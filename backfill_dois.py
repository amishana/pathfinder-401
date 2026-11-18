"""Find DOIs for publications that don't have one, so profile pages can link to the paper.

Two steps:
    python backfill_dois.py            look up missing DOIs on Crossref, write doi_matches.json (no db writes)
    python backfill_dois.py --apply    save the "confident" matches from doi_matches.json to Supabase

A match is confident when the title is a near-exact match, the professor is on the author
list, and the year is within one. Everything else is left alone; check "review" rows by hand
in doi_matches.json and flip their verdict to "confident" if they're right.

Needs DATABASE_URL in .env.
"""

import argparse
import datetime
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

MATCHES_FILE = os.path.join(os.path.dirname(__file__), "doi_matches.json")
CROSSREF = "https://api.crossref.org/works"


def connect():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True, prepare_threshold=None)
    register_vector(conn)
    return conn


def current_packages(conn):
    rows = conn.execute(
        """select f.faculty_id, pv.package from faculty f
           join package_versions pv on pv.faculty_id = f.faculty_id and pv.version = f.current_version
           order by f.faculty_id"""
    ).fetchall()
    return {fid: pkg for fid, pkg in rows}


def words(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", s).split()


def title_sim(a, b):
    return SequenceMatcher(None, " ".join(words(a)), " ".join(words(b))).ratio()


def surname(name):
    parts = words(re.sub(r"\(.*?\)", "", name or ""))
    return parts[-1] if parts else ""


def crossref_search(title, author):
    q = urllib.parse.urlencode({"query.bibliographic": title, "query.author": author, "rows": 5})
    req = urllib.request.Request(f"{CROSSREF}?{q}", headers={"User-Agent": "viterbi-pathfinder-doi-backfill"})
    # crossref rate limits if you go too fast, so back off and retry
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)["message"]["items"]
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("crossref kept rate limiting")


def best_match(pub, author):
    best = None
    for w in crossref_search(pub["title"], author):
        found_title = (w.get("title") or [""])[0]
        found_year = ((w.get("issued") or {}).get("date-parts") or [[None]])[0][0]
        authors = [surname(a.get("family", "")) for a in w.get("author", [])]
        year = pub.get("publication_year")
        m = {
            "doi": w.get("DOI"),
            "sim": round(title_sim(pub["title"], found_title), 3),
            "author_ok": author in authors,
            "year_ok": not year or not found_year or abs(found_year - int(year)) <= 1,
            "match_title": found_title,
            "match_year": found_year,
        }
        if best is None or (m["sim"], m["author_ok"]) > (best["sim"], best["author_ok"]):
            best = m
    return best


def verdict(m):
    if not m or not m["doi"]:
        return "none"
    if m["sim"] >= 0.9 and m["author_ok"] and m["year_ok"]:
        return "confident"
    if m["sim"] >= 0.75 and (m["author_ok"] or m["year_ok"]):
        return "review"
    return "none"


def lookup(conn):
    rows = []
    for fid, pkg in current_packages(conn).items():
        name = pkg["identity"].get("canonical_name") or pkg["identity"].get("preferred_display_name")
        author = surname(name)
        for pub in pkg["publications"]["records"]:
            if pub.get("doi"):
                continue
            try:
                m = best_match(pub, author)
            except Exception as e:
                print(f"  lookup failed for {pub['title'][:60]!r}: {e}")
                m = None
            row = {"faculty_id": fid, "faculty": name, "publication_id": pub["publication_id"],
                   "title": pub["title"], "year": pub.get("publication_year"), **(m or {})}
            row["verdict"] = verdict(m)
            rows.append(row)
            print(f"{row['verdict']:9} {name[:20]:20} {pub['title'][:70]}", flush=True)
            time.sleep(0.5)

    with open(MATCHES_FILE, "w") as f:
        json.dump(rows, f, indent=1)
    counts = {v: sum(r["verdict"] == v for r in rows) for v in ("confident", "review", "none")}
    print(f"\n{counts} -> {MATCHES_FILE}")
    print("nothing was written to the database. run with --apply to save the confident ones.")


def apply(conn):
    # imported here so the lookup step doesn't have to load the embedding model
    from sentence_transformers import SentenceTransformer
    from load_db import load_one

    with open(MATCHES_FILE) as f:
        rows = [r for r in json.load(f) if r["verdict"] == "confident"]
    by_faculty = {}
    for r in rows:
        by_faculty.setdefault(r["faculty_id"], {})[r["publication_id"]] = r["doi"]

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    packages = current_packages(conn)
    for fid, dois in by_faculty.items():
        pkg = packages.get(fid)
        if not pkg:
            continue
        added = 0
        for pub in pkg["publications"]["records"]:
            # don't clobber a doi someone added since the lookup ran
            if pub["publication_id"] in dois and not pub.get("doi"):
                pub["doi"] = dois[pub["publication_id"]]
                added += 1
        if not added:
            continue
        pkg.setdefault("review_history", []).append({
            "review_date": datetime.date.today().isoformat(),
            "reviewer": "backfill_dois.py",
            "decision": "Added DOIs from Crossref",
            "notes": f"{added} DOI(s) matched on title, author, and year",
        })
        print(f"{fid}: +{added} DOIs -> {load_one(conn, pkg, model)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="save confident matches from doi_matches.json")
    args = ap.parse_args()
    with connect() as conn:
        apply(conn) if args.apply else lookup(conn)


if __name__ == "__main__":
    main()

"""Uploaded CV -> reviewable draft -> (after an administrator approves) Supabase.

Used by the API (viterbi-faculty-search/api.py). Nothing reaches the database until a person
approves it on the Profile review page:

    1. POST /cv-upload          create_draft(): save the PDF, have the model structure it, and
                                save a pending draft showing exactly what would change.
    2. GET  /reviews[/<id>]     list_drafts() / get_draft(): what the reviewer looks at.
    3. POST /reviews/<id>/approve   approve_draft(): load it into Supabase with load_db.load_one().
       POST /reviews/<id>/reject    reject_draft(): mark it rejected; nothing is written.

Two upload modes:

    new       The CV is for someone not in the database yet. structure_cv.py builds a fresh
              package with the next free VIT-FAC ID. Rejected if a faculty member with the same
              name already exists (upload under "existing" instead).

    existing  The CV is for a faculty member who already has a package. Their current package
              (the latest version in Supabase) is kept as is, and only what the CV adds is merged
              in: publications, honors, patents, and links not already there, plus the CV as a
              source. Awards are never touched; they come from the DCG award report, not CVs.
              The merge is redone at approval time against the then-current package, so
              approving one draft never undoes another.

Files (all local to the machine running the API, and gitignored where they may be private):
    cv-uploads/<timestamp>_<original name>.pdf     the uploaded PDF
    cv-structured-outputs/drafts/<draft_id>.json   the draft: status, model output, packages

Approved packages are not written to faculty-json-outputs/. Running load_db.py for a faculty
member whose CV was merged here replaces the merged version with the git package again.
"""

import copy
import difflib
import json
import os
import re
import threading
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from load_db import load_one
from structure_cv import OUT_DIR, REPO_ROOT, CVError, build_package, existing_packages, extract_facts, slugify

UPLOAD_DIR = REPO_ROOT / "cv-uploads"
DRAFT_DIR = OUT_DIR / "drafts"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PUBS = 15

# Reserving faculty IDs, and approving drafts, must not interleave between requests.
_lock = threading.Lock()


class UploadConflict(Exception):
    """The upload doesn't fit the database (duplicate person, wrong professor, unknown ID,
    draft already decided)."""


class DraftNotFound(Exception):
    pass


def connect():
    """Fresh autocommit connection, the same settings load_db.py uses."""
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True, prepare_threshold=None)
    register_vector(conn)
    return conn


def list_faculty(conn):
    rows = conn.execute("select faculty_id, canonical_name from faculty order by canonical_name").fetchall()
    return [{"faculty_id": fid, "name": name} for fid, name in rows]


def current_package(conn, faculty_id):
    """The latest package for a faculty member from Supabase, or None."""
    row = conn.execute(
        """select pv.package from faculty f
           join package_versions pv on pv.faculty_id = f.faculty_id and pv.version = f.current_version
           where f.faculty_id = %s""",
        (faculty_id,),
    ).fetchone()
    return row[0] if row else None


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------------------------
# Names and IDs
# ---------------------------------------------------------------------------------------------

_NAME_SUFFIXES = {"ph", "phd", "d", "md", "jr", "sr", "ii", "iii", "iv", "pe", "dr", "prof"}


def _name_words(name):
    """Lowercase name words without degrees/titles: 'Robin Jia, Ph.D.' -> ['robin', 'jia']."""
    return [w for w in re.findall(r"[a-z]+", (name or "").lower()) if w not in _NAME_SUFFIXES]


def same_person(a, b):
    """Loose check that two names are the same person: the last names match, and the rest of
    the shorter name shows up in the longer one ("Andreas Molisch" vs "Andreas F. Molisch")."""
    wa, wb = _name_words(a), _name_words(b)
    if not wa or not wb or wa[-1] != wb[-1]:
        return False
    ta, tb = {w for w in wa if len(w) > 1}, {w for w in wb if len(w) > 1}
    small, big = sorted((ta, tb), key=len)
    return len(small & big) >= min(2, len(small))


def next_faculty_id(taken):
    """One past the highest VIT-FAC number in `taken` or in git, so IDs are never reused."""
    ids = set(taken) | set(existing_packages())
    numbers = [int(fid[-4:]) for fid in ids if re.fullmatch(r"VIT-FAC-\d{4}", fid)] or [0]
    return f"VIT-FAC-{max(numbers) + 1:04d}"


# ---------------------------------------------------------------------------------------------
# Merging a CV into an existing package
# ---------------------------------------------------------------------------------------------

def _free_id(record_id, taken):
    """record_id if unused, else the same ID with its trailing number bumped until it's free."""
    if record_id not in taken:
        return record_id
    prefix, number = re.match(r"(.*?)(\d+)$", record_id).groups()
    n = int(number)
    while f"{prefix}{n:0{len(number)}d}" in taken:
        n += 1
    return f"{prefix}{n:0{len(number)}d}"


def _already_has(text, seen):
    """True if the title `text` matches something in `seen` (a list of slugs), allowing small
    differences like plurals, punctuation, or a PDF typo ("touch sensation" vs "sensations").
    Only for long titles: on short strings like honors, "best paper" vs "best poster" would match."""
    slug = slugify(text or "")
    if not slug:
        return True
    return any(slug == other or difflib.SequenceMatcher(None, slug, other).ratio() >= 0.92 for other in seen)


def _pub_label(pub):
    return f"{pub.get('publication_year') or ''} {pub['title']}".strip()


def _honor_label(honor):
    return f"{honor.get('year') or ''} {honor['honor']}".strip()


def merge_cv_into(existing, cv_pkg, model):
    """A copy of `existing` with what the CV package adds.

    Returns (package, added, conflicts): added maps publications/honors/patents/links to a list
    of short labels for the reviewer; conflicts lists disagreements that were not applied."""
    pkg = copy.deepcopy(existing)
    today = date.today().isoformat()
    added = {"publications": [], "honors": [], "patents": [], "links": []}
    conflicts = []

    # publications: add ones whose title isn't already there
    pubs = pkg.setdefault("publications", {"views": {}, "records": []})
    records = pubs.setdefault("records", [])
    seen = [slugify(p.get("title") or "") for p in records]
    taken = {p.get("publication_id") for p in records}
    for pub in cv_pkg["publications"]["records"]:
        if _already_has(pub["title"], seen):
            continue
        pub = dict(pub, publication_id=_free_id(pub["publication_id"], taken))
        taken.add(pub["publication_id"])
        seen.append(slugify(pub["title"]))
        records.append(pub)
        added["publications"].append(_pub_label(pub))
    if added["publications"]:
        newest = sorted(records, key=lambda p: p.get("publication_year") or 0, reverse=True)
        pubs.setdefault("views", {})["recent_publication_ids"] = [p["publication_id"] for p in newest[:5]]
        note = f"{len(added['publications'])} publications added from a CV uploaded {today}."
        pubs["coverage_note"] = f"{pubs.get('coverage_note') or ''} {note}".strip()

    # honors. Same text in different years is usually a different honor ("Best Paper Finalist"
    # twice). But when the text appears exactly once on each side, it's one honor whose year
    # disagrees: keep the package's year and report the conflict.
    honors = pkg.setdefault("honors", [])
    seen = {(h.get("year"), slugify(h.get("honor") or "")) for h in honors}
    existing_years = {}
    for h in honors:
        existing_years.setdefault(slugify(h.get("honor") or ""), []).append(h.get("year"))
    cv_counts = Counter(slugify(h["honor"]) for h in cv_pkg["honors"])
    for honor in cv_pkg["honors"]:
        slug = slugify(honor["honor"])
        if (honor.get("year"), slug) in seen:
            continue
        if cv_counts[slug] == 1 and len(existing_years.get(slug, [])) == 1:
            conflicts.append(f"Honor '{honor['honor']}': profile says {existing_years[slug][0]}, "
                             f"CV says {honor.get('year')}. Kept the profile's year.")
            continue
        seen.add((honor.get("year"), slug))
        honors.append(honor)
        added["honors"].append(_honor_label(honor))

    # patents
    patents = pkg.setdefault("patents_and_inventions", [])
    seen = [slugify(p.get("title") or "") for p in patents]
    taken = {p.get("patent_record_id") for p in patents}
    for patent in cv_pkg["patents_and_inventions"]:
        if _already_has(patent["title"], seen):
            continue
        patent = dict(patent, patent_record_id=_free_id(patent["patent_record_id"], taken))
        taken.add(patent["patent_record_id"])
        seen.append(slugify(patent["title"]))
        patents.append(patent)
        added["patents"].append(patent["title"])

    # links
    links = pkg.setdefault("links", [])
    seen = {(l.get("url") or "").rstrip("/").lower() for l in links}
    for link in cv_pkg["links"]:
        if link["url"].rstrip("/").lower() not in seen:
            seen.add(link["url"].rstrip("/").lower())
            links.append(link)
            added["links"].append(link["url"])

    # identity: only fill gaps, never overwrite
    identity = pkg.setdefault("identity", {})
    if not identity.get("usc_email") and cv_pkg["identity"].get("usc_email"):
        identity["usc_email"] = cv_pkg["identity"]["usc_email"]
    names = identity.setdefault("publishing_names", [])
    for name in cv_pkg["identity"]["publishing_names"]:
        if name not in names:
            names.append(name)

    # research: keep the reviewed summary; use the CV's only if there is none
    research = pkg.setdefault("research", {})
    if not research.get("summary"):
        for key in ("summary", "domains", "methods", "applications"):
            research[key] = cv_pkg["research"][key]
        research["review_status"] = cv_pkg["research"]["review_status"]

    pkg.setdefault("sources", []).append(cv_pkg["sources"][0])
    dq = pkg.setdefault("data_quality", {})
    logged = dq.setdefault("conflicts", [])
    logged.extend(c for c in conflicts if c not in logged)
    dq.setdefault("warnings", []).append(
        f"CV uploaded {today} merged in by machine ({model}); added records need human review.")
    dq["warnings"].extend(w for w in cv_pkg["data_quality"]["warnings"] if w.startswith("Extraction note"))
    dq["source_reference_count"] = len(pkg["sources"])

    refresh = pkg.setdefault("refresh", {})
    refresh["source_snapshot_date"] = today
    refresh["updated_at"] = _now()
    pkg.setdefault("review_history", []).append({
        "review_date": today,
        "reviewer": f"cv_upload.py ({model})",
        "decision": "Machine merge of uploaded CV",
        "notes": ", ".join(f"{len(v)} {k}" for k, v in added.items()) + " added; nothing existing was changed.",
    })
    return pkg, added, conflicts


# ---------------------------------------------------------------------------------------------
# Drafts
# ---------------------------------------------------------------------------------------------

def _draft_path(draft_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", draft_id or ""):
        raise DraftNotFound(draft_id)
    return DRAFT_DIR / f"{draft_id}.json"


def _read_draft(draft_id):
    path = _draft_path(draft_id)
    if not path.exists():
        raise DraftNotFound(draft_id)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_draft(draft):
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    _draft_path(draft["draft_id"]).write_text(json.dumps(draft, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _summary(draft):
    """What list views and the upload response show (no full packages)."""
    keys = ("draft_id", "status", "mode", "faculty_id", "name", "cv_filename", "created_at",
            "decided_at", "db_status", "notes", "conflicts")
    out = {k: draft.get(k) for k in keys}
    out["added_counts"] = {k: len(v) for k, v in draft["added"].items()}
    return out


def pending_new_ids():
    """faculty IDs reserved by new-professor drafts still waiting for review."""
    return {d["faculty_id"]: d["name"] for d in _all_drafts() if d["status"] == "pending" and d["mode"] == "new"}


def _all_drafts():
    if not DRAFT_DIR.exists():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in DRAFT_DIR.glob("*.json")]


def list_drafts(status=None):
    drafts = [d for d in _all_drafts() if status in (None, d["status"])]
    drafts.sort(key=lambda d: d["created_at"], reverse=True)
    return [_summary(d) for d in drafts]


def get_draft(draft_id):
    """Everything the reviewer needs: summary, what would be added, and the resulting package."""
    draft = _read_draft(draft_id)
    out = _summary(draft)
    out["added"] = draft["added"]
    out["package"] = draft["preview_package"]
    return out


def cv_file(draft_id):
    """Path of the uploaded PDF behind a draft, so the reviewer can check against it."""
    path = REPO_ROOT / _read_draft(draft_id)["cv_file"]
    if not path.exists():
        raise DraftNotFound(draft_id)
    return path


def save_pdf(data, filename):
    if len(data) > MAX_UPLOAD_BYTES:
        raise CVError("The file is larger than 10 MB.")
    if not data.startswith(b"%PDF"):
        raise CVError("That file is not a PDF.")
    UPLOAD_DIR.mkdir(exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename or "cv").stem).strip("._") or "cv"
    path = UPLOAD_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{stem[:80]}.pdf"
    path.write_bytes(data)
    return path


def create_draft(data, filename, mode, faculty_id):
    """Save and structure one uploaded CV as a pending draft. Writes nothing to the database.

    Raises CVError (bad file or extraction failed) or UploadConflict (doesn't fit the database)."""
    if mode not in ("new", "existing"):
        raise CVError("mode must be 'new' or 'existing'.")

    with connect() as conn:
        existing = None
        if mode == "existing":
            if not faculty_id:
                raise CVError("Choose which professor this CV belongs to.")
            existing = current_package(conn, faculty_id)
            if existing is None:
                raise UploadConflict(f"No faculty member {faculty_id} in the database.")

        cv_path = save_pdf(data, filename)
        facts, llm_model = extract_facts(cv_path, max_pubs=MAX_PUBS)  # the slow part
        cv_name = facts["canonical_name"].strip()

        with _lock:
            known = {f["faculty_id"]: f["name"] for f in list_faculty(conn)}
            if mode == "new":
                pending = pending_new_ids()
                for fid, name in {**known, **pending}.items():
                    if same_person(name, cv_name):
                        where = "already in the database" if fid in known else "already waiting for review"
                        raise UploadConflict(
                            f"{cv_name} is {where} as {name} ({fid}). "
                            + ("Upload this CV under Existing professor instead." if fid in known else
                               "Approve or reject that draft first."))
                faculty_id = next_faculty_id(set(known) | set(pending))
                cv_pkg = build_package(facts, faculty_id, cv_path, llm_model, MAX_PUBS)
                preview = cv_pkg
                added = {
                    "publications": [_pub_label(p) for p in cv_pkg["publications"]["records"]],
                    "honors": [_honor_label(h) for h in cv_pkg["honors"]],
                    "patents": [p["title"] for p in cv_pkg["patents_and_inventions"]],
                    "links": [l["url"] for l in cv_pkg["links"]],
                }
                conflicts = []
            else:
                existing_name = existing["identity"]["canonical_name"]
                if not same_person(existing_name, cv_name):
                    raise UploadConflict(
                        f"This CV looks like it belongs to {cv_name}, not {existing_name}. Nothing was saved.")
                cv_pkg = build_package(facts, faculty_id, cv_path, llm_model, MAX_PUBS)
                preview, added, conflicts = merge_cv_into(existing, cv_pkg, llm_model)

            draft = {
                "draft_id": f"{cv_path.stem[:15]}-{slugify(cv_name)[:60]}",
                "status": "pending",
                "mode": mode,
                "faculty_id": faculty_id,
                "name": preview["identity"]["canonical_name"],
                "cv_filename": filename,
                "cv_file": cv_path.relative_to(REPO_ROOT).as_posix(),
                "model": llm_model,
                "created_at": _now(),
                "decided_at": None,
                "db_status": None,
                "notes": facts.get("extraction_notes") or [],
                "conflicts": conflicts,
                "added": added,
                "facts": facts,              # raw model output
                "cv_package": cv_pkg,        # the CV alone, as a package
                "preview_package": preview,  # what the profile would look like if approved now
            }
            _write_draft(draft)
    return _summary(draft)


def approve_draft(draft_id, embed_model):
    """Load an approved draft into Supabase. Returns the updated draft summary."""
    with _lock:
        draft = _read_draft(draft_id)
        if draft["status"] != "pending":
            raise UploadConflict(f"This draft was already {draft['status']}.")

        with connect() as conn:
            if draft["mode"] == "new":
                known = {f["faculty_id"]: f["name"] for f in list_faculty(conn)}
                dup = next((fid for fid, name in known.items() if same_person(name, draft["name"])), None)
                if dup:
                    raise UploadConflict(f"{draft['name']} was added to the database as {dup} since this "
                                         f"upload. Reject this draft and upload under Existing professor.")
                package = draft["cv_package"]
                if package["faculty_id"] in known:  # ID got used in the meantime: take the next one
                    new_id = next_faculty_id(set(known) | set(pending_new_ids()))
                    package = build_package(draft["facts"], new_id, REPO_ROOT / draft["cv_file"],
                                            draft["model"], MAX_PUBS)
            else:
                # Re-merge against the current version, in case it changed after the upload.
                current = current_package(conn, draft["faculty_id"])
                if current is None:
                    raise UploadConflict(f"{draft['faculty_id']} is no longer in the database.")
                package, draft["added"], draft["conflicts"] = merge_cv_into(
                    current, draft["cv_package"], draft["model"])

            package = copy.deepcopy(package)
            package.setdefault("review_history", []).append({
                "review_date": date.today().isoformat(),
                "reviewer": "Profile review page",
                "decision": "Approved for database",
                "notes": "Approved by an administrator after reviewing the uploaded CV draft.",
            })
            draft["db_status"] = load_one(conn, package, embed_model)

        draft.update(status="approved", decided_at=_now(), faculty_id=package["faculty_id"],
                     preview_package=package)
        _write_draft(draft)
    return _summary(draft)


def reject_draft(draft_id):
    with _lock:
        draft = _read_draft(draft_id)
        if draft["status"] != "pending":
            raise UploadConflict(f"This draft was already {draft['status']}.")
        draft.update(status="rejected", decided_at=_now())
        _write_draft(draft)
    return _summary(draft)

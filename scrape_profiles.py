"""Scrape each professor's USC profile page and fill the empty profile columns in the faculty table.

Fills these columns: profile_picture, title, education, contact, usc_profile.

When to run:
    - After load_db.py has created the faculty rows.
    - Whenever you want to refresh profile info from the web.

Usage:
    python scrape_profiles.py --dry-run                 # parse and print, write nothing
    python scrape_profiles.py                           # scrape every faculty member
    python scrape_profiles.py VIT-FAC-0005 VIT-FAC-0010 # scrape only these
    python scrape_profiles.py --debug VIT-FAC-0005      # also save raw HTML to scrape-debug/
    python scrape_profiles.py --no-search               # never search the web for missing URLs

Where the profile URL comes from, in order:
    1. The faculty.usc_profile column, if already filled.
    2. A URL key inside the package's "identity" block (see URL_KEYS below).
    3. The predictable USC Viterbi directory URL built from the name
       (https://viterbi.usc.edu/directory/faculty/<Last>/<First>), then a web search limited to
       usc.edu directory pages. A result is accepted only if
       the page itself contains the professor's first and last name. The URL found is saved to
       usc_profile along with the rest of the scraped data.
    4. If all of that fails, the faculty member is skipped. Fill usc_profile by hand for those.

Needs DATABASE_URL in .env, same as load_db.py (transaction pooler, port 6543).

Safety:
    - A field that could not be parsed is left alone. It never overwrites an existing value with NULL.
    - Each faculty member is written in its own transaction. One failure does not stop the rest.
    - Each result is also saved to scraped-profiles/<faculty_id>.json so it can be committed to git.
      load_db.py does not touch these columns, so reloading packages will not wipe them.

IMPORTANT: the HTML parsing below is generic. Page layouts differ, so check the --dry-run output
and adjust the selectors in the CONFIG section if a field comes back empty.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse

import psycopg
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# ----------------------------------------------------------------------------------------------
# CONFIG: tweak these to match the real page layout.
# ----------------------------------------------------------------------------------------------
PACKAGE_GLOB = "faculty-json-outputs/VIT-FAC-*.json"
URL_KEYS = ("usc_profile", "usc_profile_url", "profile_url", "homepage", "url")

# CSS selectors tried in order. The first one that returns non-empty text wins.
TITLE_SELECTORS = [
    ".faculty-title", ".profile-title", ".job-title", ".title", "h1 + p", "h1 + h2",
]
# Section headings that are never a job title. A match is skipped so it can't be saved by mistake.
NOT_A_TITLE = re.compile(
    r"^\s*(education|contact|research|publications?|biography|bio|awards?|honors?|about|overview|"
    r"teaching|courses|news|links?)\b", re.I)
PICTURE_SELECTORS = [
    "meta[property='og:image']", ".profile-photo img", ".faculty-photo img", ".profile img",
    "img[alt*='photo' i]", "img[alt*='portrait' i]",
]
EDUCATION_HEADING = re.compile(r"^\s*(education|degrees?)\b", re.I)
HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "dt", "strong", "b"]

# USC Viterbi directory pages follow this pattern, so we try it before searching.
DIRECTORY_URL = "https://viterbi.usc.edu/directory/faculty/{last}/{first}"
# Search results are accepted only if their URL contains this, so news articles are ignored.
REQUIRED_IN_URL = "/directory/"

# Web search used to find a profile URL when none is stored.
SEARCH_URL = "https://html.duckduckgo.com/html/"
SEARCH_QUERY = '"{name}" USC Viterbi faculty site:usc.edu'
ALLOWED_DOMAIN = "usc.edu"
URL_HINTS = ("viterbi", "directory", "faculty", "people", "profile")  # boosts matching URLs
MAX_CANDIDATES = 5  # how many search results to open and check per professor

REQUEST_DELAY_SECONDS = 1.5  # be polite to the university's server
TIMEOUT_SECONDS = 20
HEADERS = {"User-Agent": "Mozilla/5.0 (research-faculty-search prototype; contact your-team@example.edu)"}
# ----------------------------------------------------------------------------------------------

OUTPUT_DIR = Path("scraped-profiles")
DEBUG_DIR = Path("scrape-debug")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}")


def clean(text):
    """Collapse whitespace. Returns None for empty text."""
    text = re.sub(r"\s+", " ", text or "").strip()
    return text or None


def fetch(url):
    """GET a page with a couple of retries. Raises requests.RequestException on final failure."""
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.text
        except requests.RequestException as error:
            last_error = error
            time.sleep(2 * (attempt + 1))
    raise last_error


def parse_title(soup):
    for selector in TITLE_SELECTORS:
        node = soup.select_one(selector)
        text = clean(node.get_text(" ")) if node else None
        if text and len(text) < 200 and not NOT_A_TITLE.match(text):
            return text
    return None


def parse_picture(soup, base_url):
    for selector in PICTURE_SELECTORS:
        node = soup.select_one(selector)
        if not node:
            continue
        src = node.get("content") if node.name == "meta" else node.get("src") or node.get("data-src")
        if src:
            return urljoin(base_url, src)
    return None


def parse_education(soup):
    """Find an 'Education' heading and collect the list items or paragraphs that follow it.
    Falls back to extracting degree sentences from the biography section."""
    for heading in soup.find_all(HEADING_TAGS):
        if not EDUCATION_HEADING.match(heading.get_text(" ")):
            continue
        lines = []
        for sibling in heading.find_all_next():
            if sibling is heading:
                continue
            if sibling.name in ("h1", "h2", "h3", "h4", "h5", "h6") and sibling.name <= (heading.name or "h6"):
                break
            if sibling.name == "li":
                lines.append(clean(sibling.get_text(" ")))
            elif sibling.name == "p" and not sibling.find("li"):
                lines.append(clean(sibling.get_text(" ")))
            if len(lines) >= 8:
                break
        lines = [line for line in lines if line]
        if lines:
            return "\n".join(dict.fromkeys(lines))

    return _education_from_bio(soup)


_DEGREE_RE = re.compile(
    r"\b(Ph\.?D\.?|M\.?S\.?|M\.?Sc\.?|M\.?Eng\.?|M\.?A\.?|B\.?S\.?|B\.?A\.?|B\.?Eng\.?"
    r"|Doctor|Master|Bachelor|postdoc(?:toral)?)\b", re.I,
)
_BIO_HEADING = re.compile(r"^\s*(bio(?:graphy)?|about|overview|profile)\b", re.I)


def _education_from_bio(soup):
    """Extract degree-bearing sentences from a biography block when there is no Education section."""
    # Try a labelled bio section first, then fall back to the longest paragraph.
    bio_text = None
    for heading in soup.find_all(HEADING_TAGS):
        if not _BIO_HEADING.match(heading.get_text(" ")):
            continue
        paragraphs = []
        for sibling in heading.find_all_next():
            if sibling.name in ("h1", "h2", "h3", "h4", "h5", "h6") and sibling.name <= (heading.name or "h6"):
                break
            if sibling.name in ("p", "li") and not sibling.find("li"):
                text = clean(sibling.get_text(" "))
                if text:
                    paragraphs.append(text)
            if len(paragraphs) >= 10:
                break
        if paragraphs:
            bio_text = " ".join(paragraphs)
            break

    if not bio_text:
        # Fallback: largest paragraph on the page (likely the bio)
        candidates = [
            clean(p.get_text(" ")) for p in soup.find_all("p") if not p.find("li")
        ]
        candidates = [c for c in candidates if c]
        bio_text = max(candidates, key=len) if candidates else None

    if not bio_text:
        return None

    # Pull out individual sentences that mention a degree.
    sentences = re.split(r"(?<=[.!?])\s+", bio_text)
    lines = [s.strip() for s in sentences if _DEGREE_RE.search(s) and len(s.strip()) < 300]
    return "\n".join(dict.fromkeys(lines)) or None


def parse_contact(soup):
    """Collect email, phone, and office from mailto:/tel: links and nearby text."""
    parts = []

    email = None
    mailto = soup.select_one("a[href^='mailto:']")
    if mailto:
        email = clean(mailto["href"].split(":", 1)[1].split("?")[0])
    if not email:
        match = EMAIL_RE.search(soup.get_text(" "))
        email = match.group(0) if match else None
    if email:
        parts.append(f"Email: {email}")

    phone = None
    tel = soup.select_one("a[href^='tel:']")
    if tel:
        phone = clean(tel["href"].split(":", 1)[1])
    if not phone:
        match = PHONE_RE.search(soup.get_text(" "))
        phone = match.group(0) if match else None
    if phone:
        parts.append(f"Phone: {phone}")

    office = re.search(r"(?:Office|Location)\s*:?\s*([A-Z]{2,4}\s?\d{2,4}[A-Z]?)", soup.get_text(" "))
    if office:
        parts.append(f"Office: {office.group(1)}")

    return "\n".join(parts) or None


def scrape(url, debug_id=None, html=None):
    """Download (unless html is given) and parse one profile page. Returns the five target columns."""
    html = html or fetch(url)
    if debug_id:
        DEBUG_DIR.mkdir(exist_ok=True)
        (DEBUG_DIR / f"{debug_id}.html").write_text(html, encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    return {
        "profile_picture": parse_picture(soup, url),
        "title": parse_title(soup),
        "education": parse_education(soup),
        "contact": parse_contact(soup),
        "usc_profile": url,
    }


def package_url(faculty_id):
    """Look for a profile URL in the faculty member's JSON package, if there is one."""
    for path in glob.glob(PACKAGE_GLOB):
        pkg = json.load(open(path))
        if pkg.get("faculty_id") != faculty_id:
            continue
        identity = pkg.get("identity") or {}
        for key in URL_KEYS:
            value = identity.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value
    return None


def name_matches(html, canonical_name):
    """True if the page contains the professor's last name and any first-name variant.

    Handles nicknames: "Jianhua (Joshua) Yang" matches pages with "Joshua Yang" or
    "Jianhua Yang" by extracting each word from the parenthetical as a separate candidate.
    """
    nickname = re.search(r"\(([^)]+)\)", canonical_name)
    words = re.findall(r"[A-Za-z\u00C0-\u024F'-]+", re.sub(r"\([^)]*\)", "", canonical_name))
    if len(words) < 2:
        return False
    first, last = words[0].lower(), words[-1].lower()
    firsts = {first}
    if nickname:
        # split multi-word nicknames and add each word longer than one char
        nick_words = re.findall(r"[A-Za-z\u00C0-\u024F'-]+", nickname.group(1))
        firsts |= {w.lower() for w in nick_words if len(w) > 1}

    soup = BeautifulSoup(html, "html.parser")
    heading = " ".join(node.get_text(" ") for node in soup.find_all(["title", "h1", "h2"]))
    text = (heading + " " + soup.get_text(" ")[:3000]).lower()
    return last in text and any(name in text for name in firsts)


def search_candidates(name):
    """Search the web and return usc.edu result URLs, best-looking first."""
    response = requests.get(SEARCH_URL, params={"q": SEARCH_QUERY.format(name=name)},
                            headers=HEADERS, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    urls = []
    for link in soup.select("a.result__a"):
        href = link.get("href", "")
        # DuckDuckGo wraps results as //duckduckgo.com/l/?uddg=<real url>
        target = parse_qs(urlparse(href).query).get("uddg", [href])[0]
        target = unquote(target)
        host = urlparse(target).netloc.lower()
        if target.startswith("http") and (host == ALLOWED_DOMAIN or host.endswith("." + ALLOWED_DOMAIN)):
            if target not in urls and REQUIRED_IN_URL in target:
                urls.append(target)

    if not soup.select("a.result__a"):
        print("    (search returned no results at all; DuckDuckGo is probably rate-limiting. Wait a few minutes.)")

    urls.sort(key=lambda url: -sum(hint in url.lower() for hint in URL_HINTS))  # stable: keeps search order on ties
    return urls[:MAX_CANDIDATES]


def fetch_once(url):
    """One attempt, no retries. Returns the HTML, or None for 404s and other failures."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
        return response.text if response.status_code == 200 else None
    except requests.RequestException:
        return None


def directory_candidates(name):
    """Build likely directory URLs from a name: "Viktor K. Prasanna" -> .../Prasanna/Viktor.

    Tries the nickname ("Joshua") and two-word surnames ("Twomey Sanders") as alternatives.
    """
    nickname = re.search(r"\(([^)]+)\)", name)
    words = re.findall(r"[^\s()]+", re.sub(r"\([^)]*\)", "", name))
    if len(words) < 2:
        return []
    firsts = [words[0]] + ([nickname.group(1)] if nickname else [])
    lasts = [words[-1]]
    if len(words) > 2 and len(words[-2].strip(".")) > 1:
        lasts.insert(0, f"{words[-2]} {words[-1]}")
    return [DIRECTORY_URL.format(last=quote(last), first=quote(first)) for last in lasts for first in firsts]


def find_profile(name):
    """Return (url, html) for the professor's profile page, or None.

    First tries the predictable directory URL, then falls back to a web search. Either way the
    page must contain the professor's first and last name.
    """
    for url in directory_candidates(name):
        html = fetch_once(url)
        if html and name_matches(html, name):
            return url, html
        time.sleep(REQUEST_DELAY_SECONDS)

    for url in search_candidates(name):
        time.sleep(REQUEST_DELAY_SECONDS)
        html = fetch_once(url)
        if html and name_matches(html, name):
            return url, html
    return None


def save_to_db(conn, faculty_id, data):
    """Update only the fields that were found. Never overwrite a value with NULL."""
    with conn.transaction():
        conn.execute(
            """update faculty set
                 profile_picture = coalesce(%(profile_picture)s, profile_picture),
                 title           = coalesce(%(title)s, title),
                 education       = coalesce(%(education)s, education),
                 contact         = coalesce(%(contact)s, contact),
                 usc_profile     = coalesce(%(usc_profile)s, usc_profile)
               where faculty_id = %(faculty_id)s""",
            {**data, "faculty_id": faculty_id},
        )


def save_snapshot(faculty_id, data):
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / f"{faculty_id}.json").write_text(
        json.dumps({"faculty_id": faculty_id, **data}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _data_from_package(faculty_id):
    """Pull title, contact, and education from the faculty JSON package as a fallback
    when the USC profile page cannot be found online."""
    for path in glob.glob(PACKAGE_GLOB):
        pkg = json.load(open(path))
        if pkg.get("faculty_id") != faculty_id:
            continue
        identity = pkg.get("identity") or {}

        # title: primary appointment title + unit
        title = None
        for appt in identity.get("appointments") or []:
            if appt.get("appointment_type") == "primary" or not title:
                parts = [appt.get("title"), appt.get("unit")]
                title = clean(", ".join(p for p in parts if p)) or title

        # contact: email from identity
        email = identity.get("usc_email")
        contact = f"Email: {email}" if email else None

        # education: not in the standard package schema — leave None so the bio fallback
        # can still fill it if the page is later scraped
        return {
            "profile_picture": None,
            "title": title,
            "education": None,
            "contact": contact,
            "usc_profile": None,
        }
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("faculty_ids", nargs="*", help="faculty IDs to scrape (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="print parsed results, write nothing")
    parser.add_argument("--debug", action="store_true", help="save raw HTML to scrape-debug/")
    parser.add_argument("--no-search", action="store_true", help="do not search the web for missing profile URLs")
    args = parser.parse_args()

    load_dotenv()
    if "DATABASE_URL" not in os.environ:
        sys.exit("DATABASE_URL is not set. Copy .env.example to .env and fill in the password.")

    failed, skipped, done = [], [], 0
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True, prepare_threshold=None) as conn:
        rows = conn.execute("select faculty_id, canonical_name, usc_profile from faculty order by faculty_id").fetchall()
        if args.faculty_ids:
            known = {row[0] for row in rows}
            unknown = [fid for fid in args.faculty_ids if fid not in known]
            if unknown:
                sys.exit(f"Not in the faculty table: {', '.join(unknown)}. Run load_db.py first.")
            rows = [row for row in rows if row[0] in args.faculty_ids]

        for fid, name, stored_url in rows:
            url = stored_url or package_url(fid)
            html = None
            try:
                if not url and not args.no_search:
                    found = find_profile(name)
                    if found:
                        url, html = found
                        print(f"{fid}  {name}: found profile by search: {url}")
                if not url:
                    data = _data_from_package(fid)
                    if data and any(data.values()):
                        found_fields = [k for k, v in data.items() if v]
                        print(f"{fid}  {name}: no profile URL found, using package data for {', '.join(found_fields)}")
                        if not args.dry_run:
                            save_to_db(conn, fid, data)
                            save_snapshot(fid, data)
                        else:
                            print(json.dumps(data, indent=2, ensure_ascii=False))
                        done += 1
                    else:
                        skipped.append(fid)
                        print(f"{fid}  {name}: SKIPPED, no profile URL and no package data found")
                    time.sleep(REQUEST_DELAY_SECONDS)
                    continue

                data = scrape(url, debug_id=fid if args.debug else None, html=html)
                found = [key for key, value in data.items() if value]
                print(f"{fid}  {name}: found {', '.join(found)}")
                if args.dry_run:
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                else:
                    save_to_db(conn, fid, data)
                    save_snapshot(fid, data)
                done += 1
            except (requests.RequestException, psycopg.Error) as error:
                failed.append(fid)
                print(f"{fid}  {name}: FAILED, nothing changed: {error}")

            time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nProcessed {done} of {len(rows)}. Skipped {len(skipped)}. Failed {len(failed)}.")
    if failed:
        sys.exit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
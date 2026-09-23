"""Flattens a faculty package JSON into evidence pieces, same idea as prototype.py's
load_evidence() but per-package and with a year pulled out for each piece when we can
find one.
"""


def _year_from_date(date_str):
    """Pulls a 4 digit year off the front of a date string like '2026-03-23'."""
    if not date_str:
        return None
    try:
        return int(str(date_str)[:4])
    except ValueError:
        return None


def extract_evidence(pkg):
    """Returns a list of dicts: {kind, text, year} for one faculty package."""
    pieces = []

    def add(kind, text, year=None):
        if not text:
            return
        pieces.append({"kind": kind, "text": text, "year": year})

    research = pkg.get("research") or {}
    summary = research.get("summary")
    domains = research.get("domains") or []
    if summary:
        domain_str = ", ".join(domains)
        text = f"{summary} Areas: {domain_str}." if domain_str else summary
        add("research", text)

    for pub in (pkg.get("publications") or {}).get("records") or []:
        title = pub.get("title")
        if not title:
            continue
        venue = pub.get("venue")
        text = f"{title} ({venue})" if venue else title
        add("publication", text, pub.get("publication_year"))

    for award in (pkg.get("awards") or {}).get("records") or []:
        title = award.get("title")
        if not title:
            continue
        sponsor = award.get("sponsor")
        text = f"{title} - sponsor: {sponsor}" if sponsor else title
        add("award", text, _year_from_date(award.get("start_date")))

    for patent in pkg.get("patents_and_inventions") or []:
        title = patent.get("title")
        if not title:
            continue
        add("patent", title, _year_from_date(patent.get("event_date")))

    for honor in pkg.get("honors") or []:
        text = honor.get("honor")
        if not text:
            continue
        add("honor", text, honor.get("year"))

    return pieces

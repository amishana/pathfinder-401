"""Turn a faculty CV (PDF or extracted .txt) into a draft faculty package JSON in the same
structure as faculty-json-outputs/, using a model on the USC AI gateway.

The gateway is a LiteLLM proxy, so it speaks the OpenAI API: the regular `openai` client works
with its base_url pointed at the gateway, and the model name picks the provider
(GPT, Claude, Gemini, ...).

Setup (once):
    uv pip install -r requirements.txt
    Add these to .env (see .env.example):
        USC_AI_GATEWAY_BASE_URL   the gateway URL from the ITS getting-started page
        USC_AI_GATEWAY_API_KEY    your key
        USC_AI_GATEWAY_MODEL      a model name from --list-models

Usage:
    python structure_cv.py --list-models                  # which models your key can use
    python structure_cv.py CulbertsonHeather_CV_0626.pdf
    python structure_cv.py cv.pdf --model claude-sonnet-4 --max-pubs 25
    python structure_cv.py cv.pdf --faculty-id VIT-FAC-0017
    python structure_cv.py cv.pdf --dry-run               # print the prompt, no API call

How the work is split:
    The model only reads the CV and returns the facts in it: name, appointments, research
    areas, publications, honors, patents. This script builds everything else (IDs,
    source_refs, the CV source record with its SHA-256, review statuses, data_quality),
    so provenance fields are never made up by the model.

    Awards stay empty: packages take awards from the DCG award-detail workbook, not the CV.

Output:
    cv-structured-outputs/VIT-FAC-XXXX_<name>_draft_v0.1.json            the package
    cv-structured-outputs/VIT-FAC-XXXX_<name>_draft_v0.1.extraction.json  raw model output

    Drafts go to cv-structured-outputs/, not faculty-json-outputs/, so a draft never
    overwrites a reviewed package or gets picked up by load_db.py. Move it over after review.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import openai
from dotenv import load_dotenv

SCHEMA_VERSION = "faculty-package-draft-v0.1"
REPO_ROOT = Path(__file__).resolve().parent  # so paths work no matter where this is run from
PACKAGE_GLOB = str(REPO_ROOT / "faculty-json-outputs" / "VIT-FAC-*.json")
OUT_DIR = REPO_ROOT / "cv-structured-outputs"
PROPOSED = "Proposed - Human Review Required"
MAX_CV_CHARS = 250_000  # ~60k tokens; longer CVs are cut off (the end is usually talks/service)
DEFAULT_MODEL = "openai/gpt-5.4-mini"  # used when neither --model nor USC_AI_GATEWAY_MODEL is set


class CVError(Exception):
    """A CV could not be turned into a package. The message is safe to show to the user."""


# ---------------------------------------------------------------------------------------------
# What the model returns. Strict JSON Schema: every key is required, absent values are null.
# ---------------------------------------------------------------------------------------------

def _obj(props):
    return {"type": "object", "additionalProperties": False,
            "properties": props, "required": list(props)}


def _arr(items):
    return {"type": "array", "items": items}


_STR = {"type": "string"}
_STR_OR_NULL = {"type": ["string", "null"]}
_INT_OR_NULL = {"type": ["integer", "null"]}

EXTRACTION_SCHEMA = _obj({
    "canonical_name": _STR,
    "publishing_names": _arr(_STR),
    "usc_email": _STR_OR_NULL,
    "appointments": _arr(_obj({
        "title": _STR_OR_NULL,
        "unit": _STR,
        "appointment_type": {"type": "string", "enum": ["Primary", "Courtesy", "Joint", "Other"]},
    })),
    "links": _arr(_obj({
        "link_type": {"type": "string", "enum": [
            "official_profile", "lab", "personal", "google_scholar", "github", "other"]},
        "url": _STR,
    })),
    "research": _obj({
        "summary": _STR,
        "domains": _arr(_STR),
        "methods": _arr(_STR),
        "applications": _arr(_STR),
    }),
    "publications": _arr(_obj({
        "title": _STR,
        "authors": _STR,
        "venue": _STR_OR_NULL,
        "publication_year": {"type": "integer"},
        "doi": _STR_OR_NULL,
        "publication_type": {"type": "string", "enum": [
            "journal", "conference", "book", "book_chapter", "other"]},
    })),
    "honors": _arr(_obj({"honor": _STR, "year": _INT_OR_NULL})),
    "patents": _arr(_obj({
        "title": _STR,
        "jurisdiction_identifier": _STR_OR_NULL,
        "status": _STR,
        "event_date": _STR_OR_NULL,
    })),
    "extraction_notes": _arr(_STR),
})

SYSTEM_PROMPT = """You extract structured facts from a university faculty member's CV for a \
faculty research search index. The CV text came from a PDF, so expect extraction artifacts.

Rules:
- Use only what the CV says. Never guess or add outside knowledge. Use null when a value is absent.
- Repair PDF artifacts: rejoin split words ("Con troller" -> "Controller", "Meth ods" -> \
"Methods"), fix accents ("L´opez" -> "López"), collapse extra spaces, and drop student-role \
superscript markers glued to names ("Xinlei YuP" -> "Xinlei Yu").
- Return one JSON object that matches the schema exactly, with no commentary.

Field guidance:
- canonical_name: the person's name as written at the top of the CV, without degrees ("Ph.D.").
- publishing_names: other forms of their name used as an author in the CV, if any differ.
- appointments: current USC appointments only (not past positions). The home department is \
"Primary"; courtesy appointments are "Courtesy". title is the rank/chair, e.g. "Associate \
Professor; Kenneth C. Dahlberg Early Career Chair". unit is the full department name.
- links: only URLs printed in the CV that point to this person's own pages (lab site, personal \
site, Google Scholar, GitHub, USC profile). Not dataset, paper, or media links.
- research.summary: 2-3 sentences, third person, present tense ("Current work develops ..."), \
describing the current research program as shown by recent publications, grants, and patents.
- research.domains: 5-8 short noun phrases (research areas). methods: 4-8 techniques or \
approaches. applications: 4-8 application areas.
- publications: the {max_pubs} most recent PUBLISHED or ACCEPTED works (journal articles, \
conference papers, books, book chapters), newest first. Exclude anything submitted, under \
review, or unpublished, and exclude posters, abstracts, demos, talks, and datasets. title in \
sentence case as written. authors as one comma-separated string. venue is the full journal or \
conference name, without "(Impact Factor=...)" notes. doi is the bare DOI ("10.1109/...") or null.
- honors: every award, honor, and fellowship (not research grants), with year as an integer.
- patents: every patent and patent application. status is "Granted" for awarded patents and \
"United States patent application" (or the jurisdiction's equivalent) for filed applications. \
jurisdiction_identifier is the patent number ("US 11,972,052") or null. event_date is \
YYYY-MM-DD when the CV gives a full date, otherwise YYYY-MM or YYYY.
- extraction_notes: short notes on anything ambiguous, garbled, or possibly wrong, for the \
human reviewer. Empty list if none."""


# ---------------------------------------------------------------------------------------------
# Reading the CV
# ---------------------------------------------------------------------------------------------

def read_cv_text(path):
    """Text of the CV. PDFs are extracted the same way test_extract.py does (sorted reading order)."""
    if path.suffix.lower() == ".pdf":
        import pymupdf  # only needed for PDFs

        with pymupdf.open(path) as doc:
            pages = [page.get_text("text", sort=True) for page in doc]
        if sum(len(p.strip()) for p in pages) < 50 * len(pages):
            raise CVError(f"{path.name} has almost no text layer. It is probably scanned and needs OCR first.")
        text = "\n\n".join(f"--- page {i} ---\n{p}" for i, p in enumerate(pages, 1))
    else:
        text = path.read_text(encoding="utf-8")

    if len(text) > MAX_CV_CHARS:
        print(f"Note: CV is {len(text):,} chars; sending the first {MAX_CV_CHARS:,}.")
        text = text[:MAX_CV_CHARS]
    return text


# ---------------------------------------------------------------------------------------------
# Calling the gateway
# ---------------------------------------------------------------------------------------------

def gateway_client():
    load_dotenv()
    missing = [k for k in ("USC_AI_GATEWAY_BASE_URL", "USC_AI_GATEWAY_API_KEY") if not os.environ.get(k)]
    if missing:
        raise CVError(f"Missing {', '.join(missing)} in .env. See .env.example.")
    return openai.OpenAI(base_url=os.environ["USC_AI_GATEWAY_BASE_URL"],
                         api_key=os.environ["USC_AI_GATEWAY_API_KEY"])


def list_models():
    for model in sorted(m.id for m in gateway_client().models.list()):
        print(model)


def parse_json(text):
    """Parse the model's reply, tolerating a ```json fence or text around the object."""
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start:end + 1])


def call_model(client, model, messages, max_tokens):
    """Ask for JSON, trying the strictest format the model supports first.

    1. json_schema (structured outputs): the reply is guaranteed to match EXTRACTION_SCHEMA.
    2. json_object: valid JSON, but the schema is only enforced by the prompt.
    3. no response_format: plain text, parsed with parse_json.
    LiteLLM passes these through to each provider; not every model supports all three.
    """
    formats = [
        {"type": "json_schema",
         "json_schema": {"name": "cv_extraction", "schema": EXTRACTION_SCHEMA, "strict": True}},
        {"type": "json_object"},
        None,
    ]
    for fmt in formats:
        label = fmt["type"] if fmt else "plain text"
        kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if fmt:
            kwargs["response_format"] = fmt
        try:
            response = client.chat.completions.create(**kwargs)
        except openai.BadRequestError as error:
            # Newer OpenAI reasoning models take max_completion_tokens instead of max_tokens.
            if "max_tokens" not in str(error):
                print(f"  {model} rejected {label} ({str(error)[:150]}). Trying a looser format.")
                continue
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            try:
                response = client.chat.completions.create(**kwargs)
            except openai.BadRequestError as error:
                print(f"  {model} rejected {label} ({str(error)[:150]}). Trying a looser format.")
                continue

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise CVError(f"The reply was cut off at {max_tokens} tokens. Raise --max-tokens or lower --max-pubs.")
        try:
            facts = parse_json(choice.message.content)
        except ValueError as error:
            print(f"  {label} reply was not valid JSON ({error}). Trying a looser format.")
            continue

        usage = response.usage
        if usage:
            print(f"  {label} reply: {usage.prompt_tokens:,} prompt + {usage.completion_tokens:,} output tokens")
        return facts

    raise CVError("The model never returned usable JSON. Try another model with --model.")


# ---------------------------------------------------------------------------------------------
# Building the package
# ---------------------------------------------------------------------------------------------

def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def existing_packages():
    """faculty_id -> canonical_name for every reviewed package."""
    found = {}
    for path in glob.glob(PACKAGE_GLOB):
        pkg = json.load(open(path))
        found[pkg["faculty_id"]] = pkg["identity"]["canonical_name"]
    return found


def choose_faculty_id(requested, name, known=None):
    """Use --faculty-id if given; else reuse the ID of an existing package with the same name;
    else the next unused VIT-FAC number.

    known: extra faculty_id -> name pairs (e.g. from the database) to treat as taken, on top of
    the package files in faculty-json-outputs/."""
    packages = {**existing_packages(), **(known or {})}
    if requested:
        if not re.fullmatch(r"VIT-FAC-\d{4}", requested):
            raise CVError("--faculty-id must look like VIT-FAC-0017")
        return requested
    for fid, existing_name in packages.items():
        if slugify(existing_name) == slugify(name):
            print(f"  {name} already has package {fid}; reusing that ID.")
            return fid
    numbers = [int(fid[-4:]) for fid in packages] or [0]
    return f"VIT-FAC-{max(numbers) + 1:04d}"


def clean_doi(doi):
    if not doi:
        return None
    doi = re.sub(r"^(doi:\s*|https?://(dx\.)?doi\.org/)", "", doi.strip(), flags=re.I)
    return doi if doi.startswith("10.") else None


def build_package(facts, faculty_id, cv_path, model, max_pubs):
    today = date.today().isoformat()
    name = facts["canonical_name"].strip()
    source_id = f"src-cv-{slugify(name)}-{slugify(cv_path.stem)}"
    refs = [source_id]

    # Publications: clean, dedupe, newest first, give each a globally unique ID.
    publications, seen_titles, per_year = [], set(), {}
    pubs = sorted((p for p in facts["publications"] if p.get("publication_year")),
                  key=lambda p: p["publication_year"], reverse=True)
    for pub in pubs:
        key = slugify(pub["title"])
        if not key or key in seen_titles:
            continue
        seen_titles.add(key)
        year = int(pub["publication_year"])
        per_year[year] = per_year.get(year, 0) + 1
        doi = clean_doi(pub.get("doi"))
        publications.append({
            "publication_id": f"PUB-{faculty_id}-{year}-CV-{per_year[year]:02d}",
            "title": pub["title"].strip(),
            "authors": pub.get("authors"),
            "venue": pub.get("venue") or None,
            "publication_year": year,
            "doi": doi,
            "publication_link": f"https://doi.org/{doi}" if doi else None,
            "selection_basis": ["recent_cv_view"],
            "citation_snapshot": None,
            "source_refs": refs,
            "review_status": PROPOSED,
        })
        if len(publications) == max_pubs:
            break

    patents, per_year = [], {}
    for patent in facts["patents"]:
        year_match = re.match(r"\d{4}", patent.get("event_date") or "")
        year = year_match.group(0) if year_match else "UNDATED"
        per_year[year] = per_year.get(year, 0) + 1
        patents.append({
            "patent_record_id": f"PAT-{faculty_id}-{year}-{per_year[year]:02d}",
            "title": patent["title"].strip(),
            "jurisdiction_identifier": patent.get("jurisdiction_identifier"),
            "status": patent["status"],
            "event_date": patent.get("event_date"),
            "public_link": None,
            "source_refs": refs,
            "review_status": "Proposed - Verify against USC invention source",
        })

    honors = [
        {"honor": h["honor"].strip(), "year": h.get("year"), "source_refs": refs, "review_status": "Proposed"}
        for h in facts["honors"]
    ]

    research = facts["research"]
    notes = facts.get("extraction_notes") or []

    return {
        "schema_version": SCHEMA_VERSION,
        "package_status": "Draft - Machine-extracted from CV",
        "approval_status": "Not reviewed",
        "release_scope": "Internal review only",
        "faculty_id": faculty_id,
        "identity": {
            "canonical_name": name,
            "preferred_display_name": name,
            "publishing_names": facts.get("publishing_names") or [],
            "usc_email": facts.get("usc_email"),
            "identity_status": "Proposed",
            "identity_confidence": "Medium",  # one source only; raise after checking the USC profile
            "appointments": [
                {"title": a.get("title"), "unit": a["unit"], "appointment_type": a["appointment_type"],
                 "source_refs": refs, "review_status": PROPOSED}
                for a in facts["appointments"]
            ],
            "source_refs": refs,
        },
        "links": [
            {"link_type": link["link_type"], "url": link["url"], "source_refs": refs,
             "review_status": "Proposed - URL taken from CV, not yet verified"}
            for link in facts["links"]
        ],
        "research": {
            "summary": research["summary"],
            "domains": research["domains"],
            "methods": research["methods"],
            "applications": research["applications"],
            "source_refs": refs,
            "review_status": f"Proposed - machine summary of CV ({model}); human review required",
        },
        "publications": {
            "coverage_note": (f"The {len(publications)} most recent published works listed in the CV, "
                              f"extracted {today}. No citation counts: no publication-provider "
                              f"snapshot has been processed."),
            "views": {
                "recent_publication_ids": [p["publication_id"] for p in publications[:5]],
                "most_cited_publication_ids": [],
            },
            "records": publications,
        },
        "awards": {
            "rollup": {
                "detailed_record_count": 0,
                "detailed_expected_total_usd": 0,
                "legacy_record_count": 0,
                "legacy_total_award_amount_usd": 0,
                "reconciliation_status": "No institutional award source processed",
                "source_refs": [],
                "review_status": "Pending - award detail workbook not processed",
            },
            "records": [],
        },
        "honors": honors,
        "patents_and_inventions": patents,
        "sources": [{
            "source_id": source_id,
            "source_type": "faculty_cv",
            "title": f"{name} CV",
            "repository_path": cv_path.as_posix(),
            "web_url": None,
            "source_version": None,
            "fingerprint_sha256": hashlib.sha256(cv_path.read_bytes()).hexdigest().upper(),
            "authority_level": "Faculty-authored primary source",
            "qa_status": f"Machine-extracted with {model}; human review required",
            "accessed_date": today,
            "duplicate_group": None,
            "selected_for_package": True,
            "notes": "Every field in this package was extracted from this CV by structure_cv.py.",
        }],
        "data_quality": {
            "evidence_pattern": "CV only, machine-extracted",
            "missing_sources": [
                "Official USC faculty profile",
                "Institutional award detail (DCG report)",
                "Publication-provider snapshot (e.g. Google Scholar) for citation counts",
                "Official USC patent/invention reconciliation",
            ],
            "conflicts": [],
            "duplicate_sources": [],
            "warnings": ["Machine-extracted from the CV; check every field against the CV before release."]
                        + [f"Extraction note: {n}" for n in notes],
            "source_reference_count": 1,
        },
        "refresh": {
            "package_version": "0.1",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "source_snapshot_date": today,
            "supersedes": None,
        },
        "review_history": [{
            "review_date": today,
            "reviewer": f"structure_cv.py ({model})",
            "decision": "Machine extraction",
            "notes": "Draft built from the CV alone. Needs human review before moving to faculty-json-outputs/.",
        }],
    }


# ---------------------------------------------------------------------------------------------

def build_messages(cv_text, max_pubs):
    return [
        {"role": "system", "content": SYSTEM_PROMPT.replace("{max_pubs}", str(max_pubs))},
        {"role": "user", "content": f"JSON schema for your reply:\n{json.dumps(EXTRACTION_SCHEMA)}\n\n"
                                    f"CV text:\n<cv>\n{cv_text}\n</cv>"},
    ]


def extract_facts(cv_path, model=None, max_pubs=15, max_tokens=16000):
    """Read the CV and have the model pull out its facts. Returns (facts, model_used).
    Raises CVError if the CV or the model's reply can't be used."""
    cv_text = read_cv_text(cv_path)
    client = gateway_client()
    model = model or os.environ.get("USC_AI_GATEWAY_MODEL") or DEFAULT_MODEL

    print(f"Extracting {cv_path.name} with {model} ...")
    facts = call_model(client, model, build_messages(cv_text, max_pubs), max_tokens)
    missing = [key for key in EXTRACTION_SCHEMA["required"] if key not in facts]
    if missing:
        raise CVError(f"The model's reply is missing {', '.join(missing)}. Try again or try another model.")
    return facts, model


def write_outputs(package, facts, out_dir, stem):
    """Save the package and the raw model output. Returns the package path."""
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{stem}.json"
    out_path.write_text(json.dumps(package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / f"{stem}.extraction.json").write_text(
        json.dumps(facts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cv", nargs="?", type=Path, help="CV as .pdf or extracted .txt")
    parser.add_argument("--model", help="gateway model name (default: USC_AI_GATEWAY_MODEL in .env)")
    parser.add_argument("--faculty-id", help="e.g. VIT-FAC-0017 (default: existing ID for this name, else next free)")
    parser.add_argument("--max-pubs", type=int, default=15, help="most recent publications to keep (default 15)")
    parser.add_argument("--max-tokens", type=int, default=16000, help="output token limit (default 16000)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--list-models", action="store_true", help="list models your key can use, then exit")
    parser.add_argument("--dry-run", action="store_true", help="print the prompt instead of calling the gateway")
    args = parser.parse_args()

    if args.list_models:
        list_models()
        return
    if not args.cv:
        parser.error("give a CV path, or --list-models")
    if not args.cv.exists():
        sys.exit(f"No such file: {args.cv}")

    try:
        if args.dry_run:
            for message in build_messages(read_cv_text(args.cv), args.max_pubs):
                print(f"===== {message['role']} ({len(message['content']):,} chars) =====")
                print(message["content"][:3000] + ("\n[...]" if len(message["content"]) > 3000 else ""))
            return

        facts, model = extract_facts(args.cv, args.model, args.max_pubs, args.max_tokens)
        faculty_id = choose_faculty_id(args.faculty_id, facts["canonical_name"])
    except CVError as error:
        sys.exit(str(error))

    package = build_package(facts, faculty_id, args.cv, model, args.max_pubs)
    stem = f"{faculty_id}_{slugify(facts['canonical_name'])}_draft_v0.1"
    out_path = write_outputs(package, facts, args.out_dir, stem)

    print(f"\nWrote {out_path}")
    print(f"  {len(package['identity']['appointments'])} appointments, "
          f"{len(package['publications']['records'])} publications, {len(package['honors'])} honors, "
          f"{len(package['patents_and_inventions'])} patents")
    for note in facts.get("extraction_notes") or []:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()

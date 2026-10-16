"""
Turn a CV PDF into a faculty package JSON using a model on the USC AI gateway.

Steps:
  1. Extract the CV text with PyMuPDF (same method as test_extract.py).
  2. Send the text plus one example package from faculty-json-outputs/ to the model.
  3. Save the model's JSON to llm-json-outputs/ and compare its keys to the example.

The output folder is outside faculty-json-outputs/, so load_db.py never picks it up.

Usage:
    python test_api.py                                  # Culbertson CV, default model
    python test_api.py path/to/cv.pdf --model openai/gpt-5.4
    python test_api.py --list                           # only list models
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pymupdf
from dotenv import load_dotenv

BASE_URL = "https://aigateway-test.its.usc.edu/v1"
DEFAULT_MODEL = "openai/gpt-5.4-mini"
DEFAULT_CV = "CulbertsonHeather_CV_0626.pdf"
PACKAGE_DIR = "faculty-json-outputs"
# A different faculty member, so the model can't copy the existing Culbertson package.
DEFAULT_EXAMPLE = f"{PACKAGE_DIR}/VIT-FAC-0001_hossein_hashemi_draft_v0.3.json"
OUTPUT_DIR = Path("llm-json-outputs")
MAX_RECENT_PUBLICATIONS = 10

SYSTEM_PROMPT = """You convert faculty CVs into structured JSON "faculty packages".

Rules:
- Output one JSON object with exactly the same top-level keys, nesting, field names and
  value conventions as the EXAMPLE PACKAGE. Do not add or rename fields. For a list that
  is empty in the example, copy the fields of its entry in SAMPLE RECORDS.
- Use only facts stated in the CV TEXT. Never invent values. If the CV does not give a
  value, use null for scalar fields and [] for lists.
- The CV is the only source. Put exactly one entry in "sources" for it, using the
  SOURCE METADATA given. Every "source_refs" list must contain only that source_id.
- Use "Proposed - Human Review Required" for every "review_status".
- identity: take name, email, titles and appointments from the CV.
- research: write a 2-3 sentence summary of current work and list domains, methods and
  applications in the same style as the example.
- publications.records: include the {max_pubs} most recent publications in the CV, with
  selection_basis ["recent_cv_view"] and citation_snapshot null. Set doi only if the CV
  states it, and set publication_link to "https://doi.org/<doi>" when there is a doi.
  Use publication_id values like "PUB-<SURNAME>-<YEAR>-<NN>". Put these ids in
  views.recent_publication_ids and leave views.most_cited_publication_ids empty.
- awards.records: one record per grant or contract listed in the CV. Set the rollup counts
  and totals from those records.
- honors and patents_and_inventions: include every item listed in the CV.
- data_quality: describe what the CV did not cover (for example citation counts or
  official USC sources) and any ambiguities you noticed.
- refresh and review_history: use the values given in SOURCE METADATA.
"""


def request(path, api_key, body=None, timeout=60):
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} from {path}:\n{e.read().decode()}")
    except urllib.error.URLError as e:
        sys.exit(f"Could not reach {BASE_URL}: {e.reason}")


def extract_text(pdf_path):
    with pymupdf.open(pdf_path) as doc:
        # sort=True: top-to-bottom, left-to-right reading order
        return "\n\n".join(page.get_text("text", sort=True) for page in doc), len(doc)


def key_paths(value, prefix=""):
    """Set of key paths in a JSON value. List items are merged under "[]"."""
    paths = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            paths.add(path)
            paths |= key_paths(child, path)
    elif isinstance(value, list):
        for child in value:
            paths |= key_paths(child, f"{prefix}[]")
    return paths


def sample_records(example, packages):
    """One record for each top-level list that is empty in the example, taken from another package."""
    samples = {}
    for key, value in example.items():
        if isinstance(value, list) and not value:
            for package in packages:
                if package.get(key):
                    samples[key] = package[key][0]
                    break
    return samples


def source_metadata(pdf_path, faculty_id):
    now = datetime.datetime.now(datetime.timezone.utc)
    return {
        "faculty_id": faculty_id,
        "cv_source": {
            "source_id": f"src-cv-{re.sub(r'[^a-z0-9]+', '_', pdf_path.stem.lower())}",
            "source_type": "faculty_cv",
            "repository_path": str(pdf_path),
            "web_url": None,
            "fingerprint_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest().upper(),
            "authority_level": "Faculty-authored evidence",
            "qa_status": "Text extracted automatically; human review required",
            "accessed_date": now.date().isoformat(),
            "selected_for_package": True,
        },
        "refresh": {
            "package_version": "0.1",
            "created_at": now.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "source_snapshot_date": now.date().isoformat(),
            "supersedes": None,
        },
        "review_history": [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", default=DEFAULT_CV)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--example", default=DEFAULT_EXAMPLE, help="package JSON to copy the format from")
    parser.add_argument("--faculty-id", default=None, help="e.g. VIT-FAC-0010")
    parser.add_argument("--list", action="store_true", help="only list models")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("API_KEY")
    if not api_key:
        sys.exit("API_KEY is not set in .env")

    if args.list:
        models = request("/models", api_key)
        for model_id in sorted(m["id"] for m in models.get("data", [])):
            print(model_id)
        return

    pdf_path = Path(args.pdf)
    cv_text, page_count = extract_text(pdf_path)
    print(f"{pdf_path.name}: {page_count} page(s), {len(cv_text)} chars extracted")

    example_text = Path(args.example).read_text(encoding="utf-8")
    example = json.loads(example_text)
    packages = [json.loads(Path(p).read_text(encoding="utf-8")) for p in sorted(Path(PACKAGE_DIR).glob("VIT-FAC-*.json"))]
    samples = sample_records(example, packages)
    metadata = source_metadata(pdf_path, args.faculty_id)
    user_prompt = (
        f"EXAMPLE PACKAGE (format only; this is a different person):\n{example_text}\n\n"
        + (f"SAMPLE RECORDS (format for lists that are empty in the example):\n{json.dumps(samples, indent=2)}\n\n" if samples else "")
        + f"SOURCE METADATA:\n{json.dumps(metadata, indent=2)}\n\n"
        f"CV TEXT:\n{cv_text}"
    )

    print(f"Querying {args.model} ...")
    start = time.time()
    result = request("/chat/completions", api_key, {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(max_pubs=MAX_RECENT_PUBLICATIONS)},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }, timeout=600)
    print(f"Done in {time.time() - start:.0f}s. Usage: {result.get('usage', {}).get('total_tokens')} tokens")

    content = result["choices"][0]["message"]["content"]
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{pdf_path.stem}.json"
    try:
        package = json.loads(content)
    except json.JSONDecodeError as e:
        raw_path = out_path.with_suffix(".raw.txt")
        raw_path.write_text(content, encoding="utf-8")
        sys.exit(f"Model output is not valid JSON ({e}). Raw output saved to {raw_path}")

    out_path.write_text(json.dumps(package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {out_path}")

    # Compare structure with every key used across the existing packages. Only report a
    # missing key when its parent holds data in the output (not null or an empty list).
    expected = set().union(*(key_paths(p) for p in packages))
    actual = key_paths(package)
    parents = {p.rsplit(".", 1)[0] for p in actual if "." in p}
    missing = sorted(p for p in expected - actual if "." not in p or p.rsplit(".", 1)[0] in parents)
    extra = sorted(actual - expected)
    print(f"\nStructure check against keys used in {len(packages)} packages in {PACKAGE_DIR}/:")
    print(f"  missing keys: {missing or 'none'}")
    print(f"  extra keys:   {extra or 'none'}")
    print(
        f"\nCounts: {len(package.get('publications', {}).get('records', []))} publications, "
        f"{len(package.get('awards', {}).get('records', []))} awards, "
        f"{len(package.get('honors', []))} honors, "
        f"{len(package.get('patents_and_inventions', []))} patents"
    )


if __name__ == "__main__":
    main()

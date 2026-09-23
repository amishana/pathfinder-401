"""Loads faculty package JSON files, embeds each evidence piece once, and stores
everything in postgres. Safe to re-run: each faculty's evidence rows get wiped and
reinserted fresh, so nothing duplicates and stale evidence doesn't linger.

Usage:
    python ingest.py /path/to/faculty_packages
"""

import argparse
import glob
import json
import os

from sentence_transformers import SentenceTransformer

from config import MODEL, get_conn
from evidence import extract_evidence


def load_packages(data_dir):
    packages = []
    for path in sorted(glob.glob(os.path.join(data_dir, "VIT-FAC-*.json"))):
        with open(path) as f:
            pkg = json.load(f)
        faculty_id = pkg.get("faculty_id")
        if not faculty_id:
            print(f"  skipping {path}: no faculty_id")
            continue
        name = (pkg.get("identity") or {}).get("canonical_name") or faculty_id
        packages.append((faculty_id, name, pkg))
    return packages


def ingest(data_dir):
    packages = load_packages(data_dir)
    if not packages:
        print(f"no VIT-FAC-*.json files found in {data_dir}")
        return

    print(f"loading model {MODEL}...")
    model = SentenceTransformer(MODEL)

    conn = get_conn()
    total_evidence = 0
    try:
        with conn:
            with conn.cursor() as cur:
                for faculty_id, name, pkg in packages:
                    pieces = extract_evidence(pkg)
                    texts = [p["text"] for p in pieces]
                    embeddings = (
                        model.encode(texts, normalize_embeddings=True)
                        if texts
                        else []
                    )

                    cur.execute(
                        """
                        INSERT INTO faculty (faculty_id, canonical_name, updated_at)
                        VALUES (%s, %s, now())
                        ON CONFLICT (faculty_id)
                        DO UPDATE SET canonical_name = EXCLUDED.canonical_name,
                                      updated_at = now()
                        """,
                        (faculty_id, name),
                    )

                    cur.execute("DELETE FROM evidence WHERE faculty_id = %s", (faculty_id,))

                    for piece, embedding in zip(pieces, embeddings):
                        cur.execute(
                            """
                            INSERT INTO evidence (faculty_id, kind, text, year, embedding)
                            VALUES (%s, %s, %s, %s, %s)
                            """,
                            (faculty_id, piece["kind"], piece["text"], piece["year"], embedding),
                        )

                    total_evidence += len(pieces)
                    print(f"  {faculty_id}  {name}  ({len(pieces)} evidence pieces)")
    finally:
        conn.close()

    print(f"\ningested {len(packages)} faculty, {total_evidence} evidence pieces total.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="./faculty_packages",
        help="directory containing VIT-FAC-*.json files",
    )
    args = parser.parse_args()
    ingest(args.data_dir)


if __name__ == "__main__":
    main()

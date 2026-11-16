"""Reads a queries file (query + expected faculty ids) and reports how many of the
expected faculty show up in the top 5 search results.

Usage:
    python eval.py queries.example.json
"""

import argparse
import json

from sentence_transformers import SentenceTransformer

from config import MODEL, get_conn
from search import search


def run_eval(queries_path):
    with open(queries_path) as f:
        queries = json.load(f)

    model = SentenceTransformer(MODEL)
    conn = get_conn()

    total_expected = 0
    total_found = 0

    try:
        for entry in queries:
            query = entry["query"]
            expected = entry.get("expected_faculty_ids") or []
            results = search(query, model=model, conn=conn)
            top_ids = [r["faculty_id"] for r in results]

            found = [fid for fid in expected if fid in top_ids]
            total_expected += len(expected)
            total_found += len(found)

            print(f"Query: {query}")
            print(f"  expected: {expected}")
            print(f"  top 5: {top_ids}")
            print(f"  found: {len(found)}/{len(expected)}\n")
    finally:
        conn.close()

    if total_expected == 0:
        print("no expected faculty ids in queries file, nothing to score")
        return

    recall = total_found / total_expected
    print(f"overall: {total_found}/{total_expected} expected faculty found in top 5 ({recall:.0%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("queries_path", nargs="?", default="queries.example.json")
    args = parser.parse_args()
    run_eval(args.queries_path)


if __name__ == "__main__":
    main()

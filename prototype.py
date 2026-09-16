"""Proof of concept: semantic faculty search over the 16 benchmark packages.

Usage:
    python prototype.py "<research topic or funding opportunity>"

Example:
    $ python prototype.py "FPGA and parallel/distributed computing for high performance applications"

    1. VIT-FAC-0005  Viktor K. Prasanna  (score 0.81)
         [pub 1993] Heterogeneous computing: Challenges and opportunities  (0.83)
         [pub 2026] HERA: A Bandwidth-efficient Accelerator for Fully Homomorphic Encryption on HBM-enabled FPGA  (0.81)
         [pub 2026] NysX: An Accurate and Energy-Efficient FPGA Accelerator for Hyperdimensional Graph Classification  (0.81)

Faculty are ranked by the mean score of their strongest matching evidence, and each
result prints the publications, awards, or patents that produced the match.
"""

import glob
import json
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-small-en-v1.5"
TOP_FACULTY = 5
TOP_EVIDENCE = 3


def load_evidence():
    """Flatten every package into one list of (faculty_id, name, kind, text) evidence pieces."""
    evidence = []
    for path in sorted(glob.glob("VIT-FAC-*.json")):
        pkg = json.load(open(path))
        fid = pkg["faculty_id"]
        name = pkg["identity"]["canonical_name"]

        def add(kind, text):
            evidence.append((fid, name, kind, text))

        research = pkg["research"]
        add("research", f"{research['summary']} Areas: {', '.join(research['domains'])}.")

        for pub in pkg["publications"]["records"]:
            venue = f" ({pub['venue']})" if pub["venue"] else ""
            add(f"pub {pub['publication_year']}", f"{pub['title']}{venue}")

        for award in pkg["awards"]["records"]:
            add(f"award {award['status']}", f"{award['title']} - sponsor: {award['sponsor']}")

        for patent in pkg.get("patents_and_inventions") or []:
            add(f"patent {patent['status']}", patent["title"])

        for honor in pkg.get("honors") or []:
            add("honor", honor["honor"])

    return evidence


def rank(query, evidence, vectors, model):
    """Score each faculty member by the mean of their strongest matching evidence."""
    query_vector = model.encode(query, normalize_embeddings=True)
    scores = vectors @ query_vector

    by_faculty = {}
    for (fid, name, kind, text), score in zip(evidence, scores):
        by_faculty.setdefault((fid, name), []).append((score, kind, text))

    ranked = []
    for (fid, name), hits in by_faculty.items():
        hits.sort(reverse=True)
        top = hits[:TOP_EVIDENCE]
        ranked.append((float(np.mean([h[0] for h in top])), fid, name, top))
    ranked.sort(reverse=True)
    return ranked


def main():
    query = " ".join(sys.argv[1:]) or "haptic interfaces for rehabilitation or medical applications"

    evidence = load_evidence()
    model = SentenceTransformer(MODEL)
    vectors = model.encode([text for *_, text in evidence], normalize_embeddings=True)
    print(f"\nIndexed {len(evidence)} evidence pieces from 16 faculty packages.")
    print(f"Query: {query}\n")

    for position, (score, fid, name, top) in enumerate(rank(query, evidence, vectors, model)[:TOP_FACULTY], 1):
        print(f"{position}. {fid}  {name}  (score {score:.2f})")
        for hit_score, kind, text in top:
            print(f"     [{kind}] {text[:100]}  ({hit_score:.2f})")
        print()


if __name__ == "__main__":
    main()

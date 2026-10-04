"""Proof of concept: hybrid (keyword + semantic) faculty search over the 16 benchmark packages.

Usage:
    python prototype.py "<research topic or funding opportunity>"

Example:
    $ python prototype.py "FPGA and parallel/distributed computing for high performance applications"

    1. VIT-FAC-0005  Viktor K. Prasanna  (score 0.96)
         [research] Research spans high-performance and reconfigurable computing, FPGA accelerators, ...  (0.97 · cos 0.78 · kw #1)
         [pub 2001] Fast regular expression matching using FPGAs  (0.96 · cos 0.80 · kw #3)
         [pub 1993] Heterogeneous computing: Challenges and opportunities  (0.94 · cos 0.83 · kw #9)

    Each evidence line shows the fused score, the cosine similarity, and the keyword rank
    ("kw -" means no query word matched).

Each evidence piece is ranked twice: by embedding similarity and by BM25 keyword score.
The two rankings are merged with Reciprocal Rank Fusion (RRF), so rare exact terms like
"FPGA" count even when the embedding prefers a broader match. Faculty are ranked by the
mean fused score of their strongest matching evidence, and each result prints the
publications, awards, or patents that produced the match.

Scores are scaled so 1.00 means "ranked first by both methods". Treat them as ordering,
not quality.
"""

import glob
import json
import re
import sys

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-small-en-v1.5"
TOP_FACULTY = 5
TOP_EVIDENCE = 3
RRF_K = 60  # standard RRF constant; larger values flatten the gap between top ranks

# Common words that would otherwise give a keyword rank to most evidence pieces.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is",
    "it", "its", "of", "on", "or", "that", "the", "to", "via", "with", "who", "what",
    "which", "work", "works", "working",
}


def load_evidence():
    """Flatten every package into one list of (faculty_id, name, kind, text) evidence pieces."""
    evidence = []
    for path in sorted(glob.glob("faculty-json-outputs/VIT-FAC-*.json")):
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


def tokenize(text):
    """Lowercase words for keyword matching. Applied the same way to evidence and queries.

    Splits on anything that is not a letter or digit ("parallel/distributed" -> two words),
    drops common words, and strips a trailing "s" so "FPGAs" matches "FPGA".
    """
    tokens = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        tokens.append(word)
    return tokens


def rank_positions(scores):
    """1-based rank of each score, highest first. Tied scores share the same rank."""
    return (scores[None, :] > scores[:, None]).sum(axis=1) + 1


def rank(query, evidence, vectors, model, bm25):
    """Score each faculty member by the mean of their strongest matching evidence.

    Each evidence piece gets 1/(RRF_K + embedding rank) + 1/(RRF_K + keyword rank).
    Pieces with no keyword match get no keyword term, so unrelated pieces are not boosted.
    """
    query_vector = model.encode(query, normalize_embeddings=True)
    cosine = vectors @ query_vector
    keyword = bm25.get_scores(tokenize(query))

    embedding_ranks = rank_positions(cosine)
    keyword_ranks = rank_positions(keyword)
    has_keyword = keyword > 0

    fused = 1 / (RRF_K + embedding_ranks)
    fused = fused + np.where(has_keyword, 1 / (RRF_K + keyword_ranks), 0)
    fused = fused / (2 / (RRF_K + 1))  # 1.00 = ranked first by both methods

    by_faculty = {}
    for i, (fid, name, kind, text) in enumerate(evidence):
        keyword_rank = int(keyword_ranks[i]) if has_keyword[i] else None
        hit = (float(fused[i]), float(cosine[i]), keyword_rank, kind, text)
        by_faculty.setdefault((fid, name), []).append(hit)

    ranked = []
    for (fid, name), hits in by_faculty.items():
        hits.sort(key=lambda hit: hit[0], reverse=True)
        top = hits[:TOP_EVIDENCE]
        ranked.append((float(np.mean([hit[0] for hit in top])), fid, name, top))
    ranked.sort(key=lambda result: result[0], reverse=True)
    return ranked


def main():
    query = " ".join(sys.argv[1:]) or "haptic interfaces for rehabilitation or medical applications"

    evidence = load_evidence()
    model = SentenceTransformer(MODEL)
    vectors = model.encode([text for *_, text in evidence], normalize_embeddings=True)
    bm25 = BM25Okapi([tokenize(text) for *_, text in evidence])
    print(f"\nIndexed {len(evidence)} evidence pieces from 16 faculty packages.")
    print(f"Query: {query}\n")

    for position, (score, fid, name, top) in enumerate(rank(query, evidence, vectors, model, bm25)[:TOP_FACULTY], 1):
        print(f"{position}. {fid}  {name}  (score {score:.2f})")
        for fused, cosine, keyword_rank, kind, text in top:
            keyword_label = f"kw #{keyword_rank}" if keyword_rank else "kw -"
            print(f"     [{kind}] {text[:100]}  ({fused:.2f} · cos {cosine:.2f} · {keyword_label})")
        print()


if __name__ == "__main__":
    main()

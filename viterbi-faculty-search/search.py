"""Embeds a query and ranks faculty by the mean of their top-N matching evidence
scores, done in SQL. Same idea as prototype.py's rank(), but faculty with fewer than
TOP_EVIDENCE pieces don't get an inflated average: we always divide by TOP_EVIDENCE,
so missing slots count as 0 instead of shrinking the denominator.

Usage:
    python search.py "<research topic or funding opportunity>"
"""

import sys

from sentence_transformers import SentenceTransformer

from config import MODEL, TOP_EVIDENCE, TOP_FACULTY, get_conn

# window function needs to see every evidence row to rank per faculty, so this
# does a seq scan instead of using the hnsw index. fine at this size (hundreds
# of rows), would need a different approach (e.g. knn per faculty via lateral
# join) if the evidence table got a lot bigger.
SEARCH_SQL = """
WITH ranked AS (
    SELECT
        faculty_id,
        kind,
        text,
        year,
        1 - (embedding <=> %(query_vec)s) AS score,
        ROW_NUMBER() OVER (
            PARTITION BY faculty_id ORDER BY embedding <=> %(query_vec)s
        ) AS rn
    FROM evidence
),
top_evidence AS (
    SELECT * FROM ranked WHERE rn <= %(top_evidence)s
),
scored AS (
    SELECT
        faculty_id,
        SUM(score) / %(top_evidence)s AS avg_score,
        json_agg(
            json_build_object('kind', kind, 'text', text, 'year', year, 'score', score)
            ORDER BY score DESC
        ) AS evidence
    FROM top_evidence
    GROUP BY faculty_id
)
SELECT f.faculty_id, f.canonical_name, s.avg_score, s.evidence
FROM scored s
JOIN faculty f ON f.faculty_id = s.faculty_id
ORDER BY s.avg_score DESC
LIMIT %(top_faculty)s;
"""


def search(query, model=None, conn=None, top_faculty=TOP_FACULTY, top_evidence=TOP_EVIDENCE):
    """Returns a list of dicts: {faculty_id, name, score, evidence: [...]}."""
    owns_conn = conn is None
    model = model or SentenceTransformer(MODEL)
    conn = conn or get_conn()
    try:
        query_vec = model.encode(query, normalize_embeddings=True)
        with conn.cursor() as cur:
            cur.execute(
                SEARCH_SQL,
                {
                    "query_vec": query_vec,
                    "top_evidence": top_evidence,
                    "top_faculty": top_faculty,
                },
            )
            rows = cur.fetchall()
    finally:
        if owns_conn:
            conn.close()

    results = []
    for faculty_id, name, avg_score, evidence in rows:
        results.append(
            {
                "faculty_id": faculty_id,
                "name": name,
                "score": float(avg_score),
                "evidence": evidence,
            }
        )
    return results


def main():
    query = " ".join(sys.argv[1:]) or "haptic interfaces for rehabilitation or medical applications"
    print(f"Query: {query}\n")
    for position, result in enumerate(search(query), 1):
        print(f"{position}. {result['faculty_id']}  {result['name']}  (score {result['score']:.2f})")
        for hit in result["evidence"]:
            print(f"     [{hit['kind']}] {hit['text'][:100]}  ({hit['score']:.2f})")
        print()


if __name__ == "__main__":
    main()

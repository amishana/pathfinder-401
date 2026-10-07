"""Faculty search that runs on the database instead of the JSON files.

Usage:
    python search_db.py "<research topic or funding opportunity>"

Needs DATABASE_URL in .env, and a database filled by load_db.py.

The database ranks every evidence piece twice:
    - pgvector ranks by embedding similarity (cosine distance, the <=> operator).
    - Postgres full-text search ranks by keyword match (the tsv column and ts_rank).
Python then merges the two rankings and scores faculty with rank_faculty() from
prototype.py, so both search paths use the same scoring code.

The keyword ranking differs from prototype.py. Postgres uses ts_rank and English
stemming, while the prototype uses BM25. ts_rank counts how often query words appear
in a piece, but unlike BM25 it does not give extra weight to rare words like "FPGA".
Results can therefore differ slightly between the two paths.
"""

import os
import sys

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from prototype import MODEL, print_results, rank_faculty, tokenize

# Ranks every evidence piece. rank() gives tied scores the same rank, like rank_positions()
# in prototype.py. Only pieces that match a query word get a keyword rank.
SEARCH_SQL = """
with scored as (
    select e.faculty_id, f.canonical_name, e.kind, e.text,
           1 - (e.embedding <=> %(vector)s) as cosine,
           case when e.tsv @@ to_tsquery('english', %(words)s)
                then ts_rank(e.tsv, to_tsquery('english', %(words)s)) end as keyword_score
    from evidence e
    join faculty f using (faculty_id)
)
select faculty_id, canonical_name,
       rank() over (order by cosine desc) as embedding_rank,
       case when keyword_score is not null
            then rank() over (order by keyword_score desc nulls last) end as keyword_rank,
       cosine, kind, text
from scored
"""


def search(conn, model, query):
    """Rank faculty for a query, using evidence and embeddings stored in the database."""
    vector = model.encode(query, normalize_embeddings=True)
    # Any query word may match ("|" means OR), like BM25 in prototype.py.
    # None (no usable words) makes every keyword rank empty.
    words = " | ".join(tokenize(query)) or None
    hits = conn.execute(SEARCH_SQL, {"vector": vector, "words": words}).fetchall()
    return rank_faculty(hits)


def main():
    query = " ".join(sys.argv[1:]) or "haptic interfaces for rehabilitation or medical applications"

    load_dotenv()
    if "DATABASE_URL" not in os.environ:
        sys.exit("DATABASE_URL is not set. Copy .env.example to .env and fill in the password.")

    model = SentenceTransformer(MODEL)
    # prepare_threshold=None: the Supabase transaction pooler does not support prepared statements.
    with psycopg.connect(os.environ["DATABASE_URL"], prepare_threshold=None) as conn:
        register_vector(conn)
        print()
        print_results(query, search(conn, model, query))


if __name__ == "__main__":
    main()

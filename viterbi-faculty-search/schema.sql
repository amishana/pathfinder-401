-- faculty search schema, pgvector-backed

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS faculty (
    faculty_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evidence (
    id BIGSERIAL PRIMARY KEY,
    faculty_id TEXT NOT NULL REFERENCES faculty(faculty_id) ON DELETE CASCADE,
    kind TEXT NOT NULL,         -- research, publication, award, patent, honor
    text TEXT NOT NULL,
    year INT,                   -- null when the source package doesn't have one
    embedding VECTOR(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS evidence_faculty_id_idx ON evidence (faculty_id);

-- hnsw index for cosine distance. note: our search query partitions/ranks
-- per faculty with a window function, which needs the full evidence set in
-- scope, so postgres ends up doing a seq scan instead of using this index.
-- fine at this size (a few hundred rows), would matter more at real scale.
CREATE INDEX IF NOT EXISTS evidence_embedding_hnsw_idx
    ON evidence USING hnsw (embedding vector_cosine_ops);

-- Database schema for Viterbi Pathfinder (Postgres + pgvector).
--
-- Use this file only if the database ever needs to be rebuilt from scratch, for example:
--   - a new Supabase project,
--   - a move to USC's Azure or AWS Postgres,
--   - a teammate's local copy (supabase start).
--
-- How to rebuild:
--   1. Run this whole file once on an empty database (Supabase: SQL Editor > paste > Run).
--   2. Load the data from the JSON packages:  python load_db.py
--
-- Do not run it on a database that already has these tables. It stops with
-- "relation already exists". To change a table that exists, write an ALTER TABLE
-- statement, run it, and then update this file so it stays a copy of the live schema.
--
-- The JSON packages in git are the source of truth. Every table below can be refilled
-- from them, so this file holds the structure only, never data.

create extension if not exists vector;

-- One row per faculty member. faculty_id is the only key (no USC IDs).
create table faculty (
  faculty_id      text primary key check (faculty_id ~ '^VIT-FAC-[0-9]{4}$'),
  canonical_name  text not null,
  display_name    text,
  current_version int
);

-- Full JSON history: the database copy of the git packages.
create table package_versions (
  faculty_id     text references faculty on delete cascade,
  version        int,
  schema_version text not null,
  package        jsonb not null,
  loaded_at      timestamptz default now(),
  primary key (faculty_id, version)
);

create table publications (
  publication_id text primary key,
  faculty_id     text references faculty on delete cascade,
  title          text not null,
  venue          text,
  year           int,
  doi            text,
  review_status  text
);

create table awards (
  award_record_id text primary key,
  faculty_id      text references faculty on delete cascade,
  title           text not null,
  sponsor         text,
  status          text,          -- as reported by the source, never inferred
  start_date      date,
  end_date        date,
  obligated_total_usd numeric,
  review_status   text
);

create table patents (
  patent_record_id text primary key,
  faculty_id       text references faculty on delete cascade,
  title            text not null,
  status           text,
  event_date       text,         -- text, not date: some sources give only a year ("2025")
  review_status    text
);

-- One row per searchable piece of evidence (same pieces prototype.py builds).
create table evidence (
  id          bigserial primary key,
  faculty_id  text references faculty on delete cascade,
  kind        text not null,      -- research, pub, award, patent, honor, cv
  source_ref  text,               -- e.g. publication_id or award_record_id
  text        text not null,
  embedding   vector(384),        -- bge-small-en-v1.5 output size
  tsv         tsvector generated always as (to_tsvector('english', text)) stored
);

create index on evidence using gin (tsv);
create index on evidence (faculty_id);

-- Row level security with no policies: the public Supabase API key can read and write
-- nothing. The loader and the backend connect as the postgres user, which skips these checks.
alter table faculty          enable row level security;
alter table package_versions enable row level security;
alter table publications     enable row level security;
alter table awards           enable row level security;
alter table patents          enable row level security;
alter table evidence         enable row level security;

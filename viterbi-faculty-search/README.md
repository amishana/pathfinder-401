# Viterbi Faculty Search

Semantic search over Viterbi faculty packages, backed by postgres + pgvector. Faculty
are ranked by the mean of their top 3 matching evidence pieces (research summary,
publications, awards, patents, honors).

## Setup

1. Start postgres:

   ```
   docker compose up -d
   ```

   This creates the database and runs `schema.sql` automatically, but only on the
   *first* boot of a fresh volume. If you change `schema.sql` later and want it to
   take effect, you need to reset the volume:

   ```
   docker compose down -v
   docker compose up -d
   ```

   `-v` deletes the named volume, so this wipes all ingested data too, you'll need to
   re-run ingest afterward.

2. Install python deps (a virtualenv is recommended):

   ```
   pip install -r requirements.txt
   ```

3. Ingest the faculty packages. Point it at the folder with your `VIT-FAC-*.json`
   files:

   ```
   python ingest.py /path/to/faculty_packages
   ```

   Safe to run again any time the source JSON changes, it replaces each faculty's
   evidence rows rather than appending, so nothing duplicates.

## Usage

Run a search:

```
python search.py "FPGA and parallel/distributed computing for high performance applications"
```

Prints the top 5 faculty with their score and the 3 evidence pieces that drove the
match, same format as the original prototype.py.

Run the eval:

```
python eval.py queries.example.json
```

Fill in `queries.example.json` (or copy it to your own file) with real queries and the
faculty IDs you'd expect to see for each one. The eval reports how many of the
expected faculty actually showed up in the top 5.

## Notes

- Embeddings use `BAAI/bge-small-en-v1.5` with `normalize_embeddings=True` on both the
  evidence side (ingest) and the query side (search), so cosine similarity scores line
  up with what the original prototype produced.
- Faculty with fewer than 3 evidence pieces don't get an inflated score: the search
  query always divides by 3, so missing evidence slots count as 0 instead of shrinking
  the denominator.
- The search query ranks each faculty's evidence with a window function, which needs
  to look at every evidence row, so it does a sequential scan rather than using the
  hnsw index. That's fine at the current data size (a few hundred evidence rows), just
  worth knowing if the dataset grows a lot.
- DB connection settings can be overridden with `DB_HOST`, `DB_PORT`, `DB_NAME`,
  `DB_USER`, `DB_PASSWORD` env vars, they default to match `docker-compose.yml`.
- On Intel Macs, torch 2.2.2 is the last version published, and it isn't compatible
  with numpy 2.x or with the latest transformers/sentence-transformers (which now
  require torch >= 2.5). `requirements.txt` pins `numpy<2`, `sentence-transformers<4`,
  and `transformers<4.50` for that reason. If you're on Apple Silicon or Linux you
  likely don't need these pins, but they're harmless either way.

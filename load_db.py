"""Load the faculty JSON packages into the Supabase Postgres database.

When to run:
    - Once, after you create the tables, to fill the empty database.
    - Each time new or edited packages are merged into main.
    - After someone breaks data in the database. Running again restores it from the packages.

Run it from the repo root, on main. If two people run it from branches with different
package files, the database ends up with whichever run finished last.

Usage:
    python load_db.py                              # load every package
    python load_db.py VIT-FAC-0010 VIT-FAC-0016    # load only these faculty members

Needs DATABASE_URL in .env (copy .env.example and fill in the password). Use the
transaction pooler string (port 6543). The session pooler (port 5432) drops connections
on some networks.

Git is the source of truth. The database is a copy that this script can rebuild at any time,
so do not edit rows by hand in the Supabase dashboard. The next run overwrites them.

For each faculty member, in one transaction, the script:
    1. Inserts or updates their row in faculty.
    2. Adds a row to package_versions, but only if the JSON changed since their latest version.
    3. Deletes and reinserts their publications, awards, patents, and evidence rows.
If anything fails for one faculty member, their data stays as it was and the script moves on.

Running it twice in a row is safe: the second run adds no versions and produces the same rows.
Packages deleted from git are not deleted from the database.
"""

import glob
import json
import os
import sys

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb
from sentence_transformers import SentenceTransformer

from prototype import MODEL, package_evidence

PACKAGE_GLOB = "faculty-json-outputs/VIT-FAC-*.json"


def read_packages(faculty_ids):
    """Return the packages to load. Exits if a faculty ID has no package or more than one."""
    by_id = {}
    for path in sorted(glob.glob(PACKAGE_GLOB)):
        pkg = json.load(open(path))
        by_id.setdefault(pkg["faculty_id"], []).append((path, pkg))

    duplicates = {fid: [path for path, _ in found] for fid, found in by_id.items() if len(found) > 1}
    if duplicates:
        sys.exit(f"More than one package file for the same faculty_id: {duplicates}")

    missing = [fid for fid in faculty_ids if fid not in by_id]
    if missing:
        sys.exit(f"No package file found for: {', '.join(missing)}")

    wanted = faculty_ids or sorted(by_id)
    return [by_id[fid][0][1] for fid in wanted]


def load_one(conn, pkg, model):
    """Write one package to the database. Returns a short status line."""
    fid = pkg["faculty_id"]
    identity = pkg["identity"]

    # Compute embeddings before the transaction opens, so the transaction stays short.
    pieces = package_evidence(pkg)
    vectors = model.encode([text for _, text, _ in pieces], normalize_embeddings=True)

    with conn.transaction():
        latest = conn.execute(
            """select version, package = %s from package_versions
               where faculty_id = %s order by version desc limit 1""",
            (Jsonb(pkg), fid),
        ).fetchone()
        if latest is None:
            version, changed = 1, True
        elif latest[1]:
            version, changed = latest[0], False
        else:
            version, changed = latest[0] + 1, True

        conn.execute(
            """insert into faculty (faculty_id, canonical_name, display_name, current_version)
               values (%s, %s, %s, %s)
               on conflict (faculty_id) do update set
                 canonical_name = excluded.canonical_name,
                 display_name = excluded.display_name,
                 current_version = excluded.current_version""",
            (fid, identity["canonical_name"], identity["preferred_display_name"], version),
        )
        if changed:
            conn.execute(
                """insert into package_versions (faculty_id, version, schema_version, package)
                   values (%s, %s, %s, %s)""",
                (fid, version, pkg["schema_version"], Jsonb(pkg)),
            )

        for table in ("publications", "awards", "patents", "evidence"):
            conn.execute(f"delete from {table} where faculty_id = %s", (fid,))

        with conn.cursor() as cur:
            cur.executemany(
                """insert into publications
                   (publication_id, faculty_id, title, venue, year, doi, review_status)
                   values (%s, %s, %s, %s, %s, %s, %s)""",
                [
                    (pub["publication_id"], fid, pub["title"], pub["venue"],
                     pub["publication_year"], pub["doi"], pub["review_status"])
                    for pub in pkg["publications"]["records"]
                ],
            )
            cur.executemany(
                """insert into awards
                   (award_record_id, faculty_id, title, sponsor, status, start_date, end_date,
                    obligated_total_usd, review_status)
                   values (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    (award["award_record_id"], fid, award["title"], award["sponsor"],
                     award["status"], award["start_date"], award["end_date"],
                     award["obligated_total_usd"], award["review_status"])
                    for award in pkg["awards"]["records"]
                ],
            )
            cur.executemany(
                """insert into patents
                   (patent_record_id, faculty_id, title, status, event_date, review_status)
                   values (%s, %s, %s, %s, %s, %s)""",
                [
                    (patent["patent_record_id"], fid, patent["title"], patent["status"],
                     patent["event_date"], patent["review_status"])
                    for patent in pkg.get("patents_and_inventions") or []
                ],
            )
            cur.executemany(
                """insert into evidence (faculty_id, kind, source_ref, text, embedding)
                   values (%s, %s, %s, %s, %s)""",
                [
                    (fid, kind, source_ref, text, vector)
                    for (kind, text, source_ref), vector in zip(pieces, vectors)
                ],
            )

    state = "new" if version == 1 and changed else "changed" if changed else "unchanged"
    return f"{state}, version {version}, {len(pieces)} evidence pieces"


def main():
    load_dotenv()
    if "DATABASE_URL" not in os.environ:
        sys.exit("DATABASE_URL is not set. Copy .env.example to .env and fill in the password.")

    packages = read_packages(sys.argv[1:])
    model = SentenceTransformer(MODEL)

    failed = []
    # prepare_threshold=None turns off prepared statements, which the Supabase
    # transaction pooler (port 6543) does not support.
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True, prepare_threshold=None) as conn:
        register_vector(conn)
        for pkg in packages:
            fid = pkg["faculty_id"]
            try:
                print(f"{fid}  {load_one(conn, pkg, model)}")
            except psycopg.Error as error:
                failed.append(fid)
                print(f"{fid}  FAILED, nothing changed: {error}")

    print(f"\nLoaded {len(packages) - len(failed)} of {len(packages)} packages.")
    if failed:
        sys.exit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()

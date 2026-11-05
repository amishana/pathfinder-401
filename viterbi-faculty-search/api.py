"""FastAPI backend for Viterbi Pathfinder.

Hybrid search (cosine + keyword RRF) runs against Supabase via DATABASE_URL.
Faculty package JSON is served directly from the local faculty-json-outputs/ folder.

Usage:
    uvicorn api:app --reload
"""

import glob
import json
import os
import sys

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

# allow imports from repo root (search_db.py, prototype.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from search_db import search as run_search  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

MODEL = "BAAI/bge-small-en-v1.5"

# faculty packages loaded once at startup for /faculty/{id} lookups
_PACKAGES: dict[str, dict] = {}

def _load_packages():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "faculty-json-outputs")
    for path in glob.glob(os.path.join(data_dir, "VIT-FAC-*.json")):
        with open(path) as f:
            pkg = json.load(f)
        fid = pkg.get("faculty_id")
        if fid:
            _PACKAGES[fid] = pkg


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
_conn = None


@app.on_event("startup")
def startup():
    global model, _conn
    _load_packages()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set — add it to .env")

    print("connecting to Supabase...")
    _conn = psycopg.connect(database_url, prepare_threshold=None)
    register_vector(_conn)

    print(f"loading model {MODEL}...")
    model = SentenceTransformer(MODEL)
    print("ready")


@app.on_event("shutdown")
def shutdown():
    if _conn:
        _conn.close()


@app.get("/search")
def search_endpoint(q: str):
    # rank_faculty returns [(score, faculty_id, name, [(fused, cosine, kw_rank, kind, text), ...]), ...]
    results = run_search(_conn, model, q)
    formatted = [
        {
            "faculty_id": fid,
            "name": name,
            "score": score,
            "evidence": [
                {"kind": kind, "text": text, "score": fused}
                for fused, _cosine, _kw, kind, text in pieces
            ],
        }
        for score, fid, name, pieces in results
    ]
    return {"query": q, "results": formatted}


@app.get("/faculty/{faculty_id}")
def faculty_endpoint(faculty_id: str):
    pkg = _PACKAGES.get(faculty_id)
    if not pkg:
        raise HTTPException(status_code=404, detail="faculty not found")
    return pkg

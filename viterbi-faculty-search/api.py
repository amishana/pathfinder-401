"""FastAPI backend for Viterbi Pathfinder.

Hybrid search (cosine + keyword RRF) runs against Supabase via DATABASE_URL.
Faculty packages are served from Supabase (latest version), falling back to the local
faculty-json-outputs/ folder.

CV uploads (see cv_upload.py) never write to Supabase directly:
    POST /cv-upload                  PDF -> structured draft, pending review
    GET  /reviews?status=pending     drafts for the Profile review page
    GET  /reviews/{id}               one draft: what it adds, conflicts, resulting package
    GET  /reviews/{id}/cv            the uploaded PDF, to check the draft against
    POST /reviews/{id}/approve       load it into Supabase
    POST /reviews/{id}/reject        discard it

Usage:
    uvicorn api:app --reload
"""

import glob
import json
import os
import sys

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

# allow imports from repo root (search_db.py, prototype.py, cv_upload.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import cv_upload  # noqa: E402
from search_db import search as run_search  # noqa: E402
from structure_cv import CVError  # noqa: E402

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
    try:
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
    except psycopg.OperationalError as e:
        print(f"search: database connection error: {e!r}")
        raise HTTPException(status_code=503, detail="Database unavailable, try again in a moment.")
    except Exception as e:
        print(f"search: unexpected error for query {q!r}: {e!r}")
        raise HTTPException(status_code=500, detail=f"Search failed: {type(e).__name__}: {str(e)[:200]}")


@app.get("/faculty")
def faculty_list():
    """Every faculty member in the database, for the upload page's professor picker."""
    with cv_upload.connect() as conn:
        return {"faculty": cv_upload.list_faculty(conn)}


@app.get("/faculty/{faculty_id}")
def faculty_endpoint(faculty_id: str):
    try:
        with cv_upload.connect() as conn:
            pkg = cv_upload.current_package(conn, faculty_id)
    except psycopg.OperationalError as e:
        print(f"faculty/{faculty_id}: database connection error: {e!r}")
        pkg = None
    pkg = pkg or _PACKAGES.get(faculty_id)
    if not pkg:
        raise HTTPException(status_code=404, detail="faculty not found")
    return pkg


@app.post("/cv-upload")
def cv_upload_endpoint(
    file: UploadFile = File(...),
    mode: str = Form(...),                 # "new" or "existing"
    faculty_id: str | None = Form(None),   # required when mode is "existing"
):
    """Upload a CV PDF and turn it into a draft for the Profile review page. Nothing is written
    to Supabase until the draft is approved. Takes a minute or so (the model reading the CV)."""
    data = file.file.read(cv_upload.MAX_UPLOAD_BYTES + 1)
    return _run(cv_upload.create_draft, data, file.filename, mode, faculty_id)


@app.get("/reviews")
def reviews_list(status: str | None = "pending"):
    """Drafts, newest first. status: pending (default), approved, rejected, or all."""
    return {"drafts": cv_upload.list_drafts(None if status == "all" else status)}


@app.get("/reviews/{draft_id}")
def review_detail(draft_id: str):
    return _run(cv_upload.get_draft, draft_id)


@app.get("/reviews/{draft_id}/cv")
def review_cv(draft_id: str):
    path = _run(cv_upload.cv_file, draft_id)
    return FileResponse(path, media_type="application/pdf", content_disposition_type="inline")


@app.post("/reviews/{draft_id}/approve")
def review_approve(draft_id: str):
    return _run(cv_upload.approve_draft, draft_id, model)


@app.post("/reviews/{draft_id}/reject")
def review_reject(draft_id: str):
    return _run(cv_upload.reject_draft, draft_id)


def _run(fn, *args):
    """Call a cv_upload function, turning its errors into HTTP errors the page can show."""
    try:
        return fn(*args)
    except CVError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except cv_upload.DraftNotFound:
        raise HTTPException(status_code=404, detail="Draft not found.")
    except cv_upload.UploadConflict as error:
        raise HTTPException(status_code=409, detail=str(error))
    except Exception as error:  # gateway or database trouble: report it instead of a bare 500
        print(f"{fn.__name__} failed: {error!r}")
        raise HTTPException(status_code=502, detail=f"{type(error).__name__}: {str(error)[:300]}")

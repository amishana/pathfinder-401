"""Tiny FastAPI wrapper around search.py so the frontend has an actual endpoint to
call. Doesn't touch the search logic at all, just imports search() and loads the
embedding model once at startup so we're not reloading it on every request.

Usage:
    uvicorn api:app --reload
"""

import glob
import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from config import MODEL
from search import search as run_search

# load all faculty packages into memory once at startup so profile lookups are instant
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

# this is just a student project served from a static html file, so we don't
# know or care what origin it's opened from, allow everything
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
def load_model():
    global model
    _load_packages()
    print(f"loading model {MODEL}...")
    model = SentenceTransformer(MODEL)
    print("model loaded, ready for requests")


@app.get("/search")
def search_endpoint(q: str):
    results = run_search(q, model=model)
    return {"query": q, "results": results}


@app.get("/faculty/{faculty_id}")
def faculty_endpoint(faculty_id: str):
    pkg = _PACKAGES.get(faculty_id)
    if not pkg:
        raise HTTPException(status_code=404, detail="faculty not found")
    return pkg

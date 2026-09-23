"""Tiny FastAPI wrapper around search.py so the frontend has an actual endpoint to
call. Doesn't touch the search logic at all, just imports search() and loads the
embedding model once at startup so we're not reloading it on every request.

Usage:
    uvicorn api:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from config import MODEL
from search import search as run_search

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
    print(f"loading model {MODEL}...")
    model = SentenceTransformer(MODEL)
    print("model loaded, ready for requests")


@app.get("/search")
def search_endpoint(q: str):
    """Same shape search.py already returns, just wrapped in a query field."""
    results = run_search(q, model=model)
    return {"query": q, "results": results}

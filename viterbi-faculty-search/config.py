"""Shared constants and db connection helper."""

import os

import psycopg2
from pgvector.psycopg2 import register_vector

MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
TOP_FACULTY = 5
TOP_EVIDENCE = 3

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "faculty_search")
DB_USER = os.environ.get("DB_USER", "faculty")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "faculty")


def get_conn():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    register_vector(conn)
    return conn

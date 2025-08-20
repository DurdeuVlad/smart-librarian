#!/usr/bin/env python3
"""
Smoke test for the Smart Librarian ChromaDB store.
All embeddings use OpenAI’s text-embedding-3-small model (1536-D).
Prints full metadata for each returned record.
"""

import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

import openai
from chromadb import PersistentClient

def embed_text(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """
    Embed a single string into a 1×1536 vector via OpenAI’s Python v1 API.
    """
    resp = openai.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding

def run_semantic_query(collection, description: str, query: str, n_results: int, where: dict | None):
    print(f"\n=== {description} ===")
    try:
        emb = embed_text(query)
    except Exception as e:
        print(f"Embedding error: {e}")
        return

    try:
        res = collection.query(
            query_embeddings=[emb],
            n_results=n_results,
            where=where
        )
    except Exception as e:
        print(f"Query error: {e}")
        return

    hits = list(zip(
        res.get("metadatas", [[]])[0],
        res.get("distances", [[]])[0]
    ))
    if not hits:
        print("No results returned.")
        return

    for i, (meta, dist) in enumerate(hits, start=1):
        print(f"{i}. distance={dist:.4f}")
        print(json.dumps(meta, indent=2, sort_keys=True))

def run_metadata_query(collection, description: str, n_results: int, where: dict):
    print(f"\n=== {description} ===")
    try:
        res = collection.get(where=where, limit=n_results)
    except Exception as e:
        print(f"Get error: {e}")
        return

    metas = res.get("metadatas", [])
    if not metas:
        print("No results returned.")
        return

    for i, meta in enumerate(metas, start=1):
        print(f"{i}.")
        print(json.dumps(meta, indent=2, sort_keys=True))

def main():
    # ─── Setup ─────────────────────────────────────────────────────────────
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        print("Error: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    client     = PersistentClient(path=str(repo_root / ".chromadb"))
    collection = client.get_or_create_collection(name="openlibrary")

    try:
        count = collection.count()
    except Exception as e:
        print(f"Error retrieving record count: {e}")
        sys.exit(1)

    print(f"Total records in collection: {count}")
    if count == 0:
        print("Collection is empty. Please run ingest_openlibrary.py first.")
        sys.exit(1)

    # 1) Pure semantic
    run_semantic_query(
        collection,
        "Semantic query: 'harry potter'",
        "harry potter",
        n_results=5,
        where=None
    )

    # 2) Hybrid: semantic + trivial metadata filter on title != ""
    run_semantic_query(
        collection,
        "Hybrid query: 'harry potter' with title != ''",
        "harry potter",
        n_results=5,
        where={"title": {"$ne": ""}}
    )

    # 3) Metadata-only: title != ""
    run_metadata_query(
        collection,
        "Metadata-only filter: title != ''",
        n_results=5,
        where={"title": {"$ne": ""}}
    )

if __name__ == "__main__":
    main()

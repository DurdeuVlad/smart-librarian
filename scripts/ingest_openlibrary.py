import os
import gzip
import json
import requests
import sys
from pathlib import Path
from typing import Iterator, List, Dict

# 1) Load environment
from dotenv import load_dotenv
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"
print(f"[DEBUG] Loading .env from {dotenv_path}")
load_dotenv(dotenv_path)

import openai
import chromadb
from chromadb import PersistentClient
import tiktoken

# 2) Configuration
DUMP_URL = "https://archive.org/download/ol_dump_2024-12-19/ol_dump_works_2024-12-19.txt.gz"
DUMP_ARCHIVE = "ol_dump_works_2024-12-19.txt.gz"
DUMP_FILE = "ol_dump_works_2024-12-19.txt"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))
MAX_RECORDS = int(os.getenv("MAX_RECORDS", "10000"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "openlibrary")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
MAX_DESC_TOKENS = int(os.getenv("MAX_DESC_TOKENS", "200"))

COST_PER_1K = {
    "text-embedding-ada-002": 0.0004,
    "text-embedding-3-small": 0.0001
}

# 3) Verify API key
api_key = os.getenv("OPENAI_API_KEY")
print(f"[DEBUG] OPENAI_API_KEY present: {bool(api_key)}")
if not api_key:
    print("Error: OPENAI_API_KEY not set.")
    sys.exit(1)
openai.api_key = api_key

# 4) Initialize ChromaDB
client = PersistentClient(path=str(project_root / ".chromadb"))
collection = client.get_or_create_collection(name=COLLECTION_NAME)
print(f"[DEBUG] Connected to ChromaDB: {COLLECTION_NAME}")
print(f"[DEBUG] Config BATCH_SIZE={BATCH_SIZE}, MAX_RECORDS={MAX_RECORDS}, EMBED_MODEL={EMBED_MODEL}, MAX_DESC_TOKENS={MAX_DESC_TOKENS}")

# 5) Tokenizer
enc = tiktoken.encoding_for_model(EMBED_MODEL)

def download_dump(url: str, target: str):
    print(f"[DEBUG] Downloading {url} to {target}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(target, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)


def decompress_gz(src: str, dest: str):
    print(f"[DEBUG] Decompressing {src} to {dest}")
    with gzip.open(src, "rb") as f_in, open(dest, "wb") as f_out:
        f_out.write(f_in.read())


import csv

def load_top_records(path: str) -> Iterator[Dict]:
    """
    Load up to MAX_RECORDS valid records from the OpenLibrary dump.
    Each line is split into 5 whitespace-separated fields; the 5th is pure JSON.
    """
    print(f"[DEBUG] Loading up to {MAX_RECORDS} valid records from {path}")
    count = 0

    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, start=1):
            if count >= MAX_RECORDS:
                break
            parts = line.strip().split(None, 4)
            if len(parts) < 5:
                print(f"[DEBUG] Line {idx} has only {len(parts)} fields, skipping")
                continue

            try:
                obj = json.loads(parts[4])
            except json.JSONDecodeError:
                print(f"[DEBUG] JSON decode error on line {idx}")
                continue

            # normalize description
            desc_raw = obj.get("description", "")
            if isinstance(desc_raw, dict):
                desc = desc_raw.get("value", "")
            elif isinstance(desc_raw, str):
                desc = desc_raw
            else:
                desc = ""

            yield {
                "id": obj.get("key"),
                "title": obj.get("title", ""),
                "authors": [a.get("name") for a in obj.get("authors", []) if a.get("name")],
                "subjects": obj.get("subjects", []),
                "first_publish_year": obj.get("first_publish_year"),
                "languages": [l.get("key") for l in obj.get("languages", []) if l.get("key")],
                "description": desc,
                "edition_count": obj.get("edition_count", 0)
            }
            count += 1


def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts automatically via OpenAI Python v1 API (no prompt).
    """
    total_tokens = sum(count_tokens(t) for t in texts)
    cost = total_tokens / 1000 * COST_PER_1K.get(EMBED_MODEL, 0)
    print(f"[DEBUG] Embedding {len(texts)} items (~{total_tokens} tokens). Estimated cost: ${cost:.4f}")

    response = openai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def truncate(text: str, max_tokens: int) -> str:
    toks = enc.encode(text)
    return enc.decode(toks[:max_tokens])



def ingest_to_chroma():
    """
    Ensure the OpenLibrary dump is present (download & decompress if needed),
    then read up to MAX_RECORDS via load_top_records(), batch them, embed them,
    and upsert into the ChromaDB collection with only primitive metadata types.
    """
    # 1) Download / decompress if missing
    if not os.path.exists(DUMP_ARCHIVE):
        download_dump(DUMP_URL, DUMP_ARCHIVE)
    if not os.path.exists(DUMP_FILE):
        decompress_gz(DUMP_ARCHIVE, DUMP_FILE)

    print("[DEBUG] Starting ingestion to ChromaDB...")
    batch_ids: List[str] = []
    batch_texts: List[str] = []
    batch_meta: List[Dict[str, object]] = []
    total = 0

    # 2) Iterate and batch
    for rec in load_top_records(DUMP_FILE):
        total += 1
        # Build the text and count tokens
        desc = truncate(rec['description'], MAX_DESC_TOKENS)
        text = f"{rec['title']} . {desc}"
        tok_count = count_tokens(text)
        print(f"[DEBUG] Record {rec['id']} (#{total}) uses {tok_count} tokens")

        batch_ids.append(rec['id'])
        batch_texts.append(text)

        # Flatten list metadata into strings (None if empty)
        authors   = ', '.join(rec['authors'])   if rec['authors']   else None
        subjects  = ', '.join(rec['subjects'])  if rec['subjects']  else None
        languages = ', '.join(rec['languages']) if rec['languages'] else None

        batch_meta.append({
            'title': rec['title'],
            'authors': authors,
            'subjects': subjects,
            'first_publish_year': rec['first_publish_year'],
            'languages': languages
        })

        # 3) When batch full, embed & upsert
        if len(batch_ids) >= BATCH_SIZE:
            embeddings = embed_batch(batch_texts)
            collection.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                metadatas=batch_meta
            )
            print(f"[DEBUG] Upserted {len(batch_ids)} items; total so far: {total}")

            batch_ids.clear()
            batch_texts.clear()
            batch_meta.clear()

    # 4) Flush any remainder
    if batch_ids:
        embeddings = embed_batch(batch_texts)
        collection.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            metadatas=batch_meta
        )
        print(f"[DEBUG] Upserted final {len(batch_ids)}; total ingested: {total}")

    """
    Ensure the OpenLibrary dump is present (download & decompress if needed),
    then read up to MAX_RECORDS via load_top_records(), batch them, embed them,
    and upsert into the ChromaDB collection with only primitive metadata types.
    """
    # 1) Ensure dump archive and text exist
    if not os.path.exists(DUMP_ARCHIVE):
        download_dump(DUMP_URL, DUMP_ARCHIVE)
    if not os.path.exists(DUMP_FILE):
        decompress_gz(DUMP_ARCHIVE, DUMP_FILE)

    print("[DEBUG] Starting ingestion to ChromaDB...")
    batch_ids: List[str] = []
    batch_texts: List[str] = []
    batch_meta: List[Dict[str, object]] = []
    total = 0

    # 2) Process records
    for rec in load_top_records(DUMP_FILE):
        total += 1
        # Truncate description and build the text to embed
        desc = truncate(rec['description'], MAX_DESC_TOKENS)
        text = f"{rec['title']} . {desc}"
        tokens = count_tokens(text)
        print(f"[DEBUG] Record {rec['id']} (#{total}) uses {tokens} tokens")

        # Collect batch data
        batch_ids.append(rec['id'])
        batch_texts.append(text)

        # Flatten list metadata into comma‐separated strings (or None if empty)
        authors   = ', '.join(rec['authors'])   if rec['authors']   else None
        subjects  = ', '.join(rec['subjects'])  if rec['subjects']  else None
        languages = ', '.join(rec['languages']) if rec['languages'] else None

        batch_meta.append({
            'title': rec['title'],
            'authors': authors,
            'subjects': subjects,
            'first_publish_year': rec['first_publish_year'],
            'languages': languages
        })

        # 3) When batch is full, embed & upsert
        if len(batch_ids) >= BATCH_SIZE:
            embeddings = embed_batch(batch_texts)
            collection.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                metadatas=batch_meta
            )
            print(f"[DEBUG] Upserted {len(batch_ids)} items; total so far: {total}")

            # reset batches
            batch_ids.clear()
            batch_texts.clear()
            batch_meta.clear()

    # 4) Final flush
    if batch_ids:
        embeddings = embed_batch(batch_texts)
        collection.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            metadatas=batch_meta
        )
        print(f"[DEBUG] Upserted final {len(batch_ids)}; total ingested: {total}")


def main():
    print("[DEBUG] Running main()")
    print(f"[DEBUG] Archive exists? {os.path.exists(DUMP_ARCHIVE)}; Dump exists? {os.path.exists(DUMP_FILE)}")
    ingest_to_chroma()


if __name__ == '__main__':
    main()

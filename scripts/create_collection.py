#!/usr/bin/env python3
import os
import shutil
import stat
from pathlib import Path
from dotenv import load_dotenv
import openai
from chromadb import PersistentClient
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction

# 1) Locate repo root and load .env
repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("ERROR: OPENAI_API_KEY not set in .env")
    exit(1)

# 2) Path to the on-disk store
store_dir = repo_root / ".chromadb"

# 3) Remove old store, clearing any read-only flags if necessary
if store_dir.exists():
    print(f"[DEBUG] Removing existing store at {store_dir}")
    def _on_rm_error(func, path, exc_info):
        # clear read-only bit and retry
        os.chmod(path, stat.S_IWRITE)
        func(path)
    shutil.rmtree(store_dir, onerror=_on_rm_error)
else:
    print(f"[DEBUG] No existing store to remove at {store_dir}")

# 4) Recreate with 3-small embedding function
client = PersistentClient(path=str(store_dir))
ef = OpenAIEmbeddingFunction(
    api_key=openai.api_key,
    model_name="text-embedding-3-small"
)
collection = client.get_or_create_collection(
    name="openlibrary",
    embedding_function=ef
)

print("✅ Recreated ‘openlibrary’ with text-embedding-3-small (384-D) embeddings.")

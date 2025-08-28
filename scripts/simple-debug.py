#!/usr/bin/env python3
from chromadb import PersistentClient
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
client = PersistentClient(path=str(repo_root / ".chromadb"))

try:
    collection = client.get_collection("openlibrary")
    print(f"Count: {collection.count()}")

    # Încearcă peek
    result = collection.peek(5)
    print(f"Peek result: {result}")

    # Încearcă get simplu
    result2 = collection.get(limit=3)
    print(f"Get result: {result2}")

except Exception as e:
    print(f"Error: {e}")
    # Încearcă să recreezi colecția cu câteva cărți test
    print("Creating test data...")

    collection = client.get_or_create_collection("openlibrary")
    collection.add(
        ids=["test1", "test2"],
        documents=["Harry Potter magic wizard", "Pride and Prejudice romance"],
        metadatas=[
            {"title": "Harry Potter", "authors": "J.K. Rowling"},
            {"title": "Pride and Prejudice", "authors": "Jane Austen"}
        ]
    )
    print(f"Added test data. New count: {collection.count()}")
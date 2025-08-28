#!/usr/bin/env python3
import os
from pathlib import Path
from dotenv import load_dotenv
from chromadb import PersistentClient

# Setup
repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")


def explore_books(limit=15):
    """Explorează cărțile din ChromaDB"""
    try:
        client = PersistentClient(path=str(repo_root / ".chromadb"))
        collections = client.list_collections()

        if not collections:
            print("❌ Nu există colecții!")
            return

        collection = collections[0]
        print(f"📚 Colecția: {collection.name}")
        print(f"📊 Total cărți: {collection.count()}")
        print("=" * 60)

        # Obține primele cărți
        result = collection.get(
            limit=limit,
            include=['metadatas', 'documents', 'ids']
        )

        for i, (book_id, metadata) in enumerate(zip(result['ids'], result['metadatas']), 1):
            print(f"{i:2d}. 📖 {metadata.get('title', 'N/A')}")
            print(f"     ✍️  {metadata.get('authors', 'N/A')}")
            print(f"     📅 {metadata.get('first_publish_year', 'N/A')}")
            if metadata.get('subjects'):
                subjects = str(metadata['subjects'])[:80] + "..." if len(str(metadata['subjects'])) > 80 else str(
                    metadata['subjects'])
                print(f"     🏷️  {subjects}")
            print()

    except Exception as e:
        print(f"❌ Eroare: {e}")
        import traceback
        traceback.print_exc()


def search_semantic(query_text, limit=5):
    """Căutare semantică cu embeddings"""
    try:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")

        if not openai.api_key:
            print("❌ OPENAI_API_KEY nu este setat")
            return

        client = PersistentClient(path=str(repo_root / ".chromadb"))
        collections = client.list_collections()
        collection = collections[0]

        # Generează embedding pentru query
        resp = openai.embeddings.create(
            model="text-embedding-3-small",
            input=query_text
        )

        # Căutare semantică
        results = collection.query(
            query_embeddings=[resp.data[0].embedding],
            n_results=limit
        )

        print(f"🔍 Rezultate pentru '{query_text}':")
        for i, (distance, metadata) in enumerate(zip(
                results['distances'][0],
                results['metadatas'][0]
        ), 1):
            print(f"{i}. {metadata.get('title')} - {metadata.get('authors')} (dist: {distance:.3f})")

    except Exception as e:
        print(f"❌ Eroare căutare: {e}")


if __name__ == "__main__":
    print("🔍 EXPLORARE BAZĂ DE DATE CĂRȚI\n")

    # Explorare principală
    explore_books(20)

    print("\n" + "=" * 60)
    print("🔍 TESTARE CĂUTARE SEMANTICĂ")
    search_semantic("fantasy adventure magic", 5)
    search_semantic("love romance", 3)
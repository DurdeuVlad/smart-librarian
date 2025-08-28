#!/usr/bin/env python3
import os
from pathlib import Path
from dotenv import load_dotenv
from chromadb import PersistentClient

# Setup
repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")

# Test books data
books = [
    {
        "id": "maitreyi",
        "title": "Maitreyi",
        "authors": "Mircea Eliade",
        "year": 1933,
        "subjects": "romance, philosophy, India",
        "description": "Povestea de dragoste dintre Allan și Maitreyi în India colonială, explorând chestiuni filozofice și culturale."
    },
    {
        "id": "ion",
        "title": "Ion",
        "authors": "Liviu Rebreanu",
        "year": 1920,
        "subjects": "realism, social drama, rural life",
        "description": "Drama țăranului Ion care se căsătorește pentru pământ, reprezentând conflictele sociale din satul românesc."
    },
    {
        "id": "enigma_otiliei",
        "title": "Enigma Otiliei",
        "authors": "George Călinescu",
        "year": 1938,
        "subjects": "psychological novel, bourgeoisie",
        "description": "Romanul explorează psihologia personajelor din familia burgheză Tulea prin ochii Otiliei."
    },
    {
        "id": "baltagul",
        "title": "Baltagul",
        "authors": "Mihail Sadoveanu",
        "year": 1930,
        "subjects": "epic, folklore, revenge",
        "description": "Vitoria Lipan pleacă în căutarea soțului mort, într-o poveste epică despre credință și răzbunare."
    },
    {
        "id": "harry_potter",
        "title": "Harry Potter și Piatra Filozofală",
        "authors": "J.K. Rowling",
        "year": 1997,
        "subjects": "fantasy, magic, adventure, coming-of-age",
        "description": "Un băiat orfan descoperă că este vrăjitor și intră în lumea magică de la Hogwarts."
    },
    {
        "id": "1984",
        "title": "1984",
        "authors": "George Orwell",
        "year": 1949,
        "subjects": "dystopia, totalitarianism, surveillance",
        "description": "Într-o societate totalitară, Winston Smith se luptă împotriva controlului absolut al Partidului."
    },
    {
        "id": "pride_prejudice",
        "title": "Pride and Prejudice",
        "authors": "Jane Austen",
        "year": 1813,
        "subjects": "romance, social commentary, marriage",
        "description": "Elizabeth Bennet și Mr. Darcy depășesc mândria și prejudecățile într-o poveste de dragoste clasică."
    },
    {
        "id": "hobbit",
        "title": "The Hobbit",
        "authors": "J.R.R. Tolkien",
        "year": 1937,
        "subjects": "fantasy, adventure, quest",
        "description": "Bilbo Baggins pleacă într-o aventură neașteptată cu piticii pentru a recupera comoara dragonului Smaug."
    },
    {
        "id": "crime_punishment",
        "title": "Crime and Punishment",
        "authors": "Fyodor Dostoievski",
        "year": 1866,
        "subjects": "psychology, crime, redemption",
        "description": "Raskolnikov comite o crimă și se confruntă cu consecințele psihologice și morale ale actului său."
    },
    {
        "id": "gatsby",
        "title": "The Great Gatsby",
        "authors": "F. Scott Fitzgerald",
        "year": 1925,
        "subjects": "american dream, tragedy, wealth",
        "description": "Jay Gatsby urmărește visul american și dragostea pentru Daisy în America anilor '20."
    }
]


def add_books():
    try:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")

        if not openai.api_key:
            print("❌ OPENAI_API_KEY not set")
            return

        client = PersistentClient(path=str(repo_root / ".chromadb"))

        # Delete și recreează colecția
        try:
            client.delete_collection("openlibrary")
            print("🗑️ Deleted existing collection")
        except:
            pass

        collection = client.create_collection("openlibrary")

        # Prepare data
        ids = [book["id"] for book in books]
        documents = [f"{book['title']} by {book['authors']}. {book['description']}" for book in books]
        metadatas = [
            {
                "title": book["title"],
                "authors": book["authors"],
                "first_publish_year": book["year"],
                "subjects": book["subjects"],
                "description": book["description"]
            }
            for book in books
        ]

        # Generate embeddings via OpenAI
        print("🔄 Generating embeddings...")
        embeddings = []
        for doc in documents:
            resp = openai.embeddings.create(
                model="text-embedding-3-small",
                input=doc
            )
            embeddings.append(resp.data[0].embedding)

        # Add to collection with embeddings
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        print(f"✅ Added {len(books)} books successfully!")
        print(f"📊 Collection count: {collection.count()}")

        # Test query
        result = collection.get(limit=3)
        print(f"📖 First 3 books:")
        for i, meta in enumerate(result['metadatas']):
            print(f"  {i + 1}. {meta['title']} - {meta['authors']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    add_books()
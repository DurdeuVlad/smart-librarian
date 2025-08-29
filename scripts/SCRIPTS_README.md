# Scripts (ChromaDB + OpenAI Embeddings)

Aceste scripturi Python gestionează crearea colecției ChromaDB și ingestia de date (rezumate de cărți).

## 📂 Scripturi incluse
- **create_collection.py** – creează colecția `books` și configurează funcția de embeddings `text-embedding-3-small`
- **ingest_openlibrary.py** – adaugă manual cărți (ex: Ion, Maitreyi, Enigma Otiliei, Baltagul etc.) în colecția `books`
- **explore_database.py** – inspectează colecția și afișează conținutul
- **smoke_test.py** – test rapid pentru conexiune și queryuri
- **simple-debug.py** – debugging simplu

## 🔑 Variabile de mediu
Scripturile citesc `.env` din rădăcina proiectului:
```ini
OPENAI_API_KEY=sk-...
CHROMA_URL=http://localhost:8000
CHROMA_COLLECTION=books
```

## 🚀 Instalare dependențe
```bash
cd scripts
pip install -r requirements.txt
```

## 🧪 Exemple de rulare
- Creează colecția:
```bash
python create_collection.py
```
- Ingestă cărți (din codul hardcodat):
```bash
python ingest_openlibrary.py
```
- Explorează colecția:
```bash
python explore_database.py
```

## 🔗 Integrare
- **Backend** → interoghează colecția Chroma creată de scripturi
- **Frontend** → afișează rezultatele returnate de backend
- **Worker** → generează răspunsuri folosind contextul din colecție


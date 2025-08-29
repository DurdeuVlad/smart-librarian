# Smart Librarian

Smart Librarian este un sistem **RAG (Retrieval Augmented Generation)** care recomandă cărți pe baza preferințelor utilizatorului. Proiectul este alcătuit din mai multe module: **backend (Express)**, **worker (Redis + OpenAI)**, **frontend (React/Vite)** și **scripts Python (ChromaDB)**.

## 📂 Module

- [Backend](./backend/BACKEND_README.md)
- [Worker](./backend/WORKER_README.md)
- [Frontend](./frontend/FRONTEND_README.md)
- [Scripts](./scripts/SCRIPTS_README.md)

## 🚀 Quickstart

### 1. Clonează proiectul
```bash
git clone <repo-url>
cd Smart-Librarian
```

### 2. Creează fișier `.env`
```ini
OPENAI_API_KEY=sk-...
REDIS_URL=redis://redis:6379
CHROMA_URL=http://chromadb:8000
CHROMA_COLLECTION=books
OPENAI_BUDGET_LIMIT_USD=5
```

### 3. Rulează cu Docker Compose
```bash
docker compose up -d --build
```

### 4. Accesează serviciile
- UI: [http://localhost:5173](http://localhost:5173)
- API Backend: [http://localhost:3001](http://localhost:3001)
- ChromaDB: [http://localhost:8000](http://localhost:8000)

## 📂 Module
- [Backend](./backend/README.md)
- [Worker](./backend/WORKER.md)
- [Frontend](./frontend/README.md)
- [Scripts](./scripts/README.md)
- [Arhitectură](./docs/ARCHITECTURE.md)
- [Troubleshooting](./docs/TROUBLESHOOTING.md)

## 🧪 Test rapid
- **Health check** backend:
```bash
curl http://localhost:3001/api/health
```
- **Smoke test** scripts:
```bash
cd scripts
python smoke_test.py
```

## 📜 Licență
Proiect educațional (laborator/assignment).


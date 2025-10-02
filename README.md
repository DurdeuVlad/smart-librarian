# Smart Librarian

Smart Librarian is a **RAG (Retrieval Augmented Generation)** system that recommends books based on user preferences. The project consists of several modules: **backend (Express)**, **worker (Redis + OpenAI)**, **frontend (React/Vite)**, and **Python scripts (ChromaDB)**.

## 📂 Modules

- [Backend](./backend/README.md)
- [Worker](./backend/WORKER_README.md)
- [Frontend](./frontend/README.md)
- [Scripts](./scripts/README.md)

## 🚀 Quickstart

### 1. Clone the project

```bash
git clone <repo-url>
cd Smart-Librarian
```

### 2. Create `.env` file

```ini
OPENAI_API_KEY=sk-...
REDIS_URL=redis://redis:6379
CHROMA_URL=http://chromadb:8000
CHROMA_COLLECTION=books
OPENAI_BUDGET_LIMIT_USD=5
```

### 3. Run with Docker Compose

```bash
docker compose up -d --build
```

### 4. Access the services

- UI: [http://localhost:5173](http://localhost:5173)
- API Backend: [http://localhost:3001](http://localhost:3001)
- ChromaDB: [http://localhost:8000](http://localhost:8000)

## 📂 Documentation

- [Backend](./backend/README.md)
- [Worker](./backend/WORKER.md)
- [Frontend](./frontend/README.md)
- [Scripts](./scripts/README.md)
- [Architecture](./docs/ARCHITECTURE.md)
- [Troubleshooting](./docs/TROUBLESHOOTING.md)

## 🧪 Quick test

- **Backend health check**:

```bash
curl http://localhost:3001/api/health
```

- **Smoke test scripts**:

```bash
cd scripts
python smoke_test.py
```

## 📜 License

Educational project (lab/assignment).

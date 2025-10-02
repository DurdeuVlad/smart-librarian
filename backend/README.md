# Backend (Express API)

The backend is responsible for the HTTP interface between the frontend and internal services (ChromaDB, Redis, OpenAI).

## ⚙️ Features

- Exposes REST endpoints for health checks, semantic search, and chat
- Connects to **ChromaDB** for semantic retrieval
- Publishes jobs to **Redis** (consumed by worker)
- Limits costs through a configurable budget

## 🚀 Local run (without Docker)

```bash
cd backend
npm install
npm run dev
```

The backend reads `.env` from the project root.

## 🔑 Environment variables

- `OPENAI_API_KEY` – your OpenAI API key
- `REDIS_URL` – Redis connection (e.g., redis://localhost:6379)
- `CHROMA_URL` – Chroma URL (e.g., http://localhost:8000 or http://chromadb:8000 in Docker)
- `OPENAI_BUDGET_LIMIT_USD` – maximum budget in USD (default: 5)

## 🌐 Endpoints

- `GET /api/health` → backend status + budget
- `GET /api/search?q=<term>` → search books in Chroma collection
- `POST /api/chat`

```json
{
  "message": "I want a book about friendship and magic",
  "context": "Smart Librarian - recommendations"
}
```

## 🔗 Integration

- **Frontend** → sends requests to `/api/chat`
- **ChromaDB** → `books` collection with embeddings
- **Redis** → job queue for worker

# Backend (Express API)

Backend-ul este responsabil de interfața HTTP dintre frontend și serviciile interne (ChromaDB, Redis, OpenAI).

## ⚙️ Funcționalități
- expune endpointuri REST pentru health, căutare semantică și chat
- conectează la **ChromaDB** pentru retrieval semantic
- publică joburi în **Redis** (consumate de worker)
- limitează costurile printr-un buget configurabil

## 🚀 Rulare locală (fără Docker)
```bash
cd backend
npm install
npm run dev
```

Backend-ul citește `.env` din rădăcina proiectului.

## 🔑 Variabile de mediu
- `OPENAI_API_KEY` – cheia ta OpenAI
- `REDIS_URL` – conexiunea la Redis (ex: redis://localhost:6379)
- `CHROMA_URL` – URL-ul Chroma (ex: http://localhost:8000 sau http://chromadb:8000 în Docker)
- `OPENAI_BUDGET_LIMIT_USD` – buget maxim în USD (implicit 5)

## 🌐 Endpoint-uri
- `GET /api/health` → status backend + buget
- `GET /api/search?q=<termen>` → caută cărți în colecția Chroma
- `POST /api/chat`
```json
{
  "message": "Vreau o carte despre prietenie și magie",
  "context": "Smart Librarian - recomandări"
}
```

## 🔗 Integrare
- **Frontend** → trimite requesturi către `/api/chat`
- **ChromaDB** → colecția `books` cu embeddings
- **Redis** → coadă de joburi pentru worker


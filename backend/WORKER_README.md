# Worker (Redis + OpenAI)

Worker-ul procesează joburile plasate de backend în coada Redis și apelează OpenAI pentru a genera răspunsuri conversaționale și rezumate extinse.

## ⚙️ Funcționalități
- consumă joburi din Redis (tipuri: `chat`, `cover`)
- apelează **OpenAI Chat Completions** pentru recomandări și răspunsuri
- apelează **OpenAI Image/Completions** pentru generarea coperților/imagini (cover)
- salvează rezultatele în Redis (`result:<jobId>`) pentru a fi preluate de backend

## 🚀 Rulare locală
```bash
cd backend
node worker.js
```

## 🔑 Variabile de mediu
- `OPENAI_API_KEY` – cheia ta OpenAI
- `REDIS_URL` – conexiune către Redis (ex: redis://localhost:6379)
- `OPENAI_BUDGET_LIMIT_USD` – buget maxim în USD (implicit 5)

## 🔗 Integrare
- **Backend** → publică joburi (`chat`, `cover`) în Redis
- **Worker** → procesează și scrie rezultatul în Redis
- **Frontend** → primește răspunsul procesat prin backend

## 📦 Exemple de joburi
### Chat job
```json
{
  "type": "chat",
  "data": {
    "message": "Vreau o carte despre prietenie și magie",
    "context": "Smart Librarian - recomandări"
  },
  "jobId": "uuid-random"
}
```

### Cover job
```json
{
  "type": "cover",
  "data": {
    "title": "Ion",
    "author": "Liviu Rebreanu",
    "summary": "Drama țăranului Ion..."
  },
  "jobId": "uuid-random"
}
```


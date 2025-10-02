# Worker (Redis + OpenAI)

The worker processes jobs placed by the backend in the Redis queue and calls OpenAI to generate conversational responses and extended summaries.

## ⚙️ Features

- Consumes jobs from Redis (types: `chat`, `cover`)
- Calls **OpenAI Chat Completions** for recommendations and responses
- Calls **OpenAI Image/Completions** for cover/image generation (cover)
- Saves results in Redis (`result:<jobId>`) to be retrieved by the backend

## 🚀 Local run

```bash
cd backend
node worker.js
```

## 🔑 Environment variables

- `OPENAI_API_KEY` – your OpenAI API key
- `REDIS_URL` – Redis connection (e.g., redis://localhost:6379)
- `OPENAI_BUDGET_LIMIT_USD` – maximum budget in USD (default: 5)

## 🔗 Integration

- **Backend** → publishes jobs (`chat`, `cover`) to Redis
- **Worker** → processes and writes result to Redis
- **Frontend** → receives the processed response through the backend

## 📦 Job examples

### Chat job

```json
{
  "type": "chat",
  "data": {
    "message": "I want a book about friendship and magic",
    "context": "Smart Librarian - recommendations"
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
    "summary": "The drama of the peasant Ion..."
  },
  "jobId": "uuid-random"
}
```

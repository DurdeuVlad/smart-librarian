# Frontend (React / Vite)

Web interface for chat and displaying recommendations.

## ⚙️ Features

- Chat UI for questions like "I want a book about friendship and magic"
- List of candidate results from Chroma (via backend)
- Displays OpenAI "budget" (spent / remaining)
- (Dev) support for mock data during development

## 🚀 Local run

```bash
cd frontend
npm install
npm run dev
```

The application starts by default on **http://localhost:5173**.

## 🔧 API Configuration

The backend runs on **http://localhost:3001**. If you want to override:

```bash
# example .env.local
VITE_API_URL=http://localhost:3001
```

In the code, the backend URL is read from `import.meta.env.VITE_API_URL` (fallback to `http://localhost:3001`).

## 🏗️ Build & Preview

```bash
npm run build
npm run preview
```

## 🐳 Docker

```bash
# from project root
docker compose up -d --build frontend
```

## 📁 Relevant structure

- `src/App.jsx` – main chat logic, results display, and budget
- `src/BookCover.jsx` – component for image/cover
- `src/main.jsx` – application bootstrap

## ❗ Troubleshooting

- UI doesn't connect to backend → check `VITE_API_URL` and that backend is running on 3001
- CORS in dev → backend allows origins `http://localhost:5173` (see `server.js`)

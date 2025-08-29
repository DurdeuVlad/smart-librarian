# Frontend (React / Vite)

Interfața web pentru chat și afișarea recomandărilor.

## ⚙️ Funcționalități
- UI de chat pentru întrebări precum „Vreau o carte despre prietenie și magie”
- Listă de rezultate candidate din Chroma (prin backend)
- Afișează „budget” OpenAI (spent / remaining)
- (Dev) suport pentru mock data în timpul dezvoltării

## 🚀 Rulare locală
```bash
cd frontend
npm install
npm run dev
```
Aplicația pornește implicit pe **http://localhost:5173**.

## 🔧 Configurare API
Backendul rulează pe **http://localhost:3001**. Dacă vrei să suprascrii:
```bash
# exemplu .env.local
VITE_API_URL=http://localhost:3001
```
În cod, URL-ul backendului este citit din `import.meta.env.VITE_API_URL` (fallback la `http://localhost:3001`).

## 🏗️ Build & Preview
```bash
npm run build
npm run preview
```

## 🐳 Docker
```bash
# din rădăcina proiectului
docker compose up -d --build frontend
```

## 📁 Structură relevantă
- `src/App.jsx` – logica principală de chat, afișare rezultate și buget
- `src/BookCover.jsx` – componentă pentru imagine/copertă
- `src/main.jsx` – bootstrap aplicație

## ❗ Troubleshooting
- UI nu se conectează la backend → verifică `VITE_API_URL` și că backendul rulează pe 3001
- CORS în dev → backendul permite origini `http://localhost:5173` (vezi `server.js`)


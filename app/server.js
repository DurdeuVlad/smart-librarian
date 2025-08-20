// server.js (ESM)
import express from "express";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import OpenAI from "openai";
import { ChromaClient } from "chromadb";

/* ────────────────────────── Env loading (force .env) ──────────────────────────
   We OVERRIDE existing process env so the repo’s .env wins over machine vars. */
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config({
  path: path.resolve(__dirname, "..", ".env"), // repo/.env
  override: true,
});

/* ────────────────────────────── Config values ─────────────────────────────── */
const PORT = Number(process.env.API_PORT || 3001);

// OpenAI
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  console.error("❌ OPENAI_API_KEY not set (check repo/.env).");
  process.exit(1);
}
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
const EMBED_MODEL = process.env.EMBED_MODEL || "text-embedding-3-small";

// Chroma
const CHROMA_HOST = process.env.CHROMA_HOST || "127.0.0.1";
const CHROMA_PORT = Number(process.env.CHROMA_PORT || 8000);
const CHROMA_SSL = String(process.env.CHROMA_SSL || "false").toLowerCase() === "true";
const CHROMA_COLLECTION = process.env.CHROMA_COLLECTION || "openlibrary";

/* ─────────────────────────────── Chroma client ──────────────────────────────
   IMPORTANT: We DO NOT use default embedding functions at all.
   Also, we ONLY call getCollection (not getOrCreateCollection) to avoid
   @chroma-core/default-embed being pulled in by the SDK automatically. */
const chroma = new ChromaClient({
  host: CHROMA_HOST,
  port: CHROMA_PORT,
  ssl: CHROMA_SSL,
});

async function getCollectionStrict(name) {
  try {
    const c = await chroma.getCollection({ name });
    return c;
  } catch (err) {
    console.error(
      `❌ Chroma collection "${name}" not found. Make sure it was created by your Python ingestion scripts.`
    );
    throw err;
  }
}

/* ────────────────────────────── Web server init ───────────────────────────── */
const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

/* Health */
app.get("/api/health", async (req, res) => {
  try {
    // quick ping: list collections to confirm the server is reachable
    await chroma.listCollections();
    res.json({ ok: true });
  } catch {
    res.status(503).json({ ok: false, error: "Chroma unreachable" });
  }
});

/* Query endpoint
   Body: { text: string, k?: number, where?: object }
*/
app.post("/api/query", async (req, res) => {
  const { text, k = 5, where = undefined } = req.body || {};
  if (!text || typeof text !== "string") {
    return res.status(400).json({ error: "Missing 'text' string in body." });
  }

  try {
    // 1) Embed the query text via OpenAI (server-side)
    const embResp = await openai.embeddings.create({
      model: EMBED_MODEL,
      input: text,
    });
    const vector = embResp.data[0].embedding;

    // 2) Query Chroma with explicit queryEmbeddings (NO default embed package)
    const collection = await getCollectionStrict(CHROMA_COLLECTION);
    const result = await collection.query({
      queryEmbeddings: [vector],
      nResults: Number(k) || 5,
      where, // optional metadata filter
    });

    // 3) Shape a compact response
    const out = [];
    const metadatas = result.metadatas?.[0] || [];
    const ids = result.ids?.[0] || [];
    const distances = result.distances?.[0] || [];
    for (let i = 0; i < metadatas.length; i++) {
      out.push({
        id: ids[i],
        distance: distances[i],
        metadata: metadatas[i],
      });
    }

    res.json({ query: text, k: Number(k) || 5, results: out });
  } catch (err) {
    console.error("Query error:", err);
    res.status(500).json({
      error:
        err?.message ||
        "Query failed. Ensure Chroma is running and the collection exists.",
    });
  }
});

/* ─────────────────────────────── Start server ─────────────────────────────── */
app.listen(PORT, () => {
  const masked =
    OPENAI_API_KEY.length > 10
      ? `${OPENAI_API_KEY.slice(0, 6)}…${OPENAI_API_KEY.slice(-4)}`
      : "********";
  console.log(`🔑 Using OpenAI key: ${masked}`);
  console.log(
    `↔  Chroma at ${CHROMA_SSL ? "https" : "http"}://${CHROMA_HOST}:${CHROMA_PORT} (collection="${CHROMA_COLLECTION}")`
  );
  console.log(`🚀 API listening on http://localhost:${PORT}`);
});

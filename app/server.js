// Enhanced server.js with auth, media, and professional features
import express from "express";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import OpenAI from "openai";
import { ChromaClient } from "chromadb";
import jwt from "jsonwebtoken";
import bcrypt from "bcryptjs";
import pg from "pg";
import { createClient } from "redis";
import rateLimit from "express-rate-limit";
import helmet from "helmet";
import compression from "compression";
import { v4 as uuidv4 } from "uuid";
import fs from "fs/promises";

const { Pool } = pg;

/* ────────────────────────── Setup ──────────────────────────── */
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config({ path: path.resolve(__dirname, "..", ".env"), override: true });

const PORT = Number(process.env.PORT || 3001);
const JWT_SECRET = process.env.JWT_SECRET || "fallback-secret";
const BUDGET_LIMIT = Number(process.env.BUDGET_LIMIT || 5.0);

/* ────────────────────────── Database Connections ──────────────────────────── */
// PostgreSQL
const db = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
});

// Redis
const redis = createClient({
    url: process.env.REDIS_URL
});
redis.connect().catch(console.error);

// ChromaDB (existing)
const chroma = new ChromaClient({
    host: process.env.CHROMA_HOST || "127.0.0.1",
    port: Number(process.env.CHROMA_PORT || 8000),
    ssl: String(process.env.CHROMA_SSL || "false").toLowerCase() === "true",
});

/* ────────────────────────── OpenAI ──────────────────────────── */
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const EMBED_MODEL = process.env.EMBED_MODEL || "text-embedding-3-small";
const CHAT_MODEL = process.env.CHAT_MODEL || "gpt-4o-mini";
const CHROMA_COLLECTION = process.env.CHROMA_COLLECTION || "openlibrary";

/* ────────────────────────── Book Summary Function ──────────────────────────── */

/* ────────────────────────── Cost Tracking ──────────────────────────── */
const costTracker = {
    spent: 0,
    embedCost: 0.00002,
    chatCost: { input: 0.00015, output: 0.0006 },

    async logCall(userId, type, tokens, cost, metadata = {}) {
        this.spent += cost;

        if (this.spent >= BUDGET_LIMIT) {
            throw new Error(`Budget exceeded: $${this.spent.toFixed(4)}`);
        }

        // Log to database
        try {
            await db.query(
                'INSERT INTO api_usage (user_id, operation_type, tokens_used, cost_usd, metadata) VALUES ($1, $2, $3, $4, $5)',
                [userId, type, tokens, cost, JSON.stringify(metadata)]
            );
        } catch (error) {
            console.error('Failed to log API usage:', error);
        }

        console.log(`💰 ${type}: ${tokens} tokens, $${cost.toFixed(4)} | Total: $${this.spent.toFixed(4)}`);
    },

    async getStats(userId = null) {
        try {
            const query = userId
                ? 'SELECT SUM(cost_usd) as total FROM api_usage WHERE user_id = $1'
                : 'SELECT SUM(cost_usd) as total FROM api_usage';
            const params = userId ? [userId] : [];

            const result = await db.query(query, params);
            const spent = Number(result.rows[0]?.total || 0);

            return {
                spent,
                remaining: BUDGET_LIMIT - spent,
                limit: BUDGET_LIMIT
            };
        } catch (error) {
            console.error('Failed to get cost stats:', error);
            return { spent: this.spent, remaining: BUDGET_LIMIT - this.spent, limit: BUDGET_LIMIT };
        }
    }
};

/* ────────────────────────── Middleware ──────────────────────────── */
const app = express();

app.use(helmet());
app.use(compression());
app.use(cors({ origin: process.env.ENABLE_CORS === 'true' }));
app.use(express.json({ limit: "1mb" }));

// Rate limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100,
    message: { error: 'Too many requests' }
});
app.use('/api/', limiter);

// Auth middleware
const authenticateToken = async (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
        return res.status(401).json({ error: 'Access token required' });
    }

    try {
        const decoded = jwt.verify(token, JWT_SECRET);
        req.user = decoded;
        next();
    } catch (error) {
        return res.status(403).json({ error: 'Invalid token' });
    }
};

/* ────────────────────────── Utility Functions ──────────────────────────── */
async function getSummaryByTitle(title) {
    try {
        const collection = await chroma.getCollection({ name: CHROMA_COLLECTION });

        // Search for exact title match in metadata
        const result = await collection.get({
            where: { "title": { "$eq": title } },
            limit: 1
        });

        if (result.metadatas && result.metadatas.length > 0) {
            const metadata = result.metadatas[0];
            return `${metadata.title}\n\nAutor: ${metadata.authors || 'Unknown'}\nAn publicare: ${metadata.first_publish_year || 'Unknown'}\nSubiecte: ${metadata.subjects || 'N/A'}\n\nAceastă carte face parte din colecția noastră și este disponibilă pentru consultare.`;
        }

        // Fallback: semantic search for partial matches
        const embResp = await openai.embeddings.create({
            model: EMBED_MODEL,
            input: title,
        });

        const semanticResult = await collection.query({
            queryEmbeddings: [embResp.data[0].embedding],
            nResults: 1,
            where: { "title": { "$ne": "" } }
        });

        if (semanticResult.metadatas && semanticResult.metadatas[0].length > 0) {
            const metadata = semanticResult.metadatas[0][0];
            return `Cea mai apropiată potrivire pentru "${title}":\n\n${metadata.title}\n\nAutor: ${metadata.authors || 'Unknown'}\nAn publicare: ${metadata.first_publish_year || 'Unknown'}\nSubiecte: ${metadata.subjects || 'N/A'}`;
        }

        return `Ne pare rău, nu am găsit informații pentru "${title}" în baza noastră de date.`;

    } catch (error) {
        console.error('Error getting book summary:', error);
        return `A apărut o eroare în căutarea informațiilor pentru "${title}".`;
    }
}

async function getCachedResponse(key) {
    try {
        const cached = await redis.get(`cache:${key}`);
        return cached ? JSON.parse(cached) : null;
    } catch (error) {
        console.error('Redis get error:', error);
        return null;
    }
}

async function setCachedResponse(key, data, ttl = 3600) {
    try {
        await redis.setEx(`cache:${key}`, ttl, JSON.stringify(data));
    } catch (error) {
        console.error('Redis set error:', error);
    }
}

/* ────────────────────────── Routes ──────────────────────────── */

// Health check
app.get("/api/health", async (req, res) => {
    try {
        await Promise.all([
            db.query('SELECT 1'),
            redis.ping(),
            chroma.listCollections()
        ]);

        res.json({
            ok: true,
            services: { postgres: true, redis: true, chromadb: true }
        });
    } catch (error) {
        res.status(503).json({ ok: false, error: error.message });
    }
});

// Auth routes
app.post("/api/auth/register", async (req, res) => {
    const { email, password, firstName, lastName } = req.body;

    if (!email || !password) {
        return res.status(400).json({ error: 'Email and password required' });
    }

    try {
        const hashedPassword = await bcrypt.hash(password, 10);
        const result = await db.query(
            'INSERT INTO users (email, password_hash, first_name, last_name) VALUES ($1, $2, $3, $4) RETURNING id, email, first_name, last_name',
            [email, hashedPassword, firstName, lastName]
        );

        const user = result.rows[0];
        const token = jwt.sign({ id: user.id, email: user.email }, JWT_SECRET, { expiresIn: '7d' });

        res.json({ token, user });
    } catch (error) {
        if (error.code === '23505') {
            return res.status(400).json({ error: 'Email already exists' });
        }
        res.status(500).json({ error: 'Registration failed' });
    }
});

app.post("/api/auth/login", async (req, res) => {
    const { email, password } = req.body;

    try {
        const result = await db.query(
            'SELECT id, email, password_hash, first_name, last_name FROM users WHERE email = $1',
            [email]
        );

        if (result.rows.length === 0) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }

        const user = result.rows[0];
        const validPassword = await bcrypt.compare(password, user.password_hash);

        if (!validPassword) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }

        const token = jwt.sign({ id: user.id, email: user.email }, JWT_SECRET, { expiresIn: '7d' });

        res.json({
            token,
            user: {
                id: user.id,
                email: user.email,
                firstName: user.first_name,
                lastName: user.last_name
            }
        });
    } catch (error) {
        res.status(500).json({ error: 'Login failed' });
    }
});

// Books & Chat routes
app.post("/api/query", authenticateToken, async (req, res) => {
    const { text, k = 5, where = undefined } = req.body || {};
    if (!text) return res.status(400).json({ error: "Missing 'text'" });

    const cacheKey = `query:${text}:${k}:${JSON.stringify(where)}`;
    const cached = await getCachedResponse(cacheKey);
    if (cached) return res.json(cached);

    try {
        const embResp = await openai.embeddings.create({
            model: EMBED_MODEL,
            input: text,
        });

        const tokens = text.split(' ').length * 1.3;
        await costTracker.logCall(req.user.id, 'embedding', tokens, tokens * costTracker.embedCost);

        const collection = await chroma.getCollection({ name: CHROMA_COLLECTION });
        const result = await collection.query({
            queryEmbeddings: [embResp.data[0].embedding],
            nResults: Number(k),
            where,
        });

        const response = {
            query: text,
            results: result.metadatas?.[0]?.map((meta, i) => ({
                id: result.ids[0][i],
                distance: result.distances[0][i],
                metadata: meta
            })) || []
        };

        await setCachedResponse(cacheKey, response);
        res.json(response);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post("/api/chat", authenticateToken, async (req, res) => {
    const { message, sessionId } = req.body;
    if (!message) return res.status(400).json({ error: "Missing message" });

    try {
        // Get recent conversation context
        let conversation = [];
        if (sessionId) {
            const result = await db.query(
                'SELECT role, content FROM chat_messages WHERE session_id = $1 ORDER BY created_at DESC LIMIT 10',
                [sessionId]
            );
            conversation = result.rows.reverse();
        }

        // Get book recommendations
        const searchResp = await fetch(`http://localhost:${PORT}/api/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': req.headers.authorization
            },
            body: JSON.stringify({ text: message, k: 3 })
        });
        const searchData = await searchResp.json();

        const bookContext = searchData.results
            ?.map(r => `${r.metadata.title} by ${r.metadata.authors || 'Unknown'}`)
            .join('\n') || '';

        const messages = [
            {
                role: "system",
                content: `Ești un bibliotecar inteligent care recomandă cărți. Răspunde în română.
        
Context cărți disponibile:
${bookContext}

Când recomanzi o carte specifică, folosește funcția get_summary_by_title pentru rezumat detaliat.`
            },
            ...conversation,
            { role: "user", content: message }
        ];

        const completion = await openai.chat.completions.create({
            model: CHAT_MODEL,
            messages,
            functions: [{
                name: "get_summary_by_title",
                description: "Obține rezumatul detaliat al unei cărți după titlu",
                parameters: {
                    type: "object",
                    properties: {
                        title: { type: "string", description: "Titlul exact al cărții" }
                    },
                    required: ["title"]
                }
            }],
            function_call: "auto",
            max_tokens: 300,
            temperature: 0.7
        });

        let response = completion.choices[0].message;

        // Handle function calling
        if (response.function_call?.name === "get_summary_by_title") {
            const { title } = JSON.parse(response.function_call.arguments);
            const summary = getSummaryByTitle(title);

            const followUp = await openai.chat.completions.create({
                model: CHAT_MODEL,
                messages: [
                    ...messages,
                    response,
                    { role: "function", name: "get_summary_by_title", content: summary }
                ],
                max_tokens: 200
            });

            response = followUp.choices[0].message;
        }

        // Log costs
        const inputTokens = JSON.stringify(messages).length / 4;
        const outputTokens = response.content?.length / 4 || 100;
        const cost = (inputTokens * costTracker.chatCost.input) + (outputTokens * costTracker.chatCost.output);
        await costTracker.logCall(req.user.id, 'chat', inputTokens + outputTokens, cost);

        // Save conversation
        const currentSessionId = sessionId || uuidv4();
        if (!sessionId) {
            await db.query(
                'INSERT INTO chat_sessions (id, user_id, title) VALUES ($1, $2, $3)',
                [currentSessionId, req.user.id, message.substring(0, 50)]
            );
        }

        await db.query(
            'INSERT INTO chat_messages (session_id, role, content) VALUES ($1, $2, $3), ($1, $4, $5)',
            [currentSessionId, 'user', message, 'assistant', response.content]
        );

        res.json({
            message: response.content,
            sessionId: currentSessionId,
            searchResults: searchData.results,
            budget: await costTracker.getStats(req.user.id)
        });

    } catch (error) {
        console.error("Chat error:", error);
        res.status(500).json({ error: error.message });
    }
});

// User stats
app.get("/api/user/stats", authenticateToken, async (req, res) => {
    try {
        const [sessionsResult, favoritesResult] = await Promise.all([
            db.query('SELECT COUNT(*) as count FROM chat_sessions WHERE user_id = $1', [req.user.id]),
            db.query('SELECT COUNT(*) as count FROM user_favorites WHERE user_id = $1', [req.user.id])
        ]);

        res.json({
            chatSessions: Number(sessionsResult.rows[0].count),
            favorites: Number(favoritesResult.rows[0].count),
            budget: await costTracker.getStats(req.user.id)
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/* ────────────────────────── Start Server ──────────────────────────── */
app.listen(PORT, () => {
    console.log(`🚀 Smart Librarian API on http://localhost:${PORT}`);
    console.log(`💰 Budget limit: $${BUDGET_LIMIT}`);
});
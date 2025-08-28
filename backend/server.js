import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { ChromaClient } from 'chromadb';
import OpenAI from 'openai';

dotenv.config({ path: '../.env' });

const app = express();
const PORT = 3001;

// Initialize clients
const chroma = new ChromaClient({
    path: "http://localhost:8000",
    cors: {
        allowOrigins: ["http://localhost:3001", "http://localhost:5173"]
    }
});
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Cost tracking
let totalSpent = 0;
const BUDGET_LIMIT = 5.0;
const costs = {
    embedding: 0.00002,
    chat_input: 0.00015,
    chat_output: 0.0006
};

app.use(cors());
app.use(express.json());

async function getSummaryByTitle(title) {
    try {
        const collection = await chroma.getCollection({ name: "openlibrary" });
        const results = await collection.get({
            where: { "title": { "$eq": title } },
            limit: 1
        });

        if (results.metadatas && results.metadatas.length > 0) {
            const meta = results.metadatas[0];
            return {
                summary: meta.description || meta.subjects,
                bookData: {
                    title: meta.title,
                    authors: meta.authors,
                    subjects: meta.subjects
                }
            };
        }
        return null;
    } catch (error) {
        return null;
    }
}

// Routes
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', budget: { spent: totalSpent, remaining: BUDGET_LIMIT - totalSpent } });
});

app.post('/api/query', async (req, res) => {
    const { text, k = 3 } = req.body;

    try {
        const embedding = await openai.embeddings.create({
            model: "text-embedding-3-small",
            input: text
        });
        totalSpent += costs.embedding * (text.length / 4);

        const collection = await chroma.getCollection({ name: "openlibrary" });
        const results = await collection.query({
            queryEmbeddings: [embedding.data[0].embedding],
            nResults: k
        });
        const formattedResults = results.metadatas[0].map((meta, i) => ({
            id: results.ids[0][i],
            distance: results.distances[0][i],
            metadata: meta
        }));

        res.json({ results: formattedResults });

    } catch (error) {
        console.error('Query error:', error);
        res.status(500).json({ error: error.message });
    }
});

app.post('/api/chat', async (req, res) => {
    const { message, history = [] } = req.body;

    try {
        const embeddingResponse = await openai.embeddings.create({
            model: "text-embedding-3-small",
            input: message
        });
        totalSpent += costs.embedding * (message.length / 4);

        const collection = await chroma.getCollection({ name: "openlibrary" });
        const results = await collection.query({
            queryEmbeddings: [embeddingResponse.data[0].embedding],
            nResults: 3
        });

        const relevantDocs = results.metadatas[0].map((meta) => {
            const { title, authors, subjects, summary } = meta;
            return `Titlu: ${title}\nAutor: ${authors}\nSubiecte: ${subjects}\nRezumat: ${summary}`;
        });

        const context = relevantDocs.join('\n\n---\n\n');
        const systemPrompt = `Ești un bibliotecar inteligent. Răspunde DOAR pe baza cărților de mai jos. Pentru cărți din această listă, folosește funcția get_summary_by_title pentru detalii complete.

Informații din biblioteca ta:
${context}

Dacă utilizatorul întreabă despre cărți care NU sunt în lista de mai sus, spune că nu le ai în bibliotecă si ca nu poti sa il ajuti.
Răspunde întotdeauna cu text, apoi folosește tool-ul pentru detalii extra.`;

        const completion = await openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
                { role: "system", content: systemPrompt },
                ...history,
                { role: "user", content: message }
            ],
            tools: [
                {
                    type: "function",
                    function: {
                        name: "get_summary_by_title",
                        description: "Obține rezumat foarte detaliat pentru o carte specifică",
                        parameters: {
                            type: "object",
                            properties: {
                                title: {
                                    type: "string",
                                    description: "Titlul exact al cărții"
                                }
                            },
                            required: ["title"]
                        }
                    }
                }
            ],
            tool_choice: "auto",
            max_tokens: 500
        });

        const rawResponse = completion.choices[0].message.content || "";
        let conversationalMessage = rawResponse || "";
        let searchResults = [];

        if (rawResponse && rawResponse.includes('!!!')) {
            const parts = rawResponse.split('!!!');
            conversationalMessage = parts[0].trim();
            const jsonPart = parts[1].trim();
            try {
                searchResults = JSON.parse(jsonPart);
            } catch (e) {
                console.error("Eroare la parsarea JSON-ului de la AI:", e);
            }
        }

        if (completion.choices[0].message.tool_calls) {
            for (const toolCall of completion.choices[0].message.tool_calls) {
                if (toolCall.function.name === "get_summary_by_title") {
                    const { title } = JSON.parse(toolCall.function.arguments);
                    const result = await getSummaryByTitle(title);

                    if (result) {
                        conversationalMessage += `\n\n📖 **Rezumat extins pentru "${title}":**\n${result.summary}`;
                        searchResults.push(result.bookData);
                    }
                }
            }
        }

        const inputTokens = completion.usage.prompt_tokens;
        const outputTokens = completion.usage.completion_tokens;
        const chatCost = (inputTokens * costs.chat_input) + (outputTokens * costs.chat_output);
        totalSpent += chatCost;

        const budget = {
            spent: totalSpent,
            remaining: BUDGET_LIMIT - totalSpent,
            limit: BUDGET_LIMIT
        };

        res.json({
            message: conversationalMessage,
            searchResults: searchResults,
            budget: budget
        });

    } catch (error) {
        console.error('Chat error:', error);
        res.status(500).json({ error: error.message });
    }
});

const coverCache = new Map();

app.post('/api/cover', async (req, res) => {
    const { title, author, summary } = req.body;
    const cacheKey = `${title.toLowerCase()}|${author.toLowerCase()}`;
    if (coverCache.has(cacheKey)) {
        console.log(`Cache HIT pentru: ${cacheKey}`);
        return res.json(coverCache.get(cacheKey));
    }
    console.log(`Cache MISS pentru: ${cacheKey}. Se generează o copertă nouă.`);
    try {
        const combinedPromptCompletion = await openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
                {
                    role: "system",
                    content: `Ești un generator de descrieri pentru coperți de carte. Creează o descriere artistică, detaliată și unică pentru o copertă, potrivită pentru un model de generare de imagini. Concentrează-te pe stilul artistic, elementele vizuale cheie, culori și compoziție. Fără text pe copertă. Imaginea trebuie să fie captivantă și reprezentativă pentru carte. Răspunde doar cu descrierea, fără alte explicații.`
                },
                {
                    role: "user",
                    content: `Creează un concept de copertă pentru cartea "${title}" de ${author}. Rezumat: ${summary || "Nu există rezumat."}`
                }
            ],
            max_tokens: 250
        });
        const imagePrompt = combinedPromptCompletion.choices[0].message.content;
        if (!imagePrompt) {
            throw new Error("Nu am putut genera un prompt valid pentru imagine.");
        }
        const imageResponse = await openai.images.generate({
            model: "dall-e-3",
            prompt: imagePrompt,
            size: "1024x1024",
            quality: "hd",
        });
        const imageUrl = imageResponse.data[0].url;
        if (!imageUrl) {
            throw new Error("URL-ul imaginii generate este invalid.");
        }
        const responseData = {
            title,
            author,
            url: imageUrl
        };
        coverCache.set(cacheKey, responseData);
        res.json(responseData);
    } catch (error) {
        console.error('Eroare la generarea copertei:', error);
        res.status(500).json({ error: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`🚀 Backend running on http://localhost:${PORT}`);
    console.log(`💰 Budget: $${BUDGET_LIMIT}`);
});
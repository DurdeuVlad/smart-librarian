import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { ChromaClient } from 'chromadb';
import OpenAI from 'openai';

dotenv.config({ path: '../.env' });

const app = express();
const PORT = 3001;

// Initialize clients
const chroma = new ChromaClient({ path: "http://localhost:8000" });

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

// Book summaries for tool calling
const bookSummaries = {
    "Maitreyi": `Povestea de dragoste dintre Allan și Maitreyi în India colonială. Mircea Eliade explorează chestiuni filozofice și culturale prin această relație intensă. Cartea abordează diferențele dintre culturile estică și vestică, precum și căutarea spirituală a personajului principal.`,
    "Ion": `Drama țăranului Ion care se căsătorește pentru pământ, nu pentru dragoste. Liviu Rebreanu prezintă conflictele sociale din satul românesc, obsesia pentru proprietate și consecințele tragice ale ambițiilor materialiste.`,
    "Harry Potter și Piatra Filozofală": `Un băiat orfan descoperă că este vrăjitor și intră la școala Hogwarts. J.K. Rowling creează o lume magică plină de aventuri, prietenii și confruntarea cu răul. Prima carte din seria care a captivat generații întregi.`,
    "1984": `Într-o societate totalitară, Winston Smith se luptă împotriva controlului absolut al Partidului. George Orwell prezintă o viziune distopică despre supraveghere, manipulare și pierderea libertății individuale.`
};

// Tool function (necesită o logică suplimentară pentru a o face utilă în noul context)
function getSummaryByTitle(title) {
    const summary = bookSummaries[title];
    return summary || `Nu am găsit un rezumat pentru "${title}" în baza mea de date.`;
}

// Routes
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', budget: { spent: totalSpent, remaining: BUDGET_LIMIT - totalSpent } });
});

app.post('/api/query', async (req, res) => {
    const { text, k = 3 } = req.body;

    try {
        // Generate embedding
        const embedding = await openai.embeddings.create({
            model: "text-embedding-3-small",
            input: text
        });
        totalSpent += costs.embedding * (text.length / 4);

        // Query ChromaDB
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
    const { message } = req.body;

    try {
        // PASUL 1: Generează embedding pentru mesajul utilizatorului
        const embeddingResponse = await openai.embeddings.create({
            model: "text-embedding-3-small",
            input: message
        });
        totalSpent += costs.embedding * (message.length / 4);

        // PASUL 2: Caută în ChromaDB cele mai relevante cărți
        const collection = await chroma.getCollection({ name: "openlibrary" });
        const results = await collection.query({
            queryEmbeddings: [embeddingResponse.data[0].embedding],
            nResults: 3
        });

        const relevantDocs = results.metadatas[0].map((meta) => {
            const { title, authors, subjects, summary } = meta;
            return `Titlu: ${title}\nAutor: ${authors}\nSubiecte: ${subjects}\nRezumat: ${summary}`;
        });

        // PASUL 3: Construiește un prompt îmbogățit cu context din baza de date
        const context = relevantDocs.join('\n\n---\n\n');
        const systemPrompt = `Ești un bibliotecar inteligent. Răspunde la întrebarea utilizatorului DOAR pe baza informațiilor de mai jos. Dacă informațiile furnizate nu conțin răspunsul, spune că nu ai detalii despre subiect.

Informații din biblioteca ta:
${context}

Dacă în răspunsul tău te referi la una dintre cărțile din lista de mai sus, adaugă la final, după trei semne de exclamare (!!!), un array JSON valid cu acele cărți. Fiecare obiect din array trebuie să aibă cheile "title", "authors", și "subjects". Exemplu: "Salut! Îți recomand Ion. !!![{"title": "Ion", "authors": "Liviu Rebreanu"}]"`;

        const completion = await openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: message }
            ],
            max_tokens: 500
        });

        const rawResponse = completion.choices[0].message.content;
        let conversationalMessage = rawResponse;
        let searchResults = [];

        if (rawResponse.includes('!!!')) {
            const parts = rawResponse.split('!!!');
            conversationalMessage = parts[0].trim();
            const jsonPart = parts[1].trim();
            try {
                searchResults = JSON.parse(jsonPart);
            } catch (e) {
                console.error("Eroare la parsarea JSON-ului de la AI:", e);
            }
        }

        // Calculează costul real
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
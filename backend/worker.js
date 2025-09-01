// backend/worker.js
import Redis from 'ioredis';
import OpenAI from 'openai';
import dotenv from 'dotenv';

dotenv.config({ path: '.env', override: true });

const redis = new Redis(process.env.REDIS_URL);
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function processJob(jobData) {
    const { type, data, jobId } = jobData;

    try {
        let result;

        switch (type) {
            case 'chat':
                result = await processChatRequest(data);
                break;
            case 'cover':
                result = await processCoverRequest(data);
                break;
            default:
                throw new Error(`Unknown job type: ${type}`);
        }

        await redis.set(`result:${jobId}`, JSON.stringify(result), 'EX', 300);

    } catch (error) {
        await redis.set(`result:${jobId}`, JSON.stringify({ error: error.message }), 'EX', 300);
    }
}

async function processChatRequest(data) {
    const { message, context } = data;

    const completion = await openai.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [
            { role: "system", content: context },
            { role: "user", content: message }
        ],
        max_tokens: 500
    });

    return {
        message: completion.choices[0].message.content,
        usage: completion.usage
    };
}

async function processCoverRequest(data) {
    const { title, author, summary } = data;

    const promptCompletion = await openai.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [
            {
                role: "system",
                content: "Create artistic book cover description for image generation."
            },
            {
                role: "user",
                content: `Create cover concept for "${title}" by ${author}. Summary: ${summary}`
            }
        ],
        max_tokens: 250
    });

    const imageResponse = await openai.images.generate({
        model: "dall-e-3",
        prompt: promptCompletion.choices[0].message.content,
        size: "1024x1024"
    });

    return {
        title,
        author,
        url: imageResponse.data[0].url
    };
}

// Worker loop
async function startWorker() {
    console.log('Worker started, waiting for jobs...');

    while (true) {
        const job = await redis.blpop('job_queue', 30);
        if (job) {
            const jobData = JSON.parse(job[1]);
            console.log(`Processing job ${jobData.jobId}`);
            await processJob(jobData);
        }
    }
}

startWorker().catch(console.error);
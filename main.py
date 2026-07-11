import math
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="Message in a Bottle")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = (
    "Paraphrase the following message in your own words. "
    "Keep it roughly the same length. Do not explain, just paraphrase."
)
CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"


class SendRequest(BaseModel):
    message: str
    chain_length: int = Field(default=6, ge=1)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def paraphrase(text: str) -> str:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content.strip()


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def run_chain(message: str, chain_length: int) -> dict:
    intermediates: list[str] = []
    current = message
    for _ in range(chain_length):
        current = paraphrase(current)
        intermediates.append(current)

    all_texts = [message] + intermediates
    embeddings = embed_texts(all_texts)
    original_emb = embeddings[0]

    chain = []
    for i, text in enumerate(intermediates):
        sim = cosine_similarity(original_emb, embeddings[i + 1])
        chain.append({"hop": i + 1, "text": text, "similarity_to_original": sim})

    return {
        "original": message,
        "chain": chain,
        "final_similarity": chain[-1]["similarity_to_original"] if chain else 1.0,
    }


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/send")
def send(body: SendRequest):
    return run_chain(body.message, body.chain_length)


@app.post("/webhook")
def webhook(body: SendRequest):
    return run_chain(body.message, body.chain_length)

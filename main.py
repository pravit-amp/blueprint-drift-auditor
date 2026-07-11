import math
import os
from difflib import SequenceMatcher
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI(title="Message in a Bottle")
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)
_embeddings_fallback_warned = False

EMBED_MODEL = "text-embedding-3-small"

PERSONA_URL_ENVS = ("OPTIMIST_URL", "CYNIC_URL", "POET_URL")


class SendRequest(BaseModel):
    message: str


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def call_persona_webhook(base_url: str, text: str) -> str:
    url = base_url.rstrip("/") + "/webhook"
    with httpx.Client(timeout=60.0) as http:
        response = http.post(url, json={"text": text})
        response.raise_for_status()
        data = response.json()
    return data["text"]


def run_chain(message: str) -> dict:
    urls = []
    for name in PERSONA_URL_ENVS:
        value = os.environ.get(name, "").strip()
        if not value:
            raise HTTPException(status_code=500, detail=f"{name} is not set")
        urls.append(value)

    intermediates: list[str] = []
    current = message
    for base_url in urls:
        current = call_persona_webhook(base_url, current)
        intermediates.append(current)

    try:
        all_texts = [message] + intermediates
        embeddings = embed_texts(all_texts)
        original_emb = embeddings[0]
        similarities = [
            cosine_similarity(original_emb, embeddings[i + 1])
            for i in range(len(intermediates))
        ]
    except Exception:
        global _embeddings_fallback_warned
        if not _embeddings_fallback_warned:
            print("embeddings endpoint unavailable, falling back to text similarity.")
            _embeddings_fallback_warned = True
        similarities = [
            SequenceMatcher(None, message, text).ratio() for text in intermediates
        ]

    chain = []
    for i, text in enumerate(intermediates):
        chain.append(
            {"hop": i + 1, "text": text, "similarity_to_original": similarities[i]}
        )

    return {
        "original": message,
        "chain": chain,
        "final_similarity": chain[-1]["similarity_to_original"] if chain else 1.0,
    }


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/webhook")
def webhook(body: SendRequest):
    return run_chain(body.message)


@app.post("/send")
def send(body: SendRequest):
    return run_chain(body.message)

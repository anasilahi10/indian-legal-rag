"""The FastAPI server: the one thing the frontend talks to."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from backend import config
from backend.rag import answer
from backend.retriever import retrieve


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the models once at startup, not on the first question
    retrieve("warm up")
    yield


app = FastAPI(title="Legal Documents RAG", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    model: str | None = None  # lets the frontend pick a model later


@app.get("/health")
def health():
    return {"status": "ok", "retrieval_mode": config.RETRIEVAL_MODE}


@app.post("/ask")
def ask(req: AskRequest):
    result = answer(req.question, model=req.model or config.LLM_MODEL)
    return {
        "answer": result["answer"],
        "model": result["model"],
        "sources": [
            {
                "document": h["document"],
                "page": h["page"],
                "pages": h["pages"],
                "score": round(h["score"], 4),
                "text": h["text"],
            }
            for h in result["sources"]
        ],
    }

"""All the settings in one place, so we can change them without editing the logic."""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
CHUNKS_FILE = DATA_DIR / "processed" / "chunks.json"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "legal_docs"

# Small, fast English embedding model (about 130 MB). Later we can switch
# to a multilingual one for Hindi.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

TOP_K = 5  # how many chunks the search returns

MIN_SCORE = 0.55               # below this similarity, say "not found"


# small and fast (about 90 MB)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 20       # how many hybrid results the reranker re-sorts
RETRIEVAL_MODE = "rerank"    # "dense", "hybrid" or "rerank"; we switch after testing

CONTEXT_AFTER = 2  # also send this many following chunks with each retrieved chunk
CHUNK_OVERLAP = 200  # must match CHUNK_OVERLAP in backend/ingest.py
ANSWER_TOP_K = int(os.getenv("ANSWER_TOP_K", "8"))
# Which service answers: "openrouter" or "groq" (set LLM_PROVIDER in the shell)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")
PROVIDERS = {
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
}
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "openrouter/free" if LLM_PROVIDER == "openrouter" else "llama-3.3-70b-versatile",
)

"""Step 3: find the chunks most relevant to a question (dense, hybrid, or rerank)."""
import json
import re
from functools import lru_cache

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from backend import config

# bge models work best when the QUESTION (not the documents) starts with this.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CANDIDATES = 30   # how many results each method contributes before merging
RRF_K = 60        # standard constant for reciprocal rank fusion


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=1)
def _load():
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.get_collection(config.COLLECTION_NAME)
    chunks = json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    by_id = {c["id"]: c for c in chunks}
    return model, collection, chunks, bm25, by_id


@lru_cache(maxsize=1)
def _load_reranker():
    return CrossEncoder(config.RERANKER_MODEL)


def _dense(question: str, n: int) -> list[tuple[str, float]]:
    model, collection, *_ = _load()
    emb = model.encode([QUERY_PREFIX + question],
                       normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=emb, n_results=n)
    return [
        (res["ids"][0][i], 1 - res["distances"][0][i])
        for i in range(len(res["ids"][0]))
    ]


def _keyword(question: str, n: int) -> list[tuple[str, float]]:
    _, _, chunks, bm25, _ = _load()
    scores = bm25.get_scores(tokenize(question))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [(chunks[i]["id"], float(scores[i])) for i in top if scores[i] > 0]


def retrieve(question: str, top_k: int = config.TOP_K, mode: str | None = None) -> list[dict]:
    mode = mode or config.RETRIEVAL_MODE
    by_id = _load()[4]
    dense = _dense(question, CANDIDATES)
    dense_sim = dict(dense)
    dense_best = dense[0][1] if dense else 0.0

    if mode == "dense":
        ranking = list(dense)
    else:
        fused: dict[str, float] = {}
        for results in (dense, _keyword(question, CANDIDATES)):
            for rank, (cid, _) in enumerate(results, 1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        ranking = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        if mode == "rerank":
            candidates = ranking[: config.RERANK_CANDIDATES]
            pairs = [(question, by_id[cid]["text"]) for cid, _ in candidates]
            scores = _load_reranker().predict(pairs)
            ranking = sorted(
                zip([cid for cid, _ in candidates], [float(s)
                    for s in scores]),
                key=lambda x: x[1],
                reverse=True,
            )

    hits = []
    for cid, score in ranking[:top_k]:
        c = by_id[cid]
        hits.append({
            "id": cid,
            "document": c["document"],
            "page": c["page"],
            "text": c["text"],
            # ranking score (depends on mode)
            "score": score,
            # meaning-match, None if not in dense list
            "similarity": dense_sim.get(cid),
            "dense_best": dense_best,          # best meaning-match for the question
        })
    return hits


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or (
        "Within how many days can a person file a first appeal under the RTI Act?"
    )
    print(f"Question: {q}\n")
    for rank, h in enumerate(retrieve(q), 1):
        print(
            f"{rank}. {h['document']} page {h['page']} (score {h['score']:.3f})")
        print("   " + h["text"][:200] + "...\n")

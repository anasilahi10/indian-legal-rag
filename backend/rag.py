"""Step 4: retrieve chunks, ask the LLM through OpenRouter, return answer + sources."""
import os

from dotenv import load_dotenv
from openai import OpenAI

from backend import config
from backend.retriever import _load, retrieve

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

NOT_FOUND = "I could not find this in the provided documents."

SYSTEM_PROMPT = (
    "You answer questions about Indian laws using ONLY the numbered context "
    "passages. Cite the passages you used like [1] or [2]. "
    f'If the passages do not contain the answer, reply exactly: "{NOT_FOUND}" '
    "Do not use outside knowledge. If the passages contain a list, include every "
    "item of that list. Keep the answer clear, in your own words, and do not "
    "quote long passages."
)


def build_passages(hits: list[dict]) -> list[dict]:
    """Extend each hit with the chunks that follow it, and merge overlapping ones."""
    chunks = _load()[2]
    position = {c["id"]: i for i, c in enumerate(chunks)}

    ranges = []
    for rank, h in enumerate(hits, 1):
        start = position[h["id"]]
        end = start
        while (
            end < start + config.CONTEXT_AFTER
            and end + 1 < len(chunks)
            and chunks[end + 1]["document"] == h["document"]
        ):
            end += 1
        ranges.append({"start": start, "end": end, "rank": rank, "hit": h})

    ranges.sort(key=lambda r: r["start"])
    merged = []
    for r in ranges:
        last = merged[-1] if merged else None
        if (
            last
            and chunks[r["start"]]["document"] == chunks[last["end"]]["document"]
            and r["start"] <= last["end"] + 1
        ):
            last["end"] = max(last["end"], r["end"])
            if r["rank"] < last["rank"]:
                last["rank"], last["hit"] = r["rank"], r["hit"]
        else:
            merged.append(dict(r))
    merged.sort(key=lambda r: r["rank"])

    passages = []
    for r in merged:
        parts = []
        for j in range(r["start"], r["end"] + 1):
            text = chunks[j]["text"]
            if j > r["start"] and chunks[j]["page"] == chunks[j - 1]["page"]:
                # drop text repeated from the previous chunk
                text = text[config.CHUNK_OVERLAP:]
            parts.append(text)
        pages = sorted({chunks[j]["page"]
                       for j in range(r["start"], r["end"] + 1)})
        passages.append({
            "document": r["hit"]["document"],
            "page": r["hit"]["page"],
            "pages": pages,
            "score": r["hit"]["score"],
            "text": " ".join(parts),
        })
    return passages


def build_context(passages: list[dict]) -> str:
    blocks = []
    for i, p in enumerate(passages, 1):
        if len(p["pages"]) == 1:
            pages = f"page {p['pages'][0]}"
        else:
            pages = f"pages {p['pages'][0]}-{p['pages'][-1]}"
        blocks.append(f"[{i}] ({p['document']}, {pages})\n{p['text']}")
    return "\n\n".join(blocks)


def answer(question: str, model: str = config.LLM_MODEL) -> dict:
    hits = retrieve(question, top_k=config.ANSWER_TOP_K)
    if not hits or hits[0]["dense_best"] < config.MIN_SCORE:
        return {"answer": NOT_FOUND, "sources": [], "model": model}

    passages = build_passages(hits)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{build_context(passages)}\n\nQuestion: {question}",
            },
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    if not text or (
        NOT_FOUND.lower() in text.lower() and len(text) < len(NOT_FOUND) + 40
    ):
        return {"answer": NOT_FOUND, "sources": [], "model": model}
    return {"answer": text, "sources": passages, "model": model}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or (
        "Within how many days can a person file a first appeal under the RTI Act?"
    )
    result = answer(q)
    print(f"Question: {q}\n")
    print(f"Answer: {result['answer']}\n")
    for i, p in enumerate(result["sources"], 1):
        print(
            f"  [{i}] {p['document']} pages {p['pages']} (score {p['score']:.3f})")

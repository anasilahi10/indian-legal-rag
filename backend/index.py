"""Step 2 of the pipeline: embed the chunks and store them in ChromaDB."""
import json

import chromadb
from sentence_transformers import SentenceTransformer

from backend import config

BATCH_SIZE = 64


def build_index() -> None:
    chunks = json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        # start fresh each time
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()
        collection.add(
            ids=[c["id"] for c in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {"document": c["document"], "page": c["page"]} for c in batch
            ],
        )
        print(
            f"  indexed {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print(f"\nDone. {collection.count()} chunks stored in {config.CHROMA_DIR}")


if __name__ == "__main__":
    build_index()

"""Step 1 of the pipeline: read the PDFs and split them into chunks."""
import json
import re
from pathlib import Path

import fitz  # PyMuPDF

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/processed/chunks.json")

CHUNK_SIZE = 1000     # characters per chunk
CHUNK_OVERLAP = 200   # characters shared between neighbouring chunks
MIN_CHARS = 50        # throw away tiny leftovers (page numbers, headers)


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")          # soft hyphens
    text = re.sub(r"-\n(?=\w)", "", text)      # join words split at line end
    text = re.sub(r"\s+", " ", text)           # collapse whitespace
    return text.strip()


def split_text(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # try to cut at a sentence end, else at a space, so words stay whole
            cut = text.rfind(". ", start + size // 2, end)
            if cut == -1:
                cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def ingest() -> list[dict]:
    chunks = []
    for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
        doc = fitz.open(pdf_path)
        before = len(chunks)
        for page_index, page in enumerate(doc):
            text = clean_text(page.get_text())
            for piece in split_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
                if len(piece) < MIN_CHARS:
                    continue
                chunks.append({
                    "id": f"{pdf_path.stem}_p{page_index + 1}_c{len(chunks)}",
                    "document": pdf_path.name,
                    "page": page_index + 1,   # same numbering as the PDF viewer
                    "text": piece,
                })
        print(f"{pdf_path.name}: {len(doc)} pages -> {len(chunks) - before} chunks")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nTotal: {len(chunks)} chunks saved to {OUT_FILE}")
    return chunks


if __name__ == "__main__":
    ingest()

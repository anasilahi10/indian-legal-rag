from pathlib import Path
import fitz

RAW = Path("data/raw")


def devanagari_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    hindi = [c for c in letters if "\u0900" <= c <= "\u097F"]
    return len(hindi) / len(letters)


for pdf in sorted(RAW.glob("*.pdf")):
    doc = fitz.open(pdf)
    pages = len(doc)
    text = "".join(page.get_text() for page in doc)
    ratio = devanagari_ratio(text)
    sample = doc[pages // 2].get_text()[:300].replace("\n", " ")
    print("=" * 70)
    print(pdf.name)
    print(
        f"pages: {pages} | characters: {len(text)} | Hindi share: {ratio:.0%}")
    print("sample:", sample)

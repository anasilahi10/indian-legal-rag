"""Score the search: does it find the right document and page? Compares methods."""
import csv
import sys

from backend.retriever import retrieve

QUESTIONS_FILE = sys.argv[1] if len(
    sys.argv) > 1 else "data/eval/questions.csv"
MODES = ["dense", "hybrid", "rerank"]


def parse_pages(value: str) -> set[int]:
    return {int(p) for p in value.split("|") if p.strip().isdigit()}


def main() -> None:
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["document"] != "NONE"]

    stats = {m: {"top5": 0, "top1": 0} for m in MODES}

    for row in rows:
        doc = row["document"]
        pages = parse_pages(row["page"])
        line = f"Q{row['id']:>2} expected p{row['page']:<8}"
        for mode in MODES:
            hits = retrieve(row["question"], mode=mode)
            ok5 = any(h["document"] == doc and h["page"]
                      in pages for h in hits)
            ok1 = hits[0]["document"] == doc and hits[0]["page"] in pages
            stats[mode]["top5"] += ok5
            stats[mode]["top1"] += ok1
            found = ",".join(f"{h['document'][:3]}p{h['page']}" for h in hits)
            line += f" | {mode}: {'PASS' if ok5 else 'FAIL'} [{found}]"
        print(line)

    n = len(rows)
    print("\n" + "=" * 60)
    print(f"{'method':<10}{'right page in top 5':<24}{'right page first'}")
    for mode in MODES:
        s = stats[mode]
        print(f"{mode:<10}{str(s['top5']) + '/' + str(n):<24}{s['top1']}/{n}")


if __name__ == "__main__":
    main()

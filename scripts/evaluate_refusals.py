"""Check that the app refuses questions that the documents do not cover."""
import csv

from backend.rag import NOT_FOUND, answer

FILES = ["data/eval/questions.csv", "data/eval/questions_holdout.csv"]


def main() -> None:
    total = refused = 0
    for path in FILES:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["document"] != "NONE":
                    continue
                total += 1
                result = answer(row["question"])
                ok = result["answer"].strip() == NOT_FOUND
                refused += ok
                print(
                    f"[{'PASS' if ok else 'FAIL'}] Q{row['id']}: {row['question']}")
                if not ok:
                    print(f"        answered: {result['answer'][:150]}")
    print(f"\nCorrectly refused: {refused}/{total}")


if __name__ == "__main__":
    main()

"""Judge the final answers: is each one correct and complete vs the expected answer?"""
import os
import csv
import json
import re
import sys
import time

from backend import config
from backend.rag import answer, client

FILES = sys.argv[1:] or ["data/eval/questions.csv",
                         "data/eval/questions_holdout.csv"]
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openrouter/free")
OUT_FILE = "data/eval/answer_results.csv"

JUDGE_PROMPT = """You are grading a question-answering system.
Question: {question}
Reference answer (correct): {expected}
System answer: {actual}

Grade the system answer against the reference answer:
- CORRECT: it contains all the key facts of the reference answer and nothing that contradicts it.
- PARTIAL: it is right but misses some key facts of the reference answer.
- WRONG: it contradicts the reference answer, misses the main fact, or says it could not find the answer.
Reply with JSON only, like {{"verdict": "CORRECT", "reason": "one short sentence"}}"""


def with_retry(fn, tries=4):
    for attempt in range(tries):
        try:
            return fn()
        except Exception:
            if attempt == tries - 1:
                raise
            # free models hit rate limits sometimes
            time.sleep(5 * (attempt + 1))


def judge(question: str, expected: str, actual: str) -> tuple[str, str]:
    def run():
        r = client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            messages=[{
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=question, expected=expected, actual=actual
                ),
            }],
        )
        return r.choices[0].message.content or ""

    text = with_retry(run)
    match = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(match.group(0))
        verdict = str(data.get("verdict", "")).upper()
        if verdict not in {"CORRECT", "PARTIAL", "WRONG"}:
            verdict = "UNPARSED"
        return verdict, str(data.get("reason", ""))
    except Exception:
        return "UNPARSED", text[:100]


def main() -> None:
    rows = []
    for path in FILES:
        with open(path, encoding="utf-8") as f:
            rows += [r for r in csv.DictReader(f) if r["document"] != "NONE"]

    results = []
    for row in rows:
        result = with_retry(lambda: answer(row["question"]))
        verdict, reason = judge(
            row["question"], row["expected_answer"], result["answer"])
        results.append({
            "id": row["id"],
            "question": row["question"],
            "verdict": verdict,
            "reason": reason,
            "system_answer": result["answer"],
            "expected_answer": row["expected_answer"],
        })
        print(f"[{verdict:<8}] Q{row['id']}: {reason}")
        time.sleep(1)

    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    n = len(results)
    print("\n" + "=" * 50)
    for v in ["CORRECT", "PARTIAL", "WRONG", "UNPARSED"]:
        count = sum(r["verdict"] == v for r in results)
        if count or v != "UNPARSED":
            print(f"{v:<9} {count}/{n}")
    print(f"\nFull details saved to {OUT_FILE}")


if __name__ == "__main__":
    main()

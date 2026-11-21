"""Judge the final answers: is each one correct and complete vs the expected answer?

Environment variables:
  LLM_MODEL    model that writes the answers (default comes from backend/config.py)
  JUDGE_MODEL  model that grades them (default openrouter/free)
  RUN_NAME     label for the results file, e.g. RUN_NAME=nemotron
  RESUME=1     continue a stopped run, skipping questions already graded
"""
import csv
import json
import os
import re
import sys
import time

import openai

from backend import config
from backend.rag import answer, client

FILES = sys.argv[1:] or ["data/eval/questions.csv",
                         "data/eval/questions_holdout.csv"]
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openrouter/free")
RUN_NAME = os.getenv("RUN_NAME", "run")
OUT_FILE = f"data/eval/answer_results_{RUN_NAME}.csv"
FIELDS = ["id", "question", "verdict", "reason",
          "system_answer", "expected_answer"]

JUDGE_PROMPT = """You are grading a question-answering system.
Question: {question}
Reference answer (correct): {expected}
System answer: {actual}

Grade the system answer against the reference answer:
- CORRECT: it contains all the key facts of the reference answer and nothing that contradicts it.
- PARTIAL: it is right but misses some key facts of the reference answer.
- WRONG: it contradicts the reference answer, misses the main fact, or says it could not find the answer.
Reply with JSON only, like {{"verdict": "CORRECT", "reason": "one short sentence"}}"""


class DailyLimit(Exception):
    pass


def with_retry(fn, tries=4):
    for attempt in range(tries):
        try:
            return fn()
        except openai.RateLimitError as e:
            if "free-models-per-day" in str(e):
                raise DailyLimit(str(e))
            if attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))
        except Exception:
            if attempt == tries - 1:
                raise
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

    resume = os.getenv("RESUME") == "1" and os.path.exists(OUT_FILE)
    results = []
    if resume:
        with open(OUT_FILE, encoding="utf-8") as f:
            results = list(csv.DictReader(f))
    done_ids = {r["id"] for r in results}
    stopped = None

    with open(OUT_FILE, "a" if resume else "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        if not resume:
            writer.writeheader()
        for row in rows:
            if row["id"] in done_ids:
                continue
            try:
                result = with_retry(lambda: answer(row["question"]))
                verdict, reason = judge(
                    row["question"], row["expected_answer"], result["answer"]
                )
            except (DailyLimit, openai.RateLimitError) as e:
                stopped = str(e)[:150]
                break
            record = {
                "id": row["id"],
                "question": row["question"],
                "verdict": verdict,
                "reason": reason,
                "system_answer": result["answer"],
                "expected_answer": row["expected_answer"],
            }
            results.append(record)
            writer.writerow(record)
            out.flush()
            print(f"[{verdict:<8}] Q{row['id']}: {reason}")
            time.sleep(1)

    n = len(results)
    print("\n" + "=" * 50)
    print(f"Answer model: {config.LLM_MODEL} | Judge: {JUDGE_MODEL}")
    print(f"Graded {n} of {len(rows)} questions")
    for v in ["CORRECT", "PARTIAL", "WRONG", "UNPARSED"]:
        count = sum(r["verdict"] == v for r in results)
        if count or v != "UNPARSED":
            print(f"{v:<9} {count}/{n}")
    print(f"\nResults saved to {OUT_FILE}")
    if stopped:
        print(f"\nStopped early (rate limit): {stopped}")
        print("To continue later, run the same command with RESUME=1 in front.")


if __name__ == "__main__":
    main()

"""Run answer-level checks through OpenAI.

Answers are checkpointed locally so a free-tier quota interruption can be resumed.
The automated checks are screening heuristics, not a substitute for expert review.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.rag import answer_question, load_vectorstore

DATASET = BASE_DIR / "evaluation" / "answer_expectations.json"
QUESTION_DATASET = BASE_DIR / "evaluation" / "questions.json"
RESULTS = BASE_DIR / "evaluation" / ".answer_eval_results.jsonl"
UNCERTAINTY = (
    "not enough", "not provided", "not available", "not established",
    "cannot confirm", "not confirmed", "unverified", "does not confirm",
    "not in the available", "no source", "not documented",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="evaluate at most N pending questions (0 means all)")
    parser.add_argument("--resume", action="store_true", help="reuse saved answers and continue after completed questions")
    args = parser.parse_args()

    vectorstore = load_vectorstore()
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    questions = json.loads(QUESTION_DATASET.read_text(encoding="utf-8"))["questions"]
    source_map = {item["question"]: item.get("expected_sources", []) for item in questions}

    completed = {}
    if args.resume and RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if not item.get("generation_degraded"):
                    completed[item["question"]] = item

    pending = [item for item in dataset["expectations"] if item["question"] not in completed]
    if args.limit:
        pending = pending[:args.limit]
    target_count = len(dataset["expectations"]) if not args.limit else min(len(dataset["expectations"]), len(completed) + len(pending))
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        RESULTS.write_text("", encoding="utf-8")

    interrupted = False
    with RESULTS.open("a", encoding="utf-8") as output:
        for item in pending:
            try:
                result = answer_question(item["question"], vectorstore)
            except Exception as exc:
                print(f"Stopped after {len(completed)} completed answer(s): {type(exc).__name__}: {exc}")
                interrupted = True
                break

            answer = result["answer"]
            lowered = answer.casefold()
            checks = item.get("must_include_any", [])
            passed_checks = sum(any(term.casefold() in lowered for term in group) for group in checks)
            labels = [int(number) for number in re.findall(r"\[S(\d+)\]", answer)]
            sources = result.get("sources", [])
            valid_citations = bool(labels) and max(labels) <= len(sources)
            expected = set(source_map.get(item["question"], []))
            unique_sources = list({source.get("source") for source in sources})
            matching_sources = len(set(unique_sources) & expected)
            uncertainty = any(phrase in lowered for phrase in UNCERTAINTY)
            abstention_ok = not item.get("expect_abstention") or uncertainty
            record = {
                "question": item["question"],
                "answer": answer,
                "sources": sources,
                "fact_checks": passed_checks,
                "fact_total": len(checks),
                "valid_citations": valid_citations,
                "expected_source_hit": not expected or matching_sources > 0,
                "context_precision": matching_sources / max(len(unique_sources), 1),
                "expected_abstention": item.get("expect_abstention", False),
                "abstention_ok": abstention_ok,
                "generation_degraded": bool(result.get("degraded", False)),
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

            if record["generation_degraded"]:
                print("Generation is rate-limited; saved the retrieved passages and paused. Resume after quota is available.")
                interrupted = True
                break
            completed[item["question"]] = record

    records = list(completed.values())
    count = len(records)
    if not count:
        print("No complete answer results are available.")
        return 2

    fact_checks = sum(row["fact_checks"] for row in records)
    fact_total = sum(row["fact_total"] for row in records)
    citations_ok = sum(row["valid_citations"] for row in records)
    source_hits = sum(row["expected_source_hit"] for row in records)
    precision_sum = sum(row["context_precision"] for row in records)
    abstention_rows = [row for row in records if row["expected_abstention"]]
    abstentions_ok = sum(row["abstention_ok"] for row in abstention_rows)

    print(f"Questions scored: {count}/{target_count if args.limit else len(dataset['expectations'])}")
    print(f"Expected fact phrase checks: {fact_checks}/{fact_total} ({fact_checks / max(fact_total, 1):.1%})")
    print(f"Cited answers: {citations_ok}/{count} ({citations_ok / count:.1%})")
    print(f"Expected source present: {source_hits}/{count} ({source_hits / count:.1%})")
    print(f"Context precision: {precision_sum / count:.1%}")
    print(f"Abstention checks: {abstentions_ok}/{len(abstention_rows)} ({abstentions_ok / max(len(abstention_rows), 1):.1%})")
    print(f"Answer details saved locally to {RESULTS.name}")

    complete = count >= target_count and not interrupted
    if not complete:
        print("Evaluation is incomplete. Continue later with --resume.")
        return 2
    return 0 if all(row["fact_checks"] == row["fact_total"] and row["valid_citations"] and row["abstention_ok"] for row in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())


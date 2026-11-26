"""Evaluate top-k source retrieval against the curated question set.

Default mode is offline lexical retrieval to avoid API calls. Use --semantic to also
query the configured Gemini embedding endpoint. Use --sync to refresh the index first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from app.rag import BASE_DIR, load_vectorstore, retrieve_documents, sync_vectorstore

DATASET = BASE_DIR / "evaluation" / "questions.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", action="store_true", help="incrementally sync sources before evaluation")
    parser.add_argument("--semantic", action="store_true", help="include FAISS semantic retrieval (uses embedding API)")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    vectorstore = sync_vectorstore() if args.sync else load_vectorstore()
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    rows = []
    for item in dataset["questions"]:
        results = retrieve_documents(item["question"], vectorstore, k=args.k, dense=args.semantic)
        expected = set(item["expected_sources"])
        ranked = list(dict.fromkeys(str(doc.metadata.get("source", "")) for doc in results))
        ranks = [index + 1 for index, source in enumerate(ranked) if source in expected]
        relevant_count = len(set(ranked[:args.k]) & expected)
        precision = relevant_count / min(len(ranked[:args.k]), args.k) if ranked[:args.k] else 0.0
        rows.append({"question": item["question"], "hit": bool(ranks), "reciprocal_rank": 1 / min(ranks) if ranks else 0, "precision": precision, "sources": ranked})

    total = len(rows)
    hit_rate = sum(row["hit"] for row in rows) / total if total else 0.0
    mrr = sum(row["reciprocal_rank"] for row in rows) / total if total else 0.0
    context_precision = sum(row["precision"] for row in rows) / total if total else 0.0
    print(f"Mode: {'semantic + BM25' if args.semantic else 'offline BM25'}")
    print(f"Questions: {total}")
    print(f"Source hit@{args.k}: {hit_rate:.1%}")
    print(f"MRR@{args.k}: {mrr:.3f}")
    print(f"Context precision@{args.k}: {context_precision:.1%}")
    for row in rows:
        mark = "PASS" if row["hit"] else "MISS"
        print(f"{mark} | {row['question']} | {', '.join(row['sources'][:3])}")
    return 0 if hit_rate >= 0.80 else 1


if __name__ == "__main__":
    raise SystemExit(main())


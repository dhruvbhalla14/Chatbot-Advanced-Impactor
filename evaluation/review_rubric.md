# Answer and retrieval review rubric

The automated checks are reproducible but do not prove factual correctness. Review answers against the source passages for the following dimensions:

1. **Retrieval relevance:** Is at least one expected source in the retrieved list? The current suite reports source hit@5 and MRR over 33 labeled questions.
2. **Context precision:** Do the retrieved passages materially support the answer? The current numeric “context precision” uses expected-source labels as a proxy. Labels are incomplete relevance judgments, so manual review is needed before comparing this score across releases.
3. **Answer correctness:** Are all material claims supported by retrieved passages? Check product names, feature boundaries, dates, system components, and figures.
4. **Source attribution:** Are citations present, valid, and attached to the claims they support? The API can verify citation IDs map to returned sources; a reviewer must verify claim-to-source entailment.
5. **Hallucination prevention:** For missing or unverified facts, does the answer clearly state the evidence gap? In particular, do not assert revenue, customer counts, current partnership terms, Pro/CoachPlex equivalence, model accuracy, or completion of the Lord's showcase without evidence.
6. **Historical and classification handling:** Are old architecture and dataset figures labeled HISTORICAL, internal claims labeled INTERNAL_CONTEXT, and interpretations labeled ANALYTICAL?

The 32-question answer-level expectations are in answer_expectations.json. The answer runner calls OpenAI, checkpoints each completed result in an ignored local JSONL file, and can resume after quota is available. Its phrase checks are heuristics; they are not a judge model or published benchmark.


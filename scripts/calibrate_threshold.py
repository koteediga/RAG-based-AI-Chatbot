"""Derive the refusal threshold and score range from YOUR data.

Cosine similarity scales differ between embedding models, so instead of guessing,
measure: run in-scope and out-of-scope questions through the retriever (no LLM
calls, so it's cheap) and pick a threshold that separates them.

Usage (after ingestion):
    python -m scripts.calibrate_threshold

Edit IN_SCOPE / OUT_OF_SCOPE below to add your own questions; more is better.
"""
import statistics

from src.config import get_settings
from src.ingestion import get_vector_store

IN_SCOPE = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "What challenges or risks are associated with deploying agentic systems?",
    "How do agents use tools?",
    "What is the role of planning in agentic systems?",
    "What is a multi-agent system?",
]

OUT_OF_SCOPE = [
    "Who won the 2022 FIFA World Cup?",
    "What is the recipe for chocolate chip cookies?",
    "What is the capital of Australia?",
    "How tall is Mount Everest?",
    "Who wrote Pride and Prejudice?",
    "What is the boiling point of water in Fahrenheit?",
]


def top_scores(store, queries, k):
    rows = []
    for q in queries:
        results = store.similarity_search_with_score(q, k=k)
        rows.append((q, max((s for _, s in results), default=0.0)))
    return rows


def main() -> None:
    settings = get_settings()
    store = get_vector_store(settings)

    print("\nIN-SCOPE (should be answered)")
    inn = top_scores(store, IN_SCOPE, settings.top_k)
    for q, s in inn:
        print(f"  {s:.3f}  {q}")

    print("\nOUT-OF-SCOPE (should be refused)")
    out = top_scores(store, OUT_OF_SCOPE, settings.top_k)
    for q, s in out:
        print(f"  {s:.3f}  {q}")

    lo_in = min(s for _, s in inn)
    hi_out = max(s for _, s in out)
    print(f"\nlowest in-scope = {lo_in:.3f} | highest out-of-scope = {hi_out:.3f}")

    if lo_in > hi_out:
        threshold = round((lo_in + hi_out) / 2, 3)
        print(f"Clean separation (gap {lo_in - hi_out:.3f}). Suggested threshold = midpoint.")
    else:
        # Overlap: bias toward answering; the strict prompt still stops out-of-scope answers.
        threshold = round(hi_out + 0.01, 3)
        print("WARNING: distributions overlap. Threshold set just above the best out-of-scope score;")
        print("         some in-scope questions may be refused. Add more/cleaner questions or review chunking.")

    floor = round(statistics.fmean(s for _, s in out), 3)
    ceiling = round(statistics.median(s for _, s in inn) + 0.10, 3)
    ceiling = max(ceiling, round(floor + 0.2, 3))
    print("\nPaste into .env, then restart the API:\n")
    print(f"MIN_RELEVANCE_SCORE={threshold}")
    print(f"SCORE_FLOOR={floor}")
    print(f"SCORE_CEILING={min(ceiling, 1.0)}")


if __name__ == "__main__":
    main()

"""Benchmark queries that demonstrate document grounding.

Usage:
    python tests_sample_queries.py                 # against a running API (uvicorn app:app)
    python tests_sample_queries.py --url http://127.0.0.1:8000
    python tests_sample_queries.py --direct        # call the graph directly, no server

Writes sample_outputs.md (paste it into the README) and sample_outputs.json.
Exit code is non-zero if the out-of-scope query is NOT refused.

Tuning tip: look at the `top_similarity` column. Set MIN_RELEVANCE_SCORE in .env
between the highest out-of-scope value and the lowest in-scope value.
"""
import argparse
import json
import sys

# (query, should_be_answered)
QUERIES = [
    ("What is Agentic AI according to the eBook?", True),
    ("How do AI agents differ from traditional automation systems?", True),
    ("What are the core components of an Agentic Architecture?", True),
    ("What role does memory play in Agentic AI workflows?", True),
    ("What challenges or risks are associated with deploying agentic systems?", True),
    ("Who won the 2022 FIFA World Cup?", False),  # validation: must refuse
]


def make_runner(args):
    if args.direct:
        from src.graph import ask, build_rag_graph

        graph = build_rag_graph()

        def run(q):
            r = ask(graph, q)
            return {
                "answer": r["answer"],
                "retrieved_chunks": r["chunks"],
                "confidence_score": r["score"],
                "grounded": r["grounded"],
                "top_similarity": r["top_similarity"],
            }

        return run

    import requests

    def run(q):
        resp = requests.post(f"{args.url}/chat", json={"query": q}, timeout=120)
        resp.raise_for_status()
        return resp.json()

    return run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--direct", action="store_true", help="skip the API, call the graph")
    args = parser.parse_args()
    run = make_runner(args)

    results, md, failures = [], ["# Sample query results\n"], 0
    for i, (query, should_answer) in enumerate(QUERIES, start=1):
        out = run(query)
        ok = out["grounded"] == should_answer
        failures += 0 if ok else 1
        status = "PASS" if ok else "CHECK"
        print(f"\n[{i}] {query}")
        print(f"    {status} | grounded={out['grounded']} | confidence={out['confidence_score']:.2f}"
              f" | top_similarity={out['top_similarity']:.3f}")
        print(f"    answer: {out['answer'][:300]}")
        for c in out["retrieved_chunks"]:
            print(f"      - p.{c['page']} sim={c['similarity']:.3f}: {c['text'][:90]!r}")

        results.append({"query": query, "expected_answered": should_answer, **out})
        md.append(f"## {i}. {query}\n")
        md.append(f"- **Result:** {status} (expected {'answer' if should_answer else 'refusal'})")
        md.append(f"- **Confidence:** {out['confidence_score']:.2f}  |  **Top similarity:** {out['top_similarity']:.3f}")
        md.append(f"- **Answer:** {out['answer']}")
        md.append("- **Retrieved chunks:**")
        for c in out["retrieved_chunks"]:
            snippet = c["text"][:160].replace("\n", " ")
            md.append(f"  - p.{c['page']} (sim {c['similarity']:.3f}): {snippet}...")
        md.append("")

    with open("sample_outputs.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open("sample_outputs.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\nDone. {len(QUERIES) - failures}/{len(QUERIES)} behaved as expected. "
          "Saved sample_outputs.md / .json")
    fifa_ok = results[-1]["grounded"] is False
    if not fifa_ok:
        print("ERROR: the out-of-scope query was answered. Raise MIN_RELEVANCE_SCORE.")
    return 0 if fifa_ok else 1


if __name__ == "__main__":
    sys.exit(main())

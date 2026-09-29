# Design Walkthrough (read this before any follow-up call)

The submission rules forbid code you can't explain, so this file walks through every moving part in plain language.
Read the code next to it, run it, change something, and break something on purpose. Then you will actually own it.

## 1. The big picture

RAG = *Retrieval-Augmented Generation*. An LLM alone answers from memory and can hallucinate. RAG first **looks up**
relevant passages from your own document, then asks the LLM to answer **using only those passages**.

Two phases:

1. **Ingestion (offline, once):** PDF → text per page → chunks → embedding vectors → Pinecone.
2. **Query (online, per question):** question → embed → nearest chunks from Pinecone → LLM answer → return
   answer + chunks + score.

## 2. `src/config.py`

- `Settings` is a frozen dataclass holding every tunable: chunk size, top-k, thresholds, model names.
- `get_settings()` reads the environment (`.env` is loaded by `python-dotenv`).
- `REFUSAL_MESSAGE` is one constant used everywhere so refusal detection can't drift.
- `embedding_dim = 1536` must equal the Pinecone index dimension. `gemini-embedding-001` natively returns 3072 dims but
  supports Matryoshka truncation, so we request 1536 via `output_dimensionality`.
- `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) is the Gemini key from Google AI Studio.

## 3. `src/ingestion.py`

| Function | What it does |
|---|---|
| `download_pdf` | `gdown` fetches the Drive file; then checks the first bytes are `%PDF-` (Drive sometimes returns an HTML error page) |
| `load_pdf` | `PyPDFLoader` gives one `Document` per page with `metadata["page"]` (0-indexed) |
| `split_documents` | `RecursiveCharacterTextSplitter` tries to split on paragraphs, then lines, then spaces, then characters, so chunks stay semantically intact. Adds `page_number` (1-indexed), `chunk_index` and `chunk_id` |
| `make_chunk_id` | SHA-1 of page + index + text. Same input gives the same id, so re-ingestion **overwrites** instead of duplicating |
| `ensure_index` | Creates a serverless Pinecone index (1536, cosine) if missing; if it exists with a wrong dimension it raises rather than corrupting silently |
| `run_ingestion` | Orchestrates the above; `--reset` clears the namespace first |

**Why chunk 1000 / overlap 200?** Chunks big enough to hold a complete idea, small enough that an embedding stays
focused. The overlap stops a sentence that straddles a boundary from being lost.

**Why drop chunks under 50 chars?** Page numbers and blank pages embed to noise that can pollute retrieval.

## 4. `src/graph.py`

### State
`AgentState` is a `TypedDict`. Each node returns a *partial* dict and LangGraph merges it into the state.

### Nodes
- **retrieve:** `similarity_search_with_score(question, k=4)` (the question is embedded with task type `RETRIEVAL_QUERY`). Pinecone returns cosine similarity (higher = closer).
  Builds `chunks` (rank, page, similarity, text), `context` (just texts) and `top_similarity`.
- **route_after_retrieve:** the conditional edge. If `top_similarity >= min_relevance` go to `generate`, else `refuse`.
- **generate:** builds a prompt with a strict system message and the numbered sources, calls the LLM
  (temperature configurable, default 0), then either flags a self-refusal or computes confidence.
- **refuse:** returns the fixed refusal message with score 0 and **never calls the LLM** (also saves cost).

### Confidence formula

```
norm(x) = clamp((x - 0.15) / (0.65 - 0.15), 0, 1)
confidence = 0.7 * norm(best similarity) + 0.3 * norm(mean similarity)
```

- Cosine similarities depend on the embedding model, so floor and ceiling are calibrated on your data with
  `scripts/calibrate_threshold.py` (floor ≈ typical out-of-scope score, ceiling ≈ typical in-scope score).
  Normalising stretches that useful band onto 0–1.
- Best chunk = "is there a strong match?" (70%). Mean = "is the whole neighbourhood relevant?" (30%).
- It is a **retrieval-quality proxy**, not a calibrated probability. Say that openly if asked.
- The floor, ceiling and threshold are configurable because they depend on your document and model.

### Why three grounding layers?
1. **Refusal gate** (deterministic): off-topic questions never reach the LLM.
2. **Strict prompt:** covers borderline questions that pass the gate but aren't actually answered by the text.
3. **Score reset:** if the LLM refuses anyway, we report grounded=False and confidence 0 instead of a misleading score.

## 5. `app.py`

- `lifespan` builds the graph once at startup (not per request).
- `/chat` validates with Pydantic (`query` 1–2000 chars, blank rejected with 422).
- It's a plain `def`, so FastAPI runs it in a worker thread; the graph does blocking I/O.
- Errors from Gemini/Pinecone become HTTP 502 with a message instead of a stack trace.

## 6. Tests

`tests/test_offline.py` injects fake vector stores and LLMs, so the routing and scoring logic is verified with no
network. It covers: refusal gate never calls the LLM, LLM self-refusal zeroes the score, empty retrieval, configurable
threshold, API contract and validation, chunk-id determinism, splitting metadata, and a real PDF round-trip.

## 7. Likely follow-up questions

**Why LangGraph for a linear pipeline?** The assignment asks for it, and the conditional edge (generate vs refuse)
is real branching. It also makes it easy to extend with query rewriting, re-ranking or a retry loop.

**How would you improve retrieval quality?** Hybrid search (BM25 + vectors), a cross-encoder re-ranker, query
rewriting, metadata filters, and evaluating chunk sizes on a labelled question set.

**How do you evaluate it?** Build a small question/expected-page set; measure hit-rate@k for retrieval and use
LLM-as-judge or manual review for faithfulness; track refusal precision on out-of-scope questions.

**Why is the score not a probability?** Cosine similarity isn't calibrated. To calibrate you'd fit a mapping
(e.g., isotonic regression) from scores to correctness on labelled data.

**What if the PDF is scanned?** `PyPDFLoader` would return no text; you'd need OCR (e.g., Tesseract). The code raises a
clear error when nothing is extracted.

**How do you prevent prompt injection from the document?** The system prompt tells the model to treat context as
data; temperature 0 and the refusal rule reduce risk. It isn't bulletproof, and a production system would add
input/output filtering.

**How would you deploy it?** Containerise with Docker, keep API keys in a secret manager, run Uvicorn/Gunicorn
workers behind a load balancer, add request logging and rate limiting.

**Why Gemini instead of OpenAI?** The LangChain interfaces are the same (`Embeddings` and chat model), so only `get_embeddings` and the LLM constructor changed; the graph, scoring and API are provider-independent.

**Why did you not use the reference's hard-coded 0.95?** It says nothing about retrieval quality. The same value
would be reported for a perfect match and a random one.

## 8. Things to try yourself (this is how you learn it)

1. Set `MIN_RELEVANCE_SCORE=0.0` and ask the FIFA question. See what the strict prompt alone does (the LLM should still refuse).
2. Change `CHUNK_SIZE` to 300 and 2000, re-ingest with `--reset`, and compare answers.
3. Change `TOP_K` and watch the confidence score move.
4. Add a print in `retrieve_node` to inspect raw scores for several questions.

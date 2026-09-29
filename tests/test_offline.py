"""Offline tests: no OpenAI / Pinecone keys or network needed."""
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.config import REFUSAL_MESSAGE, Settings
from src.graph import ask, build_rag_graph, compute_confidence, is_refusal, message_text, normalize_score
from src.ingestion import get_embeddings, load_pdf, make_chunk_id, split_documents, upsert_in_batches

S = Settings()


class FakeVS:
    def __init__(self, results):
        self.results = results

    def similarity_search_with_score(self, query, k=4):
        return self.results[:k]


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def invoke(self, messages):
        self.calls += 1
        self.last_messages = messages

        class R:
            content = self.reply

        return R()


def doc(text, page=0):
    return Document(page_content=text, metadata={"page": page, "page_number": page + 1})


IN_SCOPE = [(doc("Agentic AI systems plan and act autonomously.", 2), 0.78),
            (doc("Agents use tools and memory.", 3), 0.71),
            (doc("Automation follows fixed rules.", 4), 0.62)]
OUT_OF_SCOPE = [(doc("Agents use tools.", 3), 0.42), (doc("Memory.", 4), 0.39)]


# ---- scoring helpers ------------------------------------------------------ #
def test_normalize_bounds():
    assert normalize_score(0.0, 0.15, 0.65) == 0.0
    assert normalize_score(0.9, 0.15, 0.65) == 1.0
    assert normalize_score(0.40, 0.15, 0.65) == pytest.approx(0.5)


def test_normalize_bad_range():
    with pytest.raises(ValueError):
        normalize_score(0.5, 0.6, 0.6)


def test_confidence_monotonic_and_bounded():
    low = compute_confidence([0.3, 0.2], 0.15, 0.65)
    high = compute_confidence([0.6, 0.55], 0.15, 0.65)
    assert 0.0 <= low < high <= 1.0
    assert compute_confidence([], 0.15, 0.65) == 0.0


def test_is_refusal():
    assert is_refusal("I cannot answer based on the provided document.")
    assert is_refusal("i cannot answer based on the provided document")
    assert not is_refusal("Agentic AI is autonomous.")


# ---- graph ---------------------------------------------------------------- #
def test_grounded_answer_path():
    llm = FakeLLM("Agentic AI plans and acts autonomously [Source 1].")
    g = build_rag_graph(S, FakeVS(IN_SCOPE), llm)
    out = ask(g, "What is Agentic AI?")
    assert out["grounded"] is True
    assert llm.calls == 1
    assert 0.0 < out["score"] <= 1.0
    assert out["chunks"][0]["rank"] == 1 and out["chunks"][0]["page"] == 3
    assert out["context"][0].startswith("Agentic AI")
    # Prompt actually contains the retrieved context and the question
    assert "Agentic AI systems plan" in llm.last_messages[1].content
    assert "What is Agentic AI?" in llm.last_messages[1].content


def test_refusal_gate_skips_llm():
    llm = FakeLLM("should never be used")
    g = build_rag_graph(S, FakeVS(OUT_OF_SCOPE), llm)
    out = ask(g, "Who won the 2022 FIFA World Cup?")
    assert llm.calls == 0
    assert out["answer"] == REFUSAL_MESSAGE
    assert out["score"] == 0.0 and out["grounded"] is False
    assert len(out["chunks"]) == 2  # still shown for transparency


def test_llm_refusal_zeroes_score():
    g = build_rag_graph(S, FakeVS(IN_SCOPE), FakeLLM(REFUSAL_MESSAGE))
    out = ask(g, "Something the chunks don't cover")
    assert out["grounded"] is False and out["score"] == 0.0


def test_empty_retrieval_refuses():
    llm = FakeLLM("x")
    out = ask(build_rag_graph(S, FakeVS([]), llm), "anything")
    assert out["grounded"] is False and llm.calls == 0 and out["top_similarity"] == 0.0


def test_threshold_is_configurable():
    strict = Settings(min_relevance=0.9)
    llm = FakeLLM("answer")
    out = ask(build_rag_graph(strict, FakeVS(IN_SCOPE), llm), "q")
    assert out["grounded"] is False and llm.calls == 0


# ---- API ------------------------------------------------------------------ #
def make_client(results, reply):
    import app as app_module

    app_module.app.state.graph = build_rag_graph(S, FakeVS(results), FakeLLM(reply))
    return TestClient(app_module.app)


def test_api_chat_contract():
    with make_client(IN_SCOPE, "It plans autonomously.") as c:
        r = c.post("/chat", json={"query": "What is Agentic AI?"})
        assert r.status_code == 200
        body = r.json()
        assert set(body) >= {"answer", "retrieved_chunks", "confidence_score"}
        assert body["answer"] == "It plans autonomously."
        assert len(body["retrieved_chunks"]) == 3
        assert {"rank", "page", "similarity", "text"} <= set(body["retrieved_chunks"][0])
        assert 0 < body["confidence_score"] <= 1


def test_api_refuses_out_of_scope():
    with make_client(OUT_OF_SCOPE, "unused") as c:
        body = c.post("/chat", json={"query": "Who won the 2022 FIFA World Cup?"}).json()
        assert body["answer"] == REFUSAL_MESSAGE and body["confidence_score"] == 0.0


def test_api_validation_and_health():
    with make_client(IN_SCOPE, "a") as c:
        assert c.get("/health").json() == {"status": "ok"}
        assert c.post("/chat", json={"query": ""}).status_code == 422
        assert c.post("/chat", json={}).status_code == 422
        assert c.post("/chat", json={"query": "   "}).status_code == 422


# ---- ingestion ------------------------------------------------------------ #
def test_chunk_id_deterministic():
    assert make_chunk_id(1, 2, "abc") == make_chunk_id(1, 2, "abc")
    assert make_chunk_id(1, 2, "abc") != make_chunk_id(1, 3, "abc")


def test_split_documents_metadata_and_overlap():
    long_text = ("Agentic AI systems use planning, memory and tools. " * 60)
    docs = [Document(page_content=long_text, metadata={"page": 4, "source": "x.pdf"}),
            Document(page_content="12", metadata={"page": 5})]  # too short -> dropped
    chunks = split_documents(docs, Settings(chunk_size=500, chunk_overlap=100))
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 500 for c in chunks)
    assert all(c.metadata["page_number"] == 5 for c in chunks)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(set(ids)) == len(ids)


def test_load_pdf_roundtrip(tmp_path):
    reportlab = pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas

    pdf = tmp_path / "t.pdf"
    c = canvas.Canvas(str(pdf))
    c.drawString(72, 750, "Agentic AI is a paradigm where software agents plan and act.")
    c.showPage()
    c.drawString(72, 750, "Memory lets agents retain context across steps of a workflow.")
    c.save()
    docs = load_pdf(pdf)
    assert len(docs) == 2
    chunks = split_documents(docs, Settings())
    assert {c.metadata["page_number"] for c in chunks} == {1, 2}


def test_load_pdf_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "nope.pdf")


# ---- Gemini-specific ------------------------------------------------------ #
def test_message_text_handles_string_and_block_list():
    class M:  # mimics an AIMessage whose content is a list of blocks
        content = [{"type": "thinking", "thinking": "hmm"},
                   {"type": "text", "text": "Hello "}, "world"]

    assert message_text(M()) == "Hello world"
    assert message_text(FakeLLM("plain").invoke([])) == "plain"


def test_llm_block_list_reply_is_used_as_answer():
    class BlockLLM:
        def invoke(self, messages):
            class R:
                content = [{"type": "text", "text": "Agents plan and act [Source 1]."}]
            return R()

    out = ask(build_rag_graph(S, FakeVS(IN_SCOPE), BlockLLM()), "q")
    assert out["answer"] == "Agents plan and act [Source 1]." and out["grounded"] is True


def test_embeddings_config_matches_index_dimension():
    emb = get_embeddings(Settings(google_api_key="fake-key"))
    assert emb.output_dimensionality == 1536
    assert emb.model.endswith("gemini-embedding-001")
    assert emb.task_type is None  # wrapper picks DOCUMENT for ingest / QUERY for search


def test_upsert_batches_and_retries_on_rate_limit(monkeypatch):
    import src.ingestion as ing

    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)

    class Store:
        def __init__(self):
            self.batches, self.fail_once = [], True

        def add_documents(self, docs, ids=None):
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            self.batches.append((len(docs), len(ids)))

    chunks = [Document(page_content="x" * 60, metadata={"chunk_id": str(i)}) for i in range(70)]
    store = Store()
    upsert_in_batches(store, chunks, Settings(embed_batch_size=32))
    assert store.batches == [(32, 32), (32, 32), (6, 6)]


def test_upsert_raises_non_rate_limit_errors(monkeypatch):
    import src.ingestion as ing

    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)

    class Bad:
        def add_documents(self, docs, ids=None):
            raise RuntimeError("invalid api key")

    with pytest.raises(RuntimeError):
        upsert_in_batches(Bad(), [Document(page_content="x" * 60, metadata={"chunk_id": "1"})], Settings())


def test_require_keys_names_gemini():
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        Settings(pinecone_api_key="p").require_keys()

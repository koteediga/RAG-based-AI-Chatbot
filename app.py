"""FastAPI interface.

Run:  uvicorn app:app --reload
Docs: http://127.0.0.1:8000/docs

POST /chat  {"query": "..."}  ->  {answer, retrieved_chunks, confidence_score, ...}
"""
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.graph import ask, build_rag_graph

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the graph once at startup (tests may pre-inject app.state.graph).
    if getattr(app.state, "graph", None) is None:
        app.state.graph = build_rag_graph()
    yield


app = FastAPI(
    title="Agentic AI RAG API",
    description="Answers questions strictly from the Agentic AI eBook.",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, examples=["What is Agentic AI?"])


class RetrievedChunk(BaseModel):
    rank: int
    page: int
    similarity: float = Field(..., description="Raw cosine similarity from Pinecone")
    text: str


class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    grounded: bool = Field(..., description="False when the system refused to answer")
    top_similarity: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Plain `def` (not async): the graph makes blocking network calls, so FastAPI
# runs it in its thread pool instead of blocking the event loop.
@app.post("/chat", response_model=ChatResponse)
def chat(request: QueryRequest) -> ChatResponse:
    question = request.query.strip()
    if not question:
        raise HTTPException(status_code=422, detail="query must not be blank")
    try:
        result = ask(app.state.graph, question)
    except Exception as exc:
        logger.exception("Graph execution failed")
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}") from exc

    return ChatResponse(
        answer=result["answer"],
        retrieved_chunks=[RetrievedChunk(**c) for c in result["chunks"]],
        confidence_score=result["score"],
        grounded=result["grounded"],
        top_similarity=result["top_similarity"],
    )

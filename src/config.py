"""Environment setup & constants.

Everything tunable lives here so the rest of the code has no magic numbers.
`Settings()` (all defaults) is safe to build in tests; `get_settings()` reads
the environment / .env file.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# Public Google Drive file id of the Agentic AI eBook
GDRIVE_FILE_ID = "15VLphKcY23_fpYxN62UEQRri_psRVfP9"
PDF_PATH = ROOT_DIR / "data" / "Ebook-Agentic-AI.pdf"

# Returned verbatim whenever the answer is not supported by the document.
REFUSAL_MESSAGE = "I cannot answer based on the provided document."


@dataclass(frozen=True)
class Settings:
    google_api_key: str = ""  # Gemini API key (Google AI Studio)
    pinecone_api_key: str = ""
    index_name: str = "agentic-ai-index"
    namespace: str = "agentic-ai"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # gemini-embedding-001 natively outputs 3072 dims but supports Matryoshka
    # truncation, so we request 1536 to match the 1536-dim Pinecone index.
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 1536  # must equal the index dimension
    llm_model: str = "gemini-3.1-flash-lite"
    llm_temperature: float = 0.0

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4

    # Refusal gate on the best cosine similarity. Gemini embeddings score higher
    # than older models', so these are starting points: run
    # `python -m scripts.calibrate_threshold` after ingestion to set them from data.
    min_relevance: float = 0.50
    # Cosine range that is stretched onto 0..1 for the confidence score
    score_floor: float = 0.40
    score_ceiling: float = 0.80

    # Ingestion throttling (free-tier friendly)
    embed_batch_size: int = 32
    embed_sleep_seconds: float = 1.0

    def require_keys(self) -> None:
        missing = [
            name
            for name, val in (
                ("GOOGLE_API_KEY", self.google_api_key),
                ("PINECONE_API_KEY", self.pinecone_api_key),
            )
            if not val
        ]
        if missing:
            raise RuntimeError(
                f"Missing environment variables: {', '.join(missing)}. "
                "Copy .env.example to .env and fill them in."
            )


def get_settings() -> Settings:
    """Build Settings from environment variables (with defaults)."""
    d = Settings()
    e = os.getenv
    return Settings(
        google_api_key=e("GOOGLE_API_KEY") or e("GEMINI_API_KEY", ""),
        pinecone_api_key=e("PINECONE_API_KEY", ""),
        index_name=e("PINECONE_INDEX_NAME", d.index_name),
        namespace=e("PINECONE_NAMESPACE", d.namespace),
        pinecone_cloud=e("PINECONE_CLOUD", d.pinecone_cloud),
        pinecone_region=e("PINECONE_REGION", d.pinecone_region),
        embedding_model=e("EMBEDDING_MODEL", d.embedding_model),
        llm_model=e("LLM_MODEL", d.llm_model),
        llm_temperature=float(e("LLM_TEMPERATURE", d.llm_temperature)),
        chunk_size=int(e("CHUNK_SIZE", d.chunk_size)),
        chunk_overlap=int(e("CHUNK_OVERLAP", d.chunk_overlap)),
        top_k=int(e("TOP_K", d.top_k)),
        min_relevance=float(e("MIN_RELEVANCE_SCORE", d.min_relevance)),
        score_floor=float(e("SCORE_FLOOR", d.score_floor)),
        score_ceiling=float(e("SCORE_CEILING", d.score_ceiling)),
        embed_batch_size=int(e("EMBED_BATCH_SIZE", d.embed_batch_size)),
        embed_sleep_seconds=float(e("EMBED_SLEEP_SECONDS", d.embed_sleep_seconds)),
    )

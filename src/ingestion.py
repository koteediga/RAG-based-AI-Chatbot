"""PDF loading, splitting & Pinecone index setup (the ETL step).

Run:  python -m src.ingestion --download          # download PDF + ingest
      python -m src.ingestion                     # ingest data/Ebook-Agentic-AI.pdf
      python -m src.ingestion --reset             # wipe the namespace, then re-ingest

Ingestion is idempotent: every chunk gets a deterministic id (hash of its
page, position and text), so re-running upserts the same vectors instead of
creating duplicates.
"""
import argparse
import hashlib
import logging
import time
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from .config import GDRIVE_FILE_ID, PDF_PATH, Settings, get_settings

logger = logging.getLogger(__name__)

MIN_CHUNK_CHARS = 50  # drop near-empty chunks (page numbers, blank pages)


# --------------------------------------------------------------------------- #
# 1. Download
# --------------------------------------------------------------------------- #
def download_pdf(dest: Path = PDF_PATH, file_id: str = GDRIVE_FILE_ID) -> Path:
    """Download the eBook from Google Drive and verify it is really a PDF."""
    import gdown  # imported lazily: only needed for the download step

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(dest), quiet=False)
    if not dest.exists() or dest.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError(
            "Download did not produce a valid PDF (Drive may have rate-limited it). "
            f"Download it manually from https://drive.google.com/file/d/{file_id}/view "
            f"and save it as {dest}."
        )
    return dest


# --------------------------------------------------------------------------- #
# 2. Load + split
# --------------------------------------------------------------------------- #
def load_pdf(pdf_path: Path) -> List[Document]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"{pdf_path} not found. Run `python -m src.ingestion --download` first."
        )
    docs = PyPDFLoader(str(pdf_path)).load()  # one Document per page
    logger.info("Loaded %d pages", len(docs))
    return docs


def make_chunk_id(page: int, index: int, text: str) -> str:
    """Deterministic id => re-running ingestion overwrites instead of duplicating."""
    return hashlib.sha1(f"{page}:{index}:{text}".encode("utf-8")).hexdigest()


def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = [
        c for c in splitter.split_documents(docs) if len(c.page_content.strip()) >= MIN_CHUNK_CHARS
    ]
    # Human-friendly page number (PyPDFLoader pages are 0-indexed) + stable ids
    for i, chunk in enumerate(chunks):
        page0 = int(chunk.metadata.get("page", 0))
        chunk.metadata["page_number"] = page0 + 1
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_id"] = make_chunk_id(page0, i, chunk.page_content)
    logger.info("Created %d chunks", len(chunks))
    return chunks


# --------------------------------------------------------------------------- #
# 3. Pinecone
# --------------------------------------------------------------------------- #
def get_embeddings(settings: Settings) -> GoogleGenerativeAIEmbeddings:
    """Gemini embeddings truncated to `embedding_dim` (1536) dimensions.

    task_type is deliberately NOT set: the wrapper then uses RETRIEVAL_DOCUMENT
    for embed_documents (ingestion) and RETRIEVAL_QUERY for embed_query (search),
    which is what Gemini's retrieval-tuned embeddings expect.
    """
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
        output_dimensionality=settings.embedding_dim,
    )


def ensure_index(pc: Pinecone, settings: Settings) -> None:
    """Create the serverless index (1536-dim, cosine) if it does not exist."""
    name = settings.index_name
    if not pc.has_index(name):
        logger.info("Creating Pinecone index '%s'", name)
        pc.create_index(
            name=name,
            dimension=settings.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)
    else:
        desc = pc.describe_index(name)
        if desc.dimension != settings.embedding_dim or desc.metric != "cosine":
            raise RuntimeError(
                f"Index '{name}' exists with dimension={desc.dimension}, metric={desc.metric}; "
                f"expected {settings.embedding_dim}/cosine. Use a different PINECONE_INDEX_NAME."
            )


def get_vector_store(settings: Settings) -> PineconeVectorStore:
    """Vector store bound to an existing index (used by the query graph)."""
    settings.require_keys()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    return PineconeVectorStore(
        index=pc.Index(settings.index_name),
        embedding=get_embeddings(settings),
        namespace=settings.namespace,
    )


def upsert_in_batches(store: PineconeVectorStore, chunks: List[Document], settings: Settings) -> None:
    """Embed + upsert in small batches with retry/backoff on rate limits (HTTP 429).

    Safe to retry because chunk ids are deterministic (an upsert overwrites).
    """
    size = max(1, settings.embed_batch_size)
    total = len(chunks)
    for start in range(0, total, size):
        batch = chunks[start : start + size]
        ids = [c.metadata["chunk_id"] for c in batch]
        for attempt in range(1, 7):
            try:
                store.add_documents(batch, ids=ids)
                break
            except Exception as exc:
                rate_limited = any(t in str(exc) for t in ("429", "RESOURCE_EXHAUSTED", "quota"))
                if not rate_limited or attempt == 6:
                    raise
                wait = min(60, 2**attempt * 3)
                logger.warning("Rate limited (attempt %d/6). Sleeping %ds...", attempt, wait)
                time.sleep(wait)
        logger.info("Embedded + upserted %d/%d chunks", min(start + size, total), total)
        time.sleep(settings.embed_sleep_seconds)


def run_ingestion(
    pdf_path: Path = PDF_PATH,
    settings: Optional[Settings] = None,
    reset: bool = False,
) -> PineconeVectorStore:
    settings = settings or get_settings()
    settings.require_keys()

    chunks = split_documents(load_pdf(pdf_path), settings)
    if not chunks:
        raise RuntimeError("No text extracted from the PDF (is it a scanned/image-only PDF?).")

    pc = Pinecone(api_key=settings.pinecone_api_key)
    ensure_index(pc, settings)
    index = pc.Index(settings.index_name)

    if reset:
        try:
            index.delete(delete_all=True, namespace=settings.namespace)
            logger.info("Cleared namespace '%s'", settings.namespace)
        except Exception as exc:  # namespace may not exist yet on a fresh index
            logger.info("Nothing to clear (%s)", exc)

    store = PineconeVectorStore(
        index=index, embedding=get_embeddings(settings), namespace=settings.namespace
    )
    upsert_in_batches(store, chunks, settings)
    logger.info(
        "Upserted %d chunks into '%s/%s'", len(chunks), settings.index_name, settings.namespace
    )
    return store


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest the Agentic AI eBook into Pinecone.")
    parser.add_argument("--pdf", type=Path, default=PDF_PATH, help="path to the PDF")
    parser.add_argument("--download", action="store_true", help="download the PDF from Drive first")
    parser.add_argument("--reset", action="store_true", help="clear the namespace before ingesting")
    args = parser.parse_args()

    if args.download or not args.pdf.exists():
        download_pdf(args.pdf)
    run_ingestion(args.pdf, reset=args.reset)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()

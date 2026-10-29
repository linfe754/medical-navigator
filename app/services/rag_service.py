
"""
RAG context retrieval service.

Retrieves candidate chunks using a configurable retrieval backend,
resolves their cleaned parent pages, and assembles grounded context
for the LLM.

Supported retrieval backends:
- bm25: local BM25 retrieval
- hybrid: BM25 + Redis vector retrieval using RRF

When Redis is unavailable in hybrid mode, retrieval falls back to BM25.
"""

import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

from redis.exceptions import (
    ConnectionError as RedisConnectionError,
    TimeoutError as RedisTimeoutError,
)

from app.ingestion.context_builder import build_context
from app.services.rag_retriever import get_rag_retriever
from app.services.rag_hybrid_retriever import get_hybrid_retriever


CLEANED_DIR = Path("data/cleaned")

logger = logging.getLogger(__name__)

RAG_RETRIEVER = os.getenv("RAG_RETRIEVER", "hybrid").strip().lower()

VALID_RETRIEVERS = {"bm25", "hybrid"}


def make_parent_id(
    source_id: str,
    content_hash: str,
    page_number: int,
    text: str,
) -> str:
    """Use the same parent identity as chunk generation."""
    payload = json.dumps(
        [source_id, content_hash, page_number, text],
        ensure_ascii=False,
    )
    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


@lru_cache(maxsize=1)
def load_parent_pages() -> dict[str, dict]:
    """
    Load and validate indexable parent pages.

    Cached for the lifetime of the application process.
    """
    retriever = get_rag_retriever()

    source_hashes = {}

    for chunk in retriever.candidates:
        metadata = chunk["metadata"]
        source_id = metadata["source_id"]
        content_hash = metadata["content_hash"]

        previous_hash = source_hashes.get(source_id)

        if (
            previous_hash is not None
            and previous_hash != content_hash
        ):
            raise ValueError(
                f"Inconsistent content hash: {source_id}"
            )

        source_hashes[source_id] = content_hash

    parents = {}

    for source_id, content_hash in source_hashes.items():
        path = CLEANED_DIR / f"{source_id}.json"

        document = json.loads(
            path.read_text(encoding="utf-8")
        )

        if document["content_hash"] != content_hash:
            raise ValueError(
                f"Stale cleaned document: {source_id}"
            )

        source = document["source"]

        for page in document["pages"]:
            if not page.get("indexable", True):
                continue

            text = page["text"]

            if not text.strip():
                continue

            page_number = page["page_number"]

            parent_id = make_parent_id(
                source_id,
                content_hash,
                page_number,
                text,
            )

            if parent_id in parents:
                raise ValueError(
                    f"Duplicate parent ID: {parent_id}"
                )

            parents[parent_id] = {
                "text": text,
                "metadata": {
                    **source,
                    "page_number": page_number,
                    "page_type": page["page_type"],
                    "content_hash": content_hash,
                    "cleaning_method": document["cleaning_method"],
                },
            }

    return parents


def retrieve_chunks(
    question: str,
    top_k: int = 2,
) -> list[dict]:
    """
    Retrieve candidate chunks using the configured backend.

    Hybrid mode:
        BM25 + Redis vector retrieval with RRF.

    BM25 mode:
        Local lexical retrieval without Redis.

    Redis connection or timeout failures in hybrid mode trigger
    a BM25 fallback. Other exceptions are propagated.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")

    if not question.strip():
        return []

    if RAG_RETRIEVER not in VALID_RETRIEVERS:
        raise ValueError(
            f"Unsupported RAG_RETRIEVER: {RAG_RETRIEVER}. "
            f"Expected one of: {sorted(VALID_RETRIEVERS)}"
        )

    if RAG_RETRIEVER == "bm25":
        return get_rag_retriever().retrieve(
            question,
            top_k=top_k,
        )

    try:
        return get_hybrid_retriever().retrieve(
            question,
            top_k=top_k,
        )

    except (RedisConnectionError, RedisTimeoutError):
        logger.warning(
            "Redis unavailable. Skipping RAG context retrieval.",
            exc_info=True,
        )
        return []


def retrieve_context(
    question: str,
    top_k: int = 2,
    max_characters: int = 12000,
) -> dict:
    """
    Retrieve chunks and expand them to parent-page context.

    Returns the existing build_context() structure:
    {
        "context": str,
        "sources": list[dict],
        "skipped": list,
        "character_count": int
    }
    """
    parents = load_parent_pages()

    hits = retrieve_chunks(
        question,
        top_k=top_k,
    )

    children = []

    for hit in hits:
        metadata = hit["metadata"]
        parent_id = metadata["parent_id"]

        parent = parents.get(parent_id)

        if parent is None:
            raise ValueError(
                f"Missing parent page for chunk "
                f"{hit['chunk_id']}"
            )

        if hit["text"] not in parent["text"]:
            raise ValueError(
                f"Child text does not match parent: "
                f"{hit['chunk_id']}"
            )

        if (
            metadata["content_hash"]
            != parent["metadata"]["content_hash"]
        ):
            raise ValueError(
                f"Content hash mismatch: "
                f"{hit['chunk_id']}"
            )

        children.append({
            **hit,
            "parent_id": parent_id,
        })

    return build_context(
        retrieved_children=children,
        parents_by_id=parents,
        max_characters=max_characters,
    )

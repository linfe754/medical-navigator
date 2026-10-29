
"""Index prepared document chunks in Redis vector search."""

import os
import json
import hashlib
import logging
from pathlib import Path

import numpy as np
import redis
from dotenv import load_dotenv
from redis.commands.search.field import (
    TextField,
    TagField,
    VectorField,
)
from redis.commands.search.index_definition import (
    IndexDefinition,
    IndexType,
)
from redis.exceptions import ResponseError

from app.ingestion.evaluate_retrieval import load_candidates
from app.services.llm_client import get_openai_client


load_dotenv()

logger = logging.getLogger(__name__)

INDEX_NAME = "idx:medical_navigator:chunks"
KEY_PREFIX = "mn:chunk:"
MODEL = "text-embedding-3-small"
DIMENSIONS = 1536


def get_redis_client():
    return redis.Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379"),
        decode_responses=False,
    )



def ensure_index(client):
    """Create the Redis vector index if it does not exist."""
    existing_indexes = {
        name.decode("utf-8") if isinstance(name, bytes) else name
        for name in client.execute_command("FT._LIST")
    }

    if INDEX_NAME in existing_indexes:
        print(f"Redis index already exists: {INDEX_NAME}")
        return

    schema = (
        TextField("text"),
        TagField("source_id"),
        TagField("parent_id"),
        TagField("content_hash"),
        VectorField(
            "embedding",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": DIMENSIONS,
                "DISTANCE_METRIC": "COSINE",
            },
        ),
    )

    client.ft(INDEX_NAME).create_index(
        schema,
        definition=IndexDefinition(
            prefix=[KEY_PREFIX],
            index_type=IndexType.HASH,
        ),
    )

    print(f"Created Redis index: {INDEX_NAME}")



def embedding_vector(text):
    response = get_openai_client().embeddings.create(
        model=MODEL,
        input=text,
    )

    return np.asarray(
        response.data[0].embedding,
        dtype=np.float32,
    ).tobytes()


def index_chunks():
    client = get_redis_client()
    client.ping()

    ensure_index(client)

    candidates = load_candidates()

    inserted = 0
    skipped = 0

    for chunk in candidates:
        chunk_id = chunk["chunk_id"]
        key = f"{KEY_PREFIX}{chunk_id}"

        if client.exists(key):
            skipped += 1
            continue

        metadata = chunk["metadata"]

        vector = embedding_vector(chunk["text"])

        client.hset(
            key,
            mapping={
                "chunk_id": chunk_id,
                "parent_id": metadata["parent_id"],
                "source_id": metadata["source_id"],
                "content_hash": metadata["content_hash"],
                "text": chunk["text"],
                "embedding": vector,
                "embedding_model": MODEL,
            },
        )

        inserted += 1

    print(
        f"Indexed: {inserted}, "
        f"Skipped: {skipped}, "
        f"Total candidates: {len(candidates)}"
    )


if __name__ == "__main__":
    index_chunks()

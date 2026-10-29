
"""Semantic retrieval using Redis vector search."""

import os

import numpy as np
import redis
from dotenv import load_dotenv
from redis.commands.search.query import Query

from app.services.llm_client import get_openai_client


load_dotenv()

INDEX_NAME = "idx:medical_navigator:chunks"
EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSIONS = 1536


class RedisVectorRetriever:
    def __init__(self):
        self.client = redis.Redis.from_url(
            os.getenv(
                "REDIS_URL",
                "redis://localhost:6379",
            ),
            decode_responses=False,
        )

        self.client.ping()

    def embed_query(self, question: str) -> bytes:
        response = get_openai_client().embeddings.create(
            model=EMBEDDING_MODEL,
            input=question,
        )

        vector = np.asarray(
            response.data[0].embedding,
            dtype=np.float32,
        )

        if vector.shape != (DIMENSIONS,):
            raise ValueError(
                f"Unexpected embedding dimensions: {vector.shape}"
            )

        return vector.tobytes()

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
    ) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be positive")

        if not question.strip():
            return []

        vector = self.embed_query(question)

        query = (
            Query(
                f"*=>[KNN {top_k} @embedding $query_vector "
                "AS vector_distance]"
            )
            .sort_by("vector_distance")
            .return_fields(
                "chunk_id",
                "parent_id",
                "source_id",
                "content_hash",
                "text",
                "vector_distance",
            )
            .paging(0, top_k)
            .dialect(2)
        )

        results = self.client.ft(INDEX_NAME).search(
            query,
            query_params={
                "query_vector": vector,
            },
        )

        hits = []

        for document in results.docs:
            def decode(value):
                if isinstance(value, bytes):
                    return value.decode("utf-8")
                return value

            distance = float(
                decode(document.vector_distance)
            )

            hits.append({
                "chunk_id": decode(document.chunk_id),
                "score": 1.0 - distance,
                "text": decode(document.text),
                "metadata": {
                    "parent_id": decode(document.parent_id),
                    "source_id": decode(document.source_id),
                    "content_hash": decode(document.content_hash),
                },
                "vector_distance": distance,
            })

        return hits


def get_vector_retriever():
    return RedisVectorRetriever()

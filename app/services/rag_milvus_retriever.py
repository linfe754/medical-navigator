
from pathlib import Path

from pymilvus import MilvusClient

from app.services.llm_client import get_openai_client

from functools import lru_cache

DB_PATH = Path("data/vector/milvus.db")
COLLECTION_NAME = "medical_navigator_chunks"
EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSIONS = 1536


class MilvusVectorRetriever:
    def __init__(self):
        self.client = MilvusClient(
            uri=str(DB_PATH),
            grpc_options={
                "grpc.keepalive_time_ms": 120000,
                "grpc.keepalive_timeout_ms": 20000,
                "grpc.keepalive_permit_without_calls": 0,
            },
        )
        self.client.load_collection(
            collection_name=COLLECTION_NAME,
        )

    def embed_query(self, question: str) -> list[float]:
        response = get_openai_client().embeddings.create(
            model=EMBEDDING_MODEL,
            input=question,
        )

        vector = response.data[0].embedding

        if len(vector) != DIMENSIONS:
            raise ValueError(
                f"Unexpected embedding dimensions: {len(vector)}"
            )

        return vector

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

        results = self.client.search(
            collection_name=COLLECTION_NAME,
            data=[vector],
            limit=top_k,
            output_fields=[
                "chunk_id",
                "parent_id",
                "source_id",
                "content_hash",
                "text",
            ],
            search_params={"metric_type": "COSINE"},
        )

        hits = []

        for result in results[0]:
            entity = result["entity"]
            score = float(result["distance"])

            hits.append({
                "chunk_id": entity["chunk_id"],
                "score": score,
                "text": entity["text"],
                "metadata": {
                    "parent_id": entity["parent_id"],
                    "source_id": entity["source_id"],
                    "content_hash": entity["content_hash"],
                },
                "vector_distance": 1.0 - score,
            })

        return hits

    def close(self):
        self.client.close()


@lru_cache(maxsize=1)
def get_vector_retriever() -> MilvusVectorRetriever:
    return MilvusVectorRetriever()

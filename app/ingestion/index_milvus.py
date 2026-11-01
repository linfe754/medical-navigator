
import os
from pathlib import Path

from dotenv import load_dotenv
from pymilvus import DataType, MilvusClient

from app.ingestion.evaluate_retrieval import load_candidates
from app.services.llm_client import get_openai_client


load_dotenv()

DB_PATH = Path("data/vector/milvus.db")
COLLECTION_NAME = "medical_navigator_chunks"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 32


def get_milvus_client() -> MilvusClient:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return MilvusClient(uri=str(DB_PATH))


def ensure_collection(client: MilvusClient) -> None:
    if client.has_collection(COLLECTION_NAME):
        client.load_collection(
            collection_name=COLLECTION_NAME,
        )
        return

    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )

    schema.add_field(
        field_name="chunk_id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=128,
    )

    schema.add_field(
        field_name="parent_id",
        datatype=DataType.VARCHAR,
        max_length=128,
    )

    schema.add_field(
        field_name="source_id",
        datatype=DataType.VARCHAR,
        max_length=256,
    )

    schema.add_field(
        field_name="content_hash",
        datatype=DataType.VARCHAR,
        max_length=128,
    )

    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=65535,
    )

    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=EMBEDDING_DIM,
    )

    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="embedding",
        index_type="FLAT",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )


def get_existing_chunk_ids(
    client: MilvusClient,
) -> set[str]:
    rows = client.query(
        collection_name=COLLECTION_NAME,
        filter="",
        output_fields=["chunk_id"],
        limit=16384,
    )
    return {row["chunk_id"] for row in rows}


def index_documents() -> None:
    chunks = load_candidates()
    client = get_milvus_client()

    try:
        ensure_collection(client)

        existing_ids = get_existing_chunk_ids(client)

        pending = [
            chunk
            for chunk in chunks
            if chunk["chunk_id"] not in existing_ids
        ]

        print("Total chunks:", len(chunks))
        print("Already indexed:", len(chunks) - len(pending))
        print("Pending:", len(pending))

        if not pending:
            print("Milvus index is up to date.")
            return

        openai_client = get_openai_client()

        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start:start + BATCH_SIZE]

            response = openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[chunk["text"] for chunk in batch],
            )

            embeddings = [
                item.embedding
                for item in sorted(
                    response.data,
                    key=lambda item: item.index,
                )
            ]

            rows = []

            for chunk, embedding in zip(batch, embeddings):
                metadata = chunk["metadata"]

                rows.append({
                    "chunk_id": chunk["chunk_id"],
                    "parent_id": metadata["parent_id"],
                    "source_id": metadata["source_id"],
                    "content_hash": metadata["content_hash"],
                    "text": chunk["text"],
                    "embedding": embedding,
                })

            result = client.upsert(
                collection_name=COLLECTION_NAME,
                data=rows,
            )

            print(
                f"Indexed {start + len(batch)}/{len(pending)} "
                f"| Upserted: {result['upsert_count']}"
            )

        print(
            "Final Milvus count:",
            client.get_collection_stats(COLLECTION_NAME)["row_count"],
        )

    finally:
        client.close()


if __name__ == "__main__":
    index_documents()

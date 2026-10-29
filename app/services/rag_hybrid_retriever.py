
"""Hybrid BM25 + Redis vector retrieval using RRF."""

from app.services.rag_retriever import get_rag_retriever
from app.services.rag_vector_retriever import get_vector_retriever


BM25_TOP_K = 5
VECTOR_TOP_K = 5
FINAL_TOP_K = 2

MIN_VECTOR_SIMILARITY = 0.50
RRF_K = 60


class HybridRetriever:
    def __init__(self):
        self.bm25 = get_rag_retriever()
        self.vector = get_vector_retriever()
        
        self.chunks_by_id = {
            chunk["chunk_id"]: chunk
            for chunk in self.bm25.candidates
        }

 
    def retrieve(
        self,
        question: str,
        top_k: int = FINAL_TOP_K,
    ) -> list[dict]:
        """Return relevant canonical chunks ranked by RRF."""

        if top_k < 1:
            raise ValueError("top_k must be positive")

        if not question.strip():
            return []

        # Step 1: Retrieve candidates independently.
        bm25_hits = self.bm25.retrieve(
            question,
            top_k=BM25_TOP_K,
        )

        vector_hits = self.vector.retrieve(
            question,
            top_k=VECTOR_TOP_K,
        )

        # Step 2: Apply the vector similarity relevance gate.
        vector_scores = {
            hit["chunk_id"]: hit["score"]
            for hit in vector_hits
            if hit["score"] >= MIN_VECTOR_SIMILARITY
        }

        if not vector_scores:
            return []

        # Step 3: Use canonical chunk text and metadata.
        candidates = {}

        for chunk_id, similarity in vector_scores.items():
            canonical = self.chunks_by_id.get(chunk_id)

            if canonical is None:
                continue

            candidates[chunk_id] = {
                "chunk_id": chunk_id,
                "text": canonical["text"],
                "metadata": canonical["metadata"],
                "score": 0.0,
                "bm25_rank": None,
                "vector_rank": None,
                "vector_similarity": similarity,
            }

        # Step 4: Apply BM25 RRF contributions.
        for rank, hit in enumerate(bm25_hits, start=1):
            chunk_id = hit["chunk_id"]

            if chunk_id in candidates:
                candidates[chunk_id]["score"] += (
                    1.0 / (RRF_K + rank)
                )
                candidates[chunk_id]["bm25_rank"] = rank

        # Step 5: Apply vector RRF contributions.
        for rank, hit in enumerate(vector_hits, start=1):
            chunk_id = hit["chunk_id"]

            if chunk_id in candidates:
                candidates[chunk_id]["score"] += (
                    1.0 / (RRF_K + rank)
                )
                candidates[chunk_id]["vector_rank"] = rank

        # Step 6: Rank and return at most two chunks.
        ranked = sorted(
            candidates.values(),
            key=lambda hit: (
                -hit["score"],
                -hit["vector_similarity"],
                hit["chunk_id"],
            ),
        )

        return ranked[:min(top_k, FINAL_TOP_K)]


def get_hybrid_retriever():
    return HybridRetriever()

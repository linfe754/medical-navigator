
from functools import lru_cache

from rank_bm25 import BM25Okapi

from app.ingestion.evaluate_retrieval import load_candidates, tokenize


class RAGRetriever:
    def __init__(self):
        self.candidates = load_candidates()
        corpus = [tokenize(chunk["text"]) for chunk in self.candidates]

        if any(not tokens for tokens in corpus):
            raise ValueError("A candidate contains no searchable tokens")

        self.index = BM25Okapi(corpus)

    def retrieve(self, question: str, top_k: int = 3) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be positive")

        query_tokens = tokenize(question)

        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)

        ranked = sorted(
            range(len(self.candidates)),
            key=lambda position: float(scores[position]),
            reverse=True,
        )

        hits = []

        for position in ranked:
            score = float(scores[position])

            if score <= 0:
                break

            chunk = self.candidates[position]

            hits.append({
                "chunk_id": chunk["chunk_id"],
                "score": score,
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            })

            if len(hits) >= top_k:
                break

        return hits


@lru_cache(maxsize=1)
def get_rag_retriever() -> RAGRetriever:
    return RAGRetriever()

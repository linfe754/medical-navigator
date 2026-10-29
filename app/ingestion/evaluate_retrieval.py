import hashlib
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.ingestion.prepare_documents import load_sources, save_json


def tokenize(text: str) -> list[str]:
    text = text.casefold().replace("’", "'")

    # English words, numbers, and Chinese character pairs.
    tokens = re.findall(r"[a-z0-9]+", text)

    for sequence in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", text):
        if len(sequence) == 1:
            tokens.append(sequence)
        else:
            tokens.extend(
                sequence[index:index + 2]
                for index in range(len(sequence) - 1)
            )

    return tokens


def load_candidates() -> list[dict]:
    candidates = []
    seen_ids = set()

    for source in load_sources():
        if not source["prepare"]:
            continue

        path = Path("data/chunks") / f"{source['source_id']}.json"
        document = json.loads(path.read_text(encoding="utf-8"))

        current_hash = hashlib.sha256(
            (Path("data/raw") / source["filename"]).read_bytes()
        ).hexdigest()

        for chunk in document["chunks"]:
            metadata = chunk["metadata"]

            if metadata["content_hash"] != current_hash:
                raise ValueError("Stale chunks; rerun preparation and chunking")

            if any(metadata.get(key) != value for key, value in source.items()):
                raise ValueError("Chunk metadata differs from the manifest")

            if chunk["chunk_id"] in seen_ids:
                raise ValueError("Duplicate chunk ID")

            seen_ids.add(chunk["chunk_id"])
            candidates.append(chunk)

    if not candidates:
        raise ValueError("No candidate chunks found")

    return candidates


def main() -> None:
    candidates = load_candidates()
    corpus = [tokenize(chunk["text"]) for chunk in candidates]

    if any(not tokens for tokens in corpus):
        raise ValueError("A candidate contains no searchable tokens")

    index = BM25Okapi(corpus)

    cases_path = Path("evals/retrieval_cases.json")
    cases = json.loads(
        cases_path.read_text(encoding="utf-8")
    )["cases"]

    results = []

    for case in cases:
        scores = index.get_scores(tokenize(case["question"]))

        ranked = sorted(
            range(len(candidates)),
            key=lambda position: float(scores[position]),
            reverse=True,
        )

        hits = []

        for position in ranked:
            if float(scores[position]) <= 0:
                continue

            chunk = candidates[position]

            hits.append({
                "chunk_id": chunk["chunk_id"],
                "score": float(scores[position]),
                "source_id": chunk["metadata"]["source_id"],
                "page_number": chunk["metadata"]["page_number"],
                "text": chunk["text"],
            })

            if len(hits) == 3:
                break

        expected = set(case["expected_source_ids"])
        source_hit = any(
            hit["source_id"] in expected
            for hit in hits
        )

        results.append({
            "case_id": case["id"],
            "question": case["question"],
            "required_evidence": case["required_evidence"],
            "source_hit_at_3": source_hit,
            "evidence_review": "pending",
            "hits": hits,
        })

        print(
            f"{case['id']}: "
            f"source hit@3 = {source_hit}"
        )

    hit_count = sum(result["source_hit_at_3"] for result in results)

    save_json(
        Path("data/reports/retrieval-bm25.json"),
        {
            "schema_version": 1,
            "method": "bm25_english_words_chinese_bigrams_v1",
            "candidate_count": len(candidates),
            "case_count": len(results),
            "source_hits_at_3": hit_count,
            "results": results,
        },
    )

    print(f"\nCandidate chunks: {len(candidates)}")
    print(f"Source hits@3: {hit_count}/{len(results)}")
    print("Report: data/reports/retrieval-bm25.json")
    print("Evidence correctness still requires review.")


if __name__ == "__main__":
    main()

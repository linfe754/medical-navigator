import hashlib
import json
from pathlib import Path

from app.ingestion.context_builder import build_context
from app.ingestion.evaluate_retrieval import load_candidates
from app.ingestion.prepare_documents import save_json


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    retrieval = read_json(
        Path("data/reports/retrieval-bm25.json")
    )
    cases = read_json(
        Path("evals/retrieval_cases.json")
    )["cases"]

    cases_by_id = {case["id"]: case for case in cases}

    # Reuse existing source/hash validation.
    candidates = load_candidates()
    children_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in candidates
    }

    documents = {}
    parents_by_id = {}
    results = []

    for result in retrieval["results"]:
        case_id = result["case_id"]
        case = cases_by_id[case_id]

        if (
            result["question"] != case["question"]
            or result["required_evidence"] != case["required_evidence"]
        ):
            raise ValueError(
                "Evaluation cases changed; rerun BM25 evaluation"
            )

        ranked_children = []

        for hit in result["hits"]:
            child = children_by_id.get(hit["chunk_id"])

            if child is None:
                raise ValueError(
                    "Retrieval references an old chunk; rerun evaluation"
                )

            metadata = child["metadata"]
            source_id = metadata["source_id"]

            if (
                hit["text"] != child["text"]
                or hit["source_id"] != source_id
                or hit["page_number"] != metadata["page_number"]
            ):
                raise ValueError("Retrieval output differs from current chunks")

            if source_id not in documents:
                documents[source_id] = read_json(
                    Path("data/cleaned") / f"{source_id}.json"
                )

            document = documents[source_id]

            if document["content_hash"] != metadata["content_hash"]:
                raise ValueError("Cleaned document and chunk hashes differ")

            page_number = metadata["page_number"]
            page = next(
                page
                for page in document["pages"]
                if page["page_number"] == page_number
            )

            if child["text"] not in page["text"]:
                raise ValueError("Child text no longer matches its parent")

            identity = json.dumps(
                [source_id, document["content_hash"],
                 page_number, page["text"]],
                ensure_ascii=False,
            )
            parent_id = hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()

            parents_by_id[parent_id] = {
                "parent_id": parent_id,
                "text": page["text"],
                "metadata": {
                    **document["source"],
                    "page_number": page_number,
                    "content_hash": document["content_hash"],
                },
            }

            ranked_children.append({
                **child,
                "parent_id": parent_id,
            })

        assembled = build_context(
            ranked_children,
            parents_by_id,
            max_characters=12000,
        )

        expected_sources = set(case["expected_source_ids"])

        expected_source_in_context = any(
            source["source_id"] in expected_sources
            for source in assembled["sources"]
        )

        results.append({
            "case_id": case_id,
            "question": case["question"],
            "required_evidence": case["required_evidence"],
            "child_hit_count": len(ranked_children),
            "expected_source_in_context": expected_source_in_context,
            "evidence_review": "pending",
            **assembled,
        })

        print(
            f"{case_id}: "
            f"{len(ranked_children)} hits → "
            f"{len(assembled['sources'])} parents | "
            f"{assembled['character_count']} characters | "
            f"{len(assembled['skipped'])} skipped"
        )

    output_path = Path("data/reports/context-bm25.json")

    save_json(output_path, {
        "schema_version": 1,
        "retrieval_method": retrieval["method"],
        "context_method": "deduplicated_page_parents_v1",
        "max_characters": 12000,
        "results": results,
    })

    print(f"\nSaved: {output_path}")
    print("No answers generated. Evidence reviews remain pending.")


if __name__ == "__main__":
    main()

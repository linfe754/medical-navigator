
"""
Generate retrieval chunks from cleaned PDF documents.

- Skip pages excluded during cleaning.
- Preserve source and page provenance.
- Generate deterministic parent and chunk IDs.
- Carry cleaning and quality metadata into chunks.
- Report excluded pages and review warnings.

The output is retrieval-backend independent.
"""

import hashlib
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.prepare_documents import load_sources, save_json


DATA_DIR = Path("data")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

CHUNKING_METHOD = "recursive_characters_v2"
PIPELINE_VERSION = "2.0"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "。\n",
        "。",
        "！",
        "？",
        ". ",
        "! ",
        "? ",
        "\n",
        " ",
        "",
    ],
    keep_separator="end",
    strip_whitespace=True,
)


def make_id(parts: list) -> str:
    """Generate a deterministic SHA256 identifier."""
    identity = json.dumps(
        parts,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()


def make_parent_id(
    source_id: str,
    content_hash: str,
    page_number: int,
    text: str,
) -> str:
    """
    Identify the exact cleaned parent page.

    Keep the existing rag_service.py identity formula.
    """
    identity = json.dumps(
        [
            source_id,
            content_hash,
            page_number,
            text,
        ],
        ensure_ascii=False,
    )
    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()


def make_chunks(
    source: dict,
) -> tuple[list[dict], list[dict]]:
    source_id = source["source_id"]

    cleaned_path = (
        DATA_DIR / "cleaned" / f"{source_id}.json"
    )

    document = json.loads(
        cleaned_path.read_text(encoding="utf-8")
    )

    if document.get("schema_version") != 2:
        raise ValueError(
            "Unsupported cleaned-document schema; "
            "rerun preparation"
        )

    current_hash = hashlib.sha256(
        (DATA_DIR / "raw" / source["filename"]).read_bytes()
    ).hexdigest()

    if document["content_hash"] != current_hash:
        raise ValueError(
            "Source PDF changed; rerun preparation"
        )

    if document["source"] != source:
        raise ValueError(
            "Source metadata changed; rerun preparation"
        )

    chunks = []
    warnings = []
    excluded_pages = []

    cleaning_method = document["cleaning_method"]

    for page in document["pages"]:
        page_number = page["page_number"]
        text = page["text"]

        if not page["indexable"]:
            excluded_pages.append({
                "page_number": page_number,
                "page_type": page["page_type"],
                "reason": page["exclusion_reason"],
            })
            continue

        quality_flags = sorted(set(
            page["quality"]["flags"]
            + page.get("warnings", [])
        ))

        if quality_flags:
            warnings.append({
                "page_number": page_number,
                "flags": quality_flags,
            })

        if not text.strip():
            continue

        parent_id = make_parent_id(
            source_id=source_id,
            content_hash=current_hash,
            page_number=page_number,
            text=text,
        )

        pieces = splitter.split_text(text)

        for page_chunk_index, piece in enumerate(pieces):
            chunk_id = make_id([
                source_id,
                current_hash,
                cleaning_method,
                CHUNKING_METHOD,
                CHUNK_SIZE,
                CHUNK_OVERLAP,
                page_number,
                page_chunk_index,
                piece,
            ])

            metadata = {
                **source,
                "content_hash": current_hash,
                "pipeline_version": PIPELINE_VERSION,
                "cleaning_method": cleaning_method,
                "chunking_method": CHUNKING_METHOD,
                "parent_id": parent_id,
                "page_number": page_number,
                "page_type": page["page_type"],
                "page_chunk_index": page_chunk_index,
                "indexable": True,
                "character_count": len(piece),
                "extraction_flags": quality_flags,
                "chunk_review_status": "pending",
            }

            chunks.append({
                "chunk_id": chunk_id,
                "parent_id": parent_id,
                "text": piece,
                "metadata": metadata,
            })

    # Fail on duplicate chunk identities.
    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            f"Duplicate chunk IDs in {source_id}"
        )

    return chunks, {
        "page_warnings": warnings,
        "excluded_pages": excluded_pages,
    }


def main() -> None:
    sources = load_sources()
    results = []

    for source in sources:
        if not source["prepare"]:
            continue

        source_id = source["source_id"]

        try:
            chunks, details = make_chunks(source)

            output_path = (
                DATA_DIR / "chunks" / f"{source_id}.json"
            )

            save_json(output_path, {
                "schema_version": 2,
                "source_id": source_id,
                "content_hash": (
                    chunks[0]["metadata"]["content_hash"]
                    if chunks else None
                ),
                "pipeline_version": PIPELINE_VERSION,
                "chunking_method": CHUNKING_METHOD,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "chunks": chunks,
            })

            result = {
                "source_id": source_id,
                "status": "chunked",
                "chunk_count": len(chunks),
                "excluded_page_count": len(
                    details["excluded_pages"]
                ),
                "excluded_pages": details["excluded_pages"],
                "page_warnings": details["page_warnings"],
            }

            print(
                f"{source_id}: {len(chunks)} chunks, "
                f"{len(details['excluded_pages'])} "
                "pages excluded"
            )

        except Exception as error:
            result = {
                "source_id": source_id,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }

            print(f"{source_id}: FAILED — {error}")

        results.append(result)

    save_json(
        DATA_DIR / "reports" / "chunking-report.json",
        {
            "schema_version": 2,
            "pipeline_version": PIPELINE_VERSION,
            "results": results,
        },
    )

    print("\nChunking report saved.")

    if any(
        result["status"] == "failed"
        for result in results
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

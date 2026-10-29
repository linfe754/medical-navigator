import argparse
import hashlib
import json
from pathlib import Path

from app.ingestion.prepare_documents import save_json


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_id")
    parser.add_argument(
        "--chunk",
        type=int,
        default=1,
        help="Child chunk to display, numbered from 1",
    )
    args = parser.parse_args()

    # Validate the ID before using it in paths.
    if not args.source_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
        for character in args.source_id
    ):
        raise ValueError("Invalid source ID")

    cleaned = load_json(
        Path("data/cleaned") / f"{args.source_id}.json"
    )
    chunk_document = load_json(
        Path("data/chunks") / f"{args.source_id}.json"
    )

    chunks = chunk_document["chunks"]

    if not 1 <= args.chunk <= len(chunks):
        raise ValueError(
            f"Choose a chunk from 1 to {len(chunks)}"
        )

    parents = {}
    children = []

    for chunk in chunks:
        metadata = chunk["metadata"]

        if metadata["content_hash"] != cleaned["content_hash"]:
            raise ValueError("Chunks and cleaned document versions differ")

        page_number = metadata["page_number"]

        page = next(
            page
            for page in cleaned["pages"]
            if page["page_number"] == page_number
        )

        # Confirm the child belongs to this saved page.
        if chunk["text"] not in page["text"]:
            raise ValueError(
                "Child text does not match its cleaned page; "
                "check whether outputs are stale"
            )

        parent_identity = (
            f"{args.source_id}:{cleaned['content_hash']}:"
            f"{page_number}:{page['text']}"
        )

        parent_id = hashlib.sha256(
            parent_identity.encode("utf-8")
        ).hexdigest()

        parents[parent_id] = {
            "parent_id": parent_id,
            "text": page["text"],
            "metadata": {
                **cleaned["source"],
                "page_number": page_number,
                "content_hash": cleaned["content_hash"],
                "context_type": "full_page",
            },
        }

        children.append({
            **chunk,
            "parent_id": parent_id,
        })

    output = {
        "schema_version": 1,
        "experiment": "page_parent_context_v1",
        "source_id": args.source_id,
        "parents": list(parents.values()),
        "children": children,
    }

    output_path = (
        Path("data/experiments/parent-context")
        / f"{args.source_id}.json"
    )
    save_json(output_path, output)

    selected_child = children[args.chunk - 1]
    selected_parent = parents[selected_child["parent_id"]]

    print("--- Child alone ---\n")
    print(selected_child["text"])

    print("\n--- Parent context ---\n")
    print(selected_parent["text"])

    print("\n--- Comparison ---")
    print(f"Child characters: {len(selected_child['text'])}")
    print(f"Parent characters: {len(selected_parent['text'])}")
    print(f"Saved: {output_path}")
    print("Context expansion only; no search or embedding performed.")


if __name__ == "__main__":
    main()

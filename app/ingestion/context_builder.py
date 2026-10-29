def build_context(
    retrieved_children: list[dict],
    parents_by_id: dict[str, dict],
    max_characters: int = 12000,
) -> dict:
    """Expand ranked children into unique, complete parent blocks."""
    if max_characters <= 0:
        raise ValueError("max_characters must be positive")

    blocks = []
    sources = []
    skipped = []
    seen = set()
    used_characters = 0

    for child in retrieved_children:
        parent_id = child["parent_id"]

        # Several search hits can point to the same parent.
        if parent_id in seen:
            continue

        seen.add(parent_id)
        parent = parents_by_id.get(parent_id)

        if parent is None:
            skipped.append({
                "parent_id": parent_id,
                "reason": "parent_not_found",
            })
            continue

        text = parent["text"].strip()

        if not text:
            skipped.append({
                "parent_id": parent_id,
                "reason": "empty_parent",
            })
            continue

        metadata = parent["metadata"]
        citation_id = f"S{len(sources) + 1}"

        block = (
            f"[{citation_id}]\n"
            f"Title: {metadata['title']}\n"
            f"Page: {metadata['page_number']}\n"
            f"URL: {metadata['source_url']}\n\n"
            f"{text}"
        )

        # Include separators in the budget calculation.
        extra_characters = len(block) + (2 if blocks else 0)

        if used_characters + extra_characters > max_characters:
            skipped.append({
                "parent_id": parent_id,
                "reason": "context_budget_exceeded",
                "block_characters": len(block),
            })
            continue

        blocks.append(block)
        used_characters += extra_characters

        sources.append({
            "citation_id": citation_id,
            "parent_id": parent_id,
            "source_id": metadata["source_id"],
            "title": metadata["title"],
            "source_url": metadata["source_url"],
            "page_number": metadata["page_number"],
            "content_hash": metadata["content_hash"],
        })

    return {
        "context": "\n\n".join(blocks),
        "sources": sources,
        "skipped": skipped,
        "character_count": used_characters,
    }

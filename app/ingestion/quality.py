def check_page_text(text: str) -> dict:
    """Return basic extraction signals, not a guarantee of correctness."""
    visible_characters = sum(
        not character.isspace()
        for character in text
    )

    replacement_characters = text.count("\ufffd")

    flags = []

    if visible_characters == 0:
        flags.append("no_extracted_text")
    elif visible_characters < 100:
        flags.append("little_extracted_text")

    if replacement_characters:
        flags.append("replacement_characters")

    return {
        "visible_characters": visible_characters,
        "replacement_characters": replacement_characters,
        "needs_review": bool(flags),
        "flags": flags,
    }

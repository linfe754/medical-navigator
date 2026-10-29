
"""
Reusable PDF document cleaning and page classification.

Responsibilities:
- Normalise extracted text.
- Detect repeated headers and footers.
- Remove standalone page numbers and page-count markers.
- Identify cover-only and table-of-contents-only pages.
- Flag low-content pages for review.
- Preserve cleaning decisions and original page text.

This module does not perform PDF extraction or chunking.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any


CLEANING_METHOD = "document_cleaning_v2"


@dataclass(frozen=True)
class CleaningConfig:
    # Page furniture detection
    remove_page_numbers: bool = True
    remove_repeated_headers: bool = True
    remove_repeated_footers: bool = True

    # Only inspect these lines for repeated page furniture.
    margin_lines: int = 3
    repetition_ratio: float = 0.6
    min_repeated_pages: int = 3

    # Content classification
    detect_cover_pages: bool = True
    detect_table_of_contents: bool = True

    # A warning threshold, not an exclusion threshold.
    short_page_characters: int = 100

    # Exclude only high-confidence non-substantive pages.
    exclude_cover_only: bool = True
    exclude_toc_only: bool = True

    # Limit cover classification to the first N pages.
    cover_search_pages: int = 2


PAGE_NUMBER_PATTERNS = (
    re.compile(r"^\d{1,4}$"),
    re.compile(
        r"^(?:page|p\.?)\s*\d{1,4}"
        r"(?:\s*(?:of|/)\s*\d{1,4})?$",
        re.IGNORECASE,
    ),
    re.compile(r"^第\s*\d{1,4}\s*页$"),
    re.compile(r"^第\s*\d{1,4}\s*页\s*共\s*\d{1,4}\s*页$"),
    re.compile(r"^[-–—]\s*\d{1,4}\s*[-–—]$"),
)

TOC_HEADING_PATTERN = re.compile(
    r"^(?:contents|table of contents|"
    r"what(?:'|’)?s in this (?:document|fact sheet|guide)\??|"
    r"目录|目\s*录)\s*[?？]?$",
    re.IGNORECASE,
)

TOC_ENTRY_PATTERNS = (
    re.compile(r"^.{3,100}\.{2,}\s*\d{1,4}$"),
    re.compile(r"^.{3,100}\s+\d{1,4}$"),
)

SENTENCE_ENDINGS = (".", "!", "?", "。", "！", "？")


def clean_page_text(text: str) -> str:
    """Normalise whitespace without changing document structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ")

    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.split("\n")
    ]

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _nonempty_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


def _normalise_repeated_line(line: str) -> str:
    """Normalise a margin line for cross-page comparison."""
    line = line.casefold()
    line = re.sub(r"\s+", " ", line).strip()

    # Repeated page-number patterns are handled separately.
    return line


def _is_page_number(line: str) -> bool:
    return any(
        pattern.fullmatch(line.strip())
        for pattern in PAGE_NUMBER_PATTERNS
    )


def _find_repeated_margins(
    pages: list[dict[str, Any]],
    config: CleaningConfig,
) -> tuple[set[str], set[str]]:
    """
    Find lines repeated near page tops and bottoms.

    Count each candidate at most once per page.
    """
    page_count = len(pages)

    if page_count < config.min_repeated_pages:
        return set(), set()

    header_counts: Counter[str] = Counter()
    footer_counts: Counter[str] = Counter()

    for page in pages:
        lines = _nonempty_lines(page["text"])

        headers = {
            _normalise_repeated_line(line)
            for line in lines[:config.margin_lines]
            if not _is_page_number(line)
        }

        footers = {
            _normalise_repeated_line(line)
            for line in lines[-config.margin_lines:]
            if not _is_page_number(line)
        }

        header_counts.update(headers)
        footer_counts.update(footers)

    minimum = max(
        config.min_repeated_pages,
        int(page_count * config.repetition_ratio + 0.999999),
    )

    repeated_headers = {
        line
        for line, count in header_counts.items()
        if count >= minimum
    }

    repeated_footers = {
        line
        for line, count in footer_counts.items()
        if count >= minimum
    }

    return repeated_headers, repeated_footers


def _remove_page_furniture(
    text: str,
    repeated_headers: set[str],
    repeated_footers: set[str],
    config: CleaningConfig,
) -> tuple[str, list[dict[str, str]]]:
    """Remove recognised page furniture only in margin regions."""
    lines = _nonempty_lines(text)
    removed: list[dict[str, str]] = []
    retained: list[str] = []

    total = len(lines)

    for index, line in enumerate(lines):
        near_top = index < config.margin_lines
        near_bottom = index >= total - config.margin_lines

        normalised = _normalise_repeated_line(line)
        reason = None

        if (
            config.remove_page_numbers
            and (near_top or near_bottom)
            and _is_page_number(line)
        ):
            reason = "page_number"

        elif (
            config.remove_repeated_headers
            and near_top
            and normalised in repeated_headers
        ):
            reason = "repeated_header"

        elif (
            config.remove_repeated_footers
            and near_bottom
            and normalised in repeated_footers
        ):
            reason = "repeated_footer"

        if reason:
            removed.append({
                "text": line,
                "reason": reason,
            })
        else:
            retained.append(line)

    return clean_page_text("\n".join(retained)), removed


def _is_toc_entry(line: str) -> bool:
    return any(
        pattern.fullmatch(line)
        for pattern in TOC_ENTRY_PATTERNS
    )


def _classify_toc(text: str) -> bool:
    """
    Identify TOC-only pages using an explicit heading
    and multiple section/page references.
    """
    lines = _nonempty_lines(text)

    if not lines:
        return False

    has_heading = any(
        TOC_HEADING_PATTERN.fullmatch(line)
        for line in lines[:5]
    )

    if not has_heading:
        return False

    entries = [
        line for line in lines
        if _is_toc_entry(line)
    ]

    return len(entries) >= 3


def _classify_cover(
    text: str,
    page_index: int,
    config: CleaningConfig,
) -> bool:
    """
    Identify high-confidence cover-only pages.

    Restrict detection to the beginning of a document.
    Do not treat question marks in titles as prose.
    """
    if page_index >= config.cover_search_pages:
        return False

    lines = _nonempty_lines(text)

    if not lines:
        return False

    if len(text) > 250 or len(lines) > 6:
        return False

    # A period is stronger evidence of prose than a
    # question mark, which commonly appears in titles.
    prose_lines = sum(
        line.endswith((".", "。", "!", "！"))
        for line in lines
    )

    return prose_lines == 0


def _classify_page(
    text: str,
    page_index: int,
    config: CleaningConfig,
) -> tuple[str, bool, str | None, list[str]]:
    """
    Return page_type, indexable, exclusion_reason, warnings.
    """
    warnings: list[str] = []

    if not text.strip():
        return "empty", False, "empty_page", warnings

    if len(text) < config.short_page_characters:
        warnings.append("short_page")

    if (
        config.detect_table_of_contents
        and _classify_toc(text)
    ):
        if config.exclude_toc_only:
            return (
                "table_of_contents",
                False,
                "toc_only",
                warnings,
            )

        return "table_of_contents", True, None, warnings

    if (
        config.detect_cover_pages
        and _classify_cover(text, page_index, config)
    ):
        if config.exclude_cover_only:
            return "cover", False, "cover_only", warnings

        return "cover", True, None, warnings

    return "main_content", True, None, warnings


def clean_document_pages(
    pages: list[dict[str, Any]],
    config: CleaningConfig | None = None,
) -> dict[str, Any]:
    """
    Clean and classify all pages in one document.

    Input:
        [{"page_number": 1, "text": "..."}, ...]

    Output:
        {
            "cleaning_method": "...",
            "config": {...},
            "pages": [...],
            "summary": {...}
        }
    """
    config = config or CleaningConfig()

    normalised_pages = [
        {
            "page_number": page["page_number"],
            "text": clean_page_text(page["text"]),
        }
        for page in pages
    ]

    repeated_headers, repeated_footers = (
        _find_repeated_margins(normalised_pages, config)
    )

    cleaned_pages = []
    excluded_by_reason: Counter[str] = Counter()
    removed_by_reason: Counter[str] = Counter()

    for page_index, page in enumerate(normalised_pages):
        cleaned_text, removed_lines = _remove_page_furniture(
            text=page["text"],
            repeated_headers=repeated_headers,
            repeated_footers=repeated_footers,
            config=config,
        )

        (
            page_type,
            indexable,
            exclusion_reason,
            warnings,
        ) = _classify_page(
            cleaned_text,
            page_index,
            config,
        )

        for item in removed_lines:
            removed_by_reason[item["reason"]] += 1

        if exclusion_reason:
            excluded_by_reason[exclusion_reason] += 1

        cleaned_pages.append({
            "page_number": page["page_number"],
            "original_text": page["text"],
            "text": cleaned_text,
            "page_type": page_type,
            "indexable": indexable,
            "exclusion_reason": exclusion_reason,
            "warnings": warnings,
            "removed_lines": removed_lines,
        })

    summary = {
        "total_pages": len(cleaned_pages),
        "indexable_pages": sum(
            page["indexable"]
            for page in cleaned_pages
        ),
        "excluded_pages": sum(
            not page["indexable"]
            for page in cleaned_pages
        ),
        "excluded_by_reason": dict(excluded_by_reason),
        "removed_by_reason": dict(removed_by_reason),
        "pages_with_warnings": sum(
            bool(page["warnings"])
            for page in cleaned_pages
        ),
    }

    return {
        "cleaning_method": CLEANING_METHOD,
        "config": asdict(config),
        "pages": cleaned_pages,
        "summary": summary,
    }

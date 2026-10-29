
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from app.ingestion.cleaning import clean_document_pages
from app.ingestion.extract_pdf import extract_pdf
from app.ingestion.quality import check_page_text


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
MANIFEST_PATH = DATA_DIR / "sources.json"


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_sources() -> list[dict]:
    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema version")

    sources = manifest["sources"]
    seen_ids = set()
    seen_filenames = set()

    for source in sources:
        source_id = source["source_id"]
        filename = source["filename"]

        if not source_id or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
            for character in source_id
        ):
            raise ValueError(f"Invalid source_id: {source_id!r}")

        if (
            "/" in filename
            or "\\" in filename
            or not filename.lower().endswith(".pdf")
        ):
            raise ValueError(f"Invalid PDF filename: {filename!r}")

        if source_id in seen_ids:
            raise ValueError(f"Duplicate source_id: {source_id}")

        if filename in seen_filenames:
            raise ValueError(f"Duplicate filename: {filename}")

        if not isinstance(source.get("prepare"), bool):
            raise ValueError(f"Missing boolean prepare: {source_id}")

        seen_ids.add(source_id)
        seen_filenames.add(filename)

    return sources


def prepare_source(source: dict) -> dict:
    path = RAW_DIR / source["filename"]

    if not path.resolve().is_relative_to(RAW_DIR.resolve()):
        raise ValueError("Source resolves outside data/raw")

    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    extracted = extract_pdf(path)

    document = {
        "schema_version": 1,
        "source": source,
        "content_hash": content_hash,
        "extraction_method": extracted["extraction_method"],
        "page_count": extracted["page_count"],
        "pages": extracted["pages"],
    }

    source_id = source["source_id"]

    # Preserve the original extraction.
    save_json(
        DATA_DIR / "extracted" / f"{source_id}.json",
        document,
    )

    # Run document-level cleaning and classification.
    cleaning_result = clean_document_pages(
        document["pages"]
    )

    cleaned_pages = []

    for page in cleaning_result["pages"]:
        quality = check_page_text(page["text"])

        cleaned_pages.append({
            **page,
            "quality": quality,
        })

    cleaned = {
        **document,
        "schema_version": 2,
        "cleaning_method": cleaning_result["cleaning_method"],
        "cleaning_config": cleaning_result["config"],
        "cleaning_summary": cleaning_result["summary"],
        "pages": cleaned_pages,
    }

    save_json(
        DATA_DIR / "cleaned" / f"{source_id}.json",
        cleaned,
    )

    flagged_pages = []

    for page in cleaned_pages:
        flags = list(page["quality"]["flags"])
        flags.extend(page["warnings"])

        if flags or not page["indexable"]:
            flagged_pages.append({
                "page_number": page["page_number"],
                "page_type": page["page_type"],
                "indexable": page["indexable"],
                "exclusion_reason": page["exclusion_reason"],
                "flags": sorted(set(flags)),
                "removed_lines": page["removed_lines"],
            })

    summary = cleaning_result["summary"]

    return {
        "source_id": source_id,
        "filename": source["filename"],
        "status": (
            "needs_review"
            if flagged_pages
            else "prepared"
        ),
        "content_hash": content_hash,
        "page_count": document["page_count"],
        "indexable_pages": summary["indexable_pages"],
        "excluded_pages": summary["excluded_pages"],
        "excluded_by_reason": summary["excluded_by_reason"],
        "removed_by_reason": summary["removed_by_reason"],
        "flagged_pages": flagged_pages,
        "review_notes": source.get("review_notes", []),
    }


def main() -> None:
    sources = load_sources()
    results = []

    for source in sources:
        source_id = source["source_id"]

        if not source["prepare"]:
            results.append({
                "source_id": source_id,
                "status": "skipped",
            })
            continue

        print(f"Preparing: {source_id}", flush=True)

        try:
            result = prepare_source(source)
        except Exception as error:
            result = {
                "source_id": source_id,
                "filename": source["filename"],
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }

        results.append(result)
        print(f"  {result['status']}")

    report = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(MANIFEST_PATH),
        "results": results,
    }

    report_path = (
        DATA_DIR / "reports" / "preparation-report.json"
    )
    save_json(report_path, report)

    print("\n--- Summary ---")

    for status in (
        "prepared",
        "needs_review",
        "failed",
        "skipped",
    ):
        count = sum(
            result["status"] == status
            for result in results
        )
        print(f"{status}: {count}")

    print(f"Report: {report_path}")

    if any(
        result["status"] == "failed"
        for result in results
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

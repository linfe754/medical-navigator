import argparse
import json
import re
import unicodedata
from pathlib import Path

from app.ingestion.prepare_documents import save_json


# Evaluation fixtures: expected passages, not extraction rules.
EXPECTED_PASSAGES = {
    "bulk-billing-en": {
        "provider_choice": "Not all GPs bulk bill all the time",
        "consent": "signing an Assignment of Benefit form",
        "refund_condition": "If a Medicare benefit applies",
    },
    "mental-health-navigation-en": {
        "card_access": (
            "You can access services with or without a Medicare card"
        ),
        "referral": (
            "You don't need a referral or a mental health treatment plan"
        ),
        "other_provider_fees": (
            "Other services located in the same building"
        ),
    },
    "urgent-care-selection-en": {
        "urgent_qualifier": "but is not life-threatening",
        "general_care": (
            "can wait for an appointment with your local GP"
        ),
        "emergency_instruction": "call triple zero (000)",
        "clinic_finder_url": "health.gov.au/MedicareUCC",
    },
}


def normalize_for_comparison(text: str) -> str:
    """Ignore selected formatting differences, not words or numbers."""
    text = unicodedata.normalize("NFKC", text)

    text = text.translate(str.maketrans({
        "’": "'",
        "‘": "'",
        "–": "-",
        "—": "-",
        "\u00ad": "",
    }))

    return re.sub(r"\s+", "", text).casefold()


def read_output(result: dict) -> str:
    output_dir = Path(result["output_directory"])

    if result["parser"] == "pypdf":
        return (output_dir / "document.txt").read_text(
            encoding="utf-8"
        )

    if result["parser"] == "docling":
        return (output_dir / "document.md").read_text(
            encoding="utf-8"
        )

    raise ValueError(f"Unknown parser: {result['parser']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()

    benchmark = json.loads(
        args.report.read_text(encoding="utf-8")
    )

    evaluations = []

    for result in benchmark["results"]:
        if result["status"] != "completed":
            continue

        source_id = result["source_id"]
        expected = EXPECTED_PASSAGES.get(source_id)

        if expected is None:
            continue

        text = read_output(result)
        normalized_text = normalize_for_comparison(text)

        checks = []

        for check_id, passage in expected.items():
            checks.append({
                "check_id": check_id,
                "expected_passage": passage,
                "found": (
                    normalize_for_comparison(passage)
                    in normalized_text
                ),
            })

        evaluations.append({
            "source_id": source_id,
            "parser": result["parser"],
            "checks": checks,
            "structural_review": "pending",
        })

        found_count = sum(check["found"] for check in checks)

        print(
            f"{result['parser']} | {source_id} | "
            f"passages found: {found_count}/{len(checks)}"
        )

        for check in checks:
            if not check["found"]:
                print(f"  CHECK: {check['check_id']}")

    output_path = args.report.parent / "evaluation.json"

    save_json(output_path, {
        "schema_version": 1,
        "benchmark_run_id": benchmark["run_id"],
        "scope": "Selected passage presence only",
        "results": evaluations,
    })

    print(f"\nSaved: {output_path}")
    print("Reading order and associations still require review.")


if __name__ == "__main__":
    main()

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from app.ingestion.prepare_documents import save_json


def read_json(path: Path) -> tuple[dict, str]:
    content = path.read_bytes()

    return (
        json.loads(content),
        hashlib.sha256(content).hexdigest(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()

    report, report_hash = read_json(args.report)
    rubric_path = Path("evals/answer_rubric.json")
    rubric, rubric_hash = read_json(rubric_path)

    review_path = args.report.parent / "review.json"

    if review_path.exists():
        raise FileExistsError(
            f"Review already exists: {review_path}"
        )

    reviews = []

    for result in report["results"]:
        reviews.append({
            "case_id": result["case_id"],
            "ratings": {
                criterion["id"]: {
                    "rating": "uncertain",
                    "notes": "",
                }
                for criterion in rubric["criteria"]
            },
            "review_complete": False,
        })

    save_json(review_path, {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "answer_report": str(args.report),
        "answer_report_hash": report_hash,
        "rubric_path": str(rubric_path),
        "rubric_hash": rubric_hash,
        "review_method": "manual",
        "reviewer": "",
        "reviews": reviews,
    })

    print(f"Created: {review_path}")
    print("All ratings start as uncertain; no automatic pass assigned.")


if __name__ == "__main__":
    main()

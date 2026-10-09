
import re
from collections import Counter

from app.main import ChatRequest, chat
from evals.evaluators.safety import (
    load_cases,
    check_required_text,
    check_forbidden_text,
    detect_clinical_follow_up,
)


def check_language(response: str, language: str) -> bool:
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", response))
    latin_words = len(re.findall(r"[A-Za-z]+", response))

    if language == "zh":
        return chinese_chars >= 8

    return chinese_chars == 0


def evaluate_case(case) -> dict:
    result = chat(
        ChatRequest(
            message=case.input,
            session_id=f"product-eval-{case.id}",
        )
    )

    response = result["response"]
    tags = set(case.tags)

    violations = []

    missing = check_required_text(
        response, case.expected.required_text
    )
    forbidden = check_forbidden_text(
        response, case.expected.forbidden_text
    )

    if missing:
        violations.append(f"missing_required: {missing}")

    if forbidden:
        violations.append(f"forbidden_text: {forbidden}")

    if (
        "clinical_follow_up" in case.expected.forbidden_behaviours
        and detect_clinical_follow_up(response)
    ):
        violations.append("clinical_follow_up")

    if "language" in tags and not check_language(
        response, case.language
    ):
        violations.append("language_mismatch")

    if violations:
        status = "FAIL"
    elif tags & {"gp_first", "clarification"}:
        status = "REVIEW"
    else:
        status = "PASS"

    return {
        "id": case.id,
        "tags": case.tags,
        "language": case.language,
        "status": status,
        "violations": violations,
        "response": response,
    }


def run(path: str) -> list[dict]:
    cases = load_cases(path)
    return [evaluate_case(case) for case in cases]


def print_summary(results: list[dict]) -> None:
    counts = Counter(result["status"] for result in results)

    print("\nProduct Contract Evaluation")
    print("=" * 40)
    print(f"PASS:   {counts['PASS']}")
    print(f"FAIL:   {counts['FAIL']}")
    print(f"REVIEW: {counts['REVIEW']}")
    print(f"TOTAL:  {len(results)}")

    for result in results:
        if result["status"] == "PASS":
            continue

        print(f"\n{result['id']} [{result['status']}]")
        print(f"Violations: {result['violations']}")
        print(f"Response:\n{result['response']}")


if __name__ == "__main__":
    results = run("evals/datasets/product_contract.jsonl")
    print_summary(results)

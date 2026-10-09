import json
from collections import Counter
from pathlib import Path

from app.main import get_openai_client
from app.router import classify_intent
from evals.schemas.eval_case import EvalCase
from datetime import datetime, timezone


def build_report(results: list[dict]) -> dict:
    passed = sum(result["passed"] for result in results)
    total = len(results)

    intent_metrics = calculate_intent_metrics(results)
    language_metrics = calculate_group_accuracy(results, "language")
    difficulty_metrics = calculate_group_accuracy(results, "difficulty")
    safety_metrics = calculate_safety_metrics(results)
    source_metrics = calculate_source_metrics(results)

    failures = [
        result
        for result in results
        if not result["passed"]
    ]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "passed": passed,
        "accuracy": passed / total if total else 0.0,
        "intent_metrics": intent_metrics,
        "language_metrics": language_metrics,
        "difficulty_metrics": difficulty_metrics,
        "safety_metrics": safety_metrics,
        "source_metrics": source_metrics,
        "llm_avoidance_rate": (
            sum(result["source"] == "regex" for result in results) / total
            if total else 0.0
        ),
        "failures": failures,
        "results": results,
    }
    
def save_report(
    report: dict,
    directory: str = "evals/reports",
) -> Path:
    report_dir = Path(directory)
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = report_dir / f"routing_{timestamp}.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return path

def load_cases(path: str) -> list[EvalCase]:
    cases = []

    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            cases.append(EvalCase.model_validate(json.loads(line)))

    return cases


def evaluate_case(case: EvalCase) -> dict:
    intent, source = classify_intent(
        case.input,
        get_openai_client,
    )

    actual = intent.value
    expected = case.expected.intent

    return {
        "id": case.id,
        "language": case.language,
        "difficulty": case.difficulty,
        "severity": case.severity,
        "expected": expected,
        "actual": actual,
        "source": source,
        "passed": actual == expected,
    }


def run(path: str) -> list[dict]:
    cases = load_cases(path)
    return [evaluate_case(case) for case in cases]


def print_summary(results: list[dict]) -> None:
    passed = sum(result["passed"] for result in results)
    total = len(results)

    print(f"\nRouting Evaluation")
    print(f"{'=' * 40}")
    print(f"Passed:   {passed}/{total}")
    print(f"Accuracy: {passed / total:.1%}")
    
    metrics = calculate_intent_metrics(results)

    print("\nPer-intent metrics:")
    print(
        f"  {'Intent':<15}"
        f"{'Precision':>10}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'Support':>10}"
    )

    for intent, values in metrics.items():
        print(
            f"  {intent:<15}"
            f"{values['precision']:>10.1%}"
            f"{values['recall']:>10.1%}"
            f"{values['f1']:>10.1%}"
            f"{values['support']:>10}"
        )
        
    for field, title in [
        ("language", "By language"),
        ("difficulty", "By difficulty"),
    ]:
        groups = calculate_group_accuracy(results, field)

        print(f"\n{title}:")
        for group, values in groups.items():
            print(
                f"  {group:<15}"
                f"{values['passed']:>3}/{values['total']:<3}"
                f"{values['accuracy']:>8.1%}"
            )

    safety = calculate_safety_metrics(results)

    print("\nSafety metrics:")
    print(
        f"  Emergency recall:          "
        f"{safety['emergency_recall']:.1%}"
    )
    print(
        f"  Emergency false negatives: "
        f"{safety['emergency_false_negatives']}"
    )
    print(
        f"  Clinical recall:           "
        f"{safety['clinical_recall']:.1%}"
    )
    print(
        f"  Clinical false negatives:  "
        f"{safety['clinical_false_negatives']}"
    )
    print(
        f"  Critical severity:         "
        f"{safety['critical_passed']}/"
        f"{safety['critical_total']} "
        f"({safety['critical_pass_rate']:.1%})"
    )
    
    print_confusion_matrix(results)

    source_metrics = calculate_source_metrics(results)

    print("\nBy route source:")
    for source, values in source_metrics.items():
        print(
            f"  {source:<10}"
            f"{values['passed']:>3}/{values['total']:<3}"
            f"{values['accuracy']:>8.1%}"
        )
        
    regex_count = sum(
        result["source"] == "regex"
        for result in results
    )

    print(
        f"  LLM avoidance rate: "
        f"{regex_count / len(results):.1%}"
    )

    print("\nFailures:")
    failures = [result for result in results if not result["passed"]]

    if not failures:
        print("  None")
        return

    for result in failures:
        print(
            f"  {result['id']}: "
            f"{result['expected']} -> {result['actual']} "
            f"[{result['language']}, "
            f"{result['difficulty']}, "
            f"{result['source']}]"
        )

def calculate_intent_metrics(results: list[dict]) -> dict:
    labels = sorted(
        set(result["expected"] for result in results)
        | set(result["actual"] for result in results)
    )

    metrics = {}

    for label in labels:
        tp = sum(
            r["expected"] == label and r["actual"] == label
            for r in results
        )
        fp = sum(
            r["expected"] != label and r["actual"] == label
            for r in results
        )
        fn = sum(
            r["expected"] == label and r["actual"] != label
            for r in results
        )

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    return metrics

def calculate_group_accuracy(
    results: list[dict],
    field: str,
) -> dict:
    groups = sorted(set(result[field] for result in results))
    metrics = {}

    for group in groups:
        group_results = [
            result for result in results
            if result[field] == group
        ]

        passed = sum(result["passed"] for result in group_results)
        total = len(group_results)

        metrics[group] = {
            "passed": passed,
            "total": total,
            "accuracy": passed / total if total else 0.0,
        }

    return metrics

def calculate_safety_metrics(results: list[dict]) -> dict:
    emergency_cases = [
        result for result in results
        if result["expected"] == "emergency"
    ]
    clinical_cases = [
        result for result in results
        if result["expected"] == "clinical"
    ]
    critical_cases = [
        result for result in results
        if result["severity"] == "critical"
    ]

    emergency_passed = sum(
        result["passed"] for result in emergency_cases
    )
    clinical_passed = sum(
        result["passed"] for result in clinical_cases
    )
    critical_passed = sum(
        result["passed"] for result in critical_cases
    )

    emergency_false_negatives = [
        result for result in emergency_cases
        if not result["passed"]
    ]

    clinical_false_negatives = [
        result for result in clinical_cases
        if not result["passed"]
    ]

    return {
        "emergency_recall": (
            emergency_passed / len(emergency_cases)
            if emergency_cases else 0.0
        ),
        "emergency_false_negatives": len(
            emergency_false_negatives
        ),
        "clinical_recall": (
            clinical_passed / len(clinical_cases)
            if clinical_cases else 0.0
        ),
        "clinical_false_negatives": len(
            clinical_false_negatives
        ),
        "critical_pass_rate": (
            critical_passed / len(critical_cases)
            if critical_cases else 0.0
        ),
        "critical_passed": critical_passed,
        "critical_total": len(critical_cases),
    }
    
def calculate_source_metrics(results: list[dict]) -> dict:
    sources = sorted(set(result["source"] for result in results))
    metrics = {}

    for source in sources:
        source_results = [
            result for result in results
            if result["source"] == source
        ]

        passed = sum(
            result["passed"]
            for result in source_results
        )
        total = len(source_results)

        metrics[source] = {
            "passed": passed,
            "total": total,
            "accuracy": passed / total if total else 0.0,
        }

    return metrics

def print_confusion_matrix(results: list[dict]) -> None:
    labels = sorted(
        set(result["expected"] for result in results)
        | set(result["actual"] for result in results)
    )

    matrix = {
        expected: {
            actual: 0
            for actual in labels
        }
        for expected in labels
    }

    for result in results:
        matrix[result["expected"]][result["actual"]] += 1

    width = max(12, max(len(label) for label in labels) + 2)

    print("\nConfusion matrix:")
    print(
        f"  {'Expected':<{width}}"
        + "".join(
            f"{label:>{width}}"
            for label in labels
        )
    )

    for expected in labels:
        print(
            f"  {expected:<{width}}"
            + "".join(
                f"{matrix[expected][actual]:>{width}}"
                for actual in labels
            )
        )
        
    
if __name__ == "__main__":
    results = run("evals/datasets/routing.jsonl")

    print_summary(results)

    report = build_report(results)
    report_path = save_report(report)

    print(f"\nReport saved: {report_path}")

"""Evaluate Redis vector similarity for RAG relevance gating."""

import csv
from pathlib import Path

from app.services.rag_vector_retriever import get_vector_retriever


TEST_CASES = [
    # Questions expected to be covered by our documents
    ("in_scope", "What is bulk billing?"),
    ("in_scope", "Can I see a GP without paying?"),
    ("in_scope", "How do I find a bulk-billing doctor?"),
    ("in_scope", "How does Medicare pay my GP?"),
    ("in_scope", "Do I need to consent to bulk billing?"),
    ("in_scope", "我怎样才能免费去看家庭医生？"),
    ("in_scope", "什么是全额报销？"),
    ("in_scope", "What is the PBS co-payment?"),
    ("in_scope", "How much do PBS medicines cost?"),
    ("in_scope", "What are Medicare Mental Health Centres?"),
    ("in_scope", "如何寻找免费的心理健康支持服务？"),

    # Healthcare navigation questions outside current document coverage
    ("out_of_scope", "How do I apply for a Medicare card?"),
    ("out_of_scope", "How can I book a dentist appointment?"),
    ("out_of_scope", "How much does an ambulance cost in Victoria?"),
    ("out_of_scope", "Where can I park at the Royal Children's Hospital?"),
    ("out_of_scope", "How do I get a specialist referral?"),
    ("out_of_scope", "How do I find a Chinese-speaking physiotherapist?"),
    ("out_of_scope", "How long is the waiting list for public surgery?"),
    ("out_of_scope", "如何申请 Medicare 卡？"),
    ("out_of_scope", "我怎样预约牙医？"),
    ("out_of_scope", "How do I renew my driver's licence?"),
]


def main():
    retriever = get_vector_retriever()
    results = []

    print(
        f"{'EXPECTED':<15} "
        f"{'TOP SCORE':<12} "
        f"{'SOURCE':<32} QUESTION"
    )
    print("-" * 110)

    for expected, question in TEST_CASES:
        hits = retriever.retrieve(question, top_k=3)

        top_score = hits[0]["score"] if hits else 0.0
        top_source = (
            hits[0]["metadata"]["source_id"]
            if hits else "NONE"
        )

        results.append({
            "expected": expected,
            "question": question,
            "top_score": top_score,
            "top_source": top_source,
            "second_score": (
                hits[1]["score"] if len(hits) > 1 else None
            ),
            "third_score": (
                hits[2]["score"] if len(hits) > 2 else None
            ),
        })

        print(
            f"{expected:<15} "
            f"{top_score:<12.3f} "
            f"{top_source:<32} {question}"
        )

    output_path = Path("data/reports/vector_relevance.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "expected",
                "question",
                "top_score",
                "top_source",
                "second_score",
                "third_score",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")

    print("\nThreshold evaluation")
    print(
        f"{'THRESHOLD':<12} "
        f"{'IN-SCOPE PASS':<18} "
        f"{'OUT-OF-SCOPE BLOCK'}"
    )

    for threshold in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        in_scope = [
            row for row in results
            if row["expected"] == "in_scope"
        ]
        out_scope = [
            row for row in results
            if row["expected"] == "out_of_scope"
        ]

        passed = sum(
            row["top_score"] >= threshold
            for row in in_scope
        )

        blocked = sum(
            row["top_score"] < threshold
            for row in out_scope
        )

        print(
            f"{threshold:<12.2f} "
            f"{passed}/{len(in_scope):<16} "
            f"{blocked}/{len(out_scope)}"
        )


if __name__ == "__main__":
    main()

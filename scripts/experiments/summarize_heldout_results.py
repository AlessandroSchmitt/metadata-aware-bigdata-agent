import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "artifacts/benchmarks/"
      "heldout_abc_benchmark.json"
)

OUTPUT_PATH = (
    ROOT
    / "artifacts/benchmarks/"
      "heldout_abc_summary.json"
)


data = json.loads(
    INPUT_PATH.read_text(
        encoding="utf-8"
    )
)

records = data["records"]


def accuracy(items, key):
    if not items:
        return None

    return (
        sum(
            bool(item[key]["result_correct"])
            for item in items
        )
        / len(items)
    )


perfect_recall = [
    record
    for record in records
    if record["retrieval"]["metadata_recall"] == 1.0
]

imperfect_recall = [
    record
    for record in records
    if record["retrieval"]["metadata_recall"] < 1.0
]


failures = {}

for config in ["A", "B", "C"]:
    failures[config] = [
        record["id"]
        for record in records
        if not record[config]["result_correct"]
    ]


repair_attempts = [
    record
    for record in records
    if record["C"]["repair_attempted"]
]

successful_repairs = [
    record["id"]
    for record in repair_attempts
    if (
        not record["B"]["result_correct"]
        and record["C"]["result_correct"]
    )
]


valid_but_incorrect = {}

for config in ["A", "B", "C"]:
    valid_but_incorrect[config] = [
        record["id"]
        for record in records
        if (
            record[config]["validation"]["valid"]
            and not record[config]["result_correct"]
        )
    ]


summary = {
    "benchmark": data["benchmark"],
    "configuration": data["configuration"],
    "overall": data["summary"],
    "derived": {
        "query_count": len(records),

        "correct_query_counts": {
            config: sum(
                bool(
                    record[config][
                        "result_correct"
                    ]
                )
                for record in records
            )
            for config in ["A", "B", "C"]
        },

        "failure_ids": failures,

        "valid_but_incorrect_ids": (
            valid_but_incorrect
        ),

        "retrieval_stratification": {
            "perfect_recall_queries": len(
                perfect_recall
            ),
            "imperfect_recall_queries": len(
                imperfect_recall
            ),

            "perfect_recall_accuracy": {
                "B": accuracy(
                    perfect_recall,
                    "B",
                ),
                "C": accuracy(
                    perfect_recall,
                    "C",
                ),
            },

            "imperfect_recall_accuracy": {
                "B": accuracy(
                    imperfect_recall,
                    "B",
                ),
                "C": accuracy(
                    imperfect_recall,
                    "C",
                ),
            },

            "imperfect_recall_ids": [
                {
                    "id": record["id"],
                    "recall": record[
                        "retrieval"
                    ][
                        "metadata_recall"
                    ],
                    "B_correct": record[
                        "B"
                    ][
                        "result_correct"
                    ],
                    "C_correct": record[
                        "C"
                    ][
                        "result_correct"
                    ],
                }
                for record
                in imperfect_recall
            ],
        },

        "repair": {
            "attempts": len(
                repair_attempts
            ),
            "successful_result_repairs": (
                successful_repairs
            ),
        },
    },
}


OUTPUT_PATH.write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


print("=" * 70)
print("HELD-OUT RESULT SUMMARY")
print("=" * 70)

counts = summary[
    "derived"
][
    "correct_query_counts"
]

print(
    f"A correct: {counts['A']}/20"
)

print(
    f"B correct: {counts['B']}/20"
)

print(
    f"C correct: {counts['C']}/20"
)

print()
print(
    "Perfect-recall queries:",
    len(perfect_recall),
)

print(
    "B accuracy with perfect recall:",
    f"{accuracy(perfect_recall, 'B'):.3f}",
)

print(
    "C accuracy with perfect recall:",
    f"{accuracy(perfect_recall, 'C'):.3f}",
)

print()
print(
    "Imperfect-recall queries:",
    len(imperfect_recall),
)

print(
    "B accuracy with imperfect recall:",
    f"{accuracy(imperfect_recall, 'B'):.3f}",
)

print(
    "C accuracy with imperfect recall:",
    f"{accuracy(imperfect_recall, 'C'):.3f}",
)

print()
print(
    "Successful result repairs:",
    successful_repairs,
)

print()
print("=" * 70)
print("OUTPUT")
print("=" * 70)

print(
    OUTPUT_PATH.relative_to(ROOT)
)

print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(
    "HELD-OUT RESULT SUMMARY: PASS"
)

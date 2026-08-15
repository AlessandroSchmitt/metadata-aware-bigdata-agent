import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

CANONICAL_PATH = (
    ROOT / "config/benchmark_queries.json"
)

HELDOUT_PATH = (
    ROOT / "config/benchmark_heldout_queries.json"
)


canonical = json.loads(
    CANONICAL_PATH.read_text(
        encoding="utf-8"
    )
)

heldout = json.loads(
    HELDOUT_PATH.read_text(
        encoding="utf-8"
    )
)


canonical_by_id = {
    query["id"]: query
    for query in canonical["queries"]
}


print("=" * 70)
print("HELD-OUT BENCHMARK VALIDATION")
print("=" * 70)


errors = []

questions = [
    query["question"].strip().casefold()
    for query in heldout["queries"]
]

duplicates = [
    question
    for question, count
    in Counter(questions).items()
    if count > 1
]

if duplicates:
    errors.append(
        "Duplicate held-out question text detected."
    )


if len(heldout["queries"]) != 20:
    errors.append(
        "Expected exactly 20 held-out questions."
    )


counts = Counter(
    query["source_query_id"]
    for query in heldout["queries"]
)


for source_id in sorted(
    canonical_by_id
):
    if counts[source_id] != 2:
        errors.append(
            f"{source_id}: expected 2 variants, "
            f"found {counts[source_id]}."
        )


passed = 0


for query in heldout["queries"]:
    heldout_id = query["id"]
    source_id = query[
        "source_query_id"
    ]

    print()
    print(
        f"{heldout_id} "
        f"(source={source_id})"
    )

    if source_id not in canonical_by_id:
        errors.append(
            f"{heldout_id}: unknown "
            f"source_query_id {source_id}."
        )

        print(
            "  FAIL unknown source"
        )
        continue

    source = canonical_by_id[
        source_id
    ]

    checks = {
        "question_is_new": (
            query["question"]
            .strip()
            .casefold()
            != source["question"]
            .strip()
            .casefold()
        ),
        "category_preserved": (
            query["category"]
            == source["category"]
        ),
        "expected_columns_preserved": (
            query["expected_columns"]
            == source["expected_columns"]
        ),
        "gold_sql_preserved": (
            query["gold_sql"]
            == source["gold_sql"]
        ),
        "required_metadata_preserved": (
            query["required_metadata"]
            == source["required_metadata"]
        ),
    }

    query_pass = all(
        checks.values()
    )

    for name, result in checks.items():
        print(
            f"  "
            f"{'PASS' if result else 'FAIL'} "
            f"{name}"
        )

    if not query_pass:
        errors.append(
            f"{heldout_id}: one or more "
            "invariance checks failed."
        )
    else:
        passed += 1


print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Held-out questions passed: "
    f"{passed}/20"
)

print(
    f"Unique question texts: "
    f"{len(set(questions))}/20"
)

print(
    f"Intent coverage: "
    f"{len(counts)}/10"
)


print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

if errors:
    for error in errors:
        print(
            f"ERROR: {error}"
        )

    raise RuntimeError(
        "Held-out benchmark validation failed."
    )

print(
    "HELD-OUT BENCHMARK VALIDATION: PASS"
)

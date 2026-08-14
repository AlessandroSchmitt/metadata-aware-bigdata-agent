import json
from pathlib import Path

from metadata_agent.retrieval import (
    RelationAwareMetadataRetriever,
)


ROOT = Path(__file__).resolve().parents[2]

CATALOG_PATH = (
    ROOT
    / "data/catalog/metadata_catalog.sqlite"
)

QDRANT_PATH = (
    ROOT
    / ".qdrant/metadata_catalog"
)

CONTEXT_PATH = (
    ROOT
    / "artifacts/benchmarks/"
      "relation_aware_context.txt"
)

REPORT_PATH = (
    ROOT
    / "artifacts/benchmarks/"
      "relation_aware_retrieval.json"
)


QUESTION = (
    "What was the average trip distance "
    "for Yellow Taxi trips during rainy "
    "hours in January 2024? "
    "Return exactly one column named "
    "average_trip_distance."
)


GOLD_REQUIRED = {
    "dataset:yellow_taxi",
    "dataset:weather_hourly",
    "column:yellow_taxi.trip_distance",
    (
        "column:yellow_taxi."
        "tpep_pickup_datetime"
    ),
    "column:weather_hourly.weather_hour",
    (
        "column:weather_hourly."
        "has_precipitation"
    ),
    (
        "relationship:"
        "yellow_pickup_weather_hour"
    ),
    "rule:rainy_hour",
}


def selection_keys(selection):
    keys = set()

    keys.update(
        "dataset:" + item
        for item
        in selection["datasets"]
    )

    keys.update(
        "column:" + item
        for item
        in selection["columns"]
    )

    keys.update(
        "relationship:" + item
        for item
        in selection["relationships"]
    )

    keys.update(
        "rule:" + item
        for item
        in selection["rules"]
    )

    return keys


print("=" * 70)
print("RELATION-AWARE METADATA RETRIEVAL")
print("=" * 70)

print()
print(f"Question: {QUESTION}")


retriever = (
    RelationAwareMetadataRetriever(
        catalog_path=CATALOG_PATH,
        qdrant_path=QDRANT_PATH,
    )
)


result = retriever.retrieve(
    QUESTION,
    dense_top_k=5,
)


print()
print("=" * 70)
print("DENSE SEED TOP 5")
print("=" * 70)

for point in result[
    "dense"
]["points"]:
    print(
        f"{point['rank']:2d}. "
        f"score={point['score']:.6f} "
        f"{point['entity_type']:12s} "
        f"{point['entity_key']}"
    )


print()
print("=" * 70)
print("LEXICAL SEEDS")
print("=" * 70)

for key in [
    "datasets",
    "concepts",
    "rules",
]:
    print(
        f"{key}: "
        f"{sorted(result['lexical'][key])}"
    )


print()
print("=" * 70)
print("EXPANDED SELECTION")
print("=" * 70)

for key in [
    "datasets",
    "columns",
    "concepts",
    "relationships",
    "rules",
]:
    print()
    print(
        f"{key.upper()} "
        f"({len(result['selection'][key])})"
    )

    for item in sorted(
        result["selection"][key]
    ):
        print(
            f"  - {item}"
        )


retrieved_keys = selection_keys(
    result["selection"]
)

hits = (
    GOLD_REQUIRED
    & retrieved_keys
)

missing = (
    GOLD_REQUIRED
    - retrieved_keys
)

extras = (
    retrieved_keys
    - GOLD_REQUIRED
)


print()
print("=" * 70)
print("GOLD METADATA EVALUATION")
print("=" * 70)

print(
    f"Gold required:     "
    f"{len(GOLD_REQUIRED)}"
)

print(
    f"Gold retrieved:    "
    f"{len(hits)}"
)

recall = (
    len(hits)
    / len(GOLD_REQUIRED)
)

precision = (
    len(hits)
    / len(retrieved_keys)
    if retrieved_keys
    else 0.0
)

f1 = (
    2 * precision * recall
    / (precision + recall)
    if precision + recall > 0
    else 0.0
)

print(
    f"Recall:            "
    f"{recall:.3f}"
)

print(
    f"Precision:         "
    f"{precision:.3f}"
)

print(
    f"F1:                "
    f"{f1:.3f}"
)


print()
print("Missing gold items:")

if missing:
    for item in sorted(missing):
        print(
            f"  - {item}"
        )
else:
    print("  None")


print()
print("Extra retrieved items:")

if extras:
    for item in sorted(extras):
        print(
            f"  - {item}"
        )
else:
    print("  None")


context = result["context"]


print()
print("=" * 70)
print("RETRIEVED CONTEXT STATISTICS")
print("=" * 70)

print(
    f"Characters: "
    f"{len(context):,}"
)

print(
    f"Words:      "
    f"{len(context.split()):,}"
)

print(
    f"Lines:      "
    f"{len(context.splitlines()):,}"
)


FULL_CATALOG_CHARACTERS = 8600
FULL_CATALOG_WORDS = 930


character_reduction = (
    1
    - len(context)
    / FULL_CATALOG_CHARACTERS
) * 100

word_reduction = (
    1
    - len(context.split())
    / FULL_CATALOG_WORDS
) * 100


print()
print(
    f"Full Catalog characters: "
    f"{FULL_CATALOG_CHARACTERS:,}"
)

print(
    f"Character reduction:     "
    f"{character_reduction:.2f}%"
)

print(
    f"Full Catalog words:      "
    f"{FULL_CATALOG_WORDS:,}"
)

print(
    f"Word reduction:          "
    f"{word_reduction:.2f}%"
)


print()
print("=" * 70)
print("RETRIEVED CONTEXT")
print("=" * 70)

print(context)


CONTEXT_PATH.write_text(
    context,
    encoding="utf-8",
)


report = {
    "question": QUESTION,
    "dense_top_k": 5,
    "dense_embedding": (
        result["dense"]["embedding"]
    ),
    "lexical": {
        key: sorted(value)
        for key, value
        in result["lexical"].items()
    },
    "selection": {
        key: sorted(value)
        for key, value
        in result["selection"].items()
    },
    "gold_evaluation": {
        "required": sorted(
            GOLD_REQUIRED
        ),
        "retrieved": sorted(
            retrieved_keys
        ),
        "hits": sorted(hits),
        "missing": sorted(missing),
        "extras": sorted(extras),
        "recall": recall,
        "precision": precision,
        "f1": f1,
    },
    "context": {
        "characters": len(context),
        "words": len(
            context.split()
        ),
        "lines": len(
            context.splitlines()
        ),
        "full_catalog_characters": (
            FULL_CATALOG_CHARACTERS
        ),
        "full_catalog_words": (
            FULL_CATALOG_WORDS
        ),
        "character_reduction_percent": (
            character_reduction
        ),
        "word_reduction_percent": (
            word_reduction
        ),
    },
}


REPORT_PATH.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 70)
print("OUTPUT")
print("=" * 70)

print(
    CONTEXT_PATH.relative_to(ROOT)
)

print(
    REPORT_PATH.relative_to(ROOT)
)


print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

if recall == 1.0:
    print(
        "RELATION-AWARE GOLD RECALL: PASS"
    )
else:
    print(
        "RELATION-AWARE GOLD RECALL: FAIL"
    )

print(
    "RELATION-AWARE RETRIEVAL TEST: "
    "COMPLETED"
)

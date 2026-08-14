import json
import statistics
import time
from pathlib import Path

from metadata_agent.catalog import (
    MetadataCatalog,
)
from metadata_agent.retrieval import (
    RelationAwareMetadataRetriever,
)


ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_PATH = (
    ROOT / "config/benchmark_queries.json"
)

CATALOG_PATH = (
    ROOT / "data/catalog/metadata_catalog.sqlite"
)

QDRANT_PATH = (
    ROOT / ".qdrant/metadata_catalog"
)

OUTPUT_PATH = (
    ROOT
    / "artifacts/benchmarks/"
      "retrieval_benchmark_canonical.json"
)

DENSE_TOP_K = 5


def selection_keys(selection):
    keys = set()

    keys.update(
        "dataset:" + item
        for item in selection["datasets"]
    )

    keys.update(
        "column:" + item
        for item in selection["columns"]
    )

    keys.update(
        "relationship:" + item
        for item
        in selection["relationships"]
    )

    keys.update(
        "rule:" + item
        for item in selection["rules"]
    )

    return keys


benchmark = json.loads(
    BENCHMARK_PATH.read_text(
        encoding="utf-8"
    )
)

catalog = MetadataCatalog(
    CATALOG_PATH
)

full_context = (
    catalog.render_full_catalog()
)

retriever = (
    RelationAwareMetadataRetriever(
        catalog_path=CATALOG_PATH,
        qdrant_path=QDRANT_PATH,
    )
)


print("=" * 70)
print("CANONICAL RETRIEVAL BENCHMARK")
print("=" * 70)

print(
    f"Dense Top-K:         "
    f"{DENSE_TOP_K}"
)

print(
    f"Full Catalog chars:  "
    f"{len(full_context):,}"
)

print(
    f"Full Catalog words:  "
    f"{len(full_context.split()):,}"
)


records = []


for query in benchmark["queries"]:

    print()
    print("=" * 70)
    print(
        f"{query['id']} — "
        f"{query['category']}"
    )
    print("=" * 70)

    print(
        f"Question: "
        f"{query['question']}"
    )

    start = time.perf_counter()

    result = retriever.retrieve(
        query["question"],
        dense_top_k=DENSE_TOP_K,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    selected = selection_keys(
        result["selection"]
    )

    gold = set(
        query["required_metadata"]
    )

    hits = (
        selected & gold
    )

    missing = (
        gold - selected
    )

    extras = (
        selected - gold
    )

    recall = (
        len(hits) / len(gold)
        if gold
        else 1.0
    )

    precision = (
        len(hits) / len(selected)
        if selected
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (
            precision + recall
            > 0
        )
        else 0.0
    )

    context = result["context"]

    char_reduction = (
        1
        - len(context)
        / len(full_context)
    ) * 100

    word_reduction = (
        1
        - len(context.split())
        / len(full_context.split())
    ) * 100

    print()
    print("Dense seeds:")

    for point in result[
        "dense"
    ]["points"]:
        print(
            f"  {point['rank']}. "
            f"{point['entity_type']}:"
            f"{point['entity_key']} "
            f"score={point['score']:.4f}"
        )

    print()
    print(
        "Lexical datasets: "
        f"{sorted(result['lexical']['datasets'])}"
    )

    print(
        "Lexical concepts: "
        f"{sorted(result['lexical']['concepts'])}"
    )

    print(
        "Lexical rules:    "
        f"{sorted(result['lexical']['rules'])}"
    )

    print()
    print(
        f"Selected datasets:      "
        f"{len(result['selection']['datasets'])}"
    )

    print(
        f"Selected columns:       "
        f"{len(result['selection']['columns'])}"
    )

    print(
        f"Selected relationships: "
        f"{len(result['selection']['relationships'])}"
    )

    print(
        f"Selected rules:         "
        f"{len(result['selection']['rules'])}"
    )

    print()
    print(
        f"Gold metadata:      "
        f"{len(gold)}"
    )

    print(
        f"Gold hits:          "
        f"{len(hits)}"
    )

    print(
        f"Recall:             "
        f"{recall:.3f}"
    )

    print(
        f"Precision:          "
        f"{precision:.3f}"
    )

    print(
        f"F1:                 "
        f"{f1:.3f}"
    )

    if missing:
        print("Missing:")

        for item in sorted(missing):
            print(
                f"  - {item}"
            )

    else:
        print(
            "Missing:            None"
        )

    print(
        f"Context chars:      "
        f"{len(context):,}"
    )

    print(
        f"Character reduction:"
        f" {char_reduction:.2f}%"
    )

    print(
        f"Retrieval time:     "
        f"{elapsed:.3f} s"
    )

    records.append(
        {
            "id": query["id"],
            "category": (
                query["category"]
            ),
            "question": (
                query["question"]
            ),
            "dense_top_k": (
                DENSE_TOP_K
            ),
            "dense": (
                result["dense"]["points"]
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
            "gold": sorted(gold),
            "hits": sorted(hits),
            "missing": sorted(missing),
            "extras": sorted(extras),
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "context_characters": (
                len(context)
            ),
            "context_words": (
                len(context.split())
            ),
            "character_reduction_percent": (
                char_reduction
            ),
            "word_reduction_percent": (
                word_reduction
            ),
            "retrieval_seconds": (
                elapsed
            ),
            "embedding_prompt_tokens": (
                result[
                    "dense"
                ][
                    "embedding"
                ][
                    "prompt_eval_count"
                ]
            ),
        }
    )


recalls = [
    record["recall"]
    for record in records
]

precisions = [
    record["precision"]
    for record in records
]

f1_values = [
    record["f1"]
    for record in records
]

reductions = [
    record[
        "character_reduction_percent"
    ]
    for record in records
]

retrieval_times = [
    record["retrieval_seconds"]
    for record in records
]

perfect_recall = sum(
    record["recall"] == 1.0
    for record in records
)


print()
print("=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)

print(
    f"Queries:                 "
    f"{len(records)}"
)

print(
    f"Perfect recall queries:  "
    f"{perfect_recall}/"
    f"{len(records)}"
)

print(
    f"Macro Recall:            "
    f"{statistics.mean(recalls):.3f}"
)

print(
    f"Macro Precision:         "
    f"{statistics.mean(precisions):.3f}"
)

print(
    f"Macro F1:                "
    f"{statistics.mean(f1_values):.3f}"
)

print(
    f"Mean context reduction:  "
    f"{statistics.mean(reductions):.2f}%"
)

print(
    f"Mean retrieval time:     "
    f"{statistics.mean(retrieval_times):.3f} s"
)


report = {
    "benchmark": (
        benchmark["benchmark"]
    ),
    "configuration": {
        "dense_top_k": (
            DENSE_TOP_K
        ),
        "full_catalog_characters": (
            len(full_context)
        ),
        "full_catalog_words": (
            len(full_context.split())
        ),
    },
    "summary": {
        "queries": len(records),
        "perfect_recall_queries": (
            perfect_recall
        ),
        "macro_recall": (
            statistics.mean(recalls)
        ),
        "macro_precision": (
            statistics.mean(
                precisions
            )
        ),
        "macro_f1": (
            statistics.mean(
                f1_values
            )
        ),
        "mean_context_reduction_percent": (
            statistics.mean(
                reductions
            )
        ),
        "mean_retrieval_seconds": (
            statistics.mean(
                retrieval_times
            )
        ),
    },
    "records": records,
}


OUTPUT_PATH.write_text(
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
    OUTPUT_PATH.relative_to(ROOT)
)


print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(
    "CANONICAL RETRIEVAL BENCHMARK: COMPLETED"
)

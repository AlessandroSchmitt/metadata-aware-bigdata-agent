import ast
import gc
import json
import os
import shutil
import sqlite3
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from metadata_agent.catalog import MetadataCatalog
from metadata_agent.embeddings import OllamaEmbedder
from metadata_agent.metadata_documents import (
    MetadataDocumentBuilder,
)
from metadata_agent.retrieval import (
    RelationAwareMetadataRetriever,
)


ROOT = Path(__file__).resolve().parents[2]

DEVELOPMENT_TAG = "development-freeze-v1"
HELDOUT_TAG = "heldout-benchmark-v1"
RESULTS_TAG = "heldout-results-v1"

BASE_DB = (
    ROOT / "data/catalog/metadata_catalog.sqlite"
)

DISTRACTOR_CONFIG = (
    ROOT / "config/catalog_scale_distractors.json"
)

OUTPUT_PATH = (
    ROOT
    / "artifacts/benchmarks/"
      "catalog_scale_benchmark.json"
)

SCALE_DB_DIR = (
    ROOT / "data/catalog/scaling"
)

QDRANT_ROOT = (
    ROOT / ".qdrant/catalog_scaling"
)

COLLECTION = "metadata_catalog"
EMBEDDING_MODEL = "embeddinggemma"
DENSE_TOP_K = 5
BATCH_SIZE = 20

SCALES = [4, 8, 16]

DRY_RUN = (
    os.environ.get(
        "CATALOG_SCALE_DRY_RUN",
        "0",
    )
    == "1"
)


def git_show(tag, path):
    return subprocess.check_output(
        [
            "git",
            "show",
            f"{tag}:{path}",
        ],
        cwd=ROOT,
        text=True,
    )


def git_tag_commit(tag):
    return subprocess.check_output(
        [
            "git",
            "rev-list",
            "-n",
            "1",
            tag,
        ],
        cwd=ROOT,
        text=True,
    ).strip()


def assert_frozen_code():
    paths = [
        "src/metadata_agent/catalog.py",
        "src/metadata_agent/embeddings.py",
        "src/metadata_agent/metadata_documents.py",
        "src/metadata_agent/retrieval.py",
    ]

    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            DEVELOPMENT_TAG,
            "--",
            *paths,
        ],
        cwd=ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Frozen metadata/retrieval code differs "
            "from development-freeze-v1."
        )


def table_count(conn, table):
    return conn.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]


def base_catalog_counts():
    conn = sqlite3.connect(BASE_DB)

    try:
        return {
            "datasets": table_count(
                conn,
                "datasets",
            ),
            "columns": table_count(
                conn,
                "columns",
            ),
            "semantic_concepts": table_count(
                conn,
                "semantic_concepts",
            ),
            "relationships": table_count(
                conn,
                "relationships",
            ),
            "aliases": table_count(
                conn,
                "aliases",
            ),
            "semantic_rules": table_count(
                conn,
                "semantic_rules",
            ),
        }

    finally:
        conn.close()


def load_selection_keys():
    source = git_show(
        RESULTS_TAG,
        (
            "scripts/experiments/"
            "evaluate_retrieval_heldout.py"
        ),
    )

    tree = ast.parse(source)

    node = None

    for item in tree.body:
        if (
            isinstance(item, ast.FunctionDef)
            and item.name == "selection_keys"
        ):
            node = item
            break

    if node is None:
        raise RuntimeError(
            "Frozen selection_keys() not found."
        )

    module = ast.Module(
        body=[node],
        type_ignores=[],
    )

    ast.fix_missing_locations(module)

    namespace = {}

    exec(
        compile(
            module,
            filename=(
                f"{RESULTS_TAG}:"
                "evaluate_retrieval_heldout.py"
            ),
            mode="exec",
        ),
        namespace,
    )

    return namespace["selection_keys"]


selection_keys = load_selection_keys()


def verify_base_catalog():
    counts = base_catalog_counts()

    expected = {
        "datasets": 4,
        "columns": 66,
        "semantic_concepts": 21,
        "relationships": 6,
        "aliases": 27,
        "semantic_rules": 6,
    }

    if counts != expected:
        raise RuntimeError(
            "Base SQLite catalog does not match "
            "the frozen catalog counts.\n"
            f"Expected: {expected}\n"
            f"Observed: {counts}"
        )

    current_context = (
        MetadataCatalog(
            BASE_DB
        ).render_full_catalog()
    )

    frozen_context = git_show(
        DEVELOPMENT_TAG,
        (
            "artifacts/benchmarks/"
            "full_catalog_context.txt"
        ),
    )

    if current_context != frozen_context:
        raise RuntimeError(
            "Current base SQLite catalog does not "
            "render exactly like the frozen catalog."
        )

    return counts, current_context


def insert_distractors(
    db_path,
    distractors,
):
    conn = sqlite3.connect(db_path)

    try:
        conn.execute(
            "PRAGMA foreign_keys = ON"
        )

        concepts = {
            row[0]: row[1]
            for row in conn.execute(
                """
                SELECT name, id
                FROM semantic_concepts
                """
            )
        }

        for dataset in distractors:

            cursor = conn.execute(
                """
                INSERT INTO datasets(
                    name,
                    display_name,
                    description,
                    layer,
                    format,
                    path,
                    source,
                    granularity,
                    primary_time_column,
                    temporal_start,
                    temporal_end,
                    row_count,
                    notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset["name"],
                    dataset["display_name"],
                    dataset["description"],
                    "catalog_scale",
                    "metadata_only",
                    (
                        "metadata-only://catalog-scale/"
                        + dataset["name"]
                    ),
                    (
                        "Synthetic metadata-only "
                        "catalog-scale source"
                    ),
                    dataset["granularity"],
                    dataset.get(
                        "primary_time_column"
                    ),
                    "2024-01-01 00:00:00",
                    "2024-02-01 00:00:00",
                    None,
                    (
                        "Synthetic metadata-only "
                        "hard-negative source. "
                        "No physical Spark dataset."
                    ),
                ),
            )

            dataset_id = cursor.lastrowid

            for position, column in enumerate(
                dataset["columns"],
                start=1,
            ):
                column_cursor = conn.execute(
                    """
                    INSERT INTO columns(
                        dataset_id,
                        name,
                        ordinal_position,
                        data_type,
                        nullable
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        dataset_id,
                        column["name"],
                        position,
                        column["type"],
                        1,
                    ),
                )

                concept_name = column.get(
                    "semantic_concept"
                )

                if concept_name:
                    if concept_name not in concepts:
                        raise RuntimeError(
                            f"Unknown semantic concept "
                            f"{concept_name!r} for "
                            f"{dataset['name']}."
                            f"{column['name']}"
                        )

                    conn.execute(
                        """
                        INSERT INTO column_semantics(
                            column_id,
                            concept_id,
                            semantic_role
                        )
                        VALUES (?, ?, 'represents')
                        """,
                        (
                            column_cursor.lastrowid,
                            concepts[concept_name],
                        ),
                    )

            for alias in dataset.get(
                "aliases",
                [],
            ):
                conn.execute(
                    """
                    INSERT INTO aliases(
                        entity_type,
                        entity_key,
                        alias
                    )
                    VALUES ('dataset', ?, ?)
                    """,
                    (
                        dataset["name"],
                        alias,
                    ),
                )

        conn.commit()

    finally:
        conn.close()


def build_scale_db(
    scale,
    distractors,
):
    SCALE_DB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    db_path = (
        SCALE_DB_DIR
        / f"metadata_catalog_{scale:02d}.sqlite"
    )

    shutil.copy2(
        BASE_DB,
        db_path,
    )

    extra_count = scale - 4

    insert_distractors(
        db_path,
        distractors[:extra_count],
    )

    conn = sqlite3.connect(db_path)

    try:
        counts = {
            "datasets": table_count(
                conn,
                "datasets",
            ),
            "columns": table_count(
                conn,
                "columns",
            ),
            "semantic_concepts": table_count(
                conn,
                "semantic_concepts",
            ),
            "relationships": table_count(
                conn,
                "relationships",
            ),
            "aliases": table_count(
                conn,
                "aliases",
            ),
            "semantic_rules": table_count(
                conn,
                "semantic_rules",
            ),
        }

    finally:
        conn.close()

    if counts["datasets"] != scale:
        raise RuntimeError(
            f"Scale {scale}: expected "
            f"{scale} datasets, observed "
            f"{counts['datasets']}."
        )

    return db_path, counts


def build_vector_index(
    scale,
    db_path,
):
    catalog = MetadataCatalog(
        db_path
    )

    documents = (
        MetadataDocumentBuilder(
            catalog
        ).build()
    )

    document_counts = Counter(
        document.entity_type
        for document in documents
    )

    embedder = OllamaEmbedder(
        model=EMBEDDING_MODEL
    )

    vectors = []

    prompt_tokens = 0
    embedding_wall = 0.0
    embedding_api = 0.0

    start_total = time.perf_counter()

    for start in range(
        0,
        len(documents),
        BATCH_SIZE,
    ):
        batch = documents[
            start:
            start + BATCH_SIZE
        ]

        result = embedder.embed(
            [
                document.text
                for document in batch
            ],
            keep_alive="10m",
        )

        vectors.extend(
            result["embeddings"]
        )

        prompt_tokens += (
            result["prompt_eval_count"]
        )

        embedding_wall += (
            result["wall_time_seconds"]
        )

        embedding_api += (
            result["total_duration_seconds"]
        )

    if len(vectors) != len(documents):
        raise RuntimeError(
            "Embedding/document count mismatch."
        )

    vector_size = len(
        vectors[0]
    )

    qdrant_path = (
        QDRANT_ROOT
        / f"scale_{scale:02d}"
    )

    if qdrant_path.exists():
        shutil.rmtree(
            qdrant_path
        )

    qdrant_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = QdrantClient(
        path=str(qdrant_path)
    )

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )

    points = []

    for point_id, (
        document,
        vector,
    ) in enumerate(
        zip(
            documents,
            vectors,
        ),
        start=1,
    ):
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "entity_type": (
                        document.entity_type
                    ),
                    "entity_key": (
                        document.entity_key
                    ),
                    "dataset": (
                        document.dataset
                    ),
                    "text": (
                        document.text
                    ),
                },
            )
        )

    index_start = time.perf_counter()

    client.upsert(
        collection_name=COLLECTION,
        points=points,
        wait=True,
    )

    index_seconds = (
        time.perf_counter()
        - index_start
    )

    stored = client.count(
        collection_name=COLLECTION,
        exact=True,
    ).count

    client.close()

    total_seconds = (
        time.perf_counter()
        - start_total
    )

    if stored != len(documents):
        raise RuntimeError(
            f"Scale {scale}: Qdrant point "
            "count mismatch."
        )

    return {
        "qdrant_path": qdrant_path,
        "documents": len(documents),
        "document_counts": dict(
            sorted(
                document_counts.items()
            )
        ),
        "vector_size": vector_size,
        "embedding_prompt_tokens": (
            prompt_tokens
        ),
        "embedding_wall_seconds": (
            embedding_wall
        ),
        "embedding_api_seconds": (
            embedding_api
        ),
        "qdrant_index_seconds": (
            index_seconds
        ),
        "total_build_seconds": (
            total_seconds
        ),
        "stored_points": stored,
    }


def mean(values):
    return (
        statistics.mean(values)
        if values
        else 0.0
    )


def evaluate_scale(
    scale,
    db_path,
    qdrant_path,
    queries,
):
    catalog = MetadataCatalog(
        db_path
    )

    full_context = (
        catalog.render_full_catalog()
    )

    retriever = (
        RelationAwareMetadataRetriever(
            catalog_path=db_path,
            qdrant_path=qdrant_path,
            collection=COLLECTION,
            embedding_model=EMBEDDING_MODEL,
        )
    )

    records = []

    for index, query in enumerate(
        queries,
        start=1,
    ):
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

        hits = selected & gold
        missing = gold - selected
        extras = selected - gold

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
            (
                2.0
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )
            if (
                precision
                + recall
            )
            else 0.0
        )

        context = result["context"]

        reduction = (
            100.0
            * (
                1.0
                - (
                    len(context)
                    / len(full_context)
                )
            )
        )

        records.append(
            {
                "id": query["id"],
                "category": (
                    query["category"]
                ),
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "perfect_recall": (
                    recall == 1.0
                ),
                "hits": sorted(hits),
                "missing": sorted(
                    missing
                ),
                "extras": sorted(
                    extras
                ),
                "selected_metadata_count": (
                    len(selected)
                ),
                "context_characters": (
                    len(context)
                ),
                "context_words": (
                    len(context.split())
                ),
                "context_reduction_percent": (
                    reduction
                ),
                "retrieval_seconds": (
                    elapsed
                ),
            }
        )

        print(
            f"scale={scale:2d} "
            f"{query['id']:7s} "
            f"R={recall:.3f} "
            f"P={precision:.3f} "
            f"F1={f1:.3f} "
            f"context={len(context):4d} "
            f"time={elapsed:.3f}s"
        )

    recalls = [
        item["recall"]
        for item in records
    ]

    precisions = [
        item["precision"]
        for item in records
    ]

    f1s = [
        item["f1"]
        for item in records
    ]

    retrieval_times = [
        item["retrieval_seconds"]
        for item in records
    ]

    context_chars = [
        item["context_characters"]
        for item in records
    ]

    reductions = [
        item[
            "context_reduction_percent"
        ]
        for item in records
    ]

    return {
        "full_catalog_characters": len(
            full_context
        ),
        "full_catalog_words": len(
            full_context.split()
        ),
        "full_catalog_lines": len(
            full_context.splitlines()
        ),
        "summary": {
            "queries": len(records),
            "perfect_recall_queries": sum(
                item["perfect_recall"]
                for item in records
            ),
            "macro_recall": mean(
                recalls
            ),
            "macro_precision": mean(
                precisions
            ),
            "macro_f1": mean(
                f1s
            ),
            "mean_context_characters": (
                mean(context_chars)
            ),
            "mean_context_reduction_percent": (
                mean(reductions)
            ),
            "mean_retrieval_seconds": (
                mean(retrieval_times)
            ),
            "median_retrieval_seconds": (
                statistics.median(
                    retrieval_times
                )
            ),
        },
        "records": records,
    }


print("=" * 78)
print("CATALOG-SCALE RETRIEVAL BENCHMARK")
print("=" * 78)

assert_frozen_code()

base_counts, frozen_full_context = (
    verify_base_catalog()
)

config = json.loads(
    DISTRACTOR_CONFIG.read_text(
        encoding="utf-8"
    )
)

distractors = config["datasets"]

if len(distractors) != 12:
    raise RuntimeError(
        "Expected exactly 12 distractors."
    )

names = [
    dataset["name"]
    for dataset in distractors
]

if len(set(names)) != 12:
    raise RuntimeError(
        "Distractor dataset names are not unique."
    )

heldout = json.loads(
    git_show(
        HELDOUT_TAG,
        "config/benchmark_heldout_queries.json",
    )
)

queries = heldout["queries"]

if len(queries) != 20:
    raise RuntimeError(
        "Expected exactly 20 frozen "
        "held-out queries."
    )

print(
    f"Development freeze: "
    f"{DEVELOPMENT_TAG} "
    f"({git_tag_commit(DEVELOPMENT_TAG)[:7]})"
)

print(
    f"Held-out benchmark: "
    f"{HELDOUT_TAG} "
    f"({git_tag_commit(HELDOUT_TAG)[:7]})"
)

print(
    f"Embedding model:    "
    f"{EMBEDDING_MODEL}"
)

print(
    f"Dense Top-K:        "
    f"{DENSE_TOP_K}"
)

print(
    f"Held-out queries:   "
    f"{len(queries)}"
)

print(
    f"Base datasets:      "
    f"{base_counts['datasets']}"
)

print(
    f"Base columns:       "
    f"{base_counts['columns']}"
)

print(
    f"Frozen context:     "
    f"{len(frozen_full_context):,} chars"
)

print()
print("PROJECTED SCALES")

for scale in SCALES:
    extra = scale - 4

    extra_columns = sum(
        len(dataset["columns"])
        for dataset
        in distractors[:extra]
    )

    print(
        f"  {scale:2d} sources: "
        f"+{extra:2d} distractors, "
        f"projected columns="
        f"{base_counts['columns'] + extra_columns}"
    )


if DRY_RUN:
    print()
    print("=" * 78)
    print("FINAL RESULT")
    print("=" * 78)
    print(
        "CATALOG-SCALE BENCHMARK DRY-RUN: PASS"
    )
    raise SystemExit(0)


results = {}

for scale in SCALES:

    print()
    print("=" * 78)
    print(
        f"SCALE {scale}: BUILD CATALOG"
    )
    print("=" * 78)

    db_path, counts = build_scale_db(
        scale,
        distractors,
    )

    print(
        f"Datasets: {counts['datasets']}"
    )

    print(
        f"Columns:  {counts['columns']}"
    )

    print()
    print("=" * 78)
    print(
        f"SCALE {scale}: BUILD VECTOR INDEX"
    )
    print("=" * 78)

    index_info = build_vector_index(
        scale,
        db_path,
    )

    print(
        f"Metadata documents: "
        f"{index_info['documents']}"
    )

    print(
        f"Embedding tokens:   "
        f"{index_info['embedding_prompt_tokens']}"
    )

    print(
        f"Embedding wall:     "
        f"{index_info['embedding_wall_seconds']:.2f}s"
    )

    print()
    print("=" * 78)
    print(
        f"SCALE {scale}: RETRIEVAL"
    )
    print("=" * 78)

    evaluation = evaluate_scale(
        scale,
        db_path,
        index_info["qdrant_path"],
        queries,
    )

    results[str(scale)] = {
        "catalog_counts": counts,
        "index": {
            key: (
                str(value.relative_to(ROOT))
                if (
                    key == "qdrant_path"
                )
                else value
            )
            for key, value
            in index_info.items()
        },
        "retrieval": evaluation,
    }

    del evaluation
    gc.collect()


output = {
    "benchmark": {
        "name": (
            "Frozen Relation-Aware Metadata "
            "Catalog-Scale Benchmark"
        ),
        "purpose": (
            "Measure retrieval robustness and "
            "context compression as metadata "
            "catalog size grows from 4 to 8 "
            "to 16 sources."
        ),
        "development_freeze": (
            DEVELOPMENT_TAG
        ),
        "development_commit": (
            git_tag_commit(
                DEVELOPMENT_TAG
            )
        ),
        "heldout_benchmark": (
            HELDOUT_TAG
        ),
        "heldout_commit": (
            git_tag_commit(
                HELDOUT_TAG
            )
        ),
        "scales": SCALES,
        "dense_top_k": DENSE_TOP_K,
        "embedding_model": (
            EMBEDDING_MODEL
        ),
        "distractor_type": (
            "synthetic metadata-only "
            "hard-negative sources"
        ),
    },
    "base_catalog_counts": (
        base_counts
    ),
    "results": results,
}


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH.write_text(
    json.dumps(
        output,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


print()
print("=" * 78)
print("CATALOG-SCALE SUMMARY")
print("=" * 78)

print(
    "Scale  Datasets  Columns  Docs  "
    "FullChars  MeanCtx  Perfect  "
    "Recall  Precision  F1  Reduction"
)

for scale in SCALES:
    item = results[str(scale)]

    counts = item[
        "catalog_counts"
    ]

    index_info = item[
        "index"
    ]

    retrieval = item[
        "retrieval"
    ]

    summary = retrieval[
        "summary"
    ]

    print(
        f"{scale:5d}  "
        f"{counts['datasets']:8d}  "
        f"{counts['columns']:7d}  "
        f"{index_info['documents']:4d}  "
        f"{retrieval['full_catalog_characters']:9d}  "
        f"{summary['mean_context_characters']:7.1f}  "
        f"{summary['perfect_recall_queries']:7d}/20  "
        f"{summary['macro_recall']:.3f}  "
        f"{summary['macro_precision']:.3f}  "
        f"{summary['macro_f1']:.3f}  "
        f"{summary['mean_context_reduction_percent']:.2f}%"
    )


print()
print("=" * 78)
print("OUTPUT")
print("=" * 78)

print(
    OUTPUT_PATH.relative_to(ROOT)
)

print()
print("=" * 78)
print("FINAL RESULT")
print("=" * 78)

print(
    "CATALOG-SCALE RETRIEVAL BENCHMARK: COMPLETED"
)

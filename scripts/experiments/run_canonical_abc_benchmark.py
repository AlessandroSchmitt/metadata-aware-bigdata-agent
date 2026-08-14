import json
import math
import os
import re
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

from pyspark.sql import SparkSession

from metadata_agent.catalog import MetadataCatalog
from metadata_agent.retrieval import RelationAwareMetadataRetriever
from metadata_agent.sql_repair import OllamaSQLRepairer
from metadata_agent.sql_validation import SparkSQLValidator


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
    / "artifacts/results/"
      "canonical_abc_benchmark.json"
)

DATASETS = {
    "yellow_taxi": (
        ROOT / "data/curated/yellow/2024-01"
    ),
    "green_taxi": (
        ROOT / "data/curated/green/2024-01"
    ),
    "taxi_zones": (
        ROOT / "data/curated/zones"
    ),
    "weather_hourly": (
        ROOT / "data/curated/weather/2024-01"
    ),
}

MODEL = "qwen2.5-coder:3b"
NUM_CTX = 4096
TEMPERATURE = 0
DENSE_TOP_K = 5


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def clean_sql(text):
    text = text.strip()

    text = re.sub(
        r"^```(?:sql)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    return text.strip()


def ollama_generate(
    prompt,
    keep_alive="30m",
):
    payload = {
        "model": MODEL,
        "stream": False,
        "keep_alive": keep_alive,
        "prompt": prompt,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
        },
    }

    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode(
            "utf-8"
        ),
        headers={
            "Content-Type": "application/json"
        },
    )

    start = time.perf_counter()

    with urllib.request.urlopen(
        request,
        timeout=600,
    ) as response:
        result = json.loads(
            response.read().decode(
                "utf-8"
            )
        )

    return (
        result,
        time.perf_counter() - start,
    )


def build_prompt(
    question,
    metadata_context,
):
    return f"""
You are an expert Spark SQL generator.

You must answer the user question using ONLY the metadata
context supplied below.

The datasets are already registered in Spark SQL using
exactly these table names:

- yellow_taxi
- green_taxi
- taxi_zones
- weather_hourly

Important instructions:

- Generate valid Spark SQL.
- Use only physical tables and physical columns present in the metadata context.
- Semantic concept names and aliases are metadata labels, not SQL column names.
- Use the exact physical column names shown in dataset schemas or selected_columns.
- Relationship names and JOIN RULE names are metadata identifiers, not tables.
- Never place a relationship or JOIN RULE name in FROM or JOIN.
- Use the physical_join_condition supplied by a relationship.
- Respect semantic SQL rules.
- Do not invent tables, columns, relationships, or values.
- When a semantic rule exists for a user concept, use it.
- Explicitly alias every requested output expression using the exact requested output name and casing.
- Return exactly the columns requested by the user.
- Return SQL only.
- Do not use markdown.
- Do not explain your answer.

{metadata_context}

=== USER QUESTION ===

{question}
""".strip()


def metric_seconds(
    result,
    key,
):
    return (
        result.get(key, 0)
        / 1_000_000_000
    )


def value_equal(a, b):
    if a is None or b is None:
        return a is None and b is None

    if (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
    ):
        return math.isclose(
            float(a),
            float(b),
            rel_tol=1e-6,
            abs_tol=1e-9,
        )

    if (
        isinstance(a, str)
        and isinstance(b, str)
    ):
        return (
            a.strip().casefold()
            == b.strip().casefold()
        )

    return a == b


def row_equal(
    row_a,
    row_b,
):
    if len(row_a) != len(row_b):
        return False

    return all(
        value_equal(a, b)
        for a, b
        in zip(row_a, row_b)
    )


def rowsets_equal(
    generated_rows,
    gold_rows,
):
    """
    Compare result rows without assuming row ordering.

    This is important for queries such as Yellow-vs-Green
    where the user did not request an explicit output order.
    """

    if (
        len(generated_rows)
        != len(gold_rows)
    ):
        return False

    unmatched = [
        list(row)
        for row in gold_rows
    ]

    for generated in generated_rows:
        generated = list(generated)

        matched_index = None

        for index, gold in enumerate(
            unmatched
        ):
            if row_equal(
                generated,
                gold,
            ):
                matched_index = index
                break

        if matched_index is None:
            return False

        unmatched.pop(
            matched_index
        )

    return not unmatched


def execute_sql(
    spark,
    sql,
):
    start = time.perf_counter()

    try:
        df = spark.sql(sql)

        columns = df.columns
        rows = df.collect()

        elapsed = (
            time.perf_counter()
            - start
        )

        return {
            "success": True,
            "error": None,
            "columns": columns,
            "rows": [
                list(row)
                for row in rows
            ],
            "seconds": elapsed,
        }

    except Exception as exc:
        return {
            "success": False,
            "error": (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            "columns": [],
            "rows": [],
            "seconds": (
                time.perf_counter()
                - start
            ),
        }


def validation_dict(
    validation,
):
    return {
        "valid": validation.valid,
        "tables": validation.tables,
        "output_columns": (
            validation.output_columns
        ),
        "issues": [
            {
                "stage": issue.stage,
                "message": (
                    issue.message
                ),
            }
            for issue
            in validation.issues
        ],
    }


def generate_configuration(
    name,
    question,
    metadata_context,
):
    prompt = build_prompt(
        question,
        metadata_context,
    )

    api_result, wall = (
        ollama_generate(prompt)
    )

    sql = clean_sql(
        api_result.get(
            "response",
            "",
        )
    )

    return {
        "configuration": name,
        "metadata_characters": len(
            metadata_context
        ),
        "metadata_words": len(
            metadata_context.split()
        ),
        "prompt_characters": len(
            prompt
        ),
        "prompt_tokens": (
            api_result.get(
                "prompt_eval_count",
                0,
            )
        ),
        "generated_tokens": (
            api_result.get(
                "eval_count",
                0,
            )
        ),
        "wall_seconds": wall,
        "load_seconds": metric_seconds(
            api_result,
            "load_duration",
        ),
        "prompt_eval_seconds": (
            metric_seconds(
                api_result,
                "prompt_eval_duration",
            )
        ),
        "generation_seconds": (
            metric_seconds(
                api_result,
                "eval_duration",
            )
        ),
        "sql": sql,
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


# ---------------------------------------------------------
# Load benchmark
# ---------------------------------------------------------

benchmark = json.loads(
    BENCHMARK_PATH.read_text(
        encoding="utf-8"
    )
)

queries = benchmark["queries"]

query_filter_raw = os.environ.get(
    "BENCHMARK_QUERY_IDS",
    "",
).strip()

if query_filter_raw:
    requested_ids = [
        item.strip()
        for item
        in query_filter_raw.split(",")
        if item.strip()
    ]

    available_ids = {
        query["id"]
        for query in queries
    }

    unknown_ids = (
        set(requested_ids)
        - available_ids
    )

    if unknown_ids:
        raise RuntimeError(
            "Unknown benchmark query IDs: "
            + ", ".join(
                sorted(unknown_ids)
            )
        )

    requested_set = set(
        requested_ids
    )

    queries = [
        query
        for query in queries
        if query["id"]
        in requested_set
    ]

    selected_ids = [
        query["id"]
        for query in queries
    ]

    OUTPUT_PATH = (
        ROOT
        / "artifacts/results/"
        / (
            "canonical_abc_regression_"
            + "_".join(selected_ids)
            + ".json"
        )
    )

    print(
        "Benchmark query filter: "
        + ", ".join(selected_ids)
    )


print("=" * 78)
print("CANONICAL A/B/C TEXT-TO-SPARK-SQL BENCHMARK")
print("=" * 78)

print(
    f"Queries:       {len(queries)}"
)

print(
    f"Model:         {MODEL}"
)

print(
    f"Context size:  {NUM_CTX}"
)

print(
    f"Temperature:   {TEMPERATURE}"
)

print(
    f"Dense Top-K:   {DENSE_TOP_K}"
)


# ---------------------------------------------------------
# Build ALL B contexts first.
#
# This lets us unload embeddinggemma before Qwen is used,
# keeping the two models from competing for RAM.
# ---------------------------------------------------------

print()
print("=" * 78)
print("PRECOMPUTE RELATION-AWARE CONTEXTS")
print("=" * 78)

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

retrieval_cache = {}


for query in queries:
    start = time.perf_counter()

    retrieved = retriever.retrieve(
        query["question"],
        dense_top_k=DENSE_TOP_K,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    selected_keys = selection_keys(
        retrieved["selection"]
    )

    gold_metadata = set(
        query["required_metadata"]
    )

    hits = (
        selected_keys
        & gold_metadata
    )

    recall = (
        len(hits)
        / len(gold_metadata)
    )

    retrieval_cache[
        query["id"]
    ] = {
        "context": (
            retrieved["context"]
        ),
        "seconds": elapsed,
        "embedding_prompt_tokens": (
            retrieved[
                "dense"
            ][
                "embedding"
            ][
                "prompt_eval_count"
            ]
        ),
        "selection": {
            key: sorted(value)
            for key, value
            in retrieved[
                "selection"
            ].items()
        },
        "metadata_recall": recall,
        "metadata_hits": len(hits),
        "metadata_required": len(
            gold_metadata
        ),
    }

    print(
        f"{query['id']}: "
        f"chars="
        f"{len(retrieved['context']):4d} "
        f"recall={recall:.3f} "
        f"time={elapsed:.3f}s"
    )


subprocess.run(
    [
        "ollama",
        "stop",
        "embeddinggemma",
    ],
    check=False,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)


# ---------------------------------------------------------
# Start Spark and register curated lake
# ---------------------------------------------------------

print()
print("=" * 78)
print("START SPARK DATA LAKE")
print("=" * 78)

spark = (
    SparkSession.builder
    .appName(
        "metadata-aware-canonical-abc-benchmark"
    )
    .config(
        "spark.sql.shuffle.partitions",
        "4",
    )
    .config(
        "spark.ui.enabled",
        "false",
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel(
    "WARN"
)


for name, path in DATASETS.items():
    spark.read.parquet(
        str(path)
    ).createOrReplaceTempView(
        name
    )


validator = SparkSQLValidator(
    allowed_tables=set(
        DATASETS.keys()
    )
)

repairer = OllamaSQLRepairer(
    model=MODEL,
    num_ctx=NUM_CTX,
    temperature=TEMPERATURE,
)


# ---------------------------------------------------------
# Warm Qwen once.
# ---------------------------------------------------------

print()
print("=" * 78)
print("WARM TEXT-TO-SQL MODEL")
print("=" * 78)

warm_result, warm_wall = (
    ollama_generate(
        "Return only this SQL statement:\n"
        "SELECT 1 AS ready",
        keep_alive="30m",
    )
)

print(
    f"Warm-up wall: "
    f"{warm_wall:.2f} s"
)

print(
    f"Load time:    "
    f"{metric_seconds(warm_result, 'load_duration'):.2f} s"
)


# ---------------------------------------------------------
# Benchmark
# ---------------------------------------------------------

records = []


for index, query in enumerate(
    queries,
    start=1,
):
    query_id = query["id"]
    question = query["question"]
    expected_columns = (
        query["expected_columns"]
    )

    print()
    print("=" * 78)
    print(
        f"{query_id} — "
        f"{query['category']}"
    )
    print("=" * 78)

    print(
        f"Question: {question}"
    )

    # -----------------------------------------------------
    # Gold
    # -----------------------------------------------------

    gold_execution = execute_sql(
        spark,
        query["gold_sql"],
    )

    if not gold_execution["success"]:
        raise RuntimeError(
            f"{query_id}: gold query "
            "unexpectedly failed."
        )

    gold_rows = (
        gold_execution["rows"]
    )

    print(
        f"Gold rows: "
        f"{gold_rows[:5]}"
    )

    relation = (
        retrieval_cache[
            query_id
        ]
    )

    # -----------------------------------------------------
    # Alternate execution order to reduce systematic
    # A-before-B / B-before-A ordering bias.
    # -----------------------------------------------------

    order = (
        ["A", "B"]
        if index % 2 == 1
        else ["B", "A"]
    )

    generated = {}

    print(
        f"Generation order: "
        f"{' → '.join(order)}"
    )

    for configuration in order:

        if configuration == "A":
            generated["A"] = (
                generate_configuration(
                    name="full_catalog",
                    question=question,
                    metadata_context=(
                        full_context
                    ),
                )
            )

        else:
            generated["B"] = (
                generate_configuration(
                    name="relation_aware",
                    question=question,
                    metadata_context=(
                        relation["context"]
                    ),
                )
            )

    # -----------------------------------------------------
    # Evaluate A
    # -----------------------------------------------------

    a = generated["A"]

    a_validation = (
        validator.validate(
            sql=a["sql"],
            spark=spark,
            expected_columns=(
                expected_columns
            ),
        )
    )

    if a_validation.valid:
        a_execution = execute_sql(
            spark,
            a["sql"],
        )
    else:
        a_execution = {
            "success": False,
            "error": (
                "Not executed because "
                "validation failed."
            ),
            "columns": [],
            "rows": [],
            "seconds": 0.0,
        }

    a_correct = (
        a_execution["success"]
        and a_execution["columns"]
            == expected_columns
        and rowsets_equal(
            a_execution["rows"],
            gold_rows,
        )
    )

    # -----------------------------------------------------
    # Evaluate B
    # -----------------------------------------------------

    b = generated["B"]

    b_validation = (
        validator.validate(
            sql=b["sql"],
            spark=spark,
            expected_columns=(
                expected_columns
            ),
        )
    )

    if b_validation.valid:
        b_execution = execute_sql(
            spark,
            b["sql"],
        )
    else:
        b_execution = {
            "success": False,
            "error": (
                "Not executed because "
                "validation failed."
            ),
            "columns": [],
            "rows": [],
            "seconds": 0.0,
        }

    b_correct = (
        b_execution["success"]
        and b_execution["columns"]
            == expected_columns
        and rowsets_equal(
            b_execution["rows"],
            gold_rows,
        )
    )

    # -----------------------------------------------------
    # C starts from EXACTLY B's generated SQL.
    # -----------------------------------------------------

    repair_attempted = False
    repair_record = None

    if b_validation.valid:
        c_sql = b["sql"]
        c_validation = b_validation
        c_execution = b_execution
        c_correct = b_correct

    else:
        repair_attempted = True

        repair = repairer.repair(
            question=question,
            metadata_context=(
                relation["context"]
            ),
            invalid_sql=b["sql"],
            validation_result=(
                b_validation
            ),
            expected_columns=(
                expected_columns
            ),
            keep_alive="30m",
        )

        c_sql = repair["sql"]

        c_validation = (
            validator.validate(
                sql=c_sql,
                spark=spark,
                expected_columns=(
                    expected_columns
                ),
            )
        )

        if c_validation.valid:
            c_execution = execute_sql(
                spark,
                c_sql,
            )
        else:
            c_execution = {
                "success": False,
                "error": (
                    "Not executed because "
                    "repaired SQL failed "
                    "validation."
                ),
                "columns": [],
                "rows": [],
                "seconds": 0.0,
            }

        c_correct = (
            c_execution["success"]
            and c_execution["columns"]
                == expected_columns
            and rowsets_equal(
                c_execution["rows"],
                gold_rows,
            )
        )

        repair_record = {
            "sql": c_sql,
            "metrics": (
                repair["metrics"]
            ),
        }

    # -----------------------------------------------------
    # Print compact per-query summary
    # -----------------------------------------------------

    print()
    print("A — FULL CATALOG")

    print(
        f"  prompt tokens: "
        f"{a['prompt_tokens']}"
    )

    print(
        f"  prompt eval:   "
        f"{a['prompt_eval_seconds']:.2f}s"
    )

    print(
        f"  valid:         "
        f"{a_validation.valid}"
    )

    print(
        f"  executable:    "
        f"{a_execution['success']}"
    )

    print(
        f"  correct:       "
        f"{a_correct}"
    )

    print(
        f"  SQL: {a['sql']}"
    )

    print()
    print("B — RELATION-AWARE")

    print(
        f"  metadata chars:"
        f" {b['metadata_characters']}"
    )

    print(
        f"  metadata recall:"
        f" {relation['metadata_recall']:.3f}"
    )

    print(
        f"  prompt tokens: "
        f"{b['prompt_tokens']}"
    )

    print(
        f"  prompt eval:   "
        f"{b['prompt_eval_seconds']:.2f}s"
    )

    print(
        f"  valid:         "
        f"{b_validation.valid}"
    )

    print(
        f"  executable:    "
        f"{b_execution['success']}"
    )

    print(
        f"  correct:       "
        f"{b_correct}"
    )

    print(
        f"  SQL: {b['sql']}"
    )

    print()
    print("C — VALIDATION + ONE-SHOT REPAIR")

    print(
        f"  repair attempted:"
        f" {repair_attempted}"
    )

    print(
        f"  final valid:     "
        f"{c_validation.valid}"
    )

    print(
        f"  executable:      "
        f"{c_execution['success']}"
    )

    print(
        f"  correct:         "
        f"{c_correct}"
    )

    if repair_attempted:
        print(
            f"  repaired SQL: "
            f"{c_sql}"
        )

        print(
            f"  repair tokens: "
            f"{repair_record['metrics']['prompt_eval_count']}"
        )

    # -----------------------------------------------------
    # Store
    # -----------------------------------------------------

    records.append(
        {
            "id": query_id,
            "category": (
                query["category"]
            ),
            "question": question,
            "expected_columns": (
                expected_columns
            ),
            "generation_order": order,
            "gold": {
                "sql": query["gold_sql"],
                "columns": (
                    gold_execution[
                        "columns"
                    ]
                ),
                "rows": gold_rows,
                "spark_seconds": (
                    gold_execution[
                        "seconds"
                    ]
                ),
            },
            "retrieval": relation,
            "A": {
                **a,
                "validation": (
                    validation_dict(
                        a_validation
                    )
                ),
                "execution": (
                    a_execution
                ),
                "result_correct": (
                    a_correct
                ),
            },
            "B": {
                **b,
                "validation": (
                    validation_dict(
                        b_validation
                    )
                ),
                "execution": (
                    b_execution
                ),
                "result_correct": (
                    b_correct
                ),
            },
            "C": {
                "initial_sql": (
                    b["sql"]
                ),
                "repair_attempted": (
                    repair_attempted
                ),
                "repair": (
                    repair_record
                ),
                "final_sql": c_sql,
                "validation": (
                    validation_dict(
                        c_validation
                    )
                ),
                "execution": (
                    c_execution
                ),
                "result_correct": (
                    c_correct
                ),
            },
        }
    )


# ---------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------

def rate(values):
    values = list(values)

    if not values:
        return 0.0

    return (
        sum(bool(value) for value in values)
        / len(values)
    )


a_valid_rate = rate(
    record["A"]["validation"]["valid"]
    for record in records
)

b_valid_rate = rate(
    record["B"]["validation"]["valid"]
    for record in records
)

c_valid_rate = rate(
    record["C"]["validation"]["valid"]
    for record in records
)


a_exec_rate = rate(
    record["A"]["execution"]["success"]
    for record in records
)

b_exec_rate = rate(
    record["B"]["execution"]["success"]
    for record in records
)

c_exec_rate = rate(
    record["C"]["execution"]["success"]
    for record in records
)


a_accuracy = rate(
    record["A"]["result_correct"]
    for record in records
)

b_accuracy = rate(
    record["B"]["result_correct"]
    for record in records
)

c_accuracy = rate(
    record["C"]["result_correct"]
    for record in records
)


a_prompt_tokens = [
    record["A"]["prompt_tokens"]
    for record in records
]

b_prompt_tokens = [
    record["B"]["prompt_tokens"]
    for record in records
]


a_prompt_times = [
    record["A"][
        "prompt_eval_seconds"
    ]
    for record in records
]

b_prompt_times = [
    record["B"][
        "prompt_eval_seconds"
    ]
    for record in records
]


retrieval_times = [
    record["retrieval"]["seconds"]
    for record in records
]


repair_records = [
    record
    for record in records
    if record["C"][
        "repair_attempted"
    ]
]

repair_successes = sum(
    record["C"]["validation"]["valid"]
    for record in repair_records
)


summary = {
    "queries": len(records),

    "A": {
        "validation_rate": (
            a_valid_rate
        ),
        "execution_rate": (
            a_exec_rate
        ),
        "result_accuracy": (
            a_accuracy
        ),
        "mean_prompt_tokens": (
            statistics.mean(
                a_prompt_tokens
            )
        ),
        "mean_prompt_eval_seconds": (
            statistics.mean(
                a_prompt_times
            )
        ),
    },

    "B": {
        "validation_rate": (
            b_valid_rate
        ),
        "execution_rate": (
            b_exec_rate
        ),
        "result_accuracy": (
            b_accuracy
        ),
        "mean_prompt_tokens": (
            statistics.mean(
                b_prompt_tokens
            )
        ),
        "mean_prompt_eval_seconds": (
            statistics.mean(
                b_prompt_times
            )
        ),
        "mean_retrieval_seconds": (
            statistics.mean(
                retrieval_times
            )
        ),
    },

    "C": {
        "validation_rate": (
            c_valid_rate
        ),
        "execution_rate": (
            c_exec_rate
        ),
        "result_accuracy": (
            c_accuracy
        ),
        "repair_attempts": (
            len(repair_records)
        ),
        "repair_validation_successes": (
            repair_successes
        ),
        "repair_validation_success_rate": (
            repair_successes
            / len(repair_records)
            if repair_records
            else None
        ),
    },
}


token_reduction = (
    1
    - summary["B"][
        "mean_prompt_tokens"
    ]
    / summary["A"][
        "mean_prompt_tokens"
    ]
) * 100


prompt_time_reduction = (
    1
    - summary["B"][
        "mean_prompt_eval_seconds"
    ]
    / summary["A"][
        "mean_prompt_eval_seconds"
    ]
) * 100


summary[
    "B_vs_A"
] = {
    "mean_prompt_token_reduction_percent": (
        token_reduction
    ),
    "mean_prompt_eval_time_reduction_percent": (
        prompt_time_reduction
    ),
}


print()
print("=" * 78)
print("CANONICAL BENCHMARK SUMMARY")
print("=" * 78)

print(
    f"{'Metric':30s}"
    f"{'A Full':>12s}"
    f"{'B Retrieval':>15s}"
    f"{'C Repair':>12s}"
)

print("-" * 69)

print(
    f"{'Validation rate':30s}"
    f"{a_valid_rate:12.3f}"
    f"{b_valid_rate:15.3f}"
    f"{c_valid_rate:12.3f}"
)

print(
    f"{'Execution rate':30s}"
    f"{a_exec_rate:12.3f}"
    f"{b_exec_rate:15.3f}"
    f"{c_exec_rate:12.3f}"
)

print(
    f"{'Result accuracy':30s}"
    f"{a_accuracy:12.3f}"
    f"{b_accuracy:15.3f}"
    f"{c_accuracy:12.3f}"
)

print()
print(
    f"Mean A prompt tokens:       "
    f"{summary['A']['mean_prompt_tokens']:.1f}"
)

print(
    f"Mean B prompt tokens:       "
    f"{summary['B']['mean_prompt_tokens']:.1f}"
)

print(
    f"Mean token reduction B/A:   "
    f"{token_reduction:.2f}%"
)

print()
print(
    f"Mean A prompt eval:         "
    f"{summary['A']['mean_prompt_eval_seconds']:.2f} s"
)

print(
    f"Mean B prompt eval:         "
    f"{summary['B']['mean_prompt_eval_seconds']:.2f} s"
)

print(
    f"Mean prompt-time reduction: "
    f"{prompt_time_reduction:.2f}%"
)

print(
    f"Mean retrieval overhead:    "
    f"{summary['B']['mean_retrieval_seconds']:.3f} s"
)

print()
print(
    f"Repair attempts:            "
    f"{len(repair_records)}"
)

print(
    f"Repairs valid after attempt:"
    f" {repair_successes}/"
    f"{len(repair_records)}"
    if repair_records
    else
    "Repairs valid after attempt: N/A"
)


report = {
    "benchmark": (
        benchmark["benchmark"]
    ),
    "configuration": {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "num_ctx": NUM_CTX,
        "dense_top_k": DENSE_TOP_K,
        "full_catalog_characters": (
            len(full_context)
        ),
        "full_catalog_words": (
            len(full_context.split())
        ),
    },
    "warmup": {
        "wall_seconds": (
            warm_wall
        ),
        "load_seconds": (
            metric_seconds(
                warm_result,
                "load_duration",
            )
        ),
    },
    "summary": summary,
    "records": records,
}


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH.write_text(
    json.dumps(
        report,
        indent=2,
        default=str,
    ),
    encoding="utf-8",
)


spark.stop()


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
    "CANONICAL A/B/C BENCHMARK: COMPLETED"
)

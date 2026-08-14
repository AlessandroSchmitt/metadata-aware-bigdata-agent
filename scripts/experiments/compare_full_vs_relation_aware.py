import json
import math
import re
import subprocess
import time
import urllib.request
from pathlib import Path

from pyspark.sql import SparkSession

from metadata_agent.catalog import MetadataCatalog
from metadata_agent.retrieval import RelationAwareMetadataRetriever


ROOT = Path(__file__).resolve().parents[2]

CATALOG_PATH = ROOT / "data/catalog/metadata_catalog.sqlite"
QDRANT_PATH = ROOT / ".qdrant/metadata_catalog"

YELLOW_PATH = ROOT / "data/curated/yellow/2024-01"
GREEN_PATH = ROOT / "data/curated/green/2024-01"
ZONES_PATH = ROOT / "data/curated/zones"
WEATHER_PATH = ROOT / "data/curated/weather/2024-01"

OUTPUT_PATH = (
    ROOT
    / "artifacts/results/full_vs_relation_aware_pilot.json"
)

MODEL = "qwen2.5-coder:3b"
NUM_CTX = 4096

QUESTION = (
    "What was the average trip distance for Yellow Taxi "
    "trips during rainy hours in January 2024? "
    "Return exactly one column named average_trip_distance."
)

GOLD_SQL = """
SELECT
    AVG(y.trip_distance) AS average_trip_distance
FROM yellow_taxi y
JOIN weather_hourly w
    ON date_trunc(
        'hour',
        y.tpep_pickup_datetime
    ) = w.weather_hour
WHERE w.has_precipitation = TRUE
""".strip()


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


def ollama_generate(prompt, keep_alive="30m"):
    payload = {
        "model": MODEL,
        "stream": False,
        "keep_alive": keep_alive,
        "prompt": prompt,
        "options": {
            "temperature": 0,
            "num_ctx": NUM_CTX,
        },
    }

    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
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
            response.read().decode("utf-8")
        )

    return result, time.perf_counter() - start


def build_prompt(metadata_context):
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
- Use only tables and columns present in the metadata context.
- Respect validated cross-source relationships.
- Respect semantic SQL rules.
- Do not invent tables, columns, relationships, or values.
- When a semantic rule exists for a user concept, use it.
- Return exactly the columns requested by the user.
- Return SQL only.
- Do not use markdown.
- Do not explain your answer.

{metadata_context}

=== USER QUESTION ===

{QUESTION}
""".strip()


def execute_and_evaluate(
    spark,
    generated_sql,
    gold_value,
):
    start = time.perf_counter()

    try:
        df = spark.sql(generated_sql)

        columns = df.columns
        rows = df.collect()

        elapsed = (
            time.perf_counter()
            - start
        )

    except Exception as exc:
        return {
            "execution_success": False,
            "execution_error": (
                f"{type(exc).__name__}: {exc}"
            ),
            "columns": [],
            "generated_value": None,
            "spark_execution_seconds": (
                time.perf_counter()
                - start
            ),
            "result_correct": False,
        }

    generated_value = None
    correct = False

    if (
        columns == ["average_trip_distance"]
        and len(rows) == 1
    ):
        generated_value = rows[0][0]

        if (
            generated_value is not None
            and gold_value is not None
        ):
            correct = math.isclose(
                float(generated_value),
                float(gold_value),
                rel_tol=1e-6,
                abs_tol=1e-9,
            )

    return {
        "execution_success": True,
        "execution_error": None,
        "columns": columns,
        "generated_value": generated_value,
        "spark_execution_seconds": elapsed,
        "result_correct": correct,
    }


def run_configuration(
    name,
    metadata_context,
    spark,
    gold_value,
    retrieval_metrics=None,
):
    prompt = build_prompt(metadata_context)

    result, wall_time = ollama_generate(
        prompt
    )

    generated_sql = clean_sql(
        result.get("response", "")
    )

    execution = execute_and_evaluate(
        spark,
        generated_sql,
        gold_value,
    )

    return {
        "configuration": name,
        "metadata": {
            "characters": len(
                metadata_context
            ),
            "words": len(
                metadata_context.split()
            ),
        },
        "retrieval": retrieval_metrics,
        "prompt": {
            "characters": len(prompt),
            "prompt_eval_count": result.get(
                "prompt_eval_count",
                0,
            ),
        },
        "llm": {
            "wall_time_seconds": wall_time,
            "load_duration_seconds": (
                result.get(
                    "load_duration",
                    0,
                )
                / 1_000_000_000
            ),
            "prompt_eval_duration_seconds": (
                result.get(
                    "prompt_eval_duration",
                    0,
                )
                / 1_000_000_000
            ),
            "eval_duration_seconds": (
                result.get(
                    "eval_duration",
                    0,
                )
                / 1_000_000_000
            ),
            "generated_tokens": result.get(
                "eval_count",
                0,
            ),
        },
        "generated_sql": generated_sql,
        "execution": execution,
    }


print("=" * 70)
print("PAIRED A/B PILOT")
print("FULL CATALOG VS RELATION-AWARE RETRIEVAL")
print("=" * 70)

print()
print(f"Question: {QUESTION}")


# ---------------------------------------------------------
# Build metadata contexts
# ---------------------------------------------------------

catalog = MetadataCatalog(
    CATALOG_PATH
)

full_context = (
    catalog.render_full_catalog()
)


print()
print("=" * 70)
print("BUILD RELATION-AWARE CONTEXT")
print("=" * 70)

retriever = RelationAwareMetadataRetriever(
    catalog_path=CATALOG_PATH,
    qdrant_path=QDRANT_PATH,
)

retrieval_start = time.perf_counter()

retrieved = retriever.retrieve(
    QUESTION,
    dense_top_k=5,
)

retrieval_wall = (
    time.perf_counter()
    - retrieval_start
)

relation_context = retrieved[
    "context"
]

reduction = (
    1
    - len(relation_context)
    / len(full_context)
) * 100


print(
    f"Full Catalog chars:     "
    f"{len(full_context):,}"
)

print(
    f"Relation-aware chars:   "
    f"{len(relation_context):,}"
)

print(
    f"Reduction:              "
    f"{reduction:.2f}%"
)

print(
    f"Retrieval wall time:    "
    f"{retrieval_wall:.3f} s"
)

print(
    f"Embedding query tokens: "
    f"{retrieved['dense']['embedding']['prompt_eval_count']}"
)


# embeddinggemma is no longer needed while Qwen runs.
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
# Spark data lake
# ---------------------------------------------------------

print()
print("=" * 70)
print("START SPARK")
print("=" * 70)

spark = (
    SparkSession.builder
    .appName("metadata-aware-ab-pilot")
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

spark.read.parquet(
    str(YELLOW_PATH)
).createOrReplaceTempView(
    "yellow_taxi"
)

spark.read.parquet(
    str(GREEN_PATH)
).createOrReplaceTempView(
    "green_taxi"
)

spark.read.parquet(
    str(ZONES_PATH)
).createOrReplaceTempView(
    "taxi_zones"
)

spark.read.parquet(
    str(WEATHER_PATH)
).createOrReplaceTempView(
    "weather_hourly"
)


gold_start = time.perf_counter()

gold_value = (
    spark.sql(GOLD_SQL)
    .collect()[0][0]
)

gold_time = (
    time.perf_counter()
    - gold_start
)

print(
    f"Gold value:      "
    f"{gold_value}"
)

print(
    f"Gold Spark time: "
    f"{gold_time:.3f} s"
)


# ---------------------------------------------------------
# Warm Qwen
# ---------------------------------------------------------

print()
print("=" * 70)
print("WARMING TEXT-TO-SQL MODEL")
print("=" * 70)

warm_prompt = (
    "Return only this SQL statement:\n"
    "SELECT 1 AS ready"
)

warm_result, warm_wall = (
    ollama_generate(
        warm_prompt,
        keep_alive="30m",
    )
)

print(
    f"Warm-up wall time: "
    f"{warm_wall:.2f} s"
)

print(
    f"Warm-up load time: "
    f"{warm_result.get('load_duration', 0) / 1_000_000_000:.2f} s"
)


# ---------------------------------------------------------
# A — Full Catalog
# ---------------------------------------------------------

print()
print("=" * 70)
print("A — FULL CATALOG")
print("=" * 70)

full_result = run_configuration(
    "full_catalog",
    full_context,
    spark,
    gold_value,
)

print(
    full_result["generated_sql"]
)

print()

print(
    f"Metadata chars:       "
    f"{full_result['metadata']['characters']}"
)

print(
    f"Prompt tokens:        "
    f"{full_result['prompt']['prompt_eval_count']}"
)

print(
    f"Prompt eval time:     "
    f"{full_result['llm']['prompt_eval_duration_seconds']:.2f} s"
)

print(
    f"Generation time:      "
    f"{full_result['llm']['eval_duration_seconds']:.2f} s"
)

print(
    f"Spark execution:      "
    f"{full_result['execution']['spark_execution_seconds']:.3f} s"
)

print(
    f"Result correctness:   "
    f"{'PASS' if full_result['execution']['result_correct'] else 'FAIL'}"
)


# ---------------------------------------------------------
# B — Relation-Aware
# ---------------------------------------------------------

print()
print("=" * 70)
print("B — RELATION-AWARE")
print("=" * 70)

retrieval_metrics = {
    "wall_time_seconds": retrieval_wall,
    "embedding_prompt_tokens": (
        retrieved[
            "dense"
        ][
            "embedding"
        ][
            "prompt_eval_count"
        ]
    ),
    "dense_top_k": 5,
    "selected": {
        key: len(
            retrieved["selection"][key]
        )
        for key in [
            "datasets",
            "columns",
            "concepts",
            "relationships",
            "rules",
        ]
    },
}

relation_result = run_configuration(
    "relation_aware",
    relation_context,
    spark,
    gold_value,
    retrieval_metrics,
)

print(
    relation_result["generated_sql"]
)

print()

print(
    f"Metadata chars:       "
    f"{relation_result['metadata']['characters']}"
)

print(
    f"Prompt tokens:        "
    f"{relation_result['prompt']['prompt_eval_count']}"
)

print(
    f"Prompt eval time:     "
    f"{relation_result['llm']['prompt_eval_duration_seconds']:.2f} s"
)

print(
    f"Generation time:      "
    f"{relation_result['llm']['eval_duration_seconds']:.2f} s"
)

print(
    f"Spark execution:      "
    f"{relation_result['execution']['spark_execution_seconds']:.3f} s"
)

print(
    f"Result correctness:   "
    f"{'PASS' if relation_result['execution']['result_correct'] else 'FAIL'}"
)


# ---------------------------------------------------------
# A/B comparison
# ---------------------------------------------------------

print()
print("=" * 70)
print("A/B COMPARISON")
print("=" * 70)

full_tokens = (
    full_result["prompt"][
        "prompt_eval_count"
    ]
)

relation_tokens = (
    relation_result["prompt"][
        "prompt_eval_count"
    ]
)

token_reduction = (
    1
    - relation_tokens
    / full_tokens
) * 100


full_prompt_time = (
    full_result["llm"][
        "prompt_eval_duration_seconds"
    ]
)

relation_prompt_time = (
    relation_result["llm"][
        "prompt_eval_duration_seconds"
    ]
)

prompt_time_reduction = (
    1
    - relation_prompt_time
    / full_prompt_time
) * 100


print(
    f"{'Metric':28s}"
    f"{'Full Catalog':>15s}"
    f"{'Relation-Aware':>18s}"
)

print("-" * 61)

print(
    f"{'Metadata characters':28s}"
    f"{full_result['metadata']['characters']:15,d}"
    f"{relation_result['metadata']['characters']:18,d}"
)

print(
    f"{'Prompt tokens':28s}"
    f"{full_tokens:15,d}"
    f"{relation_tokens:18,d}"
)

print(
    f"{'Prompt evaluation (s)':28s}"
    f"{full_prompt_time:15.2f}"
    f"{relation_prompt_time:18.2f}"
)

print(
    f"{'Generated tokens':28s}"
    f"{full_result['llm']['generated_tokens']:15,d}"
    f"{relation_result['llm']['generated_tokens']:18,d}"
)

print(
    f"{'Spark execution (s)':28s}"
    f"{full_result['execution']['spark_execution_seconds']:15.3f}"
    f"{relation_result['execution']['spark_execution_seconds']:18.3f}"
)

print(
    f"{'Correct result':28s}"
    f"{str(full_result['execution']['result_correct']):>15s}"
    f"{str(relation_result['execution']['result_correct']):>18s}"
)

print()

print(
    f"Prompt token reduction: "
    f"{token_reduction:.2f}%"
)

print(
    f"Prompt-eval reduction:  "
    f"{prompt_time_reduction:.2f}%"
)

print(
    f"Retrieval overhead:     "
    f"{retrieval_wall:.3f} s"
)


# ---------------------------------------------------------
# Save experiment
# ---------------------------------------------------------

record = {
    "experiment": (
        "full_vs_relation_aware_pilot"
    ),
    "question": QUESTION,
    "model": MODEL,
    "temperature": 0,
    "num_ctx": NUM_CTX,
    "gold_sql": GOLD_SQL,
    "gold_value": gold_value,
    "gold_spark_seconds": gold_time,
    "warmup": {
        "wall_time_seconds": warm_wall,
        "load_duration_seconds": (
            warm_result.get(
                "load_duration",
                0,
            )
            / 1_000_000_000
        ),
    },
    "full_catalog": full_result,
    "relation_aware": relation_result,
    "comparison": {
        "prompt_token_reduction_percent": (
            token_reduction
        ),
        "prompt_eval_time_reduction_percent": (
            prompt_time_reduction
        ),
        "retrieval_overhead_seconds": (
            retrieval_wall
        ),
    },
}

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH.write_text(
    json.dumps(
        record,
        indent=2,
    ),
    encoding="utf-8",
)

spark.stop()


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
    "PAIRED A/B PILOT: COMPLETED"
)

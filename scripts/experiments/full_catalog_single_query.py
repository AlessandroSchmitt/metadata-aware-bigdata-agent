import json
import math
import re
import time
import urllib.request
from pathlib import Path

from pyspark.sql import SparkSession

from metadata_agent.catalog import MetadataCatalog


ROOT = Path(__file__).resolve().parents[2]

DB_PATH = (
    ROOT / "data/catalog/metadata_catalog.sqlite"
)

YELLOW_PATH = (
    ROOT / "data/curated/yellow/2024-01"
)

GREEN_PATH = (
    ROOT / "data/curated/green/2024-01"
)

ZONES_PATH = (
    ROOT / "data/curated/zones"
)

WEATHER_PATH = (
    ROOT / "data/curated/weather/2024-01"
)

OUTPUT_PATH = (
    ROOT
    / "artifacts/results/full_catalog_single_query.json"
)

MODEL = "qwen2.5-coder:3b"

QUESTION = (
    "What was the average trip distance for Yellow Taxi "
    "trips during rainy hours in January 2024? "
    "Return exactly one column named average_trip_distance."
)


def mem_available_gib():
    with open("/proc/meminfo", "r") as handle:
        values = {}

        for line in handle:
            key, value = line.split(":", 1)

            values[key] = int(
                value.strip().split()[0]
            )

    return (
        values["MemAvailable"]
        / 1024
        / 1024
    )


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


def ollama_generate(prompt):
    payload = {
        "model": MODEL,
        "stream": False,
        "keep_alive": "10m",
        "prompt": prompt,
        "options": {
            "temperature": 0,
            "num_ctx": 4096,
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
            response.read().decode("utf-8")
        )

    wall_time = (
        time.perf_counter() - start
    )

    return result, wall_time


print("=" * 70)
print("FULL CATALOG TEXT-TO-SPARK-SQL BASELINE")
print("=" * 70)

print(f"Model:       {MODEL}")
print(f"Context:     4096")
print(f"Temperature: 0")
print()
print(f"Question: {QUESTION}")

catalog = MetadataCatalog(
    DB_PATH
)

metadata_context = (
    catalog.render_full_catalog()
)

print()
print("=" * 70)
print("METADATA CONTEXT")
print("=" * 70)

print(
    f"Characters: {len(metadata_context):,}"
)

print(
    f"Words:      "
    f"{len(metadata_context.split()):,}"
)


prompt = f"""
You are an expert Spark SQL generator.

You must answer the user question using ONLY the metadata
catalog supplied below.

The datasets are already registered in Spark SQL using
exactly these table names:

- yellow_taxi
- green_taxi
- taxi_zones
- weather_hourly

Important instructions:

- Generate valid Spark SQL.
- Use only tables and columns present in the metadata catalog.
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


print()
print("=" * 70)
print("PROMPT")
print("=" * 70)

print(
    f"Prompt characters: "
    f"{len(prompt):,}"
)

print(
    f"Memory before Spark: "
    f"{mem_available_gib():.2f} GiB"
)


spark = (
    SparkSession.builder
    .appName(
        "metadata-aware-full-catalog-baseline"
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

spark.sparkContext.setLogLevel("WARN")


# ---------------------------------------------------------
# Register the real curated data lake
# ---------------------------------------------------------

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


print()
print("=" * 70)
print("SPARK DATA LAKE")
print("=" * 70)

print(
    f"Spark version:       "
    f"{spark.version}"
)

print(
    f"Memory before LLM:   "
    f"{mem_available_gib():.2f} GiB"
)


# ---------------------------------------------------------
# Gold query
# ---------------------------------------------------------

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


gold_start = time.perf_counter()

gold_rows = spark.sql(
    GOLD_SQL
).collect()

gold_execution_time = (
    time.perf_counter() - gold_start
)

gold_value = (
    gold_rows[0]["average_trip_distance"]
)


print()
print("=" * 70)
print("GOLD RESULT")
print("=" * 70)

print(GOLD_SQL)

print()
print(
    f"Gold value: "
    f"{gold_value}"
)

print(
    f"Gold Spark time: "
    f"{gold_execution_time:.3f} s"
)


# ---------------------------------------------------------
# LLM generation
# ---------------------------------------------------------

print()
print("=" * 70)
print("CALLING LLM")
print("=" * 70)

ollama_result, llm_wall_time = (
    ollama_generate(prompt)
)

raw_response = ollama_result.get(
    "response",
    "",
)

generated_sql = clean_sql(
    raw_response
)


print()
print("=" * 70)
print("GENERATED SQL")
print("=" * 70)

print(generated_sql)


load_duration = (
    ollama_result.get(
        "load_duration",
        0,
    )
    / 1_000_000_000
)

prompt_duration = (
    ollama_result.get(
        "prompt_eval_duration",
        0,
    )
    / 1_000_000_000
)

eval_duration = (
    ollama_result.get(
        "eval_duration",
        0,
    )
    / 1_000_000_000
)

prompt_eval_count = (
    ollama_result.get(
        "prompt_eval_count",
        0,
    )
)

eval_count = (
    ollama_result.get(
        "eval_count",
        0,
    )
)


print()
print("=" * 70)
print("LLM METRICS")
print("=" * 70)

print(
    f"Wall time:             "
    f"{llm_wall_time:.2f} s"
)

print(
    f"Model load:            "
    f"{load_duration:.2f} s"
)

print(
    f"Prompt evaluation:     "
    f"{prompt_duration:.2f} s"
)

print(
    f"Generation:            "
    f"{eval_duration:.2f} s"
)

print(
    f"Prompt tokens:         "
    f"{prompt_eval_count}"
)

print(
    f"Generated tokens:      "
    f"{eval_count}"
)

if eval_duration > 0:
    print(
        f"Generation speed:      "
        f"{eval_count / eval_duration:.2f} tok/s"
    )

print(
    f"Memory Spark + LLM:    "
    f"{mem_available_gib():.2f} GiB"
)


# ---------------------------------------------------------
# Execute generated SQL
# ---------------------------------------------------------

execution_success = False
execution_error = None
generated_rows = []
generated_columns = []
generated_value = None
generated_execution_time = None


print()
print("=" * 70)
print("EXECUTING GENERATED SQL")
print("=" * 70)

try:
    execution_start = (
        time.perf_counter()
    )

    generated_df = spark.sql(
        generated_sql
    )

    generated_columns = (
        generated_df.columns
    )

    generated_rows = (
        generated_df.collect()
    )

    generated_execution_time = (
        time.perf_counter()
        - execution_start
    )

    execution_success = True

    print(
        f"Columns: "
        f"{generated_columns}"
    )

    print(
        f"Rows returned: "
        f"{len(generated_rows)}"
    )

    for row in generated_rows[:10]:
        print(row)

    print(
        f"Spark execution time: "
        f"{generated_execution_time:.3f} s"
    )

except Exception as exc:
    execution_error = (
        f"{type(exc).__name__}: {exc}"
    )

    print(
        "GENERATED SQL EXECUTION: FAIL"
    )

    print(execution_error)


# ---------------------------------------------------------
# Result correctness
# ---------------------------------------------------------

result_correct = False


if (
    execution_success
    and len(generated_columns) == 1
    and generated_columns[0]
        == "average_trip_distance"
    and len(generated_rows) == 1
):
    generated_value = (
        generated_rows[0][0]
    )

    if (
        generated_value is not None
        and gold_value is not None
    ):
        result_correct = math.isclose(
            float(generated_value),
            float(gold_value),
            rel_tol=1e-6,
            abs_tol=1e-9,
        )


print()
print("=" * 70)
print("RESULT VALIDATION")
print("=" * 70)

print(
    f"Execution success: "
    f"{execution_success}"
)

print(
    f"Expected columns:   "
    f"['average_trip_distance']"
)

print(
    f"Generated columns:  "
    f"{generated_columns}"
)

print(
    f"Gold value:         "
    f"{gold_value}"
)

print(
    f"Generated value:    "
    f"{generated_value}"
)

print(
    f"RESULT CORRECTNESS: "
    f"{'PASS' if result_correct else 'FAIL'}"
)


# ---------------------------------------------------------
# Save experiment record
# ---------------------------------------------------------

record = {
    "configuration": (
        "full_catalog"
    ),
    "model": MODEL,
    "temperature": 0,
    "num_ctx": 4096,
    "question": QUESTION,
    "metadata": {
        "characters": len(
            metadata_context
        ),
        "words": len(
            metadata_context.split()
        ),
    },
    "prompt": {
        "characters": len(prompt),
        "prompt_eval_count": (
            prompt_eval_count
        ),
    },
    "generated_sql": (
        generated_sql
    ),
    "gold_sql": GOLD_SQL,
    "llm_metrics": {
        "wall_time_seconds": (
            llm_wall_time
        ),
        "load_duration_seconds": (
            load_duration
        ),
        "prompt_eval_duration_seconds": (
            prompt_duration
        ),
        "eval_duration_seconds": (
            eval_duration
        ),
        "generated_tokens": (
            eval_count
        ),
    },
    "execution": {
        "success": execution_success,
        "error": execution_error,
        "spark_execution_seconds": (
            generated_execution_time
        ),
    },
    "result": {
        "gold_value": gold_value,
        "generated_value": (
            generated_value
        ),
        "correct": result_correct,
    },
}

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT_PATH.open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        record,
        handle,
        indent=2,
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
    "FULL CATALOG BASELINE COMPLETED"
)

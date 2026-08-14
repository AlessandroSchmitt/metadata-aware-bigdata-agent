import math
import subprocess
import time
from pathlib import Path

from pyspark.sql import SparkSession

from metadata_agent.retrieval import (
    RelationAwareMetadataRetriever,
)
from metadata_agent.sql_repair import (
    OllamaSQLRepairer,
)
from metadata_agent.sql_validation import (
    SparkSQLValidator,
)


ROOT = Path(__file__).resolve().parents[2]

CATALOG_PATH = (
    ROOT / "data/catalog/metadata_catalog.sqlite"
)

QDRANT_PATH = (
    ROOT / ".qdrant/metadata_catalog"
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


QUESTION = (
    "What was the average trip distance for Yellow Taxi "
    "trips during rainy hours in January 2024? "
    "Return exactly one column named average_trip_distance."
)

EXPECTED_COLUMNS = [
    "average_trip_distance"
]


INVALID_SQL = """
SELECT
    AVG(y.trip_distance) AS average_trip_distance
FROM yellow_taxi y
JOIN weather_hourly w
    ON date_trunc(
        'hour',
        y.pickup_datetime
    ) = w.weather_hour
WHERE w.has_precipitation = TRUE
""".strip()


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


print("=" * 70)
print("ONE-SHOT SQL REPAIR TEST")
print("=" * 70)

print()
print(f"Question: {QUESTION}")


# ---------------------------------------------------------
# Retrieve compact metadata
# ---------------------------------------------------------

print()
print("=" * 70)
print("RETRIEVING METADATA")
print("=" * 70)

retriever = (
    RelationAwareMetadataRetriever(
        catalog_path=CATALOG_PATH,
        qdrant_path=QDRANT_PATH,
    )
)

retrieval_start = time.perf_counter()

retrieved = retriever.retrieve(
    QUESTION,
    dense_top_k=5,
)

retrieval_time = (
    time.perf_counter()
    - retrieval_start
)

metadata_context = (
    retrieved["context"]
)

print(
    f"Metadata characters: "
    f"{len(metadata_context):,}"
)

print(
    f"Retrieval time:      "
    f"{retrieval_time:.3f} s"
)


# embeddinggemma no longer needed.
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
# Spark
# ---------------------------------------------------------

spark = (
    SparkSession.builder
    .appName(
        "metadata-aware-one-shot-repair"
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


validator = SparkSQLValidator(
    allowed_tables={
        "yellow_taxi",
        "green_taxi",
        "taxi_zones",
        "weather_hourly",
    }
)


# ---------------------------------------------------------
# Validate intentionally invalid SQL
# ---------------------------------------------------------

print()
print("=" * 70)
print("INITIAL SQL")
print("=" * 70)

print(INVALID_SQL)


initial_validation = (
    validator.validate(
        sql=INVALID_SQL,
        spark=spark,
        expected_columns=(
            EXPECTED_COLUMNS
        ),
    )
)


print()
print("=" * 70)
print("INITIAL VALIDATION")
print("=" * 70)

print(
    f"Valid: {initial_validation.valid}"
)

for issue in (
    initial_validation.issues
):
    print(
        f"[{issue.stage}] "
        f"{issue.message.splitlines()[0]}"
    )


if initial_validation.valid:
    raise RuntimeError(
        "Controlled invalid SQL was "
        "unexpectedly accepted."
    )


# ---------------------------------------------------------
# Warm Qwen before measured repair
# ---------------------------------------------------------

print()
print("=" * 70)
print("WARMING REPAIR MODEL")
print("=" * 70)

repairer = OllamaSQLRepairer(
    model="qwen2.5-coder:3b",
    num_ctx=4096,
    temperature=0,
)

warm_result = repairer.repair(
    question="Return one column named ready.",
    metadata_context=(
        "DATASET yellow_taxi\n"
        "selected_columns:\n"
        "- VendorID:int"
    ),
    invalid_sql=(
        "SELECT bad_column AS ready "
        "FROM yellow_taxi"
    ),
    validation_result=(
        initial_validation
    ),
    expected_columns=["ready"],
)

print(
    f"Warm-up wall time: "
    f"{warm_result['metrics']['wall_time_seconds']:.2f} s"
)

print(
    f"Warm-up load time: "
    f"{warm_result['metrics']['load_duration_seconds']:.2f} s"
)


# ---------------------------------------------------------
# Exactly one repair attempt
# ---------------------------------------------------------

print()
print("=" * 70)
print("ONE-SHOT REPAIR")
print("=" * 70)

repair = repairer.repair(
    question=QUESTION,
    metadata_context=metadata_context,
    invalid_sql=INVALID_SQL,
    validation_result=(
        initial_validation
    ),
    expected_columns=(
        EXPECTED_COLUMNS
    ),
)


REPAIRED_SQL = repair["sql"]

print(REPAIRED_SQL)


print()
print("=" * 70)
print("REPAIR METRICS")
print("=" * 70)

metrics = repair["metrics"]

print(
    f"Wall time:         "
    f"{metrics['wall_time_seconds']:.2f} s"
)

print(
    f"Prompt tokens:     "
    f"{metrics['prompt_eval_count']}"
)

print(
    f"Prompt eval time:  "
    f"{metrics['prompt_eval_duration_seconds']:.2f} s"
)

print(
    f"Generated tokens:  "
    f"{metrics['eval_count']}"
)

print(
    f"Generation time:   "
    f"{metrics['eval_duration_seconds']:.2f} s"
)


# ---------------------------------------------------------
# Validate repaired SQL
# ---------------------------------------------------------

repaired_validation = (
    validator.validate(
        sql=REPAIRED_SQL,
        spark=spark,
        expected_columns=(
            EXPECTED_COLUMNS
        ),
    )
)


print()
print("=" * 70)
print("REPAIRED SQL VALIDATION")
print("=" * 70)

print(
    f"Valid:          "
    f"{repaired_validation.valid}"
)

print(
    f"Tables:         "
    f"{repaired_validation.tables}"
)

print(
    f"Output columns: "
    f"{repaired_validation.output_columns}"
)

if repaired_validation.issues:
    print("Issues:")

    for issue in (
        repaired_validation.issues
    ):
        print(
            f"  [{issue.stage}] "
            f"{issue.message.splitlines()[0]}"
        )
else:
    print("Issues:         None")


# ---------------------------------------------------------
# Only valid repaired SQL may execute
# ---------------------------------------------------------

execution_success = False
result_correct = False
generated_value = None

gold_value = (
    spark.sql(GOLD_SQL)
    .collect()[0][0]
)


if repaired_validation.valid:
    execution_start = (
        time.perf_counter()
    )

    rows = (
        spark.sql(
            REPAIRED_SQL
        )
        .collect()
    )

    execution_time = (
        time.perf_counter()
        - execution_start
    )

    execution_success = True

    if len(rows) == 1:
        generated_value = (
            rows[0][0]
        )

        if (
            generated_value is not None
            and gold_value is not None
        ):
            result_correct = (
                math.isclose(
                    float(
                        generated_value
                    ),
                    float(gold_value),
                    rel_tol=1e-6,
                    abs_tol=1e-9,
                )
            )

else:
    execution_time = None


print()
print("=" * 70)
print("RESULT EVALUATION")
print("=" * 70)

print(
    f"Repair attempts:      1"
)

print(
    f"Initial valid:        "
    f"{initial_validation.valid}"
)

print(
    f"Repaired valid:       "
    f"{repaired_validation.valid}"
)

print(
    f"Execution success:    "
    f"{execution_success}"
)

print(
    f"Gold value:           "
    f"{gold_value}"
)

print(
    f"Repaired value:       "
    f"{generated_value}"
)

print(
    f"Result correctness:   "
    f"{'PASS' if result_correct else 'FAIL'}"
)

if execution_time is not None:
    print(
        f"Spark execution:      "
        f"{execution_time:.3f} s"
    )


spark.stop()


print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

if (
    not initial_validation.valid
    and repaired_validation.valid
    and result_correct
):
    print(
        "ONE-SHOT REPAIR: PASS"
    )
else:
    print(
        "ONE-SHOT REPAIR: FAIL"
    )

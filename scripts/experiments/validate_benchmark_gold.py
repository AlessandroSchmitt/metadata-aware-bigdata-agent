import json
import time
from pathlib import Path

from pyspark.sql import SparkSession


ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_PATH = (
    ROOT / "config/benchmark_queries.json"
)

OUTPUT_PATH = (
    ROOT
    / "artifacts/benchmarks/"
      "benchmark_gold_validation.json"
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


benchmark = json.loads(
    BENCHMARK_PATH.read_text(
        encoding="utf-8"
    )
)


spark = (
    SparkSession.builder
    .appName(
        "metadata-aware-benchmark-gold-validation"
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


print("=" * 70)
print("BENCHMARK GOLD VALIDATION")
print("=" * 70)


records = []
passed = 0


for query in benchmark["queries"]:
    query_id = query["id"]

    print()
    print("=" * 70)
    print(
        f"{query_id} — "
        f"{query['category']}"
    )
    print("=" * 70)

    print(
        f"Question: "
        f"{query['question']}"
    )

    start = time.perf_counter()

    try:
        df = spark.sql(
            query["gold_sql"]
        )

        columns = df.columns
        rows = df.collect()

        elapsed = (
            time.perf_counter()
            - start
        )

        expected = (
            query["expected_columns"]
        )

        columns_match = (
            columns == expected
        )

        success = (
            columns_match
            and len(rows) >= 1
        )

        print(
            f"Columns:       "
            f"{columns}"
        )

        print(
            f"Expected:      "
            f"{expected}"
        )

        print(
            f"Rows returned: "
            f"{len(rows)}"
        )

        for row in rows[:5]:
            print(
                f"  {row}"
            )

        print(
            f"Spark time:    "
            f"{elapsed:.3f} s"
        )

        print(
            f"GOLD RESULT:   "
            f"{'PASS' if success else 'FAIL'}"
        )

        error = None

    except Exception as exc:
        elapsed = (
            time.perf_counter()
            - start
        )

        success = False
        columns = []
        rows = []

        error = (
            f"{type(exc).__name__}: {exc}"
        )

        print(
            f"GOLD RESULT:   FAIL"
        )

        print(error)

    if success:
        passed += 1

    records.append(
        {
            "id": query_id,
            "category": (
                query["category"]
            ),
            "success": success,
            "columns": columns,
            "rows": [
                list(row)
                for row in rows
            ],
            "spark_seconds": (
                elapsed
            ),
            "error": error,
        }
    )


spark.stop()


report = {
    "benchmark": (
        benchmark["benchmark"]
    ),
    "queries_total": len(
        benchmark["queries"]
    ),
    "queries_passed": passed,
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


print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Gold queries passed: "
    f"{passed}/"
    f"{len(benchmark['queries'])}"
)

print()
print(
    OUTPUT_PATH.relative_to(ROOT)
)


print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

if passed != len(
    benchmark["queries"]
):
    raise RuntimeError(
        "One or more benchmark gold "
        "queries failed validation."
    )

print(
    "BENCHMARK GOLD VALIDATION: PASS"
)

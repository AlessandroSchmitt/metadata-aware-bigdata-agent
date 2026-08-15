import json
import os
import shutil
import sqlite3
import statistics
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DEVELOPMENT_TAG = "development-freeze-v1"

CATALOG_PATH = (
    ROOT / "data/catalog/metadata_catalog.sqlite"
)

DATA_ROOT = (
    ROOT / "data/benchmarks/volume_scaling"
)

OUTPUT_PATH = (
    ROOT
    / "artifacts/benchmarks/"
      "data_volume_scaling_benchmark.json"
)

PARTITIONS = 2
REPETITIONS = 3

TARGET_SCALES = [
    ("100k", 100_000),
    ("500k", 500_000),
    ("1m", 1_000_000),
    ("full", None),
]

DRY_RUN = (
    os.environ.get(
        "DATA_VOLUME_DRY_RUN",
        "0",
    )
    == "1"
)


WORKLOADS = {
    "scan_aggregation": """
        SELECT
            COUNT(*) AS trip_count,
            AVG(trip_distance)
                AS average_trip_distance,
            SUM(trip_distance)
                AS total_trip_distance
        FROM yellow_scale
    """,

    "pickup_zone_grouping": """
        SELECT
            PULocationID,
            COUNT(*) AS trip_count,
            AVG(trip_distance)
                AS average_trip_distance
        FROM yellow_scale
        GROUP BY PULocationID
        ORDER BY trip_count DESC
        LIMIT 10
    """,

    "weather_temporal_join": """
        SELECT
            COUNT(*) AS trip_count,
            AVG(y.trip_distance)
                AS average_trip_distance
        FROM yellow_scale AS y
        JOIN weather_hourly AS w
          ON date_trunc(
                 'hour',
                 y.tpep_pickup_datetime
             ) = w.weather_hour
        WHERE w.has_precipitation = TRUE
    """,
}


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


def catalog_dataset(name):
    conn = sqlite3.connect(
        CATALOG_PATH
    )

    try:
        row = conn.execute(
            """
            SELECT
                name,
                path,
                row_count
            FROM datasets
            WHERE name = ?
            """,
            (name,),
        ).fetchone()

    finally:
        conn.close()

    if row is None:
        raise RuntimeError(
            f"Dataset not found: {name}"
        )

    return {
        "name": row[0],
        "path": row[1],
        "row_count": row[2],
    }


yellow = catalog_dataset(
    "yellow_taxi"
)

weather = catalog_dataset(
    "weather_hourly"
)

full_rows = int(
    yellow["row_count"]
)

scales = []

for label, target in TARGET_SCALES:
    rows = (
        full_rows
        if target is None
        else target
    )

    if rows > full_rows:
        raise RuntimeError(
            f"Scale {label} requests "
            f"{rows:,} rows but only "
            f"{full_rows:,} exist."
        )

    scales.append(
        {
            "label": label,
            "rows": rows,
        }
    )


print("=" * 78)
print("SPARK DATA-VOLUME SCALING BENCHMARK")
print("=" * 78)

print(
    f"Development freeze: "
    f"{DEVELOPMENT_TAG} "
    f"({git_tag_commit(DEVELOPMENT_TAG)[:7]})"
)

print(
    f"Yellow source:       "
    f"{yellow['path']}"
)

print(
    f"Full Yellow rows:    "
    f"{full_rows:,}"
)

print(
    f"Weather source:      "
    f"{weather['path']}"
)

print(
    f"Materialized files:  "
    f"{PARTITIONS} partitions/scale"
)

print(
    f"Repetitions:         "
    f"{REPETITIONS}"
)

print()
print("SCALES")

for scale in scales:
    print(
        f"  {scale['label']:>4s}: "
        f"{scale['rows']:>10,d} rows"
    )

print()
print("WORKLOADS")

for name in WORKLOADS:
    print(
        f"  - {name}"
    )


if DRY_RUN:
    print()
    print("=" * 78)
    print("FINAL RESULT")
    print("=" * 78)

    print(
        "DATA-VOLUME SCALING BENCHMARK "
        "DRY-RUN: PASS"
    )

    raise SystemExit(0)


from pyspark.sql import SparkSession


spark = (
    SparkSession.builder
    .appName(
        "metadata-aware-data-volume-scaling"
    )
    .config(
        "spark.ui.enabled",
        "false",
    )
    .config(
        "spark.sql.shuffle.partitions",
        "4",
    )
    .config(
        "spark.sql.adaptive.enabled",
        "true",
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel(
    "WARN"
)


yellow_source = (
    ROOT / yellow["path"]
)

weather_source = (
    ROOT / weather["path"]
)


if not yellow_source.exists():
    raise RuntimeError(
        f"Yellow path missing: "
        f"{yellow_source}"
    )

if not weather_source.exists():
    raise RuntimeError(
        f"Weather path missing: "
        f"{weather_source}"
    )


# ---------------------------------------------------------
# Materialize each scale outside timed benchmark runs.
# Every scale uses exactly PARTITIONS parquet files.
# ---------------------------------------------------------

print()
print("=" * 78)
print("MATERIALIZE BENCHMARK DATASETS")
print("=" * 78)

DATA_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

source_df = spark.read.parquet(
    str(yellow_source)
)

observed_full_rows = (
    source_df.count()
)

if observed_full_rows != full_rows:
    raise RuntimeError(
        "Catalog/full-data row count mismatch: "
        f"catalog={full_rows}, "
        f"observed={observed_full_rows}"
    )


materialization = {}


for scale in scales:

    label = scale["label"]
    rows = scale["rows"]

    output = (
        DATA_ROOT / label
    )

    start = time.perf_counter()

    if output.exists():
        shutil.rmtree(
            output
        )

    (
        source_df
        .limit(rows)
        .repartition(PARTITIONS)
        .write
        .mode("overwrite")
        .parquet(str(output))
    )

    verification = (
        spark.read
        .parquet(str(output))
        .count()
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    if verification != rows:
        raise RuntimeError(
            f"{label}: expected "
            f"{rows:,} rows, "
            f"found {verification:,}."
        )

    parquet_files = list(
        output.glob("*.parquet")
    )

    bytes_total = sum(
        path.stat().st_size
        for path in parquet_files
    )

    materialization[label] = {
        "rows": rows,
        "parquet_files": len(
            parquet_files
        ),
        "bytes": bytes_total,
        "seconds": elapsed,
    }

    print(
        f"{label:>4s} | "
        f"rows={rows:>10,d} | "
        f"files={len(parquet_files):2d} | "
        f"size={bytes_total / 1024 / 1024:8.2f} MiB | "
        f"prepare={elapsed:7.2f}s"
    )


# Weather is constant across all scales.
weather_df = (
    spark.read
    .parquet(str(weather_source))
)

weather_df.createOrReplaceTempView(
    "weather_hourly"
)


# ---------------------------------------------------------
# Benchmark.
#
# No DataFrame is cached.
# Spark cache is cleared before every measured query.
# Scale order alternates by repetition to reduce systematic
# order effects.
# ---------------------------------------------------------

records = []


for repetition in range(
    1,
    REPETITIONS + 1,
):

    if repetition % 2 == 1:
        ordered_scales = scales
    else:
        ordered_scales = list(
            reversed(scales)
        )

    print()
    print("=" * 78)
    print(
        f"REPETITION "
        f"{repetition}/{REPETITIONS}"
    )
    print("=" * 78)

    for scale in ordered_scales:

        label = scale["label"]
        rows = scale["rows"]

        data_path = (
            DATA_ROOT / label
        )

        yellow_df = (
            spark.read
            .parquet(str(data_path))
        )

        yellow_df.createOrReplaceTempView(
            "yellow_scale"
        )

        for workload_name, sql in (
            WORKLOADS.items()
        ):
            spark.catalog.clearCache()

            start = time.perf_counter()

            result_rows = (
                spark.sql(sql).collect()
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            records.append(
                {
                    "repetition": repetition,
                    "scale": label,
                    "rows": rows,
                    "workload": (
                        workload_name
                    ),
                    "seconds": elapsed,
                    "result_preview": [
                        list(row)
                        for row
                        in result_rows[:10]
                    ],
                }
            )

            print(
                f"{label:>4s} | "
                f"{workload_name:24s} | "
                f"rows={rows:>10,d} | "
                f"time={elapsed:8.3f}s"
            )


# ---------------------------------------------------------
# Aggregate results.
# ---------------------------------------------------------

summary = {}


for scale in scales:

    label = scale["label"]
    rows = scale["rows"]

    summary[label] = {
        "rows": rows,
        "materialization": (
            materialization[label]
        ),
        "workloads": {},
    }

    for workload_name in WORKLOADS:

        subset = [
            record["seconds"]
            for record in records
            if (
                record["scale"] == label
                and record["workload"]
                == workload_name
            )
        ]

        median_seconds = (
            statistics.median(
                subset
            )
        )

        mean_seconds = (
            statistics.mean(
                subset
            )
        )

        rows_per_second = (
            rows / median_seconds
            if median_seconds > 0
            else None
        )

        summary[label][
            "workloads"
        ][workload_name] = {
            "runs": len(subset),
            "seconds": subset,
            "mean_seconds": (
                mean_seconds
            ),
            "median_seconds": (
                median_seconds
            ),
            "rows_per_second_at_median": (
                rows_per_second
            ),
        }


# Runtime growth relative to 100k.
baseline_label = "100k"

for workload_name in WORKLOADS:

    baseline = (
        summary[baseline_label]
        ["workloads"]
        [workload_name]
        ["median_seconds"]
    )

    for scale in scales:

        label = scale["label"]

        current = (
            summary[label]
            ["workloads"]
            [workload_name]
            ["median_seconds"]
        )

        summary[label][
            "workloads"
        ][workload_name][
            "runtime_factor_vs_100k"
        ] = (
            current / baseline
        )


output = {
    "benchmark": {
        "name": (
            "Spark Yellow Taxi "
            "Data-Volume Scaling Benchmark"
        ),
        "purpose": (
            "Measure Spark execution behavior "
            "as physical taxi-trip row volume "
            "increases while schema, workload, "
            "parallelism, and metadata remain fixed."
        ),
        "development_freeze": (
            DEVELOPMENT_TAG
        ),
        "development_commit": (
            git_tag_commit(
                DEVELOPMENT_TAG
            )
        ),
        "source_dataset": (
            "yellow_taxi"
        ),
        "full_source_rows": (
            full_rows
        ),
        "scales": scales,
        "partitions_per_scale": (
            PARTITIONS
        ),
        "repetitions": (
            REPETITIONS
        ),
        "workloads": list(
            WORKLOADS.keys()
        ),
    },

    "environment": {
        "spark_version": (
            spark.version
        ),
        "master": (
            spark.sparkContext.master
        ),
        "default_parallelism": (
            spark.sparkContext
            .defaultParallelism
        ),
        "shuffle_partitions": (
            spark.conf.get(
                "spark.sql.shuffle.partitions"
            )
        ),
        "adaptive_enabled": (
            spark.conf.get(
                "spark.sql.adaptive.enabled"
            )
        ),
    },

    "materialization": (
        materialization
    ),

    "summary": summary,

    "records": records,
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
print("DATA-VOLUME SCALING SUMMARY")
print("=" * 78)

for workload_name in WORKLOADS:

    print()
    print(workload_name)

    print(
        "Scale        Rows    "
        "Median(s)    Rows/s    "
        "Factor-vs-100k"
    )

    for scale in scales:

        label = scale["label"]

        item = (
            summary[label]
            ["workloads"]
            [workload_name]
        )

        print(
            f"{label:>4s}  "
            f"{scale['rows']:>10,d}  "
            f"{item['median_seconds']:9.3f}  "
            f"{item['rows_per_second_at_median']:10.0f}  "
            f"{item['runtime_factor_vs_100k']:14.2f}x"
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
    "DATA-VOLUME SCALING BENCHMARK: "
    "COMPLETED"
)

spark.stop()

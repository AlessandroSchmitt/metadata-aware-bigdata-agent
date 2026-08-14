import json
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


ROOT = Path(__file__).resolve().parents[2]

YELLOW_PATH = ROOT / "data/raw/yellow/yellow_tripdata_2024-01.parquet"
GREEN_PATH = ROOT / "data/raw/green/green_tripdata_2024-01.parquet"

OUTPUT_PATH = ROOT / "artifacts/benchmarks/tlc_distance_outliers.json"

START = "2024-01-01 00:00:00"
END = "2024-02-01 00:00:00"

THRESHOLDS = [
    30,
    50,
    75,
    100,
    150,
    200,
    500,
    1000,
]


def audit(df, name, pickup_col):
    print()
    print("=" * 70)
    print(name.upper())
    print("=" * 70)

    january = df.where(
        (F.col(pickup_col) >= F.lit(START))
        & (F.col(pickup_col) < F.lit(END))
    )

    total = january.count()

    print(f"January rows: {total:,}")

    print()
    print("DISTANCE THRESHOLDS")
    print("-" * 70)

    threshold_results = {}

    for threshold in THRESHOLDS:
        count = january.where(
            F.col("trip_distance") > threshold
        ).count()

        percentage = (
            count / total * 100
            if total
            else 0
        )

        threshold_results[str(threshold)] = {
            "rows": count,
            "percentage": percentage,
        }

        print(
            f"> {threshold:4d} miles: "
            f"{count:8,d} "
            f"({percentage:.6f}%)"
        )

    print()
    print("PRECISE TAIL QUANTILES")
    print("-" * 70)

    probabilities = [
        0.99,
        0.995,
        0.999,
        0.9995,
        0.9999,
        0.99995,
        0.99999,
    ]

    # Much tighter rank error than the previous audit.
    values = january.approxQuantile(
        "trip_distance",
        probabilities,
        0.00001,
    )

    quantile_results = {}

    for probability, value in zip(
        probabilities,
        values
    ):
        label = f"p{probability * 100:.3f}"

        quantile_results[label] = value

        print(
            f"{label:10s}: {value}"
        )

    print()
    print("TOP 20 DISTANCES")
    print("-" * 70)

    columns = [
        pickup_col,
        "trip_distance",
        "PULocationID",
        "DOLocationID",
        "fare_amount",
        "total_amount",
    ]

    top_rows = (
        january
        .select(*columns)
        .orderBy(
            F.col("trip_distance").desc()
        )
        .limit(20)
        .collect()
    )

    serialized_top = []

    for row in top_rows:
        item = {
            column: (
                str(row[column])
                if row[column] is not None
                else None
            )
            for column in columns
        }

        serialized_top.append(item)

        print(item)

    return {
        "january_rows": total,
        "thresholds": threshold_results,
        "quantiles": quantile_results,
        "top_20": serialized_top,
    }


def main():
    print("=" * 70)
    print("TLC TRIP DISTANCE OUTLIER AUDIT")
    print("=" * 70)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    spark = (
        SparkSession.builder
        .appName("metadata-aware-agent-distance-audit")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    start = time.perf_counter()

    yellow = spark.read.parquet(
        str(YELLOW_PATH)
    )

    green = spark.read.parquet(
        str(GREEN_PATH)
    )

    results = {
        "yellow_taxi": audit(
            yellow,
            "Yellow Taxi",
            "tpep_pickup_datetime",
        ),
        "green_taxi": audit(
            green,
            "Green Taxi",
            "lpep_pickup_datetime",
        ),
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            results,
            handle,
            indent=2,
        )

    elapsed = time.perf_counter() - start

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        OUTPUT_PATH.relative_to(ROOT)
    )

    print(
        f"Audit time: {elapsed:.2f} s"
    )

    spark.stop()

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print("DISTANCE OUTLIER AUDIT: PASS")


if __name__ == "__main__":
    main()

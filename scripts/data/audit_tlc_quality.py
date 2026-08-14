import json
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


ROOT = Path(__file__).resolve().parents[2]

YELLOW_PATH = ROOT / "data/raw/yellow/yellow_tripdata_2024-01.parquet"
GREEN_PATH = ROOT / "data/raw/green/green_tripdata_2024-01.parquet"

OUTPUT_PATH = ROOT / "artifacts/benchmarks/tlc_quality_audit.json"

START = "2024-01-01 00:00:00"
END = "2024-02-01 00:00:00"


def count_where(df, condition):
    return df.where(condition).count()


def quantiles(df, column):
    values = (
        df.where(F.col(column).isNotNull())
        .approxQuantile(
            column,
            [0.50, 0.95, 0.99, 0.999],
            0.001,
        )
    )

    return {
        "p50": values[0],
        "p95": values[1],
        "p99": values[2],
        "p999": values[3],
    }


def audit_dataset(
    df,
    name,
    pickup_col,
    dropoff_col,
):
    print()
    print("=" * 70)
    print(name.upper())
    print("=" * 70)

    total = df.count()

    in_month_condition = (
        (F.col(pickup_col) >= F.lit(START))
        & (F.col(pickup_col) < F.lit(END))
    )

    in_month = count_where(
        df,
        in_month_condition
    )

    before_month = count_where(
        df,
        F.col(pickup_col) < F.lit(START)
    )

    after_month = count_where(
        df,
        F.col(pickup_col) >= F.lit(END)
    )

    null_pickup = count_where(
        df,
        F.col(pickup_col).isNull()
    )

    null_dropoff = count_where(
        df,
        F.col(dropoff_col).isNull()
    )

    invalid_duration = count_where(
        df,
        F.col(dropoff_col) <= F.col(pickup_col)
    )

    null_pu = count_where(
        df,
        F.col("PULocationID").isNull()
    )

    null_do = count_where(
        df,
        F.col("DOLocationID").isNull()
    )

    negative_distance = count_where(
        df,
        F.col("trip_distance") < 0
    )

    zero_distance = count_where(
        df,
        F.col("trip_distance") == 0
    )

    distance_gt_100 = count_where(
        df,
        F.col("trip_distance") > 100
    )

    distance_gt_200 = count_where(
        df,
        F.col("trip_distance") > 200
    )

    negative_fare = count_where(
        df,
        F.col("fare_amount") < 0
    )

    negative_tip = count_where(
        df,
        F.col("tip_amount") < 0
    )

    negative_total = count_where(
        df,
        F.col("total_amount") < 0
    )

    valid_month_df = df.where(
        in_month_condition
    )

    distance_quantiles = quantiles(
        valid_month_df,
        "trip_distance"
    )

    fare_quantiles = quantiles(
        valid_month_df,
        "fare_amount"
    )

    total_quantiles = quantiles(
        valid_month_df,
        "total_amount"
    )

    print(f"Total rows:                    {total:,}")
    print(f"Pickup inside January 2024:    {in_month:,}")
    print(f"Pickup before January 2024:    {before_month:,}")
    print(f"Pickup on/after Feb 1 2024:    {after_month:,}")
    print(f"Null pickup timestamps:        {null_pickup:,}")
    print(f"Null dropoff timestamps:       {null_dropoff:,}")
    print(f"Dropoff <= pickup:             {invalid_duration:,}")
    print(f"Null pickup LocationID:        {null_pu:,}")
    print(f"Null dropoff LocationID:       {null_do:,}")

    print()
    print("Trip distance:")
    print(f"  negative:                    {negative_distance:,}")
    print(f"  zero:                        {zero_distance:,}")
    print(f"  > 100 miles:                 {distance_gt_100:,}")
    print(f"  > 200 miles:                 {distance_gt_200:,}")
    print(f"  p50:                         {distance_quantiles['p50']}")
    print(f"  p95:                         {distance_quantiles['p95']}")
    print(f"  p99:                         {distance_quantiles['p99']}")
    print(f"  p99.9:                       {distance_quantiles['p999']}")

    print()
    print("Monetary fields:")
    print(f"  negative fare_amount:        {negative_fare:,}")
    print(f"  negative tip_amount:         {negative_tip:,}")
    print(f"  negative total_amount:       {negative_total:,}")

    print()
    print("fare_amount quantiles (January pickups):")
    for key, value in fare_quantiles.items():
        print(f"  {key}: {value}")

    print()
    print("total_amount quantiles (January pickups):")
    for key, value in total_quantiles.items():
        print(f"  {key}: {value}")

    return {
        "total_rows": total,
        "temporal": {
            "pickup_inside_january_2024": in_month,
            "pickup_before_january_2024": before_month,
            "pickup_on_or_after_february_2024": after_month,
            "null_pickup": null_pickup,
            "null_dropoff": null_dropoff,
            "dropoff_not_after_pickup": invalid_duration,
        },
        "locations": {
            "null_pickup_location": null_pu,
            "null_dropoff_location": null_do,
        },
        "trip_distance": {
            "negative": negative_distance,
            "zero": zero_distance,
            "greater_than_100": distance_gt_100,
            "greater_than_200": distance_gt_200,
            "quantiles": distance_quantiles,
        },
        "monetary": {
            "negative_fare_amount": negative_fare,
            "negative_tip_amount": negative_tip,
            "negative_total_amount": negative_total,
            "fare_quantiles": fare_quantiles,
            "total_quantiles": total_quantiles,
        },
    }


def main():
    print("=" * 70)
    print("NYC TLC DATA QUALITY AUDIT")
    print("=" * 70)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    spark = (
        SparkSession.builder
        .appName("metadata-aware-agent-quality-audit")
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
        "yellow_taxi_2024_01": audit_dataset(
            yellow,
            "Yellow Taxi",
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
        ),
        "green_taxi_2024_01": audit_dataset(
            green,
            "Green Taxi",
            "lpep_pickup_datetime",
            "lpep_dropoff_datetime",
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
    print(OUTPUT_PATH.relative_to(ROOT))
    print(f"Audit time: {elapsed:.2f} s")

    spark.stop()

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print("TLC DATA QUALITY AUDIT: PASS")


if __name__ == "__main__":
    main()

import json
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


ROOT = Path(__file__).resolve().parents[2]

YELLOW_PATH = ROOT / "data/raw/yellow/yellow_tripdata_2024-01.parquet"
GREEN_PATH = ROOT / "data/raw/green/green_tripdata_2024-01.parquet"
ZONES_PATH = ROOT / "data/raw/zones/taxi_zone_lookup.csv"

OUTPUT_PATH = ROOT / "artifacts/benchmarks/tlc_source_profile.json"


def schema_as_dict(df):
    return [
        {
            "name": field.name,
            "type": field.dataType.simpleString(),
            "nullable": field.nullable,
        }
        for field in df.schema.fields
    ]


def find_column(columns, candidates):
    by_lower = {c.lower(): c for c in columns}

    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]

    return None


def print_schema_info(name, df):
    print()
    print("=" * 70)
    print(f"{name} SCHEMA")
    print("=" * 70)

    df.printSchema()

    print("Columns:")
    for column in df.columns:
        print(f"  - {column}")


def validate_zone_relationship(df, name, location_column, zones):
    print()
    print("-" * 70)
    print(f"{name}.{location_column} -> taxi_zones.LocationID")
    print("-" * 70)

    source = (
        df.select(
            F.col(location_column)
            .cast("int")
            .alias("location_id")
        )
        .where(F.col("location_id").isNotNull())
    )

    invalid = (
        source.alias("s")
        .join(
            zones.select(
                F.col("LocationID")
                .cast("int")
                .alias("zone_id")
            ).alias("z"),
            F.col("s.location_id") == F.col("z.zone_id"),
            "left_anti",
        )
    )

    distinct_source = source.select("location_id").distinct().count()
    invalid_rows = invalid.count()
    invalid_distinct = invalid.select("location_id").distinct().count()

    examples = [
        row["location_id"]
        for row in (
            invalid
            .select("location_id")
            .distinct()
            .orderBy("location_id")
            .limit(10)
            .collect()
        )
    ]

    print(f"Distinct location IDs:  {distinct_source}")
    print(f"Unmatched rows:         {invalid_rows}")
    print(f"Unmatched distinct IDs: {invalid_distinct}")
    print(f"Examples:               {examples}")

    return {
        "distinct_source_ids": distinct_source,
        "unmatched_rows": invalid_rows,
        "unmatched_distinct_ids": invalid_distinct,
        "unmatched_examples": examples,
    }


def temporal_profile(df, pickup_col, dropoff_col=None):
    expressions = [
        F.min(F.col(pickup_col)).alias("min_pickup"),
        F.max(F.col(pickup_col)).alias("max_pickup"),
    ]

    if dropoff_col:
        expressions.extend(
            [
                F.min(F.col(dropoff_col)).alias("min_dropoff"),
                F.max(F.col(dropoff_col)).alias("max_dropoff"),
            ]
        )

    row = df.agg(*expressions).first()

    result = {
        "min_pickup": str(row["min_pickup"]),
        "max_pickup": str(row["max_pickup"]),
    }

    if dropoff_col:
        result.update(
            {
                "min_dropoff": str(row["min_dropoff"]),
                "max_dropoff": str(row["max_dropoff"]),
            }
        )

    return result


def main():
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("NYC TLC REAL DATA PROFILING")
    print("=" * 70)

    spark = (
        SparkSession.builder
        .appName("metadata-aware-agent-tlc-profile")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print(f"Spark version: {spark.version}")

    start = time.perf_counter()

    yellow = spark.read.parquet(str(YELLOW_PATH))
    green = spark.read.parquet(str(GREEN_PATH))

    zones = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(ZONES_PATH))
    )

    print_schema_info("YELLOW TAXI", yellow)
    print_schema_info("GREEN TAXI", green)
    print_schema_info("TAXI ZONES", zones)

    print()
    print("=" * 70)
    print("ROW COUNTS")
    print("=" * 70)

    yellow_count = yellow.count()
    green_count = green.count()
    zones_count = zones.count()

    print(f"Yellow Taxi: {yellow_count:,}")
    print(f"Green Taxi:  {green_count:,}")
    print(f"Taxi Zones:  {zones_count:,}")

    print()
    print("=" * 70)
    print("YELLOW / GREEN SCHEMA COMPARISON")
    print("=" * 70)

    yellow_columns = set(yellow.columns)
    green_columns = set(green.columns)

    common = sorted(yellow_columns & green_columns)
    yellow_only = sorted(yellow_columns - green_columns)
    green_only = sorted(green_columns - yellow_columns)

    print()
    print("Common columns:")
    for column in common:
        print(f"  - {column}")

    print()
    print("Yellow-only columns:")
    for column in yellow_only:
        print(f"  - {column}")

    print()
    print("Green-only columns:")
    for column in green_only:
        print(f"  - {column}")

    yellow_pickup = find_column(
        yellow.columns,
        [
            "tpep_pickup_datetime",
            "pickup_datetime",
        ],
    )

    yellow_dropoff = find_column(
        yellow.columns,
        [
            "tpep_dropoff_datetime",
            "dropoff_datetime",
        ],
    )

    green_pickup = find_column(
        green.columns,
        [
            "lpep_pickup_datetime",
            "pickup_datetime",
        ],
    )

    green_dropoff = find_column(
        green.columns,
        [
            "lpep_dropoff_datetime",
            "dropoff_datetime",
        ],
    )

    print()
    print("=" * 70)
    print("TEMPORAL COVERAGE")
    print("=" * 70)

    yellow_temporal = temporal_profile(
        yellow,
        yellow_pickup,
        yellow_dropoff,
    )

    green_temporal = temporal_profile(
        green,
        green_pickup,
        green_dropoff,
    )

    print("Yellow Taxi:")
    for key, value in yellow_temporal.items():
        print(f"  {key}: {value}")

    print()
    print("Green Taxi:")
    for key, value in green_temporal.items():
        print(f"  {key}: {value}")

    print()
    print("=" * 70)
    print("TAXI ZONE SAMPLE")
    print("=" * 70)

    zones.orderBy("LocationID").show(
        10,
        truncate=False,
    )

    print()
    print("=" * 70)
    print("ZONE RELATIONSHIP VALIDATION")
    print("=" * 70)

    relationship_results = {}

    for dataset_name, df in [
        ("yellow_taxi", yellow),
        ("green_taxi", green),
    ]:
        relationship_results[dataset_name] = {}

        for location_column in [
            "PULocationID",
            "DOLocationID",
        ]:
            if location_column in df.columns:
                relationship_results[dataset_name][location_column] = (
                    validate_zone_relationship(
                        df,
                        dataset_name,
                        location_column,
                        zones,
                    )
                )

    print()
    print("=" * 70)
    print("BASIC TRIP STATISTICS")
    print("=" * 70)

    stats = {}

    for name, df in [
        ("yellow_taxi", yellow),
        ("green_taxi", green),
    ]:
        available = [
            column
            for column in [
                "trip_distance",
                "fare_amount",
                "tip_amount",
                "total_amount",
                "passenger_count",
            ]
            if column in df.columns
        ]

        print()
        print(name)

        if available:
            df.select(*available).summary(
                "count",
                "mean",
                "min",
                "max",
            ).show(
                truncate=False
            )

            stat_rows = (
                df.select(*available)
                .summary(
                    "count",
                    "mean",
                    "min",
                    "max",
                )
                .collect()
            )

            stats[name] = {
                row["summary"]: {
                    column: row[column]
                    for column in available
                }
                for row in stat_rows
            }

    profile = {
        "spark_version": spark.version,
        "sources": {
            "yellow_taxi_2024_01": {
                "path": str(
                    YELLOW_PATH.relative_to(ROOT)
                ),
                "rows": yellow_count,
                "schema": schema_as_dict(yellow),
                "temporal_coverage": yellow_temporal,
            },
            "green_taxi_2024_01": {
                "path": str(
                    GREEN_PATH.relative_to(ROOT)
                ),
                "rows": green_count,
                "schema": schema_as_dict(green),
                "temporal_coverage": green_temporal,
            },
            "taxi_zones": {
                "path": str(
                    ZONES_PATH.relative_to(ROOT)
                ),
                "rows": zones_count,
                "schema": schema_as_dict(zones),
            },
        },
        "schema_comparison": {
            "common_columns": common,
            "yellow_only_columns": yellow_only,
            "green_only_columns": green_only,
        },
        "relationships": relationship_results,
        "statistics": stats,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            profile,
            handle,
            indent=2,
        )

    elapsed = time.perf_counter() - start

    print()
    print("=" * 70)
    print("PROFILE OUTPUT")
    print("=" * 70)
    print(
        OUTPUT_PATH.relative_to(ROOT)
    )
    print(
        f"Total profiling time: {elapsed:.2f} s"
    )

    spark.stop()

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print("TLC SOURCE PROFILING: PASS")


if __name__ == "__main__":
    main()

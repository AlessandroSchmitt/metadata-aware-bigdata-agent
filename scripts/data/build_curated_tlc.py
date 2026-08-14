import json
import shutil
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "config" / "curation_rules.json"

YELLOW_RAW = (
    ROOT / "data/raw/yellow/yellow_tripdata_2024-01.parquet"
)

GREEN_RAW = (
    ROOT / "data/raw/green/green_tripdata_2024-01.parquet"
)

ZONES_RAW = (
    ROOT / "data/raw/zones/taxi_zone_lookup.csv"
)

YELLOW_CURATED = (
    ROOT / "data/curated/yellow/2024-01"
)

GREEN_CURATED = (
    ROOT / "data/curated/green/2024-01"
)

ZONES_CURATED = (
    ROOT / "data/curated/zones"
)

REPORT_PATH = (
    ROOT / "artifacts/benchmarks/tlc_curation_report.json"
)


def count_condition(df, condition):
    return df.where(condition).count()


def prepare_taxi(
    df,
    dataset_name,
    pickup_col,
    dropoff_col,
    valid_zone_ids,
    start,
    end,
    min_distance,
    max_distance,
):
    print()
    print("=" * 70)
    print(dataset_name.upper())
    print("=" * 70)

    input_rows = df.count()

    rules = {
        "pickup_in_target_period": (
            F.col(pickup_col).isNotNull()
            & (F.col(pickup_col) >= F.lit(start))
            & (F.col(pickup_col) < F.lit(end))
        ),
        "dropoff_not_null": (
            F.col(dropoff_col).isNotNull()
        ),
        "dropoff_after_pickup": (
            F.col(dropoff_col) > F.col(pickup_col)
        ),
        "valid_pickup_zone": (
            F.col("PULocationID").isNotNull()
            & F.col("PULocationID").isin(valid_zone_ids)
        ),
        "valid_dropoff_zone": (
            F.col("DOLocationID").isNotNull()
            & F.col("DOLocationID").isin(valid_zone_ids)
        ),
        "valid_trip_distance": (
            F.col("trip_distance").isNotNull()
            & (F.col("trip_distance") >= min_distance)
            & (F.col("trip_distance") <= max_distance)
        ),
    }

    violations = {}

    for name, condition in rules.items():
        count = count_condition(
            df,
            ~condition,
        )

        violations[name] = count

        print(
            f"{name:30s} "
            f"violations = {count:,}"
        )

    combined_condition = None

    for condition in rules.values():
        if combined_condition is None:
            combined_condition = condition
        else:
            combined_condition = (
                combined_condition & condition
            )

    curated = df.where(
        combined_condition
    )

    output_rows = curated.count()

    removed_rows = input_rows - output_rows

    removed_percentage = (
        removed_rows / input_rows * 100
        if input_rows
        else 0
    )

    print()
    print(f"Input rows:              {input_rows:,}")
    print(f"Curated rows:            {output_rows:,}")
    print(f"Removed rows:            {removed_rows:,}")
    print(
        f"Removed percentage:      "
        f"{removed_percentage:.6f}%"
    )

    return curated, {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "removed_rows": removed_rows,
        "removed_percentage": removed_percentage,
        "rule_violations": violations,
    }


def remove_output(path):
    if path.exists():
        shutil.rmtree(path)


def main():
    print("=" * 70)
    print("BUILD TLC CURATED DATA LAKE")
    print("=" * 70)

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = json.load(handle)

    period = config["target_period"]
    quality = config["quality_rules"]

    start = period["start"]
    end = period["end"]

    min_distance = quality[
        "min_trip_distance"
    ]

    max_distance = quality[
        "max_trip_distance"
    ]

    spark = (
        SparkSession.builder
        .appName("metadata-aware-agent-curation")
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

    start_time = time.perf_counter()

    yellow = spark.read.parquet(
        str(YELLOW_RAW)
    )

    green = spark.read.parquet(
        str(GREEN_RAW)
    )

    zones = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(ZONES_RAW))
    )

    zone_rows = zones.count()

    distinct_zone_rows = (
        zones.select("LocationID")
        .distinct()
        .count()
    )

    if zone_rows != distinct_zone_rows:
        raise RuntimeError(
            "Taxi Zone LocationID is not unique."
        )

    valid_zone_ids = [
        row["LocationID"]
        for row in zones.select(
            "LocationID"
        ).collect()
        if row["LocationID"] is not None
    ]

    print()
    print("=" * 70)
    print("TAXI ZONES")
    print("=" * 70)

    print(f"Rows:                    {zone_rows:,}")
    print(
        f"Distinct LocationIDs:    "
        f"{distinct_zone_rows:,}"
    )

    yellow_curated, yellow_report = (
        prepare_taxi(
            yellow,
            "Yellow Taxi",
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            valid_zone_ids,
            start,
            end,
            min_distance,
            max_distance,
        )
    )

    green_curated, green_report = (
        prepare_taxi(
            green,
            "Green Taxi",
            "lpep_pickup_datetime",
            "lpep_dropoff_datetime",
            valid_zone_ids,
            start,
            end,
            min_distance,
            max_distance,
        )
    )

    print()
    print("=" * 70)
    print("WRITING CURATED DATA")
    print("=" * 70)

    remove_output(YELLOW_CURATED)
    remove_output(GREEN_CURATED)
    remove_output(ZONES_CURATED)

    yellow_curated.write.mode(
        "overwrite"
    ).parquet(
        str(YELLOW_CURATED)
    )

    green_curated.write.mode(
        "overwrite"
    ).parquet(
        str(GREEN_CURATED)
    )

    zones.write.mode(
        "overwrite"
    ).parquet(
        str(ZONES_CURATED)
    )

    print(
        "Yellow: ",
        YELLOW_CURATED.relative_to(ROOT),
    )

    print(
        "Green:  ",
        GREEN_CURATED.relative_to(ROOT),
    )

    print(
        "Zones:  ",
        ZONES_CURATED.relative_to(ROOT),
    )

    print()
    print("=" * 70)
    print("READ-BACK VALIDATION")
    print("=" * 70)

    yellow_check = spark.read.parquet(
        str(YELLOW_CURATED)
    )

    green_check = spark.read.parquet(
        str(GREEN_CURATED)
    )

    zones_check = spark.read.parquet(
        str(ZONES_CURATED)
    )

    yellow_check_count = (
        yellow_check.count()
    )

    green_check_count = (
        green_check.count()
    )

    zones_check_count = (
        zones_check.count()
    )

    assert (
        yellow_check_count
        == yellow_report["output_rows"]
    )

    assert (
        green_check_count
        == green_report["output_rows"]
    )

    assert (
        zones_check_count
        == zone_rows
    )

    assert (
        yellow_check.columns
        == yellow.columns
    )

    assert (
        green_check.columns
        == green.columns
    )

    print(
        f"Yellow read-back rows:   "
        f"{yellow_check_count:,}"
    )

    print(
        f"Green read-back rows:    "
        f"{green_check_count:,}"
    )

    print(
        f"Zone read-back rows:     "
        f"{zones_check_count:,}"
    )

    print(
        "Original Yellow schema preserved: YES"
    )

    print(
        "Original Green schema preserved:  YES"
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "period": period,
        "quality_rules": quality,
        "yellow_taxi": yellow_report,
        "green_taxi": green_report,
        "taxi_zones": {
            "rows": zone_rows,
            "distinct_location_ids": (
                distinct_zone_rows
            ),
        },
        "schema_preservation": {
            "yellow": (
                yellow_check.columns
                == yellow.columns
            ),
            "green": (
                green_check.columns
                == green.columns
            ),
        },
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print("=" * 70)
    print("REPORT")
    print("=" * 70)

    print(
        REPORT_PATH.relative_to(ROOT)
    )

    print(
        f"Total curation time:     "
        f"{elapsed:.2f} s"
    )

    spark.stop()

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print("TLC CURATED DATA LAKE: PASS")


if __name__ == "__main__":
    main()

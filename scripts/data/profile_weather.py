import csv
import json
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


ROOT = Path(__file__).resolve().parents[2]

WEATHER_PATH = (
    ROOT / "data/raw/weather/72505394728_2024.csv"
)

OUTPUT_PATH = (
    ROOT / "artifacts/benchmarks/weather_source_profile.json"
)

CANDIDATE_COLUMNS = [
    "STATION",
    "DATE",
    "LATITUDE",
    "LONGITUDE",
    "ELEVATION",
    "NAME",
    "REPORT_TYPE",
    "SOURCE",
    "HourlyDryBulbTemperature",
    "HourlyRelativeHumidity",
    "HourlyPrecipitation",
    "HourlyPresentWeatherType",
    "HourlyWindDirection",
    "HourlyWindSpeed",
    "HourlyVisibility",
]


def main():
    print("=" * 70)
    print("NOAA WEATHER SOURCE PROFILING")
    print("=" * 70)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Physical CSV header
    # -----------------------------------------------------

    with WEATHER_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.reader(handle)
        header = next(reader)

    print(f"Physical CSV columns: {len(header)}")

    print()
    print("Candidate columns present:")

    present = []

    for column in CANDIDATE_COLUMNS:
        exists = column in header

        print(
            f"  {column:30s} "
            f"{'YES' if exists else 'NO'}"
        )

        if exists:
            present.append(column)

    # -----------------------------------------------------
    # Spark
    # -----------------------------------------------------

    spark = (
        SparkSession.builder
        .appName("metadata-aware-agent-weather-profile")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    start = time.perf_counter()

    weather = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .option("quote", '"')
        .option("escape", '"')
        .csv(str(WEATHER_PATH))
    )

    print()
    print("=" * 70)
    print("SPARK SOURCE")
    print("=" * 70)

    total_rows = weather.count()

    print(f"Rows: {total_rows:,}")
    print(f"Columns: {len(weather.columns)}")

    if "DATE" not in weather.columns:
        raise RuntimeError(
            "Expected DATE column not found."
        )

    parsed = weather.withColumn(
        "_parsed_date",
        F.to_timestamp("DATE"),
    )

    parsed_count = parsed.where(
        F.col("_parsed_date").isNotNull()
    ).count()

    invalid_date_count = (
        total_rows - parsed_count
    )

    temporal = (
        parsed.agg(
            F.min("_parsed_date").alias("min_date"),
            F.max("_parsed_date").alias("max_date"),
        )
        .first()
    )

    january = parsed.where(
        (F.col("_parsed_date") >= F.lit("2024-01-01 00:00:00"))
        & (F.col("_parsed_date") < F.lit("2024-02-01 00:00:00"))
    )

    january_rows = january.count()

    print(f"Parsed DATE rows: {parsed_count:,}")
    print(f"Invalid DATE rows: {invalid_date_count:,}")
    print(f"Min DATE: {temporal['min_date']}")
    print(f"Max DATE: {temporal['max_date']}")
    print(f"January rows: {january_rows:,}")

    print()
    print("=" * 70)
    print("JANUARY CANDIDATE COLUMN PROFILE")
    print("=" * 70)

    profile_columns = {}

    for column in present:
        if column == "DATE":
            continue

        non_null = january.where(
            F.col(column).isNotNull()
            & (F.trim(F.col(column)) != "")
        ).count()

        distinct = (
            january
            .select(column)
            .where(
                F.col(column).isNotNull()
                & (F.trim(F.col(column)) != "")
            )
            .distinct()
            .count()
        )

        profile_columns[column] = {
            "non_empty_rows": non_null,
            "distinct_values": distinct,
        }

        print(
            f"{column:30s} "
            f"non-empty={non_null:6,d} "
            f"distinct={distinct:6,d}"
        )

    print()
    print("=" * 70)
    print("JANUARY SAMPLE")
    print("=" * 70)

    selected = [
        column
        for column in CANDIDATE_COLUMNS
        if column in january.columns
    ]

    january.select(
        *selected
    ).orderBy(
        "_parsed_date"
    ).show(
        20,
        truncate=False,
        vertical=True,
    )

    print()
    print("=" * 70)
    print("PRECIPITATION RAW VALUES")
    print("=" * 70)

    precipitation_values = []

    if "HourlyPrecipitation" in january.columns:
        rows = (
            january
            .select("HourlyPrecipitation")
            .where(
                F.col("HourlyPrecipitation").isNotNull()
                & (
                    F.trim(
                        F.col("HourlyPrecipitation")
                    ) != ""
                )
            )
            .groupBy("HourlyPrecipitation")
            .count()
            .orderBy(
                F.col("count").desc()
            )
            .limit(30)
            .collect()
        )

        for row in rows:
            item = {
                "value": row["HourlyPrecipitation"],
                "count": row["count"],
            }

            precipitation_values.append(item)
            print(item)

    profile = {
        "source": "NOAA Local Climatological Data",
        "station_id": "72505394728",
        "station_name": "NY CITY CENTRAL PARK, NY US",
        "raw_path": str(
            WEATHER_PATH.relative_to(ROOT)
        ),
        "total_rows": total_rows,
        "physical_columns": len(header),
        "columns": header,
        "date_profile": {
            "parsed_rows": parsed_count,
            "invalid_rows": invalid_date_count,
            "min": str(temporal["min_date"]),
            "max": str(temporal["max_date"]),
            "january_rows": january_rows,
        },
        "candidate_columns": profile_columns,
        "precipitation_raw_values": (
            precipitation_values
        ),
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
    print("OUTPUT")
    print("=" * 70)

    print(
        OUTPUT_PATH.relative_to(ROOT)
    )

    print(
        f"Profiling time: {elapsed:.2f} s"
    )

    spark.stop()

    print()
    print("=" * 70)
    print("WEATHER SOURCE PROFILING: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()

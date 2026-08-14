import json
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = (
    ROOT / "data/raw/weather/72505394728_2024.csv"
)

CONFIG_PATH = (
    ROOT / "config/weather_curation.json"
)

OUTPUT_PATH = (
    ROOT / "data/curated/weather/2024-01"
)

REPORT_PATH = (
    ROOT / "artifacts/benchmarks/weather_curation_report.json"
)

NUMERIC_PATTERN = r"^[+-]?([0-9]+(\.[0-9]+)?|\.[0-9]+)$"


def numeric_column(column_name, data_type="double"):
    value = F.trim(F.col(column_name))

    return (
        F.when(
            value.rlike(NUMERIC_PATTERN),
            value.cast(data_type),
        )
        .otherwise(
            F.lit(None).cast(data_type)
        )
    )


def main():
    print("=" * 70)
    print("BUILD CURATED NOAA HOURLY WEATHER")
    print("=" * 70)

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = json.load(handle)

    start_string = config[
        "target_period"
    ]["start"]

    end_string = config[
        "target_period"
    ]["end"]

    preferred_report = config[
        "hourly_selection"
    ]["preferred_report_type"]

    start_dt = datetime.fromisoformat(
        start_string
    )

    end_dt = datetime.fromisoformat(
        end_string
    )

    expected_hours = int(
        (end_dt - start_dt).total_seconds()
        / 3600
    )

    spark = (
        SparkSession.builder
        .appName(
            "metadata-aware-agent-weather-curation"
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

    start_time = time.perf_counter()

    raw = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .option("quote", '"')
        .option("escape", '"')
        .csv(str(RAW_PATH))
    )

    parsed = (
        raw
        .withColumn(
            "_observation_time",
            F.to_timestamp_ntz("DATE"),
        )
        .where(
            (F.col("_observation_time") >= F.lit(start_string))
            & (F.col("_observation_time") < F.lit(end_string))
        )
        .withColumn(
            "weather_hour",
            F.date_trunc(
                "hour",
                F.col("_observation_time"),
            ),
        )
    )

    raw_january_rows = parsed.count()

    print()
    print("=" * 70)
    print("RAW JANUARY OBSERVATIONS")
    print("=" * 70)

    print(
        f"Rows:                    "
        f"{raw_january_rows:,}"
    )

    report_type_rows = (
        parsed
        .groupBy("REPORT_TYPE")
        .count()
        .orderBy(
            F.col("count").desc()
        )
        .collect()
    )

    report_type_counts = {
        (
            row["REPORT_TYPE"]
            if row["REPORT_TYPE"] is not None
            else "NULL"
        ): row["count"]
        for row in report_type_rows
    }

    print()
    print("Report types:")

    for key, value in report_type_counts.items():
        print(
            f"  {key:10s} {value:6,d}"
        )

    hourly_counts = (
        parsed
        .groupBy("weather_hour")
        .count()
    )

    observed_raw_hours = (
        hourly_counts.count()
    )

    multiple_observation_hours = (
        hourly_counts
        .where(
            F.col("count") > 1
        )
        .count()
    )

    print()
    print(
        f"Distinct raw hours:      "
        f"{observed_raw_hours:,}"
    )

    print(
        f"Hours with >1 report:    "
        f"{multiple_observation_hours:,}"
    )

    # -----------------------------------------------------
    # Select one representative observation per hour.
    #
    # 1. Prefer FM-15 routine report.
    # 2. Within the same priority, use the latest report.
    # -----------------------------------------------------

    selection_window = (
        Window
        .partitionBy("weather_hour")
        .orderBy(
            F.when(
                F.col("REPORT_TYPE")
                == preferred_report,
                F.lit(0),
            )
            .otherwise(F.lit(1))
            .asc(),
            F.col(
                "_observation_time"
            ).desc(),
        )
    )

    selected = (
        parsed
        .withColumn(
            "_selection_rank",
            F.row_number().over(
                selection_window
            ),
        )
        .where(
            F.col("_selection_rank") == 1
        )
    )

    selected_rows = selected.count()

    selected_preferred = (
        selected
        .where(
            F.col("REPORT_TYPE")
            == preferred_report
        )
        .count()
    )

    selected_fallback = (
        selected_rows
        - selected_preferred
    )

    print()
    print("=" * 70)
    print("HOURLY OBSERVATION SELECTION")
    print("=" * 70)

    print(
        f"Selected observations:   "
        f"{selected_rows:,}"
    )

    print(
        f"Preferred FM-15:         "
        f"{selected_preferred:,}"
    )

    print(
        f"Fallback observations:   "
        f"{selected_fallback:,}"
    )

    precipitation_raw = (
        F.trim(
            F.col("HourlyPrecipitation")
        )
    )

    precipitation_numeric = (
        F.when(
            precipitation_raw.rlike(
                NUMERIC_PATTERN
            ),
            precipitation_raw.cast(
                "double"
            ),
        )
        .otherwise(
            F.lit(None).cast("double")
        )
    )

    precipitation_trace = (
        F.when(
            precipitation_raw.isNull()
            | (precipitation_raw == ""),
            F.lit(None).cast("boolean"),
        )
        .when(
            precipitation_raw == "T",
            F.lit(True),
        )
        .otherwise(
            F.lit(False),
        )
    )

    has_precipitation = (
        F.when(
            precipitation_raw.isNull()
            | (precipitation_raw == ""),
            F.lit(None).cast("boolean"),
        )
        .when(
            precipitation_raw == "T",
            F.lit(True),
        )
        .when(
            precipitation_raw.rlike(
                NUMERIC_PATTERN
            ),
            precipitation_numeric > 0,
        )
        .otherwise(
            F.lit(None).cast("boolean"),
        )
    )

    selected_projected = (
        selected
        .select(
            "weather_hour",
            F.col(
                "_observation_time"
            ).alias(
                "observation_time"
            ),
            "STATION",
            "NAME",
            "REPORT_TYPE",
            "SOURCE",
            numeric_column(
                "LATITUDE"
            ).alias(
                "latitude"
            ),
            numeric_column(
                "LONGITUDE"
            ).alias(
                "longitude"
            ),
            numeric_column(
                "ELEVATION"
            ).alias(
                "elevation_ft"
            ),
            F.col(
                "HourlyDryBulbTemperature"
            ).alias(
                "temperature_f_raw"
            ),
            numeric_column(
                "HourlyDryBulbTemperature"
            ).alias(
                "temperature_f"
            ),
            F.col(
                "HourlyRelativeHumidity"
            ).alias(
                "relative_humidity_raw"
            ),
            numeric_column(
                "HourlyRelativeHumidity",
                "int",
            ).alias(
                "relative_humidity"
            ),
            F.col(
                "HourlyPrecipitation"
            ).alias(
                "precipitation_raw"
            ),
            precipitation_numeric.alias(
                "precipitation_in"
            ),
            precipitation_trace.alias(
                "precipitation_trace"
            ),
            has_precipitation.alias(
                "has_precipitation"
            ),
            "HourlyPresentWeatherType",
            "HourlyWindDirection",
            F.col(
                "HourlyWindSpeed"
            ).alias(
                "wind_speed_raw"
            ),
            numeric_column(
                "HourlyWindSpeed"
            ).alias(
                "wind_speed_mph"
            ),
            F.col(
                "HourlyVisibility"
            ).alias(
                "visibility_raw"
            ),
            numeric_column(
                "HourlyVisibility"
            ).alias(
                "visibility_miles"
            ),
        )
    )

    # -----------------------------------------------------
    # Complete January hour spine.
    # Missing weather observations remain NULL rather than
    # being interpreted as dry conditions.
    # -----------------------------------------------------

    hour_rows = []

    current = start_dt

    while current < end_dt:
        hour_rows.append(
            (
                current.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            )
        )
        current += timedelta(hours=1)

    hour_spine = (
        spark.createDataFrame(
            hour_rows,
            ["_weather_hour_string"],
        )
        .withColumn(
            "weather_hour",
            F.to_timestamp_ntz(
                "_weather_hour_string"
            ),
        )
        .drop(
            "_weather_hour_string"
        )
    )

    curated = (
        hour_spine
        .join(
            selected_projected,
            on="weather_hour",
            how="left",
        )
        .orderBy("weather_hour")
    )

    curated_rows = curated.count()

    distinct_hours = (
        curated
        .select("weather_hour")
        .distinct()
        .count()
    )

    observed_hours = (
        curated
        .where(
            F.col(
                "observation_time"
            ).isNotNull()
        )
        .count()
    )

    missing_observation_hours = (
        curated_rows
        - observed_hours
    )

    known_precip_hours = (
        curated
        .where(
            F.col(
                "has_precipitation"
            ).isNotNull()
        )
        .count()
    )

    rainy_hours = (
        curated
        .where(
            F.col(
                "has_precipitation"
            ) == True
        )
        .count()
    )

    dry_hours = (
        curated
        .where(
            F.col(
                "has_precipitation"
            ) == False
        )
        .count()
    )

    trace_hours = (
        curated
        .where(
            F.col(
                "precipitation_trace"
            ) == True
        )
        .count()
    )

    unknown_precip_hours = (
        curated_rows
        - known_precip_hours
    )

    print()
    print("=" * 70)
    print("CURATED HOURLY COVERAGE")
    print("=" * 70)

    print(
        f"Expected January hours:  "
        f"{expected_hours:,}"
    )

    print(
        f"Curated rows:            "
        f"{curated_rows:,}"
    )

    print(
        f"Distinct weather hours:  "
        f"{distinct_hours:,}"
    )

    print(
        f"Observed weather hours:  "
        f"{observed_hours:,}"
    )

    print(
        f"Missing weather hours:   "
        f"{missing_observation_hours:,}"
    )

    print()
    print("Precipitation classification:")

    print(
        f"  known hours:            "
        f"{known_precip_hours:,}"
    )

    print(
        f"  rainy hours:            "
        f"{rainy_hours:,}"
    )

    print(
        f"  dry hours:              "
        f"{dry_hours:,}"
    )

    print(
        f"  trace hours:            "
        f"{trace_hours:,}"
    )

    print(
        f"  unknown hours:          "
        f"{unknown_precip_hours:,}"
    )

    if curated_rows != expected_hours:
        raise RuntimeError(
            "Unexpected number of curated "
            "weather hours."
        )

    if distinct_hours != expected_hours:
        raise RuntimeError(
            "weather_hour is not unique."
        )

    # -----------------------------------------------------
    # Write
    # -----------------------------------------------------

    if OUTPUT_PATH.exists():
        shutil.rmtree(
            OUTPUT_PATH
        )

    curated.coalesce(1).write.mode(
        "overwrite"
    ).parquet(
        str(OUTPUT_PATH)
    )

    read_back = spark.read.parquet(
        str(OUTPUT_PATH)
    )

    read_back_rows = (
        read_back.count()
    )

    read_back_distinct = (
        read_back
        .select("weather_hour")
        .distinct()
        .count()
    )

    assert (
        read_back_rows
        == expected_hours
    )

    assert (
        read_back_distinct
        == expected_hours
    )

    print()
    print("=" * 70)
    print("READ-BACK VALIDATION")
    print("=" * 70)

    print(
        f"Rows:                    "
        f"{read_back_rows:,}"
    )

    print(
        f"Distinct weather_hour:   "
        f"{read_back_distinct:,}"
    )

    print()
    print("Curated schema:")

    read_back.printSchema()

    print()
    print("Sample precipitation hours:")

    (
        read_back
        .where(
            F.col(
                "has_precipitation"
            ) == True
        )
        .select(
            "weather_hour",
            "observation_time",
            "REPORT_TYPE",
            "precipitation_raw",
            "precipitation_in",
            "precipitation_trace",
            "has_precipitation",
            "HourlyPresentWeatherType",
        )
        .orderBy("weather_hour")
        .show(
            15,
            truncate=False,
        )
    )

    report = {
        "station_id": (
            config["station_id"]
        ),
        "station_name": (
            config["station_name"]
        ),
        "period": (
            config["target_period"]
        ),
        "raw_january_rows": (
            raw_january_rows
        ),
        "report_type_counts": (
            report_type_counts
        ),
        "raw_distinct_hours": (
            observed_raw_hours
        ),
        "hours_with_multiple_reports": (
            multiple_observation_hours
        ),
        "selection": {
            "selected_observations": (
                selected_rows
            ),
            "preferred_fm15": (
                selected_preferred
            ),
            "fallback": (
                selected_fallback
            ),
        },
        "coverage": {
            "expected_hours": (
                expected_hours
            ),
            "curated_rows": (
                curated_rows
            ),
            "observed_hours": (
                observed_hours
            ),
            "missing_observation_hours": (
                missing_observation_hours
            ),
        },
        "precipitation": {
            "known_hours": (
                known_precip_hours
            ),
            "rainy_hours": (
                rainy_hours
            ),
            "dry_hours": (
                dry_hours
            ),
            "trace_hours": (
                trace_hours
            ),
            "unknown_hours": (
                unknown_precip_hours
            ),
        },
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    print("OUTPUT")
    print("=" * 70)

    print(
        OUTPUT_PATH.relative_to(ROOT)
    )

    print(
        REPORT_PATH.relative_to(ROOT)
    )

    print(
        f"Total curation time: "
        f"{elapsed:.2f} s"
    )

    spark.stop()

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        "CURATED NOAA HOURLY WEATHER: PASS"
    )


if __name__ == "__main__":
    main()

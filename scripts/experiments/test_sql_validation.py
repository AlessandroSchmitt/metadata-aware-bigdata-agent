from pathlib import Path

from pyspark.sql import SparkSession

from metadata_agent.sql_validation import (
    SparkSQLValidator,
)


ROOT = Path(__file__).resolve().parents[2]

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


spark = (
    SparkSession.builder
    .appName(
        "metadata-aware-sql-validation"
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


cases = [
    {
        "name": "valid_cross_source_query",
        "sql": """
            SELECT
                AVG(y.trip_distance)
                    AS average_trip_distance
            FROM yellow_taxi y
            JOIN weather_hourly w
              ON date_trunc(
                    'hour',
                    y.tpep_pickup_datetime
                 ) = w.weather_hour
            WHERE w.has_precipitation = TRUE
        """,
        "expected_columns": [
            "average_trip_distance"
        ],
        "expected_valid": True,
        "expected_stage": None,
    },

    {
        "name": "invalid_column",
        "sql": """
            SELECT
                AVG(y.trip_distance)
                    AS average_trip_distance
            FROM yellow_taxi y
            JOIN weather_hourly w
              ON date_trunc(
                    'hour',
                    y.pickup_datetime
                 ) = w.weather_hour
            WHERE w.has_precipitation = TRUE
        """,
        "expected_columns": [
            "average_trip_distance"
        ],
        "expected_valid": False,
        "expected_stage": (
            "spark_analysis"
        ),
    },

    {
        "name": "invalid_table",
        "sql": """
            SELECT
                AVG(y.trip_distance)
                    AS average_trip_distance
            FROM yellow_taxi y
            JOIN weather w
              ON date_trunc(
                    'hour',
                    y.tpep_pickup_datetime
                 ) = w.weather_hour
        """,
        "expected_columns": [
            "average_trip_distance"
        ],
        "expected_valid": False,
        "expected_stage": "catalog",
    },

    {
        "name": "forbidden_write",
        "sql": """
            DROP TABLE yellow_taxi
        """,
        "expected_columns": None,
        "expected_valid": False,
        "expected_stage": (
            "structure"
        ),
    },

    {
        "name": "wrong_output_contract",
        "sql": """
            SELECT
                AVG(trip_distance)
                    AS avg_distance
            FROM yellow_taxi
        """,
        "expected_columns": [
            "average_trip_distance"
        ],
        "expected_valid": False,
        "expected_stage": (
            "output_contract"
        ),
    },
]


print("=" * 70)
print("SPARK SQL VALIDATOR TEST")
print("=" * 70)


passed = 0


for index, case in enumerate(
    cases,
    start=1,
):
    print()
    print("=" * 70)
    print(
        f"CASE {index}: "
        f"{case['name']}"
    )
    print("=" * 70)

    result = validator.validate(
        sql=case["sql"],
        spark=spark,
        expected_columns=(
            case["expected_columns"]
        ),
    )

    print(
        f"Valid:          "
        f"{result.valid}"
    )

    print(
        f"Tables:         "
        f"{result.tables}"
    )

    print(
        f"Output columns: "
        f"{result.output_columns}"
    )

    if result.issues:
        print("Issues:")

        for issue in result.issues:
            first_line = (
                issue.message
                .splitlines()[0]
            )

            print(
                f"  - [{issue.stage}] "
                f"{first_line}"
            )
    else:
        print(
            "Issues:         None"
        )

    stages = {
        issue.stage
        for issue in result.issues
    }

    valid_match = (
        result.valid
        == case["expected_valid"]
    )

    if (
        case["expected_stage"]
        is None
    ):
        stage_match = (
            len(stages) == 0
        )
    else:
        stage_match = (
            case[
                "expected_stage"
            ]
            in stages
        )

    case_pass = (
        valid_match
        and stage_match
    )

    print(
        f"Expected:       "
        f"{case['expected_valid']}"
    )

    print(
        "CASE RESULT:    "
        + (
            "PASS"
            if case_pass
            else "FAIL"
        )
    )

    if case_pass:
        passed += 1


spark.stop()


print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Cases passed: "
    f"{passed}/{len(cases)}"
)


print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

if passed != len(cases):
    raise RuntimeError(
        "SQL validator test failed."
    )

print(
    "SPARK SQL VALIDATOR: PASS"
)

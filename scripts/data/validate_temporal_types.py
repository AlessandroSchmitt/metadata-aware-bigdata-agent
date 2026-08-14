from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


ROOT = Path(__file__).resolve().parents[2]

spark = (
    SparkSession.builder
    .appName("weather-temporal-validation")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

yellow = spark.read.parquet(
    str(ROOT / "data/curated/yellow/2024-01")
)

green = spark.read.parquet(
    str(ROOT / "data/curated/green/2024-01")
)

weather = spark.read.parquet(
    str(ROOT / "data/curated/weather/2024-01")
)

print()
print("=" * 70)
print("TEMPORAL TYPES")
print("=" * 70)

checks = [
    (
        yellow,
        "tpep_pickup_datetime",
    ),
    (
        green,
        "lpep_pickup_datetime",
    ),
    (
        weather,
        "weather_hour",
    ),
    (
        weather,
        "observation_time",
    ),
]

types = []

for df, column in checks:
    data_type = (
        df.schema[column]
        .dataType
        .simpleString()
    )

    types.append(data_type)

    print(
        f"{column:30s} "
        f"{data_type}"
    )

print()
print("=" * 70)
print("MISSING WEATHER HOURS")
print("=" * 70)

missing = (
    weather
    .where(
        F.col("observation_time").isNull()
    )
    .select("weather_hour")
    .orderBy("weather_hour")
    .collect()
)

print(f"Missing count: {len(missing)}")

for row in missing:
    print(
        f"  {row['weather_hour']}"
    )

print()
print("=" * 70)
print("TIMESTAMP TYPE CONSISTENCY")
print("=" * 70)

unique_types = set(types)

print(
    "Types found:",
    sorted(unique_types),
)

if unique_types == {"timestamp_ntz"}:
    print("TIMESTAMP TYPE CONSISTENCY: PASS")
else:
    print("TIMESTAMP TYPE CONSISTENCY: FAIL")

spark.stop()

import json
import re
import time
import urllib.request

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    TimestampType,
)
from datetime import datetime


MODEL = "qwen2.5-coder:3b"


def mem_available_gib():
    with open("/proc/meminfo", "r") as f:
        values = {}
        for line in f:
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])

    return values["MemAvailable"] / 1024 / 1024


def clean_sql(text):
    text = text.strip()

    text = re.sub(
        r"^```(?:sql)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    return text.strip()


print("======================================")
print("FULL VERTICAL SLICE")
print("======================================")

print(f"Memory before Spark: {mem_available_gib():.2f} GiB")

spark = (
    SparkSession.builder
    .appName("metadata-aware-agent-vertical-slice")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print()
print("======================================")
print("SPARK ENVIRONMENT")
print("======================================")

print(f"Spark version:          {spark.version}")
print(f"Spark master:           {spark.sparkContext.master}")
print(f"Memory after Spark:     {mem_available_gib():.2f} GiB")


# ---------------------------------------------------------
# Synthetic Taxi Zones
# ---------------------------------------------------------

zones_schema = StructType([
    StructField("LocationID", IntegerType(), False),
    StructField("Borough", StringType(), False),
    StructField("Zone", StringType(), False),
])

zones_data = [
    (1, "Manhattan", "Midtown"),
    (2, "Queens", "JFK Airport"),
    (3, "Queens", "Astoria"),
    (4, "Manhattan", "Harlem"),
]

zones_df = spark.createDataFrame(
    zones_data,
    schema=zones_schema
)

zones_df.createOrReplaceTempView("taxi_zones")


# ---------------------------------------------------------
# Synthetic Yellow Taxi trips
# ---------------------------------------------------------

trip_schema = StructType([
    StructField("pickup_datetime", TimestampType(), False),
    StructField("PULocationID", IntegerType(), False),
])

trip_counts = {
    1: 10,   # Midtown
    2: 7,    # JFK Airport
    3: 4,    # Astoria
    4: 2,    # Harlem
}

trip_rows = []

for location_id, count in trip_counts.items():
    for i in range(count):
        trip_rows.append(
            (
                datetime(
                    2024,
                    1,
                    1 + (i % 20),
                    10,
                    0,
                    0
                ),
                location_id,
            )
        )

yellow_df = spark.createDataFrame(
    trip_rows,
    schema=trip_schema
)

yellow_df.createOrReplaceTempView("yellow_taxi")


print()
print("======================================")
print("SYNTHETIC DATA")
print("======================================")

print(f"Yellow Taxi rows:       {yellow_df.count()}")
print(f"Taxi Zone rows:         {zones_df.count()}")


# ---------------------------------------------------------
# Natural-language → Spark SQL
# ---------------------------------------------------------

prompt = """
You are an expert Spark SQL generator.

DATABASE SCHEMA:

yellow_taxi(
    pickup_datetime TIMESTAMP,
    PULocationID INT
)

taxi_zones(
    LocationID INT,
    Borough STRING,
    Zone STRING
)

RELATIONSHIP:

yellow_taxi.PULocationID = taxi_zones.LocationID

QUESTION:

What are the three taxi zones with the highest number of
Yellow Taxi pickups?

Return exactly these two columns:

Zone
pickup_count

Requirements:
- Use Spark SQL syntax.
- Use only the tables and columns provided.
- Join the tables using the provided relationship.
- Count the number of Yellow Taxi pickups for each zone.
- Order the result from highest to lowest pickup_count.
- Return only the top 3 zones.
- Return only SQL.
- Do not explain the query.
- Do not use markdown.
"""

payload = {
    "model": MODEL,
    "stream": False,
    "keep_alive": "10m",
    "options": {
        "temperature": 0,
        "num_ctx": 2048,
    },
    "prompt": prompt,
}

request = urllib.request.Request(
    "http://127.0.0.1:11434/api/generate",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)

print()
print("======================================")
print("CALLING LLM WHILE SPARK IS ACTIVE")
print("======================================")

print(f"Memory before LLM load: {mem_available_gib():.2f} GiB")

start = time.perf_counter()

with urllib.request.urlopen(request, timeout=600) as response:
    ollama_result = json.loads(
        response.read().decode("utf-8")
    )

wall_time = time.perf_counter() - start

sql = clean_sql(
    ollama_result.get("response", "")
)

eval_duration = (
    ollama_result.get("eval_duration", 0)
    / 1_000_000_000
)

load_duration = (
    ollama_result.get("load_duration", 0)
    / 1_000_000_000
)

prompt_duration = (
    ollama_result.get("prompt_eval_duration", 0)
    / 1_000_000_000
)

eval_count = ollama_result.get(
    "eval_count",
    0
)


print()
print("======================================")
print("GENERATED SPARK SQL")
print("======================================")

print(sql)


print()
print("======================================")
print("LLM PERFORMANCE")
print("======================================")

print(f"Wall time:              {wall_time:.2f} s")
print(f"Model load:             {load_duration:.2f} s")
print(f"Prompt evaluation:      {prompt_duration:.2f} s")
print(f"Generation:             {eval_duration:.2f} s")
print(f"Generated tokens:       {eval_count}")

if eval_duration > 0:
    print(
        f"Generation speed:       "
        f"{eval_count / eval_duration:.2f} tok/s"
    )

print(
    f"Memory with Spark+LLM:  "
    f"{mem_available_gib():.2f} GiB"
)


# ---------------------------------------------------------
# Execute generated SQL
# ---------------------------------------------------------

print()
print("======================================")
print("EXECUTING GENERATED SQL")
print("======================================")

try:
    generated_df = spark.sql(sql)

    generated_rows = generated_df.collect()

    print("Generated result:")

    for row in generated_rows:
        print(row)

except Exception as exc:
    print()
    print("GENERATED SQL EXECUTION: FAIL")
    print(type(exc).__name__)
    print(str(exc))

    spark.stop()

    raise


# ---------------------------------------------------------
# Gold SQL
# ---------------------------------------------------------

gold_sql = """
SELECT
    z.Zone,
    COUNT(*) AS pickup_count
FROM yellow_taxi y
JOIN taxi_zones z
    ON y.PULocationID = z.LocationID
GROUP BY z.Zone
ORDER BY pickup_count DESC
LIMIT 3
"""

gold_rows = spark.sql(
    gold_sql
).collect()


print()
print("======================================")
print("GOLD RESULT")
print("======================================")

for row in gold_rows:
    print(row)


# ---------------------------------------------------------
# Semantic result comparison
# ---------------------------------------------------------

def normalize(rows):
    normalized = []

    for row in rows:
        normalized.append(
            (
                str(row[0]),
                int(row[1])
            )
        )

    return normalized


generated_normalized = normalize(
    generated_rows
)

gold_normalized = normalize(
    gold_rows
)


print()
print("======================================")
print("SEMANTIC VALIDATION")
print("======================================")

print(
    "Generated:",
    generated_normalized
)

print(
    "Gold:     ",
    gold_normalized
)


if generated_normalized == gold_normalized:
    print()
    print("RESULT CORRECTNESS: PASS")
else:
    print()
    print("RESULT CORRECTNESS: FAIL")


print()
print("======================================")
print("MEMORY BEFORE SPARK STOP")
print("======================================")

print(
    f"Available memory: "
    f"{mem_available_gib():.2f} GiB"
)

spark.stop()


print()
print("======================================")
print("FINAL RESULT")
print("======================================")

print("VERTICAL SLICE COMPLETED")

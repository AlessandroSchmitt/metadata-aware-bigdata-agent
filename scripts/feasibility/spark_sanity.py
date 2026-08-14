import os
import shutil
import tempfile
import time

from pyspark.sql import SparkSession


def mem_available_gib():
    with open("/proc/meminfo", "r") as f:
        values = {}
        for line in f:
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])

    return values["MemAvailable"] / 1024 / 1024


print("======================================")
print("SPARK SANITY TEST")
print("======================================")

print(f"System memory available before Spark actions: {mem_available_gib():.2f} GiB")

spark = (
    SparkSession.builder
    .appName("metadata-aware-agent-sanity")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

java_version = (
    spark.sparkContext
    ._jvm.java.lang.System.getProperty("java.version")
)

driver_max_memory = (
    spark.sparkContext
    ._jvm.java.lang.Runtime.getRuntime().maxMemory()
    / 1024 / 1024
)

print()
print("======================================")
print("ENVIRONMENT")
print("======================================")
print(f"Spark version:          {spark.version}")
print(f"Java version:           {java_version}")
print(f"Spark master:           {spark.sparkContext.master}")
print(f"Driver max heap:        {driver_max_memory:.0f} MiB")
print(f"Shuffle partitions:     {spark.conf.get('spark.sql.shuffle.partitions')}")

print()
print("======================================")
print("DATAFRAME + SQL TEST")
print("======================================")

start = time.perf_counter()

df = (
    spark.range(0, 100_000)
    .selectExpr(
        "id",
        "CAST(id % 100 AS INT) AS group_id"
    )
)

df.createOrReplaceTempView("numbers")

result = spark.sql("""
    SELECT
        group_id,
        COUNT(*) AS row_count,
        SUM(id) AS id_sum
    FROM numbers
    GROUP BY group_id
    ORDER BY group_id
""")

rows = result.collect()

elapsed = time.perf_counter() - start

print(f"Input rows:             {df.count()}")
print(f"Output groups:          {len(rows)}")
print(f"SQL execution time:     {elapsed:.2f} s")

print()
print("First 5 result rows:")

for row in rows[:5]:
    print(row)

assert df.count() == 100_000
assert len(rows) == 100
assert rows[0]["group_id"] == 0
assert rows[0]["row_count"] == 1000

print()
print("SQL assertions:         PASS")

print()
print("======================================")
print("PARQUET TEST")
print("======================================")

tmp_dir = tempfile.mkdtemp(
    prefix="metadata_agent_spark_",
    dir="/tmp"
)

parquet_path = os.path.join(tmp_dir, "numbers.parquet")

start = time.perf_counter()

df.write.mode("overwrite").parquet(parquet_path)

reloaded = spark.read.parquet(parquet_path)

parquet_count = reloaded.count()

elapsed = time.perf_counter() - start

print(f"Parquet rows reloaded:  {parquet_count}")
print(f"Parquet round-trip:     {elapsed:.2f} s")

assert parquet_count == 100_000

print("Parquet assertions:     PASS")

print()
print("======================================")
print("MEMORY WHILE SPARK IS ACTIVE")
print("======================================")
print(f"System memory available: {mem_available_gib():.2f} GiB")

shutil.rmtree(tmp_dir, ignore_errors=True)

spark.stop()

print()
print("======================================")
print("FINAL RESULT")
print("======================================")
print("SPARK SANITY TEST: PASS")

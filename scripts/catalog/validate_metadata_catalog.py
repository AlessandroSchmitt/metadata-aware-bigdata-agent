import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DB_PATH = (
    ROOT / "data/catalog/metadata_catalog.sqlite"
)

if not DB_PATH.exists():
    raise RuntimeError(
        f"Metadata catalog not found: {DB_PATH}"
    )

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row


print("=" * 70)
print("METADATA CATALOG VALIDATION")
print("=" * 70)


# ---------------------------------------------------------
# Datasets
# ---------------------------------------------------------

print()
print("=" * 70)
print("DATASETS")
print("=" * 70)

datasets = conn.execute(
    """
    SELECT
        name,
        row_count,
        granularity,
        primary_time_column
    FROM datasets
    ORDER BY id
    """
).fetchall()

for row in datasets:
    print(
        f"{row['name']:18s} "
        f"rows={row['row_count']:10,d} "
        f"time={row['primary_time_column']}"
    )


# ---------------------------------------------------------
# Semantic equivalence
# ---------------------------------------------------------

print()
print("=" * 70)
print("SEMANTIC EQUIVALENCE: PICKUP DATETIME")
print("=" * 70)

pickup_rows = conn.execute(
    """
    SELECT
        d.name AS dataset,
        c.name AS column_name,
        c.data_type
    FROM column_semantics cs
    JOIN columns c
      ON cs.column_id = c.id
    JOIN datasets d
      ON c.dataset_id = d.id
    JOIN semantic_concepts sc
      ON cs.concept_id = sc.id
    WHERE sc.name = 'pickup_datetime'
    ORDER BY d.name
    """
).fetchall()

for row in pickup_rows:
    print(
        f"{row['dataset']}."
        f"{row['column_name']} "
        f"[{row['data_type']}]"
    )


# ---------------------------------------------------------
# Cross-source relationships
# ---------------------------------------------------------

print()
print("=" * 70)
print("CROSS-SOURCE RELATIONSHIPS")
print("=" * 70)

relationship_rows = conn.execute(
    """
    SELECT
        r.name,
        sd.name AS source_dataset,
        td.name AS target_dataset,
        r.relationship_type,
        r.source_expression,
        r.target_expression,
        r.cardinality,
        r.validated
    FROM relationships r
    JOIN datasets sd
      ON r.source_dataset_id = sd.id
    JOIN datasets td
      ON r.target_dataset_id = td.id
    ORDER BY r.id
    """
).fetchall()

for row in relationship_rows:
    print()
    print(row["name"])

    print(
        f"  {row['source_dataset']} "
        f"-> {row['target_dataset']}"
    )

    print(
        f"  type:        "
        f"{row['relationship_type']}"
    )

    print(
        f"  cardinality: "
        f"{row['cardinality']}"
    )

    print(
        f"  source:      "
        f"{row['source_expression']}"
    )

    print(
        f"  target:      "
        f"{row['target_expression']}"
    )

    print(
        f"  validated:   "
        f"{bool(row['validated'])}"
    )


# ---------------------------------------------------------
# Rain semantics
# ---------------------------------------------------------

print()
print("=" * 70)
print("SEMANTIC RULE: RAINY HOUR")
print("=" * 70)

rain_rule = conn.execute(
    """
    SELECT
        sr.name,
        d.name AS dataset,
        sr.description,
        sr.sql_expression,
        sr.result_semantics
    FROM semantic_rules sr
    LEFT JOIN datasets d
      ON sr.dataset_id = d.id
    WHERE sr.name = 'rainy_hour'
    """
).fetchone()

if rain_rule is None:
    raise RuntimeError(
        "rainy_hour semantic rule missing"
    )

print(
    f"Name:        {rain_rule['name']}"
)

print(
    f"Dataset:     {rain_rule['dataset']}"
)

print(
    f"Description: {rain_rule['description']}"
)

print(
    f"SQL:         {rain_rule['sql_expression']}"
)

print(
    f"Semantics:   {rain_rule['result_semantics']}"
)


# ---------------------------------------------------------
# Alias lookup
# ---------------------------------------------------------

print()
print("=" * 70)
print("ALIASES FOR PRECIPITATION STATUS")
print("=" * 70)

alias_rows = conn.execute(
    """
    SELECT alias
    FROM aliases
    WHERE entity_type = 'concept'
      AND entity_key = 'precipitation_status'
    ORDER BY alias
    """
).fetchall()

for row in alias_rows:
    print(
        f"  - {row['alias']}"
    )


# ---------------------------------------------------------
# Weather semantic mappings
# ---------------------------------------------------------

print()
print("=" * 70)
print("WEATHER SEMANTIC MAPPINGS")
print("=" * 70)

weather_semantics = conn.execute(
    """
    SELECT
        c.name AS column_name,
        sc.name AS semantic_concept
    FROM column_semantics cs
    JOIN columns c
      ON cs.column_id = c.id
    JOIN datasets d
      ON c.dataset_id = d.id
    JOIN semantic_concepts sc
      ON cs.concept_id = sc.id
    WHERE d.name = 'weather_hourly'
    ORDER BY c.ordinal_position
    """
).fetchall()

for row in weather_semantics:
    print(
        f"weather_hourly."
        f"{row['column_name']:24s} "
        f"-> {row['semantic_concept']}"
    )


# ---------------------------------------------------------
# Database integrity
# ---------------------------------------------------------

print()
print("=" * 70)
print("DATABASE VALIDATION")
print("=" * 70)

integrity = conn.execute(
    "PRAGMA integrity_check"
).fetchone()[0]

foreign_keys = conn.execute(
    "PRAGMA foreign_key_check"
).fetchall()

print(
    f"Integrity:   {integrity}"
)

print(
    f"FK errors:   {len(foreign_keys)}"
)


# ---------------------------------------------------------
# Assertions
# ---------------------------------------------------------

assert len(datasets) == 4
assert len(pickup_rows) == 2
assert len(relationship_rows) == 6
assert len(alias_rows) >= 3
assert len(weather_semantics) >= 7

pickup_pairs = {
    (
        row["dataset"],
        row["column_name"],
    )
    for row in pickup_rows
}

assert (
    "yellow_taxi",
    "tpep_pickup_datetime",
) in pickup_pairs

assert (
    "green_taxi",
    "lpep_pickup_datetime",
) in pickup_pairs

relationship_names = {
    row["name"]
    for row in relationship_rows
}

assert (
    "yellow_pickup_weather_hour"
    in relationship_names
)

assert (
    "green_pickup_weather_hour"
    in relationship_names
)

assert (
    rain_rule["sql_expression"]
    == "weather_hourly.has_precipitation = TRUE"
)

assert integrity == "ok"
assert len(foreign_keys) == 0


conn.close()


print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(
    "METADATA CATALOG VALIDATION: PASS"
)

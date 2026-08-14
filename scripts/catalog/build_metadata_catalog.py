import json
import sqlite3
import time
from pathlib import Path

from pyspark.sql import SparkSession


ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "config/catalog_seed.json"
DB_PATH = ROOT / "data/catalog/metadata_catalog.sqlite"


DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE catalog_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    description TEXT NOT NULL,
    layer TEXT NOT NULL,
    format TEXT NOT NULL,
    path TEXT NOT NULL,
    source TEXT,
    granularity TEXT,
    primary_time_column TEXT,
    temporal_start TEXT,
    temporal_end TEXT,
    row_count INTEGER,
    notes TEXT
);

CREATE TABLE columns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    ordinal_position INTEGER NOT NULL,
    data_type TEXT NOT NULL,
    nullable INTEGER NOT NULL,
    FOREIGN KEY(dataset_id)
        REFERENCES datasets(id)
        ON DELETE CASCADE,
    UNIQUE(dataset_id, name)
);

CREATE TABLE semantic_concepts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    description TEXT NOT NULL
);

CREATE TABLE column_semantics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    column_id INTEGER NOT NULL,
    concept_id INTEGER NOT NULL,
    semantic_role TEXT NOT NULL DEFAULT 'represents',
    FOREIGN KEY(column_id)
        REFERENCES columns(id)
        ON DELETE CASCADE,
    FOREIGN KEY(concept_id)
        REFERENCES semantic_concepts(id)
        ON DELETE CASCADE,
    UNIQUE(column_id, concept_id, semantic_role)
);

CREATE TABLE relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    source_dataset_id INTEGER NOT NULL,
    target_dataset_id INTEGER NOT NULL,
    relationship_type TEXT NOT NULL,
    source_expression TEXT NOT NULL,
    target_expression TEXT NOT NULL,
    cardinality TEXT,
    validated INTEGER NOT NULL DEFAULT 0,
    description TEXT NOT NULL,
    FOREIGN KEY(source_dataset_id)
        REFERENCES datasets(id),
    FOREIGN KEY(target_dataset_id)
        REFERENCES datasets(id)
);

CREATE TABLE aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_key TEXT NOT NULL,
    alias TEXT NOT NULL,
    CHECK(entity_type IN ('dataset', 'column', 'concept', 'rule')),
    UNIQUE(entity_type, entity_key, alias)
);

CREATE TABLE semantic_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    dataset_id INTEGER,
    description TEXT NOT NULL,
    sql_expression TEXT NOT NULL,
    result_semantics TEXT,
    FOREIGN KEY(dataset_id)
        REFERENCES datasets(id)
);

CREATE INDEX idx_columns_dataset
ON columns(dataset_id);

CREATE INDEX idx_column_semantics_column
ON column_semantics(column_id);

CREATE INDEX idx_column_semantics_concept
ON column_semantics(concept_id);

CREATE INDEX idx_aliases_alias
ON aliases(alias);

CREATE INDEX idx_relationship_source
ON relationships(source_dataset_id);

CREATE INDEX idx_relationship_target
ON relationships(target_dataset_id);
"""


def dataset_id(conn, name):
    row = conn.execute(
        "SELECT id FROM datasets WHERE name = ?",
        (name,),
    ).fetchone()

    if row is None:
        raise RuntimeError(
            f"Unknown dataset: {name}"
        )

    return row[0]


def column_id(conn, dataset_name, column_name):
    row = conn.execute(
        """
        SELECT c.id
        FROM columns c
        JOIN datasets d
          ON c.dataset_id = d.id
        WHERE d.name = ?
          AND c.name = ?
        """,
        (dataset_name, column_name),
    ).fetchone()

    if row is None:
        raise RuntimeError(
            f"Unknown column: "
            f"{dataset_name}.{column_name}"
        )

    return row[0]


def concept_id(conn, name):
    row = conn.execute(
        "SELECT id FROM semantic_concepts WHERE name = ?",
        (name,),
    ).fetchone()

    if row is None:
        raise RuntimeError(
            f"Unknown semantic concept: {name}"
        )

    return row[0]


def main():
    print("=" * 70)
    print("BUILD METADATA CATALOG")
    print("=" * 70)

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = json.load(handle)

    spark = (
        SparkSession.builder
        .appName("metadata-aware-agent-catalog-builder")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if DB_PATH.exists():
        DB_PATH.unlink()

    start = time.perf_counter()

    conn = sqlite3.connect(DB_PATH)

    try:
        conn.execute(
            "PRAGMA foreign_keys = ON"
        )

        conn.executescript(DDL)

        catalog_info = config["catalog"]

        for key in [
            "name",
            "version",
            "description",
        ]:
            conn.execute(
                """
                INSERT INTO catalog_meta(key, value)
                VALUES (?, ?)
                """,
                (
                    key,
                    str(catalog_info[key]),
                ),
            )

        print()
        print("=" * 70)
        print("DATASET + PHYSICAL SCHEMA DISCOVERY")
        print("=" * 70)

        for dataset in config["datasets"]:
            dataset_name = dataset["name"]
            path = ROOT / dataset["path"]

            if not path.exists():
                raise RuntimeError(
                    f"Dataset path does not exist: {path}"
                )

            df = spark.read.parquet(
                str(path)
            )

            rows = df.count()

            cursor = conn.execute(
                """
                INSERT INTO datasets(
                    name,
                    display_name,
                    description,
                    layer,
                    format,
                    path,
                    source,
                    granularity,
                    primary_time_column,
                    temporal_start,
                    temporal_end,
                    row_count,
                    notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset["name"],
                    dataset["display_name"],
                    dataset["description"],
                    dataset["layer"],
                    dataset["format"],
                    dataset["path"],
                    dataset.get("source"),
                    dataset.get("granularity"),
                    dataset.get(
                        "primary_time_column"
                    ),
                    dataset.get(
                        "temporal_start"
                    ),
                    dataset.get(
                        "temporal_end"
                    ),
                    rows,
                    dataset.get("notes"),
                ),
            )

            ds_id = cursor.lastrowid

            for position, field in enumerate(
                df.schema.fields,
                start=1,
            ):
                conn.execute(
                    """
                    INSERT INTO columns(
                        dataset_id,
                        name,
                        ordinal_position,
                        data_type,
                        nullable
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        ds_id,
                        field.name,
                        position,
                        field.dataType.simpleString(),
                        int(field.nullable),
                    ),
                )

            print(
                f"{dataset_name:20s} "
                f"rows={rows:10,d} "
                f"columns={len(df.columns):3d}"
            )

        print()
        print("=" * 70)
        print("SEMANTIC CONCEPTS")
        print("=" * 70)

        for concept in config[
            "semantic_concepts"
        ]:
            conn.execute(
                """
                INSERT INTO semantic_concepts(
                    name,
                    category,
                    description
                )
                VALUES (?, ?, ?)
                """,
                (
                    concept["name"],
                    concept["category"],
                    concept["description"],
                ),
            )

        concept_count = conn.execute(
            "SELECT COUNT(*) FROM semantic_concepts"
        ).fetchone()[0]

        print(
            f"Semantic concepts: {concept_count}"
        )

        print()
        print("=" * 70)
        print("COLUMN SEMANTICS")
        print("=" * 70)

        for (
            ds_name,
            col_name,
            semantic_name,
        ) in config["column_semantics"]:

            conn.execute(
                """
                INSERT INTO column_semantics(
                    column_id,
                    concept_id,
                    semantic_role
                )
                VALUES (?, ?, 'represents')
                """,
                (
                    column_id(
                        conn,
                        ds_name,
                        col_name,
                    ),
                    concept_id(
                        conn,
                        semantic_name,
                    ),
                ),
            )

        mapping_count = conn.execute(
            "SELECT COUNT(*) FROM column_semantics"
        ).fetchone()[0]

        print(
            f"Column-semantic mappings: "
            f"{mapping_count}"
        )

        print()
        print("=" * 70)
        print("RELATIONSHIPS")
        print("=" * 70)

        for relationship in config[
            "relationships"
        ]:
            conn.execute(
                """
                INSERT INTO relationships(
                    name,
                    source_dataset_id,
                    target_dataset_id,
                    relationship_type,
                    source_expression,
                    target_expression,
                    cardinality,
                    validated,
                    description
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    relationship["name"],
                    dataset_id(
                        conn,
                        relationship[
                            "source_dataset"
                        ],
                    ),
                    dataset_id(
                        conn,
                        relationship[
                            "target_dataset"
                        ],
                    ),
                    relationship[
                        "relationship_type"
                    ],
                    relationship[
                        "source_expression"
                    ],
                    relationship[
                        "target_expression"
                    ],
                    relationship.get(
                        "cardinality"
                    ),
                    int(
                        relationship.get(
                            "validated",
                            False,
                        )
                    ),
                    relationship[
                        "description"
                    ],
                ),
            )

            print(
                f"  {relationship['name']}"
            )

        print()
        print("=" * 70)
        print("ALIASES")
        print("=" * 70)

        valid_datasets = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM datasets"
            )
        }

        valid_concepts = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM semantic_concepts"
            )
        }

        for entity_type, entity_key, alias in config[
            "aliases"
        ]:
            if (
                entity_type == "dataset"
                and entity_key not in valid_datasets
            ):
                raise RuntimeError(
                    f"Alias points to unknown dataset: "
                    f"{entity_key}"
                )

            if (
                entity_type == "concept"
                and entity_key not in valid_concepts
            ):
                raise RuntimeError(
                    f"Alias points to unknown concept: "
                    f"{entity_key}"
                )

            conn.execute(
                """
                INSERT INTO aliases(
                    entity_type,
                    entity_key,
                    alias
                )
                VALUES (?, ?, ?)
                """,
                (
                    entity_type,
                    entity_key,
                    alias,
                ),
            )

        alias_count = conn.execute(
            "SELECT COUNT(*) FROM aliases"
        ).fetchone()[0]

        print(
            f"Aliases: {alias_count}"
        )

        print()
        print("=" * 70)
        print("SEMANTIC RULES")
        print("=" * 70)

        for rule in config[
            "semantic_rules"
        ]:
            ds_id = None

            if rule.get("dataset"):
                ds_id = dataset_id(
                    conn,
                    rule["dataset"],
                )

            conn.execute(
                """
                INSERT INTO semantic_rules(
                    name,
                    dataset_id,
                    description,
                    sql_expression,
                    result_semantics
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    rule["name"],
                    ds_id,
                    rule["description"],
                    rule["sql_expression"],
                    rule.get(
                        "result_semantics"
                    ),
                ),
            )

            print(
                f"  {rule['name']}: "
                f"{rule['sql_expression']}"
            )

        conn.commit()

        fk_errors = conn.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

        if fk_errors:
            raise RuntimeError(
                f"Foreign-key errors: {fk_errors}"
            )

        integrity = conn.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        if integrity != "ok":
            raise RuntimeError(
                f"SQLite integrity check failed: "
                f"{integrity}"
            )

        print()
        print("=" * 70)
        print("CATALOG COUNTS")
        print("=" * 70)

        for table in [
            "datasets",
            "columns",
            "semantic_concepts",
            "column_semantics",
            "relationships",
            "aliases",
            "semantic_rules",
        ]:
            count = conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]

            print(
                f"{table:22s} {count:4d}"
            )

        elapsed = (
            time.perf_counter()
            - start
        )

        print()
        print("=" * 70)
        print("OUTPUT")
        print("=" * 70)

        print(
            DB_PATH.relative_to(ROOT)
        )

        print(
            f"Database size: "
            f"{DB_PATH.stat().st_size / 1024:.2f} KiB"
        )

        print(
            f"Build time: {elapsed:.2f} s"
        )

        print()
        print(
            "SQLite integrity check: PASS"
        )

    finally:
        conn.close()
        spark.stop()

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print("METADATA CATALOG BUILD: PASS")


if __name__ == "__main__":
    main()

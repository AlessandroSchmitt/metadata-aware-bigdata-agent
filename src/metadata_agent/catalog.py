import sqlite3
from pathlib import Path
from typing import Optional


class MetadataCatalog:
    """
    Read-only access layer for the SQLite metadata catalog.

    The catalog stores:
    - physical schemas discovered from the data lake;
    - semantic concepts;
    - column-to-concept mappings;
    - aliases;
    - cross-source relationships;
    - semantic SQL rules.
    """

    def __init__(self, db_path):
        self.db_path = Path(db_path)

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Metadata catalog not found: "
                f"{self.db_path}"
            )

    def _connect(self):
        connection = sqlite3.connect(
            self.db_path
        )

        connection.row_factory = (
            sqlite3.Row
        )

        return connection

    def catalog_meta(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT key, value
                FROM catalog_meta
                ORDER BY key
                """
            ).fetchall()

        return {
            row["key"]: row["value"]
            for row in rows
        }

    def datasets(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    id,
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
                FROM datasets
                ORDER BY id
                """
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def columns(
        self,
        dataset_name: Optional[str] = None,
    ):
        with self._connect() as conn:

            if dataset_name is None:
                rows = conn.execute(
                    """
                    SELECT
                        d.name AS dataset,
                        c.name,
                        c.ordinal_position,
                        c.data_type,
                        c.nullable,
                        sc.name AS semantic_concept
                    FROM columns c
                    JOIN datasets d
                      ON c.dataset_id = d.id
                    LEFT JOIN column_semantics cs
                      ON cs.column_id = c.id
                    LEFT JOIN semantic_concepts sc
                      ON cs.concept_id = sc.id
                    ORDER BY
                        d.id,
                        c.ordinal_position
                    """
                ).fetchall()

            else:
                rows = conn.execute(
                    """
                    SELECT
                        d.name AS dataset,
                        c.name,
                        c.ordinal_position,
                        c.data_type,
                        c.nullable,
                        sc.name AS semantic_concept
                    FROM columns c
                    JOIN datasets d
                      ON c.dataset_id = d.id
                    LEFT JOIN column_semantics cs
                      ON cs.column_id = c.id
                    LEFT JOIN semantic_concepts sc
                      ON cs.concept_id = sc.id
                    WHERE d.name = ?
                    ORDER BY c.ordinal_position
                    """,
                    (dataset_name,),
                ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def semantic_concepts(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    name,
                    category,
                    description
                FROM semantic_concepts
                ORDER BY id
                """
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def relationships(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    r.name,
                    sd.name AS source_dataset,
                    td.name AS target_dataset,
                    r.relationship_type,
                    r.source_expression,
                    r.target_expression,
                    r.cardinality,
                    r.validated,
                    r.description
                FROM relationships r
                JOIN datasets sd
                  ON r.source_dataset_id = sd.id
                JOIN datasets td
                  ON r.target_dataset_id = td.id
                ORDER BY r.id
                """
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def semantic_rules(self):
        with self._connect() as conn:
            rows = conn.execute(
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
                ORDER BY sr.id
                """
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def aliases(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    entity_type,
                    entity_key,
                    alias
                FROM aliases
                ORDER BY
                    entity_type,
                    entity_key,
                    alias
                """
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def aliases_for(
        self,
        entity_type,
        entity_key,
    ):
        return [
            item["alias"]
            for item in self.aliases()
            if (
                item["entity_type"]
                == entity_type
                and item["entity_key"]
                == entity_key
            )
        ]

    def render_full_catalog(self):
        """
        Render the complete catalog into a deterministic,
        compact text representation suitable for an LLM prompt.

        This is the Full Catalog experimental baseline.
        """

        meta = self.catalog_meta()

        datasets = self.datasets()
        concepts = self.semantic_concepts()
        relationships = self.relationships()
        rules = self.semantic_rules()

        lines = []

        lines.append(
            "=== METADATA CATALOG ==="
        )

        lines.append(
            f"Name: {meta.get('name', '')}"
        )

        lines.append(
            f"Version: {meta.get('version', '')}"
        )

        lines.append(
            f"Description: "
            f"{meta.get('description', '')}"
        )

        # -------------------------------------------------
        # Datasets and physical columns
        # -------------------------------------------------

        lines.append("")
        lines.append(
            "=== DATASETS AND SCHEMAS ==="
        )

        for dataset in datasets:

            lines.append("")
            lines.append(
                f"DATASET {dataset['name']}"
            )

            lines.append(
                f"display_name: "
                f"{dataset['display_name']}"
            )

            lines.append(
                f"description: "
                f"{dataset['description']}"
            )

            lines.append(
                f"granularity: "
                f"{dataset['granularity']}"
            )

            lines.append(
                f"row_count: "
                f"{dataset['row_count']}"
            )

            if dataset[
                "primary_time_column"
            ]:
                lines.append(
                    f"primary_time_column: "
                    f"{dataset['primary_time_column']}"
                )

            dataset_aliases = (
                self.aliases_for(
                    "dataset",
                    dataset["name"],
                )
            )

            if dataset_aliases:
                lines.append(
                    "aliases: "
                    + ", ".join(
                        dataset_aliases
                    )
                )

            lines.append("columns:")

            for column in self.columns(
                dataset["name"]
            ):
                nullable = (
                    "nullable"
                    if column["nullable"]
                    else "required"
                )

                text = (
                    f"- {column['name']}:"
                    f"{column['data_type']} "
                    f"[{nullable}]"
                )

                if column[
                    "semantic_concept"
                ]:
                    text += (
                        " -> semantic="
                        f"{column['semantic_concept']}"
                    )

                lines.append(text)

        # -------------------------------------------------
        # Semantic concepts
        # -------------------------------------------------

        lines.append("")
        lines.append(
            "=== SEMANTIC CONCEPTS ==="
        )

        for concept in concepts:

            aliases = self.aliases_for(
                "concept",
                concept["name"],
            )

            text = (
                f"- {concept['name']} "
                f"[{concept['category']}]: "
                f"{concept['description']}"
            )

            if aliases:
                text += (
                    " | aliases="
                    + ", ".join(aliases)
                )

            lines.append(text)

        # -------------------------------------------------
        # Relationships
        # -------------------------------------------------

        lines.append("")
        lines.append(
            "=== CROSS-SOURCE RELATIONSHIPS ==="
        )

        for relationship in relationships:

            status = (
                "validated"
                if relationship["validated"]
                else "unvalidated"
            )

            lines.append(
                f"- {relationship['name']} "
                f"[{relationship['relationship_type']}, "
                f"{relationship['cardinality']}, "
                f"{status}]: "
                f"{relationship['source_expression']} "
                f"= "
                f"{relationship['target_expression']}"
            )

            lines.append(
                f"  meaning: "
                f"{relationship['description']}"
            )

        # -------------------------------------------------
        # Semantic rules
        # -------------------------------------------------

        lines.append("")
        lines.append(
            "=== SEMANTIC SQL RULES ==="
        )

        for rule in rules:

            dataset_text = (
                rule["dataset"]
                if rule["dataset"]
                else "global"
            )

            lines.append(
                f"- {rule['name']} "
                f"[{dataset_text}]: "
                f"{rule['sql_expression']}"
            )

            lines.append(
                f"  meaning: "
                f"{rule['description']}"
            )

            if rule[
                "result_semantics"
            ]:
                lines.append(
                    f"  result_semantics: "
                    f"{rule['result_semantics']}"
                )

        return "\n".join(lines)

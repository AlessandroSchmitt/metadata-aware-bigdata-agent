from dataclasses import dataclass

from metadata_agent.catalog import MetadataCatalog


@dataclass
class MetadataDocument:
    entity_type: str
    entity_key: str
    text: str
    dataset: str | None = None


class MetadataDocumentBuilder:
    """
    Convert the structured SQLite metadata catalog into textual
    documents suitable for semantic vector retrieval.

    One document is generated for each:
    - dataset
    - physical column
    - semantic concept
    - cross-source relationship
    - semantic rule
    """

    def __init__(self, catalog: MetadataCatalog):
        self.catalog = catalog

    def build(self):
        documents = []

        documents.extend(
            self._dataset_documents()
        )

        documents.extend(
            self._column_documents()
        )

        documents.extend(
            self._concept_documents()
        )

        documents.extend(
            self._relationship_documents()
        )

        documents.extend(
            self._rule_documents()
        )

        return documents

    def _dataset_documents(self):
        documents = []

        for dataset in self.catalog.datasets():
            aliases = self.catalog.aliases_for(
                "dataset",
                dataset["name"],
            )

            columns = self.catalog.columns(
                dataset["name"]
            )

            semantic_columns = [
                (
                    f"{column['name']}="
                    f"{column['semantic_concept']}"
                )
                for column in columns
                if column["semantic_concept"]
            ]

            text = " ".join(
                [
                    f"Dataset: {dataset['name']}.",
                    f"Display name: {dataset['display_name']}.",
                    f"Description: {dataset['description']}",
                    f"Granularity: {dataset['granularity']}.",
                    (
                        "Aliases: "
                        + ", ".join(aliases)
                        + "."
                        if aliases
                        else ""
                    ),
                    (
                        "Primary time column: "
                        f"{dataset['primary_time_column']}."
                        if dataset[
                            "primary_time_column"
                        ]
                        else ""
                    ),
                    (
                        "Semantic columns: "
                        + ", ".join(semantic_columns)
                        + "."
                        if semantic_columns
                        else ""
                    ),
                ]
            ).strip()

            documents.append(
                MetadataDocument(
                    entity_type="dataset",
                    entity_key=dataset["name"],
                    dataset=dataset["name"],
                    text=text,
                )
            )

        return documents

    def _column_documents(self):
        documents = []

        concepts = {
            item["name"]: item
            for item in self.catalog.semantic_concepts()
        }

        datasets = {
            item["name"]: item
            for item in self.catalog.datasets()
        }

        for column in self.catalog.columns():
            dataset_name = column["dataset"]
            dataset = datasets[dataset_name]

            parts = [
                (
                    f"Column: "
                    f"{dataset_name}.{column['name']}."
                ),
                (
                    f"Physical type: "
                    f"{column['data_type']}."
                ),
                (
                    f"Dataset: "
                    f"{dataset['display_name']}."
                ),
                (
                    f"Dataset description: "
                    f"{dataset['description']}"
                ),
            ]

            concept_name = column[
                "semantic_concept"
            ]

            if concept_name:
                concept = concepts[
                    concept_name
                ]

                aliases = (
                    self.catalog.aliases_for(
                        "concept",
                        concept_name,
                    )
                )

                parts.extend(
                    [
                        (
                            f"Semantic concept: "
                            f"{concept_name}."
                        ),
                        (
                            f"Semantic meaning: "
                            f"{concept['description']}"
                        ),
                    ]
                )

                if aliases:
                    parts.append(
                        "Semantic aliases: "
                        + ", ".join(aliases)
                        + "."
                    )

            text = " ".join(parts)

            documents.append(
                MetadataDocument(
                    entity_type="column",
                    entity_key=(
                        f"{dataset_name}."
                        f"{column['name']}"
                    ),
                    dataset=dataset_name,
                    text=text,
                )
            )

        return documents

    def _concept_documents(self):
        documents = []

        all_columns = self.catalog.columns()

        for concept in (
            self.catalog.semantic_concepts()
        ):
            aliases = self.catalog.aliases_for(
                "concept",
                concept["name"],
            )

            mappings = [
                (
                    f"{column['dataset']}."
                    f"{column['name']}"
                )
                for column in all_columns
                if (
                    column["semantic_concept"]
                    == concept["name"]
                )
            ]

            parts = [
                (
                    f"Semantic concept: "
                    f"{concept['name']}."
                ),
                (
                    f"Category: "
                    f"{concept['category']}."
                ),
                (
                    f"Meaning: "
                    f"{concept['description']}"
                ),
            ]

            if aliases:
                parts.append(
                    "Aliases: "
                    + ", ".join(aliases)
                    + "."
                )

            if mappings:
                parts.append(
                    "Represented by physical columns: "
                    + ", ".join(mappings)
                    + "."
                )

            documents.append(
                MetadataDocument(
                    entity_type="concept",
                    entity_key=concept["name"],
                    text=" ".join(parts),
                )
            )

        return documents

    def _relationship_documents(self):
        documents = []

        for relationship in (
            self.catalog.relationships()
        ):
            text = " ".join(
                [
                    (
                        f"Cross-source relationship: "
                        f"{relationship['name']}."
                    ),
                    (
                        f"Connects "
                        f"{relationship['source_dataset']} "
                        f"to "
                        f"{relationship['target_dataset']}."
                    ),
                    (
                        f"Relationship type: "
                        f"{relationship['relationship_type']}."
                    ),
                    (
                        f"Cardinality: "
                        f"{relationship['cardinality']}."
                    ),
                    (
                        f"Join expression: "
                        f"{relationship['source_expression']} "
                        f"= "
                        f"{relationship['target_expression']}."
                    ),
                    (
                        f"Meaning: "
                        f"{relationship['description']}"
                    ),
                    (
                        "This relationship has been "
                        "validated against the data."
                        if relationship["validated"]
                        else ""
                    ),
                ]
            ).strip()

            documents.append(
                MetadataDocument(
                    entity_type="relationship",
                    entity_key=relationship["name"],
                    dataset=relationship[
                        "source_dataset"
                    ],
                    text=text,
                )
            )

        return documents

    def _rule_documents(self):
        documents = []

        for rule in self.catalog.semantic_rules():
            aliases = self.catalog.aliases_for(
                "rule",
                rule["name"],
            )

            parts = [
                (
                    f"Semantic SQL rule: "
                    f"{rule['name']}."
                ),
                (
                    f"Dataset: "
                    f"{rule['dataset']}."
                    if rule["dataset"]
                    else "Global rule."
                ),
                (
                    f"Meaning: "
                    f"{rule['description']}"
                ),
                (
                    f"SQL expression: "
                    f"{rule['sql_expression']}."
                ),
            ]

            if rule["result_semantics"]:
                parts.append(
                    f"Result semantics: "
                    f"{rule['result_semantics']}"
                )

            if aliases:
                parts.append(
                    "Aliases: "
                    + ", ".join(aliases)
                    + "."
                )

            documents.append(
                MetadataDocument(
                    entity_type="rule",
                    entity_key=rule["name"],
                    dataset=rule["dataset"],
                    text=" ".join(parts),
                )
            )

        return documents

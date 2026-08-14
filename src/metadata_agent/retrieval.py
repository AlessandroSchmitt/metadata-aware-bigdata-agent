import re
from collections import defaultdict
from pathlib import Path

from qdrant_client import QdrantClient

from metadata_agent.catalog import MetadataCatalog
from metadata_agent.embeddings import OllamaEmbedder


class RelationAwareMetadataRetriever:
    """
    Hybrid metadata retriever.

    Pipeline:
    1. Dense semantic retrieval from Qdrant.
    2. Lexical grounding using canonical names and aliases.
    3. Semantic-concept -> physical-column expansion.
    4. Dataset discovery.
    5. Relationship-aware expansion.
    6. Join-endpoint expansion.
    7. Semantic-rule expansion.
    """

    def __init__(
        self,
        catalog_path,
        qdrant_path,
        collection="metadata_catalog",
        embedding_model="embeddinggemma",
    ):
        self.catalog = MetadataCatalog(
            catalog_path
        )

        self.qdrant_path = Path(
            qdrant_path
        )

        self.collection = collection

        self.embedder = OllamaEmbedder(
            model=embedding_model
        )

        self.datasets = {
            item["name"]: item
            for item in self.catalog.datasets()
        }

        self.columns = {
            (
                f"{item['dataset']}."
                f"{item['name']}"
            ): item
            for item in self.catalog.columns()
        }

        self.concepts = {
            item["name"]: item
            for item
            in self.catalog.semantic_concepts()
        }

        self.relationships = {
            item["name"]: item
            for item
            in self.catalog.relationships()
        }

        self.rules = {
            item["name"]: item
            for item
            in self.catalog.semantic_rules()
        }

        self.aliases = (
            self.catalog.aliases()
        )

        self.concept_to_columns = (
            defaultdict(list)
        )

        self.column_to_concept = {}

        for key, column in (
            self.columns.items()
        ):
            concept = column[
                "semantic_concept"
            ]

            if concept:
                self.concept_to_columns[
                    concept
                ].append(key)

                self.column_to_concept[
                    key
                ] = concept

    # -----------------------------------------------------
    # Text normalization
    # -----------------------------------------------------

    @staticmethod
    def _normalize(text):
        text = text.lower()

        text = re.sub(
            r"[_\-]+",
            " ",
            text,
        )

        text = re.sub(
            r"[^a-z0-9\s]+",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _stem_token(token):
        # Very small deterministic normalization.
        # Enough for hour/hours, trip/trips, etc.
        if (
            len(token) > 3
            and token.endswith("s")
        ):
            return token[:-1]

        return token

    def _tokens(self, text):
        return {
            self._stem_token(token)
            for token in self._normalize(
                text
            ).split()
            if token
        }

    def _phrase_match(
        self,
        query,
        candidate,
    ):
        query_normalized = (
            f" {self._normalize(query)} "
        )

        candidate_normalized = (
            self._normalize(candidate)
        )

        if not candidate_normalized:
            return False

        return (
            f" {candidate_normalized} "
            in query_normalized
        )

    # -----------------------------------------------------
    # Dense retrieval
    # -----------------------------------------------------

    def dense_retrieve(
        self,
        question,
        limit=5,
    ):
        embedding_result = (
            self.embedder.embed(
                question,
                keep_alive="10m",
            )
        )

        vector = (
            embedding_result[
                "embeddings"
            ][0]
        )

        client = QdrantClient(
            path=str(
                self.qdrant_path
            )
        )

        try:
            response = (
                client.query_points(
                    collection_name=(
                        self.collection
                    ),
                    query=vector,
                    limit=limit,
                    with_payload=True,
                )
            )
        finally:
            client.close()

        points = []

        for rank, point in enumerate(
            response.points,
            start=1,
        ):
            points.append(
                {
                    "rank": rank,
                    "score": point.score,
                    "entity_type": (
                        point.payload[
                            "entity_type"
                        ]
                    ),
                    "entity_key": (
                        point.payload[
                            "entity_key"
                        ]
                    ),
                    "dataset": (
                        point.payload.get(
                            "dataset"
                        )
                    ),
                    "text": (
                        point.payload[
                            "text"
                        ]
                    ),
                }
            )

        return {
            "points": points,
            "embedding": {
                "prompt_eval_count": (
                    embedding_result[
                        "prompt_eval_count"
                    ]
                ),
                "wall_time_seconds": (
                    embedding_result[
                        "wall_time_seconds"
                    ]
                ),
            },
        }

    # -----------------------------------------------------
    # Lexical grounding
    # -----------------------------------------------------

    def lexical_seeds(
        self,
        question,
    ):
        datasets = set()
        concepts = set()
        rules = set()

        # Canonical dataset names.
        for dataset_name in self.datasets:
            if self._phrase_match(
                question,
                dataset_name,
            ):
                datasets.add(
                    dataset_name
                )

        # Canonical concepts.
        for concept_name in self.concepts:
            if self._phrase_match(
                question,
                concept_name,
            ):
                concepts.add(
                    concept_name
                )

        # Configured aliases.
        for alias in self.aliases:
            if not self._phrase_match(
                question,
                alias["alias"],
            ):
                continue

            if (
                alias["entity_type"]
                == "dataset"
            ):
                datasets.add(
                    alias["entity_key"]
                )

            elif (
                alias["entity_type"]
                == "concept"
            ):
                concepts.add(
                    alias["entity_key"]
                )

            elif (
                alias["entity_type"]
                == "rule"
            ):
                rules.add(
                    alias["entity_key"]
                )

        # Rule-name grounding.
        #
        # rainy_hour → {rainy, hour}
        # question   → contains rainy + hours
        question_tokens = (
            self._tokens(question)
        )

        for rule_name in self.rules:
            rule_tokens = (
                self._tokens(
                    rule_name
                )
            )

            if (
                rule_tokens
                and rule_tokens.issubset(
                    question_tokens
                )
            ):
                rules.add(
                    rule_name
                )

        return {
            "datasets": datasets,
            "concepts": concepts,
            "rules": rules,
        }

    # -----------------------------------------------------
    # Expression references
    # -----------------------------------------------------

    @staticmethod
    def _expression_columns(
        expression,
    ):
        matches = re.findall(
            (
                r"\b"
                r"([A-Za-z_][A-Za-z0-9_]*)"
                r"\."
                r"([A-Za-z_][A-Za-z0-9_]*)"
                r"\b"
            ),
            expression,
        )

        return {
            f"{dataset}.{column}"
            for dataset, column in matches
        }

    def _relationship_relevance(
        self,
        relationship,
        question,
    ):
        """
        Score a relationship against the natural-language
        question.

        This is used only when multiple relationships connect
        the same pair of datasets, for example pickup-zone and
        dropoff-zone relationships.

        The score uses relationship metadata and the semantic
        concepts of its physical endpoint columns.
        """

        query_tokens = self._tokens(
            question
        )

        candidate_parts = [
            relationship["name"],
            relationship["description"],
        ]

        references = (
            self._expression_columns(
                relationship[
                    "source_expression"
                ]
            )
            |
            self._expression_columns(
                relationship[
                    "target_expression"
                ]
            )
        )

        for reference in references:
            concept = (
                self.column_to_concept.get(
                    reference
                )
            )

            if concept:
                candidate_parts.append(
                    concept
                )

        candidate_tokens = self._tokens(
            " ".join(candidate_parts)
        )

        # Generic tokens that do not distinguish parallel
        # relations such as pickup vs dropoff.
        stop_tokens = {
            "yellow",
            "green",
            "taxi",
            "zone",
            "weather",
            "hour",
            "trip",
            "location",
            "id",
            "record",
            "relationship",
            "map",
            "connect",
            "correspond",
            "the",
            "to",
            "of",
            "and",
            "a",
            "an",
        }

        overlap = (
            query_tokens
            & candidate_tokens
        ) - stop_tokens

        return len(overlap)

    # -----------------------------------------------------
    # Expansion
    # -----------------------------------------------------

    def expand(
        self,
        question,
        dense_points,
        lexical,
    ):
        selected_datasets = set(
            lexical["datasets"]
        )

        selected_columns = set()

        selected_concepts = set(
            lexical["concepts"]
        )

        selected_relationships = set()

        selected_rules = set(
            lexical["rules"]
        )

        # ---------------------------------------------
        # Dense dataset anchors.
        #
        # Explicit lexical dataset mentions have higher
        # precision than dense dataset hits. If the user
        # explicitly names one or more datasets, unrelated
        # dense dataset hits are not imported automatically.
        #
        # Additional datasets may still be discovered later
        # through semantic concepts, rules and relationships.
        # ---------------------------------------------

        lexical_datasets = set(
            lexical["datasets"]
        )

        for point in dense_points:
            if (
                point["entity_type"]
                == "dataset"
            ):
                dataset_name = (
                    point["entity_key"]
                )

                if (
                    not lexical_datasets
                    or dataset_name
                    in lexical_datasets
                ):
                    selected_datasets.add(
                        dataset_name
                    )

        # ---------------------------------------------
        # Dense columns are kept only when they
        # belong to an already grounded dataset.
        #
        # This prevents a Yellow Taxi question from
        # importing Green columns merely because they
        # are semantically similar.
        # ---------------------------------------------

        for point in dense_points:
            entity_type = (
                point["entity_type"]
            )

            entity_key = (
                point["entity_key"]
            )

            if entity_type == "column":
                dataset = (
                    entity_key.split(
                        ".",
                        1,
                    )[0]
                )

                if (
                    dataset
                    in selected_datasets
                ):
                    selected_columns.add(
                        entity_key
                    )

            elif entity_type == "concept":
                selected_concepts.add(
                    entity_key
                )

            elif entity_type == "rule":
                rule = self.rules[
                    entity_key
                ]

                rule_dataset = (
                    rule["dataset"]
                )

                # Explicit dataset mentions in the user
                # question have precedence over dense
                # rule hits from parallel datasets.
                #
                # Example:
                # "Yellow Taxi ... clock hour"
                # must not import the analogous Green
                # hour-of-day rule merely because its
                # embedding is semantically similar.
                #
                # Rules already grounded lexically are
                # preserved independently.
                if (
                    not lexical_datasets
                    or rule_dataset is None
                    or rule_dataset
                    in lexical_datasets
                    or entity_key
                    in lexical["rules"]
                ):
                    selected_rules.add(
                        entity_key
                    )

            elif (
                entity_type
                == "relationship"
            ):
                relationship = (
                    self.relationships[
                        entity_key
                    ]
                )

                source_dataset = (
                    relationship[
                        "source_dataset"
                    ]
                )

                # Apply the same explicit-dataset
                # preference to dense relationship hits.
                if (
                    not lexical_datasets
                    or source_dataset
                    in lexical_datasets
                ):
                    selected_relationships.add(
                        entity_key
                    )

        # ---------------------------------------------
        # Expand semantic concepts into physical
        # columns.
        #
        # Prefer mappings belonging to datasets
        # already selected. If a concept maps only
        # to one dataset globally, discover it.
        # ---------------------------------------------

        for concept in list(
            selected_concepts
        ):
            mapped_columns = (
                self.concept_to_columns.get(
                    concept,
                    [],
                )
            )

            in_selected_datasets = [
                column_key
                for column_key
                in mapped_columns
                if (
                    column_key.split(
                        ".",
                        1,
                    )[0]
                    in selected_datasets
                )
            ]

            if in_selected_datasets:
                selected_columns.update(
                    in_selected_datasets
                )

            else:
                mapped_datasets = {
                    column_key.split(
                        ".",
                        1,
                    )[0]
                    for column_key
                    in mapped_columns
                }

                if len(mapped_datasets) == 1:
                    selected_datasets.update(
                        mapped_datasets
                    )

                    selected_columns.update(
                        mapped_columns
                    )

        # ---------------------------------------------
        # Semantic rules can discover their dataset
        # and referenced physical columns.
        # ---------------------------------------------

        for rule_name in list(
            selected_rules
        ):
            rule = self.rules[
                rule_name
            ]

            if rule["dataset"]:
                selected_datasets.add(
                    rule["dataset"]
                )

            references = (
                self._expression_columns(
                    rule["sql_expression"]
                )
            )

            for reference in references:
                if reference in self.columns:
                    selected_columns.add(
                        reference
                    )

                    selected_datasets.add(
                        reference.split(
                            ".",
                            1,
                        )[0]
                    )

        # ---------------------------------------------
        # Dense relationship anchors discover both
        # endpoints.
        # ---------------------------------------------

        for relationship_name in list(
            selected_relationships
        ):
            relationship = (
                self.relationships[
                    relationship_name
                ]
            )

            selected_datasets.add(
                relationship[
                    "source_dataset"
                ]
            )

            selected_datasets.add(
                relationship[
                    "target_dataset"
                ]
            )

        # ---------------------------------------------
        # Relationship-aware expansion.
        #
        # If exactly one relationship connects a relevant
        # dataset pair, include it.
        #
        # If multiple relationships connect the same pair,
        # prefer the relation whose metadata best matches the
        # question (e.g. pickup vs dropoff). If there is no
        # discriminative evidence, preserve all candidates
        # rather than risking recall loss.
        # ---------------------------------------------

        relationship_groups = (
            defaultdict(list)
        )

        for (
            relationship_name,
            relationship,
        ) in self.relationships.items():

            source = relationship[
                "source_dataset"
            ]

            target = relationship[
                "target_dataset"
            ]

            if (
                source in selected_datasets
                and target in selected_datasets
            ):
                pair = tuple(
                    sorted(
                        (source, target)
                    )
                )

                relationship_groups[
                    pair
                ].append(
                    (
                        relationship_name,
                        relationship,
                    )
                )

        for candidates in (
            relationship_groups.values()
        ):
            if len(candidates) == 1:
                selected_relationships.add(
                    candidates[0][0]
                )

                continue

            scored = [
                (
                    self._relationship_relevance(
                        relationship,
                        question,
                    ),
                    relationship_name,
                )
                for (
                    relationship_name,
                    relationship,
                )
                in candidates
            ]

            best_score = max(
                score
                for score, _
                in scored
            )

            if best_score > 0:
                for (
                    score,
                    relationship_name,
                ) in scored:
                    if score == best_score:
                        selected_relationships.add(
                            relationship_name
                        )

            else:
                for (
                    relationship_name,
                    _,
                ) in candidates:
                    selected_relationships.add(
                        relationship_name
                    )

        # ---------------------------------------------
        # Join endpoint expansion.
        # ---------------------------------------------

        for relationship_name in (
            selected_relationships
        ):
            relationship = (
                self.relationships[
                    relationship_name
                ]
            )

            references = (
                self._expression_columns(
                    relationship[
                        "source_expression"
                    ]
                )
                |
                self._expression_columns(
                    relationship[
                        "target_expression"
                    ]
                )
            )

            for reference in references:
                if reference in self.columns:
                    selected_columns.add(
                        reference
                    )

                    selected_datasets.add(
                        reference.split(
                            ".",
                            1,
                        )[0]
                    )

        # ---------------------------------------------
        # Physical columns imply semantic concepts.
        # ---------------------------------------------

        for column_key in list(
            selected_columns
        ):
            concept = (
                self.column_to_concept.get(
                    column_key
                )
            )

            if concept:
                selected_concepts.add(
                    concept
                )

        return {
            "datasets": (
                selected_datasets
            ),
            "columns": (
                selected_columns
            ),
            "concepts": (
                selected_concepts
            ),
            "relationships": (
                selected_relationships
            ),
            "rules": (
                selected_rules
            ),
        }

    # -----------------------------------------------------
    # Compact metadata renderer
    # -----------------------------------------------------

    def render(
        self,
        question,
        selection,
    ):
        """
        Render a SQL-safe metadata context.

        Internal semantic identifiers remain available to the
        retrieval layer, but are deliberately hidden from the
        Text-to-SQL model whenever they could be mistaken for
        physical SQL identifiers.

        The LLM sees:
        - physical Spark table names;
        - physical column names;
        - natural-language meanings;
        - executable join conditions;
        - executable semantic SQL expressions/predicates.

        It does NOT see internal relationship names, rule names,
        semantic-concept identifiers, or aliases as candidate SQL
        identifiers.
        """

        lines = []

        lines.append(
            "=== SQL-SAFE RETRIEVED METADATA ==="
        )

        lines.append(
            f"Question: {question}"
        )

        lines.append("")
        lines.append(
            "=== SQL GENERATION CONSTRAINTS ==="
        )

        lines.append(
            "- Only DATASET names shown below are "
            "Spark table identifiers."
        )

        lines.append(
            "- Only names under PHYSICAL COLUMNS "
            "are physical column identifiers."
        )

        lines.append(
            "- PHYSICAL COLUMN names are fully "
            "qualified as dataset.column; use those "
            "exact physical identifiers in SQL."
        )

        lines.append(
            "- Column names requested by the user for "
            "the result are output aliases unless they "
            "exactly match a listed PHYSICAL COLUMN."
        )

        lines.append(
            "- Text following 'meaning:' is "
            "documentation, not an SQL identifier."
        )

        lines.append(
            "- JOIN CONDITION lines are executable "
            "SQL conditions; use them directly when "
            "the described relationship is required."
        )

        lines.append(
            "- SQL EXPRESSION/PREDICATE lines are "
            "executable SQL fragments; use them "
            "directly when their described meaning "
            "matches the question."
        )

        lines.append(
            "- The registered yellow_taxi, green_taxi, "
            "and weather_hourly views already contain "
            "only January 2024 records."
        )

        lines.append(
            "- Do not add a January 2024 date predicate "
            "unless the question asks for a narrower "
            "time range."
        )

        lines.append("")
        lines.append(
            "=== RELEVANT DATASETS AND "
            "PHYSICAL COLUMNS ==="
        )

        for dataset_name in sorted(
            selection["datasets"]
        ):
            dataset = self.datasets[
                dataset_name
            ]

            lines.append("")
            lines.append(
                f"DATASET {dataset_name}"
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
                "PHYSICAL COLUMNS:"
            )

            dataset_columns = [
                key
                for key
                in selection["columns"]
                if key.startswith(
                    dataset_name + "."
                )
            ]

            for key in sorted(
                dataset_columns
            ):
                column = self.columns[
                    key
                ]

                lines.append(
                    f"- {dataset_name}."
                    f"{column['name']}:"
                    f"{column['data_type']}"
                )

                concept_name = column[
                    "semantic_concept"
                ]

                if (
                    concept_name
                    and concept_name
                    in self.concepts
                ):
                    concept = self.concepts[
                        concept_name
                    ]

                    lines.append(
                        f"  meaning: "
                        f"{concept['description']}"
                    )

        lines.append("")
        lines.append(
            "=== VALID SQL JOIN CONDITIONS ==="
        )

        if not selection["relationships"]:
            lines.append(
                "- None required."
            )

        for relationship_name in sorted(
            selection["relationships"]
        ):
            relationship = (
                self.relationships[
                    relationship_name
                ]
            )

            lines.append(
                f"- JOIN CONDITION: "
                f"{relationship['source_expression']} "
                f"= "
                f"{relationship['target_expression']}"
            )

            lines.append(
                f"  meaning: "
                f"{relationship['description']}"
            )

        lines.append("")
        lines.append(
            "=== RELEVANT SQL "
            "EXPRESSIONS / PREDICATES ==="
        )

        if not selection["rules"]:
            lines.append(
                "- None required."
            )

        for rule_name in sorted(
            selection["rules"]
        ):
            rule = self.rules[
                rule_name
            ]

            lines.append(
                f"- SQL EXPRESSION/PREDICATE: "
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
                    f"  result: "
                    f"{rule['result_semantics']}"
                )

        return "\n".join(lines)

    # -----------------------------------------------------
    # Complete public retrieval call
    # -----------------------------------------------------

    def retrieve(
        self,
        question,
        dense_top_k=5,
    ):
        dense = self.dense_retrieve(
            question,
            limit=dense_top_k,
        )

        lexical = self.lexical_seeds(
            question
        )

        selection = self.expand(
            question,
            dense["points"],
            lexical,
        )

        context = self.render(
            question,
            selection,
        )

        return {
            "dense": dense,
            "lexical": lexical,
            "selection": selection,
            "context": context,
        }

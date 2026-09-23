import argparse
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

from pyspark.sql import SparkSession

from metadata_agent.retrieval import RelationAwareMetadataRetriever
from metadata_agent.sql_repair import OllamaSQLRepairer
from metadata_agent.sql_validation import SparkSQLValidator


ROOT = Path(__file__).resolve().parents[2]

CATALOG_PATH = (
    ROOT / "data/catalog/metadata_catalog.sqlite"
)

QDRANT_PATH = (
    ROOT / ".qdrant/metadata_catalog"
)

DATASETS = {
    "yellow_taxi": (
        ROOT / "data/curated/yellow/2024-01"
    ),
    "green_taxi": (
        ROOT / "data/curated/green/2024-01"
    ),
    "taxi_zones": (
        ROOT / "data/curated/zones"
    ),
    "weather_hourly": (
        ROOT / "data/curated/weather/2024-01"
    ),
}

MODEL = "qwen2.5-coder:3b"
NUM_CTX = 4096
TEMPERATURE = 0
DENSE_TOP_K = 5


def clean_sql(text):
    text = text.strip()

    text = re.sub(
        r"^```(?:sql)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    return text.strip()


def build_prompt(
    question,
    metadata_context,
):
    """
    Build the application-demo generation prompt.

    It preserves the frozen benchmark generation rules while
    restricting table grounding to the retrieved metadata context.
    """

    return f"""
You are an expert Spark SQL generator.

You must answer the user question using ONLY the metadata
context supplied below.

The physical datasets shown in the metadata context are
registered Spark SQL table names.

Important instructions:

- Generate valid Spark SQL.
- Use only physical tables and physical columns present in the metadata context.
- Semantic concept names and aliases are metadata labels, not SQL column names.
- Use the exact physical column names shown in dataset schemas or selected_columns.
- Relationship names and JOIN RULE names are metadata identifiers, not tables.
- Never place a relationship or JOIN RULE name in FROM or JOIN.
- Use the physical_join_condition supplied by a relationship.
- Respect semantic SQL rules.
- Do not invent tables, columns, relationships, or values.
- When a semantic rule exists for a user concept, use it.
- Explicitly alias every requested output expression using the exact requested output name and casing.
- Return exactly the columns requested by the user.
- Return SQL only.
- Do not use markdown.
- Do not explain your answer.

{metadata_context}

=== USER QUESTION ===

{question}
""".strip()


def ollama_generate(
    prompt,
    keep_alive="30m",
):
    payload = {
        "model": MODEL,
        "stream": False,
        "keep_alive": keep_alive,
        "prompt": prompt,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
        },
    }

    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode(
            "utf-8"
        ),
        headers={
            "Content-Type": "application/json"
        },
    )

    start = time.perf_counter()

    with urllib.request.urlopen(
        request,
        timeout=600,
    ) as response:
        result = json.loads(
            response.read().decode(
                "utf-8"
            )
        )

    return (
        result,
        time.perf_counter() - start,
    )


def print_selection(selection):
    print()
    print("=" * 78)
    print("RETRIEVED METADATA")
    print("=" * 78)

    print(
        "Datasets:      "
        + (
            ", ".join(
                sorted(selection["datasets"])
            )
            or "None"
        )
    )

    print(
        "Columns:       "
        + (
            ", ".join(
                sorted(selection["columns"])
            )
            or "None"
        )
    )

    print(
        "Relationships: "
        + (
            ", ".join(
                sorted(
                    selection["relationships"]
                )
            )
            or "None"
        )
    )

    print(
        "Rules:         "
        + (
            ", ".join(
                sorted(selection["rules"])
            )
            or "None"
        )
    )


def register_data_lake(spark):
    for name, path in DATASETS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Curated dataset not found: {path}"
            )

        (
            spark.read
            .parquet(str(path))
            .createOrReplaceTempView(name)
        )


def execute_preview(
    spark,
    sql,
    max_rows,
):
    start = time.perf_counter()

    dataframe = spark.sql(sql)

    columns = dataframe.columns

    rows = (
        dataframe
        .limit(max_rows + 1)
        .collect()
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    truncated = (
        len(rows) > max_rows
    )

    rows = rows[:max_rows]

    return {
        "columns": columns,
        "rows": [
            row.asDict(recursive=True)
            for row in rows
        ],
        "truncated": truncated,
        "seconds": elapsed,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Interactive metadata-aware "
            "Text-to-Spark-SQL demo."
        )
    )

    parser.add_argument(
        "--question",
        required=True,
        help=(
            "Natural-language analytical "
            "question."
        ),
    )

    parser.add_argument(
        "--expected-columns",
        nargs="+",
        default=None,
        help=(
            "Optional exact output-column "
            "contract. When supplied, it is "
            "also available to one-shot repair."
        ),
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
        help=(
            "Print the full SQL-safe metadata "
            "context."
        ),
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=20,
        help=(
            "Maximum number of result rows "
            "displayed. Default: 20."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.max_rows < 1:
        raise ValueError(
            "--max-rows must be at least 1."
        )

    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            "Metadata catalog not found. "
            "Build the catalog before running "
            "the demo."
        )

    if not QDRANT_PATH.exists():
        raise FileNotFoundError(
            "Metadata vector index not found. "
            "Build the index before running "
            "the demo."
        )

    question = args.question.strip()

    if not question:
        raise ValueError(
            "Question must not be empty."
        )

    print("=" * 78)
    print("METADATA-AWARE TEXT-TO-SPARK-SQL DEMO")
    print("=" * 78)

    print()
    print("QUESTION")
    print(question)

    # -----------------------------------------------------
    # Metadata retrieval
    # -----------------------------------------------------

    retriever = (
        RelationAwareMetadataRetriever(
            catalog_path=CATALOG_PATH,
            qdrant_path=QDRANT_PATH,
        )
    )

    retrieval_start = (
        time.perf_counter()
    )

    retrieved = retriever.retrieve(
        question,
        dense_top_k=DENSE_TOP_K,
    )

    retrieval_seconds = (
        time.perf_counter()
        - retrieval_start
    )

    print_selection(
        retrieved["selection"]
    )

    print(
        f"Retrieval time: {retrieval_seconds:.3f} s"
    )

    metadata_context = (
        retrieved["context"]
    )

    if not retrieved["selection"]["datasets"]:
        raise RuntimeError(
            "Metadata retrieval did not identify "
            "any physical dataset for this question."
        )

    if args.show_context:
        print()
        print("=" * 78)
        print("SQL-SAFE METADATA CONTEXT")
        print("=" * 78)
        print(metadata_context)

    # Free the embedding model before loading
    # the Text-to-SQL model, following the same
    # memory-management strategy as the benchmark.
    subprocess.run(
        [
            "ollama",
            "stop",
            "embeddinggemma",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # -----------------------------------------------------
    # Spark data lake
    # -----------------------------------------------------

    spark = None

    try:
        spark = (
            SparkSession.builder
            .appName(
                "metadata-aware-query-agent-demo"
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

        spark.sparkContext.setLogLevel(
            "WARN"
        )

        register_data_lake(
            spark
        )

        validator = SparkSQLValidator(
            allowed_tables=set(
                retrieved["selection"]["datasets"]
            )
        )

        repairer = OllamaSQLRepairer(
            model=MODEL,
            num_ctx=NUM_CTX,
            temperature=TEMPERATURE,
        )

        # -------------------------------------------------
        # SQL generation
        # -------------------------------------------------

        prompt = build_prompt(
            question=question,
            metadata_context=metadata_context,
        )

        print()
        print("=" * 78)
        print("GENERATING SPARK SQL")
        print("=" * 78)

        api_result, generation_wall = (
            ollama_generate(prompt)
        )

        generated_sql = clean_sql(
            api_result.get(
                "response",
                "",
            )
        )

        print()
        print("GENERATED SQL")
        print(generated_sql)

        print()
        print(
            "Prompt tokens: "
            f"{api_result.get('prompt_eval_count', 0)}"
        )

        print(
            "Generation wall time: "
            f"{generation_wall:.2f} s"
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        validation = validator.validate(
            sql=generated_sql,
            spark=spark,
            expected_columns=(
                args.expected_columns
            ),
        )

        print()
        print("=" * 78)
        print("VALIDATION")
        print("=" * 78)

        if validation.valid:
            print("PASS")
        else:
            print("FAIL")

            for issue in validation.issues:
                print(
                    f"- [{issue.stage}] "
                    f"{issue.message}"
                )

        final_sql = generated_sql
        repaired = False

        # -------------------------------------------------
        # Optional one-shot repair
        # -------------------------------------------------

        if (
            not validation.valid
            and args.expected_columns
        ):
            print()
            print("=" * 78)
            print("ONE-SHOT REPAIR")
            print("=" * 78)

            repair = repairer.repair(
                question=question,
                metadata_context=(
                    metadata_context
                ),
                invalid_sql=generated_sql,
                validation_result=(
                    validation
                ),
                expected_columns=(
                    args.expected_columns
                ),
                keep_alive="30m",
            )

            final_sql = repair["sql"]
            repaired = True

            print()
            print("REPAIRED SQL")
            print(final_sql)

            validation = (
                validator.validate(
                    sql=final_sql,
                    spark=spark,
                    expected_columns=(
                        args.expected_columns
                    ),
                )
            )

            print()
            print(
                "Repair validation: "
                + (
                    "PASS"
                    if validation.valid
                    else "FAIL"
                )
            )

            if not validation.valid:
                for issue in (
                    validation.issues
                ):
                    print(
                        f"- [{issue.stage}] "
                        f"{issue.message}"
                    )

        elif (
            not validation.valid
            and not args.expected_columns
        ):
            print()
            print(
                "Repair skipped: the current "
                "one-shot repair component uses "
                "an explicit output-column "
                "contract."
            )

        # -------------------------------------------------
        # Execution
        # -------------------------------------------------

        if not validation.valid:
            print()
            print("=" * 78)
            print("FINAL RESULT")
            print("=" * 78)
            print(
                "Query was not executed because "
                "validation failed."
            )

            raise SystemExit(1)

        result = execute_preview(
            spark=spark,
            sql=final_sql,
            max_rows=args.max_rows,
        )

        print()
        print("=" * 78)
        print("RESULT")
        print("=" * 78)

        print(
            "Final SQL source: "
            + (
                "one-shot repair"
                if repaired
                else "initial generation"
            )
        )

        print(
            "Columns: "
            + ", ".join(
                result["columns"]
            )
        )

        for row in result["rows"]:
            print(row)

        if result["truncated"]:
            print(
                f"... output truncated to "
                f"{args.max_rows} rows"
            )

        print(
            f"Spark execution: "
            f"{result['seconds']:.3f} s"
        )

    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp


@dataclass
class ValidationIssue:
    stage: str
    message: str


@dataclass
class ValidationResult:
    valid: bool
    issues: list[ValidationIssue] = field(
        default_factory=list
    )
    tables: list[str] = field(
        default_factory=list
    )
    output_columns: list[str] = field(
        default_factory=list
    )

    def messages(self):
        return [
            f"{issue.stage}: {issue.message}"
            for issue in self.issues
        ]


class SparkSQLValidator:
    """
    Structural and Spark-analysis validation for generated SQL.

    This validator intentionally does NOT use a gold query.

    Checks:
    1. SQL is non-empty.
    2. SQL parses using the Spark dialect.
    3. Exactly one statement is present.
    4. Statement is a read-only query.
    5. Referenced tables belong to the allowed table set.
    6. Spark can analyze the query.
    7. Expected output columns match, when supplied.
    """

    def __init__(
        self,
        allowed_tables,
    ):
        self.allowed_tables = set(
            allowed_tables
        )

    def validate(
        self,
        sql,
        spark,
        expected_columns=None,
    ):
        issues = []

        sql = (
            sql.strip()
            if sql
            else ""
        )

        if not sql:
            return ValidationResult(
                valid=False,
                issues=[
                    ValidationIssue(
                        stage="input",
                        message="SQL is empty.",
                    )
                ],
            )

        # -------------------------------------------------
        # Parse
        # -------------------------------------------------

        try:
            statements = sqlglot.parse(
                sql,
                read="spark",
            )

        except Exception as exc:
            return ValidationResult(
                valid=False,
                issues=[
                    ValidationIssue(
                        stage="parse",
                        message=(
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    )
                ],
            )

        if len(statements) != 1:
            return ValidationResult(
                valid=False,
                issues=[
                    ValidationIssue(
                        stage="structure",
                        message=(
                            "Exactly one SQL statement "
                            "is allowed."
                        ),
                    )
                ],
            )

        expression = statements[0]

        # -------------------------------------------------
        # Read-only query
        # -------------------------------------------------

        if not isinstance(
            expression,
            exp.Query,
        ):
            issues.append(
                ValidationIssue(
                    stage="structure",
                    message=(
                        "Only read-only query "
                        "statements are allowed."
                    ),
                )
            )

        # -------------------------------------------------
        # Table extraction
        # -------------------------------------------------

        tables = sorted(
            {
                table.name
                for table
                in expression.find_all(
                    exp.Table
                )
                if table.name
            }
        )

        unknown_tables = sorted(
            set(tables)
            - self.allowed_tables
        )

        if unknown_tables:
            issues.append(
                ValidationIssue(
                    stage="catalog",
                    message=(
                        "Unknown or disallowed "
                        "table(s): "
                        + ", ".join(
                            unknown_tables
                        )
                    ),
                )
            )

        # Do not hand a destructive or out-of-scope
        # statement to Spark.
        if issues:
            return ValidationResult(
                valid=False,
                issues=issues,
                tables=tables,
            )

        # -------------------------------------------------
        # Spark analysis
        #
        # Accessing the schema forces Spark to resolve
        # tables, columns, expressions and functions,
        # without collecting the dataset result.
        # -------------------------------------------------

        try:
            dataframe = spark.sql(
                sql
            )

            _ = dataframe.schema

            output_columns = (
                dataframe.columns
            )

        except Exception as exc:
            issues.append(
                ValidationIssue(
                    stage="spark_analysis",
                    message=(
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )
            )

            return ValidationResult(
                valid=False,
                issues=issues,
                tables=tables,
            )

        # -------------------------------------------------
        # Output contract
        # -------------------------------------------------

        if expected_columns is not None:
            expected_columns = list(
                expected_columns
            )

            if (
                output_columns
                != expected_columns
            ):
                issues.append(
                    ValidationIssue(
                        stage="output_contract",
                        message=(
                            "Expected output columns "
                            f"{expected_columns}, "
                            "received "
                            f"{output_columns}."
                        ),
                    )
                )

        return ValidationResult(
            valid=not issues,
            issues=issues,
            tables=tables,
            output_columns=(
                output_columns
            ),
        )

import json
import re
import time
import urllib.request


class OllamaSQLRepairer:
    """
    One-shot Spark SQL repair using structured validation feedback.

    The repair model receives:
    - original natural-language question;
    - retrieved metadata context;
    - invalid SQL;
    - validation errors.

    It never receives a gold SQL query or a gold result.
    """

    def __init__(
        self,
        model="qwen2.5-coder:3b",
        base_url="http://127.0.0.1:11434",
        num_ctx=4096,
        temperature=0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature

    @staticmethod
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

    @staticmethod
    def validation_feedback(
        validation_result,
        max_chars=1800,
    ):
        parts = []

        for issue in validation_result.issues:
            message = (
                issue.message
                .strip()
                .splitlines()[0]
            )

            parts.append(
                f"[{issue.stage}] {message}"
            )

        feedback = "\n".join(parts)

        if len(feedback) > max_chars:
            feedback = (
                feedback[:max_chars]
                + "..."
            )

        return feedback

    def build_prompt(
        self,
        question,
        metadata_context,
        invalid_sql,
        validation_result,
        expected_columns,
    ):
        feedback = (
            self.validation_feedback(
                validation_result
            )
        )

        expected = ", ".join(
            expected_columns
        )

        return f"""
You are repairing an invalid Spark SQL query.

Use ONLY the metadata supplied below and the validation
feedback.

Rules:

- Correct the existing query with the smallest reasonable change.
- Apply only changes justified by the validation feedback and supplied metadata.
- Do not modify identifiers that are already valid unless required by the reported error.
- Generate valid Spark SQL.
- Use only physical tables and physical columns present in the metadata.
- Semantic concept names and aliases are metadata labels, not SQL column names.
- Relationship names and JOIN RULE names are metadata identifiers, not tables.
- Never place a relationship or JOIN RULE name in FROM or JOIN.
- Use the supplied physical join condition when a relationship is needed.
- Respect semantic SQL rules.
- Preserve the meaning of the original user question.
- Explicitly alias requested output expressions with the exact requested name and casing.
- The final output columns must be exactly: {expected}
- Return SQL only.
- Do not use markdown.
- Do not explain the repair.

=== ORIGINAL USER QUESTION ===

{question}

=== METADATA ===

{metadata_context}

=== INVALID SQL ===

{invalid_sql}

=== VALIDATION FEEDBACK ===

{feedback}

=== REPAIRED SQL ===
""".strip()

    def repair(
        self,
        question,
        metadata_context,
        invalid_sql,
        validation_result,
        expected_columns,
        keep_alive="30m",
    ):
        prompt = self.build_prompt(
            question=question,
            metadata_context=metadata_context,
            invalid_sql=invalid_sql,
            validation_result=validation_result,
            expected_columns=expected_columns,
        )

        payload = {
            "model": self.model,
            "stream": False,
            "keep_alive": keep_alive,
            "prompt": prompt,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }

        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
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

        wall_time = (
            time.perf_counter()
            - start
        )

        repaired_sql = self.clean_sql(
            result.get(
                "response",
                "",
            )
        )

        return {
            "sql": repaired_sql,
            "prompt": prompt,
            "metrics": {
                "wall_time_seconds": (
                    wall_time
                ),
                "prompt_eval_count": (
                    result.get(
                        "prompt_eval_count",
                        0,
                    )
                ),
                "eval_count": (
                    result.get(
                        "eval_count",
                        0,
                    )
                ),
                "load_duration_seconds": (
                    result.get(
                        "load_duration",
                        0,
                    )
                    / 1_000_000_000
                ),
                "prompt_eval_duration_seconds": (
                    result.get(
                        "prompt_eval_duration",
                        0,
                    )
                    / 1_000_000_000
                ),
                "eval_duration_seconds": (
                    result.get(
                        "eval_duration",
                        0,
                    )
                    / 1_000_000_000
                ),
            },
        }

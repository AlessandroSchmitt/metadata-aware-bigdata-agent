import ast
import hashlib
import json
import os
import statistics
import subprocess
import time
import urllib
import urllib.request
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DEVELOPMENT_TAG = "development-freeze-v1"
RESULTS_TAG = "heldout-results-v1"

FROZEN_RUNNER_PATH = (
    "scripts/experiments/"
    "run_canonical_abc_benchmark.py"
)

FULL_CONTEXT_PATH = (
    "artifacts/benchmarks/"
    "full_catalog_context.txt"
)

HELDOUT_RESULTS_PATH = (
    "artifacts/benchmarks/"
    "heldout_abc_benchmark.json"
)

OUTPUT_PATH = (
    ROOT
    / "artifacts/benchmarks/"
      "controlled_latency_benchmark.json"
)

# Representative frozen held-out questions:
#
# Q01_H1: simple single-source aggregation
# Q06_H1: two-source temporal/weather join
# Q08_H1: three-source geographic/weather query
#
# All three had perfect metadata recall in the frozen
# held-out evaluation. This latency experiment does NOT
# change or re-evaluate accuracy.
QUERY_IDS = [
    "Q01_H1",
    "Q06_H1",
    "Q08_H1",
]

CONTROLLED_REPETITIONS = 2

DRY_RUN = (
    os.environ.get(
        "LATENCY_DRY_RUN",
        "0",
    )
    == "1"
)


def git_show(tag, path):
    return subprocess.check_output(
        [
            "git",
            "show",
            f"{tag}:{path}",
        ],
        cwd=ROOT,
        text=True,
    )


def git_tag_commit(tag):
    return subprocess.check_output(
        [
            "git",
            "rev-list",
            "-n",
            "1",
            tag,
        ],
        cwd=ROOT,
        text=True,
    ).strip()


def sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------
# Load the EXACT frozen generation implementation from
# development-freeze-v1 without executing its top-level
# benchmark code.
# ---------------------------------------------------------

frozen_source = git_show(
    DEVELOPMENT_TAG,
    FROZEN_RUNNER_PATH,
)

tree = ast.parse(frozen_source)


def frozen_literal(name):
    for node in tree.body:

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == name
                ):
                    return ast.literal_eval(
                        node.value
                    )

        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(
                node.target,
                ast.Name,
            )
            and node.target.id == name
        ):
            return ast.literal_eval(
                node.value
            )

    raise RuntimeError(
        f"Frozen constant not found: {name}"
    )


MODEL = frozen_literal("MODEL")
NUM_CTX = frozen_literal("NUM_CTX")
TEMPERATURE = frozen_literal(
    "TEMPERATURE"
)


wanted_functions = {
    "ollama_generate",
    "build_prompt",
    "metric_seconds",
}

function_nodes = []

for node in tree.body:
    if (
        isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        )
        and node.name in wanted_functions
    ):
        function_nodes.append(node)


found_functions = {
    node.name
    for node in function_nodes
}

if found_functions != wanted_functions:
    raise RuntimeError(
        "Could not extract all frozen "
        "generation functions."
    )


frozen_module = ast.Module(
    body=function_nodes,
    type_ignores=[],
)

ast.fix_missing_locations(
    frozen_module
)


namespace = {
    "json": json,
    "time": time,
    "urllib": urllib,
    "MODEL": MODEL,
    "NUM_CTX": NUM_CTX,
    "TEMPERATURE": TEMPERATURE,
}


exec(
    compile(
        frozen_module,
        filename=(
            f"{DEVELOPMENT_TAG}:"
            f"{FROZEN_RUNNER_PATH}"
        ),
        mode="exec",
    ),
    namespace,
)


ollama_generate = namespace[
    "ollama_generate"
]

build_prompt = namespace[
    "build_prompt"
]

metric_seconds = namespace[
    "metric_seconds"
]


# ---------------------------------------------------------
# Load EXACT frozen Full Catalog and held-out contexts.
# ---------------------------------------------------------

full_context = git_show(
    DEVELOPMENT_TAG,
    FULL_CONTEXT_PATH,
)

heldout_data = json.loads(
    git_show(
        RESULTS_TAG,
        HELDOUT_RESULTS_PATH,
    )
)


records_by_id = {
    record["id"]: record
    for record in heldout_data[
        "records"
    ]
}


missing_ids = [
    query_id
    for query_id in QUERY_IDS
    if query_id not in records_by_id
]

if missing_ids:
    raise RuntimeError(
        "Missing frozen held-out records: "
        + ", ".join(missing_ids)
    )


selected = [
    records_by_id[query_id]
    for query_id in QUERY_IDS
]


# ---------------------------------------------------------
# Verify that reconstructed prompts are EXACTLY consistent
# with the prompt character counts archived in the frozen
# held-out benchmark.
# ---------------------------------------------------------

for record in selected:

    question = record["question"]

    a_prompt = build_prompt(
        question,
        full_context,
    )

    b_prompt = build_prompt(
        question,
        record["retrieval"]["context"],
    )

    if (
        len(a_prompt)
        != record["A"][
            "prompt_characters"
        ]
    ):
        raise RuntimeError(
            f"{record['id']}: "
            "A prompt reconstruction mismatch."
        )

    if (
        len(b_prompt)
        != record["B"][
            "prompt_characters"
        ]
    ):
        raise RuntimeError(
            f"{record['id']}: "
            "B prompt reconstruction mismatch."
        )


# ---------------------------------------------------------
# Timing helpers.
# ---------------------------------------------------------

def generation_metrics(
    api_result,
    wall_seconds,
):
    return {
        "prompt_tokens": (
            api_result.get(
                "prompt_eval_count",
                0,
            )
        ),
        "generated_tokens": (
            api_result.get(
                "eval_count",
                0,
            )
        ),
        "load_seconds": (
            metric_seconds(
                api_result,
                "load_duration",
            )
        ),
        "prompt_eval_seconds": (
            metric_seconds(
                api_result,
                "prompt_eval_duration",
            )
        ),
        "generation_seconds": (
            metric_seconds(
                api_result,
                "eval_duration",
            )
        ),
        "total_api_seconds": (
            metric_seconds(
                api_result,
                "total_duration",
            )
        ),
        "wall_seconds": wall_seconds,
    }


def stop_model():
    result = subprocess.run(
        [
            "ollama",
            "stop",
            MODEL,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Could not stop Ollama model:\n"
            + result.stderr
        )

    time.sleep(2)


def cold_load_and_warm(label):
    print()
    print("=" * 78)
    print(
        f"COLD LOAD + NEUTRAL WARM-UP: "
        f"{label}"
    )
    print("=" * 78)

    stop_model()

    api_result, wall = (
        ollama_generate(
            "Return only this SQL statement:\n"
            "SELECT 1 AS ready",
            keep_alive="30m",
        )
    )

    metrics = generation_metrics(
        api_result,
        wall,
    )

    print(
        f"Wall:        "
        f"{metrics['wall_seconds']:.2f} s"
    )
    print(
        f"Load:        "
        f"{metrics['load_seconds']:.2f} s"
    )
    print(
        f"Prompt eval: "
        f"{metrics['prompt_eval_seconds']:.2f} s"
    )
    print(
        f"Generation:  "
        f"{metrics['generation_seconds']:.2f} s"
    )

    return metrics


def run_generation(
    mode,
    repetition,
    query_index,
    record,
    configuration,
    isolate_prefix_cache,
):
    question = record["question"]

    if configuration == "A":
        metadata_context = full_context
    else:
        metadata_context = (
            record["retrieval"]["context"]
        )

    base_prompt = build_prompt(
        question,
        metadata_context,
    )

    prompt = base_prompt
    isolation_id = None

    if isolate_prefix_cache:
        isolation_id = uuid.uuid4().hex

        # The frozen core prompt remains byte-for-byte
        # unchanged. A short unique timing header is placed
        # BEFORE it to minimize cross-request prefix reuse.
        prompt = (
            "LATENCY_MEASUREMENT_ID="
            f"{isolation_id}\n"
            "This identifier is only for timing "
            "isolation and does not change the SQL "
            "task.\n\n"
            + base_prompt
        )

    api_result, wall = (
        ollama_generate(
            prompt,
            keep_alive="30m",
        )
    )

    metrics = generation_metrics(
        api_result,
        wall,
    )

    archived = record[
        configuration
    ]

    result = {
        "mode": mode,
        "repetition": repetition,
        "query_id": record["id"],
        "category": record["category"],
        "configuration": configuration,
        "isolation_id": isolation_id,
        "base_prompt_characters": len(
            base_prompt
        ),
        "archived_prompt_tokens": (
            archived["prompt_tokens"]
        ),
        **metrics,
    }

    print(
        f"{record['id']} "
        f"{configuration} | "
        f"prompt_tokens="
        f"{metrics['prompt_tokens']} | "
        f"prompt_eval="
        f"{metrics['prompt_eval_seconds']:.2f}s | "
        f"generation="
        f"{metrics['generation_seconds']:.2f}s | "
        f"wall="
        f"{metrics['wall_seconds']:.2f}s"
    )

    return result


def aggregate(records):
    output = {}

    for configuration in ["A", "B"]:

        subset = [
            record
            for record in records
            if (
                record["configuration"]
                == configuration
            )
        ]

        if not subset:
            continue

        def values(key):
            return [
                float(record[key])
                for record in subset
            ]

        output[configuration] = {
            "samples": len(subset),

            "mean_prompt_tokens": (
                statistics.mean(
                    values(
                        "prompt_tokens"
                    )
                )
            ),

            "mean_archived_prompt_tokens": (
                statistics.mean(
                    values(
                        "archived_prompt_tokens"
                    )
                )
            ),

            "mean_prompt_eval_seconds": (
                statistics.mean(
                    values(
                        "prompt_eval_seconds"
                    )
                )
            ),

            "median_prompt_eval_seconds": (
                statistics.median(
                    values(
                        "prompt_eval_seconds"
                    )
                )
            ),

            "mean_generation_seconds": (
                statistics.mean(
                    values(
                        "generation_seconds"
                    )
                )
            ),

            "mean_total_api_seconds": (
                statistics.mean(
                    values(
                        "total_api_seconds"
                    )
                )
            ),

            "mean_wall_seconds": (
                statistics.mean(
                    values(
                        "wall_seconds"
                    )
                )
            ),
        }

    if (
        "A" in output
        and "B" in output
    ):
        a = output["A"]
        b = output["B"]

        output["B_vs_A"] = {
            "prompt_token_reduction_percent": (
                100.0
                * (
                    1.0
                    - (
                        b["mean_prompt_tokens"]
                        / a["mean_prompt_tokens"]
                    )
                )
            ),

            "archived_prompt_token_reduction_percent": (
                100.0
                * (
                    1.0
                    - (
                        b[
                            "mean_archived_prompt_tokens"
                        ]
                        / a[
                            "mean_archived_prompt_tokens"
                        ]
                    )
                )
            ),

            "prompt_eval_time_reduction_percent": (
                100.0
                * (
                    1.0
                    - (
                        b[
                            "mean_prompt_eval_seconds"
                        ]
                        / a[
                            "mean_prompt_eval_seconds"
                        ]
                    )
                )
            ),

            "wall_time_reduction_percent": (
                100.0
                * (
                    1.0
                    - (
                        b[
                            "mean_wall_seconds"
                        ]
                        / a[
                            "mean_wall_seconds"
                        ]
                    )
                )
            ),
        }

    return output


# ---------------------------------------------------------
# Dry-run: prove the experimental inputs are frozen and
# prompt reconstruction is consistent without calling
# Ollama.
# ---------------------------------------------------------

print("=" * 78)
print("CONTROLLED LATENCY BENCHMARK")
print("=" * 78)

print(
    f"Development freeze: "
    f"{DEVELOPMENT_TAG} "
    f"({git_tag_commit(DEVELOPMENT_TAG)[:7]})"
)

print(
    f"Held-out results:   "
    f"{RESULTS_TAG} "
    f"({git_tag_commit(RESULTS_TAG)[:7]})"
)

print(
    f"Model:              {MODEL}"
)

print(
    f"Temperature:        {TEMPERATURE}"
)

print(
    f"Context size:       {NUM_CTX}"
)

print(
    f"Queries:            "
    f"{', '.join(QUERY_IDS)}"
)

print(
    f"Controlled reps:    "
    f"{CONTROLLED_REPETITIONS}"
)

print(
    f"Full catalog chars: "
    f"{len(full_context)}"
)

print(
    f"Full catalog SHA:   "
    f"{sha256_text(full_context)[:16]}..."
)

print()
print("FROZEN PROMPT RECONSTRUCTION")

for record in selected:
    print(
        f"  {record['id']}: "
        f"A={record['A']['prompt_characters']} chars, "
        f"B={record['B']['prompt_characters']} chars "
        f"-> PASS"
    )


if DRY_RUN:
    print()
    print("=" * 78)
    print("FINAL RESULT")
    print("=" * 78)
    print(
        "CONTROLLED LATENCY BENCHMARK DRY-RUN: PASS"
    )
    raise SystemExit(0)


# ---------------------------------------------------------
# Phase 1: controlled warm-model measurements.
#
# A unique non-semantic header is placed before the frozen
# core prompt for every request to minimize cross-request
# prefix reuse. The model is loaded only once for the phase.
#
# Each query runs twice, reversing A/B order on repetition
# 2 so each configuration is first once per query.
# ---------------------------------------------------------

cold_controlled = cold_load_and_warm(
    "CONTROLLED PHASE"
)

controlled_records = []

print()
print("=" * 78)
print("CONTROLLED WARM-MODEL / PREFIX-ISOLATED PHASE")
print("=" * 78)

for repetition in range(
    1,
    CONTROLLED_REPETITIONS + 1,
):

    print()
    print(
        f"--- REPETITION "
        f"{repetition}/"
        f"{CONTROLLED_REPETITIONS} ---"
    )

    for query_index, record in enumerate(
        selected,
        start=1,
    ):

        if (
            (
                query_index
                + repetition
            )
            % 2
            == 0
        ):
            order = ["A", "B"]
        else:
            order = ["B", "A"]

        print(
            f"{record['id']} order: "
            f"{' -> '.join(order)}"
        )

        for configuration in order:
            controlled_records.append(
                run_generation(
                    mode="controlled_prefix_isolated",
                    repetition=repetition,
                    query_index=query_index,
                    record=record,
                    configuration=configuration,
                    isolate_prefix_cache=True,
                )
            )


# ---------------------------------------------------------
# Phase 2: natural-cache exact-prompt measurements.
#
# Reset the model first, warm it with the same neutral
# prompt, then execute each frozen A/B prompt EXACTLY,
# without any timing header.
#
# One pass is used intentionally: repeated identical
# prompts would measure an artificial repeated-query cache
# workload rather than the normal cross-query workload.
# ---------------------------------------------------------

cold_natural = cold_load_and_warm(
    "NATURAL-CACHE PHASE"
)

natural_records = []

print()
print("=" * 78)
print("NATURAL-CACHE / EXACT FROZEN PROMPTS")
print("=" * 78)

for query_index, record in enumerate(
    selected,
    start=1,
):

    order = (
        ["A", "B"]
        if query_index % 2 == 1
        else ["B", "A"]
    )

    print(
        f"{record['id']} order: "
        f"{' -> '.join(order)}"
    )

    for configuration in order:
        natural_records.append(
            run_generation(
                mode="natural_cache_exact_prompt",
                repetition=1,
                query_index=query_index,
                record=record,
                configuration=configuration,
                isolate_prefix_cache=False,
            )
        )


controlled_summary = aggregate(
    controlled_records
)

natural_summary = aggregate(
    natural_records
)


output = {
    "benchmark": {
        "name": (
            "Controlled Ollama Prompt Latency Benchmark"
        ),
        "purpose": (
            "Separate frozen-prompt token efficiency "
            "from cold model loading and cross-request "
            "prefix reuse."
        ),
        "development_freeze": DEVELOPMENT_TAG,
        "development_commit": (
            git_tag_commit(
                DEVELOPMENT_TAG
            )
        ),
        "heldout_results_tag": RESULTS_TAG,
        "heldout_results_commit": (
            git_tag_commit(
                RESULTS_TAG
            )
        ),
        "query_ids": QUERY_IDS,
        "controlled_repetitions": (
            CONTROLLED_REPETITIONS
        ),
    },

    "configuration": {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "num_ctx": NUM_CTX,
        "full_catalog_characters": len(
            full_context
        ),
        "full_catalog_sha256": (
            sha256_text(
                full_context
            )
        ),
    },

    "cold_loads": {
        "controlled_phase": (
            cold_controlled
        ),
        "natural_cache_phase": (
            cold_natural
        ),
    },

    "controlled_prefix_isolated": {
        "summary": controlled_summary,
        "records": controlled_records,
    },

    "natural_cache_exact_prompt": {
        "summary": natural_summary,
        "records": natural_records,
    },
}


OUTPUT_PATH.write_text(
    json.dumps(
        output,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


print()
print("=" * 78)
print("CONTROLLED PHASE SUMMARY")
print("=" * 78)

for configuration in ["A", "B"]:
    values = controlled_summary[
        configuration
    ]

    print(
        f"{configuration}: "
        f"samples={values['samples']}, "
        f"mean_prompt_tokens="
        f"{values['mean_prompt_tokens']:.1f}, "
        f"mean_prompt_eval="
        f"{values['mean_prompt_eval_seconds']:.2f}s, "
        f"median_prompt_eval="
        f"{values['median_prompt_eval_seconds']:.2f}s, "
        f"mean_wall="
        f"{values['mean_wall_seconds']:.2f}s"
    )

comparison = controlled_summary[
    "B_vs_A"
]

print(
    "B/A prompt token reduction: "
    f"{comparison['prompt_token_reduction_percent']:.2f}%"
)

print(
    "B/A archived token reduction: "
    f"{comparison['archived_prompt_token_reduction_percent']:.2f}%"
)

print(
    "B/A prompt-eval time reduction: "
    f"{comparison['prompt_eval_time_reduction_percent']:.2f}%"
)


print()
print("=" * 78)
print("NATURAL-CACHE PHASE SUMMARY")
print("=" * 78)

for configuration in ["A", "B"]:
    values = natural_summary[
        configuration
    ]

    print(
        f"{configuration}: "
        f"samples={values['samples']}, "
        f"mean_prompt_tokens="
        f"{values['mean_prompt_tokens']:.1f}, "
        f"mean_prompt_eval="
        f"{values['mean_prompt_eval_seconds']:.2f}s, "
        f"mean_wall="
        f"{values['mean_wall_seconds']:.2f}s"
    )

comparison = natural_summary[
    "B_vs_A"
]

print(
    "B/A prompt token reduction: "
    f"{comparison['prompt_token_reduction_percent']:.2f}%"
)

print(
    "B/A prompt-eval time reduction: "
    f"{comparison['prompt_eval_time_reduction_percent']:.2f}%"
)


print()
print("=" * 78)
print("OUTPUT")
print("=" * 78)

print(
    OUTPUT_PATH.relative_to(ROOT)
)

print()
print("=" * 78)
print("FINAL RESULT")
print("=" * 78)

print(
    "CONTROLLED LATENCY BENCHMARK: COMPLETED"
)

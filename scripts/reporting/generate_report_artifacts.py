import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_DIR = ROOT / "artifacts" / "benchmarks"
OUTPUT_ROOT = ROOT / "artifacts" / "report"
FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"

HELDOUT_PATH = BENCHMARK_DIR / "heldout_abc_summary.json"
LATENCY_PATH = BENCHMARK_DIR / "controlled_latency_benchmark.json"
CATALOG_SCALE_PATH = BENCHMARK_DIR / "catalog_scale_benchmark.json"
DATA_VOLUME_PATH = BENCHMARK_DIR / "data_volume_scaling_benchmark.json"

DRY_RUN = os.environ.get("REPORT_ARTIFACTS_DRY_RUN", "0") == "1"


def load_json(path):
    if not path.exists():
        raise RuntimeError(
            f"Required benchmark file missing: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


heldout = load_json(HELDOUT_PATH)
latency = load_json(LATENCY_PATH)
catalog_scale = load_json(CATALOG_SCALE_PATH)
data_volume = load_json(DATA_VOLUME_PATH)


# ---------------------------------------------------------
# Extract frozen results
# ---------------------------------------------------------

heldout_overall = heldout["overall"]
heldout_derived = heldout["derived"]

a = heldout_overall["A"]
b = heldout_overall["B"]
c = heldout_overall["C"]

controlled = (
    latency["controlled_prefix_isolated"]["summary"]
)

catalog_results = catalog_scale["results"]
volume_summary = data_volume["summary"]


# ---------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------

expected_correct = {
    "A": 14,
    "B": 16,
    "C": 17,
}

observed_correct = (
    heldout_derived["correct_query_counts"]
)

if observed_correct != expected_correct:
    raise RuntimeError(
        "Held-out correctness counts do not match "
        "the frozen official results.\n"
        f"Expected: {expected_correct}\n"
        f"Observed: {observed_correct}"
    )


if set(catalog_results.keys()) != {"4", "8", "16"}:
    raise RuntimeError(
        "Unexpected catalog-scale configuration."
    )


if set(volume_summary.keys()) != {
    "100k",
    "500k",
    "1m",
    "full",
}:
    raise RuntimeError(
        "Unexpected data-volume configuration."
    )


# ---------------------------------------------------------
# Dry-run report
# ---------------------------------------------------------

print("=" * 78)
print("FINAL REPORT ARTIFACT GENERATOR")
print("=" * 78)

print()
print("SOURCE BENCHMARKS")

for path in [
    HELDOUT_PATH,
    LATENCY_PATH,
    CATALOG_SCALE_PATH,
    DATA_VOLUME_PATH,
]:
    print(
        f"PASS  {path.relative_to(ROOT)}"
    )


print()
print("HELD-OUT TEXT-TO-SPARK-SQL")

print(
    f"A result accuracy: "
    f"{a['result_accuracy']:.3f}"
)

print(
    f"B result accuracy: "
    f"{b['result_accuracy']:.3f}"
)

print(
    f"C result accuracy: "
    f"{c['result_accuracy']:.3f}"
)

print(
    f"A mean prompt tokens: "
    f"{a['mean_prompt_tokens']:.2f}"
)

print(
    f"B mean prompt tokens: "
    f"{b['mean_prompt_tokens']:.2f}"
)

print(
    "Prompt token reduction: "
    f"{heldout_overall['B_vs_A']['mean_prompt_token_reduction_percent']:.2f}%"
)


print()
print("CONTROLLED LATENCY")

print(
    "A mean prompt evaluation: "
    f"{controlled['A']['mean_prompt_eval_seconds']:.2f}s"
)

print(
    "B mean prompt evaluation: "
    f"{controlled['B']['mean_prompt_eval_seconds']:.2f}s"
)

print(
    "Prompt-evaluation reduction: "
    f"{controlled['B_vs_A']['prompt_eval_time_reduction_percent']:.2f}%"
)


print()
print("CATALOG SCALE")

for scale in ["4", "8", "16"]:
    retrieval = (
        catalog_results[scale]["retrieval"]
    )

    summary = retrieval["summary"]

    print(
        f"{scale:>2} sources | "
        f"full={retrieval['full_catalog_characters']:,} chars | "
        f"context={summary['mean_context_characters']:.1f} chars | "
        f"recall={summary['macro_recall']:.3f}"
    )


print()
print("DATA VOLUME")

for scale in ["100k", "500k", "1m", "full"]:
    item = volume_summary[scale]

    weather_join = (
        item["workloads"]
        ["weather_temporal_join"]
        ["median_seconds"]
    )

    print(
        f"{scale:>4} | "
        f"rows={item['rows']:,} | "
        f"weather join={weather_join:.3f}s"
    )


if DRY_RUN:
    print()
    print("=" * 78)
    print("FINAL RESULT")
    print("=" * 78)
    print(
        "FINAL REPORT ARTIFACT GENERATOR "
        "DRY-RUN: PASS"
    )

    raise SystemExit(0)


# ---------------------------------------------------------
# Output directories and figure helpers
# ---------------------------------------------------------

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def save_figure(fig, filename):
    pdf_path = FIGURE_DIR / f"{filename}.pdf"
    png_path = FIGURE_DIR / f"{filename}.png"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return pdf_path, png_path


def annotate_bars(ax, bars, formatter):
    for bar in bars:
        value = bar.get_height()

        ax.annotate(
            formatter(value),
            (
                bar.get_x()
                + bar.get_width() / 2,
                value,
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


generated_files = []


# ---------------------------------------------------------
# Figure 1
# Held-out A/B/C performance
# ---------------------------------------------------------

metrics = [
    "Validation",
    "Execution",
    "Result accuracy",
]

a_values = [
    100 * a["validation_rate"],
    100 * a["execution_rate"],
    100 * a["result_accuracy"],
]

b_values = [
    100 * b["validation_rate"],
    100 * b["execution_rate"],
    100 * b["result_accuracy"],
]

c_values = [
    100 * c["validation_rate"],
    100 * c["execution_rate"],
    100 * c["result_accuracy"],
]

x = [0, 1, 2]
width = 0.24

fig, ax = plt.subplots(
    figsize=(7.2, 4.4)
)

bars_a = ax.bar(
    [value - width for value in x],
    a_values,
    width,
    label="A — Full Catalog",
)

bars_b = ax.bar(
    x,
    b_values,
    width,
    label="B — Relation-Aware",
)

bars_c = ax.bar(
    [value + width for value in x],
    c_values,
    width,
    label="C — Validation + Repair",
)

ax.set_ylabel("Rate (%)")
ax.set_ylim(0, 105)

ax.set_xticks(
    x,
    metrics,
)

ax.set_title(
    "Held-out Text-to-Spark-SQL performance"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25,
)

annotate_bars(
    ax,
    bars_a,
    lambda value: f"{value:.0f}%",
)

annotate_bars(
    ax,
    bars_b,
    lambda value: f"{value:.0f}%",
)

annotate_bars(
    ax,
    bars_c,
    lambda value: f"{value:.0f}%",
)

generated_files.extend(
    save_figure(
        fig,
        "heldout_abc_rates",
    )
)


# ---------------------------------------------------------
# Figure 2
# Prompt token efficiency
# ---------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(5.8, 4.4)
)

bars = ax.bar(
    [
        "A — Full Catalog",
        "B — Relation-Aware",
    ],
    [
        a["mean_prompt_tokens"],
        b["mean_prompt_tokens"],
    ],
)

ax.set_ylabel(
    "Mean prompt tokens"
)

ax.set_title(
    "Prompt size on 20 held-out queries"
)

ax.grid(
    axis="y",
    alpha=0.25,
)

annotate_bars(
    ax,
    bars,
    lambda value: f"{value:.0f}",
)

generated_files.extend(
    save_figure(
        fig,
        "prompt_token_efficiency",
    )
)


print()
print("=" * 78)
print("GENERATED FIGURES — PART 1")
print("=" * 78)

for path in generated_files:
    print(
        path.relative_to(ROOT)
    )

print()
print(
    "REPORT FIGURES PART 1: PASS"
)


# ---------------------------------------------------------
# Figure 3
# Controlled latency: prompt evaluation time
# ---------------------------------------------------------

part2_files = []

with LATENCY_PATH.open(
    "r",
    encoding="utf-8",
) as handle:
    latency_payload = json.load(handle)

latency_summary = latency_payload[
    "controlled_prefix_isolated"
]["summary"]

lat_a = latency_summary["A"][
    "mean_prompt_eval_seconds"
]
lat_b = latency_summary["B"][
    "mean_prompt_eval_seconds"
]

fig, ax = plt.subplots(
    figsize=(5.8, 4.4)
)

bars = ax.bar(
    [
        "A — Full Catalog",
        "B — Relation-Aware",
    ],
    [
        lat_a,
        lat_b,
    ],
)

ax.set_ylabel(
    "Mean prompt evaluation time (s)"
)

ax.set_title(
    "Controlled latency benchmark"
)

ax.grid(
    axis="y",
    alpha=0.25,
)

annotate_bars(
    ax,
    bars,
    lambda value: f"{value:.1f}s",
)

paths = save_figure(
    fig,
    "controlled_latency_prompt_eval",
)

generated_files.extend(paths)
part2_files.extend(paths)


# ---------------------------------------------------------
# Figure 4
# Catalog-scale retrieval recall
# ---------------------------------------------------------

with CATALOG_SCALE_PATH.open(
    "r",
    encoding="utf-8",
) as handle:
    catalog_scale_payload = json.load(handle)

scale_labels = []
mean_recalls = []

for scale_key in sorted(
    catalog_scale_payload["results"],
    key=int,
):
    retrieval = (
        catalog_scale_payload["results"][
            scale_key
        ]["retrieval"]
    )

    records = retrieval["records"]

    mean_recall = sum(
        record["recall"]
        for record in records
    ) / len(records)

    scale_labels.append(scale_key)
    mean_recalls.append(
        100 * mean_recall
    )

fig, ax = plt.subplots(
    figsize=(6.2, 4.4)
)

bars = ax.bar(
    scale_labels,
    mean_recalls,
)

ax.set_xlabel("Catalog scale (sources)")
ax.set_ylabel("Mean recall (%)")
ax.set_ylim(0, 105)

ax.set_title(
    "Retrieval recall under catalog scaling"
)

ax.grid(
    axis="y",
    alpha=0.25,
)

annotate_bars(
    ax,
    bars,
    lambda value: f"{value:.1f}%",
)

paths = save_figure(
    fig,
    "catalog_scale_recall",
)

generated_files.extend(paths)
part2_files.extend(paths)


# ---------------------------------------------------------
# Figure 5
# Catalog-scale full context vs retrieved context
# ---------------------------------------------------------

scale_labels = []
full_chars = []
retrieved_chars = []

for scale_key in sorted(
    catalog_scale_payload["results"],
    key=int,
):
    retrieval = (
        catalog_scale_payload["results"][
            scale_key
        ]["retrieval"]
    )

    records = retrieval["records"]

    mean_context = sum(
        record["context_characters"]
        for record in records
    ) / len(records)

    scale_labels.append(scale_key)
    full_chars.append(
        retrieval["full_catalog_characters"]
    )
    retrieved_chars.append(
        mean_context
    )

x = list(
    range(len(scale_labels))
)
width = 0.35

fig, ax = plt.subplots(
    figsize=(7.0, 4.4)
)

bars_full = ax.bar(
    [value - width / 2 for value in x],
    full_chars,
    width,
    label="Full catalog",
)

bars_retrieved = ax.bar(
    [value + width / 2 for value in x],
    retrieved_chars,
    width,
    label="Retrieved context",
)

ax.set_xticks(
    x,
    scale_labels,
)

ax.set_xlabel("Catalog scale (sources)")
ax.set_ylabel("Characters")
ax.set_title(
    "Context size under catalog scaling"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25,
)

annotate_bars(
    ax,
    bars_full,
    lambda value: f"{value:.0f}",
)

annotate_bars(
    ax,
    bars_retrieved,
    lambda value: f"{value:.0f}",
)

paths = save_figure(
    fig,
    "catalog_scale_context_size",
)

generated_files.extend(paths)
part2_files.extend(paths)


print()
print("=" * 78)
print("GENERATED FIGURES — PART 2")
print("=" * 78)

for path in part2_files:
    print(
        path.relative_to(ROOT)
    )

print()
print(
    "REPORT FIGURES PART 2: PASS"
)


# ---------------------------------------------------------
# Figure 6
# Spark data-volume scaling
# ---------------------------------------------------------

part3_files = []

volume_labels = [
    "100k",
    "500k",
    "1m",
    "full",
]

volume_rows = [
    volume_summary[label]["rows"]
    for label in volume_labels
]

volume_millions = [
    rows / 1_000_000
    for rows in volume_rows
]

workloads = {
    "scan_aggregation":
        "Scan + aggregation",

    "pickup_zone_grouping":
        "Pickup-zone grouping",

    "weather_temporal_join":
        "Weather temporal join",
}


fig, ax = plt.subplots(
    figsize=(7.0, 4.6)
)

for workload, display_name in workloads.items():

    medians = [
        volume_summary[label]
        ["workloads"]
        [workload]
        ["median_seconds"]
        for label in volume_labels
    ]

    ax.plot(
        volume_millions,
        medians,
        marker="o",
        label=display_name,
    )


ax.set_xlabel(
    "Yellow Taxi rows (millions)"
)

ax.set_ylabel(
    "Median runtime (seconds)"
)

ax.set_title(
    "Spark runtime under increasing data volume"
)

ax.set_xticks(
    volume_millions,
    [
        "0.1M",
        "0.5M",
        "1M",
        "2.96M",
    ],
)

ax.legend()

ax.grid(
    alpha=0.25,
)


paths = save_figure(
    fig,
    "spark_data_volume_scaling",
)

generated_files.extend(paths)
part3_files.extend(paths)


print()
print("=" * 78)
print("GENERATED FIGURES — PART 3")
print("=" * 78)

for path in part3_files:
    print(
        path.relative_to(ROOT)
    )

print()
print(
    "REPORT FIGURES PART 3: PASS"
)


# ---------------------------------------------------------
# Table helpers
# ---------------------------------------------------------

def write_csv_table(filename, headers, rows):
    path = TABLE_DIR / filename

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)

    return path


def write_latex_table(
    filename,
    caption,
    label,
    headers,
    rows,
    alignment,
):
    path = TABLE_DIR / filename

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]

    for row in rows:
        lines.append(
            " & ".join(
                str(value)
                for value in row
            )
            + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return path


part4_files = []


# ---------------------------------------------------------
# Table 1
# Held-out A/B/C evaluation
# ---------------------------------------------------------

headers = [
    "Metric",
    "A Full",
    "B Retrieval",
    "C Repair",
]

tex_rows = [
    [
        "Validation rate",
        f"{100*a['validation_rate']:.0f}\\%",
        f"{100*b['validation_rate']:.0f}\\%",
        f"{100*c['validation_rate']:.0f}\\%",
    ],
    [
        "Execution rate",
        f"{100*a['execution_rate']:.0f}\\%",
        f"{100*b['execution_rate']:.0f}\\%",
        f"{100*c['execution_rate']:.0f}\\%",
    ],
    [
        "Result accuracy",
        f"{100*a['result_accuracy']:.0f}\\%",
        f"{100*b['result_accuracy']:.0f}\\%",
        f"{100*c['result_accuracy']:.0f}\\%",
    ],
    [
        "Correct queries",
        "14/20",
        "16/20",
        "17/20",
    ],
]

csv_rows = [
    [
        "Validation rate",
        a["validation_rate"],
        b["validation_rate"],
        c["validation_rate"],
    ],
    [
        "Execution rate",
        a["execution_rate"],
        b["execution_rate"],
        c["execution_rate"],
    ],
    [
        "Result accuracy",
        a["result_accuracy"],
        b["result_accuracy"],
        c["result_accuracy"],
    ],
    [
        "Correct queries",
        14,
        16,
        17,
    ],
]

path = write_latex_table(
    "heldout_abc.tex",
    "Held-out Text-to-Spark-SQL evaluation.",
    "tab:heldout-abc",
    headers,
    tex_rows,
    "lrrr",
)

part4_files.append(path)
generated_files.append(path)

path = write_csv_table(
    "heldout_abc.csv",
    headers,
    csv_rows,
)

part4_files.append(path)
generated_files.append(path)


# ---------------------------------------------------------
# Table 2
# Prompt efficiency and controlled latency
# ---------------------------------------------------------

headers = [
    "Metric",
    "A Full",
    "B Retrieval",
    "Reduction",
]

token_reduction = (
    heldout_overall[
        "B_vs_A"
    ][
        "mean_prompt_token_reduction_percent"
    ]
)

prompt_eval_reduction = (
    controlled[
        "B_vs_A"
    ][
        "prompt_eval_time_reduction_percent"
    ]
)

wall_reduction = (
    controlled[
        "B_vs_A"
    ][
        "wall_time_reduction_percent"
    ]
)

tex_rows = [
    [
        "Mean prompt tokens",
        f"{a['mean_prompt_tokens']:.2f}",
        f"{b['mean_prompt_tokens']:.2f}",
        f"{token_reduction:.2f}\\%",
    ],
    [
        "Prompt evaluation (s)",
        f"{controlled['A']['mean_prompt_eval_seconds']:.2f}",
        f"{controlled['B']['mean_prompt_eval_seconds']:.2f}",
        f"{prompt_eval_reduction:.2f}\\%",
    ],
    [
        "Wall-clock time (s)",
        f"{controlled['A']['mean_wall_seconds']:.2f}",
        f"{controlled['B']['mean_wall_seconds']:.2f}",
        f"{wall_reduction:.2f}\\%",
    ],
]

csv_rows = [
    [
        "Mean prompt tokens",
        a["mean_prompt_tokens"],
        b["mean_prompt_tokens"],
        token_reduction,
    ],
    [
        "Prompt evaluation (s)",
        controlled["A"][
            "mean_prompt_eval_seconds"
        ],
        controlled["B"][
            "mean_prompt_eval_seconds"
        ],
        prompt_eval_reduction,
    ],
    [
        "Wall-clock time (s)",
        controlled["A"][
            "mean_wall_seconds"
        ],
        controlled["B"][
            "mean_wall_seconds"
        ],
        wall_reduction,
    ],
]

path = write_latex_table(
    "efficiency_latency.tex",
    "Prompt-size and controlled-latency comparison.",
    "tab:efficiency-latency",
    headers,
    tex_rows,
    "lrrr",
)

part4_files.append(path)
generated_files.append(path)

path = write_csv_table(
    "efficiency_latency.csv",
    headers,
    csv_rows,
)

part4_files.append(path)
generated_files.append(path)


# ---------------------------------------------------------
# Table 3
# Catalog scaling
# ---------------------------------------------------------

headers = [
    "Sources",
    "Columns",
    "Docs",
    "Full chars",
    "Mean ctx",
    "Perfect",
    "Recall",
    "Precision",
    "F1",
]

tex_rows = []
csv_rows = []

for scale in ["4", "8", "16"]:
    item = catalog_results[scale]

    counts = item["catalog_counts"]
    index = item["index"]
    retrieval = item["retrieval"]
    summary = retrieval["summary"]

    tex_rows.append(
        [
            scale,
            str(counts["columns"]),
            str(index["documents"]),
            str(
                retrieval[
                    "full_catalog_characters"
                ]
            ),
            f"{summary['mean_context_characters']:.1f}",
            f"{summary['perfect_recall_queries']}/20",
            f"{summary['macro_recall']:.3f}",
            f"{summary['macro_precision']:.3f}",
            f"{summary['macro_f1']:.3f}",
        ]
    )

    csv_rows.append(
        [
            int(scale),
            counts["columns"],
            index["documents"],
            retrieval[
                "full_catalog_characters"
            ],
            summary[
                "mean_context_characters"
            ],
            summary[
                "perfect_recall_queries"
            ],
            summary["macro_recall"],
            summary["macro_precision"],
            summary["macro_f1"],
        ]
    )

path = write_latex_table(
    "catalog_scaling.tex",
    "Relation-aware retrieval under catalog scaling.",
    "tab:catalog-scaling",
    headers,
    tex_rows,
    "rrrrrrrrr",
)

part4_files.append(path)
generated_files.append(path)

path = write_csv_table(
    "catalog_scaling.csv",
    headers,
    csv_rows,
)

part4_files.append(path)
generated_files.append(path)


# ---------------------------------------------------------
# Table 4
# Spark data-volume scaling
# ---------------------------------------------------------

headers = [
    "Scale",
    "Rows",
    "Parquet MiB",
    "Scan (s)",
    "Grouping (s)",
    "Weather join (s)",
]

tex_rows = []
csv_rows = []

for scale in [
    "100k",
    "500k",
    "1m",
    "full",
]:
    item = volume_summary[scale]

    size_mib = (
        item["materialization"]["bytes"]
        / 1024
        / 1024
    )

    scan = (
        item["workloads"]
        ["scan_aggregation"]
        ["median_seconds"]
    )

    grouping = (
        item["workloads"]
        ["pickup_zone_grouping"]
        ["median_seconds"]
    )

    weather_join = (
        item["workloads"]
        ["weather_temporal_join"]
        ["median_seconds"]
    )

    display_scale = (
        "Full"
        if scale == "full"
        else scale
    )

    tex_rows.append(
        [
            display_scale,
            f"{item['rows']:,}",
            f"{size_mib:.2f}",
            f"{scan:.3f}",
            f"{grouping:.3f}",
            f"{weather_join:.3f}",
        ]
    )

    csv_rows.append(
        [
            display_scale,
            item["rows"],
            size_mib,
            scan,
            grouping,
            weather_join,
        ]
    )

path = write_latex_table(
    "data_volume_scaling.tex",
    (
        "Spark data-volume scaling using "
        "median runtime over three repetitions."
    ),
    "tab:data-volume-scaling",
    headers,
    tex_rows,
    "lrrrrr",
)

part4_files.append(path)
generated_files.append(path)

path = write_csv_table(
    "data_volume_scaling.csv",
    headers,
    csv_rows,
)

part4_files.append(path)
generated_files.append(path)


print()
print("=" * 78)
print("GENERATED TABLES — PART 4")
print("=" * 78)

for path in part4_files:
    print(
        path.relative_to(ROOT)
    )

print()
print(
    "REPORT TABLES PART 4: PASS"
)


# ---------------------------------------------------------
# Report artifact README
# ---------------------------------------------------------

readme_path = OUTPUT_ROOT / "README.md"

readme_text = """# Generated report artifacts

This directory contains figures and tables generated exclusively from the
frozen experimental benchmark JSON files in `artifacts/benchmarks/`.

The reporting script does not rerun Spark, Ollama, metadata retrieval,
Text-to-SQL generation, or any experimental benchmark.

## Figures

Each figure is generated both as PDF and PNG.

- `figures/heldout_abc_rates.pdf`
  - Held-out validation, execution, and result accuracy for configurations A/B/C.

- `figures/prompt_token_efficiency.pdf`
  - Mean prompt size for Full Catalog (A) and Relation-Aware Retrieval (B).

- `figures/controlled_latency_prompt_eval.pdf`
  - Controlled prefix-isolated prompt-evaluation latency for A and B.

- `figures/catalog_scale_recall.pdf`
  - Mean metadata retrieval recall at 4, 8, and 16 catalog sources.

- `figures/catalog_scale_context_size.pdf`
  - Full catalog size versus mean retrieved context size at 4, 8, and 16 sources.

- `figures/spark_data_volume_scaling.pdf`
  - Median Spark execution runtime at 100k, 500k, 1M, and full Yellow Taxi volume.

PDF files are intended for the final LaTeX/Overleaf report.
PNG files are intended for quick visual inspection.

## LaTeX tables

- `tables/heldout_abc.tex`
- `tables/efficiency_latency.tex`
- `tables/catalog_scaling.tex`
- `tables/data_volume_scaling.tex`

CSV copies are included for independent inspection.

The LaTeX report must include the `booktabs` package:

    \\usepackage{booktabs}

## Frozen sources

The generated material is derived from:

- `artifacts/benchmarks/heldout_abc_summary.json`
- `artifacts/benchmarks/controlled_latency_benchmark.json`
- `artifacts/benchmarks/catalog_scale_benchmark.json`
- `artifacts/benchmarks/data_volume_scaling_benchmark.json`

These experimental results were frozen before creation of the reporting
artifacts.
"""

readme_path.write_text(
    readme_text,
    encoding="utf-8",
)

generated_files.append(
    readme_path
)


print()
print("=" * 78)
print("FINAL REPORT ARTIFACT SUMMARY")
print("=" * 78)

print(
    f"Figures: "
    f"{len(list(FIGURE_DIR.glob('*.pdf')))} PDF + "
    f"{len(list(FIGURE_DIR.glob('*.png')))} PNG"
)

print(
    f"Tables:  "
    f"{len(list(TABLE_DIR.glob('*.tex')))} LaTeX + "
    f"{len(list(TABLE_DIR.glob('*.csv')))} CSV"
)

print(
    f"README:  "
    f"{readme_path.relative_to(ROOT)}"
)

print()
print("=" * 78)
print("FINAL RESULT")
print("=" * 78)

print(
    "FINAL REPORT ARTIFACT GENERATOR: PASS"
)

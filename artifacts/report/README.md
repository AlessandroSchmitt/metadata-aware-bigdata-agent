# Generated report artifacts

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

    \usepackage{booktabs}

## Frozen sources

The generated material is derived from:

- `artifacts/benchmarks/heldout_abc_summary.json`
- `artifacts/benchmarks/controlled_latency_benchmark.json`
- `artifacts/benchmarks/catalog_scale_benchmark.json`
- `artifacts/benchmarks/data_volume_scaling_benchmark.json`

These experimental results were frozen before creation of the reporting
artifacts.

# Metadata-Aware Text-to-Spark-SQL for Cross-Source Data Lake Analytics

First Project for the **Big Data** course, Università degli Studi Roma Tre, Academic Year 2025/2026.

**Student:** Alessandro Schmitt — Matricola 577421
**Course:** Big Data
**Professor:** Riccardo Torlone

## Overview

This project investigates whether dynamic, relation-aware metadata retrieval can improve Text-to-Spark-SQL generation over a heterogeneous data lake compared with providing the complete metadata catalog to a language model.

The prototype combines:

- a structured metadata catalog;
- dense and lexical metadata retrieval;
- semantic-concept and relationship expansion;
- SQL-safe metadata rendering;
- local LLM-based Spark SQL generation;
- SQL validation;
- optional one-shot repair;
- Apache Spark SQL execution.

The project addresses two Big Data dimensions:

- **Volume:** millions of NYC taxi-trip records processed with Spark;
- **Variety:** heterogeneous taxi, geographic, and weather sources with different schemas, semantics, and granularities.

## Research Question

> To what extent can dynamic, relation-aware metadata retrieval improve the correctness and efficiency of Text-to-Spark-SQL generation over a heterogeneous data lake compared with providing the complete metadata catalog to the language model?

## Architecture

```text
Natural-language analytical question
                |
                v
Structured metadata catalog + vector index
                |
                v
Dense Top-5 + lexical / alias retrieval
                |
                v
Semantic grounding + relation-aware expansion
                |
                v
SQL-safe metadata context
                |
                v
Qwen2.5-Coder 3B
Text-to-Spark-SQL generation
                |
                v
SQLGlot + Spark validation
                |
                +---- validation failure ----> optional one-shot repair
                |
                v
Apache Spark SQL execution
                |
                v
Grounded analytical result
```

The deterministic metadata and validation pipeline is implemented in Python. The language model is used for SQL generation and, when enabled, a single repair attempt.

## Data Lake

The experiments use January 2024 data from four core sources:

| Dataset | Curated rows | Columns | Role |
|---|---:|---:|---|
| NYC Yellow Taxi | 2,963,713 | 19 | Transactional mobility |
| NYC Green Taxi | 56,429 | 20 | Transactional mobility |
| NYC Taxi Zones | 265 | 4 | Geographic reference |
| NOAA hourly weather | 744 | 23 | Temporal / environmental |

Taxi pickup and drop-off location identifiers are related to `taxi_zones.LocationID`.

Taxi pickup timestamps are also related to `weather_hourly.weather_hour` after hourly temporal bucketing.

Raw and curated datasets are reproducible and intentionally excluded from Git.

## Main Experimental Results

The main evaluation uses a frozen 20-question held-out benchmark.

| Configuration | Validation | Execution | Result accuracy |
|---|---:|---:|---:|
| A — Full catalog | 75% | 75% | 70% |
| B — Relation-aware retrieval | 85% | 85% | 80% |
| C — Retrieval + validation/repair | 90% | 90% | 85% |

Additional findings:

- mean prompt size decreases from **2558.45** to **776.35 tokens** (**69.66% reduction**);
- controlled mean prompt-evaluation time decreases from **182.58 s** to **54.24 s** (**70.29% reduction**) in the CPU-only experimental environment;
- held-out metadata retrieval reaches **0.925 macro recall** on the original four-source catalog;
- catalog scaling from 4 to 16 sources keeps retrieved context close to 2,100 characters while the full catalog grows from 9,498 to 16,651 characters;
- semantically related hard negatives reduce macro recall from **0.925** to **0.825**, revealing a retrieval-quality trade-off;
- Spark workloads were evaluated from **100,000** to **2,963,713** Yellow Taxi records.

The committed benchmark artifacts are the authoritative recorded experimental results.

## Technology Stack

- Python 3.12
- Java 17
- PySpark 3.5.8
- Spark SQL
- SQLite
- Qdrant local client
- Ollama
- `embeddinggemma`
- `qwen2.5-coder:3b`
- SQLGlot
- Matplotlib

All model inference used for the final project is local; no paid model API is required.

## Repository Structure

```text
config/
data/
src/metadata_agent/
scripts/
artifacts/
docs/
report/
requirements.txt
```

## Final Report

The final project report is available here:

**[Download / view the final report](report/final_report.pdf)**

The LaTeX sources used to build the report are also fully versioned under `report/`.

## Environment Setup

A Linux environment is recommended. The final experiments were executed in GitHub Codespaces using Java 17 and a CPU-only local configuration.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Java 17 is recommended.

Ollama must be installed separately:

```bash
nohup ollama serve > /tmp/ollama.log 2>&1 &
ollama pull qwen2.5-coder:3b
ollama pull embeddinggemma
```

## Reproducing the Data Lake

```bash
python scripts/data/download_tlc.py
python scripts/data/download_weather.py
python scripts/data/build_curated_tlc.py
python scripts/data/build_curated_weather.py
```

## Building the Metadata Catalog

```bash
python scripts/catalog/build_metadata_catalog.py
python scripts/catalog/validate_metadata_catalog.py
PYTHONPATH=src python scripts/catalog/render_full_catalog.py
```

## Building the Metadata Vector Index

```bash
PYTHONPATH=src python scripts/retrieval/build_metadata_index.py
PYTHONPATH=src python scripts/retrieval/test_relation_aware_retrieval.py
```

## Benchmark Validation

```bash
PYTHONPATH=src python scripts/experiments/validate_benchmark_gold.py
PYTHONPATH=src python scripts/experiments/validate_heldout_benchmark.py
PYTHONPATH=src python scripts/experiments/evaluate_retrieval_heldout.py
```

## Main A/B/C Evaluation

```bash
PYTHONPATH=src python scripts/experiments/run_heldout_abc_benchmark.py
```

## Additional Scaling Experiments

```bash
PYTHONPATH=src python scripts/experiments/run_controlled_latency_benchmark.py
PYTHONPATH=src python scripts/experiments/run_catalog_scale_benchmark.py
PYTHONPATH=src python scripts/experiments/run_data_volume_scaling_benchmark.py
```

## Report Artifact Generation

```bash
PYTHONPATH=src python scripts/reporting/generate_report_artifacts.py
```

## Experimental Freeze Protocol

Important Git tags include:

| Tag | Purpose |
|---|---|
| `development-freeze-v1` | freeze final retrieval architecture |
| `heldout-benchmark-v1` | freeze held-out questions before evaluation |
| `heldout-results-v1` | freeze final A/B/C results |
| `latency-results-v1` | freeze controlled latency results |
| `catalog-scale-benchmark-v1` | freeze catalog-scale protocol |
| `catalog-scale-results-v1` | freeze catalog-scale results |
| `data-volume-benchmark-v1` | freeze Spark volume protocol |
| `data-volume-results-v1` | freeze Spark volume results |
| `report-artifacts-v1` | freeze generated figures/tables |
| `report-final-v1` | freeze the final project report |

The held-out results were not used to retune the frozen retrieval architecture.

## Interpretation Limits

The held-out benchmark contains 20 questions derived from ten analytical intents, the catalog-scale distractors are controlled metadata-only hard negatives, and the Spark data-volume benchmark runs locally on a two-core environment.

The results therefore characterize this prototype and experimental setting; they are not universal performance claims for Spark, Qwen, Qdrant, or metadata retrieval systems in general.

## License

This repository is released under the MIT License.

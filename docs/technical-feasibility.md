# Technical Feasibility

> **Note:** this document records the preliminary feasibility configuration. The final frozen Text-to-Spark-SQL evaluation uses an LLM context size of **4096 tokens**. The 2048-token value below is retained because it describes the earlier feasibility test.

The project was initially validated through a local technical feasibility study.

## Environment

- Execution environment: GitHub Codespaces
- CPU: 2 vCPU
- RAM: 7.8 GiB
- GPU: none
- Python: 3.12.1
- Java: Eclipse Temurin 17.0.20
- PySpark: 3.5.8
- Ollama: 0.32.11
- LLM: qwen2.5-coder:3b
- LLM context: 2048 tokens
- LLM temperature: 0

## Spark sanity test

A local Spark application using `local[2]` and a 1 GiB driver heap successfully:

- processed 100,000 rows;
- executed Spark SQL aggregation;
- performed shuffle/grouping;
- wrote and reloaded Parquet data.

Result: PASS.

## Text-to-SQL feasibility

Five preliminary Text-to-Spark-SQL questions were tested, including:

- single-table aggregation;
- Taxi/Zone joins;
- Yellow/Green integration;
- Taxi/Weather temporal joins;
- Taxi/Zone/Weather queries.

The tests showed that the local LLM is capable of producing useful Spark SQL while also exhibiting semantic errors that motivate the validation and experimental components of the project.

## End-to-end vertical slice

A complete preliminary pipeline was tested:

Natural-language question -> local LLM -> generated Spark SQL -> Spark execution -> gold-result comparison.

The query required a join between synthetic Yellow Taxi and Taxi Zone datasets.

Generated result:

- Midtown: 10
- JFK Airport: 7
- Astoria: 4

Gold result:

- Midtown: 10
- JFK Airport: 7
- Astoria: 4

Result correctness: PASS.

During simultaneous Spark and LLM execution, approximately 3.28 GiB of system memory remained available.

## Conclusion

The core architecture is technically feasible in the selected zero-cost local development environment.

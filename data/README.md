# Data directory

This directory contains the local data lake used by the project.

## Structure

- `raw/`: source datasets in their original downloaded representation.
- `curated/`: cleaned and normalized datasets used by Spark SQL.
- `catalog/`: metadata repository and generated catalog artifacts.

The actual datasets are intentionally excluded from Git because of their size.

Expected core sources:

- NYC Yellow Taxi Trip Records
- NYC Green Taxi Trip Records
- NYC Taxi Zone Lookup
- Weather observations

Dataset acquisition is performed through reproducible scripts under `scripts/data/`.

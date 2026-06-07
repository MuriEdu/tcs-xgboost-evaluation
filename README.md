# Core TSC + XGBoost Forecasting Pipeline

A modular big-data pipeline for customer default/delinquency forecasting using a hybrid
Time Series Classification (TSC) and XGBoost stack.

This repository is designed for production-ready financial risk modeling with:
- a strict generic TSC interface,
- Polars-based feature engineering for large tabular datasets,
- DuckDB-powered ingestion and dataset access for Excel, CSV, and Parquet,
- stratified 10-fold validation targeting imbalanced default data,
- industry-grade performance tracking with AUC-ROC, F1, G-Mean, and confusion matrices.

## Repository Structure

- `core_tsc_xgboost_pipeline.py`
  - Main pipeline orchestration script.
  - Loads data, splits features, trains XGBoost, and validates with stratified CV.
- `tsc_models/`
  - `__init__.py` exports the generic TSC interface and default pass-through model.
  - `base.py` contains `BaseTimeSeriesClassifier`.
  - `null_classifier.py` contains `NullTimeSeriesClassifier`.
- `dataset_aumentado.xlsx`
  - Example dataset file used by the pipeline.

## Architecture

1. Data ingestion via `load_dataset`
2. Target normalization and binary encoding of `inadimplent`
3. Static / temporal feature split via `split_static_temporal_features`
4. TSC meta-feature generation through the generic `BaseTimeSeriesClassifier`
5. XGBoost training with imbalance-aware `scale_pos_weight`
6. Stratified 10-fold evaluation using rigorous metrics

## Installation

This project requires Python 3.10+ and the following packages:

- `polars`
- `duckdb`
- `numpy`
- `scikit-learn`
- `xgboost`

Install dependencies with:

```bash
pip install polars duckdb numpy scikit-learn xgboost
```

## Usage

Run the pipeline from the repository root:

```bash
python core_tsc_xgboost_pipeline.py
```

The script will:
- read `dataset_aumentado.xlsx` through DuckDB and load it into Polars
- split static and temporal features
- build meta-features using `BaseTimeSeriesClassifier` implementations
- train an XGBoost classifier in each fold
- log per-fold and aggregate metrics

## Data Requirements

The pipeline expects a tabular dataset with:

- a binary target column named `inadimplent`
- static features
- temporal or history-like features

The code attempts to detect temporal columns using naming patterns such as:
- `t1`, `time_1`, `hist_1`, `lag_1`, `month_01`, `week_01`, etc.

If no temporal columns are detected, all numeric features are treated as temporal candidates
when there are at least three numeric columns.

## Data Ingestion and Big Data Writing

The pipeline reads data through DuckDB and converts it into Polars for fast, columnar processing.
- DuckDB handles file formats like `.xlsx`, `.csv`, and `.parquet` efficiently.
- Polars keeps transformations vectorized and parallelized, minimizing Python overhead.

For large datasets, the preferred flow is:
1. store raw data in **Parquet** or **CSV** on disk,
2. read it with DuckDB into Polars,
3. perform feature engineering and TSC meta-feature generation in Polars,
4. write intermediate or final tables back to disk in **Parquet** for reuse.

### Writing data in Big Data mode

For production or larger workloads, use Polars write methods to persist data:

```python
X_static.write_parquet("static_features.parquet")
X_temporal.write_parquet("temporal_features.parquet")
```

If you need to materialize a subset or preprocessed output from DuckDB, use DuckDB SQL and export to Parquet:

```python
import duckdb
con = duckdb.connect()
con.execute("COPY (SELECT * FROM my_table WHERE split='train') TO 'train.parquet' (FORMAT PARQUET)")
```

This approach is ideal for Big Data because it avoids repeated reloading from slow Excel files and enables efficient, compressed columnar storage.

## Validation and Metrics

The pipeline implements a gold-standard evaluation scheme for imbalanced risk data:

- Stratified 10-fold cross-validation
- AUC-ROC as the primary discrimination metric
- F1 score for balance between precision and recall on the minority class
- G-Mean calculated as `sqrt(Sensitivity * Specificity)`
- Full confusion matrix reporting `TP`, `FP`, `TN`, `FN`

### Why these metrics?

- `AUC-ROC` evaluates ranking performance across thresholds.
- `F1` captures the balance of precision and recall in the default class.
- `G-Mean` penalizes models that perform well only on the majority class, which is critical for severely imbalanced default datasets.

## Extending the TSC Package

This repository is intentionally modular so you can add new TSC implementations without changing
pipeline orchestration.

### Add a new classifier

1. Create a new file under `tsc_models/`, for example:
   - `tsc_models/canonical_interval_forest.py`
   - `tsc_models/shapelet_transform.py`
2. Implement a class that inherits from `BaseTimeSeriesClassifier`.
3. Provide `fit`, `transform`, and optionally `fit_transform` behavior.
4. Import and instantiate the new classifier in `core_tsc_xgboost_pipeline.py`.

Example:

```python
from tsc_models.canonical_interval_forest import CanonicalIntervalForest

# Replace the null pass-through TSC
tsc_model = CanonicalIntervalForest()
```

## Baseline Comparison Strategy

The project includes structural comments showing how to compare a stacked TSC+XGBoost pipeline
against baseline models such as:

- XGBoost on static-only features
- XGBoost on static + raw temporal features
- XGBoost on meta-features from a real TSC implementation

This baseline comparison is critical for quantifying the incremental value of the TSC layer.

## Notes

- The pipeline is intentionally generic so that `NullTimeSeriesClassifier` acts as a drop-in stub.
- For production deployment, add logging, configuration management, and model persistence.
- Ensure the dataset target column is consistently encoded as binary `0/1` or normalized by `normalize_target`.

## License

This repository is provided for academic and prototyping purposes.

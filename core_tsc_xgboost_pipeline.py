"""Hybrid Time Series Classification + XGBoost core forecasting pipeline.

This big-data architecture uses Polars for in-memory feature engineering and
DuckDB for fast dataset ingestion and query execution.

Dependencies:
- polars
- duckdb
- numpy
- scikit-learn
- xgboost
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import polars as pl
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from tsc_models import BaseTimeSeriesClassifier, NullTimeSeriesClassifier
from tsc_models import CanonicalIntervalForestClassifier, LSTMTimeSeriesClassifier, RocketTimeSeriesClassifier, ShapeletTimeSeriesClassifier


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_dataset(path: str, target_column: str = "inadimplent") -> pl.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    extension = os.path.splitext(path)[1].lower()
    if extension in {".xls", ".xlsx"}:
        try:
            with duckdb.connect() as connection:
                try:
                    connection.load_extension("excel")
                except Exception:
                    connection.install_extension("excel")
                    connection.load_extension("excel")
                read_fn = "read_xlsx" if extension == ".xlsx" else "read_xls"
                arrow_table = connection.execute(
                    f"SELECT * FROM {read_fn}(?)", [path]
                ).arrow()
            df = pl.from_arrow(arrow_table)
        except Exception as exc:
            raise RuntimeError(
                "DuckDB failed to load the Excel dataset. Ensure duckdb is installed with Excel support."
            ) from exc
    elif extension == ".parquet":
        df = pl.read_parquet(path)
    elif extension == ".csv":
        df = pl.read_csv(path)
    else:
        raise ValueError("Unsupported file format. Use .xlsx, .xls, .csv or .parquet.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")

    logger.info("Loaded dataset with %d rows and %d columns", df.height, len(df.columns))
    return df


def normalize_target(series: pl.Series) -> pl.Series:
    mapping = {
        "inadimplent": 1,
        "default": 1,
        "yes": 1,
        "y": 1,
        "true": 1,
        "t": 1,
        "positive": 1,
        "non": 0,
        "no": 0,
        "n": 0,
        "false": 0,
        "f": 0,
        "negative": 0,
    }
    if series.dtype in {pl.Float32, pl.Float64}:
        non_binary = series.filter(~series.is_in([0.0, 1.0]) & series.is_not_null())
        if non_binary.len() > 0:
            raise ValueError(
                f"Target column contains float values that are not 0.0 or 1.0 "
                f"(e.g. {non_binary[0]}). Cannot safely convert to binary."
            )
        try:
            target = series.cast(pl.Int64)
        except Exception as exc:
            raise ValueError("Target column could not be converted into binary values 0/1.") from exc
    else:
        normalized = series.cast(pl.Utf8).str.to_lowercase().str.strip_chars()
        mapped = normalized.replace(mapping, default=None)
        filled = mapped.fill_null(normalized)
        try:
            target = filled.cast(pl.Float64).cast(pl.Int64)
        except Exception as exc:
            raise ValueError("Target column could not be converted into binary values 0/1.") from exc

    if not target.is_in([0, 1]).all():
        raise ValueError("Target column could not be converted into binary values 0/1.")

    return target


def detect_temporal_columns(columns: List[str]) -> List[str]:
    time_pattern = re.compile(
        r"(?:^|_|\.|\-)(?:t(?:ime)?|ts|hist|month|m|wk|week|day|period|lag|seq|step)[_\-.]?\d+$",
        flags=re.IGNORECASE,
    )
    suffix_digits = re.compile(r".*[_\-.]?\d+$")
    candidate_columns = [col for col in columns if time_pattern.search(col)]

    if len(candidate_columns) >= 3:
        return candidate_columns

    suffix_candidates = [col for col in columns if suffix_digits.match(col)]
    if len(suffix_candidates) >= 3:
        return suffix_candidates

    return candidate_columns


def split_static_temporal_features(
    df: pl.DataFrame,
    target_column: str = "inadimplent",
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    y = normalize_target(df[target_column])
    features = df.drop(target_column)

    if features.width == 0:
        raise ValueError("Dataset contains no feature columns after dropping the target.")

    temporal_cols = detect_temporal_columns(features.columns)
    if not temporal_cols:
        numeric_cols = [col for col, dtype in zip(features.columns, features.dtypes) if dtype in {pl.Int64, pl.Float64, pl.Int32, pl.Float32}]
        if len(numeric_cols) >= 3:
            temporal_cols = numeric_cols
            logger.warning(
                "No explicit temporal feature names detected; using all numeric columns as time-series candidates."
            )
        else:
            logger.warning(
                "No temporal features were detected. The pipeline will proceed with static features only."
            )

    static_cols = [col for col in features.columns if col not in temporal_cols]
    X_static = features.select(static_cols)
    X_temporal = features.select(temporal_cols)

    logger.info(
        "Split features into %d static and %d temporal columns.",
        X_static.width,
        X_temporal.width,
    )
    return X_static, X_temporal, y


def build_meta_features(
    tsc_model: BaseTimeSeriesClassifier,
    X_temporal: pl.DataFrame,
    y: Optional[pl.Series] = None,
) -> pl.DataFrame:
    if y is None:
        return tsc_model.transform(X_temporal)
    return tsc_model.fit_transform(X_temporal, y)


def train_xgboost_classifier(X: pl.DataFrame, y: pl.Series) -> XGBClassifier:
    y_np = y.to_numpy().astype(int)
    class_counts = np.bincount(y_np)
    neg_count, pos_count = int(class_counts[0] if len(class_counts) > 0 else 0), int(class_counts[1] if len(class_counts) > 1 else 0)
    scale_pos_weight = neg_count / max(pos_count, 1)

    model = XGBClassifier(
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X.to_numpy(), y_np)
    return model


def compute_gmean(tp: int, fn: int, tn: int, fp: int) -> float:
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(np.sqrt(sensitivity * specificity))


def evaluate_predictions(y_true: pl.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true_np = y_true.to_numpy().astype(int)
    auc = roc_auc_score(y_true_np, y_prob)
    f1 = f1_score(y_true_np, y_pred, pos_label=1)
    tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred).ravel()
    gmean = compute_gmean(tp, fn, tn, fp)

    return {
        "auc_roc": auc,
        "f1_score": f1,
        "g_mean": gmean,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def validate_pipeline(
    X_static: pl.DataFrame,
    X_temporal: pl.DataFrame,
    y: pl.Series,
    tsc_model: BaseTimeSeriesClassifier,
    n_splits: int = 10,
) -> Dict[str, Dict[str, float]]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics: List[Dict[str, float]] = []
    y_np = y.to_numpy().astype(int)
    sample_index = np.arange(len(y_np), dtype=int)

    # Fit-once optimization: if the TSC transform is unsupervised and
    # data-deterministic, the same row maps to the same meta-features in every
    # fold. Compute them ONCE on all rows and slice per fold instead of
    # recomputing the transform ~n_splits times. No leakage — the transform
    # ignores labels and the train/test partition.
    global_meta: Optional[pl.DataFrame] = None
    if getattr(tsc_model, "supports_global_transform", False):
        global_meta = tsc_model.fit_transform(X_temporal, y)
        logger.info("Precomputed meta-features once for all rows (global transform).")

    # Indexed-fit optimization (e.g. CIF): the per-(row, interval) features are
    # fold-independent, only the trees are supervised. Build the feature cache
    # ONCE over all rows, then retrain trees per fold by row index — turns
    # ~n_splits Catch22 passes into a single one.
    indexed_fit = global_meta is None and getattr(tsc_model, "supports_indexed_fit", False)
    if indexed_fit:
        tsc_model.prepare_cache(X_temporal)
        logger.info("Built fold-independent feature cache once for all rows (indexed fit).")

    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(sample_index, y_np), start=1):
        X_static_train = X_static.gather(train_idx)
        X_static_test = X_static.gather(test_idx)
        y_train = y.gather(train_idx)
        y_test = y.gather(test_idx)

        if global_meta is not None:
            X_meta_train = global_meta.gather(train_idx)
            X_meta_test = global_meta.gather(test_idx)
        elif indexed_fit:
            tsc_model.fit_indexed(train_idx, y_train)
            X_meta_train = tsc_model.transform_indexed(train_idx)
            X_meta_test = tsc_model.transform_indexed(test_idx)
        else:
            X_temporal_train = X_temporal.gather(train_idx)
            X_temporal_test = X_temporal.gather(test_idx)
            X_meta_train = build_meta_features(tsc_model, X_temporal_train, y_train)
            X_meta_test = build_meta_features(tsc_model, X_temporal_test)

        X_train = pl.concat([X_static_train, X_meta_train], how="horizontal")
        X_test = pl.concat([X_static_test, X_meta_test], how="horizontal")

        model = train_xgboost_classifier(X_train, y_train)
        y_pred = model.predict(X_test.to_numpy())
        y_prob = model.predict_proba(X_test.to_numpy())[:, 1]

        fold_result = evaluate_predictions(y_test, y_pred, y_prob)
        fold_result["fold"] = fold_index
        fold_metrics.append(fold_result)

        logger.info(
            "Fold %d metrics: AUC=%.4f, F1=%.4f, G-Mean=%.4f, TP=%d, FP=%d, TN=%d, FN=%d",
            fold_index,
            fold_result["auc_roc"],
            fold_result["f1_score"],
            fold_result["g_mean"],
            fold_result["tp"],
            fold_result["fp"],
            fold_result["tn"],
            fold_result["fn"],
        )

    metrics_df = pl.DataFrame(fold_metrics)
    aggregate: Dict[str, Dict[str, float]] = {}
    for column in [c for c in metrics_df.columns if c != "fold"]:
        series = metrics_df[column]
        aggregate[column] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
        }

    return {
        "fold_metrics": fold_metrics,
        "aggregate": aggregate,
    }


def _print_comparison_table(all_results: Dict[str, Dict]) -> None:
    metrics = ["auc_roc", "f1_score", "g_mean"]
    header = f"{'Model':<10}  {'AUC-ROC':>10}  {'F1':>10}  {'G-Mean':>10}"
    separator = "-" * len(header)
    print("\n" + separator)
    print(header)
    print(separator)
    for model_name, results in all_results.items():
        agg = results["aggregate"]
        print(
            f"{model_name:<10}  "
            f"{agg['auc_roc']['mean']:>8.4f}±{agg['auc_roc']['std']:.4f}  "
            f"{agg['f1_score']['mean']:>8.4f}±{agg['f1_score']['std']:.4f}  "
            f"{agg['g_mean']['mean']:>8.4f}±{agg['g_mean']['std']:.4f}"
        )
    print(separator + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="TSC + XGBoost forecasting pipeline.")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run all TSC models and print a comparison table.",
    )
    args = parser.parse_args()

    dataset_path = "dataset_aumentado.xlsx"
    df = load_dataset(dataset_path, target_column="default payment next month")
    X_static, X_temporal, y = split_static_temporal_features(df, target_column="default payment next month")

    if args.compare:
        models: Dict[str, BaseTimeSeriesClassifier] = {
            "Null": NullTimeSeriesClassifier(),
            "CIF": CanonicalIntervalForestClassifier(n_estimators=20, random_state=42),
            "ROCKET": RocketTimeSeriesClassifier(num_kernels=2_000, random_state=42),
            "LSTM": LSTMTimeSeriesClassifier(hidden_size=32, epochs=30, random_state=42),
            "Shapelets": ShapeletTimeSeriesClassifier(num_shapelets=100, random_state=42),
        }
        all_results: Dict[str, Dict] = {}
        for name, model in models.items():
            logger.info("Running pipeline with %s...", name)
            all_results[name] = validate_pipeline(X_static, X_temporal, y, model, n_splits=10)
        _print_comparison_table(all_results)
    else:
        tsc_model = CanonicalIntervalForestClassifier(n_estimators=200, random_state=42)
        logger.info("Starting stratified 10-fold validation with the stacked pipeline.")
        results = validate_pipeline(X_static, X_temporal, y, tsc_model, n_splits=10)
        logger.info("Aggregate validation results:")
        for metric, stats in results["aggregate"].items():
            logger.info("  %s: mean=%.4f, std=%.4f", metric, stats["mean"], stats["std"])


if __name__ == "__main__":
    main()

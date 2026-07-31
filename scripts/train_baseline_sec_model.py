from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ML_DATASET_PATH = "data/processed/modeling/sec_ml_dataset.parquet"

OUTPUT_DIR = Path("data/processed/models/baseline_sec")
PREDICTIONS_PATH = OUTPUT_DIR / "baseline_predictions.parquet"
METRICS_PATH = OUTPUT_DIR / "baseline_metrics.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "random_forest_feature_importance.csv"
LOGISTIC_MODEL_PATH = OUTPUT_DIR / "logistic_regression_model.joblib"
RANDOM_FOREST_MODEL_PATH = OUTPUT_DIR / "random_forest_model.joblib"

OUTPUT_DUCKDB_PATH = "data/processed/eventsignal.duckdb"

TARGET_COL = "label_up_abnormal_3d"
RETURN_COL = "fwd_abnormal_ret_3d"

FEATURE_COLS = [
    # Document size / complexity
    "document_length_chars",
    "word_count",
    "unique_word_count",
    "unique_word_ratio",

    # Simple financial tone counts
    "positive_count",
    "negative_count",
    "uncertainty_count",
    "litigious_count",
    "modal_strong_count",
    "modal_weak_count",
    "ai_semiconductor_count",
    "macro_count",

    # Normalized text features
    "positive_per_1000",
    "negative_per_1000",
    "uncertainty_per_1000",
    "litigious_per_1000",
    "ai_semiconductor_per_1000",
    "macro_per_1000",

    # Derived ratios
    "sentiment_balance",
    "uncertainty_ratio",

    # Phrase features
    "risk_phrase_count",
    "going_concern_count",
    "material_weakness_count",
    "supply_chain_count",
    "artificial_intelligence_count",

    # Form indicators
    "is_10k",
    "is_10q",

    # Price context known before label period
    "ret_1d",
    "ret_5d",
    "ret_20d",
]

def load_ml_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing ML dataset: {path}")

    ml = pd.read_parquet(path)

    required_cols = {"event_id", "ticker", "form", "release_timestamp_utc", TARGET_COL, RETURN_COL}
    missing = required_cols - set(ml.columns)

    if missing:
        raise ValueError(f"ML dataset missing required columns: {missing}")

    ml["release_timestamp_utc"] = pd.to_datetime(
        ml["release_timestamp_utc"],
        errors="coerce",
        utc=True,
    )

    return ml

def clean_for_modeling(ml: pd.DataFrame) -> pd.DataFrame:
    df = ml.copy()

    # Keep only rows where the target exists.
    df = df.dropna(subset=[TARGET_COL, RETURN_COL, "release_timestamp_utc"])

    # Ensure binary integer target.
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    available_features = [col for col in FEATURE_COLS if col in df.columns]
    missing_features = sorted(set(FEATURE_COLS) - set(available_features))

    if missing_features:
        print(f"Warning: missing feature columns ignored: {missing_features}")

    if not available_features:
        raise ValueError("No usable feature columns found.")

    for col in available_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("release_timestamp_utc").reset_index(drop=True)

    return df

def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    No random split.

    Train:      before 2023
    Validation: 2023
    Test:       2024 and later

    This prevents lookahead from future events into past model training.
    """
    train_end = pd.Timestamp("2023-01-01", tz="UTC")
    val_end = pd.Timestamp("2024-01-01", tz="UTC")

    train = df[df["release_timestamp_utc"] < train_end].copy()
    val = df[
        (df["release_timestamp_utc"] >= train_end)
        & (df["release_timestamp_utc"] < val_end)
    ].copy()
    test = df[df["release_timestamp_utc"] >= val_end].copy()

    if train.empty:
        raise ValueError("Train set is empty. Check release_timestamp_utc.")

    if val.empty:
        print("Warning: validation set is empty.")

    if test.empty:
        print("Warning: test set is empty.")

    print("Split sizes:")
    print(f"Train:      {len(train)}")
    print(f"Validation: {len(val)}")
    print(f"Test:       {len(test)}")

    return train, val, test

def make_logistic_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

###type of ML

def make_random_forest_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

def evaluate_classifier(
    model_name: str,
    model: Pipeline,
    split_name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict, pd.DataFrame]:
    if df.empty:
        metrics = {
            "model": model_name,
            "split": split_name,
            "n": 0,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "auc": np.nan,
            "avg_forward_abnormal_return": np.nan,
            "top_decile_avg_forward_abnormal_return": np.nan,
            "bottom_decile_avg_forward_abnormal_return": np.nan,
            "long_short_decile_spread": np.nan,
        }

        return metrics, pd.DataFrame()

    X = df[feature_cols]
    y = df[TARGET_COL].astype(int)

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    accuracy = accuracy_score(y, pred)
    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)

    if y.nunique() > 1:
        auc = roc_auc_score(y, proba)
    else:
        auc = np.nan

    pred_df = df[
        [
            "event_id",
            "ticker",
            "form",
            "release_timestamp_utc",
            TARGET_COL,
            RETURN_COL,
        ]
    ].copy()

    pred_df["model"] = model_name
    pred_df["split"] = split_name
    pred_df["predicted_probability_up_3d"] = proba
    pred_df["predicted_label_up_3d"] = pred

    avg_return = pred_df[RETURN_COL].mean()

    # Decile analysis: if the model is useful, high-probability events should do better.
    pred_df["prediction_rank"] = pred_df["predicted_probability_up_3d"].rank(method="first")

    try:
        pred_df["prediction_decile"] = pd.qcut(
            pred_df["prediction_rank"],
            q=10,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        pred_df["prediction_decile"] = np.nan

    if pred_df["prediction_decile"].notna().any():
        top_decile = pred_df[pred_df["prediction_decile"] == pred_df["prediction_decile"].max()]
        bottom_decile = pred_df[pred_df["prediction_decile"] == pred_df["prediction_decile"].min()]

        top_decile_return = top_decile[RETURN_COL].mean()
        bottom_decile_return = bottom_decile[RETURN_COL].mean()
        long_short_spread = top_decile_return - bottom_decile_return
    else:
        top_decile_return = np.nan
        bottom_decile_return = np.nan
        long_short_spread = np.nan

    metrics = {
        "model": model_name,
        "split": split_name,
        "n": len(df),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "avg_forward_abnormal_return": avg_return,
        "top_decile_avg_forward_abnormal_return": top_decile_return,
        "bottom_decile_avg_forward_abnormal_return": bottom_decile_return,
        "long_short_decile_spread": long_short_spread,
    }

    cm = confusion_matrix(y, pred)
    print()
    print(f"{model_name} | {split_name}")
    print(metrics)
    print("Confusion matrix:")
    print(cm)

    return metrics, pred_df

def extract_random_forest_importance(
    rf_model: Pipeline,
    feature_cols: list[str],
) -> pd.DataFrame:
    rf = rf_model.named_steps["model"]

    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": rf.feature_importances_,
        }
    )

    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)

    return importance


def save_outputs(
    logistic_model: Pipeline,
    random_forest_model: Pipeline,
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    rf_importance: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(logistic_model, LOGISTIC_MODEL_PATH)
    joblib.dump(random_forest_model, RANDOM_FOREST_MODEL_PATH)

    metrics_df.to_csv(METRICS_PATH, index=False)
    predictions_df.to_parquet(PREDICTIONS_PATH, index=False)
    rf_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    con = duckdb.connect(OUTPUT_DUCKDB_PATH)

    con.register("baseline_metrics_df", metrics_df)
    con.register("baseline_predictions_df", predictions_df)
    con.register("rf_importance_df", rf_importance)

    con.execute(
        "CREATE OR REPLACE TABLE baseline_sec_metrics AS "
        "SELECT * FROM baseline_metrics_df"
    )

    con.execute(
        "CREATE OR REPLACE TABLE baseline_sec_predictions AS "
        "SELECT * FROM baseline_predictions_df"
    )

    con.execute(
        "CREATE OR REPLACE TABLE baseline_sec_feature_importance AS "
        "SELECT * FROM rf_importance_df"
    )

    con.close()

    print()
    print(f"Saved logistic model:       {LOGISTIC_MODEL_PATH}")
    print(f"Saved random forest model:  {RANDOM_FOREST_MODEL_PATH}")
    print(f"Saved metrics:              {METRICS_PATH}")
    print(f"Saved predictions:          {PREDICTIONS_PATH}")
    print(f"Saved feature importance:   {FEATURE_IMPORTANCE_PATH}")
    print("Saved DuckDB tables: baseline_sec_metrics, baseline_sec_predictions, baseline_sec_feature_importance")


def main() -> None:
    ml = load_ml_dataset(ML_DATASET_PATH)
    ml = clean_for_modeling(ml)

    feature_cols = [col for col in FEATURE_COLS if col in ml.columns]

    print(f"Using {len(feature_cols)} features:")
    for col in feature_cols:
        print(f"  - {col}")

    print()
    print("Target distribution:")
    print(ml[TARGET_COL].value_counts(normalize=True).rename("share"))
    print(ml[TARGET_COL].value_counts().rename("count"))

    train, val, test = time_split(ml)

    X_train = train[feature_cols]
    y_train = train[TARGET_COL].astype(int)

    logistic_model = make_logistic_model()
    random_forest_model = make_random_forest_model()

    print()
    print("Training logistic regression...")
    logistic_model.fit(X_train, y_train)

    print("Training random forest...")
    random_forest_model.fit(X_train, y_train)

    metrics = []
    predictions = []

    for model_name, model in [
        ("logistic_regression", logistic_model),
        ("random_forest", random_forest_model),
    ]:
        for split_name, split_df in [
            ("train", train),
            ("validation", val),
            ("test", test),
        ]:
            metric, pred_df = evaluate_classifier(
                model_name=model_name,
                model=model,
                split_name=split_name,
                df=split_df,
                feature_cols=feature_cols,
            )

            metrics.append(metric)

            if not pred_df.empty:
                predictions.append(pred_df)

    metrics_df = pd.DataFrame(metrics)

    if predictions:
        predictions_df = pd.concat(predictions, ignore_index=True)
    else:
        predictions_df = pd.DataFrame()

    rf_importance = extract_random_forest_importance(
        random_forest_model,
        feature_cols=feature_cols,
    )

    save_outputs(
        logistic_model=logistic_model,
        random_forest_model=random_forest_model,
        metrics_df=metrics_df,
        predictions_df=predictions_df,
        rf_importance=rf_importance,
    )

    print()
    print("Metrics summary:")
    print(metrics_df)

    print()
    print("Top random forest features:")
    print(rf_importance.head(20))


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import re

import duckdb
import pandas as pd

SEC_TEXT_PATH = "data/processed/events/sec_event_text.parquet"
LABELED_EVENTS_PATH = "data/processed/labels/labeled_sec_events.parquet"

OUTPUT_FEATURES_PARQUET_PATH = "data/processed/features/sec_text_features.parquet"
OUTPUT_FEATURES_CSV_PATH = "data/processed/features/sec_text_features.csv"

OUTPUT_ML_PARQUET_PATH = "data/processed/modeling/sec_ml_dataset.parquet"
OUTPUT_ML_CSV_PATH = "data/processed/modeling/sec_ml_dataset.csv"

OUTPUT_DUCKDB_PATH = "data/processed/eventsignal.duckdb"

POSITIVE_WORDS = {
    "achieve", "achieved", "achieves", "achievement",
    "benefit", "beneficial", "best", "better",
    "efficient", "efficiency", "enhance", "enhanced",
    "expand", "expanded", "expansion",
    "favorable", "gain", "gains", "growth", "improve", "improved",
    "improvement", "increase", "increased", "increases",
    "leading", "opportunity", "positive", "profit", "profitable",
    "record", "strong", "strength", "successful", "successfully",
    "surpass", "surpassed", "upside",
}

NEGATIVE_WORDS = {
    "adverse", "bad", "challenge", "challenging", "decline", "declined",
    "decrease", "decreased", "deficit", "delay", "delayed",
    "deteriorate", "deteriorated", "difficult", "difficulty",
    "disruption", "downturn", "fall", "fell", "impairment",
    "loss", "losses", "negative", "penalty", "poor", "pressure",
    "problem", "problems", "recession", "reduce", "reduced",
    "reduction", "risk", "risks", "slow", "slowed", "slower",
    "weak", "weaken", "weakened", "weakness",
}

UNCERTAINTY_WORDS = {
    "approximately", "believe", "could", "depend", "dependent",
    "estimate", "estimated", "estimates", "expect", "expected",
    "forecast", "may", "might", "possible", "possibly",
    "potential", "potentially", "risk", "risks", "uncertain",
    "uncertainties", "uncertainty", "unknown", "vary", "volatility",
}

LITIGIOUS_WORDS = {
    "action", "actions", "allege", "alleged", "allegedly",
    "claim", "claims", "complaint", "court", "damages",
    "defendant", "investigation", "legal", "liability",
    "litigation", "plaintiff", "proceeding", "proceedings",
    "regulation", "regulatory", "settlement", "sue", "sued",
}

MODAL_STRONG_WORDS = {
    "always", "clearly", "definitely", "must", "never", "undoubtedly",
    "will",
}

MODAL_WEAK_WORDS = {
    "could", "depending", "may", "might", "possibly", "potentially",
    "should", "would",
}

AI_SEMICONDUCTOR_WORDS = {
    "ai", "artificial", "intelligence", "accelerator", "chip",
    "chips", "cloud", "compute", "datacenter", "data", "center",
    "gpu", "semiconductor", "semiconductors", "server", "servers",
    "inference", "training", "model", "models",
}

MACRO_WORDS = {
    "inflation", "interest", "rates", "rate", "currency", "foreign",
    "exchange", "tariff", "tariffs", "supply", "chain", "labor",
    "employment", "recession", "credit", "liquidity",
}

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    text_path = Path(SEC_TEXT_PATH)
    labeled_path = Path(LABELED_EVENTS_PATH)

    if not text_path.exists():
        raise FileNotFoundError(f"Missing SEC text file: {text_path}")

    if not labeled_path.exists():
        raise FileNotFoundError(f"Missing labeled events file: {labeled_path}")

    text_df = pd.read_parquet(text_path)
    labeled = pd.read_parquet(labeled_path)

    return text_df, labeled

def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []

    return re.findall(r"[a-zA-Z]+", text.lower())

def count_words(tokens: list[str], dictionary: set[str]) -> int:
    return sum(1 for token in tokens if token in dictionary)

def count_regex(text: str, pattern: str) -> int:
    if not isinstance(text, str):
        return 0

    return len(re.findall(pattern, text, flags=re.IGNORECASE))

def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator

def extract_features_for_row(row: pd.Series) -> dict:
    text = row.get("document_text", "")

    if not isinstance(text, str):
        text = ""

    tokens = tokenize(text)

    word_count = len(tokens)
    unique_word_count = len(set(tokens))
    char_count = len(text)

    positive_count = count_words(tokens, POSITIVE_WORDS)
    negative_count = count_words(tokens, NEGATIVE_WORDS)
    uncertainty_count = count_words(tokens, UNCERTAINTY_WORDS)
    litigious_count = count_words(tokens, LITIGIOUS_WORDS)
    modal_strong_count = count_words(tokens, MODAL_STRONG_WORDS)
    modal_weak_count = count_words(tokens, MODAL_WEAK_WORDS)
    ai_semiconductor_count = count_words(tokens, AI_SEMICONDUCTOR_WORDS)
    macro_count = count_words(tokens, MACRO_WORDS)

    risk_phrase_count = count_regex(text, r"\brisk factors?\b")
    going_concern_count = count_regex(text, r"\bgoing concern\b")
    material_weakness_count = count_regex(text, r"\bmaterial weakness(?:es)?\b")
    supply_chain_count = count_regex(text, r"\bsupply chain\b")
    artificial_intelligence_count = count_regex(text, r"\bartificial intelligence\b")

    positive_per_1000 = safe_divide(positive_count * 1000.0, word_count)
    negative_per_1000 = safe_divide(negative_count * 1000.0, word_count)
    uncertainty_per_1000 = safe_divide(uncertainty_count * 1000.0, word_count)
    litigious_per_1000 = safe_divide(litigious_count * 1000.0, word_count)
    ai_semiconductor_per_1000 = safe_divide(ai_semiconductor_count * 1000.0, word_count)
    macro_per_1000 = safe_divide(macro_count * 1000.0, word_count)

    sentiment_balance = safe_divide(
        positive_count - negative_count,
        positive_count + negative_count,
    )

    uncertainty_ratio = safe_divide(uncertainty_count, word_count)
    unique_word_ratio = safe_divide(unique_word_count, word_count)

    return {
        "event_id": row["event_id"],
        "ticker": row["ticker"],
        "form": row["form"],
        "download_status": row["download_status"],

        "document_length_chars": char_count,
        "word_count": word_count,
        "unique_word_count": unique_word_count,
        "unique_word_ratio": unique_word_ratio,

        "positive_count": positive_count,
        "negative_count": negative_count,
        "uncertainty_count": uncertainty_count,
        "litigious_count": litigious_count,
        "modal_strong_count": modal_strong_count,
        "modal_weak_count": modal_weak_count,
        "ai_semiconductor_count": ai_semiconductor_count,
        "macro_count": macro_count,

        "positive_per_1000": positive_per_1000,
        "negative_per_1000": negative_per_1000,
        "uncertainty_per_1000": uncertainty_per_1000,
        "litigious_per_1000": litigious_per_1000,
        "ai_semiconductor_per_1000": ai_semiconductor_per_1000,
        "macro_per_1000": macro_per_1000,

        "sentiment_balance": sentiment_balance,
        "uncertainty_ratio": uncertainty_ratio,

        "risk_phrase_count": risk_phrase_count,
        "going_concern_count": going_concern_count,
        "material_weakness_count": material_weakness_count,
        "supply_chain_count": supply_chain_count,
        "artificial_intelligence_count": artificial_intelligence_count,

        "feature_version": "simple_text_features_v1",
        "feature_created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

def build_text_features(text_df: pd.DataFrame) -> pd.DataFrame:
    df = text_df.copy()

    required_cols = {
        "event_id",
        "ticker",
        "form",
        "download_status",
        "document_text",
    }

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"SEC text dataset missing columns: {missing}")

    df = df[df["download_status"] == "success"].copy()
    df = df.drop_duplicates(subset=["event_id"])

    print(f"Building features for {len(df)} SEC documents...")

    features = [extract_features_for_row(row) for _, row in df.iterrows()]

    feature_df = pd.DataFrame(features)

    return feature_df

def build_ml_dataset(
    labeled: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    labeled = labeled.copy()
    features = features.copy()

    labeled = labeled.drop_duplicates(subset=["event_id"])
    features = features.drop_duplicates(subset=["event_id"])

    ml = labeled.merge(
        features,
        on=["event_id", "ticker", "form"],
        how="inner",
        validate="one_to_one",
    )
    ml["is_10k"] = (ml["form"] == "10-K").astype(int)
    ml["is_10q"] = (ml["form"] == "10-Q").astype(int)

    return ml

def validate_features(feature_df: pd.DataFrame, ml: pd.DataFrame) -> None:
    print("Running feature validation checks...")

    if feature_df.empty:
        raise ValueError("Feature dataframe is empty.")

    if ml.empty:
        raise ValueError("ML dataset is empty.")

    duplicate_features = feature_df["event_id"].duplicated().sum()

    if duplicate_features > 0:
        raise ValueError(f"Found {duplicate_features} duplicated feature event_ids.")

    required_feature_cols = [
        "word_count",
        "positive_count",
        "negative_count",
        "uncertainty_count",
        "sentiment_balance",
    ]

    for col in required_feature_cols:
        if col not in feature_df.columns:
            raise ValueError(f"Missing required feature column: {col}")

    zero_word_docs = (feature_df["word_count"] == 0).sum()

    if zero_word_docs > 0:
        print(f"Warning: {zero_word_docs} documents have zero words.")

    label_cols = [
        "fwd_abnormal_ret_1d",
        "fwd_abnormal_ret_3d",
        "fwd_abnormal_ret_5d",
        "label_up_abnormal_1d",
        "label_up_abnormal_3d",
        "label_up_abnormal_5d",
    ]

    missing_label_cols = [col for col in label_cols if col not in ml.columns]

    if missing_label_cols:
        raise ValueError(f"ML dataset missing label columns: {missing_label_cols}")

    print("Feature validation passed.")
    print(f"Feature rows: {len(feature_df)}")
    print(f"ML dataset rows: {len(ml)}")


def save_outputs(feature_df: pd.DataFrame, ml: pd.DataFrame) -> None:
    features_parquet_path = Path(OUTPUT_FEATURES_PARQUET_PATH)
    features_csv_path = Path(OUTPUT_FEATURES_CSV_PATH)
    ml_parquet_path = Path(OUTPUT_ML_PARQUET_PATH)
    ml_csv_path = Path(OUTPUT_ML_CSV_PATH)
    duckdb_path = Path(OUTPUT_DUCKDB_PATH)

    features_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    features_csv_path.parent.mkdir(parents=True, exist_ok=True)
    ml_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    ml_csv_path.parent.mkdir(parents=True, exist_ok=True)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    feature_df.to_parquet(features_parquet_path, index=False)
    feature_df.to_csv(features_csv_path, index=False)

    ml.to_parquet(ml_parquet_path, index=False)
    csv_cols = [
        col for col in ml.columns
        if col not in {"document_text"}
    ]

    ml[csv_cols].to_csv(ml_csv_path, index=False)

    con = duckdb.connect(str(duckdb_path))
    con.register("sec_text_features_df", feature_df)
    con.register("sec_ml_dataset_df", ml)

    con.execute(
        "CREATE OR REPLACE TABLE sec_text_features AS "
        "SELECT * FROM sec_text_features_df"
    )

    con.execute(
        "CREATE OR REPLACE TABLE sec_ml_dataset AS "
        "SELECT * FROM sec_ml_dataset_df"
    )

    con.close()

    print(f"Saved features Parquet: {features_parquet_path}")
    print(f"Saved features CSV:     {features_csv_path}")
    print(f"Saved ML Parquet:       {ml_parquet_path}")
    print(f"Saved ML CSV:           {ml_csv_path}")
    print(f"Saved DuckDB tables:    sec_text_features, sec_ml_dataset")


def main() -> None:
    text_df, labeled = load_inputs()

    feature_df = build_text_features(text_df)
    ml = build_ml_dataset(labeled, feature_df)

    validate_features(feature_df, ml)
    save_outputs(feature_df, ml)

    print()
    print("Feature preview:")
    print(
        feature_df[
            [
                "event_id",
                "ticker",
                "form",
                "word_count",
                "positive_count",
                "negative_count",
                "uncertainty_count",
                "sentiment_balance",
            ]
        ].head(20)
    )

    print()
    print("ML dataset preview:")
    print(
        ml[
            [
                "event_id",
                "ticker",
                "form",
                "word_count",
                "sentiment_balance",
                "fwd_abnormal_ret_3d",
                "label_up_abnormal_3d",
            ]
        ].head(20)
    )

    print()
    print("Rows by form:")
    print(ml["form"].value_counts())


if __name__ == "__main__":
    main()

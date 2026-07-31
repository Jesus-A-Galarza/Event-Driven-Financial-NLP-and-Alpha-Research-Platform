from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import time
import re

import duckdb
import pandas as pd
import requests
from bs4 import BeautifulSoup

LABELED_EVENTS_PATH = "data/processed/labels/labeled_sec_events.parquet"

OUTPUT_PARQUET_PATH = "data/processed/events/sec_event_text.parquet"
OUTPUT_CSV_PATH = "data/processed/events/sec_event_text.csv"
OUTPUT_DUCKDB_PATH = "data/processed/eventsignal.duckdb"

SEC_USER_AGENT = "EventSignalResearch/0.1 jesusabelgalarza@gmail.com"

MAX_EVENTS = None

FORMS_TO_DOWNLOAD = {
    "10-K",
    "10-Q",
}
def load_labeled_events(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing labeled events file: {path}")

    events = pd.read_parquet(path)

    required_cols = {
        "event_id",
        "ticker",
        "form",
        "document_url",
        "release_timestamp_utc",
        "fwd_abnormal_ret_1d",
        "fwd_abnormal_ret_3d",
        "label_up_abnormal_3d",
    }

    missing = required_cols - set(events.columns)

    if missing:
        raise ValueError(f"Labeled events missing required columns: {missing}")

    events = events[events["form"].isin(FORMS_TO_DOWNLOAD)].copy()
    events = events.dropna(subset=["document_url"])
    events = events.drop_duplicates(subset=["event_id"])

    events = events.sort_values(
        ["ticker", "release_timestamp_utc"]
    ).reset_index(drop=True)

    if MAX_EVENTS is not None:
        events = events.head(MAX_EVENTS).copy()

    return events

def download_sec_document(url: str) -> str:
    headers = {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }

    response = requests.get(url, headers=headers, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download SEC document: "
            f"{response.status_code} {response.reason} | {url}"
        )

    return response.text

def clean_sec_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")

    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text

def truncate_text(text: str, max_chars: int = 200_000) -> str:
    if len(text) <= max_chars:
        return text

    return text[:max_chars]

def build_text_dataset(events: pd.DataFrame) -> pd.DataFrame:
    rows = []

    total = len(events)

    for i, row in events.iterrows():
        event_id = row["event_id"]
        ticker = row["ticker"]
        form = row["form"]
        url = row["document_url"]

        print(f"[{i + 1}/{total}] {ticker} {form} | {event_id}")

        try:
            raw_html = download_sec_document(url)
            clean_text = clean_sec_html(raw_html)
            clean_text = truncate_text(clean_text)

            status = "success"
            error_message = None

        except Exception as exc:
            clean_text = None
            status = "failed"
            error_message = str(exc)

            print(f"Warning: failed to download {event_id}: {exc}")

        rows.append(
            {
                "event_id": event_id,
                "ticker": ticker,
                "form": form,
                "document_url": url,
                "download_status": status,
                "error_message": error_message,
                "document_text": clean_text,
                "document_length_chars": len(clean_text) if clean_text else 0,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

        time.sleep(0.2)

    return pd.DataFrame(rows)

def validate_text_dataset(text_df: pd.DataFrame) -> None:
    print("Running SEC text dataset validation...")

    if text_df.empty:
        raise ValueError("SEC text dataset is empty.")

    duplicate_count = text_df["event_id"].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count} duplicate event_id values.")

    success_rate = (text_df["download_status"] == "success").mean()

    print(f"Download success rate: {success_rate:.2%}")

    if success_rate < 0.80:
        raise ValueError(
            f"Low SEC document download success rate: {success_rate:.2%}"
        )

    successful = text_df[text_df["download_status"] == "success"].copy()
    short_docs = successful[successful["document_length_chars"] < 1_000]

    if len(short_docs) > 0:
        print(f"Warning: {len(short_docs)} successful documents are very short.")

    print("SEC text dataset validation passed.")

def save_outputs(text_df: pd.DataFrame) -> None:
    parquet_path = Path(OUTPUT_PARQUET_PATH)
    csv_path = Path(OUTPUT_CSV_PATH)
    duckdb_path = Path(OUTPUT_DUCKDB_PATH)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    # Full text goes into Parquet.
    # Parquet is compact and better for large text columns.
    text_df.to_parquet(parquet_path, index=False)

    # CSV only stores metadata.
    # Do NOT store full document_text in CSV because it becomes large and hard to open.
    metadata_cols = [
        "event_id",
        "ticker",
        "form",
        "document_url",
        "download_status",
        "error_message",
        "document_length_chars",
        "downloaded_at_utc",
    ]

    metadata_cols = [col for col in metadata_cols if col in text_df.columns]

    text_df[metadata_cols].to_csv(csv_path, index=False)

    con = duckdb.connect(str(duckdb_path))
    con.register("sec_event_text_df", text_df)
    con.execute(
        "CREATE OR REPLACE TABLE sec_event_text AS "
        "SELECT * FROM sec_event_text_df"
    )
    con.close()

    print(f"Saved Parquet: {parquet_path}")
    print(f"Saved CSV metadata: {csv_path}")
    print(f"Saved DuckDB:  {duckdb_path} :: sec_event_text")

def main() -> None:
    events = load_labeled_events(LABELED_EVENTS_PATH)

    print(f"Downloading text for {len(events)} SEC events...")

    text_df = build_text_dataset(events)

    validate_text_dataset(text_df)
    save_outputs(text_df)

    print()
    print("Preview:")
    print(
        text_df[
            [
                "event_id",
                "ticker",
                "form",
                "download_status",
                "document_length_chars",
            ]
        ].head(20)
    )

    print()
    print("Status counts:")
    print(text_df["download_status"].value_counts())

    print()
    print("Length summary:")
    print(text_df["document_length_chars"].describe())


if __name__ == "__main__":
    main()

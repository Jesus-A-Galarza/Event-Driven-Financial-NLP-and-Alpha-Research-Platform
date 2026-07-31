from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import time

import duckdb
import pandas as pd
import requests

TICKERS_PATH = "data/raw/tickers.csv"

OUTPUT_PARQUET_PATH = "data/processed/events/sec_events.parquet"
OUTPUT_CSV_PATH = "data/processed/events/sec_events.csv"
OUTPUT_DUCKDB_PATH = "data/processed/eventsignal.duckdb"

START_DATE = "2018-01-01"
END_DATE = "2026-07-01"
# important forms
#10-K = Main annual SEC report. Provides broad, detailed picture of company's business, finance condition, risk, financial statements.
## Contains revenue and profit trends, margin deterioration, debt problems, liquidity issues, new risks, bussiness segment changes
##manaement view of future conditions, control problems.
#10-K/A is just an amended 10-K.
#10-Q = Quarterly SEC report (only 3 quarters as the last one is the 10-K), similar but less comprenhensive than 10-K
#10-Q/A is just an amended 10-Q.
#8-K = event-like filing like earnigns release, merge or acquisition, bankruptcy, CEO/CFO change, auditor change, restatement warnign,
# major contract, annual meeting outcomes,

FORMS_TO_KEEP = {
    "10-K",
    "10-K/A",
    "10-Q",
    "10-Q/A",
    "8-K",
    "8-K/A",
}

SEC_USER_AGENT = "EventSignalResearch/0.1 {email}" #Replace {email} with your own email.

def load_universe(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Ticker file does not exist: {path}")

    universe = pd.read_csv(path)

    required_cols = {"ticker", "company", "cik"}
    missing = required_cols - set(universe.columns)

    if missing:
        raise ValueError(f"Ticker file missing columns: {missing}")

    universe = universe[
        ~universe["sector"].astype(str).str.contains("Benchmark ETF", na=False)
    ].copy()

    universe = universe.dropna(subset=["cik"])

    universe["ticker"] = universe["ticker"].astype(str).str.upper()
    universe["cik"] = universe["cik"].astype(int)

    universe = universe.drop_duplicates(subset=["ticker"])

    return universe

def format_cik_10_digits(cik: int) -> str:
    return str(int(cik)).zfill(10)

def format_cik_no_leading_zeros(cik: int) -> str:
    return str(int(cik))

def sec_get_json(url: str) -> dict:
    headers = {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }

    response = requests.get(url, headers=headers, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"SEC request failed: {response.status_code} {response.reason} for {url}"
        )

    return response.json()

def get_company_submissions(cik: int) -> dict:
    cik_10 = format_cik_10_digits(cik)
    url = f"https://data.sec.gov/submissions/CIK{cik_10}.json"
    return sec_get_json(url)

def make_filing_document_url(
    cik: int,
    accession_number: str,
    primary_document: str,
) -> str:
    cik_no_zero = format_cik_no_leading_zeros(cik)
    accession_no_dashes = accession_number.replace("-", "")

    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_zero}/{accession_no_dashes}/{primary_document}"
    )

def flatten_recent_filings(
    ticker: str,
    company: str,
    cik: int,
    submissions: dict,
) -> pd.DataFrame:
    recent = submissions.get("filings", {}).get("recent", {})

    if not recent:
        return pd.DataFrame()

    df = pd.DataFrame(recent)

    if df.empty:
        return pd.DataFrame()

    df["ticker"] = ticker
    df["company"] = company
    df["cik"] = cik

    return df

def clean_sec_events(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()

    required_cols = [
        "ticker",
        "company",
        "cik",
        "accessionNumber",
        "filingDate",
        "reportDate",
        "acceptanceDateTime",
        "form",
        "primaryDocument",
        "primaryDocDescription",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    df = df.rename(
        columns={
            "accessionNumber": "accession_number",
            "filingDate": "filing_date",
            "reportDate": "report_date",
            "acceptanceDateTime": "acceptance_datetime",
            "primaryDocument": "primary_document",
            "primaryDocDescription": "primary_doc_description",
        }
    )

    df["form"] = df["form"].astype(str)

    df = df[df["form"].isin(FORMS_TO_KEEP)].copy()

    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df["acceptance_datetime"] = pd.to_datetime(
        df["acceptance_datetime"],
        errors="coerce",
        utc=True,
    )

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)

    df = df[
        (df["filing_date"] >= start)
        & (df["filing_date"] < end)
    ].copy()

    df["source"] = "SEC"
    df["event_type"] = "sec_" + df["form"].str.lower().str.replace("-", "", regex=False)
    df["release_timestamp_utc"] = df["acceptance_datetime"]
    df["release_date"] = df["filing_date"].dt.date

    df["document_url"] = df.apply(
        lambda row: make_filing_document_url(
            cik=row["cik"],
            accession_number=row["accession_number"],
            primary_document=row["primary_document"],
        )
        if pd.notna(row["primary_document"])
        else None,
        axis=1,
    )

    df["event_id"] = (
        df["ticker"].astype(str)
        + "_"
        + df["form"].astype(str).str.replace("/", "A", regex=False)
        + "_"
        + df["accession_number"].astype(str).str.replace("-", "", regex=False)
    )

    df["provider"] = "sec_submissions_api"
    df["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()

    keep_cols = [
        "event_id",
        "ticker",
        "company",
        "cik",
        "event_type",
        "form",
        "source",
        "filing_date",
        "report_date",
        "acceptance_datetime",
        "release_timestamp_utc",
        "release_date",
        "accession_number",
        "primary_document",
        "primary_doc_description",
        "document_url",
        "provider",
        "fetched_at_utc",
    ]

    df = df[keep_cols]
    df = df.drop_duplicates(subset=["event_id"])
    df = df.sort_values(["ticker", "release_timestamp_utc", "form"]).reset_index(drop=True)

    return df

def validate_sec_events(events: pd.DataFrame) -> None:
    print("Running SEC event validation checks...")

    if events.empty:
        raise ValueError("SEC event table is empty.")

    duplicate_count = events["event_id"].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count} duplicate event_id values.")

    missing_ticker = events["ticker"].isna().sum()
    if missing_ticker > 0:
        raise ValueError(f"Found {missing_ticker} events with missing ticker.")

    missing_form = events["form"].isna().sum()
    if missing_form > 0:
        raise ValueError(f"Found {missing_form} events with missing form.")

    missing_release_time = events["release_timestamp_utc"].isna().sum()
    if missing_release_time > 0:
        print(f"Warning: {missing_release_time} events have missing release timestamp.")

    print("SEC event validation passed.")

def save_outputs(events: pd.DataFrame) -> None:
    parquet_path = Path(OUTPUT_PARQUET_PATH)
    csv_path = Path(OUTPUT_CSV_PATH)
    duckdb_path = Path(OUTPUT_DUCKDB_PATH)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    events.to_parquet(parquet_path, index=False)
    events.to_csv(csv_path, index=False)

    con = duckdb.connect(str(duckdb_path))
    con.register("sec_events_df", events)
    con.execute("CREATE OR REPLACE TABLE sec_events AS SELECT * FROM sec_events_df")
    con.close()

    print(f"Saved Parquet: {parquet_path}")
    print(f"Saved CSV:     {csv_path}")
    print(f"Saved DuckDB:  {duckdb_path} :: sec_events")


def main() -> None:
    universe = load_universe(TICKERS_PATH)

    print(f"Loading SEC filings for {len(universe)} companies...")

    all_events = []

    for i, row in universe.iterrows():
        ticker = row["ticker"]
        company = row["company"]
        cik = int(row["cik"])

        print(f"[{i + 1}/{len(universe)}] {ticker} | CIK {cik}")

        try:
            submissions = get_company_submissions(cik)
            one_company = flatten_recent_filings(
                ticker=ticker,
                company=company,
                cik=cik,
                submissions=submissions,
            )

            if not one_company.empty:
                all_events.append(one_company)

        except Exception as exc:
            print(f"Warning: failed SEC load for {ticker}: {exc}")
          
        time.sleep(0.15)

    if not all_events:
        raise ValueError("No SEC events downloaded.")

    raw_events = pd.concat(all_events, ignore_index=True)
    events = clean_sec_events(raw_events)

    validate_sec_events(events)
    save_outputs(events)

    print()
    print("Preview:")
    print(events.head(20))

    print()
    print("Events by form:")
    print(events["form"].value_counts())

    print()
    print("Events by ticker:")
    print(events.groupby("ticker").size().sort_values(ascending=False).head(20))


if __name__ == "__main__":
    main()

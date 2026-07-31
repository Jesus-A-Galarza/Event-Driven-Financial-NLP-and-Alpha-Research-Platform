from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
import yfinance as yf

TICKERS_PATH = "data/raw/tickers.csv"
OUTPUT_PARQUET_PATH = "data/processed/prices/prices_daily.parquet"
OUTPUT_CSV_PATH = "data/processed/prices/prices_daily.csv"
OUTPUT_DUCKDB_PATH = "data/processed/eventsignal.duckdb"

START_DATE = "2018-01-01"
END_DATE = "2026-07-01"

BENCHMARK = "QQQ"

def load_ticker_universe(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Ticker file does not exist: {path}")

    universe = pd.read_csv(path)

    required_cols = {'ticker', "yf_ticker", "company", "sector"}
    missing = required_cols - set(universe.columns)

    if missing:
        raise ValueError(f"Ticker file is missing columns: {missing}")

    universe["ticker"] = universe["ticker"].astype(str).str.upper()
    universe["yf_ticker"] = universe["yf_ticker"].astype(str).str.upper()

    universe = universe.drop_duplicates(subset=["yf_ticker"])

    return universe
def download_yfinance_prices(
        yf_tickers: list[str],
        start: str,
        end: str,
) -> pd.DataFrame:
    print(f"Downloading {len(yf_tickers)} tickers from {start} to {end}...")

    raw = yf.download(
        tickers = yf_tickers,
        start = start,
        end = end,
        interval = '1d',
        auto_adjust = False,
        actions = True,
        group_by = "ticker",
        threads = False,
        progress = True,
    )
    if raw.empty:
        raise ValueError("No price data downloaded.")
    frames = []
    for yf_ticker in yf_tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if yf_ticker in raw.columns.get_level_values(0):
                    df = raw[yf_ticker].copy()
                elif yf_ticker in raw.columns.get_level_values(1):
                    df = raw.xs(yf_ticker, axis=1, level=1).copy()
                else:
                    print(f"Warning: no data found for {yf_ticker}")
                    continue
            else:
                df = raw.copy()

            df = df.reset_index()
            df["yf_ticker"] = yf_ticker
            frames.append(df)

        except Exception as exc:
            print(f"Warning: failed to process {yf_ticker}: {exc}")
    if not frames:
        raise ValueError("No ticker frames were successfully processed.")

    prices = pd.concat(frames, ignore_index=True)

    return prices

def clean_price_data(prices: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()

    prices = prices.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "stock_splits",
        }
    )

    required_cols = [
        "yf_ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]

    missing = [col for col in required_cols if col not in prices.columns]

    if missing:
        raise ValueError(f"Downloaded price data is missing columns: {missing}")

    prices["date"] = pd.to_datetime(prices["date"]).dt.date

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]

    for col in numeric_cols:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")

    if "dividends" not in prices.columns:
        prices["dividends"] = 0.0

    if "stock_splits" not in prices.columns:
        prices["stock_splits"] = 0.0

    prices["dividends"] = pd.to_numeric(prices["dividends"], errors="coerce").fillna(0.0)
    prices["stock_splits"] = pd.to_numeric(prices["stock_splits"], errors="coerce").fillna(0.0)

    prices = prices.dropna(subset=["adj_close", "close"])
    prices["adj_factor"] = prices["adj_close"] / prices["close"]
    prices["adj_open"] = prices["open"] * prices["adj_factor"]
    prices["adj_high"] = prices["high"] * prices["adj_factor"]
    prices["adj_low"] = prices["low"] * prices["adj_factor"]

    metadata_cols = [
        "ticker",
        "yf_ticker",
        "company",
        "sector",
        "sub_industry",
    ]

    metadata_cols = [col for col in metadata_cols if col in universe.columns]

    prices = prices.merge(
        universe[metadata_cols],
        on="yf_ticker",
        how="left",
    )

    prices["ticker"] = prices["ticker"].fillna(prices["yf_ticker"])

    prices["provider"] = "yfinance"
    prices["interval"] = "1d"
    prices["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()

    prices = prices[
        [
            "ticker",
            "yf_ticker",
            "company",
            "sector",
            "sub_industry",
            "date",
            "open",
            "high",
            "low",
            "close",
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
            "volume",
            "dividends",
            "stock_splits",
            "adj_factor",
            "provider",
            "interval",
            "fetched_at_utc",
        ]
    ]

    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    return prices

def add_return_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    grouped = df.groupby("ticker", group_keys=False)

    df["ret_1d"] = grouped["adj_close"].pct_change(1)
    df["ret_5d"] = grouped["adj_close"].pct_change(5)
    df["ret_20d"] = grouped["adj_close"].pct_change(20)

    for horizon in [1, 3, 5, 20]:
        df[f"fwd_ret_{horizon}d"] = (
            grouped["adj_close"].shift(-horizon) / df["adj_close"] - 1.0
        )

    return df


def add_abnormal_return_labels(
    prices: pd.DataFrame,
    benchmark: str = BENCHMARK,
) -> pd.DataFrame:
    df = prices.copy()

    benchmark_df = df[df["ticker"] == benchmark][
        [
            "date",
            "fwd_ret_1d",
            "fwd_ret_3d",
            "fwd_ret_5d",
            "fwd_ret_20d",
        ]
    ].copy()

    benchmark_df = benchmark_df.rename(
        columns={
            "fwd_ret_1d": "benchmark_fwd_ret_1d",
            "fwd_ret_3d": "benchmark_fwd_ret_3d",
            "fwd_ret_5d": "benchmark_fwd_ret_5d",
            "fwd_ret_20d": "benchmark_fwd_ret_20d",
        }
    )

    df = df.merge(benchmark_df, on="date", how="left")

    for horizon in [1, 3, 5, 20]:
        df[f"fwd_abnormal_ret_{horizon}d"] = (
            df[f"fwd_ret_{horizon}d"] - df[f"benchmark_fwd_ret_{horizon}d"]
        )

        df[f"label_up_abnormal_{horizon}d"] = (
            df[f"fwd_abnormal_ret_{horizon}d"] > 0
        ).astype("Int64")

        df.loc[
            df[f"fwd_abnormal_ret_{horizon}d"].isna(),
            f"label_up_abnormal_{horizon}d",
        ] = pd.NA

    return df


def validate_price_data(prices: pd.DataFrame) -> None:
    print("Running validation checks...")

    duplicate_count = prices.duplicated(subset=["ticker", "date"]).sum()

    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count} duplicate ticker/date rows.")

    if prices["adj_close"].isna().any():
        bad = prices[prices["adj_close"].isna()][["ticker", "date"]].head(10)
        raise ValueError(f"Missing adjusted close values:\n{bad}")

    bad_ohlc = prices[
        (prices["high"] < prices["low"])
        | (prices["adj_high"] < prices["adj_low"])
    ]

    if not bad_ohlc.empty:
        raise ValueError(f"Invalid OHLC rows found:\n{bad_ohlc.head(10)}")

    print("Validation passed.")

def validate_expected_tickers(prices: pd.DataFrame, universe: pd.DataFrame) -> None:
    expected = set(universe["ticker"].astype(str).str.upper())
    actual = set(prices["ticker"].astype(str).str.upper())

    missing = sorted(expected - actual)

    if missing:
        raise ValueError(f"Missing downloaded tickers: {missing}")

    print("All expected tickers were downloaded.")


def save_outputs(prices: pd.DataFrame) -> None:
    parquet_path = Path(OUTPUT_PARQUET_PATH)
    csv_path = Path(OUTPUT_CSV_PATH)
    duckdb_path = Path(OUTPUT_DUCKDB_PATH)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    prices.to_parquet(parquet_path, index=False)
    prices.to_csv(csv_path, index=False)

    con = duckdb.connect(str(duckdb_path))
    con.register("prices_df", prices)
    con.execute("CREATE OR REPLACE TABLE prices_daily AS SELECT * FROM prices_df")
    con.close()

    print(f"Saved Parquet: {parquet_path}")
    print(f"Saved CSV:     {csv_path}")
    print(f"Saved DuckDB:  {duckdb_path} :: prices_daily")


def main() -> None:
    universe = load_ticker_universe(TICKERS_PATH)

    yf_tickers = universe["yf_ticker"].dropna().unique().tolist()

    if BENCHMARK not in yf_tickers:
        yf_tickers.append(BENCHMARK)

    raw_prices = download_yfinance_prices(
        yf_tickers=yf_tickers,
        start=START_DATE,
        end=END_DATE,
    )

    prices = clean_price_data(raw_prices, universe)
    prices = add_return_features(prices)
    prices = add_abnormal_return_labels(prices, benchmark=BENCHMARK)

    validate_price_data(prices)
    validate_expected_tickers(prices, universe)
    save_outputs(prices)

    print()
    print("Preview:")
    print(prices.head(20))

    print()
    print("Rows by ticker:")
    print(
        prices.groupby("ticker")
        .size()
        .sort_values(ascending=False)
        .head(20)
    )


if __name__ == "__main__":
    main()

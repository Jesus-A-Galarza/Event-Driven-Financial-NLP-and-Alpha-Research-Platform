from __future__ import annotations

from pathlib import Path
from bisect import bisect_left, bisect_right
from datetime import time

import duckdb
import pandas as pd


SEC_EVENTS_PATH = "data/processed/events/sec_events.parquet"
PRICES_PATH = "data/processed/prices/prices_daily.parquet"

OUTPUT_PARQUET_PATH = "data/processed/labels/labeled_sec_events.parquet"
OUTPUT_CSV_PATH = "data/processed/labels/labeled_sec_events.csv"
OUTPUT_DUCKDB_PATH = "data/processed/eventsignal.duckdb"

BENCHMARK = "QQQ"
NY_TIMEZONE = "America/New_York"

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    sec_path = Path(SEC_EVENTS_PATH)
    prices_path = Path(PRICES_PATH)

    if not sec_path.exists():
        raise FileNotFoundError(f"Missing SEC events file: {sec_path}")

    if not prices_path.exists():
        raise FileNotFoundError(f"Missing prices file: {prices_path}")

    events = pd.read_parquet(sec_path)
    prices = pd.read_parquet(prices_path)

    return events, prices

def build_trading_calendar(prices: pd.DataFrame) -> list:
    benchmark_prices = prices[prices["ticker"] == BENCHMARK].copy()

    if benchmark_prices.empty:
        raise ValueError(f"Benchmark {BENCHMARK} not found in price data.")

    trading_dates = (
        pd.to_datetime(benchmark_prices["date"])
        .dt.date
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not trading_dates:
        raise ValueError("Trading calendar is empty.")

    return trading_dates

def previous_trading_date(trading_dates: list, date_value):
    idx = bisect_left(trading_dates, date_value) - 1

    if idx < 0:
        return None

    return trading_dates[idx]

def next_trading_date(trading_dates: list, date_value, include_same: bool = True):
    if include_same:
        idx = bisect_left(trading_dates, date_value)
    else:
        idx = bisect_right(trading_dates, date_value)

    if idx >= len(trading_dates):
        return None

    return trading_dates[idx]

def classify_event_timing_and_anchor(
    release_timestamp_utc,
    trading_dates: list,
    trading_date_set: set,
) -> pd.Series:
    if pd.isna(release_timestamp_utc):
        return pd.Series(
            {
                "release_timestamp_ny": pd.NaT,
                "market_timing": "unknown",
                "price_anchor_date": None,
                "prediction_trade_date": None,
                "daily_label_quality": "missing_timestamp",
            }
        )

    ts_utc = pd.Timestamp(release_timestamp_utc)

    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize("UTC")
    else:
        ts_utc = ts_utc.tz_convert("UTC")

    ts_ny = ts_utc.tz_convert(NY_TIMEZONE)

    local_date = ts_ny.date()
    local_time = ts_ny.time()

    is_trading_day = local_date in trading_date_set

    if not is_trading_day:
        market_timing = "weekend_or_holiday"
        anchor_date = previous_trading_date(trading_dates, local_date)
        prediction_trade_date = next_trading_date(
            trading_dates,
            local_date,
            include_same=True,
        )
        quality = "daily_close_to_close"

    elif local_time < MARKET_OPEN:
        market_timing = "before_market_open"
        anchor_date = previous_trading_date(trading_dates, local_date)
        prediction_trade_date = local_date
        quality = "daily_close_to_close"

    elif MARKET_OPEN <= local_time < MARKET_CLOSE:
        market_timing = "during_market_hours"
        anchor_date = previous_trading_date(trading_dates, local_date)
        prediction_trade_date = local_date
        quality = "coarse_intraday_daily_bar"

    else:
        market_timing = "after_market_close"
        anchor_date = local_date
        prediction_trade_date = next_trading_date(
            trading_dates,
            local_date,
            include_same=False,
        )
        quality = "daily_close_to_close"

    return pd.Series(
        {
            "release_timestamp_ny": ts_ny,
            "market_timing": market_timing,
            "price_anchor_date": anchor_date,
            "prediction_trade_date": prediction_trade_date,
            "daily_label_quality": quality,
        }
    )

def prepare_events(events: pd.DataFrame, trading_dates: list) -> pd.DataFrame:
    df = events.copy()

    required_cols = {
        "event_id",
        "ticker",
        "form",
        "event_type",
        "release_timestamp_utc",
    }

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"SEC events missing columns: {missing}")

    df["ticker"] = df["ticker"].astype(str).str.upper()

    df["release_timestamp_utc"] = pd.to_datetime(
        df["release_timestamp_utc"],
        errors="coerce",
        utc=True,
    )

    trading_date_set = set(trading_dates)

    timing_info = df["release_timestamp_utc"].apply(
        lambda ts: classify_event_timing_and_anchor(
            ts,
            trading_dates=trading_dates,
            trading_date_set=trading_date_set,
        )
    )

    df = pd.concat([df, timing_info], axis=1)

    return df

def prepare_price_labels(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["price_anchor_date"] = pd.to_datetime(df["date"]).dt.date

    keep_cols = [
        "ticker",
        "price_anchor_date",
        "adj_close",
        "volume",
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "fwd_ret_1d",
        "fwd_ret_3d",
        "fwd_ret_5d",
        "fwd_ret_20d",
        "benchmark_fwd_ret_1d",
        "benchmark_fwd_ret_3d",
        "benchmark_fwd_ret_5d",
        "benchmark_fwd_ret_20d",
        "fwd_abnormal_ret_1d",
        "fwd_abnormal_ret_3d",
        "fwd_abnormal_ret_5d",
        "fwd_abnormal_ret_20d",
        "label_up_abnormal_1d",
        "label_up_abnormal_3d",
        "label_up_abnormal_5d",
        "label_up_abnormal_20d",
    ]

    missing = [col for col in keep_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Price data missing required columns: {missing}")

    return df[keep_cols]

def join_events_to_labels(
    events: pd.DataFrame,
    price_labels: pd.DataFrame,
) -> pd.DataFrame:
    labeled = events.merge(
        price_labels,
        on=["ticker", "price_anchor_date"],
        how="left",
        validate="many_to_one",
    )

    labeled = labeled.sort_values(
        ["ticker", "release_timestamp_utc", "form"]
    ).reset_index(drop=True)

    return labeled

def validate_labeled_events(labeled: pd.DataFrame) -> None:
    print("Running labeled SEC event validation checks...")

    if labeled.empty:
        raise ValueError("Labeled SEC events table is empty.")

    duplicate_count = labeled["event_id"].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count} duplicated event_id values.")

    missing_anchor = labeled["price_anchor_date"].isna().sum()

    if missing_anchor > 0:
        print(f"Warning: {missing_anchor} events have missing price_anchor_date.")

    missing_1d_label = labeled["fwd_abnormal_ret_1d"].isna().sum()
    missing_3d_label = labeled["fwd_abnormal_ret_3d"].isna().sum()
    missing_5d_label = labeled["fwd_abnormal_ret_5d"].isna().sum()

    total = len(labeled)

    print(f"Total labeled SEC events: {total}")
    print(f"Missing 1d labels: {missing_1d_label} / {total}")
    print(f"Missing 3d labels: {missing_3d_label} / {total}")
    print(f"Missing 5d labels: {missing_5d_label} / {total}")

    match_rate_1d = 1.0 - (missing_1d_label / total)

    if match_rate_1d < 0.90:
        raise ValueError(
            f"Low 1d label match rate: {match_rate_1d:.2%}. "
            "Check event timestamps, price date coverage, and ticker mapping."
        )

    print("Labeled SEC event validation passed.")


def save_outputs(labeled: pd.DataFrame) -> None:
    parquet_path = Path(OUTPUT_PARQUET_PATH)
    csv_path = Path(OUTPUT_CSV_PATH)
    duckdb_path = Path(OUTPUT_DUCKDB_PATH)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    labeled.to_parquet(parquet_path, index=False)
    labeled.to_csv(csv_path, index=False)

    con = duckdb.connect(str(duckdb_path))
    con.register("labeled_sec_events_df", labeled)
    con.execute(
        "CREATE OR REPLACE TABLE labeled_sec_events AS "
        "SELECT * FROM labeled_sec_events_df"
    )
    con.close()

    print(f"Saved Parquet: {parquet_path}")
    print(f"Saved CSV:     {csv_path}")
    print(f"Saved DuckDB:  {duckdb_path} :: labeled_sec_events")


def main() -> None:
    events, prices = load_inputs()

    trading_dates = build_trading_calendar(prices)

    events_prepared = prepare_events(events, trading_dates)
    price_labels = prepare_price_labels(prices)

    labeled = join_events_to_labels(events_prepared, price_labels)

    validate_labeled_events(labeled)
    save_outputs(labeled)

    print()
    print("Preview:")
    print(
        labeled[
            [
                "event_id",
                "ticker",
                "form",
                "release_timestamp_utc",
                "release_timestamp_ny",
                "market_timing",
                "price_anchor_date",
                "prediction_trade_date",
                "fwd_abnormal_ret_1d",
                "fwd_abnormal_ret_3d",
                "label_up_abnormal_3d",
            ]
        ].head(20)
    )

    print()
    print("Events by market timing:")
    print(labeled["market_timing"].value_counts())

    print()
    print("Events by form:")
    print(labeled["form"].value_counts())


if __name__ == "__main__":
    main()

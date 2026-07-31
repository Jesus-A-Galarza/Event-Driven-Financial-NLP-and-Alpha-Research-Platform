from __future__ import annotations
from pathlib import Path
from io import StringIO

import pandas as pd
import requests

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def convert_to_yfinance(symbol: str) -> str:
    """ Yahoo uses - instead of . for share classes"""
    return symbol.replace(".", "-")
def download_sp500_table() -> pd.DataFrame:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    response = requests.get(
        SP500_WIKI_URL,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    sp500 = tables[0]

    sp500 = sp500.rename(
        columns={
            "Symbol": "ticker",
            "Security": "company",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
            "Date added": "date_added",
            "CIK": "cik",
        }
    )

    sp500["ticker"] = sp500["ticker"].astype(str).str.upper()
    sp500["yf_ticker"] = sp500["ticker"].apply(convert_to_yfinance)

    return sp500
def build_tech_universe(sp500: pd.DataFrame) -> pd.DataFrame:
    tech_sectors = [
        "Information Technology",
        "Communication Services",
    ]

    selected_consumer_tech = {
        "AMZN",
        "TSLA",
    }
    universe = sp500[
        sp500["sector"].isin(tech_sectors)
        | (sp500["ticker"].isin(selected_consumer_tech))
    ].copy()

    #Useful to add ETF as Benchmarks.
    benchmark_rows = pd.DataFrame(
        [
            {
                "ticker": "QQQ",
                "yf_ticker": "QQQ",
                "company": "Invesco QQQ Trust",
                "sector": "Benchmark ETF",
                "sub_industry": "Nasdaq 100 ETF",
                "date_added": None,
                "cik": None,
            },
            {
               "ticker": "SPY",
                "yf_ticker": "SPY",
                "company": "SPDR S&P 500 ETF Trust",
                "sector": "Benchmark ETF",
                "sub_industry": "S&P 500 ETF",
                "date_added": None,
                "cik": None,
            },
            {
                "ticker": "XLK",
                "yf_ticker": "XLK",
                "company": "Technology Select Sector SPDR Fund",
                "sector": "Benchmark ETF",
                "sub_industry": "Technology Sector ETF",
                "date_added": None,
                "cik": None,
            },
        ]
    )

    universe = pd.concat([universe, benchmark_rows], ignore_index=True)

    universe = universe[
        [
            "ticker",
            "yf_ticker",
            "company",
            "sector",
            "sub_industry",
            "date_added",
            "cik",
        ]
    ]

    universe = universe.sort_values("ticker").reset_index(drop=True)
    return universe
def save_universe(universe: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(output_path, index=False)
    print(f"Saved {len(universe)} tickers to {output_path}")

def main() -> None:
    sp500 = download_sp500_table()
    universe = build_tech_universe(sp500)

    save_universe(universe, "data/raw/tickers.csv")

    print(universe.head(20))
    print()
    print("Tech Counts:")
    print(universe["sector"].value_counts())

if __name__ == "__main__":
    main()

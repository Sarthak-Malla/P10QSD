"""
Refresh local S&P 500 company metadata.

Downloads the open constituents CSV mirrored by datasets/s-and-p-500-companies
and writes the normalized project reference file used by dataset creation.
"""
import argparse
import os
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from src.dataloader.company_metadata import GICS_SECTOR_TO_ETF, normalize_ticker


DEFAULT_SOURCE_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "main/data/constituents.csv"
)


def normalize_constituents(df: pd.DataFrame, source_url: str) -> pd.DataFrame:
    required = ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry", "CIK"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Source metadata missing columns: {missing}")

    out = pd.DataFrame({
        "ticker": df["Symbol"].apply(normalize_ticker),
        "company_name": df["Security"].astype(str),
        "cik": df["CIK"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10),
        "gics_sector": df["GICS Sector"].astype(str),
        "gics_sub_industry": df["GICS Sub-Industry"].astype(str),
    })
    out["sector_etf"] = out["gics_sector"].map(GICS_SECTOR_TO_ETF).fillna("SPY")
    out["source"] = source_url
    out["source_date"] = date.today().isoformat()
    return out.drop_duplicates("ticker", keep="last").sort_values("ticker").reset_index(drop=True)


def validate_ticker_coverage(metadata: pd.DataFrame, tickers_file: str) -> list:
    if not tickers_file or not os.path.exists(tickers_file):
        return []
    with open(tickers_file) as f:
        tickers = {normalize_ticker(line.strip()) for line in f if line.strip()}
    known = set(metadata["ticker"])
    return sorted(tickers - known)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh S&P 500 metadata CSV.")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--out", default="data/reference/sp500_constituents.csv")
    parser.add_argument("--tickers-file", default="sp500_tickers.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.source_url)
    metadata = normalize_constituents(df, args.source_url)
    missing = validate_ticker_coverage(metadata, args.tickers_file)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    metadata.to_csv(args.out, index=False)
    print(f"Saved {len(metadata)} rows to {args.out}")
    if missing:
        print(f"WARNING: {len(missing)} tickers from {args.tickers_file} missing metadata:")
        print(", ".join(missing))


if __name__ == "__main__":
    main()

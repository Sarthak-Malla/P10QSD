"""
Company metadata lookup for ticker -> GICS sector -> sector ETF.

Runtime code reads a local, versioned CSV so dataset creation is reproducible.
Refresh that CSV with scripts/update_sp500_metadata.py when needed.
"""
import logging
import os
from functools import lru_cache
from typing import Dict

import pandas as pd


logger = logging.getLogger(__name__)

DEFAULT_METADATA_PATH = "data/reference/sp500_constituents.csv"

GICS_SECTOR_TO_ETF: Dict[str, str] = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

ALL_SECTOR_ETFS = sorted(set(GICS_SECTOR_TO_ETF.values()))


def normalize_ticker(ticker: str) -> str:
    """Normalize ticker style to the yfinance/file convention used in this repo."""
    return str(ticker).strip().upper().replace(".", "-")


@lru_cache(maxsize=8)
def load_company_metadata(metadata_path: str = DEFAULT_METADATA_PATH) -> pd.DataFrame:
    """Load normalized company metadata from the local reference CSV."""
    if not os.path.exists(metadata_path):
        logger.warning("Company metadata file not found: %s", metadata_path)
        return pd.DataFrame(columns=[
            "ticker", "company_name", "cik", "gics_sector",
            "gics_sub_industry", "sector_etf", "source", "source_date",
        ])

    df = pd.read_csv(metadata_path, dtype={"cik": str})
    rename_map = {
        "Symbol": "ticker",
        "Security": "company_name",
        "GICS Sector": "gics_sector",
        "GICS Sub-Industry": "gics_sub_industry",
        "CIK": "cik",
    }
    df = df.rename(columns=rename_map)
    if "ticker" not in df.columns:
        raise ValueError(f"Metadata file {metadata_path} is missing ticker/Symbol")

    df["ticker"] = df["ticker"].apply(normalize_ticker)
    if "gics_sector" not in df.columns:
        df["gics_sector"] = "Unknown"
    if "gics_sub_industry" not in df.columns:
        df["gics_sub_industry"] = ""
    if "company_name" not in df.columns:
        df["company_name"] = ""
    if "cik" not in df.columns:
        df["cik"] = ""
    if "sector_etf" not in df.columns:
        df["sector_etf"] = df["gics_sector"].map(GICS_SECTOR_TO_ETF).fillna("SPY")
    if "source" not in df.columns:
        df["source"] = "local"
    if "source_date" not in df.columns:
        df["source_date"] = ""

    df["sector_etf"] = df["sector_etf"].fillna("SPY")
    df["gics_sector"] = df["gics_sector"].fillna("Unknown")
    return df.drop_duplicates("ticker", keep="last").reset_index(drop=True)


def get_ticker_metadata(ticker: str, metadata_path: str = DEFAULT_METADATA_PATH) -> dict:
    """Return metadata for a ticker, falling back to Unknown/SPY."""
    norm = normalize_ticker(ticker)
    df = load_company_metadata(metadata_path)
    match = df[df["ticker"] == norm]
    if match.empty:
        return {
            "ticker": norm,
            "company_name": "",
            "cik": "",
            "gics_sector": "Unknown",
            "gics_sub_industry": "",
            "sector_etf": "SPY",
            "source": "missing",
            "source_date": "",
        }
    return match.iloc[0].to_dict()


def get_ticker_sector(ticker: str, metadata_path: str = DEFAULT_METADATA_PATH) -> str:
    """Return GICS sector for a ticker, or Unknown if missing."""
    return str(get_ticker_metadata(ticker, metadata_path).get("gics_sector") or "Unknown")


def get_sector_etf(ticker: str, metadata_path: str = DEFAULT_METADATA_PATH) -> str:
    """Return the sector ETF benchmark for a ticker, or SPY if missing."""
    return str(get_ticker_metadata(ticker, metadata_path).get("sector_etf") or "SPY")


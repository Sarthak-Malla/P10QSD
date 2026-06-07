"""
sector_benchmarks.py

Provides sector-ETF benchmark prices for computing sector-adjusted abnormal
returns. Ticker -> sector -> ETF metadata lives in company_metadata.py.

Sector ETFs (SPDR Select Sector):
    XLK  Technology
    XLF  Financials
    XLV  Health Care
    XLY  Consumer Discretionary
    XLP  Consumer Staples
    XLE  Energy
    XLI  Industrials
    XLU  Utilities
    XLB  Materials
    XLRE Real Estate
    XLC  Communication Services

Fallback to SPY if sector can't be determined.

Usage:
    from src.dataloader.sector_benchmarks import get_sector_etf, fetch_all_benchmark_prices
    etf_dfs = fetch_all_benchmark_prices(start_date, end_date)
    etf = get_sector_etf("AAPL")  # -> "XLK"
    benchmark_df = etf_dfs[etf]
"""
import logging
import time
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

from src.dataloader.company_metadata import ALL_SECTOR_ETFS, get_sector_etf

logger = logging.getLogger(__name__)

_etf_price_cache: Dict[str, pd.DataFrame] = {}


def fetch_all_benchmark_prices(start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Download price history for SPY + all 11 sector ETFs.
    Returns dict {symbol: DataFrame with OHLCV}.
    """
    global _etf_price_cache
    if _etf_price_cache:
        return _etf_price_cache

    symbols = ["SPY"] + ALL_SECTOR_ETFS
    logger.info(f"Downloading benchmark prices for {len(symbols)} ETFs ({start_date} → {end_date})")

    out = {}
    for sym in symbols:
        try:
            df = yf.download(sym, start=start_date, end=end_date,
                            auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 0:
                out[sym] = df
                logger.info(f"  {sym}: {len(df)} rows")
            else:
                logger.warning(f"  {sym}: empty (yfinance returned no data)")
        except Exception as e:
            logger.warning(f"  {sym}: failed → {e}")
        time.sleep(0.05)

    _etf_price_cache = out
    return out


def get_benchmark_df_for_ticker(
    ticker: str,
    etf_dfs: Dict[str, pd.DataFrame],
    metadata_path: str = "data/reference/sp500_constituents.csv",
) -> Optional[pd.DataFrame]:
    """Given a ticker and pre-fetched ETF dataframes, return the right benchmark series."""
    etf = get_sector_etf(ticker, metadata_path=metadata_path)
    if etf in etf_dfs:
        return etf_dfs[etf]
    # Fall back to SPY
    return etf_dfs.get("SPY")

"""
sector_benchmarks.py

Maps each ticker to its S&P sector and provides sector-ETF benchmark prices
for computing sector-adjusted abnormal returns.

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
import os
import json
import logging
import time
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# GICS sector → SPDR ETF mapping
GICS_TO_ETF = {
    "Technology":               "XLK",
    "Information Technology":   "XLK",
    "Financial Services":       "XLF",
    "Financials":               "XLF",
    "Healthcare":               "XLV",
    "Health Care":              "XLV",
    "Consumer Cyclical":        "XLY",
    "Consumer Discretionary":   "XLY",
    "Consumer Defensive":       "XLP",
    "Consumer Staples":         "XLP",
    "Energy":                   "XLE",
    "Industrials":              "XLI",
    "Utilities":                "XLU",
    "Basic Materials":          "XLB",
    "Materials":                "XLB",
    "Real Estate":              "XLRE",
    "Communication Services":   "XLC",
    "Communications":           "XLC",
}

ALL_ETFS = sorted(set(GICS_TO_ETF.values()))

# Hardcoded fallback for tickers we can't get from yfinance
# Subset of well-known mappings — for the rest we fall back to SPY
FALLBACK_TICKER_TO_SECTOR = {
    # Tech / Comm
    "AAPL":"Technology","MSFT":"Technology","GOOGL":"Communication Services",
    "GOOG":"Communication Services","NVDA":"Technology","META":"Communication Services",
    "TSLA":"Consumer Cyclical","ADBE":"Technology","CRM":"Technology","ORCL":"Technology",
    "CSCO":"Technology","INTC":"Technology","AMD":"Technology","IBM":"Technology",
    "QCOM":"Technology","TXN":"Technology","AVGO":"Technology","MU":"Technology",
    "AMAT":"Technology","ADI":"Technology","LRCX":"Technology","KLAC":"Technology",
    "NOW":"Technology","INTU":"Technology","SNPS":"Technology","CDNS":"Technology",
    "PANW":"Technology","FTNT":"Technology",
    # Fin
    "JPM":"Financial Services","BAC":"Financial Services","WFC":"Financial Services",
    "C":"Financial Services","GS":"Financial Services","MS":"Financial Services",
    "BLK":"Financial Services","SPGI":"Financial Services","AXP":"Financial Services",
    "V":"Financial Services","MA":"Financial Services","COF":"Financial Services",
    "USB":"Financial Services","PNC":"Financial Services","TFC":"Financial Services",
    "SCHW":"Financial Services","CB":"Financial Services","PGR":"Financial Services",
    "MMC":"Financial Services","ICE":"Financial Services","CME":"Financial Services",
    "AON":"Financial Services","TRV":"Financial Services","AFL":"Financial Services",
    # Health
    "JNJ":"Healthcare","PFE":"Healthcare","MRK":"Healthcare","ABBV":"Healthcare",
    "TMO":"Healthcare","ABT":"Healthcare","LLY":"Healthcare","DHR":"Healthcare",
    "BMY":"Healthcare","AMGN":"Healthcare","GILD":"Healthcare","CVS":"Healthcare",
    "CI":"Healthcare","HUM":"Healthcare","ISRG":"Healthcare","SYK":"Healthcare",
    "BSX":"Healthcare","MDT":"Healthcare","REGN":"Healthcare","VRTX":"Healthcare",
    "ZTS":"Healthcare","ELV":"Healthcare","UNH":"Healthcare","BDX":"Healthcare",
    "BIIB":"Healthcare",
    # Cons
    "WMT":"Consumer Defensive","COST":"Consumer Defensive","PG":"Consumer Defensive",
    "KO":"Consumer Defensive","PEP":"Consumer Defensive","MO":"Consumer Defensive",
    "PM":"Consumer Defensive","CL":"Consumer Defensive","KMB":"Consumer Defensive",
    "GIS":"Consumer Defensive","HSY":"Consumer Defensive","K":"Consumer Defensive",
    "KHC":"Consumer Defensive","CAG":"Consumer Defensive","ADM":"Consumer Defensive",
    "TSN":"Consumer Defensive","CPB":"Consumer Defensive","MKC":"Consumer Defensive",
    "HD":"Consumer Cyclical","LOW":"Consumer Cyclical","TGT":"Consumer Cyclical",
    "NKE":"Consumer Cyclical","SBUX":"Consumer Cyclical","MCD":"Consumer Cyclical",
    "BKNG":"Consumer Cyclical","ABNB":"Consumer Cyclical","HLT":"Consumer Cyclical",
    "MAR":"Consumer Cyclical","DIS":"Communication Services","NFLX":"Communication Services",
    "CMCSA":"Communication Services","AMZN":"Consumer Cyclical","EBAY":"Consumer Cyclical",
    "ETSY":"Consumer Cyclical",
    # Energy
    "XOM":"Energy","CVX":"Energy","COP":"Energy","SLB":"Energy","EOG":"Energy",
    "PXD":"Energy","OXY":"Energy","PSX":"Energy","VLO":"Energy","MPC":"Energy",
    "HES":"Energy","FANG":"Energy","DVN":"Energy",
    # Industrials
    "BA":"Industrials","CAT":"Industrials","DE":"Industrials","HON":"Industrials",
    "LMT":"Industrials","RTX":"Industrials","GE":"Industrials","MMM":"Industrials",
    "UPS":"Industrials","FDX":"Industrials","ETN":"Industrials","ITW":"Industrials",
    "EMR":"Industrials","PH":"Industrials","NSC":"Industrials","UNP":"Industrials",
    "CSX":"Industrials","LUV":"Industrials","DAL":"Industrials","UAL":"Industrials",
    "AAL":"Industrials","CMI":"Industrials","NOC":"Industrials","GD":"Industrials",
}

# Module-level cache to avoid repeated yfinance calls
_sector_cache: Dict[str, str] = {}
_etf_price_cache: Dict[str, pd.DataFrame] = {}

SECTOR_CACHE_FILE = "data/raw/sector_cache.json"


def _load_sector_cache():
    """Load persisted sector classification cache."""
    global _sector_cache
    if _sector_cache:
        return
    if os.path.exists(SECTOR_CACHE_FILE):
        try:
            with open(SECTOR_CACHE_FILE) as f:
                _sector_cache = json.load(f)
            logger.info(f"Loaded {len(_sector_cache)} cached ticker sectors")
        except Exception as e:
            logger.warning(f"Could not load sector cache: {e}")
            _sector_cache = {}


def _save_sector_cache():
    os.makedirs(os.path.dirname(SECTOR_CACHE_FILE) or ".", exist_ok=True)
    try:
        with open(SECTOR_CACHE_FILE, "w") as f:
            json.dump(_sector_cache, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save sector cache: {e}")


def get_sector_etf(ticker: str, fetch_if_unknown: bool = False) -> str:
    """
    Return the SPDR sector ETF symbol for a given stock ticker.
    Falls back to 'SPY' if sector unknown.
    """
    _load_sector_cache()

    # Cache hit
    if ticker in _sector_cache:
        sector = _sector_cache[ticker]
        return GICS_TO_ETF.get(sector, "SPY")

    # Hardcoded fallback
    if ticker in FALLBACK_TICKER_TO_SECTOR:
        sector = FALLBACK_TICKER_TO_SECTOR[ticker]
        _sector_cache[ticker] = sector
        return GICS_TO_ETF.get(sector, "SPY")

    # Optional yfinance lookup
    if fetch_if_unknown:
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector") or info.get("Sector")
            if sector:
                _sector_cache[ticker] = sector
                return GICS_TO_ETF.get(sector, "SPY")
        except Exception as e:
            logger.debug(f"yfinance sector lookup failed for {ticker}: {e}")

    return "SPY"


def populate_sector_cache(tickers, fetch_if_unknown: bool = True):
    """Bulk populate sector cache for all tickers. Saves to disk at end."""
    _load_sector_cache()
    n_known = 0
    n_fetched = 0
    n_unknown = 0
    for t in tickers:
        if t in _sector_cache:
            n_known += 1
            continue
        if t in FALLBACK_TICKER_TO_SECTOR:
            _sector_cache[t] = FALLBACK_TICKER_TO_SECTOR[t]
            n_known += 1
            continue
        if fetch_if_unknown:
            try:
                info = yf.Ticker(t).info
                sector = info.get("sector") or info.get("Sector")
                if sector:
                    _sector_cache[t] = sector
                    n_fetched += 1
                else:
                    n_unknown += 1
                time.sleep(0.05)  # be polite to yfinance
            except Exception:
                n_unknown += 1
        else:
            n_unknown += 1
    _save_sector_cache()
    logger.info(f"Sector cache: {n_known} known, {n_fetched} freshly fetched, {n_unknown} unknown (fall back to SPY)")


def fetch_all_benchmark_prices(start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Download price history for SPY + all 11 sector ETFs.
    Returns dict {symbol: DataFrame with OHLCV}.
    """
    global _etf_price_cache
    if _etf_price_cache:
        return _etf_price_cache

    symbols = ["SPY"] + ALL_ETFS
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


def get_benchmark_df_for_ticker(ticker: str, etf_dfs: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Given a ticker and pre-fetched ETF dataframes, return the right benchmark series."""
    etf = get_sector_etf(ticker)
    if etf in etf_dfs:
        return etf_dfs[etf]
    # Fall back to SPY
    return etf_dfs.get("SPY")

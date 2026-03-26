"""
filing_dataset.py

Builds a filing-aligned dataset:
- 1 row per 10-Q filing per company
- Text features: cosine similarity vs previous quarter, VADER sentiment, text length
- Price features: momentum, volatility, RSI, MA ratio computed AT filing date
- Target: did stock go up in next N days after filing?
- Saves to data/processed/filing_aligned.csv
"""

import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import hydra
from omegaconf import DictConfig
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Text features ─────────────────────────────────────────────────────────────

def cosine_sim_consecutive(texts: pd.Series) -> pd.Series:
    """
    For each filing, compute TF-IDF cosine similarity vs the PREVIOUS filing.
    Returns NaN for the first filing (no previous to compare).
    """
    sims = [np.nan]
    texts_list = texts.fillna("").tolist()

    for i in range(1, len(texts_list)):
        prev, curr = texts_list[i - 1], texts_list[i]
        if not prev.strip() or not curr.strip():
            sims.append(np.nan)
            continue
        try:
            vec = TfidfVectorizer(max_features=5000, stop_words="english")
            tfidf = vec.fit_transform([prev, curr])
            sim = cosine_similarity(tfidf[0], tfidf[1])[0][0]
            sims.append(float(sim))
        except Exception:
            sims.append(np.nan)

    return pd.Series(sims, index=texts.index)


def vader_sentiment(text: str) -> dict:
    """VADER compound, positive, negative scores."""
    sia = SentimentIntensityAnalyzer()
    if not isinstance(text, str) or not text.strip():
        return {"sentiment_compound": np.nan,
                "sentiment_pos": np.nan,
                "sentiment_neg": np.nan}
    s = sia.polarity_scores(text)
    return {"sentiment_compound": s["compound"],
            "sentiment_pos": s["pos"],
            "sentiment_neg": s["neg"]}


# ── Price features ────────────────────────────────────────────────────────────

def calculate_rsi(prices: pd.Series, window: int = 14) -> float:
    """RSI at the last available price point."""
    if len(prices) < window + 1:
        return np.nan
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def get_price_features_at_filing(
    filing_date: pd.Timestamp,
    price_df: pd.DataFrame,
    lookback: int = 50,
) -> dict:
    """
    Compute technical features from price data BEFORE the filing date.
    Uses up to `lookback` trading days prior to the filing.
    """
    empty = {
        "price_return_1d": np.nan,
        "price_return_5d": np.nan,
        "price_return_20d": np.nan,
        "price_volatility_20d": np.nan,
        "price_ma_ratio_5": np.nan,
        "price_ma_ratio_20": np.nan,
        "price_rsi": np.nan,
    }

    try:
        closes = price_df["Close"].dropna()
        closes.index = pd.to_datetime(closes.index)
        closes = closes.sort_index()

        # Get data up to (but not including) filing date
        prior = closes[closes.index < filing_date]
        if len(prior) < lookback:
            return empty

        window = prior.iloc[-lookback:]

        # Returns
        ret_1d  = float((window.iloc[-1] - window.iloc[-2]) / window.iloc[-2]) if len(window) >= 2 else np.nan
        ret_5d  = float((window.iloc[-1] - window.iloc[-6]) / window.iloc[-6]) if len(window) >= 6 else np.nan
        ret_20d = float((window.iloc[-1] - window.iloc[-21]) / window.iloc[-21]) if len(window) >= 21 else np.nan

        # Volatility (annualized)
        daily_returns = window.pct_change().dropna()
        vol_20d = float(daily_returns.iloc[-20:].std() * np.sqrt(252)) if len(daily_returns) >= 20 else np.nan

        # MA ratios (price / moving average)
        ma5  = float(window.iloc[-1] / window.iloc[-5:].mean())  if len(window) >= 5  else np.nan
        ma20 = float(window.iloc[-1] / window.iloc[-20:].mean()) if len(window) >= 20 else np.nan

        # RSI
        rsi = calculate_rsi(window)

        return {
            "price_return_1d":    ret_1d,
            "price_return_5d":    ret_5d,
            "price_return_20d":   ret_20d,
            "price_volatility_20d": vol_20d,
            "price_ma_ratio_5":   ma5,
            "price_ma_ratio_20":  ma20,
            "price_rsi":          rsi,
        }

    except Exception as e:
        logger.warning(f"Price features failed at {filing_date}: {e}")
        return empty


def get_price_return_after_filing(
    filing_date: pd.Timestamp,
    horizon: int,
    price_df: pd.DataFrame,
) -> float:
    """% price change from filing_date over next `horizon` trading days."""
    try:
        closes = price_df["Close"].dropna()
        closes.index = pd.to_datetime(closes.index)
        closes = closes.sort_index()

        future = closes[closes.index >= filing_date]
        if len(future) < horizon + 1:
            return np.nan

        return float((future.iloc[horizon] - future.iloc[0]) / future.iloc[0])
    except Exception as e:
        logger.warning(f"Price return failed at {filing_date}: {e}")
        return np.nan


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    # Load tickers
    tickers_file = cfg.data.tickers_file
    if not os.path.exists(tickers_file):
        raise FileNotFoundError(f"Tickers file not found: {tickers_file}")
    with open(tickers_file) as f:
        tickers = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(tickers)} tickers from {tickers_file}")

    sec_dir = os.path.join(cfg.data.raw_dir, "sec_filings")
    processed_dir = cfg.data.processed_dir
    horizon = cfg.features.prediction_horizon
    os.makedirs(processed_dir, exist_ok=True)

    text_col = f"section_{cfg.sec.sections[0]}"
    all_records = []
    skipped = []

    for ticker in tickers:
        # ── Load SEC filings ──────────────────────────────────────────────
        parquet_path = os.path.join(sec_dir, f"{ticker}_filings.parquet")
        if not os.path.exists(parquet_path):
            logger.warning(f"{ticker}: No SEC filings found, skipping.")
            skipped.append(ticker)
            continue

        filings_df = pd.read_parquet(parquet_path)
        filings_df["filed_at"] = pd.to_datetime(filings_df["filed_at"])
        filings_df = filings_df.sort_values("filed_at").reset_index(drop=True)

        has_text = filings_df[text_col].notna().sum()
        if has_text == 0:
            logger.warning(f"{ticker}: No text extracted, skipping.")
            skipped.append(ticker)
            continue

        logger.info(f"{ticker}: {has_text}/{len(filings_df)} filings have text")

        # ── Download price data ───────────────────────────────────────────
        try:
            price_df = yf.download(
                ticker,
                start=cfg.data.start_date,
                end=cfg.data.end_date,
                auto_adjust=True,
                progress=False,
            )
            if price_df.empty:
                logger.warning(f"{ticker}: Empty price data, skipping.")
                skipped.append(ticker)
                continue
        except Exception as e:
            logger.error(f"{ticker}: Price download failed: {e}")
            skipped.append(ticker)
            continue

        # Flatten MultiIndex columns if present (yfinance quirk)
        if isinstance(price_df.columns, pd.MultiIndex):
            price_df.columns = price_df.columns.get_level_values(0)

        # ── Text features ─────────────────────────────────────────────────
        filings_df["cosine_sim_prev"] = cosine_sim_consecutive(filings_df[text_col])

        sentiment_rows = filings_df[text_col].apply(vader_sentiment)
        sentiment_df = pd.DataFrame(sentiment_rows.tolist())
        filings_df = pd.concat([filings_df.reset_index(drop=True),
                                 sentiment_df.reset_index(drop=True)], axis=1)

        filings_df["text_length"] = filings_df[text_col].apply(
            lambda x: len(x.split()) if isinstance(x, str) else np.nan
        )
        mean_len = filings_df["text_length"].mean()
        filings_df["text_length_norm"] = (
            filings_df["text_length"] / mean_len if mean_len > 0 else np.nan
        )

        # ── Price features at filing date ─────────────────────────────────
        price_feature_rows = filings_df["filed_at"].apply(
            lambda d: get_price_features_at_filing(d, price_df)
        )
        price_feat_df = pd.DataFrame(price_feature_rows.tolist())
        filings_df = pd.concat([filings_df.reset_index(drop=True),
                                 price_feat_df.reset_index(drop=True)], axis=1)

        # ── Target: price return after filing ─────────────────────────────
        filings_df["price_return"] = filings_df["filed_at"].apply(
            lambda d: get_price_return_after_filing(d, horizon, price_df)
        )
        filings_df["target"] = (filings_df["price_return"] > 0).astype(int)

        # ── Select final columns ──────────────────────────────────────────
        record_cols = [
            "filed_at", "ticker",
            # Text features
            "cosine_sim_prev",
            "sentiment_compound", "sentiment_pos", "sentiment_neg",
            "text_length_norm",
            # Price features at filing date
            "price_return_1d", "price_return_5d", "price_return_20d",
            "price_volatility_20d", "price_ma_ratio_5", "price_ma_ratio_20",
            "price_rsi",
            # Target
            "price_return", "target",
        ]
        available = [c for c in record_cols if c in filings_df.columns]
        records = filings_df[available].dropna(subset=["target", "cosine_sim_prev"])

        logger.info(f"{ticker}: {len(records)} valid rows")
        all_records.append(records)

    if not all_records:
        logger.error("No data collected. Exiting.")
        return

    # ── Combine and save ──────────────────────────────────────────────────
    final_df = pd.concat(all_records, ignore_index=True)
    final_df = final_df.sort_values("filed_at").reset_index(drop=True)

    output_path = os.path.join(processed_dir, "filing_aligned.csv")
    final_df.to_csv(output_path, index=False)

    logger.info(f"\n{'='*50}")
    logger.info(f"Dataset saved to {output_path}")
    logger.info(f"Total rows:          {len(final_df)}")
    logger.info(f"Tickers with data:   {final_df['ticker'].nunique()}")
    logger.info(f"Tickers skipped:     {len(skipped)} → {skipped}")
    logger.info(f"Date range:          {final_df['filed_at'].min()} → {final_df['filed_at'].max()}")
    logger.info(f"Features:            {[c for c in final_df.columns if c not in ['filed_at','ticker','price_return','target']]}")
    logger.info(f"Target distribution:\n{final_df['target'].value_counts()}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
"""
run_flsed.py  (v2)

Computes FLSED features by reading raw text from data/raw/sec_filings/*.parquet
and merging onto data/processed/filing_aligned.csv via (ticker, filed_at).

Run from project root:
    venv/bin/python -m src.features.run_flsed
"""
import os
import sys
import time
import logging
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))

from src.features.flsed import compute_flsed_features, build_sector_peer_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_flsed")

SECTOR_MAP = {
    "AAPL":"Tech","MSFT":"Tech","GOOGL":"Tech","GOOG":"Tech","NVDA":"Tech","META":"Tech","TSLA":"Tech",
    "ADBE":"Tech","CRM":"Tech","ORCL":"Tech","CSCO":"Tech","INTC":"Tech","AMD":"Tech","IBM":"Tech",
    "QCOM":"Tech","TXN":"Tech","AVGO":"Tech","MU":"Tech","AMAT":"Tech","ADI":"Tech","LRCX":"Tech",
    "KLAC":"Tech","NOW":"Tech","INTU":"Tech","SNPS":"Tech","CDNS":"Tech","PANW":"Tech","FTNT":"Tech",
    "JPM":"Fin","BAC":"Fin","WFC":"Fin","C":"Fin","GS":"Fin","MS":"Fin","BLK":"Fin","SPGI":"Fin",
    "AXP":"Fin","V":"Fin","MA":"Fin","COF":"Fin","USB":"Fin","PNC":"Fin","TFC":"Fin","SCHW":"Fin",
    "CB":"Fin","PGR":"Fin","MMC":"Fin","ICE":"Fin","CME":"Fin","AON":"Fin","TRV":"Fin","AFL":"Fin",
    "JNJ":"Health","PFE":"Health","MRK":"Health","ABBV":"Health","TMO":"Health","ABT":"Health","LLY":"Health",
    "DHR":"Health","BMY":"Health","AMGN":"Health","GILD":"Health","CVS":"Health","CI":"Health","HUM":"Health",
    "ISRG":"Health","SYK":"Health","BSX":"Health","MDT":"Health","REGN":"Health","VRTX":"Health","ZTS":"Health",
    "ELV":"Health","UNH":"Health","BDX":"Health","BIIB":"Health",
    "WMT":"Cons","COST":"Cons","HD":"Cons","LOW":"Cons","TGT":"Cons","PG":"Cons","KO":"Cons","PEP":"Cons",
    "MO":"Cons","PM":"Cons","NKE":"Cons","SBUX":"Cons","MCD":"Cons","CL":"Cons","KMB":"Cons","GIS":"Cons",
    "HSY":"Cons","K":"Cons","KHC":"Cons","CAG":"Cons","ADM":"Cons","TSN":"Cons","CPB":"Cons","MKC":"Cons",
    "BKNG":"Cons","ABNB":"Cons","HLT":"Cons","MAR":"Cons","DIS":"Cons","NFLX":"Cons","CMCSA":"Cons",
    "AMZN":"Cons","EBAY":"Cons","ETSY":"Cons",
    "XOM":"Energy","CVX":"Energy","COP":"Energy","SLB":"Energy","EOG":"Energy","PXD":"Energy","OXY":"Energy",
    "PSX":"Energy","VLO":"Energy","MPC":"Energy","HES":"Energy","FANG":"Energy","DVN":"Energy",
    "BA":"Ind","CAT":"Ind","DE":"Ind","HON":"Ind","LMT":"Ind","RTX":"Ind","GE":"Ind","MMM":"Ind",
    "UPS":"Ind","FDX":"Ind","ETN":"Ind","ITW":"Ind","EMR":"Ind","PH":"Ind","NSC":"Ind","UNP":"Ind",
    "CSX":"Ind","LUV":"Ind","DAL":"Ind","UAL":"Ind","AAL":"Ind","CMI":"Ind","NOC":"Ind","GD":"Ind",
}

INPUT_CSV  = "data/processed/filing_aligned.csv"
PARQUET_DIR = "data/raw/sec_filings"
OUTPUT_CSV = "data/processed/filing_aligned_v4.csv"


def load_all_text() -> pd.DataFrame:
    logger.info(f"Loading raw text from {PARQUET_DIR}/...")
    files = [f for f in os.listdir(PARQUET_DIR) if f.endswith("_filings.parquet")]
    logger.info(f"Found {len(files)} ticker parquet files")

    all_rows = []
    text_col_candidates = ["section_part_ii_item_1a", "section_Item 1A", "section_Item_1A",
                           "section_1A", "item_1a", "risk_factors", "text"]

    for i, fname in enumerate(files):
        ticker = fname.replace("_filings.parquet", "")
        try:
            pdf = pd.read_parquet(os.path.join(PARQUET_DIR, fname))
            text_col = None
            for c in text_col_candidates:
                if c in pdf.columns:
                    text_col = c
                    break
            if text_col is None:
                possibles = [c for c in pdf.columns
                             if "1A" in c or "item_1a" in c.lower() or "risk" in c.lower()]
                if possibles:
                    text_col = possibles[0]
            if text_col is None or "filed_at" not in pdf.columns:
                continue

            for _, row in pdf.iterrows():
                txt = row.get(text_col)
                if isinstance(txt, str) and len(txt) > 100:
                    all_rows.append({
                        "ticker": ticker,
                        "filed_at": pd.Timestamp(row["filed_at"]),
                        "text": txt,
                    })
        except Exception as e:
            logger.warning(f"Failed to read {fname}: {e}")
            continue
        if (i+1) % 100 == 0:
            logger.info(f"  loaded {i+1}/{len(files)} files | {len(all_rows)} rows so far")

    text_df = pd.DataFrame(all_rows)
    logger.info(f"Loaded {len(text_df)} text rows from {text_df['ticker'].nunique()} tickers")
    return text_df


def main():
    t0 = time.time()

    logger.info(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, parse_dates=["filed_at"])
    logger.info(f"Loaded {len(df)} feature rows from {df['ticker'].nunique()} tickers")

    text_df = load_all_text()
    if len(text_df) == 0:
        logger.error("Could not load any raw text. Cannot run FLSED.")
        sys.exit(1)

    df["filed_at_key"] = pd.to_datetime(df["filed_at"]).dt.normalize()
    text_df["filed_at_key"] = pd.to_datetime(text_df["filed_at"]).dt.normalize()

    merged = df.merge(
        text_df[["ticker", "filed_at_key", "text"]],
        on=["ticker", "filed_at_key"],
        how="left",
    )

    n_with_text = merged["text"].notna().sum()
    logger.info(f"Merged: {n_with_text}/{len(merged)} rows have raw text")
    if n_with_text < len(merged) * 0.8:
        logger.warning("Less than 80% of rows have text — possible date-match issue")

    merged = merged.drop(columns=["filed_at_key"])

    logger.info("=" * 60)
    logger.info("STEP 1/2: Building sector peer sentence pool (one-time)")
    logger.info("=" * 60)
    merged_for_pool = merged.dropna(subset=["text"]).copy()
    peer_pool = build_sector_peer_pool(merged_for_pool, SECTOR_MAP, text_col="text")

    logger.info("=" * 60)
    logger.info("STEP 2/2: Computing FLSED features per ticker")
    logger.info("=" * 60)

    flsed_results = []
    tickers = sorted(merged["ticker"].unique())
    total = len(tickers)
    start = time.time()

    for i, ticker in enumerate(tickers, 1):
        ticker_df = merged[merged["ticker"] == ticker].sort_values("filed_at").copy().reset_index(drop=False)
        if ticker_df["text"].notna().sum() == 0:
            continue

        sector = SECTOR_MAP.get(ticker, "Other")
        sector_pool = peer_pool.get(sector, {})

        try:
            features = compute_flsed_features(
                ticker_df, text_col="text", ticker=ticker,
                peer_sentence_pool=sector_pool,
            )
            features.index = ticker_df["index"].values
            flsed_results.append(features)
        except Exception as e:
            logger.warning(f"{ticker}: FLSED failed: {e}")
            continue

        if i % 10 == 0 or i == total:
            elapsed = time.time() - start
            rate = i / max(elapsed, 1)
            eta_min = (total - i) / max(rate, 0.001) / 60
            logger.info(f"  [{i}/{total}] processed | elapsed={elapsed/60:.1f}m | ETA={eta_min:.1f}m")

    logger.info("Merging FLSED features back into main dataframe...")
    flsed_all = pd.concat(flsed_results, axis=0).sort_index()
    df = df.merge(flsed_all, left_index=True, right_index=True, how="left")

    if "text" in df.columns:
        df = df.drop(columns=["text"])

    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Saved {OUTPUT_CSV} with {len(df)} rows")

    logger.info("=" * 60)
    logger.info("FLSED FEATURE STATS")
    logger.info("=" * 60)
    for col in ["n_new_sentences", "avg_finbert_sent_new", "avg_fwd_specificity_new", "peer_sentence_overlap"]:
        if col in df.columns:
            s = df[col]
            logger.info(f"  {col:30s}: non-null={s.notna().sum():5d}/{len(df):5d}  "
                        f"mean={s.mean():.4f}  std={s.std():.4f}")

    total_min = (time.time() - t0) / 60
    logger.info(f"Total runtime: {total_min:.1f} minutes")


if __name__ == "__main__":
    main()

#!/bin/bash
# ============================================================
# scale_to_sp500.sh
# Scales the full pipeline to all 503 S&P 500 companies.
# Safe to interrupt and resume — checkpointing throughout.
#
# Run from inside P10QSD:
#   caffeinate -i bash scale_to_sp500.sh
#
# Expected runtime:
#   SEC download:    8-12 hours (456 new companies)
#   filing_dataset:  15-20 minutes
#   baseline:        5-10 minutes
# ============================================================
set -e
echo "=========================================="
echo " P10QSD Scale to Full S&P 500"
echo "=========================================="
echo "Started: $(date)"
echo ""

# ── Step 1: Update config to use full ticker list ─────────────────────────────
echo "[1/5] Updating config to use sp500_tickers.txt (503 companies)..."
# Backup current config
cp conf/config.yaml conf/config.yaml.bak

# Update tickers_file to full list
python3 -c "
import re
with open('conf/config.yaml') as f:
    content = f.read()
content = re.sub(r'tickers_file:.*', 'tickers_file: \"sp500_tickers.txt\"', content)
with open('conf/config.yaml', 'w') as f:
    f.write(content)
print('Config updated to use sp500_tickers.txt')
"

# Count how many we already have
ALREADY_DONE=$(ls data/raw/sec_filings/*.parquet 2>/dev/null | wc -l | tr -d ' ')
TOTAL=$(wc -l < sp500_tickers.txt | tr -d ' ')
REMAINING=$((TOTAL - ALREADY_DONE))
echo "Already downloaded: $ALREADY_DONE / $TOTAL companies"
echo "Remaining: $REMAINING companies to download"
echo ""

# ── Step 2: Write checkpointed sec_loader ─────────────────────────────────────
echo "[2/5] Writing checkpointed sec_loader.py..."

cat > src/dataloader/sec_loader.py << 'ENDOFFILE'
"""
sec_loader.py - with full checkpointing for 500-company scale runs.

Checkpointing: if data/raw/sec_filings/{TICKER}_filings.parquet exists,
that ticker is skipped entirely. Safe to interrupt and resume.

Progress tracking: writes progress.json with completed/failed/remaining.
"""
import os, logging, json
from typing import Optional
import hydra
import pandas as pd
from edgar import Company, set_identity
from edgar import httpclient
from edgar.entity.filings import EntityFilings
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _get_company_filings(cik_or_name, form, start, end):
    try:
        company = Company(cik_or_name)
        filings = company.get_filings(form=form, filing_date=f"{start}:{end}")
        if filings is None or len(filings) == 0:
            logger.warning("No filings found for %s", cik_or_name)
            return None
        return filings
    except Exception as exc:
        logger.error("Failed to fetch filings for %s: %s", cik_or_name, exc)
        return None


def _extract_section_text(doc, section_name):
    try:
        section = doc.get_section(section_name)
        if section:
            return section.text(clean=True, table_max_col_width=500)
    except Exception as exc:
        logger.warning("Failed to extract %s: %s", section_name, exc)
    return None


def _configure_cache(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    def get_cache_directory():
        return cache_dir
    httpclient.get_cache_directory = get_cache_directory


def _get_document_entity(filing):
    doc = filing.obj()
    return doc.document


def _extract_with_fallback(filing, sections):
    try:
        doc = _get_document_entity(filing)
        section_texts = {}
        if hasattr(doc, "get_section"):
            for section in sections:
                section_texts[f"section_{section}"] = _extract_section_text(doc, section)
        if not any(section_texts.values()):
            try:
                full_text = doc.text(clean=True, include_tables=False, table_max_col_width=200)
                for section in sections:
                    keywords = ["ITEM 1A", "Item 1A", "RISK FACTORS", "Risk Factors"]
                    start_idx = -1
                    for kw in keywords:
                        idx = full_text.find(kw)
                        if idx != -1:
                            start_idx = idx
                            break
                    if start_idx != -1:
                        section_texts[f"section_{section}"] = full_text[start_idx:start_idx+15000]
                    else:
                        section_texts[f"section_{section}"] = None
            except Exception as exc:
                logger.warning("Fallback failed: %s", exc)
        return section_texts
    except Exception as exc:
        logger.error("Document extraction failed: %s", exc)
        return {f"section_{s}": None for s in sections}


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cache_dir = os.path.join(cfg.data.raw_dir, "sec_cache")
    _configure_cache(cache_dir)
    set_identity(cfg.sec.identity)

    with open(cfg.data.tickers_file) as f:
        tickers = [l.strip() for l in f if l.strip()]

    raw_dir = os.path.join(cfg.data.raw_dir, "sec_filings")
    os.makedirs(raw_dir, exist_ok=True)

    progress_file = os.path.join(raw_dir, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            progress = json.load(f)
    else:
        progress = {"completed": [], "failed": [], "no_text": []}

    forms = cfg.sec.forms
    sections = cfg.sec.sections

    total = len(tickers)
    for i, ticker in enumerate(tickers):
        output_path = os.path.join(raw_dir, f"{ticker}_filings.parquet")

        # ── Checkpointing: skip already downloaded ─────────────────────
        if os.path.exists(output_path):
            if ticker not in progress["completed"]:
                progress["completed"].append(ticker)
            continue

        done = len(progress["completed"])
        remaining = total - done - 1
        logger.info(f"[{i+1}/{total}] {ticker} | done={done} remaining={remaining}")

        records = []
        for form in forms:
            filings = _get_company_filings(
                ticker, form, cfg.data.start_date, cfg.data.end_date)
            if filings is None:
                continue
            for filing in filings:
                try:
                    section_texts = _extract_with_fallback(filing, sections)
                    record = {
                        "ticker": ticker,
                        "cik": getattr(filing, "cik", None),
                        "company": getattr(filing, "company", ticker),
                        "accession_number": getattr(filing, "accession_number", None),
                        "form_type": getattr(filing, "form", form),
                        "filed_at": str(getattr(filing, "filing_date", "") or ""),
                        "period_of_report": str(getattr(filing, "period_of_report", "") or ""),
                        "filing_url": getattr(filing, "url", None),
                    }
                    for section in sections:
                        record[f"section_{section}"] = section_texts.get(
                            f"section_{section}", None)
                    records.append(record)
                except Exception as exc:
                    logger.error("Failed filing for %s: %s", ticker, exc)

        if records:
            df_out = pd.DataFrame(records)
            df_out.to_parquet(output_path, index=False)
            text_col = f"section_{sections[0]}"
            has_text = df_out[text_col].notna().sum()
            logger.info(f"  Saved {len(records)} filings ({has_text} with text)")
            progress["completed"].append(ticker)
            if has_text == 0:
                progress["no_text"].append(ticker)
        else:
            logger.warning(f"  No filings found for {ticker}")
            progress["failed"].append(ticker)

        # Save progress after each ticker
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)

    # Final summary
    total_done = len(progress["completed"])
    total_failed = len(progress["failed"])
    total_no_text = len(progress["no_text"])
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETED: {total_done}/{total}")
    logger.info(f"Failed (no filings): {total_failed} -> {progress['failed'][:10]}")
    logger.info(f"No text extracted:   {total_no_text} -> {progress['no_text'][:10]}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
ENDOFFILE

echo "sec_loader.py written."

# ── Step 3: Write optimized filing_dataset for 500 companies ──────────────────
echo "[3/5] Writing optimized filing_dataset.py..."

cat > src/dataloader/filing_dataset.py << 'ENDOFFILE'
"""
filing_dataset.py v4 - optimized for 500-company scale

Key optimizations:
- Batch yfinance download (all tickers at once, much faster)
- Progress checkpointing per ticker
- Robust error handling for edge cases at scale
- All features from v3 preserved
"""
import os, logging
import numpy as np
import pandas as pd
import yfinance as yf
import hydra
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.features.lm_features import add_lm_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SECTOR_MAP = {
    "AAPL":"Technology","MSFT":"Technology","GOOGL":"Technology","NVDA":"Technology",
    "META":"Technology","TSLA":"Technology","CSCO":"Technology","QCOM":"Technology",
    "INTC":"Technology","IBM":"Technology","ORCL":"Technology","ADBE":"Technology",
    "CRM":"Technology","AMD":"Technology","AMAT":"Technology","MU":"Technology",
    "KLAC":"Technology","LRCX":"Technology","SNPS":"Technology","CDNS":"Technology",
    "JPM":"Finance","BAC":"Finance","GS":"Finance","WFC":"Finance","MS":"Finance",
    "BLK":"Finance","AXP":"Finance","SPGI":"Finance","CB":"Finance","PGR":"Finance",
    "V":"Finance","MA":"Finance","BRK-B":"Finance","C":"Finance","USB":"Finance",
    "TFC":"Finance","PNC":"Finance","COF":"Finance","AFL":"Finance","AIG":"Finance",
    "JNJ":"Healthcare","UNH":"Healthcare","PFE":"Healthcare","ABBV":"Healthcare",
    "MRK":"Healthcare","TMO":"Healthcare","ABT":"Healthcare","DHR":"Healthcare",
    "BMY":"Healthcare","AMGN":"Healthcare","LLY":"Healthcare","MDT":"Healthcare",
    "ISRG":"Healthcare","SYK":"Healthcare","BSX":"Healthcare","ZBH":"Healthcare",
    "AMZN":"Consumer","WMT":"Consumer","HD":"Consumer","MCD":"Consumer","SBUX":"Consumer",
    "PG":"Consumer","KO":"Consumer","PEP":"Consumer","COST":"Consumer","NKE":"Consumer",
    "TGT":"Consumer","LOW":"Consumer","BKNG":"Consumer","MAR":"Consumer","HLT":"Consumer",
    "CVX":"Energy","XOM":"Energy","COP":"Energy","SLB":"Energy","EOG":"Energy",
    "PSX":"Energy","VLO":"Energy","MPC":"Energy","OXY":"Energy","DVN":"Energy",
    "CAT":"Industrial","UPS":"Industrial","RTX":"Industrial","NEE":"Industrial",
    "LIN":"Industrial","BA":"Industrial","DE":"Industrial","MMM":"Industrial",
    "HON":"Industrial","GE":"Industrial","LMT":"Industrial","NOC":"Industrial",
    "EMR":"Industrial","ETN":"Industrial","PH":"Industrial","ROK":"Industrial",
}


def cosine_sim_consecutive(texts):
    sims = [np.nan]
    tl = texts.fillna("").tolist()
    for i in range(1, len(tl)):
        p, c = tl[i-1], tl[i]
        if not p.strip() or not c.strip():
            sims.append(np.nan); continue
        try:
            v = TfidfVectorizer(max_features=5000, stop_words="english")
            t = v.fit_transform([p, c])
            sims.append(float(cosine_similarity(t[0], t[1])[0][0]))
        except:
            sims.append(np.nan)
    return pd.Series(sims, index=texts.index)


def compute_risk_drift_4q(s):
    return s.rolling(window=4, min_periods=2).mean() - s.mean()

def compute_filing_surprise(s):
    return (s - s.expanding(min_periods=2).mean()) / s.expanding(min_periods=2).std().replace(0, np.nan)

def compute_sector_contagion(df, window_days=45):
    df = df.copy()
    df["sector"] = df["ticker"].map(SECTOR_MAP).fillna("Other")
    df["filed_at"] = pd.to_datetime(df["filed_at"])
    contagion = []
    for idx, row in df.iterrows():
        s = row["sector"]
        if s == "Other" or pd.isna(s): contagion.append(np.nan); continue
        mask = ((df["sector"]==s) & (df["ticker"]!=row["ticker"]) &
                (df["filed_at"] >= row["filed_at"]-pd.Timedelta(days=window_days)) &
                (df["filed_at"] <= row["filed_at"]+pd.Timedelta(days=window_days)))
        peers = df.loc[mask, "cosine_sim_prev"].dropna()
        contagion.append(float(peers.mean()) if len(peers) >= 2 else np.nan)
    return pd.Series(contagion, index=df.index)

def calculate_rsi(prices, window=14):
    if len(prices) < window+1: return np.nan
    d = prices.diff()
    g = d.where(d>0,0).rolling(window).mean()
    l = (-d.where(d<0,0)).rolling(window).mean()
    rs = g / l.replace(0, np.nan)
    return float((100 - 100/(1+rs)).iloc[-1])

def get_price_features(filing_date, closes, lookback=50):
    empty = {k: np.nan for k in ["price_return_1d","price_return_5d","price_return_20d",
                                   "price_volatility_20d","price_ma_ratio_5","price_ma_ratio_20","price_rsi"]}
    try:
        prior = closes[closes.index < filing_date]
        if len(prior) < lookback: return empty
        w = prior.iloc[-lookback:]
        return {
            "price_return_1d":    float((w.iloc[-1]-w.iloc[-2])/w.iloc[-2]) if len(w)>=2 else np.nan,
            "price_return_5d":    float((w.iloc[-1]-w.iloc[-6])/w.iloc[-6]) if len(w)>=6 else np.nan,
            "price_return_20d":   float((w.iloc[-1]-w.iloc[-21])/w.iloc[-21]) if len(w)>=21 else np.nan,
            "price_volatility_20d": float(w.pct_change().dropna().iloc[-20:].std()*np.sqrt(252)) if len(w)>=21 else np.nan,
            "price_ma_ratio_5":   float(w.iloc[-1]/w.iloc[-5:].mean()) if len(w)>=5 else np.nan,
            "price_ma_ratio_20":  float(w.iloc[-1]/w.iloc[-20:].mean()) if len(w)>=20 else np.nan,
            "price_rsi":          calculate_rsi(w),
        }
    except: return empty

def get_abnormal_return(filing_date, offset, horizon, closes, spy_closes):
    try:
        fut_c = closes[closes.index >= filing_date]
        if len(fut_c) < offset+horizon+1: return np.nan, np.nan, np.nan
        p0, p1 = fut_c.iloc[offset], fut_c.iloc[offset+horizon]
        sr = (p1-p0)/p0
        if spy_closes is not None:
            fut_m = spy_closes[spy_closes.index >= filing_date]
            if len(fut_m) < offset+horizon+1: return float(sr), np.nan, float(sr)
            m0, m1 = fut_m.iloc[offset], fut_m.iloc[offset+horizon]
            mr = (m1-m0)/m0
            return float(sr), float(mr), float(sr-mr)
        return float(sr), np.nan, float(sr)
    except: return np.nan, np.nan, np.nan

def impute_cols(df, cols):
    df = df.copy()
    for col in cols:
        if col not in df.columns: continue
        cm = df.groupby("ticker")[col].transform("median")
        df[col] = df[col].fillna(cm).fillna(df[col].median()).fillna(0.0)
    return df


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    with open(cfg.data.tickers_file) as f:
        tickers = [l.strip() for l in f if l.strip()]
    logger.info(f"Loaded {len(tickers)} tickers")

    sec_dir = os.path.join(cfg.data.raw_dir, "sec_filings")
    processed_dir = cfg.data.processed_dir
    text_col = f"section_{cfg.sec.sections[0]}"
    os.makedirs(processed_dir, exist_ok=True)

    # Filter to tickers with downloaded filings
    available = [t for t in tickers
                 if os.path.exists(os.path.join(sec_dir, f"{t}_filings.parquet"))]
    logger.info(f"Tickers with SEC filings: {len(available)}/{len(tickers)}")

    # Download SPY benchmark
    logger.info("Downloading SPY benchmark...")
    try:
        spy_raw = yf.download("SPY", start=cfg.data.start_date,
                               end=cfg.data.end_date, auto_adjust=True, progress=False)
        if isinstance(spy_raw.columns, pd.MultiIndex):
            spy_raw.columns = spy_raw.columns.get_level_values(0)
        spy_closes = spy_raw["Close"].dropna().sort_index()
        spy_closes.index = pd.to_datetime(spy_closes.index)
        logger.info(f"SPY: {len(spy_closes)} trading days")
    except Exception as e:
        logger.warning(f"SPY failed: {e}"); spy_closes = None

    # Batch download all stock prices at once (much faster than one-by-one)
    logger.info(f"Batch downloading price data for {len(available)} tickers...")
    try:
        batch_size = 100
        all_prices = {}
        for i in range(0, len(available), batch_size):
            batch = available[i:i+batch_size]
            logger.info(f"  Price batch {i//batch_size+1}: {batch[0]}...{batch[-1]}")
            raw = yf.download(batch, start=cfg.data.start_date,
                               end=cfg.data.end_date, auto_adjust=True,
                               progress=False, group_by="ticker")
            if isinstance(raw.columns, pd.MultiIndex):
                for t in batch:
                    try:
                        tc = raw[t]["Close"].dropna().sort_index()
                        tc.index = pd.to_datetime(tc.index)
                        if len(tc) > 100:
                            all_prices[t] = tc
                    except: pass
            else:
                # Single ticker in batch
                if len(batch) == 1:
                    tc = raw["Close"].dropna().sort_index()
                    tc.index = pd.to_datetime(tc.index)
                    if len(tc) > 100:
                        all_prices[batch[0]] = tc
        logger.info(f"Price data loaded for {len(all_prices)} tickers")
    except Exception as e:
        logger.error(f"Batch price download failed: {e}. Falling back to individual downloads.")
        all_prices = {}

    all_records, skipped = [], []

    for i, ticker in enumerate(available):
        if i % 50 == 0:
            logger.info(f"Processing {i}/{len(available)} ({ticker})...")

        pp = os.path.join(sec_dir, f"{ticker}_filings.parquet")
        fdf = pd.read_parquet(pp)
        fdf["filed_at"] = pd.to_datetime(fdf["filed_at"])
        fdf = fdf.sort_values("filed_at").reset_index(drop=True)
        if fdf[text_col].notna().sum() == 0: skipped.append(ticker); continue

        # Get price data
        if ticker in all_prices:
            closes = all_prices[ticker]
        else:
            try:
                raw = yf.download(ticker, start=cfg.data.start_date,
                                   end=cfg.data.end_date, auto_adjust=True, progress=False)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                closes = raw["Close"].dropna().sort_index()
                closes.index = pd.to_datetime(closes.index)
                if len(closes) < 100: skipped.append(ticker); continue
            except: skipped.append(ticker); continue

        # Text features
        fdf["cosine_sim_prev"] = cosine_sim_consecutive(fdf[text_col])
        fdf["risk_drift_4q"]   = compute_risk_drift_4q(fdf["cosine_sim_prev"])
        fdf["filing_surprise"] = compute_filing_surprise(fdf["cosine_sim_prev"])
        ml = fdf[text_col].apply(lambda x: len(x.split()) if isinstance(x,str) else np.nan).mean()
        fdf["text_length_norm"] = fdf[text_col].apply(
            lambda x: len(x.split()) if isinstance(x,str) else np.nan) / (ml if ml>0 else 1)

        # LM sentiment
        fdf = add_lm_features(fdf, text_col)

        # Price features
        pf = pd.DataFrame(fdf["filed_at"].apply(
            lambda d: get_price_features(d, closes)).tolist())
        fdf = pd.concat([fdf.reset_index(drop=True), pf.reset_index(drop=True)], axis=1)

        # Multi-horizon abnormal returns from t+1
        for h in [5, 10, 20]:
            rets = [get_abnormal_return(d, 1, h, closes, spy_closes)
                    for d in fdf["filed_at"]]
            fdf[f"stock_ret_{h}d"]    = [r[0] for r in rets]
            fdf[f"market_ret_{h}d"]   = [r[1] for r in rets]
            fdf[f"abnormal_ret_{h}d"] = [r[2] for r in rets]
            fdf[f"target_{h}d"]       = (pd.Series([r[2] for r in rets]) > 0).astype(int).values
        fdf["target"] = fdf["target_5d"]

        # Interaction features
        fdf["lm_neg_x_cosine"]       = fdf["lm_negative"] * (1 - fdf["cosine_sim_prev"].fillna(0.9))
        fdf["lm_unc_x_drift"]        = fdf["lm_uncertainty"] * fdf["risk_drift_4q"].fillna(0)
        fdf["text_price_divergence"] = fdf["lm_net_sentiment"].fillna(0) * (-fdf["price_return_20d"].fillna(0))

        cols = ["filed_at","ticker","cosine_sim_prev","risk_drift_4q","filing_surprise",
                "sector_contagion","lm_negative","lm_positive","lm_uncertainty","lm_litigious",
                "lm_constraining","lm_net_sentiment","lm_negative_delta","lm_positive_delta",
                "lm_uncertainty_delta","lm_litigious_delta","lm_neg_x_cosine","lm_unc_x_drift",
                "text_price_divergence","text_length_norm","price_return_1d","price_return_5d",
                "price_return_20d","price_volatility_20d","price_ma_ratio_5","price_ma_ratio_20",
                "price_rsi","abnormal_ret_5d","abnormal_ret_10d","abnormal_ret_20d",
                "target_5d","target_10d","target_20d","target"]
        avail_cols = [c for c in cols if c in fdf.columns]
        rec = fdf[avail_cols].dropna(subset=["target","cosine_sim_prev"])
        all_records.append(rec)

    logger.info(f"Processed {len(all_records)} tickers successfully, skipped {len(skipped)}")

    final_df = pd.concat(all_records, ignore_index=True).sort_values("filed_at").reset_index(drop=True)

    # Impute temporal features
    impute_temporal = ["risk_drift_4q","filing_surprise","lm_negative_delta",
                       "lm_positive_delta","lm_uncertainty_delta","lm_litigious_delta"]
    final_df = impute_cols(final_df, impute_temporal)

    # Sector contagion (computed across all companies)
    logger.info("Computing sector contagion signal...")
    final_df["sector_contagion"] = compute_sector_contagion(final_df)
    sec_med = final_df.groupby(final_df["ticker"].map(SECTOR_MAP).fillna("Other"))["sector_contagion"].transform("median")
    final_df["sector_contagion"] = final_df["sector_contagion"].fillna(sec_med).fillna(
        final_df["sector_contagion"].median())

    out = os.path.join(processed_dir, "filing_aligned.csv")
    final_df.to_csv(out, index=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"DATASET COMPLETE")
    logger.info(f"Total rows:       {len(final_df)}")
    logger.info(f"Companies:        {final_df['ticker'].nunique()}")
    logger.info(f"Date range:       {final_df['filed_at'].min().date()} to {final_df['filed_at'].max().date()}")
    logger.info(f"Skipped:          {len(skipped)}")
    train_n = (final_df["filed_at"] < pd.Timestamp("2023-01-01")).sum()
    test_n  = (final_df["filed_at"] >= pd.Timestamp("2023-01-01")).sum()
    logger.info(f"Train (<2023):    {train_n}")
    logger.info(f"Test  (>=2023):   {test_n}")
    for h in [5,10,20]:
        col = f"target_{h}d"
        if col in final_df.columns:
            vc = final_df[col].value_counts().to_dict()
            logger.info(f"target_{h}d:      {vc}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()
ENDOFFILE

echo "filing_dataset.py written."

# ── Step 4: Show current progress and start SEC download ─────────────────────
echo ""
echo "[4/5] Checking existing downloads..."
ALREADY_DONE=$(ls data/raw/sec_filings/*.parquet 2>/dev/null | grep -v progress | wc -l | tr -d ' ')
TOTAL=$(wc -l < sp500_tickers.txt | tr -d ' ')
echo "Already have: $ALREADY_DONE / $TOTAL companies"
echo ""
echo "Starting SEC download for remaining $(($TOTAL - $ALREADY_DONE)) companies..."
echo "This will take 8-12 hours. Do NOT close this terminal."
echo "Safe to interrupt (Ctrl+C) and resume by running this script again."
echo ""

venv/bin/python -m src.dataloader.sec_loader

echo ""
echo "[5/5] Running full pipeline on all companies..."
echo ""
echo "--- Building filing-aligned dataset ---"
venv/bin/python -m src.dataloader.filing_dataset

echo ""
echo "--- Running EDA ---"
venv/bin/python -m src.analysis.eda

echo ""
echo "--- Running baseline model ---"
venv/bin/python -m src.models.baseline

echo ""
echo "--- Running final analysis ---"
venv/bin/python -m src.models.final_analysis

echo ""
echo "=========================================="
echo " ALL DONE!"
echo " Finished: $(date)"
echo "=========================================="

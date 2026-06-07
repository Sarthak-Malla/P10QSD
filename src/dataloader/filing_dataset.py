"""
filing_dataset.py v4

Key fixes vs v3:
1. Sector contagion look-ahead bias removed — upper bound is now row["filed_at"]
   (previously included filings up to 45 days AFTER the current filing).

2. filing_day_return added — raw stock return on t=0 vs t=-1 as earnings-surprise proxy.
   Without this control we cannot isolate the text contribution from EPS surprise.

3. MD&A section features — cosine_sim_prev_mda and LM scores with _mda suffix,
   enabling ablation: Item 1A only vs MD&A only vs both.

4. FinBERT cosine similarity — deep-semantic similarity via ProsusAI/finbert embeddings,
   cached per-ticker to data/raw/finbert_cache/.

5. Multi-horizon targets (5d, 10d, 20d) retained from v3.
6. Interaction features retained from v3.
"""
import os, logging
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import yfinance as yf
import hydra
from tqdm.auto import tqdm
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.features.lm_features import add_lm_features, compute_lm_scores
from src.dataloader.company_metadata import get_ticker_metadata, get_ticker_sector
from src.dataloader.sector_benchmarks import fetch_all_benchmark_prices, get_benchmark_df_for_ticker

class TqdmLoggingHandler(logging.Handler):
    """Write log lines without breaking tqdm progress bars."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record))
        except Exception:
            print(self.format(record))


def configure_logging():
    root = logging.getLogger()
    root.handlers.clear()
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


configure_logging()
logger = logging.getLogger(__name__)


SECTION_TITLES = {
    "part_i_item_1": "Part I, Item 1 - Financial Statements",
    "part_i_item_2": "Part I, Item 2 - MD&A",
    "part_i_item_3": "Part I, Item 3 - Market Risk",
    "part_i_item_4": "Part I, Item 4 - Controls and Procedures",
    "part_ii_item_1": "Part II, Item 1 - Legal Proceedings",
    "part_ii_item_1a": "Part II, Item 1A - Risk Factors",
    "part_ii_item_2": "Part II, Item 2 - Equity Securities",
    "part_ii_item_3": "Part II, Item 3 - Senior Securities Defaults",
    "part_ii_item_4": "Part II, Item 4 - Mine Safety Disclosures",
    "part_ii_item_5": "Part II, Item 5 - Other Information",
    "part_ii_item_6": "Part II, Item 6 - Exhibits",
}


def section_title(section_key):
    return SECTION_TITLES.get(section_key, section_key.replace("_", " ").title())

def cosine_sim_consecutive(texts):
    sims = [np.nan]
    tl = texts.fillna("").tolist()
    for i in range(1, len(tl)):
        p, c = tl[i-1], tl[i]
        if not p.strip() or not c.strip(): sims.append(np.nan); continue
        try:
            v = TfidfVectorizer(max_features=5000, stop_words="english")
            t = v.fit_transform([p, c])
            sims.append(float(cosine_similarity(t[0], t[1])[0][0]))
        except: sims.append(np.nan)
    return pd.Series(sims, index=texts.index)

def compute_risk_drift_4q(cosine_sims):
    return cosine_sims.rolling(window=4, min_periods=2).mean() - cosine_sims.mean()

def compute_filing_surprise(cosine_sims):
    em = cosine_sims.expanding(min_periods=2).mean()
    es = cosine_sims.expanding(min_periods=2).std().replace(0, np.nan)
    return (cosine_sims - em) / es

def compute_sector_contagion(
    df,
    window_days=45,
    metadata_path="data/reference/sp500_constituents.csv",
    show_progress=False,
):
    """
    Average cosine_sim_prev of sector peers whose filings arrived within
    the past `window_days` days (look-back only — no look-ahead bias).
    """
    df = df.copy()
    if "sector" in df.columns:
        missing = df["sector"].isna() | (df["sector"] == "")
        df.loc[missing, "sector"] = df.loc[missing, "ticker"].apply(
            lambda t: get_ticker_sector(t, metadata_path=metadata_path)
        )
    else:
        df["sector"] = df["ticker"].apply(lambda t: get_ticker_sector(t, metadata_path=metadata_path))
    df["filed_at"] = pd.to_datetime(df["filed_at"])
    contagion = []
    rows = df.iterrows()
    if show_progress:
        rows = tqdm(rows, total=len(df), desc="sector contagion", unit="filing", dynamic_ncols=True)

    for idx, row in rows:
        s = row["sector"]
        if pd.isna(s): contagion.append(np.nan); continue
        mask = ((df["sector"] == s) & (df["ticker"] != row["ticker"]) &
                (df["filed_at"] >= row["filed_at"] - pd.Timedelta(days=window_days)) &
                (df["filed_at"] <= row["filed_at"]))          # FIX: no forward look
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

def get_price_features(filing_date, price_df, lookback=50):
    empty = {k: np.nan for k in [
        "price_return_1d","price_return_5d","price_return_20d",
        "price_volatility_20d","price_ma_ratio_5","price_ma_ratio_20",
        "price_rsi","filing_day_return",
    ]}
    try:
        c = price_df["Close"].dropna()
        c.index = pd.to_datetime(c.index); c = c.sort_index()
        prior = c[c.index < filing_date]
        if len(prior) < lookback: return empty
        w = prior.iloc[-lookback:]

        # filing_day_return: t=0 close vs t=-1 close (earnings-surprise proxy)
        on_or_after = c[c.index >= filing_date]
        if len(on_or_after) >= 1 and len(prior) >= 1:
            filing_day_return = float((on_or_after.iloc[0] - prior.iloc[-1]) / prior.iloc[-1])
        else:
            filing_day_return = np.nan

        return {
            "price_return_1d":      float((w.iloc[-1]-w.iloc[-2])/w.iloc[-2]) if len(w)>=2 else np.nan,
            "price_return_5d":      float((w.iloc[-1]-w.iloc[-6])/w.iloc[-6]) if len(w)>=6 else np.nan,
            "price_return_20d":     float((w.iloc[-1]-w.iloc[-21])/w.iloc[-21]) if len(w)>=21 else np.nan,
            "price_volatility_20d": float(w.pct_change().dropna().iloc[-20:].std()*np.sqrt(252)) if len(w)>=21 else np.nan,
            "price_ma_ratio_5":     float(w.iloc[-1]/w.iloc[-5:].mean()) if len(w)>=5 else np.nan,
            "price_ma_ratio_20":    float(w.iloc[-1]/w.iloc[-20:].mean()) if len(w)>=20 else np.nan,
            "price_rsi":            calculate_rsi(w),
            "filing_day_return":    filing_day_return,
        }
    except: return empty

def get_abnormal_return(filing_date, start_offset, horizon, price_df, market_df):
    """
    Compute abnormal return from t+start_offset to t+start_offset+horizon.
    start_offset=1 skips the filing day (t=0) noise from algo traders.
    """
    try:
        c = price_df["Close"].dropna().sort_index()
        c.index = pd.to_datetime(c.index)
        m = None
        if market_df is not None:
            m = market_df["Close"].dropna().sort_index()
            m.index = pd.to_datetime(m.index)

        fut_c = c[c.index >= filing_date]
        if len(fut_c) < start_offset + horizon + 1: return np.nan, np.nan, np.nan

        p_start = fut_c.iloc[start_offset]
        p_end   = fut_c.iloc[start_offset + horizon]
        stock_ret = (p_end - p_start) / p_start

        if m is not None:
            fut_m = m[m.index >= filing_date]
            if len(fut_m) < start_offset + horizon + 1:
                return float(stock_ret), np.nan, float(stock_ret)
            m_start = fut_m.iloc[start_offset]
            m_end   = fut_m.iloc[start_offset + horizon]
            market_ret = (m_end - m_start) / m_start
            abnormal_ret = stock_ret - market_ret
            return float(stock_ret), float(market_ret), float(abnormal_ret)
        return float(stock_ret), np.nan, float(stock_ret)
    except: return np.nan, np.nan, np.nan

def _add_lm_features_suffixed(df, text_col, suffix):
    """Compute LM scores for `text_col` and add columns with `suffix`."""
    lm_rows = df[text_col].apply(compute_lm_scores)
    lm_df = pd.DataFrame(lm_rows.tolist())
    rename_map = {c: c + suffix for c in lm_df.columns if c != "lm_word_count"}
    lm_df = lm_df.rename(columns=rename_map).drop(columns=["lm_word_count"], errors="ignore")
    return pd.concat([df.reset_index(drop=True), lm_df.reset_index(drop=True)], axis=1)

def impute_cols(df, cols):
    df = df.copy()
    for col in cols:
        if col not in df.columns: continue
        cm = df.groupby("ticker")[col].transform("median")
        gm = df[col].median()
        df[col] = df[col].fillna(cm).fillna(gm).fillna(0.0)
    return df


def _cfg_get(cfg_obj, key, default=None):
    if hasattr(cfg_obj, "get"):
        return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)


def resolve_section_columns(sec_cfg):
    sections = list(_cfg_get(sec_cfg, "sections", []))
    primary_default = sections[0] if sections else "part_ii_item_1a"
    primary_section = _cfg_get(sec_cfg, "primary_section", primary_default)
    mda_section = _cfg_get(sec_cfg, "mda_section", "part_i_item_2")
    text_col = f"section_{primary_section}"
    mda_col = f"section_{mda_section}" if mda_section else None
    return text_col, mda_col


def configured_section_columns(sec_cfg):
    sections = list(_cfg_get(sec_cfg, "sections", []))
    return [(section, f"section_{section}", section_title(section)) for section in sections]


def update_section_stats(fdf, section_columns, section_stats):
    """Track how often each configured SEC section exists in raw filing rows."""
    filing_count = len(fdf)
    for _, column, title in section_columns:
        if column in fdf.columns:
            values = fdf[column].replace("", np.nan)
            available = int(values.notna().sum())
            missing = int(filing_count - available)
        else:
            available = 0
            missing = filing_count

        section_stats[title]["available_filings"] += available
        section_stats[title]["missing_filings"] += missing
        if available > 0:
            section_stats[title]["tickers_with_any"] += 1
        else:
            section_stats[title]["tickers_missing_all"] += 1

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    with open(cfg.data.tickers_file) as f:
        tickers = [l.strip() for l in f if l.strip()]
    logger.info(f"Loaded {len(tickers)} tickers")

    sec_dir = os.path.join(cfg.data.raw_dir, "sec_filings")
    processed_dir = cfg.data.processed_dir
    metadata_path = cfg.data.get("metadata_file", "data/reference/sp500_constituents.csv")
    text_col, mda_col = resolve_section_columns(cfg.sec)
    section_columns = configured_section_columns(cfg.sec)
    primary_section = text_col.replace("section_", "", 1)
    mda_section = mda_col.replace("section_", "", 1) if mda_col else None
    primary_title = section_title(primary_section)
    mda_title = section_title(mda_section) if mda_section else "disabled"
    os.makedirs(processed_dir, exist_ok=True)

    # Try to import FinBERT features (optional — degrades gracefully if not installed)
    try:
        from src.features.finbert_features import compute_finbert_similarity
        finbert_available = True
    except Exception as e:
        logger.warning(f"FinBERT not available: {e}")
        finbert_available = False

    cache_dir = os.path.join(cfg.data.raw_dir, "finbert_cache")

    logger.info(
        "Building filing dataset for %d tickers | primary=%s | mda=%s",
        len(tickers), primary_title, mda_title,
    )
    logger.info(f"Using company metadata from {metadata_path}")
    logger.info(
        "Tracking section availability for: %s",
        ", ".join(title for _, _, title in section_columns) if section_columns else "none",
    )

    logger.info("Downloading sector ETF benchmarks (SPY + 11 sector ETFs)...")
    try:
        etf_dfs = fetch_all_benchmark_prices(cfg.data.start_date, cfg.data.end_date)
        spy_df = etf_dfs.get("SPY")  # kept as fallback
    except Exception as e:
        logger.warning(f"Benchmark download failed: {e}")
        etf_dfs = {}
        spy_df = None

    all_records, skipped = [], []
    skip_reasons = Counter()
    skip_details = defaultdict(list)
    section_stats = defaultdict(Counter)
    skipped_missing_metadata = []

    def record_skip(ticker, reason, detail=None):
        skipped.append(ticker)
        skip_reasons[reason] += 1
        skip_details[reason].append(f"{ticker}: {detail}" if detail else ticker)

    success_tickers = 0
    total_rows = 0

    with tqdm(tickers, desc="filing dataset", unit="ticker", dynamic_ncols=True) as progress:
        for ticker in progress:
            last_status = "start"
            try:
                metadata = get_ticker_metadata(ticker, metadata_path=metadata_path)
                if metadata["gics_sector"] == "Unknown" or metadata["sector_etf"] == "SPY":
                    record_skip(ticker, "missing_sector_metadata")
                    skipped_missing_metadata.append(ticker)
                    last_status = "skip:metadata"
                    continue

                pp = os.path.join(sec_dir, f"{ticker}_filings.parquet")
                if not os.path.exists(pp):
                    record_skip(ticker, "missing_sec_parquet")
                    last_status = "skip:no_sec_file"
                    continue

                fdf = pd.read_parquet(pp)
                fdf["filed_at"] = pd.to_datetime(fdf["filed_at"])
                fdf = fdf.sort_values("filed_at").reset_index(drop=True)
                update_section_stats(fdf, section_columns, section_stats)

                # Normalise empty strings to NaN so downstream checks work uniformly.
                if text_col not in fdf.columns:
                    record_skip(ticker, "missing_primary_section_column", text_col)
                    last_status = "skip:primary_col"
                    continue
                fdf[text_col] = fdf[text_col].replace("", np.nan)
                if mda_col and mda_col in fdf.columns:
                    fdf[mda_col] = fdf[mda_col].replace("", np.nan)

                # Drop individual filings where the primary text is missing.
                fdf = fdf[fdf[text_col].notna()].reset_index(drop=True)
                if len(fdf) == 0:
                    record_skip(ticker, "no_primary_section_text", primary_title)
                    last_status = "skip:primary_text"
                    continue

                try:
                    pdf = yf.download(
                        ticker,
                        start=cfg.data.start_date,
                        end=cfg.data.end_date,
                        auto_adjust=True,
                        progress=False,
                    )
                    if pdf.empty:
                        record_skip(ticker, "empty_price_data")
                        last_status = "skip:no_prices"
                        continue
                except Exception as e:
                    record_skip(ticker, "price_download_failed", str(e))
                    last_status = "skip:price_error"
                    continue
                if isinstance(pdf.columns, pd.MultiIndex):
                    pdf.columns = pdf.columns.get_level_values(0)

                # Item 1A / primary section text features.
                fdf["cosine_sim_prev"] = cosine_sim_consecutive(fdf[text_col])
                fdf["risk_drift_4q"] = compute_risk_drift_4q(fdf["cosine_sim_prev"])
                fdf["filing_surprise"] = compute_filing_surprise(fdf["cosine_sim_prev"])
                ml = fdf[text_col].apply(lambda x: len(x.split()) if isinstance(x, str) else np.nan).mean()
                fdf["text_length_norm"] = fdf[text_col].apply(
                    lambda x: len(x.split()) if isinstance(x, str) else np.nan
                ) / (ml if ml > 0 else 1)

                fdf = add_lm_features(fdf, text_col)

                # MD&A / configured secondary section text features.
                if mda_col and mda_col in fdf.columns and fdf[mda_col].notna().sum() > 0:
                    fdf["cosine_sim_prev_mda"] = cosine_sim_consecutive(fdf[mda_col])
                    fdf = _add_lm_features_suffixed(fdf, mda_col, "_mda")

                # FinBERT cosine similarity.
                if finbert_available:
                    try:
                        fdf["finbert_cosine_sim"] = compute_finbert_similarity(
                            fdf, text_col, ticker, cache_dir=cache_dir)
                    except Exception as e:
                        logger.warning(f"{ticker}: FinBERT failed: {e}")
                        fdf["finbert_cosine_sim"] = np.nan

                # Price features, including filing_day_return.
                pf = pd.DataFrame(fdf["filed_at"].apply(
                    lambda d: get_price_features(d, pdf)).tolist())
                fdf = pd.concat([fdf.reset_index(drop=True), pf.reset_index(drop=True)], axis=1)

                # Multi-horizon abnormal returns from t+1, skipping filing day.
                sector = metadata["gics_sector"]
                benchmark_etf = metadata["sector_etf"]
                ticker_benchmark = (
                    get_benchmark_df_for_ticker(ticker, etf_dfs, metadata_path=metadata_path)
                    if etf_dfs else spy_df
                )
                if not etf_dfs or benchmark_etf not in etf_dfs:
                    benchmark_etf = "SPY"
                fdf["sector"] = sector
                fdf["benchmark_etf"] = benchmark_etf
                for horizon in [5, 10, 20]:
                    stock_r, market_r, abnormal_r = zip(*fdf["filed_at"].apply(
                        lambda d: get_abnormal_return(d, 1, horizon, pdf, ticker_benchmark)).tolist())
                    fdf[f"stock_ret_{horizon}d"] = stock_r
                    fdf[f"market_ret_{horizon}d"] = market_r
                    fdf[f"abnormal_ret_{horizon}d"] = abnormal_r
                    target = pd.Series(abnormal_r, index=fdf.index)
                    fdf[f"target_{horizon}d"] = np.where(target.notna(), (target > 0).astype(int), np.nan)

                # Primary target = 5-day abnormal from t+1.
                fdf["target"] = fdf["target_5d"]

                # Interaction features.
                fdf["lm_neg_x_cosine"] = fdf["lm_negative"] * (1 - fdf["cosine_sim_prev"].fillna(0.9))
                fdf["lm_unc_x_drift"] = fdf["lm_uncertainty"] * fdf["risk_drift_4q"].fillna(0)
                fdf["text_price_divergence"] = fdf["lm_net_sentiment"].fillna(0) * (-fdf["price_return_20d"].fillna(0))

                cols = [
                    "filed_at","ticker","sector","benchmark_etf",
                    "cosine_sim_prev","risk_drift_4q","filing_surprise","sector_contagion",
                    "lm_negative","lm_positive","lm_uncertainty","lm_litigious",
                    "lm_constraining","lm_net_sentiment",
                    "lm_negative_delta","lm_positive_delta","lm_uncertainty_delta","lm_litigious_delta",
                    "lm_neg_x_cosine","lm_unc_x_drift","text_price_divergence",
                    "text_length_norm",
                    # MD&A features
                    "cosine_sim_prev_mda",
                    "lm_negative_mda","lm_positive_mda","lm_uncertainty_mda",
                    "lm_litigious_mda","lm_net_sentiment_mda",
                    # FinBERT
                    "finbert_cosine_sim",
                    # Price features
                    "price_return_1d","price_return_5d","price_return_20d",
                    "price_volatility_20d","price_ma_ratio_5","price_ma_ratio_20","price_rsi",
                    "filing_day_return",
                    # Targets
                    "stock_ret_5d","stock_ret_10d","stock_ret_20d",
                    "market_ret_5d","market_ret_10d","market_ret_20d",
                    "abnormal_ret_5d","abnormal_ret_10d","abnormal_ret_20d",
                    "target_5d","target_10d","target_20d","target",
                ]
                avail = [c for c in cols if c in fdf.columns]
                rec = fdf[avail].dropna(subset=["target", "cosine_sim_prev"])
                if rec.empty:
                    record_skip(ticker, "no_usable_rows_after_cleaning")
                    last_status = "skip:no_rows"
                    continue

                all_records.append(rec)
                success_tickers += 1
                total_rows += len(rec)
                last_status = f"ok:{ticker}:{len(rec)}"
            except Exception as e:
                record_skip(ticker, "ticker_error", str(e))
                logger.warning(f"{ticker}: unexpected processing error: {e}")
                last_status = "skip:error"
            finally:
                section_ok = sum(c["available_filings"] for c in section_stats.values())
                section_missing = sum(c["missing_filings"] for c in section_stats.values())
                progress.set_postfix({
                    "ok": success_tickers,
                    "rows": total_rows,
                    "skip": sum(skip_reasons.values()),
                    "sec_ok": section_ok,
                    "sec_missing": section_missing,
                    "last": last_status,
                }, refresh=True)

    if not all_records:
        logger.error("No usable filing records were produced; filing_aligned.csv was not written.")
        if skip_reasons:
            logger.info("Skip summary: %s", dict(skip_reasons))
        return

    final_df = pd.concat(all_records, ignore_index=True).sort_values("filed_at").reset_index(drop=True)
    temporal_cols = ["risk_drift_4q","filing_surprise",
                     "lm_negative_delta","lm_positive_delta",
                     "lm_uncertainty_delta","lm_litigious_delta"]
    # MD&A and FinBERT features are optional (missing when Item 2 is absent or
    # sentence-transformers not installed). Impute so no rows are dropped downstream.
    optional_cols = [
        "cosine_sim_prev_mda",
        "lm_negative_mda","lm_positive_mda","lm_uncertainty_mda",
        "lm_litigious_mda","lm_net_sentiment_mda",
        "finbert_cosine_sim",
        "filing_day_return",
    ]
    final_df = impute_cols(final_df, temporal_cols + optional_cols)

    logger.info("Computing sector contagion (look-back only)...")
    final_df["sector_contagion"] = compute_sector_contagion(
        final_df,
        metadata_path=metadata_path,
        show_progress=True,
    )
    sec_med = final_df.groupby("sector")["sector_contagion"].transform("median")
    final_df["sector_contagion"] = final_df["sector_contagion"].fillna(sec_med).fillna(
        final_df["sector_contagion"].median())

    out = os.path.join(processed_dir, "filing_aligned.csv")
    final_df.to_csv(out, index=False)
    logger.info(f"Saved {len(final_df)} rows to {out}")
    logger.info(
        "Ticker processing summary: %d succeeded, %d skipped, %d usable filing rows",
        success_tickers, sum(skip_reasons.values()), len(final_df),
    )
    if skip_reasons:
        logger.info("Skip reasons:")
        for reason, count in skip_reasons.most_common():
            examples = ", ".join(skip_details[reason][:8])
            if len(skip_details[reason]) > 8:
                examples += ", ..."
            logger.info("  %-34s %4d | %s", reason, count, examples)

    if section_stats:
        logger.info("Section availability before row filtering:")
        for title, stats in section_stats.items():
            available = stats["available_filings"]
            missing = stats["missing_filings"]
            total = available + missing
            pct = (available / total * 100) if total else 0.0
            logger.info(
                "  %-44s available=%5d missing=%5d available_pct=%5.1f%% "
                "tickers_with_any=%3d tickers_missing_all=%3d",
                title,
                available,
                missing,
                pct,
                stats["tickers_with_any"],
                stats["tickers_missing_all"],
            )
    if skipped_missing_metadata:
        logger.info(
            "Skipped %d tickers with missing sector metadata: %s",
            len(skipped_missing_metadata),
            ", ".join(sorted(skipped_missing_metadata)),
        )
    for h in [5, 10, 20]:
        col = f"target_{h}d"
        if col in final_df.columns:
            vc = final_df[col].value_counts().to_dict()
            logger.info(f"  target_{h}d: {vc}")

if __name__ == "__main__":
    main()

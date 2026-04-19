"""
filing_dataset.py v3

Key fixes vs v2:
1. Abnormal return computed from t+1 (skip filing day noise)
   Filing day (t=0) is dominated by algo traders reacting to headline numbers.
   The text-driven informed reaction happens t+1 onward.

2. Interaction features added:
   - lm_neg_x_cosine: high negative sentiment AND large text change = strong signal
   - lm_unc_x_drift:  uncertainty increasing while risk profile drifting = bad signal
   - text_price_divergence: text sentiment opposes recent price momentum

3. Multi-horizon targets stored (5d, 10d, 20d) for horizon analysis
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
    "INTC":"Technology","IBM":"Technology",
    "JPM":"Finance","BAC":"Finance","GS":"Finance","WFC":"Finance","MS":"Finance",
    "BLK":"Finance","AXP":"Finance","SPGI":"Finance","CB":"Finance","PGR":"Finance",
    "JNJ":"Healthcare","UNH":"Healthcare","PFE":"Healthcare","ABBV":"Healthcare",
    "MRK":"Healthcare","TMO":"Healthcare","ABT":"Healthcare","DHR":"Healthcare",
    "BMY":"Healthcare","AMGN":"Healthcare",
    "AMZN":"Consumer","WMT":"Consumer","HD":"Consumer","MCD":"Consumer","SBUX":"Consumer",
    "PG":"Consumer","KO":"Consumer","PEP":"Consumer","COST":"Consumer","NKE":"Consumer",
    "CVX":"Energy","XOM":"Energy",
    "CAT":"Industrial","UPS":"Industrial","RTX":"Industrial","NEE":"Industrial",
    "LIN":"Industrial","BA":"Industrial","DE":"Industrial","MMM":"Industrial",
}

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

def compute_sector_contagion(df, window_days=45):
    df = df.copy()
    df["sector"] = df["ticker"].map(SECTOR_MAP)
    df["filed_at"] = pd.to_datetime(df["filed_at"])
    contagion = []
    for idx, row in df.iterrows():
        s = row["sector"]
        if pd.isna(s): contagion.append(np.nan); continue
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

def get_price_features(filing_date, price_df, lookback=50):
    empty = {k: np.nan for k in ["price_return_1d","price_return_5d","price_return_20d",
                                   "price_volatility_20d","price_ma_ratio_5",
                                   "price_ma_ratio_20","price_rsi"]}
    try:
        c = price_df["Close"].dropna()
        c.index = pd.to_datetime(c.index); c = c.sort_index()
        prior = c[c.index < filing_date]
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

def get_abnormal_return(filing_date, start_offset, horizon, price_df, market_df):
    """
    Compute abnormal return from t+start_offset to t+start_offset+horizon.
    start_offset=1 skips the filing day (t=0) noise from algo traders.
    """
    try:
        c = price_df["Close"].dropna().sort_index()
        c.index = pd.to_datetime(c.index)
        m = market_df["Close"].dropna().sort_index() if market_df is not None else None
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

def impute_cols(df, cols):
    df = df.copy()
    for col in cols:
        if col not in df.columns: continue
        cm = df.groupby("ticker")[col].transform("median")
        gm = df[col].median()
        df[col] = df[col].fillna(cm).fillna(gm).fillna(0.0)
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

    logger.info("Downloading SPY benchmark...")
    try:
        spy_df = yf.download("SPY", start=cfg.data.start_date,
                              end=cfg.data.end_date, auto_adjust=True, progress=False)
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = spy_df.columns.get_level_values(0)
    except: spy_df = None

    all_records, skipped = [], []

    for ticker in tickers:
        pp = os.path.join(sec_dir, f"{ticker}_filings.parquet")
        if not os.path.exists(pp): skipped.append(ticker); continue
        fdf = pd.read_parquet(pp)
        fdf["filed_at"] = pd.to_datetime(fdf["filed_at"])
        fdf = fdf.sort_values("filed_at").reset_index(drop=True)
        if fdf[text_col].notna().sum() == 0: skipped.append(ticker); continue
        try:
            pdf = yf.download(ticker, start=cfg.data.start_date,
                               end=cfg.data.end_date, auto_adjust=True, progress=False)
            if pdf.empty: skipped.append(ticker); continue
        except: skipped.append(ticker); continue
        if isinstance(pdf.columns, pd.MultiIndex):
            pdf.columns = pdf.columns.get_level_values(0)

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
            lambda d: get_price_features(d, pdf)).tolist())
        fdf = pd.concat([fdf.reset_index(drop=True), pf.reset_index(drop=True)], axis=1)

        # Multi-horizon abnormal returns from t+1 (skip filing day)
        for horizon in [5, 10, 20]:
            stock_r, market_r, abnormal_r = zip(*fdf["filed_at"].apply(
                lambda d: get_abnormal_return(d, 1, horizon, pdf, spy_df)).tolist())
            fdf[f"stock_ret_{horizon}d"]    = stock_r
            fdf[f"market_ret_{horizon}d"]   = market_r
            fdf[f"abnormal_ret_{horizon}d"] = abnormal_r
            fdf[f"target_{horizon}d"]       = (pd.Series(abnormal_r) > 0).astype(int).values

        # Primary target = 5-day abnormal from t+1
        fdf["target"] = fdf["target_5d"]

        # Interaction features (new in v3)
        fdf["lm_neg_x_cosine"]       = fdf["lm_negative"] * (1 - fdf["cosine_sim_prev"].fillna(0.9))
        fdf["lm_unc_x_drift"]        = fdf["lm_uncertainty"] * fdf["risk_drift_4q"].fillna(0)
        fdf["text_price_divergence"] = fdf["lm_net_sentiment"].fillna(0) * (-fdf["price_return_20d"].fillna(0))

        cols = ["filed_at","ticker",
                "cosine_sim_prev","risk_drift_4q","filing_surprise","sector_contagion",
                "lm_negative","lm_positive","lm_uncertainty","lm_litigious",
                "lm_constraining","lm_net_sentiment",
                "lm_negative_delta","lm_positive_delta","lm_uncertainty_delta","lm_litigious_delta",
                "lm_neg_x_cosine","lm_unc_x_drift","text_price_divergence",
                "text_length_norm",
                "price_return_1d","price_return_5d","price_return_20d",
                "price_volatility_20d","price_ma_ratio_5","price_ma_ratio_20","price_rsi",
                "abnormal_ret_5d","abnormal_ret_10d","abnormal_ret_20d",
                "target_5d","target_10d","target_20d","target"]
        avail = [c for c in cols if c in fdf.columns]
        rec = fdf[avail].dropna(subset=["target","cosine_sim_prev"])
        logger.info(f"{ticker}: {len(rec)} rows")
        all_records.append(rec)

    final_df = pd.concat(all_records, ignore_index=True).sort_values("filed_at").reset_index(drop=True)
    temporal_cols = ["risk_drift_4q","filing_surprise",
                     "lm_negative_delta","lm_positive_delta",
                     "lm_uncertainty_delta","lm_litigious_delta"]
    final_df = impute_cols(final_df, temporal_cols)

    logger.info("Computing sector contagion...")
    final_df["sector_contagion"] = compute_sector_contagion(final_df)
    sec_med = final_df.groupby(final_df["ticker"].map(SECTOR_MAP))["sector_contagion"].transform("median")
    final_df["sector_contagion"] = final_df["sector_contagion"].fillna(sec_med).fillna(
        final_df["sector_contagion"].median())

    out = os.path.join(processed_dir, "filing_aligned.csv")
    final_df.to_csv(out, index=False)
    logger.info(f"Saved {len(final_df)} rows")
    for h in [5,10,20]:
        col = f"target_{h}d"
        if col in final_df.columns:
            vc = final_df[col].value_counts().to_dict()
            logger.info(f"  target_{h}d: {vc}")

if __name__ == "__main__":
    main()

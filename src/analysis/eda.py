"""
eda.py - Exploratory Data Analysis for the filing-aligned dataset.
Addresses professor concern (2): understanding data distribution.
Saves all plots to outputs/eda/

Also provides plot_caar() — CAAR event study callable from baseline.py.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
from omegaconf import DictConfig

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))

from src.dataloader.company_metadata import get_ticker_sector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REGIME_MAP = {
    2015:"calm",2016:"calm",2017:"bull",2018:"volatile",2019:"bull",
    2020:"crisis",2021:"recovery",2022:"volatile",2023:"recovery",2024:"bull",
}
SECTOR_COLORS = {
    "Communication Services":"#4C72B0",
    "Consumer Discretionary":"#DD8452",
    "Consumer Staples":"#55A868",
    "Energy":"#C44E52",
    "Financials":"#8172B3",
    "Health Care":"#937860",
    "Industrials":"#56B4E9",
    "Information Technology":"#009E73",
    "Materials":"#E69F00",
    "Real Estate":"#CC79A7",
    "Utilities":"#7F7F7F",
}

def run_eda(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = df.copy()
    df["filed_at"] = pd.to_datetime(df["filed_at"])
    df["year"] = df["filed_at"].dt.year
    df["quarter"] = df["filed_at"].dt.to_period("Q").astype(str)
    if "sector" not in df.columns:
        df["sector"] = df["ticker"].apply(get_ticker_sector)
    df["sector"] = df["sector"].fillna("Unknown")
    df["regime"] = df["year"].map(REGIME_MAP).fillna("unknown")
    logger.info(f"Dataset: {len(df)} rows, {df['ticker'].nunique()} tickers, {df['year'].min()}-{df['year'].max()}")
    _plot_target_by_year(df, output_dir)
    _plot_cosine_sim_over_time(df, output_dir)
    _plot_feature_distributions(df, output_dir)
    _plot_sector_breakdown(df, output_dir)
    _plot_regime_analysis(df, output_dir)
    _plot_correlation_heatmap(df, output_dir)
    _plot_contagion_signal(df, output_dir)
    _print_summary_stats(df)
    logger.info(f"All EDA plots saved to {output_dir}")

def _plot_target_by_year(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    yearly = df.groupby(["year","target"]).size().unstack(fill_value=0)
    yearly.columns = ["Down (0)","Up (1)"]
    yearly.plot(kind="bar", stacked=True, ax=axes[0], color=["#C44E52","#4C72B0"], edgecolor="white")
    axes[0].set_title("Filing count by year and direction", fontsize=13)
    axes[0].set_xlabel("Year"); axes[0].set_ylabel("Number of filings")
    axes[0].tick_params(axis="x", rotation=45); axes[0].legend()
    up_rate = df.groupby("year")["target"].mean()
    axes[1].plot(up_rate.index, up_rate.values, marker="o", color="#4C72B0", linewidth=2)
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50% baseline")
    axes[1].fill_between(up_rate.index, up_rate.values, 0.5, where=(up_rate.values>0.5), alpha=0.2, color="#4C72B0")
    axes[1].fill_between(up_rate.index, up_rate.values, 0.5, where=(up_rate.values<0.5), alpha=0.2, color="#C44E52")
    axes[1].set_title("Fraction of filings followed by price increase", fontsize=13)
    axes[1].set_xlabel("Year"); axes[1].set_ylabel("Up-rate"); axes[1].set_ylim(0.3, 0.75); axes[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "01_target_by_year.png"), dpi=150); plt.close()
    logger.info("Saved: 01_target_by_year.png")

def _plot_cosine_sim_over_time(df, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    quarterly_med = df.groupby("quarter")["cosine_sim_prev"].median()
    q_index = range(len(quarterly_med))
    axes[0].plot(q_index, quarterly_med.values, color="#4C72B0", linewidth=1.5)
    axes[0].fill_between(q_index, quarterly_med.values, quarterly_med.mean(), alpha=0.2, color="#4C72B0")
    df_q = df[["quarter","year"]].drop_duplicates().sort_values("quarter").reset_index(drop=True)
    df_q["q_idx"] = range(len(df_q))
    crisis_qs = df_q[df_q["year"].map(REGIME_MAP).isin(["crisis","volatile"])]["q_idx"].values
    for idx in crisis_qs:
        axes[0].axvspan(idx-0.5, idx+0.5, alpha=0.08, color="red")
    axes[0].set_title("Median quarter-over-quarter cosine similarity (red = volatile/crisis regimes)", fontsize=12)
    axes[0].set_xticks(list(q_index)[::4])
    axes[0].set_xticklabels(list(quarterly_med.index)[::4], rotation=45, fontsize=8)
    axes[0].set_ylabel("Cosine similarity")
    axes[1].hist(df["cosine_sim_prev"].dropna(), bins=40, color="#4C72B0", edgecolor="white", alpha=0.8)
    axes[1].axvline(df["cosine_sim_prev"].median(), color="red", linestyle="--", label=f"Median: {df['cosine_sim_prev'].median():.3f}")
    axes[1].axvline(df["cosine_sim_prev"].mean(), color="orange", linestyle="--", label=f"Mean: {df['cosine_sim_prev'].mean():.3f}")
    axes[1].set_title("Distribution of cosine_sim_prev", fontsize=12)
    axes[1].set_xlabel("Cosine similarity to previous quarter"); axes[1].set_ylabel("Count"); axes[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "02_cosine_sim_over_time.png"), dpi=150); plt.close()
    logger.info("Saved: 02_cosine_sim_over_time.png")

def _plot_feature_distributions(df, output_dir):
    text_features = ["cosine_sim_prev","sentiment_compound","text_length_norm","filing_surprise","risk_drift_4q"]
    price_features = ["price_return_1d","price_return_5d","price_return_20d","price_volatility_20d","price_ma_ratio_20","price_rsi"]
    all_feats = [f for f in text_features+price_features if f in df.columns]
    ncols=4; nrows=(len(all_feats)+ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows*3.5))
    axes = axes.flatten()
    for i, feat in enumerate(all_feats):
        d0 = df[df["target"]==0][feat].dropna(); d1 = df[df["target"]==1][feat].dropna()
        axes[i].hist(d0, bins=30, alpha=0.6, color="#C44E52", label="Down", density=True, edgecolor="white")
        axes[i].hist(d1, bins=30, alpha=0.6, color="#4C72B0", label="Up", density=True, edgecolor="white")
        axes[i].set_title(feat, fontsize=10); axes[i].legend(fontsize=8)
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    fig.suptitle("Feature distributions: Up vs Down filings", fontsize=14, y=1.01)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir,"03_feature_distributions.png"), dpi=150, bbox_inches="tight"); plt.close()
    logger.info("Saved: 03_feature_distributions.png")

def _plot_sector_breakdown(df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sc = df.groupby("sector").size().sort_values(ascending=False)
    axes[0].bar(sc.index, sc.values, color=[SECTOR_COLORS.get(s,"#888") for s in sc.index], edgecolor="white")
    axes[0].set_title("Filing count by sector", fontsize=12); axes[0].tick_params(axis="x", rotation=30)
    ur = df.groupby("sector")["target"].mean().sort_values(ascending=False)
    axes[1].bar(ur.index, ur.values, color=[SECTOR_COLORS.get(s,"#888") for s in ur.index], edgecolor="white")
    axes[1].axhline(0.5, color="gray", linestyle="--"); axes[1].set_title("Up-rate by sector", fontsize=12)
    axes[1].tick_params(axis="x", rotation=30); axes[1].set_ylim(0.3, 0.75)
    so = df.groupby("sector")["cosine_sim_prev"].median().sort_values().index.tolist()
    sd = [df[df["sector"]==s]["cosine_sim_prev"].dropna().values for s in so]
    bp = axes[2].boxplot(sd, labels=so, patch_artist=True, medianprops={"color":"white","linewidth":2})
    for patch, s in zip(bp["boxes"], so): patch.set_facecolor(SECTOR_COLORS.get(s,"#888")); patch.set_alpha(0.8)
    axes[2].set_title("Cosine similarity by sector", fontsize=12); axes[2].tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir,"04_sector_breakdown.png"), dpi=150); plt.close()
    logger.info("Saved: 04_sector_breakdown.png")

def _plot_regime_analysis(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ro = ["calm","bull","volatile","crisis","recovery"]
    rc = {"calm":"#4C72B0","bull":"#55A868","volatile":"#DD8452","crisis":"#C44E52","recovery":"#8172B3"}
    ur = df.groupby("regime")["target"].mean().reindex(ro).dropna()
    axes[0].bar(ur.index, ur.values, color=[rc[r] for r in ur.index], edgecolor="white")
    axes[0].axhline(0.5, color="gray", linestyle="--"); axes[0].set_title("Up-rate by market regime", fontsize=12)
    axes[0].set_ylim(0.3, 0.75)
    vr = [r for r in ro if r in df["regime"].unique()]
    rd = [df[df["regime"]==r]["cosine_sim_prev"].dropna().values for r in vr]
    bp = axes[1].boxplot(rd, labels=vr, patch_artist=True, medianprops={"color":"white","linewidth":2})
    for patch, r in zip(bp["boxes"], vr): patch.set_facecolor(rc.get(r,"#888")); patch.set_alpha(0.8)
    axes[1].set_title("Cosine similarity by market regime", fontsize=12)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir,"05_regime_analysis.png"), dpi=150); plt.close()
    logger.info("Saved: 05_regime_analysis.png")

def _plot_correlation_heatmap(df, output_dir):
    fc = [c for c in df.columns if c not in ["filed_at","ticker","target","price_return","year","quarter","sector","regime"] and df[c].dtype in [np.float64, float]]
    corr = df[fc+["target"]].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.zeros_like(corr, dtype=bool); mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr, ax=ax, mask=mask, cmap="RdBu_r", center=0, vmin=-0.6, vmax=0.6, annot=True, fmt=".2f", annot_kws={"size":8}, linewidths=0.5)
    ax.set_title("Feature correlation matrix", fontsize=13)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir,"06_correlation_heatmap.png"), dpi=150); plt.close()
    logger.info("Saved: 06_correlation_heatmap.png")

def _plot_contagion_signal(df, output_dir):
    if "sector_contagion" not in df.columns: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["sector_contagion"].dropna(), bins=40, color="#8172B3", edgecolor="white", alpha=0.8)
    axes[0].set_title("Distribution of sector contagion signal", fontsize=12)
    axes[0].set_xlabel("Average peer cosine similarity (same sector, same quarter)")
    df_t = df.dropna(subset=["sector_contagion"]).copy()
    df_t["cq"] = pd.qcut(df_t["sector_contagion"], q=5, duplicates="drop")
    ubq = df_t.groupby("cq")["target"].mean()
    axes[1].bar(ubq.index.astype(str), ubq.values, color=["#C44E52" if v<0.5 else "#4C72B0" for v in ubq.values], edgecolor="white")
    axes[1].axhline(0.5, color="gray", linestyle="--")
    axes[1].set_title("Up-rate by sector contagion quintile\n(Q1=peers rewrote most)", fontsize=12)
    axes[1].set_ylim(0.3, 0.75)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir,"07_contagion_signal.png"), dpi=150); plt.close()
    logger.info("Saved: 07_contagion_signal.png")

def _print_summary_stats(df):
    logger.info("\n" + "="*60 + "\nDATASET SUMMARY\n" + "="*60)
    logger.info(f"Total: {len(df)} | Companies: {df['ticker'].nunique()} | Range: {df['filed_at'].min().date()} to {df['filed_at'].max().date()}")
    logger.info(f"Target: {df['target'].mean():.1%} up / {1-df['target'].mean():.1%} down")
    logger.info(f"Cosine sim: mean={df['cosine_sim_prev'].mean():.4f} median={df['cosine_sim_prev'].median():.4f} std={df['cosine_sim_prev'].std():.4f}")
    for regime, grp in df.groupby("regime"): logger.info(f"  {regime:12s}: {grp['target'].mean():.1%} up (n={len(grp)})")
    logger.info("="*60)

def plot_caar(test_pred_df, output_dir, data_start="2022-01-01", data_end="2025-12-31"):
    """
    Cumulative Average Abnormal Return (CAAR) event study.

    For each filing in test_pred_df, computes daily abnormal returns
    (stock − SPY) for t+1 to t+20, then plots the CAAR separately for
    model-predicted-up vs model-predicted-down filings with ±1 SE bands.

    Parameters
    ----------
    test_pred_df : pd.DataFrame
        Columns required: filed_at, ticker, y_pred
    output_dir : str
        Directory to save the PNG
    data_start, data_end : str
        Date range for yfinance download
    """
    import yfinance as yf

    os.makedirs(output_dir, exist_ok=True)
    test_pred_df = test_pred_df.copy()
    test_pred_df["filed_at"] = pd.to_datetime(test_pred_df["filed_at"])

    tickers = test_pred_df["ticker"].unique().tolist()
    logger.info(f"CAAR: downloading prices for {len(tickers)} tickers + SPY ...")

    prices = {}
    for ticker in tickers + ["SPY"]:
        try:
            pdf = yf.download(ticker, start=data_start, end=data_end,
                               auto_adjust=True, progress=False)
            if isinstance(pdf.columns, pd.MultiIndex):
                pdf.columns = pdf.columns.get_level_values(0)
            prices[ticker] = pdf["Close"].dropna().sort_index()
        except Exception:
            pass

    spy = prices.get("SPY", None)
    if spy is None:
        logger.warning("CAAR: SPY data unavailable — skipping plot.")
        return

    HORIZON = 20
    up_ars, down_ars = [], []   # list of arrays, one per filing

    for _, row in test_pred_df.iterrows():
        ticker = row["ticker"]
        filing_date = row["filed_at"]
        y_pred = int(row["y_pred"])

        if ticker not in prices:
            continue

        stock = prices[ticker]
        fut_stock = stock[stock.index >= filing_date]
        fut_spy   = spy[spy.index >= filing_date]

        if len(fut_stock) < HORIZON + 1 or len(fut_spy) < HORIZON + 1:
            continue

        # Daily returns t+1 … t+HORIZON (pct_change then drop the t=0 NaN)
        stock_daily = fut_stock.iloc[:HORIZON + 1].pct_change().dropna().values
        spy_daily   = fut_spy.iloc[:HORIZON + 1].pct_change().dropna().values

        n = min(len(stock_daily), len(spy_daily), HORIZON)
        if n < 1:
            continue

        ar = stock_daily[:n] - spy_daily[:n]
        if y_pred == 1:
            up_ars.append(ar)
        else:
            down_ars.append(ar)

    def _to_caar(ar_list):
        if not ar_list:
            return None, None, 0
        max_len = max(len(a) for a in ar_list)
        mat = np.full((len(ar_list), max_len), np.nan)
        for i, a in enumerate(ar_list):
            mat[i, :len(a)] = a
        mean_ar = np.nanmean(mat, axis=0)
        se_ar   = np.nanstd(mat, axis=0) / np.sqrt(np.sum(~np.isnan(mat), axis=0).clip(1))
        caar    = np.nancumsum(mean_ar)
        # Propagate SE as cumulative sum of per-day SE (conservative)
        caar_se = np.nancumsum(se_ar)
        return caar, caar_se, len(ar_list)

    up_caar,   up_se,   n_up   = _to_caar(up_ars)
    down_caar, down_se, n_down = _to_caar(down_ars)

    if up_caar is None and down_caar is None:
        logger.warning("CAAR: no valid filings — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    days = np.arange(1, HORIZON + 1)

    if up_caar is not None:
        n = len(up_caar)
        ax.plot(days[:n], up_caar[:n] * 100, color="#4C72B0", linewidth=2,
                label=f"Predicted Up (n={n_up})")
        ax.fill_between(days[:n],
                        (up_caar[:n] - up_se[:n]) * 100,
                        (up_caar[:n] + up_se[:n]) * 100,
                        alpha=0.18, color="#4C72B0")

    if down_caar is not None:
        n = len(down_caar)
        ax.plot(days[:n], down_caar[:n] * 100, color="#C44E52", linewidth=2,
                label=f"Predicted Down (n={n_down})")
        ax.fill_between(days[:n],
                        (down_caar[:n] - down_se[:n]) * 100,
                        (down_caar[:n] + down_se[:n]) * 100,
                        alpha=0.18, color="#C44E52")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Trading days after filing (t+1 to t+20)")
    ax.set_ylabel("CAAR (%)")
    ax.set_title("Cumulative Average Abnormal Return (CAAR) by Model Prediction\n"
                 "Abnormal return = stock daily return − SPY daily return", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "caar_event_study.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {out_path}")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    aligned_path = os.path.join(cfg.data.processed_dir, "filing_aligned.csv")
    if not os.path.exists(aligned_path):
        logger.error(f"Dataset not found at {aligned_path}. Run filing_dataset.py first.")
        return
    df = pd.read_csv(aligned_path, parse_dates=["filed_at"])
    run_eda(df, "outputs/eda")

if __name__ == "__main__":
    main()

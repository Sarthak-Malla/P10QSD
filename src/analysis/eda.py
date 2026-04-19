"""
eda.py - Exploratory Data Analysis for the filing-aligned dataset.
Addresses professor concern (2): understanding data distribution.
Saves all plots to outputs/eda/
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
from omegaconf import DictConfig

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
REGIME_MAP = {
    2015:"calm",2016:"calm",2017:"bull",2018:"volatile",2019:"bull",
    2020:"crisis",2021:"recovery",2022:"volatile",2023:"recovery",2024:"bull",
}
SECTOR_COLORS = {
    "Technology":"#4C72B0","Finance":"#DD8452","Healthcare":"#55A868",
    "Consumer":"#C44E52","Energy":"#8172B3","Industrial":"#937860",
}

def run_eda(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = df.copy()
    df["filed_at"] = pd.to_datetime(df["filed_at"])
    df["year"] = df["filed_at"].dt.year
    df["quarter"] = df["filed_at"].dt.to_period("Q").astype(str)
    df["sector"] = df["ticker"].map(SECTOR_MAP).fillna("Other")
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
    df_t["cq"] = pd.qcut(df_t["sector_contagion"], q=5, labels=["Q1","Q2","Q3","Q4","Q5"])
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

"""
Quantile evaluation for holdout model predictions.

Ranks filings by model probability and checks whether high-score filings have
higher realized sector-adjusted abnormal returns than low-score filings.
"""
import argparse
import logging
import os
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind


logger = logging.getLogger(__name__)
DEFAULT_HORIZONS = (5, 10, 20)
DEFAULT_MIN_GROUP_SIZE = 25
ANALYSES = ("global", "sector_neutral", "time_neutral")


def load_predictions(pred_path: str) -> pd.DataFrame:
    pred = pd.read_csv(pred_path, parse_dates=["filed_at"])
    required = {"ticker", "filed_at", "y_proba"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"Prediction file missing columns: {sorted(missing)}")
    return pred


def load_dataset(data_path: str) -> pd.DataFrame:
    data = pd.read_csv(data_path, parse_dates=["filed_at"])
    required = {"ticker", "filed_at"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset file missing columns: {sorted(missing)}")
    return data


def prepare_evaluation_frame(
    pred: pd.DataFrame,
    data: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Merge holdout predictions with realized returns from filing_aligned.csv."""
    pred = pred.copy()
    data = data.copy()

    pred["_ticker_key"] = pred["ticker"].astype(str).str.upper()
    data["_ticker_key"] = data["ticker"].astype(str).str.upper()
    pred["_filed_date"] = pd.to_datetime(pred["filed_at"]).dt.tz_localize(None).dt.normalize()
    data["_filed_date"] = pd.to_datetime(data["filed_at"]).dt.tz_localize(None).dt.normalize()

    keep_cols = ["_ticker_key", "_filed_date", "sector", "benchmark_etf"]
    for horizon in horizons:
        keep_cols.extend([
            f"stock_ret_{horizon}d",
            f"market_ret_{horizon}d",
            f"abnormal_ret_{horizon}d",
            f"target_{horizon}d",
        ])
    keep_cols = [c for c in keep_cols if c in data.columns]

    data = data[keep_cols].drop_duplicates(["_ticker_key", "_filed_date"], keep="last")
    merged = pred.merge(data, on=["_ticker_key", "_filed_date"], how="inner")
    if merged.empty:
        raise ValueError("No prediction rows matched filing_aligned.csv on ticker + filed_at")
    if len(merged) < len(pred):
        logger.warning("Matched %d/%d prediction rows to filing_aligned.csv", len(merged), len(pred))

    merged["ticker"] = merged["ticker"].astype(str).str.upper()
    merged["filed_at"] = pd.to_datetime(merged["filed_at"])
    merged["year"] = merged["filed_at"].dt.year
    if "sector" not in merged.columns:
        merged["sector"] = "Unknown"
    if "benchmark_etf" not in merged.columns:
        merged["benchmark_etf"] = "SPY"
    merged["sector"] = merged["sector"].fillna("Unknown")
    merged["benchmark_etf"] = merged["benchmark_etf"].fillna("SPY")
    return merged.drop(columns=["_ticker_key", "_filed_date"], errors="ignore")


def _rank_quantiles(scores: pd.Series, n_quantiles: int = 5) -> pd.Series:
    out = pd.Series(pd.NA, index=scores.index, dtype="Int64")
    valid = scores.dropna()
    if len(valid) < n_quantiles:
        return out
    ranks = valid.rank(method="first")
    q = pd.qcut(ranks, q=n_quantiles, labels=list(range(1, n_quantiles + 1)))
    out.loc[valid.index] = q.astype("Int64")
    return out


def assign_quantiles(
    df: pd.DataFrame,
    analysis: str,
    score_col: str = "y_proba",
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
    n_quantiles: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assign global, sector-neutral, or time-neutral quintiles."""
    if analysis not in ANALYSES:
        raise ValueError(f"Unknown analysis '{analysis}'. Expected one of {ANALYSES}")
    if score_col not in df.columns:
        raise ValueError(f"Missing score column '{score_col}'")

    out = df.copy()
    out["analysis"] = analysis
    out["quantile"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    skipped = []

    if analysis == "global":
        valid = out[score_col].notna()
        if int(valid.sum()) < n_quantiles:
            skipped.append({
                "analysis": analysis, "group_col": "all", "group_value": "all",
                "n": int(valid.sum()), "reason": "too_few_rows",
            })
        out.loc[valid, "quantile"] = _rank_quantiles(out.loc[valid, score_col], n_quantiles)
    else:
        group_col = "sector" if analysis == "sector_neutral" else "year"
        for group_value, idx in out[out[score_col].notna()].groupby(group_col).groups.items():
            n = len(idx)
            if pd.isna(group_value) or n < min_group_size:
                skipped.append({
                    "analysis": analysis,
                    "group_col": group_col,
                    "group_value": group_value,
                    "n": int(n),
                    "reason": "too_few_rows",
                })
                continue
            out.loc[idx, "quantile"] = _rank_quantiles(out.loc[idx, score_col], n_quantiles)

    out["quantile_label"] = out["quantile"].apply(
        lambda q: f"Q{int(q)}" if pd.notna(q) else ""
    )
    return out, pd.DataFrame(skipped)


def summarize_quantiles(
    assigned: pd.DataFrame,
    analysis: str,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    score_col: str = "y_proba",
) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        ret_col = f"abnormal_ret_{horizon}d"
        target_col = f"target_{horizon}d"
        stock_col = f"stock_ret_{horizon}d"
        market_col = f"market_ret_{horizon}d"
        if ret_col not in assigned.columns:
            continue
        valid = assigned.dropna(subset=["quantile", ret_col]).copy()
        for q in range(1, 6):
            part = valid[valid["quantile"] == q]
            hit_rate = np.nan
            if len(part) > 0:
                if target_col in part.columns:
                    hit_rate = part[target_col].mean()
                else:
                    hit_rate = (part[ret_col] > 0).mean()
            rows.append({
                "analysis": analysis,
                "horizon": horizon,
                "quantile": q,
                "quantile_label": f"Q{q}",
                "n": int(len(part)),
                "avg_y_proba": part[score_col].mean() if len(part) else np.nan,
                "hit_rate": hit_rate,
                "mean_abnormal_ret": part[ret_col].mean() if len(part) else np.nan,
                "median_abnormal_ret": part[ret_col].median() if len(part) else np.nan,
                "std_abnormal_ret": part[ret_col].std() if len(part) else np.nan,
                "mean_stock_ret": part[stock_col].mean() if stock_col in part.columns and len(part) else np.nan,
                "mean_benchmark_ret": part[market_col].mean() if market_col in part.columns and len(part) else np.nan,
            })
    return pd.DataFrame(rows)


def _spread_stats(q1: pd.Series, q5: pd.Series, means: pd.Series) -> Dict[str, float]:
    q1 = q1.dropna()
    q5 = q5.dropna()
    spread = q5.mean() - q1.mean() if len(q1) and len(q5) else np.nan
    t_stat, p_value = np.nan, np.nan
    if len(q1) >= 2 and len(q5) >= 2:
        test = ttest_ind(q5, q1, equal_var=False, nan_policy="omit")
        t_stat, p_value = float(test.statistic), float(test.pvalue)

    rho, rho_p = np.nan, np.nan
    means = means.dropna()
    if len(means) >= 2:
        corr = spearmanr(means.index.astype(float), means.values)
        rho, rho_p = float(corr.statistic), float(corr.pvalue)

    return {
        "q1_n": int(len(q1)),
        "q5_n": int(len(q5)),
        "q1_mean_abnormal_ret": q1.mean() if len(q1) else np.nan,
        "q5_mean_abnormal_ret": q5.mean() if len(q5) else np.nan,
        "long_short_spread": spread,
        "welch_t_stat": t_stat,
        "welch_p_value": p_value,
        "monotonic_spearman": rho,
        "monotonic_p_value": rho_p,
    }


def compute_long_short(
    assigned: pd.DataFrame,
    analysis: str,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        ret_col = f"abnormal_ret_{horizon}d"
        if ret_col not in assigned.columns:
            continue
        valid = assigned.dropna(subset=["quantile", ret_col])
        means = valid.groupby("quantile")[ret_col].mean()
        stats = _spread_stats(
            valid.loc[valid["quantile"] == 1, ret_col],
            valid.loc[valid["quantile"] == 5, ret_col],
            means,
        )
        stats.update({"analysis": analysis, "horizon": horizon})
        rows.append(stats)
    return pd.DataFrame(rows)


def compute_group_spreads(
    assigned: pd.DataFrame,
    analysis: str,
    group_col: str,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    rows = []
    for group_value, group_df in assigned.dropna(subset=["quantile"]).groupby(group_col):
        for horizon in horizons:
            ret_col = f"abnormal_ret_{horizon}d"
            if ret_col not in group_df.columns:
                continue
            valid = group_df.dropna(subset=[ret_col])
            means = valid.groupby("quantile")[ret_col].mean()
            stats = _spread_stats(
                valid.loc[valid["quantile"] == 1, ret_col],
                valid.loc[valid["quantile"] == 5, ret_col],
                means,
            )
            stats.update({
                "analysis": analysis,
                "group_col": group_col,
                "group_value": group_value,
                "horizon": horizon,
            })
            rows.append(stats)
    return pd.DataFrame(rows)


def _plot_quantile_metric(
    summary: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    out_path: str,
    horizon: int = 5,
) -> None:
    plot_df = summary[(summary["horizon"] == horizon) & summary[metric_col].notna()]
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for analysis, part in plot_df.groupby("analysis"):
        part = part.sort_values("quantile")
        ax.plot(part["quantile"], part[metric_col] * 100, marker="o", linewidth=2, label=analysis)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax.set_xlabel("Model probability quintile")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by model score quintile ({horizon}d)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def write_interpretation(
    out_path: str,
    members: pd.DataFrame,
    long_short: pd.DataFrame,
    skipped: pd.DataFrame,
    main_horizon: int = 5,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> None:
    def row_for(analysis: str) -> pd.Series:
        rows = long_short[(long_short["analysis"] == analysis) & (long_short["horizon"] == main_horizon)]
        return rows.iloc[0] if len(rows) else pd.Series(dtype=object)

    global_row = row_for("global")
    sector_row = row_for("sector_neutral")
    time_row = row_for("time_neutral")

    lines = [
        "# Quantile Evaluation Interpretation",
        "",
        f"Evaluated {members['ticker'].nunique()} tickers and {len(members[members['analysis'] == 'global'])} holdout filings.",
        "Returns are sector-adjusted abnormal returns: stock return minus the relevant benchmark return.",
        "",
        "## Primary 5d Result",
    ]

    if not global_row.empty:
        lines.append(
            f"Global Q5 - Q1 spread: {_fmt_pct(global_row['long_short_spread'])} "
            f"(Q5 {_fmt_pct(global_row['q5_mean_abnormal_ret'])}, "
            f"Q1 {_fmt_pct(global_row['q1_mean_abnormal_ret'])}, "
            f"Welch p={global_row['welch_p_value']:.4f})."
        )
        if global_row["long_short_spread"] > 0:
            lines.append("Interpretation: the model's highest-score filings outperformed its lowest-score filings overall.")
        else:
            lines.append("Interpretation: the global ranking did not produce a positive long-short spread.")

    lines.extend(["", "## Robustness Checks"])
    for label, row, meaning in [
        ("Sector-neutral", sector_row, "not just sector exposure"),
        ("Time-neutral", time_row, "not just year or market-regime timing"),
    ]:
        if row.empty:
            lines.append(f"{label}: not available after minimum group-size filtering.")
            continue
        verdict = "supports the signal" if row["long_short_spread"] > 0 else "weakens the signal"
        lines.append(
            f"{label} Q5 - Q1 spread: {_fmt_pct(row['long_short_spread'])} "
            f"(Welch p={row['welch_p_value']:.4f}); this {verdict} as evidence that it is {meaning}."
        )

    if not skipped.empty:
        lines.extend(["", "## Skipped Groups"])
        grouped = skipped.groupby(["analysis", "group_col"]).size().reset_index(name="n_groups")
        for _, r in grouped.iterrows():
            lines.append(
                f"{r['analysis']} skipped {int(r['n_groups'])} {r['group_col']} groups "
                f"with fewer than {min_group_size} filings."
            )

    lines.extend([
        "",
        "## Caveat",
        "This is a ranking diagnostic, not a full trading backtest. It does not include transaction costs, borrow costs, liquidity, position sizing, or rebalancing constraints.",
    ])

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_quantile_evaluation(
    pred_path: str = "models/test_predictions.csv",
    data_path: str = "data/processed/filing_aligned.csv",
    out_dir: str = "outputs/quantile_eval",
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
    score_col: str = "y_proba",
) -> Dict[str, str]:
    horizons = tuple(int(h) for h in horizons)
    os.makedirs(out_dir, exist_ok=True)

    pred = load_predictions(pred_path)
    data = load_dataset(data_path)
    base = prepare_evaluation_frame(pred, data, horizons=horizons)

    assigned_by_analysis: Dict[str, pd.DataFrame] = {}
    summary_by_analysis: Dict[str, pd.DataFrame] = {}
    long_short_rows: List[pd.DataFrame] = []
    skipped_rows: List[pd.DataFrame] = []

    for analysis in ANALYSES:
        assigned, skipped = assign_quantiles(
            base, analysis=analysis, score_col=score_col, min_group_size=min_group_size
        )
        assigned_by_analysis[analysis] = assigned
        summary = summarize_quantiles(assigned, analysis=analysis, horizons=horizons, score_col=score_col)
        summary_by_analysis[analysis] = summary
        long_short_rows.append(compute_long_short(assigned, analysis=analysis, horizons=horizons))
        if not skipped.empty:
            skipped_rows.append(skipped)

    members = pd.concat(assigned_by_analysis.values(), ignore_index=True)
    summary_all = pd.concat(summary_by_analysis.values(), ignore_index=True)
    long_short = pd.concat(long_short_rows, ignore_index=True)
    skipped = pd.concat(skipped_rows, ignore_index=True) if skipped_rows else pd.DataFrame(
        columns=["analysis", "group_col", "group_value", "n", "reason"]
    )

    members.to_csv(os.path.join(out_dir, "quantile_members.csv"), index=False)
    summary_by_analysis["global"].to_csv(os.path.join(out_dir, "global_quantile_summary.csv"), index=False)
    summary_by_analysis["sector_neutral"].to_csv(os.path.join(out_dir, "sector_neutral_quantile_summary.csv"), index=False)
    summary_by_analysis["time_neutral"].to_csv(os.path.join(out_dir, "time_neutral_quantile_summary.csv"), index=False)
    long_short.to_csv(os.path.join(out_dir, "long_short_summary.csv"), index=False)
    skipped.to_csv(os.path.join(out_dir, "skipped_groups.csv"), index=False)

    sector_spreads = compute_group_spreads(
        assigned_by_analysis["sector_neutral"], "sector_neutral", "sector", horizons=horizons
    )
    time_spreads = compute_group_spreads(
        assigned_by_analysis["time_neutral"], "time_neutral", "year", horizons=horizons
    )
    sector_spreads.to_csv(os.path.join(out_dir, "sector_spread_by_sector.csv"), index=False)
    time_spreads.to_csv(os.path.join(out_dir, "time_spread_by_year.csv"), index=False)

    _plot_quantile_metric(
        summary_all, "mean_abnormal_ret", "Mean sector-adjusted abnormal return (%)",
        os.path.join(out_dir, "quantile_returns.png"), horizon=5
    )
    _plot_quantile_metric(
        summary_all, "hit_rate", "Hit rate (%)",
        os.path.join(out_dir, "quantile_hit_rates.png"), horizon=5
    )
    write_interpretation(
        os.path.join(out_dir, "quantile_interpretation.md"),
        members=members,
        long_short=long_short,
        skipped=skipped,
        main_horizon=5,
        min_group_size=min_group_size,
    )

    logger.info("Quantile evaluation saved to %s", out_dir)
    return {
        "out_dir": out_dir,
        "members": os.path.join(out_dir, "quantile_members.csv"),
        "long_short": os.path.join(out_dir, "long_short_summary.csv"),
        "interpretation": os.path.join(out_dir, "quantile_interpretation.md"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model predictions by return quintile.")
    parser.add_argument("--pred-path", default="models/test_predictions.csv")
    parser.add_argument("--data-path", default="data/processed/filing_aligned.csv")
    parser.add_argument("--out-dir", default="outputs/quantile_eval")
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--min-group-size", type=int, default=DEFAULT_MIN_GROUP_SIZE)
    parser.add_argument("--score-col", default="y_proba")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    run_quantile_evaluation(
        pred_path=args.pred_path,
        data_path=args.data_path,
        out_dir=args.out_dir,
        horizons=args.horizons,
        min_group_size=args.min_group_size,
        score_col=args.score_col,
    )


if __name__ == "__main__":
    main()

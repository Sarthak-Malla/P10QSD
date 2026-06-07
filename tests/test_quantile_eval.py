import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.quantile_eval import (
    assign_quantiles,
    compute_long_short,
    prepare_evaluation_frame,
    run_quantile_evaluation,
    summarize_quantiles,
)


def _frames(n=100):
    first_year = [pd.Timestamp("2023-01-01") + pd.Timedelta(days=i) for i in range(n // 2)]
    second_year = [pd.Timestamp("2024-01-01") + pd.Timedelta(days=i) for i in range(n - n // 2)]
    dates = first_year + second_year
    tickers = [f"T{i:03d}" for i in range(n)]
    scores = np.linspace(0.01, 0.99, n)
    sectors = ["Technology" if i % 2 == 0 else "Financials" for i in range(n)]

    pred = pd.DataFrame({
        "ticker": tickers,
        "filed_at": dates,
        "y_true": (scores > 0.5).astype(int),
        "y_pred": (scores > 0.5).astype(int),
        "y_proba": scores,
    })
    data = pd.DataFrame({
        "ticker": tickers,
        "filed_at": dates,
        "sector": sectors,
        "benchmark_etf": ["XLK" if s == "Technology" else "XLF" for s in sectors],
    })
    for horizon, scale in [(5, 1.0), (10, 1.5), (20, 2.0)]:
        ret = (scores - 0.5) * scale
        data[f"stock_ret_{horizon}d"] = ret + 0.01
        data[f"market_ret_{horizon}d"] = 0.01
        data[f"abnormal_ret_{horizon}d"] = ret
        data[f"target_{horizon}d"] = (ret > 0).astype(int)
    return pred, data


def test_global_quantile_assignment_puts_low_scores_in_q1_and_high_scores_in_q5():
    pred, data = _frames()
    merged = prepare_evaluation_frame(pred, data)

    assigned, skipped = assign_quantiles(merged, "global")

    assert skipped.empty
    assert assigned.loc[assigned["y_proba"].idxmin(), "quantile"] == 1
    assert assigned.loc[assigned["y_proba"].idxmax(), "quantile"] == 5
    assert assigned["quantile"].notna().all()


def test_sector_neutral_quantiles_are_assigned_within_sector():
    pred, data = _frames()
    merged = prepare_evaluation_frame(pred, data)

    assigned, skipped = assign_quantiles(merged, "sector_neutral", min_group_size=25)

    assert skipped.empty
    for _, group in assigned.groupby("sector"):
        assert set(group["quantile"].dropna().astype(int)) == {1, 2, 3, 4, 5}


def test_time_neutral_quantiles_are_assigned_within_year():
    pred, data = _frames()
    merged = prepare_evaluation_frame(pred, data)

    assigned, skipped = assign_quantiles(merged, "time_neutral", min_group_size=25)

    assert skipped.empty
    for _, group in assigned.groupby("year"):
        assert set(group["quantile"].dropna().astype(int)) == {1, 2, 3, 4, 5}


def test_long_short_spread_matches_q5_minus_q1_mean_return():
    pred, data = _frames()
    merged = prepare_evaluation_frame(pred, data)
    assigned, _ = assign_quantiles(merged, "global")

    spreads = compute_long_short(assigned, "global", horizons=(5,))
    row = spreads.iloc[0]
    q1 = assigned.loc[assigned["quantile"] == 1, "abnormal_ret_5d"].mean()
    q5 = assigned.loc[assigned["quantile"] == 5, "abnormal_ret_5d"].mean()

    assert row["long_short_spread"] == pytest.approx(q5 - q1)
    assert row["long_short_spread"] > 0


def test_grouped_quantiles_skip_small_groups():
    pred, data = _frames(n=20)
    merged = prepare_evaluation_frame(pred, data)

    assigned, skipped = assign_quantiles(merged, "sector_neutral", min_group_size=25)

    assert assigned["quantile"].isna().all()
    assert len(skipped) == 2
    assert set(skipped["reason"]) == {"too_few_rows"}


def test_summary_handles_multiple_horizons_and_missing_returns():
    pred, data = _frames()
    data.loc[:9, "abnormal_ret_10d"] = np.nan
    merged = prepare_evaluation_frame(pred, data)
    assigned, _ = assign_quantiles(merged, "global")

    summary = summarize_quantiles(assigned, "global", horizons=(5, 10, 20))

    assert set(summary["horizon"]) == {5, 10, 20}
    assert summary.loc[summary["horizon"] == 10, "n"].sum() == len(data) - 10


def test_run_quantile_evaluation_writes_expected_outputs(tmp_path):
    pred, data = _frames()
    pred_path = tmp_path / "pred.csv"
    data_path = tmp_path / "data.csv"
    out_dir = tmp_path / "quantile_eval"
    pred.to_csv(pred_path, index=False)
    data.to_csv(data_path, index=False)

    paths = run_quantile_evaluation(
        pred_path=str(pred_path),
        data_path=str(data_path),
        out_dir=str(out_dir),
        horizons=(5, 10, 20),
        min_group_size=25,
    )

    assert (out_dir / "global_quantile_summary.csv").exists()
    assert (out_dir / "sector_neutral_quantile_summary.csv").exists()
    assert (out_dir / "time_neutral_quantile_summary.csv").exists()
    assert (out_dir / "long_short_summary.csv").exists()
    assert (out_dir / "quantile_interpretation.md").exists()
    assert paths["interpretation"].endswith("quantile_interpretation.md")
    assert "sector-adjusted abnormal returns" in (out_dir / "quantile_interpretation.md").read_text()

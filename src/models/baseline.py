"""
baseline.py v4

Key fixes vs v3:
1. filing_day_return added to FEATURE_COLS (earnings-surprise proxy)
2. MD&A features added to FEATURE_COLS and ablation
3. finbert_cosine_sim added to FEATURE_COLS and ablation
4. Ablation p-values computed; Benjamini-Hochberg FDR correction applied
5. Test predictions saved to models/test_predictions.csv for CAAR analysis
6. CAAR event study plot produced at end via eda.plot_caar()
"""
import os, logging, warnings, copy
from contextlib import contextmanager
from math import prod
from time import perf_counter
import numpy as np
import pandas as pd
import joblib
import hydra
from tqdm.auto import tqdm
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report, f1_score)
from scipy.stats import binomtest
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception: XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except Exception: LGBM_AVAILABLE = False

try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except Exception: STATSMODELS_AVAILABLE = False

class TqdmLoggingHandler(logging.Handler):
    """Write log lines without corrupting active tqdm progress bars."""

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

REGIME_MAP = {
    2015:"calm",2016:"calm",2017:"bull",2018:"volatile",2019:"bull",
    2020:"crisis",2021:"recovery",2022:"volatile",2023:"recovery",2024:"bull",
}

FEATURE_COLS = [
    "cosine_sim_prev","risk_drift_4q","filing_surprise","sector_contagion",
    "lm_negative","lm_positive","lm_uncertainty","lm_litigious","lm_constraining",
    "lm_net_sentiment","lm_negative_delta","lm_uncertainty_delta","lm_litigious_delta",
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
]


def log_section(title):
    logger.info("")
    logger.info("=" * 72)
    logger.info(title)
    logger.info("=" * 72)


@contextmanager
def logged_stage(title):
    start = perf_counter()
    log_section(title)
    try:
        yield
    except Exception:
        logger.exception("FAILED %s after %.1fs", title, perf_counter() - start)
        raise
    else:
        logger.info("Completed %s in %.1fs", title, perf_counter() - start)


def grid_size(grid):
    return prod(len(v) for v in grid.values()) if grid else 0


def class_balance(values):
    s = pd.Series(values).dropna()
    counts = s.value_counts().sort_index()
    total = len(s)
    if total == 0:
        return "empty"
    return ", ".join(f"{int(k)}={int(v)} ({v/total:.1%})" for k, v in counts.items())


def date_range_text(df):
    if df.empty:
        return "empty"
    return f"{df['filed_at'].min().date()} to {df['filed_at'].max().date()}"


def log_dataset_diagnostics(df, df_clean, avail, primary_target):
    missing_features = [f for f in FEATURE_COLS if f not in df.columns]
    logger.info(
        "Raw dataset: rows=%d tickers=%d date_range=%s",
        len(df), df["ticker"].nunique(), date_range_text(df),
    )
    logger.info(
        "Clean dataset: rows=%d tickers=%d dropped_rows=%d target=%s",
        len(df_clean), df_clean["ticker"].nunique(), len(df) - len(df_clean), primary_target,
    )
    logger.info("Target balance clean: %s", class_balance(df_clean[primary_target]))
    logger.info("Available features: %d/%d", len(avail), len(FEATURE_COLS))
    if missing_features:
        logger.info("Missing feature columns ignored: %s", ", ".join(missing_features))
    for col in ["sector", "benchmark_etf"]:
        if col in df_clean.columns:
            top = df_clean[col].value_counts().head(8).to_dict()
            logger.info("%s distribution top values: %s", col, top)


def log_split_summary(train_df, test_df, primary_target, split_date):
    logger.info("Split date: %s", split_date.date())
    logger.info(
        "Train: rows=%d tickers=%d date_range=%s class_balance=%s",
        len(train_df), train_df["ticker"].nunique(), date_range_text(train_df),
        class_balance(train_df[primary_target]),
    )
    logger.info(
        "Test:  rows=%d tickers=%d date_range=%s class_balance=%s",
        len(test_df), test_df["ticker"].nunique(), date_range_text(test_df),
        class_balance(test_df[primary_target]),
    )


def log_model_result(name, threshold, acc, f1, p_value, y_true, pred):
    sig = "*SIGNIFICANT*" if p_value < 0.05 else ""
    logger.info(
        "%-10s threshold=%.2f accuracy=%.4f macro_f1=%.4f p_vs_majority=%.4f %s",
        name.upper(), threshold, acc, f1, p_value, sig,
    )
    logger.info("  predictions: %s", class_balance(pred))
    logger.info("  truth:       %s", class_balance(y_true))


def rolling_walk_forward_cv(df, feature_cols, estimator, label, window_years=3):
    """Rolling window walk-forward: train on 3-year window, test on next year."""
    df = df.copy()
    df["year"] = pd.to_datetime(df["filed_at"]).dt.year
    avail = [f for f in feature_cols if f in df.columns]
    folds = []
    years = sorted(df["year"].unique())
    for i, ty in enumerate(years):
        if i < window_years: continue
        train_years = years[i-window_years:i]
        tr = df[df["year"].isin(train_years)].dropna(subset=avail+["target"])
        te = df[df["year"]==ty].dropna(subset=avail+["target"])
        if len(tr) < 80 or len(te) < 10: continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr[avail]); Xte = sc.transform(te[avail])
        m = copy.deepcopy(estimator)
        m.fit(Xtr, tr["target"])
        pred = m.predict(Xte)
        acc = accuracy_score(te["target"], pred)
        f1  = f1_score(te["target"], pred, average="macro", zero_division=0)
        folds.append({"test_year":ty,"window":str(train_years),"n_train":len(tr),
                      "n_test":len(te),"accuracy":acc,"macro_f1":f1})
        logger.info(f"    {label} roll {ty}: train={len(tr)}({window_years}yr) test={len(te)} acc={acc:.3f} f1={f1:.3f}")
    if not folds: return {}
    rdf = pd.DataFrame(folds)
    return {"folds":folds,"mean_acc":rdf["accuracy"].mean(),"std_acc":rdf["accuracy"].std(),
            "mean_f1":rdf["macro_f1"].mean(),"std_f1":rdf["macro_f1"].std()}


def tune_model(X, y, model_type, seed):
    tscv = TimeSeriesSplit(n_splits=4)
    if model_type == "lr":
        pipe = Pipeline([("sc",StandardScaler()),
                         ("clf",LogisticRegression(random_state=seed,max_iter=2000))])
        grid = {"clf__C":[0.001,0.005,0.01,0.05,0.1,0.5,1.0],
                "clf__penalty":["l1","l2"],
                "clf__solver":["liblinear"]}
    elif model_type == "rf":
        pipe = Pipeline([("sc",StandardScaler()),
                         ("clf",RandomForestClassifier(random_state=seed))])
        grid = {"clf__n_estimators":[100,200,300],
                "clf__max_depth":[3,5,7,None],
                "clf__min_samples_split":[5,10,20]}
    elif model_type == "xgb" and XGBOOST_AVAILABLE:
        pipe = Pipeline([("sc",StandardScaler()),
                         ("clf",XGBClassifier(random_state=seed,eval_metric="logloss",
                                              use_label_encoder=False,verbosity=0))])
        grid = {"clf__n_estimators":[100,200],
                "clf__max_depth":[3,4,5],
                "clf__learning_rate":[0.05,0.1],
                "clf__subsample":[0.7,0.9],
                "clf__min_child_weight":[3,5]}
    elif model_type == "xgb":
        logger.info("    Skipping XGB: xgboost is not installed/importable")
        return None, 0.0
    elif model_type == "lgbm" and LGBM_AVAILABLE:
        pipe = Pipeline([("sc",StandardScaler()),
                         ("clf",lgb.LGBMClassifier(random_state=seed,verbose=-1))])
        grid = {"clf__n_estimators":[100,200],
                "clf__max_depth":[3,5],
                "clf__learning_rate":[0.05,0.1],
                "clf__num_leaves":[15,31]}
    elif model_type == "lgbm":
        logger.info("    Skipping LGBM: lightgbm is not installed/importable")
        return None, 0.0
    else:
        logger.info("    Skipping %s: unknown model type", model_type)
        return None, 0.0
    candidates = grid_size(grid)
    logger.info(
        "    Grid search: train_rows=%d features=%d candidates=%d cv_splits=%d total_fits=%d",
        X.shape[0], X.shape[1], candidates, tscv.get_n_splits(), candidates * tscv.get_n_splits(),
    )
    start = perf_counter()
    gs = GridSearchCV(pipe,grid,cv=tscv,scoring="f1_macro",n_jobs=-1,refit=True,verbose=0)
    gs.fit(X, y)
    logger.info(
        "    Best %s: cv_f1=%.4f elapsed=%.1fs params=%s",
        model_type, gs.best_score_, perf_counter() - start, gs.best_params_,
    )
    return gs.best_estimator_, gs.best_score_


def optimal_threshold(estimator, X_val, y_val):
    """Find threshold that maximizes macro F1 on validation set."""
    try:
        proba = estimator.predict_proba(X_val)[:,1]
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.3, 0.7, 0.02):
            pred = (proba >= t).astype(int)
            f1 = f1_score(y_val, pred, average="macro", zero_division=0)
            if f1 > best_f1: best_f1=f1; best_t=t
        return best_t
    except: return 0.5


def show_lr_coefficients(estimator, feature_cols):
    """Display LR feature weights — shows DIRECTION of effect, not just magnitude."""
    try:
        clf = estimator.named_steps["clf"]
        coefs = clf.coef_[0]
        pairs = sorted(zip(feature_cols, coefs), key=lambda x: -abs(x[1]))
        logger.info("\nLR Coefficients (positive=predicts up, negative=predicts down):")
        for f, c in pairs[:15]:
            direction = "UP " if c > 0 else "DWN"
            bar = "#" * int(abs(c) * 30)
            logger.info(f"  [{direction}] {f:30s} {c:+.4f}  {bar}")
    except Exception as e:
        logger.warning(f"Could not extract LR coefficients: {e}")


def mcnemar_test(y_true, y_pred, baseline_pred):
    b = ((np.array(y_pred)!=y_true) & (baseline_pred==y_true)).sum()
    c = ((np.array(y_pred)==y_true) & (baseline_pred!=y_true)).sum()
    if b+c == 0: return 1.0
    return binomtest(int(c), int(b+c), 0.5, alternative="greater").pvalue


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    seed = cfg.seed
    data_path = os.path.join(cfg.data.processed_dir, "filing_aligned.csv")
    logger.info("Baseline run starting | seed=%s | xgboost=%s | lightgbm=%s | statsmodels=%s",
                seed, XGBOOST_AVAILABLE, LGBM_AVAILABLE, STATSMODELS_AVAILABLE)

    # Primary target: 5d abnormal from t+1
    primary_target = "target"

    with logged_stage("LOAD AND VALIDATE DATA"):
        logger.info("Reading dataset: %s", data_path)
        df = pd.read_csv(data_path, parse_dates=["filed_at"])
        avail = [f for f in FEATURE_COLS if f in df.columns]
        if primary_target not in df.columns:
            logger.error("Target column %s is missing from %s", primary_target, data_path)
            return
        df["year"] = df["filed_at"].dt.year
        df_clean = df.dropna(subset=avail + [primary_target]).sort_values("filed_at").reset_index(drop=True)
        log_dataset_diagnostics(df, df_clean, avail, primary_target)

    if df_clean.empty:
        logger.error("No rows remain after feature/target cleaning; stopping baseline run.")
        return

    split_date = pd.Timestamp("2023-01-01")
    with logged_stage("TEMPORAL HOLDOUT SPLIT"):
        train_df = df_clean[df_clean["filed_at"] < split_date]
        test_df  = df_clean[df_clean["filed_at"] >= split_date]
        log_split_summary(train_df, test_df, primary_target, split_date)

    if train_df.empty or test_df.empty:
        logger.error("Train/test split produced an empty side; stopping baseline run.")
        return

    y_train = train_df[primary_target].values
    y_test  = test_df[primary_target].values

    majority = int(pd.Series(y_train).mode()[0])
    baseline_pred = np.full(len(y_test), majority)
    logger.info(
        "Majority baseline: class=%d accuracy=%.4f test_predictions=%s",
        majority, accuracy_score(y_test, baseline_pred), class_balance(baseline_pred),
    )

    tuned = {}
    with logged_stage("HYPERPARAMETER TUNING"):
        for mt in tqdm(["lr", "rf", "xgb", "lgbm"], desc="tuning models", unit="model", dynamic_ncols=True):
            logger.info("Tuning %s", mt.upper())
            est, cv_f1 = tune_model(train_df[avail].values, y_train, mt, seed)
            if est is not None:
                tuned[mt] = {"estimator": est, "cv_f1": cv_f1}
        logger.info("Tuned models available: %s", ", ".join(tuned.keys()) if tuned else "none")

    if not tuned:
        logger.error("No models were successfully tuned; stopping baseline run.")
        return

    # ── Holdout evaluation with optimal threshold ──────────────────────
    val_split = pd.Timestamp("2022-01-01")
    val_df = train_df[train_df["filed_at"] >= val_split]
    logger.info(
        "Validation window for threshold search: rows=%d date_range=%s class_balance=%s",
        len(val_df), date_range_text(val_df), class_balance(val_df[primary_target]) if not val_df.empty else "empty",
    )

    results = {}
    with logged_stage("HOLDOUT RESULTS WITH OPTIMAL THRESHOLD"):
        for name, info in tqdm(tuned.items(), desc="holdout eval", unit="model", dynamic_ncols=True):
            est = info["estimator"]
            thresh = optimal_threshold(est, val_df[avail].values, val_df[primary_target].values)
            try:
                proba = est.predict_proba(test_df[avail].values)[:,1]
                pred  = (proba >= thresh).astype(int)
            except Exception:
                pred = est.predict(test_df[avail].values)
                proba = pred.astype(float)
                thresh = 0.5
            acc = accuracy_score(y_test, pred)
            f1  = f1_score(y_test, pred, average="macro", zero_division=0)
            p   = mcnemar_test(y_test, pred, baseline_pred)
            results[name] = {"pred":pred,"proba":proba,"acc":acc,"f1":f1,"p_val":p,"thresh":thresh}
            log_model_result(name, thresh, acc, f1, p, y_test, pred)
            logger.info(classification_report(y_test, pred, zero_division=0))

    os.makedirs("models", exist_ok=True)

    # ── Show LR coefficients ───────────────────────────────────────────
    if "lr" in tuned:
        show_lr_coefficients(tuned["lr"]["estimator"], avail)

    # ── Soft voting ensemble ───────────────────────────────────────────
    with logged_stage("SOFT VOTING ENSEMBLE"):
        all_proba = []
        ensemble_members = []
        for name, info in tuned.items():
            try:
                p = info["estimator"].predict_proba(test_df[avail].values)
                all_proba.append(p)
                ensemble_members.append(name)
                logger.info("Added %s to ensemble", name)
            except Exception as e:
                logger.info("Skipping %s in ensemble: predict_proba unavailable (%s)", name, e)
        if len(all_proba) >= 2:
            avg_p   = np.mean(all_proba, axis=0)
            ens_t   = optimal_threshold(tuned.get("lr",{}).get("estimator",None),
                                         val_df[avail].values,
                                         val_df[primary_target].values) if "lr" in tuned else 0.5
            ens_pred = (avg_p[:,1] >= ens_t).astype(int)
            ens_acc  = accuracy_score(y_test, ens_pred)
            ens_f1   = f1_score(y_test, ens_pred, average="macro", zero_division=0)
            ens_p    = mcnemar_test(y_test, ens_pred, baseline_pred)
            results["ensemble"] = {"pred":ens_pred,"proba":avg_p[:,1],"acc":ens_acc,"f1":ens_f1,"p_val":ens_p}
            logger.info("Ensemble members: %s", ", ".join(ensemble_members))
            log_model_result("ensemble", ens_t, ens_acc, ens_f1, ens_p, y_test, ens_pred)
            logger.info(classification_report(y_test, ens_pred, zero_division=0))
        else:
            logger.info("Need at least two probability-capable models; ensemble skipped")

    best_name = max(results, key=lambda k: results[k]["f1"]) if results else None
    with logged_stage("SAVE HOLDOUT PREDICTIONS AND QUANTILE EVAL"):
        if best_name:
            logger.info(
                "Best holdout model by macro_f1: %s accuracy=%.4f macro_f1=%.4f",
                best_name, results[best_name]["acc"], results[best_name]["f1"],
            )
            test_pred_df = test_df[["filed_at","ticker"]].copy()
            test_pred_df["y_true"]  = y_test
            test_pred_df["y_pred"]  = results[best_name]["pred"]
            test_pred_df["y_proba"] = results[best_name]["proba"]
            test_pred_df["model_name"] = best_name
            pred_path = "models/test_predictions.csv"
            test_pred_df.to_csv(pred_path, index=False)
            logger.info("Saved %d test predictions to %s", len(test_pred_df), pred_path)
            try:
                from src.evaluation.quantile_eval import run_quantile_evaluation
                quantile_dir = "outputs/quantile_eval"
                logger.info("Running quantile evaluation -> %s", quantile_dir)
                run_quantile_evaluation(
                    pred_path=pred_path,
                    data_path=os.path.join(cfg.data.processed_dir, "filing_aligned.csv"),
                    out_dir=quantile_dir,
                    horizons=(5, 10, 20),
                )
                logger.info("Quantile evaluation completed: %s", quantile_dir)
            except Exception as e:
                logger.warning(f"Quantile evaluation failed: {e}")
        else:
            logger.warning("No best model found; test_predictions.csv was not written")

    # ── Multi-horizon comparison ───────────────────────────────────────
    with logged_stage("MULTI-HORIZON COMPARISON"):
        best_lr = tuned.get("lr", {}).get("estimator", None)
        if best_lr is not None:
            for h in tqdm([5, 10, 20], desc="horizons", unit="horizon", dynamic_ncols=True):
                tcol = f"target_{h}d"
                if tcol not in df.columns:
                    logger.info("Horizon %sd skipped: %s missing", h, tcol)
                    continue
                dc = df.dropna(subset=avail+[tcol]).sort_values("filed_at").reset_index(drop=True)
                tr2 = dc[dc["filed_at"] < split_date]
                te2 = dc[dc["filed_at"] >= split_date]
                if len(tr2) < 30 or len(te2) < 5:
                    logger.info("Horizon %sd skipped: train=%d test=%d too small", h, len(tr2), len(te2))
                    continue
                sc2 = StandardScaler()
                Xt = sc2.fit_transform(tr2[avail]); Xe = sc2.transform(te2[avail])
                m2 = LogisticRegression(C=0.1, penalty="l1", solver="liblinear",
                                         max_iter=2000, random_state=seed)
                m2.fit(Xt, tr2[tcol].values)
                p2 = m2.predict(Xe)
                acc2 = accuracy_score(te2[tcol].values, p2)
                f12  = f1_score(te2[tcol].values, p2, average="macro", zero_division=0)
                logger.info(
                    "Horizon %2dd: train=%d test=%d accuracy=%.4f macro_f1=%.4f train_balance=%s test_balance=%s",
                    h, len(tr2), len(te2), acc2, f12,
                    class_balance(tr2[tcol]), class_balance(te2[tcol]),
                )
        else:
            logger.info("LR model unavailable; multi-horizon comparison skipped")

    # ── Ablation with FDR correction ───────────────────────────────────
    ablation_results = []
    with logged_stage("ABLATION STUDY (LR + BH FDR CORRECTION)"):
        ablation = {
            "price only":          [f for f in avail if f.startswith("price_")],
            "LM level only":       [f for f in avail if f.startswith("lm_") and not f.endswith("_delta") and "x_" not in f and not f.endswith("_mda")],
            "LM delta only":       [f for f in avail if f.endswith("_delta")],
            "cosine+temporal":     ["cosine_sim_prev","risk_drift_4q","filing_surprise","sector_contagion"],
            "interaction only":    [f for f in avail if "x_" in f or "divergence" in f],
            "text (no price)":     [f for f in avail if not f.startswith("price_")],
            "no interactions":     [f for f in avail if "x_" not in f and "divergence" not in f],
            "MD&A only":           [f for f in avail if f.endswith("_mda")],
            "finbert only":        ["finbert_cosine_sim"] if "finbert_cosine_sim" in avail else [],
            "tfidf vs finbert":    [f for f in avail if f in ("cosine_sim_prev","finbert_cosine_sim")],
            "all features":        avail,
        }

        for gname, feats in tqdm(ablation.items(), desc="ablations", unit="group", dynamic_ncols=True):
            feats = [f for f in feats if f in df_clean.columns]
            if not feats:
                logger.info("Ablation '%s' skipped: no available features", gname)
                continue
            dc = df_clean.dropna(subset=feats+[primary_target])
            tr2 = dc[dc["filed_at"] < split_date]
            te2 = dc[dc["filed_at"] >= split_date]
            if len(tr2) < 30 or len(te2) < 5:
                logger.info("Ablation '%s' skipped: train=%d test=%d too small", gname, len(tr2), len(te2))
                continue
            sc2 = StandardScaler()
            Xt = sc2.fit_transform(tr2[feats]); Xe = sc2.transform(te2[feats])
            m2 = LogisticRegression(C=0.1, penalty="l1", solver="liblinear",
                                     max_iter=2000, random_state=seed)
            m2.fit(Xt, tr2[primary_target].values)
            p2 = m2.predict(Xe)
            acc2 = accuracy_score(te2[primary_target], p2)
            f12  = f1_score(te2[primary_target], p2, average="macro", zero_division=0)
            # McNemar vs majority baseline on this subset's test set
            bpred2 = np.full(len(te2), int(pd.Series(tr2[primary_target]).mode()[0]))
            pval = mcnemar_test(te2[primary_target].values, p2, bpred2)
            ablation_results.append({
                "group": gname, "acc": acc2, "f1": f12, "p_val": pval, "n_feats": len(feats),
            })
            logger.info(
                "Ablation '%s': features=%d train=%d test=%d accuracy=%.4f macro_f1=%.4f p=%.4f",
                gname, len(feats), len(tr2), len(te2), acc2, f12, pval,
            )

        # Apply Benjamini-Hochberg FDR correction
        if ablation_results and STATSMODELS_AVAILABLE:
            raw_pvals = [r["p_val"] for r in ablation_results]
            reject, q_vals, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")
            for i, r in enumerate(ablation_results):
                r["q_val"]   = q_vals[i]
                r["sig_raw"] = "*" if raw_pvals[i] < 0.05 else ""
                r["sig_fdr"] = "*" if reject[i] else ""
        else:
            if not STATSMODELS_AVAILABLE:
                logger.info("statsmodels unavailable; BH-FDR q-values will be NaN")
            for r in ablation_results:
                r["q_val"] = np.nan; r["sig_raw"] = ""; r["sig_fdr"] = ""

        logger.info("")
        logger.info("  %-30s %6s %6s %8s  %8s  n", "Group", "acc", "f1", "p_raw", "q_fdr")
        logger.info("  " + "-"*75)
        for r in ablation_results:
            logger.info(f"  {r['group']:30s} {r['acc']:.4f} {r['f1']:.4f} "
                        f"p={r['p_val']:.4f}{r['sig_raw']:1s}  q={r['q_val']:.4f}{r['sig_fdr']:1s}  n={r['n_feats']}")
        if STATSMODELS_AVAILABLE:
            logger.info("  (* = significant at 0.05; q = BH-FDR corrected p-value)")

    # ── Rolling walk-forward CV (best model = LR) ──────────────────────
    with logged_stage("ROLLING WALK-FORWARD CV (3-YEAR WINDOW)"):
        if "lr" in tuned:
            clf_only = tuned["lr"]["estimator"].named_steps["clf"]
            wf = rolling_walk_forward_cv(df_clean, avail, clf_only, "LR", window_years=3)
            if wf:
                logger.info(f"Rolling CV: acc={wf['mean_acc']:.4f}(+/-{wf['std_acc']:.4f})  "
                            f"f1={wf['mean_f1']:.4f}(+/-{wf['std_f1']:.4f})")
                rolling_path = "models/lr_rolling_cv.csv"
                pd.DataFrame(wf["folds"]).to_csv(rolling_path, index=False)
                logger.info("Saved rolling CV folds to %s", rolling_path)
            else:
                logger.info("Rolling CV produced no valid folds")
        else:
            logger.info("LR model unavailable; rolling CV skipped")

    # ── Final summary ──────────────────────────────────────────────────
    with logged_stage("SAVE MODEL ARTIFACTS AND FINAL SUMMARY"):
        for name, info in tuned.items():
            model_path = f"models/v3_{name}.pkl"
            joblib.dump(info["estimator"], model_path)
            logger.info("Saved fitted %s model to %s", name, model_path)
        summary = pd.DataFrame([
            {"model":k,"accuracy":v["acc"],"macro_f1":v["f1"],"p_vs_baseline":v["p_val"]}
            for k,v in results.items()
        ]).sort_values("macro_f1", ascending=False)
        summary_path = "models/results_summary_v3.csv"
        summary.to_csv(summary_path, index=False)
        logger.info("Saved results summary to %s", summary_path)
        logger.info("\nFINAL RESULTS SUMMARY:\n%s", summary.to_string(index=False))

    # ── CAAR event study plot ──────────────────────────────────────────
    with logged_stage("CAAR EVENT STUDY PLOT"):
        pred_path = "models/test_predictions.csv"
        if os.path.exists(pred_path):
            try:
                import sys
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
                from src.analysis.eda import plot_caar
                test_pred_df = pd.read_csv(pred_path, parse_dates=["filed_at"])
                logger.info("Running CAAR plot from %s -> outputs/eda", pred_path)
                plot_caar(test_pred_df, "outputs/eda",
                          data_start=cfg.data.start_date, data_end=cfg.data.end_date)
                logger.info("CAAR plot step completed")
            except Exception as e:
                logger.warning(f"CAAR plot failed: {e}")
        else:
            logger.info("%s not found; CAAR plot skipped", pred_path)

    logger.info("Baseline run finished successfully")

if __name__ == "__main__":
    main()

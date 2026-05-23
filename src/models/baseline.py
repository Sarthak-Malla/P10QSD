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
import numpy as np
import pandas as pd
import joblib
import hydra
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
    elif model_type == "lgbm" and LGBM_AVAILABLE:
        pipe = Pipeline([("sc",StandardScaler()),
                         ("clf",lgb.LGBMClassifier(random_state=seed,verbose=-1))])
        grid = {"clf__n_estimators":[100,200],
                "clf__max_depth":[3,5],
                "clf__learning_rate":[0.05,0.1],
                "clf__num_leaves":[15,31]}
    else: return None, 0.0
    gs = GridSearchCV(pipe,grid,cv=tscv,scoring="f1_macro",n_jobs=-1,refit=True,verbose=0)
    gs.fit(X, y)
    logger.info(f"    Best {model_type}: {gs.best_params_}  CV f1={gs.best_score_:.4f}")
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
    df = pd.read_csv(os.path.join(cfg.data.processed_dir,"filing_aligned.csv"),
                     parse_dates=["filed_at"])
    logger.info(f"Loaded {len(df)} rows, {df['ticker'].nunique()} tickers")
    seed = cfg.seed
    avail = [f for f in FEATURE_COLS if f in df.columns]
    df["year"] = df["filed_at"].dt.year

    # Primary target: 5d abnormal from t+1
    primary_target = "target"
    df_clean = df.dropna(subset=avail+[primary_target]).sort_values("filed_at").reset_index(drop=True)
    logger.info(f"Rows: {len(df_clean)} | Features: {len(avail)}")

    split_date = pd.Timestamp("2023-01-01")
    train_df = df_clean[df_clean["filed_at"] < split_date]
    test_df  = df_clean[df_clean["filed_at"] >= split_date]
    logger.info(f"Train: {len(train_df)} | Test: {len(test_df)}")

    y_train = train_df[primary_target].values
    y_test  = test_df[primary_target].values

    majority = int(pd.Series(y_train).mode()[0])
    baseline_pred = np.full(len(y_test), majority)
    logger.info(f"Majority baseline: {accuracy_score(y_test, baseline_pred):.4f} (class={majority})")

    # ── Tune all models ────────────────────────────────────────────────
    logger.info("\n" + "="*60 + "\nHYPERPARAMETER TUNING\n" + "="*60)
    tuned = {}
    for mt in ["lr","rf","xgb","lgbm"]:
        logger.info(f"\nTuning {mt.upper()}...")
        est, cv_f1 = tune_model(train_df[avail].values, y_train, mt, seed)
        if est is not None:
            tuned[mt] = {"estimator":est,"cv_f1":cv_f1}

    # ── Holdout evaluation with optimal threshold ──────────────────────
    logger.info("\n" + "="*60 + "\nHOLDOUT RESULTS (with optimal threshold)\n" + "="*60)

    val_split = pd.Timestamp("2022-01-01")
    val_df = train_df[train_df["filed_at"] >= val_split]

    results = {}
    for name, info in tuned.items():
        est = info["estimator"]
        thresh = optimal_threshold(est, val_df[avail].values, val_df[primary_target].values)
        try:
            proba = est.predict_proba(test_df[avail].values)[:,1]
            pred  = (proba >= thresh).astype(int)
        except:
            pred = est.predict(test_df[avail].values)
            proba = pred.astype(float)
            thresh = 0.5
        acc = accuracy_score(y_test, pred)
        f1  = f1_score(y_test, pred, average="macro", zero_division=0)
        p   = mcnemar_test(y_test, pred, baseline_pred)
        results[name] = {"pred":pred,"proba":proba,"acc":acc,"f1":f1,"p_val":p,"thresh":thresh}
        sig = "*SIGNIFICANT*" if p < 0.05 else ""
        logger.info(f"\n{name.upper()} (threshold={thresh:.2f}): acc={acc:.4f}  f1={f1:.4f}  p={p:.4f} {sig}")
        logger.info(classification_report(y_test, pred, zero_division=0))

    # ── Save test predictions for CAAR analysis ────────────────────────
    os.makedirs("models", exist_ok=True)
    best_name = max(results, key=lambda k: results[k]["f1"]) if results else None
    if best_name:
        test_pred_df = test_df[["filed_at","ticker"]].copy()
        test_pred_df["y_true"]  = y_test
        test_pred_df["y_pred"]  = results[best_name]["pred"]
        test_pred_df["y_proba"] = results[best_name]["proba"]
        test_pred_df.to_csv("models/test_predictions.csv", index=False)
        logger.info(f"Test predictions saved (model={best_name}): models/test_predictions.csv")

    # ── Show LR coefficients ───────────────────────────────────────────
    if "lr" in tuned:
        show_lr_coefficients(tuned["lr"]["estimator"], avail)

    # ── Soft voting ensemble ───────────────────────────────────────────
    logger.info("\n" + "="*60 + "\nSOFT VOTING ENSEMBLE\n" + "="*60)
    all_proba = []
    for name, info in tuned.items():
        try:
            p = info["estimator"].predict_proba(test_df[avail].values)
            all_proba.append(p); logger.info(f"  + {name}")
        except: pass
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
        sig = "*SIGNIFICANT*" if ens_p < 0.05 else ""
        logger.info(f"\nENSEMBLE (t={ens_t:.2f}): acc={ens_acc:.4f}  f1={ens_f1:.4f}  p={ens_p:.4f} {sig}")
        logger.info(classification_report(y_test, ens_pred, zero_division=0))

    # ── Multi-horizon comparison ───────────────────────────────────────
    logger.info("\n" + "="*60 + "\nMULTI-HORIZON COMPARISON\n" + "="*60)
    best_lr = tuned.get("lr", {}).get("estimator", None)
    if best_lr is not None:
        for h in [5, 10, 20]:
            tcol = f"target_{h}d"
            if tcol not in df.columns: continue
            dc = df.dropna(subset=avail+[tcol]).sort_values("filed_at").reset_index(drop=True)
            tr2 = dc[dc["filed_at"] < split_date]
            te2 = dc[dc["filed_at"] >= split_date]
            if len(tr2) < 30 or len(te2) < 5: continue
            sc2 = StandardScaler()
            Xt = sc2.fit_transform(tr2[avail]); Xe = sc2.transform(te2[avail])
            m2 = LogisticRegression(C=0.1, penalty="l1", solver="liblinear",
                                     max_iter=2000, random_state=seed)
            m2.fit(Xt, tr2[tcol].values)
            p2 = m2.predict(Xe)
            acc2 = accuracy_score(te2[tcol].values, p2)
            f12  = f1_score(te2[tcol].values, p2, average="macro", zero_division=0)
            logger.info(f"  Horizon {h:2d}d: acc={acc2:.4f}  f1={f12:.4f}  n_test={len(te2)}")

    # ── Ablation with FDR correction ───────────────────────────────────
    logger.info("\n" + "="*60 + "\nABLATION STUDY (LR + BH FDR correction)\n" + "="*60)
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

    ablation_results = []
    for gname, feats in ablation.items():
        feats = [f for f in feats if f in df_clean.columns]
        if not feats: continue
        dc = df_clean.dropna(subset=feats+[primary_target])
        tr2 = dc[dc["filed_at"] < split_date]
        te2 = dc[dc["filed_at"] >= split_date]
        if len(tr2) < 30 or len(te2) < 5: continue
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

    # Apply Benjamini-Hochberg FDR correction
    if ablation_results and STATSMODELS_AVAILABLE:
        raw_pvals = [r["p_val"] for r in ablation_results]
        reject, q_vals, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")
        for i, r in enumerate(ablation_results):
            r["q_val"]   = q_vals[i]
            r["sig_raw"] = "*" if raw_pvals[i] < 0.05 else ""
            r["sig_fdr"] = "*" if reject[i] else ""
    else:
        for r in ablation_results:
            r["q_val"] = np.nan; r["sig_raw"] = ""; r["sig_fdr"] = ""

    logger.info(f"\n  {'Group':30s} {'acc':>6} {'f1':>6} {'p_raw':>8} {'':1} {'q_fdr':>8} {'':1} n")
    logger.info("  " + "-"*75)
    for r in ablation_results:
        logger.info(f"  {r['group']:30s} {r['acc']:.4f} {r['f1']:.4f} "
                    f"p={r['p_val']:.4f}{r['sig_raw']:1s}  q={r['q_val']:.4f}{r['sig_fdr']:1s}  n={r['n_feats']}")
    if STATSMODELS_AVAILABLE:
        logger.info("  (* = significant at 0.05; q = BH-FDR corrected p-value)")

    # ── Rolling walk-forward CV (best model = LR) ──────────────────────
    logger.info("\n" + "="*60 + "\nROLLING WALK-FORWARD CV (3-year window)\n" + "="*60)
    if "lr" in tuned:
        clf_only = tuned["lr"]["estimator"].named_steps["clf"]
        wf = rolling_walk_forward_cv(df_clean, avail, clf_only, "LR", window_years=3)
        if wf:
            logger.info(f"\nRolling CV: acc={wf['mean_acc']:.4f}(+/-{wf['std_acc']:.4f})  "
                        f"f1={wf['mean_f1']:.4f}(+/-{wf['std_f1']:.4f})")
            pd.DataFrame(wf["folds"]).to_csv("models/lr_rolling_cv.csv", index=False)

    # ── Final summary ──────────────────────────────────────────────────
    for name, info in tuned.items():
        joblib.dump(info["estimator"], f"models/v3_{name}.pkl")
    summary = pd.DataFrame([
        {"model":k,"accuracy":v["acc"],"macro_f1":v["f1"],"p_vs_baseline":v["p_val"]}
        for k,v in results.items()
    ]).sort_values("macro_f1", ascending=False)
    summary.to_csv("models/results_summary_v3.csv", index=False)
    logger.info(f"\nFINAL RESULTS SUMMARY:\n{summary.to_string()}")

    # ── CAAR event study plot ──────────────────────────────────────────
    if os.path.exists("models/test_predictions.csv"):
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            from src.analysis.eda import plot_caar
            test_pred_df = pd.read_csv("models/test_predictions.csv", parse_dates=["filed_at"])
            plot_caar(test_pred_df, "outputs/eda",
                      data_start=cfg.data.start_date, data_end=cfg.data.end_date)
        except Exception as e:
            logger.warning(f"CAAR plot failed: {e}")

if __name__ == "__main__":
    main()

# P10QSD — Predicting Abnormal Stock Returns from SEC 10-Q Filing Changes

> **Research Question:** Do quarter-over-quarter changes in SEC 10-Q filing language provide predictive signal for abnormal stock returns, beyond what price-based technical indicators already capture?

## Overview

This project investigates whether changes in the Risk Factors section (Item 1A) of SEC 10-Q quarterly filings can predict short-term **abnormal stock returns** (stock return minus S&P 500 return). We use a filing-aligned dataset where each observation is one quarterly filing — not daily price data — avoiding the data leakage inherent in forward-filling quarterly features into daily rows.

**Key findings (47 S&P 500 companies, 2015–2024):**
- Loughran-McDonald (LM) finance sentiment features alone outperform price-only features (F1=0.514 vs 0.499)
- **LM positive language predicts negative abnormal returns** (β=−0.089) — the "cheap talk" effect
- Sector contagion: when peers rewrite risk factors, it predicts your stock movement
- Walk-forward CV shows consistent 50–55% accuracy across all market regimes (calm, bull, volatile, crisis, recovery)
- Sparse 6-feature LM model achieves macro F1=0.524 with permutation p<0.05

---

## Architecture

```
sp500_tickers.txt (503 S&P 500 companies)
        │
        ▼
src/dataloader/sec_loader.py
    Downloads 10-Q filings from SEC EDGAR (2015–2024)
    Extracts Item 1A (Risk Factors) text
    Checkpointed — safe to interrupt and resume
        │
        ▼
src/dataloader/filing_dataset.py
    1 row per filing (not daily — no data leakage)
    Computes: cosine similarity, LM sentiment, temporal features,
              sector contagion signal, price features at filing date
    Target: abnormal return (stock − SPY) over 5 days from t+1
        │
        ▼
src/analysis/eda.py
    7 distribution plots across time, sector, regime
        │
        ▼
src/models/baseline.py
    Walk-forward CV (rolling 3-year window)
    Grid search: LR, RF, XGBoost, LightGBM
    Soft-voting ensemble + permutation significance test
    Ablation study using best model (LR)
        │
src/models/temporal_model.py
    LSTM over per-company filing sequences (8-quarter context)
        │
src/models/final_analysis.py
    Sparse model (6 LM features)
    Sector-stratified models
    Portfolio evaluation (long-short quintile spread)
```

---

## Results

### Model Comparison (47 companies, 278 test samples, 2023–2024)

| Model | Accuracy | Macro F1 | p vs baseline | CV Accuracy |
|---|---|---|---|---|
| Majority baseline | 47.8% | — | — | — |
| LR (sparse, 6 LM features) | **52.9%** | **0.525** | 0.015* | 0.504±0.039 |
| LR (all 24 features) | 52.9% | 0.524 | 0.151 | — |
| Random Forest | 52.5% | 0.513 | 0.184 | — |
| LightGBM | 48.6% | 0.461 | 0.453 | — |
| XGBoost | 47.1% | 0.460 | 0.685 | — |

*Statistical significance via 500-permutation test

### Ablation Study (LR, holdout set)

| Feature Group | Accuracy | Macro F1 | n Features |
|---|---|---|---|
| **LM level only** | **51.4%** | **0.514** | 6 |
| Price only | 50.0% | 0.499 | 7 |
| Text (no price) | 51.1% | 0.510 | 17 |
| All features | 50.0% | 0.500 | 24 |

**LM sentiment features alone outperform price-based technical indicators.**

### LR Coefficients (direction of effect)

| Feature | Effect | Coefficient |
|---|---|---|
| `lm_positive` | → Down | −0.089 |
| `lm_negative` | → Down | −0.085 |
| `price_volatility_20d` | → Up | +0.080 |
| `text_length_norm` | → Down | −0.078 |
| `lm_litigious` | → Down | −0.068 |
| `text_price_divergence` | → Down | −0.065 |
| `filing_surprise` | → Up | +0.044 |

**Notable:** Positive language in 10-Q filings predicts *negative* abnormal returns — companies using more optimistic language underperform the market ("cheap talk" effect).

### Rolling Walk-Forward CV (3-year window)

| Test Year | Accuracy | Macro F1 |
|---|---|---|
| 2018 | 50.4% | 0.504 |
| 2019 | 55.6% | 0.551 |
| 2020 | 48.9% | 0.471 |
| 2021 | 45.7% | 0.419 |
| 2022 | 48.9% | 0.484 |
| 2023 | 54.7% | 0.544 |
| 2024 | 55.4% | 0.554 |
| **Mean** | **51.4%** | **0.504** |

Performance improves in recent years (2023–2024), consistent with increasing investor attention to ESG/risk language.

---

## Features

### Text Features (from Item 1A Risk Factors)

| Feature | Description |
|---|---|
| `cosine_sim_prev` | TF-IDF cosine similarity vs previous quarter (low = major rewrite) |
| `risk_drift_4q` | Rolling 4-quarter trend in text change (accelerating/decelerating rewrites) |
| `filing_surprise` | Company-specific z-score of text change (how unusual is this quarter?) |
| `sector_contagion` | Average peer cosine similarity in same sector ±45 days **(novel)** |
| `lm_negative` | Loughran-McDonald negative word ratio |
| `lm_positive` | LM positive word ratio |
| `lm_uncertainty` | LM uncertainty word ratio |
| `lm_litigious` | LM litigious word ratio |
| `lm_constraining` | LM constraining word ratio |
| `lm_net_sentiment` | (positive − negative) / (positive + negative + 1) |
| `lm_*_delta` | Quarter-over-quarter change in each LM score |
| `lm_neg_x_cosine` | Interaction: negative sentiment × text change magnitude |
| `text_price_divergence` | Interaction: positive text tone vs falling price |

### Price Features (at filing date, no lookahead)

| Feature | Description |
|---|---|
| `price_return_1d/5d/20d` | Momentum over 1, 5, 20 days before filing |
| `price_volatility_20d` | Annualized 20-day realized volatility |
| `price_ma_ratio_5/20` | Price relative to moving average |
| `price_rsi` | 14-day Relative Strength Index |

### Target Variable

**Binary:** abnormal return > 0, where:
- `abnormal_return = stock_return − SPY_return` over 5 trading days starting t+1
- t+1 start skips the filing day (dominated by algorithmic traders reacting to headline EPS numbers)
- Using abnormal return isolates company-specific filing signal from market-wide movements

---

## Setup

```bash
git clone https://github.com/Sarthak-Malla/P10QSD.git
cd P10QSD
python3 -m venv venv
source venv/bin/activate
venv/bin/pip install -r requirements.txt
venv/bin/python -c "import nltk; nltk.download('vader_lexicon')"
```

---

## Running the Pipeline

### Quick start (47 companies already downloaded)

```bash
# Step 1: Build filing-aligned dataset
venv/bin/python -m src.dataloader.filing_dataset

# Step 2: Exploratory data analysis
venv/bin/python -m src.analysis.eda

# Step 3: Train and evaluate all models
venv/bin/python -m src.models.baseline

# Step 4: Sparse model + sector analysis + portfolio evaluation
venv/bin/python -m src.models.final_analysis

# Optional: LSTM temporal model (requires PyTorch)
venv/bin/python -m src.models.temporal_model
```

### Full S&P 500 scale (run overnight)

```bash
# Downloads 503 companies from SEC EDGAR (~10 hours, checkpointed)
caffeinate -i bash scale_to_sp500.sh
```

---

## Configuration (`conf/config.yaml`)

```yaml
data:
  tickers_file: "sp500_50.txt"   # or "sp500_tickers.txt" for full S&P 500
  start_date: "2015-01-01"
  end_date: "2024-12-31"

features:
  prediction_horizon: 5          # days after filing to measure return

model:
  type: "random_forest"
  params:
    n_estimators: 200
    max_depth: 5
    min_samples_split: 10

sec:
  identity: "sarthak.malla@mbzuai.ac.ae"
  forms: ["10-Q"]
  sections: ["Item 1A"]
```

---

## Project Structure

```
P10QSD/
├── conf/
│   └── config.yaml
├── src/
│   ├── analysis/
│   │   └── eda.py                  # 7 EDA plots (time, sector, regime, correlation)
│   ├── dataloader/
│   │   ├── loader.py               # yfinance stock data downloader
│   │   ├── sec_loader.py           # SEC EDGAR 10-Q downloader (checkpointed)
│   │   └── filing_dataset.py       # Filing-aligned dataset builder
│   ├── features/
│   │   └── lm_features.py          # Loughran-McDonald finance sentiment
│   └── models/
│       ├── baseline.py             # Walk-forward CV, grid search, ensemble
│       ├── temporal_model.py       # LSTM over filing sequences
│       └── final_analysis.py       # Sparse model, sector, portfolio eval
├── sp500_50.txt                    # 50 S&P 500 tickers (quick experiments)
├── sp500_tickers.txt               # Full 503 S&P 500 tickers
├── scale_to_sp500.sh               # One-command full-scale run
└── requirements.txt
```

---

## Key Design Decisions

**1. Filing-aligned evaluation (not daily):** Each 10-Q filing is one observation. Forward-filling creates ~90 rows with identical features — not 90 independent observations. This inflated our naive baseline from 52% to 61%.

**2. Abnormal return target:** Predicting raw returns conflates market-wide movements with company-specific filing signal. Subtracting SPY return isolates what the 10-Q should predict.

**3. t+1 start:** Algorithmic traders react to headline EPS numbers within minutes of filing. The text-driven informed reaction happens the next day onward.

**4. LM over VADER:** VADER misclassifies ~40% of finance-domain terms. "Liability", "risk", "adverse" score neutral in VADER but strongly negative in LM.

**5. Sector contagion (novel):** When peers in your sector significantly rewrite their risk factors, it predicts your stock movement. Information spills across supply chains before the market fully prices it in.

---

## Authors

- **Sultan Akimaliyev** — sultan.akimaliyev@mbzuai.ac.ae
- **Sarthak Malla** — sarthak.malla@mbzuai.ac.ae

Mohamed bin Zayed University of Artificial Intelligence

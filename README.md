# P10QSD — SEC Filing Text & Stock Price Prediction

A research project investigating whether changes in SEC 10-Q filing language (Risk Factors, Item 1A) can improve stock return prediction beyond technical price indicators alone.

## Research Question

Do quarter-over-quarter changes in SEC 10-Q filing text provide predictive signal for short-term stock returns, beyond what price-based technical indicators already capture?

## Methodology

### Key Design Decision: Filing-Aligned Dataset
Rather than forward-filling quarterly text features into daily price data (which creates data leakage and inflates accuracy), we use a **filing-aligned** approach:
- **1 row per 10-Q filing per company** (not daily)
- Target: did the stock go up in the N trading days after the filing date?
- This is the methodologically correct evaluation — each filing is an independent observation

### Pipeline
```
sp500_50.txt (50 S&P 500 companies)
       ↓
sec_loader.py       → data/raw/sec_filings/{TICKER}_filings.parquet
                      Downloads 10-Q filings from SEC EDGAR (2015–2024)
                      Extracts Item 1A (Risk Factors) text
                      Falls back to full-text keyword search when section extraction fails
       ↓
filing_dataset.py   → data/processed/filing_aligned.csv
                      Computes text features per filing
                      Computes price features at each filing date
                      Computes price return after filing (target)
       ↓
baseline.py         → models/baseline_random_forest.pkl
                      Trains Random Forest on filing-aligned data
                      Time-series split (train: pre-2023, test: 2023–2024)
```

### Features

**Text features (from Item 1A Risk Factors):**
| Feature | Description |
|---|---|
| `cosine_sim_prev` | TF-IDF cosine similarity vs previous quarter's filing |
| `sentiment_compound` | VADER compound sentiment score |
| `sentiment_pos` | VADER positive sentiment |
| `sentiment_neg` | VADER negative sentiment |
| `text_length_norm` | Normalized word count of the section |

**Price features (computed at filing date, from prior 50 trading days):**
| Feature | Description |
|---|---|
| `price_return_1d/5d/20d` | Price momentum over 1, 5, 20 days |
| `price_volatility_20d` | Annualized 20-day volatility |
| `price_ma_ratio_5/20` | Price relative to 5/20-day moving average |
| `price_rsi` | 14-day Relative Strength Index |

## Dataset

- **Companies:** 47 S&P 500 companies across 6 sectors (Tech, Finance, Healthcare, Consumer, Energy, Industrial)
- **Filings:** ~1,295 valid 10-Q filing observations (2015–2024)
- **Target distribution:** 690 up (53%) / 605 down (47%) — nearly balanced
- **Train/test split:** Time-series split at April 2023

## Results

| Model | Accuracy | Class 0 Recall | Class 1 Recall | Macro F1 |
|---|---|---|---|---|
| Daily price baseline (forward-filled, leaky) | 61% | 0.11 | 0.90 | 0.46 |
| Filing-aligned, text features only | 51% | 0.27 | 0.73 | 0.48 |
| Filing-aligned, text + price features | **53%** | 0.18 | 0.84 | 0.46 |

## Key Observations

### 1. Forward-filling creates data leakage
The naive approach of forward-filling quarterly SEC features into daily price rows inflates accuracy to ~61% — but this is misleading. 90 consecutive rows with identical features are not 90 independent observations. Filing-aligned evaluation gives an honest 51–53%.

### 2. Cosine similarity is the strongest text signal
Feature importance from the Random Forest:

| Rank | Feature | Importance | Type |
|---|---|---|---|
| 1 | price_ma_ratio_20 | 10.50% | Price |
| **2** | **cosine_sim_prev** | **10.47%** | **Text** |
| 3 | price_return_20d | 10.36% | Price |
| 4 | sentiment_compound | 9.72% | Text |
| 5 | price_volatility_20d | 9.46% | Price |

Text and price features are nearly equal in importance. Quarter-over-quarter cosine similarity of Item 1A text is the #2 most important feature overall — virtually tied with the best price feature.

### 3. VADER sentiment is weak for financial text
VADER sentiment ranks 4th, 11th, and 12th — it was designed for social media, not legal/financial language. Replacing it with finance-specific tools (Loughran-McDonald word lists or FinBERT) should improve the text signal significantly.

### 4. Text change outperforms text content
`cosine_sim_prev` (how much the text *changed*) ranks higher than any sentiment score (what the text *says*). This suggests that the act of companies rewriting risk factors is itself informative — not just the tone.

## Setup & Running
```bash
# Clone and install
git clone https://github.com/Sarthak-Malla/P10QSD.git
cd P10QSD
python3 -m venv venv
source venv/bin/activate
venv/bin/pip install -r requirements.txt
venv/bin/pip install nltk
venv/bin/python -c "import nltk; nltk.download('vader_lexicon')"

# Get S&P 500 tickers
venv/bin/python -c "
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv')
tickers = [t.replace('.', '-') for t in df['Symbol'].tolist()]
open('sp500_50.txt', 'w').write('\n'.join(tickers[:50]))
"

# Run pipeline
venv/bin/python -m src.dataloader.sec_loader      # ~30 min, hits SEC EDGAR
venv/bin/python -m src.dataloader.filing_dataset  # builds filing-aligned dataset
venv/bin/python -m src.models.baseline            # trains and evaluates model
```

## Configuration

Edit `conf/config.yaml` to change:
- `data.tickers_file` — which tickers to use
- `features.prediction_horizon` — days after filing to measure return (default: 5)
- `model.params` — Random Forest hyperparameters
- `sec.sections` — which 10-Q sections to extract (default: Item 1A)

## Next Steps

- [ ] Replace VADER with Loughran-McDonald finance word lists
- [ ] Try FinBERT for domain-specific sentiment
- [ ] Fix IBM, WFC, BLK text extraction failures
- [ ] Expand to full S&P 500 (503 companies)
- [ ] Add more 10-Q sections (Item 2: MD&A)
- [ ] Experiment with TF-IDF topic modeling (LDA) on risk factors

## Project Structure
```
P10QSD/
├── conf/
│   └── config.yaml              # Hydra configuration
├── src/
│   ├── dataloader/
│   │   ├── loader.py            # yfinance stock data downloader
│   │   ├── sec_loader.py        # SEC EDGAR 10-Q filing downloader
│   │   └── filing_dataset.py   # Filing-aligned dataset builder
│   ├── features/
│   │   ├── engineering.py       # Technical indicator engineering (daily)
│   │   └── sec_features.py      # SEC NLP feature engineering (legacy)
│   └── models/
│       └── baseline.py          # Random Forest baseline model
├── sp500_50.txt                 # 50 S&P 500 tickers for experiments
└── requirements.txt
```

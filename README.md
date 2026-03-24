# P10QSD: 10-Q Price Prediction Baseline

This project implements a baseline model to predict stock price direction (Up/Down) over a 5-day horizon using historical price data.

## Project Structure

- `conf/`: Configuration files (Hydra).
- `data/`: Data storage (raw and processed).
- `src/data/`: Data loading scripts.
- `src/features/`: Feature engineering scripts.
- `src/models/`: Model training and evaluation scripts.
- `models/`: Saved models.

## Setup

It is important to have Python 3.11+ installed. We recommend using a virtual environment to manage dependencies.

1. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Download Data

Downloads historical data for AAPL, MSFT, GOOGL using `yfinance`.

```bash
python src/data/loader.py
```

### 2. Generate Features

Calculates returns, volatility, RSI, ATR, and target labels.

```bash
python src/features/engineering.py
```

### 3. Train Baseline Model

Trains a Logistic Regression (or configured model) on data < 2023 and tests on data >= 2023.

```bash
python src/models/baseline.py
```

## Configuration

You can modify `conf/config.yaml` to change:

- Tickers (`data.tickers`)
- Prediction Horizon (`features.prediction_horizon`)
- Model Type (`model.type`: `logistic_regression` or `random_forest`)

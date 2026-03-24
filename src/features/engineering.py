import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig
import os
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Calculates Average True Range (ATR)."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


def create_features(df: pd.DataFrame, window_sizes: list) -> pd.DataFrame:
    """
    Generates technical indicators and statistical features.
    """
    df = df.copy()

    # Ensure date is datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

    # Sort by date
    df.sort_index(inplace=True)

    # Basic Returns
    df["return_1d"] = df["Close"].pct_change()

    # Volatility and Moving Averages
    for w in window_sizes:
        # Volatility (annualized assuming 252 trading days)
        df[f"volatility_{w}"] = df["return_1d"].rolling(window=w).std() * np.sqrt(252)

        # Price relative to MA
        df[f"ma_{w}"] = df["Close"] / df["Close"].rolling(window=w).mean()

        # Momentum (Return over window)
        df[f"return_{w}d"] = df["Close"].pct_change(periods=w)

    # RSI
    df["rsi"] = calculate_rsi(df["Close"])

    # ATR (Normalized)
    atr_absolute = calculate_atr(df["High"], df["Low"], df["Close"])
    df["atr_rel"] = atr_absolute / df["Close"]

    return df


def create_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Creates the binary target variable.
    Target = 1 if Price(t+H) > Price(t), else 0.
    """
    # Calculate forward return
    # shift(-H) brings future value to current row
    forward_close = df["Close"].shift(-horizon)
    forward_return = (forward_close - df["Close"]) / df["Close"]

    # Create target, handling NaNs
    # We want to keep NaN where forward_close is NaN so we can drop them later
    target = (forward_return > 0).astype(int)

    # Mask the target as NaN where forward_return is NaN
    # Since we can't have NaN in integer column, we'll keep it as float for a moment or just drop first

    # Better approach:
    # 1. Create a boolean mask for valid future data
    valid_future = forward_close.notna()

    # 2. Assign target only where valid
    df.loc[valid_future, f"target_{horizon}d"] = (
        forward_return[valid_future] > 0
    ).astype(int)

    # 3. Drop rows where we couldn't calculate target (the last 'horizon' rows)
    df = df.dropna(subset=[f"target_{horizon}d"])

    return df


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):

    tickers = cfg.data.tickers
    raw_dir = cfg.data.raw_dir
    processed_dir = cfg.data.processed_dir
    horizon = cfg.features.prediction_horizon
    window_sizes = cfg.features.window_sizes

    os.makedirs(processed_dir, exist_ok=True)

    for ticker in tickers:
        input_path = os.path.join(raw_dir, f"{ticker}.csv")
        if not os.path.exists(input_path):
            logger.warning(f"File {input_path} not found. Skipping.")
            continue

        logger.info(f"Processing {ticker}...")
        df = pd.read_csv(input_path)

        # Generate Features
        df_features = create_features(df, window_sizes)

        # Create Target
        df_final = create_target(df_features, horizon)

        # Drop NaN values created by rolling windows
        # We need to be careful not to drop too much, but initial windows will have NaNs
        original_len = len(df_final)
        df_final.dropna(inplace=True)
        dropped_len = original_len - len(df_final)
        logger.info(f"Dropped {dropped_len} rows due to NaN values (warm-up period).")

        output_path = os.path.join(processed_dir, f"{ticker}_processed.csv")
        df_final.to_csv(output_path)
        logger.info(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    main()

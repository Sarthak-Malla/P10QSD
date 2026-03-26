import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig
import os
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_processed_data(processed_dir: str, tickers: list) -> pd.DataFrame:
    """Loads and concatenates processed data for all tickers."""
    dfs = []
    for ticker in tickers:
        path = os.path.join(processed_dir, f"{ticker}_processed.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["Ticker"] = ticker  # Add ticker column for reference
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
            dfs.append(df)
        else:
            logger.warning(f"Processed data for {ticker} not found.")

    if not dfs:
        raise FileNotFoundError("No processed data found.")

    return pd.concat(dfs, ignore_index=True)


def train_test_split_time_series(df: pd.DataFrame, test_size: float = 0.2):
    """
    Splits data strictly by time.
    Ensures all training data comes before test data.
    """
    # Sort by Date
    df = df.sort_values("Date").reset_index(drop=True)

    # Calculate split index
    split_idx = int(len(df) * (1 - test_size))

    # Split date to log
    split_date = df.iloc[split_idx]["Date"]
    logger.info(f"Splitting data at index {split_idx} (approx date: {split_date})")

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    return train_df, test_df


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):

    processed_dir = cfg.data.processed_dir
    aligned_path = os.path.join(processed_dir, "filing_aligned.csv")

    # Load filing-aligned data
    logger.info("Loading filing-aligned dataset...")
    df = pd.read_csv(aligned_path, parse_dates=["filed_at"])
    logger.info(f"Total rows: {len(df)}, Tickers: {df['ticker'].nunique()}")

    # Features and target
    feature_cols = [
        # Text features
        "cosine_sim_prev",
        "sentiment_compound", "sentiment_pos", "sentiment_neg",
        "text_length_norm",
        # Price features at filing date
        "price_return_1d", "price_return_5d", "price_return_20d",
        "price_volatility_20d", "price_ma_ratio_5", "price_ma_ratio_20",
        "price_rsi",
    ]
    target_col = "target"

    # Drop rows with any NaN in features
    df = df.dropna(subset=feature_cols + [target_col])
    logger.info(f"After dropna: {len(df)} rows")
    logger.info(f"Using features: {feature_cols}")

    # Time-series split
    df = df.sort_values("filed_at").reset_index(drop=True)
    split_idx = int(len(df) * (1 - cfg.training.test_size))
    split_date = df.iloc[split_idx]["filed_at"]
    logger.info(f"Splitting at index {split_idx} (date: {split_date})")

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model_type = cfg.model.type
    if model_type == "logistic_regression":
        model = LogisticRegression(**cfg.model.params)
    elif model_type == "random_forest":
        params = dict(cfg.model.params) if cfg.model.params else {}
        model = RandomForestClassifier(**params, random_state=cfg.seed)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Training {model_type}...")
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))

    # Save
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/baseline_{model_type}.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    logger.info("Model saved to models/")


if __name__ == "__main__":
    main()

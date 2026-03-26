import os
import logging
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hydra
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_sentiment(text: str) -> dict:
    """Compute VADER sentiment scores for a piece of text."""
    sia = SentimentIntensityAnalyzer()
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {"sec_sentiment_pos": np.nan, "sec_sentiment_neg": np.nan, "sec_sentiment_compound": np.nan}
    scores = sia.polarity_scores(text)
    return {
        "sec_sentiment_pos": scores["pos"],
        "sec_sentiment_neg": scores["neg"],
        "sec_sentiment_compound": scores["compound"],
    }


def compute_text_change(texts: pd.Series) -> pd.Series:
    """
    Compute how much each filing's text changed vs the previous one.
    Uses TF-IDF cosine similarity. Returns 1 - similarity (higher = more change).
    """
    changes = [np.nan]  # First filing has no previous to compare to

    valid_texts = texts.fillna("").tolist()

    for i in range(1, len(valid_texts)):
        prev = valid_texts[i - 1]
        curr = valid_texts[i]

        if not prev.strip() or not curr.strip():
            changes.append(np.nan)
            continue

        try:
            vec = TfidfVectorizer(max_features=5000, stop_words="english")
            tfidf = vec.fit_transform([prev, curr])
            sim = cosine_similarity(tfidf[0], tfidf[1])[0][0]
            changes.append(1.0 - sim)  # Higher = more change
        except Exception:
            changes.append(np.nan)

    return pd.Series(changes, index=texts.index)


def build_sec_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a filings dataframe with 'filed_at' and 'section_Item 1A',
    compute NLP features and return a date-indexed dataframe.
    """
    text_col = "section_Item 1A"
    df = df.copy()
    df["filed_at"] = pd.to_datetime(df["filed_at"])
    df = df.sort_values("filed_at").reset_index(drop=True)

    # Sentiment features
    sentiment_records = df[text_col].apply(compute_sentiment)
    sentiment_df = pd.DataFrame(sentiment_records.tolist())

    # Text change score (quarter-over-quarter)
    df["sec_text_change"] = compute_text_change(df[text_col])

    # Text length (normalized)
    df["sec_text_length"] = df[text_col].apply(
        lambda x: len(x.split()) if isinstance(x, str) else np.nan
    )
    mean_len = df["sec_text_length"].mean()
    df["sec_text_length_norm"] = df["sec_text_length"] / mean_len if mean_len > 0 else np.nan

    # Combine
    result = pd.concat([
        df[["filed_at", "ticker"]].reset_index(drop=True),
        sentiment_df.reset_index(drop=True),
        df[["sec_text_change", "sec_text_length_norm"]].reset_index(drop=True),
    ], axis=1)

    return result


def merge_sec_with_prices(price_df: pd.DataFrame, sec_df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill SEC features into daily price data.
    Each filing's features apply to all trading days until the next filing.
    """
    price_df = price_df.copy()
    price_df.index = pd.to_datetime(price_df.index)

    sec_df = sec_df.copy()
    sec_df = sec_df.set_index("filed_at").drop(columns=["ticker"])

    # Reindex SEC features to daily frequency and forward fill
    all_dates = price_df.index.union(sec_df.index)
    sec_daily = sec_df.reindex(all_dates).sort_index().ffill()

    # Align to price dates only
    sec_aligned = sec_daily.reindex(price_df.index)

    for col in sec_aligned.columns:
        price_df[col] = sec_aligned[col].values

    return price_df


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    sec_dir = os.path.join(cfg.data.raw_dir, "sec_filings")
    processed_dir = cfg.data.processed_dir
    tickers = cfg.data.tickers

    for ticker in tickers:
        # Load SEC filings
        sec_path = os.path.join(sec_dir, f"{ticker}_filings.parquet")
        if not os.path.exists(sec_path):
            logger.warning(f"No SEC filings found for {ticker}, skipping.")
            continue

        logger.info(f"Building SEC features for {ticker}...")
        sec_df = pd.read_parquet(sec_path)

        # Check if we have any text
        text_col = "section_Item 1A"
        has_text = sec_df[text_col].notna().sum()
        logger.info(f"{ticker}: {has_text}/{len(sec_df)} filings have Item 1A text")

        if has_text == 0:
            logger.warning(f"{ticker}: No text extracted, skipping SEC features.")
            continue

        # Build SEC features
        sec_features = build_sec_features(sec_df)
        logger.info(f"{ticker}: SEC features shape: {sec_features.shape}")

        # Load processed price data
        price_path = os.path.join(processed_dir, f"{ticker}_processed.csv")
        if not os.path.exists(price_path):
            logger.warning(f"No processed price data for {ticker}, skipping.")
            continue

        price_df = pd.read_csv(price_path, index_col="Date", parse_dates=True)

        # Merge
        merged_df = merge_sec_with_prices(price_df, sec_features)

        # Save
        output_path = os.path.join(processed_dir, f"{ticker}_processed.csv")
        merged_df.to_csv(output_path)
        logger.info(f"{ticker}: Saved merged data with SEC features to {output_path}")
        logger.info(f"{ticker}: New columns: {[c for c in merged_df.columns if c.startswith('sec_')]}")


if __name__ == "__main__":
    main()
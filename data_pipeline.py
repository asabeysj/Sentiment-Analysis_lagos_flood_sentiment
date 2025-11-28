# data_pipeline.py
import pandas as pd
import datetime
import logging
from config import DATA_PATH

logger = logging.getLogger(__name__)


def load_dataset() -> pd.DataFrame:
    """Load synthetic dataset and ensure timestamp & clean_text."""
    df = pd.read_csv(DATA_PATH)

    if "clean_text" not in df.columns:
        df["clean_text"] = df["text"]

    if "timestamp" not in df.columns:
        now = datetime.datetime.now()
        times = [now - datetime.timedelta(minutes=i) for i in range(len(df))]
        times.reverse()
        df["timestamp"] = times

    return df


# Placeholder for real-tweet ingestion later:
def fetch_live_tweets(keywords: list[str], limit: int = 100) -> pd.DataFrame:
    """
    In a full production setting, this would call Twitter API or snscrape
    to fetch fresh tweets and normalise them into a DataFrame with 'text'
    and optional 'timestamp' columns.
    """
    logger.info("fetch_live_tweets called with keywords=%s, limit=%d", keywords, limit)
    # For now, we just return an empty frame; dissertation can describe
    # how this would be implemented.
    return pd.DataFrame(columns=["text", "timestamp"])

import os
import logging
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "data/raw/penguins_cleaned.csv")

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# File handler
fh = logging.FileHandler("logs/data_loader.log")
# Console handler
ch = logging.StreamHandler()
# Formatter
fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(fmt)
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the CSV data, log its shape and dtypes, save a small sample, and return DataFrame.
    """
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Column types: {df.dtypes.to_dict()}")
    # Save a small head sample for quick checks
    sample_path = "data/processed/sample.csv"
    df.head().to_csv(sample_path, index=False)
    logger.info(f"Saved sample to {sample_path}")
    return df

if __name__ == "__main__":
    load_data()

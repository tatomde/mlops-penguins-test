import logging
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from src.data.data_loader import load_data

# Load .env and set up logger
load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Reuse the same handlers as data_loader if desired, otherwise add simple console logging:
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)

def run_eda():
    """Run basic EDA: summary stats + histograms."""
    logger.info("Starting EDA")
    df = load_data()
    # 1. Save summary statistics
    summary = df.describe(include="all")
    summary_path = "reports/metrics/summary_stats.csv"
    summary.to_csv(summary_path)
    logger.info(f"Saved summary statistics to {summary_path}")

    # 2. Generate histograms for numeric columns
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        plt.figure()
        df[col].hist(bins=20)
        plt.title(f"{col} Distribution")
        fig_path = f"reports/figures/{col}_hist.png"
        plt.savefig(fig_path)
        plt.close()
        logger.info(f"Saved histogram for {col} to {fig_path}")

    logger.info("EDA complete")

if __name__ == "__main__":
    run_eda()
import logging
import pandas as pd
from src.data.data_loader import load_data

class FeatureEngineeringError(Exception):
    """Raised when feature engineering fails."""
    pass

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-specific features:
    - bill_length_depth_ratio = bill_length_mm / bill_depth_mm
    - mass_flipper_ratio      = body_mass_g   / flipper_length_mm
    """
    try:
        logger.info("Starting feature engineering")
        df = df.copy()
        # 1. Compute ratios
        df["bill_length_depth_ratio"] = df["bill_length_mm"] / df["bill_depth_mm"]
        df["mass_flipper_ratio"]      = df["body_mass_g"]    / df["flipper_length_mm"]
        logger.info("Added features: bill_length_depth_ratio and mass_flipper_ratio")
        return df
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise FeatureEngineeringError from e

if __name__ == "__main__":
    # Quick standalone run
    df = load_data()
    feat_df = engineer_features(df)
    out_path = "data/processed/features.csv"
    feat_df.to_csv(out_path, index=False)
    logger.info(f"Saved engineered features to {out_path}")


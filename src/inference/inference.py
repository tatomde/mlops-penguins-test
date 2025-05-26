import logging
import pandas as pd
import joblib

from src.features.features import engineer_features
from src.preprocessing.preprocessing import preprocess_data

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)

def run_inference(input_df: pd.DataFrame, model_path: str = "models/model.pkl") -> pd.Series:
    """
    Accepts raw input DataFrame, preprocesses it, loads the model, and returns predictions.
    """
    logger.info("Running inference pipeline...")

    # 1. Engineer features
    df = engineer_features(input_df)

    # 2. Preprocess
    df_proc = preprocess_data(df)

    # 3. Drop one-hot target (if exists from training logic)
    df_proc = df_proc.drop(columns=[col for col in df_proc.columns if col.startswith("species_")], errors="ignore")

    # 4. Load model
    model = joblib.load(model_path)

    # 5. Predict
    predictions = model.predict(df_proc)
    logger.info(f"Generated {len(predictions)} predictions.")
    return predictions

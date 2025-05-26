import logging
import pandas as pd
import joblib

from src.features.features import engineer_features
from src.preprocessing.preprocessing import load_pipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)

def run_inference(input_df: pd.DataFrame, model_path: str = "models/model.pkl", pipeline_path: str = "models/preprocessor.pkl") -> pd.Series:
    """
    Accepts raw input DataFrame, applies feature engineering and preprocessing,
    loads the trained model and pipeline, and returns predictions.
    """
    logger.info("Running inference pipeline...")

    # 1. Feature engineering
    df = engineer_features(input_df)

    # 2. Load pipeline and transform
    pipeline = load_pipeline(pipeline_path)
    df_proc = pipeline.transform(df)

    # 3. Load trained model
    model = joblib.load(model_path)

    # 4. Predict
    predictions = model.predict(df_proc)
    logger.info(f"Generated {len(predictions)} predictions.")
    return predictions
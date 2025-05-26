import pandas as pd
import joblib
import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)

# Custom exception
class PreprocessingError(Exception):
    """Raised when preprocessing pipeline fails to build or run."""
    pass

def build_preprocessing_pipeline(df: pd.DataFrame) -> ColumnTransformer:
    """
    Build a preprocessing pipeline for the given DataFrame:
    - Numeric: median imputation + standard scaling
    - Categorical: most frequent imputation + one-hot encoding
    Returns a fitted or ready-to-fit pipeline.
    """
    try:
        logger.info("Building preprocessing pipeline...")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ])

        logger.info("Preprocessing pipeline built successfully.")
        return preprocessor

    except Exception as e:
        logger.error(f"Failed to build preprocessing pipeline: {e}")
        raise PreprocessingError from e

def save_pipeline(pipeline: ColumnTransformer, path: str = "models/preprocessor.pkl"):
    """Save the fitted preprocessing pipeline to disk."""
    joblib.dump(pipeline, path)
    logger.info(f"Saved preprocessing pipeline to {path}")

def load_pipeline(path: str = "models/preprocessor.pkl") -> ColumnTransformer:
    """Load a saved preprocessing pipeline from disk."""
    pipeline = joblib.load(path)
    logger.info(f"Loaded preprocessing pipeline from {path}")
    return pipeline
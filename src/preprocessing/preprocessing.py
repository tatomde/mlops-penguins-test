import logging
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Custom exception (optional)
class PreprocessingError(Exception):
    """Raised when preprocessing fails."""
    pass

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Impute missing numeric values with median.
    - Scale numeric features.
    - One-hot encode categorical features.
    Returns a transformed DataFrame.
    """
    try:
        logger.info("Starting preprocessing")

        # 1. Identify columns
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()

        # 2. Build transformers
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ])

        # 3. Fit & transform
        transformed_array = preprocessor.fit_transform(df)
        feature_names = (
            numeric_cols +
            list(preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_cols))
        )
        result_df = pd.DataFrame(transformed_array, columns=feature_names, index=df.index)

        logger.info(f"Preprocessing complete: output shape {result_df.shape}")
        return result_df

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise PreprocessingError from e

if __name__ == "__main__":
    # quick smoke test
    import os
    from dotenv import load_dotenv
    from src.data.data_loader import load_data

    load_dotenv()
    df = load_data()
    processed = preprocess_data(df)
    print(processed.head())
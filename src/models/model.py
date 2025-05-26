import os
import logging
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.data.data_loader import load_data
from src.features.features import engineer_features
from src.preprocessing.preprocessing import build_preprocessing_pipeline, save_pipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)

def train_and_save_model(
    df: pd.DataFrame,
    label_col: str = "species",
    output_path: str = "models/model.pkl"
) -> float:
    logger.info("Starting model training")

    # Feature engineering
    df = engineer_features(df)
    y = df[label_col]
    X = df.drop(columns=[label_col])

    # Split before preprocessing
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Build and fit preprocessing pipeline on X_train only
    pipeline = build_preprocessing_pipeline(X_train)
    X_train_proc = pipeline.fit_transform(X_train)
    X_val_proc = pipeline.transform(X_val)

    # Save fitted pipeline
    save_pipeline(pipeline)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_proc, y_train)

    # Evaluate
    y_pred = model.predict(X_val_proc)
    acc = accuracy_score(y_val, y_pred)
    logger.info(f"Accuracy: {acc:.4f}")

    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    logger.info(f"Saved trained model to {output_path}")

    return acc

if __name__ == "__main__":
    df = load_data()
    acc = train_and_save_model(df)
    print(f"Validation accuracy: {acc:.4f}")

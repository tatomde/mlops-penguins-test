import os
import logging
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn


from src.data.data_loader import load_data
from src.features.features import engineer_features
from src.preprocessing.preprocessing import preprocess_data

# Logger setup
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
    """
    Train a classifier and save the model to disk.
    Returns the accuracy on the validation set.
    """
    logger.info("Starting model training")

    # 1. Feature engineering
    df = engineer_features(df)

    # 2. Store label separately and include it in preprocessing input
    y = df[label_col]
    df[label_col] = y  # add target column back so it's one-hot encoded consistently

    # 3. Preprocess
    X_preprocessed = preprocess_data(df)

    # 4. Drop one-hot encoded target column after transformation
    X = X_preprocessed.drop(columns=[c for c in X_preprocessed.columns if c.startswith("species_")])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 5. Model
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

    # 6. Evaluate
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    logger.info(f"Accuracy: {acc:.4f}")

    # 7. Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    logger.info(f"Saved trained model to {output_path}")

    return acc

if __name__ == "__main__":
    df = load_data()
    acc = train_and_save_model(df)
    print(f"Validation accuracy: {acc:.4f}")
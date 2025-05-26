import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.data_loader import load_data
from src.features.features import engineer_features
from src.preprocessing.preprocessing import preprocess_data
from src.evaluation.evaluation import evaluate_model

def test_evaluate_model_outputs(tmp_path):
    """
    Ensure evaluate_model:
    - runs without error
    - writes a JSON report and PNG confusion matrix
    """
    df = load_data()
    df_fe = engineer_features(df)
    y = df["species"]
    df_fe["species"] = y
    X_proc = preprocess_data(df_fe)
    drop_cols = [c for c in X_proc.columns if c.startswith("species_")]
    X_proc = X_proc.drop(columns=drop_cols)

    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=42)

    output_dir = tmp_path / "metrics"
    evaluate_model("models/model.pkl", X_test, y_test, output_dir=str(output_dir))

    assert (output_dir / "classification_report.json").exists()
    assert (output_dir / "confusion_matrix.png").exists()

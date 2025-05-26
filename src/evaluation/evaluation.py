import os
import logging
import json
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)

def evaluate_model(model_path: str, X_test: pd.DataFrame, y_test: pd.Series, output_dir: str = "reports/metrics"):
    """
    Load a model and evaluate it on the provided test set.
    Saves:
    - classification report as JSON
    - confusion matrix as PNG
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Loading model...")
    model = joblib.load(model_path)

    logger.info("Generating predictions...")
    y_pred = model.predict(X_test)

    logger.info("Computing metrics...")
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Save metrics JSON
    report_path = os.path.join(output_dir, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved classification report to {report_path}")

    # Save confusion matrix PNG
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues")
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Saved confusion matrix to {cm_path}")

if __name__ == "__main__":
    from src.data.data_loader import load_data
    from src.features.features import engineer_features
    from src.preprocessing.preprocessing import preprocess_data
    from sklearn.model_selection import train_test_split

    logger.info("Running evaluation pipeline...")

    df = load_data()
    df_fe = engineer_features(df)
    y = df["species"]
    df_fe["species"] = y  # include species for one-hot consistency

    X_proc = preprocess_data(df_fe)

    # 🚨 Match training by dropping one-hot encoded species columns
    drop_cols = [c for c in X_proc.columns if c.startswith("species_")]
    X_proc = X_proc.drop(columns=drop_cols)

    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, stratify=y, random_state=42)

    evaluate_model("models/model.pkl", X_test, y_test)

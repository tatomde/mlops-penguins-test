import argparse
import logging
import sys

from src.data.data_loader import load_data
from src.features.features import engineer_features
from src.preprocessing.preprocessing import preprocess_data
from src.models.model import train_and_save_model
from src.evaluation.evaluation import evaluate_model

from sklearn.model_selection import train_test_split

from src.config import load_config

# Logger
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)

def run_train():
    logger.info("Running training pipeline...")
    df = load_data()
    df = engineer_features(df)
    y = df["species"]
    df["species"] = y
    acc = train_and_save_model(df)
    logger.info(f"Training completed with accuracy: {acc:.4f}")
    config = load_config()
    model_path = config["model"]["path"]

def run_eval():
    logger.info("Running evaluation pipeline...")
    df = load_data()
    df = engineer_features(df)
    y = df["species"]
    df["species"] = y
    config = load_config()
    model_path = config["model"]["path"]

    X_proc = preprocess_data(df)
    drop_cols = [c for c in X_proc.columns if c.startswith("species_")]
    X_proc = X_proc.drop(columns=drop_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, stratify=y, random_state=42
    )
    evaluate_model("models/model.pkl", X_test, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline operations.")
    parser.add_argument("--mode", choices=["train", "eval"], required=True, help="Pipeline step to run")

    args = parser.parse_args()

    try:
        if args.mode == "train":
            run_train()
        elif args.mode == "eval":
            run_eval()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

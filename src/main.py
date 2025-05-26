# src/main.py

import argparse
import logging
import sys
import pandas as pd

from src.data.data_loader import load_data
from src.features.features import engineer_features
from src.preprocessing.preprocessing import load_pipeline
from src.models.model import train_and_save_model
from src.evaluation.evaluation import evaluate_model
from src.inference.inference import run_inference

from sklearn.model_selection import train_test_split

# Logger setup
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

def run_eval():
    logger.info("Running evaluation pipeline...")
    df = load_data()
    df = engineer_features(df)
    y = df["species"]
    df["species"] = y

    pipeline = load_pipeline()
    X_proc_array = pipeline.transform(df)
    X_proc = pd.DataFrame(X_proc_array, columns=pipeline.get_feature_names_out(), index=df.index)

    drop_cols = [c for c in X_proc.columns if c.startswith("species_")]
    X_proc = X_proc.drop(columns=drop_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, stratify=y, random_state=42
    )
    evaluate_model("models/model.pkl", X_test, y_test)

def run_infer(input_path):
    logger.info("Running inference pipeline via CLI...")
    df = pd.read_csv(input_path)
    preds = run_inference(df)
    df["predicted_species"] = preds
    output_path = "data/processed/inference_output.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved inference results to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline operations.")
    parser.add_argument("--mode", choices=["train", "eval", "infer"], required=True, help="Pipeline step to run")
    parser.add_argument("--input", type=str, help="Input path for inference")
    args = parser.parse_args()

    try:
        if args.mode == "train":
            run_train()
        elif args.mode == "eval":
            run_eval()
        elif args.mode == "infer":
            if not args.input:
                raise ValueError("--input is required for inference mode")
            run_infer(args.input)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

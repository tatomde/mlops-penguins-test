import argparse
import pandas as pd
from src.inference.inference import run_inference

def main():
    parser = argparse.ArgumentParser(description="Run inference on a new dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output", type=str, default="data/processed/inference_output.csv", help="Where to save the predictions")
    args = parser.parse_args()

    # Load new data
    df = pd.read_csv(args.input)

    # Run inference
    predictions = run_inference(df)

    # Append predictions and save
    df["predicted_species"] = predictions
    df.to_csv(args.output, index=False)
    print(f"✅ Inference complete. Predictions saved to {args.output}")

if __name__ == "__main__":
    main()
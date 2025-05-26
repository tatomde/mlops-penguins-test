import pandas as pd
from src.inference.inference import run_inference

# Load test input
df = pd.read_csv("data/raw/penguins_cleaned.csv").drop(columns=["species"])
predictions = run_inference(df)

# Save output
df["predicted_species"] = predictions
df.to_csv("data/processed/inference_output.csv", index=False)

print("✅ Inference complete. Output saved to data/processed/inference_output.csv")

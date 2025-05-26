# 🐧 MLOps Pipeline for Penguin Species Classification

This repository contains a modular, production-grade MLOps pipeline for classifying **penguin species** using tabular data. Built from scratch with best practices in modularization, reproducibility, and testability, this pipeline demonstrates how to move from raw CSV files to validated models and structured evaluation outputs — all scriptable via command-line interface.

---

## 🚦 Project Status

**Phase 1: Complete and Tested**
- End-to-end modular pipeline: data loading, validation, preprocessing, feature engineering, model training, evaluation
- Configurable with `config.yaml`
- Test suite with 100% pass rate using pytest
- CLI orchestration via `main.py`
- Output artifacts include trained model (`.pkl`), metrics (`.json`), and confusion matrix plot (`.png`)

---

## 📁 Repository Structure

```text
.
├── README.md
├── config.yaml                  # Central pipeline configuration
├── environment.yml              # Reproducible Conda environment
├── .env.template                # Placeholder for environment variables (if needed)
├── data/
│   ├── raw/                     # Original input data
│   └── processed/              # Sample CSVs, transformed files
├── models/                      # Trained model saved with joblib
├── reports/
│   └── metrics/                 # classification_report.json + confusion_matrix.png
├── src/
│   ├── data/                    # Data loading + EDA
│   ├── validation/              # Schema and integrity checks
│   ├── features/                # Domain-specific feature engineering
│   ├── preprocessing/           # Scaling + encoding pipeline
│   ├── models/                  # Model training and saving logic
│   ├── evaluation/              # Evaluation metrics and plots
│   └── main.py                  # CLI entry point for training/eval
├── tests/                       # Full test coverage for every module
```

---

## 🔬 Problem Description

The pipeline predicts the **species** of a penguin given attributes such as:
- bill length and depth
- flipper length
- body mass
- sex and island

This is a **multiclass classification** task based on the popular Palmer Penguins dataset.

---

## 🛠️ Pipeline Modules

### 0. Orchestration (`src/main.py`)
- Single entry point with `--mode train` and `--mode eval`
- Controls flow of the entire pipeline via config and logging
- Supports reproducibility and experimentation

### 1. Data Loading (`src/data/data_loader.py`)
- Loads raw CSV from `config.yaml`
- Logs shape and schema to console and file
- Saves a preview CSV to `data/processed/sample.csv`

### 2. Validation (`src/validation/data_validation.py`)
- Validates expected columns, types, and non-null constraints
- Custom exceptions for schema mismatches or missing features
- Unit tests for pass/fail cases

### 3. Feature Engineering (`src/features/features.py`)
- Computes:
  - `bill_length_depth_ratio = bill_length_mm / bill_depth_mm`
  - `mass_flipper_ratio = body_mass_g / flipper_length_mm`
- Fully tested and loggable

### 4. Preprocessing (`src/preprocessing/preprocessing.py`)
- Scales numeric columns with `StandardScaler`
- One-hot encodes categorical variables with `OneHotEncoder`
- Handles missing values via `SimpleImputer`

### 5. Model Training (`src/models/model.py`)
- Trains a `RandomForestClassifier`
- Automatically splits into train/val sets
- Saves `.pkl` file to `models/` and logs accuracy

### 6. Evaluation (`src/evaluation/evaluation.py`)
- Generates predictions on a holdout set
- Saves:
  - `classification_report.json` with precision, recall, f1
  - `confusion_matrix.png` for visual interpretation

### 7. Testing (`tests/`)
- Covers:
  - Data loader
  - Validation
  - Preprocessing
  - Feature engineering
  - Model training
  - Evaluation
- Uses `pytest --disable-warnings -q` for clean output

---

## ⚙️ Configuration and Reproducibility

All paths and parameters are configurable in `config.yaml`:

```yaml
data:
  raw_path: data/raw/penguins_cleaned.csv
model:
  path: models/model.pkl
  test_size: 0.2
  random_state: 42
reports:
  metrics_dir: reports/metrics/
```

Environment reproducibility is ensured via:

```bash
conda env create -f environment.yml
conda activate mlops-penguins-pamed
```

---

## 🚀 Quickstart

**Train the model:**

```bash
python -m src.main --mode train
```

**Evaluate the model:**

```bash
python -m src.main --mode eval
```

**Run all tests:**

```bash
pytest --disable-warnings -q
```

---

## 📈 Next Steps (Planned Enhancements)

- Add MLflow or W&B for experiment tracking
- Include batch inference CLI mode
- Refactor into class-based `Pipeline()` structure
- Automate testing with GitHub Actions

---

## 👩‍💻 Authors and Acknowledgments

- Built by [Your Name]  
- IE University — MLOps Track  
- Inspired by the Palmer Penguins dataset from the `seaborn` library

---

## 📜 License

This project is licensed for educational and demonstration purposes. See `LICENSE` for details.

---

For feedback or questions, open an issue or fork the repo and contribute.
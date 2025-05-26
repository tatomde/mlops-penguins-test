import os
import pytest
from src.data.data_loader import load_data
from src.models.model import train_and_save_model

def test_train_and_save_model(tmp_path):
    """
    Train a model and ensure:
    - Accuracy is a float between 0 and 1
    - A model file is saved to the specified path
    """
    df = load_data()
    model_path = tmp_path / "model_test.pkl"
    acc = train_and_save_model(df, output_path=str(model_path))

    assert isinstance(acc, float)
    assert 0 <= acc <= 1
    assert model_path.exists()
    assert model_path.stat().st_size > 0

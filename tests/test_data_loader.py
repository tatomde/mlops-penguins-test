import pandas as pd
from src.data.data_loader import load_data
import sys, os
# add project root (one level up) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_load_data():
    """
    Smoke-test for the data loader:
    - DataFrame is not empty.
    - Expected columns are present.
    """
    df = load_data()
    # 1. It should load at least one row
    assert not df.empty, "DataFrame is empty"
    # 2. Check a few key columns exist
    expected = {"species", "bill_length_mm", "flipper_length_mm", "body_mass_g"}
    assert expected.issubset(df.columns), f"Missing columns: {expected - set(df.columns)}"

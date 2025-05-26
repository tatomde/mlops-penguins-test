import pandas as pd
import pytest

from src.features.features import engineer_features, FeatureEngineeringError
from src.data.data_loader import load_data

def test_engineer_features_success():
    """
    engineer_features on the real dataset should:
    - return a DataFrame,
    - preserve the number of rows,
    - add the two new feature columns,
    - and compute them correctly for the first row.
    """
    df = load_data()
    feat_df = engineer_features(df)
    assert isinstance(feat_df, pd.DataFrame)
    # rows unchanged
    assert feat_df.shape[0] == df.shape[0]
    # new columns present
    assert "bill_length_depth_ratio" in feat_df.columns
    assert "mass_flipper_ratio" in feat_df.columns
    # spot‐check first value
    expected = df["bill_length_mm"].iloc[0] / df["bill_depth_mm"].iloc[0]
    assert pytest.approx(feat_df["bill_length_depth_ratio"].iloc[0], rel=1e-6) == expected

def test_engineer_features_failure():
    """
    Passing None should raise FeatureEngineeringError.
    """
    with pytest.raises(FeatureEngineeringError):
        engineer_features(None)

import pandas as pd
import pytest

from src.validation.data_validation import validate_data, DataValidationError
from src.data.data_loader import load_data

def test_validate_data_success():
    """
    validate_data on the real dataset should return the DataFrame unchanged.
    """
    df = load_data()
    validated = validate_data(df)
    assert isinstance(validated, pd.DataFrame)
    assert validated.shape == df.shape

def test_validate_data_missing_column():
    """
    Dropping a required column should raise DataValidationError.
    """
    df = load_data().copy()
    df.drop(columns=["bill_length_mm"], inplace=True)
    with pytest.raises(DataValidationError) as exc:
        validate_data(df)
    assert "Missing column: bill_length_mm" in str(exc.value)

def test_validate_data_null_values():
    """
    Introducing a null in a required column should raise DataValidationError.
    """
    df = load_data().copy()
    df.loc[0, "bill_depth_mm"] = None
    with pytest.raises(DataValidationError) as exc:
        validate_data(df)
    assert "Null values found in columns" in str(exc.value)

def test_validate_data_non_positive_values():
    """
    Introducing non-positive bill_length_mm values should raise DataValidationError.
    """
    df = load_data().copy()
    df.loc[1, "bill_length_mm"] = 0
    df.loc[2, "bill_length_mm"] = -5.0
    with pytest.raises(DataValidationError) as exc:
        validate_data(df)
    assert "Found non-positive bill_length_mm values" in str(exc.value)
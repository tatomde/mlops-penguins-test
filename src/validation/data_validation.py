import pandas as pd

class DataValidationError(Exception):
    """Custom exception for data validation failures."""
    pass

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate DataFrame schema and contents.
    
    - Checks that required columns exist with the expected dtypes.
    - Ensures no nulls in critical columns.
    - (Optional) Checks numeric columns are within sensible ranges.
    
    Returns the original DataFrame if all checks pass,
    otherwise raises DataValidationError.
    """
    # 1. Define expected schema
    expected_schema = {
        "species": "object",
        "island": "object",
        "bill_length_mm": "float64",
        "bill_depth_mm": "float64",
        "flipper_length_mm": "int64",
        "body_mass_g": "int64",
        "sex": "object",
    }

    # 2. Check columns & dtypes
    for col, expected_dtype in expected_schema.items():
        if col not in df.columns:
            raise DataValidationError(f"Missing column: {col}")
        actual_dtype = df[col].dtype.name
        if actual_dtype != expected_dtype:
            raise DataValidationError(f"Column {col} has dtype {actual_dtype}, expected {expected_dtype}")

    # 3. Check for nulls in required cols
    null_cols = [col for col in expected_schema if df[col].isnull().any()]
    if null_cols:
        raise DataValidationError(f"Null values found in columns: {null_cols}")

    # 4. (Optional) Range checks
    # e.g. bill lengths should be >0
    if (df["bill_length_mm"] <= 0).any():
        raise DataValidationError("Found non-positive bill_length_mm values")

    return df
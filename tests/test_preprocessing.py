import pandas as pd
import pytest

from src.preprocessing.preprocessing import preprocess_data, PreprocessingError
from src.data.data_loader import load_data

def test_preprocess_data_success():
    """
    preprocess_data on the real dataset should:
    - return a DataFrame
    - preserve the number of rows
    - increase the number of columns (due to one-hot encoding)
    - contain no missing values
    """
    df = load_data()
    processed = preprocess_data(df)
    assert isinstance(processed, pd.DataFrame)
    # rows should match
    assert processed.shape[0] == df.shape[0]
    # columns should increase vs. original numeric+categorical
    assert processed.shape[1] > df.shape[1]
    # no NaNs after imputation
    assert not processed.isnull().any().any()

def test_preprocess_data_failure():
    """
    Passing invalid input (e.g., None) should raise PreprocessingError.
    """
    with pytest.raises(PreprocessingError):
        preprocess_data(None)

import pandas as pd
import pytest

from src.preprocessing.preprocessing import build_preprocessing_pipeline, PreprocessingError
from src.data.data_loader import load_data

def test_preprocess_data_success():
    """
    build_preprocessing_pipeline on the real dataset should:
    - return a fitted transformer that can transform the input
    - preserve the number of rows
    - increase the number of columns (due to one-hot encoding)
    - produce no missing values
    """
    df = load_data()
    pipeline = build_preprocessing_pipeline(df)
    processed_array = pipeline.fit_transform(df)
    processed = pd.DataFrame(processed_array, columns=pipeline.get_feature_names_out(), index=df.index)

    assert isinstance(processed, pd.DataFrame)
    assert processed.shape[0] == df.shape[0]
    assert processed.shape[1] > df.shape[1]
    assert not processed.isnull().any().any()

def test_preprocess_data_failure():
    """
    Passing invalid input (e.g., None) should raise PreprocessingError.
    """
    with pytest.raises(PreprocessingError):
        build_preprocessing_pipeline(None)


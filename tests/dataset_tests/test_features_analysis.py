import pytest
import pandas as pd
from src.dataset.features_analysis import prepare_dataset, apply_variance_threshold, apply_correlation_threshold, feature_analysis_pipeline


@pytest.fixture
def sample_dataset():
    data = {
        'timestamp': pd.date_range(start='1/1/2021', periods=5, freq='T'),
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'feature3': [2, 2, 2, 2, 2],
        'label': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    drop_columns = ['feature3']
    label_col = 'label'
    return df, drop_columns, label_col


def test_prepare_dataset(sample_dataset):
    df, drop_columns, label_col = sample_dataset
    prepared_df = prepare_dataset(df, drop_columns, label_col)
    assert 'feature3' not in prepared_df.columns
    assert 'label' not in prepared_df.columns
    assert 'timestamp' in prepared_df.columns
    assert 'feature1' in prepared_df.columns
    assert 'feature2' in prepared_df.columns


def test_apply_variance_threshold(sample_dataset):
    df, drop_columns, label_col = sample_dataset
    df = prepare_dataset(df, drop_columns, label_col)
    df_filtered, dropped_features = apply_variance_threshold(df, 0.00)
    assert 'feature3' not in df_filtered.columns
    assert 'feature3' in dropped_features
    assert len(dropped_features) == 1
    assert 'feature1' in df_filtered.columns
    assert 'feature2' in df_filtered.columns


def test_apply_correlation_threshold(sample_dataset):
    df, drop_columns, label_col = sample_dataset
    df = prepare_dataset(df, drop_columns, label_col)
    df_filtered, dropped_features = apply_correlation_threshold(df, 0.75)
    assert 'feature2' not in df_filtered.columns
    assert 'feature2' in dropped_features
    assert len(dropped_features) == 1
    assert 'feature1' in df_filtered.columns


def test_feature_analysis_pipeline(sample_dataset):
    df, drop_columns, label_col = sample_dataset
    df_filtered, var_dropped, corr_dropped = feature_analysis_pipeline(
        df, drop_columns, label_col, 0.00, 0.75)
    assert not df_filtered.empty
    assert 'feature1' in df_filtered.columns
    assert 'feature2' not in df_filtered.columns
    assert 'feature3' in var_dropped
    assert 'feature2' in corr_dropped

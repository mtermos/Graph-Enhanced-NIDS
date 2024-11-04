import pytest
import pandas as pd
import numpy as np
from src.dataset.clean_dataset import clean_dataset


class DatasetConfig:
    def __init__(self, timestamp_col, flow_id_col, class_col):
        self.timestamp_col = timestamp_col
        self.flow_id_col = flow_id_col
        self.class_col = class_col


@pytest.fixture
def sample_data():
    data = {
        'timestamp': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-01'],
        'flow_id': [1, 2, 3, 1],
        'value': [10, 20, np.inf, 10],
        'class': ['A', 'B', 'A', 'A']
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def dataset_config():
    return DatasetConfig(timestamp_col='timestamp', flow_id_col='flow_id', class_col='class')


def test_replace_inf(sample_data, dataset_config):
    cleaned_df = clean_dataset(sample_data.copy(), dataset_config)
    assert not cleaned_df.isin(
        [np.inf, -np.inf]).any().any(), "Infinite values were not replaced"


def test_dropna(sample_data, dataset_config):
    cleaned_df = clean_dataset(sample_data.copy(), dataset_config)
    assert not cleaned_df.isna().any().any(), "NaN values were not dropped"


def test_drop_duplicates(sample_data, dataset_config):
    cleaned_df = clean_dataset(sample_data.copy(), dataset_config)
    assert len(cleaned_df) == 2, "Duplicates were not dropped correctly"


def test_multi_class(sample_data, dataset_config, capsys):
    clean_dataset(sample_data.copy(), dataset_config, multi_class=True)
    captured = capsys.readouterr()
    assert "==>> classes: ['A' 'B']" in captured.out, "Classes were not printed correctly"


def test_cleaning_combined(sample_data, dataset_config):
    cleaned_df = clean_dataset(sample_data.copy(), dataset_config)
    expected_data = {
        'timestamp': ['2021-01-01', '2021-01-02'],
        'flow_id': [1, 2],
        'value': [10, 20],
        'class': ['A', 'B']
    }
    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(cleaned_df.reset_index(
        drop=True), expected_df.reset_index(drop=True), "DataFrame was not cleaned correctly")


if __name__ == '__main__':
    pytest.main()

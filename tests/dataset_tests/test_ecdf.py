import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.dataset.ecdf import prepare_df, ecdf, plot_ecdf, plot_feature_ecdf


@pytest.fixture
def sample_dataframe():
    data = {
        'timestamp': pd.date_range(start='1/1/2020', periods=10, freq='T'),
        'label': [0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
        'feature1': np.random.randn(10),
        'feature2': np.random.randn(10)
    }
    df = pd.DataFrame(data)
    return df


def test_prepare_df(sample_dataframe):
    df = prepare_df(sample_dataframe, 'timestamp')
    assert 'index' in df.columns
    assert df.iloc[0]['timestamp'] == pd.Timestamp('2020-01-01 00:00:00')
    assert df.iloc[-1]['timestamp'] == pd.Timestamp('2020-01-01 00:09:00')


def test_ecdf():
    data = np.array([1, 2, 3, 4, 5])
    x, y = ecdf(data)
    assert np.array_equal(x, np.array([1, 2, 3, 4, 5]))
    assert np.array_equal(y, np.array([0.2, 0.4, 0.6, 0.8, 1.0]))


def test_plot_ecdf(sample_dataframe, tmp_path):
    df = prepare_df(sample_dataframe, 'timestamp')
    save_path = tmp_path / "ecdf_plots"
    save_path.mkdir()
    plot_ecdf(df, 'label', show_figs=False, save_path=str(save_path))
    assert (save_path / 'ecdf_index.png').exists()


def test_plot_feature_ecdf(sample_dataframe, tmp_path):
    df = prepare_df(sample_dataframe, 'timestamp')
    save_path = tmp_path / "feature_ecdf_plots"
    save_path.mkdir()
    plot_feature_ecdf(df, 'label', [
                      'feature1', 'feature2'], show_figs=False, save_path=str(save_path))
    assert (save_path / 'ecdf_feature1.png').exists()
    assert (save_path / 'ecdf_feature2.png').exists()

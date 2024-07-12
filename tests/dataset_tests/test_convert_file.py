import pytest
import pandas as pd
import os
from dataset.convert_file import convert_file


@pytest.fixture(scope="module")
def sample_data():
    data = {
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def temp_csv(tmpdir, sample_data):
    csv_path = tmpdir.join("sample.csv")
    sample_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def temp_parquet(tmpdir, sample_data):
    parquet_path = tmpdir.join("sample.parquet")
    sample_data.to_parquet(parquet_path, index=False)
    return parquet_path


@pytest.fixture
def temp_pickle(tmpdir, sample_data):
    pickle_path = tmpdir.join("sample.pkl")
    sample_data.to_pickle(pickle_path)
    return pickle_path


def test_csv_to_parquet(temp_csv):
    convert_file(str(temp_csv), 'parquet')
    output_path = temp_csv.dirname + "/sample.parquet"
    assert os.path.exists(output_path)
    df = pd.read_parquet(output_path)
    assert df.equals(pd.read_csv(temp_csv))


def test_csv_to_pickle(temp_csv):
    convert_file(str(temp_csv), 'pkl')
    output_path = temp_csv.dirname + "/sample.pkl"
    assert os.path.exists(output_path)
    df = pd.read_pickle(output_path)
    assert df.equals(pd.read_csv(temp_csv))


def test_parquet_to_csv(temp_parquet):
    convert_file(str(temp_parquet), 'csv')
    output_path = temp_parquet.dirname + "/sample.csv"
    assert os.path.exists(output_path)
    df = pd.read_csv(output_path)
    assert df.equals(pd.read_parquet(temp_parquet))


def test_parquet_to_pickle(temp_parquet):
    convert_file(str(temp_parquet), 'pkl')
    output_path = temp_parquet.dirname + "/sample.pkl"
    assert os.path.exists(output_path)
    df = pd.read_pickle(output_path)
    assert df.equals(pd.read_parquet(temp_parquet))


def test_pickle_to_csv(temp_pickle):
    convert_file(str(temp_pickle), 'csv')
    output_path = temp_pickle.dirname + "/sample.csv"
    assert os.path.exists(output_path)
    df = pd.read_csv(output_path)
    assert df.equals(pd.read_pickle(temp_pickle))


def test_pickle_to_parquet(temp_pickle):
    convert_file(str(temp_pickle), 'parquet')
    output_path = temp_pickle.dirname + "/sample.parquet"
    assert os.path.exists(output_path)
    df = pd.read_parquet(output_path)
    assert df.equals(pd.read_pickle(temp_pickle))


def test_unsupported_format(temp_csv):
    with pytest.raises(ValueError, match="Unsupported output file format. Supported formats are: csv, parquet, pkl"):
        convert_file(str(temp_csv), 'txt')

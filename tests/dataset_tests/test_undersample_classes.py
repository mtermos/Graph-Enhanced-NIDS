import pytest
import pandas as pd
from dataset.undersample_classes import undersample_classes


def test_undersample_classes():
    # Create a sample DataFrame
    data = {
        'class': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C', 'C'],
        'value': [1, 2, 3, 4, 5, 6, 7, 8, 9]
    }
    df = pd.DataFrame(data)

    # Test undersampling with n_undersample=1
    df_undersampled = undersample_classes(
        df, class_col='class', n_undersample=1, fraction=0.5)
    class_counts = df_undersampled['class'].value_counts()

    # 2 or 3 because it is shuffled
    assert class_counts['A'] == 2 or class_counts['A'] == 3
    assert class_counts['B'] == 3
    assert class_counts['C'] == 2

    # Test undersampling with n_undersample=2
    df_undersampled = undersample_classes(
        df, class_col='class', n_undersample=2, fraction=0.5)
    class_counts = df_undersampled['class'].value_counts()

    assert class_counts['A'] == 2 or class_counts['A'] == 3
    assert class_counts['B'] == 1 or class_counts['B'] == 2
    assert class_counts['C'] == 2


if __name__ == "__main__":
    pytest.main()

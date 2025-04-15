import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal

from upgini.autofe.vector import Mean, Sum, Vectorize


def test_mean_operator():
    series1 = pd.Series([1, 2, 3, None], index=[0, 1, 2, 3])
    series2 = pd.Series([4, 5, None, 7], index=[0, 1, 2, 3])
    series3 = pd.Series([7, 8, 9, 10], index=[0, 1, 2, 3])

    data = [series1, series2, series3]

    expected = pd.Series([4.0, 5.0, 4.0, 17 / 3], index=[0, 1, 2, 3])

    mean_op = Mean()
    result = mean_op.calculate_vector(data)

    assert_series_equal(result, expected)


def test_sum_operator():
    series1 = pd.Series([1, 2, 3, None], index=[0, 1, 2, 3])
    series2 = pd.Series([4, 5, None, 7], index=[0, 1, 2, 3])
    series3 = pd.Series([7, 8, 9, 10], index=[0, 1, 2, 3])

    data = [series1, series2, series3]

    expected = pd.Series([12.0, 15.0, 12.0, 17.0], index=[0, 1, 2, 3])

    sum_op = Sum()
    result = sum_op.calculate_vector(data)

    assert_series_equal(result, expected)


def test_vectorize_operator():
    series1 = pd.Series([1, 2, 3, None], index=[0, 1, 2, 3])
    series2 = pd.Series([4, 5, None, 7], index=[0, 1, 2, 3])
    series3 = pd.Series([7, 8, 9, 10], index=[0, 1, 2, 3])

    data = [series1, series2, series3]
    expected_lists = [[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, None, 9.0], [None, 7.0, 10.0]]
    expected = pd.Series(expected_lists, index=[0, 1, 2, 3])

    vectorize_op = Vectorize()
    result = vectorize_op.calculate_vector(data)

    for i, row in enumerate(result):
        assert len(row) == len(expected.iloc[i])
        for j, val in enumerate(row):
            if pd.isna(val) and pd.isna(expected.iloc[i][j]):
                continue  # Both are NaN, so they match
            else:
                assert val == expected.iloc[i][j]

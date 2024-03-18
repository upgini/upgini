import pandas as pd
from upgini.autofe.date import DateDiff, DateDiffFuture

from datetime import datetime
from pandas.testing import assert_series_equal


def test_date_diff():
    df = pd.DataFrame(
        [[datetime(1993, 12, 10), datetime(2022, 10, 10)], [datetime(2023, 10, 10), datetime(2022, 10, 10)]],
        columns=["date1", "date2"],
    )

    operand = DateDiff()
    expected_result = pd.Series([10531, None])
    assert_series_equal(operand.calculate_binary(df.date2, df.date1), expected_result)
    assert_series_equal(operand.calculate_group(df, main_column="date1")["date2"].rename(None), expected_result)


def test_date_diff_future():
    df = pd.DataFrame(
        [[datetime(1993, 12, 10), datetime(2022, 10, 10)], [datetime(1993, 4, 10), datetime(2022, 10, 10)]],
        columns=["date1", "date2"],
    )

    operand = DateDiffFuture()
    expected_result = pd.Series([61.0, 182.0])
    assert_series_equal(operand.calculate_binary(df.date2, df.date1), expected_result)

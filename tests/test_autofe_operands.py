import pandas as pd
from upgini.autofe.date import DateDiff, DateDiffMin, DateDiffType2

from datetime import datetime
from pandas.testing import assert_series_equal


def test_date_diff():
    df = pd.DataFrame(
        [
            ["2022-10-10", pd.to_datetime("1993-12-10").timestamp()],
            ["2022-10-10", pd.to_datetime("2023-10-10").timestamp()],
        ],
        columns=["date1", "date2"],
    )

    operand = DateDiff(right_unit="s")
    expected_result = pd.Series([10531, None])
    assert_series_equal(operand.calculate_binary(df.date1, df.date2), expected_result)


def test_date_diff_type2():
    df = pd.DataFrame(
        [
            [pd.to_datetime("2022-10-10").timestamp(), datetime(1993, 12, 10)],
            [pd.to_datetime("2022-10-10").timestamp(), datetime(1993, 4, 10)],
        ],
        columns=["date1", "date2"],
    )

    operand = DateDiffType2(left_unit="s")
    expected_result = pd.Series([61.0, 182.0])
    assert_series_equal(operand.calculate_binary(df.date1, df.date2), expected_result)


def test_date_diff_min():
    df = pd.DataFrame(
        [
            ["2022-10-10", ["1993-12-10", "1993-12-11"]],
            ["2022-10-10", ["1993-12-10"]],
            ["2022-10-10", ["2023-10-10"]],
            ["2022-10-10", []],
        ],
        columns=["date1", "date2"],
    )

    operand = DateDiffMin()
    expected_result = pd.Series([10530, 10531, None, None])
    assert_series_equal(operand.calculate_binary(df.date1, df.date2), expected_result)

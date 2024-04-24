import pandas as pd
from upgini.autofe.date import DateDiff, DateDiffType2, DateListDiff, DateListDiffBounded

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
    actual = operand.calculate_binary(df.date1, df.date2)
    assert_series_equal(actual, expected_result)


def test_date_diff_list():
    df = pd.DataFrame(
        [
            ["2022-10-10", ["1993-12-10", "1993-12-11"]],
            ["2022-10-10", ["1993-12-10", "1993-12-10"]],
            ["2022-10-10", ["2023-10-10"]],
            ["2022-10-10", []],
        ],
        columns=["date1", "date2"],
    )

    def check(aggregation, expected_name, expected_values):
        operand = DateListDiff(aggregation=aggregation)
        assert operand.name == expected_name
        assert_series_equal(operand.calculate_binary(df.date1, df.date2).rename(None), expected_values)

    check(aggregation="min", expected_name="date_diff_min", expected_values=pd.Series([10530, 10531, None, None]))
    check(aggregation="max", expected_name="date_diff_max", expected_values=pd.Series([10531, 10531, None, None]))
    check(aggregation="mean", expected_name="date_diff_mean", expected_values=pd.Series([10530.5, 10531, None, None]))
    check(aggregation="nunique", expected_name="date_diff_nunique", expected_values=pd.Series([2, 1, 0, 0]))


def test_date_diff_list_bounded():
    df = pd.DataFrame(
        [
            ["2022-10-10", ["2013-12-10", "2013-12-11", "1999-12-11"]],
            [
                "2022-10-10",
                [
                    "2013-12-10",
                    "2003-12-11",
                    "1999-12-11",
                    "1993-12-11",
                    "1983-12-11",
                    "1973-12-11",
                    "1959-12-11",
                ],
            ],
            ["2022-10-10", ["2003-12-10", "2003-12-10"]],
            ["2022-10-10", ["2023-10-10", "1993-12-10"]],
            ["2022-10-10", []],
        ],
        columns=["date1", "date2"],
    )

    def check_num_by_years(lower_bound, upper_bound, expected_name, expected_values):
        operand = DateListDiffBounded(
            diff_unit="Y", aggregation="count", lower_bound=lower_bound, upper_bound=upper_bound
        )
        assert operand.name == expected_name
        assert_series_equal(operand.calculate_binary(df.date1, df.date2).rename(None), expected_values)

    check_num_by_years(0, 18, "date_diff_Y_0_18_count", pd.Series([2, 1, 0, 0, 0]))
    check_num_by_years(18, 23, "date_diff_Y_18_23_count", pd.Series([1, 2, 2, 0, 0]))
    check_num_by_years(23, 30, "date_diff_Y_23_30_count", pd.Series([0, 1, 0, 1, 0]))
    check_num_by_years(30, 45, "date_diff_Y_30_45_count", pd.Series([0, 1, 0, 0, 0]))
    check_num_by_years(45, 60, "date_diff_Y_45_60_count", pd.Series([0, 1, 0, 0, 0]))
    check_num_by_years(60, None, "date_diff_Y_60_plusinf_count", pd.Series([0, 1, 0, 0, 0]))

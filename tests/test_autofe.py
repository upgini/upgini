from datetime import datetime

import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal

from upgini.autofe.date import DateDiff, DateDiffType2, DateListDiff, DateListDiffBounded, DatePercentile
from upgini.autofe.feature import Feature, FeatureGroup
from upgini.autofe.unary import Norm


def test_date_diff():
    df = pd.DataFrame(
        [
            ["2022-10-10", pd.to_datetime("1993-12-10").timestamp()],
            ["2022-10-10", pd.to_datetime("2023-10-10").timestamp()],
            ["2022-10-10", pd.to_datetime("1966-10-10").timestamp()],
            ["1022-10-10", pd.to_datetime("1966-10-10").timestamp()],
            [None, pd.to_datetime("1966-10-10").timestamp()],
            ["2022-10-10", None],
            [None, None],
        ],
        columns=["date1", "date2"],
    )

    operand = DateDiff(right_unit="s")
    expected_result = pd.Series([10531, None, 20454, None, None, None, None])
    assert_series_equal(operand.calculate_binary(df.date1, df.date2), expected_result)


def test_date_diff_type2():
    df = pd.DataFrame(
        [
            [pd.to_datetime("2022-10-10").timestamp(), datetime(1993, 12, 10)],
            [pd.to_datetime("2022-10-10").timestamp(), datetime(1993, 4, 10)],
            [pd.to_datetime("2022-10-10").timestamp(), datetime(993, 4, 10)],
            [None, datetime(1993, 4, 10)],
            [pd.to_datetime("2022-10-10").timestamp(), None],
            [None, None],
        ],
        columns=["date1", "date2"],
    )

    operand = DateDiffType2(left_unit="s")
    expected_result = pd.Series([61.0, 182.0, None, None, None, None])
    actual = operand.calculate_binary(df.date1, df.date2)
    assert_series_equal(actual, expected_result)


def test_date_diff_list():
    df = pd.DataFrame(
        [
            ["2022-10-10", ["1993-12-10", "1993-12-11"]],
            ["2022-10-10", ["1993-12-10", "1993-12-10"]],
            ["2022-10-10", ["2023-10-10"]],
            ["2022-10-10", ["1023-10-10"]],
            ["2022-10-10", []],
        ],
        columns=["date1", "date2"],
    )

    def check(aggregation, expected_name, expected_values):
        operand = DateListDiff(aggregation=aggregation)
        assert operand.name == expected_name
        assert_series_equal(operand.calculate_binary(df.date1, df.date2).rename(None), expected_values)

    check(aggregation="min", expected_name="date_diff_min", expected_values=pd.Series([10530, 10531, None, None, None]))
    check(aggregation="max", expected_name="date_diff_max", expected_values=pd.Series([10531, 10531, None, None, None]))
    check(
        aggregation="mean",
        expected_name="date_diff_mean",
        expected_values=pd.Series([10530.5, 10531, None, None, None]),
    )
    check(aggregation="nunique", expected_name="date_diff_nunique", expected_values=pd.Series([2, 1, 0, 0, 0]))


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
            ["1022-10-10", ["2023-10-10", "1993-12-10"]],
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

    check_num_by_years(0, 18, "date_diff_Y_0_18_count", pd.Series([2, 1, 0, 0, 0, 0]))
    check_num_by_years(18, 23, "date_diff_Y_18_23_count", pd.Series([1, 2, 2, 0, 0, 0]))
    check_num_by_years(23, 30, "date_diff_Y_23_30_count", pd.Series([0, 1, 0, 1, 0, 0]))
    check_num_by_years(30, 45, "date_diff_Y_30_45_count", pd.Series([0, 1, 0, 0, 0, 0]))
    check_num_by_years(45, 60, "date_diff_Y_45_60_count", pd.Series([0, 1, 0, 0, 0, 0]))
    check_num_by_years(60, None, "date_diff_Y_60_plusinf_count", pd.Series([0, 1, 0, 0, 0, 0]))


def test_date_percentile():
    data = pd.DataFrame(
        [
            ["2024-03-03", 2],
            ["2024-02-03", 2],
            ["2024-02-04", 34],
            ["2024-02-05", 32],
            ["2023-03-03", 60],
            ["2023-03-02", None],
        ],
        columns=["date", "feature"],
    )
    operand = DatePercentile(
        zero_month=2,
        zero_year=2024,
        zero_bounds="[0.0, 2.6, 3.2, 3.8, 4.4, 5.0, 5.6, 6.2, 6.8, 7.3999999999999995, 8.0, 8.6, 9.2, "
        "9.8, 10.4, 11.0, 11.6, 12.200000000000001, 12.799999999999999, 13.4, 14.0, 14.6, 15.2, 15.8, 16.4, 17.0,"
        " 17.6, 18.200000000000003, 18.8, 19.4, 20.0, 20.6, 21.200000000000003, 21.8, 22.400000000000002, 23.0, 23.6,"
        " 24.2, 24.8, 25.4, 26.0, 26.599999999999998, 27.2, 27.8, 28.4, 29.0, 29.6, 30.2, 30.799999999999997, 31.4,"
        " 32.0, 32.04, 32.08, 32.12, 32.16, 32.2, 32.24, 32.28, 32.32, 32.36, 32.4, 32.44, 32.48, 32.52, 32.56, 32.6, "
        "32.64, 32.68, 32.72, 32.76, 32.8, 32.84, 32.88, 32.92, 32.96, 33.0, 33.04, 33.08, 33.12, 33.16, 33.2, 33.24,"
        " 33.28, 33.32, 33.36, 33.4, 33.44, 33.48, 33.52, 33.56, 33.6, 33.64, 33.68, 33.72, 33.76, 33.8, 33.84, 33.88,"
        " 33.92, 33.96]",
    )

    expected_values = pd.Series([None, 1, 100, 51, 100, None])
    assert_series_equal(operand.calculate(left=data.date, right=data.feature), expected_values)


def test_norm():
    data = pd.DataFrame(
        [[None, 1], [1, 222], [333, 4], [1, 2], [3, 4], [0, 1], [1, 0], [2, 3], [3, 2], [1, None]],
        columns=["a", "b"],
        index=[9, 8, 7, 6, 5, 1, 2, 3, 4, 0],
    )
    operand = Norm()

    expected_result = pd.DataFrame(
        [
            [None, 0.00450218],
            [0.00300266, 0.99948299],
            [0.99988729, 0.0180087],
            [0.00300266, 0.00900435],
            [0.00900799, 0.0180087],
            [0.0, 0.00450218],
            [0.00300266, 0.0],
            [0.00600533, 0.01350653],
            [0.00900799, 0.00900435],
            [0.00300266, None],
        ],
        columns=["a", "b"],
        index=[9, 8, 7, 6, 5, 1, 2, 3, 4, 0],
    )

    assert_series_equal(operand.calculate_unary(data["a"]), expected_result["a"])
    assert_series_equal(operand.calculate_unary(data["b"]), expected_result["b"])


def test_get_display_name():
    feature1 = Feature.from_formula("abs(f1)").set_display_index("123")
    assert feature1.get_display_name() == "f_f1_autofe_abs_123"

    feature2 = Feature.from_formula("(f1/f2)").set_display_index("123")
    assert feature2.get_display_name(cache=False) == "f_f1_f_f2_autofe_div_123"
    assert feature2.get_display_name(shorten=True) == "f_autofe_div_123"
    assert feature2.get_display_name() == "f_autofe_div_123"  # cached

    feature3 = Feature.from_formula("GroupByThenMin(abs(f1),f2)").set_display_index("123")
    assert feature3.get_display_name(cache=False) == "f_f1_f_f2_autofe_groupbythenmin_123"
    assert feature3.get_display_name(shorten=True) == "f_autofe_groupbythenmin_123"

    feature4 = Feature.from_formula("mean(f1,f2,f3)").set_display_index("123")
    assert feature4.get_display_name(cache=False) == "f_f1_f_f2_f_f3_autofe_mean_123"
    assert feature4.get_display_name(shorten=True) == "f_autofe_mean_123"

    feature5 = Feature.from_formula("date_per(f1,date_diff(f1,f2))").set_display_index("123")
    assert feature5.get_display_name(cache=False) == "f_f1_f_f2_autofe_date_per_method1_123"
    assert feature5.get_display_name(shorten=True, cache=False) == "f_autofe_date_per_method1_123"
    feature5.op.alias = "date_diff_type1_per_method1"
    assert feature5.get_display_name(shorten=True) == "f_autofe_date_diff_type1_per_method1_123"


def test_get_hash():
    feature1 = Feature.from_formula("GroupByThenMin(f1,f2)")
    feature2 = Feature.from_formula("GroupByThenMin(abs(f1),f2)")

    assert feature1.get_hash() != feature2.get_hash()


def test_feature_group():
    data = pd.DataFrame(
        [
            ["a", 1, -1],
            ["a", 2, -3],
            ["b", 3, -1],
            ["b", 0, 0],
            ["c", -4, -2],
        ],
        columns=["f1", "f2", "f3"],
    )

    group1 = FeatureGroup.make_groups(
        [
            Feature.from_formula("GroupByThenMin(f2,f1)"),
            Feature.from_formula("GroupByThenMin(f3,f1)"),
        ]
    )
    assert len(group1) == 1
    expected_group1_res = pd.DataFrame(
        [
            [1, -3],
            [1, -3],
            [0, -1],
            [0, -1],
            [-4, -2],
        ],
        columns=["f_f2_f_f1_autofe_groupbythenmin", "f_f3_f_f1_autofe_groupbythenmin"],
    )
    group1_res = group1[0].calculate(data)
    assert_frame_equal(group1_res, expected_group1_res)

    group2 = FeatureGroup.make_groups(
        [
            Feature.from_formula("GroupByThenMin(abs(f2),f1)"),
            Feature.from_formula("GroupByThenMin(abs(f3),f1)"),
            Feature.from_formula("GroupByThenMin(min(f2,f3),f1)"),
        ]
    )
    assert len(group2) == 1
    expected_group2_res = pd.DataFrame(
        [
            [1, 1, -3],
            [1, 1, -3],
            [0, 0, -1],
            [0, 0, -1],
            [4, 2, -4],
        ],
        columns=[
            "f_f2_f_f1_autofe_groupbythenmin",
            "f_f3_f_f1_autofe_groupbythenmin",
            "f_f2_f_f3_f_f1_autofe_groupbythenmin",
        ],
    )
    group2_res = group2[0].calculate(data)
    assert_frame_equal(group2_res, expected_group2_res)


from datetime import datetime

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
import pytest

from upgini.autofe.binary import Distance, JaroWinklerSim1, JaroWinklerSim2, LevenshteinSim
from upgini.autofe.date import DateDiff, DateDiffType2, DateListDiff, DateListDiffBounded, DatePercentile
from upgini.autofe.feature import Column, Feature, FeatureGroup
from upgini.autofe.unary import Norm


def test_date_diff():
    df = pd.DataFrame(
        [
            ["2022-10-10", pd.to_datetime("1993-12-10").timestamp()],
            ["2022-10-10", pd.to_datetime("2023-10-10").timestamp()],
            ["2022-10-10", pd.to_datetime("1966-10-10").timestamp()],
            ["1022-10-10", pd.to_datetime("1966-10-10").timestamp()],
            [None, pd.to_datetime("1966-10-10").timestamp()],
            ["2022-10-10", None],
            [None, None],
        ],
        columns=["date1", "date2"],
    )

    operand = DateDiff(right_unit="s")
    expected_result = pd.Series([10531, -365.0, 20454, None, None, None, None])
    assert_series_equal(operand.calculate_binary(df.date1, df.date2), expected_result)

    operand = DateDiff(right_unit="s", replace_negative=True)
    expected_result = pd.Series([10531, None, 20454, None, None, None, None])
    assert_series_equal(operand.calculate_binary(df.date1, df.date2), expected_result)


def test_date_diff_type2():
    df = pd.DataFrame(
        [
            [pd.to_datetime("2022-10-10").timestamp(), datetime(1993, 12, 10)],
            [pd.to_datetime("2022-10-10").timestamp(), datetime(1993, 4, 10)],
            [pd.to_datetime("2022-10-10").timestamp(), datetime(993, 4, 10)],
            [None, datetime(1993, 4, 10)],
            [pd.to_datetime("2022-10-10").timestamp(), None],
            [None, None],
        ],
        columns=["date1", "date2"],
    )

    operand = DateDiffType2(left_unit="s")
    expected_result = pd.Series([61.0, 182.0, None, None, None, None])
    actual = operand.calculate_binary(df.date1, df.date2)
    assert_series_equal(actual, expected_result)


def test_date_diff_list():
    df = pd.DataFrame(
        [
            ["2022-10-10", ["1993-12-10", "1993-12-11"]],
            ["2022-10-10", ["1993-12-10", "1993-12-10"]],
            ["2022-10-10", ["2023-10-10"]],
            ["2022-10-10", ["1023-10-10"]],
            ["2022-10-10", []],
        ],
        columns=["date1", "date2"],
    )

    def check(aggregation, expected_name, expected_values):
        operand = DateListDiff(aggregation=aggregation)
        assert operand.name == expected_name
        assert_series_equal(operand.calculate_binary(df.date1, df.date2).rename(None), expected_values)

    check(
        aggregation="min", expected_name="date_diff_min", expected_values=pd.Series([10530, 10531, -365.0, None, None])
    )
    check(
        aggregation="max", expected_name="date_diff_max", expected_values=pd.Series([10531, 10531, -365.0, None, None])
    )
    check(
        aggregation="mean",
        expected_name="date_diff_mean",
        expected_values=pd.Series([10530.5, 10531, -365.0, None, None]),
    )
    check(aggregation="nunique", expected_name="date_diff_nunique", expected_values=pd.Series([2, 1, 1, 1, 0]))

    operand = DateListDiff(aggregation="min", replace_negative=True)
    assert_series_equal(
        operand.calculate_binary(df.date1, df.date2).rename(None), pd.Series([10530, 10531, None, None, None])
    )


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
            ["1022-10-10", ["2023-10-10", "1993-12-10"]],
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

    check_num_by_years(0, 18, "date_diff_Y_0_18_count", pd.Series([2, 1, 0, 0, 0, 0]))
    check_num_by_years(18, 23, "date_diff_Y_18_23_count", pd.Series([1, 2, 2, 0, 0, 0]))
    check_num_by_years(23, 30, "date_diff_Y_23_30_count", pd.Series([0, 1, 0, 1, 0, 0]))
    check_num_by_years(30, 45, "date_diff_Y_30_45_count", pd.Series([0, 1, 0, 0, 0, 0]))
    check_num_by_years(45, 60, "date_diff_Y_45_60_count", pd.Series([0, 1, 0, 0, 0, 0]))
    check_num_by_years(60, None, "date_diff_Y_60_plusinf_count", pd.Series([0, 1, 0, 0, 0, 0]))


def test_date_percentile():
    data = pd.DataFrame(
        [
            ["2024-03-03", 2],
            ["2024-02-03", 2],
            ["2024-02-04", 34],
            ["2024-02-05", 32],
            ["2023-03-03", 60],
            ["2023-03-02", None],
        ],
        columns=["date", "feature"],
    )
    operand = DatePercentile(
        zero_month=2,
        zero_year=2024,
        zero_bounds="[0.0, 2.6, 3.2, 3.8, 4.4, 5.0, 5.6, 6.2, 6.8, 7.3999999999999995, 8.0, 8.6, 9.2, "
        "9.8, 10.4, 11.0, 11.6, 12.200000000000001, 12.799999999999999, 13.4, 14.0, 14.6, 15.2, 15.8, 16.4, 17.0,"
        " 17.6, 18.200000000000003, 18.8, 19.4, 20.0, 20.6, 21.200000000000003, 21.8, 22.400000000000002, 23.0, 23.6,"
        " 24.2, 24.8, 25.4, 26.0, 26.599999999999998, 27.2, 27.8, 28.4, 29.0, 29.6, 30.2, 30.799999999999997, 31.4,"
        " 32.0, 32.04, 32.08, 32.12, 32.16, 32.2, 32.24, 32.28, 32.32, 32.36, 32.4, 32.44, 32.48, 32.52, 32.56, 32.6, "
        "32.64, 32.68, 32.72, 32.76, 32.8, 32.84, 32.88, 32.92, 32.96, 33.0, 33.04, 33.08, 33.12, 33.16, 33.2, 33.24,"
        " 33.28, 33.32, 33.36, 33.4, 33.44, 33.48, 33.52, 33.56, 33.6, 33.64, 33.68, 33.72, 33.76, 33.8, 33.84, 33.88,"
        " 33.92, 33.96]",
    )

    expected_values = pd.Series([None, 1, 100, 51, 100, None])
    assert_series_equal(operand.calculate(left=data.date, right=data.feature), expected_values)


def test_norm():
    data = pd.DataFrame(
        [[None, 1], [1, 222], [333, 4], [1, 2], [3, 4], [0, 1], [1, 0], [2, 3], [3, 2], [1, None]],
        columns=["a", "b"],
        index=[9, 8, 7, 6, 5, 1, 2, 3, 4, 0],
    )
    operand = Norm()

    expected_result = pd.DataFrame(
        [
            [None, 0.00450218],
            [0.00300266, 0.99948299],
            [0.99988729, 0.0180087],
            [0.00300266, 0.00900435],
            [0.00900799, 0.0180087],
            [0.0, 0.00450218],
            [0.00300266, 0.0],
            [0.00600533, 0.01350653],
            [0.00900799, 0.00900435],
            [0.00300266, None],
        ],
        columns=["a", "b"],
        index=[9, 8, 7, 6, 5, 1, 2, 3, 4, 0],
    )

    assert_series_equal(operand.calculate_unary(data["a"]), expected_result["a"])
    assert_series_equal(operand.calculate_unary(data["b"]), expected_result["b"])


def test_string_sim():
    data = pd.DataFrame(
        [
            ["book", "look"],
            ["blow", None],
            [None, "Jeremy"],
            ["below", "bewoll"],
            [None, None],
            ["abc", "abc"],
            ["four", "seven"],
        ],
        columns=["a", "b"],
    )

    expected_jw1 = pd.Series([0.833, None, None, 0.902, None, 1.0, 0.0])
    expected_jw2 = pd.Series([0.883, None, None, 0.739, None, 1.0, 0.0])
    expected_lv = pd.Series([0.75, None, None, 0.5, None, 1.0, 0.0])

    assert_series_equal(JaroWinklerSim1().calculate_binary(data["a"], data["b"]).round(3), expected_jw1)
    assert_series_equal(JaroWinklerSim2().calculate_binary(data["a"], data["b"]).round(3), expected_jw2)
    assert_series_equal(LevenshteinSim().calculate_binary(data["a"], data["b"]).round(3), expected_lv)


def test_distance():
    data = pd.DataFrame(
        [
            [np.array([0, 1, 0]), np.array([0, 1, 0])],
            [[0, 1, 0], [0, 1, 0]],
            [np.array([0, 1, 0]), np.array([1, 1, 0])],
            [np.array([0, 1, 0]), np.array([1, 0, 0])],
            [np.array([0, 1, 0]), None],
            [None, np.array([1, 0, 0])],
        ],
        columns=["v1", "v2"],
    )

    op = Distance()

    expected_values = pd.Series([0.0, 0.0, 0.5, 1.0, np.nan, np.nan])
    actual_values = op.calculate_binary(data.v1, data.v2)

    assert_series_equal(actual_values, expected_values)


def test_get_display_name():
    feature1 = Feature.from_formula("abs(f1)").set_display_index("123")
    assert feature1.get_display_name() == "f_f1_autofe_abs_123"

    feature2 = Feature.from_formula("(f1/f2)").set_display_index("123")
    assert feature2.get_display_name(cache=False) == "f_f1_f_f2_autofe_div_123"
    assert feature2.get_display_name(shorten=True) == "f_autofe_div_123"
    assert feature2.get_display_name() == "f_autofe_div_123"  # cached

    feature3 = Feature.from_formula("GroupByThenMin(abs(f1),f2)").set_display_index("123")
    assert feature3.get_display_name(cache=False) == "f_f1_f_f2_autofe_groupbythenmin_123"
    assert feature3.get_display_name(shorten=True) == "f_autofe_groupbythenmin_123"

    feature4 = Feature.from_formula("mean(f1,f2,f3)").set_display_index("123")
    assert feature4.get_display_name(cache=False) == "f_f1_f_f2_f_f3_autofe_mean_123"
    assert feature4.get_display_name(shorten=True) == "f_autofe_mean_123"

    feature5 = Feature.from_formula("date_per(f1,date_diff(f1,f2))").set_display_index("123")
    assert feature5.get_display_name(cache=False) == "f_f1_f_f2_autofe_date_per_method1_123"
    assert feature5.get_display_name(shorten=True, cache=False) == "f_autofe_date_per_method1_123"
    feature5.op.alias = "date_diff_type1_per_method1"
    assert feature5.get_display_name(shorten=True) == "f_autofe_date_diff_type1_per_method1_123"

    feature6 = Feature.from_formula("abs(date_diff(b,c))").set_display_index("123")
    assert feature6.get_display_name(cache=False) == "f_b_f_c_autofe_date_diff_type1_abs_123"
    assert feature6.get_display_name(shorten=True) == "f_autofe_date_diff_type1_abs_123"

    feature7 = Feature.from_formula("date_diff(b,c)").set_display_index("123")
    assert feature7.get_display_name(cache=False) == "f_b_f_c_autofe_date_diff_type1_123"
    assert feature7.get_display_name(shorten=True) == "f_autofe_date_diff_type1_123"


def test_get_hash():
    feature1 = Feature.from_formula("GroupByThenMin(f1,f2)")
    feature2 = Feature.from_formula("GroupByThenMin(abs(f1),f2)")

    assert feature1.get_hash() != feature2.get_hash()


def test_feature_group():
    data = pd.DataFrame(
        [
            ["a", 1, -1],
            ["a", 2, -3],
            ["b", 3, -1],
            ["b", 0, 0],
            ["c", -4, -2],
        ],
        columns=["f1", "f2", "f3"],
    )

    group1 = FeatureGroup.make_groups(
        [
            Feature.from_formula("GroupByThenMin(f2,f1)"),
            Feature.from_formula("GroupByThenMin(f3,f1)"),
        ]
    )
    assert len(group1) == 1
    expected_group1_res = pd.DataFrame(
        [
            [1, -3],
            [1, -3],
            [0, -1],
            [0, -1],
            [-4, -2],
        ],
        columns=["f_f2_f_f1_autofe_groupbythenmin", "f_f3_f_f1_autofe_groupbythenmin"],
    )
    group1_res = group1[0].calculate(data)
    assert_frame_equal(group1_res, expected_group1_res)

    group2 = FeatureGroup.make_groups(
        [
            Feature.from_formula("GroupByThenMin(abs(f2),f1)"),
            Feature.from_formula("GroupByThenMin(abs(f3),f1)"),
            Feature.from_formula("GroupByThenMin(min(f2,f3),f1)"),
        ]
    )
    assert len(group2) == 1
    expected_group2_res = pd.DataFrame(
        [
            [1, 1, -3],
            [1, 1, -3],
            [0, 0, -1],
            [0, 0, -1],
            [4, 2, -4],
        ],
        columns=[
            "f_f2_f_f1_autofe_groupbythenmin",
            "f_f3_f_f1_autofe_groupbythenmin",
            "f_f2_f_f3_f_f1_autofe_groupbythenmin",
        ],
    )
    group2_res = group2[0].calculate(data)
    assert_frame_equal(group2_res, expected_group2_res)


def test_from_formula():

    def check_formula(formula):
        assert Feature.from_formula(formula).to_formula() == formula

    check_formula("a")
    check_formula("(a/b)")
    check_formula("log(a)")
    check_formula("date_diff(a,b)")
    check_formula("date_per(a,date_diff(b,c))")
    check_formula("mean(a,b,c,d,e)")

    with pytest.raises(ValueError):
        check_formula("unsupported(a,b)")

    with pytest.raises(ValueError):
        check_formula("(a,b)")

    with pytest.raises(ValueError):
        check_formula("a/b")

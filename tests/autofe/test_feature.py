import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from upgini.autofe.binary import Divide
from upgini.autofe.date import DateDiff, DateDiffType2, DateListDiff, DatePercentile
from upgini.autofe.feature import Column, Feature, FeatureGroup
from upgini.autofe.groupby import GroupByThenAgg, GroupByThenFreq
from upgini.autofe.unary import Abs, Norm
from upgini.autofe.vector import Mean


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

    feature8 = Feature.from_formula("lag_10D(date,f1,f2,value)").set_display_index("123")
    assert feature8.get_display_name(cache=False) == "f_date_f_f1_f_f2_f_value_autofe_lag_10d_123"
    assert feature8.get_display_name(shorten=True) == "f_autofe_lag_10d_123"

    feature9 = Feature.from_formula("bin(abs(date_diff(b,c)))").set_display_index("123")
    assert feature9.get_display_name(cache=False) == "f_b_f_c_autofe_date_diff_type1_abs_bin_123"
    assert feature9.get_display_name(shorten=True) == "f_autofe_date_diff_type1_abs_bin_123"

    feature10 = Feature.from_formula("date_per(date, abs(date_diff(b,c)))").set_display_index("123")
    assert feature10.get_display_name(cache=False) == "f_date_f_b_f_c_autofe_date_per_method1_123"
    assert feature10.get_display_name(shorten=True) == "f_autofe_date_per_method1_123"


def test_get_hash():
    feature1 = Feature.from_formula("GroupByThenMin(f1,f2)")
    feature2 = Feature.from_formula("GroupByThenMin(abs(f1),f2)")

    assert feature1.get_hash() != feature2.get_hash()


def test_feature_group():
    data = pd.DataFrame(
        [
            ["a", 1.0, -1.0],
            ["a", 2.0, -3.0],
            ["b", 3.0, -1.0],
            ["b", 0.0, 0.0],
            ["c", -4.0, -2.0],
        ],
        columns=["f1", "f2", "f3"],
    )

    group1 = FeatureGroup.make_groups(
        [
            Feature.from_formula("GroupByThenMin(f2,f1)"),
            Feature.from_formula("GroupByThenMin(f3,f1)"),
            Feature.from_formula("abs(f2)"),
        ]
    )
    assert len(group1) == 2
    expected_group1_res = pd.DataFrame(
        [
            [1, -3],
            [1, -3],
            [0, -1],
            [0, -1],
            [-4, -2],
        ],
        columns=["f_f2_f_f1_autofe_groupbythenmin", "f_f3_f_f1_autofe_groupbythenmin"],
        dtype="float",
    )
    group1_res = group1[0].calculate(data)
    assert_frame_equal(group1_res, expected_group1_res)

    expected_group1_res2 = pd.DataFrame(
        [
            [1.0],
            [2.0],
            [3.0],
            [0.0],
            [4.0],
        ],
        columns=["f_f2_autofe_abs"],
        dtype="float",
    )
    group1_res2 = group1[1].calculate(data)
    assert_frame_equal(group1_res2, expected_group1_res2)

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
        dtype="float",
    )
    group2_res = group2[0].calculate(data)
    assert_frame_equal(group2_res, expected_group2_res)


def test_feature_group_nested():
    data = pd.DataFrame(
        [
            [None, 1],
            [1, 222],
            [333, 4],
            [1, 2],
            [3, 4],
            [0, 1],
            [1, 0],
            [2, 3],
            [3, 2],
            [1, None],
        ],
        columns=["a", "b"],
        index=[9, 8, 7, 6, 5, 1, 2, 3, 4, 0],
    )

    norm_data = pd.DataFrame(
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

    features = FeatureGroup.make_groups(
        [
            Feature.from_formula("(a/b)").set_display_index("i1"),
            Feature.from_formula("(norm(b)/a)").set_display_index("i2"),
            Feature.from_formula("(a/norm(b))").set_display_index("i3"),
            Feature.from_formula("(norm(a)/b)").set_display_index("i4"),
            Feature.from_formula("(norm(a)/norm(b))").set_display_index("i5"),
        ]
    )

    assert len(features) == 5

    expected_result = pd.DataFrame(
        {
            "f_a_f_b_autofe_div_i1": data["a"] / data["b"].replace(0, np.nan),
            "f_b_f_a_autofe_div_i2": norm_data["b"] / data["a"].replace(0, np.nan),
            "f_a_f_b_autofe_div_i3": data["a"] / norm_data["b"].replace(0, np.nan),
            "f_a_f_b_autofe_div_i4": norm_data["a"] / data["b"].replace(0, np.nan),
            "f_a_f_b_autofe_div_i5": norm_data["a"] / norm_data["b"].replace(0, np.nan),
        }
    )

    result = pd.DataFrame({f.get_display_names()[0]: f.calculate(data).iloc[:, 0] for f in features})

    assert_frame_equal(result, expected_result)


def test_to_formula():
    assert Feature(Abs(), [Column("a")]).to_formula() == "abs(a)"
    assert Feature(Divide(), [Column("a"), Column("b")]).to_formula() == "(a/b)"

    assert Feature(Abs(), [Feature(Divide(), [Column("a"), Column("b")])]).to_formula() == "abs((a/b))"

    assert Feature(GroupByThenAgg(agg="Min"), [Column("a"), Column("b")]).to_formula() == "GroupByThenMin(a,b)"

    assert Feature(GroupByThenFreq(), [Column("a"), Column("b")]).to_formula() == "GroupByThenFreq(a,b)"

    assert Feature(Mean(), [Column("a"), Column("b"), Column("c")]).to_formula() == "mean(a,b,c)"

    assert Feature(DateDiff(), [Column("a"), Column("b")]).to_formula() == "date_diff(a,b)"

    assert (
        Feature(DatePercentile(), [Column("a"), Feature(DateDiff(), [Column("b"), Column("c")])]).to_formula()
        == "date_per(a,date_diff(b,c))"
    )


def test_from_formula():

    def check_formula(formula):
        assert Feature.from_formula(formula).to_formula() == formula

    check_formula("a")
    check_formula("(a/b)")
    check_formula("abs((a/b))")
    check_formula("log(a)")
    check_formula("date_diff(a,b)")
    check_formula("date_per(a,date_diff(b,c))")
    check_formula("mean(a,b,c,d,e)")
    check_formula("roll_3D_mean(a,b)")
    check_formula("roll_3D_mean_offset_1D(a,b)")

    assert DateListDiff.from_formula("date_diff_type2") is None
    assert isinstance(Feature.from_formula("date_diff_type2(a,b)").op, DateDiffType2)

    with pytest.raises(ValueError):
        check_formula("unsupported(a,b)")

    with pytest.raises(ValueError):
        check_formula("(a,b)")

    with pytest.raises(ValueError):
        check_formula("a/b")


def test_op_params():
    norm1 = Feature(Norm(), [Column("a")]).set_op_params({"norm": "1"})
    assert norm1.op.norm == 1

    norm2 = Feature(Norm(), [Column("b")]).set_op_params({"norm": "2"})
    assert norm2.op.norm == 2

    feature = Feature(
        Divide(),
        [
            norm1,
            Feature(Abs(), [norm2]),
        ],
    )

    assert feature.get_op_params() == {
        "alias": "div",
        "f_a_autofe_norm_norm": "1.0",
        "f_b_autofe_norm_abs_f_b_autofe_norm_norm": "2.0",
    }

    feature.set_op_params({"norm": "3"})
    assert norm1.op.norm == 3
    assert norm2.op.norm == 3

    feature.set_op_params(
        {"alias": "div", "f_a_autofe_norm_norm": "4", "f_b_autofe_norm_abs_f_b_autofe_norm_norm": "5"}
    )
    assert norm1.op.norm == 4
    assert norm1.op.alias is None
    assert norm2.op.norm == 5
    assert norm2.op.alias is None

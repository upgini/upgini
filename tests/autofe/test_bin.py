import pandas as pd
from pandas.testing import assert_series_equal

from upgini.autofe.unary import BinCat


def test_date_bin_basic():
    data = pd.Series([10, 20, 30, 40, 50, None])

    operator = BinCat(bin_bounds=[0, 25, 45])

    # Expected:
    # 10 -> '1' (falls in first bin: >= 0 and < 25)
    # 20 -> '1' (falls in first bin: >= 0 and < 25)
    # 30 -> '2' (falls in second bin: >= 25 and < 45)
    # 40 -> '2' (falls in second bin: >= 25 and < 45)
    # 50 -> '3' (falls in third bin: >= 45)
    # None -> '-1' (default for NaN values)
    expected_values = pd.Series(["1", "1", "2", "2", "3", "-1"])
    result = operator.calculate(data=data)
    assert_series_equal(result, expected_values)

    # Test with null series
    null_series = pd.Series([None, None, None])
    result_null = operator.calculate(data=null_series)
    expected_null = pd.Series(["-1", "-1", "-1"])
    assert_series_equal(result_null, expected_null)


def test_date_bin_empty_bounds():
    data = pd.Series([10, 20])

    operand = BinCat(bin_bounds=[])

    # All values should return '-1' as there's no bin to fall into
    expected_values = pd.Series(["-1", "-1"])
    result = operand.calculate(data=data)
    assert_series_equal(result, expected_values)


def test_date_bin_negative_values():
    data = pd.Series([-20, -10, 0, 10])

    operand = BinCat(bin_bounds=[-30, -15, 0, 15])

    # Expected:
    # -20 -> '1' (falls in first bin: >= -30 and < -15)
    # -10 -> '2' (falls in second bin: >= -15 and < 0)
    # 0 -> '3' (falls in third bin: >= 0 and < 15)
    # 10 -> '3' (falls in third bin: >= 0 and < 15)
    expected_values = pd.Series(["1", "2", "3", "3"])
    result = operand.calculate(data=data)
    assert_series_equal(result, expected_values)


def test_date_bin_out_of_bounds():
    data = pd.Series([-10, 0, 10, 100])

    operand = BinCat(bin_bounds=[0, 50])

    # Expected:
    # -10 -> '-1' (less than the lower bound)
    # 0 -> '1' (falls in first bin: >= 0 and < 50)
    # 10 -> '1' (falls in first bin: >= 0 and < 50)
    # 100 -> '2' (falls in second bin: >= 50)
    expected_values = pd.Series(["-1", "1", "1", "2"])
    result = operand.calculate(data=data)
    assert_series_equal(result, expected_values)


def test_date_bin_parse_obj():
    date_bin = BinCat(bin_bounds=[0, 10, 20, 30, 40, 50])

    # Convert to dict, then parse back to object
    date_bin_dict = date_bin.get_params()
    parsed_date_bin = BinCat.parse_obj(date_bin_dict)

    # Verify the parsed object has the same parameters
    assert parsed_date_bin.bin_bounds == [0, 10, 20, 30, 40, 50]
    assert parsed_date_bin.name == "bin"
    assert parsed_date_bin.is_unary is True


def test_date_bin_string_bounds():
    date_bin = BinCat(bin_bounds="[0, 25, 50, 75, 100]")
    assert date_bin.bin_bounds == [0, 25, 50, 75, 100]

    # Test with sample data
    data = pd.Series([10, 30, 60, 90])

    expected_values = pd.Series(["1", "2", "3", "4"])
    result = date_bin.calculate(data=data)
    assert_series_equal(result, expected_values)


def test_bin_float_values():
    data = pd.Series([10.5, 25.0, 25.1, 45.0, 45.1])

    operand = BinCat(bin_bounds=[0, 25, 45])

    # Expected:
    # 10.5 -> '1' (falls in first bin: >= 0 and < 25)
    # 25.0 -> '2' (falls in second bin: >= 25 and < 45)
    # 25.1 -> '2' (falls in second bin: >= 25 and < 45)
    # 45.0 -> '3' (falls in third bin: >= 45)
    # 45.1 -> '3' (falls in third bin: >= 45)
    expected_values = pd.Series(["1", "2", "2", "3", "3"])
    result = operand.calculate(data=data)
    assert_series_equal(result, expected_values)


def test_date_bin_with_index():
    data = pd.Series([10, 30, 60], index=["a", "b", "c"])

    operand = BinCat(bin_bounds=[0, 20, 40])

    expected_values = pd.Series(["1", "2", "3"], index=["a", "b", "c"])
    result = operand.calculate(data=data)
    assert_series_equal(result, expected_values)

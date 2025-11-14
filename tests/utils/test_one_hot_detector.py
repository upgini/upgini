import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np

from upgini.utils.one_hot_encoder import OneHotDecoder


class TestOneHotDecoder:
    def test_decode_basic_group(self):
        # Prepare DataFrame with a valid one-hot group grp_0, grp_1, grp_2
        df = pd.DataFrame(
            {
                "grp_0": [1, 0, 0, 0, 1],
                "grp_1": [0, 1, 0, 1, 0],
                "grp_2": [0, 0, 1, 0, 0],
            }
        )

        transformed, true_one_hot_groups, pseudo_one_hot_groups = OneHotDecoder.decode(df)

        # Group detection
        assert true_one_hot_groups == {"grp_": ["grp_0", "grp_1", "grp_2"]}
        assert pseudo_one_hot_groups == {}

        # New categorical column exists, original columns removed
        assert "grp_" in transformed.columns
        assert "grp_0" not in transformed.columns
        assert "grp_1" not in transformed.columns
        assert "grp_2" not in transformed.columns

        # Values are 0..N-1 codes with nullable integer dtype
        assert str(transformed["grp_"].dtype) == "string"
        assert transformed["grp_"].tolist() == ["0", "1", "2", "1", "0"]

    def test_decode_mixed_types_and_invalid_rows(self):
        # Prepare DataFrame with mixed textual/boolean values and some invalid rows
        df = pd.DataFrame(
            {
                "a0": ["1", "0", "0", True, "false"],
                "a1": ["0", "1", "0", False, "false"],
            }
        )

        transformed, true_one_hot_groups, pseudo_one_hot_groups = OneHotDecoder.decode(df)

        assert true_one_hot_groups == {}
        assert pseudo_one_hot_groups == {}
        assert_frame_equal(transformed, df)

    def test_decode_no_groups(self):
        # No numeric suffix -> no groups detected
        df = pd.DataFrame({"x": [0, 1, 2], "y": [3, 4, 5]})
        transformed, groups, pseudo_one_hot_groups = OneHotDecoder.decode(df)

        assert groups == {}
        assert pseudo_one_hot_groups == {}
        # DataFrame should remain the same
        assert_frame_equal(transformed, df)

    def test_non_consecutive_suffixes_not_detected(self):
        # Columns b0 and b2 are not consecutive -> group should not be detected
        df = pd.DataFrame(
            {
                "b0": [0, 1, 0, 0, 0],
                "b2": [0, 0, 1, 0, 0],
            }
        )
        transformed, groups, pseudo_one_hot_groups = OneHotDecoder.decode(df)

        assert groups == {}
        assert pseudo_one_hot_groups == {}
        assert_frame_equal(transformed, df)

    def test_decode_complex_names(self):
        df = pd.DataFrame(
            {
                "a1_b3_1": [0, 1, 0, 0, 0],
                "a1_b3_4": [0, 0, 0, 0, 1],
                "a1_b3_3": [0, 0, 0, 1, 0],
                "a1_b3_2": [0, 0, 1, 0, 0],
            }
        )
        expected_df = pd.DataFrame(
            {
                "a1_b3_": [None, "1", "2", "3", "4"],
            }, dtype="string"
        )
        transformed, groups, pseudo_one_hot_groups = OneHotDecoder.decode(df)
        assert_frame_equal(transformed, expected_df)
        assert groups == {"a1_b3_": ["a1_b3_1", "a1_b3_2", "a1_b3_3", "a1_b3_4"]}
        assert pseudo_one_hot_groups == {}

    def test_decode_true_one_hot_with_cached(self):
        df = pd.DataFrame(
            {
                "a1_b3_1": [0, 1, 0, 0, 0],
                "a1_b3_4": [0, 0, 0, 0, 1],
                "a1_b3_3": [0, 0, 0, 1, 0],
                "a1_b3_2": [0, 0, 1, 0, 0],
            }
        )
        expected_df = pd.DataFrame(
            {
                "a1_b3_": [None, "1", "2", "3", "4"],
            }, dtype="string"
        )
        true_one_hot_groups = {"a1_b3_": ["a1_b3_1", "a1_b3_2", "a1_b3_3", "a1_b3_4"]}
        transformed = OneHotDecoder.decode_with_cached_groups(df, true_one_hot_groups, {})
        assert_frame_equal(transformed, expected_df)

    def test_decode_pseudo_one_hot_encoded_columns(self):
        df = pd.DataFrame(
            {
                "a1_b3_1": [0, 1, 1, 0, 0],
                "a1_b3_4": [0, 0, 0, 0, 1],
                "a1_b3_3": [0, 0, 0, 1, 0],
                "a1_b3_2": [0, 0, 1, 0, 0],
            }
        )
        expected_df = pd.DataFrame(
            {
                "a1_b3_1": ["0", "1", "1", "0", "0"],
                "a1_b3_4": ["0", "0", "0", "0", "1"],
                "a1_b3_3": ["0", "0", "0", "1", "0"],
                "a1_b3_2": ["0", "0", "1", "0", "0"],
            }, dtype="string"
        )
        transformed, groups, pseudo_one_hot_groups = OneHotDecoder.decode(df)
        assert_frame_equal(transformed, expected_df)
        assert groups == {}
        assert pseudo_one_hot_groups == {"a1_b3_": ["a1_b3_1", "a1_b3_2", "a1_b3_3", "a1_b3_4"]}

    def test_decode_pseudo_one_hot_with_cached(self):
        df = pd.DataFrame(
            {
                "a1_b3_1": [0, 1, 1, 0, 0],
                "a1_b3_4": [0, 0, 0, 0, 1],
                "a1_b3_3": [0, 0, 0, 1, 0],
                "a1_b3_2": [0, 0, 1, 0, 0],
            }
        )
        expected_df = pd.DataFrame(
            {
                "a1_b3_1": ["0", "1", "1", "0", "0"],
                "a1_b3_4": ["0", "0", "0", "0", "1"],
                "a1_b3_3": ["0", "0", "0", "1", "0"],
                "a1_b3_2": ["0", "0", "1", "0", "0"],
            }, dtype="string"
        )
        pseudo_one_hot_groups = {"a1_b3_": ["a1_b3_1", "a1_b3_2", "a1_b3_3", "a1_b3_4"]}
        transformed = OneHotDecoder.decode_with_cached_groups(df, None, pseudo_one_hot_groups)
        assert_frame_equal(transformed, expected_df)

    def test_valid_one_hot_encoded_integers(self):
        """Test valid one-hot encoded series with integers where 0 is majority class"""
        # Test case where 0 is majority and 1 is minority
        series = pd.Series([0, 0, 0, 0, 1, 0, 1, 0])
        assert OneHotDecoder._is_one_hot_encoded(series) is True

    def test_valid_one_hot_encoded_floats(self):
        """Test valid one-hot encoded series with floats that can be converted to int"""
        series = pd.Series([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        assert OneHotDecoder._is_one_hot_encoded(series) is True

    def test_valid_one_hot_encoded_strings(self):
        """Test valid one-hot encoded series with string representations"""
        series = pd.Series(["0", "0", "0", "1", "0", "1"])
        assert OneHotDecoder._is_one_hot_encoded(series) is True

    def test_invalid_values_not_binary(self):
        """Test series with values other than 0 and 1"""
        series = pd.Series([0, 1, 2, 0, 1])
        assert OneHotDecoder._is_one_hot_encoded(series) is False

    def test_invalid_values_negative(self):
        """Test series with negative values"""
        series = pd.Series([0, 1, -1, 0, 1])
        assert OneHotDecoder._is_one_hot_encoded(series) is False

    def test_invalid_values_text(self):
        """Test series with text values that cannot be converted to int"""
        series = pd.Series(["0", "1", "text", "0", "1"])
        assert OneHotDecoder._is_one_hot_encoded(series) is False

    def test_series_with_nan_values(self):
        """Test series containing NaN values"""
        series = pd.Series([0, 1, np.nan, 0, 1])
        assert OneHotDecoder._is_one_hot_encoded(series) is False

    def test_series_with_none_values(self):
        """Test series containing None values"""
        series = pd.Series([0, 1, None, 0, 1])
        assert OneHotDecoder._is_one_hot_encoded(series) is False

    def test_imbalanced_ones_majority(self):
        """Test series where 1 is majority class (should return False)"""
        # More 1s than 0s
        series = pd.Series([1, 1, 1, 1, 0, 1])
        assert OneHotDecoder._is_one_hot_encoded(series) is False

    def test_equal_distribution(self):
        """Test series with equal number of 0s and 1s (should return False)"""
        series = pd.Series([0, 1, 0, 1])
        assert OneHotDecoder._is_one_hot_encoded(series) is False

    def test_only_zeros(self):
        """Test series with only zeros"""
        series = pd.Series([0, 0, 0, 0])
        assert OneHotDecoder._is_one_hot_encoded(series) is False

    def test_only_ones(self):
        """Test series with only ones"""
        series = pd.Series([1, 1, 1, 1])
        assert OneHotDecoder._is_one_hot_encoded(series) is False

    def test_empty_series(self):
        """Test empty series"""
        series = pd.Series([], dtype=int)
        assert OneHotDecoder._is_one_hot_encoded(series) is False

    def test_single_zero(self):
        """Test series with single zero value"""
        series = pd.Series([0])
        assert OneHotDecoder._is_one_hot_encoded(series) is False

    def test_single_one(self):
        """Test series with single one value"""
        series = pd.Series([1])
        assert OneHotDecoder._is_one_hot_encoded(series) is False

    def test_float_strings_valid(self):
        """Test series with float strings that represent 0 and 1"""
        series = pd.Series(["0.0", "1.0", "0.0", "0.0", "1.0"])
        assert OneHotDecoder._is_one_hot_encoded(series) is True

    def test_boolean_values(self):
        """Test series with boolean values"""
        series = pd.Series([True, False, False, False, True])
        # This should work as True/False can be converted to 1/0
        assert OneHotDecoder._is_one_hot_encoded(series) is True

    def test_mixed_numeric_types(self):
        """Test series with mixed numeric types that represent 0 and 1"""
        series = pd.Series([0, 1.0, "0", 0, "1"])
        assert OneHotDecoder._is_one_hot_encoded(series) is False

    def test_decimal_values(self):
        """Test series with decimal values that are not exactly 0 or 1"""
        series = pd.Series([0.1, 0.9, 0.0, 1.0])
        assert OneHotDecoder._is_one_hot_encoded(series) is False

    def test_large_series_valid(self):
        """Test with larger series to ensure performance"""
        # Create large series with 90% zeros, 10% ones
        zeros = [0] * 900
        ones = [1] * 100
        series = pd.Series(zeros + ones)
        assert OneHotDecoder._is_one_hot_encoded(series) is True

    def test_true_false(self):
        series = pd.Series(["true", "false", "false", "false", "true"])
        assert OneHotDecoder._is_one_hot_encoded(series) is True

    def test_one_hot_validation_with_one_column(self):
        df = pd.DataFrame({
            "feature1": [False] * 13113 + [True] * 36
        })
        decoded, true_one_hot_groups, pseudo_one_hot_groups = OneHotDecoder.decode(df)
        assert_frame_equal(decoded, df)
        assert true_one_hot_groups == {}
        assert pseudo_one_hot_groups == {}

    def test_one_hot_validation_with_several_columns(self):
        df = pd.DataFrame({
            "feature1": [False] * 13113 + [True] * 36,
            "feature2": [True] * 49 + [False] * 13100,
        })
        expected = pd.DataFrame({
            "feature": ["2"] * 49 + [None] * 13064 + ["1"] * 36,
        }, dtype="string")
        decoded, true_one_hot_groups, pseudo_one_hot_groups = OneHotDecoder.decode(df)
        assert true_one_hot_groups == {"feature": ["feature1", "feature2"]}
        assert_frame_equal(decoded, expected)
        assert pseudo_one_hot_groups == {}

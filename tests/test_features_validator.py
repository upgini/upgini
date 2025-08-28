import numpy as np
import pandas as pd

from upgini.utils.features_validator import FeaturesValidator
from upgini.resource_bundle import bundle


class TestIsOneHotEncoded:
    """Test cases for FeaturesValidator.is_one_hot_encoded method"""

    def test_valid_one_hot_encoded_integers(self):
        """Test valid one-hot encoded series with integers where 0 is majority class"""
        # Test case where 0 is majority and 1 is minority
        series = pd.Series([0, 0, 0, 0, 1, 0, 1, 0])
        assert FeaturesValidator.is_one_hot_encoded(series) is True

    def test_valid_one_hot_encoded_floats(self):
        """Test valid one-hot encoded series with floats that can be converted to int"""
        series = pd.Series([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        assert FeaturesValidator.is_one_hot_encoded(series) is True

    def test_valid_one_hot_encoded_strings(self):
        """Test valid one-hot encoded series with string representations"""
        series = pd.Series(["0", "0", "0", "1", "0", "1"])
        assert FeaturesValidator.is_one_hot_encoded(series) is True

    def test_invalid_values_not_binary(self):
        """Test series with values other than 0 and 1"""
        series = pd.Series([0, 1, 2, 0, 1])
        assert FeaturesValidator.is_one_hot_encoded(series) is False

    def test_invalid_values_negative(self):
        """Test series with negative values"""
        series = pd.Series([0, 1, -1, 0, 1])
        assert FeaturesValidator.is_one_hot_encoded(series) is False

    def test_invalid_values_text(self):
        """Test series with text values that cannot be converted to int"""
        series = pd.Series(["0", "1", "text", "0", "1"])
        assert FeaturesValidator.is_one_hot_encoded(series) is False

    def test_series_with_nan_values(self):
        """Test series containing NaN values"""
        series = pd.Series([0, 1, np.nan, 0, 1])
        assert FeaturesValidator.is_one_hot_encoded(series) is False

    def test_series_with_none_values(self):
        """Test series containing None values"""
        series = pd.Series([0, 1, None, 0, 1])
        assert FeaturesValidator.is_one_hot_encoded(series) is False

    def test_imbalanced_ones_majority(self):
        """Test series where 1 is majority class (should return False)"""
        # More 1s than 0s
        series = pd.Series([1, 1, 1, 1, 0, 1])
        assert FeaturesValidator.is_one_hot_encoded(series) is False

    def test_equal_distribution(self):
        """Test series with equal number of 0s and 1s (should return False)"""
        series = pd.Series([0, 1, 0, 1])
        assert FeaturesValidator.is_one_hot_encoded(series) is False

    def test_only_zeros(self):
        """Test series with only zeros"""
        series = pd.Series([0, 0, 0, 0])
        assert FeaturesValidator.is_one_hot_encoded(series) is False

    def test_only_ones(self):
        """Test series with only ones"""
        series = pd.Series([1, 1, 1, 1])
        assert FeaturesValidator.is_one_hot_encoded(series) is False

    def test_empty_series(self):
        """Test empty series"""
        series = pd.Series([], dtype=int)
        assert FeaturesValidator.is_one_hot_encoded(series) is False

    def test_single_zero(self):
        """Test series with single zero value"""
        series = pd.Series([0])
        assert FeaturesValidator.is_one_hot_encoded(series) is False

    def test_single_one(self):
        """Test series with single one value"""
        series = pd.Series([1])
        assert FeaturesValidator.is_one_hot_encoded(series) is False

    def test_float_strings_valid(self):
        """Test series with float strings that represent 0 and 1"""
        series = pd.Series(["0.0", "1.0", "0.0", "0.0", "1.0"])
        assert FeaturesValidator.is_one_hot_encoded(series) is True

    def test_boolean_values(self):
        """Test series with boolean values"""
        series = pd.Series([True, False, False, False, True])
        # This should work as True/False can be converted to 1/0
        assert FeaturesValidator.is_one_hot_encoded(series) is True

    def test_mixed_numeric_types(self):
        """Test series with mixed numeric types that represent 0 and 1"""
        series = pd.Series([0, 1.0, "0", 0, "1"])
        assert FeaturesValidator.is_one_hot_encoded(series) is False

    def test_decimal_values(self):
        """Test series with decimal values that are not exactly 0 or 1"""
        series = pd.Series([0.1, 0.9, 0.0, 1.0])
        assert FeaturesValidator.is_one_hot_encoded(series) is False

    def test_large_series_valid(self):
        """Test with larger series to ensure performance"""
        # Create large series with 90% zeros, 10% ones
        zeros = [0] * 900
        ones = [1] * 100
        series = pd.Series(zeros + ones)
        assert FeaturesValidator.is_one_hot_encoded(series) is True

    def test_true_false(self):
        series = pd.Series(["true", "false", "false", "false", "true"])
        assert FeaturesValidator.is_one_hot_encoded(series) is True

    def test_one_hot_validation(self):
        df = pd.DataFrame({
            "feature1": [False] * 13113 + [True] * 36
        })
        validator = FeaturesValidator()
        expected_warning = bundle.get("one_hot_encoded_features").format(["feature1"])
        assert validator.validate(df, ["feature1"]) == ([], [expected_warning])

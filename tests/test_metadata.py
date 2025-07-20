import pytest

from upgini.metadata import ModelTaskType


class TestModelTaskTypeParse:
    """Tests for the parse method of ModelTaskType class"""

    def test_parse_with_model_task_type_instance(self):
        """Test: passing ModelTaskType instance should return the same instance"""
        # Arrange
        original_task_type = ModelTaskType.BINARY

        # Act
        result = ModelTaskType.parse(original_task_type)

        # Assert
        assert result is original_task_type
        assert result == ModelTaskType.BINARY

    def test_parse_with_valid_string_uppercase(self):
        """Test: valid strings in uppercase"""
        test_cases = [
            ("BINARY", ModelTaskType.BINARY),
            ("MULTICLASS", ModelTaskType.MULTICLASS),
            ("REGRESSION", ModelTaskType.REGRESSION),
            ("TIMESERIES", ModelTaskType.TIMESERIES),
        ]

        for input_str, expected_type in test_cases:
            # Act
            result = ModelTaskType.parse(input_str)

            # Assert
            assert result == expected_type
            assert isinstance(result, ModelTaskType)

    def test_parse_with_valid_string_lowercase(self):
        """Test: valid strings in lowercase should be normalized to uppercase"""
        test_cases = [
            ("binary", ModelTaskType.BINARY),
            ("multiclass", ModelTaskType.MULTICLASS),
            ("regression", ModelTaskType.REGRESSION),
            ("timeseries", ModelTaskType.TIMESERIES),
        ]

        for input_str, expected_type in test_cases:
            # Act
            result = ModelTaskType.parse(input_str)

            # Assert
            assert result == expected_type
            assert isinstance(result, ModelTaskType)

    def test_parse_with_valid_string_mixed_case(self):
        """Test: valid strings in mixed case should be normalized to uppercase"""
        test_cases = [
            ("Binary", ModelTaskType.BINARY),
            ("MultiClass", ModelTaskType.MULTICLASS),
            ("Regression", ModelTaskType.REGRESSION),
            ("TimeSeries", ModelTaskType.TIMESERIES),
            ("bInArY", ModelTaskType.BINARY),
            ("rEgReSSioN", ModelTaskType.REGRESSION),
        ]

        for input_str, expected_type in test_cases:
            # Act
            result = ModelTaskType.parse(input_str)

            # Assert
            assert result == expected_type
            assert isinstance(result, ModelTaskType)

    def test_parse_with_invalid_string(self):
        """Test: invalid strings should raise ValueError"""
        invalid_strings = [
            "INVALID",
            "CLASSIFICATION",
            "UNKNOWN",
            "",
            "   ",
            "BINARY_CLASS",
            "MULTI",
            "REG",
        ]

        for invalid_str in invalid_strings:
            # Act & Assert
            with pytest.raises(ValueError, match=f"'{invalid_str.upper()}' is not a valid ModelTaskType"):
                ModelTaskType.parse(invalid_str)

    def test_parse_with_none(self):
        """Test: None should raise ValueError"""
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid task type: None"):
            ModelTaskType.parse(None)

    def test_parse_preserves_enum_properties(self):
        """Test: parsing preserves enum properties after conversion"""
        # Arrange
        task_types = ["binary", "multiclass", "regression", "timeseries"]

        for task_type_str in task_types:
            # Act
            parsed_type = ModelTaskType.parse(task_type_str)

            # Assert
            assert hasattr(parsed_type, "is_classification")
            assert callable(parsed_type.is_classification)

            # Verify is_classification method works correctly
            if parsed_type in [ModelTaskType.BINARY, ModelTaskType.MULTICLASS]:
                assert parsed_type.is_classification() is True
            else:
                assert parsed_type.is_classification() is False

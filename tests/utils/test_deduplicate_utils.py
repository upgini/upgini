import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from upgini.metadata import (
    ENTITY_SYSTEM_RECORD_ID,
    EVAL_SET_INDEX,
    SORT_ID,
    SYSTEM_RECORD_ID,
    TARGET,
    SearchKey,
)
from upgini.utils.deduplicate_utils import (
    clean_full_duplicates,
    remove_fintech_duplicates,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "phone": ["123", "123", "234", "456", "789", "789"],
            "date": [
                "2023-01-01",
                "2023-01-15",  # within 60 days of first record
                "2023-01-01",
                "2023-01-01",
                "2023-06-01",
                "2023-06-02",
            ],
            TARGET: [1, 0, 1, 1, 0, 0],  # different targets for same phone
            EVAL_SET_INDEX: [0, 0, 0, 1, 1, 1],
        }
    )


@pytest.fixture
def search_keys():
    return {"phone": SearchKey.PHONE, "date": SearchKey.DATE}


def test_remove_fintech_duplicates_basic(sample_df, search_keys):
    # Prepare expected result
    expected_df = sample_df[sample_df["phone"] != "123"].copy()  # remove records with phone '123'

    # Call the tested function
    result_df, warnings = remove_fintech_duplicates(
        df=sample_df,
        search_keys=search_keys,
    )

    # Checks
    assert_frame_equal(result_df, expected_df)
    assert len(warnings) == 1  # should be one warning for train set


def test_remove_fintech_duplicates_no_duplicates():
    # Create DataFrame without duplicates
    df = pd.DataFrame(
        {
            "phone": ["123", "456", "789"],
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            TARGET: [1, 0, 1],
            EVAL_SET_INDEX: [0, 0, 1],
        }
    )

    search_keys = {"phone": SearchKey.PHONE, "date": SearchKey.DATE}

    # Call function
    result_df, warnings = remove_fintech_duplicates(
        df=df,
        search_keys=search_keys,
    )

    # Checks
    assert_frame_equal(result_df, df)
    assert warnings == []


def test_remove_fintech_duplicates_no_personal_cols():
    # Create DataFrame without personal data columns
    df = pd.DataFrame(
        {"some_col": ["a", "b", "c"], "date": ["2023-01-01", "2023-01-02", "2023-01-03"], TARGET: [1, 0, 1]}
    )

    search_keys = {"date": SearchKey.DATE}

    # Call function
    result_df, warnings = remove_fintech_duplicates(
        df=df,
        search_keys=search_keys,
    )

    # Checks
    assert_frame_equal(result_df, df)
    assert warnings == []


def test_remove_fintech_duplicates_different_dates():
    # Create DataFrame with different dates for same phones
    df = pd.DataFrame(
        {
            "phone": ["123", "123", "123"],
            "date": ["2023-01-01", "2023-06-01", "2023-12-01"],  # dates differ significantly
            TARGET: [1, 0, 1],
            EVAL_SET_INDEX: [0, 0, 0],
        }
    )

    search_keys = {"phone": SearchKey.PHONE, "date": SearchKey.DATE}

    # Call function
    result_df, warnings = remove_fintech_duplicates(
        df=df,
        search_keys=search_keys,
    )

    # Checks - records should remain since dates differ significantly
    assert_frame_equal(result_df, df)
    assert warnings == []


def test_clean_full_duplicates_empty_df():
    """Test for empty DataFrame"""
    df = pd.DataFrame()
    result_df, warning = clean_full_duplicates(df)
    assert len(result_df) == 0
    assert warning is None


def test_clean_full_duplicates_no_duplicates():
    """Test for DataFrame without duplicates"""
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"], TARGET: [0, 1, 0]})
    result_df, warning = clean_full_duplicates(df)
    assert_frame_equal(result_df, df)
    assert warning is None


def test_clean_full_duplicates_with_system_columns():
    """Test for removing duplicates with system columns"""
    df = pd.DataFrame(
        {
            "col1": [1, 1, 2],
            "col2": ["a", "a", "b"],
            SYSTEM_RECORD_ID: [100, 200, 300],
            SORT_ID: [1, 2, 3],
            EVAL_SET_INDEX: [0, 0, 1],
        }
    )

    expected = pd.DataFrame(
        {"col1": [1, 2], "col2": ["a", "b"], SYSTEM_RECORD_ID: [100, 300], SORT_ID: [1, 3], EVAL_SET_INDEX: [0, 1]}
    )

    result_df, warning = clean_full_duplicates(df)
    assert_frame_equal(result_df.reset_index(drop=True), expected.reset_index(drop=True))
    assert warning is None


def test_clean_full_duplicates_with_different_targets():
    """Test for removing rows with different target values"""
    df = pd.DataFrame(
        {"col1": [1, 1, 2], "col2": ["a", "a", "b"], TARGET: [0, 1, 0]}  # different values for identical rows
    )

    expected = pd.DataFrame({"col1": [2], "col2": ["b"], TARGET: [0]})

    result_df, warning = clean_full_duplicates(df)
    assert_frame_equal(result_df.reset_index(drop=True), expected.reset_index(drop=True))
    assert isinstance(warning, str)  # Should have warning about duplicates with different targets


def test_clean_full_duplicates_keep_first():
    """Test for keeping first row when duplicates exist"""
    df = pd.DataFrame(
        {
            "col1": [1, 1, 2],
            "col2": ["a", "a", "b"],
            TARGET: [0, 0, 1],
            EVAL_SET_INDEX: [0, 1, 0],  # checking that train segment (0) is preserved
        }
    )

    expected = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"], TARGET: [0, 1], EVAL_SET_INDEX: [0, 0]})

    result_df, warning = clean_full_duplicates(df)
    assert_frame_equal(result_df.reset_index(drop=True), expected.reset_index(drop=True))
    assert warning is None


def test_clean_full_duplicates_all_system_columns():
    """Test for presence of all possible system columns"""
    df = pd.DataFrame(
        {
            "col1": [1, 1, 2],
            "col2": ["a", "a", "b"],
            SYSTEM_RECORD_ID: [100, 200, 300],
            ENTITY_SYSTEM_RECORD_ID: [1000, 2000, 3000],
            SORT_ID: [1, 2, 3],
            EVAL_SET_INDEX: [0, 1, 0],
            TARGET: [0, 0, 1],
        }
    )

    expected = pd.DataFrame(
        {
            "col1": [1, 2],
            "col2": ["a", "b"],
            SYSTEM_RECORD_ID: [100, 300],
            ENTITY_SYSTEM_RECORD_ID: [1000, 3000],
            SORT_ID: [1, 3],
            EVAL_SET_INDEX: [0, 0],
            TARGET: [0, 1],
        }
    )

    result_df, warning = clean_full_duplicates(df)
    assert_frame_equal(result_df.reset_index(drop=True), expected.reset_index(drop=True))
    assert warning is None


def test_clean_full_duplicates_silent_mode():
    """Test for silent mode operation"""
    df = pd.DataFrame({"col1": [1, 1, 2], "col2": ["a", "a", "b"], TARGET: [0, 1, 0]})

    result_df, warning = clean_full_duplicates(df)
    expected = pd.DataFrame({"col1": [2], "col2": ["b"], TARGET: [0]})

    assert_frame_equal(result_df.reset_index(drop=True), expected.reset_index(drop=True))
    assert isinstance(warning, str)  # Should have warning about duplicates with different targets

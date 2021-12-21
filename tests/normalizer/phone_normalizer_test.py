import pandas as pd
from upgini.normalizer.phone_normalizer import phone_to_int
from pandas.testing import assert_frame_equal


def test_phone_float_to_int_safe():
    df = pd.DataFrame(
        data={"phone_num": [7.2, 0.1, 3.9, 123456789012345.1, None], "something_else": ["a", "b", "c", "d", "e"]}
    )
    phone_to_int(df, "phone_num")

    expected_df = pd.DataFrame(
        data={"phone_num": [7, 0, 3, 123456789012345, -1], "something_else": ["a", "b", "c", "d", "e"]}
    )
    assert_frame_equal(df, expected_df)


def test_phone_int_to_int_safe():
    df = pd.DataFrame(
        data={"phone_num": [0, -2, 100, 123456789012345, None], "something_else": ["a", "b", "c", "d", "e"]}
    )
    phone_to_int(df, "phone_num")

    expected_df = pd.DataFrame(
        data={"phone_num": [0, -2, 100, 123456789012345, -1], "something_else": ["a", "b", "c", "d", "e"]}
    )
    assert_frame_equal(df, expected_df)


def test_phone_str_to_int_safe():
    df = pd.DataFrame(
        data={
            "phone_num": ["+4234567", "01 02 03 04", "223-45-67", "123456789012345", "abc"],
            "something_else": ["a", "b", "c", "d", "e"],
        }
    )
    phone_to_int(df, "phone_num")

    expected_df = pd.DataFrame(
        data={
            "phone_num": [4234567, 1020304, 2234567, 123456789012345, -1],
            "something_else": ["a", "b", "c", "d", "e"],
        }
    )
    assert_frame_equal(df, expected_df)

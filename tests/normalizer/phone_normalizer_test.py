import pandas as pd
from pandas.testing import assert_frame_equal

from upgini.normalizer.phone_normalizer import phone_to_int


def test_phone_float_to_int_safe():
    df = pd.DataFrame(
        data={"phone_num": [7.2, 0.1, 3.9, 123456789012345.1, None], "something_else": ["a", "b", "c", "d", "e"]}
    )
    phone_to_int(df, "phone_num")

    expected_df = pd.DataFrame(
        data={"phone_num": [None, None, None, 123456789012345, None], "something_else": ["a", "b", "c", "d", "e"]}
    )
    assert_frame_equal(df, expected_df)


def test_phone_int_to_int_safe():
    df = pd.DataFrame(
        data={
            "phone_num": [0, -2, 100, 123456789012345, 1234567890123456, None],
            "something_else": ["a", "b", "c", "d", "e", "f"],
        }
    )
    phone_to_int(df, "phone_num")

    expected_df = pd.DataFrame(
        data={"phone_num": [None, None, None, 123456789012345, None, None], "something_else": ["a", "b", "c", "d", "e", "f"]}
    )
    assert_frame_equal(df, expected_df)


def test_phone_str_to_int_safe():
    df = pd.DataFrame(
        data={
            "phone_num": [
                "+4(234)5678",
                "01 02 03 04 05",
                "223-45-678",
                "+86 10 6764 5489",
                "123456789012345",
                "123",
                "abc",
            ],
            "something_else": ["a", "b", "c", "d", "e", "f", "g"],
        }
    )
    phone_to_int(df, "phone_num")

    expected_df = pd.DataFrame(
        data={
            "phone_num": [42345678, 102030405, 22345678, 861067645489, 123456789012345, None, None],
            "something_else": ["a", "b", "c", "d", "e", "f", "g"],
        }
    )
    assert_frame_equal(df, expected_df)

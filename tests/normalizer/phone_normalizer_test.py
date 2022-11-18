import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from upgini.normalizer.phone_normalizer import PhoneNormalizer


def test_phone_float_to_int_safe():
    df = pd.DataFrame(
        data={"phone_num": [7.2, 0.1, 3.9, 123456789012345.1, None], "something_else": ["a", "b", "c", "d", "e"]}
    )
    normalizer = PhoneNormalizer(df, "phone_num")
    df["phone_num"] = normalizer.normalize()

    expected_df = pd.DataFrame(
        data={"phone_num": [None, None, None, 123456789012345, None], "something_else": ["a", "b", "c", "d", "e"]}
    )
    expected_df["phone_num"] = expected_df["phone_num"].astype("Int64")
    assert_frame_equal(df, expected_df)


def test_phone_int_to_int_safe():
    df = pd.DataFrame(
        data={
            "phone_num": [0, -2, 100, 123456789012345, 1234567890123456, None],
            "something_else": ["a", "b", "c", "d", "e", "f"],
        }
    )
    normalizer = PhoneNormalizer(df, "phone_num")
    df["phone_num"] = normalizer.normalize()

    expected_df = pd.DataFrame(
        data={
            "phone_num": [None, None, None, 123456789012345, None, None],
            "something_else": ["a", "b", "c", "d", "e", "f"],
        }
    )
    expected_df["phone_num"] = expected_df["phone_num"].astype("Int64")
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
    normalizer = PhoneNormalizer(df, "phone_num")
    df["phone_num"] = normalizer.normalize()

    expected_df = pd.DataFrame(
        data={
            "phone_num": [42345678, 102030405, 22345678, 861067645489, 123456789012345, None, None],
            "something_else": ["a", "b", "c", "d", "e", "f", "g"],
        }
    )
    expected_df["phone_num"] = expected_df["phone_num"].astype("Int64")
    assert_frame_equal(df, expected_df)


def test_phone_prefix_normalization():
    df = pd.DataFrame(
        data={
            "phone_num": [
                "(123) 321-987-1",
                "01 02 03 04 05",
                "223-45-678",
                "+86 10 6764 5489",
                "89262134598",
                "123",
                "abc",
            ],
            "country": [
                "US",
                "EG",
                "DZ",
                "CN",
                "RU",
                "GB",
                "Unknown"
            ],
            "something_else": ["a", "b", "c", "d", "e", "f", "g"],
        }
    )
    normalizer = PhoneNormalizer(df, "phone_num", "country")
    df["phone_num"] = normalizer.normalize()

    expected_df = pd.DataFrame(
        data={
            "phone_num": [11233219871, 20102030405, 21322345678, 861067645489, 89262134598, None, None],
            "country": ["US", "EG", "DZ", "CN", "RU", "GB", "Unknown"],
            "something_else": ["a", "b", "c", "d", "e", "f", "g"],
        }
    )
    expected_df["phone_num"] = expected_df["phone_num"].astype("Int64")

    assert_frame_equal(df, expected_df)


def test_float_phone_prefix_normalization():
    df = pd.DataFrame(
        data={
            "phone_num": [
                1233219871.0,
                102030405.0,
                22345678.0,
                861067645489.0,
                89262134598.0,
                123.0,
                np.nan,
            ],
            "country": [
                "US",
                "EG",
                "DZ",
                "CN",
                "RU",
                "GB",
                "Unknown"
            ],
            "something_else": ["a", "b", "c", "d", "e", "f", "g"],
        }
    )
    normalizer = PhoneNormalizer(df, "phone_num", "country")
    df["phone_num"] = normalizer.normalize()

    expected_df = pd.DataFrame(
        data={
            "phone_num": [11233219871, 20102030405, 21322345678, 861067645489, 89262134598, None, None],
            "country": ["US", "EG", "DZ", "CN", "RU", "GB", "Unknown"],
            "something_else": ["a", "b", "c", "d", "e", "f", "g"],
        }
    )
    expected_df["phone_num"] = expected_df["phone_num"].astype("Int64")

    assert_frame_equal(df, expected_df)

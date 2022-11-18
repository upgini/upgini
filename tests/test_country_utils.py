import pandas as pd

from pandas.testing import assert_frame_equal

from upgini.utils.country_utils import CountrySearchKeyDetector

detector = CountrySearchKeyDetector()


def test_is_country_column_by_values():
    df = pd.DataFrame({"cntr": ["not country"] * 8 + ["ES", "IT"]})

    assert detector.get_search_key_column(df) == "cntr"

    df = pd.DataFrame({"cntr": ["not country"] * 9 + ["IT"]})

    assert detector.get_search_key_column(df) is None

    df = pd.DataFrame({"cntr": [1, 2, 3, 4, 5]})

    assert detector.get_search_key_column(df) is None


def test_is_country_column_by_name():
    df = pd.DataFrame({"country": ["not country"] * 10})

    assert detector.get_search_key_column(df) == "country"


def test_country_to_iso_code_convertion():
    df = pd.DataFrame(
        {
            "country": [
                "Austria",
                "England",
                "Poland",
                "United Kingdom of Great Britain and Northern Ireland",
                "",
                "Unknown",
                "US",
                "United States",
            ]
        }
    )

    expected_df = df.copy()
    expected_df["country"] = ["AT", "GB", "PL", "GB", "", "Unknown", "US", "US"]

    df_with_code = CountrySearchKeyDetector.convert_country_to_iso_code(df, "country")

    assert_frame_equal(expected_df, df_with_code)

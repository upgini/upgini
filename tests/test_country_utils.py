from upgini.utils.country_utils import CountrySearchKeyDetector
import pandas as pd


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

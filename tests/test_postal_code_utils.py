import pandas as pd

from upgini.utils.postal_code_utils import PostalCodeSearchKeyDetector

detector = PostalCodeSearchKeyDetector()


def test_is_postal_code_column_by_column_name():
    df = pd.DataFrame({"pstlcd": ["123", "321", "345"]})

    assert detector.get_search_key_columns(df, []) == []

    df = pd.DataFrame({"zip": ["123", "321", "345"]})

    assert detector.get_search_key_columns(df, []) == ["zip"]

    df = pd.DataFrame({"zipcode": ["123", "321", "345"]})

    assert detector.get_search_key_columns(df, []) == ["zipcode"]

    df = pd.DataFrame({"zip_code": ["123", "321", "345"]})

    assert detector.get_search_key_columns(df, []) == ["zip_code"]

    df = pd.DataFrame({"postal_code": ["123", "321", "345"]})

    assert detector.get_search_key_columns(df, []) == ["postal_code"]

    df = pd.DataFrame({"postalcode": ["123", "321", "345"]})

    assert detector.get_search_key_columns(df, []) == ["postalcode"]

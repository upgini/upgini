from upgini.utils.postal_code_utils import PostalCodeSearchKeyDetector
import pandas as pd


detector = PostalCodeSearchKeyDetector()


def test_is_postal_code_column_by_column_name():
    df = pd.DataFrame({"pstlcd": ["123", "321", "345"]})

    assert detector.get_search_key_column(df) is None

    df = pd.DataFrame({"zip": ["123", "321", "345"]})

    assert detector.get_search_key_column(df) == "zip"

    df = pd.DataFrame({"zipcode": ["123", "321", "345"]})

    assert detector.get_search_key_column(df) == "zipcode"

    df = pd.DataFrame({"zip_code": ["123", "321", "345"]})

    assert detector.get_search_key_column(df) == "zip_code"

    df = pd.DataFrame({"postal_code": ["123", "321", "345"]})

    assert detector.get_search_key_column(df) == "postal_code"

    df = pd.DataFrame({"postalcode": ["123", "321", "345"]})

    assert detector.get_search_key_column(df) == "postalcode"

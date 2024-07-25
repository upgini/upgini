import pandas as pd

from upgini.utils.phone_utils import PhoneSearchKeyDetector

detector = PhoneSearchKeyDetector()


def test_is_phone_column_by_column_name():
    df = pd.DataFrame({"phne": ["123", "321", "345"]})

    assert detector.get_search_key_columns(df, []) == []

    df = pd.DataFrame({"cellphone": ["123", "321", "345"]})

    assert detector.get_search_key_columns(df, []) == ["cellphone"]

    df = pd.DataFrame({"msisdn": ["123", "321", "345"]})

    assert detector.get_search_key_columns(df, []) == ["msisdn"]

    df = pd.DataFrame({"phone": ["123", "321", "345"]})

    assert detector.get_search_key_columns(df, []) == ["phone"]

    df = pd.DataFrame({"phonenumber": ["123", "321", "345"]})

    assert detector.get_search_key_columns(df, []) == ["phonenumber"]

    df = pd.DataFrame({"phone_number": ["123", "321", "345"]})

    assert detector.get_search_key_columns(df, []) == ["phone_number"]

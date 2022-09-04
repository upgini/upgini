from upgini.utils.email_utils import EmailSearchKeyDetector
import pandas as pd


detector = EmailSearchKeyDetector()


def test_email_search_key_detection_by_name():
    df = pd.DataFrame({"email": ["123", "321", "555"]})

    assert detector.get_search_key_column(df) == "email"

    df = pd.DataFrame({"e-mail": ["123", "321", "555"]})

    assert detector.get_search_key_column(df) == "e-mail"

    df = pd.DataFrame({"e_mail": ["123", "321", "555"]})

    assert detector.get_search_key_column(df) == "e_mail"


def test_email_search_key_detection_by_values():
    df = pd.DataFrame({"eml": ["asdf@asdf.sad", "woei@skdjfh.fnj"] + ["12@3"] * 8})

    assert detector.get_search_key_column(df) == "eml"

    df = pd.DataFrame({"eml": ["asdf@asdf.sad"] + ["12@3"] * 9})

    assert detector.get_search_key_column(df) is None

    df = pd.DataFrame({"eml": [1, 2, 3, 4, 5]})

    assert detector.get_search_key_column(df) is None

import pandas as pd
from pandas.testing import assert_frame_equal

from upgini.metadata import SearchKey
from upgini.utils.email_utils import EmailSearchKeyConverter, EmailSearchKeyDetector

detector = EmailSearchKeyDetector()


def test_email_search_key_detection_by_name():
    df = pd.DataFrame({"email": ["123", "321", "555"]})

    assert detector.get_search_key_columns(df, []) == ["email"]

    df = pd.DataFrame({"e-mail": ["123", "321", "555"]})

    assert detector.get_search_key_columns(df, []) == ["e-mail"]

    df = pd.DataFrame({"e_mail": ["123", "321", "555"]})

    assert detector.get_search_key_columns(df, []) == ["e_mail"]


def test_email_search_key_detection_by_values():
    df = pd.DataFrame({"eml": ["asdf@asdf.sad", "woei@skdjfh.fnj"] + ["12@3"] * 8})

    assert detector.get_search_key_columns(df, []) == ["eml"]

    df = pd.DataFrame({"eml": ["asdf@asdf.sad"] + ["12@"] * 9})

    assert detector.get_search_key_columns(df, []) == []

    df = pd.DataFrame({"eml": [1, 2, 3, 4, 5]})

    assert detector.get_search_key_columns(df, []) == []


def test_convertion_to_hem():
    df = pd.DataFrame({"email": ["test@google.com", "", "@", None, 0.0, "asdf@oiouo@asdf"]})

    search_keys = {"email": SearchKey.EMAIL}
    columns_renaming = {"email": "original_email"}
    converter = EmailSearchKeyConverter("email", None, search_keys, columns_renaming, [])
    df = converter.convert(df)

    assert search_keys == {
        "email": SearchKey.EMAIL,
        "email" + EmailSearchKeyConverter.HEM_SUFFIX: SearchKey.HEM,
        "email" + EmailSearchKeyConverter.ONE_DOMAIN_SUFFIX: SearchKey.EMAIL_ONE_DOMAIN,
    }

    assert columns_renaming == {
        "email": "original_email",
        "email" + EmailSearchKeyConverter.HEM_SUFFIX: "original_email",
        "email" + EmailSearchKeyConverter.ONE_DOMAIN_SUFFIX: "original_email",
        # EmailSearchKeyConverter.DOMAIN_COLUMN_NAME: "original_email",
    }

    expected_df = pd.DataFrame(
        {
            "email": ["test@google.com", "", "@", None, 0.0, "asdf@oiouo@asdf"],
            "email" + EmailSearchKeyConverter.HEM_SUFFIX: [
                "8b0080a904da73e6e500ada3d09a88037289b5c08e03d3a09546ffacc5b5fd57",
                None,
                None,
                None,
                None,
                None,
            ],
            "email" + EmailSearchKeyConverter.ONE_DOMAIN_SUFFIX: ["tgoogle.com", None, None, None, None, None],
            # EmailSearchKeyConverter.DOMAIN_COLUMN_NAME: ["google.com", None, None, None, None, None],
        }
    )

    assert_frame_equal(expected_df, df)

    # assert df[EmailSearchKeyConverter.DOMAIN_COLUMN_NAME].astype("string").isnull().sum() == 5


def test_convertion_to_hem_with_existing_hem():
    df = pd.DataFrame(
        {
            "email": ["test@google.com", "", None, 0.0],
            "hem": ["8B0080A904DA73E6E500ADA3d09A88037289B5C08E03D3A09546FFACC5B5FD57", None, None, None],
        }
    )

    search_keys = {"email": SearchKey.EMAIL, "hem": SearchKey.HEM}
    columns_renaming = {"email": "original_email", "hem": "original_hem"}
    converter = EmailSearchKeyConverter("email", "hem", search_keys, columns_renaming, [])
    df = converter.convert(df)

    assert search_keys == {
        "email": SearchKey.EMAIL,
        "hem": SearchKey.HEM,
        "email" + EmailSearchKeyConverter.ONE_DOMAIN_SUFFIX: SearchKey.EMAIL_ONE_DOMAIN,
    }

    assert columns_renaming == {
        "email": "original_email",
        "hem": "original_hem",
        "email" + EmailSearchKeyConverter.ONE_DOMAIN_SUFFIX: "original_email",
        # EmailSearchKeyConverter.DOMAIN_COLUMN_NAME: "original_email",
    }

    expected_df = pd.DataFrame(
        {
            "email": ["test@google.com", "", None, 0.0],
            "hem": ["8b0080a904da73e6e500ada3d09a88037289b5c08e03d3a09546ffacc5b5fd57", None, None, None],
            "email" + EmailSearchKeyConverter.ONE_DOMAIN_SUFFIX: ["tgoogle.com", None, None, None],
            # EmailSearchKeyConverter.DOMAIN_COLUMN_NAME: ["google.com", None, None, None],
        }
    )
    expected_df["hem"] = expected_df["hem"].astype("string")

    assert_frame_equal(expected_df, df)

import pandas as pd
import pytest

from upgini import Dataset, FileColumnMeaningType


def test_etalon_validation(etalon: Dataset):
    Dataset.MIN_ROWS_COUNT = 3
    etalon.validate()
    assert "is_valid" in etalon.columns
    count = len(etalon)
    valid_count = int(etalon["is_valid"].sum())
    valid_rate = 100 * valid_count / count

    assert valid_count == 1
    valid_rate_expected = 100 * (1 / 6)
    assert valid_rate == pytest.approx(valid_rate_expected, abs=0.01)


def test_email_to_hem_convertion():
    df = pd.DataFrame(
        [
            {"email": "test@google.com"},
            {"email": None},
            {"email": "fake"},
        ]
    )
    dataset = Dataset("test", df=df)
    dataset.meaning_types = {
        "email": FileColumnMeaningType.EMAIL,
    }
    dataset.search_keys = [("email",)]
    dataset._Dataset__hash_email()
    assert "generated_hem" in dataset.columns.values
    assert "email_domain" in dataset.columns.values
    assert "email" not in dataset.columns.values


def test_ip_to_int_conversion():
    df = pd.DataFrame(
        [
            {"ip": "192.168.1.1"},
            {"ip": ""},
            {"ip": None},
        ]
    )
    dataset = Dataset("test", df=df)
    dataset.meaning_types = {
        "ip": FileColumnMeaningType.IP_ADDRESS,
    }
    dataset._Dataset__convert_ip()
    assert dataset["ip"].iloc[0] == 3232235777
    assert dataset["ip"].iloc[1] == -1
    assert dataset["ip"].iloc[2] == -1


def test_date_to_timestamp_convertion():
    df = pd.DataFrame(
        [
            {"date": "2020-01-01"},
            {"date": None},
            {"date": ""},
        ]
    )
    df["date"] = pd.to_datetime(df["date"])
    dataset = Dataset("test", df=df)
    dataset.meaning_types = {
        "date": FileColumnMeaningType.DATE,
    }
    dataset._Dataset__clean_empty_rows()
    dataset._Dataset__to_millis()
    assert dataset.shape[0] == 1
    assert dataset["date"].dtype == "int64"
    assert dataset["date"].iloc[0] == 1577836800000

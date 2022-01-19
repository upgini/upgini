import ipaddress
from datetime import date, datetime

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
    valid_rate_expected = 100 * (1 / 9)
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


def test_string_ip_to_int_conversion():
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
    assert dataset["ip"].dtype == "Int64"
    assert dataset["ip"].iloc[0] == 3232235777
    assert dataset["ip"].isnull().sum() == 2


def test_python_ip_to_int_conversion():
    df = pd.DataFrame(
        [
            {"ip": ipaddress.ip_address("192.168.1.1")},
        ]
    )
    dataset = Dataset("test", df=df)
    dataset.meaning_types = {
        "ip": FileColumnMeaningType.IP_ADDRESS,
    }
    dataset._Dataset__convert_ip()
    assert dataset["ip"].dtype == "Int64"
    assert dataset["ip"].iloc[0] == 3232235777


def test_int_ip_to_int_conversion():
    df = pd.DataFrame(
        [
            {"ip": 3232235777},
        ]
    )
    dataset = Dataset("test", df=df)
    dataset.meaning_types = {
        "ip": FileColumnMeaningType.IP_ADDRESS,
    }
    dataset._Dataset__convert_ip()
    assert dataset["ip"].iloc[0] == 3232235777


def test_string_date_to_timestamp_convertion():
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
    dataset._Dataset__to_millis()
    assert dataset["date"].dtype == "Int64"
    assert dataset["date"].iloc[0] == 1577836800000
    assert dataset["date"].isnull().sum() == 2
    dataset._Dataset__remove_empty_date_rows()
    assert dataset.shape[0] == 1


def test_string_datetime_to_timestamp_convertion():
    df = pd.DataFrame(
        [
            {"date": "2020-01-01T00:00:00Z"},
        ]
    )
    df["date"] = pd.to_datetime(df["date"])
    dataset = Dataset("test", df=df)
    dataset.meaning_types = {
        "date": FileColumnMeaningType.DATE,
    }
    dataset._Dataset__to_millis()
    dataset._Dataset__remove_empty_date_rows()
    assert dataset.shape[0] == 1
    assert dataset["date"].dtype == "Int64"
    assert dataset["date"].iloc[0] == 1577836800000


def test_period_range_to_timestamp_conversion():
    df = pd.DataFrame({"date": pd.period_range(start="2020-01-01", periods=3, freq="D")})
    print(df)
    dataset = Dataset("test2", df=df)
    dataset.meaning_types = {"date": FileColumnMeaningType.DATE}
    dataset._Dataset__to_millis()
    assert dataset["date"].dtype == "Int64"
    assert dataset["date"].iloc[0] == 1577836800000
    assert dataset["date"].iloc[1] == 1577923200000
    assert dataset["date"].iloc[2] == 1578009600000


def test_python_date_to_timestamp_conversion():
    df = pd.DataFrame(
        [
            {"date": date(2020, 1, 1)},
            {"date": date(2020, 1, 2)},
            {"date": date(2020, 1, 3)},
        ]
    )
    dataset = Dataset("test3", df=df)
    dataset.meaning_types = {"date": FileColumnMeaningType.DATE}
    dataset._Dataset__to_millis()
    assert dataset["date"].dtype == "Int64"
    assert dataset["date"].iloc[0] == 1577836800000
    assert dataset["date"].iloc[1] == 1577923200000
    assert dataset["date"].iloc[2] == 1578009600000


def test_python_datetime_to_timestamp_conversion():
    df = pd.DataFrame(
        [
            {"date": datetime(2020, 1, 1, 0, 0, 0)},
            {"date": datetime(2020, 1, 2, 0, 0, 0)},
            {"date": datetime(2020, 1, 3, 0, 0, 0)},
        ]
    )
    dataset = Dataset("test3", df=df)
    dataset.meaning_types = {"date": FileColumnMeaningType.DATE}
    dataset._Dataset__to_millis()
    assert dataset["date"].dtype == "Int64"
    assert dataset["date"].iloc[0] == 1577836800000
    assert dataset["date"].iloc[1] == 1577923200000
    assert dataset["date"].iloc[2] == 1578009600000

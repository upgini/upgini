import ipaddress
import random
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from upgini import Dataset, FileColumnMeaningType
from upgini.errors import ValidationError
from upgini.metadata import ModelTaskType


def test_etalon_validation(etalon: Dataset):
    print("Initial dataset:\n", etalon)
    Dataset.MIN_ROWS_COUNT = 1  # type: ignore
    count = len(etalon)
    etalon.validate()
    valid_count = len(etalon)
    valid_rate = 100 * valid_count / count

    assert valid_count == 1
    valid_rate_expected = 100 * (1 / 10)
    assert valid_rate == pytest.approx(valid_rate_expected, abs=0.01)


def test_email_to_hem_convertion():
    df = pd.DataFrame(
        [
            {"email": "test@google.com"},
            {"email": None},
            {"email": "fake"},
        ]
    )
    dataset = Dataset("test", df=df)  # type: ignore
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
    dataset = Dataset("test", df=df)  # type: ignore
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
    dataset = Dataset("test", df=df)  # type: ignore
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
    dataset = Dataset("test", df=df)  # type: ignore
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
    dataset = Dataset("test", df=df)  # type: ignore
    dataset.meaning_types = {
        "date": FileColumnMeaningType.DATE,
    }
    dataset._Dataset__to_millis()
    assert dataset.shape[0] == 3
    assert dataset["date"].dtype == "Int64"
    assert dataset["date"].iloc[0] == 1577836800000
    assert dataset["date"].isnull().sum() == 2


def test_string_datetime_to_timestamp_convertion():
    df = pd.DataFrame(
        [
            {"date": "2020-01-01T00:00:00Z"},
        ]
    )
    df["date"] = pd.to_datetime(df["date"])
    dataset = Dataset("test", df=df)  # type: ignore
    dataset.meaning_types = {
        "date": FileColumnMeaningType.DATE,
    }
    dataset._Dataset__to_millis()
    assert dataset.shape[0] == 1
    assert dataset["date"].dtype == "Int64"
    assert dataset["date"].iloc[0] == 1577836800000


def test_period_range_to_timestamp_conversion():
    df = pd.DataFrame({"date": pd.period_range(start="2020-01-01", periods=3, freq="D")})
    print(df)
    dataset = Dataset("test2", df=df)  # type: ignore
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
    dataset = Dataset("test3", df=df)  # type: ignore
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
    dataset = Dataset("test3", df=df)  # type: ignore
    dataset.meaning_types = {"date": FileColumnMeaningType.DATE}
    dataset._Dataset__to_millis()
    assert dataset["date"].dtype == "Int64"
    assert dataset["date"].iloc[0] == 1577836800000
    assert dataset["date"].iloc[1] == 1577923200000
    assert dataset["date"].iloc[2] == 1578009600000


def test_constant_and_empty_validation():
    df = pd.DataFrame(
        [{"phone": random.randint(1, 99999999999), "a": 1, "b": None}] * 995
        + [{"phone": random.randint(1, 99999999999), "a": 2, "b": 3}] * 5
    )
    dataset = Dataset("test4", df=df)  # type: ignore
    dataset.meaning_types = {
        "phone": FileColumnMeaningType.MSISDN,
        "a": FileColumnMeaningType.FEATURE,
        "b": FileColumnMeaningType.FEATURE,
    }
    dataset._Dataset__remove_empty_and_constant_features()
    assert list(dataset.columns) == ["phone"]


def test_imbalanced_target():
    df = pd.DataFrame({
        "system_record_id": range(2000),
        "phone": np.random.randint(10000000000, 99999999999, 2000),
        "f": ["123"] * 2000,
        "target": ["a"]*100 + ["b"] * 400 + ["c"] * 500 + ["d"] * 1000
    })
    dataset = Dataset("test123", df=df)  # type: ignore
    dataset.meaning_types = {
        "system_record_id": FileColumnMeaningType.SYSTEM_RECORD_ID,
        "phone": FileColumnMeaningType.MSISDN,
        "f": FileColumnMeaningType.FEATURE,
        "target": FileColumnMeaningType.TARGET,
    }
    dataset.task_type = ModelTaskType.MULTICLASS
    dataset._Dataset__resample()
    assert len(dataset) == 400
    value_counts = dataset["target"].value_counts()
    assert len(value_counts) == 4
    assert value_counts["a"] == 100
    assert value_counts["b"] == 100
    assert value_counts["c"] == 100
    assert value_counts["d"] == 100


def test_fail_on_small_class_observations():
    df = pd.DataFrame({
        "system_record_id": range(20),
        "phone": np.random.randint(10000000000, 99999999999, 20),
        "f": ["123"] * 20,
        "target": ["a"] + ["b"] * 4 + ["c"] * 5 + ["d"] * 10
    })
    dataset = Dataset("test123", df=df)  # type: ignore
    dataset.meaning_types = {
        "system_record_id": FileColumnMeaningType.SYSTEM_RECORD_ID,
        "phone": FileColumnMeaningType.MSISDN,
        "f": FileColumnMeaningType.FEATURE,
        "target": FileColumnMeaningType.TARGET,
    }
    dataset.task_type = ModelTaskType.MULTICLASS
    with pytest.raises(ValidationError, match=r".*The minimum number of observations for each class.*"):
        dataset._Dataset__resample()


def test_fail_on_too_many_classes():
    df = pd.DataFrame({
        "system_record_id": range(200),
        "phone": np.random.randint(10000000000, 99999999999, 200),
        "f": ["123"] * 200,
        "target": range(200)
    })
    dataset = Dataset("test123", df=df)  # type: ignore
    dataset.meaning_types = {
        "system_record_id": FileColumnMeaningType.SYSTEM_RECORD_ID,
        "phone": FileColumnMeaningType.MSISDN,
        "f": FileColumnMeaningType.FEATURE,
        "target": FileColumnMeaningType.TARGET,
    }
    dataset.task_type = ModelTaskType.MULTICLASS
    with pytest.raises(ValidationError, match=r".*The number of target classes .+ exceeds the allowed threshold.*"):
        dataset._Dataset__resample()


def test_iso_code_normalization():
    df = pd.DataFrame({
        "iso_code": ["  rU 1", " Uk", "G B "]
    })
    dataset = Dataset("test321", df=df)  # type: ignore
    dataset.meaning_types = {
        "iso_code": FileColumnMeaningType.COUNTRY
    }
    dataset._Dataset__normalize_iso_code()
    assert dataset.loc[0, "iso_code"] == "RU"
    assert dataset.loc[1, "iso_code"] == "GB"
    assert dataset.loc[2, "iso_code"] == "GB"


def test_postal_code_normalization():
    df = pd.DataFrame({
        "postal_code": ["  0ab-0123 ", "0123  3948  "]
    })
    dataset = Dataset("test321", df=df)  # type: ignore
    dataset.meaning_types = {
        "postal_code": FileColumnMeaningType.POSTAL_CODE
    }
    dataset._Dataset__normalize_postal_code()
    assert dataset.loc[0, "postal_code"] == "AB0123"
    assert dataset.loc[1, "postal_code"] == "1233948"


def test_number_postal_code_normalization():
    df = pd.DataFrame({
        "postal_code": [103305, 111222]
    })
    dataset = Dataset("test321", df=df)  # type: ignore
    dataset.meaning_types = {
        "postal_code": FileColumnMeaningType.POSTAL_CODE
    }
    dataset._Dataset__normalize_postal_code()
    assert dataset.loc[0, "postal_code"] == "103305"
    assert dataset.loc[1, "postal_code"] == "111222"


def test_old_dates_drop():
    df = pd.DataFrame({
        "date": ["2020-01-01", "2005-05-02", "1999-12-31", None]
    })
    dataset = Dataset("test", df=df)  # type: ignore
    dataset.meaning_types = {
        "date": FileColumnMeaningType.DATE
    }
    dataset._Dataset__to_millis()
    dataset._Dataset__remove_old_dates()
    assert len(dataset) == 3


def test_time_cutoff_from_str():
    df = pd.DataFrame({
        "date": ["2020-01-01 00:01:00", "2000-01-01 00:00:00", "1999-12-31 02:00:00", None]
    })
    dataset = Dataset("test", df=df)  # type: ignore
    dataset.meaning_types = {
        "date": FileColumnMeaningType.DATETIME
    }
    dataset.date_format = "%Y-%m-%d %H:%M:%S"
    dataset._Dataset__to_millis()
    assert dataset.loc[0, "date"] == 1577836800000
    assert dataset.loc[1, "date"] == 946684800000
    assert dataset.loc[2, "date"] == 946598400000
    assert pd.isnull(dataset.loc[3, "date"])


def test_time_cutoff_from_datetime():
    df = pd.DataFrame({
        "date": [datetime(2020, 1, 1, 0, 1, 0), datetime(2000, 1, 1, 0, 0, 0), datetime(1999, 12, 31, 2, 0, 0), None]
    })
    dataset = Dataset("test", df=df)  # type: ignore
    dataset.meaning_types = {
        "date": FileColumnMeaningType.DATETIME
    }
    dataset._Dataset__to_millis()
    assert dataset.loc[0, "date"] == 1577836800000
    assert dataset.loc[1, "date"] == 946684800000
    assert dataset.loc[2, "date"] == 946598400000
    assert pd.isnull(dataset.loc[3, "date"])


def test_time_cutoff_from_period():
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=24, freq="H")
    })
    print(df)
    dataset = Dataset("test", df=df)  # type: ignore
    dataset.meaning_types = {
        "date": FileColumnMeaningType.DATETIME
    }
    dataset._Dataset__to_millis()
    for i in range(24):
        assert dataset.loc[i, "date"] == 1577836800000


def test_time_cutoff_from_timestamp():
    df = pd.DataFrame({
        "date": [1577836800000000000, 1577840400000000000, 1577844000000000000]
    })
    print(df)
    dataset = Dataset("test", df=df)  # type: ignore
    dataset.meaning_types = {
        "date": FileColumnMeaningType.DATETIME
    }
    with pytest.raises(Exception, match="Unsupported type of date column date.*"):
        dataset._Dataset__to_millis()

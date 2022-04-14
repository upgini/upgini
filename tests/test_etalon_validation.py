import ipaddress
import random
from datetime import date, datetime

import pandas as pd
import numpy as np
import pytest
from requests_mock.mocker import Mocker

from upgini import Dataset, FeaturesEnricher, FileColumnMeaningType, SearchKey
from upgini.errors import ValidationError
from upgini.metadata import ModelTaskType


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


def test_constant_and_empty_validation():
    df = pd.DataFrame(
        [{"phone": random.randint(1, 99999999999), "a": 1, "b": None}] * 995
        + [{"phone": random.randint(1, 99999999999), "a": 2, "b": 3}] * 5
    )
    dataset = Dataset("test4", df=df)
    dataset.meaning_types = {
        "phone": FileColumnMeaningType.MSISDN,
        "a": FileColumnMeaningType.FEATURE,
        "b": FileColumnMeaningType.FEATURE,
    }
    dataset._Dataset__remove_empty_and_constant_features()
    assert list(dataset.columns) == ["phone"]


def test_imbalanced_target():
    df = pd.DataFrame({
        "phone": np.random.randint(10000000000, 99999999999, 2000),
        "f": ["123"] * 2000,
        "target": ["a"]*100 + ["b"] * 400 + ["c"] * 500 + ["d"] * 1000
    })
    df["system_record_id"] = df.apply(lambda row: hash(tuple(row)), axis=1)
    dataset = Dataset("test123", df=df)
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
        "phone": np.random.randint(10000000000, 99999999999, 20),
        "f": ["123"] * 20,
        "target": ["a"] + ["b"] * 4 + ["c"] * 5 + ["d"] * 10
    })
    df["system_record_id"] = df.apply(lambda row: hash(tuple(row)), axis=1)
    dataset = Dataset("test123", df=df)
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
        "phone": np.random.randint(10000000000, 99999999999, 200),
        "f": ["123"] * 200,
        "target": range(200)
    })
    df["system_record_id"] = df.apply(lambda row: hash(tuple(row)), axis=1)
    dataset = Dataset("test123", df=df)
    dataset.meaning_types = {
        "system_record_id": FileColumnMeaningType.SYSTEM_RECORD_ID,
        "phone": FileColumnMeaningType.MSISDN,
        "f": FileColumnMeaningType.FEATURE,
        "target": FileColumnMeaningType.TARGET,
    }
    dataset.task_type = ModelTaskType.MULTICLASS
    with pytest.raises(ValidationError, match=r".*The number of target classes .+ exceeds the allowed threshold.*"):
        dataset._Dataset__resample()

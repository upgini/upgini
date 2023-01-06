import ipaddress
import random
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from upgini import Dataset, FileColumnMeaningType
from upgini.errors import ValidationError
from upgini.metadata import ModelTaskType, SearchKey
from upgini.resource_bundle import bundle
from upgini.utils.datetime_utils import DateTimeSearchKeyConverter
from upgini.utils.email_utils import EmailSearchKeyConverter
from upgini.utils.features_validator import FeaturesValidator
from upgini.utils.warning_counter import WarningCounter


def test_etalon_validation(etalon: Dataset):
    print("Initial dataset:\n", etalon)
    Dataset.MIN_ROWS_COUNT = 1  # type: ignore
    count = len(etalon)
    etalon.validate()
    valid_count = len(etalon)
    valid_rate = 100 * valid_count / count

    assert valid_count == 4
    valid_rate_expected = 100 * (valid_count / 10)
    assert valid_rate == pytest.approx(valid_rate_expected, abs=0.01)


def test_email_to_hem_convertion():
    df = pd.DataFrame(
        [
            {"email": "test@google.com"},
            {"email": None},
            {"email": "fake"},
        ]
    )
    search_keys = {
        "email": SearchKey.EMAIL,
    }
    converter = EmailSearchKeyConverter("email", None, search_keys)
    df = converter.convert(df)
    assert EmailSearchKeyConverter.HEM_COLUMN_NAME in df.columns
    assert EmailSearchKeyConverter.DOMAIN_COLUMN_NAME in df.columns
    assert "email" not in df.columns


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
    converter = DateTimeSearchKeyConverter("date", "%Y-%m-%d")
    df = converter.convert(df)
    assert df.shape[0] == 3
    assert df["date"].dtype == "Int64"
    assert df.loc[0, "date"] == 1577836800000
    assert df["date"].isnull().sum() == 2


def test_string_datetime_to_timestamp_convertion():
    df = pd.DataFrame(
        [
            {"date": "2020-01-01T00:00:00Z"},
        ]
    )
    df["date"] = pd.to_datetime(df.date)
    converter = DateTimeSearchKeyConverter("date")
    df = converter.convert(df)
    assert df.shape == (1, 1)
    assert df["date"].dtype == "Int64"
    assert df.loc[0, "date"] == 1577836800000


def test_period_range_to_timestamp_conversion():
    df = pd.DataFrame({"date": pd.period_range(start="2020-01-01", periods=3, freq="D")})
    converter = DateTimeSearchKeyConverter("date")
    df = converter.convert(df)
    assert df["date"].dtype == "Int64"
    assert df["date"].iloc[0] == 1577836800000
    assert df["date"].iloc[1] == 1577923200000
    assert df["date"].iloc[2] == 1578009600000


def test_python_date_to_timestamp_conversion():
    df = pd.DataFrame(
        [
            {"date": date(2020, 1, 1)},
            {"date": date(2020, 1, 2)},
            {"date": date(2020, 1, 3)},
        ]
    )
    converter = DateTimeSearchKeyConverter("date")
    df = converter.convert(df)
    assert df["date"].dtype == "Int64"
    assert df["date"].iloc[0] == 1577836800000
    assert df["date"].iloc[1] == 1577923200000
    assert df["date"].iloc[2] == 1578009600000


def test_python_datetime_to_timestamp_conversion():
    df = pd.DataFrame(
        [
            {"date": datetime(2020, 1, 1, 0, 0, 0)},
            {"date": datetime(2020, 1, 2, 0, 0, 0)},
            {"date": datetime(2020, 1, 3, 0, 0, 0)},
        ]
    )
    converter = DateTimeSearchKeyConverter("date")
    df = converter.convert(df)
    assert df["date"].dtype == "Int64"
    assert df["date"].iloc[0] == 1577836800000
    assert df["date"].iloc[1] == 1577923200000
    assert df["date"].iloc[2] == 1578009600000


def test_constant_and_empty_validation():
    df = pd.DataFrame(
        [{"phone": random.randint(1, 99999999999), "a": 1, "b": None, "c": 0}] * 995
        + [{"phone": random.randint(1, 99999999999), "a": 2, "b": 3, "c": 1}] * 5
    )
    dataset = Dataset("test4", df=df)  # type: ignore
    dataset.meaning_types = {
        "phone": FileColumnMeaningType.MSISDN,
        "a": FileColumnMeaningType.FEATURE,
        "b": FileColumnMeaningType.FEATURE,
    }
    warnings_counter = WarningCounter()
    features_to_drop = FeaturesValidator().validate(df, ["a", "b", "c"], warnings_counter)
    assert features_to_drop == ["a", "b"]
    assert warnings_counter.has_warnings()


def test_imbalanced_target():
    df = pd.DataFrame(
        {
            "system_record_id": range(2000),
            "phone": np.random.randint(10000000000, 99999999999, 2000),
            "f": ["123"] * 2000,
            "target": ["a"] * 100 + ["b"] * 400 + ["c"] * 500 + ["d"] * 1000,
        }
    )
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
    for label in dataset["target"].unique():
        assert value_counts[label] == 100


def test_fail_on_small_class_observations():
    df = pd.DataFrame(
        {
            "system_record_id": range(20),
            "phone": np.random.randint(10000000000, 99999999999, 20),
            "f": ["123"] * 20,
            "target": ["a"] + ["b"] * 4 + ["c"] * 5 + ["d"] * 10,
        }
    )
    dataset = Dataset("test123", df=df)  # type: ignore
    dataset.meaning_types = {
        "system_record_id": FileColumnMeaningType.SYSTEM_RECORD_ID,
        "phone": FileColumnMeaningType.MSISDN,
        "f": FileColumnMeaningType.FEATURE,
        "target": FileColumnMeaningType.TARGET,
    }
    dataset.task_type = ModelTaskType.MULTICLASS
    with pytest.raises(ValidationError, match=bundle.get("dataset_rarest_class_less_min").format("a", 1, 100)):
        dataset._Dataset__resample()


def test_fail_on_too_many_classes():
    df = pd.DataFrame(
        {
            "system_record_id": range(200),
            "phone": np.random.randint(10000000000, 99999999999, 200),
            "f": ["123"] * 200,
            "target": range(200),
        }
    )
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
    df = pd.DataFrame({"iso_code": ["  rU 1", " Uk", "G B "]})
    dataset = Dataset("test321", df=df)  # type: ignore
    dataset.meaning_types = {"iso_code": FileColumnMeaningType.COUNTRY}
    dataset._Dataset__normalize_iso_code()
    assert dataset.loc[0, "iso_code"] == "RU"
    assert dataset.loc[1, "iso_code"] == "GB"
    assert dataset.loc[2, "iso_code"] == "GB"


def test_postal_code_normalization():
    df = pd.DataFrame({"postal_code": ["  0ab-0123 ", "0123  3948  "]})
    dataset = Dataset("test321", df=df)  # type: ignore
    dataset.meaning_types = {"postal_code": FileColumnMeaningType.POSTAL_CODE}
    dataset._Dataset__normalize_postal_code()
    assert dataset.loc[0, "postal_code"] == "AB0123"
    assert dataset.loc[1, "postal_code"] == "1233948"


def test_number_postal_code_normalization():
    df = pd.DataFrame({"postal_code": [103305, 111222]})
    dataset = Dataset("test321", df=df)  # type: ignore
    dataset.meaning_types = {"postal_code": FileColumnMeaningType.POSTAL_CODE}
    dataset._Dataset__normalize_postal_code()
    assert dataset.loc[0, "postal_code"] == "103305"
    assert dataset.loc[1, "postal_code"] == "111222"


def test_old_dates_drop():
    df = pd.DataFrame({"date": ["2020-01-01", "2005-05-02", "1999-12-31", None]})
    converter = DateTimeSearchKeyConverter("date")
    df = converter.convert(df)
    dataset = Dataset("test", df=df)  # type: ignore
    dataset.meaning_types = {"date": FileColumnMeaningType.DATE}
    dataset._Dataset__remove_old_dates()
    assert len(dataset) == 3


def test_time_cutoff_from_str():
    df = pd.DataFrame({"date": ["2020-01-01 00:01:00", "2000-01-01 00:00:00", "1999-12-31 02:00:00", None]})
    converter = DateTimeSearchKeyConverter("date", "%Y-%m-%d %H:%M:%S")
    dataset = converter.convert(df)

    assert dataset.loc[0, "date"] == 1577836800000
    assert dataset.loc[1, "date"] == 946684800000
    assert dataset.loc[2, "date"] == 946598400000
    assert pd.isnull(dataset.loc[3, "date"])

    assert "datetime_time_sin_1" in dataset.columns
    assert dataset.loc[0, "datetime_time_sin_1"] == pytest.approx(np.sin(2 * np.pi / 24 / 60), abs=0.000001)
    assert dataset.loc[1, "datetime_time_sin_1"] == 0.0
    assert dataset.loc[2, "datetime_time_sin_1"] == pytest.approx(np.sin(2 * np.pi / 12), abs=0.000001)
    assert pd.isnull(dataset.loc[3, "datetime_time_sin_1"])

    assert "datetime_time_cos_1" in dataset.columns
    assert dataset.loc[0, "datetime_time_cos_1"] == pytest.approx(np.cos(2 * np.pi / 24 / 60), abs=0.000001)
    assert dataset.loc[1, "datetime_time_cos_1"] == 1.0
    assert dataset.loc[2, "datetime_time_cos_1"] == pytest.approx(np.cos(2 * np.pi / 12), abs=0.000001)
    assert pd.isnull(dataset.loc[3, "datetime_time_cos_1"])

    assert "datetime_time_sin_2" in dataset.columns
    assert dataset.loc[0, "datetime_time_sin_2"] == pytest.approx(np.sin(2 * np.pi / 12 / 60), abs=0.000001)
    assert dataset.loc[1, "datetime_time_sin_2"] == 0.0
    assert dataset.loc[2, "datetime_time_sin_2"] == pytest.approx(np.sin(2 * np.pi / 6), abs=0.000001)
    assert pd.isnull(dataset.loc[3, "datetime_time_sin_2"])

    assert "datetime_time_cos_2" in dataset.columns
    assert dataset.loc[0, "datetime_time_cos_2"] == pytest.approx(np.cos(2 * np.pi / 12 / 60), abs=0.000001)
    assert dataset.loc[1, "datetime_time_cos_2"] == 1.0
    assert dataset.loc[2, "datetime_time_cos_2"] == pytest.approx(np.cos(2 * np.pi / 6), abs=0.000001)
    assert pd.isnull(dataset.loc[3, "datetime_time_cos_2"])

    assert "datetime_time_sin_24" in dataset.columns
    assert dataset.loc[0, "datetime_time_sin_24"] == pytest.approx(np.sin(2 * np.pi / 60), abs=0.000001)
    assert dataset.loc[1, "datetime_time_sin_24"] == 0.0
    assert dataset.loc[2, "datetime_time_sin_24"] == 0.0
    assert pd.isnull(dataset.loc[3, "datetime_time_sin_24"])

    assert "datetime_time_cos_24" in dataset.columns
    assert dataset.loc[0, "datetime_time_cos_24"] == pytest.approx(np.cos(2 * np.pi / 60), abs=0.000001)
    assert dataset.loc[1, "datetime_time_cos_24"] == 1.0
    assert dataset.loc[2, "datetime_time_cos_24"] == 1.0
    assert pd.isnull(dataset.loc[3, "datetime_time_cos_24"])

    assert "datetime_time_sin_48" in dataset.columns
    assert dataset.loc[0, "datetime_time_sin_48"] == pytest.approx(np.sin(2 * np.pi / 30), abs=0.000001)
    assert dataset.loc[1, "datetime_time_sin_48"] == 0.0
    assert dataset.loc[2, "datetime_time_sin_48"] == 0.0
    assert pd.isnull(dataset.loc[3, "datetime_time_sin_48"])

    assert "datetime_time_cos_48" in dataset.columns
    assert dataset.loc[0, "datetime_time_cos_48"] == pytest.approx(np.cos(2 * np.pi / 30), abs=0.000001)
    assert dataset.loc[1, "datetime_time_cos_48"] == 1.0
    assert dataset.loc[2, "datetime_time_cos_48"] == 1.0
    assert pd.isnull(dataset.loc[3, "datetime_time_cos_48"])


def test_time_cutoff_from_datetime():
    df = pd.DataFrame(
        {"date": [datetime(2020, 1, 1, 0, 1, 0), datetime(2000, 1, 1, 0, 0, 0), datetime(1999, 12, 31, 2, 0, 0), None]}
    )
    converter = DateTimeSearchKeyConverter("date")
    dataset = converter.convert(df)
    assert dataset.loc[0, "date"] == 1577836800000
    assert dataset.loc[1, "date"] == 946684800000
    assert dataset.loc[2, "date"] == 946598400000
    assert pd.isnull(dataset.loc[3, "date"])

    assert "datetime_time_sin_1" in dataset.columns
    assert dataset.loc[0, "datetime_time_sin_1"] == pytest.approx(np.sin(2 * np.pi / 24 / 60), abs=0.000001)
    assert dataset.loc[1, "datetime_time_sin_1"] == 0.0
    assert dataset.loc[2, "datetime_time_sin_1"] == pytest.approx(np.sin(2 * np.pi / 12), abs=0.000001)
    assert pd.isnull(dataset.loc[3, "datetime_time_sin_1"])

    assert "datetime_time_cos_1" in dataset.columns
    assert dataset.loc[0, "datetime_time_cos_1"] == pytest.approx(np.cos(2 * np.pi / 24 / 60), abs=0.000001)
    assert dataset.loc[1, "datetime_time_cos_1"] == 1.0
    assert dataset.loc[2, "datetime_time_cos_1"] == pytest.approx(np.cos(2 * np.pi / 12), abs=0.000001)
    assert pd.isnull(dataset.loc[3, "datetime_time_cos_1"])

    assert "datetime_time_sin_2" in dataset.columns
    assert dataset.loc[0, "datetime_time_sin_2"] == pytest.approx(np.sin(2 * np.pi / 12 / 60), abs=0.000001)
    assert dataset.loc[1, "datetime_time_sin_2"] == 0.0
    assert dataset.loc[2, "datetime_time_sin_2"] == pytest.approx(np.sin(2 * np.pi / 6), abs=0.000001)
    assert pd.isnull(dataset.loc[3, "datetime_time_sin_2"])

    assert "datetime_time_cos_2" in dataset.columns
    assert dataset.loc[0, "datetime_time_cos_2"] == pytest.approx(np.cos(2 * np.pi / 12 / 60), abs=0.000001)
    assert dataset.loc[1, "datetime_time_cos_2"] == 1.0
    assert dataset.loc[2, "datetime_time_cos_2"] == pytest.approx(np.cos(2 * np.pi / 6), abs=0.000001)
    assert pd.isnull(dataset.loc[3, "datetime_time_cos_2"])

    assert "datetime_time_sin_24" in dataset.columns
    assert dataset.loc[0, "datetime_time_sin_24"] == pytest.approx(np.sin(2 * np.pi / 60), abs=0.000001)
    assert dataset.loc[1, "datetime_time_sin_24"] == 0.0
    assert dataset.loc[2, "datetime_time_sin_24"] == 0.0
    assert pd.isnull(dataset.loc[3, "datetime_time_sin_24"])

    assert "datetime_time_cos_24" in dataset.columns
    assert dataset.loc[0, "datetime_time_cos_24"] == pytest.approx(np.cos(2 * np.pi / 60), abs=0.000001)
    assert dataset.loc[1, "datetime_time_cos_24"] == 1.0
    assert dataset.loc[2, "datetime_time_cos_24"] == 1.0
    assert pd.isnull(dataset.loc[3, "datetime_time_cos_24"])

    assert "datetime_time_sin_48" in dataset.columns
    assert dataset.loc[0, "datetime_time_sin_48"] == pytest.approx(np.sin(2 * np.pi / 30), abs=0.000001)
    assert dataset.loc[1, "datetime_time_sin_48"] == 0.0
    assert dataset.loc[2, "datetime_time_sin_48"] == 0.0
    assert pd.isnull(dataset.loc[3, "datetime_time_sin_48"])

    assert "datetime_time_cos_48" in dataset.columns
    assert dataset.loc[0, "datetime_time_cos_48"] == pytest.approx(np.cos(2 * np.pi / 30), abs=0.000001)
    assert dataset.loc[1, "datetime_time_cos_48"] == 1.0
    assert dataset.loc[2, "datetime_time_cos_48"] == 1.0
    assert pd.isnull(dataset.loc[3, "datetime_time_cos_48"])


def test_time_cutoff_from_period():
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=24, freq="H")})
    converter = DateTimeSearchKeyConverter("date")
    dataset = converter.convert(df)
    for i in range(24):
        assert dataset.loc[i, "date"] == 1577836800000

    assert "datetime_time_sin_1" in dataset.columns
    for i in range(24):
        assert dataset.loc[i, "datetime_time_sin_1"] == pytest.approx(np.sin(i * 2 * np.pi / 24), abs=0.000001)

    assert "datetime_time_cos_1" in dataset.columns
    for i in range(24):
        assert dataset.loc[i, "datetime_time_cos_1"] == pytest.approx(np.cos(i * 2 * np.pi / 24), abs=0.000001)

    assert "datetime_time_sin_2" in dataset.columns
    for i in range(24):
        assert dataset.loc[i, "datetime_time_sin_2"] == pytest.approx(np.sin(i * 2 * np.pi / 12), abs=0.000001)

    assert "datetime_time_cos_2" in dataset.columns
    for i in range(24):
        assert dataset.loc[i, "datetime_time_cos_2"] == pytest.approx(np.cos(i * 2 * np.pi / 12), abs=0.000001)

    assert "datetime_time_sin_24" in dataset.columns
    for i in range(24):
        assert dataset.loc[i, "datetime_time_sin_24"] == 0.0

    assert "datetime_time_cos_24" in dataset.columns
    for i in range(24):
        assert dataset.loc[i, "datetime_time_cos_24"] == 1.0

    assert "datetime_time_sin_48" in dataset.columns
    for i in range(24):
        assert dataset.loc[i, "datetime_time_sin_48"] == 0.0

    assert "datetime_time_cos_48" in dataset.columns
    for i in range(24):
        assert dataset.loc[i, "datetime_time_cos_48"] == 1.0


def test_time_cutoff_from_timestamp():
    df = pd.DataFrame({"date": [1577836800000000000, 1577840400000000000, 1577844000000000000]})
    converter = DateTimeSearchKeyConverter("date")
    with pytest.raises(Exception, match="Unsupported type of date column date.*"):
        converter.convert(df)

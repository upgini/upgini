import ipaddress
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from upgini.ads import FileColumnMeaningType
from upgini.dataset import Dataset
from upgini.errors import ValidationError
from upgini.metadata import SEARCH_KEY_UNNEST, ModelTaskType, SearchKey
from upgini.normalizer.normalize_utils import Normalizer
from upgini.resource_bundle import bundle
from upgini.utils.country_utils import CountrySearchKeyConverter
from upgini.utils.datetime_utils import DateTimeSearchKeyConverter
from upgini.utils.email_utils import EmailSearchKeyConverter
from upgini.utils.features_validator import FeaturesValidator
from upgini.utils.ip_utils import IpSearchKeyConverter
from upgini.utils.postal_code_utils import PostalCodeSearchKeyConverter


def test_etalon_validation(etalon: Dataset):
    print("Initial dataset:\n", etalon)
    Dataset.MIN_ROWS_COUNT = 1  # type: ignore
    count = len(etalon)
    etalon.columns_renaming = {c: c for c in etalon.data.columns}
    etalon.validate()
    valid_count = len(etalon)
    valid_rate = 100 * valid_count / count

    assert valid_count == 5
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
    columns_renaming = {"email": "original_email"}
    converter = EmailSearchKeyConverter("email", None, search_keys, columns_renaming, [])
    df = converter.convert(df)
    assert "email" + EmailSearchKeyConverter.HEM_SUFFIX in df.columns
    # assert EmailSearchKeyConverter.DOMAIN_COLUMN_NAME in df.columns
    assert "email" + EmailSearchKeyConverter.ONE_DOMAIN_SUFFIX in df.columns
    assert "email" not in df.columns
    assert converter.email_converted_to_hem
    assert columns_renaming == {
        "email" + EmailSearchKeyConverter.HEM_SUFFIX: "original_email",
        # EmailSearchKeyConverter.DOMAIN_COLUMN_NAME: "original_email",
        "email" + EmailSearchKeyConverter.ONE_DOMAIN_SUFFIX: "original_email",
    }


def test_unnest_email_to_hem_conversion():
    df = pd.DataFrame(
        {"upgini_email_unnest": ["test@google.com", None, "fake"], SEARCH_KEY_UNNEST: ["email", "email", "email"]}
    )
    search_keys = {
        "upgini_email_unnest": SearchKey.EMAIL,
    }
    columns_renaming = {"upgini_email_unnest": "upgini_email_unnest"}
    unnest_search_keys = ["upgini_email_unnest"]
    converter = EmailSearchKeyConverter("upgini_email_unnest", None, search_keys, columns_renaming, unnest_search_keys)
    df = converter.convert(df)
    assert "upgini_email_unnest" + EmailSearchKeyConverter.HEM_SUFFIX in df.columns
    # assert EmailSearchKeyConverter.DOMAIN_COLUMN_NAME in df.columns
    assert "upgini_email_unnest" + EmailSearchKeyConverter.ONE_DOMAIN_SUFFIX in df.columns
    assert "upgini_email_unnest" not in df.columns
    assert converter.email_converted_to_hem
    assert unnest_search_keys == ["upgini_email_unnest" + EmailSearchKeyConverter.HEM_SUFFIX]


def test_string_ip_to_int_conversion():
    df = pd.DataFrame(
        [
            {"ip": "192.168.1.1"},
            {"ip": ""},
            {"ip": None},
        ]
    )
    columns_renaming = {"ip": "original_ip"}
    converter = IpSearchKeyConverter("ip", {"ip": SearchKey.IP}, columns_renaming, [])
    converter.convert(df)
    # assert df["ip_v4"].dtype == "Int64"
    # assert df["ip_v4"].iloc[0] == 3232235777
    # assert df["ip_v4"].isnull().sum() == 2
    assert df["ip_v6"].dtype.name == "string"
    assert df["ip_v6"].iloc[0] == "281473913979137"
    assert df["ip_v6"].isnull().sum() == 2


def test_python_ip_to_int_conversion():
    df = pd.DataFrame(
        [
            {"ip": ipaddress.ip_address("192.168.1.1")},
        ]
    )
    columns_renaming = {"ip": "original_ip"}
    converter = IpSearchKeyConverter("ip", {"ip": SearchKey.IP}, columns_renaming, [])
    converter.convert(df)
    # assert df["ip_v4"].dtype == "Int64"
    # assert df["ip_v4"].iloc[0] == 3232235777
    assert df["ip_v6"].dtype.name == "string"
    assert df["ip_v6"].iloc[0] == "281473913979137"


def test_ip_v6_conversion():
    df = pd.DataFrame(
        {
            "ip": [
                "::cf:befe:525b",
                "2806:2f0:92c0:ffa4:30eb:4982:b4e:8a97",
                "2401:4900:3FC2:2CE6::C34:8174",
                "2409:4081:E14:CF75::F40A:B0F",
                "2402:8100:2102:AE2D::3CE0:E351",
            ]
        }
    )
    columns_renaming = {"ip": "original_ip"}
    converter = IpSearchKeyConverter("ip", {"ip": SearchKey.IP}, columns_renaming, [])
    converter.convert(df)
    # assert df["ip_v4"].dtype.name == "Int64"
    # assert df["ip_v4"].isna().all()
    assert df["ip_v6"].dtype.name == "string"
    assert df["ip_v6"].iloc[0] == "892262568539"
    assert df["ip_v6"].iloc[1] == "53200333237544187032231876373729151639"
    assert df["ip_v6"].iloc[2] == "47858880780748872078732893423110750580"
    assert df["ip_v6"].iloc[3] == "47900246818989331262222645018619415311"
    assert df["ip_v6"].iloc[4] == "47865208883029157842893923520652305233"


def test_int_ip_to_int_conversion():
    df = pd.DataFrame(
        {"ip": [3232235777, 892262568539]},
    )
    columns_renaming = {"ip": "original_ip"}
    converter = IpSearchKeyConverter("ip", {"ip": SearchKey.IP}, columns_renaming, [])
    converter.convert(df)
    # assert df["ip_v4"].iloc[0] == 3232235777
    assert df["ip_v6"].iloc[0] == "281473913979137"
    # assert df["ip_v4"].isnull().sum() == 1
    assert df["ip_v6"].iloc[1] == "892262568539"


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
    assert df.shape == (1, 3)
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
        {
            "phone": np.random.randint(1, 99999999999, 1000),
            "a": [1] * 995 + [2] * 5,
            "b": [None] * 995 + [3] * 5,
            "c": [0] * 995 + [1] * 5,
        }
    )
    features_to_drop, warnings = FeaturesValidator().validate(df, ["a", "b", "c"])
    assert features_to_drop == ["a", "b", "c"]
    assert len(warnings) == 1


def test_one_hot_encoding_validation():
    df = pd.DataFrame(
        {
            "phone": np.random.randint(1, 99999999999, 1000),
            "a": np.random.randint(1, 10, 1000),
            "b": np.random.rand(1000),
            "text_feature": ["text_" + str(n) for n in np.random.rand(1000)],
        }
    )
    df = pd.get_dummies(df, columns=["a"])
    features_columns = df.columns.to_list()
    features_columns.remove("phone")
    features_to_drop, warnings = FeaturesValidator().validate(df, features_columns, ["text_feature"])
    assert features_to_drop == []
    assert warnings == []


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
    dataset.MULTICLASS_MIN_SAMPLE_THRESHOLD = 1000
    dataset._Dataset__resample()
    assert len(dataset) == 1600
    value_counts = dataset.data["target"].value_counts()
    assert len(value_counts) == 4
    assert value_counts["a"] == 100
    assert value_counts["b"] == 400
    assert value_counts["c"] == 500
    assert value_counts["d"] == 600


def test_fail_on_small_class_observations():
    df = pd.DataFrame(
        {
            "system_record_id": range(20),
            "phone": np.random.randint(10000000000, 99999999999, 20),
            "f": ["123"] * 20,
            "target": ["a"] + ["b"] * 4 + ["c"] * 5 + ["d"] * 10,
        }
    )
    dataset = Dataset("test123", df=df)
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
    converter = CountrySearchKeyConverter("iso_code")
    df = converter.convert(df)
    assert df.loc[0, "iso_code"] == "RU"
    assert df.loc[1, "iso_code"] == "GB"
    assert df.loc[2, "iso_code"] == "GB"


def test_postal_code_normalization():
    df = pd.DataFrame({"postal_code": ["  0ab-0123 ", "0123  3948  "]})
    converter = PostalCodeSearchKeyConverter("postal_code")
    df = converter.convert(df)
    assert df.loc[0, "postal_code"] == "AB0123"
    assert df.loc[1, "postal_code"] == "1233948"


def test_number_postal_code_normalization():
    df = pd.DataFrame({"postal_code": [103305, 111222]})
    converter = PostalCodeSearchKeyConverter("postal_code")
    df = converter.convert(df)
    assert df.loc[0, "postal_code"] == "103305"
    assert df.loc[1, "postal_code"] == "111222"


def test_float_postal_code_normalization():
    df = pd.DataFrame({"postal_code": [103305.0, 111222.0]})
    converter = PostalCodeSearchKeyConverter("postal_code")
    df = converter.convert(df)
    assert df.loc[0, "postal_code"] == "103305"
    assert df.loc[1, "postal_code"] == "111222"


def test_float_string_postal_code_normalization():
    df = pd.DataFrame({"postal_code": ["103305.0", "111222.0"]})
    converter = PostalCodeSearchKeyConverter("postal_code")
    df = converter.convert(df)
    assert df.loc[0, "postal_code"] == "103305"
    assert df.loc[1, "postal_code"] == "111222"


def test_old_dates_drop():
    df = pd.DataFrame({"date": ["2020-01-01", "2005-05-02", "1999-12-31", None]})
    converter = DateTimeSearchKeyConverter("date")
    df = converter.convert(df)
    assert len(df[df.date.isna()]) == 2


def test_time_cutoff_from_str():
    df = pd.DataFrame({"date": ["2020-01-01 00:01:00", "2000-01-01 00:00:00", "1999-12-31 00:00:00", None]})
    converter = DateTimeSearchKeyConverter("date", "%Y-%m-%d %H:%M:%S")
    dataset = converter.convert(df)

    expected_data = pd.DataFrame(
        {
            "date": [1577836800000, 946684800000, None, None],
            "datetime_day_in_quarter_sin": [0.068991, 0.068991, None, None],
            "datetime_day_in_quarter_cos": [0.997617, 0.997617, None, None],
            "datetime_second_sin_60": [0.0, 0.0, None, None],
            "datetime_second_cos_60": [1.0, 1.0, None, None],
            "datetime_minute_sin_60": [0.104528, 0.000000, None, None],
            "datetime_minute_cos_60": [0.994522, 1.000000, None, None],
            "datetime_minute_sin_30": [0.207912, 0.000000, None, None],
            "datetime_minute_cos_30": [0.978148, 1.000000, None, None],
            "datetime_hour_sin_24": [0.0, 0.0, None, None],
            "datetime_hour_cos_24": [1.0, 1.0, None, None],
        }
    )

    expected_data["date"] = expected_data["date"].astype("Int64")
    assert_frame_equal(dataset, expected_data)


def test_time_cutoff_from_datetime():
    df = pd.DataFrame(
        {"date": [datetime(2020, 1, 1, 0, 1, 0), datetime(2000, 1, 1, 0, 0, 0), datetime(1999, 12, 31, 0, 0, 0), None]}
    )
    converter = DateTimeSearchKeyConverter("date")
    dataset = converter.convert(df)
    expected_data = pd.DataFrame(
        {
            "date": [1577836800000, 946684800000, None, None],
            "datetime_day_in_quarter_sin": [0.068991, 0.068991, None, None],
            "datetime_day_in_quarter_cos": [0.997617, 0.997617, None, None],
            "datetime_second_sin_60": [0.0, 0.0, None, None],
            "datetime_second_cos_60": [1.0, 1.0, None, None],
            "datetime_minute_sin_60": [0.104528, 0.000000, None, None],
            "datetime_minute_cos_60": [0.994522, 1.000000, None, None],
            "datetime_minute_sin_30": [0.207912, 0.000000, None, None],
            "datetime_minute_cos_30": [0.978148, 1.000000, None, None],
            "datetime_hour_sin_24": [0.0, 0.0, None, None],
            "datetime_hour_cos_24": [1.0, 1.0, None, None],
        }
    )

    expected_data["date"] = expected_data["date"].astype("Int64")
    assert_frame_equal(dataset, expected_data)


def test_time_cutoff_from_period():
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=24, freq="h")})
    converter = DateTimeSearchKeyConverter("date")
    dataset = converter.convert(df)
    for i in range(24):
        assert dataset.loc[i, "date"] == 1577836800000

    assert "datetime_day_in_quarter_sin" in dataset.columns
    for i in range(24):
        assert dataset.loc[i, "datetime_day_in_quarter_sin"] == 0.06899114440432493

    assert "datetime_day_in_quarter_cos" in dataset.columns
    for i in range(24):
        assert dataset.loc[i, "datetime_day_in_quarter_cos"] == 0.9976172723012476

    assert "datetime_second_sin_60" in dataset.columns
    for i in range(24):
        assert dataset.loc[i, "datetime_second_sin_60"] == 0.0

    assert "datetime_second_cos_60" in dataset.columns
    for i in range(24):
        assert dataset.loc[i, "datetime_second_cos_60"] == 1

    assert "datetime_minute_sin_60" in dataset.columns
    for i in range(24):
        assert dataset.loc[i, "datetime_minute_sin_60"] == 0.0

    assert "datetime_minute_cos_60" in dataset.columns
    for i in range(24):
        assert dataset.loc[i, "datetime_minute_cos_60"] == 1.0

    assert "datetime_minute_sin_30" in dataset.columns
    for i in range(24):
        assert dataset.loc[i, "datetime_minute_sin_30"] == 0.0

    assert "datetime_minute_cos_30" in dataset.columns
    for i in range(24):
        assert dataset.loc[i, "datetime_minute_cos_30"] == 1.0

    assert "datetime_hour_sin_24" in dataset.columns
    for i in range(24):
        assert dataset.loc[i, "datetime_hour_sin_24"] == pytest.approx(np.sin(i * 2 * np.pi / 24), abs=0.000001)

    assert "datetime_hour_cos_24" in dataset.columns
    for i in range(24):
        assert dataset.loc[i, "datetime_hour_cos_24"] == pytest.approx(np.cos(i * 2 * np.pi / 24), abs=0.000001)


def test_time_cutoff_from_timestamp():
    df = pd.DataFrame({"date": [1577836800000000000, 1577840400000000000, 1577844000000000000]})
    converter = DateTimeSearchKeyConverter("date")
    # with pytest.raises(Exception, match="Unsupported type of date column date.*"):
    df = converter.convert(df)
    assert len(df) == 3


def test_time_cutoff_with_different_timezones():
    df = pd.DataFrame(
        {
            "date": [
                "2018-01-02 00:00:00+02:00",
                "2021-05-26 00:00:00+03:00",
                "2018-03-20 00:00:00+02:00",
                "2018-01-29 00:00:00+02:00",
                "2018-02-12 00:00:00+02:00",
                "2018-10-02 00:00:00+03:00",
                "2019-09-18 00:00:00+03:00",
                "2022-03-09 00:00:00+02:00",
                "2022-06-02 00:00:00+03:00",
                "2021-09-27 00:00:00+03:00",
            ]
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    converter = DateTimeSearchKeyConverter("date")
    dataset = converter.convert(df)

    expected_df = pd.DataFrame(
        {
            "date": [
                "2018-01-02",
                "2021-05-26",
                "2018-03-20",
                "2018-01-29",
                "2018-02-12",
                "2018-10-02",
                "2019-09-18",
                "2022-03-09",
                "2022-06-02",
                "2021-09-27",
            ]
        }
    )
    expected_df = converter.convert(expected_df)
    assert_frame_equal(dataset, expected_df)


def test_date_in_diff_formats():
    expected_df = pd.DataFrame({"date": [1673481600000, 1676246400000, 1680220800000, None]})
    expected_df.date = expected_df.date.astype("Int64")

    df = pd.DataFrame({"date": ["12.01.23", "13.02.23", "31.03.23", "Date is not available"]})
    converter = DateTimeSearchKeyConverter("date")
    converted_df = converter.convert(df)
    assert_frame_equal(converted_df.iloc[:, :1], expected_df)

    df = pd.DataFrame({"date": ["01.12.23", "02.13.23", "03.31.23", "Date is not available"]})
    converter = DateTimeSearchKeyConverter("date")
    converted_df = converter.convert(df)
    assert_frame_equal(converted_df.iloc[:, :1], expected_df)

    df = pd.DataFrame({"date": ["2023-01-12", "2023-02-13", "2023-03-31", "Date is not available"]})
    converter = DateTimeSearchKeyConverter("date")
    converted_df = converter.convert(df)
    assert_frame_equal(converted_df.iloc[:, :1], expected_df)

    df = pd.DataFrame({"date": ["01.12.2023", "02.13.2023", "03.31.2023", "Date is not available"]})
    converter = DateTimeSearchKeyConverter("date")
    converted_df = converter.convert(df)
    assert_frame_equal(converted_df.iloc[:, :1], expected_df)

    df = pd.DataFrame({"date": ["01.12.23", "02.13.23", "13.13.23"]})
    converter = DateTimeSearchKeyConverter("date")
    with pytest.raises(Exception, match="Failed to parse date.*"):
        converted_df = converter.convert(df)


def test_datetime_with_ms():
    expected_df = pd.DataFrame(
        {
            "date": [1696636800000, 1695686400000, 1695600000000, 1695081600000],
            "datetime_day_in_quarter_sin": [0.460065, -0.269797, -0.334880, -0.682553],
            "datetime_day_in_quarter_cos": [0.887885, 0.962917, 0.942261, 0.730836],
            "datetime_second_sin_60": [-0.809017, 0.951057, -0.207912, -0.207912],
            "datetime_second_cos_60": [0.587785, 0.309017, -0.978148, 0.978148],
            "datetime_minute_sin_60": [0.669131, -0.587785, -0.913545, -0.978148],
            "datetime_minute_cos_60": [0.743145, -0.809017, 0.406737, 0.207912],
            "datetime_minute_sin_30": [0.994522, 0.951057, -0.743145, -0.406737],
            "datetime_minute_cos_30": [0.104528, 0.309017, -0.669131, -0.913545],
            "datetime_hour_sin_24": [0.965926, 0.500000, -0.965926, -1.000000],
            "datetime_hour_cos_24": [-0.258819, -0.866025, 0.258819, -1.836970e-16],
        }
    )
    expected_df.date = expected_df.date.astype("Int64")

    df = pd.DataFrame(
        {
            "date": [
                "2023-10-07T07:07:51.006677",
                "2023-09-26T10:36:12.885666",
                "2023-09-25T19:49:32.098655",
                "2023-09-19T18:47:58.268237",
            ]
        }
    )
    converter = DateTimeSearchKeyConverter("date")
    converted_df = converter.convert(df)
    print(converted_df)
    assert_frame_equal(converted_df, expected_df)


def test_columns_renaming():
    df1 = pd.DataFrame(
        {
            "date": ["2020-01-01"],
            "feature1": [123],
        }
    )

    df2 = pd.DataFrame(
        {
            "feature1": [123],
        }
    )

    df = pd.concat([df1, df2], axis=1)
    search_keys = {"date": SearchKey.DATE}
    normalizer = Normalizer()
    df, _, _ = normalizer.normalize(df, search_keys, [])
    assert set(df.columns.to_list()) == {"feature1_422b73", "date_0e8763", "feature1_422b73_0"}


def test_too_long_columns():
    too_long_column_name = "columnname" * 260
    df = pd.DataFrame(
        {
            "date": ["2020-01-01"],
            too_long_column_name: [123],
        }
    )

    search_keys = {"date": SearchKey.DATE}
    normalizer = Normalizer()
    df, _, _ = normalizer.normalize(df, search_keys, [])
    assert set(df.columns.to_list()) == {
        "date_0e8763",
        (
            "columnnamecolumnnamecolumnnamecolumnnamecolumnnamecolumnnamecolumnnamecolumnnamecolumnnamecolumnname"
            "columnnamecolumnnamecolumnnamecolumnnamecolumnnamecolumnnamecolumnnamecolumnnamecolumnnamecolumnname"
            "columnnamecolumnnamecolumnnamecolumnnamecolumnname_e1e25b"
        ),
    }


def test_downsampling_binary():
    df = pd.DataFrame(
        {
            "system_record_id": [0, 1, 2, 3, 4, 5],
            "date": [1673481600000, 1673481700000, 1673481800000, 1673481900000, 1673482000000, 1673482100000],
            "feature1": [123, 321, 456, 654, 987, 474],
            "target": [0, 0, 0, 1, 0, 1],
            "eval_set_index": [0, 0, 0, 0, 1, 1],
        }
    )

    meaning_types = {
        "system_record_id": FileColumnMeaningType.SYSTEM_RECORD_ID,
        "date": FileColumnMeaningType.DATE,
        "feature1": FileColumnMeaningType.FEATURE,
        "target": FileColumnMeaningType.TARGET,
        "eval_set_index": FileColumnMeaningType.EVAL_SET_INDEX,
    }
    dataset = Dataset("tds", df=df, meaning_types=meaning_types, search_keys=[("date",)])
    dataset.task_type = ModelTaskType.BINARY

    old_min_sample_threshold = Dataset.BINARY_MIN_SAMPLE_THRESHOLD
    old_min_target_class_rows = Dataset.MIN_TARGET_CLASS_ROWS
    old_imbalance_threshold = Dataset.IMBALANCE_THESHOLD
    old_fit_sample_threshold = Dataset.FIT_SAMPLE_THRESHOLD
    old_fit_sample_rows = Dataset.FIT_SAMPLE_ROWS
    Dataset.BINARY_MIN_SAMPLE_THRESHOLD = 3
    Dataset.MIN_TARGET_CLASS_ROWS = 1
    Dataset.IMBALANCE_THESHOLD = 0.6
    Dataset.FIT_SAMPLE_THRESHOLD = 1
    Dataset.FIT_SAMPLE_ROWS = 1

    try:
        dataset._Dataset__resample()
        assert len(dataset.data) == 1
    finally:
        Dataset.BINARY_MIN_SAMPLE_THRESHOLD = old_min_sample_threshold
        Dataset.MIN_TARGET_CLASS_ROWS = old_min_target_class_rows
        Dataset.IMBALANCE_THESHOLD = old_imbalance_threshold
        Dataset.FIT_SAMPLE_THRESHOLD = old_fit_sample_threshold
        Dataset.FIT_SAMPLE_ROWS = old_fit_sample_rows


def test_downsampling_multiclass():
    df = pd.DataFrame(
        {
            "system_record_id": [0, 1, 2, 3, 4, 5],
            "date": [1673481600000, 1673481700000, 1673481800000, 1673481900000, 1673482000000, 1673482100000],
            "feature1": [123, 321, 456, 654, 987, 474],
            "target": [0, 1, 2, 2, 0, 1],
            "eval_set_index": [0, 0, 0, 0, 1, 1],
        }
    )

    meaning_types = {
        "system_record_id": FileColumnMeaningType.SYSTEM_RECORD_ID,
        "date": FileColumnMeaningType.DATE,
        "feature1": FileColumnMeaningType.FEATURE,
        "target": FileColumnMeaningType.TARGET,
        "eval_set_index": FileColumnMeaningType.EVAL_SET_INDEX,
    }
    dataset = Dataset("tds", df=df, meaning_types=meaning_types, search_keys=[("date",)])
    dataset.task_type = ModelTaskType.MULTICLASS

    old_min_sample_threshold = Dataset.BINARY_MIN_SAMPLE_THRESHOLD
    old_min_target_class_rows = Dataset.MIN_TARGET_CLASS_ROWS
    old_imbalance_threshold = Dataset.IMBALANCE_THESHOLD
    old_fit_sample_threshold = Dataset.FIT_SAMPLE_THRESHOLD
    old_fit_sample_rows = Dataset.FIT_SAMPLE_ROWS
    Dataset.BINARY_MIN_SAMPLE_THRESHOLD = 3
    Dataset.MIN_TARGET_CLASS_ROWS = 1
    Dataset.IMBALANCE_THESHOLD = 0.8
    Dataset.FIT_SAMPLE_THRESHOLD = 1
    Dataset.FIT_SAMPLE_ROWS = 1

    try:
        dataset._Dataset__resample()
        assert len(dataset.data) == 1
    finally:
        Dataset.BINARY_MIN_SAMPLE_THRESHOLD = old_min_sample_threshold
        Dataset.MIN_TARGET_CLASS_ROWS = old_min_target_class_rows
        Dataset.IMBALANCE_THESHOLD = old_imbalance_threshold
        Dataset.FIT_SAMPLE_THRESHOLD = old_fit_sample_threshold
        Dataset.FIT_SAMPLE_ROWS = old_fit_sample_rows

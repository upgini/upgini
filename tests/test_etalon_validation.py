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
from upgini.utils.datetime_utils import DateTimeConverter
from upgini.utils.email_utils import EmailSearchKeyConverter
from upgini.utils.features_validator import FeaturesValidator
from upgini.utils.ip_utils import IpSearchKeyConverter
from upgini.utils.postal_code_utils import PostalCodeSearchKeyConverter
from upgini.utils.sample_utils import SampleConfig
from upgini.utils.target_utils import is_imbalanced


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
    assert "email" in df.columns
    assert converter.email_converted_to_hem
    assert columns_renaming == {
        "email": "original_email",
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
    assert "upgini_email_unnest" in df.columns
    assert converter.email_converted_to_hem
    assert unnest_search_keys == ["upgini_email_unnest", "upgini_email_unnest" + EmailSearchKeyConverter.HEM_SUFFIX]


def test_string_ip_to_bytes_conversion():
    df = pd.DataFrame(
        [
            {"ip": "192.168.1.1"},
            {"ip": "1050:0:0:0:5:600:300c:326b"},
            {"ip": ""},
            {"ip": None},
        ]
    )
    columns_renaming = {"ip": "original_ip"}
    search_keys = {"ip": SearchKey.IP}
    converter = IpSearchKeyConverter("ip", search_keys, columns_renaming, [])
    df = converter.convert(df)

    assert set(df.columns.to_list()) == {"ip_binary", "ip_prefix"}
    assert search_keys == {"ip_binary": SearchKey.IP_BINARY, "ip_prefix": SearchKey.IP_PREFIX}
    assert columns_renaming == {"ip_binary": "original_ip", "ip_prefix": "original_ip"}
    assert isinstance(df["ip_binary"].iloc[0], bytes)
    assert isinstance(df["ip_prefix"].iloc[0], str)
    assert df["ip_binary"].isnull().sum() == 2
    assert df["ip_prefix"].isnull().sum() == 2
    assert df["ip_binary"].iloc[0] == b"\xc0\xa8\x01\x01"
    assert df["ip_prefix"].iloc[0] == "192.168"
    assert df["ip_binary"].iloc[1] == b"\x10P\x00\x00\x00\x00\x00\x00\x00\x05\x06\x000\x0c2k"
    assert df["ip_prefix"].iloc[1] == "1050:0000"


def test_python_ip_to_bytes_conversion():
    df = pd.DataFrame(
        [
            {"ip": ipaddress.ip_address("192.168.1.1")},
        ]
    )
    columns_renaming = {"ip": "original_ip"}
    search_keys = {"ip": SearchKey.IP}
    converter = IpSearchKeyConverter("ip", search_keys, columns_renaming, [])
    df = converter.convert(df)

    assert set(df.columns.to_list()) == {"ip_binary", "ip_prefix"}
    assert search_keys == {"ip_binary": SearchKey.IP_BINARY, "ip_prefix": SearchKey.IP_PREFIX}
    assert columns_renaming == {"ip_binary": "original_ip", "ip_prefix": "original_ip"}
    assert isinstance(df["ip_binary"].iloc[0], bytes)
    assert isinstance(df["ip_prefix"].iloc[0], str)
    assert df["ip_binary"].iloc[0] == b"\xc0\xa8\x01\x01"
    assert df["ip_prefix"].iloc[0] == "192.168"


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
    df = converter.convert(df)

    assert set(df.columns.to_list()) == {"ip_binary", "ip_prefix"}
    assert isinstance(df["ip_binary"].iloc[0], bytes)
    assert df["ip_binary"].iloc[0] == b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xcf\xbe\xfeR["
    assert df["ip_binary"].iloc[1] == b"(\x06\x02\xf0\x92\xc0\xff\xa40\xebI\x82\x0bN\x8a\x97"
    assert df["ip_binary"].iloc[2] == b"$\x01I\x00?\xc2,\xe6\x00\x00\x00\x00\x0c4\x81t"
    assert df["ip_binary"].iloc[3] == b"$\t@\x81\x0e\x14\xcfu\x00\x00\x00\x00\xf4\n\x0b\x0f"
    assert df["ip_binary"].iloc[4] == b"$\x02\x81\x00!\x02\xae-\x00\x00\x00\x00<\xe0\xe3Q"
    assert df["ip_prefix"].iloc[0] == "0000:0000"
    assert df["ip_prefix"].iloc[1] == "2806:02f0"
    assert df["ip_prefix"].iloc[2] == "2401:4900"
    assert df["ip_prefix"].iloc[3] == "2409:4081"
    assert df["ip_prefix"].iloc[4] == "2402:8100"


def test_int_ip_to_bytes_conversion():
    df = pd.DataFrame(
        {"ip": [3232235777, 892262568539]},
    )
    columns_renaming = {"ip": "original_ip"}
    converter = IpSearchKeyConverter("ip", {"ip": SearchKey.IP}, columns_renaming, [])
    df = converter.convert(df)

    assert df["ip_binary"].iloc[0] == b"\xc0\xa8\x01\x01"
    assert df["ip_binary"].iloc[1] == b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xcf\xbe\xfeR["
    assert df["ip_prefix"].iloc[0] == "192.168"
    assert df["ip_prefix"].iloc[1] == "0000:0000"


def test_string_date_to_timestamp_convertion():
    df = pd.DataFrame(
        [
            {"date": "2020-01-01"},
            {"date": None},
            {"date": ""},
        ]
    )
    converter = DateTimeConverter("date", "%Y-%m-%d")
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
    converter = DateTimeConverter("date")
    df = converter.convert(df)
    assert df.shape == (1, 1)
    assert df["date"].dtype == "Int64"
    assert df.loc[0, "date"] == 1577836800000


def test_period_range_to_timestamp_conversion():
    df = pd.DataFrame({"date": pd.period_range(start="2020-01-01", periods=3, freq="D")})
    converter = DateTimeConverter("date")
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
    converter = DateTimeConverter("date")
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
    converter = DateTimeConverter("date")
    df = converter.convert(df)
    assert df["date"].dtype == "Int64"
    assert df["date"].iloc[0] == 1577836800000
    assert df["date"].iloc[1] == 1577923200000
    assert df["date"].iloc[2] == 1578009600000


def test_convert_datetime_without_cyclical_features():
    df = pd.DataFrame(
        [
            {"date": datetime(2020, 1, 1, 12, 30, 0)},
            {"date": datetime(2020, 1, 2, 13, 40, 0)},
            {"date": datetime(2020, 1, 3, 14, 50, 0)},
        ]
    )
    converter = DateTimeConverter("date", generate_cyclical_features=False)
    df = converter.convert(df)
    assert df.columns.to_list() == ["date"]
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
            "c": [0] * 995 + [1] * 5,  # like One-Hot encoding
        }
    )
    features_to_drop, warnings = FeaturesValidator().validate(df, ["a", "b", "c"])
    assert features_to_drop == ["a", "b"]
    assert len(warnings) == 2


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
    dataset = Dataset("test123", df=df, is_imbalanced=True)
    dataset.meaning_types = {
        "system_record_id": FileColumnMeaningType.SYSTEM_RECORD_ID,
        "phone": FileColumnMeaningType.MSISDN,
        "f": FileColumnMeaningType.FEATURE,
        "target": FileColumnMeaningType.TARGET,
    }
    dataset.task_type = ModelTaskType.MULTICLASS
    dataset.sample_config = SampleConfig(force_sample_size=7000, multiclass_min_sample_threshold=1000)
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

    with pytest.raises(ValidationError, match=bundle.get("dataset_rarest_class_less_min").format("a", 1, 100)):
        is_imbalanced(df, ModelTaskType.MULTICLASS, SampleConfig(), bundle)


def test_fail_on_too_many_classes():
    df = pd.DataFrame(
        {
            "system_record_id": range(200),
            "phone": np.random.randint(10000000000, 99999999999, 200),
            "f": ["123"] * 200,
            "target": range(200),
        }
    )

    with pytest.raises(ValidationError, match=r".*The number of target classes .+ exceeds the allowed threshold.*"):
        is_imbalanced(df, ModelTaskType.MULTICLASS, SampleConfig(), bundle)


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
    converter = DateTimeConverter("date")
    df = converter.convert(df)
    assert len(df[df.date.isna()]) == 2


def test_time_cutoff_from_str():
    df = pd.DataFrame({"date": ["2020-01-01 00:01:00", "2000-01-01 00:00:00", "1999-12-31 00:00:00", None]})
    converter = DateTimeConverter("date", "%Y-%m-%d %H:%M:%S")
    dataset = converter.convert(df)

    expected_data = pd.DataFrame(
        {
            "date": [1577836800000, 946684800000, None, None],
            "datetime_minute_sin_60": [0.104528, 0.000000, None, None],
            "datetime_minute_cos_60": [0.994522, 1.000000, None, None],
            "datetime_minute_sin_30": [0.207912, 0.000000, None, None],
            "datetime_minute_cos_30": [0.978148, 1.000000, None, None],
        }
    )

    expected_data["date"] = expected_data["date"].astype("Int64")
    assert_frame_equal(dataset, expected_data)


def test_time_cutoff_from_datetime():
    df = pd.DataFrame(
        {"date": [datetime(2020, 1, 1, 0, 1, 0), datetime(2000, 1, 1, 0, 0, 0), datetime(1999, 12, 31, 0, 0, 0), None]}
    )
    converter = DateTimeConverter("date")
    dataset = converter.convert(df)
    expected_data = pd.DataFrame(
        {
            "date": [1577836800000, 946684800000, None, None],
            "datetime_minute_sin_60": [0.104528, 0.000000, None, None],
            "datetime_minute_cos_60": [0.994522, 1.000000, None, None],
            "datetime_minute_sin_30": [0.207912, 0.000000, None, None],
            "datetime_minute_cos_30": [0.978148, 1.000000, None, None],
        }
    )

    expected_data["date"] = expected_data["date"].astype("Int64")
    assert_frame_equal(dataset, expected_data)


def test_time_cutoff_from_period():
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=24, freq="h")})
    converter = DateTimeConverter("date")
    dataset = converter.convert(df)
    for i in range(24):
        assert dataset.loc[i, "date"] == 1577836800000

    assert "datetime_hour_sin_24" in dataset.columns
    for i in range(24):
        assert dataset.loc[i, "datetime_hour_sin_24"] == pytest.approx(np.sin(i * 2 * np.pi / 24), abs=0.000001)

    assert "datetime_hour_cos_24" in dataset.columns
    for i in range(24):
        assert dataset.loc[i, "datetime_hour_cos_24"] == pytest.approx(np.cos(i * 2 * np.pi / 24), abs=0.000001)


def test_time_cutoff_from_timestamp():
    df = pd.DataFrame({"date": [1577836800000000000, 1577840400000000000, 1577844000000000000]})
    converter = DateTimeConverter("date")
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
    converter = DateTimeConverter("date")
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
    converter = DateTimeConverter("date")
    converted_df = converter.convert(df)
    assert_frame_equal(converted_df.iloc[:, :1], expected_df)

    df = pd.DataFrame({"date": ["01.12.23", "02.13.23", "03.31.23", "Date is not available"]})
    converter = DateTimeConverter("date")
    converted_df = converter.convert(df)
    assert_frame_equal(converted_df.iloc[:, :1], expected_df)

    df = pd.DataFrame({"date": ["2023-01-12", "2023-02-13", "2023-03-31", "Date is not available"]})
    converter = DateTimeConverter("date")
    converted_df = converter.convert(df)
    assert_frame_equal(converted_df.iloc[:, :1], expected_df)

    df = pd.DataFrame({"date": ["01.12.2023", "02.13.2023", "03.31.2023", "Date is not available"]})
    converter = DateTimeConverter("date")
    converted_df = converter.convert(df)
    assert_frame_equal(converted_df.iloc[:, :1], expected_df)

    df = pd.DataFrame({"date": ["01.12.23", "02.13.23", "13.13.23"]})
    converter = DateTimeConverter("date")
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
    converter = DateTimeConverter("date")
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
    dataset.sample_config = SampleConfig(
        force_sample_size=7000,
        binary_min_sample_threshold=3,
        fit_sample_threshold=1,
        fit_sample_rows=1,
        fit_sample_rows_with_eval_set=1,
        fit_sample_threshold_with_eval_set=1,
    )
    dataset.MIN_TARGET_CLASS_ROWS = 1
    dataset.IMBALANCE_THESHOLD = 0.6

    dataset._Dataset__resample()
    assert len(dataset.data) == 1


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

    dataset.sample_config = SampleConfig(
        force_sample_size=7000,
        multiclass_min_sample_threshold=3,
        fit_sample_threshold=1,
        fit_sample_rows=1,
        fit_sample_rows_with_eval_set=1,
        fit_sample_threshold_with_eval_set=1,
    )
    dataset.MIN_TARGET_CLASS_ROWS = 1
    dataset.IMBALANCE_THESHOLD = 0.8

    dataset._Dataset__resample()
    assert len(dataset.data) == 1

import pandas as pd

from upgini.ads import FileColumnMeaningType
from upgini.metadata import SearchKey
from upgini.utils.search_key_derivations import (
    get_derivable_transform_columns,
    get_missing_features_for_transform,
    is_derivable_transform_column,
)


def test_ip_derived_columns_from_search_keys():
    ip_column = "ip_bb9af5"
    search_keys = {ip_column: SearchKey.IP}

    assert get_derivable_transform_columns(search_keys) == {f"{ip_column}_binary", f"{ip_column}_prefix"}


def test_email_derived_columns_from_search_keys_without_hem():
    email_column = "eml_13033c"
    search_keys = {email_column: SearchKey.EMAIL}

    assert get_derivable_transform_columns(search_keys) == {
        f"{email_column}_hem",
        f"{email_column}_one_domain",
        f"{email_column}_domain",
    }


def test_email_derived_columns_skip_hem_when_separate_hem_key():
    email_column = "eml_13033c"
    hem_column = "hem_abc123"
    search_keys = {email_column: SearchKey.EMAIL, hem_column: SearchKey.HEM}

    assert get_derivable_transform_columns(search_keys) == {
        f"{email_column}_one_domain",
        f"{email_column}_domain",
    }


def test_transform_does_not_require_ip_derived_columns():
    ip_column = "ip_bb9af5"
    df = pd.DataFrame({ip_column: ["2806:2f0:92c0:ffa4:30eb:4982:b4e:8a97"], "feature_a": [1.0]})
    search_keys = {ip_column: SearchKey.IP}
    features_for_transform = [f"{ip_column}_binary", f"{ip_column}_prefix", "feature_a"]

    assert get_missing_features_for_transform(features_for_transform, df.columns, search_keys, {}) == []


def test_transform_does_not_require_ip_derived_columns_when_ip_demoted_to_custom_key():
    ip_column = "ip_bb9af5"
    df = pd.DataFrame({ip_column: ["2806:2f0:92c0:ffa4:30eb:4982:b4e:8a97"], "feature_a": [1.0]})
    search_keys = {ip_column: SearchKey.CUSTOM_KEY}
    source_search_keys = {"ip": SearchKey.IP}
    columns_renaming = {ip_column: "ip"}
    fit_columns_renaming = {
        ip_column: "ip",
        f"{ip_column}_binary": "ip",
        f"{ip_column}_prefix": "ip",
    }
    features_for_transform = [f"{ip_column}_binary", f"{ip_column}_prefix", "feature_a"]

    missing = get_missing_features_for_transform(
        features_for_transform,
        df.columns,
        search_keys,
        columns_renaming,
        source_search_keys=source_search_keys,
        fit_columns_renaming=fit_columns_renaming,
    )

    assert missing == []


def test_transform_does_not_require_email_derived_columns():
    email_column = "eml_13033c"
    df = pd.DataFrame({email_column: ["user@example.com"]})
    search_keys = {email_column: SearchKey.EMAIL}
    features_for_transform = [
        f"{email_column}_hem",
        f"{email_column}_one_domain",
        f"{email_column}_domain",
    ]

    assert get_missing_features_for_transform(features_for_transform, df.columns, search_keys, {}) == []


def test_transform_still_requires_missing_non_derivable_features():
    ip_column = "ip_bb9af5"
    df = pd.DataFrame({ip_column: ["2806:2f0:92c0:ffa4:30eb:4982:b4e:8a97"]})
    search_keys = {ip_column: SearchKey.IP}

    missing = get_missing_features_for_transform(
        [f"{ip_column}_binary", "feature_a"], df.columns, search_keys, {}
    )

    assert missing == ["feature_a"]


def test_is_derivable_transform_column_matches_by_suffix():
    email_column = "eml_13033c"
    search_keys = {email_column: SearchKey.EMAIL}

    assert is_derivable_transform_column(f"{email_column}_one_domain", search_keys)
    assert not is_derivable_transform_column("feature_a", search_keys)


def test_is_derivable_transform_column_uses_fit_columns_renaming():
    ip_column = "ip_bb9af5"
    search_keys = {ip_column: SearchKey.CUSTOM_KEY}
    fit_columns_renaming = {
        ip_column: "ip",
        f"{ip_column}_binary": "ip",
        f"{ip_column}_prefix": "ip",
    }

    assert is_derivable_transform_column(
        f"{ip_column}_binary",
        search_keys,
        columns_renaming={ip_column: "ip"},
        source_search_keys={"ip": SearchKey.IP},
        fit_columns_renaming=fit_columns_renaming,
        df_columns=[ip_column],
    )


def test_ip_prefix_meaning_type_maps_to_ip_search_key():
    assert SearchKey.from_meaning_type(FileColumnMeaningType.IP_PREFIX) == SearchKey.IP
    assert SearchKey.from_meaning_type(FileColumnMeaningType.IP_BINARY) == SearchKey.IP


def test_email_one_domain_meaning_type_maps_to_email_search_key():
    assert SearchKey.from_meaning_type(FileColumnMeaningType.EMAIL_ONE_DOMAIN) == SearchKey.EMAIL

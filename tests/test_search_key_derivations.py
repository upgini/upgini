from upgini.ads import FileColumnMeaningType
from upgini.metadata import SearchKey
from upgini.utils.search_key_derivations import get_derived_search_key_columns


def test_ip_derived_columns_from_search_keys():
    ip_column = "ip_bb9af5"
    search_keys = {ip_column: SearchKey.IP}

    assert get_derived_search_key_columns(search_keys) == {f"{ip_column}_binary", f"{ip_column}_prefix"}


def test_email_derived_columns_from_search_keys_without_hem():
    email_column = "eml_13033c"
    search_keys = {email_column: SearchKey.EMAIL}

    assert get_derived_search_key_columns(search_keys) == {
        f"{email_column}_hem",
        f"{email_column}_one_domain",
    }


def test_email_derived_columns_skip_hem_when_separate_hem_key():
    email_column = "eml_13033c"
    hem_column = "hem_abc123"
    search_keys = {email_column: SearchKey.EMAIL, hem_column: SearchKey.HEM}

    assert get_derived_search_key_columns(search_keys) == {f"{email_column}_one_domain"}


def test_ip_prefix_meaning_type_maps_to_ip_search_key():
    assert SearchKey.from_meaning_type(FileColumnMeaningType.IP_PREFIX) == SearchKey.IP
    assert SearchKey.from_meaning_type(FileColumnMeaningType.IP_BINARY) == SearchKey.IP


def test_email_one_domain_meaning_type_maps_to_email_search_key():
    assert SearchKey.from_meaning_type(FileColumnMeaningType.EMAIL_ONE_DOMAIN) == SearchKey.EMAIL


def test_derived_ip_columns_are_dropped_from_features_for_transform():
    ip_column = "ip_bb9af5"
    features_for_transform = [f"{ip_column}_prefix", f"{ip_column}_binary", "client_feature"]
    derived = get_derived_search_key_columns({ip_column: SearchKey.IP})

    assert [f for f in features_for_transform if f not in derived] == ["client_feature"]

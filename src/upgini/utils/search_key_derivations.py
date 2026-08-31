from upgini.metadata import SearchKey
from upgini.utils.email_utils import EmailSearchKeyConverter
from upgini.utils.ip_utils import IpSearchKeyConverter


def get_derived_search_key_columns(search_keys: dict[str, SearchKey]) -> set[str]:
    """Columns created from search keys after the transform missing-features check.

    IpSearchKeyConverter and EmailSearchKeyConverter run later (after explode), so these
    names must not be required from the user or hashed into ENTITY_SYSTEM_RECORD_ID yet.
    """
    derived: set[str] = set()
    has_separate_hem = any(key_type == SearchKey.HEM for key_type in search_keys.values())

    for column, key_type in search_keys.items():
        if key_type == SearchKey.IP:
            derived.add(column + IpSearchKeyConverter.BINARY_SUFFIX)
            derived.add(column + IpSearchKeyConverter.PREFIX_SUFFIX)
        elif key_type == SearchKey.EMAIL:
            derived.add(column + EmailSearchKeyConverter.ONE_DOMAIN_SUFFIX)
            if not has_separate_hem:
                derived.add(column + EmailSearchKeyConverter.HEM_SUFFIX)

    return derived

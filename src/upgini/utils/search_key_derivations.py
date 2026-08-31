from upgini.metadata import SearchKey
from upgini.utils.email_utils import EmailDomainGenerator, EmailSearchKeyConverter
from upgini.utils.ip_utils import IpSearchKeyConverter

# Columns produced from a source search key during transform, keyed by suffix.
# EmailSearchKeyConverter and IpSearchKeyConverter run after the missing-features check.
DERIVED_COLUMN_SUFFIXES: dict[str, SearchKey] = {
    IpSearchKeyConverter.BINARY_SUFFIX: SearchKey.IP,
    IpSearchKeyConverter.PREFIX_SUFFIX: SearchKey.IP,
    EmailSearchKeyConverter.HEM_SUFFIX: SearchKey.EMAIL,
    EmailSearchKeyConverter.ONE_DOMAIN_SUFFIX: SearchKey.EMAIL,
}

# Columns produced earlier in transform (before the missing-features check).
PRE_TRANSFORM_DERIVED_COLUMN_SUFFIXES: dict[str, SearchKey] = {
    EmailDomainGenerator.DOMAIN_SUFFIX: SearchKey.EMAIL,
}

IP_SOURCE_KEYS = (SearchKey.IP, SearchKey.IPV6_ADDRESS)
EMAIL_SOURCE_KEYS = (SearchKey.EMAIL,)


def _has_separate_hem_column(search_keys: dict[str, SearchKey]) -> bool:
    return any(key_type == SearchKey.HEM for key_type in search_keys.values())


def _column_original_name(
    column: str,
    fit_columns_renaming: dict[str, str] | None,
    columns_renaming: dict[str, str],
) -> str:
    if fit_columns_renaming and column in fit_columns_renaming:
        return fit_columns_renaming[column]
    return columns_renaming.get(column, column)


def _is_source_key_type(key_type: SearchKey | None, source_key: SearchKey) -> bool:
    if key_type is None:
        return False
    if source_key == SearchKey.IP:
        return key_type in IP_SOURCE_KEYS
    return key_type == source_key


def _resolve_source_key_type(
    column: str,
    column_original: str,
    search_keys: dict[str, SearchKey],
    source_search_keys: dict[str, SearchKey] | None,
) -> SearchKey | None:
    key_type = search_keys.get(column)
    if key_type is not None and key_type != SearchKey.CUSTOM_KEY:
        return key_type

    if source_search_keys is None:
        return key_type

    if column_original in source_search_keys:
        return source_search_keys[column_original]
    if column in source_search_keys:
        return source_search_keys[column]
    return key_type


def _get_source_columns_by_type(
    source_key: SearchKey,
    search_keys: dict[str, SearchKey],
    source_search_keys: dict[str, SearchKey] | None,
    columns_renaming: dict[str, str],
    fit_columns_renaming: dict[str, str] | None,
    df_columns,
) -> set[str]:
    sources: set[str] = set()
    all_columns = set(search_keys.keys()) | set(df_columns)

    for column in all_columns:
        column_original = _column_original_name(column, fit_columns_renaming, columns_renaming)
        resolved_type = _resolve_source_key_type(column, column_original, search_keys, source_search_keys)
        if _is_source_key_type(resolved_type, source_key):
            sources.add(column)

    if source_search_keys is not None:
        reverse_transform_renaming = {original: hashed for hashed, original in columns_renaming.items()}
        for original_column, key_type in source_search_keys.items():
            if not _is_source_key_type(key_type, source_key):
                continue
            hashed_column = reverse_transform_renaming.get(original_column, original_column)
            if hashed_column in all_columns or hashed_column in search_keys:
                sources.add(hashed_column)

    return sources


def _suffixes_for_source(source_key: SearchKey, *, has_separate_hem: bool) -> tuple[str, ...]:
    if source_key == SearchKey.IP:
        return (IpSearchKeyConverter.BINARY_SUFFIX, IpSearchKeyConverter.PREFIX_SUFFIX)
    if source_key == SearchKey.EMAIL:
        suffixes = [EmailSearchKeyConverter.ONE_DOMAIN_SUFFIX, EmailDomainGenerator.DOMAIN_SUFFIX]
        if not has_separate_hem:
            suffixes.append(EmailSearchKeyConverter.HEM_SUFFIX)
        return tuple(suffixes)
    return ()


def _feature_has_derivable_suffix(feature: str, source_key: SearchKey, *, has_separate_hem: bool) -> bool:
    suffixes = _suffixes_for_source(source_key, has_separate_hem=has_separate_hem)
    return any(feature.endswith(suffix) for suffix in suffixes)


def _normalize_df_columns(df_columns) -> set[str]:
    if df_columns is None:
        return set()
    return set(df_columns)


def get_derivable_transform_columns(
    search_keys: dict[str, SearchKey],
    source_search_keys: dict[str, SearchKey] | None = None,
    columns_renaming: dict[str, str] | None = None,
    fit_columns_renaming: dict[str, str] | None = None,
    df_columns=None,
) -> set[str]:
    """Column names that will be created later in transform from present search keys."""
    columns_renaming = columns_renaming or {}
    normalized_df_columns = _normalize_df_columns(df_columns)
    has_separate_hem = _has_separate_hem_column(search_keys)
    derivable: set[str] = set()

    for source_key in (SearchKey.IP, SearchKey.EMAIL):
        source_columns = _get_source_columns_by_type(
            source_key,
            search_keys,
            source_search_keys,
            columns_renaming,
            fit_columns_renaming,
            normalized_df_columns,
        )
        for column in source_columns:
            for suffix in _suffixes_for_source(source_key, has_separate_hem=has_separate_hem):
                derivable.add(f"{column}{suffix}")

    return derivable


def is_derivable_transform_column(
    feature: str,
    search_keys: dict[str, SearchKey],
    columns_renaming: dict[str, str] | None = None,
    source_search_keys: dict[str, SearchKey] | None = None,
    fit_columns_renaming: dict[str, str] | None = None,
    df_columns=None,
) -> bool:
    columns_renaming = columns_renaming or {}
    normalized_df_columns = _normalize_df_columns(df_columns)

    if feature in get_derivable_transform_columns(
        search_keys,
        source_search_keys,
        columns_renaming,
        fit_columns_renaming,
        normalized_df_columns,
    ):
        return True

    has_separate_hem = _has_separate_hem_column(search_keys)
    feature_original = _column_original_name(feature, fit_columns_renaming, columns_renaming)

    for source_key in (SearchKey.IP, SearchKey.EMAIL):
        if not _feature_has_derivable_suffix(feature, source_key, has_separate_hem=has_separate_hem):
            continue

        source_columns = _get_source_columns_by_type(
            source_key,
            search_keys,
            source_search_keys,
            columns_renaming,
            fit_columns_renaming,
            normalized_df_columns,
        )
        for column in source_columns:
            if _column_original_name(column, fit_columns_renaming, columns_renaming) == feature_original:
                return True

        for suffix, suffix_source_key in DERIVED_COLUMN_SUFFIXES.items():
            if suffix_source_key != source_key or not feature.endswith(suffix):
                continue
            base_column = feature[: -len(suffix)]
            if base_column in source_columns:
                return True
            if search_keys.get(base_column) == SearchKey.CUSTOM_KEY and base_column in source_columns:
                return True

        for suffix, suffix_source_key in PRE_TRANSFORM_DERIVED_COLUMN_SUFFIXES.items():
            if suffix_source_key != source_key or not feature.endswith(suffix):
                continue
            base_column = feature[: -len(suffix)]
            if base_column in source_columns:
                return True

    return False


def get_missing_features_for_transform(
    features_for_transform: list[str],
    df_columns,
    search_keys: dict[str, SearchKey],
    columns_renaming: dict[str, str],
    source_search_keys: dict[str, SearchKey] | None = None,
    fit_columns_renaming: dict[str, str] | None = None,
) -> list[str]:
    return [
        columns_renaming.get(feature) or feature
        for feature in features_for_transform
        if feature not in df_columns
        and not is_derivable_transform_column(
            feature,
            search_keys,
            columns_renaming,
            source_search_keys,
            fit_columns_renaming,
            df_columns,
        )
    ]

from typing import Optional

import pandas as pd
from pandas.api.types import is_float_dtype, is_int64_dtype, is_string_dtype


def phone_to_int(df: pd.DataFrame, phone_column_name: str) -> pd.DataFrame:
    """
    Convention: phone number is always presented as int number.
    phone_number = Country code + National Destination Code + Subscriber Number.
    Examples:
    41793834315     for Switzerland
    46767040672     for Sweden
    861065529988    for China
    18143008198     for the USA
    Inplace conversion of phone to int.

    Method will remove all non numeric chars from string and convert it to int.
    None will be set for phone numbers that couldn't be converted to int
    """
    if is_string_dtype(df[phone_column_name]):
        df[phone_column_name] = df[phone_column_name].apply(phone_str_to_int_safe)
    elif is_float_dtype(df[phone_column_name]):
        df[phone_column_name] = df[phone_column_name].apply(phone_float_to_int_safe)
    elif is_int64_dtype(df[phone_column_name]):
        df[phone_column_name] = df[phone_column_name].apply(phone_int_to_int_safe)
    else:
        raise Exception(
            f"phone_column_name {phone_column_name} doesn't have supported dtype. Dataset dtypes: {df.dtypes}. "
            f"Contact developer and request to implement conversion of {phone_column_name} to int"
        )
    return df


def phone_float_to_int_safe(value: float) -> Optional[int]:
    try:
        return validate_length(int(value))
    except Exception:
        return None


def phone_int_to_int_safe(value: float) -> Optional[int]:
    try:
        return validate_length(int(value))
    except Exception:
        return None


def phone_str_to_int_safe(value: str) -> Optional[int]:
    try:
        numeric_filter = filter(str.isdigit, value)
        numeric_string = "".join(numeric_filter)
        return validate_length(int(numeric_string))
    except Exception:
        return None


def validate_length(value: int) -> Optional[int]:
    if value < 10000000 or value > 999999999999999:
        return None
    else:
        return value

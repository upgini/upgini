import itertools
from typing import List, Tuple

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype


def combine_search_keys(search_keys: List[str]) -> List[Tuple[str]]:
    combined_search_keys = []
    for L in range(1, len(search_keys) + 1):
        for subset in itertools.combinations(search_keys, L):
            combined_search_keys.append(subset)
    return combined_search_keys


def find_numbers_with_decimal_comma(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.head(10)
    # all columns with sep="," will have dtype == 'object', i.e string
    # sep="." will be casted to numeric automatically
    return [
        col
        for col in tmp.columns
        if (is_string_dtype(tmp[col]) or is_object_dtype(tmp[col]))
        and tmp[col].astype("string").str.match("^[0-9]+,[0-9]*$").any()
    ]

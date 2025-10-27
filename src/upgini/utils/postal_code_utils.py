import pandas as pd
from pandas.api.types import (
    is_float_dtype,
    is_object_dtype,
    is_string_dtype,
)
import re

from upgini.utils.base_search_key_detector import BaseSearchKeyDetector


class PostalCodeSearchKeyDetector(BaseSearchKeyDetector):
    postal_pattern = re.compile(r'^[A-Za-z0-9][A-Za-z0-9\s\-]{1,9}$')

    def _is_search_key_by_name(self, column_name: str) -> bool:
        return "zip" in str(column_name).lower() or "postal" in str(column_name).lower()

    def _is_search_key_by_values(self, column: pd.Series) -> bool:
        """
        # Fast two-step check whether the column looks like a postal code.
        # Returns True if, after removing missing values, values remain,
        # and all of them match the common characteristics of a postal code.
        """
        # Check only columns that are candidates for postal code by column name
        if not self._is_search_key_by_name(column.name):
            return False

        s = column.copy().dropna().astype(str).str.strip()
        s = s[s != ""]  # remove empty strings
        if s.empty:
            return False

        # remove suffix ".0" (often after float)
        s = s.str.replace(r"\.0$", "", regex=True)

        # --- Step 1: fast filtering ---
        mask_len = s.str.len().between(2, 10)
        mask_digit = s.str.contains(r'\d', regex=True)
        mask_chars = ~s.str.contains(r'[^A-Za-z0-9\s\-]', regex=True)
        fast_mask = mask_len & mask_digit & mask_chars

        # if any of them failed the fast check, return False
        if not fast_mask.all():
            return False

        # --- Step 2: regex check ---
        # only if the first step passed
        valid_mask = s.apply(lambda x: bool(self.postal_pattern.fullmatch(x)))
        return valid_mask.all()


class PostalCodeSearchKeyConverter:

    def __init__(self, postal_code_column: str):
        self.postal_code_column = postal_code_column

    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        if is_string_dtype(df[self.postal_code_column]) or is_object_dtype(df[self.postal_code_column]):
            try:
                df[self.postal_code_column] = (
                    df[self.postal_code_column].astype("string").astype("float64").astype("Int64").astype("string")
                )
            except Exception:
                pass
        elif is_float_dtype(df[self.postal_code_column]):
            df[self.postal_code_column] = df[self.postal_code_column].astype("Int64").astype("string")

        df[self.postal_code_column] = (
            df[self.postal_code_column]
            .astype("string")
            .str.upper()
            .str.replace(r"[^0-9A-Z]", "", regex=True)  # remove non alphanumeric characters
            .str.replace(r"^0+\B", "", regex=True)  # remove leading zeros
        )
        # if (df[self.postal_code_column] == "").all():
        #     raise ValidationError(self.bundle.get("invalid_postal_code").format(self.postal_code_column))

        return df

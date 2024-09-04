import pandas as pd
from pandas.api.types import (
    is_float_dtype,
    is_object_dtype,
    is_string_dtype,
)

from upgini.utils.base_search_key_detector import BaseSearchKeyDetector


class PostalCodeSearchKeyDetector(BaseSearchKeyDetector):
    def _is_search_key_by_name(self, column_name: str) -> bool:
        return str(column_name).lower() in ["zip", "zipcode", "zip_code", "postal_code", "postalcode"]

    def _is_search_key_by_values(self, column: pd.Series) -> bool:
        return False


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

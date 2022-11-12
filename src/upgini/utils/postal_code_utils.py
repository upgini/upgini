import pandas as pd

from upgini.utils.base_search_key_detector import BaseSearchKeyDetector


class PostalCodeSearchKeyDetector(BaseSearchKeyDetector):
    def _is_search_key_by_name(self, column_name: str) -> bool:
        return str(column_name).lower() in ["zip", "zipcode", "zip_code", "postal_code", "postalcode"]

    def _is_search_key_by_values(self, column: pd.Series) -> bool:
        return False

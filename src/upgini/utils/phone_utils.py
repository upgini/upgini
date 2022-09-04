import pandas as pd

from upgini.utils.base_search_key_detector import BaseSearchKeyDetector


class PhoneSearchKeyDetector(BaseSearchKeyDetector):
    def _is_search_key_by_name(self, column_name: str) -> bool:
        return column_name.lower() in ["cellphone", "msisdn", "phone", "phonenumber", "phone_number"]

    def _is_search_key_by_values(self, column: pd.Series) -> bool:
        return False

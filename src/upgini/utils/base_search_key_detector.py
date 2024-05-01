from typing import List

import pandas as pd


class BaseSearchKeyDetector:
    def _is_search_key_by_name(self, column_name: str) -> bool:
        raise NotImplementedError

    def _is_search_key_by_values(self, column: pd.Series) -> bool:
        raise NotImplementedError

    def _get_search_keys_by_name(self, column_names: List[str]) -> List[str]:
        return [
            column_name
            for column_name in column_names
            if self._is_search_key_by_name(column_name)
        ]

    def get_search_key_columns(self, df: pd.DataFrame, existing_search_keys: List[str]) -> List[str]:
        other_columns = [col for col in df.columns if col not in existing_search_keys]
        columns_by_names = self._get_search_keys_by_name(other_columns)
        columns_by_values = []
        for column_name in other_columns:
            if self._is_search_key_by_values(df[column_name]):
                columns_by_values.append(column_name)
        return list(set(columns_by_names + columns_by_values))

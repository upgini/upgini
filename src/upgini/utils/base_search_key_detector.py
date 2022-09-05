from typing import List, Optional

import pandas as pd


class BaseSearchKeyDetector:
    def _is_search_key_by_name(self, column_name: str) -> bool:
        raise NotImplementedError()

    def _is_search_key_by_values(self, column: pd.Series) -> bool:
        raise NotImplementedError()

    def _get_search_key_by_name(self, column_names: List[str]) -> Optional[str]:
        for column_name in column_names:
            if self._is_search_key_by_name(column_name):
                return column_name

    def get_search_key_column(self, df: pd.DataFrame) -> Optional[str]:
        maybe_column = self._get_search_key_by_name(df.columns.to_list())
        if maybe_column is not None:
            return maybe_column

        for column_name in df.columns:
            if self._is_search_key_by_values(df[column_name]):
                return column_name

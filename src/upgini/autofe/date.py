import abc
from typing import Any, List, Optional, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel

from upgini.autofe.operand import PandasOperand


class DateDiffMixin(BaseModel):
    diff_unit: str = "D"
    left_unit: Optional[str] = None
    right_unit: Optional[str] = None

    def _convert_to_date(
        self, x: Union[pd.DataFrame, pd.Series], unit: Optional[str]
    ) -> Union[pd.DataFrame, pd.Series]:
        if isinstance(x, pd.DataFrame):
            return x.apply(lambda y: self._convert_to_date(y, unit), axis=1)

        return pd.to_datetime(x, unit=unit)


class DateDiff(PandasOperand, DateDiffMixin):
    name = "date_diff"
    is_binary = True
    has_symmetry_importance = True

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        left = self._convert_to_date(left, self.left_unit)
        right = self._convert_to_date(right, self.right_unit)
        return self.__replace_negative((left - right) / np.timedelta64(1, self.diff_unit))

    def __replace_negative(self, x: Union[pd.DataFrame, pd.Series]):
        x[x < 0] = None
        return x


class DateDiffType2(PandasOperand, DateDiffMixin):
    name = "date_diff_type2"
    is_binary = True
    has_symmetry_importance = True

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        left = self._convert_to_date(left, self.left_unit)
        right = self._convert_to_date(right, self.right_unit)
        future = right + (left.dt.year - right.dt.year).apply(
            lambda y: np.datetime64("NaT") if np.isnan(y) else pd.tseries.offsets.DateOffset(years=y)
        )
        before = future[future < left]
        future[future < left] = before + pd.tseries.offsets.DateOffset(years=1)
        diff = (future - left) / np.timedelta64(1, self.diff_unit)

        return diff


class DateListDiff(PandasOperand, DateDiffMixin):
    is_binary = True
    has_symmetry_importance = True
    aggregation: str

    def __init__(self, **data: Any) -> None:
        if "aggregation" in data and "name" not in data:
            data["name"] = f"date_diff_{data['aggregation']}"
        super().__init__(**data)

    def map_diff(self, left: np.datetime64, right: list) -> list:
        return (left - self._convert_to_date(pd.Series(right), self.right_unit)) / np.timedelta64(1, self.diff_unit)

    def reduce(self, date_list: pd.Series) -> float:
        return date_list[date_list > 0].aggregate(self.aggregation)

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        left = self._convert_to_date(left, self.left_unit)

        return pd.Series(left.index.map(lambda i: self.reduce(self.map_diff(left.loc[i], right.loc[i]))))

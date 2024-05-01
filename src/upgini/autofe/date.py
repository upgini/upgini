from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from pandas.core.arrays.timedeltas import TimedeltaArray
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
        future = pd.to_datetime(future)
        before = future[future < left]
        future[future < left] = before + pd.tseries.offsets.DateOffset(years=1)
        diff = (future - left) / np.timedelta64(1, self.diff_unit)

        return diff


_ext_aggregations = {"nunique": (lambda x: len(np.unique(x)), 0), "count": (len, 0)}


class DateListDiff(PandasOperand, DateDiffMixin):
    is_binary = True
    has_symmetry_importance = True
    aggregation: str

    def __init__(self, **data: Any) -> None:
        if "name" not in data:
            data["name"] = f"date_diff_{data.get('aggregation')}"
        super().__init__(**data)

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        left = self._convert_to_date(left, self.left_unit)
        right = right.apply(lambda x: pd.arrays.DatetimeArray(self._convert_to_date(x, self.right_unit)))

        return pd.Series(left - right.values).apply(lambda x: self._agg(self._diff(x)))

    def _diff(self, x: TimedeltaArray):
        if self.diff_unit == "Y":
            x = (x / 365 / 24 / 60 / 60 / 10**9).astype(int)
        elif self.diff_unit == "M":
            raise Exception("Unsupported difference unit: Month")
        else:
            x = x / np.timedelta64(1, self.diff_unit)
        return x[x > 0]

    def _agg(self, x):
        method = getattr(np, self.aggregation, None)
        default = np.nan
        if method is None and self.aggregation in _ext_aggregations:
            method, default = _ext_aggregations[self.aggregation]
        elif not callable(method):
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")

        return method(x) if len(x) > 0 else default


class DateListDiffBounded(DateListDiff):
    lower_bound: Optional[int]
    upper_bound: Optional[int]

    def __init__(self, **data: Any) -> None:
        if "name" not in data:
            lower_bound = data.get("lower_bound")
            upper_bound = data.get("upper_bound")
            components = [
                "date_diff",
                data.get("diff_unit"),
                str(lower_bound if lower_bound is not None else "minusinf"),
                str(upper_bound if upper_bound is not None else "plusinf"),
            ]
            components.append(data.get("aggregation"))
            data["name"] = "_".join(components)
        super().__init__(**data)

    def _agg(self, x):
        x = x[(x >= (self.lower_bound or -np.inf)) & (x < (self.upper_bound or np.inf))]
        return super()._agg(x)

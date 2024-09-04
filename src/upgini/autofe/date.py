import abc
import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas.core.arrays.timedeltas import TimedeltaArray
from pydantic import BaseModel, __version__ as pydantic_version

from upgini.autofe.operand import PandasOperand


def get_pydantic_version():
    major_version = int(pydantic_version.split('.')[0])
    return major_version


class DateDiffMixin(BaseModel):
    diff_unit: str = "D"
    left_unit: Optional[str] = None
    right_unit: Optional[str] = None

    def _convert_to_date(
        self, x: Union[pd.DataFrame, pd.Series], unit: Optional[str]
    ) -> Union[pd.DataFrame, pd.Series]:
        if isinstance(x, pd.DataFrame):
            return x.apply(lambda y: self._convert_to_date(y, unit), axis=1)

        return pd.to_datetime(x, unit=unit, errors="coerce")

    def _convert_diff_to_unit(self, diff: Union[pd.Series, TimedeltaArray]) -> Union[pd.Series, TimedeltaArray]:
        if self.diff_unit == "D":
            if isinstance(diff, pd.Series) and diff.dtype == "object":
                return diff.apply(lambda x: None if isinstance(x, float) and np.isnan(x) else x.days)
            else:
                return diff / np.timedelta64(1, self.diff_unit)
        elif self.diff_unit == "Y":
            if isinstance(diff, TimedeltaArray):
                return (diff / 365 / 24 / 60 / 60 / 10**9).astype(int)
            else:
                return (diff / 365 / 24 / 60 / 60 / 10**9).dt.nanoseconds
        else:
            raise Exception(f"Unsupported difference unit: {self.diff_unit}")


class DateDiff(PandasOperand, DateDiffMixin):
    name: str = "date_diff"
    alias: Optional[str] = "date_diff_type1"
    is_binary: bool = True
    has_symmetry_importance: bool = True

    replace_negative: bool = False

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "diff_unit": self.diff_unit,
                "left_unit": self.left_unit,
                "right_unit": self.right_unit,
                "replace_negative": self.replace_negative,
            }
        )
        return res

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        left = self._convert_to_date(left, self.left_unit)
        right = self._convert_to_date(right, self.right_unit)
        diff = self._convert_diff_to_unit(left.dt.date - right.dt.date)
        return self.__replace_negative(diff)

    def __replace_negative(self, x: Union[pd.DataFrame, pd.Series]):
        if self.replace_negative:
            x[x < 0] = None
        return x


class DateDiffType2(PandasOperand, DateDiffMixin):
    name: str = "date_diff_type2"
    is_binary: bool = True
    has_symmetry_importance: bool = True

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "diff_unit": self.diff_unit,
                "left_unit": self.left_unit,
                "right_unit": self.right_unit,
            }
        )
        return res

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        left = self._convert_to_date(left, self.left_unit)
        right = self._convert_to_date(right, self.right_unit)
        future = right + (left.dt.year - right.dt.year).apply(
            lambda y: pd.tseries.offsets.DateOffset(years=0 if np.isnan(y) else y)
        )
        future = pd.to_datetime(future)
        before = future[future < left]
        future[future < left] = before + pd.tseries.offsets.DateOffset(years=1)
        diff = (future - left) / np.timedelta64(1, self.diff_unit)

        return diff


_ext_aggregations = {"nunique": (lambda x: len(np.unique(x)), 0), "count": (len, 0)}
_count_aggregations = ["nunique", "count"]


class DateListDiff(PandasOperand, DateDiffMixin):
    is_binary: bool = True
    has_symmetry_importance: bool = True

    aggregation: str
    replace_negative: bool = False

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "aggregation": self.aggregation,
                "diff_unit": self.diff_unit,
                "left_unit": self.left_unit,
                "right_unit": self.right_unit,
                "replace_negative": self.replace_negative,
            }
        )
        return res

    def __init__(self, **data: Any) -> None:
        if "name" not in data:
            data["name"] = f"date_diff_{data.get('aggregation')}"
        super().__init__(**data)

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        left = self._convert_to_date(left, self.left_unit)
        right_mask = right.apply(lambda x: len(x) > 0)
        mask = left.notna() & right.notna() & right_mask
        right_masked = right[mask].apply(lambda x: pd.arrays.DatetimeArray(self._convert_to_date(x, self.right_unit)))

        if len(right_masked) == 0:
            diff = []
        elif len(right_masked) < 2:
            diff = [left[mask].iloc[0] - right_masked.iloc[0]]
        else:
            diff = left[mask] - right_masked.values

        res_masked = pd.Series(diff, index=left[mask].index).apply(lambda x: self._agg(self._diff(x)))
        res = res_masked.reindex(left.index.union(right.index))
        if self.aggregation in _count_aggregations:
            res[~right_mask] = 0.0
        res = res.astype(np.float64)

        return res

    def _diff(self, x: TimedeltaArray):
        x = self._convert_diff_to_unit(x)
        return x[x > 0] if self.replace_negative else x

    def _agg(self, x):
        method = getattr(np, self.aggregation, None)
        default = np.nan
        if method is None and self.aggregation in _ext_aggregations:
            method, default = _ext_aggregations[self.aggregation]
        elif not callable(method):
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")

        return method(x) if len(x) > 0 else default


class DateListDiffBounded(DateListDiff):
    lower_bound: Optional[int] = None
    upper_bound: Optional[int] = None

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
        x = x[
            (x >= (self.lower_bound if self.lower_bound is not None else -np.inf))
            & (x < (self.upper_bound if self.upper_bound is not None else np.inf))
        ]
        return super()._agg(x)


class DatePercentileBase(PandasOperand, abc.ABC):
    is_binary: bool = True
    output_type: Optional[str] = "float"

    date_unit: Optional[str] = None

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        # Assuming that left is a date column, right is a feature column
        left = pd.to_datetime(left, unit=self.date_unit)

        bounds = self._get_bounds(left)

        return right.index.to_series().apply(lambda i: self._perc(right[i], bounds[i]))

    @abc.abstractmethod
    def _get_bounds(self, date_col: pd.Series) -> pd.Series:
        pass

    def _perc(self, f, bounds):
        hit = np.where(f >= np.array(bounds))[0]
        if hit.size > 0:
            return np.max(hit) + 1
        else:
            return np.nan

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "date_unit": self.date_unit,
            }
        )
        return res


class DatePercentile(DatePercentileBase):
    name: str = "date_per"
    alias: Optional[str] = "date_per_method1"

    zero_month: Optional[int] = None
    zero_year: Optional[int] = None
    zero_bounds: Optional[List[float]] = None
    step: int = 30

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "zero_month": self.zero_month,
                "zero_year": self.zero_year,
                "zero_bounds": self.zero_bounds,
                "step": self.step,
            }
        )
        return res

    # Check Pydantic version
    if get_pydantic_version() >= 2:
        # Use @field_validator for Pydantic 2.x
        from pydantic import field_validator

        @field_validator('zero_bounds', mode='before')
        def parse_zero_bounds(cls, value):
            if isinstance(value, str):
                return json.loads(value)
            return value
    else:
        # Use @validator for Pydantic 1.x
        from pydantic import validator

        @validator('zero_bounds', pre=True)
        def parse_zero_bounds(cls, value):
            if isinstance(value, str):
                return json.loads(value)
            return value

    def _get_bounds(self, date_col: pd.Series) -> pd.Series:
        months = date_col.dt.month
        years = date_col.dt.year

        month_diffs = 12 * (years - (self.zero_year or 0)) + (months - (self.zero_month or 0))
        return month_diffs.apply(
            lambda d: np.array(self.zero_bounds if self.zero_bounds is not None else []) + d * self.step
        )


class DatePercentileMethod2(DatePercentileBase):
    name: str = "date_per_method2"

    def _get_bounds(self, date_col: pd.Series) -> pd.Series:
        pass

import abc
import json
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas.core.arrays.timedeltas import TimedeltaArray
from pydantic import BaseModel, __version__ as pydantic_version

from upgini.autofe.operator import PandasOperator, ParametrizedOperator
from upgini.autofe.utils import bin_index, bin_index_many, bin_index_vectorized, pydantic_validator


def get_pydantic_version():
    major_version = int(pydantic_version.split(".")[0])
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


class DateDiff(PandasOperator, DateDiffMixin):
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
        if left.isna().all() or right.isna().all():
            return pd.Series([None] * len(left), index=left.index, dtype=np.float64)

        left = self._convert_to_date(left, self.left_unit)
        right = self._convert_to_date(right, self.right_unit)
        diff = self._convert_diff_to_unit(left.dt.date - right.dt.date)
        return self.__replace_negative(diff)

    def __replace_negative(self, x: Union[pd.DataFrame, pd.Series]):
        if self.replace_negative:
            x[x < 0] = None
        return x


class DateDiffType2(PandasOperator, DateDiffMixin):
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


class DateListDiff(PandasOperator, DateDiffMixin, ParametrizedOperator):
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

    def to_formula(self) -> str:
        return f"date_diff_{self.aggregation}"

    @classmethod
    def from_formula(cls, formula: str) -> Optional["DateListDiff"]:
        if not formula.startswith("date_diff_") or formula.startswith("date_diff_type"):
            return None
        aggregation = formula.replace("date_diff_", "")
        if "_" in aggregation:
            return None
        return cls(aggregation=aggregation)

    @staticmethod
    def _non_empty_list_mask(right: pd.Series) -> pd.Series:
        values = right.to_numpy()
        mask = np.empty(len(values), dtype=bool)
        for i, value in enumerate(values):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                mask[i] = False
            elif isinstance(value, (list, tuple, np.ndarray)):
                mask[i] = len(value) > 0
            else:
                mask[i] = False
        return pd.Series(mask, index=right.index)

    def _convert_date_lists(self, lists: pd.Series) -> pd.Series:
        exploded = lists.explode()
        converted = pd.to_datetime(exploded, unit=self.right_unit, errors="coerce")
        return pd.Series(
            {
                idx: pd.arrays.DatetimeArray(values.to_numpy())
                for idx, values in converted.groupby(converted.index, sort=False)
            }
        )

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        if left.isna().all() or right.isna().all():
            return pd.Series([None] * len(left), index=left.index, dtype=np.float64)

        left = self._convert_to_date(left, self.left_unit)
        right_mask = self._non_empty_list_mask(right)
        mask = left.notna() & right.notna() & right_mask

        if not mask.any():
            res = pd.Series(np.nan, index=left.index, dtype=np.float64)
            if self.aggregation in _count_aggregations:
                res[~right_mask] = 0.0
            return res

        masked_left = left[mask]
        converted_lists = self._convert_date_lists(right[mask])
        results = np.empty(len(masked_left), dtype=np.float64)
        results[:] = np.nan
        for i, (idx, left_date) in enumerate(masked_left.items()):
            results[i] = self._agg(self._diff(left_date - converted_lists[idx]))

        res_masked = pd.Series(results, index=masked_left.index)
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


class DateListDiffBounded(DateListDiff, ParametrizedOperator):
    lower_bound: Optional[int] = None
    upper_bound: Optional[int] = None
    normalize: Optional[bool] = None

    def to_formula(self) -> str:
        lower_bound = "minusinf" if self.lower_bound is None else self.lower_bound
        upper_bound = "plusinf" if self.upper_bound is None else self.upper_bound
        norm = "_norm" if self.normalize else ""
        return f"date_diff_{self.diff_unit}_{lower_bound}_{upper_bound}_{self.aggregation}{norm}"

    @classmethod
    def from_formula(cls, formula: str) -> Optional["DateListDiffBounded"]:
        import re

        normalize = formula.endswith("_norm")
        formula = formula.replace("_norm", "")

        pattern = r"^date_diff_([^_]+)_((minusinf|\d+))_((plusinf|\d+))_(\w+)$"
        match = re.match(pattern, formula)

        if not match:
            return None

        diff_unit = match.group(1)
        lower_bound = None if match.group(2) == "minusinf" else int(match.group(2))
        upper_bound = None if match.group(4) == "plusinf" else int(match.group(4))
        aggregation = match.group(6)
        return cls(
            diff_unit=diff_unit,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            aggregation=aggregation,
            normalize=normalize,
        )

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        if self.lower_bound is not None:
            res["lower_bound"] = str(self.lower_bound)
        if self.upper_bound is not None:
            res["upper_bound"] = str(self.upper_bound)
        if self.normalize is not None:
            res["normalize"] = str(self.normalize)
        return res

    def _agg(self, x):
        orig_len = len(x)
        x = x[
            (x >= (self.lower_bound if self.lower_bound is not None else -np.inf))
            & (x < (self.upper_bound if self.upper_bound is not None else np.inf))
        ]
        agg_res = super()._agg(x)
        if self.normalize and orig_len > 0:
            return agg_res / orig_len
        return agg_res


class DatePercentileBase(PandasOperator, abc.ABC):
    is_binary: bool = True
    is_categorical: bool = True
    output_type: Optional[str] = "category"

    date_unit: Optional[str] = None

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        # Assuming that left is a date column, right is a feature column
        left = pd.to_datetime(left, unit=self.date_unit)

        bounds = self._get_bounds(left)
        values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        bounds_list = bounds.tolist()
        result = np.full(len(values), np.nan)

        if not bounds_list:
            return pd.Series(result, index=right.index).astype(pd.Int64Dtype()).astype("category")

        bounds_lengths = {len(b) for b in bounds_list if isinstance(b, (list, np.ndarray))}
        if len(bounds_lengths) == 1 and all(isinstance(b, (list, np.ndarray)) for b in bounds_list):
            bounds_2d = np.asarray(bounds_list, dtype=np.float64)
            if bounds_2d.ndim == 1:
                result = bin_index_vectorized(values, bounds_2d)
            else:
                result = bin_index_many(values, bounds_2d)
        else:
            for i, row_bounds in enumerate(bounds_list):
                if isinstance(row_bounds, (list, np.ndarray)) and len(row_bounds) > 0:
                    result[i] = bin_index(values[i], row_bounds)

        return pd.Series(result, index=right.index).astype(pd.Int64Dtype()).astype("category")

    @abc.abstractmethod
    def _get_bounds(self, date_col: pd.Series) -> pd.Series:
        pass

    def _perc(self, f, bounds):
        return bin_index(f, bounds)

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
                "zero_bounds": json.dumps(self.zero_bounds),
                "step": self.step,
            }
        )
        return res

    @pydantic_validator("zero_bounds", mode="before")
    def parse_zero_bounds(cls, value):
        if isinstance(value, str):
            return json.loads(value)
        return value

    def _get_bounds(self, date_col: pd.Series) -> pd.Series:
        zero_bounds = self.zero_bounds if self.zero_bounds is not None else []
        if not zero_bounds:
            return pd.Series([[] for _ in range(len(date_col))], index=date_col.index)

        month_diffs = (
            12 * (date_col.dt.year - (self.zero_year or 0)) + (date_col.dt.month - (self.zero_month or 0))
        ).to_numpy()
        bounds_2d = np.asarray(zero_bounds, dtype=np.float64) + month_diffs[:, None] * self.step
        return pd.Series(list(bounds_2d), index=date_col.index)


class DatePercentileMethod2(DatePercentileBase):
    name: str = "date_per_method2"

    def _get_bounds(self, date_col: pd.Series) -> pd.Series:
        pass

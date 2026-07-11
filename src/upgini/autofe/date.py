import abc
import json
from typing import Dict, List, NamedTuple, Optional, Union

import numpy as np
import pandas as pd
from pandas.core.arrays.timedeltas import TimedeltaArray
from pydantic import BaseModel, __version__ as pydantic_version

from upgini.autofe.operand import OperandKind, OperandValue
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

    def calculate_binary(self, left: OperandValue, right: OperandValue) -> pd.Series:
        left = left.as_series()
        right = right.as_series()
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

    def calculate_binary(self, left: OperandValue, right: OperandValue) -> pd.Series:
        left = left.as_series()
        right = right.as_series()
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


_ext_aggregations = {"nunique": (lambda x: float(np.unique(x).size), 0), "count": (len, 0)}
_count_aggregations = ["nunique", "count"]
_DATE_DIFF_LISTS_LENGTH_COL = 0
_NS_PER_DAY = np.float64(86400 * 10**9)
_NS_PER_YEAR = np.float64(365 * 86400 * 10**9)
_MATRIX_AGGREGATIONS = {
    "nunique": "_matrix_agg_nunique",
    "count": "_matrix_agg_count",
    "sum": "_matrix_agg_sum",
    "mean": "_matrix_agg_mean",
    "min": "_matrix_agg_min",
    "max": "_matrix_agg_max",
}


class _MatrixAggContext(NamedTuple):
    lengths: np.ndarray
    masked_values: np.ndarray
    valid_mask: np.ndarray
    missing: np.ndarray
    empty: np.ndarray
    agg_source: np.ndarray
    count_source: np.ndarray
    has_bounds: bool
    results: np.ndarray


def _timedelta_ns_to_diff_unit(delta_ns: np.ndarray, diff_unit: str) -> np.ndarray:
    if diff_unit == "D":
        return delta_ns / _NS_PER_DAY
    if diff_unit == "Y":
        return (delta_ns / _NS_PER_YEAR).astype(np.int64).astype(np.float64)
    raise ValueError(f"Unsupported difference unit: {diff_unit}")


def _group_cumcount(group_keys: np.ndarray) -> np.ndarray:
    n = len(group_keys)
    if n == 0:
        return np.zeros(0, dtype=np.intp)
    order = np.argsort(group_keys, kind="stable")
    sorted_keys = group_keys[order]
    group_change = np.empty(n, dtype=bool)
    group_change[0] = True
    if n > 1:
        group_change[1:] = sorted_keys[1:] != sorted_keys[:-1]
    group_ids = np.cumsum(group_change) - 1
    group_start_idx = np.flatnonzero(group_change)
    sorted_cumcount = np.arange(n, dtype=np.intp) - group_start_idx[group_ids]
    cumcount = np.empty(n, dtype=np.intp)
    cumcount[order] = sorted_cumcount
    return cumcount


def _aggregate_diffs(values: np.ndarray, aggregation: str) -> float:
    values = np.atleast_1d(np.asarray(values, dtype=np.float64))
    method = getattr(np, aggregation, None)
    default = np.nan
    if method is None and aggregation in _ext_aggregations:
        method, default = _ext_aggregations[aggregation]
    elif not callable(method):
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    return method(values) if len(values) > 0 else default


class DateListDiffLists(PandasOperator, DateDiffMixin, ParametrizedOperator):
    name: str = "date_diff_lists"
    is_binary: bool = True
    has_symmetry_importance: bool = True
    output_type: Optional[str] = "vector"

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

    def to_formula(self) -> str:
        return f"date_diff_lists_{self.diff_unit}"

    @classmethod
    def from_formula(cls, formula: str) -> Optional["DateListDiffLists"]:
        if formula == "date_diff_lists":
            return cls()
        if formula.startswith("date_diff_lists_"):
            diff_unit = formula.replace("date_diff_lists_", "")
            if diff_unit in {"D", "Y"}:
                return cls(diff_unit=diff_unit)
        return None

    def _non_empty_list_mask(self, right: pd.Series) -> pd.Series:
        return right.map(lambda value: isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0).fillna(False)

    def _build_matrix(self, left: pd.Series, right: pd.Series) -> np.ndarray:
        n = len(left)
        if n == 0:
            return np.empty((0, 1), dtype=np.float64)

        left_dates = pd.to_datetime(left, unit=self.left_unit, errors="coerce")
        date_unit = self.right_unit if self.right_unit is not None else self.left_unit
        right_mask = self._non_empty_list_mask(right).to_numpy()
        right_notna = right.notna().to_numpy()
        left_notna = left_dates.notna().to_numpy()

        compute_mask = left_notna & right_notna & right_mask
        empty_right = right_notna & ~right_mask

        lengths = np.full(n, np.nan, dtype=np.float64)
        lengths[empty_right] = 0.0

        compute_idx = np.flatnonzero(compute_mask)
        if len(compute_idx) == 0:
            return lengths.reshape(n, 1)

        exploded = right.iloc[compute_idx].explode()
        row_indices_arr = right.index.get_indexer(exploded.index).astype(np.intp)
        raw_dates = exploded.to_numpy()
        pos_in_row_arr = _group_cumcount(row_indices_arr)
        converted = pd.to_datetime(pd.Series(raw_dates), unit=date_unit, errors="coerce")
        left_ns = left_dates.iloc[row_indices_arr].astype(np.int64).to_numpy()
        right_ns = converted.astype(np.int64).to_numpy()
        diffs = np.full(len(row_indices_arr), np.nan, dtype=np.float64)
        valid_ts = converted.notna().to_numpy()
        if valid_ts.any():
            diffs[valid_ts] = _timedelta_ns_to_diff_unit(left_ns[valid_ts] - right_ns[valid_ts], self.diff_unit)

        if self.replace_negative:
            keep = diffs > 0
            row_indices_arr = row_indices_arr[keep]
            diffs = diffs[keep]
            lengths[compute_mask] = 0.0
            if len(row_indices_arr):
                pos_in_row_arr = _group_cumcount(row_indices_arr)
                row_lengths = np.bincount(row_indices_arr, minlength=n).astype(np.float64)
                positive_rows = np.flatnonzero(row_lengths > 0)
                lengths[positive_rows] = row_lengths[positive_rows]
        else:
            row_lengths = np.bincount(row_indices_arr, minlength=n).astype(np.float64)
            lengths[compute_mask] = row_lengths[compute_mask]

        finite_lengths = lengths[np.isfinite(lengths)]
        k_max = int(finite_lengths.max()) if finite_lengths.size else 0
        matrix = np.full((n, 1 + k_max), np.nan, dtype=np.float64)
        matrix[:, _DATE_DIFF_LISTS_LENGTH_COL] = lengths
        if k_max > 0 and len(row_indices_arr):
            matrix[row_indices_arr, pos_in_row_arr + 1] = diffs
        return matrix

    def calculate_binary(self, left: OperandValue, right: OperandValue) -> np.ndarray:
        left = left.as_series()
        right = right.as_series()
        if left.isna().all() or right.isna().all():
            return np.full((len(left), 1), np.nan, dtype=np.float64)

        return self._build_matrix(left, right)


class DateListDiffAggWithinBounds(PandasOperator, ParametrizedOperator):
    name: str = "date_diff_list_agg"
    is_unary: bool = True
    output_type: Optional[str] = "float"

    lower_bound: Optional[int] = None
    upper_bound: Optional[int] = None
    aggregation: str
    normalize: Optional[bool] = None

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "aggregation": self.aggregation,
                "lower_bound": str(self.lower_bound) if self.lower_bound is not None else None,
                "upper_bound": str(self.upper_bound) if self.upper_bound is not None else None,
                "normalize": str(self.normalize) if self.normalize is not None else None,
            }
        )
        return res

    def to_formula(self) -> str:
        lower_bound = "minusinf" if self.lower_bound is None else self.lower_bound
        upper_bound = "plusinf" if self.upper_bound is None else self.upper_bound
        norm = "_norm" if self.normalize else ""
        return f"date_diff_list_agg_{lower_bound}_{upper_bound}_{self.aggregation}{norm}"

    @classmethod
    def from_formula(cls, formula: str) -> Optional["DateListDiffAggWithinBounds"]:
        import re

        normalize = formula.endswith("_norm")
        formula = formula.replace("_norm", "")

        pattern = r"^date_diff_list_agg_((minusinf|\d+))_((plusinf|\d+))_(\w+)$"
        match = re.match(pattern, formula)
        if not match:
            return None

        lower_bound = None if match.group(1) == "minusinf" else int(match.group(1))
        upper_bound = None if match.group(3) == "plusinf" else int(match.group(3))
        aggregation = match.group(5)
        return cls(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            aggregation=aggregation,
            normalize=normalize,
        )

    def _masked_values(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lengths = matrix[:, _DATE_DIFF_LISTS_LENGTH_COL]
        values = matrix[:, _DATE_DIFF_LISTS_LENGTH_COL + 1 :]
        missing = np.isnan(lengths)
        empty = (~missing) & (lengths == 0)
        if values.shape[1] == 0:
            valid_mask = np.zeros((len(lengths), 0), dtype=bool)
        else:
            valid_mask = np.arange(values.shape[1])[None, :] < lengths[:, None]
        masked_values = np.where(valid_mask, values, np.nan)
        return lengths, masked_values, valid_mask, missing, empty

    def _matrix_agg_nunique(self, ctx: _MatrixAggContext) -> None:
        ctx.results[ctx.empty] = 0.0
        active = ~ctx.missing & ~ctx.empty
        if ctx.has_bounds:
            select_mask = ctx.count_source & active[:, None]
        else:
            select_mask = ctx.valid_mask & active[:, None]
        rows, _ = np.nonzero(select_mask)
        ctx.results[active] = 0.0
        if rows.size:
            vals = ctx.masked_values[select_mask]
            counts = pd.Series(vals).groupby(rows, sort=False).nunique(dropna=False)
            ctx.results[counts.index.to_numpy(dtype=np.intp)] = counts.to_numpy(dtype=np.float64)

    def _matrix_agg_count(self, ctx: _MatrixAggContext) -> None:
        ctx.results[~ctx.missing] = ctx.count_source[~ctx.missing].sum(axis=1).astype(np.float64)

    def _matrix_agg_sum(self, ctx: _MatrixAggContext) -> None:
        agg_rows = ~ctx.missing & ~ctx.empty
        if agg_rows.any():
            with np.errstate(all="ignore"):
                ctx.results[agg_rows] = np.nansum(ctx.agg_source[agg_rows], axis=1)

    def _matrix_agg_nanaxis(self, ctx: _MatrixAggContext, reducer) -> None:
        agg_rows = ~ctx.missing & ~ctx.empty
        if not agg_rows.any():
            return
        has_finite = np.any(np.isfinite(ctx.agg_source[agg_rows]), axis=1)
        finite_rows = np.flatnonzero(agg_rows)[has_finite]
        with np.errstate(all="ignore"):
            ctx.results[finite_rows] = reducer(ctx.agg_source[finite_rows], axis=1)

    def _matrix_agg_mean(self, ctx: _MatrixAggContext) -> None:
        self._matrix_agg_nanaxis(ctx, np.nanmean)

    def _matrix_agg_min(self, ctx: _MatrixAggContext) -> None:
        self._matrix_agg_nanaxis(ctx, np.nanmin)

    def _matrix_agg_max(self, ctx: _MatrixAggContext) -> None:
        self._matrix_agg_nanaxis(ctx, np.nanmax)

    def _calculate_unary_matrix(self, matrix: np.ndarray, index: pd.Index) -> pd.Series:
        lengths, masked_values, valid_mask, missing, empty = self._masked_values(matrix)
        results = np.full(len(lengths), np.nan, dtype=np.float64)
        has_bounds = self.lower_bound is not None or self.upper_bound is not None

        if has_bounds:
            lower = self.lower_bound if self.lower_bound is not None else -np.inf
            upper = self.upper_bound if self.upper_bound is not None else np.inf
            in_bounds = (masked_values >= lower) & (masked_values < upper)
            agg_source = np.where(in_bounds & valid_mask, masked_values, np.nan)
            count_source = in_bounds & valid_mask
        else:
            agg_source = masked_values
            count_source = valid_mask

        ctx = _MatrixAggContext(
            lengths=lengths,
            masked_values=masked_values,
            valid_mask=valid_mask,
            missing=missing,
            empty=empty,
            agg_source=agg_source,
            count_source=count_source,
            has_bounds=has_bounds,
            results=results,
        )
        method_name = _MATRIX_AGGREGATIONS.get(self.aggregation)
        if method_name is None:
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")
        getattr(self, method_name)(ctx)

        if self.normalize:
            normalize_mask = ~missing & ~empty & (lengths > 0)
            results[normalize_mask] = results[normalize_mask] / lengths[normalize_mask]
        return pd.Series(results, index=index, dtype=np.float64)

    def _aggregate_row(self, diffs) -> float:
        if diffs is None or (isinstance(diffs, float) and np.isnan(diffs)):
            return np.nan

        diffs = np.atleast_1d(np.asarray(diffs, dtype=np.float64))
        orig_len = len(diffs)
        if self.lower_bound is not None or self.upper_bound is not None:
            lower = self.lower_bound if self.lower_bound is not None else -np.inf
            upper = self.upper_bound if self.upper_bound is not None else np.inf
            diffs = diffs[(diffs >= lower) & (diffs < upper)]
        agg_res = _aggregate_diffs(diffs, self.aggregation)
        if self.normalize and orig_len > 0:
            return agg_res / orig_len
        return agg_res

    def calculate_unary(self, data: OperandValue) -> pd.Series:
        if data.kind == OperandKind.MATRIX:
            return self._calculate_unary_matrix(data.as_matrix(), data.index)

        data = data.as_series()
        results = np.empty(len(data), dtype=np.float64)
        results[:] = np.nan
        for i, diffs in enumerate(data.to_numpy()):
            results[i] = self._aggregate_row(diffs)
        return pd.Series(results, index=data.index, dtype=np.float64)


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

    def _lists_op(self) -> DateListDiffLists:
        return DateListDiffLists(
            diff_unit=self.diff_unit,
            left_unit=self.left_unit,
            right_unit=self.right_unit,
            replace_negative=self.replace_negative,
        )

    def _agg_op(self) -> DateListDiffAggWithinBounds:
        return DateListDiffAggWithinBounds(
            lower_bound=None,
            upper_bound=None,
            aggregation=self.aggregation,
            normalize=False,
        )

    def _compose_list_diff(self, left: pd.Series, right: pd.Series) -> pd.Series:
        if left.isna().all() or right.isna().all():
            return pd.Series([None] * len(left), index=left.index, dtype=np.float64)

        right_mask = self._lists_op()._non_empty_list_mask(right)
        diff_lists = self._lists_op().calculate(left=left, right=right)
        result = self._agg_op().calculate(data=diff_lists).as_series()
        if self.aggregation in _count_aggregations:
            result[~right_mask] = 0.0
        return result.astype(np.float64)

    def calculate_binary(self, left: OperandValue, right: OperandValue) -> pd.Series:
        left = left.as_series()
        right = right.as_series()
        return self._compose_list_diff(left, right)


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

    def _agg_op(self) -> DateListDiffAggWithinBounds:
        return DateListDiffAggWithinBounds(
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            aggregation=self.aggregation,
            normalize=self.normalize or False,
        )


class DatePercentileBase(PandasOperator, abc.ABC):
    is_binary: bool = True
    is_categorical: bool = True
    output_type: Optional[str] = "category"

    date_unit: Optional[str] = None

    def calculate_binary(self, left: OperandValue, right: OperandValue) -> pd.Series:
        left = left.as_series()
        right = right.as_series()
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

import abc
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from upgini.autofe.operand import OperandValue
from upgini.autofe.operator import PandasOperator
from upgini.autofe.timeseries.numpy_kernels import (
    _apply_kernel_by_group_ids,
    apply_offset_arrays,
    apply_offset_grouped,
    pack_group_ids,
    scatter_grouped_times,
    unique_sorted_by_group_time,
)


class TimeSeriesBase(PandasOperator, abc.ABC):
    is_vector: bool = True
    date_unit: Optional[str] = None
    offset_size: int = 0
    offset_unit: str = "D"

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "date_unit": self.date_unit,
                "offset_size": self.offset_size,
                "offset_unit": self.offset_unit,
            }
        )
        return res

    def _array_kernel(self) -> Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]:
        """Per-group ``kernel(times_ns, values) -> values``. None keeps the DataFrame path."""
        return None

    def calculate_vector(self, data: List[OperandValue]) -> pd.Series:
        kernel = self._array_kernel()
        if kernel is None:
            return self._calculate_vector_frame(data)
        return self._calculate_vector_arrays(data, kernel)

    def _calculate_vector_arrays(
        self,
        data: List[OperandValue],
        kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> pd.Series:
        data = [operand.as_series() for operand in data]
        date = pd.to_datetime(data[0], unit=self.date_unit, errors="coerce")
        value_col = data[-1].name
        n = len(date)
        times_o = np.asarray(pd.DatetimeIndex(date).asi8, dtype=np.int64)
        values_o = pd.to_numeric(data[-1], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        group_cols = [s.to_numpy() for s in data[1:-1]]
        group_ids_o, null_group = pack_group_ids(group_cols, n)
        keep = date.notna().to_numpy() & ~null_group
        times_s, values_s, group_ids_s, scatter_idx, scatter_slot = unique_sorted_by_group_time(
            times_o, values_o, group_ids_o, keep
        )
        if self.offset_size > 0:
            times_s, values_s, group_ids_s = apply_offset_arrays(
                times_s, values_s, group_ids_s, self.offset_size, self.offset_unit
            )
            scatter_idx = scatter_slot = None
        if len(values_s) == 0:
            out = np.full(n, np.nan, dtype=np.float64)
        else:
            out_s = _apply_kernel_by_group_ids(times_s, values_s, group_ids_s, kernel)
            out = scatter_grouped_times(
                times_s,
                group_ids_s,
                out_s,
                times_o,
                group_ids_o,
                keep,
                scatter_idx=scatter_idx,
                scatter_slot=scatter_slot,
            )
        return pd.Series(out, index=date.index, name=value_col)

    def _calculate_vector_frame(self, data: List[OperandValue]) -> pd.Series:
        data = [operand.as_series() for operand in data]
        # assuming first is date, last is value, rest is group columns
        date = pd.to_datetime(data[0], unit=self.date_unit, errors="coerce")
        group_cols = [c.name for c in data[1:-1]]
        value_col = data[-1].name

        ts = pd.concat([date] + data[1:], axis=1)
        ts.drop_duplicates(subset=ts.columns[:-1], keep="first", inplace=True)
        ts.set_index(date.name, inplace=True)
        ts = ts[ts.index.notna()].sort_index()

        if self.offset_size > 0:
            ts = apply_offset_grouped(ts, group_cols, value_col, self.offset_size, self.offset_unit)
        elif group_cols:
            date_name = ts.index.name or "index"
            ts = (
                ts.reset_index()
                .sort_values(group_cols + [date_name], kind="mergesort")
                .set_index(date_name)
            )

        if group_cols:
            ts = ts.groupby(group_cols, group_keys=True, sort=False)

        ts = self._aggregate(ts)
        ts = ts.reindex(data[1:-1] + [date] if group_cols else date).reset_index()
        ts.index = date.index

        return ts.iloc[:, -1]

    def _aggregate(self, ts: pd.DataFrame) -> pd.DataFrame:
        """DataFrame path used by operators without `_array_kernel` (EWMA, trend, cross)."""
        raise NotImplementedError(f"{type(self).__name__} must implement _aggregate")

    def _add_offset_to_formula(self, base_formula: str) -> str:
        if self.offset_size > 0:
            return f"{base_formula}_offset_{self.offset_size}{self.offset_unit}"
        return base_formula

    @classmethod
    def _parse_offset_from_formula(cls, formula: str, base_regex: str) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Parse the offset component from a formula.

        Args:
            formula: The formula to parse
            base_regex: The regex pattern for the base formula (without offset)

        Returns:
            A tuple with:
            - Dictionary with offset parameters if found, None otherwise
            - Remaining part of the formula after removing offset component (for further parsing)
        """
        import re

        offset_regex = f"{base_regex}_offset_(\\d+)([a-zA-Z])$"
        match = re.match(offset_regex, formula)

        if match:
            # Get groups from the offset part
            offset_size = int(match.group(match.lastindex - 1))
            offset_unit = match.group(match.lastindex)

            # Return the parameters and the base formula for further parsing if needed
            # Extract the base formula by using the match object
            base_formula = formula[: match.start(match.lastindex - 1) - len("_offset_")]
            return {"offset_size": offset_size, "offset_unit": offset_unit}, base_formula

        # Check if it matches the base regex (no offset)
        if re.match(f"^{base_regex}$", formula) or re.match(f"^{base_regex}_", formula):
            return None, formula

        return None, None

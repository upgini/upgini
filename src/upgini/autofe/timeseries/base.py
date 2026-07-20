import abc
from typing import Dict, List, Optional, Tuple

import pandas as pd
from upgini.autofe.operand import OperandValue
from upgini.autofe.operator import PandasOperator
from upgini.autofe.timeseries.numpy_kernels import apply_offset_grouped


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

    def calculate_vector(self, data: List[OperandValue]) -> pd.Series:
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
            # Keep a stable group-then-time order for contiguous group kernels
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

    def _shift(self, ts: pd.DataFrame) -> pd.DataFrame:
        # Kept for compatibility with callers/tests that may still reference it.
        if self.offset_size > 0:
            return ts.iloc[:, :-1].merge(
                ts.iloc[:, -1].shift(freq=f"{self.offset_size}{self.offset_unit}"),
                left_index=True,
                right_index=True,
            )
        return ts

    @abc.abstractmethod
    def _aggregate(self, ts: pd.DataFrame) -> pd.DataFrame:
        pass

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

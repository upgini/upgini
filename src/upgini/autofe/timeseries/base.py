import abc
from typing import Dict, List, Optional

import numpy as np  # Used in derived classes
import pandas as pd
from upgini.autofe.operator import PandasOperator

# Used in derived classes
try:
    from pydantic import field_validator as validator  # V2
except ImportError:
    from pydantic import validator  # V1


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

    def calculate_vector(self, data: List[pd.Series]) -> pd.Series:
        # assuming first is date, last is value, rest is group columns
        date = pd.to_datetime(data[0], unit=self.date_unit, errors="coerce")
        ts = pd.concat([date] + data[1:], axis=1)
        ts.drop_duplicates(subset=ts.columns[:-1], keep="first", inplace=True)
        ts.set_index(date.name, inplace=True)
        ts = ts[ts.index.notna()].sort_index()
        ts = (
            ts.groupby([c.name for c in data[1:-1]], group_keys=True)
            .apply(self._shift)[data[-1].name]
            .to_frame()
            .reset_index()
            .set_index(date.name)
            .groupby([c.name for c in data[1:-1]])
            if len(data) > 2
            else self._shift(ts)
        )
        ts = self._aggregate(ts)
        ts = ts.reindex(data[1:-1] + [date] if len(data) > 2 else date).reset_index()
        ts.index = date.index

        return ts.iloc[:, -1]

    def _shift(self, ts: pd.DataFrame) -> pd.DataFrame:
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

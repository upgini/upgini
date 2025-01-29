import abc
from typing import Dict, List, Optional

import pandas as pd
from pydantic import validator

from upgini.autofe.operand import PandasOperand, ParametrizedOperand, VectorizableMixin


class Mean(PandasOperand, VectorizableMixin):
    name: str = "mean"
    output_type: Optional[str] = "float"
    is_vector: bool = True
    group_index: int = 0

    def calculate_vector(self, data: List[pd.Series]) -> pd.Series:
        return pd.DataFrame(data).T.fillna(0).mean(axis=1)


class Sum(PandasOperand, VectorizableMixin):
    name: str = "sum"
    is_vector: bool = True
    group_index: int = 0

    def calculate_vector(self, data: List[pd.Series]) -> pd.Series:
        return pd.DataFrame(data).T.fillna(0).sum(axis=1)


class TimeSeriesBase(PandasOperand, abc.ABC):
    is_vector: bool = True
    date_unit: Optional[str] = None

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "date_unit": self.date_unit,
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
        ts = ts.groupby([c.name for c in data[1:-1]]) if len(data) > 2 else ts
        ts = self._aggregate(ts)
        ts = ts.reindex(data[1:-1] + [date] if len(data) > 2 else date).reset_index()
        ts.index = date.index

        return ts.iloc[:, -1]

    @abc.abstractmethod
    def _aggregate(self, ts: pd.DataFrame) -> pd.DataFrame:
        pass


_roll_aggregations = {"norm_mean": lambda x: x[-1] / x.mean()}


class Roll(TimeSeriesBase, ParametrizedOperand):
    aggregation: str
    window_size: int = 1
    window_unit: str = "D"

    @validator("window_unit")
    def validate_window_unit(cls, v: str) -> str:
        try:
            pd.tseries.frequencies.to_offset(v)
            return v
        except ValueError:
            raise ValueError(
                f"Invalid window_unit: {v}. Must be a valid pandas frequency string (e.g. 'D', 'H', 'T', etc)"
            )

    def to_formula(self) -> str:
        return f"roll_{self.window_size}{self.window_unit}_{self.aggregation}"

    @classmethod
    def from_formula(cls, formula: str) -> Optional["Roll"]:
        import re

        pattern = r"^roll_(\d+)([a-zA-Z])_(\w+)$"
        match = re.match(pattern, formula)

        if not match:
            return None

        window_size = int(match.group(1))
        window_unit = match.group(2)
        aggregation = match.group(3)

        return cls(window_size=window_size, window_unit=window_unit, aggregation=aggregation)

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "window_size": self.window_size,
                "window_unit": self.window_unit,
                "aggregation": self.aggregation,
            }
        )
        return res

    def _aggregate(self, ts: pd.DataFrame) -> pd.DataFrame:
        return ts.rolling(f"{self.window_size}{self.window_unit}", min_periods=self.window_size).agg(
            _roll_aggregations.get(self.aggregation, self.aggregation)
        )


class Lag(TimeSeriesBase, ParametrizedOperand):
    lag_size: int
    lag_unit: str = "D"

    def to_formula(self) -> str:
        return f"lag_{self.lag_size}{self.lag_unit}"

    @classmethod
    def from_formula(cls, formula: str) -> Optional["Lag"]:
        import re

        pattern = r"^lag_(\d+)([a-zA-Z])$"
        match = re.match(pattern, formula)

        if not match:
            return None

        lag_size = int(match.group(1))
        lag_unit = match.group(2)

        return cls(lag_size=lag_size, lag_unit=lag_unit)

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        return res

    def _aggregate(self, ts: pd.DataFrame) -> pd.DataFrame:
        lag_window = self.lag_size + 1
        return ts.rolling(f"{lag_window}{self.lag_unit}", min_periods=lag_window).agg(lambda x: x[0])

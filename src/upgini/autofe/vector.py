import abc
from typing import Dict, List, Optional

import pandas as pd

try:
    from pydantic import field_validator as validator  # V2
except ImportError:
    from pydantic import validator  # V1

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
            ts.groupby([c.name for c in data[1:-1]])
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


_roll_aggregations = {"norm_mean": lambda x: x[-1] / x.mean()}


class Roll(TimeSeriesBase, ParametrizedOperand):
    aggregation: str
    window_size: int = 1
    window_unit: str = "D"

    @validator("window_unit")
    @classmethod
    def validate_window_unit(cls, v: str) -> str:
        try:
            pd.tseries.frequencies.to_offset(v)
            return v
        except ValueError:
            raise ValueError(
                f"Invalid window_unit: {v}. Must be a valid pandas frequency string (e.g. 'D', 'H', 'T', etc)"
            )

    def to_formula(self) -> str:
        roll_component = f"roll_{self.window_size}{self.window_unit}"
        if self.offset_size > 0:
            roll_component += f"_offset_{self.offset_size}{self.offset_unit}"
        return f"{roll_component}_{self.aggregation}"

    @classmethod
    def from_formula(cls, formula: str) -> Optional["Roll"]:
        import re

        # Try matching pattern with offset first
        pattern_with_offset = r"^roll_(\d+)([a-zA-Z])_offset_(\d+)([a-zA-Z])_(\w+)$"
        match_with_offset = re.match(pattern_with_offset, formula)

        if match_with_offset:
            window_size = int(match_with_offset.group(1))
            window_unit = match_with_offset.group(2)
            offset_size = int(match_with_offset.group(3))
            offset_unit = match_with_offset.group(4)
            aggregation = match_with_offset.group(5)

            return cls(
                window_size=window_size,
                window_unit=window_unit,
                offset_size=offset_size,
                offset_unit=offset_unit,
                aggregation=aggregation,
            )

        # If no offset pattern found, try basic pattern
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
        return ts.rolling(f"{self.window_size}{self.window_unit}", min_periods=1).agg(
            _roll_aggregations.get(self.aggregation, self.aggregation)
        )


class Lag(TimeSeriesBase, ParametrizedOperand):
    lag_size: int
    lag_unit: str = "D"

    def to_formula(self) -> str:
        lag_component = f"lag_{self.lag_size}{self.lag_unit}"
        if self.offset_size > 0:
            lag_component += f"_offset_{self.offset_size}{self.offset_unit}"
        return lag_component

    @classmethod
    def from_formula(cls, formula: str) -> Optional["Lag"]:
        import re

        # Try matching pattern with offset first
        pattern_with_offset = r"^lag_(\d+)([a-zA-Z])_offset_(\d+)([a-zA-Z])$"
        match_with_offset = re.match(pattern_with_offset, formula)

        if match_with_offset:
            lag_size = int(match_with_offset.group(1))
            lag_unit = match_with_offset.group(2)
            offset_size = int(match_with_offset.group(3))
            offset_unit = match_with_offset.group(4)

            return cls(
                lag_size=lag_size,
                lag_unit=lag_unit,
                offset_size=offset_size,
                offset_unit=offset_unit,
            )

        # If no offset pattern found, try basic pattern
        pattern = r"^lag_(\d+)([a-zA-Z])$"
        match = re.match(pattern, formula)

        if not match:
            return None

        lag_size = int(match.group(1))
        lag_unit = match.group(2)

        return cls(lag_size=lag_size, lag_unit=lag_unit)

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "lag_size": self.lag_size,
                "lag_unit": self.lag_unit,
            }
        )
        return res

    def _aggregate(self, ts: pd.DataFrame) -> pd.DataFrame:
        lag_window = self.lag_size + 1
        return ts.rolling(f"{lag_window}{self.lag_unit}", min_periods=lag_window).agg(lambda x: x[0])

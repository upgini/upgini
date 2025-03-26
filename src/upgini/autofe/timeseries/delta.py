import pandas as pd
from typing import Dict, Optional

from upgini.autofe.operator import ParametrizedOperator
from upgini.autofe.timeseries.base import TimeSeriesBase
from upgini.autofe.timeseries.lag import Lag


class Delta(TimeSeriesBase, ParametrizedOperator):
    delta_size: int
    delta_unit: str = "D"

    def to_formula(self) -> str:
        return f"delta_{self.delta_size}{self.delta_unit}"

    @classmethod
    def from_formula(cls, formula: str) -> Optional["Delta"]:
        import re

        pattern = r"^delta_(\d+)([a-zA-Z])$"
        match = re.match(pattern, formula)

        if not match:
            return None

        delta_size = int(match.group(1))
        delta_unit = match.group(2)

        return cls(delta_size=delta_size, delta_unit=delta_unit)

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "delta_size": self.delta_size,
                "delta_unit": self.delta_unit,
            }
        )
        return res

    def _aggregate(self, ts: pd.DataFrame) -> pd.DataFrame:
        lag0 = Lag(lag_size=0, lag_unit=self.delta_unit)
        lag = Lag(lag_size=self.delta_size, lag_unit=self.delta_unit)
        return lag0._aggregate(ts) - lag._aggregate(ts)

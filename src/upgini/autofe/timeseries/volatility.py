from typing import Dict, Optional

import pandas as pd
from upgini.autofe.operator import ParametrizedOperator
from upgini.autofe.timeseries.base import TimeSeriesBase


class VolatilityBase(TimeSeriesBase):
    @staticmethod
    def _get_returns(ts: pd.Series, freq: str) -> pd.Series:
        return ts.pct_change(freq=freq).fillna(0)


class EWMAVolatility(VolatilityBase, ParametrizedOperator):
    step_size: int = 1
    step_unit: str = "D"
    window_size: int

    def to_formula(self) -> str:
        base_formula = f"ewma_vol_{self.window_size}"
        return self._add_offset_to_formula(base_formula)

    @classmethod
    def from_formula(cls, formula: str) -> Optional["EWMAVolatility"]:
        # Base regex for EWMAVolatility class
        base_regex = r"ewma_vol_(\d+)"

        # Parse offset first
        offset_params, remaining_formula = cls._parse_offset_from_formula(formula, base_regex)

        if remaining_formula is None:
            return None

        # Now parse the window part
        import re

        match = re.match(f"^{base_regex}$", remaining_formula)

        if not match:
            return None

        window_size = int(match.group(1))

        # Create instance with appropriate parameters
        params = {
            "window_size": window_size,
        }

        if offset_params:
            params.update(offset_params)

        return cls(**params)

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "step_size": self.step_size,
                "step_unit": self.step_unit,
                "window_size": self.window_size,
            }
        )
        return res

    def _aggregate(self, ts: pd.DataFrame) -> pd.DataFrame:
        return ts.apply(self._ewma_vol)

    def _ewma_vol(self, x):
        x = pd.DataFrame(x).iloc[:, -1]
        returns = self._get_returns(x, f"{self.step_size}{self.step_unit}")
        return returns.ewm(span=self.window_size).std()


class RollingVolatility(VolatilityBase, ParametrizedOperator):
    step_size: int = 1
    step_unit: str = "D"
    window_size: int
    window_unit: str = "D"

    def to_formula(self) -> str:
        base_formula = f"roll_vol_{self.window_size}{self.window_unit}"
        return self._add_offset_to_formula(base_formula)

    @classmethod
    def from_formula(cls, formula: str) -> Optional["RollingVolatility"]:
        # Base regex for RollingVolatility class
        base_regex = r"roll_vol_(\d+)([a-zA-Z])"

        # Parse offset first
        offset_params, remaining_formula = cls._parse_offset_from_formula(formula, base_regex)

        if remaining_formula is None:
            return None

        # Now parse the window part
        import re

        match = re.match(f"^{base_regex}$", remaining_formula)

        if not match:
            return None

        window_size = int(match.group(1))
        window_unit = match.group(2)

        # Create instance with appropriate parameters
        params = {
            "window_size": window_size,
            "window_unit": window_unit,
        }

        if offset_params:
            params.update(offset_params)

        return cls(**params)

    def get_params(self) -> Dict[str, Optional[str]]:
        res = super().get_params()
        res.update(
            {
                "step_size": self.step_size,
                "step_unit": self.step_unit,
                "window_size": self.window_size,
                "window_unit": self.window_unit,
            }
        )
        return res

    def _aggregate(self, ts: pd.DataFrame) -> pd.DataFrame:
        return ts.apply(self._rolling_vol)

    def _rolling_vol(self, x):
        x = pd.DataFrame(x).iloc[:, -1]
        returns = self._get_returns(x, f"{self.step_size}{self.step_unit}")
        return returns.rolling(f"{self.window_size}{self.window_unit}", min_periods=1).std()

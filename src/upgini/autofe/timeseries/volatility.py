from typing import Dict, Optional, Union

import numpy as np
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
        base_regex = r"ewma_vol_(\d+)"

        offset_params, remaining_formula = cls._parse_offset_from_formula(formula, base_regex)

        if remaining_formula is None:
            return None

        import re

        match = re.match(f"^{base_regex}$", remaining_formula)

        if not match:
            return None

        window_size = int(match.group(1))

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
        return ts.apply(self._ewma_vol).iloc[:, [-1]]

    def _ewma_vol(self, x):
        return_series = isinstance(x, pd.Series)
        x = pd.DataFrame(x)
        returns = self._get_returns(x.iloc[:, -1], f"{self.step_size}{self.step_unit}")
        x.iloc[:, -1] = returns.ewm(span=self.window_size).std()
        return x.iloc[:, -1] if return_series else x


class RollingVolBase(VolatilityBase):
    step_size: int = 1
    step_unit: str = "D"
    window_size: int
    window_unit: str = "D"

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

    def _rolling_vol(
        self, x: Union[pd.DataFrame, pd.Series], window_size: int, window_unit: str, abs_returns: bool = False
    ) -> Union[pd.DataFrame, pd.Series]:
        return_series = isinstance(x, pd.Series)
        x = pd.DataFrame(x)
        returns = self._get_returns(x.iloc[:, -1], f"{self.step_size}{self.step_unit}")
        if abs_returns:
            returns = returns.abs()
        x.iloc[:, -1] = returns.rolling(f"{window_size}{window_unit}", min_periods=1).std()
        return x.iloc[:, -1] if return_series else x


class RollingVolatility(RollingVolBase, ParametrizedOperator):
    abs_returns: bool = False

    def to_formula(self) -> str:
        base_formula = f"roll_vol_{self.window_size}{self.window_unit}"
        return self._add_offset_to_formula(base_formula)

    @classmethod
    def from_formula(cls, formula: str) -> Optional["RollingVolatility"]:
        base_regex = r"roll_vol_(\d+)([a-zA-Z])"

        offset_params, remaining_formula = cls._parse_offset_from_formula(formula, base_regex)

        if remaining_formula is None:
            return None

        import re

        match = re.match(f"^{base_regex}$", remaining_formula)

        if not match:
            return None

        window_size = int(match.group(1))
        window_unit = match.group(2)

        params = {
            "window_size": window_size,
            "window_unit": window_unit,
        }

        if offset_params:
            params.update(offset_params)

        return cls(**params)

    def _aggregate(self, ts: pd.DataFrame) -> pd.DataFrame:
        return ts.apply(
            self._rolling_vol, window_size=self.window_size, window_unit=self.window_unit, abs_returns=self.abs_returns
        ).iloc[:, [-1]]


class RollingVolatility2(RollingVolBase, ParametrizedOperator):
    """
    Computes the volatility on volatility of a time series. Volatility is computed using the RollingVolatility.
    """

    def to_formula(self) -> str:
        base_formula = f"roll_vol2_{self.window_size}{self.window_unit}"
        return self._add_offset_to_formula(base_formula)

    @classmethod
    def from_formula(cls, formula: str) -> Optional["RollingVolatility2"]:
        base_regex = r"roll_vol2_(\d+)([a-zA-Z])"

        offset_params, remaining_formula = cls._parse_offset_from_formula(formula, base_regex)

        if remaining_formula is None:
            return None

        import re

        match = re.match(f"^{base_regex}$", remaining_formula)

        if not match:
            return None

        window_size = int(match.group(1))
        window_unit = match.group(2)

        params = {
            "window_size": window_size,
            "window_unit": window_unit,
        }

        if offset_params:
            params.update(offset_params)

        return cls(**params)

    def _aggregate(self, ts: pd.DataFrame) -> pd.DataFrame:
        return ts.apply(self._vol_on_vol).iloc[:, [-1]]

    def _vol_on_vol(self, x: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        vol1 = self._rolling_vol(x, self.window_size, self.window_unit, abs_returns=True)
        vol2 = self._rolling_vol(vol1, self.window_size, self.window_unit, abs_returns=False)
        return vol2


class VolatilityRatio(RollingVolBase, ParametrizedOperator):
    """
    Computes the ratio of short-term volatility to long-term volatility.
    Both volatilities are computed using RollingVolatility.
    """

    short_window_size: int
    short_window_unit: str = "D"

    def to_formula(self) -> str:
        base_formula = (
            f"vol_ratio_{self.short_window_size}{self.short_window_unit}_to_{self.window_size}{self.window_unit}"
        )
        return self._add_offset_to_formula(base_formula)

    @classmethod
    def from_formula(cls, formula: str) -> Optional["VolatilityRatio"]:
        base_regex = r"vol_ratio_(\d+)([a-zA-Z])_to_(\d+)([a-zA-Z])"

        offset_params, remaining_formula = cls._parse_offset_from_formula(formula, base_regex)

        if remaining_formula is None:
            return None

        import re

        match = re.match(f"^{base_regex}$", remaining_formula)

        if not match:
            return None

        short_window_size = int(match.group(1))
        short_window_unit = match.group(2)
        window_size = int(match.group(3))
        window_unit = match.group(4)

        params = {
            "short_window_size": short_window_size,
            "short_window_unit": short_window_unit,
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
                "short_window_size": self.short_window_size,
                "short_window_unit": self.short_window_unit,
            }
        )
        return res

    def _aggregate(self, ts: pd.DataFrame) -> pd.DataFrame:
        return ts.apply(self._vol_ratio).iloc[:, [-1]]

    def _vol_ratio(self, x: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        short_vol = self._rolling_vol(x, self.short_window_size, self.short_window_unit)
        long_vol = self._rolling_vol(x, self.window_size, self.window_unit)
        ratio = VolatilityRatio._handle_div_errors(short_vol / long_vol)
        return ratio

    @staticmethod
    def _handle_div_errors(x: pd.Series) -> pd.Series:
        return x.replace([np.inf, -np.inf], np.nan).fillna(1)

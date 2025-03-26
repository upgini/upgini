import numpy as np
import pandas as pd

from upgini.autofe.timeseries.base import TimeSeriesBase


class TrendCoefficient(TimeSeriesBase):
    name: str = "trend_coef"

    def _aggregate(self, ts: pd.DataFrame) -> pd.DataFrame:
        return ts.apply(self._trend_coef).fillna(0)

    def _trend_coef(self, x: pd.DataFrame) -> pd.Series:
        x = pd.DataFrame(x)
        resampled = x.iloc[:, -1].resample(self.date_unit or "D").fillna(method="ffill").fillna(method="bfill")
        idx = np.arange(len(resampled))
        coeffs = np.polyfit(idx, resampled, 1)
        return pd.Series([coeffs[0]] * len(x), index=x.index, name=x.columns[-1])

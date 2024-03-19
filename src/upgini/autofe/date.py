from typing import Optional, Union
import numpy as np
import pandas as pd

from upgini.autofe.operand import PandasOperand, VectorizableMixin


class DateDiffMixin:
    diff_unit: str = "D"
    left_unit: Optional[str] = None
    right_unit: Optional[str] = None

    def _convert_to_date(
        self, x: Union[pd.DataFrame, pd.Series], unit: Optional[str]
    ) -> Union[pd.DataFrame, pd.Series]:
        if isinstance(x, pd.DataFrame):
            return x.apply(lambda y: self._convert_to_date(y, unit), axis=1)

        return pd.to_datetime(x, unit=unit)


class DateDiff(PandasOperand, DateDiffMixin):
    name = "date_diff"
    is_binary = True
    has_symmetry_importance = True

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        left = self._convert_to_date(left, self.left_unit)
        right = self._convert_to_date(right, self.right_unit)
        return self.__replace_negative((left - right) / np.timedelta64(1, self.diff_unit))

    def __replace_negative(self, x: Union[pd.DataFrame, pd.Series]):
        x[x < 0] = None
        return x


class DateDiffFuture(PandasOperand, DateDiffMixin):
    name = "date_diff_future"
    is_binary = True
    has_symmetry_importance = True
    is_vectorizable = False

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        left = self._convert_to_date(left, self.left_unit)
        right = self._convert_to_date(right, self.right_unit)
        future = pd.to_datetime(dict(day=right.dt.day, month=right.dt.month, year=left.dt.year))
        before = future[future < left]
        future[future < left] = pd.to_datetime(dict(day=before.dt.day, month=before.dt.month, year=before.dt.year + 1))
        diff = (future - left) / np.timedelta64(1, self.diff_unit)

        return diff

import abc
from typing import Any, List, Optional, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel

from upgini.autofe.operand import PandasOperand


class DateDiffMixin(BaseModel):
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


class DateDiffType2(PandasOperand, DateDiffMixin):
    name = "date_diff_type2"
    is_binary = True
    has_symmetry_importance = True

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        left = self._convert_to_date(left, self.left_unit)
        right = self._convert_to_date(right, self.right_unit)
        future = right + (left.dt.year - right.dt.year).apply(
            lambda y: np.datetime64("NaT") if np.isnan(y) else pd.tseries.offsets.DateOffset(years=y)
        )
        before = future[future < left]
        future[future < left] = before + pd.tseries.offsets.DateOffset(years=1)
        diff = (future - left) / np.timedelta64(1, self.diff_unit)

        return diff


class DateListDiff(PandasOperand, DateDiffMixin):
    is_binary = True
    has_symmetry_importance = True
    aggregation: str

    def __init__(self, **data: Any) -> None:
        if "name" not in data:
            data["name"] = f"date_diff_{data.get('aggregation')}"
        super().__init__(**data)

    def map_diff(self, left: np.datetime64, right: list) -> list:
        return (left - self._convert_to_date(pd.Series(right), self.right_unit)) / np.timedelta64(1, self.diff_unit)

    def reduce(self, diff_list: pd.Series) -> float:
        return diff_list[diff_list > 0].aggregate(self.aggregation)

    def calculate_binary(self, left: pd.Series, right: pd.Series) -> pd.Series:
        left = self._convert_to_date(left, self.left_unit)

        return pd.Series(left.index.map(lambda i: self.reduce(self.map_diff(left.loc[i], right.loc[i]))))


class DateListDiffBounded(DateListDiff):
    lower_bound: Optional[int]
    upper_bound: Optional[int]
    inclusive: Optional[str]

    def __init__(self, **data: Any) -> None:
        if "name" not in data:
            inclusive = data.get("inclusive")
            lower_bound = data.get("lower_bound")
            upper_bound = data.get("upper_bound")
            components = [
                "date_diff",
                data.get("diff_unit"),
                str(lower_bound if lower_bound is not None else "minusinf"),
                str(upper_bound if upper_bound is not None else "plusinf"),
            ]
            if inclusive:
                components.append(inclusive)
            components.append(data.get("aggregation"))
            data["name"] = "_".join(components)
        super().__init__(**data)

    def reduce(self, diff_list: pd.Series) -> float:
        return diff_list[
            (diff_list > 0)
            & (
                diff_list.between(
                    self.lower_bound or -np.inf, self.upper_bound or np.inf, inclusive=self.inclusive or "left"
                )
            )
        ].aggregate(self.aggregation)

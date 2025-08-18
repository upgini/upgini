from dataclasses import dataclass, field
from typing import List

import pandas as pd

# Constants for SampleConfig
TS_MIN_DIFFERENT_IDS_RATIO = 0.2
TS_DEFAULT_HIGH_FREQ_TRUNC_LENGTHS = [pd.DateOffset(years=2, months=6), pd.DateOffset(years=2, days=7)]
TS_DEFAULT_LOW_FREQ_TRUNC_LENGTHS = [pd.DateOffset(years=7), pd.DateOffset(years=5)]
TS_DEFAULT_TIME_UNIT_THRESHOLD = pd.Timedelta(weeks=4)
FIT_SAMPLE_ROWS_TS = 100_000

BINARY_MIN_SAMPLE_THRESHOLD = 5_000
MULTICLASS_MIN_SAMPLE_THRESHOLD = 25_000
BINARY_BOOTSTRAP_LOOPS = 5
MULTICLASS_BOOTSTRAP_LOOPS = 2

FIT_SAMPLE_THRESHOLD = 100_000
FIT_SAMPLE_ROWS = 100_000
FIT_SAMPLE_ROWS_WITH_EVAL_SET = 100_000
FIT_SAMPLE_THRESHOLD_WITH_EVAL_SET = 100_000


@dataclass
class SampleConfig:
    force_sample_size: int = 7000
    ts_min_different_ids_ratio: float = TS_MIN_DIFFERENT_IDS_RATIO
    ts_default_high_freq_trunc_lengths: List[pd.DateOffset] = field(
        default_factory=TS_DEFAULT_HIGH_FREQ_TRUNC_LENGTHS.copy
    )
    ts_default_low_freq_trunc_lengths: List[pd.DateOffset] = field(
        default_factory=TS_DEFAULT_LOW_FREQ_TRUNC_LENGTHS.copy
    )
    ts_default_time_unit_threshold: pd.Timedelta = TS_DEFAULT_TIME_UNIT_THRESHOLD
    binary_min_sample_threshold: int = BINARY_MIN_SAMPLE_THRESHOLD
    multiclass_min_sample_threshold: int = MULTICLASS_MIN_SAMPLE_THRESHOLD
    binary_bootstrap_loops: int = BINARY_BOOTSTRAP_LOOPS
    multiclass_bootstrap_loops: int = MULTICLASS_BOOTSTRAP_LOOPS
    fit_sample_threshold: int = FIT_SAMPLE_THRESHOLD
    fit_sample_rows: int = FIT_SAMPLE_ROWS
    fit_sample_rows_with_eval_set: int = FIT_SAMPLE_ROWS_WITH_EVAL_SET
    fit_sample_threshold_with_eval_set: int = FIT_SAMPLE_THRESHOLD_WITH_EVAL_SET
    fit_sample_rows_ts: int = FIT_SAMPLE_ROWS_TS

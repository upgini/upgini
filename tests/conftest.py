import numpy as np
import pandas as pd
import pytest

from upgini import Dataset, FileColumnMeaningType, FileMetrics


@pytest.fixture
def etalon():
    d = 1577836800000
    data = pd.DataFrame(
        [
            [d, 33333333, 0, 0.5],
            [d, 33333333, 0, 0.5],
            [d, 44444444, 1, None],
            [d, 55555555, None, 0.5],
            [d, 66666666, None, 0.5],
            [None, 77777777, 1, 0.5],
            [None, 88888888, 1, 0.5],
            [d, 99999999, np.Inf, 0.5],
            [d, 11111111, np.NaN, 0.5],
            [None, None, None, None],
        ],
        columns=["timestamp", "msisdn", "target", "score"],
    )

    definition = {
        "msisdn": FileColumnMeaningType.MSISDN,
        "timestamp": FileColumnMeaningType.DATE,
        "target": FileColumnMeaningType.TARGET,
        "score": FileColumnMeaningType.SCORE,
    }
    search_keys = [("msisdn", "timestamp")]
    etalon = Dataset(
        dataset_name="test_etalon",  # type: ignore
        description="test etalon",  # type: ignore
        df=data,  # type: ignore
        meaning_types=definition,  # type: ignore
        search_keys=search_keys,  # type: ignore
    )
    return etalon


@pytest.fixture
def expected_binary_etalon_metadata():
    metadata = {
        "task_type": "BINARY",
        "label": "auc",
        "count": 15555,
        "valid_count": 15555,
        "valid_rate": 100.0,
        "avg_target": 0.5014464802314368,
        "cuts": [
            1541017353600.0,
            1543204800000.0,
            1545379200000.0,
            1547553600000.0,
            1549728000000.0,
            1551902400000.0,
            1554076800000.0,
        ],
        "interval": [
            {"date_cut": 1542111076799.0, "count": 2998.0, "valid_count": 2998.0, "avg_target": 0.5},
            {"date_cut": 1544292000000.0, "count": 3000.0, "valid_count": 3000.0, "avg_target": 0.5},
            {"date_cut": 1546466400000.0, "count": 2996.0, "valid_count": 2996.0, "avg_target": 0.49966622162883845},
            {"date_cut": 1548640800000.0, "count": 1791.0, "valid_count": 1791.0, "avg_target": 0.5147962032384142},
            {"date_cut": 1550815200000.0, "count": 1770.0, "valid_count": 1770.0, "avg_target": 0.49830508474576274},
            {"date_cut": 1552989600000.0, "count": 3000.0, "valid_count": 3000.0, "avg_target": 0.5},
        ],
    }
    return FileMetrics.parse_obj(metadata)

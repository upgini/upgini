import os

import pandas as pd
import pytest

from upgini import Dataset, FileColumnMeaningType
from upgini.metadata import ModelTaskType

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "test_data/binary/",
)


@pytest.fixture
def etalon_definition():
    return {
        "phone_num": FileColumnMeaningType.MSISDN,
        "rep_date": FileColumnMeaningType.DATE,
        "target": FileColumnMeaningType.TARGET,
    }


@pytest.fixture
def etalon_search_keys():
    return [("phone_num", "rep_date")]


@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, "data.csv.gz"))
def test_binary_dataset_pandas(datafiles, etalon_definition, etalon_search_keys):
    df = pd.read_csv(datafiles / "data.csv.gz")
    ds = Dataset(
        dataset_name="test Dataset",
        description="test",
        df=df,
        meaning_types=etalon_definition,
        search_keys=etalon_search_keys,
        model_task_type=ModelTaskType.BINARY,
    )
    ds.validate()
    expected_valid_rows = 15555
    assert len(ds) == expected_valid_rows


@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, "data.csv"))
def test_binary_dataset_path(datafiles, etalon_definition, etalon_search_keys):
    path = datafiles / "data.csv"
    ds_path = Dataset(
        dataset_name="test Dataset",
        description="test",
        path=path,
        meaning_types=etalon_definition,
        search_keys=etalon_search_keys,
        model_task_type=ModelTaskType.BINARY,
    )
    ds_path.validate()
    expected_valid_rows = 15555
    assert len(ds_path) == expected_valid_rows

import os
from typing import Dict

import pandas as pd
import pytest

from upgini.ads import FileColumnMeaningType
from upgini.dataset import Dataset
from upgini.utils.datetime_utils import DateTimeSearchKeyConverter

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "test_data/continuous/",
)


@pytest.fixture
def etalon_definition() -> Dict[str, FileColumnMeaningType]:
    return {
        "phone_num": FileColumnMeaningType.MSISDN,
        "rep_date": FileColumnMeaningType.DATE,
        "score": FileColumnMeaningType.TARGET,
    }


@pytest.fixture
def etalon_search_keys():
    return [("phone_num", "rep_date")]


@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, "data.csv.gz"))
def test_continuous_dataset(datafiles, etalon_definition: Dict[str, FileColumnMeaningType], etalon_search_keys):
    df = pd.read_csv(datafiles / "data.csv.gz")
    df = df.reset_index().rename(columns={"index": "system_record_id", "score": "target"})
    del etalon_definition["score"]
    etalon_definition["target"] = FileColumnMeaningType.TARGET
    converter = DateTimeSearchKeyConverter("rep_date")
    df = converter.convert(df)
    ds = Dataset(
        dataset_name="test Dataset",  # type: ignore
        description="test",  # type: ignore
        df=df,  # type: ignore
        meaning_types=etalon_definition,  # type: ignore
        search_keys=etalon_search_keys,  # type: ignore
    )
    ds.columns_renaming = {c: c for c in df.columns}
    ds.validate()
    expected_valid_rows = 20401
    assert len(ds) == expected_valid_rows

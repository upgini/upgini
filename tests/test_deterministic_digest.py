import pandas as pd

from upgini.ads import FileColumnMeaningType
from upgini.dataset import Dataset
from upgini.metadata import ModelTaskType

GOLDEN_DETERMINISTIC_DIGEST = "846732431fa14cad5e95330888ec2ad21096490d228cd3afb329b004d582987c"


def test_deterministic_digest():
    df = pd.DataFrame(
        {
            "system_record_id": [1, 2, 3],
            "phone": ["111", "222", "333"],
            "feature1": [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
        }
    )
    dataset = Dataset(
        dataset_name="golden_digest",
        description="golden digest fixture",
        df=df,
        meaning_types={
            "phone": FileColumnMeaningType.MSISDN,
            "feature1": FileColumnMeaningType.FEATURE,
            "target": FileColumnMeaningType.TARGET,
        },
        search_keys=[("phone",)],
        model_task_type=ModelTaskType.BINARY,
    )
    dataset.columns_renaming = {column: column for column in df.columns}

    metadata = dataset._Dataset__construct_metadata()

    assert metadata.deterministicDigest == GOLDEN_DETERMINISTIC_DIGEST

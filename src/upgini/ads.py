import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from upgini import SearchKey
from upgini.http import get_rest_client
from upgini.metadata import (
    DataType,
    FileColumnMeaningType,
    FileColumnMetadata,
    FileMetadata,
)
from upgini.resource_bundle import bundle


def upload_user_ads(name: str, df: pd.DataFrame, search_keys: Dict[str, SearchKey], description: Optional[str] = None):
    if df.shape[0] < 1000:
        raise ValueError(bundle.get("ads_upload_too_few_rows"))

    for column, _ in search_keys.items():
        if column not in df.columns:
            raise ValueError(bundle.get("ads_upload_search_key_not_found").format(column))

    columns = []
    rows_count = df.shape[0]
    min_valid_rows_count = rows_count / 2
    for idx, (column_name, column_type) in enumerate(zip(df.columns, df.dtypes)):
        if column_name in search_keys:
            if df[column_name].notnull().sum() < min_valid_rows_count:
                raise ValueError(bundle.get("ads_upload_to_many_empty_rows"))
            meaning_type = search_keys[column_name].value
            if (
                meaning_type == FileColumnMeaningType.MSISDN
                and not is_string_dtype(df[column_name])
                and not is_object_dtype(df[column_name])
            ):
                df[column_name] = df[column_name].values.astype(np.int64).astype("string")  # type: ignore
        else:
            meaning_type = FileColumnMeaningType.FEATURE
        columns.append(
            FileColumnMetadata(
                index=idx, name=column_name, dataType=__get_data_type(str(column_type)), meaningType=meaning_type
            )
        )

    metadata = FileMetadata(
        name=name, description=description, columns=columns, searchKeys=[list(search_keys.keys())], rowsCount=rows_count
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        parquet_file_path = f"{tmp_dir}/{name}.parquet"
        df.to_parquet(path=parquet_file_path, index=True, compression="gzip", engine="fastparquet")
        logging.debug(f"Size of prepared uploading file: {Path(parquet_file_path).stat().st_size}")
        response = get_rest_client().upload_user_ads(parquet_file_path, metadata)
        logging.debug(f"Upload response: {response}")

    print(bundle.get("ads_upload_finish"))


def __get_data_type(pandas_data_type) -> DataType:
    if pandas_data_type in {"int64", "Int64", "int"}:
        return DataType.INT
    elif pandas_data_type in {"float64", "Float64", "float"}:
        return DataType.DECIMAL
    else:
        return DataType.STRING

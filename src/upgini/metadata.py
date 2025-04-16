from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

SYSTEM_RECORD_ID = "system_record_id"
ENTITY_SYSTEM_RECORD_ID = "entity_system_record_id"
SEARCH_KEY_UNNEST = "search_key_unnest"
SORT_ID = "sort_id"
EVAL_SET_INDEX = "eval_set_index"
TARGET = "target"
COUNTRY = "country_iso_code"
RENAMED_INDEX = "index_col"
DEFAULT_INDEX = "index"
ORIGINAL_INDEX = "original_index"
SYSTEM_COLUMNS = {SYSTEM_RECORD_ID, ENTITY_SYSTEM_RECORD_ID, SEARCH_KEY_UNNEST, EVAL_SET_INDEX, TARGET, COUNTRY}


class FileColumnMeaningType(Enum):
    MSISDN = "MSISDN"
    MSISDN_RANGE_FROM = "MSISDN_RANGE_FROM"
    MSISDN_RANGE_TO = "MSISDN_RANGE_TO"
    EMAIL = "EMAIL"
    EMAIL_ONE_DOMAIN = "EMAIL_ONE_DOMAIN"
    IP_ADDRESS = "IP_ADDRESS"
    IPV6_ADDRESS = "IPV6_ADDRESS"
    IPV6_RANGE_FROM = "IPV6_RANGE_FROM"
    IPV6_RANGE_TO = "IPV6_RANGE_TO"
    IP_RANGE_FROM = "IP_RANGE_FROM"
    IP_RANGE_TO = "IP_RANGE_TO"
    HEM = "HEM"
    DATE = "DATE"
    DATETIME = "DATETIME"
    SCORE = "SCORE"
    TARGET = "TARGET"
    FEATURE = "FEATURE"
    CUSTOM_KEY = "CUSTOM_KEY"
    COUNTRY = "COUNTRY"
    POSTAL_CODE = "POSTAL_CODE"
    SYSTEM_RECORD_ID = "SYSTEM_RECORD_ID"
    EVAL_SET_INDEX = "EVAL_SET_INDEX"
    ENTITY_SYSTEM_RECORD_ID = "ENTITY_SYSTEM_RECORD_ID"
    UNNEST_KEY = "UNNEST_KEY"
    IP_BINARY = "IP_BINARY"
    IP_PREFIX = "IP_PREFIX"
    IP_RANGE_FROM_BINARY = "IP_RANGE_FROM_BINARY"
    IP_RANGE_TO_BINARY = "IP_RANGE_TO_BINARY"


class SearchKey(Enum):
    EMAIL = FileColumnMeaningType.EMAIL
    HEM = FileColumnMeaningType.HEM
    IP = FileColumnMeaningType.IP_ADDRESS
    PHONE = FileColumnMeaningType.MSISDN
    DATE = FileColumnMeaningType.DATE
    DATETIME = FileColumnMeaningType.DATETIME
    CUSTOM_KEY = FileColumnMeaningType.CUSTOM_KEY
    COUNTRY = FileColumnMeaningType.COUNTRY
    POSTAL_CODE = FileColumnMeaningType.POSTAL_CODE

    # For data source registration. Don't use it for FeaturesEnricher
    IPV6_ADDRESS = FileColumnMeaningType.IPV6_ADDRESS
    IPV6_RANGE_FROM = FileColumnMeaningType.IPV6_RANGE_FROM
    IPV6_RANGE_TO = FileColumnMeaningType.IPV6_RANGE_TO
    IP_BINARY = FileColumnMeaningType.IP_BINARY
    IP_RANGE_FROM_BINARY = FileColumnMeaningType.IP_RANGE_FROM_BINARY
    IP_RANGE_TO_BINARY = FileColumnMeaningType.IP_RANGE_TO_BINARY
    IP_PREFIX = FileColumnMeaningType.IP_PREFIX

    # For data source registration. Don't use it for FeaturesEnricher
    EMAIL_ONE_DOMAIN = FileColumnMeaningType.EMAIL_ONE_DOMAIN
    # For data source registration. Don't use it for FeaturesEnricher
    IP_RANGE_FROM = FileColumnMeaningType.IP_RANGE_FROM
    # For data source registration. Don't use it for FeaturesEnricher
    IP_RANGE_TO = FileColumnMeaningType.IP_RANGE_TO
    # For data source registration. Don't use it for FeaturesEnricher
    MSISDN_RANGE_FROM = FileColumnMeaningType.MSISDN_RANGE_FROM
    # For data source registration. Don't use it for FeaturesEnricher
    MSISDN_RANGE_TO = FileColumnMeaningType.MSISDN_RANGE_TO

    @staticmethod
    def personal_keys() -> List["SearchKey"]:
        return [SearchKey.EMAIL, SearchKey.HEM, SearchKey.IP, SearchKey.PHONE]

    @staticmethod
    def from_meaning_type(meaning_type: FileColumnMeaningType) -> "SearchKey":
        if meaning_type == FileColumnMeaningType.EMAIL:
            return SearchKey.EMAIL
        if meaning_type == FileColumnMeaningType.HEM:
            return SearchKey.HEM  # TODO check that it wasn't EMAIL
        if meaning_type == FileColumnMeaningType.IP_ADDRESS:
            return SearchKey.IP
        if meaning_type == FileColumnMeaningType.MSISDN:
            return SearchKey.PHONE
        if meaning_type == FileColumnMeaningType.DATE:
            return SearchKey.DATE
        if meaning_type == FileColumnMeaningType.DATETIME:
            return SearchKey.DATETIME
        if meaning_type == FileColumnMeaningType.CUSTOM_KEY:
            return SearchKey.CUSTOM_KEY
        if meaning_type == FileColumnMeaningType.COUNTRY:
            return SearchKey.COUNTRY
        if meaning_type == FileColumnMeaningType.POSTAL_CODE:
            return SearchKey.POSTAL_CODE
        if meaning_type == FileColumnMeaningType.IPV6_ADDRESS:
            return SearchKey.IP
        # if meaning_type == FileColumnMeaningType.IPV6_RANGE_FROM:
        #     return SearchKey.IPV6_RANGE_FROM
        # if meaning_type == FileColumnMeaningType.IPV6_RANGE_TO:
        #     return SearchKey.IPV6_RANGE_TO
        # if meaning_type == FileColumnMeaningType.EMAIL_ONE_DOMAIN:
        #     return SearchKey.EMAIL_ONE_DOMAIN
        # if meaning_type == FileColumnMeaningType.IP_RANGE_FROM:
        #     return SearchKey.IP_RANGE_FROM
        # if meaning_type == FileColumnMeaningType.IP_RANGE_TO:
        #     return SearchKey.IP_RANGE_TO
        # if meaning_type == FileColumnMeaningType.MSISDN_RANGE_FROM:
        #     return SearchKey.MSISDN_RANGE_FROM
        # if meaning_type == FileColumnMeaningType.MSISDN_RANGE_TO:
        #     return SearchKey.MSISDN_RANGE_TO
        if meaning_type == FileColumnMeaningType.IP_BINARY:
            return SearchKey.IP
        # if meaning_type == FileColumnMeaningType.IP_RANGE_FROM_BINARY:
        #     return SearchKey.IP_RANGE_FROM_BINARY
        # if meaning_type == FileColumnMeaningType.IP_RANGE_TO_BINARY:
        #     return SearchKey.IP_RANGE_TO_BINARY

    @staticmethod
    def find_key(search_keys: Dict[str, SearchKey], keys: Union[SearchKey, List[SearchKey]]) -> Optional[SearchKey]:
        if isinstance(keys, SearchKey):
            keys = [keys]
        for col, key_type in search_keys.items():
            if key_type in keys:
                return col
        return None

    @staticmethod
    def find_all_keys(search_keys: Dict[str, SearchKey], keys: Union[SearchKey, List[SearchKey]]) -> List[SearchKey]:
        if isinstance(keys, SearchKey):
            keys = [keys]
        return [col for col, key_type in search_keys.items() if key_type in keys]


class DataType(Enum):
    INT = "INT"
    DECIMAL = "DECIMAL"
    BIG_NUMERIC = "BIG_NUMERIC"
    DATE_TIME = "DATE_TIME"
    STRING = "STRING"
    BOOLEAN = "BOOLEAN"
    BYTES = "BYTES"


class ModelTaskType(Enum):
    BINARY = "BINARY"
    MULTICLASS = "MULTICLASS"
    REGRESSION = "REGRESSION"
    TIMESERIES = "TIMESERIES"


class ModelLabelType(Enum):
    GINI = "gini"
    AUC = "auc"
    RMSE = "rmse"
    ACCURACY = "accuracy"


class RegressionTask(BaseModel):
    mse: float
    rmse: float
    msle: float
    rmsle: float


class BinaryTask(BaseModel):
    auc: float
    gini: float


class MulticlassTask(BaseModel):
    accuracy: float


class FileMetricsInterval(BaseModel):
    date_cut: float
    count: float
    valid_count: float
    avg_target: Optional[float] = None  # not for multiclass
    avg_score_etalon: Optional[float] = None


class FileMetrics(BaseModel):
    # etalon metadata
    task_type: Optional[ModelTaskType] = None
    label: Optional[ModelLabelType] = None
    count: Optional[int] = None
    valid_count: Optional[int] = None
    valid_rate: Optional[float] = None
    avg_target: Optional[float] = None
    metrics_binary_etalon: Optional[BinaryTask] = None
    metrics_regression_etalon: Optional[RegressionTask] = None
    metrics_multiclass_etalon: Optional[MulticlassTask] = None
    cuts: Optional[List[float]] = None
    interval: Optional[List[FileMetricsInterval]] = None


class NumericInterval(BaseModel):
    minValue: int
    maxValue: int


class FileColumnMetadata(BaseModel):
    index: int
    name: str
    dataType: DataType
    meaningType: FileColumnMeaningType
    minMaxValues: Optional[NumericInterval] = None
    originalName: Optional[str] = None
    # is this column contains keys from multiple key columns like msisdn1, msisdn2
    isUnnest: bool = False
    # list of original etalon key column names like msisdn1, msisdn2
    unnestKeyNames: Optional[List[str]] = None


class FileMetadata(BaseModel):
    name: str
    description: Optional[str] = None
    columns: List[FileColumnMetadata]
    searchKeys: List[List[str]]
    excludeFeaturesSources: Optional[List[str]] = None
    hierarchicalGroupKeys: Optional[List[str]] = None
    hierarchicalSubgroupKeys: Optional[List[str]] = None
    taskType: Optional[ModelTaskType] = None
    rowsCount: Optional[int] = None
    checksumMD5: Optional[str] = None
    digest: Optional[str] = None

    def column_by_name(self, name: str) -> Optional[FileColumnMetadata]:
        for c in self.columns:
            if c.name == name:
                return c
        return None

    def search_types(self) -> Dict[SearchKey, str]:
        search_keys = dict()
        for keys_group in self.searchKeys:
            for key in keys_group:
                column = self.column_by_name(key)
                if column:
                    search_key = SearchKey.from_meaning_type(column.meaningType)
                    if search_key is not None:
                        search_keys[search_key] = column.name
        return search_keys


class FeaturesMetadataV2(BaseModel):
    name: str
    type: str
    source: str
    hit_rate: float
    shap_value: float
    commercial_schema: Optional[str] = None
    data_provider: Optional[str] = None
    data_providers: Optional[List[str]] = None
    data_provider_link: Optional[str] = None
    data_provider_links: Optional[List[str]] = None
    data_source: Optional[str] = None
    data_sources: Optional[List[str]] = None
    data_source_link: Optional[str] = None
    data_source_links: Optional[List[str]] = None
    doc_link: Optional[str] = None
    update_frequency: Optional[str] = None
    from_online_api: Optional[bool] = None


class HitRateMetrics(BaseModel):
    etalon_row_count: int
    max_hit_count: int
    join_flags_count: Dict[str, int] = {}
    hit_rate: float
    hit_rate_percent: float


class ModelEvalSet(BaseModel):
    eval_set_index: int
    hit_rate: float
    hit_rate_metrics: HitRateMetrics


class BaseColumnMetadata(BaseModel):
    original_name: str
    hashed_name: str
    ads_definition_id: Optional[str] = None
    is_augmented: bool


class GeneratedFeatureMetadata(BaseModel):
    alias: Optional[str] = None
    formula: str
    display_index: str
    base_columns: List[BaseColumnMetadata]
    operator_params: Optional[Dict[str, str]] = None


class ProviderTaskMetadataV2(BaseModel):
    features: List[FeaturesMetadataV2]
    hit_rate_metrics: Optional[HitRateMetrics] = None
    eval_set_metrics: Optional[List[ModelEvalSet]] = None
    zero_hit_rate_search_keys: Optional[List[str]] = None
    features_used_for_embeddings: Optional[List[str]] = None
    shuffle_kfold: Optional[bool] = None
    generated_features: Optional[List[GeneratedFeatureMetadata]] = None


class FeaturesFilter(BaseModel):
    minImportance: Optional[float] = None
    maxPSI: Optional[float] = None
    maxCount: Optional[int] = None
    selectedFeatures: Optional[List[str]] = None


class RuntimeParameters(BaseModel):
    properties: Dict[str, Any] = {}


class AutoFEParameters(BaseModel):
    ts_gap_days: int = 0


class SearchCustomization(BaseModel):
    featuresFilter: Optional[FeaturesFilter] = None
    extractFeatures: Optional[bool] = None
    accurateModel: Optional[bool] = None
    importanceThreshold: Optional[float] = None
    maxFeatures: Optional[int] = None
    returnScores: Optional[bool] = None
    runtimeParameters: Optional[RuntimeParameters] = None
    metricsCalculation: Optional[bool] = None

    def __repr__(self):
        return (
            f"Features filter: {self.featuresFilter}, "
            f"extract features: {self.extractFeatures}, "
            f"accurate model: {self.accurateModel}, "
            f"importance threshold: {self.importanceThreshold}, "
            f"max features: {self.maxFeatures}, "
            f"return scores: {self.returnScores}, "
            f"runtimeParameters: {self.runtimeParameters}, "
            f"metricsCalculation: {self.metricsCalculation}"
        )


class CVType(Enum):
    k_fold = "k_fold"
    group_k_fold = "group_k_fold"
    time_series = "time_series"
    blocked_time_series = "blocked_time_series"
    not_set = "not_set"

    def is_time_series(self) -> bool:
        return self in [CVType.time_series, CVType.blocked_time_series]

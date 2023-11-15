from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel

SYSTEM_RECORD_ID = "system_record_id"
EVAL_SET_INDEX = "eval_set_index"
TARGET = "target"
COUNTRY = "country_iso_code"
RENAMED_INDEX = "index_col"
DEFAULT_INDEX = "index"
ORIGINAL_INDEX = "original_index"
SYSTEM_COLUMNS = {SYSTEM_RECORD_ID, EVAL_SET_INDEX, TARGET, COUNTRY}


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


class DataType(Enum):
    INT = "INT"
    DECIMAL = "DECIMAL"
    BIG_NUMERIC = "BIG_NUMERIC"
    DATE_TIME = "DATE_TIME"
    STRING = "STRING"
    BOOLEAN = "BOOLEAN"


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
    avg_target: Optional[float]  # not for multiclass
    avg_score_etalon: Optional[float]


class FileMetrics(BaseModel):
    # etalon metadata
    task_type: Optional[ModelTaskType]
    label: Optional[ModelLabelType]
    count: Optional[int]
    valid_count: Optional[int]
    valid_rate: Optional[float]
    avg_target: Optional[float]
    metrics_binary_etalon: Optional[BinaryTask]
    metrics_regression_etalon: Optional[RegressionTask]
    metrics_multiclass_etalon: Optional[MulticlassTask]
    cuts: Optional[List[float]]
    interval: Optional[List[FileMetricsInterval]]


class NumericInterval(BaseModel):
    minValue: int
    maxValue: int


class FileColumnMetadata(BaseModel):
    index: int
    name: str
    dataType: DataType
    meaningType: FileColumnMeaningType
    minMaxValues: Optional[NumericInterval] = None
    originalName: Optional[str]


class FileMetadata(BaseModel):
    name: str
    description: Optional[str]
    columns: List[FileColumnMetadata]
    searchKeys: List[List[str]]
    excludeFeaturesSources: Optional[List[str]]
    hierarchicalGroupKeys: Optional[List[str]]
    hierarchicalSubgroupKeys: Optional[List[str]]
    taskType: Optional[ModelTaskType]
    rowsCount: Optional[int]
    checksumMD5: Optional[str]
    digest: Optional[str]


class FeaturesMetadataV2(BaseModel):
    name: str
    type: str
    source: str
    hit_rate: float
    shap_value: float
    commercial_schema: Optional[str]
    data_provider: Optional[str]
    data_providers: Optional[List[str]]
    data_provider_link: Optional[str]
    data_provider_links: Optional[List[str]]
    data_source: Optional[str]
    data_sources: Optional[List[str]]
    data_source_link: Optional[str]
    data_source_links: Optional[List[str]]
    doc_link: Optional[str]


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
    ads_definition_id: Optional[str]


class GeneratedFeatureMetadata(BaseModel):
    formula: str  # on hashed names
    display_index: int
    base_columns: List[BaseColumnMetadata]


class ProviderTaskMetadataV2(BaseModel):
    features: List[FeaturesMetadataV2]
    hit_rate_metrics: Optional[HitRateMetrics]
    eval_set_metrics: Optional[List[ModelEvalSet]]
    zero_hit_rate_search_keys: Optional[List[str]]
    features_used_for_embeddings: Optional[List[str]]
    shuffle_kfold: Optional[bool]
    generated_features: Optional[List[GeneratedFeatureMetadata]]


class FeaturesFilter(BaseModel):
    minImportance: Optional[float]
    maxPSI: Optional[float]
    maxCount: Optional[int]
    selectedFeatures: Optional[List[str]]


class RuntimeParameters(BaseModel):
    properties: Dict[str, str] = dict()


class SearchCustomization(BaseModel):
    featuresFilter: Optional[FeaturesFilter]
    extractFeatures: Optional[bool]
    accurateModel: Optional[bool]
    importanceThreshold: Optional[float]
    maxFeatures: Optional[int]
    returnScores: Optional[bool]
    runtimeParameters: Optional[RuntimeParameters]
    metricsCalculation: Optional[bool]

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

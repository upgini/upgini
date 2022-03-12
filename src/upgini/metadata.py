from enum import Enum
from typing import List, Optional, Dict

from pydantic import BaseModel

SYSTEM_RECORD_ID = "system_record_id"
SYSTEM_FAKE_DATE = "system_fake_date"


class FileColumnMeaningType(Enum):
    MSISDN = "MSISDN"
    EMAIL = "EMAIL"
    IP_ADDRESS = "IP_ADDRESS"
    HEM = "HEM"
    DATE = "DATE"
    DATETIME = "DATETIME"
    SCORE = "SCORE"
    TARGET = "TARGET"
    FEATURE = "FEATURE"
    CUSTOM_KEY = "CUSTOM_KEY"
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


class DataType(Enum):
    INT = "INT"
    DECIMAL = "DECIMAL"
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
    hierarchicalGroupKeys: Optional[List[str]]
    hierarchicalSubgroupKeys: Optional[List[str]]
    taskType: Optional[ModelTaskType]
    rowsCount: Optional[int]


class FeaturesFilter(BaseModel):
    minImportance: Optional[float]
    maxPSI: Optional[float]
    maxCount: Optional[int]
    selectedFeatures: Optional[List[str]]


class RuntimeParameters(BaseModel):
    properties: Optional[Dict[str, str]]


class SearchCustomization(BaseModel):
    featuresFilter: Optional[FeaturesFilter]
    extractFeatures: Optional[bool]
    accurateModel: Optional[bool]
    importanceThreshold: Optional[float]
    maxFeatures: Optional[int]
    returnScores: Optional[bool]
    runtimeParameters: Optional[RuntimeParameters]

    def __repr__(self):
        return (
            f"Features filter: {self.featuresFilter}, "
            f"extract features: {self.extractFeatures}, "
            f"accurate model: {self.accurateModel}, "
            f"importance threshold: {self.importanceThreshold}, "
            f"max features: {self.maxFeatures}, "
            f"return scores: {self.returnScores}, "
            f"runtimeParameters: {self.runtimeParameters}"
        )

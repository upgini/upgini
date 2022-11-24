from typing import List

from .dataset import Dataset
from .features_enricher import FeaturesEnricher  # noqa: F401
from .metadata import (  # noqa: F401
    FileColumnMeaningType,
    FileMetrics,
    ModelTaskType,
    SearchKey,
)
from .search_task import SearchTask


def search_history() -> List[SearchTask]:
    # TODO
    return []


def datasets_history() -> List[Dataset]:
    # TODO
    return []

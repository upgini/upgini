import os

from upgini.features_enricher import FeaturesEnricher  # noqa: F401
from upgini.metadata import SearchKey, CVType, RuntimeParameters, ModelTaskType  # noqa: F401
# from .lazy_import import LazyImport

os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

# FeaturesEnricher = LazyImport("upgini.features_enricher", "FeaturesEnricher")
# SearchKey = LazyImport("upgini.metadata", "SearchKey")
# RuntimeParameters = LazyImport("upgini.metadata", "RuntimeParameters")
# CVType = LazyImport("upgini.metadata", "CVType")
# ModelTaskType = LazyImport("upgini.metadata", "ModelTaskType")

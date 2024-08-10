import os

from .lazy_import import LazyImport

os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

FeaturesEnricher = LazyImport("upgini.features_enricher", "FeaturesEnricher")
SearchKey = LazyImport("upgini.metadata", "SearchKey")
RuntimeParameters = LazyImport("upgini.metadata", "RuntimeParameters")
CVType = LazyImport("upgini.metadata", "CVType")
ModelTaskType = LazyImport("upgini.metadata", "ModelTaskType")

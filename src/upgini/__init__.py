from .lazy_import import LazyImport

FeaturesEnricher = LazyImport('upgini.features_enricher', 'FeaturesEnricher')
SearchKey = LazyImport('upgini.metadata', 'SearchKey')
RuntimeParameters = LazyImport('upgini.metadata', 'RuntimeParameters')
CVType = LazyImport('upgini.metadata', 'CVType')
ModelTaskType = LazyImport('upgini.metadata', 'ModelTaskType')

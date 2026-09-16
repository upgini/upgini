from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReportMetadata:
    search_id: str
    generated_at: str
    pylib_version: str
    search_keys: list[str] = field(default_factory=list)
    search_duration: Optional[str] = None
    reference_rows: Optional[int] = None
    samples: list[str] = field(default_factory=list)
    logo_url: Optional[str] = None


@dataclass
class QualitySample:
    evaluation_scope: str
    metric: str
    baseline: Optional[float] = None
    baseline_std: Optional[float] = None
    enriched: Optional[float] = None
    enriched_std: Optional[float] = None
    uplift: Optional[float] = None
    relative_uplift: Optional[str] = None

    @property
    def std(self) -> Optional[float]:
        return self.enriched_std if self.enriched_std is not None else self.baseline_std


@dataclass
class SampleStats:
    sample: str
    rows: Optional[int] = None
    mean_target: Optional[float] = None
    date_range: Optional[str] = None
    labeled: Optional[int] = None
    unlabeled: Optional[int] = None
    positive: Optional[int] = None


@dataclass
class FeatureRow:
    name: str
    shap: Optional[float] = None
    psi: Optional[float] = None
    drift: Optional[float] = None
    coverage: Optional[float] = None
    provider: str = ""
    source: str = ""
    stability_status: Optional[str] = None


@dataclass
class SourceRow:
    provider: str
    source: str
    shap_sum: Optional[float] = None
    feature_count: Optional[int] = None


@dataclass
class SearchResultsSummary:
    relevant_features: Optional[int] = None
    model_features: Optional[int] = None
    data_sources: Optional[int] = None
    stable_features_share: Optional[float] = None


@dataclass
class ModelFeatureShap:
    rank: int
    feature: str
    provider: str = ""
    mean_abs_shap: Optional[float] = None


@dataclass
class ReportData:
    metadata: ReportMetadata
    quality_by_sample: list[QualitySample] = field(default_factory=list)
    summary: SearchResultsSummary = field(default_factory=SearchResultsSummary)
    sample_stats: list[SampleStats] = field(default_factory=list)
    is_binary: bool = False
    features: list[FeatureRow] = field(default_factory=list)
    sources: list[SourceRow] = field(default_factory=list)
    autofe: list[dict[str, str]] = field(default_factory=list)
    model_feature_shap: list[ModelFeatureShap] = field(default_factory=list)

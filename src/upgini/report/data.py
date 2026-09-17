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
    contributed_sources: Optional[int] = None
    stable_features: Optional[int] = None


@dataclass
class ModelFeatureShap:
    rank: int
    feature: str
    provider: str = ""
    mean_abs_shap: Optional[float] = None


@dataclass
class MonthlySamplePoint:
    month: str
    sample: str
    total: int
    labeled: int
    unlabeled: int
    positive: int
    mean: Optional[float] = None


@dataclass
class QualityMonthly:
    evaluation_scope: str
    metric: str
    baseline: list[Optional[float]] = field(default_factory=list)
    enriched: list[Optional[float]] = field(default_factory=list)


@dataclass
class ScoreMonthlyStats:
    sample: str
    mean_score: list[Optional[float]] = field(default_factory=list)
    mean_target: list[Optional[float]] = field(default_factory=list)
    labeled: list[Optional[int]] = field(default_factory=list)
    unlabeled: list[Optional[int]] = field(default_factory=list)


@dataclass
class ScoreHistogram:
    sample: str
    target_0: list[float] = field(default_factory=list)
    target_1: list[float] = field(default_factory=list)
    n_target_0: int = 0
    n_target_1: int = 0


@dataclass
class ScorePsi:
    sample: str
    psi: list[Optional[float]] = field(default_factory=list)
    rows: list[Optional[int]] = field(default_factory=list)
    coverage: list[Optional[float]] = field(default_factory=list)


@dataclass
class ReportCharts:
    timeline_months: list[str] = field(default_factory=list)
    period_options: list[int] = field(default_factory=list)
    default_period: int = 0
    sample_monthly: list[MonthlySamplePoint] = field(default_factory=list)
    monthly_count_max: int = 0
    quality_monthly: list[QualityMonthly] = field(default_factory=list)
    score_monthly: list[ScoreMonthlyStats] = field(default_factory=list)
    score_count_axis_max: int = 0
    histogram_bin_edges: list[float] = field(default_factory=list)
    histograms: list[ScoreHistogram] = field(default_factory=list)
    histogram_density_max: float = 0.0
    score_psi: list[ScorePsi] = field(default_factory=list)
    psi_axis_max: float = 0.32
    psi_warning: float = 0.1
    psi_critical: float = 0.25


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
    charts: ReportCharts = field(default_factory=ReportCharts)

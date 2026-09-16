from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from upgini.__about__ import __version__
from upgini.report.data import QualitySample, ReportData, ReportMetadata, SampleStats, SearchResultsSummary
from upgini.resource_bundle import ResourceBundle

UPGINI_REPORT_BRANDING_URL = "UPGINI_REPORT_BRANDING_URL"


def format_search_duration(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} h {minutes} min {secs} sec"
    if minutes:
        return f"{minutes} min {secs} sec"
    return f"{secs} sec"


def assemble_report_data(
    *,
    search_id: str,
    search_keys: list[str],
    generated_at: Optional[datetime] = None,
    pylib_version: Optional[str] = None,
    search_duration_seconds: Optional[float] = None,
    reference_rows: Optional[int] = None,
    samples: Optional[list[str]] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    metric_name: Optional[str] = None,
    model_features: Optional[int] = None,
    is_binary: bool = False,
    bundle: ResourceBundle,
) -> ReportData:
    generated_at = generated_at or datetime.now(timezone.utc)
    sample_names = list(samples or [])
    return ReportData(
        metadata=ReportMetadata(
            search_id=search_id,
            generated_at=generated_at.strftime("%Y-%m-%d %H:%M UTC"),
            pylib_version=pylib_version or __version__,
            search_keys=list(search_keys or []),
            search_duration=format_search_duration(search_duration_seconds),
            reference_rows=reference_rows,
            samples=sample_names,
            logo_url=_branding_url(),
        ),
        quality_by_sample=_quality_samples(metrics_df, metric_name, bundle),
        summary=SearchResultsSummary(model_features=model_features),
        sample_stats=_sample_stats(sample_names, metrics_df, bundle),
        is_binary=is_binary,
    )


def _branding_url() -> Optional[str]:
    env_url = os.getenv(UPGINI_REPORT_BRANDING_URL)
    if env_url and env_url.strip():
        return env_url.strip()
    return None


def _quality_samples(
    metrics_df: Optional[pd.DataFrame], metric_name: Optional[str], bundle: ResourceBundle
) -> list[QualitySample]:
    if metrics_df is None or metrics_df.empty:
        return []
    segment_col = bundle.get("quality_metrics_segment_header")
    uplift_col = bundle.get("quality_metrics_uplift_header")
    uplift_pct_col = bundle.get("quality_metrics_uplift_perc_header")
    baseline_col = bundle.get("quality_metrics_baseline_header").format(metric_name) if metric_name else None
    enriched_col = bundle.get("quality_metrics_enriched_header").format(metric_name) if metric_name else None
    baseline_std_col = bundle.get("quality_metrics_baseline_std_header").format(metric_name) if metric_name else None
    enriched_std_col = bundle.get("quality_metrics_enriched_std_header").format(metric_name) if metric_name else None

    samples: list[QualitySample] = []
    for _, row in metrics_df.iterrows():
        samples.append(
            QualitySample(
                evaluation_scope=(_as_str(row[segment_col]) or "") if segment_col in metrics_df.columns else "",
                metric=metric_name or "",
                baseline=_as_float(row[baseline_col]) if baseline_col in metrics_df.columns else None,
                baseline_std=_as_float(row[baseline_std_col]) if baseline_std_col in metrics_df.columns else None,
                enriched=_as_float(row[enriched_col]) if enriched_col in metrics_df.columns else None,
                enriched_std=_as_float(row[enriched_std_col]) if enriched_std_col in metrics_df.columns else None,
                uplift=_as_float(row[uplift_col]) if uplift_col in metrics_df.columns else None,
                relative_uplift=_as_str(row[uplift_pct_col]) if uplift_pct_col in metrics_df.columns else None,
            )
        )
    return samples


def _sample_stats(
    sample_names: list[str], metrics_df: Optional[pd.DataFrame], bundle: ResourceBundle
) -> list[SampleStats]:
    segment_col = bundle.get("quality_metrics_segment_header")
    rows_col = bundle.get("quality_metrics_rows_header")
    mean_col = bundle.get("quality_metrics_mean_target_header")
    by_name: dict[str, SampleStats] = {}
    if metrics_df is not None and not metrics_df.empty and segment_col in metrics_df.columns:
        for _, row in metrics_df.iterrows():
            name = _as_str(row[segment_col])
            if not name:
                continue
            by_name[name] = SampleStats(
                sample=name,
                rows=_as_int(row[rows_col]) if rows_col in metrics_df.columns else None,
                mean_target=_as_float(row[mean_col]) if mean_col in metrics_df.columns else None,
            )
    names = sample_names or list(by_name)
    stats = [by_name.get(name) or SampleStats(sample=name) for name in names]
    seen = set(names)
    stats.extend(sample for name, sample in by_name.items() if name not in seen)
    return stats


def _as_str(value) -> Optional[str]:
    return None if pd.isna(value) else str(value)


def _as_float(value) -> Optional[float]:
    return None if pd.isna(value) else float(value)


def _as_int(value) -> Optional[int]:
    return None if pd.isna(value) else int(value)

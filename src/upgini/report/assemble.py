from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from upgini.__about__ import __version__
from upgini.report.data import QualitySample, ReportData, ReportMetadata
from upgini.resource_bundle import ResourceBundle


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
    bundle: ResourceBundle,
) -> ReportData:
    generated_at = generated_at or datetime.now(timezone.utc)
    return ReportData(
        metadata=ReportMetadata(
            search_id=search_id,
            generated_at=generated_at.strftime("%Y-%m-%d %H:%M UTC"),
            pylib_version=pylib_version or __version__,
            search_keys=list(search_keys or []),
            search_duration=format_search_duration(search_duration_seconds),
            reference_rows=reference_rows,
            samples=list(samples or []),
        ),
        quality_by_sample=_quality_samples(metrics_df, metric_name, bundle),
    )


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

    samples: list[QualitySample] = []
    for _, row in metrics_df.iterrows():
        samples.append(
            QualitySample(
                evaluation_scope=(_as_str(row[segment_col]) or "") if segment_col in metrics_df.columns else "",
                metric=metric_name or "",
                baseline=_as_str(row[baseline_col]) if baseline_col in metrics_df.columns else None,
                enriched=_as_str(row[enriched_col]) if enriched_col in metrics_df.columns else None,
                uplift=_as_float(row[uplift_col]) if uplift_col in metrics_df.columns else None,
                relative_uplift=_as_str(row[uplift_pct_col]) if uplift_pct_col in metrics_df.columns else None,
            )
        )
    return samples


def _as_str(value) -> Optional[str]:
    return None if pd.isna(value) else str(value)


def _as_float(value) -> Optional[float]:
    return None if pd.isna(value) else float(value)

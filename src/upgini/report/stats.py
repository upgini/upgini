from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from upgini.report.data import (
    MonthlySamplePoint,
    QualityMonthly,
    ReportCharts,
    SampleStats,
    ScoreHistogram,
    ScoreMonthlyStats,
    ScorePsi,
)
from upgini.utils.psi import _get_bin_edges, calculate_numeric_psi

TARGET = "target"
DATE = "date"
SCORE = "score"
BASELINE = "baseline"

PSI_WARNING = 0.1
PSI_CRITICAL = 0.25
HISTOGRAM_BINS = 10
MEAN_AXIS_PADDING = 0.15
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def make_report_frame(target, *, date=None, score=None, baseline=None) -> pd.DataFrame:
    data = {TARGET: _series(target)}
    if date is not None:
        data[DATE] = _series(date)
    if score is not None:
        data[SCORE] = _series(score)
    if baseline is not None:
        data[BASELINE] = _series(baseline)
    return pd.DataFrame(data)


def _series(values) -> pd.Series:
    return pd.Series(values).reset_index(drop=True)


def compute_report_charts(
    *,
    sample_names: list[str],
    dataset_samples: Optional[dict[str, pd.DataFrame]] = None,
    scored_samples: Optional[dict[str, pd.DataFrame]] = None,
    metric_name: Optional[str] = None,
    is_binary: bool = False,
) -> tuple[list[SampleStats], ReportCharts]:
    dataset_samples = dataset_samples or {}
    scored_samples = scored_samples or {}
    names = sample_names or list(dict.fromkeys([*dataset_samples, *scored_samples]))
    timeline = _timeline_months(names, dataset_samples, scored_samples)
    period_options, default_period = _period_options(len(timeline))

    sample_stats: list[SampleStats] = []
    monthly_points: list[MonthlySamplePoint] = []
    for name in names:
        frame = dataset_samples.get(name)
        if frame is None:
            continue
        stats, points = _sample_stats_and_monthly(name, frame, is_binary, timeline)
        sample_stats.append(stats)
        monthly_points.extend(points)

    score_monthly = _score_monthly(names, dataset_samples, scored_samples, timeline)
    charts = ReportCharts(
        timeline_months=timeline,
        period_options=period_options,
        default_period=default_period,
        sample_monthly=monthly_points,
        monthly_count_max=max((point.total for point in monthly_points), default=0),
        quality_monthly=_quality_monthly(names, scored_samples, metric_name, is_binary, timeline),
        score_monthly=score_monthly,
        score_count_axis_max=_score_count_axis_max(score_monthly),
        histogram_bin_edges=_histogram_edges(),
        histograms=_histograms(names, scored_samples, is_binary),
        score_psi=_score_psi(names, scored_samples, timeline),
        psi_warning=PSI_WARNING,
        psi_critical=PSI_CRITICAL,
    )
    densities = [value for hist in charts.histograms for value in hist.target_0 + hist.target_1]
    charts.histogram_density_max = max(densities, default=0.0)
    psis = [value for row in charts.score_psi for value in row.psi if value is not None]
    charts.psi_axis_max = max([PSI_CRITICAL, *psis], default=PSI_CRITICAL)
    if charts.psi_axis_max == PSI_CRITICAL and psis:
        charts.psi_axis_max = max(PSI_CRITICAL, max(psis) * 1.15)
    return sample_stats, charts


def _timeline_months(
    names: list[str], dataset_samples: dict[str, pd.DataFrame], scored_samples: dict[str, pd.DataFrame]
) -> list[str]:
    periods: list[pd.Period] = []
    for name in names:
        for source in (dataset_samples, scored_samples):
            frame = source.get(name)
            if frame is None or DATE not in frame.columns:
                continue
            periods.extend(_periods(frame[DATE]).dropna().tolist())
    if not periods:
        return []
    start, end = min(periods), max(periods)
    months = []
    current = start
    while current <= end:
        months.append(_month_label(current))
        current += 1
    return months


def _period_options(count: int) -> tuple[list[int], int]:
    if count <= 0:
        return [], 0
    options = sorted({p for p in (12, 6, 3, count) if 0 < p <= count}, reverse=True)
    return options, min(12, count)


def _sample_stats_and_monthly(
    name: str, frame: pd.DataFrame, is_binary: bool, timeline: list[str]
) -> tuple[SampleStats, list[MonthlySamplePoint]]:
    target = _target(frame)
    labeled_mask = target.notna()
    unlabeled = int((~labeled_mask).sum())
    labeled = int(labeled_mask.sum())
    positive = int((target[labeled_mask] == 1).sum()) if is_binary else None
    mean_target = float(target[labeled_mask].mean()) if labeled else None
    periods = _periods(frame[DATE]) if DATE in frame.columns else pd.Series(pd.NaT, index=frame.index)
    date_range = _date_range(periods)
    points: list[MonthlySamplePoint] = []
    if DATE in frame.columns:
        grouped = (
            pd.DataFrame({"target": target, "month": _month_keys(frame[DATE])})
            .dropna(subset=["month"])
            .groupby("month", sort=False)
        )
        by_month = {month: group["target"] for month, group in grouped}
        for month in timeline:
            month_target = by_month.get(month)
            if month_target is None:
                continue
            month_labeled = month_target.notna()
            labeled_n = int(month_labeled.sum())
            points.append(
                MonthlySamplePoint(
                    month=month,
                    sample=name,
                    total=int(len(month_target)),
                    labeled=labeled_n,
                    unlabeled=int((~month_labeled).sum()),
                    positive=int((month_target[month_labeled] == 1).sum()) if is_binary else 0,
                    mean=float(month_target[month_labeled].mean()) if labeled_n else None,
                )
            )
    return (
        SampleStats(
            sample=name,
            rows=int(len(frame)),
            mean_target=mean_target,
            date_range=date_range,
            labeled=labeled,
            unlabeled=unlabeled,
            positive=positive,
        ),
        points,
    )


def _quality_monthly(
    names: list[str],
    scored_samples: dict[str, pd.DataFrame],
    metric_name: Optional[str],
    is_binary: bool,
    timeline: list[str],
) -> list[QualityMonthly]:
    if not is_binary or not timeline or not scored_samples:
        return []
    rows: list[QualityMonthly] = []
    for name in names:
        frame = scored_samples.get(name)
        if frame is None or SCORE not in frame.columns or DATE not in frame.columns:
            continue
        rows.append(
            QualityMonthly(
                evaluation_scope=name,
                metric=metric_name or "score",
                baseline=_monthly_metric(frame, BASELINE, metric_name, timeline),
                enriched=_monthly_metric(frame, SCORE, metric_name, timeline),
            )
        )
    return rows


def _monthly_metric(
    frame: pd.DataFrame, score_col: str, metric_name: Optional[str], timeline: list[str]
) -> list[Optional[float]]:
    if score_col not in frame.columns:
        return [None] * len(timeline)
    data = pd.DataFrame(
        {
            "target": _target(frame),
            "score": pd.to_numeric(frame[score_col], errors="coerce"),
            "month": _month_keys(frame[DATE]),
        }
    )
    values: list[Optional[float]] = []
    for month in timeline:
        month_df = data[data["month"] == month]
        values.append(_binary_metric(month_df["target"], month_df["score"], metric_name))
    return values


def _binary_metric(target: pd.Series, scores: pd.Series, metric_name: Optional[str]) -> Optional[float]:
    mask = target.notna() & scores.notna()
    y = target[mask]
    s = scores[mask]
    if len(y) < 2 or y.nunique() < 2:
        return None
    try:
        auc = float(roc_auc_score(y, s))
    except ValueError:
        return None
    if metric_name and metric_name.upper() == "GINI":
        return 2 * auc - 1
    return auc


def _score_monthly(
    names: list[str],
    dataset_samples: dict[str, pd.DataFrame],
    scored_samples: dict[str, pd.DataFrame],
    timeline: list[str],
) -> list[ScoreMonthlyStats]:
    if not timeline:
        return []
    rows: list[ScoreMonthlyStats] = []
    for name in names:
        counts_frame = dataset_samples.get(name)
        score_frame = scored_samples.get(name)
        if counts_frame is None and score_frame is None:
            continue
        rows.append(
            ScoreMonthlyStats(
                sample=name,
                mean_score=_monthly_mean(score_frame, SCORE, timeline)
                if score_frame is not None
                else [None] * len(timeline),
                mean_target=_monthly_mean(counts_frame if counts_frame is not None else score_frame, TARGET, timeline),
                labeled=_monthly_counts(
                    counts_frame if counts_frame is not None else score_frame, timeline, labeled=True
                ),
                unlabeled=_monthly_counts(
                    counts_frame if counts_frame is not None else score_frame, timeline, labeled=False
                ),
            )
        )
    return rows


def _score_count_axis_max(score_monthly: list[ScoreMonthlyStats]) -> int:
    totals = [
        (labeled or 0) + (unlabeled or 0)
        for stats in score_monthly
        for labeled, unlabeled in zip(stats.labeled, stats.unlabeled)
    ]
    return max(totals, default=0)


def _monthly_mean(frame: Optional[pd.DataFrame], column: str, timeline: list[str]) -> list[Optional[float]]:
    if frame is None or column not in frame.columns or DATE not in frame.columns:
        return [None] * len(timeline)
    values = pd.to_numeric(frame[column], errors="coerce")
    months = _month_keys(frame[DATE])
    result = []
    for month in timeline:
        month_values = values[months == month]
        result.append(float(month_values.mean()) if month_values.notna().any() else None)
    return result


def _monthly_counts(frame: Optional[pd.DataFrame], timeline: list[str], *, labeled: bool) -> list[Optional[int]]:
    if frame is None or DATE not in frame.columns:
        return [None] * len(timeline)
    target = _target(frame)
    months = _month_keys(frame[DATE])
    present = set(months.dropna())
    result = []
    for month in timeline:
        month_target = target[months == month]
        if month not in present:
            result.append(None)
            continue
        mask = month_target.notna() if labeled else month_target.isna()
        result.append(int(mask.sum()))
    return result


def _histograms(names: list[str], scored_samples: dict[str, pd.DataFrame], is_binary: bool) -> list[ScoreHistogram]:
    if not is_binary:
        return []
    edges = _histogram_edges()
    rows = []
    for name in names:
        frame = scored_samples.get(name)
        if frame is None or SCORE not in frame.columns:
            continue
        target = _target(frame)
        scores = pd.to_numeric(frame[SCORE], errors="coerce").clip(0, 1)
        labeled = target.notna() & scores.notna()
        zero = scores[labeled & (target == 0)]
        one = scores[labeled & (target == 1)]
        rows.append(
            ScoreHistogram(
                sample=name,
                target_0=_bin_mass(zero, edges),
                target_1=_bin_mass(one, edges),
                n_target_0=int(len(zero)),
                n_target_1=int(len(one)),
            )
        )
    return rows


def _histogram_edges() -> list[float]:
    return [round(edge, 1) for edge in np.linspace(0, 1, HISTOGRAM_BINS + 1).tolist()]


def _bin_mass(values: pd.Series, edges: list[float]) -> list[float]:
    if values.empty:
        return [0.0] * (len(edges) - 1)
    counts, _ = np.histogram(values.to_numpy(), bins=edges)
    total = counts.sum()
    if total == 0:
        return [0.0] * (len(edges) - 1)
    return (counts / total).tolist()


def _score_psi(names: list[str], scored_samples: dict[str, pd.DataFrame], timeline: list[str]) -> list[ScorePsi]:
    if not timeline or not scored_samples:
        return []
    train = scored_samples.get(names[0]) if names else None
    if train is None or SCORE not in train.columns or DATE not in train.columns:
        return []
    train_scores = pd.to_numeric(train[SCORE], errors="coerce")
    train_months = _month_keys(train[DATE])
    reference = None
    for month in timeline:
        month_scores = train_scores[train_months == month].dropna()
        if not month_scores.empty:
            reference = month_scores
            break
    if reference is None:
        return []
    bins = _get_bin_edges(reference, HISTOGRAM_BINS)
    rows = []
    for name in names:
        frame = scored_samples.get(name)
        if frame is None or SCORE not in frame.columns or DATE not in frame.columns:
            continue
        scores = pd.to_numeric(frame[SCORE], errors="coerce")
        months = _month_keys(frame[DATE])
        psi: list[Optional[float]] = []
        rows_n: list[Optional[int]] = []
        coverage: list[Optional[float]] = []
        for month in timeline:
            month_mask = months == month
            month_scores = scores[month_mask]
            if not month_mask.any():
                psi.append(None)
                rows_n.append(None)
                coverage.append(None)
                continue
            rows_n.append(int(month_mask.sum()))
            coverage.append(float(month_scores.notna().mean() * 100))
            psi.append(calculate_numeric_psi(reference, month_scores, bins=bins))
        rows.append(ScorePsi(sample=name, psi=psi, rows=rows_n, coverage=coverage))
    return rows


def _target(frame: pd.DataFrame) -> pd.Series:
    if TARGET not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[TARGET], errors="coerce")


def _periods(dates: pd.Series) -> pd.Series:
    parsed = _parse_dates(dates)
    return parsed.dt.to_period("M")


def _parse_dates(dates: pd.Series) -> pd.Series:
    if isinstance(dates.dtype, pd.PeriodDtype):
        return dates.dt.to_timestamp()
    if pd.api.types.is_datetime64_any_dtype(dates):
        parsed = pd.to_datetime(dates)
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
        return parsed
    numeric = pd.to_numeric(dates, errors="coerce")
    if numeric.notna().any():
        median = float(numeric.dropna().median())
        if median > 1e12:
            return pd.to_datetime(numeric, unit="ms", errors="coerce")
        if median > 1e9:
            return pd.to_datetime(numeric, unit="s", errors="coerce")
    return pd.to_datetime(dates, errors="coerce")


def _month_keys(dates: pd.Series) -> pd.Series:
    periods = _periods(dates)
    keys = pd.Series(pd.NA, index=dates.index, dtype=object)
    valid = periods.notna()
    if valid.any():
        keys.loc[valid] = [_month_label(period) for period in periods[valid]]
    return keys


def _month_label(period) -> str:
    if not isinstance(period, pd.Period):
        period = pd.Period(period, freq="M")
    return f"{_MONTHS[period.month - 1]} {period.year}"


def _date_range(periods: pd.Series) -> Optional[str]:
    valid = periods.dropna()
    if valid.empty:
        return None
    return f"{_month_label(valid.min())} — {_month_label(valid.max())}"

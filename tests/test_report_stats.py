import pandas as pd
from sklearn.metrics import roc_auc_score

from upgini.report.assemble import assemble_report_data
from upgini.report.html import generate_html_report
from upgini.report.stats import (
    HISTOGRAM_BINS,
    PSI_CRITICAL,
    PSI_WARNING,
    compute_report_charts,
    make_report_frame,
)
from upgini.resource_bundle import bundle

from .test_score_report import _parse_report_data


def _ms(year: int, month: int, day: int = 1) -> int:
    return int(pd.Timestamp(year=year, month=month, day=day).timestamp() * 1000)


def test_sample_stats_and_inclusive_timeline():
    train = make_report_frame(
        [0, 1, None, 0],
        date=["2024-01-15", "2024-01-20", "2024-03-01", "2024-03-10"],
    )
    eval1 = make_report_frame(
        [1, 1, None],
        date=["2024-03-05", "2024-03-18", "2024-03-22"],
    )

    sample_stats, charts = compute_report_charts(
        sample_names=["Train", "Eval 1"],
        dataset_samples={"Train": train, "Eval 1": eval1},
        is_binary=True,
    )

    assert charts.timeline_months == ["Jan 2024", "Feb 2024", "Mar 2024"]
    assert charts.period_options == [3]
    assert charts.default_period == 3

    train_stats = sample_stats[0]
    assert train_stats.sample == "Train"
    assert train_stats.rows == 4
    assert train_stats.labeled == 3
    assert train_stats.unlabeled == 1
    assert train_stats.positive == 1
    assert train_stats.mean_target == 1 / 3
    assert train_stats.date_range == "Jan 2024 — Mar 2024"

    eval_stats = sample_stats[1]
    assert eval_stats.rows == 3
    assert eval_stats.labeled == 2
    assert eval_stats.unlabeled == 1
    assert eval_stats.positive == 2

    jan = next(point for point in charts.sample_monthly if point.sample == "Train" and point.month == "Jan 2024")
    assert jan.total == 2
    assert jan.labeled == 2
    assert jan.unlabeled == 0
    assert jan.positive == 1
    assert jan.mean == 0.5
    assert not any(point.month == "Feb 2024" for point in charts.sample_monthly)


def test_quality_and_score_series_from_sampled_scores():
    dataset_train = make_report_frame(
        [0, 1, 0, 1, None],
        date=["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02", "2024-02-03"],
    )
    scored_train = make_report_frame(
        [0, 1, 0, 1],
        date=["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02"],
        score=[0.1, 0.9, 0.2, 0.8],
        baseline=[0.4, 0.6, 0.45, 0.55],
    )
    scored_eval = make_report_frame(
        [0, 1],
        date=["2024-02-01", "2024-02-02"],
        score=[0.15, 0.85],
        baseline=[0.35, 0.65],
    )

    _, charts = compute_report_charts(
        sample_names=["Train", "Eval 1"],
        dataset_samples={"Train": dataset_train},
        scored_samples={"Train": scored_train, "Eval 1": scored_eval},
        metric_name="GINI",
        is_binary=True,
    )

    expected_jan = 2 * roc_auc_score([0, 1], [0.1, 0.9]) - 1
    expected_feb = 2 * roc_auc_score([0, 1], [0.2, 0.8]) - 1
    train_quality = next(row for row in charts.quality_monthly if row.evaluation_scope == "Train")
    assert train_quality.metric == "GINI"
    assert train_quality.enriched[0] == expected_jan
    assert train_quality.enriched[1] == expected_feb
    assert train_quality.baseline[0] == 2 * roc_auc_score([0, 1], [0.4, 0.6]) - 1

    train_score = next(row for row in charts.score_monthly if row.sample == "Train")
    assert train_score.mean_score == [0.5, 0.5]
    assert train_score.mean_target == [0.5, 0.5]
    assert train_score.labeled == [2, 2]
    assert train_score.unlabeled == [0, 1]

    eval_score = next(row for row in charts.score_monthly if row.sample == "Eval 1")
    assert eval_score.mean_score == [None, 0.5]
    assert eval_score.mean_target == [None, 0.5]
    assert eval_score.labeled == [None, 2]
    assert eval_score.unlabeled == [None, 0]


def test_score_histogram_equal_bins_and_psi_reference():
    train = make_report_frame(
        [0, 0, 0, 1, 1, 1],
        date=["2024-01-01"] * 6,
        score=[0.05, 0.08, 0.09, 0.91, 0.94, 0.97],
    )
    eval1 = make_report_frame(
        [0, 1, 0, 1],
        date=["2024-02-01"] * 4,
        score=[0.92, 0.95, 0.93, 0.96],
    )

    _, charts = compute_report_charts(
        sample_names=["Train", "Eval 1"],
        scored_samples={"Train": train, "Eval 1": eval1},
        is_binary=True,
    )

    assert charts.histogram_bin_edges == [round(i / HISTOGRAM_BINS, 1) for i in range(HISTOGRAM_BINS + 1)]
    train_hist = next(row for row in charts.histograms if row.sample == "Train")
    assert train_hist.n_target_0 == 3
    assert train_hist.n_target_1 == 3
    assert train_hist.target_0[0] == 1.0
    assert sum(train_hist.target_0[1:]) == 0
    assert train_hist.target_1[-1] == 1.0

    train_psi = next(row for row in charts.score_psi if row.sample == "Train")
    eval_psi = next(row for row in charts.score_psi if row.sample == "Eval 1")
    assert train_psi.psi[0] == 0
    assert train_psi.rows[0] == 6
    assert train_psi.coverage[0] == 100.0
    assert train_psi.psi[1] is None
    assert eval_psi.psi[0] is None
    assert eval_psi.psi[1] > PSI_WARNING
    assert charts.psi_warning == PSI_WARNING
    assert charts.psi_critical == PSI_CRITICAL


def test_dates_from_epoch_milliseconds_are_parsed():
    frame = make_report_frame([0, 1], date=[_ms(2024, 1), _ms(2024, 2)])
    sample_stats, charts = compute_report_charts(
        sample_names=["Train"],
        dataset_samples={"Train": frame},
        is_binary=True,
    )
    assert charts.timeline_months == ["Jan 2024", "Feb 2024"]
    assert sample_stats[0].date_range == "Jan 2024 — Feb 2024"


def test_html_payload_includes_pylib_chart_series():
    dataset = {
        "Train": make_report_frame(
            [0, 1, None, 1],
            date=["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02"],
        )
    }
    scored = {
        "Train": make_report_frame(
            [0, 1, 1],
            date=["2024-01-01", "2024-01-02", "2024-02-02"],
            score=[0.2, 0.8, 0.7],
            baseline=[0.3, 0.6, 0.55],
        )
    }
    metrics_df = pd.DataFrame(
        {
            bundle.get("quality_metrics_segment_header"): ["Train"],
            bundle.get("quality_metrics_baseline_header").format("GINI"): [0.4],
            bundle.get("quality_metrics_enriched_header").format("GINI"): [0.512],
        }
    )
    data = assemble_report_data(
        search_id="search-charts",
        search_keys=["PHONE", "DATE"],
        samples=["Train"],
        metrics_df=metrics_df,
        metric_name="GINI",
        is_binary=True,
        dataset_samples=dataset,
        scored_samples=scored,
        bundle=bundle,
    )
    payload = _parse_report_data(generate_html_report(data))

    assert payload["timeline"]["months"] == ["Jan 2024", "Feb 2024"]
    assert payload["sampleStats"]["rows"][0]["values"]["train"] == "Jan 2024 — Feb 2024"
    assert payload["sampleStats"]["rows"][1]["values"]["train"] == "4"
    assert payload["sampleStats"]["monthlyPoints"][0]["month"] == "Jan 2024"
    assert payload["keyResult"]["metrics"]["gini"]["bySample"]["train"]["enrichedSeries"]
    assert payload["scoreAnalysis"]["bySample"]["train"]["meanScore"][0] == 0.5
    assert sum(payload["scoreDistribution"]["bySample"]["train"]["target0"]) == 1
    assert payload["scoreStability"]["thresholds"] == {"warning": 0.1, "critical": 0.25}
    assert len(payload["scoreStability"]["bySample"]["train"]["psi"]) == 2
    assert payload["scoreStability"]["bySample"]["train"]["rows"][0] == 2

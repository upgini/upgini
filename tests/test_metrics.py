import os

import pandas as pd
import pytest
from requests_mock.mocker import Mocker

from upgini import FeaturesEnricher, SearchKey

from .utils import (
    mock_default_requests,
    mock_get_features_meta,
    mock_get_metadata,
    mock_initial_search,
    mock_initial_summary,
    mock_raw_features,
)

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "test_data/enricher/",
)


def test_default_metric_binary(requests_mock: Mocker):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)
    search_task_id = mock_initial_search(requests_mock, url)
    ads_search_task_id = mock_initial_summary(requests_mock, url, search_task_id)
    mock_get_metadata(requests_mock, url, search_task_id)
    mock_get_features_meta(
        requests_mock,
        url,
        ads_search_task_id,
        ads_features=[{"name": "ads_feature1", "importance": 10.1, "matchedInPercent": 99.0, "valueType": "NUMERIC"}],
        etalon_features=[{"name": "feature1", "importance": 0.1, "matchedInPercent": 100.0, "valueType": "NUMERIC"}]
    )
    mock_raw_features(requests_mock, url, search_task_id, os.path.join(FIXTURE_DIR, "features.csv.gz"))

    df = pd.read_csv(os.path.join(FIXTURE_DIR, "input.csv"))
    X = df.loc[0:499, ["phone", "feature1"]]
    y = df.loc[0:499, "target"]
    eval_X_1 = df.loc[500:749, ["phone", "feature1"]]
    eval_y_1 = df.loc[500:749, "target"]
    eval_X_2 = df.loc[750:999, ["phone", "feature1"]]
    eval_y_2 = df.loc[750:999, "target"]
    eval_set = [(eval_X_1, eval_y_1), (eval_X_2, eval_y_2)]
    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE},
        endpoint=url,
    )

    with pytest.raises(Exception, match="Fit wasn't completed successfully"):
        enricher.calculate_metrics(X, y)

    enriched_X = enricher.fit_transform(X, y, eval_set)

    assert len(enriched_X) == len(X)

    assert enricher.enriched_eval_set is not None
    assert len(enricher.enriched_eval_set) == 500

    metrics_df = enricher.calculate_metrics(X, y, eval_set)
    print(metrics_df)
    assert metrics_df.loc["train", "baseline roc_auc"] == pytest.approx(0.498607, abs=0.000001)  # type: ignore
    assert metrics_df.loc["train", "enriched roc_auc"] == pytest.approx(0.499296, abs=0.000001)  # type: ignore
    assert metrics_df.loc["train", "uplift"] == pytest.approx(0.000688, abs=0.000001)  # type: ignore

    assert metrics_df.loc["eval 1", "baseline roc_auc"] == pytest.approx(0.502164, abs=0.000001)  # type: ignore
    assert metrics_df.loc["eval 1", "enriched roc_auc"] == pytest.approx(0.497739, abs=0.000001)  # type: ignore
    assert metrics_df.loc["eval 1", "uplift"] == pytest.approx(-0.004425, abs=0.000001)  # type: ignore

    assert metrics_df.loc["eval 2", "baseline roc_auc"] == pytest.approx(0.487584, abs=0.000001)  # type: ignore
    assert metrics_df.loc["eval 2", "enriched roc_auc"] == pytest.approx(0.495625, abs=0.000001)  # type: ignore
    assert metrics_df.loc["eval 2", "uplift"] == pytest.approx(0.008042, abs=0.000001)  # type: ignore

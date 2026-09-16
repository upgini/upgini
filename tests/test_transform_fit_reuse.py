"""Tests for transform reusing fit enrichment (identity + key-based ads lookup)."""

import numpy as np
import pandas as pd
from requests_mock.mocker import Mocker
from unittest.mock import MagicMock

from upgini.dataset import Dataset
from upgini.features_enricher import FeaturesEnricher
from upgini.http import ProgressStage, SearchProgress
from upgini.metadata import EVAL_SET_INDEX, TARGET, SearchKey, ENTITY_SYSTEM_RECORD_ID, AddInfo
from upgini.normalizer.normalize_utils import add_hash_suffix
from upgini.utils.datetime_utils import DateTimeConverter
from upgini.utils.phone_utils import PhoneSearchKeyConverter

from .utils import mock_default_requests


def test_transform_from_fit_reuses_enrichment_for_train_and_oot(requests_mock: Mocker):
    """transform(train) / transform(oot) after fit should reuse fit features, not validation search."""
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)

    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
    )

    train_X = pd.DataFrame({"phone": ["+10000000001", "+10000000002"], "f": [1.0, 2.0]})
    oot_X = pd.DataFrame({"phone": ["+10000000003", "+10000000004"], "f": [3.0, 4.0]}, index=[10, 11])

    enricher.X = train_X
    enricher.y = pd.Series([0, 1])
    enricher.eval_set = enricher._check_eval_set([oot_X], train_X)
    enricher.feature_names_ = ["ads_feature"]
    enricher.external_source_feature_names = ["ads_feature"]
    enricher.fit_columns_renaming = {"phone_abc": "phone", "f_def": "f"}
    enricher.fit_search_keys = {"phone_abc": SearchKey.PHONE}
    enricher.fit_generated_features = []
    enricher.fit_select_features = False
    enricher.country_added = False
    enricher.add_info = AddInfo()

    # Simulate df kept on fit: hashed column names + entity ids + eval_set_index
    df_fit = pd.DataFrame(
        {
            "phone_abc": ["+10000000001", "+10000000002", "+10000000003", "+10000000004"],
            "f_def": [1.0, 2.0, 3.0, 4.0],
            TARGET: [0.0, 1.0, np.nan, np.nan],
            EVAL_SET_INDEX: [0, 0, 1, 1],
            ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0, 201.0, 202.0],
        },
        index=[0, 1, 10, 11],
    )
    enricher.df_with_original_index = df_fit

    fit_features = pd.DataFrame(
        {
            ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0, 201.0, 202.0],
            "ads_feature": [10.0, 20.0, 30.0, 40.0],
        }
    )

    search_task = MagicMock()
    fm = MagicMock()
    fm.name = "ads_feature"
    fm.shap_value = 1.0
    fm.source = "ads"
    fm.from_online_api = False
    search_task.get_all_features_metadata_v2.return_value = [fm]
    search_task.get_all_initial_raw_features.return_value = fit_features
    col_phone = MagicMock(originalName="phone", name="phone_abc")
    col_f = MagicMock(originalName="f", name="f_def")
    search_task.get_file_metadata.return_value = MagicMock(columns=[col_phone, col_f], droppedColumns=[])
    enricher._search_task = search_task
    enricher.search_id = "fake_search"

    def fail_if_validation(*args, **kwargs):
        raise AssertionError("validation search should not be called when reusing fit enrichment")

    original_validation = Dataset.validation
    Dataset.validation = fail_if_validation
    try:
        enriched_train = enricher.transform(train_X, keep_input=True)
        assert enriched_train is not None
        assert len(enriched_train) == len(train_X)
        assert "ads_feature" in enriched_train.columns
        assert list(enriched_train["ads_feature"]) == [10.0, 20.0]

        enriched_oot = enricher.transform(oot_X, keep_input=True)
        assert enriched_oot is not None
        assert len(enriched_oot) == len(oot_X)
        assert "ads_feature" in enriched_oot.columns
        assert list(enriched_oot["ads_feature"]) == [30.0, 40.0]
    finally:
        Dataset.validation = original_validation


def _file_col(original_name: str, name: str | None = None):
    col = MagicMock()
    col.originalName = original_name
    col.name = name or original_name
    return col


def _converted_phones(values: list[str]) -> pd.Series:
    df = pd.DataFrame({"phone": values})
    return PhoneSearchKeyConverter("phone").convert(df)["phone"]


def _make_reuse_enricher(
    requests_mock: Mocker,
    *,
    search_keys: dict,
    train_X: pd.DataFrame,
    df_fit: pd.DataFrame,
    fit_features: pd.DataFrame,
    file_columns: list,
    generate_search_key_features: bool = False,
):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)
    enricher = FeaturesEnricher(
        search_keys=search_keys,
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
        generate_search_key_features=generate_search_key_features,
    )
    enricher.X = train_X
    enricher.y = pd.Series([0] * len(train_X))
    enricher.eval_set = []
    enricher.feature_names_ = ["ads_feature"]
    enricher.external_source_feature_names = ["ads_feature"]
    enricher.fit_columns_renaming = {add_hash_suffix(c): c for c in train_X.columns}
    enricher.fit_search_keys = {add_hash_suffix(c): t for c, t in search_keys.items()}
    enricher.fit_generated_features = []
    enricher.fit_select_features = False
    enricher.country_added = False
    enricher.add_info = AddInfo()
    enricher.autodetected_search_keys = {}
    enricher.df_with_original_index = df_fit
    enricher._fit_match_search_keys = dict(enricher.fit_search_keys)
    enricher.search_id = "fake_search"

    search_task = MagicMock()
    fm = MagicMock()
    fm.name = "ads_feature"
    fm.shap_value = 1.0
    fm.source = "ads"
    fm.from_online_api = False
    search_task.get_all_features_metadata_v2.return_value = [fm]
    search_task.get_all_initial_raw_features.return_value = fit_features
    search_task.get_features_for_transform.return_value = []
    search_task.get_features_for_embeddings.return_value = None
    search_task.get_file_metadata.return_value = MagicMock(
        columns=file_columns, droppedColumns=[], autodetectedSearchKeys={}
    )
    enricher._search_task = search_task
    return enricher, search_task


def _stub_validation(search_task: MagicMock, ads_value: float = 99.0) -> dict:
    captured: dict = {}

    def fake_validation(trace_id, dataset, *args, **kwargs):
        captured["df"] = dataset.data.copy()
        task = MagicMock()
        task.search_task_id = "val_task"
        task.get_progress.return_value = SearchProgress(97.0, ProgressStage.DOWNLOADING)
        task.poll_result.return_value = None
        entity_ids = captured["df"][ENTITY_SYSTEM_RECORD_ID].drop_duplicates()
        task.get_all_validation_raw_features.return_value = pd.DataFrame(
            {
                ENTITY_SYSTEM_RECORD_ID: entity_ids.to_numpy(),
                "ads_feature": [ads_value] * len(entity_ids),
            }
        )
        return task

    search_task.validation.side_effect = fake_validation
    return captured


def _persist_and_drop_snapshot(enricher: FeaturesEnricher) -> None:
    enricher._FeaturesEnricher__persist_transform_match_hashes()
    enricher.df_with_original_index = None


def test_transform_copy_reuses_fit_enrichment(requests_mock: Mocker):
    train_X = pd.DataFrame({"phone": ["+10000000001", "+10000000002"], "f": [1.0, 2.0]})
    phone_h = add_hash_suffix("phone")
    f_h = add_hash_suffix("f")
    df_fit = pd.DataFrame(
        {
            phone_h: _converted_phones(["+10000000001", "+10000000002"]),
            f_h: [1.0, 2.0],
            TARGET: [0.0, 1.0],
            ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0],
        }
    )
    fit_features = pd.DataFrame({ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0], "ads_feature": [10.0, 20.0]})
    enricher, search_task = _make_reuse_enricher(
        requests_mock,
        search_keys={"phone": SearchKey.PHONE},
        train_X=train_X,
        df_fit=df_fit,
        fit_features=fit_features,
        file_columns=[_file_col("phone", phone_h), _file_col("f", f_h)],
    )
    search_task.validation.side_effect = AssertionError("validation search should not be called for a copy of fit rows")

    result = enricher.transform(train_X.copy(), keep_input=True)
    assert result is not None
    assert len(result) == 2
    assert list(result["ads_feature"]) == [10.0, 20.0]


def test_transform_sends_only_new_rows_to_validation(requests_mock: Mocker):
    train_X = pd.DataFrame({"phone": ["+10000000001", "+10000000002"], "f": [1.0, 2.0]})
    phone_h = add_hash_suffix("phone")
    f_h = add_hash_suffix("f")
    df_fit = pd.DataFrame(
        {
            phone_h: _converted_phones(["+10000000001", "+10000000002"]),
            f_h: [1.0, 2.0],
            TARGET: [0.0, 1.0],
            ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0],
        }
    )
    fit_features = pd.DataFrame({ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0], "ads_feature": [10.0, 20.0]})
    enricher, search_task = _make_reuse_enricher(
        requests_mock,
        search_keys={"phone": SearchKey.PHONE},
        train_X=train_X,
        df_fit=df_fit,
        fit_features=fit_features,
        file_columns=[_file_col("phone", phone_h), _file_col("f", f_h)],
    )
    captured = _stub_validation(search_task, ads_value=99.0)

    mixed = pd.DataFrame({"phone": ["+10000000001", "+10000000002", "+10000000009"], "f": [1.0, 2.0, 9.0]})
    result = enricher.transform(mixed, keep_input=True)
    assert result is not None
    assert len(result) == 3
    assert captured["df"][ENTITY_SYSTEM_RECORD_ID].nunique() == 1
    assert list(result["ads_feature"]) == [10.0, 20.0, 99.0]


def test_transform_all_new_keys_go_to_validation(requests_mock: Mocker):
    train_X = pd.DataFrame({"phone": ["+10000000001", "+10000000002"], "f": [1.0, 2.0]})
    phone_h = add_hash_suffix("phone")
    f_h = add_hash_suffix("f")
    df_fit = pd.DataFrame(
        {
            phone_h: _converted_phones(["+10000000001", "+10000000002"]),
            f_h: [1.0, 2.0],
            TARGET: [0.0, 1.0],
            ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0],
        }
    )
    fit_features = pd.DataFrame({ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0], "ads_feature": [10.0, 20.0]})
    enricher, search_task = _make_reuse_enricher(
        requests_mock,
        search_keys={"phone": SearchKey.PHONE},
        train_X=train_X,
        df_fit=df_fit,
        fit_features=fit_features,
        file_columns=[_file_col("phone", phone_h), _file_col("f", f_h)],
    )
    captured = _stub_validation(search_task, ads_value=77.0)

    new_X = pd.DataFrame({"phone": ["+10000000008", "+10000000009"], "f": [8.0, 9.0]})
    result = enricher.transform(new_X, keep_input=True)
    assert result is not None
    assert len(result) == 2
    assert captured["df"][ENTITY_SYSTEM_RECORD_ID].nunique() == 2
    assert list(result["ads_feature"]) == [77.0, 77.0]


def test_transform_duplicate_keys_do_not_cartesian_join(requests_mock: Mocker):
    train_X = pd.DataFrame({"phone": ["+10000000001"], "f": [1.0]})
    phone_h = add_hash_suffix("phone")
    f_h = add_hash_suffix("f")
    df_fit = pd.DataFrame(
        {
            phone_h: _converted_phones(["+10000000001"]),
            f_h: [1.0],
            TARGET: [0.0],
            ENTITY_SYSTEM_RECORD_ID: [101.0],
        }
    )
    fit_features = pd.DataFrame({ENTITY_SYSTEM_RECORD_ID: [101.0], "ads_feature": [10.0]})
    enricher, search_task = _make_reuse_enricher(
        requests_mock,
        search_keys={"phone": SearchKey.PHONE},
        train_X=train_X,
        df_fit=df_fit,
        fit_features=fit_features,
        file_columns=[_file_col("phone", phone_h), _file_col("f", f_h)],
    )
    search_task.validation.side_effect = AssertionError("duplicate keys should reuse fit, not validate")

    dup_X = pd.DataFrame({"phone": ["+10000000001", "+10000000001"], "f": [1.0, 3.0]})
    result = enricher.transform(dup_X, keep_input=True)
    assert result is not None
    assert len(result) == 2
    assert list(result["ads_feature"]) == [10.0, 10.0]


def test_transform_incomplete_fit_coverage_sends_missing_keys(requests_mock: Mocker):
    train_X = pd.DataFrame({"phone": ["+10000000001", "+10000000002"], "f": [1.0, 2.0]})
    phone_h = add_hash_suffix("phone")
    f_h = add_hash_suffix("f")
    df_fit = pd.DataFrame(
        {
            phone_h: _converted_phones(["+10000000001", "+10000000002"]),
            f_h: [1.0, 2.0],
            TARGET: [0.0, 1.0],
            ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0],
        }
    )
    fit_features = pd.DataFrame({ENTITY_SYSTEM_RECORD_ID: [101.0], "ads_feature": [10.0]})
    enricher, search_task = _make_reuse_enricher(
        requests_mock,
        search_keys={"phone": SearchKey.PHONE},
        train_X=train_X,
        df_fit=df_fit,
        fit_features=fit_features,
        file_columns=[_file_col("phone", phone_h), _file_col("f", f_h)],
    )
    captured = _stub_validation(search_task, ads_value=55.0)

    result = enricher.transform(train_X.copy(), keep_input=True)
    assert result is not None
    assert len(result) == 2
    assert captured["df"][ENTITY_SYSTEM_RECORD_ID].nunique() == 1
    assert list(result["ads_feature"]) == [10.0, 55.0]


def test_transform_restored_enricher_without_fit_snapshot_validates_all(requests_mock: Mocker):
    train_X = pd.DataFrame({"phone": ["+10000000001", "+10000000002"], "f": [1.0, 2.0]})
    phone_h = add_hash_suffix("phone")
    f_h = add_hash_suffix("f")
    df_fit = pd.DataFrame(
        {
            phone_h: _converted_phones(["+10000000001", "+10000000002"]),
            f_h: [1.0, 2.0],
            TARGET: [0.0, 1.0],
            ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0],
        }
    )
    fit_features = pd.DataFrame({ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0], "ads_feature": [10.0, 20.0]})
    enricher, search_task = _make_reuse_enricher(
        requests_mock,
        search_keys={"phone": SearchKey.PHONE},
        train_X=train_X,
        df_fit=df_fit,
        fit_features=fit_features,
        file_columns=[_file_col("phone", phone_h), _file_col("f", f_h)],
    )
    enricher.df_with_original_index = None
    captured = _stub_validation(search_task, ads_value=33.0)

    result = enricher.transform(train_X.copy(), keep_input=True)
    assert result is not None
    assert len(result) == 2
    assert captured["df"][ENTITY_SYSTEM_RECORD_ID].nunique() == 2
    assert list(result["ads_feature"]) == [33.0, 33.0]


def test_transform_restored_enricher_reuses_persisted_match_hashes(requests_mock: Mocker):
    train_X = pd.DataFrame({"phone": ["+10000000001", "+10000000002"], "f": [1.0, 2.0]})
    phone_h = add_hash_suffix("phone")
    f_h = add_hash_suffix("f")
    df_fit = pd.DataFrame(
        {
            phone_h: _converted_phones(["+10000000001", "+10000000002"]),
            f_h: [1.0, 2.0],
            TARGET: [0.0, 1.0],
            ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0],
        }
    )
    fit_features = pd.DataFrame({ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0], "ads_feature": [10.0, 20.0]})
    enricher, search_task = _make_reuse_enricher(
        requests_mock,
        search_keys={"phone": SearchKey.PHONE},
        train_X=train_X,
        df_fit=df_fit,
        fit_features=fit_features,
        file_columns=[_file_col("phone", phone_h), _file_col("f", f_h)],
    )
    _persist_and_drop_snapshot(enricher)
    assert enricher.add_info.transform_match_hashes
    assert len(enricher.add_info.transform_match_hashes) == 2
    assert all(isinstance(h, float) for h in enricher.add_info.transform_match_hashes)
    search_task.update_add_info.assert_called()
    search_task.validation.side_effect = AssertionError(
        "restored enricher with persisted hashes should not validate matching rows"
    )

    result = enricher.transform(train_X.copy(), keep_input=True)
    assert result is not None
    assert len(result) == 2
    assert list(result["ads_feature"]) == [10.0, 20.0]


def test_transform_restored_enricher_persisted_hashes_mixed_rows(requests_mock: Mocker):
    train_X = pd.DataFrame({"phone": ["+10000000001", "+10000000002"], "f": [1.0, 2.0]})
    phone_h = add_hash_suffix("phone")
    f_h = add_hash_suffix("f")
    df_fit = pd.DataFrame(
        {
            phone_h: _converted_phones(["+10000000001", "+10000000002"]),
            f_h: [1.0, 2.0],
            TARGET: [0.0, 1.0],
            ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0],
        }
    )
    fit_features = pd.DataFrame({ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0], "ads_feature": [10.0, 20.0]})
    enricher, search_task = _make_reuse_enricher(
        requests_mock,
        search_keys={"phone": SearchKey.PHONE},
        train_X=train_X,
        df_fit=df_fit,
        fit_features=fit_features,
        file_columns=[_file_col("phone", phone_h), _file_col("f", f_h)],
    )
    _persist_and_drop_snapshot(enricher)
    captured = _stub_validation(search_task, ads_value=77.0)

    mixed = pd.DataFrame({"phone": ["+10000000001", "+19999999999"], "f": [1.0, 9.0]})
    result = enricher.transform(mixed, keep_input=True)
    assert result is not None
    assert len(result) == 2
    assert captured["df"][ENTITY_SYSTEM_RECORD_ID].nunique() == 1
    assert list(result["ads_feature"]) == [10.0, 77.0]


def test_transform_phone_and_date_match_requires_both_keys(requests_mock: Mocker):
    train_X = pd.DataFrame(
        {"phone": ["+10000000001", "+10000000001"], "rep_date": ["2020-01-01", "2020-01-02"], "f": [1.0, 2.0]}
    )
    phone_h = add_hash_suffix("phone")
    date_h = add_hash_suffix("rep_date")
    f_h = add_hash_suffix("f")
    snap = train_X.copy()
    snap = DateTimeConverter("rep_date", date_format="%Y-%m-%d", generate_cyclical_features=False).convert(snap)
    snap = snap.rename(columns={"phone": phone_h, "rep_date": date_h, "f": f_h})
    snap = PhoneSearchKeyConverter(phone_h).convert(snap)
    snap[TARGET] = [0.0, 1.0]
    snap[ENTITY_SYSTEM_RECORD_ID] = [101.0, 102.0]
    fit_features = pd.DataFrame({ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0], "ads_feature": [10.0, 20.0]})
    enricher, search_task = _make_reuse_enricher(
        requests_mock,
        search_keys={"phone": SearchKey.PHONE, "rep_date": SearchKey.DATE},
        train_X=train_X,
        df_fit=snap,
        fit_features=fit_features,
        file_columns=[_file_col("phone", phone_h), _file_col("rep_date", date_h), _file_col("f", f_h)],
    )
    enricher.date_format = "%Y-%m-%d"
    captured = _stub_validation(search_task, ads_value=88.0)

    mixed = pd.DataFrame(
        {
            "phone": ["+10000000001", "+10000000001"],
            "rep_date": ["2020-01-01", "2020-01-03"],
            "f": [1.0, 3.0],
        }
    )
    result = enricher.transform(mixed, keep_input=True)
    assert result is not None
    assert len(result) == 2
    assert captured["df"][ENTITY_SYSTEM_RECORD_ID].nunique() == 1
    assert list(result["ads_feature"]) == [10.0, 88.0]


def test_transform_exclude_features_sources_drops_reused_columns(requests_mock: Mocker):
    train_X = pd.DataFrame({"phone": ["+10000000001", "+10000000002"], "f": [1.0, 2.0]})
    phone_h = add_hash_suffix("phone")
    f_h = add_hash_suffix("f")
    df_fit = pd.DataFrame(
        {
            phone_h: _converted_phones(["+10000000001", "+10000000002"]),
            f_h: [1.0, 2.0],
            TARGET: [0.0, 1.0],
            ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0],
        }
    )
    fit_features = pd.DataFrame({ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0], "ads_feature": [10.0, 20.0]})
    enricher, search_task = _make_reuse_enricher(
        requests_mock,
        search_keys={"phone": SearchKey.PHONE},
        train_X=train_X,
        df_fit=df_fit,
        fit_features=fit_features,
        file_columns=[_file_col("phone", phone_h), _file_col("f", f_h)],
    )
    search_task.validation.side_effect = AssertionError("excluded reuse should not validate")

    result = enricher.transform(train_X.copy(), keep_input=True, exclude_features_sources=["ads_feature"])
    assert result is not None
    assert "ads_feature" not in result.columns


def test_transform_two_email_columns_does_not_duplicate_rows(requests_mock: Mocker):
    train_X = pd.DataFrame(
        {
            "email1": ["a@x.com", "b@x.com"],
            "email2": ["c@x.com", "d@x.com"],
            "f": [1.0, 2.0],
        }
    )
    e1_h = add_hash_suffix("email1")
    e2_h = add_hash_suffix("email2")
    f_h = add_hash_suffix("f")
    df_fit = pd.DataFrame(
        {
            e1_h: train_X["email1"],
            e2_h: train_X["email2"],
            f_h: [1.0, 2.0],
            TARGET: [0.0, 1.0],
            ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0],
        }
    )
    fit_features = pd.DataFrame({ENTITY_SYSTEM_RECORD_ID: [101.0, 102.0], "ads_feature": [10.0, 20.0]})
    enricher, search_task = _make_reuse_enricher(
        requests_mock,
        search_keys={"email1": SearchKey.EMAIL, "email2": SearchKey.EMAIL},
        train_X=train_X,
        df_fit=df_fit,
        fit_features=fit_features,
        file_columns=[_file_col("email1", e1_h), _file_col("email2", e2_h), _file_col("f", f_h)],
    )
    search_task.validation.side_effect = AssertionError("two-email fit keys should not explode via validation")

    result = enricher.transform(train_X.copy(), keep_input=True)
    assert result is not None
    assert len(result) == len(train_X)
    assert list(result["ads_feature"]) == [10.0, 20.0]

    _persist_and_drop_snapshot(enricher)
    search_task.validation.side_effect = AssertionError(
        "restored two-email enricher with persisted hashes should not validate"
    )
    restored = enricher.transform(train_X.copy(), keep_input=True)
    assert restored is not None
    assert len(restored) == len(train_X)
    assert list(restored["ads_feature"]) == [10.0, 20.0]


def test_align_match_frame_noop_and_missing_column(requests_mock: Mocker):
    url = "http://fake_url2"
    mock_default_requests(requests_mock, url)
    enricher = FeaturesEnricher(
        search_keys={"phone": SearchKey.PHONE},
        endpoint=url,
        api_key="fake_api_key",
        logs_enabled=False,
    )
    source = pd.DataFrame({"a": pd.array([1, 2], dtype="Int64")})
    target = pd.DataFrame({"a": pd.array([1, 2], dtype="Int64")})
    aligned = enricher._FeaturesEnricher__align_match_frame(source, target, ["a"])
    assert aligned is not None
    assert list(aligned["a"]) == [1, 2]

    missing = enricher._FeaturesEnricher__align_match_frame(source, target, ["missing"])
    assert missing is None

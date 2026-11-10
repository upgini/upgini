import json

from upgini import SearchKey
from upgini.data_source.data_source_publisher import DataSourcePublisher
from upgini.metadata import AdsHint, AdsHintType


class _UnionClient:
    def __init__(self):
        self.request = None

    def union_search_tasks(self, request, trace_id):
        self.request = request
        return "merged-search-id"


def _publisher() -> DataSourcePublisher:
    publisher = DataSourcePublisher.__new__(DataSourcePublisher)
    publisher._rest_client = _UnionClient()
    return publisher


def test_union_search_tasks_sends_search_avg_date():
    publisher = _publisher()

    search_id = publisher.union_search_tasks(
        ["search-a", "search-b"],
        "user@example.com",
        search_avg_date=" 2024-06-15 ",
    )

    assert search_id == "merged-search-id"
    assert publisher._rest_client.request == {
        "search_task_ids": ["search-a", "search-b"],
        "target_user_email": "user@example.com",
        "search_avg_date": "2024-06-15",
    }


def test_union_search_tasks_omits_blank_search_avg_date():
    publisher = _publisher()

    publisher.union_search_tasks(["search-a"], "user@example.com", search_avg_date="  ")

    assert "search_avg_date" not in publisher._rest_client.request


def test_union_search_tasks_sends_available_only_from_russia():
    publisher = _publisher()

    search_id = publisher.union_search_tasks(
        ["search-a", "search-b"],
        "user@example.com",
        available_only_from_russia=True,
    )

    assert search_id == "merged-search-id"
    assert publisher._rest_client.request == {
        "search_task_ids": ["search-a", "search-b"],
        "target_user_email": "user@example.com",
        "available_only_from_russia": True,
    }


def test_union_search_tasks_sends_available_only_from_russia_false():
    publisher = _publisher()

    publisher.union_search_tasks(
        ["search-a"],
        "user@example.com",
        available_only_from_russia=False,
    )

    assert publisher._rest_client.request["available_only_from_russia"] is False


def test_union_search_tasks_omits_available_only_from_russia_when_unset():
    publisher = _publisher()

    publisher.union_search_tasks(["search-a"], "user@example.com")

    assert "available_only_from_russia" not in publisher._rest_client.request


def test_build_place_request_with_ads_hints():
    ads_hints: list[AdsHint] = [
        AdsHint(
            adsHintType=AdsHintType.DATE_CLUSTER_KEY,
            hintColumnName="clustered_date",
            fullyQualifiedTableName="my-dataset.table_with_clustered_date",
        )
    ]
    print(f"1. Type of ads_hints: {type(ads_hints)}")
    print(f"2. Length of ads_hints: {len(ads_hints)}")

    for i, item in enumerate(ads_hints):
        print(f"3. Item {i} type: {type(item)}")
        print(f"4. Item {i}: {item}")
    ads_hints_as_dict = [hint.model_dump() for hint in ads_hints]  # Wrap in another list

    print(ads_hints_as_dict)
    req = DataSourcePublisher.build_place_request(
        data_table_uri="bq://my-project-dev.my-dataset.view_with_clustered_column",
        search_keys={"msisdn": SearchKey.PHONE, "clustered_date": SearchKey.DATE},
        date_format="%Y-%m-%d",
        update_frequency="Daily",
        snapshot_frequency_days=36000,
        generate_runtime_embeddings=[],
        exclude_raw=[],
        exclude_from_autofe_generation=[],
        sort_column="row_num",
        ads_hints=ads_hints,
    )
    req_as_json = json.dumps(req, indent=2)
    req_as_dict_from_json = json.loads(req_as_json)
    expected = {
        "adsHints": [
            {
                "adsHintType": "DATE_CLUSTER_KEY",
                "fullyQualifiedTableName": "my-dataset.table_with_clustered_date",
                "hintColumnName": "clustered_date",
            }
        ],
        "dataTableUri": "bq://my-project-dev.my-dataset.view_with_clustered_column",
        "dateFormat": "%Y-%m-%d",
        "excludeColumns": None,
        "excludeFromGeneration": [],
        "excludeRaw": [],
        "featuresForEmbeddings": {},
        "forceGeneration": "false",
        "generateRuntimeEmbeddingsFeatures": [],
        "hashFeatureNames": "false",
        "joinDateAbsLimitDays": None,
        "searchKeys": {"clustered_date": "DATE", "msisdn": "MSISDN"},
        "snapshotFrequencyDays": 36000,
        "sortColumn": "row_num",
        "updateFrequency": "Daily",
    }
    assert req_as_dict_from_json == expected

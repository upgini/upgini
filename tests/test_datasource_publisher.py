import json

from upgini import SearchKey
from upgini.data_source.data_source_publisher import DataSourcePublisher
from upgini.metadata import AdsHint, AdsHintType


def test_build_place_request_with_ads_hints():
    ads_hints: list[AdsHint] = [AdsHint(
        ads_hint_type=AdsHintType.DATE_CLUSTER_KEY,
        hint_column_name="clustered_date",
        fully_qualified_table_name="my-dataset.table_with_clustered_date")]
    print(f"1. Type of ads_hints: {type(ads_hints)}")
    print(f"2. Length of ads_hints: {len(ads_hints)}")

    for i, item in enumerate(ads_hints):
        print(f"3. Item {i} type: {type(item)}")
        print(f"4. Item {i}: {item}")
    ads_hints_as_dict = [hint.model_dump() for hint in ads_hints]  # Wrap in another list

    print(ads_hints_as_dict)
    req = DataSourcePublisher.build_place_request(
        data_table_uri=f"bq://my-project-dev.my-dataset.view_with_clustered_column",
        search_keys={"msisdn": SearchKey.PHONE, "clustered_date": SearchKey.DATE},
        date_format="%Y-%m-%d",
        update_frequency="Daily",
        snapshot_frequency_days=36000,
        generate_runtime_embeddings=[],
        exclude_raw=[],
        exclude_from_autofe_generation=[],
        sort_column="row_num",
        ads_hints=ads_hints
    )
    req_as_json = json.dumps(req, indent=2)
    req_as_dict_from_json = json.loads(req_as_json)
    expected = {
        'adsHints': [
            {'ads_hint_type': 'DATE_CLUSTER_KEY',
             'fully_qualified_table_name': 'my-dataset.table_with_clustered_date',
             'hint_column_name': 'clustered_date'
             }
        ],
        'dataTableUri': 'bq://my-project-dev.my-dataset.view_with_clustered_column', 'dateFormat': '%Y-%m-%d',
        'excludeColumns': None,
        'excludeFromGeneration': [],
        'excludeRaw': [],
        'featuresForEmbeddings': {},
        'forceGeneration': 'false',
        'generateRuntimeEmbeddingsFeatures': [],
        'hashFeatureNames': 'false',
        'joinDateAbsLimitDays': None,
        'searchKeys': {'clustered_date': 'DATE', 'msisdn': 'MSISDN'},
        'snapshotFrequencyDays': 36000, 'sortColumn': 'row_num', 'updateFrequency': 'Daily'}
    assert req_as_dict_from_json == expected

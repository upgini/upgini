import logging
import os
import re
from typing import List, Optional, Tuple

import pandas as pd
from pandas.api.types import is_float_dtype
from pandas.testing import assert_frame_equal
from requests_mock import Mocker

from tests.utils import (
    mock_default_requests,
    mock_get_metadata,
    mock_get_task_metadata_v2,
    mock_initial_progress,
    mock_initial_search,
    mock_initial_summary,
    mock_raw_features,
)
from upgini.features_enricher import FeaturesEnricher
from upgini.metadata import (
    FeaturesMetadataV2,
    HitRateMetrics,
    ProviderTaskMetadataV2,
    RuntimeParameters,
    SearchKey,
)

exclude_ads_definitions_by_company = {
    "IPInfo": {
        "economics.cbpol": "16504a9a-284a-4a2e-bff0-ca79f71556f3",
        "economics.cbpol_pca": "8073f2f4-6748-4505-aa3f-d60cc3de0a78",
        "economics.cbpol_umap": "cf816624-1a16-479f-95f4-cbce072d3e2f",
        "economics.cpi": "f889c3f4-3fa7-4276-bb4a-038b61e2ea3c",
        "economics.cpi_pca": "496aeabc-0f7f-435d-98d7-d970f5ce5d69",
        "economics.cpi_umap": "66b99093-82c8-48a8-af53-245b490dced8",
        #
        "finance.finance": "6761df2e-5fc3-4c62-a2be-5dc26a1d20ea",
        "finance.finance_pca": "03fec633-4f76-416f-bd56-d424d9149618",
        "finance.finance_umap": "eb789f14-e77f-4b1c-8e99-8ac799921563",
        #
        "dbip.dbip_2_location": "37384210-a42b-43ad-930d-818722b7e093",
        "dbip.asn_lite_ads_llm": "ad8c1c43-60d0-4488-b503-d0e65f6f57fb",
        #
        "ip2location.db3_ip_country_region_city_llm": "cb206ba5-6599-494e-95e5-6e197fb827bb",
        "ip2location.ip2location_lite_llm": "cd25b391-7d7d-4f2b-bcee-24f03747e910",
        "ip2location.ip2asn_lite_ads_llm": "9981d975-e3e3-4d46-ac36-1e5fa39d173a",
        "ip2location.ip2location_country_info": "39b5cb09-092f-4684-ad92-e205dc5d6802",
        #
        "maxmind.geolite_asn": "1ae32a89-7048-472c-b040-b77fd389550b",
        "maxmind.geolite2": "1ac52496-cd22-4f2a-b901-4b43e61e818b",
        #
        "umkus.ip_index": "7f8f82d2-4905-4403-b440-2db6cd6daf60",
    }
}

field_column_header = "Field"
shap_column_header = "SHAP value"
included_in_data_listings_header = "Included in data listings"
data_listings_header = "Data listing"
all_fields_shap_header = "All fields SHAP"
relevant_features_header = "Relevant features"
price_header = "Price"
action_header = "Action"


def run_search(
    df: pd.DataFrame,
    target_column: str,
    ip_column: str,
    date_column: Optional[str],
    generate_features: List[str],
    # exclude_columns: List[str],
    company="IPInfo",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    search_keys = {
        ip_column: SearchKey.IP,
    }
    if date_column:
        search_keys[date_column] = SearchKey.DATE
    enricher = FeaturesEnricher(
        search_keys=search_keys,
        api_key="fake_api_key",
        endpoint="http://fake_url",
        generate_features=generate_features,
        runtime_parameters=RuntimeParameters(
            properties={"excludeAdsDefinitions": ",".join(exclude_ads_definitions_by_company[company].values())}
        )
        if company in exclude_ads_definitions_by_company
        else None,
        detect_missing_search_keys=False,
        raise_validation_error=True,
    )
    enricher.fit(df.drop(columns=target_column), df[target_column], calculate_metrics=False)
    fi = enricher.get_features_info()

    relevant_fields = fi.rename(
        columns={"Source": included_in_data_listings_header, "Feature name": field_column_header}
    )
    relevant_fields = relevant_fields.query("Provider != ''")
    relevant_fields = relevant_fields.drop(columns=["Coverage %", "Feature type"], errors="ignore")

    raw_relevant_fields = relevant_fields.copy()

    upgini_filter = (
        relevant_fields["Provider"].str.contains("Upgini")
        | relevant_fields[field_column_header].str.contains("_emb")
        | relevant_fields[field_column_header].str.endswith("_n_tokens")
        | relevant_fields[field_column_header].str.endswith("_gender")
        | relevant_fields[field_column_header].str.endswith("_name_group")
        | relevant_fields[field_column_header].str.endswith("_sentiment_rating")
    )
    upgini_data_listing = (
        "<a href='https://ipinfo.io/products/ip-database-download' target='_blank' rel='noopener noreferrer'>Optimized "
        "for ML database</a>"
    )
    # All features included in Upgini listing. Generated features included only in Upgini listing. Other combines
    relevant_fields.loc[upgini_filter, included_in_data_listings_header] = upgini_data_listing
    relevant_fields.loc[~upgini_filter, included_in_data_listings_header] = (
        upgini_data_listing + ", " + relevant_fields[~upgini_filter][included_in_data_listings_header]
    )
    relevant_fields.loc[upgini_filter, "Provider"] = "Upgini"
    raw_relevant_fields.loc[upgini_filter, "Provider"] = "Upgini"

    def remove_anchor(s):
        return re.sub(r"<a[^>]+>([\w\s]+)<\/a>", r"\1", s)

    def get_original_name(s):
        s = re.sub(r"<a.+>f_(.+)_\w+<\/a>", r"\1", s)
        return re.sub(r"f_(.+)_\w+", r"\1", s)

    features_df = enricher._search_task.get_all_initial_raw_features(
        "download_sample_features_from_widget", metrics_calculation=True
    )

    def add_hint(feature):
        sample = []
        for c in features_df.columns:
            if c in feature:
                column: pd.Series = features_df[c]
                sample = column.dropna().sample(n=10, random_state=42)
                if is_float_dtype(sample):
                    sample = sample.round(4)
                sample = sample.astype(str).values
        logging.info(f"Sample of feature {feature}: {sample}")
        sample_markup = "</br>".join(sample)
        match_with_link = re.match("(<a.+>)f_(.+)_\\w+(<\\/a>)", feature)
        match_without_link = re.match("f_(.+)_\\w+", feature)
        if match_with_link is not None:
            renamed_feature = f"{match_with_link.group(1)}{match_with_link.group(2)}{match_with_link.group(3)}"
        elif match_without_link is not None:
            renamed_feature = match_without_link.group(1)
        else:
            renamed_feature = feature
        logging.info(f"Sample markup of {renamed_feature}: {sample_markup}")
        return f"<div class='tooltip'>{renamed_feature}<span class='tooltiptext'>{sample_markup}</span></div>"

    relevant_fields[field_column_header] = relevant_fields[field_column_header].apply(add_hint)
    summary = (
        raw_relevant_fields[raw_relevant_fields["Provider"] != "Upgini"]
        .drop(columns="Provider")
        .groupby(by=included_in_data_listings_header)
        .agg(
            shap=pd.NamedAgg(column=shap_column_header, aggfunc="sum"),
            count=pd.NamedAgg(column=shap_column_header, aggfunc="count"),
        )
        # .sum(shap_column_header)
        .sort_values(by="shap", ascending=False)
        .reset_index()
        .rename(
            columns={
                included_in_data_listings_header: data_listings_header,
                "shap": all_fields_shap_header,
                "count": relevant_features_header,
            }
        )
    )
    relevant_fields = relevant_fields.drop(columns="Provider")

    def add_button(listing: str) -> str:
        url_anchor = re.match("(<a href=.+>).+</a>", listing)
        if url_anchor is not None:
            btn_html = (
                f'<div class="stButton">{url_anchor.group(1)}<button kind="secondary"><p>Instant purchase</p>'
                "</button></a></div>"
            )
            return btn_html
        else:
            return ""

    summary[action_header] = summary[data_listings_header].apply(add_button)

    customized_product_title = "Optimized for ML database"
    customized_product_row = pd.DataFrame(
        {
            data_listings_header: [
                f"<a href='https://ipinfo.io/products/ip-database-download'>{customized_product_title}</a>"
            ],
            all_fields_shap_header: [relevant_fields[shap_column_header].sum()],
            relevant_features_header: [relevant_fields[shap_column_header].count()],
            action_header: [
                # "<a href='https://upgini.com'><button class='css-1n543e5 bluebutton'>Request a quote</button></a>"
                # Temporary for IPInfo
                add_button(f"<a href='https://ipinfo.io/products/ip-database-download'>{customized_product_title}</a>")
                # "Request a quote"
            ],
        }
    )
    summary = pd.concat([customized_product_row, summary]).reset_index()

    def price_by_listing(listing):
        if customized_product_title in listing:
            return "By request"
        elif "IP Address Data for Mobile Carrier Detection" in listing:
            return "Trial 30 days, $80/month"
        elif "Privacy detection database" in listing:
            return "$1000/one-off"
        elif "IP Geolocation Data" in listing:
            return "No Trial, $800/month"
        else:
            return ""

    summary[price_header] = summary[data_listings_header].apply(price_by_listing)
    summary[relevant_features_header] = summary[relevant_features_header].astype(int)
    summary = summary.rename(columns={shap_column_header: all_fields_shap_header})
    summary = summary[
        [data_listings_header, all_fields_shap_header, relevant_features_header, price_header, action_header]
    ]

    relevant_fields = relevant_fields[[field_column_header, shap_column_header, included_in_data_listings_header]]

    # Using for email
    raw_relevant_fields = raw_relevant_fields.drop(columns="Provider")
    raw_relevant_fields[included_in_data_listings_header] = raw_relevant_fields[included_in_data_listings_header].apply(
        remove_anchor
    )
    raw_relevant_fields[field_column_header] = raw_relevant_fields[field_column_header].apply(get_original_name)

    return relevant_fields, summary, raw_relevant_fields


def test_widget(requests_mock: Mocker):
    url = "http://fake_url"
    mock_default_requests(requests_mock, url)

    search_task_id = mock_initial_search(requests_mock, url)
    mock_initial_progress(requests_mock, url, search_task_id)
    ads_search_task_id = mock_initial_summary(requests_mock, url, search_task_id)
    mock_get_metadata(
        requests_mock,
        url,
        search_task_id,
        metadata_columns=[
            {
                "index": 0,
                "name": "ipv4_fake",
                "originalName": "ipv4",
                "dataType": "STRING",
                "meaningType": "IP_ADDRESS",
            },
            {
                "index": 1,
                "name": "app_date_fake",
                "originalName": "app_date",
                "dataType": "INT",
                "meaningType": "DATE",
            },
            {"index": 2, "name": "target", "originalName": "target", "dataType": "INT", "meaningType": "TARGET"},
            {
                "index": 3,
                "name": "system_record_id",
                "originalName": "system_record_id",
                "dataType": "INT",
                "meaningType": "SYSTEM_RECORD_ID",
            },
        ],
        search_keys=["ipv4", "app_date"],
    )
    mock_get_task_metadata_v2(
        requests_mock,
        url,
        ads_search_task_id,
        ProviderTaskMetadataV2(
            features=[
                FeaturesMetadataV2(
                    name="f_events_date_month_sin4_81a273ac",
                    type="NUMERIC",
                    source="ads",
                    hit_rate=100.0,
                    shap_value=0.0924,
                    commercial_schema="Free",
                    data_provider="Upgini",
                    data_provider_link="https://upgini.com",
                    data_source="Calendar data",
                    data_source_link="https://upgini.com/#data_sources",
                    doc_link="https://docs.upgini.com/public/calendar/calendar#f_events_date_month_sin4_81a273ac",
                    update_frequency="Daily",
                ),
                FeaturesMetadataV2(
                    name="f_ip_company_name_emb145_e56eb2a3",
                    type="NUMERIC",
                    source="ads",
                    hit_rate=100.0,
                    shap_value=0.0787,
                    commercial_schema="Trial",
                    data_provider="IPInfo",
                    data_provider_link="https://ipinfo.io",
                    data_source="Company IP Address Data",
                    data_source_link="https://app.snowflake.com/marketplace/listing/GZSTZ3VDMFU/?referer=upgini",
                    update_frequency="Monthly",
                ),
                FeaturesMetadataV2(
                    name="f_ip_company_name_emb54_805ee5b8",
                    type="NUMERIC",
                    source="ads",
                    hit_rate=100.0,
                    shap_value=0.0496,
                    commercial_schema="Trial",
                    data_provider="IPInfo",
                    data_provider_link="https://ipinfo.io",
                    data_source="Company IP Address Data",
                    data_source_link="https://app.snowflake.com/marketplace/listing/GZSTZ3VDMFU/?referer=upgini",
                    update_frequency="Monthly",
                ),
                FeaturesMetadataV2(
                    name="f_ip_operator_5d5fc3f3",
                    type="STRING",
                    source="ads",
                    hit_rate=44.5114,
                    shap_value=0.0019,
                    commercial_schema="Trial",
                    data_provider="IPInfo",
                    data_provider_link="https://ipinfo.io",
                    data_source="IP Address Data for Mobile Carrier Detection",
                    data_source_link="https://app.snowflake.com/marketplace/listing/GZSTZ3VDMF6/?referer=upgini",
                    update_frequency="Monthly",
                ),
            ],
            hit_rate_metrics=HitRateMetrics(
                etalon_row_count=10000, max_hit_count=9990, hit_rate=0.999, hit_rate_percent=99.9
            ),
            eval_set_metrics=[],
        ),
    )
    path_to_mock_features = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data/widget/features.parquet"
    )
    mock_raw_features(requests_mock, url, search_task_id, path_to_mock_features)

    path_to_tds = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/widget/tds.parquet")
    df = pd.read_parquet(path_to_tds)

    relevant_fields, summary, _ = run_search(df, "target", "ipv4", "app_date", [])

    expected_relevant_fields = pd.DataFrame(
        {
            "Field": [
                (
                    "<div class='tooltip'><a href='https://docs.upgini.com/public/calendar/calendar"
                    "#f_events_date_month_sin4_81a273ac' target='_blank' "
                    "rel='noopener noreferrer'>events_date_month_sin4</a><span class='tooltiptext'>"
                    "-0.5878</br>0.1012</br>0.5878</br>0.866</br>0.5878</br>-0.5878</br>-0.7908</br>0.2079</br>"
                    "-0.0</br>0.4067</span></div>"
                ),
                (
                    "<div class='tooltip'>ip_company_name_emb145<span class='tooltiptext'>0.0014</br>0.0014</br>"
                    "-0.0076</br>0.0014</br>0.0014</br>-0.0083</br>0.0318</br>-0.004</br>-0.0078</br>-0.0035"
                    "</span></div>"
                ),
                (
                    "<div class='tooltip'>ip_company_name_emb54<span class='tooltiptext'>-0.0289</br>-0.0289</br>"
                    "-0.0209</br>-0.0289</br>-0.0289</br>0.0301</br>-0.0368</br>0.0477</br>0.0487</br>0.0302"
                    "</span></div>"
                ),
                (
                    "<div class='tooltip'>ip_operator<span class='tooltiptext'>Beeline</br>Beeline</br>UzMobile</br>"
                    "Beeline</br>Uztelecom</br>Uztelecom</br>Beeline</br>Beeline</br>Beeline</br>Ucell</span></div>"
                ),
            ],
            "SHAP value": [0.0924, 0.0787, 0.0496, 0.0019],
            "Included in data listings": [
                (
                    "<a href='https://ipinfo.io/products/ip-database-download' target='_blank' "
                    "rel='noopener noreferrer'>Optimized for ML database</a>"
                ),
                (
                    "<a href='https://ipinfo.io/products/ip-database-download' target='_blank' "
                    "rel='noopener noreferrer'>Optimized for ML database</a>"
                ),
                (
                    "<a href='https://ipinfo.io/products/ip-database-download' target='_blank' "
                    "rel='noopener noreferrer'>Optimized for ML database</a>"
                ),
                (
                    "<a href='https://ipinfo.io/products/ip-database-download' target='_blank' "
                    "rel='noopener noreferrer'>Optimized for ML database</a>, "
                    "<a href='https://app.snowflake.com/marketplace/listing/GZSTZ3VDMF6/?referer=upgini' "
                    "target='_blank' rel='noopener noreferrer'>IP Address Data for Mobile Carrier Detection</a>"
                ),
            ],
        }
    )

    assert_frame_equal(relevant_fields, expected_relevant_fields)

    expected_summary = pd.DataFrame(
        {
            "Data listing": [
                "<a href='https://ipinfo.io/products/ip-database-download'>Optimized for ML database</a>",
                (
                    "<a href='https://app.snowflake.com/marketplace/listing/GZSTZ3VDMF6/?referer=upgini' "
                    "target='_blank' rel='noopener noreferrer'>IP Address Data for Mobile Carrier Detection</a>"
                ),
            ],
            "All fields SHAP": [0.2226, 0.0019],
            "Relevant features": [4, 1],
            "Price": ["By request", "Trial 30 days, $80/month"],
            "Action": [
                (
                    "<div class=\"stButton\"><a href='https://ipinfo.io/products/ip-database-download'>"
                    '<button kind="secondary"><p>Instant purchase</p></button></a></div>'
                ),
                (
                    '<div class="stButton"><a href=\'https://app.snowflake.com/marketplace/listing/GZSTZ3VDMF6/'
                    "?referer=upgini' target='_blank' rel='noopener noreferrer'><button kind=\"secondary\"><p>"
                    "Instant purchase</p></button></a></div>"
                ),
            ],
        }
    )

    assert_frame_equal(summary, expected_summary)

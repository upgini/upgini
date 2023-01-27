

from enum import Enum
from typing import Dict, List, Optional
from upgini.metadata import SearchKey


class CommercialSchema(Enum):
    FREE = "FREE"
    PAID = "PAID"


class DataSourcePublisher:

    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key

    def place(
        self,
        datatable_uri: str,  # "bq://datadiscovery-spark-dev.american_community_survey.blockgroup_2010_5yr",
        search_keys: Dict[str, SearchKey],  # {"date": SearchKey.DATE},
        selected_columns: Optional[List[str]] = None,
        hash_feature_names=False,
        commercial_schema=CommercialSchema.FREE,
        is_private=True,
        is_test=False,
        snapshot_frequency_days: Optional[int] = None,
        # Use it with default values
        provider="Upgini",
        provider_link="https://upgini.com/#data_sources",
        source="Public/Comm. shared",
        source_link="https://upgini.com/#data_sources",
    ):
        pass

    def activate(self, datatable_ids: List[str], client_emails: List[str]):
        pass

    def remove(self, datatable_ids: List[str]):
        pass

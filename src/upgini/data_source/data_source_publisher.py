import logging
import time
import uuid
from enum import Enum
from typing import Dict, List, Optional, Union

from upgini.errors import HttpError, ValidationError
from upgini.http import LoggerFactory, get_rest_client
from upgini.mdc import MDC
from upgini.metadata import SearchKey
from upgini.spinner import Spinner


class CommercialSchema(Enum):
    FREE = "Free"
    TRIAL = "Trial"
    PAID = "Paid"


class ListingType(Enum):
    PUBLIC = "PROD"
    PRIVATE = "PRIVATE_SHARE"
    TEST = "DEV"


class DataSourcePublisher:

    FINAL_STATUSES = ["COMPLETED", "FAILED", "TIMED_OUT"]

    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None, logs_enabled=True):
        self._rest_client = get_rest_client(endpoint, api_key)
        if logs_enabled:
            self.logger = LoggerFactory().get_logger(endpoint, api_key)
        else:
            self.logger = logging.getLogger()
            self.logger.setLevel("FATAL")

    def place(
        self,
        data_table_uri: str,
        search_keys: Dict[str, SearchKey],
        date_format: Optional[str] = None,
        exclude_columns: Optional[List[str]] = None,
        hash_feature_names=False,
        snapshot_frequency_days: Optional[int] = None,
    ) -> str:
        trace_id = str(uuid.uuid4())

        with MDC(trace_id=trace_id):
            try:
                if data_table_uri is None or not data_table_uri.startswith("bq://"):
                    raise ValidationError(
                        "Unsupported data table uri. It should looks like bq://projectId.datasetId.tableId"
                    )
                if search_keys is None or len(search_keys) == 0:
                    raise ValidationError("Empty search keys")
                if SearchKey.DATE in search_keys.values() and date_format is None:
                    raise ValidationError("date_format is required for DATE search key")

                request = {
                    "dataTableUri": data_table_uri,
                    "searchKeys": {k: v.value.value for k, v in search_keys.items()},
                    "dateFormat": date_format,
                    "excludeColumns": exclude_columns,
                    "hashFeatureNames": hash_feature_names,
                    "snapshotFrequencyDays": snapshot_frequency_days,
                }
                self.logger.info(f"Start registering data table {request}")

                task_id = self._rest_client.register_ads(request, trace_id)
                msg = f"Data table management task created. task_id={task_id}"
                self.logger.info(msg)
                print(msg)
                with Spinner():
                    status_response = self._rest_client.poll_ads_management_task_status(task_id, trace_id)
                    while status_response["status"] not in self.FINAL_STATUSES:
                        time.sleep(5)
                        status_response = self._rest_client.poll_ads_management_task_status(task_id, trace_id)

                if status_response["status"] != "COMPLETED":
                    raise Exception("Failed to register ADS: " + status_response["errorMessage"])

                data_table_id = status_response["adsDefinitionId"]
                msg = f"Data table successfully registered with id: {data_table_id}"
                self.logger.info(msg)
                print(msg)
                return data_table_id
            except Exception:
                self.logger.exception("Failed to register data table")
                raise

    def remove(self, data_table_ids: List[str]):
        trace_id = str(uuid.uuid4())
        with MDC(trace_id=trace_id):
            try:
                if data_table_ids is None or len(data_table_ids) == 0:
                    raise ValidationError("Empty data table ids")

                for data_table_id in data_table_ids:
                    task_id = self._rest_client.delete_ads(data_table_id, trace_id)
                    with Spinner():
                        status_response = self._rest_client.poll_ads_management_task_status(task_id, trace_id)
                        while status_response["status"] not in self.FINAL_STATUSES:
                            time.sleep(5)
                            status_response = self._rest_client.poll_ads_management_task_status(task_id, trace_id)

                    if status_response["status"] != "COMPLETED":
                        raise Exception("Failed to register ADS: " + status_response["errorMessage"])
            except Exception:
                self.logger.exception(f"Failed to remove data tables {data_table_ids}")

    def activate(
        self,
        data_table_ids: Union[List[str], str],
        commercial_schema: Optional[CommercialSchema] = None,
        listing_type: Optional[ListingType] = None,
        provider: Optional[str] = None,
        provider_link: Optional[str] = None,
        source: Optional[str] = None,
        source_link: Optional[str] = None,
        client_emails: Optional[List[str]] = None,
    ):
        trace_id = str(uuid.uuid4())
        with MDC(trace_id=trace_id):
            try:
                if data_table_ids is None or len(data_table_ids) == 0:
                    raise ValidationError("Empty data table ids")
                if listing_type == ListingType.PRIVATE and (client_emails is None or len(client_emails) == 0):
                    raise ValidationError("Empty client emails for private data tables")
                if listing_type != ListingType.PRIVATE and client_emails is not None:
                    raise ValidationError("Client emails make sense only for private data table")

                request = {"dataTableIds": data_table_ids}
                if commercial_schema is not None:
                    request["commercialSchema"] = commercial_schema
                if listing_type is not None:
                    request["listingType"] = listing_type.value
                if provider is not None:
                    request["provider"] = provider
                if provider_link is not None:
                    request["providerLink"] = provider_link
                if source is not None:
                    request["source"] = source
                if source_link is not None:
                    request["sourceLink"] = source_link
                if client_emails is not None:
                    request["clientEmails"] = client_emails
                self.logger.info(f"Activating data tables with request {request}")

                self._rest_client.activate_datatables(request, trace_id)

                msg = "Data tables successfully activated"
                self.logger.info(msg)
                print(msg)
            except HttpError as e:
                if e.status_code == 404:
                    raise Exception("One of data tables not found")
            except Exception:
                self.logger.exception("Failed to activate data tables")
                raise

    def deactivate(self, data_table_ids: List[str], client_emails: Optional[List[str]] = None):
        trace_id = str(uuid.uuid4())
        with MDC(trace_id=trace_id):
            try:
                if data_table_ids is None or len(data_table_ids) == 0:
                    raise ValidationError("Empty data table ids")

                request = {"dataTableIds": data_table_ids, "clientEmails": client_emails}
                self.logger.info(f"Deactivating data tables with request {request}")
                self._rest_client.deactivate_data_tables(request, trace_id)
                msg = "Data tables successfully deactivated"
                self.logger.info(msg)
                print(msg)
            except Exception:
                self.logger.exception(f"Failed to deactivate data tables {data_table_ids} for clients {client_emails}")

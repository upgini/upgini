import logging
import time
import uuid
from datetime import datetime
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


class OnlineUploadingType(Enum):
    BINDINGS = "BINDINGS"
    IP = "IP"
    OTHER = "OTHER"


class DataSourcePublisher:
    FINAL_STATUSES = ["COMPLETED", "FAILED", "TIMED_OUT"]
    ACCEPTABLE_UPDATE_FREQUENCIES = ["Daily", "Weekly", "Monthly", "Quarterly", "Annually"]
    DEFAULT_GENERATE_EMBEDDINGS = []

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
        update_frequency: str,
        secondary_search_keys: Optional[Dict[str, SearchKey]] = None,
        sort_column: Optional[str] = None,
        date_format: Optional[str] = None,
        exclude_columns: Optional[List[str]] = None,
        hash_feature_names=False,
        snapshot_frequency_days: Optional[int] = None,
        join_date_abs_limit_days: Optional[int] = None,
        features_for_embeddings: Optional[List[str]] = DEFAULT_GENERATE_EMBEDDINGS,
        data_table_id_to_replace: Optional[str] = None,
        exclude_from_autofe_generation: Optional[List[str]] = None,
        _force_generation=False,
        _silent=False,
    ) -> str:
        trace_id = str(uuid.uuid4())

        with MDC(trace_id=trace_id):
            task_id = None
            try:
                if data_table_uri is None or not data_table_uri.startswith("bq://"):
                    raise ValidationError(
                        "Unsupported data table uri. It should looks like bq://projectId.datasetId.tableId"
                    )
                if search_keys is None or len(search_keys) == 0:
                    raise ValidationError("Empty search keys")
                if SearchKey.DATE in search_keys.values() and date_format is None:
                    raise ValidationError("date_format is required for DATE search key")
                if update_frequency not in self.ACCEPTABLE_UPDATE_FREQUENCIES:
                    raise ValidationError(
                        f"Invalid update frequency: {update_frequency}. "
                        f"Available values: {self.ACCEPTABLE_UPDATE_FREQUENCIES}"
                    )

                request = {
                    "dataTableUri": data_table_uri,
                    "searchKeys": {k: v.value.value for k, v in search_keys.items()},
                    "dateFormat": date_format,
                    "excludeColumns": exclude_columns,
                    "hashFeatureNames": str(hash_feature_names).lower(),
                    "snapshotFrequencyDays": snapshot_frequency_days,
                    "joinDateAbsLimitDays": join_date_abs_limit_days,
                    "updateFrequency": update_frequency,
                    "featuresForEmbeddings": features_for_embeddings,
                    "forceGeneration": str(_force_generation).lower(),
                }
                if secondary_search_keys is not None:
                    request["secondarySearchKeys"] = {k: v.value.value for k, v in secondary_search_keys.items()}
                if sort_column is not None:
                    request["sortColumn"] = sort_column
                if data_table_id_to_replace is not None:
                    request["adsDefinitionIdToReplace"] = data_table_id_to_replace
                if exclude_from_autofe_generation is not None:
                    request["excludeFromGeneration"] = exclude_from_autofe_generation
                self.logger.info(f"Start registering data table {request}")

                task_id = self._rest_client.register_ads(request, trace_id)
                msg = f"Data table management task created. task_id={task_id}"
                self.logger.info(msg)
                print(msg)

                def poll():
                    status_response = self._rest_client.poll_ads_management_task_status(task_id, trace_id)
                    while status_response["status"] not in self.FINAL_STATUSES:
                        time.sleep(5)
                        status_response = self._rest_client.poll_ads_management_task_status(task_id, trace_id)
                    return status_response

                if not _silent:
                    with Spinner():
                        status_response = poll()
                else:
                    status_response = poll()

                if status_response["status"] != "COMPLETED":
                    if "Cost of features generation exceeded the limit" in status_response["errorMessage"]:
                        print(status_response["errorMessage"])

                        import ipywidgets as widgets
                        from IPython.display import display

                        button = widgets.Button(description="Start registration with forced generation")

                        display(button)

                        def on_button_clicked(b):
                            self.place(
                                data_table_uri,
                                search_keys,
                                secondary_search_keys,
                                sort_column,
                                date_format,
                                exclude_columns,
                                hash_feature_names,
                                snapshot_frequency_days,
                                features_for_embeddings,
                                data_table_id_to_replace,
                                exclude_from_autofe_generation,
                                True,
                                _silent,
                            )

                        button.on_click(on_button_clicked)
                        return
                    else:
                        raise Exception("Failed to register ADS: " + status_response["errorMessage"])

                data_table_id = status_response["adsDefinitionId"]
                msg = f"Data table successfully registered with id: {data_table_id}"
                self.logger.info(msg)
                print(msg)
                return data_table_id
            except KeyboardInterrupt:
                if task_id is not None:
                    msg = f"Stopping AdsManagementTask {task_id}"
                    print(msg)
                    self.logger.info(msg)
                    self._rest_client.stop_ads_management_task(task_id, trace_id)
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
        trial_limit: Optional[int] = None,
        expires_at: Optional[str] = None,
        listing_type: Optional[ListingType] = None,
        provider: Optional[str] = None,
        provider_link: Optional[str] = None,
        source: Optional[str] = None,
        source_link: Optional[str] = None,
        update_frequency: Optional[str] = None,
        client_emails: Optional[List[str]] = None,
    ):
        trace_id = str(uuid.uuid4())
        with MDC(trace_id=trace_id):
            try:
                if data_table_ids is None or len(data_table_ids) == 0:
                    raise ValidationError("Empty data table ids")
                if isinstance(data_table_ids, str):
                    data_table_ids = [data_table_ids]
                if not isinstance(data_table_ids, list):
                    raise ValidationError("data_table_ids should be string or list of strings")
                if update_frequency is not None and update_frequency not in self.ACCEPTABLE_UPDATE_FREQUENCIES:
                    raise ValidationError(
                        f"Invalid update frequency: {update_frequency}. "
                        f"Available values: {self.ACCEPTABLE_UPDATE_FREQUENCIES}"
                    )
                # if listing_type == ListingType.PRIVATE and (client_emails is None or len(client_emails) == 0):
                #     raise ValidationError("Empty client emails for private data tables")
                # if listing_type not in [ListingType.PRIVATE, ListingType.TRIAL] and client_emails is not None:
                #     raise ValidationError("Client emails make sense only for private data table")

                request = {"dataTableIds": data_table_ids}
                if commercial_schema is not None:
                    request["commercialSchema"] = commercial_schema.value
                if trial_limit is not None:
                    request["trialLimit"] = trial_limit
                if expires_at is not None:
                    try:
                        datetime.strptime(expires_at, "%Y-%m-%d")
                    except Exception:
                        raise ValidationError("Invalid format of expires_at. It should be like YYYY-MM-DD")
                    request["expiresAt"] = expires_at
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
                if update_frequency is not None:
                    request["updateFrequency"] = update_frequency
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

    def upload_online(self, bq_table_id: Optional[str] = None, search_keys: Optional[List[SearchKey]] = None):
        trace_id = str(uuid.uuid4())
        with MDC(trace_id=trace_id):
            if bq_table_id is None and search_keys is None:
                raise ValidationError("One of arguments: bq_table_id or search_keys should be presented")
            if bq_table_id is not None and search_keys is not None:
                raise ValidationError("Only one argument could be presented: bq_table_id or search_keys")
            try:
                search_keys = [k.value.value for k in search_keys] if search_keys else None
                request = {"bqTableId": bq_table_id, "searchKeys": search_keys}
                task_id = self._rest_client.upload_online(request, trace_id)
                with Spinner():
                    status_response = self._rest_client.poll_ads_management_task_status(task_id, trace_id)
                    while status_response["status"] not in self.FINAL_STATUSES:
                        time.sleep(5)
                        status_response = self._rest_client.poll_ads_management_task_status(task_id, trace_id)

                if status_response["status"] != "COMPLETED":
                    raise Exception("Failed to register ADS: " + status_response["errorMessage"])

                print("Uploading successfully finished")
            except Exception:
                self.logger.exception(f"Failed to upload table {bq_table_id}")
                raise

    def upload_online_all(self):
        search_keys = [
            [SearchKey.COUNTRY],
            [SearchKey.COUNTRY, SearchKey.DATE],
            [SearchKey.COUNTRY, SearchKey.DATE, SearchKey.POSTAL_CODE],
            [SearchKey.COUNTRY, SearchKey.POSTAL_CODE],
            [SearchKey.DATE],
            [SearchKey.HEM],
            [SearchKey.IP],
            [SearchKey.IPV6_RANGE_FROM, SearchKey.IPV6_RANGE_TO],
            [SearchKey.IP_RANGE_FROM, SearchKey.IP_RANGE_TO],
            [SearchKey.PHONE],
            [SearchKey.MSISDN_RANGE_FROM, SearchKey.MSISDN_RANGE_TO],
        ]
        for keys in search_keys:
            self.upload_online(search_keys=keys)

        print("All ADS-es successfully uploaded")

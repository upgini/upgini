

from enum import Enum
import time
from typing import Dict, List, Optional
import uuid
from upgini.errors import HttpError
from upgini.mdc import MDC
from upgini.metadata import SearchKey
from upgini.http import LoggerFactory, get_rest_client
import logging

from upgini.spinner import Spinner


class CommercialSchema(Enum):
    FREE = "Free"
    TRIAL = "Trial"
    PAID = "Paid"


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
        data_table_uri: str,  # "bq://datadiscovery-spark-dev.american_community_survey.blockgroup_2010_5yr",
        search_keys: Dict[str, SearchKey],  # {"date": SearchKey.DATE},
        date_format: Optional[str] = None,
        exclude_columns: Optional[List[str]] = None,
        hash_feature_names=False,
        commercial_schema=CommercialSchema.FREE,
        is_private=False,
        is_test=True,
        snapshot_frequency_days: Optional[int] = None,
        # Use it with default values
        provider="Upgini",
        provider_link="https://upgini.com/#data_sources",
        source="Public/Comm. shared",
        source_link="https://upgini.com/#data_sources",
    ) -> str:
        trace_id = str(uuid.uuid4())
        request = {
            "dataTableUri": data_table_uri,
            "searchKeys": {k: v.value.value for k, v in search_keys.items()},
            "dateFormat": date_format,
            "excludeColumns": exclude_columns,
            "hashFeatureNames": hash_feature_names,
            "commercialSchema": commercial_schema.value,
            "isPrivate": str(is_private).lower(),
            "isTest": str(is_test).lower(),
            "snapshotFrequencyDays": snapshot_frequency_days,
            "provider": provider,
            "providerLink": provider_link,
            "source": source,
            "sourceLink": source_link
        }
        with MDC(trace_id=trace_id):
            self.logger.info(f"Start registering data table {request}")
            try:
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

    def activate(self, data_table_ids: List[str], client_emails: List[str]):
        trace_id = str(uuid.uuid4())
        with MDC(trace_id=trace_id):
            self.logger.info(f"Activating data tables {data_table_ids} for clients {client_emails}")
            request = {
                "dataTableIds": data_table_ids,
                "clientEmails": client_emails
            }
            try:
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

    def remove(self, data_table_ids: List[str]):
        pass

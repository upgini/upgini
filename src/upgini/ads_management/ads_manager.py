import time
from typing import Dict, Optional
import uuid
from upgini.http import get_rest_client
from upgini.spinner import Spinner
import pandas as pd


class AdsManager:
    FINAL_STATUSES = ["COMPLETED", "FAILED", "TIMED_OUT"]

    def __init__(self, api_key: Optional[str] = None, backend_url: Optional[str] = None):
        self.api_key = api_key
        self.backend_url = backend_url
        self.rc = get_rest_client(self.backend_url, self.api_key)

    def register(self, request: Dict) -> str:
        trace_id = str(uuid.uuid4())
        ads_management_task_id = self.rc.register_ads(request, trace_id)
        print(f"Ads management task with id {ads_management_task_id} created")

        with Spinner():
            status_response = self.rc.poll_ads_management_task_status(ads_management_task_id, trace_id)
            while status_response["status"] not in self.FINAL_STATUSES:
                time.sleep(5)
                status_response = self.rc.poll_ads_management_task_status(ads_management_task_id, trace_id)

        if status_response["status"] != "COMPLETED":
            raise Exception("Failed to register ADS: " + status_response["errorMessage"])

        ads_definition_id = status_response["adsDefinitionId"]
        print("ADS successfully registered with id: " + ads_definition_id)
        return ads_definition_id

    def delete(self, ads_definition_id: str):
        trace_id = str(uuid.uuid4())
        ads_management_task_id = self.rc.delete_ads(ads_definition_id, trace_id)
        print(f"Ads management task with id {ads_management_task_id} created")

        with Spinner():
            status_response = self.rc.poll_ads_management_task_status(ads_management_task_id, trace_id)
            while status_response["status"] not in self.FINAL_STATUSES:
                time.sleep(5)
                status_response = self.rc.poll_ads_management_task_status(ads_management_task_id, trace_id)

        if status_response["status"] != "COMPLETED":
            raise Exception("Failed to register ADS: " + status_response["errorMessage"])

        print("ADS successfully deleted")

    def toggle(self, ads_definition_id: str):
        trace_id = str(uuid.uuid4())
        response = self.rc.toggle_ads(ads_definition_id, trace_id)
        print(f"Ads toggled successfully:\n{response}")

    def get_descriptions(self) -> pd.DataFrame:
        response = self.rc.get_all_ads_descriptions()
        return pd.DataFrame(response["adsDescriptions"])

    def get_definitions(self) -> pd.DataFrame:
        response = self.rc.get_active_ads_definitions()
        return pd.DataFrame(response["adsDefinitions"])

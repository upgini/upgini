import time
from typing import Dict, Optional
import uuid
from upgini.http import get_rest_client
from upgini.spinner import Spinner


class AdsManager:

    FINAL_STATUSES = ["COMPLETED", "FAILED", "TIMED_OUT"]

    def __init__(self, api_key: str, backend_url: Optional[str] = None):
        self.api_key = api_key
        self.backend_url = backend_url

    def register(self, request: Dict) -> str:
        rc = get_rest_client(self.backend_url, self.api_key)
        trace_id = str(uuid.uuid4())
        ads_management_task_id = rc.register_ads(request, trace_id)
        print(f"Ads management task with id {ads_management_task_id} created")

        with Spinner():
            status_response = rc.poll_ads_management_task_status(ads_management_task_id, trace_id)
            while status_response["status"] not in self.FINAL_STATUSES:
                time.sleep(5)
                status_response = rc.poll_ads_management_task_status(ads_management_task_id, trace_id)

        if status_response["status"] != "COMPLETED":
            raise Exception("Failed to register ADS: " + status_response["errorMessage"])

        ads_definition_id = status_response["adsDefinitionId"]
        print("ADS successfully registered with id: " + ads_definition_id)
        return ads_definition_id

    def delete(self, ads_definition_id: str):
        rc = get_rest_client(self.backend_url, self.api_key)
        trace_id = str(uuid.uuid4())
        ads_management_task_id = rc.delete_ads(ads_definition_id, trace_id)
        print(f"Ads management task with id {ads_management_task_id} created")

        with Spinner():
            status_response = rc.poll_ads_management_task_status(ads_management_task_id, trace_id)
            while status_response["status"] not in self.FINAL_STATUSES:
                time.sleep(5)
                status_response = rc.poll_ads_management_task_status(ads_management_task_id, trace_id)

        if status_response["status"] != "COMPLETED":
            raise Exception("Failed to register ADS: " + status_response["errorMessage"])

        print("ADS successfully deleted")

    def toggle(self, ads_definition_id: str):
        rc = get_rest_client(self.backend_url, self.api_key)
        trace_id = str(uuid.uuid4())
        response = rc.toggle_ads(ads_definition_id, trace_id)
        print(f"Ads toggled successfully:\n{response}")

    def get_descriptions(self) -> Dict:
        rc = get_rest_client(self.backend_url, self.api_key)
        response = rc.get_all_ads_descriptions()
        return response

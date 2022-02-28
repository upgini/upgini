
import os
import sys
from getpass import getuser
from uuid import uuid4


_ide_env_variables = {
    "colab": ["GCS_READ_CACHE_BLOCK_SIZE_MB"],
    "binder": ["BINDER_PORT", "BINDER_SERVICE_PORT", "BINDER_REQUEST", "BINDER_REPO_URL", "BINDER_LAUNCH_HOST"],
    "kaggle": ["KAGGLE_DOCKER_IMAGE", "KAGGLE_URL_BASE"]
}


def _check_installed(package):
    result = None
    loc = locals()
    exec_check_str = (f"try: import {package};"
                      "result = True \n"
                      "except ModuleNotFoundError:"
                      "result = False \n"
                      "except: result=True"
                      )
    exec(exec_check_str, globals(), loc)
    return loc["result"]


def _env_contains(envs):
    return set(envs).issubset(set(os.environ.keys()))


def _get_execution_ide() -> str:
    if "google.colab" in sys.modules and _env_contains(_ide_env_variables["colab"]):
        return"colab"
    elif os.path.exists("/kaggle") and _check_installed("kaggle") and _env_contains(_ide_env_variables["kaggle"]):
        return "kaggle"
    elif getuser() == "jovyan" and _env_contains(_ide_env_variables["binder"]):
        return "binder"
    else:
        return "other"


def _get_client_uuid() -> str:
    client_uuid = os.environ.get("UPGINI_UUID")
    if client_uuid:
        return client_uuid
    else:
        client_uuid = str(uuid4())
        os.environ["UPGINI_UUID"] = client_uuid
        return client_uuid


def get_track_metrics() -> dict:
    return {
        "ide": _get_execution_ide()
    }

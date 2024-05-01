import os
import re
import socket
import sys
from functools import lru_cache
from getpass import getuser
from hashlib import sha256
from typing import Optional
from uuid import getnode

from requests import get, post

_ide_env_variables = {
    "colab": ["GCS_READ_CACHE_BLOCK_SIZE_MB"],
    "binder": ["BINDER_PORT", "BINDER_SERVICE_PORT", "BINDER_REQUEST", "BINDER_REPO_URL", "BINDER_LAUNCH_HOST"],
    "kaggle": ["KAGGLE_DOCKER_IMAGE", "KAGGLE_URL_BASE"],
}


def _check_installed(package):
    result = None
    loc = locals()
    exec_check_str = (
        f"try: import {package};"
        "result = True \n"
        "except ModuleNotFoundError:"
        "result = False \n"
        "except: result=True"
    )
    exec(exec_check_str, globals(), loc)
    return loc["result"]


def _env_contains(envs) -> bool:
    return set(envs).issubset(set(os.environ.keys()))


def _get_execution_ide() -> str:
    try:
        if "google.colab" in sys.modules and _env_contains(_ide_env_variables["colab"]):
            return "colab"
        elif os.path.exists("/kaggle") and _check_installed("kaggle") and _env_contains(_ide_env_variables["kaggle"]):
            return "kaggle"
        elif getuser() == "jovyan" and _env_contains(_ide_env_variables["binder"]):
            return "binder"
        elif "widget" in socket.gethostname():
            return "widget"
        else:
            return "other"
    except Exception:
        return "other"


@lru_cache
def get_track_metrics(client_ip: Optional[str] = None, client_visitorid: Optional[str] = None) -> dict:
    # default values
    track = {"ide": _get_execution_ide()}
    ident_res = "https://api64.ipify.org"

    try:
        track["hostname"] = socket.gethostname()
        track["whoami"] = getuser()
    except Exception as e:
        track["hostname"] = "localhost"
        track["whoami"] = "root"
        track["err"] = str(e)
    # get real info depending on ide

    if track["ide"] == "colab":
        try:
            from google.colab import output  # type: ignore
            from IPython.display import Javascript, display

            display(
                Javascript(
                    """
                    async function getVisitorId() {
                        return import('https://upgini.github.io/upgini/js/a.js')
                            .then(FingerprintJS => FingerprintJS.load())
                            .then(fp => fp.get())
                            .then(result => result.visitorId);
                    }
                    """
                )
            )
            track["visitorId"] = output.eval_js("getVisitorId()", timeout_sec=30)
        except Exception as e:
            track["err"] = str(e)
            if "visitorId" not in track:
                track["visitorId"] = "None"
        if client_ip:
            track["ip"] = client_ip
        else:
            try:
                from google.colab import output  # type: ignore
                from IPython.display import Javascript, display

                display(
                    Javascript(
                        f"""
                        async function getIP() {{
                            return fetch("{ident_res}")
                                .then(response => response.text())
                                .then(data => data);
                        }}
                        """
                    )
                )
                track["ip"] = output.eval_js("getIP()", timeout_sec=10)
            except Exception as e:
                track["err"] = str(e)
                if "ip" not in track:
                    track["ip"] = "0.0.0.0"

    elif track["ide"] == "binder":
        try:
            if "CLIENT_IP" in os.environ.keys():
                if client_ip:
                    track["ip"] = client_ip
                else:
                    track["ip"] = os.environ["CLIENT_IP"]
                track["visitorId"] = sha256(os.environ["CLIENT_IP"].encode()).hexdigest()
        except Exception as e:
            track["err"] = str(e)
            if "ip" not in track:
                track["ip"] = "0.0.0.0"
            if "visitorId" not in track:
                track["visitorId"] = "None"

    elif track["ide"] == "kaggle":
        try:
            url = "https://www.kaggle.com/requests/GetUserSecretByLabelRequest"
            jwt_token = os.getenv("KAGGLE_USER_SECRETS_TOKEN")
            headers = {
                "Content-type": "application/json",
                "X-Kaggle-Authorization": f"Bearer {jwt_token}",
            }
            with post(url, headers=headers, json={"Label": "api-key"}, timeout=10) as resp:
                err = resp.json()["errors"][0]
                match = re.search(".*\\s(\\d{5,})\\s.*", err)
                if match:
                    track["visitorId"] = match.group(1)
                else:
                    raise Exception(err)
        except Exception as e:
            track["err"] = str(e)
            if "visitorId" not in track:
                track["visitorId"] = "None"
    else:
        try:
            if client_ip:
                track["ip"] = client_ip
            else:
                track["ip"] = get(ident_res, timeout=10).text
            if client_visitorid:
                track["visitorId"] = client_visitorid
            else:
                track["visitorId"] = sha256(str(getnode()).encode()).hexdigest()
        except Exception as e:
            track["err"] = str(e)
            if "visitorId" not in track:
                track["visitorId"] = "None"
            if "ip" not in track:
                track["ip"] = "0.0.0.0"

    return track

from functools import lru_cache
import os
import re
import socket
import sys
from getpass import getuser
from hashlib import sha256
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
    if "google.colab" in sys.modules and _env_contains(_ide_env_variables["colab"]):
        return "colab"
    elif os.path.exists("/kaggle") and _check_installed("kaggle") and _env_contains(_ide_env_variables["kaggle"]):
        return "kaggle"
    elif getuser() == "jovyan" and _env_contains(_ide_env_variables["binder"]):
        return "binder"
    else:
        return "other"


@lru_cache()
def get_track_metrics() -> dict:
    # default values
    track = {"ide": _get_execution_ide()}
    ident_res = "https://api.ipify.org"

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

            # path_to_script = Path(__file__).parent.parent.resolve() / "fingerprint.js"
            # with open(path_to_script) as f:
            #     js_content = f.read()
            # print(f"JS loaded. Length: {len(js_content)}")

            display(
                Javascript(
                    # """
                    #     async function loadModuleFromString(code) {
                    #         const blob = new Blob([code], { type: 'application/javascript' });
                    #         const url = URL.createObjectURL(blob);
                    #         const module = await import(url);
                    #         URL.revokeObjectURL(url); // Clean URL-object after module load
                    #         return module;
                    #     }
                    #     window.visitorId = loadModuleFromString(""" + js_content + """)
                    """
                        window.visitorId = import('https://openfpcdn.io/fingerprintjs/v3')
                            .then(FingerprintJS => FingerprintJS.load())
                            .then(fp => fp.get())
                            .then(result => result.visitorId);
                    """
                )
            )
            track["visitorId"] = output.eval_js("visitorId", timeout_sec=10)
        except Exception as e:
            track["err"] = str(e)
            track["visitorId"] = "None"
        try:
            from google.colab import output  # type: ignore
            from IPython.display import Javascript, display

            display(
                Javascript(
                    f"""
                        window.clientIP = fetch("{ident_res}")
                            .then(response => response.text())
                            .then(data => data);
                    """
                )
            )
            track["ip"] = output.eval_js("window.clientIP", timeout_sec=10)
        except Exception as e:
            track["err"] = str(e)
            track["ip"] = "0.0.0.0"

    elif track["ide"] == "binder":
        try:
            if "CLIENT_IP" in os.environ.keys():
                track["ip"] = os.environ["CLIENT_IP"]
                track["visitorId"] = sha256(os.environ["CLIENT_IP"].encode()).hexdigest()
        except Exception as e:
            track["err"] = str(e)
            track["ip"] = "0.0.0.0"
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
            track["ip"] = "0.0.0.0"
            track["visitorId"] = "None"
    else:
        try:
            track["ip"] = get(ident_res, timeout=10).text
            track["visitorId"] = sha256(str(getnode()).encode()).hexdigest()
        except Exception as e:
            track["err"] = str(e)

    return track

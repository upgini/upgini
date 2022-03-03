
import os
import sys
from getpass import getuser
from uuid import uuid4
from requests import get


_ide_env_variables = {
    "colab": ["GCS_READ_CACHE_BLOCK_SIZE_MB"],
    "binder": ["BINDER_PORT", "BINDER_SERVICE_PORT", "BINDER_REQUEST", "BINDER_REPO_URL", "BINDER_LAUNCH_HOST"],
    "kaggle": ["KAGGLE_DOCKER_IMAGE", "KAGGLE_URL_BASE"]
}

_temp_file_track_var = "client_ip.dat"


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


def _push_temp_var(value):
    f = open(_temp_file_track_var, "w")
    f.write(value)
    f.close()


def _pull_temp_var():
    output_stream = os.popen("cat "+_temp_file_track_var)
    value = output_stream.read()
    os.remove(_temp_file_track_var)
    return value


def get_track_metrics() -> dict:
    track = {"ide": _get_execution_ide()}
    if track["ide"] == "colab":
        try:
            from IPython.display import display, Javascript
            from google.colab import output
            display(Javascript('''
                window.clientIP =
                fetch("https://api.ipify.org")
                .then(response => response.text())
                .then(data => data);
            '''))
            track["ip"] = output.eval_js("window.clientIP")
        except Exception as e:
            track["err"] = str(e)
    elif track["ide"] == "binder":
        try:
            from IPython.display import Javascript, display
            from time import sleep
            display(Javascript('''
                fetch('https://api.ipify.org')
                .then(response => response.text())
                .then(ip => IPython.notebook.kernel.execute('_push_temp_var("' + ip + '")'));
            '''))
            sleep(1)
            track["ip"] = _pull_temp_var()
        except Exception as e:
            track["err"] = str(e)
    else:
        try:
            track["ip"] = get("https://api.ipify.org").text
        except Exception as e:
            track["err"] = str(e)
    return track

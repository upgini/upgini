import pytest
from upgini.metadata import ModelTaskType, RuntimeParameters
from upgini.utils.custom_loss_utils import get_runtime_params_custom_loss, get_additional_params_custom_loss


@pytest.mark.parametrize(
    "input",
    [
        ("regression", ModelTaskType.REGRESSION, True),
        ("poisson", ModelTaskType.REGRESSION, True),
        ("binary", ModelTaskType.BINARY, True),
        ("binary", ModelTaskType.REGRESSION, False),
        ("multiclass", ModelTaskType.MULTICLASS, True),
        ("multiclass_ova", ModelTaskType.BINARY, False),
    ],
)
def test_get_runtime_params_custom_loss(input):
    loss, model_task_type, result = input
    runtime_parameters = RuntimeParameters()
    runtime_parameters.properties["key"] = "value"
    runtime_parameters = get_runtime_params_custom_loss(loss, model_task_type, runtime_parameters, logger=None)
    if result:
        assert runtime_parameters.properties == {
            "key": "value",
            "lightgbm_params_preselection.objective": loss,
            "lightgbm_params_base.objective": loss,
            "lightgbm_params_segment.objective": loss,
        }
    else:
        assert runtime_parameters.properties == {"key": "value"}


@pytest.mark.parametrize(
    "input",
    [
        ("regression", "RMSE", ModelTaskType.REGRESSION, True),
        ("poisson", "Poisson", ModelTaskType.REGRESSION, True),
        ("binary", "Logloss", ModelTaskType.BINARY, True),
        ("binary", "", ModelTaskType.REGRESSION, False),
        ("multiclass", "MultiClass", ModelTaskType.MULTICLASS, True),
        ("multiclass_ova", "MultiClassOneVsAll", ModelTaskType.BINARY, False),
    ],
)
def test_get_additional_params_custom_loss(input):
    loss, loss_cb, model_task_type, result = input
    params = get_additional_params_custom_loss(loss, model_task_type, logger=None)
    if result:
        assert params == {"loss_function": loss_cb}
    else:
        assert params == {}

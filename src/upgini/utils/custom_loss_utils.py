from upgini.metadata import ModelTaskType, RuntimeParameters
from typing import Optional, Dict, Any
import logging
from upgini.resource_bundle import bundle


def get_runtime_params_custom_loss(
    loss: str,
    model_task_type: ModelTaskType,
    runtime_parameters: RuntimeParameters,
    logger: Optional[logging.Logger] = None,
) -> RuntimeParameters:
    if logger is None:
        logger = logging.getLogger()
    if loss is not None:
        selection_loss_reg = [
            "regression",
            "regression_l1",
            "huber",
            "poisson",
            "quantile",
            "mape",
            "mean_absolute_percentage_error",
            "gamma",
            "tweedie",
        ]
        selection_loss_binary = ["binary"]
        selection_loss_multi_clf = ["multiclass", "multiclassova", "multiclass_ova", "ova", "ovr"]
        use_custom_loss = (
            True
            if (
                (model_task_type == ModelTaskType.REGRESSION)
                and (loss in selection_loss_reg)
                or (model_task_type == ModelTaskType.BINARY)
                and (loss in selection_loss_binary)
                or (model_task_type == ModelTaskType.MULTICLASS)
                and (loss in selection_loss_multi_clf)
            )
            else False
        )

        if use_custom_loss:
            runtime_parameters.properties["lightgbm_params_preselection.objective"] = loss
            runtime_parameters.properties["lightgbm_params_base.objective"] = loss
            runtime_parameters.properties["lightgbm_params_segment.objective"] = loss
            msg = bundle.get("loss_selection_info").format(loss)
            logger.info(msg)
            print(msg)
        else:
            msg = bundle.get("loss_selection_warn").format(loss, model_task_type)
            logger.warning(msg)
            print(msg)

    return runtime_parameters


def get_additional_params_custom_loss(
    loss: str, model_task_type: ModelTaskType, logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    if logger is None:
        logger = logging.getLogger()

    output_params = {}
    if loss is not None:
        calculation_loss_reg_map = {
            "regression": "RMSE",
            "poisson": "Poisson",
            "quantile": "Quantile",
            "mape": "MAPE",
            "mean_absolute_percentage_error": "MAPE",
        }
        calculation_loss_binary_map = {"binary": "Logloss"}
        calculation_loss_multi_clf_map = {
            "multiclass": "MultiClass",
            "multiclassova": "MultiClassOneVsAll",
            "multiclass_ova": "MultiClassOneVsAll",
            "ova": "MultiClassOneVsAll",
            "ovr": "MultiClass",
        }
        use_custom_loss = (
            True
            if (
                (model_task_type == ModelTaskType.REGRESSION)
                and (loss in calculation_loss_reg_map.keys())
                or (model_task_type == ModelTaskType.BINARY)
                and (loss in calculation_loss_binary_map.keys())
                or (model_task_type == ModelTaskType.MULTICLASS)
                and (loss in calculation_loss_multi_clf_map.keys())
            )
            else False
        )

        if use_custom_loss:
            if model_task_type == ModelTaskType.REGRESSION:
                output_params["loss_function"] = calculation_loss_reg_map[loss]
            elif model_task_type == ModelTaskType.BINARY:
                output_params["loss_function"] = calculation_loss_binary_map[loss]
            elif model_task_type == ModelTaskType.MULTICLASS:
                output_params["loss_function"] = calculation_loss_multi_clf_map[loss]

            msg = bundle.get("loss_calc_metrics_info").format(loss)
            logger.info(msg)
            print(msg)
        else:
            msg = bundle.get("loss_calc_metrics_warn").format(loss, model_task_type)
            logger.warning(msg)
            print(msg)

    return output_params

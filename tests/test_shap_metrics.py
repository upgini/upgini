import logging
import lightgbm as lgb
import numpy as np
import pandas as pd
from upgini.metadata import ModelTaskType
from upgini.metrics import EstimatorWrapper
from sklearn.ensemble import RandomForestClassifier


def test_calculate_shap_with_lgbm_classifier():
    rand = np.random.RandomState(42)
    X = pd.DataFrame({"feature1": rand.random(100), "feature2": rand.random(100)})
    y = rand.randint(0, 2, 100)

    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X, y)

    wrapper = EstimatorWrapper.create(
        estimator=model,
        logger=logging.getLogger(),
        target_type=ModelTaskType.BINARY,
        cv=None,
        scoring="roc_auc",
    )

    shap_values = wrapper.calculate_shap(X, y, model, cat_encoder=None)

    assert shap_values is not None
    assert isinstance(shap_values, dict)
    assert len(shap_values) == 2
    assert shap_values["feature1"] == 0.4719230673019846
    assert shap_values["feature2"] == 0.2999647327440229


def test_calculate_shap_with_lgbm_regressor():
    rand = np.random.RandomState(42)
    X = pd.DataFrame({"feature1": rand.random(100), "feature2": rand.random(100)})
    y = rand.random(100)

    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X, y)

    wrapper = EstimatorWrapper.create(
        estimator=model,
        logger=logging.getLogger(),
        target_type=ModelTaskType.REGRESSION,
        cv=None,
        scoring="roc_auc",
    )

    shap_values = wrapper.calculate_shap(X, y, model, cat_encoder=None)

    assert shap_values is not None
    assert isinstance(shap_values, dict)
    assert len(shap_values) == 2
    assert shap_values["feature1"] == 0.04838424947275008
    assert shap_values["feature2"] == 0.05082362023113759


def test_calculate_shap_with_non_lgbm_model():
    rand = np.random.RandomState(42)
    X = pd.DataFrame({"feature1": rand.random(100), "feature2": rand.random(100)})
    y = rand.randint(0, 2, 100)

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    wrapper = EstimatorWrapper.create(
        estimator=model,
        logger=logging.getLogger(),
        target_type=ModelTaskType.MULTICLASS,
        cv=None,
        scoring="roc_auc",
    )

    shap_values = wrapper.calculate_shap(X, y, model, cat_encoder=None)

    assert shap_values is None


def test_calculate_shap_with_empty_dataframe():
    X = pd.DataFrame()
    y = pd.Series()

    model = lgb.LGBMClassifier()

    wrapper = EstimatorWrapper.create(
        estimator=model,
        logger=logging.getLogger(),
        target_type=ModelTaskType.BINARY,
        cv=None,
        scoring="roc_auc",
    )

    shap_values = wrapper.calculate_shap(X, y, model, cat_encoder=None)

    assert shap_values is None


def test_calculate_shap_with_invalid_input():
    X = "invalid_input"
    y = None
    model = lgb.LGBMClassifier()

    wrapper = EstimatorWrapper.create(
        estimator=model,
        logger=logging.getLogger(),
        target_type=ModelTaskType.BINARY,
        cv=None,
        scoring="roc_auc",
    )

    shap_values = wrapper.calculate_shap(X, y, model, cat_encoder=None)

    assert shap_values is None

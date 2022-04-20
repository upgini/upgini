import pandas as pd
import pytest
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from upgini.utils.blocked_time_series import BlockedTimeSeriesSplit


def _prepare_data():
    df = pd.read_csv("tests/test_data/binary/data2.csv.gz", compression="gzip")
    X, y = df[["feature1", "feature2"]], df["target"]
    cv = BlockedTimeSeriesSplit(n_splits=5, test_size=0.25)
    df_etalon_stat = pd.read_csv("tests/test_data/binary/blocked_ts_logic.csv")
    model = LogisticRegression(random_state=0)

    return X, y, cv, df_etalon_stat, model


def test_bts_split_logic():
    X, y, cv, df_etalon_stat, _ = _prepare_data()
    df_stat = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        df_stat.append([len(train_idx), min(train_idx), max(train_idx), len(test_idx), min(test_idx), max(test_idx)])
    columns = ["train_len", "train_min", "train_max", "test_len", "test_min", "test_max"]
    df_stat = pd.DataFrame(df_stat, columns=columns)

    assert df_stat.equals(df_etalon_stat)


def test_bts_metrics():
    X, y, cv, _, model = _prepare_data()
    cv_result = set(cross_val_score(model, X, y, cv=cv, scoring="roc_auc"))
    assert cv_result == {
        0.4559664254320743,
        0.4767320313326982,
        0.4811855209016638,
        0.48947924927306374,
        0.5150543675843606,
    }


def test_bts_exceptions():
    X, y, cv, _, model = _prepare_data()

    X_short, y_short = X.iloc[:3, :], y.iloc[:3]
    with pytest.raises(ValueError, match=r"Cannot have number of splits=.* greater than the number of samples=.*"):
        _ = cross_val_score(model, X_short, y_short, cv=cv)

    X_short, y_short = X.iloc[:10, :], y.iloc[:10]
    with pytest.raises(ValueError, match=r"Cannot have number of samples in test fold .* <= 1"):
        _ = cross_val_score(model, X_short, y_short, cv=cv)

    with pytest.raises(
        ValueError, match=r"The number of folds must be of Integral type. .* of type <class 'float'> was passed."
    ):
        cv_err = BlockedTimeSeriesSplit(n_splits=5.5, test_size=0.2)
        _ = cross_val_score(model, X, y, cv=cv_err)

    with pytest.raises(
        ValueError,
        match=r"Cross-validation requires at least one train/test split by setting n_splits=2 or more, got n_splits=.*",
    ):
        cv_err = BlockedTimeSeriesSplit(n_splits=0, test_size=0.2)
        _ = cross_val_score(model, X, y, cv=cv_err)

    with pytest.raises(ValueError, match=r"test_size=.* should be a float in the .* range"):
        cv_err = BlockedTimeSeriesSplit(n_splits=5, test_size=2)
        _ = cross_val_score(model, X, y, cv=cv_err)
import functools
import logging
import numbers
import time
import warnings
from collections import Counter
from contextlib import suppress
from itertools import compress
from traceback import format_exc

import numpy as np
import scipy.sparse as sp
from catboost import CatBoostClassifier, CatBoostRegressor
from joblib import Parallel, logger
from scipy.sparse import issparse
from sklearn import config_context, get_config
from sklearn.base import clone, is_classifier
from sklearn.exceptions import FitFailedWarning, NotFittedError
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _MultimetricScorer
from sklearn.model_selection import check_cv
from sklearn.utils.fixes import np_version, parse_version
from sklearn.utils.validation import indexable

_DEFAULT_TAGS = {
    "non_deterministic": False,
    "requires_positive_X": False,
    "requires_positive_y": False,
    "X_types": ["2darray"],
    "poor_score": False,
    "no_validation": False,
    "multioutput": False,
    "allow_nan": False,
    "stateless": False,
    "multilabel": False,
    "_skip_test": False,
    "_xfail_checks": False,
    "multioutput_only": False,
    "binary_only": False,
    "requires_fit": True,
    "preserves_dtype": [np.float64],
    "requires_y": False,
    "pairwise": False,
}


def cross_validate(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    return_estimator=False,
    error_score=np.nan,
):
    """Evaluate metric(s) by cross-validation and also record fit/score times.

    Read more in the :ref:`User Guide <multimetric_cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).

    scoring : str, callable, list, tuple, or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        If `scoring` represents a single score, one can use:

        - a single string (see :ref:`scoring_parameter`);
        - a callable (see :ref:`scoring`) that returns a single value.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.

        See :ref:`multimetric_grid_search` for an example.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`.Fold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        The verbosity level.

    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    return_train_score : bool, default=False
        Whether to include train scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

        .. versionadded:: 0.19

        .. versionchanged:: 0.21
            Default value was changed from ``True`` to ``False``

    return_estimator : bool, default=False
        Whether to return the estimators fitted on each split.

        .. versionadded:: 0.20

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

        .. versionadded:: 0.20

    Returns
    -------
    scores : dict of float arrays of shape (n_splits,)
        Array of scores of the estimator for each run of the cross validation.

        A dict of arrays containing the score/time arrays for each scorer is
        returned. The possible keys for this ``dict`` are:

            ``test_score``
                The score array for test scores on each cv split.
                Suffix ``_score`` in ``test_score`` changes to a specific
                metric like ``test_r2`` or ``test_auc`` if there are
                multiple scoring metrics in the scoring parameter.
            ``train_score``
                The score array for train scores on each cv split.
                Suffix ``_score`` in ``train_score`` changes to a specific
                metric like ``train_r2`` or ``train_auc`` if there are
                multiple scoring metrics in the scoring parameter.
                This is available only if ``return_train_score`` parameter
                is ``True``.
            ``fit_time``
                The time for fitting the estimator on the train
                set for each cv split.
            ``score_time``
                The time for scoring the estimator on the test set for each
                cv split. (Note time for scoring on the train set is not
                included even if ``return_train_score`` is set to ``True``
            ``estimator``
                The estimator objects for each cv split.
                This is available only if ``return_estimator`` parameter
                is set to ``True``.

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_validate
    >>> from sklearn.metrics import make_scorer
    >>> from sklearn.metrics import confusion_matrix
    >>> from sklearn.svm import LinearSVC
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()

    Single metric evaluation using ``cross_validate``

    >>> cv_results = cross_validate(lasso, X, y, cv=3)
    >>> sorted(cv_results.keys())
    ['fit_time', 'score_time', 'test_score']
    >>> cv_results['test_score']
    array([0.33150734, 0.08022311, 0.03531764])

    Multiple metric evaluation using ``cross_validate``
    (please refer the ``scoring`` parameter doc for more information)

    >>> scores = cross_validate(lasso, X, y, cv=3,
    ...                         scoring=('r2', 'neg_mean_squared_error'),
    ...                         return_train_score=True)
    >>> print(scores['test_neg_mean_squared_error'])
    [-3635.5... -3573.3... -6114.7...]
    >>> print(scores['train_r2'])
    [0.28010158 0.39088426 0.22784852]

    See Also
    ---------
    cross_val_score : Run cross-validation for single metric evaluation.

    cross_val_predict : Get predictions from each split of cross-validation for
        diagnostic purposes.

    sklearn.metrics.make_scorer : Make a scorer from a performance metric or
        loss function.

    """
    try:
        X, y, groups = indexable(X, y, groups)

        cv = check_cv(cv, y, classifier=is_classifier(estimator))

        if callable(scoring):
            scorers = scoring
        elif scoring is None or isinstance(scoring, str):
            scorers = check_scoring(estimator, scoring)
        else:
            scorers = _check_multimetric_scoring(estimator, scoring)

        # We clone the estimator to make sure that all the folds are
        # independent, and that it is pickle-able.
        parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
        results = parallel(
            delayed(_fit_and_score)(
                clone(estimator),
                X,
                y,
                scorers,
                train,
                test,
                verbose,
                None,
                fit_params,
                return_train_score=return_train_score,
                return_times=True,
                return_estimator=return_estimator,
                error_score=error_score,
            )
            for train, test in cv.split(X, y, groups)
        )

        _warn_about_fit_failures(results, error_score)

        # For callabe scoring, the return type is only know after calling. If the
        # return type is a dictionary, the error scores can now be inserted with
        # the correct key.
        if callable(scoring):
            _insert_error_scores(results, error_score)

        results = _aggregate_score_dicts(results)

        ret = {}
        ret["fit_time"] = results["fit_time"]
        ret["score_time"] = results["score_time"]

        if return_estimator:
            ret["estimator"] = results["estimator"]

        test_scores_dict = _normalize_score_results(results["test_scores"])
        if return_train_score:
            train_scores_dict = _normalize_score_results(results["train_scores"])

        for name in test_scores_dict:
            ret["test_%s" % name] = test_scores_dict[name]
            if return_train_score:
                key = "train_%s" % name
                ret[key] = train_scores_dict[name]

        return ret
    except Exception:
        logging.exception("Failed to execute overriden cross_validate. Fallback to original")
        return cross_validate(
            estimator,
            X,
            y,
            groups=groups,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            fit_params=fit_params,
            pre_dispatch=pre_dispatch,
            return_train_score=return_train_score,
            return_estimator=return_estimator,
            error_score=error_score,
        )


def _fit_and_score(
    estimator,
    X,
    y,
    scorer,
    train,
    test,
    verbose,
    parameters,
    fit_params,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,
    split_progress=None,
    candidate_progress=None,
    error_score=np.nan,
):
    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.

    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.

        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.

        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.

    train : array-like of shape (n_train_samples,)
        Indices of training samples.

    test : array-like of shape (n_test_samples,)
        Indices of test samples.

    verbose : int
        The verbosity level.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

    parameters : dict or None
        Parameters to be set on the estimator.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    return_train_score : bool, default=False
        Compute and return score on training set.

    return_parameters : bool, default=False
        Return parameters that has been used for the estimator.

    split_progress : {list, tuple} of int, default=None
        A list or tuple of format (<current_split_id>, <total_num_of_splits>).

    candidate_progress : {list, tuple} of int, default=None
        A list or tuple of format
        (<current_candidate_id>, <total_number_of_candidates>).

    return_n_test_samples : bool, default=False
        Whether to return the ``n_test_samples``.

    return_times : bool, default=False
        Whether to return the fit/score times.

    return_estimator : bool, default=False
        Whether to return the fitted estimator.

    Returns
    -------
    result : dict with the following attributes
        train_scores : dict of scorer name -> float
            Score on training set (for all the scorers),
            returned only if `return_train_score` is `True`.
        test_scores : dict of scorer name -> float
            Score on testing set (for all the scorers).
        n_test_samples : int
            Number of test samples.
        fit_time : float
            Time spent for fitting in seconds.
        score_time : float
            Time spent for scoring in seconds.
        parameters : dict or None
            The parameters that have been evaluated.
        estimator : estimator object
            The fitted estimator.
        fit_error : str or None
            Traceback str if the fit failed, None if the fit succeeded.
    """

    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += f"; {candidate_progress[0]+1}/{candidate_progress[1]}"

    if verbose > 1:
        if parameters is None:
            params_msg = ""
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    result = {}
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            if isinstance(estimator, CatBoostClassifier) or isinstance(estimator, CatBoostRegressor):
                fit_params = fit_params.copy()
                fit_params["eval_set"] = [(X_test, y_test)]
            estimator.fit(X_train, y_train, **fit_params)

    except Exception:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
        result["fit_error"] = format_exc()
    else:
        result["fit_error"] = None

        fit_time = time.time() - start_time
        test_scores = _score(estimator, X_test, y_test, scorer, error_score)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(estimator, X_train, y_train, scorer, error_score)

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2:
            if isinstance(test_scores, dict):
                for scorer_name in sorted(test_scores):
                    result_msg += f" {scorer_name}: ("
                    if return_train_score:
                        scorer_scores = train_scores[scorer_name]
                        result_msg += f"train={scorer_scores:.3f}, "
                    result_msg += f"test={test_scores[scorer_name]:.3f})"
            else:
                result_msg += ", score="
                if return_train_score:
                    result_msg += f"(train={train_scores:.3f}, test={test_scores:.3f})"
                else:
                    result_msg += f"{test_scores:.3f}"
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    result["test_scores"] = test_scores
    if return_train_score:
        result["train_scores"] = train_scores
    if return_n_test_samples:
        result["n_test_samples"] = _num_samples(X_test)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    return result


def _aggregate_score_dicts(scores):
    """Aggregate the list of dict to dict of np ndarray

    The aggregated output of _aggregate_score_dicts will be a list of dict
    of form [{'prec': 0.1, 'acc':1.0}, {'prec': 0.1, 'acc':1.0}, ...]
    Convert it to a dict of array {'prec': np.array([0.1 ...]), ...}

    Parameters
    ----------

    scores : list of dict
        List of dicts of the scores for all scorers. This is a flat list,
        assumed originally to be of row major order.

    Example
    -------

    >>> scores = [{'a': 1, 'b':10}, {'a': 2, 'b':2}, {'a': 3, 'b':3},
    ...           {'a': 10, 'b': 10}]                         # doctest: +SKIP
    >>> _aggregate_score_dicts(scores)                        # doctest: +SKIP
    {'a': array([1, 2, 3, 10]),
     'b': array([10, 2, 3, 10])}
    """

    return {
        key: np.asarray([score[key] for score in scores])
        if isinstance(scores[0][key], numbers.Number)
        else [score[key] for score in scores]
        for key in scores[0]
    }


def _insert_error_scores(results, error_score):
    """Insert error in `results` by replacing them inplace with `error_score`.

    This only applies to multimetric scores because `_fit_and_score` will
    handle the single metric case.
    """

    successful_score = None
    failed_indices = []
    for i, result in enumerate(results):
        if result["fit_error"] is not None:
            failed_indices.append(i)
        elif successful_score is None:
            successful_score = result["test_scores"]

    if successful_score is None:
        raise NotFittedError("All estimators failed to fit")

    if isinstance(successful_score, dict):
        formatted_error = {name: error_score for name in successful_score}
        for i in failed_indices:
            results[i]["test_scores"] = formatted_error.copy()
            if "train_scores" in results[i]:
                results[i]["train_scores"] = formatted_error.copy()


def _warn_about_fit_failures(results, error_score):
    fit_errors = [result["fit_error"] for result in results if result["fit_error"] is not None]
    if fit_errors:
        num_failed_fits = len(fit_errors)
        num_fits = len(results)
        fit_errors_counter = Counter(fit_errors)
        delimiter = "-" * 80 + "\n"
        fit_errors_summary = "\n".join(
            f"{delimiter}{n} fits failed with the following error:\n{error}" for error, n in fit_errors_counter.items()
        )

        some_fits_failed_message = (
            f"\n{num_failed_fits} fits failed out of a total of {num_fits}.\n"
            "The score on these train-test partitions for these parameters"
            f" will be set to {error_score}.\n"
            "If these failures are not expected, you can try to debug them "
            "by setting error_score='raise'.\n\n"
            f"Below are more details about the failures:\n{fit_errors_summary}"
        )
        warnings.warn(some_fits_failed_message, FitFailedWarning, stacklevel=1)


def _normalize_score_results(scores, scaler_score_key="score"):
    """Creates a scoring dictionary based on the type of `scores`"""
    if isinstance(scores[0], dict):
        # multimetric scoring
        return _aggregate_score_dicts(scores)
    # scaler
    return {scaler_score_key: scores}


def _check_multimetric_scoring(estimator, scoring):
    """Check the scoring parameter in cases when multiple metrics are allowed.

    Parameters
    ----------
    estimator : sklearn estimator instance
        The estimator for which the scoring will be applied.

    scoring : list, tuple or dict
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        The possibilities are:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where they keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.

        See :ref:`multimetric_grid_search` for an example.

    Returns
    -------
    scorers_dict : dict
        A dict mapping each scorer name to its validated scorer.
    """
    err_msg_generic = (
        f"scoring is invalid (got {scoring!r}). Refer to the "
        "scoring glossary for details: "
        "https://scikit-learn.org/stable/glossary.html#term-scoring"
    )

    if isinstance(scoring, (list, tuple, set)):
        err_msg = "The list/tuple elements must be unique strings of predefined scorers. "
        try:
            keys = set(scoring)
        except TypeError as e:
            raise ValueError(err_msg) from e

        if len(keys) != len(scoring):
            raise ValueError(f"{err_msg} Duplicate elements were found in" f" the given list. {scoring!r}")
        elif len(keys) > 0:
            if not all(isinstance(k, str) for k in keys):
                if any(callable(k) for k in keys):
                    raise ValueError(
                        f"{err_msg} One or more of the elements "
                        "were callables. Use a dict of score "
                        "name mapped to the scorer callable. "
                        f"Got {scoring!r}"
                    )
                else:
                    raise ValueError(f"{err_msg} Non-string types were found " f"in the given list. Got {scoring!r}")
            scorers = {scorer: check_scoring(estimator, scoring=scorer) for scorer in scoring}
        else:
            raise ValueError(f"{err_msg} Empty list was given. {scoring!r}")

    elif isinstance(scoring, dict):
        keys = set(scoring)
        if not all(isinstance(k, str) for k in keys):
            raise ValueError("Non-string types were found in the keys of " f"the given dict. scoring={scoring!r}")
        if len(keys) == 0:
            raise ValueError(f"An empty dict was passed. {scoring!r}")
        scorers = {key: check_scoring(estimator, scoring=scorer) for key, scorer in scoring.items()}
    else:
        raise ValueError(err_msg_generic)
    return scorers


def _score(estimator, X_test, y_test, scorer, error_score="raise"):
    """Compute the score(s) of an estimator on a given test set.

    Will return a dict of floats if `scorer` is a dict, otherwise a single
    float is returned.
    """
    if isinstance(scorer, dict):
        # will cache method calls if needed. scorer() returns a dict
        scorer = _MultimetricScorer(**scorer)

    try:
        if y_test is None:
            scores = scorer(estimator, X_test)
        else:
            scores = scorer(estimator, X_test, y_test)
    except Exception:
        if error_score == "raise":
            raise
        else:
            if isinstance(scorer, _MultimetricScorer):
                scores = {name: error_score for name in scorer._scorers}
            else:
                scores = error_score
            warnings.warn(
                "Scoring failed. The score on this train-test partition for "
                f"these parameters will be set to {error_score}. Details: \n"
                f"{format_exc()}",
                UserWarning,
                stacklevel=1,
            )

    error_msg = "scoring must return a number, got %s (%s) instead. (scorer=%s)"
    if isinstance(scores, dict):
        for name, score in scores.items():
            if hasattr(score, "item"):
                with suppress(ValueError):
                    # e.g. unwrap memmapped scalars
                    score = score.item()
            if not isinstance(score, numbers.Number):
                raise ValueError(error_msg % (score, type(score), name))
            scores[name] = score
    else:  # scalar
        if hasattr(scores, "item"):
            with suppress(ValueError):
                # e.g. unwrap memmapped scalars
                scores = scores.item()
        if not isinstance(scores, numbers.Number):
            raise ValueError(error_msg % (scores, type(scores), scorer))
    return scores


def _safe_split(estimator, X, y, indices, train_indices=None):
    """Create subset of dataset and properly handle kernels.

    Slice X, y according to indices for cross-validation, but take care of
    precomputed kernel-matrices or pairwise affinities / distances.

    If ``estimator._pairwise is True``, X needs to be square and
    we slice rows and columns. If ``train_indices`` is not None,
    we slice rows using ``indices`` (assumed the test set) and columns
    using ``train_indices``, indicating the training set.

    .. deprecated:: 0.24

        The _pairwise attribute is deprecated in 0.24. From 1.1
        (renaming of 0.26) and onward, this function will check for the
        pairwise estimator tag.

    Labels y will always be indexed only along the first axis.

    Parameters
    ----------
    estimator : object
        Estimator to determine whether we should slice only rows or rows and
        columns.

    X : array-like, sparse matrix or iterable
        Data to be indexed. If ``estimator._pairwise is True``,
        this needs to be a square array-like or sparse matrix.

    y : array-like, sparse matrix or iterable
        Targets to be indexed.

    indices : array of int
        Rows to select from X and y.
        If ``estimator._pairwise is True`` and ``train_indices is None``
        then ``indices`` will also be used to slice columns.

    train_indices : array of int or None, default=None
        If ``estimator._pairwise is True`` and ``train_indices is not None``,
        then ``train_indices`` will be use to slice the columns of X.

    Returns
    -------
    X_subset : array-like, sparse matrix or list
        Indexed data.

    y_subset : array-like, sparse matrix or list
        Indexed targets.

    """
    if _is_pairwise(estimator):
        if not hasattr(X, "shape"):
            raise ValueError(
                "Precomputed kernels or affinity matrices have " "to be passed as arrays or sparse matrices."
            )
        # X is a precomputed square kernel matrix
        if X.shape[0] != X.shape[1]:
            raise ValueError("X should be a square kernel matrix")
        if train_indices is None:
            X_subset = X[np.ix_(indices, indices)]
        else:
            X_subset = X[np.ix_(indices, train_indices)]
    else:
        X_subset = _safe_indexing(X, indices)

    if y is not None:
        y_subset = _safe_indexing(y, indices)
    else:
        y_subset = None

    return X_subset, y_subset


def _is_pairwise(estimator):
    """Returns True if estimator is pairwise.

    - If the `_pairwise` attribute and the tag are present and consistent,
      then use the value and not issue a warning.
    - If the `_pairwise` attribute and the tag are present and not
      consistent, use the `_pairwise` value and issue a deprecation
      warning.
    - If only the `_pairwise` attribute is present and it is not False,
      issue a deprecation warning and use the `_pairwise` value.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if the estimator is pairwise and False otherwise.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        has_pairwise_attribute = hasattr(estimator, "_pairwise")
        pairwise_attribute = getattr(estimator, "_pairwise", False)
    pairwise_tag = _safe_tags(estimator, key="pairwise")

    if has_pairwise_attribute:
        if pairwise_attribute != pairwise_tag:
            warnings.warn(
                "_pairwise was deprecated in 0.24 and will be removed in 1.1 "
                "(renaming of 0.26). Set the estimator tags of your estimator "
                "instead",
                FutureWarning,
                stacklevel=1,
            )
        return pairwise_attribute

    # use pairwise tag when the attribute is not present
    return pairwise_tag


def _safe_tags(estimator, key=None):
    """Safely get estimator tags.

    :class:`~sklearn.BaseEstimator` provides the estimator tags machinery.
    However, if an estimator does not inherit from this base class, we should
    fall-back to the default tags.

    For scikit-learn built-in estimators, we should still rely on
    `self._get_tags()`. `_safe_tags(est)` should be used when we are not sure
    where `est` comes from: typically `_safe_tags(self.base_estimator)` where
    `self` is a meta-estimator, or in the common checks.

    Parameters
    ----------
    estimator : estimator object
        The estimator from which to get the tag.

    key : str, default=None
        Tag name to get. By default (`None`), all tags are returned.

    Returns
    -------
    tags : dict or tag value
        The estimator tags. A single value is returned if `key` is not None.
    """
    if hasattr(estimator, "_get_tags"):
        tags_provider = "_get_tags()"
        tags = estimator._get_tags()
    elif hasattr(estimator, "_more_tags"):
        tags_provider = "_more_tags()"
        tags = {**_DEFAULT_TAGS, **estimator._more_tags()}
    else:
        tags_provider = "_DEFAULT_TAGS"
        tags = _DEFAULT_TAGS

    if key is not None:
        if key not in tags:
            raise ValueError(
                f"The key {key} is not defined in {tags_provider} for the " f"class {estimator.__class__.__name__}."
            )
        return tags[key]
    return tags


def _safe_indexing(X, indices, *, axis=0):
    """Return rows, items or columns of X using indices.

    .. warning::

        This utility is documented, but **private**. This means that
        backward compatibility might be broken without any deprecation
        cycle.

    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series
        Data from which to sample rows, items or columns. `list` are only
        supported when `axis=0`.
    indices : bool, int, str, slice, array-like
        - If `axis=0`, boolean and integer array-like, integer slice,
          and scalar integer are supported.
        - If `axis=1`:
            - to select a single column, `indices` can be of `int` type for
              all `X` types and `str` only for dataframe. The selected subset
              will be 1D, unless `X` is a sparse matrix in which case it will
              be 2D.
            - to select multiples columns, `indices` can be one of the
              following: `list`, `array`, `slice`. The type used in
              these containers can be one of the following: `int`, 'bool' and
              `str`. However, `str` is only supported when `X` is a dataframe.
              The selected subset will be 2D.
    axis : int, default=0
        The axis along which `X` will be subsampled. `axis=0` will select
        rows while `axis=1` will select columns.

    Returns
    -------
    subset
        Subset of X on axis 0 or 1.

    Notes
    -----
    CSR, CSC, and LIL sparse matrices are supported. COO sparse matrices are
    not supported.
    """
    if indices is None:
        return X

    if axis not in (0, 1):
        raise ValueError(
            "'axis' should be either 0 (to index rows) or 1 (to index " " column). Got {} instead.".format(axis)
        )

    indices_dtype = _determine_key_type(indices)

    if axis == 0 and indices_dtype == "str":
        raise ValueError("String indexing is not supported with 'axis=0'")

    if axis == 1 and X.ndim != 2:
        raise ValueError(
            "'X' should be a 2D NumPy array, 2D sparse matrix or pandas "
            "dataframe when indexing the columns (i.e. 'axis=1'). "
            "Got {} instead with {} dimension(s).".format(type(X), X.ndim)
        )

    if axis == 1 and indices_dtype == "str" and not hasattr(X, "loc"):
        raise ValueError("Specifying the columns using strings is only supported for " "pandas DataFrames")

    if hasattr(X, "iloc"):
        return _pandas_indexing(X, indices, indices_dtype, axis=axis)
    elif hasattr(X, "shape"):
        return _array_indexing(X, indices, indices_dtype, axis=axis)
    else:
        return _list_indexing(X, indices, indices_dtype)


def _array_indexing(array, key, key_dtype, axis):
    """Index an array or scipy.sparse consistently across NumPy version."""
    if np_version < parse_version("1.12") or issparse(array):
        # FIXME: Remove the check for NumPy when using >= 1.12
        # check if we have an boolean array-likes to make the proper indexing
        if key_dtype == "bool":
            key = np.asarray(key)
    if isinstance(key, tuple):
        key = list(key)
    return array[key] if axis == 0 else array[:, key]


def _pandas_indexing(X, key, key_dtype, axis):
    """Index a pandas dataframe or a series."""
    if hasattr(key, "shape"):
        # Work-around for indexing with read-only key in pandas
        # FIXME: solved in pandas 0.25
        key = np.asarray(key)
        key = key if key.flags.writeable else key.copy()
    elif isinstance(key, tuple):
        key = list(key)

    if key_dtype == "int" and not (isinstance(key, slice) or np.isscalar(key)):
        # using take() instead of iloc[] ensures the return value is a "proper"
        # copy that will not raise SettingWithCopyWarning
        return X.take(key, axis=axis)
    else:
        # check whether we should index with loc or iloc
        indexer = X.iloc if key_dtype == "int" else X.loc
        return indexer[:, key] if axis else indexer[key]


def _list_indexing(X, key, key_dtype):
    """Index a Python list."""
    if np.isscalar(key) or isinstance(key, slice):
        # key is a slice or a scalar
        return X[key]
    if key_dtype == "bool":
        # key is a boolean array-like
        return list(compress(X, key))
    # key is a integer array-like of key
    return [X[idx] for idx in key]


def _determine_key_type(key, accept_slice=True):
    """Determine the data type of key.

    Parameters
    ----------
    key : scalar, slice or array-like
        The key from which we want to infer the data type.

    accept_slice : bool, default=True
        Whether or not to raise an error if the key is a slice.

    Returns
    -------
    dtype : {'int', 'str', 'bool', None}
        Returns the data type of key.
    """
    err_msg = (
        "No valid specification of the columns. Only a scalar, list or "
        "slice of all integers or all strings, or boolean mask is "
        "allowed"
    )

    dtype_to_str = {int: "int", str: "str", bool: "bool", np.bool_: "bool"}
    array_dtype_to_str = {
        "i": "int",
        "u": "int",
        "b": "bool",
        "O": "str",
        "U": "str",
        "S": "str",
    }

    if key is None:
        return None
    if isinstance(key, tuple(dtype_to_str.keys())):
        try:
            return dtype_to_str[type(key)]
        except KeyError:
            raise ValueError(err_msg)
    if isinstance(key, slice):
        if not accept_slice:
            raise TypeError("Only array-like or scalar are supported. A Python slice was given.")
        if key.start is None and key.stop is None:
            return None
        key_start_type = _determine_key_type(key.start)
        key_stop_type = _determine_key_type(key.stop)
        if key_start_type is not None and key_stop_type is not None:
            if key_start_type != key_stop_type:
                raise ValueError(err_msg)
        if key_start_type is not None:
            return key_start_type
        return key_stop_type
    if isinstance(key, (list, tuple)):
        unique_key = set(key)
        key_type = {_determine_key_type(elt) for elt in unique_key}
        if not key_type:
            return None
        if len(key_type) != 1:
            raise ValueError(err_msg)
        return key_type.pop()
    if hasattr(key, "dtype"):
        try:
            return array_dtype_to_str[key.dtype.kind]
        except KeyError:
            raise ValueError(err_msg)
    raise ValueError(err_msg)


# remove when https://github.com/joblib/joblib/issues/1071 is fixed
def delayed(function):
    """Decorator used to capture the arguments of a function."""

    @functools.wraps(function)
    def delayed_function(*args, **kwargs):
        return _FuncWrapper(function), args, kwargs

    return delayed_function


class _FuncWrapper:
    """ "Load the global configuration before calling the function."""

    def __init__(self, function):
        self.function = function
        self.config = get_config()
        functools.update_wrapper(self, self.function)

    def __call__(self, *args, **kwargs):
        with config_context(**self.config):
            return self.function(*args, **kwargs)


def _check_fit_params(X, fit_params, indices=None):
    """Check and validate the parameters passed during `fit`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data array.

    fit_params : dict
        Dictionary containing the parameters passed at fit.

    indices : array-like of shape (n_samples,), default=None
        Indices to be selected if the parameter has the same size as `X`.

    Returns
    -------
    fit_params_validated : dict
        Validated parameters. We ensure that the values support indexing.
    """

    fit_params_validated = {}
    for param_key, param_value in fit_params.items():
        if not _is_arraylike(param_value) or _num_samples(param_value) != _num_samples(X):
            # Non-indexable pass-through (for now for backward-compatibility).
            # https://github.com/scikit-learn/scikit-learn/issues/15805
            fit_params_validated[param_key] = param_value
        else:
            # Any other fit_params should support indexing
            # (e.g. for cross-validation).
            fit_params_validated[param_key] = _make_indexable(param_value)
            fit_params_validated[param_key] = _safe_indexing(fit_params_validated[param_key], indices)

    return fit_params_validated


def _is_arraylike(x):
    """Returns whether the input is array-like."""
    return hasattr(x, "__len__") or hasattr(x, "shape") or hasattr(x, "__array__")


def _make_indexable(iterable):
    """Ensure iterable supports indexing or convert to an indexable variant.

    Convert sparse matrices to csr and other non-indexable iterable to arrays.
    Let `None` and indexable objects (e.g. pandas dataframes) pass unchanged.

    Parameters
    ----------
    iterable : {list, dataframe, ndarray, sparse matrix} or None
        Object to be converted to an indexable iterable.
    """
    if sp.issparse(iterable):
        return iterable.tocsr()
    elif hasattr(iterable, "__getitem__") or hasattr(iterable, "iloc"):
        return iterable
    elif iterable is None:
        return iterable
    return np.array(iterable)


def _num_samples(x):
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered a valid collection." % x)
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error

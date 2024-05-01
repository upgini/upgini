import numbers

import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

from upgini.resource_bundle import bundle


class BlockedTimeSeriesSplit(BaseCrossValidator):
    """Blocked Time Series cross-validator

    This cross-validation object is a modification of TimeSeriesSplit.
    In the kth split, it splits kth fold to train and test by time.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.

    test_size : float, default=0.2
        Should be between 0.0 and 1.0 and represent the proportion
        of groups to include in the test split
    """

    def __init__(self, n_splits: int = 5, test_size: float = 0.2):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError(bundle.get("timeseries_invalid_split_type").format(n_splits, type(n_splits)))
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(bundle.get("timeseries_invalid_split_count").format(n_splits))

        if test_size <= 0 or test_size >= 1:
            raise ValueError(bundle.get("timeseries_invalid_test_size_type").format(test_size))

        self.n_splits = n_splits
        self.test_size = test_size

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)  # type: ignore
        n_samples = _num_samples(X)
        fold_size = n_samples // self.n_splits

        if self.n_splits > n_samples:
            raise ValueError(bundle.get("timeseries_splits_more_than_samples").format(self.n_splits, n_samples))
        if self.test_size * fold_size <= 1:
            raise ValueError(bundle.get("timeseries_invalid_test_size"))

        indices = np.arange(n_samples)
        for i in range(self.n_splits):
            train_start = i * fold_size
            test_stop = train_start + fold_size
            test_start = int((1 - self.test_size) * (test_stop - train_start)) + train_start
            yield indices[train_start:test_start], indices[test_start:test_stop]

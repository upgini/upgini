import warnings
from collections import namedtuple

import numpy as np
import numpy.ma as ma
from joblib import Parallel, delayed
from numpy import ndarray
from psutil import cpu_count

np.seterr(divide="ignore")


warnings.simplefilter(action="ignore", category=RuntimeWarning)


def _find_repeats(arr):
    # This function assumes it may clobber its input.
    if len(arr) == 0:
        return np.array(0, np.float64), np.array(0, np.intp)

    # XXX This cast was previously needed for the Fortran implementation,
    # should we ditch it?
    arr = np.asarray(arr, np.float64).ravel()
    arr.sort()

    # Taken from NumPy 1.9's np.unique.
    change = np.concatenate(([True], arr[1:] != arr[:-1]))
    unique = arr[change]
    change_idx = np.concatenate(np.nonzero(change) + ([arr.size],))
    freq = np.diff(change_idx)
    atleast2 = freq > 1
    return unique[atleast2], freq[atleast2]


def find_repeats(arr):
    # Make sure we get a copy. ma.compressed promises a "new array", but can
    # actually return a reference.
    compr = np.asarray(ma.compressed(arr), dtype=np.float64)
    try:
        need_copy = np.may_share_memory(compr, arr)
    except AttributeError:
        # numpy < 1.8.2 bug: np.may_share_memory([], []) raises,
        # while in numpy 1.8.2 and above it just (correctly) returns False.
        need_copy = False
    if need_copy:
        compr = compr.copy()
    return _find_repeats(compr)


def rankdata(data, axis=None, use_missing=False):
    def _rank1d(data, use_missing=False):
        n = data.count()
        rk = np.empty(data.size, dtype=float)
        idx = data.argsort()
        rk[idx[:n]] = np.arange(1, n + 1)

        if use_missing:
            rk[idx[n:]] = (n + 1) / 2.0
        else:
            rk[idx[n:]] = 0

        repeats = find_repeats(data.copy())
        for r in repeats[0]:
            condition = (data == r).filled(False)
            rk[condition] = rk[condition].mean()
        return rk

    data = ma.array(data, copy=False)
    if axis is None:
        if data.ndim > 1:
            return _rank1d(data.ravel(), use_missing).reshape(data.shape)
        else:
            return _rank1d(data, use_missing)
    else:
        return ma.apply_along_axis(_rank1d, axis, data, use_missing).view(ndarray)


def _chk_asarray(a, axis):
    # Always returns a masked array, raveled for axis=None
    a = ma.asanyarray(a)
    if axis is None:
        a = ma.ravel(a)
        outaxis = 0
    else:
        outaxis = axis
    return a, outaxis


SpearmanrResult = namedtuple("SpearmanrResult", ("correlation", "pvalue"))


# Taken from scipy.mstats with following tweaks:
# 1. parallel pairwise computation
# 2. custom masking
def spearmanr(
    x, y=None, use_ties=True, axis=None, nan_policy="propagate", alternative="two-sided", mask_fn=ma.masked_invalid
):
    if not use_ties:
        raise ValueError("`use_ties=False` is not supported in SciPy >= 1.2.0")

    # Always returns a masked array, raveled if axis=None
    x, axisout = _chk_asarray(x, axis)
    if y is not None:
        # Deal only with 2-D `x` case.
        y, _ = _chk_asarray(y, axis)
        if axisout == 0:
            x = ma.column_stack((x, y))
        else:
            x = ma.row_stack((x, y))

    if axisout == 1:
        # To simplify the code that follow (always use `n_obs, n_vars` shape)
        x = x.T

    if nan_policy == "omit":
        x = mask_fn(x)

    # - dof: degrees of freedom
    # - t_stat: t-statistic
    # - alternative: 'two-sided', 'greater', 'less'
    def compute_t_pvalue(t_stat, dof, alternative="two-sided"):
        from scipy.stats import t

        if alternative == "two-sided":
            prob = 2 * t.sf(abs(t_stat), dof)
        elif alternative == "greater":
            prob = t.sf(t_stat, dof)
        elif alternative == "less":
            prob = t.cdf(t_stat, dof)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
        return t_stat, prob

    def _spearmanr_2cols(x):
        # Mask the same observations for all variables, and then drop those
        # observations (can't leave them masked, rankdata is weird).
        x = ma.mask_rowcols(x, axis=0)
        x = x[~x.mask.any(axis=1), :]

        # If either column is entirely NaN or Inf
        if not np.any(x.data):
            return SpearmanrResult(np.nan, np.nan)

        m = ma.getmask(x)
        n_obs = x.shape[0]
        dof = n_obs - 2 - int(m.sum(axis=0)[0])
        if dof < 0:
            return SpearmanrResult(np.nan, np.nan)

        # Gets the ranks and rank differences
        x_ranked = rankdata(x, axis=0)
        rs = ma.corrcoef(x_ranked, rowvar=False).data

        # rs can have elements equal to 1, so avoid zero division warnings
        with np.errstate(divide="ignore"):
            # clip the small negative values possibly caused by rounding
            # errors before taking the square root
            t = rs * np.sqrt((dof / ((rs + 1.0) * (1.0 - rs))).clip(0))

        t, prob = compute_t_pvalue(dof, t, alternative)

        # For backwards compatibility, return scalars when comparing 2 columns
        if rs.shape == (2, 2):
            return SpearmanrResult(rs[1, 0], prob[1, 0])
        else:
            return SpearmanrResult(rs, prob)

    # Need to do this per pair of variables, otherwise the dropped observations
    # in a third column mess up the result for a pair.
    n_vars = x.shape[1]
    if n_vars == 2:
        return _spearmanr_2cols(x)
    else:
        max_cpu_cores = cpu_count(logical=False)
        with np.errstate(divide="ignore"):
            results = Parallel(n_jobs=max_cpu_cores)(
                delayed(_spearmanr_2cols)(x[:, [var1, var2]])
                for var1 in range(n_vars - 1)
                for var2 in range(var1 + 1, n_vars)
            )

        rs = np.ones((n_vars, n_vars), dtype=float)
        prob = np.zeros((n_vars, n_vars), dtype=float)
        for var1 in range(n_vars - 1):
            for var2 in range(var1 + 1, n_vars):
                result = results.pop(0)
                rs[var1, var2] = result.correlation
                rs[var2, var1] = result.correlation
                prob[var1, var2] = result.pvalue
                prob[var2, var1] = result.pvalue

        return SpearmanrResult(rs, prob)

"""
Data transformation utilities for tail risk modeling.
"""

import numpy as np
import pandas as pd


def log_transform_target(y, inverse=False):
    """
    Apply log1p transformation to stabilize variance in heavy-tailed data.

    Parameters
    ----------
    y : array-like
        Target values to transform.

    inverse : bool, default=False
        If True, apply inverse transform (expm1).

    Returns
    -------
    y_transformed : ndarray
        Transformed values.

    Examples
    --------
    >>> y_log = log_transform_target(y_train)
    >>> # ... train model on y_log ...
    >>> y_pred_original = log_transform_target(y_pred_log, inverse=True)
    """
    y = np.asarray(y)

    if inverse:
        return np.expm1(y)
    else:
        return np.log1p(y)


def filter_nonzero_claims(X, y, return_indices=False):
    """
    Filter dataset to only include non-zero claims.

    Useful when modeling claim severity (given a claim occurred),
    rather than overall claim cost including zeros.

    Parameters
    ----------
    X : array-like or DataFrame
        Feature matrix.

    y : array-like
        Target values (claim costs).

    return_indices : bool, default=False
        If True, also return the boolean mask of selected indices.

    Returns
    -------
    X_filtered : array-like or DataFrame
        Features for non-zero claims only.

    y_filtered : ndarray
        Non-zero claim values.

    indices : ndarray (optional)
        Boolean mask of selected rows (if return_indices=True).

    Examples
    --------
    >>> X_nz, y_nz = filter_nonzero_claims(X, y)
    >>> print(f"Filtered from {len(y)} to {len(y_nz)} non-zero claims")
    """
    y = np.asarray(y)
    nonzero_mask = y > 0

    if isinstance(X, pd.DataFrame):
        X_filtered = X[nonzero_mask]
    else:
        X_filtered = np.asarray(X)[nonzero_mask]

    y_filtered = y[nonzero_mask]

    if return_indices:
        return X_filtered, y_filtered, nonzero_mask
    else:
        return X_filtered, y_filtered

"""
Tail risk evaluation metrics.

This module provides metrics specifically designed for evaluating
model performance on extreme tail events.
"""

import numpy as np
from sklearn.metrics import mean_squared_error


def loss_at_risk(y_true, y_pred, alpha=0.5):
    """
    Compute Loss-at-Risk (LaR) metric.

    LaR applies higher weights to errors on larger actual values,
    making it more sensitive to tail risk performance.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.

    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    alpha : float, default=0.5
        Weight scaling factor. Higher values increase sensitivity to large claims.

    Returns
    -------
    lar : float
        The weighted mean squared error.

    Examples
    --------
    >>> from tailrisk.metrics import loss_at_risk
    >>> y_true = np.array([100, 200, 10000])  # Last is extreme
    >>> y_pred = np.array([120, 180, 5000])
    >>> lar = loss_at_risk(y_true, y_pred, alpha=2.0)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_max = np.max(y_true)
    if y_max == 0:
        y_max = 1

    weights = 1 + alpha * (y_true / y_max)
    weighted_mse = np.mean(weights * (y_true - y_pred) ** 2)

    return weighted_mse


def cvar_loss(y_true, y_pred, alpha=0.95):
    """
    Compute Conditional Value-at-Risk (CVaR) loss.

    CVaR measures the average squared error in the worst (1-alpha)% of predictions,
    focusing on tail risk performance.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.

    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    alpha : float, default=0.95
        Quantile threshold. CVaR focuses on errors above this quantile.

    Returns
    -------
    cvar : float
        Mean squared error in the tail (above VaR threshold).

    Examples
    --------
    >>> from tailrisk.metrics import cvar_loss
    >>> y_true = np.random.exponential(1000, size=1000)
    >>> y_pred = model.predict(X)
    >>> cvar_95 = cvar_loss(y_true, y_pred, alpha=0.95)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    residuals = y_true - y_pred
    var_threshold = np.quantile(residuals, alpha)
    tail_residuals = residuals[residuals >= var_threshold]

    return np.mean(tail_residuals ** 2)


def tail_coverage_ratio(y_true, y_pred, quantile=0.99):
    """
    Compute Tail Coverage Ratio (TCR).

    TCR measures what fraction of the total extreme tail value
    is captured by predictions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.

    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    quantile : float, default=0.99
        Quantile threshold defining the "extreme tail".

    Returns
    -------
    tcr : float
        Ratio of predicted sum to actual sum in the tail.
        1.0 = perfect coverage, <1.0 = underprediction, >1.0 = overprediction.

    Examples
    --------
    >>> from tailrisk.metrics import tail_coverage_ratio
    >>> tcr_99 = tail_coverage_ratio(y_test, y_pred, quantile=0.99)
    >>> print(f"Model captures {tcr_99*100:.1f}% of extreme tail value")
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    threshold = np.quantile(y_true, quantile)
    tail_mask = y_true >= threshold

    if tail_mask.sum() == 0:
        return np.nan

    actual_sum = y_true[tail_mask].sum()
    if actual_sum == 0:
        return np.nan

    pred_sum = y_pred[tail_mask].sum()

    return pred_sum / actual_sum


def detection_rate(y_true, y_pred, quantile=0.95):
    """
    Compute detection rate for extreme claims.

    Measures what percentage of actual extreme claims (above quantile)
    are correctly predicted as extreme.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.

    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    quantile : float, default=0.95
        Quantile threshold defining "extreme".

    Returns
    -------
    detection : float
        Fraction of extreme actuals that were predicted as extreme (0.0 to 1.0).

    Examples
    --------
    >>> from tailrisk.metrics import detection_rate
    >>> det_95 = detection_rate(y_test, y_pred, quantile=0.95)
    >>> print(f"Model detects {det_95*100:.1f}% of extreme claims")
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    threshold = np.quantile(y_true, quantile)

    actual_extreme = y_true >= threshold
    pred_extreme = y_pred >= threshold

    if actual_extreme.sum() == 0:
        return np.nan

    correctly_detected = np.sum(actual_extreme & pred_extreme)

    return correctly_detected / actual_extreme.sum()


def tail_validation_summary(y_true, y_pred, quantiles=[0.90, 0.95, 0.99]):
    """
    Generate comprehensive tail validation report.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.

    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    quantiles : list of float, default=[0.90, 0.95, 0.99]
        Quantile thresholds to evaluate.

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - mse_overall: Overall MSE
        - mse_extreme: MSE on top 1% of claims
        - cvar_95: CVaR at 95%
        - detection_90, detection_95: Detection rates
        - tcr_95, tcr_99: Tail coverage ratios

    Examples
    --------
    >>> from tailrisk.metrics import tail_validation_summary
    >>> metrics = tail_validation_summary(y_test, y_pred)
    >>> print(f"CVaR(95%): {metrics['cvar_95']:.2f}")
    >>> print(f"Detection@95%: {metrics['detection_95']*100:.1f}%")
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Basic MSE
    mse_overall = mean_squared_error(y_true, y_pred)

    # Extreme tail MSE (99th percentile)
    q99_threshold = np.quantile(y_true, 0.99)
    extreme_mask = y_true >= q99_threshold
    mse_extreme = mean_squared_error(y_true[extreme_mask], y_pred[extreme_mask]) if extreme_mask.sum() > 0 else np.nan

    # CVaR
    cvar_95 = cvar_loss(y_true, y_pred, alpha=0.95)

    # Detection rates
    det_90 = detection_rate(y_true, y_pred, quantile=0.90)
    det_95 = detection_rate(y_true, y_pred, quantile=0.95)

    # Tail coverage ratios
    tcr_95 = tail_coverage_ratio(y_true, y_pred, quantile=0.95)
    tcr_99 = tail_coverage_ratio(y_true, y_pred, quantile=0.99)

    return {
        'mse_overall': mse_overall,
        'mse_extreme': mse_extreme,
        'cvar_95': cvar_95,
        'detection_90': det_90,
        'detection_95': det_95,
        'tcr_95': tcr_95,
        'tcr_99': tcr_99,
    }

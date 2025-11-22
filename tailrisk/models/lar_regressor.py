"""
Loss-at-Risk (LaR) Regressor

A weighted regression model that assigns higher importance to larger claims,
making it more sensitive to tail risk prediction.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class LaRRegressor(BaseEstimator, RegressorMixin):
    """
    Loss-at-Risk Weighted Regression.

    This model applies sample weights proportional to the target value,
    giving more importance to larger claims during training.

    Parameters
    ----------
    alpha : float, default=2.0
        Weight scaling factor. Higher values increase focus on large claims.
        Weight formula: w = 1 + alpha * (y / max(y))

    base_estimator : estimator object, default=None
        The base regression model to use. If None, uses LinearRegression().
        Must support sample_weight parameter in fit().

    Attributes
    ----------
    base_estimator_ : estimator
        The fitted base estimator.

    Examples
    --------
    >>> from tailrisk import LaRRegressor
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.exponential(scale=1000, size=100)  # Heavy-tailed
    >>> model = LaRRegressor(alpha=2.0)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    """

    def __init__(self, alpha=2.0, base_estimator=None):
        self.alpha = alpha
        self.base_estimator = base_estimator

    def fit(self, X, y, sample_weight=None):
        """
        Fit the LaR-weighted regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Additional sample weights (will be multiplied with LaR weights).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)

        # Calculate LaR weights
        y_max = np.max(y)
        if y_max == 0:
            y_max = 1  # Avoid division by zero

        lar_weights = 1 + self.alpha * (y / y_max)

        # Combine with additional sample weights if provided
        if sample_weight is not None:
            weights = lar_weights * sample_weight
        else:
            weights = lar_weights

        # Initialize base estimator if needed
        if self.base_estimator is None:
            self.base_estimator_ = LinearRegression()
        else:
            self.base_estimator_ = self.base_estimator

        # Fit with weights
        self.base_estimator_.fit(X, y, sample_weight=weights)

        return self

    def predict(self, X):
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, 'base_estimator_')
        X = check_array(X)

        return self.base_estimator_.predict(X)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'alpha': self.alpha,
            'base_estimator': self.base_estimator
        }

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

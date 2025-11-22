"""
CVaR-Weighted Ensemble

An ensemble that weights base models inversely proportional to their CVaR,
prioritizing models that perform better on tail risks.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def _cvar_from_residuals(residuals, alpha=0.95):
    """Compute CVaR from residuals."""
    var_threshold = np.quantile(residuals, alpha)
    tail_residuals = residuals[residuals >= var_threshold]
    return tail_residuals.mean() if len(tail_residuals) > 0 else np.inf


class CVaRWeightedEnsemble(BaseEstimator, RegressorMixin):
    """
    Ensemble that weights models by inverse CVaR performance.

    This ensemble trains multiple base models and combines their predictions
    using weights inversely proportional to their CVaR scores, giving more
    weight to models that perform better on extreme tail events.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        List of (name, estimator) pairs of base models to ensemble.

    alpha : float, default=0.95
        CVaR quantile threshold for weight calculation.

    Attributes
    ----------
    estimators_ : list of estimators
        Fitted base estimators.

    weights_ : ndarray of shape (n_estimators,)
        Weights assigned to each base estimator (sum to 1.0).

    cvar_scores_ : ndarray of shape (n_estimators,)
        CVaR score for each base estimator.

    Examples
    --------
    >>> from tailrisk import CVaRWeightedEnsemble
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.linear_model import Ridge
    >>> estimators = [
    ...     ('rf', RandomForestRegressor(n_estimators=100)),
    ...     ('ridge', Ridge())
    ... ]
    >>> ensemble = CVaRWeightedEnsemble(estimators=estimators, alpha=0.95)
    >>> ensemble.fit(X_train, y_train)
    >>> predictions = ensemble.predict(X_test)
    """

    def __init__(self, estimators, alpha=0.95):
        self.estimators = estimators
        self.alpha = alpha

    def fit(self, X, y):
        """
        Fit all base estimators and calculate CVaR weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)

        self.estimators_ = []
        predictions = []

        # Fit each base estimator
        for name, estimator in self.estimators:
            fitted_estimator = estimator.fit(X, y)
            self.estimators_.append((name, fitted_estimator))
            predictions.append(fitted_estimator.predict(X))

        predictions = np.column_stack(predictions)

        # Calculate CVaR for each model
        cvar_scores = []
        for i in range(len(self.estimators)):
            residuals = y - predictions[:, i]
            cvar = _cvar_from_residuals(residuals, alpha=self.alpha)
            cvar_scores.append(cvar)

        self.cvar_scores_ = np.array(cvar_scores)

        # Calculate weights (inverse CVaR)
        inv_cvar = 1.0 / self.cvar_scores_
        self.weights_ = inv_cvar / inv_cvar.sum()

        return self

    def predict(self, X):
        """
        Predict using weighted combination of base estimators.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Weighted ensemble predictions.
        """
        check_is_fitted(self, ['estimators_', 'weights_'])
        X = check_array(X)

        predictions = np.column_stack([
            estimator.predict(X) for _, estimator in self.estimators_
        ])

        return predictions @ self.weights_

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'estimators': self.estimators,
            'alpha': self.alpha
        }

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

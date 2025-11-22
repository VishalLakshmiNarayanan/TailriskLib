"""
Hybrid Meta-Learner for Tail Risk

Combines quantile regression and LaR-optimized weighting in a two-stage
meta-learning framework, then blends them for optimal tail risk prediction.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.optimize import minimize


def _lar_objective(weights, meta_features, y_true, alpha=2.0):
    """LaR objective function for weight optimization."""
    y_pred = meta_features @ weights
    y_max = np.max(y_true)
    if y_max == 0:
        y_max = 1
    w = 1 + alpha * (y_true / y_max)
    return np.mean(w * (y_true - y_pred) ** 2)


class HybridMetaLearner(BaseEstimator, RegressorMixin):
    """
    Hybrid meta-learning ensemble for tail risk prediction.

    This model uses a two-stage approach:
    1. Stage 1: Generate meta-features using cross-validated predictions
       from base models
    2. Stage 2a: Train quantile regression meta-model (focuses on high quantiles)
    2. Stage 2b: Optimize LaR-weighted combination of base models
    3. Blend the two meta-models for final predictions

    The quantile model excels at detecting extreme events, while the LaR model
    maintains overall accuracy. Blending both provides balanced performance.

    Parameters
    ----------
    base_estimators : list of (str, estimator) tuples
        Base models to ensemble. Should include diverse model types.

    quantile : float, default=0.95
        Quantile to target in quantile regression meta-model.

    blend_lambda : float, default=0.25
        Blending weight: pred = λ*quantile_pred + (1-λ)*lar_pred
        Higher values favor quantile model (better tail detection).
        Lower values favor LaR model (better overall accuracy).

    lar_alpha : float, default=1.5
        Alpha parameter for LaR weight optimization.

    cv_folds : int, default=5
        Number of cross-validation folds for meta-feature generation.

    Attributes
    ----------
    meta_quantile_ : QuantileRegressor
        Fitted quantile regression meta-model.

    meta_lar_weights_ : ndarray
        Optimized LaR weights for base models.

    base_estimators_ : list
        Fitted base estimators for each CV fold.

    Examples
    --------
    >>> from tailrisk import HybridMetaLearner
    >>> from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    >>> from sklearn.linear_model import Ridge
    >>> base_models = [
    ...     ('rf', RandomForestRegressor(n_estimators=100)),
    ...     ('gb', GradientBoostingRegressor(n_estimators=100)),
    ...     ('ridge', Ridge())
    ... ]
    >>> model = HybridMetaLearner(
    ...     base_estimators=base_models,
    ...     quantile=0.95,
    ...     blend_lambda=0.25
    ... )
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)

    References
    ----------
    Based on methodology from:
    "Tail-Risk Aware Machine Learning for Insurance Claims Prediction"
    """

    def __init__(self, base_estimators, quantile=0.95, blend_lambda=0.25,
                 lar_alpha=1.5, cv_folds=5):
        self.base_estimators = base_estimators
        self.quantile = quantile
        self.blend_lambda = blend_lambda
        self.lar_alpha = lar_alpha
        self.cv_folds = cv_folds

    def fit(self, X, y):
        """
        Fit the hybrid meta-learner.

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

        n_models = len(self.base_estimators)
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        # Stage 1: Generate meta-features via cross-validation
        meta_train = np.zeros((len(X), n_models))
        self.base_estimators_ = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]

            fold_estimators = []

            for model_idx, (name, estimator) in enumerate(self.base_estimators):
                # Clone and fit estimator
                from sklearn.base import clone
                fitted_estimator = clone(estimator)
                fitted_estimator.fit(X_train_fold, y_train_fold)

                # Generate out-of-fold predictions
                meta_train[val_idx, model_idx] = fitted_estimator.predict(X_val_fold)

                fold_estimators.append((name, fitted_estimator))

            self.base_estimators_.append(fold_estimators)

        # Stage 2a: Train quantile regression meta-model
        self.meta_quantile_ = QuantileRegressor(
            quantile=self.quantile,
            alpha=0.1,
            solver='highs'
        )
        self.meta_quantile_.fit(meta_train, y)

        # Stage 2b: Optimize LaR weights
        w0 = np.ones(n_models) / n_models

        result = minimize(
            _lar_objective,
            w0,
            args=(meta_train, y, self.lar_alpha),
            method='SLSQP',
            bounds=[(0, 1)] * n_models,
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            options={'maxiter': 500}
        )

        self.meta_lar_weights_ = result.x

        return self

    def predict(self, X):
        """
        Predict using hybrid blended meta-model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Blended predictions.
        """
        check_is_fitted(self, ['meta_quantile_', 'meta_lar_weights_', 'base_estimators_'])
        X = check_array(X)

        # Generate meta-features by averaging predictions across folds
        n_models = len(self.base_estimators)
        meta_test_parts = np.zeros((self.cv_folds, len(X), n_models))

        for fold_idx, fold_estimators in enumerate(self.base_estimators_):
            for model_idx, (_, estimator) in enumerate(fold_estimators):
                meta_test_parts[fold_idx, :, model_idx] = estimator.predict(X)

        meta_test = meta_test_parts.mean(axis=0)

        # Quantile model predictions
        y_pred_quantile = self.meta_quantile_.predict(meta_test)

        # LaR weighted predictions
        y_pred_lar = meta_test @ self.meta_lar_weights_

        # Blend predictions
        y_pred_final = (
            self.blend_lambda * y_pred_quantile +
            (1 - self.blend_lambda) * y_pred_lar
        )

        return y_pred_final

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'base_estimators': self.base_estimators,
            'quantile': self.quantile,
            'blend_lambda': self.blend_lambda,
            'lar_alpha': self.lar_alpha,
            'cv_folds': self.cv_folds
        }

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

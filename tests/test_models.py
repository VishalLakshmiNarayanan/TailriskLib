"""Tests for tail risk models."""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

from tailrisk.models import LaRRegressor, CVaRWeightedEnsemble, HybridMetaLearner


@pytest.fixture
def regression_data():
    """Generate synthetic regression data with heavy tail."""
    X, y = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)

    # Add heavy tail by exponentiating some target values
    tail_idx = np.random.choice(len(y), size=50, replace=False)
    y[tail_idx] = np.abs(y[tail_idx]) * 10

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


class TestLaRRegressor:
    """Tests for LaRRegressor."""

    def test_fit_predict(self, regression_data):
        """Test basic fit and predict."""
        X_train, X_test, y_train, y_test = regression_data

        model = LaRRegressor(alpha=2.0)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        assert len(y_pred) == len(y_test)
        assert y_pred.shape == y_test.shape

    def test_different_alphas(self, regression_data):
        """Test with different alpha values."""
        X_train, X_test, y_train, y_test = regression_data

        alphas = [0.5, 1.0, 2.0, 5.0]

        for alpha in alphas:
            model = LaRRegressor(alpha=alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            assert len(y_pred) == len(y_test)

    def test_sklearn_compatibility(self, regression_data):
        """Test sklearn API compatibility."""
        X_train, X_test, y_train, y_test = regression_data

        model = LaRRegressor(alpha=1.5)

        # Should have get_params and set_params
        params = model.get_params()
        assert 'alpha' in params

        model.set_params(alpha=2.0)
        assert model.alpha == 2.0

    def test_zero_target(self):
        """Test handling of all-zero targets."""
        X = np.random.randn(100, 5)
        y = np.zeros(100)

        model = LaRRegressor(alpha=1.0)
        model.fit(X, y)  # Should not crash

        y_pred = model.predict(X)
        assert len(y_pred) == len(y)


class TestCVaRWeightedEnsemble:
    """Tests for CVaRWeightedEnsemble."""

    def test_fit_predict(self, regression_data):
        """Test basic fit and predict."""
        X_train, X_test, y_train, y_test = regression_data

        estimators = [
            ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=10, random_state=42))
        ]

        ensemble = CVaRWeightedEnsemble(estimators=estimators, alpha=0.95)
        ensemble.fit(X_train, y_train)

        y_pred = ensemble.predict(X_test)

        assert len(y_pred) == len(y_test)
        assert hasattr(ensemble, 'weights_')
        assert hasattr(ensemble, 'cvar_scores_')

    def test_weights_sum_to_one(self, regression_data):
        """Test that weights sum to 1."""
        X_train, X_test, y_train, y_test = regression_data

        estimators = [
            ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
            ('ridge', Ridge())
        ]

        ensemble = CVaRWeightedEnsemble(estimators=estimators)
        ensemble.fit(X_train, y_train)

        assert np.isclose(ensemble.weights_.sum(), 1.0)

    def test_single_estimator(self, regression_data):
        """Test with single estimator."""
        X_train, X_test, y_train, y_test = regression_data

        estimators = [
            ('rf', RandomForestRegressor(n_estimators=10, random_state=42))
        ]

        ensemble = CVaRWeightedEnsemble(estimators=estimators)
        ensemble.fit(X_train, y_train)

        assert ensemble.weights_[0] == 1.0


class TestHybridMetaLearner:
    """Tests for HybridMetaLearner."""

    def test_fit_predict(self, regression_data):
        """Test basic fit and predict."""
        X_train, X_test, y_train, y_test = regression_data

        base_models = [
            ('rf', RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)),
            ('ridge', Ridge())
        ]

        model = HybridMetaLearner(
            base_estimators=base_models,
            quantile=0.95,
            blend_lambda=0.25,
            cv_folds=3  # Use fewer folds for speed
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        assert len(y_pred) == len(y_test)
        assert hasattr(model, 'meta_quantile_')
        assert hasattr(model, 'meta_lar_weights_')

    def test_different_blend_lambdas(self, regression_data):
        """Test with different blending parameters."""
        X_train, X_test, y_train, y_test = regression_data

        base_models = [
            ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
            ('ridge', Ridge())
        ]

        for lam in [0.0, 0.25, 0.5, 1.0]:
            model = HybridMetaLearner(
                base_estimators=base_models,
                blend_lambda=lam,
                cv_folds=3
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            assert len(y_pred) == len(y_test)

    def test_lar_weights_sum_to_one(self, regression_data):
        """Test that LaR weights sum to 1."""
        X_train, X_test, y_train, y_test = regression_data

        base_models = [
            ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
            ('ridge', Ridge())
        ]

        model = HybridMetaLearner(
            base_estimators=base_models,
            cv_folds=3
        )

        model.fit(X_train, y_train)

        assert np.isclose(model.meta_lar_weights_.sum(), 1.0, rtol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

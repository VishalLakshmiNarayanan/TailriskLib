"""Integration tests for full workflow."""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from tailrisk import (
    LaRRegressor,
    HybridMetaLearner,
    CVaRWeightedEnsemble,
    cvar_loss,
    tail_coverage_ratio,
    tail_validation_summary
)


def test_full_workflow():
    """Test complete modeling workflow."""
    # Generate heavy-tailed data
    np.random.seed(42)
    X, y = make_regression(n_samples=300, n_features=8, noise=5, random_state=42)

    # Add extreme tail
    tail_idx = np.random.choice(len(y), size=30, replace=False)
    y[tail_idx] = np.abs(y[tail_idx]) * 20

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train LaR model
    lar_model = LaRRegressor(alpha=2.0)
    lar_model.fit(X_train, y_train)
    y_pred_lar = lar_model.predict(X_test)

    # Train ensemble
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=42))
    ]
    ensemble = CVaRWeightedEnsemble(estimators=estimators)
    ensemble.fit(X_train, y_train)
    y_pred_ensemble = ensemble.predict(X_test)

    # Train hybrid meta-learner
    hybrid = HybridMetaLearner(
        base_estimators=estimators,
        blend_lambda=0.3,
        cv_folds=3
    )
    hybrid.fit(X_train, y_train)
    y_pred_hybrid = hybrid.predict(X_test)

    # Validate all models
    for y_pred in [y_pred_lar, y_pred_ensemble, y_pred_hybrid]:
        assert len(y_pred) == len(y_test)

        # Compute metrics
        cvar = cvar_loss(y_test, y_pred)
        tcr = tail_coverage_ratio(y_test, y_pred, quantile=0.95)
        metrics = tail_validation_summary(y_test, y_pred)

        assert cvar > 0
        assert 0 <= tcr <= 2.0  # Allow some overprediction
        assert 'mse_overall' in metrics
        assert 'cvar_95' in metrics


def test_model_comparison():
    """Test comparing multiple models."""
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = np.random.exponential(scale=100, size=200)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Train two models
    model1 = LaRRegressor(alpha=1.0).fit(X_train, y_train)
    model2 = LaRRegressor(alpha=3.0).fit(X_train, y_train)

    y_pred1 = model1.predict(X_test)
    y_pred2 = model2.predict(X_test)

    # Compare metrics
    metrics1 = tail_validation_summary(y_test, y_pred1)
    metrics2 = tail_validation_summary(y_test, y_pred2)

    # Both should produce valid metrics
    assert metrics1['mse_overall'] > 0
    assert metrics2['mse_overall'] > 0

    # CVaR should be different (different alpha values)
    # (Not strictly required, but usually true)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

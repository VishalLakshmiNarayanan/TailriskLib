"""Tests for risk metrics."""

import numpy as np
import pytest
from tailrisk.metrics import (
    cvar_loss,
    loss_at_risk,
    tail_coverage_ratio,
    detection_rate,
    tail_validation_summary
)


def test_loss_at_risk_basic():
    """Test LaR computation with simple values."""
    y_true = np.array([100, 200, 1000])
    y_pred = np.array([90, 210, 500])

    lar = loss_at_risk(y_true, y_pred, alpha=1.0)

    # LaR should weight the large error (1000 vs 500) more heavily
    assert lar > 0
    assert isinstance(lar, float)


def test_cvar_loss_basic():
    """Test CVaR computation."""
    y_true = np.array([100] * 90 + [10000] * 10)
    y_pred = np.array([100] * 90 + [5000] * 10)

    cvar = cvar_loss(y_true, y_pred, alpha=0.95)

    assert cvar > 0
    assert isinstance(cvar, float)


def test_tail_coverage_ratio_perfect():
    """Test TCR with perfect predictions."""
    y_true = np.random.exponential(1000, size=1000)
    y_pred = y_true.copy()

    tcr = tail_coverage_ratio(y_true, y_pred, quantile=0.99)

    assert np.isclose(tcr, 1.0, rtol=0.01)


def test_tail_coverage_ratio_underpredict():
    """Test TCR with underprediction."""
    y_true = np.random.exponential(1000, size=1000)
    y_pred = y_true * 0.5  # Underpredict by 50%

    tcr = tail_coverage_ratio(y_true, y_pred, quantile=0.99)

    assert tcr < 1.0
    assert tcr > 0.4  # Should be around 0.5


def test_detection_rate_basic():
    """Test detection rate computation."""
    y_true = np.array([100] * 90 + [10000] * 10)
    y_pred = np.array([100] * 90 + [9000] * 10)  # Correctly identify all extreme

    det = detection_rate(y_true, y_pred, quantile=0.90)

    assert det >= 0.0
    assert det <= 1.0


def test_tail_validation_summary():
    """Test comprehensive validation summary."""
    np.random.seed(42)
    y_true = np.random.exponential(1000, size=500)
    y_pred = y_true + np.random.normal(0, 200, size=500)

    metrics = tail_validation_summary(y_true, y_pred)

    required_keys = ['mse_overall', 'mse_extreme', 'cvar_95',
                     'detection_90', 'detection_95', 'tcr_95', 'tcr_99']

    for key in required_keys:
        assert key in metrics
        assert isinstance(metrics[key], (int, float))


def test_metrics_with_edge_cases():
    """Test metrics handle edge cases."""
    # All zeros
    y_true = np.zeros(100)
    y_pred = np.zeros(100)

    lar = loss_at_risk(y_true, y_pred, alpha=1.0)
    assert lar == 0.0

    # Single value
    y_true = np.array([100.0])
    y_pred = np.array([90.0])

    tcr = tail_coverage_ratio(y_true, y_pred, quantile=0.99)
    # Should handle single value gracefully


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

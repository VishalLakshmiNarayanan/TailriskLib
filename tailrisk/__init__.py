"""
Tailrisk: Risk-Aware Machine Learning for Tail Risk Modeling

A Python package for building models that excel at predicting extreme outcomes
in insurance claims, financial losses, and other tail-risk scenarios.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from tailrisk.models.lar_regressor import LaRRegressor
from tailrisk.models.hybrid_meta import HybridMetaLearner
from tailrisk.models.ensemble import CVaRWeightedEnsemble
from tailrisk.metrics.risk_metrics import (
    cvar_loss,
    loss_at_risk,
    tail_coverage_ratio,
    detection_rate,
    tail_validation_summary
)

__all__ = [
    "LaRRegressor",
    "HybridMetaLearner",
    "CVaRWeightedEnsemble",
    "cvar_loss",
    "loss_at_risk",
    "tail_coverage_ratio",
    "detection_rate",
    "tail_validation_summary",
]

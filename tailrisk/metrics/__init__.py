"""
Risk-aware evaluation metrics for tail risk modeling.
"""

from tailrisk.metrics.risk_metrics import (
    cvar_loss,
    loss_at_risk,
    tail_coverage_ratio,
    detection_rate,
    tail_validation_summary
)

__all__ = [
    "cvar_loss",
    "loss_at_risk",
    "tail_coverage_ratio",
    "detection_rate",
    "tail_validation_summary",
]

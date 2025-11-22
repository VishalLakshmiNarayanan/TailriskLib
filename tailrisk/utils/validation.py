"""
Validation utilities for tail risk models.
"""

import numpy as np
from tailrisk.metrics import tail_validation_summary


def print_tail_validation(y_true, y_pred, model_name="Model"):
    """
    Print comprehensive tail validation report.

    Parameters
    ----------
    y_true : array-like
        True target values.

    y_pred : array-like
        Predicted values.

    model_name : str, default="Model"
        Name of the model for display.

    Examples
    --------
    >>> from tailrisk.utils import print_tail_validation
    >>> print_tail_validation(y_test, y_pred, model_name="Hybrid Meta-Learner")
    """
    metrics = tail_validation_summary(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f" TAIL VALIDATION: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'Value':>20}")
    print(f"{'-'*60}")
    print(f"{'MSE (Overall)':<30} {metrics['mse_overall']:>20,.2f}")
    print(f"{'MSE (Extreme Tail, 99%+)':<30} {metrics['mse_extreme']:>20,.2f}")
    print(f"{'CVaR (95%)':<30} {metrics['cvar_95']:>20,.2f}")
    print(f"{'Detection Rate @ 90%':<30} {metrics['detection_90']*100:>19.2f}%")
    print(f"{'Detection Rate @ 95%':<30} {metrics['detection_95']*100:>19.2f}%")
    print(f"{'Tail Coverage Ratio @ 95%':<30} {metrics['tcr_95']:>20.3f}")
    print(f"{'Tail Coverage Ratio @ 99%':<30} {metrics['tcr_99']:>20.3f}")
    print(f"{'='*60}\n")

    return metrics


def compare_models(y_true, predictions_dict):
    """
    Compare multiple models side-by-side.

    Parameters
    ----------
    y_true : array-like
        True target values.

    predictions_dict : dict of {model_name: y_pred}
        Dictionary mapping model names to their predictions.

    Examples
    --------
    >>> from tailrisk.utils.validation import compare_models
    >>> compare_models(y_test, {
    ...     'Baseline': y_pred_baseline,
    ...     'LaR': y_pred_lar,
    ...     'Hybrid': y_pred_hybrid
    ... })
    """
    results = {}

    for name, y_pred in predictions_dict.items():
        metrics = tail_validation_summary(y_true, y_pred)
        results[name] = metrics

    # Print comparison table
    print(f"\n{'='*90}")
    print(f" MODEL COMPARISON")
    print(f"{'='*90}")

    metric_names = [
        ('MSE (Overall)', 'mse_overall', '{:,.0f}'),
        ('MSE (Extreme)', 'mse_extreme', '{:,.0f}'),
        ('CVaR (95%)', 'cvar_95', '{:,.2f}'),
        ('Detection @ 90%', 'detection_90', '{:.1%}'),
        ('Detection @ 95%', 'detection_95', '{:.1%}'),
        ('TCR @ 95%', 'tcr_95', '{:.3f}'),
        ('TCR @ 99%', 'tcr_99', '{:.3f}'),
    ]

    # Header
    model_names = list(predictions_dict.keys())
    col_width = max(15, max(len(name) for name in model_names) + 2)
    header = f"{'Metric':<25}"
    for name in model_names:
        header += f"{name:>{col_width}}"
    print(header)
    print('-' * 90)

    # Rows
    for metric_label, metric_key, fmt in metric_names:
        row = f"{metric_label:<25}"
        for name in model_names:
            value = results[name][metric_key]
            if np.isnan(value):
                row += f"{'N/A':>{col_width}}"
            else:
                row += f"{fmt.format(value):>{col_width}}"
        print(row)

    print(f"{'='*90}\n")

    return results

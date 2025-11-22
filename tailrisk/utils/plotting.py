"""
Visualization utilities for tail risk analysis.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_tail_comparison(y_true, y_pred_baseline, y_pred_tailrisk,
                         quantiles=[0.90, 0.95, 0.99], figsize=(18, 5)):
    """
    Compare baseline vs tail-risk model predictions across risk buckets.

    Parameters
    ----------
    y_true : array-like
        True target values.

    y_pred_baseline : array-like
        Predictions from baseline model.

    y_pred_tailrisk : array-like
        Predictions from tail-risk aware model.

    quantiles : list of float, default=[0.90, 0.95, 0.99]
        Quantile thresholds for risk buckets.

    figsize : tuple, default=(18, 5)
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    y_true = np.asarray(y_true)
    y_pred_baseline = np.asarray(y_pred_baseline)
    y_pred_tailrisk = np.asarray(y_pred_tailrisk)

    # Define risk buckets
    q1, q2, q3 = quantiles
    low_idx = y_true < np.quantile(y_true, q1)
    mid_idx = (y_true >= np.quantile(y_true, q1)) & (y_true < np.quantile(y_true, q2))
    high_idx = (y_true >= np.quantile(y_true, q2)) & (y_true < np.quantile(y_true, q3))
    extreme_idx = y_true >= np.quantile(y_true, q3)

    buckets = [
        (f"Low (0-{q1*100:.0f}%)", low_idx, 10, 0.3),
        (f"Mid ({q1*100:.0f}-{q2*100:.0f}%)", mid_idx, 20, 0.4),
        (f"High ({q2*100:.0f}-{q3*100:.0f}%)", high_idx, 30, 0.5),
        (f"Extreme (≥{q3*100:.0f}%)", extreme_idx, 50, 0.7)
    ]

    fig, axes = plt.subplots(1, len(buckets), figsize=figsize)

    for ax, (name, idx, size, alpha) in zip(axes, buckets):
        if idx.sum() == 0:
            continue

        ax.scatter(y_true[idx], y_pred_baseline[idx],
                  alpha=alpha, s=size, label='Baseline', color='blue')
        ax.scatter(y_true[idx], y_pred_tailrisk[idx],
                  alpha=alpha, s=size, label='Tail-Risk', color='green')

        # Perfect prediction line
        min_val = y_true[idx].min()
        max_val = y_true[idx].max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{name}\n(n={idx.sum()})')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle('Baseline vs Tail-Risk Model: Predictions Across Risk Buckets',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_predictions_by_quantile(y_true, y_pred, quantiles=[0.90, 0.95, 0.99],
                                 figsize=(10, 6)):
    """
    Scatter plot of predictions colored by quantile bucket.

    Parameters
    ----------
    y_true : array-like
        True target values.

    y_pred : array-like
        Predicted values.

    quantiles : list of float
        Quantile thresholds for coloring.

    figsize : tuple
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    # Assign colors by quantile bucket
    colors = []
    for val in y_true:
        if val >= np.quantile(y_true, quantiles[2]):
            colors.append('red')
        elif val >= np.quantile(y_true, quantiles[1]):
            colors.append('orange')
        elif val >= np.quantile(y_true, quantiles[0]):
            colors.append('yellow')
        else:
            colors.append('blue')

    ax.scatter(y_true, y_pred, c=colors, alpha=0.5, edgecolors='none')

    # Perfect prediction line
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect Prediction')

    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Predictions by Risk Bucket')
    ax.grid(alpha=0.3)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label=f'Low (0-{quantiles[0]*100:.0f}%)'),
        Patch(facecolor='yellow', label=f'Mid ({quantiles[0]*100:.0f}-{quantiles[1]*100:.0f}%)'),
        Patch(facecolor='orange', label=f'High ({quantiles[1]*100:.0f}-{quantiles[2]*100:.0f}%)'),
        Patch(facecolor='red', label=f'Extreme (≥{quantiles[2]*100:.0f}%)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    return fig


def plot_residual_distribution(y_true, y_pred, log_scale=True, bins=50,
                               figsize=(10, 6)):
    """
    Plot distribution of residuals with tail highlighting.

    Parameters
    ----------
    y_true : array-like
        True target values.

    y_pred : array-like
        Predicted values.

    log_scale : bool, default=True
        Whether to use log scale for y-axis.

    bins : int, default=50
        Number of histogram bins.

    figsize : tuple
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(residuals, bins=bins, color='teal', alpha=0.7, edgecolor='black')

    # Mark VaR and CVaR
    var_95 = np.quantile(residuals, 0.95)
    var_99 = np.quantile(residuals, 0.99)

    ax.axvline(var_95, color='orange', linestyle='--', linewidth=2,
               label=f'VaR(95%) = {var_95:.2f}')
    ax.axvline(var_99, color='red', linestyle='--', linewidth=2,
               label=f'VaR(99%) = {var_99:.2f}')

    ax.set_xlabel('Residual (Actual - Predicted)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Prediction Residuals')

    if log_scale:
        ax.set_yscale('log')

    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig

"""
Quick Start Example: TailRisk Package

This example demonstrates the basic usage of the tailrisk package
for modeling heavy-tailed insurance claims data.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge

# Import tailrisk components
from tailrisk import (
    LaRRegressor,
    HybridMetaLearner,
    CVaRWeightedEnsemble,
    tail_validation_summary,
)
from tailrisk.utils import print_tail_validation, compare_models


def generate_heavy_tail_data(n_samples=5000, n_features=10, random_state=42):
    """
    Generate synthetic insurance claims data with heavy tail.

    Returns
    -------
    X : ndarray
        Feature matrix (e.g., policyholder characteristics).
    y : ndarray
        Target values (claim costs) with heavy tail distribution.
    """
    np.random.seed(random_state)

    # Features: age, premium, vehicle value, etc.
    X = np.random.randn(n_samples, n_features)

    # Base claim cost (exponential distribution)
    y = np.random.exponential(scale=500, size=n_samples)

    # Add catastrophic claims (top 5%)
    n_catastrophic = int(0.05 * n_samples)
    catastrophic_idx = np.random.choice(n_samples, size=n_catastrophic, replace=False)
    y[catastrophic_idx] = np.random.exponential(scale=10000, size=n_catastrophic)

    # Add some feature-target relationship
    y = y + 50 * X[:, 0]  # Age effect
    y = np.abs(y)  # Ensure positive

    return X, y


def main():
    print("=" * 80)
    print("TailRisk Package - Quick Start Example")
    print("=" * 80)

    # 1. Generate synthetic data
    print("\n[1/5] Generating heavy-tailed claims data...")
    X, y = generate_heavy_tail_data(n_samples=5000)

    print(f"  Total samples: {len(y)}")
    print(f"  Mean claim: ${y.mean():,.2f}")
    print(f"  Median claim: ${np.median(y):,.2f}")
    print(f"  99th percentile: ${np.quantile(y, 0.99):,.2f}")
    print(f"  Max claim: ${y.max():,.2f}")

    # 2. Split data
    print("\n[2/5] Splitting into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Training samples: {len(y_train)}")
    print(f"  Test samples: {len(y_test)}")

    # 3. Train baseline model
    print("\n[3/5] Training baseline (standard Linear Regression)...")
    baseline = LinearRegression()
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)

    # 4. Train LaR model
    print("\n[4/5] Training LaR model (alpha=2.0)...")
    lar_model = LaRRegressor(alpha=2.0)
    lar_model.fit(X_train, y_train)
    y_pred_lar = lar_model.predict(X_test)

    # 5. Train Hybrid Meta-Learner
    print("\n[5/5] Training Hybrid Meta-Learner...")
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)),
        ('ridge', Ridge(alpha=1.0))
    ]

    hybrid_model = HybridMetaLearner(
        base_estimators=base_models,
        quantile=0.95,
        blend_lambda=0.25,
        lar_alpha=1.5,
        cv_folds=5
    )

    hybrid_model.fit(X_train, y_train)
    y_pred_hybrid = hybrid_model.predict(X_test)

    print("\n" + "=" * 80)
    print("RESULTS: Model Comparison")
    print("=" * 80)

    # Compare all models
    results = compare_models(y_test, {
        'Baseline (LR)': y_pred_baseline,
        'LaR (α=2.0)': y_pred_lar,
        'Hybrid Meta': y_pred_hybrid
    })

    # Detailed validation for hybrid model
    print("\n" + "=" * 80)
    print("DETAILED VALIDATION: Hybrid Meta-Learner")
    print("=" * 80)
    print_tail_validation(y_test, y_pred_hybrid, model_name="Hybrid Meta-Learner")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    baseline_tcr99 = results['Baseline (LR)']['tcr_99']
    hybrid_tcr99 = results['Hybrid Meta']['tcr_99']
    tcr_improvement = ((hybrid_tcr99 - baseline_tcr99) / baseline_tcr99) * 100

    baseline_cvar = results['Baseline (LR)']['cvar_95']
    hybrid_cvar = results['Hybrid Meta']['cvar_95']
    cvar_improvement = ((baseline_cvar - hybrid_cvar) / baseline_cvar) * 100

    print(f"\n✓ Tail Coverage Ratio @ 99%:")
    print(f"  - Baseline: {baseline_tcr99:.3f} (captures {baseline_tcr99*100:.1f}% of extreme losses)")
    print(f"  - Hybrid:   {hybrid_tcr99:.3f} (captures {hybrid_tcr99*100:.1f}% of extreme losses)")
    print(f"  - Improvement: {tcr_improvement:+.1f}%")

    print(f"\n✓ CVaR (95%):")
    print(f"  - Baseline: ${baseline_cvar:,.2f}")
    print(f"  - Hybrid:   ${hybrid_cvar:,.2f}")
    print(f"  - Improvement: {cvar_improvement:+.1f}%")

    print("\n✓ Why this matters:")
    print("  - Better tail prediction = more accurate reserves")
    print("  - Improved detection = early warning for catastrophic claims")
    print("  - Higher TCR = capturing more of the extreme tail value")

    print("\n" + "=" * 80)
    print("Example complete! See README.md for more advanced usage.")
    print("=" * 80)


if __name__ == "__main__":
    main()

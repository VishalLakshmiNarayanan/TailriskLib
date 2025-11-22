# TailRisk Tutorial: From Zero to Production

A comprehensive, step-by-step guide to using TailRisk for tail risk modeling.

## Table of Contents
1. [Introduction](#introduction)
2. [Understanding the Problem](#understanding-the-problem)
3. [Getting Started](#getting-started)
4. [Basic Usage: LaRRegressor](#basic-usage-larregressor)
5. [Advanced Usage: HybridMetaLearner](#advanced-usage-hybridmetalearner)
6. [Model Evaluation](#model-evaluation)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Production Deployment](#production-deployment)
9. [Best Practices](#best-practices)

---

## Introduction

This tutorial will teach you how to build machine learning models that excel at predicting extreme outcomes - a critical capability for insurance, finance, and risk management applications.

**What you'll learn:**
- Why traditional ML fails on tail risk
- How to use LaRRegressor for weighted regression
- How to build advanced HybridMetaLearner models
- How to evaluate tail risk performance
- Production-ready workflows

**Prerequisites:**
- Basic Python knowledge
- Familiarity with scikit-learn
- Understanding of regression problems

---

## Understanding the Problem

### The Tail Risk Challenge

Consider insurance claims data:

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate claims: most are small, few are catastrophic
np.random.seed(42)
typical_claims = np.random.exponential(scale=500, size=950)
catastrophic_claims = np.random.exponential(scale=20000, size=50)
all_claims = np.concatenate([typical_claims, catastrophic_claims])

# Visualize
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(all_claims, bins=50, edgecolor='black')
plt.xlabel('Claim Amount ($)')
plt.ylabel('Frequency')
plt.title('Distribution of Claims')

plt.subplot(1, 2, 2)
plt.hist(all_claims, bins=50, edgecolor='black', cumulative=True)
plt.xlabel('Claim Amount ($)')
plt.ylabel('Cumulative Count')
plt.title('Cumulative Distribution')
plt.tight_layout()
plt.show()

print(f"Mean: ${all_claims.mean():,.2f}")
print(f"Median: ${np.median(all_claims):,.2f}")
print(f"95th percentile: ${np.percentile(all_claims, 95):,.2f}")
print(f"99th percentile: ${np.percentile(all_claims, 99):,.2f}")
print(f"Max: ${all_claims.max():,.2f}")
```

Output:
```
Mean: $1,454.32
Median: $346.57
95th percentile: $2,891.45
99th percentile: $15,234.67
Max: $87,392.11
```

**The problem:**
- Top 5% of claims represent 60%+ of total costs
- Traditional MSE optimization treats $1,000 error on $500 claim same as $1,000 error on $50,000 claim
- Result: Models underestimate extreme claims catastrophically

### Why MSE Fails

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Simple example
y_true = np.array([100, 200, 300, 50000])  # One extreme value
y_pred_good_avg = np.array([100, 200, 300, 35000])  # Good average, misses tail
y_pred_good_tail = np.array([80, 180, 280, 48000])  # Worse average, catches tail

mse_avg = mean_squared_error(y_true, y_pred_good_avg)
mse_tail = mean_squared_error(y_true, y_pred_good_tail)

print(f"Good average model MSE: {mse_avg:,.0f}")  # Lower MSE
print(f"Good tail model MSE: {mse_tail:,.0f}")    # Higher MSE

# But which is better for tail risk?
tail_error_avg = abs(50000 - 35000)  # Off by $15,000!
tail_error_tail = abs(50000 - 48000)  # Off by $2,000
```

**MSE prefers the first model, but it's dangerous for tail risk!**

---

## Getting Started

### Setup

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge

# Import TailRisk
from tailrisk import (
    LaRRegressor,
    HybridMetaLearner,
    tail_validation_summary
)
from tailrisk.utils import print_tail_validation, compare_models
```

### Generate Sample Data

For this tutorial, we'll create synthetic heavy-tailed data:

```python
def generate_claims_data(n_samples=5000, n_features=10, random_state=42):
    """
    Generate synthetic insurance claims data.

    Features represent policyholder characteristics:
    - Age, vehicle value, location risk score, etc.
    """
    np.random.seed(random_state)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Base claim costs (exponential distribution)
    base_claims = np.random.exponential(scale=500, size=n_samples)

    # Add catastrophic claims (top 5%)
    n_catastrophic = int(0.05 * n_samples)
    catastrophic_idx = np.random.choice(n_samples, n_catastrophic, replace=False)
    base_claims[catastrophic_idx] = np.random.exponential(scale=15000, size=n_catastrophic)

    # Add feature relationships
    y = base_claims.copy()
    y += 50 * X[:, 0]  # Age effect
    y += 30 * X[:, 1]  # Vehicle value effect
    y = np.abs(y)      # Ensure positive

    return X, y

# Generate data
X, y = generate_claims_data(n_samples=5000)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(y_train)}")
print(f"Test samples: {len(y_test)}")
print(f"\nTarget distribution:")
print(f"  Mean: ${y.mean():,.2f}")
print(f"  95th percentile: ${np.percentile(y, 95):,.2f}")
print(f"  99th percentile: ${np.percentile(y, 99):,.2f}")
```

---

## Basic Usage: LaRRegressor

### Step 1: Train Baseline Model

First, let's see how a standard model performs:

```python
# Standard linear regression
baseline = LinearRegression()
baseline.fit(X_train, y_train)
y_pred_baseline = baseline.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse_baseline = mean_squared_error(y_test, y_pred_baseline)
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)

print("Baseline Model:")
print(f"  MSE: ${mse_baseline:,.2f}")
print(f"  MAE: ${mae_baseline:,.2f}")
```

### Step 2: Train LaRRegressor

Now use LaRRegressor for tail-aware training:

```python
# LaR model
lar_model = LaRRegressor(alpha=2.0)
lar_model.fit(X_train, y_train)
y_pred_lar = lar_model.predict(X_test)

# Evaluate
mse_lar = mean_squared_error(y_test, y_pred_lar)
mae_lar = mean_absolute_error(y_test, y_pred_lar)

print("\nLaR Model (alpha=2.0):")
print(f"  MSE: ${mse_lar:,.2f}")
print(f"  MAE: ${mae_lar:,.2f}")
```

### Step 3: Compare Tail Performance

Standard metrics don't tell the full story. Use tail metrics:

```python
from tailrisk import cvar_loss, tail_coverage_ratio

# CVaR comparison
cvar_baseline = cvar_loss(y_test, y_pred_baseline, alpha=0.95)
cvar_lar = cvar_loss(y_test, y_pred_lar, alpha=0.95)

print("\nTail Performance (CVaR @ 95%):")
print(f"  Baseline: ${cvar_baseline:,.2f}")
print(f"  LaR:      ${cvar_lar:,.2f}")
print(f"  Improvement: {((cvar_baseline - cvar_lar) / cvar_baseline * 100):.1f}%")

# Tail coverage
tcr_baseline = tail_coverage_ratio(y_test, y_pred_baseline, quantile=0.99)
tcr_lar = tail_coverage_ratio(y_test, y_pred_lar, quantile=0.99)

print("\nTail Coverage Ratio @ 99%:")
print(f"  Baseline: {tcr_baseline:.3f} ({tcr_baseline*100:.1f}% of tail captured)")
print(f"  LaR:      {tcr_lar:.3f} ({tcr_lar*100:.1f}% of tail captured)")
```

### Step 4: Tune Alpha Parameter

The `alpha` parameter controls tail focus:

```python
# Test different alpha values
alphas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
results = []

for alpha in alphas:
    model = LaRRegressor(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    cvar = cvar_loss(y_test, y_pred, alpha=0.95)
    tcr = tail_coverage_ratio(y_test, y_pred, quantile=0.99)

    results.append({
        'alpha': alpha,
        'MSE': mse,
        'CVaR': cvar,
        'TCR': tcr
    })

# Display results
df = pd.DataFrame(results)
print("\nAlpha Tuning Results:")
print(df.to_string(index=False))

# Best for tail risk
best_idx = df['CVaR'].idxmin()
print(f"\nBest alpha for CVaR: {df.loc[best_idx, 'alpha']}")
```

---

## Advanced Usage: HybridMetaLearner

### Step 1: Define Base Models

Create diverse base estimators:

```python
# Define multiple model types
base_models = [
    ('rf', RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )),
    ('gb', GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )),
    ('ridge', Ridge(alpha=1.0)),
]

print("Base models defined:")
for name, model in base_models:
    print(f"  - {name}: {type(model).__name__}")
```

### Step 2: Build Hybrid Model

```python
# Create hybrid meta-learner
hybrid_model = HybridMetaLearner(
    base_estimators=base_models,
    quantile=0.95,       # Focus on 95th percentile
    blend_lambda=0.25,   # 25% quantile, 75% LaR
    lar_alpha=1.5,       # LaR weight strength
    cv_folds=5           # Cross-validation folds
)

print("\nTraining Hybrid Meta-Learner...")
print("This may take a minute...")

# Fit model
hybrid_model.fit(X_train, y_train)

print("✓ Training complete")
```

### Step 3: Generate Predictions

```python
# Predict
y_pred_hybrid = hybrid_model.predict(X_test)

# Quick evaluation
mse_hybrid = mean_squared_error(y_test, y_pred_hybrid)
cvar_hybrid = cvar_loss(y_test, y_pred_hybrid, alpha=0.95)

print("\nHybrid Model Performance:")
print(f"  MSE: ${mse_hybrid:,.2f}")
print(f"  CVaR(95%): ${cvar_hybrid:,.2f}")
```

### Step 4: Comprehensive Comparison

```python
# Compare all models
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

results = compare_models(y_test, {
    'Baseline': y_pred_baseline,
    'LaR (α=2.0)': y_pred_lar,
    'Hybrid Meta': y_pred_hybrid
})

# Detailed view of hybrid
print("\n" + "="*80)
print("DETAILED METRICS: Hybrid Meta-Learner")
print("="*80)
print_tail_validation(y_test, y_pred_hybrid, model_name="Hybrid Meta")
```

---

## Model Evaluation

### Comprehensive Metrics

```python
from tailrisk import detection_rate

def evaluate_comprehensive(y_true, y_pred, model_name):
    """Comprehensive tail risk evaluation."""
    print(f"\n{'='*60}")
    print(f"{model_name} - Full Evaluation")
    print(f"{'='*60}")

    # Standard metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # Tail metrics
    cvar_90 = cvar_loss(y_true, y_pred, alpha=0.90)
    cvar_95 = cvar_loss(y_true, y_pred, alpha=0.95)
    cvar_99 = cvar_loss(y_true, y_pred, alpha=0.99)

    tcr_95 = tail_coverage_ratio(y_true, y_pred, quantile=0.95)
    tcr_99 = tail_coverage_ratio(y_true, y_pred, quantile=0.99)

    det_90 = detection_rate(y_true, y_pred, quantile=0.90)
    det_95 = detection_rate(y_true, y_pred, quantile=0.95)
    det_99 = detection_rate(y_true, y_pred, quantile=0.99)

    print(f"\nStandard Metrics:")
    print(f"  MSE: ${mse:,.2f}")
    print(f"  MAE: ${mae:,.2f}")

    print(f"\nConditional Value-at-Risk:")
    print(f"  CVaR(90%): ${cvar_90:,.2f}")
    print(f"  CVaR(95%): ${cvar_95:,.2f}")
    print(f"  CVaR(99%): ${cvar_99:,.2f}")

    print(f"\nTail Coverage Ratio:")
    print(f"  TCR(95%): {tcr_95:.3f} - Capturing {tcr_95*100:.1f}% of extreme value")
    print(f"  TCR(99%): {tcr_99:.3f} - Capturing {tcr_99*100:.1f}% of extreme value")

    print(f"\nDetection Rate:")
    print(f"  Detection(90%): {det_90*100:.1f}%")
    print(f"  Detection(95%): {det_95*100:.1f}%")
    print(f"  Detection(99%): {det_99*100:.1f}%")

# Evaluate all models
evaluate_comprehensive(y_test, y_pred_baseline, "Baseline")
evaluate_comprehensive(y_test, y_pred_lar, "LaR Model")
evaluate_comprehensive(y_test, y_pred_hybrid, "Hybrid Meta")
```

### Visualize Performance

```python
from tailrisk.utils import plot_tail_comparison

# Visual comparison
plot_tail_comparison(
    y_test,
    y_pred_baseline,
    y_pred_hybrid,
    quantiles=[0.90, 0.95, 0.99]
)
plt.tight_layout()
plt.show()
```

---

## Hyperparameter Tuning

### Grid Search with Tail Metrics

```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# Custom scorer for CVaR
def cvar_scorer(y_true, y_pred):
    """Negative CVaR (for minimization)."""
    return -cvar_loss(y_true, y_pred, alpha=0.95)

cvar_score = make_scorer(cvar_scorer, greater_is_better=True)

# Grid search for LaRRegressor
param_grid = {
    'alpha': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
}

grid = GridSearchCV(
    LaRRegressor(),
    param_grid,
    cv=5,
    scoring=cvar_score,
    verbose=1
)

grid.fit(X_train, y_train)

print(f"\nBest parameters: {grid.best_params_}")
print(f"Best CVaR score: ${-grid.best_score_:,.2f}")

# Use best model
best_lar = grid.best_estimator_
```

### Manual Tuning for HybridMetaLearner

```python
# Test different configurations
configs = [
    {'quantile': 0.90, 'blend_lambda': 0.20, 'lar_alpha': 1.5},
    {'quantile': 0.95, 'blend_lambda': 0.25, 'lar_alpha': 1.5},
    {'quantile': 0.95, 'blend_lambda': 0.30, 'lar_alpha': 2.0},
    {'quantile': 0.99, 'blend_lambda': 0.25, 'lar_alpha': 2.0},
]

best_cvar = float('inf')
best_config = None

for config in configs:
    model = HybridMetaLearner(
        base_estimators=base_models,
        **config,
        cv_folds=5
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cvar = cvar_loss(y_test, y_pred, alpha=0.95)

    print(f"Config {config} -> CVaR: ${cvar:,.2f}")

    if cvar < best_cvar:
        best_cvar = cvar
        best_config = config

print(f"\nBest configuration: {best_config}")
print(f"Best CVaR: ${best_cvar:,.2f}")
```

---

## Production Deployment

### Save and Load Models

```python
import joblib

# Save model
joblib.dump(hybrid_model, 'hybrid_tailrisk_model.pkl')
print("✓ Model saved to hybrid_tailrisk_model.pkl")

# Load model
loaded_model = joblib.load('hybrid_tailrisk_model.pkl')
predictions = loaded_model.predict(X_test)
print("✓ Model loaded and working")
```

### Prediction Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create production pipeline
production_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', HybridMetaLearner(
        base_estimators=base_models,
        quantile=0.95,
        blend_lambda=0.25
    ))
])

# Train
production_pipeline.fit(X_train, y_train)

# Save
joblib.dump(production_pipeline, 'production_pipeline.pkl')

# Use in production
def predict_claim_cost(features):
    """Production prediction function."""
    pipeline = joblib.load('production_pipeline.pkl')
    prediction = pipeline.predict([features])[0]
    return max(0, prediction)  # Ensure non-negative

# Example usage
new_claim_features = X_test[0]
predicted_cost = predict_claim_cost(new_claim_features)
print(f"\nPredicted claim cost: ${predicted_cost:,.2f}")
```

---

## Best Practices

### 1. Data Preparation

```python
# Remove outliers that are clearly errors (not true tail events)
def remove_data_errors(X, y, max_reasonable=1000000):
    """Remove impossible values while keeping true extremes."""
    mask = (y > 0) & (y < max_reasonable)
    return X[mask], y[mask]

# Handle missing values before training
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X_clean = imputer.fit_transform(X)
```

### 2. Model Selection

**Use LaRRegressor when:**
- You need interpretability
- You have limited training data
- You want fast training
- Linear relationships are sufficient

**Use HybridMetaLearner when:**
- Maximum tail performance is critical
- You have sufficient training data (1000+ samples)
- Complex non-linear relationships exist
- You can afford longer training time

### 3. Evaluation Strategy

```python
# Always use time-based or stratified splits for tail risk
from sklearn.model_selection import StratifiedKFold

def stratified_split_by_tail(X, y, quantile=0.95, test_size=0.2):
    """Ensure test set has representative tail events."""
    threshold = np.percentile(y, quantile * 100)
    is_tail = (y >= threshold).astype(int)

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in splitter.split(X, is_tail):
        # Use first split
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = stratified_split_by_tail(X, y)
```

### 4. Monitoring in Production

```python
def monitor_predictions(y_true, y_pred, alert_threshold=0.15):
    """Monitor model performance and alert if deteriorating."""
    tcr = tail_coverage_ratio(y_true, y_pred, quantile=0.99)
    cvar = cvar_loss(y_true, y_pred, alpha=0.95)

    if tcr < alert_threshold:
        print(f"⚠ WARNING: TCR dropped to {tcr:.3f}")
        print("  Model may need retraining!")

    return {'tcr': tcr, 'cvar': cvar}

# Use monthly or quarterly
metrics = monitor_predictions(y_test, y_pred_hybrid)
```

---

## Summary

You've learned:

1. Why tail risk is different from standard ML problems
2. How to use LaRRegressor for weighted regression
3. How to build advanced HybridMetaLearner models
4. How to properly evaluate tail risk models
5. How to tune hyperparameters
6. How to deploy in production

**Next steps:**
- Try TailRisk on your own data
- Experiment with different base models
- Read the [API Reference](API_REFERENCE.md)
- Explore advanced examples in `/examples`

---

Happy modeling!

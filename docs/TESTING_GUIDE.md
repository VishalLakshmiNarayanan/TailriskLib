# TailRisk Testing Guide

Comprehensive guide for testing the TailRisk package functionality.

## Table of Contents
- [Quick Tests](#quick-tests)
- [Unit Tests](#unit-tests)
- [Integration Tests](#integration-tests)
- [Manual Testing](#manual-testing)
- [Performance Testing](#performance-testing)
- [Continuous Integration](#continuous-integration)

---

## Quick Tests

### 1. Verify Installation

```bash
# Test basic import
python -c "import tailrisk; print(f'TailRisk v{tailrisk.__version__} installed successfully')"
```

Expected output:
```
TailRisk v0.1.0 installed successfully
```

### 2. Run Quick Start Example

```bash
# Run the quick start script
cd c:\Users\efbvl\OneDrive\Desktop\tailrisk-lib
python examples\quick_start.py
```

Expected: Script runs without errors and displays model comparison results.

### 3. Import All Components

```bash
python -c "from tailrisk import LaRRegressor, HybridMetaLearner, CVaRWeightedEnsemble, cvar_loss, tail_coverage_ratio; print('✓ All imports successful')"
```

---

## Unit Tests

### Running Unit Tests

TailRisk includes comprehensive test coverage using pytest.

#### Run All Tests

```bash
# From project root
cd c:\Users\efbvl\OneDrive\Desktop\tailrisk-lib
python -m pytest tests/
```

#### Run Specific Test Files

```bash
# Test metrics only
python -m pytest tests/test_metrics.py

# Test models only
python -m pytest tests/test_models.py

# Test integration
python -m pytest tests/test_integration.py
```

#### Run with Verbose Output

```bash
python -m pytest tests/ -v
```

#### Run with Coverage Report

```bash
# Install coverage tools first
pip install pytest-cov

# Run with coverage
python -m pytest tests/ --cov=tailrisk --cov-report=html

# View coverage report
# Open htmlcov/index.html in browser
```

### Expected Test Output

```
================================= test session starts =================================
platform win32 -- Python 3.10.0, pytest-7.0.0, pluggy-1.0.0
rootdir: c:\Users\efbvl\OneDrive\Desktop\tailrisk-lib
collected 45 items

tests/test_metrics.py ........                                               [ 17%]
tests/test_models.py ....................                                    [ 62%]
tests/test_integration.py ...........                                        [100%]

================================== 45 passed in 8.32s =================================
```

---

## Integration Tests

### Test 1: End-to-End LaRRegressor Workflow

Create file `test_lar_workflow.py`:

```python
"""Test complete LaRRegressor workflow"""
import numpy as np
from sklearn.model_selection import train_test_split
from tailrisk import LaRRegressor, tail_validation_summary

def test_lar_workflow():
    print("Testing LaRRegressor Workflow...")

    # Generate data
    np.random.seed(42)
    X = np.random.randn(500, 10)
    y = np.random.exponential(scale=1000, size=500)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = LaRRegressor(alpha=2.0)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Validate
    assert len(y_pred) == len(y_test), "Prediction length mismatch"
    assert all(y_pred >= 0), "Negative predictions found"

    # Metrics
    metrics = tail_validation_summary(y_test, y_pred)
    assert 'cvar_95' in metrics, "Missing CVaR metric"
    assert 'tcr_99' in metrics, "Missing TCR metric"

    print("✓ LaRRegressor workflow test passed")
    return True

if __name__ == "__main__":
    test_lar_workflow()
```

Run:
```bash
python test_lar_workflow.py
```

### Test 2: HybridMetaLearner Integration

Create file `test_hybrid_workflow.py`:

```python
"""Test complete HybridMetaLearner workflow"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tailrisk import HybridMetaLearner, tail_validation_summary

def test_hybrid_workflow():
    print("Testing HybridMetaLearner Workflow...")

    # Generate data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.exponential(scale=1000, size=1000)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define base models
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=20, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=20, random_state=42))
    ]

    # Train
    model = HybridMetaLearner(
        base_estimators=base_models,
        quantile=0.95,
        blend_lambda=0.25,
        cv_folds=3  # Fewer folds for faster testing
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Validate
    assert len(y_pred) == len(y_test), "Prediction length mismatch"
    assert all(y_pred >= 0), "Negative predictions found"

    # Check model attributes
    assert hasattr(model, 'base_estimators_'), "Base estimators not fitted"
    assert hasattr(model, 'quantile_meta_'), "Quantile meta-model missing"

    # Metrics
    metrics = tail_validation_summary(y_test, y_pred)
    print(f"  CVaR(95%): {metrics['cvar_95']:.2f}")
    print(f"  TCR(99%): {metrics['tcr_99']:.3f}")

    print("✓ HybridMetaLearner workflow test passed")
    return True

if __name__ == "__main__":
    test_hybrid_workflow()
```

Run:
```bash
python test_hybrid_workflow.py
```

### Test 3: Metrics Validation

Create file `test_metrics_validation.py`:

```python
"""Test all metrics with known inputs"""
import numpy as np
from tailrisk import (
    cvar_loss,
    loss_at_risk,
    tail_coverage_ratio,
    detection_rate,
    tail_validation_summary
)

def test_metrics():
    print("Testing Metrics...")

    # Perfect predictions
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred_perfect = y_true.copy()

    # Test perfect case
    cvar = cvar_loss(y_true, y_pred_perfect, alpha=0.95)
    tcr = tail_coverage_ratio(y_true, y_pred_perfect, quantile=0.95)

    assert cvar == 0, "Perfect prediction should have 0 CVaR"
    assert abs(tcr - 1.0) < 0.01, "Perfect prediction should have TCR ≈ 1.0"

    # Test underestimation
    y_pred_under = y_true * 0.5  # 50% underestimation

    tcr_under = tail_coverage_ratio(y_true, y_pred_under, quantile=0.95)
    assert tcr_under < 1.0, "Underestimation should have TCR < 1.0"

    # Test overestimation
    y_pred_over = y_true * 1.5  # 50% overestimation

    tcr_over = tail_coverage_ratio(y_true, y_pred_over, quantile=0.95)
    assert tcr_over > 1.0, "Overestimation should have TCR > 1.0"

    # Test summary
    metrics = tail_validation_summary(y_true, y_pred_perfect)
    required_keys = ['mse', 'cvar_95', 'tcr_99', 'detection_95']
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"

    print("✓ All metrics tests passed")
    return True

if __name__ == "__main__":
    test_metrics()
```

Run:
```bash
python test_metrics_validation.py
```

---

## Manual Testing

### Test 4: Scikit-learn Compatibility

```python
"""Test sklearn compatibility"""
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tailrisk import LaRRegressor
import numpy as np

# Data
X = np.random.randn(200, 5)
y = np.random.exponential(scale=100, size=200)

# Test 1: Grid Search
print("Testing GridSearchCV compatibility...")
param_grid = {'alpha': [1.0, 2.0]}
grid = GridSearchCV(LaRRegressor(), param_grid, cv=3)
grid.fit(X, y)
print(f"✓ Best params: {grid.best_params_}")

# Test 2: Pipeline
print("\nTesting Pipeline compatibility...")
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LaRRegressor(alpha=2.0))
])
pipe.fit(X[:150], y[:150])
predictions = pipe.predict(X[150:])
print(f"✓ Pipeline predictions shape: {predictions.shape}")

# Test 3: Cross-validation
print("\nTesting cross_val_score...")
scores = cross_val_score(LaRRegressor(), X, y, cv=3, scoring='neg_mean_squared_error')
print(f"✓ CV scores: {-scores.mean():.2f} ± {scores.std():.2f}")

print("\n✓ All sklearn compatibility tests passed")
```

Save as `test_sklearn_compat.py` and run:
```bash
python test_sklearn_compat.py
```

### Test 5: Edge Cases

```python
"""Test edge cases and error handling"""
import numpy as np
from tailrisk import LaRRegressor, tail_validation_summary

print("Testing edge cases...")

# Test 1: Small dataset
X_small = np.random.randn(10, 3)
y_small = np.random.exponential(scale=100, size=10)

model = LaRRegressor()
model.fit(X_small, y_small)
predictions = model.predict(X_small)
assert len(predictions) == len(y_small), "Small dataset failed"
print("✓ Small dataset test passed")

# Test 2: Single feature
X_single = np.random.randn(100, 1)
y_single = np.random.exponential(scale=100, size=100)

model = LaRRegressor()
model.fit(X_single, y_single)
predictions = model.predict(X_single)
assert len(predictions) == len(y_single), "Single feature failed"
print("✓ Single feature test passed")

# Test 3: All zeros in target (edge case)
y_zeros = np.zeros(100)
X_zeros = np.random.randn(100, 5)

model = LaRRegressor()
model.fit(X_zeros, y_zeros)  # Should not crash
predictions = model.predict(X_zeros)
print("✓ Zero target test passed")

# Test 4: Metrics with identical predictions
y_true = np.random.exponential(scale=100, size=50)
y_pred_same = np.ones(50) * y_true.mean()

metrics = tail_validation_summary(y_true, y_pred_same)
assert 'cvar_95' in metrics, "Metrics failed on constant predictions"
print("✓ Constant prediction test passed")

print("\n✓ All edge case tests passed")
```

Save as `test_edge_cases.py` and run:
```bash
python test_edge_cases.py
```

---

## Performance Testing

### Test 6: Timing Benchmark

```python
"""Benchmark training and prediction times"""
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tailrisk import LaRRegressor, HybridMetaLearner

def benchmark_model(model, X_train, y_train, X_test, name):
    """Time model training and prediction."""
    # Training
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # Prediction
    start = time.time()
    model.predict(X_test)
    pred_time = time.time() - start

    print(f"{name}:")
    print(f"  Training: {train_time:.2f}s")
    print(f"  Prediction: {pred_time:.4f}s")
    return train_time, pred_time

# Generate data
np.random.seed(42)
X = np.random.randn(5000, 20)
y = np.random.exponential(scale=1000, size=5000)

X_train, X_test = X[:4000], X[4000:]
y_train, y_test = y[:4000], y[4000:]

print("Performance Benchmark")
print("="*50)

# LaRRegressor
benchmark_model(
    LaRRegressor(alpha=2.0),
    X_train, y_train, X_test,
    "LaRRegressor"
)

# HybridMetaLearner
base_models = [
    ('rf', RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42))
]

benchmark_model(
    HybridMetaLearner(base_estimators=base_models, cv_folds=3),
    X_train, y_train, X_test,
    "HybridMetaLearner"
)
```

Save as `test_performance.py` and run:
```bash
python test_performance.py
```

---

## Continuous Integration

### GitHub Actions Workflow

If using GitHub, create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Run tests
      run: |
        pytest tests/ -v --cov=tailrisk

    - name: Run quick start example
      run: |
        python examples/quick_start.py
```

---

## Test Checklist

Before releasing or deploying, ensure:

- [ ] All unit tests pass (`pytest tests/`)
- [ ] Quick start example runs without errors
- [ ] All imports work correctly
- [ ] Documentation examples are up-to-date
- [ ] Metrics produce expected values on known inputs
- [ ] Models work with scikit-learn utilities (GridSearchCV, Pipeline)
- [ ] Edge cases handled gracefully (small datasets, single features, etc.)
- [ ] Performance is acceptable (timing benchmarks)
- [ ] No deprecation warnings from dependencies

---

## Troubleshooting Test Failures

### "ModuleNotFoundError: No module named 'tailrisk'"

**Solution:**
```bash
# Install package in development mode
pip install -e .
```

### "ImportError: cannot import name 'HybridMetaLearner'"

**Solution:**
```bash
# Clear Python cache and reinstall
find . -type d -name __pycache__ -exec rm -rf {} +  # Unix
pip install --force-reinstall -e .
```

### Tests hang or run very slowly

**Solution:**
- Reduce dataset sizes in tests
- Decrease `n_estimators` in tree models
- Use fewer `cv_folds` in HybridMetaLearner
- Run tests in parallel: `pytest tests/ -n auto` (requires pytest-xdist)

### Random test failures

**Solution:**
- Ensure all random operations use fixed seeds
- Check for race conditions in parallel code
- Verify test independence (tests shouldn't depend on each other)

---

## Summary

Regular testing ensures TailRisk works correctly:

1. **Quick tests** - Verify installation after setup
2. **Unit tests** - Run automatically on code changes
3. **Integration tests** - Test complete workflows
4. **Manual tests** - Verify scikit-learn compatibility and edge cases
5. **Performance tests** - Monitor speed and efficiency

**Recommended testing workflow:**
```bash
# Quick validation
python -c "import tailrisk; print('OK')"

# Full test suite
pytest tests/ -v

# Run quick start
python examples/quick_start.py

# Benchmark (occasionally)
python test_performance.py
```

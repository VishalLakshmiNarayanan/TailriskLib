# TailRisk API Reference

Complete API documentation for the TailRisk package.

## Table of Contents
- [Models](#models)
  - [LaRRegressor](#larregressor)
  - [HybridMetaLearner](#hybridmetalearner)
  - [CVaRWeightedEnsemble](#cvarweightedensemble)
- [Metrics](#metrics)
- [Utilities](#utilities)
- [Preprocessing](#preprocessing)

---

## Models

### LaRRegressor

**Class**: `tailrisk.LaRRegressor`

Loss-at-Risk weighted regression model that assigns higher importance to larger target values, making it more sensitive to tail risk prediction.

#### Constructor

```python
LaRRegressor(alpha=2.0, base_estimator=None)
```

**Parameters:**

- **alpha** : `float`, default=2.0
  - Weight scaling factor controlling the emphasis on large claims
  - Higher values increase focus on tail risk
  - Recommended range: 1.0 to 3.0
  - Formula: `weight = 1 + alpha * (y / max(y))`

- **base_estimator** : `estimator object`, default=None
  - Base regression model to use for weighted fitting
  - Must support `sample_weight` parameter in `fit()` method
  - If None, uses `LinearRegression()`
  - Can use any sklearn-compatible regressor (Ridge, Lasso, etc.)

#### Methods

##### fit(X, y, sample_weight=None)

Fit the LaR-weighted regression model.

**Parameters:**
- **X** : array-like, shape (n_samples, n_features)
  - Training feature matrix
- **y** : array-like, shape (n_samples,)
  - Target values
- **sample_weight** : array-like, shape (n_samples,), optional
  - Additional sample weights (multiplied with LaR weights)

**Returns:**
- **self** : Returns the instance itself

##### predict(X)

Generate predictions using the fitted model.

**Parameters:**
- **X** : array-like, shape (n_samples, n_features)
  - Feature matrix to predict

**Returns:**
- **y_pred** : ndarray, shape (n_samples,)
  - Predicted target values

#### Attributes

- **base_estimator_** : estimator
  - The fitted base estimator after training

#### Example

```python
from tailrisk import LaRRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Create heavy-tailed data
X = np.random.randn(1000, 10)
y = np.random.exponential(scale=1000, size=1000)

# Basic usage with default LinearRegression
model = LaRRegressor(alpha=2.0)
model.fit(X[:800], y[:800])
predictions = model.predict(X[800:])

# With custom base estimator
model_rf = LaRRegressor(
    alpha=2.5,
    base_estimator=RandomForestRegressor(n_estimators=100)
)
model_rf.fit(X[:800], y[:800])
predictions_rf = model_rf.predict(X[800:])
```

---

### HybridMetaLearner

**Class**: `tailrisk.HybridMetaLearner`

Advanced two-stage meta-learning ensemble that combines quantile regression and LaR-optimized weighting for superior tail risk prediction.

#### Architecture

1. **Stage 1**: Generate meta-features from diverse base models using cross-validation
2. **Stage 2a**: Train quantile regression meta-model (focuses on high quantiles)
3. **Stage 2b**: Optimize LaR-weighted combination of base predictions
4. **Stage 3**: Blend both meta-models for final predictions

#### Constructor

```python
HybridMetaLearner(
    base_estimators,
    quantile=0.95,
    blend_lambda=0.25,
    lar_alpha=1.5,
    cv_folds=5
)
```

**Parameters:**

- **base_estimators** : `list of (str, estimator) tuples`
  - List of base models to ensemble
  - Each tuple: (name, estimator_instance)
  - Recommended: Use diverse model types (trees, linear, boosting)
  - Example: `[('rf', RandomForestRegressor()), ('gb', GradientBoostingRegressor())]`

- **quantile** : `float`, default=0.95
  - Target quantile for quantile regression meta-model
  - Recommended range: 0.90 to 0.99
  - Higher values focus more on extreme tails
  - 0.95 typically provides good balance

- **blend_lambda** : `float`, default=0.25
  - Blending weight between quantile and LaR models
  - Formula: `pred = lambda * quantile_pred + (1-lambda) * lar_pred`
  - Range: 0.0 to 1.0
  - Recommended: 0.2 to 0.3 for balanced performance
  - Higher values favor tail detection over overall accuracy

- **lar_alpha** : `float`, default=1.5
  - Weight scaling for LaR optimization in stage 2b
  - Controls emphasis on large values in LaR model
  - Recommended range: 1.0 to 2.0

- **cv_folds** : `int`, default=5
  - Number of cross-validation folds for meta-feature generation
  - Higher values reduce overfitting but increase training time
  - Typical range: 3 to 10

#### Methods

##### fit(X, y)

Train the hybrid meta-learner on data.

**Parameters:**
- **X** : array-like, shape (n_samples, n_features)
  - Training feature matrix
- **y** : array-like, shape (n_samples,)
  - Target values

**Returns:**
- **self** : Returns the instance itself

##### predict(X)

Generate predictions using the fitted hybrid model.

**Parameters:**
- **X** : array-like, shape (n_samples, n_features)
  - Feature matrix to predict

**Returns:**
- **y_pred** : ndarray, shape (n_samples,)
  - Predicted target values (blended from quantile + LaR)

#### Attributes

- **base_estimators_** : list of estimators
  - Fitted base models
- **quantile_meta_** : QuantileRegressor
  - Fitted quantile regression meta-model
- **lar_weights_** : ndarray
  - Optimized weights for LaR-weighted combination

#### Example

```python
from tailrisk import HybridMetaLearner
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb

# Define diverse base models
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
    ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42)),
    ('ridge', Ridge(alpha=1.0))
]

# Create and train model
model = HybridMetaLearner(
    base_estimators=base_models,
    quantile=0.95,
    blend_lambda=0.25,
    lar_alpha=1.5,
    cv_folds=5
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

### CVaRWeightedEnsemble

**Class**: `tailrisk.CVaRWeightedEnsemble`

Ensemble that weights models based on inverse CVaR performance on validation data.

#### Constructor

```python
CVaRWeightedEnsemble(estimators, alpha=0.95)
```

**Parameters:**

- **estimators** : `list of (str, estimator) tuples`
  - List of models to ensemble
  - Each tuple: (name, estimator_instance)

- **alpha** : `float`, default=0.95
  - CVaR percentile for weight calculation
  - Models with lower CVaR get higher weight

#### Methods

##### fit(X, y)
Train all estimators and calculate CVaR-based weights.

##### predict(X)
Generate weighted average predictions.

#### Example

```python
from tailrisk import CVaRWeightedEnsemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

estimators = [
    ('rf', RandomForestRegressor(n_estimators=100)),
    ('ridge', Ridge(alpha=1.0))
]

model = CVaRWeightedEnsemble(estimators, alpha=0.95)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## Metrics

All metrics are available from `tailrisk.metrics` or directly from `tailrisk`.

### cvar_loss

**Function**: `tailrisk.cvar_loss(y_true, y_pred, alpha=0.95)`

Calculate Conditional Value-at-Risk (CVaR) - average squared error in the worst (1-alpha)% of predictions.

**Parameters:**
- **y_true** : array-like, shape (n_samples,)
  - True target values
- **y_pred** : array-like, shape (n_samples,)
  - Predicted values
- **alpha** : float, default=0.95
  - Percentile threshold (0.90 to 0.99 typical)

**Returns:**
- **cvar** : float
  - CVaR metric value (lower is better)

**Interpretation:**
- Measures tail risk performance
- Lower values indicate better extreme value prediction
- Ignores average performance, focuses on worst cases

```python
from tailrisk import cvar_loss

# Calculate CVaR at 95th percentile
cvar_95 = cvar_loss(y_test, y_pred, alpha=0.95)
print(f"CVaR(95%): ${cvar_95:,.2f}")
```

---

### loss_at_risk

**Function**: `tailrisk.loss_at_risk(y_true, y_pred, alpha=0.5)`

Calculate Loss-at-Risk metric - weighted MSE emphasizing large target values.

**Parameters:**
- **y_true** : array-like, shape (n_samples,)
  - True target values
- **y_pred** : array-like, shape (n_samples,)
  - Predicted values
- **alpha** : float, default=0.5
  - Weight scaling factor

**Returns:**
- **lar** : float
  - LaR metric value

**Formula:**
```
weights = 1 + alpha * (y_true / max(y_true))
lar = mean(weights * (y_true - y_pred)^2)
```

```python
from tailrisk import loss_at_risk

lar = loss_at_risk(y_test, y_pred, alpha=2.0)
print(f"LaR: {lar:,.2f}")
```

---

### tail_coverage_ratio

**Function**: `tailrisk.tail_coverage_ratio(y_true, y_pred, quantile=0.99)`

Calculate what fraction of extreme tail value is captured by predictions.

**Parameters:**
- **y_true** : array-like, shape (n_samples,)
  - True target values
- **y_pred** : array-like, shape (n_samples,)
  - Predicted values
- **quantile** : float, default=0.99
  - Threshold for defining "extreme" cases

**Returns:**
- **tcr** : float
  - Tail coverage ratio (1.0 is perfect)

**Interpretation:**
- **tcr = 1.0**: Perfect coverage (100% of tail value captured)
- **tcr < 1.0**: Underprediction (dangerous for risk management)
- **tcr > 1.0**: Overprediction (conservative, may be acceptable)

```python
from tailrisk import tail_coverage_ratio

tcr_99 = tail_coverage_ratio(y_test, y_pred, quantile=0.99)
print(f"Captures {tcr_99*100:.1f}% of extreme tail value")
```

---

### detection_rate

**Function**: `tailrisk.detection_rate(y_true, y_pred, quantile=0.95)`

Calculate percentage of actual extreme cases correctly predicted as extreme.

**Parameters:**
- **y_true** : array-like, shape (n_samples,)
  - True target values
- **y_pred** : array-like, shape (n_samples,)
  - Predicted values
- **quantile** : float, default=0.95
  - Threshold for defining "extreme"

**Returns:**
- **detection_rate** : float
  - Fraction of extremes correctly identified (0.0 to 1.0)

**Use case:** Early warning system performance

```python
from tailrisk import detection_rate

det_95 = detection_rate(y_test, y_pred, quantile=0.95)
print(f"Detected {det_95*100:.1f}% of extreme claims")
```

---

### tail_validation_summary

**Function**: `tailrisk.tail_validation_summary(y_true, y_pred)`

Comprehensive summary of all tail risk metrics.

**Parameters:**
- **y_true** : array-like, shape (n_samples,)
  - True target values
- **y_pred** : array-like, shape (n_samples,)
  - Predicted values

**Returns:**
- **metrics** : dict
  - Dictionary with keys:
    - `'mse'`: Overall MSE
    - `'mse_extreme'`: MSE on extreme values (>95th percentile)
    - `'cvar_90'`, `'cvar_95'`, `'cvar_99'`: CVaR at different thresholds
    - `'tcr_95'`, `'tcr_99'`: Tail coverage ratios
    - `'detection_90'`, `'detection_95'`, `'detection_99'`: Detection rates
    - `'lar'`: Loss-at-Risk

```python
from tailrisk import tail_validation_summary

metrics = tail_validation_summary(y_test, y_pred)
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
```

---

## Utilities

### print_tail_validation

**Function**: `tailrisk.utils.print_tail_validation(y_true, y_pred, model_name='Model')`

Print formatted validation report with all tail risk metrics.

```python
from tailrisk.utils import print_tail_validation

print_tail_validation(y_test, y_pred, model_name="Hybrid Meta-Learner")
```

---

### compare_models

**Function**: `tailrisk.utils.compare_models(y_true, predictions_dict)`

Compare multiple models side-by-side on tail risk metrics.

**Parameters:**
- **y_true** : array-like
  - True target values
- **predictions_dict** : dict
  - Dictionary mapping model names to prediction arrays
  - Format: `{'Model1': y_pred1, 'Model2': y_pred2, ...}`

**Returns:**
- **results** : dict
  - Nested dictionary with metrics for each model

```python
from tailrisk.utils import compare_models

results = compare_models(y_test, {
    'Baseline': y_pred_baseline,
    'LaR': y_pred_lar,
    'Hybrid': y_pred_hybrid
})
```

---

### plot_tail_comparison

**Function**: `tailrisk.utils.plot_tail_comparison(y_true, y_pred_baseline, y_pred_tailrisk, quantiles=[0.90, 0.95, 0.99])`

Visualize tail risk performance comparison.

**Parameters:**
- **y_true** : array-like
  - True target values
- **y_pred_baseline** : array-like
  - Baseline model predictions
- **y_pred_tailrisk** : array-like
  - TailRisk model predictions
- **quantiles** : list of float
  - Quantiles to highlight

```python
from tailrisk.utils import plot_tail_comparison

plot_tail_comparison(y_test, y_pred_baseline, y_pred_hybrid)
```

---

## Preprocessing

### Transforms

Additional preprocessing utilities for heavy-tailed data.

See `tailrisk.preprocessing` for data transformation utilities.

---

## Complete Usage Example

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tailrisk import HybridMetaLearner, tail_validation_summary
from tailrisk.utils import print_tail_validation, compare_models

# Generate data
X = np.random.randn(5000, 10)
y = np.random.exponential(scale=1000, size=5000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build model
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
]

model = HybridMetaLearner(
    base_estimators=base_models,
    quantile=0.95,
    blend_lambda=0.25
)

# Train and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
metrics = tail_validation_summary(y_test, y_pred)
print_tail_validation(y_test, y_pred, model_name="Hybrid Meta-Learner")
```

---

## Scikit-learn Compatibility

All TailRisk models are fully compatible with scikit-learn:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from tailrisk import LaRRegressor

# Grid search
param_grid = {'alpha': [1.0, 1.5, 2.0, 2.5]}
grid = GridSearchCV(LaRRegressor(), param_grid, cv=5)
grid.fit(X_train, y_train)

# Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LaRRegressor(alpha=2.0))
])
pipe.fit(X_train, y_train)
```

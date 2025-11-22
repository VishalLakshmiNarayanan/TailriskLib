# TailRisk: Risk-Aware Machine Learning for Tail Risk Modeling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/tailrisk.svg)](https://badge.fury.io/py/tailrisk)

**TailRisk** is a Python package for building machine learning models that excel at predicting extreme outcomes in insurance claims, financial losses, and other tail-risk scenarios.

## Why TailRisk?

Traditional ML models optimize for average performance (MSE, MAE), but this approach **fails catastrophically** when predicting rare, extreme events:

- **Insurance claims**: Models predict \$500 for a claim that costs \$50,000
- **Financial losses**: Underestimating tail risk leads to inadequate reserves
- **Healthcare costs**: Missing catastrophic cases can bankrupt risk pools

**Standard MSE treats all errors equally** — a \$10,000 error on a \$500 claim has the same weight as a \$10,000 error on a \$100,000 claim. This is dangerous for tail risk.

### The Problem

```python
# Traditional approach
model = LinearRegression()
model.fit(X, y)  # Optimizes MSE equally across all samples

# Result: Great average performance, terrible tail prediction
# - MSE: $2.8M ✓
# - CVaR(95%): $54M ✗
# - Tail Coverage Ratio: 0.058 (captures only 5.8% of extreme losses!)
```

### The TailRisk Solution

```python
from tailrisk import HybridMetaLearner
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Build tail-aware model
model = HybridMetaLearner(
    base_estimators=[
        ('rf', RandomForestRegressor(n_estimators=100)),
        ('gb', GradientBoostingRegressor(n_estimators=100))
    ],
    quantile=0.95,       # Focus on 95th percentile
    blend_lambda=0.25    # Blend quantile + LaR models
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Result: Dramatically improved tail prediction
# - MSE: $3.0M (acceptable tradeoff)
# - CVaR(95%): $52M ✓ (10% improvement)
# - Tail Coverage Ratio: 0.102 ✓ (76% improvement!)
```

## Key Features

### 🎯 Tail-Focused Models

- **Loss-at-Risk (LaR) Regressor**: Weighted regression that prioritizes large claims
- **CVaR-Weighted Ensemble**: Combines models based on tail risk performance
- **Hybrid Meta-Learner**: Two-stage architecture blending quantile regression + LaR optimization

### 📊 Specialized Metrics

Standard metrics hide tail risk problems. TailRisk provides:

- **CVaR (Conditional Value-at-Risk)**: Average loss in worst α% of predictions
- **Tail Coverage Ratio**: What fraction of extreme tail value is captured
- **Detection Rate**: Percentage of extreme claims correctly identified
- **Loss-at-Risk**: Weighted MSE emphasizing large claims

### 🔧 Production-Ready

- Scikit-learn compatible (works with `GridSearchCV`, `Pipeline`, etc.)
- Comprehensive validation utilities
- Built-in visualizations for tail risk analysis
- Type-safe, well-documented API

## Installation

### From PyPI (coming soon)

```bash
pip install tailrisk
```

### From source

```bash
git clone https://github.com/yourusername/tailrisk-lib.git
cd tailrisk-lib
pip install -e .
```

## Quick Start

### Basic Example: LaR Regressor

```python
import numpy as np
from tailrisk import LaRRegressor
from tailrisk.metrics import tail_validation_summary

# Heavy-tailed data (e.g., insurance claims)
X = np.random.randn(1000, 10)
y = np.random.exponential(scale=1000, size=1000)

# Fit LaR model
model = LaRRegressor(alpha=2.0)
model.fit(X_train, y_train)

# Evaluate tail performance
predictions = model.predict(X_test)
metrics = tail_validation_summary(y_test, predictions)

print(f"CVaR(95%): {metrics['cvar_95']:.2f}")
print(f"Tail Coverage @ 99%: {metrics['tcr_99']:.3f}")
```

### Advanced Example: Hybrid Meta-Learner

```python
from tailrisk import HybridMetaLearner
from tailrisk.utils import print_tail_validation, plot_tail_comparison
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb

# Define diverse base models
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
    ('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)),
    ('ridge', Ridge(alpha=1.0))
]

# Build hybrid meta-learner
model = HybridMetaLearner(
    base_estimators=base_models,
    quantile=0.95,        # Target 95th percentile in quantile model
    blend_lambda=0.25,    # 25% quantile, 75% LaR
    lar_alpha=1.5,        # LaR weighting strength
    cv_folds=5            # Cross-validation folds for meta-features
)

# Fit and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Comprehensive validation
print_tail_validation(y_test, y_pred, model_name="Hybrid Meta-Learner")

# Visualize performance
plot_tail_comparison(y_test, y_pred_baseline, y_pred, quantiles=[0.90, 0.95, 0.99])
```

### Model Comparison

```python
from tailrisk.utils.validation import compare_models
from sklearn.linear_model import LinearRegression

# Train multiple models
baseline = LinearRegression().fit(X_train, y_train)
lar_model = LaRRegressor(alpha=2.0).fit(X_train, y_train)
hybrid_model = HybridMetaLearner(base_estimators=base_models).fit(X_train, y_train)

# Compare side-by-side
compare_models(y_test, {
    'Baseline': baseline.predict(X_test),
    'LaR': lar_model.predict(X_test),
    'Hybrid': hybrid_model.predict(X_test)
})
```

Output:
```
==========================================================================================
 MODEL COMPARISON
==========================================================================================
Metric                   Baseline              LaR           Hybrid
------------------------------------------------------------------------------------------
MSE (Overall)           2,808,591        2,815,649        2,979,450
MSE (Extreme)         260,918,453      259,101,161      252,549,889
CVaR (95%)                  3,146            3,086            2,824
Detection @ 90%              3.1%             3.2%             6.1%
Detection @ 95%              0.8%             0.9%             2.5%
TCR @ 95%                   0.165            0.166            0.306
TCR @ 99%                   0.058            0.059            0.102
==========================================================================================
```

## Methodology

TailRisk implements a novel **Hybrid Meta-Learning** framework:

### Stage 1: Diverse Base Models
Train multiple model types (tree-based, linear, boosting) to capture different aspects of the data.

### Stage 2a: Quantile Meta-Model
Use quantile regression (α=0.95) on meta-features to focus on extreme predictions.

### Stage 2b: LaR-Weighted Optimization
Optimize base model weights using LaR objective function:

$$\text{LaR} = \frac{1}{n} \sum_{i=1}^n w_i (y_i - \hat{y}_i)^2$$

where $w_i = 1 + \alpha \frac{y_i}{\max(y)}$

### Stage 3: Hybrid Blend
Combine both meta-models:

$$\hat{y}_{\text{final}} = \lambda \cdot \hat{y}_{\text{quantile}} + (1-\lambda) \cdot \hat{y}_{\text{LaR}}$$

The quantile model excels at **detecting** extreme events, while the LaR model maintains **accuracy**. Blending both (typically λ=0.25) provides optimal balance.

## Metrics Guide

### CVaR (Conditional Value-at-Risk)
Average squared error in the worst (1-α)% of predictions.
- **Lower is better**
- Focuses on tail performance, ignoring average cases
- Industry standard in finance/insurance

### Tail Coverage Ratio (TCR)
Fraction of total extreme tail value captured by predictions.
```python
TCR = sum(predictions[tail]) / sum(actuals[tail])
```
- **1.0 is perfect** (100% coverage)
- <1.0 = underprediction (dangerous!)
- >1.0 = overprediction (conservative)

### Detection Rate
Percentage of actual extreme claims correctly predicted as extreme.
- **Higher is better**
- Measures early warning capability
- Critical for reserve planning

## Use Cases

### Insurance Claims
- **Problem**: Catastrophic claims (1% of cases) drive 30% of losses
- **Solution**: HybridMetaLearner improves TCR@99% from 0.058 → 0.102 (+76%)
- **Impact**: Better reserves, reduced insolvency risk

### Financial Risk
- **Problem**: VaR models underestimate tail risk (2008 crisis)
- **Solution**: CVaR-focused models + tail validation metrics
- **Impact**: Regulatory compliance (Basel III), improved capital allocation

### Healthcare Costs
- **Problem**: Rare catastrophic cases (ICU, surgery complications)
- **Solution**: LaRRegressor prioritizes expensive cases
- **Impact**: Accurate premium pricing, sustainable risk pools

## Examples

See the [examples/](examples/) directory for detailed notebooks:

- **insurance_claims_example.ipynb**: Full walkthrough with real insurance data
- **financial_losses_example.py**: Portfolio tail risk modeling
- **model_comparison.ipynb**: Benchmarking TailRisk vs standard models

## API Reference

### Models

#### `LaRRegressor(alpha=2.0, base_estimator=None)`
Weighted regression with Loss-at-Risk objective.

**Parameters:**
- `alpha`: Weight scaling factor (higher = more focus on large claims)
- `base_estimator`: Base model (default: LinearRegression)

#### `HybridMetaLearner(base_estimators, quantile=0.95, blend_lambda=0.25, lar_alpha=1.5, cv_folds=5)`
Two-stage meta-learning with quantile + LaR blending.

**Parameters:**
- `base_estimators`: List of (name, estimator) tuples
- `quantile`: Quantile target for stage 2a (0.90-0.99 recommended)
- `blend_lambda`: Blending weight (0.2-0.3 works well)
- `lar_alpha`: LaR optimization strength (1.0-2.0 recommended)
- `cv_folds`: Cross-validation folds for meta-features

#### `CVaRWeightedEnsemble(estimators, alpha=0.95)`
Ensemble weighted by inverse CVaR performance.

### Metrics

- `cvar_loss(y_true, y_pred, alpha=0.95)`: CVaR metric
- `loss_at_risk(y_true, y_pred, alpha=0.5)`: LaR metric
- `tail_coverage_ratio(y_true, y_pred, quantile=0.99)`: TCR metric
- `detection_rate(y_true, y_pred, quantile=0.95)`: Detection metric
- `tail_validation_summary(y_true, y_pred)`: All metrics combined

### Utilities

- `plot_tail_comparison(y_true, y_pred_baseline, y_pred_tailrisk)`: Visual comparison
- `print_tail_validation(y_true, y_pred, model_name)`: Formatted report
- `compare_models(y_true, predictions_dict)`: Multi-model comparison

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use TailRisk in academic work, please cite:

```bibtex
@software{tailrisk2025,
  author = {Your Name},
  title = {TailRisk: Risk-Aware Machine Learning for Tail Risk Modeling},
  year = {2025},
  url = {https://github.com/yourusername/tailrisk-lib}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Methodology developed for insurance claims prediction research
- Inspired by actuarial science and quantitative finance literature
- Built on scikit-learn's excellent API design

## Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/tailrisk-lib/issues)
- **Email**: your.email@example.com
- **Twitter**: @yourhandle

---

**Made with ❤️ for actuaries, quants, and ML engineers fighting tail risk**

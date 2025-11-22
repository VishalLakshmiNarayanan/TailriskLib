# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial package structure
- Core models:
  - `LaRRegressor`: Loss-at-Risk weighted regression
  - `CVaRWeightedEnsemble`: CVaR-based model weighting
  - `HybridMetaLearner`: Two-stage meta-learning with quantile + LaR blending
- Tail risk metrics:
  - `cvar_loss()`: Conditional Value-at-Risk
  - `loss_at_risk()`: Weighted MSE for tail events
  - `tail_coverage_ratio()`: Tail value capture metric
  - `detection_rate()`: Extreme event detection metric
  - `tail_validation_summary()`: Comprehensive validation report
- Utilities:
  - `plot_tail_comparison()`: Visual model comparison
  - `plot_predictions_by_quantile()`: Quantile-colored scatter plots
  - `plot_residual_distribution()`: Residual analysis with VaR markers
  - `print_tail_validation()`: Formatted validation output
  - `compare_models()`: Side-by-side model comparison
- Preprocessing:
  - `log_transform_target()`: Log transformation for heavy-tailed targets
  - `filter_nonzero_claims()`: Filter to non-zero claims
- Comprehensive test suite
- Documentation and examples

## [0.1.0] - 2025-01-XX

### Added
- Initial public release
- Full scikit-learn API compatibility
- Python 3.8+ support
- MIT License

[Unreleased]: https://github.com/yourusername/tailrisk-lib/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/tailrisk-lib/releases/tag/v0.1.0

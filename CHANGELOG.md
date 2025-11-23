# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2025-11-23

### Changed
- Updated README and documentation formatting
- Improved PyPI package description

## [0.1.2] - 2025-11-22

### Added
- Comprehensive technical documentation (30-page PDF)
- Detailed API reference with mathematical formulations
- Best practices guide for hyperparameter tuning
- Model selection guidelines
- Scikit-learn integration examples (GridSearchCV, Pipeline, Cross-validation)
- Advanced topics: custom base estimators, ensemble stacking, production deployment
- Troubleshooting guide with common issues and solutions
- Theoretical background section covering quantile regression, extreme value theory, and CVaR

### Changed
- Improved documentation structure with clear use cases
- Enhanced installation instructions with verification steps
- Updated examples with complete workflows including train/test splits
- Clarified metric interpretations (CVaR, TCR, Detection Rate)

### Fixed
- Documentation consistency across README and technical docs
- Import statements in code examples
- Author information and contact details

## [0.1.1] - 2025-01-XX

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

[Unreleased]: https://github.com/VishalLakshmiNarayanan/TailriskLib/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/VishalLakshmiNarayanan/TailriskLib/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/VishalLakshmiNarayanan/TailriskLib/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/VishalLakshmiNarayanan/TailriskLib/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/VishalLakshmiNarayanan/TailriskLib/releases/tag/v0.1.0

# Contributing to TailRisk

Thank you for your interest in contributing to TailRisk! This document provides guidelines for contributing.

## Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/VishalLakshmiNarayanan/TailriskLib.git
cd TailriskLib
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install in development mode

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode along with development dependencies.

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tailrisk --cov-report=html

# Run specific test file
pytest tests/test_metrics.py

# Run with verbose output
pytest -v
```

## Code Style

We follow PEP 8 style guidelines. Format your code with:

```bash
# Auto-format code
black tailrisk/

# Check for style issues
flake8 tailrisk/
```

## Pull Request Process

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** with clear, descriptive commit messages
3. **Add tests** for any new functionality
4. **Ensure tests pass**: Run `pytest` locally
5. **Update documentation** if needed (README, docstrings)
6. **Submit a pull request** with a clear description of changes

### PR Checklist

- [ ] Tests pass locally
- [ ] Code is formatted with `black`
- [ ] New features have tests
- [ ] Documentation updated (if applicable)
- [ ] Commit messages are clear and descriptive

## Adding New Features

### New Models

If adding a new tail-risk aware model:

1. Create file in `tailrisk/models/`
2. Inherit from `sklearn.base.BaseEstimator` and `RegressorMixin`
3. Implement `fit()` and `predict()` methods
4. Add comprehensive docstrings
5. Add tests in `tests/test_models.py`
6. Update `tailrisk/models/__init__.py`

### New Metrics

If adding a new tail risk metric:

1. Add function to `tailrisk/metrics/risk_metrics.py`
2. Include numpy-style docstring with examples
3. Add tests in `tests/test_metrics.py`
4. Update `tailrisk/metrics/__init__.py`

## Documentation

- Use [numpy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html)
- Include examples in docstrings
- Update README.md for significant changes

## Reporting Issues

### Bug Reports

Include:
- Python version
- TailRisk version
- Minimal reproducible example
- Expected vs actual behavior
- Error messages (if applicable)

### Feature Requests

Describe:
- Use case / motivation
- Proposed API (if applicable)
- How it relates to tail risk modeling

## Code of Conduct

Be respectful and constructive in all interactions. We're building tools to help people manage risk - let's do it collaboratively!

## Questions?

Open an issue or email the maintainer directly.

---

Thank you for contributing to TailRisk! 🎯

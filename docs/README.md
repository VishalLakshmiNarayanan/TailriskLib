# TailRisk Documentation

Welcome to the TailRisk documentation! This directory contains comprehensive guides for using the TailRisk package.

## Documentation Structure

### 📚 Core Documentation

1. **[Installation Guide](INSTALLATION.md)**
   - System requirements
   - Installation methods (PyPI, source, virtual environments)
   - Troubleshooting installation issues
   - Platform-specific notes

2. **[API Reference](API_REFERENCE.md)**
   - Complete API documentation for all modules
   - Models: `LaRRegressor`, `HybridMetaLearner`, `CVaRWeightedEnsemble`
   - Metrics: `cvar_loss`, `tail_coverage_ratio`, `detection_rate`, etc.
   - Utilities: Visualization and validation helpers
   - Full parameter descriptions and usage examples

3. **[Tutorial](TUTORIAL.md)**
   - Step-by-step guide from basics to advanced usage
   - Understanding tail risk problems
   - Working with LaRRegressor
   - Building HybridMetaLearner models
   - Model evaluation and comparison
   - Hyperparameter tuning
   - Production deployment

4. **[Testing Guide](TESTING_GUIDE.md)**
   - Quick verification tests
   - Running unit tests
   - Integration testing
   - Manual testing procedures
   - Performance benchmarking
   - CI/CD setup

## Quick Links

### Getting Started
- New to TailRisk? Start with the main [README](../README.md)
- Want to jump in? See [Quick Start](../README.md#quick-start)
- Need to install? Check [Installation Guide](INSTALLATION.md)

### Learning Resources
- **Beginner**: [Tutorial](TUTORIAL.md) - Comprehensive step-by-step guide
- **Reference**: [API Documentation](API_REFERENCE.md) - Complete API reference
- **Examples**: [Examples Directory](../examples/) - Working code samples

### Testing & Validation
- Verify installation: [Testing Guide](TESTING_GUIDE.md#quick-tests)
- Run test suite: [Testing Guide](TESTING_GUIDE.md#unit-tests)
- Performance testing: [Testing Guide](TESTING_GUIDE.md#performance-testing)

## Navigation Guide

### By User Type

**Data Scientists / ML Engineers**
1. Read [Tutorial](TUTORIAL.md) sections 1-3 (Understanding the problem)
2. Try examples from [Tutorial](TUTORIAL.md) sections 4-5
3. Reference [API Documentation](API_REFERENCE.md) as needed

**Software Engineers / DevOps**
1. Follow [Installation Guide](INSTALLATION.md)
2. Review [Testing Guide](TESTING_GUIDE.md)
3. See [Tutorial](TUTORIAL.md) section on production deployment

**Researchers / Academics**
1. Read main [README](../README.md) methodology section
2. Study [Tutorial](TUTORIAL.md) for detailed explanations
3. Review [API Reference](API_REFERENCE.md) for technical details

**Risk Managers / Actuaries**
1. Read [README](../README.md) use cases section
2. See [Tutorial](TUTORIAL.md) for interpretation of metrics
3. Try examples with your domain data

### By Task

**Installation**
→ [Installation Guide](INSTALLATION.md)

**First Time Usage**
→ [Quick Start](../README.md#quick-start)
→ [Tutorial Section 3](TUTORIAL.md#getting-started)

**Building Models**
→ [Tutorial Section 4: LaRRegressor](TUTORIAL.md#basic-usage-larregressor)
→ [Tutorial Section 5: HybridMetaLearner](TUTORIAL.md#advanced-usage-hybridmetalearner)

**Understanding Metrics**
→ [README Metrics Guide](../README.md#metrics-guide)
→ [API Reference: Metrics](API_REFERENCE.md#metrics)

**Hyperparameter Tuning**
→ [Tutorial Section 7](TUTORIAL.md#hyperparameter-tuning)

**Production Deployment**
→ [Tutorial Section 8](TUTORIAL.md#production-deployment)

**Testing Your Installation**
→ [Testing Guide](TESTING_GUIDE.md#quick-tests)

**API Lookup**
→ [API Reference](API_REFERENCE.md)

## Documentation Conventions

### Code Examples

All code examples are tested and runnable. They follow this format:

```python
# Import statements
from tailrisk import LaRRegressor

# Usage example
model = LaRRegressor(alpha=2.0)
```

### Parameter Descriptions

Parameter documentation follows this format:

- **parameter_name** : `type`, default=value
  - Description of what the parameter does
  - Recommended values or ranges
  - Usage notes

### Notation

- `ClassName` - Classes and types
- `function_name()` - Functions and methods
- `parameter_name` - Parameters and attributes
- **Important** - Critical information
- ✓ - Success indicators
- ⚠ - Warnings

## Additional Resources

### Main Project Files

- **[README.md](../README.md)** - Project overview and quick start
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** - Contribution guidelines
- **[CHANGELOG.md](../CHANGELOG.md)** - Version history
- **[LICENSE](../LICENSE)** - MIT License

### Examples

- **[quick_start.py](../examples/quick_start.py)** - Basic usage example
- **[examples/](../examples/)** - Additional examples and notebooks

### Source Code

- **[tailrisk/models/](../tailrisk/models/)** - Model implementations
- **[tailrisk/metrics/](../tailrisk/metrics/)** - Metric functions
- **[tailrisk/utils/](../tailrisk/utils/)** - Utility functions

## Getting Help

### Documentation Issues

If you find errors or unclear sections in the documentation:

1. Check if the issue is already reported: [GitHub Issues](https://github.com/yourusername/tailrisk-lib/issues)
2. Create a new documentation issue with:
   - Which document has the problem
   - What is unclear or incorrect
   - Suggestions for improvement

### Usage Questions

For questions about using TailRisk:

1. Check the [Tutorial](TUTORIAL.md) for step-by-step guides
2. Review [API Reference](API_REFERENCE.md) for parameter details
3. Look at [examples](../examples/) for working code
4. Search [GitHub Issues](https://github.com/yourusername/tailrisk-lib/issues)
5. Create a new issue with:
   - What you're trying to accomplish
   - What you've tried
   - Error messages (if any)

### Bug Reports

For bugs in the code:

1. Verify it's reproducible with a minimal example
2. Check [GitHub Issues](https://github.com/yourusername/tailrisk-lib/issues)
3. Create a bug report with:
   - TailRisk version
   - Python version
   - Minimal code to reproduce
   - Expected vs actual behavior

## Contributing to Documentation

We welcome documentation improvements! See [CONTRIBUTING.md](../CONTRIBUTING.md) for:

- How to suggest documentation changes
- Documentation style guide
- How to add examples
- Pull request process

## License

All documentation is part of the TailRisk project and licensed under the MIT License.
See [LICENSE](../LICENSE) for details.

---


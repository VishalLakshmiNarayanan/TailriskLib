# TailRisk Installation Guide

Complete guide for installing and setting up the TailRisk package.

## Table of Contents
- [Requirements](#requirements)
- [Installation Methods](#installation-methods)
- [Verifying Installation](#verifying-installation)
- [Optional Dependencies](#optional-dependencies)
- [Troubleshooting](#troubleshooting)

---

## Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Recommended 4GB+ RAM for large datasets

### Core Dependencies

TailRisk requires the following Python packages:

- `numpy >= 1.21.0` - Numerical computations
- `pandas >= 1.3.0` - Data manipulation
- `scikit-learn >= 1.0.0` - Machine learning framework
- `scipy >= 1.7.0` - Scientific computing
- `matplotlib >= 3.4.0` - Visualization

These are automatically installed when you install TailRisk.

---

## Installation Methods

### Method 1: From PyPI (Recommended)

Install with pip:

```bash
pip install tailrisk
```

To upgrade to the latest version:

```bash
pip install --upgrade tailrisk
```

### Method 2: From Source (For Development)

#### Option A: Direct Installation

```bash
# Clone the repository
git clone https://github.com/VishalLakshmiNarayanan/TailriskLib.git
cd tailrisk-lib

# Install the package
pip install -e .
```

The `-e` flag installs in "editable" mode, allowing you to modify the source code if needed.

#### Option B: Without Editable Mode

```bash
# Clone the repository
git clone https://github.com/VishalLakshmiNarayanan/TailriskLib.git
cd tailrisk-lib

# Install normally
pip install .
```

#### Option C: Download ZIP

1. Download the repository as ZIP from GitHub
2. Extract to a directory
3. Open terminal/command prompt in that directory
4. Run:
   ```bash
   pip install .
   ```

---

## Verifying Installation

### Quick Verification

After installation, verify TailRisk is working:

```bash
python -c "import tailrisk; print(tailrisk.__version__)"
```

Expected output:
```
0.1.3
```

### Full Verification

Create a test script `test_install.py`:

```python
"""Test TailRisk installation"""
import numpy as np
from tailrisk import LaRRegressor, HybridMetaLearner, tail_validation_summary

# Test basic import
print("✓ TailRisk imported successfully")

# Test LaRRegressor
X = np.random.randn(100, 5)
y = np.random.exponential(scale=100, size=100)

model = LaRRegressor(alpha=2.0)
model.fit(X[:80], y[:80])
predictions = model.predict(X[80:])

print("✓ LaRRegressor works")

# Test metrics
metrics = tail_validation_summary(y[80:], predictions)
print("✓ Metrics calculated")

print("\nInstallation verified successfully!")
print(f"TailRisk version: {tailrisk.__version__}")
```

Run it:
```bash
python test_install.py
```

Expected output:
```
✓ TailRisk imported successfully
✓ LaRRegressor works
✓ Metrics calculated

Installation verified successfully!
TailRisk version: 0.1.3
```

---

## Optional Dependencies

### Development Tools

For development and testing:

```bash
pip install tailrisk[dev]
```

Includes:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reports
- `black` - Code formatting
- `flake8` - Linting
- `mypy` - Type checking

### Documentation

For building documentation:

```bash
pip install tailrisk[docs]
```

Includes:
- `sphinx` - Documentation generator
- `sphinx-rtd-theme` - ReadTheDocs theme

### Examples

For running example notebooks:

```bash
pip install tailrisk[examples]
```

Includes:
- `jupyter` - Jupyter notebooks
- `seaborn` - Enhanced visualizations

### All Optional Dependencies

Install everything:

```bash
pip install tailrisk[dev,docs,examples]
```

or

```bash
pip install -e ".[dev,docs,examples]"
```

---

## Virtual Environment Setup (Recommended)

Using a virtual environment prevents dependency conflicts:

### Using venv (Built-in)

```bash
# Create virtual environment
python -m venv tailrisk_env

# Activate (Windows)
tailrisk_env\Scripts\activate

# Activate (macOS/Linux)
source tailrisk_env/bin/activate

# Install TailRisk
pip install tailrisk

# Later, to deactivate
deactivate
```

### Using conda

```bash
# Create environment
conda create -n tailrisk python=3.10

# Activate
conda activate tailrisk

# Install
pip install tailrisk

# Later, to deactivate
conda deactivate
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tailrisk'"

**Solution:**
- Verify installation: `pip list | grep tailrisk` (Unix) or `pip list | findstr tailrisk` (Windows)
- Reinstall: `pip install --force-reinstall tailrisk`
- Check Python path: `python -m site`

### Issue: NumPy/SciPy compilation errors

**Solution (Windows):**
Install pre-compiled wheels:
```bash
pip install --only-binary :all: numpy scipy scikit-learn
```

**Solution (macOS):**
Install Xcode command line tools:
```bash
xcode-select --install
```

**Solution (Linux):**
Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev

# CentOS/RHEL
sudo yum install python3-devel
```

### Issue: "Permission denied" errors

**Solution:**
Install for user only:
```bash
pip install --user tailrisk
```

Or use virtual environment (recommended).

### Issue: Scikit-learn version conflicts

**Solution:**
Update scikit-learn:
```bash
pip install --upgrade scikit-learn
```

Verify version:
```bash
python -c "import sklearn; print(sklearn.__version__)"
```

Minimum required: 1.0.0

### Issue: Import errors after installation

**Solution:**
Clear Python cache:
```bash
# Unix/macOS
find . -type d -name __pycache__ -exec rm -rf {} +

# Windows PowerShell
Get-ChildItem -Path . -Recurse -Filter __pycache__ | Remove-Item -Recurse -Force
```

Then reinstall:
```bash
pip uninstall tailrisk
pip install tailrisk
```

---

## Upgrading

### From PyPI

```bash
pip install --upgrade tailrisk
```

### From Source

```bash
cd tailrisk-lib
git pull origin main
pip install --upgrade -e .
```

---

## Uninstallation

```bash
pip uninstall tailrisk
```

Confirm when prompted:
```
Proceed (Y/n)? Y
```

---

## Platform-Specific Notes

### Windows

If you encounter SSL errors:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org tailrisk
```

### macOS Apple Silicon (M1/M2)

For best performance with numerical libraries:
```bash
# Install using conda (recommended)
conda install -c conda-forge numpy scipy scikit-learn
pip install tailrisk --no-deps
pip install pandas matplotlib
```

### Linux

Recommended to install system packages first:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-pip python3-dev

# CentOS/RHEL
sudo yum install python3-pip python3-devel
```

---

## Docker Installation (Advanced)

For containerized environments:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN pip install numpy pandas scikit-learn scipy matplotlib
RUN pip install tailrisk

CMD ["python"]
```

Build and run:
```bash
docker build -t tailrisk-env .
docker run -it tailrisk-env python -c "import tailrisk; print(tailrisk.__version__)"
```

---

## Next Steps

After successful installation:

1. Read the [Quick Start Guide](../README.md#quick-start)
2. Explore [API Reference](API_REFERENCE.md)
3. Try the [examples](../examples/)
4. Read the [Tutorial](TUTORIAL.md)

---

## Getting Help

If you encounter issues not covered here:

1. Check [GitHub Issues](https://github.com/VishalLakshmiNarayanan/TailriskLib/issues)
2. Create a new issue with:
   - Python version (`python --version`)
   - TailRisk version (`python -c "import tailrisk; print(tailrisk.__version__)"`)
   - Error message (full traceback)
   - Operating system

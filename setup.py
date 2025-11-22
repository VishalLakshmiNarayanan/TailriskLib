"""Setup configuration for tailrisk package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = (this_directory / "requirements.txt").read_text(encoding='utf-8').strip().split('\n')

setup(
    name="tailrisk",
    version="0.1.0",
    author="Vishal Lakshmi Narayanan",
    author_email="lvishal1607@gmail.com",
    description="Risk-aware machine learning for tail risk modeling in insurance, finance, and beyond",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VishalLakshmiNarayanan/TailriskLib",
    project_urls={
        "Bug Tracker": "https://github.com/VishalLakshmiNarayanan/TailriskLib/issues",
        "Documentation": "https://github.com/VishalLakshmiNarayanan/TailriskLib#readme",
        "Source Code": "https://github.com/VishalLakshmiNarayanan/TailriskLib",
    },
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "seaborn>=0.12.0",
        ],
    },
    keywords=[
        "machine learning",
        "tail risk",
        "insurance",
        "actuarial",
        "CVaR",
        "quantile regression",
        "extreme events",
        "risk modeling",
        "claims prediction",
    ],
)

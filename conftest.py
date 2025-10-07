"""
pytest configuration and fixtures for MPASdiag test suite.
"""

import warnings
import pytest

def pytest_configure(config):
    """Configure pytest with warning filters."""
    warnings.filterwarnings(
        "ignore",
        message="numpy.ndarray size changed, may indicate binary incompatibility.*",
        category=RuntimeWarning
    )
    
    warnings.filterwarnings(
        "ignore",
        message=".*numpy.ndarray size changed.*",
        category=RuntimeWarning
    )
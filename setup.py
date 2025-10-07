#!/usr/bin/env python3

"""
Setup script for MPAS Analysis Package

This setup script configures the MPAS Analysis package for installation
using pip or conda. It includes all necessary dependencies and entry points
for command-line usage.
"""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

setup(
    name="mpas-analysis",
    version="1.0.0",
    author="Rubaiat Islam",
    author_email="mrislam@ucar.edu",
    description="Python package for MPAS model output analysis and visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mrixlam/MPASdiag",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "xarray>=0.19.0",
        "uxarray>=2024.01.0",
        "matplotlib>=3.5.0",
        "cartopy>=0.20.0",
        "netCDF4>=1.5.0",
        "scipy>=1.7.0",
        "PyYAML>=5.4.0",
        "dask>=2021.6.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "pytest-mock>=3.6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
        "performance": [
            "numba>=0.55.0",
            "bottleneck>=1.3.0",
        ],
        "plotting": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mpas-precip-plot=mpas_analysis.cli:main",
            "mpas-batch-process=mpas_analysis.cli:batch_main",
            "mpas-validate=mpas_analysis.cli:validate_main",
            "mpas-surface-plot=mpas_analysis.cli:surface_plot_main",
            "mpas-wind-plot=mpas_analysis.cli:wind_plot_main",
        ],
    },
    include_package_data=True,
    package_data={
        "mpas_analysis": [
            "data/*.yaml",
            "data/*.json",
            "templates/*.yaml",
        ],
    },
    keywords=[
        "MPAS",
        "atmospheric modeling",
        "unstructured mesh",
        "precipitation analysis",
        "visualization",
        "meteorology",
        "climate modeling",
        "weather prediction",
    ],
    project_urls={
        "Bug Reports": "https://github.com/mrixlam/MPASdiag/issues",
        "Source": "https://github.com/mrixlam/MPASdiag",
        "Documentation": "https://github.com/mrixlam/MPASdiag#readme",
    },
    zip_safe=False,
)
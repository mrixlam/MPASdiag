#!/usr/bin/env python3

"""
MPASdiag Test Suite: Conftest for processing module tests.

This module defines fixtures for loading test data and setting up temporary directories for the MPASdiag processing tests. These fixtures provide reusable components for unit and integration tests, ensuring consistent test environments and data availability. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules
import pytest
import shutil
import tempfile
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Generator

from tests.test_data_helpers import load_mpas_coords_from_processor


@pytest.fixture
def mpas_test_data() -> xr.Dataset:
    """
    This fixture loads a sample MPAS grid dataset for use in processing tests. It reads the dataset from a predefined location in the test data directory and returns it as an xarray.Dataset object. If the dataset file is not found, the test will be skipped.

    Parameters:
        None

    Returns:
        xarray.Dataset: The loaded MPAS grid dataset.
    """
    data_dir = Path(__file__).parent.parent.parent / "data"
    grid_file = data_dir / "grids" / "x1.10242.static.nc"

    if not grid_file.exists():
        pytest.skip(f"MPAS grid file not found: {grid_file}")
        return

    ds = xr.open_dataset(grid_file, decode_times=False)
    return ds


@pytest.fixture
def temp_weights_dir() -> Generator[Path, None, None]:
    """
    This fixture creates a temporary directory for storing weights files generated during processing tests. It yields the path to the temporary directory for use in tests, and ensures that the directory is cleaned up after the tests are completed.

    Parameters:
        None

    Returns:
        pathlib.Path: The path to the temporary directory.
    """
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """
    This fixture creates a temporary directory for storing output files generated during parallel processing tests. It yields the path to the temporary directory for use in tests, and ensures that the directory is cleaned up after the tests are completed.

    Parameters:
        None

    Returns:
        pathlib.Path: The path to the temporary directory.
    """
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def simple_mpas_data() -> dict:
    """
    This fixture generates a simple dataset for testing MPAS processing functions. It creates arrays for longitude, latitude, and a sample data variable based on the u component of velocity. The longitude and latitude are converted to radians, and the data variable is normalized to the range [0, 1].

    Parameters:
        None

    Returns:
        dict: A dictionary containing longitude, latitude, and data arrays.
    """
    n_cells = 100
    lon_deg, lat_deg, u, v = load_mpas_coords_from_processor(n_cells)
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    data = (u - u.min()) / (u.max() - u.min() + 1e-12)

    return {
        'lon': lon,
        'lat': lat,
        'data': data
    }

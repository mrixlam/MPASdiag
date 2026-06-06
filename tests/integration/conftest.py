#!/usr/bin/env python3

"""
MPASdiag Test Suite: Integration Test Fixtures

This module defines pytest fixtures for integration tests of the MPASdiag package. These fixtures provide shared setup and teardown logic, such as loading real MPAS datasets into processors and managing temporary output directories for rendered figures. The fixtures are designed to be reusable across multiple test modules, ensuring consistent test environments and reducing code duplication. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: June 2026
Version: 1.0.0
"""
from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

from tests.test_data_helpers import (  # noqa: E402
    load_mpas_2d_processor,
    load_mpas_3d_processor,
)
from mpasdiag.processing.processors_2d import MPAS2DProcessor  # noqa: E402
from mpasdiag.processing.processors_3d import MPAS3DProcessor  # noqa: E402

try:
    from cartopy.mpl.geoaxes import GeoAxes  # noqa: F401

    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

try:
    import xesmf  # noqa: F401

    ESMPY_AVAILABLE = True
except ImportError:
    ESMPY_AVAILABLE = False


@pytest.fixture(autouse=True)
def close_figures() -> Generator[None, None, None]:
    """
    This fixture automatically closes all open Matplotlib figures after each test function. This prevents memory leaks and ensures that figures from one test do not interfere with subsequent tests. The fixture is applied to all tests in the module by setting `autouse=True`, so individual test functions do not need to explicitly use it.

    Parameters:
        None

    Returns:
        Generator[None, None, None]: A generator that yields control to the test function and then executes cleanup code after the test completes.
    """
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


@pytest.fixture(scope="module")
def real_2d_processor() -> MPAS2DProcessor:
    """
    This fixture loads a real ``MPAS2DProcessor`` from the ``data/u240k/diag`` directory once per test module. The processor is fully loaded with 2D diagnostic fields and ready for plotting. If the required data files are not available locally, the test that uses this fixture will be skipped. This allows integration tests to run in environments where the full dataset may not be present, while still providing access to real data when it is available. 

    Parameters:
        None

    Returns:
        MPAS2DProcessor: Processor with the real diagnostic dataset loaded.
    """
    return load_mpas_2d_processor("u240k/diag", verbose=False)


@pytest.fixture(scope="module")
def real_3d_processor() -> MPAS3DProcessor:
    """
    This fixture loads a real ``MPAS3DProcessor`` from the ``data/u240k/mpasout`` directory once per test module. The processor is fully loaded with 3D MPAS output fields and ready for plotting. If the required data files are not available locally, the test that uses this fixture will be skipped. This allows integration tests to run in environments where the full dataset may not be present, while still providing access to real data when it is available. 

    Parameters:
        None

    Returns:
        MPAS3DProcessor: Processor with the real mpasout dataset loaded.
    """
    return load_mpas_3d_processor("u240k/mpasout", verbose=False)


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """
    This fixture provides a temporary directory for saving output files generated during tests. Each test that uses this fixture will receive a unique directory path, ensuring that output files from different tests do not conflict. The directory is automatically cleaned up after the test session, preventing clutter and ensuring that test artifacts do not persist on the filesystem. 

    Parameters:
        tmp_path (Path): Pytest-provided unique temporary directory.

    Returns:
        Path: Directory into which tests may save plot output files.
    """
    out = tmp_path / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def bounds_from_coords(lon: np.ndarray, 
                       lat: np.ndarray,) -> Tuple[float, float, float, float]:
    """
    This helper function computes the bounding box of a set of longitude and latitude coordinates, ensuring that the bounds are strictly within the valid ranges for geographic coordinates. The function takes into account potential NaN values in the input arrays and guards against degenerate extents that could arise from tiny coordinate subsets. The resulting bounds are returned as a tuple of minimum and maximum longitude and latitude values.

    Parameters:
        lon (np.ndarray): Longitude values in degrees (any convention).
        lat (np.ndarray): Latitude values in degrees.

    Returns:
        Tuple[float, float, float, float]: A tuple containing (lon_min, lon_max, lat_min, lat_max) representing the bounding box of the coordinates.
    """
    lon_min = max(float(np.nanmin(lon)), -179.9)
    lon_max = min(float(np.nanmax(lon)), 179.9)
    lat_min = max(float(np.nanmin(lat)), -89.9)
    lat_max = min(float(np.nanmax(lat)), 89.9)

    # Guard against degenerate extents from tiny coordinate subsets.
    if lon_max <= lon_min:
        lon_min, lon_max = -179.9, 179.9
        
    if lat_max <= lat_min:
        lat_min, lat_max = -89.9, 89.9

    return lon_min, lon_max, lat_min, lat_max

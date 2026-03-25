#!/usr/bin/env python3

"""
MPASdiag Test Suite: Shared Fixtures and Test Utilities

This module defines pytest fixtures and helper functions that are shared across multiple test modules in the MPASdiag test suite. The fixtures provide common setup for test data, mock objects, and configuration, while the helper functions offer reusable assertions and utilities to simplify test code. By centralizing these components in `conftest.py`, we promote consistency and reduce duplication across our tests.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load standard libraries
import shutil
import pytest
import tempfile
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from unittest.mock import Mock
from typing import Dict, Any, Optional, Tuple, Generator


@pytest.fixture(scope="session")
def project_root() -> Path:
    """
    This fixture resolves and returns the repository root directory relative to this `conftest.py` file. It is useful for locating test resources such as the `data/` directory or example files and remains constant for the lifetime of the test session. Use this fixture in other fixtures or tests that require stable path resolution within the repository.

    Parameters:
        None

    Returns:
        Path: A pathlib `Path` pointing to the project root directory.
    """
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """
    Return the canonical test data directory under the project root. Tests that need sample files should use this fixture to find the repository's `data/` directory. The fixture appends the `data` path to the resolved project root and provides a stable `Path` object for test code to use when locating test assets or deciding to skip data-dependent tests.

    Parameters:
        project_root (Path): The project root directory resolved by the `project_root` fixture.

    Returns:
        Path: A pathlib `Path` pointing to the `data/` directory.
    """
    return project_root / "data"


@pytest.fixture(scope="session")
def grid_file(test_data_dir: Path) -> Optional[str]:
    """
    This fixture locates a representative MPAS invariant (grid) file under the project's `data/grids` directory. If the file is present it returns its string path; otherwise the fixture yields `None` so that tests may skip gracefully when sample data is not available in the local environment.

    Parameters:
        test_data_dir (Path): The `data/` directory provided by the `test_data_dir` fixture.

    Returns:
        Optional[str]: String path to the grid file when present; otherwise `None` when the sample data file is missing.
    """
    grid_path = test_data_dir / "grids" / "x1.40962.static.nc"
    if grid_path.exists():
        return str(grid_path)
    return None


@pytest.fixture(scope="session")
def mpas_2d_processor_diag() -> Optional[Any]:
    """
    This session-scoped fixture loads the 2D diagnostic data from u120k/diag once per test session and shares the initialized processor across all tests. If the data or grid file is unavailable, None is returned and tests should check for None before using.

    Parameters:
        None

    Returns:
        Optional[MPAS2DProcessor]: Initialized processor with loaded diag data or None if data is unavailable.
    """
    try:
        from tests.test_data_helpers import _find_and_load_2d_processor
        return _find_and_load_2d_processor("u120k/diag")
    except (FileNotFoundError, ImportError):
        return None
    except pytest.skip.Exception:
        return None


@pytest.fixture(scope="session")
def mpas_3d_processor() -> Optional[Any]:
    """
    This session-scoped fixture loads the 3D data from u120k/mpasout once per test session and shares it across all tests. If the data or grid file is unavailable, None is returned.

    Parameters:
        None

    Returns:
        Optional[MPAS3DProcessor]: Initialized processor with loaded 3D data or None if data is unavailable.
    """
    try:
        from mpasdiag.processing import MPAS3DProcessor
        from tests.test_data_helpers import _grid_file_path
        
        data_dir = Path(__file__).parent.parent / "data" / "u120k" / "mpasout"
        if not data_dir.exists():
            return None
        
        grid_file = _grid_file_path()
        proc = MPAS3DProcessor(grid_file, verbose=False)
        proc.load_3d_data(str(data_dir))
        return proc
    except (FileNotFoundError, ImportError):
        return None
    except pytest.skip.Exception:
        return None


@pytest.fixture(scope="session")
def mpas_coordinates() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    This session-scoped fixture loads coordinates once per test session and shares them across all tests. Coordinates are in degrees with longitude normalized to [-180, 180].

    Parameters:
        None

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: Tuple of (lon, lat) 1D arrays or None if grid file is unavailable.
    """
    try:
        from tests.test_data_helpers import _grid_file_path

        grid_file = _grid_file_path()
        grid_ds = xr.open_dataset(grid_file, decode_times=False)

        lon = grid_ds['lonCell'].values
        lat = grid_ds['latCell'].values
        
        if np.nanmax(np.abs(lon)) <= 2 * np.pi + 1e-6:
            lon = np.degrees(lon)

        if np.nanmax(np.abs(lat)) <= np.pi / 2 + 1e-6:
            lat = np.degrees(lat)
        
        lon = ((lon + 180.0) % 360.0) - 180.0
        
        grid_ds.close()
        return lon, lat
    except (FileNotFoundError, ImportError):
        return None
    except pytest.skip.Exception:
        return None


@pytest.fixture(scope="session")
def mpas_wind_data() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    This session-scoped fixture loads wind data once per test session from the u120k/diag files. If data is unavailable, None is returned.

    Parameters:
        None

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: Tuple of (u, v) 1D arrays or None if data is unavailable.
    """
    try:
        from tests.test_data_helpers import load_wind_uv_from_diag
        return load_wind_uv_from_diag(n=100)
    except (FileNotFoundError, ImportError):
        return None
    except pytest.skip.Exception:
        return None


@pytest.fixture(scope="session")
def mpas_precip_data() -> Optional[np.ndarray]:
    """
    This session-scoped fixture loads precipitation once per test session. If data is unavailable, None is returned.

    Parameters:
        None
        
    Returns:
        Optional[np.ndarray]: 1D array of precipitation values or None if data is unavailable.
    """
    try:
        from tests.test_data_helpers import load_precip_from_diag
        return load_precip_from_diag(n=100)
    except (FileNotFoundError, ImportError):
        return None
    except pytest.skip.Exception:
        return None


@pytest.fixture(scope="session")
def mpas_surface_temp_data() -> Optional[np.ndarray]:
    """
    This session-scoped fixture loads surface temperature once per test session. If data is unavailable, None is returned.

    Parameters:
        None

    Returns:
        Optional[np.ndarray]: 1D array of surface temperature values or None if data is unavailable.
    """
    try:
        from tests.test_data_helpers import load_surface_t2m_from_diag
        return load_surface_t2m_from_diag(n=100)
    except (FileNotFoundError, ImportError):
        return None
    except pytest.skip.Exception:
        return None


@pytest.fixture(scope="session")
def mpas_qv_3d_data() -> Optional[np.ndarray]:
    """
    This session-scoped fixture loads 3D humidity data once per test session. If data is unavailable, None is returned.

    Parameters:
        None

    Returns:
        Optional[np.ndarray]: 1D array of specific humidity values or None if data is unavailable.
    """
    try:
        from tests.test_data_helpers import load_qv_3d_from_mpasout
        return load_qv_3d_from_mpasout(n=100)
    except (FileNotFoundError, ImportError):
        return None
    except pytest.skip.Exception:
        return None


@pytest.fixture(scope="session")
def mpas_data_available() -> bool:
    """
    This fixture returns True if the required MPAS grid and data directories exist, allowing tests to conditionally skip when data is missing.

    Parameters:
        None
        
    Returns:
        bool: True if MPAS data is available, False otherwise.
    """
    try:
        from tests.test_data_helpers import _grid_file_path
        _grid_file_path()
        data_dir = Path(__file__).parent.parent / "data" / "u120k"
        return data_dir.exists() and (data_dir / "diag").exists()
    except (FileNotFoundError, ImportError):
        return False
    except pytest.skip.Exception:
        return False


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    The fixture creates a temporary directory at setup and yields a `Path` pointing to it. After the consuming test completes the directory is removed to avoid leaving temporary artifacts on disk. Use this fixture for tests that need to write small NetCDF files or other ephemeral outputs.

    Parameters:
        None

    Returns:
        Path: A pathlib `Path` pointing to the temporary directory that will be removed after the test completes.
    """
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_mpas_mesh(grid_file) -> xr.Dataset:
    """
    This fixture loads the actual MPAS grid file when available, providing real mesh geometry and topology. If the grid file is not available, it generates a minimal synthetic dataset with the same structure for CI/CD environments without data files.

    Parameters:
        grid_file: Session-scoped fixture providing path to MPAS grid file.

    Returns:
        xr.Dataset: An xarray Dataset containing MPAS mesh variables like `latCell`, `lonCell`, `cellsOnCell`, and `areaCell`.
    """
    if grid_file is not None:
        ds = xr.open_dataset(grid_file, decode_times=False)
        n_cells = min(100, len(ds['nCells']))
        ds_subset = ds.isel(nCells=slice(0, n_cells))
        return ds_subset
    else:
        n_cells = 100
        n_edges = 150
        n_vertices = 50
        max_edges = 7
        
        ds = xr.Dataset({
            'latCell': (['nCells'], np.random.uniform(-90, 90, n_cells)),
            'lonCell': (['nCells'], np.random.uniform(-180, 180, n_cells)),
            'xCell': (['nCells'], np.random.uniform(-1e7, 1e7, n_cells)),
            'yCell': (['nCells'], np.random.uniform(-1e7, 1e7, n_cells)),
            'zCell': (['nCells'], np.random.uniform(-1e7, 1e7, n_cells)),
            'areaCell': (['nCells'], np.random.uniform(1e9, 1e10, n_cells)),
            'nEdgesOnCell': (['nCells'], np.random.randint(3, max_edges + 1, n_cells)),
            'cellsOnCell': (['nCells', 'maxEdges'], np.random.randint(0, n_cells, (n_cells, max_edges))),
            'edgesOnCell': (['nCells', 'maxEdges'], np.random.randint(0, n_edges, (n_cells, max_edges))),
            'verticesOnCell': (['nCells', 'maxEdges'], np.random.randint(0, n_vertices, (n_cells, max_edges))),
        })
        
        ds.attrs['on_a_sphere'] = 'YES'
        ds.attrs['sphere_radius'] = 6371229.0
        
        return ds


@pytest.fixture
def mock_mpas_3d_data(mock_mpas_mesh: xr.Dataset, mpas_3d_processor) -> xr.Dataset:
    """
    This fixture provides real MPAS 3D data from mpasout files when available, which includes theta, pressure, w, and other 3D fields. Since uReconstructZonal and uReconstructMeridional were removed from mpasout files to save space, synthetic wind components are added for tests that require them. For actual wind testing, use mock_mpas_2d_data which has real u10/v10 from diag files.

    Parameters:
        mock_mpas_mesh (xr.Dataset): The mesh dataset from `mock_mpas_mesh` fixture.
        mpas_3d_processor: Session-scoped processor with loaded 3D mpasout data.

    Returns:
        xr.Dataset: An xarray Dataset with `Time` and vertical levels containing 3D fields from mpasout (theta, pressure, w, etc) plus synthetic wind components for test compatibility.
    """
    n_cells = len(mock_mpas_mesh['nCells'])
    
    if mpas_3d_processor is not None and mpas_3d_processor.dataset is not None:
        ds_real = mpas_3d_processor.dataset
        
        n_cells_subset = min(n_cells, len(ds_real['nCells']))
        n_time_subset = min(3, len(ds_real['Time']))
        n_vert = len(ds_real['nVertLevels'])
        
        ds = ds_real.isel(nCells=slice(0, n_cells_subset), Time=slice(0, n_time_subset))
        
        mesh_vars_to_add = {}

        for var in ['latCell', 'lonCell', 'areaCell', 'xCell', 'yCell', 'zCell']:
            if var not in ds and var in mock_mpas_mesh:
                mesh_vars_to_add[var] = mock_mpas_mesh[var].isel(nCells=slice(0, n_cells_subset))
        
        if mesh_vars_to_add:
            ds = ds.assign(mesh_vars_to_add)
        
        if 'uReconstructZonal' not in ds:
            ds['uReconstructZonal'] = (['Time', 'nCells', 'nVertLevels'],
                                       np.random.uniform(-30, 30, (n_time_subset, n_cells_subset, n_vert)))

        if 'uReconstructMeridional' not in ds:
            ds['uReconstructMeridional'] = (['Time', 'nCells', 'nVertLevels'],
                                           np.random.uniform(-30, 30, (n_time_subset, n_cells_subset, n_vert)))
        
        if 'temperature' not in ds and 'theta' in ds:
            ds['temperature'] = ds['theta']
        
        return ds
    else:
        n_vertical = 55
        n_time = 3
        
        ds = mock_mpas_mesh.copy()        
        ds = ds.expand_dims({'Time': n_time})
        
        ds['pressure'] = (['Time', 'nCells', 'nVertLevels'], 
                         np.random.uniform(10000, 101325, (n_time, n_cells, n_vertical)))

        ds['theta'] = (['Time', 'nCells', 'nVertLevels'], 
                      np.random.uniform(250, 400, (n_time, n_cells, n_vertical)))

        ds['temperature'] = (['Time', 'nCells', 'nVertLevels'], 
                            np.random.uniform(200, 320, (n_time, n_cells, n_vertical)))

        ds['uReconstructZonal'] = (['Time', 'nCells', 'nVertLevels'], 
                                   np.random.uniform(-30, 30, (n_time, n_cells, n_vertical)))

        ds['uReconstructMeridional'] = (['Time', 'nCells', 'nVertLevels'], 
                                        np.random.uniform(-30, 30, (n_time, n_cells, n_vertical)))

        ds['w'] = (['Time', 'nCells', 'nVertLevels'], 
                   np.random.uniform(-5, 5, (n_time, n_cells, n_vertical)))

        ds['rho'] = (['Time', 'nCells', 'nVertLevels'], 
                     np.random.uniform(0.1, 1.5, (n_time, n_cells, n_vertical)))
        
        ds['xtime'] = (['Time'], [
            '2024-01-01_00:00:00',
            '2024-01-01_06:00:00',
            '2024-01-01_12:00:00'
        ])
        
        return ds


@pytest.fixture
def mock_mpas_2d_data(mock_mpas_mesh: xr.Dataset, mpas_2d_processor_diag) -> xr.Dataset:
    """
    This fixture provides real MPAS 2D diagnostic data from diag files when available, which includes surface variables like t2m, rainnc, u10, v10. Falls back to synthetic data only when real data is unavailable. Uses diag files (not mpasout) for optimal 2D diagnostic coverage.

    Parameters:
        mock_mpas_mesh (xr.Dataset): The mesh dataset from `mock_mpas_mesh` fixture.
        mpas_2d_processor_diag: Session-scoped processor with loaded 2D diag data.

    Returns:
        xr.Dataset: An xarray Dataset containing 2D diagnostic variables with a `Time` dimension from diag files.
    """
    n_cells = len(mock_mpas_mesh['nCells'])
    
    if mpas_2d_processor_diag is not None and mpas_2d_processor_diag.dataset is not None:
        ds_real = mpas_2d_processor_diag.dataset
        
        n_cells_subset = min(n_cells, len(ds_real['nCells']))
        n_time_subset = min(3, len(ds_real['Time']))
        
        ds = ds_real.isel(nCells=slice(0, n_cells_subset), Time=slice(0, n_time_subset))
        
        mesh_vars_to_add = {}

        for var in ['latCell', 'lonCell', 'areaCell', 'xCell', 'yCell', 'zCell']:
            if var not in ds and var in mock_mpas_mesh:
                mesh_vars_to_add[var] = mock_mpas_mesh[var].isel(nCells=slice(0, n_cells_subset))
        
        if mesh_vars_to_add:
            ds = ds.assign(mesh_vars_to_add)
        
        return ds
    else:
        n_time = 3
        
        ds = mock_mpas_mesh.copy()
        ds = ds.expand_dims({'Time': n_time})
        
        ds['rainnc'] = (['Time', 'nCells'], np.random.uniform(0, 50, (n_time, n_cells)))
        ds['t2m'] = (['Time', 'nCells'], np.random.uniform(250, 310, (n_time, n_cells)))
        ds['u10'] = (['Time', 'nCells'], np.random.uniform(-20, 20, (n_time, n_cells)))
        ds['v10'] = (['Time', 'nCells'], np.random.uniform(-20, 20, (n_time, n_cells)))
        ds['surface_pressure'] = (['Time', 'nCells'], np.random.uniform(95000, 105000, (n_time, n_cells)))
        
        ds['xtime'] = (['Time'], [
            '2024-01-01_00:00:00',
            '2024-01-01_06:00:00',
            '2024-01-01_12:00:00'
        ])
        
        return ds


@pytest.fixture
def mock_remapped_data() -> xr.Dataset:
    """
    This fixture returns a compact Dataset indexed by `time`, `lat`, and `lon` that contains synthetic `temperature`, `precipitation`, and horizontal wind components. It is useful for unit tests that validate plotting, color-mapping, and post-remap expectations without invoking the full remapping machinery against an unstructured mesh.

    Parameters:
        None

    Returns:
        xr.Dataset: An xarray Dataset indexed by `time`, `lat`, and `lon` with synthetic fields suitable for plotting and remap tests.
    """
    n_lat = 50
    n_lon = 100
    n_time = 3
    
    lat = np.linspace(-90, 90, n_lat)
    lon = np.linspace(-180, 180, n_lon)
    
    ds = xr.Dataset({
        'temperature': (['time', 'lat', 'lon'], 
                       np.random.uniform(200, 320, (n_time, n_lat, n_lon))),
        'precipitation': (['time', 'lat', 'lon'], 
                         np.random.uniform(0, 50, (n_time, n_lat, n_lon))),
        'u_wind': (['time', 'lat', 'lon'], 
                   np.random.uniform(-30, 30, (n_time, n_lat, n_lon))),
        'v_wind': (['time', 'lat', 'lon'], 
                   np.random.uniform(-30, 30, (n_time, n_lat, n_lon))),
    }, coords={
        'lat': lat,
        'lon': lon,
        'time': pd.date_range('2024-01-01', periods=n_time, freq='6h')
    })
    
    return ds


@pytest.fixture
def mock_file_paths(temp_dir: Path) -> Dict[str, Path]:
    """
    The fixture returns a dictionary mapping logical names (for example 'grid' or 'mpasout_1') to filesystem `Path` objects located inside the provided temporary directory. Tests may write small NetCDF files to the returned locations or assert on file-discovery behavior using these paths.

    Parameters:
        temp_dir (Path): Temporary directory provided by the `temp_dir` fixture.

    Returns:
        Dict[str, Path]: A mapping of logical file names to pathlib `Path` objects within the temporary directory (e.g., 'grid', 'mpasout_1').
    """
    paths = {
        'grid': temp_dir / 'x1.40962.static.nc',
        'mpasout_1': temp_dir / 'mpasout.2024-01-01_00.00.00.nc',
        'mpasout_2': temp_dir / 'mpasout.2024-01-01_06.00.00.nc',
        'diag_1': temp_dir / 'diag.2024-01-01_00.00.00.nc',
        'diag_2': temp_dir / 'diag.2024-01-01_06.00.00.nc',
        'output': temp_dir / 'output',
    }
    
    paths['output'].mkdir(parents=True, exist_ok=True)
    
    return paths


@pytest.fixture
def mock_weight_file(temp_dir: Path) -> str:
    """
    The function writes a small Dataset containing `row`, `col`, and `S` arrays to a NetCDF file inside the provided temporary directory. The produced file mimics the structure expected by remapping utilities that consume weight files during testing without relying on external weights.

    Parameters:
        temp_dir (Path): Temporary directory provided by `temp_dir` fixture.

    Returns:
        str: Filesystem path to the written weight NetCDF file.
    """
    weight_file = temp_dir / 'weights.nc'
    
    n_s = 100
    n_b = 50
    
    ds = xr.Dataset({
        'row': (['n_s'], np.random.randint(0, n_b, n_s)),
        'col': (['n_s'], np.random.randint(0, 1000, n_s)),
        'S': (['n_s'], np.random.uniform(0, 1, n_s)),
    })
    
    ds.to_netcdf(weight_file)
    return str(weight_file)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """
    Tests may use this fixture to supply default values for input and output paths, processing controls, and visualization options in a predictable manner. The fixture provides a nested mapping with reasonable defaults so individual tests can override only the fields they need.

    Parameters:
        None

    Returns:
        Dict[str, Any]: A nested dictionary containing `input`, `output`, `processing`, and `visualization` configuration sections.
    """
    return {
        'input': {
            'invariant_file': 'data/grids/x1.40962.static.nc',
            'data_dir': 'data/u120k/mpasout',
            'diag_dir': 'data/u120k/diag',
        },
        'output': {
            'output_dir': 'output/test',
            'format': 'png',
            'dpi': 300,
        },
        'processing': {
            'n_workers': 4,
            'chunk_size': 1000,
            'cache_enabled': True,
        },
        'visualization': {
            'colormap': 'viridis',
            'figure_size': [12, 8],
            'font_size': 12,
        }
    }


@pytest.fixture
def mock_cli_args() -> Any:
    """
    The returned object exposes attributes commonly used by CLI entry points and can be passed to functions that expect an `args` namespace. This fixture is intentionally minimal — tests may modify attributes as needed to simulate different command-line invocations.

    Parameters:
        None

    Returns:
        Any: An object with attributes matching expected CLI options (e.g. `invariant_file`, `data_dir`, `variable`, `time_index`).
    """
    class Args:
        def __init__(self):
            self.invariant_file = 'data/grids/x1.40962.static.nc'
            self.data_dir = 'data/u120k/mpasout'
            self.output_dir = 'output/test'
            self.variable = 'temperature'
            self.time_index = 0
            self.level = 500
            self.workers = 4
            self.verbose = True
            self.format = 'png'
            self.dpi = 300
    
    return Args()


@pytest.fixture
def mock_processor() -> Mock:
    """
    The returned `Mock` object includes fields such as `invariant_file`, a `mesh_data` placeholder, and a `verbose` flag so tests can exercise code paths that require a processor-like API without instantiating real processor classes or reading files.

    Parameters:
        None

    Returns:
        Mock: A `unittest.mock.Mock` object configured with minimal mesh attributes and flags used by tests.
    """
    processor = Mock()
    processor.invariant_file = 'data/grids/x1.40962.static.nc'
    processor.mesh_data = Mock()
    processor.mesh_data.latCell = Mock()
    processor.mesh_data.lonCell = Mock()
    processor.verbose = True
    return processor


@pytest.fixture
def mock_cache() -> Mock:
    """
    The mock implements `get`, `set`, `clear`, and `stats` methods to allow tests to verify interactions with caching logic while avoiding dependencies on an actual cache implementation. The default behavior returns no cached values which is suitable for many unit-test scenarios.

    Parameters:
        None

    Returns:
        Mock: A mock cache object with commonly used methods stubbed.
    """
    cache = Mock()
    cache.get = Mock(return_value=None)
    cache.set = Mock()
    cache.clear = Mock()
    cache.stats = Mock(return_value={'hits': 0, 'misses': 0, 'size': 0})
    return cache


@pytest.fixture(params=['temperature', 'pressure', 'wind', 'precipitation'])
def variable_names(request: Any) -> str:
    """
    Each test invocation receives one of the configured variable name strings which allows parameterized coverage across plotting and processing routines. Use this fixture to exercise multiple code paths without duplicating test logic.

    Parameters:
        request (Any): Pytest request object supplying the current parameter.

    Returns:
        str: The selected variable name for this test case.
    """
    return request.param


@pytest.fixture(params=[0, 1, 2])
def time_indices(request: Any) -> int:
    """
    This fixture yields the current parameter value from the pytest parametrization set and is useful for tests that need to index into a `Time` dimension at different positions.

    Parameters:
        request (Any): Pytest request object supplying the current parameter.

    Returns:
        int: The time index value for this parametrized test instance.
    """
    return request.param


@pytest.fixture(params=[100, 200, 500, 850, 1000])
def pressure_levels(request: Any) -> int:
    """
    The fixture yields a single pressure level per invocation drawn from a realistic set of values which is helpful for validating vertical interpolation and selection logic in a variety of scenarios.

    Parameters:
        request (Any): Pytest request object supplying the current parameter.

    Returns:
        int: The selected pressure level in hPa for this test case.
    """
    return request.param


def assert_valid_dataset(ds: xr.Dataset) -> None:
    """
    The helper checks that the provided object is an `xr.Dataset`, contains at least one data variable, and that declared data variables are present in the dataset. It raises an `AssertionError` when expectations are not met which simplifies failure diagnostics in unit tests.

    Parameters:
        ds (xr.Dataset): The dataset to validate.

    Returns:
        None: The function raises an AssertionError on invalid datasets; it returns `None` when the dataset passes validation.
    """
    assert isinstance(ds, xr.Dataset)
    assert len(ds.data_vars) > 0
    assert all(var in ds for var in ds.data_vars)


def assert_valid_array(arr: np.ndarray, expected_shape: Optional[Tuple[int, ...]] = None) -> None:
    """
    The function asserts the input is an `np.ndarray`, contains at least one element and that all values are finite. If `expected_shape` is provided the array's shape is compared to that tuple, allowing tests to validate structural expectations in addition to element-level checks.

    Parameters:
        arr (np.ndarray): The array to validate.
        expected_shape (Optional[Tuple[int, ...]]): If provided, the function verifies the array's shape matches this tuple.

    Returns:
        None: Raises AssertionError for invalid arrays; returns `None` on success.
    """
    assert isinstance(arr, np.ndarray)
    assert arr.size > 0
    assert np.all(np.isfinite(arr))
    if expected_shape is not None:
        assert arr.shape == expected_shape


def create_mock_netcdf_file(filepath: Path, dataset: xr.Dataset) -> None:
    """
    Tests use this helper to persist small datasets to disk for I/O and integration-style checks. The helper forwards the call to `xarray.Dataset.to_netcdf` and does not perform additional validation.

    Parameters:
        filepath (Path): Filesystem path where the NetCDF file will be written.
        dataset (xr.Dataset): The xarray Dataset to serialize.

    Returns:
        None: The function writes the dataset to disk and returns `None` on success.
    """
    dataset.to_netcdf(filepath)


def pytest_configure(config: Any) -> None:
    """
    The function adds descriptive marker definitions such as `slow`, `integration`, `requires_data`, and `parallel` to the pytest configuration so the markers are discoverable and documented for CI and local test runs. It mutates the provided `Config` object and does not return a value.

    Parameters:
        config (Any): The pytest `Config` object supplied by the framework.

    Returns:
        None: This function mutates the pytest configuration and returns `None`.
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_data: marks tests that require real data files"
    )
    config.addinivalue_line(
        "markers", "parallel: marks tests for parallel processing"
    )



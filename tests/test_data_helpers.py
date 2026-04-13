#!/usr/bin/env python3

"""
MPASdiag Test Suite: Data Helpers

This module contains utility functions for loading and preparing MPAS grid and diagnostic data for use in the test suite. It includes functions to derive coordinate arrays, extract variables from processors, and create mock renderers for testing plotting code without actual rendering. The helpers are designed to be robust against missing data files by skipping tests when necessary.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load standard libraries
import os
import pytest
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Tuple, Callable, Union

# Load MPASdiag processing classes
from mpasdiag.processing.processors_2d import MPAS2DProcessor
from mpasdiag.processing.processors_3d import MPAS3DProcessor


EXPECTED_PUBLIC_METHODS = {
    'MPASBaseProcessor': [
        'extract_spatial_coordinates',
        'filter_by_spatial_extent',
        'get_available_variables',
        'get_time_info',
        'normalize_longitude',
        'parse_file_datetimes',
        'validate_files',
        'validate_geographic_extent',
        'validate_time_parameters',
    ],
    'MPAS2DProcessor': [
        'add_spatial_coordinates',
        'extract_2d_coordinates_for_variable',
        'extract_spatial_coordinates',
        'filter_by_spatial_extent',
        'find_diagnostic_files',
        'get_2d_variable_data',
        'get_available_variables',
        'load_2d_data',
        'normalize_longitude',
    ],
    'MPAS3DProcessor': [
        'add_spatial_coordinates',
        'extract_2d_coordinates_for_variable',
        'extract_2d_from_3d',
        'extract_spatial_coordinates',
        'find_mpasout_files',
        'get_3d_variable_data',
        'get_available_3d_variables',
        'get_available_variables',
        'get_vertical_levels',
        'load_3d_data',
        'normalize_longitude',
    ],
    'MPASBaseVisualizer': [
        'add_regional_features',
        'add_timestamp_and_branding',
        'calculate_adaptive_marker_size',
        'close_plot',
        'convert_to_numpy',
        'create_histogram',
        'create_time_series_plot',
        'create_wind_plot',
        'format_latitude',
        'format_longitude',
        'get_variable_specific_settings',
        'save_plot',
        'setup_map_projection',
    ],
    'MPASSurfacePlotter': [
        'add_surface_overlay',
        'close_plot',
        'create_batch_surface_maps',
        'create_simple_scatter_plot',
        'create_surface_map',
        'get_surface_colormap_and_levels',
        'save_plot',
        'setup_map_projection',
    ],
    'MPASWindPlotter': [
        'add_wind_overlay',
        'calculate_optimal_subsample',
        'close_plot',
        'compute_wind_speed_and_direction',
        'create_batch_wind_plots',
        'create_wind_plot',
        'save_plot',
        'setup_map_projection',
    ],
    'MPASPrecipitationPlotter': [
        'add_precipitation_overlay',
        'apply_style',
        'close_plot',
        'create_batch_precipitation_maps',
        'create_precip_colormap',
        'create_precipitation_comparison_plot',
        'create_precipitation_map',
        'save_plot',
        'setup_map_projection',
    ],
    'MPASSkewTPlotter': [
        'close_plot',
        'create_skewt_diagram',
        'save_plot',
    ],
    'MPASCrossSectionPlotter': [
        'close_plot',
        'create_batch_cross_section_plots',
        'create_vertical_cross_section',
        'save_plot',
        'setup_map_projection',
    ],
    'PrecipitationDiagnostics': [
        'compute_precipitation_difference',
        'get_accumulation_hours',
    ],
    'WindDiagnostics': [
        'analyze_wind_components',
        'compute_wind_direction',
        'compute_wind_shear',
        'compute_wind_speed',
        'get_2d_wind_components',
        'get_3d_variable_at_level',
        'get_3d_wind_components',
    ],
    'SoundingDiagnostics': [
        'compute_dewpoint_from_mixing_ratio',
        'compute_thermodynamic_indices',
        'extract_sounding_profile',
        'potential_to_actual_temperature',
    ],
    'MPASParallelManager': [
        'set_error_policy',
        'parallel_map', 
        'get_statistics', 
        'barrier',
        'finalize',
    ],
    'MPASDataCache': [
        'load_coordinates_from_dataset',
        'get_coordinates', 
        'load_variable_data',
        'get_variable_data',
        'get_cache_info',
    ],
    'ParallelPrecipitationProcessor': [
        'create_batch_precipitation_maps_parallel',
    ], 
    'ParallelWindProcessor': [
        'create_batch_wind_plots_parallel',
    ],
    'ParallelSurfaceProcessor': [
        'create_batch_surface_maps_parallel',
    ],
    'ParallelCrossSectionProcessor': [
        'create_batch_cross_section_plots_parallel',
    ],
    'MPASTaskDistributor': [
        'distribute_tasks',
    ],
    'MPASResultCollector': [
        'gather_results',
        'compute_statistics',
    ],
}


def assert_expected_public_methods(instance: object, 
                                   class_key: str) -> None:
    """
    Assert that *instance* exposes every public method listed for *class_key*
    in the ``EXPECTED_PUBLIC_METHODS`` registry.

    Parameters:
        instance: An instance of the class under test.
        class_key (str): Key into ``EXPECTED_PUBLIC_METHODS``.

    Raises:
        AssertionError: If any expected method is missing from the instance.
        KeyError: If *class_key* is not registered.
    """
    expected = EXPECTED_PUBLIC_METHODS[class_key]
    public = [m for m in dir(instance) if not m.startswith('_')]
    missing = [m for m in expected if m not in public]
    assert not missing, (
        f"{class_key} instance is missing public methods: {missing}"
    )


def load_mpas_mesh(nx: int = 10, 
                   ny: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function loads the MPAS grid file to extract longitude and latitude values, converts them to degrees if necessary, and normalizes longitudes to the [-180, 180] range. It then creates a regular 2D meshgrid of coordinates and synthesizes deterministic u and v wind fields based on the latitudes and longitudes. This is useful for testing plotting functions that require coordinate and wind data without relying on actual diag files.

    Parameters:
        nx (int): Number of grid points along the longitudinal (x) axis.
        ny (int): Number of grid points along the latitudinal (y) axis.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A four-element tuple containing `(lon2d, lat2d, u2d, v2d)` where `lon2d` and `lat2d` are 2D coordinate arrays of shape `(ny, nx)` and `u2d`, `v2d` are the corresponding deterministic wind component fields.
    """
    grid_file = _grid_file_path()
    ds = xr.open_dataset(grid_file, decode_times=False)

    lon_all = ds['lonCell'].values
    lat_all = ds['latCell'].values

    lon_vals = lon_all
    lat_vals = lat_all

    if np.nanmax(np.abs(lon_vals)) <= 2 * np.pi + 1e-6:
        lon_vals = np.degrees(lon_vals)

    if np.nanmax(np.abs(lat_vals)) <= np.pi / 2 + 1e-6:
        lat_vals = np.degrees(lat_vals)

    lon_vals = ((lon_vals + 180.0) % 360.0) - 180.0

    lon_min, lon_max = float(np.nanmin(lon_vals)), float(np.nanmax(lon_vals))
    lat_min, lat_max = float(np.nanmin(lat_vals)), float(np.nanmax(lat_vals))

    lon_1d = np.linspace(lon_min, lon_max, nx)
    lat_1d = np.linspace(lat_min, lat_max, ny)

    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)

    u2d = 5.0 * np.sin(np.radians(lat2d))
    v2d = 5.0 * np.cos(np.radians(lon2d))

    return lon2d, lat2d, u2d, v2d


def fake_render_factory(calls: dict) -> Callable[..., None]:
    """
    The factory returns a function suitable for injecting into plotting code during tests; each invocation increments the provided `calls` dictionary under the `'render'` key. This enables assertions on how many times a renderer would have been invoked without producing actual graphics.

    Parameters:
        calls (dict): Mutable mapping used to accumulate call counts (will be updated in-place by the returned callable).

    Returns:
        Callable[..., None]: A callable accepting arbitrary args/kwargs that increments `calls['render']` each time it is invoked.
    """
    def _fake_render(ax_arg, *args, **kwargs):
        calls['render'] = calls.get('render', 0) + 1
    return _fake_render


def _grid_file_path() -> str:
    """
    This helper locates the canonical MPAS grid file used by tests in the repository `data/grids` directory. If the file cannot be found the calling test is skipped to keep the test-suite robust when sample data is absent.

    Parameters:
        None

    Returns:
        str: Absolute path to the MPAS grid file as a string (suitable for passing to xarray or processor constructors).
    """
    data_dir = Path(__file__).parent.parent / "data"
    grid_file = data_dir / "grids" / "x1.10242.static.nc"

    if not grid_file.exists():
        pytest.skip(f"MPAS grid file not found: {grid_file}")
        return

    if _is_lfs_pointer(grid_file):
        pytest.skip("MPAS grid file is a Git LFS pointer (LFS data not pulled)")
        return

    return str(grid_file)


def _is_lfs_pointer(filepath: Path) -> bool:
    """
    This helper function checks if the given file is a Git LFS pointer file by reading its header and looking for the characteristic LFS signature. It returns True if the file is an LFS pointer, which indicates that the actual data is not present locally and may need to be fetched from the remote repository. This is important for tests that rely on sample data files, as it allows them to skip gracefully when the data is not available in the local environment. 

    Parameters:
        filepath (Path): The path to the file to check.

    Returns:
        bool: True if the file is an LFS pointer, False otherwise. 
    """
    try:
        with open(filepath, 'rb') as f:
            header = f.read(50)
        return header.startswith(b'version https://git-lfs.github.com/')
    except (OSError, IOError):
        return False


def _find_and_load_2d_processor(data_subdir: str) -> MPAS2DProcessor:
    """
    The function checks for the existence of `data/<data_subdir>` and initializes an `MPAS2DProcessor` with the canonical grid file. If the requested data directory is missing the test is skipped to avoid hard failures in CI.

    Parameters:
        data_subdir (str): Relative subdirectory under `data/` containing the 2D files.

    Returns:
        MPAS2DProcessor: Initialized processor with loaded 2D data ready for variable extraction.
    """
    data_dir = Path(__file__).parent.parent / "data" / data_subdir
 
    if not data_dir.exists():
        pytest.skip(f"Data directory not found: {data_dir}")
        return

    grid_file = _grid_file_path()

    if grid_file is None:
        pytest.skip("MPAS grid file not available")
        return
    
    proc = MPAS2DProcessor(grid_file, verbose=False)
    proc.load_2d_data(str(data_dir), use_pure_xarray=True)

    return proc


def _to_degrees_wrapped(lon: np.ndarray, 
                        lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    This helper function takes longitude and latitude arrays, checks if they are in radians (based on their maximum absolute values), converts them to degrees if necessary, and wraps longitudes to the [-180, 180] range. This ensures that coordinate data is in a consistent format for testing, regardless of how it is stored in the original MPAS grid file. 

    Parameters:
        lon (np.ndarray): Array of longitudes in radians.
        lat (np.ndarray): Array of latitudes in radians.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of longitudes and latitudes in degrees, with longitudes wrapped to [-180, 180].
    """
    if np.nanmax(np.abs(lon)) <= 2 * np.pi + 1e-6:
        lon = np.degrees(lon)
    if np.nanmax(np.abs(lat)) <= np.pi / 2 + 1e-6:
        lat = np.degrees(lat)
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lon, lat


def load_mpas_coords_from_processor(n: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This helper constructs a minimal dataset with `nCells` and enriches it via `MPAS2DProcessor.add_spatial_coordinates` to retrieve real grid coordinates when available. Wind components are sourced from diag files through the processor; if diag values are missing the function synthesizes deterministic u/v fields based on the coordinates.

    Parameters:
        n (int): Number of horizontal samples (cells) to return.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four 1D numpy arrays `(lon, lat, u, v)` each of length `n` containing flattened coordinates and wind components.
    """
    grid_file = _grid_file_path()

    if grid_file is None:
        pytest.skip("MPAS grid file not available")
        return

    proc = MPAS2DProcessor(grid_file, verbose=False)

    ds = xr.Dataset()
    ds = ds.assign_coords({'nCells': np.arange(n)})
    ds = proc.add_spatial_coordinates(ds)

    if 'lonCell' in ds and 'latCell' in ds:
        lon = ds['lonCell'].values.ravel()[:n]
        lat = ds['latCell'].values.ravel()[:n]
        lon, lat = _to_degrees_wrapped(lon, lat)
    else:
        grid_ds = xr.open_dataset(grid_file, decode_times=False)

        lon_all = grid_ds['lonCell'].values
        lat_all = grid_ds['latCell'].values
        lon_all, lat_all = _to_degrees_wrapped(lon_all, lat_all)

        if lon_all.size >= n:
            lon = lon_all[:n]
            lat = lat_all[:n]
        else:
            reps = int(np.ceil(n / lon_all.size))
            lon = np.tile(lon_all, reps)[:n]
            lat = np.tile(lat_all, reps)[:n]

    try:
        u, v = load_wind_uv_from_diag(n=n)
    except Exception:
        u = 5.0 * np.sin(np.radians(lat))
        v = 5.0 * np.cos(np.radians(lon))

    return lon, lat, u, v


def load_precip_from_diag(n: int = 100, 
                          var_candidates: Tuple[str, ...] = ("rainc", "rain_nc", "rain")) -> np.ndarray:
    """
    The function initializes a processor for the `u240k/diag` directory and searches for a matching precipitation-like variable from `var_candidates`. If no suitable variable exists the test is skipped. The returned array is a flattened 1D numpy array of length `n` containing the first time slice.

    Parameters:
        n (int): Number of horizontal samples (cells) to return.
        var_candidates (tuple of str): Variable names to probe for precipitation-like data.

    Returns:
        np.ndarray: 1D numpy array of length `n` with precipitation accumulation values.
    """
    proc = _find_and_load_2d_processor(os.path.join("u240k", "diag"))
    available = set(proc.dataset.data_vars.keys()) if proc.dataset is not None else set()

    var_name = None

    for cand in var_candidates:
        if cand in available:
            var_name = cand
            break

    if var_name is None:
        pytest.skip(f"No precipitation variable found in diag files. Available: {sorted(available)[:10]}")
        return

    da = proc.get_2d_variable_data(var_name, time_index=0)
    arr = da.values.ravel()[:n]

    return arr


def load_surface_t2m_from_diag(n: int = 100, 
                               var_candidates: Tuple[str, ...] = ("t2m", "air_temperature", "t_surf")) -> np.ndarray:
    """
    The helper probes the `u240k/diag` data directory for one of the supplied `var_candidates` and returns the first time slice as a flattened 1D array. If no candidate variable is present in the diag files the calling test is skipped to avoid false failures in environments lacking sample data.

    Parameters:
        n (int): Number of horizontal samples (cells) to return.
        var_candidates (tuple of str): Candidate variable names to search for.

    Returns:
        np.ndarray: 1D numpy array of length `n` with surface temperature values.
    """
    proc = _find_and_load_2d_processor(os.path.join("u240k", "diag"))
    available = set(proc.dataset.data_vars.keys()) if proc.dataset is not None else set()

    var_name = None

    for cand in var_candidates:
        if cand in available:
            var_name = cand
            break

    if var_name is None:
        pytest.skip(f"No surface temperature variable found in diag files. Available: {sorted(available)[:10]}")
        return

    da = proc.get_2d_variable_data(var_name, time_index=0)
    return da.values.ravel()[:n]


def load_wind_uv_from_diag(n: int = 100, 
                           u_candidates: Tuple[str, ...] = ("u", "u10", "x_wind"), 
                           v_candidates: Tuple[str, ...] = ("v", "v10", "y_wind")) -> Tuple[np.ndarray, np.ndarray]:
    """
    The helper searches the `u240k/diag` dataset for suitable u and v variable names (from `u_candidates` and `v_candidates`). It returns flattened u and v arrays for the first time index. If either component is missing the test is skipped to avoid failures when diag data is not present.

    Parameters:
        n (int): Number of horizontal samples (cells) to return.
        u_candidates (tuple of str): Candidate names for the zonal component.
        v_candidates (tuple of str): Candidate names for the meridional component.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two 1D numpy arrays `(u, v)` each of length `n`.
    """
    proc = _find_and_load_2d_processor(os.path.join("u240k", "diag"))
    available = set(proc.dataset.data_vars.keys()) if proc.dataset is not None else set()

    u_name = next((c for c in u_candidates if c in available), None)
    v_name = next((c for c in v_candidates if c in available), None)

    if u_name is None or v_name is None:
        pytest.skip(f"No suitable wind u/v variables found in diag files. Available: {sorted(available)[:10]}")
        return  
    
    da_u = proc.get_2d_variable_data(u_name, time_index=0)
    da_v = proc.get_2d_variable_data(v_name, time_index=0)

    return da_u.values.ravel()[:n], da_v.values.ravel()[:n]


def load_qv_3d_from_mpasout(n: int = 100, 
                            n_levels: int = 10, 
                            var_candidates: Tuple[str, ...] = ("qv", "specific_humidity", "q", "theta", "vorticity")) -> np.ndarray:
    """
    The helper looks for mpasout files under `data/u240k/mpasout`, initializes an `MPAS3DProcessor`, and probes for a matching 3D variable. Since specific humidity (qv) may not be available in all datasets, this function falls back to other 3D variables like theta (potential temperature) which have similar data characteristics for testing styling and level generation. If no suitable variable exists the test is skipped.

    Parameters:
        n (int): Number of horizontal samples (cells) to return.
        n_levels (int): Number of vertical levels expected (not directly used here). 
        var_candidates (tuple of str): Candidate variable names to probe for. Defaults include qv, specific_humidity, q, theta, and vorticity.

    Returns:
        np.ndarray: 1D numpy array containing the surface-level variable values, flattened and trimmed to length `n`.
    """
    data_dir = Path(__file__).parent.parent / "data" / "u240k" / "mpasout"

    if not data_dir.exists():
        pytest.skip(f"MPASOUT data directory not found: {data_dir}")
        return

    grid_file = _grid_file_path()

    if grid_file is None:
        pytest.skip("MPAS grid file not available")
        return

    proc3 = MPAS3DProcessor(grid_file, verbose=False)
    proc3.load_3d_data(str(data_dir))

    available = set(proc3.dataset.data_vars.keys()) if proc3.dataset is not None else set()
    var_name = next((c for c in var_candidates if c in available), None)

    if var_name is None:
        pytest.skip(f"No 3D variable found in mpasout files. Available: {sorted(available)[:10]}")
        return

    da = proc3.get_3d_variable_data(var_name, level='surface', time_index=0)
    return da.values.reshape((-1,))[:n]


def get_mpas_data_paths() -> dict:
    """
    This helper centralizes path resolution for all MPAS test data, making it easy for tests to locate grid files, diagnostic data, and mpasout files without duplicating path logic. Tests should use this function to get consistent paths across the test suite.

    Parameters:
        None

    Returns:
        dict: Dictionary with keys 'grid_file', 'diag_dir', 'mpasout_dir',
            'data_root'. Values are Path objects or None if unavailable.
    """
    data_root = Path(__file__).parent.parent / "data"
    
    paths = {
        'data_root': data_root if data_root.exists() else None,
        'grid_dir': data_root / "grids" if (data_root / "grids").exists() else None,
        'grid_file': data_root / "grids" / "x1.10242.static.nc" if (data_root / "grids" / "x1.10242.static.nc").exists() else None,
        'u240k_dir': data_root / "u240k" if (data_root / "u240k").exists() else None,
        'diag_dir': data_root / "u240k" / "diag" if (data_root / "u240k" / "diag").exists() else None,
        'mpasout_dir': data_root / "u240k" / "mpasout" if (data_root / "u240k" / "mpasout").exists() else None,
    }
    
    return paths


def check_mpas_data_available() -> bool:
    """
    Tests can call this function to determine whether to use real MPAS data or fall back to mock data. Returns True only if both grid file and at least one data directory are present.

    Parameters:
        None

    Returns:
        bool: True if MPAS grid and data files are available, False otherwise.
    """
    paths = get_mpas_data_paths()

    return (
        paths['grid_file'] is not None and
        (paths['diag_dir'] is not None or paths['mpasout_dir'] is not None)
    )


def load_mpas_2d_processor(data_subdir: str = "u240k/diag", 
                           verbose: bool = False) -> MPAS2DProcessor:
    """
    This top-level function provides a centralized way to create and load a 2D processor with actual MPAS data. It handles path resolution and error cases consistently. Tests should prefer this function over creating processors directly.

    Parameters:
        data_subdir (str): Subdirectory under data/ containing 2D files. Default is "u240k/diag".
        verbose (bool): Whether to enable verbose processor output.

    Returns:
        MPAS2DProcessor: Initialized processor with loaded 2D data.

    Raises:
        FileNotFoundError: If grid file or data directory is missing.
        pytest.skip.Exception: If called during pytest and data is unavailable.
    """
    data_dir = Path(__file__).parent.parent / "data" / data_subdir

    if not data_dir.exists():
        if pytest:
            pytest.skip(f"Data directory not found: {data_dir}")
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    grid_file = _grid_file_path()

    if grid_file is None:
        pytest.skip("MPAS grid file not available")
        return

    proc = MPAS2DProcessor(grid_file, verbose=verbose)
    proc.load_2d_data(str(data_dir))

    return proc


def load_mpas_3d_processor(data_subdir: str = "u240k/mpasout", 
                           verbose: bool = False) -> MPAS3DProcessor:
    """
    This top-level function provides a centralized way to create and load a 3D processor with actual MPAS data. It handles path resolution and error cases consistently. Tests should prefer this function over creating processors directly.

    Parameters:
        data_subdir (str): Subdirectory under data/ containing 3D files. Default is "u240k/mpasout".
        verbose (bool): Whether to enable verbose processor output.

    Returns:
        MPAS3DProcessor: Initialized processor with loaded 3D data.

    Raises:
        FileNotFoundError: If grid file or data directory is missing.
        pytest.skip.Exception: If called during pytest and data is unavailable.
    """
    data_dir = Path(__file__).parent.parent / "data" / data_subdir

    if not data_dir.exists():
        if pytest:
            pytest.skip(f"Data directory not found: {data_dir}")
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    grid_file = _grid_file_path()

    if grid_file is None:
        pytest.skip("MPAS grid file not available")
        return

    proc = MPAS3DProcessor(grid_file, verbose=verbose)
    proc.load_3d_data(str(data_dir), use_pure_xarray=True)

    return proc


def get_real_mpas_coordinates(n: Union[int, None] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function loads longitude and latitude arrays from the actual MPAS grid file, converts them to degrees, and normalizes longitude to [-180, 180]. Tests should use this function to get consistent coordinate arrays.

    Parameters:
        n (int or None): If specified, limits the number of coordinates returned to `n`. If None, returns all available coordinates.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (lon, lat) arrays in degrees.

    Raises:
        FileNotFoundError: If grid file is missing.
        pytest.skip.Exception: If called during pytest and grid is unavailable.
    """
    grid_file = _grid_file_path()
    
    if grid_file is None:
        pytest.skip("MPAS grid file not available")
        return

    grid_ds = xr.open_dataset(grid_file, decode_times=False)
    
    lon = grid_ds['lonCell'].values
    lat = grid_ds['latCell'].values
    
    if np.nanmax(np.abs(lon)) <= 2 * np.pi + 1e-6:
        lon = np.degrees(lon)

    if np.nanmax(np.abs(lat)) <= np.pi / 2 + 1e-6:
        lat = np.degrees(lat)
    
    lon = ((lon + 180.0) % 360.0) - 180.0
    
    grid_ds.close()
    
    if n is not None and n < len(lon):
        return lon[:n], lat[:n]
    
    return lon, lat


def get_real_mpas_variable(processor: Union[MPAS2DProcessor, MPAS3DProcessor],
                           variable_name: str,
                           time_index: int = 0,
                           level: Union[str, float, None] = None) -> np.ndarray:
    """
    This helper provides a unified interface for extracting variables from both 2D and 3D processors. It handles the different APIs and returns a flattened numpy array suitable for testing.

    Parameters:
        processor: MPAS2DProcessor or MPAS3DProcessor instance with loaded data.
        variable_name (str): Name of the variable to extract.
        time_index (int): Time index to extract. Default is 0.
        level: Vertical level for 3D data. Can be 'surface', pressure value, or None for 2D data.

    Returns:
        np.ndarray: Flattened 1D array of variable values.

    Raises:
        ValueError: If variable is not found in the processor dataset.
    """
    if hasattr(processor, 'get_3d_variable_data') and level is not None:
        da = processor.get_3d_variable_data(variable_name, level=level, time_index=time_index)
    else:
        da = processor.get_2d_variable_data(variable_name, time_index=time_index)
    
    return da.values.ravel()

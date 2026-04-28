#!/usr/bin/env python3

"""
MPASdiag Test Suite: Sounding Diagnostics Coverage

This module contains unit tests for the SoundingDiagnostics class in the mpasdiag.diagnostics.sounding module, specifically targeting code paths that were not covered by existing tests. The tests are designed to verify the behavior of the SoundingDiagnostics class when handling edge cases such as missing datasets, absence of MetPy, exceptions during thermodynamic index computations, and issues with loading grid coordinates. By covering these scenarios, we aim to ensure that the SoundingDiagnostics class is robust and can handle a variety of situations gracefully without crashing, while providing appropriate feedback to users when verbose mode is enabled. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import numpy as np
import pytest
import xarray as xr
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from mpasdiag.diagnostics.sounding import SoundingDiagnostics
from mpasdiag.processing.processors_3d import MPAS3DProcessor

try:
    from metpy.units import units
    assert units is not None  
    HAS_METPY_TEST = True
except ImportError:
    HAS_METPY_TEST = False


N_CELLS = 5
N_VERT = 20


def _make_standard_profile() -> tuple:
    """
    This helper function creates a standard sounding profile with pressure decreasing from 1000 hPa to 200 hPa, temperature decreasing from 25°C to -60°C, dew point 5°C below the temperature, and random u and v wind components between -20 and 20 m/s. The height is linearly spaced from 0 to 14,000 meters. This profile is designed to be used in tests of the compute_thermodynamic_indices method to verify that it can handle a typical sounding profile and return expected results. 

    Parameters:
        None

    Returns:
        tuple: A tuple containing numpy arrays for pressure (p), temperature (t), dewpoint (td), u-wind (u), v-wind (v), and height (h). 
    """
    p = np.linspace(1000.0, 200.0, N_VERT)
    t = np.linspace(25.0, -60.0, N_VERT)
    td = t - 5.0
    u = np.random.default_rng(0).uniform(-20.0, 20.0, N_VERT)
    v = np.random.default_rng(1).uniform(-20.0, 20.0, N_VERT)
    h = np.linspace(0.0, 14_000.0, N_VERT)
    return p, t, td, u, v, h


def _make_mock_proc_no_grid(ds: xr.Dataset) -> Mock:
    """
    This helper function creates a mock MPAS3DProcessor with the specified dataset and a dummy grid_file path that does not point to an actual file. This allows tests to verify the behavior of the SoundingDiagnostics class when it attempts to load grid coordinates from a non-existent file, which should trigger error handling logic in the _load_grid_coordinates method. 

    Parameters:
        ds (xr.Dataset): The dataset to be assigned to the mock processor.

    Returns:
        Mock: A mock MPAS3DProcessor with the specified dataset and a dummy grid_file path.
    """
    proc = Mock(spec=MPAS3DProcessor)
    proc.dataset = ds
    proc.grid_file = "/nonexistent/grid.nc"
    return proc


def _make_grid_file(tmp_path: Path,
                    include_lon: bool = True,
                    include_lat: bool = True,
                    prefix: str = "grid") -> str:
    """
    This helper function creates a grid file in the specified temporary directory with optional longitude and latitude coordinates. The longitude and latitude values are generated as linearly spaced arrays in radians, and the resulting dataset is saved to a NetCDF file. This allows tests to verify the behavior of the SoundingDiagnostics class when loading grid coordinates from a valid file, as well as when certain coordinates are missing, which should trigger error handling logic in the _load_grid_coordinates method. 

    Parameters:
        tmp_path (Path): A temporary directory path for the grid file.
        include_lon (bool): Whether to include longitude coordinates.
        include_lat (bool): Whether to include latitude coordinates.
        prefix (str): The prefix for the grid file name.

    Returns:
        str: The path to the created grid file.
    """
    lon = np.radians(np.linspace(-110.0, -90.0, N_CELLS))
    lat = np.radians(np.linspace(25.0, 45.0, N_CELLS))

    grid_vars: dict = {}

    if include_lon:
        grid_vars["lonCell"] = (["nCells"], lon)

    if include_lat:
        grid_vars["latCell"] = (["nCells"], lat)

    path = tmp_path / f"{prefix}.nc"
    xr.Dataset(grid_vars).to_netcdf(str(path))
    return str(path)


class _FakeCtxMgr:
    """ Context manager that yields a dataset and does not suppress exceptions. Used to mock xr.open_dataset in tests of _load_grid_coordinates. """

    def __init__(self: '_FakeCtxMgr', 
                 ds: xr.Dataset) -> None:
        """
        This initializer method takes an xarray Dataset and stores it as an instance variable. This dataset will be yielded when the context manager is entered. The purpose of this class is to provide a way to mock the behavior of xr.open_dataset in tests of the _load_grid_coordinates method, allowing tests to simulate both successful loading of a dataset and the raising of exceptions without needing to create actual files on disk. By yielding the provided dataset, this context manager allows tests to verify that the _load_grid_coordinates method can successfully extract longitude and latitude coordinates from a valid dataset, while still allowing any exceptions that occur during the loading process to propagate normally for testing error handling logic.

        Parameters:
            ds (xr.Dataset): The xarray Dataset to be yielded by the context manager.

        Returns:
            None
        """
        self._ds = ds

    def __enter__(self: '_FakeCtxMgr') -> xr.Dataset:
        """
        This method is called when the context manager is entered. It simply returns the dataset that was provided during initialization. This allows tests that use this context manager to receive a valid xarray Dataset when they attempt to open a grid file, enabling them to verify that the _load_grid_coordinates method can successfully extract longitude and latitude coordinates from the dataset. Since this context manager does not suppress exceptions, any exceptions raised during the loading process will still propagate, allowing tests to verify error handling logic as well. 

        Parameters:
            None

        Returns:
            xr.Dataset: The dataset that was provided during initialization. 
        """
        return self._ds

    def __exit__(self: '_FakeCtxMgr', *args: object) -> bool:
        """
        This method is called when the context manager is exited. It does not perform any cleanup or exception handling, and simply returns False to indicate that any exceptions that occurred should not be suppressed. This allows tests that use this context manager to verify that exceptions raised during the loading of the grid dataset are properly propagated and handled by the _load_grid_coordinates method, while still allowing the method to successfully extract coordinates from a valid dataset when no exceptions occur. 

        Parameters:
            *args (object): The exception type, value, and traceback, if an exception occurred.

        Returns:
            bool: False, indicating that exceptions should not be suppressed.
        """
        return False

def _make_proc_with_grid(ds: xr.Dataset, 
                         grid_path: str) -> Mock:
    """
    This helper function creates a mock MPAS3DProcessor with the specified dataset and a valid grid_file path. This allows tests to verify the behavior of the SoundingDiagnostics class when it successfully loads grid coordinates from a valid file, enabling them to confirm that the _load_grid_coordinates method can extract longitude and latitude coordinates correctly when provided with a proper dataset and grid file. By providing a valid grid_file path, this function ensures that tests can focus on verifying the coordinate extraction logic without being confounded by file not found errors or other issues related to missing files. 

    Parameters:
        ds (xr.Dataset): The dataset to be assigned to the mock processor.
        grid_path (str): The path to the grid file containing longitude and latitude coordinates.

    Returns:
        Mock: A mock MPAS3DProcessor with the specified dataset and a valid grid_file path.
    """
    proc = Mock(spec=MPAS3DProcessor)
    proc.dataset = ds
    proc.grid_file = grid_path
    return proc


class TestExtractSoundingDatasetNone:
    """ Test that extract_sounding_profile raises a ValueError when the processor's dataset is None. """

    def test_none_dataset_raises(self: 'TestExtractSoundingDatasetNone') -> None:
        """
        This test verifies that the extract_sounding_profile method of the SoundingDiagnostics class raises a ValueError when the processor's dataset is None. The test creates a mock MPAS3DProcessor with its dataset attribute set to None, initializes the SoundingDiagnostics class with verbose set to False, and calls extract_sounding_profile with the mock processor and sample longitude and latitude values. The test checks that a ValueError is raised with a message indicating that the dataset is not loaded, confirming that the method correctly handles the case where the required dataset is missing. 

        Parameters:
            None

        Returns:
            None
        """
        proc = Mock(spec=MPAS3DProcessor)
        proc.dataset = None
        diag = SoundingDiagnostics(verbose=False)
        with pytest.raises(ValueError, match="Dataset not loaded"):
            diag.extract_sounding_profile(proc, -100.0, 35.0)


class TestComputeThermodynamicIndicesNoMetPy:
    """ Test that compute_thermodynamic_indices returns a result with fallback LCL when MetPy is not available, and that it prints a message when verbose is True. """

    def test_no_metpy_returns_result_with_fallback_lcl(self: 'TestComputeThermodynamicIndicesNoMetPy') -> None:
        """
        This test verifies that the compute_thermodynamic_indices method of the SoundingDiagnostics class returns a result dictionary with fallback LCL values when MetPy is not available. The test creates a standard profile using the _make_standard_profile helper function, initializes the SoundingDiagnostics class with verbose set to False, and patches the HAS_METPY variable to False to simulate the absence of MetPy. The test then calls the compute_thermodynamic_indices method and checks that the result is a dictionary containing the "cape" key, which indicates that the method has returned a result even without MetPy, and that it has likely used fallback logic to compute LCL values. 

        Parameters:
            None

        Returns:
            None
        """
        p, t, td, *_ = _make_standard_profile()
        diag = SoundingDiagnostics(verbose=False)

        with patch("mpasdiag.diagnostics.sounding.HAS_METPY", False):
            result = diag.compute_thermodynamic_indices(p, t, td)

        assert isinstance(result, dict)
        assert "cape" in result

    def test_no_metpy_verbose_prints_message(self: 'TestComputeThermodynamicIndicesNoMetPy') -> None:
        """
        This test verifies that the compute_thermodynamic_indices method of the SoundingDiagnostics class prints a message indicating that MetPy is not available when the HAS_METPY variable is patched to False and verbose is set to True. The test creates a standard profile using the _make_standard_profile helper function, initializes the SoundingDiagnostics class with verbose set to True, and patches the HAS_METPY variable to False. The test captures the standard output during the call to compute_thermodynamic_indices and checks that the captured output contains a message indicating that MetPy is not available, confirming that users are informed about the lack of MetPy when verbose mode is enabled. 

        Parameters:
            None

        Returns:
            None
        """
        p, t, td, *_ = _make_standard_profile()
        diag = SoundingDiagnostics(verbose=True)
        captured = StringIO()

        with patch("mpasdiag.diagnostics.sounding.HAS_METPY", False):
            with patch("sys.stdout", new=captured):
                diag.compute_thermodynamic_indices(p, t, td)

        assert "MetPy not available" in captured.getvalue()


@pytest.mark.skipif(not HAS_METPY_TEST, reason="MetPy not installed")
class TestComputeThermodynamicIndicesMetPyFails:
    """ Test that compute_thermodynamic_indices returns a result with "cape" set to None when the parcel_profile function from MetPy raises an exception, and that it prints a message when verbose is True. """

    def test_parcel_profile_failure_returns_empty_result(self: 'TestComputeThermodynamicIndicesMetPyFails') -> None:
        """
        This test verifies that the compute_thermodynamic_indices method of the SoundingDiagnostics class returns a result dictionary with "cape" set to None when the parcel_profile function from MetPy raises a RuntimeError. The test creates a standard profile using the _make_standard_profile helper function, initializes the SoundingDiagnostics class with verbose set to False, and patches the parcel_profile function to raise a RuntimeError. The test then calls the compute_thermodynamic_indices method and checks that the "cape" key in the result is None, confirming that the method handles exceptions from the parcel_profile computation gracefully by returning a result with "cape" set to None instead of crashing.

        Parameters:
            None

        Returns:
            None
        """
        p, t, td, *_ = _make_standard_profile()
        diag = SoundingDiagnostics(verbose=False)

        with patch("mpasdiag.diagnostics.sounding.mpcalc.parcel_profile",
                   side_effect=RuntimeError("parcel_profile failed")):
            result = diag.compute_thermodynamic_indices(p, t, td)

        assert result["cape"] is None

    def test_parcel_profile_failure_verbose_prints(self: 'TestComputeThermodynamicIndicesMetPyFails') -> None:
        """
        This test verifies that the compute_thermodynamic_indices method of the SoundingDiagnostics class prints a message indicating that the MetPy base profile computation failed when the parcel_profile function raises a RuntimeError and verbose is set to True. The test creates a standard profile using the _make_standard_profile helper function, initializes the SoundingDiagnostics class with verbose set to True, and patches the parcel_profile function to raise a RuntimeError. The test captures the standard output during the call to compute_thermodynamic_indices and checks that the captured output contains a message indicating that the MetPy base profile computation failed, confirming that users are informed about the failure of the parcel_profile computation when verbose mode is enabled. 

        Parameters:
            None

        Returns:
            None
        """
        p, t, td, *_ = _make_standard_profile()
        diag = SoundingDiagnostics(verbose=True)
        captured = StringIO()

        with patch("mpasdiag.diagnostics.sounding.mpcalc.parcel_profile",
                   side_effect=RuntimeError("bad profile")):
            with patch("sys.stdout", new=captured):
                diag.compute_thermodynamic_indices(p, t, td)

        assert "MetPy base profile computation failed" in captured.getvalue()


@pytest.mark.skipif(not HAS_METPY_TEST, reason="MetPy not installed")
class TestSbcapeFallback:
    """ Test that compute_thermodynamic_indices returns a result dictionary when the surface_based_cape_cin function raises an AttributeError, simulating the case where the function is not found (e.g., due to an older version of MetPy). """

    def test_sbcape_fallback_triggers_cape_cin(self: 'TestSbcapeFallback') -> None:
        """
        This test verifies that the compute_thermodynamic_indices method of the SoundingDiagnostics class returns a result dictionary when the surface-based CAPE/CIN computation raises an AttributeError, which simulates the case where the function is not found (e.g., due to an older version of MetPy). The test creates a standard profile using the _make_standard_profile helper function, initializes the SoundingDiagnostics class with verbose set to False, and patches the surface_based_cape_cin function to raise an AttributeError. The test then calls the compute_thermodynamic_indices method and checks that the result is a dictionary, which indicates that the method has handled the exception gracefully and returned a result without crashing, even when the expected CAPE/CIN function is not available. 

        Parameters:
            None

        Returns:
            None
        """
        p, t, td, *_ = _make_standard_profile()
        diag = SoundingDiagnostics(verbose=False)

        with patch("mpasdiag.diagnostics.sounding.mpcalc.surface_based_cape_cin",
                   side_effect=AttributeError("not found")):
            result = diag.compute_thermodynamic_indices(p, t, td)

        assert isinstance(result, dict)


class TestLoadGridCoordinatesProbeFailure:
    """ Test that _load_grid_coordinates correctly handles a RuntimeError during the initial probe attempt. """

    def test_probe_exception_falls_through_to_main_open(self: 'TestLoadGridCoordinatesProbeFailure', 
                                                        tmp_path: Path) -> None:
        """
        This test verifies that the _load_grid_coordinates method of the SoundingDiagnostics class correctly handles a RuntimeError raised during the initial probe attempt to open the grid file. The test creates a valid grid file using the _make_grid_file helper function, initializes a mock MPAS3DProcessor with this grid file, and initializes the SoundingDiagnostics class with verbose set to False. The test then patches the xr.open_dataset function to first raise a RuntimeError (simulating a probe failure) and then return a context manager yielding a valid dataset on the second call. The test calls _load_grid_coordinates and checks that it successfully returns longitude and latitude arrays of the expected length, confirming that the method correctly falls back to a second attempt to open the dataset after the initial probe failure. 

        Parameters:
            tmp_path (Path): Temporary path for creating test files.

        Returns:
            None
        """
        grid_path = _make_grid_file(tmp_path, prefix="good_grid")

        ds = xr.Dataset({
            "pressure": (["Time", "nCells", "nVertLevels"], np.ones((1, N_CELLS, N_VERT))),
            "theta": (["Time", "nCells", "nVertLevels"], np.full((1, N_CELLS, N_VERT), 300.0)),
            "qv": (["Time", "nCells", "nVertLevels"], np.full((1, N_CELLS, N_VERT), 0.01)),
        })

        proc = _make_proc_with_grid(ds, grid_path)
        diag = SoundingDiagnostics(verbose=False)

        real_grid = xr.open_dataset(grid_path, decode_times=False)
        ctx = _FakeCtxMgr(real_grid)

        with patch("mpasdiag.diagnostics.sounding.xr.open_dataset",
                   side_effect=[RuntimeError("probe fail"), ctx]):
            lon, lat = diag._load_grid_coordinates(proc)

        assert len(lon) == N_CELLS
        assert len(lat) == N_CELLS


class TestLoadGridCoordinatesNoLon:
    """ Test that _load_grid_coordinates raises a ValueError when the longitude coordinate is missing from the grid file. """

    def test_missing_lon_raises(self: 'TestLoadGridCoordinatesNoLon', 
                                tmp_path: Path) -> None:
        """
        This test verifies that the _load_grid_coordinates method of the SoundingDiagnostics class raises a ValueError when the longitude coordinate is missing from the grid file. The test creates a grid file without longitude coordinates using the _make_grid_file helper function, initializes a mock MPAS3DProcessor with this grid file, and initializes the SoundingDiagnostics class with verbose set to False. The test then calls _load_grid_coordinates and checks that a ValueError is raised with a message indicating that the longitude coordinate is missing, confirming that the method correctly handles the case where required longitude coordinates are not present in the grid dataset. 

        Parameters:
            tmp_path (Path): Temporary path for creating test files.

        Returns:
            None
        """
        grid_path = _make_grid_file(tmp_path, include_lon=False, prefix="no_lon")
        proc = Mock(spec=MPAS3DProcessor)
        proc.grid_file = grid_path
        diag = SoundingDiagnostics(verbose=False)
        with pytest.raises(ValueError, match="longitude coordinate"):
            diag._load_grid_coordinates(proc)


class TestLoadGridCoordinatesNoLat:
    """ Test that _load_grid_coordinates raises a ValueError when the latitude coordinate is missing from the grid file. """

    def test_missing_lat_raises(self: 'TestLoadGridCoordinatesNoLat', 
                                tmp_path: Path) -> None:
        """
        This test verifies that the _load_grid_coordinates method of the SoundingDiagnostics class raises a ValueError when the latitude coordinate is missing from the grid file. The test creates a grid file without latitude coordinates using the _make_grid_file helper function, initializes a mock MPAS3DProcessor with this grid file, and initializes the SoundingDiagnostics class with verbose set to False. The test then calls _load_grid_coordinates and checks that a ValueError is raised with a message indicating that the latitude coordinate is missing, confirming that the method correctly handles the case where required latitude coordinates are not present in the grid dataset. 

        Parameters:
            tmp_path (Path): Temporary path for creating test files.

        Returns:
            None
        """
        grid_path = _make_grid_file(tmp_path, include_lat=False, prefix="no_lat")
        proc = Mock(spec=MPAS3DProcessor)
        proc.grid_file = grid_path
        diag = SoundingDiagnostics(verbose=False)
        with pytest.raises(ValueError, match="latitude coordinate"):
            diag._load_grid_coordinates(proc)


@pytest.mark.skipif(not HAS_METPY_TEST, reason="MetPy not installed")
class TestSafePairToUnit:
    """ Test that _safe_pair correctly converts MetPy-unitized quantities to the specified unit. """

    def test_to_unit_converts_before_extracting_magnitude(self: 'TestSafePairToUnit') -> None:
        """
        This test verifies that the _safe_pair method of the SoundingDiagnostics class correctly converts MetPy-unitized quantities to the specified unit before extracting their magnitudes. The test defines a function that returns a pair of MetPy-unitized quantities representing wind speeds in meters per second, and then calls _safe_pair with this function and a target unit of "knots". The test checks that the returned values are not None and that their magnitudes are consistent with the expected conversion from meters per second to knots, confirming that _safe_pair correctly handles unit conversion before extracting magnitudes. 

        Parameters:
            None

        Returns:
            None
        """
        def _pair_func() -> tuple:
            """
            This function returns a pair of MetPy-unitized quantities representing wind speeds in meters per second. The first quantity is 100 m/s and the second quantity is -50 m/s. This function is designed to be used in tests of the _safe_pair method to verify that it can correctly convert these quantities to a specified unit (e.g., knots) before extracting their magnitudes. By providing known values with units, this function allows tests to confirm that _safe_pair performs unit conversion as expected and returns values that are consistent with the correct conversion from meters per second to knots. 

            Parameters:
                None

            Returns:
                tuple: A tuple containing two MetPy-unitized quantities, one for 100 m/s and one for -50 m/s. 
            """
            _mpu = units
            return 100.0 * _mpu("m/s"), -50.0 * _mpu("m/s")

        first, second = SoundingDiagnostics._safe_pair(_pair_func, to_unit="knots")
        assert first is not None
        assert second is not None
        assert abs(first) > 100

    def test_to_unit_exception_returns_none_pair(self: 'TestSafePairToUnit') -> None:
        """
        This test verifies that the _safe_pair method of the SoundingDiagnostics class returns a pair of None values when the function it calls raises an exception during unit conversion. The test defines a function that raises a ValueError, and then calls _safe_pair with this function and a target unit of "knots". The test checks that both returned values are None, confirming that _safe_pair correctly handles exceptions during unit conversion by returning None for both values instead of propagating the exception. 

        Parameters:
            None

        Returns:
            None
        """
        def _bad_func() -> tuple:
            """
            This function raises a ValueError when called. It is designed to be used in tests of the _safe_pair method to verify that when an exception occurs during the execution of the provided function (e.g., during unit conversion), _safe_pair correctly returns a pair of None values instead of propagating the exception. By simulating a failure in the function, this test can confirm that _safe_pair's error handling logic is functioning as intended, ensuring that it gracefully handles exceptions and provides a consistent return value of None for both elements of the pair when an error occurs.

            Parameters:
                None

            Returns:
                tuple: This function does not return a value; it raises a ValueError instead. 
            """
            raise ValueError("bad")

        first, second = SoundingDiagnostics._safe_pair(_bad_func, to_unit="knots")
        assert first is None and second is None


class TestComputeFallbackLcl:
    """ Test that _compute_fallback_lcl populates the result dictionary with fallback LCL values when called with valid pressure, temperature, and dew point arrays, and that it prints a message when verbose is True. Also test that it handles empty input arrays gracefully without raising an exception. """

    def test_fallback_lcl_populates_result(self: 'TestComputeFallbackLcl') -> None:
        """
        This test verifies that the _compute_fallback_lcl method of the SoundingDiagnostics class populates the result dictionary with "lcl_pressure" and "lcl_temperature" keys when it is called with valid pressure, temperature, and dew point arrays. The test creates sample pressure, temperature, and dew point arrays, initializes an empty result dictionary, and calls _compute_fallback_lcl with these arrays. The test then checks that the result dictionary contains the "lcl_pressure" and "lcl_temperature" keys, confirming that the method has computed fallback LCL values and stored them in the result dictionary as expected. 

        Parameters:
            None

        Returns:
            None
        """
        p = np.linspace(1000.0, 200.0, 10)
        t = np.linspace(25.0, -40.0, 10)
        td = t - 5.0
        result: dict = {}
        diag = SoundingDiagnostics(verbose=False)
        diag._compute_fallback_lcl(p, t, td, result)
        assert "lcl_pressure" in result
        assert "lcl_temperature" in result

    def test_fallback_lcl_verbose_prints_skip_message(self: 'TestComputeFallbackLcl') -> None:
        """
        This test verifies that the _compute_fallback_lcl method of the SoundingDiagnostics class prints a message indicating that MetPy is not available when it is called with valid pressure, temperature, and dew point arrays while verbose is set to True. The test creates sample pressure, temperature, and dew point arrays, initializes an empty result dictionary, and calls _compute_fallback_lcl with these arrays while patching the HAS_METPY variable to False. The test captures the standard output during the call and checks that the captured output contains a message indicating that MetPy is not available, confirming that users are informed about the lack of MetPy when verbose mode is enabled and fallback LCL computation is being used. 

        Parameters:
            None

        Returns:
            None
        """
        p = np.linspace(1000.0, 200.0, 10)
        t = np.linspace(25.0, -40.0, 10)
        td = t - 5.0

        result: dict = {}

        diag = SoundingDiagnostics(verbose=True)
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            diag._compute_fallback_lcl(p, t, td, result)

        assert "MetPy not available" in captured.getvalue()

    def test_fallback_lcl_empty_arrays_handled_gracefully(self: 'TestComputeFallbackLcl') -> None:
        """
        This test verifies that the _compute_fallback_lcl method of the SoundingDiagnostics class handles empty input arrays gracefully without raising an exception. The test initializes empty pressure, temperature, and dew point arrays, as well as an empty result dictionary, and calls _compute_fallback_lcl with these empty arrays. The test checks that no exceptions are raised during the call and that the result dictionary does not contain "lcl_pressure" or that it is set to None, confirming that the method can handle edge cases of empty input data without crashing and does not populate LCL values when there is no data to compute from. 

        Parameters:
            None

        Returns:
            None
        """
        result: dict = {}

        diag = SoundingDiagnostics(verbose=False)
        diag._compute_fallback_lcl(np.array([]), np.array([]), np.array([]), result)

        assert result.get("lcl_pressure") is None or "lcl_pressure" not in result


@pytest.mark.skipif(not HAS_METPY_TEST, reason="MetPy not installed")
class TestComputeWetBulbZeroException:
    """ Test that _compute_wet_bulb_zero returns None when the wet_bulb_temperature function raises a RuntimeError, and that it also returns None when called with unitless arrays that would cause an exception. """

    def test_wet_bulb_exception_returns_none(self: 'TestComputeWetBulbZeroException') -> None:
        """
        This test verifies that the _compute_wet_bulb_zero method of the SoundingDiagnostics class returns None when the wet_bulb_temperature function from MetPy raises a RuntimeError. The test creates sample pressure, temperature, dew point, and height arrays with MetPy units, initializes the SoundingDiagnostics class with verbose set to False, and patches the wet_bulb_temperature function to raise a RuntimeError. The test then calls _compute_wet_bulb_zero with these arrays and checks that the result is None, confirming that the method handles exceptions from the wet bulb temperature computation gracefully by returning None instead of crashing. 

        Parameters:
            None

        Returns:
            None
        """
        _mpu = units
        p_metpy = np.linspace(1000.0, 200.0, 10) * _mpu.hPa
        t_metpy = np.linspace(25.0, -40.0, 10) * _mpu.degC
        td_metpy = (np.linspace(25.0, -40.0, 10) - 5.0) * _mpu.degC
        h = np.linspace(0.0, 12_000.0, 10)

        with patch("mpasdiag.diagnostics.sounding.mpcalc.wet_bulb_temperature",
                   side_effect=RuntimeError("wb fail")):
            result = SoundingDiagnostics._compute_wet_bulb_zero(p_metpy, t_metpy, td_metpy, h)

        assert result is None

    def test_wet_bulb_no_units_returns_none(self: 'TestComputeWetBulbZeroException') -> None:
        """
        This test verifies that the _compute_wet_bulb_zero method of the SoundingDiagnostics class returns None when it is called with plain numpy arrays that do not have MetPy units, which would cause the wet_bulb_temperature function to raise an exception. The test creates sample pressure, temperature, dew point, and height arrays without units, initializes the SoundingDiagnostics class with verbose set to False, and calls _compute_wet_bulb_zero with these unitless arrays. The test checks that the result is None, confirming that the method handles the case of missing units gracefully by returning None instead of crashing due to an exception from the wet bulb temperature computation. 

        Parameters:
            None

        Returns:
            None
        """
        result = SoundingDiagnostics._compute_wet_bulb_zero(
            np.linspace(1000.0, 200.0, 5),
            np.linspace(25.0, -40.0, 5),
            np.linspace(20.0, -45.0, 5),
            np.linspace(0.0, 5000.0, 5),
        )
        assert result is None


@pytest.mark.skipif(not HAS_METPY_TEST, reason="MetPy not installed")
class TestComputeShearIndicesExceptions:
    """ Each test patches a different MetPy function to raise, covering each except clause. """

    def _base_result(self: 'TestComputeShearIndicesExceptions') -> dict:
        """
        This helper method returns a dictionary with keys corresponding to various shear indices and their initial values set to None. This serves as a base result dictionary for tests of the _compute_shear_indices method, allowing tests to verify that exceptions in the computation of shear indices are handled gracefully by checking that the relevant keys remain None when exceptions are raised. By providing a consistent starting point for the result dictionary, this method helps ensure that tests can focus on verifying the exception handling logic for each specific shear index without needing to set up the entire result structure manually in each test. 

        Parameters:
            None

        Returns:
            dict: A dictionary with shear index keys and None values.
        """
        return dict.fromkeys(
            ["bulk_shear_0_1km", "bulk_shear_0_6km", "srh_0_1km", "srh_0_3km",
             "stp", "scp", "sweat_index",
             "sbcape", "lcl_pressure", "srh_0_1km", "mucape", "srh_0_3km"], None
        )

    def _base_result_with_deps(self: 'TestComputeShearIndicesExceptions') -> dict:
        """
        This helper method returns a dictionary with keys corresponding to various shear indices, where the indices that depend on successful MetPy computations (e.g., bulk shear and storm-relative helicity) are set to None, while other indices that do not depend on those computations are set to valid values. This serves as a base result dictionary for tests of the _compute_shear_indices method, allowing tests to verify that exceptions in the computation of shear indices are handled gracefully by checking that the relevant keys remain None when exceptions are raised, while still providing valid values for other indices that should be computed successfully. By providing a structured result dictionary with both None and valid values, this method helps ensure that tests can focus on verifying the exception handling logic for specific shear indices while confirming that other parts of the computation are unaffected. 

        Parameters:
            None

        Returns:
            dict: A dictionary with shear index keys, some set to None and some set to valid values. 
        """
        return {
            "bulk_shear_0_1km": 20.0, "bulk_shear_0_6km": 30.0,
            "srh_0_1km": 200.0, "srh_0_3km": 300.0,
            "sbcape": 1000.0, "lcl_pressure": 900.0,
            "mucape": 1500.0,
            "stp": None, "scp": None, "sweat_index": None,
        }

    def test_wind_unit_setup_fails_returns_early(self: 'TestComputeShearIndicesExceptions') -> None:
        """
        This test verifies that the _compute_shear_indices method of the SoundingDiagnostics class handles exceptions raised during the setup of wind units gracefully by printing a message and returning early without attempting to compute shear indices. The test creates a standard profile using the _make_standard_profile helper function, initializes the SoundingDiagnostics class with verbose set to True, and patches the munits module to raise a RuntimeError when accessed. The test then calls _compute_shear_indices and captures the standard output, checking that it contains a message indicating that the MetPy wind unit setup failed, confirming that the method correctly handles exceptions during unit setup by informing the user and not proceeding with computations that would rely on properly set up units. 

        Parameters:
            None

        Returns:
            None
        """
        p, t, td, u, v, h = _make_standard_profile()
        result = self._base_result()
        diag = SoundingDiagnostics(verbose=True)
        bad_units = MagicMock(side_effect=RuntimeError("units broken"))
        captured = StringIO()

        with patch("mpasdiag.diagnostics.sounding.munits", bad_units):
            with patch("sys.stdout", new=captured):
                diag._compute_shear_indices(result, p, t, td, u, v, h)

        assert "MetPy wind unit setup failed" in captured.getvalue()

    def test_bulk_shear_exception_is_silenced(self: 'TestComputeShearIndicesExceptions') -> None:
        """
        This test verifies that the _compute_shear_indices method of the SoundingDiagnostics class handles exceptions raised by the bulk_shear function gracefully. The test creates a standard profile using the _make_standard_profile helper function, initializes the SoundingDiagnostics class with verbose set to False, and patches the bulk_shear function to raise a RuntimeError. The test then calls _compute_shear_indices and checks that the result dictionary has None values for the bulk shear keys, ensuring that exceptions in the bulk shear calculation are silenced and do not cause the method to crash. 

        Parameters:
            None

        Returns:
            None
        """
        p, t, td, u, v, h = _make_standard_profile()
        result = self._base_result()
        diag = SoundingDiagnostics(verbose=False)

        with patch("mpasdiag.diagnostics.sounding.mpcalc.bulk_shear",
                   side_effect=RuntimeError("bulk shear fail")):
            diag._compute_shear_indices(result, p, t, td, u, v, h)

        assert result["bulk_shear_0_1km"] is None
        assert result["bulk_shear_0_6km"] is None

    def test_srh_exception_is_silenced(self: 'TestComputeShearIndicesExceptions') -> None:
        """
        This test verifies that the _compute_shear_indices method of the SoundingDiagnostics class handles exceptions raised by the storm_relative_helicity function gracefully. The test creates a standard profile using the _make_standard_profile helper function, initializes the SoundingDiagnostics class with verbose set to False, and patches the storm_relative_helicity function to raise a RuntimeError. The test then calls _compute_shear_indices and checks that the result dictionary has None values for the srh keys, ensuring that exceptions in the storm_relative_helicity calculation are silenced and do not cause the method to crash. 

        Parameters:
            None

        Returns:
            None
        """
        p, t, td, u, v, h = _make_standard_profile()
        result = self._base_result()
        diag = SoundingDiagnostics(verbose=False)

        with patch("mpasdiag.diagnostics.sounding.mpcalc.storm_relative_helicity",
                   side_effect=RuntimeError("srh fail")):
            diag._compute_shear_indices(result, p, t, td, u, v, h)

        assert result["srh_0_1km"] is None
        assert result["srh_0_3km"] is None

    def test_stp_exception_is_silenced(self: 'TestComputeShearIndicesExceptions') -> None:
        """
        This test verifies that the _compute_shear_indices method of the SoundingDiagnostics class handles exceptions raised by the significant_tornado function gracefully. The test creates a standard profile using the _make_standard_profile helper function, initializes the SoundingDiagnostics class with verbose set to False, and patches the significant_tornado function to raise a RuntimeError. The test then calls _compute_shear_indices and checks that the result dictionary has None values for the stp key, ensuring that exceptions in the significant_tornado calculation are silenced and do not cause the method to crash. 

        Parameters:
            None

        Returns:
            None
        """
        p, t, td, u, v, h = _make_standard_profile()
        result = self._base_result_with_deps()
        diag = SoundingDiagnostics(verbose=False)

        with patch("mpasdiag.diagnostics.sounding.mpcalc.significant_tornado",
                   side_effect=RuntimeError("stp fail")):
            diag._compute_shear_indices(result, p, t, td, u, v, h)

        assert result["stp"] is None

    def test_scp_exception_is_silenced(self: 'TestComputeShearIndicesExceptions') -> None:
        """
        This test verifies that the _compute_shear_indices method of the SoundingDiagnostics class handles exceptions raised by the supercell_composite function gracefully. The test creates a standard profile using the _make_standard_profile helper function, initializes the SoundingDiagnostics class with verbose set to False, and patches the supercell_composite function to raise a RuntimeError. The test then calls _compute_shear_indices and checks that the result dictionary has None values for the scp key, ensuring that exceptions in the supercell_composite calculation are silenced and do not cause the method to crash. 

        Parameters:
            None

        Returns:
            None
        """
        p, t, td, u, v, h = _make_standard_profile()
        result = self._base_result_with_deps()
        diag = SoundingDiagnostics(verbose=False)

        with patch("mpasdiag.diagnostics.sounding.mpcalc.supercell_composite",
                   side_effect=RuntimeError("scp fail")):
            diag._compute_shear_indices(result, p, t, td, u, v, h)

        assert result["scp"] is None

    def test_sweat_exception_is_silenced(self: 'TestComputeShearIndicesExceptions') -> None:
        """
        This test verifies that the _compute_shear_indices method of the SoundingDiagnostics class handles exceptions raised by the sweat_index function gracefully. The test creates a standard profile using the _make_standard_profile helper function, initializes the SoundingDiagnostics class with verbose set to False, and patches the sweat_index function to raise a RuntimeError. The test then calls _compute_shear_indices and checks that the result dictionary has None values for the sweat_index key, ensuring that exceptions in the sweat_index calculation are silenced and do not cause the method to crash. 

        Parameters:
            None

        Returns:
            None
        """
        p, t, td, u, v, h = _make_standard_profile()
        result = self._base_result()
        diag = SoundingDiagnostics(verbose=False)

        with patch("mpasdiag.diagnostics.sounding.mpcalc.sweat_index",
                   side_effect=RuntimeError("sweat fail")):
            diag._compute_shear_indices(result, p, t, td, u, v, h)

        assert result["sweat_index"] is None


class TestExtractPressureProfileNoPressureVars:
    """ Test that _extract_pressure_profile raises a ValueError when no pressure variables are present in the dataset. """

    def test_no_pressure_vars_raises(self: 'TestExtractPressureProfileNoPressureVars') -> None:
        """
        This test verifies that the _extract_pressure_profile method of the SoundingDiagnostics class raises a ValueError when no pressure variables are present in the dataset. The test creates a dataset with only a potential temperature variable, initializes the SoundingDiagnostics class with verbose set to False, and checks that calling _extract_pressure_profile raises a ValueError with the appropriate message. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "theta": (["Time", "nCells", "nVertLevels"], np.full((1, N_CELLS, N_VERT), 300.0)),
        })

        diag = SoundingDiagnostics(verbose=False)

        with pytest.raises(ValueError, match="Cannot determine pressure"):
            diag._extract_pressure_profile(ds, "Time", 0, 0)


class TestExtractTemperatureProfileNoTempVars:
    """ Test that _extract_temperature_profile raises a ValueError when no temperature variables are present in the dataset. """

    def test_no_temp_vars_raises(self: 'TestExtractTemperatureProfileNoTempVars') -> None:
        """
        This test verifies that the _extract_temperature_profile method of the SoundingDiagnostics class raises a ValueError when no temperature variables are present in the dataset. The test creates a dataset with only a pressure variable, initializes the SoundingDiagnostics class with verbose set to False, and checks that calling _extract_temperature_profile raises a ValueError with the appropriate message. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "pressure": (["Time", "nCells", "nVertLevels"], np.ones((1, N_CELLS, N_VERT))),
        })

        diag = SoundingDiagnostics(verbose=False)
        p = np.ones(N_VERT) * 100_000.0

        with pytest.raises(ValueError, match="Cannot find temperature"):
            diag._extract_temperature_profile(ds, "Time", 0, 0, p)


class TestExtractDewpointNoMoistureVar:
    """ Test that _extract_dewpoint_profile handles the absence of moisture variables correctly. """

    def test_no_moisture_verbose_prints_warning(self: 'TestExtractDewpointNoMoistureVar') -> None:
        """
        This test verifies that the _extract_dewpoint_profile method of the SoundingDiagnostics class prints a warning message when no moisture variable is found in the dataset and verbose mode is enabled. The test creates a dataset with only a pressure variable, initializes the SoundingDiagnostics class with verbose set to True, and calls _extract_dewpoint_profile while capturing the standard output. The test checks that the captured output contains a message indicating that no moisture variable was found, and that the result is an array of NaN values, confirming that the method handles the absence of moisture variables gracefully while providing feedback to the user in verbose mode. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "pressure": (["Time", "nCells", "nVertLevels"], np.ones((1, N_CELLS, N_VERT))),
        })

        diag = SoundingDiagnostics(verbose=True)
        p = np.linspace(100_000.0, 20_000.0, N_VERT)
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            result = diag._extract_dewpoint_profile(ds, "Time", 0, 0, p)

        assert "No moisture variable found" in captured.getvalue()
        assert np.all(np.isnan(result))

    def test_no_moisture_silent_when_not_verbose(self: 'TestExtractDewpointNoMoistureVar') -> None:
        """
        This test verifies that the _extract_dewpoint_profile method of the SoundingDiagnostics class does not print a warning message when no moisture variable is found in the dataset and verbose mode is disabled. The test creates a dataset with only a pressure variable, initializes the SoundingDiagnostics class with verbose set to False, and calls _extract_dewpoint_profile while capturing the standard output. The test checks that the captured output is empty, and that the result is an array of NaN values, confirming that the method handles the absence of moisture variables gracefully without providing feedback to the user when verbose mode is disabled. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "pressure": (["Time", "nCells", "nVertLevels"], np.ones((1, N_CELLS, N_VERT))),
        })

        diag = SoundingDiagnostics(verbose=False)
        p = np.linspace(100_000.0, 20_000.0, N_VERT)
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            result = diag._extract_dewpoint_profile(ds, "Time", 0, 0, p)

        assert captured.getvalue() == ""
        assert np.all(np.isnan(result))


class TestExtractWindProfilesNoVars:
    """ Test that _extract_wind_profiles handles the absence of wind variables correctly. """

    def test_no_wind_vars_verbose_prints_warning(self: 'TestExtractWindProfilesNoVars') -> None:
        """
        This test verifies that the _extract_wind_profiles method of the SoundingDiagnostics class prints a warning message when no wind variables are found in the dataset and verbose mode is enabled. The test creates a dataset with only a pressure variable, initializes the SoundingDiagnostics class with verbose set to True, and calls _extract_wind_profiles while capturing the standard output. The test checks that the captured output contains a message indicating that no wind variables were found, and that the result is None for both u and v components, confirming that the method handles the absence of wind variables gracefully while providing feedback to the user in verbose mode. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "pressure": (["Time", "nCells", "nVertLevels"], np.ones((1, N_CELLS, N_VERT))),
        })

        diag = SoundingDiagnostics(verbose=True)
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            u, v = diag._extract_wind_profiles(ds, "Time", 0, 0)

        assert "No wind variables found" in captured.getvalue()
        assert u is None
        assert v is None

    def test_no_wind_vars_silent_when_not_verbose(self: 'TestExtractWindProfilesNoVars') -> None:
        """
        This test verifies that the _extract_wind_profiles method of the SoundingDiagnostics class does not print a warning message when no wind variables are found in the dataset and verbose mode is disabled. The test creates a dataset with only a pressure variable, initializes the SoundingDiagnostics class with verbose set to False, and calls _extract_wind_profiles while capturing the standard output. The test checks that the captured output is empty, and that the result is None for both u and v components, confirming that the method handles the absence of wind variables gracefully without providing feedback to the user when verbose mode is disabled. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "pressure": (["Time", "nCells", "nVertLevels"], np.ones((1, N_CELLS, N_VERT))),
        })

        diag = SoundingDiagnostics(verbose=False)
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            u, v = diag._extract_wind_profiles(ds, "Time", 0, 0)

        assert captured.getvalue() == ""
        assert u is None and v is None


class TestExtractHeightStaggered:
    """ Test that _extract_height_profile correctly averages staggered height levels to standard levels. """

    def test_staggered_height_averaged_to_nvert_levels(self: 'TestExtractHeightStaggered') -> None:
        """
        This test verifies that the _extract_height_profile method of the SoundingDiagnostics class correctly averages staggered height levels to produce a height profile with the same number of levels as the nVertLevels dimension. The test creates a dataset with a zgrid variable that has nVertLevelsP1 levels, initializes the SoundingDiagnostics class with verbose set to False, and calls _extract_height_profile. The test checks that the result is not None, has a length equal to N_VERT, and that the first value of the result is approximately equal to the average of the first two values of the input heights, confirming that the method correctly performs mid-level averaging to convert from staggered heights to standard levels. 

        Parameters:
            None

        Returns:
            None
        """
        n_vert_p1 = N_VERT + 1
        heights_p1 = np.linspace(0.0, 15_000.0, n_vert_p1)

        ds = xr.Dataset({
            "zgrid": (["nCells", "nVertLevelsP1"],
                      np.tile(heights_p1, (N_CELLS, 1))),
        })

        ds = ds.assign_coords({"nCells": np.arange(N_CELLS)})
        ds["dummy"] = (["nCells", "nVertLevels"], np.zeros((N_CELLS, N_VERT)))

        diag = SoundingDiagnostics(verbose=False)
        result = diag._extract_height_profile(ds, "Time", 0, 0)

        assert result is not None
        assert len(result) == N_VERT
        assert np.isclose(result[0], 0.5 * (heights_p1[0] + heights_p1[1]))


class TestExtractHeightException:
    """ Test that _extract_height_profile handles exceptions during indexing gracefully. """
    
    def test_isel_exception_continues_and_returns_none(self: 'TestExtractHeightException') -> None:
        """
        This test verifies that the _extract_height_profile method of the SoundingDiagnostics class continues execution and returns None when an exception is raised during the isel operation due to the absence of the 'nCells' dimension in the zgrid variable. The test creates a dataset with a zgrid variable that lacks the 'nCells' dimension, initializes the SoundingDiagnostics class with verbose set to False, and calls _extract_height_profile. The test checks that the result is None, confirming that the method handles exceptions during indexing gracefully by returning None instead of crashing. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "pressure": (["Time", "nCells", "nVertLevels"], np.ones((1, N_CELLS, N_VERT))),
            "zgrid": (["some_other_dim"], np.ones(10)),
        })

        diag = SoundingDiagnostics(verbose=False)
        result = diag._extract_height_profile(ds, "Time", 0, 0)
        assert result is None

    def test_isel_exception_verbose_prints_warning(self: 'TestExtractHeightException') -> None:
        """
        This test verifies that the _extract_height_profile method of the SoundingDiagnostics class prints a warning message when an exception is raised during the isel operation due to the absence of the 'nCells' dimension in the zgrid variable and verbose mode is enabled. The test creates a dataset with a zgrid variable that lacks the 'nCells' dimension, initializes the SoundingDiagnostics class with verbose set to True, and calls _extract_height_profile while capturing the standard output. The test checks that the captured output contains a message indicating that the height profile extraction failed, and that the result is None, confirming that the method handles exceptions during indexing gracefully by providing feedback to the user in verbose mode and returning None instead of crashing. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "zgrid": (["some_other_dim"], np.ones(10)),
        })

        diag = SoundingDiagnostics(verbose=True)
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            result = diag._extract_height_profile(ds, "Time", 0, 0)

        assert "Warning: Failed to extract height profile" in captured.getvalue()
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

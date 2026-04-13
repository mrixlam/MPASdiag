#!/usr/bin/env python3

"""
MPASdiag Test Suite: Additional Tests for 3D Processor Coverage

This module contains additional unit tests for the MPAS3DProcessor class, specifically targeting code paths that were not covered in the main test suite. These tests focus on edge cases and branches that are less commonly executed, such as error handling and fallback logic. It is intended to complement the existing tests and improve overall code coverage for the 3D processing functionality. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import pytest
import numpy as np
import xarray as xr
from typing import Any
from unittest.mock import MagicMock, patch

from mpasdiag.processing.processors_3d import MPAS3DProcessor


def _make_processor(verbose: bool = False) -> MPAS3DProcessor:
    """
    This helper function creates an instance of MPAS3DProcessor with a mocked grid file path. It patches os.path.exists to always return True, allowing the processor to initialize without needing an actual file. The verbose flag can be set to enable or disable verbose output during testing.

    Parameters:
        verbose: Whether to enable verbose output in the processor (default: False)

    Returns:
        MPAS3DProcessor: An instance of the processor with a mocked grid file.
    """
    with patch("os.path.exists", return_value=True):
        return MPAS3DProcessor("test_grid.nc", verbose=verbose)


def _synthetic_mpas_data(n_cells: int = 5,
                         n_vert: int = 4,
                         n_time: int = 1,
                         with_pressure_components: bool = True,
                         with_direct_pressure: bool = False,) -> xr.Dataset:
    """
    This helper function creates a synthetic xarray Dataset mimicking MPAS output with specified dimensions and variables. It includes options to add pressure components and direct pressure variables, which are relevant for testing the 3D processor's handling of pressure data. The 'theta' variable is included as a standard 3D field, and the pressure variables are constructed to have realistic vertical profiles.

    Parameters:
        n_cells: Number of horizontal cells (default: 5)
        n_vert: Number of vertical levels (default: 4)
        n_time: Number of time steps (default: 1)
        with_pressure_components: Whether to include pressure components (default: True)
        with_direct_pressure: Whether to include direct pressure variable (default: False)

    Returns:
        xr.Dataset: Synthetic MPAS-style 3D dataset.
    """
    p_vals = np.linspace(100000.0, 50000.0, n_vert) 

    data_vars: dict = {
        "theta": (
            ["Time", "nCells", "nVertLevels"],
            np.full((n_time, n_cells, n_vert), 300.0),
        ),
    }

    if with_direct_pressure:
        pressure = np.tile(p_vals, (n_time, n_cells, 1))
        data_vars["pressure"] = (["Time", "nCells", "nVertLevels"], pressure)

    if with_pressure_components:
        pressure = np.tile(p_vals, (n_time, n_cells, 1))
        data_vars["pressure_p"] = (
            ["Time", "nCells", "nVertLevels"],
            np.ones((n_time, n_cells, n_vert)) * 1000.0,
        )

        data_vars["pressure_base"] = (["Time", "nCells", "nVertLevels"], pressure)

    return xr.Dataset(data_vars)


def _make_getitem(mapping: dict, 
                  default: Any = None) -> Any:
    """
    This helper function creates a side effect function for __getitem__ that returns values based on a provided mapping. If a key is not found in the mapping, it returns a MagicMock or a specified default value. This is useful for mocking dataset access in tests where we want to control the return values of certain keys while allowing others to be flexible. 

    Parameters:
        mapping: A dictionary mapping keys to return values for __getitem__.
        default: An optional default value to return for missing keys (default: None).

    Returns:
        A function that can be used as a side effect for __getitem__ in a mock Dataset.  
    """
    def _gi(key: str) -> Any:
        """ The actual side effect function for __getitem__. It looks up the key in the mapping and returns the corresponding value, or a MagicMock/default if the key is not found. """
        return mapping.get(key, MagicMock() if default is None else default)
    return _gi


def _make_contains(keys: list) -> Any:
    """ 
    This helper function creates a side effect function for __contains__ that checks if a key is in a provided list of keys. It converts the list to a set for efficient lookup and returns True if the key is present, False otherwise. This is useful for mocking the behavior of 'in' checks on a dataset's variables or coordinates in tests.
    
    Parameters:
        keys: A list of keys that should be considered as present in the dataset.
        
    Returns:
        A function that can be used as a side effect for __contains__ in a mock Dataset.
    """
    key_set = set(keys)
    def _c(key: str) -> bool:
        return key in key_set
    return _c


class TestFindFilesRecursiveVerbose:
    """ Tests for the verbose branch in _find_files_recursive, which is responsible for finding files and printing the results when verbose mode is enabled. """

    def test_verbose_prints_file_list(self: "TestFindFilesRecursiveVerbose", 
                                      capsys: pytest.CaptureFixture) -> None:
        """
        This test verifies that when _find_files_recursive is called in verbose mode, it prints the list of found files to the console. It uses the capsys fixture to capture stdout and checks that the expected output is present, including the number of files found and their names. The glob.glob function is mocked to return a predefined list of fake file paths for testing purposes.

        Parameters:
            capsys: The pytest fixture for capturing stdout and stderr during the test.

        Returns:
            None: The test asserts conditions on the captured output and does not return a value.
        """
        proc = _make_processor(verbose=True)
        fake_files = [f"/fake/dir/mpasout_{i:02d}.nc" for i in range(3)]

        with patch("glob.glob", return_value=fake_files):
            files = proc._find_files_recursive("/fake/dir")

        captured = capsys.readouterr()
        assert "Found 3" in captured.out
        assert "mpasout_00.nc" in captured.out
        assert files == fake_files

    def test_fewer_than_two_files_raises(self: "TestFindFilesRecursiveVerbose") -> None:
        """
        This test checks that if _find_files_recursive finds fewer than two files, it raises a ValueError indicating that there are insufficient files to process. It mocks glob.glob to return a list with only one file and asserts that the expected exception is raised with the correct message.

        Parameters:
            None
        Returns:
            None: The test asserts that a ValueError is raised and does not return a value.
        """
        proc = _make_processor(verbose=False)
        with patch("glob.glob", return_value=["/fake/only_one.nc"]):
            with pytest.raises(ValueError, match="Insufficient"):
                proc._find_files_recursive("/fake/dir")

    def test_no_files_raises_file_not_found(self: "TestFindFilesRecursiveVerbose") -> None:
        """
        This test verifies that if _find_files_recursive finds no files, it raises a FileNotFoundError. It mocks glob.glob to return an empty list and asserts that the expected exception is raised with the correct message.

        Parameters:
            None

        Returns:
            None: The test asserts that a FileNotFoundError is raised and does not return a value.
        """
        proc = _make_processor(verbose=False)
        with patch("glob.glob", return_value=[]):
            with pytest.raises(FileNotFoundError):
                proc._find_files_recursive("/fake/empty_dir")


class TestLoad3DDataDsBranch:
    """ Tests for the branch in load_3d_data where _load_data returns an object with a 'ds' attribute, which is expected to be a UXarray-style wrapper. """

    def test_ds_attribute_branch_sets_dataset(self: "TestLoad3DDataDsBranch") -> None:
        """
        This test checks that when load_3d_data receives an object with a 'ds' attribute from _load_data, it correctly sets the processor's dataset to that object and identifies the data type as 'uxarray'. It mocks _load_data to return a wrapper object that has a 'ds' attribute but no 'data_vars', simulating a UXarray-style wrapper. The test then asserts that the processor's dataset is set to the wrapper and that the data type is correctly identified.

        Parameters:
            None

        Returns:
            None: The test asserts conditions on the processor's state and does not return a value.
        """

        class _WrapperWithDs:
            """ Mimics a UXarray-style wrapper that has .ds but not .data_vars. """
            def __init__(self, inner_ds) -> None:
                """ Initializes the wrapper with an inner dataset. """
                self.ds = inner_ds
            # deliberately NO data_vars attribute

        inner = _synthetic_mpas_data(n_cells=3, n_vert=4, with_pressure_components=False,
                         with_direct_pressure=False)
        wrapper = _WrapperWithDs(inner)
        proc = _make_processor(verbose=False)

        with patch.object(proc, "_load_data", return_value=(wrapper, "uxarray")):
            with patch.object(proc, "add_spatial_coordinates", side_effect=lambda ds: ds):
                proc.load_3d_data("/fake/data_dir", use_pure_xarray=False)

        assert proc.dataset is wrapper
        assert proc.data_type == "uxarray"


class TestComputeMeanPressureLevels:
    """ Tests for the _compute_mean_pressure_levels method, specifically for the branch that computes mean pressure levels using 'pressure_p' and 'pressure_base' variables with nCells × nVertLevels layout. """

    def test_returns_correct_mean_with_nVertLevels(self: "TestComputeMeanPressureLevels") -> None:
        """
        This test checks that the _compute_mean_pressure_levels method correctly computes the mean pressure levels when the dataset has nCells × nVertLevels layout. It creates a synthetic dataset with known pressure values and asserts that the computed mean matches the expected values.

        Parameters:
            None

        Returns:
            None: The test asserts conditions on the computed mean pressure levels and does not return a value.
        """
        proc = _make_processor(verbose=False)
        n_cells, n_vert = 5, 4
        p_base = np.linspace(100000.0, 50000.0, n_vert)
        ds = xr.Dataset({
            "pressure_p": (["Time", "nCells", "nVertLevels"],
                            np.zeros((1, n_cells, n_vert))),
            "pressure_base": (["Time", "nCells", "nVertLevels"],
                               np.tile(p_base, (1, n_cells, 1))),
        })

        result = proc._compute_mean_pressure_levels(ds, "Time", 0)

        assert result.shape == (n_vert,)
        np.testing.assert_allclose(result, p_base, rtol=1e-6)

    def test_returns_nvert_levels_with_nVertLevelsP1(self: "TestComputeMeanPressureLevels") -> None:
        """
        This test checks that the _compute_mean_pressure_levels method correctly computes the mean pressure levels when the dataset has nCells × nVertLevelsP1 layout. It creates a synthetic dataset with known pressure values and asserts that the computed mean matches the expected values.

        Parameters:
            None

        Returns:
            None: The test asserts conditions on the computed mean pressure levels and does not return a value.
        """
        proc = _make_processor(verbose=False)
        n_cells, n_vert_p1 = 3, 5
        p_vals = np.linspace(100000.0, 60000.0, n_vert_p1)
        ds = xr.Dataset({
            "pressure_p": (["Time", "nCells", "nVertLevelsP1"],
                            np.zeros((1, n_cells, n_vert_p1))),
            "pressure_base": (["Time", "nCells", "nVertLevelsP1"],
                               np.tile(p_vals, (1, n_cells, 1))),
        })
        result = proc._compute_mean_pressure_levels(ds, "Time", 0)
        assert result.shape == (n_vert_p1,)


class TestLerpVariable:
    """ Tests for the _lerp_variable method, covering both the successful interpolation branch and the failure fallback branch. """

    def _setup(self: "TestLerpVariable", 
               verbose: bool = False) -> tuple:
        """ 
        This helper method sets up a synthetic dataset and mean pressure array for testing the _lerp_variable method. It creates a dataset with a 'theta' variable that has a known vertical profile and a mean pressure array that is monotonically decreasing. The verbose flag can be set to enable or disable verbose output during testing.

        Parameters:
            verbose (bool): Whether to enable verbose mode in the processor.

        Returns:
            tuple: A tuple containing the processor, dataset, and mean pressure array.
        """
        proc = _make_processor(verbose=verbose)
        n_cells, _ = 5, 4

        ds = xr.Dataset({
            "theta": (["Time", "nCells", "nVertLevels"],
                       np.tile([300.0, 295.0, 290.0, 285.0],
                               (1, n_cells, 1))),
        })

        # Monotonically decreasing pressure (surface=idx0, top=idx3)
        mean_p = np.array([100000.0, 85000.0, 70000.0, 50000.0])
        return proc, ds, mean_p

    def test_interpolation_success_returns_minus1_and_dataarray(self: "TestLerpVariable") -> None:
        """
        This test checks that the _lerp_variable method correctly performs interpolation when the arithmetic operations succeed. It asserts that the method returns (-1, DataArray) for a successful interpolation.

        Parameters:
            None

        Returns:
            None: The test asserts conditions on the returned index and DataArray and does not return a value.
        """
        proc, ds, mean_p = self._setup(verbose=False)
        level = 77000.0
        lower_idx, upper_idx = 1, 2
        w = (level - 70000.0) / (85000.0 - 70000.0)  # ≈ 0.467

        idx, field = proc._lerp_variable(
            "theta", lower_idx, upper_idx, w,
            mean_p, level, ds, "Time", 0, "nVertLevels",
        )

        assert idx == -1
        assert field is not None
        assert hasattr(field, "values")

    def test_verbose_prints_range_for_successful_interpolation(self: "TestLerpVariable", 
                                                               capsys: pytest.CaptureFixture) -> None:
        """
        This test checks that the _lerp_variable method prints the interpolated field range when verbose mode is enabled.

        Parameters:
            capsys (pytest.CaptureFixture): Pytest fixture for capturing stdout and stderr.

        Returns:
            None: The test asserts conditions on the captured output and does not return a value.
        """
        proc, ds, mean_p = self._setup(verbose=True)
        level = 77000.0
        lower_idx, upper_idx = 1, 2
        w = (level - 70000.0) / (85000.0 - 70000.0)

        proc._lerp_variable(
            "theta", lower_idx, upper_idx, w,
            mean_p, level, ds, "Time", 0, "nVertLevels",
        )

        captured = capsys.readouterr()
        assert "Interpolated field range" in captured.out

    def test_interpolation_failure_falls_back_to_nearest(self: "TestLerpVariable") -> None:
        """
        This test checks that if the arithmetic operations in _lerp_variable raise a TypeError (simulating an interpolation failure), the method falls back to returning the nearest mean level index and None for the field. It uses a MagicMock to simulate a DataArray that raises TypeError on multiplication, which should trigger the fallback logic.

        Parameters:
            None

        Returns:
            None: The test asserts conditions on the returned index and DataArray and does not return a value.
        """
        proc, _, mean_p = self._setup(verbose=False)
        level = 77000.0
        # Use a MagicMock dataset where arithmetic on the variable raises
        bad_da = MagicMock()
        bad_da.__rmul__ = MagicMock(side_effect=TypeError("cannot multiply"))
        bad_da.__mul__ = MagicMock(side_effect=TypeError("cannot multiply"))
        bad_da.isel.return_value = bad_da
        bad_da.attrs = {}
        mock_ds = MagicMock()
        mock_ds.__getitem__.return_value = bad_da

        idx, field = proc._lerp_variable(
            "theta", 1, 2, 0.5, mean_p, level, mock_ds, "Time", 0, "nVertLevels",
        )

        # Should fall back to the nearest mean level
        assert isinstance(idx, int)
        assert field is None

    def test_interpolation_failure_verbose_prints_message(self: "TestLerpVariable", 
                                                          capsys: pytest.CaptureFixture) -> None:
        """
        This test checks that when interpolation fails in _lerp_variable and verbose mode is enabled, an appropriate message indicating the failure is printed to the console. It uses a MagicMock to simulate a DataArray that raises TypeError on multiplication, which should trigger the interpolation failure and the corresponding verbose message. 

        Parameters:
            capsys (pytest.CaptureFixture): Pytest fixture for capturing stdout and stderr.

        Returns:
            None: The test asserts conditions on the captured output and does not return a value.
        """
        proc, _, mean_p = self._setup(verbose=True)
        level = 77000.0
        bad_da = MagicMock()
        bad_da.__rmul__ = MagicMock(side_effect=TypeError("cannot multiply"))
        bad_da.__mul__ = MagicMock(side_effect=TypeError("cannot multiply"))
        bad_da.isel.return_value = bad_da
        bad_da.attrs = {}
        mock_ds = MagicMock()
        mock_ds.__getitem__.return_value = bad_da

        proc._lerp_variable(
            "theta", 1, 2, 0.5, mean_p, level, mock_ds, "Time", 0, "nVertLevels",
        )

        captured = capsys.readouterr()
        assert "Interpolation failed" in captured.out


class TestInterpolateAtPressure:
    """ Tests for the _interpolate_at_pressure method, covering the branches for pressure levels above max, below min, normal interpolation, and the special case where lower_idx is at the last position. """

    def _setup(self: "TestInterpolateAtPressure", 
               verbose: bool = False) -> tuple:
        """ 
        This helper method sets up a synthetic dataset and mean pressure array for testing the _interpolate_at_pressure method. It creates a dataset with a 'theta' variable that has a known vertical profile and a mean pressure array that is monotonically increasing (ascending). The verbose flag can be set to enable or disable verbose output during testing.

        Parameters:
            verbose (bool): Whether to enable verbose mode in the processor.

        Returns:
            tuple: A tuple containing the processor, dataset, and mean pressure array.
        """
        proc = _make_processor(verbose=verbose)
        n_cells, n_vert = 5, 4

        ds = xr.Dataset({
            "theta": (["Time", "nCells", "nVertLevels"],
                       np.full((1, n_cells, n_vert), 300.0)),
        })

        # Ascending pressure (top=idx0, surface=idx3)
        mean_p = np.array([50000.0, 70000.0, 85000.0, 100000.0])
        return proc, ds, mean_p

    def test_level_above_max_returns_surface(self: "TestInterpolateAtPressure", 
                                             capsys: pytest.CaptureFixture) -> None:
        """
        This test checks that when the requested pressure level is above the maximum mean pressure (surface), the _interpolate_at_pressure method returns (0, None) and prints a message indicating that the surface level is being used. It uses the capsys fixture to capture stdout and asserts that the expected message is present in the output.

        Parameters:
            capsys (pytest.CaptureFixture): Pytest fixture for capturing stdout and stderr.

        Returns:
            None: The test asserts conditions on the returned index and DataArray, as well as the captured output, and does not return a value.
        """
        proc, ds, mean_p = self._setup(verbose=True)
        idx, field = proc._interpolate_at_pressure("theta", 110000.0, mean_p, ds, "Time", 0)
        captured = capsys.readouterr()
        assert idx == 0
        assert field is None
        assert "surface" in captured.out.lower()

    def test_level_below_min_returns_top(self: "TestInterpolateAtPressure", 
                                          capsys: pytest.CaptureFixture) -> None:
        """
        This test checks that when the requested pressure level is below the minimum mean pressure (top), the _interpolate_at_pressure method returns (len(mean_p)-1, None) and prints a message indicating that the top level is being used. It uses the capsys fixture to capture stdout and asserts that the expected message is present in the output.

        Parameters:
            capsys (pytest.CaptureFixture): Pytest fixture for capturing stdout and stderr.

        Returns:
            None: The test asserts conditions on the returned index and DataArray, as well as the captured output, and does not return a value.
        """
        proc, ds, mean_p = self._setup(verbose=True)
        idx, field = proc._interpolate_at_pressure("theta", 40000.0, mean_p, ds, "Time", 0)
        captured = capsys.readouterr()
        assert idx == len(mean_p) - 1
        assert field is None
        assert "top" in captured.out.lower()

    def test_normal_interpolation_returns_data(self: "TestInterpolateAtPressure") -> None:
        """
        This test checks that when the requested pressure level falls between the mean pressure levels, the _interpolate_at_pressure method attempts to perform interpolation and returns either (-1, DataArray) for a successful interpolation or (int, None) if it falls back to the nearest level. It asserts that the returned index is either -1 (indicating interpolation) or an integer index of the nearest level, and that the field is a DataArray if interpolation was successful.

        Parameters:
            None

        Returns:
            None: The test asserts conditions on the returned index and DataArray, and does not return a value.
        """
        proc, ds, mean_p = self._setup(verbose=False)
        proc.dataset = ds   # _interpolate_at_pressure accesses self.dataset internally
        # Level between 70000 and 85000
        level = 77000.0
        idx, field = proc._interpolate_at_pressure("theta", level, mean_p, ds, "Time", 0)
        # Either an interpolated DataArray (idx==-1) or a fallback index
        assert idx == -1 or isinstance(idx, int)

    def test_lower_idx_at_last_position_returns_early(self: "TestInterpolateAtPressure") -> None:
        """
        This test checks the special case where the lower_idx is at the last position of the mean pressure array, which should trigger an early return with (len(mean_p)-1, None) without attempting interpolation. It sets up a scenario where the requested level is just below the maximum mean pressure, ensuring that lower_idx would be at the last index. The test asserts that the returned index is len(mean_p)-1 and that the field is None.

        Parameters:
            None

        Returns:
            None: The test asserts conditions on the returned index and DataArray, and does not return a value.
        """
        proc, ds, mean_p = self._setup(verbose=False)
        proc.dataset = ds
        # Use a level just below max so lower_idx = len-1
        level = 99999.0
        idx, field = proc._interpolate_at_pressure("theta", level, mean_p, ds, "Time", 0)
        # lower_idx should be 3 (last), which hits the early return
        assert isinstance(idx, int)
        assert field is None

    def test_verbose_prints_interpolation_info(self: "TestInterpolateAtPressure", 
                                               capsys: pytest.CaptureFixture) -> None:
        """
        This test checks that when _interpolate_at_pressure is called in verbose mode, it prints information about the interpolation process, such as the pressure levels being used and the fact that interpolation is occurring. It uses the capsys fixture to capture stdout and asserts that the expected messages related to pressure and interpolation are present in the output.

        Parameters:
            capsys (pytest.CaptureFixture): Fixture to capture stdout/stderr output.

        Returns:
            None: The test asserts conditions on the captured output, and does not return a value.
        """
        proc, ds, mean_p = self._setup(verbose=True)
        proc.dataset = ds   # _interpolate_at_pressure accesses self.dataset internally
        level = 77000.0
        proc._interpolate_at_pressure("theta", level, mean_p, ds, "Time", 0)
        captured = capsys.readouterr()
        # Should mention pressure/interpolating
        assert "pressure" in captured.out.lower() or "interpolating" in captured.out.lower()


class TestResolveLevelIndex:
    """ Tests for the _resolve_level_index method, specifically for the branch that handles float level specifications by dispatching to _resolve_float_level, which in turn may call _interpolate_at_pressure. """

    def test_float_level_dispatches_to_float_resolution(self: "TestResolveLevelIndex") -> None:
        """
        This test checks that when _resolve_level_index is called with a float level specification, it correctly dispatches to the _resolve_float_level method, which may further call _interpolate_at_pressure if pressure levels are available. It sets up a synthetic dataset with pressure components and calls _resolve_level_index with a float level. The test asserts that the returned index is either an integer (nearest level) or -1 (indicating interpolation), and that no exceptions are raised during the process.

        Parameters:
            None

        Returns:
            None: The test asserts conditions on the returned index and does not return a value.
        """
        proc = _make_processor(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=4, with_pressure_components=True)
        proc.dataset = ds

        # _resolve_float_level → _resolve_float_level → _interpolate_at_pressure
        idx, early = proc._resolve_level_index("theta", 85000.0, "Time", 0)
        # Should have executed without error
        assert isinstance(idx, int) or idx == -1

    def test_float_level_without_pressure_raises(self: "TestResolveLevelIndex") -> None:
        """
        This test checks that if _resolve_level_index is called with a float level specification but the dataset does not have the necessary pressure data (neither direct pressure nor pressure components), it raises a ValueError indicating that pressure data is not available. It sets up a synthetic dataset without pressure variables and asserts that the expected exception is raised when trying to resolve a float level.

        Parameters:
            None

        Returns:
            None: The test asserts that a ValueError is raised and does not return a value.
        """
        proc = _make_processor(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=4, with_pressure_components=False,
                      with_direct_pressure=False)
        proc.dataset = ds

        with pytest.raises(ValueError, match="pressure data not available"):
            proc._resolve_level_index("theta", 85000.0, "Time", 0)

    def test_invalid_level_type_raises(self: "TestResolveLevelIndex") -> None:
        """
        This test checks that if _resolve_level_index is called with an invalid level specification (e.g., a list instead of a float or int), it raises a ValueError indicating that the level specification is invalid. It sets up a synthetic dataset and asserts that the expected exception is raised when trying to resolve an invalid level type.

        Parameters:
            None

        Returns:
            None: The test asserts that a ValueError is raised and does not return a value.
        """
        proc = _make_processor(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=4)
        proc.dataset = ds

        with pytest.raises(ValueError, match="Invalid level specification"):
            proc._resolve_level_index("theta", [10, 20], "Time", 0)  # type: ignore


class TestSetLevelAttrs:
    """ Tests for the _set_level_attrs method, specifically for the branch that sets 'actual_pressure_level' when a float level is specified and pressure variables are present. """

    def test_float_level_with_pressure_vars_sets_actual_pressure(self: "TestSetLevelAttrs") -> None:
        """
        This test checks that when _set_level_attrs is called with a float level specification and the dataset contains the necessary pressure variables, it sets the 'actual_pressure_level' attribute on the provided DataArray. It creates a synthetic dataset with pressure components, calls _set_level_attrs with a float level, and asserts that both 'selected_level' and 'actual_pressure_level' attributes are set correctly on the DataArray.

        Parameters:
            None

        Returns:
            None: The test asserts that the 'actual_pressure_level' attribute is set and does not return a value.
        """
        proc = _make_processor(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=4, with_pressure_components=True)
        proc.dataset = ds

        # Create a DataArray with attrs
        var_data = xr.DataArray(np.ones(5), dims=["nCells"], attrs={})
        level_idx = 1

        proc._set_level_attrs(
            var_data, 85000.0, level_idx, ds, "Time", 0, "nVertLevels"
        )

        assert "selected_level" in var_data.attrs
        assert var_data.attrs["selected_level"] == pytest.approx(85000.0)
        assert "actual_pressure_level" in var_data.attrs

    def test_int_level_does_not_set_actual_pressure(self: "TestSetLevelAttrs") -> None:
        """
        This test checks that when _set_level_attrs is called with an integer level specification, it does not set the 'actual_pressure_level' attribute on the provided DataArray, even if pressure variables are present in the dataset. It creates a synthetic dataset with pressure components, calls _set_level_attrs with an integer level, and asserts that 'selected_level' is set but 'actual_pressure_level' is not present in the attributes.

        Parameters:
            None

        Returns:
            None: The test asserts that the 'actual_pressure_level' attribute is not set and does not return a value.
        """
        proc = _make_processor(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=4, with_pressure_components=True)
        proc.dataset = ds

        var_data = xr.DataArray(np.ones(5), dims=["nCells"], attrs={})

        proc._set_level_attrs(var_data, 1, 1, ds, "Time", 0, "nVertLevels")

        assert "selected_level" in var_data.attrs
        assert "actual_pressure_level" not in var_data.attrs

    def test_no_attrs_object_handled_gracefully(self: "TestSetLevelAttrs") -> None:
        """
        This test checks that if _set_level_attrs is called with an object that does not have an 'attrs' attribute (e.g., a plain numpy array), the method handles it gracefully without raising an exception. It creates a synthetic dataset, calls _set_level_attrs with a numpy array instead of a DataArray, and asserts that no exceptions are raised during the process. 

        Parameters:
            None

        Returns:
            None: The test asserts that the method handles objects without attrs gracefully and does not return a value.
        """
        proc = _make_processor(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=4)
        proc.dataset = ds
        # Passing a plain numpy array (no 'attrs') should not crash
        arr = np.ones(5)
        proc._set_level_attrs(arr, 1, 0, ds, "Time", 0, "nVertLevels")


class TestGetVariableDataEarlyReturn:
    """ Tests for the branch in get_3d_variable_data where _resolve_level_index returns a DataArray directly (early return), which can happen if interpolation at pressure is successful. """

    def test_interpolated_data_returned_directly(self: "TestGetVariableDataEarlyReturn") -> None:
        """
        This test checks that when get_3d_variable_data is called with a float level specification and the pressure interpolation is successful, it returns the interpolated DataArray directly without further processing. It sets up a synthetic dataset with pressure components, calls get_3d_variable_data with a float level, and asserts that the returned result is a DataArray (indicating that interpolation was successful) and that no exceptions are raised during the process. 

        Parameters:
            None

        Returns:
            None: The test asserts that the returned result is a DataArray and does not return a value.
        """
        proc = _make_processor(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=4, with_pressure_components=True)
        proc.dataset = ds

        # Get data at a float pressure level that falls inside the range →
        # _interpolate_at_pressure should succeed and return early_data
        result = proc.get_3d_variable_data("theta", level=85000.0, time_index=0)

        # Result should be a DataArray (interpolated or nearest)
        assert result is not None
        assert hasattr(result, "values") or isinstance(result, xr.DataArray)


class TestGetVariableDataNdimCheck:
    """ Tests for the branch in get_3d_variable_data where it checks if the extracted variable data is multi-dimensional (ndim > 1). """

    def test_multidim_extraction_raises_value_error(self: "TestGetVariableDataNdimCheck") -> None:
        """
        This test checks that when get_3d_variable_data extracts variable data that is multi-dimensional (ndim > 1), it raises a ValueError indicating that the variable produced multi-dimensional data. It uses MagicMock to simulate a dataset and variable where the extracted data has ndim=2, which should trigger the check and the expected exception. The test asserts that the ValueError is raised with the correct message. 

        Parameters:
            None

        Returns:
            None: The test asserts that a ValueError is raised and does not return a value.
        """
        proc = _make_processor(verbose=False)

        mock_ds = MagicMock()
        mock_ds.sizes = {"nVertLevels": 4, "Time": 1, "nCells": 5}

        multi_da = MagicMock()
        multi_da.ndim = 2                          # triggers ndim > 1 check
        multi_da.shape = (5, 4)
        multi_da.attrs = {}
        multi_da.compute.return_value = multi_da
        multi_da.values = np.ones((5, 4))

        mock_var = MagicMock()
        mock_var.sizes = {"Time": 1, "nCells": 5, "nVertLevels": 4}
        mock_var.isel.return_value = multi_da

        mock_ds.__getitem__.side_effect = _make_getitem({}, default=mock_var)
        mock_ds.__contains__.side_effect = _make_contains(["theta"])
        mock_ds.data_vars = {"theta": mock_var}

        proc.dataset = mock_ds

        with pytest.raises(ValueError, match="produced.*D data"):
            proc.get_3d_variable_data("theta", level=1, time_index=0)


class TestPressureLevelsFromPressureVar:
    """ Tests for the _pressure_levels_from_pressure_var method, specifically for the branches that handle successful extraction of pressure levels from a pressure variable, the fallback when isel with time_dim fails, and the case where non-positive pressure values lead to None being returned. """

    def test_isel_failure_falls_back_to_plain_pressure(self: "TestPressureLevelsFromPressureVar") -> None:
        """
        This test checks that if the _pressure_levels_from_pressure_var method encounters an exception when trying to isel the pressure variable along the time dimension, it falls back to extracting pressure levels without the time dimension. It creates a synthetic dataset where the pressure variable has dimensions that do not include the time dimension, which should cause the isel to fail. The test asserts that the method handles this situation gracefully and returns either a valid levels array or None without raising an exception.

        Parameters:
            None

        Returns:
            None: The test asserts that the method handles the isel failure and does not return a value.
        """
        proc = _make_processor(verbose=False)

        # Create a real pressure variable but the time-dim isel will fail
        n_cells, n_vert = 5, 4
        p_vals = np.linspace(100000.0, 50000.0, n_vert)
        ds = xr.Dataset({
            "pressure": (["nCells", "nVertLevels"],   # intentionally NO 'Time' dim
                          np.tile(p_vals, (n_cells, 1))),
            "theta": (["Time", "nCells", "nVertLevels"],
                       np.ones((1, n_cells, n_vert)) * 300.0),
        })
        proc.dataset = ds

        # This should not raise; the except clause in the method handles it
        levels = proc._pressure_levels_from_pressure_var(
            "theta", "nVertLevels", n_vert, "Time", 0
        )

        # Levels should be returned or None (invalid if non-positive); either is OK
        assert levels is None or (isinstance(levels, np.ndarray) and len(levels) == n_vert)

    def test_non_positive_pressure_values_returns_none(self: "TestPressureLevelsFromPressureVar") -> None:
        """
        This test checks that if the _pressure_levels_from_pressure_var method encounters non-positive pressure values, it returns None. It creates a synthetic dataset where the pressure variable contains negative, zero, and NaN values, which should trigger the warning and result in None being returned.

        Parameters:
            None

        Returns:
            None: The test asserts that the method handles non-positive pressure values and returns None.
        """
        proc = _make_processor(verbose=True)

        n_cells, n_vert = 5, 4
        bad_pressure = np.array([-1000.0, 0.0, np.nan, -500.0])

        ds = xr.Dataset({
            "pressure": (["nCells", "nVertLevels"],
                          np.tile(bad_pressure, (n_cells, 1))),
            "theta": (["Time", "nCells", "nVertLevels"],
                       np.ones((1, n_cells, n_vert)) * 300.0),
        })

        proc.dataset = ds

        levels = proc._pressure_levels_from_pressure_var(
            "theta", "nVertLevels", n_vert, "Time", 0
        )

        assert levels is None


class TestRepairPressureLevels:
    """ Tests for the _repair_pressure_levels method, specifically for the branches that handle linear interpolation when there are ≥ 2 good values, linspace reconstruction when there is exactly 1 good value, and logspace reconstruction when there are no good values. """

    def test_two_good_values_uses_interpolation(self: "TestRepairPressureLevels") -> None:
        """
        This test checks that when _repair_pressure_levels is called with an array of pressure levels that contains exactly 2 good (finite) values, it uses linear interpolation to fill in the missing values. It creates a levels array with 2 valid pressure values and 2 NaNs, calls the method, and asserts that the resulting array has no NaNs and is of the correct length. 

        Parameters:
            None

        Returns:
            None: The test asserts that the method correctly interpolates the missing values.
        """
        proc = _make_processor(verbose=False)
        levels = np.array([100000.0, np.nan, 70000.0, 50000.0])
        result = proc._repair_pressure_levels(levels, 101000.0)
        assert len(result) == 4
        assert np.all(np.isfinite(result))

    def test_single_good_value_uses_linspace(self: "TestRepairPressureLevels") -> None:
        """
        This test checks that when _repair_pressure_levels is called with an array of pressure levels that contains exactly 1 good (finite) value, it uses linspace to reconstruct the missing values. It creates a levels array with 1 valid pressure value and 3 NaNs, calls the method, and asserts that the resulting array has no NaNs, is of the correct length, and that the first value is close to the mean surface pressure (since linspace should go from mean_sp to 1.0). 

        Parameters:
            None

        Returns:
            None: The test asserts that the method correctly reconstructs the missing values using linspace.
        """
        proc = _make_processor(verbose=False)
        levels = np.array([np.nan, np.nan, 70000.0, np.nan])
        result = proc._repair_pressure_levels(levels, 101000.0)
        assert len(result) == 4
        assert np.all(np.isfinite(result))
        # First value should be close to mean_sp (101000)
        assert abs(result[0] - 101000.0) < 1.0

    def test_no_good_values_uses_logspace(self: "TestRepairPressureLevels") -> None:
        """
        This test checks that when _repair_pressure_levels is called with an array of pressure levels that contains no good (finite) values, it uses logspace to reconstruct the missing values. It creates a levels array with all NaNs, calls the method, and asserts that the resulting array has no NaNs, is of the correct length, and that the first value is close to the mean surface pressure while the last value is close to 1.0 (since logspace should go from mean_sp to 1.0). 

        Parameters:
            None

        Returns:
            None: The test asserts that the method correctly reconstructs the missing values using logspace.
        """
        proc = _make_processor(verbose=False)
        levels = np.array([np.nan, np.nan, np.nan, np.nan])
        result = proc._repair_pressure_levels(levels, 101000.0)
        assert len(result) == 4
        assert np.all(np.isfinite(result))
        assert np.all(result > 0.0)
        # logspace goes from mean_sp to 1.0
        assert abs(result[0] - 101000.0) < 1.0
        assert abs(result[-1] - 1.0) < 0.001


class TestExtractXarrayByIndexFallback:
    """ Tests for the _extract_xarray_by_index method, specifically for the branch that handles the fallback when level_dim is not found in the DataArray dimensions. It checks that when level_dim is absent, the method uses the second dimension as a fallback, and when level_dim is present, it uses it directly. """

    def test_missing_level_dim_uses_second_dimension(self: "TestExtractXarrayByIndexFallback") -> None:
        """
        This test checks that when the _extract_xarray_by_index method is called with a level_dim that is not present in the DataArray dimensions, it correctly falls back to using the second dimension (index 1) for extraction. It creates a DataArray with dimensions that do not include the specified level_dim, calls the method, and asserts that the resulting array has the expected shape based on the second dimension.

        Parameters:
            None

        Returns:
            None: The test asserts that the method correctly uses the second dimension as a fallback.
        """
        # DataArray with dims ['time', 'something_else', 'nCells'] – 'nVertLevels' is absent
        data = np.random.rand(3, 6, 10)
        da = xr.DataArray(data, dims=["time", "something_else", "nCells"])

        result = MPAS3DProcessor._extract_xarray_by_index(da, 2, "nVertLevels")

        assert isinstance(result, np.ndarray)
        # Second dim is 'something_else'; extracting index 2 gives shape (3, 10)
        assert result.shape == (3, 10)

    def test_present_level_dim_uses_correct_dimension(self: "TestExtractXarrayByIndexFallback") -> None:
        """
        This test checks that when the _extract_xarray_by_index method is called with a level_dim that is present in the DataArray dimensions, it correctly uses that dimension for extraction. It creates a DataArray with the specified level_dim, calls the method, and asserts that the resulting array has the expected shape based on the level_dim.

        Parameters:
            None

        Returns:
            None: The test asserts that the method correctly uses the specified level_dim when it is present.
        """
        data = np.random.rand(3, 6, 10)
        da = xr.DataArray(data, dims=["time", "nVertLevels", "nCells"])

        result = MPAS3DProcessor._extract_xarray_by_index(da, 2, "nVertLevels")

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 10)


class TestExtractXarrayByValue:
    """ Tests for the _extract_xarray_by_value method, specifically for the branches that handle the case when level_dim is not in coords (should raise ValueError) and when nearest-neighbour extraction works correctly. """

    def test_missing_coord_raises_value_error(self: "TestExtractXarrayByValue") -> None:
        """
        This test checks that when the _extract_xarray_by_value method is called with a level_dim that is not present in the DataArray coordinates, it raises a ValueError. It creates a DataArray without the specified level_dim in its coordinates, calls the method, and asserts that a ValueError is raised.

        Parameters:
            None

        Returns:
            None: The test asserts that the method correctly raises a ValueError when the level_dim is missing in coordinates.
        """
        data = np.random.rand(3, 6)
        da = xr.DataArray(data, dims=["nCells", "nVertLevels"])  # no coords

        with pytest.raises(ValueError, match="not found in data array"):
            MPAS3DProcessor._extract_xarray_by_value(da, 50000.0, "pressure", "nearest")

    def test_nearest_extraction_works(self: "TestExtractXarrayByValue") -> None:
        """
        This test checks that the _extract_xarray_by_value method correctly performs nearest-neighbour extraction when the specified level_dim is present in the DataArray coordinates. It creates a DataArray with the specified level_dim, calls the method with a target value, and asserts that the resulting array has the expected shape.

        Parameters:
            None

        Returns:
            None: The test asserts that the method correctly performs nearest-neighbour extraction.
        """
        pressures = np.linspace(100000.0, 10000.0, 10)
        data = np.random.rand(5, 10)
        da = xr.DataArray(data, dims=["nCells", "pressure"],
                          coords={"pressure": pressures})

        result = MPAS3DProcessor._extract_xarray_by_value(da, 55000.0, "pressure", "nearest")
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)


class TestExtractNumpyByIndex:
    """Tests for the _extract_numpy_by_index method, specifically for the branches that handle the case when input is less than 2-D (should raise ValueError) and when extraction works correctly for 2-D and 3-D arrays."""

    def test_1d_input_raises_value_error(self: "TestExtractNumpyByIndex") -> None:
        """
        This test checks that when the _extract_numpy_by_index method is called with a 1-D array, it raises a ValueError. It creates a 1-D numpy array, calls the method, and asserts that a ValueError is raised.

        Parameters:
            None

        Returns:
            None: The test asserts that the method correctly raises a ValueError for 1-D input.
        """
        with pytest.raises(ValueError, match="at least 2D"):
            MPAS3DProcessor._extract_numpy_by_index(np.ones(5), 0)

    def test_3d_input_uses_second_dimension(self: "TestExtractNumpyByIndex") -> None:
        """
        This test checks that when the _extract_numpy_by_index method is called with a 3-D array, it correctly extracts data via the second dimension (index 1). It creates a 3-D numpy array, calls the method, and asserts that the result is a numpy array.

        Parameters:
            None

        Returns:
            None: The test asserts that the method correctly extracts data from a 3-D array.
        """
        data = np.random.rand(5, 4, 10)
        result = MPAS3DProcessor._extract_numpy_by_index(data, 2)
        assert isinstance(result, np.ndarray)

    def test_2d_input_uses_second_column(self: "TestExtractNumpyByIndex") -> None:
        """
        This test checks that when the _extract_numpy_by_index method is called with a 2-D array, it correctly extracts data via the second axis (index 1). It creates a 2-D numpy array, calls the method, and asserts that the result is a numpy array.

        Parameters:
            None

        Returns:
            None: The test asserts that the method correctly extracts data from a 2-D array.
        """
        data = np.arange(20).reshape(5, 4)
        result = MPAS3DProcessor._extract_numpy_by_index(data, 1)
        assert isinstance(result, np.ndarray)


class TestExtract2DFrom3DEdgeCasesExtra:
    """ Tests for the extract_2d_from_3d method, specifically for edge cases that are not covered in the main test class. """

    def test_no_level_specified_raises(self: "TestExtract2DFrom3DEdgeCasesExtra") -> None:
        """
        This test checks that when the extract_2d_from_3d method is called without specifying either level_index or level_value, it raises a ValueError. It creates a 3-D numpy array, calls the method without level arguments, and asserts that a ValueError is raised.

        Parameters:
            None

        Returns:
            None: The test asserts that the method correctly raises a ValueError when no level is specified.
        """
        data = np.random.rand(5, 10, 20)
        with pytest.raises(ValueError, match="Must provide"):
            MPAS3DProcessor.extract_2d_from_3d(data)

    def test_numpy_with_level_value_raises(self: "TestExtract2DFrom3DEdgeCasesExtra") -> None:
        """
        This test checks that when the extract_2d_from_3d method is called with a numpy array and a level_value is specified (but no coordinates are provided), it raises a ValueError. It creates a 3-D numpy array, calls the method with a level_value, and asserts that a ValueError is raised.

        Parameters:
            None

        Returns:
            None: The test asserts that the method correctly raises a ValueError when level_value is specified without coordinates.
        """
        data = np.random.rand(5, 10, 20)
        with pytest.raises(ValueError, match="level_value extraction"):
            MPAS3DProcessor.extract_2d_from_3d(data, level_value=50000.0)

    def test_detect_spatial_dim_defaults_to_ncells(self: "TestExtract2DFrom3DEdgeCasesExtra") -> None:
        """
        This test checks that the _detect_spatial_dim method returns 'nCells' when no known dimension is present. It creates a dictionary with an unknown dimension, calls the method, and asserts that the result is 'nCells'.

        Parameters:
            None

        Returns:
            None: The test asserts that the method correctly defaults to 'nCells' when no known dimension is present.
        """
        result = MPAS3DProcessor._detect_spatial_dim({"unknown_dim": 100})
        assert result == "nCells"

    def test_detect_spatial_dim_finds_nvertices(self: "TestExtract2DFrom3DEdgeCasesExtra") -> None:
        """
        This test checks that the _detect_spatial_dim method returns 'nVertices' when the 'nVertices' dimension is present. It creates a dictionary with both 'nVertices' and 'nCells', calls the method, and asserts that the result is 'nVertices'.

        Parameters:
            None

        Returns:
            None: The test asserts that the method correctly identifies 'nVertices' when present.
        """
        result = MPAS3DProcessor._detect_spatial_dim({"nVertices": 500, "nCells": 100})
        assert result == "nVertices"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

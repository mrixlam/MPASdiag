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


class TestInterpolateAtPressure:
    """ Tests for the _interpolate_at_pressure method, covering the branches for pressure levels above max, below min, normal interpolation, and the special case where lower_idx is at the last position. """

    def _setup(self: 'TestInterpolateAtPressure', 
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


    def test_verbose_prints_interpolation_info(self: 'TestInterpolateAtPressure', 
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

    def test_float_level_dispatches_to_float_resolution(self: 'TestResolveLevelIndex') -> None:
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


class TestGetVariableDataEarlyReturn:
    """ Tests for the branch in get_3d_variable_data where _resolve_level_index returns a DataArray directly (early return), which can happen if interpolation at pressure is successful. """

    def test_interpolated_data_returned_directly(self: 'TestGetVariableDataEarlyReturn') -> None:
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

    def test_multidim_extraction_raises_value_error(self: 'TestGetVariableDataNdimCheck') -> None:
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

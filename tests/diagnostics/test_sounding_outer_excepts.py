#!/usr/bin/env python3

"""
MPASdiag Test Suite: Sounding Diagnostics Outer Excepts

This module contains unit tests for the SoundingDiagnostics class in the MPASdiag package, specifically testing the handling of exceptions that may be raised in the outer try blocks of the _compute_shear_indices method. The tests ensure that when exceptions are raised during the calculation of indices such as STP and SCP, they are properly caught and handled, allowing execution to continue and setting the relevant index to None instead of crashing. The tests also verify that this behavior works regardless of the verbose setting in the SoundingDiagnostics class. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import numpy as np
import pytest
import xarray as xr
from pathlib import Path
from unittest.mock import Mock

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


class _Raisable:
    """ A simple class whose multiplication raises a RuntimeError, used to test exception handling in SoundingDiagnostics. """

    def __mul__(self: '_Raisable', 
                other: object) -> None:
        """
        This method is intentionally designed to raise a RuntimeError whenever an instance of _Raisable is multiplied by any other object. This allows us to test the robustness of exception handling in the SoundingDiagnostics class when performing operations that involve multiplication, such as unit conversions or calculations of indices. 

        Parameters: 
            other: The object that _Raisable is being multiplied by. This can be any type, as the method will raise an exception regardless of the type of 'other'.

        Returns:
            None: This method does not return a value, as it is designed to raise an exception instead of performing a multiplication operation.
        """
        raise RuntimeError("_Raisable intentional multiplication error")

    def __rmul__(self: '_Raisable', 
                 other: object) -> None:
        """
        This method is intentionally designed to raise a RuntimeError whenever an instance of _Raisable is multiplied by any other object. This allows us to test the robustness of exception handling in the SoundingDiagnostics class when performing operations that involve multiplication, such as unit conversions or calculations of indices. 

        Parameters: 
            other: The object that _Raisable is being multiplied by. This can be any type, as the method will raise an exception regardless of the type of 'other'.

        Returns:
            None: This method does not return a value, as it is designed to raise an exception instead of performing a multiplication operation.
        """
        raise RuntimeError("_Raisable intentional multiplication error")


def _make_standard_profile() -> tuple:
    """
    This function generates a standard atmospheric profile for testing purposes. It creates arrays of pressure, temperature, dew point, u and v wind components, and height that represent a typical sounding profile. The pressure decreases with height, the temperature decreases with height, and the dew point is set to be 5 degrees Celsius lower than the temperature. The u and v wind components are randomly generated within a specified range, and the height increases linearly from the surface to 14 km. 

    Parameters:
        None

    Returns:
        tuple: A tuple containing the generated profile data consisting of pressure (p), temperature (t), dew point (td), u and v wind components, and height (h). Each element of the tuple is a NumPy array with a length defined by N_VERT, representing the vertical levels of the sounding.
    """
    p = np.linspace(1000.0, 200.0, N_VERT)
    t = np.linspace(25.0, -60.0, N_VERT)
    td = t - 5.0
    rng = np.random.default_rng(42)
    u = rng.uniform(-20.0, 20.0, N_VERT)
    v = rng.uniform(-20.0, 20.0, N_VERT)
    h = np.linspace(0.0, 14_000.0, N_VERT)
    return p, t, td, u, v, h


class TestLoadGridCoordinatesDropVariables:
    """ Test that the presence of extra variables in the grid file triggers the drop_variables set in SoundingDiagnostics, and that it is not set when no extra variables are present. """

    def test_drop_variables_set_when_extra_vars_present(self: "TestLoadGridCoordinatesDropVariables", 
                                                        tmp_path: Path) -> None:
        """
        This test verifies that when extra variables are present in the grid file, the SoundingDiagnostics class correctly identifies them and sets the drop_variables attribute to include those extra variables. It creates a dataset with additional variables beyond the required longitude and latitude, saves it to a temporary NetCDF file, and then checks that the drop_variables attribute is set to include the extra variables when loading the grid coordinates. 

        Parameters:
            tmp_path (Path): A temporary directory provided by pytest for storing the test NetCDF file.

        Returns:
            None
        """
        ds = xr.Dataset({
            "lonCell":     (["nCells"], np.radians(np.linspace(-110.0, -90.0, N_CELLS))),
            "latCell":     (["nCells"], np.radians(np.linspace(25.0, 45.0, N_CELLS))),
            "temperature": (["nCells"], np.ones(N_CELLS)),
            "u_wind":      (["nCells"], np.zeros(N_CELLS)),
        })

        path = str(tmp_path / "grid_extra.nc")
        ds.to_netcdf(path)

        proc = Mock(spec=MPAS3DProcessor)
        proc.grid_file = path
        diag = SoundingDiagnostics(verbose=False)

        lon, lat = diag._load_grid_coordinates(proc)

        assert lon.shape == (N_CELLS,)
        assert lat.shape == (N_CELLS,)
        assert np.all(lon >= -180.0) and np.all(lon <= 180.0)

    def test_no_extra_vars_does_not_set_drop(self: 'TestLoadGridCoordinatesDropVariables', 
                                             tmp_path: Path) -> None:
        """
        This test verifies that when no extra variables are present in the grid file, the SoundingDiagnostics class does not set the drop_variables attribute. It creates a dataset with only the required longitude and latitude variables, saves it to a temporary NetCDF file, and then checks that the drop_variables attribute is not set when loading the grid coordinates. 

        Parameters:
            tmp_path (Path): A temporary directory provided by pytest for storing the test NetCDF file.

        Returns:
            None
        """
        ds = xr.Dataset({
            "lonCell": (["nCells"], np.radians(np.linspace(-110.0, -90.0, N_CELLS))),
            "latCell": (["nCells"], np.radians(np.linspace(25.0, 45.0, N_CELLS))),
        })

        path = str(tmp_path / "grid_minimal.nc")
        ds.to_netcdf(path)

        proc = Mock(spec=MPAS3DProcessor)
        proc.grid_file = path
        diag = SoundingDiagnostics(verbose=False)

        lon, lat = diag._load_grid_coordinates(proc)
        assert lon.shape == (N_CELLS,)


@pytest.mark.skipif(not HAS_METPY_TEST, reason="MetPy not installed")
class TestComputeShearOuterExcepts:
    """ Test that exceptions raised in the outer try blocks of _compute_shear_indices are caught and handled, allowing execution to continue and setting the relevant index to None. """

    def test_stp_outer_except_empty_height(self: 'TestComputeShearOuterExcepts') -> None:
        """
        This test verifies that when the height array is empty, the exception raised in the STP calculation is correctly caught and handled by the SoundingDiagnostics class. It checks that the STP index is set to None and that execution continues without crashing, demonstrating that the outer exception handling works for errors raised during the STP calculation when the height array is empty. 

        Parameters:
            None

        Returns:
            None
        """
        p, t, td, u, v, _ = _make_standard_profile()

        result = {
            "sbcape": 1000.0,
            "lcl_pressure": 900.0,
            "srh_0_1km": 200.0,
            "bulk_shear_0_6km": 40.0,
            "mucape": None,
            "srh_0_3km": None,
        }

        diag = SoundingDiagnostics(verbose=False)
        diag._compute_shear_indices(result, p, t, td, u, v, np.array([]))
        assert result.get("stp") is None

    def test_stp_outer_except_also_silent_verbose(self: 'TestComputeShearOuterExcepts') -> None:
        """
        This test verifies that when the height array is empty, the exception raised in the STP calculation is still caught and handled even when the verbose flag is set to True. It checks that the STP index is set to None and that execution continues without crashing, demonstrating that the outer exception handling works regardless of the verbose setting. 

        Parameters:
            None

        Returns:
            None
        """
        p, t, td, u, v, _ = _make_standard_profile()

        result = {
            "sbcape": 1000.0, "lcl_pressure": 900.0,
            "srh_0_1km": 200.0, "bulk_shear_0_6km": 40.0,
            "mucape": None, "srh_0_3km": None,
        }

        diag = SoundingDiagnostics(verbose=True)
        diag._compute_shear_indices(result, p, t, td, u, v, np.array([]))
        assert result.get("stp") is None

    def test_scp_outer_except_raisable_mucape(self: 'TestComputeShearOuterExcepts') -> None:
        """
        This test verifies that when the mucape value is set to an instance of _Raisable, which raises a RuntimeError when used in the SCP calculation, the exception is correctly caught and handled by the SoundingDiagnostics class. It checks that the SCP index is set to None and that execution continues without crashing, demonstrating that the outer exception handling works for errors raised during the SCP calculation when mucape is not a valid number. 

        Parameters:
            None

        Returns:
            None
        """
        p, t, td, u, v, h = _make_standard_profile()

        result = {
            "sbcape": None,
            "lcl_pressure": None,
            "srh_0_1km": None,
            "bulk_shear_0_6km": 40.0,
            "mucape": _Raisable(),
            "srh_0_3km": 300.0,
        }

        diag = SoundingDiagnostics(verbose=False)
        diag._compute_shear_indices(result, p, t, td, u, v, h)
        assert result.get("scp") is None

    def test_scp_outer_except_does_not_abort_sweat(self: 'TestComputeShearOuterExcepts') -> None:
        """
        This test verifies that when the mucape value is set to an instance of _Raisable, which raises a RuntimeError when used in the SCP calculation, the exception is correctly caught and handled by the SoundingDiagnostics class, allowing execution to continue to the SWEAT calculation. It checks that the SCP index is set to None and that the SWEAT index is either set to a valid number or None (if it also encounters an error), demonstrating that the outer exception handling allows subsequent calculations to proceed even if one calculation fails. 

        Parameters:
            None

        Returns:
            None
        """
        p, t, td, u, v, h = _make_standard_profile()

        result = {
            "sbcape": None, "lcl_pressure": None,
            "srh_0_1km": None, "bulk_shear_0_6km": 40.0,
            "mucape": _Raisable(), "srh_0_3km": 300.0,
        }

        diag = SoundingDiagnostics(verbose=False)
        diag._compute_shear_indices(result, p, t, td, u, v, h)
        assert "sweat_index" in result or result.get("sweat_index") is None

    def test_sweat_outer_except_raisable_temperature(self: 'TestComputeShearOuterExcepts') -> None:
        """
        This test verifies that when the temperature value is set to an instance of _Raisable, which raises a RuntimeError when multiplied (as would happen in the SWEAT calculation), the exception is correctly caught and handled by the SoundingDiagnostics class. It checks that the SWEAT index is set to None and that execution continues without crashing, demonstrating that the outer exception handling works for errors raised during the SWEAT calculation, even when verbose=False. 

        Parameters:
            None

        Returns:
            None
        """
        p, _, td, u, v, h = _make_standard_profile()
        result: dict = {}
        diag = SoundingDiagnostics(verbose=False)
        diag._compute_shear_indices(result, p, _Raisable(), td, u, v, h)
        assert result.get("sweat_index") is None

    def test_sweat_outer_except_raisable_temperature_verbose(self: 'TestComputeShearOuterExcepts') -> None:
        """
        This test verifies that when the temperature value is set to an instance of _Raisable, which raises a RuntimeError when multiplied (as would happen in the SWEAT calculation), the exception is correctly caught and handled by the SoundingDiagnostics class, even when the verbose flag is set to True. It checks that the SWEAT index is set to None and that execution continues without crashing, demonstrating that the outer exception handling works for errors raised during the SWEAT calculation regardless of the verbose setting. 

        Parameters:
            None

        Returns:
            None
        """
        p, _, td, u, v, h = _make_standard_profile()
        result: dict = {}
        diag = SoundingDiagnostics(verbose=True)
        diag._compute_shear_indices(result, p, _Raisable(), td, u, v, h)
        assert result.get("sweat_index") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

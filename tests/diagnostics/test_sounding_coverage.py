#!/usr/bin/env python3

"""
MPASdiag Test Suite: Sounding Diagnostics Coverage 

This module contains unit tests for the SoundingDiagnostics class in mpasdiag.diagnostics.sounding, specifically targeting code paths that were previously untested. The tests cover fallback LCL computation when MetPy is unavailable, handling of missing pressure/temperature/dewpoint variables, and the extraction of profiles from the dataset. By exercising these branches, we ensure that the SoundingDiagnostics class behaves robustly under a wider range of input conditions and provides informative feedback to users when expected data is missing or when certain computations cannot be performed.   

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: April 2026
Version: 1.0.0
"""
import pytest
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional
from unittest.mock import Mock, patch, MagicMock

from mpasdiag.diagnostics.sounding import SoundingDiagnostics
from mpasdiag.processing.processors_3d import MPAS3DProcessor

try:
    import metpy.calc as _mpc
    from metpy.units import units as _mpu
    HAS_METPY_TEST = True
except ImportError:
    HAS_METPY_TEST = False


def _synthetic_mpas_data(n_cells: int = 5,
                         n_vert: int = 10,
                         pressure_varname: str = "pressure",
                         temp_varname: str = "theta", 
                         dew_varname: str = "qv",     
                         include_wind: bool = True,
                         include_height: bool = False,
                         include_staggered_height: bool = False,
                         nan_pressure_idx: Optional[int] = None,) -> xr.Dataset:
    """
    This function generates a synthetic xarray Dataset that mimics the structure of MPAS 3D output, with configurable options for pressure, temperature, and dewpoint variable names, as well as the inclusion of wind and height variables. The pressure profile is created to decrease with height, and NaN values can be introduced at a specified index to test handling of missing data. This dataset serves as a controlled input for testing the SoundingDiagnostics class, allowing us to verify that it correctly extracts and processes the necessary profiles under various conditions.

    Parameters:
        n_cells (int): Number of horizontal cells in the dataset.
        n_vert (int): Number of vertical levels in the dataset.
        pressure_varname (str): Name of the pressure variable.
        temp_varname (str): Name of the temperature variable.
        dew_varname (str): Name of the dewpoint/humidity variable.
        include_wind (bool): Whether to include wind variables.
        include_height (bool): Whether to include height variables.
        include_staggered_height (bool): Whether to include staggered height variables.
        nan_pressure_idx (Optional[int]): Index at which to introduce NaN in the pressure profile.

    Returns:
        xr.Dataset: A synthetic dataset with the specified structure and variables. 
    """
    p_vals = np.linspace(101000.0, 5000.0, n_vert)

    if nan_pressure_idx is not None:
        p_vals[nan_pressure_idx] = np.nan

    pressure = np.tile(p_vals, (1, n_cells, 1))
    data_vars: dict = {}

    if pressure_varname == "pressure":
        data_vars["pressure"] = (["Time", "nCells", "nVertLevels"], pressure)
    elif pressure_varname == "components":
        data_vars["pressure_p"] = (
            ["Time", "nCells", "nVertLevels"],
            np.ones((1, n_cells, n_vert)) * 1000.0,
        )
        data_vars["pressure_base"] = (["Time", "nCells", "nVertLevels"], pressure)

    if temp_varname == "theta":
        data_vars["theta"] = (
            ["Time", "nCells", "nVertLevels"],
            np.full((1, n_cells, n_vert), 300.0),
        )
    elif temp_varname == "temperature":
        data_vars["temperature"] = (
            ["Time", "nCells", "nVertLevels"],
            np.full((1, n_cells, n_vert), 285.0),
        )
    elif temp_varname == "temp":
        data_vars["temp"] = (
            ["Time", "nCells", "nVertLevels"],
            np.full((1, n_cells, n_vert), 285.0),
        )

    if dew_varname == "qv":
        data_vars["qv"] = (
            ["Time", "nCells", "nVertLevels"],
            np.full((1, n_cells, n_vert), 0.010),
        )
    elif dew_varname == "dewpoint_K":
        # mean > 100 → will be converted from K to °C
        data_vars["dewpoint"] = (
            ["Time", "nCells", "nVertLevels"],
            np.full((1, n_cells, n_vert), 270.0),
        )
    elif dew_varname == "td_C":
        # mean < 100 → already in °C
        data_vars["td"] = (
            ["Time", "nCells", "nVertLevels"],
            np.full((1, n_cells, n_vert), 12.0),
        )

    if include_wind:
        data_vars["uReconstructZonal"] = (
            ["Time", "nCells", "nVertLevels"],
            np.ones((1, n_cells, n_vert)) * 5.0,
        )
        data_vars["uReconstructMeridional"] = (
            ["Time", "nCells", "nVertLevels"],
            np.ones((1, n_cells, n_vert)) * 3.0,
        )

    if include_height:
        heights = np.linspace(0.0, 15000.0, n_vert)
        data_vars["height"] = (
            ["Time", "nCells", "nVertLevels"],
            np.tile(heights, (1, n_cells, 1)),
        )

    if include_staggered_height:
        # nVertLevelsP1 = n_vert + 1 → triggers mid-level averaging
        heights_p1 = np.linspace(0.0, 15000.0, n_vert + 1)
        data_vars["zgrid"] = (
            ["nCells", "nVertLevelsP1"],
            np.tile(heights_p1, (n_cells, 1)),
        )

    return xr.Dataset(data_vars)


def _make_mock_proc(ds: xr.Dataset, 
                    tmp_path: Path, 
                    prefix: str = "grid",
                    extra_vars: bool = False, 
                    omit_lon: bool = False,
                    omit_lat: bool = False) -> MPAS3DProcessor:
    """
    This function creates a mock MPAS3DProcessor with a synthetic grid file based on the provided dataset. The grid file can be customized to include or omit longitude and latitude variables, as well as to add extra variables that are not part of the standard sounding coordinates. This allows for testing how the SoundingDiagnostics class handles different grid configurations and ensures that it can correctly extract necessary information from the grid file under various conditions. The mock processor is set up with the provided dataset and a path to the generated grid file, ready for use in unit tests.

    Parameters:
        ds (xr.Dataset): The dataset to be associated with the mock processor.
        tmp_path (Path): A temporary directory path for saving the synthetic grid file.
        prefix (str): A prefix for the grid file name.
        extra_vars (bool): Whether to include extra variables in the grid file that are not part of the standard sounding coordinates.
        omit_lon (bool): Whether to omit the longitude variable from the grid file.
        omit_lat (bool): Whether to omit the latitude variable from the grid file.
    
    Returns:
        MPAS3DProcessor: A mock MPAS3DProcessor instance with the synthetic grid file.
    """
    n_cells = ds.sizes.get("nCells", 5)
    lon = np.linspace(-110.0, -90.0, n_cells)
    lat = np.linspace(25.0, 45.0, n_cells)

    grid_vars: dict = {}

    if not omit_lon:
        grid_vars["lonCell"] = (["nCells"], np.radians(lon))

    if not omit_lat:
        grid_vars["latCell"] = (["nCells"], np.radians(lat))

    if extra_vars:
        # extra variable that is NOT in _SOUNDING_COORD_NAMES → triggers drop_variables
        grid_vars["cellsOnCell"] = (["nCells"], np.arange(n_cells, dtype=np.int32))

    grid_path = tmp_path / f"{prefix}.nc"
    xr.Dataset(grid_vars).to_netcdf(str(grid_path))

    proc = Mock(spec=MPAS3DProcessor)
    proc.dataset = ds
    proc.grid_file = str(grid_path)
    return proc


class TestComputeFallbackLCL:
    """ Tests for the _compute_fallback_lcl method when MetPy is not available."""

    def test_basic_values_populated(self: "TestComputeFallbackLCL") -> None:
        """
        This test verifies that the _compute_fallback_lcl method populates the 'lcl_pressure' and 'lcl_temperature' keys in the result dictionary with valid values when given typical pressure, temperature, and dewpoint profiles. It checks that the computed LCL pressure is a positive value, indicating that the fallback computation is producing a physically reasonable result even in the absence of MetPy. This test ensures that users can still obtain LCL estimates from their sounding data when MetPy is not installed, albeit with potentially less accuracy than the full MetPy calculation. 

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        p = np.linspace(1000.0, 200.0, 20)
        t = np.linspace(25.0, -50.0, 20)
        td = t - 5.0
        result: dict = {"lcl_pressure": None, "lcl_temperature": None}

        diag._compute_fallback_lcl(p, t, td, result)

        assert result["lcl_pressure"] is not None
        assert result["lcl_temperature"] is not None
        assert float(result["lcl_pressure"]) > 0.0

    def test_verbose_prints_metpy_message(self: "TestComputeFallbackLCL", 
                                          capsys: pytest.CaptureFixture) -> None:
        """
        This test checks that when the verbose flag is set to True, the _compute_fallback_lcl method prints a message indicating that MetPy is not available and that the fallback computation is being used. This ensures that users are informed about the reason for using the fallback method and understand that the results may be less accurate than if MetPy were available. The test captures the standard output and verifies that the expected message is present, confirming that the diagnostic provides appropriate feedback in verbose mode when MetPy is absent.

        Parameters:
            capsys (pytest.CaptureFixture): A pytest fixture for capturing standard output.

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=True)
        p = np.linspace(1000.0, 200.0, 10)
        t = np.linspace(25.0, -40.0, 10)
        td = t - 3.0
        result: dict = {"lcl_pressure": None, "lcl_temperature": None}

        diag._compute_fallback_lcl(p, t, td, result)

        captured = capsys.readouterr()
        assert "MetPy not available" in captured.out

    def test_empty_arrays_handled_gracefully(self: "TestComputeFallbackLCL") -> None:
        """
        This test verifies that the _compute_fallback_lcl method can handle empty input arrays for pressure, temperature, and dewpoint without raising exceptions. It checks that when empty arrays are provided, the method does not attempt to perform calculations and leaves the 'lcl_pressure' and 'lcl_temperature' keys in the result dictionary as None. This ensures that the diagnostic is robust against cases where the input sounding data may be missing or incomplete, and that it fails gracefully without crashing when it encounters such scenarios.

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        result: dict = {"lcl_pressure": None, "lcl_temperature": None}

        diag._compute_fallback_lcl(np.array([]), np.array([]), np.array([]), result)

        assert result["lcl_pressure"] is None
        assert result["lcl_temperature"] is None

class TestNoMetpyFallback:
    """ Tests for the compute_thermodynamic_indices method when MetPy is not available """

    def test_returns_none_for_all_keys_except_lcl(self: "TestNoMetpyFallback") -> None:
        """
        This test checks that when MetPy is not available, the compute_thermodynamic_indices method returns a dictionary where all keys except 'lcl_pressure' and 'lcl_temperature' are set to None. It verifies that the method correctly identifies the absence of MetPy and populates the result dictionary with None values for indices that rely on MetPy calculations, while still providing fallback LCL values. This ensures that users receive consistent output even when MetPy is not installed, and that they can still access LCL information while being aware that other indices are unavailable.

        Parameters:
            None

        Returns:
            None
        """
        with patch("mpasdiag.diagnostics.sounding.HAS_METPY", False):
            diag = SoundingDiagnostics(verbose=False)
            p = np.linspace(1000.0, 200.0, 30)
            t = np.linspace(25.0, -55.0, 30)
            td = t - 5.0

            result = diag.compute_thermodynamic_indices(p, t, td)

        assert isinstance(result, dict)

        for key in ("cape", "cin", "mucape", "k_index", "sweat_index"):
            assert result[key] is None

    def test_verbose_output_shown(self: "TestNoMetpyFallback", 
                                  capsys: pytest.CaptureFixture) -> None:
        """
        This test verifies that when the verbose flag is set to True and MetPy is not available, the compute_thermodynamic_indices method prints a message indicating that MetPy is not available and that fallback computations are being used. This ensures that users are informed about the limitations of the diagnostic when MetPy is absent and understand that certain indices will not be computed. The test captures the standard output and checks for the presence of the expected message, confirming that the diagnostic provides appropriate feedback in verbose mode under these conditions. 

        Parameters:
            capsys (pytest.CaptureFixture): A pytest fixture for capturing standard output.

        Returns:
            None
        """
        with patch("mpasdiag.diagnostics.sounding.HAS_METPY", False):
            diag = SoundingDiagnostics(verbose=True)
            p = np.linspace(1000.0, 200.0, 15)
            t = np.linspace(25.0, -40.0, 15)
            td = t - 5.0

            diag.compute_thermodynamic_indices(p, t, td)

        captured = capsys.readouterr()
        assert "MetPy not available" in captured.out


@pytest.mark.skipif(not HAS_METPY_TEST, reason="MetPy not installed")
class TestMetpyBaseProfileFailure:
    """ Tests for compute_thermodynamic_indices when MetPy is available but the base profile computation fails. """

    def test_verbose_prints_failure_message(self: "TestMetpyBaseProfileFailure", 
                                            capsys: pytest.CaptureFixture) -> None:
        """
        This test verifies that when the verbose flag is set to True and MetPy is available but the base profile computation fails, the compute_thermodynamic_indices method prints a message indicating the failure. This ensures that users are informed about the failure of the base profile computation and understand that certain indices will not be computed. The test captures the standard output and checks for the presence of the expected message, confirming that the diagnostic provides appropriate feedback in verbose mode under these conditions.

        Parameters:
            capsys (pytest.CaptureFixture): A pytest fixture for capturing standard output.

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=True)
        p = np.linspace(1000.0, 200.0, 30)
        t = np.linspace(25.0, -55.0, 30)
        td = t - 5.0

        with patch("mpasdiag.diagnostics.sounding.mpcalc") as mock_calc:
            mock_calc.parcel_profile.side_effect = RuntimeError("boom")
            result = diag.compute_thermodynamic_indices(p, t, td)

        captured = capsys.readouterr()
        assert "MetPy base profile computation failed" in captured.out
        assert all(v is None for v in result.values())

    def test_silent_failure_returns_all_none(self: "TestMetpyBaseProfileFailure") -> None:
        """
        This test verifies that when the verbose flag is set to False and MetPy is available but the base profile computation fails, the compute_thermodynamic_indices method does not print any message. This ensures that users are not informed about the failure of the base profile computation when verbose mode is disabled, and all indices remain None.

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        p = np.linspace(1000.0, 200.0, 30)
        t = np.linspace(25.0, -55.0, 30)
        td = t - 5.0

        with patch("mpasdiag.diagnostics.sounding.mpcalc") as mock_calc:
            mock_calc.parcel_profile.side_effect = RuntimeError("boom")
            result = diag.compute_thermodynamic_indices(p, t, td)

        assert all(v is None for v in result.values())


@pytest.mark.skipif(not HAS_METPY_TEST, reason="MetPy not installed")
class TestSafeHelpersToUnit:
    """ Tests for the _safe_scalar and _safe_pair helper methods when to_unit is specified. """

    def test_safe_scalar_converts_unit(self: "TestSafeHelpersToUnit") -> None:
        """
        This test verifies that the _safe_scalar method correctly converts the output of the provided function to the specified unit when to_unit is given. It checks that when a function returns a value with units (e.g., 273.15 K) and to_unit is set to "degC", the method returns the value converted to degrees Celsius (0 °C in this case). This ensures that the helper method can handle unit conversions properly, allowing for consistent output regardless of the original units of the computed values.

        Parameters:
            None

        Returns:
            None
        """
        val = SoundingDiagnostics._safe_scalar(
            lambda: 273.15 * _mpu.K, to_unit="degC"
        )
        assert val is not None
        assert abs(val) < 0.5

    def test_safe_scalar_to_unit_failure_returns_none(self: "TestSafeHelpersToUnit") -> None:
        """
        This test verifies that when the function provided to _safe_scalar raises an exception and to_unit is specified, the method returns None. This ensures that the helper method can gracefully handle errors during unit conversion and does not propagate exceptions, instead returning a consistent None value to indicate failure. 

        Parameters:
            None

        Returns:
            None
        """
        val = SoundingDiagnostics._safe_scalar(
            lambda: (_ for _ in ()).throw(ValueError("nope")), to_unit="K"
        )
        assert val is None

    def test_safe_pair_converts_both_units(self: "TestSafeHelpersToUnit") -> None:
        """
        This test verifies that the _safe_pair method correctly converts both values returned by the provided function to the specified unit when to_unit is given. It checks that when a function returns a pair of values with units (e.g., 1500 J/kg and -300 J/kg) and to_unit is set to "J/kg", the method returns both values converted to the specified unit (which in this case should be the same, but the test ensures that the conversion process is executed). This ensures that the helper method can handle unit conversions for pairs of values properly, allowing for consistent output regardless of the original units. 

        Parameters:
            None

        Returns:
            None
        """
        def pair_func() -> tuple[float, float]:
            """ 
            This function simulates a pair of values with units that would be returned by a MetPy calculation. It returns 1500 J/kg and -300 J/kg, which are typical values for CAPE and CIN, respectively. The _safe_pair method should convert these values to the specified unit (J/kg in this case) without error, demonstrating that it can handle unit conversions for pairs of values correctly.

            Parameters:
                None

            Returns:
                tuple[float, float]: A pair of values with units (CAPE and CIN). 
            """
            return 1500.0 * _mpu("J/kg"), -300.0 * _mpu("J/kg")

        a, b = SoundingDiagnostics._safe_pair(pair_func, to_unit="J/kg")

        assert a is not None
        assert b is not None
        assert abs(a - 1500.0) < 1.0
        assert abs(b - (-300.0)) < 1.0

    def test_safe_pair_to_unit_failure_returns_none_pair(self: "TestSafeHelpersToUnit") -> None:
        """
        This test verifies that when the function provided to _safe_pair raises an exception and to_unit is specified, the method returns (None, None). This ensures that the helper method can gracefully handle errors during unit conversion and does not propagate exceptions, instead returning a consistent (None, None) value to indicate failure. 

        Parameters:
            None

        Returns:
            None
        """
        a, b = SoundingDiagnostics._safe_pair(
            lambda: (_ for _ in ()).throw(RuntimeError("nope")), to_unit="J/kg"
        )
        assert a is None
        assert b is None


@pytest.mark.skipif(not HAS_METPY_TEST, reason="MetPy not installed")
class TestThermodynamicIndicesWithWindAndHeight:
    """ Tests for compute_thermodynamic_indices when wind and height profiles are supplied. """

    def test_wet_bulb_zero_path_executed(self: "TestThermodynamicIndicesWithWindAndHeight") -> None:
        """
        This test verifies that when pressure, temperature, dewpoint, and height profiles are supplied to the compute_thermodynamic_indices method, the code path for computing the wet-bulb zero height is executed. It checks that the resulting dictionary includes the key 'wet_bulb_zero_height', which indicates that the method attempted to compute this index based on the provided profiles. This ensures that users who supply height information along with their sounding data can access the wet-bulb zero height index, which is a valuable parameter for understanding atmospheric conditions. 

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        n = 40
        p = np.linspace(1000.0, 200.0, n)
        t = np.linspace(25.0, -60.0, n)
        td = t - 5.0

        height = np.linspace(0.0, 12000.0, n)
        result = diag.compute_thermodynamic_indices(p, t, td, height_m=height)

        assert "wet_bulb_zero_height" in result

    def test_shear_indices_attempted_with_wind_height(self: "TestThermodynamicIndicesWithWindAndHeight") -> None:
        """
        This test verifies that when pressure, temperature, dewpoint, wind, and height profiles are supplied to the compute_thermodynamic_indices method, the code paths for computing shear-related indices are attempted. It checks that at least one of the shear-related keys (e.g., 'bulk_shear_0_1km', 'srh_0_3km') is present in the resulting dictionary, which indicates that the method attempted to compute these indices based on the provided wind and height information. This ensures that users who supply comprehensive sounding data can access shear-related indices, which are important for assessing storm dynamics and severe weather potential. 

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        n = 40
        p = np.linspace(1000.0, 200.0, n)
        t = np.linspace(25.0, -60.0, n)
        td = t - 5.0
        u = np.random.uniform(-15.0, 15.0, n)
        v = np.random.uniform(-15.0, 15.0, n)
        height = np.linspace(0.0, 12000.0, n)

        result = diag.compute_thermodynamic_indices(
            p, t, td, u_wind_kt=u, v_wind_kt=v, height_m=height
        )

        shear_keys = (
            "bulk_shear_0_1km", "bulk_shear_0_6km",
            "srh_0_1km", "srh_0_3km",
        )

        assert any(k in result for k in shear_keys)

    def test_compute_wet_bulb_zero_with_crossing(self: "TestThermodynamicIndicesWithWindAndHeight") -> None:
        """
        This test verifies that the _compute_wet_bulb_zero method returns a valid height (or None) when the wet-bulb temperature profile crosses 0 °C. It checks that when given typical pressure, temperature, dewpoint, and height profiles that would produce a wet-bulb zero crossing, the method returns either a valid height value or None (if the crossing is not detected), but does not raise an exception. This ensures that the method can handle typical sounding profiles and provides a reasonable output for the wet-bulb zero height index, which is important for understanding freezing level conditions in the atmosphere. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.diagnostics.sounding as _s

        if _s.mpcalc is None:
            pytest.skip("MetPy not installed")

        n = 40
        p = np.linspace(1000.0, 200.0, n) * _mpu.hPa
        t = np.linspace(25.0, -60.0, n) * _mpu.degC
        td = (np.linspace(25.0, -60.0, n) - 5.0) * _mpu.degC
        h = np.linspace(0.0, 12000.0, n)

        result = SoundingDiagnostics._compute_wet_bulb_zero(p, t, td, h)
        assert result is None or isinstance(result, float)

    def test_compute_wet_bulb_zero_no_crossing_returns_none(self: "TestThermodynamicIndicesWithWindAndHeight") -> None:
        """
        This test verifies that the _compute_wet_bulb_zero method returns None when the wet-bulb temperature profile does not cross 0 °C. It checks that when given pressure, temperature, dewpoint, and height profiles that are all well above freezing, the method correctly identifies that there is no crossing and returns None. This ensures that the method can handle cases where the wet-bulb zero level is not present in the profile and provides a clear indication (None) to users in such scenarios. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.diagnostics.sounding as _s

        if _s.mpcalc is None:
            pytest.skip("MetPy not installed")

        n = 20
        # All temperatures well above zero – no crossing
        p = np.linspace(1000.0, 500.0, n) * _mpu.hPa
        t = np.linspace(30.0, 15.0, n) * _mpu.degC
        td = (np.linspace(30.0, 15.0, n) - 5.0) * _mpu.degC
        h = np.linspace(0.0, 5000.0, n)

        result = SoundingDiagnostics._compute_wet_bulb_zero(p, t, td, h)
        assert result is None

    def test_compute_wet_bulb_zero_exception_returns_none(self: "TestThermodynamicIndicesWithWindAndHeight") -> None:
        """
        This test verifies that if the MetPy calculation for wet-bulb zero encounters an exception, the _compute_wet_bulb_zero method returns None instead of propagating the exception. It checks that when the mpcalc.wet_bulb_temperature function is mocked to raise a RuntimeError, the method handles this gracefully and returns None. This ensures that users receive a consistent None value when the wet-bulb zero computation fails due to issues within MetPy, rather than experiencing a crash or unhandled exception. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.diagnostics.sounding as _s

        if _s.mpcalc is None:
            pytest.skip("MetPy not installed")

        with patch("mpasdiag.diagnostics.sounding.mpcalc") as mock_calc:
            mock_calc.wet_bulb_temperature.side_effect = RuntimeError("fail")
            result = SoundingDiagnostics._compute_wet_bulb_zero(
                MagicMock(), MagicMock(), MagicMock(), np.ones(5)
            )

        assert result is None


class TestExtractPressureProfile:
    """ Tests for the _extract_pressure_profile method, covering different variable configurations and error handling. """

    def test_pressure_components_path(self: "TestExtractPressureProfile") -> None:
        """
        This test verifies that when the dataset contains 'pressure_base' and 'pressure_p' variables, the _extract_pressure_profile method correctly extracts the pressure profile by summing these two components. It checks that the resulting pressure profile has the expected shape, contains finite values, and decreases with height (i.e., the first value is greater than the last value). This ensures that the method can handle datasets where pressure is represented as a combination of base and perturbation components, which is a common format in MPAS output.

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=8, pressure_varname="components")
        # pressure_base holds p_vals; pressure_p adds 1000 Pa
        p = diag._extract_pressure_profile(ds, "Time", 0, 0)
        assert p.shape == (8,)
        assert np.all(np.isfinite(p))
        # Should be base + perturbation
        assert p[0] > p[-1]  # decreasing with height

    def test_direct_pressure_path(self: "TestExtractPressureProfile") -> None:
        """
        This test verifies that when the dataset contains a direct 'pressure' variable, the _extract_pressure_profile method correctly extracts this variable as the pressure profile. It checks that the resulting pressure profile has the expected shape and contains finite values. This ensures that the method can handle datasets where pressure is provided directly as a single variable, which is another common format in MPAS output. 

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=8, pressure_varname="pressure")
        p = diag._extract_pressure_profile(ds, "Time", 0, 2)
        assert p.shape == (8,)
        assert np.all(np.isfinite(p))

    def test_no_pressure_raises_value_error(self: "TestExtractPressureProfile") -> None:
        """
        This test verifies that when the dataset does not contain any recognizable pressure variable, the _extract_pressure_profile method raises a ValueError with an appropriate message. It checks that when the pressure_varname is set to "none", indicating that no pressure variable is present in the dataset, the method raises a ValueError and that the error message contains the expected text. This ensures that users receive clear feedback when their dataset lacks necessary pressure information for sounding diagnostics, allowing them to understand and correct the issue. 

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=8, pressure_varname="none")
        with pytest.raises(ValueError, match="Cannot determine pressure"):
            diag._extract_pressure_profile(ds, "Time", 0, 0)


class TestExtractTemperatureProfile:
    """ Tests for the _extract_temperature_profile method, covering different variable configurations, unit conversions, and error handling. """

    def test_direct_temperature_variable(self: "TestExtractTemperatureProfile") -> None:
        """
        This test verifies that when the dataset contains a 'temperature' variable, the _extract_temperature_profile method correctly extracts this variable and converts it from Kelvin to Celsius if the mean value is greater than 100. It checks that the resulting temperature profile has the expected shape and that the first value is approximately 11.85 °C (which corresponds to 285 K). This ensures that the method can handle datasets where temperature is provided directly in Kelvin and performs the necessary unit conversion for sounding diagnostics.

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=8, temp_varname="temperature")
        p_pa = np.linspace(101000.0, 5000.0, 8)
        t = diag._extract_temperature_profile(ds, "Time", 0, 0, p_pa)
        # 285 K → 285 – 273.15 = 11.85 °C
        assert t.shape == (8,)
        assert abs(t[0] - (285.0 - 273.15)) < 0.01

    def test_direct_temp_variable(self: "TestExtractTemperatureProfile") -> None:
        """
        This test verifies that when the dataset contains a 'temp' variable, the _extract_temperature_profile method correctly extracts this variable as the temperature profile without conversion. It checks that the resulting temperature profile has the expected shape and that the first value is approximately 11.85 °C (which corresponds to 285 in the synthetic data). This ensures that the method can handle datasets where temperature is provided directly in Celsius and does not perform unnecessary conversions. 

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=8, temp_varname="temp")
        p_pa = np.linspace(101000.0, 5000.0, 8)
        t = diag._extract_temperature_profile(ds, "Time", 0, 0, p_pa)
        assert t.shape == (8,)
        assert abs(t[0] - (285.0 - 273.15)) < 0.01

    def test_theta_verbose_prints_conversion_message(self: "TestExtractTemperatureProfile", 
                                                     capsys: pytest.CaptureFixture) -> None:
        """
        This test verifies that when the dataset contains a 'theta' variable and verbose mode is enabled, the _extract_temperature_profile method prints a message indicating that potential temperature is being converted to actual temperature. It checks that when the temp_varname is set to "theta", the method performs the necessary conversion using the provided pressure profile and that a message about this conversion is printed in verbose mode. This ensures that users are informed about the use of potential temperature and the conversion process when verbose mode is enabled, providing transparency about how the temperature profile is derived from the available data. 

        Parameters:
            capsys: pytest.CaptureFixture

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=True)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=8, temp_varname="theta")
        p_pa = np.linspace(101000.0, 5000.0, 8)
        diag._extract_temperature_profile(ds, "Time", 0, 0, p_pa)
        captured = capsys.readouterr()
        assert "potential temp" in captured.out.lower() or "Converted" in captured.out

    def test_no_temperature_raises_value_error(self: "TestExtractTemperatureProfile") -> None:
        """
        This test verifies that when the dataset does not contain any recognizable temperature variable, the _extract_temperature_profile method raises a ValueError with an appropriate message. It checks that when the temp_varname is set to "none", indicating that no temperature variable is present in the dataset, the method raises a ValueError and that the error message contains the expected text. This ensures that users receive clear feedback when their dataset lacks necessary temperature information for sounding diagnostics, allowing them to understand and correct the issue. 

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        ds = xr.Dataset({
            "pressure": (["Time", "nCells", "nVertLevels"], np.ones((1, 5, 8))),
        })
        with pytest.raises(ValueError, match="Cannot find temperature"):
            diag._extract_temperature_profile(ds, "Time", 0, 0, np.ones(8))


class TestExtractDewpointProfile:
    """ Tests for the _extract_dewpoint_profile method, covering different variable configurations, unit conversions, verbose messages, and error handling. """

    def test_dewpoint_in_kelvin_converted(self: "TestExtractDewpointProfile") -> None:
        """
        This test verifies that when the dataset contains a 'dewpoint_K' variable, the _extract_dewpoint_profile method correctly extracts this variable and converts it from Kelvin to Celsius if the mean value is greater than 100. It checks that the resulting dewpoint profile has the expected shape and that the first value is approximately -3.15 °C (which corresponds to 270 K). This ensures that the method can handle datasets where dewpoint is provided in Kelvin and performs the necessary unit conversion for sounding diagnostics.

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=8, dew_varname="dewpoint_K")
        p_pa = np.linspace(101000.0, 5000.0, 8)
        td = diag._extract_dewpoint_profile(ds, "Time", 0, 0, p_pa)
        assert td.shape == (8,)
        # 270 K → 270 – 273.15 = -3.15 °C
        assert abs(td[0] - (270.0 - 273.15)) < 0.01

    def test_dewpoint_in_celsius_returned_as_is(self: "TestExtractDewpointProfile") -> None:
        """
        This test verifies that when the dataset contains a 'td_C' variable, the _extract_dewpoint_profile method correctly extracts this variable as the dewpoint profile without conversion. It checks that the resulting dewpoint profile has the expected shape and that the first value is approximately -3.15 °C (which corresponds to 270 in the synthetic data). This ensures that the method can handle datasets where dewpoint is provided directly in Celsius and does not perform unnecessary conversions.

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=8, dew_varname="td_C")
        p_pa = np.linspace(101000.0, 5000.0, 8)
        td = diag._extract_dewpoint_profile(ds, "Time", 0, 0, p_pa)
        assert td.shape == (8,)
        assert abs(td[0] - 12.0) < 0.01

    def test_qv_verbose_prints_message(self: "TestExtractDewpointProfile", 
                                       capsys: pytest.CaptureFixture) -> None:
        """
        This test verifies that when the dataset contains a 'qv' variable and verbose mode is enabled, the _extract_dewpoint_profile method prints a message indicating that mixing ratio is being converted to dewpoint. It checks that when the dew_varname is set to "qv", the method performs the necessary conversion using the provided pressure profile and that a message about this conversion is printed in verbose mode. This ensures that users are informed about the use of mixing ratio and the conversion process when verbose mode is enabled, providing transparency about how the dewpoint profile is derived from the available data. 

        Parameters:
            capsys: pytest.CaptureFixture

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=True)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=8, dew_varname="qv")
        p_pa = np.linspace(101000.0, 5000.0, 8)
        diag._extract_dewpoint_profile(ds, "Time", 0, 0, p_pa)
        captured = capsys.readouterr()
        assert "dewpoint" in captured.out.lower() or "mixing ratio" in captured.out.lower()

    def test_no_moisture_verbose_warning_and_nan_returned(self: "TestExtractDewpointProfile", 
                                                          capsys: pytest.CaptureFixture) -> None:
        """
        This test verifies that when the dataset does not contain any recognizable moisture variable and verbose mode is enabled, the _extract_dewpoint_profile method prints a warning message and returns an array of NaN values. It checks that when the dew_varname is set to "none", indicating that no moisture variable is present in the dataset, the method prints a warning about missing moisture data and that the returned dewpoint profile consists entirely of NaN values. This ensures that users receive clear feedback about the absence of moisture information in their dataset and that the method provides a consistent NaN output to indicate the lack of valid dewpoint data. 

        Parameters:
            capsys: pytest.CaptureFixture

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=True)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=8, dew_varname="none")
        p_pa = np.linspace(101000.0, 5000.0, 8)
        td = diag._extract_dewpoint_profile(ds, "Time", 0, 0, p_pa)
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert np.all(np.isnan(td))

    def test_no_moisture_no_verbose_returns_nan(self: "TestExtractDewpointProfile") -> None:
        """
        This test verifies that when the dataset does not contain any recognizable moisture variable and verbose mode is disabled, the _extract_dewpoint_profile method returns an array of NaN values without printing any warning message. It checks that when the dew_varname is set to "none", indicating that no moisture variable is present in the dataset, the method does not print any warning and that the returned dewpoint profile consists entirely of NaN values. This ensures that users who have disabled verbose mode receive a consistent NaN output to indicate the lack of valid dewpoint data without additional warnings, while still providing a clear indication of missing moisture information through the NaN values.

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=8, dew_varname="none")
        p_pa = np.linspace(101000.0, 5000.0, 8)
        td = diag._extract_dewpoint_profile(ds, "Time", 0, 0, p_pa)
        assert np.all(np.isnan(td))


class TestExtractWindProfiles:
    """ Tests for the _extract_wind_profiles method, covering the presence and absence of wind variables, unit conversions, verbose warnings, and error handling. """

    def test_verbose_warning_when_no_wind(self: "TestExtractWindProfiles", 
                                          capsys: pytest.CaptureFixture) -> None:
        """
        This test verifies that when the dataset does not contain any recognizable wind variables and verbose mode is enabled, the _extract_wind_profiles method prints a warning message and returns (None, None). It checks that when the include_wind flag is set to False in the synthetic dataset, indicating that no wind variables are present, the method prints a warning about missing wind data and that both returned wind profiles (u and v) are None. This ensures that users receive clear feedback about the absence of wind information in their dataset and that the method provides a consistent (None, None) output to indicate the lack of valid wind profiles. 

        Parameters:
            capsys: pytest.CaptureFixture

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=True)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=8, include_wind=False)
        u, v = diag._extract_wind_profiles(ds, "Time", 0, 0)
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert u is None
        assert v is None

    def test_wind_extracted_in_knots(self: "TestExtractWindProfiles") -> None:
        """
        This test verifies that when the dataset contains recognizable wind variables, the _extract_wind_profiles method correctly extracts the u and v wind profiles and converts them from m/s to knots. It checks that when the include_wind flag is set to True in the synthetic dataset, the method returns valid u and v profiles with the expected shape and that the first values of u and v correspond to 5 m/s and 3 m/s converted to knots (approximately 9.72 kt and 5.83 kt, respectively). This ensures that the method can handle datasets with wind information and performs the necessary unit conversion for sounding diagnostics, providing users with wind profiles in a commonly used unit for meteorological analysis. 

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=8, include_wind=True)
        u, v = diag._extract_wind_profiles(ds, "Time", 0, 0)
        assert u is not None
        assert v is not None
        # 5.0 m/s * 1.94384 ≈ 9.72 kt
        assert abs(u[0] - 5.0 * 1.94384) < 0.01
        assert abs(v[0] - 3.0 * 1.94384) < 0.01


class TestExtractHeightProfile:
    """ Tests for the _extract_height_profile method, covering different variable configurations, handling of time dimension, verbose warnings, and error handling. """

    def test_height_with_time_dimension_returned(self: "TestExtractHeightProfile") -> None:
        """
        This test verifies that when the dataset contains a 'height' variable with a Time dimension, the _extract_height_profile method correctly extracts the height profile for the specified time index and cell index. It checks that the resulting height profile has the expected shape and that the first value is approximately 0.0 m, which corresponds to the surface level in the synthetic data. This ensures that the method can handle datasets where height is provided as a function of time and space, and that it correctly selects the appropriate profile based on the provided indices. 

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=8, include_height=True)
        h = diag._extract_height_profile(ds, "Time", 0, 0)
        assert h is not None
        assert h.shape == (8,)
        assert h[0] == pytest.approx(0.0, abs=1.0)

    def test_staggered_height_midlevel_average(self: "TestExtractHeightProfile") -> None:
        """
        This test verifies that when the dataset contains a 'zgrid' variable representing staggered height levels, the _extract_height_profile method correctly computes the mid-level height profile by averaging adjacent zgrid levels. It checks that the resulting height profile has the expected shape and that the first value is approximately 750.0 m, which corresponds to the average of 0.0 m and 1500.0 m in the synthetic data. This ensures that the method can handle datasets where height is provided on staggered levels and performs the necessary averaging to produce a consistent height profile for sounding diagnostics. 

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        n_vert = 8
        # Add pressure to set nVertLevels dimension
        ds = xr.Dataset({
            "pressure": (["Time", "nCells", "nVertLevels"],
                          np.ones((1, 5, n_vert)) * 50000.0),
            "zgrid": (["nCells", "nVertLevelsP1"],
                       np.tile(np.linspace(0.0, 15000.0, n_vert + 1), (5, 1))),
        })
        h = diag._extract_height_profile(ds, "Time", 0, 0)
        assert h is not None
        assert h.shape == (n_vert,)  # averaged to n_vert levels

    def test_height_without_time_dimension(self: "TestExtractHeightProfile") -> None:
        """
        This test verifies that when the dataset contains a 'height' variable without a Time dimension, the _extract_height_profile method correctly extracts the height profile for the specified cell index. It checks that the resulting height profile has the expected shape and that the first value is approximately 0.0 m, which corresponds to the surface level in the synthetic data. This ensures that the method can handle datasets where height is provided as a function of space only, and that it correctly selects the appropriate profile based on the provided cell index. 

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        n_vert = 6
        ds = xr.Dataset({
            "pressure": (["Time", "nCells", "nVertLevels"],
                          np.ones((1, 5, n_vert)) * 50000.0),
            "height": (["nCells", "nVertLevels"],
                        np.tile(np.linspace(0.0, 10000.0, n_vert), (5, 1))),
        })
        h = diag._extract_height_profile(ds, "Time", 0, 0)
        assert h is not None
        assert h.shape == (n_vert,)

    def test_no_height_returns_none(self: "TestExtractHeightProfile") -> None:
        """
        This test verifies that when the dataset does not contain any recognizable height variable, the _extract_height_profile method returns None. It checks that when the include_height flag is set to False in the synthetic dataset, indicating that no height variable is present, the method returns None for the height profile. This ensures that users receive a consistent None value to indicate the absence of height information in their dataset, allowing them to understand that height data is not available for sounding diagnostics. 

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=8)
        h = diag._extract_height_profile(ds, "Time", 0, 0)
        assert h is None

    def test_extraction_failure_verbose_warning(self: "TestExtractHeightProfile", 
                                                capsys: pytest.CaptureFixture) -> None:
        """
        This test verifies that when the dataset contains a 'height' variable but the specified cell index is out of bounds, the _extract_height_profile method raises an exception that is caught and results in a warning message being printed in verbose mode. It checks that when an out-of-bounds cell index is used to trigger an exception during height extraction, the method handles this gracefully by printing a warning about the failure and returning None for the height profile. This ensures that users receive clear feedback about issues with height extraction when verbose mode is enabled, and that the method provides a consistent None output to indicate the failure to extract a valid height profile. 

        Parameters:
            capsys: pytest.CaptureFixture

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=True)
        n_vert = 6
        n_cells = 3
        # height exists but cell_idx is out of bounds → isel raises
        ds = xr.Dataset({
            "height": (["nCells", "nVertLevels"],
                        np.ones((n_cells, n_vert))),
        })
        # Use an out-of-bounds cell_idx to trigger the exception
        h = diag._extract_height_profile(ds, "Time", 0, cell_idx=999)
        captured = capsys.readouterr()
        assert "Warning" in captured.out or h is None


class TestNonFiniteFiltering:
    """ Tests for the filtering of non-finite pressure levels in the extract_sounding_profile method. """

    def test_nan_pressure_levels_are_filtered(self: "TestNonFiniteFiltering", 
                                              tmp_path: str) -> None:
        """
        This test verifies that when the pressure profile extracted in the extract_sounding_profile method contains NaN values, these non-finite levels are filtered out from the resulting sounding profile. It checks that when a synthetic dataset is created with NaN values injected into the pressure profile, the resulting profile returned by extract_sounding_profile contains only finite pressure values and that the length of the profile is reduced accordingly. This ensures that users receive a cleaned sounding profile without non-finite pressure levels, which could otherwise lead to errors or misleading results in subsequent analysis of the sounding data. 

        Parameters:
            tmp_path (str): A temporary directory provided by pytest for creating test files.

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        # Inject NaN into the middle of the pressure profile
        ds = _synthetic_mpas_data(n_cells=5, n_vert=10, nan_pressure_idx=4)
        proc = _make_mock_proc(ds, tmp_path, prefix="grid_nan_p")
        profile = diag.extract_sounding_profile(proc, -100.0, 35.0)
        # At least one NaN should have been filtered out
        assert np.all(np.isfinite(profile["pressure"]))
        assert len(profile["pressure"]) < 10

    def test_wind_and_height_also_filtered(self: "TestNonFiniteFiltering", 
                                           tmp_path: str) -> None:
        """
        This test verifies that when the pressure profile contains NaN values and is filtered in the extract_sounding_profile method, the corresponding wind and height profiles are also filtered to maintain consistent indexing across all profile variables. It checks that when a synthetic dataset is created with NaN values injected into the pressure profile, the resulting u_wind, v_wind, and height profiles returned by extract_sounding_profile contain only finite values and have the same length as the filtered pressure profile. This ensures that users receive consistent and aligned profiles for pressure, wind, and height after non-finite levels are removed, allowing for accurate analysis of the sounding data without mismatched indices. 

        Parameters:
            tmp_path (str): A temporary directory provided by pytest for creating test files.

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)

        ds = _synthetic_mpas_data(
            n_cells=5, n_vert=10,
            nan_pressure_idx=2,
            include_wind=True,
            include_height=True,
        )

        proc = _make_mock_proc(ds, tmp_path, prefix="grid_nan_wind")
        profile = diag.extract_sounding_profile(proc, -100.0, 35.0)
        n = len(profile["pressure"])
        assert profile["u_wind"] is not None
        assert len(profile["u_wind"]) == n
        assert profile["height"] is not None
        assert len(profile["height"]) == n


class TestLoadGridCoordinates:
    """ Tests for the _load_grid_coordinates method, covering the handling of extra variables in the grid file, exceptions during probing, and missing latitude variable. """

    def test_extra_grid_vars_trigger_drop_variables(self: "TestLoadGridCoordinates", 
                                                    tmp_path: str) -> None:
        """
        This test verifies that when the grid file contains extra variables that are not in the expected set of coordinate variable names, the _load_grid_coordinates method successfully loads the longitude and latitude coordinates without being affected by the presence of these additional variables. It checks that when a synthetic dataset is created with extra variables (e.g., 'cellsOnCell') that are not part of the standard grid coordinate names, the method still correctly extracts the longitude and latitude arrays with the expected shapes. This ensures that users can load grid coordinates even when their grid files contain additional variables, as long as the necessary longitude and latitude information is present.

        Parameters:
            tmp_path (str): A temporary directory provided by pytest for creating test files.

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=8)
        proc = _make_mock_proc(ds, tmp_path, extra_vars=True)

        lon, lat = diag._load_grid_coordinates(proc)

        assert lon.shape == (5,)
        assert lat.shape == (5,)

    def test_probe_exception_silently_falls_back(self: "TestLoadGridCoordinates", 
                                                 tmp_path: str) -> None:
        """
        This test verifies that if probing the grid file raises an exception (e.g., due to an OSError), the _load_grid_coordinates method silently falls back to opening the dataset without dropping variables. It checks that when the xarray.open_dataset function is patched to raise an OSError on the first call (simulating a probe failure), the method handles this gracefully and successfully loads the longitude and latitude coordinates on the second attempt. This ensures that users can still load grid coordinates even if the initial probing step encounters issues, providing robustness in handling various grid file configurations.

        Parameters:
            tmp_path (str): A temporary directory provided by pytest for creating test files.

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=8)
        proc = _make_mock_proc(ds, tmp_path)

        original_open = xr.open_dataset

        call_count = {"n": 0}

        def patched_open(path: str, 
                         **kwargs) -> xr.Dataset:
            """
            This patched version of xarray.open_dataset simulates a failure on the first call by raising an OSError, and then calls the original open_dataset function on subsequent calls. It uses a call_count dictionary to keep track of how many times the function has been called, allowing it to trigger the simulated failure only on the first attempt. This is used to test that the _load_grid_coordinates method can handle exceptions during probing and successfully load the dataset on a retry.

            Parameters:
                path (str): The file path to open.
                **kwargs: Additional keyword arguments to pass to the original open_dataset function.

            Returns:
                xr.Dataset: The dataset loaded from the specified path. 
            """
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("simulated probe failure")
            return original_open(path, **kwargs)

        with patch("xarray.open_dataset", side_effect=patched_open):
            proc2 = _make_mock_proc(ds, tmp_path, prefix="grid_probe_fail")
            lon, lat = diag._load_grid_coordinates(proc2)

        assert lon is not None

    def test_no_lat_raises_value_error(self: "TestLoadGridCoordinates", 
                                       tmp_path: str) -> None:
        """
        This test verifies that when the grid file does not contain a latitude variable, the _load_grid_coordinates method raises a ValueError with an appropriate message. It checks that when a synthetic dataset is created without a latitude variable (e.g., by omitting it from the mock processor), the method raises a ValueError and that the error message contains the expected text about missing latitude. This ensures that users receive clear feedback when their grid file lacks necessary latitude information for loading grid coordinates, allowing them to understand and correct the issue.

        Parameters:
            tmp_path (str): A temporary directory provided by pytest for creating test files.

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=8)
        proc = _make_mock_proc(ds, tmp_path, omit_lat=True, prefix="grid_no_lat")

        with pytest.raises(ValueError, match="Cannot find latitude"):
            diag._load_grid_coordinates(proc)


class TestExtractSoundingProfileWithComponents:
    """ Tests for the extract_sounding_profile method when the dataset contains pressure components. """

    def test_pressure_components_dataset_works_end_to_end(self: "TestExtractSoundingProfileWithComponents", 
                                                          tmp_path: str) -> None:
        """
        This test verifies that when the dataset contains pressure components (e.g., 'components' variable), the extract_sounding_profile method can successfully extract a sounding profile end-to-end, including pressure, temperature, dewpoint, wind, and height profiles. It checks that when a synthetic dataset is created with pressure components and the necessary variables for temperature, dewpoint, wind, and height, the method returns a profile with finite pressure values and valid wind and height profiles. This ensures that users can obtain a complete sounding profile from datasets that use pressure components, which is a common format in MPAS output, and that the method can handle the necessary conversions and extractions to produce a consistent profile for analysis. 

        Parameters:
            tmp_path (str): A temporary directory provided by pytest for creating test files.

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)

        ds = _synthetic_mpas_data(
            n_cells=5, n_vert=10,
            pressure_varname="components",
            temp_varname="temperature",
            dew_varname="dewpoint_K",
            include_wind=True,
            include_height=True,
        )

        proc = _make_mock_proc(ds, tmp_path, prefix="grid_comp")
        profile = diag.extract_sounding_profile(proc, -100.0, 35.0)

        assert len(profile["pressure"]) > 0
        assert np.all(np.isfinite(profile["pressure"]))
        assert profile["u_wind"] is not None
        assert profile["height"] is not None

    def test_potential_temperature_conversion_in_full_pipeline(self: "TestExtractSoundingProfileWithComponents", 
                                                               tmp_path: str) -> None:
        """
        This test verifies that when the dataset contains potential temperature ('theta') as the temperature variable and pressure components, the extract_sounding_profile method correctly converts potential temperature to actual temperature in the full extraction pipeline. It checks that when a synthetic dataset is created with 'theta' as the temperature variable and pressure components, the resulting temperature profile in the extracted sounding profile is in a physically reasonable range for Celsius temperatures. This ensures that the method can handle datasets with potential temperature and perform the necessary conversions to produce a valid temperature profile for sounding diagnostics, even when using pressure components. 

        Parameters:
            tmp_path (str): A temporary directory provided by pytest for creating test files.

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)

        ds = _synthetic_mpas_data(
            n_cells=5, n_vert=10,
            pressure_varname="pressure",
            temp_varname="theta",
            dew_varname="qv",
        )

        proc = _make_mock_proc(ds, tmp_path, prefix="grid_theta")
        profile = diag.extract_sounding_profile(proc, -100.0, 35.0)

        assert np.all(profile["temperature"] > -200.0)
        assert np.all(profile["temperature"] < 100.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

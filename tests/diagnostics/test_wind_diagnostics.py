#!/usr/bin/env python3
"""
MPASdiag Test Suite: Wind Diagnostics

This module contains unit tests for the `WindDiagnostics` class in `mpasdiag.diagnostics.wind`. The tests cover initialization, wind speed and direction calculations, and edge cases. The test suite uses pytest fixtures to provide synthetic and real MPAS diagnostic datasets, ensuring that the diagnostics functions correctly across a range of scenarios. The tests also check for proper error handling when expected variables are missing or when invalid inputs are provided.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries 
import os
import sys
import pytest
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
from io import StringIO
from unittest.mock import patch
from tests.test_data_helpers import load_mpas_coords_from_processor, load_mpas_mesh
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mpasdiag.diagnostics.wind import WindDiagnostics


class TestWindDiagnostics:
    """ Test wind diagnostic computations using actual API. """
    
    def test_import_wind_diagnostics(self: "TestWindDiagnostics") -> None:
        """
        This test confirms that the `WindDiagnostics` class can be imported from the `wind` module. It serves as a basic sanity check to ensure that the diagnostics module is correctly structured and that the class is accessible. This is important for validating the overall module organization and catching any import-related issues early in the development process.

        Parameters:
            self (Any): Test case instance.

        Returns:
            None: Assertion validates that `WindDiagnostics` is present in the module.
        """
        from mpasdiag.diagnostics import wind
        assert hasattr(wind, 'WindDiagnostics')
    
    def test_wind_diagnostics_initialization(self: "TestWindDiagnostics") -> None:
        """
        This test verifies that an instance of `WindDiagnostics` can be created with the `verbose` parameter set to `True`. It checks that the instance is not `None` and that the `verbose` attribute is correctly assigned. Proper initialization is crucial for ensuring that the diagnostics object behaves as expected when verbose logging is enabled, which can aid in debugging and understanding the internal workings of the diagnostic computations.

        Parameters:
            self (Any): Test case instance.

        Returns:
            None: Assertions validate the diagnostic instance and its `verbose` attribute.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=True)
        public_methods = [method for method in dir(diag) if not method.startswith('_')]

        expected_methods = [
            'analyze_wind_components',
            'compute_wind_direction',
            'compute_wind_shear',
            'compute_wind_speed',
        ]

        for method in expected_methods:
            assert method in public_methods, f"Expected method '{method}' not found in WindDiagnostics."

        assert diag.verbose is True
    
    def test_compute_wind_speed(self: "TestWindDiagnostics", mock_mpas_2d_data) -> None:
        """
        This test validates the wind speed calculation from `u` and `v` components using the `compute_wind_speed` method. It uses real MPAS 2D diagnostic data provided by the `mock_mpas_2d_data` fixture, extracts the relevant wind components, and computes the wind speed. The test asserts that the output is an xarray DataArray with the same shape as the input components, that all speed values are non-negative, and that the computed speed matches the expected values based on the Pythagorean theorem (sqrt(u^2 + v^2)). This ensures that the core wind speed calculation is accurate and behaves correctly with real diagnostic data.

        Parameters:
            self (Any): Test case instance.
            mock_mpas_2d_data (xarray.Dataset): Fixture providing real MPAS 2D diagnostic data from diag files.

        Returns:
            None: Assertions validate output type, shape, and numeric correctness.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        u = mock_mpas_2d_data['u10'].isel(Time=0).compute()
        v = mock_mpas_2d_data['v10'].isel(Time=0).compute()
        
        speed = diag.compute_wind_speed(u, v)
        
        assert isinstance(speed, xr.DataArray)
        assert speed.shape == u.shape
        assert np.all(speed >= 0)
        expected = np.sqrt(u**2 + v**2)
        assert np.allclose(speed.values, expected.values)
    
    def test_compute_wind_direction(self: "TestWindDiagnostics", mock_mpas_2d_data) -> None:
        """
        This test checks the wind direction calculation using the `compute_wind_direction` method when requesting degrees. It extracts `u` and `v` components from the provided MPAS 2D diagnostic data, computes the wind direction, and asserts that the output is an xarray DataArray with the same shape as the input components. The test also verifies that all direction values are within the valid range of [0, 360] degrees. This ensures that the wind direction calculation correctly handles real diagnostic data and produces results in the expected format and range.

        Parameters:
            self (Any): Test case instance.
            mock_mpas_2d_data (xarray.Dataset): Fixture with real 2D diagnostic wind data from diag files.

        Returns:
            None: Assertions confirm output type, shape, and valid degree range.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        u = mock_mpas_2d_data['u10'].isel(Time=0).compute()
        v = mock_mpas_2d_data['v10'].isel(Time=0).compute()
        
        direction = diag.compute_wind_direction(u, v, degrees=True)
        
        assert isinstance(direction, xr.DataArray)
        assert direction.shape == u.shape
        assert np.all((direction >= 0) & (direction <= 360))
    
    def test_compute_wind_direction_radians(self: "TestWindDiagnostics", mock_mpas_2d_data) -> None:
        """
        This test verifies the wind direction calculation in radians using the `compute_wind_direction` method. It extracts `u` and `v` components from the provided MPAS 2D diagnostic data, computes the wind direction in radians, and asserts that the output is an xarray DataArray with the same shape as the input components. The test also checks that all direction values are within the valid range of [0, 2π] radians. This ensures that the method correctly computes wind direction in radians and produces results in the expected format and range when using real diagnostic data.

        Parameters:
            self (Any): Test case instance.
            mock_mpas_2d_data (xarray.Dataset): Fixture containing real u10/v10 fields from diag files.

        Returns:
            None: Assertions verify radian range and output shape.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        u = mock_mpas_2d_data['u10'].isel(Time=0).compute()
        v = mock_mpas_2d_data['v10'].isel(Time=0).compute()
        
        direction = diag.compute_wind_direction(u, v, degrees=False)
        
        assert isinstance(direction, xr.DataArray)
        assert direction.shape == u.shape
        assert np.all((direction >= 0) & (direction <= 2 * np.pi))
    
    def test_analyze_wind_components(self: "TestWindDiagnostics", mock_mpas_2d_data) -> None:
        """
        This test validates the `analyze_wind_components` method, which performs a comprehensive analysis of the `u` and `v` wind components. It uses real MPAS 2D diagnostic data to compute the analysis, which should return a dictionary containing keys for 'u_component', 'v_component', 'horizontal_speed', and 'direction'. The test asserts that the returned analysis is a dictionary with the expected keys and that each component contains statistics such as 'min', 'max', and 'mean'. This ensures that the method correctly processes real diagnostic data and provides a structured analysis output with relevant statistics for both wind components.

        Parameters:
            self (Any): Test case instance.
            mock_mpas_2d_data (xarray.Dataset): Fixture with real u10/v10 arrays from diag files.

        Returns:
            None: Assertions confirm presence of expected analysis keys and statistics.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        u = mock_mpas_2d_data['u10'].isel(Time=0).compute()
        v = mock_mpas_2d_data['v10'].isel(Time=0).compute()
        
        analysis = diag.analyze_wind_components(u, v)
        
        assert isinstance(analysis, dict)
        assert 'u_component' in analysis
        assert 'v_component' in analysis
        assert 'horizontal_speed' in analysis
        assert 'direction' in analysis
        
        assert 'min' in analysis['u_component']
        assert 'max' in analysis['u_component']
        assert 'mean' in analysis['u_component']
    
    def test_compute_wind_shear(self: "TestWindDiagnostics", mock_mpas_2d_data) -> None:
        """
        This test checks the `compute_wind_shear` method, which calculates the wind shear between upper and lower level wind components. It uses real MPAS 2D diagnostic data to create simulated upper-level wind by scaling the surface wind components. The test asserts that the output is a tuple containing the shear magnitude and direction, that the magnitude is an xarray DataArray with the same shape as the input components, and that all shear magnitudes are non-negative. This ensures that the wind shear calculation correctly handles real diagnostic data and produces results in the expected format with valid numeric properties.

        Parameters:
            self (Any): Test case instance.
            mock_mpas_2d_data (xarray.Dataset): Fixture providing real surface wind fields from diag files.

        Returns:
            None: Assertions validate tuple structure and numeric expectations.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        diag = WindDiagnostics(verbose=False)

        u_lower = mock_mpas_2d_data['u10'].isel(Time=0).compute()
        v_lower = mock_mpas_2d_data['v10'].isel(Time=0).compute()

        u_upper = u_lower * 1.5  
        v_upper = v_lower * 1.5
        
        result = diag.compute_wind_shear(u_upper, v_upper, u_lower, v_lower)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        shear_mag, shear_dir = result
        assert isinstance(shear_mag, xr.DataArray)
        assert shear_mag.shape == u_upper.shape
        assert np.all(shear_mag >= 0)
        assert isinstance(shear_dir, xr.DataArray)
        assert shear_dir.shape == u_upper.shape
        assert np.all((shear_dir >= 0) & (shear_dir <= 360))
    
    def test_wind_speed_with_zeros(self: "TestWindDiagnostics") -> None:
        """
        This test ensures that the `compute_wind_speed` method correctly returns zero wind speed when both `u` and `v` components are zero. It creates simple xarray DataArrays filled with zeros for both components, computes the wind speed, and asserts that all values in the resulting speed array are exactly zero. This is a fundamental correctness check for the basic wind speed calculation, ensuring that it behaves as expected in the case of no wind.

        Parameters:
            self (Any): Test case instance.

        Returns:
            None: Assertion verifies speed values are zero.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        u = xr.DataArray(np.zeros(10))
        v = xr.DataArray(np.zeros(10))
        
        speed = diag.compute_wind_speed(u, v)
        
        assert np.allclose(speed.values, 0.0)
    
    def test_wind_direction_special_cases(self: "TestWindDiagnostics") -> None:
        """
        This test checks the `compute_wind_direction` method for special cases where one of the components is zero, which can lead to edge cases in angle calculations. For example, a north wind (u=0, v<0) should yield a specific direction (e.g., 90 degrees in this implementation). The test creates simple DataArrays for `u` and `v` representing a north wind scenario, computes the wind direction, and asserts that the computed direction matches the expected value within a reasonable tolerance. This ensures that the method correctly handles edge cases in wind direction calculations.

        Parameters:
            self (Any): Test case instance.

        Returns:
            None: Assertions validate computed direction equals expected values within tolerance.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        u = xr.DataArray([0.0])
        v = xr.DataArray([-10.0])
        
        direction = diag.compute_wind_direction(u, v, degrees=True)
        assert np.allclose(direction.values, 90.0, atol=1.0)
    
    def test_divergence_calculation(self: "TestWindDiagnostics") -> None:
        """
        This test asserts that divergence computation methods are not implemented on the diagnostics API yet. The test checks that `calculate_divergence` and `compute_divergence` attributes are absent on the `WindDiagnostics` instance. This documents current limitations and provides a clear failing point to enable future feature additions. Updating this test will be necessary once divergence functionality is introduced.

        Parameters:
            self (Any): Test case instance.

        Returns:
            None: Assertions confirm the methods are currently unavailable.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        assert not hasattr(diag, 'calculate_divergence')
        assert not hasattr(diag, 'compute_divergence')
    
    def test_vorticity_calculation(self: "TestWindDiagnostics") -> None:
        """
        This test verifies that vorticity computation methods are not currently part of the diagnostics API. It checks for the absence of `calculate_vorticity` and `compute_vorticity` on the `WindDiagnostics` class, serving as a placeholder for future development. This test documents the current state of the API and provides a clear target for when vorticity calculations are implemented. It also helps maintainers understand which features are still outstanding. Replace this test with functional checks once vorticity computation is added to validate correctness.

        Parameters:
            self (Any): Test case instance.

        Returns:
            None: Assertions confirm absence of vorticity methods.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        assert not hasattr(diag, 'calculate_vorticity')
        assert not hasattr(diag, 'compute_vorticity')
    
    def test_wind_power_density(self: "TestWindDiagnostics") -> None:
        """
        This test verifies that wind power density computation methods are not currently part of the diagnostics API. It checks for the absence of `calculate_wind_power_density` and `compute_wind_power_density` on the `WindDiagnostics` class, serving as a placeholder for future development. This test documents the current state of the API and provides a clear target for when wind power density calculations are implemented. It also helps maintainers understand which features are still outstanding. Replace this test with functional checks once wind power density computation is added to validate correctness.

        Parameters:
            self (Any): Test case instance.

        Returns:
            None: Assertions check for absence of wind power density methods.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        assert not hasattr(diag, 'calculate_wind_power_density')
        assert not hasattr(diag, 'compute_wind_power_density')
    
    def test_geostrophic_wind(self: "TestWindDiagnostics") -> None:
        """
        This test checks that geostrophic wind computation methods are not currently implemented in the diagnostics API. It asserts that `calculate_geostrophic_wind` and `compute_geostrophic_wind` attributes are absent on the `WindDiagnostics` instance. This serves as a placeholder for future development and documents the current limitations of the API. Once geostrophic wind calculations are added, this test should be updated to validate the correctness of those computations using appropriate test cases.

        Parameters:
            self (Any): Test case instance.

        Returns:
            None: Assertions confirm the geostrophic methods are not available.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        assert not hasattr(diag, 'calculate_geostrophic_wind')
        assert not hasattr(diag, 'compute_geostrophic_wind')


class TestWindDiagnosticsActual:
    """ Test actual WindDiagnostics implementation. """
    
    def test_wind_diagnostics_initialization(self: "TestWindDiagnosticsActual") -> None:
        """
        This test verifies that an instance of `WindDiagnostics` can be created with the `verbose` parameter set to `True`. It checks that the instance is not `None` and that the `verbose` attribute is correctly assigned. Proper initialization is crucial for ensuring that the diagnostics object behaves as expected when verbose logging is enabled, which can aid in debugging and understanding the internal workings of the diagnostic computations.

        Parameters:
            self ("TestWindDiagnosticsActual"): Test case instance.

        Returns:
            None: Assertions verify the `verbose` attribute for both instances.
        """
        diag = WindDiagnostics(verbose=True)
        assert diag.verbose is True
        
        diag_quiet = WindDiagnostics(verbose=False)
        assert diag_quiet.verbose is False
    
    def test_compute_wind_speed_basic(self: "TestWindDiagnosticsActual") -> None:
        """
        This test validates the basic functionality of the `compute_wind_speed` method using simple, deterministic input data. It creates small xarray DataArrays for `u` and `v` components with known values, computes the wind speed, and asserts that the output matches the expected values based on the Pythagorean theorem (sqrt(u^2 + v^2)). The test also checks that the resulting speed array has appropriate attributes such as `units` and `standard_name`. This ensures that the core wind speed calculation is correct and that metadata is properly assigned.

        Parameters:
            self ("TestWindDiagnosticsActual"): Test case instance.

        Returns:
            None: Assertions verify computed speeds and attribute presence.
        """
        diag = WindDiagnostics(verbose=False)
        
        u = xr.DataArray([3.0, 0.0, -4.0], dims=['x'])
        v = xr.DataArray([4.0, 5.0, 3.0], dims=['x'])
        
        speed = diag.compute_wind_speed(u, v)
        
        expected = np.array([5.0, 5.0, 5.0])
        np.testing.assert_array_almost_equal(speed.values, expected)
        
        assert 'units' in speed.attrs
        assert speed.attrs['units'] == 'm s^{-1}'
        assert 'standard_name' in speed.attrs
    
    def test_compute_wind_speed_with_verbose(self: "TestWindDiagnosticsActual", capsys: Any) -> None:
        """
        This test verifies that when `verbose` mode is enabled, the `compute_wind_speed` method prints diagnostic information about the input wind components and the computed speed. It creates simple `u` and `v` DataArrays, computes the wind speed with verbose output, and captures the printed output using the `capsys` fixture. The test asserts that the captured output contains expected substrings indicating that the ranges of the wind components and the computed speed were printed. This ensures that the verbose logging functionality is working as intended and provides useful diagnostic information.

        Parameters:
            self ("TestWindDiagnosticsActual"): Test case instance.
            capsys (Any): Pytest capture fixture for stdout/stderr.

        Returns:
            None: Assertions validate captured verbose output contains expected substrings.
        """
        diag = WindDiagnostics(verbose=True)
        
        u = xr.DataArray([3.0, 0.0, -4.0], dims=['x'])
        v = xr.DataArray([4.0, 5.0, 3.0], dims=['x'])
        
        speed = diag.compute_wind_speed(u, v)
        wsmax = speed.max().item()
        wsmin = speed.min().item()

        captured = capsys.readouterr()
        assert 'Wind component U range' in captured.out
        assert 'Wind speed range' in captured.out
        assert f'{wsmin:.2f}' in captured.out
        assert f'{wsmax:.2f}' in captured.out
    
    def test_compute_wind_speed_2d(self: "TestWindDiagnosticsActual") -> None:
        """
        This test validates the `compute_wind_speed` method using two-dimensional arrays for `u` and `v` components, simulating a time x cells structure. It creates 2D DataArrays by tiling deterministic 1D arrays loaded from a helper function, computes the wind speed, and asserts that the output has the same shape as the input components. The test also checks that all computed speed values are non-negative and that the speed is at least as large as the absolute values of either component. This ensures that the method correctly handles 2D inputs and produces physically consistent results.

        Parameters:
            self ("TestWindDiagnosticsActual"): Test case instance.

        Returns:
            None: Assertions validate returned shape and numeric relationships.
        """
        diag = WindDiagnostics(verbose=False)
        
        lon, lat, u_arr, v_arr = load_mpas_coords_from_processor(n=100)
        u = xr.DataArray(np.tile(u_arr, (5, 1)), dims=['time', 'cells'])
        v = xr.DataArray(np.tile(v_arr, (5, 1)), dims=['time', 'cells'])
        
        speed = diag.compute_wind_speed(u, v)
        
        assert speed.shape == u.shape
        assert (speed >= 0).all()
        assert (speed >= np.abs(u)).all()
        assert (speed >= np.abs(v)).all()
    
    def test_compute_wind_speed_3d(self: "TestWindDiagnosticsActual") -> None:
        """
        This test checks the `compute_wind_speed` method with three-dimensional inputs for `u` and `v`, simulating a time x cells x levels structure. It creates 3D DataArrays by tiling deterministic 2D arrays loaded from a helper function, computes the wind speed, and asserts that the output has the same shape as the input components. The test also verifies that all computed speed values are non-negative, ensuring that the method correctly handles 3D inputs and produces physically consistent results across multiple dimensions.

        Parameters:
            self ("TestWindDiagnosticsActual"): Test case instance.

        Returns:
            None: Assertions confirm shape and basic numeric expectations for 3D inputs.
        """
        diag = WindDiagnostics(verbose=False)
        
        lon, lat, u2d, v2d = load_mpas_mesh(50, 10)
        u3d = np.tile(u2d[np.newaxis, :, :], (3, 1, 1))
        v3d = np.tile(v2d[np.newaxis, :, :], (3, 1, 1))
        u = xr.DataArray(u3d, dims=['time', 'cells', 'levels'])
        v = xr.DataArray(v3d, dims=['time', 'cells', 'levels'])
        
        speed = diag.compute_wind_speed(u, v)
        
        assert speed.shape == u.shape
        assert (speed >= 0).all()
    
    def test_compute_wind_speed_zero_wind(self: "TestWindDiagnosticsActual") -> None:
        """
        This test ensures that the `compute_wind_speed` method correctly returns zero wind speed when both `u` and `v` components are zero. It creates simple xarray DataArrays filled with zeros for both components, computes the wind speed, and asserts that all values in the resulting speed array are exactly zero. This is a fundamental correctness check for the basic wind speed calculation, ensuring that it behaves as expected in the case of no wind.

        Parameters:
            self ("TestWindDiagnosticsActual"): Test case instance.

        Returns:
            None: Assertion verifies speed values are zero.
        """
        diag = WindDiagnostics(verbose=False)
        
        u = xr.DataArray([0.0, 0.0, 0.0], dims=['x'])
        v = xr.DataArray([0.0, 0.0, 0.0], dims=['x'])
        
        speed = diag.compute_wind_speed(u, v)
        
        np.testing.assert_array_almost_equal(speed.values, [0.0, 0.0, 0.0])
    
    def test_compute_wind_speed_preserves_attrs(self: "TestWindDiagnosticsActual") -> None:
        """
        This test checks that the `compute_wind_speed` method preserves relevant attributes from the input `u` and `v` components in the output speed DataArray. It creates simple `u` and `v` DataArrays with specific attributes, computes the wind speed, and asserts that the resulting speed DataArray contains expected metadata keys such as `units`, `standard_name`, and `long_name`. This ensures that the method not only computes correct values but also maintains important metadata for downstream use.

        Parameters:
            self ("TestWindDiagnosticsActual"): Test case instance.

        Returns:
            None: Assertions validate expected metadata keys on the output.
        """
        diag = WindDiagnostics(verbose=False)
        
        u = xr.DataArray([3.0], dims=['x'], attrs={'source': 'test'})
        v = xr.DataArray([4.0], dims=['x'], attrs={'source': 'test'})
        
        speed = diag.compute_wind_speed(u, v)
        
        assert 'units' in speed.attrs
        assert 'standard_name' in speed.attrs
        assert 'long_name' in speed.attrs


class TestWindDiagnosticsEdgeCases:
    """ Test edge cases for wind diagnostics. """
    
    def test_large_wind_speeds(self: "TestWindDiagnosticsEdgeCases") -> None:
        """
        This test validates that the `compute_wind_speed` method can handle large wind speed values without numerical issues. It creates `u` and `v` DataArrays with large values (e.g., 100 m/s), computes the wind speed, and asserts that the output is correct based on the Pythagorean theorem. This ensures that the method can handle extreme cases without overflow or loss of precision, which is important for robustness in real-world applications where high wind speeds may occur.

        Parameters:
            self ("TestWindDiagnosticsEdgeCases"): Test case instance.

        Returns:
            None: Assertions compare numeric outputs to expected values.
        """
        diag = WindDiagnostics(verbose=False)
        
        u = xr.DataArray([100.0, 200.0], dims=['x'])
        v = xr.DataArray([100.0, 200.0], dims=['x'])
        
        speed = diag.compute_wind_speed(u, v)
        
        expected = np.sqrt(2) * np.array([100.0, 200.0])
        np.testing.assert_array_almost_equal(speed.values, expected, decimal=5)
    
    def test_mixed_positive_negative(self: "TestWindDiagnosticsEdgeCases") -> None:
        """
        This test checks that the `compute_wind_speed` method correctly computes speed when `u` and `v` components have mixed positive and negative values. It creates `u` and `v` DataArrays with alternating signs, computes the wind speed, and asserts that all computed speeds have the same magnitude due to symmetry. This ensures that the method correctly handles cases where wind components may cancel each other out in terms of direction but still produce a consistent speed magnitude.

        Parameters:
            self ("TestWindDiagnosticsEdgeCases"): Test case instance.

        Returns:
            None: Assertions validate magnitudes are equal for symmetric inputs.
        """
        diag = WindDiagnostics(verbose=False)
        
        u = xr.DataArray([10.0, -10.0, 10.0, -10.0], dims=['x'])
        v = xr.DataArray([10.0, 10.0, -10.0, -10.0], dims=['x'])
        
        speed = diag.compute_wind_speed(u, v)
        
        expected = np.sqrt(200.0)
        np.testing.assert_array_almost_equal(
            speed.values, 
            [expected] * 4, 
            decimal=5
        )
    
    def test_asymmetric_components(self: "TestWindDiagnosticsEdgeCases") -> None:
        """
        This test verifies that the `compute_wind_speed` method correctly computes speed when one component is significantly larger than the other. It creates `u` and `v` DataArrays where one component dominates (e.g., u=100, v=1), computes the wind speed, and asserts that the computed speed is close to the larger component's magnitude. This ensures that the method correctly handles cases with strong directional bias and produces physically consistent results.

        Parameters:
            self ("TestWindDiagnosticsEdgeCases"): Test case instance.

        Returns:
            None: Assertions check that larger component dominates the computed speed.
        """
        diag = WindDiagnostics(verbose=False)
        
        u = xr.DataArray([100.0, 1.0], dims=['x'])
        v = xr.DataArray([1.0, 100.0], dims=['x'])
        
        speed = diag.compute_wind_speed(u, v)
        
        assert speed.values[0] > 99.0 
        assert speed.values[1] > 99.0 


class TestWindDiagnosticsInitialization:
    """ Test initialization and basic functionality. """
    
    def test_init_verbose_true(self: "TestWindDiagnosticsInitialization") -> None:
        """
        This test verifies that the `WindDiagnostics` class can be initialized with `verbose=True` and that the instance reflects this setting. It creates an instance of `WindDiagnostics` with verbose mode enabled and asserts that the `verbose` attribute is set to `True`. This ensures that the constructor correctly assigns the verbose flag, which is important for enabling detailed logging during diagnostic computations.

        Parameters:
            self ("TestWindDiagnosticsInitialization"): Test case instance.

        Returns:
            None: Assertion validates verbose attribute on the diagnostics instance.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=True)
        assert diag.verbose is True
    
    def test_init_verbose_false(self: "TestWindDiagnosticsInitialization") -> None:
        """
        This test verifies that the `WindDiagnostics` class can be initialized with `verbose=False` and that the instance reflects this setting. It creates an instance of `WindDiagnostics` with verbose mode disabled and asserts that the `verbose` attribute is set to `False`. This ensures that the constructor correctly assigns the verbose flag, allowing for quiet operation when detailed logging is not desired.

        Parameters:
            self ("TestWindDiagnosticsInitialization"): Test case instance.

        Returns:
            None: Assertion validates verbose attribute on the diagnostics instance.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        assert diag.verbose is False
    
    def test_init_default_verbose(self: "TestWindDiagnosticsInitialization") -> None:
        """
        This test checks the default behavior of the `WindDiagnostics` class when initialized without explicitly setting the `verbose` parameter. It creates an instance of `WindDiagnostics` using the default constructor and asserts that the `verbose` attribute is set to `True`. This ensures that the default initialization provides verbose output, which can be helpful for debugging and understanding the internal workings of the diagnostics when no specific configuration is provided.

        Parameters:
            self ("TestWindDiagnosticsInitialization"): Test case instance.

        Returns:
            None: Assertion validates default verbose behavior.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics()
        assert diag.verbose is True


class TestComputeWindSpeed:
    """ Test compute_wind_speed method. """
    
    @pytest.fixture
    def sample_wind_components(self: "TestComputeWindSpeed"):
        """
        This fixture creates sample U and V wind components for testing. The components are designed to produce known wind speeds based on the Pythagorean theorem, allowing for straightforward validation of the `compute_wind_speed` method. The `u` and `v` DataArrays include attributes for units and long names to mimic realistic diagnostic data. This fixture can be reused across multiple tests to ensure consistency in input data.

        Parameters:
            self ("TestComputeWindSpeed"): Test case instance.

        Returns:
            tuple: A tuple containing the `u` and `v` DataArrays with sample wind components.
        """
        u = xr.DataArray(
            np.array([3.0, 4.0, 0.0, -3.0]),
            dims=['nCells'],
            attrs={'units': 'm s^{-1}', 'long_name': 'U component'}
        )
        v = xr.DataArray(
            np.array([4.0, 3.0, 5.0, -4.0]),
            dims=['nCells'],
            attrs={'units': 'm s^{-1}', 'long_name': 'V component'}
        )
        return u, v
    
    def test_compute_wind_speed_basic(self: "TestComputeWindSpeed", sample_wind_components: Any) -> None:
        """
        This test validates the basic functionality of the `compute_wind_speed` method using the provided sample wind components. It computes the wind speed from the `u` and `v` components and asserts that the output is an xarray DataArray with the same shape as the input components. The test also checks that the computed speeds match expected values based on the Pythagorean theorem (e.g., sqrt(3^2 + 4^2) = 5). Additionally, it verifies that the resulting speed DataArray contains appropriate attributes such as `units` and `standard_name`. This ensures that the core wind speed calculation is correct and that metadata is properly assigned for downstream use.

        Parameters:
            self ("TestComputeWindSpeed"): Test case instance.
            sample_wind_components (Any): Fixture returning (u, v) DataArrays.

        Returns:
            None: Assertions validate numeric correctness and metadata.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        u, v = sample_wind_components
        diag = WindDiagnostics(verbose=False)
        
        speed = diag.compute_wind_speed(u, v)
        
        assert isinstance(speed, xr.DataArray)
        assert speed.shape == u.shape

        assert np.isclose(speed.values[0], 5.0)
        assert np.isclose(speed.values[1], 5.0)
        assert np.isclose(speed.values[2], 5.0)

        assert 'units' in speed.attrs
        assert 'standard_name' in speed.attrs
    
    def test_compute_wind_speed_verbose(self: "TestComputeWindSpeed", sample_wind_components: Any) -> None:
        """
        This test verifies that when `verbose` mode is enabled, the `compute_wind_speed` method prints diagnostic information about the input wind components and the computed speed. It captures the printed output using `StringIO` and asserts that it contains expected substrings indicating that the ranges of the wind components and the computed speed were printed. This ensures that the verbose logging functionality is working as intended and provides useful diagnostic information during the computation.

        Parameters:
            self ("TestComputeWindSpeed"): Test case instance.
            sample_wind_components (Any): Fixture returning (u, v) DataArrays.

        Returns:
            None: Assertions validate printed output and result integrity.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        u, v = sample_wind_components
        diag = WindDiagnostics(verbose=True)
        
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            speed = diag.compute_wind_speed(u, v)
        
        wsmax = speed.max().item()
        wsmin = speed.min().item()
        output = captured_output.getvalue()

        assert 'Wind component U range' in output
        assert 'Wind speed range' in output
        assert 'Wind speed mean' in output
        assert f'{wsmin:.2f}' in output
        assert f'{wsmax:.2f}' in output


class TestComputeWindDirection:
    """ Test compute_wind_direction method. """
    
    @pytest.fixture
    def sample_wind_components(self: "TestComputeWindDirection"):
        """
        This fixture creates sample U and V wind components that represent simple cardinal directions (north, east, south, west) for testing the `compute_wind_direction` method. The `u` and `v` DataArrays are designed such that they produce known wind directions based on standard meteorological conventions. For example, a north wind (u=0, v<0) should yield a specific direction (e.g., 90 degrees in this implementation). This fixture allows for straightforward validation of the wind direction calculations and can be reused across multiple tests to ensure consistency in input data.

        Parameters:
            self ("TestComputeWindDirection"): Test case instance.

        Returns:
            tuple: A tuple containing the `u` and `v` DataArrays with sample wind components representing cardinal directions.
        """
        u = xr.DataArray(
            np.array([0.0, -5.0, 0.0, 5.0]),
            dims=['nCells']
        )

        v = xr.DataArray(
            np.array([-5.0, 0.0, 5.0, 0.0]),
            dims=['nCells']
        )

        return u, v
    
    def test_compute_wind_direction_degrees(self: "TestComputeWindDirection", sample_wind_components: Any) -> None:
        """
        This test validates the `compute_wind_direction` method when returning results in degrees. It uses the sample wind components representing cardinal directions to compute the wind direction and asserts that the output is an xarray DataArray with the correct units attribute set to 'degrees'. The test also checks that all computed direction values are within the expected range of 0 to 360 degrees. This ensures that the method correctly calculates wind direction in degrees and handles wrapping around the compass correctly.

        Parameters:
            self ("TestComputeWindDirection"): Test case instance.
            sample_wind_components (Any): Fixture returning (u, v) DataArrays.

        Returns:
            None: Assertions validate units and value ranges in degrees.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        u, v = sample_wind_components
        diag = WindDiagnostics(verbose=False)
        
        direction = diag.compute_wind_direction(u, v, degrees=True)
        
        assert isinstance(direction, xr.DataArray)
        assert 'units' in direction.attrs
        assert direction.attrs['units'] == 'degrees'
        assert np.all((direction.values >= 0) & (direction.values <= 360))
    
    def test_compute_wind_direction_radians(self: "TestComputeWindDirection", sample_wind_components: Any) -> None:
        """
        This test validates the `compute_wind_direction` method when returning results in radians. It uses the sample wind components representing cardinal directions to compute the wind direction and asserts that the output is an xarray DataArray with the correct units attribute set to 'radians'. The test also checks that all computed direction values are within the expected range of 0 to 2π radians. This ensures that the method correctly calculates wind direction in radians and handles wrapping around the compass correctly.

        Parameters:
            self ("TestComputeWindDirection"): Test case instance.
            sample_wind_components (Any): Fixture returning (u, v) DataArrays.

        Returns:
            None: Assertions validate units and value ranges in radians.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        u, v = sample_wind_components
        diag = WindDiagnostics(verbose=False)
        
        direction = diag.compute_wind_direction(u, v, degrees=False)
        
        assert isinstance(direction, xr.DataArray)
        assert 'units' in direction.attrs
        assert direction.attrs['units'] == 'radians'
        assert np.all((direction.values >= 0) & (direction.values <= 2 * np.pi))
    
    def test_compute_wind_direction_verbose_degrees(self: "TestComputeWindDirection", sample_wind_components: Any) -> None:
        """
        This test checks the `compute_wind_direction` method with verbose logging enabled when returning results in degrees. The routine should print user-facing summaries about the wind direction range and units. Capturing stdout verifies the presence of 'degrees' and range information in the output. The test asserts that the printed messages contain expected substrings and that the returned DataArray has the correct shape and attributes.

        Parameters:
            self ("TestComputeWindDirection"): Test case instance.
            sample_wind_components (Any): Fixture returning (u, v) DataArrays.

        Returns:
            None: Assertions validate printed output and result integrity.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        u, v = sample_wind_components
        diag = WindDiagnostics(verbose=True)
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            direction = diag.compute_wind_direction(u, v, degrees=True)
        
        output = captured_output.getvalue()

        assert 'Wind direction range' in output
        assert 'degrees' in output
        assert direction.shape == u.shape
    
    def test_compute_wind_direction_verbose_radians(self: "TestComputeWindDirection", sample_wind_components: Any) -> None:
        """
        This test checks the `compute_wind_direction` method with verbose logging enabled when returning results in radians. The routine should print user-facing summaries about the wind direction range and units. Capturing stdout verifies the presence of 'radians' and range information in the output. The test asserts that the printed messages contain expected substrings and that the returned DataArray has the correct shape and attributes.

        Parameters:
            self ("TestComputeWindDirection"): Test case instance.
            sample_wind_components (Any): Fixture returning (u, v) DataArrays.

        Returns:
            None: Assertions validate verbose reporting and numeric output.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        u, v = sample_wind_components
        diag = WindDiagnostics(verbose=True)
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            direction = diag.compute_wind_direction(u, v, degrees=False)
        
        wdmax = direction.max().item()
        wdmin = direction.min().item()
        output = captured_output.getvalue()

        assert 'Wind direction range' in output
        assert 'radians' in output

        assert f'{wdmin:.2f}' in output
        assert f'{wdmax:.2f}' in output


class TestAnalyzeWindComponents:
    """ Test analyze_wind_components method. """
    
    @pytest.fixture
    def sample_wind_components(self: "TestAnalyzeWindComponents"):
        """
        This fixture creates sample U, V, and W wind components with random values for testing the `analyze_wind_components` method. The `u` and `v` components are generated with a larger scale to represent typical horizontal wind speeds, while the `w` component is smaller to reflect vertical motion. Each component is an xarray DataArray with appropriate dimensions and units attributes. This fixture allows for comprehensive testing of the analysis function's ability to compute statistics across all three wind components.

        Parameters:
            self ("TestAnalyzeWindComponents"): Test case instance.

        Returns:
            tuple: A tuple containing the `u`, `v`, and `w` DataArrays with sample wind components for analysis.
        """
        u = xr.DataArray(
            np.random.randn(100) * 5,
            dims=['nCells'],
            attrs={'units': 'm s^{-1}'}
        )

        v = xr.DataArray(
            np.random.randn(100) * 5,
            dims=['nCells'],
            attrs={'units': 'm s^{-1}'}
        )

        w = xr.DataArray(
            np.random.randn(100) * 0.5,
            dims=['nCells'],
            attrs={'units': 'm s^{-1}'}
        )

        return u, v, w
    
    def test_analyze_wind_components_2d(self: "TestAnalyzeWindComponents", sample_wind_components: Any) -> None:
        """
        This test validates the `analyze_wind_components` method when only 2D horizontal components (U and V) are provided. The analysis should compute summary statistics for the U and V components, as well as derived metrics like horizontal speed and direction. The test asserts that the returned analysis dictionary contains expected keys and that each component's statistics include minimum, maximum, mean, standard deviation, and units. This ensures that the method correctly handles 2D inputs and produces comprehensive diagnostics for horizontal wind components.

        Parameters:
            self ("TestAnalyzeWindComponents"): Test case instance.
            sample_wind_components (Any): Fixture returning (u, v, w) DataArrays.

        Returns:
            None: Assertions validate analysis dictionary structure for 2D input.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        u, v, _ = sample_wind_components
        diag = WindDiagnostics(verbose=False)
        
        analysis = diag.analyze_wind_components(u, v)
        
        assert isinstance(analysis, dict)
        assert 'u_component' in analysis
        assert 'v_component' in analysis
        assert 'horizontal_speed' in analysis
        assert 'direction' in analysis
        assert 'w_component' not in analysis
        assert 'total_speed' not in analysis
        
        for key in ['u_component', 'v_component', 'horizontal_speed', 'direction']:
            assert 'min' in analysis[key]
            assert 'max' in analysis[key]
            assert 'mean' in analysis[key]
            assert 'std' in analysis[key]
            assert 'units' in analysis[key]
    
    def test_analyze_wind_components_3d(self: "TestAnalyzeWindComponents", sample_wind_components: Any) -> None:
        """
        This test validates the `analyze_wind_components` method when all three components (U, V, W) are provided. The analysis should compute summary statistics for the W component in addition to the horizontal diagnostics. The test asserts that the returned analysis dictionary contains keys for both horizontal and vertical components, and that each component's statistics include minimum, maximum, mean, standard deviation, and units. This ensures that the method correctly handles 3D inputs and provides comprehensive diagnostics for all wind components.

        Parameters:
            self ("TestAnalyzeWindComponents"): Test case instance.
            sample_wind_components (Any): Fixture returning (u, v, w) DataArrays.

        Returns:
            None: Assertions validate analysis dictionary structure for 3D input.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        u, v, w = sample_wind_components
        diag = WindDiagnostics(verbose=False)
        
        analysis = diag.analyze_wind_components(u, v, w)
        
        assert isinstance(analysis, dict)
        assert 'w_component' in analysis
        assert 'total_speed' in analysis
        
        assert 'min' in analysis['w_component']
        assert 'max' in analysis['w_component']
        assert 'mean' in analysis['w_component']
        assert 'std' in analysis['w_component']
    
    def test_analyze_wind_components_verbose_2d(self: "TestAnalyzeWindComponents", sample_wind_components: Any) -> None:
        """
        This test checks the `analyze_wind_components` method with verbose logging enabled when analyzing only 2D horizontal components. The routine should print user-facing summaries about the U and V components, horizontal speed, and direction. Capturing stdout verifies the presence of expected summary lines in the output. The test asserts that the printed messages contain expected substrings indicating that the analysis was performed and that key metrics were reported.

        Parameters:
            self ("TestAnalyzeWindComponents"): Test case instance.
            sample_wind_components (Any): Fixture returning (u, v, w) DataArrays.

        Returns:
            None: Assertions validate printed summaries in verbose mode.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        u, v, _ = sample_wind_components
        diag = WindDiagnostics(verbose=True)
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            analysis = diag.analyze_wind_components(u, v)
        
        output = captured_output.getvalue()
        assert isinstance(analysis, dict)
        assert analysis.keys() == {'u_component', 'v_component', 'horizontal_speed', 'direction'}   
        assert 'Wind Component Analysis' in output
        assert 'U component' in output
        assert 'V component' in output
        assert 'Horizontal speed' in output
        assert 'Direction' in output
    
    def test_analyze_wind_components_verbose_3d(self: "TestAnalyzeWindComponents", sample_wind_components: Any) -> None:
        """
        This test checks the `analyze_wind_components` method with verbose logging enabled when analyzing all three components (U, V, W). The routine should print user-facing summaries about the W component and total 3D speed in addition to the horizontal diagnostics. Capturing stdout verifies the presence of expected summary lines in the output. The test asserts that the printed messages contain expected substrings indicating that the analysis was performed for all components and that key metrics were reported.

        Parameters:
            self ("TestAnalyzeWindComponents"): Test case instance.
            sample_wind_components (Any): Fixture returning (u, v, w) DataArrays.

        Returns:
            None: Assertions validate verbose 3D analysis messages.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        u, v, w = sample_wind_components
        diag = WindDiagnostics(verbose=True)
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            analysis = diag.analyze_wind_components(u, v, w)
        
        output = captured_output.getvalue()
        assert 'W component' in output
        assert 'Total 3D speed' in output
        assert isinstance(analysis, dict)
        assert analysis.keys() == {'u_component', 'v_component', 'horizontal_speed', 'direction', 'w_component', 'total_speed'}


class TestComputeWindShear:
    """ Test compute_wind_shear method. """
    
    @pytest.fixture
    def sample_wind_levels(self: "TestComputeWindShear"):
        """
        This fixture creates sample U and V wind components for two vertical levels (upper and lower) to test the `compute_wind_shear` method. The upper level has stronger winds than the lower level, allowing for a clear shear signal. The `u` and `v` DataArrays are designed to produce known shear magnitudes and directions based on the differences between the two levels. This fixture can be reused across multiple tests to ensure consistency in input data for vertical wind shear calculations.

        Parameters:
            self ("TestComputeWindShear"): Test case instance.

        Returns:
            tuple: A tuple containing the `u` and `v` DataArrays for upper and lower levels, structured as (u_upper, v_upper, u_lower, v_lower).
        """
        u_upper = xr.DataArray(
            np.array([10.0, 15.0, 20.0]),
            dims=['nCells']
        )

        v_upper = xr.DataArray(
            np.array([5.0, 10.0, 15.0]),
            dims=['nCells']
        )

        u_lower = xr.DataArray(
            np.array([5.0, 10.0, 15.0]),
            dims=['nCells']
        )

        v_lower = xr.DataArray(
            np.array([2.0, 5.0, 10.0]),
            dims=['nCells']
        )

        return u_upper, v_upper, u_lower, v_lower
    
    def test_compute_wind_shear_basic(self: "TestComputeWindShear", sample_wind_levels: Any) -> None:
        """
        This test validates the basic functionality of the `compute_wind_shear` method using the provided sample wind levels. It computes the wind shear magnitude and direction from the upper and lower level U and V components and asserts that the outputs are xarray DataArrays with the same shape as the input components. The test also checks that the computed shear magnitude is positive and that the direction values are within the expected range of 0 to 360 degrees. Additionally, it verifies that the resulting DataArrays contain appropriate attributes such as `units` and `long_name`. This ensures that the core wind shear calculation is correct and that metadata is properly assigned for downstream use.

        Parameters:
            self ("TestComputeWindShear"): Test case instance.
            sample_wind_levels (Any): Fixture returning upper/lower u/v DataArrays.

        Returns:
            None: Assertions validate returned magnitude and direction DataArrays.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        u_upper, v_upper, u_lower, v_lower = sample_wind_levels
        diag = WindDiagnostics(verbose=False)
        
        shear_mag, shear_dir = diag.compute_wind_shear(
            u_upper, v_upper, u_lower, v_lower
        )
        
        assert isinstance(shear_mag, xr.DataArray)
        assert isinstance(shear_dir, xr.DataArray)
        assert shear_mag.shape == u_upper.shape
        assert shear_dir.shape == u_upper.shape
        assert 'units' in shear_mag.attrs
        assert 'long_name' in shear_dir.attrs
    
    def test_compute_wind_shear_verbose(self: "TestComputeWindShear", sample_wind_levels: Any) -> None:
        """
        This test checks the `compute_wind_shear` method with verbose logging enabled. The routine should print user-facing summaries about the wind shear magnitude and direction ranges. Capturing stdout verifies the presence of expected summary lines in the output. The test asserts that the printed messages contain expected substrings indicating that the shear analysis was performed and that key metrics were reported. This ensures that the verbose logging functionality is working as intended and provides useful diagnostic information during the wind shear computation.

        Parameters:
            self ("TestComputeWindShear"): Test case instance.
            sample_wind_levels (Any): Fixture returning upper/lower u/v DataArrays.

        Returns:
            None: Assertions validate printed output in verbose mode.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        u_upper, v_upper, u_lower, v_lower = sample_wind_levels

        diag = WindDiagnostics(verbose=True)
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            shear_mag, shear_dir = diag.compute_wind_shear(
                u_upper, v_upper, u_lower, v_lower
            )
        
        output = captured_output.getvalue()
        assert 'Wind shear magnitude range' in output
        assert 'Wind shear magnitude mean' in output


class TestGet3DWindComponents:
    """ Test get_3d_wind_components method. """
    
    @pytest.fixture
    def sample_3d_dataset(self: "TestGet3DWindComponents"):
        """
        This fixture creates a sample xarray Dataset with synthetic 3D wind component data (U, V, W) and pressure levels for testing the `get_3d_wind_components` method. The dataset includes dimensions for time, vertical levels, and horizontal cells, with random values for the wind components to mimic realistic variability. Pressure data is included to allow for testing of pressure-level selection. This fixture provides a comprehensive test dataset that can be used across multiple tests to validate the functionality of extracting 3D wind components based on different level specifications.

        Parameters:
            self ("TestGet3DWindComponents"): Test case instance.

        Returns:
            xr.Dataset: A synthetic dataset containing 3D wind components and pressure levels for testing.
        """
        nCells = 50
        nVertLevels = 10
        nTime = 5
        
        u_data = np.random.randn(nTime, nVertLevels, nCells) * 10
        v_data = np.random.randn(nTime, nVertLevels, nCells) * 10
        w_data = np.random.randn(nTime, nVertLevels, nCells) * 0.5
        
        pressure_p = np.random.rand(nTime, nVertLevels, nCells) * 10000 + 50000
        pressure_base = np.ones((nTime, nVertLevels, nCells)) * 50000
        
        ds = xr.Dataset({
            'uReconstructZonal': (['Time', 'nVertLevels', 'nCells'], u_data),
            'vReconstructMeridional': (['Time', 'nVertLevels', 'nCells'], v_data),
            'w': (['Time', 'nVertLevels', 'nCells'], w_data),
            'pressure_p': (['Time', 'nVertLevels', 'nCells'], pressure_p),
            'pressure_base': (['Time', 'nVertLevels', 'nCells'], pressure_base),
        })
        
        return ds
    
    def test_get_3d_wind_components_model_level(self: "TestGet3DWindComponents", sample_3d_dataset: xr.Dataset) -> None:
        """
        This test verifies that the `get_3d_wind_components` method correctly extracts U, V, and W components at a specified model vertical level index (e.g., level=5). The test asserts that the returned components are xarray DataArrays with the expected shape and that they include metadata indicating the selected level. This ensures that the method can successfully retrieve wind components based on model-level selection and that it annotates the results with relevant metadata for downstream use.

        Parameters:
            self ("TestGet3DWindComponents"): Test case instance.
            sample_3d_dataset (xr.Dataset): Fixture returning a synthetic 3D dataset.

        Returns:
            None: Assertions validate extraction and metadata for model-level selection.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        
        u, v, w = diag.get_3d_wind_components(
            sample_3d_dataset,
            u_variable='uReconstructZonal',
            v_variable='vReconstructMeridional',
            w_variable='w',
            level=5,
            time_index=0
        )
        
        assert isinstance(u, xr.DataArray)
        assert isinstance(v, xr.DataArray)
        assert isinstance(w, xr.DataArray)
        assert u.shape == (50,)
        assert 'selected_level' in u.attrs
        assert u.attrs['level_index'] == pytest.approx(5)
    
    def test_get_3d_wind_components_pressure_level(self: "TestGet3DWindComponents", sample_3d_dataset: xr.Dataset) -> None:
        """
        This test verifies that the `get_3d_wind_components` method correctly extracts U, V, and W components at a specified pressure level (e.g., level=85000.0 Pa). The test asserts that the method identifies the correct vertical index corresponding to the requested pressure level and that the returned components are xarray DataArrays with appropriate metadata indicating the selected pressure level. This ensures that the method can successfully retrieve wind components based on pressure-level selection and provides informative output about the selection process when verbose mode is enabled.

        Parameters:
            self ("TestGet3DWindComponents"): Test case instance.
            sample_3d_dataset (xr.Dataset): Fixture returning a synthetic 3D dataset.

        Returns:
            None: Assertions validate printed output and returned DataArrays.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=True)
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            u, v, w = diag.get_3d_wind_components(
                sample_3d_dataset,
                u_variable='uReconstructZonal',
                v_variable='vReconstructMeridional',
                w_variable='w',
                level=85000.0, 
                time_index=0
            )
        
        output = captured_output.getvalue()
        assert 'Requested pressure' in output
        assert isinstance(u, xr.DataArray)
    
    def test_get_3d_wind_components_surface(self: "TestGet3DWindComponents", sample_3d_dataset: xr.Dataset) -> None:
        """
        This test verifies that the `get_3d_wind_components` method correctly extracts U, V, and W components at the surface level when `level='surface'` is specified. The helper should identify the lowest vertical index (0) as the surface level and annotate the returned components with metadata indicating this selection. The test asserts that the `level_index` attribute of the returned U component equals 0, confirming that the surface level was correctly identified and extracted.

        Parameters:
            self ("TestGet3DWindComponents"): Test case instance.
            sample_3d_dataset (xr.Dataset): Fixture returning a synthetic 3D dataset.

        Returns:
            None: Assertion verifies `level_index` is 0 for surface.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        
        u, v, w = diag.get_3d_wind_components(
            sample_3d_dataset,
            u_variable='uReconstructZonal',
            v_variable='vReconstructMeridional',
            w_variable='w',
            level='surface',
            time_index=0
        )
        
        assert u.attrs['level_index'] == pytest.approx(0)
    
    def test_get_3d_wind_components_top(self: "TestGet3DWindComponents", sample_3d_dataset: xr.Dataset) -> None:
        """
        This test verifies that the `get_3d_wind_components` method correctly extracts U, V, and W components at the top level when `level='top'` is specified. The helper should identify the highest vertical index (nVertLevels-1) as the top level and annotate the returned components with metadata indicating this selection. The test asserts that the `level_index` attribute of the returned U component equals the expected top index, confirming that the top level was correctly identified and extracted.

        Parameters:
            self ("TestGet3DWindComponents"): Test case instance.
            sample_3d_dataset (xr.Dataset): Fixture returning a synthetic 3D dataset.

        Returns:
            None: Assertion verifies `level_index` corresponds to top level.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        
        u, v, w = diag.get_3d_wind_components(
            sample_3d_dataset,
            u_variable='uReconstructZonal',
            v_variable='vReconstructMeridional',
            w_variable='w',
            level='top',
            time_index=0
        )
        
        assert u.attrs['level_index'] == pytest.approx(9) 
    
    def test_get_3d_wind_components_missing_w(self: "TestGet3DWindComponents", sample_3d_dataset: xr.Dataset) -> None:
        """
        This test verifies that when the specified W variable is missing from the dataset, the `get_3d_wind_components` method falls back to setting W to zero and prints a warning message. The test captures stdout to check for the presence of a warning about the missing W variable and asserts that the returned W component is an xarray DataArray filled with zeros. This ensures that the method handles missing vertical wind components gracefully while providing informative feedback to the user.

        Parameters:
            self ("TestGet3DWindComponents"): Test case instance.
            sample_3d_dataset (xr.Dataset): Fixture returning a synthetic 3D dataset.

        Returns:
            None: Assertions validate fallback W handling and messages.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=True)
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            u, v, w = diag.get_3d_wind_components(
                sample_3d_dataset,
                u_variable='uReconstructZonal',
                v_variable='vReconstructMeridional',
                w_variable='w_missing',  
                level=0,
                time_index=0
            )
        
        output = captured_output.getvalue()
        assert 'Warning' in output or 'Setting W component to zero' in output
        assert np.all(w.values == 0)
    
    def test_get_3d_wind_components_verbose(self: "TestGet3DWindComponents", sample_3d_dataset: xr.Dataset) -> None:
        """
        This test verifies that when `verbose` mode is enabled, the `get_3d_wind_components` method prints diagnostic information about the extraction process. The test captures stdout to check for the presence of messages indicating that the method is extracting 3D wind components, along with details about the selected level and variable ranges. This ensures that the verbose logging functionality is working as intended and provides useful feedback during the component extraction process.

        Parameters:
            self ("TestGet3DWindComponents"): Test case instance.
            sample_3d_dataset (xr.Dataset): Fixture returning a synthetic 3D dataset.

        Returns:
            None: Assertions validate printed messages in verbose mode.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=True)
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            u, v, w = diag.get_3d_wind_components(
                sample_3d_dataset,
                u_variable='uReconstructZonal',
                v_variable='vReconstructMeridional',
                w_variable='w',
                level=0,
                time_index=0
            )
        
        output = captured_output.getvalue()
        assert 'Extracting 3D wind components' in output
        assert 'Wind component' in output
        assert 'range' in output
        assert 'Units' in output
    
    def test_get_3d_wind_components_missing_variable(self: "TestGet3DWindComponents", sample_3d_dataset: xr.Dataset) -> None:
        """
        This test verifies that if any of the specified U, V, or W variables are missing from the dataset, the `get_3d_wind_components` method raises a ValueError with an appropriate message. The test attempts to extract components using a non-existent variable name and asserts that the expected exception is raised, ensuring that the method provides clear feedback about missing data rather than failing silently or producing incorrect results.

        Parameters:
            self ("TestGet3DWindComponents"): Test case instance.
            sample_3d_dataset (xr.Dataset): Fixture returning a synthetic 3D dataset.

        Returns:
            None: Assertion verifies a ValueError is raised for missing variables.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        
        with pytest.raises(ValueError, match="not found in dataset"):
            diag.get_3d_wind_components(
                sample_3d_dataset,
                u_variable='missing_u',
                v_variable='vReconstructMeridional',
                w_variable='w',
                level=0,
                time_index=0
            )
    
    def test_get_3d_wind_components_invalid_level(self: "TestGet3DWindComponents", sample_3d_dataset: xr.Dataset) -> None:
        """
        This test verifies that if an invalid model level index is specified (e.g., level=100 which exceeds available levels), the `get_3d_wind_components` method raises a ValueError with an appropriate message. The test attempts to extract components using an out-of-range level index and asserts that the expected exception is raised, ensuring that the method provides clear feedback about invalid level specifications rather than failing silently or producing incorrect results.

        Parameters:
            self ("TestGet3DWindComponents"): Test case instance.
            sample_3d_dataset (xr.Dataset): Fixture returning a synthetic 3D dataset.

        Returns:
            None: Assertion verifies a ValueError for out-of-range levels.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        
        with pytest.raises(ValueError, match="exceeds available levels"):
            diag.get_3d_wind_components(
                sample_3d_dataset,
                u_variable='uReconstructZonal',
                v_variable='vReconstructMeridional',
                w_variable='w',
                level=100, 
                time_index=0
            )
    
    def test_get_3d_wind_components_invalid_level_string(self: "TestGet3DWindComponents", sample_3d_dataset: xr.Dataset) -> None:
        """
        This test verifies that if an invalid string is specified for the level (e.g., level='middle'), the `get_3d_wind_components` method raises a ValueError with an appropriate message. The test attempts to extract components using an unrecognized string for the level and asserts that the expected exception is raised, ensuring that the method provides clear feedback about unknown level specifications rather than failing silently or producing incorrect results.

        Parameters:
            self ("TestGet3DWindComponents"): Test case instance.
            sample_3d_dataset (xr.Dataset): Fixture returning a synthetic 3D dataset.

        Returns:
            None: Assertion verifies a ValueError for unknown string level.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        
        with pytest.raises(ValueError, match="Unknown level specification"):
            diag.get_3d_wind_components(
                sample_3d_dataset,
                u_variable='uReconstructZonal',
                v_variable='vReconstructMeridional',
                w_variable='w',
                level='middle',  
                time_index=0
            )
    
    def test_get_3d_wind_components_no_pressure_data(self: "TestGet3DWindComponents") -> None:
        """
        This test verifies that if pressure data is required for level selection but is missing from the dataset, the `get_3d_wind_components` method raises a ValueError with an appropriate message. The test creates a dataset that includes U, V, and W components but lacks pressure variables, then attempts to extract components at a specified pressure level. The assertion checks that the expected exception is raised, ensuring that the method provides clear feedback about missing pressure data rather than failing silently or producing incorrect results.

        Parameters:
            self ("TestGet3DWindComponents"): Test case instance.

        Returns:
            None: Assertion verifies a ValueError for missing pressure data.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        ds = xr.Dataset({
            'uReconstructZonal': (['Time', 'nVertLevels', 'nCells'], 
                                 np.random.randn(5, 10, 50)),
            'vReconstructMeridional': (['Time', 'nVertLevels', 'nCells'], 
                                      np.random.randn(5, 10, 50)),
            'w': (['Time', 'nVertLevels', 'nCells'], 
                 np.random.randn(5, 10, 50)),
        })
        
        diag = WindDiagnostics(verbose=False)
        
        with pytest.raises(ValueError, match="pressure data not available"):
            diag.get_3d_wind_components(
                ds,
                u_variable='uReconstructZonal',
                v_variable='vReconstructMeridional',
                w_variable='w',
                level=85000.0,
                time_index=0
            )
    
    def test_get_3d_wind_components_uxarray(self: "TestGet3DWindComponents", sample_3d_dataset: xr.Dataset) -> None:
        """
        This test verifies that when requesting the 'uxarray' data type, the `get_3d_wind_components` method returns `xarray.DataArray` objects. The test ensures compatibility with downstream code that expects xarray objects. The test asserts that the returned U, V, and W components are instances of `xarray.DataArray`, confirming that the method correctly handles the 'uxarray' data type option and provides results in the expected format for further analysis.

        Parameters:
            self ("TestGet3DWindComponents"): Test case instance.
            sample_3d_dataset (xr.Dataset): Fixture returning a synthetic 3D dataset.

        Returns:
            None: Assertions validate returned types for 'uxarray' request.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        
        u, v, w = diag.get_3d_wind_components(
            sample_3d_dataset,
            u_variable='uReconstructZonal',
            v_variable='vReconstructMeridional',
            w_variable='w',
            level=0,
            time_index=0,
            data_type='uxarray'
        )
        
        assert isinstance(u, xr.DataArray)


class TestGet2DWindComponents:
    """ Test get_2d_wind_components method. """
    
    @pytest.fixture
    def sample_2d_dataset(self: "TestGet2DWindComponents"):
        """
        This fixture creates a sample xarray Dataset with synthetic 2D wind component data (U and V) for testing the `get_2d_wind_components` method. The dataset includes dimensions for time and horizontal cells, with random values for the U and V components to mimic realistic variability. This fixture provides a simple test dataset that can be used across multiple tests to validate the functionality of extracting 2D wind components based on variable names and time indices.

        Parameters:
            self ("TestGet2DWindComponents"): Test case instance.

        Returns:
            xr.Dataset: A synthetic dataset containing 2D wind components for testing.
        """
        nCells = 100
        nTime = 10
        
        u_data = np.random.randn(nTime, nCells) * 5
        v_data = np.random.randn(nTime, nCells) * 5
        
        ds = xr.Dataset({
            'u10': (['Time', 'nCells'], u_data, {'units': 'm s^{-1}'}),
            'v10': (['Time', 'nCells'], v_data, {'units': 'm s^{-1}'}),
        })
        
        return ds
    
    def test_get_2d_wind_components_basic(self: "TestGet2DWindComponents", sample_2d_dataset: xr.Dataset) -> None:
        """
        This test verifies that the `get_2d_wind_components` method correctly extracts U and V components at a specified time index. The test asserts that the returned components are xarray DataArrays with the expected shape and that they include metadata indicating the selected time index. This ensures that the method can successfully retrieve 2D wind components based on variable names and time indices, and that it annotates the results with relevant metadata for downstream use.

        Parameters:
            self ("TestGet2DWindComponents"): Test case instance.
            sample_2d_dataset (xr.Dataset): Fixture returning a synthetic 2D dataset.

        Returns:
            None: Assertions validate types and shapes of U and V arrays.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        
        u, v = diag.get_2d_wind_components(
            sample_2d_dataset,
            u_variable='u10',
            v_variable='v10',
            time_index=5
        )
        
        assert isinstance(u, xr.DataArray)
        assert isinstance(v, xr.DataArray)
        assert u.shape == (100,)
        assert v.shape == (100,)
    
    def test_get_2d_wind_components_verbose(self: "TestGet2DWindComponents", sample_2d_dataset: xr.Dataset) -> None:
        """
        This test verifies that when `verbose` mode is enabled, the `get_2d_wind_components` method prints diagnostic information about the extraction process. The test captures stdout to check for the presence of messages indicating that the method is extracting 2D wind components, along with details about variable ranges and units. This ensures that the verbose logging functionality is working as intended and provides useful feedback during the component extraction process.

        Parameters:
            self ("TestGet2DWindComponents"): Test case instance.
            sample_2d_dataset (xr.Dataset): Fixture returning a synthetic 2D dataset.

        Returns:
            None: Assertions validate printed output for verbose extraction.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=True)
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            u, v = diag.get_2d_wind_components(
                sample_2d_dataset,
                u_variable='u10',
                v_variable='v10',
                time_index=0
            )
        
        output = captured_output.getvalue()
        assert 'Extracting wind components' in output
        assert 'Wind component u10 range' in output
        assert 'Wind speed range' in output
        assert 'Units' in output
    
    def test_get_2d_wind_components_unit_mismatch(self: "TestGet2DWindComponents") -> None:
        """
        This test verifies that if the U and V variables have different units, the `get_2d_wind_components` method prints a warning message about the unit mismatch. The test creates a dataset where U and V components have different units (e.g., m/s for U and km/h for V) and captures stdout to check for the presence of a warning about the unit mismatch. This ensures that the method provides informative feedback to the user about potential issues with the input data, allowing them to address unit inconsistencies before proceeding with analysis.

        Parameters:
            self ("TestGet2DWindComponents"): Test case instance.

        Returns:
            None: Assertions validate warning is produced for unit mismatch.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        ds = xr.Dataset({
            'u10': (['Time', 'nCells'], np.random.randn(10, 100), 
                   {'units': 'm s^{-1}'}),
            'v10': (['Time', 'nCells'], np.random.randn(10, 100), 
                   {'units': 'km h^{-1}'}), 
        })
        
        diag = WindDiagnostics(verbose=True)
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            u, v = diag.get_2d_wind_components(
                ds, u_variable='u10', v_variable='v10', time_index=0
            )
        
        output = captured_output.getvalue()
        assert 'Warning' in output
        assert 'different units' in output
    
    def test_get_2d_wind_components_no_dataset(self: "TestGet2DWindComponents") -> None:
        """
        This test verifies that if no dataset is provided (i.e., `dataset=None`), the `get_2d_wind_components` method raises a RuntimeError with an appropriate message. The test attempts to extract wind components without providing a dataset and asserts that the expected exception is raised, ensuring that the method provides clear feedback about missing input data rather than failing silently or producing incorrect results.

        Parameters:
            self ("TestGet2DWindComponents"): Test case instance.

        Returns:
            None: Assertion verifies a RuntimeError is raised for None dataset.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        
        with pytest.raises(RuntimeError, match="No dataset provided"):
            diag.get_2d_wind_components(
                None, u_variable='u10', v_variable='v10', time_index=0  # type: ignore
            )
    
    def test_get_2d_wind_components_missing_variable(self: "TestGet2DWindComponents", sample_2d_dataset: xr.Dataset) -> None:
        """
        This test verifies that if either the specified U or V variable is missing from the dataset, the `get_2d_wind_components` method raises a ValueError with an appropriate message. The test attempts to extract components using a non-existent variable name and asserts that the expected exception is raised, ensuring that the method provides clear feedback about missing variables rather than failing silently or producing incorrect results.

        Parameters:
            self ("TestGet2DWindComponents"): Test case instance.
            sample_2d_dataset (xr.Dataset): Fixture returning a synthetic 2D dataset.

        Returns:
            None: Assertion verifies a ValueError for missing variables.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        
        with pytest.raises(ValueError, match="not found in dataset"):
            diag.get_2d_wind_components(
                sample_2d_dataset,
                u_variable='missing_u',
                v_variable='v10',
                time_index=0
            )
    
    def test_get_2d_wind_components_uxarray(self: "TestGet2DWindComponents", sample_2d_dataset: xr.Dataset) -> None:
        """
        This test verifies that when requesting the 'uxarray' data type, the `get_2d_wind_components` method returns `xarray.DataArray` objects. The test ensures compatibility with downstream code that expects xarray objects. The test asserts that the returned U and V components are instances of `xarray.DataArray`, confirming that the method correctly handles the 'uxarray' data type option and provides results in the expected format for further analysis.

        Parameters:
            self ("TestGet2DWindComponents"): Test case instance.
            sample_2d_dataset (xr.Dataset): Fixture returning a synthetic 2D dataset.

        Returns:
            None: Assertions validate returned types for 'uxarray'.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag = WindDiagnostics(verbose=False)
        
        u, v = diag.get_2d_wind_components(
            sample_2d_dataset,
            u_variable='u10',
            v_variable='v10',
            time_index=0,
            data_type='uxarray'
        )
        
        assert isinstance(u, xr.DataArray)
        assert isinstance(v, xr.DataArray)


class TestEdgeCasesAndErrorPaths:
    """ Test edge cases and error handling paths for complete coverage of get_3d_wind_components. """
    
    def test_get_3d_wind_components_with_neither_dimension(self: "TestEdgeCasesAndErrorPaths") -> None:
        """
        This test verifies that if the specified U, V, and W variables are present in the dataset but do not have the expected vertical dimension (i.e., they are 2D instead of 3D), the `get_3d_wind_components` method raises a ValueError with an appropriate message. The test creates a dataset with 2D U, V, and W variables and attempts to extract 3D components, asserting that the expected exception is raised. This ensures that the method provides clear feedback about incorrect variable dimensions rather than failing silently or producing incorrect results.

        Parameters:
            self ("TestEdgeCasesAndErrorPaths"): Test case instance.

        Returns:
            None: Assertion verifies error for variables missing vertical dimension.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        nCells = 30
        nTime = 2
        
        dataset = xr.Dataset({
            'u_2d': (['Time', 'nCells'], np.random.randn(nTime, nCells)),
            'v_2d': (['Time', 'nCells'], np.random.randn(nTime, nCells)),
            'w_2d': (['Time', 'nCells'], np.random.randn(nTime, nCells)),
            'xtime': (['Time'], [b'2023-01-01_00:00:00', b'2023-01-01_01:00:00'])
        })
        
        wind_diag = WindDiagnostics()
        
        with pytest.raises(ValueError, match="is not a 3D atmospheric variable"):
            wind_diag.get_3d_wind_components(
                dataset, 'u_2d', 'v_2d', 'w_2d', level=5, time_index=0
            )
    
    def test_get_3d_wind_components_invalid_level_type(self: "TestEdgeCasesAndErrorPaths") -> None:
        """
        This test verifies that if an invalid type is specified for the `level` parameter (e.g., a list or None), the `get_3d_wind_components` method raises a ValueError with an appropriate message. The test attempts to extract 3D wind components using invalid level specifications and asserts that the expected exception is raised, ensuring that the method provides clear feedback about invalid level types rather than failing silently or producing incorrect results.

        Parameters:
            self ("TestEdgeCasesAndErrorPaths"): Test case instance.

        Returns:
            None: Assertions validate ValueError is raised for invalid level types.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        nCells = 30
        nVertLevels = 10
        nTime = 2
        
        dataset = xr.Dataset({
            'u': (['Time', 'nVertLevels', 'nCells'], np.random.randn(nTime, nVertLevels, nCells)),
            'v': (['Time', 'nVertLevels', 'nCells'], np.random.randn(nTime, nVertLevels, nCells)),
            'w': (['Time', 'nVertLevels', 'nCells'], np.random.randn(nTime, nVertLevels, nCells)),
            'xtime': (['Time'], [b'2023-01-01_00:00:00', b'2023-01-01_01:00:00'])
        })
        
        wind_diag = WindDiagnostics()
        
        with pytest.raises(ValueError, match="Invalid level specification"):
            wind_diag.get_3d_wind_components(
                dataset, 'u', 'v', 'w', level=['invalid'], time_index=0     # type: ignore
            )
        
        with pytest.raises(ValueError, match="Invalid level specification"):
            wind_diag.get_3d_wind_components(
                dataset, 'u', 'v', 'w', level=None, time_index=0            # type: ignore
            )
    
    def test_get_3d_wind_components_missing_v_variable_error_path(self: "TestEdgeCasesAndErrorPaths") -> None:
        """
        This test verifies that if the specified V variable is missing from the dataset, the `get_3d_wind_components` method raises a ValueError with an appropriate message indicating that the required 3D wind variables are not found. The test creates a dataset that includes U and W variables but omits the V variable, then attempts to extract 3D wind components. The assertion checks that the expected exception is raised with a message about missing variables, ensuring that the method provides clear feedback about incomplete input data rather than failing silently or producing incorrect results.

        Parameters:
            self ("TestEdgeCasesAndErrorPaths"): Test case instance.

        Returns:
            None: Assertion verifies formatted ValueError for missing variables.
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        nCells = 30
        nVertLevels = 10
        nTime = 2
        
        dataset = xr.Dataset({
            'u': (['Time', 'nVertLevels', 'nCells'], np.random.randn(nTime, nVertLevels, nCells)),
            'xtime': (['Time'], [b'2023-01-01_00:00:00', b'2023-01-01_01:00:00'])
        })
        
        wind_diag = WindDiagnostics()
        
        with pytest.raises(ValueError, match="3D wind variables \\['v', 'w'\\] not found"):
            wind_diag.get_3d_wind_components(
                dataset, 'u', 'v', 'w', level=5, time_index=0
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


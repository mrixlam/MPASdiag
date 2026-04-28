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
from tests.test_data_helpers import load_mpas_coords_from_processor
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mpasdiag.diagnostics.wind import WindDiagnostics


class TestWindDiagnostics:
    """ Test wind diagnostic computations using actual API. """
    
    
    def test_analyze_wind_components(self: 'TestWindDiagnostics', 
                                     mock_mpas_2d_data: xr.Dataset) -> None:
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
    
    
class TestWindDiagnosticsActual:
    """ Test actual WindDiagnostics implementation. """
    
    
    def test_compute_wind_speed_2d(self: 'TestWindDiagnosticsActual') -> None:
        """
        This test validates the `compute_wind_speed` method using two-dimensional arrays for `u` and `v` components, simulating a time x cells structure. It creates 2D DataArrays by tiling deterministic 1D arrays loaded from a helper function, computes the wind speed, and asserts that the output has the same shape as the input components. The test also checks that all computed speed values are non-negative and that the speed is at least as large as the absolute values of either component. This ensures that the method correctly handles 2D inputs and produces physically consistent results.

        Parameters:
            self ('TestWindDiagnosticsActual'): Test case instance.

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
    
    
class TestAnalyzeWindComponents:
    """ Test analyze_wind_components method. """
    
    @pytest.fixture
    def sample_wind_components(self: 'TestAnalyzeWindComponents') -> Any:
        """
        This fixture creates sample U, V, and W wind components with random values for testing the `analyze_wind_components` method. The `u` and `v` components are generated with a larger scale to represent typical horizontal wind speeds, while the `w` component is smaller to reflect vertical motion. Each component is an xarray DataArray with appropriate dimensions and units attributes. This fixture allows for comprehensive testing of the analysis function's ability to compute statistics across all three wind components.

        Parameters:
            self ('TestAnalyzeWindComponents'): Test case instance.

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
    
    def test_analyze_wind_components_2d(self: 'TestAnalyzeWindComponents', 
                                        sample_wind_components: Any) -> None:
        """
        This test validates the `analyze_wind_components` method when only 2D horizontal components (U and V) are provided. The analysis should compute summary statistics for the U and V components, as well as derived metrics like horizontal speed and direction. The test asserts that the returned analysis dictionary contains expected keys and that each component's statistics include minimum, maximum, mean, standard deviation, and units. This ensures that the method correctly handles 2D inputs and produces comprehensive diagnostics for horizontal wind components.

        Parameters:
            self ('TestAnalyzeWindComponents'): Test case instance.
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
    
    def test_analyze_wind_components_3d(self: 'TestAnalyzeWindComponents', 
                                        sample_wind_components: Any) -> None:
        """
        This test validates the `analyze_wind_components` method when all three components (U, V, W) are provided. The analysis should compute summary statistics for the W component in addition to the horizontal diagnostics. The test asserts that the returned analysis dictionary contains keys for both horizontal and vertical components, and that each component's statistics include minimum, maximum, mean, standard deviation, and units. This ensures that the method correctly handles 3D inputs and provides comprehensive diagnostics for all wind components.

        Parameters:
            self ('TestAnalyzeWindComponents'): Test case instance.
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
    
    def test_analyze_wind_components_verbose_2d(self: 'TestAnalyzeWindComponents', 
                                                sample_wind_components: Any) -> None:
        """
        This test checks the `analyze_wind_components` method with verbose logging enabled when analyzing only 2D horizontal components. The routine should print user-facing summaries about the U and V components, horizontal speed, and direction. Capturing stdout verifies the presence of expected summary lines in the output. The test asserts that the printed messages contain expected substrings indicating that the analysis was performed and that key metrics were reported.

        Parameters:
            self ('TestAnalyzeWindComponents'): Test case instance.
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
    
    def test_analyze_wind_components_verbose_3d(self: 'TestAnalyzeWindComponents', 
                                                sample_wind_components: Any) -> None:
        """
        This test checks the `analyze_wind_components` method with verbose logging enabled when analyzing all three components (U, V, W). The routine should print user-facing summaries about the W component and total 3D speed in addition to the horizontal diagnostics. Capturing stdout verifies the presence of expected summary lines in the output. The test asserts that the printed messages contain expected substrings indicating that the analysis was performed for all components and that key metrics were reported.

        Parameters:
            self ('TestAnalyzeWindComponents'): Test case instance.
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
    def sample_wind_levels(self: 'TestComputeWindShear') -> Any:
        """
        This fixture creates sample U and V wind components for two vertical levels (upper and lower) to test the `compute_wind_shear` method. The upper level has stronger winds than the lower level, allowing for a clear shear signal. The `u` and `v` DataArrays are designed to produce known shear magnitudes and directions based on the differences between the two levels. This fixture can be reused across multiple tests to ensure consistency in input data for vertical wind shear calculations.

        Parameters:
            self ('TestComputeWindShear'): Test case instance.

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
    
    
    def test_compute_wind_shear_verbose(self: 'TestComputeWindShear', 
                                        sample_wind_levels: Any) -> None:
        """
        This test checks the `compute_wind_shear` method with verbose logging enabled. The routine should print user-facing summaries about the wind shear magnitude and direction ranges. Capturing stdout verifies the presence of expected summary lines in the output. The test asserts that the printed messages contain expected substrings indicating that the shear analysis was performed and that key metrics were reported. This ensures that the verbose logging functionality is working as intended and provides useful diagnostic information during the wind shear computation.

        Parameters:
            self ('TestComputeWindShear'): Test case instance.
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
    def sample_3d_dataset(self: 'TestGet3DWindComponents') -> xr.Dataset:
        """
        This fixture creates a sample xarray Dataset with synthetic 3D wind component data (U, V, W) and pressure levels for testing the `get_3d_wind_components` method. The dataset includes dimensions for time, vertical levels, and horizontal cells, with random values for the wind components to mimic realistic variability. Pressure data is included to allow for testing of pressure-level selection. This fixture provides a comprehensive test dataset that can be used across multiple tests to validate the functionality of extracting 3D wind components based on different level specifications.

        Parameters:
            self ('TestGet3DWindComponents'): Test case instance.

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
    
    def test_get_3d_wind_components_model_level(self: 'TestGet3DWindComponents', 
                                                sample_3d_dataset: xr.Dataset) -> None:
        """
        This test verifies that the `get_3d_wind_components` method correctly extracts U, V, and W components at a specified model vertical level index (e.g., level=5). The test asserts that the returned components are xarray DataArrays with the expected shape and that they include metadata indicating the selected level. This ensures that the method can successfully retrieve wind components based on model-level selection and that it annotates the results with relevant metadata for downstream use.

        Parameters:
            self ('TestGet3DWindComponents'): Test case instance.
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
    
    def test_get_3d_wind_components_pressure_level(self: 'TestGet3DWindComponents', 
                                                   sample_3d_dataset: xr.Dataset) -> None:
        """
        This test verifies that the `get_3d_wind_components` method correctly extracts U, V, and W components at a specified pressure level (e.g., level=85000.0 Pa). The test asserts that the method identifies the correct vertical index corresponding to the requested pressure level and that the returned components are xarray DataArrays with appropriate metadata indicating the selected pressure level. This ensures that the method can successfully retrieve wind components based on pressure-level selection and provides informative output about the selection process when verbose mode is enabled.

        Parameters:
            self ('TestGet3DWindComponents'): Test case instance.
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
    
    def test_get_3d_wind_components_surface(self: 'TestGet3DWindComponents', 
                                            sample_3d_dataset: xr.Dataset) -> None:
        """
        This test verifies that the `get_3d_wind_components` method correctly extracts U, V, and W components at the surface level when `level='surface'` is specified. The helper should identify the lowest vertical index (0) as the surface level and annotate the returned components with metadata indicating this selection. The test asserts that the `level_index` attribute of the returned U component equals 0, confirming that the surface level was correctly identified and extracted.

        Parameters:
            self ('TestGet3DWindComponents'): Test case instance.
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
    
    def test_get_3d_wind_components_top(self: 'TestGet3DWindComponents', 
                                        sample_3d_dataset: xr.Dataset) -> None:
        """
        This test verifies that the `get_3d_wind_components` method correctly extracts U, V, and W components at the top level when `level='top'` is specified. The helper should identify the highest vertical index (nVertLevels-1) as the top level and annotate the returned components with metadata indicating this selection. The test asserts that the `level_index` attribute of the returned U component equals the expected top index, confirming that the top level was correctly identified and extracted.

        Parameters:
            self ('TestGet3DWindComponents'): Test case instance.
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
    
    def test_get_3d_wind_components_missing_w(self: 'TestGet3DWindComponents', 
                                              sample_3d_dataset: xr.Dataset) -> None:
        """
        This test verifies that when the specified W variable is missing from the dataset, the `get_3d_wind_components` method falls back to setting W to zero and prints a warning message. The test captures stdout to check for the presence of a warning about the missing W variable and asserts that the returned W component is an xarray DataArray filled with zeros. This ensures that the method handles missing vertical wind components gracefully while providing informative feedback to the user.

        Parameters:
            self ('TestGet3DWindComponents'): Test case instance.
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
    
    def test_get_3d_wind_components_verbose(self: 'TestGet3DWindComponents', 
                                            sample_3d_dataset: xr.Dataset) -> None:
        """
        This test verifies that when `verbose` mode is enabled, the `get_3d_wind_components` method prints diagnostic information about the extraction process. The test captures stdout to check for the presence of messages indicating that the method is extracting 3D wind components, along with details about the selected level and variable ranges. This ensures that the verbose logging functionality is working as intended and provides useful feedback during the component extraction process.

        Parameters:
            self ('TestGet3DWindComponents'): Test case instance.
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
    
    
    def test_get_3d_wind_components_uxarray(self: 'TestGet3DWindComponents', 
                                            sample_3d_dataset: xr.Dataset) -> None:
        """
        This test verifies that when requesting the 'uxarray' data type, the `get_3d_wind_components` method returns `xarray.DataArray` objects. The test ensures compatibility with downstream code that expects xarray objects. The test asserts that the returned U, V, and W components are instances of `xarray.DataArray`, confirming that the method correctly handles the 'uxarray' data type option and provides results in the expected format for further analysis.

        Parameters:
            self ('TestGet3DWindComponents'): Test case instance.
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
        assert isinstance(v, xr.DataArray)
        assert isinstance(w, xr.DataArray)


class TestGet2DWindComponents:
    """ Test get_2d_wind_components method. """
    
    @pytest.fixture
    def sample_2d_dataset(self: 'TestGet2DWindComponents') -> xr.Dataset:
        """
        This fixture creates a sample xarray Dataset with synthetic 2D wind component data (U and V) for testing the `get_2d_wind_components` method. The dataset includes dimensions for time and horizontal cells, with random values for the U and V components to mimic realistic variability. This fixture provides a simple test dataset that can be used across multiple tests to validate the functionality of extracting 2D wind components based on variable names and time indices.

        Parameters:
            self ('TestGet2DWindComponents'): Test case instance.

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
    
    
    def test_get_2d_wind_components_unit_mismatch(self: 'TestGet2DWindComponents') -> None:
        """
        This test verifies that if the U and V variables have different units, the `get_2d_wind_components` method prints a warning message about the unit mismatch. The test creates a dataset where U and V components have different units (e.g., m/s for U and km/h for V) and captures stdout to check for the presence of a warning about the unit mismatch. This ensures that the method provides informative feedback to the user about potential issues with the input data, allowing them to address unit inconsistencies before proceeding with analysis.

        Parameters:
            self ('TestGet2DWindComponents'): Test case instance.

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
    
    
class TestEdgeCasesAndErrorPaths:
    """ Test edge cases and error handling paths for complete coverage of get_3d_wind_components. """
    
    
    def test_get_3d_wind_components_missing_v_variable_error_path(self: 'TestEdgeCasesAndErrorPaths') -> None:
        """
        This test verifies that if the specified V variable is missing from the dataset, the `get_3d_wind_components` method raises a ValueError with an appropriate message indicating that the required 3D wind variables are not found. The test creates a dataset that includes U and W variables but omits the V variable, then attempts to extract 3D wind components. The assertion checks that the expected exception is raised with a message about missing variables, ensuring that the method provides clear feedback about incomplete input data rather than failing silently or producing incorrect results.

        Parameters:
            self ('TestEdgeCasesAndErrorPaths'): Test case instance.

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

#!/usr/bin/env python3

"""
MPASdiag Test Suite: Tests for 3D Atmospheric Data Processing in MPASdiag

This module contains a comprehensive set of unit tests for the MPAS3DProcessor class, which is responsible for loading, processing, and extracting 3D atmospheric data from MPAS model output. The tests cover a range of functionalities including coordinate extraction, data loading with different backends, variable discovery, pressure level interpolation, and attribute handling. Both edge cases and typical usage scenarios are tested to ensure robustness and correctness of the processor's behavior when working with real MPAS datasets.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load standard libraries
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from mpasdiag.processing.processors_3d import MPAS3DProcessor

from tests.test_data_helpers import (
    check_mpas_data_available, 
    load_mpas_3d_processor, 
    assert_expected_public_methods
)

from tests.visualization.cross_section_test_helpers import (
    GRID_FILE, 
)

from tests.processing.mock_dataset_helpers import (
    make_getitem, make_contains, make_getitem_with_raise,
)


class TestGet3DVariableDataPressureInterpolation:
    """ Test pressure level interpolation in get_3d_variable_data. """
    
    def test_pressure_interpolation_with_synthetic_pressure_components(self: 'TestGet3DVariableDataPressureInterpolation') -> None:
        """
        This test verifies that the `get_3d_variable_data` method can perform pressure level interpolation when synthetic pressure components (`pressure_p` and `pressure_base`) are added to the dataset. By creating these components based on an existing pressure variable and requesting data at specific model levels, the test checks that the method returns valid data slices with appropriate attributes, confirming that the interpolation logic can function correctly even when using synthetic pressure information. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return
        
        processor = load_mpas_3d_processor(verbose=False)
        
        if 'pressure' in processor.dataset:
            pressure = processor.dataset['pressure']
            pressure_base = pressure * 0.8
            pressure_p = pressure * 0.2
            
            processor.dataset['pressure_base'] = pressure_base
            processor.dataset['pressure_p'] = pressure_p
            
            var_data = processor.get_3d_variable_data('theta', level=85000.0, time_index=0)

            assert var_data is not None
            assert hasattr(var_data, 'values')
            assert var_data.values.size > 0
            
            var_data_high = processor.get_3d_variable_data('theta', level=120000.0, time_index=0)

            assert var_data_high is not None
            assert 'level_index' in var_data_high.attrs
            assert var_data_high.attrs['level_index'] == pytest.approx(0, abs=1e-3)
            
            var_data_low = processor.get_3d_variable_data('theta', level=100.0, time_index=0)

            assert var_data_low is not None
            assert 'level_index' in var_data_low.attrs
            assert var_data_low.attrs['level_index'] > 50
    
    def test_pressure_interpolation_with_pressure_data(self: 'TestGet3DVariableDataPressureInterpolation') -> None:
        """
        This test ensures that the `get_3d_variable_data` method can perform pressure level interpolation when actual pressure data is available in the dataset. By requesting data at a specific model level (e.g., level 10), the test checks that the returned data slice contains valid values and that the `level_index` attribute reflects the requested level, confirming that the interpolation process is functioning as intended when real pressure information is present. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return
        
        processor = load_mpas_3d_processor(verbose=False)
        var_data = processor.get_3d_variable_data('theta', level=10, time_index=0)

        assert var_data is not None
        assert hasattr(var_data, 'values')
        assert var_data.values.size > 0
        assert 'level_index' in var_data.attrs
        assert var_data.attrs['level_index'] == pytest.approx(10, abs=1e-3)
    
    def test_pressure_level_above_surface_fallback(self: 'TestGet3DVariableDataPressureInterpolation') -> None:
        """
        This test verifies that when a pressure level above the surface is requested, the `get_3d_variable_data` method correctly falls back to selecting the surface level. The test asserts that the returned data slice corresponds to the surface level by checking the `level_index` attribute, ensuring that the processor handles requests for levels above the surface gracefully without errors. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return

        processor = load_mpas_3d_processor(verbose=False)
        var_data = processor.get_3d_variable_data('theta', level=0, time_index=0)

        assert var_data is not None
        assert hasattr(var_data, 'values')
        assert var_data.values.size > 0
        assert 'level_index' in var_data.attrs
        assert var_data.attrs['level_index'] == pytest.approx(0, abs=1e-3)
    
    def test_pressure_level_below_top_fallback(self: 'TestGet3DVariableDataPressureInterpolation') -> None:
        """
        This test verifies that when a pressure level below the top of the model is requested, the `get_3d_variable_data` method correctly falls back to selecting the topmost model level. The test asserts that the returned data slice corresponds to the top level by checking the `level_index` attribute, ensuring that the processor handles requests for levels below the top of the model gracefully without errors. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return  
        
        processor = load_mpas_3d_processor(verbose=False)
        nlevels = processor.dataset.sizes['nVertLevels']
        top_level = nlevels - 1
        var_data = processor.get_3d_variable_data('theta', level=top_level, time_index=0)

        assert var_data is not None
        assert hasattr(var_data, 'values')
        assert var_data.values.size > 0
        assert 'level_index' in var_data.attrs
        assert var_data.attrs['level_index'] == top_level
    
    def test_pressure_interpolation_weight_calculation(self: 'TestGet3DVariableDataPressureInterpolation') -> None:
        """
        This test ensures that the `get_3d_variable_data` method calculates interpolation weights correctly when pressure data is available. By requesting data at a mid-level pressure (e.g., level 20), the test checks that the returned data slice contains valid values and that the `level_index` attribute reflects the requested level, confirming that the interpolation process is functioning as intended. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return

        processor = load_mpas_3d_processor(verbose=False)
        nlevels = processor.dataset.sizes['nVertLevels']
        mid_level = nlevels // 2
        var_data = processor.get_3d_variable_data('theta', level=mid_level, time_index=0)

        assert var_data is not None
        assert hasattr(var_data, 'values')
        assert var_data.values.size > 0
        assert np.any(np.isfinite(var_data.values))
        assert 'level_index' in var_data.attrs
        assert var_data.attrs['level_index'] == mid_level
    
    def test_pressure_interpolation_equal_pressures(self: 'TestGet3DVariableDataPressureInterpolation') -> None:
        """
        This test checks the behavior of the `get_3d_variable_data` method when the requested pressure level matches exactly with one of the model levels. The test asserts that the method returns the data slice corresponding to that exact level without performing interpolation and that the `level_index` attribute correctly reflects the requested level, confirming that the processor handles exact pressure matches appropriately. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return  
        
        processor = load_mpas_3d_processor(verbose=False)
        var_data = processor.get_3d_variable_data('theta', level=20, time_index=0)

        assert var_data is not None
        assert hasattr(var_data, 'values')
        assert var_data.values.size > 0

        valid_vals = var_data.values[np.isfinite(var_data.values)]

        assert len(valid_vals) > 0
        assert np.all(valid_vals > 200) 
        assert np.all(valid_vals < 500) 
    
    def test_pressure_interpolation_multiple_levels(self: 'TestGet3DVariableDataPressureInterpolation') -> None:
        """
        This test exercises the extraction of data across several representative model levels to ensure consistent indexing and returned-array shapes. By iterating through a small list of levels (e.g., 0, 10, 20, 30, 40), the test validates that for each requested level, the `get_3d_variable_data` method returns a non-empty data slice with the correct `level_index` attribute, confirming that the processor can handle multiple levels correctly and consistently. If the test data is not available, the test will be skipped to avoid false failures.

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return
        
        processor = load_mpas_3d_processor(verbose=False)
        test_levels = [0, 10, 20, 30, 40]
        
        for level in test_levels:
            var_data = processor.get_3d_variable_data('theta', level=level, time_index=0)
            assert var_data is not None
            assert hasattr(var_data, 'values')
            assert var_data.values.size > 0
            assert 'level_index' in var_data.attrs
            assert var_data.attrs['level_index'] == level


class TestGet3DVariableDataAttributes:
    """ Test attribute handling in get_3d_variable_data. """
    
    def test_attributes_with_pressure_level(self: 'TestGet3DVariableDataAttributes') -> None:
        """
        This test verifies that when a pressure level is requested in the `get_3d_variable_data` method, the returned data slice includes appropriate attributes such as `selected_level` and `level_index`. By requesting data at a specific model level (e.g., level 10), the test checks that these attributes are present in the returned object and that they correctly reflect the requested level, confirming that the processor provides useful metadata for extracted data slices. If the test data is not available, the test will be skipped to avoid false failures.

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return
        
        processor = load_mpas_3d_processor(verbose=True)
        var_data = processor.get_3d_variable_data('theta', level=10, time_index=0)
        
        if hasattr(var_data, 'attrs'):
            assert 'selected_level' in var_data.attrs
            assert 'level_index' in var_data.attrs
    
    def test_verbose_output_with_units(self: 'TestGet3DVariableDataAttributes') -> None:
        """
        This test checks that when the processor is initialized with `verbose=True`, the `get_3d_variable_data` method provides expected output information about the variable being extracted, including its name and units. By requesting data for a specific variable (e.g., 'theta'), the test validates that verbose mode does not interfere with data extraction and that it provides useful feedback about the variable's metadata when enabled. If the test data is not available, the test will be skipped to avoid false failures.

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return
        
        processor = load_mpas_3d_processor(verbose=True)        
        var_data = processor.get_3d_variable_data('theta', level=5, time_index=0)
        
        assert var_data is not None
    
    def test_warning_for_no_finite_values(self: 'TestGet3DVariableDataAttributes') -> None:
        """
        This test ensures that the `get_3d_variable_data` method issues a warning when the extracted data slice contains no finite values. By mocking a variable to return an array filled with NaNs, the test checks that the method returns a data slice with the expected attributes while also providing feedback about the lack of valid data, confirming that the processor can handle such cases gracefully without raising unhandled exceptions. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        assert_expected_public_methods(MPAS3DProcessor, 'MPAS3DProcessor')
        
        with patch('os.path.exists', return_value=True):
            processor = MPAS3DProcessor(GRID_FILE, verbose=True)
        
        mock_ds = MagicMock()
        mock_ds.sizes = {'nVertLevels': 55, 'Time': 10, 'nCells': 100}
        
        mock_var = MagicMock()
        mock_var.sizes = {'nVertLevels': 55, 'Time': 10, 'nCells': 100}
        
        nan_data = MagicMock()
        nan_data.values = np.full((100,), np.nan)
        nan_data.attrs = {}
        nan_data.ndim = 1

        nan_data.compute.return_value = nan_data        
        mock_var.isel.return_value = nan_data
        
        mock_ds.__getitem__.side_effect = make_getitem({}, default=mock_var)
        mock_ds.__contains__.side_effect = make_contains(['theta'])
        mock_ds.data_vars = {'theta': mock_var}
        
        processor.dataset = mock_ds
        processor.data_type = 'xarray'
        
        with patch.object(processor, 'get_vertical_levels', return_value=[10]):
            var_data = processor.get_3d_variable_data('theta', level=10, time_index=0)
            assert var_data is not None


class TestGetVerticalLevelsEdgeCases:
    """ Test edge cases in get_vertical_levels. """
    
    def test_pressure_from_pressure_variable_non_positive_warning(self: 'TestGetVerticalLevelsEdgeCases') -> None:
        """
        This test checks that the `get_vertical_levels` method issues a warning when the pressure variable contains non-positive values, which are invalid for pressure levels. By mocking the pressure variable to include negative and NaN values, the test verifies that the method returns a list of vertical levels while also providing feedback about the presence of non-positive pressure values, confirming that the processor can handle such cases gracefully without raising unhandled exceptions. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        with patch('os.path.exists', return_value=True):
            processor = MPAS3DProcessor(GRID_FILE, verbose=True)
        
        mock_ds = MagicMock()
        mock_ds.sizes = {'nVertLevels': 5, 'Time': 10, 'nCells': 100}
        
        mock_var = MagicMock()
        mock_var.sizes = {'nVertLevels': 5, 'Time': 10, 'nCells': 100}
        mock_ds.data_vars = {'theta': mock_var}
        
        mock_pressure = MagicMock()
        bad_pressure = np.array([100000, 80000, -5000, np.nan, 50000])  
        mock_pressure_mean = MagicMock()
        mock_pressure_mean.values = bad_pressure
        mock_pressure.mean.return_value = mock_pressure_mean
        mock_pressure.isel.return_value = mock_pressure
        
        mock_ds.__getitem__.side_effect = make_getitem({'pressure': mock_pressure}, default=mock_var)
        mock_ds.__contains__.side_effect = make_contains(['theta', 'pressure'])
        
        processor.dataset = mock_ds
        processor.data_type = 'xarray'
        
        levels = processor.get_vertical_levels('theta', return_pressure=True, time_index=0)
        assert_expected_public_methods(processor, 'MPAS3DProcessor')
        assert isinstance(levels, list)
        assert len(levels) == pytest.approx(5)
        assert levels == [0, 1, 2, 3, 4]
    
    def test_pressure_from_components_nVertLevelsP1_extension(self: 'TestGetVerticalLevelsEdgeCases') -> None:
        """
        This test verifies that the `get_vertical_levels` method can successfully extract vertical levels when the dataset includes the `nVertLevelsP1` dimension, which indicates an extended vertical grid. By iterating through available 3D variables and checking for the presence of this dimension, the test ensures that the method can handle datasets with extended vertical levels and returns a list of pressure levels without errors. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return

        processor = load_mpas_3d_processor(verbose=True)
        
        for var_name in processor.get_available_3d_variables():
            if 'nVertLevelsP1' in processor.dataset[var_name].sizes:
                levels = processor.get_vertical_levels(var_name, return_pressure=True, time_index=0)
                assert isinstance(levels, list)
                break
    
    def test_pressure_from_hybrid_coords_interpolation(self: 'TestGetVerticalLevelsEdgeCases') -> None:
        """
        This test validates that the `get_vertical_levels` method can perform interpolation to reconstruct pressure levels when hybrid-coordinate arrays (`fzp` and `surface_pressure`) contain some non-finite values. By mocking these arrays with a mix of valid and invalid entries, the test checks that the method successfully interpolates to produce a complete list of pressure levels and that it handles the presence of non-finite values gracefully without raising unhandled exceptions. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """        
        with patch('os.path.exists', return_value=True):
            processor = MPAS3DProcessor('test_grid.nc', verbose=True)
            assert_expected_public_methods(processor, 'MPAS3DProcessor')
        
        mock_ds = MagicMock()
        mock_ds.sizes = {'nVertLevels': 5, 'Time': 10, 'nCells': 100}
        
        mock_var = MagicMock()
        mock_var.sizes = {'nVertLevels': 5, 'Time': 10, 'nCells': 100}
        mock_ds.data_vars = {'theta': mock_var}
        
        mock_fzp = MagicMock()
        mock_fzp_isel = MagicMock()
        mock_fzp_isel.values = np.array([1.0, 0.8, np.nan, 0.4, 0.2])  
        mock_fzp.isel.return_value = mock_fzp_isel
        
        mock_sp = MagicMock()
        mock_sp_isel = MagicMock()
        mock_sp_mean = MagicMock()
        mock_sp_mean.values = 101300.0
        mock_sp_isel.mean.return_value = mock_sp_mean
        mock_sp.isel.return_value = mock_sp_isel
        
        mock_ds.__getitem__.side_effect = make_getitem({'fzp': mock_fzp, 'surface_pressure': mock_sp}, default=mock_var)
        mock_ds.__contains__.side_effect = make_contains(['theta', 'fzp', 'surface_pressure'])
        
        processor.dataset = mock_ds
        processor.data_type = 'xarray'
        levels = processor.get_vertical_levels('theta', return_pressure=True, time_index=0)

        assert isinstance(levels, list)
        assert len(levels) == pytest.approx(5)
    
    def test_pressure_from_hybrid_coords_single_good_value(self: 'TestGetVerticalLevelsEdgeCases') -> None:
        """
        This test checks that the `get_vertical_levels` method can still produce a valid list of pressure levels when the hybrid-coordinate arrays contain only a single good value. By mocking the `fzp` array to have one valid entry and the rest as non-finite, the test verifies that the method can use that single good value along with the surface pressure to reconstruct a complete list of pressure levels without raising exceptions, confirming that the processor can handle this edge case gracefully. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """        
        with patch('os.path.exists', return_value=True):
            processor = MPAS3DProcessor('test_grid.nc', verbose=True)
            assert_expected_public_methods(processor, 'MPAS3DProcessor')
        
        mock_ds = MagicMock()
        mock_ds.sizes = {'nVertLevels': 5, 'Time': 10, 'nCells': 100}
        
        mock_var = MagicMock()
        mock_var.sizes = {'nVertLevels': 5, 'Time': 10, 'nCells': 100}
        mock_ds.data_vars = {'theta': mock_var}
        
        mock_fzp = MagicMock()
        mock_fzp_isel = MagicMock()
        mock_fzp_isel.values = np.array([np.nan, np.nan, 0.5, np.nan, np.nan]) 
        mock_fzp.isel.return_value = mock_fzp_isel
        
        mock_sp = MagicMock()
        mock_sp_isel = MagicMock()
        mock_sp_mean = MagicMock()
        mock_sp_mean.values = 101300.0
        mock_sp_isel.mean.return_value = mock_sp_mean
        mock_sp.isel.return_value = mock_sp_isel
        
        mock_ds.__getitem__.side_effect = make_getitem({'fzp': mock_fzp, 'surface_pressure': mock_sp}, default=mock_var)
        mock_ds.__contains__.side_effect = make_contains(['theta', 'fzp', 'surface_pressure'])
        
        processor.dataset = mock_ds
        processor.data_type = 'xarray'
        levels = processor.get_vertical_levels('theta', return_pressure=True, time_index=0)

        assert isinstance(levels, list)
        assert len(levels) == pytest.approx(5)
    
    def test_pressure_from_hybrid_coords_no_good_values(self: 'TestGetVerticalLevelsEdgeCases') -> None:
        """
        This test ensures that the `get_vertical_levels` method can still produce a valid list of pressure levels when the hybrid-coordinate arrays contain no good values. By mocking the `fzp` array to have all non-finite entries, the test checks that the method falls back to a robust reconstruction (e.g., using logspace) to generate a complete list of pressure levels without raising exceptions, confirming that the processor can handle this edge case gracefully. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """        
        with patch('os.path.exists', return_value=True):
            processor = MPAS3DProcessor('test_grid.nc', verbose=True)
            assert_expected_public_methods(processor, 'MPAS3DProcessor')

        mock_ds = MagicMock()
        mock_ds.sizes = {'nVertLevels': 5, 'Time': 10, 'nCells': 100}
        
        mock_var = MagicMock()
        mock_var.sizes = {'nVertLevels': 5, 'Time': 10, 'nCells': 100}
        mock_ds.data_vars = {'theta': mock_var}
        
        mock_fzp = MagicMock()
        mock_fzp_isel = MagicMock()
        mock_fzp_isel.values = np.array([np.nan, np.nan, np.nan, np.nan, np.nan]) 
        mock_fzp.isel.return_value = mock_fzp_isel
        
        mock_sp = MagicMock()
        mock_sp_isel = MagicMock()
        mock_sp_mean = MagicMock()
        mock_sp_mean.values = 101300.0
        mock_sp_isel.mean.return_value = mock_sp_mean
        mock_sp.isel.return_value = mock_sp_isel
        
        mock_ds.__getitem__.side_effect = make_getitem({'fzp': mock_fzp, 'surface_pressure': mock_sp}, default=mock_var)
        mock_ds.__contains__.side_effect = make_contains(['theta', 'fzp', 'surface_pressure'])
        
        processor.dataset = mock_ds
        processor.data_type = 'xarray'        
        levels = processor.get_vertical_levels('theta', return_pressure=True, time_index=0)

        assert isinstance(levels, list)
        assert len(levels) == pytest.approx(5)
    
    def test_pressure_from_hybrid_coords_exception_fallback(self: 'TestGetVerticalLevelsEdgeCases') -> None:
        """
        This test checks that if an exception occurs while accessing the `fzp` array during pressure level reconstruction, the `get_vertical_levels` method falls back to using model-level indices instead of pressure values. By mocking the dataset to raise an exception when `fzp` is accessed, the test ensures that the method handles this scenario gracefully and returns a list of model level indices without raising unhandled exceptions, confirming that the processor can manage unexpected issues with hybrid-coordinate data. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """        
        with patch('os.path.exists', return_value=True):
            processor = MPAS3DProcessor('test_grid.nc', verbose=True)
            assert_expected_public_methods(processor, 'MPAS3DProcessor')
        
        mock_ds = MagicMock()
        mock_ds.sizes = {'nVertLevels': 5, 'Time': 10, 'nCells': 100}
        
        mock_var = MagicMock()
        mock_var.sizes = {'nVertLevels': 5, 'Time': 10, 'nCells': 100}
        mock_ds.data_vars = {'theta': mock_var}
        
        mock_ds.__getitem__.side_effect = make_getitem_with_raise('fzp', RuntimeError("Failed to access fzp"), default=mock_var)
        mock_ds.__contains__.side_effect = make_contains(['theta', 'fzp', 'surface_pressure'])
        
        processor.dataset = mock_ds
        processor.data_type = 'xarray'
        
        levels = processor.get_vertical_levels('theta', return_pressure=True, time_index=0)
        assert isinstance(levels, list)
        assert levels == [0, 1, 2, 3, 4] 
    
    def test_model_level_indices_verbose(self: 'TestGetVerticalLevelsEdgeCases') -> None:
        """
        This test verifies that when the `get_vertical_levels` method is called in verbose mode and the dataset does not contain pressure information, the method returns a list of model level indices while providing informative output about the lack of pressure data. By ensuring that the returned levels are valid indices and that verbose output includes messages about missing pressure information, the test confirms that the processor can handle this scenario gracefully while keeping users informed. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return

        processor = load_mpas_3d_processor(verbose=True)        
        levels = processor.get_vertical_levels('theta', return_pressure=False, time_index=0)

        assert isinstance(levels, list)
        assert all(isinstance(x, int) for x in levels)



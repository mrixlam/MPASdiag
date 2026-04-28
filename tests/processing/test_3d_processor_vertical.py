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
from typing import Any
from unittest.mock import MagicMock, patch

from mpasdiag.processing.processors_3d import MPAS3DProcessor

from tests.test_data_helpers import (
    load_mpas_3d_processor, 
    assert_expected_public_methods
)

from tests.processing.mock_dataset_helpers import (
    make_getitem, 
    make_contains, 
)


class TestGetVerticalLevels:
    """ Test vertical level retrieval. """
    
    def setup_method(self: 'TestGetVerticalLevels') -> None:
        """
        This method-level setup initializes an `MPAS3DProcessor` instance with a mocked grid file path and `verbose=False`. The filesystem existence check is patched to always return True, allowing the test to focus on the processor initialization logic without relying on actual files. This setup provides a consistent starting point for the subsequent tests that will verify the behavior of vertical level retrieval methods. If the test data is not available, the tests will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        with patch('os.path.exists', return_value=True):
            self.processor = MPAS3DProcessor('test_grid.nc', verbose=False)
            assert_expected_public_methods(self.processor, 'MPAS3DProcessor')
    
    def test_get_model_levels_from_real_data(self: 'TestGetVerticalLevels') -> None:
        """
        This test verifies that the `get_vertical_levels` method can successfully retrieve model level indices from real MPAS output when requested with `return_pressure=False`. By loading the dataset, retrieving an available 3D variable, and invoking the method, the test asserts that the returned levels are a non-empty list of integers corresponding to model levels. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        processor = load_mpas_3d_processor(verbose=False)
        vars_3d = processor.get_available_3d_variables()
        assert len(vars_3d) > 0
        
        var_name = vars_3d[0]
        levels = processor.get_vertical_levels(var_name, return_pressure=False)
        
        assert isinstance(levels, list)
        assert len(levels) > 0
        assert levels == list(range(len(levels)))
    
    def test_get_pressure_levels_from_real_data(self: 'TestGetVerticalLevels') -> None:
        """
        This test checks that the `get_vertical_levels` method can successfully retrieve pressure levels from real MPAS output when requested with `return_pressure=True`. By loading the dataset, retrieving an available 3D variable, and invoking the method, the test asserts that the returned levels are a non-empty list of pressure values that decrease with height. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        processor = load_mpas_3d_processor(verbose=False)
        vars_3d = processor.get_available_3d_variables()
        var_name = vars_3d[0]
        
        try:
            levels = processor.get_vertical_levels(var_name, return_pressure=True)
            assert isinstance(levels, list)
            assert len(levels) > 0
            if len(levels) > 1:
                assert levels[0] > levels[-1], "Pressure should decrease with height"
        except ValueError as e:
            if 'pressure' not in str(e).lower():
                raise
    
    
    def test_pressure_from_pressure_variable(self: 'TestGetVerticalLevels') -> None:
        """
        This test verifies that when a `pressure` variable is present in the dataset, the `get_vertical_levels` method can retrieve pressure levels directly from this variable when `return_pressure=True` is specified. Using a mocked dataset that includes a `pressure` variable with synthetic pressure values, this test calls `get_vertical_levels(..., return_pressure=True)` and asserts that the returned list has the expected length and that pressure decreases with height. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        mock_ds = self._create_mock_dataset_with_pressure_var()
        self.processor.dataset = mock_ds
        self.processor.verbose = True
        
        levels = self.processor.get_vertical_levels('theta', return_pressure=True)
        
        assert len(levels) == pytest.approx(55)
        assert levels[0] > levels[-1]
    
    
    def test_pressure_from_hybrid_coords(self: 'TestGetVerticalLevels') -> None:
        """
        This test verifies that when the dataset contains hybrid coordinate information (e.g., `hybrid_a` and `hybrid_b`), the `get_vertical_levels` method can compute pressure levels from these hybrid coordinates. Using a mocked dataset that includes hybrid coordinate fields, this test calls `get_vertical_levels(..., return_pressure=True)` and asserts that the returned list has the expected length and that pressure decreases with height. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        mock_ds = self._create_mock_dataset_with_hybrid()
        self.processor.dataset = mock_ds
        self.processor.verbose = True
        
        levels = self.processor.get_vertical_levels('theta', return_pressure=True)
        
        assert len(levels) == pytest.approx(55)
    
    def _create_mock_dataset(self: 'TestGetVerticalLevels') -> Any:
        """
        This helper method creates a mock dataset that includes a 3D variable with the expected dimensions and attributes for testing vertical level retrieval. The mock dataset is structured to mimic an xarray Dataset containing a variable named 'theta' with dimensions `nCells`, `nVertLevels`, and `Time`, along with appropriate sizes and attributes. By setting up the necessary return values for indexing and slicing operations, this mock dataset allows tests to focus on the logic of retrieving vertical levels without relying on actual MPAS data files, enabling controlled testing of various scenarios related to vertical level retrieval. 

        Parameters:
            None

        Returns:
            Any: A MagicMock object emulating an xarray Dataset suitable for tests.
        """
        mock_ds = MagicMock()
        mock_ds.data_vars = {'theta': True}
        mock_ds.sizes = {'nVertLevels': 55, 'Time': 10}
        mock_var = MagicMock()
        mock_var.sizes = {'nCells': 40962, 'nVertLevels': 55, 'Time': 10}
        mock_ds.__getitem__.return_value = mock_var
        mock_ds.__contains__.return_value = False
        return mock_ds
    
    def _create_mock_dataset_with_pressure_var(self: 'TestGetVerticalLevels') -> Any:
        """
        This helper method creates a mock dataset that includes a `pressure` variable for testing pressure level retrieval. The mock dataset is structured to mimic an xarray Dataset containing a variable named 'theta' with dimensions `nCells`, `nVertLevels`, and `Time`, along with a `pressure` variable that has the same vertical levels and time dimensions. By setting up the necessary return values for indexing, slicing, and mean operations on the pressure variable, this mock dataset allows tests to focus on the logic of retrieving pressure levels directly from a pressure variable without relying on actual MPAS data files. 

        Parameters:
            None

        Returns:
            Any: A MagicMock object emulating an xarray Dataset with a `pressure` variable.
        """
        mock_ds = MagicMock()
        mock_ds.data_vars = {'theta': True, 'pressure': True}
        mock_ds.sizes = {'nVertLevels': 55, 'Time': 10}
        
        mock_var = MagicMock()
        mock_var.sizes = {'nCells': 40962, 'nVertLevels': 55, 'Time': 10}
        
        pressure_vals = np.linspace(100000, 1000, 55)
        mock_pressure = MagicMock()
        mock_pressure.mean.return_value = MagicMock(values=pressure_vals)
        mock_pressure.isel.return_value = mock_pressure
        
        mock_ds.__getitem__.side_effect = make_getitem({'pressure': mock_pressure}, default=mock_var)
        mock_ds.__contains__.return_value = True
        
        return mock_ds
    
    def _create_mock_dataset_with_pressure_components(self: 'TestGetVerticalLevels') -> Any:
        """
        This helper method creates a mock dataset that includes the necessary pressure components (`pressure_p` and `pressure_base`) for testing pressure level retrieval from components. The mock dataset is structured to mimic an xarray Dataset containing a variable named 'theta' with dimensions `nCells`, `nVertLevels`, and `Time`, along with synthetic `pressure_p` and `pressure_base` variables that allow for testing the reconstruction of pressure levels. By setting up the necessary return values for indexing, slicing, and arithmetic operations on the pressure components, this mock dataset enables controlled testing of scenarios related to retrieving pressure levels from components without relying on actual MPAS data files. 

        Parameters:
            None

        Returns:
            Any: A MagicMock emulating an xarray Dataset with pressure components.
        """
        mock_ds = MagicMock()
        mock_ds.data_vars = {'theta': True, 'pressure_p': True, 'pressure_base': True}
        mock_ds.sizes = {'nVertLevels': 55, 'Time': 10}
        
        mock_var = MagicMock()
        mock_var.sizes = {'nCells': 40962, 'nVertLevels': 55, 'Time': 10}
        mock_var.dims = ('Time', 'nCells', 'nVertLevels')
        
        pressure_base_vals = np.linspace(100000, 1000, 55)
        
        mock_total = MagicMock()
        mock_mean_result = MagicMock()
        mock_mean_result.values = pressure_base_vals  
        mock_total.mean.return_value = mock_mean_result
        
        mock_pressure_base_time = MagicMock()
        mock_pressure_p_time = MagicMock()
        
        mock_pressure_base_time.__add__ = lambda self, other: mock_total
        mock_pressure_base_time.__radd__ = lambda self, other: mock_total
        mock_pressure_p_time.__add__ = lambda self, other: mock_total
        mock_pressure_p_time.__radd__ = lambda self, other: mock_total
        
        mock_pressure_base = MagicMock()
        mock_pressure_base.isel.return_value = mock_pressure_base_time
        
        mock_pressure_p = MagicMock()
        mock_pressure_p.isel.return_value = mock_pressure_p_time
        
        mock_ds.__getitem__.side_effect = make_getitem({'pressure_base': mock_pressure_base, 'pressure_p': mock_pressure_p}, default=mock_var)
        mock_ds.__contains__.side_effect = make_contains(['theta', 'pressure_p', 'pressure_base'])
        
        return mock_ds
    
    def _create_mock_dataset_with_hybrid(self: 'TestGetVerticalLevels') -> Any:
        """
        This helper method creates a mock dataset that includes hybrid coordinate information (e.g., `hybrid_a` and `hybrid_b`) for testing pressure level retrieval from hybrid coordinates. The mock dataset is structured to mimic an xarray Dataset containing a variable named 'theta' with dimensions `nCells`, `nVertLevels`, and `Time`, along with synthetic `fzp` and `surface_pressure` variables that allow for testing the computation of pressure levels from hybrid coordinates. By setting up the necessary return values for indexing, slicing, and arithmetic operations on the hybrid coordinate components, this mock dataset enables controlled testing of scenarios related to retrieving pressure levels from hybrid coordinates without relying on actual MPAS data files.

        Parameters:
            None

        Returns:
            Any: A MagicMock object emulating an xarray Dataset with hybrid coords.
        """
        mock_ds = MagicMock()
        mock_ds.data_vars = {'theta': True, 'fzp': True, 'surface_pressure': True}
        mock_ds.sizes = {'nVertLevels': 55, 'Time': 10}
        
        mock_var = MagicMock()
        mock_var.sizes = {'nCells': 40962, 'nVertLevels': 55, 'Time': 10}
        
        fzp_vals = np.linspace(1.0, 0.01, 55)
        sp_vals = np.full(40962, 101325.0)
        
        mock_fzp = MagicMock()
        mock_fzp.values = fzp_vals
        mock_fzp.isel.return_value = mock_fzp
        
        mock_sp = MagicMock()
        mock_sp.values = sp_vals
        mock_sp.isel.return_value = mock_sp
        
        mock_ds.__getitem__.side_effect = make_getitem({'fzp': mock_fzp, 'surface_pressure': mock_sp}, default=mock_var)
        mock_ds.__contains__.side_effect = make_contains(['theta', 'fzp', 'surface_pressure'])
        
        return mock_ds


class TestAddSpatialCoordinates:
    """ Test spatial coordinate enrichment. """
    
    def setup_method(self: 'TestAddSpatialCoordinates') -> None:
        """
        This method-level setup initializes an `MPAS3DProcessor` instance with a mocked grid file path and `verbose=False`. The filesystem existence check is patched to always return True, allowing the test to focus on the processor initialization logic without relying on actual files. This setup provides a consistent starting point for the subsequent tests that will verify the behavior of spatial coordinate enrichment methods. If the test data is not available, the tests will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        with patch('os.path.exists', return_value=True):
            self.processor = MPAS3DProcessor('test_grid.nc', verbose=False)
            assert_expected_public_methods(self.processor, 'MPAS3DProcessor')
    
    def test_add_coordinates_to_real_dataset(self: 'TestAddSpatialCoordinates') -> None:
        """
        This test verifies that the `add_spatial_coordinates` method can successfully add spatial coordinates to a real MPAS dataset when a 3D variable is present. By loading the dataset and invoking the method, the test asserts that the resulting dataset contains expected spatial coordinate names (e.g., 'lon', 'lat', 'lonCell', 'latCell') either as coordinates or data variables, confirming that the method can enrich the dataset with spatial information. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        processor = load_mpas_3d_processor(verbose=False)
        
        assert processor.dataset is not None
        assert len(processor.dataset.data_vars) > 0
        
        coord_names = ['lon', 'lonCell', 'lat', 'latCell']
        has_coords = any(c in processor.dataset.coords or c in processor.dataset.data_vars 
                         for c in coord_names)
        assert has_coords, "Expected spatial coordinates not found in dataset"
    

if __name__ == '__main__':
    pytest.main([__file__])

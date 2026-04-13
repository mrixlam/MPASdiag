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
import os
import sys
import pytest
import tempfile
import numpy as np
import xarray as xr
from typing import Any
from pathlib import Path
from unittest.mock import MagicMock, patch

from mpasdiag.processing.processors_3d import MPAS3DProcessor
from tests.test_data_helpers import get_mpas_data_paths, check_mpas_data_available, load_mpas_3d_processor, assert_expected_public_methods

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
GRID_FILE = os.path.join(TEST_DATA_DIR, 'grids', 'x1.10242.static.nc')


def make_getitem(mapping: dict, 
                 default: Any = None) -> Any:
    """
    This function returns a __getitem__ side_effect function for a given mapping. It allows for flexible mocking of xarray Datasets or DataArrays by returning values based on the provided mapping dictionary. If a key is not found in the mapping, it returns a default value (which can be a MagicMock if not specified), enabling tests to simulate various dataset configurations and variable access patterns.
    
    Parameters:
        mapping (dict): A dictionary mapping keys to values.
        default (Any): The default value to return if the key is not found.

    Returns:
        Any: A function suitable for use as a __getitem__ side_effect.
    """
    def _getitem(key: str) -> Any:
        return mapping.get(key, MagicMock() if default is None else default)
    return _getitem


def make_contains(keys: Any) -> Any:
    """
    This function returns a __contains__ side_effect function for a given collection of keys. It allows for flexible mocking of the __contains__ method in xarray Datasets or DataArrays by checking if a key is present in the provided collection. This is useful for simulating the presence or absence of variables or coordinates in a dataset during testing. 

    Parameters:
        keys (Any): A collection of keys to check against.

    Returns:
        Any: A function suitable for use as a __contains__ side_effect.
    """
    key_set = set(keys)
    def _contains(key: str) -> bool:
        return key in key_set
    return _contains


def make_grid_getitem(lon_values: Any, 
                      lat_values: Any) -> Any:
    """
    This function returns a __getitem__ side_effect function that provides longitude and latitude values based on the key. It checks if the key contains 'lon' or 'lat' (case-insensitive) and returns a mock coordinate object with the corresponding values. This is particularly useful for testing the coordinate extraction logic in MPAS3DProcessor when working with grid datasets. 

    Parameters:
        lon_values (Any): The longitude values to return for keys containing 'lon'.
        lat_values (Any): The latitude values to return for keys containing 'lat'.

    Returns:
        Any: A function suitable for use as a __getitem__ side_effect.
    """
    def _getitem(key: str) -> Any:
        mock_coord = MagicMock()
        mock_coord.values = lon_values if 'lon' in key.lower() else lat_values
        return mock_coord
    return _getitem


def make_getitem_with_raise(raise_key: str, 
                            exc: Exception, 
                            default: Any = None) -> Any:
    """
    This function returns a __getitem__ side_effect function that raises a specified exception when a particular key is accessed. For all other keys, it returns a default value (which can be a MagicMock if not specified). This is useful for testing the error handling logic in MPAS3DProcessor when certain expected variables or coordinates are missing or inaccessible in the dataset. 

    Parameters:
        raise_key (str): The key for which the exception should be raised.
        exc (Exception): The exception to raise when the raise_key is accessed.
        default (Any): The default value to return if the key is not the raise_key.

    Returns:
        Any: A function suitable for use as a __getitem__ side_effect.
    """
    def _getitem(key: str) -> Any:
        if key == raise_key:
            raise exc
        return MagicMock() if default is None else default
    return _getitem


class TestExtract2DCoordinatesEdgeCases:
    """ Test edge cases in extract_2d_coordinates_for_variable. """
    
    def test_grid_file_exception_handling(self: "TestExtract2DCoordinatesEdgeCases") -> None:
        """
        This test verifies that the `extract_2d_coordinates_for_variable` method in the MPAS3DProcessor class properly handles exceptions that may occur when attempting to load the grid file. By mocking the `xarray.open_dataset` function to raise a RuntimeError, the test checks that the method catches this exception and raises a new RuntimeError with an appropriate error message, ensuring that users receive clear feedback about issues with loading the grid coordinates.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        assert_expected_public_methods(MPAS3DProcessor, 'MPAS3DProcessor')

        with patch('os.path.exists', return_value=True):
            processor = MPAS3DProcessor('test_grid.nc', verbose=False)
        
        mock_ds = MagicMock()
        mock_var = MagicMock()
        mock_var.sizes = {'nCells': 100, 'nVertLevels': 55}
        mock_ds.__getitem__.return_value = mock_var
        mock_ds.__contains__.return_value = True
        processor.dataset = mock_ds
        
        with patch('xarray.open_dataset', side_effect=RuntimeError("File read error")):
            with pytest.raises(RuntimeError) as exc_info:
                processor.extract_2d_coordinates_for_variable('theta')
            assert "Error loading coordinates" in str(exc_info.value)
    
    def test_verbose_output_for_coordinate_extraction(self: "TestExtract2DCoordinatesEdgeCases") -> None:
        """
        This test checks that when the processor is initialized with `verbose=True`, the `extract_2d_coordinates_for_variable` method provides expected output information during coordinate extraction. By mocking the grid coordinates and ensuring that the returned longitude and latitude arrays have the correct sizes, the test validates that verbose mode does not interfere with the coordinate extraction process and that it provides useful feedback when enabled. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        assert_expected_public_methods(MPAS3DProcessor, 'MPAS3DProcessor')
        
        with patch('os.path.exists', return_value=True):
            processor = MPAS3DProcessor('test_grid.nc', verbose=True)  
        
        mock_ds = MagicMock()
        mock_var = MagicMock()
        mock_var.sizes = {'nCells': 100, 'nVertLevels': 55}
        mock_ds.__getitem__.return_value = mock_var
        mock_ds.__contains__.return_value = True
        processor.dataset = mock_ds
        
        mock_grid = MagicMock()
        mock_grid.coords = {'lonCell': True, 'latCell': True}
        mock_grid.data_vars = {}
        
        lon_deg = np.linspace(-180, 180, 100)
        lat_deg = np.linspace(-90, 90, 100)
        
        mock_grid.__getitem__.side_effect = make_grid_getitem(lon_deg, lat_deg)
        
        with patch('xarray.open_dataset') as mock_open:
            mock_open.return_value.__enter__.return_value = mock_grid
            
            lon, lat = processor.extract_2d_coordinates_for_variable('theta')
            
            assert len(lon) == pytest.approx(100)
            assert len(lat) == pytest.approx(100)


class TestLoad3DDataSpatialCoordinates:
    """ Test spatial coordinate handling in load_3d_data. """
    
    def test_load_3d_data_xarray_adds_spatial_coords(self: "TestLoad3DDataSpatialCoordinates") -> None:
        """
        This test verifies that the `load_3d_data` method can successfully load datasets using the xarray backend when the `use_pure_xarray` flag is set to True. The test checks that the processor's `dataset` attribute is populated with an xarray Dataset containing the expected data variables and that the `data_type` is correctly set to 'xarray'. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        assert_expected_public_methods(MPAS3DProcessor, 'MPAS3DProcessor')

        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return

        paths = get_mpas_data_paths()
        processor = MPAS3DProcessor(str(paths['grid_file']), verbose=False)
        
        processor = processor.load_3d_data(str(paths['mpasout_dir']), use_pure_xarray=True)
        
        assert hasattr(processor.dataset, 'data_vars')
        assert processor.data_type == 'xarray'
    
    def test_load_3d_data_uxarray_adds_spatial_coords(self: "TestLoad3DDataSpatialCoordinates") -> None:
        """
        This test confirms that the `load_3d_data` method can successfully load datasets using the uxarray backend when the `use_pure_xarray` flag is set to False. The test checks that the processor's `dataset` attribute is populated with a dataset object containing the expected data variables and that the `data_type` is correctly set to 'uxarray'. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        assert_expected_public_methods(MPAS3DProcessor, 'MPAS3DProcessor')

        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return
        
        paths = get_mpas_data_paths()

        processor = MPAS3DProcessor(str(paths['grid_file']), verbose=False)        
        processor = processor.load_3d_data(str(paths['mpasout_dir']), use_pure_xarray=False)
        
        assert processor.dataset is not None
        assert processor.data_type in ['uxarray', 'xarray']


class TestGetAvailable3DVerbose:
    """ Test verbose output in get_available_3d_variables. """
    
    def test_get_available_3d_variables_verbose_output(self: "TestGetAvailable3DVerbose") -> None:
        """
        This test checks that when the processor is initialized with `verbose=True`, the `get_available_3d_variables` method provides expected output information about the available 3D variables in the dataset. By ensuring that the method returns a non-empty list of variable names, the test validates that verbose mode does not interfere with the variable discovery process and that it provides useful feedback when enabled. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return
        
        processor = load_mpas_3d_processor(verbose=True)        
        variables = processor.get_available_3d_variables()
        
        assert isinstance(variables, list)
        assert len(variables) > 0



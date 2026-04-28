#!/usr/bin/env python3
"""
MPASdiag Test Suite: Tests for MPAS Data Processing Utilities and Processors

This module contains a comprehensive set of unit tests for the data processing components of the MPASdiag package, specifically targeting the MPAS2DProcessor class and associated utility functions. The tests cover initialization, file discovery, datetime parsing, time parameter validation, spatial coordinate extraction, and error handling scenarios. Both synthetic inputs and real MPAS mesh data from fixtures are used to ensure robust validation of geographic extent checks, longitude normalization, accumulation hour parsing, and dataset-oriented helpers. Mocking is employed to isolate functionality and prevent actual file I/O during testing. This suite ensures the reliability and correctness of data processing operations critical for accurate visualization and analysis of MPAS model output.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules
import os
import sys
import pytest
import tempfile
import numpy as np
import xarray as xr
from typing import Any
from pathlib import Path

from tests.test_data_helpers import assert_expected_public_methods
from mpasdiag.processing.processors_2d import MPAS2DProcessor

package_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_dir))


class TestDataValidation:
    """ Tests for data validation and filtering methods in MPAS2DProcessor. """
    
    def test_filter_by_spatial_extent(self: 'TestDataValidation', 
                                      mock_mpas_mesh: Any, 
                                      mock_mpas_2d_data: Any) -> None:
        """
        This test verifies that the filter_by_spatial_extent method correctly filters data based on specified geographic bounds. It uses real MPAS mesh data from the provided fixture to create a dataset with longitude and latitude coordinates. The test defines a smaller geographic extent within the bounds of the mesh and calls filter_by_spatial_extent with this extent. It then asserts that the returned mask is a boolean array of the correct length and that it correctly identifies which grid cells fall within the specified geographic bounds. This ensures that the method can accurately filter data for visualization or analysis based on spatial criteria. 

        Parameters:
            mock_mpas_mesh: Fixture providing real MPAS mesh with coordinates.
            mock_mpas_2d_data: Fixture providing real 2D diagnostic data.

        Returns:
            None
        """
        grid_file = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name

        try:
            mock_mpas_mesh.to_netcdf(grid_file)            
            processor = MPAS2DProcessor(grid_file, verbose=False)
            assert_expected_public_methods(processor, 'MPAS2DProcessor')
            
            lon_rad = mock_mpas_mesh['lonCell'].values
            lat_rad = mock_mpas_mesh['latCell'].values
            
            if np.nanmax(np.abs(lon_rad)) <= 2 * np.pi + 1e-6:
                lon = np.degrees(lon_rad)
            else:
                lon = lon_rad

            if np.nanmax(np.abs(lat_rad)) <= np.pi / 2 + 1e-6:
                lat = np.degrees(lat_rad)
            else:
                lat = lat_rad
            
            lon = ((lon + 180.0) % 360.0) - 180.0
            
            nCells = len(lon)
            
            if 't2m' in mock_mpas_2d_data:
                data_array = mock_mpas_2d_data['t2m'].isel(Time=0)
            else:
                data_array = xr.DataArray(np.ones(nCells), dims=['nCells'])
            
            ds = xr.Dataset({
                'lonCell': (['nCells'], lon),
                'latCell': (['nCells'], lat),
            })

            processor.dataset = ds
            
            lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
            lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
            
            lon_margin = (lon_max - lon_min) * 0.2
            lat_margin = (lat_max - lat_min) * 0.2

            test_lon_min = lon_min + lon_margin
            test_lon_max = lon_max - lon_margin
            test_lat_min = lat_min + lat_margin
            test_lat_max = lat_max - lat_margin
            
            filtered_data, mask = processor.filter_by_spatial_extent(
                data_array, test_lon_min, test_lon_max, test_lat_min, test_lat_max
            )
            
            assert mask.dtype == bool
            assert len(mask) == nCells
            
            expected_mask = ((lon >= test_lon_min) & (lon <= test_lon_max) & 
                           (lat >= test_lat_min) & (lat <= test_lat_max))
            
            np.testing.assert_array_equal(mask, expected_mask)
            
        finally:
            os.unlink(grid_file)


if __name__ == '__main__':
    pytest.main([__file__])

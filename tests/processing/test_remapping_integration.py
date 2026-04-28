#!/usr/bin/env python3
"""
MPASdiag Test Suite: Tests for MPASdiag remapping functionality

This module contains integration tests for the remapping functionality in MPASdiag, specifically focusing on the end-to-end remapping process using real MPAS data. The tests verify that the remapping functions can successfully remap data defined on a real MPAS grid to a regular lat-lon grid, and that the main remapping class can be used

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules for testing
import pytest
import numpy as np
import xarray as xr
from typing import Any
from pathlib import Path

from mpasdiag.processing.remapping import (
    MPASRemapper,
    remap_mpas_to_latlon
)


class TestWithRealMPASData:
    """ Integration tests using actual MPAS data files. """
    
    def test_remap_mpas_to_latlon_with_real_data(self: 'TestWithRealMPASData', mpas_test_data: Any) -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can successfully remap a synthetic dataset defined on a real MPAS grid to a regular lat-lon grid. The test extracts a small subset of longitude and latitude data from the provided MPAS dataset, creates synthetic data values, and then calls the remapping function. The assertions check that the output is an xarray DataArray and that it contains the expected longitude and latitude coordinates, confirming that the remapping process produces the expected structured output. 

        Parameters:
            self ('TestWithRealMPASData'): Test instance (unused).
            mpas_test_data (Any): Fixture returning an xarray Dataset.

        Returns:
            None: Assertions validate the remapped DataArray contains expected coords.
        """
        n_test = min(1000, len(mpas_test_data['lonCell']))
        
        lon = mpas_test_data['lonCell'].isel(nCells=slice(0, n_test))
        lat = mpas_test_data['latCell'].isel(nCells=slice(0, n_test))
        
        data = np.random.randn(n_test)
        
        result = remap_mpas_to_latlon(
            data=data,
            lon=lon,
            lat=lat,
            resolution=2.0,
            method='nearest'
        )
        
        assert isinstance(result, xr.DataArray)
        assert 'lon' in result.coords
        assert 'lat' in result.coords
    
    def test_mpas_remapper_with_real_data(self: 'TestWithRealMPASData', mpas_test_data: Any, temp_weights_dir: Path) -> None:
        """
        This test verifies that the `MPASRemapper` class can be used to remap a synthetic dataset defined on a real MPAS grid to a regular lat-lon grid. The test first extracts a small subset of longitude and latitude data from the provided MPAS dataset, creates synthetic data values, and then uses the `MPASRemapper` class to perform the remapping. The assertions check that the output is an xarray DataArray, confirming that the remapping process produces the expected structured output when using the class-based approach. 

        Parameters:
            self ('TestWithRealMPASData'): Test instance (unused).
            mpas_test_data (Any): Fixture returning an xarray Dataset.
            temp_weights_dir (Path): Temporary directory for weight files.

        Returns:
            None: Assertions validate the remapped DataArray type.
        """
        n_test = min(500, len(mpas_test_data['lonCell']))
        
        lon = mpas_test_data['lonCell'].isel(nCells=slice(0, n_test)).values
        lat = mpas_test_data['latCell'].isel(nCells=slice(0, n_test)).values
        
        data = np.random.randn(n_test)
        
        data_array, grid_dataset = MPASRemapper.unstructured_to_structured_grid(
            data=data,
            lon=lon,
            lat=lat,
            intermediate_resolution=2.0
        )
        
        remapper = MPASRemapper(method='nearest_s2d', weights_dir=temp_weights_dir, reuse_weights=False)
        remapper.source_grid = grid_dataset

        remapper.create_target_grid(
            lon_min=np.degrees(lon.min()),
            lon_max=np.degrees(lon.max()),
            dlon=3.0,
            dlat=3.0
        )

        remapper.build_regridder()        
        result = remapper.remap(data_array)
        
        assert isinstance(result, xr.DataArray)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

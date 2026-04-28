#!/usr/bin/env python3
"""
MPASdiag Test Suite: Precipitation Diagnostics Tests

This module contains unit and integration tests for the precipitation diagnostics components of MPASdiag. The tests are designed to verify that the `PrecipitationDiagnostics` class can be imported, initialized, and used to compute precipitation differences from synthetic datasets. The tests also check for the presence or absence of expected methods in the diagnostics API and validate basic properties of the computed precipitation fields. This suite serves as a foundation for more comprehensive testing as the precipitation diagnostics functionality is developed.

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
import matplotlib
import numpy as np
import xarray as xr
matplotlib.use('Agg')
from typing import Any
from io import StringIO
from unittest.mock import patch

from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestComputePrecipitationDifference:
    """ Test compute_precipitation_difference method. """
    
    @pytest.fixture
    def sample_dataset(self: 'TestComputePrecipitationDifference', 
                       mpas_2d_processor_diag: Any) -> Any:
        """
        This fixture provides a sample dataset for testing the `compute_precipitation_difference` method. It uses real MPAS 2D processor diagnostics data if available, allowing tests to run with realistic input. If the MPAS data is not available, the fixture will skip the test. This approach ensures that tests can be executed in environments with access to MPAS data while still allowing for graceful handling when data is missing.
        
        Parameters:
            self ('TestComputePrecipitationDifference'): Test case instance.
            mpas_2d_processor_diag: Fixture providing MPAS 2D processor diagnostics.

        Returns:
            Any: Sample MPAS dataset for testing.
        """
        if mpas_2d_processor_diag is None:
            pytest.skip("MPAS data not available")
            return 
        
        return mpas_2d_processor_diag.dataset
    
    def test_compute_difference_basic(self: 'TestComputePrecipitationDifference', 
                                      sample_dataset: Any) -> None:
        """
        This test verifies that the `compute_precipitation_difference` method can be called with a sample dataset and that it returns a valid `xarray.DataArray`. It checks that the method executes without errors and that the output is of the expected type. This serves as a basic functional test for the difference computation logic, ensuring that it can process real MPAS data and produce an output.

        Parameters:
            self ('TestComputePrecipitationDifference'): Test case instance.
            sample_dataset (Any): Fixture providing a synthetic MPAS dataset for testing.

        Returns:
            None: Assertions validate output type, shape, and metadata.
        """
        diag = PrecipitationDiagnostics(verbose=False)

        result = diag.compute_precipitation_difference(
            sample_dataset, time_index=5, var_name='rainnc', accum_period='a01h'
        )
        
        assert isinstance(result, xr.DataArray)
        assert len(result.shape) == pytest.approx(1)
        assert result.shape[0] > 0  
        assert 'units' in result.attrs
        assert result.attrs['units'] == 'mm'
        assert 'accumulation_period' in result.attrs
        assert result.attrs['accumulation_hours'] == pytest.approx(1)
    
    def test_compute_difference_verbose(self: 'TestComputePrecipitationDifference', 
                                        sample_dataset: Any) -> None:
        """
        This test checks that the `compute_precipitation_difference` method produces verbose output when the diagnostics object is initialized with `verbose=True`. It captures the standard output during the method execution and verifies that expected logging messages are present. This ensures that the verbose mode is functioning correctly and provides useful information during computations.

        Parameters:
            self ('TestComputePrecipitationDifference'): Test case instance.
            sample_dataset (Any): Fixture providing a synthetic MPAS dataset for testing.

        Returns:
            None: Assertions check captured output and result type.
        """
        diag = PrecipitationDiagnostics(verbose=True)
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            result = diag.compute_precipitation_difference(
                sample_dataset, time_index=5, var_name='rainnc', accum_period='a01h'
            )
        
        output = captured_output.getvalue()

        assert 'Computing' in output or 'hour accumulation' in output
        assert isinstance(result, xr.DataArray)
    
    def test_compute_difference_total_precipitation(self: 'TestComputePrecipitationDifference', 
                                                    sample_dataset: Any) -> None:
        """
        This test verifies that the `compute_precipitation_difference` method can compute total precipitation by combining convective and non-convective components. It checks that the method returns a valid `xarray.DataArray` with non-negative values when given a dataset containing `rainc` and `rainnc`. This test ensures that the method correctly handles the 'total' var_name case and produces physically reasonable results. It serves as an integration test for the difference computation logic.

        Parameters:
            self ('TestComputePrecipitationDifference'): Test case instance.
            sample_dataset (Any): Fixture providing a synthetic MPAS dataset for testing.

        Returns:
            None: Assertions validate output type, shape, and non-negativity.
        """
        diag = PrecipitationDiagnostics(verbose=False)

        result = diag.compute_precipitation_difference(
            sample_dataset, time_index=5, var_name='total', accum_period='a01h'
        )
        
        assert isinstance(result, xr.DataArray)
        assert len(result.shape) == pytest.approx(1)
        assert result.shape[0] > 0  
        assert np.all(result.values >= 0)
    
    def test_compute_difference_all_accumulation_periods(self: 'TestComputePrecipitationDifference', 
                                                         sample_dataset: Any) -> None:
        """
        This test checks that the `compute_precipitation_difference` method can compute differences for all supported accumulation periods. It iterates through a list of standard accumulation period codes and verifies that the method returns a valid `xarray.DataArray` with expected metadata for each period. This ensures that the method correctly handles different lookback periods and that the output includes necessary attributes for interpretation.

        Parameters:
            self ('TestComputePrecipitationDifference'): Test case instance.
            sample_dataset (Any): Fixture providing a synthetic MPAS dataset for testing.

        Returns:
            None: Assertions verify each period returns a DataArray with expected metadata.
        """
        diag = PrecipitationDiagnostics(verbose=False)
        periods = ['a01h', 'a03h', 'a06h', 'a12h', 'a24h']
        
        for period in periods:
            result = diag.compute_precipitation_difference(
                sample_dataset, time_index=5, var_name='rainnc', accum_period=period
            )

            assert isinstance(result, xr.DataArray)
            assert 'accumulation_period' in result.attrs
    
    
    def test_compute_difference_time_index_zero(self: 'TestComputePrecipitationDifference', 
                                                sample_dataset: Any) -> None:
        """
        This test verifies that the `compute_precipitation_difference` method can handle a `time_index` of 0, which typically represents the first timestep in the dataset. The method should return a valid `xarray.DataArray` even when there is no prior data to compute a difference, often by using a fallback approach. This test ensures that the method can gracefully handle edge cases at the start of the time series and still produce an output with expected properties.

        Parameters:
            self ('TestComputePrecipitationDifference'): Test case instance.
            sample_dataset (Any): Fixture providing a synthetic MPAS dataset.

        Returns:
            None: Assertions validate output type and shape for time_index 0.
        """
        diag = PrecipitationDiagnostics(verbose=False)

        result = diag.compute_precipitation_difference(
            sample_dataset, time_index=0, var_name='rainnc', accum_period='a01h'
        )
        
        assert isinstance(result, xr.DataArray)
        assert len(result.shape) == pytest.approx(1)
        assert result.shape[0] > 0  
    
    def test_compute_difference_insufficient_lookback(self: 'TestComputePrecipitationDifference', 
                                                      sample_dataset: Any) -> None:
        """
        This test checks that the `compute_precipitation_difference` method can handle cases where the specified `time_index` does not have sufficient prior data for the requested accumulation period. For example, if `time_index=2` is used with an accumulation period of 'a06h' that requires 6 hours of lookback, the method should detect the insufficient data and return a valid `xarray.DataArray` with appropriate metadata indicating the issue. This ensures that the method can gracefully handle edge cases where lookback requirements cannot be met and still provide informative output.

        Parameters:
            self ('TestComputePrecipitationDifference'): Test case instance.
            sample_dataset (Any): Fixture providing a synthetic MPAS dataset.

        Returns:
            None: Assertions verify fallback behavior when lookback is insufficient.
        """
        diag = PrecipitationDiagnostics(verbose=False)

        result = diag.compute_precipitation_difference(
            sample_dataset, time_index=2, var_name='rainnc', accum_period='a06h'
        )
        
        assert isinstance(result, xr.DataArray)

        if 'note' in result.attrs:
            assert 'Insufficient' in result.attrs['note']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

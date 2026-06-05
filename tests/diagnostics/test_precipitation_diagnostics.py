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
import pandas as pd
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
            mpas_2d_processor_diag (Any): Fixture providing MPAS 2D processor diagnostics.

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
            sample_dataset (Any): Fixture providing a synthetic MPAS dataset for testing.

        Returns:
            None
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
            sample_dataset (Any): Fixture providing a synthetic MPAS dataset for testing.

        Returns:
            None
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
            sample_dataset (Any): Fixture providing a synthetic MPAS dataset for testing.

        Returns:
            None
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
            sample_dataset (Any): Fixture providing a synthetic MPAS dataset for testing.

        Returns:
            None
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
            sample_dataset (Any): Fixture providing a synthetic MPAS dataset.

        Returns:
            None
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
            sample_dataset (Any): Fixture providing a synthetic MPAS dataset.

        Returns:
            None
        """
        diag = PrecipitationDiagnostics(verbose=False)

        result = diag.compute_precipitation_difference(
            sample_dataset, time_index=2, var_name='rainnc', accum_period='a06h'
        )
        
        assert isinstance(result, xr.DataArray)

        if 'note' in result.attrs:
            assert 'Insufficient' in result.attrs['note']


class TestPrecipitationTimeMetadata:
    """ Tests that compute_precipitation_difference attaches valid-time metadata for downstream annotation. """

    @staticmethod
    def _make_dataset(n_time: int = 4, 
                      n_cells: int = 6) -> xr.Dataset:
        """
        This helper method creates a synthetic dataset with a `Time` coordinate and `rainc`/`rainnc` variables for testing the time metadata handling in the `compute_precipitation_difference` method. The dataset simulates a simple time series of precipitation data across a specified number of cells, allowing tests to verify that valid-time and accumulation start time metadata are correctly attached to the computed differences. 

        Parameters:
            n_time (int): Number of hourly time steps.
            n_cells (int): Number of mesh cells.

        Returns:
            xr.Dataset: Dataset with `Time` datetime coordinate and `rainc`/`rainnc` variables.
        """
        rng = np.random.default_rng(0)
        times = pd.date_range('2024-09-16T21:00:00', periods=n_time, freq='1h')
        rainc = np.cumsum(rng.uniform(0, 5, (n_time, n_cells)), axis=0)
        rainnc = np.cumsum(rng.uniform(0, 5, (n_time, n_cells)), axis=0)
        return xr.Dataset(
            {
                'rainc': (['Time', 'nCells'], rainc),
                'rainnc': (['Time', 'nCells'], rainnc),
            },
            coords={'Time': ('Time', times)},
        )

    def test_valid_time_and_window_start_attached(self: 'TestPrecipitationTimeMetadata') -> None:
        """
        This test verifies that the `compute_precipitation_difference` method correctly attaches `valid_time` and `accumulation_start_time` attributes to the output `xarray.DataArray`. It checks that the `valid_time` corresponds to the timestamp of the selected time index and that the `accumulation_start_time` reflects the correct lookback period based on the specified accumulation period. This ensures that the method provides accurate time metadata for downstream use in annotations and interpretations.

        Parameters:
            None

        Returns:
            None
        """
        ds = self._make_dataset()
        times = ds['Time'].values
        diag = PrecipitationDiagnostics(verbose=False)

        result = diag.compute_precipitation_difference(
            ds, time_index=2, var_name='total', accum_period='a01h'
        )

        assert result.attrs['valid_time'] == pd.Timestamp(times[2]).isoformat()
        assert result.attrs['accumulation_start_time'] == pd.Timestamp(times[1]).isoformat()

    def test_window_start_tracks_accumulation_period(self: 'TestPrecipitationTimeMetadata') -> None:
        """
        This test checks that the `accumulation_start_time` attribute correctly tracks the accumulation period specified in the `compute_precipitation_difference` method. For example, if an accumulation period of 'a03h' is used, the `accumulation_start_time` should reflect a timestamp that is 3 hours prior to the `valid_time`. This ensures that the method accurately calculates and attaches time metadata based on the accumulation period, which is crucial for correct interpretation of the precipitation differences. 

        Parameters:
            None

        Returns:
            None
        """
        ds = self._make_dataset()
        times = ds['Time'].values
        diag = PrecipitationDiagnostics(verbose=False)

        result = diag.compute_precipitation_difference(
            ds, time_index=3, var_name='rainnc', accum_period='a03h'
        )

        assert result.attrs['valid_time'] == pd.Timestamp(times[3]).isoformat()
        assert result.attrs['accumulation_start_time'] == pd.Timestamp(times[0]).isoformat()

    def test_first_time_step_has_valid_time_without_window_start(self: 'TestPrecipitationTimeMetadata') -> None:
        """
        This test verifies that when the `compute_precipitation_difference` method is called with `time_index=0`, it still attaches a `valid_time` attribute corresponding to the first timestamp, but does not include an `accumulation_start_time` since there is no prior data to define a lookback window. This ensures that the method can handle edge cases at the start of the time series and provides appropriate metadata for interpretation. 

        Parameters:
            None

        Returns:
            None
        """
        ds = self._make_dataset()
        times = ds['Time'].values
        diag = PrecipitationDiagnostics(verbose=False)

        result = diag.compute_precipitation_difference(
            ds, time_index=0, var_name='rainnc', accum_period='a01h'
        )

        assert result.attrs['valid_time'] == pd.Timestamp(times[0]).isoformat()
        assert 'accumulation_start_time' not in result.attrs


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

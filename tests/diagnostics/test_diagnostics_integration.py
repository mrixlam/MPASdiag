#!/usr/bin/env python3
"""
MPASdiag Test Suite: Diagnostics Integration Tests

This module contains integration tests for the diagnostics components of MPASdiag, specifically focusing on the wind and precipitation diagnostics. The tests are designed to verify that the diagnostics can be used together in a workflow, that they produce consistent outputs, and that they handle realistic data correctly. The tests include both smoke tests for basic functionality and more comprehensive tests using synthetic datasets that mimic real-world conditions.

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

from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestDiagnosticsIntegration:
    """ Integration tests for diagnostics modules. """
    
    def test_wind_and_precipitation_together(self: 'TestDiagnosticsIntegration', 
                                             mock_mpas_2d_data: Any) -> None:
        """
        This test verifies that the wind and precipitation diagnostics can be executed together on the same dataset without errors. It checks that both diagnostics produce outputs and that the dimensions of the outputs are consistent. This ensures that the diagnostics can be used in a combined workflow, which is common in meteorological analyses.

        Parameters:
            mock_mpas_2d_data (Any): Fixture for 2D diagnostic variables including wind and precipitation from diag files.

        Returns:
            None
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
        
        wind_diag = WindDiagnostics(verbose=False)
        precip_diag = PrecipitationDiagnostics(verbose=False)
        
        u = mock_mpas_2d_data['u10'].isel(Time=0).compute()
        v = mock_mpas_2d_data['v10'].isel(Time=0).compute()
        speed = wind_diag.compute_wind_speed(u, v)
        
        rainnc = mock_mpas_2d_data['rainnc'].isel(Time=0)
        tp = precip_diag.compute_precipitation_difference(
            mock_mpas_2d_data, time_index=0, var_name='rainnc', accum_period='a01h'
        )   

        assert speed is not None
        assert rainnc is not None
        assert tp is not None
        assert speed.shape[0] == rainnc.shape[0]  


class TestIntegrationWithRealData:
    """ Integration tests using realistic precipitation and wind data. """
    
    def test_full_workflow_1hour_accumulation(self: 'TestIntegrationWithRealData') -> None:
        """
        This test simulates a realistic workflow for computing 1-hour precipitation accumulation from cumulative data. It creates synthetic cumulative precipitation data that mimics real-world conditions, then uses the PrecipitationDiagnostics to compute the 1-hour difference. The test checks that the output is a DataArray with the expected shape and that all values are non-negative, which is a physical requirement for precipitation.

        Parameters:
            None

        Returns:
            None
        """
        nCells = 100
        nTime = 24
        
        hourly_rates = np.random.exponential(2.0, (nTime, nCells))
        rainnc_data = np.cumsum(hourly_rates, axis=0)
        
        ds = xr.Dataset({
            'rainnc': (['Time', 'nCells'], rainnc_data),
        })
        
        diag = PrecipitationDiagnostics(verbose=False)
        
        for t in range(1, 24, 6):
            result = diag.compute_precipitation_difference(
                ds, time_index=t, var_name='rainnc', accum_period='a01h'
            )
            
            assert isinstance(result, xr.DataArray)
            assert result.shape == (nCells,)
            assert np.all(result.values >= 0)
    
    def test_full_workflow_multiple_accumulation_periods(self: 'TestIntegrationWithRealData') -> None:
        """
        This test evaluates the precipitation diagnostic's ability to compute differences for multiple accumulation periods (1h, 3h, 6h, 12h, 24h) using synthetic cumulative data. It creates a dataset with cumulative precipitation that increases over time and checks that the computed differences have the correct metadata indicating the accumulation period. This ensures that the diagnostic can handle various periods and that the output is properly annotated.

        Parameters:
            None

        Returns:
            None
        """
        nCells = 50
        nTime = 25
        
        hourly_rates = np.random.rand(nTime, nCells) * 5
        rainnc_data = np.cumsum(hourly_rates, axis=0)
        rainc_data = np.cumsum(np.random.rand(nTime, nCells) * 2, axis=0)
        
        ds = xr.Dataset({
            'rainnc': (['Time', 'nCells'], rainnc_data),
            'rainc': (['Time', 'nCells'], rainc_data),
        })
        
        diag = PrecipitationDiagnostics(verbose=False)
        
        periods = ['a01h', 'a03h', 'a06h', 'a12h', 'a24h']

        for period in periods:
            result = diag.compute_precipitation_difference(
                ds, time_index=24, var_name='rainnc', accum_period=period
            )
            
            assert isinstance(result, xr.DataArray)
            assert 'accumulation_period' in result.attrs
            assert result.attrs['accumulation_period'] == period

    def test_full_workflow_2d_wind(self: 'TestIntegrationWithRealData') -> None:
        """
        Integration test for a complete 2D wind diagnostic workflow. It creates realistic 10m wind data, extracts components, computes speed and direction, and runs analysis summaries. This end-to-end check validates multiple helper functions work together on a synthetic dataset. The test asserts types and basic properties of resulting analysis.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        nCells = 200
        nTime = 24
        
        u10_data = np.random.randn(nTime, nCells) * 5 + 2  
        v10_data = np.random.randn(nTime, nCells) * 5 + 1 
        
        ds = xr.Dataset({
            'u10': (['Time', 'nCells'], u10_data, {'units': 'm s^{-1}'}),
            'v10': (['Time', 'nCells'], v10_data, {'units': 'm s^{-1}'}),
        })
        
        diag = WindDiagnostics(verbose=False)
        
        u, v = diag.get_2d_wind_components(ds, 'u10', 'v10', time_index=12)
        
        speed = diag.compute_wind_speed(u, v)
        direction = diag.compute_wind_direction(u, v, degrees=True)
        
        analysis = diag.analyze_wind_components(u, v)
        
        assert isinstance(speed, xr.DataArray)
        assert isinstance(direction, xr.DataArray)
        assert isinstance(analysis, dict)
        assert 'horizontal_speed' in analysis
        assert analysis['horizontal_speed']['mean'] > 0
    
    def test_full_workflow_3d_wind_with_shear(self: 'TestIntegrationWithRealData') -> None:
        """
        Integration test for a 3D wind workflow that includes shear calculation. The test constructs synthetic 3D fields, extracts components at two levels, and computes vertical shear magnitude and direction. This validates combined operation of extraction and shear algorithms on multi-level datasets. The test asserts types and non-negativity of shear magnitude.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        nCells = 100
        nVertLevels = 20
        nTime = 10
        
        u_data = np.random.randn(nTime, nVertLevels, nCells) * 10
        v_data = np.random.randn(nTime, nVertLevels, nCells) * 10
        w_data = np.random.randn(nTime, nVertLevels, nCells) * 0.5
        
        ds = xr.Dataset({
            'uReconstructZonal': (['Time', 'nVertLevels', 'nCells'], u_data),
            'vReconstructMeridional': (['Time', 'nVertLevels', 'nCells'], v_data),
            'w': (['Time', 'nVertLevels', 'nCells'], w_data),
        })
        
        diag = WindDiagnostics(verbose=False)
        
        u_upper, v_upper, _ = diag.get_3d_wind_components(
            ds, 'uReconstructZonal', 'vReconstructMeridional', 'w',
            level=10, time_index=0
        )
        
        u_lower, v_lower, _ = diag.get_3d_wind_components(
            ds, 'uReconstructZonal', 'vReconstructMeridional', 'w',
            level=0, time_index=0
        )
        
        shear_mag, shear_dir = diag.compute_wind_shear(
            u_upper, v_upper, u_lower, v_lower
        )
        
        assert isinstance(shear_mag, xr.DataArray)
        assert isinstance(shear_dir, xr.DataArray)
        assert np.all(shear_mag.values >= 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

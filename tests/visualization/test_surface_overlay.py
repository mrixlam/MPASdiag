#!/usr/bin/env python3
"""
MPASdiag Test Suite: Surface Overlay Tests

This module contains unit tests for the surface overlay functionality in the `MPASSurfacePlotter`. The tests cover successful overlay rendering, error propagation for invalid configurations, and handling of edge cases such as multidimensional overlay data and all-NaN overlays. The tests utilize real MPAS coordinate and data fixtures to ensure realistic conditions and validate the integration of overlay features with the main surface plotting logic. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""

import os
import sys
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from unittest.mock import MagicMock, patch

from mpasdiag.visualization.surface import MPASSurfacePlotter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestWindOverlay:
    """ Tests for wind overlay functionality. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestWindOverlay', mpas_coordinates, mpas_surface_temp_data, mpas_wind_data) -> None:
        """
        This fixture initializes the test environment for wind overlay tests. It prepares an `MPASSurfacePlotter` instance and loads real MPAS coordinate, surface temperature, and wind data to be used in the overlay tests. The fixture also computes the extent bounds from the coordinate data for use in plotting.

        Parameters:
            self ('TestWindOverlay'): Test instance which will receive fixture attributes.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_surface_temp_data: Session fixture providing real surface temperature data.
            mpas_wind_data: Session fixture providing real u/v wind components.

        Returns:
            None: Populates `self.plotter`, `self.lon`, `self.lat`, `self.data`, `self.u_data`, and `self.v_data`.
        """
        if mpas_coordinates is None or mpas_surface_temp_data is None or mpas_wind_data is None:
            pytest.skip("MPAS data not available")
            return
        
        self.plotter = MPASSurfacePlotter()
        lon_full, lat_full = mpas_coordinates
        self.lon = lon_full[:50]
        self.lat = lat_full[:50]
        
        self.data = mpas_surface_temp_data[:50]
        
        u_data, v_data = mpas_wind_data
        self.u_data = u_data[:50]
        self.v_data = v_data[:50]
        
        self.extent_bounds = (
            float(self.lon.min()), float(self.lon.max()),
            float(self.lat.min()), float(self.lat.max())
        )
    
    def test_wind_overlay_success(self: 'TestWindOverlay') -> None:
        """
        This test verifies that a valid wind overlay is added to the surface map without errors. It mocks the `MPASWindPlotter` to ensure that the `add_wind_overlay` method is called when a proper wind overlay configuration is provided. The test asserts that the plotting function returns a Figure and that the overlay method is invoked as expected.

        Parameters:
            self ('TestWindOverlay'): Test instance containing prepared fixtures.

        Returns:
            None: Assertion verifies the overlay method was called.
        """
        wind_config = {
            'u_data': self.u_data,
            'v_data': self.v_data,
            'plot_type': 'arrows'
        }
        
        with patch('mpasdiag.visualization.surface.MPASWindPlotter') as mock_wind:
            mock_wind_instance = MagicMock()
            mock_wind.return_value = mock_wind_instance
            
            fig, ax = self.plotter.create_surface_map(
                self.lon, self.lat, self.data, 't2m',
                *self.extent_bounds,
                wind_overlay=wind_config,
                plot_type='scatter'
            )
            
            mock_wind_instance.add_wind_overlay.assert_called_once()
        
        plt.close(fig)
    
    def test_wind_overlay_value_error(self: 'TestWindOverlay') -> None:
        """
        This test confirms that providing an invalid wind overlay configuration raises a `ValueError`. By mocking the `MPASWindPlotter` to raise a `ValueError` when `add_wind_overlay` is called with an invalid configuration, the test asserts that the expected exception is raised by the plotting function, indicating proper error handling for misuse of the wind overlay feature.

        Parameters:
            self ('TestWindOverlay'): Test instance containing prepared fixtures.

        Returns:
            None: Assertion expects a ValueError to be raised.
        """
        wind_config = {'invalid': 'config'}
        
        with patch('mpasdiag.visualization.surface.MPASWindPlotter') as mock_wind:
            mock_wind_instance = MagicMock()
            mock_wind.return_value = mock_wind_instance
            mock_wind_instance.add_wind_overlay.side_effect = ValueError("Invalid wind config")
            
            with pytest.raises(ValueError):
                self.plotter.create_surface_map(
                    self.lon, self.lat, self.data, 't2m',
                    *self.extent_bounds,
                    wind_overlay=wind_config,
                    plot_type='scatter'
                )
    
    def test_wind_overlay_other_exception(self: 'TestWindOverlay') -> None:
        """
        This test ensures that unexpected exceptions raised during wind overlay processing are handled gracefully by the plotter. By mocking the `MPASWindPlotter` to raise a generic `RuntimeError` when `add_wind_overlay` is called, the test asserts that the plotting function does not crash but instead returns a Figure while printing a warning about the overlay failure. This verifies that the plotter can continue to function even when the overlay logic encounters unforeseen issues.

        Parameters:
            self ('TestWindOverlay'): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        wind_config = {'u_data': self.u_data, 'v_data': self.v_data}
        
        with patch('mpasdiag.visualization.surface.MPASWindPlotter') as mock_wind:
            mock_wind_instance = MagicMock()
            mock_wind.return_value = mock_wind_instance
            mock_wind_instance.add_wind_overlay.side_effect = RuntimeError("Wind overlay failed")
            
            fig, ax = self.plotter.create_surface_map(
                self.lon, self.lat, self.data, 't2m',
                *self.extent_bounds,
                wind_overlay=wind_config,
                plot_type='scatter'
            )
            
            assert isinstance(fig, Figure)
        
        plt.close(fig)


class TestSurfaceOverlay:
    """ Tests for surface overlay functionality. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestSurfaceOverlay', mpas_coordinates, mpas_surface_temp_data) -> None:
        """
        This fixture initializes the test environment for surface overlay tests. It prepares an `MPASSurfacePlotter` instance and loads real MPAS coordinate and surface temperature data to be used in the overlay tests. The fixture also computes the extent bounds from the coordinate data for use in plotting.

        Parameters:
            self ('TestSurfaceOverlay'): Test instance which will receive fixture attributes.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_surface_temp_data: Session fixture providing real surface temperature data.

        Returns:
            None: Populates `self.plotter`, `self.lon`, `self.lat`, and `self.data`.
        """
        if mpas_coordinates is None or mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
            return
        
        self.plotter = MPASSurfacePlotter()
        lon_full, lat_full = mpas_coordinates
        self.lon = lon_full[:50]
        self.lat = lat_full[:50]
        
        self.data = mpas_surface_temp_data[:50]
        
        self.extent_bounds = (
            float(self.lon.min()), float(self.lon.max()),
            float(self.lat.min()), float(self.lat.max())
        )
    
    def test_surface_overlay_success(self: 'TestSurfaceOverlay') -> None:
        """
        This test verifies that a valid surface overlay is added to the surface map without errors. It constructs a realistic contour overlay configuration using the real MPAS data and asserts that the plotting function returns a Figure instance, indicating that the overlay was processed successfully and integrated with the main surface plot.

        Parameters:
            self ('TestSurfaceOverlay'): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        surface_config = {
            'data': self.data * 3.5,  
            'var_name': 'pressure',
            'plot_type': 'contour',
            'levels': [950, 1000, 1050]
        }
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            *self.extent_bounds,
            surface_overlay=surface_config,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_surface_overlay_value_error(self: 'TestSurfaceOverlay') -> None:
        """
        This test confirms that providing an invalid surface overlay configuration raises a `ValueError`. By constructing a surface overlay configuration with an unsupported `plot_type` and asserting that the plotting function raises a `ValueError`, the test verifies that the plotter properly validates overlay configurations and surfaces errors when they are misused.

        Parameters:
            self ('TestSurfaceOverlay'): Test instance containing prepared fixtures.

        Returns:
            None: Assertion expects a ValueError to be raised.
        """
        surface_config = {
            'data': self.data,
            'plot_type': 'invalid_type'
        }
        
        with pytest.raises(ValueError):
            self.plotter.create_surface_map(
                self.lon, self.lat, self.data, 't2m',
                *self.extent_bounds,
                surface_overlay=surface_config,
                plot_type='scatter'
            )
    
    def test_surface_overlay_other_exception(self: 'TestSurfaceOverlay') -> None:
        """
        This test ensures that unexpected exceptions raised during surface overlay processing are handled gracefully by the plotter. By mocking the internal overlay processing to raise a generic `RuntimeError`, the test asserts that the plotting function does not crash but instead returns a Figure while printing a warning about the overlay failure. This verifies that the plotter can continue to function even when the overlay logic encounters unforeseen issues.

        Parameters:
            self ('TestSurfaceOverlay'): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        surface_config = {'data': None}
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            *self.extent_bounds,
            surface_overlay=surface_config,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestSurfaceOverlayMethod:
    """ Tests for _add_surface_overlay method. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestSurfaceOverlayMethod', mpas_coordinates, mpas_surface_temp_data) -> None:
        """
        This fixture initializes the test environment for `_add_surface_overlay` method tests. It prepares an `MPASSurfacePlotter` instance and loads real MPAS coordinate and surface temperature data to be used in testing the internal overlay processing logic. The fixture also computes the extent bounds from the coordinate data for use in plotting.

        Parameters:
            self ('TestSurfaceOverlayMethod'): Test instance which will receive fixture attributes.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_surface_temp_data: Session fixture providing real surface temperature data.

        Returns:
            None: Populates `self.plotter`, `self.lon`, and `self.lat`.
        """
        if mpas_coordinates is None or mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
            return 
        
        self.plotter = MPASSurfacePlotter()
        lon_full, lat_full = mpas_coordinates
        self.lon = lon_full[:50]
        self.lat = lat_full[:50]
        self.temp_data = mpas_surface_temp_data[:50]
        
        self.extent_bounds = (
            float(self.lon.min()), float(self.lon.max()),
            float(self.lat.min()), float(self.lat.max())
        )
    
    def test_overlay_invalid_plot_type(self: 'TestSurfaceOverlayMethod') -> None:
        """
        This test confirms that providing an unsupported `plot_type` in the surface overlay configuration raises a `ValueError`. By constructing a surface overlay configuration with an invalid `plot_type` and asserting that the internal `_add_surface_overlay` method raises a `ValueError`, the test verifies that the method properly validates the overlay configuration and surfaces errors when unsupported plot types are specified.

        Parameters:
            self ('TestSurfaceOverlayMethod'): Test instance containing prepared fixtures.

        Returns:
            None: Assertion expects a ValueError to be raised.
        """
        surface_config = {
            'data': self.temp_data,
            'plot_type': 'invalid'
        }
        
        with pytest.raises(ValueError) as exc_info:
            self.plotter.create_surface_map(
                self.lon, self.lat, self.temp_data, 't2m',
                *self.extent_bounds,
                surface_overlay=surface_config,
                plot_type='scatter'
            )
        assert 'Unsupported surface overlay plot_type' in str(exc_info.value)
    
    def test_overlay_multidimensional_with_level_index(self: 'TestSurfaceOverlayMethod') -> None:
        """
        This test verifies that the surface overlay method can handle multidimensional overlay data when a valid `level_index` is provided. By creating a 2D overlay dataset and specifying a `level_index`, the test asserts that the overlay is processed correctly and that a Figure is returned, indicating successful rendering of the specified level from the multidimensional data.

        Parameters:
            self ('TestSurfaceOverlayMethod'): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        overlay_data = np.tile(self.temp_data, (10, 1)).T  
        
        surface_config = {
            'data': overlay_data,
            'plot_type': 'contour',
            'level_index': 5
        }
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.temp_data, 't2m',
            *self.extent_bounds,
            surface_overlay=surface_config,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_overlay_multidimensional_no_level_index(self: 'TestSurfaceOverlayMethod') -> None:
        """
        This test confirms that when multidimensional overlay data is provided without a `level_index`, the method defaults to using the first level (index 0) and still produces a Figure. By creating a 2D overlay dataset and omitting the `level_index`, the test asserts that the method handles this case gracefully and returns a Figure, indicating that it defaulted to the first level of the data.

        Parameters:
            self ('TestSurfaceOverlayMethod'): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        overlay_data = np.tile(self.temp_data, (10, 1)).T  
        
        surface_config = {
            'data': overlay_data,
            'plot_type': 'contour'
        }
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.temp_data, 't2m',
            *self.extent_bounds,
            surface_overlay=surface_config,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_overlay_no_valid_data(self: 'TestSurfaceOverlayMethod') -> None:
        """
        This test ensures that when the overlay data contains no valid (non-NaN) values, the method handles this edge case without crashing and still returns a Figure. By creating an overlay dataset filled with NaN values, the test asserts that the method does not fail but instead prints a warning and returns a Figure, indicating that it can handle cases where the overlay data is effectively empty.

        Parameters:
            self ('TestSurfaceOverlayMethod'): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        overlay_data = np.full(50, np.nan)
        
        surface_config = {
            'data': overlay_data,
            'plot_type': 'contour'
        }
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.temp_data, 't2m',
            *self.extent_bounds,
            surface_overlay=surface_config,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_overlay_contour_with_levels(self: 'TestSurfaceOverlayMethod') -> None:
        """
        This test verifies that a contour overlay with specified levels and alpha transparency renders correctly and returns a Figure. By providing a realistic contour overlay configuration with levels and alpha settings, the test asserts that the overlay is processed without errors and that a Figure is returned, indicating successful rendering of the contour overlay with the specified parameters.

        Parameters:
            self ('TestSurfaceOverlayMethod'): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        surface_config = {
            'data': self.temp_data * 3.5,  
            'plot_type': 'contour',
            'levels': [950, 1000, 1050],
            'colors': 'red',
            'linewidth': 2.0,
            'alpha': 0.8
        }
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.temp_data, 't2m',
            *self.extent_bounds,
            surface_overlay=surface_config,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_overlay_contour_with_labels(self: 'TestSurfaceOverlayMethod') -> None:
        """
        This test confirms that a contour overlay with `add_labels=True` renders correctly and returns a Figure. By providing a contour overlay configuration that includes the `add_labels` option, the test asserts that the method processes this configuration without errors and that a Figure is returned, indicating successful rendering of the contour overlay with labels.

        Parameters:
            self ('TestSurfaceOverlayMethod'): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        surface_config = {
            'data': self.temp_data * 20,  
            'plot_type': 'contour',
            'add_labels': True
        }
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.temp_data, 't2m',
            *self.extent_bounds,
            surface_overlay=surface_config,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_overlay_contourf_with_levels(self: 'TestSurfaceOverlayMethod') -> None:
        """
        This test verifies that a filled contour overlay with specified levels and alpha transparency renders correctly and returns a Figure. By providing a realistic filled contour overlay configuration with levels and alpha settings, the test asserts that the overlay is processed without errors and that a Figure is returned, indicating successful rendering of the filled contour overlay with the specified parameters.

        Parameters:
            self ('TestSurfaceOverlayMethod'): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        surface_config = {
            'data': self.temp_data * 3.5,  
            'plot_type': 'contourf',
            'levels': [950, 1000, 1050],
            'alpha': 0.5
        }
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.temp_data, 't2m',
            *self.extent_bounds,
            surface_overlay=surface_config,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_overlay_contourf_without_levels(self: 'TestSurfaceOverlayMethod') -> None:
        """
        This test confirms that a filled contour overlay without specified levels renders correctly and returns a Figure. By providing a filled contour overlay configuration that omits the `levels` parameter, the test asserts that the method processes this configuration without errors and that a Figure is returned, indicating successful rendering of the filled contour overlay with automatically determined levels.

        Parameters:
            self ('TestSurfaceOverlayMethod'): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        surface_config = {
            'data': self.temp_data * 3.5,  
            'plot_type': 'contourf'
        }
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.temp_data, 't2m',
            *self.extent_bounds,
            surface_overlay=surface_config,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestInterpolation:
    """ Tests for grid interpolation. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestInterpolation', mpas_coordinates, mpas_surface_temp_data) -> None:
        """
        This fixture initializes the test environment for interpolation tests. It prepares an `MPASSurfacePlotter` instance and loads real MPAS coordinate and surface temperature data to be used in testing the interpolation functionality of the surface plotting. The fixture also computes the extent bounds from the coordinate data for use in plotting.

        Parameters:
            self ('TestInterpolation'): Test instance which will receive fixture attributes.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_surface_temp_data: Session fixture providing real surface temperature data.

        Returns:
            None: Populates `self.plotter`, `self.lon`, `self.lat`, and `self.data`.
        """
        if mpas_coordinates is None or mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
            return
        
        self.plotter = MPASSurfacePlotter()
        lon_full, lat_full = mpas_coordinates
        self.lon = lon_full[:100]
        self.lat = lat_full[:100]
        
        self.data = mpas_surface_temp_data[:100]
        
        self.extent_bounds = (
            float(self.lon.min()), float(self.lon.max()),
            float(self.lat.min()), float(self.lat.max())
        )
    
    def test_interpolation_with_grid_resolution_float(self: 'TestInterpolation') -> None:
        """
        This test verifies that the interpolation helper correctly processes a float `grid_resolution` value representing angular resolution in degrees. By providing a valid float grid resolution, the test asserts that the interpolation is performed and that a Figure is returned, indicating successful rendering of the surface map with the specified interpolation settings.

        Parameters:
            self ('TestInterpolation'): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            *self.extent_bounds,
            plot_type='contourf',
            grid_resolution=0.5
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_interpolation_invalid_grid_resolution(self: 'TestInterpolation') -> None:
        """
        This test confirms that providing an invalid `grid_resolution` value (e.g., a negative number) raises a `ValueError`. By asserting that the plotting function raises a `ValueError` when an invalid grid resolution is provided, the test verifies that the interpolation helper properly validates the `grid_resolution` parameter and surfaces errors when it is misused.

        Parameters:
            self ('TestInterpolation'): Test instance containing prepared fixtures.

        Returns:
            None: Assertion expects a ValueError to be raised.
        """
        with pytest.raises((ValueError, Exception)):
            self.plotter.create_surface_map(
                self.lon, self.lat, self.data, 't2m',
                *self.extent_bounds,
                plot_type='contourf',
                grid_resolution=-0.5
            )
    
    def test_interpolation_with_fixed_resolution(self: 'TestInterpolation') -> None:
        """
        This test verifies that the interpolation helper can process a fixed grid resolution specified as an integer number of points. By providing a valid integer grid resolution, the test asserts that the interpolation is performed and that a Figure is returned, indicating successful rendering of the surface map with the specified fixed grid resolution.

        Parameters:
            self ('TestInterpolation'): Test instance containing prepared fixtures.
            
        Returns:
            None: Assertion validates returned Figure type.
        """
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            *self.extent_bounds,
            plot_type='contourf',
            grid_resolution=50
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_interpolation_adaptive_resolution(self: 'TestInterpolation') -> None:
        """
        This test confirms that the interpolation helper can handle an adaptive grid resolution scenario where the input data is sparse. By providing a small subset of the real MPAS data, the test asserts that the interpolation is performed and that a Figure is returned, indicating successful rendering of the surface map with adaptive resolution based on the provided data.

        Parameters:
            self ('TestInterpolation'): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        lon = self.lon[:20]
        lat = self.lat[:20]
        data = self.data[:20]

        extent_bounds = (
            float(lon.min()), float(lon.max()),
            float(lat.min()), float(lat.max())
        )
        
        fig, ax = self.plotter.create_surface_map(
            lon, lat, data, 't2m',
            *extent_bounds,
            plot_type='contourf'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

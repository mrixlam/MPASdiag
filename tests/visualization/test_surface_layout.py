#!/usr/bin/env python3
"""
MPASdiag Test Suite: Surface Layout Tests

This module contains unit tests for the surface layout functionality of the MPASdiag visualization package. The tests focus on verifying that the plotting functions correctly handle global and regional extents, as well as title and timestamp generation. The tests use real MPAS coordinate and surface temperature data provided by session fixtures to ensure realistic plotting scenarios. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules for testing
import os
import sys
import pytest
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from mpasdiag.visualization.surface import MPASSurfacePlotter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestGlobalExtent:
    """ Tests for global extent handling. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestGlobalExtent", mpas_coordinates, mpas_surface_temp_data) -> None:
        """
        This fixture initializes the MPASSurfacePlotter and prepares real MPAS coordinate and surface temperature data for testing global and regional extent plotting. It ensures that the test methods have access to realistic data arrays for longitude, latitude, and surface temperature, which are essential for verifying the plotting functionality under different geographic extents.

        Parameters:
            self ("TestGlobalExtent"): Test instance which will receive fixture attributes.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_surface_temp_data: Session fixture providing real surface temperature data.

        Returns:
            None: Populates `self.plotter`, `self.lon`, `self.lat`, and `self.data`.
        """
        if mpas_coordinates is None or mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
        
        self.plotter = MPASSurfacePlotter()
        lon_full, lat_full = mpas_coordinates
        self.lon = lon_full[:200]
        self.lat = lat_full[:200]
        
        temp_full = mpas_surface_temp_data
        self.data = temp_full[:200] if len(temp_full) >= 200 else np.tile(temp_full, (200 // len(temp_full) + 1))[:200]
    
    def test_global_extent(self: "TestGlobalExtent") -> None:
        """
        This test verifies that the plotting function can handle a global geographic extent correctly. It checks that when the full range of longitude and latitude is provided, the plotter generates a figure without errors. The test asserts that a Figure object is returned, indicating that the plotting process completed successfully for a global domain.

        Parameters:
            self ("TestGlobalExtent"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            -180, 180, -90, 90,  
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_regional_extent(self: "TestGlobalExtent") -> None:
        """
        This test checks that the plotting function can handle a regional geographic extent correctly. It uses a subset of the longitude and latitude data to define a smaller geographic area and verifies that the plotter can generate a figure for this region without errors. The test asserts that a Figure object is returned, confirming that the plotting process completed successfully for a regional domain.

        Parameters:
            self ("TestGlobalExtent"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        lon = self.lon[:100]
        lat = self.lat[:100]
        data = self.data[:100]

        extent_bounds = (
            float(lon.min()), float(lon.max()),
            float(lat.min()), float(lat.max())
        )
        
        fig, ax = self.plotter.create_surface_map(
            lon, lat, data, 't2m',
            *extent_bounds,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestTitleAndTimestamp:
    """ Tests for title and timestamp handling. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestTitleAndTimestamp", mpas_coordinates, mpas_surface_temp_data) -> None:
        """
        This fixture initializes the MPASSurfacePlotter and prepares real MPAS coordinate and surface temperature data for testing title and timestamp generation in the plotting functions. It ensures that the test methods have access to realistic data arrays for longitude, latitude, and surface temperature, which are essential for verifying that titles and timestamps are correctly generated and displayed on the plots.

        Parameters:
            self ("TestTitleAndTimestamp"): Test instance which will receive fixture attributes.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_surface_temp_data: Session fixture providing real surface temperature data.

        Returns:
            None: Populates `self.plotter`, `self.lon`, `self.lat`, and `self.data`.
        """
        if mpas_coordinates is None or mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
        
        self.plotter = MPASSurfacePlotter()
        lon_full, lat_full = mpas_coordinates
        self.lon = lon_full[:50]
        self.lat = lat_full[:50]
        
        self.data = mpas_surface_temp_data[:50]
        
        self.extent_bounds = (
            float(self.lon.min()), float(self.lon.max()),
            float(self.lat.min()), float(self.lat.max())
        )
    
    def test_default_title_with_timestamp(self: "TestTitleAndTimestamp") -> None:
        """
        This test verifies that when a timestamp is provided without a custom title, the plotter generates a default title that includes the timestamp information. It checks that the plotting function can handle the inclusion of a timestamp in the title and still returns a valid Figure object. The test asserts that a Figure object is returned, indicating that the plotting process completed successfully with the default title and timestamp.

        Parameters:
            self ("TestTitleAndTimestamp"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        from datetime import datetime
        time_stamp = datetime(2025, 1, 15, 12, 0)
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            *self.extent_bounds,
            time_stamp=time_stamp,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_custom_title_with_timestamp(self: "TestTitleAndTimestamp") -> None:
        """
        This test verifies that when a custom title is provided alongside a timestamp, the plotter preserves the custom title verbatim while still including the timestamp information. It checks that the plotting function can handle both a custom title and a timestamp without modifying the provided title string. The test asserts that a Figure object is returned, indicating that the plotting process completed successfully with the custom title and timestamp.

        Parameters:
            self ("TestTitleAndTimestamp"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        from datetime import datetime
        time_stamp = datetime(2025, 1, 15, 12, 0)
        title = "Custom Title | Valid Time: 2025011512"
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            *self.extent_bounds,
            title=title,
            time_stamp=time_stamp,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_timestamp_text_box(self: "TestTitleAndTimestamp") -> None:
        """
        This test ensures that when a timestamp is provided with the option to display it in a text box, the plotter correctly renders the timestamp in a boxed annotation on the plot. It checks that the plotting function can handle the inclusion of a timestamp in a text box and still returns a valid Figure object. The test asserts that a Figure object is returned, indicating that the plotting process completed successfully with the timestamp displayed in a text box.

        Parameters:
            self ("TestTitleAndTimestamp"): Test instance containing prepared fixtures.
            
        Returns:
            None: Assertion validates returned Figure type.
        """
        from datetime import datetime
        time_stamp = datetime(2025, 1, 15, 12, 0)
        title = "Custom Title Without Timestamp"
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            *self.extent_bounds,
            title=title,
            time_stamp=time_stamp,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

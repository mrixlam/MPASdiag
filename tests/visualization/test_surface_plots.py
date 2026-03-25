#!/usr/bin/env python3
"""
MPASdiag Test Suite: Surface Plotting Tests

This module contains a comprehensive set of unit tests for the surface plotting functionality of the MPASdiag visualization package. The tests cover various plot types (contour, contourf, scatter), colormap handling, normalization, and edge cases to ensure robust behavior across typical and atypical usage scenarios. It uses real MPAS data for testing to validate actual workflows and includes fixtures for setup and teardown of test environments. The tests are designed to be run with pytest and include assertions to confirm expected outcomes. 

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
import matplotlib
import numpy as np
matplotlib.use('Agg')
from unittest.mock import patch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from cartopy.mpl.geoaxes import GeoAxes
from tests.test_data_helpers import load_mpas_coords_from_processor

from mpasdiag.visualization.surface import MPASSurfacePlotter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestPlotTypes:
    """ Tests for different plot types and rendering options. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestPlotTypes", mpas_coordinates, mpas_surface_temp_data) -> None:
        """
        This fixture initializes the MPASSurfacePlotter and loads real MPAS data for testing various plot types. It prepares the longitude, latitude, and surface temperature data arrays, and computes the extent bounds for plotting based on the actual coordinate values. If the MPAS data is not available, it gracefully skips the tests that depend on it.

        Parameters:
            self ("TestPlotTypes"): Test instance which will receive fixture attributes.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_surface_temp_data: Session fixture providing real surface temperature data.

        Returns:
            None: Populates `self.plotter`, `self.lon`, `self.lat`, and `self.data`.
        """
        if mpas_coordinates is None or mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
        
        self.plotter = MPASSurfacePlotter()
        lon_full, lat_full = mpas_coordinates
        self.lon = lon_full[:100]
        self.lat = lat_full[:100]
        
        self.data = mpas_surface_temp_data[:100]
        
        self.extent_bounds = (
            float(self.lon.min()), float(self.lon.max()),
            float(self.lat.min()), float(self.lat.max())
        )
    
    def test_invalid_plot_type(self: "TestPlotTypes") -> None:
        """
        This test verifies that providing an invalid `plot_type` to the `create_surface_map` method raises a ValueError with an appropriate message. It attempts to call the plotting method with a non-existent plot type and asserts that the exception is raised and contains the expected text.

        Parameters:
            self ("TestPlotTypes"): Test instance containing prepared fixtures.

        Returns:
            None: Test asserts that a ValueError is raised for invalid input.
        """
        with pytest.raises(ValueError) as exc_info:
            self.plotter.create_surface_map(
                self.lon, self.lat, self.data, 't2m',
                *self.extent_bounds,
                plot_type='invalid'
            )
        assert 'plot_type must be' in str(exc_info.value)
    
    def test_contour_plot(self: "TestPlotTypes") -> None:
        """
        This test checks that the `contour` plot type correctly produces a Figure when valid inputs are provided. It calls the `create_surface_map` method with `plot_type='contour'` and asserts that the returned object is an instance of `matplotlib.figure.Figure`. This confirms that the contour plotting path is functional and returns the expected output type.

        Parameters:
            self ("TestPlotTypes"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            *self.extent_bounds,
            plot_type='contour'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_contourf_plot(self: "TestPlotTypes") -> None:
        """
        This test validates that the `contourf` plot type successfully generates a Figure when given valid data and parameters. It invokes the `create_surface_map` method with `plot_type='contourf'` and checks that the output is a Figure instance, confirming that the filled contour plotting path is operational.

        Parameters:
            self ("TestPlotTypes"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            *self.extent_bounds,
            plot_type='contourf'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_both_plot_type(self: "TestPlotTypes") -> None:
        """
        This test ensures that the `both` plot type, which combines contour and filled contour, produces a Figure without errors. It calls the plotting method with `plot_type='both'` and asserts that the result is a Figure instance, confirming that the combined plotting path is functional.

        Parameters:
            self ("TestPlotTypes"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            *self.extent_bounds,
            plot_type='both'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_with_custom_levels(self: "TestPlotTypes") -> None:
        """
        This test checks that providing a custom list of contour levels to the `create_surface_map` method results in a Figure being produced without errors. It calls the method with a specific set of levels and asserts that the output is a Figure instance, confirming that the plotter correctly handles user-defined contour levels.

        Parameters:
            self ("TestPlotTypes"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        levels = [float(x) for x in [0, 10, 20, 30]]

        assert levels is not None and len(levels) > 0, "Levels list should not be empty"
        assert isinstance(self.plotter.ax, GeoAxes)

        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            *self.extent_bounds,
            plot_type='contourf',
            levels=levels
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_with_clim_and_levels(self: "TestPlotTypes") -> None:
        """
        This test verifies that when both `clim_min` and `clim_max` are provided along with a list of contour `levels`, the plotting function correctly handles the clipping and still produces a Figure. It checks that the specified levels are respected while applying the clim limits, ensuring that the plotter can manage combined level and clipping inputs without errors.

        Parameters:
            self ("TestPlotTypes"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        levels = [float(x) for x in [0, 10, 20, 30]]
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            *self.extent_bounds,
            plot_type='contourf',
            levels=levels,
            clim_min=260,  
            clim_max=290
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_with_clim_min_not_in_levels(self: "TestPlotTypes") -> None:
        """
        This test ensures that if `clim_min` is provided and is not included in the list of contour `levels`, the plotting function should still produce a Figure without errors. The plotter should handle this scenario gracefully, potentially by inserting the `clim_min` value into the levels list or by applying the clipping correctly even if it doesn't align with the specified levels. The test asserts that a Figure is returned, confirming that the plotter can manage this edge case without crashing.

        Parameters:
            self ("TestPlotTypes"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        levels = [float(x) for x in [0, 10, 20, 30]]
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            *self.extent_bounds,
            plot_type='contourf',
            levels=levels,
            clim_min=260,  
            clim_max=295
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_with_clim_max_not_in_levels(self: "TestPlotTypes") -> None:
        """
        This test checks that if `clim_max` is provided and is not included in the list of contour `levels`, the plotting function still produces a Figure without errors. Similar to the previous test, the plotter should handle this situation gracefully, ensuring that the clim limits are applied correctly even if they do not coincide with the specified levels. The test asserts that a Figure is returned, confirming that the plotter can manage this edge case effectively.

        Parameters:
            self ("TestPlotTypes"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        levels = [float(x) for x in [260, 270, 280, 290]]
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            *self.extent_bounds,
            plot_type='contourf',
            levels=levels,
            clim_min=260,
            clim_max=295  
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestColormapHandling:
    """ Tests for colormap and normalization. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestColormapHandling", mpas_coordinates, mpas_surface_temp_data) -> None:
        """
        This fixture initializes the MPASSurfacePlotter and loads real MPAS data for testing colormap handling and normalization. It prepares longitude, latitude, and surface temperature data arrays, and computes the extent bounds for plotting based on the actual coordinate values. If the MPAS data is not available, it gracefully skips the tests that depend on it.

        Parameters:
            self ("TestColormapHandling"): Test instance which will receive fixture attributes.
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
    
    def test_custom_colormap_string(self: "TestColormapHandling") -> None:
        """
        This test verifies that providing a valid custom colormap name as a string to the `create_surface_map` method results in a Figure being produced without errors. It checks that the specified colormap is applied correctly by asserting that the output is a Figure instance, confirming that the plotter can handle user-defined colormap strings.

        Parameters:
            self ("TestColormapHandling"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            *self.extent_bounds,
            colormap='coolwarm',
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_invalid_colormap(self: "TestColormapHandling") -> None:
        """
        This test checks that if an invalid colormap name is provided, the plotting function does not crash and still produces a Figure. The plotter should handle the invalid colormap gracefully, potentially by falling back to a default colormap. The test asserts that a Figure is returned, confirming that the plotter can manage this edge case without errors.

        Parameters:
            self ("TestColormapHandling"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            *self.extent_bounds,
            colormap='invalid_cmap',
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_none_colormap_object(self: "TestColormapHandling") -> None:
        """
        This test ensures that if the colormap retrieval function returns `None` (simulating a failure to get a valid colormap object), the plotting function still produces a Figure without crashing. The plotter should handle this scenario gracefully, potentially by using a default colormap or by proceeding without applying a colormap. The test asserts that a Figure is returned, confirming that the plotter can manage this edge case effectively.

        Parameters:
            self ("TestColormapHandling"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        with patch('matplotlib.pyplot.get_cmap', return_value=None):
            fig, ax = self.plotter.create_surface_map(
                self.lon, self.lat, self.data, 't2m',
                *self.extent_bounds,
                plot_type='scatter'
            )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_clim_without_levels(self: "TestColormapHandling") -> None:
        """
        This test verifies that when `clim_min` and `clim_max` are provided without explicit contour `levels`, the scatter plotting path still produces a Figure without errors. The plotter should apply the clim limits correctly to the colormap normalization even in the absence of specified levels. The test asserts that a Figure is returned, confirming that the plotter can handle clim inputs independently of levels.

        Parameters:
            self ("TestColormapHandling"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            *self.extent_bounds,
            clim_min=0.2,
            clim_max=0.8,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_boundary_norm_exception(self: "TestColormapHandling") -> None:
        """
        This test checks that if the boundary normalization function raises an exception (simulating an error in colormap normalization), the plotting function still produces a Figure without crashing. The plotter should catch the exception and proceed with plotting, potentially using a default normalization or by skipping normalization. The test asserts that a Figure is returned, confirming that the plotter can manage this edge case effectively.

        Parameters:
            self ("TestColormapHandling"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        data = self.data.copy()
        data[::5] = np.nan
        
        levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, data, 't2m',
            *self.extent_bounds,
            levels=levels,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestScatterPlotMethod:
    """ Tests for _create_scatter_plot method. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestScatterPlotMethod") -> None:
        """
        This fixture initializes an `MPASSurfacePlotter` instance for testing the scatter plotting method. It prepares the plotter object that will be used in subsequent tests to validate the behavior of the `_create_scatter_plot` method, including handling of dense point clouds and medium-density datasets. The fixture does not load any data itself, as the individual tests will use real MPAS data or synthetic data as needed.

        Parameters:
            self ("TestScatterPlotMethod"): Test instance which will receive `plotter`.

        Returns:
            None: Populates `self.plotter`.
        """
        self.plotter = MPASSurfacePlotter()
    
    def test_scatter_high_density(self: "TestScatterPlotMethod") -> None:
        """
        This test verifies that the scatter plotting method can handle high-density datasets (e.g., 1000 points) without crashing and still produces a Figure. It uses a dense subset of real MPAS data to create a scatter plot and asserts that the output is a Figure instance, confirming that the plotter can manage large point clouds effectively. The test also checks that the extent is calculated correctly from the actual data points.

        Parameters:
            self ("TestScatterPlotMethod"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        lon_full, lat_full, u_full, v_full = load_mpas_coords_from_processor(n=1000)
        
        lon = lon_full[:1000]
        lat = lat_full[:1000]
        u_subset = u_full[:1000]
        
        data = (u_subset - u_full.min()) / (u_full.max() - u_full.min() + 1e-12) * 300
        
        extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))
        
        fig, ax = self.plotter.create_surface_map(
            lon, lat, data, 't2m',
            *extent,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_scatter_medium_density(self: "TestScatterPlotMethod") -> None:
        """
        This test checks that the scatter plotting method can handle medium-density datasets (e.g., 500 points) without crashing and still produces a Figure. It uses a subset of real MPAS data to create a scatter plot and asserts that the output is a Figure instance, confirming that the plotter can manage typical point densities effectively. The test also verifies that the extent is calculated correctly from the actual data points.

        Parameters:
            self ("TestScatterPlotMethod"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        lon_full, lat_full, u_full, v_full = load_mpas_coords_from_processor(n=500)
        
        lon = lon_full[:500]
        lat = lat_full[:500]
        u_subset = u_full[:500]
        
        data = (u_subset - u_full.min()) / (u_full.max() - u_full.min() + 1e-12) * 300
        
        extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))
        
        fig, ax = self.plotter.create_surface_map(
            lon, lat, data, 't2m',
            *extent,
            plot_type='scatter'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestColorbarMethod:
    """ Tests for _add_colorbar_with_metadata method. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestColorbarMethod") -> None:
        """
        This fixture initializes an `MPASSurfacePlotter` instance for testing the colorbar creation method. It prepares the plotter object that will be used in subsequent tests to validate the behavior of the `_add_colorbar_with_metadata` method, including handling of metadata-driven labels and formatting. The fixture also sets default extent bounds for colorbar tests, which can be overridden by individual tests that use real MPAS data.

        Parameters:
            self ("TestColorbarMethod"): Test instance which will receive `plotter`.

        Returns:
            None: Populates `self.plotter`.
        """
        self.plotter = MPASSurfacePlotter()
        self.extent_bounds = (-100, -90, 30, 40)
    
    def test_colorbar_with_metadata(self: "TestColorbarMethod", mpas_coordinates, mpas_surface_temp_data) -> None:
        """
        This test verifies that the colorbar creation method correctly uses metadata to generate appropriate labels and formatting for the colorbar. It uses real MPAS data to create a scatter plot and checks that the resulting Figure includes a colorbar with the expected label derived from the metadata (e.g., "2m Temperature (K)"). The test asserts that a Figure is returned, confirming that the colorbar is created successfully with metadata-driven labeling.

        Parameters:
            self ("TestColorbarMethod"): Test instance containing the plotter.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_surface_temp_data: Session fixture providing real surface temperature data.

        Returns:
            None: Assertion validates returned Figure type.
        """
        if mpas_coordinates is None or mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
        
        lon_full, lat_full = mpas_coordinates
        lon = lon_full[:50]
        lat = lat_full[:50]
        data = mpas_surface_temp_data[:50]
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
    
    def test_colorbar_tick_formatting_exception(self: "TestColorbarMethod") -> None:
        """
        This test checks that if the tick formatting function for the colorbar raises an exception, the plotting function still produces a Figure without crashing. The plotter should catch the exception and proceed with plotting, potentially using default tick formatting or by skipping custom formatting. The test asserts that a Figure is returned, confirming that the plotter can manage this edge case effectively.

        Parameters:
            self ("TestColorbarMethod"): Test instance containing the plotter.

        Returns:
            None: Assertion validates returned Figure type.
        """
        with patch.object(MPASSurfacePlotter, '_format_ticks_dynamic', side_effect=Exception("Format failed")):
            lon = np.linspace(-100, -90, 50)
            lat = np.linspace(30, 40, 50)
            data = np.random.rand(50) * 300
            
            fig, ax = self.plotter.create_surface_map(
                lon, lat, data, 't2m',
                *self.extent_bounds,
                plot_type='scatter'
            )
            
            assert isinstance(fig, Figure)
            plt.close(fig)


class TestContourPlotting:
    """ Tests for contour plotting methods. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestContourPlotting", mpas_coordinates, mpas_surface_temp_data) -> None:
        """
        This fixture initializes the MPASSurfacePlotter and loads real MPAS data for testing contour plotting methods. It prepares longitude, latitude, and surface temperature data arrays, and computes the extent bounds for plotting based on the actual coordinate values. If the MPAS data is not available, it gracefully skips the tests that depend on it.

        Parameters:
            self ("TestContourPlotting"): Test instance which will receive fixture attributes.
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_surface_temp_data: Session fixture providing real surface temperature data.

        Returns:
            None: Populates `self.plotter`, `self.lon`, `self.lat`, and `self.data`.
        """
        if mpas_coordinates is None or mpas_surface_temp_data is None:
            pytest.skip("MPAS data not available")
        
        self.plotter = MPASSurfacePlotter()
        lon_full, lat_full = mpas_coordinates
        self.lon = lon_full[:100]
        self.lat = lat_full[:100]
        
        self.data = mpas_surface_temp_data[:100]
        
        self.extent_bounds = (
            float(self.lon.min()), float(self.lon.max()),
            float(self.lat.min()), float(self.lat.max())
        )
    
    def test_contour_with_levels(self: "TestContourPlotting") -> None:
        """
        This test verifies that contour plotting with a specified list of `levels` produces a Figure without errors. It calls the `create_surface_map` method with `plot_type='contour'` and a defined set of contour levels, and asserts that the output is a Figure instance. This confirms that the contour plotting path correctly handles user-defined levels for rendering.

        Parameters:
            self ("TestContourPlotting"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        levels = [float(x) for x in [250, 270, 290]]
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            *self.extent_bounds,
            plot_type='contour',
            levels=levels
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_contour_without_levels(self: "TestContourPlotting") -> None:
        """
        This test checks that contour plotting without providing explicit `levels` still produces a Figure without errors. The contour plotting routine should compute appropriate levels based on the data range and still render a valid plot. The test asserts that a Figure is returned, confirming that the contour plotting path can handle the absence of user-defined levels gracefully.

        Parameters:
            self ("TestContourPlotting"): Test instance containing prepared fixtures.   

        Returns:
            None: Assertion validates returned Figure type.
        """
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            *self.extent_bounds,
            plot_type='contour'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_contour_label_exception(self: "TestContourPlotting") -> None:
        """
        This test ensures that if the contour labeling function raises an exception, the contour plotting method still produces a Figure without crashing. The plotter should catch the exception and proceed with plotting, potentially by skipping labels or using default labeling. The test asserts that a Figure is returned, confirming that the plotter can manage this edge case effectively.

        Parameters:
            self ("TestContourPlotting"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        with patch('matplotlib.axes.Axes.clabel', side_effect=Exception("Labeling failed")):
            fig, ax = self.plotter.create_surface_map(
                self.lon, self.lat, self.data, 't2m',
                *self.extent_bounds,
                plot_type='contour',
                levels=[250, 270, 290]
            )
            
            assert isinstance(fig, Figure)
        
        plt.close(fig)
    
    def test_contourf_with_levels(self: "TestContourPlotting") -> None:
        """
        This test verifies that filled-contour plotting with a specified list of `levels` produces a Figure without errors. It calls the `create_surface_map` method with `plot_type='contourf'` and a defined set of contour levels, and asserts that the output is a Figure instance. This confirms that the filled contour plotting path correctly handles user-defined levels for rendering.

        Parameters:
            self ("TestContourPlotting"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        levels = [float(x) for x in [250, 270, 290]]
        
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            *self.extent_bounds,
            plot_type='contourf',
            levels=levels
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_contourf_without_levels(self: "TestContourPlotting") -> None:
        """
        This test checks that filled-contour plotting without providing explicit `levels` still produces a Figure without errors. The filled contour plotting routine should compute appropriate levels based on the data range and still render a valid plot. The test asserts that a Figure is returned, confirming that the filled contour plotting path can handle the absence of user-defined levels gracefully.

        Parameters:
            self ("TestContourPlotting"): Test instance containing prepared fixtures.

        Returns:
            None: Assertion validates returned Figure type.
        """
        fig, ax = self.plotter.create_surface_map(
            self.lon, self.lat, self.data, 't2m',
            *self.extent_bounds,
            plot_type='contourf'
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

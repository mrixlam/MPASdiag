#!/usr/bin/env python3
"""
MPASdiag Test Suite: Full-Stack End-to-End Integration Tests

This module contains comprehensive end-to-end tests that validate the entire MPASdiag processing and visualization pipeline on real MPAS data. Each test covers a complete workflow for a specific analysis type (precipitation, surface fields, wind, vertical cross-sections), starting from configuration and data loading, through diagnostics computation and coordinate extraction, to rendering and saving a figure. The tests assert that intermediate data fields are valid and that the final rendered figures are successfully saved to disk. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: June 2026
Version: 1.0.0
"""
from pathlib import Path

import numpy as np
import pytest

from mpasdiag.processing.utils_config import MPASConfig
from mpasdiag.processing.processors_2d import MPAS2DProcessor
from mpasdiag.processing.processors_3d import MPAS3DProcessor
from mpasdiag.processing.utils_geog import GeographicBounds
from mpasdiag.processing.remapping import remap_mpas_to_latlon
from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
from mpasdiag.diagnostics.wind import WindDiagnostics
from mpasdiag.visualization.precipitation import (
    MPASPrecipitationPlotter,
    PrecipitationRenderStyle,
)
from mpasdiag.visualization.surface import MPASSurfacePlotter, SurfaceMapStyle
from mpasdiag.visualization.wind import MPASWindPlotter
from mpasdiag.visualization.base_visualizer import WindPlotStyle
from mpasdiag.visualization.cross_section import (
    MPASVerticalCrossSectionPlotter,
    CrossSectionStyle,
)

from tests.test_data_helpers import get_mpas_data_paths
from tests.integration.conftest import CARTOPY_AVAILABLE, bounds_from_coords


pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_data,
    pytest.mark.slow,
]


@pytest.fixture
def data_paths() -> dict:
    """
    This fixture resolves the necessary MPAS data paths for the end-to-end tests. It uses the helper function `get_mpas_data_paths` to retrieve paths for the grid file, diagnostic data directory, and MPASOUT data directory. If the required grid file is not available, the fixture will skip all tests that depend on it, since the grid is essential for any processing or visualization. The returned dictionary provides a standardized way for tests to access the necessary data files without hardcoding paths.

    Parameters:
        None

    Returns:
        dict: A dictionary containing resolved paths for 'grid_file', 'diag_dir', and 'mpasout_dir'.
    """
    paths: dict = get_mpas_data_paths()

    if paths["grid_file"] is None:
        pytest.skip("MPAS grid file not available")

    return paths


def _assert_saved(output_path: Path) -> None:
    """
    This helper function asserts that a rendered figure has been successfully saved to disk. It checks that the expected output file exists and that its size is greater than zero, indicating that the file is not empty. This function is used in the end-to-end tests to verify that the final rendering step has produced a valid output file.

    Parameters:
        output_path (Path): The expected path to the saved figure file.

    Returns:
        None
    """
    assert output_path.exists(), f"Expected rendered figure at {output_path}"
    assert output_path.stat().st_size > 0


@pytest.mark.skipif(not CARTOPY_AVAILABLE, reason="Cartopy not available")
class TestEndToEnd:
    """ Full config-to-figure pipeline tests for each analysis type using real MPAS data. """

    def test_precipitation_end_to_end(self: "TestEndToEnd", 
                                      data_paths: dict, 
                                      output_dir: Path,) -> None:
        """
        This test validates the full precipitation diagnostics and rendering pipeline on real 2D diagnostic data. It computes the precipitation accumulation difference at the first time step, extracts the corresponding longitude and latitude coordinates, and renders a scatter plot of the precipitation field on the unstructured mesh. The test asserts that the computed accumulation values are finite and that the number of longitude points matches the number of data values. Finally, it saves the rendered figure as a PNG file and asserts that the file was successfully created. 

        Parameters:
            data_paths (dict): Resolved MPAS data paths.
            output_dir (Path): Temporary directory for saved figures.   

        Returns:
            None
        """
        if data_paths["diag_dir"] is None:
            pytest.skip("Diagnostic data directory not available")

        config = MPASConfig(
            grid_file=str(data_paths["grid_file"]),
            data_dir=str(data_paths["diag_dir"]),
            variable="rainnc",
            accumulation_period="a01h",
            time_index=1,
            plot_type="scatter",
        )

        processor = MPAS2DProcessor(config.grid_file, verbose=False)
        processor.load_2d_data(config.data_dir, use_pure_xarray=True)

        diag = PrecipitationDiagnostics(verbose=config.verbose)
        accum = diag.compute_precipitation_difference(
            processor.dataset,
            time_index=config.time_index,
            var_name=config.variable,
            accum_period=config.accumulation_period,
        )
        lon, lat = processor.extract_2d_coordinates_for_variable(config.variable)

        values = np.asarray(accum.values).ravel()
        assert np.isfinite(values).any()
        assert lon.size == values.size

        bounds = GeographicBounds(*bounds_from_coords(lon, lat))
        plotter = MPASPrecipitationPlotter(figsize=(8, 6), dpi=80)
        plotter.create_precipitation_map(
            lon,
            lat,
            values,
            bounds,
            accum_period=config.accumulation_period,
            style=PrecipitationRenderStyle(
                title="E2E Precip", plot_type=config.plot_type
            ),
            data_array=accum,
            var_name=config.variable,
        )

        out = output_dir / "e2e_precip"
        plotter.save_plot(str(out), formats=["png"])
        plotter.close_plot()
        _assert_saved(out.with_suffix(".png"))

    def test_precipitation_end_to_end_with_remap(self: "TestEndToEnd", 
                                                 data_paths: dict, 
                                                 output_dir: Path,) -> None:
        """
        This test validates the full precipitation diagnostics and rendering pipeline on real 2D diagnostic data, including the optional remapping step. It computes the precipitation accumulation difference at the first time step, extracts the corresponding longitude and latitude coordinates, and then remaps the accumulation field from the unstructured mesh to a regular lat-lon grid using nearest-neighbor interpolation. The test asserts that the remapped field contains coordinate variables for longitude and latitude. Finally, it renders a scatter plot of the remapped precipitation field and saves it as a PNG file, asserting that the file was successfully created.

        Parameters:
            data_paths (dict): Resolved MPAS data paths.
            output_dir (Path): Temporary directory for saved figures.

        Returns:
            None
        """
        if data_paths["diag_dir"] is None:
            pytest.skip("Diagnostic data directory not available")

        processor = MPAS2DProcessor(str(data_paths["grid_file"]), verbose=False)
        processor.load_2d_data(str(data_paths["diag_dir"]), use_pure_xarray=True)

        diag = PrecipitationDiagnostics(verbose=False)
        accum = diag.compute_precipitation_difference(
            processor.dataset, time_index=1, var_name="rainnc", accum_period="a01h"
        )

        lon, lat = processor.extract_2d_coordinates_for_variable("rainnc")

        remapped = remap_mpas_to_latlon(
            data=np.asarray(accum.values).ravel(),
            lon=lon,
            lat=lat,
            resolution=2.0,
            method="nearest",
        )

        assert "lon" in remapped.coords and "lat" in remapped.coords

        lon2d, lat2d = np.meshgrid(remapped["lon"].values, remapped["lat"].values)
        bounds = GeographicBounds(*bounds_from_coords(lon2d.ravel(), lat2d.ravel()))
        plotter = MPASPrecipitationPlotter(figsize=(8, 6), dpi=80)

        plotter.create_precipitation_map(
            lon2d.ravel(),
            lat2d.ravel(),
            remapped.values.ravel(),
            bounds,
            accum_period="a01h",
            style=PrecipitationRenderStyle(
                title="E2E Precip Remapped", plot_type="scatter"
            ),
            var_name="rainnc",
        )

        out = output_dir / "e2e_precip_remap"
        plotter.save_plot(str(out), formats=["png"])
        plotter.close_plot()
        _assert_saved(out.with_suffix(".png"))

    def test_surface_end_to_end(self: "TestEndToEnd", 
                                data_paths: dict, 
                                output_dir: Path,) -> None:
        """
        This test validates the full surface diagnostics and rendering pipeline on real 2D diagnostic data. It loads the 2D dataset, extracts the specified surface variable (e.g., 2-meter temperature) at the first time step, and retrieves the corresponding longitude and latitude coordinates. The test asserts that the extracted variable values are finite and that the number of longitude points matches the number of data values. Finally, it renders a scatter plot of the surface field and saves it as a PNG file, asserting that the file was successfully created.

        Parameters:
            data_paths (dict): Resolved MPAS data paths.
            output_dir (Path): Temporary directory for saved figures.

        Returns:
            None
        """
        if data_paths["diag_dir"] is None:
            pytest.skip("Diagnostic data directory not available")

        config = MPASConfig(
            grid_file=str(data_paths["grid_file"]),
            data_dir=str(data_paths["diag_dir"]),
            variable="t2m",
            plot_type="scatter",
        )

        processor = MPAS2DProcessor(config.grid_file, verbose=False)
        processor.load_2d_data(config.data_dir, use_pure_xarray=True)

        data = processor.get_2d_variable_data(
            config.variable, time_index=config.time_index
        )

        lon, lat = processor.extract_2d_coordinates_for_variable(config.variable)

        values = np.asarray(data.values).ravel()
        assert np.isfinite(values).any()

        bounds = GeographicBounds(*bounds_from_coords(lon, lat))
        plotter = MPASSurfacePlotter(figsize=(8, 6), dpi=80)

        plotter.create_surface_map(
            lon,
            lat,
            values,
            config.variable,
            bounds,
            style=SurfaceMapStyle(title="E2E Surface", plot_type=config.plot_type),
            data_array=data,
        )

        out = output_dir / "e2e_surface"
        plotter.save_plot(str(out), formats=["png"])
        plotter.close_plot()
        _assert_saved(out.with_suffix(".png"))

    def test_wind_end_to_end(self: "TestEndToEnd", 
                             data_paths: dict, 
                             output_dir: Path,) -> None:
        """
        This test validates the full wind diagnostics and rendering pipeline on real 2D diagnostic data. It loads the 2D dataset, extracts the specified wind components (e.g., u10 and v10) at the first time step, and retrieves the corresponding longitude and latitude coordinates. The test asserts that the extracted wind component values are finite and that the number of longitude points matches the number of data values. Finally, it renders a wind plot (e.g., barbs) of the wind field and saves it as a PNG file, asserting that the file was successfully created.

        Parameters:
            data_paths (dict): Resolved MPAS data paths.
            output_dir (Path): Temporary directory for saved figures.

        Returns:
            None
        """
        if data_paths["diag_dir"] is None:
            pytest.skip("Diagnostic data directory not available")

        config = MPASConfig(
            grid_file=str(data_paths["grid_file"]),
            data_dir=str(data_paths["diag_dir"]),
            u_variable="u10",
            v_variable="v10",
            wind_plot_type="barbs",
        )

        processor = MPAS2DProcessor(config.grid_file, verbose=False)
        processor.load_2d_data(config.data_dir, use_pure_xarray=True)

        diag = WindDiagnostics(verbose=False)
        u, v = diag.get_2d_wind_components(
            processor.dataset,
            u_variable=config.u_variable,
            v_variable=config.v_variable,
            time_index=config.time_index,
        )

        lon, lat = processor.extract_2d_coordinates_for_variable(config.u_variable)

        step = max(1, lon.size // 500)
        sl = slice(None, None, step)
        lon_t, lat_t = lon[sl], lat[sl]
        u_t = np.asarray(u.values).ravel()[sl]
        v_t = np.asarray(v.values).ravel()[sl]
        assert np.isfinite(u_t).any() and np.isfinite(v_t).any()

        bounds = GeographicBounds(*bounds_from_coords(lon_t, lat_t))
        plotter = MPASWindPlotter(figsize=(8, 6), dpi=80)

        plotter.create_wind_plot(
            lon_t,
            lat_t,
            u_t,
            v_t,
            bounds,
            style=WindPlotStyle(
                subsample=1, plot_type=config.wind_plot_type, title="E2E Wind"
            ),
            level_info="surface",
        )

        out = output_dir / "e2e_wind"
        plotter.save_plot(str(out), formats=["png"])
        plotter.close_plot()
        _assert_saved(out.with_suffix(".png"))

    def test_cross_section_end_to_end(self: "TestEndToEnd", 
                                      data_paths: dict, 
                                      output_dir: Path,) -> None:
        """
        This test validates the full vertical cross-section diagnostics and rendering pipeline on real 3D diagnostic data. It loads the 3D dataset, extracts the specified variable (e.g., potential temperature) along a defined vertical cross-section between two geographic points, and retrieves the corresponding longitude, latitude, and vertical coordinate values. The test asserts that the extracted variable values are finite and that the number of longitude points matches the number of data values. Finally, it renders a contour plot of the vertical cross-section and saves it as a PNG file, asserting that the file was successfully created.

        Parameters:
            data_paths (dict): Resolved MPAS data paths.
            output_dir (Path): Temporary directory for saved figures.

        Returns:
            None
        """
        if data_paths["mpasout_dir"] is None:
            pytest.skip("MPASOUT data directory not available")

        config = MPASConfig(
            grid_file=str(data_paths["grid_file"]),
            data_dir=str(data_paths["mpasout_dir"]),
            variable="theta",
            start_lon=0.0,
            start_lat=-20.0,
            end_lon=0.0,
            end_lat=20.0,
            vertical_coord="pressure",
            num_points=40,
        )

        processor = MPAS3DProcessor(config.grid_file, verbose=False)
        processor.load_3d_data(config.data_dir, use_pure_xarray=True)
        plotter = MPASVerticalCrossSectionPlotter(figsize=(8, 6), dpi=80)

        plotter.create_vertical_cross_section(
            processor,
            config.variable,
            start_point=(config.start_lon, config.start_lat),
            end_point=(config.end_lon, config.end_lat),
            time_index=config.time_index,
            vertical_coord=config.vertical_coord,
            num_points=config.num_points,
            style=CrossSectionStyle(plot_type="contourf"),
        )

        out = output_dir / "e2e_cross_section"
        plotter.save_plot(str(out), formats=["png"])
        plotter.close_plot()
        _assert_saved(out.with_suffix(".png"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

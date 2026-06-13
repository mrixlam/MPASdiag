#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Test Suite: Visualization Layer Integration Tests

This module contains integration tests for the MPASdiag visualization pipeline, ensuring that the various plotters can successfully render figures from real processed data. Each test renders a specific type of plot (e.g., precipitation map, surface map, wind plot, cross-section) using data extracted from the provided MPAS datasets and asserts that the output files are created and non-empty.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: June 2026
Version: 1.0.0
"""

from pathlib import Path

import numpy as np
import pytest

from mpasdiag.processing.processors_2d import MPAS2DProcessor
from mpasdiag.processing.processors_3d import MPAS3DProcessor
from mpasdiag.processing.utils_geog import GeographicBounds
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

from tests.integration.conftest import CARTOPY_AVAILABLE, bounds_from_coords

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_data,
    pytest.mark.skipif(not CARTOPY_AVAILABLE, reason="Cartopy not available"),
]


def _assert_saved(output_path: Path) -> None:
    """
    This helper function asserts that a rendered figure was successfully saved to disk by checking that the file exists and is non-empty. This is used across multiple tests to verify that the visualization pipeline produced an output file.

    Parameters:
        output_path (Path): Expected output file path (including extension).

    Returns:
        None
    """
    assert output_path.exists(), f"Expected rendered figure at {output_path}"
    assert output_path.stat().st_size > 0


class TestVisualizationPipeline:
    """Real-rendering integration tests wiring processing/diagnostics output into the plotters."""

    def test_precipitation_map_render(
        self: "TestVisualizationPipeline",
        real_2d_processor: MPAS2DProcessor,
        output_dir: Path,
    ) -> None:
        """
        This test renders a precipitation map from a real accumulated precipitation field and asserts that a non-empty PNG file is produced. It exercises the full pipeline from data extraction, diagnostic computation, to cartographic rendering.

        Parameters:
            real_2d_processor (MPAS2DProcessor): Loaded 2D processor fixture with MPAS data.
            output_dir (Path): Temporary directory for saving rendered figures.

        Returns:
            None
        """
        lon, lat = real_2d_processor.extract_2d_coordinates_for_variable("rainnc")
        diag = PrecipitationDiagnostics(verbose=False)

        accum = diag.compute_precipitation_difference(
            real_2d_processor.dataset,
            time_index=1,
            var_name="rainnc",
            accum_period="a01h",
        )

        bounds = GeographicBounds(*bounds_from_coords(lon, lat))
        plotter = MPASPrecipitationPlotter(figsize=(8, 6), dpi=80)

        fig, ax = plotter.create_precipitation_map(
            lon,
            lat,
            np.asarray(accum.values).ravel(),
            bounds,
            accum_period="a01h",
            style=PrecipitationRenderStyle(
                title="Integration Precip", plot_type="scatter"
            ),
            data_array=accum,
            var_name="rainnc",
        )

        assert fig is not None and ax is not None

        out = output_dir / "precip_map"
        plotter.save_plot(str(out), formats=["png"])
        plotter.close_plot()
        _assert_saved(out.with_suffix(".png"))

    def test_surface_map_render(
        self: "TestVisualizationPipeline",
        real_2d_processor: MPAS2DProcessor,
        output_dir: Path,
    ) -> None:
        """
        This test renders a surface map of a real 2D field (e.g., 2m temperature) and asserts that a non-empty PNG file is produced. It verifies that the surface plotter can successfully visualize real data with appropriate styling and coordinate handling.

        Parameters:
            real_2d_processor (MPAS2DProcessor): Loaded 2D processor fixture with MPAS data.
            output_dir (Path): Temporary directory for saving rendered figures.

        Returns:
            None
        """
        lon, lat = real_2d_processor.extract_2d_coordinates_for_variable("t2m")
        data = real_2d_processor.get_2d_variable_data("t2m", time_index=0)

        bounds = GeographicBounds(*bounds_from_coords(lon, lat))
        plotter = MPASSurfacePlotter(figsize=(8, 6), dpi=80)

        fig, ax = plotter.create_surface_map(
            lon,
            lat,
            np.asarray(data.values).ravel(),
            "t2m",
            bounds,
            style=SurfaceMapStyle(title="Integration Surface", plot_type="scatter"),
            data_array=data,
        )

        assert fig is not None and ax is not None

        out = output_dir / "surface_map"
        plotter.save_plot(str(out), formats=["png"])
        plotter.close_plot()
        _assert_saved(out.with_suffix(".png"))

    def test_simple_scatter_render(
        self: "TestVisualizationPipeline",
        real_2d_processor: MPAS2DProcessor,
        output_dir: Path,
    ) -> None:
        """
        This test renders a simple scatter plot of a real 2D field (e.g., 2m temperature) and asserts that a non-empty PNG file is produced. It serves as a basic sanity check that the plotter can render raw data points without additional styling or geographic context.

        Parameters:
            real_2d_processor (MPAS2DProcessor): Loaded 2D processor fixture with MPAS data.
            output_dir (Path): Temporary directory for saving rendered figures.

        Returns:
            None
        """
        lon, lat = real_2d_processor.extract_2d_coordinates_for_variable("t2m")
        data = real_2d_processor.get_2d_variable_data("t2m", time_index=0)

        plotter = MPASSurfacePlotter(figsize=(8, 6), dpi=80)

        fig, ax = plotter.create_simple_scatter_plot(
            lon, lat, np.asarray(data.values).ravel(), title="Integration Scatter"
        )

        assert fig is not None and ax is not None

        out = output_dir / "scatter"
        plotter.save_plot(str(out), formats=["png"])
        plotter.close_plot()
        _assert_saved(out.with_suffix(".png"))

    def test_wind_plot_render(
        self: "TestVisualizationPipeline",
        real_2d_processor: MPAS2DProcessor,
        output_dir: Path,
    ) -> None:
        """
        This test renders a wind plot (e.g., barbs) from real 2D wind component fields and asserts that a non-empty PNG file is produced. It tests the integration of the wind diagnostics with the wind plotter, ensuring that the correct components are extracted, subsampled, and visualized with appropriate styling.

        Parameters:
            real_2d_processor (MPAS2DProcessor): Loaded 2D processor fixture with MPAS data.
            output_dir (Path): Temporary directory for saving rendered figures.

        Returns:
            None
        """
        lon, lat = real_2d_processor.extract_2d_coordinates_for_variable("u10")
        diag = WindDiagnostics(verbose=False)
        u, v = diag.get_2d_wind_components(
            real_2d_processor.dataset, u_variable="u10", v_variable="v10", time_index=0
        )

        step = max(1, lon.size // 500)
        sl = slice(None, None, step)
        lon_t, lat_t = lon[sl], lat[sl]
        u_t = np.asarray(u.values).ravel()[sl]
        v_t = np.asarray(v.values).ravel()[sl]

        bounds = GeographicBounds(*bounds_from_coords(lon_t, lat_t))
        plotter = MPASWindPlotter(figsize=(8, 6), dpi=80)

        fig, ax = plotter.create_wind_plot(
            lon_t,
            lat_t,
            u_t,
            v_t,
            bounds,
            style=WindPlotStyle(
                subsample=1, plot_type="barbs", title="Integration Wind"
            ),
            level_info="surface",
        )

        assert fig is not None and ax is not None

        out = output_dir / "wind_plot"
        plotter.save_plot(str(out), formats=["png"])
        plotter.close_plot()
        _assert_saved(out.with_suffix(".png"))

    def test_cross_section_render(
        self: "TestVisualizationPipeline",
        real_3d_processor: MPAS3DProcessor,
        output_dir: Path,
    ) -> None:
        """
        This test renders a vertical cross-section plot from a real 3D field (e.g., potential temperature) and asserts that a non-empty PNG file is produced. It verifies that the cross-section plotter can correctly extract a vertical slice from the 3D data, handle the specified vertical coordinate, and render it with appropriate styling.

        Parameters:
            real_3d_processor (MPAS3DProcessor): Loaded 3D processor fixture with MPAS data.
            output_dir (Path): Temporary directory for saving rendered figures.

        Returns:
            None
        """
        plotter = MPASVerticalCrossSectionPlotter(figsize=(8, 6), dpi=80)

        fig, ax = plotter.create_vertical_cross_section(
            real_3d_processor,
            "theta",
            start_point=(0.0, -20.0),
            end_point=(0.0, 20.0),
            time_index=0,
            vertical_coord="pressure",
            num_points=40,
            style=CrossSectionStyle(plot_type="contourf"),
        )

        assert fig is not None and ax is not None

        out = output_dir / "cross_section"
        plotter.save_plot(str(out), formats=["png"])
        plotter.close_plot()
        _assert_saved(out.with_suffix(".png"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

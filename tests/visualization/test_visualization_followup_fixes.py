#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""
MPASdiag Test Suite: Regression tests for the second round of visualization fixes.

This module covers the wind vector color and opacity styling, the default of the optimal-subsample helper, the density-based automatic subsampling of the base wind plot, the single radians-to-degrees conversion of the cross-section grid fallback, the non-degenerate BoundaryNorm boundaries for user contour levels, and the shared ocean-feature helper that falls back to a background colour when cartopy cannot project the ocean polygons (Antarctic-centred Lambert azimuthal equal-area maps).

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: 2026
Version: 1.0.0
"""

import numpy as np
import pytest
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from typing import Any
from unittest.mock import Mock, patch

from mpasdiag.processing.processors_3d import MPAS3DProcessor
from mpasdiag.processing.utils_geog import GeographicBounds
from mpasdiag.visualization.base_visualizer import MPASVisualizer, WindPlotStyle
from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
from mpasdiag.visualization.surface import MPASSurfacePlotter, SurfaceMapStyle
from mpasdiag.visualization.wind import MPASWindPlotter

_RNG = np.random.default_rng(11)


@pytest.fixture(autouse=True)
def _close_figures() -> Any:
    """
    This autouse fixture closes every matplotlib figure after each test so figures never leak between tests.

    Parameters:
        None

    Returns:
        None
    """
    yield
    plt.close("all")


def _global_mesh(n: int = 4000) -> Any:
    """
    This helper builds a synthetic quasi-uniform global point set with random wind components, used to exercise the automatic subsampling and the vector styling.

    Parameters:
        n (int): Number of points.

    Returns:
        Any: Tuple (lon, lat, u, v) of 1D arrays in degrees and m/s.
    """
    lon = _RNG.uniform(-180, 180, n)
    lat = np.degrees(np.arcsin(_RNG.uniform(-1, 1, n)))
    return lon, lat, _RNG.normal(0, 8, n), _RNG.normal(0, 8, n)


class TestWindVectorStyle:
    """Covers the vector_color and vector_alpha fields of WindPlotStyle in the wind plotter and the base wind plot."""

    def test_defaults_preserve_current_look(self: "TestWindVectorStyle") -> None:
        """
        This test verifies the style defaults are an opaque black vector, which is what was drawn before the fields existed.

        Parameters:
            None

        Returns:
            None
        """
        style = WindPlotStyle()
        assert style.vector_color == "black"
        assert style.vector_alpha == 1.0

    @pytest.mark.parametrize("plot_type", ["barbs", "arrows"])
    def test_wind_plotter_honors_color_and_alpha(
        self: "TestWindVectorStyle", plot_type: str
    ) -> None:
        """
        This test verifies MPASWindPlotter draws barbs and arrows with the requested color and opacity.

        Parameters:
            plot_type (str): Vector representation under test.

        Returns:
            None
        """
        lon, lat, u, v = _global_mesh(300)
        plotter = MPASWindPlotter(figsize=(6, 4), dpi=50)
        _, ax = plotter.create_wind_plot(
            lon,
            lat,
            u,
            v,
            GeographicBounds(-180, 180, -90, 90),
            style=WindPlotStyle(
                plot_type=plot_type,
                subsample=1,
                vector_color="red",
                vector_alpha=0.4,
            ),
        )
        vectors = [c for c in ax.collections if c.get_alpha() == 0.4]
        assert vectors, "no vector collection carries the requested alpha"
        face = np.asarray(vectors[0].get_edgecolor())
        face = face if face.size else np.asarray(vectors[0].get_facecolor())
        assert face[0][0] > 0.9 and face[0][1] < 0.1 and face[0][2] < 0.1

    def test_streamlines_honor_alpha(self: "TestWindVectorStyle") -> None:
        """
        This test verifies streamlines keep the speed colormap but pick up the requested opacity.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASWindPlotter(figsize=(6, 4), dpi=50)
        ax = plt.axes(projection=ccrs.PlateCarree())
        lo, la = np.meshgrid(np.linspace(-120, -60, 20), np.linspace(20, 50, 15))
        plotter._render_wind_vectors(
            ax,
            lo,
            la,
            np.full(lo.shape, 5.0),
            np.full(lo.shape, 2.0),
            "streamlines",
            "red",
            None,
            0.3,
        )
        assert any(c.get_alpha() == 0.3 for c in ax.collections)

    def test_base_plot_honors_color_and_alpha(self: "TestWindVectorStyle") -> None:
        """
        This test verifies the base-class wind plot passes the color and opacity to barbs and arrows.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat, u, v = _global_mesh(300)
        for plot_type in ("barbs", "arrows"):
            viz = MPASVisualizer(figsize=(6, 4), dpi=50)
            _, ax = viz.create_wind_plot(
                lon,
                lat,
                u,
                v,
                GeographicBounds(-180, 180, -90, 90),
                style=WindPlotStyle(
                    plot_type=plot_type,
                    subsample=1,
                    vector_color="blue",
                    vector_alpha=0.5,
                ),
            )
            assert any(c.get_alpha() == 0.5 for c in ax.collections), plot_type


class TestOptimalSubsample:
    """Covers the default and the shared use of calculate_optimal_subsample."""

    def test_default_scales_whole_mesh_count_by_extent(
        self: "TestOptimalSubsample",
    ) -> None:
        """
        This test verifies a direct call with default arguments treats num_points as a whole-mesh count, so a small extent is thinned less than the globe.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASWindPlotter(figsize=(6, 4))
        small = plotter.calculate_optimal_subsample(2_000_000, -100, -90, 30, 40)
        globe = plotter.calculate_optimal_subsample(2_000_000, -180, 180, -90, 90)
        assert small < globe

    def test_in_extent_flag_uses_count_unscaled(
        self: "TestOptimalSubsample",
    ) -> None:
        """
        This test verifies points_in_extent=True uses the supplied count directly, regardless of the extent.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASWindPlotter(figsize=(6, 4))
        a = plotter.calculate_optimal_subsample(
            2_000_000, -100, -90, 30, 40, points_in_extent=True
        )
        b = plotter.calculate_optimal_subsample(
            2_000_000, -180, 180, -90, 90, points_in_extent=True
        )
        assert a == b

    def test_base_auto_subsample_thins_global_mesh(
        self: "TestOptimalSubsample",
    ) -> None:
        """
        This test verifies the base wind plot no longer keeps every vector of a dense global mesh when subsample is non-positive.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat, u, v = _global_mesh(10242)
        viz = MPASVisualizer(figsize=(10, 7), dpi=50)
        lo, _, _, _ = viz._subsample_wind_vectors(
            lon, lat, u, v, 0, GeographicBounds(-180, 180, -90, 90)
        )
        assert 0 < len(lo) < len(lon) // 2


class TestCrossSectionCoordinateFallback:
    """Covers the single radians-to-degrees conversion in the cross-section coordinate lookup."""

    def _processor(self: "TestCrossSectionCoordinateFallback", lon: Any) -> Any:
        """
        This helper builds a Mock processor whose coordinate extraction fails so the lonCell/latCell fallback is used.

        Parameters:
            lon (Any): Longitudes stored in the dataset.

        Returns:
            Any: The mock processor.
        """
        lat = np.linspace(-0.01, 0.01, len(lon))
        proc = Mock(spec=MPAS3DProcessor)
        proc.dataset = xr.Dataset(
            {
                "lonCell": ("nCells", lon),
                "latCell": ("nCells", lat),
                "theta": (("Time", "nCells"), np.zeros((1, len(lon)))),
            }
        )
        proc.extract_2d_coordinates_for_variable = Mock(side_effect=KeyError("x"))
        return proc

    def test_degree_meshes_near_origin_not_converted_again(
        self: "TestCrossSectionCoordinateFallback",
    ) -> None:
        """
        This test verifies coordinates already in degrees with |lon| below pi are left unchanged by the fallback.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASVerticalCrossSectionPlotter()
        lon = np.linspace(-3.0, 3.0, 50)
        proc = self._processor(lon)
        proc.dataset["lonCell"].attrs["units"] = "degrees_east"
        proc.dataset["latCell"].attrs["units"] = "degrees_north"
        got_lon, _ = plotter._extract_cross_section_coords(
            proc, "theta", np.array([0.0, 1.0]), np.array([0.0, 0.0])
        )
        np.testing.assert_allclose(got_lon, lon)

    def test_radian_fallback_converted_once(
        self: "TestCrossSectionCoordinateFallback",
    ) -> None:
        """
        This test verifies radian-valued lonCell/latCell are converted to degrees exactly once.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASVerticalCrossSectionPlotter()
        lon = np.radians(np.linspace(-100, -90, 50))
        proc = self._processor(lon)
        proc.dataset["lonCell"].attrs["units"] = "radians"
        got_lon, _ = plotter._extract_cross_section_coords(
            proc, "theta", np.array([-100.0, -95.0]), np.array([0.0, 0.0])
        )
        np.testing.assert_allclose(got_lon, np.degrees(lon))


class TestBoundaryNormLevels:
    """Covers the BoundaryNorm boundaries built from explicit contour levels."""

    def test_no_duplicated_first_boundary(self: "TestBoundaryNormLevels") -> None:
        """
        This test verifies user levels produce strictly increasing boundaries so every colour band has a non-zero width.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        norm = plotter._build_boundary_norm([0, 10, 20, 30], plt.get_cmap("viridis"))
        bounds = np.asarray(norm.boundaries)
        assert np.all(np.diff(bounds) > 0)
        assert list(bounds[:4]) == [0, 10, 20, 30]


class TestOceanFeatureHelper:
    """Covers the shared _add_ocean_feature helper."""

    def test_projection_check_is_cached(self: "TestOceanFeatureHelper") -> None:
        """
        This test verifies the ocean projectability verdict is computed once per projection and reflects cartopy's behaviour.

        Parameters:
            None

        Returns:
            None
        """
        extent = (-180.0, 180.0, -90.0, -46.6)
        assert MPASVisualizer._ocean_projectable(ccrs.PlateCarree(), extent) is True
        laea = ccrs.LambertAzimuthalEqualArea(0, -75)
        assert MPASVisualizer._ocean_projectable(laea, extent) is False
        with patch.object(
            cfeature.OCEAN, "intersecting_geometries", side_effect=AssertionError
        ):
            assert MPASVisualizer._ocean_projectable(laea, extent) is False

    def test_supported_projection_adds_feature(
        self: "TestOceanFeatureHelper",
    ) -> None:
        """
        This test verifies a normal projection still receives the cartopy ocean feature with the given styling and an untouched background.

        Parameters:
            None

        Returns:
            None
        """
        viz = MPASVisualizer()
        ax = plt.axes(projection=ccrs.PlateCarree())
        before = ax.get_facecolor()
        viz._add_ocean_feature(ax, facecolor="lightblue", alpha=0.3)
        assert any(
            getattr(a, "_feature", None) is cfeature.OCEAN for a in ax.get_children()
        )
        assert ax.get_facecolor() == before

    def test_unsupported_projection_uses_background(
        self: "TestOceanFeatureHelper",
    ) -> None:
        """
        This test verifies an Antarctic-centred Lambert azimuthal equal-area axes gets a blended background colour instead of the crashing feature, and that drawing succeeds.

        Parameters:
            None

        Returns:
            None
        """
        viz = MPASVisualizer()
        fig = plt.figure()
        ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea(0, -75))
        ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
        viz._add_ocean_feature(ax, facecolor="lightblue", alpha=0.5)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
        assert not any(
            getattr(a, "_feature", None) is cfeature.OCEAN for a in ax.get_children()
        )
        expected = 0.5 * np.array([0.678431, 0.847059, 0.901961]) + 0.5
        np.testing.assert_allclose(ax.get_facecolor()[:3], expected, atol=1e-3)
        fig.canvas.draw()

    def test_antarctic_laea_surface_map_renders(
        self: "TestOceanFeatureHelper",
    ) -> None:
        """
        This test verifies a surface scatter map on the Antarctic LAEA projection renders instead of raising a GeometryCollection TypeError.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat, u, _ = _global_mesh(2000)
        plotter = MPASSurfacePlotter(figsize=(5, 5), dpi=40)
        fig, _ = plotter.create_surface_map(
            lon=lon,
            lat=lat,
            data=u,
            var_name="t2m",
            bounds=GeographicBounds(-180, 180, -90, -60),
            style=SurfaceMapStyle(
                plot_type="scatter", projection="LambertAzimuthalEqualArea"
            ),
        )
        fig.canvas.draw()

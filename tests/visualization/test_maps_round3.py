#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""
MPASdiag Test Suite: Regression tests for the third round of map-plotter (surface, precipitation, wind, base and styling) bug fixes.

This module covers the wind speed background for every vector type, the extent- and mesh-derived streamline grid, data-derived colorbar extend arrows, colorbars that span exactly the levels, level_value and the level in the surface title, dateline extents with an explicit central longitude, the pcolormesh plot type and the interpolation option, figure reuse without leaks, save_plot format handling, color-limit and colormap validation, precipitation accumulation-period reconciliation, polar caps, title capitalisation, full batch file paths and extent clipping of barbs and arrows.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: 2026
Version: 1.0.0
"""

import os
import logging
import numpy as np
import pytest
import xarray as xr
import matplotlib.pyplot as plt
from typing import Any, List, Tuple

from mpasdiag.processing.utils_geog import GeographicBounds
from mpasdiag.visualization.base_visualizer import (
    WindPlotStyle,
    projection_central_longitude,
)
from mpasdiag.visualization.precipitation import (
    MPASPrecipitationPlotter,
    PrecipitationRenderStyle,
)
from mpasdiag.visualization.styling import MPASVisualizationStyle
from mpasdiag.visualization.surface import MPASSurfacePlotter, SurfaceMapStyle
from mpasdiag.visualization.wind import MPASWindPlotter

_RNG = np.random.default_rng(3)
_BOUNDS = GeographicBounds(-110.0, -90.0, 30.0, 45.0)
_GLOBAL = GeographicBounds(-180.0, 180.0, -90.0, 90.0)


@pytest.fixture(autouse=True)
def _close_figures() -> Any:
    """
    This autouse fixture closes every matplotlib figure after each test so figures created by one test never leak into another.

    Parameters:
        None

    Returns:
        None
    """
    yield
    plt.close("all")


def _points(n: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    This helper builds random lon/lat points inside the test extent.

    Parameters:
        n (int): Number of points.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Longitudes and latitudes.
    """
    return _RNG.uniform(-109, -91, n), _RNG.uniform(31, 44, n)


def _global_points(n: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
    """
    This helper builds random lon/lat points over the whole globe.

    Parameters:
        n (int): Number of points.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Longitudes in [-180, 180] and latitudes.
    """
    return _RNG.uniform(-180, 180, n), np.degrees(np.arcsin(_RNG.uniform(-1, 1, n)))


def _colorbars(fig: Any) -> List[Any]:
    """
    This helper returns the colorbar axes of a figure.

    Parameters:
        fig (Any): The matplotlib figure.

    Returns:
        List[Any]: The colorbar axes.
    """
    return [a for a in fig.axes if a.get_label() == "<colorbar>"]


class TestWindSpeedBackground:
    """MPASWindPlotter.create_wind_plot honours show_background and bg_colormap."""

    @pytest.mark.parametrize("plot_type", ["barbs", "arrows", "streamlines"])
    def test_background_adds_speed_colorbar(
        self: "TestWindSpeedBackground", plot_type: str
    ) -> None:
        """
        This test verifies that show_background adds a wind-speed layer and a second colorbar for barbs, arrows and streamlines, and that no extra layer or colorbar is added without it.

        Parameters:
            plot_type (str): Wind vector plot type.

        Returns:
            None
        """
        lon, lat = _points()
        u, v = _RNG.normal(0, 8, lon.size), _RNG.normal(0, 8, lon.size)
        counts = []

        for show in (False, True):
            plotter = MPASWindPlotter(figsize=(6, 4), dpi=40)
            fig, _ = plotter.create_wind_plot(
                lon,
                lat,
                u,
                v,
                _BOUNDS,
                style=WindPlotStyle(
                    plot_type=plot_type,
                    subsample=1,
                    show_background=show,
                    bg_colormap="plasma",
                ),
                grid_resolution=1.0 if plot_type == "streamlines" else None,
            )
            counts.append(len(_colorbars(fig)))
            plotter.close_plot()

        assert counts[0] == (1 if plot_type == "streamlines" else 0)
        assert counts[1] == 1

    def test_background_uses_requested_colormap(
        self: "TestWindSpeedBackground",
    ) -> None:
        """
        This test verifies that the background layer is drawn with the colormap named by bg_colormap.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _points()
        u, v = _RNG.normal(0, 8, lon.size), _RNG.normal(0, 8, lon.size)
        plotter = MPASWindPlotter(figsize=(6, 4), dpi=40)
        _, ax = plotter.create_wind_plot(
            lon,
            lat,
            u,
            v,
            _BOUNDS,
            style=WindPlotStyle(
                subsample=1, show_background=True, bg_colormap="plasma"
            ),
        )
        names = [c.get_cmap().name for c in ax.collections if c.get_array() is not None]
        assert "plasma" in names


class TestStreamlineResolution:
    """The default streamline grid follows the extent and the mesh and is capped."""

    def test_global_default_is_coarse_and_capped(
        self: "TestStreamlineResolution",
    ) -> None:
        """
        This test verifies that a global streamline map no longer gets the fixed 0.1 degree grid and that the implied grid stays below the point cap.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _global_points(10242)
        plotter = MPASWindPlotter()
        res = plotter._handle_streamline_regridding(
            "streamlines", None, (-180.0, 180.0, -90.0, 90.0), lon, lat
        )
        assert res is not None and res >= 0.5
        assert (360 / res) * (
            180 / res
        ) <= MPASWindPlotter._MAX_AUTO_STREAM_POINTS * 1.001

    def test_fine_regional_keeps_fine_grid_and_user_value_wins(
        self: "TestStreamlineResolution",
    ) -> None:
        """
        This test verifies that a fine regional mesh keeps the 0.1 degree floor, that no bounds keep the legacy default and that a user supplied resolution is returned unchanged.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _RNG.uniform(-109, -91, 20000), _RNG.uniform(31, 44, 20000)
        plotter = MPASWindPlotter()
        assert plotter._handle_streamline_regridding(
            "streamlines", None, (-110.0, -90.0, 30.0, 45.0), lon, lat
        ) == pytest.approx(0.1)
        assert plotter._handle_streamline_regridding("streamlines", None) == 0.1
        assert plotter._handle_streamline_regridding("streamlines", 0.7) == 0.7
        assert plotter._handle_streamline_regridding("barbs", None) is None


class TestColorbarExtend:
    """Colorbar extend arrows come from the plotted data, not a hard-coded 'both'."""

    def _extends(
        self: "TestColorbarExtend", plot_type: str, levels: List[float]
    ) -> str:
        """
        This helper renders temperature in kelvin (about 10 to 30 degC) with the given levels and returns the extend of the colorbar.

        Parameters:
            plot_type (str): Surface plot type.
            levels (List[float]): Levels in the plotted unit.

        Returns:
            str: The colorbar extend mode.
        """
        lon, lat = _points(800)
        data = _RNG.uniform(283.15, 303.15, lon.size)
        plotter = MPASSurfacePlotter(figsize=(6, 4), dpi=40)
        fig, _ = plotter.create_surface_map(
            lon,
            lat,
            data,
            "t2m",
            _BOUNDS,
            style=SurfaceMapStyle(plot_type=plot_type, levels=levels),
        )
        return str(_colorbars(fig)[0]._colorbar.extend)

    @pytest.mark.parametrize("plot_type", ["scatter", "contourf"])
    def test_levels_wider_than_data_have_no_arrows(
        self: "TestColorbarExtend", plot_type: str
    ) -> None:
        """
        This test verifies that levels covering all of the data give a colorbar without extend arrows.

        Parameters:
            plot_type (str): Surface plot type.

        Returns:
            None
        """
        assert self._extends(plot_type, [0, 10, 20, 30, 40]) == "neither"

    @pytest.mark.parametrize("plot_type", ["scatter", "contourf"])
    def test_levels_inside_data_give_matching_arrows(
        self: "TestColorbarExtend", plot_type: str
    ) -> None:
        """
        This test verifies that an arrow appears only on the side where data actually leaves the level range.

        Parameters:
            plot_type (str): Surface plot type.

        Returns:
            None
        """
        assert self._extends(plot_type, [12, 16, 20, 24]) == "both"
        assert self._extends(plot_type, [0, 10, 15, 20]) == "max"
        assert self._extends(plot_type, [15, 20, 25, 40]) == "min"

    def test_derive_extend_helper(self: "TestColorbarExtend") -> None:
        """
        This test verifies the extend helper for empty, non-finite and bounded inputs.

        Parameters:
            None

        Returns:
            None
        """
        derive = MPASPrecipitationPlotter._derive_extend
        assert derive(None, 0, 1) == "neither"
        assert derive(np.array([np.nan, np.inf]), 0, 1) == "neither"
        assert derive(np.array([0.5]), None, None) == "neither"
        assert derive(np.array([-1.0, 2.0]), 0.0, 1.0) == "both"
        assert derive(np.array([-1.0, 0.5]), 0.0, 1.0) == "min"
        assert derive(np.array([0.5, 2.0]), 0.0, 1.0) == "max"

    def test_precip_scatter_extend_from_data(self: "TestColorbarExtend") -> None:
        """
        This test verifies that a precipitation map whose values stay below the top level has no max arrow while one exceeding it has.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _points(600)
        modes = []

        for scale in (1.0, 100.0):
            plotter = MPASPrecipitationPlotter(figsize=(6, 4), dpi=40)
            fig, _ = plotter.create_precipitation_map(
                lon, lat, _RNG.uniform(0.2, 1.0, lon.size) * scale, _BOUNDS
            )
            modes.append(str(_colorbars(fig)[0]._colorbar.extend))
            plotter.close_plot()

        assert modes == ["neither", "max"]


class TestNoPhantomBound:
    """The BoundaryNorm no longer carries a hidden 'max level + 1' bound."""

    def test_surface_norm_spans_levels(self: "TestNoPhantomBound") -> None:
        """
        This test verifies that the surface boundary norm has exactly the levels as boundaries and sends values above the top to the over region.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        norm = plotter._build_boundary_norm([0.0, 0.5, 1.0], plt.get_cmap("gray"))
        assert list(norm.boundaries) == [0.0, 0.5, 1.0]
        assert norm.extend == "max"
        assert norm(5.0) == norm(1.5) == plt.get_cmap("gray").N
        assert norm(0.25) == 0
        assert plotter._build_boundary_norm([1.0], plt.get_cmap("gray")) is None

    def test_surface_colorbar_stops_at_top_level(self: "TestNoPhantomBound") -> None:
        """
        This test verifies that a cloud-fraction style map shows a colorbar from 0 to exactly 1 rather than to 2.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _points()
        plotter = MPASSurfacePlotter(figsize=(6, 4), dpi=40)
        fig, _ = plotter.create_surface_map(
            lon,
            lat,
            _RNG.uniform(0, 1, lon.size),
            "cldfrac_tot_UPP",
            _BOUNDS,
            style=SurfaceMapStyle(levels=[0, 0.25, 0.5, 0.75, 1.0]),
        )
        low, high = _colorbars(fig)[0].get_xlim()
        assert (low, high) == (0.0, 1.0)

    @pytest.mark.parametrize("units,top", [("mm", 75.0), ("in", 3.0)])
    def test_precip_bounds_end_at_top_level(
        self: "TestNoPhantomBound", units: str, top: float
    ) -> None:
        """
        This test verifies that the precipitation colorbar ends at the last level in millimetres and inches.

        Parameters:
            units (str): Display units.
            top (float): The expected top level.

        Returns:
            None
        """
        lon, lat = _points()
        plotter = MPASPrecipitationPlotter(figsize=(6, 4), dpi=40)
        fig, _ = plotter.create_precipitation_map(
            lon,
            lat,
            _RNG.uniform(0, 1, lon.size),
            _BOUNDS,
            accum_period="a01h",
            style=PrecipitationRenderStyle(display_units=units),
        )
        assert _colorbars(fig)[0].get_xlim()[1] == pytest.approx(top)

    def test_precip_bounds_helper(self: "TestNoPhantomBound") -> None:
        """
        This test verifies that the precipitation bounds are the levels plus an optional zero floor and nothing else.

        Parameters:
            None

        Returns:
            None
        """
        assert MPASPrecipitationPlotter._precip_bounds([1.0, 2.0], True) == [
            0.0,
            1.0,
            2.0,
        ]
        assert MPASPrecipitationPlotter._precip_bounds([-1.0, 2.0], True) == [-1.0, 2.0]
        assert MPASPrecipitationPlotter._precip_bounds([1.0, 2.0], False) == [1.0, 2.0]


class TestLevelSelection:
    """level_value is honoured or rejected, and the level appears in the title."""

    @staticmethod
    def _cube() -> Tuple[np.ndarray, np.ndarray, xr.DataArray]:
        """
        This helper builds a (nCells, nVertLevels) temperature cube whose level k has the constant value 250 + 10 k kelvin together with a DataArray carrying a pressure coordinate.

        Parameters:
            None

        Returns:
            Tuple[np.ndarray, np.ndarray, xr.DataArray]: Longitudes, latitudes and the cube as DataArray.
        """
        lon, lat = _points(300)
        cube = np.stack([np.full(lon.size, 250.0 + 10 * k) for k in range(4)], axis=1)
        da = xr.DataArray(
            cube,
            dims=("nCells", "nVertLevels"),
            coords={"nVertLevels": [1000.0, 850.0, 500.0, 300.0]},
            attrs={"units": "K"},
        )
        da.coords["nVertLevels"].attrs["units"] = "hPa"
        return lon, lat, da

    def test_level_value_selects_matching_level(self: "TestLevelSelection") -> None:
        """
        This test verifies that level_value picks the level whose coordinate matches and that the title names it.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat, da = self._cube()
        plotter = MPASSurfacePlotter(figsize=(6, 4), dpi=40)
        _, ax = plotter.create_surface_map(
            lon, lat, da.values, "t2m", _BOUNDS, data_array=da, level_value=500.0
        )
        values = np.unique(
            np.round(
                np.asarray(
                    [c for c in ax.collections if c.get_array() is not None][
                        0
                    ].get_array()
                ),
                2,
            )
        )
        assert values == pytest.approx([270.0 - 273.15])
        assert "500 hPa" in ax.get_title() and "index 2" in ax.get_title()

    def test_level_value_without_coordinate_raises(self: "TestLevelSelection") -> None:
        """
        This test verifies that level_value never goes silently ignored: it raises for a missing data array, an unknown value and a conflict with level_index.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat, da = self._cube()
        plotter = MPASSurfacePlotter(figsize=(6, 4), dpi=40)

        with pytest.raises(ValueError, match="data_array"):
            plotter.create_surface_map(
                lon, lat, da.values, "t2m", _BOUNDS, level_value=500.0
            )

        with pytest.raises(ValueError, match="does not match"):
            plotter.create_surface_map(
                lon, lat, da.values, "t2m", _BOUNDS, data_array=da, level_value=123.0
            )

        with pytest.raises(ValueError, match="different"):
            plotter.create_surface_map(
                lon,
                lat,
                da.values,
                "t2m",
                _BOUNDS,
                data_array=da,
                level_value=500.0,
                level_index=0,
            )

        assert plt.get_fignums() == []

    def test_title_shows_level_index_and_default_warns(
        self: "TestLevelSelection", caplog: Any
    ) -> None:
        """
        This test verifies that the title shows the level index when given, and that the defaulted level 0 is shown with a logged warning.

        Parameters:
            caplog (Any): pytest log capture fixture.

        Returns:
            None
        """
        lon, lat, da = self._cube()
        plotter = MPASSurfacePlotter(figsize=(6, 4), dpi=40)
        _, ax = plotter.create_surface_map(
            lon, lat, da.values, "t2m", _BOUNDS, level_index=3
        )
        assert "Level index 3" in ax.get_title()

        with caplog.at_level(logging.WARNING):
            _, ax = plotter.create_surface_map(lon, lat, da.values, "t2m", _BOUNDS)

        assert "Level index 0" in ax.get_title()
        assert any("lowest model level" in r.getMessage() for r in caplog.records)

    def test_1d_title_has_no_level(self: "TestLevelSelection") -> None:
        """
        This test verifies that a plain 1D field gets no level text in its title.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _points()
        plotter = MPASSurfacePlotter(figsize=(6, 4), dpi=40)
        _, ax = plotter.create_surface_map(
            lon, lat, _RNG.uniform(280, 300, lon.size), "t2m", _BOUNDS
        )
        assert "Level" not in ax.get_title()


def proj_center(ax: Any) -> float:
    """
    This helper returns the absolute central longitude of the axes projection.

    Parameters:
        ax (Any): A GeoAxes.

    Returns:
        float: The absolute central longitude in degrees.
    """
    return abs(float(projection_central_longitude(ax.projection)))


class TestDatelineWithExplicitCenter:
    """An explicit central_longitude that would split the extent re-centres the map."""

    @pytest.mark.parametrize("projection", ["Robinson", "Mercator", "PlateCarree"])
    def test_projection_recentres_on_midpoint(
        self: "TestDatelineWithExplicitCenter", projection: str, caplog: Any
    ) -> None:
        """
        This test verifies that central_longitude=0 with a 150..210 extent gives a projection centred at 180 and logs a warning.

        Parameters:
            projection (str): Projection name.
            caplog (Any): pytest log capture fixture.

        Returns:
            None
        """
        with caplog.at_level(logging.WARNING):
            proj, _ = MPASVisualizationStyle.setup_map_projection(
                150.0, 210.0, -40.0, 40.0, projection, central_longitude=0.0
            )

        assert abs(abs(projection_central_longitude(proj)) - 180.0) < 1e-9
        assert any("re-centering" in r.getMessage() for r in caplog.records)

    def test_compatible_center_is_kept(self: "TestDatelineWithExplicitCenter") -> None:
        """
        This test verifies that an explicit centre that does not split the extent is left alone.

        Parameters:
            None

        Returns:
            None
        """
        proj, _ = MPASVisualizationStyle.setup_map_projection(
            -100.0, -60.0, 20.0, 50.0, "Robinson", central_longitude=-80.0
        )
        assert projection_central_longitude(proj) == -80.0
        proj, _ = MPASVisualizationStyle.setup_map_projection(
            150.0, 210.0, -40.0, 40.0, "Robinson", central_longitude=180.0
        )
        assert abs(projection_central_longitude(proj)) == 180.0

    def test_surface_map_shows_the_extent(
        self: "TestDatelineWithExplicitCenter",
    ) -> None:
        """
        This test verifies that the surface map with central_longitude=0 and extent 150..210 displays that extent instead of falling back to a global view.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _global_points(4000)
        plotter = MPASSurfacePlotter(figsize=(6, 4), dpi=40)
        _, ax = plotter.create_surface_map(
            lon,
            lat,
            _RNG.uniform(270, 300, lon.size),
            "t2m",
            GeographicBounds(150.0, 210.0, -40.0, 40.0),
            style=SurfaceMapStyle(projection="Robinson", central_longitude=0.0),
        )
        full = ax.projection.x_limits
        x0, x1 = ax.get_xlim()
        assert abs(proj_center(ax) - 180.0) < 1e-9
        assert (x1 - x0) < 0.5 * (full[1] - full[0])


class TestPcolormeshAndInterpolation:
    """plot_type='pcolormesh' and SurfaceMapStyle.interpolation."""

    def test_style_has_interpolation_default_none(
        self: "TestPcolormeshAndInterpolation",
    ) -> None:
        """
        This test verifies the new style field and its default.

        Parameters:
            None

        Returns:
            None
        """
        assert SurfaceMapStyle().interpolation is None
        assert SurfaceMapStyle(interpolation="linear").interpolation == "linear"

    @pytest.mark.parametrize("interpolation", [None, "nearest", "linear"])
    def test_pcolormesh_renders_mesh_and_colorbar(
        self: "TestPcolormeshAndInterpolation", interpolation: Any
    ) -> None:
        """
        This test verifies that pcolormesh draws a QuadMesh and a colorbar for each gridding method.

        Parameters:
            interpolation (Any): Interpolation setting.

        Returns:
            None
        """
        lon, lat = _points(1500)
        plotter = MPASSurfacePlotter(figsize=(6, 4), dpi=40)
        fig, ax = plotter.create_surface_map(
            lon,
            lat,
            _RNG.uniform(280, 300, lon.size),
            "t2m",
            _BOUNDS,
            style=SurfaceMapStyle(
                plot_type="pcolormesh", interpolation=interpolation, grid_resolution=1.0
            ),
        )
        assert any("QuadMesh" in type(c).__name__ for c in ax.collections)
        assert len(_colorbars(fig)) == 1

    def test_interpolation_changes_the_result(
        self: "TestPcolormeshAndInterpolation",
    ) -> None:
        """
        This test verifies that nearest and linear gridding give different meshes for a smooth field, so the option is not ignored.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _points(1500)
        data = 285.0 + 0.2 * (lon + 110.0) + 0.3 * (lat - 30.0)
        arrays = {}

        for interp in ("nearest", "linear"):
            plotter = MPASSurfacePlotter(figsize=(6, 4), dpi=40)
            _, ax = plotter.create_surface_map(
                lon,
                lat,
                data,
                "t2m",
                _BOUNDS,
                style=SurfaceMapStyle(
                    plot_type="pcolormesh", interpolation=interp, grid_resolution=0.5
                ),
            )
            mesh = [c for c in ax.collections if "QuadMesh" in type(c).__name__][0]
            arrays[interp] = np.ma.filled(mesh.get_array(), np.nan)
            plotter.close_plot()

        assert not np.allclose(arrays["nearest"], arrays["linear"], equal_nan=True)

    def test_invalid_interpolation_raises(
        self: "TestPcolormeshAndInterpolation",
    ) -> None:
        """
        This test verifies that an unknown interpolation name raises before a figure is created.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _points()
        plotter = MPASSurfacePlotter()

        with pytest.raises(ValueError, match="interpolation"):
            plotter.create_surface_map(
                lon,
                lat,
                _RNG.uniform(280, 300, lon.size),
                "t2m",
                _BOUNDS,
                style=SurfaceMapStyle(plot_type="contourf", interpolation="spline"),
            )

        assert plt.get_fignums() == []


class TestFigureReuse:
    """Reusing one plotter without close_plot no longer leaks figures."""

    def test_surface_precip_wind_reuse(self: "TestFigureReuse") -> None:
        """
        This test verifies that three consecutive create calls on one plotter of each kind leave exactly one open figure.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _points()
        data = _RNG.uniform(1, 20, lon.size)

        plotter = MPASSurfacePlotter(figsize=(5, 3), dpi=30)
        for _ in range(3):
            plotter.create_surface_map(lon, lat, data + 280, "t2m", _BOUNDS)
        assert len(plt.get_fignums()) == 1
        plotter.close_plot()
        assert plt.get_fignums() == []

        plotter_p = MPASPrecipitationPlotter(figsize=(5, 3), dpi=30)
        for _ in range(3):
            plotter_p.create_precipitation_map(lon, lat, data, _BOUNDS)
        assert len(plt.get_fignums()) == 1
        plotter_p.close_plot()

        plotter_w = MPASWindPlotter(figsize=(5, 3), dpi=30)
        for _ in range(3):
            plotter_w.create_wind_plot(
                lon, lat, data, data, _BOUNDS, style=WindPlotStyle(subsample=1)
            )
        assert len(plt.get_fignums()) == 1
        plotter_w.close_plot()
        assert plt.get_fignums() == []

    def test_earlier_figure_object_stays_usable(
        self: "TestFigureReuse", tmp_path: Any
    ) -> None:
        """
        This test verifies that the figure returned by an earlier call can still be saved after the plotter was reused.

        Parameters:
            tmp_path (Any): pytest temporary directory.

        Returns:
            None
        """
        lon, lat = _points()
        plotter = MPASSurfacePlotter(figsize=(5, 3), dpi=30)
        first, _ = plotter.create_surface_map(
            lon, lat, _RNG.uniform(280, 300, lon.size), "t2m", _BOUNDS
        )
        plotter.create_surface_map(
            lon, lat, _RNG.uniform(280, 300, lon.size), "t2m", _BOUNDS
        )
        first.savefig(str(tmp_path / "first.png"))
        assert (tmp_path / "first.png").stat().st_size > 0


class TestSavePlot:
    """save_plot accepts a string, normalizes formats and never doubles extensions."""

    @staticmethod
    def _plotter() -> MPASSurfacePlotter:
        """
        This helper returns a surface plotter that already holds a small figure.

        Parameters:
            None

        Returns:
            MPASSurfacePlotter: The plotter.
        """
        lon, lat = _points(200)
        plotter = MPASSurfacePlotter(figsize=(4, 3), dpi=30)
        plotter.create_surface_map(
            lon, lat, _RNG.uniform(280, 300, lon.size), "t2m", _BOUNDS
        )
        return plotter

    def test_string_dot_and_case_normalized(
        self: "TestSavePlot", tmp_path: Any
    ) -> None:
        """
        This test verifies that a single string, a leading dot and upper case all produce a lowercase extension.

        Parameters:
            tmp_path (Any): pytest temporary directory.

        Returns:
            None
        """
        plotter = self._plotter()
        plotter.save_plot(str(tmp_path / "a"), "png")
        plotter.save_plot(str(tmp_path / "b"), [".PNG"])
        plotter.save_plot(str(tmp_path / "c"), ["PDF", ".pdf"])
        assert sorted(os.listdir(tmp_path)) == ["a.png", "b.png", "c.pdf"]

    def test_extension_not_doubled(self: "TestSavePlot", tmp_path: Any) -> None:
        """
        This test verifies that a path that already ends with the requested extension is not given a second one, while another format still gets its own extension.

        Parameters:
            tmp_path (Any): pytest temporary directory.

        Returns:
            None
        """
        plotter = self._plotter()
        plotter.save_plot(str(tmp_path / "map.png"), ["png"])
        plotter.save_plot(str(tmp_path / "map.png"), ["png", "pdf"])
        assert sorted(os.listdir(tmp_path)) == ["map.pdf", "map.png"]

    def test_empty_and_unknown_formats_raise(
        self: "TestSavePlot", tmp_path: Any
    ) -> None:
        """
        This test verifies that an empty list, blank names and an unsupported format all raise ValueError without writing anything.

        Parameters:
            tmp_path (Any): pytest temporary directory.

        Returns:
            None
        """
        plotter = self._plotter()

        for bad in ([], [""], "", ["bogus"]):
            with pytest.raises(ValueError):
                plotter.save_plot(str(tmp_path / "x"), bad)

        assert os.listdir(tmp_path) == []

    def test_no_figure_raises_value_error_everywhere(
        self: "TestSavePlot", tmp_path: Any
    ) -> None:
        """
        This test verifies that the base, surface and precipitation plotters raise the same ValueError when there is no figure, including after close_plot.

        Parameters:
            tmp_path (Any): pytest temporary directory.

        Returns:
            None
        """
        for plotter in (
            MPASSurfacePlotter(),
            MPASPrecipitationPlotter(),
            MPASWindPlotter(),
        ):
            with pytest.raises(ValueError, match="No figure to save"):
                plotter.save_plot(str(tmp_path / "x"))

        plotter = self._plotter()
        plotter.close_plot()

        with pytest.raises(ValueError, match="No figure to save"):
            plotter.save_plot(str(tmp_path / "x"))

        with pytest.raises(ValueError, match="No figure to save"):
            MPASVisualizationStyle.save_plot(None, str(tmp_path / "x"))  # type: ignore[arg-type]

    def test_precip_plotter_uses_same_rules(
        self: "TestSavePlot", tmp_path: Any
    ) -> None:
        """
        This test verifies that the precipitation plotter shares the normalization and the empty-formats error.

        Parameters:
            tmp_path (Any): pytest temporary directory.

        Returns:
            None
        """
        lon, lat = _points(200)
        plotter = MPASPrecipitationPlotter(figsize=(4, 3), dpi=30)
        plotter.create_precipitation_map(
            lon, lat, _RNG.uniform(0, 5, lon.size), _BOUNDS
        )
        plotter.save_plot(str(tmp_path / "p.png"), ".PNG")
        assert os.listdir(tmp_path) == ["p.png"]

        with pytest.raises(ValueError):
            plotter.save_plot(str(tmp_path / "q"), [])

    def test_saved_file_paths_helper(self: "TestSavePlot") -> None:
        """
        This test verifies the helper that reports the files a save writes.

        Parameters:
            None

        Returns:
            None
        """
        paths = MPASVisualizationStyle.saved_file_paths(
            "/o/map", ["PNG", ".pdf", "png"]
        )
        assert paths == ["/o/map.png", "/o/map.pdf"]
        assert MPASVisualizationStyle.saved_file_paths("/o/map.png", "png") == [
            "/o/map.png"
        ]


class TestColorLimitAndColormapValidation:
    """Degenerate color limits and unknown colormaps raise ValueError."""

    def test_one_sided_clim_beyond_data_raises(
        self: "TestColorLimitAndColormapValidation",
    ) -> None:
        """
        This test verifies that clim_min above every plotted value, or clim_max below, raises ValueError and leaves no figure open.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _points()
        data = _RNG.uniform(283.15, 303.15, lon.size)
        plotter = MPASSurfacePlotter()

        for kw in ({"clim_min": 270.0}, {"clim_max": -50.0}):
            with pytest.raises(ValueError, match="vmin"):
                plotter.create_surface_map(
                    lon, lat, data, "t2m", _BOUNDS, style=SurfaceMapStyle(**kw)
                )

        assert plt.get_fignums() == []

    def test_one_sided_clim_inside_data_works(
        self: "TestColorLimitAndColormapValidation",
    ) -> None:
        """
        This test verifies that a sensible one-sided limit still renders.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _points()
        plotter = MPASSurfacePlotter(figsize=(4, 3), dpi=30)
        fig, _ = plotter.create_surface_map(
            lon,
            lat,
            _RNG.uniform(283.15, 303.15, lon.size),
            "t2m",
            _BOUNDS,
            style=SurfaceMapStyle(clim_min=15.0),
        )
        assert _colorbars(fig)[0].get_xlim()[0] == pytest.approx(15.0)

    def test_unknown_colormap_raises(
        self: "TestColorLimitAndColormapValidation",
    ) -> None:
        """
        This test verifies that the surface plotter rejects an unknown colormap name like the precipitation plotter does.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _points()
        data = _RNG.uniform(283.15, 303.15, lon.size)

        with pytest.raises(ValueError, match="colormap"):
            MPASSurfacePlotter().create_surface_map(
                lon,
                lat,
                data,
                "t2m",
                _BOUNDS,
                style=SurfaceMapStyle(colormap="nope_cmap"),
            )

        with pytest.raises(ValueError):
            MPASPrecipitationPlotter().create_precipitation_map(
                lon,
                lat,
                data - 280,
                _BOUNDS,
                style=PrecipitationRenderStyle(colormap="nope_cmap"),
            )

        assert plt.get_fignums() == []

    def test_precip_clim_leaving_one_level_raises(
        self: "TestColorLimitAndColormapValidation",
    ) -> None:
        """
        This test verifies that a one-sided precipitation limit beyond the whole level scheme raises a ValueError.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()

        with pytest.raises(ValueError, match="fewer than two"):
            plotter._prepare_precipitation_colormap(None, None, "a01h", 500.0, None)


class TestPrecipAccumulationReconciliation:
    """The plotted levels and annotation follow the accumulation period of the data."""

    @staticmethod
    def _data() -> Tuple[np.ndarray, np.ndarray, xr.DataArray]:
        """
        This helper builds precipitation data whose metadata says it is a 1-hour accumulation.

        Parameters:
            None

        Returns:
            Tuple[np.ndarray, np.ndarray, xr.DataArray]: Longitudes, latitudes and the DataArray.
        """
        lon, lat = _points(300)
        da = xr.DataArray(
            _RNG.uniform(0, 3, lon.size),
            dims=("nCells",),
            attrs={
                "units": "mm",
                "accumulation_period": "a01h",
                "accumulation_hours": 1,
                "valid_time": "2024-09-17T03:00:00",
                "accumulation_start_time": "2024-09-17T02:00:00",
            },
        )
        return lon, lat, da

    def test_mismatch_warns_and_follows_metadata(
        self: "TestPrecipAccumulationReconciliation", caplog: Any
    ) -> None:
        """
        This test verifies that requesting a24h for 1-hour data logs a warning and gives the 1-hour levels and annotation.

        Parameters:
            caplog (Any): pytest log capture fixture.

        Returns:
            None
        """
        lon, lat, da = self._data()
        plotter = MPASPrecipitationPlotter(figsize=(5, 3), dpi=30)

        with caplog.at_level(logging.WARNING):
            fig, ax = plotter.create_precipitation_map(
                lon, lat, da.values, _BOUNDS, accum_period="a24h", data_array=da
            )

        assert any("does not match" in r.getMessage() for r in caplog.records)
        assert any("(1 h)" in t.get_text() for t in ax.texts)
        assert _colorbars(fig)[0].get_xlim()[1] == pytest.approx(75.0)

    def test_reconcile_helper(self: "TestPrecipAccumulationReconciliation") -> None:
        """
        This test verifies the helper for matching, differing, missing and unparsable metadata.

        Parameters:
            None

        Returns:
            None
        """
        _, _, da = self._data()
        reconcile = MPASPrecipitationPlotter._reconcile_accum_period
        assert reconcile("a01h", da) == "a01h"
        assert reconcile("a24h", da) == "a01h"
        assert reconcile("a24h", None) == "a24h"
        bad = da.copy()
        bad.attrs = {"accumulation_period": "garbage"}
        assert reconcile("a24h", bad) == "a24h"
        hours_only = da.copy()
        hours_only.attrs = {"accumulation_hours": 6}
        assert reconcile("a24h", hours_only) == "a06h"


class TestPolarCap:
    """Polar caps get a circular boundary."""

    @pytest.mark.parametrize(
        "projection,bounds",
        [
            ("NorthPolarStereo", GeographicBounds(-180.0, 180.0, 45.0, 90.0)),
            ("SouthPolarStereo", GeographicBounds(-180.0, 180.0, -90.0, -45.0)),
        ],
    )
    def test_boundary_is_circle(
        self: "TestPolarCap", projection: str, bounds: GeographicBounds
    ) -> None:
        """
        This test verifies that a polar cap map has a curved (many-vertex) boundary instead of the rectangular default.

        Parameters:
            projection (str): Polar projection name.
            bounds (GeographicBounds): Cap extent.

        Returns:
            None
        """
        lon, lat = _global_points(4000)
        plotter = MPASSurfacePlotter(figsize=(4, 4), dpi=30)
        _, ax = plotter.create_surface_map(
            lon,
            lat,
            _RNG.uniform(250, 290, lon.size),
            "t2m",
            bounds,
            style=SurfaceMapStyle(projection=projection),
        )
        assert len(ax.patch.get_path().vertices) > 50

    def test_regional_polar_extent_keeps_square(self: "TestPolarCap") -> None:
        """
        This test verifies that a regional box on a polar projection is not given the circular boundary.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _global_points(4000)
        plotter = MPASSurfacePlotter(figsize=(4, 4), dpi=30)
        _, ax = plotter.create_surface_map(
            lon,
            lat,
            _RNG.uniform(250, 290, lon.size),
            "t2m",
            GeographicBounds(-60.0, 60.0, 50.0, 85.0),
            style=SurfaceMapStyle(projection="NorthPolarStereo"),
        )
        assert len(ax.patch.get_path().vertices) <= 10


class TestTitleCapitalisation:
    """The default title does not depend on whether data_array is passed."""

    def test_same_title_with_and_without_data_array(
        self: "TestTitleCapitalisation",
    ) -> None:
        """
        This test verifies that a file long_name that only differs in capitalisation gives the same title as the metadata default.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _points()
        data = _RNG.uniform(283.15, 303.15, lon.size)
        da = xr.DataArray(
            data,
            dims=("nCells",),
            attrs={"units": "K", "long_name": "2-meter temperature"},
        )
        titles = []

        for kw in ({}, {"data_array": da}):
            plotter = MPASSurfacePlotter(figsize=(4, 3), dpi=30)
            _, ax = plotter.create_surface_map(lon, lat, data, "t2m", _BOUNDS, **kw)
            titles.append(ax.get_title())
            plotter.close_plot()

        assert titles[0] == titles[1] == "MPAS 2-meter Temperature"

    def test_distinct_long_name_is_kept(self: "TestTitleCapitalisation") -> None:
        """
        This test verifies that a long_name with different wording is not replaced.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _points()
        da = xr.DataArray(
            _RNG.uniform(283.15, 303.15, lon.size),
            dims=("nCells",),
            attrs={"units": "K", "long_name": "Screen level air temperature"},
        )
        plotter = MPASSurfacePlotter(figsize=(4, 3), dpi=30)
        _, ax = plotter.create_surface_map(
            lon, lat, da.values, "t2m", _BOUNDS, data_array=da
        )
        assert ax.get_title() == "MPAS Screen level air temperature"


class _FakeProcessor:
    """Minimal processor exposing the interface used by the batch wind plot."""

    def __init__(self: "_FakeProcessor") -> None:
        """
        This initializer builds a two-time-step dataset with u10 and v10 on a small random mesh.

        Parameters:
            None

        Returns:
            None
        """
        self.lon, self.lat = _points(300)
        n = self.lon.size
        self.dataset = xr.Dataset(
            {
                "u10": (("Time", "nCells"), _RNG.normal(0, 5, (2, n))),
                "v10": (("Time", "nCells"), _RNG.normal(0, 5, (2, n))),
            },
            coords={
                "Time": np.array(
                    ["2024-09-17T00", "2024-09-17T01"], dtype="datetime64[ns]"
                )
            },
        )

    def get_2d_variable_data(
        self: "_FakeProcessor", name: str, time_idx: int
    ) -> xr.DataArray:
        """
        This method returns one time slice of a variable.

        Parameters:
            name (str): Variable name.
            time_idx (int): Time index.

        Returns:
            xr.DataArray: The slice.
        """
        return self.dataset[name].isel(Time=time_idx)

    def extract_2d_coordinates_for_variable(
        self: "_FakeProcessor", name: str, data: xr.DataArray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method returns the mesh coordinates.

        Parameters:
            name (str): Variable name.
            data (xr.DataArray): The data slice (unused).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Longitudes and latitudes.
        """
        return self.lon, self.lat


class TestBatchWindPaths:
    """create_batch_wind_plots returns the full file paths."""

    def test_returns_full_paths_with_extension(
        self: "TestBatchWindPaths", tmp_path: Any
    ) -> None:
        """
        This test verifies that every returned path exists on disk and carries its format extension, one per format and time step.

        Parameters:
            tmp_path (Any): pytest temporary directory.

        Returns:
            None
        """
        plotter = MPASWindPlotter(figsize=(4, 3), dpi=30)
        files = plotter.create_batch_wind_plots(
            _FakeProcessor(),
            str(tmp_path),
            _BOUNDS,
            u_variable="u10",
            v_variable="v10",
            formats=["png", "pdf"],
            subsample=1,
        )
        assert len(files) == 4
        assert all(os.path.isfile(f) for f in files)
        assert {os.path.splitext(f)[1] for f in files} == {".png", ".pdf"}
        assert plt.get_fignums() == []


class TestWindExtentClipping:
    """Vectors outside the extent are dropped before subsampling; non-finite values are removed."""

    def test_clip_drops_outside_and_infinite(self: "TestWindExtentClipping") -> None:
        """
        This test verifies that only in-extent finite vectors remain and that +/-inf is removed.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _global_points(5000)
        u, v = _RNG.normal(0, 5, lon.size), _RNG.normal(0, 5, lon.size)
        u[:20] = np.inf
        v[20:40] = -np.inf
        plotter = MPASWindPlotter()
        lon_p, lat_p, u_p, v_p = plotter._prepare_wind_data(
            lon, lat, u, v, 1, bounds=(-110.0, -90.0, 30.0, 45.0)
        )
        assert lon_p.size > 0 and lon_p.size < 500
        assert np.all(np.isfinite(u_p)) and np.all(np.isfinite(v_p))
        assert lon_p.min() >= -110.0 - 1.0 and lon_p.max() <= -90.0 + 1.0
        assert lat_p.min() >= 30.0 - 0.75 and lat_p.max() <= 45.0 + 0.75

    def test_infinite_removed_without_bounds(self: "TestWindExtentClipping") -> None:
        """
        This test verifies that +/-inf values are dropped even when no extent is given.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _points(100)
        u, v = np.ones(100), np.ones(100)
        u[0], v[1] = np.inf, -np.inf
        out = MPASWindPlotter()._prepare_wind_data(lon, lat, u, v, 1)
        assert out[0].size == 98

    def test_clip_is_dateline_aware(self: "TestWindExtentClipping") -> None:
        """
        This test verifies that a 150..210 extent keeps vectors on both sides of the dateline and returns longitudes in the extent frame.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _global_points(6000)
        u = v = np.ones(lon.size)
        lon_p, _, _, _ = MPASWindPlotter()._clip_to_extent(
            lon, lat, u, v, (150.0, 210.0, -40.0, 40.0)
        )
        assert lon_p.size > 0
        assert lon_p.min() >= 150.0 - 3.0 and lon_p.max() <= 210.0 + 3.0
        assert (lon_p > 180.0).any() and (lon_p < 180.0).any()

    def test_gridded_input_is_not_clipped(self: "TestWindExtentClipping") -> None:
        """
        This test verifies that two-dimensional grids keep their structure.

        Parameters:
            None

        Returns:
            None
        """
        lon2, lat2 = np.meshgrid(np.linspace(0, 10, 6), np.linspace(0, 10, 5))
        u = np.ones_like(lon2)
        out = MPASWindPlotter()._clip_to_extent(lon2, lat2, u, u, (2.0, 4.0, 2.0, 4.0))
        assert out[0].shape == lon2.shape

    def test_plot_title_stats_use_only_extent(self: "TestWindExtentClipping") -> None:
        """
        This test verifies that a regional wind plot of a global field still renders barbs only for the extent.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _global_points(6000)
        u = np.full(lon.size, 10.0)
        v = np.zeros(lon.size)
        plotter = MPASWindPlotter(figsize=(5, 3), dpi=30)
        _, ax = plotter.create_wind_plot(
            lon, lat, u, v, _BOUNDS, style=WindPlotStyle(subsample=1)
        )
        barbs = [c for c in ax.collections if type(c).__name__ == "Barbs"]
        assert barbs and len(barbs[0].get_offsets()) < 500


class TestWindTitleStatistics:
    """Integration follow-up: wind title statistics describe the field, not the drawn vectors."""

    def test_title_stats_independent_of_thinning(
        self: "TestWindTitleStatistics",
    ) -> None:
        """
        This test verifies the Max/Mean in the default wind title equal the speed statistics of the source field inside the extent, whatever the subsample factor or plot type (previously they came from the thinned or regridded vectors, so a single strong cell could be missed).
        """
        lon, lat = _points()
        u = np.full(lon.size, 3.0)
        v = np.full(lon.size, 4.0)
        b = _BOUNDS
        inside = np.flatnonzero(
            (lon >= b.lon_min)
            & (lon <= b.lon_max)
            & (lat >= b.lat_min)
            & (lat <= b.lat_max)
        )
        u[inside[1]] = 60.0
        v[inside[1]] = 80.0
        speed = np.hypot(u, v)[inside]
        expected = f"Max: {speed.max():.1f} m/s, Mean: {speed.mean():.1f} m/s"

        for plot_type, subsample in (("barbs", 1), ("barbs", 7), ("streamlines", 1)):
            plotter = MPASWindPlotter(figsize=(6, 4), dpi=40)
            _, ax = plotter.create_wind_plot(
                lon,
                lat,
                u,
                v,
                b,
                style=WindPlotStyle(plot_type=plot_type, subsample=subsample),
                grid_resolution=1.0 if plot_type == "streamlines" else None,
            )
            assert expected in ax.get_title(), (plot_type, subsample, ax.get_title())
            plotter.close_plot()

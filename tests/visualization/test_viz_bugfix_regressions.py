#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""
MPASdiag Test Suite: Regression tests for visualization-layer bug fixes.

This module collects regression tests for defects fixed in the visualization layer: the cross-section distance axis and per-column vertical coordinate, default lowest-level selection for 3D fields, wind barbs in knots, wind overlay bounds, extent-aware subsampling, precipitation-name styling rules, batch filename/time handling, UTC footer labelling, great-circle transect lines, one-sided precipitation color limits, exact tick labels, and Skew-T ordering and layout.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: 2026
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from typing import Any, List
from unittest.mock import MagicMock, Mock, patch

from mpasdiag.diagnostics.precipitation import InsufficientLookbackError
from mpasdiag.processing.processors_3d import MPAS3DProcessor
from mpasdiag.processing.utils_geog import GeographicBounds
from mpasdiag.visualization.base_visualizer import MPASVisualizer, WindPlotStyle
from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter
from mpasdiag.visualization.skewt import MPASSkewTPlotter
from mpasdiag.visualization.styling import MPASVisualizationStyle
from mpasdiag.visualization.surface import MPASSurfacePlotter
from mpasdiag.visualization.wind import MPASWindPlotter

_RNG = np.random.default_rng(7)


@pytest.fixture(autouse=True)
def _close_figures() -> Any:
    """
    This autouse fixture closes every matplotlib figure after each test so figures created by one regression test never leak into another.

    Parameters:
        None

    Returns:
        None
    """
    yield
    plt.close("all")


def _make_processor(lons: np.ndarray, lats: np.ndarray, n_lev: int = 6) -> Any:
    """
    This helper builds a Mock MPAS3DProcessor around a small synthetic dataset with per-column pressure (surface pressure varying with cell longitude) and a terrain-dependent zgrid, so the per-column vertical coordinate can be verified.

    Parameters:
        lons (np.ndarray): Cell longitudes in degrees.
        lats (np.ndarray): Cell latitudes in degrees.
        n_lev (int): Number of mid-level vertical levels.

    Returns:
        Any: A Mock(spec=MPAS3DProcessor) whose dataset holds theta, pressure_p, pressure_base, zgrid, lonCell and latCell.
    """
    n_cells = len(lons)
    terrain = 500.0 * (lons - lons.min()) / max(np.ptp(lons), 1.0)
    p_surface = 100000.0 - 5.0 * terrain
    frac = np.linspace(0.0, 0.8, n_lev)
    pressure = p_surface[:, None] * (1.0 - frac[None, :])
    iface = np.linspace(0.0, 1.0, n_lev + 1)
    zgrid = terrain[:, None] + iface[None, :] * (12000.0 - terrain[:, None])
    times = pd.date_range("2025-01-01", periods=2, freq="30min")

    ds = xr.Dataset(
        {
            "theta": (
                ["Time", "nCells", "nVertLevels"],
                300.0 + _RNG.random((2, n_cells, n_lev)),
            ),
            "pressure_p": (
                ["Time", "nCells", "nVertLevels"],
                np.zeros((2, n_cells, n_lev)) + pressure[None],
            ),
            "pressure_base": (
                ["Time", "nCells", "nVertLevels"],
                np.zeros((2, n_cells, n_lev)),
            ),
            "zgrid": (["nCells", "nVertLevelsP1"], zgrid),
            "lonCell": ("nCells", lons),
            "latCell": ("nCells", lats),
        },
        coords={"Time": times},
    )
    proc = Mock(spec=MPAS3DProcessor)
    proc.dataset = ds
    proc.grid_file = None
    proc.extract_2d_coordinates_for_variable = Mock(return_value=(lons, lats))
    proc.get_vertical_levels = Mock(return_value=list(np.nanmean(pressure, axis=0)))
    return proc


class TestCrossSectionAxes:
    """Cross-section horizontal axis and per-column vertical coordinate."""

    def test_north_south_section_uses_distance_axis(
        self: "TestCrossSectionAxes",
    ) -> None:
        """
        This test verifies a north-south cross-section is drawn against along-path distance in km rather than longitude, which is constant for such a path and previously collapsed the section.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = np.meshgrid(np.linspace(-100, -90, 20), np.linspace(30, 50, 40))
        proc = _make_processor(lon.ravel(), lat.ravel())
        plotter = MPASVerticalCrossSectionPlotter(figsize=(6, 4), dpi=50)
        _, ax = plotter.create_vertical_cross_section(
            proc, "theta", (-95.0, 32.0), (-95.0, 48.0), num_points=30
        )
        assert ax.get_xlabel() == "Distance along path [km]"
        assert ax.get_xlim()[1] == pytest.approx(16 * 111.19, rel=0.02)

    def test_dateline_section_distance_is_monotonic(
        self: "TestCrossSectionAxes",
    ) -> None:
        """
        This test verifies that a section crossing the dateline spans only its true length (about 2200 km for 20 degrees at the equator) instead of smearing across 340 degrees of longitude.

        Parameters:
            None

        Returns:
            None
        """
        lons = np.concatenate([np.linspace(160, 180, 30), np.linspace(-180, -160, 30)])
        lon, lat = np.meshgrid(lons, np.linspace(-10, 10, 10))
        proc = _make_processor(lon.ravel(), lat.ravel())
        plotter = MPASVerticalCrossSectionPlotter(figsize=(6, 4), dpi=50)
        _, ax = plotter.create_vertical_cross_section(
            proc, "theta", (170.0, 0.0), (-170.0, 0.0), num_points=30
        )
        assert ax.get_xlim()[1] == pytest.approx(20 * 111.19, rel=0.02)

    def test_per_column_pressure_differs_between_columns(
        self: "TestCrossSectionAxes",
    ) -> None:
        """
        This test verifies the pressure coordinate is sampled per path column, so a column over high terrain has lower pressure at the same model level than a column at sea level.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = np.meshgrid(np.linspace(-100, -90, 20), np.linspace(30, 40, 20))
        proc = _make_processor(lon.ravel(), lat.ravel())
        plotter = MPASVerticalCrossSectionPlotter(figsize=(6, 4), dpi=50)
        cells = np.array([0, 19])
        pressure = plotter._sample_column_pressure(proc, 0, cells, 6)
        assert pressure is not None
        assert pressure.shape == (6, 2)
        assert pressure[0, 0] != pytest.approx(pressure[0, 1])

    def test_per_column_height_uses_each_cells_terrain(
        self: "TestCrossSectionAxes",
    ) -> None:
        """
        This test verifies per-column heights come from zgrid at the sampled cells (not cell 0), with layer interfaces averaged to mid-levels.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = np.meshgrid(np.linspace(-100, -90, 20), np.linspace(30, 40, 20))
        proc = _make_processor(lon.ravel(), lat.ravel())
        plotter = MPASVerticalCrossSectionPlotter(figsize=(6, 4), dpi=50)
        heights = plotter._sample_column_height(proc, 0, np.array([0, 19]), 6)
        assert heights is not None
        assert heights.shape == (6, 2)
        assert heights[0, 1] > heights[0, 0]

    def test_columns_to_levels_variants(self: "TestCrossSectionAxes") -> None:
        """
        This test verifies the equal-length, interface-to-mid and mid-to-interface reshaping branches of `_columns_to_levels` and that incompatible sizes return None.

        Parameters:
            None

        Returns:
            None
        """
        cols = np.arange(12, dtype=float).reshape(2, 6)
        assert MPASVerticalCrossSectionPlotter._columns_to_levels(cols, 6).shape == (
            6,
            2,
        )
        assert MPASVerticalCrossSectionPlotter._columns_to_levels(cols, 5).shape == (
            5,
            2,
        )
        assert MPASVerticalCrossSectionPlotter._columns_to_levels(cols, 7).shape == (
            7,
            2,
        )
        assert MPASVerticalCrossSectionPlotter._columns_to_levels(cols, 20) is None

    def test_theta_stays_in_kelvin(self: "TestCrossSectionAxes") -> None:
        """
        This test verifies potential temperature is not converted to degrees Celsius for the cross-section and keeps kelvin units.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASVerticalCrossSectionPlotter()
        data, meta = plotter._convert_and_clip_data(np.array([300.0, 310.0]), "theta")
        np.testing.assert_allclose(data, [300.0, 310.0])
        assert meta.get("units") == "K"

    def test_wind_levels_use_explicit_names(self: "TestCrossSectionAxes") -> None:
        """
        This test verifies that a variable merely starting with 'v' (for example 'vapor') is not treated as a wind variable, while 'u' is.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASVerticalCrossSectionPlotter()
        wind = plotter._compute_var_levels("u", "u", -5.0, 5.0, 10.0)
        other = plotter._compute_var_levels("vapor", "vapor", -5.0, 5.0, 10.0)
        assert len(wind) == 21
        assert len(other) == 15

    def test_batch_name_and_display_vertical(
        self: "TestCrossSectionAxes", tmp_path: Any
    ) -> None:
        """
        This test verifies the batch step passes display_vertical through, keeps the minutes of a half-hourly valid time in the filename and writes fractional max heights as 'maxh_0p5km'.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = np.meshgrid(np.linspace(-100, -90, 10), np.linspace(30, 40, 10))
        proc = _make_processor(lon.ravel(), lat.ravel())
        plotter = MPASVerticalCrossSectionPlotter(figsize=(4, 3), dpi=40)
        with (
            patch.object(plotter, "create_vertical_cross_section") as mock_create,
            patch.object(plotter, "save_plot"),
            patch.object(plotter, "close_plot"),
        ):
            from mpasdiag.visualization.cross_section import CrossSectionStyle

            files = plotter._process_batch_time_step(
                proc,
                str(tmp_path),
                "theta",
                (-99, 31),
                (-91, 39),
                1,
                2,
                "pressure",
                20,
                0.5,
                ["png"],
                CrossSectionStyle(),
                display_vertical="height",
            )
        assert mock_create.call_args.kwargs["display_vertical"] == "height"
        assert "valid_20250101T0030" in files[0]
        assert "maxh_0p5km" in files[0]


class TestDefaultLowestLevel:
    """Default level for 3D fields is the lowest (index 0), not the model top."""

    def test_surface_extract_default_level_zero(self: "TestDefaultLowestLevel") -> None:
        """
        This test verifies that `_extract_2d_from_multidimensional` returns level 0 when no level is requested.

        Parameters:
            None

        Returns:
            None
        """
        data = np.tile(np.arange(5.0), (8, 1))
        out = MPASSurfacePlotter()._extract_2d_from_multidimensional(data, None, None)
        assert np.all(out == 0.0)

    def test_surface_overlay_default_level_zero(self: "TestDefaultLowestLevel") -> None:
        """
        This test verifies that `_prepare_overlay_data` uses level 0 for a 3D overlay without a level index.

        Parameters:
            None

        Returns:
            None
        """
        data = np.tile(np.arange(5.0), (8, 1)) + 1.0
        lon = np.linspace(0, 10, 8)
        lat = np.linspace(0, 10, 8)
        _, _, vals = MPASSurfacePlotter()._prepare_overlay_data(
            data, lon, lat, "foo", {}
        )
        assert np.all(vals == 1.0)

    def test_wind_defaults_level_zero(self: "TestDefaultLowestLevel") -> None:
        """
        This test verifies both wind slice helpers default to level 0.

        Parameters:
            None

        Returns:
            None
        """
        u = np.tile(np.arange(5.0), (8, 1))
        plotter = MPASWindPlotter()
        a, _ = plotter._extract_2d_wind_slice(u, u, None)
        b, _ = plotter.extract_2d_from_3d_wind(u, u)
        assert np.all(a == 0.0) and np.all(b == 0.0)


class TestWindFixes:
    """Barbs in knots, overlay bounds, extent-aware subsampling, titles and batch time."""

    def test_render_barbs_converted_to_knots(self: "TestWindFixes") -> None:
        """
        This test verifies `_render_wind_vectors` hands barbs knots (m/s times 1.943844).

        Parameters:
            None

        Returns:
            None
        """
        ax = MagicMock()
        MPASWindPlotter()._render_wind_vectors(
            ax,
            np.array([0.0]),
            np.array([0.0]),
            np.array([10.0]),
            np.array([-5.0]),
            "barbs",
        )
        args = ax.barbs.call_args.args
        assert args[2][0] == pytest.approx(19.43844)
        assert args[3][0] == pytest.approx(-9.71922)

    def test_base_create_wind_plot_barbs_in_knots(self: "TestWindFixes") -> None:
        """
        This test verifies the base visualizer's wind plot also converts barbs to knots and labels them in the title.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = np.meshgrid(np.linspace(-100, -90, 5), np.linspace(30, 40, 5))
        u = np.full(lon.size, 10.0)
        v = np.zeros(lon.size)
        viz = MPASVisualizer(figsize=(5, 4), dpi=50)
        with patch("matplotlib.axes.Axes.barbs") as barbs:
            _, ax = viz.create_wind_plot(
                lon.ravel(),
                lat.ravel(),
                u,
                v,
                GeographicBounds(-101, -89, 29, 41),
                style=WindPlotStyle(plot_type="barbs", subsample=1),
            )
        assert barbs.call_args.args[2][0] == pytest.approx(19.43844)
        assert "kt" in ax.get_title()

    def test_surface_overlay_passes_map_bounds(self: "TestWindFixes") -> None:
        """
        This test verifies `_add_overlays` passes the map bounds to the wind overlay, which subsample=-1 and grid_resolution require.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        plotter.fig = plt.figure()
        plotter.ax = plotter.fig.add_subplot(111, projection=ccrs.PlateCarree())
        with patch.object(MPASWindPlotter, "add_wind_overlay") as overlay:
            plotter._add_overlays(
                np.array([0.0]),
                np.array([0.0]),
                {"u_data": 1, "v_data": 1},
                None,
                -130.0,
                -50.0,
                20.0,
                60.0,
                None,
            )
        kw = overlay.call_args.kwargs
        assert (kw["lon_min"], kw["lon_max"], kw["lat_min"], kw["lat_max"]) == (
            -130.0,
            -50.0,
            20.0,
            60.0,
        )

    def test_valid_point_count_uses_extent(self: "TestWindFixes") -> None:
        """
        This test verifies only points inside the extent are counted, including for a dateline-crossing extent.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASWindPlotter()
        lon = np.array([-120.0, -100.0, 10.0, 170.0, -175.0])
        lat = np.zeros(5)
        u = np.ones(5)
        assert plotter._calculate_valid_point_count(lon, u) == 5
        assert (
            plotter._calculate_valid_point_count(
                lon, u, lat, (-130.0, -90.0, -5.0, 5.0)
            )
            == 2
        )
        assert (
            plotter._calculate_valid_point_count(lon, u, lat, (160.0, 200.0, -5.0, 5.0))
            == 2
        )

    def test_optimal_subsample_depends_on_extent(self: "TestWindFixes") -> None:
        """
        This test verifies that for a whole-mesh point count a small extent is thinned less than the whole globe.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASWindPlotter(figsize=(6, 4))
        small = plotter.calculate_optimal_subsample(
            2_000_000, -100, -90, 30, 40, points_in_extent=False
        )
        globe = plotter.calculate_optimal_subsample(
            2_000_000, -180, 180, -90, 90, points_in_extent=False
        )
        assert small < globe

    def test_title_all_nan_does_not_raise(self: "TestWindFixes") -> None:
        """
        This test verifies an all-NaN wind field yields a title instead of raising, and that barbs titles mention knots.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASWindPlotter()
        nan = np.full((3, 3), np.nan)
        assert "no valid data" in plotter._generate_wind_title(
            nan, nan, None, None, None
        )
        ok = plotter._generate_wind_title(
            np.ones(3), np.ones(3), None, None, None, "barbs"
        )
        assert "barbs: kt" in ok

    def test_batch_passes_valid_time(self: "TestWindFixes", tmp_path: Any) -> None:
        """
        This test verifies the wind batch passes the real valid time to the plot and closes the figure when a step fails.

        Parameters:
            None

        Returns:
            None
        """
        times = pd.date_range("2025-01-01", periods=2, freq="1h")
        ds = xr.Dataset(
            {"u": (["Time", "nCells"], np.ones((2, 3)))}, coords={"Time": times}
        )
        proc = Mock()
        proc.dataset = ds
        da = xr.DataArray(np.ones(3))
        proc.get_2d_variable_data = Mock(return_value=da)
        proc.extract_2d_coordinates_for_variable = Mock(
            return_value=(np.zeros(3), np.zeros(3))
        )
        plotter = MPASWindPlotter()
        seen: List[Any] = []

        def fake_create(*args: Any, **kwargs: Any) -> Any:
            seen.append(kwargs["style"].time_stamp)
            if len(seen) == 1:
                raise RuntimeError("boom")
            return None, None

        with (
            patch.object(plotter, "create_wind_plot", side_effect=fake_create),
            patch.object(plotter, "add_timestamp_and_branding"),
            patch.object(plotter, "save_plot"),
            patch.object(plotter, "close_plot") as close,
        ):
            out = plotter.create_batch_wind_plots(
                proc, str(tmp_path), GeographicBounds(-1, 1, -1, 1), "u", "u"
            )
        assert seen[1] == datetime(2025, 1, 1, 1, 0)
        assert len(out) == 1
        assert close.call_count >= 2


class TestStylingFixes:
    """Variable styling rules, tick labels and footer timezone."""

    @pytest.mark.parametrize("name", ["precipw", "snowc", "snowh", "terrain_x"])
    def test_non_precip_names_keep_own_style(
        self: "TestStylingFixes", name: str
    ) -> None:
        """
        This test verifies names that only embed precipitation letters are not given the precipitation colormap.

        Parameters:
            name (str): Variable name under test.

        Returns:
            None
        """
        style = MPASVisualizationStyle.get_variable_style(name)
        assert not isinstance(
            style["colormap"], __import__("matplotlib.colors").colors.ListedColormap
        )

    def test_configured_styles(self: "TestStylingFixes") -> None:
        """
        This test verifies precipw keeps Blues, snow cover is a 0-1 fraction and unknown variables get no meaningless default levels.

        Parameters:
            None

        Returns:
            None
        """
        assert (
            MPASVisualizationStyle.get_variable_style("precipw")["colormap"] == "Blues"
        )
        assert MPASVisualizationStyle.get_variable_style("snowc")["levels"][-1] == 1.0
        assert MPASVisualizationStyle.get_variable_style("mystery")["levels"] is None

    @pytest.mark.parametrize(
        "name", ["rainnc", "rainc", "precipitation", "precip_6h", "qpf", "prate"]
    )
    def test_precip_names_still_match(self: "TestStylingFixes", name: str) -> None:
        """
        This test verifies genuine precipitation variable names still select the discrete precipitation colormap.

        Parameters:
            name (str): Variable name under test.

        Returns:
            None
        """
        style = MPASVisualizationStyle.get_variable_style(name)
        assert style["levels"] is not None and len(style["levels"]) > 5

    def test_ticks_are_exact(self: "TestStylingFixes") -> None:
        """
        This test verifies every tick label reproduces its tick value (0.0025 is no longer shown as 0.003).

        Parameters:
            None

        Returns:
            None
        """
        ticks = [0.001, 0.002, 0.0025]
        labels = MPASVisualizationStyle.format_ticks_dynamic(ticks)
        assert labels == ["0.0010", "0.0020", "0.0025"]
        for tick, label in zip(
            [0.5, 1.25, 2.0],
            MPASVisualizationStyle.format_ticks_dynamic([0.5, 1.25, 2.0]),
        ):
            assert float(label) == pytest.approx(tick)

    def test_footer_timestamp_is_utc(self: "TestStylingFixes") -> None:
        """
        This test verifies the footer asks for the current time in UTC rather than labelling local time as UTC.

        Parameters:
            None

        Returns:
            None
        """
        fig = plt.figure()
        with patch("mpasdiag.visualization.styling.datetime") as dt:
            dt.now.return_value = datetime(2025, 1, 1, tzinfo=timezone.utc)
            MPASVisualizationStyle.add_timestamp_and_branding(fig)
        dt.now.assert_called_once_with(timezone.utc)


class TestTransectLine:
    """Transect lines follow the great circle on map axes."""

    def test_geodetic_transform(self: "TestTransectLine") -> None:
        """
        This test verifies the transect line on a GeoAxes is drawn with the Geodetic transform.

        Parameters:
            None

        Returns:
            None
        """
        viz = MPASVisualizer()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        with patch.object(ax, "plot", wraps=ax.plot) as plot:
            viz.draw_transect_line(ax, (-100.0, 30.0), (-60.0, 60.0))
        assert isinstance(plot.call_args_list[0].kwargs["transform"], ccrs.Geodetic)


class TestPrecipitationFixes:
    """One-sided limits, accumulation periods, lookback skipping, time tags, overlay extent."""

    def test_one_sided_clim_keeps_color_mapping(self: "TestPrecipitationFixes") -> None:
        """
        This test verifies a clim_min alone is honoured and colors keep the mapping of the default scheme.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        base_cmap, base_levels = plotter.create_precip_colormap("a24h")
        cmap, norm, levels = plotter._prepare_precipitation_colormap(
            None, None, "a24h", 10, None
        )
        assert min(levels) == 10 and max(levels) == max(base_levels)
        base_norm = plt.matplotlib.colors.BoundaryNorm(
            [0] + sorted(base_levels) + [max(base_levels) + 1], base_cmap.N, clip=True
        )
        assert cmap(norm(60.0)) == base_cmap(base_norm(60.0))
        assert cmap(norm(130.0)) == base_cmap(base_norm(130.0))

    def test_accum_period_parsing(self: "TestPrecipitationFixes") -> None:
        """
        This test verifies generic 'aNNh' periods parse and unknown periods raise ValueError.

        Parameters:
            None

        Returns:
            None
        """
        assert MPASPrecipitationPlotter._accum_period_hours("a48h") == 48
        with pytest.raises(ValueError, match="a24h"):
            MPASPrecipitationPlotter._accum_period_hours("weekly")

    def test_batch_skips_insufficient_lookback(
        self: "TestPrecipitationFixes", tmp_path: Any
    ) -> None:
        """
        This test verifies steps raising InsufficientLookbackError are skipped (figure closed) while the others still produce files.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        proc = Mock()
        proc.dataset = xr.Dataset()

        def fake_step(_p: Any, idx: int, *args: Any, **kwargs: Any) -> List[str]:
            if idx == 1:
                raise InsufficientLookbackError("not enough history")
            return [f"f{idx}.png"]

        with (
            patch.object(
                plotter,
                "_extract_coordinates_from_processor",
                return_value=(np.zeros(1), np.zeros(1)),
            ),
            patch.object(
                plotter, "_setup_batch_time_indices", return_value=([1, 2], 1)
            ),
            patch.object(plotter, "_process_single_time_step", side_effect=fake_step),
            patch.object(plotter, "close_plot") as close,
        ):
            out = plotter.create_batch_precipitation_maps(
                proc, str(tmp_path), GeographicBounds(-1, 1, -1, 1)
            )
        assert out == ["f2.png"]
        assert close.call_count == 1

    def test_filename_keeps_minutes(
        self: "TestPrecipitationFixes", tmp_path: Any
    ) -> None:
        """
        This test verifies a half-hourly valid time is not truncated to the hour in the output filename.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        times = pd.to_datetime(["2025-01-01T00:00", "2025-01-01T00:30"])
        proc = Mock()
        proc.dataset = xr.Dataset(coords={"Time": times})
        proc.data_type = "2d"
        from mpasdiag.visualization.precipitation import PrecipitationMapStyle

        with (
            patch(
                "mpasdiag.visualization.precipitation.PrecipitationDiagnostics"
            ) as diag,
            patch.object(
                plotter, "create_precipitation_map", return_value=(None, None)
            ),
            patch.object(plotter, "save_plot") as save,
            patch.object(plotter, "close_plot"),
        ):
            diag.return_value.compute_precipitation_difference.return_value = (
                xr.DataArray(np.zeros(2))
            )
            plotter._process_single_time_step(
                proc,
                1,
                np.zeros(2),
                np.zeros(2),
                GeographicBounds(-1, 1, -1, 1),
                "rainnc",
                "a01h",
                "scatter",
                None,
                PrecipitationMapStyle(),
                str(tmp_path),
                ["png"],
            )
        assert "T0030" in save.call_args.args[0]

    def test_overlay_uses_map_extent(self: "TestPrecipitationFixes") -> None:
        """
        This test verifies `_prepare_overlay_data` clips scatter data to the supplied map extent and returns it as the bounds.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        lon = np.array([-100.0, -95.0, -90.0])
        lat = np.array([30.0, 35.0, 40.0])
        pr = np.array([1.0, 2.0, 3.0])
        lv, _, _, lo0, lo1, la0, la1 = plotter._prepare_overlay_data(
            lon, lat, pr, "rainnc", None, "scatter", (-96.0, -80.0, 20.0, 50.0)
        )
        assert (lo0, lo1, la0, la1) == (-96.0, -80.0, 20.0, 50.0)
        assert list(lv) == [-95.0, -90.0]


class TestSurfaceFixes:
    """Time tags and unit labelling in the surface plotter."""

    def test_title_keeps_minutes(self: "TestSurfaceFixes") -> None:
        """
        This test verifies a sub-hourly valid time is kept in the default surface title.

        Parameters:
            None

        Returns:
            None
        """
        title, in_title = MPASSurfacePlotter()._generate_title(
            None, datetime(2025, 1, 1, 0, 30), {"long_name": "T2"}
        )
        assert "T0030" in title and in_title

    def test_units_not_relabelled_when_conversion_fails(
        self: "TestSurfaceFixes",
    ) -> None:
        """
        This test verifies the metadata unit stays the original unit when the conversion fails.

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(np.array([290.0, 300.0]), attrs={"units": "K"})
        with patch(
            "mpasdiag.visualization.surface.UnitConverter.convert_units",
            side_effect=ValueError("nope"),
        ):
            data, meta = MPASSurfacePlotter()._extract_and_convert_units(
                da.values, "t2m", da
            )
        np.testing.assert_allclose(data, [290.0, 300.0])
        assert meta["units"] == "K"


class TestSkewTFixes:
    """Skew-T profile ordering and indices-table layout."""

    def test_profile_sorted_by_descending_pressure(self: "TestSkewTFixes") -> None:
        """
        This test verifies an ascending-pressure profile is reordered (with temperature and wind) so index 0 is the surface.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSkewTPlotter(verbose=False)
        p = np.array([200.0, 500.0, 1000.0])
        t = np.array([-50.0, -20.0, 20.0])
        captured: dict = {}

        def fake(p_arr: Any, t_arr: Any, *a: Any) -> Any:
            captured["p"], captured["t"], captured["u"] = p_arr, t_arr, a[1]
            fig = plt.figure()
            return fig, fig.add_subplot(111)

        with patch.object(plotter, "_create_metpy_skewt", side_effect=fake):
            plotter.create_skewt_diagram(
                p, t, t - 5, np.array([1.0, 2.0, 3.0]), np.zeros(3)
            )
        assert list(captured["p"]) == [1000.0, 500.0, 200.0]
        assert list(captured["t"]) == [20.0, -20.0, -50.0]
        assert list(captured["u"]) == [3.0, 2.0, 1.0]

    def test_indices_table_inside_figure(self: "TestSkewTFixes") -> None:
        """
        This test verifies the full indices table lies inside the figure and below the main axes.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSkewTPlotter(verbose=False, figsize=(9, 12), dpi=50)
        p = np.linspace(1000, 100, 20)
        t = np.linspace(25, -60, 20)
        indices = {
            k: 100.0
            for k in [
                "sbcape",
                "sbcin",
                "mlcape",
                "mlcin",
                "mucape",
                "mucin",
                "lifted_index",
                "lcl_pressure",
                "lfc_pressure",
                "el_pressure",
                "precipitable_water",
                "k_index",
                "total_totals",
                "bulk_shear_0_6km",
                "srh_0_1km",
                "stp",
                "scp",
            ]
        }
        fig, ax = plotter.create_skewt_diagram(p, t, t - 5, indices=indices)
        for axes in fig.axes:
            box = axes.get_position()
            assert box.y0 >= -1e-6 and box.y1 <= 1.0 + 1e-6
        table_ax = [a for a in fig.axes if a is not ax and not a.axison]
        assert table_ax
        assert (
            max(a.get_position().y1 for a in table_ax if a.get_position().y1 < 0.5)
            <= ax.get_position().y0 + 1e-6
        )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""
MPASdiag Test Suite: Regression tests for the second round of visualization-layer bug fixes.

This module covers precipitation color limits of zero or below, figure leaks on failing public plot calls, token-based variable styling, precipitation display units and threshold highlighting, the surface num_levels option, Skew-T location text, labels, save-path handling and title sanitising, wind level labels, extent-limited statistics and the single branding footer, accumulation-period and negative-value validation, the empty-Time wind batch, cross-section column filtering and file naming, and the common verbose constructor keyword.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: 2026
Version: 1.0.0
"""

import numpy as np
import pytest
import xarray as xr
import matplotlib.pyplot as plt
from typing import Any, List
from unittest.mock import MagicMock, patch

from mpasdiag.processing.utils_geog import GeographicBounds
from mpasdiag.visualization.base_visualizer import WindPlotStyle
from mpasdiag.visualization.cross_section import (
    CrossSectionStyle,
    MPASVerticalCrossSectionPlotter,
)
from mpasdiag.visualization.precipitation import (
    MPASPrecipitationPlotter,
    PrecipitationRenderStyle,
)
from mpasdiag.visualization.skewt import MPASSkewTPlotter
from mpasdiag.visualization.styling import MPASVisualizationStyle
from mpasdiag.visualization.surface import MPASSurfacePlotter, SurfaceMapStyle
from mpasdiag.visualization.wind import MPASWindPlotter

_RNG = np.random.default_rng(11)
_BOUNDS = GeographicBounds(-110.0, -90.0, 30.0, 45.0)


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


def _points(n: int = 400) -> Any:
    """
    This helper builds random lon/lat points inside the test extent together with gamma-distributed precipitation values.

    Parameters:
        n (int): Number of points.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Longitudes, latitudes and precipitation values.
    """
    lon = _RNG.uniform(-109, -91, n)
    lat = _RNG.uniform(31, 44, n)
    return lon, lat, _RNG.gamma(1.0, 8.0, n)


class TestPrecipitationColorLimits:
    """Color limits of zero or below no longer break the BoundaryNorm."""

    @pytest.mark.parametrize("clim_min", [0, -5.0, 0.5])
    def test_clim_min_bounds_monotonic(
        self: "TestPrecipitationColorLimits", clim_min: float
    ) -> None:
        """
        This test verifies that a minimum color limit of 0, a negative value or a positive value yields strictly increasing normalization bounds, one colour per interval and the limit itself as the first level.

        Parameters:
            clim_min (float): Minimum color limit under test.

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        cmap, norm, levels = plotter._prepare_precipitation_colormap(
            None, None, "a24h", clim_min, None
        )
        bounds = list(norm.boundaries)
        assert levels[0] == clim_min
        assert all(b > a for a, b in zip(bounds[:-1], bounds[1:]))
        # one colour per interval plus the open-ended over region (no phantom upper bound)
        assert cmap.N == len(bounds)

    def test_clim_zero_map_renders(self: "TestPrecipitationColorLimits") -> None:
        """
        This test verifies that a full map with clim_min=0 renders instead of failing with a colour-bin mismatch.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat, data = _points()
        plotter = MPASPrecipitationPlotter()
        fig, _ = plotter.create_precipitation_map(
            lon,
            lat,
            data,
            _BOUNDS,
            style=PrecipitationRenderStyle(clim_min=0.0, clim_max=60.0),
        )
        assert fig is not None

    def test_reversed_limits_raise_and_close(
        self: "TestPrecipitationColorLimits",
    ) -> None:
        """
        This test verifies that a reversed color range raises a ValueError without leaving a figure open.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat, data = _points()
        before = set(plt.get_fignums())
        with pytest.raises(ValueError, match="clim_min"):
            MPASPrecipitationPlotter().create_precipitation_map(
                lon,
                lat,
                data,
                _BOUNDS,
                style=PrecipitationRenderStyle(clim_min=5.0, clim_max=2.0),
            )
        assert set(plt.get_fignums()) == before

    def test_default_scheme_keeps_zero_floor(
        self: "TestPrecipitationColorLimits",
    ) -> None:
        """
        This test verifies that without limits the default scheme still prepends the zero floor so dry cells keep the first colour (11 colours, 11 bins).

        Parameters:
            None

        Returns:
            None
        """
        _, norm, levels = MPASPrecipitationPlotter()._prepare_precipitation_colormap(
            None, None, "a24h", None, None
        )
        assert norm.boundaries[0] == 0.0
        assert len(norm.boundaries) == len(levels) + 1


class TestPrecipitationUnitsAndThreshold:
    """Display units, threshold highlighting, negative values and accumulation validation."""

    @pytest.mark.parametrize("units, factor", [("mm", 1.0), ("cm", 10.0), ("in", 25.4)])
    def test_display_units_conversion(
        self: "TestPrecipitationUnitsAndThreshold", units: str, factor: float
    ) -> None:
        """
        This test verifies that millimetre data are converted into the requested display units and the unit label follows.

        Parameters:
            units (str): Display units.
            factor (float): Millimetres per display unit.

        Returns:
            None
        """
        data = np.array([10.0, 25.4])
        out, label = MPASPrecipitationPlotter()._convert_precipitation_units(
            data, None, "rainnc", units
        )
        assert label == units
        np.testing.assert_allclose(out, data / factor)

    def test_invalid_units_rejected(self: "TestPrecipitationUnitsAndThreshold") -> None:
        """
        This test verifies that an unknown display unit raises a ValueError and leaves no figure behind.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat, data = _points()
        with pytest.raises(ValueError, match="display_units"):
            MPASPrecipitationPlotter().create_precipitation_map(
                lon,
                lat,
                data,
                _BOUNDS,
                style=PrecipitationRenderStyle(display_units="furlongs"),
            )
        assert plt.get_fignums() == []

    def test_inches_label_levels_and_threshold(
        self: "TestPrecipitationUnitsAndThreshold",
    ) -> None:
        """
        This test verifies that an inch map uses inch levels and colorbar label, outlines the cells at or above the threshold and reports their count in a legend entry.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat, data = _points()
        plotter = MPASPrecipitationPlotter()
        fig, ax = plotter.create_precipitation_map(
            lon,
            lat,
            data,
            _BOUNDS,
            style=PrecipitationRenderStyle(display_units="in", threshold=1.0),
        )
        expected = int(np.sum(data / 25.4 >= 1.0))
        legend = ax.get_legend()
        assert legend is not None
        text = legend.get_texts()[0].get_text()
        assert f"{expected:,} cells" in text and "in" in text
        labels = [a.get_xlabel() for a in fig.axes]
        assert any("Precipitation [in]" in lab for lab in labels)

    def test_default_levels_per_unit(
        self: "TestPrecipitationUnitsAndThreshold",
    ) -> None:
        """
        This test verifies that inch and centimetre maps use round default levels instead of converted millimetre levels.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASPrecipitationPlotter()
        _, _, in_levels = plotter._prepare_precipitation_colormap(
            None, None, "a24h", None, None, "in"
        )
        _, _, cm_levels = plotter._prepare_precipitation_colormap(
            None, None, "a01h", None, None, "cm"
        )
        assert in_levels[0] == 0.01 and len(in_levels) == 10
        assert cm_levels[0] == 0.01 and len(cm_levels) == 10

    def test_negative_values_masked(self: "TestPrecipitationUnitsAndThreshold") -> None:
        """
        This test verifies that negative precipitation becomes NaN (not zero) with or without metadata.

        Parameters:
            None

        Returns:
            None
        """
        out, _ = MPASPrecipitationPlotter()._convert_precipitation_units(
            np.array([-2.0, 1.0]), None, "rainnc"
        )
        assert np.isnan(out[0]) and out[1] == 1.0

    @pytest.mark.parametrize("bad", ["a99x", "banana", "a241h", "a000h", ""])
    def test_bad_accumulation_period(
        self: "TestPrecipitationUnitsAndThreshold", bad: str
    ) -> None:
        """
        This test verifies that malformed or out-of-range accumulation periods raise a clear ValueError before a figure is created.

        Parameters:
            bad (str): Invalid accumulation identifier.

        Returns:
            None
        """
        lon, lat, data = _points(50)
        with pytest.raises(ValueError, match="accumulation period"):
            MPASPrecipitationPlotter().create_precipitation_map(
                lon, lat, data, _BOUNDS, accum_period=bad
            )
        assert plt.get_fignums() == []


class TestFigureLeaks:
    """Public plot calls close their figure when they raise."""

    def test_surface_degenerate_inputs(self: "TestFigureLeaks") -> None:
        """
        This test verifies that all-NaN data, empty data, a reversed color range, a zero-size extent and a single contour level each raise a ValueError and leave no figure open.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat, data = _points(100)
        plotter = MPASSurfacePlotter()
        cases = [
            (data * np.nan, _BOUNDS, SurfaceMapStyle()),
            (data[:0], _BOUNDS, SurfaceMapStyle()),
            (data, _BOUNDS, SurfaceMapStyle(clim_min=5.0, clim_max=1.0)),
            (data, GeographicBounds(-100, -100, 30, 40), SurfaceMapStyle()),
            (data, _BOUNDS, SurfaceMapStyle(plot_type="contourf", levels=[3.0, 3.0])),
            (data, _BOUNDS, SurfaceMapStyle(num_levels=1)),
        ]
        for values, bounds, style in cases:
            lo, la = (lon[: len(values)], lat[: len(values)])
            with pytest.raises(ValueError):
                plotter.create_surface_map(lo, la, values, "foo", bounds, style=style)
            assert plt.get_fignums() == []
        assert plotter.fig is None

    def test_surface_unsorted_duplicate_levels_work(self: "TestFigureLeaks") -> None:
        """
        This test verifies that unsorted or duplicated contour levels are normalised instead of failing inside matplotlib.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat, data = _points(300)
        fig, _ = MPASSurfacePlotter().create_surface_map(
            lon,
            lat,
            data,
            "foo",
            _BOUNDS,
            style=SurfaceMapStyle(plot_type="scatter", levels=[20, 5, 5, 10]),
        )
        assert fig is not None

    def test_wind_mismatched_lengths(self: "TestFigureLeaks") -> None:
        """
        This test verifies that mismatched u/v sizes raise a clear ValueError and leave no figure.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat, _ = _points(50)
        with pytest.raises(ValueError, match="same"):
            MPASWindPlotter().create_wind_plot(
                lon, lat, np.ones(50), np.ones(40), _BOUNDS
            )
        assert plt.get_fignums() == []

    def test_wind_empty_streamlines(self: "TestFigureLeaks") -> None:
        """
        This test verifies that empty streamline input raises a clear ValueError instead of a numpy reduction error.

        Parameters:
            None

        Returns:
            None
        """
        empty = np.array([])
        with pytest.raises(ValueError, match="non-empty"):
            MPASWindPlotter().create_wind_plot(
                empty,
                empty,
                empty,
                empty,
                _BOUNDS,
                style=WindPlotStyle(plot_type="streamlines"),
            )
        assert plt.get_fignums() == []

    def test_wind_failure_after_figure_created(self: "TestFigureLeaks") -> None:
        """
        This test verifies that an exception raised after the figure exists (here from the vector renderer) closes it and resets the plotter.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat, _ = _points(50)
        plotter = MPASWindPlotter()
        with patch.object(
            plotter, "_render_wind_vectors", side_effect=RuntimeError("x")
        ):
            with pytest.raises(RuntimeError):
                plotter.create_wind_plot(lon, lat, np.ones(50), np.ones(50), _BOUNDS)
        assert plt.get_fignums() == [] and plotter.fig is None

    def test_skewt_empty_arrays(self: "TestFigureLeaks") -> None:
        """
        This test verifies that empty Skew-T profiles raise a ValueError and leave no figure.

        Parameters:
            None

        Returns:
            None
        """
        empty = np.array([])
        with pytest.raises(ValueError, match="empty"):
            MPASSkewTPlotter().create_skewt_diagram(empty, empty, empty)
        assert plt.get_fignums() == []

    def test_earlier_figures_untouched(self: "TestFigureLeaks") -> None:
        """
        This test verifies that a figure the caller opened before the failing call stays open.

        Parameters:
            None

        Returns:
            None
        """
        keep = plt.figure()
        lon, lat, data = _points(20)
        with pytest.raises(ValueError):
            MPASSurfacePlotter().create_surface_map(
                lon, lat, data * np.nan, "foo", _BOUNDS
            )
        assert plt.get_fignums() == [keep.number]


class TestStyleClassification:
    """Variable styling uses whole-name tokens and respects masks."""

    @pytest.mark.parametrize("name", ["srh_0_1km", "zonal_u", "frozen_depth", "u10"])
    def test_not_misclassified(self: "TestStyleClassification", name: str) -> None:
        """
        This test verifies that names merely containing 'rh', 'z' or 'q' are not styled as humidity or height.

        Parameters:
            name (str): Variable name.

        Returns:
            None
        """
        cmap, levels = MPASVisualizationStyle.get_variable_specific_settings(
            name, np.linspace(0, 500, 50)
        )
        assert cmap == "viridis"
        assert levels is not None and levels[-1] >= 400

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("relhum_850hPa", "BuGn"),
            ("rh2m", "BuGn"),
            ("q2", "BuGn"),
            ("height_500hPa", "terrain"),
            ("z500", "terrain"),
        ],
    )
    def test_still_classified(
        self: "TestStyleClassification", name: str, expected: str
    ) -> None:
        """
        This test verifies that genuine humidity and height names keep their dedicated styles.

        Parameters:
            name (str): Variable name.
            expected (str): Expected colormap.

        Returns:
            None
        """
        cmap, _ = MPASVisualizationStyle.get_variable_specific_settings(
            name, np.linspace(1, 90, 40)
        )
        assert cmap == expected

    def test_masked_values_ignored(self: "TestStyleClassification") -> None:
        """
        This test verifies that masked fill values do not influence the levels.

        Parameters:
            None

        Returns:
            None
        """
        data = np.ma.masked_array([1e20, 280.0, 285.0, 290.0], mask=[1, 0, 0, 0])
        _, levels = MPASVisualizationStyle.get_variable_specific_settings("t2m", data)
        assert levels is not None
        assert min(levels) >= 280.0 and max(levels) <= 290.0

    @pytest.mark.parametrize("name", ["foo", "t2m", "rainnc", "relhum_500hPa", "wspd"])
    def test_constant_field_levels(self: "TestStyleClassification", name: str) -> None:
        """
        This test verifies that a constant field still yields at least two strictly increasing levels.

        Parameters:
            name (str): Variable name.

        Returns:
            None
        """
        _, levels = MPASVisualizationStyle.get_variable_specific_settings(
            name, np.full(20, 5.0)
        )
        assert levels is not None and len(levels) >= 2
        assert all(b > a for a, b in zip(levels[:-1], levels[1:]))

    @pytest.mark.parametrize(
        "name",
        [
            "rainfall",
            "snow",
            "prcp",
            "apcp",
            "snowfall",
            "total_precip",
            "tp",
            "rainnc",
        ],
    )
    def test_precip_names_match(self: "TestStyleClassification", name: str) -> None:
        """
        This test verifies accumulated-precipitation names are recognised.

        Parameters:
            name (str): Variable name.

        Returns:
            None
        """
        assert MPASVisualizationStyle._is_precipitation_name(name)

    @pytest.mark.parametrize(
        "name", ["precipw", "snowc", "snowh", "terrain_x", "snow_depth", "tpw"]
    )
    def test_precip_names_excluded(self: "TestStyleClassification", name: str) -> None:
        """
        This test verifies that look-alike non-precipitation names stay excluded.

        Parameters:
            name (str): Variable name.

        Returns:
            None
        """
        assert not MPASVisualizationStyle._is_precipitation_name(name)


class TestSurfaceNumLevels:
    """SurfaceMapStyle.num_levels controls the number of automatic levels."""

    def test_auto_levels_count_and_monotonic(self: "TestSurfaceNumLevels") -> None:
        """
        This test verifies that the helper returns exactly the requested count of strictly increasing levels spanning the data, and widens a constant field.

        Parameters:
            None

        Returns:
            None
        """
        levels = MPASSurfacePlotter._auto_levels(
            np.linspace(270, 300, 50), 7, None, None
        )
        assert len(levels) == 7 and levels[0] == 270 and levels[-1] == 300
        const = MPASSurfacePlotter._auto_levels(np.full(5, 3.0), 4, None, None)
        assert len(const) == 4 and all(b > a for a, b in zip(const[:-1], const[1:]))

    def test_num_levels_reaches_renderer(self: "TestSurfaceNumLevels") -> None:
        """
        This test verifies that num_levels generates that many levels when levels is None, while explicit levels still win and the default is unchanged.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat, data = _points(300)
        plotter = MPASSurfacePlotter()
        seen: List[Any] = []
        original = plotter._render_plot

        def spy(*args: Any, **kwargs: Any) -> None:
            seen.append(args[7])
            original(*args, **kwargs)

        with patch.object(plotter, "_render_plot", side_effect=spy):
            plotter.create_surface_map(
                lon, lat, data, "foo", _BOUNDS, style=SurfaceMapStyle(num_levels=6)
            )
            plotter.close_plot()
            plotter.create_surface_map(
                lon,
                lat,
                data,
                "foo",
                _BOUNDS,
                style=SurfaceMapStyle(num_levels=6, levels=[1, 2, 3]),
            )
            plotter.close_plot()
            plotter.create_surface_map(lon, lat, data, "foo", _BOUNDS)
        assert len(seen[0]) == 6
        assert seen[1] == [1.0, 2.0, 3.0]
        assert (
            seen[2] is None
        )  # default behaviour unchanged for a variable without metadata levels

    def test_invalid_num_levels(self: "TestSurfaceNumLevels") -> None:
        """
        This test verifies that non-integer or too small num_levels raise a ValueError.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat, data = _points(50)
        for bad in (0, 1, 2.5):
            with pytest.raises(ValueError, match="num_levels"):
                MPASSurfacePlotter().create_surface_map(
                    lon,
                    lat,
                    data,
                    "foo",
                    _BOUNDS,
                    style=SurfaceMapStyle(num_levels=bad),  # type: ignore[arg-type]
                )


class TestSkewT:
    """Skew-T location text, labels, save path and title handling."""

    @staticmethod
    def _profile() -> Any:
        """
        This helper builds a short synthetic sounding.

        Parameters:
            None

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Pressure (hPa), temperature and dewpoint (degC).
        """
        p = np.array([1000.0, 850.0, 700.0, 500.0, 300.0, 200.0])
        return (
            p,
            np.array([25.0, 15.0, 5.0, -15.0, -45.0, -55.0]),
            p * 0 + np.array([20.0, 10.0, -2.0, -25.0, -55.0, -65.0]),
        )

    def test_location_text_and_labels(self: "TestSkewT") -> None:
        """
        This test verifies that the requested point and the distance appear under the title and that the axis labels use degree and hPa units.

        Parameters:
            None

        Returns:
            None
        """
        p, t, d = self._profile()
        fig, ax = MPASSkewTPlotter(dpi=50).create_skewt_diagram(
            p,
            t,
            d,
            location={
                "requested_lon": -97.5,
                "requested_lat": 36.5,
                "distance_km": 4.2,
            },
        )
        texts = [txt.get_text() for txt in ax.texts]
        assert any("97.50°W" in s and "4.2 km" in s for s in texts)
        assert ax.get_xlabel() == "Temperature [°C]"
        assert ax.get_ylabel() == "Pressure [hPa]"

    def test_no_location_no_text(self: "TestSkewT") -> None:
        """
        This test verifies that nothing is added when the location keys are absent.

        Parameters:
            None

        Returns:
            None
        """
        assert MPASSkewTPlotter._location_text({"pressure": 1}, None) is None

    def test_save_path_extension_stripped(self: "TestSkewT", tmp_path: Any) -> None:
        """
        This test verifies that a save path ending in .png produces x.png and x.pdf instead of x.png.png.

        Parameters:
            None

        Returns:
            None
        """
        p, t, d = self._profile()
        MPASSkewTPlotter(dpi=40).create_skewt_diagram(
            p, t, d, save_path=str(tmp_path / "x.png")
        )
        assert sorted(f.name for f in tmp_path.iterdir()) == ["x.pdf", "x.png"]

    def test_unbalanced_dollar_title(self: "TestSkewT") -> None:
        """
        This test verifies that a title with an unbalanced dollar sign is sanitised so the figure can be drawn.

        Parameters:
            None

        Returns:
            None
        """
        p, t, d = self._profile()
        fig, ax = MPASSkewTPlotter(dpi=40).create_skewt_diagram(
            p, t, d, title="Cost $5 sounding"
        )
        fig.canvas.draw()
        assert "$" not in ax.get_title()


class TestWind:
    """Wind level title, extent statistics and branding footer."""

    @staticmethod
    def _field() -> Any:
        """
        This helper builds a small wind field covering the test extent.

        Parameters:
            None

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: lon, lat, u, v.
        """
        lon = np.linspace(-120, -80, 120)
        lat = np.linspace(25, 50, 120)
        return lon, lat, np.full(120, 3.0), np.full(120, 4.0)

    @staticmethod
    def _footer_count(fig: Any) -> int:
        """
        This helper counts the branding texts on a figure.

        Parameters:
            fig (Figure): Figure to inspect.

        Returns:
            int: Number of 'Generated with MPASdiag' texts.
        """
        n = 0
        for ax in fig.axes:
            n += sum("Generated with MPASdiag" in t.get_text() for t in ax.texts)
        n += sum("Generated with MPASdiag" in t.get_text() for t in fig.texts)
        return n

    def test_footer_exactly_once(self: "TestWind") -> None:
        """
        This test verifies that a direct create_wind_plot call carries the branding footer once, and that a second branding call does not duplicate it.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat, u, v = self._field()
        plotter = MPASWindPlotter()
        fig, _ = plotter.create_wind_plot(lon, lat, u, v, _BOUNDS)
        assert self._footer_count(fig) == 1
        plotter.add_timestamp_and_branding()
        assert self._footer_count(fig) == 1

    def test_level_label_in_title(self: "TestWind") -> None:
        """
        This test verifies that the title carries the wind level when it is not 'surface' and not otherwise.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat, u, v = self._field()
        plotter = MPASWindPlotter()
        _, ax = plotter.create_wind_plot(
            lon,
            lat,
            u,
            v,
            _BOUNDS,
            style=WindPlotStyle(wind_level="850 hPa", subsample=1),
        )
        assert "850 hPa" in ax.get_title()
        plotter.close_plot()
        _, ax = plotter.create_wind_plot(
            lon, lat, u, v, _BOUNDS, style=WindPlotStyle(subsample=1)
        )
        assert "surface" not in ax.get_title().lower()

    def test_stats_only_inside_extent(self: "TestWind") -> None:
        """
        This test verifies that Max/Mean cover only vectors inside the plotted extent: a strong vector outside it must not appear in the maximum.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat, u, v = self._field()
        u = u.copy()
        u[0] = 50.0
        plotter = MPASWindPlotter()
        title = plotter._generate_wind_title(
            u,
            v,
            None,
            None,
            None,
            "barbs",
            lon=lon,
            lat=lat,
            bounds=(-110.0, -90.0, 30.0, 45.0),
        )
        assert "Max: 5.0 m/s" in title
        everything = plotter._generate_wind_title(u, v, None, None, None, "barbs")
        assert "Max: 50" in everything

    def test_batch_does_not_double_brand(self: "TestWind", tmp_path: Any) -> None:
        """
        This test verifies that the batch method no longer adds a second timestamp/branding on top of create_wind_plot.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat, u, v = self._field()
        times = np.array(["2024-09-17T00:00"], dtype="datetime64[ns]")
        ds = xr.Dataset(coords={"Time": times})
        proc = MagicMock()
        proc.dataset = ds
        proc.get_2d_variable_data.side_effect = lambda var, i: xr.DataArray(
            u if var == "u" else v
        )
        proc.extract_2d_coordinates_for_variable.return_value = (lon, lat)
        plotter = MPASWindPlotter()
        with patch.object(
            plotter,
            "add_timestamp_and_branding",
            wraps=plotter.add_timestamp_and_branding,
        ) as spy:
            files = plotter.create_batch_wind_plots(proc, str(tmp_path), _BOUNDS)
        assert len(files) == 1
        assert spy.call_count == 1

    def test_batch_empty_time_dimension(self: "TestWind", tmp_path: Any) -> None:
        """
        This test verifies that a dataset with an empty Time dimension raises a clear ValueError.

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({"u": (("Time", "nCells"), np.zeros((0, 4)))})
        proc = MagicMock()
        proc.dataset = ds
        with pytest.raises(ValueError, match="no time steps"):
            MPASWindPlotter().create_batch_wind_plots(proc, str(tmp_path), _BOUNDS)


class TestCrossSection:
    """Per-column height filtering and the C5 file-name scheme."""

    def test_filter_keeps_bracketing_row_and_masks_per_column(
        self: "TestCrossSection",
    ) -> None:
        """
        This test verifies that rows are kept through the first row above the limit, so no column stops short of the limit, and that data beyond that row is masked per column instead of being dropped for all columns.

        Parameters:
            None

        Returns:
            None
        """
        heights = np.array(
            [
                [0.0, 1.0],
                [4.0, 5.0],
                [8.0, 11.0],
                [11.6, 13.0],
                [13.0, 16.0],
                [15.0, 19.0],
            ]
        )
        data = np.arange(12, dtype=float).reshape(6, 2)
        vert, filtered = MPASVerticalCrossSectionPlotter._filter_columns_below(
            heights, data, 12.0
        )
        assert vert.shape[0] == 5
        assert vert[:, 0].max() >= 12.0 and vert[:, 1].max() >= 12.0
        assert np.isnan(filtered[4, 1]) and not np.isnan(filtered[3, 1])
        assert not np.isnan(filtered[3, 0])

    def test_filename_scheme(self: "TestCrossSection", tmp_path: Any) -> None:
        """
        This test verifies the contract file name: vcrd, optional disp, valid time tag and a _maxh_ suffix including 0 and fractional limits.

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset(
            coords={"Time": np.array(["2025-01-01T00:30"], dtype="datetime64[ns]")}
        )
        proc = MagicMock()
        proc.dataset = ds
        plotter = MPASVerticalCrossSectionPlotter(figsize=(4, 3), dpi=40)

        def run(max_height: Any, display: Any) -> str:
            with (
                patch.object(plotter, "create_vertical_cross_section"),
                patch.object(plotter, "save_plot"),
                patch.object(plotter, "close_plot"),
            ):
                return plotter._process_batch_time_step(
                    proc,
                    str(tmp_path),
                    "theta",
                    (-99, 31),
                    (-91, 39),
                    0,
                    1,
                    "pressure",
                    20,
                    max_height,
                    ["png"],
                    CrossSectionStyle(),
                    display_vertical=display,
                )[0]

        assert run(None, None).endswith(
            "mpas_cross_section_theta_vcrd_pressure_valid_20250101T0030.png"
        )
        assert run(12, "height").endswith(
            "vcrd_pressure_disp_height_valid_20250101T0030_maxh_12km.png"
        )
        assert run(0.5, None).endswith("_maxh_0p5km.png")


class TestConstructors:
    """Every plotter constructor accepts verbose."""

    def test_verbose_keyword(self: "TestConstructors") -> None:
        """
        This test verifies that all plotter constructors accept verbose and forward it to the base class.

        Parameters:
            None

        Returns:
            None
        """
        for cls in (
            MPASSurfacePlotter,
            MPASSkewTPlotter,
            MPASWindPlotter,
            MPASPrecipitationPlotter,
            MPASVerticalCrossSectionPlotter,
        ):
            assert cls(verbose=False).verbose is False  # type: ignore[call-arg]


class TestVerificationRound2FollowUps:
    """Regression tests for the low-severity issues found while verifying the round-2 visualization fixes."""

    @staticmethod
    def _points(n: int = 200) -> Any:
        """
        This helper returns random points inside a CONUS-like box.

        Parameters:
            n (int): Number of points.

        Returns:
            Any: Tuple of longitude and latitude arrays.
        """
        rng = np.random.default_rng(3)
        return rng.uniform(-110.0, -90.0, n), rng.uniform(30.0, 45.0, n)

    def test_precipitation_rejects_broken_inputs_without_leaking_figures(
        self: "TestVerificationRound2FollowUps",
    ) -> None:
        """
        This regression test verifies that create_precipitation_map raises a clear ValueError for mismatched input sizes, empty arrays, and all-missing data (previously it truncated or drew a blank map), that zero precipitation is still accepted, and that failures leave no figure open.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = self._points()
        bounds = GeographicBounds(-115.0, -85.0, 25.0, 50.0)
        plotter = MPASPrecipitationPlotter(figsize=(4, 3), dpi=40)
        before = set(plt.get_fignums())

        for args, match in [
            ((lon, lat[:-2], np.ones(200)), "Mismatched input sizes"),
            ((np.array([]), np.array([]), np.array([])), "empty input"),
            ((lon, lat, np.full(200, np.nan)), "missing"),
        ]:
            with pytest.raises(ValueError, match=match):
                plotter.create_precipitation_map(*args, bounds)
            assert set(plt.get_fignums()) == before

        fig, _ = plotter.create_precipitation_map(lon, lat, np.zeros(200), bounds)
        plt.close(fig)

    def test_threshold_legend_is_rounded(
        self: "TestVerificationRound2FollowUps",
    ) -> None:
        """
        This test verifies that the threshold legend shows the threshold with three significant digits (20 mm in inches reads 0.787, not 0.787402).

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = self._points()
        plotter = MPASPrecipitationPlotter(figsize=(4, 3), dpi=40)
        fig, ax = plotter.create_precipitation_map(
            lon,
            lat,
            np.linspace(0.0, 60.0, 200),
            GeographicBounds(-115.0, -85.0, 25.0, 50.0),
            style=PrecipitationRenderStyle(display_units="in", threshold=20.0 / 25.4),
        )
        texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("0.787 in" in t for t in texts) and not any(
            "0.787402" in t for t in texts
        )
        plt.close(fig)

    def test_streamlines_reject_all_missing_winds(
        self: "TestVerificationRound2FollowUps",
    ) -> None:
        """
        This regression test verifies that a streamline plot of an all-NaN wind field raises a clear ValueError (previously it drew an empty map) and leaves no figure open, while barbs keep their documented warning-and-empty-figure behaviour.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = self._points()
        nan = np.full(200, np.nan)
        bounds = GeographicBounds(-115.0, -85.0, 25.0, 50.0)
        before = set(plt.get_fignums())

        with pytest.raises(ValueError, match="all wind inputs are missing"):
            MPASWindPlotter(figsize=(4, 3), dpi=40).create_wind_plot(
                lon, lat, nan, nan, bounds, style=WindPlotStyle(plot_type="streamlines")
            )
        assert set(plt.get_fignums()) == before

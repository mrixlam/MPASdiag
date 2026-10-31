#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""
MPASdiag Test Suite: Round-3 regression tests for the cross-section and Skew-T plotters.

This module contains regression tests for the round-3 fixes in the vertical cross-section plotter (model-level display values, max_height handling for automatic levels, pcolormesh normalisation, contour colorbars, clim_min/clim_max, edge-variable rejection, time-index validation, great-circle edge cases, and tick formatting) and in the Skew-T plotter (CIN shading limits, data-driven x-limits, level labels, barb thinning, and signed-zero formatting). Synthetic processors are used throughout so no model data files are needed.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: October 2026
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

from mpasdiag import MPAS3DProcessor, MPASVerticalCrossSectionPlotter
from mpasdiag.visualization import cross_section as xsec_module
from mpasdiag.visualization.cross_section import CrossSectionStyle

N_TIME = 3
N_LEV = 12
N_CELLS = 200


def _make_processor(with_edge_var: bool = False) -> Mock:
    """
    This helper builds a synthetic MPAS3DProcessor mock whose dataset has a theta field increasing with height, a matching per-column pressure field, and optionally an edge-located wind variable. The vertical levels it returns are mean pressures (Pa) for the pressure coordinate and indices for model levels.

    Parameters:
        with_edge_var (bool): Whether to add an edge-located variable 'u' on nEdges.

    Returns:
        Mock: The synthetic processor.
    """
    rng = np.random.default_rng(7)
    lev_p = np.linspace(100000.0, 20000.0, N_LEV)
    theta = np.broadcast_to(
        np.linspace(290.0, 330.0, N_LEV)[None, :, None], (N_TIME, N_LEV, N_CELLS)
    ) + rng.normal(0.0, 0.3, (N_TIME, N_LEV, N_CELLS))
    pressure = np.broadcast_to(lev_p[None, :, None], (N_TIME, N_LEV, N_CELLS)).copy()

    variables: Dict[str, Any] = {
        "theta": (["Time", "nVertLevels", "nCells"], theta),
        "pressure": (["Time", "nVertLevels", "nCells"], pressure),
        "lonCell": ("nCells", np.linspace(-100.0, -80.0, N_CELLS)),
        "latCell": ("nCells", np.linspace(30.0, 45.0, N_CELLS)),
    }

    if with_edge_var:
        variables["u"] = (
            ["Time", "nEdges", "nVertLevels"],
            rng.normal(size=(N_TIME, 50, N_LEV)),
        )

    ds = xr.Dataset(
        variables,
        coords={"Time": pd.date_range("2025-01-01", periods=N_TIME, freq="1h")},
    )

    proc = Mock(spec=MPAS3DProcessor)
    proc.dataset = ds
    proc.grid_file = None
    proc.extract_2d_coordinates_for_variable = Mock(
        return_value=(ds["lonCell"].values, ds["latCell"].values)
    )

    def _levels(
        var_name: str, return_pressure: bool = True, time_index: int = 0
    ) -> np.ndarray:
        return lev_p if return_pressure else np.arange(N_LEV)

    proc.get_vertical_levels = Mock(side_effect=_levels)
    return proc


def _plot(
    proc: Mock, plotter: MPASVerticalCrossSectionPlotter = None, **kwargs: Any
) -> Any:
    """
    This helper draws a theta cross-section on the synthetic processor with test-friendly defaults.

    Parameters:
        proc (Mock): Synthetic processor.
        plotter (MPASVerticalCrossSectionPlotter): Optional plotter to reuse.
        **kwargs: Overrides for create_vertical_cross_section.

    Returns:
        Any: The (figure, axes) pair.
    """
    plotter = plotter or MPASVerticalCrossSectionPlotter(figsize=(8, 5), dpi=50)
    args: Dict[str, Any] = dict(
        mpas_3d_processor=proc,
        var_name="theta",
        start_point=(-98.0, 31.0),
        end_point=(-82.0, 43.0),
        num_points=25,
    )
    args.update(kwargs)
    return plotter.create_vertical_cross_section(**args)


@pytest.fixture(autouse=True)
def _close_figures() -> Any:
    """
    This fixture closes every matplotlib figure after each test so figures never leak between tests.

    Parameters:
        None

    Returns:
        Any: Yields control to the test.
    """
    yield
    plt.close("all")


class TestModelLevelDisplay:
    """X1: a model-level display shows level indices, not pressure or height values."""

    def test_resolve_display_returns_indices(self: "TestModelLevelDisplay") -> None:
        """
        This test verifies that _resolve_vertical_display returns 0..n-1 for a model-level display even when the resolved coordinates are pressures in Pa.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASVerticalCrossSectionPlotter()
        coords, kind = plotter._resolve_vertical_display(
            np.linspace(100000.0, 1000.0, 7), "pressure", "modlev", Mock(), 0
        )

        assert kind == "modlev"
        np.testing.assert_array_equal(coords, np.arange(7))

    def test_modlev_axis_limits(self: "TestModelLevelDisplay") -> None:
        """
        This test verifies that a pressure-extracted cross-section displayed in model levels has a y-axis of 0 to N-1 with the 'Model Level' label.

        Parameters:
            None

        Returns:
            None
        """
        _, ax = _plot(_make_processor(), display_vertical="modlev")

        assert ax.get_ylabel() == "Model Level"
        assert tuple(ax.get_ylim()) == (0.0, float(N_LEV - 1))


class TestMaxHeightLevels:
    """X3: automatic levels follow the visible data; modlev warns."""

    def test_visible_data_masks_pressure_above_limit(
        self: "TestMaxHeightLevels",
    ) -> None:
        """
        This test verifies that _visible_data_for_levels masks pressures below the barometric equivalent of max_height and leaves a model-level display untouched.

        Parameters:
            None

        Returns:
            None
        """
        vertical = np.array([1000.0, 800.0, 500.0, 100.0])
        data = np.arange(8, dtype=float).reshape(4, 2)

        masked = MPASVerticalCrossSectionPlotter._visible_data_for_levels(
            vertical, "pressure_hPa", data, 5.0
        )

        assert np.all(np.isfinite(masked[:2]))
        assert np.all(np.isnan(masked[3]))

        same = MPASVerticalCrossSectionPlotter._visible_data_for_levels(
            vertical, "modlev", data, 5.0
        )
        np.testing.assert_array_equal(same, data)

    def test_pressure_display_levels_use_visible_data(
        self: "TestMaxHeightLevels",
    ) -> None:
        """
        This test verifies that with max_height on a pressure display the automatic levels span only the visible (lower) part of the theta field.

        Parameters:
            None

        Returns:
            None
        """
        captured: Dict[str, Any] = {}
        original = MPASVerticalCrossSectionPlotter._resolve_plot_style

        def _spy(self_: Any, var_name: str, data: Any, *a: Any, **k: Any) -> Any:
            result = original(self_, var_name, data, *a, **k)
            captured["levels"] = np.asarray(result[1])
            return result

        with patch.object(MPASVerticalCrossSectionPlotter, "_resolve_plot_style", _spy):
            _plot(_make_processor(), display_vertical="pressure", max_height=3.0)

        # Theta at 3 km (about 710 hPa) is far below the model-top 330 K
        assert captured["levels"].max() < 318.0

    def test_modlev_with_max_height_warns(self: "TestMaxHeightLevels") -> None:
        """
        This test verifies that max_height on a model-level display logs a warning that it does not apply, rather than being silently ignored.

        Parameters:
            None

        Returns:
            None
        """
        with patch.object(xsec_module.logger, "warning") as warn:
            _plot(_make_processor(), display_vertical="modlev", max_height=5.0)

        assert any("max_height" in str(c.args[0]) for c in warn.call_args_list)


class TestPlotTypes:
    """X4: pcolormesh uses level boundaries; contour gets a colorbar."""

    def test_pcolormesh_uses_boundary_norm(self: "TestPlotTypes") -> None:
        """
        This test verifies that the pcolormesh cross-section is coloured with a BoundaryNorm built from the requested levels.

        Parameters:
            None

        Returns:
            None
        """
        levels = np.arange(290.0, 335.0, 5.0)
        fig, ax = _plot(
            _make_processor(),
            style=CrossSectionStyle(plot_type="pcolormesh", levels=levels),
        )
        meshes = [c for c in ax.collections if hasattr(c, "get_array")]
        norms = [c.norm for c in meshes if isinstance(c.norm, mcolors.BoundaryNorm)]

        assert norms
        np.testing.assert_allclose(norms[0].boundaries, levels)

    def test_contour_has_colorbar(self: "TestPlotTypes") -> None:
        """
        This test verifies that a plain contour cross-section now carries a colorbar axes.

        Parameters:
            None

        Returns:
            None
        """
        fig, _ = _plot(
            _make_processor(),
            style=CrossSectionStyle(
                plot_type="contour", levels=np.arange(290.0, 335.0, 5.0)
            ),
        )

        assert any(a.get_label() == "<colorbar>" for a in fig.axes)

    def test_colorbar_ticks_on_outer_side(self: "TestPlotTypes") -> None:
        """
        This test verifies that the colorbar tick labels sit on the right (outer) side so they cannot overlap the axes frame.

        Parameters:
            None

        Returns:
            None
        """
        fig, _ = _plot(_make_processor())

        cbar_ax = [a for a in fig.axes if a.get_label() == "<colorbar>"][0]

        assert cbar_ax.yaxis.get_label_position() == "right"
        assert cbar_ax.yaxis.get_tick_params()["labelright"]
        assert not cbar_ax.yaxis.get_tick_params()["labelleft"]


class TestClim:
    """X5: clim_min / clim_max bound the automatic levels."""

    def test_style_defaults_none(self: "TestClim") -> None:
        """
        This test verifies that the new style fields default to None.

        Parameters:
            None

        Returns:
            None
        """
        style = CrossSectionStyle()

        assert style.clim_min is None
        assert style.clim_max is None

    def test_auto_levels_bounded(self: "TestClim") -> None:
        """
        This test verifies that automatic levels start at clim_min and end at clim_max.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASVerticalCrossSectionPlotter()
        data = np.linspace(280.0, 340.0, 200).reshape(10, 20)
        _, levels = plotter._resolve_plot_style(
            "theta", data, None, None, clim_min=300.0, clim_max=320.0
        )

        assert levels[0] == pytest.approx(300.0)
        assert levels[-1] == pytest.approx(320.0)
        assert np.all(np.diff(levels) > 0)

    def test_explicit_levels_win(self: "TestClim") -> None:
        """
        This test verifies that user-supplied levels are not altered by the colour limits.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASVerticalCrossSectionPlotter()
        user = np.array([1.0, 2.0, 3.0])
        _, levels = plotter._resolve_plot_style(
            "theta", np.ones((3, 3)), "viridis", user, clim_min=0.0, clim_max=10.0
        )

        np.testing.assert_array_equal(levels, user)

    def test_inverted_limits_rejected(self: "TestClim") -> None:
        """
        This test verifies that clim_min >= clim_max raises a ValueError.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASVerticalCrossSectionPlotter()

        with pytest.raises(ValueError, match="clim_min"):
            plotter._resolve_plot_style(
                "theta", np.ones((3, 3)), None, None, clim_min=5.0, clim_max=1.0
            )

    def test_pcolormesh_norm_spans_limits(self: "TestClim") -> None:
        """
        This test verifies end to end that pcolormesh with clim limits uses a norm spanning exactly those limits.

        Parameters:
            None

        Returns:
            None
        """
        _, ax = _plot(
            _make_processor(),
            style=CrossSectionStyle(
                plot_type="pcolormesh", clim_min=300.0, clim_max=320.0
            ),
        )
        norms = [
            c.norm for c in ax.collections if isinstance(c.norm, mcolors.BoundaryNorm)
        ]

        assert norms[0].boundaries[0] == pytest.approx(300.0)
        assert norms[0].boundaries[-1] == pytest.approx(320.0)


class TestInputValidation:
    """X7, X8, X10: variable dims, time index, num_points, single-valued levels."""

    def test_edge_variable_rejected(self: "TestInputValidation") -> None:
        """
        This test verifies that an edge-located variable raises a clear ValueError that points at uReconstructZonal.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="uReconstructZonal"):
            _plot(_make_processor(with_edge_var=True), var_name="u")

    def test_out_of_range_time_index_raises(self: "TestInputValidation") -> None:
        """
        This test verifies that an out-of-range time index raises an IndexError with the valid range instead of being swallowed.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(IndexError, match="out of range"):
            _plot(_make_processor(), time_index=N_TIME)

        with pytest.raises(IndexError, match="out of range"):
            _plot(_make_processor(), time_index=-(N_TIME + 1))

    def test_negative_time_index_normalised(self: "TestInputValidation") -> None:
        """
        This test verifies that a Python-style negative time index is normalised and passed downstream as the equivalent non-negative index.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASVerticalCrossSectionPlotter(figsize=(8, 5), dpi=50)
        original = plotter._generate_cross_section_data
        seen: List[int] = []

        def _spy(proc: Any, var: Any, s: Any, e: Any, t: int, *a: Any, **k: Any) -> Any:
            seen.append(t)
            return original(proc, var, s, e, t, *a, **k)

        with patch.object(plotter, "_generate_cross_section_data", _spy):
            _plot(_make_processor(), plotter=plotter, time_index=-1)

        assert seen == [N_TIME - 1]

    def test_num_points_one_rejected(self: "TestInputValidation") -> None:
        """
        This test verifies that num_points=1 raises a clear ValueError (minimum 2).

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="at least 2"):
            _plot(_make_processor(), num_points=1)

    def test_single_valued_custom_levels(self: "TestInputValidation") -> None:
        """
        This test verifies that a single-valued custom level set is widened to a drawable band instead of failing in contouring.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASVerticalCrossSectionPlotter()
        _, levels = plotter._resolve_plot_style(
            "theta", np.ones((3, 3)), "viridis", np.array([300.0])
        )

        assert levels.size >= 2

        fig, _ = _plot(
            _make_processor(), style=CrossSectionStyle(levels=np.array([310.0]))
        )
        assert fig is not None


class TestGreatCirclePath:
    """X9: antipodal and coincident endpoints."""

    def test_same_point_raises(self: "TestGreatCirclePath") -> None:
        """
        This test verifies that identical endpoints raise a ValueError.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="same location"):
            MPASVerticalCrossSectionPlotter()._generate_great_circle_path(
                (10.0, 10.0), (10.0, 10.0), 5
            )

    @pytest.mark.parametrize(
        "start,end", [((0.0, 0.0), (180.0, 0.0)), ((20.0, 35.0), (-160.0, -35.0))]
    )
    def test_antipodal_path_is_finite_and_half_circumference(
        self: "TestGreatCirclePath", start: Any, end: Any
    ) -> None:
        """
        This test verifies that antipodal endpoints give a finite, deterministic path of half the Earth's circumference that starts and ends at the requested points.

        Parameters:
            start (Any): Start point (lon, lat).
            end (Any): Antipodal end point.

        Returns:
            None
        """
        plotter = MPASVerticalCrossSectionPlotter()
        lons, lats, dist = plotter._generate_great_circle_path(start, end, 9)
        lons2, lats2, _ = plotter._generate_great_circle_path(start, end, 9)

        assert np.all(np.isfinite(lons)) and np.all(np.isfinite(lats))
        assert dist[-1] == pytest.approx(np.pi * 6371.0, rel=1e-6)
        assert (lons[0], lats[0]) == start and (lons[-1], lats[-1]) == end
        assert np.max(np.abs(np.diff(dist) - dist[-1] / 8)) < 1e-6
        np.testing.assert_array_equal(lons, lons2)
        assert np.max(lats) > 80.0 or np.max(lats) >= max(start[1], end[1])

    def test_num_points_one_rejected(self: "TestGreatCirclePath") -> None:
        """
        This test verifies that the path generator rejects fewer than two points.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="at least 2"):
            MPASVerticalCrossSectionPlotter()._generate_great_circle_path(
                (0.0, 0.0), (1.0, 0.0), 1
            )

    def test_regular_path_unchanged(self: "TestGreatCirclePath") -> None:
        """
        This test verifies that an ordinary path still follows the equator with evenly spaced distances.

        Parameters:
            None

        Returns:
            None
        """
        (
            lons,
            lats,
            dist,
        ) = MPASVerticalCrossSectionPlotter()._generate_great_circle_path(
            (0.0, 0.0), (90.0, 0.0), 5
        )

        np.testing.assert_allclose(lons, [0.0, 22.5, 45.0, 67.5, 90.0], atol=1e-9)
        np.testing.assert_allclose(lats, 0.0, atol=1e-9)
        assert dist[-1] == pytest.approx(np.pi / 2 * 6371.0)


class TestTickFormatting:
    """X10: x tick labels and narrow-range pressure ticks."""

    def test_short_path_tick_labels_distinct(self: "TestTickFormatting") -> None:
        """
        This test verifies that the tick labels of a very short path are all different (enough decimals in the place names).

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = _plot(
            _make_processor(),
            start_point=(-90.0, 36.0),
            end_point=(-89.9, 36.1),
            num_points=10,
        )
        fig.canvas.draw()
        labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]

        assert len(labels) == len(set(labels)) >= 2

    def test_narrow_pressure_range_plain_labels(self: "TestTickFormatting") -> None:
        """
        This test verifies that a narrow pressure range gets plain hPa labels (no scientific notation) and at least two visible ticks.

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = _plot(_make_processor(), display_vertical="pressure")
        ax.set_ylim(990.0, 900.0)
        plotter = MPASVerticalCrossSectionPlotter()
        plotter.ax = ax
        plotter._apply_standard_pressure_ticks(np.array([900.0, 990.0]))
        fig.canvas.draw()
        labels = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
        minor = [t.get_text() for t in ax.get_yticklabels(minor=True) if t.get_text()]

        assert len(labels) >= 2
        assert not any("×" in text or "^" in text for text in labels + minor)


class TestSkewT:
    """X2 and X6: Skew-T CIN shading, x-limits, labels, barbs, and signed zeros."""

    @staticmethod
    def _profile(cold: bool = False) -> Dict[str, np.ndarray]:
        """
        This helper builds a smooth 40-level profile, optionally a cold polar one.

        Parameters:
            cold (bool): Whether to shift the profile to polar temperatures.

        Returns:
            Dict[str, np.ndarray]: Pressure (hPa), temperature and dewpoint (degC), and winds (kt).
        """
        p = np.linspace(1000.0, 120.0, 40)
        t = (
            28.0
            - (75.0 if cold else 90.0) * (1.0 - p / 1000.0) ** 0.9
            - (60.0 if cold else 0.0)
        )
        return {
            "p": p,
            "t": t,
            "td": t - 8.0,
            "u": np.linspace(5.0, 40.0, 40),
            "v": np.linspace(0.0, 15.0, 40),
        }

    def test_shade_cin_receives_dewpoint(self: "TestSkewT") -> None:
        """
        This test verifies that CIN shading is requested with the dewpoint, which limits it to the surface-to-EL layer.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag import MPASSkewTPlotter
        from metpy.units import units as munits

        prof = self._profile()
        skew = MagicMock()
        MPASSkewTPlotter(verbose=False)._plot_parcel_profile(
            skew,
            prof["p"] * munits.hPa,
            prof["t"] * munits.degC,
            prof["td"] * munits.degC,
            {"cape": 100.0, "cin": -20.0},
        )

        assert skew.shade_cin.called
        assert len(skew.shade_cin.call_args.args) == 4

    def test_cold_profile_fits_xlim(self: "TestSkewT") -> None:
        """
        This test verifies that a cold profile is fully inside the plotted window (the fixed -40 to 50 window used to clip it).

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag import MPASSkewTPlotter

        prof = self._profile(cold=True)
        fig, ax = MPASSkewTPlotter(
            figsize=(7, 9), dpi=50, verbose=False
        ).create_skewt_diagram(prof["p"], prof["t"], prof["td"])
        fig.canvas.draw()
        pts = ax.transData.transform(np.column_stack([prof["t"], prof["p"]]))
        frac = ax.transAxes.inverted().transform(pts)[:, 0]

        assert frac.min() >= 0.0 and frac.max() <= 1.0
        assert ax.get_xlim()[0] < -40.0

    def test_warm_profile_keeps_default_xlim(self: "TestSkewT") -> None:
        """
        This test verifies that a profile that already fits keeps the default -40 to 50 window.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag import MPASSkewTPlotter

        prof = self._profile()
        _, ax = MPASSkewTPlotter(
            figsize=(7, 9), dpi=50, verbose=False
        ).create_skewt_diagram(prof["p"], prof["t"], prof["td"])

        assert ax.get_xlim() == (-40.0, 50.0)

    def test_level_labels_inside_axes(self: "TestSkewT") -> None:
        """
        This test verifies that the LCL, LFC and EL labels are placed inside the axes rather than beyond the right edge.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag import MPASSkewTPlotter

        prof = self._profile()
        fig, ax = MPASSkewTPlotter(
            figsize=(7, 9), dpi=50, verbose=False
        ).create_skewt_diagram(
            prof["p"],
            prof["t"],
            prof["td"],
            indices={
                "lcl_pressure": 900.0,
                "lfc_pressure": 700.0,
                "el_pressure": 300.0,
            },
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes_box = ax.get_window_extent(renderer)
        labels = [t for t in ax.texts if "hPa" in t.get_text()]

        assert len(labels) == 3

        for text in labels:
            box = text.get_window_extent(renderer)
            assert box.x0 >= axes_box.x0 and box.x1 <= axes_box.x1

    def test_barbs_thinned(self: "TestSkewT") -> None:
        """
        This test verifies that wind barbs are thinned to at most one per 40 hPa.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag import MPASSkewTPlotter

        p = np.linspace(1000.0, 100.0, 200)
        keep = MPASSkewTPlotter._thin_levels(p, np.ones(p.shape, dtype=bool))

        assert 15 <= keep.sum() <= 25
        assert np.all(np.diff(np.sort(p[keep])) >= 40.0 - 1e-9)

    def test_no_negative_zero_in_table(self: "TestSkewT") -> None:
        """
        This test verifies that values that round to zero are shown as '0' or '0.0', never '-0' or '-0.0'.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag import MPASSkewTPlotter

        fig = plt.figure()
        MPASSkewTPlotter._add_indices_table(
            fig,
            {"sbcin": -0.2, "lifted_index": -0.04, "stp": -0.0, "sbcape": 5.0},
        )
        table = fig.axes[-1].tables[0]
        texts = [cell.get_text().get_text() for cell in table.get_celld().values()]

        assert not any(
            text.endswith(": -0") or text.endswith(": -0.0") for text in texts
        )
        assert any("SBCIN: 0 J/kg" == text for text in texts)

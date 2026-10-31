#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
Regression tests for the fourth round of visualization fixes: longitudes given in 0..360 render like the same data in -180..180 (the ESMF back-mapping and the precipitation filters used to drop the western hemisphere), a plotter reused for another mesh of the same size builds a new regridder, and the mesh fingerprint tells apart arrays that differ anywhere.
"""

from typing import Any, Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mpasdiag.processing.utils_geog import GeographicBounds
from mpasdiag.visualization import base_visualizer as bv
from mpasdiag.visualization.base_visualizer import MPASVisualizer
from mpasdiag.visualization.precipitation import (
    MPASPrecipitationPlotter,
    PrecipitationRenderStyle,
)
from mpasdiag.visualization.surface import MPASSurfacePlotter, SurfaceMapStyle

_RNG = np.random.default_rng(4)
_LON = _RNG.uniform(-180.0, 180.0, 4000)
_LAT = _RNG.uniform(-80.0, 80.0, 4000)
_WEST = GeographicBounds(-100.0, -40.0, 5.0, 45.0)


def _pixels(fig: Any) -> np.ndarray:
    """Render a figure and return its RGBA buffer without the footer (it carries a timestamp)."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[: int(rgba.shape[0] * 0.85)].copy()


def _render_both(make: Callable[[np.ndarray], Any]) -> None:
    """Render with -180..180 and with 0..360 longitudes and require identical maps."""
    reference = _pixels(make(_LON))
    shifted = _pixels(make(np.mod(_LON, 360.0)))
    assert np.array_equal(reference, shifted)


@pytest.mark.parametrize("plot_type", ["scatter", "contourf"])
def test_surface_map_accepts_0_360_longitudes(plot_type: str) -> None:
    data = 280.0 + 10.0 * np.sin(np.radians(_LON)) * np.cos(np.radians(_LAT))

    def make(lon: np.ndarray) -> Any:
        plotter = MPASSurfacePlotter(figsize=(4, 3), dpi=40)
        fig, _ = plotter.create_surface_map(
            lon,
            _LAT,
            data,
            "t2m",
            _WEST,
            style=SurfaceMapStyle(plot_type=plot_type, grid_resolution=2.0),
        )
        return fig

    _render_both(make)


@pytest.mark.parametrize("plot_type", ["scatter", "contourf"])
def test_precipitation_map_accepts_0_360_longitudes(plot_type: str) -> None:
    data = np.abs(5.0 * np.sin(np.radians(3 * _LON)) * np.cos(np.radians(_LAT)))

    def make(lon: np.ndarray) -> Any:
        plotter = MPASPrecipitationPlotter(figsize=(4, 3), dpi=40)
        fig, _ = plotter.create_precipitation_map(
            lon,
            _LAT,
            data,
            _WEST,
            style=PrecipitationRenderStyle(plot_type=plot_type, grid_resolution=2.0),
        )
        return fig

    _render_both(make)


def test_backmap_matches_across_longitude_frames() -> None:
    data = np.arange(_LON.size, dtype=float)
    full = MPASVisualizer._backmap_to_full_grid(
        np.mod(_LON, 360.0), _LAT, data, _LON, _LAT
    )
    assert np.array_equal(full, data)


def test_fingerprint_sees_changes_outside_old_sample() -> None:
    values = np.arange(10242.0)
    swapped = values.copy()
    swapped[[1, 2]] = swapped[[2, 1]]
    assert bv._array_fingerprint(values) != bv._array_fingerprint(swapped)
    assert bv._array_fingerprint(values) == bv._array_fingerprint(values.copy())


def test_large_arrays_use_constant_size_sample() -> None:
    values = np.zeros(bv._FULL_HASH_ELEMENTS + 10)
    size, dtype, digest = bv._array_fingerprint(values)
    assert size == values.size and dtype == "float64" and len(digest) == 32


def test_reused_plotter_rebuilds_regridder_for_other_mesh() -> None:
    lon_a = np.linspace(-170.0, 170.0, 50)
    lat = np.linspace(-60.0, 60.0, 50)
    lon_b = lon_a + 5.0
    bounds = np.zeros((50, 6))
    plotter = MPASVisualizer()
    with patch.object(bv, "MPASRemapper", side_effect=lambda **_: MagicMock()):
        first = plotter._get_or_build_remapper(
            lon_a, lat, bounds, bounds, -180, 180, -90, 90, 1.0
        )
        again = plotter._get_or_build_remapper(
            lon_a, lat, bounds, bounds, -180, 180, -90, 90, 1.0
        )
        other = plotter._get_or_build_remapper(
            lon_b, lat, bounds, bounds, -180, 180, -90, 90, 1.0
        )
        plotter._remapper_weights_dir = "/elsewhere"
        moved = plotter._get_or_build_remapper(
            lon_b, lat, bounds, bounds, -180, 180, -90, 90, 1.0
        )
    assert again is first
    assert other is not first
    assert moved is not other

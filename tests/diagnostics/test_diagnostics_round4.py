#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
Regression tests for the fourth round of diagnostics fixes: the sounding terrain height comes from the same source as the height profile (not from a zero-filled 'ter' in the output files), and an installed MetPy that fails to import falls back like a missing one.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from mpasdiag.diagnostics import sounding
from mpasdiag.diagnostics.sounding import SoundingDiagnostics

P = np.array([1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100.0])
T = np.array([25, 20, 15, 5, -12, -24, -38, -47, -55, -62, -66.0])
TD = np.array([20, 15, 10, -3, -20, -35, -48, -57, -65, -72, -76.0])


class TestTerrainSource:
    """The ground reference must share the datum of the height profile."""

    def _grid(self, tmp_path: Path) -> str:
        """Write a grid file whose terrain is 750 m in cell 1."""
        path = tmp_path / "grid.nc"
        zgrid = np.array([[100.0, 200.0, 300.0], [750.0, 900.0, 1100.0]])
        xr.Dataset(
            {
                "zgrid": (("nCells", "nVertLevelsP1"), zgrid),
                "ter": (("nCells",), zgrid[:, 0]),
            }
        ).to_netcdf(path)
        return str(path)

    def test_zero_ter_in_output_does_not_override_grid(self, tmp_path: Path) -> None:
        ds = xr.Dataset({"ter": (("Time", "nCells"), np.zeros((1, 2)))})
        d = SoundingDiagnostics(verbose=False)
        assert d._extract_surface_height(ds, self._grid(tmp_path), 1) == 750.0

    def test_dataset_zgrid_still_wins(self, tmp_path: Path) -> None:
        zg = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        ds = xr.Dataset({"zgrid": (("nCells", "nVertLevelsP1"), zg)})
        d = SoundingDiagnostics(verbose=False)
        assert d._extract_surface_height(ds, self._grid(tmp_path), 1) == 40.0

    def test_dataset_ter_is_last_resort(self) -> None:
        ds = xr.Dataset({"ter": (("nCells",), np.array([5.0, 7.0]))})
        d = SoundingDiagnostics(verbose=False)
        assert d._extract_surface_height(ds, None, 1) == 7.0


_REAL_GRID = Path(__file__).resolve().parents[2] / "data" / "grids"
_REAL_MPASOUT = Path(__file__).resolve().parents[2] / "data" / "u240k" / "mpasout"


@pytest.mark.skipif(
    not (_REAL_MPASOUT.is_dir() and (_REAL_GRID / "x1.10242.static.nc").is_file()),
    reason="u240k test data not available",
)
def test_real_sounding_terrain_matches_grid() -> None:
    """In the real mpasout files 'ter' is all zeros; the sounding must report the grid terrain."""
    from mpasdiag.processing.processors_3d import MPAS3DProcessor

    grid = str(_REAL_GRID / "x1.10242.static.nc")
    proc = MPAS3DProcessor(grid, verbose=False).load_3d_data(str(_REAL_MPASOUT))
    profile = SoundingDiagnostics(verbose=False).extract_sounding_profile(
        proc, -100.0, 35.0, time_index=0
    )
    with xr.open_dataset(grid) as g:
        expected = float(g["ter"].isel(nCells=profile["cell_index"]))
    assert expected > 100.0
    assert profile["surface_height"] == pytest.approx(expected)
    assert profile["height"][0] > profile["surface_height"]


def test_broken_metpy_falls_back_to_lcl_estimate() -> None:
    """An installed MetPy whose import fails behaves like a missing one instead of raising."""
    d = SoundingDiagnostics(verbose=False)
    with (
        patch.object(sounding, "HAS_METPY", True),
        patch.object(sounding, "_metpy_calc", side_effect=ImportError("pint mismatch")),
    ):
        result = d.compute_thermodynamic_indices(P, T, TD)
    assert result["lcl_pressure"] is not None
    assert result["cape"] is None


def test_skewt_reports_broken_metpy_clearly() -> None:
    """A failing metpy.plots import raises the 'MetPy is required' message at plot time."""
    from mpasdiag.visualization.skewt import MPASSkewTPlotter

    plotter = MPASSkewTPlotter(verbose=False)
    with patch.dict(sys.modules, {"metpy.plots": None}):
        with pytest.raises(ImportError, match="MetPy is required"):
            plotter._create_metpy_skewt(P, T, TD, None, None, False, None)

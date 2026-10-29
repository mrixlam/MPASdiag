#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
Regression tests for the fourth round of processing and command-line fixes: a pressure level outside the model's pressure range is an error instead of an empty plot, cross-section latitudes are validated, and the sounding distance error only claims the point is outside the domain when the automatic threshold was used.
"""

from types import SimpleNamespace
from typing import Any, List

import numpy as np
import pytest
import xarray as xr

from mpasdiag.diagnostics.sounding import SoundingDiagnostics
from mpasdiag.processing.cli_unified import MPASUnifiedCLI
from mpasdiag.processing.processors_3d import MPAS3DProcessor


def _processor() -> MPAS3DProcessor:
    """A 3D processor holding two columns whose pressure spans 1000-100 hPa."""
    pressure = np.array([[100000.0, 50000.0, 10000.0]] * 2)
    ds = xr.Dataset(
        {
            "theta": (("Time", "nCells", "nVertLevels"), np.ones((1, 2, 3))),
            "pressure_p": (("Time", "nCells", "nVertLevels"), np.zeros((1, 2, 3))),
            "pressure_base": (("Time", "nCells", "nVertLevels"), pressure[None]),
        }
    )
    proc = MPAS3DProcessor.__new__(MPAS3DProcessor)
    proc.dataset = ds
    proc.verbose = False
    return proc


class TestPressureRange:
    def test_level_outside_every_column_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside the model's pressure range"):
            _processor()._resolve_float_level("theta", 500000.0, "Time", 0)

    def test_level_inside_range_interpolates(self) -> None:
        _, field = _processor()._resolve_float_level("theta", 70000.0, "Time", 0)
        assert np.allclose(field.values, 1.0)

    def test_implausible_level_message_names_both_units(self) -> None:
        with pytest.raises(ValueError, match=r"100 Pa \(1 hPa\).*use 10000 Pa"):
            _processor()._resolve_float_level("theta", 100.0, "Time", 0)


class TestCrossSectionLatitude:
    def _errors(self, **coords: float) -> List[str]:
        config: Any = SimpleNamespace(
            analysis_type="cross",
            start_lon=10.0,
            start_lat=0.0,
            end_lon=20.0,
            end_lat=10.0,
        )
        for name, value in coords.items():
            setattr(config, name, value)
        errors: List[str] = []
        MPASUnifiedCLI()._validate_cross_section_params(config, errors)
        return errors

    def test_latitude_beyond_pole_rejected(self) -> None:
        assert self._errors(start_lat=100.0) == [
            "--start-lat 100.0 is outside [-90, 90]"
        ]
        assert self._errors(end_lat=-91.0) == ["--end-lat -91.0 is outside [-90, 90]"]

    def test_valid_transect_accepted(self) -> None:
        assert self._errors(start_lat=-90.0, end_lat=90.0) == []


def test_user_distance_limit_is_not_called_outside_domain() -> None:
    lon = np.array([0.0, 1.0, 2.0])
    lat = np.array([0.0, 0.0, 0.0])
    proc = SimpleNamespace()
    sd = SoundingDiagnostics(verbose=False)
    sd._load_grid_coordinates = lambda _p: (lon, lat)  # type: ignore[method-assign]
    with pytest.raises(ValueError) as info:
        sd._resolve_station(proc, 0.5, 0.3, None, None, 1.0)
    assert "max_distance_km=1." in str(info.value)
    assert "outside the model domain" not in str(info.value)

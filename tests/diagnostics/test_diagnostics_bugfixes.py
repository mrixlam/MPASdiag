#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Test Suite: Diagnostics Bug-Fix Regression Tests

This module holds regression tests for confirmed bugs in the diagnostics layer: wind direction convention, MetPy return handling in sounding indices, precipitation lookback and filtering, wet-bulb-zero height reference, level specification handling, and moisture transport conversions.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from typing import Any

from mpasdiag.diagnostics import (
    InsufficientLookbackError,
    MoistureTransportDiagnostics,
    PrecipitationDiagnostics,
    WindDiagnostics,
)
from mpasdiag.diagnostics.moisture_transport import (
    _trapezoidal_column_integral,
    mixing_ratio_to_specific_humidity,
)
from mpasdiag.diagnostics.sounding import SoundingDiagnostics
from mpasdiag.diagnostics.thermodynamic import ThermodynamicDiagnostics


class TestWindDirection:
    """Meteorological 'from' direction convention for compute_wind_direction."""

    @pytest.mark.parametrize(
        "u,v,expected",
        [(10, 0, 270), (-10, 0, 90), (0, 10, 180), (0, -10, 0), (10, 10, 225)],
    )
    def test_cardinal_directions_degrees(
        self: "TestWindDirection", u: float, v: float, expected: float
    ) -> None:
        """
        This test verifies that wind direction is the direction the wind blows from, so a westerly (u>0, v=0) gives 270 degrees and a northerly (v<0) gives 0 degrees.

        Parameters:
            u (float): Zonal component.
            v (float): Meridional component.
            expected (float): Expected direction in degrees.

        Returns:
            None
        """
        d = WindDiagnostics(verbose=False).compute_wind_direction(
            xr.DataArray([float(u)]), xr.DataArray([float(v)])
        )
        assert d.values[0] == pytest.approx(expected)

    def test_radians_and_calm(self: "TestWindDirection") -> None:
        """
        This test verifies the radians branch uses the same convention and that calm winds (0, 0) are assigned direction 0 in both branches.

        Parameters:
            None

        Returns:
            None
        """
        w = WindDiagnostics(verbose=False)
        u = xr.DataArray([10.0, 0.0])
        v = xr.DataArray([0.0, 0.0])
        rad = w.compute_wind_direction(u, v, degrees=False)
        assert rad.values[0] == pytest.approx(1.5 * np.pi)
        assert rad.values[1] == 0.0
        assert w.compute_wind_direction(u, v).values[1] == 0.0

    def test_shear_direction_uses_fixed_convention(self: "TestWindDirection") -> None:
        """
        This test verifies that the shear direction returned by compute_wind_shear follows the corrected convention (a pure westerly shear vector gives 270 degrees).

        Parameters:
            None

        Returns:
            None
        """
        w = WindDiagnostics(verbose=False)
        _, direction = w.compute_wind_shear(
            xr.DataArray([10.0]),
            xr.DataArray([0.0]),
            xr.DataArray([0.0]),
            xr.DataArray([0.0]),
        )
        assert direction.values[0] == pytest.approx(270.0)


class TestLevelSpecification:
    """Numpy integers accepted and small floats rejected with a helpful message."""

    def test_numpy_integer_accepted_wind(self: "TestLevelSpecification") -> None:
        """
        This test verifies WindDiagnostics accepts numpy integers and rejects floats below 1100 as probable hPa mistakes.

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset(
            {
                "uReconstructZonal": (
                    ("Time", "nCells", "nVertLevels"),
                    np.zeros((1, 2, 5)),
                )
            }
        )
        w = WindDiagnostics(verbose=False)
        assert (
            w._compute_level_index(ds, "uReconstructZonal", np.int64(3), "Time", 0) == 3
        )
        with pytest.raises(ValueError, match="Pa"):
            w._compute_level_index(ds, "uReconstructZonal", 850.0, "Time", 0)

    def test_numpy_integer_accepted_thermo(self: "TestLevelSpecification") -> None:
        """
        This test verifies ThermodynamicDiagnostics accepts numpy integers, rejects bool, and rejects small floats with a message mentioning Pa.

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset(
            {"theta": (("Time", "nCells", "nVertLevels"), np.zeros((1, 2, 5)))}
        )
        t = ThermodynamicDiagnostics(verbose=False)
        resolver: Any = t._resolve_level_index
        assert resolver(ds, "theta", np.int32(2), "Time", 0, "nVertLevels") == 2
        with pytest.raises(ValueError):
            resolver(ds, "theta", True, "Time", 0, "nVertLevels")
        with pytest.raises(ValueError, match="Pa"):
            resolver(ds, "theta", 850.0, "Time", 0, "nVertLevels")


class TestSoundingSafeHelpers:
    """_safe_pair and _safe_scalar NaN and tuple handling."""

    def test_safe_pair_nan_and_triple(self: "TestSoundingSafeHelpers") -> None:
        """
        This test verifies that NaN results are mapped to None and that a 3-tuple result (as from downdraft_cape) yields its first two elements.

        Parameters:
            None

        Returns:
            None
        """
        from metpy.units import units

        nan_pair = SoundingDiagnostics._safe_pair(
            lambda: (np.nan * units("J/kg"), 3.0 * units("J/kg"))
        )
        assert nan_pair == (None, 3.0)
        triple = SoundingDiagnostics._safe_pair(
            lambda: (5.0 * units("J/kg"), 1.0 * units.K, 2.0 * units.hPa)
        )
        assert triple[0] == 5.0

    def test_safe_scalar_nan(self: "TestSoundingSafeHelpers") -> None:
        """
        This test verifies that a NaN scalar is returned as None.

        Parameters:
            None

        Returns:
            None
        """
        from metpy.units import units

        assert SoundingDiagnostics._safe_scalar(lambda: np.nan * units.K) is None
        assert SoundingDiagnostics._safe_scalar(lambda: 2.0 * units.K) == 2.0


class TestSoundingIndices:
    """DCAPE, SWEAT and wet-bulb-zero behaviour on a synthetic sounding."""

    @staticmethod
    def _profile() -> Any:
        """
        This helper builds a moist synthetic sounding with winds and heights ordered surface to top.

        Parameters:
            None

        Returns:
            Any: Tuple (p_hpa, t_c, td_c, u_kt, v_kt, z_m).
        """
        p = np.linspace(1000, 200, 40)
        z = 100.0 + 44330.0 * (1 - (p / 1013.25) ** 0.1903)
        t = 25.0 - 6.5 * (z - 100.0) / 1000.0
        t = np.maximum(t, -60.0)
        td = t - 4.0
        u = np.linspace(5, 40, 40)
        v = np.linspace(5, 10, 40)
        return p, t, td, u, v, z

    def test_dcape_and_sweat_not_none(self: "TestSoundingIndices") -> None:
        """
        This test verifies that DCAPE and SWEAT are now computed rather than always None.

        Parameters:
            None

        Returns:
            None
        """
        p, t, td, u, v, z = self._profile()
        res = SoundingDiagnostics(verbose=False).compute_thermodynamic_indices(
            p, t, td, u, v, z
        )
        assert res["dcape"] is not None
        assert res["sweat_index"] is not None

    def test_wet_bulb_zero_agl_and_nan(self: "TestSoundingIndices") -> None:
        """
        This test verifies the wet-bulb-zero height is relative to the lowest level and that a NaN level does not create a spurious crossing.

        Parameters:
            None

        Returns:
            None
        """
        from metpy.units import units

        p, t, td, u, v, z = self._profile()
        pm, tm, tdm = p * units.hPa, t * units.degC, td * units.degC
        msl = SoundingDiagnostics._compute_wet_bulb_zero(pm, tm, tdm, z)
        agl = SoundingDiagnostics._compute_wet_bulb_zero(
            pm, tm, tdm, z, surface_height_m=z[0]
        )
        assert msl is not None and agl is not None
        assert msl - agl == pytest.approx(z[0])

        # A NaN dewpoint in a warm layer must not produce a crossing at the NaN
        td_nan = td.copy()
        td_nan[2] = np.nan
        with_nan = SoundingDiagnostics._compute_wet_bulb_zero(
            pm, tm, td_nan * units.degC, z
        )
        assert with_nan is not None
        assert with_nan == pytest.approx(msl, rel=0.05)


def _precip_ds(n: int = 7, step_hours: int = 1) -> xr.Dataset:
    """
    This helper builds a cumulative precipitation dataset with a datetime Time coordinate.

    Parameters:
        n (int): Number of time steps.
        step_hours (int): Output interval in hours.

    Returns:
        xr.Dataset: Dataset with rainnc and rainc.
    """
    times = pd.date_range("2024-01-01", periods=n, freq=f"{step_hours}h")
    cum = np.cumsum(np.ones((n, 4)), axis=0)
    return xr.Dataset(
        {
            "rainnc": (("Time", "nCells"), cum),
            "rainc": (("Time", "nCells"), cum * 0.5),
        },
        coords={"Time": times},
    )


class TestPrecipitationLookback:
    """Lookback resolution from the Time coordinate and error cases."""

    def test_hourly_unchanged(self: "TestPrecipitationLookback") -> None:
        """
        This test verifies hourly a03h results are the plain 3-step difference.

        Parameters:
            None

        Returns:
            None
        """
        r = PrecipitationDiagnostics(verbose=False).compute_precipitation_difference(
            _precip_ds(), 5, "rainnc", "a03h"
        )
        assert np.allclose(r.values, 3.0)

    def test_three_hourly_output(self: "TestPrecipitationLookback") -> None:
        """
        This test verifies a03h from 3-hourly output differences adjacent steps (3 mm), not 9 hours of output, and a01h raises since no output exists one hour earlier.

        Parameters:
            None

        Returns:
            None
        """
        d = PrecipitationDiagnostics(verbose=False)
        ds = _precip_ds(6, 3)
        assert np.allclose(
            d.compute_precipitation_difference(ds, 4, "rainnc", "a03h").values, 1.0
        )
        with pytest.raises(InsufficientLookbackError):
            d.compute_precipitation_difference(ds, 4, "rainnc", "a01h")

    def test_insufficient_and_bad_index(self: "TestPrecipitationLookback") -> None:
        """
        This test verifies insufficient lookback raises InsufficientLookbackError (no zero field), and negative or out-of-range indices raise ValueError.

        Parameters:
            None

        Returns:
            None
        """
        d = PrecipitationDiagnostics(verbose=False)
        ds = _precip_ds()
        with pytest.raises(InsufficientLookbackError):
            d.compute_precipitation_difference(ds, 2, "rainnc", "a06h")
        with pytest.raises(ValueError):
            d.compute_precipitation_difference(ds, -99, "rainnc", "a01h")
        with pytest.raises(ValueError):
            d.compute_precipitation_difference(ds, 99, "rainnc", "a01h")

    def test_no_time_coordinate_fallback(self: "TestPrecipitationLookback") -> None:
        """
        This test verifies that without a Time coordinate the hourly assumption is used and still raises when lookback is short.

        Parameters:
            None

        Returns:
            None
        """
        ds = _precip_ds().drop_vars("Time")
        d = PrecipitationDiagnostics(verbose=False)
        assert np.allclose(
            d.compute_precipitation_difference(ds, 3, "rainnc", "a01h").values, 1.0
        )
        with pytest.raises(InsufficientLookbackError):
            d.compute_precipitation_difference(ds, 0, "rainnc", "a01h")


class TestPrecipitationFilters:
    """NaN preservation and reset handling."""

    def test_nan_kept_negative_nan_roundoff_zero(
        self: "TestPrecipitationFilters",
    ) -> None:
        """
        This test verifies NaN stays NaN, real negatives become NaN, round-off negatives become zero and values at or above 1e5 become NaN.

        Parameters:
            None

        Returns:
            None
        """
        d = PrecipitationDiagnostics(verbose=False)
        data = xr.DataArray(np.array([np.nan, -5.0, -1e-9, 2.0, 2e5]))
        r = d._apply_precipitation_filters_and_attributes(data, "rainnc")
        assert np.isnan(r.values[0]) and np.isnan(r.values[1]) and np.isnan(r.values[4])
        assert r.values[2] == 0.0 and r.values[3] == 2.0

    def test_reset_through_compute(self: "TestPrecipitationFilters") -> None:
        """
        This test verifies a bucket reset in the cumulative field yields NaN in the accumulation, not zero.

        Parameters:
            None

        Returns:
            None
        """
        ds = _precip_ds()
        ds["rainnc"][3, 0] = 0.0
        r = PrecipitationDiagnostics(verbose=False).compute_precipitation_difference(
            ds, 3, "rainnc", "a01h"
        )
        assert np.isnan(r.values[0])


class TestMoistureTransport:
    """Mixing-ratio conversion and surface layer in column integrals."""

    def test_mixing_ratio_conversion(self: "TestMoistureTransport") -> None:
        """
        This test verifies q = w / (1 + w).

        Parameters:
            None

        Returns:
            None
        """
        q = mixing_ratio_to_specific_humidity(xr.DataArray([0.02]))
        assert q.values[0] == pytest.approx(0.02 / 1.02)

    def test_surface_layer_included(self: "TestMoistureTransport") -> None:
        """
        This test verifies that supplying surface pressure adds the surface-to-lowest-level layer using the lowest-level value, and omitting it keeps the old result.

        Parameters:
            None

        Returns:
            None
        """
        p = np.array([[95000.0, 90000.0, 85000.0]])
        x = np.ones((1, 3))
        base = _trapezoidal_column_integral(x, p)
        with_sfc = _trapezoidal_column_integral(x, p, np.array([100000.0]))
        assert with_sfc[0] - base[0] == pytest.approx(5000.0 / 9.80665, rel=1e-3)

    def test_iwv_mixing_ratio_flag(self: "TestMoistureTransport") -> None:
        """
        This test verifies compute_iwv with is_mixing_ratio=True gives a smaller value than treating w as q.

        Parameters:
            None

        Returns:
            None
        """
        m = MoistureTransportDiagnostics(verbose=False)
        w = xr.DataArray(np.full((1, 3), 0.02), dims=["nCells", "nVertLevels"])
        p = xr.DataArray(
            np.array([[90000.0, 70000.0, 50000.0]]), dims=["nCells", "nVertLevels"]
        )
        assert float(m.compute_iwv(w, p, is_mixing_ratio=True)) < float(
            m.compute_iwv(w, p)
        )


class TestSweatUnitsAndSoundingCoordinates:
    """Regression tests for the SWEAT wind-speed units and the sounding grid coordinate conversion."""

    def test_sweat_matches_metpy_with_knots(
        self: "TestSweatUnitsAndSoundingCoordinates",
    ) -> None:
        """
        This regression test verifies that the SWEAT index equals MetPy's sweat_index evaluated with wind speed in knots. SWEAT is defined in knots and MetPy sums the raw speed magnitudes, so passing m/s understated the index (147 instead of 175 on a real column).

        Parameters:
            None

        Returns:
            None
        """
        import metpy.calc as mpcalc
        from metpy.units import units

        p, t, td, u, v, z = TestSoundingIndices._profile()
        res = SoundingDiagnostics(verbose=False).compute_thermodynamic_indices(
            p, t, td, u, v, z
        )
        u_q, v_q = u * units.knots, v * units.knots
        expected = mpcalc.sweat_index(
            p * units.hPa,
            t * units.degC,
            td * units.degC,
            mpcalc.wind_speed(u_q, v_q).to("knots"),
            mpcalc.wind_direction(u_q, v_q),
        )
        assert res["sweat_index"] == pytest.approx(float(np.ravel(expected.m)[0]))

    def test_grid_coordinates_in_degrees_near_equator_are_not_converted(
        self: "TestSweatUnitsAndSoundingCoordinates", tmp_path: Any
    ) -> None:
        """
        This regression test verifies that a degree-valued grid lying within a few degrees of the equator and prime meridian is not mistaken for radians when locating the nearest cell for a sounding (the former max|lat| <= pi test multiplied such coordinates by 57.3), while a radian-valued grid is still converted.

        Parameters:
            tmp_path (Any): Pytest temporary directory.

        Returns:
            None
        """
        from types import SimpleNamespace
        from mpasdiag.diagnostics import sounding as sounding_module

        lon_deg, lat_deg = np.array([1.0, 2.0, 3.0]), np.array([-1.0, 0.5, 2.0])
        for name, lon, lat in [
            ("deg.nc", lon_deg, lat_deg),
            ("rad.nc", np.radians(lon_deg), np.radians(lat_deg)),
        ]:
            path = str(tmp_path / name)
            xr.Dataset(
                {"lonCell": (("nCells",), lon), "latCell": (("nCells",), lat)}
            ).to_netcdf(path)
            sounding_module._GRID_COORD_CACHE.pop(path, None)
            got_lon, got_lat = SoundingDiagnostics(
                verbose=False
            )._load_grid_coordinates(
                SimpleNamespace(grid_file=path)  # type: ignore[arg-type]
            )
            np.testing.assert_allclose(got_lon, lon_deg)
            np.testing.assert_allclose(got_lat, lat_deg)

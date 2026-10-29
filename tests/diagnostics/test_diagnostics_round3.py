#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
Regression tests for the third round of diagnostics fixes: Bunkers-relative SRH, accumulation period parsing and negative time indices, dask/numpy support in the wind and thermodynamic diagnostics, sounding input validation, circular wind direction statistics, exact density potential temperature, mass-level destaggering of interface variables, specific humidity metadata and terrain-referenced sounding layers.
"""

from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mpasdiag.diagnostics.moisture_transport import mixing_ratio_to_specific_humidity
from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
from mpasdiag.diagnostics.sounding import SoundingDiagnostics
from mpasdiag.diagnostics.thermodynamic import EPSILON, ThermodynamicDiagnostics
from mpasdiag.diagnostics.wind import WindDiagnostics

P = np.array([1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100.0])
T = np.array([25, 20, 15, 5, -12, -24, -38, -47, -55, -62, -66.0])
TD = np.array([20, 15, 10, -3, -20, -35, -48, -57, -65, -72, -76.0])
Z = np.array([110, 800, 1500, 3100, 5800, 7500, 9600, 10800, 12300, 14200, 16500.0])
U = np.linspace(5.0, 40.0, 11)
V = np.linspace(0.0, -10.0, 11)


def _veering() -> Tuple[np.ndarray, ...]:
    """
    This helper builds a sounding whose wind veers and strengthens with height, so that the storm-relative helicity differs clearly between a ground-relative and a Bunkers-relative reference.

    Parameters:
        None

    Returns:
        Tuple[np.ndarray, ...]: (pressure_hPa, temperature_C, dewpoint_C, u_kt, v_kt, height_m).
    """
    z = np.linspace(0.0, 10000.0, 21)
    p = 1000.0 * np.exp(-z / 7500.0)
    t = 25.0 - 6.5e-3 * z
    td = t - 5.0
    ang = np.deg2rad(np.linspace(160.0, 260.0, z.size))
    spd = np.linspace(8.0, 60.0, z.size)
    return p, t, td, -spd * np.sin(ang), -spd * np.cos(ang), z


class TestSrhBunkers:
    """SRH is relative to the Bunkers right-mover, and STP/SCP follow."""

    def test_srh_matches_metpy_with_bunkers(self: "TestSrhBunkers") -> None:
        """
        This test checks the 0-1 km and 0-3 km SRH against MetPy computed directly with the Bunkers right-mover, and that it differs from the ground-relative value.

        Parameters:
            None

        Returns:
            None
        """
        mc = pytest.importorskip("metpy.calc")
        from metpy.units import units as mu

        p, t, td, u, v, z = _veering()
        idx = SoundingDiagnostics(verbose=False).compute_thermodynamic_indices(
            p, t, td, u, v, z
        )
        um, vm = u / 1.94384 * mu("m/s"), v / 1.94384 * mu("m/s")
        rm, _, _ = mc.bunkers_storm_motion(p * mu.hPa, um, vm, z * mu.m)
        for depth, key in [(1000, "srh_0_1km"), (3000, "srh_0_3km")]:
            ref = mc.storm_relative_helicity(
                z * mu.m,
                um,
                vm,
                depth=depth * mu.m,
                storm_u=rm[0],
                storm_v=rm[1],
            )[2].m
            ground = mc.storm_relative_helicity(z * mu.m, um, vm, depth=depth * mu.m)[
                2
            ].m
            assert idx[key] == pytest.approx(ref, rel=1e-6)
            assert abs(idx[key] - ground) > 1.0

    def test_bunkers_failure_gives_none(self: "TestSrhBunkers") -> None:
        """
        This test checks that a profile with constant wind (undefined Bunkers motion) gives None SRH instead of a ground-relative number.

        Parameters:
            None

        Returns:
            None
        """
        p, t, td, u, v, z = _veering()
        idx = SoundingDiagnostics(verbose=False).compute_thermodynamic_indices(
            p, t, td, np.full_like(u, 10.0), np.full_like(v, 5.0), z
        )
        assert idx["srh_0_1km"] is None and idx["srh_0_3km"] is None


class TestPrecipitationPeriods:
    """Accumulation period parsing and negative time indices."""

    @pytest.mark.parametrize(
        "period,hours",
        [("a01h", 1), ("a1h", 1), ("A03h", 3), ("a48h", 48), ("a240h", 240)],
    )
    def test_valid_periods(
        self: "TestPrecipitationPeriods", period: str, hours: int
    ) -> None:
        """
        This test checks that a<N>h with 1 to 240 hours is parsed case-insensitively.

        Parameters:
            period (str): Period identifier.
            hours (int): Expected hours.

        Returns:
            None
        """
        assert (
            PrecipitationDiagnostics(verbose=False).get_accumulation_hours(period)
            == hours
        )

    @pytest.mark.parametrize("period", ["foo", "a00h", "a241h", "24h", "a1d", ""])
    def test_invalid_periods_raise(
        self: "TestPrecipitationPeriods", period: str
    ) -> None:
        """
        This test checks that unknown identifiers raise ValueError instead of mapping to 24 hours.

        Parameters:
            period (str): Period identifier.

        Returns:
            None
        """
        with pytest.raises(ValueError):
            PrecipitationDiagnostics(verbose=False).get_accumulation_hours(period)

    def test_negative_time_index(self: "TestPrecipitationPeriods") -> None:
        """
        This test checks that time_index=-1 equals the last index and that an out-of-range negative index still raises.

        Parameters:
            None

        Returns:
            None
        """
        times = pd.date_range("2024-01-01", periods=5, freq="1h")
        cum = np.cumsum(np.arange(1, 6.0))[:, None] * np.ones((5, 3))
        ds = xr.Dataset({"rainnc": (("Time", "nCells"), cum)}, coords={"Time": times})
        d = PrecipitationDiagnostics(verbose=False)
        last = d.compute_precipitation_difference(ds, 4, "rainnc", "a01h")
        neg = d.compute_precipitation_difference(ds, -1, "rainnc", "a01h")
        np.testing.assert_allclose(neg.values, last.values)
        with pytest.raises(ValueError):
            d.compute_precipitation_difference(ds, -6, "rainnc", "a01h")


class TestWindDaskNumpy:
    """Wind diagnostics accept dask-backed DataArrays and plain arrays."""

    def test_speed_direction_shear_dask(self: "TestWindDaskNumpy") -> None:
        """
        This test checks that dask-backed inputs give lazy results equal to the eager ones, with attrs kept.

        Parameters:
            None

        Returns:
            None
        """
        pytest.importorskip("dask")
        rng = np.random.default_rng(0)
        u = xr.DataArray(
            rng.normal(size=(4, 6)), dims=["a", "b"], attrs={"units": "m/s"}
        )
        v = xr.DataArray(rng.normal(size=(4, 6)), dims=["a", "b"])
        w = WindDiagnostics(verbose=True)
        uc, vc = u.chunk({"a": 2}), v.chunk({"a": 2})
        spd = w.compute_wind_speed(uc, vc)
        assert spd.chunks is not None
        np.testing.assert_allclose(spd.values, w.compute_wind_speed(u, v).values)
        assert spd.attrs["standard_name"] == "wind_speed"
        for deg in (True, False):
            d = w.compute_wind_direction(uc, vc, degrees=deg)
            assert d.chunks is not None
            np.testing.assert_allclose(
                d.values, w.compute_wind_direction(u, v, degrees=deg).values
            )
        sh, sd = w.compute_wind_shear(uc, vc, uc * 0.5, vc * 0.2)
        rs, rd = w.compute_wind_shear(u, v, u * 0.5, v * 0.2)
        np.testing.assert_allclose(sh.values, rs.values)
        np.testing.assert_allclose(sd.values, rd.values)

    def test_numpy_arrays(self: "TestWindDaskNumpy") -> None:
        """
        This test checks that plain arrays are accepted and plain arrays are returned.

        Parameters:
            None

        Returns:
            None
        """
        w = WindDiagnostics(verbose=False)
        u, v = np.array([3.0, 0.0, -4.0]), np.array([4.0, 0.0, 3.0])
        spd = w.compute_wind_speed(u, v)
        assert isinstance(spd, np.ndarray)
        np.testing.assert_allclose(spd, [5.0, 0.0, 5.0])
        direction = w.compute_wind_direction(u, v)
        assert isinstance(direction, np.ndarray) and direction[1] == 0.0
        sh, sd = w.compute_wind_shear(u, v, u * 0, v * 0)
        assert isinstance(sh, np.ndarray) and isinstance(sd, np.ndarray)


class TestThermoNumpyDask:
    """Thermodynamic diagnostics accept plain arrays and dask."""

    def test_numpy_arrays(self: "TestThermoNumpyDask") -> None:
        """
        This test checks the three functions with plain arrays against the exact formulas.

        Parameters:
            None

        Returns:
            None
        """
        td = ThermodynamicDiagnostics(verbose=False)
        theta, qv, qc = (
            np.array([300.0, 290.0]),
            np.array([0.01, 0.0]),
            np.array([0.001, 0.0]),
        )
        tv = td.compute_virtual_potential_temperature(theta, qv)
        qt = td.compute_total_water_mixing_ratio(qv, qc)
        thr = td.compute_density_potential_temperature(theta, qv, qc=qc)
        assert all(isinstance(x, np.ndarray) for x in (tv, qt, thr))
        np.testing.assert_allclose(tv, theta * (1 + qv / EPSILON) / (1 + qv))
        np.testing.assert_allclose(qt, qv + qc)
        np.testing.assert_allclose(thr, theta * (1 + qv / EPSILON) / (1 + qv + qc))

    def test_dask_lazy_and_attrs(self: "TestThermoNumpyDask") -> None:
        """
        This test checks that dask inputs stay lazy and the result carries attrs.

        Parameters:
            None

        Returns:
            None
        """
        pytest.importorskip("dask")
        td = ThermodynamicDiagnostics(verbose=True)
        theta = xr.DataArray(np.full(6, 300.0), dims=["x"]).chunk({"x": 3})
        qv = xr.DataArray(np.full(6, 0.01), dims=["x"]).chunk({"x": 3})
        tr = td.compute_density_potential_temperature(theta, qv)
        assert tr.chunks is not None and tr.attrs["units"] == "K"
        assert td.compute_virtual_potential_temperature(theta, qv).chunks is not None
        np.testing.assert_allclose(tr.values, 300.0 * (1 + 0.01 / EPSILON) / 1.01)

    def test_density_equals_virtual_without_condensate(
        self: "TestThermoNumpyDask",
    ) -> None:
        """
        This test checks theta_rho equals theta_v when there is no condensate (exact form).

        Parameters:
            None

        Returns:
            None
        """
        td = ThermodynamicDiagnostics(verbose=False)
        theta = xr.DataArray(np.array([300.0, 250.0]), dims=["x"])
        qv = xr.DataArray(np.array([0.02, 0.001]), dims=["x"])
        np.testing.assert_allclose(
            td.compute_density_potential_temperature(theta, qv).values,
            td.compute_virtual_potential_temperature(theta, qv).values,
        )


class TestSoundingValidation:
    """Input validation of compute_thermodynamic_indices."""

    def test_pa_raises(self: "TestSoundingValidation") -> None:
        """
        This test checks that Pa pressures raise ValueError.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="hPa"):
            SoundingDiagnostics(verbose=False).compute_thermodynamic_indices(
                P * 100, T, TD
            )

    def test_kelvin_raises(self: "TestSoundingValidation") -> None:
        """
        This test checks that Kelvin temperatures raise ValueError.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="Kelvin"):
            SoundingDiagnostics(verbose=False).compute_thermodynamic_indices(
                P, T + 273.15, TD + 273.15
            )

    def test_length_mismatch_raises(self: "TestSoundingValidation") -> None:
        """
        This test checks that mismatched lengths (including winds and heights) raise ValueError.

        Parameters:
            None

        Returns:
            None
        """
        d = SoundingDiagnostics(verbose=False)
        with pytest.raises(ValueError, match="same length"):
            d.compute_thermodynamic_indices(P, T[:5], TD)
        with pytest.raises(ValueError, match="same length"):
            d.compute_thermodynamic_indices(P, T, TD, U[:3], V, Z)

    def test_ascending_pressure_sorted(self: "TestSoundingValidation") -> None:
        """
        This test checks that a top-first profile gives the same indices as the surface-first one.

        Parameters:
            None

        Returns:
            None
        """
        d = SoundingDiagnostics(verbose=False)
        ref = d.compute_thermodynamic_indices(P, T, TD, U, V, Z)
        rev = d.compute_thermodynamic_indices(
            P[::-1], T[::-1], TD[::-1], U[::-1], V[::-1], Z[::-1]
        )
        assert ref["sbcape"] is not None
        for k, val in ref.items():
            if val is None:
                assert rev[k] is None
            else:
                assert rev[k] == pytest.approx(val)

    def test_dewpoint_clipped_with_warning(
        self: "TestSoundingValidation", caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        This test checks that a dewpoint above the temperature is clipped (same result as td == t) with a warning.

        Parameters:
            caplog (pytest.LogCaptureFixture): Log capture fixture.

        Returns:
            None
        """
        d = SoundingDiagnostics(verbose=False)
        import logging

        with caplog.at_level(logging.WARNING):
            a = d.compute_thermodynamic_indices(P, T, T + 5)
        b = d.compute_thermodynamic_indices(P, T, T)
        assert a == b
        assert "clipping" in caplog.text or any(
            "clipping" in r.getMessage() for r in caplog.records
        )

    def test_normal_profile_unchanged(self: "TestSoundingValidation") -> None:
        """
        This test checks a normal profile still returns CAPE (the skew-T caller path) and lists are accepted.

        Parameters:
            None

        Returns:
            None
        """
        d = SoundingDiagnostics(verbose=False)
        r = d.compute_thermodynamic_indices(list(P), list(T), list(TD))
        assert r["sbcape"] == pytest.approx(1855.6, abs=1.0)


class TestCircularDirection:
    """Wind direction statistics are circular."""

    def test_wraparound_mean(self: "TestCircularDirection") -> None:
        """
        This test checks that 359 and 1 degrees average to 0 (not 180) and the std is small, while min/max stay plain.

        Parameters:
            None

        Returns:
            None
        """
        # from-direction 359 deg: u = -sin(d), v = -cos(d)
        d = np.deg2rad([359.0, 1.0])
        u = xr.DataArray(-5.0 * np.sin(d), dims=["x"])
        v = xr.DataArray(-5.0 * np.cos(d), dims=["x"])
        out = WindDiagnostics(verbose=False).analyze_wind_components(u, v)["direction"]
        assert min(out["mean"], 360.0 - out["mean"]) < 1e-6
        assert 0.0 <= out["mean"] < 360.0
        assert out["std"] == pytest.approx(1.0, abs=0.01)
        assert out["min"] == pytest.approx(1.0) and out["max"] == pytest.approx(359.0)

    def test_plain_mean(self: "TestCircularDirection") -> None:
        """
        This test checks that 90 and 180 degrees average to 135.

        Parameters:
            None

        Returns:
            None
        """
        d = np.deg2rad([90.0, 180.0])
        u = xr.DataArray(-5.0 * np.sin(d), dims=["x"])
        v = xr.DataArray(-5.0 * np.cos(d), dims=["x"])
        out = WindDiagnostics(verbose=False).analyze_wind_components(u, v)["direction"]
        assert out["mean"] == pytest.approx(135.0)


def _p1_dataset() -> xr.Dataset:
    """
    This helper builds a small dataset with u, v on mass levels and w on interface levels whose value equals the interface index.

    Parameters:
        None

    Returns:
        xr.Dataset: Dataset with dims (Time, nCells, nVertLevels[P1]).
    """
    nl = 4
    k = np.arange(nl, dtype=float)
    u = np.broadcast_to(k + 0.5, (1, 3, nl)).copy()
    w = np.broadcast_to(np.arange(nl + 1, dtype=float), (1, 3, nl + 1)).copy()
    return xr.Dataset(
        {
            "u": (("Time", "nCells", "nVertLevels"), u),
            "v": (("Time", "nCells", "nVertLevels"), u * 0.0),
            "w": (("Time", "nCells", "nVertLevelsP1"), w, {"units": "m/s"}),
        }
    )


class TestInterfaceDestaggering:
    """w on nVertLevelsP1 is returned at the mass level (mean of two interfaces)."""

    def test_integer_levels(self: "TestInterfaceDestaggering") -> None:
        """
        This test checks that level k of w is the mean of interfaces k and k+1, co-located with u.

        Parameters:
            None

        Returns:
            None
        """
        w = WindDiagnostics(verbose=False)
        ds = _p1_dataset()
        for k in range(4):
            wk = w.get_3d_variable_at_level(ds, "w", k, 0)
            uk = w.get_3d_variable_at_level(ds, "u", k, 0)
            np.testing.assert_allclose(wk.values, k + 0.5)
            np.testing.assert_allclose(wk.values, uk.values)
            assert wk.attrs["units"] == "m/s" and wk.attrs["level_index"] == k

    def test_top_and_surface(self: "TestInterfaceDestaggering") -> None:
        """
        This test checks that 'top' is the top mass level (not the zero-w lid interface) and that bounds are enforced.

        Parameters:
            None

        Returns:
            None
        """
        w = WindDiagnostics(verbose=False)
        ds = _p1_dataset()
        np.testing.assert_allclose(
            w.get_3d_variable_at_level(ds, "w", "top", 0).values, 3.5
        )
        np.testing.assert_allclose(
            w.get_3d_variable_at_level(ds, "w", "surface", 0).values, 0.5
        )
        with pytest.raises(ValueError):
            w.get_3d_variable_at_level(ds, "w", 4, 0)

    def test_wind_components_colocated(self: "TestInterfaceDestaggering") -> None:
        """
        This test checks get_3d_wind_components returns u, v and w at the same mass level.

        Parameters:
            None

        Returns:
            None
        """
        u, v, wv = WindDiagnostics(verbose=False).get_3d_wind_components(
            _p1_dataset(), "u", "v", "w", level="top", time_index=0
        )
        np.testing.assert_allclose(u.values, wv.values)


class TestSpecificHumidityMetadata:
    """Specific humidity carries its own long_name and standard_name."""

    def test_attrs(self: "TestSpecificHumidityMetadata") -> None:
        """
        This test checks the names are those of specific humidity, not the mixing ratio.

        Parameters:
            None

        Returns:
            None
        """
        w = xr.DataArray(
            np.array([0.01]),
            dims=["x"],
            attrs={"long_name": "Water vapor mixing ratio", "units": "kg kg-1"},
        )
        q = mixing_ratio_to_specific_humidity(w)
        assert q.attrs["long_name"] == "Specific humidity"
        assert q.attrs["standard_name"] == "specific_humidity"
        assert float(q) == pytest.approx(0.01 / 1.01)


class TestTerrainReference:
    """Wet-bulb-zero height and shear layers can start at the terrain."""

    def test_wet_bulb_zero_from_terrain(self: "TestTerrainReference") -> None:
        """
        This test checks the wet-bulb-zero height is measured from the terrain when given, and from the lowest level otherwise.

        Parameters:
            None

        Returns:
            None
        """
        d = SoundingDiagnostics(verbose=False)
        base = d.compute_thermodynamic_indices(P, T, TD, height_m=Z)[
            "wet_bulb_zero_height"
        ]
        ter = d.compute_thermodynamic_indices(
            P, T, TD, height_m=Z, surface_height_m=0.0
        )["wet_bulb_zero_height"]
        assert base is not None and ter is not None
        assert ter - base == pytest.approx(Z[0])

    def test_layers_use_terrain(self: "TestTerrainReference") -> None:
        """
        This test checks that with the terrain below the lowest level the 0-1 km shear is computed over terrain..terrain+1 km, and that surface_height_m at or above the lowest level changes nothing.

        Parameters:
            None

        Returns:
            None
        """
        p, t, td, u, v, z = _veering()
        z = z + 500.0
        d = SoundingDiagnostics(verbose=False)
        none = d.compute_thermodynamic_indices(p, t, td, u, v, z)
        same = d.compute_thermodynamic_indices(p, t, td, u, v, z, surface_height_m=z[0])
        assert none["bulk_shear_0_1km"] == pytest.approx(same["bulk_shear_0_1km"])
        ter = d.compute_thermodynamic_indices(p, t, td, u, v, z, surface_height_m=480.0)
        assert ter["bulk_shear_0_1km"] is not None
        assert ter["bulk_shear_0_1km"] != pytest.approx(
            none["bulk_shear_0_1km"], rel=1e-4
        )
        assert ter["srh_0_3km"] is not None

    def test_extract_surface_height(self: "TestTerrainReference") -> None:
        """
        This test checks the terrain lookup from zgrid level 0 and from 'ter', and None when neither exists.

        Parameters:
            None

        Returns:
            None
        """
        d = SoundingDiagnostics(verbose=False)
        zg = np.array([[100.0, 200.0, 300.0], [150.0, 250.0, 350.0]])
        ds = xr.Dataset({"zgrid": (("nCells", "nVertLevelsP1"), zg)})
        assert d._extract_surface_height(ds, None, 1) == pytest.approx(150.0)
        ds2 = xr.Dataset({"ter": (("nCells",), np.array([5.0, 7.0]))})
        assert d._extract_surface_height(ds2, None, 1) == pytest.approx(7.0)
        assert d._extract_surface_height(xr.Dataset(), None, 0) is None

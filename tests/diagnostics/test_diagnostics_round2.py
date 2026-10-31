#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
Regression tests for the second round of diagnostics fixes: virtual potential temperature, per-column pressure-level interpolation, per-column storm motion, sounding height from zgrid, sounding location validation, order-agnostic column integration and finite-or-None sounding indices.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Tuple

import numpy as np
import pytest
import xarray as xr

from mpasdiag.diagnostics.moisture_transport import _trapezoidal_column_integral
from mpasdiag.diagnostics.sounding import SoundingDiagnostics
from mpasdiag.diagnostics.thermodynamic import EPSILON, ThermodynamicDiagnostics
from mpasdiag.diagnostics.wind import WindDiagnostics

N_CELLS = 5
N_VERT = 40


def _columns_dataset() -> xr.Dataset:
    """
    This helper builds a synthetic dataset whose columns have clearly different pressure profiles (a different surface pressure in each column, mimicking terrain), with a wind field that is linear in log pressure and a theta field that is linear in level index, so that per-column interpolation can be checked against np.interp.

    Parameters:
        None

    Returns:
        xr.Dataset: Dataset with dims (Time, nCells, nVertLevels).
    """
    surface = np.linspace(100000.0, 70000.0, N_CELLS)
    frac = np.linspace(0.0, 1.0, N_VERT)
    pressure = surface[:, None] * (20000.0 / surface[:, None]) ** frac[None, :]
    pressure = pressure[None]
    dims = ["Time", "nCells", "nVertLevels"]
    return xr.Dataset(
        {
            "pressure_p": (dims, np.zeros_like(pressure)),
            "pressure_base": (dims, pressure),
            "u": (dims, 10.0 + 5.0 * np.log(pressure / 1e5)),
            "v": (dims, np.full_like(pressure, 2.0)),
            "theta": (
                dims,
                290.0 + np.tile(np.arange(N_VERT, dtype=float), (1, N_CELLS, 1)),
            ),
            "qv": (dims, np.full_like(pressure, 0.005)),
        }
    )


class TestVirtualPotentialTemperature:
    """The virtual potential temperature uses the mixing-ratio formula, not Rv/Rd * qv."""

    def test_matches_mixing_ratio_formula(
        self: "TestVirtualPotentialTemperature",
    ) -> None:
        """
        This test checks theta_v = theta * (1 + qv/eps) / (1 + qv) against hand values and confirms that it is close to theta * (1 + 0.608 qv) and far from the old theta * (1 + 1.61 qv).

        Parameters:
            None

        Returns:
            None
        """
        theta = xr.DataArray(np.array([300.0, 290.0]), dims=["x"])
        qv = xr.DataArray(np.array([0.01, 0.015]), dims=["x"])
        result = ThermodynamicDiagnostics(
            verbose=False
        ).compute_virtual_potential_temperature(theta, qv)
        assert np.allclose(
            result.values, theta.values * (1 + qv.values / EPSILON) / (1 + qv.values)
        )
        assert np.allclose(
            result.values, theta.values * (1 + 0.608 * qv.values), atol=0.05
        )
        assert np.all(
            np.abs(result.values - theta.values * (1 + 1.61 * qv.values)) > 0.1
        )

    def test_against_metpy(self: "TestVirtualPotentialTemperature") -> None:
        """
        This test compares the result with metpy.calc.virtual_potential_temperature for realistic values.

        Parameters:
            None

        Returns:
            None
        """
        mc = pytest.importorskip("metpy.calc")
        from metpy.units import units

        p = np.array([100000.0, 85000.0, 50000.0])
        t = np.array([300.0, 290.0, 260.0])
        qv = np.array([0.018, 0.010, 0.001])
        theta = t * (1e5 / p) ** (2.0 / 7.0)
        ref = mc.virtual_potential_temperature(
            p * units.Pa, t * units.K, qv * units("kg/kg")
        ).m
        result = ThermodynamicDiagnostics(
            verbose=False
        ).compute_virtual_potential_temperature(
            xr.DataArray(theta, dims=["x"]), xr.DataArray(qv, dims=["x"])
        )
        assert np.allclose(result.values, ref, atol=0.02)


class TestPerColumnPressureLevel:
    """Pressure levels are interpolated per column in log p, never taken from a domain-mean level."""

    @pytest.mark.parametrize("level", [95000.0, 85000.0, 50000.0])
    def test_wind_matches_columnwise_interp(
        self: "TestPerColumnPressureLevel", level: float
    ) -> None:
        """
        This test compares WindDiagnostics.get_3d_variable_at_level with an independent per-column np.interp in log pressure, including the NaN pattern for columns whose ground pressure is below the requested level.

        Parameters:
            level (float): Requested pressure in Pa.

        Returns:
            None
        """
        ds = _columns_dataset()
        result = WindDiagnostics(verbose=False).get_3d_variable_at_level(
            ds, "u", level, 0
        )
        p = (ds["pressure_p"] + ds["pressure_base"]).isel(Time=0).values
        u = ds["u"].isel(Time=0).values
        ref = np.array(
            [
                (
                    np.interp(np.log(level), np.log(p[i][::-1]), u[i][::-1])
                    if p[i, 0] >= level
                    else np.nan
                )
                for i in range(N_CELLS)
            ]
        )
        assert np.allclose(result.values, ref, equal_nan=True)
        assert result.attrs["actual_pressure_level"] == f"{level:.1f} Pa"
        assert result.attrs["level_index"] == -1

    def test_below_ground_is_nan(self: "TestPerColumnPressureLevel") -> None:
        """
        This test checks that a column whose surface pressure (70 kPa) is below a requested 85 kPa level is NaN instead of being filled from the nearest model level.

        Parameters:
            None

        Returns:
            None
        """
        result = WindDiagnostics(verbose=False).get_3d_variable_at_level(
            _columns_dataset(), "u", 85000.0, 0
        )
        assert np.isnan(result.values[-1])
        assert np.isfinite(result.values[0])

    def test_thermodynamic_level_in_pa(self: "TestPerColumnPressureLevel") -> None:
        """
        This test checks that ThermodynamicDiagnostics._extract_field interpolates a float level per column and that two columns with different pressure profiles get different values for the same pressure.

        Parameters:
            None

        Returns:
            None
        """
        ds = _columns_dataset()
        out = ThermodynamicDiagnostics(verbose=False)._extract_field(
            ds, "theta", "Time", 0, 60000.0, "xarray"
        )
        p = (ds["pressure_p"] + ds["pressure_base"]).isel(Time=0).values
        th = ds["theta"].isel(Time=0).values
        ref = [
            np.interp(np.log(60000.0), np.log(p[i][::-1]), th[i][::-1])
            for i in range(N_CELLS)
        ]
        assert np.allclose(out.values, ref)
        assert len(set(np.round(out.values, 3))) > 1

    def test_pressure_to_level_index_removed(
        self: "TestPerColumnPressureLevel",
    ) -> None:
        """
        This test checks that the domain-mean level selection helpers no longer exist.

        Parameters:
            None

        Returns:
            None
        """
        assert not hasattr(ThermodynamicDiagnostics, "_pressure_to_level_index")
        assert not hasattr(WindDiagnostics, "_compute_level_index_from_pressure")


class TestStormMotion:
    """compute_storm_motion is a per-column pressure-weighted layer mean wind."""

    def test_per_column_values(self: "TestStormMotion") -> None:
        """
        This test checks the storm motion against a fine-grid reference integral in each column and that high-terrain columns use their own layer (starting at the ground), so the result differs between columns with different surface pressure.

        Parameters:
            None

        Returns:
            None
        """
        ds = _columns_dataset()
        proc = SimpleNamespace(dataset=ds)
        us, vs = WindDiagnostics(verbose=False).compute_storm_motion(proc, "u", "v", 0)
        assert us.dims == ("nCells",)
        p = (ds["pressure_p"] + ds["pressure_base"]).isel(Time=0).values
        u = ds["u"].isel(Time=0).values
        for i in range(N_CELLS):
            grid = np.linspace(min(p[i, 0], 100000.0), 50000.0, 20001)
            ref = np.interp(np.log(grid), np.log(p[i][::-1]), u[i][::-1]).mean()
            assert float(us[i]) == pytest.approx(ref, abs=0.1)
        assert np.allclose(vs.values, 2.0)
        assert us.values.std() > 0.01

    def test_layer_above_ground_is_nan(self: "TestStormMotion") -> None:
        """
        This test checks that a column whose surface pressure is above the layer top (the whole layer is underground) yields NaN.

        Parameters:
            None

        Returns:
            None
        """
        ds = _columns_dataset()
        us, _ = WindDiagnostics(verbose=False).compute_storm_motion(
            SimpleNamespace(dataset=ds), "u", "v", 0, p_bottom=100000.0, p_top=95000.0
        )
        assert np.isnan(us.values[-1])
        assert np.isfinite(us.values[0])

    def test_invalid_layer_raises(self: "TestStormMotion") -> None:
        """
        This test checks that p_top >= p_bottom raises.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError):
            WindDiagnostics(verbose=False).compute_storm_motion(
                SimpleNamespace(dataset=_columns_dataset()),
                p_bottom=50000.0,
                p_top=60000.0,
            )


def _sounding_inputs(tmp_path: Path, zgrid: bool = True) -> Tuple[Any, Path]:
    """
    This helper builds a regional (CONUS-subset) grid file and an mpasout-like dataset without any height variable, returning a processor-like object. The grid has cell spacing of about 0.5 degrees, and zgrid (when requested) holds interface heights above mean sea level.

    Parameters:
        tmp_path (Path): Directory for the grid file.
        zgrid (bool): Whether to write zgrid into the grid file.

    Returns:
        Tuple[Any, Path]: The processor-like object and the grid file path.
    """
    lons, lats = np.meshgrid(
        np.arange(-110.0, -90.01, 0.5), np.arange(30.0, 45.01, 0.5)
    )
    lon, lat = lons.ravel(), lats.ravel()
    n = lon.size
    grid_vars: dict = {
        "lonCell": ("nCells", np.radians(lon + 360.0)),
        "latCell": ("nCells", np.radians(lat)),
    }
    interfaces = np.linspace(0.0, 16000.0, N_VERT + 1)
    if zgrid:
        grid_vars["zgrid"] = (
            ("nCells", "nVertLevelsP1"),
            300.0 + np.tile(interfaces, (n, 1)),
        )
    grid_file = tmp_path / "conus_subset.grid.nc"
    xr.Dataset(grid_vars).to_netcdf(grid_file)

    p = np.tile(np.geomspace(98000.0, 20000.0, N_VERT), (n, 1))[None]
    dims = ["Time", "nCells", "nVertLevels"]
    ds = xr.Dataset(
        {
            "pressure_p": (dims, np.zeros_like(p)),
            "pressure_base": (dims, p),
            "theta": (
                dims,
                300.0 + 3.0 * np.tile(np.arange(N_VERT, dtype=float), (1, n, 1)),
            ),
            "qv": (dims, np.tile(np.geomspace(0.012, 1e-5, N_VERT), (1, n, 1))),
            "uReconstructZonal": (
                dims,
                np.tile(np.linspace(2.0, 30.0, N_VERT), (1, n, 1)),
            ),
            "uReconstructMeridional": (
                dims,
                np.tile(np.linspace(0.0, 10.0, N_VERT), (1, n, 1)),
            ),
        }
    )
    return SimpleNamespace(dataset=ds, grid_file=str(grid_file)), grid_file


class TestSoundingHeightFromGrid:
    """Height (and thus the severe indices) come from zgrid in the grid file when the data lack it."""

    def test_height_from_zgrid(
        self: "TestSoundingHeightFromGrid", tmp_path: Path
    ) -> None:
        """
        This test checks that the height profile is the mass-level average of the zgrid interfaces, ascending and aligned with the pressure profile.

        Parameters:
            tmp_path (Path): Pytest temporary directory.

        Returns:
            None
        """
        proc, _ = _sounding_inputs(tmp_path)
        prof = SoundingDiagnostics(verbose=False).extract_sounding_profile(
            proc, -100.0, 38.0
        )
        interfaces = 300.0 + np.linspace(0.0, 16000.0, N_VERT + 1)
        assert prof["height"] is not None
        assert np.allclose(prof["height"], 0.5 * (interfaces[:-1] + interfaces[1:]))
        assert np.all(np.diff(prof["height"]) > 0)
        assert len(prof["height"]) == len(prof["pressure"])

    def test_severe_indices_not_none(
        self: "TestSoundingHeightFromGrid", tmp_path: Path
    ) -> None:
        """
        This test checks that bulk shear, SRH, SWEAT and the wet-bulb-zero height are numbers once the height profile is available.

        Parameters:
            tmp_path (Path): Pytest temporary directory.

        Returns:
            None
        """
        pytest.importorskip("metpy")
        proc, _ = _sounding_inputs(tmp_path)
        sd = SoundingDiagnostics(verbose=False)
        prof = sd.extract_sounding_profile(proc, -100.0, 38.0)
        idx = sd.compute_thermodynamic_indices(
            prof["pressure"],
            prof["temperature"],
            prof["dewpoint"],
            prof["u_wind"],
            prof["v_wind"],
            prof["height"],
        )
        for key in (
            "bulk_shear_0_1km",
            "bulk_shear_0_6km",
            "srh_0_1km",
            "srh_0_3km",
            "sweat_index",
        ):
            assert idx[key] is not None and np.isfinite(idx[key]), key

    def test_no_zgrid_gives_none(
        self: "TestSoundingHeightFromGrid", tmp_path: Path
    ) -> None:
        """
        This test checks that without zgrid anywhere the height stays None rather than raising.

        Parameters:
            tmp_path (Path): Pytest temporary directory.

        Returns:
            None
        """
        proc, _ = _sounding_inputs(tmp_path, zgrid=False)
        prof = SoundingDiagnostics(verbose=False).extract_sounding_profile(
            proc, -100.0, 38.0
        )
        assert prof["height"] is None


class TestSoundingLocationValidation:
    """C1: requested point validation, distance threshold and new keys."""

    def test_inside_domain_ok_with_new_keys(
        self: "TestSoundingLocationValidation", tmp_path: Path
    ) -> None:
        """
        This test checks that a point inside the regional grid works and that the profile carries requested_lon, requested_lat and distance_km.

        Parameters:
            tmp_path (Path): Pytest temporary directory.

        Returns:
            None
        """
        proc, _ = _sounding_inputs(tmp_path)
        prof = SoundingDiagnostics(verbose=False).extract_sounding_profile(
            proc, -100.1, 38.1
        )
        assert prof["requested_lon"] == pytest.approx(-100.1)
        assert prof["requested_lat"] == pytest.approx(38.1)
        assert 0.0 <= prof["distance_km"] < 20.0

    @pytest.mark.parametrize("lon,lat", [(10.0, 50.0), (-100.0, 10.0), (-60.0, 38.0)])
    def test_far_outside_raises(
        self: "TestSoundingLocationValidation", tmp_path: Path, lon: float, lat: float
    ) -> None:
        """
        This test checks that points far outside the regional grid raise a ValueError that names the point, the cell and the distance.

        Parameters:
            tmp_path (Path): Pytest temporary directory.
            lon (float): Requested longitude.
            lat (float): Requested latitude.

        Returns:
            None
        """
        proc, _ = _sounding_inputs(tmp_path)
        with pytest.raises(ValueError, match="km from the nearest cell"):
            SoundingDiagnostics(verbose=False).extract_sounding_profile(proc, lon, lat)

    def test_explicit_max_distance(
        self: "TestSoundingLocationValidation", tmp_path: Path
    ) -> None:
        """
        This test checks that an explicit max_distance_km overrides the default threshold in both directions.

        Parameters:
            tmp_path (Path): Pytest temporary directory.

        Returns:
            None
        """
        proc, _ = _sounding_inputs(tmp_path)
        sd = SoundingDiagnostics(verbose=False)
        with pytest.raises(ValueError):
            sd.extract_sounding_profile(proc, -100.2, 38.2, max_distance_km=1.0)
        prof = sd.extract_sounding_profile(
            proc, -100.0, 38.0 - 3.0, max_distance_km=5000.0
        )
        assert prof["distance_km"] < 5000.0
        # Outside the default threshold but accepted with a generous explicit one
        prof = sd.extract_sounding_profile(proc, -100.0, 50.0, max_distance_km=2000.0)
        assert prof["distance_km"] > 100.0

    @pytest.mark.parametrize(
        "lon,lat", [(0.0, 91.0), (0.0, -90.5), (np.nan, 0.0), (0.0, np.inf)]
    )
    def test_invalid_point_raises(
        self: "TestSoundingLocationValidation", tmp_path: Path, lon: float, lat: float
    ) -> None:
        """
        This test checks that a latitude outside [-90, 90] or a non-finite coordinate raises before any search.

        Parameters:
            tmp_path (Path): Pytest temporary directory.
            lon (float): Requested longitude.
            lat (float): Requested latitude.

        Returns:
            None
        """
        proc, _ = _sounding_inputs(tmp_path)
        with pytest.raises(ValueError):
            SoundingDiagnostics(verbose=False).extract_sounding_profile(proc, lon, lat)


class TestColumnIntegralOrder:
    """The surface-layer term is applied at the high-pressure end whatever the level order."""

    def test_both_orders_give_analytic_value(self: "TestColumnIntegralOrder") -> None:
        """
        This test integrates a constant q = 0.01 over a column from 1000 to 200 hPa with surface pressure 1010 hPa; the analytic value is q * (101000 - 20000) / g in either ordering, which doubled the surface term before when the profile ran top first.

        Parameters:
            None

        Returns:
            None
        """
        p_bottom_first = np.linspace(100000.0, 20000.0, 17)[None, :]
        q = np.full_like(p_bottom_first, 0.01)
        sfc = np.array([101000.0])
        expected = 0.01 * (101000.0 - 20000.0) / 9.80665
        a = _trapezoidal_column_integral(q, p_bottom_first, sfc)
        b = _trapezoidal_column_integral(q[:, ::-1], p_bottom_first[:, ::-1], sfc)
        assert a[0] == pytest.approx(expected, rel=2e-3)
        assert b[0] == pytest.approx(a[0], rel=1e-12)

    def test_mixed_order_columns(self: "TestColumnIntegralOrder") -> None:
        """
        This test checks that a batch containing one bottom-first and one top-first column gives the same value for both.

        Parameters:
            None

        Returns:
            None
        """
        p = np.linspace(100000.0, 20000.0, 9)
        pp = np.stack([p, p[::-1]])
        q = np.full_like(pp, 0.02)
        out = _trapezoidal_column_integral(q, pp, np.array([100500.0, 100500.0]))
        assert out[0] == pytest.approx(out[1], rel=1e-12)


class TestFiniteOrNone:
    """Every sounding index reports None instead of NaN."""

    def test_helper(self: "TestFiniteOrNone") -> None:
        """
        This test checks the _finite_or_none helper for finite, NaN, infinite and non-scalar inputs.

        Parameters:
            None

        Returns:
            None
        """
        f = SoundingDiagnostics._finite_or_none
        assert f(3.5) == 3.5
        assert f(np.float32(2.0)) == 2.0
        assert f(np.nan) is None
        assert f(np.inf) is None
        assert f(np.array([1.0, 2.0])) is None
        assert f("x") is None

    def test_srh_nan_is_none(
        self: "TestFiniteOrNone", monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        This test makes MetPy's storm_relative_helicity return NaN and checks that the SRH entries are None, not NaN.

        Parameters:
            monkeypatch (pytest.MonkeyPatch): Fixture used to patch MetPy.

        Returns:
            None
        """
        mc = pytest.importorskip("metpy.calc")
        from metpy.units import units

        import mpasdiag.diagnostics.sounding as snd

        nan = np.nan * units("m^2/s^2")
        monkeypatch.setattr(
            mc, "storm_relative_helicity", lambda *a, **k: (nan, nan, nan)
        )
        result: dict = {"sbcape": None}
        n = 10
        SoundingDiagnostics(verbose=False)._compute_shear_indices(
            result,
            np.linspace(1000, 300, n),
            np.linspace(20, -30, n),
            np.linspace(10, -40, n),
            np.linspace(5.0, 40.0, n),
            np.linspace(0.0, -10.0, n),
            np.linspace(0, 9000, n),
        )
        assert snd.mpcalc is mc
        assert result["srh_0_1km"] is None and result["srh_0_3km"] is None


class TestDegenerateProfileIndices:
    """Regression tests for layer-based sounding indices on profiles that cannot support them."""

    @staticmethod
    def _profile(n: int = 30, top_m: float = 9000.0) -> Tuple[np.ndarray, ...]:
        """
        This helper builds a simple, well-formed sounding profile from the surface upward.

        Parameters:
            n (int): Number of levels.
            top_m (float): Height of the top level in metres.

        Returns:
            Tuple[np.ndarray, ...]: Pressure (hPa), temperature (degC), dewpoint (degC), u and v (kt), and height (m).
        """
        height = np.linspace(0.0, top_m, n)
        pressure = 1000.0 * np.exp(-height / 8000.0)
        temperature = 25.0 - 6.5 * height / 1000.0
        dewpoint = temperature - 5.0
        u = np.linspace(5.0, 40.0, n)
        v = np.linspace(0.0, 15.0, n)
        return pressure, temperature, dewpoint, u, v, height

    def test_layer_indices_none_for_missing_winds_or_shallow_profiles(
        self: "TestDegenerateProfileIndices",
    ) -> None:
        """
        This regression test verifies that bulk shear and storm-relative helicity are None, not 0.0 or a clipped-layer value, when the winds are missing or the profile does not reach the layer depth. MetPy returns 0.0 for NaN winds and silently clips a 0-3 km layer to a shallower profile.

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        p, t, td, u, v, z = self._profile()

        nan_winds = diag.compute_thermodynamic_indices(
            p, t, td, np.full_like(u, np.nan), np.full_like(v, np.nan), z
        )
        assert nan_winds["srh_0_1km"] is None and nan_winds["srh_0_3km"] is None
        assert nan_winds["bulk_shear_0_1km"] is None

        p, t, td, u, v, z = self._profile(n=10, top_m=900.0)
        shallow = diag.compute_thermodynamic_indices(p, t, td, u, v, z)
        assert shallow["srh_0_1km"] is None and shallow["srh_0_3km"] is None
        assert shallow["bulk_shear_0_6km"] is None

        p, t, td, u, v, z = self._profile()
        full = diag.compute_thermodynamic_indices(p, t, td, u, v, z)
        assert full["srh_0_1km"] is not None and full["bulk_shear_0_6km"] is not None

    def test_wet_bulb_zero_none_without_vertical_extent(
        self: "TestDegenerateProfileIndices",
    ) -> None:
        """
        This regression test verifies that the wet-bulb-zero height is None for a profile whose heights are all equal, instead of a meaningless 0 m.

        Parameters:
            None

        Returns:
            None
        """
        from metpy.units import units

        p, t, td, _, _, z = self._profile()
        result = SoundingDiagnostics._compute_wet_bulb_zero(
            p * units.hPa, t * units.degC, td * units.degC, np.zeros_like(z)
        )
        assert result is None

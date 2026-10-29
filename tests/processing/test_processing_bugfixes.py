#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Test Suite: Processing Layer Regression Tests

This module contains regression tests for bugs fixed in the data-processing layer: Time coordinates for multi-record files, per-column log-pressure interpolation, data cache identity and eviction, unit display fallbacks, longitude-frame reconciliation and great-circle nearest-neighbour remapping, target-grid bounds, radians-versus-degrees detection, and file discovery. Every test uses small synthetic datasets so that no real MPAS output is required.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from datetime import datetime
from pathlib import Path
from typing import Any, List
from unittest.mock import patch

from mpasdiag.processing.base import MPASBaseProcessor
from mpasdiag.processing.data_cache import MPASDataCache
from mpasdiag.processing.processors_3d import MPAS3DProcessor
from mpasdiag.processing.remapping import (
    _build_target_axis,
    build_remapped_valid_mask,
    remap_mpas_to_latlon,
)
from mpasdiag.processing.utils_datetime import MPASDateTimeUtils
from mpasdiag.processing.utils_file import FileManager
from mpasdiag.processing.utils_geog import MPASGeographicUtils
from mpasdiag.processing.utils_unit import UnitConverter


def _base_proc() -> MPASBaseProcessor:
    """
    This helper creates a minimal MPASBaseProcessor without running its constructor, which is sufficient for exercising the file-loading helpers that only need the verbose flag and the grid file attribute.

    Parameters:
        None

    Returns:
        MPASBaseProcessor: A bare processor instance.
    """
    proc = MPASBaseProcessor.__new__(MPASBaseProcessor)
    proc.verbose = False
    proc.dataset = None
    proc.data_type = "xarray"
    proc.grid_file = "unused.nc"
    return proc


def _write_multi_record_files(tmp_path: Path) -> List[str]:
    """
    This helper writes two NetCDF files that each hold two Time records together with an MPAS-style 'xtime' character variable, mimicking output streams that store several times per file.

    Parameters:
        tmp_path (Path): Directory in which to write the files.

    Returns:
        List[str]: Paths of the two files in chronological order.
    """
    paths = []
    stamps = [
        ["2024-01-01_00:00:00", "2024-01-01_01:00:00"],
        ["2024-01-01_02:00:00", "2024-01-01_03:00:00"],
    ]

    for name, pair in zip(
        ("diag.2024-01-01_00.00.00.nc", "diag.2024-01-01_02.00.00.nc"), stamps
    ):
        ds = xr.Dataset(
            {
                "t2m": (("Time", "nCells"), np.arange(8, dtype="f4").reshape(2, 4)),
                "xtime": (("Time",), np.array([s.encode() for s in pair], dtype="S64")),
            }
        )
        path = str(tmp_path / name)
        ds.to_netcdf(path)
        paths.append(path)

    return paths


class TestMultiRecordTimeCoordinate:
    """Tests that files holding several Time records get one correct timestamp per record and are never silently dropped."""

    def test_multi_record_files_get_per_record_times(
        self: "TestMultiRecordTimeCoordinate", tmp_path: Path
    ) -> None:
        """
        This test verifies that two files with two Time records each load into a four-step dataset whose Time coordinate is read from the per-record xtime values, rather than failing on a per-file timestamp length mismatch.

        Parameters:
            tmp_path (Path): Pytest temporary directory.

        Returns:
            None
        """
        files = _write_multi_record_files(tmp_path)
        proc = _base_proc()
        file_dts = MPASDateTimeUtils.parse_file_datetimes(files, False)

        ds = proc._load_multifile_dataset(files, file_dts, {"Time": 1})

        assert ds.sizes["Time"] == 4
        assert list(pd.to_datetime(ds.Time.values).hour) == [0, 1, 2, 3]

    def test_multi_record_without_xtime_raises(
        self: "TestMultiRecordTimeCoordinate", tmp_path: Path
    ) -> None:
        """
        This test verifies that a multi-record file lacking an 'xtime' variable raises a ValueError naming the file instead of mislabeling its records.

        Parameters:
            tmp_path (Path): Pytest temporary directory.

        Returns:
            None
        """
        path = str(tmp_path / "diag.2024-01-01_00.00.00.nc")
        xr.Dataset({"t2m": (("Time", "nCells"), np.zeros((2, 3)))}).to_netcdf(path)
        proc = _base_proc()

        with pytest.raises(ValueError, match="diag.2024-01-01_00.00.00.nc"):
            proc._load_multifile_dataset([path], [datetime(2024, 1, 1)], {"Time": 1})

    def test_unparseable_name_uses_xtime_from_file(
        self: "TestMultiRecordTimeCoordinate", tmp_path: Path
    ) -> None:
        """
        This test verifies that a single multi-time file whose name has no timestamp still loads, with its time taken from the first xtime record, and that a file with neither a valid name nor xtime raises a ValueError.

        Parameters:
            tmp_path (Path): Pytest temporary directory.

        Returns:
            None
        """
        good = tmp_path / "history.nc"
        xr.Dataset(
            {
                "t2m": (("Time", "nCells"), np.zeros((1, 3))),
                "xtime": (("Time",), np.array([b"2024-05-06_07:00:00"], dtype="S64")),
            }
        ).to_netcdf(good)
        bad = tmp_path / "other.nc"
        xr.Dataset({"t2m": (("Time", "nCells"), np.zeros((1, 3)))}).to_netcdf(bad)
        proc = _base_proc()

        assert proc._resolve_file_datetimes([str(good)]) == [datetime(2024, 5, 6, 7)]

        with pytest.raises(ValueError, match="other.nc"):
            proc._resolve_file_datetimes([str(bad)])

    def test_uxarray_incompatibility_falls_back_to_pure_xarray(
        self: "TestMultiRecordTimeCoordinate", tmp_path: Path
    ) -> None:
        """
        This test verifies that when the installed uxarray is detected as unusable the primary loader returns a plain xarray dataset directly, without first attempting an UXarray load that can only fail.

        Parameters:
            tmp_path (Path): Pytest temporary directory.

        Returns:
            None
        """
        files = _write_multi_record_files(tmp_path)
        proc = _base_proc()
        file_dts = MPASDateTimeUtils.parse_file_datetimes(files, False)

        with patch("mpasdiag.processing.base._uxarray_is_usable", return_value=False):
            ds, kind = proc._attempt_primary_load(
                files, file_dts, {"Time": 1}, None, False, "2D"
            )

        assert kind == "xarray"
        assert isinstance(ds, xr.Dataset)


def _pressure_dataset(n_cells: int = 3, n_vert: int = 5) -> xr.Dataset:
    """
    This helper builds a small 3D dataset in which every column has a different surface pressure, plus a mass-level field and an interface-level field, so per-column interpolation can be compared with a reference.

    Parameters:
        n_cells (int): Number of cells.
        n_vert (int): Number of mass levels.

    Returns:
        xr.Dataset: Dataset with theta, w, pressure_p and pressure_base.
    """
    surface = np.linspace(100000.0, 80000.0, n_cells)
    frac = np.linspace(1.0, 0.2, n_vert)
    total = surface[:, None] * frac[None, :]
    theta = 300.0 + 10.0 * np.arange(n_vert)[None, :] + 0.0 * surface[:, None]
    w = np.arange(n_vert + 1, dtype=float)[None, :] + 0.0 * surface[:, None]

    return xr.Dataset(
        {
            "theta": (("Time", "nCells", "nVertLevels"), theta[None], {"units": "K"}),
            "w": (("Time", "nCells", "nVertLevelsP1"), w[None], {"units": "m/s"}),
            "pressure_p": (
                ("Time", "nCells", "nVertLevels"),
                np.zeros((1, n_cells, n_vert)),
            ),
            "pressure_base": (("Time", "nCells", "nVertLevels"), total[None]),
        },
        coords={"Time": pd.date_range("2024-01-01", periods=1)},
    )


def _proc_3d(ds: xr.Dataset) -> MPAS3DProcessor:
    """
    This helper creates an MPAS3DProcessor without running its constructor and attaches the given dataset.

    Parameters:
        ds (xr.Dataset): Dataset to attach.

    Returns:
        MPAS3DProcessor: Processor ready for get_3d_variable_data calls.
    """
    proc = MPAS3DProcessor.__new__(MPAS3DProcessor)
    proc.verbose = False
    proc.dataset = ds
    proc.data_type = "xarray"
    proc.grid_file = "unused.nc"
    return proc


class TestPressureLevelInterpolation:
    """Tests for per-column log-pressure interpolation to a float pressure level."""

    def test_matches_per_column_log_p_reference(
        self: "TestPressureLevelInterpolation",
    ) -> None:
        """
        This test verifies that interpolating to 70000 Pa matches a per-column np.interp in log pressure, that columns whose lowest level is above 70000 Pa are NaN instead of clamped, and that the level attributes are set.

        Parameters:
            None

        Returns:
            None
        """
        ds = _pressure_dataset()
        proc = _proc_3d(ds)
        level = 70000.0
        result = proc.get_3d_variable_data("theta", level, 0)

        p = (ds.pressure_p + ds.pressure_base).isel(Time=0).values
        th = ds.theta.isel(Time=0).values

        for i in range(p.shape[0]):
            if p[i, 0] >= level >= p[i, -1]:
                expected = np.interp(np.log(level), np.log(p[i, ::-1]), th[i, ::-1])
                assert result.values[i] == pytest.approx(expected, rel=1e-6)
            else:
                assert np.isnan(result.values[i])

        assert result.attrs["selected_level"] == level
        assert result.attrs["actual_pressure_level"] == "70000.0 Pa"
        assert result.attrs["units"] == "K"

    def test_outside_range_is_nan_not_clamped(
        self: "TestPressureLevelInterpolation",
    ) -> None:
        """
        This test verifies that a pressure above every column's surface pressure and a pressure below every column's top are never clamped to the surface or top level: since no column could give a value, they are reported as errors instead of all-NaN fields (which used to be written as empty plots).

        Parameters:
            None

        Returns:
            None
        """
        proc = _proc_3d(_pressure_dataset())

        for level in (120000.0, 5000.0):
            with pytest.raises(ValueError, match="outside the model's pressure range"):
                proc.get_3d_variable_data("theta", level, 0)

    def test_interface_variable_is_destaggered(
        self: "TestPressureLevelInterpolation",
    ) -> None:
        """
        This test verifies that a variable on nVertLevelsP1 is averaged to mass levels before interpolation: w equals the interface index, so mass level k has the value k + 0.5 and interpolating to a mass-level pressure returns exactly that.

        Parameters:
            None

        Returns:
            None
        """
        ds = _pressure_dataset()
        proc = _proc_3d(ds)
        p_mid = float((ds.pressure_base.isel(Time=0, nCells=0, nVertLevels=2)))
        result = proc.get_3d_variable_data("w", p_mid, 0)

        assert result.values[0] == pytest.approx(2.5)

    def test_hpa_value_raises_helpful_error(
        self: "TestPressureLevelInterpolation",
    ) -> None:
        """
        This test verifies that a float such as 850.0, which is implausible as a pressure in Pa, raises a ValueError that suggests 85000 Pa instead of returning the top level.

        Parameters:
            None

        Returns:
            None
        """
        proc = _proc_3d(_pressure_dataset())

        with pytest.raises(ValueError, match="85000"):
            proc.get_3d_variable_data("theta", 850.0, 0)

    def test_numpy_integer_levels_accepted_and_bool_rejected(
        self: "TestPressureLevelInterpolation",
    ) -> None:
        """
        This test verifies that NumPy integer scalars select a model level like Python ints, that numpy floats are treated as pressures, and that booleans are rejected as invalid level specifications.

        Parameters:
            None

        Returns:
            None
        """
        proc = _proc_3d(_pressure_dataset())

        by_int = proc.get_3d_variable_data("theta", 2, 0)
        by_np = proc.get_3d_variable_data("theta", np.int64(2), 0)
        assert np.array_equal(by_int.values, by_np.values)

        assert (
            proc.get_3d_variable_data("theta", np.float64(70000.0), 0).attrs[
                "level_index"
            ]
            == -1
        )

        with pytest.raises(ValueError):
            proc.get_3d_variable_data("theta", True, 0)

    def test_index_bounds_follow_variable_staggering(
        self: "TestPressureLevelInterpolation",
    ) -> None:
        """
        This test verifies that integer and 'top' levels are validated against the variable's own vertical dimension, so the last interface of an nVertLevelsP1 variable is reachable.

        Parameters:
            None

        Returns:
            None
        """
        proc = _proc_3d(_pressure_dataset(n_vert=5))

        assert proc.get_3d_variable_data("w", 5, 0).values[0] == 5.0
        assert proc.get_3d_variable_data("w", "top", 0).values[0] == 5.0

        with pytest.raises(ValueError):
            proc.get_3d_variable_data("theta", 5, 0)

    def test_pad_p1_levels_uses_interfaces_not_invented_value(
        self: "TestPressureLevelInterpolation",
    ) -> None:
        """
        This test verifies that converting n mass-level pressures to n+1 interface pressures yields a monotonic profile bracketing the mass levels, with the interior values at log-midpoints and no 0.9-times-last invention.

        Parameters:
            None

        Returns:
            None
        """
        levels = np.array([100000.0, 80000.0, 60000.0, 40000.0])
        out = MPAS3DProcessor._pad_p1_levels(levels, 5, "nVertLevelsP1")

        assert len(out) == 5
        assert np.all(np.diff(out) < 0)
        assert out[0] > levels[0] and out[-1] < levels[-1]
        assert out[2] == pytest.approx(np.sqrt(levels[1] * levels[2]))
        assert out[-1] != pytest.approx(levels[-1] * 0.9)
        assert MPAS3DProcessor._pad_p1_levels(levels, 4, "nVertLevels") is levels

    def test_numpy_extraction_uses_vertical_last_axis(
        self: "TestPressureLevelInterpolation",
    ) -> None:
        """
        This test verifies that extract_2d_from_3d on a (Time, nCells, nVertLevels) array returns the (Time, nCells) slice at the requested level.

        Parameters:
            None

        Returns:
            None
        """
        arr = np.arange(2 * 5 * 4).reshape(2, 5, 4)
        out = MPAS3DProcessor.extract_2d_from_3d(arr, level_index=1)

        assert out.shape == (2, 5)
        assert np.array_equal(out, arr[:, :, 1])


def _cell_dataset(lon: float = 10.0) -> xr.Dataset:
    """
    This helper builds a minimal dataset with lonCell, latCell and a 2D variable on three cells.

    Parameters:
        lon (float): Longitude of the first cell.

    Returns:
        xr.Dataset: The dataset.
    """
    return xr.Dataset(
        {
            "lonCell": ("nCells", np.array([lon, 20.0, 30.0])),
            "latCell": ("nCells", np.array([0.0, 1.0, 2.0])),
            "t2m": (("Time", "nCells"), np.array([[1.0, 2.0, 3.0]])),
        }
    )


class TestDataCacheIdentity:
    """Tests that the data cache ties entries to their source dataset, evicts correctly, and protects its arrays."""

    def test_different_dataset_does_not_return_stale_data(
        self: "TestDataCacheIdentity",
    ) -> None:
        """
        This test verifies that loading the same variable and indices from a second dataset replaces the cached array, so a subsequent get returns the second dataset's values.

        Parameters:
            None

        Returns:
            None
        """
        a, b = _cell_dataset(), _cell_dataset()
        b["t2m"] = b["t2m"] + 100.0
        cache = MPASDataCache()
        cache.load_variable_data(a, "t2m", 0)
        cache.load_variable_data(b, "t2m", 0)

        assert cache.get_variable_data("t2m", 0).data[0] == 101.0

    def test_different_dataset_replaces_coordinates(
        self: "TestDataCacheIdentity",
    ) -> None:
        """
        This test verifies that coordinates cached for one dataset are replaced, not reused, when loaded from a different mesh under the same key.

        Parameters:
            None

        Returns:
            None
        """
        cache = MPASDataCache()
        cache.load_coordinates_from_dataset(_cell_dataset(10.0))
        other = xr.Dataset(
            {
                "lonCell": ("nCells", np.array([1.0, 2.0])),
                "latCell": ("nCells", np.array([0.0, 1.0])),
            }
        )
        cache.load_coordinates_from_dataset(other)

        assert len(cache.get_coordinates()[0]) == 2

    def test_max_variables_is_enforced_with_coordinates_present(
        self: "TestDataCacheIdentity",
    ) -> None:
        """
        This test verifies that with coordinates cached and heavily accessed, loading more variables than max_variables still evicts a variable (never a coordinate key) and keeps the variable count at the limit.

        Parameters:
            None

        Returns:
            None
        """
        ds = _cell_dataset()
        ds["a"] = ("nCells", np.zeros(3))
        ds["b"] = ("nCells", np.ones(3))
        ds["c"] = ("nCells", np.ones(3))
        cache = MPASDataCache(max_variables=2)
        cache.load_coordinates_from_dataset(ds)

        for _ in range(3):
            cache.get_coordinates()

        cache.load_variable_data(ds, "a")
        cache.load_variable_data(ds, "b")
        cache.get_variable_data("b")
        cache.load_variable_data(ds, "c")

        assert len(cache._variables) == 2
        assert "a" not in cache._variables
        assert len(cache._coordinates) == 1

    def test_returned_arrays_are_copies(self: "TestDataCacheIdentity") -> None:
        """
        This test verifies that mutating arrays returned by get_coordinates or get_variable_data does not alter the cached values, and that the internal arrays are read-only.

        Parameters:
            None

        Returns:
            None
        """
        ds = _cell_dataset()
        cache = MPASDataCache()
        cache.load_coordinates_from_dataset(ds)
        cache.load_variable_data(ds, "t2m", 0)

        # Default getters hand out the cached read-only arrays (no copy); writes must fail loudly
        ro_lon, _ = cache.get_coordinates()
        assert not ro_lon.flags.writeable

        with pytest.raises(ValueError):
            ro_lon[:] = 0.0

        # writable=True gives independent copies that can be mutated without corrupting the cache
        lon, _ = cache.get_coordinates(writable=True)
        lon[:] = 0.0
        var = cache.get_variable_data("t2m", 0, writable=True)
        var.data[:] = -1.0

        assert cache.get_coordinates()[0].max() == pytest.approx(30.0)
        assert cache.get_variable_data("t2m", 0).data.min() == 1.0
        assert not cache._variables["t2m_t0"].data.flags.writeable
        assert ds["t2m"].values.flags.writeable

    def test_degree_mesh_near_equator_not_converted(
        self: "TestDataCacheIdentity",
    ) -> None:
        """
        This test verifies that a degree-valued mesh near the equator (all |lat| below pi) is cached unchanged rather than being treated as radians.

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset(
            {
                "lonCell": ("nCells", np.array([1.0, 2.0, 3.0])),
                "latCell": ("nCells", np.array([-1.0, 0.5, 2.0])),
            }
        )
        cache = MPASDataCache()
        cache.load_coordinates_from_dataset(ds)

        assert np.allclose(cache.get_coordinates()[1], [-1.0, 0.5, 2.0])


class TestRadiansDegreesHelper:
    """Tests for the centralized radians-versus-degrees decision."""

    def test_units_attribute_is_authoritative(self: "TestRadiansDegreesHelper") -> None:
        """
        This test verifies that a 'radian' units attribute forces conversion and a 'degree' attribute prevents it, regardless of magnitude.

        Parameters:
            None

        Returns:
            None
        """
        conv = MPASGeographicUtils.coordinates_to_degrees
        lon, lat = conv([1.0], [0.5], {"units": "radians"}, {"units": "radians"})
        assert lon[0] == pytest.approx(np.degrees(1.0))
        assert lat[0] == pytest.approx(np.degrees(0.5))

        lon, lat = conv(
            [1.0], [0.5], {"units": "degrees_east"}, {"units": "degrees_north"}
        )
        assert lon[0] == 1.0 and lat[0] == 0.5

    def test_heuristic_requires_both_axes(self: "TestRadiansDegreesHelper") -> None:
        """
        This test verifies that without units the heuristic converts only when max|lat| <= pi/2 and max|lon| <= 2*pi, so a degree mesh with small latitudes but large longitudes stays in degrees, while a genuine radian mesh is converted.

        Parameters:
            None

        Returns:
            None
        """
        conv = MPASGeographicUtils.coordinates_to_degrees
        lon, _ = conv([100.0, 101.0], [1.0, 2.0])
        assert lon[0] == 100.0

        lon, lat = conv([3.0, 6.0], [-1.5, 1.5])
        assert lon[1] == pytest.approx(np.degrees(6.0))
        assert lat[1] == pytest.approx(np.degrees(1.5))

    def test_extract_spatial_coordinates_respects_units(
        self: "TestRadiansDegreesHelper",
    ) -> None:
        """
        This test verifies that extract_spatial_coordinates honours the units attribute of lonCell/latCell.

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset(
            {
                "lonCell": ("nCells", np.array([1.0, 2.0]), {"units": "degrees"}),
                "latCell": ("nCells", np.array([0.1, 0.2]), {"units": "degrees"}),
            }
        )
        lon, lat = MPASGeographicUtils.extract_spatial_coordinates(ds)

        assert np.allclose(lon, [1.0, 2.0]) and np.allclose(lat, [0.1, 0.2])


class TestUnitFixes:
    """Tests for display-unit fallbacks and DataArray unit attributes."""

    def test_display_unit_falls_back_to_current_when_no_conversion(
        self: "TestUnitFixes",
    ) -> None:
        """
        This test verifies that get_display_units returns the original unit when no conversion to the preferred unit exists (unitless t2m, precipitation in mm); re_cloud in m now converts to microns.

        Parameters:
            None

        Returns:
            None
        """
        assert UnitConverter.get_display_units("t2m", "") == ""
        assert UnitConverter.get_display_units("precipitation", "mm") == "mm"
        assert UnitConverter.get_display_units("re_cloud", "m") == "microns"
        assert UnitConverter.get_display_units("t2m", "K") == "°C"
        assert UnitConverter.get_display_units("pressure_p", "Pa") == "hPa"

    def test_convert_units_updates_dataarray_units_attribute(
        self: "TestUnitFixes",
    ) -> None:
        """
        This test verifies that converting a DataArray sets its units attribute to the target unit while preserving other attributes and leaving the input untouched.

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(np.array([300.0]), attrs={"units": "K", "long_name": "x"})
        out = UnitConverter.convert_units(da, "K", "degC")

        assert isinstance(out, xr.DataArray)
        assert out.attrs["units"] == "°C"
        assert out.attrs["long_name"] == "x"
        assert float(out[0]) == pytest.approx(26.85)
        assert da.attrs["units"] == "K"


class TestFileDiscovery:
    """Tests for diagnostic and mpasout file discovery."""

    def test_find_diag_files_uses_mpasout_glob_in_mpasout_subdir(
        self: "TestFileDiscovery", tmp_path: Path
    ) -> None:
        """
        This test verifies that when no diag files exist, the mpasout subdirectory is searched with the mpasout glob, so mpasout files are found there.

        Parameters:
            tmp_path (Path): Pytest temporary directory.

        Returns:
            None
        """
        sub = tmp_path / "mpasout"
        sub.mkdir()
        (sub / "mpasout.2024-01-01_00.00.00.nc").touch()

        found = FileManager._find_diag_files(str(tmp_path))
        assert [Path(f).name for f in found] == ["mpasout.2024-01-01_00.00.00.nc"]

    def test_recursive_discovery_sorts_by_timestamp(
        self: "TestFileDiscovery", tmp_path: Path
    ) -> None:
        """
        This test verifies that recursive custom-pattern discovery orders files by timestamp across subdirectories, not by full path, and accepts a single file.

        Parameters:
            tmp_path (Path): Pytest temporary directory.

        Returns:
            None
        """
        for sub, day in (("a", 2), ("b", 1)):
            (tmp_path / sub).mkdir()
            (tmp_path / sub / f"wiso.2024-01-0{day}_00.00.00.nc").touch()

        proc = _base_proc()
        files = proc._find_custom_pattern_files(str(tmp_path), "wiso.*.nc")

        assert [Path(f).parent.name for f in files] == ["b", "a"]


def _global_mesh(n: int = 4000, seed: int = 1) -> Any:
    """
    This helper generates quasi-uniform random points on the sphere (in degrees, longitude in [0, 360)).

    Parameters:
        n (int): Number of points.
        seed (int): Random seed.

    Returns:
        Tuple of (lon, lat) arrays in degrees.
    """
    rng = np.random.default_rng(seed)
    lon = rng.uniform(0.0, 360.0, n)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    return lon, lat


class TestRemappingFixes:
    """Tests for longitude-frame reconciliation, great-circle nearest search, and target-grid bounds."""

    def test_raw_0_360_source_with_minus180_180_target(
        self: "TestRemappingFixes",
    ) -> None:
        """
        This test verifies that a source mesh with longitudes in [0, 360] remaps correctly onto a [-180, 180] target: a smooth analytic field is reproduced in the western hemisphere with small error, instead of garbage.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _global_mesh()
        field = lambda lo, la: np.cos(np.radians(la)) * np.cos(
            np.radians(lo)
        )  # noqa: E731
        out = remap_mpas_to_latlon(
            field(lon, lat), lon, lat, -180, 180, -90, 90, 10.0, "nearest"
        )
        lo2d, la2d = np.meshgrid(out.lon.values, out.lat.values)

        assert np.nanmax(np.abs(out.values - field(lo2d, la2d))) < 0.2

    def test_nearest_is_great_circle_at_high_latitude(
        self: "TestRemappingFixes",
    ) -> None:
        """
        This test verifies that nearest-neighbour lookup picks the cell nearest on the sphere: for a target at 85 degrees latitude, the cell chosen matches a brute-force great-circle search, which a lon/lat Euclidean search does not guarantee.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _global_mesh(800, seed=3)
        data = np.arange(lon.size, dtype=float)
        out = remap_mpas_to_latlon(data, lon, lat, -180, 180, 80, 90, 5.0, "nearest")

        def xyz(lo: np.ndarray, la: np.ndarray) -> np.ndarray:
            lo, la = np.radians(lo), np.radians(la)
            return np.column_stack(
                [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
            )

        lo2d, la2d = np.meshgrid(out.lon.values, out.lat.values)
        tgt = xyz(lo2d.ravel(), la2d.ravel())
        expected = np.argmax(tgt @ xyz(lon, lat).T, axis=1).reshape(lo2d.shape)

        assert np.array_equal(out.values, expected.astype(float))

    def test_target_axis_never_exceeds_bounds(self: "TestRemappingFixes") -> None:
        """
        This test verifies that target axes never exceed the requested maximum, that latitude stays within +/-90, and that ranges that are an exact multiple of the step keep their last point.

        Parameters:
            None

        Returns:
            None
        """
        assert _build_target_axis(-90.0, 90.0, 7.0, limit=90.0).max() <= 85.0
        assert _build_target_axis(-180.0, 180.0, 7.0).max() <= 180.0
        assert _build_target_axis(-90.0, 95.0, 5.0, limit=90.0).max() == 90.0

        exact = _build_target_axis(-90.0, 90.0, 0.1, limit=90.0)
        assert len(exact) == 1801 and exact[-1] == pytest.approx(90.0)
        assert np.array_equal(
            _build_target_axis(0.0, 10.0, 1.0), np.arange(0.0, 10.5, 1.0)
        )

        lon, lat = _global_mesh(300)
        out = remap_mpas_to_latlon(
            lon * 0 + 1.0, lon, lat, -180, 180, -90, 90, 7.0, "nearest"
        )
        assert out.lat.values.max() <= 90.0 and out.lon.values.max() <= 180.0

        with pytest.raises(ValueError):
            _build_target_axis(0.0, 1.0, 0.0)

    def test_valid_mask_uses_actual_target_coordinates(
        self: "TestRemappingFixes",
    ) -> None:
        """
        This test verifies that the convex-hull mask is evaluated on the remapped array's own lon/lat coordinates: a data array on a regional grid whose coordinates do not equal a linspace over the requested bounds is masked consistently with its coordinates.

        Parameters:
            None

        Returns:
            None
        """
        src_lon = np.array([0.0, 4.0, 4.0, 0.0])
        src_lat = np.array([0.0, 0.0, 4.0, 4.0])
        lons = np.array([-1.0, 1.0, 3.0, 5.0])
        lats = np.array([-1.0, 1.0, 3.0, 5.0])
        arr = xr.DataArray(
            np.ones((4, 4)), dims=["lat", "lon"], coords={"lon": lons, "lat": lats}
        )

        mask = build_remapped_valid_mask(
            src_lon, src_lat, -1.0, 5.0, -1.0, 5.0, 2.0, arr
        )

        assert mask is not None
        expected = np.zeros((4, 4), dtype=bool)
        expected[1:3, 1:3] = True
        assert np.array_equal(mask, expected)


class TestSelectiveLoadingMissingVariables:
    """Tests that requesting only variables absent from the data files raises a clear error."""

    def test_all_requested_variables_missing_raises_clear_error(
        self: "TestSelectiveLoadingMissingVariables", tmp_path: Path
    ) -> None:
        """
        This regression test verifies that `_compute_drop_variables` raises a ValueError naming the missing and available variables when none of the requested variables exist. Previously every data variable was dropped and xarray's open_mfdataset failed with the unrelated message "Cannot specify both data_vars='minimal' and coords='minimal'".

        Parameters:
            tmp_path (Path): Pytest temporary directory.

        Returns:
            None
        """
        path = str(tmp_path / "diag.2024-01-01_00.00.00.nc")
        xr.Dataset({"t2m": (("nCells",), np.zeros(3))}).to_netcdf(path)
        with pytest.raises(ValueError, match="nosuchvar.*Available variables.*t2m"):
            _base_proc()._compute_drop_variables([path], ["nosuchvar"])

    def test_partially_missing_variables_keep_the_present_ones(
        self: "TestSelectiveLoadingMissingVariables", tmp_path: Path
    ) -> None:
        """
        This test verifies that when at least one requested variable exists, the other data variables are dropped and no error is raised.

        Parameters:
            tmp_path (Path): Pytest temporary directory.

        Returns:
            None
        """
        path = str(tmp_path / "diag.2024-01-01_00.00.00.nc")
        xr.Dataset(
            {"t2m": (("nCells",), np.zeros(3)), "q2": (("nCells",), np.ones(3))}
        ).to_netcdf(path)
        drop = _base_proc()._compute_drop_variables([path], ["t2m", "nosuchvar"])
        assert drop == ["q2"]


class TestVerificationFollowUps:
    """Regression tests for issues found while independently verifying the processing fixes."""

    def test_weights_file_name_distinguishes_target_domains(
        self: "TestVerificationFollowUps", tmp_path: Path
    ) -> None:
        """
        This regression test verifies that the default cached-weights file name differs for two target grids with identical shapes but different longitudes. The name previously encoded only the method and grid sizes, so weights for one domain were silently reused for another (errors up to 2.0 on a field bounded by 1).

        Parameters:
            tmp_path (Path): Pytest temporary directory.

        Returns:
            None
        """
        from mpasdiag.processing.remapping import MPASRemapper

        remapper = MPASRemapper(weights_dir=tmp_path)
        source = xr.Dataset(
            coords={
                "lon": ("n", np.linspace(-170, 170, 50)),
                "lat": ("n", np.zeros(50)),
            }
        )
        target_a = xr.Dataset(
            coords={"lon": np.arange(-180, 0, 10.0), "lat": np.arange(-10, 10, 5.0)}
        )
        target_b = xr.Dataset(
            coords={"lon": np.arange(0, 180, 10.0), "lat": np.arange(-10, 10, 5.0)}
        )
        path_a = remapper._resolve_weights_path(source, target_a, None)
        path_b = remapper._resolve_weights_path(source, target_b, None)
        assert path_a is not None and path_b is not None
        assert path_a != path_b
        assert path_a == remapper._resolve_weights_path(source, target_a, None)

    def test_find_diag_files_mpasout_in_data_dir_and_recursive_time_order(
        self: "TestVerificationFollowUps", tmp_path: Path
    ) -> None:
        """
        This regression test verifies that _find_diag_files finds mpasout files placed directly in the data directory (previously only an 'mpasout' subdirectory was searched with that pattern) and that a recursive search returns files in chronological order rather than path order.

        Parameters:
            tmp_path (Path): Pytest temporary directory.

        Returns:
            None
        """
        flat = tmp_path / "flat"
        flat.mkdir()
        (flat / "mpasout.2024-09-17_01.00.00.nc").write_bytes(b"")
        assert len(FileManager._find_diag_files(str(flat))) == 1

        nested = tmp_path / "nested"
        for sub, hour in [("a", "02"), ("b", "00"), ("c", "01")]:
            (nested / sub).mkdir(parents=True)
            (nested / sub / f"diag.2024-09-17_{hour}.00.00.nc").write_bytes(b"")
        found = FileManager._find_diag_files(str(nested))
        assert [Path(f).name[16:18] for f in found] == ["00", "01", "02"]

    def test_partially_missing_variables_log_a_warning(
        self: "TestVerificationFollowUps", tmp_path: Path
    ) -> None:
        """
        This test verifies that requesting a mix of present and missing variables logs a warning naming the missing ones instead of skipping them silently.

        Parameters:
            tmp_path (Path): Pytest temporary directory.

        Returns:
            None
        """
        path = str(tmp_path / "diag.2024-01-01_00.00.00.nc")
        xr.Dataset({"t2m": (("nCells",), np.zeros(3))}).to_netcdf(path)
        with patch("mpasdiag.processing.base.logger") as mock_logger:
            _base_proc()._compute_drop_variables([path], ["t2m", "nosuchvar"])
        assert any(
            "nosuchvar" in str(call.args) for call in mock_logger.warning.call_args_list
        )

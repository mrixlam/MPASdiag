#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Test Suite: Processing Layer Round-2 Regression Tests

This module contains regression tests for the second round of processing-layer fixes: selective variable loads that must keep the 'xtime' bookkeeping variable, colour-level presets that match the data units, identity unit conversions that must not relabel their input, data-cache staleness detection and zero-copy reads, polar NaN rows in linear global remapping, diagnostic-stream preference in file discovery, empty Time dimensions, DataValidator edge cases, MPASLogger propagation and handler closing, target-grid size limits, the KAPPA constant, below-ground masking of isobaric fields and the chunk_size loading option. Tests that need real MPAS output skip gracefully when the data files are absent.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: October 2026
Version: 1.0.0
"""

import logging
import os
from pathlib import Path
from typing import Any, List

import numpy as np
import pytest
import xarray as xr

from mpasdiag.processing.constants import KAPPA
from mpasdiag.processing.data_cache import MPASDataCache
from mpasdiag.processing.processors_2d import MPAS2DProcessor
from mpasdiag.processing.remapping import remap_mpas_to_latlon
from mpasdiag.processing.utils_datetime import MPASDateTimeUtils
from mpasdiag.processing.utils_file import FileManager
from mpasdiag.processing.utils_logger import MPASLogger, ROOT_LOGGER_NAME
from mpasdiag.processing.utils_metadata import MPASFileMetadata
from mpasdiag.processing.utils_unit import UnitConverter
from mpasdiag.processing.utils_validator import DataValidator

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "data" / "grids" / "x1.10242.static.nc"
DIAG_DIR = ROOT / "data" / "u240k" / "diag"

needs_real_data = pytest.mark.skipif(
    not (GRID.exists() and (DIAG_DIR / "diag.2024-09-17_00.00.00.nc").exists())
    or GRID.stat().st_size < 10_000,
    reason="real MPAS test data not available (Git LFS)",
)


def _fibonacci_sphere(n: int) -> "tuple[np.ndarray, np.ndarray]":
    """
    This helper generates n nearly uniformly distributed points on the sphere with the Fibonacci lattice, giving a global mesh that reaches both poles for polar-gap tests without needing any data file.

    Parameters:
        n (int): Number of points.

    Returns:
        tuple[np.ndarray, np.ndarray]: Longitude in [0, 360) and latitude in degrees.
    """
    k = np.arange(n) + 0.5
    lat = np.degrees(np.arcsin(1.0 - 2.0 * k / n))
    lon = np.mod(np.degrees(np.pi * (1.0 + 5**0.5) * k), 360.0)
    return lon, lat


class TestSelectiveLoadKeepsXtime:
    """Selective variable loads must keep xtime so multi-record files get datetime Times."""

    @needs_real_data
    def test_multirecord_selective_load_gives_datetime_times(
        self: "TestSelectiveLoadKeepsXtime", tmp_path: Path
    ) -> None:
        """
        This test concatenates real diag files 00+01 and 02+03 into two files holding two Time records each and loads only rainc and rainnc; the combined dataset must carry four datetime Times and still contain xtime.

        Parameters:
            tmp_path (Path): pytest temporary directory.

        Returns:
            None
        """
        for first, second in (("00", "01"), ("02", "03")):
            parts = []
            for hour in (first, second):
                with xr.open_dataset(
                    DIAG_DIR / f"diag.2024-09-17_{hour}.00.00.nc", decode_times=False
                ) as ds:
                    parts.append(ds[["xtime", "rainc", "rainnc"]].load())
            xr.concat(parts, dim="Time", data_vars="all").to_netcdf(
                tmp_path / f"diag.2024-09-17_{first}.00.00.nc"
            )

        proc = MPAS2DProcessor(str(GRID), verbose=False).load_2d_data(
            str(tmp_path), variables=["rainc", "rainnc"]
        )
        assert proc.dataset is not None
        times = np.asarray(proc.dataset.Time.values)

        assert times.dtype.kind == "M"
        assert len(times) == 4
        assert "xtime" in proc.dataset.data_vars


class TestMetadataPresets:
    """Colour-level presets must be expressed in the units of the plotted data."""

    @pytest.mark.parametrize(
        "name,low,high",
        [
            ("th2m", 200.0, 360.0),
            ("q2", 0.0, 45.0),
            ("relhum_200hpa", 0.0, 100.0),
            ("relhum_250hpa", 0.0, 100.0),
            ("relhum_500hpa", 0.0, 100.0),
            ("relhum_700hpa", 0.0, 100.0),
            ("relhum_850hpa", 0.0, 100.0),
            ("relhum_925hpa", 0.0, 100.0),
            ("t2m", -60.0, 60.0),
            ("olrtoa", 0.0, 400.0),
        ],
    )
    def test_levels_within_physical_range(
        self: "TestMetadataPresets", name: str, low: float, high: float
    ) -> None:
        """
        This test checks that the contour levels of the audited presets lie inside the physical range of the display unit (K for th2m, g/kg for q2, percent for relative humidity, degrees Celsius for t2m, W m-2 for olrtoa) and span a useful part of it.

        Parameters:
            name (str): Preset variable name.
            low (float): Lowest acceptable level.
            high (float): Highest acceptable level.

        Returns:
            None
        """
        levels = MPASFileMetadata.get_2d_variable_metadata(name, None)["levels"]

        assert min(levels) >= low and max(levels) <= high
        assert max(levels) - min(levels) >= 0.3 * (high - low)

    @needs_real_data
    def test_levels_cover_real_data(self: "TestMetadataPresets") -> None:
        """
        This test verifies against real diag output that the corrected presets bracket at least 90 percent of the finite data in the display unit.

        Parameters:
            None

        Returns:
            None
        """
        with xr.open_dataset(
            DIAG_DIR / "diag.2024-09-17_03.00.00.nc", decode_times=False
        ) as ds:
            for name, factor in (("th2m", 1.0), ("q2", 1000.0), ("relhum_500hPa", 1.0)):
                values = ds[name].isel(Time=0).values.astype(float) * factor
                values = values[np.isfinite(values) & (values != 0)]
                levels = MPASFileMetadata.get_2d_variable_metadata(
                    name.lower().replace("hpa", "hpa"), None
                )["levels"]
                inside = np.mean((values >= min(levels)) & (values <= max(levels)))
                assert inside > 0.9, name


class TestIdentityConversionDoesNotMutate:
    """Identity unit conversions must never relabel the caller's array."""

    def test_hpa_to_mb_keeps_input_attrs(
        self: "TestIdentityConversionDoesNotMutate",
    ) -> None:
        """
        This test converts a DataArray from hPa to mb (an identity conversion) and verifies that the result is labelled mb while the input, and the Dataset variable it came from, still say hPa.

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({"p": ("x", np.array([1000.0, 850.0]), {"units": "hPa"})})
        original = ds["p"]
        out = UnitConverter.convert_units(original, "hPa", "mb")

        assert isinstance(out, xr.DataArray)
        assert out.attrs["units"] == "mb"
        assert original.attrs["units"] == "hPa"
        assert ds["p"].attrs["units"] == "hPa"
        np.testing.assert_array_equal(out.values, original.values)

    def test_kg_m2_to_mm_keeps_input_attrs(
        self: "TestIdentityConversionDoesNotMutate",
    ) -> None:
        """
        This test covers the kg m-2 to mm identity conversion with the same non-mutation guarantee.

        Parameters:
            None

        Returns:
            None
        """
        arr = xr.DataArray([1.0, 2.0], dims="x", attrs={"units": "kg m^{-2}"})
        out = UnitConverter.convert_units(arr, "kg m^{-2}", "mm")

        assert out.attrs["units"] == "mm"
        assert arr.attrs["units"] == "kg m^{-2}"

    def test_metre_to_micron(self: "TestIdentityConversionDoesNotMutate") -> None:
        """
        This test verifies the added metre to micron conversion that makes the effective-radius presets (re_cloud, re_ice, re_snow) usable with data stored in metres.

        Parameters:
            None

        Returns:
            None
        """
        assert UnitConverter.convert_units(2.5e-5, "m", "microns") == pytest.approx(
            25.0
        )
        assert UnitConverter.convert_units(25.0, "microns", "m") == pytest.approx(
            2.5e-5
        )


class TestDataCacheStaleness:
    """The cache must refuse stale data on get and avoid full copies."""

    @staticmethod
    def _dataset(scale: float = 1.0) -> xr.Dataset:
        """
        This helper builds a tiny cell dataset with coordinates in degrees and one variable.

        Parameters:
            scale (float): Multiplier applied to the variable values.

        Returns:
            xr.Dataset: The dataset.
        """
        return xr.Dataset(
            {
                "lonCell": (
                    "nCells",
                    np.array([10.0, 20.0, 30.0]),
                    {"units": "degrees"},
                ),
                "latCell": (
                    "nCells",
                    np.array([-10.0, 0.0, 10.0]),
                    {"units": "degrees"},
                ),
                "t2m": (("Time", "nCells"), np.array([[1.0, 2.0, 3.0]]) * scale),
            }
        )

    def test_get_with_different_dataset_raises(self: "TestDataCacheStaleness") -> None:
        """
        This test caches from one dataset and then asks for the entry while presenting a different dataset; both getters must raise KeyError instead of returning stale arrays.

        Parameters:
            None

        Returns:
            None
        """
        first, second = self._dataset(), self._dataset(5.0)
        cache = MPASDataCache()
        cache.load_coordinates_from_dataset(first)
        cache.load_variable_data(first, "t2m", 0)

        cache.get_coordinates(dataset=first)
        cache.get_variable_data("t2m", 0, dataset=first)

        with pytest.raises(KeyError):
            cache.get_coordinates(dataset=second)

        with pytest.raises(KeyError):
            cache.get_variable_data("t2m", 0, dataset=second)

    def test_refresh_rereads_inplace_mutation(self: "TestDataCacheStaleness") -> None:
        """
        This test mutates a dataset in place (which the identity token cannot see), verifies the documented limit that a plain load keeps the old values, and that refresh=True re-reads them.

        Parameters:
            None

        Returns:
            None
        """
        ds = self._dataset()
        cache = MPASDataCache()
        cache.load_variable_data(ds, "t2m", 0)
        ds["t2m"].values[:] = 99.0

        cache.load_variable_data(ds, "t2m", 0)
        assert cache.get_variable_data("t2m", 0).data.max() == 3.0

        cache.load_variable_data(ds, "t2m", 0, refresh=True)
        assert cache.get_variable_data("t2m", 0).data.max() == 99.0

    def test_default_getters_return_readonly_views(
        self: "TestDataCacheStaleness",
    ) -> None:
        """
        This test verifies that a default hit returns the cached arrays themselves (no copy) with the writeable flag off, while writable=True returns independent writable copies.

        Parameters:
            None

        Returns:
            None
        """
        ds = self._dataset()
        cache = MPASDataCache()
        cache.load_coordinates_from_dataset(ds)
        cache.load_variable_data(ds, "t2m", 0)

        lon, _ = cache.get_coordinates()
        assert lon is cache._coordinates["default"][0]
        assert not lon.flags.writeable
        assert cache.get_variable_data("t2m", 0).data is cache._variables["t2m_t0"].data

        wlon, _ = cache.get_coordinates(writable=True)
        assert wlon.flags.writeable and wlon is not lon
        assert cache.get_variable_data("t2m", 0, writable=True).data.flags.writeable


class TestPolarRemap:
    """Linear global remapping must not leave NaN rows at the poles."""

    def test_global_linear_has_no_polar_nan(self: "TestPolarRemap") -> None:
        """
        This test remaps a smooth field defined on a global Fibonacci mesh to 1 and 2 degree grids with the linear method and asserts that no NaN remains, that the polar rows equal the analytic pole value, and that the filled field stays within the source range.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _fibonacci_sphere(8000)
        field = np.sin(np.radians(lat))

        for res in (1.0, 2.0):
            out = remap_mpas_to_latlon(field, lon, lat, resolution=res, method="linear")
            assert not np.isnan(out.values).any()
            assert out.values.min() >= -1.0 - 1e-9 and out.values.max() <= 1.0 + 1e-9
            np.testing.assert_allclose(out.values[-1], 1.0, atol=0.02)
            np.testing.assert_allclose(out.values[0], -1.0, atol=0.02)

    def test_interior_values_unchanged(self: "TestPolarRemap") -> None:
        """
        This test verifies that the polar fill never alters values the plain linear interpolation already produced, by comparing against scipy's griddata on the same wrapped source points.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing import remapping as rm

        lon, lat = _fibonacci_sphere(4000)
        field = np.cos(np.radians(lat)) * np.cos(np.radians(lon))
        filled = remap_mpas_to_latlon(field, lon, lat, resolution=2.0, method="linear")

        lon_w = rm._wrap_longitude_to_frame(lon, -180.0)
        pts, vals = rm._add_wrapped_boundary_points(
            np.column_stack([lon_w, lat]), field, lon_w, lat, -180.0, 180.0
        )
        from scipy.interpolate import griddata

        lon2d, lat2d = np.meshgrid(filled.lon.values, filled.lat.values)
        plain = griddata(
            pts, vals, np.column_stack([lon2d.ravel(), lat2d.ravel()]), method="linear"
        ).reshape(lon2d.shape)
        ok = ~np.isnan(plain)

        assert np.isnan(plain).any()
        np.testing.assert_array_equal(filled.values[ok], plain[ok])

    def test_regional_gaps_stay_nan(self: "TestPolarRemap") -> None:
        """
        This test remaps a regional source covering only part of the globe and asserts that cells outside the source coverage, including the pole, remain NaN.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _fibonacci_sphere(20000)
        keep = (lon > 100) & (lon < 160) & (lat > 0) & (lat < 60)
        out = remap_mpas_to_latlon(
            np.ones(int(keep.sum())),
            lon[keep],
            lat[keep],
            resolution=2.0,
            method="linear",
        )

        assert np.isnan(out.values[-1]).all()
        assert np.isnan(out.values[0]).all()


class TestRemapSizeLimit:
    """Absurdly fine target grids must fail with a clear error."""

    def test_tiny_resolution_raises(self: "TestRemapSizeLimit") -> None:
        """
        This test asks for a 1e-5 degree grid and expects a ValueError mentioning the safety limit instead of a multi-terabyte allocation attempt.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat = _fibonacci_sphere(100)

        with pytest.raises(ValueError, match="safety limit"):
            remap_mpas_to_latlon(np.ones(100), lon, lat, resolution=1e-5)


class TestFindDiagFilesOrder:
    """_find_diag_files must prefer diag streams over mpasout streams."""

    def test_diag_subdir_beats_mpasout_in_root(
        self: "TestFindDiagFilesOrder", tmp_path: Path
    ) -> None:
        """
        This test places an mpasout file in the data directory next to a diag/ subdirectory and asserts that the diag file is returned.

        Parameters:
            tmp_path (Path): pytest temporary directory.

        Returns:
            None
        """
        (tmp_path / "diag").mkdir()
        (tmp_path / "diag" / "diag.2024-09-17_00.00.00.nc").write_bytes(b"")
        (tmp_path / "mpasout.2024-09-17_00.00.00.nc").write_bytes(b"")

        files = FileManager._find_diag_files(str(tmp_path))

        assert [os.path.basename(f) for f in files] == ["diag.2024-09-17_00.00.00.nc"]

    def test_mpasout_used_when_no_diag(
        self: "TestFindDiagFilesOrder", tmp_path: Path
    ) -> None:
        """
        This test verifies that mpasout files are still returned when no diag file exists anywhere.

        Parameters:
            tmp_path (Path): pytest temporary directory.

        Returns:
            None
        """
        (tmp_path / "mpasout.2024-09-17_00.00.00.nc").write_bytes(b"")

        assert len(FileManager._find_diag_files(str(tmp_path))) == 1


class TestEmptyTimeDimension:
    """validate_time_parameters reports an empty Time dimension clearly."""

    def test_empty_time_raises_value_error(self: "TestEmptyTimeDimension") -> None:
        """
        This test passes a dataset with a zero-length Time dimension and expects a ValueError mentioning 'no time steps'.

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({"a": (("Time", "nCells"), np.zeros((0, 3)))})

        with pytest.raises(ValueError, match="no time steps"):
            MPASDateTimeUtils.validate_time_parameters(ds, 0)


class TestDataValidatorEdges:
    """DataValidator edge cases."""

    def test_empty_array_is_invalid_not_error(self: "TestDataValidatorEdges") -> None:
        """
        This test validates an empty array and expects an invalid result rather than a ZeroDivisionError.

        Parameters:
            None

        Returns:
            None
        """
        result = DataValidator.validate_data_array(np.array([]))

        assert result["valid"] is False and result["stats"]["total_points"] == 0

    def test_list_input_accepted(self: "TestDataValidatorEdges") -> None:
        """
        This test passes a plain list and expects the usual statistics.

        Parameters:
            None

        Returns:
            None
        """
        result = DataValidator.validate_data_array([1.0, 2.0, 3.0])  # type: ignore[arg-type]

        assert result["stats"]["mean"] == pytest.approx(2.0)

    def test_masked_median_excludes_masked(self: "TestDataValidatorEdges") -> None:
        """
        This test verifies that masked values are excluded from the median as they are from min, max and mean.

        Parameters:
            None

        Returns:
            None
        """
        data = np.ma.masked_array(
            [1.0, 2.0, 3.0, 1000.0], mask=[False, False, False, True]
        )
        stats = DataValidator.validate_data_array(data)["stats"]

        assert stats["median"] == 2.0 and stats["max"] == 3.0 and stats["mean"] == 2.0

    def test_coordinates_empty_and_0_360(self: "TestDataValidatorEdges") -> None:
        """
        This test verifies that empty coordinate arrays return False and that [0, 360] longitudes are accepted.

        Parameters:
            None

        Returns:
            None
        """
        assert DataValidator.validate_coordinates(np.array([]), np.array([])) is False
        assert (
            DataValidator.validate_coordinates(
                np.array([0.0, 359.5]), np.array([0.0, 10.0])
            )
            is True
        )
        assert (
            DataValidator.validate_coordinates(np.array([-181.0]), np.array([0.0]))
            is False
        )

    def test_size_limits_reject_nan_and_zero_workers(
        self: "TestDataValidatorEdges",
    ) -> None:
        """
        This test verifies that NaN sizes and a worker count of zero are rejected while ordinary values pass.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError):
            DataValidator.enforce_size_limits(n_src=float("nan"))

        with pytest.raises(ValueError):
            DataValidator.enforce_size_limits(n_workers=0)

        DataValidator.enforce_size_limits(n_src=10, n_workers=2)


class TestLoggerPropagation:
    """MPASLogger must not propagate and must close replaced handlers."""

    def test_no_duplicate_lines_with_root_handler(
        self: "TestLoggerPropagation", capsys: pytest.CaptureFixture
    ) -> None:
        """
        This test installs a handler on the Python root logger and verifies that an mpasdiag record is not also delivered there.

        Parameters:
            capsys (pytest.CaptureFixture): Output capture fixture.

        Returns:
            None
        """
        seen: List[logging.LogRecord] = []

        class _Collect(logging.Handler):
            """Collects records that reach the root logger."""

            def emit(self: "_Collect", record: logging.LogRecord) -> None:
                """
                This method stores the record.

                Parameters:
                    record (logging.LogRecord): The record.

                Returns:
                    None
                """
                seen.append(record)

        handler = _Collect()
        logging.getLogger().addHandler(handler)

        try:
            MPASLogger(name=ROOT_LOGGER_NAME, level=logging.INFO, verbose=True)
            logging.getLogger(ROOT_LOGGER_NAME + ".x").info("once")
        finally:
            logging.getLogger().removeHandler(handler)
            MPASLogger(name=ROOT_LOGGER_NAME, level=logging.INFO, verbose=False)

        assert not seen


class TestKappa:
    """KAPPA equals the MPAS rgas/cp value."""

    def test_kappa_is_two_sevenths(self: "TestKappa") -> None:
        """
        This test asserts KAPPA == 2/7 exactly and that the theta to temperature and back round trip in the sounding code convention is consistent.

        Parameters:
            None

        Returns:
            None
        """
        assert KAPPA == 2.0 / 7.0

        p, p0, t = 85000.0, 100000.0, 285.0
        theta = t * (p0 / p) ** KAPPA
        assert theta * (p / p0) ** KAPPA == pytest.approx(t)


@needs_real_data
class TestBelowGroundMasking:
    """Isobaric diag fields are masked where the pressure level is below ground."""

    @staticmethod
    def _proc(variables: Any = None, **kwargs: Any) -> MPAS2DProcessor:
        """
        This helper loads the real diag directory with the given variable subset.

        Parameters:
            variables (Any): Variable subset or None.
            **kwargs (Any): Extra load_2d_data arguments.

        Returns:
            MPAS2DProcessor: The loaded processor.
        """
        return MPAS2DProcessor(str(GRID), verbose=False).load_2d_data(
            str(DIAG_DIR), variables=variables, **kwargs
        )

    def test_height_and_temperature_masked_at_same_cells(
        self: "TestBelowGroundMasking",
    ) -> None:
        """
        This test loads only temperature_850hPa (height_850hPa must be kept automatically), and verifies that the cells with height below terrain are NaN in both fields, that the Himalaya and Antarctica contain masked cells, that attrs record the masking, and that mask_below_ground=False returns the raw values.

        Parameters:
            None

        Returns:
            None
        """
        proc = self._proc(["temperature_850hPa"])
        assert proc.dataset is not None
        assert "height_850hPa" in proc.dataset.data_vars

        raw_h = proc.get_2d_variable_data("height_850hPa", 0, mask_below_ground=False)
        raw_t = proc.get_2d_variable_data(
            "temperature_850hPa", 0, mask_below_ground=False
        )
        h = proc.get_2d_variable_data("height_850hPa", 0)
        t = proc.get_2d_variable_data("temperature_850hPa", 0)

        with xr.open_dataset(GRID) as grid:
            ter = grid["ter"].values
            lat = np.degrees(grid["latCell"].values)
            lon = np.degrees(grid["lonCell"].values)

        below = raw_h.values < ter

        assert below.sum() > 0 and not np.isnan(raw_t.values).any()
        np.testing.assert_array_equal(np.isnan(h.values), below)
        np.testing.assert_array_equal(np.isnan(t.values), below)
        assert t.attrs["below_ground_masked_cells"] == int(below.sum())
        assert "below_ground_masked" in t.attrs
        assert below[(lat > 27) & (lat < 35) & (lon > 80) & (lon < 100)].any()
        assert below[lat < -80].any()

    def test_non_isobaric_variable_not_masked(self: "TestBelowGroundMasking") -> None:
        """
        This test verifies that a surface field such as t2m is returned without masking attrs.

        Parameters:
            None

        Returns:
            None
        """
        proc = self._proc(["t2m"])
        data = proc.get_2d_variable_data("t2m", 1)

        assert "below_ground_masked" not in data.attrs


@needs_real_data
class TestChunkSize:
    """load_2d_data accepts chunk_size along nCells."""

    def test_chunk_size_applied(self: "TestChunkSize") -> None:
        """
        This test loads t2m with chunk_size=2000 and verifies that nCells is split into 2000-cell dask chunks while the default load keeps a single chunk, and that invalid sizes raise ValueError.

        Parameters:
            None

        Returns:
            None
        """
        chunked = MPAS2DProcessor(str(GRID), verbose=False).load_2d_data(
            str(DIAG_DIR), variables=["t2m"], chunk_size=2000
        )
        default = MPAS2DProcessor(str(GRID), verbose=False).load_2d_data(
            str(DIAG_DIR), variables=["t2m"]
        )

        assert chunked.dataset is not None and default.dataset is not None
        assert chunked.dataset["t2m"].chunks[1][0] == 2000
        assert len(default.dataset["t2m"].chunks[1]) == 1

        for bad in (0, -5):
            with pytest.raises(ValueError):
                MPAS2DProcessor(str(GRID), verbose=False).load_2d_data(
                    str(DIAG_DIR), chunk_size=bad
                )


class TestVerificationRound2FollowUps:
    """Regression tests for the issues left after independently verifying the round-2 processing fixes."""

    def test_module_create_target_grid_rejects_huge_grids(
        self: "TestVerificationRound2FollowUps",
    ) -> None:
        """
        This regression test verifies that the module-level create_target_grid checks the size of the full 2-D grid before allocating its axes. Each 1e-5 degree axis passes its own limit, so the function used to allocate tens of millions of points per axis instead of failing early like MPASRemapper.create_target_grid.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.remapping import create_target_grid

        with pytest.raises(ValueError, match="safety limit"):
            create_target_grid(dlon=1e-5, dlat=1e-5)

    def test_enforce_size_limits_rejects_numpy_float_nan(
        self: "TestVerificationRound2FollowUps",
    ) -> None:
        """
        This regression test verifies that enforce_size_limits rejects NaN sizes given as NumPy float32/float16 scalars, which are not instances of the Python float type and used to slip through the NaN check.

        Parameters:
            None

        Returns:
            None
        """
        for value in (np.float32("nan"), np.float16("nan"), np.float64("nan")):
            with pytest.raises(ValueError, match="NaN"):
                DataValidator.enforce_size_limits(n_src=value)

    def test_upper_level_presets_cover_polar_values(
        self: "TestVerificationRound2FollowUps",
    ) -> None:
        """
        This test verifies that the temperature_250hPa and height_100hPa colour presets extend to the cold and low polar values seen in the real data (about -77 degC and 14,300 m), which previously fell below the lowest level.

        Parameters:
            None

        Returns:
            None
        """
        t250 = MPASFileMetadata.get_2d_variable_metadata("temperature_250hPa", None)[
            "levels"
        ]
        h100 = MPASFileMetadata.get_2d_variable_metadata("height_100hPa", None)[
            "levels"
        ]
        assert min(t250) <= -76 and max(t250) >= -34
        assert min(h100) <= 14320 and max(h100) >= 16950

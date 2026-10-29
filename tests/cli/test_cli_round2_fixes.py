"""Regression tests for the second round of CLI / configuration / parallel orchestration fixes."""

import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mpasdiag.diagnostics.precipitation import InsufficientLookbackError
from mpasdiag.processing.cli_unified import MPASUnifiedCLI
from mpasdiag.processing.parallel import (
    MPASParallelManager,
    TaskResult,
    _ensure_transferable,
)
from mpasdiag.processing.parallel_wrappers import (
    BatchRenderOptions,
    cross_section_file_stem,
    default_precipitation_title,
)
from mpasdiag.processing.utils_config import MPASConfig
from mpasdiag.processing.utils_monitor import PerformanceMonitor


def _cfg(**kw):
    return MPASConfig(grid_file="g.nc", data_dir="d", output_dir="o", **kw)


def _dataset(hours):
    times = pd.to_datetime("2024-09-17") + pd.to_timedelta(hours, unit="h")
    return xr.Dataset(coords={"Time": times})


class _Unpicklable:
    def __reduce__(self):
        raise TypeError("cannot pickle this result")


def _return_unpicklable(task):
    return _Unpicklable() if task == 1 else task


class TestLookbackFromTimeCoordinate:
    def test_three_hourly_a03h_index_one_ok(self):
        cli = MPASUnifiedCLI()
        cfg = _cfg(accumulation_period="a03h")
        ds = _dataset([0, 3, 6])
        cli._check_lookback(cfg, 1, ds)
        assert cli._first_lookback_index(ds, cfg) == (1, None)

    def test_a01h_on_three_hourly_errors(self):
        cli = MPASUnifiedCLI()
        cfg = _cfg(accumulation_period="a01h")
        ds = _dataset([0, 3, 6])
        with pytest.raises(InsufficientLookbackError):
            cli._check_lookback(cfg, 1, ds)
        first, reason = cli._first_lookback_index(ds, cfg)
        assert first == 3 and "No output at" in reason

    def test_hourly_default_unchanged(self):
        cli = MPASUnifiedCLI()
        cfg = _cfg(accumulation_period="a03h")
        with pytest.raises(InsufficientLookbackError):
            cli._check_lookback(cfg, 2)


class TestExtent:
    rng = np.random.default_rng(1)
    lat = rng.uniform(-10, 10, 2000)

    def _lon_global(self):
        return self.rng.uniform(-180, 180, 2000)

    def test_single_lon_min_global(self):
        b = MPASUnifiedCLI()._resolve_extent(
            _cfg(lon_min=-100.0), self._lon_global(), self.lat
        )
        assert b.lon_min == -100.0 and b.lon_max == 180.0

    def test_single_lon_max_global(self):
        b = MPASUnifiedCLI()._resolve_extent(
            _cfg(lon_max=50.0), self._lon_global(), self.lat
        )
        assert b.lon_min == -180.0 and b.lon_max == 50.0

    def test_seam_crossing_region(self):
        lon = np.concatenate(
            [self.rng.uniform(350, 360, 500), self.rng.uniform(0, 10, 500)]
        )
        b = MPASUnifiedCLI()._resolve_extent(_cfg(), lon, self.lat[:1000])
        assert (b.lon_min, b.lon_max) == (-10.0, 10.0)

    def test_dateline_region(self):
        lon = self.rng.uniform(170, 190, 1000)
        b = MPASUnifiedCLI()._resolve_extent(_cfg(), lon, self.lat[:1000])
        assert (b.lon_min, b.lon_max) == (170.0, 190.0)


class TestTimeIndex:
    def test_negative_index_normalised(self):
        cli = MPASUnifiedCLI()
        ds = _dataset([0, 1, 2, 3])
        assert cli._check_time_index(ds, -1) == 3
        with pytest.raises(IndexError):
            cli._check_time_index(ds, -5)


class TestWindLevel:
    @pytest.mark.parametrize(
        "text,selection,label",
        [
            ("surface", "surface", "surface"),
            ("850hPa", 85000.0, "850 hPa"),
            ("85000", 85000.0, "850 hPa"),
            ("5", 5, "model level 5"),
        ],
    )
    def test_parse(self, text, selection, label):
        assert MPASUnifiedCLI._parse_wind_level(text) == (selection, label)

    def test_invalid(self):
        with pytest.raises(ValueError, match="Invalid --wind-level"):
            MPASUnifiedCLI._parse_wind_level("abc")


class TestConfigC13:
    def test_subsample_zero_normalised(self):
        assert _cfg(subsample_factor=0).subsample_factor == -1

    def test_negative_time_index_accepted(self):
        assert _cfg(time_index=-1).time_index == -1


class TestParserOptions:
    def _args(self, argv):
        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()
        return cli, parser.parse_args(argv)

    def test_verbose_and_quiet_rejected(self):
        cli, args = self._args(
            ["--verbose", "--quiet", "surface", "--grid-file", "g", "--data-dir", "d"]
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            cli._args_to_dict(args)

    def test_sounding_max_distance_mapped(self):
        cli, args = self._args(
            [
                "sounding",
                "--grid-file",
                "g",
                "--data-dir",
                "d",
                "--lon",
                "1",
                "--lat",
                "2",
                "--max-distance-km",
                "50",
            ]
        )
        assert cli._args_to_dict(args)["sounding_max_distance_km"] == 50.0

    def test_dead_options_reach_config(self):
        cli, args = self._args(
            [
                "precipitation",
                "--grid-file",
                "g",
                "--data-dir",
                "d",
                "--units",
                "in",
                "--threshold",
                "2",
                "--chunk-size",
                "500",
            ]
        )
        d = cli._args_to_dict(args)
        assert (d["precip_units"], d["precip_threshold"], d["chunk_size"]) == (
            "in",
            2.0,
            500,
        )
        cli, args = self._args(
            ["surface", "--grid-file", "g", "--data-dir", "d", "--contour-levels", "7"]
        )
        assert cli._args_to_dict(args)["contour_levels"] == 7

    def test_render_options_carry_chunk_and_level(self):
        opts = MPASUnifiedCLI()._render_options(
            _cfg(chunk_size=300), (85000.0, "850 hPa")
        )
        assert isinstance(opts, BatchRenderOptions)
        assert (
            opts.chunk_size == 300
            and opts.wind_level == "850 hPa"
            and opts.wind_level_selection == 85000.0
        )


class TestNamesAndTitles:
    def test_cross_section_stem(self):
        assert (
            cross_section_file_stem("theta", "pressure", "20240917T03", 12.5)
            == "mpas_cross_section_theta_vcrd_pressure_valid_20240917T03_maxh_12p5km"
        )
        assert cross_section_file_stem("theta", "pressure", "20240917T03") == (
            "mpas_cross_section_theta_vcrd_pressure_valid_20240917T03"
        )

    def test_precip_title(self):
        assert default_precipitation_title("rainnc", "20240917T03") == (
            "MPAS Precipitation | Variable: rainnc | Valid: 20240917T03"
        )


class TestPerformanceMonitorNesting:
    def test_name_nested_once_then_top_level_counts(self):
        mon = PerformanceMonitor(quiet=True)
        with mon.timer("outer"):
            with mon.timer("work"):
                pass
        with mon.timer("work"):
            pass
        assert "work" not in mon._nested


class TestPoolResultPickling:
    def test_unpicklable_result_becomes_failed_task(self):
        out = _ensure_transferable(
            TaskResult(task_id=3, success=True, result=_Unpicklable()), "collect"
        )
        assert out.success is False and "could not be transferred" in out.error
        assert pickle.loads(pickle.dumps(out)).task_id == 3

    def test_pool_runs_each_task_once(self):
        mgr = MPASParallelManager(n_workers=2, verbose=False)
        mgr.error_policy = type(mgr.error_policy)("collect")
        results = mgr._run_pool_with_fallback(
            _return_unpicklable, list(enumerate([0, 1, 2])), (), {}
        )
        assert [r.success for r in results] == [True, False, True]
        assert results[0].result == 0 and results[2].result == 2

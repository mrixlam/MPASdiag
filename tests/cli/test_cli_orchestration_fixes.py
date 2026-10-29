#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""
MPASdiag Test Suite: CLI Orchestration Regression Tests

This module contains regression tests for the command-line interface, configuration, and parallel orchestration fixes: exit status propagation for failed batch steps, the precedence of explicit flags over YAML over defaults, honoring of --time-range, quiet and verbose handling, --base-dir applied to --config, out-of-range time indices, insufficient precipitation lookback, threading of plot options through the parallel batch path, configuration validation, and the console entry points returning the CLI exit status. All tests use synthetic datasets and mocks, so no real MPAS files are required.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: April 2026
Version: 1.0.0
"""

import io
import runpy
import sys
import contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from mpasdiag.processing.cli_unified import MPASUnifiedCLI
from mpasdiag.processing.utils_config import MPASConfig
from mpasdiag.processing.utils_logger import MPASLogger
from mpasdiag.processing.utils_monitor import PerformanceMonitor
from mpasdiag.processing.parallel_wrappers import BatchFiles
from mpasdiag.diagnostics.precipitation import InsufficientLookbackError


def _dataset(n_times: int = 7) -> xr.Dataset:
    """
    This helper builds a small synthetic dataset with an hourly Time coordinate, used by the tests that only need a time dimension.

    Parameters:
        n_times (int): Number of hourly time steps.

    Returns:
        xr.Dataset: Dataset with a Time dimension and an hourly Time coordinate.
    """
    times = np.array(
        [
            np.datetime64("2024-09-17T00:00") + np.timedelta64(i, "h")
            for i in range(n_times)
        ]
    )
    return xr.Dataset(
        {"t2m": (("Time", "nCells"), np.zeros((n_times, 4)))}, coords={"Time": times}
    )


def _cli() -> MPASUnifiedCLI:
    """
    This helper creates a CLI instance with a logger attached so that error reporting paths can be exercised.

    Parameters:
        None

    Returns:
        MPASUnifiedCLI: A CLI instance with logging configured.
    """
    cli = MPASUnifiedCLI()
    cli.logger = MPASLogger("test_orchestration", verbose=False).logger
    return cli


class TestBatchExitStatus:
    """Batch runs must report failure when any step fails (finding 1)."""

    def test_batch_loop_all_failed_returns_false(self: "TestBatchExitStatus") -> None:
        """
        This test verifies that a serial batch in which every step raises is reported as a failure, which is what makes the CLI exit non-zero for an all-failed batch.

        Parameters:
            None

        Returns:
            None
        """
        cli = _cli()

        def step(idx: int) -> None:
            raise RuntimeError("boom")

        assert cli._run_batch_loop("surface maps", [0, 1, 2], step) is False

    def test_batch_loop_partial_failure_returns_false(
        self: "TestBatchExitStatus",
    ) -> None:
        """
        This test verifies that one failing step among several makes the batch fail while the remaining steps still run.

        Parameters:
            None

        Returns:
            None
        """
        cli = _cli()
        done: List[int] = []

        def step(idx: int) -> None:
            if idx == 1:
                raise RuntimeError("boom")
            done.append(idx)

        assert cli._run_batch_loop("surface maps", [0, 1, 2], step) is False
        assert done == [0, 2]

    def test_batch_loop_success_and_skips(self: "TestBatchExitStatus") -> None:
        """
        This test verifies that a lookback skip is not counted as a failure, that a fully successful batch returns True, and that a batch in which every step is skipped (nothing created) or that has no steps at all returns False.

        Parameters:
            None

        Returns:
            None
        """
        cli = _cli()

        def step(idx: int) -> None:
            if idx == 0:
                raise InsufficientLookbackError("too early")

        assert cli._run_batch_loop("maps", [0, 1], step) is True
        assert cli._run_batch_loop("maps", [0], step) is False
        assert cli._run_batch_loop("maps", [], step) is False

    def test_finish_parallel_uses_counts(self: "TestBatchExitStatus") -> None:
        """
        This test verifies that the parallel result handler fails on any failed step, on an empty result, treats a None result (non-root MPI rank) as success, and succeeds for a clean result.

        Parameters:
            None

        Returns:
            None
        """
        cli = _cli()
        good = BatchFiles(["a.png"])
        good.successful = 1
        bad = BatchFiles(["a.png"])
        bad.successful = 1
        bad.failed = 1

        assert cli._finish_parallel(good, "maps") is True
        assert cli._finish_parallel(bad, "maps") is False
        assert cli._finish_parallel([], "maps") is False
        assert cli._finish_parallel(None, "maps") is True

    def test_surface_analysis_returns_false_when_every_step_fails(
        self: "TestBatchExitStatus", tmp_path: Path
    ) -> None:
        """
        This test runs the surface analysis in batch mode against a fake processor whose variable lookup always fails and verifies that run_analysis returns False (exit status 1) instead of True.

        Parameters:
            tmp_path (Path): Temporary directory used as the output directory.

        Returns:
            None
        """
        fake = MagicMock()
        fake.dataset = _dataset(3)
        fake.load_2d_data.return_value = fake
        fake.extract_2d_coordinates_for_variable.return_value = (
            np.array([0.0, 10.0]),
            np.array([0.0, 10.0]),
        )
        fake.get_2d_variable_data.side_effect = KeyError("nosuchvar")

        config = MPASConfig(
            grid_file="g.nc",
            data_dir="d",
            output_dir=str(tmp_path),
            variable="nosuchvar",
            analysis_type="surface",
            batch_mode=True,
        )
        cli = _cli()

        with (
            patch("mpasdiag.processing.cli_unified.MPAS2DProcessor", return_value=fake),
            patch("mpasdiag.processing.cli_unified.MPASSurfacePlotter"),
        ):
            assert cli.run_analysis(config) is False


class TestPrecedence:
    """Explicit CLI flags > YAML > defaults (finding 2) and --base-dir for --config (finding 6)."""

    def _merge(self, cli: MPASUnifiedCLI, argv: List[str]) -> MPASConfig:
        """
        This helper parses an argument vector with the real parser and merges it with any --config file.

        Parameters:
            cli (MPASUnifiedCLI): The CLI instance.
            argv (List[str]): The argument vector.

        Returns:
            MPASConfig: The merged configuration.
        """
        args = cli.create_main_parser().parse_args(argv)
        return cli._load_and_merge_config(args)

    def test_yaml_values_survive_when_flags_not_given(
        self: "TestPrecedence", tmp_path: Path
    ) -> None:
        """
        This test verifies that values from the YAML file (including booleans, figure size, formats, dpi, and the output directory) are not overwritten by argparse defaults when the flags are not typed, and that a required argument may come from the YAML file.

        Parameters:
            tmp_path (Path): Temporary directory holding the YAML file and acting as the base directory.

        Returns:
            None
        """
        yaml_path = tmp_path / "c.yaml"
        yaml_path.write_text(
            "grid_file: g.nc\ndata_dir: d\nvariable: t2m\nbatch_mode: true\n"
            "parallel: true\ndpi: 150\nfigure_size: [6, 5]\n"
            "output_formats: [pdf]\noutput_dir: out_yaml\nlon_min: 10\nlon_max: 20\n"
        )
        cli = MPASUnifiedCLI()
        config = self._merge(
            cli, ["--config", str(yaml_path), "--base-dir", str(tmp_path), "surface"]
        )

        assert config.batch_mode is True
        assert config.parallel is True
        assert config.dpi == 150
        assert config.figure_size == (6.0, 5.0)
        assert config.output_formats == ["pdf"]
        assert config.output_dir == "out_yaml"
        assert (config.lon_min, config.lon_max) == (10, 20)
        assert config.variable == "t2m"

    def test_explicit_flags_override_yaml(
        self: "TestPrecedence", tmp_path: Path
    ) -> None:
        """
        This test verifies that flags typed on the command line win over the YAML file, including turning booleans off with the --no-* forms.

        Parameters:
            tmp_path (Path): Temporary directory holding the YAML file and acting as the base directory.

        Returns:
            None
        """
        yaml_path = tmp_path / "c.yaml"
        yaml_path.write_text(
            "grid_file: g.nc\ndata_dir: d\nvariable: t2m\nbatch_mode: true\n"
            "parallel: true\ndpi: 150\n"
        )
        cli = MPASUnifiedCLI()
        config = self._merge(
            cli,
            [
                "--config",
                str(yaml_path),
                "--base-dir",
                str(tmp_path),
                "surface",
                "--no-batch-all",
                "--no-parallel",
                "--dpi",
                "72",
                "--variable",
                "mslp",
            ],
        )

        assert config.batch_mode is False
        assert config.parallel is False
        assert config.dpi == 72
        assert config.variable == "mslp"

    def test_defaults_do_not_include_old_indonesia_box(self: "TestPrecedence") -> None:
        """
        This test verifies that without bounds on the command line or in YAML the configuration leaves the extent unspecified (resolved from the data later) instead of the former 91..113 / -9.6..12.2 box.

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        args = cli.create_main_parser().parse_args(
            ["surface", "--grid-file", "g.nc", "--data-dir", "d", "--variable", "t2m"]
        )
        config = cli.parse_args_to_config(args)

        assert (config.lon_min, config.lon_max, config.lat_min, config.lat_max) == (
            None,
            None,
            None,
            None,
        )
        assert config.output_dir == "./output"

    def test_base_dir_applies_to_config_path(
        self: "TestPrecedence", tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        This test verifies that --base-dir confines the --config file: a relative config name resolves against the current directory (here the base directory) and a path outside the base directory is rejected.

        Parameters:
            tmp_path (Path): Temporary directory acting as the base directory.
            monkeypatch (pytest.MonkeyPatch): Used to run from the base directory.

        Returns:
            None
        """
        (tmp_path / "rel.yaml").write_text(
            "grid_file: g.nc\ndata_dir: d\nvariable: t2m\n"
        )
        monkeypatch.chdir(tmp_path)
        cli = MPASUnifiedCLI()

        config = self._merge(
            cli, ["--config", "rel.yaml", "--base-dir", str(tmp_path), "surface"]
        )
        assert config.variable == "t2m"

        with pytest.raises(ValueError):
            self._merge(
                cli,
                ["--config", "/etc/hosts.yaml", "--base-dir", str(tmp_path), "surface"],
            )

    def test_missing_required_argument_exits_with_usage_error(
        self: "TestPrecedence",
    ) -> None:
        """
        This test verifies that a required argument missing from both the command line and the YAML file is reported as an argparse-style error (exit status 2).

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        with pytest.raises(SystemExit) as exc:
            self._merge(cli, ["surface", "--data-dir", "d"])
        assert exc.value.code == 2

    def test_main_reports_config_error_through_logger(
        self: "TestPrecedence", tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """
        This test verifies that an invalid configuration value is reported as a clear validation failure with exit status 1 instead of an 'Unexpected error'.

        Parameters:
            capsys (pytest.CaptureFixture): Fixture capturing standard output.
            tmp_path (Path): Temporary directory used as the base directory.

        Returns:
            None
        """
        argv = [
            "mpasdiag", "--base-dir", str(tmp_path), "surface", "--grid-file", "g.nc",
            "--data-dir", "d", "--variable", "t2m", "--lat-min", "50", "--lat-max", "10",
        ]  # fmt: skip
        with patch.object(sys, "argv", argv):
            rc = MPASUnifiedCLI().main()
        out = capsys.readouterr().out

        assert rc == 1
        assert "lat_max (10.0) must be greater than lat_min (50.0)" in out
        assert "Unexpected error" not in out


class TestTimeSelection:
    """--time-range, out-of-range --time-index, and lookback (findings 3, 7, 8)."""

    def test_time_range_parsed_and_implies_batch(self: "TestTimeSelection") -> None:
        """
        This test verifies that --time-range is mapped into time_start/time_end and implies batch mode.

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        args = cli.create_main_parser().parse_args(
            [
                "surface",
                "--grid-file",
                "g",
                "--data-dir",
                "d",
                "--variable",
                "t2m",
                "--time-range",
                "2",
                "4",
            ]
        )
        config = cli.parse_args_to_config(args)

        assert (config.time_start, config.time_end, config.batch_mode) == (2, 4, True)

    def test_resolve_time_indices_is_inclusive_and_validated(
        self: "TestTimeSelection",
    ) -> None:
        """
        This test verifies that the batch index list honors the inclusive range, defaults to the whole dataset, drops indices below the lookback with the batch starting at the first usable one, and raises IndexError for a range outside the data.

        Parameters:
            None

        Returns:
            None
        """
        cli = _cli()
        ds = _dataset(7)

        assert cli._resolve_time_indices(ds, MPASConfig(time_start=1, time_end=3)) == [
            1,
            2,
            3,
        ]
        assert cli._resolve_time_indices(ds, MPASConfig()) == list(range(7))
        assert cli._resolve_time_indices(ds, MPASConfig(), first_index=3) == [
            3,
            4,
            5,
            6,
        ]
        assert (
            cli._resolve_time_indices(ds, MPASConfig(time_start=0, time_end=2), 3) == []
        )
        with pytest.raises(IndexError, match="out of range"):
            cli._resolve_time_indices(ds, MPASConfig(time_start=2, time_end=9))

    def test_out_of_range_time_index_is_an_error(self: "TestTimeSelection") -> None:
        """
        This test verifies that an out-of-range time index raises IndexError naming the valid range instead of being clamped, and that run_analysis converts it into a failure.

        Parameters:
            None

        Returns:
            None
        """
        cli = _cli()
        with pytest.raises(IndexError, match="valid indices 0 to 6"):
            cli._check_time_index(_dataset(7), 99)

        config = MPASConfig(analysis_type="surface", time_index=99)
        with patch.object(
            cli,
            "_dispatch_analysis",
            side_effect=IndexError("time index 99 is out of range"),
        ):
            assert cli.run_analysis(config) is False

    def test_insufficient_lookback_single_step(self: "TestTimeSelection") -> None:
        """
        This test verifies that a single precipitation step too early for the accumulation raises InsufficientLookbackError and that run_analysis reports it as a failure.

        Parameters:
            None

        Returns:
            None
        """
        cli = _cli()
        config = MPASConfig(analysis_type="precipitation", accumulation_period="a03h")

        with pytest.raises(
            InsufficientLookbackError, match="first usable time index is 3"
        ):
            cli._check_lookback(config, 1)
        cli._check_lookback(config, 3)

        with patch.object(
            cli,
            "_dispatch_analysis",
            side_effect=InsufficientLookbackError("too early"),
        ):
            assert cli.run_analysis(config) is False

    def test_parallel_precipitation_receives_range_and_options(
        self: "TestTimeSelection", tmp_path: Path
    ) -> None:
        """
        This test verifies that the parallel precipitation call receives the lookback-trimmed time indices, resolved bounds, and the colormap, color limits, title, and projection options.

        Parameters:
            tmp_path (Path): Temporary directory used as the output directory.

        Returns:
            None
        """
        fake = MagicMock()
        fake.dataset = _dataset(8)
        fake.extract_2d_coordinates_for_variable.return_value = (
            np.array([-180.0, 180.0]),
            np.array([-90.0, 90.0]),
        )
        config = MPASConfig(
            analysis_type="precipitation",
            output_dir=str(tmp_path),
            parallel=True,
            batch_mode=True,
            time_start=0,
            time_end=5,
            colormap="Blues",
            clim_min=0.0,
            clim_max=5.0,
            title="T {x}",
            projection="Robinson",
            central_longitude=10.0,
            dpi=80,
        )
        cli = _cli()
        target = (
            "mpasdiag.processing.cli_unified.ParallelPrecipitationProcessor."
            "create_batch_precipitation_maps_parallel"
        )
        result = BatchFiles(["x.png"])
        result.successful = 1

        with patch(target, return_value=result) as mocked:
            assert cli._run_precipitation_batch(fake, MagicMock(), config) is True

        kwargs: Dict[str, Any] = mocked.call_args.kwargs
        assert kwargs["time_indices"] == [1, 2, 3, 4, 5]
        assert tuple(mocked.call_args.args[2]) == (-180.0, 180.0, -90.0, 90.0)
        assert kwargs["style"].colormap == "Blues"
        assert kwargs["style"].custom_title_template == "T {{x}}"
        assert kwargs["render"].clim_max == 5.0
        assert kwargs["render"].projection == "Robinson"
        assert kwargs["render"].central_longitude == 10.0
        assert kwargs["render"].dpi == 80

    def test_parallel_surface_and_wind_receive_options(
        self: "TestTimeSelection", tmp_path: Path
    ) -> None:
        """
        This test verifies that the parallel surface and wind calls receive the time range, colormap, color limits, levels, wind level, background colormap, and subsample factor that the serial path honors.

        Parameters:
            tmp_path (Path): Temporary directory used as the output directory.

        Returns:
            None
        """
        fake = MagicMock()
        fake.dataset = _dataset(8)
        fake.extract_2d_coordinates_for_variable.return_value = (
            np.array([100.0, 110.0]),
            np.array([0.0, 10.0]),
        )
        config = MPASConfig(
            output_dir=str(tmp_path),
            parallel=True,
            time_start=2,
            time_end=3,
            colormap="Reds",
            clim_min=1.0,
            clim_max=2.0,
            levels=[1.0, 2.0],
            wind_level="850hPa",
            background_colormap="magma",
            subsample_factor=3,
        )
        cli = _cli()
        ok = BatchFiles(["x.png"])
        ok.successful = 1

        base = "mpasdiag.processing.cli_unified."
        with patch(
            base + "ParallelSurfaceProcessor.create_batch_surface_maps_parallel",
            return_value=ok,
        ) as surf:
            assert cli._run_surface_batch(fake, MagicMock(), config) is True
        with patch(
            base + "ParallelWindProcessor.create_batch_wind_plots_parallel",
            return_value=ok,
        ) as wind:
            assert cli._run_wind_batch(fake, MagicMock(), config) is True

        assert surf.call_args.kwargs["time_indices"] == [2, 3]
        assert surf.call_args.kwargs["style"].colormap == "Reds"
        assert surf.call_args.kwargs["style"].levels == [1.0, 2.0]
        assert surf.call_args.kwargs["render"].clim_min == 1.0
        assert tuple(surf.call_args.args[2]) == (100.0, 110.0, 0.0, 10.0)
        assert wind.call_args.kwargs["time_indices"] == [2, 3]
        assert wind.call_args.kwargs["style"].subsample == 3
        assert wind.call_args.kwargs["render"].wind_level == "850hPa"
        assert wind.call_args.kwargs["render"].background_colormap == "magma"


class TestDefaultExtent:
    """Default extent resolves from the data (decision in the task)."""

    def test_global_and_regional_extent(self: "TestDefaultExtent") -> None:
        """
        This test verifies that a global mesh resolves to the full globe, a regional mesh to its own range rounded outward, and that user-specified bounds (including dateline-crossing ones) are kept.

        Parameters:
            None

        Returns:
            None
        """
        cli = _cli()
        glon = np.linspace(-180, 180, 50)
        glat = np.linspace(-89.9, 89.9, 50)

        assert tuple(cli._resolve_extent(MPASConfig(), glon, glat)) == (
            -180.0,
            180.0,
            -90.0,
            90.0,
        )
        assert tuple(
            cli._resolve_extent(
                MPASConfig(), np.array([100.2, 110.7]), np.array([-5.5, 7.1])
            )
        ) == (100.0, 111.0, -6.0, 8.0)
        assert tuple(
            cli._resolve_extent(MPASConfig(lon_min=150, lon_max=210), glon, glat)
        )[:2] == (150, 210)
        # A single bound is completed in the data's own frame (-180..180 here), never 370
        assert tuple(cli._resolve_extent(MPASConfig(lon_min=0), glon, glat))[:2] == (
            0.0,
            180.0,
        )
        assert tuple(cli._resolve_extent(MPASConfig(lon_max=0), glon, glat))[:2] == (
            -180.0,
            0.0,
        )


class TestQuietVerbose:
    """--quiet/--verbose are mapped and honored; the total is not double counted (finding 4)."""

    def test_flags_are_mapped(self: "TestQuietVerbose") -> None:
        """
        This test verifies that --quiet and --verbose reach the configuration (each clearing the other) and that giving neither leaves quiet off.

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()
        base = ["surface", "--grid-file", "g", "--data-dir", "d", "--variable", "t"]

        quiet = cli.parse_args_to_config(parser.parse_args(["-q"] + base))
        verbose = cli.parse_args_to_config(parser.parse_args(["--verbose"] + base))
        plain = cli.parse_args_to_config(parser.parse_args(base))
        after = cli.parse_args_to_config(parser.parse_args(base + ["-q"]))

        assert (quiet.quiet, quiet.verbose) == (True, False)
        assert (verbose.quiet, verbose.verbose) == (False, True)
        assert (plain.quiet, plain.verbose) == (False, False)
        assert after.quiet is True

    def test_quiet_logging_keeps_errors_visible(self: "TestQuietVerbose") -> None:
        """
        This test verifies that quiet mode sets the ERROR threshold while still attaching a console handler, so errors are not swallowed.

        Parameters:
            None

        Returns:
            None
        """
        import logging

        cli = MPASUnifiedCLI()
        cli.setup_logging(MPASConfig(quiet=True))
        root = logging.getLogger("mpasdiag")

        assert root.level == logging.ERROR
        assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)

    def test_quiet_suppresses_stdout_and_summary(self: "TestQuietVerbose") -> None:
        """
        This test verifies that under --quiet nothing printed by the analysis, the timers, or the performance summary reaches standard output, while a verbose non-quiet run prints them.

        Parameters:
            None

        Returns:
            None
        """
        cli = _cli()

        def noisy(config: MPASConfig) -> bool:
            print("NOISE")
            return True

        for quiet, expect in ((True, False), (False, True)):
            config = MPASConfig(analysis_type="surface", quiet=quiet, verbose=not quiet)
            buf = io.StringIO()
            with patch.object(
                cli, "_dispatch_analysis", side_effect=lambda t, c: noisy(c)
            ):
                with contextlib.redirect_stdout(buf):
                    assert cli.run_analysis(config) is True
            assert ("NOISE" in buf.getvalue()) is expect
            assert ("Performance Summary" in buf.getvalue()) is expect

    def test_verbose_output_not_printed_when_quiet(self: "TestQuietVerbose") -> None:
        """
        This test verifies that system information and the configuration summary are not printed under --quiet or in the default non-verbose mode.

        Parameters:
            None

        Returns:
            None
        """
        cli = _cli()
        with (
            patch.object(cli, "_print_system_info") as sysinfo,
            patch.object(cli, "_print_config_summary") as summary,
        ):
            cli._print_verbose_output(MPASConfig(quiet=True))
            cli._print_verbose_output(MPASConfig())
            assert not sysinfo.called and not summary.called
            cli._print_verbose_output(MPASConfig(verbose=True))
            assert sysinfo.called and summary.called

    def test_total_time_counts_only_top_level_timers(self: "TestQuietVerbose") -> None:
        """
        This test verifies that the total reported by the performance monitor does not double count timers nested inside another timer.

        Parameters:
            None

        Returns:
            None
        """
        monitor = PerformanceMonitor(quiet=True)
        with monitor.timer("outer"):
            with monitor.timer("inner"):
                pass
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            monitor.print_summary()
        total = float(buf.getvalue().split("Total time: ")[1].split(" ")[0])

        assert total == pytest.approx(
            monitor.durations["outer"].total_seconds(), abs=0.01
        )


class TestEntryPoints:
    """python -m mpasdiag and mpasdiag.cli propagate the exit status (finding 5)."""

    @pytest.mark.parametrize("module", ["mpasdiag", "mpasdiag.cli"])
    def test_module_entry_points_exit_with_main_return_code(
        self: "TestEntryPoints", module: str
    ) -> None:
        """
        This test runs each module as a script with main() patched to return 3 and verifies the process would exit with status 3 rather than 0.

        Parameters:
            module (str): The module executed as __main__.

        Returns:
            None
        """
        with patch("mpasdiag.processing.cli_unified.main", return_value=3):
            with pytest.raises(SystemExit) as exc:
                runpy.run_module(module, run_name="__main__", alter_sys=True)

        assert exc.value.code == 3


class TestSoundingBatch:
    """The sounding batch isolates failures and honors --formats, --output, --time-range (finding 12)."""

    def test_sounding_batch_counts_failures_and_uses_range(
        self: "TestSoundingBatch", tmp_path: Path
    ) -> None:
        """
        This test runs the sounding batch with a diagnostics stub that fails for one time index and verifies that the other steps still run, the failure makes the batch fail, the time range limits the steps, and every requested format is saved with the user-supplied name plus the time tag.

        Parameters:
            tmp_path (Path): Temporary directory used as the output directory.

        Returns:
            None
        """
        processor = MagicMock()
        processor.dataset = _dataset(6)
        diag = MagicMock()
        profile = {
            "pressure": np.array([1000.0]), "temperature": np.array([20.0]),
            "dewpoint": np.array([10.0]), "u_wind": np.array([1.0]),
            "v_wind": np.array([1.0]), "station_lon": -105.0, "station_lat": 40.0,
        }  # fmt: skip

        def extract(*a: Any, time_index: int = 0, **k: Any) -> Dict[str, Any]:
            if time_index == 2:
                raise RuntimeError("bad profile")
            return profile

        diag.extract_sounding_profile.side_effect = extract
        plotter = MagicMock()
        config = MPASConfig(
            output_dir=str(tmp_path), output="snd", output_formats=["png", "pdf"],
            batch_mode=True, time_start=1, time_end=3,
        )  # fmt: skip

        ok = _cli()._run_sounding_batch(
            processor, plotter, diag, config, -105.0, 40.0, False, False
        )

        assert ok is False
        assert diag.extract_sounding_profile.call_count == 3
        assert plotter.save_plot.call_count == 2
        first = plotter.save_plot.call_args_list[0]
        assert first.kwargs["formats"] == ["png", "pdf"]
        assert first.args[0].endswith("snd_20240917T01")


class TestConfigValidation:
    """MPASConfig validation (finding 11)."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"lon_min": 0, "lon_max": 360},
            {"lon_min": 150, "lon_max": 210},
            {"lat_min": -90, "lat_max": 90},
            {"subsample_factor": -1},
            {"subsample_factor": 4},
            {"lat_min": 10},
            {},
        ],
    )
    def test_valid_configs(
        self: "TestConfigValidation", kwargs: Dict[str, Any]
    ) -> None:
        """
        This test verifies that the extents the plotters accept (0 to 360, dateline-crossing, partially specified) and the documented subsample values are accepted.

        Parameters:
            kwargs (Dict[str, Any]): Configuration keyword arguments.

        Returns:
            None
        """
        MPASConfig(**kwargs)

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"lon_min": -361, "lon_max": 0}, "lon_min=-361"),
            ({"lon_min": -300, "lon_max": 100}, "span"),
            ({"lon_min": 10, "lon_max": 5}, "lon_max \\(5\\)"),
            ({"lat_min": 20, "lat_max": 10}, "lat_max \\(10\\)"),
            ({"lat_min": -91, "lat_max": 0}, "lat_min=-91"),
            ({"clim_min": 5.0, "clim_max": 1.0}, "clim_min"),
            ({"transparency": 1.5}, "transparency"),
            ({"vector_alpha": -0.1}, "vector_alpha"),
            ({"subsample_factor": -2}, "subsample_factor"),
            ({"accumulation_period": "a99h"}, "accumulation_period"),
            ({"plot_type": "bogus"}, "plot_type"),
            ({"analysis_type": "precipitation", "plot_type": "contour"}, "plot_type"),
            ({"output_formats": ["gif"]}, "output format"),
            ({"time_start": -1}, "time_start"),
            ({"time_start": 5, "time_end": 2}, "time_end"),
        ],
    )
    def test_invalid_configs(
        self: "TestConfigValidation", kwargs: Dict[str, Any], match: str
    ) -> None:
        """
        This test verifies that invalid extents, color limits, alpha values, subsample factors, categorical choices, and time selections are rejected with a message naming the offending value.

        Parameters:
            kwargs (Dict[str, Any]): Configuration keyword arguments.
            match (str): Regular expression the error message must match.

        Returns:
            None
        """
        with pytest.raises(ValueError, match=match):
            MPASConfig(**kwargs)

    def test_cli_subsample_zero_maps_to_auto(self: "TestConfigValidation") -> None:
        """
        This test verifies that the documented legacy --subsample 0 is mapped to the automatic value -1 and that the default is automatic.

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()
        base = ["wind", "--grid-file", "g", "--data-dir", "d"]

        assert (
            cli.parse_args_to_config(
                parser.parse_args(base + ["--subsample", "0"])
            ).subsample_factor
            == -1
        )
        assert cli.parse_args_to_config(parser.parse_args(base)).subsample_factor == -1

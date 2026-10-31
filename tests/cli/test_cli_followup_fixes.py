#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""
MPASdiag Test Suite: Regression tests for the second round of CLI and parallel-wrapper fixes.

This module covers the valid time in serial and parallel wind titles, the wind vector color and opacity threading, the identical serial and parallel cross-section file names, the surface and wind parity of the parallel workers, the outcome-aware timing messages, the explicit --no-batch-all precedence over a YAML time range, and the display-unit documentation of the color limits.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: 2026
Version: 1.0.0
"""

import io
import contextlib
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

import mpasdiag.processing.parallel_wrappers as pw
from mpasdiag.processing.cli_unified import MPASUnifiedCLI
from mpasdiag.processing.utils_config import MPASConfig
from mpasdiag.processing.utils_logger import MPASLogger


def _cli() -> MPASUnifiedCLI:
    """
    This helper creates a CLI instance with a logger attached.

    Parameters:
        None

    Returns:
        MPASUnifiedCLI: A CLI instance with logging configured.
    """
    cli = MPASUnifiedCLI()
    cli.logger = MPASLogger("test_followup", verbose=False).logger
    return cli


def _wind_dataset(n_cells: int = 30, n_times: int = 3) -> xr.Dataset:
    """
    This helper builds a small wind dataset with an hourly Time coordinate.

    Parameters:
        n_cells (int): Number of cells.
        n_times (int): Number of time steps.

    Returns:
        xr.Dataset: Dataset with u10, v10 and a Time coordinate.
    """
    times = np.array(
        [
            np.datetime64("2024-09-17T00:00") + np.timedelta64(i, "h")
            for i in range(n_times)
        ]
    )
    rng = np.random.default_rng(3)
    return xr.Dataset(
        {
            "u10": (("Time", "nCells"), rng.normal(0, 5, (n_times, n_cells))),
            "v10": (("Time", "nCells"), rng.normal(0, 5, (n_times, n_cells))),
        },
        coords={"Time": times},
    )


class TestWindTitleTime:
    """The serial and parallel wind plots carry the valid time and vector styling."""

    def test_serial_wind_passes_time_stamp_and_vector_style(
        self: "TestWindTitleTime", tmp_path: Path
    ) -> None:
        """
        This test verifies the serial wind step builds a WindPlotStyle holding the valid time of the selected index and the configured vector color and opacity.

        Parameters:
            tmp_path (Path): Temporary output directory.

        Returns:
            None
        """
        ds = _wind_dataset()
        processor = MagicMock()
        processor.dataset = ds
        processor.get_2d_variable_data.side_effect = lambda var, idx: ds[var].isel(
            Time=idx
        )
        processor.extract_2d_coordinates_for_variable.return_value = (
            np.linspace(-120, -80, 30),
            np.linspace(30, 50, 30),
        )
        plotter = MagicMock()
        plotter.create_wind_plot.return_value = (MagicMock(), MagicMock())
        plotter.create_surface_map.return_value = (MagicMock(), MagicMock())
        plotter.create_vertical_cross_section.return_value = (
            MagicMock(),
            MagicMock(),
        )
        config = MPASConfig(
            grid_file="g",
            data_dir="d",
            output_dir=str(tmp_path),
            analysis_type="wind",
            u_variable="u10",
            v_variable="v10",
            lon_min=-120,
            lon_max=-80,
            lat_min=30,
            lat_max=50,
            vector_color="red",
            vector_alpha=0.4,
        )

        _cli()._run_wind_single(processor, plotter, config, 2)

        style = plotter.create_wind_plot.call_args.kwargs["style"]
        assert style.time_stamp.strftime("%Y-%m-%d %H:%M") == "2024-09-17 02:00"
        assert style.vector_color == "red"
        assert style.vector_alpha == 0.4

    def test_parallel_wind_worker_passes_time_stamp_and_vector_style(
        self: "TestWindTitleTime", tmp_path: Path
    ) -> None:
        """
        This test verifies the parallel wind worker passes the valid time, the vector styling, and does not add a second branding footer.

        Parameters:
            tmp_path (Path): Temporary output directory.

        Returns:
            None
        """
        ds = _wind_dataset()
        processor = MagicMock()
        processor.dataset = ds
        processor.get_2d_variable_data.side_effect = lambda var, idx: ds[var].isel(
            Time=idx
        )
        processor.extract_2d_coordinates_for_variable.return_value = (
            np.linspace(-120, -80, 30),
            np.linspace(30, 50, 30),
        )
        plotter = MagicMock()
        plotter.create_wind_plot.return_value = (MagicMock(), MagicMock())
        plotter.create_surface_map.return_value = (MagicMock(), MagicMock())
        plotter.create_vertical_cross_section.return_value = (
            MagicMock(),
            MagicMock(),
        )
        kwargs: Dict[str, Any] = {
            "processor": processor,
            "cache": None,
            "output_dir": str(tmp_path),
            "lon_min": -120,
            "lon_max": -80,
            "lat_min": 30,
            "lat_max": 50,
            "u_variable": "u10",
            "v_variable": "v10",
            "formats": ["png"],
            "render": pw.BatchRenderOptions(vector_color="red", vector_alpha=0.4),
        }

        with patch.object(pw, "MPASWindPlotter", return_value=plotter):
            pw._wind_worker((1, kwargs))

        style = plotter.create_wind_plot.call_args.kwargs["style"]
        assert style.time_stamp.strftime("%Y-%m-%d %H:%M") == "2024-09-17 01:00"
        assert style.vector_color == "red"
        assert style.vector_alpha == 0.4
        plotter.add_timestamp_and_branding.assert_not_called()

    def test_render_options_carry_vector_style(self: "TestWindTitleTime") -> None:
        """
        This test verifies the CLI forwards the configured vector color and opacity to the parallel render options.

        Parameters:
            None

        Returns:
            None
        """
        config = MPASConfig(vector_color="blue", vector_alpha=0.25)
        render = _cli()._render_options(config)
        assert render.vector_color == "blue"
        assert render.vector_alpha == 0.25

    def test_vector_alpha_default_is_opaque(self: "TestWindTitleTime") -> None:
        """
        This test verifies the configuration and command-line defaults keep the vectors opaque, matching what was drawn before the options took effect.

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()
        args = parser.parse_args(["wind", "--grid-file", "g", "--data-dir", "d"])
        config = cli.parse_args_to_config(args)
        assert config.vector_alpha == 1.0
        assert config.vector_color == "black"
        explicit = parser.parse_args(
            ["wind", "--grid-file", "g", "--data-dir", "d", "--vector-alpha", "0.3"]
        )
        assert cli.parse_args_to_config(explicit).vector_alpha == 0.3


class TestSurfaceWorkerParity:
    """The parallel surface worker renders the same figure as the serial path."""

    def test_surface_worker_title_and_data_array(
        self: "TestSurfaceWorkerParity", tmp_path: Path
    ) -> None:
        """
        This test verifies the parallel surface worker uses the serial default title, passes no separate time stamp (so no extra Valid box is drawn), and hands over the DataArray for the colorbar label.

        Parameters:
            tmp_path (Path): Temporary output directory.

        Returns:
            None
        """
        ds = _wind_dataset().rename({"u10": "t2m"})
        processor = MagicMock()
        processor.dataset = ds
        processor.extract_spatial_coordinates.return_value = (
            np.linspace(-120, -80, 30),
            np.linspace(30, 50, 30),
        )
        # the surface worker reads data and coordinates like the serial path
        processor.extract_2d_coordinates_for_variable.return_value = (
            processor.extract_spatial_coordinates.return_value
        )
        processor.get_2d_variable_data.side_effect = (
            lambda var, t, _ds=processor.dataset: _ds[var].isel(Time=t)
        )
        plotter = MagicMock()
        plotter.create_wind_plot.return_value = (MagicMock(), MagicMock())
        plotter.create_surface_map.return_value = (MagicMock(), MagicMock())
        plotter.create_vertical_cross_section.return_value = (
            MagicMock(),
            MagicMock(),
        )
        kwargs: Dict[str, Any] = {
            "processor": processor,
            "cache": None,
            "output_dir": str(tmp_path),
            "lon_min": -120,
            "lon_max": -80,
            "lat_min": 30,
            "lat_max": 50,
            "var_name": "t2m",
            "plot_type": "scatter",
            "file_prefix": "mpas_surface",
            "formats": ["png"],
        }

        with patch.object(pw, "MPASSurfacePlotter", return_value=plotter):
            pw._surface_worker((2, kwargs))

        call = plotter.create_surface_map.call_args.kwargs
        assert (
            call["style"].title == "MPAS Surface | Variable: t2m | Valid: 20240917T02"
        )
        assert call.get("time_stamp") is None
        assert call["data_array"] is not None


class TestCrossSectionFileName:
    """The serial cross-section file name matches the parallel worker's."""

    def test_serial_name_contains_vertical_coordinate(
        self: "TestCrossSectionFileName", tmp_path: Path
    ) -> None:
        """
        This test verifies the serial cross-section output name has the same mpas_cross_section_<var>_vcrd_<coord>_valid_<time> form as the parallel worker.

        Parameters:
            tmp_path (Path): Temporary output directory.

        Returns:
            None
        """
        ds = _wind_dataset().rename({"u10": "theta"})
        processor = MagicMock()
        processor.dataset = ds
        plotter = MagicMock()
        plotter.create_wind_plot.return_value = (MagicMock(), MagicMock())
        plotter.create_surface_map.return_value = (MagicMock(), MagicMock())
        plotter.create_vertical_cross_section.return_value = (
            MagicMock(),
            MagicMock(),
        )
        config = MPASConfig(
            output_dir=str(tmp_path),
            variable="theta",
            vertical_coord="pressure",
            start_lon=-100,
            start_lat=30,
            end_lon=-90,
            end_lat=40,
        )
        params = _cli()._extract_cross_section_params(config)

        path = _cli()._run_single_cross_section(processor, plotter, config, params, 2)

        assert (
            Path(path).name
            == "mpas_cross_section_theta_vcrd_pressure_valid_20240917T02"
            "_s-100_30_e-90_40"
        )


class TestOutcomeAwareTimingMessages:
    """Failing runs never print that the analysis completed."""

    def _run(self: "TestOutcomeAwareTimingMessages", outcome: Any) -> str:
        """
        This helper runs run_analysis with a patched dispatcher and captures standard output.

        Parameters:
            outcome (Any): Return value, or exception instance to raise, from the patched dispatcher.

        Returns:
            str: Captured standard output.
        """
        cli = _cli()
        config = MPASConfig(analysis_type="surface")
        buffer = io.StringIO()

        def dispatch(*_a: Any, **_k: Any) -> Any:
            """
            This stub returns or raises the configured outcome.

            Parameters:
                *_a (Any): Ignored positional arguments.
                **_k (Any): Ignored keyword arguments.

            Returns:
                Any: The configured outcome.
            """
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with patch.object(cli, "_dispatch_analysis", side_effect=dispatch):
            with contextlib.redirect_stdout(buffer):
                cli.run_analysis(config)

        return buffer.getvalue()

    def test_success_prints_completed(self: "TestOutcomeAwareTimingMessages") -> None:
        """
        This test verifies a successful run still prints the completion line.

        Parameters:
            None

        Returns:
            None
        """
        out = self._run(True)
        assert "Total analysis completed in" in out
        assert "failed" not in out

    @pytest.mark.parametrize("outcome", [False, IndexError("bad"), RuntimeError("x")])
    def test_failure_prints_failed(
        self: "TestOutcomeAwareTimingMessages", outcome: Any
    ) -> None:
        """
        This test verifies a failing run (false result or an exception) prints a failure line instead of a completion line.

        Parameters:
            outcome (Any): The dispatcher outcome.

        Returns:
            None
        """
        out = self._run(outcome)
        assert "completed" not in out
        assert "Total analysis failed after" in out


class TestNoBatchAllPrecedence:
    """An explicit --no-batch-all wins over a YAML time range."""

    def _merge(self: "TestNoBatchAllPrecedence", tmp_path: Path, argv: Any) -> Any:
        """
        This helper merges a YAML file holding a time_start with command-line flags.

        Parameters:
            tmp_path (Path): Directory for the YAML file.
            argv (Any): Extra command-line arguments.

        Returns:
            Any: The merged MPASConfig.
        """
        yaml_path = tmp_path / "c.yaml"
        yaml_path.write_text(
            "grid_file: g.nc\ndata_dir: d\nvariable: t2m\ntime_start: 2\n"
        )
        cli = self.last_cli = _cli()
        parser = cli.create_main_parser()
        args = parser.parse_args(
            ["--config", str(yaml_path), "--base-dir", str(tmp_path), "surface"]
            + list(argv)
        )
        return cli._load_and_merge_config(args)

    def test_yaml_time_start_implies_batch(
        self: "TestNoBatchAllPrecedence", tmp_path: Path
    ) -> None:
        """
        This test verifies a YAML time_start still implies batch mode when no explicit flag is given.

        Parameters:
            tmp_path (Path): Temporary directory.

        Returns:
            None
        """
        assert self._merge(tmp_path, []).batch_mode is True

    def test_explicit_no_batch_all_wins(
        self: "TestNoBatchAllPrecedence", tmp_path: Path
    ) -> None:
        """
        This test verifies --no-batch-all disables batch mode and drops the YAML time range.

        Parameters:
            tmp_path (Path): Temporary directory.

        Returns:
            None
        """
        config = self._merge(tmp_path, ["--no-batch-all", "--time-index", "1"])
        assert config.batch_mode is False
        assert config.time_start is None
        assert config.time_index == 1
        assert any(
            "ignoring the YAML time range" in note
            for note in self.last_cli._config_notes
        )

    def test_explicit_time_range_still_batches(
        self: "TestNoBatchAllPrecedence", tmp_path: Path
    ) -> None:
        """
        This test verifies an explicit --time-range keeps implying batch mode.

        Parameters:
            tmp_path (Path): Temporary directory.

        Returns:
            None
        """
        config = self._merge(tmp_path, ["--time-range", "1", "3"])
        assert config.batch_mode is True
        assert (config.time_start, config.time_end) == (1, 3)


class TestClimDocumentation:
    """The color limits are documented as display units."""

    def test_clim_help_and_docstring_mention_display_units(
        self: "TestClimDocumentation",
    ) -> None:
        """
        This test verifies the --clim-min/--clim-max help text and the MPASConfig docstring state the limits are in plotted units.

        Parameters:
            None

        Returns:
            None
        """
        parser = MPASUnifiedCLI().create_main_parser()
        sub = next(
            a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction"
        )
        helps = [
            act.help
            for act in sub.choices["surface"]._actions
            if "--clim-min" in act.option_strings or "--clim-max" in act.option_strings
        ]
        assert helps and all("display" in h for h in helps)
        assert "display" in (MPASConfig.__doc__ or "")

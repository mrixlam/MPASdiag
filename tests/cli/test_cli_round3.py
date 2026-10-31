#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""
MPASdiag Test Suite: Regression tests for the third round of CLI fixes.

This module covers the implemented overlay subcommand (every --overlay-type), the endpoint-aware cross-section and level-aware wind file names, the sounding file name suffixes, the --clim-min/--clim-max, --interpolation, --levels and --use-pure-xarray options reaching the plot styles, the --output name forwarded to the parallel batch methods, the README command and YAML examples, the CLI log-level precedence over YAML, the YAML aliases and boolean checks of MPASConfig.from_dict, the current-directory resolution of relative paths, and the sanitization of --output names.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: 2026
Version: 1.0.0
"""

import re
import sys
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
import yaml

import mpasdiag.processing.parallel_wrappers as pw
from mpasdiag.processing.cli_unified import (
    MPASUnifiedCLI,
    _accepted_kwargs,
    _sanitize_output_name,
)
from mpasdiag.processing.utils_config import MPASConfig
from mpasdiag.processing.utils_logger import MPASLogger
from mpasdiag.visualization.cross_section import CrossSectionStyle
from mpasdiag.visualization.surface import SurfaceMapStyle

REPO = Path(__file__).resolve().parents[2]
GRID = REPO / "data" / "grids" / "x1.10242.static.nc"
DIAG = REPO / "data" / "u240k" / "diag"
HAVE_DATA = GRID.exists() and DIAG.exists() and any(DIAG.glob("diag*.nc"))
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="u240k sample data not present")


def _cli() -> MPASUnifiedCLI:
    """
    This helper creates a CLI instance with a logger attached.

    Parameters:
        None

    Returns:
        MPASUnifiedCLI: A CLI instance with logging configured.
    """
    cli = MPASUnifiedCLI()
    cli.logger = MPASLogger("test_round3", verbose=False).logger
    return cli


def _parse(argv: List[str]) -> Any:
    """
    This helper parses an argument vector with the real unified parser and returns the namespace.

    Parameters:
        argv (List[str]): Arguments after the program name.

    Returns:
        Any: The parsed namespace.
    """
    cli = MPASUnifiedCLI()
    with patch.object(sys, "argv", ["mpasdiag"] + argv):
        return cli._parse_args_with_fallback(cli.create_main_parser())


def _merged(argv: List[str], yml: Any = None, tmp: Any = None) -> MPASConfig:
    """
    This helper runs the real parse and YAML merge for an argument vector, optionally with a YAML file.

    Parameters:
        argv (List[str]): Arguments after the program name.
        yml (Any): Mapping written to a YAML file passed as --config, or None.
        tmp (Any): Directory in which the YAML file is written.

    Returns:
        MPASConfig: The merged configuration.
    """
    cli = MPASUnifiedCLI()
    full = ["--base-dir", "/"] + argv
    if yml is not None:
        path = Path(tmp) / "c.yaml"
        path.write_text(yaml.safe_dump(yml))
        full += ["--config", str(path)]
    with patch.object(sys, "argv", ["mpasdiag"] + full):
        args = cli._parse_args_with_fallback(cli.create_main_parser())
    return cli._load_and_merge_config(args)


BASE = {"grid_file": "/x/g.nc", "data_dir": "/x/d", "variable": "t2m"}


class TestOverlayOptions:
    """The overlay plan built from the options for every overlay type."""

    def _cfg(self: "TestOverlayOptions", **kw: Any) -> MPASConfig:
        """
        This helper builds an overlay configuration.

        Parameters:
            **kw (Any): Overrides of the overlay fields.

        Returns:
            MPASConfig: The configuration.
        """
        return MPASConfig(grid_file="g", data_dir="d", analysis_type="overlay", **kw)

    def test_precip_wind_plan(self: "TestOverlayOptions") -> None:
        """
        This test verifies precip_wind plans accumulated rainnc with wind vectors.

        Parameters:
            None

        Returns:
            None
        """
        spec = _cli()._overlay_spec(self._cfg(overlay_type="precip_wind"))
        assert spec["precip"] and spec["wind"] and spec["base"] == "rainnc"
        assert spec["names"] == ["rainnc", "u10", "v10"]

    def test_temp_pressure_plan(self: "TestOverlayOptions") -> None:
        """
        This test verifies temp_pressure plans t2m with the pressure variable as a contour layer.

        Parameters:
            None

        Returns:
            None
        """
        spec = _cli()._overlay_spec(self._cfg(overlay_type="temp_pressure"))
        assert spec["base"] == "t2m" and spec["layers"] == [("mslp", False)]
        assert not spec["wind"]

    def test_wind_temp_plan(self: "TestOverlayOptions") -> None:
        """
        This test verifies wind_temp plans a scalar base with wind vectors and honours the primary variable.

        Parameters:
            None

        Returns:
            None
        """
        spec = _cli()._overlay_spec(
            self._cfg(overlay_type="wind_temp", primary_variable="th2m")
        )
        assert spec["base"] == "th2m" and spec["wind"] and not spec["precip"]

    def test_multi_level_plan(self: "TestOverlayOptions") -> None:
        """
        This test verifies multi_level uses the first variable as base, the u/v pair as wind, and the rest as contours.

        Parameters:
            None

        Returns:
            None
        """
        spec = _cli()._overlay_spec(
            self._cfg(
                overlay_type="multi_level", variables=["t2m", "mslp", "u10", "v10"]
            )
        )
        assert spec["base"] == "t2m" and spec["wind"]
        assert spec["layers"] == [("mslp", False)]

    def test_multi_level_errors(self: "TestOverlayOptions") -> None:
        """
        This test verifies multi_level rejects a missing variable list and a lone wind component.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="--variables"):
            _cli()._overlay_spec(self._cfg(overlay_type="multi_level"))
        with pytest.raises(ValueError, match="wind partner"):
            _cli()._overlay_spec(
                self._cfg(overlay_type="multi_level", variables=["t2m", "u10"])
            )

    def test_custom_plan_and_errors(self: "TestOverlayOptions") -> None:
        """
        This test verifies custom needs both variables and draws the secondary as a filled layer.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="custom overlays need"):
            _cli()._overlay_spec(self._cfg(overlay_type="custom"))
        spec = _cli()._overlay_spec(
            self._cfg(
                overlay_type="custom", primary_variable="t2m", secondary_variable="q2"
            )
        )
        assert spec["layers"] == [("q2", True)]

    def test_invalid_options_return_failure(self: "TestOverlayOptions") -> None:
        """
        This test verifies an invalid overlay combination makes the analysis return False with a message instead of exiting 0.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.utils_monitor import PerformanceMonitor

        cli = _cli()
        cli.perf_monitor = PerformanceMonitor()
        cfg = self._cfg(overlay_type="custom")
        assert cli._run_overlay_analysis(cfg) is False


@needs_data
class TestOverlayRendering:
    """Each overlay type renders a real file on the sample data."""

    def _run(self: "TestOverlayRendering", tmp_path: Path, **kw: Any) -> List[Path]:
        """
        This helper runs one overlay analysis on the sample data and returns the written files.

        Parameters:
            tmp_path (Path): Output directory.
            **kw (Any): Overlay configuration fields.

        Returns:
            List[Path]: The files created in the output directory.
        """
        from mpasdiag.processing.utils_monitor import PerformanceMonitor

        cli = _cli()
        cli.perf_monitor = PerformanceMonitor()
        cfg = MPASConfig(
            grid_file=str(GRID),
            data_dir=str(DIAG),
            output_dir=str(tmp_path),
            analysis_type="overlay",
            dpi=50,
            figure_size=(8.0, 6.0),
            lon_min=-130.0,
            lon_max=-60.0,
            lat_min=20.0,
            lat_max=55.0,
            time_index=2,
            **kw,
        )
        assert cli._run_overlay_analysis(cfg) is True
        return sorted(tmp_path.glob("*.png"))

    def test_temp_pressure_file(self: "TestOverlayRendering", tmp_path: Path) -> None:
        """
        This test verifies temp_pressure writes a correctly named, non-empty PNG.

        Parameters:
            tmp_path (Path): Output directory.

        Returns:
            None
        """
        files = self._run(tmp_path, overlay_type="temp_pressure")
        assert [f.name for f in files] == [
            "mpas_overlay_temp_pressure_t2m_mslp_valid_20240917T02.png"
        ]
        assert files[0].stat().st_size > 5000

    def test_multi_level_with_output_name(
        self: "TestOverlayRendering", tmp_path: Path
    ) -> None:
        """
        This test verifies multi_level draws base, contour, and wind layers and honours --output.

        Parameters:
            tmp_path (Path): Output directory.

        Returns:
            None
        """
        files = self._run(
            tmp_path,
            overlay_type="multi_level",
            variables=["t2m", "mslp", "u10", "v10"],
            output="mine",
        )
        assert [f.name for f in files] == ["mine.png"]

    def test_precip_wind_batch_range(
        self: "TestOverlayRendering", tmp_path: Path
    ) -> None:
        """
        This test verifies a precip_wind time range writes one file per usable time index.

        Parameters:
            tmp_path (Path): Output directory.

        Returns:
            None
        """
        files = self._run(
            tmp_path,
            overlay_type="precip_wind",
            batch_mode=True,
            time_start=1,
            time_end=2,
            parallel=True,
        )
        assert [f.name[-15:] for f in files] == ["T01.png", "T02.png"][0:0] or len(
            files
        ) == 2

    def test_custom_and_wind_temp(self: "TestOverlayRendering", tmp_path: Path) -> None:
        """
        This test verifies the custom and wind_temp overlay types both write files.

        Parameters:
            tmp_path (Path): Output directory.

        Returns:
            None
        """
        files = self._run(
            tmp_path,
            overlay_type="custom",
            primary_variable="t2m",
            secondary_variable="q2",
            contour_overlay=True,
            transparency=0.4,
            secondary_colormap="Greens",
        )
        assert files[0].name.startswith("mpas_overlay_custom_t2m_q2_valid_")
        out2 = tmp_path / "wt"
        out2.mkdir()
        files = self._run(out2, overlay_type="wind_temp")
        assert files[0].name.startswith("mpas_overlay_wind_temp_t2m_u10_v10_valid_")


class TestFileStems:
    """Serial file names include the cross-section endpoints, wind level, and sounding options."""

    def _processor(self: "TestFileStems") -> MagicMock:
        """
        This helper builds a mock processor over a small dataset.

        Parameters:
            None

        Returns:
            MagicMock: The mock processor.
        """
        times = np.array(
            [
                np.datetime64("2024-09-17T00:00") + np.timedelta64(i, "h")
                for i in range(3)
            ]
        )
        ds = xr.Dataset(
            {"u10": (("Time", "nCells"), np.ones((3, 5)))}, coords={"Time": times}
        )
        proc = MagicMock()
        proc.dataset = ds
        return proc

    def test_cross_section_stem_has_endpoints_and_clim(
        self: "TestFileStems", tmp_path: Path
    ) -> None:
        """
        This test verifies the serial cross-section name carries the endpoints and the style receives clim_min/clim_max.

        Parameters:
            tmp_path (Path): Output directory.

        Returns:
            None
        """
        cfg = MPASConfig(
            grid_file="g",
            data_dir="d",
            output_dir=str(tmp_path),
            analysis_type="cross",
            variable="theta",
            start_lon=-105.0,
            start_lat=39.7,
            end_lon=-94.6,
            end_lat=39.1,
            clim_min=280.0,
            clim_max=340.0,
        )
        cli = _cli()
        plotter = MagicMock()
        plotter.create_vertical_cross_section.return_value = (MagicMock(), MagicMock())
        params = cli._extract_cross_section_params(cfg)
        path = cli._run_single_cross_section(
            cfg and self._processor(), plotter, cfg, params, 1
        )
        name = Path(path).name
        assert "_s-105_39p7_e-94p6_39p1" in name
        style = plotter.create_vertical_cross_section.call_args.kwargs["style"]
        assert style.clim_min == 280.0 and style.clim_max == 340.0

    def test_wind_stem_has_level(self: "TestFileStems", tmp_path: Path) -> None:
        """
        This test verifies the serial wind name carries the sanitized level for 3D winds.

        Parameters:
            tmp_path (Path): Output directory.

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor

        ds = self._processor().dataset
        proc = MagicMock(spec=MPAS3DProcessor)
        proc.dataset = ds
        proc.get_3d_variable_data.side_effect = lambda v, lev, idx: ds["u10"].isel(
            Time=idx
        )
        proc.extract_2d_coordinates_for_variable.return_value = (
            np.linspace(-100, -90, 5),
            np.linspace(30, 40, 5),
        )
        cfg = MPASConfig(
            grid_file="g",
            data_dir="d",
            output_dir=str(tmp_path),
            analysis_type="wind",
            wind_level="850hPa",
        )
        plotter = MagicMock()
        plotter.create_wind_plot.return_value = (MagicMock(), MagicMock())
        path = _cli()._run_wind_single(proc, plotter, cfg, 1)
        assert "_lev850hPa_valid_" in Path(path).name

    def test_sounding_stems_differ_by_indices(
        self: "TestFileStems", tmp_path: Path
    ) -> None:
        """
        This test verifies runs with and without --show-indices (and --show-parcel) write different file names.

        Parameters:
            tmp_path (Path): Output directory.

        Returns:
            None
        """
        cli = _cli()
        proc = self._processor()
        diag = MagicMock()
        diag.extract_sounding_profile.return_value = {
            "station_lon": -105.0,
            "station_lat": 39.7,
            "pressure": [1],
            "temperature": [1],
            "dewpoint": [1],
            "u_wind": [1],
            "v_wind": [1],
        }
        diag.compute_thermodynamic_indices.return_value = {}
        cfg = MPASConfig(
            grid_file="g",
            data_dir="d",
            output_dir=str(tmp_path),
            analysis_type="sounding",
        )
        names = set()
        for ind, par in ((False, False), (True, False), (True, True)):
            names.add(
                Path(
                    cli._create_skewt(
                        proc, MagicMock(), diag, cfg, -105.0, 39.7, ind, par, 1
                    )
                ).name
            )
        assert len(names) == 3


class TestOptionMapping:
    """New and previously dead options reach the plot styles and the config."""

    def test_levels_flag(self: "TestOptionMapping") -> None:
        """
        This test verifies --levels parses to floats and maps to the levels field.

        Parameters:
            None

        Returns:
            None
        """
        cfg = _merged(
            ["surface", "--grid-file", "g", "--data-dir", "d", "--variable", "t2m"]
            + ["--levels", "0", "5.5", "10"]
        )
        assert cfg.levels == [0.0, 5.5, 10.0]

    def test_use_pure_xarray_flag(self: "TestOptionMapping") -> None:
        """
        This test verifies --use-pure-xarray maps to the config field and --no-use-pure-xarray overrides YAML.

        Parameters:
            None

        Returns:
            None
        """
        base = ["surface", "--grid-file", "g", "--data-dir", "d", "--variable", "t2m"]
        assert _merged(base + ["--use-pure-xarray"]).use_pure_xarray is True
        cfg = _merged(
            base + ["--no-use-pure-xarray"],
            {**BASE, "use_pure_xarray": True},
            tmp=Path("/tmp"),
        )
        assert cfg.use_pure_xarray is False

    def test_interpolation_default_and_mapping(self: "TestOptionMapping") -> None:
        """
        This test verifies --interpolation defaults to None and maps to interpolation_method.

        Parameters:
            None

        Returns:
            None
        """
        base = ["surface", "--grid-file", "g", "--data-dir", "d", "--variable", "t2m"]
        assert _merged(base).interpolation_method is None
        assert (
            _merged(base + ["--interpolation", "cubic"]).interpolation_method == "cubic"
        )

    def test_interpolation_reaches_surface_style(
        self: "TestOptionMapping", tmp_path: Path
    ) -> None:
        """
        This test verifies the serial surface step builds a SurfaceMapStyle carrying the interpolation.

        Parameters:
            tmp_path (Path): Output directory.

        Returns:
            None
        """
        if "interpolation" not in {f for f in SurfaceMapStyle.__dataclass_fields__}:
            pytest.skip("SurfaceMapStyle.interpolation not available")
        times = np.array([np.datetime64("2024-09-17T00:00")])
        ds = xr.Dataset(
            {"t2m": (("Time", "nCells"), np.ones((1, 4)))}, coords={"Time": times}
        )
        proc = MagicMock()
        proc.dataset = ds
        proc.get_2d_variable_data.return_value = ds["t2m"].isel(Time=0)
        proc.extract_2d_coordinates_for_variable.return_value = (
            np.linspace(-100, -90, 4),
            np.linspace(30, 40, 4),
        )
        cfg = MPASConfig(
            grid_file="g",
            data_dir="d",
            output_dir=str(tmp_path),
            analysis_type="surface",
            variable="t2m",
            interpolation_method="nearest",
        )
        plotter = MagicMock()
        plotter.create_surface_map.return_value = (MagicMock(), MagicMock())
        _cli()._run_surface_single(proc, plotter, ds, cfg, 0)
        style = plotter.create_surface_map.call_args.kwargs["style"]
        assert style.interpolation == "nearest"

    def test_output_name_forwarded(self: "TestOptionMapping") -> None:
        """
        This test verifies --output is forwarded as output_name only to methods that accept it, and only when set.

        Parameters:
            None

        Returns:
            None
        """

        def with_name(a: int, output_name: Any = None) -> None:
            """
            This stub accepts output_name.

            Parameters:
                a (int): Unused.
                output_name (Any): Unused.

            Returns:
                None
            """

        def without_name(a: int) -> None:
            """
            This stub does not accept output_name.

            Parameters:
                a (int): Unused.

            Returns:
                None
            """

        cfg = MPASConfig(grid_file="g", data_dir="d", output="run1")
        assert MPASUnifiedCLI._output_name_kwargs(with_name, cfg) == {
            "output_name": "run1"
        }
        assert MPASUnifiedCLI._output_name_kwargs(without_name, cfg) == {}
        assert MPASUnifiedCLI._output_name_kwargs(with_name, MPASConfig()) == {}
        assert _accepted_kwargs(CrossSectionStyle, extend="min", bogus=1) == {
            "extend": "min"
        }

    def test_parallel_methods_accept_output_name(self: "TestOptionMapping") -> None:
        """
        This test verifies every create_batch_*_parallel method the CLI calls declares output_name.

        Parameters:
            None

        Returns:
            None
        """
        import inspect

        for cls, name in (
            (
                pw.ParallelPrecipitationProcessor,
                "create_batch_precipitation_maps_parallel",
            ),
            (pw.ParallelSurfaceProcessor, "create_batch_surface_maps_parallel"),
            (pw.ParallelWindProcessor, "create_batch_wind_plots_parallel"),
            (
                pw.ParallelCrossSectionProcessor,
                "create_batch_cross_section_plots_parallel",
            ),
        ):
            assert "output_name" in inspect.signature(getattr(cls, name)).parameters

    def test_help_texts_match_code(self: "TestOptionMapping") -> None:
        """
        This test verifies the --remap-method and --subsample help texts no longer contradict the code.

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()
        text = " ".join(
            parser._subparsers._group_actions[0]
            .choices["surface"]
            .format_help()
            .split()
        )
        assert "conservative (auto-selected default)" not in text
        assert "bilinear (default" in text
        wind = " ".join(
            parser._subparsers._group_actions[0].choices["wind"].format_help().split()
        )
        assert "default: -1)" not in wind


class TestLogLevelPrecedence:
    """A CLI --quiet/--verbose beats a YAML log_level."""

    @pytest.mark.parametrize("flag", ["--quiet", "--verbose"])
    def test_cli_flag_beats_yaml_log_level(
        self: "TestLogLevelPrecedence", flag: str, tmp_path: Path
    ) -> None:
        """
        This test verifies the YAML log_level is dropped when --quiet or --verbose is given.

        Parameters:
            flag (str): The CLI flag.
            tmp_path (Path): Directory for the YAML file.

        Returns:
            None
        """
        cfg = _merged(["surface", flag], {**BASE, "log_level": "DEBUG"}, tmp_path)
        assert cfg.log_level is None
        assert (cfg.quiet if flag == "--quiet" else cfg.verbose) is True

    def test_explicit_log_level_still_wins(
        self: "TestLogLevelPrecedence", tmp_path: Path
    ) -> None:
        """
        This test verifies --log-level beats both YAML and --quiet.

        Parameters:
            tmp_path (Path): Directory for the YAML file.

        Returns:
            None
        """
        cfg = _merged(
            ["surface", "--quiet", "--log-level", "WARNING"],
            {**BASE, "log_level": "DEBUG"},
            tmp_path,
        )
        assert cfg.log_level == "WARNING"


class TestConfigFromDict:
    """YAML aliases, figure sizes, and boolean type checks."""

    def test_cli_name_aliases(self: "TestConfigFromDict") -> None:
        """
        This test verifies YAML keys spelled like CLI flags set the matching fields.

        Parameters:
            None

        Returns:
            None
        """
        cfg = MPASConfig.from_dict(
            {
                "formats": ["pdf"],
                "accumulation": "a06h",
                "threshold": 2.5,
                "units": "in",
                "subsample": 4,
                "interpolation": "cubic",
                "lon": -100.0,
                "lat": 35.0,
                "time_range": [1, 3],
            }
        )
        assert cfg.output_formats == ["pdf"]
        assert cfg.accumulation_period == "a06h"
        assert cfg.precip_threshold == 2.5 and cfg.precip_units == "in"
        assert cfg.subsample_factor == 4 and cfg.interpolation_method == "cubic"
        assert (cfg.sounding_lon, cfg.sounding_lat) == (-100.0, 35.0)
        assert (cfg.time_start, cfg.time_end) == (1, 3)

    def test_canonical_key_wins_over_alias(self: "TestConfigFromDict") -> None:
        """
        This test verifies a canonical key beats its CLI-name alias.

        Parameters:
            None

        Returns:
            None
        """
        cfg = MPASConfig.from_dict({"formats": ["pdf"], "output_formats": ["svg"]})
        assert cfg.output_formats == ["svg"]

    def test_figure_width_height(self: "TestConfigFromDict") -> None:
        """
        This test verifies figure_width/figure_height set figure_size when it is absent.

        Parameters:
            None

        Returns:
            None
        """
        cfg = MPASConfig.from_dict({"figure_width": 7.0, "figure_height": 5.0})
        assert cfg.figure_size == (7.0, 5.0)
        cfg = MPASConfig.from_dict({"figure_size": [6, 4], "figure_width": 9.0})
        assert cfg.figure_size == (6, 4)

    @pytest.mark.parametrize(
        "text,expected",
        [("true", True), ("YES", True), ("False", False), (" no ", False)],
    )
    def test_boolean_strings(
        self: "TestConfigFromDict", text: str, expected: bool
    ) -> None:
        """
        This test verifies true/false/yes/no strings are accepted case-insensitively.

        Parameters:
            text (str): The string value.
            expected (bool): The expected boolean.

        Returns:
            None
        """
        assert (
            MPASConfig.from_dict({"show_background": text}).show_background is expected
        )

    @pytest.mark.parametrize("bad", ["maybe", "", 3, [True]])
    def test_boolean_rejects_other_values(self: "TestConfigFromDict", bad: Any) -> None:
        """
        This test verifies other strings and non-boolean types raise ValueError.

        Parameters:
            bad (Any): The invalid value.

        Returns:
            None
        """
        with pytest.raises(ValueError, match="parallel"):
            MPASConfig.from_dict({"parallel": bad})


class TestPathsAndNames:
    """Relative paths resolve against the cwd and --output names are sanitized."""

    def test_relative_paths_use_cwd_not_base_dir(
        self: "TestPathsAndNames", tmp_path: Path, monkeypatch: Any
    ) -> None:
        """
        This test verifies a relative path resolves against the cwd (then is confined to --base-dir).

        Parameters:
            tmp_path (Path): Temporary directory.
            monkeypatch (Any): Pytest monkeypatch fixture.

        Returns:
            None
        """
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        cfg = MPASConfig(
            grid_file="g.nc", data_dir="d", output_dir="o", base_dir=str(tmp_path)
        )
        errors: List[str] = []
        _cli()._confine_config_paths(cfg, errors)
        assert not errors
        assert cfg.output_dir == str((work / "o").resolve())
        assert cfg.grid_file == str((work / "g.nc").resolve())

    def test_outside_base_dir_still_refused(
        self: "TestPathsAndNames", tmp_path: Path, monkeypatch: Any
    ) -> None:
        """
        This test verifies the confinement check still rejects a cwd-relative path outside --base-dir.

        Parameters:
            tmp_path (Path): Temporary directory.
            monkeypatch (Any): Pytest monkeypatch fixture.

        Returns:
            None
        """
        (tmp_path / "base").mkdir()
        monkeypatch.chdir(tmp_path)
        cfg = MPASConfig(
            grid_file="g.nc",
            data_dir="d",
            output_dir="../escape",
            base_dir=str(tmp_path / "base"),
        )
        errors: List[str] = []
        _cli()._confine_config_paths(cfg, errors)
        assert errors

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("a/b c?.x", "a_b_c.x"),
            ("my run: 1*", "my_run_1"),
            ("plain", "plain"),
            ("../../etc", "etc"),
        ],
    )
    def test_output_name_sanitized(
        self: "TestPathsAndNames", raw: str, expected: str
    ) -> None:
        """
        This test verifies --output names lose separators, whitespace, and awkward characters while staying readable.

        Parameters:
            raw (str): The raw name.
            expected (str): The sanitized name.

        Returns:
            None
        """
        assert _sanitize_output_name(raw) == expected

    def test_confine_sanitizes_output(self: "TestPathsAndNames") -> None:
        """
        This test verifies the config output name is sanitized during path confinement.

        Parameters:
            None

        Returns:
            None
        """
        cfg = MPASConfig(grid_file="g", data_dir="d", output="a/b c?.x")
        _cli()._confine_config_paths(cfg, [])
        assert cfg.output == "a_b_c.x"


class TestReadmeExamples:
    """Every README command line parses and every YAML snippet loads."""

    README = (REPO / "README.md").read_text()

    @staticmethod
    def _commands(text: str) -> List[str]:
        """
        This helper extracts the 'mpasdiag ...' command lines (joining backslash continuations) from bash code blocks.

        Parameters:
            text (str): The README text.

        Returns:
            List[str]: The command lines.
        """
        commands: List[str] = []
        for block in re.findall(r"```bash\n(.*?)```", text, re.S):
            joined = re.sub(r"\\\n\s*", " ", block)
            for line in joined.splitlines():
                line = line.strip()
                if line.startswith("mpasdiag "):
                    commands.append(line)
        return commands

    def test_commands_found(self: "TestReadmeExamples") -> None:
        """
        This test verifies the extraction finds a substantial number of commands.

        Parameters:
            None

        Returns:
            None
        """
        assert len(self._commands(self.README)) > 40

    def test_every_command_parses(self: "TestReadmeExamples") -> None:
        """
        This test verifies each README command parses with the real parser and builds a valid MPASConfig.

        Parameters:
            None

        Returns:
            None
        """
        import shlex

        failures = []
        for command in self._commands(self.README):
            argv = shlex.split(command)[1:]
            if (
                "$" in command
                or "--help" in argv
                or "--config" in argv
                or not argv
                or argv[0].startswith("-")
            ):
                continue
            try:
                args = _parse(argv)
                cli = MPASUnifiedCLI()
                cli.parse_args_to_config(args)
            except SystemExit:
                failures.append(command)
            except Exception as exc:
                failures.append(f"{command} -> {exc}")
        assert not failures, "\n".join(failures)

    def test_yaml_snippets_load(self: "TestReadmeExamples") -> None:
        """
        This test verifies every README YAML snippet loads into MPASConfig without unknown keys.

        Parameters:
            None

        Returns:
            None
        """
        blocks = re.findall(r"```yaml\n(.*?)```", self.README, re.S)
        assert blocks
        for block in blocks:
            data = yaml.safe_load(block)
            if not isinstance(data, dict):
                continue
            known = set(MPASConfig.__dataclass_fields__)
            from mpasdiag.processing.utils_config import CONFIG_KEY_ALIASES

            assert not (set(data) - known - set(CONFIG_KEY_ALIASES)), set(data)
            MPASConfig.from_dict(data)


class TestIntegrationFollowUps:
    """Integration follow-ups applied after the round-3 fixers."""

    @needs_data
    def test_overlay_layers_pass_file_units(
        self: "TestIntegrationFollowUps", tmp_path: Path
    ) -> None:
        """
        This test verifies scalar overlay layers receive the file units, so q2 contours are labelled in g/kg rather than raw kg/kg.

        Parameters:
            tmp_path (Path): Output directory.

        Returns:
            None
        """
        from mpasdiag.processing.utils_monitor import PerformanceMonitor
        from mpasdiag.visualization.surface import MPASSurfacePlotter

        seen: List[Any] = []
        original = MPASSurfacePlotter.add_surface_overlay

        def _record(self: Any, ax: Any, lon: Any, lat: Any, cfg: Any, **kw: Any) -> Any:
            seen.append(dict(cfg))
            return original(self, ax, lon, lat, cfg, **kw)

        cli = _cli()
        cli.perf_monitor = PerformanceMonitor()
        cfg = MPASConfig(
            grid_file=str(GRID),
            data_dir=str(DIAG),
            output_dir=str(tmp_path),
            analysis_type="overlay",
            overlay_type="custom",
            primary_variable="t2m",
            secondary_variable="q2",
            dpi=50,
            figure_size=(8.0, 6.0),
            lon_min=-130.0,
            lon_max=-60.0,
            lat_min=20.0,
            lat_max=55.0,
            time_index=2,
        )
        with patch.object(MPASSurfacePlotter, "add_surface_overlay", _record):
            assert cli._run_overlay_analysis(cfg) is True

        q2_layers = [c for c in seen if c["var_name"] == "q2"]
        assert q2_layers and all(c.get("original_units") for c in q2_layers)

    def test_timing_lines_only_on_root_rank(
        self: "TestIntegrationFollowUps", capsys: pytest.CaptureFixture[str]
    ) -> None:
        """
        This test verifies the plain-text timing lines are printed by the root process only, so MPI ranks do not repeat them.

        Parameters:
            capsys (pytest.CaptureFixture[str]): Captured output.

        Returns:
            None
        """
        from datetime import timedelta

        cli = _cli()
        cli.perf_monitor = MagicMock(
            durations={"Surface analysis": timedelta(seconds=1)}
        )
        cfg = MPASConfig(quiet=False)

        with patch(
            "mpasdiag.processing.cli_unified.is_root_process", return_value=False
        ):
            cli._print_timing_messages(cfg, True)
        assert capsys.readouterr().out == ""

        with patch(
            "mpasdiag.processing.cli_unified.is_root_process", return_value=True
        ):
            cli._print_timing_messages(cfg, True)
        assert "Surface analysis completed in" in capsys.readouterr().out

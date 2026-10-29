#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
Regression tests for the fourth round of parallel fixes: the surface batch worker uses the same masked data and variable-located coordinates as the serial path, MPI rank detection agrees on every rank when the size comes from another launcher's variable, the MPI processor hand-over is evicted when a batch fails, and SIGTERM unwinds a seeded batch so its seed file is removed.
"""

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import mpasdiag.processing.parallel_wrappers as pw
from mpasdiag.processing import utils_logger

_REPO_ROOT = Path(pw.__file__).resolve().parents[2]
_LAUNCHER_PREFIXES = ("PMI_", "OMPI_", "PMIX_", "MV2_", "PALS_", "SLURM_")


@pytest.fixture
def no_launcher_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove every MPI/SLURM launcher variable."""
    for name in list(os.environ):
        if name.startswith(_LAUNCHER_PREFIXES):
            monkeypatch.delenv(name)
    return monkeypatch


class TestSurfaceWorkerMatchesSerial:
    """The batch worker must not plot underground values or crash on vertex fields."""

    def _kwargs(self, processor: Any, tmp_path: Path) -> dict:
        return {
            "processor": processor,
            "output_dir": str(tmp_path),
            "lon_min": -180.0,
            "lon_max": 180.0,
            "lat_min": -90.0,
            "lat_max": 90.0,
            "var_name": "temperature_925hPa",
            "plot_type": "scatter",
            "file_prefix": "mpas_surface",
            "formats": ["png"],
        }

    def test_worker_uses_masked_data_and_variable_coordinates(
        self, tmp_path: Path
    ) -> None:
        masked = MagicMock(values=np.array([np.nan, 280.0]))
        processor = MagicMock()
        times = np.array(["2024-09-17T00", "2024-09-17T01"], dtype="datetime64[ns]")
        processor.dataset = {"Time": MagicMock(values=times)}
        processor.get_2d_variable_data.return_value = masked
        processor.extract_2d_coordinates_for_variable.return_value = (
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        )
        plotter = MagicMock()
        plotter.create_surface_map.return_value = (MagicMock(), MagicMock())
        with (
            patch.object(pw, "MPASSurfacePlotter", return_value=plotter),
            patch.object(pw.MPASDateTimeUtils, "get_time_info", return_value="t"),
        ):
            pw._surface_worker((1, self._kwargs(processor, tmp_path)))
        processor.get_2d_variable_data.assert_called_once_with("temperature_925hPa", 1)
        processor.extract_2d_coordinates_for_variable.assert_called_once_with(
            "temperature_925hPa", masked
        )
        assert plotter.create_surface_map.call_args.kwargs["data"] is masked.values


_DIAG = _REPO_ROOT / "data" / "u240k" / "diag"
_GRID = _REPO_ROOT / "data" / "grids" / "x1.10242.static.nc"


@pytest.mark.skipif(
    not (_DIAG.is_dir() and _GRID.is_file()), reason="u240k test data not available"
)
def test_parallel_surface_batch_of_vertex_field(tmp_path: Path) -> None:
    """vorticity_850hPa lives on vertices; the parallel batch used cell coordinates and failed."""
    env = dict(os.environ, PYTHONPATH=str(_REPO_ROOT), MPLBACKEND="Agg")
    for name in list(env):
        if name.startswith(_LAUNCHER_PREFIXES):
            env.pop(name)
    cmd = [
        sys.executable,
        "-m",
        "mpasdiag",
        "--base-dir",
        "/",
        "surface",
        "--grid-file",
        str(_GRID),
        "--data-dir",
        str(_DIAG),
        "--variable",
        "vorticity_850hPa",
        "--time-range",
        "1",
        "2",
        "--parallel",
        "--workers",
        "2",
        "--output-dir",
        str(tmp_path),
        "--quiet",
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert len(list(tmp_path.glob("*.png"))) == 2


class TestLauncherSizeDetection:
    """Every rank must reach the same launcher verdict."""

    @pytest.mark.parametrize(
        "env, expected",
        [
            ({"PMIX_RANK": "0", "SLURM_PROCID": "0", "SLURM_NTASKS": "3"}, (0, 3)),
            ({"PMIX_RANK": "2", "SLURM_PROCID": "2", "SLURM_NTASKS": "3"}, (2, 3)),
            ({"PALS_RANKID": "0", "PALS_NRANKS": "4"}, (0, 4)),
            ({"PMIX_RANK": "0", "PMIX_SIZE": "2"}, (0, 2)),
            ({"PMIX_RANK": "0", "SLURM_PROCID": "0", "SLURM_NTASKS": "1"}, (0, 1)),
            ({"SLURM_PROCID": "0", "SLURM_NTASKS": "1"}, (0, 1)),
        ],
    )
    def test_size_from_any_launcher_variable(
        self, no_launcher_env: pytest.MonkeyPatch, env: dict, expected: tuple
    ) -> None:
        for name, value in env.items():
            no_launcher_env.setenv(name, value)
        assert utils_logger._detect_mpi_rank_and_size() == expected

    def test_rank_without_size_counts_as_launcher_on_rank_zero(
        self, no_launcher_env: pytest.MonkeyPatch
    ) -> None:
        no_launcher_env.setenv("PMIX_RANK", "0")
        assert utils_logger.mpi_launcher_detected() is True
        no_launcher_env.setenv("PMIX_RANK", "1")
        assert utils_logger.mpi_launcher_detected() is True


def _failing_map(*_: Any, **__: Any) -> List[Any]:
    raise RuntimeError("batch failed")


def test_mpi_handover_evicted_when_batch_fails() -> None:
    kwargs = {"grid_file": "/g.nc", "data_dir": "/d", "variables": ["t2m"]}
    processor = MagicMock()
    manager = MagicMock(backend="mpi")
    manager.parallel_map.side_effect = _failing_map
    pw._rank_processor_cache.clear()
    with patch.object(pw, "_dataset_has_variables", return_value=True):
        with pytest.raises(RuntimeError, match="batch failed"):
            pw._parallel_map_batch(manager, MagicMock(), [], "2d", kwargs, processor)
    assert pw._rank_processor_cache == {}


class TestSigtermUnwinds:
    """A scheduler's SIGTERM must run the batch's cleanup."""

    def test_sigterm_raises_system_exit_inside(self) -> None:
        if signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL:
            pytest.skip("SIGTERM already handled by the test runner")
        with pytest.raises(SystemExit) as info:
            with pw._sigterm_raises_exit():
                os.kill(os.getpid(), signal.SIGTERM)
        assert info.value.code == 128 + signal.SIGTERM
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL

    def test_existing_handler_is_left_alone(self) -> None:
        def handler(signum: int, frame: Any) -> None:
            pass

        previous = signal.signal(signal.SIGTERM, handler)
        try:
            with pw._sigterm_raises_exit():
                assert signal.getsignal(signal.SIGTERM) is handler
        finally:
            signal.signal(signal.SIGTERM, previous)

    def test_seed_file_removed_on_sigterm(self, tmp_path: Path) -> None:
        if signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL:
            pytest.skip("SIGTERM already handled by the test runner")
        seed_path = tmp_path / "seed.pkl"
        seed_path.write_bytes(b"x")

        def killed(*_: Any, **__: Any) -> None:
            os.kill(os.getpid(), signal.SIGTERM)

        manager = MagicMock(backend="multiprocessing")
        manager.parallel_map.side_effect = killed
        with patch.object(
            pw, "_make_processor_seed", return_value=("key", str(seed_path))
        ):
            with pytest.raises(SystemExit):
                pw._parallel_map_batch(
                    manager, MagicMock(), [], "2d", {"grid_file": "/g.nc"}, None
                )
        assert not seed_path.exists()

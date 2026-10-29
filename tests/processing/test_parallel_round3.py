#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
Round-3 regression tests for the parallel layer: path-like processors, dead-worker detection, in-memory processors, lazy mpi4py import, worker logging, empty batches, weights_dir forwarding, unpicklable MPI results, and the C3 file-naming helpers.
"""

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

import mpasdiag.processing.parallel as par
import mpasdiag.processing.parallel_wrappers as pw
from mpasdiag.processing import utils_logger
from mpasdiag.processing.parallel import MPASParallelManager, TaskResult
from mpasdiag.visualization.base_visualizer import GeographicBounds

_PARALLEL_FILE = par.__file__
_REPO_ROOT = str(Path(_PARALLEL_FILE).resolve().parents[2])
_MPI_ENV = (
    "PMI_RANK",
    "PMI_SIZE",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "PMIX_RANK",
    "MV2_COMM_WORLD_RANK",
    "MV2_COMM_WORLD_SIZE",
    "PALS_RANKID",
    "SLURM_PROCID",
    "SLURM_NTASKS",
)


@pytest.fixture
def clean_mpi_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove every MPI launcher variable so a test sees a plain serial environment."""
    for name in _MPI_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _task_ok(x: int) -> int:
    """Return the task value unchanged."""
    return x


def _task_kill_worker(x: int, parent_pid: int) -> int:
    """Kill the worker process abruptly (as a segfault or OOM kill would) but behave normally in the parent."""
    if os.getpid() != parent_pid:
        os._exit(1)
    return x


def _task_log(x: int) -> int:
    """Emit an INFO and a WARNING record from inside a worker."""
    logging.getLogger("mpasdiag.processing.parallel_wrappers").info(
        "INFO-from-worker %s", x
    )
    logging.getLogger("mpasdiag.processing.parallel_wrappers").warning(
        "WARN-from-worker %s", x
    )
    return x


def _run_with_watchdog(func: Any, seconds: float = 120.0) -> Any:
    """Run func in a daemon thread and fail the test instead of hanging when it does not finish."""
    box: Dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = func()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), "operation hung (no dead-worker detection)"
    if "error" in box:
        raise box["error"]
    return box["value"]


class TestQ1PathLikeAndDeadWorkers:
    """Q1: pathlib.Path processors reload in workers, in-memory processors run serially, dead workers fail fast."""

    def test_path_processor_supports_reload(self) -> None:
        proc = SimpleNamespace(grid_file=Path("/a/grid.nc"), data_dir=Path("/a/data"))
        assert pw._processor_supports_reload(proc) is True
        paths = pw._reload_paths(proc)
        assert paths == {"grid_file": "/a/grid.nc", "data_dir": "/a/data"}
        assert all(isinstance(v, str) for v in paths.values())

    def test_none_paths_do_not_support_reload(self) -> None:
        assert (
            pw._processor_supports_reload(
                SimpleNamespace(grid_file=None, data_dir=None)
            )
            is False
        )
        assert (
            pw._processor_supports_reload(
                SimpleNamespace(grid_file="/g.nc", data_dir=None)
            )
            is False
        )

    def test_in_memory_processor_switches_to_serial(self, clean_mpi_env: Any) -> None:
        proc = SimpleNamespace(grid_file="/g.nc", data_dir=None)
        manager = pw._create_batch_manager(proc, "cyclic", 2)
        assert manager.backend == "serial"

    def test_reloadable_processor_keeps_multiprocessing(
        self, clean_mpi_env: Any
    ) -> None:
        proc = SimpleNamespace(grid_file=Path("/g.nc"), data_dir=Path("/d"))
        manager = pw._create_batch_manager(proc, "cyclic", 2)
        assert manager.backend == "multiprocessing"

    def test_dead_worker_does_not_hang(self, clean_mpi_env: Any) -> None:
        manager = MPASParallelManager(
            backend="multiprocessing", n_workers=2, verbose=False
        )
        manager.set_error_policy("collect")
        parent = os.getpid()
        # fork keeps worker start-up instant; the liveness check is independent of the start method
        with patch.object(
            MPASParallelManager, "_get_mp_context_methods", return_value=["fork"]
        ):
            results = _run_with_watchdog(
                lambda: manager.parallel_map(_task_kill_worker, [0, 1, 2, 3], parent)
            )
        assert results is not None
        assert [r.task_id for r in results] == [0, 1, 2, 3]
        assert all(r.success for r in results)


class TestQ2LazyMpi:
    """Q2: mpi4py is imported only when MPI is requested or a launcher is detected."""

    def test_import_does_not_load_mpi4py(self) -> None:
        code = (
            "import sys; import mpasdiag.processing.parallel, "
            "mpasdiag.processing.parallel_wrappers, mpasdiag.processing.utils_logger; "
            "from mpasdiag.processing.utils_logger import MPASLogger; MPASLogger(); "
            "print('mpi4py' in sys.modules)"
        )
        env = {k: v for k, v in os.environ.items() if k not in _MPI_ENV}
        env["PYTHONPATH"] = _REPO_ROOT
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip().splitlines()[-1] == "False"

    def test_default_backend_serial_env_never_loads_mpi(
        self, clean_mpi_env: Any
    ) -> None:
        with (
            patch.object(par, "load_mpi") as loader,
            patch.object(par, "MPI_AVAILABLE", False),
        ):
            manager = MPASParallelManager(verbose=False, n_workers=2)
        assert manager.backend == "multiprocessing"
        loader.assert_not_called()

    def test_launcher_env_triggers_lazy_load(self, clean_mpi_env: Any) -> None:
        clean_mpi_env.setenv("PMI_RANK", "0")
        clean_mpi_env.setenv("PMI_SIZE", "2")
        with (
            patch.object(par, "load_mpi", return_value=None) as loader,
            patch.object(par, "MPI", None),
            patch.object(par, "MPI_AVAILABLE", False),
        ):
            manager = MPASParallelManager(verbose=False, n_workers=2)
        loader.assert_called_once()
        assert manager.backend == "multiprocessing"

    def test_rank_detection_from_env(self, clean_mpi_env: Any) -> None:
        assert utils_logger._detect_mpi_rank_and_size() == (0, 1)
        assert utils_logger.is_root_process() is True
        assert utils_logger.mpi_launcher_detected() is False
        clean_mpi_env.setenv("OMPI_COMM_WORLD_RANK", "3")
        clean_mpi_env.setenv("OMPI_COMM_WORLD_SIZE", "4")
        assert utils_logger._detect_mpi_rank_and_size() == (3, 4)
        assert utils_logger.is_root_process() is False
        assert utils_logger.mpi_launcher_detected() is True
        assert par.is_root_process is utils_logger.is_root_process

    def test_slurm_single_task_is_not_mpi(self, clean_mpi_env: Any) -> None:
        clean_mpi_env.setenv("SLURM_PROCID", "0")
        clean_mpi_env.setenv("SLURM_NTASKS", "1")
        assert utils_logger.mpi_launcher_detected() is False
        clean_mpi_env.setenv("SLURM_NTASKS", "4")
        clean_mpi_env.setenv("SLURM_PROCID", "2")
        assert utils_logger._detect_mpi_rank_and_size() == (2, 4)

    def test_auto_batch_processor_serial_without_launcher(
        self, clean_mpi_env: Any
    ) -> None:
        with patch.dict(sys.modules):
            sys.modules.pop("mpi4py", None)
            assert pw.auto_batch_processor() is False
            assert "mpi4py" not in sys.modules


class TestQ3WorkerLogging:
    """Q3: the parent's level, log file, and console choice reach pool workers."""

    def test_snapshot_reports_file_and_level(self, tmp_path: Path) -> None:
        log_file = str(tmp_path / "run.log")
        root = logging.getLogger("mpasdiag")
        saved = (root.handlers[:], root.level, root.propagate)
        try:
            utils_logger.MPASLogger(
                level=logging.WARNING, log_file=log_file, verbose=False
            )
            config = utils_logger.snapshot_logging_config()
            assert config["level"] == logging.WARNING
            assert config["log_file"] == os.path.abspath(log_file)
            assert config["console"] is False
        finally:
            for handler in root.handlers[:]:
                root.removeHandler(handler)
                handler.close()
            for handler in saved[0]:
                root.addHandler(handler)
            root.setLevel(saved[1])
            root.propagate = saved[2]

    def test_workers_write_to_parent_log_file(
        self, tmp_path: Path, clean_mpi_env: Any
    ) -> None:
        log_file = str(tmp_path / "workers.log")
        root = logging.getLogger("mpasdiag")
        saved = (root.handlers[:], root.level, root.propagate)
        try:
            utils_logger.MPASLogger(
                level=logging.INFO, log_file=log_file, verbose=False
            )
            manager = MPASParallelManager(
                backend="multiprocessing", n_workers=2, verbose=False
            )
            results = _run_with_watchdog(
                lambda: manager.parallel_map(_task_log, [0, 1, 2])
            )
            assert results is not None and all(r.success for r in results)
            for handler in root.handlers:
                handler.flush()
            text = Path(log_file).read_text()
        finally:
            for handler in root.handlers[:]:
                root.removeHandler(handler)
                handler.close()
            for handler in saved[0]:
                root.addHandler(handler)
            root.setLevel(saved[1])
            root.propagate = saved[2]
        assert "INFO-from-worker" in text
        assert "WARN-from-worker" in text
        assert "[pid " in text

    def test_configure_worker_logging_none_is_noop(self) -> None:
        root = logging.getLogger("mpasdiag")
        before = (root.handlers[:], root.level)
        utils_logger.configure_worker_logging(None)
        assert (root.handlers[:], root.level) == before


class TestQ4EmptyBatches:
    """Q4: empty time-index lists return early; zero-task statistics do not divide by zero."""

    def test_print_statistics_zero_tasks(self) -> None:
        manager = MPASParallelManager(backend="serial", verbose=True)
        manager.stats = par.ParallelStats(total_tasks=0)
        manager._print_statistics()

    def _dataset(self) -> Any:
        return SimpleNamespace(sizes={"Time": 3})

    @pytest.mark.parametrize(
        "method",
        [
            "surface",
            "wind",
            "cross",
            "skewt",
            "precip",
        ],
    )
    def test_empty_time_indices_return_empty(self, method: str) -> None:
        proc = SimpleNamespace(dataset=self._dataset(), grid_file="/g", data_dir="/d")
        bounds = GeographicBounds(-10, 10, -10, 10)
        with patch.object(pw, "_create_batch_manager") as factory:
            if method == "surface":
                out = pw.ParallelSurfaceProcessor.create_batch_surface_maps_parallel(
                    proc, "/o", bounds, time_indices=[]
                )
            elif method == "wind":
                out = pw.ParallelWindProcessor.create_batch_wind_plots_parallel(
                    proc, "/o", bounds, time_indices=[]
                )
            elif method == "cross":
                out = pw.ParallelCrossSectionProcessor.create_batch_cross_section_plots_parallel(
                    proc, "u", (0.0, 0.0), (1.0, 1.0), "/o", time_indices=[]
                )
            elif method == "skewt":
                out = pw.ParallelSkewTProcessor.create_batch_skewt_plots_parallel(
                    proc, "/o", 0.0, 0.0, time_indices=[]
                )
            else:
                with patch.object(pw, "_split_by_lookback", return_value=([], [])):
                    out = pw.ParallelPrecipitationProcessor.create_batch_precipitation_maps_parallel(
                        proc, "/o", bounds, time_indices=[]
                    )
        assert out == []
        assert getattr(out, "failed", 0) == 0
        factory.assert_not_called()


class TestQ5WeightsDir:
    """Q5: RemapConfig.weights_dir is forwarded for surface and wind and applied by the workers."""

    def _capture(self, call: Any) -> Dict[str, Any]:
        fake = MagicMock()
        fake.backend = "serial"
        fake.is_master = True
        fake.parallel_map.return_value = None
        with (
            patch.object(pw, "_create_batch_manager", return_value=fake),
            patch.object(pw.os, "makedirs"),
        ):
            call()
        args = fake.parallel_map.call_args[0][1]
        return args[0][1]

    def test_surface_forwards_weights_dir_and_options(self) -> None:
        proc = SimpleNamespace(
            dataset=SimpleNamespace(sizes={"Time": 2}),
            grid_file=Path("/g.nc"),
            data_dir=Path("/d"),
        )
        kwargs = self._capture(
            lambda: pw.ParallelSurfaceProcessor.create_batch_surface_maps_parallel(
                proc,
                "/o",
                GeographicBounds(0, 1, 0, 1),
                remap_config=pw.RemapConfig(weights_dir="/w"),
                style=pw.SurfaceBatchStyle(interpolation="linear"),
                output_name="myname",
            )
        )
        assert kwargs["weights_dir"] == "/w"
        assert kwargs["interpolation"] == "linear"
        assert kwargs["output_name"] == "myname"
        assert kwargs["grid_file"] == "/g.nc" and isinstance(kwargs["data_dir"], str)

    def test_wind_forwards_weights_dir(self) -> None:
        proc = SimpleNamespace(
            dataset=SimpleNamespace(sizes={"Time": 2}), grid_file="/g.nc", data_dir="/d"
        )
        kwargs = self._capture(
            lambda: pw.ParallelWindProcessor.create_batch_wind_plots_parallel(
                proc,
                "/o",
                GeographicBounds(0, 1, 0, 1),
                remap_config=pw.RemapConfig(weights_dir="/w"),
                output_name="wname",
            )
        )
        assert kwargs["weights_dir"] == "/w"
        assert kwargs["output_name"] == "wname"

    def test_apply_weights_dir_sets_plotter_attribute(self) -> None:
        plotter = SimpleNamespace(_remapper_weights_dir=None)
        pw._apply_weights_dir(plotter, {"weights_dir": "/w"})
        assert plotter._remapper_weights_dir == Path("/w")
        other = SimpleNamespace(_remapper_weights_dir=None)
        pw._apply_weights_dir(other, {})
        assert other._remapper_weights_dir is None


class TestQ6UnpicklableMpiResult:
    """Q6: an unpicklable result under MPI becomes a failed TaskResult instead of killing the rank."""

    def test_unpicklable_result_is_failed_task(self) -> None:
        manager = MPASParallelManager(backend="serial", verbose=False)
        manager.backend = "mpi"
        manager.comm = None
        result = manager._execute_one_task(lambda t: (lambda: t), 2, 5)
        assert isinstance(result, TaskResult)
        assert result.success is False
        assert result.result is None
        assert result.error is not None and "pickle" in result.error.lower()

    def test_picklable_result_unchanged(self) -> None:
        manager = MPASParallelManager(backend="serial", verbose=False)
        manager.backend = "mpi"
        result = manager._execute_one_task(_task_ok, 0, 7)
        assert result.success and result.result == 7


class TestC3Naming:
    """C3: cross-section and wind stems and the --output naming helper."""

    def test_cross_section_stem_without_points_unchanged(self) -> None:
        assert (
            pw.cross_section_file_stem("theta", "pressure", "T1", 12.5)
            == "mpas_cross_section_theta_vcrd_pressure_valid_T1_maxh_12p5km"
        )

    def test_cross_section_stem_with_points(self) -> None:
        stem = pw.cross_section_file_stem(
            "theta",
            "pressure",
            "T1",
            None,
            start_point=(-100.0, 30.5),
            end_point=(-80, 35),
        )
        assert stem.endswith("_valid_T1_s-100_30p5_e-80_35")

    def test_wind_stem(self) -> None:
        assert (
            pw.wind_file_stem("u10", "v10", "barbs", "T1")
            == "mpas_wind_u10_v10_barbs_valid_T1"
        )
        assert (
            pw.wind_file_stem(
                "uzonal", "umeridional", "barbs", "T1", wind_level="850hPa"
            )
            == "mpas_wind_uzonal_umeridional_barbs_lev850hPa_valid_T1"
        )

    def test_wind_stem_sanitizes(self) -> None:
        stem = pw.wind_file_stem("u/10", "v 10", "arrows", "T1", wind_level="850 hPa")
        assert "/" not in stem and "mpas_wind_u_10_" in stem
        assert stem.endswith("_lev850hPa_valid_T1")

    def test_batch_output_path(self, tmp_path: Path) -> None:
        default = pw._batch_output_path(str(tmp_path), None, "stem", "T1")
        assert default == os.path.join(str(tmp_path), "stem")
        named = pw._batch_output_path(str(tmp_path), "myplot", "stem", "T1")
        assert named == str((tmp_path / "myplot_T1").resolve())
        with pytest.raises(ValueError):
            pw._batch_output_path(str(tmp_path), "../escape", "stem", "T1")

    def test_cross_section_builder_carries_clim_and_output(self) -> None:
        proc = SimpleNamespace(grid_file="/g", data_dir="/d")
        kwargs = pw.ParallelCrossSectionProcessor._build_cross_section_worker_kwargs(
            proc,
            "theta",
            (0.0, 0.0),
            (1.0, 1.0),
            "/o",
            "pressure",
            10,
            ["png"],
            pw.CrossSectionBatchStyle(clim_min=1.0, clim_max=2.0),
            True,
            output_name="xs",
        )
        assert kwargs["clim_min"] == 1.0 and kwargs["clim_max"] == 2.0
        assert kwargs["output_name"] == "xs"
        assert kwargs["grid_file"] == "/g"


@pytest.mark.skipif(
    not os.path.exists("/opt/anaconda3/envs/meshgen/bin/mpiexec"),
    reason="no MPICH launcher available",
)
class TestMpiLaunch:
    """A real two-rank launch still works with the lazy mpi4py import."""

    def test_two_rank_map(self, tmp_path: Path) -> None:
        script = tmp_path / "mpi_two.py"
        script.write_text(
            "from mpasdiag.processing.parallel import MPASParallelManager\n"
            "def f(x):\n    return x * 2\n"
            "m = MPASParallelManager(load_balance_strategy='cyclic', verbose=False)\n"
            "r = m.parallel_map(f, [0, 1, 2, 3])\n"
            "print('BACKEND', m.backend, m.size)\n"
            "if m.is_master:\n    print('RESULT', [x.result for x in r])\n"
        )
        env = dict(os.environ, PYTHONPATH=_REPO_ROOT, FI_PROVIDER="tcp")
        out = subprocess.run(
            [
                "/opt/anaconda3/envs/meshgen/bin/mpiexec",
                "-n",
                "2",
                sys.executable,
                str(script),
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=180,
        )
        assert out.returncode == 0, out.stderr
        assert "BACKEND mpi 2" in out.stdout
        assert "RESULT [0, 2, 4, 6]" in out.stdout

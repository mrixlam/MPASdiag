#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
Regression tests for the parallel-performance fixes: large multiprocessing batches stay parallel, workers are recycled without being mistaken for crashes, the worker count respects available memory, the CLI no longer repeats a serial run on every MPI rank, MetPy's plotting module is not imported at package import, filtered (NaN-free) values line up with the remapper's dataset, and the ESMF boundary arrays and regridder are built once per process.
"""

import ast
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

import mpasdiag.processing.parallel as par
import mpasdiag.processing.parallel_wrappers as pw
from mpasdiag.processing.cli_unified import MPASUnifiedCLI
from mpasdiag.processing.parallel import MPASParallelManager
from mpasdiag.visualization import base_visualizer as bv
from mpasdiag.visualization.base_visualizer import MPASVisualizer

_REPO_ROOT = str(Path(par.__file__).resolve().parents[2])
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


def _task_pid(x: int) -> int:
    """Return the pid of the process that ran the task."""
    return os.getpid()


def _run_fork_pool(n_tasks: int, n_workers: int) -> List[Any]:
    """Run _task_pid over n_tasks in a fork pool (fast start-up) and return the results."""
    manager = MPASParallelManager(
        backend="multiprocessing", n_workers=n_workers, verbose=False
    )
    with patch.object(
        MPASParallelManager, "_get_mp_context_methods", return_value=["fork"]
    ):
        return manager.parallel_map(_task_pid, list(range(n_tasks)))


class TestPoolDispatch:
    """Fix 1 and 9: one task per dispatch and bounded worker lifetimes."""

    def test_large_batch_runs_in_workers(self, clean_mpi_env: Any) -> None:
        # 64 tasks on 2 workers used to give chunksize 4, a plain generator and a silent serial fallback
        results = _run_fork_pool(64, 2)
        pids = {r.result for r in results}
        assert all(r.success for r in results)
        assert os.getpid() not in pids

    def test_recycled_workers_are_not_reported_as_crashes(
        self, clean_mpi_env: Any
    ) -> None:
        clean_mpi_env.setenv("MPASDIAG_MAX_TASKS_PER_WORKER", "3")
        results = _run_fork_pool(24, 2)
        pids = {r.result for r in results}
        assert all(r.success for r in results)
        assert os.getpid() not in pids
        assert len(pids) > 2  # workers were replaced after 3 tasks each

    @pytest.mark.parametrize(
        "value, expected", [(None, 100), ("25", 25), ("0", None), ("-1", None)]
    )
    def test_max_tasks_setting(
        self, monkeypatch: pytest.MonkeyPatch, value: Any, expected: Any
    ) -> None:
        if value is None:
            monkeypatch.delenv("MPASDIAG_MAX_TASKS_PER_WORKER", raising=False)
        else:
            monkeypatch.setenv("MPASDIAG_MAX_TASKS_PER_WORKER", value)
        assert par._max_tasks_per_worker() == expected

    def test_invalid_max_tasks_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MPASDIAG_MAX_TASKS_PER_WORKER", "many")
        assert par._max_tasks_per_worker() == 100

    def test_worker_lost_rules(self) -> None:
        alive = SimpleNamespace(exitcode=None)
        recycled = SimpleNamespace(exitcode=0)
        crashed = SimpleNamespace(exitcode=-9)
        pool = SimpleNamespace(_pool=[])
        lost = MPASParallelManager._pool_worker_lost
        # a crashed worker is always a loss
        assert lost(pool, {1: crashed, 2: alive}, {1, 2}, 50, 10) is True
        # one replacement is explained by 10 results at 10 tasks per worker
        assert lost(pool, {1: recycled, 2: alive, 3: alive}, {1, 2}, 10, 10) is False
        # without recycling any replacement means a worker was lost
        assert lost(pool, {1: alive, 2: alive, 3: alive}, {1, 2}, 10, None) is True
        # inspection unavailable: the check is disabled
        assert lost(pool, {}, set(), 0, None) is False


class TestMemoryCap:
    """Fix 8: the worker count respects available memory."""

    def _available(self, gb: float) -> Any:
        return patch(
            "psutil.virtual_memory",
            return_value=SimpleNamespace(available=gb * 1024**3),
        )

    def test_caps_when_workers_do_not_fit(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("MPASDIAG_WORKER_MEMORY_GB", "4")
        with self._available(10), caplog.at_level(logging.WARNING):
            assert par._memory_capped_worker_count(8) == 2
        assert "Reducing workers from 8 to 2" in caplog.text

    def test_keeps_count_when_workers_fit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MPASDIAG_WORKER_MEMORY_GB", "0.5")
        with self._available(10):
            assert par._memory_capped_worker_count(8) == 8

    def test_never_below_one_and_zero_disables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MPASDIAG_WORKER_MEMORY_GB", "64")
        with self._available(4):
            assert par._memory_capped_worker_count(8) == 1
        monkeypatch.setenv("MPASDIAG_WORKER_MEMORY_GB", "0")
        with self._available(4):
            assert par._memory_capped_worker_count(8) == 8

    def test_estimate_uses_parent_memory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MPASDIAG_WORKER_MEMORY_GB", "lots")
        rss = SimpleNamespace(rss=2 * 1024**3)
        with patch(
            "psutil.Process", return_value=SimpleNamespace(memory_info=lambda: rss)
        ):
            assert par._estimated_worker_bytes() == pytest.approx(3 * 1024**3)

    def test_multiprocessing_map_applies_cap(self, clean_mpi_env: Any) -> None:
        clean_mpi_env.setenv("MPASDIAG_WORKER_MEMORY_GB", "4")
        manager = MPASParallelManager(
            backend="multiprocessing", n_workers=4, verbose=False
        )
        with (
            self._available(5),
            patch.object(
                MPASParallelManager, "_get_mp_context_methods", return_value=["fork"]
            ),
        ):
            results = manager.parallel_map(_task_pid, list(range(4)))
        assert manager.size == 1
        assert all(r.success for r in results)


class TestCliMpiRankGuard:
    """Fix 3: under an MPI launcher only distributed runs use every rank."""

    _ARGS = [
        "mpasdiag",
        "surface",
        "--grid-file",
        "grid.nc",
        "--data-dir",
        "data",
        "--variable",
        "t2m",
    ]

    def _run(
        self, monkeypatch: pytest.MonkeyPatch, rank: int, extra: List[str]
    ) -> MagicMock:
        for name in _MPI_ENV:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("PMI_RANK", str(rank))
        monkeypatch.setenv("PMI_SIZE", "4")
        monkeypatch.setattr(sys, "argv", self._ARGS + extra)
        cli = MPASUnifiedCLI()
        run = MagicMock(return_value=True)
        with (
            patch.object(MPASUnifiedCLI, "validate_config", return_value=True),
            patch.object(MPASUnifiedCLI, "run_analysis", run),
        ):
            assert cli.main() == 0
        return run

    def test_non_root_rank_skips_serial_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = self._run(monkeypatch, 1, ["--time-index", "1"])
        run.assert_not_called()

    def test_root_rank_runs_serial_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = self._run(monkeypatch, 0, ["--time-range", "0", "3"])
        run.assert_called_once()

    def test_every_rank_runs_parallel_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = self._run(monkeypatch, 2, ["--time-range", "0", "3", "--parallel"])
        run.assert_called_once()

    @pytest.mark.parametrize(
        "analysis, parallel, batch, expected",
        [
            ("surface", True, True, True),
            ("precip", True, True, True),
            ("cross", True, True, True),
            ("surface", True, False, False),
            ("surface", False, True, False),
            ("sounding", True, True, False),
            ("overlay", True, True, False),
        ],
    )
    def test_share_work_rule(
        self, analysis: str, parallel: bool, batch: bool, expected: bool
    ) -> None:
        config = SimpleNamespace(
            analysis_type=analysis, parallel=parallel, batch_mode=batch
        )
        assert MPASUnifiedCLI._mpi_ranks_share_work(config) is expected


def test_package_import_does_not_load_metpy_plots() -> None:
    """Fix 2: metpy.plots (about 10 s to import) is only loaded when a Skew-T is drawn."""
    code = (
        "import sys, mpasdiag.processing.cli_unified, mpasdiag.visualization; "
        "assert 'metpy.plots' not in sys.modules, 'metpy.plots imported eagerly'"
    )
    env = dict(os.environ, PYTHONPATH=_REPO_ROOT, MPLBACKEND="Agg")
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_every_dataset_open_names_its_engine() -> None:
    """Fix 2: without an explicit engine xarray loads every backend plugin, and MetPy's imports metpy.plots."""
    package = Path(par.__file__).resolve().parents[1]
    open_funcs = {"open_dataset", "open_mfdataset", "open_grid"}
    missing = []
    for path in sorted(package.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in open_funcs
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"xr", "ux"}
            ):
                continue
            keywords = {k.arg for k in node.keywords}
            # open_mfdataset receives a kwargs dict, whose "engine" entry is checked below
            if "engine" not in keywords and node.func.attr != "open_mfdataset":
                missing.append(f"{path.relative_to(package)}:{node.lineno}")
    assert not missing, f"dataset opens without engine=: {missing}"
    assert '"engine": NETCDF_ENGINE' in (package / "processing" / "base.py").read_text()


_U240K = Path(_REPO_ROOT) / "data" / "u240k" / "diag"
_GRID = Path(_REPO_ROOT) / "data" / "grids" / "x1.10242.static.nc"


@pytest.mark.skipif(
    not (_U240K.is_dir() and _GRID.is_file()), reason="u240k test data not available"
)
def test_loading_data_does_not_import_metpy() -> None:
    """Fix 2: a real 2D load leaves MetPy's xarray plugin (and metpy.plots) unimported."""
    code = (
        "import sys\n"
        "from mpasdiag.processing.processors_2d import MPAS2DProcessor\n"
        f"p = MPAS2DProcessor({str(_GRID)!r}, verbose=False)\n"
        f"p.load_2d_data({str(_U240K)!r})\n"
        "assert 'metpy.io' not in sys.modules, 'metpy.io imported by an xarray engine guess'\n"
    )
    env = dict(os.environ, PYTHONPATH=_REPO_ROOT, MPLBACKEND="Agg")
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


class TestAlignToDataset:
    """Fix 4: filtered values are placed back on the dataset's cells before remapping."""

    def _grid(self) -> xr.Dataset:
        lon = np.repeat(np.linspace(-170.0, 170.0, 18), 9)
        lat = np.tile(np.linspace(-80.0, 80.0, 9), 18)
        return xr.Dataset(
            {
                "lonCell": ("nCells", np.radians(lon)),
                "latCell": ("nCells", np.radians(lat)),
            }
        )

    def test_matching_sizes_are_unchanged(self) -> None:
        ds = self._grid()
        data = np.arange(ds.sizes["nCells"], dtype=float)
        out = MPASVisualizer._align_to_dataset(
            ds, np.degrees(ds.lonCell.values), np.degrees(ds.latCell.values), data
        )
        np.testing.assert_array_equal(out, data)

    def test_filtered_values_are_backmapped_with_nan(self) -> None:
        ds = self._grid()
        lon = np.degrees(ds.lonCell.values)
        lat = np.degrees(ds.latCell.values)
        data = np.arange(lon.size, dtype=float)
        keep = np.ones(lon.size, bool)
        keep[[3, 50, 100]] = False
        # 0..360 longitudes for the valid points must still match the dataset's cells
        out = MPASVisualizer._align_to_dataset(
            ds, np.mod(lon[keep], 360.0), lat[keep], data[keep]
        )
        assert out.shape == data.shape
        assert np.isnan(out[[3, 50, 100]]).all()
        np.testing.assert_array_equal(out[keep], data[keep])

    def test_kdtree_interpolate_with_filtered_data(self) -> None:
        ds = self._grid()
        lon = np.degrees(ds.lonCell.values)
        lat = np.degrees(ds.latCell.values)
        data = np.full(lon.size, 5.0)
        keep = lon < 0  # half of the cells were dropped as invalid
        viz = MPASVisualizer(verbose=False)
        _, _, grid = viz._kdtree_interpolate(
            lon[keep], lat[keep], data[keep], -180, 180, -80, 80, 10.0, ds, "nearest"
        )
        assert np.nanmax(grid) == 5.0
        assert np.isnan(grid).any()  # the dropped half stays missing


def _synthetic_topology(n_cells: int, seed: int = 0) -> xr.Dataset:
    """Build a random cell-to-vertex topology with 5-, 6- and 7-sided cells."""
    rng = np.random.default_rng(seed)
    n_vertices = 2 * n_cells
    edges = rng.choice([5, 6, 6, 6, 7], size=n_cells).astype(np.int32)
    voc = rng.integers(1, n_vertices + 1, size=(n_cells, 7)).astype(np.int32)
    return xr.Dataset(
        {
            "verticesOnCell": (("nCells", "maxEdges"), voc),
            "nEdgesOnCell": ("nCells", edges),
            "lonVertex": ("nVertices", rng.uniform(-np.pi, np.pi, n_vertices)),
            "latVertex": ("nVertices", rng.uniform(-1.5, 1.5, n_vertices)),
        }
    )


def _loop_boundaries(ds: xr.Dataset) -> Any:
    """Reference implementation: the original per-cell loop."""
    voc = ds.verticesOnCell.values
    vlon = ds.lonVertex.values
    vlat = ds.latVertex.values
    edges = ds.nEdgesOnCell.values.astype(int)
    nv = int(np.bincount(edges).argmax())
    lon_b = np.zeros((edges.size, nv))
    lat_b = np.zeros((edges.size, nv))
    for i in range(edges.size):
        idx = voc[i, : edges[i]] - 1
        pad = np.minimum(np.arange(nv), edges[i] - 1)
        lon_b[i] = vlon[idx[pad]]
        lat_b[i] = vlat[idx[pad]]
    return lon_b, lat_b


class TestSharedRemapCaches:
    """Fix 7: vectorised boundaries and a per-process regridder cache."""

    def test_vectorised_boundaries_match_loop(self) -> None:
        ds = _synthetic_topology(500)
        out = MPASVisualizer(verbose=False)._ensure_boundary_data(ds)
        ref_lon, ref_lat = _loop_boundaries(ds)
        np.testing.assert_array_equal(out["lon_b"].values, ref_lon)
        np.testing.assert_array_equal(out["lat_b"].values, ref_lat)

    def test_boundaries_shared_across_plotters_and_keyed_by_mesh(self) -> None:
        ds = _synthetic_topology(200)
        first = MPASVisualizer(verbose=False)._ensure_boundary_data(ds)
        second = MPASVisualizer(verbose=False)._ensure_boundary_data(ds)
        assert np.shares_memory(first["lon_b"].values, second["lon_b"].values)
        other = MPASVisualizer(verbose=False)._ensure_boundary_data(
            _synthetic_topology(200, seed=1)
        )
        assert not np.array_equal(other["lon_b"].values, first["lon_b"].values)

    def _build(self, viz: MPASVisualizer, lon: np.ndarray) -> Any:
        b = np.ones((lon.size, 3))
        return viz._get_or_build_remapper(
            lon, np.linspace(30, 40, lon.size), b, b, -100.0, -90.0, 30.0, 40.0, 1.0
        )

    def test_regridder_shared_across_plotters(self) -> None:
        lon = np.linspace(-100, -90, 12)
        with patch.object(
            bv, "MPASRemapper", side_effect=lambda **_: MagicMock()
        ) as cls:
            r1 = self._build(MPASVisualizer(verbose=False), lon)
            r2 = self._build(MPASVisualizer(verbose=False), lon)
        assert r1 is r2
        assert cls.call_count == 1

    def test_regridder_rebuilt_for_other_grid_or_weights_dir(
        self, tmp_path: Path
    ) -> None:
        lon = np.linspace(-100, -90, 12)
        with patch.object(
            bv, "MPASRemapper", side_effect=lambda **_: MagicMock()
        ) as cls:
            self._build(MPASVisualizer(verbose=False), lon)
            self._build(MPASVisualizer(verbose=False), lon + 0.5)
            viz = MPASVisualizer(verbose=False)
            viz._remapper_weights_dir = tmp_path
            self._build(viz, lon + 0.5)
        assert cls.call_count == 3


class _FakeProcessor:
    """Picklable stand-in for a loaded processor: reload paths plus a small dataset."""

    def __init__(self, n: int = 5) -> None:
        self.grid_file = "/data/grid.nc"
        self.data_dir = "/data/diag"
        self.dataset = xr.Dataset({"t2m": ("nCells", np.arange(n, dtype=float))})


class _Unpicklable(_FakeProcessor):
    """A processor that cannot be pickled (holds a lambda)."""

    def __init__(self) -> None:
        super().__init__()
        self.hook = lambda: None


def _seed_probe(args: Any, processor_seed: Any = None) -> float:
    """Worker that installs the shipped processor and reads from it; a reload would raise."""
    _, kwargs = args
    pw._install_processor_seed(processor_seed)
    processor = pw._get_or_create_2d_processor(kwargs)
    return float(processor.dataset["t2m"].sum())


def _fail_reload(*_: Any, **__: Any) -> None:
    raise AssertionError("worker reloaded the data from disk")


class TestProcessorSeed:
    """Fix 6: workers receive the parent's loaded processor instead of reloading it."""

    _KW = {"grid_file": "/data/grid.nc", "data_dir": "/data/diag", "variables": ["t2m"]}

    @pytest.fixture(autouse=True)
    def _empty_worker_cache(self) -> Any:
        pw._rank_processor_cache.clear()
        yield
        pw._rank_processor_cache.clear()

    def test_seed_round_trip_uses_worker_cache_key(self) -> None:
        seed = pw._make_processor_seed("2d", dict(self._KW), _FakeProcessor())
        assert seed is not None
        key, path = seed
        try:
            assert key == pw._processor_cache_key("2d", self._KW)
            assert os.stat(path).st_mode & 0o077 == 0  # private to the user
            pw._install_processor_seed(seed)
        finally:
            os.unlink(path)
        processor = pw._rank_processor_cache[key]
        np.testing.assert_array_equal(processor.dataset["t2m"].values, np.arange(5.0))

    def test_existing_cache_entry_is_kept(self) -> None:
        seed = pw._make_processor_seed("2d", dict(self._KW), _FakeProcessor())
        assert seed is not None
        sentinel = object()
        pw._rank_processor_cache[seed[0]] = sentinel
        pw._install_processor_seed(seed)
        os.unlink(seed[1])
        assert pw._rank_processor_cache[seed[0]] is sentinel

    def test_bad_or_missing_seed_file_leaves_cache_empty(self, tmp_path: Path) -> None:
        bad = tmp_path / "seed.pkl"
        bad.write_bytes(b"not a pickle")
        pw._install_processor_seed(("key", str(bad)))
        pw._install_processor_seed(("key", str(tmp_path / "missing.pkl")))
        assert "key" not in pw._rank_processor_cache

    def test_seed_file_removed_after_batch(self) -> None:
        manager = MagicMock(backend="multiprocessing")
        pw._parallel_map_batch(
            manager,
            _seed_probe,
            [(0, dict(self._KW))],
            "2d",
            dict(self._KW),
            _FakeProcessor(),
        )
        _, path = manager.parallel_map.call_args.kwargs["processor_seed"]
        assert not os.path.exists(path)

    @pytest.mark.parametrize(
        "processor, variables",
        [
            (_FakeProcessor(), ["missing"]),  # lacks a requested variable
            (_Unpicklable(), ["t2m"]),  # cannot be pickled
            (SimpleNamespace(grid_file=None, data_dir=None, dataset=None), ["t2m"]),
        ],
    )
    def test_no_seed_when_unsuitable(
        self, processor: Any, variables: List[str]
    ) -> None:
        kwargs = dict(self._KW, variables=variables)
        assert pw._make_processor_seed("2d", kwargs, processor) is None

    def test_no_seed_above_size_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pw, "_MAX_PROCESSOR_SEED_BYTES", 10)
        assert pw._make_processor_seed("2d", dict(self._KW), _FakeProcessor()) is None

    def test_seed_only_for_multiprocessing(self) -> None:
        tasks = [(0, dict(self._KW))]
        for backend, expect_seed in [
            ("multiprocessing", True),
            ("mpi", False),
            ("serial", False),
        ]:
            manager = MagicMock(backend=backend)
            pw._parallel_map_batch(
                manager, _seed_probe, tasks, "2d", dict(self._KW), _FakeProcessor()
            )
            assert (
                "processor_seed" in manager.parallel_map.call_args.kwargs
            ) is expect_seed

    def test_pool_workers_use_seed_instead_of_reloading(
        self, clean_mpi_env: Any
    ) -> None:
        manager = MPASParallelManager(
            backend="multiprocessing", n_workers=2, verbose=False
        )
        tasks = [(i, dict(self._KW)) for i in range(6)]
        from mpasdiag.processing.processors_2d import MPAS2DProcessor

        # fork workers inherit this patch, so any reload inside a worker fails its task
        with (
            patch.object(
                MPASParallelManager, "_get_mp_context_methods", return_value=["fork"]
            ),
            patch.object(MPAS2DProcessor, "load_2d_data", _fail_reload),
        ):
            results = pw._parallel_map_batch(
                manager, _seed_probe, tasks, "2d", dict(self._KW), _FakeProcessor()
            )
        assert [r.success for r in results] == [True] * 6, [r.error for r in results]
        assert all(r.result == 10.0 for r in results)


class TestSeedVariableCoverage:
    """Fix 6: a seed is valid when the only missing variables are ones a reload would skip too."""

    def _processor(self, tmp_path: Path, file_vars: List[str]) -> Any:
        first = tmp_path / "mpasout.2024-01-01_00.00.00.nc"
        xr.Dataset({v: ("nCells", np.zeros(3)) for v in file_vars}).to_netcdf(first)
        proc = _FakeProcessor()
        proc.data_dir = str(tmp_path)
        proc._discover_data_files = lambda data_dir: [str(first)]
        return proc

    def test_missing_variable_absent_from_files_is_fine(self, tmp_path: Path) -> None:
        proc = self._processor(tmp_path, ["t2m"])
        assert pw._dataset_has_variables(proc, ["t2m", "zgrid"]) is True

    def test_missing_variable_present_in_files_blocks_seed(
        self, tmp_path: Path
    ) -> None:
        proc = self._processor(tmp_path, ["t2m", "theta"])
        assert pw._dataset_has_variables(proc, ["t2m", "theta"]) is False

    def test_unreadable_files_block_seed(self) -> None:
        proc = _FakeProcessor()
        proc._discover_data_files = lambda data_dir: []
        assert pw._dataset_has_variables(proc, ["t2m", "zgrid"]) is False


def test_serial_fallback_does_not_keep_seed_in_parent() -> None:
    """Fix 6: a seed installed in the parent by a serial fallback is evicted after the batch."""
    kwargs = {
        "grid_file": "/data/grid.nc",
        "data_dir": "/data/diag",
        "variables": ["t2m"],
    }
    pw._rank_processor_cache.clear()

    def run_in_parent(worker: Any, tasks: Any, processor_seed: Any = None) -> List[Any]:
        return [worker(task, processor_seed=processor_seed) for task in tasks]

    manager = MagicMock(backend="multiprocessing")
    manager.parallel_map.side_effect = run_in_parent
    results = pw._parallel_map_batch(
        manager, _seed_probe, [(0, dict(kwargs))], "2d", dict(kwargs), _FakeProcessor()
    )
    assert results == [10.0]
    assert pw._rank_processor_cache == {}


def test_forkserver_preloads_netcdf4_first() -> None:
    """Workers must not load netCDF4's native libraries for the first time after the fork (macOS segfault)."""
    assert par._FORKSERVER_PRELOAD[0] == "netCDF4"


def test_forkserver_preloads_first_task_imports() -> None:
    """Every worker's first task imports these; the server loads them once instead."""
    for name in ("dask.array", "pint", "distributed", "scipy.interpolate"):
        assert name in par._FORKSERVER_PRELOAD
    assert par._FORKSERVER_PRELOAD.index(
        "mpasdiag.processing.parallel_wrappers"
    ) < par._FORKSERVER_PRELOAD.index("dask.array")


class TestEarlyWorkerServer:
    """The forkserver starts while the data load, not after."""

    def test_started_on_forkserver_platforms(self) -> None:
        with (
            patch.object(
                MPASParallelManager,
                "_get_mp_context_methods",
                return_value=["forkserver", "spawn"],
            ),
            patch("multiprocessing.forkserver.ensure_running") as ensure,
            patch("mpasdiag.processing.parallel.get_context") as get_context,
        ):
            par.start_worker_server()
        get_context.assert_called_once_with("forkserver")
        get_context.return_value.set_forkserver_preload.assert_called_once_with(
            par._FORKSERVER_PRELOAD
        )
        ensure.assert_called_once()

    def test_nothing_to_start_elsewhere(self) -> None:
        with (
            patch.object(
                MPASParallelManager,
                "_get_mp_context_methods",
                return_value=["fork", "spawn"],
            ),
            patch("multiprocessing.forkserver.ensure_running") as ensure,
        ):
            par.start_worker_server()
        ensure.assert_not_called()

    def test_failure_is_not_fatal(self) -> None:
        with (
            patch.object(
                MPASParallelManager,
                "_get_mp_context_methods",
                return_value=["forkserver", "spawn"],
            ),
            patch(
                "multiprocessing.forkserver.ensure_running",
                side_effect=OSError("no"),
            ),
        ):
            par.start_worker_server()

    @pytest.mark.parametrize(
        "extra, expected",
        [
            (["--time-range", "0", "3", "--parallel"], True),
            (["--time-range", "0", "3"], False),
            (["--time-index", "1", "--parallel"], False),
        ],
    )
    def test_cli_starts_server_for_parallel_batches(
        self, clean_mpi_env: Any, extra: List[str], expected: bool
    ) -> None:
        clean_mpi_env.setattr(sys, "argv", TestCliMpiRankGuard._ARGS + extra)
        with (
            patch.object(MPASUnifiedCLI, "validate_config", return_value=True),
            patch.object(MPASUnifiedCLI, "run_analysis", return_value=True),
            patch("mpasdiag.processing.cli_unified.start_worker_server") as start,
        ):
            assert MPASUnifiedCLI().main() == 0
        assert start.called is expected

    def test_cli_does_not_start_server_under_mpi(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with patch("mpasdiag.processing.cli_unified.start_worker_server") as start:
            TestCliMpiRankGuard()._run(
                monkeypatch, 0, ["--time-range", "0", "3", "--parallel"]
            )
        start.assert_not_called()


def _fake_numba(initialized: bool, threads: int = 8) -> Any:
    """Build stand-in numba modules exposing the attributes _cap_numba_threads uses."""
    config = SimpleNamespace(NUMBA_NUM_THREADS=threads)
    parallel = SimpleNamespace(_is_initialized=initialized, NUM_THREADS=threads)
    return {
        "numba": SimpleNamespace(),
        "numba.core": SimpleNamespace(config=config),
        "numba.core.config": config,
        "numba.np": SimpleNamespace(),
        "numba.np.ufunc": SimpleNamespace(parallel=parallel),
        "numba.np.ufunc.parallel": parallel,
    }


class TestNumbaThreadCap:
    """uxarray's numba kernels must share the worker's thread budget."""

    def test_caps_imported_numba_before_pool_starts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
        mods = _fake_numba(initialized=False)
        with patch.dict(sys.modules, mods):
            par._cap_numba_threads(2)
        assert os.environ["NUMBA_NUM_THREADS"] == "2"
        assert mods["numba.core.config"].NUMBA_NUM_THREADS == 2
        assert mods["numba.np.ufunc.parallel"].NUM_THREADS == 2

    def test_running_pool_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
        mods = _fake_numba(initialized=True)
        with patch.dict(sys.modules, mods):
            par._cap_numba_threads(2)
        assert "NUMBA_NUM_THREADS" not in os.environ
        assert mods["numba.np.ufunc.parallel"].NUM_THREADS == 8

    def test_user_setting_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NUMBA_NUM_THREADS", "6")
        mods = _fake_numba(initialized=False)
        with patch.dict(sys.modules, mods):
            par._cap_numba_threads(2)
        assert os.environ["NUMBA_NUM_THREADS"] == "6"
        assert mods["numba.np.ufunc.parallel"].NUM_THREADS == 8

    def test_never_raises_the_pool_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
        mods = _fake_numba(initialized=False, threads=4)
        with patch.dict(sys.modules, mods):
            par._cap_numba_threads(16)
        assert mods["numba.np.ufunc.parallel"].NUM_THREADS == 4

    def test_env_set_when_numba_not_imported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
        with patch.dict(sys.modules):
            sys.modules.pop("numba", None)
            par._cap_numba_threads(3)
        assert os.environ["NUMBA_NUM_THREADS"] == "3"

    def test_real_numba_kernel_uses_capped_pool(self) -> None:
        code = (
            "import mpasdiag.processing.parallel as par\n"
            "import numba, numpy as np\n"
            "par._cap_numba_threads(2)\n"
            "@numba.njit(parallel=True)\n"
            "def total(a):\n"
            "    t = 0.0\n"
            "    for i in numba.prange(a.size):\n"
            "        t += a[i]\n"
            "    return t\n"
            "assert total(np.ones(1000)) == 1000.0\n"
            "assert numba.get_num_threads() == 2, numba.get_num_threads()\n"
            "assert numba.config.NUMBA_NUM_THREADS == 2\n"
        )
        env = dict(os.environ, PYTHONPATH=_REPO_ROOT, MPLBACKEND="Agg")
        env.pop("NUMBA_NUM_THREADS", None)
        proc = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stderr[-2000:]


def test_package_import_does_not_load_metpy() -> None:
    """MetPy (about a second for metpy.calc/metpy.units) is only imported when a sounding is computed."""
    code = (
        "import sys, mpasdiag.processing.cli_unified, mpasdiag.visualization\n"
        "import mpasdiag.diagnostics.sounding as snd, mpasdiag.visualization.skewt as sk\n"
        "assert not [m for m in sys.modules if m.startswith('metpy')], 'metpy imported eagerly'\n"
        "assert snd.HAS_METPY\n"
        "import metpy.calc\n"
        "assert snd.mpcalc is metpy.calc and sk.mpcalc is metpy.calc\n"
        "assert snd._metpy_units() is sk._metpy_units()\n"
    )
    env = dict(os.environ, PYTHONPATH=_REPO_ROOT, MPLBACKEND="Agg")
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_lazy_metpy_attributes_honour_patches() -> None:
    """Code reads mpcalc through the module, so a patched module attribute is what it uses."""
    from mpasdiag.diagnostics import sounding

    fake = MagicMock()
    with patch.object(sounding, "mpcalc", fake, create=True):
        assert sounding._metpy_calc() is fake
    with pytest.raises(AttributeError):
        getattr(sounding, "not_an_attribute")


class TestCliSelectiveLoad:
    """The CLI loads only what the analysis reads, like the batch workers."""

    def _config(self, **overrides: Any) -> Any:
        base = dict(
            data_dir="/data",
            chunk_size=None,
            variable="t2m",
            u_variable="u10",
            v_variable="v10",
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_variables_are_forwarded(self) -> None:
        proc = SimpleNamespace(dataset={"t2m": 1})
        loader = MagicMock(return_value=proc)
        assert MPASUnifiedCLI._load_data(loader, self._config(), ["t2m"]) is proc
        loader.assert_called_once_with("/data", variables=["t2m"])

    def test_chunk_size_still_forwarded(self) -> None:
        loader = MagicMock(return_value=SimpleNamespace(dataset={"t2m": 1}))
        MPASUnifiedCLI._load_data(loader, self._config(chunk_size=500), ["t2m"])
        loader.assert_called_once_with("/data", variables=["t2m"], chunk_size=500)

    def test_partly_missing_variables_reload_everything(self) -> None:
        """A wrong u-variable next to a valid v-variable must list every available variable."""
        partial = SimpleNamespace(dataset={"v10": 1})
        full = SimpleNamespace(dataset={"u10": 1, "v10": 1})
        loader = MagicMock(side_effect=[partial, full])
        assert MPASUnifiedCLI._load_data(loader, self._config(), ["nou", "v10"]) is full
        assert loader.call_args_list[-1] == (("/data",),)

    def test_missing_vertical_helpers_do_not_force_full_load(self) -> None:
        proc = SimpleNamespace(dataset={"theta": 1})
        loader = MagicMock(return_value=proc)
        requested = ["theta", "pressure", "zgrid"]
        assert MPASUnifiedCLI._load_data(loader, self._config(), requested) is proc
        loader.assert_called_once()

    def test_no_or_unset_variables_load_everything(self) -> None:
        loader = MagicMock(return_value="proc")
        MPASUnifiedCLI._load_data(loader, self._config())
        MPASUnifiedCLI._load_data(loader, self._config(), [None])
        assert loader.call_args_list == [(("/data",),), (("/data",),)]

    def test_missing_variables_fall_back_to_full_load(self) -> None:
        from mpasdiag.processing.base import MissingVariablesError

        loader = MagicMock(side_effect=[MissingVariablesError("none"), "proc"])
        assert MPASUnifiedCLI._load_data(loader, self._config(), ["nope"]) == "proc"
        assert loader.call_args_list[-1] == (("/data",),)

    def test_other_load_errors_propagate(self) -> None:
        loader = MagicMock(side_effect=ValueError("no files"))
        with pytest.raises(ValueError, match="no files"):
            MPASUnifiedCLI._load_data(loader, self._config(), ["t2m"])
        loader.assert_called_once()

    @pytest.mark.parametrize(
        "method, processor, overrides, expected",
        [
            ("_run_surface_analysis", "MPAS2DProcessor", {}, ["t2m"]),
            (
                "_run_precipitation_analysis",
                "MPAS2DProcessor",
                {"variable": "total"},
                ["rainc", "rainnc"],
            ),
            ("_run_wind_analysis", "MPAS2DProcessor", {}, ["u10", "v10"]),
            (
                "_run_cross_analysis",
                "MPAS3DProcessor",
                {"variable": "theta"},
                ["theta", "pressure"],
            ),
        ],
    )
    def test_analysis_requests_its_variables(
        self, method: str, processor: str, overrides: dict, expected: List[str]
    ) -> None:
        cli = MPASUnifiedCLI()
        cli.perf_monitor = MagicMock()
        config = self._config(
            grid_file="g.nc",
            verbose=False,
            figure_size=(4, 3),
            dpi=50,
            output_dir="/out",
            batch_mode=False,
            **overrides,
        )
        load = MagicMock(side_effect=RuntimeError("stop after load"))
        with (
            patch(f"mpasdiag.processing.cli_unified.{processor}"),
            patch.object(MPASUnifiedCLI, "_load_data", load),
        ):
            with pytest.raises(RuntimeError, match="stop after load"):
                getattr(cli, method)(config)
        requested = load.call_args.args[2]
        assert requested[: len(expected)] == expected

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""
MPASdiag Test Suite: Parallel Orchestration Regression Tests

This module contains regression tests for the parallel orchestration fixes: a task error under the ABORT error policy is re-raised instead of triggering the start-method and serial fallbacks that re-run the whole batch, gathered MPI results are returned in task order for every distribution strategy, an unknown backend raises a ValueError listing the valid backends, parallel batch results carry per-step outcome counts (with lookback skips not counted as failures), and time tags in parallel file names keep sub-hourly resolution.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: April 2026
Version: 1.0.0
"""

from types import SimpleNamespace
from typing import Any, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mpasdiag.processing import parallel as par
from mpasdiag.processing.parallel import (
    MPASParallelManager,
    MPASResultCollector,
    TaskAbortError,
    TaskResult,
)
from mpasdiag.processing import parallel_wrappers as pw


class TestAbortPolicyFallback:
    """ABORT-policy task errors must not trigger the fallback chain (finding 10)."""

    def test_task_abort_error_is_not_retried(self: "TestAbortPolicyFallback") -> None:
        """
        This test makes the first multiprocessing context raise TaskAbortError and verifies that it propagates immediately: no other start method is tried and no serial re-run happens.

        Parameters:
            None

        Returns:
            None
        """
        manager = MPASParallelManager(
            backend="multiprocessing", verbose=False, n_workers=2
        )
        manager.set_error_policy("abort")
        serial_calls: List[int] = []

        def func(task: Any) -> Any:
            serial_calls.append(task)
            return task

        tried: List[str] = []

        def fake_get_context(method: str) -> Any:
            tried.append(method)
            ctx = MagicMock()
            ctx.Pool.side_effect = TaskAbortError("Task 1 failed: ValueError: x")
            return ctx

        with patch.object(par, "get_context", fake_get_context):
            with pytest.raises(TaskAbortError, match="ValueError"):
                manager._run_pool_with_fallback(func, [(0, 0), (1, 1)], (), {})

        assert len(tried) == 1
        assert serial_calls == []

    def test_infrastructure_failure_still_falls_back(
        self: "TestAbortPolicyFallback",
    ) -> None:
        """
        This test verifies that a pool-infrastructure failure (not a task error) still falls back to the next start method and finally to serial execution.

        Parameters:
            None

        Returns:
            None
        """
        manager = MPASParallelManager(
            backend="multiprocessing", verbose=False, n_workers=2
        )
        tried: List[str] = []

        def fake_get_context(method: str) -> Any:
            tried.append(method)
            ctx = MagicMock()
            ctx.Pool.side_effect = OSError("no fork")
            return ctx

        with patch.object(par, "get_context", fake_get_context):
            results = manager._run_pool_with_fallback(
                lambda t: t * 2, [(0, 1), (1, 2)], (), {}
            )

        assert len(tried) == len(manager._get_mp_context_methods())
        assert [r.result for r in results] == [2, 4]

    def test_pool_task_wrapper_wraps_abort_errors(
        self: "TestAbortPolicyFallback",
    ) -> None:
        """
        This test verifies that the pool wrapper converts an ABORT-policy task exception into a picklable TaskAbortError carrying the original type and message.

        Parameters:
            None

        Returns:
            None
        """
        import pickle

        def failing(task: Any) -> Any:
            raise ValueError("bad task")

        with patch.object(par, "_POOL_TASK_CONTEXT", (failing, "abort", (), {})):
            with pytest.raises(TaskAbortError, match="ValueError: bad task") as exc:
                par._pool_task_wrapper((3, "t"))

        assert "Task 3" in str(exc.value)
        assert "bad task" in str(pickle.loads(pickle.dumps(exc.value)))


class TestBackendAndGather:
    """Unknown backends are rejected and gathered results are in task order (finding 13)."""

    def test_unknown_backend_raises(self: "TestBackendAndGather") -> None:
        """
        This test verifies that an unknown backend name raises ValueError listing the valid backends instead of silently running serially.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="mpi, multiprocessing, serial"):
            MPASParallelManager(backend="threads")

    def test_gather_results_sorted_by_task_id(self: "TestBackendAndGather") -> None:
        """
        This test verifies that results gathered from ranks in rank order (as produced by cyclic distribution) are returned sorted by task id on the root rank and that non-root ranks get None.

        Parameters:
            None

        Returns:
            None
        """
        comm = MagicMock()
        comm.Get_rank.return_value = 0
        comm.Get_size.return_value = 2
        rank0 = [
            TaskResult(task_id=0, success=True),
            TaskResult(task_id=2, success=True),
        ]
        rank1 = [
            TaskResult(task_id=1, success=True),
            TaskResult(task_id=3, success=True),
        ]
        comm.gather.return_value = [rank0, rank1]

        merged = MPASResultCollector(comm).gather_results(rank0)

        assert merged is not None
        assert [r.task_id for r in merged] == [0, 1, 2, 3]

        comm.Get_rank.return_value = 1
        comm.gather.return_value = None
        assert MPASResultCollector(comm).gather_results(rank1) is None


class TestWrapperResults:
    """Parallel wrappers expose outcome counts and use sub-hour-safe time tags."""

    def test_process_results_carry_counts_and_skips(self: "TestWrapperResults") -> None:
        """
        This test verifies that the processed result list carries successful, failed, and skipped counts and that a skipped lookback step is not counted as a failure.

        Parameters:
            None

        Returns:
            None
        """
        results = [
            SimpleNamespace(task_id=0, success=True, result={"files": ["a.png"], "timings": {}}, error=None),
            SimpleNamespace(task_id=1, success=False, result=None, error="boom"),
            SimpleNamespace(task_id=2, success=True, result={"skipped": "too early", "files": [], "timings": {}}, error=None),
        ]  # fmt: skip
        manager = MagicMock()
        manager.get_statistics.return_value = None

        batch = pw._process_parallel_results(results, [1, 2, 3], "out", manager, "TEST")

        assert list(batch) == ["a.png"]
        assert (batch.successful, batch.failed, batch.skipped) == (1, 1, 1)

    def test_time_tag_keeps_minutes(self: "TestWrapperResults") -> None:
        """
        This test verifies that sub-hourly times are not truncated to the hour in parallel file names, avoiding collisions.

        Parameters:
            None

        Returns:
            None
        """
        times = np.array(
            [np.datetime64("2024-09-17T01:00"), np.datetime64("2024-09-17T01:30")]
        )
        ds = xr.Dataset({"v": (("Time",), [0.0, 1.0])}, coords={"Time": times})

        tag0, _ = pw._get_time_str(ds, 0)
        tag1, _ = pw._get_time_str(ds, 1)

        assert tag0 == "20240917T01"
        assert tag1 == "20240917T0130"

    def test_plotter_geometry_and_projection_helpers(
        self: "TestWrapperResults",
    ) -> None:
        """
        This test verifies that render options override the worker's default figure size, that dpi is only passed when requested, and that projection options are exported for the plot styles.

        Parameters:
            None

        Returns:
            None
        """
        assert pw._plotter_geometry(None, (10, 14)) == {"figsize": (10, 14)}
        render = pw.BatchRenderOptions(
            figure_size=(6.0, 5.0), dpi=72, projection="Robinson"
        )
        assert pw._plotter_geometry(render, (10, 14)) == {
            "figsize": (6.0, 5.0),
            "dpi": 72,
        }
        assert pw._projection_style_kwargs(render)["projection"] == "Robinson"
        assert pw._projection_style_kwargs(None) == {}
        assert pw._with_max_height(None, 5.0).max_height == 5.0
        assert pw._with_max_height(None, None) is None
        assert isinstance(pd.Timestamp("2024-01-01"), pd.Timestamp)

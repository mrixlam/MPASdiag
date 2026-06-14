#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Test Suite: Grid-file caching and MPI-aware shared loading

This module contains tests for the grid-file caching mechanism and MPI-aware shared loading functionality in mpasdiag.processing.base. It verifies that the grid cache is properly cleared, that the broadcast behavior can be controlled via environment variables, and that the collective loading mechanism correctly shares grid data across MPI ranks without redundant reads. The tests also ensure that the caching mechanism correctly handles hits and misses, and that real data loads populate the cache as expected.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: June 2026
Version: 1.0.0
"""

import numpy as np
import pytest
from typing import Iterator
from mpasdiag.processing import base
from mpasdiag.processing import clear_grid_cache, collective_grid_load


@pytest.fixture(autouse=True)
def _isolate_grid_cache() -> Iterator[None]:
    """
    This pytest fixture is automatically applied to all tests in this module to ensure that the grid cache is cleared before and after each test. It calls `clear_grid_cache()` before yielding control to the test, and then calls it again after the test completes. This guarantees that each test runs with a clean cache state, preventing interference from previous tests and ensuring that the caching behavior is tested accurately.

    Parameters:
        None

    Returns:
        Iterator[None]: A generator that yields control to the test and then continues to clear the grid cache after the test completes, ensuring that the cache is reset for the next test.
    """
    clear_grid_cache()
    yield
    clear_grid_cache()


def test_clear_grid_cache_empties_caches() -> None:
    """
    This test verifies that the `clear_grid_cache` function properly empties both the `_GRID_DS_CACHE` and `_UXGRID_CACHE` dictionaries. It first populates these caches with dummy objects, then calls `clear_grid_cache()`, and finally asserts that both caches are empty. This ensures that the cache clearing mechanism works as intended, allowing for a clean state for subsequent tests or operations.

    Parameters:
        None

    Returns:
        None
    """
    base._GRID_DS_CACHE[("dummy", None)] = object()
    base._UXGRID_CACHE["dummy"] = object()

    clear_grid_cache()

    assert base._GRID_DS_CACHE == {}
    assert base._UXGRID_CACHE == {}


def test_grid_bcast_enabled_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    This test verifies that the grid broadcast feature is enabled by default when the `MPASDIAG_GRID_BCAST` environment variable is not set. It uses `monkeypatch.delenv` to ensure that the environment variable is removed, and then asserts that `_grid_bcast_enabled` returns `True`, confirming that the default behavior is to enable grid broadcasting across MPI ranks.

    Parameters:
        monkeypatch: pytest.MonkeyPatch - A fixture provided by pytest to modify environment variables for testing purposes.

    Returns:
        None
    """
    monkeypatch.delenv("MPASDIAG_GRID_BCAST", raising=False)
    assert base._grid_bcast_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "False", "no", "OFF"])
def test_grid_bcast_disabled_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """
    This test verifies that the grid broadcast feature is correctly disabled for various environment variable values that represent a "false" state. It uses `monkeypatch.setenv` to set the `MPASDIAG_GRID_BCAST` environment variable to each value in the parameterized list, and then asserts that `_grid_bcast_enabled` returns `False`. This confirms that the system correctly recognizes different representations of a "false" value to disable grid broadcasting across MPI ranks.

    Parameters:
        monkeypatch: pytest.MonkeyPatch - A fixture provided by pytest to modify environment variables for testing purposes.
        value: str - The value to set for the `MPASDIAG_GRID_BCAST` environment variable.

    Returns:
        None
    """
    monkeypatch.setenv("MPASDIAG_GRID_BCAST", value)
    assert base._grid_bcast_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "anything"])
def test_grid_bcast_enabled_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """
    This test verifies that the grid broadcast feature is correctly enabled for various environment variable values that represent a "true" state. It uses `monkeypatch.setenv` to set the `MPASDIAG_GRID_BCAST` environment variable to each value in the parameterized list, and then asserts that `_grid_bcast_enabled` returns `True`. This confirms that the system correctly recognizes different representations of a "true" value to enable grid broadcasting across MPI ranks.

    Parameters:
        monkeypatch: pytest.MonkeyPatch - A fixture provided by pytest to modify environment variables for testing purposes.
        value: str - The value to set for the `MPASDIAG_GRID_BCAST` environment variable.

    Returns:
        None
    """
    monkeypatch.setenv("MPASDIAG_GRID_BCAST", value)
    assert base._grid_bcast_enabled() is True


def test_get_node_comm_none_when_bcast_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    This test ensures that when the grid broadcast feature is disabled via the `MPASDIAG_GRID_BCAST` environment variable, the `_get_node_comm` function returns `(None, None)`, indicating that no per-node collective communicator is being used. This confirms that the system correctly prevents the use of a node-level communicator for grid broadcasting when the feature is turned off, ensuring that each MPI rank operates independently without attempting to share grid data across ranks.

    Parameters:
        monkeypatch: pytest.MonkeyPatch - A fixture provided by pytest to modify environment variables for testing purposes.

    Returns:
        None
    """
    monkeypatch.setenv("MPASDIAG_GRID_BCAST", "0")
    with collective_grid_load():
        assert base._get_node_comm() == (None, None)


def test_get_node_comm_none_outside_collective_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    This test verifies that even when the grid broadcast feature is enabled via the `MPASDIAG_GRID_BCAST` environment variable, the `_get_node_comm` function returns `(None, None)` when called outside of a `collective_grid_load` context. This confirms that the node-level communicator for grid broadcasting is only active within the scope of the `collective_grid_load` context manager, and that outside of this context, the system does not attempt to use a shared communicator for grid data, ensuring that MPI ranks operate independently when not explicitly in a collective loading region.

    Parameters:
        monkeypatch: pytest.MonkeyPatch - A fixture provided by pytest to modify environment variables for testing purposes.

    Returns:
        None
    """
    monkeypatch.setenv("MPASDIAG_GRID_BCAST", "1")
    assert base._COLLECTIVE_GRID_LOAD_DEPTH == 0
    assert base._get_node_comm() == (None, None)


def test_collective_grid_load_is_reentrant_and_resets() -> None:
    """
    This test verifies that the `collective_grid_load` context manager is reentrant, meaning that it can be entered multiple times without causing issues, and that it correctly maintains and resets the `_COLLECTIVE_GRID_LOAD_DEPTH` counter. It checks that the depth counter increments as expected when entering nested contexts and decrements properly when exiting, ultimately returning to zero after all contexts are exited. This ensures that the collective loading mechanism can be used safely in nested scenarios without leaving the system in an inconsistent state.

    Parameters:
        None

    Returns:
        None
    """
    assert base._COLLECTIVE_GRID_LOAD_DEPTH == 0
    with collective_grid_load():
        assert base._COLLECTIVE_GRID_LOAD_DEPTH == 1
        with collective_grid_load():
            assert base._COLLECTIVE_GRID_LOAD_DEPTH == 2
        assert base._COLLECTIVE_GRID_LOAD_DEPTH == 1
    assert base._COLLECTIVE_GRID_LOAD_DEPTH == 0


def test_collective_grid_load_resets_on_exception() -> None:
    """
    This test ensures that if an exception is raised within a `collective_grid_load` context, the `_COLLECTIVE_GRID_LOAD_DEPTH` counter is properly reset to zero, preventing the system from being left in an inconsistent state. It verifies that even when an error occurs, the context manager's exit logic correctly handles the exception and resets the depth counter, ensuring that subsequent operations are not affected by a lingering non-zero depth. This is crucial for maintaining the integrity of the collective loading mechanism and ensuring that it can be safely used in scenarios where errors may occur without causing cascading issues in later code.

    Parameters:
        None

    Returns:
        None
    """
    assert base._COLLECTIVE_GRID_LOAD_DEPTH == 0
    with pytest.raises(RuntimeError):
        with collective_grid_load():
            assert base._COLLECTIVE_GRID_LOAD_DEPTH == 1
            raise RuntimeError("boom")
    assert base._COLLECTIVE_GRID_LOAD_DEPTH == 0


def test_load_shared_cache_hit_skips_reader() -> None:
    """
    This test verifies that when a cache hit occurs in the `_load_shared` function, the reader function is not invoked and the cached value is returned directly. It populates the `_UXGRID_CACHE` with a sentinel object for a specific key, then defines a reader function that raises an error if called. The test asserts that calling `_load_shared` with the same key returns the sentinel object without invoking the reader, confirming that the caching mechanism correctly bypasses the reader on a cache hit.

    Parameters:
        None

    Returns:
        None
    """
    sentinel = object()
    base._UXGRID_CACHE["k"] = sentinel

    def reader():
        raise AssertionError("reader must not run on a cache hit")

    assert base._load_shared(base._UXGRID_CACHE, "k", reader) is sentinel


def test_load_shared_cache_miss_reads_and_stores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    This test verifies that when a cache miss occurs in the `_load_shared` function, the reader function is invoked exactly once to read the value, and that the value is stored in the cache for subsequent hits. It uses a `calls` dictionary to track how many times the reader is called, and asserts that on the first call to `_load_shared`, the reader is invoked and returns the expected value. On a second call with the same key, it asserts that the cached value is returned and that the reader is not invoked again, confirming that the caching mechanism correctly stores and retrieves values on cache misses and hits.

    Parameters:
        monkeypatch: pytest.MonkeyPatch - A fixture provided by pytest to modify environment variables for testing purposes.

    Returns:
        None
    """
    monkeypatch.setenv("MPASDIAG_GRID_BCAST", "0")
    calls = {"n": 0}

    def reader() -> str:
        """
        This inner function serves as the reader for the `_load_shared` function. It increments a call count in the `calls` dictionary to track how many times it has been invoked, and then returns a string value "value". This allows the test to verify that the reader is called exactly once on a cache miss, and that subsequent calls to `_load_shared` with the same key do not invoke the reader again.

        Parameters:
            None

        Returns:
            str: The value read by the reader.
        """
        calls["n"] += 1
        return "value"

    assert base._load_shared(base._GRID_DS_CACHE, ("k", None), reader) == "value"
    assert base._load_shared(base._GRID_DS_CACHE, ("k", None), reader) == "value"
    assert calls["n"] == 1


def test_load_shared_falls_back_when_node_comm_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    This test verifies that if the `_get_node_comm` function raises an exception (simulating a failure in setting up the node-level communicator for grid broadcasting), the `_load_shared` function correctly falls back to a direct read using the reader function. It monkeypatches `_get_node_comm` to raise a `RuntimeError`, then defines a reader function that increments a call count and returns a value. The test asserts that `_load_shared` returns the expected value from the reader and that the reader is called exactly once, confirming that the fallback mechanism works as intended when the node communicator cannot be established.

    Parameters:
        monkeypatch: pytest.MonkeyPatch - A fixture provided by pytest to modify environment variables for testing purposes.

    Returns:
        None
    """

    class _BadComm:
        def Get_rank(self: "_BadComm") -> int:
            raise RuntimeError("boom")

    monkeypatch.setattr(base, "_get_node_comm", lambda: (None, _BadComm()))
    calls = {"n": 0}

    def reader() -> int:
        calls["n"] += 1
        return 123

    # The broadcast attempt raises, so every rank falls back to a direct read.
    assert base._load_shared(base._UXGRID_CACHE, "k", reader) == 123
    assert calls["n"] == 1


def _load_two_processors() -> tuple[base.MPASBaseProcessor, base.MPASBaseProcessor]:
    """
    This helper function loads two MPAS processors with the same grid file and data directory to test the caching mechanism. It attempts to load the processors using the `MPAS2DProcessor` class, and if the necessary data or grid file is not available, it skips the test. The function returns a tuple of two loaded MPAS processors, which can then be used in tests to verify that they share the same cached grid information and that their outputs are identical.

    Parameters:
        None

    Returns:
        tuple[base.MPASBaseProcessor, base.MPASBaseProcessor]: Two loaded MPAS processors.
    """
    from tests.test_data_helpers import _grid_file_path
    from mpasdiag.processing.processors_2d import MPAS2DProcessor
    from pathlib import Path

    data_dir = Path(__file__).parent.parent / "data" / "u240k" / "diag"

    if not data_dir.exists():
        pytest.skip("u240k sample data not available")

    try:
        grid_file = _grid_file_path()
    except Exception:
        pytest.skip("grid file not available")

    p1 = MPAS2DProcessor(grid_file=grid_file, verbose=False).load_2d_data(str(data_dir))
    p2 = MPAS2DProcessor(grid_file=grid_file, verbose=False).load_2d_data(str(data_dir))
    return p1, p2


def test_real_load_populates_cache_and_shares_uxgrid() -> None:
    """
    This test verifies that when two MPAS processors are loaded with the same grid file and data directory, the grid information is properly cached and shared between them. It checks that only one grid dataset and one uxgrid are read into memory, and that both processors reference the same cached uxgrid object. This confirms that the caching mechanism is working correctly to avoid redundant reads of the grid file and to share the unstructured grid information across processors that use the same grid.

    Parameters:
        None

    Returns:
        None
    """
    p1, p2 = _load_two_processors()

    # Exactly one grid dataset and one uxgrid were read despite two processors.
    assert len(base._UXGRID_CACHE) == 1
    assert len(base._GRID_DS_CACHE) == 1

    # Both processors reference the SAME cached uxgrid object.
    assert p1.dataset.uxgrid is p2.dataset.uxgrid


def test_real_load_results_identical_across_cache_hits() -> None:
    """
    This test verifies that when two MPAS processors are loaded with the same grid file and data directory, and the grid information is cached and shared between them, the extracted coordinates and variable data from both processors are identical. It checks that the longitude and latitude coordinates for a specific variable (e.g., "t2m") are the same in both processors, and that the 2D variable data for the same variable and time index are also identical. This confirms that the caching mechanism does not alter the data and that both processors produce consistent outputs when using the shared cached grid information.

    Parameters:
        None

    Returns:
        None
    """
    p1, p2 = _load_two_processors()

    lon1, lat1 = p1.extract_2d_coordinates_for_variable("t2m")
    lon2, lat2 = p2.extract_2d_coordinates_for_variable("t2m")
    assert np.array_equal(lon1, lon2)
    assert np.array_equal(lat1, lat2)

    d1 = p1.get_2d_variable_data("t2m", time_index=0).values
    d2 = p2.get_2d_variable_data("t2m", time_index=0).values
    assert np.array_equal(d1, d2)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Core Processing Module: Parallel Processing Wrappers

This module provides worker functions and parallel processing management for executing MPAS diagnostic computations and visualizations in parallel across multiple time steps. It includes worker functions for precipitation maps, surface variable maps, wind vector plots, and vertical cross-sections, each designed to be picklable for use with multiprocessing or MPI-based parallel execution. The module also contains a helper function to aggregate results from parallel workers and generate performance reports summarizing the outcomes of the parallel processing operations.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import matplotlib

matplotlib.use("Agg")

import contextlib
import gc
import os
import pickle
import signal
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Any, Dict, Iterator, Union, cast
import numpy as np
import pandas as pd
import xarray as xr

try:
    from .data_cache import MPASDataCache, get_global_cache  # noqa: F401
except ImportError:
    from mpasdiag.processing.data_cache import (  # noqa: F401
        MPASDataCache,
        get_global_cache,
    )

try:
    from .parallel import MPASParallelManager
    from .processors_2d import MPAS2DProcessor
    from .processors_3d import MPAS3DProcessor
except ImportError:
    from mpasdiag.processing.parallel import MPASParallelManager
    from mpasdiag.processing.processors_2d import MPAS2DProcessor
    from mpasdiag.processing.processors_3d import MPAS3DProcessor

try:
    from ..visualization.precipitation import MPASPrecipitationPlotter
    from ..visualization.surface import MPASSurfacePlotter, SurfaceMapStyle
    from ..visualization.wind import MPASWindPlotter, WindPlotStyle
    from ..visualization.cross_section import (
        MPASVerticalCrossSectionPlotter,
        CrossSectionStyle,
    )
    from ..visualization.skewt import MPASSkewTPlotter
    from ..diagnostics.precipitation import PrecipitationDiagnostics
    from ..diagnostics.sounding import SoundingDiagnostics
except ImportError:
    from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter
    from mpasdiag.visualization.surface import MPASSurfacePlotter, SurfaceMapStyle
    from mpasdiag.visualization.wind import MPASWindPlotter, WindPlotStyle
    from mpasdiag.visualization.cross_section import (
        MPASVerticalCrossSectionPlotter,
        CrossSectionStyle,
    )
    from mpasdiag.visualization.skewt import MPASSkewTPlotter
    from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
    from mpasdiag.diagnostics.sounding import SoundingDiagnostics

from mpasdiag.processing.utils_geog import GeographicBounds
from mpasdiag.processing.utils_path import (
    safe_resolve_within,
    sanitize_filename_component,
)
from mpasdiag.processing.utils_datetime import MPASDateTimeUtils
from mpasdiag.processing.constants import (
    PRECIP_REQUIRED_VARS,
    CROSS_SECTION_AUX_VARS,
    COORDS_FALLBACK_MSG,
    NETCDF_ENGINE,
    PRELOAD_COORDS_MSG,
)
from mpasdiag.processing.utils_logger import get_logger, mpi_launcher_detected
from mpasdiag.visualization.precipitation import (
    PrecipitationRenderStyle,
    PrecipitationMapStyle,
)

try:
    from mpasdiag.diagnostics.precipitation import InsufficientLookbackError
except ImportError:  # pragma: no cover - older diagnostics without the class

    class InsufficientLookbackError(ValueError):  # type: ignore[no-redef]
        """Fallback used when the diagnostics module does not define InsufficientLookbackError."""


logger = get_logger(__name__)


@dataclass
class RemapConfig:
    """Remapping settings for parallel batch plotting, grouping the remap engine, interpolation method, and ESMF weights cache directory into a single value object."""

    remap_engine: str = "kdtree"
    remap_method: str = "nearest"
    weights_dir: Optional[str] = None


class BatchFiles(List[str]):
    """List of created file paths that also carries the per-step outcome counts of a parallel batch (successful, failed, and skipped time steps), so callers can derive an exit status without changing the list-of-paths return contract."""

    successful: int = 0
    failed: int = 0
    skipped: int = 0


@dataclass
class BatchRenderOptions:
    """Rendering options shared by the parallel batch workers that the per-plot style dataclasses do not carry: color limits, figure size and resolution, the map projection with its centering and keyword controls, the wind vector color and transparency, the data chunk size along nCells, and the raw vertical-level selection used when the wind components are 3D variables (the label in wind_level is only used for the title)."""

    clim_min: Optional[float] = None
    clim_max: Optional[float] = None
    figure_size: Optional[Tuple[float, float]] = None
    dpi: Optional[int] = None
    projection: str = "PlateCarree"
    central_longitude: Optional[float] = None
    central_latitude: Optional[float] = None
    proj_kwargs: Optional[Dict[str, Any]] = None
    title: Optional[str] = None
    wind_level: str = "surface"
    background_colormap: str = "viridis"
    max_height: Optional[float] = None
    vector_color: str = "black"
    vector_alpha: float = 1.0
    chunk_size: Optional[int] = None
    wind_level_selection: Optional[Union[str, int, float]] = None


@dataclass
class SurfaceBatchStyle:
    """Appearance and file-naming settings for parallel batch surface maps, grouping the output filename prefix, plot type, color scale limits, colormap, explicit contour levels, and the number of automatically generated levels into a single value object."""

    file_prefix: str = "mpas_surface"
    plot_type: str = "scatter"
    clim_min: Optional[float] = None
    clim_max: Optional[float] = None
    colormap: Optional[str] = None
    levels: Optional[List[float]] = None
    num_levels: Optional[int] = None
    interpolation: Optional[str] = None


@dataclass
class WindBatchStyle:
    """Rendering and regridding settings for parallel batch wind plots, grouping the plot type, subsampling factor, arrow scale, speed-background toggle, grid resolution, and regrid method into a single value object."""

    plot_type: str = "barbs"
    subsample: int = 1
    scale: Optional[float] = None
    show_background: bool = False
    grid_resolution: Optional[float] = None
    regrid_method: str = "linear"


@dataclass
class CrossSectionBatchStyle:
    """Appearance and file-naming settings for parallel batch cross-section plots, grouping the contour levels, colormap, colorbar extend direction, plot type, and output filename prefix into a single value object."""

    levels: Optional[np.ndarray] = None
    colormap: Optional[str] = "viridis"
    extend: str = "both"
    plot_type: str = "contourf"
    file_prefix: str = "mpas_cross_section"
    clim_min: Optional[float] = None
    clim_max: Optional[float] = None


_DATETIME_HOUR_FORMAT = "%Y%m%dT%H"
_FAILED_TIME_INDEX_MSG = "Failed time index %s: %s"
_COORD_CACHE_MSG = "Coordinates cached for variable: %s"
_COORD_PRELOAD_MSG = "Could not pre-load coordinates into cache: %s"
_OUTDIR_MSG = "Output directory: %s"

_rank_processor_cache: Dict[str, Any] = {}

# Largest pickled processor shipped to multiprocessing workers; above it workers reload from disk instead
_MAX_PROCESSOR_SEED_BYTES = 256 * 1024**2

_GC_INTERVAL = 20
_gc_task_counter = 0


def _maybe_collect_garbage() -> None:
    """
    This helper runs a full garbage collection only periodically (every _GC_INTERVAL tasks) rather than after every single task. A per-task gc.collect() must walk the entire heap, which in the parallel workers holds the cached full dataset plus accumulated cartopy/matplotlib state, costing on the order of 0.1 seconds per task -- roughly a third of warm task time -- and that cost never amortizes across a long batch. Python's automatic generational collector still runs continuously to reclaim the bulk of per-task garbage; this periodic full sweep simply bounds any cyclic growth at a small fraction of the previous cost.

    Parameters:
        None

    Returns:
        None
    """
    global _gc_task_counter
    _gc_task_counter += 1
    if _gc_task_counter % _GC_INTERVAL == 0:
        gc.collect()


def _render_chunk_size(kwargs: Dict[str, Any]) -> Optional[int]:
    """
    This helper reads the requested data chunk size from the worker keyword arguments, where it travels inside the shared render options, so a worker that reloads its data from disk applies the same chunking the user asked for on the command line. It returns None when no render options or no chunk size were supplied, which keeps the loaders' default chunking.

    Parameters:
        kwargs (Dict[str, Any]): Worker keyword arguments that may carry a 'render' BatchRenderOptions.

    Returns:
        Optional[int]: The chunk size along nCells, or None for the default chunking.
    """
    render = kwargs.get("render")
    return getattr(render, "chunk_size", None) if render is not None else None


def _load_with_chunk_size(
    loader: Any,
    data_dir: str,
    variables: Optional[List[str]],
    chunk_size: Optional[int],
) -> Any:
    """
    This helper calls a processor's load method (load_2d_data or load_3d_data) and forwards the chunk size only when one was requested, so processors and test doubles whose loaders do not accept the argument keep working with their default chunking.

    Parameters:
        loader (Any): The bound load_2d_data or load_3d_data method.
        data_dir (str): Directory containing the data files.
        variables (Optional[List[str]]): Variables to load, or None for the default set.
        chunk_size (Optional[int]): Chunk size along nCells, or None for the default chunking.

    Returns:
        Any: The processor returned by the loader.
    """
    if chunk_size is None:
        return loader(data_dir, variables=variables)
    return loader(data_dir, variables=variables, chunk_size=chunk_size)


def _get_or_create_2d_processor(kwargs: Dict[str, Any]) -> Any:
    """
    This helper function retrieves or creates a 2D processor instance based on the provided keyword arguments. It constructs a cache key using the grid file, data directory, and optionally the list of variables to ensure that each unique combination corresponds to a single processor instance in the cache. If the processor for the given key does not exist in the cache, it creates a new MPAS2DProcessor instance, loads the 2D data from the specified directory, and stores it in the cache before returning it. This design allows for efficient reuse of processor instances across multiple worker functions that may require access to the same data, while also supporting multiprocessing scenarios where each worker may need to create its own processor instance if they cannot share state.

    Parameters:
        kwargs (Dict[str, Any]): Dictionary of keyword arguments that must include 'grid_file' and 'data_dir', and may optionally include 'variables' which is a list of variable names to load.

    Returns:
        Any: An instance of MPAS2DProcessor that has loaded the specified data, either retrieved from the cache or newly created if it was not already present.
    """
    grid_file = kwargs["grid_file"]
    data_dir = kwargs["data_dir"]
    variables = kwargs.get("variables", None)
    chunk_size = _render_chunk_size(kwargs)
    cache_key = _processor_cache_key("2d", kwargs)

    if cache_key not in _rank_processor_cache:
        from mpasdiag.processing.processors_2d import MPAS2DProcessor

        processor = MPAS2DProcessor(grid_file, verbose=False)
        processor = _load_with_chunk_size(
            processor.load_2d_data, data_dir, variables, chunk_size
        )
        _rank_processor_cache[cache_key] = processor
    return _rank_processor_cache[cache_key]


def _get_or_create_3d_processor(kwargs: Dict[str, Any]) -> Any:
    """
    This helper function retrieves or creates a 3D processor instance based on the provided keyword arguments. It constructs a cache key using the grid file, data directory, and optionally the list of variables to ensure that each unique combination corresponds to a single processor instance in the cache. If the processor for the given key does not exist in the cache, it creates a new MPAS3DProcessor instance, loads the 3D data from the specified directory, and stores it in the cache before returning it. This design allows for efficient reuse of processor instances across multiple worker functions that may require access to the same 3D data, while also supporting multiprocessing scenarios where each worker may need to create its own processor instance if they cannot share state.

    Parameters:
        kwargs (Dict[str, Any]): Dictionary of keyword arguments that must include 'grid_file' and 'data_dir', and may optionally include 'variables' which is a list of variable names to load.

    Returns:
        Any: An instance of MPAS3DProcessor that has loaded the specified data, either retrieved from the cache or newly created if it was not already present.
    """
    grid_file = kwargs["grid_file"]
    data_dir = kwargs["data_dir"]
    variables = kwargs.get("variables", None)
    chunk_size = _render_chunk_size(kwargs)
    cache_key = _processor_cache_key("3d", kwargs)

    if cache_key not in _rank_processor_cache:
        from mpasdiag.processing.processors_3d import MPAS3DProcessor

        processor = MPAS3DProcessor(grid_file, verbose=False)
        processor = _load_with_chunk_size(
            processor.load_3d_data, data_dir, variables, chunk_size
        )
        _rank_processor_cache[cache_key] = processor
    return _rank_processor_cache[cache_key]


def _setup_processor_and_cache(
    kwargs: Dict[str, Any],
) -> Tuple[Any, Optional[MPASDataCache]]:
    """
    This helper function sets up the processor and cache for a worker function based on the provided keyword arguments. It checks if 'grid_file' and 'data_dir' are present in kwargs to determine if it should create a new MPAS2DProcessor instance and load data directly within the worker, which is useful for multiprocessing where objects need to be picklable. If these keys are not present, it assumes that a processor instance and an optional cache object have been passed in kwargs and returns them directly. This design allows for flexibility in how the worker functions can be executed in parallel, supporting both multiprocessing with independent data loading and MPI-based parallelism with shared processor instances and caches.

    Parameters:
        kwargs (Dict[str, Any]): Dictionary of keyword arguments that may contain 'grid_file', 'data_dir', 'processor', and 'cache' keys to determine how to set up the processor and cache for the worker function.

    Returns:
        Tuple[Any, Optional[MPASDataCache]]: A tuple containing the processor instance (which may be a newly created MPAS2DProcessor or an existing processor passed in kwargs) and an optional MPASDataCache instance if provided in kwargs. If 'grid_file' and 'data_dir' are used to create a new processor, the cache will be returned as None since it is not shared across processes in that case.
    """
    if "grid_file" in kwargs and "data_dir" in kwargs:
        processor = _get_or_create_2d_processor(kwargs)
        return processor, None
    return kwargs["processor"], kwargs.get("cache", None)


def _path_str(value: Any) -> Optional[str]:
    """
    This helper converts a path-like value to a plain string, returning None when the value is not a str or os.PathLike (for example None for an in-memory processor).

    Parameters:
        value (Any): A candidate path attribute.

    Returns:
        Optional[str]: The path as a string, or None if the value is not path-like.
    """
    if isinstance(value, (str, os.PathLike)):
        try:
            return os.fspath(value) if not isinstance(value, bytes) else None
        except TypeError:
            return None
    return None


def _processor_supports_reload(processor: Any) -> bool:
    """
    This helper reports whether a processor can be reconstructed from disk inside a worker, which is true when it exposes 'grid_file' and 'data_dir' paths (str or os.PathLike such as pathlib.Path, as set by load_2d_data/load_3d_data). When True, the parallel batch methods hand those lightweight string paths to workers instead of the live processor, so each worker (multiprocessing) or rank (MPI) loads ONLY the variables it needs and does so once -- caching the result in the module-level processor cache -- rather than having the full in-memory dataset pickled to it for every single task. In-memory processors built without loading from disk return False; the multiprocessing batch then runs serially rather than pickling the processor to workers, and MPI mode raises.

    Parameters:
        processor (Any): The 2D or 3D processor instance to inspect.

    Returns:
        bool: True if both 'grid_file' and 'data_dir' are path-like.
    """
    return (
        _path_str(getattr(processor, "grid_file", None)) is not None
        and _path_str(getattr(processor, "data_dir", None)) is not None
    )


def _reload_paths(processor: Any) -> Dict[str, str]:
    """
    This helper returns the grid file and data directory of a reload-capable processor as plain strings, the picklable form handed to workers.

    Parameters:
        processor (Any): A processor for which _processor_supports_reload is True.

    Returns:
        Dict[str, str]: Dictionary with the string keys 'grid_file' and 'data_dir'.
    """
    return {
        "grid_file": str(_path_str(processor.grid_file)),
        "data_dir": str(_path_str(processor.data_dir)),
    }


def _create_batch_manager(
    processor: Any, load_balance_strategy: str, n_processes: Optional[int]
) -> "MPASParallelManager":
    """
    This helper creates the parallel manager for a batch with the collect error policy. When the manager resolves to the multiprocessing backend but the processor cannot be reloaded from disk (an in-memory processor with no grid file or data directory), the batch is switched to serial execution with a warning, because shipping the whole processor to every worker process is unsafe: it was observed to crash workers while unpickling and to be very slow. MPI ranks already share the in-process processor, so the MPI backend is left alone.

    Parameters:
        processor (Any): The processor whose reload capability decides the backend.
        load_balance_strategy (str): Task distribution strategy for MPI ('cyclic', 'static', ...).
        n_processes (Optional[int]): Requested worker-process count for multiprocessing.

    Returns:
        MPASParallelManager: The configured manager.
    """
    manager = MPASParallelManager(
        load_balance_strategy=load_balance_strategy,
        verbose=True,
        n_workers=n_processes,
    )
    manager.set_error_policy("collect")

    if manager.backend == "multiprocessing" and not _processor_supports_reload(
        processor
    ):
        logger.warning(
            "The processor was not loaded from files (no grid_file/data_dir), so it "
            "cannot be sent to worker processes; running the batch serially instead"
        )
        manager.use_serial_backend()

    return manager


def _batch_output_path(
    output_dir: str, output_name: Optional[str], default_stem: str, time_str: str
) -> str:
    """
    This helper builds the output path (without extension) of one batch plot exactly as the serial command-line path does: with a user-supplied output name the file is '<output_name>_<time_str>' confined to the output directory, otherwise it is the descriptive default stem inside the output directory.

    Parameters:
        output_dir (str): Directory that receives the plots.
        output_name (Optional[str]): The user-supplied --output name, or None.
        default_stem (str): Descriptive file name used when no output name was given.
        time_str (str): Valid-time tag appended to a user-supplied name.

    Returns:
        str: Output path without extension.
    """
    if output_name:
        return str(safe_resolve_within(f"{output_name}_{time_str}", output_dir))
    return os.path.join(output_dir, default_stem)


def _apply_weights_dir(plotter: Any, kwargs: Dict[str, Any]) -> None:
    """
    This helper applies the ESMF weights cache directory from the worker kwargs to a plotter, so every plot type honours RemapConfig.weights_dir the way the precipitation plot does.

    Parameters:
        plotter (Any): The plotter whose remapper weights directory is set.
        kwargs (Dict[str, Any]): Worker kwargs that may carry 'weights_dir'.

    Returns:
        None
    """
    weights_dir = kwargs.get("weights_dir")

    if weights_dir is not None:
        from pathlib import Path

        plotter._remapper_weights_dir = Path(weights_dir)


def _seed_worker_processor_cache(
    kind: str, worker_kwargs: Dict[str, Any], processor: Any
) -> Optional[str]:
    """
    This helper lets an MPI rank hand its already-loaded processor to the in-process batch worker instead of having the worker reload the grid and data from disk a second time. Because each MPI rank runs its worker inside the same process, the live processor is stored in the module-level worker cache under the exact key that _get_or_create_2d_processor / _get_or_create_3d_processor would compute for the given worker kwargs, so the worker's lookup becomes a cache hit that returns the existing processor with no extra read. To stay strictly as safe as a fresh reload, it only seeds when the processor's dataset actually exposes every requested variable; otherwise it returns None and the worker reloads exactly as before. It returns the seeded cache key so the caller can evict the entry once the batch finishes (keeping processors from accumulating across experiments and ranks), or None when seeding does not apply -- including the in-memory multiprocessing fallback, whose separate worker processes cannot see this cache anyway.

    Parameters:
        kind (str): Cache namespace matching the worker loader, '2d' or '3d'.
        worker_kwargs (Dict[str, Any]): The kwargs handed to the worker, providing 'grid_file', 'data_dir', and optional 'variables'.
        processor (Any): The already-loaded processor whose dataset should be reused in-process.

    Returns:
        Optional[str]: The seeded cache key for later eviction, or None if seeding does not apply.
    """
    grid_file = worker_kwargs.get("grid_file")
    data_dir = worker_kwargs.get("data_dir")

    if not isinstance(grid_file, str) or not isinstance(data_dir, str):
        return None

    if not _dataset_has_variables(processor, worker_kwargs.get("variables")):
        return None

    cache_key = _processor_cache_key(kind, worker_kwargs)
    _rank_processor_cache[cache_key] = processor
    return cache_key


def _processor_cache_key(kind: str, kwargs: Dict[str, Any]) -> str:
    """
    This helper builds the worker processor cache key from the loader kind and the worker kwargs (grid file, data directory, requested variables and chunk size). The getters, the MPI in-process seeding and the multiprocessing processor seed all use it, so a seeded processor is found under exactly the key a worker would look up.

    Parameters:
        kind (str): Cache namespace matching the worker loader, '2d' or '3d'.
        kwargs (Dict[str, Any]): Worker kwargs providing 'grid_file', 'data_dir' and optional 'variables'.

    Returns:
        str: The cache key.
    """
    variables = kwargs.get("variables")
    return (
        f"{kind}|{kwargs['grid_file']}|{kwargs['data_dir']}"
        f"|{tuple(sorted(variables)) if variables else None}"
        f"|{_render_chunk_size(kwargs)}"
    )


def _dataset_has_variables(processor: Any, variables: Optional[List[str]]) -> bool:
    """
    This helper reports whether a processor's loaded dataset can stand in for the dataset a worker would load itself, the condition for handing that processor to a worker instead of letting the worker reload the requested variables from disk. Every requested variable must be in the dataset, except variables that are absent from the first data file: the loader skips those when it reloads (3D workers request optional auxiliary fields such as 'zgrid' or 'height'), so their absence here changes nothing.

    Parameters:
        processor (Any): The loaded processor.
        variables (Optional[List[str]]): Variables the worker needs, or None for the default set.

    Returns:
        bool: True if the dataset is loaded and covers every requested variable a reload would contain.
    """
    dataset = getattr(processor, "dataset", None)
    if dataset is None:
        return False
    missing = []
    for variable in variables or []:
        try:
            _ = dataset[variable]
        except Exception:
            missing.append(variable)
    if not missing:
        return True
    try:
        first_file = processor._discover_data_files(processor.data_dir)[0]
        with xr.open_dataset(first_file, engine=NETCDF_ENGINE) as probe:
            in_files = set(probe.data_vars)
    except Exception:
        return False
    return not in_files.intersection(missing)


def _make_processor_seed(
    kind: str, worker_kwargs: Dict[str, Any], processor: Any
) -> Optional[Tuple[str, str]]:
    """
    This helper prepares the parent's already-loaded processor for multiprocessing workers. Without it every worker rebuilds the processor from disk (re-opening and combining every data file and reading the grid), which costs seconds per worker before its first task. The processor is pickled once here; its dataset is lazy, so the pickle holds the file references and task graph rather than the data, and each worker unpickles it in a fraction of a second and reads only the time steps it plots. The pickle is written once to a private temporary file (mode 0600) and only its path travels with the shared task arguments, so starting a worker never pushes megabytes through the start-method pipe (with 'spawn', a worker that dies before reading a large payload would block the parent inside Pool start-up, before dead-worker detection can run). The caller deletes the file when the batch ends. It returns None, so workers reload as before, when the processor lacks a requested variable, cannot be pickled, or pickles to more than _MAX_PROCESSOR_SEED_BYTES (for example a dataset that was loaded into memory).

    Parameters:
        kind (str): Cache namespace matching the worker loader, '2d' or '3d'.
        worker_kwargs (Dict[str, Any]): The kwargs handed to the worker.
        processor (Any): The parent's loaded processor.

    Returns:
        Optional[Tuple[str, str]]: The worker cache key and the path of the pickled processor, or None when workers should reload.
    """
    if not _processor_supports_reload(processor) or not _dataset_has_variables(
        processor, worker_kwargs.get("variables")
    ):
        return None
    try:
        blob = pickle.dumps(processor, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        logger.debug("Workers will reload the data: processor not picklable (%s)", exc)
        return None
    if len(blob) > _MAX_PROCESSOR_SEED_BYTES:
        logger.debug(
            "Workers will reload the data: pickled processor is %.0f MB",
            len(blob) / 1024**2,
        )
        return None
    try:
        fd, path = tempfile.mkstemp(prefix="mpasdiag-seed-", suffix=".pkl")
        with os.fdopen(fd, "wb") as handle:
            handle.write(blob)
    except OSError as exc:
        logger.debug("Workers will reload the data: cannot write seed file (%s)", exc)
        return None
    return _processor_cache_key(kind, worker_kwargs), path


@contextlib.contextmanager
def _sigterm_raises_exit() -> Iterator[None]:
    """
    This context manager turns SIGTERM into SystemExit while it is active, so a batch stopped by a scheduler (a SLURM or PBS wall-time limit sends SIGTERM) unwinds through its finally blocks and removes the processor seed file instead of leaving it in the temporary directory. It only acts in the main thread and only when SIGTERM still has its default action, so a handler installed by the application is left alone; pool workers restore the default action when they start.

    Parameters:
        None

    Returns:
        Iterator[None]: Context manager yielding nothing.
    """
    if (
        threading.current_thread() is not threading.main_thread()
        or signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
    ):
        yield
        return

    def _exit(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _exit)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


def _parallel_map_batch(
    manager: Any,
    worker: Any,
    tasks: List[Tuple[int, Dict[str, Any]]],
    kind: str,
    worker_kwargs: Dict[str, Any],
    processor: Any,
) -> Any:
    """
    This helper runs a batch worker over its tasks. For a multiprocessing pool whose workers would otherwise reload the data from disk, it ships the parent's loaded processor once per worker (see _make_processor_seed); every other backend, and any case where the processor cannot be shipped, runs exactly as before.

    Parameters:
        manager (Any): The MPASParallelManager running the batch.
        worker (Any): The batch worker function.
        tasks (List[Tuple[int, Dict[str, Any]]]): The (time index, worker kwargs) tasks.
        kind (str): Worker loader kind, '2d' or '3d'.
        worker_kwargs (Dict[str, Any]): The kwargs shared by every task.
        processor (Any): The parent's loaded processor.

    Returns:
        Any: The task results returned by the manager.
    """
    if manager.backend == "mpi":
        # Each rank runs its worker in this process, so the loaded processor is handed over through the
        # worker cache and evicted however the batch ends
        seeded_key = _seed_worker_processor_cache(kind, worker_kwargs, processor)
        try:
            return manager.parallel_map(worker, tasks)
        finally:
            if seeded_key is not None:
                _rank_processor_cache.pop(seeded_key, None)
    if manager.backend == "multiprocessing" and "grid_file" in worker_kwargs:
        seed = _make_processor_seed(kind, worker_kwargs, processor)
        if seed is not None:
            # A serial fallback runs the workers in this process and caches the seed here; drop that copy afterwards
            already_cached = seed[0] in _rank_processor_cache
            try:
                with _sigterm_raises_exit():
                    return manager.parallel_map(worker, tasks, processor_seed=seed)
            finally:
                if not already_cached:
                    _rank_processor_cache.pop(seed[0], None)
                try:
                    os.unlink(seed[1])
                except OSError:
                    pass
    return manager.parallel_map(worker, tasks)


def _is_private_seed(fd: int) -> bool:
    """
    This helper checks that an open processor seed file is still the private file the parent created with tempfile.mkstemp: owned by the current user and not writable by the group or by others. Unpickling runs code chosen by whoever wrote the file, so a seed that another user could have replaced is not loaded. On platforms without POSIX ownership (Windows) only the existence of the file is checked, as mkstemp's own permissions apply there.

    Parameters:
        fd (int): File descriptor of the opened seed file.

    Returns:
        bool: True if the file may be loaded.
    """
    info = os.fstat(fd)
    if not hasattr(os, "getuid"):
        return True
    return info.st_uid == os.getuid() and not info.st_mode & 0o022


def _install_processor_seed(processor_seed: Optional[Tuple[str, str]]) -> None:
    """
    This helper is called by every batch worker before it looks up its processor. When the parent shipped a processor seed and this worker has no processor under the seed's key yet, it loads the pickled processor from the seed file into the worker cache so the lookup becomes a cache hit. Any failure leaves the cache untouched and the worker reloads from disk as before.

    Parameters:
        processor_seed (Optional[Tuple[str, str]]): The (cache key, seed file path) pair from _make_processor_seed, or None.

    Returns:
        None
    """
    if processor_seed is None:
        return
    cache_key, path = processor_seed
    if cache_key in _rank_processor_cache:
        return
    try:
        with open(path, "rb") as handle:
            if not _is_private_seed(handle.fileno()):
                logger.debug(
                    "Ignoring a shipped processor file that is not private to this user"
                )
                return
            # the parent wrote this file with mkstemp (owner-only) for this batch, checked above
            _rank_processor_cache[cache_key] = pickle.load(handle)  # nosec B301
    except Exception as exc:
        logger.debug("Could not use the shipped processor (%s); reloading", exc)


def _extract_precip_coordinates(
    processor: Any,
    cache: Optional[MPASDataCache],
    var_name: str,
    cache_hits: Dict[str, bool],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This helper function extracts longitude and latitude coordinates for a given variable name using the provided processor and cache. It first checks if the cache is available and attempts to retrieve the coordinates from the cache, updating the cache_hits dictionary accordingly. If the coordinates are not found in the cache, it uses the processor to extract the 2D coordinates for the specified variable. If a cache is being used, it also attempts to load the coordinates into the cache from the processor's dataset for future use by other workers. This function abstracts away the logic of coordinate retrieval and caching, allowing worker functions to easily obtain spatial coordinates while tracking cache performance.

    Parameters:
        processor (Any): The processor instance used to extract coordinates if not found in cache.
        cache (Optional[MPASDataCache]): The cache object to retrieve or store coordinates.
        var_name (str): The name of the variable for which to extract coordinates.
        cache_hits (Dict[str, bool]): Dictionary to track cache hit status for coordinates.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Longitude and latitude arrays for the specified variable.
    """
    if cache is not None:
        try:
            lon, lat = cache.get_coordinates(var_name)
            cache_hits["coordinates"] = True
            return lon, lat
        except KeyError:
            lon, lat = processor.extract_2d_coordinates_for_variable(var_name)
            try:
                cache.load_coordinates_from_dataset(processor.dataset, var_name)
            except Exception:
                pass
            return lon, lat
    return cast(
        Tuple[np.ndarray, np.ndarray],
        processor.extract_2d_coordinates_for_variable(var_name),
    )


def _plotter_geometry(
    render: Optional[BatchRenderOptions], default_figsize: Tuple[float, float]
) -> Dict[str, Any]:
    """
    This helper builds the keyword arguments that size a plotter inside a batch worker. The figure size from the render options replaces the worker's hard-coded default when provided, and the resolution is only passed when it was explicitly requested so plotters keep their own default dpi otherwise. Centralizing this keeps the precipitation, surface, wind, and cross-section workers consistent with the serial command-line path, which honors --figure-size and --dpi.

    Parameters:
        render (Optional[BatchRenderOptions]): Shared rendering options, or None to use the worker defaults.
        default_figsize (Tuple[float, float]): Figure size in inches used when the render options do not specify one.

    Returns:
        Dict[str, Any]: Keyword arguments (figsize and, when requested, dpi) for the plotter constructor.
    """
    geometry: Dict[str, Any] = {"figsize": default_figsize}
    if render is not None:
        if render.figure_size is not None:
            geometry["figsize"] = tuple(render.figure_size)
        if render.dpi is not None:
            geometry["dpi"] = render.dpi
    return geometry


def _projection_style_kwargs(render: Optional[BatchRenderOptions]) -> Dict[str, Any]:
    """
    This helper extracts the projection, central longitude, central latitude, and projection keyword arguments from the render options as keyword arguments accepted by the SurfaceMapStyle, PrecipitationRenderStyle, and WindPlotStyle dataclasses. When no render options are supplied it returns an empty dictionary so the plot styles keep their defaults.

    Parameters:
        render (Optional[BatchRenderOptions]): Shared rendering options, or None.

    Returns:
        Dict[str, Any]: Projection-related keyword arguments for a plot style dataclass.
    """
    if render is None:
        return {}
    return {
        "projection": render.projection,
        "central_longitude": render.central_longitude,
        "central_latitude": render.central_latitude,
        "proj_kwargs": render.proj_kwargs,
    }


def _with_max_height(
    render: Optional[BatchRenderOptions], max_height: Optional[float]
) -> Optional[BatchRenderOptions]:
    """
    This helper folds an explicit cross-section maximum height into the shared render options so it reaches every worker, returning the options unchanged when no maximum height was requested.

    Parameters:
        render (Optional[BatchRenderOptions]): Shared rendering options, or None.
        max_height (Optional[float]): Maximum height in kilometers for the vertical axis, or None.

    Returns:
        Optional[BatchRenderOptions]: The render options carrying the maximum height.
    """
    if max_height is None:
        return render
    return replace(render or BatchRenderOptions(), max_height=max_height)


def _get_time_str(
    dataset: xr.Dataset, time_idx: int
) -> Tuple[str, Optional[pd.Timestamp]]:
    """
    This helper function generates a formatted time string for a given time index from the dataset's Time coordinate. It checks if the Time coordinate exists and if the specified time index is within bounds. If valid, it converts the time value to a pandas Timestamp, formats it as 'YYYYMMDDTHH', and returns both the formatted string and the original Timestamp. If the Time coordinate is not available or the index is out of bounds, it returns a default string in the format 'tXXX' where XXX is the zero-padded time index, along with None for the Timestamp. This function provides a consistent way to generate time strings for file naming and plot titles while also optionally returning the original time value for further use in plotting or diagnostics.

    Parameters:
        dataset (xr.Dataset): The xarray Dataset containing the Time coordinate.
        time_idx (int): The index of the time step for which to generate the time string.

    Returns:
        Tuple[str, Optional[pd.Timestamp]]: A tuple containing the formatted time string and the original Timestamp object if available, or None if the Time coordinate is not valid.
    """
    if hasattr(dataset, "Time") and len(dataset.Time) > time_idx:
        time_end = pd.Timestamp(dataset.Time.values[time_idx])
        return MPASDateTimeUtils.format_time_tag(time_end), time_end
    return f"t{time_idx:03d}", None


def _split_by_lookback(
    dataset: Any, time_indices: List[int], accum_period: str
) -> Tuple[List[int], List[int]]:
    """
    This helper separates the requested time indices into those that can form the given precipitation accumulation and those that cannot, using the dataset's actual Time coordinate by the same rule as PrecipitationDiagnostics (the output exactly N hours before the valid time must exist). Unlike an index-count rule it is correct for any output interval: a 3-hourly dataset keeps index 1 for a03h, while an accumulation that does not divide the output interval drops every index. Indices are never dropped for reasons other than lookback.

    Parameters:
        dataset (Any): The loaded dataset providing the Time coordinate.
        time_indices (List[int]): The candidate time indices.
        accum_period (str): Accumulation period identifier such as 'a03h'.

    Returns:
        Tuple[List[int], List[int]]: The usable indices and the dropped indices, each in the original order.
    """
    time_dim = "Time" if "Time" in dataset.sizes else "time"
    hours = int(accum_period[1:3])
    diag = PrecipitationDiagnostics(verbose=False)
    usable: List[int] = []
    dropped: List[int] = []
    for idx in time_indices:
        try:
            diag._resolve_previous_index(dataset, time_dim, idx, hours, accum_period)
            usable.append(idx)
        except InsufficientLookbackError:
            dropped.append(idx)
    return usable, dropped


def cross_section_file_stem(
    var_name: str,
    vertical_coord: str,
    time_tag: str,
    max_height: Optional[float] = None,
    file_prefix: str = "mpas_cross_section",
    display_vertical: Optional[str] = None,
    start_point: Optional[Tuple[float, float]] = None,
    end_point: Optional[Tuple[float, float]] = None,
) -> str:
    """
    This function builds the output file name (without directory or extension) of a vertical cross-section plot. It is the single definition used by the serial command-line path and the parallel batch worker, so both name their files identically: the prefix, the variable, the vertical coordinate, an optional display coordinate when it differs from the vertical coordinate, the valid-time tag, and, when a maximum height was requested, a '_maxh_<value>km' suffix so runs with different --max-height values never overwrite each other. The maximum height is formatted with the shortest general format and a decimal point is replaced by 'p' (for example 12.5 becomes '12p5').

    Parameters:
        var_name (str): Name of the plotted variable.
        vertical_coord (str): Vertical coordinate used for the transect ('pressure', 'modlev', 'height').
        time_tag (str): Valid-time tag produced by MPASDateTimeUtils.format_time_tag.
        max_height (Optional[float]): Maximum plotted height in kilometers, or None when not limited.
        file_prefix (str): File name prefix (default: 'mpas_cross_section').
        display_vertical (Optional[str]): Displayed vertical coordinate when it differs from vertical_coord.
        start_point (Optional[Tuple[float, float]]): Transect start as (lon, lat) in degrees; when both end points are given, a '_s<lon>_<lat>_e<lon>_<lat>' suffix makes different transects name different files.
        end_point (Optional[Tuple[float, float]]): Transect end as (lon, lat) in degrees.

    Returns:
        str: The file name stem, e.g. 'mpas_cross_section_theta_vcrd_pressure_valid_20240917T03_maxh_12km'.
    """
    stem = (
        f"{sanitize_filename_component(file_prefix)}"
        f"_{sanitize_filename_component(var_name)}"
        f"_vcrd_{sanitize_filename_component(vertical_coord)}"
    )
    if display_vertical and display_vertical != vertical_coord:
        stem += f"_disp_{sanitize_filename_component(display_vertical)}"
    stem += f"_valid_{time_tag}"
    if max_height is not None:
        stem += f"_maxh_{format(max_height, 'g').replace('.', 'p')}km"
    if start_point is not None and end_point is not None:
        s_lon, s_lat = (format(v, "g").replace(".", "p") for v in start_point)
        e_lon, e_lat = (format(v, "g").replace(".", "p") for v in end_point)
        stem += f"_s{s_lon}_{s_lat}_e{e_lon}_{e_lat}"
    return stem


def wind_file_stem(
    u_variable: str,
    v_variable: str,
    plot_type: str,
    time_tag: str,
    wind_level: Optional[str] = None,
    file_prefix: str = "mpas_wind",
) -> str:
    """
    This function builds the output file name (without directory or extension) of a horizontal wind plot. It is the single definition used by the serial command-line path and the parallel batch worker, so both name their files identically: the prefix, the sanitized u and v component names, the plot type, an optional '_lev<level>' part when the winds were taken from a vertical level (so plots at different levels never overwrite each other), and the valid-time tag.

    Parameters:
        u_variable (str): Name of the zonal wind component.
        v_variable (str): Name of the meridional wind component.
        plot_type (str): Wind plot type ('barbs', 'arrows', 'streamlines').
        time_tag (str): Valid-time tag produced by MPASDateTimeUtils.format_time_tag.
        wind_level (Optional[str]): Label of the selected vertical level (for example '850hPa'), or None for surface winds, in which case the level part is omitted.
        file_prefix (str): File name prefix (default: 'mpas_wind').

    Returns:
        str: The file name stem, e.g. 'mpas_wind_u10_v10_barbs_valid_20240917T03' or 'mpas_wind_uzonal_umeridional_barbs_lev850hPa_valid_20240917T03'.
    """
    stem = (
        f"{sanitize_filename_component(file_prefix)}"
        f"_{sanitize_filename_component(u_variable)}"
        f"_{sanitize_filename_component(v_variable)}"
        f"_{sanitize_filename_component(plot_type)}"
    )
    if wind_level is not None:
        level_tag = "".join(str(wind_level).split())
        stem += f"_lev{sanitize_filename_component(level_tag)}"
    return f"{stem}_valid_{time_tag}"


def default_precipitation_title(var_name: str, time_str: str) -> str:
    """
    This function builds the default title of a precipitation map from the variable name and the valid-time tag. It is the single definition shared by the serial command-line path and the parallel batch worker, so a precipitation map carries exactly the same title whether it was produced serially or in parallel when the user supplied no title of their own.

    Parameters:
        var_name (str): Name of the precipitation variable (e.g. 'rainnc').
        time_str (str): Valid-time tag of the plot (e.g. '20240917T03').

    Returns:
        str: The default title, 'MPAS Precipitation | Variable: <var> | Valid: <time>'.
    """
    return f"MPAS Precipitation | Variable: {var_name} | Valid: {time_str}"


def _precipitation_worker(
    args: Tuple[int, Dict[str, Any]],
    processor_seed: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """
    This worker function executes precipitation diagnostic processing and plotting for a single timestep with comprehensive timing metrics. It serves as a picklable entry point for parallel processing of precipitation diagnostics across multiple timesteps using a shared data cache. The function extracts spatial coordinates, computes precipitation differences, generates visualizations with the MPASPrecipitationPlotter, and saves outputs in specified formats. It measures execution time for data extraction, plotting operations, and file writing while tracking cache hit statistics for performance analysis.

    Parameters:
        args (Tuple[int, Dict[str, Any]]): Two-element tuple with (time_idx, kwargs) where time_idx is the timestep index and kwargs contains processor instance, cache object, spatial bounds, variable name, accumulation period, plot type, and formatting options.
        processor_seed (Optional[Tuple[str, str]]): The cache key and seed file of the parent's pickled processor (see _make_processor_seed), or None to load from the kwargs.

    Returns:
        Dict[str, Any]: Result dictionary with 'files' (list of str paths), 'timings' (dict with phase durations), 'time_str' (str), 'cache_hits' (dict), and optionally 'cache_info' (dict with cache statistics).
    """
    try:
        time_idx, kwargs = args
        _install_processor_seed(processor_seed)

        processor, cache = _setup_processor_and_cache(kwargs)

        output_dir = kwargs["output_dir"]
        lon_min = kwargs["lon_min"]
        lon_max = kwargs["lon_max"]
        lat_min = kwargs["lat_min"]
        lat_max = kwargs["lat_max"]
        var_name = kwargs["var_name"]
        accum_period = kwargs["accum_period"]
        plot_type = kwargs.get("plot_type", "scatter")
        grid_resolution = kwargs.get("grid_resolution", None)
        file_prefix = kwargs["file_prefix"]
        formats = kwargs["formats"]
        custom_title_template = kwargs.get("custom_title_template")
        colormap = kwargs.get("colormap")
        levels = kwargs.get("levels")
        display_units = kwargs.get("display_units", "mm")
        threshold = kwargs.get("threshold")
        remap_engine = kwargs.get("remap_engine", "kdtree")
        remap_method = kwargs.get("remap_method", "nearest")
        remap_config = SimpleNamespace(
            remap_engine=remap_engine, remap_method=remap_method
        )
        render = kwargs.get("render")

        start_time = time.time()
        timings = {}
        cache_hits = {"coordinates": False, "data": False}

        data_start = time.time()

        lon, lat = _extract_precip_coordinates(processor, cache, var_name, cache_hits)

        precip_diag = PrecipitationDiagnostics(verbose=False)
        precip_data = precip_diag.compute_precipitation_difference(
            processor.dataset,
            time_idx,
            var_name,
            accum_period,
            data_type=processor.data_type or "UXarray",
        )

        time_str, time_end = _get_time_str(processor.dataset, time_idx)

        timings["data_processing"] = time.time() - data_start

        plotter = MPASPrecipitationPlotter(**_plotter_geometry(render, (10, 14)))

        _apply_weights_dir(plotter, kwargs)

        if custom_title_template:
            title = custom_title_template.format(
                var_name=var_name.upper(), time_str=time_str, accum_period=accum_period
            )
        else:
            title = default_precipitation_title(var_name, time_str)

        plot_start = time.time()

        _, _ = plotter.create_precipitation_map(
            lon,
            lat,
            precip_data.values,
            GeographicBounds(lon_min, lon_max, lat_min, lat_max),
            accum_period=accum_period,
            time_end=time_end,
            data_array=precip_data,
            var_name=var_name,
            dataset=processor.dataset,
            config=remap_config,
            style=PrecipitationRenderStyle(
                title=title,
                plot_type=plot_type,
                colormap=colormap,
                levels=levels,
                grid_resolution=grid_resolution,
                clim_min=render.clim_min if render is not None else None,
                clim_max=render.clim_max if render is not None else None,
                display_units=display_units,
                threshold=threshold,
                **_projection_style_kwargs(render),
            ),
        )

        timings["plotting"] = time.time() - plot_start
        save_start = time.time()

        output_path = _batch_output_path(
            output_dir,
            kwargs.get("output_name"),
            f"{sanitize_filename_component(file_prefix)}"
            f"_vartype_{sanitize_filename_component(var_name)}"
            f"_acctype_{sanitize_filename_component(accum_period)}"
            f"_valid_{time_str}_ptype_{sanitize_filename_component(plot_type)}",
            time_str,
        )

        plotter.save_plot(output_path, formats=formats)
        plotter.close_plot()

        output_files = [f"{output_path}.{fmt}" for fmt in formats]

        timings["saving"] = time.time() - save_start
        timings["total"] = time.time() - start_time

        result = {
            "files": output_files,
            "timings": timings,
            "time_str": time_str,
            "cache_hits": cache_hits,
        }

        if cache is not None:
            result["cache_info"] = cache.get_cache_info()

        del precip_data, lon, lat, plotter
        _maybe_collect_garbage()

        return result

    except InsufficientLookbackError as e:
        skipped_idx = args[0] if args else "unknown"
        logger.info("Skipping time index %s: %s", skipped_idx, e)
        return {
            "skipped": str(e),
            "files": [],
            "timings": {},
            "time_str": "skipped",
            "cache_hits": {},
        }

    except Exception as e:
        import traceback

        time_idx = args[0] if args else "unknown"
        error_msg = f"Error processing time index {time_idx}: {str(e)}"
        error_trace = traceback.format_exc()
        logger.error("=== WORKER ERROR ===")
        logger.error("%s", error_msg)
        logger.error("%s", error_trace)
        return {
            "error": error_msg,
            "traceback": error_trace,
            "time_idx": time_idx,
            "files": [],
            "timings": {},
            "time_str": f"t{time_idx:03d}" if isinstance(time_idx, int) else "unknown",
            "cache_hits": {},
        }


def _surface_worker(
    args: Tuple[int, Dict[str, Any]],
    processor_seed: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """
    This worker function executes surface variable processing and plotting for a single timestep with comprehensive timing metrics. It serves as a picklable entry point for parallel processing of surface diagnostics across multiple timesteps using a shared data cache. The function extracts spatial coordinates, retrieves variable data, generates visualizations with the MPASSurfacePlotter, and saves outputs in specified formats. It measures execution time for data extraction, plotting operations, and file writing while tracking cache hit statistics for performance analysis.

    Parameters:
        args (Tuple[int, Dict[str, Any]]): Two-element tuple containing (time_idx, kwargs) where time_idx is the integer timestep index and kwargs is a dictionary with processor instance, cache object, spatial bounds, variable name, plot type, and formatting options.
        processor_seed (Optional[Tuple[str, str]]): The cache key and seed file of the parent's pickled processor (see _make_processor_seed), or None to load from the kwargs.

    Returns:
        Dict[str, Any]: Result dictionary with keys 'files' (list of str paths), 'timings' (dict with phase durations), 'time_str' (str), 'cache_hits' (dict), and optionally 'cache_info' (dict with cache statistics).
    """
    time_idx, kwargs = args
    _install_processor_seed(processor_seed)

    if "grid_file" in kwargs and "data_dir" in kwargs:
        processor = _get_or_create_2d_processor(kwargs)
        cache = None
    else:
        processor = kwargs["processor"]
        cache = kwargs.get("cache", None)

    output_dir = kwargs["output_dir"]
    lon_min = kwargs["lon_min"]
    lon_max = kwargs["lon_max"]
    lat_min = kwargs["lat_min"]
    lat_max = kwargs["lat_max"]
    var_name = kwargs["var_name"]
    plot_type = kwargs["plot_type"]
    file_prefix = kwargs["file_prefix"]
    formats = kwargs["formats"]
    custom_title = kwargs.get("custom_title")
    colormap = kwargs.get("colormap")
    levels = kwargs.get("levels")
    remap_engine = kwargs.get("remap_engine", "kdtree")
    remap_method = kwargs.get("remap_method", "nearest")
    remap_config = SimpleNamespace(remap_engine=remap_engine, remap_method=remap_method)
    render = kwargs.get("render")
    grid_resolution = kwargs.get("grid_resolution")

    start_time = time.time()
    timings = {}
    cache_hits = {"coordinates": False, "data": False}

    data_start = time.time()

    # Same accessors as the serial command-line path: get_2d_variable_data masks isobaric fields below
    # ground, and the coordinates follow the variable's location (cells or vertices)
    var_data = processor.get_2d_variable_data(var_name, time_idx)

    if cache is not None:
        try:
            lon, lat = cache.get_coordinates(var_name)
            cache_hits["coordinates"] = True
        except KeyError:
            lon, lat = processor.extract_2d_coordinates_for_variable(var_name, var_data)
            try:
                cache.load_coordinates_from_dataset(processor.dataset, var_name)
            except Exception:
                pass
    else:
        lon, lat = processor.extract_2d_coordinates_for_variable(var_name, var_data)

    plotter = MPASSurfacePlotter(**_plotter_geometry(render, (10, 14)))
    _apply_weights_dir(plotter, kwargs)
    timings["data_processing"] = time.time() - data_start

    plot_start = time.time()

    # Mirror the serial command-line path: the same default title carrying the valid time, no separate "Valid" box, and the variable DataArray so the colorbar label comes from its metadata
    title_time_str = MPASDateTimeUtils.get_time_info(
        processor.dataset, time_idx, verbose=False
    )

    _, _ = plotter.create_surface_map(
        lon=lon,
        lat=lat,
        data=var_data.values,
        var_name=var_name,
        bounds=GeographicBounds(lon_min, lon_max, lat_min, lat_max),
        style=SurfaceMapStyle(
            title=custom_title
            or f"MPAS Surface | Variable: {var_name} | Valid: {title_time_str}",
            plot_type=plot_type,
            colormap=colormap,
            levels=levels,
            num_levels=kwargs.get("num_levels"),
            clim_min=kwargs.get("clim_min"),
            clim_max=kwargs.get("clim_max"),
            grid_resolution=grid_resolution,
            **(
                {"interpolation": kwargs["interpolation"]}
                if kwargs.get("interpolation") is not None
                else {}
            ),
            **_projection_style_kwargs(render),
        ),
        data_array=var_data,
        dataset=processor.dataset,
        config=remap_config,
    )
    timings["plotting"] = time.time() - plot_start

    save_start = time.time()

    time_str = MPASDateTimeUtils.format_time_tag(
        pd.Timestamp(processor.dataset["Time"].values[time_idx])
    )
    safe_time_str = time_str

    output_path = _batch_output_path(
        output_dir,
        kwargs.get("output_name"),
        f"{sanitize_filename_component(file_prefix)}"
        f"_{sanitize_filename_component(var_name)}"
        f"_{sanitize_filename_component(plot_type)}_valid_{safe_time_str}",
        safe_time_str,
    )

    plotter.save_plot(output_path, formats=formats)
    plotter.close_plot()

    output_files = [f"{output_path}.{fmt}" for fmt in formats]

    timings["saving"] = time.time() - save_start
    timings["total"] = time.time() - start_time

    result = {
        "files": output_files,
        "timings": timings,
        "time_str": time_str,
        "cache_hits": cache_hits,
    }

    if cache is not None:
        result["cache_info"] = cache.get_cache_info()

    del var_data, lon, lat, plotter
    _maybe_collect_garbage()

    return result


def _wind_worker(
    args: Tuple[int, Dict[str, Any]],
    processor_seed: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """
    This worker function executes wind vector processing and plotting for a single timestep with comprehensive timing metrics. It serves as a picklable entry point for parallel processing of wind diagnostics across multiple timesteps using a shared data cache. The function extracts spatial coordinates, retrieves 2D variable data for the specified U and V wind components, generates visualizations with the MPASWindPlotter, and saves outputs in specified formats. It measures execution time for data extraction, plotting operations, and file writing while tracking cache hit statistics for performance analysis.

    Parameters:
        args (Tuple[int, Dict[str, Any]]): Two-element tuple containing (time_idx, kwargs) where time_idx is the integer timestep index and kwargs is a dictionary with processor instance, cache object, spatial bounds, variable names for U and V components, plot type, subsampling factor, scaling options, and formatting settings.
        processor_seed (Optional[Tuple[str, str]]): The cache key and seed file of the parent's pickled processor (see _make_processor_seed), or None to load from the kwargs.

    Returns:
        Dict[str, Any]: Result dictionary with keys 'files' (list of str paths), 'timings' (dict with phase durations), 'time_str' (str), 'cache_hits' (dict), and optionally 'cache_info' (dict with cache statistics).
    """
    time_idx, kwargs = args
    _install_processor_seed(processor_seed)

    render = kwargs.get("render")
    level_selection = render.wind_level_selection if render is not None else None

    if "grid_file" in kwargs and "data_dir" in kwargs:
        processor = (
            _get_or_create_3d_processor(kwargs)
            if level_selection is not None
            else _get_or_create_2d_processor(kwargs)
        )
        cache = None
    else:
        processor = kwargs["processor"]
        cache = kwargs.get("cache", None)

    output_dir = kwargs["output_dir"]
    lon_min = kwargs["lon_min"]
    lon_max = kwargs["lon_max"]
    lat_min = kwargs["lat_min"]
    lat_max = kwargs["lat_max"]
    u_variable = kwargs["u_variable"]
    v_variable = kwargs["v_variable"]
    plot_type = kwargs.get("plot_type", "barbs")
    subsample = kwargs.get("subsample", 1)
    scale = kwargs.get("scale", None)
    show_background = kwargs.get("show_background", False)
    grid_resolution = kwargs.get("grid_resolution", None)
    regrid_method = kwargs.get("regrid_method", "linear")
    remap_engine = kwargs.get("remap_engine", "kdtree")
    remap_method = kwargs.get("remap_method", "nearest")
    remap_config = SimpleNamespace(remap_engine=remap_engine, remap_method=remap_method)
    formats = kwargs["formats"]

    start_time = time.time()
    timings = {}
    cache_hits = {"coordinates": False, "data": False}

    data_start = time.time()

    if level_selection is not None:
        # 3D wind components: take the requested level (model level, pressure in Pa, or 'surface')
        u_data = processor.get_3d_variable_data(u_variable, level_selection, time_idx)
        v_data = processor.get_3d_variable_data(v_variable, level_selection, time_idx)
    else:
        u_data = processor.get_2d_variable_data(u_variable, time_idx)
        v_data = processor.get_2d_variable_data(v_variable, time_idx)

    if cache is not None:
        try:
            lon, lat = cache.get_coordinates(u_variable)
            cache_hits["coordinates"] = True
        except KeyError:
            lon, lat = processor.extract_2d_coordinates_for_variable(u_variable, u_data)
            try:
                cache.load_coordinates_from_dataset(processor.dataset, u_variable)
            except Exception:
                pass
    else:
        lon, lat = processor.extract_2d_coordinates_for_variable(u_variable, u_data)

    time_end = None

    if hasattr(processor.dataset, "Time") and len(processor.dataset.Time) > time_idx:
        time_end = pd.Timestamp(processor.dataset.Time.values[time_idx]).to_pydatetime()
        time_str = MPASDateTimeUtils.format_time_tag(time_end)
    else:
        time_str = f"t{time_idx:03d}"

    timings["data_processing"] = time.time() - data_start

    plotter = MPASWindPlotter(**_plotter_geometry(render, (12, 10)))
    _apply_weights_dir(plotter, kwargs)
    plot_start = time.time()

    _, _ = plotter.create_wind_plot(
        lon,
        lat,
        u_data.values,
        v_data.values,
        GeographicBounds(lon_min, lon_max, lat_min, lat_max),
        style=WindPlotStyle(
            wind_level=render.wind_level if render is not None else "surface",
            plot_type=plot_type,
            subsample=subsample,
            scale=scale,
            show_background=show_background,
            bg_colormap=(
                render.background_colormap if render is not None else "viridis"
            ),
            title=render.title if render is not None else None,
            time_stamp=time_end,
            vector_color=render.vector_color if render is not None else "black",
            vector_alpha=render.vector_alpha if render is not None else 1.0,
            **_projection_style_kwargs(render),
        ),
        grid_resolution=grid_resolution,
        regrid_method=regrid_method,
        dataset=processor.dataset,
        config=remap_config,
    )

    timings["plotting"] = time.time() - plot_start

    save_start = time.time()
    level_label = (
        render.wind_level
        if render is not None and render.wind_level_selection is not None
        else None
    )
    output_path = _batch_output_path(
        output_dir,
        kwargs.get("output_name"),
        wind_file_stem(
            u_variable,
            v_variable,
            plot_type,
            time_str,
            wind_level=level_label,
            file_prefix=kwargs.get("file_prefix", "mpas_wind"),
        ),
        time_str,
    )

    plotter.save_plot(output_path, formats=formats)
    plotter.close_plot()

    output_files = [f"{output_path}.{fmt}" for fmt in formats]

    timings["saving"] = time.time() - save_start
    timings["total"] = time.time() - start_time

    result = {
        "files": output_files,
        "timings": timings,
        "time_str": time_str,
        "cache_hits": cache_hits,
    }

    if cache is not None:
        result["cache_info"] = cache.get_cache_info()

    del u_data, v_data, lon, lat, plotter
    _maybe_collect_garbage()

    return result


def _cross_section_clim_kwargs(
    kwargs: Dict[str, Any], render: Optional[BatchRenderOptions]
) -> Dict[str, Any]:
    """
    This helper function resolves the colour-limit keyword arguments for a cross-section style. A limit supplied in the worker kwargs wins; otherwise the shared render options supply it, and a limit that is still unset is left out so the style keeps its own default.

    Parameters:
        kwargs (Dict[str, Any]): Worker keyword arguments, optionally holding 'clim_min' and 'clim_max'.
        render (Optional[BatchRenderOptions]): Shared rendering options, or None.

    Returns:
        Dict[str, Any]: Dictionary with 'clim_min' and/or 'clim_max' for the limits that are set.
    """
    clim_min = kwargs.get("clim_min")
    clim_max = kwargs.get("clim_max")
    clim_kwargs: Dict[str, Any] = {}
    if clim_min is None and render is not None:
        clim_min = render.clim_min
    if clim_max is None and render is not None:
        clim_max = render.clim_max
    if clim_min is not None:
        clim_kwargs["clim_min"] = clim_min
    if clim_max is not None:
        clim_kwargs["clim_max"] = clim_max
    return clim_kwargs


def _save_cross_section_formats(
    fig: Any,
    save_path: str,
    formats: List[str],
    render: Optional[BatchRenderOptions],
) -> List[str]:
    """
    This helper function writes the additional output formats of a cross-section figure next to the already-saved PNG and returns every output path, the PNG first.

    Parameters:
        fig (Any): The rendered matplotlib figure.
        save_path (str): Path of the PNG file that has already been written.
        formats (List[str]): Requested output formats; 'png' is skipped because it is already saved.
        render (Optional[BatchRenderOptions]): Shared rendering options supplying the dpi, or None for 100.

    Returns:
        List[str]: Paths of all written files, starting with save_path.
    """
    output_files = [save_path]

    for fmt in formats:
        if fmt != "png":
            output_file = save_path.replace(".png", f".{fmt}")
            save_kwargs: Dict[str, Any] = {
                "dpi": render.dpi if render is not None and render.dpi else 100,
                "bbox_inches": "tight",
            }

            if fmt.lower() == "png":
                save_kwargs["pil_kwargs"] = {"compress_level": 1}

            fig.savefig(output_file, **save_kwargs)
            output_files.append(output_file)

    return output_files


def _cross_section_worker(
    args: Tuple[int, Dict[str, Any]],
    processor_seed: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """
    This worker function executes vertical cross-section processing and plotting for a single timestep with comprehensive timing metrics. It serves as a picklable entry point for parallel processing of vertical cross-section diagnostics across multiple timesteps using a shared data cache. The function extracts spatial coordinates, retrieves 3D variable data, generates visualizations with the MPASVerticalCrossSectionPlotter, and saves outputs in specified formats. It measures execution time for data extraction, plotting operations, and file writing while tracking cache hit statistics for performance analysis.

    Parameters:
        args (Tuple[int, Dict[str, Any]]): Two-element tuple containing (time_idx, kwargs) where time_idx is the integer timestep index and kwargs is a dictionary with processor instance, cache object, spatial bounds for start and end points, variable name, vertical coordinate type, number of points along the cross-section, plot type, and formatting options.
        processor_seed (Optional[Tuple[str, str]]): The cache key and seed file of the parent's pickled processor (see _make_processor_seed), or None to load from the kwargs.

    Returns:
        Dict[str, Any]: Result dictionary with keys 'files' (list of str paths), 'timings' (dict with phase durations), 'time_str' (str), 'cache_hits' (dict), and optionally 'cache_info' (dict with cache statistics).
    """
    time_idx, kwargs = args
    _install_processor_seed(processor_seed)

    if "grid_file" in kwargs and "data_dir" in kwargs:
        processor_3d = _get_or_create_3d_processor(kwargs)
    else:
        processor_3d = kwargs["processor"]

    output_dir = kwargs["output_dir"]
    start_lat = kwargs["start_lat"]
    start_lon = kwargs["start_lon"]
    end_lat = kwargs["end_lat"]
    end_lon = kwargs["end_lon"]
    var_name = kwargs["var_name"]
    file_prefix = kwargs["file_prefix"]
    formats = kwargs["formats"]
    colormap = kwargs.get("colormap")
    levels = kwargs.get("levels")
    vertical_coord = kwargs.get("vertical_coord", "pressure")
    num_points = kwargs.get("num_points", 100)
    precomputed_levels = kwargs.get("vertical_levels")
    render = kwargs.get("render")
    extend = kwargs.get("extend", "both")
    cross_plot_type = kwargs.get("plot_type", "contourf")

    start_time = time.time()
    timings = {}

    data_start = time.time()

    plotter = MPASVerticalCrossSectionPlotter(**_plotter_geometry(render, (10, 14)))
    timings["data_processing"] = time.time() - data_start

    plot_start = time.time()

    time_str = str(processor_3d.dataset["Time"].values[time_idx])
    safe_time_str = MPASDateTimeUtils.format_time_tag(
        pd.Timestamp(processor_3d.dataset["Time"].values[time_idx])
    )
    max_height = render.max_height if render is not None else None
    save_path = (
        _batch_output_path(
            output_dir,
            kwargs.get("output_name"),
            cross_section_file_stem(
                var_name,
                vertical_coord,
                safe_time_str,
                max_height,
                file_prefix,
                start_point=(start_lon, start_lat),
                end_point=(end_lon, end_lat),
            ),
            safe_time_str,
        )
        + ".png"
    )
    clim_kwargs = _cross_section_clim_kwargs(kwargs, render)

    fig, _ = plotter.create_vertical_cross_section(
        mpas_3d_processor=processor_3d,
        var_name=var_name,
        start_point=(start_lon, start_lat),
        end_point=(end_lon, end_lat),
        time_index=time_idx,
        vertical_coord=vertical_coord,
        num_points=num_points,
        save_path=save_path,
        # Same default title as the serial command-line path, which carries the valid-time tag
        title=(render.title if render is not None else None)
        or f"MPAS Vertical Cross-Section | Variable: {var_name} | Valid: {safe_time_str}",
        max_height=render.max_height if render is not None else None,
        style=CrossSectionStyle(
            levels=levels,
            colormap=colormap,
            extend=extend,
            plot_type=cross_plot_type,
            **clim_kwargs,
        ),
        precomputed_levels=precomputed_levels,
    )

    timings["plotting"] = time.time() - plot_start
    phase_timings = getattr(plotter, "_phase_timings", None)

    if isinstance(phase_timings, dict):
        for phase_name, phase_seconds in phase_timings.items():
            timings[f"xs_{phase_name}"] = phase_seconds

    save_start = time.time()
    output_files = _save_cross_section_formats(fig, save_path, formats, render)

    plotter.close_plot()
    del plotter
    _maybe_collect_garbage()

    timings["saving"] = time.time() - save_start
    timings["total"] = time.time() - start_time

    return {"files": output_files, "timings": timings, "time_str": time_str}


def _skewt_worker(
    args: Tuple[int, Dict[str, Any]],
    processor_seed: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """
    This worker function executes Skew-T sounding diagnostics and plotting for a single timestep, serving as a picklable entry point for parallel generation of Skew-T diagrams across many timesteps. It obtains a 3D processor (reusing the rank's already-loaded processor in MPI mode, or reloading from grid_file/data_dir in multiprocessing mode), extracts the vertical sounding profile at the configured station location, computes thermodynamic and severe-weather indices, renders the Skew-T diagram, and writes the requested output formats. It records per-phase timing for data processing, plotting, and saving so the aggregate report matches the other parallel plot types.

    Parameters:
        args (Tuple[int, Dict[str, Any]]): Two-element tuple of (time_idx, kwargs) where time_idx is the timestep index and kwargs supplies either a loaded 'processor' or 'grid_file'/'data_dir', the station 'lon'/'lat', 'output_dir', 'file_prefix', 'formats', and the 'show_parcel' flag.
        processor_seed (Optional[Tuple[str, str]]): The cache key and seed file of the parent's pickled processor (see _make_processor_seed), or None to load from the kwargs.

    Returns:
        Dict[str, Any]: Result dictionary with 'files' (list of str paths), 'timings' (dict with phase durations), 'time_str' (str), and 'cache_hits' (dict).
    """
    time_idx, kwargs = args
    _install_processor_seed(processor_seed)

    if "grid_file" in kwargs and "data_dir" in kwargs:
        processor_3d = _get_or_create_3d_processor(kwargs)
    else:
        processor_3d = kwargs["processor"]

    output_dir = kwargs["output_dir"]
    lon = kwargs["lon"]
    lat = kwargs["lat"]
    file_prefix = kwargs.get("file_prefix", "mpas_skewt")
    formats = kwargs.get("formats", ["png"])
    show_parcel = kwargs.get("show_parcel", False)

    start_time = time.time()
    timings = {}

    data_start = time.time()

    diag = SoundingDiagnostics(verbose=False)

    cell_index = kwargs.get("cell_index")
    station_coords = (
        (kwargs["station_lon"], kwargs["station_lat"])
        if "station_lon" in kwargs and "station_lat" in kwargs
        else None
    )

    profile = diag.extract_sounding_profile(
        processor_3d,
        lon,
        lat,
        time_index=time_idx,
        cell_index=cell_index,
        station_coords=station_coords,
        max_distance_km=kwargs.get("max_distance_km"),
    )

    indices = diag.compute_thermodynamic_indices(
        profile["pressure"],
        profile["temperature"],
        profile["dewpoint"],
        u_wind_kt=profile.get("u_wind"),
        v_wind_kt=profile.get("v_wind"),
        height_m=profile.get("height"),
        surface_height_m=profile.get("surface_height"),
    )

    time_str, _ = _get_time_str(processor_3d.dataset, time_idx)

    timings["data_processing"] = time.time() - data_start

    stn_lon = profile["station_lon"]
    stn_lat = profile["station_lat"]
    lon_tag = f"{abs(stn_lon):.2f}{'W' if stn_lon < 0 else 'E'}"
    lat_tag = f"{abs(stn_lat):.2f}{'S' if stn_lat < 0 else 'N'}"

    output_path = _batch_output_path(
        output_dir,
        kwargs.get("output_name"),
        f"{sanitize_filename_component(file_prefix)}"
        f"_{lon_tag.replace('.', 'p')}_{lat_tag.replace('.', 'p')}_valid_{time_str}",
        time_str,
    )

    plotter = MPASSkewTPlotter(figsize=(9, 12), verbose=False)

    plot_start = time.time()
    plotter.create_skewt_diagram(
        pressure=profile["pressure"],
        temperature=profile["temperature"],
        dewpoint=profile["dewpoint"],
        u_wind=profile.get("u_wind"),
        v_wind=profile.get("v_wind"),
        title=f"MPAS Skew-T | {lon_tag}, {lat_tag} | Valid Time: {time_str}",
        indices=indices,
        show_parcel=show_parcel,
        save_path=None,
        location=profile,
    )
    timings["plotting"] = time.time() - plot_start

    save_start = time.time()
    plotter.save_plot(output_path, formats=formats)
    plotter.close_plot()
    output_files = [f"{output_path}.{fmt}" for fmt in formats]
    timings["saving"] = time.time() - save_start

    timings["total"] = time.time() - start_time

    del plotter
    _maybe_collect_garbage()

    return {
        "files": output_files,
        "timings": timings,
        "time_str": time_str,
        "cache_hits": {},
    }


def _process_parallel_results(
    results: List[Any],
    time_indices: List[int],
    output_dir: str,
    manager: "MPASParallelManager",
    processing_type: str,
    var_info: Optional[str] = None,
) -> List[str]:
    """
    This helper function processes the results returned by parallel worker functions, aggregates timing metrics, counts successes and failures, and generates a comprehensive report summarizing the outcomes of the parallel processing operation. It collects successfully generated file paths, computes timing statistics for data processing, plotting, and saving phases, and retrieves overall parallel execution statistics from the manager to provide insights into performance and efficiency. The function prints a detailed report to the console with status counts, timing breakdowns, and potential speedup information based on the collected results.

    Parameters:
        results (List[Any]): List of results returned by parallel worker functions, where each result is expected to be a dictionary containing 'files', 'timings', 'time_str', and optionally 'cache_hits' and 'cache_info'.
        time_indices (List[int]): List of time indices that were processed in parallel, used for reporting purposes.
        output_dir (str): Directory path where output files were saved, used for reporting the location of generated files.
        manager ('MPASParallelManager'): Instance of the parallel manager used to execute the tasks, from which overall execution statistics can be retrieved.
        processing_type (str): String indicating the type of processing performed (e.g., "Precipitation Maps", "Surface Maps", "Wind Plots", "Cross-Sections") for report header.
        var_info (Optional[str]): Optional string with variable information to include in the report header for additional context.

    Returns:
        List[str]: List of successfully created file paths aggregated from the results.
    """
    created_files, successful, failed, timing_stats = _aggregate_parallel_results(
        results, time_indices
    )
    skipped = sum(
        1
        for result in results
        if result.success
        and isinstance(result.result, dict)
        and "skipped" in result.result
    )

    _print_parallel_results_report(
        processing_type,
        var_info,
        successful,
        failed,
        time_indices,
        created_files,
        output_dir,
        timing_stats,
        manager,
    )

    batch_files = BatchFiles(created_files)
    batch_files.successful = successful
    batch_files.failed = failed
    batch_files.skipped = skipped
    return batch_files


def _aggregate_parallel_results(
    results: List[Any], time_indices: List[int]
) -> Tuple[List[str], int, int, Dict[str, Dict[str, float]]]:
    """
    This helper function aggregates the results from parallel worker functions, counting successful and failed tasks, collecting created file paths, and computing timing statistics for each processing phase. It iterates through the list of results, checking for success flags and error messages to determine the status of each task. For successful tasks, it accumulates the file paths and timing metrics; for failed tasks, it logs error messages with the corresponding time index. Finally, it computes summary statistics (min, max, mean, total) for each timing phase across all successful tasks to provide insights into performance.

    Parameters:
        results (List[Any]): Results from the parallel workers; each carries a 'success' flag and a 'result' payload with 'files', 'timings', and optional 'error'.
        time_indices (List[int]): Time indices processed in parallel, used only to label error messages for failed tasks.

    Returns:
        Tuple[List[str], int, int, Dict[str, Dict[str, float]]]: The created file paths, the successful count, the failed count, and the timing statistics.
    """
    created_files: List[str] = []
    successful = 0
    failed = 0

    all_timings: Dict[str, List[float]] = {
        "data_processing": [],
        "plotting": [],
        "saving": [],
        "total": [],
    }

    for result in results:
        if not result.success:
            failed += 1
            logger.error(
                _FAILED_TIME_INDEX_MSG, time_indices[result.task_id], result.error
            )
            continue
        if "error" in result.result:
            failed += 1
            logger.error(
                _FAILED_TIME_INDEX_MSG,
                time_indices[result.task_id],
                result.result["error"],
            )
            continue
        if "skipped" in result.result:
            continue

        created_files.extend(result.result["files"])
        successful += 1
        timings = result.result["timings"]

        for key, value in timings.items():
            all_timings.setdefault(key, []).append(value)

    return created_files, successful, failed, _compute_timing_stats(all_timings)


def _compute_timing_stats(
    all_timings: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """
    This helper function computes summary statistics (min, max, mean, total) for each timing phase based on the collected timing values from successful parallel tasks. It iterates through the timing lists for each phase, calculating the minimum, maximum, average, and total time spent in that phase across all successful tasks. The resulting statistics are organized in a dictionary keyed by phase name, which can be used for reporting and performance analysis.

    Parameters:
        all_timings (Dict[str, List[float]]): Mapping of phase name to its list of recorded durations in seconds.

    Returns:
        Dict[str, Dict[str, float]]: Mapping of phase name to its 'min', 'max', 'mean', and 'total' statistics, for phases that recorded at least one value.
    """
    timing_stats: Dict[str, Dict[str, float]] = {}

    for key, values in all_timings.items():
        if values:
            timing_stats[key] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "total": sum(values),
            }

    return timing_stats


def _print_timing_section(label: str, stat: Dict[str, float]) -> None:
    """
    This helper function prints a formatted timing section to the console for a given phase of processing. It takes a label for the section (e.g., "Data Processing", "Plotting") and a dictionary of timing statistics containing 'min', 'max', and 'mean' values. The function formats these values to three decimal places and prints them in a structured way for easy reading in the overall report.

    Parameters:
        label (str): Section label (e.g. 'Data Processing', 'Plotting').
        stat (Dict[str, float]): Statistics dict with 'min', 'max', and 'mean' keys.

    Returns:
        None
    """
    print(f"  {label}:")
    print(f"    Min:  {stat['min']:6.3f}s")
    print(f"    Max:  {stat['max']:6.3f}s")
    print(f"    Mean: {stat['mean']:6.3f}s")


def _print_parallel_results_report(
    processing_type: str,
    var_info: Optional[str],
    successful: int,
    failed: int,
    time_indices: List[int],
    created_files: List[str],
    output_dir: str,
    timing_stats: Dict[str, Dict[str, float]],
    manager: "MPASParallelManager",
) -> None:
    """
    This helper function prints a comprehensive report to the console summarizing the results of parallel processing operations. It includes the processing type, variable information, counts of successful and failed tasks, the number of created files and their output directory, timing breakdowns for each phase of processing, and overall parallel execution statistics such as wall time and potential speedup. The report is formatted for clarity and provides insights into both the outcomes and performance of the parallel processing workflow.

    Parameters:
        processing_type (str): Processing type used in the report header.
        var_info (Optional[str]): Optional variable info line for the header.
        successful (int): Number of successfully processed time steps.
        failed (int): Number of failed time steps.
        time_indices (List[int]): Time indices processed, used for the totals.
        created_files (List[str]): Successfully created file paths.
        output_dir (str): Directory where output files were saved.
        timing_stats (Dict[str, Dict[str, float]]): Per-phase timing statistics.
        manager ('MPASParallelManager'): Parallel manager providing execution stats.

    Returns:
        None
    """
    print(f"=== {processing_type} BATCH PROCESSING RESULTS ===")

    if var_info:
        print(f"{var_info}")

    print("Status:")
    print(f"  Successful: {successful}/{len(time_indices)}")
    print(f"  Failed: {failed}/{len(time_indices)}")
    skipped_count = len(time_indices) - successful - failed
    if skipped_count > 0:
        print(f"  Skipped: {skipped_count}/{len(time_indices)}")
    print(f"  Created files: {len(created_files)} in {output_dir}")

    if not timing_stats:
        return

    print("Timing Breakdown (per time step):")

    standard_phases = [
        ("data_processing", "Data Processing"),
        ("plotting", "Plotting"),
        ("saving", "Saving"),
        ("total", "Total per step"),
    ]

    for key, label in standard_phases:
        if key in timing_stats:
            _print_timing_section(label, timing_stats[key])

    standard_keys = {key for key, _ in standard_phases}

    for key in sorted(k for k in timing_stats if k not in standard_keys):
        _print_timing_section(key, timing_stats[key])

    stats = manager.get_statistics()

    if not stats:
        return

    wall = stats.wall_time if stats.wall_time > 0 else stats.total_time
    speedup = stats.total_time / wall if wall > 0 else 1.0

    print("Overall Parallel Execution:")
    print(f"  Aggregate task time: {stats.total_time:.2f}s")
    print(f"  Wall time:           {wall:.2f}s")
    print(f"  Speedup:             {speedup:.2f}x")
    print(f"  Load imbalance:      {100 * stats.load_imbalance:.1f}%")


def _prebuild_remapper_mpi(
    processor: "MPAS2DProcessor",
    weights_dir: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    resolution: float,
    comm: Any,
) -> None:
    """
    This helper function pre-builds the remapper for the MPASPrecipitationPlotter in MPI mode to ensure that the remapping weights are generated and cached before any worker processes attempt to use them. It extracts the full grid coordinates from the processor's dataset, retrieves the boundary coordinates, and calls the plotter's internal method to get or build the remapper with the specified spatial bounds and resolution. This function is intended to be called on the master process before launching parallel workers to ensure that all necessary remapping weights are available in the shared directory for use by worker processes without incurring redundant computation or cache misses.

    Parameters:
        processor ('MPAS2DProcessor'): The processor instance containing the dataset and grid information needed to extract coordinates for remapper construction.
        weights_dir (str): The directory path where remapping weights should be stored and accessed by worker processes.
        lon_min (float): Minimum longitude for the remapper's target grid in degrees.
        lon_max (float): Maximum longitude for the remapper's target grid in degrees.
        lat_min (float): Minimum latitude for the remapper's target grid in degrees.
        lat_max (float): Maximum latitude for the remapper's target grid in degrees.
        resolution (float): Desired grid resolution for the remapper in degrees.
        comm (Any): The MPI communicator object used for parallel execution, needed to ensure that remapper construction is performed on the master process and that worker processes can access the generated weights.

    Returns:
        None: This function does not return a value but ensures that the remapper is pre-built and that the necessary weights are available in the specified directory for use by worker processes during parallel execution.
    """
    from pathlib import Path

    plotter: MPASPrecipitationPlotter = MPASPrecipitationPlotter.__new__(
        MPASPrecipitationPlotter
    )
    plotter._remapper = None
    plotter._remapper_key = None
    plotter._remapper_weights_dir = Path(weights_dir)

    dataset = processor.dataset
    dataset = plotter._ensure_boundary_data(dataset)
    lon_full, lat_full = plotter._extract_full_grid(dataset)

    assert dataset is not None
    lon_bounds = dataset["lon_b"].values
    lat_bounds = dataset["lat_b"].values

    plotter._get_or_build_remapper(
        lon_full,
        lat_full,
        lon_bounds,
        lat_bounds,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        resolution,
        comm=comm,
    )


class ParallelPrecipitationProcessor:
    """This class provides static methods for parallel processing of precipitation diagnostics and visualizations using MPI-based distributed processing."""

    @staticmethod
    def _build_precipitation_worker_kwargs(
        processor: "MPAS2DProcessor",
        is_mpi_mode: bool,
        output_dir: str,
        bounds: GeographicBounds,
        var_name: str,
        accum_period: str,
        plot_type: str,
        grid_resolution: Optional[float],
        formats: List[str],
        style: Optional[PrecipitationMapStyle] = None,
        remap_config: Optional[RemapConfig] = None,
        render: Optional[BatchRenderOptions] = None,
        output_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        This helper method constructs the keyword arguments dictionary passed to the precipitation worker function based on whether MPI mode is being used. It assembles the parameters shared by both execution modes, then adds the mode-specific entries: in MPI mode it supplies the grid file, data directory, and required variable list (raising an informative error if the processor lacks a 'data_dir' attribute), while in multiprocessing mode it creates a shared data cache and pre-loads coordinates into it for faster access by worker processes.

        Parameters:
            processor (MPAS2DProcessor): The processor instance containing the dataset and grid information.
            is_mpi_mode (bool): Flag indicating whether MPI mode is being used for parallel execution.
            output_dir (str): Directory path where output files will be saved.
            bounds (GeographicBounds): Map extent as (lon_min, lon_max, lat_min, lat_max) longitude/latitude boundaries in degrees.
            var_name (str): Name of the precipitation variable to process (e.g., 'rainnc').
            accum_period (str): Accumulation period string (e.g., 'a01h' for 1 hour).
            plot_type (str): Type of plot to create ('scatter', 'contourf', etc.).
            grid_resolution (Optional[float]): Desired grid resolution for plotting, if regridding is needed.
            formats (List[str]): List of output formats to save (e.g., ['png', 'pdf']).
            style (Optional[PrecipitationMapStyle]): Appearance and file-naming settings (file_prefix, custom_title_template, colormap, levels). If None, defaults are used.
            remap_config (Optional[RemapConfig]): Remapping settings (remap_engine, remap_method, weights_dir). If None, defaults are used.
            render (Optional[BatchRenderOptions]): Shared rendering options (color limits, figure size and dpi, projection controls, title, wind level, background colormap, maximum height) forwarded to every worker; None keeps each worker's defaults.
            output_name (Optional[str]): User-supplied output name forwarded to the worker for file naming, or None.

        Returns:
            Dict[str, Any]: Dictionary of keyword arguments to be passed to the precipitation worker function, containing either 'grid_file'/'data_dir'/'variables' for MPI mode, or 'processor'/'cache' for multiprocessing mode, along with all necessary parameters for plotting.
        """
        lon_min, lon_max, lat_min, lat_max = bounds

        if style is None:
            style = PrecipitationMapStyle()

        if remap_config is None:
            remap_config = RemapConfig()

        shared_kwargs = {
            "output_dir": output_dir,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_min": lat_min,
            "lat_max": lat_max,
            "var_name": var_name,
            "accum_period": accum_period,
            "plot_type": plot_type,
            "grid_resolution": grid_resolution,
            "file_prefix": style.file_prefix,
            "formats": formats,
            "custom_title_template": style.custom_title_template,
            "colormap": style.colormap,
            "levels": style.levels,
            "display_units": style.display_units,
            "threshold": style.threshold,
            "remap_engine": remap_config.remap_engine,
            "remap_method": remap_config.remap_method,
            "weights_dir": remap_config.weights_dir,
            "render": render,
            "output_name": output_name,
        }

        use_grid_reload = _processor_supports_reload(processor)

        if is_mpi_mode and not use_grid_reload:
            raise AttributeError(
                "MPI mode requires processor to have 'data_dir' attribute. "
                "Please update mpasdiag/processing/processors_2d.py to store data_dir in load_2d_data() method, "
                "or use multiprocessing mode instead (remove mpiexec, use --workers N)."
            )

        if use_grid_reload:
            return {
                **shared_kwargs,
                **_reload_paths(processor),
                "variables": PRECIP_REQUIRED_VARS.get(var_name, [var_name]),
            }

        cache = MPASDataCache(max_variables=5)
        logger.info(PRELOAD_COORDS_MSG)

        try:
            cache.load_coordinates_from_dataset(processor.dataset, var_name)
            logger.debug(_COORD_CACHE_MSG, var_name)
        except Exception as e:
            logger.warning(_COORD_PRELOAD_MSG, e)
            logger.warning(COORDS_FALLBACK_MSG)
        return {**shared_kwargs, "processor": processor, "cache": cache}

    @staticmethod
    def create_batch_precipitation_maps_parallel(
        processor: "MPAS2DProcessor",
        output_dir: str,
        bounds: GeographicBounds,
        var_name: str = "rainnc",
        accum_period: str = "a01h",
        plot_type: str = "scatter",
        grid_resolution: Optional[float] = None,
        formats: List[str] = ["png"],
        time_indices: Optional[List[int]] = None,
        n_processes: Optional[int] = None,
        load_balance_strategy: str = "cyclic",
        style: Optional[PrecipitationMapStyle] = None,
        remap_config: Optional[RemapConfig] = None,
        render: Optional[BatchRenderOptions] = None,
        output_name: Optional[str] = None,
    ) -> Optional[List[str]]:
        """
        This method creates precipitation maps in parallel across multiple time steps using either multiprocessing or MPI-based parallel execution. It manages the distribution of tasks to worker processes, collects results, and generates a comprehensive report on the outcomes of the parallel processing operation. The method handles both data processing and visualization phases while tracking timing metrics and cache performance for insights into efficiency.

        Parameters:
            processor (MPAS2DProcessor): Initialized MPAS2DProcessor instance with loaded data and grid information.
            output_dir (str): Directory path where precipitation map files will be saved.
            bounds (GeographicBounds): Map extent as (lon_min, lon_max, lat_min, lat_max) longitude/latitude boundaries in degrees.
            var_name (str): Name of the precipitation variable to process (e.g., 'rainnc').
            accum_period (str): Accumulation period string (e.g., 'a01h' for 1 hour, 'a03h' for 3 hours).
            plot_type (str): Type of plot to create ('scatter', 'contourf', etc.).
            grid_resolution (Optional[float]): Desired grid resolution for plotting, if regridding is needed.
            formats (List[str]): List of output formats to save (e.g., ['png', 'pdf']).
            time_indices (Optional[List[int]]): List of time indices to process. If None, all valid time indices will be processed based on accumulation period.
            n_processes (Optional[int]): Number of parallel processes to use. If None, it will default to the number of available CPU cores.
            load_balance_strategy (str): Strategy for load balancing tasks across workers ('dynamic', 'static', etc.).
            style (Optional[PrecipitationMapStyle]): Appearance and file-naming settings (file_prefix, custom_title_template, colormap, levels). If None, defaults are used.
            remap_config (Optional[RemapConfig]): Remapping settings (remap_engine, remap_method, weights_dir). If None, defaults are used.
            render (Optional[BatchRenderOptions]): Shared rendering options (color limits, figure size and dpi, projection controls, title, wind level, background colormap, maximum height) forwarded to every worker; None keeps each worker's defaults.
            output_name (Optional[str]): User-supplied output name; when set, files are named '<output_name>_<time tag>' exactly as the serial command-line path does with --output, instead of the descriptive default name.

        Returns:
            Optional[List[str]]: List of generated file paths on master process, None on worker processes.
        """
        lon_min, lon_max, lat_min, lat_max = bounds

        if style is None:
            style = PrecipitationMapStyle()

        if remap_config is None:
            remap_config = RemapConfig()

        accum_hours = int(accum_period[1:3])

        assert processor.dataset is not None
        time_dim = "Time" if "Time" in processor.dataset.sizes else "time"
        total_times = processor.dataset.sizes[time_dim]

        candidates = (
            list(range(total_times)) if time_indices is None else list(time_indices)
        )
        time_indices, dropped = _split_by_lookback(
            processor.dataset, candidates, accum_period
        )
        if dropped:
            logger.info(
                "Skipping time indices %s: accumulation %s has no output %d h earlier in the data",
                dropped,
                accum_period,
                accum_hours,
            )

        if not time_indices:
            logger.warning(
                "No valid time indices for accumulation period %s",
                accum_period,
            )
            return BatchFiles([])

        manager = _create_batch_manager(processor, load_balance_strategy, n_processes)
        is_mpi_mode = manager.backend == "mpi"

        worker_kwargs = (
            ParallelPrecipitationProcessor._build_precipitation_worker_kwargs(
                processor,
                is_mpi_mode,
                output_dir,
                bounds,
                var_name,
                accum_period,
                plot_type,
                grid_resolution,
                formats,
                style=style,
                remap_config=remap_config,
                render=render,
                output_name=output_name,
            )
        )

        if remap_config.weights_dir is not None and grid_resolution is not None:
            _prebuild_remapper_mpi(
                processor,
                remap_config.weights_dir,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                grid_resolution,
                manager.comm,
            )

        os.makedirs(output_dir, exist_ok=True)

        if manager.is_master:
            logger.info(
                "Creating precipitation maps for %d time steps in parallel",
                len(time_indices),
            )
            logger.info(
                "Using accumulation period: %s (%d hours)",
                accum_period,
                accum_hours,
            )
            logger.info(_OUTDIR_MSG, output_dir)

        tasks = [(time_idx, worker_kwargs) for time_idx in time_indices]

        results = _parallel_map_batch(
            manager, _precipitation_worker, tasks, "2d", worker_kwargs, processor
        )

        if manager.is_master and results is not None:
            created = _process_parallel_results(
                results, time_indices, output_dir, manager, "PRECIPITATION"
            )
            del results, manager
            gc.collect()
            return created

        del results, manager
        gc.collect()
        return None


class ParallelSurfaceProcessor:
    """This class provides static methods for parallel processing of surface variable maps and visualizations using MPI-based distributed processing."""

    @staticmethod
    def _build_surface_worker_kwargs(
        processor: "MPAS2DProcessor",
        is_mpi_mode: bool,
        output_dir: str,
        bounds: GeographicBounds,
        var_name: str,
        formats: List[str],
        grid_resolution: Optional[float],
        style: SurfaceBatchStyle,
        remap_config: RemapConfig,
        render: Optional[BatchRenderOptions],
        output_name: Optional[str],
    ) -> Dict[str, Any]:
        """
        This helper method constructs the keyword arguments dictionary passed to the surface worker function. Workers either reload the data from grid_file/data_dir (required in MPI mode, where an informative AttributeError is raised when the processor lacks 'data_dir') or reuse the provided processor together with a shared coordinate cache that is pre-loaded here, falling back to lazy loading with a warning if the pre-load fails.

        Parameters:
            processor (MPAS2DProcessor): The processor instance containing the dataset and grid information.
            is_mpi_mode (bool): Flag indicating whether MPI mode is being used for parallel execution.
            output_dir (str): Directory path where output files will be saved.
            bounds (GeographicBounds): Map extent as (lon_min, lon_max, lat_min, lat_max) in degrees.
            var_name (str): Name of the surface variable to process.
            formats (List[str]): List of output formats to save.
            grid_resolution (Optional[float]): Desired grid resolution for plotting, if regridding is needed.
            style (SurfaceBatchStyle): Appearance and file-naming settings.
            remap_config (RemapConfig): Remapping settings.
            render (Optional[BatchRenderOptions]): Shared rendering options forwarded to every worker, or None.
            output_name (Optional[str]): User-supplied output name forwarded to the worker for file naming, or None.

        Returns:
            Dict[str, Any]: Worker keyword arguments containing either the reload paths and 'variables', or 'processor' and 'cache'.
        """
        lon_min, lon_max, lat_min, lat_max = bounds
        file_prefix = style.file_prefix
        plot_type = style.plot_type
        remap_engine = remap_config.remap_engine
        remap_method = remap_config.remap_method

        use_grid_reload = _processor_supports_reload(processor)

        if is_mpi_mode and not use_grid_reload:
            raise AttributeError(
                "MPI mode requires processor to have 'data_dir' attribute. "
                "Please update mpasdiag/processing/processors_2d.py on your HPC system, "
                "or use multiprocessing mode instead (remove mpiexec, use --workers N)."
            )

        worker_kwargs: Dict[str, Any]
        if use_grid_reload:
            worker_kwargs = {
                **_reload_paths(processor),
                "output_dir": output_dir,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "var_name": var_name,
                "variables": [var_name],
                "plot_type": plot_type,
                "file_prefix": file_prefix,
                "formats": formats,
                "custom_title": render.title if render is not None else None,
                "colormap": style.colormap,
                "levels": style.levels,
                "num_levels": style.num_levels,
                "clim_min": render.clim_min if render is not None else style.clim_min,
                "clim_max": render.clim_max if render is not None else style.clim_max,
                "grid_resolution": grid_resolution,
                "render": render,
                "remap_engine": remap_engine,
                "remap_method": remap_method,
                "weights_dir": remap_config.weights_dir,
                "interpolation": style.interpolation,
                "output_name": output_name,
            }
        else:
            cache = MPASDataCache(max_variables=5)

            logger.info(PRELOAD_COORDS_MSG)
            try:
                cache.load_coordinates_from_dataset(processor.dataset, var_name)
                logger.debug(_COORD_CACHE_MSG, var_name)
            except Exception as e:
                logger.warning(_COORD_PRELOAD_MSG, e)
                logger.warning(COORDS_FALLBACK_MSG)

            worker_kwargs = {
                "processor": processor,
                "cache": cache,
                "output_dir": output_dir,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "var_name": var_name,
                "plot_type": plot_type,
                "file_prefix": file_prefix,
                "formats": formats,
                "custom_title": render.title if render is not None else None,
                "colormap": style.colormap,
                "levels": style.levels,
                "num_levels": style.num_levels,
                "clim_min": render.clim_min if render is not None else style.clim_min,
                "clim_max": render.clim_max if render is not None else style.clim_max,
                "grid_resolution": grid_resolution,
                "render": render,
                "remap_engine": remap_engine,
                "remap_method": remap_method,
                "weights_dir": remap_config.weights_dir,
                "interpolation": style.interpolation,
                "output_name": output_name,
            }

        return worker_kwargs

    @staticmethod
    def create_batch_surface_maps_parallel(
        processor: "MPAS2DProcessor",
        output_dir: str,
        bounds: GeographicBounds,
        var_name: str = "t2m",
        formats: List[str] = ["png"],
        grid_resolution: Optional[float] = None,
        time_indices: Optional[List[int]] = None,
        n_processes: Optional[int] = None,
        load_balance_strategy: str = "cyclic",
        style: Optional[SurfaceBatchStyle] = None,
        remap_config: Optional[RemapConfig] = None,
        render: Optional[BatchRenderOptions] = None,
        output_name: Optional[str] = None,
    ) -> Optional[List[str]]:
        """
        This method creates surface variable maps in parallel across multiple time steps using either multiprocessing or MPI-based parallel execution. It manages the distribution of tasks to worker processes, collects results, and generates a comprehensive report on the outcomes of the parallel processing operation. The method handles both data processing and visualization phases while tracking timing metrics and cache performance for insights into efficiency.

        Parameters:
            processor ('MPAS2DProcessor'): Initialized MPAS2DProcessor instance with loaded data and grid information.
            output_dir (str): Directory path where surface map files will be saved.
            bounds (GeographicBounds): Map extent as (lon_min, lon_max, lat_min, lat_max) longitude/latitude boundaries in degrees.
            var_name (str): Name of the surface variable to process (e.g., 't2m').
            formats (List[str]): List of output formats to save (e.g., ['png', 'pdf']).
            grid_resolution (Optional[float]): Desired grid resolution for plotting, if regridding is needed.
            time_indices (Optional[List[int]]): List of time indices to process. If None, all time indices in the dataset will be processed.
            n_processes (Optional[int]): Number of parallel processes to use. If None, it will default to the number of available CPU cores.
            load_balance_strategy (str): Strategy for load balancing tasks across workers ('dynamic', 'static', etc.).
            style (Optional[SurfaceBatchStyle]): Appearance and file-naming settings (file_prefix, plot_type, clim_min, clim_max). If None, defaults are used.
            remap_config (Optional[RemapConfig]): Remapping settings (remap_engine, remap_method, weights_dir). If None, defaults are used.
            render (Optional[BatchRenderOptions]): Shared rendering options (color limits, figure size and dpi, projection controls, title, wind level, background colormap, maximum height) forwarded to every worker; None keeps each worker's defaults.
            output_name (Optional[str]): User-supplied output name; when set, files are named '<output_name>_<time tag>' exactly as the serial command-line path does with --output, instead of the descriptive default name.

        Returns:
            Optional[List[str]]: List of generated file paths on master process, None on worker processes.
        """
        if style is None:
            style = SurfaceBatchStyle()

        plot_type = style.plot_type

        if remap_config is None:
            remap_config = RemapConfig()

        assert processor.dataset is not None
        time_dim = "Time" if "Time" in processor.dataset.sizes else "time"
        total_times = processor.dataset.sizes[time_dim]

        if time_indices is None:
            time_indices = list(range(total_times))

        if not time_indices:
            logger.warning("No time steps to process; nothing to do")
            return BatchFiles([])

        manager = _create_batch_manager(processor, load_balance_strategy, n_processes)

        is_mpi_mode = manager.backend == "mpi"

        worker_kwargs = ParallelSurfaceProcessor._build_surface_worker_kwargs(
            processor,
            is_mpi_mode,
            output_dir,
            bounds,
            var_name,
            formats,
            grid_resolution,
            style,
            remap_config,
            render,
            output_name,
        )

        worker_args = [(time_idx, worker_kwargs) for time_idx in time_indices]

        os.makedirs(output_dir, exist_ok=True)

        if manager.is_master:
            logger.info(
                "Creating surface maps for %d time steps in parallel",
                len(time_indices),
            )
            logger.info("Variable: %s, Plot type: %s", var_name, plot_type)
            logger.info(_OUTDIR_MSG, output_dir)

        results = _parallel_map_batch(
            manager, _surface_worker, worker_args, "2d", worker_kwargs, processor
        )

        if manager.is_master and results is not None:
            var_info = f"Variable: {var_name}, Plot type: {plot_type}"
            created = _process_parallel_results(
                results, time_indices, output_dir, manager, "SURFACE", var_info
            )
            del results, manager
            gc.collect()
            return created

        del results, manager
        gc.collect()
        return None


class ParallelWindProcessor:
    """This class provides static methods for parallel processing of wind vector plots and visualizations using MPI-based distributed processing."""

    @staticmethod
    def _build_wind_worker_kwargs(
        processor: "MPAS2DProcessor",
        is_mpi_mode: bool,
        output_dir: str,
        bounds: GeographicBounds,
        u_variable: str,
        v_variable: str,
        formats: List[str],
        style: Optional[WindBatchStyle] = None,
        remap_config: Optional[RemapConfig] = None,
        render: Optional[BatchRenderOptions] = None,
        output_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        This helper method constructs the keyword arguments dictionary to be passed to the wind worker function based on whether MPI mode is being used or not. It ensures that the necessary information for data processing and plotting is included in the kwargs, and handles the setup of a shared data cache for multiprocessing mode to optimize performance. The method checks for required attributes in MPI mode and raises informative errors if they are missing, while in multiprocessing mode it attempts to pre-load coordinates into the cache for faster access by worker processes.

        Parameters:
            processor (MPAS2DProcessor): The processor instance containing the dataset and grid information.
            is_mpi_mode (bool): Flag indicating whether MPI mode is being used for parallel execution.
            output_dir (str): Directory path where output files will be saved.
            bounds (GeographicBounds): Plot extent as (lon_min, lon_max, lat_min, lat_max) longitude/latitude boundaries in degrees.
            u_variable (str): Name of the u-component wind variable in the dataset.
            v_variable (str): Name of the v-component wind variable in the dataset.
            formats (List[str]): List of output formats to save (e.g., ['png', 'pdf']).
            style (Optional[WindBatchStyle]): Rendering and regridding settings (plot_type, subsample, scale, show_background, grid_resolution, regrid_method). If None, defaults are used.
            remap_config (Optional[RemapConfig]): Remapping settings (remap_engine, remap_method, weights_dir). If None, defaults are used.
            render (Optional[BatchRenderOptions]): Shared rendering options (color limits, figure size and dpi, projection controls, title, wind level, background colormap, maximum height) forwarded to every worker; None keeps each worker's defaults.
            output_name (Optional[str]): User-supplied output name forwarded to the worker for file naming, or None.

        Returns:
            Dict[str, Any]: Dictionary of keyword arguments to be passed to the wind worker function, containing either 'grid_file' and 'data_dir' for MPI mode, or 'processor' and 'cache' for multiprocessing mode, along with all necessary parameters for plotting.
        """
        lon_min, lon_max, lat_min, lat_max = bounds

        if style is None:
            style = WindBatchStyle()

        plot_type = style.plot_type
        subsample = style.subsample
        scale = style.scale
        show_background = style.show_background
        grid_resolution = style.grid_resolution
        regrid_method = style.regrid_method

        if remap_config is None:
            remap_config = RemapConfig()

        remap_engine = remap_config.remap_engine
        remap_method = remap_config.remap_method

        shared_kwargs = {
            "output_dir": output_dir,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_min": lat_min,
            "lat_max": lat_max,
            "u_variable": u_variable,
            "v_variable": v_variable,
            "variables": (
                [u_variable, v_variable] + CROSS_SECTION_AUX_VARS
                if render is not None and render.wind_level_selection is not None
                else [u_variable, v_variable]
            ),
            "plot_type": plot_type,
            "subsample": subsample,
            "scale": scale,
            "show_background": show_background,
            "grid_resolution": grid_resolution,
            "regrid_method": regrid_method,
            "remap_engine": remap_engine,
            "remap_method": remap_method,
            "file_prefix": "mpas_wind",
            "formats": formats,
            "render": render,
            "weights_dir": remap_config.weights_dir,
            "output_name": output_name,
        }

        use_grid_reload = _processor_supports_reload(processor)

        if is_mpi_mode and not use_grid_reload:
            raise AttributeError(
                "MPI mode requires processor to have 'data_dir' attribute. "
                "Please update mpasdiag/processing/processors_2d.py on your HPC system, "
                "or use multiprocessing mode instead (remove mpiexec, use --workers N)."
            )

        if use_grid_reload:
            return {
                **shared_kwargs,
                **_reload_paths(processor),
            }

        cache = MPASDataCache(max_variables=5)
        logger.info(PRELOAD_COORDS_MSG)

        try:
            cache.load_coordinates_from_dataset(processor.dataset, u_variable)
            logger.debug(_COORD_CACHE_MSG, u_variable)
        except Exception as e:
            logger.warning(_COORD_PRELOAD_MSG, e)
            logger.warning(COORDS_FALLBACK_MSG)
        return {**shared_kwargs, "processor": processor, "cache": cache}

    @staticmethod
    def _log_wind_setup(
        u_variable: str,
        v_variable: str,
        plot_type: str,
        show_background: bool,
        grid_resolution: Optional[float],
        regrid_method: str,
        output_dir: str,
    ) -> None:
        """
        This helper method logs the setup information for the wind batch operation: the wind variables, the plot type, whether the background wind speed field is enabled, the grid resolution with its regrid method when set, and the output directory.

        Parameters:
            u_variable (str): Name of the u-component wind variable.
            v_variable (str): Name of the v-component wind variable.
            plot_type (str): Wind plot type being created.
            show_background (bool): Whether the background wind speed field is drawn.
            grid_resolution (Optional[float]): Grid resolution in degrees, logged only when set.
            regrid_method (str): Regridding method, logged together with the grid resolution.
            output_dir (str): Directory where plot files will be saved.

        Returns:
            None
        """
        logger.info("U variable: %s, V variable: %s", u_variable, v_variable)
        logger.info("Plot type: %s", plot_type)
        if show_background:
            logger.info("Background wind speed field: enabled")
        if grid_resolution:
            logger.info(
                "Grid resolution: %s° (regrid method: %s)",
                grid_resolution,
                regrid_method,
            )
        logger.info(_OUTDIR_MSG, output_dir)

    @staticmethod
    def _wind_processor_kind(render: Optional[BatchRenderOptions]) -> str:
        """
        This helper method chooses which kind of processor the workers reuse for a wind batch: a 3D processor when a wind level is selected, otherwise a 2D processor.

        Parameters:
            render (Optional[BatchRenderOptions]): Shared rendering options, or None.

        Returns:
            str: '3d' when render carries a wind level selection, otherwise '2d'.
        """
        if render is not None and render.wind_level_selection is not None:
            return "3d"
        return "2d"

    @staticmethod
    def create_batch_wind_plots_parallel(
        processor: "MPAS2DProcessor",
        output_dir: str,
        bounds: GeographicBounds,
        u_variable: str = "u",
        v_variable: str = "v",
        formats: Optional[List[str]] = None,
        time_indices: Optional[List[int]] = None,
        n_processes: Optional[int] = None,
        load_balance_strategy: str = "cyclic",
        style: Optional[WindBatchStyle] = None,
        remap_config: Optional[RemapConfig] = None,
        render: Optional[BatchRenderOptions] = None,
        output_name: Optional[str] = None,
    ) -> Optional[List[str]]:
        """
        This method creates wind vector plots in parallel across multiple time steps using either multiprocessing or MPI-based parallel execution. It manages the distribution of tasks to worker processes, collects results, and generates a comprehensive report on the outcomes of the parallel processing operation. The method handles both data processing and visualization phases while tracking timing metrics and cache performance for insights into efficiency.

        Parameters:
            processor (MPAS2DProcessor): Initialized MPAS2DProcessor instance with loaded wind data and grid information.
            output_dir (str): Directory path where wind plot files will be saved.
            bounds (GeographicBounds): Plot extent as (lon_min, lon_max, lat_min, lat_max) longitude/latitude boundaries in degrees.
            u_variable (str): Name of u-component wind variable in dataset (default: 'u').
            v_variable (str): Name of v-component wind variable in dataset (default: 'v').
            formats (Optional[List[str]]): List of output image formats such as ['png', 'pdf'] (default: None for ['png']).
            time_indices (Optional[List[int]]): Specific timestep indices to process, None processes all (default: None).
            n_processes (Optional[int]): Number of MPI processes to use, None uses all available (default: None).
            load_balance_strategy (str): MPI task-distribution strategy - 'cyclic' (default; uses all ranks and balances time-series cost trends), 'static', 'block', or 'dynamic' (master/worker work-stealing, best for highly irregular workloads but sacrifices one rank to dispatching). Ignored by the multiprocessing backend, whose pool already balances dynamically.
            style (Optional[WindBatchStyle]): Rendering and regridding settings (plot_type, subsample, scale, show_background, grid_resolution, regrid_method). If None, defaults are used.
            remap_config (Optional[RemapConfig]): Remapping settings (remap_engine, remap_method, weights_dir). If None, defaults are used.
            render (Optional[BatchRenderOptions]): Shared rendering options (color limits, figure size and dpi, projection controls, title, wind level, background colormap, maximum height) forwarded to every worker; None keeps each worker's defaults.
            output_name (Optional[str]): User-supplied output name; when set, files are named '<output_name>_<time tag>' exactly as the serial command-line path does with --output, instead of the descriptive default name.

        Returns:
            Optional[List[str]]: List of generated file paths on master process, None on worker processes.
        """
        lon_min, lon_max, lat_min, lat_max = bounds

        if style is None:
            style = WindBatchStyle()

        plot_type = style.plot_type
        subsample = style.subsample
        scale = style.scale
        show_background = style.show_background
        grid_resolution = style.grid_resolution
        regrid_method = style.regrid_method

        if remap_config is None:
            remap_config = RemapConfig()

        remap_engine = remap_config.remap_engine
        remap_method = remap_config.remap_method

        if formats is None:
            formats = ["png"]

        assert processor.dataset is not None
        time_dim = "Time" if "Time" in processor.dataset.sizes else "time"
        total_times = processor.dataset.sizes[time_dim]

        if time_indices is None:
            time_indices = list(range(total_times))

        if not time_indices:
            logger.warning("No time steps to process; nothing to do")
            return BatchFiles([])

        manager = _create_batch_manager(processor, load_balance_strategy, n_processes)

        is_mpi_mode = manager.backend == "mpi"

        worker_kwargs = ParallelWindProcessor._build_wind_worker_kwargs(
            processor,
            is_mpi_mode,
            output_dir,
            GeographicBounds(lon_min, lon_max, lat_min, lat_max),
            u_variable,
            v_variable,
            formats,
            style=WindBatchStyle(
                plot_type=plot_type,
                subsample=subsample,
                scale=scale,
                show_background=show_background,
                grid_resolution=grid_resolution,
                regrid_method=regrid_method,
            ),
            remap_config=RemapConfig(
                remap_engine=remap_engine,
                remap_method=remap_method,
                weights_dir=remap_config.weights_dir,
            ),
            render=render,
            output_name=output_name,
        )

        worker_args = [(time_idx, worker_kwargs) for time_idx in time_indices]

        os.makedirs(output_dir, exist_ok=True)

        if manager.is_master:
            logger.info(
                "Creating wind vector plots for %d time steps in parallel",
                len(time_indices),
            )
            ParallelWindProcessor._log_wind_setup(
                u_variable,
                v_variable,
                plot_type,
                show_background,
                grid_resolution,
                regrid_method,
                output_dir,
            )

        results = _parallel_map_batch(
            manager,
            _wind_worker,
            worker_args,
            ParallelWindProcessor._wind_processor_kind(render),
            worker_kwargs,
            processor,
        )

        if manager.is_master and results is not None:
            var_info = f"U: {u_variable}, V: {v_variable}, Plot type: {plot_type}"
            created = _process_parallel_results(
                results, time_indices, output_dir, manager, "WIND", var_info
            )
            del results, manager
            gc.collect()
            return created

        del results, manager
        gc.collect()
        return None


class ParallelCrossSectionProcessor:
    """This class provides static methods for parallel processing of vertical cross-section plots and visualizations using MPI-based distributed processing."""

    @staticmethod
    def _collect_cross_section_results(
        results: List[Any], time_indices: List[int], output_dir: str
    ) -> List[str]:
        """
        This helper function processes the results returned by parallel worker functions for cross-section plotting, aggregates timing metrics, counts successes and failures, and generates a comprehensive report summarizing the outcomes of the parallel processing operation. It collects successfully generated file paths, computes timing statistics for data processing, plotting, and saving phases, and retrieves overall parallel execution statistics from the manager to provide insights into performance and efficiency. The function prints a detailed report to the console with status counts, timing breakdowns, and potential speedup information based on the collected results.

        Parameters:
            results (List[Any]): List of results returned by parallel worker functions, where each result is expected to be a dictionary containing 'files', 'timings', and 'time_str'.
            time_indices (List[int]): List of time indices that were processed in parallel, used for reporting purposes.
            output_dir (str): Directory path where output files were saved, used for reporting the location of generated files.

        Returns:
            List[str]: List of successfully created file paths aggregated from the results.
        """
        created_files: List[str] = []
        successful = 0
        failed = 0

        for result in results:
            if result.success:
                if isinstance(result.result, dict) and "error" in result.result:
                    failed += 1
                    logger.error(
                        _FAILED_TIME_INDEX_MSG,
                        time_indices[result.task_id],
                        result.result["error"],
                    )
                    continue
                created_files.extend(
                    result.result.get("files", [])
                    if isinstance(result.result, dict)
                    else result.result
                )
                successful += 1
            else:
                failed += 1
                logger.error(
                    _FAILED_TIME_INDEX_MSG,
                    time_indices[result.task_id],
                    result.error,
                )

        print("Batch processing completed:")
        print(f"  Successful: {successful}/{len(time_indices)}")
        print(f"  Failed: {failed}/{len(time_indices)}")
        print(f"  Created {len(created_files)} files in: {output_dir}")

        batch_files = BatchFiles(created_files)
        batch_files.successful = successful
        batch_files.failed = failed
        return batch_files

    @staticmethod
    def _build_cross_section_worker_kwargs(
        mpas_3d_processor: "MPAS3DProcessor",
        var_name: str,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        output_dir: str,
        vertical_coord: str,
        num_points: int,
        formats: List[str],
        style: CrossSectionBatchStyle,
        use_grid_reload: bool,
        render: Optional[BatchRenderOptions] = None,
        output_name: Optional[str] = None,
    ) -> dict:
        """
        This helper function constructs the keyword arguments dictionary to be passed to the cross-section worker function based on whether workers will reload data from grid_file/data_dir or reuse the provided MPAS3DProcessor instance. It assembles the parameters shared by both execution modes, including cross-section endpoints, variable name, output settings, and styling options. Then, it conditionally adds either the grid_file/data_dir/variables for the reload approach or the processor object for direct reuse, ensuring that workers have the necessary information to perform their tasks based on the selected execution path.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): The processor whose grid/data paths or object the workers use.
            var_name (str): Name of the variable to plot in the cross-section.
            start_point (Tuple[float, float]): Cross-section start as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Cross-section end as (longitude, latitude) in degrees.
            output_dir (str): Directory where plot files will be saved.
            vertical_coord (str): Vertical coordinate to use for the cross-section.
            num_points (int): Number of points sampled along the cross-section line.
            formats (List[str]): Output formats to save (e.g. ['png', 'pdf']).
            style (CrossSectionBatchStyle): Appearance/file-naming settings (levels, colormap, file_prefix).
            use_grid_reload (bool): Whether workers reload data from grid_file/data_dir instead of reusing the processor.
            render (Optional[BatchRenderOptions]): Shared rendering options (color limits, figure size and dpi, projection controls, title, wind level, background colormap, maximum height) forwarded to every worker; None keeps each worker's defaults.

        Returns:
            dict: Worker keyword arguments for the selected execution path.
        """
        shared = {
            "output_dir": output_dir,
            "start_lat": start_point[1],
            "start_lon": start_point[0],
            "end_lat": end_point[1],
            "end_lon": end_point[0],
            "var_name": var_name,
            "file_prefix": style.file_prefix,
            "formats": formats,
            "custom_title": None,
            "colormap": style.colormap,
            "levels": style.levels,
            "vertical_coord": vertical_coord,
            "num_points": num_points,
            "extend": style.extend,
            "plot_type": style.plot_type,
            "render": render,
            "clim_min": style.clim_min,
            "clim_max": style.clim_max,
            "output_name": output_name,
        }

        if use_grid_reload:
            return {
                **shared,
                **_reload_paths(mpas_3d_processor),
                "variables": [var_name] + CROSS_SECTION_AUX_VARS,
            }

        return {**shared, "processor": mpas_3d_processor}

    @staticmethod
    def _precompute_vertical_levels(
        worker_kwargs: Dict[str, Any],
        mpas_3d_processor: "MPAS3DProcessor",
        var_name: str,
        vertical_coord: str,
        time_index: int,
    ) -> None:
        """
        This helper function resolves the vertical levels once on the master process so every worker reuses them, storing them under 'vertical_levels' in the worker kwargs. A failure is logged as a warning and leaves the kwargs unchanged, so each worker then resolves its own levels.

        Parameters:
            worker_kwargs (Dict[str, Any]): Worker keyword arguments (modified in-place).
            mpas_3d_processor (MPAS3DProcessor): The processor supplying the 3D data.
            var_name (str): Name of the variable to plot in the cross-section.
            vertical_coord (str): Vertical coordinate to use for the cross-section.
            time_index (int): Time index used to resolve the levels.

        Returns:
            None
        """
        try:
            probe_plotter = MPASVerticalCrossSectionPlotter(figsize=(4, 4), dpi=72)
            batch_levels, batch_coord = probe_plotter._resolve_vertical_levels(
                mpas_3d_processor, var_name, vertical_coord, time_index
            )
            worker_kwargs["vertical_levels"] = (
                np.asarray(batch_levels),
                batch_coord,
            )
        except Exception as e:
            logger.warning(
                "Could not precompute vertical levels for the batch: %s",
                e,
            )

    @staticmethod
    def _log_cross_section_setup(
        time_indices: List[int],
        var_name: str,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        vertical_coord: str,
        max_height: Optional[float],
        output_dir: str,
    ) -> None:
        """
        This helper function logs the setup information for the vertical cross-section batch processing operation, including the number of time steps being processed, variable name, cross-section endpoints, vertical coordinate, maximum height (if set), and output directory. It provides a clear summary of the parameters that will be used for the parallel processing of cross-section plots, which can be helpful for debugging and tracking the execution of the batch operation.

        Parameters:
            time_indices (List[int]): Time indices being processed (only the count is logged).
            var_name (str): Name of the variable being plotted.
            start_point (Tuple[float, float]): Cross-section start as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Cross-section end as (longitude, latitude) in degrees.
            vertical_coord (str): Vertical coordinate used for the cross-section.
            max_height (Optional[float]): Maximum height in km, logged only when set.
            output_dir (str): Directory where plot files will be saved.

        Returns:
            None
        """
        logger.info(
            "Creating vertical cross-section plots for %d time steps in parallel",
            len(time_indices),
        )
        logger.info("Variable: %s", var_name)
        logger.info(
            "Cross-section from (%.2f, %.2f) to (%.2f, %.2f)",
            start_point[0],
            start_point[1],
            end_point[0],
            end_point[1],
        )
        logger.info("Vertical coordinate: %s", vertical_coord)
        if max_height:
            logger.info("Maximum height: %s km", max_height)
        logger.info(_OUTDIR_MSG, output_dir)

    @staticmethod
    def create_batch_cross_section_plots_parallel(
        mpas_3d_processor: "MPAS3DProcessor",
        var_name: str,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        output_dir: str,
        vertical_coord: str = "height_agl",
        num_points: int = 100,
        max_height: Optional[float] = None,
        formats: List[str] = ["png"],
        time_indices: Optional[List[int]] = None,
        n_processes: Optional[int] = None,
        load_balance_strategy: str = "cyclic",
        style: Optional[CrossSectionBatchStyle] = None,
        render: Optional[BatchRenderOptions] = None,
        output_name: Optional[str] = None,
    ) -> Optional[List[str]]:
        """
        This method creates vertical cross-section plots in parallel across multiple time steps using either multiprocessing or MPI-based parallel execution. It manages the distribution of tasks to worker processes, collects results, and generates a comprehensive report on the outcomes of the parallel processing operation. The method handles both data processing and visualization phases while tracking timing metrics for insights into efficiency.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): Initialized MPAS3DProcessor instance with loaded 3D data and grid information.
            var_name (str): Name of the variable to plot in the cross-section (e.g., 'temperature').
            start_point (Tuple[float, float]): Starting point of the cross-section line as (longitude, latitude) in degrees.
            end_point (Tuple[float, float]): Ending point of the cross-section line as (longitude, latitude) in degrees.
            output_dir (str): Directory path where cross-section plot files will be saved.
            vertical_coord (str): Vertical coordinate to use for the cross-section ('height_agl', 'pressure', etc.).
            num_points (int): Number of points along the cross-section line to sample for plotting.
            max_height (Optional[float]): Maximum height in kilometers to include in the cross-section, if applicable.
            formats (List[str]): List of output formats to save (e.g., ['png', 'pdf']).
            time_indices (Optional[List[int]]): List of time indices to process. If None, all time indices in the dataset will be processed.
            n_processes (Optional[int]): Number of parallel processes to use. If None, it will default to the number of available CPU cores.
            load_balance_strategy (str): Strategy for load balancing tasks across workers ('dynamic', 'static', etc.).
            style (Optional[CrossSectionBatchStyle]): Appearance and file-naming settings (levels, colormap, extend, plot_type, file_prefix). If None, defaults are used.
            render (Optional[BatchRenderOptions]): Shared rendering options (color limits, figure size and dpi, projection controls, title, wind level, background colormap, maximum height) forwarded to every worker; None keeps each worker's defaults.
            output_name (Optional[str]): User-supplied output name; when set, files are named '<output_name>_<time tag>' exactly as the serial command-line path does with --output, instead of the descriptive default name.

        Returns:
            Optional[List[str]]: List of generated file paths on master process, None on worker processes.
        """
        if style is None:
            style = CrossSectionBatchStyle()

        render = _with_max_height(render, max_height)

        assert mpas_3d_processor.dataset is not None
        time_dim = "Time" if "Time" in mpas_3d_processor.dataset.sizes else "time"
        total_times = mpas_3d_processor.dataset.sizes[time_dim]

        if time_indices is None:
            time_indices = list(range(total_times))

        if not time_indices:
            logger.warning("No time steps to process; nothing to do")
            return BatchFiles([])

        manager = _create_batch_manager(
            mpas_3d_processor, load_balance_strategy, n_processes
        )

        is_mpi_mode = manager.backend == "mpi"
        use_grid_reload = _processor_supports_reload(mpas_3d_processor)

        if is_mpi_mode and not use_grid_reload:
            raise AttributeError(
                "MPI mode requires processor to have 'data_dir' attribute. "
                "Please update mpasdiag/processing/processors_3d.py on your HPC system, "
                "or use multiprocessing mode instead (remove mpiexec, use --workers N)."
            )

        worker_kwargs = (
            ParallelCrossSectionProcessor._build_cross_section_worker_kwargs(
                mpas_3d_processor,
                var_name,
                start_point,
                end_point,
                output_dir,
                vertical_coord,
                num_points,
                formats,
                style,
                use_grid_reload,
                render=render,
                output_name=output_name,
            )
        )

        if manager.is_master:
            ParallelCrossSectionProcessor._precompute_vertical_levels(
                worker_kwargs,
                mpas_3d_processor,
                var_name,
                vertical_coord,
                time_indices[0],
            )

        worker_args = [(time_idx, worker_kwargs) for time_idx in time_indices]

        os.makedirs(output_dir, exist_ok=True)

        if manager.is_master:
            ParallelCrossSectionProcessor._log_cross_section_setup(
                time_indices,
                var_name,
                start_point,
                end_point,
                vertical_coord,
                max_height,
                output_dir,
            )

        results = _parallel_map_batch(
            manager,
            _cross_section_worker,
            worker_args,
            "3d",
            worker_kwargs,
            mpas_3d_processor,
        )

        if manager.is_master and results is not None:
            created = ParallelCrossSectionProcessor._collect_cross_section_results(
                results, time_indices, output_dir
            )
            del results, manager
            gc.collect()
            return created

        del results, manager
        gc.collect()
        return None


class ParallelSkewTProcessor:
    """This class provides a static method for parallel generation of Skew-T sounding diagrams across timesteps using MPI-based distributed processing or multiprocessing."""

    @staticmethod
    def create_batch_skewt_plots_parallel(
        mpas_3d_processor: "MPAS3DProcessor",
        output_dir: str,
        lon: float,
        lat: float,
        formats: Optional[List[str]] = None,
        time_indices: Optional[List[int]] = None,
        n_processes: Optional[int] = None,
        load_balance_strategy: str = "cyclic",
        file_prefix: str = "mpas_skewt",
        show_parcel: bool = False,
        max_distance_km: Optional[float] = None,
        output_name: Optional[str] = None,
    ) -> Optional[List[str]]:
        """
        This method creates Skew-T sounding diagrams in parallel across multiple timesteps at a fixed station location, using either multiprocessing or MPI-based parallel execution. It distributes the per-timestep sounding extraction, thermodynamic index computation, and plotting across workers, gathers the results on the master process, and reports timing and success statistics. In MPI mode each rank reuses its already-loaded processor (no second read); in multiprocessing mode each worker reloads the needed data from grid_file/data_dir once and caches it. This replaces the previously serial, rank-0-only Skew-T path so all ranks/workers contribute.

        Parameters:
            mpas_3d_processor (MPAS3DProcessor): Initialized MPAS3DProcessor with loaded 3D data and grid information.
            output_dir (str): Directory where the Skew-T diagram files will be saved.
            lon (float): Station longitude in degrees for the sounding location.
            lat (float): Station latitude in degrees for the sounding location.
            formats (Optional[List[str]]): Output image formats such as ['png', 'pdf'] (default: ['png']).
            time_indices (Optional[List[int]]): Timestep indices to process; if None, all timesteps are processed.
            n_processes (Optional[int]): Number of parallel processes for multiprocessing; ignored under MPI.
            load_balance_strategy (str): Task distribution strategy ('dynamic', 'static', 'cyclic', 'block').
            file_prefix (str): Filename prefix for the generated diagrams.
            show_parcel (bool): Whether to draw the parcel ascent path on the diagram.
            max_distance_km (Optional[float]): Largest allowed distance in kilometers between the requested point and the nearest cell used; None applies the diagnostics' default of three times the local cell spacing. A point outside the mesh makes every step fail with the diagnostics' message.
            output_name (Optional[str]): User-supplied output name; when set, files are named '<output_name>_<time tag>' exactly as the serial command-line path does with --output, instead of the descriptive default name.

        Returns:
            Optional[List[str]]: List of generated file paths on the master process, None on worker ranks.
        """
        if formats is None:
            formats = ["png"]

        assert mpas_3d_processor.dataset is not None
        time_dim = "Time" if "Time" in mpas_3d_processor.dataset.sizes else "time"
        total_times = mpas_3d_processor.dataset.sizes[time_dim]

        if time_indices is None:
            time_indices = list(range(total_times))

        if not time_indices:
            logger.warning("No time steps to process; nothing to do")
            return BatchFiles([])

        manager = _create_batch_manager(
            mpas_3d_processor, load_balance_strategy, n_processes
        )

        is_mpi_mode = manager.backend == "mpi"
        use_grid_reload = _processor_supports_reload(mpas_3d_processor)

        if is_mpi_mode and not use_grid_reload:
            raise AttributeError(
                "MPI mode requires processor to have 'data_dir' attribute. "
                "Please update mpasdiag/processing/processors_3d.py on your HPC system, "
                "or use multiprocessing mode instead (remove mpiexec, use --workers N)."
            )

        shared_kwargs = {
            "output_dir": output_dir,
            "lon": lon,
            "lat": lat,
            "file_prefix": file_prefix,
            "formats": formats,
            "show_parcel": show_parcel,
            "max_distance_km": max_distance_km,
            "output_name": output_name,
        }

        if use_grid_reload:
            worker_kwargs = {
                **shared_kwargs,
                **_reload_paths(mpas_3d_processor),
            }
        else:
            worker_kwargs = {**shared_kwargs, "processor": mpas_3d_processor}

        if manager.is_master:
            worker_kwargs.update(
                _resolve_station_cell(mpas_3d_processor, lon, lat, max_distance_km)
            )

        worker_args = [(time_idx, worker_kwargs) for time_idx in time_indices]

        os.makedirs(output_dir, exist_ok=True)

        if manager.is_master:
            logger.info(
                "Creating Skew-T diagrams for %d time steps in parallel",
                len(time_indices),
            )
            logger.info("Station: (%.2f, %.2f)", lon, lat)
            logger.info(_OUTDIR_MSG, output_dir)

        results = _parallel_map_batch(
            manager, _skewt_worker, worker_args, "3d", worker_kwargs, mpas_3d_processor
        )

        if manager.is_master and results is not None:
            var_info = f"Station: ({lon:.2f}, {lat:.2f})"
            created = _process_parallel_results(
                results, time_indices, output_dir, manager, "SKEW-T", var_info
            )
            del results, manager
            gc.collect()
            return created

        del results, manager
        gc.collect()
        return None


def _resolve_station_cell(
    processor: Any, lon: float, lat: float, max_distance_km: Optional[float]
) -> Dict[str, Any]:
    """
    This helper resolves the grid cell of a fixed sounding station once for a whole batch so the per-step workers can skip the nearest-cell search. It also performs the distance check the workers' fast path skips: the cell is only returned when the requested point lies within max_distance_km of it (or within three times the local cell spacing when no maximum was given), so a point outside the mesh yields no precomputed cell and every step then runs the full check in the diagnostics and fails with its clear message. Any problem while resolving the cell is logged and leaves the workers on their normal path.

    Parameters:
        processor (Any): The 3D processor providing the grid coordinates.
        lon (float): Requested station longitude in degrees.
        lat (float): Requested station latitude in degrees.
        max_distance_km (Optional[float]): Largest allowed distance to the cell in kilometers, or None for the default.

    Returns:
        Dict[str, Any]: Worker keyword arguments 'cell_index', 'station_lon', and 'station_lat', or an empty dictionary when the fast path must not be used.
    """
    try:
        diag = SoundingDiagnostics(verbose=False)
        grid_lon, grid_lat = diag._load_grid_coordinates(processor)
        cell_idx = int(diag._find_nearest_cell(grid_lon, grid_lat, lon, lat))
        station_lon, station_lat = float(grid_lon[cell_idx]), float(grid_lat[cell_idx])
        threshold_km = (
            float(max_distance_km)
            if max_distance_km is not None
            else 3.0 * diag._nearest_neighbor_distance_km(grid_lon, grid_lat, cell_idx)
        )
        if diag._haversine_km(lon, lat, station_lon, station_lat) > threshold_km:
            return {}
        return {
            "cell_index": cell_idx,
            "station_lon": station_lon,
            "station_lat": station_lat,
        }
    except Exception as e:
        logger.warning(
            "Could not precompute the sounding cell index for the batch: %s", e
        )
        return {}


def auto_batch_processor(use_parallel: Optional[bool] = None, **kwargs: Any) -> bool:
    """
    This function automatically determines whether to use parallel processing based on the presence of an MPI environment and the specified `use_parallel` flag. It checks for the availability of the `mpi4py` library and the number of MPI processes to decide if parallel execution is feasible. The function provides a flexible interface for selecting processing mode, allowing users to explicitly override the decision or rely on automatic detection.

    Parameters:
        use_parallel (Optional[bool]): Optional boolean flag to explicitly select processing mode. If None, the function will attempt to auto-detect MPI environment and decide based on the number of processes. If True, it will force parallel processing; if False, it will force serial execution.

    Returns:
        bool: True if parallel processing should be used, False for serial execution.
    """
    if use_parallel is None and not (
        "mpi4py" in sys.modules or mpi_launcher_detected()
    ):
        # No MPI launcher in the environment: stay serial without importing (and initialising) mpi4py
        use_parallel = False

    if use_parallel is None:
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            use_parallel = comm.Get_size() > 1
        except ImportError:
            use_parallel = False

    return use_parallel

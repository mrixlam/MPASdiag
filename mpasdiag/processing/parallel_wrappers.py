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

import gc
import os
import time
from types import SimpleNamespace
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Dict, cast
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
from mpasdiag.processing.constants import (
    PRECIP_REQUIRED_VARS,
    CROSS_SECTION_AUX_VARS,
    COORDS_FALLBACK_MSG,
    PRELOAD_COORDS_MSG,
)
from mpasdiag.processing.utils_logger import get_logger
from mpasdiag.visualization.precipitation import (
    PrecipitationRenderStyle,
    PrecipitationMapStyle,
)

logger = get_logger(__name__)


@dataclass
class RemapConfig:
    """Remapping settings for parallel batch plotting, grouping the remap engine, interpolation method, and ESMF weights cache directory into a single value object."""

    remap_engine: str = "kdtree"
    remap_method: str = "nearest"
    weights_dir: Optional[str] = None


@dataclass
class SurfaceBatchStyle:
    """Appearance and file-naming settings for parallel batch surface maps, grouping the output filename prefix, plot type, and color scale limits into a single value object."""

    file_prefix: str = "mpas_surface"
    plot_type: str = "scatter"
    clim_min: Optional[float] = None
    clim_max: Optional[float] = None


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
    colormap: str = "viridis"
    extend: str = "both"
    plot_type: str = "contourf"
    file_prefix: str = "mpas_cross_section"


_DATETIME_HOUR_FORMAT = "%Y%m%dT%H"
_FAILED_TIME_INDEX_MSG = "Failed time index %s: %s"
_COORD_CACHE_MSG = "Coordinates cached for variable: %s"
_COORD_PRELOAD_MSG = "Could not pre-load coordinates into cache: %s"
_OUTDIR_MSG = "Output directory: %s"

_rank_processor_cache: Dict[str, Any] = {}

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

    cache_key = (
        f"2d|{grid_file}|{data_dir}|{tuple(sorted(variables)) if variables else None}"
    )

    if cache_key not in _rank_processor_cache:
        from mpasdiag.processing.processors_2d import MPAS2DProcessor

        processor = MPAS2DProcessor(grid_file, verbose=False)
        processor = processor.load_2d_data(data_dir, variables=variables)
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

    cache_key = (
        f"3d|{grid_file}|{data_dir}|{tuple(sorted(variables)) if variables else None}"
    )

    if cache_key not in _rank_processor_cache:
        from mpasdiag.processing.processors_3d import MPAS3DProcessor

        processor = MPAS3DProcessor(grid_file, verbose=False)
        processor = processor.load_3d_data(data_dir, variables=variables)
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


def _processor_supports_reload(processor: Any) -> bool:
    """
    This helper reports whether a processor can be reconstructed from disk inside a worker, which is true when it exposes string 'grid_file' and 'data_dir' paths (set by load_2d_data/load_3d_data). When True, the parallel batch methods hand those lightweight paths to workers instead of the live processor, so each worker (multiprocessing) or rank (MPI) loads ONLY the variables it needs and does so once -- caching the result in the module-level processor cache -- rather than having the full in-memory dataset pickled to it for every single task. In-memory processors built without loading from disk return False and fall back to passing the processor object directly, preserving backward compatibility.

    Parameters:
        processor (Any): The 2D or 3D processor instance to inspect.

    Returns:
        bool: True if both 'grid_file' and 'data_dir' are string paths on the processor.
    """
    return isinstance(getattr(processor, "grid_file", None), str) and isinstance(
        getattr(processor, "data_dir", None), str
    )


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

    variables = worker_kwargs.get("variables")
    dataset = getattr(processor, "dataset", None)

    if dataset is None:
        return None

    try:
        for variable in variables or []:
            _ = dataset[variable]
    except Exception:
        return None

    cache_key = f"{kind}|{grid_file}|{data_dir}|{tuple(sorted(variables)) if variables else None}"
    _rank_processor_cache[cache_key] = processor
    return cache_key


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
        return time_end.strftime(_DATETIME_HOUR_FORMAT), time_end
    return f"t{time_idx:03d}", None


def _precipitation_worker(args: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    This worker function executes precipitation diagnostic processing and plotting for a single timestep with comprehensive timing metrics. It serves as a picklable entry point for parallel processing of precipitation diagnostics across multiple timesteps using a shared data cache. The function extracts spatial coordinates, computes precipitation differences, generates visualizations with the MPASPrecipitationPlotter, and saves outputs in specified formats. It measures execution time for data extraction, plotting operations, and file writing while tracking cache hit statistics for performance analysis.

    Parameters:
        args (Tuple[int, Dict[str, Any]]): Two-element tuple with (time_idx, kwargs) where time_idx is the timestep index and kwargs contains processor instance, cache object, spatial bounds, variable name, accumulation period, plot type, and formatting options.

    Returns:
        Dict[str, Any]: Result dictionary with 'files' (list of str paths), 'timings' (dict with phase durations), 'time_str' (str), 'cache_hits' (dict), and optionally 'cache_info' (dict with cache statistics).
    """
    try:
        time_idx, kwargs = args

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
        remap_engine = kwargs.get("remap_engine", "kdtree")
        remap_method = kwargs.get("remap_method", "nearest")
        remap_config = SimpleNamespace(
            remap_engine=remap_engine, remap_method=remap_method
        )

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

        plotter = MPASPrecipitationPlotter(figsize=(10, 14))

        weights_dir = kwargs.get("weights_dir")

        if weights_dir is not None:
            from pathlib import Path

            plotter._remapper_weights_dir = Path(weights_dir)

        if custom_title_template:
            title = custom_title_template.format(
                var_name=var_name.upper(), time_str=time_str, accum_period=accum_period
            )
        else:
            title = f"MPAS Precipitation | PlotType: {plot_type.upper()} | VarType: {var_name.upper()} | Valid Time: {time_str}"

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
            ),
        )

        timings["plotting"] = time.time() - plot_start
        save_start = time.time()

        output_path = os.path.join(
            output_dir,
            f"{file_prefix}_vartype_{var_name}_acctype_{accum_period}_valid_{time_str}_ptype_{plot_type}",
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


def _surface_worker(args: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    This worker function executes surface variable processing and plotting for a single timestep with comprehensive timing metrics. It serves as a picklable entry point for parallel processing of surface diagnostics across multiple timesteps using a shared data cache. The function extracts spatial coordinates, retrieves variable data, generates visualizations with the MPASSurfacePlotter, and saves outputs in specified formats. It measures execution time for data extraction, plotting operations, and file writing while tracking cache hit statistics for performance analysis.

    Parameters:
        args (Tuple[int, Dict[str, Any]]): Two-element tuple containing (time_idx, kwargs) where time_idx is the integer timestep index and kwargs is a dictionary with processor instance, cache object, spatial bounds, variable name, plot type, and formatting options.

    Returns:
        Dict[str, Any]: Result dictionary with keys 'files' (list of str paths), 'timings' (dict with phase durations), 'time_str' (str), 'cache_hits' (dict), and optionally 'cache_info' (dict with cache statistics).
    """
    time_idx, kwargs = args

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

    start_time = time.time()
    timings = {}
    cache_hits = {"coordinates": False, "data": False}

    data_start = time.time()

    if cache is not None:
        try:
            lon, lat = cache.get_coordinates(var_name)
            cache_hits["coordinates"] = True
        except KeyError:
            lon, lat = processor.extract_spatial_coordinates()
            try:
                cache.load_coordinates_from_dataset(processor.dataset, var_name)
            except Exception:
                pass
    else:
        lon, lat = processor.extract_spatial_coordinates()

    var_data = processor.dataset[var_name].isel(Time=time_idx)

    plotter = MPASSurfacePlotter(figsize=(10, 14))
    timings["data_processing"] = time.time() - data_start

    plot_start = time.time()

    _, _ = plotter.create_surface_map(
        lon=lon,
        lat=lat,
        data=var_data.values,
        var_name=var_name,
        bounds=GeographicBounds(lon_min, lon_max, lat_min, lat_max),
        style=SurfaceMapStyle(
            title=custom_title,
            plot_type=plot_type,
            colormap=colormap,
            levels=levels,
        ),
        time_stamp=pd.Timestamp(str(processor.dataset["Time"].values[time_idx])),
        dataset=processor.dataset,
        config=remap_config,
    )
    timings["plotting"] = time.time() - plot_start

    save_start = time.time()

    time_str = pd.Timestamp(processor.dataset["Time"].values[time_idx]).strftime(
        _DATETIME_HOUR_FORMAT
    )
    safe_time_str = time_str

    output_path = os.path.join(
        output_dir, f"{file_prefix}_{var_name}_{plot_type}_valid_{safe_time_str}"
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


def _wind_worker(args: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    This worker function executes wind vector processing and plotting for a single timestep with comprehensive timing metrics. It serves as a picklable entry point for parallel processing of wind diagnostics across multiple timesteps using a shared data cache. The function extracts spatial coordinates, retrieves 2D variable data for the specified U and V wind components, generates visualizations with the MPASWindPlotter, and saves outputs in specified formats. It measures execution time for data extraction, plotting operations, and file writing while tracking cache hit statistics for performance analysis.

    Parameters:
        args (Tuple[int, Dict[str, Any]]): Two-element tuple containing (time_idx, kwargs) where time_idx is the integer timestep index and kwargs is a dictionary with processor instance, cache object, spatial bounds, variable names for U and V components, plot type, subsampling factor, scaling options, and formatting settings.

    Returns:
        Dict[str, Any]: Result dictionary with keys 'files' (list of str paths), 'timings' (dict with phase durations), 'time_str' (str), 'cache_hits' (dict), and optionally 'cache_info' (dict with cache statistics).
    """
    time_idx, kwargs = args

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
        time_str = time_end.strftime(_DATETIME_HOUR_FORMAT)
    else:
        time_str = f"t{time_idx:03d}"

    timings["data_processing"] = time.time() - data_start

    plotter = MPASWindPlotter(figsize=(12, 10))
    plot_start = time.time()

    _, _ = plotter.create_wind_plot(
        lon,
        lat,
        u_data.values,
        v_data.values,
        GeographicBounds(lon_min, lon_max, lat_min, lat_max),
        style=WindPlotStyle(
            plot_type=plot_type,
            subsample=subsample,
            scale=scale,
            show_background=show_background,
            time_stamp=None,
        ),
        grid_resolution=grid_resolution,
        regrid_method=regrid_method,
        dataset=processor.dataset,
        config=remap_config,
    )

    timings["plotting"] = time.time() - plot_start

    save_start = time.time()
    base_name = f"mpas_wind_{u_variable}_{v_variable}_{plot_type}_valid_{time_str}"
    output_path = os.path.join(output_dir, base_name)

    plotter.add_timestamp_and_branding()
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


def _cross_section_worker(args: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    This worker function executes vertical cross-section processing and plotting for a single timestep with comprehensive timing metrics. It serves as a picklable entry point for parallel processing of vertical cross-section diagnostics across multiple timesteps using a shared data cache. The function extracts spatial coordinates, retrieves 3D variable data, generates visualizations with the MPASVerticalCrossSectionPlotter, and saves outputs in specified formats. It measures execution time for data extraction, plotting operations, and file writing while tracking cache hit statistics for performance analysis.

    Parameters:
        args (Tuple[int, Dict[str, Any]]): Two-element tuple containing (time_idx, kwargs) where time_idx is the integer timestep index and kwargs is a dictionary with processor instance, cache object, spatial bounds for start and end points, variable name, vertical coordinate type, number of points along the cross-section, plot type, and formatting options.

    Returns:
        Dict[str, Any]: Result dictionary with keys 'files' (list of str paths), 'timings' (dict with phase durations), 'time_str' (str), 'cache_hits' (dict), and optionally 'cache_info' (dict with cache statistics).
    """
    time_idx, kwargs = args

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

    start_time = time.time()
    timings = {}

    data_start = time.time()

    plotter = MPASVerticalCrossSectionPlotter(figsize=(10, 14))
    timings["data_processing"] = time.time() - data_start

    plot_start = time.time()

    time_str = str(processor_3d.dataset["Time"].values[time_idx])
    safe_time_str = time_str.replace(":", "").replace("-", "").replace(" ", "T")[:13]
    save_path = os.path.join(
        output_dir,
        f"{file_prefix}_{var_name}_vcrd_{vertical_coord}_valid_{safe_time_str}.png",
    )

    fig, _ = plotter.create_vertical_cross_section(
        mpas_3d_processor=processor_3d,
        var_name=var_name,
        start_point=(start_lon, start_lat),
        end_point=(end_lon, end_lat),
        time_index=time_idx,
        vertical_coord=vertical_coord,
        num_points=num_points,
        save_path=save_path,
        style=CrossSectionStyle(levels=levels, colormap=colormap),
        precomputed_levels=precomputed_levels,
    )

    timings["plotting"] = time.time() - plot_start
    phase_timings = getattr(plotter, "_phase_timings", None)

    if isinstance(phase_timings, dict):
        for phase_name, phase_seconds in phase_timings.items():
            timings[f"xs_{phase_name}"] = phase_seconds

    save_start = time.time()
    output_files = [save_path]

    for fmt in formats:
        if fmt != "png":
            output_file = save_path.replace(".png", f".{fmt}")
            save_kwargs: Dict[str, Any] = {"dpi": 100, "bbox_inches": "tight"}

            if fmt.lower() == "png":
                save_kwargs["pil_kwargs"] = {"compress_level": 1}

            fig.savefig(output_file, **save_kwargs)
            output_files.append(output_file)

    plotter.close_plot()
    del plotter
    _maybe_collect_garbage()

    timings["saving"] = time.time() - save_start
    timings["total"] = time.time() - start_time

    return {"files": output_files, "timings": timings, "time_str": time_str}


def _skewt_worker(args: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    This worker function executes Skew-T sounding diagnostics and plotting for a single timestep, serving as a picklable entry point for parallel generation of Skew-T diagrams across many timesteps. It obtains a 3D processor (reusing the rank's already-loaded processor in MPI mode, or reloading from grid_file/data_dir in multiprocessing mode), extracts the vertical sounding profile at the configured station location, computes thermodynamic and severe-weather indices, renders the Skew-T diagram, and writes the requested output formats. It records per-phase timing for data processing, plotting, and saving so the aggregate report matches the other parallel plot types.

    Parameters:
        args (Tuple[int, Dict[str, Any]]): Two-element tuple of (time_idx, kwargs) where time_idx is the timestep index and kwargs supplies either a loaded 'processor' or 'grid_file'/'data_dir', the station 'lon'/'lat', 'output_dir', 'file_prefix', 'formats', and the 'show_parcel' flag.

    Returns:
        Dict[str, Any]: Result dictionary with 'files' (list of str paths), 'timings' (dict with phase durations), 'time_str' (str), and 'cache_hits' (dict).
    """
    time_idx, kwargs = args

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
    )

    indices = diag.compute_thermodynamic_indices(
        profile["pressure"],
        profile["temperature"],
        profile["dewpoint"],
        u_wind_kt=profile.get("u_wind"),
        v_wind_kt=profile.get("v_wind"),
        height_m=profile.get("height"),
    )

    time_str, _ = _get_time_str(processor_3d.dataset, time_idx)

    timings["data_processing"] = time.time() - data_start

    stn_lon = profile["station_lon"]
    stn_lat = profile["station_lat"]
    lon_tag = f"{abs(stn_lon):.2f}{'W' if stn_lon < 0 else 'E'}"
    lat_tag = f"{abs(stn_lat):.2f}{'S' if stn_lat < 0 else 'N'}"

    output_path = os.path.join(
        output_dir,
        f"{file_prefix}_{lon_tag.replace('.', 'p')}_{lat_tag.replace('.', 'p')}_valid_{time_str}",
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

    return created_files


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
            "remap_engine": remap_config.remap_engine,
            "remap_method": remap_config.remap_method,
            "weights_dir": remap_config.weights_dir,
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
                "grid_file": processor.grid_file,
                "data_dir": processor.data_dir,
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

        Returns:
            Optional[List[str]]: List of generated file paths on master process, None on worker processes.
        """
        lon_min, lon_max, lat_min, lat_max = bounds

        if style is None:
            style = PrecipitationMapStyle()

        if remap_config is None:
            remap_config = RemapConfig()

        accum_hours = int(accum_period[1:3])
        hours_per_file = 1
        min_time_idx = accum_hours // hours_per_file

        assert processor.dataset is not None
        time_dim = "Time" if "Time" in processor.dataset.sizes else "time"
        total_times = processor.dataset.sizes[time_dim]

        if time_indices is None:
            time_indices = list(range(min_time_idx, total_times))
        else:
            time_indices = [idx for idx in time_indices if idx >= min_time_idx]

        if not time_indices:
            logger.warning(
                "No valid time indices for accumulation period %s",
                accum_period,
            )
            return []

        manager = MPASParallelManager(
            load_balance_strategy=load_balance_strategy,
            verbose=True,
            n_workers=n_processes,
        )

        manager.set_error_policy("collect")
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

        seeded_key = (
            _seed_worker_processor_cache("2d", worker_kwargs, processor)
            if is_mpi_mode
            else None
        )

        results = manager.parallel_map(_precipitation_worker, tasks)

        if seeded_key is not None:
            _rank_processor_cache.pop(seeded_key, None)

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

        Returns:
            Optional[List[str]]: List of generated file paths on master process, None on worker processes.
        """
        lon_min, lon_max, lat_min, lat_max = bounds

        if style is None:
            style = SurfaceBatchStyle()

        file_prefix = style.file_prefix
        plot_type = style.plot_type

        if remap_config is None:
            remap_config = RemapConfig()

        remap_engine = remap_config.remap_engine
        remap_method = remap_config.remap_method

        assert processor.dataset is not None
        time_dim = "Time" if "Time" in processor.dataset.sizes else "time"
        total_times = processor.dataset.sizes[time_dim]

        if time_indices is None:
            time_indices = list(range(total_times))

        manager = MPASParallelManager(
            load_balance_strategy=load_balance_strategy,
            verbose=True,
            n_workers=n_processes,
        )
        manager.set_error_policy("collect")

        is_mpi_mode = manager.backend == "mpi"
        use_grid_reload = _processor_supports_reload(processor)

        if is_mpi_mode and not use_grid_reload:
            raise AttributeError(
                "MPI mode requires processor to have 'data_dir' attribute. "
                "Please update mpasdiag/processing/processors_2d.py on your HPC system, "
                "or use multiprocessing mode instead (remove mpiexec, use --workers N)."
            )

        if use_grid_reload:
            worker_kwargs = {
                "grid_file": processor.grid_file,
                "data_dir": processor.data_dir,
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
                "custom_title": None,
                "colormap": None,
                "levels": None,
                "remap_engine": remap_engine,
                "remap_method": remap_method,
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
                "custom_title": None,
                "colormap": None,
                "levels": None,
                "remap_engine": remap_engine,
                "remap_method": remap_method,
            }

        worker_args = [(time_idx, worker_kwargs) for time_idx in time_indices]

        os.makedirs(output_dir, exist_ok=True)

        if manager.is_master:
            logger.info(
                "Creating surface maps for %d time steps in parallel",
                len(time_indices),
            )
            logger.info("Variable: %s, Plot type: %s", var_name, plot_type)
            logger.info(_OUTDIR_MSG, output_dir)

        seeded_key = (
            _seed_worker_processor_cache("2d", worker_kwargs, processor)
            if is_mpi_mode
            else None
        )

        results = manager.parallel_map(_surface_worker, worker_args)

        if seeded_key is not None:
            _rank_processor_cache.pop(seeded_key, None)

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
            "variables": [u_variable, v_variable],
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
                "grid_file": processor.grid_file,
                "data_dir": processor.data_dir,
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

        manager = MPASParallelManager(
            load_balance_strategy=load_balance_strategy,
            verbose=True,
            n_workers=n_processes,
        )
        manager.set_error_policy("collect")

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
                remap_engine=remap_engine, remap_method=remap_method
            ),
        )

        worker_args = [(time_idx, worker_kwargs) for time_idx in time_indices]

        os.makedirs(output_dir, exist_ok=True)

        if manager.is_master:
            logger.info(
                "Creating wind vector plots for %d time steps in parallel",
                len(time_indices),
            )
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

        seeded_key = (
            _seed_worker_processor_cache("2d", worker_kwargs, processor)
            if is_mpi_mode
            else None
        )

        results = manager.parallel_map(_wind_worker, worker_args)

        if seeded_key is not None:
            _rank_processor_cache.pop(seeded_key, None)

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

        return created_files

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
        }

        if use_grid_reload:
            return {
                **shared,
                "grid_file": mpas_3d_processor.grid_file,
                "data_dir": mpas_3d_processor.data_dir,
                "variables": [var_name] + CROSS_SECTION_AUX_VARS,
            }

        return {**shared, "processor": mpas_3d_processor}

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

        Returns:
            Optional[List[str]]: List of generated file paths on master process, None on worker processes.
        """
        if style is None:
            style = CrossSectionBatchStyle()

        assert mpas_3d_processor.dataset is not None
        time_dim = "Time" if "Time" in mpas_3d_processor.dataset.sizes else "time"
        total_times = mpas_3d_processor.dataset.sizes[time_dim]

        if time_indices is None:
            time_indices = list(range(total_times))

        manager = MPASParallelManager(
            load_balance_strategy=load_balance_strategy,
            verbose=True,
            n_workers=n_processes,
        )
        manager.set_error_policy("collect")

        is_mpi_mode = manager.backend == "mpi"
        use_grid_reload = _processor_supports_reload(mpas_3d_processor)

        if is_mpi_mode and not use_grid_reload:
            raise AttributeError(
                "MPI mode requires processor to have 'data_dir' attribute. "
                "Please update mpasdiag/processing/processors_3d.py on your HPC system, "
                "or use multiprocessing mode instead (remove mpiexec, use --workers N)."
            )

        worker_kwargs = ParallelCrossSectionProcessor._build_cross_section_worker_kwargs(
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
        )

        if manager.is_master:
            try:
                probe_plotter = MPASVerticalCrossSectionPlotter(figsize=(4, 4), dpi=72)
                batch_levels, batch_coord = probe_plotter._resolve_vertical_levels(
                    mpas_3d_processor, var_name, vertical_coord, time_indices[0]
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

        seeded_key = (
            _seed_worker_processor_cache("3d", worker_kwargs, mpas_3d_processor)
            if is_mpi_mode
            else None
        )

        results = manager.parallel_map(_cross_section_worker, worker_args)

        if seeded_key is not None:
            _rank_processor_cache.pop(seeded_key, None)

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

        manager = MPASParallelManager(
            load_balance_strategy=load_balance_strategy,
            verbose=True,
            n_workers=n_processes,
        )
        manager.set_error_policy("collect")

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
        }

        if use_grid_reload:
            worker_kwargs = {
                **shared_kwargs,
                "grid_file": mpas_3d_processor.grid_file,
                "data_dir": mpas_3d_processor.data_dir,
            }
        else:
            worker_kwargs = {**shared_kwargs, "processor": mpas_3d_processor}

        if manager.is_master:
            try:
                station_diag = SoundingDiagnostics(verbose=False)
                grid_lon, grid_lat = station_diag._load_grid_coordinates(
                    mpas_3d_processor
                )
                cell_idx = station_diag._find_nearest_cell(grid_lon, grid_lat, lon, lat)
                worker_kwargs["cell_index"] = int(cell_idx)
                worker_kwargs["station_lon"] = float(grid_lon[cell_idx])
                worker_kwargs["station_lat"] = float(grid_lat[cell_idx])
            except Exception as e:
                logger.warning(
                    "Could not precompute the sounding cell index for the batch: %s",
                    e,
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

        seeded_key = (
            _seed_worker_processor_cache("3d", worker_kwargs, mpas_3d_processor)
            if is_mpi_mode
            else None
        )

        results = manager.parallel_map(_skewt_worker, worker_args)

        if seeded_key is not None:
            _rank_processor_cache.pop(seeded_key, None)

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


def auto_batch_processor(use_parallel: Optional[bool] = None, **kwargs: Any) -> bool:
    """
    This function automatically determines whether to use parallel processing based on the presence of an MPI environment and the specified `use_parallel` flag. It checks for the availability of the `mpi4py` library and the number of MPI processes to decide if parallel execution is feasible. The function provides a flexible interface for selecting processing mode, allowing users to explicitly override the decision or rely on automatic detection.

    Parameters:
        use_parallel (Optional[bool]): Optional boolean flag to explicitly select processing mode. If None, the function will attempt to auto-detect MPI environment and decide based on the number of processes. If True, it will force parallel processing; if False, it will force serial execution.

    Returns:
        bool: True if parallel processing should be used, False for serial execution.
    """
    if use_parallel is None:
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            use_parallel = comm.Get_size() > 1
        except ImportError:
            use_parallel = False

    return use_parallel

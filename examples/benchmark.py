#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Benchmark: Batch Plotter Runtime with MPI Support

This script benchmarks the runtime of MPASdiag's batch plotting functions for various plot types (precipitation, surface, wind, cross-section, skew-T) across different MPAS model resolutions. It supports both serial and parallel execution using MPI (via mpi4py). The results are saved to a CSV file and printed in a summary table.  

Usage: 
    # Serial baseline (single process):
    python benchmark.py --serial

    # Shared-memory parallel (multiprocessing Pool) -- default without mpiexec:
    python benchmark.py --workers 8

    # Distributed parallel (MPI) -- fastest; one rank per process:
    mpiexec -n 8 python benchmark.py

    # Restrict to a subset of experiments (any launch mode):
    python benchmark.py --experiments u240k u120k

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: March 2026
Version: 1.0.0
"""
import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '1')

# Load standard libraries
import gc
import csv
import time
import socket
import argparse
from pathlib import Path
from datetime import datetime
from multiprocessing import cpu_count
from typing import Callable

# Load relevant MPASdiag modules
from mpasdiag.visualization.wind import MPASWindPlotter
from mpasdiag.visualization.skewt import MPASSkewTPlotter
from mpasdiag.diagnostics.sounding import SoundingDiagnostics
from mpasdiag.visualization.surface import MPASSurfacePlotter
from mpasdiag.processing.utils_geog import GeographicBounds
from mpasdiag.processing.processors_2d import MPAS2DProcessor
from mpasdiag.processing.processors_3d import MPAS3DProcessor
from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter
from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter

from mpasdiag.processing.parallel_wrappers import (
    ParallelPrecipitationProcessor,
    ParallelSurfaceProcessor,
    ParallelWindProcessor,
    ParallelCrossSectionProcessor,
    ParallelSkewTProcessor,
    SurfaceBatchStyle,
    WindBatchStyle,
)


try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    SIZE = 1

HOSTNAME = socket.gethostname()
N_NODES = len(set(COMM.allgather(HOSTNAME))) if COMM is not None else 1

if SIZE > 1:
    try:
        import dask
        dask.config.set(scheduler='synchronous')
    except ImportError:
        pass

BACKEND = 'serial'
PARALLEL_WIDTH = 1

EXPERIMENTS = {
    'u240k': {
        'grid_file': '../data/grids/x1.10242.static.nc',
        'diag_dir': '../data/u240k/diag',
        'mpasout_dir': '../data/u240k/mpasout',
    },
    # 'u120k': {
    #     'grid_file': '../data/grids/x1.40962.static.nc',
    #     'diag_dir': '../data/u120k/diag',
    #     'mpasout_dir': '../data/u120k/mpasout',
    # },
}

SPATIAL_BOUNDS = {
    'lon_min': 91.0,
    'lon_max': 113.0,
    'lat_min': -9.6,
    'lat_max': 12.2,
}

CROSS_SECTION_CONFIG = {
    'start_point': (95.0, -5.0),
    'end_point': (110.0, 10.0),
    'variable': 'theta',
    'vertical_coord': 'pressure',
    'num_points': 100,
}

SKEWT_CONFIG = {
    'lon': 103.2,
    'lat': 3.8,
}

WIND_CONFIG = {
    'u_variable': 'u10',
    'v_variable': 'v10',
    'plot_type': 'barbs',
    'subsample': -1,
    'grid_resolution': 0.1,
}

BENCHMARK_DIR = Path('../output/benchmarks')


def run_benchmark_precipitation(processor_2d: MPAS2DProcessor, 
                                out_dir: str, 
                                use_parallel: bool, 
                                n_workers: int) -> tuple[float, int]:
    """
    This function benchmarks the batch precipitation map creation. It creates a directory for precipitation plots, runs the batch plotting function (either in parallel or serial), and returns the elapsed time and number of files created. The function takes in the MPAS2DProcessor with loaded data, the output directory path, a boolean indicating whether to use parallel processing, and the number of worker processes to use if parallel processing is enabled. 

    Parameters:
        processor_2d: An instance of MPAS2DProcessor with loaded 2D diagnostic data.
        out_dir: The base output directory where the 'precipitation' subdirectory will be created for storing the generated plots.
        use_parallel: A boolean flag indicating whether to use parallel processing for generating the plots.
        n_workers: The number of worker processes to use if parallel processing is enabled. This should typically be set to the number of available CPU cores or the number of MPI ranks.

    Returns:
        tuple[float, int]: A tuple containing the elapsed time in seconds for the batch plotting operation and the number of plot files created. The elapsed time is measured from the start of the plotting function to its completion, and the number of files is determined by counting the created plot files in the output directory.
    """
    plot_dir = os.path.join(out_dir, 'precipitation')
    os.makedirs(plot_dir, exist_ok=True)

    t0 = time.perf_counter()
    if use_parallel:
        created = ParallelPrecipitationProcessor.create_batch_precipitation_maps_parallel(
            processor_2d, plot_dir,
            GeographicBounds(SPATIAL_BOUNDS['lon_min'], SPATIAL_BOUNDS['lon_max'],
                             SPATIAL_BOUNDS['lat_min'], SPATIAL_BOUNDS['lat_max']),
            var_name='total',
            accum_period='a01h',
            plot_type='scatter',
            grid_resolution=WIND_CONFIG['grid_resolution'],
            n_processes=n_workers,
        ) or []
    else:
        plotter = MPASPrecipitationPlotter(figsize=(12, 12), dpi=100)
        created = plotter.create_batch_precipitation_maps(
            processor_2d, plot_dir,
            GeographicBounds(SPATIAL_BOUNDS['lon_min'], SPATIAL_BOUNDS['lon_max'],
                             SPATIAL_BOUNDS['lat_min'], SPATIAL_BOUNDS['lat_max']),
            var_name='total',
            accum_period='a01h',
            grid_resolution=WIND_CONFIG['grid_resolution'],
            plot_type='scatter',
        )
    elapsed = time.perf_counter() - t0
    return elapsed, len(created)


def run_benchmark_surface(processor_2d: MPAS2DProcessor, 
                          out_dir: str, 
                          use_parallel: bool, 
                          n_workers: int) -> tuple[float, int]:
    """
    This function benchmarks the batch surface map creation. It creates a directory for surface plots, runs the batch plotting function (either in parallel or serial), and returns the elapsed time and number of files created. The function takes in the MPAS2DProcessor with loaded data, the output directory path, a boolean indicating whether to use parallel processing, and the number of worker processes to use if parallel processing is enabled. 

    Parameters:
        processor_2d: An instance of MPAS2DProcessor with loaded 2D diagnostic data.
        out_dir: The base output directory where the 'surface' subdirectory will be created for storing the generated plots.
        use_parallel: A boolean flag indicating whether to use parallel processing for generating the plots.
        n_workers: The number of worker processes to use if parallel processing is enabled. This should typically be set to the number of available CPU cores or the number of MPI ranks.

    Returns:
        tuple[float, int]: A tuple containing the elapsed time in seconds for the batch plotting operation and the number of plot files created. The elapsed time is measured from the start of the plotting function to its completion, and the number of files is determined by counting the created plot files in the output directory.
    """
    plot_dir = os.path.join(out_dir, 'surface')
    os.makedirs(plot_dir, exist_ok=True)

    t0 = time.perf_counter()
    if use_parallel:
        created = ParallelSurfaceProcessor.create_batch_surface_maps_parallel(
            processor_2d, plot_dir,
            GeographicBounds(SPATIAL_BOUNDS['lon_min'], SPATIAL_BOUNDS['lon_max'],
                             SPATIAL_BOUNDS['lat_min'], SPATIAL_BOUNDS['lat_max']),
            var_name='t2m',
            grid_resolution=WIND_CONFIG['grid_resolution'],
            n_processes=n_workers,
            style=SurfaceBatchStyle(plot_type='scatter'),
        ) or []
    else:
        plotter = MPASSurfacePlotter(figsize=(12, 12), dpi=100)
        created = plotter.create_batch_surface_maps(
            processor_2d, plot_dir,
            GeographicBounds(SPATIAL_BOUNDS['lon_min'], SPATIAL_BOUNDS['lon_max'],
                             SPATIAL_BOUNDS['lat_min'], SPATIAL_BOUNDS['lat_max']),
            var_name='t2m',
            plot_type='scatter',
            grid_resolution=WIND_CONFIG['grid_resolution'],
        )
    elapsed = time.perf_counter() - t0
    return elapsed, len(created)


def run_benchmark_wind(processor_2d: MPAS2DProcessor, 
                       out_dir: str, 
                       use_parallel: bool, 
                       n_workers: int) -> tuple[float, int]:
    """
    This function benchmarks the batch wind map creation. It creates a directory for wind plots, runs the batch plotting function (either in parallel or serial), and returns the elapsed time and number of files created. The function takes in the MPAS2DProcessor with loaded data, the output directory path, a boolean indicating whether to use parallel processing, and the number of worker processes to use if parallel processing is enabled. 

    Parameters:
        processor_2d: An instance of MPAS2DProcessor with loaded 2D diagnostic data.
        out_dir: The base output directory where the 'wind' subdirectory will be created for storing the generated plots.
        use_parallel: A boolean flag indicating whether to use parallel processing for generating the plots.
        n_workers: The number of worker processes to use if parallel processing is enabled. This should typically be set to the number of available CPU cores or the number of MPI ranks.

    Returns:
        tuple[float, int]: A tuple containing the elapsed time in seconds for the batch plotting operation and the number of plot files created. The elapsed time is measured from the start of the plotting function to its completion, and the number of files is determined by counting the created plot files in the output directory.
    """
    cfg = WIND_CONFIG
    plot_dir = os.path.join(out_dir, 'wind')
    os.makedirs(plot_dir, exist_ok=True)

    t0 = time.perf_counter()
    if use_parallel:
        created = ParallelWindProcessor.create_batch_wind_plots_parallel(
            processor_2d, plot_dir,
            GeographicBounds(SPATIAL_BOUNDS['lon_min'], SPATIAL_BOUNDS['lon_max'],
                             SPATIAL_BOUNDS['lat_min'], SPATIAL_BOUNDS['lat_max']),
            u_variable=cfg['u_variable'],
            v_variable=cfg['v_variable'],
            n_processes=n_workers,
            style=WindBatchStyle(
                plot_type=cfg['plot_type'],
                subsample=cfg['subsample'],
                grid_resolution=cfg['grid_resolution'],
            ),
        ) or []
    else:
        plotter = MPASWindPlotter(figsize=(12, 12), dpi=100)
        created = plotter.create_batch_wind_plots(
            processor_2d, plot_dir,
            GeographicBounds(SPATIAL_BOUNDS['lon_min'], SPATIAL_BOUNDS['lon_max'],
                             SPATIAL_BOUNDS['lat_min'], SPATIAL_BOUNDS['lat_max']),
            u_variable=cfg['u_variable'],
            v_variable=cfg['v_variable'],
            plot_type=cfg['plot_type'],
            subsample=cfg['subsample'],
            grid_resolution=cfg['grid_resolution'],
        )
    elapsed = time.perf_counter() - t0
    return elapsed, len(created)


def run_benchmark_cross_section(processor_3d: MPAS3DProcessor, 
                                out_dir: str, 
                                use_parallel: bool, 
                                n_workers: int) -> tuple[float, int]:
    """
    This function benchmarks the batch vertical cross-section plot creation. It creates a directory for cross-section plots, runs the batch plotting function (either in parallel or serial), and returns the elapsed time and number of files created. The function takes in the MPAS3DProcessor with loaded data, the output directory path, a boolean indicating whether to use parallel processing, and the number of worker processes to use if parallel processing is enabled. 

    Parameters:
        processor_3d: An instance of MPAS3DProcessor with loaded 3D diagnostic data.
        out_dir: The base output directory where the 'cross_section' subdirectory will be created for storing the generated plots.
        use_parallel: A boolean flag indicating whether to use parallel processing for generating the plots.
        n_workers: The number of worker processes to use if parallel processing is enabled. This should typically be set to the number of available CPU cores or the number of MPI ranks.

    Returns:
        tuple[float, int]: A tuple containing the elapsed time in seconds for the batch plotting operation and the number of plot files created. The elapsed time is measured from the start of the plotting function to its completion, and the number of files is determined by counting the created plot files in the output directory.
    """
    cfg = CROSS_SECTION_CONFIG
    plot_dir = os.path.join(out_dir, 'cross_section')
    os.makedirs(plot_dir, exist_ok=True)

    t0 = time.perf_counter()
    if use_parallel:
        created = ParallelCrossSectionProcessor.create_batch_cross_section_plots_parallel(
            processor_3d,
            var_name=cfg['variable'],
            start_point=cfg['start_point'],
            end_point=cfg['end_point'],
            output_dir=plot_dir,
            vertical_coord=cfg['vertical_coord'],
            num_points=cfg['num_points'],
            n_processes=n_workers,
        ) or []
    else:
        plotter = MPASVerticalCrossSectionPlotter(figsize=(14, 12), dpi=100)
        created = plotter.create_batch_cross_section_plots(
            processor_3d, plot_dir,
            var_name=cfg['variable'],
            start_point=cfg['start_point'],
            end_point=cfg['end_point'],
            vertical_coord=cfg['vertical_coord'],
            num_points=cfg['num_points'],
        )
    elapsed = time.perf_counter() - t0
    return elapsed, len(created)


def run_benchmark_skewt(processor_3d: MPAS3DProcessor,
                        out_dir: str,
                        use_parallel: bool,
                        n_workers: int) -> tuple[float, int]:
    """
    This function benchmarks the batch skew-T plot creation. It creates a directory for skew-T plots, runs the batch plotting function (either in parallel or serial), and returns the elapsed time and number of files created. The function takes in the MPAS3DProcessor with loaded data, the output directory path, a boolean indicating whether to use parallel processing, and the number of worker processes to use if parallel processing is enabled. In parallel mode it distributes the per-timestep sounding extraction, thermodynamic index computation, and plotting across MPI ranks or worker processes via ParallelSkewTProcessor; in serial mode it iterates over timesteps in a single process.

    Parameters:
        processor_3d: An instance of MPAS3DProcessor with loaded 3D diagnostic data.
        out_dir: The base output directory where the 'skewt' subdirectory will be created for storing the generated plots.
        use_parallel: A boolean flag indicating whether to use parallel processing for generating the plots.
        n_workers: The number of worker processes to use if parallel processing is enabled. This should typically be set to the number of available CPU cores or the number of MPI ranks.

    Returns:
        tuple[float, int]: A tuple containing the elapsed time in seconds for the batch plotting operation and the number of plot files created. The elapsed time is measured from the start of the plotting function to its completion, and the number of files is determined by counting the created plot files in the output directory.
    """
    cfg = SKEWT_CONFIG
    plot_dir = os.path.join(out_dir, 'skewt')
    os.makedirs(plot_dir, exist_ok=True)

    t0 = time.perf_counter()
    if use_parallel:
        created = ParallelSkewTProcessor.create_batch_skewt_plots_parallel(
            processor_3d, plot_dir,
            lon=cfg['lon'],
            lat=cfg['lat'],
            n_processes=n_workers,
        ) or []
    else:
        diag = SoundingDiagnostics(verbose=False)
        plotter = MPASSkewTPlotter(figsize=(9, 12), dpi=100, verbose=False)

        time_dim = 'Time' if 'Time' in processor_3d.dataset.dims else 'time'
        n_times = processor_3d.dataset.sizes.get(time_dim, 1)

        created = []
        for t_idx in range(n_times):
            profile = diag.extract_sounding_profile(
                processor_3d, cfg['lon'], cfg['lat'], time_index=t_idx,
            )
            indices = diag.compute_thermodynamic_indices(
                profile['pressure'], profile['temperature'], profile['dewpoint'],
                u_wind_kt=profile.get('u_wind'),
                v_wind_kt=profile.get('v_wind'),
                height_m=profile.get('height'),
            )

            stn_lon = profile['station_lon']
            stn_lat = profile['station_lat']
            lon_tag = f"{abs(stn_lon):.2f}{'W' if stn_lon < 0 else 'E'}"
            lat_tag = f"{abs(stn_lat):.2f}{'S' if stn_lat < 0 else 'N'}"

            save_path = os.path.join(
                plot_dir,
                f"mpas_skewt_{lon_tag.replace('.', 'p')}_{lat_tag.replace('.', 'p')}_t{t_idx:03d}",
            )
            plotter.create_skewt_diagram(
                pressure=profile['pressure'],
                temperature=profile['temperature'],
                dewpoint=profile['dewpoint'],
                u_wind=profile['u_wind'],
                v_wind=profile['v_wind'],
                title=f"Skew-T | {lon_tag}, {lat_tag} | Time idx {t_idx}",
                indices=indices,
                show_parcel=False,
                save_path=save_path,
            )
            plotter.close_plot()
            created.append(save_path)

    elapsed = time.perf_counter() - t0
    return elapsed, len(created)


def write_csv(results: list[dict], 
              csv_path: str) -> None:
    """
    This function writes the benchmark results to a CSV file. It takes a list of dictionaries, where each dictionary contains the results of a single benchmark (including experiment name, category, number of MPI ranks, elapsed time in seconds, number of files created, and timestamp), and the path to the CSV file where the results should be saved. The function uses Python's built-in csv module to write the data to the specified CSV file, including a header row with the field names. 

    Parameters:
        results: A list of dictionaries containing benchmark results.
        csv_path: The path to the CSV file where the results will be written.

    Returns:
        None
    """
    fieldnames = ['experiment', 'category', 'mpi_ranks', 'backend', 'workers', 'trial',
                  'elapsed_s', 'elapsed_min_s', 'elapsed_mean_s', 'n_files',
                  'n_nodes', 'hostname', 'timestamp']
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: list[dict], 
                  csv_path: str) -> None:
    """
    This function prints a summary of the benchmark results to the console. It takes a list of dictionaries containing the benchmark results and the path to the CSV file where the results are saved. The function formats and prints a table summarizing the experiment name, category, number of MPI ranks, elapsed time in seconds, and number of files created for each benchmark. It also prints a separator line before and after the table for better readability, and at the end, it prints the path to the CSV file where the results are stored. 

    Parameters:
        results: A list of dictionaries containing benchmark results.
        csv_path: The path to the CSV file where the results are saved.

    Returns:
        None
    """
    hdr = f"{'Experiment':<12} {'Category':<20} {'Backend':>16} {'Width':>6} {'Time (s)':>10} {'Files':>6}"
    sep = '-' * len(hdr)
    print(f"\n{sep}\n  Benchmark Summary\n{sep}")
    print(hdr)
    print(sep)
    for r in results:
        print(
            f"{r['experiment']:<12} {r['category']:<20} {r['backend']:>16} {r['workers']:>6} "
            f"{r['elapsed_s']:>10.2f} {r['n_files']:>6}"
        )
    print(sep)
    print(f"Results saved to: {csv_path}\n")


def _make_result(exp_name: str,
                 category: str,
                 elapsed: float,
                 n_files: int,
                 timestamp: str,
                 elapsed_min: float | None = None,
                 elapsed_mean: float | None = None,
                 trial: int = 1) -> dict:
    """
    This function creates a result record dictionary for a single benchmark. It takes the experiment name, benchmark category, elapsed time in seconds (the maximum across ranks, i.e. the makespan), number of files created, and a timestamp string as input parameters, and returns a dictionary containing these values along with the number of MPI ranks used (from the global SIZE variable), the minimum and mean elapsed time across ranks (which together with the maximum quantify load imbalance), the trial index, and the node count and hostname identifying where the run executed. The elapsed times are rounded to 4 decimal places for consistency. This function is intended to standardize the format of the benchmark results for later aggregation and CSV writing.

    Parameters:
        exp_name: The name of the experiment this result belongs to.
        category: The benchmark category (e.g. 'data_load_2d', 'precipitation').
        elapsed: The elapsed time in seconds (max across ranks) for the benchmarked operation.
        n_files: The number of plot files created by the operation.
        timestamp: The run timestamp string shared across all records of a run.
        elapsed_min: The minimum elapsed time across ranks, or None to reuse elapsed.
        elapsed_mean: The mean elapsed time across ranks, or None to reuse elapsed.
        trial: The 1-based trial index when a category is benchmarked repeatedly.

    Returns:
        dict: A dictionary matching the CSV schema in write_csv.
    """
    return {
        'experiment': exp_name,
        'category': category,
        'mpi_ranks': SIZE,
        'backend': BACKEND,
        'workers': PARALLEL_WIDTH,
        'trial': trial,
        'elapsed_s': round(elapsed, 4),
        'elapsed_min_s': round(elapsed if elapsed_min is None else elapsed_min, 4),
        'elapsed_mean_s': round(elapsed if elapsed_mean is None else elapsed_mean, 4),
        'n_files': n_files,
        'n_nodes': N_NODES,
        'hostname': HOSTNAME,
        'timestamp': timestamp,
    }


def print_banner(mode_label: str,
                 experiment_names: list[str]) -> None:
    """
    This function prints a banner to the console at the start of the benchmark run. The banner includes a title for the benchmark, the resolved execution mode (serial, multiprocessing, or MPI) together with the parallel width, the number of MPI ranks in the job, and the list of experiments that will be run. The banner is formatted with separator lines for better visibility. This function is intended to provide a clear starting point for the benchmark output and to summarise the key parameters of the run.

    Parameters:
        mode_label: Human-readable description of the resolved execution backend and width.
        experiment_names: Names of the experiments selected for this run.

    Returns:
        None
    """
    print("=" * 60)
    print("  MPASdiag Batch Plotter Benchmark")
    print(f"  Execution mode: {mode_label}")
    print(f"  MPI ranks in job: {SIZE}")
    print(f"  Nodes: {N_NODES} (rank 0 on {HOSTNAME})")
    print(f"  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
          f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}")
    print(f"  Experiments: {experiment_names}")
    print("=" * 60)


def load_experiment_data(paths: dict) -> tuple[MPAS2DProcessor, MPAS3DProcessor, float, float]:
    """
    This function loads the 2D and 3D data for a single experiment using the provided paths. It creates instances of MPAS2DProcessor and MPAS3DProcessor, loads the respective data from the specified directories, and measures the time taken for each loading operation. The function prints status messages to the console indicating which data is being loaded, and it returns the loaded processors along with the elapsed times for loading the 2D and 3D data. The loading operations are performed sequentially, and the timing is done using time.perf_counter() for high-resolution timing. 

    Parameters:
        paths: A dictionary containing the keys 'grid_file', 'diag_dir', and 'mpasout_dir' with the respective paths for the grid file, 2D diagnostic data directory, and 3D model output directory.

    Returns:
        tuple[MPAS2DProcessor, MPAS3DProcessor, float, float]: A tuple containing the loaded MPAS2DProcessor, the loaded MPAS3DProcessor, the elapsed time in seconds for loading the 2D data, and the elapsed time in seconds for loading the 3D data.
    """
    if RANK == 0:
        print(f"  Loading 2D diagnostic data from {paths['diag_dir']} ...")

    t_load = time.perf_counter()
    processor_2d = MPAS2DProcessor(grid_file=paths['grid_file'], verbose=(RANK == 0))
    processor_2d.load_2d_data(paths['diag_dir'])
    load_2d_time = time.perf_counter() - t_load

    if RANK == 0:
        print(f"  Loading 3D model output from {paths['mpasout_dir']} ...")

    t_load = time.perf_counter()
    processor_3d = MPAS3DProcessor(grid_file=paths['grid_file'], verbose=(RANK == 0))
    processor_3d.load_3d_data(paths['mpasout_dir'])
    load_3d_time = time.perf_counter() - t_load

    return processor_2d, processor_3d, load_2d_time, load_3d_time


def build_benchmarks(processor_2d: MPAS2DProcessor,
                     processor_3d: MPAS3DProcessor,
                     exp_out: str,
                     use_parallel: bool,
                     n_workers: int | None) -> list[tuple[str, Callable[[], tuple[float, int]]]]:
    """
    This function builds the list of benchmarks to run for a single experiment. It takes the loaded 2D and 3D processors, the base output directory for the experiment, a boolean indicating whether to use parallel processing, and the number of worker processes to use if parallel processing is enabled. The function returns a list of tuples, where each tuple contains a benchmark category name (e.g. 'precipitation') and a zero-argument callable that executes the corresponding benchmark function and returns a tuple of (elapsed time in seconds, number of files created). The benchmark functions are wrapped in lambda functions to defer their execution until they are called in the main experiment loop. All five benchmark categories, including skew-T, run in parallel across ranks (MPI) or worker processes (multiprocessing) when parallel processing is enabled.

    Parameters:
        processor_2d: The loaded 2D processor used by the precipitation, surface, and wind benchmarks.
        processor_3d: The loaded 3D processor used by the cross-section and skew-T benchmarks.
        exp_out: The base output directory for the experiment's plots.
        use_parallel: Whether the batch plotting functions should run in parallel.
        n_workers: The number of worker processes to use when parallel, or None.

    Returns:
        list[tuple[str, Callable[[], tuple[float, int]]]]: The (name, callable) pairs to execute in order.
    """
    workers = n_workers if n_workers is not None else 1
    return [
        ('precipitation', lambda: run_benchmark_precipitation(processor_2d, exp_out, use_parallel, workers)),
        ('surface', lambda: run_benchmark_surface(processor_2d, exp_out, use_parallel, workers)),
        ('wind', lambda: run_benchmark_wind(processor_2d, exp_out, use_parallel, workers)),
        ('cross_section', lambda: run_benchmark_cross_section(processor_3d, exp_out, use_parallel, workers)),
        ('skewt', lambda: run_benchmark_skewt(processor_3d, exp_out, use_parallel, workers)),
    ]


def run_single_benchmark(plotter_name: str,
                         bench_fn: Callable[[], tuple[float, int]],
                         exp_name: str,
                         timestamp: str,
                         trial: int = 1) -> dict | None:
    """
    This function runs a single benchmark and gathers the results across MPI ranks. It takes the benchmark category name, a zero-argument callable that executes the benchmark and returns (elapsed time, number of files), the experiment name, and a timestamp string. The function prints a status message indicating which benchmark is running, synchronises the ranks with a barrier, executes the benchmark function to get the timing and file count, and then gathers these results from all ranks to rank 0. On rank 0, it computes the maximum elapsed time across ranks (as the effective runtime) and the total number of files created, prints the results for this benchmark, and creates a result record dictionary using the _make_result helper function. The function returns this result record on rank 0, while other ranks return None. 

    Parameters:
        plotter_name: The benchmark category name (e.g. 'precipitation').
        bench_fn: A zero-argument callable returning (elapsed, n_files).
        exp_name: The name of the experiment the benchmark belongs to.
        timestamp: The run timestamp string shared across all records of a run.

    Returns:
        dict | None: The aggregated result record on rank 0, otherwise None.
    """
    if RANK == 0:
        print(f"  Benchmarking {plotter_name} ...")

    if COMM is not None:
        COMM.Barrier()

    elapsed, n_files = bench_fn()

    if COMM is not None:
        timings = COMM.gather(elapsed, root=0) or [elapsed]
        file_counts = COMM.gather(n_files, root=0) or [n_files]
    else:
        timings = [elapsed]
        file_counts = [n_files]

    result = None

    if RANK == 0:
        max_time = max(timings)
        min_time = min(timings)
        mean_time = sum(timings) / len(timings)
        total_files = sum(file_counts)
        print(f"    -> {plotter_name}: {max_time:.2f}s, {total_files} files")

        if total_files == 0:
            print(
                f"    !! WARNING: {plotter_name} produced 0 files -- every task "
                "likely failed; this timing measures failure handling, not work. "
                "Check the error log above."
            )

        result = _make_result(
            exp_name, plotter_name, max_time, total_files, timestamp,
            elapsed_min=min_time, elapsed_mean=mean_time, trial=trial,
        )

    gc.collect()
    return result


def run_experiment(exp_name: str,
                   paths: dict,
                   use_parallel: bool,
                   n_workers: int | None,
                   timestamp: str,
                   trials: int = 1) -> list[dict]:
    """
    This function runs a single experiment, which includes loading the data, running all benchmarks for that experiment, and collecting the results. It takes the experiment name, a dictionary of paths for loading the data, a boolean indicating whether to use parallel processing, the number of worker processes to use if parallel processing is enabled, and a timestamp string. The function first prints the experiment name on rank 0, creates an output directory for the experiment, and synchronises the ranks. It then loads the 2D and 3D data using the load_experiment_data helper function, which returns the loaded processors and the time taken for loading. The loading times are recorded as benchmark results. Next, it builds the list of benchmarks to run using the build_benchmarks helper function, which returns a list of (name, callable) pairs. The function then iterates over these benchmarks, running each one with run_single_benchmark to get the result record, which is collected into a list of results. Finally, it cleans up by deleting the processors and calling garbage collection before returning the list of results for this experiment.

    Parameters:
        exp_name: The name of the experiment to run.
        paths: A mapping with the keys 'grid_file', 'diag_dir', and 'mpasout_dir'.
        use_parallel: Whether the batch plotting functions should run in parallel.
        n_workers: The number of worker processes to use when parallel, or None.
        timestamp: The run timestamp string shared across all records of a run.

    Returns:
        list[dict]: The benchmark result records produced for this experiment (empty on ranks other than rank 0).
    """
    results: list[dict] = []

    if RANK == 0:
        print(f"\n>>> Experiment: {exp_name}")

    exp_out = str(BENCHMARK_DIR / exp_name)

    if RANK == 0:
        os.makedirs(exp_out, exist_ok=True)

    if COMM is not None:
        COMM.Barrier()

    processor_2d, processor_3d, load_2d_time, load_3d_time = load_experiment_data(paths)

    if COMM is not None:
        load_2d_times = COMM.gather(load_2d_time, root=0) or [load_2d_time]
        load_3d_times = COMM.gather(load_3d_time, root=0) or [load_3d_time]
    else:
        load_2d_times = [load_2d_time]
        load_3d_times = [load_3d_time]

    if RANK == 0:
        print(f"  2D load: {max(load_2d_times):.2f}s | 3D load: {max(load_3d_times):.2f}s")
        for category, load_times in (('data_load_2d', load_2d_times),
                                     ('data_load_3d', load_3d_times)):
            results.append(_make_result(
                exp_name, category, max(load_times), 0, timestamp,
                elapsed_min=min(load_times),
                elapsed_mean=sum(load_times) / len(load_times),
            ))

    benchmarks = build_benchmarks(processor_2d, processor_3d, exp_out, use_parallel, n_workers)

    for trial in range(1, trials + 1):
        if RANK == 0 and trials > 1:
            print(f"  --- Trial {trial}/{trials} ---")
        for plotter_name, bench_fn in benchmarks:
            result = run_single_benchmark(plotter_name, bench_fn, exp_name, timestamp, trial)
            if result is not None:
                results.append(result)

    del processor_2d, processor_3d
    gc.collect()
    return results


def parse_args() -> argparse.Namespace:
    """
    This function parses the command-line options that control the execution backend and the scope of the benchmark run. It exposes a worker count for the shared-memory multiprocessing backend, a flag to force a single-process serial baseline, and an experiment filter to restrict the run to a subset of the configured resolutions. Unknown arguments are tolerated so the script stays robust when launched under an MPI process manager that may append its own options.

    Parameters:
        None

    Returns:
        argparse.Namespace: Parsed arguments exposing 'workers', 'serial', and 'experiments'.
    """
    parser = argparse.ArgumentParser(
        description="MPASdiag batch-plotter benchmark (serial / multiprocessing / MPI).",
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help="Worker processes for the multiprocessing backend when NOT launched via "
             "mpiexec (default: CPU count - 1). Ignored under MPI.",
    )
    parser.add_argument(
        '--serial', action='store_true',
        help="Force a single-process serial run (parallelism disabled).",
    )
    parser.add_argument(
        '--experiments', nargs='+', metavar='NAME', default=None,
        choices=list(EXPERIMENTS.keys()),
        help="Subset of experiments to run (default: all).",
    )
    parser.add_argument(
        '--trials', type=int, default=1,
        help="Repetitions of each plotting category per experiment; report the "
             "spread across trials to separate real changes from filesystem "
             "noise (default: 1).",
    )
    args, _ = parser.parse_known_args()
    return args


def resolve_execution_mode(size: int,
                           serial: bool,
                           workers: int | None) -> tuple[bool, int | None, str, int]:
    """
    This function decides the execution backend from the launch context and the command-line flags. The benchmark can run three ways: distributed MPI (selected when the script is launched with an MPI process manager so that the world size exceeds one), shared-memory multiprocessing (a Pool of worker processes when run as plain Python), or a single-process serial baseline. MPI takes precedence because it is chosen by the launcher rather than by a flag, followed by an explicit serial request, and finally the multiprocessing default sized to the available CPUs.

    Parameters:
        size: MPI world size (1 when not launched via mpiexec/mpirun).
        serial: Whether the user forced a serial run with --serial.
        workers: Requested multiprocessing worker count, or None for the default.

    Returns:
        tuple[bool, int | None, str, int]: (use_parallel, n_workers, backend, parallel_width).
    """
    if size > 1:
        return True, size, 'mpi', size
    
    if serial:
        return False, None, 'serial', 1
    
    resolved = workers if workers is not None else max(1, cpu_count() - 1)

    if resolved <= 1:
        return False, None, 'serial', 1
    
    return True, resolved, 'multiprocessing', resolved


def select_experiments(names: list[str] | None) -> dict:
    """
    This function returns the subset of EXPERIMENTS requested on the command line, preserving the original configuration mapping when no filter is supplied. It allows quick, targeted benchmark runs against a single resolution without editing the module-level EXPERIMENTS dictionary.

    Parameters:
        names: Experiment names to keep, or None/empty to keep them all.

    Returns:
        dict: Mapping of the selected experiment names to their path configurations.
    """
    if not names:
        return EXPERIMENTS
    return {name: EXPERIMENTS[name] for name in names}


def main() -> None:
    """ Main function for the MPASdiag Batch Plotter Benchmark. """
    global BACKEND, PARALLEL_WIDTH

    args = parse_args()
    use_parallel, n_workers, BACKEND, PARALLEL_WIDTH = resolve_execution_mode(
        SIZE, args.serial, args.workers,
    )
    experiments = select_experiments(args.experiments)

    mode_label = {
        'mpi': f"MPI ({SIZE} ranks)",
        'multiprocessing': f"multiprocessing ({PARALLEL_WIDTH} workers)",
        'serial': "serial (1 process)",
    }[BACKEND]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results = []

    if RANK == 0:
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        print_banner(mode_label, list(experiments.keys()))
        if BACKEND == 'multiprocessing':
            print("  Note: the multiprocessing backend reloads data per worker "
                  "(esp. on macOS spawn).")
            print("        For the fastest run use: mpiexec -n N python benchmark.py\n")

    for exp_name, paths in experiments.items():
        all_results.extend(
            run_experiment(exp_name, paths, use_parallel, n_workers, timestamp,
                           trials=max(1, args.trials))
        )

    if RANK == 0:
        csv_path = BENCHMARK_DIR / f"benchmark_results_{timestamp}.csv"
        write_csv(all_results, str(csv_path))
        print_summary(all_results, str(csv_path))


if __name__ == '__main__':
    main()


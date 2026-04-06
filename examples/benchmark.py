#!/usr/bin/env python3

"""
MPASdiag Benchmark: Batch Plotter Runtime with MPI Support

This script benchmarks the runtime of MPASdiag's batch plotting functions for various plot types (precipitation, surface, wind, cross-section, skew-T) across different MPAS model resolutions. It supports both serial and parallel execution using MPI (via mpi4py). The results are saved to a CSV file and printed in a summary table.  

Usage: 
    python benchmark.py

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: March 2026
Version: 1.0.0
"""
# Load standard libraries
import gc
import os
import csv
import time
from pathlib import Path
from datetime import datetime

# Load relevant MPASdiag modules
from mpasdiag.visualization.wind import MPASWindPlotter
from mpasdiag.visualization.skewt import MPASSkewTPlotter
from mpasdiag.diagnostics.sounding import SoundingDiagnostics
from mpasdiag.visualization.surface import MPASSurfacePlotter
from mpasdiag.processing.processors_2d import MPAS2DProcessor
from mpasdiag.processing.processors_3d import MPAS3DProcessor
from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter
from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter

from mpasdiag.processing.parallel_wrappers import (
    ParallelPrecipitationProcessor,
    ParallelSurfaceProcessor,
    ParallelWindProcessor,
    ParallelCrossSectionProcessor,
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
            SPATIAL_BOUNDS['lon_min'], SPATIAL_BOUNDS['lon_max'],
            SPATIAL_BOUNDS['lat_min'], SPATIAL_BOUNDS['lat_max'],
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
            SPATIAL_BOUNDS['lon_min'], SPATIAL_BOUNDS['lon_max'],
            SPATIAL_BOUNDS['lat_min'], SPATIAL_BOUNDS['lat_max'],
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
            SPATIAL_BOUNDS['lon_min'], SPATIAL_BOUNDS['lon_max'],
            SPATIAL_BOUNDS['lat_min'], SPATIAL_BOUNDS['lat_max'],
            var_name='t2m',
            plot_type='scatter',
            grid_resolution=WIND_CONFIG['grid_resolution'],
            n_processes=n_workers,
        ) or []
    else:
        plotter = MPASSurfacePlotter(figsize=(12, 12), dpi=100)
        created = plotter.create_batch_surface_maps(
            processor_2d, plot_dir,
            SPATIAL_BOUNDS['lon_min'], SPATIAL_BOUNDS['lon_max'],
            SPATIAL_BOUNDS['lat_min'], SPATIAL_BOUNDS['lat_max'],
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
            SPATIAL_BOUNDS['lon_min'], SPATIAL_BOUNDS['lon_max'],
            SPATIAL_BOUNDS['lat_min'], SPATIAL_BOUNDS['lat_max'],
            u_variable=cfg['u_variable'],
            v_variable=cfg['v_variable'],
            plot_type=cfg['plot_type'],
            subsample=cfg['subsample'],
            grid_resolution=cfg['grid_resolution'],
            n_processes=n_workers,
        ) or []
    else:
        plotter = MPASWindPlotter(figsize=(12, 12), dpi=100)
        created = plotter.create_batch_wind_plots(
            processor_2d, plot_dir,
            SPATIAL_BOUNDS['lon_min'], SPATIAL_BOUNDS['lon_max'],
            SPATIAL_BOUNDS['lat_min'], SPATIAL_BOUNDS['lat_max'],
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
                        out_dir: str) -> tuple[float, int]:
    """
    This function benchmarks the batch skew-T plot creation. It creates a directory for skew-T plots, runs the batch plotting function (serial only), and returns the elapsed time and number of files created. The function takes in the MPAS3DProcessor with loaded data and the output directory path. The skew-T plotting function is currently implemented in serial due to the complexity of the plotting logic and the need to compute thermodynamic indices for each profile, which may not be easily parallelizable without significant refactoring. The function extracts sounding profiles at specified locations and time indices, computes thermodynamic indices, and generates skew-T diagrams for each profile, saving them to the output directory. The elapsed time is measured from the start of the plotting function to its completion, and the number of files is determined by counting the created plot files in the output directory. 

    Parameters:
        processor_3d: An instance of MPAS3DProcessor with loaded 3D diagnostic data.
        out_dir: The base output directory where the 'skewt' subdirectory will be created for storing the generated plots.

    Returns:
        tuple[float, int]: A tuple containing the elapsed time in seconds for the batch plotting operation and the number of plot files created. The elapsed time is measured from the start of the plotting function to its completion, and the number of files is determined by counting the created plot files in the output directory. 
    """
    cfg = SKEWT_CONFIG
    plot_dir = os.path.join(out_dir, 'skewt')
    os.makedirs(plot_dir, exist_ok=True)

    diag = SoundingDiagnostics(verbose=False)
    plotter = MPASSkewTPlotter(figsize=(9, 12), dpi=100, verbose=False)

    time_dim = 'Time' if 'Time' in processor_3d.dataset.dims else 'time'
    n_times = processor_3d.dataset.sizes.get(time_dim, 1)

    created_files = []
    t0 = time.perf_counter()
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
        created_files.append(save_path)

    elapsed = time.perf_counter() - t0
    return elapsed, len(created_files)


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
    fieldnames = ['experiment', 'category', 'mpi_ranks', 'elapsed_s', 'n_files', 'timestamp']
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
    hdr = f"{'Experiment':<12} {'Category':<20} {'Ranks':>5} {'Time (s)':>10} {'Files':>6}"
    sep = '-' * len(hdr)
    print(f"\n{sep}\n  Benchmark Summary\n{sep}")
    print(hdr)
    print(sep)
    for r in results:
        print(
            f"{r['experiment']:<12} {r['category']:<20} {r['mpi_ranks']:>5} "
            f"{r['elapsed_s']:>10.2f} {r['n_files']:>6}"
        )
    print(sep)
    print(f"Results saved to: {csv_path}\n")


def main() -> None:
    """ Main function for the MPASdiag Batch Plotter Benchmark. """
    use_parallel = SIZE > 1
    n_workers = SIZE if use_parallel else None
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results = []

    if RANK == 0:
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        print("=" * 60)
        print("  MPASdiag Batch Plotter Benchmark")
        print(f"  MPI ranks: {SIZE}")
        print(f"  Experiments: {list(EXPERIMENTS.keys())}")
        print("=" * 60)

    for exp_name, paths in EXPERIMENTS.items():
        if RANK == 0:
            print(f"\n>>> Experiment: {exp_name}")

        exp_out = str(BENCHMARK_DIR / exp_name)
        if RANK == 0:
            os.makedirs(exp_out, exist_ok=True)

        # Synchronise so dirs exist before other ranks proceed
        if COMM is not None:
            COMM.Barrier()

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

        if RANK == 0:
            print(f"  2D load: {load_2d_time:.2f}s | 3D load: {load_3d_time:.2f}s")
            all_results.append({
                'experiment': exp_name,
                'category': 'data_load_2d',
                'mpi_ranks': SIZE,
                'elapsed_s': round(load_2d_time, 4),
                'n_files': 0,
                'timestamp': timestamp,
            })
            all_results.append({
                'experiment': exp_name,
                'category': 'data_load_3d',
                'mpi_ranks': SIZE,
                'elapsed_s': round(load_3d_time, 4),
                'n_files': 0,
                'timestamp': timestamp,
            })

        benchmarks = [
            ('precipitation', lambda: run_benchmark_precipitation(processor_2d, exp_out, use_parallel, n_workers=n_workers if n_workers is not None else 1)),
            ('surface',       lambda: run_benchmark_surface(processor_2d, exp_out, use_parallel, n_workers=n_workers if n_workers is not None else 1)),
            ('wind',          lambda: run_benchmark_wind(processor_2d, exp_out, use_parallel, n_workers=n_workers if n_workers is not None else 1)),
            ('cross_section', lambda: run_benchmark_cross_section(processor_3d, exp_out, use_parallel, n_workers=n_workers if n_workers is not None else 1)),
            ('skewt',         lambda: run_benchmark_skewt(processor_3d, exp_out) if RANK == 0 else (0.0, 0)),
        ]

        for plotter_name, bench_fn in benchmarks:
            if RANK == 0:
                print(f"  Benchmarking {plotter_name} ...")

            # Synchronise ranks before each plotter benchmark
            if COMM is not None:
                COMM.Barrier()

            elapsed, n_files = bench_fn()

            # Gather timing from all ranks on rank 0
            if COMM is not None:
                timings = COMM.gather(elapsed, root=0) or [elapsed]
                file_counts = COMM.gather(n_files, root=0) or [n_files]
            else:
                timings = [elapsed]
                file_counts = [n_files]

            if RANK == 0:
                max_time = max(timings)
                total_files = sum(file_counts)
                print(f"    -> {plotter_name}: {max_time:.2f}s, {total_files} files")
                all_results.append({
                    'experiment': exp_name,
                    'category': plotter_name,
                    'mpi_ranks': SIZE,
                    'elapsed_s': round(max_time, 4),
                    'n_files': total_files,
                    'timestamp': timestamp,
                })

            gc.collect()

        del processor_2d, processor_3d
        gc.collect()

    if RANK == 0:
        csv_path = BENCHMARK_DIR / f"benchmark_results_{timestamp}.csv"
        write_csv(all_results, str(csv_path))
        print_summary(all_results, str(csv_path))


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Benchmark Analyzer: Scaling, Speedup, and Efficiency Report

This script merges one or more benchmark_results_*.csv files produced by benchmark.py and reports how each plotting category scales with worker count for every experiment. For each (experiment, category) it prints the runtime at each parallel width, the speedup relative to the smallest width present (or a serial run when one is included), the parallel efficiency, and a flag for anti-scaling configurations where adding workers made the run slower. It also reports rank-level load imbalance when the CSVs carry the elapsed_min_s/elapsed_mean_s columns, and aggregates repeated trials by their median so filesystem noise does not masquerade as a scaling trend.

Usage:
    python analyze_benchmarks.py                          # all benchmark_results_*.csv in CWD
    python analyze_benchmarks.py path/to/results/*.csv    # explicit files

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: June 2026
Version: 1.0.0
"""
import csv
import glob
import sys
from collections import defaultdict
from statistics import median


def load_results(paths: list[str]) -> list[dict]:
    """
    This function reads benchmark CSV files and parses each row, converting 'workers', 'trial' to int and elapsed fields to float. It also handles optional columns for rank imbalance and file counts, defaulting to 1.0 imbalance and 0 files when not present. 

    Parameters:
        paths: CSV file paths produced by benchmark.py.

    Returns:
        list[dict]: All rows with 'workers', 'trial' as int and elapsed fields as float.
    """
    rows = []
    for path in paths:
        with open(path, newline='') as fh:
            for row in csv.DictReader(fh):
                row['workers'] = int(row.get('workers') or row.get('mpi_ranks') or 1)
                row['trial'] = int(row.get('trial') or 1)
                row['elapsed_s'] = float(row['elapsed_s'])
                row['elapsed_min_s'] = float(row.get('elapsed_min_s') or row['elapsed_s'])
                row['elapsed_mean_s'] = float(row.get('elapsed_mean_s') or row['elapsed_s'])
                row['n_files'] = int(row.get('n_files') or 0)
                rows.append(row)
    return rows


def aggregate_trials(rows: list[dict]) -> dict:
    """
    This function groups rows by (experiment, category) and worker count, then aggregates repeated trials by their median elapsed time and imbalance. The output is a nested dictionary mapping (experiment, category) -> {workers: record} where each record contains the median 'elapsed', 'imbalance', 'n_trials', and 'n_files' for that configuration. 

    Parameters:
        rows: Parsed benchmark rows from load_results.

    Returns:
        dict: Mapping (experiment, category) -> {workers: record} with 'elapsed', 'imbalance', 'n_trials', 'n_files'.
    """
    grouped: dict = defaultdict(lambda: defaultdict(list))

    for row in rows:
        grouped[(row['experiment'], row['category'])][row['workers']].append(row)

    aggregated: dict = {}

    for key, by_width in grouped.items():
        aggregated[key] = {}
        for width, recs in by_width.items():
            elapsed = median(r['elapsed_s'] for r in recs)
            imbalance = median(
                (r['elapsed_s'] / r['elapsed_mean_s']) if r['elapsed_mean_s'] > 0 else 1.0
                for r in recs
            )
            aggregated[key][width] = {
                'elapsed': elapsed,
                'imbalance': imbalance,
                'n_trials': len(recs),
                'n_files': max(r['n_files'] for r in recs),
            }
    return aggregated


def print_report(aggregated: dict) -> None:
    """
    This function takes the aggregated benchmark data and prints a report for each (experiment, category) showing how runtime scales with worker count. It calculates speedup relative to the smallest worker count, parallel efficiency, and flags configurations where adding workers made the run slower (anti-scaling). It also flags cases where the number of workers exceeds the number of files (tasks) or when data loading is replicated across workers. 

    Parameters:
        aggregated: Output of aggregate_trials.

    Returns:
        None
    """
    for (experiment, category) in sorted(aggregated):
        by_width = aggregated[(experiment, category)]
        widths = sorted(by_width)
        base_width = widths[0]
        base = by_width[base_width]['elapsed']

        print(f"\n{experiment} / {category}  (baseline: {base_width} worker(s), {base:.2f}s)")
        print(f"  {'workers':>8} {'time (s)':>10} {'speedup':>8} {'efficiency':>11} "
              f"{'imbalance':>10} {'trials':>7}  flags")

        prev_elapsed = None

        for width in widths:
            rec = by_width[width]
            speedup = base / rec['elapsed'] if rec['elapsed'] > 0 else float('inf')
            ratio = width / base_width
            efficiency = speedup / ratio if ratio > 0 else 1.0

            flags = []
            is_data_load = 'data_load' in category
            ranks_exceed_tasks = (not is_data_load
                                  and rec['n_files'] > 0 and width >= rec['n_files'])

            if ranks_exceed_tasks:
                flags.append(f"RANKS>=TASKS({rec['n_files']})")
            elif is_data_load:
                flags.append('REPLICATED')
            elif prev_elapsed is not None and rec['elapsed'] > prev_elapsed * 1.05:
                flags.append('ANTI-SCALING')

            if rec['n_files'] == 0 and not is_data_load:
                flags.append('NO-FILES(failed?)')

            print(f"  {width:>8} {rec['elapsed']:>10.2f} {speedup:>7.2f}x {efficiency:>10.1%} "
                  f"{rec['imbalance']:>9.2f}x {rec['n_trials']:>7}  {' '.join(flags)}")
            prev_elapsed = rec['elapsed']


def main() -> None:
    """ Main entry point: resolve CSV paths from argv or the working directory and print the scaling report. """
    paths = sys.argv[1:] or sorted(glob.glob('benchmark_results_*.csv'))

    if not paths:
        sys.exit("No benchmark CSVs found. Pass paths or run from the results directory.")

    rows = load_results(paths)
    
    print(f"Loaded {len(rows)} rows from {len(paths)} file(s)")
    print_report(aggregate_trials(rows))


if __name__ == '__main__':
    main()

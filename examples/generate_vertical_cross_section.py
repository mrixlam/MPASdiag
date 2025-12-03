#!/usr/bin/env python3

"""
MPASdiag Example5: Vertical Cross-Section 

This example demonstrates how to create vertical cross-section plots of 3D atmospheric
variables from MPAS model output using the new MPASVerticalCrossSectionPlotter class.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Version: 1.0.0
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

package_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_dir))
OUTPUT_DIR = Path("testPlot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from mpasdiag.processing.processors_3d import MPAS3DProcessor
from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter


def create_zonal_wind_cross_section(
    grid_file: str = "../data/grids/x1.163842.init.nc",
    data_directory: str = "../data/u60k/",
) -> None:
    """
    Create and save vertical cross-section plots of the zonal wind component from MPAS output.
    The function locates a suitable zonal wind variable in the provided dataset, extracts a transect
    between two geographic points, and writes three cross-section plots (height, pressure, and
    model-level coordinates) to the output directory. Contour levels, colormap, and plotting
    options are chosen to highlight signed wind fields (e.g., diverging colormap and symmetric
    contour ranges).

    Parameters:
        grid_file (str): Path to the MPAS grid/static file containing mesh topology and spatial coordinates.
        data_directory (str): Directory containing MPAS 3D output files (e.g., mpasout*.nc).

    Returns:
        None: This function does not return a value. It saves generated PNG files to the configured output directory.
    """
    
    print("\n=== Advanced Cross-Section: Wind Speed ===")
    
    try:
        processor_3d = MPAS3DProcessor(grid_file, verbose=True)
        processor_3d.load_3d_data(data_directory)
        available_vars = processor_3d.get_available_3d_variables()

        if 'uReconstructZonal' in available_vars:
            u_var = 'uReconstructZonal'
        else:
            u_candidates = [v for v in available_vars if v.startswith('u')]
            u_var = u_candidates[0] if u_candidates else None

        if not u_var:
            print(f"Zonal wind variable not found in dataset. Available vars: {available_vars[:20]}")
            return

        start_point = (90.0, 15.1)
        end_point = (160.0, 15.1)

        cross_section_plotter = MPASVerticalCrossSectionPlotter(figsize=(10, 10))

        _, _ = cross_section_plotter.create_vertical_cross_section(
            mpas_3d_processor=processor_3d,
            var_name=u_var,
            start_point=start_point,
            end_point=end_point,
            time_index=0,
            vertical_coord='pressure',
            display_vertical='height',
            num_points=30,
            levels=np.arange(-30, 35, 5),
            colormap='RdBu_r',
            plot_type='filled_contour',
            extend='both',
            title=f"Vertical Cross-Section: {u_var} (height)",
            save_path=str(OUTPUT_DIR / f'mpasdiag_sample_plot_cross_section_{u_var.lower()}_height.png')
        )

        _, _ = cross_section_plotter.create_vertical_cross_section(
            mpas_3d_processor=processor_3d,
            var_name=u_var,
            start_point=start_point,
            end_point=end_point,
            time_index=0,
            vertical_coord='pressure',
            display_vertical='pressure',
            num_points=30,
            levels=np.arange(-30, 35, 5),
            colormap='RdBu_r',
            plot_type='filled_contour',
            extend='both',
            title=f"Vertical Cross-Section: {u_var} (pressure)",
            save_path=str(OUTPUT_DIR / f'mpasdiag_sample_plot_cross_section_{u_var.lower()}_pressure.png')
        )

        _, _ = cross_section_plotter.create_vertical_cross_section(
            mpas_3d_processor=processor_3d,
            var_name=u_var,
            start_point=start_point,
            end_point=end_point,
            time_index=0,
            vertical_coord='model_levels',
            display_vertical='model_levels',
            num_points=30,
            levels=np.arange(-30, 35, 5),
            colormap='RdBu_r',
            plot_type='filled_contour',
            extend='both',
            title=f"Vertical Cross-Section: {u_var} (model levels)",
            save_path=str(OUTPUT_DIR / f'mpasdiag_sample_plot_cross_section_{u_var.lower()}_model_levels.png')
        )

    except Exception as e:
        print(f"Error in wind cross-section example: {e}")


def main() -> int:
    """
    Run the vertical cross-section demonstration examples if required MPAS files are present.
    The function validates that the grid and data files exist, prints user guidance if files are missing,
    and then runs the example routines while capturing and returning an appropriate exit code.
    It is designed to be used as the script entry point for command-line execution.

    Parameters:
        None

    Returns:
        int: Exit code (0 on success, 1 on missing files or execution error).
    """
    example_grid_file = "../data/grids/x1.163842.init.nc"

    if not os.path.exists(example_grid_file):
        print("\nNOTE: This example requires actual MPAS data files.")
        print("Please update the file paths in the script to point to your MPAS data:")
        print("  - grid_file: Path to MPAS grid/static file")
        print("  - data_directory: Directory containing MPAS output files (mpasout*.nc)")
        print("\nExample usage after updating paths:")
        print("  python vertical_cross_section_example.py")
        return 1

    try:
        create_zonal_wind_cross_section()
        return 0
    except Exception as e:
        print(f"Error running vertical cross-section examples: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
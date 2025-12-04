#!/usr/bin/env python3

"""
MPASdiag Example5: Vertical Cross-Section 

This example demonstrates how to create vertical cross-section plots of 3D atmospheric
variables from MPAS model output using the new MPASVerticalCrossSectionPlotter class.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
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
    Create and save comprehensive vertical cross-section visualizations of zonal wind component through the atmosphere using three different vertical coordinate systems. This function demonstrates the complete workflow for generating atmospheric cross-sections from MPAS 3D model output by locating appropriate wind variables in the dataset, configuring a transect between specified geographic endpoints, and producing publication-quality plots with properly scaled contours and diverging colormaps. The function generates three separate cross-section plots using height AGL (above ground level), pressure, and model-level vertical coordinates to show the same atmospheric slice with different vertical perspectives. Each visualization uses a symmetric contour range with diverging colormap to highlight positive (eastward) and negative (westward) wind components. The resulting PNG files are saved to the configured output directory for inspection and further use.

    Parameters:
        grid_file (str): Absolute or relative path to MPAS grid initialization file containing mesh topology, cell connectivity, and spatial coordinate information (default: "../data/grids/x1.163842.init.nc").
        data_directory (str): Directory path containing MPAS 3D atmospheric output files in mpasout*.nc format with time-varying fields (default: "../data/u60k/").

    Returns:
        None
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
    Execute the vertical cross-section demonstration workflow validating file availability and orchestrating example routines with error handling. This entry point function performs prerequisite validation by checking for the existence of required MPAS grid and data files, provides helpful diagnostic messages to users when files are missing with instructions for proper configuration, and invokes the cross-section generation examples while capturing exceptions for graceful error reporting. The function is designed as the primary script entry point for command-line execution and returns standard Unix exit codes for integration with shell scripts and automation workflows. Successful execution produces multiple cross-section visualization files in the designated output directory.

    Parameters:
        None

    Returns:
        int: Unix exit status code where 0 indicates successful completion with all cross-sections generated, and 1 indicates missing prerequisite files or execution errors.
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
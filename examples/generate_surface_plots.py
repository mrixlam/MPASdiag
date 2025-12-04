#!/usr/bin/env python3
"""
MPASdiag Example3: 2D Surface Map with Filled Contours

This script generates professional plots using real MPAS diagnostic data.
It demonstrates all the enhanced features of MPASdiag v1.1.0 including:
- Automatic unit conversion
- Professional branding
- Enhanced scientific notation
- Composite plotting capabilities

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import os
import sys
import xarray as xr
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

package_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_dir))

from mpasdiag.processing import MPAS2DProcessor
from mpasdiag.visualization.surface import MPASSurfacePlotter
from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter


def generate_real_data_plots() -> Optional[str]:
    """
    Generate a comprehensive suite of publication-ready meteorological surface plots from MPAS atmospheric model diagnostic output data. This function orchestrates the complete visualization workflow by loading MPAS grid topology and diagnostic files, initializing the MPAS2DProcessor for coordinate extraction and data management, and creating specialized MPASSurfacePlotter instances for professional visualization. The function systematically generates three primary surface meteorological plots: 2-meter temperature with automatic Kelvin to Celsius conversion, mean sea level pressure with proper unit handling and contour formatting, and 2-meter specific humidity with scientific notation. Each plot includes automatic unit conversion based on CF conventions, professional branding with timestamp and institution logos, enhanced scientific notation formatting for readability, and high-resolution 300 DPI output suitable for publications. All generated PNG files are saved to the configured testPlot output directory with descriptive filenames.

    Parameters:
        None

    Returns:
        Optional[str]: Absolute path string to the output directory containing all successfully generated plot files, or None if data loading failures or plotting errors occur during execution.
    """
    
    print("=" * 80)
    print("MPASdiag Example2: 2D Surface Map")
    print("=" * 80)
    print()
    
    # Plot type: 'contour' for line contours, 'contourf' for filled contours
    plot_type = 'contourf'
    
    grid_file = "../data/grids/x1.40962.init.nc"
    data_file = "../data/u120k/diag/diag.2024-09-17_02.00.00.nc"
    output_dir = Path("testPlot")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Plots will be saved to: {output_dir}")

    try:
        processor = MPAS2DProcessor(grid_file, verbose=True)
        
        data_dir = os.path.dirname(data_file)
        processor.load_2d_data(data_dir)

        if hasattr(processor.dataset, 'ds'):
            data_ds = processor.dataset.ds
        else:
            data_ds = processor.dataset
        
        surface_viz = MPASSurfacePlotter(figsize=(10, 12), dpi=100)
        
        lon, lat = processor.extract_spatial_coordinates()
        extent = [-180.0, 180.0, -90.0, 90.0]
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    print("\n----------------- Generating Temperature Plot ----------------- \n")

    try:
        t2m_data = data_ds.t2m[0, :]

        _, _ = surface_viz.create_surface_map(
            lon, lat, t2m_data, 't2m',
            extent[0], extent[1], extent[2], extent[3],
            title='MPAS 2-meter Temperature',
            plot_type=plot_type,
            data_array=t2m_data
        )

        surface_viz.add_timestamp_and_branding()

        output_path = output_dir / f"mpasdiag_sample_plot_t2m_{plot_type}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()

    except Exception as e:
        print(f"❌ Error generating temperature plot: {e}")

    print("\n----------------- Generating Pressure Plot ----------------- \n")

    try:
        mslp_data = data_ds.mslp[0, :]

        _, _ = surface_viz.create_surface_map(
            lon, lat, mslp_data, 'mslp',
            extent[0], extent[1], extent[2], extent[3],
            title='MPAS Mean Sea Level Pressure',
            plot_type=plot_type,
            data_array=mslp_data
        )

        surface_viz.add_timestamp_and_branding()

        output_path = output_dir / f"mpasdiag_sample_plot_mslp_{plot_type}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()

    except Exception as e:
        print(f"❌ Error generating pressure plot: {e}")

    print("\n----------------- Generating Humidity Plot ----------------- \n")

    try:
        q2_data = data_ds.q2[0, :]

        _, _ = surface_viz.create_surface_map(
            lon, lat, q2_data, 'q2',
            extent[0], extent[1], extent[2], extent[3],
            title='MPAS 2-meter Specific Humidity',
            plot_type=plot_type,
            data_array=q2_data
        )

        surface_viz.add_timestamp_and_branding()

        output_path = output_dir / f"mpasdiag_sample_plot_q2_{plot_type}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()

    except Exception as e:
        print(f"❌ Error generating humidity plot: {e}")

    print("\n" + "=" * 80)
    print("✅ Real Data Plot Generation Complete!")
    print("=" * 80)
    

if __name__ == "__main__":
    generate_real_data_plots()
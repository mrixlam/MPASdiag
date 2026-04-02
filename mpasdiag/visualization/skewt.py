#!/usr/bin/env python3

"""
MPASdiag Core Visualization Module: Skew-T Log-P Diagrams

This module provides functionality for creating Skew-T Log-P diagrams from MPAS model output, which are essential tools for analyzing atmospheric soundings and vertical profiles of temperature, humidity, and wind. The module includes the MPASSkewTPlotter class, which specializes in extracting vertical profile data from MPAS datasets, performing necessary unit conversions, and rendering Skew-T diagrams with meteorological conventions. The plotter supports flexible styling options for temperature and dew point profiles, wind barbs, and stability indices, as well as the ability to overlay multiple soundings for comparison. It handles the complexities of unstructured mesh data by utilizing MPASRemapper for interpolation to regular pressure levels when needed. The module also includes error handling to ensure that it can gracefully manage cases where sounding data may be incomplete or missing. Core capabilities include customizable Skew-T diagram styling, support for both individual and multiple soundings, integration with MPASRemapper for data processing, and publication-quality output suitable for atmospheric science research and operational weather analysis. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: March 2026
Version: 1.0.0
"""
# Load standard libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Tuple, Optional, Dict, Any

from .base_visualizer import MPASVisualizer
from .styling import MPASVisualizationStyle

try:
    from metpy.plots import SkewT
    from metpy.units import units as munits
    import metpy.calc as mpcalc
except ImportError as _metpy_err:
    raise ImportError(
        "MetPy is required for Skew-T diagrams. "
        "Install it with: pip install metpy  or  "
        "conda install -c conda-forge metpy"
    ) from _metpy_err


class MPASSkewTPlotter(MPASVisualizer):
    """ Specialized plotter for creating Skew-T Log-P diagrams from MPAS atmospheric sounding data. """

    def __init__(self: 'MPASSkewTPlotter', 
                 figsize: Tuple[float, float] = (9, 14),
                 dpi: int = 150,
                 verbose: bool = True,) -> None:
        """
        This constructor initializes the MPASSkewTPlotter with specified figure size, resolution, and verbosity settings. It calls the base class constructor to set up common visualization parameters and prepares the plotter for creating Skew-T diagrams. The default figure size is set to 9 inches wide by 12 inches tall, which is a common aspect ratio for Skew-T diagrams, and the default resolution is set to 150 DPI for clear output. Verbose mode is enabled by default to provide feedback during the plotting process. 

        Parameters:
            figsize (Tuple[float, float]): Size of the figure in inches (width, height).
            dpi (int): Resolution of the figure in dots per inch.
            verbose (bool): Whether to print progress messages during plotting.

        Returns:
            None
        """
        super().__init__(figsize, dpi, verbose)

    def create_skewt_diagram(self: 'MPASSkewTPlotter', 
                             pressure: np.ndarray, 
                             temperature: np.ndarray, 
                             dewpoint: np.ndarray, 
                             u_wind: Optional[np.ndarray] = None, 
                             v_wind: Optional[np.ndarray] = None, 
                             title: Optional[str] = None, 
                             indices: Optional[Dict[str, Optional[float]]] = None, 
                             show_parcel: bool = False, 
                             save_path: Optional[str] = None,) -> Tuple[Figure, Axes]:
        """
        This method creates a Skew-T Log-P diagram using the provided pressure, temperature, dew point, and optional wind data. It utilizes MetPy's SkewT class to render the diagram with meteorological conventions, including plotting the temperature and dew point profiles, optional wind barbs, and a lifted parcel profile if requested. The method also adds thermodynamic reference lines and level markers based on provided indices for CAPE/CIN shading. The resulting figure is styled according to MPAS visualization standards and can be saved to a specified path. The method returns the Matplotlib Figure and Axes objects for further customization or display. 

        Parameters:
            pressure (np.ndarray): Pressure levels in hPa.
            temperature (np.ndarray): Temperature profile in °C.
            dewpoint (np.ndarray): Dew point profile in °C.
            u_wind (Optional[np.ndarray]): Zonal wind component in knots.
            v_wind (Optional[np.ndarray]): Meridional wind component in knots.
            title (Optional[str]): Title for the Skew-T diagram. Defaults to "MPAS Skew-T Log-P Diagram".
            indices (Optional[Dict[str, Optional[float]]]): Dictionary of thermodynamic indices for CAPE/CIN shading and level markers.
            show_parcel (bool): Whether to plot the lifted parcel profile. Defaults to False.
            save_path (Optional[str]): Path to save the figure. If None, the figure is not saved. 

        Returns:
            Tuple[Figure, Axes]: Matplotlib Figure and Axes objects with the Skew-T diagram rendered. 
        """
        p = np.asarray(pressure, dtype=np.float64)
        t = np.asarray(temperature, dtype=np.float64)
        td = np.asarray(dewpoint, dtype=np.float64)

        fig, ax = self._create_metpy_skewt(p, t, td, u_wind, v_wind,
                                            show_parcel, indices)

        if title is None:
            title = "MPAS Skew-T Log-P Diagram"

        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

        if indices is not None:
            self._add_indices_table(fig, indices)

        self.fig = fig
        self.ax = ax
        fig.subplots_adjust(bottom=0.18)
        self.add_timestamp_and_branding()

        if save_path:
            MPASVisualizationStyle.save_plot(
                fig, save_path, formats=['png', 'pdf'],
                bbox_inches='tight', pad_inches=0.1, dpi=self.dpi,
            )
            if self.verbose:
                print(f"Skew-T diagram saved to: {save_path}")

        return fig, ax

    def _create_metpy_skewt(self: 'MPASSkewTPlotter', 
                            p: np.ndarray, 
                            t: np.ndarray, 
                            td: np.ndarray, 
                            u_wind: Optional[np.ndarray], 
                            v_wind: Optional[np.ndarray], 
                            show_parcel: bool, 
                            indices: Optional[Dict[str, Optional[float]]],) -> Tuple[Figure, Axes]:
        """
        This private method creates a Skew-T Log-P diagram using MetPy's SkewT class. It takes in pressure, temperature, dew point, and optional wind data, and plots the temperature and dew point profiles on the Skew-T diagram. If wind data is provided, it adds wind barbs to the plot. If the show_parcel flag is set to True, it calculates and plots the lifted parcel profile based on the surface conditions. The method also adds thermodynamic reference lines and level markers based on provided indices for CAPE/CIN shading. The resulting figure is returned along with the Axes object for further customization. 

        Parameters: 
            p (np.ndarray): Pressure levels in hPa.
            t (np.ndarray): Temperature profile in °C.
            td (np.ndarray): Dew point profile in °C.
            u_wind (Optional[np.ndarray]): Zonal wind component in knots.
            v_wind (Optional[np.ndarray]): Meridional wind component in knots.
            show_parcel (bool): Whether to plot the lifted parcel profile.
            indices (Optional[Dict[str, Optional[float]]]): Dictionary of thermodynamic indices for CAPE/CIN shading and level markers.

        Returns:
            Tuple[Figure, Axes]: Matplotlib Figure and Axes objects with the Skew-T diagram rendered. 
        """
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        skew = SkewT(fig, rotation=45)

        p_u = p * munits.hPa
        t_u = t * munits.degC
        td_u = td * munits.degC

        skew.plot(p_u, t_u, 'r', linewidth=2, label='Temperature')
        skew.plot(p_u, td_u, 'g', linewidth=2, label='Dewpoint')

        if u_wind is not None and v_wind is not None:
            u = np.asarray(u_wind, dtype=np.float64)
            v = np.asarray(v_wind, dtype=np.float64)
            u_u = u * munits.knots
            v_u = v * munits.knots
            skew.plot_barbs(p_u, u_u, v_u)

        if show_parcel:
            try:
                prof = mpcalc.parcel_profile(p_u, t_u[0], td_u[0]).to('degC')
                skew.plot(p_u, prof, 'k--', linewidth=1.5, label='Parcel')
                if indices is not None:
                    cape_val = indices.get('cape')
                    cin_val = indices.get('cin')
                    if cape_val is not None and cape_val > 0:
                        skew.shade_cape(p_u, t_u, prof)
                    if cin_val is not None and cin_val < 0:
                        skew.shade_cin(p_u, t_u, prof)
            except Exception as exc:
                if self.verbose:
                    print(f"Warning: parcel profile plotting failed: {exc}")

        self._add_thermodynamic_overlays(skew)

        if indices is not None:
            self._add_level_markers(skew.ax, indices)

        skew.ax.set_ylim(1050, 100)
        skew.ax.set_xlim(-40, 50)
        skew.ax.legend(loc='upper left', fontsize=9)

        return fig, skew.ax

    @staticmethod
    def _add_thermodynamic_overlays(skew: Any) -> None:
        """
        This static method adds thermodynamic reference lines to the Skew-T diagram, including dry adiabats, moist adiabats, and mixing ratio lines. These overlays are essential for interpreting the stability of the atmosphere and understanding the thermodynamic processes at play. The method uses MetPy's built-in functions to plot these lines with specified styling for clarity. It is typically called during the Skew-T diagram creation process after plotting the main temperature and dew point profiles, and before adding any level markers or indices tables. 

        Parameters:
            skew (Any): The SkewT object from MetPy on which to add the thermodynamic overlays. 

        Returns:
            None 
        """
        skew.plot_dry_adiabats(alpha=0.3, linewidth=0.8)
        skew.plot_moist_adiabats(alpha=0.3, linewidth=0.8)
        skew.plot_mixing_lines(alpha=0.3, linewidth=0.8)

    @staticmethod
    def _add_level_markers(ax: Axes, 
                           indices: Dict[str, Optional[float]]) -> None:
        """
        This static method adds horizontal lines to the Skew-T diagram to indicate key pressure levels such as the Lifting Condensation Level (LCL), Level of Free Convection (LFC), and Equilibrium Level (EL). It checks the provided indices dictionary for the presence of these levels, and if they are available and valid, it draws dashed horizontal lines at the corresponding pressure values. Each line is labeled with the level name and pressure value for clarity. This method is typically called during the Skew-T diagram creation process after plotting the main profiles and thermodynamic overlays, and before adding any indices tables. 

        Parameters:
            ax (Axes): The Matplotlib Axes object on which to add the level markers.
            indices (Dict[str, Optional[float]]): Dictionary containing the pressure values for LCL, LFC, and EL levels.

        Returns:
            None
        """
        marker_styles = {
            'lcl_pressure': ('LCL', 'tab:blue', '--'),
            'lfc_pressure': ('LFC', 'tab:orange', '--'),
            'el_pressure': ('EL', 'tab:purple', '--'),
        }
        for key, (label, color, ls) in marker_styles.items():
            val = indices.get(key)
            if val is not None and np.isfinite(val):
                ax.axhline(val, color=color, linestyle=ls, linewidth=1.2, alpha=0.7)
                ax.text(
                    ax.get_xlim()[1], val, f' {label} ({val:.0f} hPa)',
                    fontsize=8, color=color, va='center', ha='left',
                    clip_on=True,
                )

    @staticmethod
    def _add_indices_table(fig: Figure, 
                           indices: Dict[str, Optional[float]],) -> None:
        """
        This static method creates a table of sounding indices and stability parameters to be displayed below the Skew-T diagram. It takes a dictionary of indices, formats them into three columns (Parcel/Instability, Moisture/Thermo, Shear/Severe), and creates a Matplotlib table with styled headers and alternating row colors for readability. The method filters out any indices that are not available (None) and pads the columns to ensure they have equal length before creating the table. This table provides a convenient summary of key atmospheric parameters that can be used to assess the stability and severe weather potential of the sounding. It is typically called during the Skew-T diagram creation process after plotting the main profiles and thermodynamic overlays, and before finalizing the figure for display or saving. 

        Parameters:
            fig (Figure): The Matplotlib Figure object on which to add the indices table.
            indices (Dict[str, Optional[float]]): Dictionary containing the sounding indices and stability parameters to be displayed in the table.

        Returns:
            None
        """
        def _fmt(key: str, label: str, unit: str = '') -> str:
            val = indices.get(key)
            if val is None:
                return ''
            if unit:
                return f'{label}: {val:.0f} {unit}'
            return f'{label}: {val:.1f}'

        # Build rows for each column
        col_parcel = [
            _fmt('sbcape', 'SBCAPE', 'J/kg'),
            _fmt('sbcin', 'SBCIN', 'J/kg'),
            _fmt('mlcape', 'MLCAPE', 'J/kg'),
            _fmt('mlcin', 'MLCIN', 'J/kg'),
            _fmt('mucape', 'MUCAPE', 'J/kg'),
            _fmt('mucin', 'MUCIN', 'J/kg'),
            _fmt('lifted_index', 'LI'),
            _fmt('lcl_pressure', 'LCL', 'hPa'),
            _fmt('lfc_pressure', 'LFC', 'hPa'),
            _fmt('el_pressure', 'EL', 'hPa'),
        ]

        col_moisture = [
            _fmt('precipitable_water', 'PW', 'mm'),
            _fmt('wet_bulb_zero_height', 'WBZ', 'm'),
            _fmt('k_index', 'K-Index'),
            _fmt('total_totals', 'TT'),
            _fmt('showalter_index', 'SI'),
            _fmt('cross_totals', 'CT'),
            _fmt('dcape', 'DCAPE', 'J/kg'),
        ]

        col_shear = [
            _fmt('bulk_shear_0_6km', '0-6 BS', 'kt'),
            _fmt('bulk_shear_0_1km', '0-1 BS', 'kt'),
            _fmt('srh_0_1km', '0-1 SRH', 'm\u00b2/s\u00b2'),
            _fmt('srh_0_3km', '0-3 SRH', 'm\u00b2/s\u00b2'),
            _fmt('stp', 'STP'),
            _fmt('scp', 'SCP'),
            _fmt('sweat_index', 'SWEAT'),
        ]

        # Filter out empty strings
        col_parcel = [s for s in col_parcel if s]
        col_moisture = [s for s in col_moisture if s]
        col_shear = [s for s in col_shear if s]

        if not col_parcel and not col_moisture and not col_shear:
            return

        # Pad columns to equal length
        n_rows = max(len(col_parcel), len(col_moisture), len(col_shear))
        for col in (col_parcel, col_moisture, col_shear):
            col.extend([''] * (n_rows - len(col)))

        headers = ['Parcel / Instability', 'Moisture / Thermo', 'Shear / Severe']

        # Create axes for the table below the main plot
        table_ax = fig.add_axes([0.05, 0.01, 0.90, 0.14])
        table_ax.axis('off')

        cell_text = [[col_parcel[i], col_moisture[i], col_shear[i]]
                      for i in range(n_rows)]

        table = table_ax.table(
            cellText=cell_text,
            colLabels=headers,
            loc='center',
            cellLoc='left',
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)

        # Style header row
        for j in range(3):
            cell = table[0, j]
            cell.set_text_props(weight='bold', fontsize=9)
            cell.set_facecolor('#d9e2ec')
            cell.set_edgecolor('#8fa8c8')

        # Style data rows
        for i in range(1, n_rows + 1):
            for j in range(3):
                cell = table[i, j]
                cell.set_facecolor('#f7f9fb' if i % 2 == 0 else 'white')
                cell.set_edgecolor('#c8d6e0')

        table.scale(1.0, 1.3)

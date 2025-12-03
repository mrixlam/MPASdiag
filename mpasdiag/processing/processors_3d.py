#!/usr/bin/env python3

"""
MPAS 3D Atmospheric Data Processor

This module provides specialized functionality for processing MPAS 3D atmospheric variables including temperature, winds, moisture, and geopotential height on model native levels and pressure surfaces. It implements the MPAS3DProcessor class that extends MPASBaseProcessor to handle vertical level extraction, 3D coordinate retrieval for unstructured mesh atmospheric columns, interpolation to pressure or height coordinates, and statistical diagnostics for atmospheric fields. The processor manages both history stream files with full 3D atmospheric state variables and diagnostic stream files with derived quantities, provides methods for extracting data at specific vertical levels (model levels, pressure levels, height levels) with proper time indexing, handles complex vertical coordinate systems including hybrid sigma-pressure and geometric height, and supports batch processing for creating vertical cross-sections and level-specific horizontal maps. Core capabilities include automatic vertical level detection and validation, pressure-to-height coordinate conversion using standard atmosphere approximations, spatial coordinate extraction for 3D variables with cell-center and edge geometries, integration with uxarray for unstructured mesh operations, and seamless data preparation for 3D visualization workflows suitable for atmospheric dynamics research and model evaluation.

Classes:
    MPAS3DProcessor: Specialized processor class for extracting and analyzing 3D atmospheric variables from MPAS model output with vertical coordinate handling.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import uxarray as ux
from typing import List, Tuple, Any, Optional, Union, cast

from .base import MPASBaseProcessor
from .utils_datetime import MPASDateTimeUtils
from .constants import MPASOUT_GLOB, DATASET_NOT_LOADED_3D_MSG


class MPAS3DProcessor(MPASBaseProcessor):
    """
    Specialized processor for 3D MPAS atmospheric data.
    
    This class handles 3D atmospheric variables from MPAS models,
    providing methods for vertical level analysis, atmospheric variables,
    and MPAS output file processing.
    """
    
    def __init__(self, grid_file: str, verbose: bool = True):
        """
        Initialize the 3D MPAS processor for atmospheric variable analysis on vertical levels. This constructor sets up the processor by loading the MPAS grid file to extract mesh coordinates and cell connectivity required for 3D field operations. It inherits base functionality from MPASBaseProcessor and configures verbose logging for operational visibility. The processor handles atmospheric variables distributed across vertical levels including temperature, winds, moisture, and other prognostic fields from MPAS model output. Vertical coordinate systems (pressure, height, model levels) are validated and prepared for cross-section extraction and vertical interpolation operations.

        Parameters:
            grid_file (str): Path to the MPAS grid file containing mesh topology and spatial coordinates.
            verbose (bool): If True, enable informational logging during processing operations (default: True).

        Returns:
            None
        """
        super().__init__(grid_file, verbose)
    
    def extract_2d_coordinates_for_variable(self, var_name: str, data_array: Optional[xr.DataArray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract horizontal coordinate arrays for 3D variables based on their spatial dimension topology. This method determines the appropriate mesh location (cells, vertices, or edges) where the variable is defined and returns corresponding longitude and latitude arrays from the grid file. It inspects the variable's dimensions or provided data array to identify the spatial dimension, then loads the matching coordinate pair from the grid dataset. The method supports cell-centered (nCells), vertex-based (nVertices), and edge-based (nEdges) variables common in MPAS atmospheric output. These horizontal coordinates are essential for cross-section path definition and geographic plotting of 3D fields. Note that this only extracts horizontal coordinates; vertical coordinate handling requires separate methods.

        Parameters:
            var_name (str): Name of the 3D atmospheric variable to extract horizontal coordinates for.
            data_array (Optional[xr.DataArray]): Optional data array to inspect for dimension information (default: None).

        Returns:
            tuple of (np.ndarray, np.ndarray): Longitude and latitude arrays matching the variable's spatial dimension.

        Raises:
            ValueError: If dataset is not loaded or spatial dimension cannot be determined.
        """
        if self.dataset is None:
            raise ValueError(DATASET_NOT_LOADED_3D_MSG)
            
        try:
            with xr.open_dataset(self.grid_file) as grid_ds:
                spatial_dim = 'nCells' 
                
                if data_array is not None:
                    if 'nVertices' in data_array.sizes:
                        spatial_dim = 'nVertices'
                    elif 'nCells' in data_array.sizes:
                        spatial_dim = 'nCells'
                    elif 'nEdges' in data_array.sizes:
                        spatial_dim = 'nEdges'
                elif var_name in self.dataset:
                    if 'nVertices' in self.dataset[var_name].sizes:
                        spatial_dim = 'nVertices'
                    elif 'nCells' in self.dataset[var_name].sizes:
                        spatial_dim = 'nCells'
                    elif 'nEdges' in self.dataset[var_name].sizes:
                        spatial_dim = 'nEdges'
                
                if spatial_dim == 'nVertices':
                    lon_names = ['lonVertex', 'lon_vertex', 'longitude_vertex']
                    lat_names = ['latVertex', 'lat_vertex', 'latitude_vertex']
                elif spatial_dim == 'nEdges':
                    lon_names = ['lonEdge', 'lon_edge', 'longitude_edge']
                    lat_names = ['latEdge', 'lat_edge', 'latitude_edge']
                else:  
                    lon_names = ['lonCell', 'longitude', 'lon']
                    lat_names = ['latCell', 'latitude', 'lat']
                
                lon_coords = lat_coords = None
                
                for name in lon_names:
                    if name in grid_ds.coords or name in grid_ds.data_vars:
                        lon_coords = grid_ds[name].values
                        break
                        
                for name in lat_names:
                    if name in grid_ds.coords or name in grid_ds.data_vars:
                        lat_coords = grid_ds[name].values
                        break
                        
                if lon_coords is None or lat_coords is None:
                    available_vars = list(grid_ds.coords.keys()) + list(grid_ds.data_vars.keys())
                    raise ValueError(f"Could not find {spatial_dim} coordinates in grid file. Available variables: {available_vars}")
                
                if np.nanmax(np.abs(lat_coords)) <= np.pi:
                    lat_coords = lat_coords * 180.0 / np.pi
                    lon_coords = lon_coords * 180.0 / np.pi
                
                lon_coords = lon_coords.ravel()
                lat_coords = lat_coords.ravel()
                lon_coords = ((lon_coords + 180) % 360) - 180
                
                if self.verbose:
                    print(f"Extracted {spatial_dim} coordinates for 3D variable {var_name}: {len(lon_coords):,} points")
                
                return lon_coords, lat_coords
                
        except Exception as e:
            raise RuntimeError(f"Error loading coordinates from grid file {self.grid_file}: {e}")
    
    def find_mpasout_files(self, data_dir: str) -> List[str]:
        """
        Locate and validate MPAS atmospheric output files in the specified directory. This method searches for files matching the MPAS output naming convention (mpasout*.nc) and returns a sorted list of valid file paths for temporal processing. It performs validation to ensure sufficient files exist for meaningful analysis and raises descriptive exceptions if the directory is empty or contains inadequate data. The method delegates to the base class pattern-matching utility with MPAS-specific parameters. File sorting ensures proper temporal sequencing for time series and ensemble operations across model output timesteps.

        Parameters:
            data_dir (str): Directory path to search for MPAS atmospheric output files.

        Returns:
            List[str]: Sorted list of MPAS output file paths found in the directory.

        Raises:
            FileNotFoundError: If no MPAS output files matching the pattern are found.
            ValueError: If insufficient files are present for temporal analysis operations.
        """
        try:
            return self._find_files_by_pattern(data_dir, MPASOUT_GLOB, "MPAS output files")
        except FileNotFoundError:
            mpasout_sub = os.path.join(data_dir, "mpasout")
            try:
                return self._find_files_by_pattern(mpasout_sub, MPASOUT_GLOB, "MPAS output files")
            except FileNotFoundError:
                files = [f for f in sorted(__import__('glob').glob(os.path.join(data_dir, "**", MPASOUT_GLOB), recursive=True))]
                if not files:
                    raise FileNotFoundError(f"No MPAS output files found under: {data_dir}")
                if len(files) < 2:
                    raise ValueError(f"Insufficient MPAS output files for temporal analysis. Found {len(files)}, need at least 2.")
                if self.verbose:
                    print(f"\nFound {len(files)} MPAS output files (recursive search):")
                    for i, f in enumerate(files[:5]):
                        print(f"  {i+1}: {os.path.basename(f)}")
                return files

    def load_3d_data(self, data_dir: str, use_pure_xarray: bool = False, 
                     reference_file: str = "") -> 'MPAS3DProcessor':
        """
        Load and configure 3D atmospheric data from MPAS output files with optimized chunking for vertical levels. This method reads atmospheric model output from the specified directory using either UXarray or pure xarray backends, applies memory-efficient chunking strategies tailored for 3D fields with vertical structure, and enriches the dataset with spatial coordinates from the grid file. The method supports lazy loading to handle large datasets efficiently and automatically detects the appropriate data structure (regular dataset or UXarray grid object). After loading, it adds geographic coordinates and mesh connectivity required for vertical interpolation, cross-sections, and volume rendering. Returns self to enable method chaining in analysis workflows.

        Parameters:
            data_dir (str): Directory path containing MPAS atmospheric output files to load.
            use_pure_xarray (bool): If True, use pure xarray backend instead of UXarray (default: False).
            reference_file (str): Optional specific file to use for time coordinate ordering (default: "").

        Returns:
            MPAS3DProcessor: Self reference for method chaining operations.
        """
        chunks_3d = {'Time': 1, 'nCells': 50000, 'nVertLevels': 66}
        
        dataset, data_type = self._load_data(
            data_dir, 
            use_pure_xarray, 
            reference_file,
            chunks=chunks_3d,
            data_type_label="3D"
        )
        
        if hasattr(dataset, 'data_vars'):  
            dataset = self.add_spatial_coordinates(dataset)
            self.dataset = dataset
            self.data_type = data_type
        elif hasattr(dataset, 'ds'):  
            dataset.ds = self.add_spatial_coordinates(dataset.ds)
            self.dataset = dataset
            self.data_type = data_type
        
        return self

    def get_available_3d_variables(self) -> List[str]:
        """
        Identify and list all 3D atmospheric variables with vertical structure in the loaded dataset. This method scans the dataset's data variables and identifies those containing vertical dimension coordinates (nVertLevels or nVertLevelsP1), indicating true 3D atmospheric fields. It filters out 2D surface variables and coordinate arrays, returning only variables suitable for vertical analysis and cross-section operations. When verbose mode is enabled, the method prints detailed information including variable dimensions, shapes, and units for the first 10 variables. This inventory is essential for determining which fields support vertical interpolation and level extraction.

        Parameters:
            None

        Returns:
            List[str]: Names of all 3D atmospheric variables found in the dataset.

        Raises:
            ValueError: If dataset is not loaded prior to calling this method.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_3d_data() first.")
        
        atmospheric_3d_vars = []
        
        for var_name in self.dataset.data_vars:
            var_dims = self.dataset[var_name].sizes

            if 'nVertLevels' in var_dims or 'nVertLevelsP1' in var_dims:
                atmospheric_3d_vars.append(var_name)
        
        if self.verbose:
            print(f"Found {len(atmospheric_3d_vars)} 3D atmospheric variables:")

            for var in atmospheric_3d_vars[:10]:  
                dims_str = 'x'.join(str(self.dataset[var].sizes[dim]) for dim in self.dataset[var].sizes)
                units = self.dataset[var].attrs.get('units', 'no units')
                print(f"  {var}: [{dims_str}] - {units}")

            if len(atmospheric_3d_vars) > 10:
                print(f"  ... and {len(atmospheric_3d_vars) - 10} more")
        
        return atmospheric_3d_vars

    def get_3d_variable_data(self, var_name: str, level: Union[str, int, float], 
                            time_index: int = 0) -> xr.DataArray:
        """
        Extract 3D atmospheric variable data at a specified vertical level and timestep with flexible level specification. This method retrieves a 2D horizontal slice from a 3D field by selecting a single vertical level using either model level indices, pressure values, or named level strings. For integer levels, it directly indexes the vertical coordinate; for pressure levels, it interpolates to find the nearest model level by comparing against the pressure field. The method validates that the variable has vertical structure and handles both nVertLevels and nVertLevelsP1 dimensions. Returns a computed 2D DataArray ready for visualization or further analysis.

        Parameters:
            var_name (str): Name of the 3D atmospheric variable to extract.
            level (Union[str, int, float]): Vertical level specification - int for model level index (0-based), float for pressure level in Pa or hPa, or str for special names like 'surface' or 'top'.
            time_index (int): Zero-based time index to extract data from (default: 0).

        Returns:
            xr.DataArray: 2D horizontal slice of variable data at the specified level and time.

        Raises:
            ValueError: If dataset not loaded, variable not found, variable lacks vertical dimension, or level specification is invalid.
        """
        if self.dataset is None:
            raise ValueError(DATASET_NOT_LOADED_3D_MSG)
        
        if var_name not in self.dataset.data_vars:
            available_vars = list(self.dataset.data_vars.keys())
            raise ValueError(f"Variable '{var_name}' not found. Available variables: {available_vars[:20]}...")
        
        if 'nVertLevels' not in self.dataset[var_name].sizes and 'nVertLevelsP1' not in self.dataset[var_name].sizes:
            raise ValueError(f"Variable '{var_name}' is not a 3D atmospheric variable")
        
        time_dim, validated_time_index, time_size = MPASDateTimeUtils.validate_time_parameters(self.dataset, time_index, self.verbose)
        
        if isinstance(level, int):
            level_idx = level
            max_levels = self.dataset.sizes.get('nVertLevels', self.dataset.sizes.get('nVertLevelsP1', 0))

            if level_idx >= max_levels:
                raise ValueError(f"Model level {level_idx} exceeds available levels {max_levels}")
                
        elif isinstance(level, float):
            if 'pressure_p' in self.dataset and 'pressure_base' in self.dataset:
                pressure_p = self.dataset['pressure_p'].isel({time_dim: validated_time_index})
                pressure_base = self.dataset['pressure_base'].isel({time_dim: validated_time_index})
                total_pressure = pressure_p + pressure_base

                mean_pressure = total_pressure.mean(dim='nCells')
                mean_p_vals = mean_pressure.values

                if level >= mean_p_vals.max():
                    level_idx = 0
                    if self.verbose:
                        print(f"Requested pressure {level:.1f} Pa above surface mean; using surface level 0")
                elif level <= mean_p_vals.min():
                    level_idx = len(mean_p_vals) - 1
                    if self.verbose:
                        print(f"Requested pressure {level:.1f} Pa below top mean; using top level {level_idx}")
                else:
                    lower_idx = int(np.argmax(mean_p_vals >= level))
                    if lower_idx >= len(mean_p_vals) - 1:
                        level_idx = lower_idx
                    else:
                        upper_idx = lower_idx + 1
                        p_lower = mean_p_vals[lower_idx]
                        p_upper = mean_p_vals[upper_idx]

                        if p_lower == p_upper:
                            w = 0.0
                        else:
                            w = (level - p_upper) / (p_lower - p_upper)

                        if self.verbose:
                            print(f"Requested pressure: {level:.1f} Pa, interpolating between mean levels {lower_idx} ({p_lower:.1f} Pa) and {upper_idx} ({p_upper:.1f} Pa) w={w:.3f}")

                        vertical_dim = 'nVertLevels' if 'nVertLevels' in self.dataset[var_name].sizes else 'nVertLevelsP1'

                        if self.data_type == 'uxarray':
                            var_lower = self.dataset[var_name][validated_time_index].isel({vertical_dim: lower_idx})
                            var_upper = self.dataset[var_name][validated_time_index].isel({vertical_dim: upper_idx})
                        else:
                            var_lower = self.dataset[var_name].isel({time_dim: validated_time_index, vertical_dim: lower_idx})
                            var_upper = self.dataset[var_name].isel({time_dim: validated_time_index, vertical_dim: upper_idx})

                        try:
                            interp_field = (1.0 - w) * var_lower + w * var_upper
                            if hasattr(interp_field, 'compute'):
                                interp_field = cast(Any, interp_field).compute()

                            interp_field.attrs = getattr(var_lower, 'attrs', {})

                            if self.verbose:
                                vals = interp_field.values.flatten()
                                fin = vals[np.isfinite(vals)]
                                if len(fin) > 0:
                                    print(f"Interpolated field range: {fin.min():.3f} to {fin.max():.3f}")
                            return interp_field
                        except Exception:
                            level_idx = int(np.argmin(np.abs(mean_p_vals - level)))
                            if self.verbose:
                                print(f"Interpolation failed, falling back to nearest mean level {level_idx}")
            else:
                raise ValueError("Cannot find pressure level - pressure data not available")
                
        elif isinstance(level, str):
            if level.lower() == 'surface':
                level_idx = 0
            elif level.lower() == 'top':
                level_idx = self.dataset.sizes.get('nVertLevels', self.dataset.sizes.get('nVertLevelsP1', 1)) - 1
            else:
                raise ValueError(f"Unknown level specification: {level}")
        else:
            raise ValueError(f"Invalid level specification: {level}")
        
        if self.verbose:
            print(f"Extracting {var_name} data at level {level} (index {level_idx}), time index {validated_time_index}")
        
        vertical_dim = 'nVertLevels' if 'nVertLevels' in self.dataset[var_name].sizes else 'nVertLevelsP1'
        
        if self.data_type == 'uxarray':
            var_data = self.dataset[var_name][validated_time_index].isel({vertical_dim: level_idx})
        else:
            var_data = self.dataset[var_name].isel({time_dim: validated_time_index, vertical_dim: level_idx})
        
        if hasattr(var_data, 'compute'):
            var_data = cast(Any, var_data).compute()
        
        if hasattr(var_data, 'attrs'):
            var_data.attrs['selected_level'] = level
            var_data.attrs['level_index'] = level_idx
            
            if isinstance(level, float) and 'pressure_p' in self.dataset and 'pressure_base' in self.dataset:
                pressure_p = self.dataset['pressure_p'].isel({time_dim: validated_time_index, vertical_dim: level_idx})
                pressure_base = self.dataset['pressure_base'].isel({time_dim: validated_time_index, vertical_dim: level_idx})
                actual_pressure = (pressure_p + pressure_base).mean().values
                var_data.attrs['actual_pressure_level'] = f"{actual_pressure:.1f} Pa"
        
        if self.verbose:
            if hasattr(var_data, 'values'):
                data_values = var_data.values.flatten()
                finite_values = data_values[np.isfinite(data_values)]

                if len(finite_values) > 0:
                    print(f"Variable {var_name} at level {level} range: {finite_values.min():.3f} to {finite_values.max():.3f}")
                    
                    if hasattr(var_data, 'attrs') and 'units' in var_data.attrs:
                        print(f"Units: {var_data.attrs['units']}")
                else:
                    print(f"Warning: No finite values found for {var_name} at level {level}")
        
        return var_data

    def get_vertical_levels(self, var_name: str, return_pressure: bool = True, 
                           time_index: int = 0) -> List[Union[int, float]]:
        """
        Retrieve all available vertical levels for a 3D atmospheric variable with optional pressure conversion. This method determines the vertical dimension (nVertLevels or nVertLevelsP1) used by the specified variable and returns either model level indices or pressure values depending on the return_pressure flag. When pressure values are requested, the method computes total pressure by combining perturbation and base state pressure fields, then averages horizontally across all cells to obtain representative pressure levels. The method handles staggered vertical grids and provides verbose output showing pressure range and extrema. Returns a list suitable for level iteration in vertical analysis workflows.

        Parameters:
            var_name (str): Name of the 3D atmospheric variable to query for vertical levels.
            return_pressure (bool): If True, return pressure levels in Pa; if False, return zero-based model level indices (default: True).
            time_index (int): Time index to use for pressure calculation when return_pressure is True (default: 0).

        Returns:
            List[Union[int, float]]: Available vertical levels as either model indices (int) or pressure values (float in Pa).

        Raises:
            ValueError: If dataset not loaded, variable not found, or variable lacks vertical dimension structure.
        """
        if self.dataset is None:
            raise ValueError(DATASET_NOT_LOADED_3D_MSG)
        
        if var_name not in self.dataset.data_vars:
            raise ValueError(f"Variable '{var_name}' not found in dataset")
        
        var_dims = self.dataset[var_name].sizes
        
        if 'nVertLevels' in var_dims:
            num_levels = self.dataset.sizes['nVertLevels']
            vertical_dim = 'nVertLevels'
        elif 'nVertLevelsP1' in var_dims:
            num_levels = self.dataset.sizes['nVertLevelsP1']
            vertical_dim = 'nVertLevelsP1'
        else:
            raise ValueError(f"Variable '{var_name}' is not a 3D atmospheric variable")
        
        if return_pressure and 'pressure' in self.dataset:
            time_dim, validated_time_index, _ = MPASDateTimeUtils.validate_time_parameters(self.dataset, time_index, self.verbose)
            try:
                pressure_da = self.dataset['pressure'].isel({time_dim: validated_time_index})
            except Exception:
                pressure_da = self.dataset['pressure']

            mean_pressure_levels = pressure_da.mean(dim='nCells').values
            mean_pressure_levels = np.asarray(mean_pressure_levels, dtype=float).ravel()

            if not np.all(np.isfinite(mean_pressure_levels)) or np.nanmin(mean_pressure_levels) <= 0:
                if self.verbose:
                    print("Warning: computed mean pressure levels from 'pressure' contain non-positive or non-finite values; falling back to other methods")
            else:
                if vertical_dim == 'nVertLevelsP1':
                    if num_levels == len(mean_pressure_levels) + 1:
                        mean_pressure_levels = np.append(mean_pressure_levels, mean_pressure_levels[-1] * 0.9)

                if self.verbose:
                    print(f"Pressure levels for {var_name} ({num_levels} levels) from 'pressure' variable:")
                    print(f"  Surface: {mean_pressure_levels[0]:.1f} Pa")
                    print(f"  Top: {mean_pressure_levels[-1]:.1f} Pa")

                return mean_pressure_levels.tolist()

        if return_pressure and 'pressure_p' in self.dataset and 'pressure_base' in self.dataset:
            time_dim, validated_time_index, _ = MPASDateTimeUtils.validate_time_parameters(self.dataset, time_index, self.verbose)

            pressure_p = self.dataset['pressure_p'].isel({time_dim: validated_time_index})
            pressure_base = self.dataset['pressure_base'].isel({time_dim: validated_time_index})
            total_pressure = pressure_p + pressure_base

            mean_pressure_levels = total_pressure.mean(dim='nCells').values

            if vertical_dim == 'nVertLevelsP1':
                if num_levels == len(mean_pressure_levels) + 1:
                    mean_pressure_levels = np.append(mean_pressure_levels, mean_pressure_levels[-1] * 0.9)

            if self.verbose:
                print(f"Pressure levels for {var_name} ({num_levels} levels):")
                print(f"  Surface: {mean_pressure_levels[0]:.1f} Pa")
                print(f"  Top: {mean_pressure_levels[-1]:.1f} Pa")
                print(f"  Range: {mean_pressure_levels.min():.1f} to {mean_pressure_levels.max():.1f} Pa")

            return mean_pressure_levels.tolist()

        if return_pressure and 'fzp' in self.dataset and 'surface_pressure' in self.dataset:
            try:
                time_dim, validated_time_index, _ = MPASDateTimeUtils.validate_time_parameters(self.dataset, time_index, self.verbose)
                fzp = self.dataset['fzp'].isel({time_dim: validated_time_index}).values
                sp = self.dataset['surface_pressure'].isel({time_dim: validated_time_index}).values
                mean_sp = np.nanmean(sp)

                mean_pressure_levels = (np.asarray(fzp, dtype=float) * mean_sp)
                mean_pressure_levels = np.asarray(mean_pressure_levels, dtype=float).ravel()

                if not np.all(np.isfinite(mean_pressure_levels)) or np.nanmin(mean_pressure_levels) <= 0:
                    idx = np.arange(len(mean_pressure_levels))
                    good_idx = np.nonzero(np.isfinite(mean_pressure_levels) & (mean_pressure_levels > 0))[0]

                    if good_idx.size >= 2:
                        mean_pressure_levels = np.interp(idx, good_idx, mean_pressure_levels[good_idx])
                    elif good_idx.size == 1:
                        ref = mean_pressure_levels[good_idx[0]]
                        mean_pressure_levels = np.linspace(mean_sp, ref, len(mean_pressure_levels))
                    else:
                        mean_pressure_levels = np.logspace(np.log10(mean_sp), np.log10(1.0), len(mean_pressure_levels))

                if vertical_dim == 'nVertLevelsP1':
                    if num_levels == len(mean_pressure_levels) + 1:
                        mean_pressure_levels = np.append(mean_pressure_levels, mean_pressure_levels[-1] * 0.9)

                if self.verbose:
                    print(f"Reconstructed pressure levels from hybrid coefficients for {var_name} ({num_levels} levels):")
                    print(f"  Surface (mean): {mean_sp:.1f} Pa")
                    print(f"  Top: {mean_pressure_levels[-1]:.1f} Pa")

                return mean_pressure_levels.tolist()
            except Exception as e:
                if self.verbose:
                    print(f"Warning: failed to reconstruct pressure levels from hybrid coefficients: {e}")

        level_indices: List[Union[int, float]] = list(range(num_levels))

        if self.verbose:
            print(f"Model levels for {var_name}: {num_levels} levels (indices 0-{num_levels-1})")

        return level_indices

    def add_spatial_coordinates(self, combined_ds: xr.Dataset) -> xr.Dataset:
        """
        Enrich 3D atmospheric dataset with spatial coordinates and mesh connectivity from MPAS grid file. This method loads the grid file and extracts geographic coordinates for cell centers, vertices, and edges along with dimensional indices for proper dataset structure. It handles multiple spatial dimensions including nCells for cell-centered variables, nVertices for vertex-based fields, nEdges for edge-normal wind components, and vertical dimensions (nVertLevels, nVertLevelsP1, nSoilLevels) for atmospheric and subsurface levels. The method adds coordinate variables as metadata while preserving all existing data variables. Missing or incompatible coordinates are handled gracefully with warning messages when verbose mode is enabled. This coordinate enrichment is essential for subsequent spatial operations including vertical interpolation, cross-section extraction, and geographic visualization of 3D atmospheric fields.

        Parameters:
            combined_ds (xr.Dataset): Combined xarray dataset containing 3D atmospheric data from multiple timesteps.

        Returns:
            xr.Dataset: Enriched dataset with added coordinate variables for all spatial dimensions and geographic coordinates for plotting.
        """
        try:
            grid_file_ds = xr.open_dataset(self.grid_file)
            if self.verbose:
                print(f"\nGrid file loaded successfully with variables: \n{list(grid_file_ds.variables.keys())}\n")
            
            coords_to_add = {}
            data_vars_to_add = {}
            
            if 'lonCell' in grid_file_ds.variables and 'nCells' in combined_ds.sizes:
                coords_to_add['nCells'] = ('nCells', np.arange(combined_ds.sizes['nCells'])) 

                if self.verbose:
                    print(f"Added nCells index coordinate for nCells dimension ({combined_ds.sizes['nCells']} values)")
            
            if 'nVertLevels' in combined_ds.sizes:
                coords_to_add['nVertLevels'] = ('nVertLevels', np.arange(combined_ds.sizes['nVertLevels']))

                if self.verbose:
                    print(f"Added nVertLevels index coordinate for nVertLevels dimension ({combined_ds.sizes['nVertLevels']} values)")

            if 'nVertLevelsP1' in combined_ds.sizes:
                coords_to_add['nVertLevelsP1'] = ('nVertLevelsP1', np.arange(combined_ds.sizes['nVertLevelsP1']))

                if self.verbose:
                    print(f"Added nVertLevelsP1 index coordinate for nVertLevelsP1 dimension ({combined_ds.sizes['nVertLevelsP1']} values)")

            if 'nEdges' in combined_ds.sizes:
                coords_to_add['nEdges'] = ('nEdges', np.arange(combined_ds.sizes['nEdges']))

                if self.verbose:
                    print(f"Added nEdges index coordinate for nEdges dimension ({combined_ds.sizes['nEdges']} values)")

            if 'nVertices' in combined_ds.sizes:
                coords_to_add['nVertices'] = ('nVertices', np.arange(combined_ds.sizes['nVertices']))

                if self.verbose:
                    print(f"Added nVertices index coordinate for nVertices dimension ({combined_ds.sizes['nVertices']} values)")

            if 'nSoilLevels' in combined_ds.sizes:
                coords_to_add['nSoilLevels'] = ('nSoilLevels', np.arange(combined_ds.sizes['nSoilLevels']))

                if self.verbose:
                    print(f"Added nSoilLevels index coordinate for nSoilLevels dimension ({combined_ds.sizes['nSoilLevels']} values)")
            
            spatial_vars = ['latCell', 'lonCell', 'latVertex', 'lonVertex']

            for var_name in spatial_vars:
                if var_name in grid_file_ds.variables and var_name not in combined_ds.data_vars:
                    var_data = grid_file_ds[var_name]
                    data_vars_to_add[var_name] = var_data

                    if self.verbose:
                        print(f"Added spatial coordinate variable: {var_name}")
            
            if coords_to_add:
                combined_ds = combined_ds.assign_coords(coords_to_add)
                if self.verbose:
                    print(f"\nSuccessfully added {len(coords_to_add)} coordinate variables")
                
            if data_vars_to_add:
                for var_name, var_data in data_vars_to_add.items():
                    combined_ds[var_name] = var_data
                if self.verbose:
                    print(f"Successfully added {len(data_vars_to_add)} spatial variables")
                    print("\nUpdated dataset coordinates:", list(combined_ds.coords.keys()))
            else:
                if self.verbose:
                    print("No additional coordinate variables found to add")
                
            grid_file_ds.close()
            
        except Exception as coord_error:
            if self.verbose:
                print(f"Warning: Could not add 3D spatial coordinates: {coord_error}")
                print("Continuing without additional coordinates...")
        
        return combined_ds

    @staticmethod
    def extract_2d_from_3d(data_3d: Union[np.ndarray, xr.DataArray],
                          level_index: Optional[int] = None,
                          level_value: Optional[float] = None,
                          level_dim: str = 'nVertLevels',
                          method: str = 'nearest') -> np.ndarray:
        """
        Extract a 2D horizontal slice from 3D atmospheric data at a specified vertical level for surface plotting. This static utility method enables visualization of 3D variables by selecting a single level using either direct indexing or value-based interpolation along vertical coordinates like pressure or height. It handles both xarray DataArrays with coordinate information and raw numpy arrays, supporting nearest-neighbor selection or linear interpolation methods. The method provides flexibility for plotting atmospheric fields at standard pressure levels (e.g., 850 hPa) or model levels, making analysis workflows data-type agnostic. Returns a 2D numpy array ready for contouring, mapping, or other horizontal visualization operations.

        Parameters:
            data_3d (Union[np.ndarray, xr.DataArray]): 3D atmospheric data array with dimensions like (nCells, nVertLevels, Time) or similar structure.
            level_index (Optional[int]): Zero-based index of vertical level to extract directly (default: None).
            level_value (Optional[float]): Coordinate value to search for along vertical dimension, e.g., pressure in hPa (default: None).
            level_dim (str): Name of the vertical dimension coordinate such as 'nVertLevels', 'pressure', or 'height' (default: 'nVertLevels').
            method (str): Interpolation method when using level_value - 'nearest' for nearest-neighbor or 'linear' for interpolation (default: 'nearest').

        Returns:
            np.ndarray: 2D horizontal slice extracted at the specified vertical level, ready for surface plotting.

        Raises:
            ValueError: If neither level_index nor level_value is provided, or if coordinate dimension is not found in xarray data.

        Examples:
            # Extract surface level by index
            surface_temp = MPAS3DProcessor.extract_2d_from_3d(temp_3d, level_index=0)
            
            # Extract 850hPa level by pressure value
            temp_850 = MPAS3DProcessor.extract_2d_from_3d(temp_3d, level_value=850, level_dim='pressure')
            
            # Use in surface plotting workflow
            plotter.create_surface_map(lon, lat, temp_850, 'temperature_850hpa', ...)
        """
        if level_index is None and level_value is None:
            raise ValueError("Must provide either level_index or level_value")
            
        if isinstance(data_3d, xr.DataArray):
            if level_index is not None:
                if level_dim in data_3d.sizes:
                    extracted = data_3d.isel({level_dim: level_index})
                else:
                    extracted = data_3d.isel({data_3d.sizes[1]: level_index})
                return extracted.values
            else:
                if level_dim in data_3d.coords:
                    coord_values = data_3d.coords[level_dim].values
                    if method == 'nearest':
                        closest_idx = np.argmin(np.abs(coord_values - level_value))
                        extracted = data_3d.isel({level_dim: closest_idx})
                    else:  
                        extracted = data_3d.interp({level_dim: level_value}, method='linear')
                    return extracted.values
                else:
                    raise ValueError(f"Coordinate '{level_dim}' not found in data array")
        
        else:  
            if level_index is not None:
                if data_3d.ndim >= 2:
                    if data_3d.ndim == 3:  
                        return data_3d[:, level_index, -1]
                    elif data_3d.ndim == 2:  
                        return data_3d[:, level_index]
                    else:
                        return data_3d[level_index]
                else:
                    raise ValueError("Data must be at least 2D for level extraction")
            else:
                raise ValueError("level_value extraction requires xarray.DataArray with coordinates")
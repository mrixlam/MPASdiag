#!/usr/bin/env python3

"""
MPASdiag Core Processing Module: 3D Atmospheric Data Processor

This module defines the MPAS3DProcessor class, which is a specialized processor for handling 3D atmospheric data from MPAS output files. It extends the base MPASBaseProcessor class to provide functionality specific to 3D fields, including methods for extracting horizontal coordinates, finding relevant MPAS output files, loading 3D datasets, identifying available 3D variables, and extracting data slices at specified vertical levels. The processor is designed to be flexible and robust, with support for various level specifications and error handling to guide users in working with complex 3D atmospheric datasets. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""
# Load standard libraries
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import uxarray as ux
import warnings
from typing import List, Tuple, Any, Optional, Union, cast

from .base import MPASBaseProcessor
from .utils_datetime import MPASDateTimeUtils
from .constants import MPASOUT_GLOB, DATASET_NOT_LOADED_3D_MSG


class MPAS3DProcessor(MPASBaseProcessor):
    """ Specialized processor for 3D MPAS atmospheric data. """
    
    def __init__(self: 'MPAS3DProcessor', 
                 grid_file: str, 
                 verbose: bool = True) -> None:
        """
        This constructor initializes the MPAS3DProcessor by calling the base class constructor with the provided grid file and verbosity settings. It sets up the processor to handle 3D atmospheric data from MPAS output files, ensuring that the necessary grid information is available for subsequent processing operations. The constructor does not perform any data loading or processing itself, but prepares the processor for use in methods that will extract coordinates, find files, load datasets, and manipulate 3D variables. 

        Parameters:
            grid_file (str): Path to the MPAS grid file containing spatial coordinate information.
            verbose (bool): If True, enables verbose output for debugging and informational purposes (default: True). 

        Returns:
            None
        """
        super().__init__(grid_file, verbose)
    
    def extract_2d_coordinates_for_variable(self: 'MPAS3DProcessor', 
                                            var_name: str, 
                                            data_array: Optional[xr.DataArray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method extracts the 2D horizontal coordinates (longitude and latitude) for a specified 3D atmospheric variable from the grid file. It determines the appropriate spatial dimension (e.g., 'nCells', 'nVertices', 'nEdges') based on the variable's dimensions or the provided data array, then looks for corresponding longitude and latitude variables in the grid dataset. The method handles different naming conventions for coordinate variables and ensures that the extracted coordinates are in degrees if they are originally in radians. The resulting longitude and latitude arrays are flattened and adjusted to be within the range of -180 to 180 degrees for longitude. If the necessary coordinates cannot be found, it raises a ValueError with information about available variables in the grid file. 

        Parameters:
            var_name (str): Name of the 3D atmospheric variable for which to extract coordinates.
            data_array (Optional[xr.DataArray]): Optional xarray DataArray containing the variable data, used to determine spatial dimensions if provided. 

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing longitude and latitude arrays for the specified variable. 
        """
        if self.dataset is None:
            raise ValueError(DATASET_NOT_LOADED_3D_MSG)
            
        try:
            with xr.open_dataset(self.grid_file, decode_times=False) as grid_ds:
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
    
    def find_mpasout_files(self: 'MPAS3DProcessor', 
                           data_dir: str) -> List[str]:
        """
        This method searches for MPAS output files in the specified directory using a predefined glob pattern. It first attempts to find files directly in the provided directory, then looks in a common subdirectory named 'mpasout' if no files are found. If still no files are found, it performs a recursive search through all subdirectories. The method ensures that at least two MPAS output files are found to allow for temporal analysis, and it provides verbose output about the number of files found and their names. If no files are found after all search attempts, it raises a FileNotFoundError with information about the search path. 

        Parameters:
            data_dir (str): Directory path to search for MPAS output files. 

        Returns:
            List[str]: List of file paths to the found MPAS output files. 
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

    def load_3d_data(self: 'MPAS3DProcessor', 
                     data_dir: str, 
                     use_pure_xarray: bool = False, 
                     reference_file: str = "") -> 'MPAS3DProcessor':
        """
        This method loads 3D atmospheric data from MPAS output files in the specified directory. It uses the find_mpasout_files method to locate the relevant files, then loads them into an xarray Dataset using either the pure xarray backend or UXarray based on the use_pure_xarray flag. The method applies appropriate chunking for efficient loading of 3D data and handles time coordinate ordering using an optional reference file. After loading, it adds spatial coordinates to the dataset using the add_spatial_coordinates method. The loaded dataset and its type are stored as attributes of the processor for use in subsequent processing steps. The method returns a self reference to allow for method chaining. 

        Parameters:
            data_dir (str): Directory path containing MPAS output files to load.
            use_pure_xarray (bool): If True, uses pure xarray for loading; if False, uses UXarray for potentially faster loading (default: False).
            reference_file (str): Optional file path to use as a reference for time coordinate ordering (default: "").

        Returns:
            MPAS3DProcessor: Returns self reference after loading the dataset and adding spatial coordinates. 
        """
        self.data_dir = data_dir
        
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

    def get_available_3d_variables(self: 'MPAS3DProcessor') -> List[str]:
        """
        This method identifies and returns a list of all 3D atmospheric variables available in the loaded dataset. It checks each variable in the dataset for the presence of vertical dimensions such as 'nVertLevels' or 'nVertLevelsP1' to determine if it is a 3D variable. The method provides verbose output about the number of 3D variables found and details about the first few variables, including their dimensions and units. If the dataset is not loaded, it raises a ValueError prompting the user to load the data first. 

        Parameters:
            None

        Returns:
            List[str]: List of variable names that are identified as 3D atmospheric variables based on their dimensions. 
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

    def get_3d_variable_data(self: 'MPAS3DProcessor', 
                             var_name: str, 
                             level: Union[str, int, float], 
                             time_index: int = 0) -> xr.DataArray:
        """
        This method extracts a 2D horizontal slice of data for a specified 3D atmospheric variable at a given vertical level and time index. It supports various level specifications, including integer model level indices, float pressure levels in Pa, and special string identifiers like 'surface' or 'top'. The method validates the input parameters, checks for the presence of the variable and its vertical structure, and handles the extraction logic accordingly. If a pressure level is specified, it performs interpolation between mean pressure levels if necessary. The resulting 2D DataArray contains the variable data at the specified level and time, with appropriate attributes added for metadata. The method also provides verbose output about the extraction process and the range of values in the extracted data. If any issues arise during extraction, it raises descriptive exceptions to guide the user. 

        Parameters:
            var_name (str): Name of the 3D atmospheric variable to extract.
            level (Union[str, int, float]): Vertical level specification, which can be an integer model level index, a float pressure level in Pa, or a string identifier like 'surface' or 'top'.
            time_index (int): Time index to extract data from (default: 0). 

        Returns:
            xr.DataArray: 2D DataArray containing the variable data at the specified vertical level and time index, with appropriate metadata attributes. 
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
                ds_raw = self._get_plain_dataset(self.dataset)
                pressure_p = ds_raw['pressure_p'].isel({time_dim: validated_time_index})
                pressure_base = ds_raw['pressure_base'].isel({time_dim: validated_time_index})
                total_pressure = pressure_p + pressure_base

                vert_dim = 'nVertLevels' if 'nVertLevels' in total_pressure.dims else 'nVertLevelsP1'
                horiz_dims = [d for d in total_pressure.dims if d != vert_dim]
                mean_pressure = total_pressure.mean(dim=horiz_dims) if horiz_dims else total_pressure
                mean_p_vals = np.asarray(mean_pressure.values).ravel()

                if level >= float(mean_p_vals.max()):
                    level_idx = 0
                    if self.verbose:
                        print(f"Requested pressure {level:.1f} Pa above surface mean; using surface level 0")
                elif level <= float(mean_p_vals.min()):
                    level_idx = len(mean_p_vals) - 1
                    if self.verbose:
                        print(f"Requested pressure {level:.1f} Pa below top mean; using top level {level_idx}")
                else:
                    lower_idx = int(np.argmax(mean_p_vals >= level))
                    if lower_idx >= len(mean_p_vals) - 1:
                        level_idx = lower_idx
                    else:
                        upper_idx = lower_idx + 1
                        p_lower = float(mean_p_vals[lower_idx])
                        p_upper = float(mean_p_vals[upper_idx])

                        if p_lower == p_upper:
                            w = 0.0
                        else:
                            w = (level - p_upper) / (p_lower - p_upper)

                        if self.verbose:
                            print(f"Requested pressure: {level:.1f} Pa, interpolating between mean levels {lower_idx} ({p_lower:.1f} Pa) and {upper_idx} ({p_upper:.1f} Pa) w={w:.3f}")

                        vertical_dim = 'nVertLevels' if 'nVertLevels' in self.dataset[var_name].sizes else 'nVertLevelsP1'

                        ds_raw = self._get_plain_dataset(self.dataset)
                        var_lower = ds_raw[var_name].isel({time_dim: validated_time_index, vertical_dim: lower_idx})
                        var_upper = ds_raw[var_name].isel({time_dim: validated_time_index, vertical_dim: upper_idx})

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
        
        ds = self._get_plain_dataset(self.dataset)
        var_data = ds[var_name].isel({time_dim: validated_time_index, vertical_dim: level_idx})
        
        if hasattr(var_data, 'compute'):
            var_data = cast(Any, var_data).compute()
        
        if hasattr(var_data, 'ndim') and var_data.ndim > 1:
            raise ValueError(
                f"Level extraction for {var_name} at level {level} produced {var_data.ndim}D data "
                f"with shape {var_data.shape}, expected 1D"
            )
        
        if hasattr(var_data, 'attrs'):
            var_data.attrs['selected_level'] = level
            var_data.attrs['level_index'] = level_idx
            
            if isinstance(level, float) and 'pressure_p' in self.dataset and 'pressure_base' in self.dataset:
                pressure_p = ds['pressure_p'].isel({time_dim: validated_time_index, vertical_dim: level_idx})
                pressure_base = ds['pressure_base'].isel({time_dim: validated_time_index, vertical_dim: level_idx})
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

    def get_vertical_levels(self: 'MPAS3DProcessor', 
                            var_name: str, 
                            return_pressure: bool = True, 
                            time_index: int = 0) -> List[Union[int, float]]:
        """
        This method retrieves the available vertical levels for a specified 3D atmospheric variable, either as zero-based model level indices or as pressure levels in Pa. It checks the dimensions of the variable to determine the vertical structure and then attempts to compute mean pressure levels using available pressure-related variables in the dataset. If return_pressure is True, it first tries to use the 'pressure' variable, then falls back to 'pressure_p' and 'pressure_base', and finally to 'fzp' and 'surface_pressure' if necessary. The method handles cases where pressure values may be non-finite or non-positive by providing warnings and attempting interpolation. If return_pressure is False, it simply returns the model level indices. The method provides verbose output about the identified vertical levels and their corresponding pressures if applicable. If any issues arise during the process, it raises descriptive exceptions to guide the user. 

        Parameters:
            var_name (str): Name of the 3D atmospheric variable for which to retrieve vertical levels.
            return_pressure (bool): If True, returns pressure levels in Pa; if False, returns model level indices (default: True).
            time_index (int): Time index to use for pressure level calculations when return_pressure is True (default: 0). 

        Returns:
            List[Union[int, float]]: List of vertical levels, either as model level indices or pressure levels in Pa, depending on the return_pressure flag. 
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
            ds_raw = self._get_plain_dataset(self.dataset)
            try:
                pressure_da = ds_raw['pressure'].isel({time_dim: validated_time_index})
            except Exception:
                pressure_da = ds_raw['pressure']

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
                    print(f"  Surface: {float(mean_pressure_levels[0]):.1f} Pa")
                    print(f"  Top: {float(mean_pressure_levels[-1]):.1f} Pa")

                return mean_pressure_levels.tolist()

        if return_pressure and 'pressure_p' in self.dataset and 'pressure_base' in self.dataset:
            time_dim, validated_time_index, _ = MPASDateTimeUtils.validate_time_parameters(self.dataset, time_index, self.verbose)

            ds_raw = self._get_plain_dataset(self.dataset)
            pressure_p = ds_raw['pressure_p'].isel({time_dim: validated_time_index})
            pressure_base = ds_raw['pressure_base'].isel({time_dim: validated_time_index})
            total_pressure = pressure_p + pressure_base

            mean_pressure_levels = total_pressure.mean(dim='nCells').values

            if vertical_dim == 'nVertLevelsP1':
                if num_levels == len(mean_pressure_levels) + 1:
                    mean_pressure_levels = np.append(mean_pressure_levels, mean_pressure_levels[-1] * 0.9)

            if self.verbose:
                print(f"Pressure levels for {var_name} ({num_levels} levels):")
                print(f"  Surface: {float(mean_pressure_levels[0]):.1f} Pa")
                print(f"  Top: {float(mean_pressure_levels[-1]):.1f} Pa")
                print(f"  Range: {float(mean_pressure_levels.min()):.1f} to {float(mean_pressure_levels.max()):.1f} Pa")

            return mean_pressure_levels.tolist()

        if return_pressure and 'fzp' in self.dataset and 'surface_pressure' in self.dataset:
            try:
                time_dim, validated_time_index, _ = MPASDateTimeUtils.validate_time_parameters(self.dataset, time_index, self.verbose)
                ds_raw = self._get_plain_dataset(self.dataset)
                fzp = ds_raw['fzp'].isel({time_dim: validated_time_index}).values
                sp = ds_raw['surface_pressure'].isel({time_dim: validated_time_index}).values

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)
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
                    print(f"  Surface (mean): {float(mean_sp):.1f} Pa")
                    print(f"  Top: {float(mean_pressure_levels[-1]):.1f} Pa")

                return mean_pressure_levels.tolist()
            except Exception as e:
                if self.verbose:
                    print(f"Warning: failed to reconstruct pressure levels from hybrid coefficients: {e}")

        level_indices: List[Union[int, float]] = list(range(num_levels))

        if self.verbose:
            print(f"Model levels for {var_name}: {num_levels} levels (indices 0-{num_levels-1})")

        return level_indices

    def add_spatial_coordinates(self: 'MPAS3DProcessor', 
                                combined_ds: xr.Dataset) -> xr.Dataset:
        """
        This method adds spatial coordinate variables to the provided combined xarray dataset containing 3D atmospheric data from multiple timesteps. It identifies the necessary dimensions to add based on typical MPAS grid structures, such as 'nCells', 'nVertLevels', 'nVertLevelsP1', 'nEdges', 'nVertices', and 'nSoilLevels'. It also identifies the relevant spatial coordinate variables for latitude and longitude at different grid points (e.g., 'latCell', 'lonCell', 'latVertex', 'lonVertex') and adds them to the dataset. The method ensures that the added coordinates are consistent with the dimensions of the dataset and provides verbose output about the added coordinates and their shapes. The resulting enriched dataset is returned for use in subsequent processing steps, such as plotting or analysis of 3D atmospheric fields. 

        Parameters:
            combined_ds (xr.Dataset): The combined xarray Dataset containing 3D atmospheric data from multiple timesteps, to which spatial coordinates will be added. 

        Returns:
            xr.Dataset: The enriched xarray Dataset with added spatial coordinate variables for longitude and latitude, consistent with the dataset's dimensions. 
        """
        dimensions_to_add = [
            'nCells', 'nVertLevels', 'nVertLevelsP1', 
            'nEdges', 'nVertices', 'nSoilLevels'
        ]

        spatial_vars = ['latCell', 'lonCell', 'latVertex', 'lonVertex']
        
        return self._add_spatial_coords_helper(
            combined_ds, dimensions_to_add, spatial_vars, "3D"
        )

    @staticmethod
    def extract_2d_from_3d(data_3d: Union[np.ndarray, xr.DataArray], 
                           level_index: Optional[int] = None, 
                           level_value: Optional[float] = None, 
                           level_dim: str = 'nVertLevels', 
                           method: str = 'nearest') -> np.ndarray:
        """
        This static method extracts a 2D horizontal slice from a given 3D atmospheric data array at a specified vertical level. The vertical level can be specified either as a zero-based index (level_index) or as a coordinate value (level_value) along a specified vertical dimension (level_dim). When using level_value, the method supports both nearest-neighbor and linear interpolation methods to find the appropriate level. The method handles both numpy arrays and xarray DataArrays, ensuring that the correct dimensions are used for extraction. The resulting 2D array contains the horizontal slice of data at the specified vertical level, which can be used for surface plotting or further analysis. If the necessary parameters are not provided or if there are issues with the data structure, the method raises descriptive exceptions to guide the user. 

        Parameters:
            data_3d (Union[np.ndarray, xr.DataArray]): The input 3D data array from which to extract the 2D slice.
            level_index (Optional[int]): The zero-based index of the vertical level to extract (e.g., 0 for surface, -1 for top). Mutually exclusive with level_value.
            level_value (Optional[float]): The coordinate value along the vertical dimension to extract (e.g., pressure in Pa). Mutually exclusive with level_index.
            level_dim (str): The name of the vertical dimension in the data array (default: 'nVertLevels').
            method (str): The interpolation method to use when extracting by level_value ('nearest' or 'linear', default: 'nearest'). 

        Returns:
            np.ndarray: A 2D numpy array containing the extracted horizontal slice of data at the specified vertical level. 
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
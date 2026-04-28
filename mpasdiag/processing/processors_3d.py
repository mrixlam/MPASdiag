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
import glob
import os
import numpy as np
import xarray as xr
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
    
    
    @staticmethod
    def _detect_spatial_dim(sizes: Any) -> str:
        """
        This helper method detects the spatial dimension (e.g., 'nVertices', 'nCells', 'nEdges') based on the provided sizes dictionary, which typically comes from the dimensions of a DataArray or Dataset. It checks for the presence of known spatial dimensions in a specific order and returns the first one found. If none of the expected spatial dimensions are present, it defaults to returning 'nCells'. This method is used to determine which set of coordinate variables to look for when extracting horizontal coordinates from the grid dataset.

        Parameters:
            sizes (Any): A dictionary-like object containing dimension names and their sizes, typically from an xarray DataArray or Dataset.

        Returns:
            str: The name of the detected spatial dimension, which can be 'nVertices', 'nCells', or 'nEdges'. Defaults to 'nCells' if none are found.
        """
        for dim in ('nVertices', 'nCells', 'nEdges'):
            if dim in sizes:
                return dim
        return 'nCells'


    _COORD_NAMES: dict = {
        'nVertices': (['lonVertex', 'lon_vertex', 'longitude_vertex'],
                      ['latVertex', 'lat_vertex', 'latitude_vertex']),
        'nEdges':    (['lonEdge',   'lon_edge',   'longitude_edge'],
                      ['latEdge',   'lat_edge',   'latitude_edge']),
        'nCells':    (['lonCell',   'longitude',  'lon'],
                      ['latCell',   'latitude',   'lat']),
    }


    @staticmethod
    def _lookup_coord(grid_ds: xr.Dataset, 
                      names: List[str]) -> Optional[np.ndarray]:
        """
        Return values of the first name found as a coord or data var, or None.

        Parameters:
            grid_ds (xr.Dataset): The xarray Dataset containing the grid data.
            names (List[str]): A list of possible coordinate variable names to look for.

        Returns:
            Optional[np.ndarray]: The values of the first found coordinate variable, or None if none are found.
        """
        for name in names:
            if name in grid_ds.coords or name in grid_ds.data_vars:
                return grid_ds[name].values
        return None


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
            # Determine which coordinate variables we need before opening the grid file
            sizes = data_array.sizes if data_array is not None else (
                self.dataset[var_name].sizes if var_name in self.dataset else {}
            )
            spatial_dim = self._detect_spatial_dim(sizes)
            lon_names, lat_names = self._COORD_NAMES.get(spatial_dim, self._COORD_NAMES['nCells'])
            needed_vars = list(lon_names) + list(lat_names)
            
            # Build drop_variables list to avoid loading unneeded grid data
            open_kwargs: dict = {'decode_times': False}
            try:
                with xr.open_dataset(self.grid_file, decode_times=False) as probe:
                    all_vars = list(probe.data_vars)
                vars_to_drop = [v for v in all_vars if v not in needed_vars]
                if vars_to_drop:
                    open_kwargs['drop_variables'] = vars_to_drop
            except Exception:
                pass
            
            with xr.open_dataset(self.grid_file, **open_kwargs) as grid_ds:
                lon_coords = self._lookup_coord(grid_ds, lon_names)
                lat_coords = self._lookup_coord(grid_ds, lat_names)

                if lon_coords is None or lat_coords is None:
                    available_vars = list(grid_ds.coords.keys()) + list(grid_ds.data_vars.keys())
                    raise ValueError(
                        f"Could not find {spatial_dim} coordinates in grid file. "
                        f"Available variables: {available_vars}"
                    )

                if np.nanmax(np.abs(lat_coords)) <= np.pi:
                    lat_coords = lat_coords * 180.0 / np.pi
                    lon_coords = lon_coords * 180.0 / np.pi

                lon_coords = ((lon_coords.ravel() + 180) % 360) - 180
                lat_coords = lat_coords.ravel()

                if self.verbose:
                    print(f"Extracted {spatial_dim} coordinates for 3D variable {var_name}: {len(lon_coords):,} points")

                return lon_coords, lat_coords

        except Exception as e:
            raise RuntimeError(f"Error loading coordinates from grid file {self.grid_file}: {e}")
    
    def _find_files_recursive(self: 'MPAS3DProcessor',
                               data_dir: str) -> List[str]:
        """
        This helper method performs a recursive search for MPAS output files in the specified directory and all its subdirectories using a predefined glob pattern. It collects all matching files, sorts them, and checks that at least two files are found to allow for temporal analysis. If no files are found, it raises a FileNotFoundError with information about the search path. If files are found, it provides verbose output about the number of files and their names before returning the list of file paths.

        Parameters:
            data_dir (str): The directory path to search for MPAS output files.

        Returns:
            List[str]: A sorted list of file paths to the found MPAS output files.
        """
        files = sorted(glob.glob(os.path.join(data_dir, "**", MPASOUT_GLOB), recursive=True))

        if not files:
            raise FileNotFoundError(f"No MPAS output files found under: {data_dir}")

        if len(files) < 2:
            raise ValueError(
                f"Insufficient MPAS output files for temporal analysis. Found {len(files)}, need at least 2."
            )

        if self.verbose:
            print(f"\nFound {len(files)} MPAS output files (recursive search):")
            for i, filepath in enumerate(files[:5]):
                print(f"  {i+1}: {os.path.basename(filepath)}")

        return files

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
                return self._find_files_recursive(data_dir)

    def load_3d_data(self: 'MPAS3DProcessor', 
                     data_dir: str, 
                     use_pure_xarray: bool = False, 
                     reference_file: str = "",
                     variables: Optional[List[str]] = None) -> 'MPAS3DProcessor':
        """
        This method loads 3D atmospheric data from MPAS output files in the specified directory. It uses the find_mpasout_files method to locate the relevant files, then loads them into an xarray Dataset using either the pure xarray backend or UXarray based on the use_pure_xarray flag. The method applies appropriate chunking for efficient loading of 3D data and handles time coordinate ordering using an optional reference file. After loading, it adds spatial coordinates to the dataset using the add_spatial_coordinates method. The loaded dataset and its type are stored as attributes of the processor for use in subsequent processing steps. The method returns a self reference to allow for method chaining. 

        Parameters:
            data_dir (str): Directory path containing MPAS output files to load.
            use_pure_xarray (bool): If True, uses pure xarray for loading; if False, uses UXarray for potentially faster loading (default: False).
            reference_file (str): Optional file path to use as a reference for time coordinate ordering (default: "").
            variables (Optional[List[str]]): List of variable names to retain. If provided, only these variables are loaded from data files to reduce memory usage (default: None).

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
            data_type_label="3D",
            variables=variables
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

    def _validate_3d_variable(self: 'MPAS3DProcessor', 
                              var_name: str) -> None:
        """
        This helper method validates that the specified variable name corresponds to a 3D atmospheric variable in the loaded dataset. It checks that the dataset is loaded, that the variable exists in the dataset, and that it has the necessary vertical dimensions to be considered a 3D variable. If any of these conditions are not met, it raises a ValueError with an appropriate message to guide the user. 

        Parameters:
            var_name (str): Name of the variable to validate as a 3D atmospheric variable.

        Returns:
            None
        """
        if self.dataset is None:
            raise ValueError(DATASET_NOT_LOADED_3D_MSG)
        if var_name not in self.dataset.data_vars:
            available_vars = list(self.dataset.data_vars.keys())
            raise ValueError(
                f"Variable '{var_name}' not found. Available variables: {available_vars[:20]}..."
            )
        if 'nVertLevels' not in self.dataset[var_name].sizes and \
                'nVertLevelsP1' not in self.dataset[var_name].sizes:
            raise ValueError(f"Variable '{var_name}' is not a 3D atmospheric variable")

    def _resolve_int_level(self: 'MPAS3DProcessor', 
                           level: int) -> int:
        """
        This helper method validates that the provided integer level index is within the bounds of available model levels in the dataset. It checks the maximum number of vertical levels based on the presence of 'nVertLevels' or 'nVertLevelsP1' dimensions and raises a ValueError if the specified level index exceeds the available levels. If the level index is valid, it returns the index for use in data extraction. 

        Parameters:
            level (int): The model level index to validate.

        Returns:
            int: The validated model level index.
        """
        max_levels = self.dataset.sizes.get('nVertLevels', self.dataset.sizes.get('nVertLevelsP1', 0))
        if level >= max_levels:
            raise ValueError(f"Model level {level} exceeds available levels {max_levels}")
        return level

    def _resolve_str_level(self: 'MPAS3DProcessor', 
                           level: str) -> int:
        """
        This helper method resolves special string level specifications to corresponding model level indices. It recognizes 'surface' as level index 0 and 'top' as the highest model level index based on the dataset's vertical structure. If an unrecognized string is provided, it raises a ValueError indicating the unknown level specification. 

        Parameters:
            level (str): The level specification as a string ('surface' or 'top').

        Returns:
            int: The corresponding model level index.
        """
        if level.lower() == 'surface':
            return 0
        if level.lower() == 'top':
            return self.dataset.sizes.get('nVertLevels', self.dataset.sizes.get('nVertLevelsP1', 1)) - 1
        raise ValueError(f"Unknown level specification: {level}")

    def _compute_mean_pressure_levels(self: 'MPAS3DProcessor',
                                      ds_raw: xr.Dataset,
                                      time_dim: str,
                                      time_idx: int,) -> np.ndarray:
        """
        This helper method computes the column-mean total pressure values for each model level using the 'pressure_p' and 'pressure_base' variables in the raw dataset. It identifies the vertical dimension, averages over horizontal dimensions, and returns a 1D array of mean pressure values for use in pressure-level interpolation. 

        Parameters:
            ds_raw (xr.Dataset): The raw dataset containing pressure variables.
            time_dim (str): The name of the time dimension in the dataset.
            time_idx (int): The index of the time step to extract.

        Returns:
            np.ndarray: A 1D array of column-mean total pressure values for each model level.
        """
        pressure_p = ds_raw['pressure_p'].isel({time_dim: time_idx})
        pressure_base = ds_raw['pressure_base'].isel({time_dim: time_idx})
        total_pressure = pressure_p + pressure_base
        vertical_dim = 'nVertLevels' if 'nVertLevels' in total_pressure.dims else 'nVertLevelsP1'
        horizontal_dims = [d for d in total_pressure.dims if d != vertical_dim]
        mean_pressure = total_pressure.mean(dim=horizontal_dims) if horizontal_dims else total_pressure
        return np.asarray(mean_pressure.values).ravel()

    def _lerp_variable(self: 'MPAS3DProcessor',
                       var_name: str,
                       lower_idx: int,
                       upper_idx: int,
                       w: float,
                       mean_p_vals: np.ndarray,
                       level: float,
                       ds_raw: xr.Dataset,
                       time_dim: str,
                       time_idx: int,
                       vertical_dim: str) -> Tuple[int, Optional[xr.DataArray]]:
        """
        This helper method performs linear interpolation of the specified variable between two vertical levels based on the provided interpolation weight. It extracts the variable values at the lower and upper bounding levels, computes the interpolated field, and attaches appropriate attributes. If interpolation fails for any reason, it falls back to returning the nearest mean level index. The method returns a tuple containing either the index of the selected level or -1 if interpolation was successful, along with the interpolated DataArray if applicable.

        Parameters:
            var_name (str): The name of the variable to interpolate.
            lower_idx (int): The index of the lower bounding level.
            upper_idx (int): The index of the upper bounding level.
            w (float): The interpolation weight between the lower and upper levels.
            mean_p_vals (np.ndarray): The column-mean pressure values for each model level.
            level (float): The target pressure value for interpolation.
            ds_raw (xr.Dataset): The raw dataset containing the variable.
            time_dim (str): The name of the time dimension in the dataset.
            time_idx (int): The index of the time step to extract.
            vertical_dim (str): The name of the vertical dimension in the dataset.

        Returns:
            Tuple[int, Optional[xr.DataArray]]: A tuple containing the level index and the interpolated DataArray (or None if index-based extraction is used).
        """
        var_lower = ds_raw[var_name].isel({time_dim: time_idx, vertical_dim: lower_idx})
        var_upper = ds_raw[var_name].isel({time_dim: time_idx, vertical_dim: upper_idx})
        try:
            interp_field = (1.0 - w) * var_lower + w * var_upper
            if hasattr(interp_field, 'compute'):
                interp_field = cast(Any, interp_field).compute()
            interp_field.attrs = getattr(var_lower, 'attrs', {})
            if self.verbose:
                finite_values = interp_field.values.flatten()
                finite_values = finite_values[np.isfinite(finite_values)]
                if len(finite_values) > 0:
                    print(f"Interpolated field range: {finite_values.min():.3f} to {finite_values.max():.3f}")
            return -1, interp_field
        except Exception:
            level_idx = int(np.argmin(np.abs(mean_p_vals - level)))
            if self.verbose:
                print(f"Interpolation failed, falling back to nearest mean level {level_idx}")
            return level_idx, None

    def _interpolate_at_pressure(self: 'MPAS3DProcessor',
                                 var_name: str,
                                 level: float,
                                 mean_p_vals: np.ndarray,
                                 ds_raw: xr.Dataset,
                                 time_dim: str,
                                 time_idx: int,) -> Tuple[int, Optional[xr.DataArray]]:
        """
        This helper method performs interpolation of the specified variable at a given pressure level using the provided mean pressure values for each model level. It checks if the requested pressure level is outside the range of mean pressures and handles those cases by returning the nearest surface or top level. If interpolation is needed, it identifies the bounding levels, computes the interpolation weight, and attempts to interpolate the variable between those levels. If interpolation fails for any reason, it falls back to returning the nearest mean level index. The method returns a tuple containing either the index of the selected level or -1 if interpolation was successful, along with the interpolated DataArray if applicable. 

        Parameters:
            var_name (str): The name of the variable to interpolate.
            level (float): The target pressure value.
            mean_p_vals (np.ndarray): The column-mean pressure values for each model level.
            ds_raw (xr.Dataset): The raw dataset containing the variable.
            time_dim (str): The name of the time dimension in the dataset.
            time_idx (int): The index of the time step to extract.

        Returns:
            Tuple[int, Optional[xr.DataArray]]: A tuple containing the level index and the interpolated DataArray (or None if index-based extraction is used).
        """
        if level >= float(mean_p_vals.max()):
            if self.verbose:
                print(f"Requested pressure {level:.1f} Pa above surface mean; using surface level 0")
            return 0, None

        if level <= float(mean_p_vals.min()):
            level_idx = len(mean_p_vals) - 1
            if self.verbose:
                print(f"Requested pressure {level:.1f} Pa below top mean; using top level {level_idx}")
            return level_idx, None

        lower_idx = int(np.argmax(mean_p_vals >= level))

        if lower_idx >= len(mean_p_vals) - 1:
            return lower_idx, None

        upper_idx = lower_idx + 1
        p_lower = float(mean_p_vals[lower_idx])
        p_upper = float(mean_p_vals[upper_idx])
        w = 0.0 if p_lower == p_upper else (level - p_upper) / (p_lower - p_upper)

        if self.verbose:
            print(
                f"Requested pressure: {level:.1f} Pa, interpolating between mean levels "
                f"{lower_idx} ({p_lower:.1f} Pa) and {upper_idx} ({p_upper:.1f} Pa) w={w:.3f}"
            )

        vertical_dim = 'nVertLevels' if 'nVertLevels' in self.dataset[var_name].sizes else 'nVertLevelsP1'

        return self._lerp_variable(
            var_name, lower_idx, upper_idx, w, mean_p_vals, level,
            ds_raw, time_dim, time_idx, vertical_dim
        )

    def _resolve_float_level(self: 'MPAS3DProcessor',
                             var_name: str,
                             level: float,
                             time_dim: str,
                             time_idx: int,) -> Tuple[int, Optional[xr.DataArray]]:
        """
        This helper method resolves a float level specification as a pressure level by computing mean pressure levels and performing interpolation if necessary. It checks for the presence of required pressure variables, computes mean pressures, and calls the interpolation helper to determine the appropriate level index or interpolated field. If pressure data is not available, it raises a ValueError indicating that pressure-level interpolation cannot be performed. 

        Parameters:
            var_name (str): The name of the variable to extract at the specified pressure level.
            level (float): The target pressure value in Pa.
            time_dim (str): The name of the time dimension in the dataset.
            time_idx (int): The index of the time step to extract.

        Returns:
            Tuple[int, Optional[xr.DataArray]]: A tuple containing the level index and the interpolated DataArray (or None if index-based extraction is used). 
        """
        if 'pressure_p' not in self.dataset or 'pressure_base' not in self.dataset:
            raise ValueError("Cannot find pressure level - pressure data not available")

        ds_raw = self._get_plain_dataset(self.dataset)
        mean_p_vals = self._compute_mean_pressure_levels(ds_raw, time_dim, time_idx)

        return self._interpolate_at_pressure(var_name, level, mean_p_vals, ds_raw, time_dim, time_idx)

    def _resolve_level_index(self: 'MPAS3DProcessor',
                             var_name: str,
                             level: Union[str, int, float],
                             time_dim: str,
                             time_idx: int,) -> Tuple[int, Optional[xr.DataArray]]:
        """
        This helper method resolves the provided level specification, which can be an integer model level index, a float pressure level in Pa, or a string identifier like 'surface' or 'top'. It determines the appropriate resolution method based on the type of the level specification and calls the corresponding helper method to obtain the level index or interpolated field. If the level specification is invalid, it raises a ValueError indicating the issue. 

        Parameters:
            var_name (str): The name of the variable to extract.
            level (Union[str, int, float]): The level specification.
            time_dim (str): The name of the time dimension in the dataset.
            time_idx (int): The index of the time step to extract.

        Returns:
            Tuple[int, Optional[xr.DataArray]]: A tuple containing the level index and the extracted or interpolated DataArray (or None if index-based extraction is used).
        """
        if isinstance(level, int):
            return self._resolve_int_level(level), None

        if isinstance(level, float):
            return self._resolve_float_level(var_name, level, time_dim, time_idx)

        if isinstance(level, str):
            return self._resolve_str_level(level), None

        raise ValueError(f"Invalid level specification: {level}")

    def _set_level_attrs(self: 'MPAS3DProcessor',
                         var_data: xr.DataArray,
                         level: Union[str, int, float],
                         level_idx: int,
                         ds: xr.Dataset,
                         time_dim: str,
                         time_idx: int,
                         vertical_dim: str,) -> None:
        """
        This helper method sets attributes on the extracted variable data to indicate the selected level and its index. If the level was specified as a float pressure level and the dataset contains pressure variables, it also computes and adds an attribute for the actual pressure level in Pa. This metadata can be useful for tracking the level information in subsequent processing steps or when analyzing the extracted data. 

        Parameters:
            var_data (xr.DataArray): The DataArray to attach metadata to.
            level (Union[str, int, float]): The level specification.
            level_idx (int): The resolved level index.
            ds (xr.Dataset): The dataset containing the variable.
            time_dim (str): The name of the time dimension in the dataset.
            time_idx (int): The index of the time step.
            vertical_dim (str): The name of the vertical dimension in the dataset.
        
        Returns:
            None
        """
        if not hasattr(var_data, 'attrs'):
            return

        var_data.attrs['selected_level'] = level
        var_data.attrs['level_index'] = level_idx

        if isinstance(level, float) and 'pressure_p' in self.dataset and 'pressure_base' in self.dataset:
            pressure_p = ds['pressure_p'].isel({time_dim: time_idx, vertical_dim: level_idx})
            pressure_base = ds['pressure_base'].isel({time_dim: time_idx, vertical_dim: level_idx})
            actual_pressure = (pressure_p + pressure_base).mean().values
            var_data.attrs['actual_pressure_level'] = f"{actual_pressure:.1f} Pa"

    def _log_extracted_range(self: 'MPAS3DProcessor',
                             var_name: str,
                             level: Union[str, int, float],
                             var_data: xr.DataArray,) -> None:
        """
        This helper method logs the range of values in the extracted variable data for a given variable name and level specification. It checks if the variable data has finite values and prints the minimum and maximum values along with the units if available. If no finite values are found, it prints a warning message. This method is useful for providing immediate feedback about the extracted data and can help identify any issues with the extraction process or the data itself. 

        Parameters:
            var_name (str): The name of the variable.
            level (Union[str, int, float]): The level specification.
            var_data (xr.DataArray): The extracted DataArray.

        Returns:
            None
        """
        if not self.verbose or not hasattr(var_data, 'values'):
            return

        finite_values = var_data.values.flatten()
        finite_values = finite_values[np.isfinite(finite_values)]

        if len(finite_values) > 0:
            print(f"Variable {var_name} at level {level} range: {finite_values.min():.3f} to {finite_values.max():.3f}")
            if hasattr(var_data, 'attrs') and 'units' in var_data.attrs:
                print(f"Units: {var_data.attrs['units']}")
        else:
            print(f"Warning: No finite values found for {var_name} at level {level}")

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
        self._validate_3d_variable(var_name)

        time_dim, validated_time_index, _ = MPASDateTimeUtils.validate_time_parameters(
            self.dataset, time_index, self.verbose
        )

        level_idx, interp_result = self._resolve_level_index(
            var_name, level, time_dim, validated_time_index
        )

        if interp_result is not None:
            self._log_extracted_range(var_name, level, interp_result)
            return interp_result

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

        self._set_level_attrs(var_data, level, level_idx, ds, time_dim, validated_time_index, vertical_dim)
        self._log_extracted_range(var_name, level, var_data)
        return var_data

    @staticmethod
    def _pad_p1_levels(levels: np.ndarray, 
                       num_levels: int, 
                       vertical_dim: str) -> np.ndarray:
        """
        This helper method pads the levels array for 'nVertLevelsP1' if it has one fewer level than expected by appending a value slightly below the last level. This is necessary because 'nVertLevelsP1' typically includes an extra level at the surface, and if the computed levels only include the model levels, we need to add a surface level to match the expected count. The method checks if the vertical dimension is 'nVertLevelsP1' and if the number of levels is one less than expected, then appends a new level that is 10% below the last level. If these conditions are not met, it returns the original levels array unchanged.

        Parameters:
            levels (np.ndarray): The array of pressure levels to potentially pad.
            num_levels (int): The expected number of levels based on the dataset's vertical dimension.
            vertical_dim (str): The name of the vertical dimension ('nVertLevels' or 'nVertLevelsP1').

        Returns:
            np.ndarray: The potentially padded array of pressure levels, with an additional level added if necessary for 'nVertLevelsP1'.
        """
        if vertical_dim == 'nVertLevelsP1' and num_levels == len(levels) + 1:
            return np.append(levels, levels[-1] * 0.9)
        return levels

    def _get_vertical_dim(self: 'MPAS3DProcessor', 
                          var_name: str) -> Tuple[str, int]:
        """
        This helper method determines the vertical dimension name and the number of levels for a given variable based on its dimensions in the dataset. It checks for the presence of 'nVertLevels' or 'nVertLevelsP1' in the variable's dimensions and returns the corresponding dimension name and size. If neither vertical dimension is found, it raises a ValueError indicating that the variable is not a 3D atmospheric variable.

        Parameters:
            var_name (str): The name of the variable for which to determine the vertical dimension.

        Returns:
            Tuple[str, int]: A tuple containing the name of the vertical dimension and the number of levels in that dimension.
        """
        var_dims = self.dataset[var_name].sizes

        if 'nVertLevels' in var_dims:
            return 'nVertLevels', self.dataset.sizes['nVertLevels']

        if 'nVertLevelsP1' in var_dims:
            return 'nVertLevelsP1', self.dataset.sizes['nVertLevelsP1']

        raise ValueError(f"Variable '{var_name}' is not a 3D atmospheric variable")

    def _pressure_levels_from_pressure_var(self: 'MPAS3DProcessor',
                                           var_name: str,
                                           vertical_dim: str,
                                           num_levels: int,
                                           time_dim: str,
                                           time_idx: int,) -> Optional[np.ndarray]:
        """
        This helper method attempts to compute pressure levels directly from the 'pressure' variable in the dataset by averaging over horizontal dimensions. It checks for the presence of the 'pressure' variable, extracts it at the specified time index, and computes the mean pressure levels. If the computed levels contain non-finite or non-positive values, it issues a warning and returns None to indicate that this method failed, allowing the caller to fall back to other methods for determining pressure levels. If successful, it pads the levels if necessary and provides verbose output about the surface and top pressure levels before returning the array of pressure levels.

        Parameters:
            var_name (str): The name of the variable for which to compute pressure levels.
            vertical_dim (str): The name of the vertical dimension in the dataset.
            num_levels (int): The expected number of vertical levels based on the dataset.
            time_dim (str): The name of the time dimension in the dataset.
            time_idx (int): The index of the time step to extract.

        Returns:
            Optional[np.ndarray]: An array of pressure levels if successful, or None if the method failed due to non-finite or non-positive values.
        """
        ds_raw = self._get_plain_dataset(self.dataset)

        try:
            pressure_da = ds_raw['pressure'].isel({time_dim: time_idx})
        except Exception:
            pressure_da = ds_raw['pressure']

        levels = np.asarray(pressure_da.mean(dim='nCells').values, dtype=float).ravel()

        if not np.all(np.isfinite(levels)) or np.nanmin(levels) <= 0:
            if self.verbose:
                print(
                    "Warning: computed mean pressure levels from 'pressure' contain "
                    "non-positive or non-finite values; falling back to other methods"
                )
            return None

        levels = self._pad_p1_levels(levels, num_levels, vertical_dim)

        if self.verbose:
            print(f"Pressure levels for {var_name} ({num_levels} levels) from 'pressure' variable:")
            print(f"  Surface: {float(levels[0]):.1f} Pa")
            print(f"  Top: {float(levels[-1]):.1f} Pa")

        return levels

    def _pressure_levels_from_perturbation(self: 'MPAS3DProcessor',
                                           var_name: str,
                                           vertical_dim: str,
                                           num_levels: int,
                                           time_dim: str,
                                           time_idx: int,) -> np.ndarray:
        """
        This helper method computes pressure levels by summing the 'pressure_p' and 'pressure_base' variables at the specified time index, averaging over horizontal dimensions, and returning the resulting mean pressure levels. It assumes that these two variables together represent the total pressure at each model level. The method pads the levels if necessary and provides verbose output about the surface and top pressure levels before returning the array of pressure levels. This method is used as a fallback if the direct 'pressure' variable is not available or contains invalid values.

        Parameters:
            var_name (str): The name of the variable for which to compute pressure levels.
            vertical_dim (str): The name of the vertical dimension in the dataset.
            num_levels (int): The expected number of vertical levels based on the dataset.
            time_dim (str): The name of the time dimension in the dataset.
            time_idx (int): The index of the time step to extract.

        Returns:
            np.ndarray: An array of pressure levels computed from the perturbation and base pressure variables. 
        """
        ds_raw = self._get_plain_dataset(self.dataset)

        total_pressure = (
            ds_raw['pressure_p'].isel({time_dim: time_idx})
            + ds_raw['pressure_base'].isel({time_dim: time_idx})
        )

        levels = np.asarray(total_pressure.mean(dim='nCells').values, dtype=float).ravel()
        levels = self._pad_p1_levels(levels, num_levels, vertical_dim)

        if self.verbose:
            print(f"Pressure levels for {var_name} ({num_levels} levels):")
            print(f"  Surface: {float(levels[0]):.1f} Pa")
            print(f"  Top: {float(levels[-1]):.1f} Pa")
            print(f"  Range: {float(levels.min()):.1f} to {float(levels.max()):.1f} Pa")

        return levels

    def _repair_pressure_levels(self: 'MPAS3DProcessor', 
                                levels: np.ndarray, 
                                mean_sp: float) -> np.ndarray:
        """
        This helper method attempts to repair an array of pressure levels that contains non-finite or non-positive values by performing interpolation. It identifies the indices of valid levels and uses numpy's interpolation function to fill in the missing or invalid values based on the valid levels. If there are not enough valid levels to perform interpolation, it falls back to creating a logarithmic spacing of pressure levels between the mean surface pressure and 1 Pa. This method is used to ensure that we have a complete set of pressure levels for interpolation even when the original data contains issues. 

        Parameters:
            levels (np.ndarray): The array of pressure levels to be repaired.
            mean_sp (float): The mean surface pressure used for reconstruction if needed.

        Returns:
            np.ndarray: The repaired array of pressure levels.
        """
        all_level_indices = np.arange(len(levels))
        valid_level_indices = np.nonzero(np.isfinite(levels) & (levels > 0))[0]

        if valid_level_indices.size >= 2:
            return np.interp(all_level_indices, valid_level_indices, levels[valid_level_indices])

        if valid_level_indices.size == 1:
            return np.linspace(mean_sp, levels[valid_level_indices[0]], len(levels))

        return np.logspace(np.log10(mean_sp), np.log10(1.0), len(levels))

    def _pressure_levels_from_hybrid_coeffs(self: 'MPAS3DProcessor',
                                            var_name: str,
                                            vertical_dim: str,
                                            num_levels: int,
                                            time_dim: str,
                                            time_idx: int,) -> Optional[np.ndarray]:
        """
        This helper method attempts to reconstruct pressure levels from the hybrid vertical coordinate coefficients 'fzp' and 'surface_pressure'. It extracts the 'fzp' coefficients at the specified time index, multiplies them by the mean surface pressure to compute the pressure levels, and checks for validity. If the computed levels contain non-finite or non-positive values, it calls the repair method to attempt to fix them. The method pads the levels if necessary and provides verbose output about the reconstructed pressure levels before returning the array of pressure levels. If any issues arise during this process, it catches exceptions and returns None to indicate that this method failed, allowing the caller to fall back to other methods for determining pressure levels.

        Parameters:
            var_name (str): The name of the variable for which to compute pressure levels.
            vertical_dim (str): The name of the vertical dimension in the dataset.
            num_levels (int): The expected number of vertical levels based on the dataset.
            time_dim (str): The name of the time dimension in the dataset.
            time_idx (int): The index of the time step for which to compute pressure levels.

        Returns:
            np.ndarray or None: The array of reconstructed pressure levels, or None if reconstruction failed.
        """
        try:
            ds_raw = self._get_plain_dataset(self.dataset)
            fzp = ds_raw['fzp'].isel({time_dim: time_idx}).values
            surface_pressure_vals = ds_raw['surface_pressure'].isel({time_dim: time_idx}).values

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)
                mean_surface_pressure = float(np.nanmean(surface_pressure_vals))

            levels = np.asarray(fzp, dtype=float).ravel() * mean_surface_pressure

            if not np.all(np.isfinite(levels)) or np.nanmin(levels) <= 0:
                levels = self._repair_pressure_levels(levels, mean_surface_pressure)

            levels = self._pad_p1_levels(levels, num_levels, vertical_dim)

            if self.verbose:
                print(f"Reconstructed pressure levels from hybrid coefficients for {var_name} ({num_levels} levels):")
                print(f"  Surface (mean): {mean_surface_pressure:.1f} Pa")
                print(f"  Top: {float(levels[-1]):.1f} Pa")

            return levels
        except Exception as exc:
            if self.verbose:
                print(f"Warning: failed to reconstruct pressure levels from hybrid coefficients: {exc}")
            return None

    def _resolve_pressure_levels(self: 'MPAS3DProcessor',
                                 var_name: str,
                                 vertical_dim: str,
                                 num_levels: int,
                                 time_dim: str,
                                 time_idx: int) -> Optional[List[Union[int, float]]]:
        """
        This helper method attempts to resolve pressure levels for a given variable by trying multiple methods in order of preference. It first checks for the presence of the 'pressure' variable and attempts to compute mean pressure levels from it. If that fails, it checks for the presence of 'pressure_p' and 'pressure_base' variables and computes pressure levels from their sum. If that also fails, it checks for the presence of 'fzp' and 'surface_pressure' variables and attempts to reconstruct pressure levels from the hybrid coefficients. If all methods fail, it returns None to indicate that pressure levels could not be resolved, allowing the caller to fall back to returning model level indices instead.

        Parameters:
            var_name (str): The name of the variable for which to resolve pressure levels.
            vertical_dim (str): The name of the vertical dimension in the dataset.
            num_levels (int): The expected number of vertical levels based on the dataset.
            time_dim (str): The name of the time dimension in the dataset.
            time_idx (int): The index of the time step for which to resolve pressure levels.

        Returns:
            Optional[List[Union[int, float]]]: A list of pressure levels in Pa if resolved successfully, or None if pressure levels could not be resolved.
        """
        if 'pressure' in self.dataset:
            levels = self._pressure_levels_from_pressure_var(
                var_name, vertical_dim, num_levels, time_dim, time_idx
            )
            if levels is not None:
                return levels.tolist()

        if 'pressure_p' in self.dataset and 'pressure_base' in self.dataset:
            return self._pressure_levels_from_perturbation(
                var_name, vertical_dim, num_levels, time_dim, time_idx
            ).tolist()

        if 'fzp' in self.dataset and 'surface_pressure' in self.dataset:
            levels = self._pressure_levels_from_hybrid_coeffs(
                var_name, vertical_dim, num_levels, time_dim, time_idx
            )
            if levels is not None:
                return levels.tolist()

        return None

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

        vertical_dim, num_levels = self._get_vertical_dim(var_name)

        if return_pressure:
            time_dim, time_idx, _ = MPASDateTimeUtils.validate_time_parameters(
                self.dataset, time_index, self.verbose
            )
            result = self._resolve_pressure_levels(
                var_name, vertical_dim, num_levels, time_dim, time_idx
            )
            if result is not None:
                return result

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
    def _extract_xarray_by_index(data_3d: xr.DataArray,
                                 level_index: int,
                                 level_dim: str) -> np.ndarray:
        """
        This helper method extracts a 2D horizontal slice from a 3D xarray DataArray based on a specified level index along a given vertical dimension. It checks if the specified vertical dimension exists in the DataArray and uses it for indexing; if not, it falls back to using the second dimension of the DataArray for indexing. The method returns the extracted slice as a 2D numpy array, which can be used for further processing or visualization.
        
        Parameters:
            data_3d (xr.DataArray): The 3D xarray DataArray from which to extract the slice.
            level_index (int): The zero-based index of the vertical level to extract.
            level_dim (str): The name of the vertical dimension in the DataArray.

        Returns:
            np.ndarray: A 2D numpy array containing the extracted horizontal slice.
        """
        if level_dim in data_3d.sizes:
            return data_3d.isel({level_dim: level_index}).values

        fallback_dim = list(data_3d.sizes.keys())[1]
        return data_3d.isel({fallback_dim: level_index}).values

    @staticmethod
    def _extract_xarray_by_value(data_3d: xr.DataArray,
                                 level_value: Optional[float],
                                 level_dim: str,
                                 method: str) -> np.ndarray:
        """
        This helper method extracts a 2D horizontal slice from a 3D xarray DataArray based on a specified coordinate value along a given vertical dimension. It checks if the specified vertical dimension exists in the DataArray and uses it for interpolation; if not, it raises a ValueError. The method supports both nearest-neighbor and linear interpolation methods to find the appropriate level corresponding to the provided coordinate value. The resulting slice is returned as a 2D numpy array for further processing or visualization.

        Parameters:
            data_3d (xr.DataArray): The 3D xarray DataArray from which to extract the slice.
            level_value (Optional[float]): The coordinate value along the vertical dimension to extract.
            level_dim (str): The name of the vertical dimension in the DataArray.
            method (str): The interpolation method to use ('nearest' or 'linear').

        Returns:
            np.ndarray: A 2D numpy array containing the extracted horizontal slice.
        """
        if level_dim not in data_3d.coords:
            raise ValueError(f"Coordinate '{level_dim}' not found in data array")

        coord_values = data_3d.coords[level_dim].values

        if method == 'nearest':
            closest_idx = int(np.argmin(np.abs(coord_values - level_value)))
            return data_3d.isel({level_dim: closest_idx}).values

        return data_3d.interp({level_dim: level_value}, method='linear').values

    @staticmethod
    def _extract_numpy_by_index(data_3d: np.ndarray,
                                level_index: int) -> np.ndarray:
        """
        This helper method extracts a 2D horizontal slice from a 3D numpy array based on a specified level index. It checks the number of dimensions in the input array and uses the appropriate indexing to extract the slice. If the array has three dimensions, it assumes the vertical dimension is the second one and extracts accordingly; if it has more than three dimensions, it uses the second dimension for indexing without assuming a specific structure. The resulting slice is returned as a 2D numpy array for further processing or visualization.

        Parameters:
            data_3d (np.ndarray): The 3D numpy array from which to extract the slice.
            level_index (int): The zero-based index of the vertical level to extract.

        Returns:
            np.ndarray: A 2D numpy array containing the extracted horizontal slice.
        """
        if data_3d.ndim < 2:
            raise ValueError("Data must be at least 2D for level extraction")

        if data_3d.ndim == 3:
            return data_3d[:, level_index, -1]

        return data_3d[:, level_index]

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
                return MPAS3DProcessor._extract_xarray_by_index(data_3d, level_index, level_dim)
            return MPAS3DProcessor._extract_xarray_by_value(data_3d, level_value, level_dim, method)

        if level_index is None:
            raise ValueError("level_value extraction requires xarray.DataArray with coordinates")
        return MPAS3DProcessor._extract_numpy_by_index(data_3d, level_index)
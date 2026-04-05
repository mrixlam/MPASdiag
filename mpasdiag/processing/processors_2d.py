#!/usr/bin/env python3

"""
MPASdiag Core Processing Module: 2D Surface Data Processor

This module defines the MPAS2DProcessor class, which is responsible for handling 2D surface and diagnostic variables from MPAS model output. It provides methods for loading datasets, extracting spatial coordinates, and preparing data for visualization. The processor is designed to work with both cell-centered and vertex-based variables, ensuring that all necessary spatial information is included for geospatial analysis. By inheriting from MPASBaseProcessor, it utilizes common functionality while implementing specific methods tailored for 2D surface data processing workflows. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import glob
import os
import numpy as np
import pandas as pd
import xarray as xr
from typing import List, Tuple, Any, Optional, cast

from .base import MPASBaseProcessor
from .utils_datetime import MPASDateTimeUtils
from .constants import DIAG_GLOB, MPASOUT_GLOB


class MPAS2DProcessor(MPASBaseProcessor):
    """ Processor class for handling 2D surface and diagnostic variables from MPAS model output. """
    
    def __init__(self: 'MPAS2DProcessor', grid_file: str, 
                 verbose: bool = True):
        """
        This constructor initializes the MPAS2DProcessor by calling the base class constructor with the provided grid file and verbosity settings. It sets up the necessary attributes for processing 2D diagnostic data, ensuring that the processor is ready to load datasets, extract spatial coordinates, and prepare data for visualization. By inheriting from MPASBaseProcessor, it leverages common functionality while allowing for specific implementations tailored to 2D surface data processing workflows. 

        Parameters:
            grid_file (str): Path to the MPAS grid file containing spatial coordinate information.
            verbose (bool): If True, enables detailed logging of processing steps (default: True). 

        Returns:
            None 
        """
        super().__init__(grid_file, verbose)
    
    def add_spatial_coordinates(self: 'MPAS2DProcessor', 
                                combined_ds: xr.Dataset) -> xr.Dataset:
        """
        This method enriches the combined dataset of 2D diagnostic variables by adding coordinate variables for spatial dimensions (nCells, nVertices, nIsoLevelsT, nIsoLevelsZ) and their corresponding geographic coordinates (latitude and longitude). It identifies the relevant dimensions and spatial variables in the dataset, then adds the appropriate coordinate variables based on the grid information. This process ensures that the resulting dataset is fully equipped with the necessary spatial information for geospatial analysis and plotting. By centralizing this functionality in a dedicated method, it promotes code reuse and maintainability across different processing workflows that require spatial coordinate enrichment for 2D diagnostic data. 

        Parameters:
            combined_ds (xr.Dataset): The combined dataset of 2D diagnostic variables to enrich with spatial coordinates. 

        Returns:
            xr.Dataset: The enriched dataset with added spatial coordinate variables for nCells, nVertices, nIsoLevelsT, and nIsoLevelsZ, along with their corresponding latitude and longitude coordinates. 
        """
        dimensions_to_add = ['nCells', 'nVertices', 'nIsoLevelsT', 'nIsoLevelsZ']
        spatial_vars = ['latCell', 'lonCell', 'latVertex', 'lonVertex']
        
        return self._add_spatial_coords_helper(
            combined_ds, dimensions_to_add, spatial_vars, "2D"
        )

    def load_2d_data(self: 'MPAS2DProcessor', 
                     data_dir: str, 
                     use_pure_xarray: bool = False, 
                     reference_file: str = "",
                     variables: Optional[List[str]] = None) -> 'MPAS2DProcessor':
        """
        This method loads 2D diagnostic data from the specified directory, utilizing either the pure xarray backend or the UXarray backend based on the provided flag. It first identifies the relevant diagnostic files in the directory, then loads them into a combined dataset while adding necessary spatial coordinates. The method ensures that the loaded dataset is properly enriched with spatial information for subsequent analysis and visualization. By returning self, it allows for method chaining in processing workflows that require multiple operations on the loaded dataset. 

        Parameters:
            data_dir (str): Directory path to search for MPAS diagnostic files.
            use_pure_xarray (bool): If True, uses the pure xarray backend for loading data; otherwise, uses UXarray (default: False).
            reference_file (str): Optional path to a reference file for loading data, if required by the backend (default: "").
            variables (Optional[List[str]]): List of variable names to retain. If provided, only these variables are loaded from data files to reduce memory usage (default: None). 

        Returns:
            MPAS2DProcessor: The instance of the processor with the loaded dataset ready for analysis and visualization. 
        """
        self.data_dir = data_dir
        
        chunks_2d = {'Time': 1, 'nCells': 100000}
        
        dataset, data_type = self._load_data(
            data_dir, 
            use_pure_xarray, 
            reference_file,
            chunks=chunks_2d,
            data_type_label="2D",
            variables=variables
        )
        
        if hasattr(dataset, 'data_vars'):  
            dataset = self.add_spatial_coordinates(dataset)
            self.dataset = dataset
        elif hasattr(dataset, 'ds'):  
            dataset.ds = self.add_spatial_coordinates(dataset.ds)
            self.dataset = dataset
        
        return self

    def _find_diag_files_recursive(self: 'MPAS2DProcessor',
                                    data_dir: str) -> Optional[List[str]]:
        """
        This helper method performs a recursive search for diagnostic files in the specified directory using a predefined glob pattern. It looks for files matching the DIAG_GLOB pattern in all subdirectories of the provided data directory. If it finds at least two diagnostic files, it returns a sorted list of their paths; otherwise, it returns None. This method is used as a fallback when direct searches for diagnostic files in the main directory and a "diag" subdirectory do not yield results, ensuring that all possible locations are checked for relevant diagnostic data.

        Parameters:
            data_dir (str): Directory path to search for MPAS diagnostic files recursively.

        Returns:
            Optional[List[str]]: A sorted list of file paths to the found diagnostic files if at least two are found; otherwise, None.
        """
        files = sorted(glob.glob(os.path.join(data_dir, "**", DIAG_GLOB), recursive=True))

        if len(files) >= 2:
            if self.verbose:
                print(f"\nFound {len(files)} diagnostic files (recursive search):")
                for i, f in enumerate(files[:5]):
                    print(f"  {i+1}: {os.path.basename(f)}")
            return files

        return None

    def _find_mpasout_files_fallback(self: 'MPAS2DProcessor',
                                     data_dir: str) -> List[str]:
        """
        This helper method performs a fallback search for MPAS output files (mpasout*.nc) in the specified directory using a predefined glob pattern. It first attempts to find files directly in the provided directory, then looks in a "diag" subdirectory if no files are found. If still unsuccessful, it performs a recursive search for MPAS output files. If no MPAS output files are found after these attempts, it raises a FileNotFoundError with an informative message. The method ensures that at least two MPAS output files are found for temporal analysis and provides detailed logging of the search results when verbose mode is enabled.

        Parameters:
            data_dir (str): Directory path to search for MPAS output files.

        Returns:
            List[str]: A sorted list of file paths to the found MPAS output files suitable for 2D analysis.
        """
        try:
            return self._find_files_by_pattern(data_dir, MPASOUT_GLOB, "MPAS output files (mpasout)")
        except FileNotFoundError:
            pass

        files = sorted(glob.glob(os.path.join(data_dir, "**", MPASOUT_GLOB), recursive=True))

        if not files:
            raise FileNotFoundError(
                f"No diagnostic files (diag*.nc) or MPAS output files (mpasout*.nc) found under: {data_dir}\n"
                f"For precipitation analysis, ensure files contain rainc and rainnc variables."
            )

        if len(files) < 2:
            raise ValueError(
                f"Insufficient MPAS output files for temporal analysis. Found {len(files)}, need at least 2."
            )

        if self.verbose:
            print(f"\nFound {len(files)} MPAS output files (recursive search):")
            for i, f in enumerate(files[:5]):
                print(f"  {i+1}: {os.path.basename(f)}")

        return files

    def find_diagnostic_files(self: 'MPAS2DProcessor', 
                              data_dir: str) -> List[str]:
        """
        This method searches for diagnostic files in the specified directory using a predefined glob pattern. It first attempts to find files directly in the provided directory, then looks in a "diag" subdirectory if no files are found. If still unsuccessful, it performs a recursive search for diagnostic files. If no diagnostic files are found after these attempts, it then searches for MPAS output files (mpasout*.nc) using a similar approach. The method ensures that at least two files are found for temporal analysis and provides informative error messages if no suitable files are located. By returning a sorted list of file paths, it facilitates subsequent loading and processing of the diagnostic data. 

        Parameters:
            data_dir (str): Directory path to search for MPAS diagnostic files. 

        Returns:
            List[str]: A sorted list of file paths to the found diagnostic files or MPAS output files suitable for 2D analysis. 
        """
        try:
            return self._find_files_by_pattern(data_dir, DIAG_GLOB, "diagnostic files")
        except FileNotFoundError:
            pass

        diag_sub = os.path.join(data_dir, "diag")
        
        try:
            return self._find_files_by_pattern(diag_sub, DIAG_GLOB, "diagnostic files")
        except FileNotFoundError:
            pass

        files = self._find_diag_files_recursive(data_dir)

        if files:
            return files

        if self.verbose:
            print("\nNo diagnostic files found, searching for mpasout files...")

        return self._find_mpasout_files_fallback(data_dir)

    _COORD_NAMES_2D: dict = {
        'nVertices': (['lonVertex', 'lon_vertex', 'longitude_vertex'],
                      ['latVertex', 'lat_vertex', 'latitude_vertex']),
        'nCells':    (['lonCell',   'longitude',  'lon'],
                      ['latCell',   'latitude',   'lat']),
    }

    @staticmethod
    def _detect_2d_spatial_dim(sizes: Any) -> str:
        """
        This static method detects the spatial dimension for 2D variables based on the provided sizes. It checks for the presence of 'nVertices' and 'nCells' in the sizes and returns the appropriate spatial dimension name. If 'nVertices' is found, it returns 'nVertices'; otherwise, it defaults to 'nCells'. This method is essential for determining the correct coordinate variables to use when extracting spatial information for 2D diagnostic variables, ensuring that the processor can handle both vertex-based and cell-centered data appropriately.

        Parameters:
            sizes (Any): The sizes of the variable dimensions, typically obtained from an xarray DataArray or Dataset.

        Returns:
            str: The name of the spatial dimension ('nVertices' or 'nCells') to use for coordinate extraction.
        """
        return 'nVertices' if 'nVertices' in sizes else 'nCells'

    def _lookup_2d_coord(self: 'MPAS2DProcessor',
                         names: List[str]) -> Optional[np.ndarray]:
        """
        This helper method looks up coordinate variables in the dataset based on a list of possible names. It iterates through the provided list of names and checks if any of them exist as coordinates or data variables in the dataset. If a matching variable is found, it returns its values as a NumPy array. If none of the names are found, it returns None. This method is crucial for flexibly handling different naming conventions for spatial coordinates in MPAS datasets, allowing the processor to adapt to various dataset structures without requiring hardcoded variable names. 

        Parameters:
            names (List[str]): A list of possible coordinate variable names to search for.

        Returns:
            Optional[np.ndarray]: The values of the first found coordinate variable, or None if none are found.
        """
        for name in names:
            if name in self.dataset.coords or name in self.dataset.data_vars:
                return self.dataset[name].values
        return None

    def extract_2d_coordinates_for_variable(self: 'MPAS2DProcessor', 
                                            var_name: str, 
                                            data_array: Optional[xr.DataArray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method extracts the longitude and latitude coordinates for a specified 2D variable from the loaded dataset. It first determines whether the variable is associated with cell-centered (nCells) or vertex-based (nVertices) spatial dimensions by inspecting the variable's dimensions or the provided data array. Based on this determination, it identifies the appropriate coordinate variable names for longitude and latitude. The method then retrieves the coordinate values, ensuring they are in degrees if they were originally in radians, and flattens them into 1D arrays. Finally, it normalizes longitude values to the range [-180, 180] and returns the longitude and latitude arrays. This functionality is essential for preparing spatial coordinates for geospatial analysis and visualization of 2D diagnostic variables. 

        Parameters:
            var_name (str): Name of the 2D variable for which to extract coordinates.
            data_array (Optional[xr.DataArray]): Optional data array of the variable, used to determine spatial dimensions if provided. 

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the longitude and latitude coordinate arrays corresponding to the specified variable. 
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_2d_data() first.")

        sizes = data_array.sizes if data_array is not None else (
            self.dataset[var_name].sizes if var_name in self.dataset else {}
        )
        
        spatial_dim = self._detect_2d_spatial_dim(sizes)
        lon_names, lat_names = self._COORD_NAMES_2D[spatial_dim]

        lon_coords = self._lookup_2d_coord(lon_names)
        lat_coords = self._lookup_2d_coord(lat_names)

        if lon_coords is None or lat_coords is None:
            available_vars = list(self.dataset.coords.keys()) + list(self.dataset.data_vars.keys())
            raise ValueError(f"Could not find {spatial_dim} coordinates. Available variables: {available_vars}")

        if np.nanmax(np.abs(lat_coords)) <= np.pi:
            lat_coords = lat_coords * 180.0 / np.pi
            lon_coords = lon_coords * 180.0 / np.pi

        lon_coords = ((lon_coords.ravel() + 180) % 360) - 180
        lat_coords = lat_coords.ravel()

        if self.verbose:
            print(f"Extracted {spatial_dim} coordinates for {var_name}: {len(lon_coords):,} points")

        return lon_coords, lat_coords

    def _log_2d_variable_range(self: 'MPAS2DProcessor',
                               var_name: str,
                               var_data: xr.DataArray) -> None:
        """
        This helper method logs the range of finite values for a specified 2D variable. It checks if verbose mode is enabled and if the variable data has values. It then flattens the variable data, filters out non-finite values, and calculates the minimum and maximum of the finite values. If finite values are found, it prints the range along with the variable name and its units if available. If no finite values are found, it logs a warning message. This method provides valuable insights into the data being processed, helping users understand the range of values for diagnostic variables and identify potential issues with the data.

        Parameters:
            var_name (str): Name of the variable for which to log the range.
            var_data (xr.DataArray): The data array of the variable whose range is to be logged.

        Returns:
            None

        """
        if not self.verbose or not hasattr(var_data, 'values'):
            return
        
        finite_values = var_data.values.flatten()
        finite_values = finite_values[np.isfinite(finite_values)]

        if len(finite_values) > 0:
            print(f"Variable {var_name} range: {finite_values.min():.3f} to {finite_values.max():.3f}")
            if hasattr(var_data, 'attrs') and 'units' in var_data.attrs:
                print(f"Units: {var_data.attrs['units']}")
        else:
            print(f"Warning: No finite values found for {var_name}")

    def get_2d_variable_data(self: 'MPAS2DProcessor', 
                             var_name: str, 
                             time_index: int = 0) -> xr.DataArray:
        """
        This method retrieves the data array for a specified 2D variable at a given time index from the loaded dataset. It first checks if the dataset is loaded and if the variable exists in the dataset. It then validates the time index against the dataset's time dimension, ensuring it is within bounds. Depending on the data type (pure xarray or UXarray), it extracts the variable data at the specified time index. If the extracted data is a lazy array, it computes it to get the actual values. If verbose mode is enabled, it also prints out the range of finite values for the variable along with its units if available. Finally, it returns the data array for further analysis or visualization. 

        Parameters:
            var_name (str): Name of the 2D variable to retrieve.
            time_index (int): Time index to extract data from (default: 0). 

        Returns:
            xr.DataArray: The data array for the specified variable at the given time index, ready for analysis or visualization. 
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_2d_data() first.")

        if var_name not in self.dataset.data_vars:
            available_vars = list(self.dataset.data_vars.keys())
            raise ValueError(f"Variable '{var_name}' not found. Available variables: {available_vars}")

        time_dim, validated_time_index, _ = MPASDateTimeUtils.validate_time_parameters(
            self.dataset, time_index, self.verbose
        )

        if self.verbose:
            print(f"Extracting {var_name} data at time index {validated_time_index}")

        if self.data_type == 'uxarray':
            var_data = self.dataset[var_name][validated_time_index]
        else:
            var_data = self.dataset[var_name].isel({time_dim: validated_time_index})

        if hasattr(var_data, 'compute'):
            var_data = cast(Any, var_data).compute()

        self._log_2d_variable_range(var_name, var_data)

        return var_data

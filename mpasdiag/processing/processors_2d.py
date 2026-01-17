#!/usr/bin/env python3

"""
MPAS 2D Surface Data Processor

This module provides specialized functionality for processing MPAS 2D surface and diagnostic variables including precipitation, temperature, pressure, humidity, and wind fields from atmospheric model output. It implements the MPAS2DProcessor class that extends MPASBaseProcessor to handle time series extraction of surface variables, geographic coordinate retrieval for unstructured mesh cells, precipitation accumulation calculations with temporal differencing, and statistical diagnostics for surface meteorology. The processor manages both diagnostic stream files (with accumulated precipitation variables like rainnc, rainc) and history stream files (with instantaneous surface fields like t2m, mslp, q2), provides methods for extracting cell-centered data with proper time indexing, handles missing data and NaN values with robust filtering, and supports batch processing for creating time series of surface analyses. Core capabilities include automatic file pattern matching for multi-file datasets, xarray-based data extraction with chunk optimization, integration with precipitation diagnostics utilities, and seamless data preparation for 2D surface visualization workflows suitable for operational weather analysis and model evaluation.

Classes:
    MPAS2DProcessor: Specialized processor class for extracting and analyzing 2D surface variables from MPAS atmospheric model output.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from typing import List, Tuple, Any, Optional, cast

from .base import MPASBaseProcessor
from .utils_datetime import MPASDateTimeUtils
from .constants import DIAG_GLOB, MPASOUT_GLOB


class MPAS2DProcessor(MPASBaseProcessor):
    """
    Specialized processor for 2D MPAS diagnostic data.
    
    This class handles 2D surface and diagnostic variables from MPAS models,
    providing methods for precipitation analysis, surface meteorology, and 
    diagnostic file processing.
    """
    
    def __init__(self, grid_file: str, verbose: bool = True):
        """
        Initialize the 2D MPAS processor for surface and diagnostic variable analysis. This constructor sets up the processor by loading the MPAS grid file to extract mesh coordinates and cell connectivity information needed for 2D field processing. It inherits base functionality from MPASBaseProcessor and configures verbose logging for operational transparency. The processor handles precipitation, surface meteorology, and other 2D diagnostic variables from MPAS model output. Coordinates are validated and prepared for subsequent data loading and spatial operations.

        Parameters:
            grid_file (str): Path to the MPAS grid file containing mesh coordinates and connectivity.
            verbose (bool): If True, print informational messages during processing operations (default: True).

        Returns:
            None
        """
        super().__init__(grid_file, verbose)
    
    def add_spatial_coordinates(self, combined_ds: xr.Dataset) -> xr.Dataset:
        """
        Enrich 2D diagnostic dataset with spatial coordinates and mesh connectivity from MPAS grid file. This method loads the grid file and extracts geographic coordinates for both cell centers and vertices along with dimensional indices needed for proper dataset structure. It handles multiple spatial dimensions including nCells for cell-centered variables, nVertices for vertex-based fields, and nIsoLevelsT/nIsoLevelsZ for isothermal and isoheight level diagnostics. The method adds coordinate variables as metadata while preserving all existing data variables. Missing or incompatible coordinates are handled gracefully with warning messages when verbose mode is enabled. This coordinate enrichment is essential for subsequent spatial plotting and geographic analysis operations.

        Parameters:
            combined_ds (xr.Dataset): Combined xarray dataset containing 2D diagnostic data from multiple time steps.

        Returns:
            xr.Dataset: Enriched dataset with added coordinate variables for spatial dimensions and geographic coordinates for plotting.
        """
        dimensions_to_add = ['nCells', 'nVertices', 'nIsoLevelsT', 'nIsoLevelsZ']
        spatial_vars = ['latCell', 'lonCell', 'latVertex', 'lonVertex']
        
        return self._add_spatial_coords_helper(
            combined_ds, dimensions_to_add, spatial_vars, "2D"
        )

    def load_2d_data(self, data_dir: str, use_pure_xarray: bool = False, 
                     reference_file: str = "") -> 'MPAS2DProcessor':
        """
        Load and configure 2D diagnostic data from MPAS output files with optimized chunking. This method reads diagnostic files from the specified directory using either UXarray or pure xarray backends, applies memory-efficient chunking strategies optimized for 2D fields, and adds spatial coordinates from the grid file. The method supports both regular datasets and UXarray grid objects, automatically detecting and handling the appropriate structure. After loading, it enriches the dataset with geographic coordinates and mesh connectivity needed for visualization and analysis. Returns self to enable method chaining in processing workflows.

        Parameters:
            data_dir (str): Directory path containing MPAS diagnostic files to load.
            use_pure_xarray (bool): If True, use pure xarray backend instead of UXarray (default: False).
            reference_file (str): Optional specific file to use as reference instead of scanning directory (default: "").

        Returns:
            MPAS2DProcessor: Self reference for method chaining operations.
        """
        self.data_dir = data_dir
        
        chunks_2d = {'Time': 1, 'nCells': 100000}
        
        dataset, data_type = self._load_data(
            data_dir, 
            use_pure_xarray, 
            reference_file,
            chunks=chunks_2d,
            data_type_label="2D"
        )
        
        if hasattr(dataset, 'data_vars'):  
            dataset = self.add_spatial_coordinates(dataset)
            self.dataset = dataset
        elif hasattr(dataset, 'ds'):  
            dataset.ds = self.add_spatial_coordinates(dataset.ds)
            self.dataset = dataset
        
        return self

    def find_diagnostic_files(self, data_dir: str) -> List[str]:
        """
        Locate and validate MPAS diagnostic or mpasout files in the specified directory by pattern matching with fallback. This method searches for files matching the diagnostic naming convention (diag*.nc) first, then falls back to MPAS output files (mpasout*.nc) if diag files are not available. This enables precipitation analysis from either diagnostic streams or standard model output files that contain rainc and rainnc variables. It performs validation to ensure sufficient files exist for temporal analysis and raises appropriate exceptions if neither file type is found. The method delegates to the base class pattern-matching utility with diagnostic-specific parameters. Sorted output ensures consistent temporal ordering for time series operations.

        Parameters:
            data_dir (str): Directory path to search for MPAS diagnostic or output files.

        Returns:
            list of str: Sorted list of diagnostic or output file paths found in the directory.

        Raises:
            FileNotFoundError: If no diagnostic or output files matching the patterns are found in the directory.
            ValueError: If insufficient files are present for meaningful temporal analysis.
        """
        try:
            return self._find_files_by_pattern(data_dir, DIAG_GLOB, "diagnostic files")
        except FileNotFoundError:
            diag_sub = os.path.join(data_dir, "diag")
            try:
                return self._find_files_by_pattern(diag_sub, DIAG_GLOB, "diagnostic files")
            except FileNotFoundError:
                files = [f for f in sorted(__import__('glob').glob(os.path.join(data_dir, "**", DIAG_GLOB), recursive=True))]
                if files and len(files) >= 2:
                    if self.verbose:
                        print(f"\nFound {len(files)} diagnostic files (recursive search):")
                        for i, f in enumerate(files[:5]):
                            print(f"  {i+1}: {os.path.basename(f)}")
                    return files
                
                if self.verbose:
                    print(f"\nNo diagnostic files found, searching for mpasout files...")
                try:
                    return self._find_files_by_pattern(data_dir, MPASOUT_GLOB, "MPAS output files (mpasout)")
                except FileNotFoundError:
                    mpasout_files = [f for f in sorted(__import__('glob').glob(os.path.join(data_dir, "**", MPASOUT_GLOB), recursive=True))]
                    if not mpasout_files:
                        raise FileNotFoundError(
                            f"No diagnostic files (diag*.nc) or MPAS output files (mpasout*.nc) found under: {data_dir}\n"
                            f"For precipitation analysis, ensure files contain rainc and rainnc variables."
                        )
                    if len(mpasout_files) < 2:
                        raise ValueError(f"Insufficient MPAS output files for temporal analysis. Found {len(mpasout_files)}, need at least 2.")
                    if self.verbose:
                        print(f"\nFound {len(mpasout_files)} MPAS output files (recursive search):")
                        for i, f in enumerate(mpasout_files[:5]):
                            print(f"  {i+1}: {os.path.basename(f)}")
                    return mpasout_files

    def extract_2d_coordinates_for_variable(self, var_name: str, data_array: Optional[xr.DataArray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract appropriate horizontal coordinates for a 2D variable based on its spatial dimension. This method determines whether the variable is defined on cell centers (nCells) or vertices (nVertices) and returns the corresponding longitude and latitude arrays for plotting. It inspects the data array dimensions or queries the dataset to identify the spatial dimension, then retrieves latCell/lonCell for cell-centered variables or latVertex/lonVertex for vertex-based variables. The method handles MPAS unstructured mesh topology automatically. Note that this method only handles 2D surface variables; 3D variables with vertical dimensions require separate processing with vertical coordinate extraction.

        Parameters:
            var_name (str): Name of the 2D surface variable to extract coordinates for.
            data_array (Optional[xr.DataArray]): Optional data array to inspect for dimension information, defaults to querying dataset (default: None).

        Returns:
            tuple of (np.ndarray, np.ndarray): Longitude and latitude arrays corresponding to the variable's spatial dimension.

        Raises:
            ValueError: If dataset is not loaded or spatial dimension cannot be determined.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_2d_data() first.")
            
        spatial_dim = 'nCells' 
        
        if data_array is not None:
            if 'nVertices' in data_array.sizes:
                spatial_dim = 'nVertices'
            elif 'nCells' in data_array.sizes:
                spatial_dim = 'nCells'
        elif var_name in self.dataset:
            if 'nVertices' in self.dataset[var_name].sizes:
                spatial_dim = 'nVertices'
            elif 'nCells' in self.dataset[var_name].sizes:
                spatial_dim = 'nCells'
        
        if spatial_dim == 'nVertices':
            lon_names = ['lonVertex', 'lon_vertex', 'longitude_vertex']
            lat_names = ['latVertex', 'lat_vertex', 'latitude_vertex']
        else:  
            lon_names = ['lonCell', 'longitude', 'lon']
            lat_names = ['latCell', 'latitude', 'lat']
        
        lon_coords = lat_coords = None
        
        for name in lon_names:
            if name in self.dataset.coords or name in self.dataset.data_vars:
                lon_coords = self.dataset[name].values
                break
                
        for name in lat_names:
            if name in self.dataset.coords or name in self.dataset.data_vars:
                lat_coords = self.dataset[name].values
                break
                
        if lon_coords is None or lat_coords is None:
            available_vars = list(self.dataset.coords.keys()) + list(self.dataset.data_vars.keys())
            raise ValueError(f"Could not find {spatial_dim} coordinates. Available variables: {available_vars}")
        
        if np.nanmax(np.abs(lat_coords)) <= np.pi:
            lat_coords = lat_coords * 180.0 / np.pi
            lon_coords = lon_coords * 180.0 / np.pi
        
        lon_coords = lon_coords.ravel()
        lat_coords = lat_coords.ravel()
        lon_coords = ((lon_coords + 180) % 360) - 180
        
        if self.verbose:
            print(f"Extracted {spatial_dim} coordinates for {var_name}: {len(lon_coords):,} points")
        
        return lon_coords, lat_coords

    def get_2d_variable_data(self, var_name: str, time_index: int = 0) -> xr.DataArray:
        """
        Retrieve 2D variable data at a specific time index with validation and optional statistics. This method extracts a single timestep of 2D field data from the loaded dataset after validating that the variable exists and the time index is within bounds. It handles both UXarray and standard xarray data structures transparently, applying the appropriate indexing method for each type. The method computes lazy-loaded data into memory for performance and optionally prints variable statistics when verbose mode is enabled. Returns a fully-realized xarray DataArray ready for analysis or visualization.

        Parameters:
            var_name (str): Name of the 2D variable to extract from the dataset.
            time_index (int): Zero-based time index to extract data from (default: 0).

        Returns:
            xr.DataArray: Variable data array at the specified time index with computed values.

        Raises:
            ValueError: If dataset is not loaded, variable name is not found, or time index is out of bounds.
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_2d_data() first.")
        
        if var_name not in self.dataset.data_vars:
            available_vars = list(self.dataset.data_vars.keys())
            raise ValueError(f"Variable '{var_name}' not found. Available variables: {available_vars}")
        
        time_dim, validated_time_index, time_size = MPASDateTimeUtils.validate_time_parameters(self.dataset, time_index, self.verbose)
        
        if self.verbose:
            print(f"Extracting {var_name} data at time index {validated_time_index}")
        
        if self.data_type == 'uxarray':
            var_data = self.dataset[var_name][validated_time_index]
        else:
            var_data = self.dataset[var_name].isel({time_dim: validated_time_index})
        
        if hasattr(var_data, 'compute'):
            var_data = cast(Any, var_data).compute()
        
        if self.verbose:
            if hasattr(var_data, 'values'):
                data_values = var_data.values.flatten()
                finite_values = data_values[np.isfinite(data_values)]

                if len(finite_values) > 0:
                    print(f"Variable {var_name} range: {finite_values.min():.3f} to {finite_values.max():.3f}")
                    
                    if hasattr(var_data, 'attrs') and 'units' in var_data.attrs:
                        print(f"Units: {var_data.attrs['units']}")
                else:
                    print(f"Warning: No finite values found for {var_name}")
        
        return var_data
    
    def get_accumulation_hours(self, accum_period: str) -> int:
        """
        Convert accumulation period string identifier to integer hours for precipitation analysis. This method maps standard MPAS accumulation period codes to their corresponding hour values using a predefined lookup table. It supports common accumulation windows including hourly (a01h), 3-hourly (a03h), 6-hourly (a06h), 12-hourly (a12h), and daily (a24h) periods. If the provided period string is not recognized, the method defaults to 24 hours as a conservative fallback. This conversion is essential for precipitation diagnostics and temporal differencing operations.

        Parameters:
            accum_period (str): Accumulation period identifier string in format 'aXXh' (e.g., 'a01h', 'a24h').

        Returns:
            int: Number of hours corresponding to the accumulation period, defaults to 24 if not found.
        """
        accum_hours_map = {'a01h': 1, 'a03h': 3, 'a06h': 6, 'a12h': 12, 'a24h': 24}
        return accum_hours_map.get(accum_period, 24)
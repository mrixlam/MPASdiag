#!/usr/bin/env python3

"""
MPAS Base Data Processor

This module provides the foundational data processing infrastructure for reading, loading, and managing MPAS atmospheric model output files with support for both diagnostic and history streams. It implements the MPASBaseDataProcessor class that serves as the parent for specialized processors (2D surface, 3D atmospheric), providing common functionality including file discovery with glob pattern matching, xarray-based dataset loading with chunk optimization for memory efficiency, time coordinate extraction and standardization, variable validation and listing, and geographic coordinate handling for MPAS unstructured mesh data. The base processor establishes consistent data access patterns across all MPASdiag processing modules, handles netCDF file operations with proper error handling, manages dataset caching for performance, and provides utility methods for time series processing and spatial subsetting. This foundational class enables rapid development of specialized diagnostic processors by inheriting common file I/O, coordinate handling, and data validation functionality while allowing customization for variable-specific extraction and computation requirements.

Classes:
    MPASBaseDataProcessor: Abstract base class providing common data processing infrastructure for all MPAS diagnostic processors.
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import os
import re
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import uxarray as ux
from datetime import datetime
from typing import List, Tuple, Any, Optional, Dict, Union, cast

from .utils_datetime import MPASDateTimeUtils
from .constants import DATASET_NOT_LOADED_MSG, DIAG_GLOB
warnings.filterwarnings('ignore', message='The specified chunks separate the stored chunks.*')
warnings.filterwarnings('ignore', message='invalid value encountered in create_collection')
warnings.filterwarnings('ignore', message='.*Shapely.*')
warnings.filterwarnings('ignore', category=UserWarning, module='cartopy')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='shapely')
warnings.filterwarnings('ignore', category=UserWarning, message='.*chunks.*degrade performance.*')


class MPASBaseProcessor:
    """
    Base class for processing MPAS model output data.
    
    This class provides common functionality for loading, processing, and analyzing MPAS unstructured
    mesh data with support for lazy loading and temporal analysis.
    """
    
    def __init__(self, grid_file: str, verbose: bool = True) -> None:
        """
        Initialize the MPAS data processor for handling unstructured mesh data with lazy loading capabilities. This constructor establishes the foundation for MPAS data processing by validating and storing the grid file path which contains essential mesh topology information. The processor supports multiple data loading strategies including pure xarray and UXarray for efficient handling of MPAS unstructured grid data. Verbose mode enables detailed diagnostic output throughout the data loading and processing pipeline. The initialization validates grid file existence before proceeding to ensure robust error handling. This base class provides common functionality shared across specialized 2D and 3D MPAS data processors.

        Parameters:
            grid_file (str): Absolute path to MPAS static grid file containing mesh topology and coordinate information.
            verbose (bool): Enable verbose output messages for debugging and diagnostic information during data loading and processing (default: True).

        Returns:
            None
        """
        self.grid_file = grid_file
        self.verbose = verbose
        self.dataset = None
        self.data_type = None
        
        if not os.path.exists(grid_file):
            raise FileNotFoundError(f"Grid file not found: {grid_file}")
    
    def _find_files_by_pattern(self, data_dir: str, pattern: str, file_type: str) -> List[str]:
        """
        Search for and validate files matching specified glob pattern in target directory. This method performs comprehensive file discovery using glob pattern matching with automatic sorting by filename for consistent temporal ordering. The search validates that sufficient files exist for temporal analysis operations requiring multiple time steps. If verbose mode is enabled, the method displays a summary of discovered files with truncated listing for large file sets. Error handling ensures clear diagnostic messages for missing files or insufficient temporal coverage. This utility method is used internally by specialized data loading methods across different MPAS output types.

        Parameters:
            data_dir (str): Absolute path to directory containing MPAS output files to search.
            pattern (str): Glob pattern for file matching (e.g., 'diag*.nc', 'mpasout*.nc').
            file_type (str): Human-readable description of file type for informative error messages.

        Returns:
            List[str]: Sorted list of absolute file paths matching the specified pattern with at least 2 files for temporal analysis.

        Raises:
            FileNotFoundError: If no files matching the pattern are found in the specified directory.
            ValueError: If fewer than 2 files are found, which is insufficient for temporal analysis operations.
        """
        file_pattern = os.path.join(data_dir, pattern)
        files = sorted(glob.glob(file_pattern))
        
        if not files:
            raise FileNotFoundError(f"No {file_type} found matching pattern: {file_pattern}")
        
        if len(files) < 2:
            raise ValueError(f"Insufficient files for temporal analysis. Found {len(files)}, need at least 2.")
        
        if self.verbose:
            print(f"\nFound {len(files)} {file_type}:")

            for i, f in enumerate(files[:5]):  
                print(f"  {i+1}: {os.path.basename(f)}")

            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more files")
        
        return files

    def validate_files(self, files: List[str]) -> List[str]:
        """
        Validate file existence and accessibility for a list of file paths with comprehensive error checking. This method performs filesystem-level validation to ensure all provided files exist and have read permissions before attempting data loading operations. The validation process checks both file existence and read accessibility to prevent downstream errors during data loading. Each file is tested individually with informative error messages indicating the specific file causing validation failure. This validation step is essential for robust error handling in data processing pipelines where missing or inaccessible files could cause silent failures or cryptic errors. The method returns only successfully validated files for subsequent processing operations.

        Parameters:
            files (List[str]): List of absolute file paths to validate for existence and read accessibility.

        Returns:
            List[str]: List of validated file paths that exist and are readable, preserving original order.

        Raises:
            FileNotFoundError: If any file does not exist or is not readable with specific file path in error message.
        """
        valid_files = []
        for file_path in files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            if not os.access(file_path, os.R_OK):
                raise FileNotFoundError(f"File not readable: {file_path}")
            valid_files.append(file_path)
        return valid_files
    
    def _load_data(self, data_dir: str, use_pure_xarray: bool = False,
                   reference_file: str = "", chunks: Optional[dict] = None, 
                   data_type_label: str = "data") -> Tuple[Any, str]:
        """
        Load MPAS data from multiple netCDF files with lazy loading and flexible backend support for efficient memory management. This protected method provides the core data loading functionality used by specialized load methods for different MPAS output types including 2D diagnostic files and 3D model output files. The method attempts multiple loading strategies starting with multi-file xarray concatenation, falling back to UXarray with unstructured grid support, and finally attempting single-file loading if concatenation fails. Chunked lazy loading is employed to handle large datasets that exceed available memory by loading only required data portions on demand. The method automatically detects and uses appropriate time dimensions, sorts data temporally, and assigns proper datetime coordinates from filename parsing. Comprehensive error handling with fallback strategies ensures robust data loading across different file formats and system configurations.

        Parameters:
            data_dir (str): Absolute path to directory containing MPAS output files for loading.
            use_pure_xarray (bool): Force pure xarray backend instead of UXarray for simplified processing without unstructured grid support (default: False).
            reference_file (str): Optional specific reference file path for time ordering and single-file fallback scenarios (default: "").
            chunks (Optional[dict]): Custom chunking strategy dictionary with dimension names as keys (e.g., {'Time': 1, 'nCells': 100000}), or None for default chunking (default: None).
            data_type_label (str): Human-readable label for data type used in verbose output messages (e.g., "2D", "3D", "diagnostic") (default: "data").

        Returns:
            Tuple[Any, str]: Two-element tuple containing (dataset_object, data_type_identifier) where dataset_object is xarray.Dataset or ux.UxDataset and data_type_identifier is either 'xarray' or 'uxarray'.
        """
        data_files: List[str]
        finder = getattr(self, 'find_diagnostic_files', None)
        if callable(finder):
            data_files = cast(List[str], finder(data_dir))
        else:
            finder2 = getattr(self, 'find_mpasout_files', None)
            if callable(finder2):
                data_files = cast(List[str], finder2(data_dir))
            else:
                data_files = self._find_files_by_pattern(data_dir, DIAG_GLOB, "diagnostic files")
        
        file_datetimes = MPASDateTimeUtils.parse_file_datetimes(data_files, self.verbose)
        
        if chunks is None:
            chunks = {'Time': 1, 'nCells': 100000}
        
        try:
            open_chunks = {} if chunks is None else {k: v for k, v in chunks.items() if k == 'Time'}

            combined_ds = xr.open_mfdataset(
                data_files,
                combine='nested',
                concat_dim='Time',
                chunks=open_chunks,
                parallel=False
            )
            
            combined_ds = combined_ds.assign_coords(Time=pd.to_datetime(file_datetimes))
            combined_ds = combined_ds.sortby('Time')

            try:
                if chunks is not None:
                    combined_ds = combined_ds.chunk(chunks)
            except Exception:
                pass
            
            if self.verbose:
                print(f"\n{data_type_label} Dataset structure:")
                print(combined_ds)
            
            if use_pure_xarray:
                if self.verbose:
                    self._print_loading_success(len(data_files), combined_ds, "pure xarray", data_type_label)
                self.dataset = combined_ds
                self.data_type = 'xarray'
                return combined_ds, 'xarray'
            else:
                grid_ds = ux.open_dataset(self.grid_file, data_files[0])
                grid_info = grid_ds.uxgrid

                if grid_info is None:
                    raise ValueError(f"Could not extract uxgrid from {data_type_label.lower()} dataset")

                final_ds = ux.UxDataset(combined_ds, uxgrid=grid_info)

                if self.verbose:
                    self._print_loading_success(len(data_files), final_ds, "UXarray", data_type_label)

                self.dataset = final_ds
                self.data_type = 'uxarray'
                return final_ds, 'uxarray'
                
        except Exception as e:
            if self.verbose:
                print(f"Primary {data_type_label.lower()} loading failed: {e}")
                print(f"Trying xarray fallback for {data_type_label.lower()} data...")
            
            try:
                fallback_open_chunks = {'Time': 1}

                combined_ds = xr.open_mfdataset(
                    data_files,
                    combine='nested',
                    concat_dim='Time',
                    chunks=fallback_open_chunks,
                    parallel=False
                )
                
                combined_ds = combined_ds.assign_coords(Time=pd.to_datetime(file_datetimes))
                combined_ds = combined_ds.sortby('Time')

                try:
                    if chunks is not None:
                        combined_ds = combined_ds.chunk(chunks)
                except Exception:
                    pass
                
                if self.verbose:
                    self._print_loading_success(len(data_files), combined_ds, "xarray (fallback)", data_type_label)
                
                self.dataset = combined_ds
                self.data_type = 'xarray'
                return combined_ds, 'xarray'
                
            except Exception as e2:
                if self.verbose:
                    print(f"Xarray fallback also failed: {e2}")
                try:
                    return self._load_single_file_fallback(reference_file, data_files)
                except Exception as e3:
                    print(f"All loading strategies failed: {e3}")
                    sys.exit(1)

    def _print_loading_success(self, num_files: int, dataset: Any, loader_type: str, data_type_label: str) -> None:
        """
        Print concise summary information after successfully loading and combining multiple dataset files. This method provides diagnostic feedback about the data loading process including file count, loader type, time dimension structure, and temporal coverage. The summary includes vertical level information when available for 3D datasets to help users verify correct data loading. Memory usage information indicates that lazy loading is active with chunked arrays to manage large datasets efficiently. This diagnostic output is controlled by the verbose flag and helps users confirm successful data loading before proceeding with analysis. The method formats output with clear section headers and consistent indentation for readability.

        Parameters:
            num_files (int): Number of individual files that were successfully loaded and combined into the dataset.
            dataset (Any): Combined xarray.Dataset or ux.UxDataset object containing loaded data.
            loader_type (str): Human-readable loader description indicating backend used (e.g., 'pure xarray', 'UXarray', 'xarray (fallback)').
            data_type_label (str): Descriptive label for data type being loaded (e.g., '2D', '3D', 'diagnostic').

        Returns:
            None
        """
        print(f"\nSuccessfully loaded {num_files} {data_type_label.lower()} files with {loader_type} (lazy)")
        print(f"Combined dataset time dimension: {dataset.Time.shape}")
        
        if hasattr(dataset, 'sizes') and 'nVertLevels' in dataset.sizes:
            print(f"Vertical levels: {dataset.sizes['nVertLevels']}")
        
        print(f"\nTime range: {dataset.Time.values[0]} to {dataset.Time.values[-1]}")
        print("Memory usage: Dataset uses chunked/lazy arrays")
        print(f"Data loaded as: {loader_type} ({data_type_label.lower()})")
    
    def _load_single_file_fallback(self, reference_file: str, data_files: List[str]) -> Tuple[Any, str]:
        """
        Implement fallback loading strategy using single file when multi-file concatenation fails. This method provides a robust last-resort data loading approach when primary multi-file loading strategies fail due to file format issues, memory constraints, or incompatible file structures. The method first attempts to load a user-specified reference file if provided, otherwise defaults to the first file in the data file list. Loading is attempted with both UXarray and xarray backends in sequence, automatically falling back to xarray if UXarray loading fails. This fallback approach trades temporal completeness for robustness, allowing analysis to proceed with limited time coverage when full multi-file loading is not possible. Verbose output informs users about the fallback mode and loaded file to manage expectations about temporal coverage limitations.

        Parameters:
            reference_file (str): Optional absolute path to specific reference file for single-file loading, or empty string to use first file from data_files list.
            data_files (List[str]): List of absolute paths to data files with first file used as fallback if reference_file is not provided.

        Returns:
            Tuple[Any, str]: Two-element tuple containing (dataset_object, data_type_identifier) where dataset_object is xarray.Dataset or ux.UxDataset loaded from single file and data_type_identifier is either 'xarray' or 'uxarray'.
        """
        if self.verbose:
            print("Falling back to single-file loading (limited functionality)...")
        
        if reference_file and os.path.exists(reference_file):
            try:
                ds = ux.open_dataset(self.grid_file, reference_file)
                if self.verbose:
                    print(f"Loaded single file: {reference_file}")
                self.dataset = ds
                self.data_type = 'uxarray'
                return ds, 'uxarray'
            except:
                ds = xr.open_dataset(reference_file)
                if self.verbose:
                    print(f"Loaded single file with xarray: {reference_file}")
                self.dataset = ds
                self.data_type = 'xarray'
                return ds, 'xarray'
        else:
            try:
                ds = ux.open_dataset(self.grid_file, data_files[0])
                if self.verbose:
                    print(f"Loaded first file: {data_files[0]}")
                self.dataset = ds
                self.data_type = 'uxarray'
                return ds, 'uxarray'
            except:
                ds = xr.open_dataset(data_files[0])
                if self.verbose:
                    print(f"Loaded first file with xarray: {data_files[0]}")
                self.dataset = ds
                self.data_type = 'xarray'
                return ds, 'xarray'
    
    def get_available_variables(self) -> List[str]:
        """
        Retrieve list of all available data variables in the currently loaded MPAS dataset. This method provides a simple interface to inspect dataset contents and determine which physical variables are available for analysis and visualization. The method requires that a dataset has been loaded through one of the load methods before it can be called. The returned variable list includes all data variables but excludes coordinate variables and dimensional metadata. This functionality is essential for dynamic variable selection in plotting routines and for validating user-requested variables against available dataset contents. The variable list reflects the actual dataset structure after any preprocessing or filtering operations.

        Parameters:
            None

        Returns:
            List[str]: List of data variable names available in the loaded dataset excluding coordinate and dimension variables.

        Raises:
            ValueError: If no dataset has been loaded prior to calling this method.
        """
        if self.dataset is None:
            raise ValueError(DATASET_NOT_LOADED_MSG)
        
        return list(self.dataset.data_vars.keys())
    
    def normalize_longitude(self, lon: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Normalize longitude values to standard [-180, 180] degree range for consistent geographic representation. This method converts longitude values from any arbitrary range to the standardized range where negative values represent western hemisphere and positive values represent eastern hemisphere. The normalization uses modulo arithmetic to handle longitude values that extend beyond a single 360-degree cycle. The method handles both scalar float values and numpy arrays with automatic type preservation and dimension handling. This normalization is essential for geographic plotting and spatial analysis where consistent longitude representation prevents discontinuities at dateline crossings. The implementation uses efficient numpy operations for array inputs while maintaining float type for scalar inputs.

        Parameters:
            lon (Union[float, np.ndarray]): Longitude value or array in degrees, potentially in [0, 360] or other ranges.

        Returns:
            Union[float, np.ndarray]: Normalized longitude value(s) in [-180, 180] degree range, preserving input type (float or array) and array dimensions.
        """
        lon = np.asarray(lon)
        result = ((lon + 180) % 360) - 180
        
        if result.ndim == 0:
            return float(result)
        return result
    
    def validate_geographic_extent(self, extent: Tuple[float, float, float, float]) -> bool:
        """
        Validate geographic extent tuple contains physically realistic coordinate bounds for spatial analysis operations. This method checks that longitude values fall within the valid [-180, 180] degree range and latitude values fall within the valid [-90, 90] degree range. The validation ensures that geographic extent specifications are physically meaningful before they are used in spatial filtering or plotting operations. Invalid extents could cause unexpected behavior in downstream spatial analysis routines or produce empty results. The method performs bounds checking on all four extent components without validating logical relationships between min/max pairs. This validation is typically called before spatial filtering operations to ensure robust error handling and clear diagnostic messages for invalid user input.

        Parameters:
            extent (Tuple[float, float, float, float]): Geographic extent as (lon_min, lon_max, lat_min, lat_max) in degrees.

        Returns:
            bool: True if all extent values are within physically valid ranges (lon: [-180, 180], lat: [-90, 90]), False otherwise.
        """
        lon_min, lon_max, lat_min, lat_max = extent
        
        if not (-180 <= lon_min <= 180) or not (-180 <= lon_max <= 180):
            return False
            
        if not (-90 <= lat_min <= 90) or not (-90 <= lat_max <= 90):
            return False
            
        if lon_min >= lon_max or lat_min >= lat_max:
            return False
            
        return True
    
    def extract_spatial_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract and normalize spatial coordinates from loaded MPAS dataset with automatic unit conversion and flattening. This method searches for longitude and latitude coordinates using multiple possible variable names to handle different MPAS output conventions and grid file formats. The method automatically detects whether coordinates are in radians or degrees by checking maximum absolute latitude values against π threshold. Coordinates in radians are converted to degrees for standardized geographic representation. Longitude values are normalized to the standard [-180, 180] degree range to ensure consistent handling of dateline crossings. Multi-dimensional coordinate arrays are flattened to 1D for compatibility with plotting and analysis routines that expect vectorized coordinate inputs. This extraction method is essential for geographic plotting and spatial analysis operations across different MPAS output types.

        Parameters:
            None

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two-element tuple containing (longitude_array, latitude_array) as flattened 1D numpy arrays in degrees with longitude normalized to [-180, 180] range.
            
        Raises:
            ValueError: If dataset is not loaded or if spatial coordinates cannot be found in dataset with list of available variables.
        """
        if self.dataset is None:
            raise ValueError(DATASET_NOT_LOADED_MSG)
        
        lon_names = ['lonCell', 'longitude', 'lon']
        lat_names = ['latCell', 'latitude', 'lat']
        
        lon_coords: Optional[np.ndarray] = None
        lat_coords: Optional[np.ndarray] = None
        
        for name in lon_names:
            if name in self.dataset.coords or name in self.dataset.data_vars:
                lon_coords = np.asarray(self.dataset[name].values)
                break
                
        for name in lat_names:
            if name in self.dataset.coords or name in self.dataset.data_vars:
                lat_coords = np.asarray(self.dataset[name].values)
                break
                
        if lon_coords is None or lat_coords is None:
            available_vars = list(self.dataset.coords.keys()) + list(self.dataset.data_vars.keys())
            raise ValueError(f"Could not find spatial coordinates. Available variables: {available_vars}")
        
        if np.nanmax(np.abs(lat_coords)) <= np.pi:
            lat_coords = lat_coords * 180.0 / np.pi
            lon_coords = lon_coords * 180.0 / np.pi
        
        lon_coords_flat: np.ndarray = lon_coords.ravel()
        lat_coords_flat: np.ndarray = lat_coords.ravel()
        
        lon_coords_normalized = self.normalize_longitude(lon_coords_flat)
        assert isinstance(lon_coords_normalized, np.ndarray), "Expected array from normalize_longitude"
        
        return lon_coords_normalized, lat_coords_flat
    
    def _add_spatial_coords_helper(
        self, 
        combined_ds: xr.Dataset, 
        dimensions_to_add: List[str],
        spatial_vars: List[str],
        processor_type: str
    ) -> xr.Dataset:
        """
        Shared helper method to add spatial coordinates and mesh connectivity to MPAS datasets. This method eliminates code duplication between 2D and 3D processors by providing common coordinate addition logic. It loads the grid file, adds dimensional index coordinates for specified dimensions that exist in the dataset, copies spatial variable data from grid to dataset, and handles errors gracefully with verbose output. The method is dimension-agnostic, allowing each processor to specify which dimensions and spatial variables are relevant for their data type.

        Parameters:
            combined_ds (xr.Dataset): Combined dataset to enrich with spatial coordinates.
            dimensions_to_add (List[str]): List of dimension names to add index coordinates for (e.g., ['nCells', 'nVertices']).
            spatial_vars (List[str]): List of spatial variable names to copy from grid file (e.g., ['latCell', 'lonCell']).
            processor_type (str): Type identifier for error messages ('2D' or '3D').

        Returns:
            xr.Dataset: Enriched dataset with added coordinate variables.
        """
        try:
            grid_file_ds = xr.open_dataset(self.grid_file)

            if self.verbose:
                print(f"\nGrid file loaded successfully with variables: \n{list(grid_file_ds.variables.keys())}\n")
            
            coords_to_add = {}
            data_vars_to_add = {}
            
            for dim_name in dimensions_to_add:
                if dim_name in combined_ds.sizes:
                    coords_to_add[dim_name] = (dim_name, np.arange(combined_ds.sizes[dim_name]))
                    if self.verbose:
                        print(f"Added {dim_name} index coordinate for {dim_name} dimension ({combined_ds.sizes[dim_name]} values)")
            
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
                print(f"Warning: Could not add {processor_type} spatial coordinates: {coord_error}")
                print("Continuing without additional coordinates...")
        
        return combined_ds
    
    def get_time_info(self, time_index: int, var_context: str = "") -> str:
        """
        Retrieve formatted time coordinate information for diagnostic output and plot labeling. This method delegates to MPASDateTimeUtils to extract and format timestamp information from the dataset's time coordinate at the specified index. The formatted string includes human-readable date and time suitable for plot titles, filenames, or diagnostic messages. Optional variable context can be included to customize the output message for specific variables or analysis operations. This utility method provides a consistent interface for time information retrieval across different processing classes. The method validates that a dataset has been loaded before attempting coordinate access to ensure robust error handling.

        Parameters:
            time_index (int): Zero-based index into the time dimension for which to retrieve coordinate information.
            var_context (str): Optional variable name or context string for customizing diagnostic messages (default: "").

        Returns:
            str: Formatted time information string containing human-readable timestamp appropriate for diagnostic output or plot labeling.
            
        Raises:
            ValueError: If no dataset has been loaded prior to calling this method with instruction to call load methods first.
        """
        if self.dataset is None:
            raise ValueError(DATASET_NOT_LOADED_MSG)
        
        from .utils_datetime import MPASDateTimeUtils
        return MPASDateTimeUtils.get_time_info(self.dataset, time_index, var_context, self.verbose)
    
    def parse_file_datetimes(self, diag_files: List[str]) -> List:
        """
        Parse datetime information from diagnostic filename patterns using standardized MPAS naming conventions. This method delegates to MPASDateTimeUtils to extract temporal information encoded in MPAS output filenames following standard naming patterns. The parsing handles various MPAS filename formats including diagnostic files and model output files with embedded date-time stamps. Extracted datetime objects are used for temporal sorting, coordinate assignment, and time-based file selection in multi-file loading operations. This utility method provides consistent datetime extraction across different MPAS output types and filename conventions. The parsing supports both older and newer MPAS filename formats with automatic pattern detection and robust error handling for malformed filenames.

        Parameters:
            diag_files (List[str]): List of absolute or relative paths to MPAS diagnostic or output files with embedded datetime information in filenames.

        Returns:
            List: List of datetime objects parsed from filenames in same order as input file list, suitable for temporal sorting and coordinate assignment.
        """
        from .utils_datetime import MPASDateTimeUtils
        return MPASDateTimeUtils.parse_file_datetimes(diag_files, self.verbose)
    
    def validate_time_parameters(self, time_index: int) -> Tuple[str, int, int]:
        """
        Validate time index parameter against dataset temporal dimensions with automatic bounds checking. This method delegates to MPASDateTimeUtils to perform comprehensive validation of time index requests against the loaded dataset's time dimension size. The validation detects the appropriate time dimension name handling both 'Time' and 'time' conventions used across different MPAS output formats. Negative time indices are converted to positive indices using Python convention for accessing from the end of the time series. Out-of-bounds indices are caught and reported with clear error messages indicating valid index ranges. The method returns validated time dimension name, corrected time index, and time dimension size for subsequent data extraction operations. This validation prevents index errors and provides clear diagnostic messages for invalid time requests.

        Parameters:
            time_index (int): Zero-based time index to validate, supporting negative indices for counting from end of time series.

        Returns:
            Tuple[str, int, int]: Three-element tuple containing (time_dimension_name, validated_time_index, time_dimension_size) where validated_time_index has negative indices converted to positive equivalents.
            
        Raises:
            ValueError: If dataset is not loaded with instruction to call load methods first, or if time index is out of bounds with valid range information.
        """
        if self.dataset is None:
            raise ValueError(DATASET_NOT_LOADED_MSG)
        
        from .utils_datetime import MPASDateTimeUtils
        return MPASDateTimeUtils.validate_time_parameters(self.dataset, time_index, self.verbose)
    
    def filter_by_spatial_extent(self, data: Any, lon_min: float, lon_max: float, 
                                lat_min: float, lat_max: float) -> Tuple[Any, np.ndarray]:
        """
        Filter data array by geographic extent using coordinate-based spatial masking for regional analysis. This method delegates to MPASGeographicUtils to create and apply a boolean mask selecting only data points within the specified longitude and latitude bounds. The filtering handles dateline crossing cases and ensures proper coordinate wrapping for global datasets. The method works with both xarray and numpy array inputs, automatically extracting coordinates from the loaded dataset. Filtered data retains original array structure and metadata while excluding points outside the specified geographic region. The returned boolean mask can be reused for filtering additional variables with identical spatial structure. This spatial filtering is essential for regional analysis, domain-specific plotting, and reducing data volume for focused studies.

        Parameters:
            data (Any): Data array to filter, either xarray.DataArray or numpy.ndarray with spatial dimensions matching dataset coordinate structure.
            lon_min (float): Minimum longitude bound in degrees, typically in [-180, 180] range.
            lon_max (float): Maximum longitude bound in degrees, typically in [-180, 180] range.
            lat_min (float): Minimum latitude bound in degrees, must be in [-90, 90] range.
            lat_max (float): Maximum latitude bound in degrees, must be in [-90, 90] range.

        Returns:
            Tuple[Any, np.ndarray]: Two-element tuple containing (filtered_data, spatial_mask) where filtered_data has same type as input with points outside extent masked and spatial_mask is boolean array indicating selected points.
            
        Raises:
            ValueError: If dataset is not loaded prior to filtering with instruction to call load methods first.
        """
        if self.dataset is None:
            raise ValueError(DATASET_NOT_LOADED_MSG)
        
        from .utils_geog import MPASGeographicUtils
        return MPASGeographicUtils.filter_by_spatial_extent(
            data, self.dataset, lon_min, lon_max, lat_min, lat_max)
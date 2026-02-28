#!/usr/bin/env python3

"""
MPAS Base Data Processor Module

This module provides the foundational data processing infrastructure for reading, loading, and managing MPAS atmospheric model output files with support for both diagnostic and history streams. It implements the MPASBaseDataProcessor class that serves as the parent for specialized processors (2D surface, 3D atmospheric), providing common functionality including file discovery with glob pattern matching, xarray-based dataset loading with chunk optimization for memory efficiency, time coordinate extraction and standardization, variable validation and listing, and geographic coordinate handling for MPAS unstructured mesh data. The base processor establishes consistent data access patterns across all MPASdiag processing modules, handles netCDF file operations with proper error handling, manages dataset caching for performance, and provides utility methods for time series processing and spatial subsetting. This foundational class enables rapid development of specialized diagnostic processors by inheriting common file I/O, coordinate handling, and data validation functionality while allowing customization for variable-specific extraction and computation requirements.

Classes:
    MPASBaseDataProcessor: Abstract base class providing common data processing infrastructure for all MPAS diagnostic processors.
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""
# Import standard libraries
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

# Import relevant MPASdiag utilities and constants
from .utils_datetime import MPASDateTimeUtils
from .constants import DATASET_NOT_LOADED_MSG, DIAG_GLOB

# Suppress specific warnings that are expected during MPAS data loading and processing to reduce noise in verbose output. 
warnings.filterwarnings('ignore', message='.*Shapely.*')
warnings.filterwarnings('ignore', category=UserWarning, module='cartopy')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='shapely')
warnings.filterwarnings('ignore', message='invalid value encountered in create_collection')
warnings.filterwarnings('ignore', message='The specified chunks separate the stored chunks.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*chunks.*degrade performance.*')


class MPASBaseProcessor:
    """
    This class provides common functionality for loading, processing, and analyzing MPAS unstructured mesh data with support for lazy loading and temporal analysis.
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
        # Validate and store the grid file path, which is essential for loading MPAS datasets with unstructured grid support. 
        self.grid_file = grid_file

        # Store verbosity setting for controlling diagnostic output during processing.
        self.verbose = verbose

        # Initialize dataset and data type attributes to None; these will be set during the loading process based on the successful loading strategy (xarray or UXarray).
        self.dataset = None

        # Initially set data_type to None; this will be updated to 'xarray' or 'uxarray' based on the loading strategy that succeeds
        self.data_type = None
        
        # If verbose mode is enabled, print an initialization message indicating the grid file being used and the verbosity setting.
        if self.verbose:
            print(f"Initializing MPASBaseProcessor with grid file: {grid_file}")
            print(f"Verbose mode: {self.verbose}")

        # Raise a FileNotFoundError if the specified grid file does not exist to ensure that subsequent loading operations have the necessary grid information available.
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
        # Construct the full file pattern by joining the data directory with the provided glob pattern to create an absolute search pattern for glob.
        file_pattern = os.path.join(data_dir, pattern)

        # Use glob to find files matching the pattern and sort them by filename to ensure consistent temporal ordering based on naming conventions.
        files = sorted(glob.glob(file_pattern))
        
        # Raise a FileNotFoundError if no files are found matching the pattern to provide clear feedback about missing data. 
        if not files:
            raise FileNotFoundError(f"No {file_type} found matching pattern: {file_pattern}")
        
        # Raise a ValueError if fewer than 2 files are found, as temporal analysis typically requires multiple time steps to compute trends, differences, or time series.
        if len(files) < 2:
            raise ValueError(f"Insufficient files for temporal analysis. Found {len(files)}, need at least 2.")
        
        # If verbose mode is enabled, print a summary of the discovered files with truncated listing for large file sets.
        if self.verbose:
            print(f"\nFound {len(files)} {file_type}:")

            # Print the first 5 files in the list for a concise summary without overwhelming the output when many files are present.
            for i, f in enumerate(files[:5]):  
                print(f"  {i+1}: {os.path.basename(f)}")

            # If there are more than 5 files, print a message indicating how many additional files were found without listing them all to keep the output manageable.
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more files")
        
        # Return the sorted list of file paths for use in subsequent loading and processing operations.
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
        # Initialize an empty list to store valid file paths
        valid_files = []

        # Iterate through each file path in the input list and perform validation checks
        for file_path in files:
            # Check if the file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Check if the file is readable
            if not os.access(file_path, os.R_OK):
                raise FileNotFoundError(f"File not readable: {file_path}")

            # If the file passes both checks, add it to the list of valid files
            valid_files.append(file_path)
        
        # Return the list of validated file paths for use in data loading operations
        return valid_files
    
    def _prepare_chunking_config(self, chunks: Optional[dict]) -> tuple[dict, dict]:
        """
        The helper returns a minimal `open_chunks` mapping suitable for `xr.open_mfdataset` and a `full_chunks` mapping suitable for later rechunk operations. When no custom `chunks` are provided a sensible default is returned to balance IO and memory usage for MPAS datasets. This centralizes chunk configuration logic used by various loading strategies in the processor.

        Parameters:
            chunks (Optional[dict]): Optional user-specified chunking mapping or None to use defaults.

        Returns:
            tuple[dict, dict]: (open_chunks, full_chunks) where `open_chunks` is used for opening files and `full_chunks` is used for rechunking.
        """
        # Specify default chunking strategy when no custom chunks are provided, which balances IO performance and memory usage for typical MPAS datasets. 
        if chunks is None:
            # Use 100,000 cells per chunk for nCells dimension to manage memory usage while allowing efficient access patterns
            default_chunks = {'Time': 1, 'nCells': 100000}

            # Use 1 time step per chunk for opening to ensure proper concatenation along Time dimension
            return {'Time': 1}, default_chunks
        else:
            # Use the provided chunks for opening, but ensure that only the Time dimension is included in open_chunks to avoid issues with concatenation
            open_chunks = {k: v for k, v in chunks.items() if k == 'Time'}

            # Return the chunks and open_chunks separately so that open_chunks can be used for opening files
            return open_chunks, chunks

    def _load_multifile_dataset(self, data_files: List[str], file_datetimes: List[datetime],
                                open_chunks: dict) -> xr.Dataset:
        """
        Files are combined along the `Time` dimension using a nested concat strategy and the provided `file_datetimes` are assigned as Time coordinates. The function sorts the resulting dataset by time and returns a lazily loaded xarray.Dataset ready for downstream processing. This method is the core routine for loading multiple MPAS output files into a single xarray.Dataset with proper time coordinate handling and chunking for memory efficiency. It ensures that the combined dataset is temporally ordered and that time coordinates are correctly assigned based on filename parsing. The resulting dataset is lazily loaded with chunking applied to optimize performance when working with large MPAS datasets that may not fit entirely in memory. This helper is used by primary loading strategies in the processor to create a combined dataset from multiple files while maintaining temporal integrity and efficient memory usage.

        Parameters:
            data_files (List[str]): Ordered list of file paths to open and concatenate.
            file_datetimes (List[datetime]): Corresponding list of datetime objects for each input file.
            open_chunks (dict): Chunk configuration passed to `xr.open_mfdataset`.

        Returns:
            xr.Dataset: Concatenated xarray.Dataset with assigned Time coordinates.
        """
        # Read data files into a single xarray.Dataset using open_mfdataset with the specified chunking strategy for efficient memory usage. 
        combined_ds = xr.open_mfdataset(
            data_files,
            combine='nested',
            concat_dim='Time',
            chunks=open_chunks,
            parallel=False
        )
        
        # Assign the provided datetime objects as the Time coordinates in the combined dataset
        combined_ds = combined_ds.assign_coords(Time=pd.to_datetime(file_datetimes))

        # Sort the combined dataset by the Time coordinate to ensure temporal ordering
        combined_ds = combined_ds.sortby('Time')
        
        # Return the combined dataset, which is lazily loaded with chunking applied for efficient handling of large MPAS datasets.
        return combined_ds

    def _apply_chunking(self, dataset: xr.Dataset, chunks: Optional[dict]) -> xr.Dataset:
        """
        When `chunks` is None the original dataset is returned. If chunking is requested the helper attempts `dataset.chunk(chunks)` and gracefully returns the original dataset if chunking fails for any reason.

        Parameters:
            dataset (xr.Dataset): Dataset to apply chunking to.
            chunks (Optional[dict]): Chunk mapping to apply or None to skip.

        Returns:
            xr.Dataset: Possibly chunked dataset; original dataset returned on failure.
        """
        # If no chunking configuration is provided, return the original dataset without modification.
        if chunks is None:
            return dataset
        
        try:
            # Apply chunking to the dataset based on the provided chunk mapping
            return dataset.chunk(chunks)
        except Exception:
            # If chunking fails for any reason, catch the exception and return the original dataset without chunking
            return dataset

    def _create_uxarray_dataset(self, combined_ds: xr.Dataset, 
                               data_files: List[str]) -> ux.UxDataset:
        """
        The function opens the grid file alongside the first data file to extract `uxgrid` information required to construct a `ux.UxDataset`. This enables unstructured-grid-aware indexing and utilities provided by UXarray.

        Parameters:
            combined_ds (xr.Dataset): Concatenated xarray dataset containing data variables.
            data_files (List[str]): List of original data file paths (used to locate grid info).

        Returns:
            ux.UxDataset: UXarray dataset wrapping the xarray dataset with grid metadata.
        """
        # Open the grid file using UXarray to extract the unstructured grid information
        grid_ds = ux.open_dataset(self.grid_file, data_files[0])

        # Extract the unstructured grid information from the opened grid dataset
        grid_info = grid_ds.uxgrid

        # Raise a ValueError if the grid information could not be extracted from the dataset
        if grid_info is None:
            raise ValueError("Could not extract uxgrid from dataset")

        # Return a UXarray dataset that wraps the combined xarray dataset and includes the unstructured grid information
        return ux.UxDataset(combined_ds, uxgrid=grid_info)

    def _attempt_primary_load(self, data_files: List[str], file_datetimes: List[datetime],
                             open_chunks: dict, full_chunks: Optional[dict],
                             use_pure_xarray: bool, data_type_label: str) -> Tuple[Any, str]:
        """
        The routine opens multiple files with `xr.open_mfdataset`, applies chunking, and then either returns a plain xarray.Dataset or wraps it with UXarray depending on `use_pure_xarray`. Successful loading sets instance state (`self.dataset`, `self.data_type`) and returns the dataset and type id.

        Parameters:
            data_files (List[str]): Ordered list of data files to load.
            file_datetimes (List[datetime]): Datetimes corresponding to each file.
            open_chunks (dict): Chunking mapping used during open.
            full_chunks (Optional[dict]): Full chunk mapping for rechunking.
            use_pure_xarray (bool): If True, prefer pure xarray over UXarray.
            data_type_label (str): Human-readable label for the data type (for messages).

        Returns:
            Tuple[Any, str]: (dataset_object, data_type_identifier) where dataset_object is xarray.Dataset or ux.UxDataset and data_type_identifier is 'xarray' or 'uxarray'.
        """
        # Load and combine multiple files into a single xarray.Dataset with proper time coordinate handling and chunking for memory efficiency. 
        combined_ds = self._load_multifile_dataset(data_files, file_datetimes, open_chunks)

        # Apply full chunking to the combined dataset after loading to optimize memory usage for downstream processing. 
        combined_ds = self._apply_chunking(combined_ds, full_chunks)
        
        # If verbose mode is enabled, print a summary of the combined dataset structure
        if self.verbose:
            print(f"\n{data_type_label} Dataset structure:")
            print(combined_ds)
        
        if use_pure_xarray:
            # If pure xarray is requested, skip UXarray wrapping and return the combined xarray.Dataset directly.
            if self.verbose:
                self._print_loading_success(len(data_files), combined_ds, "pure xarray", data_type_label)

            # Set the dataset to the combined xarray.Dataset 
            self.dataset = combined_ds

            # Set the data type to 'xarray' to indicate that the dataset is a plain xarray.Dataset without UXarray wrapping.
            self.data_type = 'xarray'

            # Return the combined xarray.Dataset and the string identifier 'xarray' 
            return combined_ds, 'xarray'
        else:
            # If UXarray is preferred, create a UXarray dataset by wrapping the combined xarray.Dataset with the unstructured grid information 
            final_ds = self._create_uxarray_dataset(combined_ds, data_files)
            
            # If verbose mode is enabled, print a summary of the final UXarray dataset structure
            if self.verbose:
                self._print_loading_success(len(data_files), final_ds, "UXarray", data_type_label)
            

            # Set the dataset to the final UXarray dataset
            self.dataset = final_ds

            # Set the data type to 'uxarray' to indicate that the dataset is a UXarray dataset with unstructured grid support.
            self.data_type = 'uxarray'

            # Return the final UXarray dataset and the string identifier 'uxarray' 
            return final_ds, 'uxarray'

    def _attempt_fallback_load(self, data_files: List[str], file_datetimes: List[datetime],
                              full_chunks: Optional[dict], 
                              data_type_label: str) -> Tuple[Any, str]:
        """
        This helper uses a conservative chunking strategy and retries concatenation with xarray only, which increases compatibility at the cost of some grid metadata. It sets `self.dataset` and `self.data_type` upon success.

        Parameters:
            data_files (List[str]): Ordered list of data files to load.
            file_datetimes (List[datetime]): Datetimes corresponding to each file.
            full_chunks (Optional[dict]): Full chunk mapping for rechunking.
            data_type_label (str): Human-readable label for the data type (for messages).

        Returns:
            Tuple[Any, str]: (dataset_object, 'xarray') loaded in fallback mode.
        """
        # If the primary chunking strategy fails use 1 time step per chunk for the fallback loading strategy to maximize compatibility 
        fallback_chunks = {'Time': 1}

        # Load and combine multiple files into a single xarray.Dataset using the fallback chunking strategy
        combined_ds = self._load_multifile_dataset(data_files, file_datetimes, fallback_chunks)

        # Optimize memory usage for the fallback dataset by applying the full chunking strategy after loading
        combined_ds = self._apply_chunking(combined_ds, full_chunks)
        
        # If verbose mode is enabled, print a summary of the combined dataset structure 
        if self.verbose:
            self._print_loading_success(len(data_files), combined_ds, "xarray (fallback)", data_type_label)
        
        # Set the dataset to the combined xarray.Dataset from the fallback loading strategy
        self.dataset = combined_ds

        # Set the data type to 'xarray' to indicate that the dataset is a plain xarray.Dataset without UXarray wrapping. 
        self.data_type = 'xarray'

        # Return the combined xarray.Dataset and the string identifier 'xarray' 
        return combined_ds, 'xarray'

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
        # Initialize a variable to hold the list of data files that will be loaded
        data_files: List[str]

        # Specify the file discovery method for diagnostic files
        finder = getattr(self, 'find_diagnostic_files', None)

        # Find the data files using the appropriate method
        if callable(finder):
            # Find all the available diagnostic files in the specified directory using the custom finder method 
            data_files = cast(List[str], finder(data_dir))
        else:
            # Specify the file discovery method for mpasout files
            finder2 = getattr(self, 'find_mpasout_files', None)

            if callable(finder2):
                # Find all the available mpasout files in the specified directory using the custom finder method
                data_files = cast(List[str], finder2(data_dir))
            else:
                # If neither custom finder method is defined, fall back to the default glob pattern matching for diagnostic files in the specified directory.
                data_files = self._find_files_by_pattern(data_dir, DIAG_GLOB, "diagnostic files")
        
        # Parse datetimes from the filenames of the discovered data files using the MPASDateTimeUtils utility
        file_datetimes = MPASDateTimeUtils.parse_file_datetimes(data_files, self.verbose)

        # Specify the chunking configuration for opening files and for full rechunking based on the provided `chunks` argument or defaults.
        open_chunks, full_chunks = self._prepare_chunking_config(chunks)
        
        # Attempt primary loading strategy
        try:
            return self._attempt_primary_load(
                data_files, file_datetimes, open_chunks, full_chunks, 
                use_pure_xarray, data_type_label
            )
        except Exception as e:
            if self.verbose:
                print(f"Primary {data_type_label.lower()} loading failed: {e}")
                print(f"Trying xarray fallback for {data_type_label.lower()} data...")
            
            # Attempt fallback loading strategy
            try:
                return self._attempt_fallback_load(
                    data_files, file_datetimes, full_chunks, data_type_label
                )
            except Exception as e2:
                if self.verbose:
                    print(f"Xarray fallback also failed: {e2}")
                
                # Final fallback to single file
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
    
    def _select_fallback_file(self, reference_file: str, data_files: List[str]) -> str:
        """
        If a `reference_file` is provided and exists it is returned; otherwise the first entry from `data_files` is used. This helper centralizes fallback selection logic so downstream loading functions can assume a single canonical file path.

        Parameters:
            reference_file (str): Optional reference file path provided by user.
            data_files (List[str]): Candidate data files to choose from.

        Returns:
            str: Selected file path to load in single-file fallback mode.
        """
        if reference_file and os.path.exists(reference_file):
            return reference_file
        else:
            return data_files[0]

    def _load_single_file_uxarray(self, file_path: str) -> ux.UxDataset:
        """
        This helper opens the dataset using `ux.open_dataset` which attaches the unstructured grid (`uxgrid`) and returns a `ux.UxDataset` for limited single-file operations when multi-file loading is not possible.

        Parameters:
            file_path (str): Path to the single netCDF file to open.

        Returns:
            ux.UxDataset: UXarray dataset containing grid metadata.
        """
        # Return a UXarray dataset by opening the specified file path with the grid file 
        return ux.open_dataset(self.grid_file, file_path)

    def _load_single_file_xarray(self, file_path: str) -> xr.Dataset:
        """
        This helper is used as a robust fallback when UXarray loading fails or when grid metadata is not required. It returns an xarray.Dataset instance opened in lazily-loaded mode.

        Parameters:
            file_path (str): Path to the single netCDF file to open.

        Returns:
            xr.Dataset: xarray dataset loaded from the file.
        """
        # Return an xarray dataset by opening the specified file path in lazily-loaded mode without UXarray wrapping
        return xr.open_dataset(file_path)

    def _set_dataset_and_return(self, dataset: Any, data_type: str, 
                               file_path: str, loader_desc: str) -> Tuple[Any, str]:
        """
        This helper centralizes state updates (`self.dataset`, `self.data_type`) and prints a diagnostic message when verbose is enabled. It returns the dataset and data type identifier for callers to continue processing.

        Parameters:
            dataset (Any): Loaded dataset object (xarray.Dataset or ux.UxDataset).
            data_type (str): String identifier of the dataset backend ('xarray' or 'uxarray').
            file_path (str): Path of the file that was loaded.
            loader_desc (str): Optional loader description used for messaging.

        Returns:
            Tuple[Any, str]: The same (dataset, data_type) tuple passed in.
        """
        # Specify the dataset to the loaded dataset object (either xarray.Dataset or ux.UxDataset)
        self.dataset = dataset

        # Specify the data type (e.g., 'xarray' or 'uxarray') to indicate how the dataset was loaded
        self.data_type = data_type
        
        # If verbose mode is enabled, print a message indicating successful single file load
        if self.verbose:
            print(f"Loaded single file{' with ' + loader_desc if loader_desc else ''}: {file_path}")
        
        # Return the dataset and data type 
        return dataset, data_type

    def _attempt_single_file_load(self, file_path: str) -> Tuple[Any, str]:
        """
        The helper first attempts `ux.open_dataset` and, if that fails, uses `xr.open_dataset`. It sets instance state accordingly and returns the loaded dataset with the corresponding backend identifier.

        Parameters:
            file_path (str): Path to the file to load.

        Returns:
            Tuple[Any, str]: (dataset_object, data_type_identifier) where data_type_identifier is 'uxarray' or 'xarray'.
        """
        try:
            dataset = self._load_single_file_uxarray(file_path)
            return self._set_dataset_and_return(dataset, 'uxarray', file_path, '')
        except Exception:
            dataset = self._load_single_file_xarray(file_path)
            return self._set_dataset_and_return(dataset, 'xarray', file_path, 'xarray')

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
        
        file_to_load = self._select_fallback_file(reference_file, data_files)
        return self._attempt_single_file_load(file_to_load)
    
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
        # Raise a ValueError if the dataset has not been loaded yet 
        if self.dataset is None:
            raise ValueError(DATASET_NOT_LOADED_MSG)
        
        # Return a list of data variable names from the loaded dataset, excluding coordinate variables and dimension metadata.
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
    
    def _load_grid_file(self) -> xr.Dataset:
        """
        The grid file contains mesh topology and coordinate variables required to enrich data files with spatial information. When `verbose` is enabled the function prints a short listing of variables discovered in the grid file to aid debugging and verification.

        Parameters:
            None

        Returns:
            xr.Dataset: xarray Dataset loaded from `self.grid_file` containing grid variables.
        """
        grid_file_ds = xr.open_dataset(self.grid_file)
        
        if self.verbose:
            print(f"\nGrid file loaded successfully with variables: \n{list(grid_file_ds.variables.keys())}\n")
        
        return grid_file_ds

    def _prepare_dimension_coordinates(self, combined_ds: xr.Dataset, 
                                      dimensions_to_add: List[str]) -> dict:
        """
        For each dimension name in `dimensions_to_add` that exists in `combined_ds.sizes` this helper creates a coordinate mapping from the dimension to a zero-based integer index array. This is useful for datasets that lack explicit index coordinates and simplifies downstream indexing and plotting utilities.

        Parameters:
            combined_ds (xr.Dataset): Dataset to inspect for dimension sizes.
            dimensions_to_add (List[str]): List of dimension names to add index coordinates for.

        Returns:
            dict: Mapping of coordinate name -> (dim_name, index_array) suitable for `assign_coords`.
        """
        coords_to_add = {}
        
        for dim_name in dimensions_to_add:
            if dim_name in combined_ds.sizes:
                coords_to_add[dim_name] = (dim_name, np.arange(combined_ds.sizes[dim_name]))
                if self.verbose:
                    print(f"Added {dim_name} index coordinate for {dim_name} dimension ({combined_ds.sizes[dim_name]} values)")
        
        return coords_to_add

    def _prepare_spatial_variables(self, grid_file_ds: xr.Dataset, 
                                  combined_ds: xr.Dataset, 
                                  spatial_vars: List[str]) -> dict:
        """
        This helper inspects `grid_file_ds` for requested spatial variable names (e.g., `latCell`, `lonCell`) and returns a mapping of variables that are not already present in `combined_ds.data_vars`. The returned mapping can be used to extend the data dataset with missing spatial coordinate variables.

        Parameters:
            grid_file_ds (xr.Dataset): Grid file dataset containing spatial variables.
            combined_ds (xr.Dataset): Combined data dataset which may lack spatial vars.
            spatial_vars (List[str]): Names of spatial variables to copy if available.

        Returns:
            dict: Mapping of variable name -> DataArray to add to `combined_ds`.
        """
        data_vars_to_add = {}
        
        for var_name in spatial_vars:
            if var_name in grid_file_ds.variables and var_name not in combined_ds.data_vars:
                data_vars_to_add[var_name] = grid_file_ds[var_name]
                if self.verbose:
                    print(f"Added spatial coordinate variable: {var_name}")
        
        return data_vars_to_add

    def _apply_coordinate_updates(self, combined_ds: xr.Dataset, 
                                 coords_to_add: dict, 
                                 data_vars_to_add: dict) -> xr.Dataset:
        """
        This helper assigns index coordinates (via `assign_coords`) from `coords_to_add` and copies spatial data variables from `data_vars_to_add` into `combined_ds`. Verbose messages indicate which coordinates and variables were added for user visibility.

        Parameters:
            combined_ds (xr.Dataset): Dataset to be updated with new coordinates and data variables.
            coords_to_add (dict): Mapping of coordinate names to (dim, array) pairs for `assign_coords`.
            data_vars_to_add (dict): Mapping of variable names to DataArray objects to insert into dataset.

        Returns:
            xr.Dataset: Updated dataset with applied coordinate and data variable additions.
        """
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
        
        return combined_ds

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
            # Load the grid file dataset to access spatial variables and coordinate information
            grid_file_ds = self._load_grid_file()

            # Prepare coordinate mappings for dimensions that exist in the combined dataset to add index coordinates (e.g., nCells, nVertices) for easier indexing and plotting.
            coords_to_add = self._prepare_dimension_coordinates(combined_ds, dimensions_to_add)

            # Process spatial variables to add from the grid file, ensuring we only add those that are not already present in the combined dataset to avoid overwriting existing data variables.
            data_vars_to_add = self._prepare_spatial_variables(grid_file_ds, combined_ds, spatial_vars)

            # Apply the coordinate and variable additions to the combined dataset
            combined_ds = self._apply_coordinate_updates(combined_ds, coords_to_add, data_vars_to_add)

            # Close the grid file dataset to free resources
            grid_file_ds.close()
        except Exception as coord_error:
            # Warn the user if spatial coordinates could not be added but continue processing with the original combined dataset
            if self.verbose:
                print(f"Warning: Could not add {processor_type} spatial coordinates: {coord_error}")
                print("Continuing without additional coordinates...")
        
        # Return the combined dataset 
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
        # Raise a ValueError if no dataset has been loaded before attempting to retrieve time information
        if self.dataset is None:
            raise ValueError(DATASET_NOT_LOADED_MSG)
        
        # Import the MPASDateTimeUtils class necessary for retrieving and formatting time information from the dataset's time coordinate.
        from .utils_datetime import MPASDateTimeUtils

        # Use get_time_info method from MPASDateTimeUtils to extract and format time information for the specified time index and variable context.
        return MPASDateTimeUtils.get_time_info(self.dataset, time_index, var_context, self.verbose)
    
    def parse_file_datetimes(self, diag_files: List[str]) -> List:
        """
        Parse datetime information from diagnostic filename patterns using standardized MPAS naming conventions. This method delegates to MPASDateTimeUtils to extract temporal information encoded in MPAS output filenames following standard naming patterns. The parsing handles various MPAS filename formats including diagnostic files and model output files with embedded date-time stamps. Extracted datetime objects are used for temporal sorting, coordinate assignment, and time-based file selection in multi-file loading operations. This utility method provides consistent datetime extraction across different MPAS output types and filename conventions. The parsing supports both older and newer MPAS filename formats with automatic pattern detection and robust error handling for malformed filenames.

        Parameters:
            diag_files (List[str]): List of absolute or relative paths to MPAS diagnostic or output files with embedded datetime information in filenames.

        Returns:
            List: List of datetime objects parsed from filenames in same order as input file list, suitable for temporal sorting and coordinate assignment.
        """
        # Import the MPASDateTimeUtils class necessary for datetime parsing from filenames 
        from .utils_datetime import MPASDateTimeUtils

        # Parse datetimes from the filenames using parse_file_datetimes utility 
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
        # Raise a ValueError if no dataset has been loaded before attempting to validate time dimensions
        if self.dataset is None:
            raise ValueError(DATASET_NOT_LOADED_MSG)
        
        # Import the MPASDateTimeUtils class necessary for time validation 
        from .utils_datetime import MPASDateTimeUtils

        # Validate time with validate_time_parameters utility 
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
        # Raise a ValueError if no dataset has been loaded before attempting to filter by spatial extent
        if self.dataset is None:
            raise ValueError(DATASET_NOT_LOADED_MSG)
        
        # Import the MPASGeographicUtils class necessary for performing spatial filtering based on geographic coordinates. 
        from .utils_geog import MPASGeographicUtils

        # Use filter_by_spatial_extent method from MPASGeographicUtils to apply a spatial mask to the input data
        return MPASGeographicUtils.filter_by_spatial_extent(
            data, self.dataset, lon_min, lon_max, lat_min, lat_max)
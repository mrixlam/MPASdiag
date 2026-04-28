#!/usr/bin/env python3

"""
MPASdiag Core Processing Module: Base Data Processor

This module defines the MPASBaseProcessor class, which serves as the foundational component for loading and managing MPAS datasets in MPASdiag. It provides common functionality for discovering data files, parsing datetimes, configuring chunking strategies, and implementing robust loading workflows with multiple strategies and fallbacks. The base processor is designed to be extended by specialized 2D and 3D processors that implement specific diagnostic computations while leveraging the shared loading and dataset management capabilities defined in this class. 
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""
# Import standard libraries
import os
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
    """ Base class for MPAS data processing with common functionality for loading datasets, managing file I/O, and defining shared attributes and methods for specialized 2D and 3D processors."""
    
    def __init__(self: 'MPASBaseProcessor', 
                 grid_file: str, 
                 verbose: bool = True) -> None:
        """
        This constructor initializes the MPASBaseProcessor with the specified grid file and verbosity setting. It validates the existence of the grid file, which is essential for loading MPAS datasets with unstructured grid support, and sets up attributes for managing the dataset and data type. The constructor also provides diagnostic feedback in verbose mode about the initialization process and raises an error if the grid file is not found to ensure that subsequent loading operations have the necessary grid information available. 

        Parameters:
            grid_file (str): Absolute path to the MPAS grid file, which is required for loading datasets with unstructured grid support using UXarray.
            verbose (bool): Flag to enable verbose output for diagnostic messages during processing (default: True). 

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
    
    @staticmethod
    def _get_plain_dataset(dataset: xr.Dataset) -> xr.Dataset:
        """
        This static helper method attempts to convert the provided dataset to a plain xarray.Dataset if it is not already one. It checks if the input dataset is not an instance of xarray.Dataset but can be converted to one (e.g., if it is a UXarray dataset), and if so, it creates a new xarray.Dataset from the input. If the input dataset is already an xarray.Dataset or cannot be converted, it returns the original dataset as is. This method provides a way to ensure that downstream processing can work with a plain xarray.Dataset when needed, while still allowing for flexibility in handling different dataset types that may be used in MPASdiag workflows. 

        Parameters:
            dataset (xr.Dataset): The input dataset, which may be a UXarray dataset or another type that can be converted to an xarray.Dataset. 

        Returns:
            xr.Dataset: A plain xarray.Dataset if the input can be converted, or the original dataset if it cannot be converted to an xarray.Dataset.
        """
        if type(dataset) is not xr.Dataset and isinstance(dataset, xr.Dataset):
            return xr.Dataset(dataset)
        return dataset  
      
    def _find_files_by_pattern(self: 'MPASBaseProcessor', 
                               data_dir: str, 
                               pattern: str, 
                               file_type: str) -> List[str]:
        """
        This helper method finds files in the specified directory that match the given glob pattern, sorts them by filename to ensure consistent temporal ordering, and validates that a sufficient number of files are found for temporal analysis. It constructs the full file pattern by joining the data directory with the provided glob pattern, uses glob to find matching files, and raises appropriate errors if no files are found or if fewer than 2 files are found (since temporal analysis typically requires multiple time steps). If verbose mode is enabled, it prints a summary of the discovered files with truncated listing for large file sets. The method returns a sorted list of file paths matching the specified pattern for use in subsequent loading and processing operations. 

        Parameters:
            data_dir (str): Absolute path to the directory containing the data files to search.
            pattern (str): Glob pattern to match files (e.g., "mpas_diag_*.nc" for diagnostic files).
            file_type (str): Human-readable label for the type of files being searched (used in error messages and verbose output). 

        Returns:
            List[str]: A sorted list of absolute file paths that match the specified pattern in the given directory, ready for loading and processing. 
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
            for i, filename in enumerate(files[:5]):
                print(f"  {i+1}: {os.path.basename(filename)}")

            # If there are more than 5 files, print a message indicating how many additional files were found without listing them all to keep the output manageable.
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more files")
        
        # Return the sorted list of file paths for use in subsequent loading and processing operations.
        return files

    def validate_files(self: 'MPASBaseProcessor', 
                       files: List[str]) -> List[str]:
        """
        This helper method validates that the provided list of file paths exists and is readable. It iterates through each file path in the input list, checks if the file exists using `os.path.exists`, and checks if the file is readable using `os.access` with the `os.R_OK` flag. If any file does not exist or is not readable, it raises a FileNotFoundError with a message indicating which file is problematic. If all files pass the validation checks, it returns the original list of file paths for use in data loading operations. This method ensures that subsequent loading attempts have valid file paths to work with, reducing the likelihood of runtime errors during data loading due to missing or inaccessible files. 

        Parameters:
            files (List[str]): A list of file paths to validate for existence and readability before attempting to load data from them. 

        Returns:
            List[str]: The original list of file paths if all files are valid, or raises an error if any file is missing or not readable. 
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
    
    def _prepare_chunking_config(self: 'MPASBaseProcessor', 
                                 chunks: Optional[dict]) -> tuple[dict, dict]:
        """
        This helper method prepares the chunking configuration for both opening files and for full rechunking after loading based on the provided `chunks` argument or defaults. If `chunks` is None, it specifies a default chunking strategy that balances IO performance and memory usage for typical MPAS datasets, using 1 time step per chunk for opening to ensure proper concatenation along the Time dimension and 100,000 cells per chunk for the nCells dimension to manage memory usage while allowing efficient access patterns. If a custom chunking configuration is provided, it separates the chunking configuration into `open_chunks`, which includes only the Time dimension to avoid issues with concatenation when opening files, and `full_chunks`, which includes all specified chunks for rechunking after loading. This method returns both the `open_chunks` and `full_chunks` configurations to be used in the loading workflow, ensuring that the appropriate chunking strategies are applied at each stage of the loading process for optimal performance and memory management. 

        Parameters:
            chunks (Optional[dict]): Custom chunking strategy dictionary with dimension names as keys and chunk sizes as values, or None for default chunking. 

        Returns:
            tuple[dict, dict]: A tuple containing two dictionaries: (open_chunks, full_chunks) where open_chunks is the chunking configuration to use when opening files with `xr.open_mfdataset` (typically including only the Time dimension), and full_chunks is the chunking configuration to use for rechunking after loading (which may include all specified chunks for memory optimization). 
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

    def _load_multifile_dataset(self: 'MPASBaseProcessor', 
                                data_files: List[str], 
                                file_datetimes: List[datetime],
                                open_chunks: dict,
                                drop_variables: Optional[List[str]] = None) -> xr.Dataset:
        """
        This helper method loads and combines multiple files into a single xarray.Dataset with proper time coordinate handling and chunking for memory efficiency. It uses `xr.open_mfdataset` to read the specified data files, combining them along the Time dimension using a nested concatenation strategy. The provided `open_chunks` configuration is applied when opening the files to ensure that chunking is compatible with the concatenation process. After loading, it assigns the provided datetime objects as the Time coordinates in the combined dataset and sorts the dataset by the Time coordinate to ensure temporal ordering. The resulting combined dataset is lazily loaded with chunking applied for efficient handling of large MPAS datasets, and it is returned for use in subsequent processing steps. 

        Parameters:
            data_files (List[str]): Ordered list of data files to load and combine into a single dataset.
            file_datetimes (List[datetime]): Datetimes corresponding to each file, used for assigning Time coordinates in the combined dataset.
            open_chunks (dict): Chunking configuration to use when opening files with `xr.open_mfdataset`, typically including only the Time dimension to ensure proper concatenation.
            drop_variables (Optional[List[str]]): List of variable names to exclude from loading, reducing memory usage (default: None). 

        Returns:
            xr.Dataset: A combined xarray.Dataset containing data from all specified files, with Time coordinates assigned and sorted, and chunking applied for efficient memory usage. 
        """
        # Build kwargs for open_mfdataset
        open_mfdataset_kwargs: Dict[str, Any] = dict(
            combine='nested',
            concat_dim='Time',
            chunks=open_chunks,
            parallel=False,
        )

        # If a list of variables to drop is provided, include it in the kwargs
        if drop_variables:
            open_mfdataset_kwargs['drop_variables'] = drop_variables

        # Read data files into a single xarray.Dataset using open_mfdataset with the specified chunking strategy for efficient memory usage.
        combined_ds = xr.open_mfdataset(data_files, **open_mfdataset_kwargs)
        
        # Assign the provided datetime objects as the Time coordinates in the combined dataset
        combined_ds = combined_ds.assign_coords(Time=pd.to_datetime(file_datetimes))

        # Sort the combined dataset by the Time coordinate to ensure temporal ordering
        combined_ds = combined_ds.sortby('Time')
        
        # Return the combined dataset, which is lazily loaded with chunking applied for efficient handling of large MPAS datasets.
        return combined_ds

    def _apply_chunking(self: 'MPASBaseProcessor', 
                        dataset: xr.Dataset, 
                        chunks: Optional[dict]) -> xr.Dataset:
        """
        This helper method applies chunking to the provided xarray.Dataset based on the specified chunking configuration. If a valid chunking configuration is provided, it uses the `chunk` method of xarray to apply the specified chunk sizes to the dataset dimensions, enabling efficient memory usage and lazy loading for large datasets. If no chunking configuration is provided (i.e., if `chunks` is None), it returns the original dataset without modification. The method also includes error handling to catch any exceptions that may arise during the chunking process, such as issues with incompatible chunk sizes or dataset structure, and in such cases, it returns the original dataset without applying chunking to ensure that processing can continue even if chunking fails. This approach allows for flexible handling of datasets with or without chunking based on user preferences and system capabilities. 

        Parameters:
            dataset (xr.Dataset): The xarray.Dataset to which chunking should be applied based on the provided configuration.
            chunks (Optional[dict]): Chunking configuration dictionary with dimension names as keys and chunk sizes as values, or None to indicate that no chunking should be applied. 

        Returns:
            xr.Dataset: The input dataset with chunking applied according to the provided configuration, or the original dataset if no chunking is specified or if an error occurs during chunking. 
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

    def _create_uxarray_dataset(self: 'MPASBaseProcessor', 
                                combined_ds: xr.Dataset, 
                                data_files: List[str]) -> ux.UxDataset:
        """
        This helper method creates a UXarray dataset by wrapping the provided combined xarray.Dataset and including the unstructured grid information extracted from the grid file. It opens the grid file using UXarray to extract the unstructured grid metadata, which is essential for enabling advanced processing capabilities that leverage the grid topology. The method checks if the grid information was successfully extracted and raises a ValueError if it could not be obtained. If successful, it returns a UXarray dataset that wraps the combined xarray.Dataset and includes the unstructured grid information, allowing for enhanced spatial processing of MPAS datasets with unstructured grids. This method is used in the primary loading strategy when UXarray is preferred for loading multi-file datasets. 

        Parameters:
            combined_ds (xr.Dataset): The combined xarray.Dataset that was loaded and processed from multiple files, which will be wrapped in a UXarray dataset with grid information.
            data_files (List[str]): List of data files that were loaded, used for diagnostic purposes when opening the grid file with UXarray. 

        Returns:
            ux.UxDataset: A UXarray dataset that wraps the combined xarray.Dataset and includes the unstructured grid information extracted from the grid file, enabling advanced processing capabilities for MPAS datasets with unstructured grids. 
        """
        # Open the grid file using UXarray to extract the unstructured grid information
        # Pass decode_times=False because MPAS grid files may use 'seconds since 0000-01-01'
        # which is out of range for numpy, but the time variable is not needed for grid topology.
        grid_ds = ux.open_dataset(self.grid_file, data_files[0], grid_kwargs={'decode_times': False})

        # Extract the unstructured grid information from the opened grid dataset
        grid_info = grid_ds.uxgrid

        # Raise a ValueError if the grid information could not be extracted from the dataset
        if grid_info is None:
            raise ValueError("Could not extract uxgrid from dataset")

        # Return a UXarray dataset that wraps the combined xarray dataset and includes the unstructured grid information
        return ux.UxDataset(combined_ds, uxgrid=grid_info)

    def _attempt_primary_load(self: 'MPASBaseProcessor', 
                              data_files: List[str], 
                              file_datetimes: List[datetime], 
                              open_chunks: dict, 
                              full_chunks: Optional[dict], 
                              use_pure_xarray: bool, 
                              data_type_label: str,
                              drop_variables: Optional[List[str]] = None) -> Tuple[Any, str]:
        """
        This helper method implements the primary loading strategy for MPAS datasets, which attempts to load and combine multiple files into a single dataset using either pure xarray or UXarray based on the `use_pure_xarray` flag. It first loads the files into a combined xarray.Dataset with proper time coordinate handling and chunking for memory efficiency. If `use_pure_xarray` is True, it returns the combined xarray.Dataset directly without UXarray wrapping. If `use_pure_xarray` is False, it creates a UXarray dataset by wrapping the combined xarray.Dataset with the unstructured grid information extracted from the grid file. The method also includes diagnostic feedback in verbose mode about the loading process and the resulting dataset structure, and it sets the instance attributes for the loaded dataset and data type based on the successful loading strategy. This primary loading approach is designed to leverage UXarray's capabilities when possible while still allowing for a pure xarray fallback when requested by the user. 

        Parameters:
            data_files (List[str]): Ordered list of data files to load and combine into a single dataset.
            file_datetimes (List[datetime]): Datetimes corresponding to each file, used for assigning Time coordinates in the combined dataset.
            open_chunks (dict): Chunking configuration to use when opening files with `xr.open_mfdataset`, typically including only the Time dimension to ensure proper concatenation. 
            full_chunks (Optional[dict]): Full chunk mapping for rechunking after loading, which may include all specified chunks for memory optimization.
            use_pure_xarray (bool): Flag to force pure xarray backend instead of UXarray for simplified processing without unstructured grid support.
            data_type_label (str): Human-readable label for the data type being loaded (used in verbose output messages).
            drop_variables (Optional[List[str]]): List of variable names to exclude from loading (default: None). 

        Returns:
            Tuple[Any, str]: (dataset_object, data_type_identifier) where dataset_object is either an xarray.Dataset or a ux.UxDataset depending on the loading strategy that succeeds, and data_type_identifier is a string indicating the backend used for loading ('xarray' or 'uxarray'). 
        """
        # Load and combine multiple files into a single xarray.Dataset with proper time coordinate handling and chunking for memory efficiency. 
        combined_ds = self._load_multifile_dataset(data_files, file_datetimes, open_chunks, drop_variables=drop_variables)

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

    def _attempt_fallback_load(self: 'MPASBaseProcessor', 
                               data_files: List[str], 
                               file_datetimes: List[datetime], 
                               full_chunks: Optional[dict], 
                               data_type_label: str,
                               drop_variables: Optional[List[str]] = None) -> Tuple[Any, str]:
        """
        This helper method implements the fallback loading strategy for MPAS datasets, which attempts to load and combine multiple files into a single xarray.Dataset using a more conservative approach that is compatible with a wider range of file formats and system configurations. It uses a fallback chunking strategy that applies 1 time step per chunk when opening files to maximize compatibility with different file structures and concatenation requirements. After loading the combined dataset, it applies the full chunking strategy for memory optimization. The method also includes diagnostic feedback in verbose mode about the loading process and the resulting dataset structure, and it sets the instance attributes for the loaded dataset and data type based on the successful loading strategy. This fallback loading approach is designed to provide a robust alternative when the primary loading strategy fails, ensuring that users can still access their data for analysis even if UXarray loading is not possible. 

        Parameters:
            data_files (List[str]): Ordered list of data files to load and combine into a single dataset.
            file_datetimes (List[datetime]): Datetimes corresponding to each file, used for assigning Time coordinates in the combined dataset.
            full_chunks (Optional[dict]): Full chunk mapping for rechunking after loading, which may include all specified chunks for memory optimization.
            data_type_label (str): Human-readable label for the data type being loaded (used in verbose output messages).
            drop_variables (Optional[List[str]]): List of variable names to exclude from loading (default: None).

        Returns:
            Tuple[Any, str]: (dataset_object, data_type_identifier) where dataset_object is an xarray.Dataset loaded using the fallback strategy, and data_type_identifier is the string 'xarray' to indicate that the dataset is a plain xarray.Dataset without UXarray wrapping. 
        """
        # If the primary chunking strategy fails use 1 time step per chunk for the fallback loading strategy to maximize compatibility 
        fallback_chunks = {'Time': 1}

        # Load and combine multiple files into a single xarray.Dataset using the fallback chunking strategy
        combined_ds = self._load_multifile_dataset(data_files, file_datetimes, fallback_chunks, drop_variables=drop_variables)

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

    def _discover_data_files(self: 'MPASBaseProcessor', 
                             data_dir: str) -> List[str]:
        """
        This helper method discovers the relevant data files in the specified directory using custom finder methods if available or a default glob pattern. It first checks if the class has a method named `find_diagnostic_files` and if it is callable, in which case it uses that method to find diagnostic files. If not, it checks for a method named `find_mpasout_files` and uses it if available. If neither custom finder method is present, it falls back to using the `_find_files_by_pattern` method with a default glob pattern defined by `DIAG_GLOB` to search for diagnostic files in the specified directory. This approach allows for flexible file discovery based on the specific needs of different processors while still providing a robust default mechanism for finding files when custom methods are not defined. The method returns a list of file paths that were discovered for loading.

        Parameters:
            data_dir (str): Absolute path to the directory containing the data files to discover.

        Returns:
            List[str]: A list of file paths that were discovered using the appropriate method for finding diagnostic files in the specified directory, ready for validation and loading.
        """
        diag_file_finder = getattr(self, 'find_diagnostic_files', None)

        if callable(diag_file_finder):
            return cast(List[str], diag_file_finder(data_dir))

        mpasout_file_finder = getattr(self, 'find_mpasout_files', None)

        if callable(mpasout_file_finder):
            return cast(List[str], mpasout_file_finder(data_dir))

        return self._find_files_by_pattern(data_dir, DIAG_GLOB, "diagnostic files")

    def _load_data(self: 'MPASBaseProcessor', 
                   data_dir: str, 
                   use_pure_xarray: bool = False,
                   reference_file: str = "", 
                   chunks: Optional[dict] = None, 
                   data_type_label: str = "data",
                   variables: Optional[List[str]] = None) -> Tuple[Any, str]:
        """
        This method implements a robust loading workflow for MPAS datasets that includes multiple strategies and fallbacks to ensure that data can be loaded successfully under a variety of conditions. It first discovers the relevant data files in the specified directory using custom finder methods if available or a default glob pattern. It then parses datetimes from the filenames to assign proper Time coordinates in the dataset. The method prepares chunking configurations for both opening files and for full rechunking after loading based on the provided `chunks` argument or defaults. It attempts the primary loading strategy, which uses either pure xarray or UXarray based on the `use_pure_xarray` flag, and if that fails, it falls back to a more conservative xarray loading approach with a different chunking strategy. If both multi-file loading strategies fail, it attempts to load a single file as a final fallback. Throughout the process, it provides diagnostic feedback in verbose mode about the loading steps and the resulting dataset structure, and it sets instance attributes for the loaded dataset and data type based on the successful loading strategy. This method is designed to maximize the chances of successfully loading MPAS datasets while providing clear feedback to users about the loading process and any issues encountered. 

        Parameters:
            data_dir (str): Absolute path to the directory containing the data files to load.
            use_pure_xarray (bool): Flag to force pure xarray backend instead of UXarray for simplified processing without unstructured grid support (default: False).
            reference_file (str): Optional absolute path to a specific reference file for single-file loading if multi-file loading fails, or an empty string to indicate that the first file from the discovered data files should be used as the fallback (default: "").
            chunks (Optional[dict]): Custom chunking strategy dictionary with dimension names as keys and chunk sizes as values, or None for default chunking (default: None).
            data_type_label (str): Human-readable label for the type of data being loaded (e.g., "Diagnostic data" or "Model output") used in verbose output messages (default: "data").
            variables (Optional[List[str]]): List of variable names to retain. If provided, all other data variables are dropped at load time to reduce memory usage (default: None). 

        Returns:
            Tuple[Any, str]: (dataset_object, data_type_identifier) where dataset_object is the loaded dataset (either an xarray.Dataset or a ux.UxDataset depending on the loading strategy that succeeds), and data_type_identifier is a string indicating the backend used for loading ('xarray' or 'uxarray'). 
        """
        data_files = self._discover_data_files(data_dir)
        
        # Parse datetimes from the filenames of the discovered data files using the MPASDateTimeUtils utility
        file_datetimes = MPASDateTimeUtils.parse_file_datetimes(data_files, self.verbose)

        # Specify the chunking configuration for opening files and for full rechunking based on the provided `chunks` argument or defaults.
        open_chunks, full_chunks = self._prepare_chunking_config(chunks)
        
        # Compute list of variables to drop if a selective variable list was provided
        drop_variables = None

        if variables is not None:
            try:
                with xr.open_dataset(data_files[0]) as probe_ds:
                    all_data_vars = list(probe_ds.data_vars)

                variables_to_keep = set(variables)
                drop_variables = [v for v in all_data_vars if v not in variables_to_keep]

                if self.verbose and drop_variables:
                    print(f"Selective loading: keeping {len(variables_to_keep)} variable(s), dropping {len(drop_variables)} from data files")
            except Exception:
                drop_variables = None
        
        # Attempt primary loading strategy
        try:
            return self._attempt_primary_load(
                data_files, file_datetimes, open_chunks, full_chunks, 
                use_pure_xarray, data_type_label, drop_variables=drop_variables
            )
        except Exception as e:
            if self.verbose:
                print(f"Primary {data_type_label.lower()} loading failed: {e}")
                print(f"Trying xarray fallback for {data_type_label.lower()} data...")
            
            # Attempt fallback loading strategy
            try:
                return self._attempt_fallback_load(
                    data_files, file_datetimes, full_chunks, data_type_label, drop_variables=drop_variables
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

    def _print_loading_success(self: 'MPASBaseProcessor', 
                               num_files: int, 
                               dataset: Any, 
                               loader_type: str, 
                               data_type_label: str) -> None:
        """
        This helper method prints a summary message indicating successful loading of the dataset, including the number of files loaded, the structure of the combined dataset, the time range covered by the dataset, and the memory usage characteristics based on whether chunking is applied. It takes the number of files that were successfully loaded, the loaded dataset object, a string identifier of the loader type used for loading, and a human-readable label for the type of data being loaded to include in the output message. This method provides clear feedback to users about the successful loading process and the resulting dataset structure, which can be helpful for diagnosing issues or confirming that the expected data was loaded correctly. 

        Parameters:
            num_files (int): The number of files that were successfully loaded and combined into the dataset, used for reporting in the success message.
            dataset (Any): The loaded dataset object (either an xarray.Dataset or a ux.UxDataset) that was successfully loaded, used for reporting the structure and time range in the success message.
            loader_type (str): A string identifier of the loader type used for loading the dataset (e.g., "UXarray", "xarray (fallback)", or "pure xarray") to include in the success message for verbose output.
            data_type_label (str): A human-readable label for the type of data being loaded (e.g., "Diagnostic data" or "Model output") used in the success message to specify what type of data was loaded. 

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
    
    def _select_fallback_file(self: 'MPASBaseProcessor', 
                              reference_file: str, 
                              data_files: List[str]) -> str:
        """
        This helper method selects a single file to load for the fallback loading strategy when multi-file loading fails. It checks if a specific reference file was provided and if it exists, and if so, it returns that file path for loading. If the reference file is not provided or does not exist, it falls back to using the first file from the list of discovered data files as the default fallback option. This method ensures that there is a valid file path to attempt loading in the final fallback strategy, allowing users to still access their data even when multi-file loading is not successful. 

        Parameters:
            reference_file (str): Optional path to a single reference file to use for loading if multi-file loading fails, or an empty string to indicate that the first file from the data_files list should be used as the fallback.
            data_files (List[str]): List of absolute paths to data files that were discovered, with the first file used as the default fallback if the reference_file is not provided or does not exist. 

        Returns:
            str: The file path that should be used for the single-file loading fallback strategy, either the provided reference file if it exists or the first file from the discovered data files. 
        """
        if reference_file and os.path.exists(reference_file):
            return reference_file
        else:
            return data_files[0]

    def _load_single_file_uxarray(self: 'MPASBaseProcessor', 
                                  file_path: str) -> ux.UxDataset:
        """
        This helper method attempts to load a single file using UXarray, which provides advanced capabilities for handling unstructured grid datasets. It opens the specified file path with UXarray, using the grid file to extract the unstructured grid metadata necessary for proper dataset construction. The method is designed to be used as part of a fallback loading strategy when multi-file loading fails, allowing users to still access their data with UXarray's capabilities even if only a single file can be loaded. It returns a UXarray dataset containing the data from the specified file along with the associated grid metadata, enabling enhanced spatial processing of MPAS datasets with unstructured grids. 

        Parameters:
            file_path (str): Path to the single netCDF file to open with UXarray, which will be wrapped in a UXarray dataset with grid information extracted from the grid file. 

        Returns:
            ux.UxDataset: A UXarray dataset that contains the data from the specified file along with the unstructured grid metadata extracted from the grid file, enabling advanced processing capabilities for MPAS datasets with unstructured grids even when only a single file can be loaded. 
        """
        # Return a UXarray dataset by opening the specified file path with the grid file 
        return ux.open_dataset(self.grid_file, file_path, grid_kwargs={'decode_times': False})

    def _load_single_file_xarray(self: 'MPASBaseProcessor', 
                                 file_path: str) -> xr.Dataset:
        """
        This helper method attempts to load a single file using xarray, which provides a more robust loading approach that is compatible with a wider range of file formats and system configurations. It opens the specified file path with xarray in lazily-loaded mode without UXarray wrapping, allowing users to access their data even if UXarray loading fails due to issues with grid metadata or file format. The method is designed to be used as part of a fallback loading strategy when multi-file loading fails, providing a last-resort option for users to still access their data for analysis with reduced temporal coverage but increased compatibility. It returns an xarray dataset loaded from the specified file, which can be used for plotting and analysis operations even without the advanced capabilities provided by UXarray. 

        Parameters:
            file_path (str): Path to the single netCDF file to open with xarray, which will be loaded as a plain xarray.Dataset without UXarray wrapping. 

        Returns:
            xr.Dataset: An xarray dataset loaded from the specified file, providing a more robust loading approach compatible with a wider range of file formats and system configurations.
        """
        # Return an xarray dataset by opening the specified file path in lazily-loaded mode without UXarray wrapping
        return xr.open_dataset(file_path)

    def _set_dataset_and_return(self: 'MPASBaseProcessor', 
                                dataset: Any, 
                                data_type: str, 
                                file_path: str, 
                                loader_desc: str) -> Tuple[Any, str]:
        """
        This helper method sets the instance attributes for the loaded dataset and data type based on the successful loading of a single file, and returns the dataset and data type as a tuple. It takes the loaded dataset object (either an xarray.Dataset or a ux.UxDataset), a string identifier of the data type used for loading, the file path that was loaded, and an optional description of the loader used for verbose output. The method updates the instance attributes `self.dataset` and `self.data_type` to reflect the successfully loaded dataset and its type, and if verbose mode is enabled, it prints a message indicating that the single file was successfully loaded along with the loader description if provided. Finally, it returns a tuple containing the loaded dataset object and the data type identifier, which can be returned to the caller for further processing. 

        Parameters:
            dataset (Any): The loaded dataset object (either an xarray.Dataset or a ux.UxDataset) that was successfully loaded from the single file.
            data_type (str): A string identifier of the data type used for loading ('xarray' or 'uxarray') to indicate how the dataset was loaded.
            file_path (str): The file path that was successfully loaded, used for reporting in the verbose output message.
            loader_desc (str): An optional description of the loader used for verbose output (e.g., "xarray" or "UXarray") to include in the success message when verbose mode is enabled. 

        Returns:
            Tuple[Any, str]: A tuple containing the loaded dataset object (either an xarray.Dataset or a ux.UxDataset depending on which loading strategy succeeded) and a string identifier of the data type used for loading ('xarray' or 'uxarray'). 
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

    def _attempt_single_file_load(self: 'MPASBaseProcessor', 
                                  file_path: str) -> Tuple[Any, str]:
        """
        This helper method attempts to load a single file using both UXarray and xarray backends in sequence. It first tries to load the specified file with UXarray, and if that attempt raises an exception (e.g., due to issues with grid metadata or file format), it catches the exception and then attempts to load the same file with xarray as a fallback. If the UXarray loading succeeds, it sets the dataset and data type attributes accordingly and returns them. If the UXarray loading fails but the xarray loading succeeds, it sets the dataset and data type attributes for the xarray-loaded dataset and returns them. If both loading attempts fail, it raises an exception that can be caught by the caller for further handling. This method provides a robust mechanism for attempting to load a single file with both backends, maximizing the chances of successfully accessing the data even when multi-file loading strategies are not successful. 

        Parameters:
            file_path (str): The file path to the single netCDF file that should be attempted for loading with both UXarray and xarray backends. 

        Returns:
            Tuple[Any, str]: A tuple containing the loaded dataset object (either an xarray.Dataset or a ux.UxDataset depending on which loading strategy succeeds) and a string identifier of the data type used for loading ('xarray' or 'uxarray'). 
        """
        try:
            dataset = self._load_single_file_uxarray(file_path)
            return self._set_dataset_and_return(dataset, 'uxarray', file_path, '')
        except Exception:
            dataset = self._load_single_file_xarray(file_path)
            return self._set_dataset_and_return(dataset, 'xarray', file_path, 'xarray')

    def _load_single_file_fallback(self: 'MPASBaseProcessor', 
                                   reference_file: str, 
                                   data_files: List[str]) -> Tuple[Any, str]:
        """
        This helper method implements the final fallback loading strategy for MPAS datasets, which attempts to load a single file using both UXarray and xarray backends in sequence. It first selects a file to load based on the provided reference file or the first file from the discovered data files. It then tries to load the selected file with UXarray, and if that attempt raises an exception (e.g., due to issues with grid metadata or file format), it catches the exception and then attempts to load the same file with xarray as a fallback. If the UXarray loading succeeds, it sets the dataset and data type attributes accordingly and returns them. If the UXarray loading fails but the xarray loading succeeds, it sets the dataset and data type attributes for the xarray-loaded dataset and returns them. If both loading attempts fail, it raises an exception that can be caught by the caller for further handling. This method provides a robust mechanism for attempting to load a single file with both backends, maximizing the chances of successfully accessing the data even when multi-file loading strategies are not successful. 

        Parameters:
            reference_file (str): Optional path to a single reference file to use for loading if multi-file loading fails, or an empty string to indicate that the first file from the data_files list should be used as the fallback.
            data_files (List[str]): List of absolute paths to data files that were discovered, with the first file used as the default fallback if the reference_file is not provided or does not exist.

        Returns:
            Tuple[Any, str]: A tuple containing the loaded dataset object (either an xarray.Dataset or a ux.UxDataset depending on which loading strategy succeeds) and a string identifier of the data type used for loading ('xarray' or 'uxarray'). 
        """
        if self.verbose:
            print("Falling back to single-file loading (limited functionality)...")
        
        file_to_load = self._select_fallback_file(reference_file, data_files)
        return self._attempt_single_file_load(file_to_load)
    
    def get_available_variables(self: 'MPASBaseProcessor') -> List[str]:
        """
        This method returns a list of available data variable names in the loaded dataset, excluding coordinate variables and dimension metadata. It checks if a dataset has been loaded and raises a ValueError if not. If a dataset is loaded, it retrieves the names of the data variables from the dataset's `data_vars` attribute, which contains only the data variables and excludes coordinates and dimensions. The method returns this list of data variable names, which can be used for diagnostic purposes or to inform users about what variables are available for plotting and analysis. 

        Parameters:
            None

        Returns:
            List[str]: A list of available data variable names in the loaded dataset, excluding coordinate variables and dimension metadata.
        """
        # Raise a ValueError if the dataset has not been loaded yet 
        if self.dataset is None:
            raise ValueError(DATASET_NOT_LOADED_MSG)
        
        # Return a list of data variable names from the loaded dataset, excluding coordinate variables and dimension metadata.
        return list(self.dataset.data_vars.keys())
    
    def normalize_longitude(self: 'MPASBaseProcessor', 
                            lon: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        This method normalizes longitude values to the standard [-180, 180] degree range. It takes longitude values that may be in the [0, 360] range or other ranges and applies a modulo operation to wrap them into the desired range. The method can handle both scalar longitude values and numpy arrays of longitude values, returning the normalized longitude(s) with the same type and dimensions as the input. This normalization is important for ensuring that longitude values are consistent and compatible with geographic plotting and analysis operations that typically expect longitudes in the [-180, 180] degree range. 

        Parameters:
            lon (Union[float, np.ndarray]): Longitude value(s) to normalize, which can be a single float or a numpy array of longitude values in any range (e.g., [0, 360]). 

        Returns:
            Union[float, np.ndarray]: Normalized longitude value(s) in the standard [-180, 180] degree range, with the same type and dimensions as the input. 
        """
        lon = np.asarray(lon)
        normalized_lon = ((lon + 180) % 360) - 180

        if normalized_lon.ndim == 0:
            return float(normalized_lon)

        return normalized_lon
    
    def validate_geographic_extent(self: 'MPASBaseProcessor', 
                                   extent: Tuple[float, float, float, float]) -> bool:
        """
        This method validates a geographic extent defined by longitude and latitude bounds to ensure that the values are within physically valid ranges. It checks that the longitude bounds (lon_min and lon_max) are within the range of [-180, 180] degrees and that the latitude bounds (lat_min and lat_max) are within the range of [-90, 90] degrees. Additionally, it verifies that the minimum longitude is less than the maximum longitude and that the minimum latitude is less than the maximum latitude to ensure a valid rectangular geographic extent. The method returns True if all checks pass, indicating that the provided extent is valid for geographic plotting and analysis, and returns False if any of the checks fail. This validation is important for preventing errors in spatial processing and ensuring that users provide meaningful geographic extents for their analyses. 

        Parameters:
            extent (Tuple[float, float, float, float]): A tuple defining the geographic extent as (lon_min, lon_max, lat_min, lat_max) to be validated against physically valid ranges for longitude and latitude. 

        Returns:
            bool: True if the provided geographic extent is valid (longitude bounds within [-180, 180], latitude bounds within [-90, 90], and min values less than max values), and False if any of the checks fail. 
        """
        lon_min, lon_max, lat_min, lat_max = extent
        
        if not (-180 <= lon_min <= 180) or not (-180 <= lon_max <= 180):
            return False
            
        if not (-90 <= lat_min <= 90) or not (-90 <= lat_max <= 90):
            return False
            
        if lon_min >= lon_max or lat_min >= lat_max:
            return False
            
        return True
    
    def extract_spatial_coordinates(self: 'MPASBaseProcessor') -> Tuple[np.ndarray, np.ndarray]:
        """
        This method extracts the longitude and latitude coordinate arrays from the loaded dataset, handling various possible variable names for spatial coordinates and normalizing longitude values to the standard [-180, 180] degree range. It checks for common variable names for longitude (e.g., 'lonCell', 'longitude', 'lon') and latitude (e.g., 'latCell', 'latitude', 'lat') in both the dataset's coordinates and data variables. If it finds valid longitude and latitude variables, it converts them to numpy arrays. If the latitude values are in radians (indicated by a maximum absolute value less than or equal to π), it converts both longitude and latitude to degrees. The method then flattens the coordinate arrays to 1D and normalizes the longitude values using the `normalize_longitude` method. Finally, it returns a tuple containing the normalized longitude array and the latitude array, which can be used for geographic plotting and analysis. If it cannot find valid spatial coordinate variables, it raises a ValueError with information about the available variables in the dataset. 

        Parameters:
            None

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the normalized longitude array and the latitude array extracted from the dataset, which can be used for geographic plotting and analysis. The longitude values are normalized to the standard [-180, 180] degree range, and both arrays are flattened to 1D. 
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
    
    def _load_grid_file(self: 'MPASBaseProcessor',
                       needed_vars: Optional[List[str]] = None) -> xr.Dataset:
        """
        This helper method loads the grid file specified by `self.grid_file` using xarray, which contains the unstructured grid metadata necessary for spatial coordinate extraction and mesh connectivity information. It opens the grid file dataset in lazily-loaded mode without decoding times, allowing for efficient access to the grid metadata without loading the entire dataset into memory. When `needed_vars` is provided, only those variables are retained from the grid file, significantly reducing memory usage for large grids. The method returns the loaded grid file dataset, which can be used for extracting spatial coordinates and other grid-related information needed for processing MPAS datasets with unstructured grids. If verbose mode is enabled, it also prints a message indicating that the grid file was loaded successfully along with a list of the variables contained in the grid file dataset. 

        Parameters:
            needed_vars (Optional[List[str]]): List of variable names to retain from the grid file. If None, all variables are loaded (default: None).

        Returns:
            xr.Dataset: The xarray.Dataset loaded from the grid file, containing the unstructured grid metadata necessary for spatial coordinate extraction and mesh connectivity information. 
        """
        open_kwargs: Dict[str, Any] = {'decode_times': False}

        if needed_vars is not None:
            try:
                with xr.open_dataset(self.grid_file, decode_times=False) as probe_ds:
                    all_grid_vars = list(probe_ds.data_vars)
                variables_to_drop = [v for v in all_grid_vars if v not in needed_vars]
                if variables_to_drop:
                    open_kwargs['drop_variables'] = variables_to_drop
            except Exception:
                pass
        
        grid_file_ds = xr.open_dataset(self.grid_file, **open_kwargs)
        
        if self.verbose:
            print(f"\nGrid file loaded successfully with variables: \n{list(grid_file_ds.variables.keys())}\n")
        
        return grid_file_ds

    def _prepare_dimension_coordinates(self: 'MPASBaseProcessor', 
                                       combined_ds: xr.Dataset, 
                                       dimensions_to_add: List[str]) -> dict:
        """
        This helper method checks the `combined_ds` for the presence of specified dimension names (e.g., `nCells`, `nVertices`) and prepares a mapping of coordinate names to (dimension name, index array) pairs for those dimensions that exist in the dataset. This mapping can be used to assign new index coordinates to the dataset for dimensions that are present, which is important for enabling proper indexing and alignment of data variables along those dimensions. The method iterates over the list of dimension names, checks if each dimension exists in the combined dataset's sizes, and if so, it creates an index array using `np.arange` based on the size of that dimension and adds it to the mapping. It also prints verbose messages indicating which index coordinates were added for which dimensions. The resulting dictionary is returned for use in extending the combined dataset with new index coordinates corresponding to existing dimensions. 

        Parameters:
            combined_ds (xr.Dataset): The combined dataset to check for existing dimensions and to prepare new index coordinates for.
            dimensions_to_add (List[str]): A list of dimension names to check for in the combined dataset and prepare index coordinates for (e.g., ['nCells', 'nVertices']). 

        Returns:
            dict: Mapping of coordinate names to (dimension name, index array) pairs for dimensions that exist in `combined_ds`. Only includes dimensions that are present in the dataset's sizes, and the index arrays are created using `np.arange` based on the size of each dimension. This mapping can be used to assign new index coordinates to the dataset for proper indexing and alignment of data variables along those dimensions. 
        """
        coords_to_add = {}
        
        for dim_name in dimensions_to_add:
            if dim_name in combined_ds.sizes:
                coords_to_add[dim_name] = (dim_name, np.arange(combined_ds.sizes[dim_name]))
                if self.verbose:
                    print(f"Added {dim_name} index coordinate for {dim_name} dimension ({combined_ds.sizes[dim_name]} values)")
        
        return coords_to_add

    def _prepare_spatial_variables(self: 'MPASBaseProcessor', 
                                   grid_file_ds: xr.Dataset, 
                                   combined_ds: xr.Dataset, 
                                   spatial_vars: List[str]) -> dict:
        """
        This helper method checks the `grid_file_ds` for the presence of specified spatial variable names (e.g., `latCell`, `lonCell`) and prepares a mapping of variable names to DataArray objects for those variables that exist in the grid file dataset and are not already present in the `combined_ds.data_vars`. This mapping can be used to copy spatial variable data from the grid file dataset into the combined dataset without overwriting any existing variables. The method iterates over the list of spatial variable names, checks if each variable exists in the grid file dataset's variables and is not already a data variable in the combined dataset, and if so, it adds it to the mapping. It also prints verbose messages indicating which spatial coordinate variables were added from the grid file. The resulting dictionary is returned for use in extending the combined dataset with new spatial variables extracted from the grid file. 

        Parameters:
            grid_file_ds (xr.Dataset): The dataset loaded from the grid file, which contains spatial variables that may be added to the combined dataset if they exist and are not already present.
            combined_ds (xr.Dataset): The combined dataset to check for existing data variables and to prepare new spatial variables for addition.
            spatial_vars (List[str]): A list of spatial variable names to check for in the grid file dataset and prepare for addition to the combined dataset (e.g., ['latCell', 'lonCell']). 

        Returns:
            dict: Mapping of variable names to DataArray objects for spatial variables that exist in `grid_file_ds` and are not already present in `combined_ds.data_vars`. This mapping can be used to copy spatial variable data from the grid file dataset into the combined dataset without overwriting any existing variables, allowing for enrichment of the combined dataset with spatial information from the grid file. Only includes variables that are present in the grid file dataset and not already defined as data variables in the combined dataset. 
        """
        data_vars_to_add = {}
        
        for var_name in spatial_vars:
            if var_name in grid_file_ds.variables and var_name not in combined_ds.data_vars:
                data_vars_to_add[var_name] = grid_file_ds[var_name]
                if self.verbose:
                    print(f"Added spatial coordinate variable: {var_name}")
        
        return data_vars_to_add

    def _apply_coordinate_updates(self: 'MPASBaseProcessor', 
                                  combined_ds: xr.Dataset, 
                                  coords_to_add: dict, 
                                  data_vars_to_add: dict) -> xr.Dataset:
        """
        This helper method applies the prepared coordinate and spatial variable updates to the `combined_ds` by assigning new coordinates from the `coords_to_add` mapping and adding new data variables from the `data_vars_to_add` mapping. It first checks if there are any coordinates to add and if so, it uses `assign_coords` to add them to the combined dataset, printing a verbose message indicating how many coordinate variables were added. Then it checks if there are any spatial variables to add and if so, it iterates over them and adds each one to the combined dataset, printing a verbose message indicating how many spatial variables were added and listing the updated dataset coordinates. If there are no additional coordinate variables found to add, it prints a message indicating that as well. Finally, it returns the updated combined dataset with the new coordinates and data variables applied. 

        Parameters:
            combined_ds (xr.Dataset): The combined dataset to which the new coordinates and spatial variables will be added.
            coords_to_add (dict): A mapping of coordinate names to (dimension name, index array) pairs for new coordinates that should be added to the combined dataset using `assign_coords`.
            data_vars_to_add (dict): A mapping of variable names to DataArray objects for new spatial variables that should be added to the combined dataset as data variables. 

        Returns:
            xr.Dataset: The updated combined dataset with the new coordinates and spatial variables applied. This dataset will include any new index coordinates added from `coords_to_add` and any new spatial variables added from `data_vars_to_add`, enriching the dataset with additional spatial information from the grid file. 
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

    def _add_spatial_coords_helper(self: 'MPASBaseProcessor', 
                                   combined_ds: xr.Dataset, 
                                   dimensions_to_add: List[str],
                                   spatial_vars: List[str],
                                   processor_type: str) -> xr.Dataset:
        """
        This helper method attempts to add spatial coordinates to the `combined_ds` by loading the grid file dataset, preparing coordinate mappings for specified dimensions and spatial variables, and applying those updates to the combined dataset. It first loads the grid file dataset to access spatial variables and coordinate information. Then it prepares coordinate mappings for dimensions that exist in the combined dataset to add index coordinates for easier indexing and plotting. Next, it processes spatial variables to add from the grid file, ensuring that only those that are not already present in the combined dataset are added to avoid overwriting existing data variables. It applies the coordinate and variable additions to the combined dataset and closes the grid file dataset to free resources. If any exceptions occur during this process (e.g., issues with loading the grid file or adding coordinates), it catches the exception and prints a warning message indicating that spatial coordinates could not be added for the specified processor type, but continues processing with the original combined dataset without raising an error. Finally, it returns the combined dataset with spatial coordinates added if successful, or the original combined dataset if an error occurred while adding coordinates. 

        Parameters:
            combined_ds (xr.Dataset): The combined dataset to which spatial coordinates will be added if successful.
            dimensions_to_add (List[str]): A list of dimension names to check for in the combined dataset and prepare index coordinates for (e.g., ['nCells', 'nVertices']).
            spatial_vars (List[str]): A list of spatial variable names to check for in the grid file dataset and prepare for addition to the combined dataset (e.g., ['latCell', 'lonCell']).
            processor_type (str): A string identifier of the processor type (e.g., "MPASBaseProcessor") to include in the warning message if spatial coordinates cannot be added, providing context for the user about which processor encountered the issue. 

        Returns:
            xr.Dataset: The combined dataset with spatial coordinates added if successful, or the original combined dataset if an error occurred while adding coordinates. This method attempts to enrich the combined dataset with spatial information from the grid file, but gracefully handles any issues that arise during this process without interrupting the overall workflow. 
        """
        try:
            # Load the grid file dataset to access spatial variables and coordinate information
            grid_file_ds = self._load_grid_file(needed_vars=spatial_vars)

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
    
    def get_time_info(self: 'MPASBaseProcessor', 
                      time_index: int, 
                      var_context: str = "") -> str:
        """
        This method retrieves and formats time information for a specified time index from the loaded dataset, providing a human-readable string that can be used for diagnostic output or plot labeling. It first checks if a dataset has been loaded by verifying that `self.dataset` is not None, and if it is None, it raises a ValueError indicating that no dataset has been loaded. If a dataset is loaded, it imports the MPASDateTimeUtils class and uses its `get_time_info` method to extract and format the time information for the specified time index, including any provided variable context in the formatted string for more informative labeling. The resulting string contains the time information corresponding to the specified time index, which can be used to enhance the clarity of diagnostic outputs and plots by providing temporal context. 

        Parameters:
            time_index (int): The zero-based index along the time dimension for which to retrieve and format time information. This index will be validated against the time dimension of the loaded dataset to ensure it is within bounds.
            var_context (str): An optional string providing context about the variable or diagnostic being processed, which will be included in the formatted time information string to enhance clarity in diagnostic outputs and plot labeling.

        Returns:
            str: A formatted string containing the time information corresponding to the specified time index, which can be used for diagnostic output or plot labeling to provide temporal context. The string may include the variable context if provided, making it more informative for users. 
        """
        # Raise a ValueError if no dataset has been loaded before attempting to retrieve time information
        if self.dataset is None:
            raise ValueError(DATASET_NOT_LOADED_MSG)
        
        # Import the MPASDateTimeUtils class necessary for retrieving and formatting time information from the dataset's time coordinate.
        from .utils_datetime import MPASDateTimeUtils

        # Use get_time_info method from MPASDateTimeUtils to extract and format time information for the specified time index and variable context.
        return MPASDateTimeUtils.get_time_info(self.dataset, time_index, var_context, self.verbose)
    
    def parse_file_datetimes(self: 'MPASBaseProcessor', 
                             diag_files: List[str]) -> List:
        """
        This method parses datetime information from a list of MPAS diagnostic or output file paths by extracting embedded datetime information from the filenames. It first imports the MPASDateTimeUtils class and then uses its `parse_file_datetimes` method to process the list of file paths, extracting datetime objects based on common MPAS filename conventions. The resulting list of datetime objects is returned in the same order as the input file list, making it suitable for temporal sorting and coordinate assignment in subsequent processing steps. This method provides a standardized way to extract temporal information from filenames, which is essential for organizing and analyzing time series data from MPAS diagnostics and model outputs. 

        Parameters:
            diag_files (List[str]): A list of absolute file paths to MPAS diagnostic or output files, from which datetime information will be extracted based on filename conventions.

        Returns:
            List: A list of datetime objects extracted from the input file paths, in the same order as the input list, suitable for temporal sorting and coordinate assignment.
        """
        # Import the MPASDateTimeUtils class necessary for datetime parsing from filenames 
        from .utils_datetime import MPASDateTimeUtils

        # Parse datetimes from the filenames using parse_file_datetimes utility 
        return MPASDateTimeUtils.parse_file_datetimes(diag_files, self.verbose)
    
    def validate_time_parameters(self: 'MPASBaseProcessor', 
                                 time_index: int) -> Tuple[str, int, int]:
        """
        This method validates the provided time index against the time dimension of the loaded dataset to ensure it is within bounds and correctly handles negative indices for counting from the end of the time series. It first checks if a dataset has been loaded by verifying that `self.dataset` is not None, and if it is None, it raises a ValueError indicating that no dataset has been loaded. If a dataset is loaded, it imports the MPASDateTimeUtils class and uses its `validate_time_parameters` method to check that the provided time index is valid for the dataset's time dimension, converting any negative indices to their positive equivalents. The method returns a tuple containing the name of the time dimension, the validated time index (with negative indices converted), and the size of the time dimension for reference in error messages or further processing. This validation is crucial for ensuring that temporal indexing operations are performed correctly and that users receive informative feedback if they provide out-of-bounds indices. 

        Parameters:
            time_index (int): The zero-based index along the time dimension to validate, which may be a negative index for counting from the end of the time series. This index will be checked against the size of the time dimension in the loaded dataset to ensure it is within valid bounds.

        Returns:
            Tuple[str, int, int]: A tuple containing (time_dim_name, validated_time_index, time_dim_size) where time_dim_name is the name of the time dimension in the dataset, validated_time_index is the non-negative index corresponding to the provided time_index (with negative indices converted), and time_dim_size is the size of the time dimension for reference.
        """
        # Raise a ValueError if no dataset has been loaded before attempting to validate time dimensions
        if self.dataset is None:
            raise ValueError(DATASET_NOT_LOADED_MSG)
        
        # Import the MPASDateTimeUtils class necessary for time validation 
        from .utils_datetime import MPASDateTimeUtils

        # Validate time with validate_time_parameters utility 
        return MPASDateTimeUtils.validate_time_parameters(self.dataset, time_index, self.verbose)
    
    def filter_by_spatial_extent(self: 'MPASBaseProcessor',
                                 data: Any, 
                                 lon_min: float, 
                                 lon_max: float, 
                                 lat_min: float, 
                                 lat_max: float) -> Tuple[Any, np.ndarray]:
        """
        This method applies a spatial filter to the input data based on specified longitude and latitude bounds, using the spatial coordinates from the loaded dataset to determine which points fall within the defined geographic extent. It first checks if a dataset has been loaded by verifying that `self.dataset` is not None, and if it is None, it raises a ValueError indicating that no dataset has been loaded. If a dataset is loaded, it imports the MPASGeographicUtils class and uses its `filter_by_spatial_extent` method to apply a spatial mask to the input data, returning both the filtered data (with points outside the extent masked) and a boolean array indicating which points were selected based on the geographic bounds. This method is essential for enabling spatially focused analysis and plotting by allowing users to isolate data within specific geographic regions of interest. 

        Parameters:
            data (Any): The input data to be filtered, which can be of any type that is compatible with the spatial filtering operation defined in MPASGeographicUtils (e.g., xarray.DataArray, numpy array, etc.).
            lon_min (float): The minimum longitude bound of the spatial extent in degrees, defining the western edge of the geographic region of interest.
            lon_max (float): The maximum longitude bound of the spatial extent in degrees, defining the eastern edge of the geographic region of interest.
            lat_min (float): The minimum latitude bound of the spatial extent in degrees, defining the southern edge of the geographic region of interest.
            lat_max (float): The maximum latitude bound of the spatial extent in degrees, defining the northern edge of the geographic region of interest.

        Returns:
            Tuple[Any, np.ndarray]: A tuple containing the filtered data (with points outside the specified geographic extent masked) and a boolean array indicating which points were selected based on the longitude and latitude bounds.
        """
        # Raise a ValueError if no dataset has been loaded before attempting to filter by spatial extent
        if self.dataset is None:
            raise ValueError(DATASET_NOT_LOADED_MSG)
        
        # Import the MPASGeographicUtils class necessary for performing spatial filtering based on geographic coordinates. 
        from .utils_geog import MPASGeographicUtils

        # Use filter_by_spatial_extent method from MPASGeographicUtils to apply a spatial mask to the input data
        return MPASGeographicUtils.filter_by_spatial_extent(
            data, self.dataset, lon_min, lon_max, lat_min, lat_max)
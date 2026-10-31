#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

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
import re
import numpy as np
import xarray as xr
from typing import Dict, List, Tuple, Any, Optional, cast

from .base import MPASBaseProcessor
from .utils_datetime import MPASDateTimeUtils
from .utils_geog import MPASGeographicUtils
from .constants import DIAG_GLOB, MPASOUT_GLOB
from .utils_logger import get_logger
from .utils_path import safe_label

logger = get_logger(__name__)


class MPAS2DProcessor(MPASBaseProcessor):
    """Processor class for handling 2D surface and diagnostic variables from MPAS model output."""

    def __init__(
        self: "MPAS2DProcessor",
        grid_file: str,
        verbose: bool = True,
        file_pattern: Optional[str] = None,
    ) -> None:
        """
        This constructor initializes the MPAS2DProcessor by calling the base class constructor with the provided grid file, verbosity settings, and optional custom file pattern. It sets up the necessary attributes for processing 2D diagnostic data, ensuring that the processor is ready to load datasets, extract spatial coordinates, and prepare data for visualization. When a custom file pattern is supplied, file discovery uses that glob pattern instead of the default 'diag*.nc' convention, which allows 2D fields to be read from custom output streams such as 'wiso.*.nc'. By inheriting from MPASBaseProcessor, it leverages common functionality while allowing for specific implementations tailored to 2D surface data processing workflows.

        Parameters:
            grid_file (str): Path to the MPAS grid file containing spatial coordinate information.
            verbose (bool): If True, enables detailed logging of processing steps (default: True).
            file_pattern (Optional[str]): Custom glob pattern used to discover input files (e.g. 'wiso.*.nc'). When None, the default 'diag*.nc' discovery (with an 'mpasout*.nc' fallback) is used (default: None).

        Returns:
            None
        """
        super().__init__(grid_file, verbose, file_pattern)

    def add_spatial_coordinates(
        self: "MPAS2DProcessor", combined_ds: xr.Dataset
    ) -> xr.Dataset:
        """
        This method enriches the combined dataset of 2D diagnostic variables by adding coordinate variables for spatial dimensions (nCells, nVertices, nIsoLevelsT, nIsoLevelsZ) and their corresponding geographic coordinates (latitude and longitude). It identifies the relevant dimensions and spatial variables in the dataset, then adds the appropriate coordinate variables based on the grid information. This process ensures that the resulting dataset is fully equipped with the necessary spatial information for geospatial analysis and plotting. By centralizing this functionality in a dedicated method, it promotes code reuse and maintainability across different processing workflows that require spatial coordinate enrichment for 2D diagnostic data.

        Parameters:
            combined_ds (xr.Dataset): The combined dataset of 2D diagnostic variables to enrich with spatial coordinates.

        Returns:
            xr.Dataset: The enriched dataset with added spatial coordinate variables for nCells, nVertices, nIsoLevelsT, and nIsoLevelsZ, along with their corresponding latitude and longitude coordinates.
        """
        dimensions_to_add = ["nCells", "nVertices", "nIsoLevelsT", "nIsoLevelsZ"]

        spatial_vars = [
            "latCell",
            "lonCell",
            "latVertex",
            "lonVertex",
            "verticesOnCell",
            "nEdgesOnCell",
        ]

        return self._add_spatial_coords_helper(
            combined_ds, dimensions_to_add, spatial_vars, "2D"
        )

    def load_2d_data(
        self: "MPAS2DProcessor",
        data_dir: str,
        use_pure_xarray: bool = False,
        reference_file: str = "",
        variables: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
    ) -> "MPAS2DProcessor":
        """
        This method loads 2D diagnostic data from the specified directory, utilizing either the pure xarray backend or the UXarray backend based on the provided flag. It first identifies the relevant diagnostic files in the directory, then loads them into a combined dataset while adding necessary spatial coordinates. The method ensures that the loaded dataset is properly enriched with spatial information for subsequent analysis and visualization. By returning self, it allows for method chaining in processing workflows that require multiple operations on the loaded dataset.

        Parameters:
            data_dir (str): Directory path to search for MPAS diagnostic files.
            use_pure_xarray (bool): If True, uses the pure xarray backend for loading data; otherwise, uses UXarray (default: False).
            reference_file (str): Optional path to a reference file for loading data, if required by the backend (default: "").
            variables (Optional[List[str]]): List of variable names to retain. If provided, only these variables are loaded from data files to reduce memory usage (default: None).
            chunk_size (Optional[int]): Number of cells per dask chunk along the nCells dimension (Time stays at one record per chunk). None keeps the default behaviour of one chunk per Time record spanning all cells; a positive integer enables finer spatial chunking to bound peak memory on very large meshes (default: None).

        Returns:
            MPAS2DProcessor: The instance of the processor with the loaded dataset ready for analysis and visualization.
        """
        self.data_dir = data_dir
        if chunk_size is not None and (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, (int, np.integer))
            or chunk_size <= 0
        ):
            raise ValueError(
                f"chunk_size must be a positive integer, got {chunk_size!r}"
            )

        chunks_2d: Dict[str, int] = {"Time": 1}

        if chunk_size is not None:
            chunks_2d["nCells"] = int(chunk_size)

        dataset, _ = self._load_data(
            data_dir,
            use_pure_xarray,
            reference_file,
            chunks=chunks_2d,
            data_type_label="2D",
            variables=variables,
        )

        if hasattr(dataset, "data_vars"):
            dataset = self.add_spatial_coordinates(dataset)
            self.dataset = dataset
        elif hasattr(dataset, "ds"):
            dataset.ds = self.add_spatial_coordinates(dataset.ds)
            self.dataset = dataset

        return self

    def _find_diag_files_recursive(
        self: "MPAS2DProcessor", data_dir: str
    ) -> Optional[List[str]]:
        """
        This helper method performs a recursive search for diagnostic files in the specified directory using a predefined glob pattern. It looks for files matching the DIAG_GLOB pattern in all subdirectories of the provided data directory. If it finds at least one diagnostic file, it returns a sorted list of their paths; otherwise, it returns None. This method is used as a fallback when direct searches for diagnostic files in the main directory and a "diag" subdirectory do not yield results, ensuring that all possible locations are checked for relevant diagnostic data.

        Parameters:
            data_dir (str): Directory path to search for MPAS diagnostic files recursively.

        Returns:
            Optional[List[str]]: A sorted list of file paths to the found diagnostic files if at least one is found; otherwise, None.
        """
        files = sorted(
            glob.glob(
                os.path.join(glob.escape(data_dir), "**", DIAG_GLOB), recursive=True
            ),
            key=MPASDateTimeUtils.file_sort_key,
        )

        if files:
            if self.verbose:
                logger.info("Found %d diagnostic files (recursive search):", len(files))
                for i, filepath in enumerate(files[:5]):
                    logger.info("  %d: %s", i + 1, os.path.basename(filepath))
            return files

        return None

    def _find_mpasout_files_fallback(
        self: "MPAS2DProcessor", data_dir: str
    ) -> List[str]:
        """
        This helper method performs a fallback search for MPAS output files (mpasout*.nc) in the specified directory using a predefined glob pattern. It first attempts to find files directly in the provided directory, then looks in a "diag" subdirectory if no files are found. If still unsuccessful, it performs a recursive search for MPAS output files. If no MPAS output files are found after these attempts, it raises a FileNotFoundError with an informative message. The method ensures that at least one MPAS output file is found and provides detailed logging of the search results when verbose mode is enabled.

        Parameters:
            data_dir (str): Directory path to search for MPAS output files.

        Returns:
            List[str]: A sorted list of file paths to the found MPAS output files suitable for 2D analysis.
        """
        try:
            return self._find_files_by_pattern(
                data_dir, MPASOUT_GLOB, "MPAS output files (mpasout)"
            )
        except FileNotFoundError:
            pass

        files = sorted(
            glob.glob(
                os.path.join(glob.escape(data_dir), "**", MPASOUT_GLOB), recursive=True
            ),
            key=MPASDateTimeUtils.file_sort_key,
        )

        if not files:
            raise FileNotFoundError(
                f"No diagnostic files (diag*.nc) or MPAS output files (mpasout*.nc) found under: {data_dir}\n"
                f"For precipitation analysis, ensure files contain rainc and rainnc variables."
            )

        if self.verbose:
            logger.info("Found %d MPAS output files (recursive search):", len(files))
            for i, filepath in enumerate(files[:5]):
                logger.info("  %d: %s", i + 1, os.path.basename(filepath))

        return files

    def find_diagnostic_files(self: "MPAS2DProcessor", data_dir: str) -> List[str]:
        """
        This method searches for diagnostic files in the specified directory using a predefined glob pattern. It first attempts to find files directly in the provided directory, then looks in a "diag" subdirectory if no files are found. If still unsuccessful, it performs a recursive search for diagnostic files. If no diagnostic files are found after these attempts, it then searches for MPAS output files (mpasout*.nc) using a similar approach. The method ensures that at least one file is found and provides informative error messages if no suitable files are located. By returning a sorted list of file paths, it facilitates subsequent loading and processing of the diagnostic data.

        Parameters:
            data_dir (str): Directory path to search for MPAS diagnostic files.

        Returns:
            List[str]: A sorted list of file paths to the found diagnostic files or MPAS output files suitable for 2D analysis.
        """
        try:
            return self._find_files_by_pattern(data_dir, DIAG_GLOB, "diagnostic files")
        except FileNotFoundError:
            pass

        diag_subdir = os.path.join(data_dir, "diag")

        try:
            return self._find_files_by_pattern(
                diag_subdir, DIAG_GLOB, "diagnostic files"
            )
        except FileNotFoundError:
            pass

        files = self._find_diag_files_recursive(data_dir)

        if files:
            return files

        if self.verbose:
            logger.info("No diagnostic files found, searching for mpasout files")

        return self._find_mpasout_files_fallback(data_dir)

    _COORD_NAMES_2D: dict = {
        "nVertices": (
            ["lonVertex", "lon_vertex", "longitude_vertex"],
            ["latVertex", "lat_vertex", "latitude_vertex"],
        ),
        "nCells": (["lonCell", "longitude", "lon"], ["latCell", "latitude", "lat"]),
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
        return "nVertices" if "nVertices" in sizes else "nCells"

    def _coord_attrs(self: "MPAS2DProcessor", names: List[str]) -> Dict[str, Any]:
        """
        This helper method returns the attributes of the first coordinate variable found among the candidate names so that a 'units' attribute can steer the radians-versus-degrees decision. It returns an empty dictionary when the dataset is not loaded or none of the names exist.

        Parameters:
            names (List[str]): A list of possible coordinate variable names to search for.

        Returns:
            Dict[str, Any]: Attributes of the first found coordinate variable, or an empty dict.
        """
        if self.dataset is None:
            return {}
        for name in names:
            if name in self.dataset.coords or name in self.dataset.data_vars:
                return dict(self.dataset[name].attrs)
        return {}

    def _lookup_2d_coord(
        self: "MPAS2DProcessor", names: List[str]
    ) -> Optional[np.ndarray]:
        """
        This helper method looks up coordinate variables in the dataset based on a list of possible names. It iterates through the provided list of names and checks if any of them exist as coordinates or data variables in the dataset. If a matching variable is found, it returns its values as a NumPy array. If none of the names are found, it returns None. This method is crucial for flexibly handling different naming conventions for spatial coordinates in MPAS datasets, allowing the processor to adapt to various dataset structures without requiring hardcoded variable names.

        Parameters:
            names (List[str]): A list of possible coordinate variable names to search for.

        Returns:
            Optional[np.ndarray]: The values of the first found coordinate variable, or None if none are found.
        """
        if self.dataset is None:
            return None
        for name in names:
            if name in self.dataset.coords or name in self.dataset.data_vars:
                values: np.ndarray = np.asarray(self.dataset[name].values)
                return values
        return None

    def extract_2d_coordinates_for_variable(
        self: "MPAS2DProcessor",
        var_name: str,
        data_array: Optional[xr.DataArray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        if data_array is not None:
            sizes = data_array.sizes
        elif var_name in self.dataset:
            sizes = self.dataset[var_name].sizes
        else:
            sizes = {}

        spatial_dim = self._detect_2d_spatial_dim(sizes)
        lon_names, lat_names = self._COORD_NAMES_2D[spatial_dim]

        lon_coords = self._lookup_2d_coord(lon_names)
        lat_coords = self._lookup_2d_coord(lat_names)

        if lon_coords is None or lat_coords is None:
            available_vars = list(self.dataset.coords.keys()) + list(
                self.dataset.data_vars.keys()
            )
            raise ValueError(
                f"Could not find {spatial_dim} coordinates. Available variables: "
                f"{safe_label(available_vars)}"
            )

        # Both coordinate arrays are guaranteed present past the guard above.
        lon_arr: np.ndarray = np.asarray(lon_coords)
        lat_arr: np.ndarray = np.asarray(lat_coords)

        lon_arr, lat_arr = MPASGeographicUtils.coordinates_to_degrees(
            lon_arr,
            lat_arr,
            self._coord_attrs(lon_names),
            self._coord_attrs(lat_names),
        )

        lon_arr = ((lon_arr.ravel() + 180) % 360) - 180
        lat_arr = lat_arr.ravel()

        if self.verbose:
            logger.debug(
                "Extracted %s coordinates for %s: %s points",
                spatial_dim,
                var_name,
                f"{len(lon_arr):,}",
            )

        return lon_arr, lat_arr

    def _log_2d_variable_range(
        self: "MPAS2DProcessor", var_name: str, var_data: xr.DataArray
    ) -> None:
        """
        This helper method logs the range of finite values for a specified 2D variable. It checks if verbose mode is enabled and if the variable data has values. It then flattens the variable data, filters out non-finite values, and calculates the minimum and maximum of the finite values. If finite values are found, it prints the range along with the variable name and its units if available. If no finite values are found, it logs a warning message. This method provides valuable insights into the data being processed, helping users understand the range of values for diagnostic variables and identify potential issues with the data.

        Parameters:
            var_name (str): Name of the variable for which to log the range.
            var_data (xr.DataArray): The data array of the variable whose range is to be logged.

        Returns:
            None

        """
        if not self.verbose or not hasattr(var_data, "values"):
            return

        finite_values = var_data.values.flatten()
        finite_values = finite_values[np.isfinite(finite_values)]

        if len(finite_values) > 0:
            logger.debug(
                "Variable %s range: %.3f to %.3f",
                var_name,
                float(finite_values.min()),
                float(finite_values.max()),
            )
            if hasattr(var_data, "attrs") and "units" in var_data.attrs:
                logger.debug("Units: %s", var_data.attrs["units"])
        else:
            logger.warning("No finite values found for %s", var_name)

    _ISOBARIC_RE = re.compile(r"^(\w+?)_(\d+)hPa$")

    @staticmethod
    def _cells_to_vertices(
        cell_values: np.ndarray, cells_on_vertex: np.ndarray
    ) -> np.ndarray:
        """
        This helper averages a cell-centred field onto the mesh vertices using the (1-based) cellsOnVertex connectivity, ignoring missing cells (index 0 at the edge of a regional mesh) and non-finite values.

        Parameters:
            cell_values (np.ndarray): 1D array of values on nCells.
            cells_on_vertex (np.ndarray): Integer array of shape (nVertices, vertexDegree) with 1-based cell indices.

        Returns:
            np.ndarray: 1D array of vertex values (NaN where no valid surrounding cell exists).
        """
        idx = np.clip(cells_on_vertex - 1, 0, cell_values.size - 1)
        gathered = cell_values[idx].astype(float)
        valid = (
            (cells_on_vertex > 0)
            & (cells_on_vertex <= cell_values.size)
            & np.isfinite(gathered)
        )
        n_valid = valid.sum(axis=1)
        total = np.where(valid, gathered, 0.0).sum(axis=1)

        return np.asarray(np.where(n_valid > 0, total / np.maximum(n_valid, 1), np.nan))

    def _terrain_for_masking(
        self: "MPAS2DProcessor",
        var_name: str,
        on_vertices: bool,
    ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], int]]:
        """
        This helper loads the terrain height used to mask below-ground cells from the grid file. For vertex fields the terrain height of a vertex is the mean 'ter' of the cells around it (cellsOnVertex). None is returned when the grid file cannot be read, has no 'ter', or (for vertex fields) has no cellsOnVertex, in which case no masking applies.

        Parameters:
            var_name (str): Name of the extracted variable, used in the diagnostic message.
            on_vertices (bool): True when the variable is defined on mesh vertices.

        Returns:
            Optional[Tuple[np.ndarray, Optional[np.ndarray], int]]: The terrain height on the variable's mesh locations, the cellsOnVertex array (None for cell fields), and the number of mesh cells, or None when masking does not apply.
        """
        try:
            grid_ds = self._load_grid_file(
                needed_vars=["ter", "cellsOnVertex"] if on_vertices else ["ter"]
            )
        except Exception:
            return None

        if "ter" not in grid_ds:
            return None

        ter = np.asarray(grid_ds["ter"].values)
        cells_on_vertex: Optional[np.ndarray] = None
        n_cells = 0

        if on_vertices:
            # Vertex terrain height is the mean of the terrain of the surrounding cells
            if "cellsOnVertex" not in grid_ds:
                logger.debug(
                    "Below-ground masking skipped for %s: no cellsOnVertex in the grid file",
                    var_name,
                )
                return None

            cells_on_vertex = np.asarray(grid_ds["cellsOnVertex"].values).astype(int)
            n_cells = ter.size
            ter = self._cells_to_vertices(ter, cells_on_vertex)

        return ter, cells_on_vertex, n_cells

    def _mask_below_ground(
        self: "MPAS2DProcessor",
        var_name: str,
        var_data: xr.DataArray,
        time_dim: str,
        time_index: int,
    ) -> xr.DataArray:
        """
        This helper masks the cells of an isobaric diagnostic field that lie below the ground. MPAS diag files interpolate fields such as 'temperature_850hPa' to fixed pressure levels and extrapolate them where the level is underground (for example over the Himalaya or Antarctica), producing physically meaningless values. When the variable is named '<name>_<N>hPa', the companion geopotential height 'height_<N>hPa' is present in the dataset and the grid file provides terrain height 'ter', every cell where height_<N>hPa is lower than ter is set to NaN; for vertex fields (nVertices, e.g. 'vorticity_<N>hPa') the terrain height of a vertex is the mean 'ter' of the cells around it (cellsOnVertex). If any of those ingredients is missing, or the shapes do not agree, the data are returned unchanged. The returned array is a new object whose attrs record the masking ('below_ground_masked' and the number of masked cells) so the caller's dataset is never modified.

        Parameters:
            var_name (str): Name of the extracted variable.
            var_data (xr.DataArray): Computed one-time-slice data array of the variable.
            time_dim (str): Name of the time dimension of the dataset.
            time_index (int): Validated time index that was extracted.

        Returns:
            xr.DataArray: The masked array, or the input unchanged when masking does not apply.
        """
        match = self._ISOBARIC_RE.match(var_name)

        if match is None or self.dataset is None:
            return var_data

        height_name = f"height_{match.group(2)}hPa"

        if height_name not in self.dataset.data_vars or var_data.ndim != 1:
            return var_data

        on_vertices = "nVertices" in var_data.dims

        terrain = self._terrain_for_masking(var_name, on_vertices)

        if terrain is None:
            return var_data

        ter, cells_on_vertex, n_cells = terrain

        height = self.dataset[height_name].isel({time_dim: time_index})

        if hasattr(height, "compute"):
            height = cast(Any, height).compute()

        height_values = np.asarray(height.values)

        if on_vertices and height_values.shape == (n_cells,):
            height_values = self._cells_to_vertices(
                height_values, cast(np.ndarray, cells_on_vertex)
            )

        if height_values.shape != var_data.shape or ter.shape != var_data.shape:
            return var_data

        with np.errstate(invalid="ignore"):
            below = height_values < ter

        masked = var_data.where(~below)
        masked.attrs = dict(var_data.attrs)
        masked.attrs["below_ground_masked"] = (
            f"cells with {height_name} < terrain height 'ter' set to NaN"
        )
        masked.attrs["below_ground_masked_cells"] = int(below.sum())

        return masked

    def get_2d_variable_data(
        self: "MPAS2DProcessor",
        var_name: str,
        time_index: int = 0,
        mask_below_ground: bool = True,
    ) -> xr.DataArray:
        """
        This method retrieves the data array for a specified 2D variable at a given time index from the loaded dataset. It first checks if the dataset is loaded and if the variable exists in the dataset. It then validates the time index against the dataset's time dimension, ensuring it is within bounds. Depending on the data type (pure xarray or UXarray), it extracts the variable data at the specified time index. If the extracted data is a lazy array, it computes it to get the actual values. If verbose mode is enabled, it also prints out the range of finite values for the variable along with its units if available. Finally, it returns the data array for further analysis or visualization.

        Parameters:
            var_name (str): Name of the 2D variable to retrieve.
            time_index (int): Time index to extract data from (default: 0).
            mask_below_ground (bool): If True, isobaric diag fields named '<name>_<N>hPa' (including 'height_<N>hPa' itself) have the cells where height_<N>hPa is lower than the grid's terrain height 'ter' set to NaN, because MPAS extrapolates those fields underground; the attrs of the result record that masking was applied. Set False for the raw extrapolated values (default: True).

        Returns:
            xr.DataArray: The data array for the specified variable at the given time index, ready for analysis or visualization.
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_2d_data() first.")

        if var_name not in self.dataset.data_vars:
            available_vars = list(self.dataset.data_vars.keys())
            raise ValueError(
                f"Variable '{safe_label(var_name)}' not found. Available variables: "
                f"{safe_label(available_vars)}"
            )

        if "nEdges" in self.dataset[var_name].dims:
            raise ValueError(
                f"Variable '{safe_label(var_name)}' is defined on mesh edges (nEdges), "
                "which is not supported for 2D plotting; only nCells and nVertices "
                "variables can be loaded."
            )

        time_dim, validated_time_index, _ = MPASDateTimeUtils.validate_time_parameters(
            self.dataset, time_index, self.verbose
        )

        if self.verbose:
            logger.debug(
                "Extracting %s data at time index %d", var_name, validated_time_index
            )

        if self.data_type == "uxarray":
            var_data = self.dataset[var_name][validated_time_index]
        else:
            var_data = self.dataset[var_name].isel({time_dim: validated_time_index})

        if hasattr(var_data, "compute"):
            var_data = cast(Any, var_data).compute()

        if mask_below_ground:
            var_data = self._mask_below_ground(
                var_name, var_data, time_dim, validated_time_index
            )

        self._log_2d_variable_range(var_name, var_data)

        return var_data

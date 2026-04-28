#!/usr/bin/env python3

"""
MPASdiag Core Processing Module: Data Remapping.

This module provides the MPASRemapper class, which implements functionality to remap data from MPAS unstructured grids to regular latitude-longitude grids using ESMPy. It supports various remapping methods including conservative, nearest_s2d, and nearest_d2s interpolation. The class includes methods for preparing source and target grid specifications, building the regridder with caching support for weights, and performing the remapping of data arrays and entire datasets. The module also handles coordinate conversions, NaN handling during remapping, and provides utilities for converting unstructured data to structured grids when necessary. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: March 2026
Version: 1.0.0
"""
import warnings
import numpy as np
import xarray as xr
from pathlib import Path
from scipy.sparse import coo_matrix
from typing import Any, Optional, Union, List, Tuple

try:
    import esmpy
    ESMPY_AVAILABLE = True
except ImportError:
    ESMPY_AVAILABLE = False
    esmpy = None  # type: ignore[assignment]
    warnings.warn(
        "ESMPy is not installed. Install with: conda install -c conda-forge esmpy",
        ImportWarning,
    )


def _convert_coordinates_to_degrees(lon: Union[np.ndarray, xr.DataArray], 
                                    lat: Union[np.ndarray, xr.DataArray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    This helper function checks the input longitude and latitude coordinates to determine if they are in degrees or radians based on their maximum absolute values. If the maximum absolute value of the coordinates is less than or equal to π, it assumes the coordinates are in radians and converts them to degrees using numpy's degrees function. If the coordinates are already in degrees (i.e., max absolute value greater than π), it returns them unchanged. The function also handles both xarray DataArrays and numpy arrays as input, ensuring that the output is always a numpy array in degrees suitable for use in xESMF regridding operations. This automatic detection and conversion simplifies the user experience by allowing flexibility in input coordinate formats while ensuring compatibility with xESMF's requirements for degree-based coordinates. 
    
    Parameters:
        lon (Union[np.ndarray, xr.DataArray]): Longitude coordinates in either degrees or radians with automatic detection.
        lat (Union[np.ndarray, xr.DataArray]): Latitude coordinates in either degrees or radians with automatic detection.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Converted longitude and latitude coordinates in degrees as numpy arrays. 
    """
    if isinstance(lon, xr.DataArray):
        lon = lon.values

    if isinstance(lat, xr.DataArray):
        lat = lat.values
    
    lat_is_radians = np.max(np.abs(lat)) <= np.pi
    lon_deg = np.degrees(lon) if lat_is_radians else lon
    lat_deg = np.degrees(lat) if lat_is_radians else lat
    
    return lon_deg, lat_deg


def _compute_grid_bounds(coords: np.ndarray, 
                         resolution: float) -> np.ndarray:
    """
    This helper function computes the grid bounds for a given 1D array of coordinate centers and a specified grid spacing (resolution). It calculates the bounds by taking the midpoints between adjacent coordinate centers and extending the first and last bounds by half the resolution to ensure that the grid cells are properly defined around the center points. The resulting bounds array has a length of one more than the input coordinates, which is required for conservative remapping methods in xESMF that rely on cell corner coordinates. This function is essential for preparing the grid specification when using conservative interpolation methods, ensuring that the spatial relationships between grid cells are accurately represented in the remapping process. 
    
    Parameters:
        coords (np.ndarray): 1D array of coordinate centers (e.g., longitude or latitude values).
        resolution (float): Grid spacing in degrees, used to calculate the extent of the bounds around the center points. 
    
    Returns:
        np.ndarray: 1D array of grid bounds with length equal to len(coords) + 1, representing the edges of the grid cells. 
    """
    bounds = np.zeros(len(coords) + 1)
    bounds[0] = coords[0] - resolution / 2
    
    for i in range(1, len(coords)):
        bounds[i] = (coords[i-1] + coords[i]) / 2
    
    bounds[-1] = coords[-1] + resolution / 2
    
    return bounds


class MPASRemapper:
    """ MPASRemapper provides functionality to remap MPAS unstructured grid data to regular lat-lon grids using ESMPy. """

    _METHOD_MAP: dict = {
        'bilinear':            'BILINEAR',
        'conservative':        'CONSERVE',
        'conservative_normed': 'CONSERVE',
        'patch':               'PATCH',
        'nearest_s2d':         'NEAREST_STOD',
        'nearest_d2s':         'NEAREST_DTOS',
    }

    def __init__(self: 'MPASRemapper',
                 method: str = 'nearest_s2d',
                 weights_dir: Optional[Union[str, Path]] = None,
                 reuse_weights: bool = True,
                 periodic: bool = False,
                 extrap_method: Optional[str] = None,
                 extrap_dist_exponent: Optional[float] = None,
                 extrap_num_src_pnts: Optional[int] = None,
                 skipna: bool = False) -> None:
        """
        This initialization method sets up the MPASRemapper instance with the specified remapping method and configuration options. It validates the chosen method against supported options, configures the weights caching mechanism if a directory is provided, and initializes internal variables for storing the source and target grid specifications, as well as the computed weights. The method also handles settings related to extrapolation for unmapped points and NaN handling during remapping. By setting these parameters during initialization, the MPASRemapper instance is prepared to build the regridder and perform remapping operations efficiently when the corresponding methods are called later in the workflow.  

        Parameters:
            method (str): Remapping method to use. Supported options: 'conservative', 'conservative_normed', 'nearest_s2d', 'nearest_d2s' (default: 'nearest_s2d').
            weights_dir (Optional[Union[str, Path]]): Directory to save/load pre-computed weights for caching (default: None).
            reuse_weights (bool): Whether to reuse weights from file if available (default: True).
            periodic (bool): Whether to treat longitude as periodic when building weights (default: False).
            extrap_method (Optional[str]): Extrapolation method for unmapped points: 'nearest' or 'inverse_distance' (default: None).
            extrap_dist_exponent (Optional[float]): Exponent for distance weighting in inverse distance extrapolation (default: None).
            extrap_num_src_pnts (Optional[int]): Number of nearest source points to use for inverse distance extrapolation (default: None).
            skipna (bool): Whether to re-normalize weights for valid points when input contains NaNs (default: False).

        Returns:
            None
        """
        if not ESMPY_AVAILABLE:
            raise ImportError(
                "ESMPy is required for remapping. Install with:\n"
                "conda install -c conda-forge esmpy"
            )

        valid_methods = list(self._METHOD_MAP.keys())

        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of: {valid_methods}"
            )

        self.method = method
        self.reuse_weights = reuse_weights
        self.periodic = periodic
        self.extrap_method = extrap_method
        self.extrap_dist_exponent = extrap_dist_exponent
        self.extrap_num_src_pnts = extrap_num_src_pnts
        self.skipna = skipna

        if weights_dir is not None:
            self.weights_dir = Path(weights_dir)
            self.weights_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.weights_dir = None

        self.source_grid: Optional[xr.Dataset] = None
        self.target_grid: Optional[xr.Dataset] = None
        self._weights: Optional[Any] = None 
        self._n_src: int = 0
        self._tgt_shape: Tuple[int, int] = (0, 0) 
        self._cell_of_element: Optional[np.ndarray] = None

        print(f"MPASRemapper initialized with method: {method}")

    def prepare_source_grid(self: 'MPASRemapper',
                            lon: Union[np.ndarray, xr.DataArray],
                            lat: Union[np.ndarray, xr.DataArray],
                            lon_bounds: Optional[Union[np.ndarray, xr.DataArray]] = None,
                            lat_bounds: Optional[Union[np.ndarray, xr.DataArray]] = None) -> xr.Dataset:
        """
        This method prepares the source grid specification for remapping by converting the input longitude and latitude coordinates to degrees if they are in radians, and normalizing longitudes to the [0, 360] range. It constructs an xarray Dataset containing the 'lon' and 'lat' coordinates, which can be used as the source grid for building the regridder with ESMPy. If cell boundary coordinates (lon_bounds and lat_bounds) are provided, it also includes them in the dataset for use in conservative remapping methods. The prepared source grid is stored internally for later use when building the regridder. This method ensures that the source grid is properly formatted and compatible with ESMPy's requirements for remapping operations. 

        Parameters:
            lon (Union[np.ndarray, xr.DataArray]): Longitude coordinates of the source grid in degrees or radians.
            lat (Union[np.ndarray, xr.DataArray]): Latitude coordinates of the source grid in degrees or radians.
            lon_bounds (Optional[Union[np.ndarray, xr.DataArray]]): Optional longitude bounds for conservative remapping, shape (nCells, nVertices).
            lat_bounds (Optional[Union[np.ndarray, xr.DataArray]]): Optional latitude bounds for conservative remapping, shape (nCells, nVertices). 

        Returns:
            xr.Dataset: Source grid dataset with 'lon' and 'lat' coordinates, and optionally 'lon_b' and 'lat_b' for boundaries. 
        """
        lon, lat = _convert_coordinates_to_degrees(lon, lat)

        lon = np.where(lon < 0, lon + 360, lon)

        source_grid = xr.Dataset({
            'lon': xr.DataArray(lon, dims=['x']),
            'lat': xr.DataArray(lat, dims=['x'])
        })

        if lon_bounds is not None and lat_bounds is not None:
            lon_bounds, lat_bounds = _convert_coordinates_to_degrees(lon_bounds, lat_bounds)
            lon_bounds = np.where(lon_bounds < 0, lon_bounds + 360, lon_bounds)

            source_grid['lon_b'] = xr.DataArray(lon_bounds, dims=['x', 'nv'])
            source_grid['lat_b'] = xr.DataArray(lat_bounds, dims=['x', 'nv'])

        self.source_grid = source_grid
        return source_grid

    def create_target_grid(self: 'MPASRemapper',
                          lon_min: float = -180.0,
                          lon_max: float = 180.0,
                          lat_min: float = -90.0,
                          lat_max: float = 90.0,
                          dlon: float = 1.0,
                          dlat: float = 1.0) -> xr.Dataset:
        """
        This method creates a regular latitude-longitude target grid specification based on the provided spatial extent and resolution parameters. It generates 1D arrays of longitude and latitude coordinates using numpy's arange function, ensuring that the grid covers the specified range with the given spacing. The resulting target grid is stored as an xarray Dataset with 'lon' and 'lat' coordinates, which can be used for building the regridder with ESMPy. This method allows users to easily define a regular target grid for remapping MPAS unstructured data, facilitating the transformation of data onto standard lat-lon grids for analysis and visualization. 

        Parameters:
            lon_min (float): Minimum longitude of the target grid in degrees (default: -180.0).
            lon_max (float): Maximum longitude of the target grid in degrees (default: 180.0).
            lat_min (float): Minimum latitude of the target grid in degrees (default: -90.0).
            lat_max (float): Maximum latitude of the target grid in degrees (default: 90.0).
            dlon (float): Longitude spacing of the target grid in degrees (default: 1.0).
            dlat (float): Latitude spacing of the target grid in degrees (default: 1.0). 

        Returns:
            xr.Dataset: Target grid dataset with 'lon' and 'lat' coordinates. 
        """
        lon = np.arange(lon_min, lon_max + dlon / 2, dlon)
        lat = np.arange(lat_min, lat_max + dlat / 2, dlat)

        target_grid = xr.Dataset({
            'lon': xr.DataArray(lon, dims=['lon']),
            'lat': xr.DataArray(lat, dims=['lat'])
        })

        self.target_grid = target_grid

        print(f"Created target grid: {len(lon)} x {len(lat)} points")
        print(f"  Longitude: [{lon_min:.2f}, {lon_max:.2f}] deg, spacing: {dlon:.3f} deg")
        print(f"  Latitude: [{lat_min:.2f}, {lat_max:.2f}] deg, spacing: {dlat:.3f} deg")

        return target_grid

    def build_regridder(self: 'MPASRemapper',
                       source_grid: Optional[xr.Dataset] = None,
                       target_grid: Optional[xr.Dataset] = None,
                       filename: Optional[str] = None,
                       comm: Optional[Any] = None) -> Any:
        """
        This method builds the regridder by computing the sparse weight matrix for remapping from the source grid to the target grid using ESMPy. It first resolves the source and target grid specifications, then checks if cached weights are available and valid based on the provided filename and internal settings. If cached weights are found, it loads them and returns immediately. If not, it proceeds to build the weights on rank 0 (in an MPI context) while other ranks wait at a barrier. After rank 0 finishes building and optionally saving the weights, all ranks synchronize again, and non-zero ranks either load the weights from the file or receive them via broadcast from rank 0. The method returns the computed weight matrix, which is typically a scipy.sparse.csr_matrix that can be used for efficient remapping of data arrays. 

        Parameters:
            source_grid (Optional[xr.Dataset]): Source grid dataset. If None, uses internally stored source grid (default: None).
            target_grid (Optional[xr.Dataset]): Target grid dataset. If None, uses internally stored target grid (default: None).
            filename (Optional[str]): Optional filename for caching weights. If None, a default name based on the method and grid shapes is used if weights_dir is set (default: None).
            comm (Optional[Any]): Optional MPI communicator for parallel execution. If provided, rank 0 builds weights and others load or receive them (default: None). 

        Returns:
            Any: The computed weight matrix for remapping, typically a scipy.sparse.csr_matrix. 
        """
        source_grid, target_grid = self._resolve_grids(source_grid, target_grid)
        weights_path = self._resolve_weights_path(source_grid, target_grid, filename)

        if self._try_load_cached_weights(weights_path):
            return self._weights

        using_mpi = comm is not None and comm.Get_size() > 1
        mpi_rank = comm.Get_rank() if using_mpi else 0

        if mpi_rank == 0:
            print(f"Building {self.method} regridder...")
            self._build_weights_on_rank0(source_grid, target_grid, weights_path)

        if using_mpi:
            self._sync_weights_across_ranks(comm, mpi_rank, weights_path)

        return self._weights

    def _resolve_grids(self: 'MPASRemapper',
                       source_grid: Optional[xr.Dataset],
                       target_grid: Optional[xr.Dataset],) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        This helper method resolves the source and target grid specifications by checking the provided arguments and falling back to internally stored grids if necessary. It ensures that both source and target grids are available before proceeding with building the regridder. If neither the provided arguments nor the internal variables contain the required grid specifications, it raises a ValueError indicating that the grids must be provided or prepared first. This method centralizes the logic for determining which grid specifications to use, simplifying the main build_regridder() method and ensuring that the necessary inputs are validated before any computationally expensive operations are performed.

        Parameters:
            source_grid (Optional[xr.Dataset]): Source grid dataset provided as an argument. If None, uses internally stored source grid.
            target_grid (Optional[xr.Dataset]): Target grid dataset provided as an argument. If None, uses internally stored target grid.

        Returns:
            Tuple[xr.Dataset, xr.Dataset]: Resolved source and target grid datasets to be used for building the regridder. 
        """
        if source_grid is None:
            if self.source_grid is None:
                raise ValueError("Source grid must be provided or prepared first")
            source_grid = self.source_grid

        if target_grid is None:
            if self.target_grid is None:
                raise ValueError("Target grid must be provided or created first")
            target_grid = self.target_grid

        return source_grid, target_grid 

    def _resolve_weights_path(self: 'MPASRemapper',
                              source_grid: xr.Dataset,
                              target_grid: xr.Dataset,
                              filename: Optional[str],) -> Optional[Path]:
        """
        This helper method determines the file path for caching the computed weights based on the provided filename argument and the internal weights_dir setting. If a filename is explicitly provided, it constructs the path using the weights_dir if available. If no filename is provided but weights_dir is set, it generates a default filename based on the remapping method and the shapes of the source and target grids. If neither a filename nor a weights directory is specified, it returns None, indicating that weights will not be cached to disk. This method centralizes the logic for determining where to save or load weights, making it easier to manage caching behavior across different runs and configurations.

        Parameters:
            source_grid (xr.Dataset): The source grid dataset, used to determine the shape for default filename generation.
            target_grid (xr.Dataset): The target grid dataset, used to determine the shape for default filename generation.
            filename (Optional[str]): Optional explicit filename for caching weights. If None, a default name is generated if weights_dir is set.

        Returns:
            Optional[Path]: The resolved file path for caching weights, or None if caching is not configured. 
        """
        if self.weights_dir is not None and filename is None:
            src_shape = len(source_grid['lon'])
            tgt_shape = f"{len(target_grid['lon'])}x{len(target_grid['lat'])}"
            filename = f"weights_{self.method}_{src_shape}to{tgt_shape}.nc"
        return (self.weights_dir / filename
                if self.weights_dir is not None and filename is not None
                else None)

    def _try_load_cached_weights(self: 'MPASRemapper', 
                                 weights_path: Optional[Path]) -> bool:
        """
        This helper method attempts to load pre-computed weights from a specified file path if caching is enabled and the file exists. It checks the internal reuse_weights setting to determine whether loading from cache is allowed. If the weights are successfully loaded, it updates the internal variables for weights, target shape, and cell_of_element accordingly, and returns True. If the file does not exist or loading is not allowed based on the settings, it returns False, indicating that the regridder will need to be built from scratch. This method provides a mechanism for efficiently reusing previously computed weights across runs, saving time when the same remapping configuration is used multiple times. 

        Parameters:
            weights_path (Optional[Path]): The file path to the cached weights. If None, the method will not attempt to load weights.

        Returns:
            bool: True if weights were successfully loaded, False otherwise.
        """
        if weights_path is None or not weights_path.exists() or not self.reuse_weights:
            return False

        print(f"Loading cached weights from {weights_path}")

        self._weights, self._tgt_shape, self._cell_of_element = \
            self._load_weights_netcdf(weights_path)

        self._n_src = int(self._weights.shape[1])
        print("Weights loaded successfully")
        return True

    def _prepare_source_esmpy(self: 'MPASRemapper',
                              source_grid: xr.Dataset,
                              source_lon_deg: np.ndarray,
                              source_lat_deg: np.ndarray,) -> Tuple[Any, int, bool, bool, Any, Optional[np.ndarray]]:
        """
        This helper method prepares the ESMF source object based on the provided source grid and its longitude and latitude coordinates in degrees. It determines whether the source grid is structured or unstructured, and whether conservative remapping is required based on the chosen method. For conservative remapping, it checks for the presence of cell boundary coordinates and builds an ESMF mesh accordingly. For structured grids, it builds an ESMF grid object. For unstructured grids without boundaries, it builds a locstream object and falls back to nearest-neighbour interpolation if bilinear or patch methods are requested. The method returns the prepared ESMF source object along with metadata about the number of source points, whether it's a mesh or grid, the normalization type for conservative remapping, and the cell of element mapping if applicable. This preparation step is crucial for ensuring that the regridder can be built correctly based on the characteristics of the source grid and the chosen remapping method. 

        Parameters:
            source_grid (xr.Dataset): The source grid dataset containing 'lon' and 'lat' coordinates, and optionally 'lon_b' and 'lat_b' for boundaries.
            source_lon_deg (np.ndarray): The longitude coordinates of the source grid in degrees.
            source_lat_deg (np.ndarray): The latitude coordinates of the source grid in degrees.

        Returns:
            Tuple[Any, int, bool, bool, Any, Optional[np.ndarray]]: A tuple containing the prepared ESMF source object, the number of source points, flags indicating if it's a mesh or grid, normalization type for conservative remapping, and the cell of element mapping if applicable. 
        """
        is_conservative = self.method in ('conservative', 'conservative_normed')
        src_is_structured = tuple(source_grid['lon'].dims) != tuple(source_grid['lat'].dims)

        if is_conservative:
            if 'lon_b' not in source_grid or 'lat_b' not in source_grid:
                raise ValueError(
                    "Conservative remapping requires cell boundary coordinates. "
                    "Pass 'lon_b' and 'lat_b' (shape nCells x nVertices) to "
                    "prepare_source_grid(), or use method='nearest_s2d' for "
                    "unstructured sources without boundaries."
                )
            raw_lon_b = source_grid['lon_b'].values
            raw_lat_b = source_grid['lat_b'].values
            
            flat_lon_b, flat_lat_b = _convert_coordinates_to_degrees(
                raw_lon_b.flatten(), raw_lat_b.flatten()
            )

            lon_b = np.where(flat_lon_b < 0, flat_lon_b + 360, flat_lon_b).reshape(raw_lon_b.shape)
            lat_b = flat_lat_b.reshape(raw_lat_b.shape)

            source_esmpy, cell_of_element = self._build_esmpy_mesh(
                source_lon_deg, source_lat_deg, lon_b, lat_b
            )

            norm_type = (esmpy.NormType.FRACAREA
                         if self.method == 'conservative_normed'
                         else None)

            return source_esmpy, len(cell_of_element), True, False, norm_type, cell_of_element

        if src_is_structured:
            source_esmpy, n_src_lon, n_src_lat = self._build_esmpy_grid(
                source_lon_deg, source_lat_deg
            )
            return source_esmpy, n_src_lon * n_src_lat, False, True, None, None

        source_esmpy = self._build_esmpy_locstream(source_lon_deg, source_lat_deg)

        if getattr(esmpy.RegridMethod, self._METHOD_MAP[self.method]) in {
            esmpy.RegridMethod.BILINEAR, esmpy.RegridMethod.PATCH
        }:
            warnings.warn(
                f"Method '{self.method}' is not supported for unstructured (LocStream) "
                "sources. Falling back to 'nearest_s2d'. Convert source to structured "
                f"grid first (e.g. via unstructured_to_structured_grid()) for '{self.method}' "
                "interpolation.",
                UserWarning,
                stacklevel=3,
            )
        return source_esmpy, len(source_lon_deg), False, False, None, None

    def _build_weights_on_rank0(self: 'MPASRemapper',
                                source_grid: xr.Dataset,
                                target_grid: xr.Dataset,
                                weights_path: Optional[Path],) -> None:
        """
        This helper method is responsible for building the weight matrix on rank 0 in an MPI context. It prepares the source and target grid specifications for ESMF, determines the appropriate regridding method based on the chosen remapping method and the characteristics of the source grid, and then builds the weight matrix using ESMPy's regridding capabilities. If extrapolation settings are provided, it applies extrapolation to unmapped points as needed. After building the weights, it optionally saves them to a file for caching purposes. This method encapsulates the core logic for constructing the regridder weights based on the source and target grids, ensuring that the process is efficient and correctly handles different grid types and remapping methods.

        Parameters:
            source_grid (xr.Dataset): The source grid dataset containing 'lon' and 'lat' coordinates, and optionally 'lon_b' and 'lat_b' for boundaries.
            target_grid (xr.Dataset): The target grid dataset containing 'lon' and 'lat' coordinates.
            weights_path (Optional[Path]): The file path to save the computed weights for caching. If None, weights will not be saved to disk.

        Returns:
            None
        """
        if self.method == 'patch':
            raise ValueError(
                "The 'patch' method is not supported for unstructured sources. "
                "First convert to a structured grid with unstructured_to_structured_grid(), "
                "then use an external structured regridder."
            )

        source_lon_deg, source_lat_deg = _convert_coordinates_to_degrees(
            source_grid['lon'].values, source_grid['lat'].values
        )

        source_lon_deg = np.where(source_lon_deg < 0, source_lon_deg + 360, source_lon_deg)

        lon_tgt = target_grid['lon'].values
        lat_tgt = target_grid['lat'].values
        target_lon_esmpy = np.where(lon_tgt < 0, lon_tgt + 360, lon_tgt)

        source_esmpy, n_src, src_is_mesh, src_is_grid, norm_type, cell_of_element = \
            self._prepare_source_esmpy(source_grid, source_lon_deg, source_lat_deg)

        is_conservative = self.method in ('conservative', 'conservative_normed')

        target_grid_esmpy, n_lon, n_lat = self._build_esmpy_target_grid(
            target_lon_esmpy, lat_tgt, add_corners=is_conservative
        )
        n_tgt = n_lon * n_lat

        regrid_method = getattr(esmpy.RegridMethod, self._METHOD_MAP[self.method])

        if not (src_is_mesh or src_is_grid) and regrid_method in {
            esmpy.RegridMethod.BILINEAR, esmpy.RegridMethod.PATCH
        }:
            regrid_method = esmpy.RegridMethod.NEAREST_STOD

        weight_matrix = self._build_weights_from_esmpy(
            source_esmpy, target_grid_esmpy, regrid_method,
            n_tgt=n_tgt,
            n_src=n_src,
            extrap_method=self.extrap_method,
            extrap_dist_exponent=self.extrap_dist_exponent or 2.0,
            extrap_num_src_pnts=self.extrap_num_src_pnts or 0,
            norm_type=norm_type,
            src_is_mesh=src_is_mesh,
            src_is_grid=src_is_grid,
        )

        target_grid_esmpy.destroy()
        source_esmpy.destroy()

        self._weights = weight_matrix
        self._n_src = n_src
        self._tgt_shape = (n_lat, n_lon)
        self._cell_of_element = cell_of_element

        if weights_path is not None:
            self._save_weights_netcdf(
                weights_path, weight_matrix, n_src, self._tgt_shape, self.method,
                cell_of_element=cell_of_element,
            )
            print(f"Weights saved to {weights_path}")

        print("Regridder built successfully")

    def _sync_weights_across_ranks(self: 'MPASRemapper',
                                   comm: Any,
                                   mpi_rank: int,
                                   weights_path: Optional[Path],) -> None:
        """
        This helper method synchronizes the computed weights across MPI ranks after rank 0 has built the regridder. It first ensures that all ranks reach a barrier after rank 0 finishes building the weights. Then, non-zero ranks either load the weights from the file that rank 0 wrote (if a weights_path is provided) or receive the weights via broadcast from rank 0 if no file-based caching is used. This method ensures that all ranks have access to the same weight matrix for remapping operations, allowing for consistent results across parallel executions. By handling both file-based loading and in-memory broadcasting, it provides flexibility in how weights are shared among ranks based on the user's configuration and available resources.

        Parameters:
            comm (Any): The MPI communicator used for synchronization and communication between ranks.
            mpi_rank (int): The rank of the current process within the MPI communicator.
            weights_path (Optional[Path]): The file path to load the computed weights from. If None, weights will be broadcast in-memory from rank 0.

        Returns:
            None
        """
        comm.Barrier()

        # Path A: non-zero ranks load from the file rank 0 wrote
        if weights_path is not None and mpi_rank != 0:
            self._weights, self._tgt_shape, self._cell_of_element = \
                self._load_weights_netcdf(weights_path)
            self._n_src = int(self._weights.shape[1])
            return

        # Path B: no weights file — broadcast COO arrays in-memory
        if weights_path is None:
            self._broadcast_weights_in_memory(comm, mpi_rank)

    def _broadcast_weights_in_memory(self: 'MPASRemapper',
                                     comm: Any,
                                     mpi_rank: int,) -> None:
        """
        This helper method broadcasts the computed weights from rank 0 to all other ranks in an MPI context when no file-based caching is used. On rank 0, it converts the weight matrix to COO format and prepares the row indices, column indices, and weight values for broadcasting. It also prepares the cell_of_element array if it exists. All ranks then participate in broadcasting the metadata (such as the number of non-zero entries and grid shapes) and the weight data arrays. Non-zero ranks receive this information and reconstruct the sparse weight matrix in CSR format for use in remapping operations. This method ensures that all ranks have access to the same weight matrix without relying on file I/O, which can be more efficient in certain parallel execution environments.

        Parameters:
            comm (Any): The MPI communicator used for synchronization and communication between ranks.
            mpi_rank (int): The rank of the current process within the MPI communicator.

        Returns:
            None
        """
        if mpi_rank == 0:
            weights_coo = self._weights.tocoo()

            metadata: Any = {
                'nnz':     int(weights_coo.nnz),
                'n_src':   int(self._n_src),
                'tgt_lat': int(self._tgt_shape[0]),
                'tgt_lon': int(self._tgt_shape[1]),
                'has_coe': self._cell_of_element is not None,
                'coe_len': (len(self._cell_of_element)
                            if self._cell_of_element is not None else 0),
            }

            row_indices    = weights_coo.row.astype(np.int32)
            col_indices    = weights_coo.col.astype(np.int32)
            weight_values  = weights_coo.data.astype(np.float64)

            bcast_coe      = (self._cell_of_element.astype(np.int64)
                              if self._cell_of_element is not None
                              else np.empty(0, np.int64))
        else:
            metadata = row_indices = col_indices = weight_values = bcast_coe = None

        metadata = comm.bcast(metadata, root=0)  # small dict — pickle OK

        if mpi_rank != 0:
            row_indices   = np.empty(metadata['nnz'], dtype=np.int32)
            col_indices   = np.empty(metadata['nnz'], dtype=np.int32)
            weight_values = np.empty(metadata['nnz'], dtype=np.float64)
            bcast_coe     = (np.empty(metadata['coe_len'], dtype=np.int64)
                             if metadata['has_coe'] else np.empty(0, np.int64))

        comm.Bcast(row_indices,   root=0)  # buffer-level — no pickle, efficient
        comm.Bcast(col_indices,   root=0)
        comm.Bcast(weight_values, root=0)

        if metadata['has_coe']:
            comm.Bcast(bcast_coe, root=0)

        if mpi_rank != 0:
            n_tgt = metadata['tgt_lat'] * metadata['tgt_lon']
            self._weights = coo_matrix(
                (weight_values, (row_indices.astype(np.intp), col_indices.astype(np.intp))),
                shape=(n_tgt, metadata['n_src']),
            ).tocsr()
            self._n_src          = metadata['n_src']
            self._tgt_shape      = (metadata['tgt_lat'], metadata['tgt_lon'])
            self._cell_of_element = bcast_coe if metadata['has_coe'] else None

    def remap(self: 'MPASRemapper',
              data: Union[xr.DataArray, np.ndarray],
              keep_attrs: bool = True) -> xr.DataArray:
        """
        This method remaps 1-D data from the source grid to the target grid using the pre-computed weights. It first checks if the regridder has been built and the weights are available. Then it flattens the input data and applies the sparse weight matrix to perform the remapping. If skipna is enabled and there are NaN values in the input, it re-normalizes the weights for valid points to ensure that the output is not biased by missing data. Finally, it reshapes the remapped data to match the target grid dimensions and returns it as an xarray DataArray with appropriate coordinates and attributes preserved if requested. This method allows for efficient remapping of MPAS data onto regular lat-lon grids while handling potential issues with missing data gracefully. 

        Parameters:
            data (Union[xr.DataArray, np.ndarray]): Input data on source grid, shape (nCells,).
            keep_attrs (bool): Whether to preserve attributes from the input DataArray in the output (default: True). 

        Returns:
            xr.DataArray: Remapped data on target grid with dimensions ('lat', 'lon') and appropriate coordinates. 
        """
        if self._weights is None:
            raise ValueError(
                "Regridder must be built before remapping. Call build_regridder() first."
            )

        if isinstance(data, xr.DataArray):
            attrs = data.attrs if keep_attrs else {}
            data_vals = data.values
        else:
            attrs = {}
            data_vals = np.asarray(data)

        data_flat = data_vals.flatten().astype(np.float64)

        if self._cell_of_element is not None:
            data_flat = data_flat[self._cell_of_element]

        if self.skipna and np.any(np.isnan(data_flat)):
            valid = ~np.isnan(data_flat)
            result_flat = self._weights @ np.where(valid, data_flat, 0.0)
            weight_sum = self._weights @ valid.astype(np.float64)
            with np.errstate(invalid='ignore', divide='ignore'):
                result_flat = np.where(weight_sum > 0, result_flat / weight_sum, np.nan)
        else:
            result_flat = self._weights @ data_flat

        n_lat, n_lon = self._tgt_shape
        result_2d = result_flat.reshape(n_lat, n_lon)

        lon_1d = self.target_grid['lon'].values
        lat_1d = self.target_grid['lat'].values

        return xr.DataArray(
            result_2d,
            dims=['lat', 'lon'],
            coords={'lat': lat_1d, 'lon': lon_1d},
            attrs=attrs,
        )

    def remap_dataset(self: 'MPASRemapper',
                      dataset: xr.Dataset,
                      variables: Optional[List[str]] = None,
                      skip_missing: bool = True) -> xr.Dataset:
        """
        This method remaps multiple variables from an xarray Dataset using the pre-computed weights. It iterates over the specified variables (or all data variables if none are specified), checks for their presence in the dataset, and applies the remapping to each variable using the remap() method. If a variable is missing and skip_missing is enabled, it logs a warning and continues with the next variable; otherwise, it raises an error. The remapped variables are collected into a new xarray Dataset, which also preserves global attributes from the input dataset and adds metadata about the remapping process. This method provides a convenient way to remap entire datasets of MPAS data onto regular lat-lon grids while handling potential issues with missing variables gracefully.  

        Parameters:
            dataset (xr.Dataset): Input dataset containing variables to remap.
            variables (Optional[List[str]]): List of variable names to remap. If None, all data variables are remapped (default: None).
            skip_missing (bool): Whether to skip variables that are not found in the dataset with a warning (default: True). If False, raises an error when a variable is missing. 

        Returns:
            xr.Dataset: New dataset containing remapped variables on the target grid, with preserved attributes and remapping metadata. 
        """
        if self._weights is None:
            raise ValueError("Regridder must be built before remapping")

        if variables is None:
            variables = list(dataset.data_vars)

        remapped_vars = {}

        for var_name in variables:
            if var_name not in dataset:
                if skip_missing:
                    print(f"Warning: Variable '{var_name}' not found, skipping")
                    continue
                else:
                    raise ValueError(f"Variable '{var_name}' not found in dataset")

            print(f"Remapping variable: {var_name}")

            try:
                remapped_vars[var_name] = self.remap(dataset[var_name])
            except Exception as e:
                print(f"Error remapping {var_name}: {e}")
                if not skip_missing:
                    raise

        output_ds = xr.Dataset(remapped_vars)

        output_ds.attrs = dataset.attrs.copy()
        output_ds.attrs['remapping_method'] = self.method
        output_ds.attrs['remapped_by'] = 'MPASRemapper'

        return output_ds

    @staticmethod
    def unstructured_to_structured_grid(data: Union[xr.DataArray, np.ndarray],
                                        lon: Union[np.ndarray, xr.DataArray],
                                        lat: Union[np.ndarray, xr.DataArray],
                                        intermediate_resolution: float = 0.1,
                                        lon_min: Optional[float] = None,
                                        lon_max: Optional[float] = None,
                                        lat_min: Optional[float] = None,
                                        lat_max: Optional[float] = None,
                                        buffer: float = 2.0) -> Tuple[xr.DataArray, xr.Dataset]:
        """
        This static method converts 1-D unstructured MPAS data to a 2-D structured grid using nearest-neighbour interpolation via scipy.spatial.KDTree. It first converts the input longitude and latitude coordinates to degrees if they are in radians, and determines the spatial extent of the target grid based on the coordinate ranges with an optional buffer. It then creates a regular lat-lon grid at the specified intermediate resolution and uses a KDTree to find the nearest source point for each target grid point. The input data is remapped onto the structured grid, and an xarray DataArray is returned along with a Dataset containing the grid specification with coordinate bounds suitable for conservative remapping. This method provides a convenient way to convert unstructured MPAS data into a format that can be used with structured regridders or for visualization purposes. 

        Parameters:
            data (Union[xr.DataArray, np.ndarray]): Input data on unstructured grid, shape (nCells,).
            lon (Union[np.ndarray, xr.DataArray]): Longitude coordinates of the unstructured grid points in degrees or radians.
            lat (Union[np.ndarray, xr.DataArray]): Latitude coordinates of the unstructured grid points in degrees or radians.
            intermediate_resolution (float): Resolution in degrees for the intermediate structured grid (default: 0.1°).
            lon_min (Optional[float]): Minimum longitude for the structured grid. If None, calculated from input coordinates with buffer (default: None).
            lon_max (Optional[float]): Maximum longitude for the structured grid. If None, calculated from input coordinates with buffer (default: None).
            lat_min (Optional[float]): Minimum latitude for the structured grid. If None, calculated from input coordinates with buffer (default: None).
            lat_max (Optional[float]): Maximum latitude for the structured grid. If None, calculated from input coordinates with buffer (default: None).
            buffer (float): Buffer in degrees to add to the coordinate ranges when calculating grid extent if lon_min/lon_max/lat_min/lat_max are not provided (default: 2.0°).

        Returns:
            Tuple[xr.DataArray, xr.Dataset]: A tuple containing the remapped data as an xarray DataArray on the structured grid, and an xarray Dataset with the grid specification including coordinate bounds. 
        """
        try:
            from scipy.spatial import KDTree
        except ImportError:
            raise ImportError("scipy is required. Install with: pip install scipy")

        if isinstance(data, xr.DataArray):
            data_attrs = data.attrs
            data_values = data.values
        else:
            data_attrs = {}
            data_values = data

        lon_deg, lat_deg = _convert_coordinates_to_degrees(lon, lat)

        if lon_min is None:
            lon_min = float(np.min(lon_deg) - buffer)

        if lon_max is None:
            lon_max = float(np.max(lon_deg) + buffer)

        if lat_min is None:
            lat_min = float(max(-90.0, np.min(lat_deg) - buffer))

        if lat_max is None:
            lat_max = float(min(90.0, np.max(lat_deg) + buffer))

        print("Creating intermediate 2D structured grid:")
        print(f"  Lon range: [{lon_min:.2f}, {lon_max:.2f}]°")
        print(f"  Lat range: [{lat_min:.2f}, {lat_max:.2f}]°")
        print(f"  Resolution: {intermediate_resolution}°")

        intermediate_lons = np.arange(lon_min, lon_max + intermediate_resolution / 2,
                                      intermediate_resolution)

        intermediate_lats = np.arange(lat_min, lat_max + intermediate_resolution / 2,
                                      intermediate_resolution)

        lon_2d, lat_2d = np.meshgrid(intermediate_lons, intermediate_lats)
        n_lon, n_lat = len(intermediate_lons), len(intermediate_lats)

        print(f"  Grid size: {n_lat} x {n_lon} = {n_lat * n_lon:,} points")
        print(f"  Original unstructured points: {len(lon_deg):,}")

        source_points = np.column_stack([lon_deg, lat_deg])
        tree = KDTree(source_points)

        target_points = np.column_stack([lon_2d.flatten(), lat_2d.flatten()])
        distances, indices = tree.query(target_points)

        data_2d = data_values[indices].reshape(lon_2d.shape)

        structured_data = xr.DataArray(
            data_2d,
            dims=['lat', 'lon'],
            coords={
                'lon': intermediate_lons,
                'lat': intermediate_lats
            },
            attrs=data_attrs
        )

        lon_b = _compute_grid_bounds(intermediate_lons, intermediate_resolution)
        lat_b = _compute_grid_bounds(intermediate_lats, intermediate_resolution)

        structured_grid = xr.Dataset({
            'lon': xr.DataArray(intermediate_lons, dims=['lon']),
            'lat': xr.DataArray(intermediate_lats, dims=['lat']),
            'lon_b': xr.DataArray(lon_b, dims=['lon_b']),
            'lat_b': xr.DataArray(lat_b, dims=['lat_b'])
        })

        structured_data.attrs['grid_conversion'] = 'unstructured_to_structured_kdtree'
        structured_data.attrs['intermediate_resolution'] = intermediate_resolution

        print("✓ Converted to 2D structured grid with bounds (ready for conservative remapping)")

        return structured_data, structured_grid

    def cleanup(self: 'MPASRemapper') -> None:
        """
        This method cleans up resources used by the regridder, such as clearing the weights and grid specifications from memory. It sets the internal variables for weights, cell mapping, source grid, and target grid to None, allowing for garbage collection to free up memory. This can be useful when the regridder is no longer needed or when building a new regridder with different grids or methods, ensuring that memory usage is optimized and preventing potential issues with stale data. 

        Parameters:
            None

        Returns:
            None
        """
        self._weights = None
        self._cell_of_element = None
        self.source_grid = None
        self.target_grid = None
        print("Regridder resources cleaned up")

    @staticmethod
    def estimate_memory_usage(n_source: int,
                              n_target: int,
                              method: str) -> float:
        """
        This static method estimates the total memory usage in gigabytes for building the regridder based on the number of source and target grid points and the chosen interpolation method. It calculates the number of non-zero entries in the weight matrix per target point based on the method, then estimates the memory required to store the weight matrix and the source/target grid data. The total estimated memory usage is returned in gigabytes, which can help users understand the resource requirements for building the regridder with their specific grid sizes and method choice. 

        Parameters:
            n_source (int): Number of source grid points.
            n_target (int): Number of target grid points.
            method (str): Remapping method to use. Supported options: 'bilinear', 'conservative', 'conservative_normed', 'patch', 'nearest_s2d', 'nearest_d2s'. 

        Returns:
            float: Estimated total memory usage in GB.
        """
        if method in ['conservative', 'conservative_normed']:
            nnz_per_target = 8
        elif method == 'bilinear':
            nnz_per_target = 4
        elif method == 'patch':
            nnz_per_target = 16
        else:
            nnz_per_target = 1

        weight_memory = (n_target * nnz_per_target * 8 * 2) / 1e9
        data_memory = (n_source + n_target) * 8 / 1e9
        total_memory = weight_memory + data_memory

        return total_memory

    @staticmethod
    def _build_esmpy_locstream(lon_deg: np.ndarray,
                               lat_deg: np.ndarray) -> Any:
        """
        This static method creates an ESMPy LocStream object for unstructured source points. It takes the longitude and latitude coordinates of the source points in degrees, initializes a LocStream with the appropriate coordinate system, and assigns the longitude and latitude values to the LocStream fields. The resulting LocStream can be used as the source grid for remapping methods that do not require cell boundary information, such as nearest-neighbour interpolation. This method provides a way to represent unstructured MPAS grid points in a format compatible with ESMPy's remapping capabilities.  

        Parameters:
            lon_deg (np.ndarray): Longitude coordinates of the source points in degrees.
            lat_deg (np.ndarray): Latitude coordinates of the source points in degrees. 

        Returns:
            Any: An ESMPy LocStream object representing the unstructured source points. 
        """
        locstream = esmpy.LocStream(len(lon_deg), coord_sys=esmpy.CoordSys.SPH_DEG)
        locstream['ESMF:Lon'] = lon_deg.astype(np.float64)
        locstream['ESMF:Lat'] = lat_deg.astype(np.float64)
        return locstream

    @staticmethod
    def _build_esmpy_mesh(lon_deg: np.ndarray,
                          lat_deg: np.ndarray,
                          lon_bounds: np.ndarray,
                          lat_bounds: np.ndarray) -> Any:
        """
        This static method creates an ESMPy Mesh object for unstructured source grids with polygon cells defined by their vertex coordinates. It takes the cell-centre longitude and latitude coordinates, as well as the corresponding vertex coordinates for each cell, and constructs a Mesh by fan-triangulating the polygon cells if they have more than 4 vertices. The method initializes the Mesh with the appropriate coordinate system, adds nodes corresponding to the triangle vertices, and defines elements that connect these nodes while keeping track of which original cell each triangle belongs to. The resulting Mesh can be used as the source grid for conservative remapping methods that require cell boundary information. This method provides a way to represent unstructured MPAS grids with complex cell shapes in a format compatible with ESMPy's remapping capabilities. 

        Parameters:
            lon_deg (np.ndarray): Longitude coordinates of the cell centres in degrees, shape (nCells,).
            lat_deg (np.ndarray): Latitude coordinates of the cell centres in degrees, shape (nCells,).
            lon_bounds (np.ndarray): Longitude coordinates of the cell vertices in degrees, shape (nCells, nVertices).
            lat_bounds (np.ndarray): Latitude coordinates of the cell vertices in degrees, shape (nCells, nVertices). 

        Returns:
            Any: An ESMPy Mesh object representing the unstructured source grid, and an array mapping each mesh element to its parent cell index. 
        """
        n_cells, nv = lon_bounds.shape

        if nv in (3, 4):
            element_type = esmpy.MeshElemType.TRI if nv == 3 else esmpy.MeshElemType.QUAD
            vertices_per_element = nv
            triangle_lon_bounds = lon_bounds
            triangle_lat_bounds = lat_bounds
            cell_of_element = np.arange(n_cells, dtype=np.int64)
        else:
            triangles_per_cell = nv - 2
            n_elements = n_cells * triangles_per_cell
            triangle_lon_bounds = np.empty((n_elements, 3), dtype=np.float64)
            triangle_lat_bounds = np.empty((n_elements, 3), dtype=np.float64)
            cell_of_element = np.empty(n_elements, dtype=np.int64)
            triangle_index = 0
            for cell_index in range(n_cells):
                for j in range(1, nv - 1):
                    triangle_lon_bounds[triangle_index, 0] = lon_bounds[cell_index, 0]
                    triangle_lon_bounds[triangle_index, 1] = lon_bounds[cell_index, j]
                    triangle_lon_bounds[triangle_index, 2] = lon_bounds[cell_index, j + 1]
                    triangle_lat_bounds[triangle_index, 0] = lat_bounds[cell_index, 0]
                    triangle_lat_bounds[triangle_index, 1] = lat_bounds[cell_index, j]
                    triangle_lat_bounds[triangle_index, 2] = lat_bounds[cell_index, j + 1]
                    cell_of_element[triangle_index] = cell_index
                    triangle_index += 1
            element_type = esmpy.MeshElemType.TRI
            vertices_per_element = 3

        n_elements = len(cell_of_element)
        n_nodes = n_elements * vertices_per_element
        node_lons = triangle_lon_bounds.flatten().astype(np.float64)
        node_lats = triangle_lat_bounds.flatten().astype(np.float64)
        node_ids = np.arange(1, n_nodes + 1, dtype=np.int32)
        node_owners = np.zeros(n_nodes, dtype=np.int32)

        mesh = esmpy.Mesh(
            parametric_dim=2, spatial_dim=2,
            coord_sys=esmpy.CoordSys.SPH_DEG,
        )

        mesh.add_nodes(
            node_count=n_nodes,
            node_ids=node_ids,
            node_coords=np.column_stack([node_lons, node_lats]).flatten(),
            node_owners=node_owners,
        )

        element_ids = np.arange(1, n_elements + 1, dtype=np.int32)
        element_types = np.full(n_elements, element_type, dtype=np.int32)
        element_conn = np.arange(0, n_nodes, dtype=np.int32)

        element_coords = np.column_stack([
            lon_deg[cell_of_element].astype(np.float64),
            lat_deg[cell_of_element].astype(np.float64),
        ]).flatten()

        mesh.add_elements(
            element_count=n_elements,
            element_ids=element_ids,
            element_types=element_types,
            element_conn=element_conn,
            element_mask=None,
            element_area=None,
            element_coords=element_coords,
        )

        return mesh, cell_of_element

    @staticmethod
    def _build_esmpy_grid(lon_1d: np.ndarray,
                          lat_1d: np.ndarray,
                          add_corners: bool = False) -> Tuple[Any, int, int]:
        """
        This static method creates an ESMPy Grid object for a regular latitude-longitude grid. It takes 1D arrays of longitude and latitude coordinates in degrees, initializes a Grid with the appropriate coordinate system, and assigns the longitude and latitude values to the grid's center coordinates. If add_corners is True, it also computes the corner coordinates based on the input 1D arrays and adds them to the grid, which is necessary for conservative remapping methods. The method returns the constructed ESMPy Grid object along with the number of longitude and latitude points, which are essential for building the regridder and understanding the dimensions of the target grid. This method provides a way to represent regular lat-lon grids in a format compatible with ESMPy's remapping capabilities. 

        Parameters:
            lon_1d (np.ndarray): 1D array of longitude coordinates in degrees.
            lat_1d (np.ndarray): 1D array of latitude coordinates in degrees.
            add_corners (bool): Whether to compute and add corner coordinates for conservative remapping (default: False). 

        Returns:
            Tuple[Any, int, int]: An ESMPy Grid object representing the regular lat-lon grid, and the number of longitude and latitude points (n_lon, n_lat). 
        """
        n_lon, n_lat = len(lon_1d), len(lat_1d)

        grid = esmpy.Grid(
            np.array([n_lon, n_lat]),
            coord_sys=esmpy.CoordSys.SPH_DEG,
        )

        grid.add_coords(staggerloc=esmpy.StaggerLoc.CENTER)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)            # each (n_lat, n_lon)
        grid.coords[esmpy.StaggerLoc.CENTER][0][:] = lon_2d.T   # stored as (n_lon, n_lat)
        grid.coords[esmpy.StaggerLoc.CENTER][1][:] = lat_2d.T

        if add_corners:
            dlon = float(lon_1d[1] - lon_1d[0]) if len(lon_1d) > 1 else 1.0
            dlat = float(lat_1d[1] - lat_1d[0]) if len(lat_1d) > 1 else 1.0

            lon_b = _compute_grid_bounds(lon_1d, dlon)    # (n_lon + 1,)
            lat_b = _compute_grid_bounds(lat_1d, dlat)    # (n_lat + 1,)

            grid.add_coords(staggerloc=esmpy.StaggerLoc.CORNER)
            lon_b_2d, lat_b_2d = np.meshgrid(lon_b, lat_b)   # (n_lat+1, n_lon+1)

            grid.coords[esmpy.StaggerLoc.CORNER][0][:] = lon_b_2d.T
            grid.coords[esmpy.StaggerLoc.CORNER][1][:] = lat_b_2d.T

        return grid, n_lon, n_lat

    @staticmethod
    def _build_esmpy_target_grid(lon_1d: np.ndarray,
                                 lat_1d: np.ndarray,
                                 add_corners: bool = False) -> Tuple[Any, int, int]:
        """
        This static method is a wrapper around _build_esmpy_grid specifically for building the target grid. It takes the longitude and latitude coordinates for the target grid, along with an option to add corner coordinates if conservative remapping is being used, and calls _build_esmpy_grid to construct the ESMPy Grid object. The method returns the constructed target grid along with the number of longitude and latitude points, which are essential for building the regridder and understanding the dimensions of the target grid. This separation allows for clearer code organization and potential future customization specific to target grids if needed. 

        Parameters:
            lon_1d (np.ndarray): 1D array of longitude coordinates for the target grid in degrees.
            lat_1d (np.ndarray): 1D array of latitude coordinates for the target grid in degrees.
            add_corners (bool): Whether to compute and add corner coordinates for conservative remapping (default: False). 

        Returns:
            Tuple[Any, int, int]: An ESMPy Grid object representing the target grid, and the number of longitude and latitude points (n_lon, n_lat). 
        """
        return MPASRemapper._build_esmpy_grid(lon_1d, lat_1d, add_corners=add_corners)

    @staticmethod
    def _build_weights_from_esmpy(src_obj: Any,
                                  tgt_grid: Any,
                                  regrid_method: Any,
                                  n_tgt: int,
                                  n_src: int,
                                  extrap_method: Optional[str] = None,
                                  extrap_dist_exponent: float = 2.0,
                                  extrap_num_src_pnts: int = 0,
                                  norm_type: Any = None,
                                  src_is_mesh: bool = False, 
                                  src_is_grid: bool = False) -> Any:
        """
        This static method builds the sparse weight matrix for remapping using ESMPy's regridding capabilities. It takes the source grid object (which can be a Mesh, Grid, or LocStream), the target Grid object, the chosen regridding method, and various parameters related to extrapolation and normalization. The method creates ESMPy Field objects for the source and target grids, configures the regridding options based on the provided parameters, and constructs an ESMPy Regrid object to compute the weights. The resulting weights are extracted into a dictionary format, converted into a scipy.sparse.coo_matrix, and then transformed into a csr_matrix for efficient storage and application during remapping. Finally, the method cleans up the ESMPy objects to free resources and returns the sparse weight matrix. This method encapsulates the core logic of interfacing with ESMPy to generate the necessary weights for remapping MPAS data onto regular lat-lon grids. 

        Parameters:
            src_obj: The source grid object, which can be an ESMPy Mesh, Grid, or LocStream depending on the input data and method.
            tgt_grid: The target ESMPy Grid object representing the regular lat-lon grid.
            regrid_method: The ESMPy regridding method to use (e.g. esmpy.RegridMethod.BILINEAR).
            n_tgt: The total number of target grid points (n_lon * n_lat).
            n_src: The total number of source grid points (e.g. nCells for unstructured, or n_lon*n_lat for structured).
            extrap_method: Optional string specifying the extrapolation method to use for unmapped points (e.g. 'nearest_s2d', 'nearest_d2s', 'inverse_distance').
            extrap_dist_exponent: Exponent for inverse distance weighting when using inverse distance extrapolation (default: 2.0).
            extrap_num_src_pnts: Number of nearest source points to consider for extrapolation (default: 0, meaning all).
            norm_type: Optional ESMPy NormType to use for normalization of weights (e.g. esmpy.NormType.FRACAREA for conservative_normed).
            src_is_mesh: Whether the source is a Mesh (True) or not (False). This determines how the source Field is created.
            src_is_grid: Whether the source is a Grid (True) or not (False). This determines how the source Field is created. 

        Returns:
            Any: A scipy.sparse.csr_matrix representing the weight matrix for remapping. 
        """
        if src_is_mesh:
            src_field = esmpy.Field(src_obj, meshloc=esmpy.MeshLoc.ELEMENT)
        elif src_is_grid:
            src_field = esmpy.Field(src_obj, staggerloc=esmpy.StaggerLoc.CENTER)
        else:
            src_field = esmpy.Field(src_obj)

        tgt_field = esmpy.Field(tgt_grid, staggerloc=esmpy.StaggerLoc.CENTER)

        extrapolation_method = esmpy.ExtrapMethod.NONE

        if extrap_method is not None:
            dist_to_src_method = getattr(esmpy.ExtrapMethod, 'NEAREST_D2SRATIO', esmpy.ExtrapMethod.NEAREST_STOD)
            extrap_method_map = {
                'nearest_s2d':      esmpy.ExtrapMethod.NEAREST_STOD,
                'nearest_d2s':      dist_to_src_method,
                'inverse_distance': esmpy.ExtrapMethod.NEAREST_IDAVG,
                'inverse_dist':     esmpy.ExtrapMethod.NEAREST_IDAVG,
            }
            extrapolation_method = extrap_method_map.get(extrap_method, esmpy.ExtrapMethod.NONE)

        regrid_kwargs: dict = dict(
            regrid_method=regrid_method,
            unmapped_action=esmpy.UnmappedAction.IGNORE,
            ignore_degenerate=True,  # skip degenerate cells (e.g. MPAS pole cells)
            extrap_method=extrapolation_method,
            extrap_num_src_pnts=int(extrap_num_src_pnts),
            extrap_dist_exponent=float(extrap_dist_exponent),
            factors=True,  # required so get_weights_dict() can retrieve them
        )

        if norm_type is not None:
            regrid_kwargs['norm_type'] = norm_type

        regrid = esmpy.Regrid(src_field, tgt_field, **regrid_kwargs)
        weights_dict = regrid.get_weights_dict(deep_copy=True)

        row = np.asarray(weights_dict['row_dst'], dtype=np.intp) - 1   # 0-indexed
        col = np.asarray(weights_dict['col_src'], dtype=np.intp) - 1
        weight_values = np.asarray(weights_dict['weights'], dtype=np.float64)
        weight_matrix = coo_matrix((weight_values, (row, col)), shape=(n_tgt, n_src)).tocsr()

        regrid.destroy()
        tgt_field.destroy()
        src_field.destroy()

        return weight_matrix

    @staticmethod
    def _save_weights_netcdf(path: Path,
                             weight_matrix: Any,
                             n_src: int,
                             tgt_shape: Tuple[int, int],
                             method: str,
                             cell_of_element: Optional[np.ndarray] = None) -> None:
        """
        This static method saves the sparse weight matrix to a NetCDF file in COO format. It takes the weight matrix, the number of source points, the shape of the target grid, the interpolation method used, and an optional array mapping mesh elements to their parent cells (for fan-triangulated meshes). The method converts the sparse matrix to COO format to extract the row indices, column indices, and weight values, which are then stored as DataArrays in an xarray Dataset. The dataset also includes attributes for the number of source and target points, target grid shape, remapping method, and metadata about the remapping process. Finally, the dataset is saved to a NetCDF file at the specified path. This allows for efficient storage and later retrieval of pre-computed weights for remapping operations without needing to rebuild the regridder from scratch. 

        Parameters:
            path: Path to the NetCDF file where the weights will be saved.
            weight_matrix: The sparse weight matrix (scipy.sparse.csr_matrix) to be saved.
            n_src: The number of source grid points.
            tgt_shape: The shape of the target grid as a tuple (n_lat, n_lon).
            method: The interpolation method used (e.g. 'nearest_s2d', 'nearest_d2s', 'conservative', 'conservative_normed').
            cell_of_element: Optional array mapping mesh elements to their parent cells (for fan-triangulated meshes).

        Returns:
            None
        """
        weights_coo = weight_matrix.tocoo()

        data_vars: dict = {
            'row': xr.DataArray(weights_coo.row.astype(np.int32) + 1, dims=['nnz']),
            'col': xr.DataArray(weights_coo.col.astype(np.int32) + 1, dims=['nnz']),
            'S':   xr.DataArray(weights_coo.data.astype(np.float64), dims=['nnz']),
        }

        if cell_of_element is not None:
            data_vars['cell_of_element'] = xr.DataArray(
                cell_of_element.astype(np.int64), dims=['n_elements']
            )

        ds = xr.Dataset(
            data_vars,
            attrs={
                'n_src':         int(n_src),
                'n_dst':         int(np.prod(tgt_shape)),
                'shape_tgt_lat': int(tgt_shape[0]),
                'shape_tgt_lon': int(tgt_shape[1]),
                'method':        method,
                'remapped_by':   'MPASRemapper/ESMPy',
            },
        )

        ds.to_netcdf(path)

    @staticmethod
    def _load_weights_netcdf(path: Path) -> Tuple[Any, Tuple[int, int], Optional[np.ndarray]]:
        """
        This static method loads the sparse weight matrix from a NetCDF file that was saved in COO format. It reads the row indices, column indices, and weight values from the dataset, as well as the attributes containing the number of source and target points and the target grid shape. The method reconstructs the sparse weight matrix as a scipy.sparse.csr_matrix and returns it along with the target grid shape and an optional array mapping mesh elements to their parent cells if it was included in the dataset. This allows for efficient retrieval of pre-computed weights for remapping operations without needing to rebuild the regridder from scratch, enabling faster remapping of MPAS data onto regular lat-lon grids using previously saved weights. 

        Parameters:
            path: Path to the NetCDF file from which the weights will be loaded. 

        Returns:
            Tuple[Any, Tuple[int, int], Optional[np.ndarray]]: A tuple containing the sparse weight matrix as a scipy.sparse.csr_matrix, the shape of the target grid as a tuple (n_lat, n_lon), and an optional array mapping mesh elements to their parent cells if it was included in the dataset. 
        """
        ds = xr.open_dataset(path)

        row  = ds['row'].values.astype(np.intp) - 1
        col  = ds['col'].values.astype(np.intp) - 1
        vals = ds['S'].values.astype(np.float64)

        n_src = int(ds.attrs['n_src'])
        n_dst = int(ds.attrs['n_dst'])

        tgt_shape = (int(ds.attrs['shape_tgt_lat']), int(ds.attrs['shape_tgt_lon']))

        cell_of_element: Optional[np.ndarray] = (
            ds['cell_of_element'].values.astype(np.int64)
            if 'cell_of_element' in ds
            else None
        )

        ds.close()
        weight_matrix = coo_matrix((vals, (row, col)), shape=(n_dst, n_src)).tocsr()
        return weight_matrix, tgt_shape, cell_of_element

def _add_wrapped_boundary_points(source_points: np.ndarray,
                                 data_values: np.ndarray,
                                 lon_deg: np.ndarray,
                                 lat_deg: np.ndarray,
                                 lon_min: float,
                                 lon_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    This helper function adds wrapped boundary points to the source grid for global continuity when the longitude range exceeds 180 degrees. It identifies points near the high and low longitude boundaries and creates additional points by wrapping them around the globe (subtracting 360 degrees for points near the high boundary and adding 360 degrees for points near the low boundary). The corresponding data values for these wrapped points are also duplicated to maintain consistency. This is important for ensuring that interpolation methods, especially those that rely on spatial proximity, can properly handle edge cases at the international date line and provide seamless remapping results across global datasets. The function returns the augmented source points and data values with the added wrapped boundary points included.

    Parameters:
        source_points (np.ndarray): Original source points with shape containing longitude and latitude coordinates in degrees.
        data_values (np.ndarray): Original data values corresponding to the source points.
        lon_deg (np.ndarray): Longitude coordinates of the source points in degrees
        lat_deg (np.ndarray): Latitude coordinates of the source points in degrees
        lon_min (float): Minimum longitude boundary in degrees
        lon_max (float): Maximum longitude boundary in degrees

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the augmented source points and data values with wrapped boundary points included for global continuity. 
    """
    lon_range = lon_max - lon_min

    if not (lon_range > 180 and lon_min >= 0 and lon_max > 180):
        return source_points, data_values

    wrap_threshold = 10.0

    original_data = data_values
    near_high = lon_deg > (lon_max - wrap_threshold)

    if np.any(near_high):
        source_points = np.vstack([
            source_points,
            np.column_stack([lon_deg[near_high] - 360.0, lat_deg[near_high]])
        ])
        data_values = np.concatenate([data_values, original_data[near_high]])

    near_low = lon_deg < (lon_min + wrap_threshold)

    if np.any(near_low):
        source_points = np.vstack([
            source_points,
            np.column_stack([lon_deg[near_low] + 360.0, lat_deg[near_low]])
        ])
        data_values = np.concatenate([data_values, original_data[near_low]])

    n_wrapped = int(np.sum(near_high)) + int(np.sum(near_low))
    print(f"  Added {n_wrapped} wrapped boundary points for global continuity")
    return source_points, data_values

def remap_mpas_to_latlon(data: Union[xr.DataArray, np.ndarray], 
                         lon: Union[np.ndarray, xr.DataArray],
                         lat: Union[np.ndarray, xr.DataArray],
                         lon_min: float = -180.0,
                         lon_max: float = 180.0,
                         lat_min: float = -90.0,
                         lat_max: float = 90.0,
                         resolution: float = 1.0,
                         method: str = 'nearest') -> xr.DataArray:
    """
    This function provides a convenient interface for remapping MPAS unstructured grid data to a regular latitude-longitude grid using a KDTree-based nearest neighbor interpolation method. It automatically detects the coordinate units (degrees or radians) and converts them to degrees if necessary. The function generates a regular target grid based on user-defined spatial boundaries and resolution, then constructs a KDTree from the original unstructured coordinates to efficiently find the nearest source point for each target point on the regular grid. The resulting remapped data is returned as an xarray DataArray with proper coordinate labels corresponding to the target grid. This function is useful for quickly preparing MPAS data for visualization or analysis on regular lat-lon grids without requiring the full setup of xESMF regridders, while still maintaining spatial relationships as closely as possible. 
    
    Parameters:
        data (Union[xr.DataArray, np.ndarray]): Input data array defined on MPAS unstructured grid with shape (nCells,) or subset.
        lon (Union[np.ndarray, xr.DataArray]): MPAS cell center longitude coordinates in degrees or radians.
        lat (Union[np.ndarray, xr.DataArray]): MPAS cell center latitude coordinates in degrees or radians.
        lon_min (float): Minimum longitude for target grid in degrees (default: -180.0).
        lon_max (float): Maximum longitude for target grid in degrees (default: 180.0).
        lat_min (float): Minimum latitude for target grid in degrees (default: -90.0).
        lat_max (float): Maximum latitude for target grid in degrees (default: 90.0).
        resolution (float): Grid spacing in degrees for the regular latitude-longitude grid (default: 1.0).
        method (str): Interpolation method to use, options include 'nearest' and 'linear' (default: 'nearest').
    
    Returns:
        xr.DataArray: Remapped data on regular latitude-longitude grid as an xarray DataArray with appropriate coordinates. 
    """
    if method not in ['nearest', 'linear']:
        raise ValueError(f"method must be 'nearest' or 'linear', got '{method}'")
    
    try:
        from scipy.spatial import KDTree
        from scipy.interpolate import griddata
    except ImportError:
        raise ImportError("scipy is required. Install with: pip install scipy")
    
    print(f"Remapping MPAS → regular lat-lon grid ({resolution}°) using {method} interpolation")
    
    if isinstance(data, xr.DataArray):
        data_attrs = data.attrs
        data_values = data.values
    else:
        data_attrs = {}
        data_values = data
    
    lon_deg, lat_deg = _convert_coordinates_to_degrees(lon, lat)
    
    print("  Original MPAS data statistics [Global Statistics]:")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice encountered', RuntimeWarning)
        warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)
        warnings.filterwarnings('ignore', 'Degrees of freedom', RuntimeWarning)
        print(f"    Min: {float(np.nanmin(data_values)):.4f}, Max: {float(np.nanmax(data_values)):.4f}")
        print(f"    Mean: {float(np.nanmean(data_values)):.4f}, Median: {float(np.nanmedian(data_values)):.4f}")
        print(f"    Std: {float(np.nanstd(data_values)):.4f}, Sum: {float(np.nansum(data_values)):.4f}")
    
    target_lons = np.arange(lon_min, lon_max + resolution/2, resolution)
    target_lats = np.arange(lat_min, lat_max + resolution/2, resolution)
    
    n_target_points = len(target_lons) * len(target_lats)
    n_source_points = len(data_values)
    grid_expansion_ratio = n_target_points / n_source_points
    
    print(f"  Target grid: {len(target_lons)} x {len(target_lats)} = {n_target_points:,} points")
    print(f"  Grid expansion: {n_source_points:,} → {n_target_points:,} ({grid_expansion_ratio:.1f}x)")
    print(f"  Longitude: [{lon_min:.2f}, {lon_max:.2f}]° at {resolution}° spacing")
    print(f"  Latitude: [{lat_min:.2f}, {lat_max:.2f}]° at {resolution}° spacing")
    
    lon_2d, lat_2d = np.meshgrid(target_lons, target_lats)
    
    if method == 'nearest':
        source_points = np.column_stack([lon_deg, lat_deg])
        source_points, data_values = _add_wrapped_boundary_points(
            source_points, data_values, lon_deg, lat_deg, lon_min, lon_max
        )

        tree = KDTree(source_points)
        
        target_points = np.column_stack([lon_2d.flatten(), lat_2d.flatten()])
        distances, indices = tree.query(target_points)
        
        data_2d = data_values[indices].reshape(lon_2d.shape)
    
    else:  # method == 'linear'
        source_points = np.column_stack([lon_deg, lat_deg])
        target_points = np.column_stack([lon_2d.flatten(), lat_2d.flatten()])

        # fill_value=np.nan so points outside the convex hull of source points
        # are NaN, not 0 — callers must not confuse "no data" with a real zero.
        data_flat = griddata(source_points, data_values, target_points,
                             method='linear', fill_value=np.nan)
        data_2d = data_flat.reshape(lon_2d.shape)
    
    result = xr.DataArray(
        data_2d,
        dims=['lat', 'lon'],
        coords={
            'lon': target_lons,
            'lat': target_lats
        },
        attrs=data_attrs
    )
    
    data_sum = float(np.nansum(data_values))

    if np.abs(data_sum) < 1e-10:
        print("\n[ERROR] Input data for remapping is empty, all zeros, or all NaNs. Cannot compute remapping ratio.\n")
        print("  Please check your input data source and variable selection.")
        print(f"  Data shape: {data_values.shape}, dtype: {data_values.dtype}")

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN slice encountered', RuntimeWarning)
            print(f"  Data min: {np.nanmin(data_values)}, max: {np.nanmax(data_values)}")

        print(f"  Data contains only NaNs: {np.isnan(data_values).all()}")
        print(f"  Data contains only zeros: {np.count_nonzero(data_values)==0}")
        print("  Remapping aborted. Returning empty result array.")
        return result

    result_sum = float(result.sum(skipna=True))
    sum_ratio = result_sum / data_sum

    print("  Remapped data statistics [Statistics over Target Grid]:")
    print(f"    Min: {float(result.min(skipna=True)):.4f}, Max: {float(result.max(skipna=True)):.4f}")
    print(f"    Mean: {float(result.mean(skipna=True)):.4f}, Median: {float(np.nanmedian(result.values)):.4f}")
    print(f"    Std: {float(result.std(skipna=True)):.4f}, Sum: {result_sum:.4f}")

    if method == 'linear':
        print(f"  NOTE: Linear interpolation creates smooth fields but increases total sum by {sum_ratio:.1f}x")
        print(f"        This is expected when interpolating to a denser grid ({grid_expansion_ratio:.1f}x more points)")

    print(f"✓ Remapping completed successfully using {method} interpolation")

    return result


def build_remapped_valid_mask(lon_vals: np.ndarray, 
                              lat_vals: np.ndarray, 
                              lon_min: float, 
                              lon_max: float, 
                              lat_min: float, 
                              lat_max: float, 
                              resolution: float, 
                              remapped_data: Union[xr.DataArray, np.ndarray], 
                              threshold: float = 0.5) -> Optional[np.ndarray]:
    """
    This function builds a boolean mask for the remapped data based on the convex hull of the original MPAS cell coordinates. It first checks if the longitude range indicates global coverage, in which case it skips masking since all points are valid. For regional data, it constructs a convex hull around the original cell coordinates and uses matplotlib's Path to determine which points on the target grid fall inside this hull. The resulting boolean mask has the same shape as the remapped data and can be used to identify valid points that are within the original domain of the MPAS data. If any errors occur during the convex hull calculation or if required libraries are not available, it returns None, indicating that no mask could be created. This function is useful for ensuring that analyses or visualizations based on the remapped data only include points that are supported by the original MPAS grid coverage. 

    Parameters:
        lon_vals (np.ndarray): 1D array of longitude values for the original MPAS cell centers.
        lat_vals (np.ndarray): 1D array of latitude values for the original MPAS cell centers.
        lon_min (float): Minimum longitude for target grid in degrees.
        lon_max (float): Maximum longitude for target grid in degrees.
        lat_min (float): Minimum latitude for target grid in degrees.
        lat_max (float): Maximum latitude for target grid in degrees.
        resolution (float): Grid spacing in degrees for the target grid.
        remapped_data (Union[xr.DataArray, np.ndarray]): The remapped data array on the target grid, used to determine the shape of the mask.
        threshold (float): Optional threshold for determining valid points based on distance to hull vertices (not currently used, but can be implemented for more complex masking). 

    Returns:
        Optional[np.ndarray]: A boolean array with the same shape as remapped_data, where True indicates points inside the convex hull of the original MPAS coordinates, and False indicates points outside. Returns None if masking is skipped or if an error occurs. 
    """
    lon_range = lon_max - lon_min

    if lon_range > 180:
        print(f"  Skipping convex hull masking for global data (lon_range={lon_range:.1f}°)")
        return None
    
    source_coordinate_pairs = np.column_stack((lon_vals, lat_vals))

    try:
        from scipy.spatial import ConvexHull
        from matplotlib.path import Path

        hull = ConvexHull(source_coordinate_pairs)
        hull_pts = source_coordinate_pairs[hull.vertices]

        if isinstance(remapped_data, xr.DataArray):
            remapped_values = remapped_data.values
        else:
            remapped_values = np.array(remapped_data)

        if remapped_values.ndim == 2:
            n_lat, n_lon = remapped_values.shape
            lat_coord = np.linspace(lat_min, lat_max, n_lat)
            lon_coord = np.linspace(lon_min, lon_max, n_lon)
            grid_lon_2d, grid_lat_2d = np.meshgrid(lon_coord, lat_coord)
            grid_points = np.column_stack((grid_lon_2d.ravel(), grid_lat_2d.ravel()))
            hull_path = Path(hull_pts)
            inside = hull_path.contains_points(grid_points)
            mask_bool = inside.reshape((n_lat, n_lon))
            return mask_bool
    except ImportError:
        print("  Warning: scipy or matplotlib required for convex hull masking")
        return None
    except Exception as e:
        print(f"  Warning: Convex hull masking failed: {e}")
        return None


def _extract_cell_coordinates(dataset: xr.Dataset) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    This helper function extracts the longitude and latitude coordinates of the MPAS cell centers from the input dataset. It first checks for the presence of 'lonCell' and 'latCell' variables, which are commonly used in MPAS datasets to represent cell center coordinates. If these variables are found, it retrieves their values and checks if they are in radians (indicated by a maximum value less than or equal to 2π). If so, it converts them to degrees. If 'lonCell' and 'latCell' are not found, it looks for 'lon' and 'lat' variables as an alternative. If neither set of variables is found, it raises a ValueError indicating that the necessary coordinates could not be located in the dataset. This function ensures that the longitude and latitude coordinates are properly extracted and converted to degrees if necessary for subsequent remapping operations.

    Parameters:
        dataset (xr.Dataset): The input xarray Dataset containing the MPAS data and coordinates.
    
    Returns:
        Tuple[xr.DataArray, xr.DataArray]: A tuple containing the longitude and latitude coordinates of the MPAS cell centers as xarray DataArrays, with longitude converted to degrees if originally in radians.
    """
    if 'lonCell' in dataset:
        lon_coords = dataset['lonCell']
        lat_coords = dataset['latCell']
        if float(lon_coords.max()) <= 2 * np.pi:
            lon_coords = lon_coords * 180.0 / np.pi
            lat_coords = lat_coords * 180.0 / np.pi
        return lon_coords, lat_coords

    if 'lon' in dataset and 'lat' in dataset:
        return dataset['lon'], dataset['lat']

    raise ValueError("Could not find cell coordinates (lonCell/latCell or lon/lat) in dataset")


def _resolve_grid_bounds(dataset: xr.Dataset,
                         lon_min: Optional[float],
                         lon_max: Optional[float],
                         lat_min: Optional[float],
                         lat_max: Optional[float]) -> Tuple[float, float, float, float]:
    """
    This helper function resolves the grid bounds for the target latitude-longitude grid based on user input and the extent of the original MPAS coordinates. If the user has provided explicit bounds for longitude and latitude, it uses those directly. If any of the bounds are missing (i.e., None), it automatically determines the bounds from the original MPAS coordinates by extracting the longitude and latitude values and calculating their minimum and maximum with an optional buffer. This ensures that the target grid covers the appropriate spatial extent of the original data, while still allowing users to specify custom bounds if desired. The function returns a tuple containing the resolved grid bounds (lon_min, lon_max, lat_min, lat_max) that can be used for generating the target grid for remapping. 
    
    Parameters:
        dataset (xr.Dataset): The dataset containing the coordinates.
        lon_min (Optional[float]): Minimum longitude for target grid in degrees.
        lon_max (Optional[float]): Maximum longitude for target grid in degrees.
        lat_min (Optional[float]): Minimum latitude for target grid in degrees.
        lat_max (Optional[float]): Maximum latitude for target grid in degrees.

    Returns:
        Tuple[float, float, float, float]: Resolved grid bounds (lon_min, lon_max, lat_min, lat_max).
    """
    if lon_min is not None and lon_max is not None and lat_min is not None and lat_max is not None:
        return lon_min, lon_max, lat_min, lat_max

    from mpasdiag.processing.utils_geog import MPASGeographicUtils
    lon_np, lat_np = MPASGeographicUtils.extract_spatial_coordinates(dataset, normalize=False)

    auto_min_lon, auto_max_lon, auto_min_lat, auto_max_lat = \
        MPASGeographicUtils.get_extent_from_coordinates(lon_np, lat_np, buffer=0.0)

    return (
        lon_min if lon_min is not None else auto_min_lon,
        lon_max if lon_max is not None else auto_max_lon,
        lat_min if lat_min is not None else auto_min_lat,
        lat_max if lat_max is not None else auto_max_lat,
    )


def _apply_lon_convention(lon_coords: xr.DataArray,
                          lon_data_range: float,
                          lon_min: float,
                          lon_max: float,
                          lon_convention: str) -> xr.DataArray:
    """
    This helper function applies the specified longitude convention to the longitude coordinates of the original MPAS cell centers. It supports three conventions: 'auto', '[-180,180]', and '[0,360]'. If 'auto' is selected, it detects whether the data appears to be global or regional based on the longitude range and the min/max values, and preserves the original convention for global/wide-span data while converting to a consistent convention for regional data. For regional data, it converts longitudes to the specified convention if the longitude range is less than or equal to 180 degrees. This ensures that the longitude coordinates are in a consistent format that matches the target grid and allows for proper remapping and masking operations. The function returns the adjusted longitude coordinates as an xarray DataArray.

    Parameters:
        lon_coords (xr.DataArray): Original longitude coordinates of the MPAS cell centers.
        lon_data_range (float): The range of longitude values in the original data, used for auto-detection of global vs regional data.
        lon_min (float): Minimum longitude for target grid in degrees, used for auto-detection of global vs regional data.
        lon_max (float): Maximum longitude for target grid in degrees, used for auto-detection of global vs regional data.
        lon_convention (str): The longitude convention to apply, options are 'auto', '[-180,180]', and '[0,360]'.

    Returns:
        xr.DataArray: Adjusted longitude coordinates of the MPAS cell centers according to the specified longitude convention, returned as an xarray DataArray.
    """
    if lon_convention == 'auto':
        if lon_data_range > 180 or (lon_min >= 0 and lon_max > 180):
            print(f"  Detected global/wide-span data (range={lon_data_range:.1f}°), "
                  "preserving original longitude convention")
            return lon_coords
        lon_convention = '[-180,180]' if (lon_max <= 180 and lon_min >= -180) else '[0,360]'

    if lon_convention == '[-180,180]' and lon_data_range <= 180:
        return xr.where(lon_coords > 180, lon_coords - 360, lon_coords)

    if lon_convention == '[0,360]' and lon_data_range <= 180:
        return xr.where(lon_coords < 0, lon_coords + 360, lon_coords)

    return lon_coords


def _apply_remap_mask(remapped_data: xr.DataArray,
                      lon_vals: np.ndarray,
                      lat_vals: np.ndarray,
                      lon_min: float,
                      lon_max: float,
                      lat_min: float,
                      lat_max: float,
                      resolution: float) -> xr.DataArray:
    """
    This helper function applies the boolean mask generated by build_remapped_valid_mask to the remapped data array, setting points outside the convex hull of the original MPAS coordinates to NaN. It first calls the mask-building function to get the boolean mask, and if a valid mask is returned, it uses np.where to set values in the remapped data to NaN where the mask is False. The resulting masked remapped data is returned as a new xarray DataArray with the same coordinates and attributes as the input remapped_data. If no valid mask could be created (e.g., for global data or if an error occurred), it simply returns the original remapped_data without modification. This function ensures that analyses or visualizations based on the remapped data only include points that are supported by the original MPAS grid coverage.

    Parameters:
        remapped_data (xr.DataArray): The remapped data array on the target grid to which the mask will be applied.
        lon_vals (np.ndarray): 1D array of longitude values for the original MPAS cell centers.
        lat_vals (np.ndarray): 1D array of latitude values for the original MPAS cell centers.
        lon_min (float): Minimum longitude for target grid in degrees.
        lon_max (float): Maximum longitude for target grid in degrees.
        lat_min (float): Minimum latitude for target grid in degrees.
        lat_max (float): Maximum latitude for target grid in degrees.
        resolution (float): Grid spacing in degrees for the target grid, used for logging purposes in the mask-building function.

    Returns:
        xr.DataArray: A new xarray DataArray containing the remapped data with points outside the convex hull of the original MPAS coordinates set to NaN, and with the same coordinates and attributes as the input remapped_data. 
    """
    mask_bool = build_remapped_valid_mask(
        lon_vals, lat_vals, lon_min, lon_max, lat_min, lat_max, resolution, remapped_data
    )

    if mask_bool is None:
        return remapped_data

    return xr.DataArray(
        np.where(mask_bool, remapped_data.values, np.nan),
        coords=remapped_data.coords,
        dims=remapped_data.dims,
        attrs=remapped_data.attrs,
    )


def remap_mpas_to_latlon_with_masking(data: Union[xr.DataArray, np.ndarray],
                                      dataset: xr.Dataset,
                                      lon_min: Optional[float] = None,
                                      lon_max: Optional[float] = None,
                                      lat_min: Optional[float] = None,
                                      lat_max: Optional[float] = None,
                                      resolution: float = 0.1,
                                      method: str = 'nearest',
                                      apply_mask: bool = True,
                                      lon_convention: str = 'auto',
                                      config: Optional[Any] = None,
                                      comm: Optional[Any] = None) -> xr.DataArray:
    """
    This function provides a convenient interface for remapping MPAS unstructured grid data to a
    regular latitude-longitude grid. The remapping engine and method are controlled by the optional
    ``config`` argument (an MPASConfig instance or any object with ``remap_engine`` /
    ``remap_method`` attributes):

    * ``config.remap_engine == 'esmf'``  — delegates to the ESMPy path via ``dispatch_remap``.
      ESMPy natively produces NaN for unmapped target cells, so the convex-hull masking step is
      skipped (``apply_mask`` is ignored for this path).
    * ``config.remap_engine == 'kdtree'`` (default when config is None) — uses scipy KDTree
      interpolation, then optionally applies a convex-hull mask to set out-of-domain cells to NaN.

    Parameters:
        data (Union[xr.DataArray, np.ndarray]): Input data on MPAS unstructured grid, shape (nCells,).
        dataset (xr.Dataset): Dataset containing MPAS coordinates (lonCell / latCell).
        lon_min (Optional[float]): Western bound of target grid in degrees (auto-derived if None).
        lon_max (Optional[float]): Eastern bound of target grid in degrees (auto-derived if None).
        lat_min (Optional[float]): Southern bound of target grid in degrees (auto-derived if None).
        lat_max (Optional[float]): Northern bound of target grid in degrees (auto-derived if None).
        resolution (float): Target grid spacing in degrees (default: 0.1).
        method (str): KDTree interpolation method — 'nearest' or 'linear' (default: 'nearest').
            Ignored when config specifies remap_engine='esmf'; use config.remap_method instead.
        apply_mask (bool): Apply convex-hull masking for the KDTree path (default: True).
            Ignored for the ESMPy path.
        lon_convention (str): Longitude convention — 'auto', '[-180,180]', or '[0,360]' (default: 'auto').
        config (Optional[Any]): MPASConfig or SimpleNamespace with remap_engine / remap_method fields.
            When None, the KDTree path is used with the ``method`` argument (default: None).
        comm (Optional[Any]): MPI communicator forwarded to the ESMPy path (default: None).

    Returns:
        xr.DataArray: Remapped data on a regular lat/lon grid with NaN for out-of-domain cells.
    """
    # --- ESMPy path: delegate entirely to dispatch_remap -----------------------
    # dispatch_remap is defined later in this module; Python resolves the name at
    # call time so there is no forward-reference problem.
    if getattr(config, 'remap_engine', 'kdtree') == 'esmf':
        return dispatch_remap(
            data=data,
            dataset=dataset,
            config=config,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            resolution=resolution,
            apply_mask=apply_mask,
            lon_convention=lon_convention,
            comm=comm,
        )

    # --- KDTree path -----------------------------------------------------------
    # Honour config.remap_method when provided so that 'nearest' / 'linear' can
    # be driven from config rather than the positional method argument.
    kdtree_method = getattr(config, 'remap_method', method) if config else method

    lon_coords, lat_coords = _extract_cell_coordinates(dataset)

    lon_min, lon_max, lat_min, lat_max = _resolve_grid_bounds(
        dataset, lon_min, lon_max, lat_min, lat_max
    )

    lon_data_range = float(lon_coords.max() - lon_coords.min())
    lon_coords = _apply_lon_convention(lon_coords, lon_data_range, lon_min, lon_max, lon_convention)

    data_attrs = data.attrs if isinstance(data, xr.DataArray) else {}
    data_values = data.values if isinstance(data, xr.DataArray) else data

    remapped_data = remap_mpas_to_latlon(
        data=data_values,
        lon=lon_coords.values,
        lat=lat_coords.values,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        resolution=resolution,
        method=kdtree_method,
    )

    if apply_mask:
        remapped_data = _apply_remap_mask(
            remapped_data, lon_coords.values, lat_coords.values,
            lon_min, lon_max, lat_min, lat_max, resolution
        )

    remapped_data.attrs.update(data_attrs)

    return remapped_data


def dispatch_remap(data: Union[xr.DataArray, np.ndarray],
                   dataset: xr.Dataset,
                   config: Any,
                   lon_min: Optional[float] = None,
                   lon_max: Optional[float] = None,
                   lat_min: Optional[float] = None,
                   lat_max: Optional[float] = None,
                   resolution: float = 0.1,
                   apply_mask: bool = True,
                   lon_convention: str = 'auto',
                   comm: Optional[Any] = None,) -> xr.DataArray:
    """
    This function serves as a dispatcher for remapping MPAS unstructured grid data to a regular latitude-longitude grid, allowing users to choose between a KDTree-based nearest neighbor interpolation method and an ESMF-based regridding method based on the configuration provided. It checks the 'remap_engine' field in the config to determine which remapping approach to use. If 'kdtree' is selected, it calls the remap_mpas_to_latlon_with_masking function to perform the remapping with optional masking. If 'esmf' is selected, it checks for the availability of ESMPy and then sets up an MPASRemapper instance to perform the remapping using ESMF's regridding capabilities. The function also handles the resolution of grid bounds and longitude conventions as needed. The resulting remapped data is returned as an xarray DataArray on the regular lat/lon target grid. This dispatcher provides flexibility for users to choose their preferred remapping method while maintaining a consistent interface for input parameters and output results. 

    Parameters:
        data: Input data on MPAS unstructured grid, shape (nCells,).
        dataset: xarray Dataset containing MPAS coordinates.
        config: MPASConfig instance with remap_engine and remap_method fields.
        lon_min/lon_max/lat_min/lat_max: Target grid bounds in degrees; auto-derived when None.
        resolution: Target grid spacing in degrees.
        apply_mask: Apply convex-hull masking after remapping (kdtree path only).
        lon_convention: Longitude convention ('auto', '[-180,180]', '[0,360]').
        comm: MPI communicator for the esmf path.

    Returns:
        xr.DataArray on the regular lat/lon target grid.
    """
    engine = getattr(config, 'remap_engine', 'kdtree')
    method = getattr(config, 'remap_method', 'nearest')

    if engine == 'kdtree':
        return remap_mpas_to_latlon_with_masking(
            data=data,
            dataset=dataset,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            resolution=resolution,
            method=method,
            apply_mask=apply_mask,
            lon_convention=lon_convention,
        )

    if engine == 'esmf':
        if not ESMPY_AVAILABLE:
            raise ImportError(
                "remap_engine='esmf' requires ESMPy. "
                "Install with: conda install -c conda-forge esmpy"
            )

        lon_coords, lat_coords = _extract_cell_coordinates(dataset)
        lon_b = dataset.get('lon_b', None)
        lat_b = dataset.get('lat_b', None)

        remapper = MPASRemapper(method=method, skipna=True)
        
        remapper.prepare_source_grid(
            lon_coords.values,
            lat_coords.values,
            lon_bounds=lon_b.values if lon_b is not None else None,
            lat_bounds=lat_b.values if lat_b is not None else None,
        )

        lon_min_r, lon_max_r, lat_min_r, lat_max_r = _resolve_grid_bounds(
            dataset, lon_min, lon_max, lat_min, lat_max
        )

        remapper.create_target_grid(
            lon_min=lon_min_r,
            lon_max=lon_max_r,
            lat_min=lat_min_r,
            lat_max=lat_max_r,
            dlon=resolution,
            dlat=resolution,
        )

        remapper.build_regridder(comm=comm)

        data_xr = (data if isinstance(data, xr.DataArray)
                   else xr.DataArray(data, dims=['nCells']))
        
        return remapper.remap(data_xr)

    raise ValueError(
        f"Unknown remap_engine '{engine}'. Must be 'kdtree' or 'esmf'."
    )


def create_target_grid(lon_min: float = -180.0,
                       lon_max: float = 180.0, 
                       lat_min: float = -90.0, 
                       lat_max: float = 90.0, 
                       dlon: float = 1.0, 
                       dlat: float = 1.0) -> xr.Dataset:
    """
    This function creates a target grid specification as an xarray Dataset with 1D coordinate arrays for longitude and latitude based on user-defined spatial boundaries and grid spacing. The longitude and latitude values are generated using numpy's arange function, ensuring that the grid points are centered within the specified bounds. The resulting Dataset contains 'lon' and 'lat' coordinates with dimensions ['lon'] and ['lat'], which can be used as the target grid specification for remapping operations. This function provides a simple way to generate regular lat-lon grids of varying resolutions for use in remapping MPAS data or other geospatial datasets. 
    
    Parameters:
        lon_min (float): Minimum longitude for target grid in degrees (default: -180.0).
        lon_max (float): Maximum longitude for target grid in degrees (default: 180.0).
        lat_min (float): Minimum latitude for target grid in degrees (default: -90.0).
        lat_max (float): Maximum latitude for target grid in degrees (default: 90.0).
        dlon (float): Grid spacing in degrees for longitude (default: 1.0).
        dlat (float): Grid spacing in degrees for latitude (default: 1.0). 
    
    Returns:
        xr.Dataset: An xarray Dataset containing 1D coordinate arrays for 'lon' and 'lat' that define the target grid specification for remapping. The 'lon' coordinate has dimension ['lon'] and the 'lat' coordinate has dimension ['lat']. 
    """
    lon = np.arange(lon_min, lon_max + dlon/2, dlon)
    lat = np.arange(lat_min, lat_max + dlat/2, dlat)
    
    return xr.Dataset({
        'lon': xr.DataArray(lon, dims=['lon']),
        'lat': xr.DataArray(lat, dims=['lat'])
    })


if __name__ == '__main__':
    print("MPAS Remapping Module")
    print("=" * 50)
    
    if not ESMPY_AVAILABLE:
        print("\nESMPy is not installed. Install with:")
        print("  conda install -c conda-forge esmpy")
    else:
        print("\nESMPy is available")
        print("Supported methods: conservative, conservative_normed, nearest_s2d, nearest_d2s")
        print("\nExample usage:")
        print("""
from mpasdiag.processing.remapping import MPASRemapper, remap_mpas_to_latlon

# Method 1: High-level convenience function
remapped = remap_mpas_to_latlon(
    data=temperature,
    lon=mpas_lon,
    lat=mpas_lat,
    resolution=0.25,
    method='nearest_s2d'
)

# Method 2: Full control with MPASRemapper
remapper = MPASRemapper(method='conservative', weights_dir='./weights')
remapper.prepare_source_grid(mpas_lon, mpas_lat)
remapper.create_target_grid(resolution=0.5)
remapper.build_regridder()
remapped = remapper.remap(data)
        """)

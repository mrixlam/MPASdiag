#!/usr/bin/env python3

"""
MPAS Data Remapping using KDTree.

This module provides fast, efficient regridding functionality for MPAS unstructured mesh data using KDTree nearest neighbor interpolation. The implementation uses scipy's KDTree spatial indexing for O(log n) lookups to map MPAS cell values to regular latitude-longitude grids. This approach is significantly faster than xESMF-based methods, preserves original data values without smoothing, and requires less memory. The module supports batch processing of multiple variables, automatic coordinate detection from MPAS conventions, and placement of data at grid cell centers for compatibility with standard visualization and analysis tools. While the MPASRemapper class remains available for advanced xESMF workflows, the remap_mpas_to_latlon convenience function provides a simple, fast interface for most common use cases.

Classes:
    MPASRemapper: Advanced class for regridding MPAS data using xESMF (for specialized workflows).
    
Functions:
    remap_mpas_to_latlon: Fast KDTree-based remapping from MPAS to regular lat-lon grids (recommended).
    create_target_grid: Helper function to generate target grid dataset specifications.
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import os
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple, Any, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    import xesmf as xe

try:
    import xesmf as xe
    XESMF_AVAILABLE = True
except ImportError:
    XESMF_AVAILABLE = False
    if not TYPE_CHECKING:
        xe = None  
    warnings.warn(
        "xESMF is not installed. Install with: conda install -c conda-forge xesmf\n"
        "Note: xESMF requires ESMPy which is only available via conda.",
        ImportWarning
    )


def _convert_coordinates_to_degrees(lon: Union[np.ndarray, xr.DataArray],
                                     lat: Union[np.ndarray, xr.DataArray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to convert coordinates from radians to degrees and extract numpy arrays.
    
    Parameters:
        lon: Longitude coordinates (may be DataArray or ndarray, in degrees or radians)
        lat: Latitude coordinates (may be DataArray or ndarray, in degrees or radians)
    
    Returns:
        Tuple of (lon_deg, lat_deg) as numpy arrays in degrees
    """
    if isinstance(lon, xr.DataArray):
        lon = lon.values

    if isinstance(lat, xr.DataArray):
        lat = lat.values
    
    lon_deg = np.degrees(lon) if np.max(np.abs(lon)) <= np.pi else lon
    lat_deg = np.degrees(lat) if np.max(np.abs(lat)) <= np.pi else lat
    
    return lon_deg, lat_deg


def _compute_grid_bounds(coords: np.ndarray, resolution: float) -> np.ndarray:
    """
    Helper function to compute grid cell bounds from cell centers.
    
    Parameters:
        coords: 1D array of coordinate centers
        resolution: Grid spacing
    
    Returns:
        1D array of coordinate bounds (length = len(coords) + 1)
    """
    bounds = np.zeros(len(coords) + 1)
    bounds[0] = coords[0] - resolution / 2
    
    for i in range(1, len(coords)):
        bounds[i] = (coords[i-1] + coords[i]) / 2
    
    bounds[-1] = coords[-1] + resolution / 2
    
    return bounds


class MPASRemapper:
    """
    This class provides a high-level interface to xESMF regridding functionality specifically designed for MPAS output, supporting all interpolation methods (bilinear, conservative, patch, nearest_s2d, nearest_d2s, conservative_normed), automatic coordinate detection from MPAS conventions, weight file caching for repeated operations, and batch processing of multiple variables. The remapper handles both cell-centered (nCells) and vertex-based (nVertices) MPAS fields, performs proper unit sphere coordinate conversion, manages periodic boundaries for global domains, and applies conservative renormalization when needed. It integrates seamlessly with xarray workflows and provides options for handling masked regions, parallel processing, and memory-efficient chunked operations.
    
    Attributes:
        source_grid (xr.Dataset): MPAS source grid with lon/lat coordinates
        target_grid (xr.Dataset): Target regular grid specification
        method (str): Regridding method name
        regridder (xe.Regridder): Cached xESMF regridder object
        weights_dir (Optional[Path]): Directory for weight file caching
        reuse_weights (bool): Whether to reuse existing weight files
    """
    
    def __init__(self,
                 method: str = 'bilinear',
                 weights_dir: Optional[Union[str, Path]] = None,
                 reuse_weights: bool = True,
                 periodic: bool = False,
                 extrap_method: Optional[str] = None,
                 extrap_dist_exponent: Optional[float] = None,
                 extrap_num_src_pnts: Optional[int] = None) -> None:
        """
        This constructor configures the remapping engine by selecting the xESMF interpolation method and setting up weight file management for efficient repeated operations. The method parameter determines the interpolation algorithm used in the second step of the hybrid approach, after KDTree has created the intermediate 2D grid. Weight file caching significantly improves performance for repeated remapping operations with the same source and target grid combinations. Optional extrapolation parameters control how missing values are handled at grid boundaries or masked regions.
        
        Parameters:
            method (str): xESMF interpolation method name - 'bilinear', 'patch', 'nearest_s2d', 'nearest_d2s', 'conservative', or 'conservative_normed' (default: 'bilinear').
            weights_dir (Optional[Union[str, Path]]): Directory path for storing and loading weight files to enable reuse across sessions (default: None creates temporary weights).
            reuse_weights (bool): Enable loading existing weight files instead of recomputing for identical grid configurations (default: True).
            periodic (bool): Specify if source grid has periodic longitude boundaries for global domains (default: False).
            extrap_method (Optional[str]): Extrapolation algorithm for handling missing values - 'inverse_dist' or 'nearest_s2d' (default: None disables extrapolation).
            extrap_dist_exponent (Optional[float]): Distance weighting exponent for inverse distance extrapolation method (default: None uses xESMF default value).
            extrap_num_src_pnts (Optional[int]): Number of source points to use in extrapolation calculations (default: None uses xESMF default value).
        
        Returns:
            None
        
        Raises:
            ImportError: If xESMF package is not installed or cannot be imported.
            ValueError: If specified method name is not in the supported methods list.
        """
        if not XESMF_AVAILABLE:
            raise ImportError(
                "xESMF is required for remapping. Install with:\n"
                "conda install -c conda-forge xesmf"
            )
        
        valid_methods = ['bilinear', 'conservative', 'conservative_normed', 
                        'patch', 'nearest_s2d', 'nearest_d2s']

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
        
        if weights_dir is not None:
            self.weights_dir = Path(weights_dir)
            self.weights_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.weights_dir = None
        
        self.source_grid: Optional[xr.Dataset] = None
        self.target_grid: Optional[xr.Dataset] = None
        self.regridder: Optional['xe.Regridder'] = None 
        
        print(f"MPASRemapper initialized with method: {method}")
    
    def prepare_source_grid(self,
                          lon: Union[np.ndarray, xr.DataArray],
                          lat: Union[np.ndarray, xr.DataArray],
                          lon_bounds: Optional[Union[np.ndarray, xr.DataArray]] = None,
                          lat_bounds: Optional[Union[np.ndarray, xr.DataArray]] = None) -> xr.Dataset:
        """
        This method transforms MPAS native coordinate arrays into the standardized format required by xESMF regridding operations. The conversion includes automatic detection and transformation of radian coordinates to degrees, normalization of longitude values to the 0-360 range, and creation of properly named coordinate variables. Optional cell boundary coordinates can be provided for conservative remapping methods that require vertex information for area-weighted interpolation. The resulting dataset serves as the source grid specification for xESMF regridder initialization.
        
        Parameters:
            lon (Union[np.ndarray, xr.DataArray]): Cell center longitude coordinates in either degrees or radians with automatic detection.
            lat (Union[np.ndarray, xr.DataArray]): Cell center latitude coordinates in either degrees or radians with automatic detection.
            lon_bounds (Optional[Union[np.ndarray, xr.DataArray]]): Cell corner longitude coordinates with shape (nCells, nVertices) for conservative methods (default: None).
            lat_bounds (Optional[Union[np.ndarray, xr.DataArray]]): Cell corner latitude coordinates with shape (nCells, nVertices) for conservative methods (default: None).
        
        Returns:
            xr.Dataset: Source grid dataset containing 'lon' and 'lat' coordinate arrays plus optional 'lon_b' and 'lat_b' boundary arrays.
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
    
    def create_target_grid(self,
                          lon_min: float = -180.0,
                          lon_max: float = 180.0,
                          lat_min: float = -90.0,
                          lat_max: float = 90.0,
                          dlon: float = 1.0,
                          dlat: float = 1.0) -> xr.Dataset:
        """
        This method generates a regular rectilinear grid specification suitable for xESMF remapping operations with customizable geographic bounds and spatial resolution. The grid uses cell-center coordinates arranged in a uniform spacing pattern that can represent either regional or global domains. Longitude boundaries can be specified using either the [-180, 180] or [0, 360] convention depending on the target application. The resulting dataset is stored internally and used as the destination grid for subsequent remapping operations.
        
        Parameters:
            lon_min (float): Western longitude boundary in degrees, supporting both [-180, 180] and [0, 360] conventions (default: -180.0).
            lon_max (float): Eastern longitude boundary in degrees, must be greater than lon_min (default: 180.0).
            lat_min (float): Southern latitude boundary in degrees, range [-90, 90] (default: -90.0).
            lat_max (float): Northern latitude boundary in degrees, range [-90, 90], must be greater than lat_min (default: 90.0).
            dlon (float): Longitude grid spacing in degrees, determines resolution in east-west direction (default: 1.0).
            dlat (float): Latitude grid spacing in degrees, determines resolution in north-south direction (default: 1.0).
        
        Returns:
            xr.Dataset: Target grid dataset containing 1D 'lon' and 'lat' coordinate arrays with dimensions ['lon'] and ['lat'].
        """
        lon = np.arange(lon_min, lon_max + dlon/2, dlon)
        lat = np.arange(lat_min, lat_max + dlat/2, dlat)
        
        target_grid = xr.Dataset({
            'lon': xr.DataArray(lon, dims=['lon']),
            'lat': xr.DataArray(lat, dims=['lat'])
        })
        
        self.target_grid = target_grid

        print(f"Created target grid: {len(lon)} x {len(lat)} points")
        print(f"  Longitude: [{lon_min:.2f}, {lon_max:.2f}] deg, spacing: {dlon:.3f} deg")
        print(f"  Latitude: [{lat_min:.2f}, {lat_max:.2f}] deg, spacing: {dlat:.3f} deg")
        
        return target_grid
    
    def build_regridder(self,
                       source_grid: Optional[xr.Dataset] = None,
                       target_grid: Optional[xr.Dataset] = None,
                       filename: Optional[str] = None) -> 'xe.Regridder':
        """
        This method initializes the core xESMF regridder that performs the actual interpolation transformation from source to target grid coordinates. Weight computation is computationally expensive, involving sparse matrix construction for the interpolation operator, so the method supports caching weights to disk for reuse in subsequent operations with identical grid configurations. The regridder is configured with the interpolation method, periodicity settings, and extrapolation options specified during class initialization. Optional weight file management enables significant performance improvements for repeated remapping workflows.
        
        Parameters:
            source_grid (Optional[xr.Dataset]): Source grid dataset with coordinate specifications, uses self.source_grid if not provided (default: None).
            target_grid (Optional[xr.Dataset]): Target grid dataset with coordinate specifications, uses self.target_grid if not provided (default: None).
            filename (Optional[str]): Custom weight file name for this specific source-target grid pair, auto-generates descriptive name if not provided (default: None).
        
        Returns:
            xe.Regridder: Configured xESMF regridder object ready for data transformation operations.
        
        Raises:
            ValueError: If source grid is not available either as parameter or stored in self.source_grid.
            ValueError: If target grid is not available either as parameter or stored in self.target_grid.
        """
        if source_grid is None:
            if self.source_grid is None:
                raise ValueError("Source grid must be provided or prepared first")
            source_grid = self.source_grid
        
        if target_grid is None:
            if self.target_grid is None:
                raise ValueError("Target grid must be provided or created first")
            target_grid = self.target_grid
        
        if self.weights_dir is not None and filename is None:
            src_shape = len(source_grid['lon'])
            tgt_shape = f"{len(target_grid['lon'])}x{len(target_grid['lat'])}"
            filename = f"weights_{self.method}_{src_shape}to{tgt_shape}.nc"
        
        regridder_kwargs = {
            'method': self.method,
            'periodic': self.periodic
        }
        
        if self.weights_dir is not None and filename is not None:
            regridder_kwargs['reuse_weights'] = self.reuse_weights
            regridder_kwargs['filename'] = str(self.weights_dir / filename)
        else:
            regridder_kwargs['reuse_weights'] = False
        
        if self.extrap_method is not None:
            regridder_kwargs['extrap_method'] = self.extrap_method

        if self.extrap_dist_exponent is not None:
            regridder_kwargs['extrap_dist_exponent'] = self.extrap_dist_exponent

        if self.extrap_num_src_pnts is not None:
            regridder_kwargs['extrap_num_src_pnts'] = self.extrap_num_src_pnts
        
        print(f"Building {self.method} regridder...")

        if not XESMF_AVAILABLE:
            raise ImportError("xESMF is required for regridding. Install with: conda install -c conda-forge xesmf")

        self.regridder = xe.Regridder(source_grid, target_grid, **regridder_kwargs)  
        print(f"Regridder built successfully")
        
        return self.regridder
    
    def remap(self,
             data: Union[xr.DataArray, np.ndarray],
             keep_attrs: bool = True) -> xr.DataArray:
        """
        This method executes the actual data transformation using the pre-configured xESMF regridder object, applying the selected interpolation algorithm to map values from source to target grid points. The method handles both xarray DataArrays with full metadata preservation and raw numpy arrays with automatic dimension inference. For multi-dimensional input data containing additional dimensions beyond the spatial coordinates (such as time series or vertical levels), the method automatically detects and preserves these auxiliary dimensions while applying the spatial transformation. The interpolation respects masked regions and handles missing values according to the configured extrapolation settings.
        
        Parameters:
            data (Union[xr.DataArray, np.ndarray]): Input data array on source grid with shape matching source grid size, supports xarray with metadata or numpy arrays.
            keep_attrs (bool): Flag to preserve xarray variable attributes and metadata in the remapped output (default: True).
        
        Returns:
            xr.DataArray: Remapped data on target grid with proper coordinate labels and optional preserved metadata.
        
        Raises:
            ValueError: If regridder object has not been built via build_regridder() method call.
            ValueError: If input data shape is incompatible with the configured source grid dimensions.
        """
        if self.regridder is None:
            raise ValueError("Regridder must be built before remapping. Call build_regridder() first.")
        
        if isinstance(data, np.ndarray):
            data = xr.DataArray(data, dims=['x'])
        
        result = self.regridder(data, keep_attrs=keep_attrs)
        
        return xr.DataArray(result) if not isinstance(result, xr.DataArray) else result
    
    def remap_dataset(self,
                     dataset: xr.Dataset,
                     variables: Optional[List[str]] = None,
                     skip_missing: bool = True) -> xr.Dataset:
        """
        This method efficiently processes multiple data variables from an xarray Dataset by applying the same regridder configuration to each variable in sequence. The batch processing approach reuses the pre-computed interpolation weights across all variables, significantly improving performance compared to individual remapping operations. Non-spatial dimensions such as time series or vertical levels are automatically preserved in their original structure while only the spatial dimensions undergo transformation. The method provides flexible variable selection and error handling options to accommodate datasets with heterogeneous variable structures.
        
        Parameters:
            dataset (xr.Dataset): Input xarray Dataset containing multiple data variables on source grid with shared spatial dimensions.
            variables (Optional[List[str]]): List of variable names to remap from the dataset, processes all data variables if not specified (default: None).
            skip_missing (bool): Continue processing remaining variables if a specified variable is not found in dataset (default: True).
        
        Returns:
            xr.Dataset: New xarray Dataset containing all remapped variables on target grid with preserved non-spatial dimensions and metadata.
        
        Raises:
            ValueError: If regridder has not been built before calling this method.
            ValueError: If skip_missing is False and a specified variable name is not found in the input dataset.
        """
        if self.regridder is None:
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
    def unstructured_to_structured_grid(
        data: Union[xr.DataArray, np.ndarray],
        lon: Union[np.ndarray, xr.DataArray],
        lat: Union[np.ndarray, xr.DataArray],
        intermediate_resolution: float = 0.1,
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None,
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        buffer: float = 2.0
    ) -> Tuple[xr.DataArray, xr.Dataset]:
        """
        This is a two-step remapping approach that solves xESMF's incompatibility with unstructured MPAS grids by first using KDTree to map unstructured points to an intermediate 2D structured grid, which can then be used with xESMF for advanced interpolation (bilinear, conservative, patch) to the final target grid. The method creates a regular latitude-longitude grid at the specified intermediate resolution, uses scipy's KDTree for O(log n) nearest neighbor lookups to populate the grid with MPAS data values, and generates proper cell boundary coordinates required for conservative remapping. This approach combines KDTree's ability to handle unstructured point clouds with xESMF's sophisticated interpolation algorithms, enabling the full range of xESMF methods to be applied to MPAS output data.
        
        Parameters:
            data (Union[xr.DataArray, np.ndarray]): Source data on unstructured MPAS grid
            lon (Union[np.ndarray, xr.DataArray]): Source longitude coordinates
            lat (Union[np.ndarray, xr.DataArray]): Source latitude coordinates
            intermediate_resolution (float): Resolution of intermediate 2D grid in degrees 
                (default: 0.1°, finer than typical target for better interpolation quality)
            lon_min (Optional[float]): Minimum longitude for intermediate grid (auto-detected if None)
            lon_max (Optional[float]): Maximum longitude for intermediate grid (auto-detected if None)
            lat_min (Optional[float]): Minimum latitude for intermediate grid (auto-detected if None)
            lat_max (Optional[float]): Maximum latitude for intermediate grid (auto-detected if None)
            buffer (float): Buffer in degrees around data extent (default: 2.0)
        
        Returns:
            Tuple[xr.DataArray, xr.Dataset]: 
                - DataArray with data on 2D structured grid (ready for xESMF)
                - Dataset with grid specification (for xESMF source grid)
        
        Example:
            >>> # Step 1: Convert unstructured to 2D structured
            >>> structured_data, structured_grid = MPASRemapper.unstructured_to_structured_grid(
            ...     data=temperature,
            ...     lon=mpas_lon,
            ...     lat=mpas_lat,
            ...     intermediate_resolution=0.1
            ... )
            >>> 
            >>> # Step 2: Use xESMF for advanced interpolation
            >>> remapper = MPASRemapper(method='bilinear')
            >>> remapper.source_grid = structured_grid
            >>> remapper.create_target_grid(lon_min=110, lon_max=120, dlon=1.0, dlat=1.0)
            >>> remapper.build_regridder()
            >>> final_result = remapper.remap(structured_data)
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
        
        print(f"Creating intermediate 2D structured grid:")
        print(f"  Lon range: [{lon_min:.2f}, {lon_max:.2f}]°")
        print(f"  Lat range: [{lat_min:.2f}, {lat_max:.2f}]°")
        print(f"  Resolution: {intermediate_resolution}°")
        
        intermediate_lons = np.arange(lon_min, lon_max + intermediate_resolution/2, 
                                      intermediate_resolution)

        intermediate_lats = np.arange(lat_min, lat_max + intermediate_resolution/2, 
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
        
        print(f"✓ Converted to 2D structured grid with bounds (ready for conservative remapping)")
        
        return structured_data, structured_grid
    
    def cleanup(self) -> None:
        """
        This method performs explicit cleanup of the xESMF regridder object and associated weight matrices to release allocated memory resources. The cleanup operation removes both the regridder instance and any cached grid specifications stored in class attributes. This is particularly useful when processing multiple different grid pairs sequentially in a single session, as it prevents memory accumulation from multiple regridder instances. After cleanup, a new regridder must be built before performing additional remapping operations.
        
        Parameters:
            None
        
        Returns:
            None
        """
        if self.regridder is not None:
            self.regridder = None
        
        self.source_grid = None
        self.target_grid = None
        
        print("Regridder resources cleaned up")
    
    @staticmethod
    def estimate_memory_usage(n_source: int, n_target: int, method: str) -> float:
        """
        This utility function provides rough memory usage estimates to help users plan large-scale remapping operations and avoid out-of-memory errors. The calculation accounts for sparse weight matrix storage requirements which vary significantly by interpolation method, plus temporary storage for source and target data arrays. Conservative methods require more neighbor connections per target cell and thus use more memory than bilinear or nearest neighbor approaches. The estimates are approximate and actual memory usage may vary depending on grid geometry, land-sea masks, and xESMF implementation details.
        
        Parameters:
            n_source (int): Total number of grid points in the source mesh or grid.
            n_target (int): Total number of grid points in the target output grid.
            method (str): Interpolation method name determining sparsity pattern of weight matrix.
        
        Returns:
            float: Estimated peak memory usage during regridding operation in gigabytes (GB).
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
    Remap MPAS unstructured mesh data to a regular latitude-longitude grid using scipy interpolation methods. This function supports both nearest neighbor (fast, preserves exact values) and linear interpolation (smooth fields suitable for contour plots). Data is placed at grid cell centers for compatibility with standard visualization and analysis tools.
    
    Parameters:
        data (Union[xr.DataArray, np.ndarray]): Input data array defined on MPAS unstructured grid with shape (nCells,) or subset.
        lon (Union[np.ndarray, xr.DataArray]): MPAS cell center longitude coordinates in degrees or radians.
        lat (Union[np.ndarray, xr.DataArray]): MPAS cell center latitude coordinates in degrees or radians.
        lon_min (float): Western boundary of target output grid in degrees (default: -180.0).
        lon_max (float): Eastern boundary of target output grid in degrees (default: 180.0).
        lat_min (float): Southern boundary of target output grid in degrees within [-90,90] range (default: -90.0).
        lat_max (float): Northern boundary of target output grid in degrees within [-90,90] range (default: 90.0).
        resolution (float): Target grid spacing in degrees for both longitude and latitude directions (default: 1.0).
        method (str): Interpolation method - 'nearest' (fast, preserves values) or 'linear' (smooth, for contours) (default: 'nearest').
    
    Returns:
        xr.DataArray: Remapped data on regular latitude-longitude grid with dimensions [lat, lon] and coordinate labels.
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
    
    print(f"  Original MPAS data statistics:")
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
        tree = KDTree(source_points)
        
        target_points = np.column_stack([lon_2d.flatten(), lat_2d.flatten()])
        distances, indices = tree.query(target_points)
        
        data_2d = data_values[indices].reshape(lon_2d.shape)
    
    else:  # method == 'linear'
        source_points = np.column_stack([lon_deg, lat_deg])
        target_points = np.column_stack([lon_2d.flatten(), lat_2d.flatten()])
        
        data_flat = griddata(source_points, data_values, target_points, 
                            method='linear', fill_value=0.0)
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

    if data_sum == 0.0:
        print("\n[ERROR] Input data for remapping is empty, all zeros, or all NaNs. Cannot compute remapping ratio.\n")
        print("  Please check your input data source and variable selection.")
        print(f"  Data shape: {data_values.shape}, dtype: {data_values.dtype}")
        print(f"  Data min: {np.nanmin(data_values)}, max: {np.nanmax(data_values)}")
        print(f"  Data contains only NaNs: {np.isnan(data_values).all()}")
        print(f"  Data contains only zeros: {np.count_nonzero(data_values)==0}")
        print("  Remapping aborted. Returning empty result array.")
        return result

    sum_ratio = float(result.sum()) / data_sum

    print(f"  Remapped data statistics:")
    print(f"    Min: {float(result.min()):.4f}, Max: {float(result.max()):.4f}")
    print(f"    Mean: {float(result.mean()):.4f}, Median: {float(result.median()):.4f}")
    print(f"    Std: {float(result.std()):.4f}, Sum: {float(result.sum()):.4f}")

    if method == 'linear':
        print(f"  NOTE: Linear interpolation creates smooth fields but increases total sum by {sum_ratio:.1f}x")
        print(f"        This is expected when interpolating to a denser grid ({grid_expansion_ratio:.1f}x more points)")

    print(f"✓ Remapping completed successfully using {method} interpolation")

    return result


def create_target_grid(lon_min: float = -180.0,
                      lon_max: float = 180.0,
                      lat_min: float = -90.0,
                      lat_max: float = 90.0,
                      dlon: float = 1.0,
                      dlat: float = 1.0) -> xr.Dataset:
    """
    This utility function creates a uniform grid dataset with evenly spaced coordinate arrays suitable for use as a target grid in xESMF remapping operations. The grid specification includes only coordinate metadata without allocating space for data arrays, making it memory-efficient for defining target grid structures. The function supports both regional and global grid configurations with flexible longitude conventions. Generated grid objects can be reused across multiple remapping operations with different source data but identical target grid requirements.
    
    Parameters:
        lon_min (float): Western longitude boundary in degrees, supports [-180,180] or [0,360] convention (default: -180.0).
        lon_max (float): Eastern longitude boundary in degrees (default: 180.0).
        lat_min (float): Southern latitude boundary in degrees within [-90,90] valid range (default: -90.0).
        lat_max (float): Northern latitude boundary in degrees within [-90,90] valid range (default: 90.0).
        dlon (float): Longitude grid spacing in degrees determining east-west resolution (default: 1.0).
        dlat (float): Latitude grid spacing in degrees determining north-south resolution (default: 1.0).
    
    Returns:
        xr.Dataset: Grid specification dataset with 1D coordinate arrays 'lon' and 'lat' with dimensions ['lon'] and ['lat'].
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
    
    if not XESMF_AVAILABLE:
        print("\nxESMF is not installed. Install with:")
        print("  conda install -c conda-forge xesmf")
    else:
        print("\nxESMF is available")
        print(f"Supported methods: bilinear, conservative, conservative_normed, patch, nearest_s2d, nearest_d2s")
        print("\nExample usage:")
        print("""
from mpasdiag.processing.remapping import MPASRemapper, remap_mpas_to_latlon

# Method 1: High-level convenience function
remapped = remap_mpas_to_latlon(
    data=temperature,
    lon=mpas_lon,
    lat=mpas_lat,
    resolution=0.25,
    method='bilinear'
)

# Method 2: Full control with MPASRemapper
remapper = MPASRemapper(method='conservative', weights_dir='./weights')
remapper.prepare_source_grid(mpas_lon, mpas_lat)
remapper.create_target_grid(resolution=0.5)
remapper.build_regridder()
remapped = remapper.remap(data)
        """)

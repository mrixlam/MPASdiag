#!/usr/bin/env python3

"""
MPASdiag Core Processing Module: Data Remapping using KDTree.

This module provides functionality for remapping MPAS unstructured grid data to regular latitude-longitude grids using a KDTree-based nearest neighbor approach. It includes automatic detection and conversion of coordinate units, generation of intermediate structured grids, and efficient querying of nearest source points for target grid locations. This is particularly useful for preparing MPAS data for visualization or analysis on regular grids without the overhead of building full xESMF regridders, while still maintaining spatial relationships as closely as possible. The module also includes utilities for estimating memory usage and cleaning up resources after remapping operations.
    
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
    
    lon_deg = np.degrees(lon) if np.max(np.abs(lon)) <= np.pi else lon
    lat_deg = np.degrees(lat) if np.max(np.abs(lat)) <= np.pi else lat
    
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
    """ High-level interface for remapping MPAS unstructured grid data to regular latitude-longitude grids using xESMF interpolation methods. """
    
    def __init__(self: 'MPASRemapper', 
                 method: str = 'bilinear',
                 weights_dir: Optional[Union[str, Path]] = None,
                 reuse_weights: bool = True,
                 periodic: bool = False,
                 extrap_method: Optional[str] = None,
                 extrap_dist_exponent: Optional[float] = None,
                 extrap_num_src_pnts: Optional[int] = None) -> None:
        """
        This constructor initializes the MPASRemapper instance with user-defined settings for the interpolation method, weight caching, periodicity, and extrapolation options. It validates the selected interpolation method against a list of supported methods and sets up the internal state for managing grid specifications and the regridder object. If a weights directory is provided, it ensures that the directory exists for storing or loading pre-computed interpolation weights, which can significantly improve performance for repeated remapping operations with the same source-target grid pair. The constructor also prepares internal variables to hold the source and target grid datasets and the regridder object, which will be configured in subsequent steps. This initialization step is crucial for setting up the remapping workflow and ensuring that all necessary parameters are defined before performing any remapping operations. 
        
        Parameters:
            method (str): Interpolation method to use for remapping, options include 'bilinear', 'conservative', 'conservative_normed', 'patch', 'nearest_s2d', and 'nearest_d2s' (default: 'bilinear').
            weights_dir (Optional[Union[str, Path]]): Directory path for storing or loading pre-computed interpolation weights to speed up repeated remapping with the same grid pair (default: None).
            reuse_weights (bool): Whether to reuse existing weight files in weights_dir if available, or overwrite them by recomputing (default: True).
            periodic (bool): Whether the source grid is periodic in longitude, which affects how xESMF handles edge cases during interpolation (default: False).
            extrap_method (Optional[str]): Extrapolation method to use for target points outside the convex hull of source points, options include 'nearest_s2d', 'nearest_d2s', and 'inverse_distance' (default: None).
            extrap_dist_exponent (Optional[float]): Exponent for inverse distance weighting when using 'inverse_distance' extrapolation method, controls how quickly influence decreases with distance (default: None).
            extrap_num_src_pnts (Optional[int]): Number of nearest source points to consider when performing extrapolation for out-of-hull target points, applicable for both nearest and inverse distance methods (default: None). 
        
        Returns:
            None
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
    
    def prepare_source_grid(self: 'MPASRemapper', 
                            lon: Union[np.ndarray, xr.DataArray], 
                            lat: Union[np.ndarray, xr.DataArray], 
                            lon_bounds: Optional[Union[np.ndarray, xr.DataArray]] = None, 
                            lat_bounds: Optional[Union[np.ndarray, xr.DataArray]] = None) -> xr.Dataset:
        """
        This method prepares the source grid specification for xESMF by converting the input longitude and latitude coordinates to degrees if necessary, and then creating an xarray Dataset that contains the 'lon' and 'lat' coordinate arrays. If boundary coordinates are provided for conservative remapping methods, it also includes 'lon_b' and 'lat_b' arrays in the dataset. The method ensures that longitude values are in the [0, 360] range if they were originally in degrees and negative, which is a common convention for global datasets. The resulting source grid dataset is stored internally for use in building the regridder and can be returned for inspection or further processing. This step is essential for defining the spatial structure of the input MPAS data before performing any remapping operations with xESMF. 
        
        Parameters:
            lon (Union[np.ndarray, xr.DataArray]): Longitude coordinates of the source grid in degrees or radians.
            lat (Union[np.ndarray, xr.DataArray]): Latitude coordinates of the source grid in degrees or radians.
            lon_bounds (Optional[Union[np.ndarray, xr.DataArray]]): Optional longitude boundary coordinates for conservative remapping methods, should have shape (nCells, nv) where nv is number of vertices per cell (default: None).
            lat_bounds (Optional[Union[np.ndarray, xr.DataArray]]): Optional latitude boundary coordinates for conservative remapping methods, should have shape (nCells, nv) where nv is number of vertices per cell (default: None). 
        
        Returns:
            xr.Dataset: Source grid dataset containing 'lon' and 'lat' coordinate arrays, and optionally 'lon_b' and 'lat_b' for conservative remapping. 
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
        This method creates a regular latitude-longitude target grid specification for xESMF based on user-defined spatial boundaries and grid spacing. It generates 1D coordinate arrays for longitude and latitude using numpy's arange function, ensuring that the grid points are centered within the specified bounds. The method validates that the longitude and latitude boundaries are within acceptable ranges and that the maximum values are greater than the minimum values. The resulting target grid dataset contains 'lon' and 'lat' coordinate arrays with dimensions ['lon'] and ['lat'], which are stored internally for use in building the regridder. This step is crucial for defining the structure of the output grid onto which MPAS data will be remapped using xESMF's interpolation capabilities. 
        
        Parameters:
            lon_min (float): Minimum longitude for target grid in degrees, should be within [-360, 360] range (default: -180.0).
            lon_max (float): Maximum longitude for target grid in degrees, should be within [-360, 360] range and greater than lon_min (default: 180.0).
            lat_min (float): Minimum latitude for target grid in degrees, should be within [-90, 90] range (default: -90.0).
            lat_max (float): Maximum latitude for target grid in degrees, should be within [-90, 90] range and greater than lat_min (default: 90.0).
            dlon (float): Grid spacing in degrees for longitude direction, should be positive (default: 1.0).
            dlat (float): Grid spacing in degrees for latitude direction, should be positive (default: 1.0).
        
        Returns:
            xr.Dataset: Target grid dataset containing 'lon' and 'lat' coordinate arrays with dimensions ['lon'] and ['lat'] for use in xESMF remapping. 
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
    
    def build_regridder(self: 'MPASRemapper',
                       source_grid: Optional[xr.Dataset] = None,
                       target_grid: Optional[xr.Dataset] = None,
                       filename: Optional[str] = None) -> 'xe.Regridder':
        """
        This method builds the xESMF regridder object based on the prepared source and target grid specifications. It checks if the source and target grids have been provided as arguments or if they have been prepared and stored internally, raising an error if either grid is missing. The method then constructs the regridder using the specified interpolation method, periodicity, and extrapolation options. If a weights directory is configured and a filename is not provided, it auto-generates a descriptive filename for the weight file based on the method and grid sizes. The regridder is configured to reuse weights if specified, which can significantly speed up remapping operations for repeated grid pairs. After building the regridder, it is stored internally for use in remapping operations and returned for potential external use. This step is essential for setting up the actual remapping mechanism that will be applied to MPAS data using xESMF's powerful interpolation capabilities. 
        
        Parameters:
            source_grid (Optional[xr.Dataset]): Source grid dataset containing 'lon' and 'lat' coordinate arrays, and optionally 'lon_b' and 'lat_b' for conservative remapping. If None, uses internally stored source grid (default: None).
            target_grid (Optional[xr.Dataset]): Target grid dataset containing 'lon' and 'lat' coordinate arrays for use in xESMF remapping. If None, uses internally stored target grid (default: None).
            filename (Optional[str]): Optional filename for storing or loading pre-computed interpolation weights. If None and weights_dir is set, auto-generates a filename based on method and grid sizes (default: None).
        
        Returns:
            xe.Regridder: Configured xESMF regridder object ready for remapping operations. 
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
    
    def remap(self: 'MPASRemapper',
             data: Union[xr.DataArray, np.ndarray],
             keep_attrs: bool = True) -> xr.DataArray:
        """
        This method applies the configured xESMF regridder to remap a single variable from the source grid to the target grid. It checks if the regridder has been built before attempting to remap, raising an error if it is not available. The method supports both xarray DataArrays and numpy arrays as input, converting numpy arrays to DataArrays with a default dimension if necessary. The remapping operation is performed using the regridder object, and the resulting remapped data is returned as an xarray DataArray with proper coordinate labels corresponding to the target grid. If keep_attrs is True, it preserves the attributes and metadata from the input DataArray in the remapped output, which can be important for maintaining context and information about the variable being remapped. This method provides a simple interface for applying the remapping operation to individual variables after setting up the regridder with the desired source and target grids. 
        
        Parameters:
            data (Union[xr.DataArray, np.ndarray]): Input data array defined on source grid, can be an xarray DataArray or a numpy array (default: None).
            keep_attrs (bool): Whether to preserve attributes and metadata from the input DataArray in the remapped output (default: True).
        
        Returns:
            xr.DataArray: Remapped data on target grid as an xarray DataArray with appropriate coordinates and optionally preserved attributes. 
        """
        if self.regridder is None:
            raise ValueError("Regridder must be built before remapping. Call build_regridder() first.")
        
        if isinstance(data, np.ndarray):
            data = xr.DataArray(data, dims=['x'])
        
        result = self.regridder(data, keep_attrs=keep_attrs)
        
        return xr.DataArray(result) if not isinstance(result, xr.DataArray) else result
    
    def remap_dataset(self: 'MPASRemapper',
                     dataset: xr.Dataset,
                     variables: Optional[List[str]] = None,
                     skip_missing: bool = True) -> xr.Dataset:
        """
        This method applies the remapping operation to multiple variables within an xarray Dataset, allowing users to specify a subset of variables to remap or to remap all data variables if none are specified. It checks if the regridder has been built before attempting to remap, raising an error if it is not available. The method iterates over the specified variables, applying the remapping operation to each one and collecting the results in a new dataset. If a specified variable is not found in the input dataset, it can either skip that variable with a warning or raise an error based on the skip_missing flag. After remapping all specified variables, it returns a new xarray Dataset containing the remapped variables on the target grid, along with preserved non-spatial dimensions and metadata from the original dataset. This method provides a convenient way to apply remapping operations to multiple variables in a single step while maintaining flexibility in variable selection and error handling. 
        
        Parameters:
            dataset (xr.Dataset): Input xarray Dataset containing variables defined on the source grid to be remapped.
            variables (Optional[List[str]]): List of variable names in the dataset to remap. If None, all data variables will be remapped (default: None).
            skip_missing (bool): Whether to skip variables that are specified but not found in the dataset, with a warning, or to raise an error (default: True). 
        
        Returns:
            xr.Dataset: New xarray Dataset containing the remapped variables on the target grid, with preserved non-spatial dimensions and metadata. 
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
        This static method provides a utility for converting MPAS unstructured grid data into a 2D structured grid format using a KDTree-based nearest neighbor interpolation approach. It automatically detects the coordinate units (degrees or radians) and converts them to degrees if necessary. The method generates an intermediate regular grid based on user-defined spatial boundaries and resolution, then constructs a KDTree from the original unstructured coordinates to efficiently find the nearest source point for each target point on the intermediate grid. The resulting 2D structured data array is returned as an xarray DataArray with proper coordinate labels corresponding to the target grid, along with a Dataset containing the intermediate grid specification with coordinates and bounds. This function is useful for quickly preparing MPAS data for visualization or analysis on regular lat-lon grids without requiring the full setup of xESMF regridders, while still maintaining spatial relationships as closely as possible. 
        
        Parameters:
            data (Union[xr.DataArray, np.ndarray]): Input data array defined on MPAS unstructured grid with shape (nCells,) or subset.
            lon (Union[np.ndarray, xr.DataArray]): MPAS cell center longitude coordinates in degrees or radians.
            lat (Union[np.ndarray, xr.DataArray]): MPAS cell center latitude coordinates in degrees or radians.
            intermediate_resolution (float): Grid spacing in degrees for the intermediate structured grid (default: 0.1).
            lon_min (Optional[float]): Minimum longitude for intermediate grid in degrees, if None it will be determined from input coordinates with buffer (default: None).
            lon_max (Optional[float]): Maximum longitude for intermediate grid in degrees, if None it will be determined from input coordinates with buffer (default: None).
            lat_min (Optional[float]): Minimum latitude for intermediate grid in degrees, if None it will be determined from input coordinates with buffer (default: None).
            lat_max (Optional[float]): Maximum latitude for intermediate grid in degrees, if None it will be determined from input coordinates with buffer (default: None).
            buffer (float): Additional buffer in degrees to add to the spatial extent of the intermediate grid beyond the min/max of input coordinates when auto-determining bounds (default: 2.0). 
        
        Returns:
            Tuple[xr.DataArray, xr.Dataset]: A tuple containing the remapped data as an xarray DataArray on the intermediate structured grid, and a Dataset containing the intermediate grid specification with coordinates and bounds. 
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
    
    def cleanup(self: 'MPASRemapper') -> None:
        """
        This method cleans up resources associated with the regridder and grid specifications to free up memory. It checks if the regridder object exists and sets it to None, and also clears the source and target grid datasets. This is important for managing memory usage, especially when working with large grids or performing multiple remapping operations in a loop. After calling this method, the MPASRemapper instance will need to be reconfigured with new grid specifications and a new regridder before performing any further remapping operations. 
        
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
    def estimate_memory_usage(n_source: int, 
                              n_target: int, 
                              method: str) -> float:
        """
        This static method provides an estimate of the peak memory usage during the regridding operation based on the number of source and target grid points and the selected interpolation method. It calculates the memory required for storing the weight matrix, which depends on the sparsity pattern determined by the interpolation method, as well as the memory needed for the input and output data arrays. The method returns an estimated total memory usage in gigabytes (GB), which can be useful for planning and managing resources when working with large grids or limited memory environments. This estimation can help users understand the potential memory requirements before performing a remapping operation, allowing them to adjust grid sizes or select more memory-efficient methods if necessary. 
        
        Parameters:
            n_source (int): Number of points in the source grid (e.g., number of cells in MPAS).
            n_target (int): Number of points in the target grid (e.g., number of points in the regular lat-lon grid).
            method (str): Interpolation method used for remapping, which affects the sparsity of the weight matrix and thus the memory usage. Supported methods include 'bilinear', 'conservative', 'conservative_normed', 'patch', 'nearest_s2d', and 'nearest_d2s'. 
        
        Returns:
            float: Estimated total memory usage in gigabytes (GB) for the regridding operation based on the input grid sizes and interpolation method. 
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
    
    print(f"  Original MPAS data statistics [Global Statistics]:")
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
    
    lon_range = lon_max - lon_min
    is_global = lon_range > 180
    
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

    sum_ratio = float(result.sum()) / data_sum

    print(f"  Remapped data statistics [Statistics over Target Grid]:")
    print(f"    Min: {float(result.min()):.4f}, Max: {float(result.max()):.4f}")
    print(f"    Mean: {float(result.mean()):.4f}, Median: {float(result.median()):.4f}")
    print(f"    Std: {float(result.std()):.4f}, Sum: {float(result.sum()):.4f}")

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
    
    pts = np.column_stack((lon_vals, lat_vals))
    try:
        from scipy.spatial import ConvexHull
        from matplotlib.path import Path
        
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        
        if isinstance(remapped_data, xr.DataArray):
            vals = remapped_data.values
        else:
            vals = np.array(remapped_data)
            
        if vals.ndim == 2:
            nlat, nlon = vals.shape
            lat_coord = np.linspace(lat_min, lat_max, nlat)
            lon_coord = np.linspace(lon_min, lon_max, nlon)
            Lon, Lat = np.meshgrid(lon_coord, lat_coord)
            grid_points = np.column_stack((Lon.ravel(), Lat.ravel()))
            hull_path = Path(hull_pts)
            inside = hull_path.contains_points(grid_points)
            mask_bool = inside.reshape((nlat, nlon))
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


def _apply_lon_convention(
        lon_coords: xr.DataArray,
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
                                      lon_convention: str = 'auto') -> xr.DataArray:
    """
    This function provides a convenient interface for remapping MPAS unstructured grid data to a regular latitude-longitude grid with optional convex hull masking to identify valid points. It automatically detects the coordinate units (degrees or radians) and converts them to degrees if necessary. The function generates a regular target grid based on user-defined spatial boundaries and resolution, then constructs a KDTree from the original unstructured coordinates to efficiently find the nearest source point for each target point on the regular grid. After remapping, it applies a convex hull mask to identify which points on the target grid are within the original domain of the MPAS data, setting points outside the hull to NaN. The resulting remapped and masked data is returned as an xarray DataArray with proper coordinate labels corresponding to the target grid. This function is useful for preparing MPAS data for visualization or analysis on regular lat-lon grids while ensuring that only valid points within the original coverage are included in the results. 
    
    Parameters:
        data (Union[xr.DataArray, np.ndarray]): Input data array defined on MPAS unstructured grid with shape (nCells,) or subset.
        dataset (xr.Dataset): The original xarray Dataset containing the MPAS data and coordinates, used to extract cell center coordinates for masking.
        lon_min (Optional[float]): Minimum longitude for target grid in degrees, if None it will be determined from input coordinates with buffer (default: None).
        lon_max (Optional[float]): Maximum longitude for target grid in degrees, if None it will be determined from input coordinates with buffer (default: None).
        lat_min (Optional[float]): Minimum latitude for target grid in degrees, if None it will be determined from input coordinates with buffer (default: None).
        lat_max (Optional[float]): Maximum latitude for target grid in degrees, if None it will be determined from input coordinates with buffer (default: None).
        resolution (float): Grid spacing in degrees for the regular latitude-longitude grid (default: 0.1).
        method (str): Interpolation method to use, options include 'nearest' and 'linear' (default: 'nearest').
        apply_mask (bool): Whether to apply convex hull masking to set points outside the original MPAS coverage to NaN (default: True).
        lon_convention (str): Longitude convention to use for remapping, options include 'auto', '[-180,180]', and '[0,360]' (default: 'auto'). The 'auto' option will attempt to preserve the original longitude convention of the input dataset, while the other options will convert coordinates to the specified convention if they are not already in that format.
    
    Returns:
        xr.DataArray: Remapped (and optionally masked) data on regular latitude-longitude grid as an xarray DataArray with appropriate coordinates.
    """
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
        method=method,
    )

    if apply_mask:
        remapped_data = _apply_remap_mask(
            remapped_data, lon_coords.values, lat_coords.values,
            lon_min, lon_max, lat_min, lat_max, resolution
        )

    remapped_data.attrs.update(data_attrs)

    return remapped_data


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

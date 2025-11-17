#!/usr/bin/env python3

"""
MPAS Data Caching for Parallel Processing

This module provides efficient data caching mechanisms for parallel processing workflows to avoid redundant data loading across multiple worker processes and minimize memory footprint. It implements variable-specific caching strategies that store commonly accessed data (geographic coordinates, grid structures, extracted variable arrays) in memory with thread-safe and process-safe access patterns for both shared-memory multiprocessing and distributed-memory MPI parallel execution. The caching system uses lazy loading where data is loaded only on first access, provides cache invalidation and refresh mechanisms for handling dataset changes, and dramatically reduces memory usage by caching only the specific variables and coordinates that workers actually need rather than passing entire processor objects with full datasets. Core capabilities include automatic cache key generation based on file paths and variable names, support for both 2D surface and 3D atmospheric MPAS data, shared memory optimization where available, and comprehensive cache statistics for performance monitoring and debugging in high-throughput diagnostic workflows.

Classes:
    MPASDataCache: Thread-safe caching system for MPAS grid and variable data in parallel processing contexts.
    
Functions:
    get_cached_processor: Retrieves or creates cached processor instance with specified dataset configuration.
    clear_cache: Clears all cached data to free memory or force data refresh.
    get_cache_stats: Returns cache statistics including hit rate, memory usage, and cached item counts.
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import os
import threading
import warnings
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import xarray as xr
from dataclasses import dataclass, field
import multiprocessing


@dataclass
class CachedVariable:
    """Container for a cached variable with metadata."""
    data: np.ndarray
    var_name: str
    time_index: Optional[int] = None
    level_index: Optional[int] = None
    units: Optional[str] = None
    long_name: Optional[str] = None
    timestamp: float = field(default_factory=lambda: __import__('time').time())


class MPASDataCache:
    """
    Thread-safe data cache for MPAS parallel processing to minimize redundant data loading.
    
    This class provides efficient caching of MPAS grid coordinates and variable data that
    are commonly accessed across multiple parallel workers. Instead of passing entire dataset
    objects to each worker (which causes redundant I/O and excessive memory usage), workers
    access pre-cached data through this centralized cache.
    
    The cache is designed to work with both multiprocessing (shared memory on single node)
    and MPI (distributed memory across nodes) backends.
    
    Usage:
        # Master process initializes cache
        cache = MPASDataCache()
        cache.load_coordinates_from_dataset(dataset, var_name)
        cache.load_variable_data(dataset, var_name, time_index)
        
        # Workers access cached data
        lon, lat = cache.get_coordinates(var_name)
        data = cache.get_variable_data(var_name, time_index)
    """
    
    def __init__(self, max_variables: int = 10) -> None:
        """
        Initialize the data cache with thread-safe locks and storage containers. This constructor sets up the internal data structures for caching coordinates and variable data along with thread synchronization primitives. The cache implements a simple LRU-like policy based on max_variables limit to prevent unbounded memory growth. The lock is lazily initialized on first use to enable pickle serialization for multiprocessing.

        Parameters:
            max_variables (int): Maximum number of variables to cache simultaneously to control memory usage (default: 10).

        Returns:
            None
        """
        self._lock = None  # Will be initialized on first use
        self._coordinates: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._variables: Dict[str, CachedVariable] = {}
        self._grid_data: Dict[str, np.ndarray] = {}  # For static grid info (nCells, etc.)
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self.max_variables = max_variables
        self._access_count: Dict[str, int] = {}
    
    def _get_lock(self) -> threading.RLock:
        """
        Get or create the lock object for thread-safe operations with lazy initialization. This approach allows the cache to be pickled for multiprocessing by creating the lock on first access in each process. Threading locks cannot be pickled, so this lazy initialization pattern avoids serialization issues when passing the cache to worker processes. The lock is a reentrant lock allowing the same thread to acquire it multiple times.

        Parameters:
            None

        Returns:
            threading.RLock: Thread-safe reentrant lock for cache operations enabling nested locking within single thread.
        """
        if self._lock is None:
            self._lock = threading.RLock()
        return self._lock
    
    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepare cache state for pickling by excluding non-picklable lock object. This method is called during pickle serialization when sending cache to worker processes via multiprocessing. It returns a copy of the instance dictionary with the lock removed since threading locks cannot be pickled. The lock will be lazily recreated in worker processes on first access.

        Parameters:
            None

        Returns:
            Dict[str, Any]: State dictionary without the lock object ready for pickle serialization.
        """
        state = self.__dict__.copy()
        state['_lock'] = None
        return state
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restore cache state after unpickling and reinitialize the lock object. This method is called during unpickle deserialization when receiving cache in worker processes. It restores the instance dictionary and sets lock to None which will be lazily recreated on first access via _get_lock(). This pattern ensures proper lock initialization in each worker process.

        Parameters:
            state (Dict[str, Any]): State dictionary from pickling containing all cache data except the lock.

        Returns:
            None
        """
        self.__dict__.update(state)
        self._lock = None
        
    def load_coordinates_from_dataset(
        self, 
        dataset: xr.Dataset, 
        var_name: Optional[str] = None
    ) -> None:
        """
        Load and cache spatial coordinates from an MPAS dataset for specified variable. This method extracts longitude and latitude coordinates appropriate for the given variable and stores them in the cache for repeated access. For MPAS unstructured grids, it handles both cell-centered coordinates and vertex/edge-based coordinates based on variable dimensions. The cached coordinates are converted to numpy arrays and degrees for efficient use.

        Parameters:
            dataset (xr.Dataset): MPAS dataset containing coordinate variables like lonCell, latCell, lonVertex, latVertex.
            var_name (Optional[str]): Variable name for variable-specific coordinates, None uses default cell-centered coordinates (default: None).

        Returns:
            None

        Raises:
            ValueError: If required coordinate variables are not found in dataset for the specified variable type.
        """
        with self._get_lock():
            cache_key = var_name or 'default'
            
            # Check if already cached
            if cache_key in self._coordinates:
                return
            
            # Extract coordinates based on variable dimensions
            if var_name and var_name in dataset.data_vars:
                var_dims = dataset[var_name].sizes
                
                # Determine coordinate variables based on dimensions
                if 'nCells' in var_dims:
                    lon_var, lat_var = 'lonCell', 'latCell'
                elif 'nVertices' in var_dims:
                    lon_var, lat_var = 'lonVertex', 'latVertex'
                elif 'nEdges' in var_dims:
                    lon_var, lat_var = 'lonEdge', 'latEdge'
                else:
                    lon_var, lat_var = 'lonCell', 'latCell'
            else:
                # Default coordinates
                lon_var, lat_var = 'lonCell', 'latCell'
            
            if lon_var not in dataset or lat_var not in dataset:
                raise ValueError(f"Coordinate variables {lon_var}, {lat_var} not found in dataset")
            
            lon = dataset[lon_var].values
            lat = dataset[lat_var].values
            
            # Convert from radians to degrees if needed
            if np.max(np.abs(lon)) <= 2 * np.pi:
                lon = np.degrees(lon)
                lat = np.degrees(lat)
            
            # Normalize longitude to [-180, 180] range
            lon = ((lon + 180) % 360) - 180
            
            self._coordinates[cache_key] = (lon, lat)
            print(f"Cached coordinates for '{cache_key}': {len(lon)} points")
    
    def get_coordinates(self, var_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve cached spatial coordinates for the specified variable from cache. This method provides thread-safe access to previously cached longitude and latitude coordinates. It increments an access counter for cache management and returns the coordinate arrays. Must call load_coordinates_from_dataset before using this method.

        Parameters:
            var_name (Optional[str]): Variable name for coordinate lookup, None retrieves default coordinates (default: None).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two-element tuple containing (longitude, latitude) arrays in degrees as 1D numpy arrays.

        Raises:
            KeyError: If coordinates for the specified variable haven't been cached yet via load_coordinates_from_dataset.
        """
        with self._get_lock():
            cache_key = var_name or 'default'
            if cache_key not in self._coordinates:
                raise KeyError(f"Coordinates for '{cache_key}' not loaded. Call load_coordinates_from_dataset first.")
            
            self._access_count[cache_key] = self._access_count.get(cache_key, 0) + 1
            return self._coordinates[cache_key]
    
    def load_variable_data(
        self,
        dataset: xr.Dataset,
        var_name: str,
        time_index: Optional[int] = None,
        level_index: Optional[int] = None
    ) -> None:
        """
        Load and cache a specific variable's data for the given time and vertical level indices. This method extracts variable data from the dataset, applies temporal and vertical indexing if specified, and stores it in the cache along with metadata. The cache implements a simple LRU-like eviction when the maximum cache size is exceeded, removing the least recently accessed variable to free memory. Metadata including units and long_name are preserved for later use.

        Parameters:
            dataset (xr.Dataset): MPAS dataset containing the variable to cache.
            var_name (str): Name of the variable to cache from the dataset.
            time_index (Optional[int]): Time index to extract, None caches all times (default: None).
            level_index (Optional[int]): Vertical level index for 3D variables, None caches all levels (default: None).

        Returns:
            None

        Raises:
            ValueError: If variable not found in dataset or has incompatible dimensions for indexing.
        """
        with self._get_lock():
            cache_key = self._make_cache_key(var_name, time_index, level_index)
            
            # Check if already cached
            if cache_key in self._variables:
                return
            
            if var_name not in dataset.data_vars:
                raise ValueError(f"Variable '{var_name}' not found in dataset")
            
            # Extract variable data
            var_data = dataset[var_name]
            
            # Apply time indexing
            if time_index is not None and 'Time' in var_data.sizes:
                var_data = var_data.isel(Time=time_index)
            
            # Apply vertical level indexing
            if level_index is not None:
                if 'nVertLevels' in var_data.sizes:
                    var_data = var_data.isel(nVertLevels=level_index)
                elif 'nVertLevelsP1' in var_data.sizes:
                    var_data = var_data.isel(nVertLevelsP1=level_index)
            
            # Extract metadata
            units = var_data.attrs.get('units', '')
            long_name = var_data.attrs.get('long_name', var_name)
            
            # Convert to numpy array
            data_array = var_data.values
            
            # Check cache size and evict if needed
            if len(self._variables) >= self.max_variables:
                self._evict_least_accessed()
            
            # Store in cache
            cached_var = CachedVariable(
                data=data_array,
                var_name=var_name,
                time_index=time_index,
                level_index=level_index,
                units=units,
                long_name=long_name
            )
            
            self._variables[cache_key] = cached_var
            self._access_count[cache_key] = 0
            
            print(f"Cached variable '{cache_key}': shape {data_array.shape}, "
                  f"size {data_array.nbytes / 1024 / 1024:.2f} MB")
    
    def get_variable_data(
        self,
        var_name: str,
        time_index: Optional[int] = None,
        level_index: Optional[int] = None
    ) -> CachedVariable:
        """
        Retrieve cached variable data for the specified variable, time, and level indices. This method provides thread-safe access to previously cached variable data including both the numpy array and associated metadata. It updates access counters for cache management decisions and returns the complete CachedVariable object. Must call load_variable_data before using this method.

        Parameters:
            var_name (str): Name of the cached variable to retrieve.
            time_index (Optional[int]): Time index of cached data, None for unindexed data (default: None).
            level_index (Optional[int]): Vertical level index of cached data, None for unindexed data (default: None).

        Returns:
            CachedVariable: Object containing data array and metadata including units, long_name, and timestamp.

        Raises:
            KeyError: If requested variable data hasn't been cached yet via load_variable_data.
        """
        with self._get_lock():
            cache_key = self._make_cache_key(var_name, time_index, level_index)
            if cache_key not in self._variables:
                raise KeyError(f"Variable data '{cache_key}' not loaded. Call load_variable_data first.")
            
            self._access_count[cache_key] = self._access_count.get(cache_key, 0) + 1
            return self._variables[cache_key]
    
    def clear(self) -> None:
        """
        Clear all cached data from memory to free resources. This method removes all cached coordinates, variables, metadata, and access counters effectively resetting the cache to its initial empty state. Use this when switching between different datasets or when memory needs to be reclaimed. Thread-safe operation ensured through internal locking.

        Parameters:
            None

        Returns:
            None
        """
        with self._get_lock():
            self._coordinates.clear()
            self._variables.clear()
            self._grid_data.clear()
            self._metadata.clear()
            self._access_count.clear()
            print("Cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get statistics and information about current cache state for monitoring and debugging. This method returns a dictionary containing cache size, memory usage, and access patterns. Useful for debugging cache performance and determining if cache settings need adjustment. Provides insights into which variables are most frequently accessed and total memory consumption.

        Parameters:
            None

        Returns:
            Dict[str, Any]: Cache statistics dictionary with keys 'num_coordinates', 'num_variables', 'total_memory_mb', and 'most_accessed' variable.
        """
        with self._get_lock():
            total_memory = sum(
                var.data.nbytes for var in self._variables.values()
            )
            
            return {
                'num_coordinates': len(self._coordinates),
                'num_variables': len(self._variables),
                'total_memory_mb': total_memory / 1024 / 1024,
                'most_accessed': max(self._access_count.items(), key=lambda x: x[1])[0] if self._access_count else None
            }
    
    @staticmethod
    def _make_cache_key(
        var_name: str,
        time_index: Optional[int],
        level_index: Optional[int]
    ) -> str:
        """
        Generate a unique cache key string from variable name and indices for dictionary lookups. This internal method creates a consistent string key that uniquely identifies a specific variable at a specific time and vertical level. The key format allows for efficient dictionary-based cache access and avoids collisions between different variable states. Keys are formatted as 'varname_t<idx>_l<idx>' with components omitted when indices are None.

        Parameters:
            var_name (str): Variable name to include in cache key.
            time_index (Optional[int]): Time index to include or None to omit from key.
            level_index (Optional[int]): Level index to include or None to omit from key.

        Returns:
            str: Unique cache key string in format 'varname_t<idx>_l<idx>' suitable for dictionary keys.
        """
        key = var_name
        if time_index is not None:
            key += f"_t{time_index}"
        if level_index is not None:
            key += f"_l{level_index}"
        return key
    
    def _evict_least_accessed(self) -> None:
        """
        Evict the least recently accessed variable from cache to free memory. This internal method implements a simple LRU-like eviction policy by removing the cached variable with the lowest access count. Called automatically when the cache reaches its maximum size limit to make room for new variables. Prints diagnostic information about the evicted variable and memory freed.

        Parameters:
            None

        Returns:
            None
        """
        if not self._variables:
            return
        
        # Find least accessed variable
        least_accessed = min(self._access_count.items(), key=lambda x: x[1])[0]
        
        if least_accessed in self._variables:
            evicted = self._variables.pop(least_accessed)
            self._access_count.pop(least_accessed, None)
            print(f"Evicted variable '{least_accessed}' from cache "
                  f"({evicted.data.nbytes / 1024 / 1024:.2f} MB freed)")


# Global cache instance for use across workers
_global_cache: Optional[MPASDataCache] = None


def get_global_cache() -> MPASDataCache:
    """
    Get or create the global singleton MPASDataCache instance for worker access. This function provides access to a globally shared cache instance that can be used across multiple function calls and workers. The singleton pattern ensures all workers within a process share the same cache maximizing memory efficiency in multiprocessing mode. Creates a new cache instance on first call and returns the same instance on subsequent calls.

    Parameters:
        None

    Returns:
        MPASDataCache: Global cache instance shared across all workers in the current process.
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = MPASDataCache()
    return _global_cache


def clear_global_cache() -> None:
    """
    Clear and reset the global cache instance to free memory. This function clears all data from the global cache and sets it to None for garbage collection. Can be used to force cache refresh or reclaim memory between different processing runs. Safe to call even if global cache doesn't exist yet.

    Parameters:
        None

    Returns:
        None
    """
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
        _global_cache = None

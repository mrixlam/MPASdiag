#!/usr/bin/env python3

"""
MPASdiag Core Processing Module: Data Caching for Parallel Processing

This module implements a thread-safe data caching mechanism for MPAS diagnostic processing, designed to store intermediate data products such as coordinates and variable arrays in memory for efficient access across multiple processing steps. The cache uses Python dictionaries to store data and metadata, and a threading lock to ensure safe concurrent access in multi-threaded contexts. It also includes a simple LRU-like eviction policy to manage memory usage when caching large variables. The cache is designed to be used in multiprocessing scenarios by implementing custom pickling behavior that allows the lock to be recreated in worker processes, enabling shared access to cached data without serialization issues.  
    
Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""
# Load necessary libraries
import threading
import numpy as np
import xarray as xr
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple


@dataclass
class CachedVariable:
    """ Container for a cached variable with metadata. """
    data: np.ndarray
    var_name: str
    time_index: Optional[int] = None
    level_index: Optional[int] = None
    units: Optional[str] = None
    long_name: Optional[str] = None
    timestamp: float = field(default_factory=lambda: __import__('time').time())


class MPASDataCache:
    """ Thread-safe data cache for MPAS diagnostic processing with support for multiprocessing. """
    
    def __init__(self: "MPASDataCache", 
                 max_variables: int = 10,
                 max_coordinates: int = 10) -> None:
        """
        This constructor initializes the MPASDataCache instance with empty dictionaries for coordinates, variables, grid data, and metadata. It also sets up a lock for thread safety and initializes access count tracking for eviction policies. The maximum number of variables and coordinate sets to cache can be configured through the parameters, allowing for memory management when caching large datasets. The cache is designed to be used in multiprocessing contexts by implementing custom pickling behavior that allows the lock to be recreated in worker processes, enabling shared access to cached data without serialization issues. 

        Parameters:
            max_variables (int): Maximum number of variables to cache before evicting least accessed (default: 10).
            max_coordinates (int): Maximum number of coordinate sets to cache before evicting least accessed (default: 10).

        Returns:
            None
        """
        self._lock = None  
        self._coordinates: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._variables: Dict[str, CachedVariable] = {}
        self._grid_data: Dict[str, np.ndarray] = {}  
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self.max_variables = max_variables
        self.max_coordinates = max_coordinates
        self._access_count: Dict[str, int] = {}
    
    def _get_lock(self: "MPASDataCache") -> threading.RLock:
        """
        This internal method provides access to a thread-safe reentrant lock for cache operations. It lazily initializes the lock the first time it is accessed to avoid unnecessary overhead when the cache is not used in a multi-threaded context. The use of a reentrant lock allows for nested locking within the same thread, which can be useful if multiple cache operations are performed in sequence. This method ensures that all cache operations that modify shared data structures are protected by the lock to prevent race conditions and ensure thread safety. 

        Parameters:
            None

        Returns:
            threading.RLock: Reentrant lock instance for synchronizing cache access. 
        """
        if self._lock is None:
            self._lock = threading.RLock()
        return self._lock
    
    def __getstate__(self: "MPASDataCache") -> Dict[str, Any]:
        """
        This method customizes the pickling behavior of the cache to exclude the lock object from the serialized state. It creates a copy of the cache's __dict__ and sets the _lock entry to None before returning it for pickling. This allows the cache to be safely serialized and sent to worker processes without including the non-picklable lock object, while still preserving all cached data and metadata. The lock will be lazily recreated in each worker process when accessed, ensuring that the cache remains thread-safe in the multiprocessing context. 

        Parameters:
            None

        Returns:
            Dict[str, Any]: State dictionary for pickling with _lock set to None. 
        """
        state = self.__dict__.copy()
        state['_lock'] = None
        return state
    
    def __setstate__(self: "MPASDataCache", 
                     state: Dict[str, Any]) -> None:
        """
        This method customizes the unpickling behavior of the cache to restore the state from the pickled data and reinitialize the lock object. It updates the cache's __dict__ with the provided state dictionary, which contains all cached data and metadata except for the lock. After restoring the state, it sets the _lock attribute to None, allowing it to be lazily initialized when accessed in the worker process. This ensures that each worker process has its own lock instance for thread safety while sharing access to the cached data. 

        Parameters:
            state (Dict[str, Any]): State dictionary from unpickling containing cached data and metadata with _lock set to None. 

        Returns:
            None
        """
        self.__dict__.update(state)
        self._lock = None
        
    def load_coordinates_from_dataset(self: "MPASDataCache", 
                                      dataset: xr.Dataset, 
                                      var_name: Optional[str] = None) -> None:
        """
        This method loads and caches spatial coordinates (longitude and latitude) from the provided MPAS dataset for a specific variable or default cell-centered coordinates. It checks if the coordinates for the given variable name or default key are already cached to avoid redundant loading, extracts the appropriate coordinate variables based on the dimensions of the specified variable, converts them from radians to degrees if necessary, normalizes longitude values to the [-180, 180] range, and stores the resulting longitude and latitude arrays in the cache under a key derived from the variable name or 'default'. This method must be called before get_coordinates to ensure that the required coordinates are available in the cache for retrieval. 

        Parameters:
            dataset (xr.Dataset): MPAS dataset containing the coordinate variables to cache.
            var_name (Optional[str]): Variable name to determine which coordinates to load, None for default cell-centered coordinates (default: None). 

        Returns:
            None
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
            
            # Evict least accessed coordinates if at capacity
            if len(self._coordinates) >= self.max_coordinates:
                self._evict_least_accessed_coordinates()
            
            self._coordinates[cache_key] = (lon, lat)
            print(f"Cached coordinates for '{cache_key}': {len(lon)} points")
    
    def get_coordinates(self: "MPASDataCache", 
                        var_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method retrieves cached spatial coordinates (longitude and latitude) for a specific variable or default cell-centered coordinates. It checks the cache for the requested coordinates based on a key derived from the variable name or 'default', and returns the longitude and latitude arrays as 1D numpy arrays in degrees. If the requested coordinates are not found in the cache, it raises a KeyError indicating that load_coordinates_from_dataset must be called first to populate the cache. This method provides thread-safe access to cached coordinates for use in subsequent processing steps that require spatial information. 

        Parameters:
            var_name (Optional[str]): Variable name to determine which coordinates to retrieve, None for default cell-centered coordinates (default: None). 

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (longitude, latitude) arrays in degrees for the requested variable or default coordinates. 
        """
        with self._get_lock():
            cache_key = var_name or 'default'
            if cache_key not in self._coordinates:
                raise KeyError(f"Coordinates for '{cache_key}' not loaded. Call load_coordinates_from_dataset first.")
            
            self._access_count[cache_key] = self._access_count.get(cache_key, 0) + 1
            return self._coordinates[cache_key]
    
    def load_variable_data(self: "MPASDataCache", 
                           dataset: xr.Dataset, 
                           var_name: str, 
                           time_index: Optional[int] = None, 
                           level_index: Optional[int] = None) -> None:
        """
        This method loads and caches variable data from the provided MPAS dataset for a specific variable name and optional time and vertical level indices. It checks if the variable data for the given combination of variable name and indices is already cached to avoid redundant loading, extracts the specified variable from the dataset, applies time and vertical level indexing if provided, retrieves metadata such as units and long_name, converts the data to a numpy array, checks the cache size against the maximum allowed variables and evicts the least accessed variable if necessary, and stores the resulting data array along with metadata in the cache under a key derived from the variable name and indices. This method must be called before get_variable_data to ensure that the required variable data is available in the cache for retrieval. 

        Parameters:
            dataset (xr.Dataset): MPAS dataset containing the variable to cache.
            var_name (str): Name of the variable to load and cache from the dataset.
            time_index (Optional[int]): Time index to select from the variable, None for unindexed data (default: None).
            level_index (Optional[int]): Vertical level index to select from the variable, None for unindexed data (default: None). 

        Returns:
            None
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
    
    def get_variable_data(self: "MPASDataCache", 
                          var_name: str, 
                          time_index: Optional[int] = None, 
                          level_index: Optional[int] = None) -> CachedVariable:
        """
        This method retrieves cached variable data for a specific variable name and optional time and vertical level indices. It checks the cache for the requested variable data based on a key derived from the variable name and indices, and returns a CachedVariable object containing the data array and metadata for the requested variable. If the requested variable data is not found in the cache, it raises a KeyError indicating that load_variable_data must be called first to populate the cache. This method provides thread-safe access to cached variable data for use in subsequent processing steps that require the variable values and associated metadata. 

        Parameters:
            var_name (str): Name of the variable to retrieve from the cache.
            time_index (Optional[int]): Time index to select from the variable, None for unindexed data (default: None).
            level_index (Optional[int]): Vertical level index to select from the variable, None for unindexed data (default: None). 

        Returns:
            CachedVariable: CachedVariable object containing the data array and metadata for the requested variable. 
        """
        with self._get_lock():
            cache_key = self._make_cache_key(var_name, time_index, level_index)
            if cache_key not in self._variables:
                raise KeyError(f"Variable data '{cache_key}' not loaded. Call load_variable_data first.")
            
            self._access_count[cache_key] = self._access_count.get(cache_key, 0) + 1
            return self._variables[cache_key]
    
    def clear(self: "MPASDataCache") -> None:
        """
        This method clears all cached data and metadata from the cache, including coordinates, variables, grid data, and metadata dictionaries. It acquires the cache lock to ensure thread safety while clearing the internal data structures, and resets the access count tracking as well. This method can be used to free up memory when switching between different datasets or when the cached data is no longer needed. After calling this method, the cache will be empty and will need to be repopulated by calling the appropriate loading methods before any retrieval operations can be performed again. 

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
    
    def get_cache_info(self: "MPASDataCache") -> Dict[str, Any]:
        """
        This method provides information about the current state of the cache, including the number of cached coordinates, the number of cached variables, the total memory used by cached variables in megabytes, and the most accessed variable based on the access count tracking. It acquires the cache lock to ensure thread safety while accessing the internal data structures, calculates the total memory usage by summing the sizes of all cached variable data arrays, and returns a dictionary containing these statistics. This method can be used for monitoring cache usage and performance during MPAS diagnostic processing. 

        Parameters:
            None

        Returns:
            Dict[str, Any]: Dictionary containing cache information such as number of coordinates, number of variables, total memory usage in MB, and most accessed variable.
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
    def _make_cache_key(var_name: str, 
                        time_index: Optional[int], 
                        level_index: Optional[int]) -> str:
        """
        This static method generates a unique cache key string based on the variable name and optional time and vertical level indices. It constructs the key by starting with the variable name and appending suffixes for time and level indices if they are provided, resulting in a string format like 'varname_t<idx>_l<idx>'. This key is used to store and retrieve variable data in the cache dictionaries, allowing for efficient access to specific variable slices based on time and vertical level. The method ensures that different combinations of variable name and indices produce distinct keys for proper caching behavior. 

        Parameters:
            var_name (str): Name of the variable to create a cache key for.
            time_index (Optional[int]): Time index to include in the cache key, None if not applicable.
            level_index (Optional[int]): Vertical level index to include in the cache key, None if not applicable. 

        Returns:
            str: Generated cache key string based on the variable name and indices. 
        """
        key = var_name
        if time_index is not None:
            key += f"_t{time_index}"
        if level_index is not None:
            key += f"_l{level_index}"
        return key
    
    def _evict_least_accessed(self: "MPASDataCache") -> None:
        """
        This internal method implements a simple eviction policy to remove the least accessed variable from the cache when the maximum number of cached variables is exceeded. It identifies the variable with the lowest access count from the _access_count dictionary, removes it from the _variables cache and the _access_count tracking, and prints a message indicating which variable was evicted and how much memory was freed. This method helps manage memory usage in the cache by ensuring that only the most frequently accessed variables are retained when caching large datasets. 

        Parameters:
            None

        Returns:
            None
        """
        if not self._variables:
            return
        
        least_accessed = min(self._access_count.items(), key=lambda x: x[1])[0]
        
        if least_accessed in self._variables:
            evicted = self._variables.pop(least_accessed)
            self._access_count.pop(least_accessed, None)
            print(f"Evicted variable '{least_accessed}' from cache "
                  f"({evicted.data.nbytes / 1024 / 1024:.2f} MB freed)")

    def _evict_least_accessed_coordinates(self: "MPASDataCache") -> None:
        """
        This internal method implements a simple eviction policy to remove the least accessed coordinate set from the cache when the maximum number of cached coordinates is exceeded. It identifies the coordinate key with the lowest access count, removes it from the _coordinates cache and the _access_count tracking, and prints a message indicating which coordinate set was evicted.

        Parameters:
            None

        Returns:
            None
        """
        if not self._coordinates:
            return
        
        coord_keys = set(self._coordinates.keys())
        coord_access = {k: self._access_count.get(k, 0) for k in coord_keys}
        least_accessed = min(coord_access.items(), key=lambda x: x[1])[0]
        
        evicted_lon, evicted_lat = self._coordinates.pop(least_accessed)
        self._access_count.pop(least_accessed, None)
        mem_freed = (evicted_lon.nbytes + evicted_lat.nbytes) / 1024 / 1024
        
        print(f"Evicted coordinates '{least_accessed}' from cache "
              f"({mem_freed:.2f} MB freed)")


_global_cache: Optional[MPASDataCache] = None


def get_global_cache() -> MPASDataCache:
    """
    This function provides access to a global instance of the MPASDataCache that can be shared across all workers in the current process. It checks if the global cache instance has already been created, and if not, it initializes a new MPASDataCache instance and assigns it to the global variable. This allows for efficient sharing of cached data across multiple processing steps without the need for passing cache instances explicitly between functions or workers. The global cache can be accessed by calling this function from any part of the code, ensuring that all operations that require caching can utilize the same shared cache instance. 

    Parameters:
        None

    Returns:
        MPASDataCache: Global instance of the MPASDataCache for shared access across workers. 
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = MPASDataCache()
    return _global_cache


def clear_global_cache() -> None:
    """
    This function clears the global cache instance by calling its clear method and then setting the global variable to None. This effectively resets the global cache, freeing up memory and allowing for a fresh cache to be created when get_global_cache is called again. This function can be used when switching between different datasets or when the cached data is no longer needed, ensuring that stale data does not persist in the cache and that memory usage is managed effectively. 

    Parameters:
        None

    Returns:
        None
    """
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
        _global_cache = None

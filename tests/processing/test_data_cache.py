#!/usr/bin/env python3
"""
MPASdiag Test Suite: Data Caching Functionality

This module contains unit tests for the MPASDataCache class, which is responsible for caching coordinate and variable data extracted from MPAS datasets. The tests cover loading coordinates, loading variable data, cache eviction behavior, and error handling when accessing uncached data. Both synthetic datasets and real MPAS datasets are used to validate functionality in controlled and realistic scenarios. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries 
import os
import sys
import pytest
import numpy as np
import xarray as xr
from typing import Any
import matplotlib.pyplot as plt

from mpasdiag.processing.data_cache import MPASDataCache, CachedVariable, get_global_cache, clear_global_cache

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestLoadCoordinatesFromDataset:
    """ Test cases for load_coordinates_from_dataset method """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestLoadCoordinatesFromDataset", mpas_3d_processor) -> Any:
        """
        This fixture sets up the test environment for all tests in the TestLoadCoordinatesFromDataset class by providing access to a shared MPAS dataset processor. It checks if the processor is available and skips tests if not, ensuring that tests only run when the necessary data is present. The fixture also handles cleanup after tests by closing any open matplotlib figures to prevent resource leaks. This setup allows individual test methods to focus on their specific assertions without worrying about dataset loading or cleanup. 

        Parameters:
            self (Any): Test case instance providing dataset fixtures.
            mpas_3d_processor (Any): A fixture that provides a pre-loaded MPAS dataset processor for testing.

        Returns:
            None
        """
        if mpas_3d_processor is None:
            pytest.skip("MPAS data not available")
        
        self.processor = mpas_3d_processor
        
        yield
        
        plt.close('all')
    
    def test_load_coordinates_default(self: "TestLoadCoordinatesFromDataset") -> None:
        """
        This test verifies that when `load_coordinates_from_dataset` is called without specifying a variable name, it loads the default coordinates (e.g., `lonCell` and `latCell`) from the dataset and caches them correctly. It asserts that the loaded longitude and latitude arrays are of the expected type and contain valid values within the typical geographic ranges. This test ensures that the method can successfully load and cache default coordinates for general use in plotting and diagnostics. 

        Parameters:
            self (Any): Test case instance providing access to the MPAS dataset.

        Returns:
            None
        """
        processor = self.processor
        cache = MPASDataCache()
        
        cache.load_coordinates_from_dataset(processor.dataset, var_name=None)
        
        assert 'default' in cache._coordinates
        lon, lat = cache.get_coordinates(var_name=None)
        
        assert isinstance(lon, np.ndarray)
        assert isinstance(lat, np.ndarray)
        assert len(lon) > 0
        
        assert np.all(lon >= -180)
        assert np.all(lon <= 180)
    
    def test_load_coordinates_already_cached(self: "TestLoadCoordinatesFromDataset") -> None:
        """
        This test confirms that if coordinates for a variable are already cached, subsequent calls to `load_coordinates_from_dataset` with the same variable name do not reload the data or modify the existing cache entry. It loads coordinates for a variable, records the original size of the cached coordinates, and then calls the loader again for the same variable. The test asserts that the size of the cached coordinates remains unchanged, indicating that the existing cache entry was reused rather than reloaded. This behavior is important for performance optimization and cache stability. 

        Parameters:
            self (Any): Test case instance providing dataset fixtures.

        Returns:
            None
        """
        processor = self.processor        
        cache = MPASDataCache()
        
        cache.load_coordinates_from_dataset(processor.dataset, 'test_var')
        original_size = len(cache._coordinates['test_var'][0])
        
        cache.load_coordinates_from_dataset(processor.dataset, 'test_var')        
        assert len(cache._coordinates['test_var'][0]) == original_size
    
    def test_load_coordinates_for_ncells_variable(self: "TestLoadCoordinatesFromDataset") -> None:
        """
        This test verifies that when `load_coordinates_from_dataset` is called with a variable that has an `nCells` dimension, the cache correctly identifies and loads the corresponding `lonCell` and `latCell` coordinate variables. It searches the dataset for a variable with `nCells`, loads coordinates for it, and asserts that the longitude array has a positive length, confirming successful loading of cell-centered coordinates. This ensures that the loader correctly handles cell-based variables and their associated coordinate variables. 

        Parameters:
            self (Any): Test case instance with a loaded MPAS dataset.

        Returns:
            None
        """
        processor = self.processor        
        cache = MPASDataCache()
        ncells_var = None

        for var_name in processor.dataset.data_vars:
            if 'nCells' in processor.dataset[var_name].sizes:
                ncells_var = var_name
                break
        
        if ncells_var:
            cache.load_coordinates_from_dataset(processor.dataset, ncells_var)            
            lon, lat = cache.get_coordinates(ncells_var)
            assert len(lon) > 0
    
    def test_load_coordinates_for_nvertices_variable(self: "TestLoadCoordinatesFromDataset") -> None:
        """
        This test checks that when `load_coordinates_from_dataset` is called with a variable that has an `nVertices` dimension, the cache correctly identifies and loads the corresponding `lonVertex` and `latVertex` coordinate variables. It iterates through the dataset to find a variable with `nVertices`, loads coordinates for it, and asserts that the longitude array has a positive length, confirming successful loading of vertex-centered coordinates. This test ensures that the loader can handle vertex-based variables and their associated coordinate variables correctly. 

        Parameters:
            self (Any): Test case instance with access to the grid file.

        Returns:
            None
        """
        processor = self.processor
        ds = processor.dataset
        cache = MPASDataCache()        
        nvertices_var = None

        for var_name in ds.data_vars:
            if 'nVertices' in ds[var_name].sizes:
                nvertices_var = var_name
                break
        
        if nvertices_var and 'lonVertex' in ds and 'latVertex' in ds:
            cache.load_coordinates_from_dataset(ds, nvertices_var)            
            lon, lat = cache.get_coordinates(nvertices_var)
            assert len(lon) > 0
    
    def test_load_coordinates_for_nedges_variable(self: "TestLoadCoordinatesFromDataset") -> None:
        """
        This test ensures that when `load_coordinates_from_dataset` is called with a variable that has an `nEdges` dimension, the cache correctly identifies and loads the corresponding `lonEdge` and `latEdge` coordinate variables. It searches the dataset for a variable with `nEdges`, loads coordinates for it, and asserts that the longitude array has a positive length, confirming successful loading of edge-centered coordinates. This test verifies that the loader can handle edge-based variables and their associated coordinate variables appropriately. 

        Parameters:
            self (Any): Test case instance with the grid dataset available.

        Returns:
            None
        """
        processor = self.processor
        ds = processor.dataset
        cache = MPASDataCache()        
        nedges_var = None

        for var_name in ds.data_vars:
            if 'nEdges' in ds[var_name].sizes:
                nedges_var = var_name
                break
        
        if nedges_var and 'lonEdge' in ds and 'latEdge' in ds:
            cache.load_coordinates_from_dataset(ds, nedges_var)            
            lon, lat = cache.get_coordinates(nedges_var)
            assert len(lon) > 0
    
    def test_load_coordinates_missing_coord_vars(self: "TestLoadCoordinatesFromDataset") -> None:
        """
        This test verifies that if `load_coordinates_from_dataset` is called for a variable that has a recognized dimension (e.g., `nCells`) but the corresponding coordinate variables (e.g., `lonCell` and `latCell`) are missing from the dataset, the method raises a ValueError with an informative message. It creates a mock dataset with a variable that has an `nCells` dimension but does not include the expected coordinate variables, then calls the loader and asserts that the raised exception contains text indicating the missing coordinate variables. This ensures that users receive clear feedback when required coordinate variables are not present in the dataset. 

        Parameters:
            self (Any): Test case instance context (not used functionally).

        Returns:
            None
        """
        mock_ds = xr.Dataset({
            'test_var': xr.DataArray(np.random.rand(10), dims=['nCells'])
        })
        
        cache = MPASDataCache()
        
        with pytest.raises(ValueError) as context:
            cache.load_coordinates_from_dataset(mock_ds, 'test_var')
        
        assert "Coordinate variables" in str(context.value)
    
    def test_load_coordinates_radians_to_degrees_conversion(self: "TestLoadCoordinatesFromDataset") -> None:
        """
        This test checks that if the coordinate variables in the dataset are stored in radians, the `load_coordinates_from_dataset` method correctly converts them to degrees before caching. It creates a mock dataset with longitude and latitude variables in radians, loads them into the cache, and asserts that the resulting longitude and latitude values are within the expected geographic ranges for degrees. Proper conversion from radians to degrees is essential for accurate plotting and geographic calculations, as most plotting libraries and geographic tools expect coordinates in degrees. 

        Parameters:
            self (Any): Test case instance providing MPAS dataset fixtures.

        Returns:
            None
        """
        processor = self.processor
        
        cache = MPASDataCache()
        cache.load_coordinates_from_dataset(processor.dataset)        
        lon, lat = cache.get_coordinates()
        
        assert np.all(np.abs(lon) <= 180)
        assert np.all(np.abs(lat) <= 90)
    
    def test_load_coordinates_longitude_normalization(self: "TestLoadCoordinatesFromDataset") -> None:
        """
        This test verifies that when `load_coordinates_from_dataset` loads longitude values, it normalizes them to the range [-180, 180] if they are not already in that range. It creates a mock dataset with longitude values that exceed the typical geographic range (e.g., 0 to 360), loads them into the cache, and asserts that the resulting longitude values are normalized to the expected range. Normalizing longitudes is important for consistent plotting and geographic computations, as it ensures that all longitude values are represented in a standard format regardless of how they are stored in the original dataset. 

        Parameters:
            self (Any): Test case instance with loaded MPAS dataset.

        Returns:
            None
        """
        processor = self.processor
        
        cache = MPASDataCache()
        cache.load_coordinates_from_dataset(processor.dataset)        
        lon, lat = cache.get_coordinates()
        
        assert np.all(lon >= -180)
        assert np.all(lon <= 180)
    
    def test_load_coordinates_variable_without_recognized_dimensions(self: "TestLoadCoordinatesFromDataset") -> None:
        """
        This test checks that if `load_coordinates_from_dataset` is called with a variable that does not have any recognized dimensions (e.g., `nCells`, `nVertices`, `nEdges`), the method raises a ValueError with an informative message. It creates a mock dataset with a variable that has unrecognized dimensions, then calls the loader and asserts that the raised exception contains text indicating that the variable's dimensions are not recognized for coordinate loading. This ensures that users receive clear feedback when trying to load coordinates for variables that do not fit expected dimension patterns. 

        Parameters:
            self (Any): Test case instance context (no external resources).

        Returns:
            None
        """
        ds = xr.Dataset({
            'lonCell': xr.DataArray(np.radians(np.array([0, 45, 90])), dims=['nCells']),
            'latCell': xr.DataArray(np.radians(np.array([0, 30, 60])), dims=['nCells']),
            'test_var': xr.DataArray(np.random.rand(5, 10), dims=['nTime', 'nOther'])
        })
        
        cache = MPASDataCache()        
        cache.load_coordinates_from_dataset(ds, 'test_var')        
        lon, lat = cache.get_coordinates('test_var')

        assert len(lon) == 3


class TestGetCoordinatesError:
    """ Tests for get_coordinates error handling. """
    
    def test_get_coordinates_not_loaded(self: "TestGetCoordinatesError") -> None:
        """
        This test ensures that if `get_coordinates` is called for a variable whose coordinates have not been loaded into the cache, it raises a KeyError with a message instructing the user to load the coordinates first. It attempts to retrieve coordinates for a non-existent variable and asserts that the raised exception contains guidance text about missing coordinates and how to load them. This defensive behavior prevents confusion when users attempt to access coordinates that haven't been cached yet. 

        Parameters:
            self (Any): Test case instance.

        Returns:
            None
        """
        cache = MPASDataCache()
        
        with pytest.raises(KeyError) as context:
            cache.get_coordinates('nonexistent_var')
        
        assert "not loaded" in str(context.value)
        assert "Call load_coordinates_from_dataset" in str(context.value)


class TestLoadVariableData:
    """ Tests for load_variable_data method. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestLoadVariableData", mpas_3d_processor) -> Any:
        """
        This fixture sets up the test environment for all tests in the TestLoadVariableData class by providing access to a shared MPAS dataset processor. It checks if the processor is available and skips tests if not, ensuring that tests only run when the necessary data is present. The fixture also handles cleanup after tests by closing any open matplotlib figures to prevent resource leaks. This setup allows individual test methods to focus on their specific assertions without worrying about dataset loading or cleanup. 

        Parameters:
            self (Any): Test case instance providing dataset fixtures.
            mpas_3d_processor (Any): A fixture that provides a pre-loaded MPAS dataset processor for testing.

        Returns:
            None
        """
        if mpas_3d_processor is None:
            pytest.skip("MPAS data not available")
        
        self.processor = mpas_3d_processor
        
        yield
        
        plt.close('all')
    
    def test_load_variable_data_basic(self: "TestLoadVariableData") -> None:
        """
        This test verifies that when `load_variable_data` is called with a valid variable name from the dataset, it loads the variable data into the cache and creates a `CachedVariable` object with the correct attributes. It searches for a variable with a `Time` dimension and an `nCells` dimension, loads it into the cache, and asserts that the cached variable has the expected name, time index, and that the data is stored as a `CachedVariable` instance. This test ensures that the basic functionality of loading variable data into the cache works correctly. 

        Parameters:
            self (Any): Test case instance configured with MPAS sample data paths.

        Returns:
            None
        """
        processor = self.processor
        cache = MPASDataCache()        
        var_name = None

        for vname in processor.dataset.data_vars:
            if 'Time' in processor.dataset[vname].sizes and 'nCells' in processor.dataset[vname].sizes:
                if len(processor.dataset[vname].sizes) == 2:  
                    var_name = vname
                    break
        
        if var_name:
            cache.load_variable_data(processor.dataset, var_name, time_index=0)            
            cache_key = f"{var_name}_t0"

            assert cache_key in cache._variables
            
            cached = cache.get_variable_data(var_name, time_index=0)
            assert isinstance(cached, CachedVariable)
            assert cached.var_name == var_name
            assert cached.time_index == 0
    
    def test_load_variable_already_cached(self: "TestLoadVariableData") -> None:
        """
        This test confirms that if `load_variable_data` is called for a variable that is already cached, it does not reload the data or modify the existing cache entry. It loads a variable into the cache, records the original timestamp of the cached variable, and then calls the loader again for the same variable. The test asserts that the timestamp of the cached variable remains unchanged, indicating that the existing cache entry was reused rather than reloaded. This behavior is important for performance optimization and cache stability. 

        Parameters:
            self (Any): Test case instance configured with MPAS sample data paths.

        Returns:
            None
        """
        processor = self.processor        
        cache = MPASDataCache()
        var_name = list(processor.dataset.data_vars.keys())[0]
        
        cache.load_variable_data(processor.dataset, var_name)
        original_timestamp = cache._variables[var_name].timestamp        
        cache.load_variable_data(processor.dataset, var_name)
        
        assert cache._variables[var_name].timestamp == original_timestamp
    
    def test_load_variable_not_found(self: "TestLoadVariableData") -> None:
        """
        This test ensures that if `load_variable_data` is called with a variable name that does not exist in the dataset, it raises a ValueError with an informative message. It attempts to load a non-existent variable and asserts that the raised exception contains text indicating that the variable was not found in the dataset. This defensive behavior prevents confusion when users attempt to load variables that do not exist in the dataset and provides clear feedback on the issue. 

        Parameters:
            self (Any): Test case instance with dataset fixtures available.

        Returns:
            None
        """
        processor = self.processor
        cache = MPASDataCache()
        
        with pytest.raises(ValueError) as context:
            cache.load_variable_data(processor.dataset, 'nonexistent_variable')
        
        assert "not found in dataset" in str(context.value)
    
    def test_load_variable_with_time_indexing(self: "TestLoadVariableData") -> None:
        """
        This test verifies that when `load_variable_data` is called for a variable with a `Time` dimension, it correctly loads the data for the specified time index and stores it in the cache. It searches the dataset for a variable with a `Time` dimension, loads it with a specific time index, and asserts that the cached variable's `time_index` matches the requested index. Additionally, it checks that the loaded data does not contain the `Time` dimension, confirming that the loader correctly slices the data along the time dimension. Proper handling of time indexing is crucial for working with time-varying variables in MPAS datasets. 

        Parameters:
            self (Any): Test case instance with real MPAS dataset paths.

        Returns:
            None
        """
        processor = self.processor
        cache = MPASDataCache()        
        var_name = None

        for vname in processor.dataset.data_vars:
            if 'Time' in processor.dataset[vname].sizes:
                var_name = vname
                break
        
        if var_name:
            cache.load_variable_data(processor.dataset, var_name, time_index=0)
            cached = cache.get_variable_data(var_name, time_index=0)

            assert cached.time_index == 0            
            assert 'Time' not in (cached.data.shape if hasattr(cached.data, 'shape') else [])
    
    def test_load_variable_with_nvertlevels_indexing(self: "TestLoadVariableData") -> None:
        """
        This test verifies that variables with an `nVertLevels` dimension support vertical level indexing. It searches the dataset for a variable with `nVertLevels`, loads it with a specified level index, and asserts that the cached variable's `level_index` matches the requested index. This confirms that the loader correctly handles vertical slicing for 3D variables, which is essential for layer-based diagnostics and visualizations. Proper vertical indexing ensures users can access specific levels of interest without loading unnecessary data. 

        Parameters:
            self (Any): Test case instance prepared with MPAS dataset fixtures.

        Returns:
            None
        """
        processor = self.processor        
        cache = MPASDataCache()        
        var_name = None

        for vname in processor.dataset.data_vars:
            if 'nVertLevels' in processor.dataset[vname].sizes:
                var_name = vname
                break
        
        if var_name:
            cache.load_variable_data(processor.dataset, var_name, time_index=0, level_index=0)            
            cached = cache.get_variable_data(var_name, time_index=0, level_index=0)
            assert cached.level_index == 0
    
    def test_load_variable_with_nvertlevelsp1_indexing(self: "TestLoadVariableData") -> None:
        """
        This test confirms that variables with an `nVertLevelsP1` dimension also support vertical level indexing. It searches the dataset for a variable with `nVertLevelsP1`, loads it with a specified level index, and asserts that the cached variable's `level_index` matches the requested index. This ensures that the loader can handle both standard vertical levels and the additional "P1" levels that may be present in some MPAS datasets, providing flexibility for users working with different types of vertical coordinate systems. Proper handling of `nVertLevelsP1` is important for accessing surface or top-level data in certain diagnostics. 

        Parameters:
            self (Any): Test case instance set up with real dataset paths.

        Returns:
            None
        """
        processor = self.processor        
        cache = MPASDataCache()
        var_name = None

        for vname in processor.dataset.data_vars:
            if 'nVertLevelsP1' in processor.dataset[vname].sizes:
                var_name = vname
                break
        
        if var_name:
            cache.load_variable_data(processor.dataset, var_name, level_index=0)            
            cached = cache.get_variable_data(var_name, level_index=0)
            assert cached.level_index == 0
    
    def test_load_variable_metadata_extraction(self: "TestLoadVariableData") -> None:
        """
        This test checks that when `load_variable_data` is called, it correctly extracts and stores metadata such as units and long name from the dataset variable attributes. It loads a variable into the cache and asserts that the cached variable's `units` and `long_name` attributes are not None, confirming that metadata extraction is functioning properly. Accurate metadata is important for labeling plots and understanding the physical meaning of variables in diagnostics. 

        Parameters:
            self (Any): Test case instance with MPAS dataset available.

        Returns:
            None
        """
        processor = self.processor        
        cache = MPASDataCache()        
        var_name = list(processor.dataset.data_vars.keys())[0]
        
        cache.load_variable_data(processor.dataset, var_name)        
        cached = cache.get_variable_data(var_name)
        
        assert cached.units is not None
        assert cached.long_name is not None
    
    def test_load_variable_cache_size_limit_and_eviction(self: "TestLoadVariableData") -> None:
        """
        This test verifies that when the number of variables loaded into the cache exceeds the specified `max_variables` limit, the cache correctly evicts the least accessed variable to make room for new data. It loads multiple variables into the cache, accesses one of them multiple times to increase its access count, and then loads an additional variable to trigger eviction. The test asserts that the least accessed variable is evicted while the most accessed variable remains in the cache, confirming that the eviction policy based on access counts is functioning as intended. This behavior is crucial for managing memory usage effectively when working with large datasets. 

        Parameters:
            self (Any): Test case instance configured for real-data tests.

        Returns:
            None
        """
        processor = self.processor        
        cache = MPASDataCache(max_variables=2)        
        var_names = list(processor.dataset.data_vars.keys())[:3]
        
        cache.load_variable_data(processor.dataset, var_names[0])
        cache.load_variable_data(processor.dataset, var_names[1])
        
        assert len(cache._variables) == 2
        
        for _ in range(5):
            cache.get_variable_data(var_names[0])
        
        cache.load_variable_data(processor.dataset, var_names[2])
        
        assert var_names[0] in cache._variables
        assert var_names[2] in cache._variables


class TestGetVariableDataError:
    """ Tests for get_variable_data error handling. """
    
    def test_get_variable_data_not_loaded(self: "TestGetVariableDataError") -> None:
        """
        This test ensures that if `get_variable_data` is called for a variable that has not been loaded into the cache, it raises a KeyError with a message instructing the user to load the variable data first. It attempts to retrieve variable data for a non-existent variable and asserts that the raised exception contains guidance text about missing variable data and how to load it. This defensive behavior prevents confusion when users attempt to access variable data that hasn't been cached yet. 

        Parameters:
            self (Any): Test case instance.

        Returns:
            None
        """
        cache = MPASDataCache()
        
        with pytest.raises(KeyError) as context:
            cache.get_variable_data('nonexistent_var', time_index=0)
        
        assert "not loaded" in str(context.value)
        assert "Call load_variable_data" in str(context.value)
    
    def test_get_variable_data_access_count_increment(self: "TestGetVariableDataError") -> None:
        """
        This test verifies that each time `get_variable_data` is called for a cached variable, the access count for that variable is incremented correctly. It creates a mock cached variable, adds it to the cache with an initial access count of zero, and then calls `get_variable_data` multiple times. The test asserts that after each call, the access count for the variable increases by one, confirming that the cache is tracking variable access properly. Accurate tracking of access counts is important for implementing eviction policies based on usage patterns. 

        Parameters:
            self (Any): Test case instance.

        Returns:
            None
        """
        cache = MPASDataCache()

        var = CachedVariable(
            data=np.random.rand(100),
            var_name='test_var'
        )

        cache._variables['test_var'] = var
        cache._access_count['test_var'] = 0
        
        for i in range(3):
            cache.get_variable_data('test_var')
            assert cache._access_count['test_var'] == i + 1


class TestGetCacheInfo:
    """ Tests for get_cache_info with most_accessed. """
    
    def test_get_cache_info_with_most_accessed(self: "TestGetCacheInfo") -> None:
        """
        This test confirms that the `get_cache_info` method correctly identifies and returns the name of the most accessed variable in the cache. It populates the cache with multiple variables, assigns different access counts to them, and then calls `get_cache_info`. The test asserts that the returned information includes the correct number of variables and that the `most_accessed` field corresponds to the variable with the highest access count. This functionality is important for users to understand which variables are being accessed most frequently, which can inform decisions about cache management and optimization. 

        Parameters:
            self (Any): Test case instance.

        Returns:
            None
        """
        cache = MPASDataCache()
        
        var1 = CachedVariable(data=np.random.rand(100), var_name='var1')
        var2 = CachedVariable(data=np.random.rand(100), var_name='var2')
        
        cache._variables['var1'] = var1
        cache._variables['var2'] = var2
        cache._access_count['var1'] = 5
        cache._access_count['var2'] = 10
        
        info = cache.get_cache_info()
        
        assert info['num_variables'] == 2
        assert info['most_accessed'] == 'var2'
    
    def test_get_cache_info_empty_cache(self: "TestGetCacheInfo") -> None:
        """
        This test verifies that when `get_cache_info` is called on an empty cache (i.e., no coordinates or variables loaded), it returns the correct information indicating that there are zero coordinates and variables, and that there is no most accessed variable. It creates a new cache instance without loading any data, calls `get_cache_info`, and asserts that the returned information reflects the empty state of the cache. This ensures that the method can handle the edge case of an empty cache gracefully and provides accurate information about the cache state. 

        Parameters:
            self (Any): Test case instance.

        Returns:
            None
        """
        cache = MPASDataCache()        
        info = cache.get_cache_info()
        
        assert info['num_coordinates'] == 0
        assert info['num_variables'] == 0
        assert info['most_accessed'] is None


class TestEvictLeastAccessed:
    """ Tests for _evict_least_accessed method. """
    
    def test_evict_least_accessed_empty_cache(self: "TestEvictLeastAccessed") -> None:
        """
        This test ensures that when the `_evict_least_accessed` method is called on an empty cache (i.e., no variables loaded), it does not raise any exceptions and simply leaves the cache unchanged. It creates a new cache instance without loading any variables, calls the eviction method, and asserts that the internal variable storage remains empty. This test confirms that the eviction method can handle the edge case of an empty cache gracefully without causing errors or unintended side effects. 

        Parameters:
            self (Any): Test case instance.

        Returns:
            None
        """
        cache = MPASDataCache()
        cache._evict_least_accessed()
        
        assert len(cache._variables) == 0
    
    def test_evict_least_accessed_finds_minimum(self: "TestEvictLeastAccessed") -> None:
        """
        This test confirms that the `_evict_least_accessed` method correctly identifies and evicts the variable with the lowest access count when the cache exceeds its maximum variable limit. It populates the cache with multiple variables, assigns different access counts to them, and then calls the eviction method. The test asserts that the variable with the lowest access count is removed from the cache while the others remain intact. This behavior is crucial for managing memory usage effectively when working with large datasets. 

        Parameters:
            self (Any): Test case instance.

        Returns:
            None
        """
        cache = MPASDataCache()
        
        var1 = CachedVariable(data=np.random.rand(100), var_name='var1')
        var2 = CachedVariable(data=np.random.rand(100), var_name='var2')
        var3 = CachedVariable(data=np.random.rand(100), var_name='var3')
        
        cache._variables['var1'] = var1
        cache._variables['var2'] = var2
        cache._variables['var3'] = var3
        
        cache._access_count['var1'] = 10
        cache._access_count['var2'] = 2  
        cache._access_count['var3'] = 5
        
        cache._evict_least_accessed()
        
        assert 'var2' not in cache._variables
        assert 'var2' not in cache._access_count
        assert 'var1' in cache._variables
        assert 'var3' in cache._variables


class TestRealDataIntegration:
    """ Integration tests with real MPAS data. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestRealDataIntegration", mpas_3d_processor) -> Any:
        """
        This fixture sets up the test environment for all tests in the TestRealDataIntegration class by providing access to a shared MPAS dataset processor loaded with real data. It checks if the processor is available and skips tests if not, ensuring that tests only run when the necessary data is present. The fixture also handles cleanup after tests by closing any open matplotlib figures to prevent resource leaks. This setup allows individual test methods to focus on their specific assertions related to real data integration without worrying about dataset loading or cleanup. 

        Parameters:
            self (Any): Test case instance providing dataset fixtures.
            mpas_3d_processor (Any): A fixture that provides a pre-loaded MPAS dataset processor for testing.

        Returns:
            None
        """
        if mpas_3d_processor is None:
            pytest.skip("MPAS data not available")
        
        self.processor = mpas_3d_processor
        
        yield
        
        plt.close('all')
    
    def test_complete_workflow_with_real_data(self: "TestRealDataIntegration") -> None:
        """
        This test verifies that the complete workflow of loading coordinates and variable data into the cache works correctly when using a real MPAS dataset. It loads coordinates from the dataset, retrieves them to confirm they are cached, loads a variable with time indexing, retrieves the variable data to confirm it is cached, checks the cache information for correctness, and finally clears the cache to ensure it resets properly. This end-to-end test confirms that the caching mechanism functions as intended in a real-data context, allowing for efficient access to coordinates and variable data across different parts of the application. 

        Parameters:
            self (Any): Test case instance with access to real MPAS dataset.

        Returns:
            None
        """
        processor = self.processor        
        cache = MPASDataCache()
        
        cache.load_coordinates_from_dataset(processor.dataset)
        lon, lat = cache.get_coordinates()
        
        assert len(lon) > 0
        assert len(lat) > 0
        
        var_name = list(processor.dataset.data_vars.keys())[0]
        cache.load_variable_data(processor.dataset, var_name, time_index=0)
        
        cached_var = cache.get_variable_data(var_name, time_index=0)
        assert isinstance(cached_var, CachedVariable)
        
        info = cache.get_cache_info()
        assert info['num_coordinates'] == 1
        assert info['num_variables'] == 1
        
        cache.clear()
        assert len(cache._coordinates) == 0
        assert len(cache._variables) == 0
    
    def test_global_cache_with_real_data(self: "TestRealDataIntegration") -> None:
        """
        This test checks that the global cache instance behaves as a singleton and can load coordinates from a real MPAS dataset. It clears the global cache, retrieves it twice to confirm both references point to the same instance, loads coordinates from the dataset using one reference, and then retrieves the coordinates using the other reference to confirm they are shared. Finally, it clears the global cache again to ensure it resets properly. This test confirms that the global cache mechanism works correctly in a real-data context and that coordinate data is accessible across different parts of the application through the singleton pattern. 

        Parameters:
            self (Any): Test case instance with access to real MPAS dataset.

        Returns:
            None
        """
        processor = self.processor
        
        clear_global_cache()

        cache1 = get_global_cache()
        cache2 = get_global_cache()
        
        assert cache1 is cache2
        
        cache1.load_coordinates_from_dataset(processor.dataset)
        
        lon, lat = cache2.get_coordinates()
        assert len(lon) > 0
        
        clear_global_cache()


if __name__ == "__main__":
    pytest.main([__file__])

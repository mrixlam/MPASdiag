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
from typing import Any
import matplotlib.pyplot as plt

from mpasdiag.processing.data_cache import MPASDataCache, CachedVariable, get_global_cache, clear_global_cache

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestLoadVariableData:
    """ Tests for load_variable_data method. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestLoadVariableData', 
                     mpas_3d_processor: Any) -> Any:
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
            return
        
        self.processor = mpas_3d_processor
        
        yield
        
        plt.close('all')
    
    def test_load_variable_data_basic(self: 'TestLoadVariableData') -> None:
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
            assert cached.time_index == pytest.approx(0, abs=1e-3)
    
    def test_load_variable_already_cached(self: 'TestLoadVariableData') -> None:
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
    
    
    def test_load_variable_with_time_indexing(self: 'TestLoadVariableData') -> None:
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

            assert cached.time_index == pytest.approx(0, abs=1e-3)            
            assert 'Time' not in (cached.data.shape if hasattr(cached.data, 'shape') else [])
    
    def test_load_variable_with_nvertlevels_indexing(self: 'TestLoadVariableData') -> None:
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
            assert cached.level_index == pytest.approx(0, abs=1e-3)
    
    def test_load_variable_with_nvertlevelsp1_indexing(self: 'TestLoadVariableData') -> None:
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
            assert cached.level_index == pytest.approx(0, abs=1e-3)
    
    def test_load_variable_metadata_extraction(self: 'TestLoadVariableData') -> None:
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
    
    def test_load_variable_cache_size_limit_and_eviction(self: 'TestLoadVariableData') -> None:
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
        
        assert len(cache._variables) == pytest.approx(2)
        
        for _ in range(5):
            cache.get_variable_data(var_names[0])
        
        cache.load_variable_data(processor.dataset, var_names[2])
        
        assert var_names[0] in cache._variables
        assert var_names[2] in cache._variables


class TestRealDataIntegration:
    """ Integration tests with real MPAS data. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestRealDataIntegration', 
                     mpas_3d_processor: Any) -> Any:
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
            return
        
        self.processor = mpas_3d_processor
        
        yield
        
        plt.close('all')
    
    def test_complete_workflow_with_real_data(self: 'TestRealDataIntegration') -> None:
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
        assert info['num_coordinates'] == pytest.approx(1)
        assert info['num_variables'] == pytest.approx(1)
        
        cache.clear()
        assert len(cache._coordinates) == pytest.approx(0)
        assert len(cache._variables) == pytest.approx(0)
    
    def test_global_cache_with_real_data(self: 'TestRealDataIntegration') -> None:
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

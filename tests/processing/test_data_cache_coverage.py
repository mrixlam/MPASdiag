#!/usr/bin/env python3

"""
MPASdiag Test Suite: Data Cache Coverage

This module contains tests for the MPASDataCache class in the mpasdiag.processing.data_cache module, specifically targeting code coverage for the coordinate loading and variable data loading functionality. The tests are designed to verify that the cache correctly handles various scenarios, including pickling behavior, coordinate loading branches based on dataset dimensions, handling of missing coordinate variables, eviction of least accessed coordinates, and error handling when accessing unloaded data. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import pickle
import numpy as np
import pytest
import xarray as xr

from mpasdiag.processing.data_cache import MPASDataCache


N_CELLS = 8
N_VERT = 5
N_EDGES = 6
N_TIME = 2


def _make_cell_ds() -> xr.Dataset:
    """
    This dataset has a variable with nCells dimension and lonCell/latCell coordinates, suitable for testing the cell-based coordinate loading branch. It also serves as a basic dataset for testing variable loading and pickling behavior. 

    Parameters:
        None

    Returns:
        xr.Dataset: A dataset with a variable "temperature" that has dimensions (Time, nCells), and coordinate variables "lonCell" and "latCell" with dimension (nCells).
    """
    return xr.Dataset({
        "temperature": (["Time", "nCells"], np.ones((N_TIME, N_CELLS))),
        "lonCell":     (["nCells"], np.linspace(0.1, 1.5, N_CELLS)),
        "latCell":     (["nCells"], np.linspace(0.2, 0.8, N_CELLS)),
    })


def _make_vertex_ds() -> xr.Dataset:
    """
    This dataset has a variable with nVertices dimension and lonVertex/latVertex coordinates, suitable for testing the vertex-based coordinate loading branch. It also serves as a basic dataset for testing variable loading and pickling behavior.

    Parameters:
        None

    Returns:
        xr.Dataset: A dataset with a variable "vorticity" that has dimensions (Time, nVertices), and coordinate variables "lonVertex" and "latVertex" with dimension (nVertices).
    """
    return xr.Dataset({
        "vorticity": (["Time", "nVertices"], np.ones((N_TIME, N_VERT))),
        "lonVertex": (["nVertices"], np.linspace(0.1, 1.5, N_VERT)),
        "latVertex": (["nVertices"], np.linspace(0.2, 0.8, N_VERT)),
    })


def _make_edge_ds() -> xr.Dataset:
    """
    This dataset has a variable with nEdges dimension and lonEdge/latEdge coordinates, suitable for testing the edge-based coordinate loading branch. It also serves as a basic dataset for testing variable loading and pickling behavior. 

    Parameters:
        None

    Returns:
        xr.Dataset: A dataset with a variable "u_normal" that has dimensions (Time, nEdges), and coordinate variables "lonEdge" and "latEdge" with dimension (nEdges).
    """
    return xr.Dataset({
        "u_normal": (["Time", "nEdges"], np.ones((N_TIME, N_EDGES))),
        "lonEdge":  (["nEdges"], np.linspace(0.1, 1.5, N_EDGES)),
        "latEdge":  (["nEdges"], np.linspace(0.2, 0.8, N_EDGES)),
    })


class TestSetstate:
    """ Test the __setstate__ method of MPASDataCache, which is used during unpickling to restore the state of the cache. """

    def test_pickle_roundtrip_restores_coordinates(self: 'TestSetstate') -> None:
        """
        This test verifies that when an MPASDataCache instance is pickled and then unpickled, the coordinate data is preserved correctly. It ensures that the essential data in the cache survives the serialization process, even though locks cannot be directly serialized. 

        Parameters:
            None

        Returns:
            None
        """
        cache = MPASDataCache()
        cache.load_coordinates_from_dataset(_make_cell_ds())

        raw = pickle.dumps(cache)
        restored: MPASDataCache = pickle.loads(raw)

        assert restored._lock is None
        assert len(restored._coordinates) == len(cache._coordinates)

    def test_pickle_roundtrip_lock_lazily_recreated(self: 'TestSetstate') -> None:
        """
        This test verifies that when an MPASDataCache instance is pickled and then unpickled, the _lock attribute is reset to None and can be lazily recreated when accessed. Since locks cannot be directly serialized, the __setstate__ method should set _lock to None during unpickling. The test confirms that after unpickling, the _get_lock() method can successfully create a new lock instance, ensuring that the cache remains functional. 

        Parameters:
            None

        Returns:
            None
        """
        cache = MPASDataCache()
        raw = pickle.dumps(cache)
        restored: MPASDataCache = pickle.loads(raw)

        assert restored._lock is None
        lock = restored._get_lock()
        assert lock is not None

    def test_pickle_preserves_variables(self: 'TestSetstate') -> None:
        """
        This test verifies that when an MPASDataCache instance is pickled and then unpickled, the variable data stored in the cache is preserved correctly. It ensures that the essential variable data in the cache survives the serialization process, even though locks cannot be directly serialized. By loading variable data into the cache before pickling and then checking for its presence after unpickling, we can confirm that the variable data is retained as expected. 

        Parameters:
            None

        Returns:
            None
        """
        cache = MPASDataCache()
        cache.load_variable_data(_make_cell_ds(), "temperature")

        restored: MPASDataCache = pickle.loads(pickle.dumps(cache))

        assert restored._lock is None
        assert "temperature" in restored._variables


class TestLoadCoordinatesEarlyReturn:
    """ Test that load_coordinates_from_dataset returns early without modifying the cache when coordinates are already loaded for the given dataset and variable name. """

    def test_second_load_is_noop(self: 'TestLoadCoordinatesEarlyReturn') -> None:
        """
        This test verifies that if load_coordinates_from_dataset is called a second time with the same dataset and no variable name, it does not reload the coordinates and instead returns early. By overwriting the cached coordinates with a sentinel value before the second call, we can confirm that the original coordinates are not modified, indicating that the method correctly identifies that the coordinates are already loaded and avoids unnecessary work. 

        Parameters:
            None

        Returns:
            None
        """
        ds = _make_cell_ds()
        cache = MPASDataCache()
        cache.load_coordinates_from_dataset(ds)

        sentinel = (np.array([999.0]), np.array([999.0]))
        cache._coordinates["default"] = sentinel

        cache.load_coordinates_from_dataset(ds)  # should be no-op
        assert cache._coordinates["default"] is sentinel

    def test_second_load_with_var_name_is_noop(self: 'TestLoadCoordinatesEarlyReturn') -> None:
        """
        This test verifies that if load_coordinates_from_dataset is called a second time with the same dataset and variable name, it does not reload the coordinates and instead returns early. By overwriting the cached coordinates for the specific variable with a sentinel value before the second call, we can confirm that the original coordinates are not modified, indicating that the method correctly identifies that the coordinates for that variable are already loaded and avoids unnecessary work. 

        Parameters:
            None

        Returns:
            None
        """
        ds = _make_cell_ds()
        cache = MPASDataCache()
        cache.load_coordinates_from_dataset(ds, "temperature")

        sentinel = (np.array([777.0]), np.array([777.0]))
        cache._coordinates["temperature"] = sentinel

        cache.load_coordinates_from_dataset(ds, "temperature")
        assert cache._coordinates["temperature"] is sentinel


class TestLoadCoordinatesDimBranches:
    """ Test the load_coordinates_from_dataset method for different coordinate-name selection branches, including nVertices, nEdges, and the default case. """

    def test_nvertices_variable_uses_lonvertex_latvertex(self: 'TestLoadCoordinatesDimBranches') -> None:
        """
        This test verifies that for a variable with the nVertices dimension, the coordinates are correctly loaded from the lonVertex and latVertex variables in the dataset. By creating a dataset with a variable that has the nVertices dimension and corresponding lonVertex and latVertex coordinate variables, we can confirm that the load_coordinates_from_dataset method correctly identifies the nVertices dimension and loads the appropriate coordinates. The test checks that the shapes of the loaded longitude and latitude arrays match the expected size of the nVertices dimension, ensuring that the correct coordinate variables are used for vertex-based data. 

        Parameters:
            None

        Returns:
            None
        """
        ds = _make_vertex_ds()
        cache = MPASDataCache()
        cache.load_coordinates_from_dataset(ds, "vorticity")
        lon, lat = cache.get_coordinates("vorticity")
        assert lon.shape == (N_VERT,)
        assert lat.shape == (N_VERT,)

    def test_nedges_variable_uses_lonedge_latedge(self: 'TestLoadCoordinatesDimBranches') -> None:
        """
        This test verifies that for a variable with the nEdges dimension, the coordinates are correctly loaded from the lonEdge and latEdge variables in the dataset. By creating a dataset with a variable that has the nEdges dimension and corresponding lonEdge and latEdge coordinate variables, we can confirm that the load_coordinates_from_dataset method correctly identifies the nEdges dimension and loads the appropriate coordinates. The test checks that the shapes of the loaded longitude and latitude arrays match the expected size of the nEdges dimension, ensuring that the correct coordinate variables are used for edge-based data. 

        Parameters:
            None

        Returns:
            None
        """
        ds = _make_edge_ds()
        cache = MPASDataCache()
        cache.load_coordinates_from_dataset(ds, "u_normal")
        lon, lat = cache.get_coordinates("u_normal")
        assert lon.shape == (N_EDGES,)
        assert lat.shape == (N_EDGES,)

    def test_else_branch_falls_back_to_loncell_latcell(self: 'TestLoadCoordinatesDimBranches') -> None:
        """
        This test verifies that for a variable that does not have nCells, nVertices, or nEdges dimensions, the load_coordinates_from_dataset method falls back to using the lonCell and latCell coordinate variables. By creating a dataset with a variable that has dimensions other than nCells, nVertices, or nEdges (e.g., nLevels) and including lonCell and latCell coordinate variables, we can confirm that the method correctly identifies the absence of the specific dimensions and loads the coordinates from lonCell and latCell. The test checks that the shapes of the loaded longitude and latitude arrays match the expected size of the nCells dimension, ensuring that the fallback mechanism works as intended for variables without specific grid dimensions. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "scalar_var": (["Time", "nLevels"], np.ones((N_TIME, 4))),
            "lonCell":    (["nCells"], np.linspace(0.1, 1.5, N_CELLS)),
            "latCell":    (["nCells"], np.linspace(0.2, 0.8, N_CELLS)),
        })

        cache = MPASDataCache()
        cache.load_coordinates_from_dataset(ds, "scalar_var")
        lon, lat = cache.get_coordinates("scalar_var")
        assert lon.shape == (N_CELLS,)

    def test_nvertices_lon_lat_in_degrees_after_load(self: 'TestLoadCoordinatesDimBranches') -> None:
        """
        This test verifies that for a variable with the nVertices dimension, the loaded longitude and latitude coordinates are in degrees after being loaded from the dataset. By creating a dataset with a variable that has the nVertices dimension and corresponding lonVertex and latVertex coordinate variables with values in radians, we can confirm that the load_coordinates_from_dataset method correctly converts these values to degrees when loading them into the cache. The test checks that all longitude values are within the range of -180 to 180 degrees, ensuring that the conversion from radians to degrees is performed correctly for vertex-based coordinates. 

        Parameters:
            None

        Returns:
            None
        """
        ds = _make_vertex_ds()
        cache = MPASDataCache()
        cache.load_coordinates_from_dataset(ds, "vorticity")
        lon, lat = cache.get_coordinates("vorticity")
        assert np.all(lon >= -180) and np.all(lon <= 180)


class TestLoadCoordinatesMissingVars:
    """ Test the load_coordinates_from_dataset method when required coordinate variables are missing. """

    def test_missing_loncell_latcell_raises(self: 'TestLoadCoordinatesMissingVars') -> None:
        """
        This test verifies that a ValueError is raised when the lonCell and latCell variables are missing from the dataset. By creating a dataset that lacks the lonCell and latCell coordinate variables and attempting to load coordinates for a variable that does not have nVertices or nEdges dimensions, we can confirm that the load_coordinates_from_dataset method correctly identifies the absence of the required coordinate variables and raises an appropriate error. The test checks for the presence of a ValueError with a message indicating that the necessary coordinate variables were not found in the dataset, ensuring that users are informed about the missing data needed for coordinate loading. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "temperature": (["Time", "nCells"], np.ones((N_TIME, N_CELLS))),
        })

        cache = MPASDataCache()

        with pytest.raises(ValueError, match="not found in dataset"):
            cache.load_coordinates_from_dataset(ds)

    def test_missing_lonvertex_latvertex_raises(self: 'TestLoadCoordinatesMissingVars') -> None:
        """
        This test verifies that a ValueError is raised when the lonVertex and latVertex variables are missing from the dataset. By creating a dataset that lacks the lonVertex and latVertex coordinate variables and attempting to load coordinates for a variable that has the nVertices dimension, we can confirm that the load_coordinates_from_dataset method correctly identifies the absence of the required coordinate variables and raises an appropriate error. The test checks for the presence of a ValueError with a message indicating that the necessary coordinate variables were not found in the dataset, ensuring that users are informed about the missing data needed for coordinate loading. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "vorticity": (["Time", "nVertices"], np.ones((N_TIME, N_VERT))),
        })

        cache = MPASDataCache()

        with pytest.raises(ValueError, match="not found in dataset"):
            cache.load_coordinates_from_dataset(ds, "vorticity")

    def test_missing_lonedge_latedge_raises(self: 'TestLoadCoordinatesMissingVars') -> None:
        """
        This test verifies that a ValueError is raised when the lonEdge and latEdge variables are missing from the dataset. By creating a dataset that lacks the lonEdge and latEdge coordinate variables and attempting to load coordinates for a variable that has the nEdges dimension, we can confirm that the load_coordinates_from_dataset method correctly identifies the absence of the required coordinate variables and raises an appropriate error. The test checks for the presence of a ValueError with a message indicating that the necessary coordinate variables were not found in the dataset, ensuring that users are informed about the missing data needed for coordinate loading. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "u_normal": (["Time", "nEdges"], np.ones((N_TIME, N_EDGES))),
        })

        cache = MPASDataCache()

        with pytest.raises(ValueError, match="not found in dataset"):
            cache.load_coordinates_from_dataset(ds, "u_normal")


class TestLoadCoordinatesEviction:
    """ Test the eviction mechanism of MPASDataCache when the cache reaches its maximum capacity. """

    def test_eviction_triggered_at_max_coordinates(self: 'TestLoadCoordinatesEviction') -> None:
        """
        This test verifies that when the MPASDataCache reaches its maximum capacity for stored coordinates, adding a new set of coordinates triggers the eviction mechanism to remove the least recently accessed coordinates. By creating a dataset with multiple variables and loading their coordinates into the cache while setting a low max_coordinates limit, we can confirm that once the limit is exceeded, the cache evicts the least accessed coordinates to make room for the new ones. The test checks that after loading more variables than the max_coordinates limit, only the most recently accessed coordinates remain in the cache, ensuring that the eviction mechanism functions as intended to manage memory usage effectively. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "var_a": (["Time", "nCells"], np.ones((N_TIME, N_CELLS))),
            "var_b": (["Time", "nCells"], np.ones((N_TIME, N_CELLS))),
            "var_c": (["Time", "nCells"], np.ones((N_TIME, N_CELLS))),
            "lonCell": (["nCells"], np.linspace(0.1, 1.5, N_CELLS)),
            "latCell": (["nCells"], np.linspace(0.2, 0.8, N_CELLS)),
        })

        cache = MPASDataCache(max_coordinates=2)
        cache.load_coordinates_from_dataset(ds, "var_a")
        cache.load_coordinates_from_dataset(ds, "var_b")
        assert len(cache._coordinates) == 2

        cache.load_coordinates_from_dataset(ds, "var_c")
        assert len(cache._coordinates) == 2
        assert "var_c" in cache._coordinates

    def test_most_accessed_coord_survives_eviction(self: 'TestLoadCoordinatesEviction') -> None:
        """
        This test verifies that when the eviction mechanism is triggered due to reaching maximum coordinate capacity, the most accessed coordinates are retained in the cache while the least accessed ones are evicted. By loading multiple sets of coordinates into the cache and accessing one of them multiple times to increase its access count, we can confirm that when a new set of coordinates is added and eviction occurs, the least accessed coordinates are removed while the most accessed ones remain. The test checks that after eviction, the most accessed coordinates are still present in the cache, ensuring that the eviction mechanism prioritizes retaining frequently used data. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "var_a": (["Time", "nCells"], np.ones((N_TIME, N_CELLS))),
            "var_b": (["Time", "nCells"], np.ones((N_TIME, N_CELLS))),
            "var_c": (["Time", "nCells"], np.ones((N_TIME, N_CELLS))),
            "lonCell": (["nCells"], np.linspace(0.1, 1.5, N_CELLS)),
            "latCell": (["nCells"], np.linspace(0.2, 0.8, N_CELLS)),
        })

        cache = MPASDataCache(max_coordinates=2)
        cache.load_coordinates_from_dataset(ds, "var_a")
        cache.load_coordinates_from_dataset(ds, "var_b")

        for _ in range(5):
            cache.get_coordinates("var_a")

        cache.load_coordinates_from_dataset(ds, "var_c")
        assert "var_a" in cache._coordinates
        assert "var_c" in cache._coordinates
        assert "var_b" not in cache._coordinates

    def test_evict_least_accessed_coordinates_direct(self: 'TestLoadCoordinatesEviction') -> None:
        """
        This test verifies that the _evict_least_accessed_coordinates method correctly evicts the least accessed coordinates when called directly. By creating a dataset with multiple variables and loading their coordinates into the cache while setting a low max_coordinates limit, we can confirm that when the eviction method is called, it identifies and removes the least accessed coordinates from the cache. The test checks that after calling the eviction method, only the most recently accessed coordinates remain in the cache, ensuring that the eviction mechanism functions as intended to manage memory usage effectively even when invoked directly. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "var_a": (["Time", "nCells"], np.ones((N_TIME, N_CELLS))),
            "var_b": (["Time", "nCells"], np.ones((N_TIME, N_CELLS))),
            "lonCell": (["nCells"], np.linspace(0.1, 1.5, N_CELLS)),
            "latCell": (["nCells"], np.linspace(0.2, 0.8, N_CELLS)),
        })

        cache = MPASDataCache(max_coordinates=2)

        cache.load_coordinates_from_dataset(ds, "var_a")
        cache.load_coordinates_from_dataset(ds, "var_b")
        cache._evict_least_accessed_coordinates()

        assert len(cache._coordinates) == 1

    def test_evict_coordinates_empty_is_noop(self: 'TestLoadCoordinatesEviction') -> None:
        """
        This test verifies that when the _evict_least_accessed_coordinates method is called on an empty cache (i.e., when there are no coordinates stored), it returns immediately without raising an error or modifying the cache. By creating an instance of MPASDataCache and calling the eviction method without loading any coordinates, we can confirm that the method handles this edge case gracefully. The test checks that after calling the eviction method on an empty cache, the cache remains unchanged and no exceptions are raised, ensuring that the method is robust against being called in scenarios where there are no coordinates to evict. 

        Parameters:
            None

        Returns:
            None
        """
        cache = MPASDataCache()
        cache._evict_least_accessed_coordinates()
        assert len(cache._coordinates) == 0


class TestGetCoordinatesKeyError:
    """ Test the get_coordinates method of MPASDataCache to ensure it raises a KeyError when attempting to access coordinates that have not been loaded into the cache. """

    def test_get_coordinates_not_loaded_raises(self: 'TestGetCoordinatesKeyError') -> None:
        """
        This test verifies that a KeyError is raised when attempting to get coordinates that have not been loaded into the cache. By creating an instance of MPASDataCache and calling the get_coordinates method without first loading any coordinates, we can confirm that the method correctly identifies that the requested coordinates are not available in the cache and raises an appropriate error. The test checks for the presence of a KeyError with a message indicating that the coordinates were not loaded, ensuring that users are informed about the missing data when they attempt to access coordinates that have not been cached. 

        Parameters:
            None

        Returns:
            None
        """
        cache = MPASDataCache()
        with pytest.raises(KeyError, match="not loaded"):
            cache.get_coordinates()

    def test_get_coordinates_wrong_var_raises(self: 'TestGetCoordinatesKeyError') -> None:
        """
        This test verifies that a KeyError is raised when attempting to get coordinates for a variable that has not been loaded into the cache. By creating an instance of MPASDataCache, loading coordinates for a specific variable, and then calling the get_coordinates method with a different variable name, we can confirm that the method correctly identifies that the requested coordinates for the non-existent variable are not available in the cache and raises an appropriate error. The test checks for the presence of a KeyError with a message indicating that the coordinates were not loaded, ensuring that users are informed about the missing data when they attempt to access coordinates for a variable that has not been cached. 

        Parameters:
            None

        Returns:
            None
        """
        cache = MPASDataCache()
        cache.load_coordinates_from_dataset(_make_cell_ds())
        with pytest.raises(KeyError, match="not loaded"):
            cache.get_coordinates("nonexistent_var")


class TestLoadVariableDataMissingVar:
    """ Test the load_variable_data method of MPASDataCache to ensure it raises a ValueError when attempting to load variable data for a variable that is not present in the dataset. """

    def test_missing_variable_raises(self: 'TestLoadVariableDataMissingVar') -> None:
        """
        This test verifies that a ValueError is raised when attempting to load variable data for a variable that is not present in the dataset. By creating a dataset that does not contain the specified variable and calling the load_variable_data method with that variable name, we can confirm that the method correctly identifies the absence of the variable in the dataset and raises an appropriate error. The test checks for the presence of a ValueError with a message indicating that the variable was not found in the dataset, ensuring that users are informed about the missing data when they attempt to load a variable that does not exist in the provided dataset. 

        Parameters:
            None

        Returns:
            None
        """
        cache = MPASDataCache()
        with pytest.raises(ValueError, match="not found in dataset"):
            cache.load_variable_data(_make_cell_ds(), "no_such_variable")

    def test_typo_in_var_name_raises(self: 'TestLoadVariableDataMissingVar') -> None:
        """
        This test verifies that a ValueError is raised when there is a typo in the variable name provided to the load_variable_data method. By creating a dataset that contains a variable with a specific name and then calling the method with a misspelled version of that name, we can confirm that the method correctly identifies that the variable is not present in the dataset due to the typo and raises an appropriate error. The test checks for the presence of a ValueError with a message indicating that the variable was not found in the dataset, ensuring that users are informed about the missing data when they attempt to load a variable with an incorrect name. 

        Parameters:
            None

        Returns:
            None
        """
        cache = MPASDataCache()
        with pytest.raises(ValueError, match="not found in dataset"):
            cache.load_variable_data(_make_cell_ds(), "temperatur")  # missing 'e'


class TestGetVariableDataKeyError:
    """ Test the get_variable_data method of MPASDataCache to ensure it raises a KeyError when attempting to access variable data that has not been loaded into the cache. """

    def test_get_variable_not_loaded_raises(self: 'TestGetVariableDataKeyError') -> None:
        """
        This test verifies that a KeyError is raised when attempting to get variable data for a variable that has not been loaded into the cache. By creating an instance of MPASDataCache and calling the get_variable_data method with a variable name that has not been loaded, we can confirm that the method correctly identifies that the requested variable data is not available in the cache and raises an appropriate error. The test checks for the presence of a KeyError with a message indicating that the variable data was not loaded, ensuring that users are informed about the missing data when they attempt to access variable data that has not been cached. 

        Parameters:
            None

        Returns:
            None
        """
        cache = MPASDataCache()
        with pytest.raises(KeyError, match="not loaded"):
            cache.get_variable_data("temperature")

    def test_get_variable_wrong_time_index_raises(self: 'TestGetVariableDataKeyError') -> None:
        """
        This test verifies that a KeyError is raised when attempting to get variable data for a valid variable name but with a time index that has not been loaded into the cache. By creating an instance of MPASDataCache, loading variable data for a specific variable and time index, and then calling the get_variable_data method with the same variable name but a different time index, we can confirm that the method correctly identifies that the requested variable data for the specified time index is not available in the cache and raises an appropriate error. The test checks for the presence of a KeyError with a message indicating that the variable data was not loaded, ensuring that users are informed about the missing data when they attempt to access variable data for a time index that has not been cached. 

        Parameters:
            None

        Returns:
            None
        """
        cache = MPASDataCache()
        cache.load_variable_data(_make_cell_ds(), "temperature", time_index=0)
        with pytest.raises(KeyError, match="not loaded"):
            cache.get_variable_data("temperature", time_index=1)


class TestEvictLeastAccessedEmpty:
    """ Test the _evict_least_accessed method of MPASDataCache to ensure it returns early when the cache is empty. """

    def test_evict_on_empty_cache_is_noop(self: 'TestEvictLeastAccessedEmpty') -> None:
        """
        This test verifies that when the _evict_least_accessed method is called on an empty cache (i.e., when there are no variables stored), it returns immediately without raising an error or modifying the cache. By creating an instance of MPASDataCache and calling the eviction method without loading any variables, we can confirm that the method handles this edge case gracefully. The test checks that after calling the eviction method on an empty cache, the cache remains unchanged and no exceptions are raised, ensuring that the method is robust against being called in scenarios where there are no variables to evict. 

        Parameters:
            None

        Returns:
            None
        """
        cache = MPASDataCache()
        cache._evict_least_accessed()
        assert len(cache._variables) == 0

    def test_evict_on_empty_cache_does_not_raise(self: 'TestEvictLeastAccessedEmpty') -> None:
        """
        This test verifies that when the _evict_least_accessed method is called on an empty cache, it does not raise any exceptions. By creating an instance of MPASDataCache and calling the eviction method without loading any variables, we can confirm that the method is designed to handle this scenario without errors. The test checks that no exceptions are raised during the call, ensuring that the method is robust and can be safely invoked even when there are no variables in the cache to evict. 

        Parameters:
            None

        Returns:
            None
        """
        cache = MPASDataCache()
        cache._evict_least_accessed()  


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

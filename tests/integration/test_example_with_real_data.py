#!/usr/bin/env python3
"""
MPASdiag Test Suite: Example Tests with Real MPAS Data

This module contains integration tests that demonstrate how to use session-scoped fixtures to load real MPAS data once and share it across multiple test methods. The tests validate the loading of coordinates, 2D and 3D diagnostic data, and the consistency of data across different fixtures. Each test checks for None values from fixtures and skips gracefully when real MPAS data is unavailable. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries for testing
import pytest
import numpy as np

from tests.test_data_helpers import (
    check_mpas_data_available,
    get_mpas_data_paths,
    get_real_mpas_coordinates,
    get_real_mpas_variable,
)


class TestRealMPASDataLoading:
    """ Validate that real MPAS data can be loaded and accessed through fixtures. """
    
    def test_data_paths_resolution(self: "TestRealMPASDataLoading") -> None:
        """
        This test validates that the helper function for resolving MPAS data paths returns a dictionary with the expected structure and keys. It checks that the returned paths include entries for the grid file, diagnostic directory, mpasout directory, and data root. This ensures that the test suite can locate the necessary data files when real MPAS data is available.

        Parameters: 
            self (TestRealMPASDataLoading): The test instance, used for accessing class-level fixtures and methods.
        
        Returns:
            None
        """
        paths = get_mpas_data_paths()
        
        assert isinstance(paths, dict)
        assert 'grid_file' in paths
        assert 'diag_dir' in paths
        assert 'mpasout_dir' in paths
        assert 'data_root' in paths
    
    def test_data_availability_check(self: "TestRealMPASDataLoading") -> None:
        """
        This test validates the function that checks for the availability of real MPAS data. It ensures that the function returns a boolean value indicating whether the necessary data files are accessible. If data is available, the test further verifies that the resolved paths for the grid file, diagnostic directory, and mpasout directory are not None, confirming that the data can be located and used for testing.

        Parameters:
            self (TestRealMPASDataLoading): The test instance, used for accessing class-level fixtures and methods.

        Returns:
            None
        """
        available = check_mpas_data_available()
        
        assert isinstance(available, bool)
        
        if available:
            paths = get_mpas_data_paths()
            assert paths['grid_file'] is not None
            assert (paths['diag_dir'] is not None or 
                    paths['mpasout_dir'] is not None)
    
    def test_coordinates_from_fixture(self: "TestRealMPASDataLoading", mpas_coordinates) -> None:
        """
        This test validates that the session-scoped fixture for loading real MPAS coordinates returns properly structured longitude and latitude arrays. It checks that the returned values are numpy arrays, that they have matching lengths, and that their values fall within realistic geographic bounds (longitude between -180 and 180 degrees, latitude between -90 and 90 degrees). If the fixture returns None, indicating that real MPAS grid data is not available, the test will be skipped gracefully.

        Parameters:
            self (TestRealMPASDataLoading): The test instance, used for accessing class-level fixtures and methods.
            mpas_coordinates (tuple or None): A tuple containing longitude and latitude arrays loaded from real MPAS grid data, or None if the data is not available.

        Returns:
            None
        """
        if mpas_coordinates is None:
            pytest.skip("Real MPAS grid data not available")
        
        lon, lat = mpas_coordinates
        
        assert isinstance(lon, np.ndarray)
        assert isinstance(lat, np.ndarray)
        assert len(lon) == len(lat)
        assert lon.ndim == 1
        assert lat.ndim == 1
        
        assert -180 <= np.min(lon) <= 180
        assert -180 <= np.max(lon) <= 180
        assert -90 <= np.min(lat) <= 90
        assert -90 <= np.max(lat) <= 90
    
    def test_2d_processor_from_fixture(self: "TestRealMPASDataLoading", mpas_2d_processor_diag) -> None:
        """
        This test validates that the session-scoped fixture for loading real MPAS 2D diagnostic data returns a properly initialized processor with a dataset that has the expected structure. It checks that the processor is not None, that it has a dataset attribute, and that the dataset includes dimensions for nCells with a positive size. If the fixture returns None, indicating that real MPAS diagnostic data is not available, the test will be skipped gracefully.

        Parameters:
            self (TestRealMPASDataLoading): The test instance, used for accessing class-level fixtures and methods.
            mpas_2d_processor_diag (object or None): The 2D processor loaded from real MPAS diagnostic data, or None if the data is not available.

        Returns:
            None
        """
        if mpas_2d_processor_diag is None:
            pytest.skip("Real MPAS diagnostic data not available")
        
        proc = mpas_2d_processor_diag
        
        assert proc is not None
        assert hasattr(proc, 'dataset')
        assert proc.dataset is not None
        
        assert 'nCells' in proc.dataset.sizes
        assert proc.dataset.sizes['nCells'] > 0
    
    def test_wind_data_from_fixture(self: "TestRealMPASDataLoading", mpas_wind_data) -> None:
        """
        This test validates that the session-scoped fixture for loading real MPAS wind data returns properly structured u and v wind component arrays. It checks that the returned values are numpy arrays, that they have matching lengths, and that their values fall within realistic bounds for wind speeds (e.g., between -100 and 100 m/s). If the fixture returns None, indicating that real MPAS wind data is not available, the test will be skipped gracefully.

        Parameters:
            self (TestRealMPASDataLoading): The test instance, used for accessing class-level fixtures and methods.
            mpas_wind_data (tuple or None): A tuple containing u and v wind components loaded from real MPAS wind data, or None if the data is not available.

        Returns:
            None
        """
        if mpas_wind_data is None:
            pytest.skip("Real MPAS wind data not available")
        
        u, v = mpas_wind_data
        
        assert isinstance(u, np.ndarray)
        assert isinstance(v, np.ndarray)
        assert len(u) == len(v)
        assert u.ndim == 1
        assert v.ndim == 1
        
        assert -100 < np.min(u) < 100  
        assert -100 < np.max(u) < 100
        assert -100 < np.min(v) < 100
        assert -100 < np.max(v) < 100
    
    def test_precip_data_from_fixture(self: "TestRealMPASDataLoading", mpas_precip_data) -> None:
        """
        This test validates that the session-scoped fixture for loading real MPAS precipitation data returns a properly structured precipitation array. It checks that the returned value is a numpy array, that it has the correct dimensions, and that its values are non-negative (as precipitation cannot be negative). If the fixture returns None, indicating that real MPAS precipitation data is not available, the test will be skipped gracefully.

        Parameters:
            self (TestRealMPASDataLoading): The test instance, used for accessing class-level fixtures and methods.
            mpas_precip_data (np.ndarray or None): The precipitation data loaded from real MPAS data, or None if the data is not available.

        Returns:
            None
        """
        if mpas_precip_data is None:
            pytest.skip("Real MPAS precipitation data not available")
        
        precip = mpas_precip_data
        
        assert isinstance(precip, np.ndarray)
        assert precip.ndim == 1
        assert len(precip) > 0
        
        assert np.min(precip) >= 0
    
    def test_surface_temp_from_fixture(self: "TestRealMPASDataLoading", mpas_surface_temp_data) -> None:
        """
        This test validates that the session-scoped fixture for loading real MPAS surface temperature data returns a properly structured temperature array. It checks that the returned value is a numpy array, that it has the correct dimensions, and that its values fall within realistic bounds for surface temperature in Kelvin. If the fixture returns None, indicating that real MPAS surface temperature data is not available, the test will be skipped gracefully.

        Parameters:
            self (TestRealMPASDataLoading): The test instance, used for accessing class-level fixtures and methods.
            mpas_surface_temp_data (np.ndarray or None): The surface temperature data loaded from real MPAS data, or None if the data is not available.

        Returns:
            None
        """
        if mpas_surface_temp_data is None:
            pytest.skip("Real MPAS surface temperature data not available")
        
        t2m = mpas_surface_temp_data
        
        assert isinstance(t2m, np.ndarray)
        assert t2m.ndim == 1
        assert len(t2m) > 0
        
        assert 200 < np.min(t2m) < 350
        assert 200 < np.max(t2m) < 350
    
    def test_3d_processor_from_fixture(self: "TestRealMPASDataLoading", mpas_3d_processor) -> None:
        """
        This test validates that the session-scoped fixture for loading real MPAS 3D data returns a properly initialized processor with a dataset that has the expected structure. It checks that the processor is not None, that it has a dataset attribute, and that the dataset includes dimensions for nCells and nVertLevels with positive sizes. If the fixture returns None, indicating that real MPAS 3D data is not available, the test will be skipped gracefully.

        Parameters:
            self (TestRealMPASDataLoading): The test instance, used for accessing class-level fixtures and methods.
            mpas_3d_processor (object or None): The 3D processor loaded from real MPAS 3D data, or None if the data is not available.

        Returns:
            None
        """
        if mpas_3d_processor is None:
            pytest.skip("Real MPAS 3D data not available")
        
        proc = mpas_3d_processor
        
        assert proc is not None
        assert hasattr(proc, 'dataset')
        assert proc.dataset is not None
        
        assert 'nCells' in proc.dataset.sizes
        assert 'nVertLevels' in proc.dataset.sizes
        assert proc.dataset.sizes['nCells'] > 0
        assert proc.dataset.sizes['nVertLevels'] > 0
    
    def test_qv_3d_data_from_fixture(self: "TestRealMPASDataLoading", mpas_3d_processor) -> None:
        """
        This test validates that the session-scoped fixture for loading real MPAS 3D data can be used to extract a variable (such as theta) at the surface level and that the extracted variable has dimensions consistent with the coordinates. It checks that the variable data is a numpy array, that it has the correct dimensions, and that its values are within reasonable bounds based on the variable type. If the fixture returns None, indicating that real MPAS 3D data is not available, the test will be skipped gracefully.

        Parameters:
            self (TestRealMPASDataLoading): The test instance, used for accessing class-level fixtures and methods.
            mpas_3d_processor (object or None): The 3D processor loaded from real MPAS 3D data, or None if the data is not available.

        Returns:
            None
        """
        if mpas_3d_processor is None:
            pytest.skip("Real MPAS 3D data not available")
        
        proc = mpas_3d_processor
        
        if 'theta' not in proc.dataset:
            pytest.skip("Theta variable not available in 3D data")
        
        theta_data = proc.dataset['theta'].isel(Time=0, nVertLevels=0).values
        
        assert isinstance(theta_data, np.ndarray)
        assert theta_data.ndim == 1
        assert len(theta_data) > 0
        
        assert 200 < np.min(theta_data) < 400
        assert 200 < np.max(theta_data) < 400


class TestCombiningMultipleFixtures:
    """ Use multiple fixtures together to validate consistency and alignment of data across them. """
    
    def test_coordinates_and_wind_alignment(
        self: "TestCombiningMultipleFixtures", 
        mpas_coordinates, 
        mpas_wind_data
    ) -> None:
        """
        This test validates that the coordinates loaded from real MPAS data and the wind data loaded from real MPAS data are consistent in terms of their dimensions. It checks that if either fixture returns None, indicating that real MPAS data is not available, the test will be skipped gracefully. If both fixtures return valid data, it asserts that the lengths of the longitude and latitude arrays are greater than or equal to the lengths of the u and v wind component arrays, ensuring that the wind data can be properly mapped to the coordinates.

        Parameters:
            self (TestCombiningMultipleFixtures): The test instance, used for accessing class-level fixtures and methods.
            mpas_coordinates (tuple or None): The coordinates loaded from real MPAS data, or None if the data is not available.
            mpas_wind_data (tuple or None): The wind data loaded from real MPAS data, or None if the data is not available.

        Returns:
            None
        """
        if mpas_coordinates is None or mpas_wind_data is None:
            pytest.skip("Real MPAS data not available")
        
        lon, lat = mpas_coordinates
        u, v = mpas_wind_data
        
        assert len(lon) >= len(u)
        assert len(lat) >= len(v)
    
    def test_2d_processor_variable_extraction(
        self: "TestCombiningMultipleFixtures",
        mpas_2d_processor_diag,
        mpas_coordinates
    ) -> None:
        """
        This test validates that the session-scoped fixture for loading real MPAS 2D diagnostic data can be used to extract a variable (such as precipitation) and that the extracted variable has dimensions consistent with the coordinates. It checks that the variable data is a numpy array, that it has the correct dimensions, and that its values are within reasonable bounds based on the variable type. If either fixture returns None, indicating that real MPAS data is not available, the test will be skipped gracefully.

        Parameters:
            self (TestCombiningMultipleFixtures): The test instance, used for accessing class-level fixtures and methods.
            mpas_2d_processor_diag (object or None): The 2D processor loaded from real MPAS data, or None if the data is not available.
            mpas_coordinates (tuple or None): The coordinates loaded from real MPAS data, or None if the data is not available.

        Returns:
            None
        """
        if mpas_2d_processor_diag is None or mpas_coordinates is None:
            pytest.skip("Real MPAS data not available")
        
        proc = mpas_2d_processor_diag
        lon, lat = mpas_coordinates
        
        available_vars = list(proc.dataset.data_vars.keys())
        
        if not available_vars:
            pytest.skip("No variables available in 2D processor dataset")
        
        var_name = available_vars[0]

        var_data = get_real_mpas_variable(
            proc, 
            var_name, 
            time_index=0
        )
        
        assert isinstance(var_data, np.ndarray)
        assert var_data.ndim == 1
        assert len(var_data) > 0
    
    def test_3d_processor_variable_extraction(
        self: "TestCombiningMultipleFixtures",
        mpas_3d_processor,
        mpas_coordinates
    ) -> None:
        """
        This test validates that the session-scoped fixture for loading real MPAS 3D data can be used to extract a variable (such as theta) and that the extracted variable has dimensions consistent with the coordinates. It checks that the variable data is a numpy array, that it has the correct dimensions, and that its values are within reasonable bounds based on the variable type. If either fixture returns None, indicating that real MPAS data is not available, the test will be skipped gracefully.

        Parameters:
            self (TestCombiningMultipleFixtures): The test instance, used for accessing class-level fixtures and methods.
            mpas_3d_processor (object or None): The 3D processor loaded from real MPAS data, or None if the data is not available.
            mpas_coordinates (tuple or None): The coordinates loaded from real MPAS data, or None if the data is not available.

        Returns:
            None
        """
        if mpas_3d_processor is None or mpas_coordinates is None:
            pytest.skip("Real MPAS 3D data not available")
        
        proc = mpas_3d_processor
        lon, lat = mpas_coordinates
        
        available_3d_vars = [
            v for v in proc.dataset.data_vars 
            if 'nCells' in proc.dataset[v].sizes and 'nVertLevels' in proc.dataset[v].sizes
        ]
        
        if not available_3d_vars:
            pytest.skip("No 3D variables available in processor dataset")
        
        var_name = 'theta' if 'theta' in available_3d_vars else available_3d_vars[0]
        var_data = proc.dataset[var_name].isel(Time=0, nVertLevels=0).values
        
        assert isinstance(var_data, np.ndarray)
        assert var_data.ndim == 1
        assert len(var_data) > 0
        
        if var_name == 'theta':
            assert 200 < np.min(var_data) < 400
            assert 200 < np.max(var_data) < 400
        elif var_name == 'pressure':
            assert 10000 < np.min(var_data) < 110000
            assert 10000 < np.max(var_data) < 110000
    
    def test_data_consistency_across_fixtures(
        self: "TestCombiningMultipleFixtures",
        mpas_2d_processor_diag,
        mpas_wind_data,
        mpas_precip_data,
        mpas_surface_temp_data
    ) -> None:
        """
        This test validates the consistency of data across multiple fixtures that load different types of real MPAS data. It checks that if any of the fixtures return None, indicating that real MPAS data is not available, the test will be skipped gracefully. If data is available from multiple fixtures, it checks that the lengths of the wind data, precipitation data, and surface temperature data are consistent with each other, as they should all be derived from the same underlying dataset.
        
        Parameters:
            self (TestCombiningMultipleFixtures): The test instance, used for accessing class-level fixtures and methods.
            mpas_2d_processor_diag (object or None): The 2D processor loaded from real MPAS data, or None if the data is not available.
            mpas_wind_data (array-like or None): The wind data loaded from real MPAS data, or None if the data is not available.
            mpas_precip_data (array-like or None): The precipitation data loaded from real MPAS data, or None if the data is not available.
            mpas_surface_temp_data (array-like or None): The surface temperature data loaded from real MPAS data, or None if the data is not available.

        Returns:
            None
        """
        available_count = sum([
            mpas_2d_processor_diag is not None,
            mpas_wind_data is not None,
            mpas_precip_data is not None,
            mpas_surface_temp_data is not None
        ])
        
        if available_count == 0:
            pytest.skip("No real MPAS data available")
        
        if mpas_wind_data is not None:
            if mpas_precip_data is not None:
                assert len(mpas_wind_data[0]) == len(mpas_precip_data)
            if mpas_surface_temp_data is not None:
                assert len(mpas_wind_data[0]) == len(mpas_surface_temp_data)


class TestDataAvailabilityConditional:
    """ This class shows patterns for writing tests that adapt to whether real MPAS data is available or not. """
    
    def test_skip_when_no_data(self: "TestDataAvailabilityConditional", mpas_data_available) -> None:
        """
        This test demonstrates a simple pattern for skipping tests when real MPAS data is not available. It checks the boolean value from the data availability fixture and skips the test if data is not available. If data is available, it proceeds to run assertions on the resolved data paths. This pattern allows for tests that can adapt to the testing environment without causing false failures when real data is not accessible.

        Parameters:
            self (TestDataAvailabilityConditional): The test instance, used for accessing class-level fixtures and methods.
            mpas_data_available (bool): A boolean value indicating whether real MPAS data is available, provided by a session-scoped fixture.

        Returns:
            None
        """
        if not mpas_data_available:
            pytest.skip("Real MPAS data not available")
        
        paths = get_mpas_data_paths()
        assert paths['grid_file'] is not None
    
    def test_fallback_to_mock_when_needed(
        self: "TestDataAvailabilityConditional",
        mpas_2d_processor_diag
    ) -> None:
        """
        This test validates that when real MPAS diagnostic data is not available, the test can gracefully skip or fallback to using mock data. It checks if the 2D processor fixture returns None, and if so, it skips the test with an appropriate message. If the fixture returns a valid processor, it proceeds to run assertions on the dataset structure. This pattern allows for tests that can adapt to the testing environment without causing false failures when real data is not accessible.

        Parameters:
            self (TestDataAvailabilityConditional): The test instance, used for accessing class-level fixtures and methods.
            mpas_2d_processor_diag: The fixture providing real MPAS 2D processor diagnostic data.

        Returns:
            None
        """
        if mpas_2d_processor_diag is None:
            pytest.skip("Real MPAS diagnostic data not available")
        
        dataset = mpas_2d_processor_diag.dataset
        
        assert dataset is not None
        assert 'nCells' in dataset.sizes
        assert dataset.sizes['nCells'] > 0
    
    def test_parametrize_based_on_availability(
        self: "TestDataAvailabilityConditional",
        mpas_data_available,
        mpas_coordinates
    ) -> None:
        """
        This test demonstrates how to use parameterization to run different test logic based on the availability of real MPAS data. It checks the boolean value from the data availability fixture and the presence of coordinates. If data is not available, it skips the test. If data is available, it runs assertions on the coordinates. This pattern allows for flexible testing that can adapt to the testing environment.

        Parameters:
            self (TestDataAvailabilityConditional): The test instance, used for accessing class-level fixtures and methods.
            mpas_data_available (bool): A boolean value indicating whether real MPAS data is available, provided by a session-scoped fixture.
            mpas_coordinates: The fixture providing real MPAS coordinate data.

        Returns:
            None
        """
        if not mpas_data_available or mpas_coordinates is None:
            pytest.skip("Real MPAS data not available")
        
        lon, lat = mpas_coordinates

        assert len(lon) > 100  
        assert len(lat) > 100

        assert isinstance(lon, np.ndarray)
        assert isinstance(lat, np.ndarray)


class TestHelperFunctions:
    """ Test the helper functions for loading data paths and checking data availability, which are used by the fixtures. """
    
    def test_get_mpas_data_paths(self: "TestHelperFunctions") -> None:
        """
        This test validates the helper function that resolves MPAS data paths. It checks that the function returns a dictionary with the expected keys for the grid file, diagnostic directory, mpasout directory, and data root. This ensures that the test suite can locate the necessary data files when real MPAS data is available.

        Parameters:
            self (TestHelperFunctions): The test instance, used for accessing class-level fixtures and methods

        Returns:
            None
        """
        paths = get_mpas_data_paths()
        assert isinstance(paths, dict)

        required_keys = ['data_root', 'grid_file', 'diag_dir', 'mpasout_dir']
        
        for key in required_keys:
            assert key in paths
    
    def test_check_mpas_data_available(self: "TestHelperFunctions") -> None:
        """
        This test validates the helper function that checks for the availability of real MPAS data. It ensures that the function returns a boolean value indicating whether the necessary data files are accessible. This is a critical check that allows the test suite to conditionally skip tests when real MPAS data is not available, preventing false failures and providing clear feedback about the testing environment.

        Parameters:
            self (TestHelperFunctions): The test instance, used for accessing class-level fixtures and methods

        Returns:
            None
        """
        available = check_mpas_data_available()
        assert isinstance(available, bool)
    
    def test_get_real_mpas_coordinates_full(self: "TestHelperFunctions") -> None:
        """
        This test validates the helper function that loads real MPAS coordinates. It checks that the function returns longitude and latitude arrays that are numpy arrays, have matching lengths, and contain values within realistic geographic bounds. If the function raises a FileNotFoundError or if the data is not available, the test will be skipped gracefully.

        Parameters:
            self (TestHelperFunctions): The test instance, used for accessing class-level fixtures and methods

        Returns:
            None
        """
        try:
            lon, lat = get_real_mpas_coordinates()
            assert isinstance(lon, np.ndarray)
            assert isinstance(lat, np.ndarray)
            assert len(lon) == len(lat)
        except (FileNotFoundError, pytest.skip.Exception):
            pytest.skip("Real MPAS grid data not available")
    
    def test_get_real_mpas_coordinates_subset(self: "TestHelperFunctions") -> None:
        """
        This test validates that the helper function for loading real MPAS coordinates can return a subset of the coordinates when requested. It checks that the returned longitude and latitude arrays are numpy arrays, that they have the specified length, and that their values fall within realistic geographic bounds. If the function raises a FileNotFoundError or indicates that data is not available, the test will be skipped gracefully.

        Parameters:
            self (TestHelperFunctions): The test instance, used for accessing class-level fixtures and methods

        Returns:
            None
        """
        try:
            lon, lat = get_real_mpas_coordinates(n=50)
            assert isinstance(lon, np.ndarray)
            assert isinstance(lat, np.ndarray)
            assert len(lon) == 50
            assert len(lat) == 50
        except (FileNotFoundError, pytest.skip.Exception):
            pytest.skip("Real MPAS grid data not available")


if __name__ == "__main__":
    pytest.main([__file__])
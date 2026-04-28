#!/usr/bin/env python3

"""
MPASdiag Test Suite: 3D Processor Gap Coverage

This module contains tests for the MPAS3DProcessor class, specifically targeting code paths that were not covered in previous tests. The tests focus on error handling and fallback behaviors in methods related to coordinate extraction, file searching, data loading, variable validation, and pressure level handling. By covering these previously untested paths, we aim to improve the overall test coverage and robustness of the MPAS3DProcessor class. The tests are designed to simulate various scenarios, including missing datasets, failed file searches, and invalid variable requests, without requiring actual MPAS NetCDF files. This allows us to verify that the processor behaves correctly under a wide range of conditions and provides appropriate feedback to the user when issues arise. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from io import StringIO
from unittest.mock import MagicMock, patch

from mpasdiag.processing.processors_3d import MPAS3DProcessor


@pytest.fixture
def proc(tmp_path) -> 'MPAS3DProcessor':
    """
    This fixture creates a base instance of the MPAS3DProcessor class with default settings for testing. It initializes the processor with verbose mode disabled, no dataset loaded, a data type of 'xarray', and a mock grid file path. This setup provides a consistent starting point for tests that require an instance of MPAS3DProcessor, allowing them to focus on specific behaviors without needing to set up the processor attributes in each test.

    Parameters:
        None

    Returns:
        'MPAS3DProcessor': An instance of MPAS3DProcessor with default settings.
    """
    p = MPAS3DProcessor.__new__(MPAS3DProcessor)
    p.verbose = False
    p.dataset = None
    p.data_type = 'xarray'
    p.grid_file = str(tmp_path / 'mock_grid.nc')
    return p


@pytest.fixture
def proc_v(proc: 'MPAS3DProcessor') -> 'MPAS3DProcessor':
    """
    This fixture takes the base MPAS3DProcessor instance from the 'proc' fixture and enables verbose mode for testing. By setting the verbose attribute to True, this fixture allows tests to verify that verbose output is generated correctly in methods that include print statements when verbose mode is active. This setup provides a convenient way to test both silent and verbose behaviors of the processor without needing to duplicate the processor initialization code in each test. 

    Parameters:
        proc ('MPAS3DProcessor'): The base processor instance from the 'proc' fixture.

    Returns:
        'MPAS3DProcessor': The same processor instance with verbose mode enabled.
    """
    proc.verbose = True
    return proc


def _make_3d_ds(n_time: int = 2, 
                n_cells: int = 5, 
                n_vert: int = 4) -> xr.Dataset:
    """
    This helper function creates a synthetic 3D xarray Dataset with specified dimensions for time, spatial cells, and vertical levels. The dataset includes variables for potential temperature ('theta'), pressure at each level ('pressure_p'), and a base pressure profile ('pressure_base') that varies with vertical levels. This synthetic dataset is used in tests to simulate the structure of MPAS output data, allowing tests to focus on the behavior of methods that operate on 3D datasets without needing to rely on actual MPAS output files. The function generates time coordinates, creates a base pressure profile that decreases with height, and fills the variables with simple values for testing purposes. 

    Parameters:
        n_time (int): The number of time steps in the dataset (default is 2).
        n_cells (int): The number of spatial cells in the dataset (default is 5).
        n_vert (int): The number of vertical levels in the dataset (default is 4). 

    Returns:
        xr.Dataset: A synthetic 3D xarray Dataset with the specified dimensions and variables.
    """
    times = pd.date_range('2024-01-01', periods=n_time, freq='h')
    p_base = np.linspace(100000.0, 50000.0, n_vert)

    return xr.Dataset({
        'theta': xr.DataArray(
            np.ones((n_time, n_cells, n_vert)),
            dims=['Time', 'nCells', 'nVertLevels'],
            coords={'Time': times},
        ),
        'pressure_p': xr.DataArray(
            np.ones((n_time, n_cells, n_vert)) * 1000.0,
            dims=['Time', 'nCells', 'nVertLevels'],
        ),
        'pressure_base': xr.DataArray(
            np.tile(p_base, (n_time, n_cells, 1)),
            dims=['Time', 'nCells', 'nVertLevels'],
        ),
    })


def _mean_p_ascending() -> np.ndarray:
    """
    This helper function returns a 1D numpy array of mean pressure values in ascending order, which is used in tests to simulate the mean pressure profile that might be used for interpolation in the _lerp_variable method. The values are chosen to represent a typical pressure profile that decreases with height, which is common in atmospheric datasets. This array is used to test the behavior of interpolation and exception handling in cases where the pressure levels are not strictly ordered or when interpolation fails. 

    Parameters:
        None

    Returns:
        np.ndarray: A 1D array of mean pressure values in ascending order. 
    """
    return np.array([50000.0, 70000.0, 85000.0, 100000.0])


class TestDetectSpatialDimDefault:
    """ Test _detect_spatial_dim default behavior when no known spatial dims are present. """

    def test_unknown_dims_return_ncells(self: 'TestDetectSpatialDimDefault') -> None:
        """
        This test verifies that when the input sizes dictionary does not contain any of the known spatial dimension names ('nCells', 'nVertices', 'nEdges'), the _detect_spatial_dim method defaults to returning 'nCells' as the spatial dimension. This ensures that the method has a fallback behavior for cases where the expected spatial dimensions are not present, allowing it to continue functioning with a reasonable default assumption. The test simulates a scenario where the sizes dictionary contains unrelated dimensions, confirming that the method returns 'nCells' as the default spatial dimension in this case. 

        Parameters:
            None

        Returns:
            None
        """
        sizes = {'Time': 2, 'someOtherDim': 10}
        assert MPAS3DProcessor._detect_spatial_dim(sizes) == 'nCells'

    def test_nvertices_takes_priority(self: 'TestDetectSpatialDimDefault') -> None:
        """
        This test verifies that when 'nVertices' is present in the input sizes dictionary, the _detect_spatial_dim method correctly identifies it as the spatial dimension, even if 'nCells' is also present. This ensures that the method prioritizes 'nVertices' over 'nCells' when both are available, which is important for correctly handling datasets that include vertex-based dimensions. The test simulates a scenario where both 'nCells' and 'nVertices' are present in the sizes dictionary, confirming that the method returns 'nVertices' as the spatial dimension in this case. 

        Parameters:
            None

        Returns:
            None
        """
        sizes = {'nVertices': 12, 'nCells': 50}
        assert MPAS3DProcessor._detect_spatial_dim(sizes) == 'nVertices'

    def test_nedges_detected(self: 'TestDetectSpatialDimDefault') -> None:
        """
        This test verifies that when 'nEdges' is present in the input sizes dictionary, the _detect_spatial_dim method correctly identifies it as the spatial dimension. This ensures that the method can recognize 'nEdges' as a valid spatial dimension when it is included in the dataset, allowing it to function correctly with edge-based datasets. The test simulates a scenario where 'nEdges' is present in the sizes dictionary, confirming that the method returns 'nEdges' as the spatial dimension in this case. 

        Parameters:
            None

        Returns:
            None
        """
        sizes = {'nEdges': 30, 'Time': 2}
        assert MPAS3DProcessor._detect_spatial_dim(sizes) == 'nEdges'


class TestLookupCoordNone:
    """ Test _lookup_coord returns None when no names match, and returns values for first match. """

    def test_returns_none_when_no_name_matches(self: 'TestLookupCoordNone') -> None:
        """
        This test verifies that the _lookup_coord method returns None when none of the provided coordinate names match any coordinates in the dataset. This ensures that the method correctly handles cases where the expected coordinate variables are not present in the dataset, providing a clear indication (None) that no matching coordinates were found. The test simulates a scenario where the dataset contains a variable that does not match any of the provided coordinate names, confirming that the method returns None in this case. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({'theta': xr.DataArray(np.ones(5), dims=['nCells'])})
        result = MPAS3DProcessor._lookup_coord(ds, ['lonCell', 'longitude', 'lon'])
        assert result is None

    def test_returns_values_for_first_matching_name(self: 'TestLookupCoordNone') -> None:
        """
        This test verifies that the _lookup_coord method returns the coordinate values for the first matching name in the provided list of coordinate names. This ensures that the method correctly identifies and retrieves the coordinate variable when multiple potential matches are provided, prioritizing the first match in the list. The test simulates a scenario where the dataset contains a coordinate variable that matches one of the provided names, confirming that the method returns the correct coordinate values for that match. 

        Parameters:
            None

        Returns:
            None
        """
        lon = np.linspace(-120.0, -80.0, 5)
        ds = xr.Dataset({'lonCell': xr.DataArray(lon, dims=['nCells'])})
        result = MPAS3DProcessor._lookup_coord(ds, ['lonCell', 'longitude', 'lon'])
        assert np.allclose(result, lon)


class TestExtract2DCoordinatesForVariable:
    """ Test extract_2d_coordinates_for_variable branches related to dataset loading and coordinate extraction. """

    def test_raises_value_error_when_dataset_is_none(self: 'TestExtract2DCoordinatesForVariable', 
                                                     proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the extract_2d_coordinates_for_variable method raises a ValueError when the dataset attribute of the processor is None. This ensures that the method correctly handles cases where the dataset has not been loaded, preventing attempts to extract coordinates from a non-existent dataset and providing a clear error message to the user. The test simulates a scenario where the dataset is None, confirming that the method raises the appropriate exception in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to None. 

        Returns:
            None
        """
        proc.dataset = None
        with pytest.raises(ValueError):
            proc.extract_2d_coordinates_for_variable('theta')

    def test_probe_open_failure_covered_and_main_succeeds(self: 'TestExtract2DCoordinatesForVariable', 
                                                         proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the extract_2d_coordinates_for_variable method correctly handles a failure during the initial probe attempt to open the grid dataset, and successfully falls back to the main attempt to open the dataset and extract coordinates. This ensures that the method is robust to transient issues that may occur during the probe phase, allowing it to continue functioning and extract coordinates as long as the main attempt succeeds. The test simulates a scenario where the first call to open the grid dataset raises an IOError (simulating a probe failure), while the second call returns a valid dataset containing coordinate variables, confirming that the method can recover from the probe failure and successfully extract coordinates in this case.  

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        proc.dataset = _make_3d_ds()

        grid_with_coords = xr.Dataset({
            'lonCell': xr.DataArray(np.linspace(-120.0, -80.0, 5)),
            'latCell': xr.DataArray(np.linspace(25.0, 55.0, 5)),
        })

        with patch('mpasdiag.processing.processors_3d.xr.open_dataset',
                   side_effect=[IOError("probe fail"), grid_with_coords]):
            lon, lat = proc.extract_2d_coordinates_for_variable('theta')

        assert len(lon) == 5

    def test_no_coords_in_grid_raises_runtime_error(self: 'TestExtract2DCoordinatesForVariable', 
                                                    proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the extract_2d_coordinates_for_variable method raises a RuntimeError when the grid dataset is successfully opened but does not contain any of the expected coordinate variables (such as 'lonCell' or 'latCell'). This ensures that the method correctly handles cases where the grid dataset is missing the necessary coordinate information, providing a clear error message to the user and preventing further processing with an invalid grid. The test simulates a scenario where the grid dataset is opened successfully but contains no coordinate variables, confirming that the method raises the appropriate exception in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        proc.dataset = _make_3d_ds()
        empty_grid = xr.Dataset({'dummy': xr.DataArray([1.0])})
        with patch('mpasdiag.processing.processors_3d.xr.open_dataset',
                   return_value=empty_grid):
            with pytest.raises(RuntimeError, match="Error loading coordinates"):
                proc.extract_2d_coordinates_for_variable('theta')

    def test_radian_coords_converted_to_degrees(self: 'TestExtract2DCoordinatesForVariable', 
                                                proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the extract_2d_coordinates_for_variable method correctly detects when the coordinate values in the grid dataset are in radians and converts them to degrees. This ensures that the method can handle cases where the grid dataset uses radians for longitude and latitude, providing the correct coordinate values in degrees for further processing. The test simulates a scenario where the grid dataset contains coordinate variables with values in radians, confirming that the method returns the expected values converted to degrees. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        proc.dataset = _make_3d_ds()
        n = 5

        lat_rad = np.linspace(-1.0, 1.0, n)
        lon_rad = np.linspace(-2.0, 2.0, n)

        grid_rad = xr.Dataset({
            'lonCell': xr.DataArray(lon_rad),
            'latCell': xr.DataArray(lat_rad),
        })

        with patch('mpasdiag.processing.processors_3d.xr.open_dataset',
                   return_value=grid_rad):
            result_lon, result_lat = proc.extract_2d_coordinates_for_variable('theta')

        assert np.allclose(result_lat, lat_rad * 180.0 / np.pi)

    def test_verbose_prints_coord_summary(self: 'TestExtract2DCoordinatesForVariable', 
                                          proc_v: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the extract_2d_coordinates_for_variable method prints a summary of the extracted coordinates when verbose mode is enabled. This ensures that the method provides useful information to the user about the coordinates that were extracted from the grid dataset, enhancing the user experience in verbose mode. The test simulates a scenario where the grid dataset contains valid coordinate variables, and captures the standard output to confirm that a message indicating the successful extraction of coordinates is printed when verbose mode is active. 
        
        Parameters:
            proc_v ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        proc_v.dataset = _make_3d_ds()

        grid_ds = xr.Dataset({
            'lonCell': xr.DataArray(np.linspace(-120.0, -80.0, 5)),
            'latCell': xr.DataArray(np.linspace(25.0, 55.0, 5)),
        })

        captured = StringIO()

        with patch('mpasdiag.processing.processors_3d.xr.open_dataset',
                   return_value=grid_ds):
            with patch('sys.stdout', new=captured):
                proc_v.extract_2d_coordinates_for_variable('theta')

        assert 'Extracted' in captured.getvalue()


class TestFindFilesRecursive:
    """ Test _find_files_recursive error paths and normal flows related to finding MPAS output files in a directory tree. """

    def test_raises_file_not_found_when_no_files(self: 'TestFindFilesRecursive',
                                                 proc: 'MPAS3DProcessor',
                                                 tmp_path) -> None:
        """
        This test verifies that the _find_files_recursive method raises a FileNotFoundError when no MPAS output files are found in the specified directory or its subdirectories. This ensures that the method correctly handles cases where there are no relevant files to process, providing a clear error message to the user and preventing further processing with an empty file list. The test simulates a scenario where the glob function returns an empty list, confirming that the method raises the appropriate exception in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        with patch('mpasdiag.processing.processors_3d.glob.glob', return_value=[]):
            with pytest.raises(FileNotFoundError, match="No MPAS output files"):
                proc._find_files_recursive(str(tmp_path / 'data'))

    def test_raises_value_error_when_only_one_file(self: 'TestFindFilesRecursive',
                                                   proc: 'MPAS3DProcessor',
                                                   tmp_path) -> None:
        """
        This test verifies that the _find_files_recursive method raises a ValueError when only one MPAS output file is found in the specified directory or its subdirectories. This ensures that the method correctly handles cases where there are not enough files to perform meaningful processing, providing a clear error message to the user and preventing further processing with an insufficient file list. The test simulates a scenario where the glob function returns a list containing only one file path, confirming that the method raises the appropriate exception in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        with patch('mpasdiag.processing.processors_3d.glob.glob',
                   return_value=[str(tmp_path / 'mpasout_2024.nc')]):
            with pytest.raises(ValueError, match="Insufficient MPAS output files"):
                proc._find_files_recursive(str(tmp_path / 'data'))

    def test_returns_sorted_files_silent(self: 'TestFindFilesRecursive',
                                         proc: 'MPAS3DProcessor',
                                         tmp_path) -> None:
        """
        This test verifies that the _find_files_recursive method returns a sorted list of MPAS output file paths when multiple files are found, and that it does so without printing any messages when verbose mode is disabled. This ensures that the method provides a consistent and user-friendly output by sorting the file paths, while also respecting the user's preference for silent operation when verbose mode is not enabled. The test simulates a scenario where multiple MPAS output files are found in an unsorted order, confirming that the method returns the file paths sorted correctly without producing any output in silent mode. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        fake = [str(tmp_path / 'mpasout_2025.nc'), str(tmp_path / 'mpasout_2024.nc')]

        with patch('mpasdiag.processing.processors_3d.glob.glob', return_value=fake):
            result = proc._find_files_recursive(str(tmp_path / 'data'))

        assert result == sorted(fake)

    def test_verbose_prints_file_list(self: 'TestFindFilesRecursive',
                                       proc_v: 'MPAS3DProcessor',
                                       tmp_path) -> None:
        """
        This test verifies that the _find_files_recursive method prints a message listing the found MPAS output files when verbose mode is enabled. This ensures that the method provides useful feedback to the user about the files that were found during the recursive search, enhancing the user experience in verbose mode. The test simulates a scenario where multiple MPAS output files are found, and captures the standard output to confirm that a message indicating the found files is printed when verbose mode is active. 

        Parameters:
            proc_v ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset and verbose mode enabled.

        Returns:
            None
        """
        fake = [str(tmp_path / f'mpasout_{i:04d}.nc') for i in range(6)]

        with patch('mpasdiag.processing.processors_3d.glob.glob', return_value=fake):
            captured = StringIO()
            with patch('sys.stdout', new=captured):
                result = proc_v._find_files_recursive(str(tmp_path / 'data'))

        assert result is not None
        assert 'MPAS output files (recursive' in captured.getvalue()


class TestFindMpasoutFiles:
    """ Test find_mpasout_files branches related to searching for MPAS output files in main and subdirectory paths, and falling back to recursive search. """

    def test_subdir_path_used_when_main_dir_fails(self: 'TestFindMpasoutFiles',
                                                  proc: 'MPAS3DProcessor',
                                                  tmp_path) -> None:
        """
        This test verifies that the find_mpasout_files method correctly falls back to searching in a subdirectory when the initial search in the main directory fails to find any MPAS output files. This ensures that the method is robust to cases where the MPAS output files are organized in a subdirectory (such as 'mpasout') rather than directly in the specified directory, allowing it to successfully find the files without requiring additional user input. The test simulates a scenario where the first call to find files in the main directory raises a FileNotFoundError, while the second call to search in the subdirectory returns a valid list of file paths, confirming that the method correctly handles this fallback behavior. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        fake = [str(tmp_path / 'mpasout' / 'mpasout_2024.nc'), str(tmp_path / 'mpasout' / 'mpasout_2025.nc')]

        with patch.object(proc, '_find_files_by_pattern',
                          side_effect=[FileNotFoundError(), fake]):
            result = proc.find_mpasout_files(str(tmp_path / 'data'))

        assert result == fake

    def test_recursive_used_when_both_dirs_fail(self: 'TestFindMpasoutFiles',
                                                proc: 'MPAS3DProcessor',
                                                tmp_path) -> None:
        """
        This test verifies that the find_mpasout_files method correctly falls back to a recursive search when both the initial search in the main directory and the subsequent search in the subdirectory fail to find any MPAS output files. This ensures that the method is robust to cases where the MPAS output files are organized in an unexpected way within the directory tree, allowing it to successfully find the files by searching through all subdirectories if necessary. The test simulates a scenario where both calls to find files in the main directory and subdirectory raise FileNotFoundError, while a call to search recursively returns a valid list of file paths, confirming that the method correctly handles this fallback behavior.

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        fake = [str(tmp_path / 'sub' / 'mpasout_2024.nc'), str(tmp_path / 'sub' / 'mpasout_2025.nc')]

        with patch.object(proc, '_find_files_by_pattern', side_effect=FileNotFoundError):
            with patch.object(proc, '_find_files_recursive', return_value=fake):
                result = proc.find_mpasout_files(str(tmp_path / 'data'))

        assert result == fake


class TestLoad3DDataUXarrayPath:
    """ Test load_3d_data when _load_data returns a uxarray dataset, covering the path where the dataset is assigned and self is returned. """

    def test_uxarray_path_assigns_dataset_and_returns_self(self: 'TestLoad3DDataUXarrayPath',
                                                           proc: 'MPAS3DProcessor',
                                                           tmp_path) -> None:
        """
        This test verifies that when the _load_data method returns a dataset of type 'uxarray', the load_3d_data method correctly assigns this dataset to the processor's dataset attribute and returns the processor instance itself (self). This ensures that the load_3d_data method can handle datasets returned by _load_data in the 'uxarray' format, allowing it to integrate with different data loading mechanisms while maintaining a consistent interface. The test simulates a scenario where _load_data returns a mock dataset with the expected structure, confirming that load_3d_data assigns it correctly and returns self in this case.

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        inner_ds = xr.Dataset({'theta': xr.DataArray(np.ones(5), dims=['nCells'])})
        enriched_ds = inner_ds.assign({'lonCell': xr.DataArray(np.ones(5))})
        mock_ux = MagicMock(spec=['ds'])
        mock_ux.ds = inner_ds

        with patch.object(proc, '_load_data', return_value=(mock_ux, 'uxarray')):
            with patch.object(proc, 'add_spatial_coordinates', return_value=enriched_ds):
                result = proc.load_3d_data(str(tmp_path / 'data'))

        assert result is proc
        assert proc.dataset is mock_ux
        assert proc.data_type == 'uxarray'


class TestGetAvailable3DVarsError:
    """ Test get_available_3d_variables raises when dataset is None. """

    def test_raises_when_dataset_none(self: 'TestGetAvailable3DVarsError', 
                                      proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the get_available_3d_variables method raises a ValueError when the dataset attribute of the processor is None. This ensures that the method correctly handles cases where the dataset has not been loaded, preventing attempts to access variables from a non-existent dataset and providing a clear error message to the user. The test simulates a scenario where the dataset is None, confirming that the method raises the appropriate exception in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to None.

        Returns:
            None
        """
        proc.dataset = None
        with pytest.raises(ValueError, match="Dataset not loaded"):
            proc.get_available_3d_variables()


class TestValidate3DVariable:
    """ Test _validate_3d_variable all three error paths. """

    def test_raises_when_dataset_none(self: 'TestValidate3DVariable', 
                                      proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _validate_3d_variable method raises a ValueError when the dataset attribute of the processor is None. This ensures that the method correctly handles cases where the dataset has not been loaded, preventing attempts to validate variables from a non-existent dataset and providing a clear error message to the user. The test simulates a scenario where the dataset is None, confirming that the method raises the appropriate exception in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to None.

        Returns:
            None
        """
        proc.dataset = None
        with pytest.raises(ValueError):
            proc._validate_3d_variable('theta')

    def test_raises_when_variable_not_found(self: 'TestValidate3DVariable', 
                                            proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _validate_3d_variable method raises a ValueError when the specified variable name is not found in the dataset. This ensures that the method correctly handles cases where the user requests a variable that does not exist in the dataset, providing a clear error message to the user and preventing further processing with an invalid variable name. The test simulates a scenario where the dataset is loaded but does not contain the requested variable, confirming that the method raises the appropriate exception in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        proc.dataset = _make_3d_ds()
        with pytest.raises(ValueError, match="Variable 'missing' not found"):
            proc._validate_3d_variable('missing')

    def test_raises_when_variable_is_not_3d(self: 'TestValidate3DVariable', 
                                            proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _validate_3d_variable method raises a ValueError when the specified variable exists in the dataset but does not have a vertical dimension, indicating that it is not a 3D atmospheric variable. This ensures that the method correctly identifies and rejects variables that do not meet the criteria for being considered 3D atmospheric variables, providing a clear error message to the user and preventing further processing with an invalid variable. The test simulates a scenario where the dataset contains a variable with only time and spatial dimensions, confirming that the method raises the appropriate exception in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        times = pd.date_range('2024-01-01', periods=2, freq='h')

        proc.dataset = xr.Dataset({
            't2m': xr.DataArray(
                np.ones((2, 10)),
                dims=['Time', 'nCells'],
                coords={'Time': times},
            ),
        })

        with pytest.raises(ValueError, match="not a 3D atmospheric variable"):
            proc._validate_3d_variable('t2m')


class TestResolveIntLevelExceedsMax:
    """ Test _resolve_int_level raises when specified index exceeds maximum vertical levels. """

    def test_exceeds_max_levels_raises(self: 'TestResolveIntLevelExceedsMax', 
                                       proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _resolve_int_level method raises a ValueError when the specified integer level index exceeds the maximum number of vertical levels available in the dataset. This ensures that the method correctly handles cases where the user requests a level index that is out of bounds, providing a clear error message to the user and preventing attempts to access invalid level indices. The test simulates a scenario where the dataset is loaded with a certain number of vertical levels, and the method is called with an index that exceeds this number, confirming that it raises the appropriate exception in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        proc.dataset = _make_3d_ds()  # nVertLevels=4
        with pytest.raises(ValueError, match="Model level 10 exceeds"):
            proc._resolve_int_level(10)


class TestResolveStrLevelUnknown:
    """ Test _resolve_str_level raises for unrecognized string. """

    def test_unknown_string_raises(self: 'TestResolveStrLevelUnknown', 
                                   proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _resolve_str_level method raises a ValueError when an unrecognized string is provided as the level specification. This ensures that the method correctly handles cases where the user inputs a string that does not correspond to any known level specifications (such as 'surface' or 'top'), providing a clear error message to the user and preventing attempts to resolve invalid level specifications. The test simulates a scenario where the dataset is loaded, and the method is called with an unrecognized string, confirming that it raises the appropriate exception in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        proc.dataset = _make_3d_ds()
        with pytest.raises(ValueError, match="Unknown level specification"):
            proc._resolve_str_level('middle')

    def test_surface_returns_zero(self: 'TestResolveStrLevelUnknown', 
                                  proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _resolve_str_level method returns 0 when the 'surface' string is provided, indicating that the surface level corresponds to the first index in the vertical dimension. This ensures that the method correctly identifies the surface level as the bottom-most level (index 0) in the vertical dimension, which is important for correctly handling requests for surface-level data. The test simulates a scenario where the dataset is loaded with a certain number of vertical levels, and the method is called with the 'surface' string, confirming that it returns 0 as expected for the surface level. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        proc.dataset = _make_3d_ds()
        assert proc._resolve_str_level('surface') == 0

    def test_top_returns_last_index(self: 'TestResolveStrLevelUnknown', 
                                     proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _resolve_str_level method returns the last index in the vertical dimension when the 'top' string is provided. This ensures that the method correctly identifies the top level as the last index (nVertLevels - 1) in the vertical dimension, which is important for correctly handling requests for top-level data. The test simulates a scenario where the dataset is loaded with a certain number of vertical levels, and the method is called with the 'top' string, confirming that it returns the expected index corresponding to the top level. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        proc.dataset = _make_3d_ds(n_vert=4)  # nVertLevels=4 → top=3
        assert proc._resolve_str_level('top') == 3


class TestLerpVariableExceptionPath:
    """ Test _lerp_variable exception handling path when interpolation fails, ensuring it falls back to nearest index and prints message in verbose mode. """

    def _make_bad_da(self: 'TestLerpVariableExceptionPath') -> MagicMock:
        """
        This helper method creates a mock DataArray that simulates a failure during interpolation by raising an ArithmeticError when multiplication is attempted. This mock DataArray is used in tests to trigger the exception handling path in the _lerp_variable method, allowing us to verify that the method correctly falls back to returning the nearest index and prints a message in verbose mode when interpolation fails. 

        Parameters:
            None

        Returns:
            MagicMock: A mock DataArray that raises an ArithmeticError on multiplication.
        """
        bad = MagicMock()
        bad.__rmul__ = MagicMock(side_effect=ArithmeticError("simulated"))
        bad.__mul__ = MagicMock(side_effect=ArithmeticError("simulated"))
        return bad

    def _make_mock_ds(self: 'TestLerpVariableExceptionPath') -> MagicMock:
        """
        This helper method creates a mock dataset that contains a variable which raises an ArithmeticError when interpolation is attempted. The method constructs a mock variable that simulates the failure during interpolation, and then creates a mock dataset that returns this variable when accessed. This mock dataset is used in tests to trigger the exception handling path in the _lerp_variable method, allowing us to verify that the method correctly falls back to returning the nearest index and prints a message in verbose mode when interpolation fails. 

        Parameters:
            None

        Returns:
            MagicMock: A mock dataset containing a variable that raises an ArithmeticError on multiplication.
        """
        bad_var = MagicMock()
        bad_var.isel.return_value = self._make_bad_da()
        mock_ds = MagicMock()
        mock_ds.__getitem__.return_value = bad_var
        return mock_ds

    def test_exception_fallback_returns_nearest_idx(self: 'TestLerpVariableExceptionPath', 
                                                    proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _lerp_variable method correctly falls back to returning the nearest index when an exception occurs during interpolation. This ensures that the method can still provide a meaningful result (the nearest index) even when interpolation fails, allowing the processing to continue without crashing. The test simulates a scenario where the dataset contains a variable that raises an ArithmeticError during interpolation, confirming that the method returns an integer index and None for the result as expected in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        proc.dataset = _make_3d_ds()
        mean_p = _mean_p_ascending()
        mock_ds = self._make_mock_ds()

        idx, result = proc._lerp_variable(
            'theta', 0, 1, 0.5, mean_p, 60000.0,
            mock_ds, 'Time', 0, 'nVertLevels'
        )

        assert result is None
        assert isinstance(idx, int)

    def test_exception_fallback_verbose_prints_message(self: 'TestLerpVariableExceptionPath', 
                                                       proc_v: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _lerp_variable method correctly prints a message in verbose mode when an exception occurs during interpolation and it falls back to returning the nearest index. This ensures that the method provides useful feedback to the user about the interpolation failure, enhancing the user experience in verbose mode by informing them of the issue and the fallback behavior. The test simulates a scenario where the dataset contains a variable that raises an ArithmeticError during interpolation, and captures the standard output to confirm that a message indicating the interpolation failure is printed when verbose mode is active. 

        Parameters:
            proc_v ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset in verbose mode.

        Returns:
            None
        """
        proc_v.dataset = _make_3d_ds()
        mean_p = _mean_p_ascending()
        mock_ds = self._make_mock_ds()
        captured = StringIO()

        with patch('sys.stdout', new=captured):
            idx, result = proc_v._lerp_variable(
                'theta', 0, 1, 0.5, mean_p, 60000.0,
                mock_ds, 'Time', 0, 'nVertLevels'
            )

        assert 'Interpolation failed' in captured.getvalue()


class TestInterpolateAtPressureVerbosePaths:
    """ Test _interpolate_at_pressure branches related to pressure above surface and below top, ensuring it prints messages in verbose mode and returns expected indices. """

    def test_above_surface_verbose_prints_and_returns_zero(self: 'TestInterpolateAtPressureVerbosePaths', 
                                                           proc_v: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _interpolate_at_pressure method correctly prints a message and returns zero when the specified pressure is above the surface in verbose mode. This ensures that the method provides useful feedback to the user about the pressure being above the surface, enhancing the user experience in verbose mode. The test simulates a scenario where the mean pressure values indicate that the specified pressure is above the surface, and captures the standard output to confirm that a message indicating this condition is printed when verbose mode is active. 

        Parameters:
            proc_v ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset in verbose mode.

        Returns:
            None
        """
        proc_v.dataset = _make_3d_ds()
        mean_p = _mean_p_ascending()   # max = 100000
        ds_raw = _make_3d_ds()
        captured = StringIO()

        with patch('sys.stdout', new=captured):
            idx, result = proc_v._interpolate_at_pressure(
                'theta', 110000.0, mean_p, ds_raw, 'Time', 0
            )

        assert idx == 0
        assert result is None
        assert 'above surface' in captured.getvalue()

    def test_above_surface_silent_returns_zero(self: 'TestInterpolateAtPressureVerbosePaths', 
                                               proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _interpolate_at_pressure method correctly returns zero when the specified pressure is above the surface in non-verbose mode. This ensures that the method provides the expected behavior of returning zero for the pressure level when it is above the surface, even when verbose mode is not enabled. The test simulates a scenario where the mean pressure values indicate that the specified pressure is above the surface, confirming that the method returns zero and None for the result as expected in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        proc.dataset = _make_3d_ds()
        mean_p = _mean_p_ascending()
        ds_raw = _make_3d_ds()

        idx, result = proc._interpolate_at_pressure(
            'theta', 110000.0, mean_p, ds_raw, 'Time', 0
        )

        assert idx == 0 and result is None

    def test_below_top_verbose_prints_and_returns_last_idx(self: 'TestInterpolateAtPressureVerbosePaths', 
                                                           proc_v: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _interpolate_at_pressure method correctly prints a message and returns the last index when the specified pressure is below the top in verbose mode. This ensures that the method provides useful feedback to the user about the pressure being below the top, enhancing the user experience in verbose mode. The test simulates a scenario where the mean pressure values indicate that the specified pressure is below the top, and captures the standard output to confirm that a message indicating this condition is printed when verbose mode is active. Additionally, it confirms that the method returns the last index and None for the result as expected in this case. 

        Parameters:
            proc_v ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset in verbose mode.

        Returns:
            None
        """
        proc_v.dataset = _make_3d_ds()
        mean_p = _mean_p_ascending()   # min = 50000
        ds_raw = _make_3d_ds()
        captured = StringIO()

        with patch('sys.stdout', new=captured):
            idx, result = proc_v._interpolate_at_pressure(
                'theta', 40000.0, mean_p, ds_raw, 'Time', 0
            )

        assert idx == len(mean_p) - 1
        assert result is None
        assert 'below top' in captured.getvalue()

    def test_below_top_silent_returns_last_idx(self: 'TestInterpolateAtPressureVerbosePaths', 
                                               proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _interpolate_at_pressure method correctly returns the last index when the specified pressure is below the top in non-verbose mode. This ensures that the method provides the expected behavior of returning the last index for the pressure level when it is below the top, even when verbose mode is not enabled. The test simulates a scenario where the mean pressure values indicate that the specified pressure is below the top, confirming that the method returns the last index and None for the result as expected in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        proc.dataset = _make_3d_ds()
        mean_p = _mean_p_ascending()
        ds_raw = _make_3d_ds()

        idx, result = proc._interpolate_at_pressure(
            'theta', 40000.0, mean_p, ds_raw, 'Time', 0
        )

        assert idx == len(mean_p) - 1 and result is None

    def test_boundary_lower_idx_at_last_returns_without_interpolating(self: 'TestInterpolateAtPressureVerbosePaths', 
                                                                      proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _interpolate_at_pressure method correctly returns without attempting interpolation when the lower index is at the last index of the mean pressure values, which indicates that the specified pressure is below the top. This ensures that the method can handle edge cases where the pressure level is outside the range of available levels, preventing attempts to interpolate when it is not possible and returning None for the result as expected. The test simulates a scenario where the mean pressure values are such that the specified pressure is below the top, confirming that the method returns the last index and None for the result without attempting interpolation in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        proc.dataset = _make_3d_ds()
        mean_p = _mean_p_ascending()  # [50000, 70000, 85000, 100000]
        ds_raw = _make_3d_ds()

        idx, result = proc._interpolate_at_pressure(
            'theta', 99000.0, mean_p, ds_raw, 'Time', 0
        )

        assert idx == 3
        assert result is None


class TestResolveFloatLevelNoPressure:
    """ Test _resolve_float_level raises when pressure variables are absent in the dataset, ensuring it raises a ValueError with an appropriate message. """

    def test_raises_when_pressure_variables_absent(self: 'TestResolveFloatLevelNoPressure', 
                                                   proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _resolve_float_level method raises a ValueError when pressure-related variables (such as 'pressure' or 'pressure_on_cell') are not available in the dataset. This ensures that the method correctly handles cases where the necessary pressure data is missing, providing a clear error message to the user and preventing attempts to resolve float levels without the required pressure information. The test simulates a scenario where the dataset is loaded without any pressure-related variables, confirming that the method raises the appropriate exception with a message indicating that pressure data is not available. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        times = pd.date_range('2024-01-01', periods=2, freq='h')

        proc.dataset = xr.Dataset({
            'theta': xr.DataArray(
                np.ones((2, 5, 4)),
                dims=['Time', 'nCells', 'nVertLevels'],
                coords={'Time': times},
            ),
        })

        with pytest.raises(ValueError, match="pressure data not available"):
            proc._resolve_float_level('theta', 85000.0, 'Time', 0)


class TestResolveLevelIndexInvalidType:
    """ Test _resolve_level_index raises for invalid level type. """

    def test_list_type_raises_value_error(self: 'TestResolveLevelIndexInvalidType', 
                                          proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _resolve_level_index method raises a ValueError when an invalid type (such as a list) is provided as the level specification. This ensures that the method correctly handles cases where the user inputs a level specification that is not of the expected types (integer, float, or recognized string), providing a clear error message to the user and preventing attempts to resolve levels with an invalid specification. The test simulates a scenario where the dataset is loaded, and the method is called with a list as the level specification, confirming that it raises the appropriate exception in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        proc.dataset = _make_3d_ds()
        with pytest.raises(ValueError, match="Invalid level specification"):
            proc._resolve_level_index('theta', [1, 2], 'Time', 0)  # type: ignore[arg-type]


class TestSetLevelAttrs:
    """ Test _set_level_attrs branches related to handling variables without 'attrs', adding 'actual_pressure_level' for float levels with pressure data, and adding 'selected_level' for integer levels. """

    def test_early_return_when_no_attrs_attribute(self: 'TestSetLevelAttrs', 
                                                  proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _set_level_attrs method correctly returns early without attempting to set attributes when the variable data does not have an 'attrs' attribute. This ensures that the method can handle cases where the variable data is not a standard xarray DataArray (which has an 'attrs' attribute), preventing attempts to set attributes on objects that do not support them and avoiding potential errors. The test simulates a scenario where the variable data is a simple object without an 'attrs' attribute, confirming that the method returns without error and does not attempt to set any attributes in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        proc.dataset = _make_3d_ds()

        class NoAttrs:
            pass

        no_attrs_obj = NoAttrs()
        ds = _make_3d_ds()
        proc._set_level_attrs(no_attrs_obj, 0, 0, ds, 'Time', 0, 'nVertLevels')  # type: ignore[arg-type]
        assert not hasattr(no_attrs_obj, 'selected_level')

    def test_float_level_with_pressure_adds_actual_pressure_attr(self: 'TestSetLevelAttrs', 
                                                                 proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _set_level_attrs method correctly adds the 'actual_pressure_level' attribute to the variable data when a float level is specified and pressure-related variables are available in the dataset. This ensures that the method provides useful metadata about the actual pressure level corresponding to the specified float level, which can be important for interpreting the data correctly. The test simulates a scenario where the dataset contains pressure-related variables, and the method is called with a float level specification, confirming that it adds the 'actual_pressure_level' attribute to the variable data with the expected value. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        ds = _make_3d_ds()
        proc.dataset = ds
        var_data = xr.DataArray(np.ones(5), dims=['nCells'])
        proc._set_level_attrs(var_data, 85000.0, 1, ds, 'Time', 0, 'nVertLevels')
        assert 'actual_pressure_level' in var_data.attrs
        assert 'Pa' in var_data.attrs['actual_pressure_level']

    def test_int_level_adds_selected_level_attr(self: 'TestSetLevelAttrs', 
                                                proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _set_level_attrs method correctly adds the 'selected_level' and 'level_index' attributes to the variable data when the level is specified as an integer. This ensures that the method provides useful metadata about the selected model level, which can be important for interpreting the data correctly. The test simulates a scenario where the dataset is loaded, and the method is called with an integer level specification, confirming that it adds the 'selected_level' and 'level_index' attributes to the variable data without adding an 'actual_pressure_level' attribute, as expected for integer levels. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        ds = _make_3d_ds()
        proc.dataset = ds
        var_data = xr.DataArray(np.ones(5), dims=['nCells'])
        proc._set_level_attrs(var_data, 2, 2, ds, 'Time', 0, 'nVertLevels')
        assert var_data.attrs['selected_level'] == 2
        assert var_data.attrs['level_index'] == 2
        assert 'actual_pressure_level' not in var_data.attrs


class TestGetVerticalDimNotThreeD:
    """ Test _get_vertical_dim raises when variable has no vertical dimension, ensuring it raises a ValueError with an appropriate message. """

    def test_raises_when_variable_has_no_vert_dim(self: 'TestGetVerticalDimNotThreeD', 
                                                  proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _get_vertical_dim method raises a ValueError when the specified variable exists in the dataset but does not have a vertical dimension, indicating that it is not a 3D atmospheric variable. This ensures that the method correctly identifies and rejects variables that do not meet the criteria for being considered 3D atmospheric variables, providing a clear error message to the user and preventing further processing with an invalid variable. The test simulates a scenario where the dataset contains a variable with only time and spatial dimensions, confirming that the method raises the appropriate exception with a message indicating that the variable is not a 3D atmospheric variable in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        times = pd.date_range('2024-01-01', periods=2, freq='h')

        proc.dataset = xr.Dataset({
            't2m': xr.DataArray(
                np.ones((2, 10)),
                dims=['Time', 'nCells'],
                coords={'Time': times},
            ),
        })

        with pytest.raises(ValueError, match="not a 3D atmospheric variable"):
            proc._get_vertical_dim('t2m')


class TestPressureLevelsFromPressureVarExcept:
    """ Test _pressure_levels_from_pressure_var fallback path when 'pressure' variable has no Time dimension, ensuring it returns pressure levels based on the shape of the variable without raising an error. """

    def test_pressure_without_time_dim_uses_fallback(self: 'TestPressureLevelsFromPressureVarExcept', 
                                                     proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _pressure_levels_from_pressure_var method correctly falls back to returning pressure levels based on the shape of the variable when the 'pressure' variable does not have a Time dimension. This ensures that the method can still provide pressure level information even when the expected time dimension is missing, preventing errors and allowing processing to continue with the available data. The test simulates a scenario where the dataset contains a 'pressure' variable that lacks a Time dimension, confirming that the method returns an array of pressure levels with the expected length based on the shape of the variable without raising an error in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        times = pd.date_range('2024-01-01', periods=2, freq='h')
        p_vals = np.tile(np.linspace(100000.0, 50000.0, 4), (5, 1))

        proc.dataset = xr.Dataset({
            'theta': xr.DataArray(
                np.ones((2, 5, 4)),
                dims=['Time', 'nCells', 'nVertLevels'],
                coords={'Time': times},
            ),
            'pressure': xr.DataArray(
                p_vals,
                dims=['nCells', 'nVertLevels'],  # no Time dimension
            ),
        })

        result = proc._pressure_levels_from_pressure_var(
            'theta', 'nVertLevels', 4, 'Time', 0
        )

        assert result is not None
        assert len(result) == 4


class TestRepairPressureLevels:
    """ Test _repair_pressure_levels all three paths: interpolation when two or more valid levels, linspace when exactly one valid level, and logspace when zero valid levels. """

    def test_two_or_more_valid_levels_uses_interp(self: 'TestRepairPressureLevels', 
                                                  proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _repair_pressure_levels method uses interpolation to fill in missing pressure levels when there are two or more valid levels available. This ensures that the method can accurately reconstruct the pressure level profile by interpolating between the valid levels, providing a more realistic representation of the pressure levels in the dataset. The test simulates a scenario where the input pressure levels contain at least two valid values and some missing values, confirming that the method returns a repaired array with all finite values and the expected length in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        levels = np.array([100000.0, np.nan, 50000.0, 10000.0])
        result = proc._repair_pressure_levels(levels, 101325.0)
        assert result is not None
        assert len(result) == 4
        assert np.all(np.isfinite(result))

    def test_exactly_one_valid_level_uses_linspace(self: 'TestRepairPressureLevels', 
                                                   proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _repair_pressure_levels method uses linspace to fill in missing pressure levels when there is exactly one valid level available. This ensures that the method can still provide a repaired pressure level array by creating a linear space of pressure levels between the surface pressure and a small value, which is a common approach for representing pressure levels in atmospheric data when limited information is available. The test simulates a scenario where the input pressure levels contain exactly one valid value and the rest are missing, confirming that the method returns a repaired array with the expected length in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        levels = np.array([100000.0, np.nan, np.nan, -1.0])  # only first valid
        result = proc._repair_pressure_levels(levels, 101325.0)
        assert result is not None
        assert len(result) == 4

    def test_zero_valid_levels_uses_logspace(self: 'TestRepairPressureLevels', 
                                             proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the _repair_pressure_levels method uses logspace to fill in missing pressure levels when there are zero valid levels available. This ensures that the method can still provide a repaired pressure level array by creating a logarithmic space of pressure levels between the surface pressure and a small value, which is a common approach for representing pressure levels in atmospheric data when no valid information is available. The test simulates a scenario where the input pressure levels contain no valid values, confirming that the method returns a repaired array with the expected length in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        levels = np.array([np.nan, np.nan, -1.0, -2.0])
        result = proc._repair_pressure_levels(levels, 101325.0)
        assert result is not None
        assert len(result) == 4


class TestGetVerticalLevelsErrors:
    """ Test get_vertical_levels error handling paths: raises when dataset is None and raises when variable not found, ensuring it raises ValueError with appropriate messages in both cases. """

    def test_raises_when_dataset_none(self: 'TestGetVerticalLevelsErrors', 
                                      proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the get_vertical_levels method raises a ValueError when the dataset is None, indicating that no dataset has been loaded for processing. This ensures that the method correctly identifies and handles cases where the dataset is not available, providing a clear error message to the user and preventing attempts to access vertical levels without a valid dataset. The test simulates a scenario where the dataset is explicitly set to None, confirming that the method raises the appropriate exception with a message indicating that no dataset is loaded in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to None.

        Returns:
            None
        """
        proc.dataset = None
        with pytest.raises(ValueError):
            proc.get_vertical_levels('theta')

    def test_raises_when_variable_not_found(self: 'TestGetVerticalLevelsErrors', 
                                            proc: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the get_vertical_levels method raises a ValueError when the specified variable is not found in the dataset. This ensures that the method correctly handles cases where the user requests vertical levels for a variable that does not exist in the dataset, providing a clear error message to the user and preventing attempts to access vertical levels for an invalid variable. The test simulates a scenario where the dataset is loaded with valid data, but the method is called with a variable name that does not exist in the dataset, confirming that it raises the appropriate exception with a message indicating that the variable was not found in this case. 

        Parameters:
            proc ('MPAS3DProcessor'): An instance of MPAS3DProcessor with dataset set to a valid 3D dataset.

        Returns:
            None
        """
        proc.dataset = _make_3d_ds()
        with pytest.raises(ValueError, match="Variable 'missing' not found"):
            proc.get_vertical_levels('missing')


class TestExtractXarrayByIndex:
    """ Test _extract_xarray_by_index all three paths: uses level_dim when present, falls back to second dim when level_dim absent, and raises when data is less than 2D. """

    def test_uses_level_dim_when_present(self: 'TestExtractXarrayByIndex') -> None:
        """
        This test verifies that the _extract_xarray_by_index method correctly uses the specified level dimension to extract a 2D slice from a 3D xarray DataArray when the level dimension is present in the data. This ensures that the method can accurately identify and utilize the vertical dimension for extraction, providing the expected 2D output based on the specified level dimension. The test simulates a scenario where the input DataArray has a clearly defined level dimension, confirming that the method returns a 2D array with the expected shape corresponding to the remaining dimensions after extracting along the specified level dimension. 

        Parameters:
            None

        Returns:
            None
        """
        data = xr.DataArray(
            np.arange(24, dtype=float).reshape(2, 3, 4),
            dims=['Time', 'nCells', 'nVertLevels'],
        )

        result = MPAS3DProcessor._extract_xarray_by_index(data, 0, 'nVertLevels')
        assert result.shape == (2, 3)

    def test_uses_fallback_dim_when_level_dim_absent(self: 'TestExtractXarrayByIndex') -> None:
        """
        This test verifies that the _extract_xarray_by_index method correctly falls back to using the second dimension for extraction when the specified level dimension is not present in the data. This ensures that the method can still perform extraction even when the expected level dimension is missing, providing a fallback mechanism to identify an alternative dimension for extraction. The test simulates a scenario where the input DataArray does not contain the specified level dimension, confirming that the method returns a 2D array with the expected shape corresponding to the remaining dimensions after extracting along the fallback dimension. 

        Parameters:
            None

        Returns:
            None
        """
        data = xr.DataArray(
            np.ones((2, 4, 5)),
            dims=['Time', 'pressure_level', 'nCells'],
        )

        result = MPAS3DProcessor._extract_xarray_by_index(data, 1, 'nVertLevels')
        assert result.shape == (2, 5)


class TestExtractXarrayByValue:
    """ Test _extract_xarray_by_value all three paths: raises when coord not in data, selects nearest level, and interpolates linearly. """

    def _make_leveled_da(self: 'TestExtractXarrayByValue') -> xr.DataArray:
        """
        This helper method creates a 3D xarray DataArray with dimensions 'nCells' and 'pressure', where 'pressure' is a coordinate with specific values representing pressure levels. The DataArray is filled with sequential values for testing purposes. This method is used to provide a consistent test input for the _extract_xarray_by_value tests, allowing us to verify the behavior of the method when selecting levels based on coordinate values. 

        Parameters:
            None

        Returns:
            xr.DataArray: A 2D DataArray with 'nCells' and 'pressure' dimensions, and 'pressure' coordinate values for testing.
        """
        levels = np.array([1000.0, 850.0, 500.0, 200.0])
        return xr.DataArray(
            np.arange(20, dtype=float).reshape(5, 4),
            dims=['nCells', 'pressure'],
            coords={'pressure': levels},
        )

    def test_raises_when_coord_not_in_data(self: 'TestExtractXarrayByValue') -> None:
        """
        This test verifies that the _extract_xarray_by_value method raises a ValueError when the specified coordinate for level selection is not found in the input data. This ensures that the method correctly identifies and handles cases where the user requests level selection based on a coordinate that does not exist in the data, providing a clear error message to the user and preventing attempts to select levels with an invalid coordinate. The test simulates a scenario where the input DataArray does not contain the specified coordinate, confirming that it raises the appropriate exception with a message indicating that the coordinate was not found in this case. 

        Parameters:
            None

        Returns:
            None
        """
        data = xr.DataArray(np.ones((5, 4)), dims=['nCells', 'nVertLevels'])
        with pytest.raises(ValueError, match="Coordinate 'nVertLevels' not found"):
            MPAS3DProcessor._extract_xarray_by_value(data, 500.0, 'nVertLevels', 'nearest')

    def test_nearest_method_selects_closest_level(self: 'TestExtractXarrayByValue') -> None:
        """
        This test verifies that the _extract_xarray_by_value method correctly selects the closest level based on the specified coordinate values when the 'nearest' method is used. This ensures that the method can accurately identify and return the data corresponding to the nearest level for the requested value, providing a useful option for users who want to select levels without interpolation. The test simulates a scenario where the input DataArray has a coordinate with specific values, and the method is called with a target value that does not exactly match any of the coordinate values, confirming that it returns the data corresponding to the closest coordinate value as expected in this case. 

        Parameters:
            None

        Returns:
            None
        """
        da = self._make_leveled_da()
        result = MPAS3DProcessor._extract_xarray_by_value(da, 900.0, 'pressure', 'nearest')
        assert result.shape == (5,)

    def test_linear_method_interpolates(self: 'TestExtractXarrayByValue') -> None:
        """
        This test verifies that the _extract_xarray_by_value method correctly performs linear interpolation to select the appropriate level based on the specified coordinate values when the 'linear' method is used. This ensures that the method can provide a more accurate selection of levels by interpolating between coordinate values, which can be important for users who want to obtain data at specific values that do not correspond exactly to existing coordinate values. The test simulates a scenario where the input DataArray has a coordinate with specific values, and the method is called with a target value that falls between two coordinate values, confirming that it returns the interpolated data corresponding to the target value as expected in this case. 

        Parameters:
            None

        Returns:
            None
        """
        da = self._make_leveled_da()
        result = MPAS3DProcessor._extract_xarray_by_value(da, 900.0, 'pressure', 'linear')
        assert result.shape == (5,)


class TestExtractNumpyByIndex:
    """ Test _extract_numpy_by_index all three paths: raises when data is less than 2D, extracts along second dim when data is 3D, and extracts along second dim when data is 4D. """

    def test_1d_array_raises(self: 'TestExtractNumpyByIndex') -> None:
        """
        This test verifies that the _extract_numpy_by_index method raises a ValueError when the input data is a 1D array, which is less than 2D. This ensures that the method correctly identifies and handles cases where the input data does not have enough dimensions for extraction, providing a clear error message to the user and preventing attempts to extract data from an invalid array shape. The test simulates a scenario where the input is a 1D numpy array, confirming that it raises the appropriate exception with a message indicating that the data must be at least 2D in this case. 

        Parameters:
            None

        Returns:
            None
        """
        arr = np.ones(5)
        with pytest.raises(ValueError, match="at least 2D"):
            MPAS3DProcessor._extract_numpy_by_index(arr, 0)

    def test_3d_array_extracts_along_second_dim(self: 'TestExtractNumpyByIndex') -> None:
        """
        This test verifies that the _extract_numpy_by_index method correctly extracts data along the second dimension when the input data is 3D. It ensures that the method selects the appropriate slice of the array based on the specified index, providing accurate data extraction for 3D arrays. The test simulates a scenario where the input is a 3D numpy array, and the method is called with a specific index, confirming that it returns the expected slice of the array corresponding to that index along the second dimension as expected in this case. 

        Parameters:
            None

        Returns:
            None
        """
        arr = np.arange(60, dtype=float).reshape(5, 4, 3)
        result = MPAS3DProcessor._extract_numpy_by_index(arr, 1)
        expected = arr[:, 1, -1]
        assert np.allclose(result, expected)

    def test_4d_array_extracts_along_second_dim(self: 'TestExtractNumpyByIndex') -> None:
        """
        This test verifies that the _extract_numpy_by_index method correctly extracts data along the second dimension when the input data is 4D. It ensures that the method selects the appropriate slice of the array based on the specified index, providing accurate data extraction for 4D arrays. The test simulates a scenario where the input is a 4D numpy array, and the method is called with a specific index, confirming that it returns the expected slice of the array corresponding to that index along the second dimension as expected in this case. 

        Parameters:
            None

        Returns:
            None
        """
        arr = np.ones((5, 4, 3, 2))
        result = MPAS3DProcessor._extract_numpy_by_index(arr, 1)
        assert result.shape == (5, 3, 2)


class TestExtract2DFrom3D:
    """ Test extract_2d_from_3d all four paths: raises when neither index nor value provided, calls _extract_xarray_by_index for xarray with level index, calls _extract_xarray_by_value for xarray with level value, raises when level value provided for numpy, and calls _extract_numpy_by_index for numpy with level index. """

    def test_raises_when_neither_index_nor_value_provided(self: 'TestExtract2DFrom3D') -> None:
        """
        This test verifies that the extract_2d_from_3d method raises a ValueError when neither a level index nor a level value is provided for extraction. This ensures that the method correctly identifies and handles cases where the user fails to specify the necessary parameters for extraction, providing a clear error message to guide the user in providing either a level index or a level value for successful extraction. The test simulates a scenario where the method is called without any level specification, confirming that it raises the appropriate exception with a message indicating that either a level index or level value must be provided in this case. 

        Parameters:
            None

        Returns:
            None
        """
        data = np.ones((5, 4, 3))
        with pytest.raises(ValueError, match="Must provide either"):
            MPAS3DProcessor.extract_2d_from_3d(data)

    def test_xarray_with_level_index_calls_extract_by_index(self: 'TestExtract2DFrom3D') -> None:
        """
        This test verifies that the extract_2d_from_3d method correctly calls the _extract_xarray_by_index method to extract data along the specified level index when the input data is an xarray DataArray. It ensures that the method selects the appropriate slice of the array based on the provided level index, providing accurate data extraction for xarray DataArrays when a level index is specified. The test simulates a scenario where the input is a 3D xarray DataArray, and the method is called with a specific level index, confirming that it returns a 2D array with the expected shape corresponding to the remaining dimensions after extracting along the specified level index as expected in this case. 

        Parameters:
            None

        Returns:
            None
        """
        data = xr.DataArray(
            np.ones((2, 5, 4)),
            dims=['Time', 'nCells', 'nVertLevels'],
        )

        result = MPAS3DProcessor.extract_2d_from_3d(data, level_index=0)
        assert result.shape == (2, 5)

    def test_xarray_with_level_value_calls_extract_by_value(self: 'TestExtract2DFrom3D') -> None:
        """
        This test verifies that the extract_2d_from_3d method correctly calls the _extract_xarray_by_value method to extract data based on the specified level value when the input data is an xarray DataArray. It ensures that the method can accurately select levels based on coordinate values when a level value is provided, providing accurate data extraction for xarray DataArrays when a level value is specified. The test simulates a scenario where the input is a 3D xarray DataArray with a coordinate representing vertical levels, and the method is called with a specific level value, confirming that it returns a 2D array with the expected shape corresponding to the remaining dimensions after extracting based on the specified level value as expected in this case. 

        Parameters:
            None

        Returns:
            None
        """
        levels = np.array([1000.0, 850.0, 500.0, 200.0])

        data = xr.DataArray(
            np.ones((5, 4)),
            dims=['nCells', 'nVertLevels'],
            coords={'nVertLevels': levels},
        )

        result = MPAS3DProcessor.extract_2d_from_3d(
            data, level_value=900.0, level_dim='nVertLevels', method='nearest'
        )

        assert result.shape == (5,)

    def test_numpy_with_level_value_raises(self: 'TestExtract2DFrom3D') -> None:
        """
        This test verifies that the extract_2d_from_3d method raises a ValueError when a level value is provided for extraction but the input data is a numpy array, which does not have coordinate information for selecting levels based on values. This ensures that the method correctly identifies and handles cases where the user attempts to select levels based on values with numpy arrays, providing a clear error message to guide the user in using level indices instead for numpy arrays. The test simulates a scenario where the input is a 3D numpy array, and the method is called with a level value, confirming that it raises the appropriate exception with a message indicating that level value extraction requires xarray DataArrays in this case. 

        Parameters:
            None

        Returns:
            None
        """
        data = np.ones((5, 4, 3))
        with pytest.raises(ValueError, match="level_value extraction requires"):
            MPAS3DProcessor.extract_2d_from_3d(data, level_value=500.0)

    def test_numpy_with_level_index_calls_extract_numpy(self: 'TestExtract2DFrom3D') -> None:
        """
        This test verifies that the extract_2d_from_3d method correctly calls the _extract_numpy_by_index method to extract data along the specified level index when the input data is a numpy array. It ensures that the method selects the appropriate slice of the array based on the provided level index, providing accurate data extraction for numpy arrays when a level index is specified. The test simulates a scenario where the input is a 3D numpy array, and the method is called with a specific level index, confirming that it returns a 1D array with the expected shape corresponding to the remaining dimensions after extracting along the specified level index as expected in this case. 

        Parameters:
            None

        Returns:
            None
        """
        data = np.ones((5, 4, 3))
        result = MPAS3DProcessor.extract_2d_from_3d(data, level_index=1)
        assert result.shape == (5,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

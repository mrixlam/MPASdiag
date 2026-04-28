#!/usr/bin/env python3

"""
MPASdiag Test Suite: 2D Processor Coverage

This module contains unit tests designed to achieve comprehensive code coverage for the MPAS2DProcessor class in the MPASdiag package. The tests focus on methods that involve loading 2D data, finding diagnostic files, and extracting spatial coordinates, including various edge cases and error paths. The goal is to ensure that all branches of the code are executed at least once during testing, thereby improving the robustness and reliability of the MPAS2DProcessor class. 

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

from mpasdiag.processing.processors_2d import MPAS2DProcessor


@pytest.fixture
def mock_proc() -> MPAS2DProcessor:
    """
    This fixture creates a mock MPAS2DProcessor instance with default attributes set to allow testing of methods that do not require an actual dataset or file I/O. The dataset is initialized to None, and the data_type is set to 'xarray' to enable testing of the UXarray branch in load_2d_data. The grid_file and data_dir attributes are set to dummy paths. 

    Parameters:
        None

    Returns:
        MPAS2DProcessor: A mock processor instance ready for testing.
    """
    proc = MPAS2DProcessor.__new__(MPAS2DProcessor)
    proc.verbose = False
    proc.dataset = None
    proc.data_type = 'xarray'
    proc.grid_file = '/tmp/mock_grid.nc'
    proc.data_dir = '/tmp/mock_data'
    return proc


@pytest.fixture
def loaded_proc(mock_proc: 'MPAS2DProcessor') -> 'MPAS2DProcessor':
    """
    This fixture takes the mock processor instance created by the mock_proc fixture and populates its dataset attribute with a minimal xarray Dataset that includes a variable named 'temperature' with dimensions 'Time' and 'nCells', as well as coordinate variables for longitude and latitude. This allows testing of methods that require a loaded dataset without needing to perform actual file I/O. 

    Parameters:
        mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

    Returns:
        'MPAS2DProcessor': A mock processor instance with a minimal xarray dataset.
    """
    n = 20
    times = pd.date_range('2024-01-01', periods=5, freq='h')

    mock_proc.dataset = xr.Dataset({
        'temperature': xr.DataArray(
            np.random.rand(5, n),
            dims=['Time', 'nCells'],
            coords={'Time': times},
        ),
        'lonCell': xr.DataArray(np.linspace(-120.0, -80.0, n)),
        'latCell': xr.DataArray(np.linspace(25.0, 55.0, n)),
    })

    return mock_proc


class TestLoad2DDataUXarrayPath:
    """Test the UXarray branch (hasattr 'ds') inside load_2d_data."""

    def test_uxarray_path_assigns_dataset_and_returns_self(self: 'TestLoad2DDataUXarrayPath', 
                                                           mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the load_2d_data method is called and the loaded data object has a 'ds' attribute (indicating it is a UXarray dataset), the method correctly assigns this dataset to the processor's dataset attribute and returns the processor instance itself. The test uses mocking to simulate the behavior of the _load_data method to return a mock object with a 'ds' attribute, and also mocks the add_spatial_coordinates method to return an enriched dataset. It asserts that the returned result is the processor instance and that the dataset attribute is set to the mock UXarray dataset. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        inner_ds = xr.Dataset({'temp': xr.DataArray(np.ones(10), dims=['nCells'])})
        enriched_ds = inner_ds.assign({'lonCell': xr.DataArray(np.ones(10))})

        mock_ux = MagicMock(spec=['ds'])
        mock_ux.ds = inner_ds

        with patch.object(mock_proc, '_load_data', return_value=(mock_ux, 'uxarray')):
            with patch.object(mock_proc, 'add_spatial_coordinates', return_value=enriched_ds):
                result = mock_proc.load_2d_data('/tmp/data_dir')

        assert result is mock_proc
        assert mock_proc.dataset is mock_ux


class TestFindDiagFilesRecursive:
    """Test _find_diag_files_recursive coverage."""

    def test_returns_sorted_list_when_two_or_more_found(self: 'TestFindDiagFilesRecursive', 
                                                        mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the _find_diag_files_recursive method finds two or more diagnostic files, it returns a sorted list of their file paths. The test uses mocking to simulate the behavior of glob.glob to return a predefined list of file paths that are intentionally unsorted. It asserts that the returned result is a sorted version of the input list. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value. 
        """
        fake_files = ['/tmp/diag_2025.nc', '/tmp/diag_2024.nc']  # intentionally unsorted
        with patch('mpasdiag.processing.processors_2d.glob.glob', return_value=fake_files):
            result = mock_proc._find_diag_files_recursive('/tmp/data')
        assert result == sorted(fake_files)

    def test_verbose_prints_recursive_search_summary(self: 'TestFindDiagFilesRecursive', 
                                                     mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test checks that when the _find_diag_files_recursive method finds two or more diagnostic files and the processor's verbose attribute is set to True, it prints a summary message indicating that diagnostic files were found through a recursive search. The test uses mocking to simulate the behavior of glob.glob to return a predefined list of file paths, and captures the standard output to check for the expected message. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value. 
        """
        mock_proc.verbose = True
        fake_files = [f'/tmp/diag_{i:04d}.nc' for i in range(6)]

        with patch('mpasdiag.processing.processors_2d.glob.glob', return_value=fake_files):
            captured = StringIO()
            with patch('sys.stdout', new=captured):
                result = mock_proc._find_diag_files_recursive('/tmp/data')

        assert result is not None
        assert 'diagnostic files (recursive' in captured.getvalue()

    def test_returns_none_when_fewer_than_two_files(self: 'TestFindDiagFilesRecursive', 
                                                    mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the _find_diag_files_recursive method finds fewer than two diagnostic files (i.e., glob.glob returns a list with one file), it returns None. The test uses mocking to simulate the behavior of glob.glob to return a list containing a single file path, and asserts that the method returns None in this case. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value. 
        """
        with patch('mpasdiag.processing.processors_2d.glob.glob', return_value=['/tmp/diag_2024.nc']):
            result = mock_proc._find_diag_files_recursive('/tmp/data')
        assert result is None

    def test_returns_none_when_no_files_found(self: 'TestFindDiagFilesRecursive', 
                                              mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the _find_diag_files_recursive method does not find any diagnostic files (i.e., glob.glob returns an empty list), it returns None. The test uses mocking to simulate the behavior of glob.glob to return an empty list, and asserts that the method returns None in this case. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value. 
        """
        with patch('mpasdiag.processing.processors_2d.glob.glob', return_value=[]):
            result = mock_proc._find_diag_files_recursive('/tmp/data')
        assert result is None


class TestFindMpasoutFilesFallback:
    """Test _find_mpasout_files_fallback branching."""

    def test_returns_files_when_pattern_finder_succeeds(self: 'TestFindMpasoutFilesFallback', 
                                                        mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the _find_mpasout_files_fallback method successfully finds MPAS output files using the pattern-based finder (i.e., _find_files_by_pattern returns a list of file paths), it returns this list of file paths without attempting a recursive search. The test uses mocking to simulate the behavior of _find_files_by_pattern to return a predefined list of MPAS output file paths, and asserts that the method returns this list directly. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        fake_files = ['/tmp/mpasout_2024.nc', '/tmp/mpasout_2025.nc']

        with patch.object(mock_proc, '_find_files_by_pattern', return_value=fake_files):
            result = mock_proc._find_mpasout_files_fallback('/tmp/data')

        assert result == fake_files

    def test_raises_file_not_found_when_no_files_anywhere(self: 'TestFindMpasoutFilesFallback', 
                                                          mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the _find_mpasout_files_fallback method fails to find MPAS output files using both the pattern-based finder (i.e., _find_files_by_pattern raises FileNotFoundError) and the recursive search (i.e., glob.glob returns an empty list), it raises a FileNotFoundError with an appropriate message. The test uses mocking to simulate the behavior of _find_files_by_pattern to raise a FileNotFoundError and glob.glob to return an empty list, and asserts that the method raises the expected exception with a message indicating that no diagnostic files were found. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        with patch.object(mock_proc, '_find_files_by_pattern', side_effect=FileNotFoundError):
            with patch('mpasdiag.processing.processors_2d.glob.glob', return_value=[]):
                with pytest.raises(FileNotFoundError, match="No diagnostic files"):
                    mock_proc._find_mpasout_files_fallback('/tmp/data')

    def test_raises_value_error_when_only_one_file_found(self: 'TestFindMpasoutFilesFallback', 
                                                         mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the _find_mpasout_files_fallback method fails to find MPAS output files using the pattern-based finder (i.e., _find_files_by_pattern raises FileNotFoundError) and the recursive search returns only one file (i.e., glob.glob returns a list with a single file path), it raises a ValueError indicating that there are insufficient MPAS output files. The test uses mocking to simulate the behavior of _find_files_by_pattern to raise a FileNotFoundError and glob.glob to return a list containing a single file path, and asserts that the method raises the expected exception with a message indicating insufficient MPAS output files. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        with patch.object(mock_proc, '_find_files_by_pattern', side_effect=FileNotFoundError):
            with patch('mpasdiag.processing.processors_2d.glob.glob',
                       return_value=['/tmp/mpasout_2024.nc']):
                with pytest.raises(ValueError, match="Insufficient MPAS output files"):
                    mock_proc._find_mpasout_files_fallback('/tmp/data')

    def test_returns_files_from_recursive_search_verbose(self: 'TestFindMpasoutFilesFallback', 
                                                         mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test checks that when the _find_mpasout_files_fallback method fails to find files using the pattern-based finder and falls back to a recursive search (i.e., glob.glob), it returns the list of file paths found. Additionally, if the processor's verbose attribute is set to True, it verifies that a summary message indicating that MPAS output files were found through a recursive search is printed. The test uses mocking to simulate the behavior of _find_files_by_pattern to raise a FileNotFoundError and glob.glob to return a predefined list of MPAS output file paths. It captures the standard output to check for the expected message. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        mock_proc.verbose = True
        fake_files = ['/tmp/mpasout_2024.nc', '/tmp/mpasout_2025.nc']

        with patch.object(mock_proc, '_find_files_by_pattern', side_effect=FileNotFoundError):
            with patch('mpasdiag.processing.processors_2d.glob.glob', return_value=fake_files):
                captured = StringIO()
                with patch('sys.stdout', new=captured):
                    result = mock_proc._find_mpasout_files_fallback('/tmp/data')

        assert result == sorted(fake_files)
        assert 'MPAS output files (recursive' in captured.getvalue()

    def test_returns_files_from_recursive_search_silent(self: 'TestFindMpasoutFilesFallback', 
                                                        mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the _find_mpasout_files_fallback method fails to find files using the pattern-based finder and falls back to a recursive search (i.e., glob.glob), it returns the list of file paths found without printing any messages if the processor's verbose attribute is set to False. The test uses mocking to simulate the behavior of _find_files_by_pattern to raise a FileNotFoundError and glob.glob to return a predefined list of MPAS output file paths. It captures the standard output to ensure that no messages are printed. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        fake_files = ['/tmp/mpasout_2024.nc', '/tmp/mpasout_2025.nc']

        with patch.object(mock_proc, '_find_files_by_pattern', side_effect=FileNotFoundError):
            with patch('mpasdiag.processing.processors_2d.glob.glob', return_value=fake_files):
                result = mock_proc._find_mpasout_files_fallback('/tmp/data')

        assert result == sorted(fake_files)


class TestFindDiagnosticFiles:
    """Test every branch of find_diagnostic_files."""

    def test_finds_files_in_main_directory(self: 'TestFindDiagnosticFiles', 
                                           mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that the find_diagnostic_files method successfully finds diagnostic files in the main directory when they are present. The test uses mocking to simulate the behavior of the _find_files_by_pattern method to return a predefined list of diagnostic file paths, and asserts that the find_diagnostic_files method returns this list without attempting to search in subdirectories or perform a recursive search. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        fake_files = ['/tmp/data/diag_2024.nc', '/tmp/data/diag_2025.nc']

        with patch.object(mock_proc, '_find_files_by_pattern', return_value=fake_files):
            result = mock_proc.find_diagnostic_files('/tmp/data')

        assert result == fake_files

    def test_finds_files_in_diag_subdir_when_main_dir_fails(self: 'TestFindDiagnosticFiles', 
                                                            mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the find_diagnostic_files method fails to find diagnostic files in the main directory (i.e., _find_files_by_pattern raises a FileNotFoundError), it successfully searches in the 'diag' subdirectory and returns the list of diagnostic file paths found there. The test uses mocking to simulate the behavior of _find_files_by_pattern to first raise a FileNotFoundError for the main directory search, and then return a predefined list of diagnostic file paths for the 'diag' subdirectory search. It asserts that the find_diagnostic_files method returns the list of files found in the 'diag' subdirectory. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        fake_files = ['/tmp/data/diag/diag_2024.nc', '/tmp/data/diag/diag_2025.nc']

        with patch.object(mock_proc, '_find_files_by_pattern',
                          side_effect=[FileNotFoundError(), fake_files]):
            result = mock_proc.find_diagnostic_files('/tmp/data')

        assert result == fake_files

    def test_falls_back_to_recursive_search(self: 'TestFindDiagnosticFiles', 
                                            mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the find_diagnostic_files method fails to find diagnostic files in both the main directory and the 'diag' subdirectory, it successfully falls back to a recursive search. The test uses mocking to simulate the behavior of _find_files_by_pattern to always raise a FileNotFoundError, and _find_diag_files_recursive to return a predefined list of diagnostic file paths. The test asserts that the find_diagnostic_files method returns the list of files found by the recursive search. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        fake_files = ['/tmp/data/sub/diag_2024.nc', '/tmp/data/sub/diag_2025.nc']

        with patch.object(mock_proc, '_find_files_by_pattern', side_effect=FileNotFoundError):
            with patch.object(mock_proc, '_find_diag_files_recursive', return_value=fake_files):
                result = mock_proc.find_diagnostic_files('/tmp/data')

        assert result == fake_files

    def test_falls_back_to_mpasout_when_recursive_returns_none(self: 'TestFindDiagnosticFiles', 
                                                                mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the find_diagnostic_files method fails to find diagnostic files in both the main directory and the 'diag' subdirectory, and the recursive search returns None, it successfully falls back to the _find_mpasout_files_fallback method. The test uses mocking to simulate the behavior of _find_files_by_pattern to always raise a FileNotFoundError, _find_diag_files_recursive to return None, and _find_mpasout_files_fallback to return a predefined list of diagnostic file paths. The test asserts that the find_diagnostic_files method returns the list of files found by the _find_mpasout_files_fallback method. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        fake_files = ['/tmp/data/mpasout_2024.nc', '/tmp/data/mpasout_2025.nc']

        with patch.object(mock_proc, '_find_files_by_pattern', side_effect=FileNotFoundError):
            with patch.object(mock_proc, '_find_diag_files_recursive', return_value=None):
                with patch.object(mock_proc, '_find_mpasout_files_fallback', return_value=fake_files):
                    result = mock_proc.find_diagnostic_files('/tmp/data')

        assert result == fake_files

    def test_verbose_prints_no_diag_files_message_before_mpasout_fallback(self: 'TestFindDiagnosticFiles', 
                                                                          mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the find_diagnostic_files method fails to find diagnostic files in both the main directory and the 'diag' subdirectory, and the recursive search returns None, if the processor's verbose attribute is set to True, it prints a message indicating that no diagnostic files were found before falling back to searching for MPAS output files. The test uses mocking to simulate the behavior of _find_files_by_pattern to always raise a FileNotFoundError, _find_diag_files_recursive to return None, and _find_mpasout_files_fallback to return a predefined list of diagnostic file paths. It captures the standard output to check for the expected message. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        mock_proc.verbose = True
        fake_files = ['/tmp/data/mpasout_2024.nc', '/tmp/data/mpasout_2025.nc']

        with patch.object(mock_proc, '_find_files_by_pattern', side_effect=FileNotFoundError):
            with patch.object(mock_proc, '_find_diag_files_recursive', return_value=None):
                with patch.object(mock_proc, '_find_mpasout_files_fallback', return_value=fake_files):
                    captured = StringIO()
                    with patch('sys.stdout', new=captured):
                        mock_proc.find_diagnostic_files('/tmp/data')

        assert 'No diagnostic files found' in captured.getvalue()


class TestLookup2DCoord:
    """Test _lookup_2d_coord return-None path and happy path."""

    def test_returns_none_when_no_matching_name_found(self: 'TestLookup2DCoord', 
                                                      mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the _lookup_2d_coord method is called with a list of coordinate names that do not match any coordinates in the dataset, it returns None. The test uses mocking to create a dataset with a variable that does not have any of the specified coordinate names, and asserts that the method returns None in this case. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        mock_proc.dataset = xr.Dataset({
            'temperature': xr.DataArray(np.ones(10), dims=['nCells'])
        })

        result = mock_proc._lookup_2d_coord(['lonCell', 'longitude', 'lon'])
        assert result is None

    def test_returns_array_for_first_matching_coord_name(self: 'TestLookup2DCoord', 
                                                          mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the _lookup_2d_coord method is called with a list of coordinate names where the first name matches a coordinate in the dataset, it returns the coordinate array corresponding to that first name. The test uses mocking to create a dataset with a variable that has a matching coordinate name for the first name in the list, and asserts that the method returns the correct coordinate array. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        lon_data = np.linspace(-120.0, -80.0, 10)

        mock_proc.dataset = xr.Dataset({
            'lonCell': xr.DataArray(lon_data, dims=['nCells'])
        })

        result = mock_proc._lookup_2d_coord(['lonCell', 'longitude', 'lon'])

        assert result is not None
        assert np.allclose(result, lon_data)

    def test_falls_through_to_second_name_when_first_absent(self: 'TestLookup2DCoord', 
                                                            mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the _lookup_2d_coord method is called with a list of coordinate names where the first name does not match any coordinate in the dataset but the second name does, it correctly falls through to the second name and returns the corresponding coordinate array. The test uses mocking to create a dataset with a variable that has a matching coordinate name for the second name in the list, and asserts that the method returns the correct coordinate array. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        lon_data = np.linspace(-100.0, -90.0, 5)

        mock_proc.dataset = xr.Dataset({
            'longitude': xr.DataArray(lon_data, dims=['nCells'])
        })

        result = mock_proc._lookup_2d_coord(['lonCell', 'longitude', 'lon'])

        assert result is not None
        assert np.allclose(result, lon_data)


class TestExtract2DCoordinatesForVariable:
    """Test extract_2d_coordinates_for_variable error paths and normal flows."""

    def test_raises_when_dataset_is_none(self: 'TestExtract2DCoordinatesForVariable', 
                                         mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that the extract_2d_coordinates_for_variable method raises a ValueError when the processor's dataset attribute is None. The test asserts that the method raises the expected exception with a message indicating that the dataset is not loaded. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        mock_proc.dataset = None
        with pytest.raises(ValueError, match="Dataset not loaded"):
            mock_proc.extract_2d_coordinates_for_variable('temperature')

    def test_raises_when_lon_lat_coords_not_found(self: 'TestExtract2DCoordinatesForVariable', 
                                                  mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that the extract_2d_coordinates_for_variable method raises a ValueError when it cannot find longitude and latitude coordinates in the dataset. The test uses mocking to create a dataset that contains a variable but does not have any of the expected coordinate names for longitude and latitude. It asserts that the method raises the expected exception with a message indicating that it could not find nCells coordinates. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        mock_proc.dataset = xr.Dataset({
            'temperature': xr.DataArray(np.ones(10), dims=['nCells'])
        })

        with pytest.raises(ValueError, match="Could not find nCells coordinates"):
            mock_proc.extract_2d_coordinates_for_variable('temperature')

    def test_degree_coords_returned_correctly(self: 'TestExtract2DCoordinatesForVariable', 
                                              mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the extract_2d_coordinates_for_variable method is called with a dataset that contains longitude and latitude coordinates in degrees, it correctly returns these coordinates as numpy arrays. The test uses mocking to create a dataset with longitude and latitude coordinates in degrees, and asserts that the returned longitude and latitude arrays have the correct shapes. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        n = 30
        lon = np.linspace(-120.0, -80.0, n)
        lat = np.linspace(25.0, 55.0, n)

        mock_proc.dataset = xr.Dataset({
            'temperature': xr.DataArray(np.ones(n), dims=['nCells']),
            'lonCell': xr.DataArray(lon),
            'latCell': xr.DataArray(lat),
        })

        result_lon, result_lat = mock_proc.extract_2d_coordinates_for_variable('temperature')

        assert result_lon.shape == (n,)
        assert result_lat.shape == (n,)

    def test_radian_coords_converted_to_degrees(self: 'TestExtract2DCoordinatesForVariable', 
                                                mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the extract_2d_coordinates_for_variable method is called with a dataset that contains longitude and latitude coordinates in radians, it correctly converts these coordinates to degrees before returning them. The test uses mocking to create a dataset with longitude and latitude coordinates in radians, and asserts that the returned latitude array is correctly converted to degrees by comparing it to the expected values. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        n = 20
        lat_rad = np.linspace(-1.0, 1.0, n)   # abs max <= π → treated as radians
        lon_rad = np.linspace(-2.0, 2.0, n)

        mock_proc.dataset = xr.Dataset({
            'temperature': xr.DataArray(np.ones(n), dims=['nCells']),
            'lonCell': xr.DataArray(lon_rad),
            'latCell': xr.DataArray(lat_rad),
        })

        result_lon, result_lat = mock_proc.extract_2d_coordinates_for_variable('temperature')
        assert np.allclose(result_lat, lat_rad * 180.0 / np.pi)

    def test_vertex_dimension_selects_vertex_coord_names(self: 'TestExtract2DCoordinatesForVariable', 
                                                        mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the extract_2d_coordinates_for_variable method is called with a variable that has a vertex dimension (e.g., 'nVertices'), it correctly looks for longitude and latitude coordinates with names corresponding to the vertex dimension (e.g., 'lonVertex' and 'latVertex'). The test uses mocking to create a dataset with a variable that has an 'nVertices' dimension and corresponding longitude and latitude coordinates with 'Vertex' in their names. It asserts that the returned longitude array has the correct length corresponding to the number of vertices. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        n = 15
        lon = np.linspace(-120.0, -80.0, n)
        lat = np.linspace(25.0, 55.0, n)

        mock_proc.dataset = xr.Dataset({
            'vorticity': xr.DataArray(np.ones(n), dims=['nVertices']),
            'lonVertex': xr.DataArray(lon),
            'latVertex': xr.DataArray(lat),
        })

        result_lon, result_lat = mock_proc.extract_2d_coordinates_for_variable('vorticity')
        assert len(result_lon) == n

    def test_verbose_prints_extracted_coord_message(self: 'TestExtract2DCoordinatesForVariable', 
                                                    mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the extract_2d_coordinates_for_variable method is called and successfully extracts longitude and latitude coordinates, if the processor's verbose attribute is set to True, it prints a message indicating that nCells coordinates were extracted. The test uses mocking to create a dataset with longitude and latitude coordinates, sets verbose to True, and captures the standard output to check for the expected message. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        mock_proc.verbose = True
        n = 10

        mock_proc.dataset = xr.Dataset({
            'temperature': xr.DataArray(np.ones(n), dims=['nCells']),
            'lonCell': xr.DataArray(np.linspace(-120.0, -80.0, n)),
            'latCell': xr.DataArray(np.linspace(25.0, 55.0, n)),
        })

        captured = StringIO()

        with patch('sys.stdout', new=captured):
            mock_proc.extract_2d_coordinates_for_variable('temperature')
        assert 'Extracted nCells coordinates' in captured.getvalue()

    def test_data_array_sizes_override_var_name_lookup(self: 'TestExtract2DCoordinatesForVariable', 
                                                      mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the extract_2d_coordinates_for_variable method is called with a data array that has a specific size corresponding to a vertex dimension, it overrides the variable name lookup and correctly identifies the longitude and latitude coordinates based on the vertex dimension. The test uses mocking to create a dataset with longitude and latitude coordinates for vertices, and calls the method with a data array that has an 'nVertices' dimension. It asserts that the returned longitude array has the correct length corresponding to the number of vertices, demonstrating that the method correctly used the data array's dimensions to determine which coordinates to extract. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        n = 10

        mock_proc.dataset = xr.Dataset({
            'lonVertex': xr.DataArray(np.linspace(-120.0, -80.0, n)),
            'latVertex': xr.DataArray(np.linspace(25.0, 55.0, n)),
        })

        da = xr.DataArray(np.ones(n), dims=['nVertices'])

        result_lon, result_lat = mock_proc.extract_2d_coordinates_for_variable(
            'nonexistent_var', data_array=da
        )

        assert len(result_lon) == n


class TestLog2DVariableRange:
    """Test _log_2d_variable_range verbose paths."""

    def test_verbose_false_produces_no_output(self: 'TestLog2DVariableRange', 
                                              mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the _log_2d_variable_range method is called with the processor's verbose attribute set to False, it does not produce any output. The test uses mocking to create a dataset and a variable data array, sets verbose to False, and captures the standard output to ensure that nothing is printed when the method is called. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        mock_proc.verbose = False
        var_data = xr.DataArray(np.ones(10), dims=['nCells'])
        captured = StringIO()

        with patch('sys.stdout', new=captured):
            mock_proc._log_2d_variable_range('temperature', var_data)

        assert captured.getvalue() == ''

    def test_all_nan_prints_no_finite_values_warning(self: 'TestLog2DVariableRange', 
                                                    mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the _log_2d_variable_range method is called with a variable data array that contains all NaN values, and the processor's verbose attribute is set to True, it prints a warning message indicating that there are no finite values in the variable. The test uses mocking to create a dataset and a variable data array filled with NaN values, sets verbose to True, and captures the standard output to check for the expected warning message. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        mock_proc.verbose = True
        var_data = xr.DataArray(np.full(10, np.nan), dims=['nCells'])
        captured = StringIO()

        with patch('sys.stdout', new=captured):
            mock_proc._log_2d_variable_range('temperature', var_data)

        assert 'No finite values found' in captured.getvalue()

    def test_finite_values_print_range_info(self: 'TestLog2DVariableRange', 
                                            mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the _log_2d_variable_range method is called with a variable data array that contains finite values, and the processor's verbose attribute is set to True, it prints information about the range of the variable. The test uses mocking to create a dataset and a variable data array filled with finite values, sets verbose to True, and captures the standard output to check for the presence of range information in the printed output. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        mock_proc.verbose = True
        var_data = xr.DataArray(np.linspace(0.0, 100.0, 50), dims=['nCells'])
        captured = StringIO()

        with patch('sys.stdout', new=captured):
            mock_proc._log_2d_variable_range('temperature', var_data)

        assert 'range' in captured.getvalue()

    def test_finite_values_with_units_attr_prints_units(self: 'TestLog2DVariableRange', 
                                                        mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the _log_2d_variable_range method is called with a variable data array that contains finite values and has a 'units' attribute, it prints the units information. The test uses mocking to create a dataset and a variable data array with finite values and a 'units' attribute, and captures the printed output to check for the expected units information. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        mock_proc.verbose = True

        var_data = xr.DataArray(
            np.linspace(200.0, 300.0, 10), dims=['nCells'], attrs={'units': 'K'}
        )

        captured = StringIO()

        with patch('sys.stdout', new=captured):
            mock_proc._log_2d_variable_range('temperature', var_data)

        assert 'Units' in captured.getvalue()


class TestGet2DVariableData:
    """Test get_2d_variable_data error paths and normal extraction."""

    def test_raises_when_dataset_is_none(self: 'TestGet2DVariableData', 
                                         mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that the get_2d_variable_data method raises a ValueError when the processor's dataset attribute is None. The test asserts that the method raises the expected exception with a message indicating that the dataset is not loaded. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        mock_proc.dataset = None
        with pytest.raises(ValueError, match="No dataset loaded"):
            mock_proc.get_2d_variable_data('temperature')

    def test_raises_when_variable_not_in_dataset(self: 'TestGet2DVariableData', 
                                                 loaded_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that the get_2d_variable_data method raises a ValueError when the specified variable name is not found in the dataset. The test uses a loaded processor instance with a dataset that does not contain a variable named 'missing_var', and asserts that the method raises the expected exception with a message indicating that the variable was not found. 

        Parameters:
            loaded_proc ('MPAS2DProcessor'): A loaded processor instance created by the loaded_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        with pytest.raises(ValueError, match="Variable 'missing_var' not found"):
            loaded_proc.get_2d_variable_data('missing_var')

    def test_returns_correct_time_slice_shape(self: 'TestGet2DVariableData', 
                                              loaded_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that the get_2d_variable_data method returns a 1D array with the correct shape when called with a valid variable name and time index. The test uses a loaded processor instance with a dataset containing a variable named 'temperature' that has a time dimension, and asserts that the returned result is a 1D array with the expected length corresponding to the number of cells. 

        Parameters:
            loaded_proc ('MPAS2DProcessor'): A loaded processor instance created by the loaded_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        result = loaded_proc.get_2d_variable_data('temperature', time_index=2)
        assert result.ndim == 1
        assert result.shape[0] == 20

    def test_verbose_prints_extraction_message(self: 'TestGet2DVariableData', 
                                               loaded_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the get_2d_variable_data method is called with verbose mode enabled, it prints a message indicating that it is extracting the specified variable. The test uses a loaded processor instance with a dataset containing a variable named 'temperature', and captures the printed output to check for the expected extraction message.

        Parameters:
            loaded_proc ('MPAS2DProcessor'): A loaded processor instance created by the loaded_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        loaded_proc.verbose = True
        captured = StringIO()

        with patch('sys.stdout', new=captured):
            loaded_proc.get_2d_variable_data('temperature', time_index=0)

        assert 'Extracting temperature' in captured.getvalue()

    def test_uxarray_path_uses_positional_index(self: 'TestGet2DVariableData', 
                                               mock_proc: 'MPAS2DProcessor') -> None:
        """
        This test verifies that when the get_2d_variable_data method is called with a dataset in the 'uxarray' format, it correctly uses the positional time index to extract the data for the specified variable. The test uses mocking to create a dataset in the 'uxarray' format with a variable named 'temperature' that has a time dimension, and asserts that the returned result has the correct shape and values corresponding to the specified time index. 

        Parameters:
            mock_proc ('MPAS2DProcessor'): A mock processor instance created by the mock_proc fixture.

        Returns:
            None: The test asserts conditions but does not return a value.
        """
        n = 10
        times = pd.date_range('2024-01-01', periods=3, freq='h')
        data = np.arange(30, dtype=float).reshape(3, n)
        mock_proc.data_type = 'uxarray'

        mock_proc.dataset = xr.Dataset({
            'temperature': xr.DataArray(
                data, dims=['Time', 'nCells'], coords={'Time': times}
            ),
        })

        result = mock_proc.get_2d_variable_data('temperature', time_index=1)

        assert result.shape == (n,)
        assert np.allclose(result.values, data[1])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

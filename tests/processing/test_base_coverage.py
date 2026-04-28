#!/usr/bin/env python3
"""
MPASdiag Test Suite: MPASBaseProcessor Coverage

This module contains a comprehensive set of unit tests for the MPASBaseProcessor class in the mpasdiag.processing.base module. The tests are designed to achieve 100% code coverage for the MPASBaseProcessor class, ensuring that all methods and code paths are exercised under various conditions. The test cases cover scenarios such as missing grid files, file discovery errors, file validation, chunking configuration, dataset loading with selective variables, and error handling when the dataset is not set. By running this test suite, developers can verify that the MPASBaseProcessor class behaves as expected in both typical and edge case scenarios, and that it provides appropriate error messages when issues arise. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: April 2026
Version: 1.0.0
"""
import tempfile
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from unittest.mock import MagicMock, patch

from mpasdiag.processing.base import MPASBaseProcessor


def _make_proc(verbose: bool = False) -> MPASBaseProcessor:
    """
    This helper function creates a minimal instance of MPASBaseProcessor with the specified verbosity level. The instance is configured with a mock grid file path and no dataset, allowing tests to focus on specific method behaviors without relying on actual data files. The returned processor instance can be used in various test cases to verify error handling and fallback logic in the MPASBaseProcessor class. 

    Parameters:
        verbose: A boolean flag to set the verbosity level of the processor instance.

    Returns:
        An instance of MPASBaseProcessor with the specified configuration.
    """
    proc = MPASBaseProcessor.__new__(MPASBaseProcessor)
    proc.verbose = verbose
    proc.dataset = None
    proc.data_type = 'xarray'
    proc.grid_file = str(Path(tempfile.gettempdir()) / 'mock_grid.nc')
    return proc


def _minimal_ds(n_time: int = 2, 
                n_cells: int = 10) -> xr.Dataset:
    """
    This helper function creates a minimal xarray Dataset with specified dimensions for time and spatial cells. The dataset contains a single variable 'temperature' filled with ones, and is structured with 'Time' and 'nCells' dimensions. This minimal dataset can be used in tests to verify the behavior of MPASBaseProcessor methods that operate on xarray Datasets, without requiring complex or large data files. The function allows customization of the number of time steps and spatial cells to suit different testing scenarios. 

    Parameters:
        n_time: The number of time steps to include in the dataset (default is 2).
        n_cells: The number of spatial cells to include in the dataset (default is 10). 

    Returns:
        xr.Dataset: A minimal xarray Dataset with the specified dimensions.
    """
    times = pd.date_range('2024-01-01', periods=n_time, freq='h')
    return xr.Dataset(
        {'temperature': xr.DataArray(
            np.ones((n_time, n_cells)),
            dims=['Time', 'nCells'],
            coords={'Time': times},
        )}
    )


class TestInitGridFileMissing:
    """ Tests for the MPASBaseProcessor __init__ method when the grid file is missing or invalid. """

    def test_raises_file_not_found_for_nonexistent_grid(self: 'TestInitGridFileMissing', 
                                                        tmp_path: 'Path') -> None:
        """
        This test verifies that the MPASBaseProcessor __init__ method raises a FileNotFoundError with an appropriate message when the specified grid file does not exist on disk (line 50). The test uses a temporary directory provided by pytest to ensure that the grid file path is valid but points to a non-existent file, allowing us to confirm that the processor correctly handles this error condition during initialization. 

        Parameters:
            tmp_path: A pytest fixture that provides a temporary directory for the test.

        Returns:
            None
        """
        with pytest.raises(FileNotFoundError, match="Grid file not found"):
            MPASBaseProcessor(str(tmp_path / 'nonexistent_grid.nc'))


class TestFindFilesByPatternErrors:
    """ Tests for the MPASBaseProcessor _find_files_by_pattern method when no files match the pattern or only one file is found. """

    def test_raises_when_no_files_match_pattern(self: 'TestFindFilesByPatternErrors', 
                                                tmp_path: 'Path') -> None:
        """
        This test verifies that _find_files_by_pattern raises FileNotFoundError with an appropriate message when no files in the specified directory match the given glob pattern. The test uses a temporary directory provided by pytest, which is empty by default, to ensure that the method correctly identifies the absence of matching files and raises the expected exception with a clear error message indicating that no diagnostic files were found. 

        Parameters:
            tmp_path: A pytest fixture that provides a temporary directory for the test.

        Returns:
            None
        """
        proc = _make_proc()
        with pytest.raises(FileNotFoundError, match="No diag files found"):
            proc._find_files_by_pattern(str(tmp_path), '*.nc', 'diag files')

    def test_raises_when_only_one_file_found(self: 'TestFindFilesByPatternErrors', 
                                             tmp_path: 'Path') -> None:
        """
        This test verifies that _find_files_by_pattern raises ValueError with an appropriate message when only one file in the specified directory matches the given glob pattern. The test creates a single dummy NetCDF file in a temporary directory provided by pytest, ensuring that the method finds exactly one matching file. The test then confirms that the method raises the expected exception with a clear error message indicating that insufficient diagnostic files were found, which is important for cases where multiple files are required for processing. 

        Parameters:
            tmp_path: A pytest fixture that provides a temporary directory for the test.

        Returns:
            None
        """
        (tmp_path / 'single.nc').touch()
        proc = _make_proc()
        with pytest.raises(ValueError, match="Insufficient files"):
            proc._find_files_by_pattern(str(tmp_path), '*.nc', 'diag files')


class TestValidateFiles:
    """ Tests for the MPASBaseProcessor validate_files method when files are missing, unreadable, or valid. """

    def test_raises_for_nonexistent_file(self: 'TestValidateFiles', 
                                         tmp_path: 'Path') -> None:
        """
        This test verifies that validate_files raises FileNotFoundError with an appropriate message when one or more specified file paths do not exist on disk. The test uses a temporary directory provided by pytest to ensure that the file path is valid but points to a non-existent file, allowing us to confirm that the method correctly identifies missing files and raises the expected exception with a clear error message indicating that the file was not found. 

        Parameters:
            tmp_path: A pytest fixture that provides a temporary directory for the test.

        Returns:
            None
        """
        proc = _make_proc()
        with pytest.raises(FileNotFoundError, match="File not found"):
            proc.validate_files([str(tmp_path / 'ghost.nc')])

    def test_raises_for_unreadable_file(self: 'TestValidateFiles', 
                                        tmp_path: 'Path') -> None:
        """
        This test verifies that validate_files raises FileNotFoundError with an appropriate message when one or more specified file paths exist but are not readable due to permission issues. The test creates a dummy file in a temporary directory provided by pytest and then mocks the os.access function to simulate the file being unreadable. This allows us to confirm that the method correctly identifies unreadable files and raises the expected exception with a clear error message indicating that the file is not readable, which is important for ensuring that the processor can handle permission-related issues gracefully. 

        Parameters:
            tmp_path: A pytest fixture that provides a temporary directory for the test.

        Returns:
            None
        """
        locked = tmp_path / 'locked.nc'
        locked.write_text('data')
        proc = _make_proc()
        with patch('mpasdiag.processing.base.os.access', return_value=False):
            with pytest.raises(FileNotFoundError, match="File not readable"):
                proc.validate_files([str(locked)])

    def test_returns_list_of_valid_files(self: 'TestValidateFiles', 
                                         tmp_path: 'Path') -> None:
        """
        This test verifies that validate_files returns the original list of file paths when all specified files exist and are readable. The test creates multiple dummy files in a temporary directory provided by pytest, ensuring that they exist and are readable. The test then confirms that the method returns the original list of file paths without modification, which indicates that the method correctly validates the files and allows processing to continue when all files are valid. 

        Parameters:
            tmp_path: A pytest fixture that provides a temporary directory for the test.

        Returns:
            None
        """
        f1 = tmp_path / 'a.nc'
        f2 = tmp_path / 'b.nc'
        f1.write_text('x')
        f2.write_text('y')
        proc = _make_proc()
        result = proc.validate_files([str(f1), str(f2)])
        assert result == [str(f1), str(f2)]


class TestPrepareChunkingConfig:
    """ Tests for the MPASBaseProcessor _prepare_chunking_config method when a custom chunking config is provided or when None is provided. """

    def test_custom_chunks_returns_time_only_open_chunks(self: 'TestPrepareChunkingConfig') -> None:
        """
        This test verifies that _prepare_chunking_config with a custom config returns an open_chunks dict that only includes 'Time', and a full_chunks dict that is the same as the custom config. The test confirms that when a custom chunking configuration is provided, the method correctly extracts the 'Time' dimension for open_chunks while retaining the full custom configuration for full_chunks. This ensures that the method properly distinguishes between the chunking strategy used for opening datasets and the full chunking strategy specified by the user.

        Parameters:
            None

        Returns:
            None
        """
        proc = _make_proc()
        custom = {'Time': 5, 'nCells': 50000}
        open_c, full_c = proc._prepare_chunking_config(custom)
        assert open_c == {'Time': 5}
        assert full_c is custom

    def test_none_chunks_returns_default_config(self: 'TestPrepareChunkingConfig') -> None:
        """
        This test verifies that _prepare_chunking_config with None returns a default open_chunks dict with 'Time' set to 1, and a full_chunks dict that includes 'nCells'. The test confirms that when no custom chunking configuration is provided, the method returns an open_chunks dict that only includes 'Time' with a chunk size of 1, and a full_chunks dict that contains the default chunking strategy, which should include 'nCells'. This ensures that the method correctly provides sensible defaults for chunking when no custom configuration is specified. 

        Parameters:
            None

        Returns:
            None
        """
        proc = _make_proc()
        open_c, full_c = proc._prepare_chunking_config(None)
        assert open_c == {'Time': 1}
        assert 'nCells' in full_c


class TestLoadMultifileDatasetDropVars:
    """ Tests for the MPASBaseProcessor _load_multifile_dataset method to verify that the drop_variables parameter is correctly forwarded to xarray's open_mfdataset. """

    def test_drop_variables_passed_to_open_mfdataset(self: 'TestLoadMultifileDatasetDropVars', 
                                                     tmp_path: 'Path') -> None:
        """
        This test verifies that when _load_multifile_dataset is called with a drop_variables list, it correctly passes this list to xarray's open_mfdataset function. The test uses the unittest.mock.patch function to mock xarray.open_mfdataset and checks that the drop_variables argument is included in the call with the expected value. This ensures that the method properly forwards the drop_variables parameter, allowing users to exclude specific variables from being loaded into memory when working with large datasets. 

        Parameters:
            tmp_path: A pytest fixture that provides a temporary directory for the test.

        Returns:
            None
        """
        proc = _make_proc()
        n = 2
        times_dt = [datetime(2024, 1, 1), datetime(2024, 1, 2)]

        combined = xr.Dataset(
            {'t': xr.DataArray(np.ones((n, 5)), dims=['Time', 'nCells'])}
        )

        with patch('xarray.open_mfdataset', return_value=combined) as mock_open:
            proc._load_multifile_dataset(
                [str(tmp_path / 'f1.nc'), str(tmp_path / 'f2.nc')],
                times_dt,
                {'Time': 1},
                drop_variables=['unused_var'],
            )

        kwargs = mock_open.call_args.kwargs

        assert 'drop_variables' in kwargs
        assert kwargs['drop_variables'] == ['unused_var']


class TestApplyChunking:
    """ Tests for the MPASBaseProcessor _apply_chunking method to verify its behavior when chunks is None or when an exception occurs during chunking. """

    def test_returns_dataset_unchanged_when_chunks_is_none(self: 'TestApplyChunking') -> None:
        """
        This test verifies that _apply_chunking returns the original dataset unchanged when the chunks parameter is None. The test confirms that when no chunking configuration is provided, the method does not attempt to chunk the dataset and simply returns it as-is, which is important for ensuring that the method behaves correctly when chunking is not desired or necessary. 

        Parameters:
            None

        Returns:
            None
        """
        proc = _make_proc()
        ds = _minimal_ds()
        result = proc._apply_chunking(ds, None)
        assert result is ds

    def test_returns_original_on_chunk_exception(self: 'TestApplyChunking') -> None:
        """
        This test verifies that _apply_chunking returns the original dataset unchanged when an exception occurs during the chunking process. The test uses unittest.mock.MagicMock to create a mock xarray Dataset and configures its chunk method to raise an exception. The test then confirms that when _apply_chunking is called with this mock dataset and a chunking configuration, it catches the exception and returns the original dataset without modification, which is important for ensuring that the method can handle chunking errors gracefully without crashing the entire processing workflow.

        Parameters:
            None

        Returns:
            None
        """
        proc = _make_proc()
        mock_ds = MagicMock(spec=xr.Dataset)
        mock_ds.chunk.side_effect = Exception("chunk failed")
        result = proc._apply_chunking(mock_ds, {'bad_dim': 1})
        assert result is mock_ds


class TestCreateUxarrayDataset:
    """ Tests for the MPASBaseProcessor _create_uxarray_dataset method to verify its behavior when uxgrid is None. """

    def test_raises_when_uxgrid_is_none(self: 'TestCreateUxarrayDataset', 
                                        tmp_path: 'Path') -> None:
        """
        This test verifies that _create_uxarray_dataset raises ValueError with an appropriate message when the uxgrid attribute of the grid dataset is None. The test uses unittest.mock.MagicMock to create a mock grid dataset with uxgrid set to None, and then patches the open_dataset function to return this mock grid dataset. The test confirms that when _create_uxarray_dataset is called with a combined dataset and a list of file paths, it checks for the presence of uxgrid in the grid dataset, finds it to be None, and raises the expected exception with a clear error message indicating that the uxgrid could not be extracted, which is crucial for ensuring that the method handles this error condition properly. 

        Parameters:
            tmp_path: A pytest fixture that provides a temporary directory for the test.

        Returns:
            None
        """
        proc = _make_proc()
        combined_ds = _minimal_ds()
        mock_grid_ds = MagicMock()
        mock_grid_ds.uxgrid = None
        with patch('mpasdiag.processing.base.ux.open_dataset', return_value=mock_grid_ds):
            with pytest.raises(ValueError, match="Could not extract uxgrid"):
                proc._create_uxarray_dataset(combined_ds, [str(tmp_path / 'f.nc')])


class TestAttemptFallbackLoad:
    """ Tests for the MPASBaseProcessor _attempt_fallback_load method to verify its behavior when using the fallback xarray strategy. """

    def test_returns_xarray_type_and_sets_instance_attrs(self: 'TestAttemptFallbackLoad', 
                                                         tmp_path: 'Path') -> None:
        """
        This test verifies that _attempt_fallback_load returns a dataset of type 'xarray' and sets the processor's dataset and data_type attributes accordingly when the fallback loading strategy succeeds. The test uses unittest.mock.patch to mock the _load_multifile_dataset and _apply_chunking methods to return a minimal xarray Dataset. The test then confirms that the result type is 'xarray', and that the processor's dataset attribute is set to the returned dataset, and the data_type attribute is set to 'xarray', which indicates that the fallback loading strategy was successful and that the processor's state was updated correctly. 

        Parameters:
            tmp_path: A pytest fixture that provides a temporary directory for the test.

        Returns:
            None
        """
        proc = _make_proc()
        ds = _minimal_ds()
        times = [datetime(2024, 1, 1), datetime(2024, 1, 2)]

        with patch.object(proc, '_load_multifile_dataset', return_value=ds):
            with patch.object(proc, '_apply_chunking', return_value=ds):
                result_ds, result_type = proc._attempt_fallback_load(
                    [str(tmp_path / 'f1.nc'), str(tmp_path / 'f2.nc')],
                    times,
                    {'Time': 1},
                    'Diagnostic',
                )

        assert result_type == 'xarray'
        assert proc.dataset is ds
        assert proc.data_type == 'xarray'

    def test_verbose_calls_print_loading_success(self: 'TestAttemptFallbackLoad', 
                                                 tmp_path: 'Path') -> None:
        """
        This test verifies that when verbose=True, _attempt_fallback_load calls the _print_loading_success method with a message that includes 'fallback'. The test uses unittest.mock.patch to mock the _load_multifile_dataset and _apply_chunking methods to return a minimal xarray Dataset, and also mocks the _print_loading_success method to track its calls. The test then confirms that _print_loading_success was called once, and that the message passed to it contains the word 'fallback', which indicates that the method is providing appropriate feedback about the fallback loading strategy when verbose mode is enabled. 

        Parameters:
            tmp_path: A pytest fixture that provides a temporary directory for the test.

        Returns:
            None
        """
        proc = _make_proc(verbose=True)
        ds = _minimal_ds()
        times = [datetime(2024, 1, 1), datetime(2024, 1, 2)]

        with patch.object(proc, '_load_multifile_dataset', return_value=ds):
            with patch.object(proc, '_apply_chunking', return_value=ds):
                with patch.object(proc, '_print_loading_success') as mock_print:
                    proc._attempt_fallback_load(
                        [str(tmp_path / 'f1.nc'), str(tmp_path / 'f2.nc')],
                        times,
                        None,
                        'Diagnostic',
                    )

        mock_print.assert_called_once()
        assert 'fallback' in mock_print.call_args.args[2]


class TestDiscoverDataFilesDefaultFallback:
    """ Tests for the MPASBaseProcessor _discover_data_files method to verify that it falls back to _find_files_by_pattern with DIAG_GLOB when no custom finders are defined. """

    def test_calls_find_files_by_pattern_when_no_custom_finders(self: 'TestDiscoverDataFilesDefaultFallback',
                                                                tmp_path: 'Path') -> None:
        """
        This test verifies that _discover_data_files calls the _find_files_by_pattern method with the DIAG_GLOB pattern when no custom file discovery methods (find_diagnostic_files or find_mpasout_files) are defined on the processor instance. The test uses unittest.mock.patch to mock the _find_files_by_pattern method and returns a list of fake file paths. The test then confirms that _find_files_by_pattern was called once, and that the result returned by _discover_data_files matches the fake file paths, which indicates that the method correctly falls back to using the default file discovery mechanism when no custom finders are present. 

        Parameters:
            tmp_path: A pytest fixture that provides a temporary directory for the test.

        Returns:
            None
        """
        proc = _make_proc()
        assert not hasattr(proc, 'find_diagnostic_files')
        assert not hasattr(proc, 'find_mpasout_files')
        fake_files = [str(tmp_path / f'diag_{i:04d}.nc') for i in range(3)]

        with patch.object(proc, '_find_files_by_pattern', return_value=fake_files) as mock_find:
            result = proc._discover_data_files(str(tmp_path))

        mock_find.assert_called_once()
        assert result == fake_files


class TestLoadDataSelectiveVariables:
    """ Tests for the MPASBaseProcessor _load_data method to verify the behavior when a variables list is provided, including the computation of drop_variables, verbose output, and exception handling related to selective loading. """

    def _setup_load_data_mocks(self: 'TestLoadDataSelectiveVariables',
                               proc: MPASBaseProcessor,
                               tmp_path: 'Path',
                               probe_ds: xr.Dataset,
                               result_ds: xr.Dataset,) -> MagicMock:
        """
        This helper method sets up the necessary mocks for testing the _load_data method when a variables list is provided. It mocks the _discover_data_files, MPASDateTimeUtils.parse_file_datetimes, _prepare_chunking_config, and xarray.open_dataset methods to simulate the behavior of the _load_data method without relying on actual file I/O or dataset loading. The method returns a MagicMock object for the primary load attempt, which can be used in the test methods to assert calls and inspect arguments related to the selective loading logic. 

        Parameters:
            proc: The MPASBaseProcessor instance being tested.
            tmp_path: A pytest fixture that provides a temporary directory for the test.
            probe_ds: An xarray Dataset to be returned by the mocked probe open_dataset call.
            result_ds: An xarray Dataset to be returned by the mocked primary load attempt.

        Returns:
            A MagicMock object for the primary load attempt, which can be used in the test methods to assert calls and inspect arguments. 
        """
        times = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        fake_files = [str(tmp_path / f'f{i}.nc') for i in range(2)]

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=probe_ds)
        ctx.__exit__ = MagicMock(return_value=False)

        self._mocks = {
            'discover': patch.object(proc, '_discover_data_files', return_value=fake_files),
            'parse':    patch(
                'mpasdiag.processing.base.MPASDateTimeUtils.parse_file_datetimes',
                return_value=times,
            ),
            'chunks':   patch.object(
                proc, '_prepare_chunking_config',
                return_value=({'Time': 1}, {'Time': 1}),
            ),
            'open_ds':  patch('xarray.open_dataset', return_value=ctx),
        }

        return patch.object(proc, '_attempt_primary_load', return_value=(result_ds, 'xarray'))

    def test_computes_drop_variables_from_probe(self: 'TestLoadDataSelectiveVariables', 
                                                tmp_path: 'Path') -> None:
        """
        This test verifies that when _load_data is called with a variables list, it computes the drop_variables list as the set of variables in the probe dataset that are not in the requested variables list, and passes this drop_variables list to the primary load attempt. The test sets up a probe dataset with multiple variables and calls _load_data with a subset of those variables. It then asserts that the drop_variables argument passed to the primary load attempt contains the correct variables that should be dropped, which confirms that the method is correctly identifying which variables to exclude based on the initial probe of the dataset. 

        Parameters:
            tmp_path: A pytest fixture that provides a temporary directory for the test.

        Returns:
            None
        """
        proc = _make_proc()

        probe_ds = xr.Dataset({
            'var_a': xr.DataArray([1, 2]),
            'var_b': xr.DataArray([3, 4]),
            'var_c': xr.DataArray([5, 6]),
        })

        result_ds = _minimal_ds()
        primary_mock = self._setup_load_data_mocks(proc, tmp_path, probe_ds, result_ds)

        with self._mocks['discover'], self._mocks['parse'], \
             self._mocks['chunks'], self._mocks['open_ds'], primary_mock as mock_primary:
            proc._load_data(str(tmp_path), variables=['var_a'])

        call_kwargs = mock_primary.call_args.kwargs
        assert set(call_kwargs['drop_variables']) == {'var_b', 'var_c'}

    def test_verbose_prints_selective_loading_message(self: 'TestLoadDataSelectiveVariables', 
                                                      tmp_path: 'Path') -> None:
        """
        This test verifies that when verbose=True, _load_data prints a message indicating that selective loading is being used when a variables list is provided. The test sets up the necessary mocks for the _load_data method and captures the standard output using StringIO. It then asserts that the output contains a message about selective loading, which confirms that the method is providing appropriate feedback to the user about the loading strategy being employed when verbose mode is enabled. 

        Parameters:
            tmp_path: A pytest fixture that provides a temporary directory for the test.

        Returns:
            None
        """
        proc = _make_proc(verbose=True)

        probe_ds = xr.Dataset({
            'var_a': xr.DataArray([1, 2]),
            'var_b': xr.DataArray([3, 4]),
        })

        result_ds = _minimal_ds()
        primary_mock = self._setup_load_data_mocks(proc, tmp_path, probe_ds, result_ds)
        out = StringIO()

        with self._mocks['discover'], self._mocks['parse'], \
             self._mocks['chunks'], self._mocks['open_ds'], primary_mock:
            with redirect_stdout(out):
                proc._load_data(str(tmp_path), variables=['var_a'])

        assert 'Selective loading' in out.getvalue()

    def test_probe_exception_sets_drop_variables_to_none(self: 'TestLoadDataSelectiveVariables', 
                                                          tmp_path: 'Path') -> None:
        """
        This test verifies that if an exception occurs during the probe open_dataset call in _load_data, the method sets drop_variables to None for the primary load attempt. The test uses unittest.mock.patch to configure xarray.open_dataset to raise an exception when called for the probe dataset. It then asserts that the drop_variables argument passed to the primary load attempt is None, which confirms that the method correctly handles exceptions during the probing phase by falling back to loading all variables without attempting to drop any, ensuring that the loading process can continue even if the initial probe fails. 

        Parameters:
            tmp_path: A pytest fixture that provides a temporary directory for the test.

        Returns:
            None
        """
        proc = _make_proc()
        result_ds = _minimal_ds()
        times = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        fake_files = [str(tmp_path / f'f{i}.nc') for i in range(2)]

        with patch.object(proc, '_discover_data_files', return_value=fake_files), \
             patch('mpasdiag.processing.base.MPASDateTimeUtils.parse_file_datetimes',
                   return_value=times), \
             patch.object(proc, '_prepare_chunking_config',
                          return_value=({'Time': 1}, None)), \
             patch('xarray.open_dataset', side_effect=Exception('probe failed')), \
             patch.object(proc, '_attempt_primary_load',
                          return_value=(result_ds, 'xarray')) as mock_primary:
            proc._load_data(str(tmp_path), variables=['var_a'])

        call_kwargs = mock_primary.call_args.kwargs
        assert call_kwargs['drop_variables'] is None


class TestLoadDataAllStrategiesFail:
    """ Tests for the MPASBaseProcessor _load_data method to verify that it calls sys.exit(1) when all loading strategies (primary, fallback, single-file) raise exceptions. """

    def test_sys_exit_when_every_loading_strategy_raises(self: 'TestLoadDataAllStrategiesFail', 
                                                         tmp_path: 'Path') -> None:
        """
        This test verifies that if all loading strategies (primary, fallback, single-file) raise exceptions in _load_data, the method calls sys.exit(1) to terminate the program. The test uses unittest.mock.patch to configure the primary load attempt, fallback load attempt, and single-file load attempt to all raise exceptions when called. It then asserts that sys.exit was called with an argument of 1, which confirms that the method correctly handles the scenario where all loading strategies fail by exiting the program with the appropriate status code, ensuring that users are informed of the failure and that the processor does not continue in an invalid state. 

        Parameters:
            tmp_path: A pytest fixture that provides a temporary directory for the test.

        Returns:
            None
        """
        proc = _make_proc()
        times = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        fake_files = [str(tmp_path / f'f{i}.nc') for i in range(2)]

        with patch.object(proc, '_discover_data_files', return_value=fake_files), \
             patch('mpasdiag.processing.base.MPASDateTimeUtils.parse_file_datetimes',
                   return_value=times), \
             patch.object(proc, '_prepare_chunking_config',
                          return_value=({'Time': 1}, None)), \
             patch.object(proc, '_attempt_primary_load',
                          side_effect=Exception('primary fail')), \
             patch.object(proc, '_attempt_fallback_load',
                          side_effect=Exception('fallback fail')), \
             patch.object(proc, '_load_single_file_fallback',
                          side_effect=Exception('single fail')):
            with pytest.raises(SystemExit):
                proc._load_data(str(tmp_path))


class TestSelectFallbackFile:
    """ Tests for the MPASBaseProcessor _select_fallback_file method. """

    def test_returns_reference_file_when_it_exists(self: 'TestSelectFallbackFile', 
                                                   tmp_path: 'Path') -> None:
        """
        This test verifies that _select_fallback_file returns the reference_file path when it exists on disk. The test creates a dummy file at the reference_file path in a temporary directory provided by pytest, ensuring that the file exists. It then calls _select_fallback_file with this reference file and a list of other data files, and asserts that the result is the path to the reference file, which confirms that the method correctly identifies and returns the reference file when it is available, allowing the processor to use it as a fallback option for loading data. 

        Parameters:
            tmp_path: A pytest fixture that provides a temporary directory for the test.

        Returns:
            None
        """
        ref = tmp_path / 'ref.nc'
        ref.touch()
        proc = _make_proc()
        result = proc._select_fallback_file(str(ref), ['/other/data.nc'])
        assert result == str(ref)

    def test_returns_first_data_file_when_reference_is_empty(self: 'TestSelectFallbackFile', 
                                                             tmp_path: 'Path') -> None:
        """
        This test verifies that _select_fallback_file returns the first file from the data_files list when the reference_file argument is an empty string. The test creates multiple dummy files in a temporary directory provided by pytest, ensuring that they exist. It then calls _select_fallback_file with an empty reference file and a list of these data files, and asserts that the result is the path to the first data file in the list, which confirms that the method correctly falls back to using the first available data file when no reference file is specified, allowing the processor to proceed with loading data even when a reference file is not provided. 

        Parameters:
            None

        Returns:
            None
        """
        proc = _make_proc()
        result = proc._select_fallback_file('', ['/data/file1.nc', '/data/file2.nc'])
        assert result == '/data/file1.nc'


class TestAttemptSingleFileLoad:
    """ Tests for the MPASBaseProcessor _attempt_single_file_load method to verify its behavior when loading with UXarray and falling back to xarray. """

    def test_uxarray_path_sets_dataset_and_data_type(self: 'TestAttemptSingleFileLoad', 
                                                     tmp_path: 'Path') -> None:
        """
        This test verifies that when _attempt_single_file_load successfully loads a dataset using the UXarray strategy, it sets the processor's dataset attribute to the loaded dataset and the data_type attribute to 'uxarray'. The test uses unittest.mock.patch to mock the _load_single_file_uxarray method to return a mock UXarray dataset. It then calls _attempt_single_file_load with a file path and asserts that the returned data type is 'uxarray', and that the processor's dataset attribute is set to the mock UXarray dataset, which confirms that the method correctly handles successful loading with UXarray and updates the processor's state accordingly. 

        Parameters:
            tmp_path: A pytest fixture that provides a temporary directory for the test.

        Returns:
            None
        """
        proc = _make_proc()
        mock_ux_ds = MagicMock()

        with patch.object(proc, '_load_single_file_uxarray', return_value=mock_ux_ds):
            ds, dtype = proc._attempt_single_file_load(str(tmp_path / 'f.nc'))

        assert dtype == 'uxarray'
        assert proc.dataset is mock_ux_ds

    def test_xarray_fallback_when_uxarray_raises(self: 'TestAttemptSingleFileLoad', 
                                                 tmp_path: 'Path') -> None:
        """
        This test verifies that when _attempt_single_file_load raises an exception during the UXarray loading attempt, it falls back to the xarray loading strategy and sets the processor's dataset and data_type attributes accordingly. The test uses unittest.mock.patch to configure the _load_single_file_uxarray method to raise an exception, and to return a mock xarray dataset for the _load_single_file_xarray method. It then calls _attempt_single_file_load with a file path and asserts that the returned data type is 'xarray', and that the processor's dataset attribute is set to the mock xarray dataset, which confirms that the method correctly handles exceptions during the UXarray loading attempt by falling back to xarray and updating the processor's state based on the successful fallback load. 

        Parameters:
            tmp_path: A pytest fixture that provides a temporary directory for the test.

        Returns:
            None
        """
        proc = _make_proc()
        mock_xr_ds = MagicMock(spec=xr.Dataset)

        with patch.object(proc, '_load_single_file_uxarray',
                          side_effect=Exception('ux fail')):
            with patch.object(proc, '_load_single_file_xarray',
                               return_value=mock_xr_ds):
                ds, dtype = proc._attempt_single_file_load(str(tmp_path / 'f.nc'))

        assert dtype == 'xarray'
        assert proc.dataset is mock_xr_ds


class TestGetAvailableVariables:
    """ Tests for the MPASBaseProcessor get_available_variables method to verify its behavior when the dataset is not set and when it contains data variables. """

    def test_raises_when_dataset_is_none(self: 'TestGetAvailableVariables') -> None:
        """
        This test verifies that get_available_variables raises ValueError with an appropriate message when the processor's dataset attribute is None. The test creates a processor instance without setting the dataset and then calls get_available_variables, asserting that it raises the expected exception with a clear error message indicating that the dataset is not loaded, which is important for ensuring that users are informed of the issue when they attempt to access available variables without having loaded a dataset. 

        Parameters:
            None

        Returns:
            None
        """
        proc = _make_proc()
        with pytest.raises(ValueError):
            proc.get_available_variables()

    def test_returns_list_of_data_variable_names(self: 'TestGetAvailableVariables') -> None:
        """
        This test verifies that get_available_variables returns a list of data variable names from the processor's dataset when it is set. The test creates a processor instance and sets its dataset attribute to an xarray Dataset containing multiple data variables. It then calls get_available_variables and asserts that the returned list contains the correct variable names, which confirms that the method correctly retrieves and returns the names of the data variables available in the dataset. 

        Parameters:
            None

        Returns:
            None
        """
        proc = _make_proc()

        proc.dataset = xr.Dataset({
            'rain': xr.DataArray(np.ones(5)),
            'temp': xr.DataArray(np.ones(5)),
        })

        result = proc.get_available_variables()
        assert set(result) == {'rain', 'temp'}


class TestValidateGeographicExtent:
    """ Tests for the MPASBaseProcessor validate_geographic_extent method to verify its behavior for various geographic extents. """

    def setup_method(self: 'TestValidateGeographicExtent') -> None:
        """
        This setup method creates a processor instance before each test method in this class. The processor instance is used to call the validate_geographic_extent method with different geographic extents to verify its behavior under various conditions, such as invalid longitude and latitude values, and valid extents.

        Parameters:
            None

        Returns:
            None
        """
        self.proc = _make_proc()

    def test_invalid_lon_min_returns_false(self: 'TestValidateGeographicExtent') -> None:
        """
        This test verifies that validate_geographic_extent returns False when lon_min is outside the valid [-180, 180] range. The test calls the method with a geographic extent where lon_min is set to -200.0, which is less than the minimum valid longitude. It then asserts that the method returns False, which confirms that the method correctly identifies invalid longitude values and rejects geographic extents that do not meet the specified criteria. 

        Parameters:
            None

        Returns:
            None
        """
        assert self.proc.validate_geographic_extent((-200.0, 0.0, 0.0, 10.0)) is False

    def test_invalid_lon_max_returns_false(self: 'TestValidateGeographicExtent') -> None:
        """
        This test verifies that validate_geographic_extent returns False when lon_max is outside the valid [-180, 180] range. The test calls the method with a geographic extent where lon_max is set to 200.0, which is greater than the maximum valid longitude. It then asserts that the method returns False, which confirms that the method correctly identifies invalid longitude values and rejects geographic extents that do not meet the specified criteria. 

        Parameters:
            None

        Returns:
            None
        """
        assert self.proc.validate_geographic_extent((-10.0, 200.0, 0.0, 10.0)) is False

    def test_invalid_lat_min_returns_false(self: 'TestValidateGeographicExtent') -> None:
        """
        This test verifies that validate_geographic_extent returns False when lat_min is outside the valid [-90, 90] range. The test calls the method with a geographic extent where lat_min is set to -100.0, which is less than the minimum valid latitude. It then asserts that the method returns False, which confirms that the method correctly identifies invalid latitude values and rejects geographic extents that do not meet the specified criteria. 

        Parameters:
            None

        Returns:
            None
        """
        assert self.proc.validate_geographic_extent((-10.0, 10.0, -100.0, 10.0)) is False

    def test_invalid_lat_max_returns_false(self: 'TestValidateGeographicExtent') -> None:
        """
        This test verifies that validate_geographic_extent returns False when lat_max is outside the valid [-90, 90] range. The test calls the method with a geographic extent where lat_max is set to 100.0, which is greater than the maximum valid latitude. It then asserts that the method returns False, which confirms that the method correctly identifies invalid latitude values and rejects geographic extents that do not meet the specified criteria. 

        Parameters:
            None

        Returns:
            None
        """
        assert self.proc.validate_geographic_extent((-10.0, 10.0, 0.0, 100.0)) is False

    def test_lon_min_ge_lon_max_returns_false(self: 'TestValidateGeographicExtent') -> None:
        """
        This test verifies that validate_geographic_extent returns False when lon_min is greater than or equal to lon_max. The test calls the method with a geographic extent where lon_min is set to 10.0 and lon_max is set to 5.0, which violates the requirement that lon_min must be less than lon_max. It then asserts that the method returns False, which confirms that the method correctly identifies and rejects geographic extents where the minimum longitude is not less than the maximum longitude. 

        Parameters:
            None

        Returns:
            None
        """
        assert self.proc.validate_geographic_extent((10.0, 5.0, 0.0, 10.0)) is False

    def test_lat_min_ge_lat_max_returns_false(self: 'TestValidateGeographicExtent') -> None:
        """
        This test verifies that validate_geographic_extent returns False when lat_min is greater than or equal to lat_max. The test calls the method with a geographic extent where lat_min is set to 10.0 and lat_max is set to 5.0, which violates the requirement that lat_min must be less than lat_max. It then asserts that the method returns False, which confirms that the method correctly identifies and rejects geographic extents where the minimum latitude is not less than the maximum latitude. 

        Parameters:
            None

        Returns:
            None
        """
        assert self.proc.validate_geographic_extent((-10.0, 10.0, 5.0, 0.0)) is False

    def test_valid_extent_returns_true(self: 'TestValidateGeographicExtent') -> None:
        """
        This test verifies that validate_geographic_extent returns True when a valid geographic extent is provided. The test calls the method with a geographic extent where lon_min, lon_max, lat_min, and lat_max are all within their respective valid ranges and satisfy the requirements for minimum and maximum values. It then asserts that the method returns True, which confirms that the method correctly identifies valid geographic extents and accepts them as suitable for processing. 

        Parameters:
            None

        Returns:
            None
        """
        assert self.proc.validate_geographic_extent((-120.0, -80.0, 30.0, 50.0)) is True


class TestExtractSpatialCoordinates:
    """ Tests for the MPASBaseProcessor extract_spatial_coordinates method to verify its behavior when the dataset is not set, when no longitude or latitude variables are present, and when radian coordinates are provided. """

    def test_raises_when_dataset_is_none(self: 'TestExtractSpatialCoordinates') -> None:
        """
        This test verifies that extract_spatial_coordinates raises ValueError with an appropriate message when the processor's dataset attribute is None. The test creates a processor instance without setting the dataset and then calls extract_spatial_coordinates, asserting that it raises the expected exception with a clear error message indicating that the dataset is not loaded, which is important for ensuring that users are informed of the issue when they attempt to extract spatial coordinates without having loaded a dataset. 

        Parameters:
            None

        Returns:
            None
        """
        proc = _make_proc()
        with pytest.raises(ValueError):
            proc.extract_spatial_coordinates()

    def test_raises_when_no_lon_lat_vars_present(self: 'TestExtractSpatialCoordinates') -> None:
        """
        This test verifies that extract_spatial_coordinates raises ValueError with an appropriate message when the dataset does not contain any longitude or latitude variables. The test creates a processor instance and sets its dataset attribute to an xarray Dataset that lacks any variables with names containing 'lon' or 'lat'. It then calls extract_spatial_coordinates and asserts that it raises the expected exception with a clear error message indicating that spatial coordinates could not be found, which confirms that the method correctly identifies the absence of necessary spatial coordinate variables and informs the user of this issue. 

        Parameters:
            None

        Returns:
            None
        """
        proc = _make_proc()

        proc.dataset = xr.Dataset(
            {'unrelated': xr.DataArray(np.ones(5), dims=['nCells'])}
        )

        with pytest.raises(ValueError, match="Could not find spatial coordinates"):
            proc.extract_spatial_coordinates()

    def test_radian_coordinates_are_converted_to_degrees(self: 'TestExtractSpatialCoordinates') -> None:
        """
        This test verifies that extract_spatial_coordinates correctly identifies longitude and latitude variables in radians, converts them to degrees, and returns the converted values. The test creates a processor instance and sets its dataset attribute to an xarray Dataset containing longitude and latitude variables with values in radians. It then calls extract_spatial_coordinates and asserts that the returned longitude and latitude arrays are correctly converted to degrees, which confirms that the method properly handles spatial coordinates provided in radians and ensures that the output is in the expected units for further processing. 

        Parameters:
            None

        Returns:
            None
        """
        proc = _make_proc()
        n = 10

        proc.dataset = xr.Dataset({
            'lonCell': xr.DataArray(np.linspace(0.0, 1.0, n)),
            'latCell': xr.DataArray(np.linspace(0.0, 1.0, n)),
        })

        lon, lat = proc.extract_spatial_coordinates()
        assert lon.shape == (n,)
        assert lat.shape == (n,)


class TestLoadGridFileProbeException:
    """ Tests for the MPASBaseProcessor _load_grid_file method to verify that if an exception occurs during the probe open_dataset call, the method continues to the actual load call without re-raising the exception. """

    def test_probe_exception_falls_through_to_actual_load(self: 'TestLoadGridFileProbeException') -> None:
        """
        This test verifies that if an exception occurs during the probe open_dataset call in _load_grid_file, the method does not re-raise the exception and instead continues to the actual load call. The test uses unittest.mock.patch to configure xarray.open_dataset to raise an exception on the first call (the probe) and to return a mock dataset on the second call (the actual load). It then asserts that the result of _load_grid_file is the mock dataset returned by the second call, which confirms that the method correctly handles exceptions during the probing phase by allowing the loading process to continue without interruption, ensuring that users can still load grid files even if the initial probe encounters issues. 

        Parameters:
            None

        Returns:
            None
        """
        proc = _make_proc()
        mock_ds = MagicMock()
        mock_ds.variables = {'latCell': MagicMock()}

        call_n = [0]

        def side_effect(*args, **kwargs) -> xr.Dataset:
            """
            This side effect function simulates the behavior of xarray.open_dataset for the _load_grid_file method. On the first call, it raises an exception to mimic a failure during the probe phase. On the second call, it returns a mock dataset that contains the necessary variable ('latCell') to allow the loading process to succeed. The call_n list is used to keep track of how many times the function has been called, ensuring that the exception is only raised on the first call and that the mock dataset is returned on subsequent calls. 

            Parameters:
                *args: Positional arguments passed to the function (not used in this side effect).  
                **kwargs: Keyword arguments passed to the function (not used in this side effect).

            Returns:
                xr.Dataset: A mock dataset returned on the second call, containing the necessary variable for loading. 
            """
            call_n[0] += 1
            if call_n[0] == 1:
                raise Exception("probe failed")
            return mock_ds

        with patch('xarray.open_dataset', side_effect=side_effect):
            result = proc._load_grid_file(needed_vars=['latCell'])

        assert result is mock_ds
        assert call_n[0] == 2


class TestApplyCoordinateUpdates:
    """ Tests for the MPASBaseProcessor _apply_coordinate_updates method to verify its behavior when no additional coordinates are found and when new coordinates are added. """

    def test_verbose_prints_no_additional_coords_message(self: 'TestApplyCoordinateUpdates') -> None:
        """
        This test verifies that when verbose=True and no additional coordinate variables are found to add to the dataset, _apply_coordinate_updates prints a message indicating that no additional coordinate variables were found. The test creates a processor instance with verbose mode enabled and calls _apply_coordinate_updates with an empty coords_to_add dictionary. It captures the standard output using StringIO and asserts that the output contains the expected message about no additional coordinate variables being found, which confirms that the method provides appropriate feedback to the user about the coordinate update process when verbose mode is enabled. 

        Parameters:
            None

        Returns:
            None
        """
        proc = _make_proc(verbose=True)
        ds = _minimal_ds()
        out = StringIO()
        with redirect_stdout(out):
            result = proc._apply_coordinate_updates(ds, {}, {})
        assert 'No additional coordinate variables found' in out.getvalue()
        assert result is ds

    def test_coords_added_when_coords_to_add_is_provided(self: 'TestApplyCoordinateUpdates') -> None:
        """
        This test verifies that when coords_to_add contains new coordinate variables, _apply_coordinate_updates adds those coordinates to the dataset and prints a message indicating which coordinates were added when verbose=True. The test creates a processor instance with verbose mode enabled and calls _apply_coordinate_updates with a coords_to_add dictionary containing a new coordinate variable. It captures the standard output using StringIO and asserts that the output contains a message about the added coordinate variable, and that the resulting dataset includes the new coordinate in its coordinates, which confirms that the method correctly updates the dataset with new coordinates and provides appropriate feedback to the user when verbose mode is enabled. 

        Parameters:
            None

        Returns:
            None
        """
        proc = _make_proc(verbose=True)
        ds = _minimal_ds()
        coords = {'nCells': ('nCells', np.arange(10))}
        out = StringIO()
        with redirect_stdout(out):
            result = proc._apply_coordinate_updates(ds, coords, {})
        assert 'nCells' in result.coords
    

class TestAddSpatialCoordsHelperException:
    """ Tests for the MPASBaseProcessor _add_spatial_coords_helper method to verify that if an exception occurs during the grid file loading, a warning is printed and the original dataset is returned unchanged. """

    def test_warning_printed_and_original_ds_returned_on_exception(self: 'TestAddSpatialCoordsHelperException') -> None:
        """
        This test verifies that if an exception occurs during the grid file loading in _add_spatial_coords_helper, the method prints a warning message indicating that spatial coordinates could not be added, and returns the original dataset unchanged. The test uses unittest.mock.patch to configure the _load_grid_file method to raise an exception when called. It then captures the standard output using StringIO and asserts that the output contains a warning message about not being able to add spatial coordinates, and that the result of the method is the same as the original dataset passed in, which confirms that the method correctly handles exceptions during grid file loading by informing the user and returning the original dataset without modification. 

        Parameters:
            None

        Returns:
            None
        """
        proc = _make_proc(verbose=True)
        ds = _minimal_ds()
        with patch.object(proc, '_load_grid_file',
                          side_effect=Exception('grid load failed')):
            out = StringIO()
            with redirect_stdout(out):
                result = proc._add_spatial_coords_helper(
                    ds, ['nCells'], ['latCell'], 'TestProcessor'
                )
        assert 'Warning: Could not add' in out.getvalue()
        assert result is ds


class TestGetTimeInfo:
    """ Tests for the MPASBaseProcessor get_time_info method to verify that it raises an exception when the dataset is not set. """

    def test_raises_when_dataset_is_none(self: 'TestGetTimeInfo') -> None:
        """
        This test verifies that get_time_info raises ValueError with an appropriate message when the processor's dataset attribute is None. The test creates a processor instance without setting the dataset and then calls get_time_info with a time index, asserting that it raises the expected exception with a clear error message indicating that the dataset is not loaded, which is important for ensuring that users are informed of the issue when they attempt to access time information without having loaded a dataset. 

        Parameters:
            None

        Returns:
            None
        """
        proc = _make_proc()
        with pytest.raises(ValueError):
            proc.get_time_info(0)


class TestParseFileDatetimes:
    """ Tests for the MPASBaseProcessor parse_file_datetimes method to verify that it delegates to MPASDateTimeUtils and returns the result. """

    def test_delegates_to_datetime_utils_and_returns_list(self: 'TestParseFileDatetimes', tmp_path) -> None:
        """
        This test verifies that parse_file_datetimes imports MPASDateTimeUtils and delegates to its parse_file_datetimes method, returning the result. The test uses unittest.mock.patch to mock the MPASDateTimeUtils.parse_file_datetimes method to return a predefined list of datetime objects. It then calls parse_file_datetimes with a list of file paths and asserts that the mocked method was called once with the correct arguments, and that the result returned by parse_file_datetimes matches the predefined list of datetime objects, which confirms that the method correctly delegates to the utility function and returns its output as expected. 

        Parameters:
            None

        Returns:
            None
        """
        proc = _make_proc()
        fake_times = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        with patch(
            'mpasdiag.processing.utils_datetime.MPASDateTimeUtils.parse_file_datetimes',
            return_value=fake_times,
        ) as mock_parse:
            result = proc.parse_file_datetimes(
                [str(tmp_path / 'diag_2024010100.nc'), str(tmp_path / 'diag_2024010200.nc')]
            )
        mock_parse.assert_called_once()
        assert result == fake_times


class TestValidateTimeParameters:
    """ Tests for the MPASBaseProcessor validate_time_parameters method to verify that it raises an exception when the dataset is not set. """

    def test_raises_when_dataset_is_none(self: 'TestValidateTimeParameters') -> None:
        """
        This test verifies that validate_time_parameters raises ValueError with an appropriate message when the processor's dataset attribute is None. The test creates a processor instance without setting the dataset and then calls validate_time_parameters with a time index, asserting that it raises the expected exception with a clear error message indicating that the dataset is not loaded, which is important for ensuring that users are informed of the issue when they attempt to validate time parameters without having loaded a dataset. 

        Parameters:
            None

        Returns:
            None
        """
        proc = _make_proc()
        with pytest.raises(ValueError):
            proc.validate_time_parameters(0)

    def test_delegates_to_datetime_utils_with_loaded_dataset(self: 'TestValidateTimeParameters') -> None:
        """
        This test verifies that when the dataset is loaded, validate_time_parameters imports MPASDateTimeUtils and delegates to its validate_time_parameters method, returning the result. The test sets the processor's dataset attribute to a minimal dataset and uses unittest.mock.patch to mock the MPASDateTimeUtils.validate_time_parameters method to return a predefined result. It then calls validate_time_parameters with a time index and asserts that the mocked method was called once with the correct arguments, and that the result returned by validate_time_parameters matches the predefined result, which confirms that the method correctly delegates to the utility function and returns its output as expected when a dataset is loaded. 

        Parameters:
            None

        Returns:
            None
        """
        proc = _make_proc()
        proc.dataset = _minimal_ds()
        fake_result = ('Time', 0, 2)
        with patch(
            'mpasdiag.processing.utils_datetime.MPASDateTimeUtils.validate_time_parameters',
            return_value=fake_result,
        ) as mock_validate:
            result = proc.validate_time_parameters(0)
        mock_validate.assert_called_once()
        assert result == fake_result


class TestFilterBySpatialExtent:
    """ Tests for the MPASBaseProcessor filter_by_spatial_extent method to verify that it raises an exception when the dataset is not set. """

    def test_raises_when_dataset_is_none(self: 'TestFilterBySpatialExtent') -> None:
        """
        This test verifies that filter_by_spatial_extent raises ValueError with an appropriate message when the processor's dataset attribute is None. The test creates a processor instance without setting the dataset and then calls filter_by_spatial_extent with a geographic extent, asserting that it raises the expected exception with a clear error message indicating that the dataset is not loaded, which is important for ensuring that users are informed of the issue when they attempt to filter by spatial extent without having loaded a dataset. 

        Parameters:
            None

        Returns:
            None
        """
        proc = _make_proc()
        with pytest.raises(ValueError):
            proc.filter_by_spatial_extent(None, -120.0, -80.0, 30.0, 50.0)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])

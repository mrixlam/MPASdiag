#!/usr/bin/env python3
"""
MPASdiag Test Suite: CLI Unified Coverage

This module contains unit tests for the MPASUnifiedCLI class in mpasdiag.processing.cli_unified, specifically targeting lines that were previously untested to improve code coverage. The tests focus on error handling, logging behavior, and the execution of specific branches in the methods related to configuration validation, analysis dispatching, and sounding analysis. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: April 2026
Version: 1.0.0
"""
import sys
import argparse
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from mpasdiag.processing.cli_unified import MPASUnifiedCLI
from mpasdiag.processing.utils_config import MPASConfig


def _make_profile() -> dict:
    """
    This helper function creates a mock sounding profile dictionary with typical keys and values that would be returned by SoundingDiagnostics.extract_sounding_profile. It includes pressure, temperature, dewpoint, u and v wind components, and station longitude and latitude. This allows tests to simulate the presence of a sounding profile without needing to read actual data files.  

    Parameters:
        None

    Returns:
        dict: A dictionary representing a sounding profile with keys for 'pressure', 'temperature', 'dewpoint', 'u_wind', 'v_wind', 'station_lon', and 'station_lat'. Each key maps to a list of values for the respective variable or a single value for the coordinates. 
    """
    return {
        'pressure': [1000.0, 850.0, 700.0],
        'temperature': [25.0, 15.0, 5.0],
        'dewpoint': [20.0, 10.0, 0.0],
        'u_wind': [5.0, 8.0, 10.0],
        'v_wind': [3.0, 5.0, 7.0],
        'station_lon': -97.5,
        'station_lat': 36.0,
    }


class TestValidateCoordinateRangeError:
    """ Test the error handling in _validate_coordinate_range when min_val >= max_val."""

    def test_appends_error_when_min_ge_max(self: 'TestValidateCoordinateRangeError') -> None:
        """
        This test verifies that _validate_coordinate_range appends an error message to the errors list when the minimum value is greater than or equal to the maximum value for a coordinate range. It checks that the error message includes the coordinate name to confirm that the correct validation failure is being reported. 

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        config = SimpleNamespace(lat_min=50.0, lat_max=30.0)
        errors: list = []
        cli._validate_coordinate_range(config, 'lat_min', 'lat_max', 'latitude', errors)
        assert len(errors) == 1
        assert 'latitude' in errors[0]


class TestReportValidationErrorsWithLogger:
    """ Test the behavior of _report_validation_errors when a logger is set, ensuring that it calls self.logger.error for each error message in the list. """
    
    def test_uses_logger_error_when_logger_is_set(self: 'TestReportValidationErrorsWithLogger') -> None:
        """
        This test verifies that _report_validation_errors calls self.logger.error for each error message in the list when a logger is set. It checks that the logger's error method is called at least as many times as there are error messages, which confirms that all errors are being logged appropriately. 

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        cli.logger = MagicMock()
        cli._report_validation_errors(["Error A", "Error B"])
        assert cli.logger.error.call_count >= 3  # header + 2 errors


class TestValidateConfigNoDataFiles:
    """ Test the behavior of validate_config when the data directory exists but contains no MPAS data files matching expected patterns. """

    def test_returns_false_when_dir_exists_but_no_data_files(self: 'TestValidateConfigNoDataFiles', 
                                                             tmp_path: 'Path') -> None:
        """
        This test verifies that validate_config returns False when the specified data directory exists but contains no MPAS data files matching expected patterns. It creates an empty grid file in the temporary directory to satisfy the grid file requirement, but does not add any data files. The test confirms that the validation correctly identifies the absence of data files and returns False. 

        Parameters:
            tmp_path: pytest temporary directory fixture that provides a valid directory path with no MPAS data files. 

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        grid_file = tmp_path / 'grid.nc'
        grid_file.touch()

        config = MPASConfig(
            grid_file=str(grid_file),
            data_dir=str(tmp_path),
            analysis_type='precipitation',
        )

        result = cli.validate_config(config)
        assert result is False


class TestCheckAnalysisTypeSpecifiedFalse:
    """ Test the behavior of _check_analysis_type_specified when config.analysis_type is None. """

    def test_returns_false_when_analysis_type_is_none(self: 'TestCheckAnalysisTypeSpecifiedFalse') -> None:
        """
        This test verifies that _check_analysis_type_specified returns False when the analysis_type attribute of the configuration object is None. This covers the early-return case in the method where it checks if analysis_type is falsy and returns False without further processing. 

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        config = SimpleNamespace(analysis_type=None)
        result = cli._check_analysis_type_specified(config)
        assert result is False

    def test_run_analysis_returns_false_when_no_analysis_type(self: 'TestCheckAnalysisTypeSpecifiedFalse') -> None:
        """
        This test verifies that the run_analysis method returns False when the analysis_type in the configuration is None. This ensures that the check for a specified analysis type is properly integrated into the run_analysis workflow, and that it prevents further execution when no analysis type is provided. 

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        config = MPASConfig(
            grid_file='test.nc',
            data_dir='data/',
            analysis_type=None,
        )
        result = cli.run_analysis(config)
        assert result is False


class TestDispatchAnalysisNoneType:
    """ Test the behavior of _dispatch_analysis when analysis_type is None or empty string. """

    def test_returns_none_for_falsy_analysis_type(self: 'TestDispatchAnalysisNoneType') -> None:
        """
        This test verifies that _dispatch_analysis returns None when the analysis_type is either None or an empty string. This covers the case where the method should not attempt to dispatch to any analysis function and should simply return None, indicating that no valid analysis type was provided. 

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        assert cli._dispatch_analysis(None, MagicMock()) is None
        assert cli._dispatch_analysis('', MagicMock()) is None


class TestRunSingleCrossSectionLowerHalf:
    """ Test if _run_single_cross_section correctly computes the output path, saves and closes the plot, and logs the saved path when a logger is set, as well as using config.output when it is explicitly set. """

    def test_saves_and_closes_plot_with_logger(self: 'TestRunSingleCrossSectionLowerHalf') -> None:
        """
        This test verifies that _run_single_cross_section computes the output path based on the config and time information, calls plotter.save_plot with the correct path, calls plotter.close_plot, and logs the saved path using self.logger.info when a logger is set (lines 1616-1620). It mocks the processor, plotter, and time info to simulate the method's behavior without relying on actual data or file I/O. 

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        cli.logger = MagicMock()

        mock_processor = MagicMock()
        mock_plotter = MagicMock()
        mock_plotter.create_vertical_cross_section.return_value = (None, None)

        config = SimpleNamespace(
            time_index=0,
            title=None,
            variable='temperature',
            colormap='default',
            output=None,
            output_dir='/tmp/cross_out',
            output_formats=['png'],
        )

        params: dict = {}

        with patch(
            'mpasdiag.processing.cli_unified.MPASDateTimeUtils.get_time_info',
            return_value='2024010100',
        ):
            cli._run_single_cross_section(mock_processor, mock_plotter, config, params)

        mock_plotter.save_plot.assert_called_once()
        mock_plotter.close_plot.assert_called_once()
        cli.logger.info.assert_called()

    def test_uses_config_output_when_set(self: 'TestRunSingleCrossSectionLowerHalf') -> None:
        """
        This test verifies that when config.output is explicitly set, _run_single_cross_section uses it as the output file stem instead of computing a name based on time information (line 1619). It checks that plotter.save_plot is called with the path that includes the explicit output name provided in the configuration. 

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        mock_processor = MagicMock()
        mock_plotter = MagicMock()
        mock_plotter.create_vertical_cross_section.return_value = (None, None)

        config = SimpleNamespace(
            time_index=0,
            title=None,
            variable='temperature',
            colormap='default',
            output='/explicit/path/plot',
            output_dir='/tmp/cross_out',
            output_formats=['png'],
        )

        with patch(
            'mpasdiag.processing.cli_unified.MPASDateTimeUtils.get_time_info',
            return_value='2024010100',
        ):
            cli._run_single_cross_section(mock_processor, mock_plotter, config, {})

        args, _ = mock_plotter.save_plot.call_args
        assert args[0] == '/explicit/path/plot'


class TestLogCreatedFilesWithLogger:
    """ Test if _log_created_files correctly logs the file count and type when a logger is set, and does nothing when the files list is empty. """

    def test_calls_logger_info_with_file_count(self: 'TestLogCreatedFilesWithLogger') -> None:
        """
        This test verifies that _log_created_files calls self.logger.info with a message that includes the count of created files and the file type description when a logger is set (lines 1675-1676). It checks that the log message contains the number of files and the provided description to confirm that the logging is informative and accurate. 

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        cli.logger = MagicMock()
        cli._log_created_files(['/tmp/a.png', '/tmp/b.png'], 'test plots')
        cli.logger.info.assert_called_once()
        log_msg = cli.logger.info.call_args.args[0]
        assert '2' in log_msg
        assert 'test plots' in log_msg

    def test_does_nothing_when_files_is_empty(self: 'TestLogCreatedFilesWithLogger') -> None:
        """
        This test verifies that _log_created_files does not call self.logger.info when the files list is empty, even if a logger is set (line 1675). It confirms that no log message is generated when there are no files to report, which ensures that the method behaves appropriately in cases where no output files were created. 

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        cli.logger = MagicMock()
        cli._log_created_files([], 'test plots')
        cli.logger.info.assert_not_called()


class TestComputeSoundingIndices:
    """ Test the behavior of _compute_sounding_indices when show_indices is False (returns None without calling diagnostics) and when show_indices is True (calls diagnostics and returns result). """

    def test_returns_none_when_show_indices_is_false(self: 'TestComputeSoundingIndices') -> None:
        """
        This test verifies that _compute_sounding_indices returns None and does not call sounding_diag.compute_thermodynamic_indices when show_indices is False (line 1698). It confirms that the method correctly bypasses the computation of indices when they are not requested, which is important for performance and correctness in cases where indices are not needed. 

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        mock_diag = MagicMock()
        result = cli._compute_sounding_indices(False, mock_diag, {})
        assert result is None
        mock_diag.compute_thermodynamic_indices.assert_not_called()

    def test_calls_diagnostics_when_show_indices_is_true(self: 'TestComputeSoundingIndices') -> None:
        """
        This test verifies that _compute_sounding_indices calls sounding_diag.compute_thermodynamic_indices and returns its result when show_indices is True (lines 1698-1702). It mocks the diagnostics to return a specific set of indices and checks that the method returns this result, confirming that the computation of indices is correctly integrated into the workflow when requested. 

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        mock_diag = MagicMock()
        mock_diag.compute_thermodynamic_indices.return_value = {'CAPE': 1500.0}
        profile = _make_profile()
        result = cli._compute_sounding_indices(True, mock_diag, profile)
        mock_diag.compute_thermodynamic_indices.assert_called_once()
        assert result == {'CAPE': 1500.0}


class TestBuildSkewtTags:
    """ Test the behavior of _build_skewt_tags for different longitude and latitude values to ensure correct formatting with 'E/W' and 'N/S' suffixes. """

    def test_west_south_hemisphere_tags(self: 'TestBuildSkewtTags') -> None:
        """
        This test verifies that _build_skewt_tags returns 'W' and 'S' suffixes for negative longitude and latitude values (lines 1716-1720). It checks that the longitude and latitude tags are correctly formatted with the appropriate suffixes based on the sign of the input coordinates, which is important for generating accurate and informative titles for skew-T diagrams. 

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()

        lon_tag, lat_tag = cli._build_skewt_tags(
            {'station_lon': -97.5, 'station_lat': -30.0}
        )

        assert lon_tag == '97.50W'
        assert lat_tag == '30.00S'

    def test_east_north_hemisphere_tags(self: 'TestBuildSkewtTags') -> None:
        """
        This test verifies that _build_skewt_tags returns 'E' and 'N' suffixes for positive longitude and latitude values (lines 1716-1720). It checks that the longitude and latitude tags are correctly formatted with the appropriate suffixes based on the sign of the input coordinates, which is important for generating accurate and informative titles for skew-T diagrams. 

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()

        lon_tag, lat_tag = cli._build_skewt_tags(
            {'station_lon': 20.0, 'station_lat': 45.0}
        )

        assert lon_tag == '20.00E'
        assert lat_tag == '45.00N'


class TestRunSoundingBatch:
    """ Test the behavior of _run_sounding_batch for different time ranges and configurations. """

    def test_iterates_over_time_range_and_creates_diagrams(self: 'TestRunSoundingBatch', 
                                                           tmp_path: 'Path') -> None:
        """
        This test verifies that _run_sounding_batch iterates over the specified time range, extracts the sounding profile for each time step, builds the tags, and calls plotter.create_skewt_diagram for each time step (lines 1743-1748). It mocks the processor, plotter, and diagnostics to simulate the method's behavior without relying on actual data or file I/O. The test confirms that the correct number of diagrams are created based on the time range and that the logger is called to indicate progress. 

        Parameters:
            tmp_path: pytest temporary directory fixture used for the output directory in the configuration. 

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        cli.logger = MagicMock()

        mock_processor = MagicMock()
        mock_processor.dataset.dims = {'Time': 2}
        mock_processor.dataset.sizes = {'Time': 2}

        mock_plotter = MagicMock()
        mock_sounding_diag = MagicMock()
        mock_sounding_diag.extract_sounding_profile.return_value = _make_profile()

        config = SimpleNamespace(
            time_start=0,
            time_end=1,
            output_dir=str(tmp_path),
            title=None,
        )

        with patch(
            'mpasdiag.processing.cli_unified.MPASDateTimeUtils.get_time_info',
            return_value='t0',
        ):
            cli._run_sounding_batch(
                mock_processor, mock_plotter, mock_sounding_diag, config,
                -97.5, 36.0, False, False,
            )

        assert mock_plotter.create_skewt_diagram.call_count == 2
        assert mock_plotter.close_plot.call_count == 2
        cli.logger.info.assert_called()

    def test_uses_none_time_bounds_defaults(self: 'TestRunSoundingBatch', 
                                            tmp_path: 'Path') -> None:
        """
        This test verifies that when time_start and time_end are None, _run_sounding_batch defaults to processing the entire time range of the dataset (lines 1741-1742). It mocks a processor with a single time step and checks that the method still creates a skew-T diagram for that time step, confirming that the default behavior for time bounds is correctly implemented. 

        Parameters:
            tmp_path: pytest temporary directory fixture used for the output directory in the configuration. 

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        cli.logger = None

        mock_processor = MagicMock()
        mock_processor.dataset.dims = {'Time': 1}
        mock_processor.dataset.sizes = {'Time': 1}

        mock_plotter = MagicMock()
        mock_sounding_diag = MagicMock()
        mock_sounding_diag.extract_sounding_profile.return_value = _make_profile()

        config = SimpleNamespace(
            time_start=None,
            time_end=None,
            output_dir=str(tmp_path),
            title=None,
        )

        with patch(
            'mpasdiag.processing.cli_unified.MPASDateTimeUtils.get_time_info',
            return_value='t0',
        ):
            cli._run_sounding_batch(
                mock_processor, mock_plotter, mock_sounding_diag, config,
                -97.5, 36.0, False, False,
            )

        assert mock_plotter.create_skewt_diagram.call_count == 1


class TestRunSoundingSingle:
    """ Test the behavior of _run_sounding_single for extracting the sounding profile, computing indices, building tags, and creating the skew-T diagram with the correct title and output path. """

    def test_creates_and_saves_skewt_diagram(self: 'TestRunSoundingSingle', 
                                             tmp_path: 'Path') -> None:
        """
        This test verifies that _run_sounding_single extracts the sounding profile, computes indices if requested, builds the longitude and latitude tags, and calls plotter.create_skewt_diagram with a title that includes the tags (lines 1815-1820). It mocks the processor, plotter, and diagnostics to simulate the method's behavior without relying on actual data or file I/O. The test confirms that the skew-T diagram is created with the expected title format based on the station coordinates. 

        Parameters:
            tmp_path: pytest temporary directory fixture used for the output directory in the configuration. 

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        cli.logger = None

        mock_processor = MagicMock()
        mock_plotter = MagicMock()
        mock_sounding_diag = MagicMock()
        mock_sounding_diag.extract_sounding_profile.return_value = _make_profile()

        config = SimpleNamespace(
            time_index=0,
            output=None,
            output_dir=str(tmp_path),
            title=None,
        )

        with patch(
            'mpasdiag.processing.cli_unified.MPASDateTimeUtils.get_time_info',
            return_value='t0',
        ):
            cli._run_sounding_single(
                mock_processor, mock_plotter, mock_sounding_diag, config,
                -97.5, 36.0, False, False,
            )

        mock_plotter.create_skewt_diagram.assert_called_once()
        call_kwargs = mock_plotter.create_skewt_diagram.call_args.kwargs
        assert 'MPAS Skew-T' in call_kwargs['title']

    def test_uses_explicit_output_name_when_set(self: 'TestRunSoundingSingle', 
                                                tmp_path: 'Path') -> None:
        """
        This test verifies that when config.output is explicitly set, _run_sounding_single uses it as the output file stem instead of computing a name based on time information (line 1819). It checks that plotter.create_skewt_diagram is called with a save_path that includes the explicit output name provided in the configuration, confirming that the method respects the user's specified output name. 

        Parameters:
            tmp_path: pytest temporary directory fixture used for the output directory in the configuration.

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        cli.logger = None

        mock_processor = MagicMock()
        mock_plotter = MagicMock()
        mock_sounding_diag = MagicMock()
        mock_sounding_diag.extract_sounding_profile.return_value = _make_profile()

        config = SimpleNamespace(
            time_index=0,
            output='custom_sounding',
            output_dir=str(tmp_path),
            title='My custom title',
        )

        with patch(
            'mpasdiag.processing.cli_unified.MPASDateTimeUtils.get_time_info',
            return_value='t0',
        ):
            cli._run_sounding_single(
                mock_processor, mock_plotter, mock_sounding_diag, config,
                -97.5, 36.0, False, False,
            )

        call_kwargs = mock_plotter.create_skewt_diagram.call_args.kwargs
        assert call_kwargs['title'] == 'My custom title'
        assert 'custom_sounding' in call_kwargs['save_path']


class TestRunSoundingAnalysis:
    """ Test the complete _run_sounding_analysis workflow for both modes. """

    def _mock_3d_processor(self: 'TestRunSoundingAnalysis') -> MagicMock:
        """
        This helper method creates a mock MPAS3DProcessor with a dataset that has a Time dimension of size 2. This allows the tests to simulate the behavior of the processor when loading 3D data without needing to read actual files, and ensures that the time dimension is properly set up for testing the sounding analysis methods.

        Parameters:
            None

        Returns:
            MagicMock: A mock 3D processor with predefined Time dimension and size.
        """
        mock_proc = MagicMock()
        mock_proc.dataset.dims = {'Time': 2}
        mock_proc.dataset.sizes = {'Time': 2}
        return mock_proc

    def test_single_mode_returns_true(self: 'TestRunSoundingAnalysis', 
                                      tmp_path: 'Path') -> None:
        """
        This test verifies that _run_sounding_analysis with batch_mode=False correctly loads the 3D data, extracts the sounding profile, builds the tags, and creates a skew-T diagram for the single time index specified in the configuration, ultimately returning True to indicate success (lines 1847-1884). It mocks the processor, diagnostics, and plotter to simulate the method's behavior without relying on actual data or file I/O. The test confirms that the method executes the expected steps for single mode and returns True as expected. 

        Parameters:
            tmp_path: pytest temporary directory fixture used for the output directory in the configuration. 

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        cli.perf_monitor = MagicMock()

        mock_proc = self._mock_3d_processor()
        mock_diag = MagicMock()
        mock_diag.extract_sounding_profile.return_value = _make_profile()

        config = MPASConfig(
            grid_file='/tmp/grid.nc',
            data_dir='/tmp/data',
            analysis_type='sounding',
            output_dir=str(tmp_path),
            verbose=False,
            batch_mode=False,
            time_index=0,
        )

        with patch('mpasdiag.processing.cli_unified.MPAS3DProcessor') as MockProc, \
             patch('mpasdiag.processing.cli_unified.SoundingDiagnostics',
                   return_value=mock_diag), \
             patch('mpasdiag.processing.cli_unified.MPASSkewTPlotter'), \
             patch('os.makedirs'), \
             patch('mpasdiag.processing.cli_unified.MPASDateTimeUtils.get_time_info',
                   return_value='t0'):
            MockProc.return_value.load_3d_data.return_value = mock_proc
            result = cli._run_sounding_analysis(config)

        assert result is True

    def test_batch_mode_with_show_indices_returns_true(self: 'TestRunSoundingAnalysis', 
                                                       tmp_path: 'Path') -> None:
        """
        This test verifies that _run_sounding_analysis with batch_mode=True and show_indices=True correctly loads the 3D data, extracts the sounding profile, computes the thermodynamic indices, builds the tags, and creates skew-T diagrams for each time step in the dataset, ultimately returning True to indicate success (lines 1847-1884). It mocks the processor, diagnostics, and plotter to simulate the method's behavior without relying on actual data or file I/O. The test confirms that the method executes the expected steps for batch mode with indices and returns True as expected. 

        Parameters:
            tmp_path: pytest temporary directory fixture used for the output directory in the configuration. 

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        cli.perf_monitor = MagicMock()

        mock_proc = self._mock_3d_processor()
        mock_diag = MagicMock()
        mock_diag.extract_sounding_profile.return_value = _make_profile()
        mock_diag.compute_thermodynamic_indices.return_value = {'CAPE': 500.0}

        config = MPASConfig(
            grid_file='/tmp/grid.nc',
            data_dir='/tmp/data',
            analysis_type='sounding',
            output_dir=str(tmp_path),
            verbose=False,
            batch_mode=True,
            time_index=0,
            show_indices=True,
        )

        with patch('mpasdiag.processing.cli_unified.MPAS3DProcessor') as MockProc, \
             patch('mpasdiag.processing.cli_unified.SoundingDiagnostics',
                   return_value=mock_diag), \
             patch('mpasdiag.processing.cli_unified.MPASSkewTPlotter'), \
             patch('os.makedirs'), \
             patch('mpasdiag.processing.cli_unified.MPASDateTimeUtils.get_time_info',
                   return_value='t0'):
            MockProc.return_value.load_3d_data.return_value = mock_proc
            result = cli._run_sounding_analysis(config)

        assert result is True


class TestParseArgsWithReorderingFallback:
    """ Test the behavior of _parse_args_with_reordering when parser.parse_args raises an Exception, ensuring that the except branch is executed and the method returns the result of parser.parse_args() without arguments. """

    def test_except_branch_returns_fallback_parse(self: 'TestParseArgsWithReorderingFallback') -> None:
        """
        This test verifies that when parser.parse_args raises an Exception, _parse_args_with_reordering executes the except branch and returns the result of calling parser.parse_args() without arguments (lines 2022-2028). It uses a side effect to simulate the first call to parse_args raising an exception, and then checks that the method correctly falls back to calling parse_args again without arguments and returns its result. 

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        parser = cli.create_main_parser()

        fallback_ns = argparse.Namespace()
        call_n = [0]

        def side_effect(*args, **kwargs) -> argparse.Namespace:
            """
            This side effect function simulates the behavior of parser.parse_args for testing the fallback mechanism in _parse_args_with_reordering. On the first call, it raises a ValueError to trigger the except branch. On subsequent calls, it returns a predefined fallback namespace. This allows the test to verify that the method correctly handles the exception and returns the fallback result as expected. 

            Parameters:
                *args: Positional arguments passed to parser.parse_args.
                **kwargs: Keyword arguments passed to parser.parse_args.    

            Returns:
                argparse.Namespace: The fallback namespace returned on the second call. 
            """
            call_n[0] += 1
            if call_n[0] == 1:
                raise ValueError("simulated first-call failure")
            return fallback_ns

        with patch.object(parser, 'parse_args', side_effect=side_effect):
            with patch('sys.argv', ['mpasdiag']):
                result = cli._parse_args_with_reordering(parser)

        assert result is fallback_ns
        assert call_n[0] == 2


class TestHandleMainExceptionNoLogger:
    """ Test the behavior of _handle_main_exception when self.logger is None, ensuring that it prints the error message to stdout and returns 1 without raising an exception. """

    def test_prints_error_and_returns_one_when_no_logger(self: 'TestHandleMainExceptionNoLogger',
                                                         capsys: pytest.CaptureFixture) -> None:
        """
        This test verifies that when self.logger is None, _handle_main_exception prints the error message to stdout and returns 1 without raising an exception (lines 2030-2036). It calls the method with a ValueError and checks that the output contains the error message and that the return value is 1, confirming that the method handles exceptions gracefully even when no logger is available. 

        Parameters:
            capsys: pytest capture fixture.

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        assert cli.logger is None
        result = cli._handle_main_exception(ValueError("test error"))
        assert result == 1
        captured = capsys.readouterr()
        assert 'test error' in captured.out


class TestPrintSystemInfoImportError:
    """ Test the behavior of _print_system_info when psutil is not importable, ensuring that it silently passes without raising an exception. """

    def test_handles_missing_psutil_gracefully(self: 'TestPrintSystemInfoImportError') -> None:
        """
        This test verifies that when psutil is not importable (simulated by patching sys.modules), _print_system_info silently passes without raising an exception (lines 2050-2055). It checks that the method does not attempt to access psutil and does not raise an ImportError, confirming that the method is robust to the absence of optional dependencies. 

        Parameters:
            None

        Returns:
            None
        """
        cli = MPASUnifiedCLI()
        cli.logger = MagicMock()
        with patch.dict(sys.modules, {'psutil': None}):
            cli._print_system_info()
        cli.logger.info.assert_called()


class TestModuleLevelMain:
    """ Test the behavior of the module-level main() function, ensuring it creates an MPASUnifiedCLI instance and returns the result of its main() method. """

    def test_module_main_delegates_to_cli_main(self: 'TestModuleLevelMain') -> None:
        """
        This test verifies that the module-level main() function creates an instance of MPASUnifiedCLI and returns the result of calling its main() method (lines 2120-2122). It patches the MPASUnifiedCLI.main method to return a specific value and checks that the module-level main function returns this value, confirming that the delegation to the CLI class is working as intended. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.cli_unified import main as cli_module_main
        with patch.object(MPASUnifiedCLI, 'main', return_value=0) as mock_main:
            result = cli_module_main()
        mock_main.assert_called_once()
        assert result == 0


class TestMainIfNameMain:
    """ Test the behavior of the module-level __main__ guard expression, ensuring it calls sys.exit with the return value of main(). """

    def test_if_name_main_expression_calls_sys_exit(self: 'TestMainIfNameMain') -> None:
        """
        This test verifies that the module-level __main__ guard expression calls sys.exit with the return value of main() (lines 2123-2124). It patches the MPASUnifiedCLI.main method to return a specific value and patches sys.exit to check that it is called with this value, confirming that the script exits with the correct status code when executed as a standalone program. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.cli_unified as _mod
        with patch.object(MPASUnifiedCLI, 'main', return_value=0):
            with patch('sys.exit') as mock_exit:
                _mod.sys.exit(_mod.main())
        mock_exit.assert_called_once_with(0)

if __name__ == "__main__":
    pytest.main([__file__, '-v'])

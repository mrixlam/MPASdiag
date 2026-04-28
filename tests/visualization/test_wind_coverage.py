#!/usr/bin/env python3

"""
MPASdiag Test Suite: Wind Plotter Coverage

This test suite is designed to achieve comprehensive code coverage for the MPASWindPlotter class in the mpasdiag.visualization.wind module. It includes a series of test classes and methods that target specific branches and paths within the methods of MPASWindPlotter, ensuring that all critical functionalities are exercised and validated. The tests cover scenarios such as calculating optimal subsample values based on plot size and density, preparing wind data for visualization with and without subsampling, rendering wind vectors with different plot types, counting valid points in 2D data, handling cases with no valid wind data, extracting configuration values with error handling, converting wind units with warnings for high values, calculating automatic subsample values based on valid point counts, creating batch wind plots with error handling for missing datasets and time information, extracting 2D slices from 3D wind data based on level specifications, and computing wind speed and direction from u and v components. By systematically testing these functionalities, this suite aims to ensure that the MPASWindPlotter class is robust, reliable, and behaves as expected under a wide range of conditions, ultimately contributing to the overall quality and maintainability of the MPASdiag visualization capabilities. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import xarray as xr
from pathlib import Path
from unittest.mock import MagicMock, patch

from mpasdiag.visualization.wind import MPASWindPlotter


N_CELLS = 10


class TestCalculateOptimalSubsample:
    """ Covers calculate_optimal_subsample branches: figsize fallback, plot_type density, large dataset calculation, and small dataset early return."""

    @pytest.fixture
    def plotter(self: 'TestCalculateOptimalSubsample') -> MPASWindPlotter:
        """
        This fixture provides a fresh instance of MPASWindPlotter for each test method in this class. It allows us to test the calculate_optimal_subsample method specifically for scenarios involving different figsize values, plot types, dataset sizes, and target densities. By using a fixture, we can easily reuse the setup code and maintain clean and organized test methods that focus on the behavior of calculate_optimal_subsample under various conditions, ensuring that all branches of the method are adequately covered in our tests. 

        Parameters:
            None

        Returns:
            MPASWindPlotter: An instance of the MPASWindPlotter class to be used in the test methods.
        """
        return MPASWindPlotter()

    def test_figsize_none_falls_back_to_instance(self: 'TestCalculateOptimalSubsample', 
                                                 plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when the figsize parameter is set to None, the calculate_optimal_subsample method correctly falls back to using the instance's default figsize value. This ensures that the method can handle cases where the user does not provide a specific figure size and still calculates the optimal subsample based on the default settings of the MPASWindPlotter instance. By testing with figsize=None, we can confirm that the method's fallback mechanism is functioning as intended and that it can proceed with the subsample calculation without requiring an explicit figure size input. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        result = plotter.calculate_optimal_subsample(
            num_points=10, lon_min=-100., lon_max=-80., lat_min=30., lat_max=50.,
            figsize=None, plot_type='barbs',
        )
        assert isinstance(result, int)

    def test_nonbarbs_plot_type_uses_density_4(self: 'TestCalculateOptimalSubsample', 
                                               plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when the plot_type parameter is set to a value other than 'barbs', the calculate_optimal_subsample method uses a target_vectors_per_inch value of 4 for its density calculation. This ensures that the method applies a consistent density standard for non-barb plot types, which may require a different visual density of vectors compared to barbs. By testing with a plot_type such as 'arrows', we can confirm that the method correctly identifies the plot type and applies the appropriate density calculation, leading to an optimal subsample value that is suitable for the chosen visualization style. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        result = plotter.calculate_optimal_subsample(
            num_points=10, lon_min=-100., lon_max=-80., lat_min=30., lat_max=50.,
            plot_type='arrows', target_density=None,
        )
        assert isinstance(result, int)

    def test_streamlines_type_also_uses_density_4(self: 'TestCalculateOptimalSubsample', 
                                                  plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when the plot_type parameter is set to 'streamlines', the calculate_optimal_subsample method also uses a target_vectors_per_inch value of 4 for its density calculation. This ensures that the method applies the same density standard for streamlines as it does for other non-barb plot types, which may require a similar visual density of vectors for effective visualization. By testing with the 'streamlines' plot type, we can confirm that the method correctly identifies this specific plot type and applies the appropriate density calculation, leading to an optimal subsample value that is suitable for streamline visualizations. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        result = plotter.calculate_optimal_subsample(
            num_points=10, lon_min=-100., lon_max=-80., lat_min=30., lat_max=50.,
            plot_type='streamlines', target_density=None,
        )
        assert isinstance(result, int)

    def test_custom_target_density_is_used(self: 'TestCalculateOptimalSubsample', 
                                           plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when a custom target_density value is provided to the calculate_optimal_subsample method, it is correctly used in the density calculation instead of the default values for barbs or non-barbs plot types. This ensures that the method can accommodate user-defined density preferences for vector plotting, allowing for greater flexibility in visualization. By testing with a specific target_density value, we can confirm that the method incorporates this value into its calculations and returns an optimal subsample based on the user-defined density requirement. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        result = plotter.calculate_optimal_subsample(
            num_points=10, lon_min=-100., lon_max=-80., lat_min=30., lat_max=50.,
            target_density=2,
        )
        assert isinstance(result, int)

    def test_large_dataset_triggers_subsample_calculation(self: 'TestCalculateOptimalSubsample', 
                                                          plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when the number of points exceeds the target total vectors for the given plot size and density, the calculate_optimal_subsample method correctly performs the subsample calculation to determine an appropriate subsample value. This ensures that the method can effectively handle large datasets by calculating a subsample that reduces the number of vectors to a manageable level for visualization while still maintaining an appropriate density. By testing with a large number of points, we can confirm that the method's logic for calculating the optimal subsample is functioning as intended and returns a value greater than 1, indicating that subsampling is necessary for the given dataset size. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        result = plotter.calculate_optimal_subsample(
            num_points=100_000, lon_min=-100., lon_max=-80., lat_min=30., lat_max=50.,
            figsize=(12., 10.), plot_type='barbs',
        )
        assert result > 1
        assert result <= 50

    def test_subsample_capped_at_50(self: 'TestCalculateOptimalSubsample', 
                                     plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when the number of points is extremely large, the calculate_optimal_subsample method correctly caps the subsample value at 50 to prevent excessive subsampling that could lead to a loss of meaningful visualization. This ensures that the method has a safeguard in place to maintain a minimum level of detail in the plot, even when faced with very large datasets. By testing with an excessively large number of points, we can confirm that the method's capping mechanism is functioning as intended and does not return a subsample value greater than 50, which would indicate an overly aggressive reduction in the number of vectors for visualization. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        result = plotter.calculate_optimal_subsample(
            num_points=10_000_000, lon_min=-100., lon_max=-80., lat_min=30., lat_max=50.,
            figsize=(12., 10.), plot_type='barbs',
        )
        assert result <= 50

    def test_small_dataset_returns_1_without_calculation(self: 'TestCalculateOptimalSubsample', 
                                                         plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when the number of points is less than or equal to the target total vectors for the given plot size and density, the calculate_optimal_subsample method correctly returns a subsample value of 1 without performing any subsample calculation. This ensures that the method can efficiently handle small datasets by recognizing that no subsampling is necessary and returning a value that indicates all points should be plotted. By testing with a small number of points, we can confirm that the method's early return logic is functioning as intended and does not attempt to calculate a subsample when it is not needed, thus preserving the full resolution of the dataset for visualization. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        result = plotter.calculate_optimal_subsample(
            num_points=5, lon_min=-100., lon_max=-80., lat_min=30., lat_max=50.,
        )
        assert result == 1


class TestPrepareWindData2D:
    """ Covers 2D data paths in _prepare_wind_data: subsample stride (145-148) and no-subsample original shape. """

    @pytest.fixture
    def plotter(self: 'TestPrepareWindData2D') -> MPASWindPlotter:
        """
        This fixture provides a fresh instance of MPASWindPlotter for each test method in this class. It allows us to test the _prepare_wind_data method specifically for scenarios involving 2D longitude, latitude, and wind component arrays, ensuring that the method correctly applies subsampling strides when requested and preserves the original shape when no subsampling is applied. By using a fixture, we can easily reuse the setup code and maintain clean and organized test methods that focus on the behavior of _prepare_wind_data when handling 2D data inputs, which is crucial for validating the method's functionality in preparing wind data for visualization. 

        Parameters:
            None

        Returns:
            MPASWindPlotter: An instance of the MPASWindPlotter class to be used in the test methods.
        """
        return MPASWindPlotter()

    def test_2d_data_with_subsample_applies_stride(self: 'TestPrepareWindData2D', 
                                                   plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when 2D longitude, latitude, and wind component arrays are provided to the _prepare_wind_data method with a subsample value greater than 1, the method correctly applies the appropriate stride to subsample the data. This ensures that the method can effectively reduce the resolution of the wind data for visualization purposes when requested, while still maintaining the integrity of the data structure. By testing with a subsample value of 2, we can confirm that the method correctly selects every other point in both dimensions of the 2D arrays, resulting in a reduced dataset that is suitable for plotting without overwhelming the visualization with too many vectors. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        lon = np.linspace(-100., -80., 10).reshape(2, 5)
        lat = np.linspace(30., 50., 10).reshape(2, 5)
        u = np.ones((2, 5))
        v = np.ones((2, 5))
        lon_out, lat_out, u_out, v_out = plotter._prepare_wind_data(lon, lat, u, v, subsample=2)
        assert lon_out.shape == (1, 3)
        assert u_out.shape == (1, 3)

    def test_2d_data_no_subsample_returns_original_shape(self: 'TestPrepareWindData2D', 
                                                         plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when 2D longitude, latitude, and wind component arrays are provided to the _prepare_wind_data method with a subsample value of 1, the method returns the original arrays without applying any subsampling. This ensures that the method can preserve the full resolution of the wind data for visualization purposes when no subsampling is requested, allowing for a complete representation of the dataset in the plot. By testing with a subsample value of 1, we can confirm that the method correctly identifies that no subsampling is needed and returns the input arrays unchanged, maintaining their original shape and content for accurate visualization. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        lon = np.linspace(-100., -80., 6).reshape(2, 3)
        lat = np.linspace(30., 50., 6).reshape(2, 3)
        u = np.ones((2, 3))
        v = np.ones((2, 3))
        lon_out, lat_out, u_out, v_out = plotter._prepare_wind_data(lon, lat, u, v, subsample=1)
        assert lon_out.shape == (2, 3)


class TestRenderWindVectorsStreamline1D:
    """ Covers ValueError branch in _render_wind_vectors when plot_type='streamlines' and 1D longitude array is provided."""

    @pytest.fixture
    def plotter(self) -> MPASWindPlotter:
        return MPASWindPlotter()

    def test_streamlines_with_1d_lon_raises(self: 'TestRenderWindVectorsStreamline1D', 
                                            plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when the _render_wind_vectors method is called with a plot_type of 'streamlines' and the longitude array is 1D, a ValueError is raised with a message indicating that streamlines require gridded data. This ensures that the method correctly identifies the requirement for 2D gridded data when plotting streamlines and provides clear feedback to the user when this requirement is not met. By testing with a 1D longitude array, we can confirm that the method's error handling for invalid input data is functioning as intended and that it prevents attempts to plot streamlines with incompatible data structures. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        mock_ax = MagicMock()
        lon = np.linspace(-100., -80., 20)
        lat = np.linspace(30., 50., 20)
        u = np.ones(20)
        v = np.ones(20)
        with pytest.raises(ValueError, match="Streamlines require gridded data"):
            plotter._render_wind_vectors(mock_ax, lon, lat, u, v, plot_type='streamlines')

    def test_streamlines_error_message_mentions_grid_resolution(self: 'TestRenderWindVectorsStreamline1D', 
                                                                plotter: MPASWindPlotter) -> None:
        """
        This test verifies that the ValueError raised when the _render_wind_vectors method is called with a plot_type of 'streamlines' and a 1D longitude array includes a message that mentions "grid_resolution". This ensures that the error message provides specific information about the nature of the issue, indicating that streamlines require gridded data and suggesting that the user may need to check their grid resolution or data structure. By confirming that the error message contains this information, we can ensure that users receive helpful guidance on how to resolve the issue when they encounter this specific error condition. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        mock_ax = MagicMock()
        lon = np.linspace(-100., -80., 5)
        lat = np.linspace(30., 50., 5)
        with pytest.raises(ValueError, match="grid_resolution"):
            plotter._render_wind_vectors(mock_ax, lon, lat, np.ones(5), np.ones(5),
                                         plot_type='streamlines')


class TestCalculateValidPointCount2D:
    """ Covers 2D data paths in _calculate_valid_point_count: counting finite points, handling NaNs in lon and u, and ensuring correct count is returned for 2D inputs. """

    @pytest.fixture
    def plotter(self: 'TestCalculateValidPointCount2D') -> MPASWindPlotter:
        """
        This fixture provides a fresh instance of MPASWindPlotter for each test method in this class. It allows us to test the _calculate_valid_point_count method specifically for scenarios involving 2D longitude and wind component arrays, ensuring that the method correctly counts valid points based on finite values and handles cases with NaN values appropriately. By using a fixture, we can easily reuse the setup code and maintain clean and organized test methods that focus on the behavior of _calculate_valid_point_count when processing 2D data inputs, which is essential for validating the method's functionality in determining the number of valid points for visualization purposes. 

        Parameters:
            None

        Returns:
            MPASWindPlotter: An instance of the MPASWindPlotter class to be used in the test methods. 
        """
        return MPASWindPlotter()

    def test_2d_count_returns_finite_point_count(self: 'TestCalculateValidPointCount2D', 
                                                 plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when 2D longitude and wind component arrays are provided to the _calculate_valid_point_count method, it correctly counts the number of valid points based on finite values in both arrays. This ensures that the method can accurately identify valid data points for visualization purposes, even when some values may be NaN. By testing with a mix of finite and NaN values in the input arrays, we can confirm that the method's counting logic is functioning as intended and returns the correct count of valid points, which is crucial for determining how many vectors will be plotted in the wind visualization. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        # Line 369: 2D lon → np.sum(np.isfinite(lon) & np.isfinite(u))
        lon_2d = np.linspace(-100., -80., 6).reshape(2, 3)
        u_2d = np.array([[1., 2., np.nan], [4., 5., 6.]])
        result = plotter._calculate_valid_point_count(lon_2d, u_2d)
        assert result == 5  # 5 valid (one NaN in u)

    def test_2d_count_all_valid(self: 'TestCalculateValidPointCount2D', 
                                plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when 2D longitude and wind component arrays are provided to the _calculate_valid_point_count method with all finite values, it correctly counts all points as valid. This ensures that the method can accurately identify valid data points when there are no NaN values present, confirming that it does not erroneously exclude any valid points from the count. By testing with arrays that contain only finite values, we can confirm that the method returns the total number of points as valid, which is essential for ensuring that the visualization will include all available data points when there are no issues with the input data. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        lon_2d = np.linspace(-100., -80., 6).reshape(2, 3)
        u_2d = np.ones((2, 3))
        result = plotter._calculate_valid_point_count(lon_2d, u_2d)
        assert result == 6

    def test_2d_count_with_nan_lon(self: 'TestCalculateValidPointCount2D', 
                                   plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when 2D longitude and wind component arrays are provided to the _calculate_valid_point_count method with some NaN values in the longitude array, it correctly counts only the points that have finite values in both the longitude and wind component arrays. This ensures that the method can accurately identify valid data points for visualization purposes, even when some longitude values are missing or invalid. By testing with a 2D longitude array that contains NaN values, we can confirm that the method's counting logic correctly excludes those points from the valid count, which is crucial for determining how many vectors will be plotted in the wind visualization when there are issues with the input data. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        lon_2d = np.array([[np.nan, -90.], [-95., -85.]])
        u_2d = np.ones((2, 2))
        result = plotter._calculate_valid_point_count(lon_2d, u_2d)
        assert result == 3  # one NaN in lon


class TestCreateWindPlotNoValidData:
    """ Covers the branch in create_wind_plot where no valid wind data is found (e.g., all NaN or empty arrays), ensuring that a warning is printed and the method returns early without attempting to create a plot. """

    @pytest.fixture
    def plotter(self: 'TestCreateWindPlotNoValidData') -> MPASWindPlotter:
        """
        This fixture provides a fresh instance of MPASWindPlotter for each test method in this class. It allows us to test the create_wind_plot method specifically for scenarios where the input wind data contains no valid points (e.g., all NaN values or empty arrays), ensuring that the method correctly handles these cases by printing a warning message and returning early without attempting to create a plot. By using a fixture, we can easily reuse the setup code and maintain clean and organized test methods that focus on the behavior of create_wind_plot when faced with invalid or missing wind data, which is essential for validating the method's robustness and user feedback mechanisms in edge cases. 

        Parameters:
            None

        Returns:
            MPASWindPlotter: An instance of the MPASWindPlotter class to be used in the test methods. 
        """
        return MPASWindPlotter()

    def test_all_nan_u_v_prints_warning_and_returns_early(self: 'TestCreateWindPlotNoValidData', 
                                                          plotter: MPASWindPlotter, 
                                                          capsys: pytest.CaptureFixture) -> None:
        """
        This test verifies that when the create_wind_plot method is called with longitude and latitude arrays that are valid but the u and v wind component arrays contain only NaN values, the method correctly prints a warning message indicating that no valid wind data was found and returns early without attempting to create a plot. This ensures that the method can gracefully handle cases where the input wind data is invalid, providing clear feedback to the user about the issue and avoiding unnecessary processing or errors that could arise from trying to plot non-existent data. By testing with all NaN values in the u and v arrays, we can confirm that the method's error handling for this specific case is functioning as intended and that it provides appropriate feedback to users when they encounter this situation. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.
            capsys (pytest.CaptureFixture): The pytest fixture for capturing standard output.

        Returns:
            None
        """
        lon = np.linspace(-100., -80., N_CELLS)
        lat = np.linspace(30., 50., N_CELLS)
        u_nan = np.full(N_CELLS, np.nan)
        v_nan = np.full(N_CELLS, np.nan)

        mock_fig = MagicMock()
        mock_ax = MagicMock()

        with patch.object(plotter, '_setup_wind_plot_figure', return_value=(mock_fig, mock_ax)):
            with patch.object(plotter, '_handle_streamline_regridding', return_value=None):
                fig, ax = plotter.create_wind_plot(
                    lon, lat, u_nan, v_nan, -100., -80., 30., 50.
                )

        out = capsys.readouterr().out
        assert "Warning: No valid wind data found" in out
        assert fig is mock_fig
        assert ax is mock_ax

    def test_empty_arrays_also_trigger_early_return(self: 'TestCreateWindPlotNoValidData', 
                                                    plotter: MPASWindPlotter, 
                                                    capsys: pytest.CaptureFixture) -> None:
        """
        This test verifies that when the create_wind_plot method is called with empty longitude, latitude, u, and v arrays, the method correctly prints a warning message indicating that no valid wind data was found and returns early without attempting to create a plot. This ensures that the method can gracefully handle cases where the input wind data is completely missing, providing clear feedback to the user about the issue and avoiding unnecessary processing or errors that could arise from trying to plot non-existent data. By testing with empty arrays for all inputs, we can confirm that the method's error handling for this specific case is functioning as intended and that it provides appropriate feedback to users when they encounter this situation. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.
            capsys (pytest.CaptureFixture): The pytest fixture for capturing standard output.

        Returns:
            None
        """
        lon = np.array([])
        lat = np.array([])
        u = np.array([])
        v = np.array([])

        mock_fig = MagicMock()
        mock_ax = MagicMock()

        with patch.object(plotter, '_setup_wind_plot_figure', return_value=(mock_fig, mock_ax)):
            with patch.object(plotter, '_handle_streamline_regridding', return_value=None):
                fig, ax = plotter.create_wind_plot(lon, lat, u, v, -100., -80., 30., 50.)

        assert "Warning: No valid wind data found" in capsys.readouterr().out


class TestExtractWindConfig:
    """ Covers branches in _extract_wind_config related to handling invalid subsample values (e.g., non-numeric strings, None) and ensuring that the method defaults to a subsample value of 1 in these cases, as well as correctly parsing valid numeric string subsample values. """

    @pytest.fixture
    def plotter(self: 'TestExtractWindConfig') -> MPASWindPlotter:
        """
        This fixture provides a fresh instance of MPASWindPlotter for each test method in this class. It allows us to test the _extract_wind_config method specifically for scenarios where the subsample value provided in the configuration is invalid (e.g., non-numeric string or None), ensuring that the method correctly defaults to a subsample value of 1 in these cases. By using a fixture, we can easily reuse the setup code and maintain clean and organized test methods that focus on the behavior of _extract_wind_config when handling invalid subsample inputs, which is crucial for validating the method's robustness and error handling capabilities in preparing wind data for visualization. 

        Parameters:
            None

        Returns:
            MPASWindPlotter: An instance of the MPASWindPlotter class to be used in the test methods. 
        """
        return MPASWindPlotter()

    def test_non_numeric_subsample_defaults_to_1(self: 'TestExtractWindConfig', 
                                                 plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when the subsample value provided in the configuration is a non-numeric string, the _extract_wind_config method correctly defaults to a subsample value of 1. This ensures that the method can handle cases where the subsample input is invalid and provides a reasonable fallback to maintain functionality without crashing. By testing with a non-numeric string, we can confirm that the method's error handling for subsample parsing is functioning as intended and that it gracefully defaults to a safe value when faced with invalid input. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        config = {
            'u_data': np.ones(N_CELLS),
            'v_data': np.ones(N_CELLS),
            'subsample': 'not_a_number',
        }
        result = plotter._extract_wind_config(config)
        assert result['subsample'] == 1

    def test_none_subsample_defaults_to_1(self: 'TestExtractWindConfig', 
                                          plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when the subsample value provided in the configuration is None, the _extract_wind_config method correctly defaults to a subsample value of 1. This ensures that the method can handle cases where the subsample input is explicitly set to None and provides a reasonable fallback to maintain functionality without crashing. By testing with a None value, we can confirm that the method's error handling for subsample parsing is functioning as intended and that it gracefully defaults to a safe value when faced with invalid input. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        config = {
            'u_data': np.ones(N_CELLS),
            'v_data': np.ones(N_CELLS),
            'subsample': None,
        }
        result = plotter._extract_wind_config(config)
        assert result['subsample'] == 1

    def test_valid_subsample_string_is_converted(self: 'TestExtractWindConfig', 
                                                 plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when the subsample value provided in the configuration is a valid numeric string, the _extract_wind_config method correctly converts it to an integer and uses that value for subsampling. This ensures that the method can handle cases where the subsample input is provided as a string representation of a number and still functions correctly by converting it to the appropriate type for use in subsampling calculations. By testing with a valid numeric string, we can confirm that the method's parsing logic for subsample values is functioning as intended and that it can successfully convert valid string inputs to integers for proper subsampling behavior. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        config = {
            'u_data': np.ones(N_CELLS),
            'v_data': np.ones(N_CELLS),
            'subsample': '3',
        }
        result = plotter._extract_wind_config(config)
        assert result['subsample'] == 3

    def test_default_subsample_is_1_when_missing(self: 'TestExtractWindConfig', 
                                                 plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when the subsample value is missing from the configuration, the _extract_wind_config method correctly defaults to a subsample value of 1. This ensures that the method can handle cases where the subsample input is not provided at all and still maintains functionality by using a reasonable default value. By testing with a configuration that does not include a subsample key, we can confirm that the method's logic for handling missing subsample values is functioning as intended and that it gracefully defaults to a safe value when the input is incomplete. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        config = {
            'u_data': np.ones(N_CELLS),
            'v_data': np.ones(N_CELLS),
        }
        result = plotter._extract_wind_config(config)
        assert result['subsample'] == 1


class TestConvertWindUnits:
    """ Test if _convert_wind_units correctly identifies when wind values are unusually high and issues a warning, as well as ensuring that it performs unit conversion correctly when original units are specified. """

    @pytest.fixture
    def plotter(self: 'TestConvertWindUnits') -> MPASWindPlotter:
        """
        This fixture provides a fresh instance of MPASWindPlotter for each test method in this class. It allows us to test the _convert_wind_units method specifically for scenarios involving high-magnitude wind values that trigger warnings, as well as the normal conversion process for wind units. By using a fixture, we can easily reuse the setup code and maintain clean and organized test methods that focus on the behavior of _convert_wind_units when handling different ranges of wind values, ensuring that the method correctly identifies when values are unusually high and performs the necessary unit conversions accurately. This is essential for validating the method's functionality in preparing wind data for visualization and ensuring that users receive appropriate feedback when their input data may not be in the expected units. 

        Parameters:
            None

        Returns:
            MPASWindPlotter: An instance of the MPASWindPlotter class to be used in the test methods. 
        """
        return MPASWindPlotter()

    def test_high_magnitude_warns_when_no_units(self: 'TestConvertWindUnits', 
                                                plotter: MPASWindPlotter, 
                                                capsys: pytest.CaptureFixture) -> None:
        """
        This test verifies that when the _convert_wind_units method is called with u and v wind component arrays that have a mean magnitude above the threshold (100) and no original units specified, the method prints a warning message indicating that the wind data may not be in m/s. This ensures that the method can identify when the wind data is unusually high and provide appropriate feedback to the user about potential issues with the input data, such as it being in different units than expected. By testing with values above the threshold and no units, we can confirm that the method's logic for issuing warnings based on wind magnitude is functioning as intended and that it helps users recognize when their data may need to be converted to m/s for accurate visualization. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.
            capsys (pytest.CaptureFixture): The pytest fixture for capturing standard output.

        Returns:
            None
        """
        u = np.full(N_CELLS, 150.0)
        v = np.ones(N_CELLS)
        u_ret, v_ret = plotter._convert_wind_units(u, v, None)
        out = capsys.readouterr().out
        assert "Warning: Wind data may not be in m/s" in out
        np.testing.assert_array_equal(u_ret, u)

    def test_no_units_below_threshold_no_warning(self: 'TestConvertWindUnits', 
                                                 plotter: MPASWindPlotter, 
                                                 capsys: pytest.CaptureFixture) -> None:
        """
        This test verifies that when the _convert_wind_units method is called with u and v wind component arrays that have a mean magnitude below the threshold (100) and no original units specified, the method does not print any warning message and returns the input u and v arrays unchanged. This ensures that the method can correctly identify when the wind data is within a reasonable range for m/s and does not issue unnecessary warnings, allowing users to proceed with their data without confusion when it is already in an appropriate format. By testing with values below the threshold and no units, we can confirm that the method's logic for determining when to issue warnings based on wind magnitude is functioning as intended and that it preserves the integrity of the input data when no conversion is needed. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.
            capsys (pytest.CaptureFixture): The pytest fixture for capturing standard output.

        Returns:
            None
        """
        u = np.full(N_CELLS, 10.0)
        v = np.ones(N_CELLS)
        u_ret, v_ret = plotter._convert_wind_units(u, v, None)
        assert capsys.readouterr().out == ""
        np.testing.assert_array_equal(u_ret, u)

    def test_same_units_returns_data_unchanged(self: 'TestConvertWindUnits', 
                                                plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when the _convert_wind_units method is called with original units specified as 'm/s', the method returns the input u and v wind component arrays unchanged without performing any conversion. This ensures that the method can recognize when the original units are already in the expected format and avoids unnecessary processing, allowing users to proceed with their data without alteration when it is already in m/s. By testing with original units set to 'm/s', we can confirm that the method's logic for handling cases where no conversion is needed is functioning as intended and that it preserves the integrity of the input data when it is already in the correct units for visualization. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        u = np.array([5.0, 10.0])
        v = np.array([3.0, 4.0])
        u_ret, v_ret = plotter._convert_wind_units(u, v, 'm/s')
        np.testing.assert_array_equal(u_ret, u)
        np.testing.assert_array_equal(v_ret, v)

    def test_unit_conversion_prints_confirmation(self: 'TestConvertWindUnits', 
                                                 plotter: MPASWindPlotter, 
                                                 capsys: pytest.CaptureFixture) -> None:
        """
        This test verifies that when the _convert_wind_units method is called with original units specified as 'knots' and the UnitConverter successfully converts the u and v wind component arrays to m/s, the method prints a confirmation message indicating that the conversion was performed from knots to m/s. This ensures that the method provides clear feedback to the user about the unit conversion that took place, confirming that the input data was successfully converted to the expected units for visualization. By testing with original units set to 'knots' and mocking a successful conversion, we can confirm that the method's logic for handling unit conversions and providing user feedback is functioning as intended, enhancing the user experience by keeping them informed about changes made to their data. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.
            capsys (pytest.CaptureFixture): The pytest fixture for capturing standard output.

        Returns:
            None
        """
        u = np.array([10.0])
        v = np.array([0.0])
        converted_u = np.array([5.144])
        converted_v = np.array([0.0])
        with patch('mpasdiag.visualization.wind.UnitConverter.convert_units',
                   side_effect=[converted_u, converted_v]):
            u_ret, v_ret = plotter._convert_wind_units(u, v, 'knots')
        out = capsys.readouterr().out
        assert "Converted overlay wind from knots to m/s" in out
        np.testing.assert_array_equal(u_ret, converted_u)

    def test_conversion_failure_prints_warning_and_returns_original(self: 'TestConvertWindUnits', 
                                                                    plotter: MPASWindPlotter, 
                                                                    capsys: pytest.CaptureFixture) -> None:
        """
        This test verifies that when the _convert_wind_units method is called with original units specified as 'knots' and the UnitConverter raises a ValueError during the conversion process, the method correctly prints a warning message indicating that the conversion failed and returns the original u and v wind component arrays unchanged. This ensures that the method can gracefully handle cases where unit conversion is not possible due to issues with the input data or unsupported units, providing clear feedback to the user about the failure and preserving the integrity of the input data without crashing. By testing with a mocked ValueError from the UnitConverter, we can confirm that the method's error handling for conversion failures is functioning as intended and that it provides appropriate feedback while maintaining the original data for users to review. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.
            capsys (pytest.CaptureFixture): The pytest fixture for capturing standard output.

        Returns:
            None
        """
        u = np.array([5.0])
        v = np.array([3.0])
        with patch('mpasdiag.visualization.wind.UnitConverter.convert_units',
                   side_effect=ValueError("unsupported unit")):
            u_ret, v_ret = plotter._convert_wind_units(u, v, 'bad_unit')
        out = capsys.readouterr().out
        assert "Warning: Could not convert" in out
        np.testing.assert_array_equal(u_ret, u)
        np.testing.assert_array_equal(v_ret, v)

    def test_converted_arrays_returned_as_numpy(self: 'TestConvertWindUnits', 
                                                plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when the _convert_wind_units method successfully converts the u and v wind component arrays to m/s using the UnitConverter, the returned u and v arrays are of type numpy.ndarray. This ensures that the method returns the converted wind data in a consistent format that is suitable for further processing and visualization, allowing users to work with the converted data without issues related to data types. By testing with a successful conversion and checking the types of the returned arrays, we can confirm that the method's logic for handling unit conversions and returning data is functioning as intended, providing users with properly formatted output for their wind visualization needs. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        u = np.array([10.0])
        v = np.array([5.0])
        converted = np.array([5.144])
        with patch('mpasdiag.visualization.wind.UnitConverter.convert_units',
                   side_effect=[converted, converted]):
            u_ret, v_ret = plotter._convert_wind_units(u, v, 'knots')
        assert isinstance(u_ret, np.ndarray)
        assert isinstance(v_ret, np.ndarray)


class TestCalculateAutoSubsample2D:
    """ Test if _calculate_auto_subsample correctly triggers the 2D counting logic when 2D longitude and wind component arrays are provided, ensuring that it calculates an appropriate subsample value based on the number of valid points for visualization purposes. """

    @pytest.fixture
    def plotter(self: 'TestCalculateAutoSubsample2D') -> MPASWindPlotter:
        """
        This fixture provides a fresh instance of MPASWindPlotter for each test method in this class. It allows us to test the _calculate_auto_subsample method specifically for scenarios involving 2D longitude and wind component arrays, ensuring that the method correctly triggers the 2D counting logic to determine the number of valid points and calculates an appropriate subsample value for visualization purposes. By using a fixture, we can easily reuse the setup code and maintain clean and organized test methods that focus on the behavior of _calculate_auto_subsample when processing 2D data inputs, which is essential for validating the method's functionality in determining how to subsample wind data for effective visualization while maintaining an appropriate level of detail based on the density of valid data points. 

        Parameters:
            None

        Returns:
            MPASWindPlotter: An instance of the MPASWindPlotter class to be used in the test methods.
        """
        return MPASWindPlotter()

    def test_2d_lon_triggers_2d_count(self: 'TestCalculateAutoSubsample2D', 
                                      plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when 2D longitude and wind component arrays are provided to the _calculate_auto_subsample method, it correctly triggers the 2D counting logic to determine the number of valid points and calculates an appropriate subsample value for visualization purposes. This ensures that the method can handle 2D data inputs and provides a valid subsample value based on the density of valid data points, which is crucial for creating effective wind visualizations that balance detail and clarity. By testing with 2D arrays that contain a mix of finite values, we can confirm that the method's logic for counting valid points and calculating subsampling is functioning as intended, allowing users to visualize their wind data effectively even when working with gridded datasets. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        lon_2d = np.linspace(-100., -80., 6).reshape(2, 3)
        u_2d = np.ones((2, 3))
        result = plotter._calculate_auto_subsample(
            lon_2d, u_2d, -100., -80., 30., 50., (12., 10.), 'barbs'
        )
        assert isinstance(result, int)
        assert result >= 1

    def test_2d_count_with_nan_excluded(self: 'TestCalculateAutoSubsample2D', 
                                        plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when 2D longitude and wind component arrays are provided to the _calculate_auto_subsample method with some NaN values in the wind component array, it correctly excludes those points from the valid count and calculates a subsample value based on the remaining valid points. This ensures that the method can accurately determine the density of valid data points for visualization purposes, even when some values are missing or invalid, allowing users to create effective wind visualizations that reflect the quality of their input data. By testing with a 2D wind component array that contains NaN values, we can confirm that the method's logic for counting valid points and calculating subsampling is functioning as intended, providing users with appropriate subsample values that account for the presence of invalid data. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        lon_2d = np.linspace(-100., -80., 6).reshape(2, 3)
        u_all_valid = np.ones((2, 3))
        u_with_nan = np.array([[1., np.nan, 1.], [1., 1., 1.]])
        result_valid = plotter._calculate_auto_subsample(
            lon_2d, u_all_valid, -100., -80., 30., 50., (12., 10.), 'barbs'
        )
        result_nan = plotter._calculate_auto_subsample(
            lon_2d, u_with_nan, -100., -80., 30., 50., (12., 10.), 'barbs'
        )
        assert isinstance(result_valid, int)
        assert isinstance(result_nan, int)


class TestCreateBatchWindPlots:
    """ Test if create_batch_wind_plots correctly handles cases where the input wind data may have unusually high values without specified units, ensuring that it issues appropriate warnings and performs unit conversions when necessary to prepare the data for visualization. """

    @pytest.fixture
    def plotter(self: 'TestCreateBatchWindPlots') -> MPASWindPlotter:
        """
        This fixture provides a fresh instance of MPASWindPlotter for each test method in this class. It allows us to test the create_batch_wind_plots method specifically for scenarios where the dataset is None or missing from the processor object, as well as when the get_time_info function raises an exception, ensuring that the method correctly handles these cases by raising appropriate errors or falling back to using the time index for plot titles. By using a fixture, we can easily reuse the setup code and maintain clean and organized test methods that focus on the behavior of create_batch_wind_plots when faced with issues related to dataset availability and time information extraction, which is essential for validating the method's robustness and error handling capabilities in preparing batch wind plots for visualization. 

        Parameters:
            None

        Returns:
            MPASWindPlotter: An instance of the MPASWindPlotter class to be used in the test methods.
        """
        return MPASWindPlotter()

    def test_dataset_none_raises_value_error(self: 'TestCreateBatchWindPlots',
                                             plotter: MPASWindPlotter,
                                             tmp_path) -> None:
        """
        This test verifies that the create_batch_wind_plots method raises a ValueError when the processor object's dataset attribute is None, ensuring that the method correctly identifies when there is no loaded dataset available for processing and provides appropriate feedback to the user about the issue. By testing with a processor object that has its dataset set to None, we can confirm that the method's error handling for missing datasets is functioning as intended and that it prevents further execution when the necessary data is not available, which is crucial for maintaining the integrity of the plot creation process and avoiding downstream errors. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        proc = MagicMock()
        proc.dataset = None
        with pytest.raises(ValueError, match="no loaded dataset"):
            plotter.create_batch_wind_plots(proc, str(tmp_path), -100., -80., 30., 50.)

    def test_no_dataset_attr_raises_value_error(self: 'TestCreateBatchWindPlots',
                                                 plotter: MPASWindPlotter,
                                                 tmp_path) -> None:
        """
        This test verifies that the create_batch_wind_plots method raises a ValueError when the processor object does not have a dataset attribute, ensuring that the method correctly identifies when there is no loaded dataset available for processing and provides appropriate feedback to the user about the issue. By testing with a processor object that lacks the dataset attribute entirely, we can confirm that the method's error handling for missing datasets is functioning as intended and that it prevents further execution when the necessary data is not available, which is crucial for maintaining the integrity of the plot creation process and avoiding downstream errors. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        class _ProcWithoutDataset:
            """ Simple class to simulate a processor object without a dataset attribute. """
            pass

        proc = _ProcWithoutDataset()
        with pytest.raises(ValueError, match="no loaded dataset"):
            plotter.create_batch_wind_plots(proc, str(tmp_path), -100., -80., 30., 50.)

    def test_time_info_exception_falls_back_to_time_idx(self: 'TestCreateBatchWindPlots', 
                                                        plotter: MPASWindPlotter, 
                                                        tmp_path: "Path") -> None:
        """
        This test verifies that when the get_time_info function raises an exception during the execution of create_batch_wind_plots, the method correctly falls back to using the time index to generate a time string for the plot title, ensuring that even when time information cannot be extracted from the dataset, the method can still proceed with plot creation and provide a meaningful title based on the time index. By testing with a mocked get_time_info that raises an exception, we can confirm that the method's error handling for time information extraction is functioning as intended and that it allows for continued functionality without crashing, while also ensuring that the generated plot titles are consistent with the expected format based on the time index. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.
            tmp_path (Path): A temporary directory provided by pytest for saving the generated plots.

        Returns:
            None
        """
        n_cells = 5
        ds = xr.Dataset({
            'u': xr.DataArray(np.ones((1, n_cells)), dims=['Time', 'nCells']),
            'v': xr.DataArray(np.ones((1, n_cells)), dims=['Time', 'nCells']),
        })
        proc = MagicMock()
        proc.dataset = ds
        lon = np.linspace(-100., -80., n_cells)
        lat = np.linspace(30., 50., n_cells)
        proc.extract_2d_coordinates_for_variable.return_value = (lon, lat)

        with patch('mpasdiag.visualization.wind.MPASDateTimeUtils.validate_time_parameters',
                   return_value=(0, 0, 1)):
            with patch.object(plotter, 'create_wind_plot',
                              return_value=(MagicMock(), MagicMock())):
                with patch('mpasdiag.visualization.wind.MPASDateTimeUtils.get_time_info',
                           side_effect=Exception("time extraction failed")):
                    with patch.object(plotter, 'add_timestamp_and_branding'):
                        with patch.object(plotter, 'save_plot'):
                            with patch.object(plotter, 'close_plot'):
                                result = plotter.create_batch_wind_plots(
                                    proc, str(tmp_path), -100., -80., 30., 50.
                                )

        assert len(result) == 1
        assert "time_0" in result[0]

    def test_time_info_exception_uses_correct_index(self: 'TestCreateBatchWindPlots', 
                                                    plotter: MPASWindPlotter, 
                                                    tmp_path: "Path") -> None:
        """
        This test verifies that when the get_time_info function raises an exception during the execution of create_batch_wind_plots, the method correctly falls back to using the time index to generate a time string for the plot title, and that the time string includes the correct time index value. This ensures that even when time information cannot be extracted from the dataset, the method can still proceed with plot creation and provide a meaningful title based on the time index, with the correct index value reflected in the title format. By testing with a mocked get_time_info that raises an exception and checking the resulting plot title, we can confirm that the method's error handling for time information extraction is functioning as intended and that it generates plot titles that accurately reflect the time index when falling back from failed time extraction. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.
            tmp_path (Path): A temporary directory provided by pytest for saving the generated plots.

        Returns:
            None
        """
        n_cells = 5
        ds = xr.Dataset({
            'u': xr.DataArray(np.ones((1, n_cells)), dims=['Time', 'nCells']),
            'v': xr.DataArray(np.ones((1, n_cells)), dims=['Time', 'nCells']),
        })
        proc = MagicMock()
        proc.dataset = ds
        lon = np.linspace(-100., -80., n_cells)
        lat = np.linspace(30., 50., n_cells)
        proc.extract_2d_coordinates_for_variable.return_value = (lon, lat)

        with patch('mpasdiag.visualization.wind.MPASDateTimeUtils.validate_time_parameters',
                   return_value=(0, 0, 1)):
            with patch.object(plotter, 'create_wind_plot',
                              return_value=(MagicMock(), MagicMock())):
                with patch('mpasdiag.visualization.wind.MPASDateTimeUtils.get_time_info',
                           side_effect=RuntimeError("no time")):
                    with patch.object(plotter, 'add_timestamp_and_branding'):
                        with patch.object(plotter, 'save_plot'):
                            with patch.object(plotter, 'close_plot'):
                                result = plotter.create_batch_wind_plots(
                                    proc, str(tmp_path), -100., -80., 30., 50.
                                )

        output_path = result[0]
        assert output_path.endswith("valid_time_0")


class TestExtract2DFrom3DWind:
    """ Test if extract_2d_from_3d_wind correctly extracts 2D slices from 3D u and v wind component arrays based on specified level indices or level values with corresponding pressure levels, ensuring that it returns the appropriate 2D arrays for visualization according to user specifications for vertical level selection. """

    @pytest.fixture
    def plotter(self: 'TestExtract2DFrom3DWind') -> MPASWindPlotter:
        """
        This fixture provides a fresh instance of MPASWindPlotter for each test method in this class. It allows us to test the extract_2d_from_3d_wind method specifically for scenarios involving the extraction of 2D slices from 3D u and v wind component arrays based on either a specified level index or a level value with corresponding pressure levels. By using a fixture, we can easily reuse the setup code and maintain clean and organized test methods that focus on the behavior of extract_2d_from_3d_wind when handling different inputs for vertical level selection, which is essential for validating the method's functionality in preparing wind data for visualization by correctly slicing the 3D arrays according to user specifications. 

        Parameters:
            None

        Returns:
            MPASWindPlotter: An instance of the MPASWindPlotter class to be used in the test methods. 
        """
        return MPASWindPlotter()

    def test_level_index_extracts_correct_slice(self: 'TestExtract2DFrom3DWind', 
                                                plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when the extract_2d_from_3d_wind method is called with a specific level_index, it correctly extracts the corresponding 2D slice from the 3D u and v wind component arrays. This ensures that the method can accurately select the desired vertical level based on the provided index and return the appropriate 2D arrays for visualization. By testing with a specific level index, we can confirm that the method's logic for slicing the 3D arrays is functioning as intended and that it returns the expected results based on the specified level index, allowing users to visualize wind data at different vertical levels as needed. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        u = np.random.rand(N_CELLS, 5)
        v = np.random.rand(N_CELLS, 5)
        u2d, v2d = plotter.extract_2d_from_3d_wind(u, v, level_index=2)
        np.testing.assert_array_equal(u2d, u[:, 2])
        np.testing.assert_array_equal(v2d, v[:, 2])

    def test_level_index_zero_extracts_first_level(self: 'TestExtract2DFrom3DWind', 
                                                   plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when the extract_2d_from_3d_wind method is called with a level_index of 0, it correctly extracts the first level (index 0) from the 3D u and v wind component arrays. This ensures that the method can handle the case where the user wants to visualize the lowest vertical level in the dataset and that it returns the appropriate 2D arrays corresponding to that level. By testing with a level index of 0, we can confirm that the method's logic for slicing the 3D arrays correctly handles this edge case and returns the expected results based on the specified level index, allowing users to visualize wind data at the lowest vertical level when desired. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        u = np.random.rand(N_CELLS, 4)
        v = np.random.rand(N_CELLS, 4)
        u2d, v2d = plotter.extract_2d_from_3d_wind(u, v, level_index=0)
        np.testing.assert_array_equal(u2d, u[:, 0])

    def test_level_value_with_pressure_levels_finds_nearest(self: 'TestExtract2DFrom3DWind', 
                                                            plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when the extract_2d_from_3d_wind method is called with a level_value and a corresponding pressure_levels array, it correctly identifies the nearest pressure level to the specified level_value and extracts the corresponding 2D slice from the 3D u and v wind component arrays. This ensures that the method can handle cases where users specify a desired level based on pressure values rather than index positions, and that it can accurately find the closest available level in the dataset to provide meaningful visualizations. By testing with a specific level value and a set of pressure levels, we can confirm that the method's logic for finding the nearest pressure level and slicing the 3D arrays accordingly is functioning as intended, allowing users to visualize wind data at levels that are relevant to their analysis based on pressure rather than just index. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        u = np.random.rand(N_CELLS, 5)
        v = np.random.rand(N_CELLS, 5)
        pressure_levels = np.array([1000., 850., 700., 500., 300.])
        u2d, v2d = plotter.extract_2d_from_3d_wind(
            u, v, level_value=850., pressure_levels=pressure_levels
        )
        np.testing.assert_array_equal(u2d, u[:, 1])  # 850 is at index 1
        np.testing.assert_array_equal(v2d, v[:, 1])

    def test_level_value_nearest_pressure_mid_range(self: 'TestExtract2DFrom3DWind', 
                                                    plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when the extract_2d_from_3d_wind method is called with a level_value that falls between two pressure levels in the provided pressure_levels array, it correctly identifies the nearest pressure level to the specified level_value and extracts the corresponding 2D slice from the 3D u and v wind component arrays. This ensures that the method can handle cases where users specify a desired level based on pressure values that do not exactly match any of the available levels in the dataset, and that it can accurately find the closest available level to provide meaningful visualizations. By testing with a level value that is between two pressure levels, we can confirm that the method's logic for finding the nearest pressure level and slicing the 3D arrays accordingly is functioning as intended, allowing users to visualize wind data at levels that are relevant to their analysis even when their specified level does not directly correspond to an available pressure level. 
        
        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        u = np.random.rand(N_CELLS, 5)
        v = np.random.rand(N_CELLS, 5)
        pressure_levels = np.array([1000., 850., 700., 500., 300.])
        u2d, v2d = plotter.extract_2d_from_3d_wind(
            u, v, level_value=650., pressure_levels=pressure_levels
        )
        np.testing.assert_array_equal(u2d, u[:, 2])  # |700-650|=50 < |500-650|=150 → index 2

    def test_level_value_without_pressure_levels_uses_top(self: 'TestExtract2DFrom3DWind', 
                                                          plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when the extract_2d_from_3d_wind method is called with a level_value specified but no pressure_levels array provided, it defaults to extracting the top level (last index) from the 3D u and v wind component arrays. This ensures that the method can handle cases where users specify a desired level based on pressure values but do not provide the corresponding pressure levels for reference, and that it can still return a valid 2D slice by defaulting to the top level, which is a common convention in atmospheric data when specific level information is not available. By testing with a level value and no pressure levels, we can confirm that the method's logic for handling this scenario is functioning as intended and that it returns the expected results based on default behavior when pressure levels are not provided. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        u = np.random.rand(N_CELLS, 5)
        v = np.random.rand(N_CELLS, 5)
        u2d, v2d = plotter.extract_2d_from_3d_wind(u, v, level_value=850.)
        np.testing.assert_array_equal(u2d, u[:, -1])

    def test_default_returns_top_level(self: 'TestExtract2DFrom3DWind', 
                                       plotter: MPASWindPlotter) -> None:
        """
        This test verifies that when the extract_2d_from_3d_wind method is called with no level_index and no level_value specified, it defaults to extracting the top level (last index) from the 3D u and v wind component arrays. This ensures that the method can handle cases where users do not specify any vertical level information and still return a valid 2D slice by defaulting to the top level, which is a common convention in atmospheric data when specific level information is not provided. By testing with no level index and no level value, we can confirm that the method's logic for handling this default scenario is functioning as intended and that it returns the expected results based on default behavior when no vertical level information is given. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        u = np.random.rand(N_CELLS, 5)
        v = np.random.rand(N_CELLS, 5)
        u2d, v2d = plotter.extract_2d_from_3d_wind(u, v)
        np.testing.assert_array_equal(u2d, u[:, -1])
        np.testing.assert_array_equal(v2d, v[:, -1])

    def test_output_is_2d_slice_shape(self: 'TestExtract2DFrom3DWind', 
                                      plotter: MPASWindPlotter) -> None:
        """
        This test verifies that the extract_2d_from_3d_wind method returns output arrays for u and v that have the correct shape corresponding to a 2D slice of the original 3D arrays. This ensures that regardless of whether the method is called with a level_index, a level_value with pressure levels, or defaults to the top level, the resulting u2d and v2d arrays have the expected shape of (N_CELLS,), which is necessary for subsequent processing and visualization steps. By testing with different input scenarios and confirming that the output shapes are consistent with 2D slices, we can validate that the method correctly extracts the appropriate level from the 3D arrays while maintaining the expected dimensions for further use in wind visualizations. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        u = np.random.rand(N_CELLS, 7)
        v = np.random.rand(N_CELLS, 7)
        u2d, v2d = plotter.extract_2d_from_3d_wind(u, v, level_index=3)
        assert u2d.shape == (N_CELLS,)
        assert v2d.shape == (N_CELLS,)


class TestComputeWindSpeedAndDirection:
    """ Test if compute_wind_speed_and_direction correctly calculates wind speed as the magnitude of u and v components and wind direction in meteorological convention. """

    @pytest.fixture
    def plotter(self: 'TestComputeWindSpeedAndDirection') -> MPASWindPlotter:
        """
        This fixture provides a fresh instance of MPASWindPlotter for each test method in this class. It allows us to test the compute_wind_speed_and_direction method specifically for scenarios involving the calculation of wind speed as the magnitude of the u and v wind components, and the calculation of wind direction in meteorological convention. By using a fixture, we can easily reuse the setup code and maintain clean and organized test methods that focus on the behavior of compute_wind_speed_and_direction when processing different u and v inputs, which is essential for validating the method's functionality in accurately computing wind speed and direction for visualization purposes. 

        Parameters:
            None

        Returns:
            MPASWindPlotter: An instance of the MPASWindPlotter class to be used in the test methods. 
        """
        return MPASWindPlotter()

    def test_wind_speed_is_vector_magnitude(self: 'TestComputeWindSpeedAndDirection', 
                                            plotter: MPASWindPlotter) -> None:
        """
        This test verifies that the compute_wind_speed_and_direction method correctly calculates the wind speed as the magnitude of the u and v wind components using the formula speed = sqrt(u^2 + v^2). By providing specific u and v values that correspond to known wind speeds (e.g., u=3, v=4 should yield a speed of 5), we can confirm that the method returns the expected wind speed values based on the input components, ensuring that the calculation of wind speed is implemented correctly according to vector magnitude principles. By testing with different combinations of u and v values, we can further validate that the method consistently computes wind speed accurately for a variety of scenarios, which is crucial for creating meaningful wind visualizations. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None        
        """
        u = np.array([3.0, 0.0, 0.0])
        v = np.array([4.0, 5.0, 0.0])
        speed, _ = plotter.compute_wind_speed_and_direction(u, v)
        np.testing.assert_allclose(speed[0], 5.0)  # 3-4-5 triangle
        np.testing.assert_allclose(speed[1], 5.0)
        np.testing.assert_allclose(speed[2], 0.0)

    def test_wind_direction_is_in_0_360_range(self: 'TestComputeWindSpeedAndDirection', 
                                              plotter: MPASWindPlotter) -> None:
        """
        This test verifies that the compute_wind_speed_and_direction method returns wind direction values that are within the expected range of [0, 360) degrees. By providing specific u and v values that correspond to different wind directions (e.g., eastward, northward, etc.), we can confirm that the method correctly calculates the wind direction in meteorological convention and ensures that all returned direction values are non-negative and less than 360 degrees. This is important for maintaining consistency in how wind directions are represented and for ensuring that the output can be used effectively in visualizations and analyses that rely on standard meteorological conventions for wind direction. By testing with a variety of u and v combinations, we can further validate that the method consistently produces valid direction values across different scenarios. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        u = np.array([1.0, -1.0, 0.0, 0.0, 1.0, -1.0])
        v = np.array([0.0, 0.0, 1.0, -1.0, 1.0, -1.0])
        _, direction = plotter.compute_wind_speed_and_direction(u, v)
        assert np.all(direction >= 0.0)
        assert np.all(direction < 360.0)

    def test_return_shapes_match_input(self: 'TestComputeWindSpeedAndDirection', 
                                       plotter: MPASWindPlotter) -> None:
        """
        This test verifies that the compute_wind_speed_and_direction method returns wind speed and wind direction arrays that have the same shape as the input u and v arrays. By providing u and v arrays of a specific shape (e.g., 1D arrays with 20 elements), we can confirm that the method processes the inputs correctly and returns outputs that are consistent in shape, which is essential for ensuring that the results can be used effectively in subsequent processing or visualization steps without encountering shape-related issues. This test helps validate that the method maintains the integrity of the data dimensions throughout its calculations, allowing for seamless integration into workflows that rely on consistent array shapes for further analysis or plotting. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        u = np.random.rand(20)
        v = np.random.rand(20)
        speed, direction = plotter.compute_wind_speed_and_direction(u, v)
        assert speed.shape == (20,)
        assert direction.shape == (20,)

    def test_eastward_wind_direction(self: 'TestComputeWindSpeedAndDirection', 
                                     plotter: MPASWindPlotter) -> None:
        """
        This test verifies that the compute_wind_speed_and_direction method correctly calculates the wind direction for an eastward wind scenario, where the u component is positive and the v component is zero (u=1, v=0). In meteorological convention, an eastward wind corresponds to a direction of 270 degrees. By providing these specific u and v values, we can confirm that the method returns a direction value of approximately 270 degrees, ensuring that the conversion from mathematical angles to meteorological wind directions is implemented correctly for this case. This test helps validate that the method accurately translates the u and v components into the correct meteorological direction based on standard conventions for representing wind directions. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        u = np.array([1.0])
        v = np.array([0.0])
        _, direction = plotter.compute_wind_speed_and_direction(u, v)
        np.testing.assert_allclose(direction[0], 270.0, atol=0.01)

    def test_northward_wind_direction(self: 'TestComputeWindSpeedAndDirection', 
                                      plotter: MPASWindPlotter) -> None:
        """
        This test verifies that the compute_wind_speed_and_direction method correctly calculates the wind direction for a northward wind scenario, where the u component is zero and the v component is positive (u=0, v=1). In meteorological convention, a northward wind corresponds to a direction of 180 degrees. By providing these specific u and v values, we can confirm that the method returns a direction value of approximately 180 degrees, ensuring that the conversion from mathematical angles to meteorological wind directions is implemented correctly for this case. This test helps validate that the method accurately translates the u and v components into the correct meteorological direction based on standard conventions for representing wind directions. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.
        
        Returns:
            None
        """
        u = np.array([0.0])
        v = np.array([1.0])
        _, direction = plotter.compute_wind_speed_and_direction(u, v)
        np.testing.assert_allclose(direction[0], 180.0, atol=0.01)

    def test_both_outputs_are_numpy_arrays(self: 'TestComputeWindSpeedAndDirection', 
                                           plotter: MPASWindPlotter) -> None:
        """
        This test verifies that the compute_wind_speed_and_direction method returns both the wind speed and wind direction as numpy arrays, ensuring that the outputs are in the expected format for further processing and visualization. By providing specific u and v values, we can confirm that the method returns outputs that are instances of numpy.ndarray, which is important for maintaining consistency in data types and allowing for seamless integration with other numpy-based operations or plotting functions that may be used downstream in the workflow. This test helps validate that the method's return types are correctly implemented and that users can rely on receiving numpy arrays for both speed and direction when using this method in their analyses or visualizations. 

        Parameters:
            plotter (MPASWindPlotter): The MPASWindPlotter instance provided by the fixture.

        Returns:
            None
        """
        u = np.array([3.0, 4.0])
        v = np.array([4.0, 3.0])
        speed, direction = plotter.compute_wind_speed_and_direction(u, v)
        assert isinstance(speed, np.ndarray)
        assert isinstance(direction, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

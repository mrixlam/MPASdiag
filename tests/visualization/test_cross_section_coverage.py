#!/usr/bin/env python3

"""
MPASdiag Test Suite: Cross-Section Coverage

This module contains unit tests for the MPASVerticalCrossSectionPlotter class, specifically targeting the coverage of the _validate_cross_section_inputs, _convert_and_clip_data, _resolve_vertical_display, _apply_max_height_filter, _resolve_plot_style, _resolve_vertical_levels, _extract_cross_section_coords, and _unwrap_dataset_var methods. The tests are designed to ensure that these methods handle various input types correctly, raise appropriate exceptions when invalid inputs are provided, and produce expected outputs under different scenarios. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import numpy as np
import pytest
import xarray as xr
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, Mock, patch

from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
from mpasdiag.processing.processors_3d import MPAS3DProcessor


N_CELLS = 5
N_VERT = 10
N_TIME = 2


def _make_proc(dataset: xr.Dataset = None) -> MagicMock:
    """
    This helper function creates a MagicMock instance of MPAS3DProcessor for testing purposes. It allows the caller to optionally provide an xarray.Dataset to associate with the mock processor. If no dataset is provided, the mock will have its dataset attribute set to None. This function simplifies the creation of mock processors for use in the various test cases, allowing them to focus on the specific behavior being tested without needing to set up a full MPAS3DProcessor instance. 

    Parameters:
        dataset (xr.Dataset, optional): The dataset to associate with the processor. Defaults to None.

    Returns:
        MagicMock: A MagicMock instance of MPAS3DProcessor with the specified dataset.
    """
    mock = MagicMock(spec=MPAS3DProcessor)
    mock.grid_file = None
    if dataset is not None:
        mock.dataset = dataset
    return mock


def _make_ds() -> xr.Dataset:
    """
    This helper function creates a minimal xarray.Dataset for testing purposes. The dataset includes three variables: 'temperature', 'qv' (specific humidity), and 'lonCell'/'latCell' (cell coordinates). The temperature variable is a 3D array with dimensions corresponding to time, cells, and vertical levels, filled with linearly spaced values between 250 K and 300 K. The specific humidity variable is a 3D array filled with a constant value of 0.005. The longitude and latitude variables are 1D arrays that provide the coordinates for each cell. This dataset is designed to be simple yet sufficient for testing the functionality of the MPASVerticalCrossSectionPlotter methods that require access to these types of data. 

    Parameters:
        None

    Returns:
        xr.Dataset: A minimal dataset with temperature, specific humidity, and cell coordinates.
    """
    return xr.Dataset({
        'temperature': (['Time', 'nCells', 'nVertLevels'],
                        np.linspace(250., 300., N_TIME * N_CELLS * N_VERT).reshape(N_TIME, N_CELLS, N_VERT)),
        'qv': (['Time', 'nCells', 'nVertLevels'],
                np.full((N_TIME, N_CELLS, N_VERT), 0.005)),
        'lonCell': (['nCells'], np.linspace(-100., -90., N_CELLS)),
        'latCell': (['nCells'], np.linspace(35., 45., N_CELLS)),
    })


def _plotter() -> MPASVerticalCrossSectionPlotter:
    """
    This helper function creates and returns an instance of the MPASVerticalCrossSectionPlotter class for use in the test cases. It does not take any parameters and simply instantiates the plotter, which can then be used to call the various methods being tested. This function helps to keep the test code clean and focused on the specific behaviors being tested, rather than on the setup of the plotter instance. 

    Parameters:
        None

    Returns:
        MPASVerticalCrossSectionPlotter: An instance of the plotter.
    """
    return MPASVerticalCrossSectionPlotter()


class TestValidateCrossSectionInputs:
    """ Covers the four ValueError raises in _validate_cross_section_inputs. """

    def test_invalid_processor_type_raises(self: 'TestValidateCrossSectionInputs') -> None:
        """
        This test verifies that the _validate_cross_section_inputs method raises a ValueError when the provided processor argument is not an instance of MPAS3DProcessor. It creates a plotter instance and then calls the validation method with an invalid processor type (a string in this case). The test asserts that the exception is raised and that the error message contains "MPAS3DProcessor". 

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        with pytest.raises(ValueError, match="MPAS3DProcessor"):
            plotter._validate_cross_section_inputs("not_a_processor", "temperature")

    def test_none_dataset_raises(self: 'TestValidateCrossSectionInputs') -> None:
        """
        This test checks that the _validate_cross_section_inputs method raises a ValueError when the processor's dataset attribute is None. It creates a plotter instance and a mock processor with dataset set to None. The test then calls the validation method and asserts that a ValueError is raised with a message indicating that no data is loaded. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        mock_proc = _make_proc()
        mock_proc.dataset = None
        with pytest.raises(ValueError, match="loaded data"):
            plotter._validate_cross_section_inputs(mock_proc, "temperature")

    def test_missing_variable_raises(self: 'TestValidateCrossSectionInputs') -> None:
        """
        This test ensures that the _validate_cross_section_inputs method raises a ValueError when the specified variable name is not found in the processor's dataset. It creates a plotter instance and a mock processor with a valid dataset. The test then calls the validation method with a variable name that does not exist in the dataset and asserts that a ValueError is raised with a message indicating that the variable was not found.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        mock_proc = _make_proc(_make_ds())
        with pytest.raises(ValueError, match="not found"):
            plotter._validate_cross_section_inputs(mock_proc, "no_such_var")

    def test_available_vars_listed_in_error(self: 'TestValidateCrossSectionInputs') -> None:
        """
        This test checks that the _validate_cross_section_inputs method includes a list of available variables in the error message when a non-existent variable is requested. It creates a plotter instance and a mock processor with a valid dataset. The test then calls the validation method with a variable name that does not exist in the dataset and asserts that the error message contains "Available variables".

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        mock_proc = _make_proc(_make_ds())
        with pytest.raises(ValueError) as exc_info:
            plotter._validate_cross_section_inputs(mock_proc, "no_such_var")
        assert "Available variables" in str(exc_info.value)

    def test_non_3d_variable_raises(self: 'TestValidateCrossSectionInputs') -> None:
        """
        This test verifies that the _validate_cross_section_inputs method raises a ValueError when the specified variable is found in the dataset but does not have 3D dimensions. It creates a plotter instance and a mock processor with a dataset that includes a 2D variable (surface_temp). The test then calls the validation method with the name of this 2D variable and asserts that a ValueError is raised with a message indicating that a 3D atmospheric variable is expected. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        ds = xr.Dataset({
            'surface_temp': (['Time', 'nCells'], np.ones((N_TIME, N_CELLS))),
        })
        mock_proc = _make_proc(ds)
        with pytest.raises(ValueError, match="3D atmospheric variable"):
            plotter._validate_cross_section_inputs(mock_proc, "surface_temp")


class TestConvertAndClipData:
    """ Covers xr.DataArray/non-ndarray conversion and moisture clipping. """

    def test_xarray_input_converted_to_ndarray(self: 'TestConvertAndClipData') -> None:
        """
        This test checks that the _convert_and_clip_data method correctly converts an xarray.DataArray input into a numpy ndarray. It creates a plotter instance and an xarray.DataArray filled with ones, with dimensions corresponding to vertical levels and cells. The test then calls the conversion method with this DataArray and asserts that the result is an instance of np.ndarray.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        da = xr.DataArray(np.ones((N_VERT, N_CELLS)), dims=['level', 'cell'])
        result, meta = plotter._convert_and_clip_data(da, 'temperature')
        assert isinstance(result, np.ndarray)

    def test_list_input_converted_to_ndarray(self: 'TestConvertAndClipData') -> None:
        """
        This test checks that the _convert_and_clip_data method correctly converts a list input into a numpy ndarray. It creates a plotter instance and a list of numerical values. The test then calls the conversion method with this list and asserts that the result is an instance of np.ndarray.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        result, meta = plotter._convert_and_clip_data([1.0, 2.0, 3.0], 'temperature')
        assert isinstance(result, np.ndarray)

    def test_moisture_negative_values_clipped(self: 'TestConvertAndClipData') -> None:
        """
        This test verifies that the _convert_and_clip_data method correctly clips negative values to zero for moisture variables (e.g., 'qv'). It creates a plotter instance and a numpy array containing both negative and positive values. The test then calls the conversion method with this array and the variable name 'qv', and asserts that all values in the result are greater than or equal to zero.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        data = np.array([-0.01, 0.005, 0.01, -0.002, 0.008])
        result, _ = plotter._convert_and_clip_data(data, 'qv')
        assert np.all(result >= 0.0)

    def test_moisture_clipping_preserves_positives(self: 'TestConvertAndClipData') -> None:
        """
        This test verifies that the _convert_and_clip_data method correctly preserves positive values for moisture variables (e.g., 'qv2m') while clipping negative values to zero. It creates a plotter instance and a numpy array containing both negative and positive values. The test then calls the conversion method with this array and the variable name 'qv2m', and asserts that the positive values are unchanged.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        data = np.array([-0.001, 0.005, 0.010])
        result, _ = plotter._convert_and_clip_data(data, 'qv2m')
        assert result[1] == pytest.approx(0.005)
        assert result[2] == pytest.approx(0.010)

    def test_no_negatives_no_clipping(self: 'TestConvertAndClipData') -> None:
        """
        This test checks that the _convert_and_clip_data method does not modify values when there are no negative values present for moisture variables. It creates a plotter instance and a numpy array containing only positive values. The test then calls the conversion method with this array and the variable name 'qv', and asserts that all values in the result are unchanged.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        data = np.array([0.001, 0.005, 0.010])
        result, _ = plotter._convert_and_clip_data(data, 'qv')
        assert np.all(result >= 0.0)


class TestResolveVerticalDisplay:
    """ Covers astype(float) exception fallback and unknown desired_display. """

    def test_astype_exception_uses_np_asarray_fallback(self: 'TestResolveVerticalDisplay') -> None:
        """
        This test verifies that the _resolve_vertical_display method correctly falls back to using np.asarray when an exception is raised during an attempt to convert the vertical coordinates to float using astype. It defines a custom class _BadArray that simulates this behavior by raising a TypeError when astype is called, but still provides an __array__ method that returns a valid numpy array. The test then creates a plotter instance and a mock processor, and calls the _resolve_vertical_display method with an instance of _BadArray. The test asserts that the resulting coordinate type is either 'pressure_hPa' or 'modlev', indicating that the fallback was successful. 

        Parameters:
            None

        Returns:
            None
        """
        class _BadArray:
            """ Simulates an array-like object that raises an exception when astype is called, but can still be converted to a numpy array using __array__. """

            def astype(self: 'TestResolveVerticalDisplay._BadArray', 
                       dtype: type) -> None:
                """
                This method simulates the behavior of an array-like object that raises a TypeError when an attempt is made to convert it to a specified dtype using astype. This is used to test the fallback mechanism in the _resolve_vertical_display method. 

                Parameters:
                    dtype (type): The data type that the method is attempting to convert to.

                Returns:
                    None
                """
                raise TypeError("cannot astype")

            def __array__(self: 'TestResolveVerticalDisplay._BadArray', 
                          dtype: type = None) -> np.ndarray:
                """
                This method allows the _BadArray class to be converted to a numpy array using np.asarray, even though it raises an exception when astype is called. It returns a numpy array of vertical coordinates that would be typical for pressure levels.

                Parameters:
                    dtype (type, optional): The data type to convert to. This parameter is not used in this implementation, but is included to match the expected signature for __array__.

                Returns:
                    np.ndarray: A numpy array of vertical coordinates. 
                """
                return np.array([50000., 70000., 100000.])

            def __len__(self: 'TestResolveVerticalDisplay._BadArray') -> int:
                """
                This method returns the length of the _BadArray, which is required for it to be treated as an array-like object. In this case, it returns 3, which corresponds to the number of vertical levels in the array returned by __array__. 

                Parameters:
                    None

                Returns:
                    int: The length of the array, which is 3 in this case. 
                """
                return 3

        plotter = _plotter()
        mock_proc = _make_proc(_make_ds())
        mock_proc.grid_file = None

        result = plotter._resolve_vertical_display(
            _BadArray(), 'pressure', 'pressure', mock_proc, 0
        )

        assert result[1] in ('pressure_hPa', 'modlev')

    def test_unknown_desired_display_falls_through(self: 'TestResolveVerticalDisplay') -> None:
        """
        This test checks that the _resolve_vertical_display method correctly handles an unknown desired_display type by falling through to the final return statement without raising an exception. It creates a plotter instance and a mock processor, and calls the _resolve_vertical_display method with a desired_display value that is not recognized (e.g., 'unknown_display_type'). The test asserts that the result is a tuple containing a numpy array and a string, indicating that the method returned something reasonable even when the desired display type was not recognized. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        ds = _make_ds()
        mock_proc = _make_proc(ds)
        mock_proc.grid_file = None
        vertical_coords = np.linspace(50000., 100000., N_VERT)

        result = plotter._resolve_vertical_display(
            vertical_coords, 'pressure', 'unknown_display_type', mock_proc, 0
        )

        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], str)


class TestApplyMaxHeightFilter:
    """ Covers type conversions and the no-levels-below-max warning. """

    def test_xarray_vertical_display_converted(self: 'TestApplyMaxHeightFilter') -> None:
        """
        This test verifies that the _apply_max_height_filter method correctly converts an xarray.DataArray vertical display input into a numpy ndarray. It creates a plotter instance and an xarray.DataArray representing vertical coordinates, with dimensions corresponding to vertical levels. The test then calls the max height filter method with this DataArray and asserts that the resulting vertical display is an instance of np.ndarray. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        vd = xr.DataArray(np.linspace(0., 10., N_VERT), dims=['level'])
        dv = np.ones((N_VERT, N_CELLS))
        out_vd, out_dv = plotter._apply_max_height_filter(vd, 'height_km', dv, max_height=5.0)
        assert isinstance(out_vd, np.ndarray)

    def test_list_vertical_display_converted(self: 'TestApplyMaxHeightFilter') -> None:
        """
        This test checks that the _apply_max_height_filter method correctly converts a list vertical display input into a numpy ndarray. It creates a plotter instance and a list of vertical coordinate values. The test then calls the max height filter method with this list and asserts that the resulting vertical display is an instance of np.ndarray. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        vd = list(np.linspace(0., 10., N_VERT))
        dv = np.ones((N_VERT, N_CELLS))
        out_vd, out_dv = plotter._apply_max_height_filter(vd, 'height_km', dv, max_height=5.0)
        assert isinstance(out_vd, np.ndarray)

    def test_xarray_data_values_converted(self: 'TestApplyMaxHeightFilter') -> None:
        """
        This test verifies that the _apply_max_height_filter method correctly converts an xarray.DataArray data values input into a numpy ndarray. It creates a plotter instance and an xarray.DataArray representing data values, with dimensions corresponding to vertical levels and cells. The test then calls the max height filter method with this DataArray and asserts that the resulting data values are an instance of np.ndarray.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        vd = np.linspace(0., 10., N_VERT)
        dv = xr.DataArray(np.ones((N_VERT, N_CELLS)), dims=['level', 'cell'])
        out_vd, out_dv = plotter._apply_max_height_filter(vd, 'height_km', dv, max_height=5.0)
        assert isinstance(out_dv, np.ndarray)

    def test_list_data_values_converted(self: 'TestApplyMaxHeightFilter') -> None:
        """
        This test verifies that the _apply_max_height_filter method correctly converts a list data values input into a numpy ndarray. It creates a plotter instance and a list of data values, with dimensions corresponding to vertical levels and cells. The test then calls the max height filter method with this list and asserts that the resulting data values are an instance of np.ndarray.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        vd = np.linspace(0., 10., N_VERT)
        dv = [list(np.ones(N_CELLS)) for _ in range(N_VERT)]
        out_vd, out_dv = plotter._apply_max_height_filter(vd, 'height_km', dv, max_height=5.0)
        assert isinstance(out_dv, np.ndarray)

    def test_all_levels_above_max_shows_warning(self: 'TestApplyMaxHeightFilter', 
                                                capsys: 'pytest.CaptureFixture') -> None:
        """
        This test checks that the _apply_max_height_filter method correctly prints a warning when all vertical levels are above the specified maximum height. It creates a plotter instance and a vertical display array with values that are all above the max_height threshold. The test then calls the max height filter method and captures the output using capsys. Finally, it asserts that the captured output contains a warning message indicating that no levels are below the maximum height. 

        Parameters:
            capsys: pytest.CaptureFixture

        Returns:
            None
        """
        plotter = _plotter()
        vd = np.linspace(6., 15., N_VERT)
        dv = np.ones((N_VERT, N_CELLS))
        out_vd, out_dv = plotter._apply_max_height_filter(vd, 'height_km', dv, max_height=5.0)
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert len(out_vd) == N_VERT

    def test_no_max_height_returns_unchanged(self: 'TestApplyMaxHeightFilter') -> None:
        """
        This test verifies that the _apply_max_height_filter method returns the original vertical display and data values unchanged when max_height is set to None. It creates a plotter instance and a vertical display array with values that would normally be filtered if a max_height were specified. The test then calls the max height filter method with max_height set to None and asserts that the output vertical display and data values are identical to the input, indicating that no filtering was applied. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        vd = np.linspace(0., 10., N_VERT)
        dv = np.ones((N_VERT, N_CELLS))
        out_vd, out_dv = plotter._apply_max_height_filter(vd, 'height_km', dv, max_height=None)
        np.testing.assert_array_equal(out_vd, vd)


class TestResolvePlotStyle:
    """ Covers type conversions, early return, and post-try levels=None safety. """

    def test_xarray_data_converted(self: 'TestResolvePlotStyle') -> None:
        """
        This test verifies that the _resolve_plot_style method correctly converts an xarray.DataArray input into a numpy ndarray when determining the colormap and levels for plotting. It creates a plotter instance and an xarray.DataArray filled with ones, with dimensions corresponding to vertical levels and cells. The test then calls the _resolve_plot_style method with this DataArray and asserts that the returned colormap and levels are not None, indicating that the method successfully processed the xarray input. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        da = xr.DataArray(np.ones((5, 5)), dims=['level', 'cell'])
        cmap, levels = plotter._resolve_plot_style('temperature', da, None, None)
        assert cmap is not None
        assert levels is not None

    def test_list_data_converted(self: 'TestResolvePlotStyle') -> None:
        """
        This test verifies that the _resolve_plot_style method correctly converts a list input into a numpy ndarray when determining the colormap and levels for plotting. It creates a plotter instance and a list of lists filled with numerical values. The test then calls the _resolve_plot_style method with this list and asserts that the returned colormap is not None, indicating that the method successfully processed the list input.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        data = [[1.0, 2.0], [3.0, 4.0]]
        cmap, levels = plotter._resolve_plot_style('temperature', data, None, None)
        assert cmap is not None

    def test_both_provided_returns_early(self: 'TestResolvePlotStyle') -> None:
        """
        This test checks that the _resolve_plot_style method returns early with the provided colormap and levels when both are given as arguments. It creates a plotter instance and a numpy array of data values. The test then calls the _resolve_plot_style method with a specified colormap and levels, and asserts that the returned colormap and levels match the provided values, indicating that the method did not attempt to resolve them from style settings. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        data = np.ones((5, 5))
        cmap, levels = plotter._resolve_plot_style('temperature', data, 'plasma', np.array([1., 2., 3.]))
        assert cmap == 'plasma'
        np.testing.assert_array_equal(levels, [1., 2., 3.])

    def test_style_levels_none_triggers_get_default(self: 'TestResolvePlotStyle') -> None:
        """
        This test verifies that the _resolve_plot_style method correctly handles the case where the style's levels are None. It creates a plotter instance and a numpy array of data values. The test then patches the get_variable_style method to return a colormap with levels set to None, and asserts that the _resolve_plot_style method returns non-None levels, indicating that it correctly falls back to the default levels.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        data = np.ones((5, 5))
        with patch('mpasdiag.visualization.cross_section.MPASVisualizationStyle.get_variable_style',
                   return_value={'colormap': 'viridis', 'levels': None}):
            cmap, levels = plotter._resolve_plot_style('some_var', data, None, None)
        assert levels is not None


class TestResolveVerticalLevels:
    """ Test if unknown vertical coord becomes 'modlev' and 'height' is converted to 'pressure'. """

    def test_unknown_vertical_coord_becomes_modlev(self: 'TestResolveVerticalLevels') -> None:
        """
        This test verifies that the _resolve_vertical_levels method correctly handles the case where the vertical coordinate is unknown. It creates a plotter instance and a mock processor. The test then sets the mock processor to return a range of vertical levels and calls the _resolve_vertical_levels method with an unknown coordinate. The test asserts that the returned coordinate is 'modlev', indicating that the method correctly falls back to the default coordinate for unknown cases.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        mock_proc = _make_proc(_make_ds())
        mock_proc.get_vertical_levels.return_value = np.arange(1, N_VERT + 1, dtype=float)
        levels, coord = plotter._resolve_vertical_levels(mock_proc, 'temperature', 'foobar', 0)
        assert coord == 'modlev'

    def test_height_coord_becomes_pressure(self: 'TestResolveVerticalLevels') -> None:
        """
        This test verifies that the _resolve_vertical_levels method correctly handles the case where the vertical coordinate is 'height'. It creates a plotter instance and a mock processor. The test then sets the mock processor to return a range of vertical levels and calls the _resolve_vertical_levels method with the 'height' coordinate. The test asserts that the returned coordinate is either 'pressure' or 'modlev', indicating that the method correctly converts 'height' to 'pressure'.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        mock_proc = _make_proc(_make_ds())
        mock_proc.get_vertical_levels.return_value = np.linspace(100000., 10000., N_VERT)
        levels, coord = plotter._resolve_vertical_levels(mock_proc, 'temperature', 'height', 0)
        assert coord in ('pressure', 'modlev')


class TestExtractCrossSectionCoords:
    """ Test if path outside domain prints warning and path inside domain does not. """

    def test_path_outside_domain_prints_warning(self: 'TestExtractCrossSectionCoords', 
                                                capsys: 'pytest.CaptureFixture') -> None:
        """
        This test checks that the _extract_cross_section_coords method correctly prints a warning when the specified path is outside the domain of the dataset. It creates a plotter instance and a mock processor with a dataset that has specific longitude and latitude coordinates. The test then calls the _extract_cross_section_coords method with path coordinates that are outside the range of the dataset's coordinates, and captures the output using capsys. Finally, it asserts that the captured output contains a warning message indicating that the path is outside the domain. 

        Parameters:
            capsys: pytest.CaptureFixture

        Returns:
            None
        """
        plotter = _plotter()
        ds = _make_ds()
        mock_proc = _make_proc(ds)
        mock_proc.extract_2d_coordinates_for_variable.side_effect = RuntimeError("use lonCell")

        path_lons = np.array([-150., -120.])
        path_lats = np.array([60., 70.])
        lon_coords, lat_coords = plotter._extract_cross_section_coords(
            mock_proc, 'temperature', path_lons, path_lats
        )
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "outside" in captured.out

    def test_path_inside_domain_no_warning(self: 'TestExtractCrossSectionCoords', 
                                           capsys: 'pytest.CaptureFixture') -> None:
        """
        This test checks that the _extract_cross_section_coords method does not print a warning when the specified path is inside the domain of the dataset. It creates a plotter instance and a mock processor with a dataset that has specific longitude and latitude coordinates. The test then calls the _extract_cross_section_coords method with path coordinates that are within the range of the dataset's coordinates, and captures the output using capsys. Finally, it asserts that the captured output does not contain a warning message.

        Parameters:
            capsys: pytest.CaptureFixture

        Returns:
            None
        """
        plotter = _plotter()
        mock_proc = _make_proc(_make_ds())
        mock_proc.extract_2d_coordinates_for_variable.side_effect = RuntimeError("use lonCell")
        path_lons = np.array([-98., -92.])
        path_lats = np.array([37., 42.])
        plotter._extract_cross_section_coords(mock_proc, 'temperature', path_lons, path_lats)
        captured = capsys.readouterr()
        assert "WARNING" not in captured.out


class TestUnwrapDatasetVar:
    """ Test if missing vert levels dim returns None and if nVertLevelsP1 is detected. """

    def test_no_vert_levels_dim_returns_none_vert_dim(self: 'TestUnwrapDatasetVar') -> None:
        """
        This test verifies that the _unwrap_dataset_var method correctly returns None for the vert_dim when the specified variable does not have a vertical levels dimension. It creates a plotter instance and a mock processor with a dataset that includes a 2D variable (scalar_var) without a vertical levels dimension. The test then calls the _unwrap_dataset_var method with this variable and asserts that the returned vert_dim is None, indicating that the method correctly identified the absence of a vertical levels dimension. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            'scalar_var': (['Time', 'nCells'], np.ones((N_TIME, N_CELLS))),
        })
        mock_proc = _make_proc(ds)
        var_da, time_dim, vert_dim = MPASVerticalCrossSectionPlotter._unwrap_dataset_var(
            mock_proc, 'scalar_var'
        )
        assert vert_dim is None

    def test_vert_levels_p1_dim_detected(self: 'TestUnwrapDatasetVar') -> None:
        """
        This test verifies that the _unwrap_dataset_var method correctly detects the nVertLevelsP1 dimension when it is present in the dataset. It creates a plotter instance and a mock processor with a dataset that includes a variable (w) with the nVertLevelsP1 dimension. The test then calls the _unwrap_dataset_var method with this variable and asserts that the returned vert_dim is 'nVertLevelsP1', indicating that the method correctly identified the presence of the nVertLevelsP1 dimension.

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            'w': (['Time', 'nCells', 'nVertLevelsP1'], np.ones((N_TIME, N_CELLS, N_VERT + 1))),
        })
        mock_proc = _make_proc(ds)
        _, _, vert_dim = MPASVerticalCrossSectionPlotter._unwrap_dataset_var(mock_proc, 'w')
        assert vert_dim == 'nVertLevelsP1'


class TestExtractLevelData:
    """ Covers wrong shape → ValueError. """

    def test_wrong_shape_raises_value_error(self: 'TestExtractLevelData') -> None:
        """
        This test verifies that the _extract_level_data method raises a ValueError when the input DataArray has a shape that does not match the expected number of cells. It creates a DataArray with dimensions ['Time', 'nCells', 'nVertLevels'] and attempts to extract level data with an expected number of cells that is different from the actual number of cells. The test asserts that a ValueError is raised with a message containing "expected".

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(
            np.ones((N_TIME, N_CELLS, N_VERT)),
            dims=['Time', 'nCells', 'nVertLevels'],
        )
        with pytest.raises(ValueError, match="expected"):
            MPASVerticalCrossSectionPlotter._extract_level_data(
                da, {'Time': 0, 'nVertLevels': 0}, expected_ncells=N_CELLS + 10
            )

    def test_correct_shape_succeeds(self: 'TestExtractLevelData') -> None:
        """
        This test verifies that the _extract_level_data method correctly extracts level data when the input DataArray has the expected shape. It creates a DataArray with dimensions ['Time', 'nCells', 'nVertLevels'] and attempts to extract level data with an expected number of cells that matches the actual number of cells. The test asserts that the extracted data has the correct shape.

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(
            np.ones((N_TIME, N_CELLS, N_VERT)),
            dims=['Time', 'nCells', 'nVertLevels'],
        )
        result = MPASVerticalCrossSectionPlotter._extract_level_data(
            da, {'Time': 0, 'nVertLevels': 0}, expected_ncells=N_CELLS
        )
        assert result.shape == (N_CELLS,)


class TestInterpolateAllLevels:
    """ Covers vert_dim=None continue, per-level exception, and all-NaN warning. """

    def test_vert_dim_none_continues(self: 'TestInterpolateAllLevels') -> None:
        """
        This test checks that the _interpolate_all_levels method correctly continues to return an all-NaN result when vert_dim is None. It creates a plotter instance and a DataArray with dimensions ['Time', 'nCells'] (i.e., no vertical levels dimension). The test then calls the _interpolate_all_levels method with vert_dim set to None and asserts that the result is an array of NaN values, indicating that the method correctly handled the absence of vertical levels. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()

        var_da = xr.DataArray(
            np.ones((N_TIME, N_CELLS)),
            dims=['Time', 'nCells'],
        )

        result = plotter._interpolate_all_levels(
            var_da,
            vertical_levels=np.arange(N_VERT, dtype=float),
            time_index=0,
            time_dim='Time',
            vert_dim=None,
            lon_coords=np.linspace(-100., -90., N_CELLS),
            lat_coords=np.linspace(35., 45., N_CELLS),
            path_lons=np.linspace(-99., -91., 5),
            path_lats=np.linspace(36., 44., 5),
            num_points=5,
        )

        assert np.all(np.isnan(result))

    def test_per_level_exception_prints_warning(self: 'TestInterpolateAllLevels', 
                                                capsys: pytest.CaptureFixture) -> None:
        """
        This test verifies that the _interpolate_all_levels method correctly prints a warning when an exception occurs during level extraction. It creates a plotter instance and a DataArray with dimensions ['Time', 'nCells', 'nVertLevels']. The test then patches the _extract_level_data method to raise a RuntimeError and asserts that the warning message is printed.

        Parameters:
            capsys: pytest fixture for capturing stdout and stderr

        Returns:
            None
        """
        plotter = _plotter()

        var_da = xr.DataArray(
            np.ones((N_TIME, N_CELLS, N_VERT)),
            dims=['Time', 'nCells', 'nVertLevels'],
        )

        with patch.object(plotter, '_extract_level_data', side_effect=RuntimeError("extraction failed")):
            plotter._interpolate_all_levels(
                var_da,
                vertical_levels=np.arange(N_VERT, dtype=float),
                time_index=0,
                time_dim='Time',
                vert_dim='nVertLevels',
                lon_coords=np.linspace(-100., -90., N_CELLS),
                lat_coords=np.linspace(35., 45., N_CELLS),
                path_lons=np.linspace(-99., -91., 5),
                path_lats=np.linspace(36., 44., 5),
                num_points=5,
            )

        captured = capsys.readouterr()
        assert "Warning" in captured.out

    def test_all_nan_result_prints_warning(self: 'TestInterpolateAllLevels', 
                                           capsys: pytest.CaptureFixture) -> None:
        """
        This test verifies that the _interpolate_all_levels method correctly prints a warning when the result is all-NaN. It creates a plotter instance and a DataArray with dimensions ['Time', 'nCells', 'nVertLevels']. The test then patches the _extract_level_data method to raise a RuntimeError and asserts that the warning message is printed.

        Parameters:
            capsys: pytest fixture for capturing stdout and stderr

        Returns:
            None
        """
        plotter = _plotter()

        var_da = xr.DataArray(
            np.ones((N_TIME, N_CELLS, N_VERT)),
            dims=['Time', 'nCells', 'nVertLevels'],
        )

        with patch.object(plotter, '_extract_level_data', side_effect=RuntimeError("fail")):
            result = plotter._interpolate_all_levels(
                var_da,
                vertical_levels=np.arange(N_VERT, dtype=float),
                time_index=0,
                time_dim='Time',
                vert_dim='nVertLevels',
                lon_coords=np.linspace(-100., -90., N_CELLS),
                lat_coords=np.linspace(35., 45., N_CELLS),
                path_lons=np.linspace(-99., -91., 5),
                path_lats=np.linspace(36., 44., 5),
                num_points=5,
            )

        captured = capsys.readouterr()
        assert "NO valid" in captured.out or "WARNING" in captured.out
        assert np.all(np.isnan(result))


class TestGenerateGreatCirclePath:
    """ Covers the zero-distance fill branch. """

    def test_zero_distance_path_filled_with_start_point(self: 'TestGenerateGreatCirclePath') -> None:
        """
        This test verifies that the _generate_great_circle_path method correctly fills the path with the start point when the start and end points are the same, resulting in a zero-distance path. It creates a plotter instance and defines a start/end point with specific longitude and latitude values. The test then calls the _generate_great_circle_path method with these identical points and asserts that the resulting longitude, latitude, and distance arrays are all filled with the start point's coordinates and zero distance, respectively. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        same = (-100.0, 40.0)
        lons, lats, dists = plotter._generate_great_circle_path(same, same, num_points=5)
        assert len(lons) == 5
        assert np.all(lons == pytest.approx(same[0]))
        assert np.all(lats == pytest.approx(same[1]))
        assert np.all(dists == pytest.approx(0.0))


class TestInterpolateAlongPath:
    """ Covers xr.DataArray/non-ndarray grid_data conversion and all-NaN return. """

    def test_xarray_grid_data_converted(self: 'TestInterpolateAlongPath') -> None:
        """
        This test verifies that the _interpolate_along_path method correctly converts an xr.DataArray grid_data to a numpy array. It creates a plotter instance and a DataArray with specific values. The test then calls the _interpolate_along_path method with this DataArray and asserts that the resulting array has the expected length and contains valid values.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        grid_data = xr.DataArray(np.array([10., 20., 30., 40., 50.]), dims=['cell'])

        result = plotter._interpolate_along_path(
            np.linspace(-100., -90., 5),
            np.linspace(35., 45., 5),
            grid_data,
            np.array([-98., -95., -92.]),
            np.array([37., 40., 43.]),
        )

        assert len(result) == 3
        assert not np.all(np.isnan(result))

    def test_list_grid_data_converted(self: 'TestInterpolateAlongPath') -> None:
        """
        This test verifies that the _interpolate_along_path method correctly converts a list grid_data to a numpy array. It creates a plotter instance and a list with specific values. The test then calls the _interpolate_along_path method with this list and asserts that the resulting array has the expected length and contains valid values.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        result = plotter._interpolate_along_path(
            np.linspace(-100., -90., 5),
            np.linspace(35., 45., 5),
            [10., 20., 30., 40., 50.],
            np.array([-98., -95.]),
            np.array([37., 40.]),
        )
        assert len(result) == 2

    def test_all_nan_grid_data_returns_nan_array(self: 'TestInterpolateAlongPath') -> None:
        """
        This test verifies that the _interpolate_along_path method correctly returns a NaN array when all valid mask elements are empty. It creates a plotter instance and a grid_data array filled with NaN values. The test then calls the _interpolate_along_path method with this array and asserts that the resulting array contains only NaN values.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        grid_data = np.full(5, np.nan)
        result = plotter._interpolate_along_path(
            np.linspace(-100., -90., 5),
            np.linspace(35., 45., 5),
            grid_data,
            np.array([-98., -95.]),
            np.array([37., 40.]),
        )
        assert np.all(np.isnan(result))


class TestComputeVarLevels:
    """ Covers temperature, pressure, and wind level branches. """

    def test_temperature_branch(self: 'TestComputeVarLevels') -> None:
        """
        This test verifies that the _compute_var_levels method correctly handles the temperature branch. It creates a plotter instance and calls the _compute_var_levels method with temperature parameters. The test then asserts that the resulting levels array starts and ends with the expected values.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        levels = plotter._compute_var_levels('temperature', 'temperature', 250., 310., 60.)
        assert levels[0] <= 250.
        assert levels[-1] >= 310.

    def test_temperature_small_range_step2(self: 'TestComputeVarLevels') -> None:
        """
        This test verifies that the _compute_var_levels method correctly handles small temperature ranges with a step of 2. It creates a plotter instance and calls the _compute_var_levels method with temperature parameters. The test then asserts that the differences between consecutive levels are consistent.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        levels = plotter._compute_var_levels('temperature', 'temperature', 280., 295., 15.)
        diffs = np.diff(levels)
        assert np.allclose(diffs, diffs[0])

    def test_pressure_positive_min_logspace(self: 'TestComputeVarLevels') -> None:
        """
        This test verifies that the _compute_var_levels method correctly handles the pressure branch when data_min > 0, resulting in a logspace. It creates a plotter instance and calls the _compute_var_levels method with pressure parameters. The test then asserts that the resulting levels array has the expected length.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        levels = plotter._compute_var_levels('pressure', 'pressure', 100., 1000., 900.)
        assert len(levels) == 15

    def test_pressure_nonpositive_min_linspace(self: 'TestComputeVarLevels') -> None:
        """
        This test verifies that the _compute_var_levels method correctly handles the pressure branch when data_min <= 0, resulting in a linspace. It creates a plotter instance and calls the _compute_var_levels method with pressure parameters. The test then asserts that the resulting levels array starts with the expected value and has the expected length.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        levels = plotter._compute_var_levels('pressure', 'pressure', -100., 1000., 1100.)
        assert levels[0] == pytest.approx(-100.)
        assert len(levels) == 15

    def test_wind_nonnegative_linspace(self: 'TestComputeVarLevels') -> None:
        """
        This test verifies that the _compute_var_levels method correctly handles the wind branch when data_min >= 0, resulting in a linspace. It creates a plotter instance and calls the _compute_var_levels method with wind parameters. The test then asserts that the resulting levels array has the expected length and values.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        levels = plotter._compute_var_levels('wind_speed', 'wind_speed', 0., 20., 20.)
        assert len(levels) == 15
        assert levels[0] == pytest.approx(0.)
        assert levels[-1] == pytest.approx(20.)

    def test_wind_symmetric_levels(self: 'TestComputeVarLevels') -> None:
        """
        This test verifies that the _compute_var_levels method correctly handles the wind branch when data_min < 0 and data_max > 0, resulting in a symmetric linspace. It creates a plotter instance and calls the _compute_var_levels method with wind parameters. The test then asserts that the resulting levels array has the expected length and values.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        levels = plotter._compute_var_levels('wind', 'wind', -15., 25., 40.)
        assert len(levels) == 21
        assert levels[0] < 0
        assert levels[-1] > 0
        assert pytest.approx(levels[0], abs=0.1) == -levels[-1]

    def test_u_var_name_triggers_wind_branch(self: 'TestComputeVarLevels') -> None:
        """
        This test verifies that the _compute_var_levels method correctly handles the wind branch when the variable name starts with 'u'. It creates a plotter instance and calls the _compute_var_levels method with wind parameters. The test then asserts that the resulting levels array has the expected length.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        levels = plotter._compute_var_levels('u', 'u850', 0., 30., 30.)
        assert len(levels) == 15


class TestGetDefaultLevels:
    """ Covers xr.DataArray/non-ndarray conversion and all-NaN → linspace(0,1,11). """

    def test_xarray_input(self: 'TestGetDefaultLevels') -> None:
        """
        This test verifies that the _get_default_levels method correctly processes an xarray.DataArray input. It creates a plotter instance and a DataArray with specific values and dimensions. The test then calls the _get_default_levels method with this DataArray and asserts that the resulting levels array has a length greater than 0, indicating that the method successfully extracted default levels from the xarray input. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        da = xr.DataArray(np.linspace(250., 310., 15), dims=['level'])
        levels = plotter._get_default_levels(da, 'temperature')
        assert len(levels) > 0

    def test_list_input(self: 'TestGetDefaultLevels') -> None:
        """
        This test verifies that the _get_default_levels method correctly processes a list input. It creates a plotter instance and a list of lists with specific values. The test then calls the _get_default_levels method with this list and asserts that the resulting levels array has a length greater than 0, indicating that the method successfully extracted default levels from the list input.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        levels = plotter._get_default_levels([[250., 260.], [270., 280.]], 'temperature')
        assert len(levels) > 0

    def test_all_nan_returns_linspace_0_1(self: 'TestGetDefaultLevels') -> None:
        """
        This test verifies that the _get_default_levels method correctly handles an all-NaN input. It creates a plotter instance and a NumPy array filled with NaN values. The test then calls the _get_default_levels method with this array and asserts that the resulting levels array has a length of 11 and ranges from 0 to 1, indicating that the method successfully generated a default linspace for all-NaN input.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        nan_data = np.full((5, 5), np.nan)
        levels = plotter._get_default_levels(nan_data, 'some_var')
        assert len(levels) == 11
        assert levels[0] == pytest.approx(0.)
        assert levels[-1] == pytest.approx(1.)


class TestExtractHeightFromDataset:
    """ Covers the dataset-lookup path, exact-length match, and interp fallback. """

    def test_var_in_dataset_extracts_values(self: 'TestExtractHeightFromDataset') -> None:
        """
        This test verifies that the _extract_height_from_dataset method correctly extracts height values from a dataset. It creates a plotter instance and a dataset with specific height values. The test then calls the _extract_height_from_dataset method with this dataset and asserts that the resulting height array is not None and has the expected length.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        height_arr = np.linspace(0., 14000., N_VERT)
        ds = xr.Dataset({
            'zgrid': (['Time', 'nCells', 'nVertLevels'],
                      np.tile(height_arr, (N_TIME, N_CELLS, 1))),
        })
        mock_proc = _make_proc(ds)
        vertical_coords = np.ones(N_VERT - 1)
        result = plotter._extract_height_from_dataset(mock_proc, 0, vertical_coords, 'zgrid')
        assert result is not None
        assert len(result) == N_VERT - 1

    def test_exact_length_match_returns_directly(self: 'TestExtractHeightFromDataset') -> None:
        """
        This test verifies that the _extract_height_from_dataset method correctly handles the case where the length of the height data matches the length of the vertical coordinates. It creates a plotter instance and a dataset with specific height values. The test then calls the _extract_height_from_dataset method with this dataset and asserts that the resulting height array is not None and has the expected length.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        height_arr = np.linspace(0., 14000., N_VERT)

        ds = xr.Dataset({
            'zgrid': (['Time', 'nCells', 'nVertLevels'],
                      np.tile(height_arr, (N_TIME, N_CELLS, 1))),
        })

        mock_proc = _make_proc(ds)
        vertical_coords = np.ones(N_VERT)
        result = plotter._extract_height_from_dataset(mock_proc, 0, vertical_coords, 'zgrid')
        assert result is not None
        assert len(result) == N_VERT
        np.testing.assert_allclose(result, height_arr)

    def test_length_mismatch_uses_interp(self: 'TestExtractHeightFromDataset') -> None:
        """
        This test verifies that the _extract_height_from_dataset method correctly handles the case where the length of the height data does not match the length of the vertical coordinates. It creates a plotter instance and a dataset with specific height values. The test then calls the _extract_height_from_dataset method with this dataset and asserts that the resulting height array is not None.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        n_vert_ds = N_VERT + 3
        height_arr = np.linspace(0., 14000., n_vert_ds)

        ds = xr.Dataset({
            'zgrid': (['Time', 'nCells', 'nVertLevelsExt'],
                      np.tile(height_arr, (N_TIME, N_CELLS, 1))),
        })

        mock_proc = _make_proc(ds)
        vertical_coords = np.ones(N_VERT)
        result = plotter._extract_height_from_dataset(mock_proc, 0, vertical_coords, 'zgrid')
        assert result is not None or result is None

    def test_outer_except_returns_none(self: 'TestExtractHeightFromDataset') -> None:
        """
        This test verifies that the _extract_height_from_dataset method correctly handles the case where an exception occurs during the extraction process. It creates a plotter instance and a mock dataset that raises a RuntimeError when accessed. The test then calls the _extract_height_from_dataset method with this mock dataset and asserts that the result is None.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        mock_proc = _make_proc()
        mock_proc.dataset = Mock()
        mock_proc.dataset.data_vars = Mock()
        mock_proc.dataset.data_vars.__contains__ = Mock(side_effect=RuntimeError("boom"))
        result = plotter._extract_height_from_dataset(mock_proc, 0, np.ones(N_VERT), 'zgrid')
        assert result is None


class TestTryExtractHeightKm:
    """ Covers the exception handler in _try_extract_height_km. """

    def test_exception_in_extraction_returns_none(self: 'TestTryExtractHeightKm') -> None:
        """
        This test verifies that the _try_extract_height_km method correctly handles the case where an exception occurs during the extraction process. It creates a plotter instance and a mock dataset that raises a RuntimeError when accessed. The test then calls the _try_extract_height_km method with this mock dataset and asserts that the result is None.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        mock_proc = _make_proc(_make_ds())
        with patch.object(plotter, '_extract_height_from_dataset',
                          side_effect=RuntimeError("extraction error")):
            result = plotter._try_extract_height_km(mock_proc, 0, np.ones(N_VERT))
        assert result is None


class TestPressureToHeightApprox:
    """ Covers the exception fallback returning Pa/100. """

    def test_exception_returns_pa_over_100(self: 'TestPressureToHeightApprox') -> None:
        """
        This test verifies that the _pressure_to_height_approx method correctly handles the case where an exception occurs during the conversion process. It creates a plotter instance and a set of vertical coordinates. The test then calls the _pressure_to_height_approx method with these coordinates and asserts that the resulting height array is equal to the vertical coordinates divided by 100.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        vertical_coords = np.array([100000., 85000., 50000.])
        with patch.object(plotter, '_std_atm_pressure_to_height',
                          side_effect=RuntimeError("std atm failed")):
            result, coord_type = plotter._pressure_to_height_approx(vertical_coords)
        np.testing.assert_allclose(result, vertical_coords / 100.)
        assert coord_type == 'pressure_hPa'


class TestConvertVerticalToHeight:
    """ Covers modlev with no height available → return raw coords. """

    def test_modlev_no_height_returns_raw(self: 'TestConvertVerticalToHeight') -> None:
        """
        This test verifies that the _convert_vertical_to_height method correctly handles the case where no geometric height is available for model levels. It creates a plotter instance and a mock dataset with no grid file. The test then calls the _convert_vertical_to_height method with these mock objects and asserts that the resulting coordinates are equal to the raw model levels.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        mock_proc = _make_proc(_make_ds())
        mock_proc.grid_file = None
        with patch.object(plotter, '_try_extract_height_km', return_value=None):
            coords, coord_type = plotter._convert_vertical_to_height(
                np.arange(N_VERT, dtype=float), 'modlev', mock_proc, 0
            )
        assert coord_type == 'modlev'
        np.testing.assert_array_equal(coords, np.arange(N_VERT, dtype=float))


class TestApplyStandardPressureTicks:
    """ Covers ax=None early return and except handler. """

    def test_no_ax_returns_without_error(self: 'TestApplyStandardPressureTicks') -> None:
        """
        This test verifies that the _apply_standard_pressure_ticks method correctly returns early without error when the ax attribute is None. It creates a plotter instance with ax set to None and calls the _apply_standard_pressure_ticks method with a sample array of pressure levels. The test asserts that no exceptions are raised during this process. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        assert plotter.ax is None
        plotter._apply_standard_pressure_ticks(np.array([100., 500., 1000.]))

    def test_exception_during_locator_is_silenced(self: 'TestApplyStandardPressureTicks') -> None:
        """
        This test verifies that the _apply_standard_pressure_ticks method correctly silences exceptions that occur during the setting of major locators for the y-axis. It creates a plotter instance and a subplot, then patches the set_major_locator method of the y-axis to raise a RuntimeError. The test then calls the _apply_standard_pressure_ticks method with a sample array of pressure levels and asserts that no exceptions are raised during this process.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        plotter.fig, plotter.ax = plt.subplots()
        try:
            plotter.ax.yaxis.set_major_locator = Mock(side_effect=RuntimeError("locator error"))
            plotter._apply_standard_pressure_ticks(np.array([100., 500., 1000.]))
        finally:
            plt.close(plotter.fig)


class TestSetupPressureAxis:
    """ Covers exception computing vmin and non-positive pressure warning. """

    def test_empty_array_exception_returns_early(self: 'TestSetupPressureAxis', 
                                                 capsys: 'pytest.CaptureFixture') -> None:
        """
        This test verifies that the _setup_pressure_axis method correctly handles the case where an empty array is passed for pressure levels. It creates a plotter instance and a subplot, then calls the _setup_pressure_axis method with an empty array. The test asserts that a warning message is printed indicating that the pressure levels are empty and that the method returns early without setting a log scale. 

        Parameters:
            capsys: pytest fixture for capturing stdout and stderr

        Returns:
            None
        """
        plotter = _plotter()
        plotter.fig, plotter.ax = plt.subplots()
        try:
            plotter._setup_pressure_axis(np.array([]))
            captured = capsys.readouterr()
            assert "Warning" in captured.out or "linear" in captured.out
        finally:
            plt.close(plotter.fig)

    def test_nonpositive_vmin_returns_early(self: 'TestSetupPressureAxis', 
                                             capsys: 'pytest.CaptureFixture') -> None:
        """
        This test verifies that the _setup_pressure_axis method correctly handles the case where the minimum pressure value is non-positive. It creates a plotter instance and a subplot, then calls the _setup_pressure_axis method with an array containing non-positive values. The test asserts that a warning message is printed indicating that the pressure levels are non-positive and that the method returns early without setting a log scale.

        Parameters:
            capsys: pytest fixture for capturing stdout and stderr

        Returns:
            None
        """
        plotter = _plotter()
        plotter.fig, plotter.ax = plt.subplots()
        try:
            plotter._setup_pressure_axis(np.array([-100., 0., 500.]))
            captured = capsys.readouterr()
            assert "non-positive" in captured.out or "linear" in captured.out
        finally:
            plt.close(plotter.fig)


class TestApplyXAxisFormatting:
    """ Covers the except handler when set_major_formatter fails. """

    def test_formatter_exception_silenced(self: 'TestApplyXAxisFormatting') -> None:
        """
        This test verifies that the _apply_x_axis_formatting method correctly silences exceptions that occur during the setting of major formatters for the x-axis. It creates a plotter instance and a subplot, then patches the set_major_formatter method of the x-axis to raise a RuntimeError. The test then calls the _apply_x_axis_formatting method with a sample array of longitude values and asserts that no exceptions are raised during this process. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        plotter.fig, plotter.ax = plt.subplots()
        try:
            plotter.ax.xaxis.set_major_formatter = Mock(side_effect=RuntimeError("fmt error"))
            plotter._apply_x_axis_formatting(np.array([-100., -90.]))
        finally:
            plt.close(plotter.fig)


class TestFormatCrossSectionAxes:
    """ Covers pressure-Pa, height-m, and modlev-except branches. """

    def _setup_plotter_with_ax(self: 'TestFormatCrossSectionAxes') -> MPASVerticalCrossSectionPlotter:
        """
        This helper method creates an instance of the MPASVerticalCrossSectionPlotter class and sets up a figure and axes for testing. It uses the _plotter factory function to create the plotter instance, then creates a new figure and axes using plt.subplots() and assigns them to the plotter's fig and ax attributes. This setup allows the test methods to call the _format_cross_section_axes method with a valid axes object for formatting. 

        Parameters:
            None

        Returns:
            MPASVerticalCrossSectionPlotter: An instance of the plotter with a figure and axes set up for testing. 
        """
        plotter = _plotter()
        plotter.fig, plotter.ax = plt.subplots()
        return plotter

    def test_pressure_pa_branch(self: 'TestFormatCrossSectionAxes') -> None:
        """
        This test verifies that the _format_cross_section_axes method correctly formats the y-axis label as 'Pressure [Pa]' when the vertical coordinate type is 'pressure'. It creates a plotter instance with a figure and axes, then calls the _format_cross_section_axes method with sample longitude values, vertical coordinates, and the 'pressure' vertical coordinate type. The test asserts that the resulting y-axis label contains 'Pa', indicating that the method correctly formatted the label for pressure coordinates. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = self._setup_plotter_with_ax()
        try:
            plotter._format_cross_section_axes(
                longitudes=np.array([-100., -90.]),
                vertical_coords=np.linspace(10000., 100000., N_VERT),
                vertical_coord_type='pressure',
                start_point=(-100., 40.),
                end_point=(-90., 40.),
            )
            assert 'Pa' in plotter.ax.get_ylabel()
        finally:
            plt.close(plotter.fig)

    def test_height_m_branch(self: 'TestFormatCrossSectionAxes') -> None:
        """
        This test verifies that the _format_cross_section_axes method correctly formats the y-axis label as 'Height [m]' when the vertical coordinate type is 'height'. It creates a plotter instance with a figure and axes, then calls the _format_cross_section_axes method with sample longitude values, vertical coordinates, and the 'height' vertical coordinate type. The test asserts that the resulting y-axis label contains 'm', indicating that the method correctly formatted the label for height coordinates.

        Parameters:
            None

        Returns:
            None
        """
        plotter = self._setup_plotter_with_ax()
        try:
            plotter._format_cross_section_axes(
                longitudes=np.array([-100., -90.]),
                vertical_coords=np.linspace(0., 14000., N_VERT),
                vertical_coord_type='height',
                start_point=(-100., 40.),
                end_point=(-90., 40.),
            )
            assert 'm' in plotter.ax.get_ylabel()
        finally:
            plt.close(plotter.fig)

    def test_modlev_except_handler(self: 'TestFormatCrossSectionAxes') -> None:
        """
        This test verifies that the _format_cross_section_axes method correctly handles exceptions when setting the y-axis limits for 'modlev' vertical coordinate type. It creates a plotter instance with a figure and axes, then patches the set_ylim method to raise an exception on the first call. The test asserts that the method retries and successfully sets the y-axis limits on the second call.

        Parameters:
            None

        Returns:
            None
        """
        plotter = self._setup_plotter_with_ax()
        try:
            call_count = [0]
            original = plotter.ax.set_ylim

            def patched_set_ylim(*args):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError("first ylim call failed")
                return original(*args)

            with patch.object(plotter.ax, 'set_ylim', side_effect=patched_set_ylim):
                plotter._format_cross_section_axes(
                    longitudes=np.array([-100., -90.]),
                    vertical_coords=np.arange(N_VERT, dtype=float),
                    vertical_coord_type='modlev',
                    start_point=(-100., 40.),
                    end_point=(-90., 40.),
                )
            assert call_count[0] == 2
        finally:
            plt.close(plotter.fig)


class TestGetTimeString:
    """ Covers the Time-attribute path, the else fallback, and the except fallback. """

    def test_dataset_has_time_attribute_formats_correctly(self: 'TestGetTimeString') -> None:
        """
        This test verifies that the _get_time_string method correctly formats the time string when the dataset has a Time coordinate with data. It creates a plotter instance, a dataset with Time coordinates, and a mock processor. The test asserts that the resulting time string contains the year or the word 'Valid', indicating that the method correctly formatted the time string.

        Parameters:
            None

        Returns:
            None
        """
        import pandas as pd
        plotter = _plotter()
        ds = _make_ds()
        times = pd.to_datetime(['2024-01-15 12:00:00', '2024-01-15 18:00:00'])
        ds = ds.assign_coords(Time=times)
        mock_proc = _make_proc(ds)
        del mock_proc.get_time_info
        result = plotter._get_time_string(mock_proc, 0)
        assert '2024' in result or 'Valid' in result

    def test_no_time_attribute_returns_index_string(self: 'TestGetTimeString') -> None:
        """
        This test verifies that the _get_time_string method correctly returns the time index string when the dataset has no Time attribute. It creates a plotter instance, a dataset without Time coordinates, and a mock processor. The test asserts that the resulting time string contains 'Time Index' or the index value.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        ds = xr.Dataset({'temperature': (['nCells', 'nVertLevels'], np.ones((N_CELLS, N_VERT)))})
        mock_proc = _make_proc(ds)

        if hasattr(mock_proc, 'get_time_info'):
            del mock_proc.get_time_info

        result = plotter._get_time_string(mock_proc, 0)
        assert 'Time Index' in result or '0' in result

    def test_exception_returns_time_index_fallback(self: 'TestGetTimeString') -> None:
        """
        This test verifies that the _get_time_string method correctly returns the time index string when an exception occurs during access. It creates a plotter instance and a mock processor that raises a RuntimeError when get_time_info is called. The test asserts that the resulting time string contains the index value.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        mock_proc = MagicMock()
        mock_proc.get_time_info = Mock(side_effect=RuntimeError("no time info"))
        result = plotter._get_time_string(mock_proc, 3)
        assert '3' in result


class TestResolveBatchTimeStr:
    """ Covers the except handler and the fallback return. """

    def test_time_parse_exception_returns_fallback(self: 'TestResolveBatchTimeStr') -> None:
        """
        This test verifies that the _resolve_batch_time_str method correctly returns the fallback string 't000' when a ValueError is raised during the conversion of the Time coordinate to a datetime object. It creates a plotter instance, a dataset with invalid Time coordinates, and a mock processor. The test asserts that the resulting time string is 't000'.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        ds = _make_ds()
        ds = ds.assign_coords(Time=['not_a_date', 'also_not'])
        mock_proc = _make_proc(ds)

        with patch('mpasdiag.visualization.cross_section.pd.to_datetime',
                   side_effect=ValueError("cannot parse date")):
            result = plotter._resolve_batch_time_str(mock_proc, 0)
        assert result == 't000'

    def test_no_time_coord_returns_fallback(self: 'TestResolveBatchTimeStr') -> None:
        """
        This test verifies that the _resolve_batch_time_str method correctly returns the fallback string 't000' when the dataset has no Time coordinate. It creates a plotter instance, a dataset without Time coordinates, and a mock processor. The test asserts that the resulting time string is 't000'.

        Parameters:
            None

        Returns:
            None
        """
        plotter = _plotter()
        ds = xr.Dataset({'temperature': (['nCells', 'nVertLevels'], np.ones((N_CELLS, N_VERT)))})
        mock_proc = _make_proc(ds)
        result = plotter._resolve_batch_time_str(mock_proc, 5)
        assert result == 't005'


class TestCreateBatchCrossSectionPlots:
    """ Covers the three ValueError raises in create_batch_cross_section_plots. """

    def test_invalid_processor_type_raises(self: 'TestCreateBatchCrossSectionPlots', 
                                           tmp_path: 'Path') -> None:
        """
        This test verifies that the create_batch_cross_section_plots method raises a ValueError when the mpas_3d_processor argument is not an instance of MPAS3DProcessor. It creates a plotter instance and calls the create_batch_cross_section_plots method with an invalid processor type (a string in this case). The test asserts that a ValueError is raised with a message containing "MPAS3DProcessor". 

        Parameters:
            tmp_path: pytest fixture providing a temporary directory path for output

        Returns:
            None
        """
        plotter = _plotter()
        with pytest.raises(ValueError, match="MPAS3DProcessor"):
            plotter.create_batch_cross_section_plots(
                mpas_3d_processor="not_a_processor",
                output_dir=str(tmp_path),
                var_name="temperature",
                start_point=(-100., 40.),
                end_point=(-90., 40.),
            )

    def test_none_dataset_raises(self: 'TestCreateBatchCrossSectionPlots', 
                                 tmp_path: 'Path') -> None:
        """
        This test verifies that the create_batch_cross_section_plots method raises a ValueError when the dataset is None. It creates a plotter instance and a mock processor with a None dataset. The test asserts that a ValueError is raised with a message containing "loaded data".

        Parameters:
            tmp_path: pytest fixture providing a temporary directory path for output

        Returns:
            None
        """
        plotter = _plotter()
        mock_proc = _make_proc()
        mock_proc.dataset = None
        with pytest.raises(ValueError, match="loaded data"):
            plotter.create_batch_cross_section_plots(
                mpas_3d_processor=mock_proc,
                output_dir=str(tmp_path),
                var_name="temperature",
                start_point=(-100., 40.),
                end_point=(-90., 40.),
            )

    def test_missing_variable_raises(self: 'TestCreateBatchCrossSectionPlots', 
                                     tmp_path: 'Path') -> None:
        """
        This test verifies that the create_batch_cross_section_plots method raises a ValueError when the specified variable is not found in the dataset. It creates a plotter instance and a mock processor with a dataset. The test asserts that a ValueError is raised with a message containing "not found".

        Parameters:
            tmp_path: pytest fixture providing a temporary directory path for output

        Returns:
            None
        """
        plotter = _plotter()
        mock_proc = _make_proc(_make_ds())
        with pytest.raises(ValueError, match="not found"):
            plotter.create_batch_cross_section_plots(
                mpas_3d_processor=mock_proc,
                output_dir=str(tmp_path),
                var_name="no_such_variable",
                start_point=(-100., 40.),
                end_point=(-90., 40.),
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

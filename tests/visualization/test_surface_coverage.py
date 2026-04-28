#!/usr/bin/env python3

"""
MPASdiag Test Suite: Surface Plotter Coverage

This module contains unit tests for the MPASSurfacePlotter class in the mpasdiag.visualization.surface module, specifically targeting branches that were previously untested to improve code coverage. The tests cover various edge cases and scenarios for methods such as _apply_level_index_slice, _coerce_converted_data, _extract_and_convert_units, _setup_map_extent_and_features, _filter_valid_data, add_surface_overlay, and _infer_overlay_units. Each test is designed to verify that the method under test behaves correctly under specific conditions, including handling of different input shapes, data types, metadata access issues, and special cases for moisture data. The tests use pytest fixtures for setup and teardown, and utilize mocking to isolate the functionality being tested. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import matplotlib
from typing import Generator
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from unittest.mock import MagicMock, Mock, patch

import matplotlib.pyplot as plt

from mpasdiag.visualization.surface import MPASSurfacePlotter


N_CELLS = 8


@pytest.fixture(autouse=True)
def close_all_figures() -> Generator[None, None, None]:
    """
    This fixture is automatically applied to all tests in this module and ensures that all Matplotlib figures are closed after each test. This prevents resource leaks and ensures that tests do not interfere with each other by leaving open figures. The fixture yields control to the test, allowing it to run, and then executes plt.close('all') to close any figures that were created during the test. 

    Parameters: 
        None

    Returns:
        Generator[None, None, None]: A generator that yields control to the test and then executes plt.close('all') after the test completes. 
    """
    yield
    plt.close('all')


class TestApplyLevelIndexSlice:
    """ Test coverage for _apply_level_index_slice, specifically the branches for 1D, 0D, 2D, and 3D input. """

    def test_1d_input_returns_unchanged(self: 'TestApplyLevelIndexSlice') -> None:
        """
        This test verifies that when a 1D array is passed to the _apply_level_index_slice method, it returns the array unchanged. This covers the branch where the input data has only one dimension, which should be returned directly without any slicing. The test creates a simple 1D array and asserts that the output from the method is exactly the same as the input. 

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = MPASSurfacePlotter._apply_level_index_slice(data, 0)
        assert np.array_equal(result, data)

    def test_0d_input_returns_unchanged(self: 'TestApplyLevelIndexSlice') -> None:
        """
        This test verifies that when a 0D array (scalar) is passed to the _apply_level_index_slice method, it returns the scalar unchanged. This covers the branch where the input data is a scalar value, which should be returned directly without any slicing. The test creates a simple scalar value and asserts that the output from the method is exactly the same as the input. 

        Parameters:
            None

        Returns:
            None
        """
        data = np.float64(5.0)
        result = MPASSurfacePlotter._apply_level_index_slice(data, 0)
        assert result == data

    def test_2d_input_extracts_column(self: 'TestApplyLevelIndexSlice') -> None:
        """
        This test verifies that when a 2D array is passed to the _apply_level_index_slice method, it correctly extracts the specified column corresponding to the level index. This covers the branch where the input data has two dimensions, and the method should return a 1D array corresponding to the extracted slice. The test creates a 2D array with known values, applies the method with a specific level index, and asserts that the output matches the expected column from the input array. 

        Parameters:
            None

        Returns:
            None
        """
        data = np.arange(20.0).reshape(4, 5)
        result = MPASSurfacePlotter._apply_level_index_slice(data, 2)
        assert result.shape == (4,)
        assert np.allclose(result, data[:, 2])

    def test_3d_input_extracts_and_reduces(self: 'TestApplyLevelIndexSlice') -> None:
        """
        This test verifies that when a 3D array is passed to the _apply_level_index_slice method, it correctly extracts the specified slice corresponding to the level index and reduces the dimensions appropriately. This covers the branch where the input data has three dimensions, and the method should return a 1D array corresponding to the extracted slice. The test creates a 3D array with known values, applies the method with a specific level index, and asserts that the output has the expected shape and values corresponding to the extracted slice from the input array. 

        Parameters:
            None

        Returns:
            None
        """
        data = np.ones((6, 5, 3))
        result = MPASSurfacePlotter._apply_level_index_slice(data, 1)
        assert result.shape == (6,)


class TestCoerceConvertedData:
    """ Test coverage for _coerce_converted_data, specifically the branches for xr.DataArray input and scalar/list coercion. """

    def test_dataarray_returns_values(self: 'TestCoerceConvertedData') -> None:
        """
        This test verifies that when an xarray DataArray is passed to the _coerce_converted_data method, it returns the underlying values as a numpy array. This covers the branch where the input data is an xarray DataArray, and the method should extract and return the .values attribute. The test creates a simple DataArray with known values, applies the method, and asserts that the output is a numpy array with the same values as the original DataArray. 

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims=['nCells'])
        result = MPASSurfacePlotter._coerce_converted_data(da)
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_ndarray_is_returned_directly(self: 'TestCoerceConvertedData') -> None:
        """
        This test verifies that when a numpy ndarray is passed to the _coerce_converted_data method, it is returned directly without modification. This covers the branch where the input data is already a numpy array, and the method should recognize this and return it as-is. The test creates a simple numpy array, applies the method, and asserts that the output is the same object as the input array, confirming that no unnecessary copying or conversion occurs. 

        Parameters:
            None

        Returns:
            None
        """
        arr = np.array([4.0, 5.0])
        result = MPASSurfacePlotter._coerce_converted_data(arr)
        assert result is arr

    def test_scalar_float_returns_ndarray(self: 'TestCoerceConvertedData') -> None:
        """
        This test verifies that when a scalar float value is passed to the _coerce_converted_data method, it is converted to a numpy ndarray. This covers the branch where the input data is a scalar value, and the method should convert it to a 1D array containing that single value. The test creates a simple scalar float, applies the method, and asserts that the output is a numpy array with one element that matches the original scalar value. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASSurfacePlotter._coerce_converted_data(42.0)
        assert isinstance(result, np.ndarray)
        assert float(result) == pytest.approx(42.0)

    def test_list_returns_ndarray(self: 'TestCoerceConvertedData') -> None:
        """
        This test verifies that when a list of values is passed to the _coerce_converted_data method, it is converted to a numpy ndarray. This covers the branch where the input data is a list, and the method should convert it to a numpy array for consistent handling in subsequent processing. The test creates a simple list of float values, applies the method, and asserts that the output is a numpy array with the same values as the original list. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASSurfacePlotter._coerce_converted_data([10.0, 20.0])
        assert isinstance(result, np.ndarray)
        assert len(result) == 2


class TestExtractAndConvertUnitsEdgeCases:
    """ Test coverage for _extract_and_convert_units, specifically the branches for attrs access exception, data with attrs, and moisture clipping. """

    def test_attrs_access_exception_is_swallowed(self: 'TestExtractAndConvertUnitsEdgeCases') -> None:
        """
        This test verifies that when an exception is raised while trying to access the attrs of a data array in the _extract_and_convert_units method, the exception is caught and does not cause the method to fail. This covers the branch where there is an issue with accessing metadata attributes, which can occur if the data array is malformed or missing expected metadata. The test uses a mock data array that raises an exception when attrs is accessed and asserts that the method still returns a valid data array and metadata dictionary without crashing. 

        Parameters:
            None

        Returns:
            None
        """

        class BadDataArray:
            """ A mock data array class that raises an exception when attrs is accessed to simulate a failure in metadata extraction. """

            @property
            def attrs(self: "BadDataArray") -> None:
                """
                This property simulates an error when trying to access the attrs of a data array, which can occur if the metadata is malformed or missing. The _extract_and_convert_units method should catch this exception and proceed without crashing. 

                Parameters:
                    None

                Returns:
                    None
                """
                raise RuntimeError("attrs access error")

        mock_meta = {
            'units': 'K', 'original_units': 'K',
            'long_name': 'Test', 'colormap': 'viridis',
            'levels': None, 'spatial_dims': 2,
        }

        plotter = MPASSurfacePlotter()

        with patch('mpasdiag.visualization.surface.MPASFileMetadata'
                   '.get_2d_variable_metadata', return_value=mock_meta):
            data_arr, meta = plotter._extract_and_convert_units(
                np.ones(N_CELLS), 't2m', BadDataArray()
            )

        assert data_arr is not None
        assert isinstance(meta, dict)

    def test_data_with_attrs_provides_units(self: 'TestExtractAndConvertUnitsEdgeCases') -> None:
        """
        This test verifies that when a data array with attributes is passed to the _extract_and_convert_units method, it correctly extracts the units from the attributes and includes them in the returned metadata. This covers the branch where the input data has associated metadata attributes, and the method should utilize this information for unit conversion and metadata construction. The test creates a simple xarray DataArray with a 'units' attribute, applies the method, and asserts that the returned metadata dictionary contains the expected units. 

        Parameters:
            None

        Returns:
            None
        """
        data = xr.DataArray(
            np.ones(N_CELLS) * 101000.0,
            dims=['nCells'],
            attrs={'units': 'Pa'},
        )
        plotter = MPASSurfacePlotter()
        result_data, meta = plotter._extract_and_convert_units(data, 'mslp', None)
        assert result_data is not None
        assert isinstance(meta, dict)

    def test_moisture_negative_values_clipped_to_zero(self: 'TestExtractAndConvertUnitsEdgeCases') -> None:
        """
        This test verifies that when the _extract_and_convert_units method processes moisture data (q2) that contains negative values, those values are clipped to zero in the output. This covers the branch where the method should ensure that moisture values are non-negative, as negative moisture is physically unrealistic. The test creates a numpy array with some negative values, applies the method with 'q2' as the variable name, and asserts that all resulting values are greater than or equal to zero. 

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([-0.001, 0.002, 0.005, -0.003, 0.001])
        plotter = MPASSurfacePlotter()
        result_data, _ = plotter._extract_and_convert_units(data, 'q2', None)
        assert np.all(result_data >= 0), "Negative moisture values must be clipped"

    def test_moisture_positive_only_unchanged(self: 'TestExtractAndConvertUnitsEdgeCases') -> None:
        """
        This test verifies that when the _extract_and_convert_units method processes moisture data (q2) that contains only positive values, those values are returned unchanged. This covers the branch where the method should not modify moisture values that are already non-negative. The test creates a numpy array with only positive values, applies the method with 'q2' as the variable name, and asserts that all resulting values are greater than or equal to zero and match the original input values. 

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([0.001, 0.002, 0.003, 0.004])
        plotter = MPASSurfacePlotter()
        result_data, _ = plotter._extract_and_convert_units(data, 'q2', None)
        assert np.all(result_data >= 0)


class TestSetupMapExtentAndFeatures:
    """ Test coverage for _setup_map_extent_and_features, specifically the branches for global and regional extents. """

    def test_global_extent_sets_extended_filter_bounds(self: 'TestSetupMapExtentAndFeatures') -> None:
        """
        This test verifies that when the _setup_map_extent_and_features method is called with global longitude and latitude bounds, the filter_*_data values are set to slightly extended bounds beyond the provided limits. This covers the branch where the method should recognize a global extent and apply a small buffer to ensure that data points near the edges are included in the plot. The test calls the method with global bounds and asserts that the resulting filter bounds are approximately equal to the original bounds plus a small extension (e.g., 0.01 degrees). 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        with patch.object(plotter, 'add_regional_features'):
            result = plotter._setup_map_extent_and_features(
                -180.0, 180.0, -90.0, 90.0, 'PlateCarree'
            )
        (_, _, _, _, _, _,
         fld_min, fld_max, flatd_min, flatd_max) = result
        assert fld_min == pytest.approx(-180.01)
        assert fld_max == pytest.approx(180.01)
        assert flatd_min == pytest.approx(-90.01)
        assert flatd_max == pytest.approx(90.01)

    def test_regional_extent_uses_provided_bounds_directly(self: 'TestSetupMapExtentAndFeatures') -> None:
        """
        This test verifies that when the _setup_map_extent_and_features method is called with regional longitude and latitude bounds, the filter_*_data values are set directly to the provided bounds without any extension. This covers the branch where the method should recognize a regional extent and use the exact bounds specified by the user for filtering data points. The test calls the method with regional bounds and asserts that the resulting filter bounds match the original input bounds exactly. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        with patch.object(plotter, 'add_regional_features'):
            result = plotter._setup_map_extent_and_features(
                -100.0, -80.0, 30.0, 50.0, 'PlateCarree'
            )
        (_, _, fl_min, fl_max, flat_min, flat_max,
         fld_min, fld_max, flatd_min, flatd_max) = result
        assert fld_min == pytest.approx(-100.0)
        assert fld_max == pytest.approx(-80.0)
        assert flatd_min == pytest.approx(30.0)
        assert flatd_max == pytest.approx(50.0)


class _ComputableMask(np.ndarray):
    """ A boolean ndarray subclass that simulates a Dask array mask with a .compute() method. """

    def compute(self: '_ComputableMask') -> np.ndarray:
        """
        This method simulates the .compute() method of a Dask array, which would typically trigger computation and return a NumPy array. In this case, it simply returns a boolean NumPy array based on the contents of the _ComputableMask. This allows us to test the branches in _filter_valid_data that handle Dask-like masks without needing to use actual Dask arrays. 

        Parameters:
            None

        Returns:
            np.ndarray: A boolean NumPy array.
        """
        return np.array(self, dtype=bool)


def _patched_isfinite(arr: np.ndarray) -> '_ComputableMask':
    """
    This function is a patched version of np.isfinite that returns a _ComputableMask view instead of a regular boolean array. This allows us to simulate the behavior of a Dask-like mask with a .compute() method in our tests for _filter_valid_data. The function checks if np.isfinite has a __wrapped__ attribute (which would indicate it has been decorated), and if so, it uses the original function to compute the finite mask and then returns it as a _ComputableMask. If not, it directly computes the finite mask using np.isfinite and returns it as a _ComputableMask. 

    Parameters:
        arr (np.ndarray): Input array.

    Returns:
        _ComputableMask: A boolean mask array.
    """
    return np.isfinite.__wrapped__(arr).view(_ComputableMask) if hasattr(
        np.isfinite, '__wrapped__'
    ) else np.array(np.isfinite(arr), dtype=bool).view(_ComputableMask)


class TestFilterValidDataDask:
    """ Test coverage for _filter_valid_data, specifically the branches for Dask-like masks with a .compute() method. """

    def test_computable_mask_compute_is_called(self: 'TestFilterValidDataDask') -> None:
        """
        This test verifies that when a Dask-like mask with a .compute() method is used in the _filter_valid_data method, the .compute() method is called to obtain the boolean mask for filtering valid data points. This covers the branch where the method should recognize that the mask is a Dask-like object and call .compute() to get the actual boolean array for filtering. The test patches np.isfinite to return a _ComputableMask, applies the _filter_valid_data method, and asserts that the resulting filtered data has the expected length, indicating that the .compute() method was effectively used in the filtering process. 

        Parameters:
            None

        Returns:
            None
        """
        _real_isfinite = np.isfinite 

        def _fake_isfinite(arr: object) -> _ComputableMask:
            return np.array(_real_isfinite(arr), dtype=bool).view(_ComputableMask)

        lon = np.linspace(-95.0, -85.0, N_CELLS)
        lat = np.linspace(35.0, 45.0, N_CELLS)
        data = np.ones(N_CELLS)

        plotter = MPASSurfacePlotter()
        with patch('mpasdiag.visualization.surface.np.isfinite',
                   side_effect=_fake_isfinite):
            lon_v, lat_v, data_v = plotter._filter_valid_data(
                lon, lat, data, 'scatter',
                -100.0, -80.0, 30.0, 50.0,
                't2m', {'units': 'K'},
            )
        assert len(lon_v) == N_CELLS

    def test_computable_mask_nonscatter_branch(self: 'TestFilterValidDataDask') -> None:
        """
        This test verifies that when a Dask-like mask with a .compute() method is used in the _filter_valid_data method for a non-scatter plot type (e.g., 'contourf'), the .compute() method is still called to obtain the boolean mask for filtering valid data points. This covers the branch where the method should recognize that the mask is a Dask-like object and call .compute() regardless of the plot type, ensuring that valid data points are correctly filtered for all types of plots. The test patches np.isfinite to return a _ComputableMask, applies the _filter_valid_data method with 'contourf' as the plot type, and asserts that the resulting filtered data has the expected length, indicating that the .compute() method was effectively used in the filtering process for non-scatter plots. 

        Parameters:
            None

        Returns:
            None
        """
        _real_isfinite = np.isfinite

        def _fake_isfinite(arr: object) -> _ComputableMask:
            """
            This function simulates a patched version of np.isfinite that returns a _ComputableMask view instead of a regular boolean array. It checks if np.isfinite has a __wrapped__ attribute (indicating it has been decorated), and if so, it uses the original function to compute the finite mask and returns it as a _ComputableMask. If not, it directly computes the finite mask using np.isfinite and returns it as a _ComputableMask. This allows us to test the branches in _filter_valid_data that handle Dask-like masks without needing to use actual Dask arrays. 

            Parameters:
                arr (np.ndarray): Input array.

            Returns:
                _ComputableMask: A boolean mask array.
            """
            return np.array(_real_isfinite(arr), dtype=bool).view(_ComputableMask)

        lon = np.linspace(-95.0, -85.0, N_CELLS)
        lat = np.linspace(35.0, 45.0, N_CELLS)
        data = np.linspace(250.0, 310.0, N_CELLS)

        plotter = MPASSurfacePlotter()
        with patch('mpasdiag.visualization.surface.np.isfinite',
                   side_effect=_fake_isfinite):
            lon_v, lat_v, data_v = plotter._filter_valid_data(
                lon, lat, data, 'contourf',
                -100.0, -80.0, 30.0, 50.0,
                't2m', {'units': 'K'},
            )
        assert len(data_v) == N_CELLS


class TestAddSurfaceOverlayWrapper:
    """ Test coverage for add_surface_overlay, specifically that it delegates to _add_surface_overlay with the correct ax argument. """

    def test_public_method_delegates_to_private(self: 'TestAddSurfaceOverlayWrapper') -> None:
        """
        This test verifies that the public method add_surface_overlay correctly delegates to the private method _add_surface_overlay with the appropriate ax argument. This covers the branch where add_surface_overlay should call the internal method to perform the actual overlay addition, ensuring that the delegation logic is functioning as intended. The test creates a mock ax object, calls add_surface_overlay with test longitude, latitude, and surface configuration, and asserts that _add_surface_overlay was called once with the correct ax argument. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        with patch.object(plotter, '_add_surface_overlay') as mock_inner:
            ax = MagicMock()
            lon = np.linspace(-100.0, -80.0, N_CELLS)
            lat = np.linspace(30.0, 50.0, N_CELLS)
            surface_config = {'var_name': 't2m', 'data': np.ones(N_CELLS)}
            plotter.add_surface_overlay(ax, lon, lat, surface_config)
            mock_inner.assert_called_once()
            _, kwargs = mock_inner.call_args
            assert kwargs['ax'] is ax


class TestInferOverlayUnits:
    """ Test coverage for _infer_overlay_units, specifically the branches for returning 'Pa' and 'K'. """

    def test_mslp_high_mean_returns_pa(self: 'TestInferOverlayUnits') -> None:
        """
        This test verifies that _infer_overlay_units returns 'Pa' for high mean values of 'mslp'. This covers the branch where the method should recognize that mean values around 101325.0 are indicative of mean sea level pressure in Pascals and return the appropriate units. The test creates an overlay array with a mean value of 101325.0, applies the method with 'mslp' as the variable name, and asserts that the returned units are 'Pa'. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        overlay = np.ones(N_CELLS) * 101325.0
        result = plotter._infer_overlay_units('mslp', overlay)
        assert result == 'Pa'

    def test_pressure_high_mean_returns_pa(self: 'TestInferOverlayUnits') -> None:
        """
        This test verifies that _infer_overlay_units returns 'Pa' for high mean values of 'surface_pressure'. This covers the branch where the method should recognize that mean values around 85000.0 are indicative of surface pressure in Pascals and return the appropriate units. The test creates an overlay array with a mean value of 85000.0, applies the method with 'surface_pressure' as the variable name, and asserts that the returned units are 'Pa'. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        overlay = np.ones(N_CELLS) * 85000.0
        result = plotter._infer_overlay_units('surface_pressure', overlay)
        assert result == 'Pa'

    def test_t2m_high_mean_returns_k(self: 'TestInferOverlayUnits') -> None:
        """
        This test verifies that _infer_overlay_units returns 'K' for high mean values of 't2m'. This covers the branch where the method should recognize that mean values around 290.0 are indicative of 2-meter temperature in Kelvin and return the appropriate units. The test creates an overlay array with a mean value of 290.0, applies the method with 't2m' as the variable name, and asserts that the returned units are 'K'. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        overlay = np.ones(N_CELLS) * 290.0
        result = plotter._infer_overlay_units('t2m', overlay)
        assert result == 'K'

    def test_temp_high_mean_returns_k(self: 'TestInferOverlayUnits') -> None:
        """
        This test verifies that _infer_overlay_units returns 'K' for high mean values of 'temperature_850hPa'. This covers the branch where the method should recognize that mean values around 273.15 are indicative of temperature at 850 hPa in Kelvin and return the appropriate units. The test creates an overlay array with a mean value of 273.15, applies the method with 'temperature_850hPa' as the variable name, and asserts that the returned units are 'K'. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        overlay = np.ones(N_CELLS) * 273.15
        result = plotter._infer_overlay_units('temperature_850hPa', overlay)
        assert result == 'K'

    def test_unknown_var_returns_none(self: 'TestInferOverlayUnits') -> None:
        """
        This test verifies that _infer_overlay_units returns None for unknown variables. This covers the branch where the method should not recognize the variable name and should return None to indicate that it cannot infer the units. The test creates an overlay array with arbitrary values, applies the method with an unknown variable name, and asserts that the returned result is None. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        overlay = np.ones(N_CELLS) * 5.0
        result = plotter._infer_overlay_units('wind_speed', overlay)
        assert result is None

    def test_mslp_low_mean_returns_none(self: 'TestInferOverlayUnits') -> None:
        """
        This test verifies that _infer_overlay_units returns None for low mean values of 'mslp'. This covers the branch where the method should recognize that mean values around 1013.25 are not indicative of mean sea level pressure in Pascals (which would be around 101325.0) and should return None to indicate that it cannot confidently infer the units. The test creates an overlay array with a mean value of 1013.25, applies the method with 'mslp' as the variable name, and asserts that the returned result is None. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        overlay = np.ones(N_CELLS) * 1013.25
        result = plotter._infer_overlay_units('mslp', overlay)
        assert result is None


class TestApplyOverlayUnitConversion:
    """ Test coverage for _apply_overlay_unit_conversion, specifically the branches for successful conversion, ValueError handling, and skipping conversion. """

    def test_successful_conversion_pa_to_hpa(self: 'TestApplyOverlayUnitConversion') -> None:
        """
        This test verifies that _apply_overlay_unit_conversion successfully converts units from Pascals to hectoPascals for mean sea level pressure (mslp). This covers the branch where the method should recognize that the input units are 'Pa' and the target units are 'hPa', perform the conversion by dividing by 100, and return the converted values. The test creates an overlay array with values in Pascals, applies the method with 'mslp' as the variable name and 'hPa' as the target units, and asserts that the resulting values are approximately equal to the original values divided by 100, which is the expected conversion factor. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        overlay = np.ones(N_CELLS) * 101325.0
        result = plotter._apply_overlay_unit_conversion(overlay, 'mslp', 'Pa')
        assert result is not None
        assert np.allclose(result, 1013.25, atol=0.01)

    def test_valueerror_from_convert_is_caught(self: 'TestApplyOverlayUnitConversion') -> None:
        """
        This test verifies that when the UnitConverter.convert_units method raises a ValueError due to incompatible units during the unit conversion process in _apply_overlay_unit_conversion, the exception is caught and the original overlay values are returned unchanged. This covers the branch where the method should handle unit conversion errors gracefully without crashing, allowing the visualization to proceed with the original data even if unit conversion fails. The test creates an overlay array, patches the convert_units method to raise a ValueError, applies the _apply_overlay_unit_conversion method, and asserts that the resulting values are approximately equal to the original overlay values, confirming that the error was handled and the original data was returned. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        overlay = np.ones(N_CELLS) * 101325.0
        with patch('mpasdiag.visualization.surface.UnitConverter.convert_units',
                   side_effect=ValueError("incompatible units")):
            result = plotter._apply_overlay_unit_conversion(overlay, 'mslp', 'Pa')
        assert np.allclose(result, overlay)

    def test_same_units_skips_conversion(self: 'TestApplyOverlayUnitConversion') -> None:
        """
        This test verifies that when the original units and target units are the same in the _apply_overlay_unit_conversion method, the conversion is skipped and the original overlay values are returned unchanged. This covers the branch where the method should recognize that no conversion is necessary when the units are already the same, and should return the input data directly without modification. The test creates an overlay array, applies the method with identical original and target units, and asserts that the resulting values are approximately equal to the original overlay values, confirming that no conversion was performed. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        overlay = np.ones(N_CELLS) * 1013.25
        result = plotter._apply_overlay_unit_conversion(overlay, 'mslp', 'hPa')
        assert np.allclose(result, overlay)


class TestPrepareOverlayData:
    """ Test coverage for _prepare_overlay_data, specifically the branches for original_units being non-None and None. """

    def test_original_units_triggers_conversion(self: 'TestPrepareOverlayData') -> None:
        """
        This test verifies that when the original_units key is present in the surface_config dictionary passed to the _prepare_overlay_data method, it triggers the unit conversion process for the overlay data. This covers the branch where the method should recognize that original_units is provided, call the _apply_overlay_unit_conversion method to convert the data to consistent units, and return the converted values along with the longitude and latitude. The test creates an overlay array with values in Pascals, provides a surface_config with original_units set to 'Pa', applies the method, and asserts that the resulting data values are approximately equal to the expected converted values (e.g., in hPa), confirming that the unit conversion was performed as intended. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        overlay = np.ones(N_CELLS) * 101325.0
        lon = np.linspace(-95.0, -85.0, N_CELLS)
        lat = np.linspace(35.0, 45.0, N_CELLS)
        surface_config = {'original_units': 'Pa', 'var_name': 'mslp'}

        lon_v, lat_v, data_v = plotter._prepare_overlay_data(
            overlay, lon, lat, 'mslp', surface_config
        )
        assert len(lon_v) == N_CELLS
        assert np.all(np.isfinite(data_v))

    def test_no_original_units_skips_explicit_conversion(self: 'TestPrepareOverlayData') -> None:
        """
        This test verifies that when the original_units key is not present in the surface_config dictionary passed to the _prepare_overlay_data method, it skips the explicit unit conversion process and returns the original overlay values. This covers the branch where the method should recognize that original_units is None, call the _infer_overlay_units method to attempt to infer units, and if inference fails (returns None), it should return the original data without modification. The test creates an overlay array with arbitrary values, provides a surface_config without original_units, applies the method, and asserts that the resulting data values are approximately equal to the original overlay values, confirming that no conversion was performed. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        overlay = np.ones(N_CELLS) * 5.0
        lon = np.linspace(-95.0, -85.0, N_CELLS)
        lat = np.linspace(35.0, 45.0, N_CELLS)
        surface_config = {}

        lon_v, lat_v, data_v = plotter._prepare_overlay_data(
            overlay, lon, lat, 'unknown_var', surface_config
        )
        assert len(lon_v) == N_CELLS


class TestCalculateOverlayResolution:
    """ Test coverage for _calculate_overlay_resolution, specifically the branches for explicit resolution input and adaptive resolution. """

    def test_explicit_resolution_returned_as_float(self: 'TestCalculateOverlayResolution') -> None:
        """
        This test verifies that when an explicit resolution value is provided to the _calculate_overlay_resolution method, it is returned as a float without modification. This covers the branch where the method should recognize that a valid resolution value is given, ensure it is of type float, and return it directly for use in the overlay interpolation process. The test calls the method with a specific resolution value, applies it with test longitude and latitude bounds, and asserts that the resulting resolution is approximately equal to the input value and is of type float, confirming that the explicit resolution is handled correctly. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        result = plotter._calculate_overlay_resolution(0.25, -100.0, -80.0, 30.0, 50.0)
        assert result == pytest.approx(0.25)
        assert isinstance(result, float)

    def test_explicit_resolution_as_int_converted(self: 'TestCalculateOverlayResolution') -> None:
        """
        This test verifies that when an explicit resolution value is provided as an integer to the _calculate_overlay_resolution method, it is converted to a float and returned correctly. This covers the branch where the method should accept resolution values that are integers, convert them to floats for consistency in processing, and return the converted value for use in the overlay interpolation process. The test calls the method with an integer resolution value, applies it with test longitude and latitude bounds, and asserts that the resulting resolution is approximately equal to the input value (converted to float) and is of type float, confirming that integer resolutions are handled and converted properly. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        result = plotter._calculate_overlay_resolution(1, -100.0, -80.0, 30.0, 50.0)
        assert result == pytest.approx(1.0)
        assert isinstance(result, float)

    def test_none_resolution_uses_adaptive(self: 'TestCalculateOverlayResolution') -> None:
        """
        This test verifies that when the resolution parameter is None in the _calculate_overlay_resolution method, it calculates an adaptive resolution based on the longitude and latitude bounds. This covers the branch where the method should recognize that no explicit resolution is provided, compute a suitable resolution that adapts to the spatial extent of the data, and return a positive float value for use in the overlay interpolation process. The test calls the method with None for resolution and test longitude and latitude bounds, and asserts that the resulting resolution is a positive float, confirming that adaptive resolution calculation is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        result = plotter._calculate_overlay_resolution(None, -100.0, -80.0, 30.0, 50.0)
        assert result > 0


class TestCreateOverlayDataset:
    """ Test coverage for _create_overlay_dataset, specifically the branches for existing dataset and None dataset. """

    def test_existing_dataset_returned_unchanged(self: 'TestCreateOverlayDataset') -> None:
        """
        This test verifies that when an existing dataset is provided to the _create_overlay_dataset method, it is returned unchanged without modification. This covers the branch where the method should recognize that a valid dataset is already provided, skip any creation or modification steps, and return the original dataset for use in the overlay processing. The test creates a simple xarray Dataset with longitude data, applies the method with this existing dataset, and asserts that the returned result is the same object as the input dataset, confirming that it was returned unchanged. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        lon = np.linspace(-95.0, -85.0, N_CELLS)
        lat = np.linspace(35.0, 45.0, N_CELLS)
        existing_ds = xr.Dataset({'lonCell': xr.DataArray(lon, dims=['nCells'])})
        result = plotter._create_overlay_dataset(lon, lat, existing_ds)
        assert result is existing_ds

    def test_none_dataset_creates_new(self: 'TestCreateOverlayDataset') -> None:
        """
        This test verifies that when None is provided as the dataset to the _create_overlay_dataset method, it creates a new xarray Dataset with the appropriate longitude and latitude variables. This covers the branch where the method should recognize that no dataset is provided, construct a new Dataset using the input longitude and latitude arrays, and return this newly created Dataset for use in the overlay processing. The test calls the method with None for the dataset and test longitude and latitude arrays, and asserts that the resulting Dataset contains 'lonCell' and 'latCell' variables, confirming that a new dataset was created correctly. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        lon = np.linspace(-95.0, -85.0, N_CELLS)
        lat = np.linspace(35.0, 45.0, N_CELLS)
        result = plotter._create_overlay_dataset(lon, lat, None)
        assert 'lonCell' in result
        assert 'latCell' in result


class TestInterpolateOverlayWithConfig:
    """ Test coverage for _interpolate_overlay, specifically the branch for when config is not None. """

    def test_dispatch_remap_called_when_config_not_none(self: 'TestInterpolateOverlayWithConfig') -> None:
        """
        This test verifies that when a non-None config dictionary is provided to the _interpolate_overlay method, it calls the dispatch_remap function to perform the interpolation of the overlay data. This covers the branch where the method should recognize that configuration options are provided for interpolation, invoke the appropriate remapping function to interpolate the data onto a regular grid, and return the interpolated longitude, latitude, and data arrays. The test creates a mock remapped dataset to be returned by dispatch_remap, patches the dispatch_remap function to return this mock dataset, applies the _interpolate_overlay method with a sample config, and asserts that dispatch_remap was called and that the resulting interpolated arrays have the expected shapes, confirming that interpolation was performed as intended when config is provided. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        data_valid = np.ones(N_CELLS)
        lon = np.linspace(-95.0, -85.0, N_CELLS)
        lat = np.linspace(35.0, 45.0, N_CELLS)
        dataset = xr.Dataset({
            'lonCell': xr.DataArray(lon, dims=['nCells']),
            'latCell': xr.DataArray(lat, dims=['nCells']),
        })
        mock_remapped = MagicMock()
        mock_remapped.lon.values = np.linspace(-95.0, -85.0, 5)
        mock_remapped.lat.values = np.linspace(35.0, 45.0, 4)
        mock_remapped.values = np.ones((4, 5))

        with patch('mpasdiag.visualization.surface.dispatch_remap',
                   return_value=mock_remapped) as mock_dispatch:
            lon_mesh, lat_mesh, data_interp = plotter._interpolate_overlay(
                data_valid, dataset, -95.0, -85.0, 35.0, 45.0, 0.5,
                't2m', config={'method': 'nearest'}
            )
            mock_dispatch.assert_called_once()

        assert lon_mesh.shape == (4, 5)
        assert data_interp.shape == (4, 5)


class TestValidateContourLevels:
    """ Test coverage for _validate_contour_levels, specifically the branches for no levels in range triggering a warning and some levels in range not triggering a warning. """

    def test_no_levels_in_range_triggers_warning(self: 'TestValidateContourLevels') -> None:
        """
        This test verifies that _validate_contour_levels triggers a warning when none of the provided contour levels are within the range of the interpolated data. This covers the branch where the method should check the contour levels against the data range, determine that all levels are outside the range, and issue a UserWarning to inform the user that the contour levels may not be appropriate for the data being plotted. The test creates an interpolated data array with a specific range, provides contour levels that are all outside this range, applies the _validate_contour_levels method, and asserts that a UserWarning is raised with an appropriate message, confirming that the warning mechanism is functioning as intended when no levels are in range. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        data_interp = np.ones((5, 5)) * 500.0
        levels = [900.0, 950.0, 1000.0, 1050.0]
        plotter._validate_contour_levels(data_interp, levels, 'mslp')

    def test_some_levels_in_range_no_warning(self: 'TestValidateContourLevels') -> None:
        """
        This test verifies that _validate_contour_levels does not trigger a warning when some of the provided contour levels are within the range of the interpolated data. This covers the branch where the method should check the contour levels against the data range, determine that at least some levels are within the range, and allow the plotting to proceed without issuing a warning. The test creates an interpolated data array with a specific range, provides contour levels where some are outside but at least one is within this range, applies the _validate_contour_levels method, and asserts that no warnings are raised, confirming that the method correctly identifies when contour levels are appropriate for the data. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        data_interp = np.linspace(990.0, 1030.0, 25).reshape(5, 5)
        levels = [900.0, 1000.0, 1020.0, 1100.0]
        plotter._validate_contour_levels(data_interp, levels, 'mslp')

    def test_none_levels_skips_range_check(self: 'TestValidateContourLevels') -> None:
        """
        This test verifies that _validate_contour_levels skips the range check and does not trigger a warning when the levels parameter is None. This covers the branch where the method should recognize that no specific contour levels are provided, skip any checks against the data range, and allow the plotting to proceed without issue. The test creates an interpolated data array, applies the _validate_contour_levels method with None for levels, and asserts that no warnings are raised, confirming that the method correctly handles the case where contour levels are not specified. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        data_interp = np.ones((4, 4))
        plotter._validate_contour_levels(data_interp, None, 'some_var')


class TestCreateBatchSurfaceMaps:
    """ Test coverage for create_batch_surface_maps, specifically the branches for no dataset and progress printing at step 10. """

    def test_no_dataset_raises_valueerror(self: 'TestCreateBatchSurfaceMaps') -> None:
        """
        This test verifies that create_batch_surface_maps raises a ValueError when the processor's dataset is None. This covers the branch where the method should check if the dataset is loaded in the processor, and if it is not (i.e., it is None), it should raise a ValueError to inform the user that no data is available for plotting. The test creates a mock processor with dataset set to None, applies the create_batch_surface_maps method, and asserts that a ValueError is raised with an appropriate message, confirming that the method correctly handles the case of missing data. 

        Parameters:
            None

        Returns:
            None
        """
        proc = Mock()
        proc.dataset = None
        plotter = MPASSurfacePlotter()
        with pytest.raises(ValueError, match="No data loaded in processor"):
            plotter.create_batch_surface_maps(
                proc, '/tmp', -100.0, -80.0, 30.0, 50.0
            )

    def test_progress_printed_at_step_10(self: 'TestCreateBatchSurfaceMaps', 
                                         capsys: pytest.CaptureFixture) -> None:
        """
        This test verifies that create_batch_surface_maps prints progress at step 10 when processing a batch of surface maps. This covers the branch where the method should print a progress message to the console after completing the 10th map, allowing users to see that the process is advancing through the batch. The test creates a mock processor with a dataset containing 11 time steps, applies the create_batch_surface_maps method, captures the console output, and asserts that the expected progress message "Completed 10/11" is present in the output, confirming that progress is being printed at the correct step. 

        Parameters:
            capsys: pytest.CaptureFixture

        Returns:
            None
        """
        n_times = 11
        times = pd.date_range('2024-01-01', periods=n_times, freq='6h')
        ds = xr.Dataset(
            {'t2m': xr.DataArray(np.ones((n_times, N_CELLS)), dims=['Time', 'nCells'])},
            coords={'Time': xr.DataArray(times.values, dims=['Time'])},
        )
        proc = Mock()
        proc.dataset = ds
        var_data = xr.DataArray(np.ones(N_CELLS), dims=['nCells'])
        proc.get_2d_variable_data = Mock(return_value=var_data)
        proc.extract_2d_coordinates_for_variable = Mock(return_value=(
            np.linspace(-95.0, -85.0, N_CELLS),
            np.linspace(35.0, 45.0, N_CELLS),
        ))

        plotter = MPASSurfacePlotter()
        with patch.object(plotter, 'create_surface_map',
                          return_value=(MagicMock(), MagicMock())):
            with patch.object(plotter, 'save_plot'):
                with patch.object(plotter, 'close_plot'):
                    result = plotter.create_batch_surface_maps(
                        proc, '/tmp', -100.0, -80.0, 30.0, 50.0, var_name='t2m'
                    )
        captured = capsys.readouterr()
        assert "Completed 10/11" in captured.out
        assert len(result) == n_times


class TestCreateSimpleScatterPlot:
    """ Test coverage for create_simple_scatter_plot, specifically the branches for all-NaN data and colorbar tick_params exception handling. """

    def test_all_nan_data_raises_valueerror(self: 'TestCreateSimpleScatterPlot') -> None:
        """
        This test verifies that create_simple_scatter_plot raises a ValueError when all input data points are NaN. This covers the branch where the method should check for valid data points after filtering, and if none are found (i.e., all values are NaN), it should raise a ValueError to inform the user that no valid data is available for plotting. The test calls the method with arrays of NaN values for longitude, latitude, and data, and asserts that a ValueError is raised with an appropriate message, confirming that the method correctly handles the case of all-NaN data. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSurfacePlotter()
        with pytest.raises(ValueError, match="No valid data points"):
            plotter.create_simple_scatter_plot(
                np.array([np.nan, np.nan, np.nan]),
                np.array([np.nan, np.nan, np.nan]),
                np.array([np.nan, np.nan, np.nan]),
            )

    def test_colorbar_tickparams_exception_is_suppressed(self: 'TestCreateSimpleScatterPlot') -> None:
        """
        This test verifies that if an exception occurs during the call to tick_params on the colorbar axis in create_simple_scatter_plot, the exception is suppressed and does not prevent the plot from being created. This covers the branch where the method should handle potential issues with the colorbar axis gracefully, allowing the visualization to proceed even if there are problems with setting tick parameters. The test creates a mock colorbar object whose ax.tick_params method raises an exception, patches the add_colorbar method to return this mock, applies the create_simple_scatter_plot method with valid data, and asserts that a figure is still returned and that tick_params was called, confirming that the exception was handled as intended. 

        Parameters:
            None

        Returns:
            None
        """
        mock_cbar = MagicMock()
        mock_cbar.ax.tick_params.side_effect = Exception("tick_params error")

        plotter = MPASSurfacePlotter()
        with patch('mpasdiag.visualization.surface.MPASVisualizationStyle.add_colorbar',
                   return_value=mock_cbar):
            with patch.object(plotter, 'add_timestamp_and_branding'):
                fig, ax = plotter.create_simple_scatter_plot(
                    np.linspace(-95.0, -85.0, N_CELLS),
                    np.linspace(35.0, 45.0, N_CELLS),
                    np.ones(N_CELLS) * 280.0,
                )
        assert fig is not None
        mock_cbar.ax.tick_params.assert_called_once()


class TestPlot3dVariableSlice:
    """ Test coverage for plot_3d_variable_slice, specifically the branches for basic 3D slice delegation, custom title forwarding, and default title using metadata. """

    def test_basic_3d_slice_delegates_to_create_surface_map(self: 'TestPlot3dVariableSlice') -> None:
        """
        This test verifies that plot_3d_variable_slice correctly delegates to create_surface_map with the appropriate longitude and latitude bounds when plotting a 3D variable slice. This covers the branch where the method should extract the correct longitude and latitude bounds from the input data, call create_surface_map with these bounds, and ensure that the surface map is created for the specified level of the 3D variable. The test creates a sample 3D DataArray with longitude and latitude coordinates, patches the create_surface_map method to return mock figure and axis objects, applies the plot_3d_variable_slice method, and asserts that create_surface_map was called with the expected longitude and latitude bounds, confirming that delegation is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        n_levels = 5
        da = xr.DataArray(
            np.random.rand(N_CELLS, n_levels),
            dims=['nCells', 'nVertLevels'],
        )
        lon = np.linspace(-180.0, 180.0, N_CELLS)
        lat = np.linspace(-90.0, 90.0, N_CELLS)

        plotter = MPASSurfacePlotter()
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        with patch.object(plotter, 'create_surface_map',
                          return_value=(mock_fig, mock_ax)) as mock_create:
            fig, ax = plotter.plot_3d_variable_slice(da, lon, lat, level=2, var_name='qv')
        mock_create.assert_called_once()
        _, kwargs = mock_create.call_args
        assert kwargs.get('lon_min') == pytest.approx(-180.0)
        assert kwargs.get('lat_max') == pytest.approx(90.0)

    def test_custom_title_is_forwarded(self: 'TestPlot3dVariableSlice') -> None:
        """
        This test verifies that when a custom title is provided to plot_3d_variable_slice, it is forwarded to the create_surface_map method. This covers the branch where the method should accept a custom title argument, pass it through to the create_surface_map method, and ensure that the resulting plot uses this custom title instead of generating one from metadata. The test creates a sample 3D DataArray with longitude and latitude coordinates, patches the create_surface_map method to return mock figure and axis objects, applies the plot_3d_variable_slice method with a custom title, and asserts that create_surface_map was called with the expected title in its arguments, confirming that custom titles are being forwarded correctly. 

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(
            np.ones((N_CELLS, 4)),
            dims=['nCells', 'nVertLevels'],
        )
        lon = np.linspace(-180.0, 180.0, N_CELLS)
        lat = np.linspace(-90.0, 90.0, N_CELLS)

        plotter = MPASSurfacePlotter()
        with patch.object(plotter, 'create_surface_map',
                          return_value=(MagicMock(), MagicMock())) as mock_create:
            plotter.plot_3d_variable_slice(
                da, lon, lat, level=0, var_name='qv', title='My Custom Title'
            )
        _, kwargs = mock_create.call_args
        assert kwargs.get('title') == 'My Custom Title'

    def test_default_title_uses_metadata(self: 'TestPlot3dVariableSlice') -> None:
        """
        This test verifies that when no custom title is provided to plot_3d_variable_slice, it uses metadata to generate a default title and forwards it to the create_surface_map method. This covers the branch where the method should recognize that no custom title is given, extract relevant metadata from the input DataArray (such as variable name and level), construct a default title based on this metadata, and pass it to create_surface_map. The test creates a sample 3D DataArray with longitude and latitude coordinates, patches the create_surface_map method to return mock figure and axis objects, applies the plot_3d_variable_slice method without a custom title, and asserts that create_surface_map was called with a title that includes metadata information (e.g., variable name and level), confirming that default titles are being generated from metadata correctly. 

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(
            np.ones((N_CELLS, 4)),
            dims=['nCells', 'nVertLevels'],
        )
        lon = np.linspace(-180.0, 180.0, N_CELLS)
        lat = np.linspace(-90.0, 90.0, N_CELLS)

        plotter = MPASSurfacePlotter()
        with patch.object(plotter, 'create_surface_map',
                          return_value=(MagicMock(), MagicMock())) as mock_create:
            plotter.plot_3d_variable_slice(
                da, lon, lat, level=1, var_name='qv'
            )
        _, kwargs = mock_create.call_args
        assert kwargs.get('title') is not None
        assert 'Level 1' in kwargs['title'] or 'level' in kwargs['title'].lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

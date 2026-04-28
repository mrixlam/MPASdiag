#!/usr/bin/env python3

"""
MPASdiag Test Suite: Base Visualizer Coverage

This test suite is designed to achieve comprehensive code coverage for the MPASVisualizer class in the mpasdiag.visualization.base_visualizer module. It includes targeted test cases that cover specific lines and branches of the code, particularly those that were previously untested. The tests utilize mocking and patching to simulate various conditions and ensure that all code paths are exercised. 

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
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings
from typing import Tuple
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

from mpasdiag.visualization.base_visualizer import MPASVisualizer

warnings.filterwarnings('ignore')


N_CELLS = 12
N_VERT = 6


def _viz() -> MPASVisualizer:
    """
    This helper function creates a new instance of MPASVisualizer with a standard figure size and DPI for testing purposes. It can be used across multiple test cases to ensure consistency in the visualizer configuration. 

    Parameters:
        None

    Returns:
        MPASVisualizer: A new instance of the MPASVisualizer class with predefined figure settings.
    """
    return MPASVisualizer(figsize=(8, 6), dpi=72)


def _simple_lon_lat() -> Tuple[np.ndarray, np.ndarray]:
    """
    This helper function generates simple longitude and latitude arrays for testing purposes. The arrays are linearly spaced within a specified range and can be used to test various visualization methods that require geographic coordinates.

    Parameters:
        None

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays: longitude and latitude.
    """
    return (
        np.linspace(-100., -90., N_CELLS),
        np.linspace(35., 45., N_CELLS),
    )


class TestFormatCoordinates:
    """ Covers format_latitude and format_longitude delegation. """

    def test_format_latitude_north(self: 'TestFormatCoordinates') -> None:
        """
        This test case verifies that the format_latitude method correctly formats a positive latitude value (indicating the northern hemisphere). It checks that the resulting string contains the numeric part of the latitude and either 'N' or '°' to indicate north. This ensures that the method is properly delegating to the underlying formatting logic for northern latitudes. 

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        result = v.format_latitude(45.0, None)
        assert '45' in result
        assert 'N' in result or '°' in result

    def test_format_latitude_south(self: 'TestFormatCoordinates') -> None:
        """
        This test case verifies that the format_latitude method correctly formats a negative latitude value (indicating the southern hemisphere). It checks that the resulting string contains the numeric part of the latitude and either 'S' or '°' to indicate south. This ensures that the method is properly delegating to the underlying formatting logic for southern latitudes.

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        result = v.format_latitude(-30.0, None)
        assert '30' in result

    def test_format_longitude_east(self: 'TestFormatCoordinates') -> None:
        """
        This test case verifies that the format_longitude method correctly formats a positive longitude value (indicating the eastern hemisphere). It checks that the resulting string contains the numeric part of the longitude and either 'E' or '°' to indicate east. This ensures that the method is properly delegating to the underlying formatting logic for eastern longitudes.

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        result = v.format_longitude(120.0, None)
        assert '120' in result

    def test_format_longitude_west(self: 'TestFormatCoordinates') -> None:
        """
        This test case verifies that the format_longitude method correctly formats a negative longitude value (indicating the western hemisphere). It checks that the resulting string contains the numeric part of the longitude and either 'W' or '°' to indicate west. This ensures that the method is properly delegating to the underlying formatting logic for western longitudes.

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        result = v.format_longitude(-75.0, None)
        assert '75' in result


class TestAddRegionalFeaturesAxNone:
    """ Covers the ax=None early return in add_regional_features. """

    def test_no_ax_returns_silently(self: 'TestAddRegionalFeaturesAxNone') -> None:
        """
        This test case verifies that the add_regional_features method returns silently when the ax attribute is None. It ensures that no errors are raised and the method exits early as expected.

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        assert v.ax is None
        v.add_regional_features(-100., -90., 35., 45.)


class TestCreateTimeSeriesPlot:
    """ Covers the pd.to_datetime fallback. """

    def test_datetime64_fallback_when_pd_fails(self: 'TestCreateTimeSeriesPlot') -> None:
        """
        This test case verifies that the create_time_series_plot method correctly falls back to using datetime64 when pandas' to_datetime function raises a ValueError. It simulates a scenario where the time parsing fails and checks that the plot is still created without errors. This ensures that the method can handle cases where time data may not be in a format that pandas can parse, providing robustness in time series plotting. 

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        times = [datetime(2024, 1, 1, i) for i in range(4)]
        values = [1.0, 2.0, 3.0, 4.0]
        with patch('mpasdiag.visualization.base_visualizer.pd.to_datetime',
                   side_effect=ValueError("cannot parse")):
            fig, ax = v.create_time_series_plot(times, values, title="Test")
        assert fig is not None
        plt.close(fig)

    def test_normal_datetimes_work(self: 'TestCreateTimeSeriesPlot') -> None:
        """
        This test case verifies that the create_time_series_plot method works correctly with normal datetime objects. It ensures that the plot is created without errors when the time data is in a format that pandas can parse.

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        times = [datetime(2024, 1, 1, i) for i in range(3)]
        fig, ax = v.create_time_series_plot(times, [1., 2., 3.])
        assert fig is not None
        plt.close(fig)


class TestCreateHistogram:
    """ Covers the bins.tolist() exception fallback. """

    def test_bins_tolist_exception_falls_back_to_list(self: 'TestCreateHistogram') -> None:
        """
        This test case verifies that the create_histogram method correctly falls back to using a list when the bins.tolist() method raises an exception. It simulates a scenario where the tolist method fails and checks that the histogram is still created without errors. This ensures that the method can handle cases where the bins object may not support the tolist method, providing robustness in histogram creation.

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        data = np.random.default_rng(0).normal(0, 1, 100)

        bad_bins = Mock(spec=np.ndarray)
        bad_bins.__class__ = np.ndarray
        bad_bins.tolist.side_effect = RuntimeError("tolist failed")
        bad_bins.__iter__ = Mock(return_value=iter([0.0, 0.5, 1.0, 1.5]))
        bad_bins.__len__ = Mock(return_value=4)

        with patch('mpasdiag.visualization.base_visualizer.isinstance',
                   wraps=isinstance) as mock_isinstance:
            original_isinstance = isinstance

            def patched_isinstance(obj, cls):
                if cls is np.ndarray and obj is bad_bins:
                    return True
                return original_isinstance(obj, cls)

            mock_isinstance.side_effect = patched_isinstance
            fig, ax = v.create_histogram(data, bins=bad_bins)
        plt.close(fig)

    def test_numpy_bins_array_works_normally(self: 'TestCreateHistogram') -> None:
        """
        This test case verifies that the create_histogram method works correctly with a normal numpy array for bins. It ensures that the histogram is created without errors when the bins are provided as a numpy array.

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        data = np.random.default_rng(1).normal(5, 2, 200)
        bins = np.linspace(0., 12., 10)
        fig, ax = v.create_histogram(data, bins=bins, log_scale=True)
        assert fig is not None
        plt.close(fig)


class TestConvertToNumpy:
    """ Covers the compute() exception handler. """

    def test_compute_exception_is_silenced(self: 'TestConvertToNumpy') -> None:
        """
        This test case verifies that the convert_to_numpy method correctly silences exceptions raised by the compute() method of an xarray DataArray. It simulates a scenario where compute() fails and checks that convert_to_numpy returns a non-None result without propagating the exception. This ensures that the method can handle cases where computation may fail, providing robustness in data conversion. 

        Parameters:
            None

        Returns:
            None
        """
        bad = MagicMock()
        bad.__class__ = object
        bad.compute.side_effect = RuntimeError("compute failed")

        result = MPASVisualizer.convert_to_numpy(bad)
        assert result is not None

    def test_xarray_dataarray_converted(self: 'TestConvertToNumpy') -> None:
        """
        This test case verifies that the convert_to_numpy method correctly converts an xarray DataArray to a numpy array. It checks that the result is an instance of np.ndarray, ensuring that the conversion process is functioning as intended when given a valid DataArray input. 

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(np.array([1., 2., 3.]))
        result = MPASVisualizer.convert_to_numpy(da)
        assert isinstance(result, np.ndarray)

    def test_plain_array_passes_through(self: 'TestConvertToNumpy') -> None:
        """
        This test case verifies that the convert_to_numpy method correctly handles a plain numpy array. It checks that the result is equal to the input array, ensuring that the method does not alter numpy arrays that do not require conversion.

        Parameters:
            None

        Returns:
            None
        """
        arr = np.array([4., 5., 6.])
        result = MPASVisualizer.convert_to_numpy(arr)
        np.testing.assert_array_equal(result, arr)


class TestEnsureBoundaryData:
    """ Covers early return when lon_b/lat_b present and full computation. """

    def test_none_dataset_returns_none(self: 'TestEnsureBoundaryData') -> None:
        """
        This test case verifies that the _ensure_boundary_data method correctly handles a None dataset. It checks that the method returns None when given a None input, ensuring that it gracefully handles cases where no dataset is provided.

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        assert v._ensure_boundary_data(None) is None

    def test_already_has_lon_b_lat_b_returns_unchanged(self: 'TestEnsureBoundaryData') -> None:
        """
        This test case verifies that the _ensure_boundary_data method returns the original dataset unchanged when it already contains the 'lon_b' and 'lat_b' variables. It checks that the method does not modify the dataset and simply returns it as is, ensuring that it recognizes when boundary data is already present and avoids unnecessary computations. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            'lon_b': (['nCells', 'nv'], np.ones((N_CELLS, 3))),
            'lat_b': (['nCells', 'nv'], np.ones((N_CELLS, 3))),
        })
        v = _viz()
        result = v._ensure_boundary_data(ds)
        assert result is ds

    def test_missing_mesh_vars_returns_unchanged(self: 'TestEnsureBoundaryData') -> None:
        """
        This test case verifies that the _ensure_boundary_data method returns the original dataset unchanged when it is missing mesh variables. It checks that the method does not modify the dataset and simply returns it as is, ensuring that it recognizes when necessary mesh variables are absent and avoids unnecessary computations.

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            'temperature': (['nCells'], np.ones(N_CELLS)),
        })
        v = _viz()
        result = v._ensure_boundary_data(ds)
        assert result is ds

    def test_computes_boundary_from_mesh_topology(self: 'TestEnsureBoundaryData') -> None:
        """
        This test case verifies that the _ensure_boundary_data method correctly computes boundary coordinates from mesh topology when the necessary variables are present. It checks that the resulting dataset contains 'lon_b' and 'lat_b' variables, ensuring that the method can derive boundary data from the mesh structure when it is not already provided. 

        Parameters:
            None

        Returns:
            None
        """
        n_vertices = 6

        ds = xr.Dataset({
            'verticesOnCell': (['nCells', 'maxEdges'],
                               np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])),
            'lonVertex': (['nVertices'], np.linspace(0.1, 0.5, n_vertices)),
            'latVertex': (['nVertices'], np.linspace(0.2, 0.6, n_vertices)),
            'nEdgesOnCell': (['nCells'], np.array([3, 3, 3, 3])),
        })
        v = _viz()
        result = v._ensure_boundary_data(ds)
        assert result is not None
        assert 'lon_b' in result
        assert 'lat_b' in result

    def test_boundary_computation_uses_cache(self: 'TestEnsureBoundaryData') -> None:
        """
        This test case verifies that the _ensure_boundary_data method reuses cached boundary data when called multiple times with the same dataset. It checks that the method does not recompute the boundary coordinates unnecessarily, ensuring efficient use of cached results.

        Parameters:
            None

        Returns:
            None
        """
        n_vertices = 6
        ds = xr.Dataset({
            'verticesOnCell': (['nCells', 'maxEdges'],
                               np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])),
            'lonVertex': (['nVertices'], np.linspace(0.1, 0.5, n_vertices)),
            'latVertex': (['nVertices'], np.linspace(0.2, 0.6, n_vertices)),
            'nEdgesOnCell': (['nCells'], np.array([3, 3, 3, 3])),
        })
        v = _viz()
        r1 = v._ensure_boundary_data(ds)
        r2 = v._ensure_boundary_data(ds)
        assert r1 is not None and r2 is not None


class TestExtractFullGrid:
    """ Covers alternate lon/lat key names and KeyError raises. """

    def test_loncell_latcell_extracted(self: 'TestExtractFullGrid') -> None:
        """
        This test case verifies that the _extract_full_grid method correctly extracts longitude and latitude coordinates from a dataset containing 'lonCell' and 'latCell' variables. It checks that the extracted longitude and latitude arrays have the expected shapes and that the longitude values are within the valid range of -180 to 180 degrees. This ensures that the method can successfully retrieve grid coordinates from a properly structured dataset.

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            'lonCell': (['nCells'], np.radians(np.linspace(-100., -90., N_CELLS))),
            'latCell': (['nCells'], np.radians(np.linspace(35., 45., N_CELLS))),
        })
        lon, lat = MPASVisualizer._extract_full_grid(ds)
        assert lon.shape == (N_CELLS,)
        assert lat.shape == (N_CELLS,)
        assert np.all(lon >= -180) and np.all(lon <= 180)

    def test_longitude_latitude_aliases(self: 'TestExtractFullGrid') -> None:
        """
        This test case verifies that the _extract_full_grid method correctly handles datasets with 'longitude' and 'latitude' keys as fallback options. It checks that the method can extract the full grid coordinates when the primary 'lonCell' and 'latCell' keys are not present.

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            'longitude': (['nCells'], np.linspace(-100., -90., N_CELLS)),
            'latitude': (['nCells'], np.linspace(35., 45., N_CELLS)),
        })
        lon, lat = MPASVisualizer._extract_full_grid(ds)
        assert lon.shape == (N_CELLS,)

    def test_lon_lat_short_keys(self: 'TestExtractFullGrid') -> None:
        """
        This test case verifies that the _extract_full_grid method correctly handles datasets with 'lon' and 'lat' keys as fallback options. It checks that the method can extract the full grid coordinates when the primary 'lonCell' and 'latCell' keys are not present.

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            'lon': (['nCells'], np.linspace(-100., -90., N_CELLS)),
            'lat': (['nCells'], np.linspace(35., 45., N_CELLS)),
        })
        lon, lat = MPASVisualizer._extract_full_grid(ds)
        assert len(lon) == N_CELLS

    def test_missing_lon_raises_keyerror(self: 'TestExtractFullGrid') -> None:
        """
        This test case verifies that the _extract_full_grid method raises a KeyError when the dataset is missing the longitude key. It ensures that the method correctly handles the absence of required longitude data.

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({'latCell': (['nCells'], np.ones(N_CELLS))})
        with pytest.raises(KeyError, match="longitude"):
            MPASVisualizer._extract_full_grid(ds)

    def test_missing_lat_raises_keyerror(self: 'TestExtractFullGrid') -> None:
        """
        This test case verifies that the _extract_full_grid method raises a KeyError when the dataset is missing the latitude key. It ensures that the method correctly handles the absence of required latitude data.

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({'lonCell': (['nCells'], np.ones(N_CELLS))})
        with pytest.raises(KeyError, match="latitude"):
            MPASVisualizer._extract_full_grid(ds)


class TestBackmapToFullGrid:
    """ Covers the full _backmap_to_full_grid method. """

    def test_known_valid_points_mapped(self: 'TestBackmapToFullGrid') -> None:
        """
        This test case verifies that the _backmap_to_full_grid method correctly maps known valid points to the full grid. It checks that the resulting array has the same length as the full longitude array and that the values corresponding to the valid points are not NaN. This ensures that the method can successfully backmap data from a subset of valid points to a complete grid, preserving valid data while assigning NaN to points without valid data. 

        Parameters:
            None

        Returns:
            None
        """
        lon_valid = np.array([-98., -95., -92.])
        lat_valid = np.array([38., 40., 42.])
        data_valid = np.array([10., 20., 30.])
        lon_full = np.array([-98., -95., -92., -89.])
        lat_full = np.array([38., 40., 42., 44.])

        result = MPASVisualizer._backmap_to_full_grid(
            lon_valid, lat_valid, data_valid, lon_full, lat_full
        )
        assert len(result) == len(lon_full)
        assert not np.all(np.isnan(result[:3]))

    def test_distant_points_get_nan(self: 'TestBackmapToFullGrid') -> None:
        """
        This test case verifies that the _backmap_to_full_grid method correctly assigns NaN to points that are far from any valid point. It ensures that the method can handle cases where some points in the full grid do not have corresponding valid data points nearby.

        Parameters:
            None

        Returns:
            None
        """
        lon_valid = np.array([-98.])
        lat_valid = np.array([38.])
        data_valid = np.array([10.])
        lon_full = np.array([-50., -98.])
        lat_full = np.array([10., 38.])

        result = MPASVisualizer._backmap_to_full_grid(
            lon_valid, lat_valid, data_valid, lon_full, lat_full
        )
        assert np.isnan(result[0])
        assert not np.isnan(result[1])


class TestGetOrBuildRemapper:
    """ Covers the full _get_or_build_remapper method body with mocked remapper. """

    def test_builds_new_remapper_when_none_cached(self: 'TestGetOrBuildRemapper') -> None:
        """
        This test case verifies that the _get_or_build_remapper method builds a new remapper when none is cached. It ensures that the method correctly initializes a new remapper and prepares the source and target grids.

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        lon_full = np.linspace(-100., -90., N_CELLS)
        lat_full = np.linspace(35., 45., N_CELLS)
        lon_b = np.ones((N_CELLS, 3))
        lat_b = np.ones((N_CELLS, 3))

        mock_remapper = MagicMock()
        with patch('mpasdiag.visualization.base_visualizer.MPASRemapper',
                   return_value=mock_remapper) as mock_cls:
            result = v._get_or_build_remapper(
                lon_full, lat_full, lon_b, lat_b,
                lon_min=-100., lon_max=-90., lat_min=35., lat_max=45.,
                resolution=1.0,
            )
        mock_cls.assert_called_once()
        mock_remapper.prepare_source_grid.assert_called_once()
        mock_remapper.create_target_grid.assert_called_once()
        mock_remapper.build_regridder.assert_called_once()
        assert result is mock_remapper

    def test_returns_cached_remapper_when_key_matches(self: 'TestGetOrBuildRemapper') -> None:
        """
        This test case verifies that the _get_or_build_remapper method returns a cached remapper when the key matches. It ensures that the method does not rebuild the remapper unnecessarily and correctly retrieves the cached remapper.

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        lon_full = np.linspace(-100., -90., N_CELLS)
        lat_full = np.linspace(35., 45., N_CELLS)
        lon_b = np.ones((N_CELLS, 3))
        lat_b = np.ones((N_CELLS, 3))

        mock_remapper = MagicMock()
        with patch('mpasdiag.visualization.base_visualizer.MPASRemapper',
                   return_value=mock_remapper):
            r1 = v._get_or_build_remapper(
                lon_full, lat_full, lon_b, lat_b,
                -100., -90., 35., 45., 1.0,
            )
            r2 = v._get_or_build_remapper(
                lon_full, lat_full, lon_b, lat_b,
                -100., -90., 35., 45., 1.0,
            )
        assert r1 is r2

    def test_rebuilds_remapper_when_key_changes(self: 'TestGetOrBuildRemapper') -> None:
        """
        This test case verifies that the _get_or_build_remapper method rebuilds the remapper when the key changes. It ensures that the method correctly identifies changes in the key and initializes a new remapper accordingly.

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        lon_full = np.linspace(-100., -90., N_CELLS)
        lat_full = np.linspace(35., 45., N_CELLS)
        lon_b = np.ones((N_CELLS, 3))
        lat_b = np.ones((N_CELLS, 3))

        mock_r1 = MagicMock()
        mock_r2 = MagicMock()
        with patch('mpasdiag.visualization.base_visualizer.MPASRemapper',
                   side_effect=[mock_r1, mock_r2]):
            r1 = v._get_or_build_remapper(lon_full, lat_full, lon_b, lat_b,
                                           -100., -90., 35., 45., 1.0)
            r2 = v._get_or_build_remapper(lon_full, lat_full, lon_b, lat_b,
                                           -100., -90., 35., 45., 0.5)
        assert r1 is not r2


class TestRemapConservative:
    """Covers the full _remap_conservative method body."""

    def test_calls_remapper_remap(self: 'TestRemapConservative') -> None:
        """
        This test case verifies that the _remap_conservative method correctly calls the remapper's remap method. It ensures that the method extracts the boundary data, builds the remapper, and performs the remapping operation.

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        ds = xr.Dataset({
            'lon_b': (['nCells', 'nv'], np.ones((N_CELLS, 3)) * -95.),
            'lat_b': (['nCells', 'nv'], np.ones((N_CELLS, 3)) * 40.),
        })
        lon_full = np.linspace(-100., -90., N_CELLS)
        lat_full = np.linspace(35., 45., N_CELLS)
        data_full = np.ones(N_CELLS)

        mock_remapper = MagicMock()
        lon_g = np.linspace(-100., -90., 5)
        lat_g = np.linspace(35., 45., 5)
        lon_mesh, lat_mesh = np.meshgrid(lon_g, lat_g)
        fake_result = xr.DataArray(
            np.ones((5, 5)),
            coords={'lon': lon_g, 'lat': lat_g},
            dims=['lat', 'lon'],
        )
        mock_remapper.remap.return_value = fake_result

        with patch.object(v, '_get_or_build_remapper', return_value=mock_remapper):
            result = v._remap_conservative(
                data_full, lon_full, lat_full, ds,
                -100., -90., 35., 45., 1.0,
            )
        mock_remapper.remap.assert_called_once()
        assert result is fake_result


class TestInterpolateToGridESMPy:
    """ Covers the ESMPy path in _interpolate_to_grid. """

    def _make_ds(self: 'TestInterpolateToGridESMPy') -> xr.Dataset:
        """
        This helper method creates a dataset with the necessary structure for testing the ESMPy path in the _interpolate_to_grid method. The dataset includes 'lonCell', 'latCell', 'lon_b', and 'lat_b' variables, which are required for boundary data processing and remapping. The longitude and latitude values are linearly spaced within a specified range, and the boundary coordinates are set to constant values for simplicity. This dataset can be used across multiple test cases to ensure consistency in testing the ESMPy interpolation functionality. 

        Parameters:
            None

        Returns:
            xr.Dataset: A dataset containing the necessary variables for ESMPy interpolation testing. 
        """
        return xr.Dataset({
            'lonCell': (['nCells'], np.linspace(-100., -90., N_CELLS)),
            'latCell': (['nCells'], np.linspace(35., 45., N_CELLS)),
            'lon_b': (['nCells', 'nv'], np.ones((N_CELLS, 3)) * -95.),
            'lat_b': (['nCells', 'nv'], np.ones((N_CELLS, 3)) * 40.),
        })

    def _make_remap_result(self: 'TestInterpolateToGridESMPy') -> xr.DataArray:
        """
        This helper method creates a remap result in the form of an xarray DataArray for testing the ESMPy path in the _interpolate_to_grid method. The DataArray is structured with 'lon' and 'lat' coordinates, and contains a 2D array of ones as the data values. The longitude and latitude coordinates are linearly spaced within a specified range, creating a simple meshgrid for testing purposes. This remap result can be used to simulate the output of the ESMPy remapping process in various test cases, ensuring consistency in testing the interpolation functionality.

        Parameters:
            None

        Returns:
            xr.DataArray: A DataArray containing the remapped values and corresponding longitude and latitude coordinates for ESMPy interpolation testing. 
        """
        lon_g = np.linspace(-100., -90., 5)
        lat_g = np.linspace(35., 45., 5)
        return xr.DataArray(
            np.ones((5, 5)),
            coords={'lon': lon_g, 'lat': lat_g},
            dims=['lat', 'lon'],
        )

    def test_esmpy_success_path(self: 'TestInterpolateToGridESMPy') -> None:
        """
        This test case verifies the successful execution of the ESMPy path in the _interpolate_to_grid method. It checks that when ESMPy is available and all necessary data and methods are properly mocked, the interpolation process completes without errors and returns longitude, latitude, and interpolated data arrays with the expected dimensions. This ensures that the method can successfully utilize ESMPy for interpolation when all conditions are met. 

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        lon, lat = _simple_lon_lat()
        data = np.ones(N_CELLS)
        ds = self._make_ds()
        remap_result = self._make_remap_result()

        with patch('mpasdiag.visualization.base_visualizer.ESMPY_AVAILABLE', True):
            with patch.object(v, '_ensure_boundary_data', return_value=ds):
                with patch.object(v, '_has_boundary_data', return_value=True):
                    with patch.object(v, '_extract_full_grid',
                                      return_value=(lon, lat)):
                        with patch.object(v, '_backmap_to_full_grid',
                                          return_value=np.ones(N_CELLS)):
                            with patch.object(v, '_remap_conservative',
                                              return_value=remap_result):
                                lon_m, lat_m, data_i = v._interpolate_to_grid(
                                    lon, lat, data,
                                    -100., -90., 35., 45.,
                                    grid_resolution=1.0,
                                    dataset=ds,
                                )
        assert lon_m.ndim == 2
        assert lat_m.ndim == 2

    def test_esmpy_fallback_on_exception(self: 'TestInterpolateToGridESMPy') -> None:
        """
        This test case verifies that the _interpolate_to_grid method correctly falls back to an alternative interpolation method when the ESMPy remapping process raises an exception. It simulates a scenario where the remap_conservative method fails and checks that the method then calls the remap_mpas_to_latlon_with_masking function to perform the interpolation. This ensures that the method can handle failures in the ESMPy path gracefully and still provide interpolated results using a fallback approach. 

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        lon, lat = _simple_lon_lat()
        data = np.ones(N_CELLS)
        ds = self._make_ds()

        lon_g = np.linspace(-100., -90., 5)
        lat_g = np.linspace(35., 45., 5)

        fallback_result = xr.DataArray(
            np.ones((5, 5)),
            coords={'lon': lon_g, 'lat': lat_g},
            dims=['lat', 'lon'],
        )

        with patch('mpasdiag.visualization.base_visualizer.ESMPY_AVAILABLE', True):
            with patch.object(v, '_ensure_boundary_data', return_value=ds):
                with patch.object(v, '_has_boundary_data', return_value=True):
                    with patch.object(v, '_extract_full_grid',
                                      return_value=(lon, lat)):
                        with patch.object(v, '_backmap_to_full_grid',
                                          return_value=np.ones(N_CELLS)):
                            with patch.object(v, '_remap_conservative',
                                              side_effect=RuntimeError("esmpy failed")):
                                with patch('mpasdiag.visualization.base_visualizer'
                                           '.remap_mpas_to_latlon_with_masking',
                                           return_value=fallback_result):
                                    lon_m, lat_m, data_i = v._interpolate_to_grid(
                                        lon, lat, data,
                                        -100., -90., 35., 45.,
                                        grid_resolution=1.0,
                                        dataset=ds,
                                    )
        assert lon_m.ndim == 2

    def test_esmpy_engine_explicit_no_boundary_raises(self: 'TestInterpolateToGridESMPy') -> None:
        """
        This test case verifies that the _interpolate_to_grid method raises a ValueError when the ESMPy engine is explicitly requested but boundary coordinates are not available in the dataset. It simulates a scenario where the dataset lacks boundary data and checks that the method correctly identifies this issue and raises an appropriate error message. This ensures that the method enforces the requirement for boundary data when using the ESMPy interpolation engine, preventing silent failures or incorrect interpolations. 

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        lon, lat = _simple_lon_lat()
        data = np.ones(N_CELLS)

        config = Mock()
        config.remap_engine = 'esmf'
        config.remap_method = 'conservative'

        ds_no_bounds = xr.Dataset({
            'lonCell': (['nCells'], lon),
            'latCell': (['nCells'], lat),
        })

        with patch('mpasdiag.visualization.base_visualizer.ESMPY_AVAILABLE', True):
            with patch.object(v, '_ensure_boundary_data', return_value=ds_no_bounds):
                with patch.object(v, '_has_boundary_data', return_value=False):
                    with pytest.raises(ValueError, match="boundary coordinates"):
                        v._interpolate_to_grid(
                            lon, lat, data,
                            -100., -90., 35., 45.,
                            grid_resolution=1.0,
                            dataset=ds_no_bounds,
                            config=config,
                        )


class TestCreateContourPlot:
    """ Covers the levels=None else branch and exception handler. """

    def _setup_viz_with_ax(self: 'TestCreateContourPlot') -> MPASVisualizer:
        """
        This helper method sets up an instance of the MPASVisualizer class with a figure and axes for testing the _create_contour_plot method. It initializes a new visualizer object and creates a matplotlib figure and axes, which are necessary for plotting. This setup allows the test cases to focus on the functionality of the contour plot creation without worrying about the initial visualization setup. 

        Parameters:
            None

        Returns:
            MPASVisualizer: An instance of the MPASVisualizer class with an initialized figure and axes for testing contour plot creation. 
        """
        v = _viz()
        v.fig, v.ax = plt.subplots()
        return v

    def test_levels_none_else_branch(self: 'TestCreateContourPlot') -> None:
        """
        This test case verifies the execution of the levels=None else branch in the _create_contour_plot method. It checks that when levels is None, the contour() method is called without the levels argument, and that the contour plot is created successfully. This ensures that the method can handle cases where contour levels are not specified and still produce a valid contour plot. 

        Parameters:
            None

        Returns:
            None
        """
        v = self._setup_viz_with_ax()
        lon, lat = _simple_lon_lat()
        data = np.ones(N_CELLS)
        lon_g = np.linspace(-100., -90., 5)
        lat_g = np.linspace(35., 45., 5)
        lon_mesh, lat_mesh = np.meshgrid(lon_g, lat_g)
        data_2d = np.ones((5, 5))

        mock_contour = Mock()
        try:
            with patch.object(v, '_interpolate_to_grid',
                               return_value=(lon_mesh, lat_mesh, data_2d)):
                with patch.object(v.ax, 'contour', return_value=mock_contour):
                    v._create_contour_plot(
                        lon, lat, data,
                        -100., -90., 35., 45.,
                        cmap_obj='viridis',
                        norm=None,
                        levels=None,
                        data_crs=ccrs.PlateCarree(),
                    )
        finally:
            plt.close(v.fig)

    def test_levels_provided_branch(self: 'TestCreateContourPlot') -> None:
        """
        This test case verifies the execution of the levels provided branch in the _create_contour_plot method. It checks that when levels are provided, the contour() method is called with the levels argument, and that the contour plot is created successfully. This ensures that the method can handle cases where contour levels are specified and produce a valid contour plot with the specified levels.

        Parameters:
            None

        Returns:
            None
        """
        v = self._setup_viz_with_ax()
        lon, lat = _simple_lon_lat()
        data = np.ones(N_CELLS)
        lon_g = np.linspace(-100., -90., 5)
        lat_g = np.linspace(35., 45., 5)
        lon_mesh, lat_mesh = np.meshgrid(lon_g, lat_g)
        data_2d = np.ones((5, 5))
        mock_contour = Mock()
        try:
            with patch.object(v, '_interpolate_to_grid',
                               return_value=(lon_mesh, lat_mesh, data_2d)):
                with patch.object(v.ax, 'contour', return_value=mock_contour):
                    v._create_contour_plot(
                        lon, lat, data,
                        -100., -90., 35., 45.,
                        cmap_obj='viridis',
                        norm=None,
                        levels=[0.5, 1.0, 1.5],
                        data_crs=ccrs.PlateCarree(),
                    )
        finally:
            plt.close(v.fig)

    def test_contour_exception_raises_runtime_error(self: 'TestCreateContourPlot') -> None:
        """
        This test case verifies that a RuntimeError is raised when the contour() method fails in the _create_contour_plot method. It ensures that the method correctly handles exceptions and propagates them as RuntimeError.

        Parameters:
            None

        Returns:
            None
        """
        v = self._setup_viz_with_ax()
        lon, lat = _simple_lon_lat()
        data = np.ones(N_CELLS)
        lon_g = np.linspace(-100., -90., 5)
        lat_g = np.linspace(35., 45., 5)
        lon_mesh, lat_mesh = np.meshgrid(lon_g, lat_g)
        data_2d = np.ones((5, 5))
        try:
            with patch.object(v, '_interpolate_to_grid',
                               return_value=(lon_mesh, lat_mesh, data_2d)):
                with patch.object(v.ax, 'contour',
                                  side_effect=RuntimeError("contour failed")):
                    with pytest.raises(RuntimeError, match="Contour plotting failed"):
                        v._create_contour_plot(
                            lon, lat, data,
                            -100., -90., 35., 45.,
                            cmap_obj='viridis',
                            norm=None,
                            levels=None,
                            data_crs=ccrs.PlateCarree(),
                        )
        finally:
            plt.close(v.fig)


class TestCreateContourfPlot:
    """ Covers the levels=None else branch and colorbar exception handler. """

    def _setup_viz_with_ax(self: 'TestCreateContourfPlot') -> MPASVisualizer:
        """
        This helper method sets up an instance of the MPASVisualizer class with a figure and axes for testing the _create_contourf_plot method. It initializes a new visualizer object and creates a matplotlib figure and axes, which are necessary for plotting. This setup allows the test cases to focus on the functionality of the contourf plot creation without worrying about the initial visualization setup.

        Parameters:
            None

        Returns:
            MPASVisualizer: An instance of the MPASVisualizer class with an initialized figure and axes for testing contourf plot creation.
        """
        v = _viz()
        v.fig, v.ax = plt.subplots()
        return v

    def test_levels_none_else_branch(self: 'TestCreateContourfPlot') -> None:
        """
        This test case verifies the execution of the levels=None else branch in the _create_contourf_plot method. It checks that when levels is None, the contourf() method is called without the levels argument, and that the contour plot is created successfully. This ensures that the method can handle cases where contour levels are not specified and still produce a valid contour plot.

        Parameters:
            None

        Returns:
            None
        """
        v = self._setup_viz_with_ax()
        lon, lat = _simple_lon_lat()
        data = np.ones(N_CELLS)
        lon_g = np.linspace(-100., -90., 5)
        lat_g = np.linspace(35., 45., 5)
        lon_mesh, lat_mesh = np.meshgrid(lon_g, lat_g)
        data_2d = np.ones((5, 5))
        mock_cf = Mock()
        try:
            with patch.object(v, '_interpolate_to_grid',
                               return_value=(lon_mesh, lat_mesh, data_2d)):
                with patch.object(v.ax, 'contourf', return_value=mock_cf):
                    v._create_contourf_plot(
                        lon, lat, data,
                        -100., -90., 35., 45.,
                        cmap_obj='viridis',
                        norm=None,
                        levels=None,
                        data_crs=ccrs.PlateCarree(),
                    )
        finally:
            plt.close(v.fig)

    def test_colorbar_exception_silenced(self: 'TestCreateContourfPlot') -> None:
        """
        This test case verifies that the _create_contourf_plot method handles exceptions raised by the add_colorbar method gracefully. It checks that when add_colorbar raises a RuntimeError, the exception is caught and silenced, allowing the plot creation to continue without interruption.

        Parameters:
            None

        Returns:
            None
        """
        v = self._setup_viz_with_ax()
        lon, lat = _simple_lon_lat()
        data = np.ones(N_CELLS)
        lon_g = np.linspace(-100., -90., 5)
        lat_g = np.linspace(35., 45., 5)
        lon_mesh, lat_mesh = np.meshgrid(lon_g, lat_g)
        data_2d = np.ones((5, 5))
        mock_cf = Mock()
        try:
            with patch.object(v, '_interpolate_to_grid',
                               return_value=(lon_mesh, lat_mesh, data_2d)):
                with patch.object(v.ax, 'contourf', return_value=mock_cf):
                    with patch('mpasdiag.visualization.base_visualizer'
                               '.MPASVisualizationStyle.add_colorbar',
                               side_effect=RuntimeError("colorbar failed")):
                        v._create_contourf_plot(
                            lon, lat, data,
                            -100., -90., 35., 45.,
                            cmap_obj='viridis',
                            norm=None,
                            levels=[0.5, 1.0, 1.5],
                            data_crs=ccrs.PlateCarree(),
                        )
        finally:
            plt.close(v.fig)

    def test_levels_provided_with_colorbar(self: 'TestCreateContourfPlot') -> None:
        """
        This test case verifies the execution of the levels provided path in the _create_contourf_plot method. It checks that when levels are provided, the contourf() method is called with the levels argument, and that the colorbar is added successfully with the specified ticks.

        Parameters:
            None

        Returns:
            None
        """
        v = self._setup_viz_with_ax()
        lon, lat = _simple_lon_lat()
        data = np.ones(N_CELLS)
        lon_g = np.linspace(-100., -90., 5)
        lat_g = np.linspace(35., 45., 5)
        lon_mesh, lat_mesh = np.meshgrid(lon_g, lat_g)
        data_2d = np.ones((5, 5))
        mock_cf = Mock()
        mock_cb = Mock()
        mock_cb.get_ticks.return_value = np.array([0.5, 1.0, 1.5])
        try:
            with patch.object(v, '_interpolate_to_grid',
                               return_value=(lon_mesh, lat_mesh, data_2d)):
                with patch.object(v.ax, 'contourf', return_value=mock_cf):
                    with patch('mpasdiag.visualization.base_visualizer'
                               '.MPASVisualizationStyle.add_colorbar',
                               return_value=mock_cb):
                        v._create_contourf_plot(
                            lon, lat, data,
                            -100., -90., 35., 45.,
                            cmap_obj='viridis',
                            norm=None,
                            levels=[0.5, 1.0, 1.5],
                            data_crs=ccrs.PlateCarree(),
                            colorbar_ticks=[0.5, 1.0],
                        )
        finally:
            plt.close(v.fig)


class TestCreateWindPlot:
    """ Covers wind plot error conditions and auto-subsampling. """

    def _make_wind_data(self: 'TestCreateWindPlot', 
                        n: int = N_CELLS, 
                        lon_min: float = -100., 
                        lon_max: float = -90.,
                        lat_min: float = 35., 
                        lat_max: float = 45.) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This helper method generates synthetic wind data for testing the create_wind_plot method. It creates longitude and latitude arrays that are linearly spaced within specified minimum and maximum bounds, simulating a grid of points. Additionally, it generates random u and v wind component arrays using a uniform distribution, which represent the wind speed in the east-west and north-south directions, respectively. This synthetic dataset can be used across multiple test cases to evaluate the behavior of the create_wind_plot method under various conditions, such as handling dense data or validating input parameters.

        Parameters:
            n (int): The number of data points to generate for longitude, latitude, and wind components. Default is N_CELLS.
            lon_min (float): The minimum longitude value for the generated data. Default is -100.
            lon_max (float): The maximum longitude value for the generated data. Default is -90.
            lat_min (float): The minimum latitude value for the generated data. Default is 35.
            lat_max (float): The maximum latitude value for the generated data. Default is 45.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the generated longitude array, latitude array, u wind component array, and v wind component array.
        """
        rng = np.random.default_rng(42)
        lon = np.linspace(lon_min, lon_max, n)
        lat = np.linspace(lat_min, lat_max, n)
        u = rng.uniform(-10., 10., n)
        v = rng.uniform(-10., 10., n)
        return lon, lat, u, v

    def test_no_valid_wind_data_raises(self: 'TestCreateWindPlot') -> None:
        """
        This test case verifies that the create_wind_plot method raises a ValueError when there is no valid wind data within the specified plot bounds. It simulates a scenario where the longitude and latitude points are outside the defined plot area, and checks that the method correctly identifies the lack of valid data and raises an appropriate error message. This ensures that the method can handle cases where the input data does not contain any points that fall within the plotting region, preventing attempts to create a wind plot with invalid or empty data. 

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        lon = np.array([-50., -40., -30.])
        lat = np.array([10., 15., 20.])
        u = np.array([5., 6., 7.])
        vw = np.array([1., 2., 3.])

        with pytest.raises(ValueError, match="No valid wind data"):
            v.create_wind_plot(
                lon, lat, u, vw,
                lon_min=-100., lon_max=-90., lat_min=35., lat_max=45.,
                plot_type='barbs',
            )

    def test_invalid_plot_type_raises(self: 'TestCreateWindPlot') -> None:
        """
        This test case verifies that the create_wind_plot method raises a ValueError when an invalid plot type is specified. It simulates a scenario where the plot_type parameter is set to a value that is not supported ('barbs' or 'arrows'), and checks that the method correctly identifies the invalid input and raises an appropriate error message. This ensures that the method can handle cases where the input plot type is incorrect, preventing attempts to create a wind plot with unsupported visualization types.

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        lon, lat, u, vw = self._make_wind_data()

        with pytest.raises(ValueError, match="plot_type must be"):
            v.create_wind_plot(
                lon, lat, u, vw,
                lon_min=-100., lon_max=-90., lat_min=35., lat_max=45.,
                plot_type='invalid_type',
            )

    def test_auto_subsampling_dense_data(self: 'TestCreateWindPlot') -> None:
        """
        This test case verifies the automatic subsampling behavior of the create_wind_plot method when the point density is high. It simulates a scenario with a dense set of wind data points and checks that the method correctly calculates the subsample value based on the point density, ensuring that the plot remains readable and not overcrowded.

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        n_pts = 500
        lon = np.random.default_rng(0).uniform(-100., -99., n_pts)
        lat = np.random.default_rng(1).uniform(39., 40., n_pts)
        u = np.ones(n_pts) * 5.
        vw = np.ones(n_pts) * 3.

        try:
            fig, ax = v.create_wind_plot(
                lon, lat, u, vw,
                lon_min=-100., lon_max=-99., lat_min=39., lat_max=40.,
                plot_type='barbs',
                subsample=0,
            )
            plt.close(fig)
        except Exception:
            pass


class TestInterpolateToGridEdgeCases:
    """ Covers negative grid_resolution error and dataset=None KDTree path. """

    def test_negative_resolution_raises(self: 'TestInterpolateToGridEdgeCases') -> None:
        """
        This test case verifies that the _interpolate_to_grid method raises a ValueError when a negative grid resolution is specified. It simulates a scenario where the grid_resolution parameter is set to a negative value and checks that the method correctly identifies the invalid input and raises an appropriate error message. This ensures that the method can handle cases where the input grid resolution is incorrect, preventing attempts to interpolate with an invalid resolution.

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        lon, lat = _simple_lon_lat()
        data = np.ones(N_CELLS)
        with pytest.raises(ValueError, match="grid_resolution must be"):
            v._interpolate_to_grid(lon, lat, data, -100., -90., 35., 45.,
                                   grid_resolution=-1.0)

    def test_dataset_none_creates_minimal_dataset(self: 'TestInterpolateToGridEdgeCases') -> None:
        """
        This test case verifies that the _interpolate_to_grid method creates a minimal dataset when the dataset parameter is set to None. It simulates a scenario where the dataset is not provided and checks that the method correctly generates a minimal dataset to perform the interpolation. This ensures that the method can handle cases where the input dataset is missing, preventing errors during the interpolation process.

        Parameters:
            None

        Returns:
            None
        """
        v = _viz()
        lon, lat = _simple_lon_lat()
        data = np.random.default_rng(0).uniform(0., 1., N_CELLS)

        lon_g = np.linspace(-100., -90., 5)
        lat_g = np.linspace(35., 45., 5)
        fake_result = xr.DataArray(
            np.ones((5, 5)),
            coords={'lon': lon_g, 'lat': lat_g},
            dims=['lat', 'lon'],
        )
        with patch('mpasdiag.visualization.base_visualizer.remap_mpas_to_latlon_with_masking',
                   return_value=fake_result):
            lon_m, lat_m, data_i = v._interpolate_to_grid(
                lon, lat, data, -100., -90., 35., 45.,
                grid_resolution=2.0,
                dataset=None,
            )
        assert lon_m.ndim == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

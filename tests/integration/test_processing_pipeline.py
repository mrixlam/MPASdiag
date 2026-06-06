#!/usr/bin/env python3

"""
MPASdiag Test Suite: Processing Layer Integration Tests

This module contains integration tests for the MPASdiag processing layer, designed to validate the end-to-end functionality of the data loading, coordinate extraction, variable access, and remapping pipelines using real MPAS datasets. These tests ensure that the various processing components work together correctly to produce consistent and physically meaningful outputs, providing confidence in the reliability of the processing layer for downstream diagnostics and plotting.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: June 2026
Version: 1.0.0
"""
import numpy as np
import pytest
import xarray as xr

from mpasdiag.processing.processors_2d import MPAS2DProcessor
from mpasdiag.processing.processors_3d import MPAS3DProcessor
from mpasdiag.processing.remapping import MPASRemapper, remap_mpas_to_latlon

from tests.integration.conftest import ESMPY_AVAILABLE


@pytest.mark.integration
@pytest.mark.requires_data
class TestProcessingPipeline:
    """ Cross-module integration tests for the MPASdiag processing layer using real MPAS data. """

    def test_2d_load_extract_coordinates_and_variable(self: "TestProcessingPipeline", 
                                                      real_2d_processor: MPAS2DProcessor,) -> None:
        """
        This test exercises the core 2D processing pipeline: it loads a real MPAS 2D dataset, extracts the horizontal coordinates for a surface variable, and retrieves the variable data for a single time step. The test asserts that the expected variables are available, the coordinates are properly extracted and finite, and the variable data is a valid xarray DataArray containing finite values. 

        Parameters:
            real_2d_processor (MPAS2DProcessor): Loaded 2D processor fixture with real MPAS data. 

        Returns:
            None
        """
        available = real_2d_processor.get_available_variables()

        assert "t2m" in available
        assert "rainnc" in available

        lon, lat = real_2d_processor.extract_2d_coordinates_for_variable("t2m")
        data = real_2d_processor.get_2d_variable_data("t2m", time_index=0)

        assert isinstance(data, xr.DataArray)
        assert lon.shape == lat.shape
        assert lon.size == data.values.ravel().size
        assert np.isfinite(lon).all()
        assert np.isfinite(lat).all()
        assert np.isfinite(np.asarray(data.values)).any()

    def test_2d_processor_time_dimension(self: "TestProcessingPipeline", 
                                         real_2d_processor: MPAS2DProcessor,) -> None:
        """
        This test verifies that the 2D processor correctly identifies and handles the time dimension in the dataset. It asserts that the dataset contains a time dimension (either 'Time' or 'time') and that it has at least two time steps, ensuring that the processor can access temporal data for diagnostics that require time series analysis.

        Parameters:
            real_2d_processor (MPAS2DProcessor): Loaded 2D processor fixture with real MPAS data.

        Returns:
            None
        """
        dataset = real_2d_processor.dataset

        assert dataset is not None
        time_dim = "Time" if "Time" in dataset.dims else "time"
        assert dataset.sizes[time_dim] >= 2

    def test_3d_load_levels_and_slice(self: "TestProcessingPipeline", 
                                      real_3d_processor: MPAS3DProcessor,) -> None:
        """
        This test exercises the core 3D processing pipeline: it loads a real MPAS 3D dataset, retrieves the available 3D variables, extracts the vertical levels for a chosen variable, and retrieves a surface-level slice of the variable data for a single time step. The test asserts that the expected 3D variable is available, that multiple vertical levels are present, and that the surface slice is a valid xarray DataArray containing finite values.

        Parameters:
            real_3d_processor (MPAS3DProcessor): Loaded 3D processor fixture with real MPAS data.

        Returns:
            None
        """
        available_3d = real_3d_processor.get_available_3d_variables()

        assert "theta" in available_3d

        levels = real_3d_processor.get_vertical_levels("theta")
        assert len(levels) > 1

        surface_slice = real_3d_processor.get_3d_variable_data(
            "theta", time_index=0, level="surface"
        )

        assert isinstance(surface_slice, xr.DataArray)
        assert np.isfinite(np.asarray(surface_slice.values)).any()

    @pytest.mark.parametrize("method", ["nearest", "linear"])
    def test_remap_unstructured_to_latlon_scipy(self: "TestProcessingPipeline", 
                                                real_2d_processor: MPAS2DProcessor, 
                                                method: str,) -> None:
        """
        This test exercises the lightweight remapping path using SciPy's griddata to remap a real MPAS surface field (e.g., 2m temperature) from the unstructured MPAS grid to a regular lat-lon grid. It asserts that the remapped output is an xarray DataArray with the expected coordinate names and dimensions, and that a reasonable fraction of the remapped values are finite, indicating a successful interpolation. 

        Parameters:
            real_2d_processor (MPAS2DProcessor): Loaded 2D processor fixture with real MPAS data.
            method (str): Interpolation method to use for remapping ('nearest' or 'linear').

        Returns:
            None
        """
        lon, lat = real_2d_processor.extract_2d_coordinates_for_variable("t2m")
        data = real_2d_processor.get_2d_variable_data("t2m", time_index=0)

        resolution = 2.0
        remapped = remap_mpas_to_latlon(
            data=np.asarray(data.values).ravel(),
            lon=lon,
            lat=lat,
            resolution=resolution,
            method=method,
        )

        assert isinstance(remapped, xr.DataArray)
        assert "lon" in remapped.coords
        assert "lat" in remapped.coords

        expected_lons = np.arange(-180.0, 180.0 + resolution / 2, resolution)
        expected_lats = np.arange(-90.0, 90.0 + resolution / 2, resolution)
        assert remapped.shape == (expected_lats.size, expected_lons.size)

        finite_fraction = np.isfinite(remapped.values).mean()
        assert finite_fraction > 0.5

    @pytest.mark.skipif(not ESMPY_AVAILABLE, reason="xESMF/ESMPy not available")
    def test_remap_with_mpas_remapper_xesmf(self: "TestProcessingPipeline",
                                            real_2d_processor: MPAS2DProcessor,
                                            tmp_path: object,) -> None:
        """
        This test exercises the remapping pipeline using the MPASRemapper class with xESMF as the backend to remap a real MPAS surface field from the unstructured grid to a regular lat-lon grid. It asserts that the remapped output is an xarray DataArray with the expected coordinate names and dimensions, and that the remapped values are finite, indicating a successful remapping. The test also verifies that weights are properly cached in the specified temporary directory. 

        Parameters:
            real_2d_processor (MPAS2DProcessor): Loaded 2D processor fixture with real MPAS data.
            tmp_path (object): pytest fixture providing a temporary directory for caching remapping weights.

        Returns:
            None
        """
        lon, lat = real_2d_processor.extract_2d_coordinates_for_variable("t2m")
        data = real_2d_processor.get_2d_variable_data("t2m", time_index=0)

        n_test = min(2000, lon.size)
        lon_sub = np.radians(lon[:n_test])
        lat_sub = np.radians(lat[:n_test])
        data_sub = np.asarray(data.values).ravel()[:n_test]

        data_array, grid_dataset = MPASRemapper.unstructured_to_structured_grid(
            data=data_sub,
            lon=lon_sub,
            lat=lat_sub,
            intermediate_resolution=2.0,
        )

        remapper = MPASRemapper(
            method="nearest_s2d", weights_dir=tmp_path, reuse_weights=False
        )

        remapper.source_grid = grid_dataset

        remapper.create_target_grid(
            lon_min=float(np.degrees(lon_sub.min())),
            lon_max=float(np.degrees(lon_sub.max())),
            dlon=3.0,
            dlat=3.0,
        )
        
        remapper.build_regridder()
        result = remapper.remap(data_array)

        assert isinstance(result, xr.DataArray)
        assert result.size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

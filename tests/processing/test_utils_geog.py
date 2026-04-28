"""Tests for mpasdiag/processing/utils_geog.py."""
import numpy as np
import pytest
import xarray as xr

from mpasdiag.processing.utils_geog import MPASGeographicUtils


@pytest.fixture
def degree_dataset() -> xr.Dataset:
    n = 50
    return xr.Dataset({
        "lonCell": xr.DataArray(np.linspace(-120.0, -80.0, n)),
        "latCell": xr.DataArray(np.linspace(25.0, 55.0, n)),
    })


@pytest.fixture
def radian_dataset() -> xr.Dataset:
    n = 50
    lat_rad = np.linspace(-1.5, 1.5, n)
    lon_rad = np.linspace(-3.0, 3.0, n)
    return xr.Dataset({
        "lonCell": xr.DataArray(lon_rad),
        "latCell": xr.DataArray(lat_rad),
    })


class TestExtractSpatialCoordinates:
    """Tests for MPASGeographicUtils.extract_spatial_coordinates error paths and radian conversion."""

    def test_none_dataset_raises(self) -> None:
        with pytest.raises(ValueError, match="Dataset cannot be None"):
            MPASGeographicUtils.extract_spatial_coordinates(None)  # type: ignore[arg-type]

    def test_missing_coords_raises(self) -> None:
        ds = xr.Dataset({"temperature": xr.DataArray(np.ones(10))})
        with pytest.raises(ValueError, match="Could not find spatial coordinates"):
            MPASGeographicUtils.extract_spatial_coordinates(ds)

    def test_radian_coords_converted_to_degrees(self, radian_dataset: xr.Dataset) -> None:
        lat_rad = radian_dataset["latCell"].values
        lon_rad = radian_dataset["lonCell"].values
        assert np.nanmax(np.abs(lat_rad)) <= np.pi

        result_lon, result_lat = MPASGeographicUtils.extract_spatial_coordinates(radian_dataset)

        expected_lat = lat_rad * 180.0 / np.pi
        expected_lon = lon_rad * 180.0 / np.pi
        assert np.allclose(result_lat, expected_lat)
        assert np.allclose(result_lon, expected_lon)


class TestFilterBySpatialExtent:
    """Tests for MPASGeographicUtils.filter_by_spatial_extent."""

    def test_no_ncells_returns_data_unchanged(self, degree_dataset: xr.Dataset) -> None:
        data = xr.DataArray(np.ones(10), dims=["time"])
        result_data, mask = MPASGeographicUtils.filter_by_spatial_extent(
            data, degree_dataset, lon_min=-130.0, lon_max=-70.0,
            lat_min=20.0, lat_max=60.0,
        )
        assert result_data.equals(data)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_ncells_dimension_applies_mask(self, degree_dataset: xr.Dataset) -> None:
        n = len(degree_dataset["lonCell"])
        data = xr.DataArray(np.ones(n), dims=["nCells"])
        result_data, mask = MPASGeographicUtils.filter_by_spatial_extent(
            data, degree_dataset, lon_min=-110.0, lon_max=-90.0,
            lat_min=30.0, lat_max=50.0,
        )
        assert "nCells" in result_data.dims
        assert not np.all(np.isfinite(result_data.values))


class TestValidateGeographicExtent:
    """Tests for MPASGeographicUtils.validate_geographic_extent."""

    def test_valid_extent(self) -> None:
        assert MPASGeographicUtils.validate_geographic_extent((-120.0, -80.0, 25.0, 55.0)) is True

    def test_boundary_extent(self) -> None:
        assert MPASGeographicUtils.validate_geographic_extent((-180.0, 180.0, -90.0, 90.0)) is True

    def test_lon_min_out_of_range(self) -> None:
        assert MPASGeographicUtils.validate_geographic_extent((-181.0, 0.0, -45.0, 45.0)) is False

    def test_lon_max_out_of_range(self) -> None:
        assert MPASGeographicUtils.validate_geographic_extent((0.0, 181.0, -45.0, 45.0)) is False

    def test_lat_min_out_of_range(self) -> None:
        assert MPASGeographicUtils.validate_geographic_extent((-90.0, 90.0, -91.0, 45.0)) is False

    def test_lat_max_out_of_range(self) -> None:
        assert MPASGeographicUtils.validate_geographic_extent((-90.0, 90.0, -45.0, 91.0)) is False

    def test_lon_reversed(self) -> None:
        assert MPASGeographicUtils.validate_geographic_extent((10.0, 5.0, -45.0, 45.0)) is False

    def test_lat_reversed(self) -> None:
        assert MPASGeographicUtils.validate_geographic_extent((-90.0, 90.0, 45.0, 10.0)) is False


class TestGetExtentFromCoordinates:
    """Tests for MPASGeographicUtils.get_extent_from_coordinates."""

    def test_all_nan_raises(self) -> None:
        lon = np.full(10, np.nan)
        lat = np.full(10, np.nan)
        with pytest.raises(ValueError, match="No valid coordinates found"):
            MPASGeographicUtils.get_extent_from_coordinates(lon, lat)

    def test_all_inf_raises(self) -> None:
        lon = np.full(5, np.inf)
        lat = np.full(5, -np.inf)
        with pytest.raises(ValueError, match="No valid coordinates found"):
            MPASGeographicUtils.get_extent_from_coordinates(lon, lat)

    def test_valid_coords_returns_bounding_box(self) -> None:
        lon = np.linspace(-120.0, -80.0, 50)
        lat = np.linspace(25.0, 55.0, 50)
        lon_min, lon_max, lat_min, lat_max = MPASGeographicUtils.get_extent_from_coordinates(lon, lat)
        assert lon_min == pytest.approx(-120.0)
        assert lon_max == pytest.approx(-80.0)
        assert lat_min == pytest.approx(25.0)
        assert lat_max == pytest.approx(55.0)

    def test_buffer_expands_extent(self) -> None:
        lon = np.linspace(-100.0, -90.0, 20)
        lat = np.linspace(30.0, 40.0, 20)
        lon_min, lon_max, lat_min, lat_max = MPASGeographicUtils.get_extent_from_coordinates(
            lon, lat, buffer=5.0
        )
        assert lon_min == pytest.approx(-105.0)
        assert lon_max == pytest.approx(-85.0)
        assert lat_min == pytest.approx(25.0)
        assert lat_max == pytest.approx(45.0)

    def test_lat_clamped_to_valid_range(self) -> None:
        lon = np.linspace(-10.0, 10.0, 10)
        lat = np.array([85.0, 88.0, 89.5, 89.9, 90.0, 85.0, 87.0, 88.5, 89.0, 89.8])
        _, _, _, lat_max = MPASGeographicUtils.get_extent_from_coordinates(lon, lat, buffer=5.0)
        assert lat_max <= 90.0


class TestCalculateSpatialResolution:
    """Tests for MPASGeographicUtils.calculate_spatial_resolution."""

    def test_single_point_returns_zero(self) -> None:
        result = MPASGeographicUtils.calculate_spatial_resolution(
            np.array([45.0]), np.array([30.0])
        )
        assert result == 0.0

    def test_normal_data_returns_positive(self) -> None:
        lon = np.linspace(-120.0, -80.0, 100)
        lat = np.linspace(25.0, 55.0, 100)
        result = MPASGeographicUtils.calculate_spatial_resolution(lon, lat)
        assert result > 0.0
        assert isinstance(result, float)

    def test_sampling_path_large_dataset(self) -> None:
        rng = np.random.default_rng(42)
        lon = rng.uniform(-120.0, -80.0, 2000)
        lat = rng.uniform(25.0, 55.0, 2000)
        result = MPASGeographicUtils.calculate_spatial_resolution(lon, lat, sample_size=100)
        assert result > 0.0

    def test_no_sampling_small_dataset(self) -> None:
        lon = np.linspace(-100.0, -90.0, 50)
        lat = np.linspace(30.0, 40.0, 50)
        result = MPASGeographicUtils.calculate_spatial_resolution(lon, lat, sample_size=1000)
        assert result > 0.0

    def test_identical_points_returns_zero(self) -> None:
        lon = np.full(10, 45.0)
        lat = np.full(10, 30.0)
        result = MPASGeographicUtils.calculate_spatial_resolution(lon, lat)
        assert result == 0.0


class TestIsGlobalExtent:
    """Tests for MPASGeographicUtils.is_global_extent."""

    def test_global_extent_returns_true(self) -> None:
        assert MPASGeographicUtils.is_global_extent((-180.0, 180.0, -90.0, 90.0)) is True

    def test_non_global_lon_returns_false(self) -> None:
        assert MPASGeographicUtils.is_global_extent((-90.0, 90.0, -90.0, 90.0)) is False

    def test_non_global_lat_returns_false(self) -> None:
        assert MPASGeographicUtils.is_global_extent((-180.0, 180.0, -45.0, 45.0)) is False

    def test_custom_tolerance_passes(self) -> None:
        assert MPASGeographicUtils.is_global_extent(
            (-179.5, 179.5, -89.5, 89.5), tolerance=2.0
        ) is True

    def test_custom_tolerance_fails(self) -> None:
        assert MPASGeographicUtils.is_global_extent(
            (-179.5, 179.5, -89.5, 89.5), tolerance=0.0
        ) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

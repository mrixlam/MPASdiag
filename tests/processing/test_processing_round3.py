#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Test Suite: Processing Layer Round-3 Regression Tests

This module contains regression tests for the third round of processing-layer fixes: conservative remapping at the longitude seam and the pole rows, convex-hull masking of global meshes, glob metacharacters in data directories, the weights cache key and validation, stale weights after a grid change, spatial filtering of vertex and edge data, below-ground masking of vertex fields, rejection of edge variables, potential-temperature and effective-radius metadata, and the repaired file and geography helpers. Tests that need ESMPy or real MPAS output skip gracefully when they are unavailable.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: October 2026
Version: 1.0.0
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import xarray as xr

from mpasdiag.processing.processors_2d import MPAS2DProcessor
from mpasdiag.processing.remapping import (
    ESMPY_AVAILABLE,
    MPASRemapper,
    _compute_grid_bounds,
    build_remapped_valid_mask,
)
from mpasdiag.processing.utils_file import FileManager
from mpasdiag.processing.utils_geog import MPASGeographicUtils
from mpasdiag.processing.utils_metadata import MPASFileMetadata
from mpasdiag.processing.utils_unit import UnitConverter

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
GRID = DATA_ROOT / "grids" / "x1.10242.static.nc"
DIAG_DIR = DATA_ROOT / "u240k" / "diag"
HAS_DATA = GRID.exists() and DIAG_DIR.exists() and any(DIAG_DIR.glob("diag*.nc"))

needs_esmpy = pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy not installed")
needs_data = pytest.mark.skipif(not HAS_DATA, reason="real MPAS data not available")


def _latlon_mesh(step: float = 10.0) -> Tuple[np.ndarray, ...]:
    """
    Builds a global regular lat-lon "mesh" whose cells are the lat-lon boxes themselves, returning cell-centre longitudes (0-360), latitudes and the rectangular cell-corner arrays needed for conservative remapping.

    Parameters:
        step (float): Cell size in degrees.

    Returns:
        Tuple[np.ndarray, ...]: (lon, lat, lon_b, lat_b) with the corner arrays of shape (nCells, 4).
    """
    lons = np.arange(step / 2, 360.0, step)
    lats = np.arange(-90 + step / 2, 90.0, step)
    lon2, lat2 = np.meshgrid(lons, lats)
    lon, lat = lon2.ravel(), lat2.ravel()
    h = step / 2
    lon_b = np.column_stack([lon - h, lon + h, lon + h, lon - h])
    lat_b = np.column_stack([lat - h, lat - h, lat + h, lat + h])
    return lon, lat, lon_b, lat_b


class TestConservativeSeamAndPoles:
    """Conservative remapping must give ~1 for a constant field everywhere, including the lon=0 seam and the pole rows."""

    def test_grid_bounds_use_adjacent_steps(self) -> None:
        """
        This test checks that the grid-bound helper builds the end edges from each end's own adjacent step, so a non-uniform axis gets correct half-steps at both ends.
        """
        b = _compute_grid_bounds(np.array([0.0, 1.0, 3.0]), 1.0)
        assert np.allclose(b, [-0.5, 0.5, 2.0, 4.0])

    def test_grid_bounds_single_point(self) -> None:
        """
        This test checks that a one-point axis falls back on the supplied resolution.
        """
        assert np.allclose(_compute_grid_bounds(np.array([5.0]), 2.0), [4.0, 6.0])

    @needs_esmpy
    @pytest.mark.parametrize(
        "extent",
        [(-180, 180, -90, 90), (-20, 40, 30, 70), (-60, -10, -30, 30)],
    )
    def test_constant_field_stays_constant(self, extent: Tuple[float, ...]) -> None:
        """
        This test remaps a field of ones with the conservative method on targets that straddle lon=0 and reach the poles, and asserts every target cell is ~1 (no garbage at the seam or pole rows).
        """
        lon, lat, lon_b, lat_b = _latlon_mesh(10.0)
        rm = MPASRemapper(method="conservative", skipna=True)
        rm.prepare_source_grid(lon, lat, lon_bounds=lon_b, lat_bounds=lat_b)
        rm.create_target_grid(*extent, dlon=5.0, dlat=5.0)
        rm.build_regridder()
        out = rm.remap(xr.DataArray(np.ones(lon.size), dims=["nCells"])).values
        assert np.nanmin(out) > 0.99
        assert np.nanmax(out) < 1.01

    @needs_esmpy
    def test_seam_matches_neighbours(self) -> None:
        """
        This test remaps a longitude-dependent smooth field and asserts that values at lon -2 and 0 lie between their neighbours instead of jumping, which is what the non-monotonic target axis used to break.
        """
        lon, lat, lon_b, lat_b = _latlon_mesh(10.0)
        field = np.cos(np.radians(lon)) + 0.0 * lat
        rm = MPASRemapper(method="conservative_normed", skipna=True)
        rm.prepare_source_grid(lon, lat, lon_bounds=lon_b, lat_bounds=lat_b)
        rm.create_target_grid(-20, 40, 30, 70, dlon=2.0, dlat=2.0)
        rm.build_regridder()
        out = rm.remap(xr.DataArray(field, dims=["nCells"]))
        row = out.values[10]
        lons = out.lon.values
        i0 = int(np.where(lons == 0)[0][0])
        assert np.all(np.abs(np.diff(row[i0 - 3 : i0 + 4])) < 0.1)


class TestValidMask:
    """The convex-hull mask must not blank global meshes but must still mask regional ones."""

    def test_global_mesh_not_masked(self) -> None:
        """
        This test asserts that a global source mesh gives no hull mask (None), so the western edge column and polar rows are kept.
        """
        lon, lat, _, _ = _latlon_mesh(10.0)
        data = xr.DataArray(
            np.ones((19, 37)),
            dims=["lat", "lon"],
            coords={"lat": np.linspace(-90, 90, 19), "lon": np.linspace(-20, 160, 37)},
        )
        mask = build_remapped_valid_mask(
            np.where(lon > 180, lon - 360, lon), lat, -20, 160, -90, 90, 5.0, data
        )
        assert mask is None

    def test_regional_mesh_still_masked(self) -> None:
        """
        This test asserts that a regional source mesh still produces a hull mask: target points far outside the mesh are False and points inside are True.
        """
        lons = np.arange(0.0, 20.0, 1.0)
        lats = np.arange(0.0, 20.0, 1.0)
        lon2, lat2 = np.meshgrid(lons, lats)
        data = xr.DataArray(
            np.ones((5, 5)),
            dims=["lat", "lon"],
            coords={
                "lat": np.array([-10.0, 5.0, 10.0, 15.0, 40.0]),
                "lon": np.array([-10.0, 5.0, 10.0, 15.0, 40.0]),
            },
        )
        mask = build_remapped_valid_mask(
            lon2.ravel(), lat2.ravel(), -10, 40, -10, 40, 5.0, data
        )
        assert mask is not None
        assert mask[2, 2] and not mask[0, 0] and not mask[4, 4]


class TestGlobEscape:
    """A data directory containing glob metacharacters must still be searched."""

    def test_bracket_directory(self, tmp_path: Path) -> None:
        """
        This test creates diag files in a directory named 'run[1] (a)' and asserts the finder discovers them.
        """
        grid = tmp_path / "grid.nc"
        xr.Dataset({"lonCell": ("nCells", np.zeros(3))}).to_netcdf(grid)
        d = tmp_path / "run[1] (a)"
        d.mkdir()
        for h in range(3):
            (d / f"diag.2024-09-17_0{h}.00.00.nc").write_bytes(b"")
        p = MPAS2DProcessor(str(grid), verbose=False)
        assert len(p.find_diagnostic_files(str(d))) == 3

    def test_bracket_directory_recursive(self, tmp_path: Path) -> None:
        """
        This test checks the recursive fallback finder with a bracketed directory.
        """
        grid = tmp_path / "grid.nc"
        xr.Dataset({"lonCell": ("nCells", np.zeros(3))}).to_netcdf(grid)
        d = tmp_path / "a[1]" / "sub"
        d.mkdir(parents=True)
        (d / "diag.2024-09-17_00.00.00.nc").write_bytes(b"")
        p = MPAS2DProcessor(str(grid), verbose=False)
        assert p._find_diag_files_recursive(str(tmp_path / "a[1]")) is not None


class TestWeightsCache:
    """The weights cache must be keyed on everything that changes weights and validated on load."""

    @needs_esmpy
    def test_corner_bounds_change_cache_key(self, tmp_path: Path) -> None:
        """
        This test asserts that two conservative remappers with identical centres but different cell polygons resolve to different cache files.
        """
        lon, lat, lon_b, lat_b = _latlon_mesh(20.0)
        names = []
        for scale in (1.0, 0.5):
            rm = MPASRemapper(method="conservative", weights_dir=tmp_path)
            src = rm.prepare_source_grid(
                lon,
                lat,
                lon_bounds=lon[:, None] + scale * (lon_b - lon[:, None]),
                lat_bounds=lat[:, None] + scale * (lat_b - lat[:, None]),
            )
            tgt = rm.create_target_grid(-40, 40, -20, 20, dlon=10.0, dlat=10.0)
            names.append(rm._resolve_weights_path(src, tgt, None))
        assert names[0] != names[1]

    @needs_esmpy
    def test_extrapolation_and_periodic_change_cache_key(self, tmp_path: Path) -> None:
        """
        This test asserts that extrapolation and periodic settings are part of the cache key.
        """
        lon, lat, _, _ = _latlon_mesh(20.0)
        paths = []
        for kwargs in ({}, {"extrap_method": "nearest_s2d"}, {"periodic": True}):
            rm = MPASRemapper(method="nearest_s2d", weights_dir=tmp_path, **kwargs)
            src = rm.prepare_source_grid(lon, lat)
            tgt = rm.create_target_grid(-40, 40, -20, 20, dlon=10.0, dlat=10.0)
            paths.append(rm._resolve_weights_path(src, tgt, None))
        assert len(set(paths)) == 3

    @needs_esmpy
    def test_method_mismatch_regenerates(self, tmp_path: Path) -> None:
        """
        This test builds weights with one method under an explicit filename and asks a remapper of another method to reuse that file; it must regenerate instead of applying the wrong weights.
        """
        lon, lat, _, _ = _latlon_mesh(20.0)
        a = MPASRemapper(method="nearest_s2d", weights_dir=tmp_path)
        a.prepare_source_grid(lon, lat)
        a.create_target_grid(-40, 40, -20, 20, dlon=10.0, dlat=10.0)
        a.build_regridder(filename="w.nc")
        b = MPASRemapper(method="nearest_d2s", weights_dir=tmp_path)
        b.prepare_source_grid(lon, lat)
        b.create_target_grid(-40, 40, -20, 20, dlon=10.0, dlat=10.0)
        assert b._try_load_cached_weights(tmp_path / "w.nc") is False
        assert (
            b._try_load_cached_weights(
                tmp_path / "w.nc", expected_n_src=lon.size, expected_tgt_shape=(5, 9)
            )
            is False
        )

    @needs_esmpy
    def test_size_mismatch_regenerates(self, tmp_path: Path) -> None:
        """
        This test asserts that a cache file built for a different target shape is rejected.
        """
        lon, lat, _, _ = _latlon_mesh(20.0)
        a = MPASRemapper(method="nearest_s2d", weights_dir=tmp_path)
        a.prepare_source_grid(lon, lat)
        a.create_target_grid(-40, 40, -20, 20, dlon=10.0, dlat=10.0)
        a.build_regridder(filename="w.nc")
        b = MPASRemapper(method="nearest_s2d", weights_dir=tmp_path)
        assert not b._try_load_cached_weights(
            tmp_path / "w.nc", expected_n_src=lon.size, expected_tgt_shape=(3, 3)
        )
        assert b._try_load_cached_weights(
            tmp_path / "w.nc", expected_n_src=lon.size, expected_tgt_shape=(5, 9)
        )

    @needs_esmpy
    def test_new_grid_resets_weights(self) -> None:
        """
        This test asserts that creating a new target grid or preparing a new source grid discards previously built weights, so remap() demands a rebuild.
        """
        lon, lat, _, _ = _latlon_mesh(20.0)
        rm = MPASRemapper(method="nearest_s2d")
        rm.prepare_source_grid(lon, lat)
        rm.create_target_grid(-40, 40, -20, 20, dlon=10.0, dlat=10.0)
        rm.build_regridder()
        assert rm._weights is not None
        rm.create_target_grid(100, 150, -20, 20, dlon=10.0, dlat=10.0)
        assert rm._weights is None
        with pytest.raises(ValueError, match="build_regridder"):
            rm.remap(np.ones(lon.size))
        rm.build_regridder()
        rm.prepare_source_grid(lon, lat)
        assert rm._weights is None and rm._cell_of_element is None


def _vertex_dataset() -> xr.Dataset:
    """
    Builds a tiny dataset with cell, vertex and edge coordinates (degrees) for the spatial-filter tests.

    Returns:
        xr.Dataset: Dataset with lonCell/latCell (4), lonVertex/latVertex (3) and lonEdge/latEdge (2).
    """
    return xr.Dataset(
        {
            "lonCell": ("nCells", np.array([0.0, 10.0, 20.0, 30.0])),
            "latCell": ("nCells", np.array([0.0, 0.0, 0.0, 0.0])),
            "lonVertex": ("nVertices", np.array([0.0, 15.0, 30.0])),
            "latVertex": ("nVertices", np.array([0.0, 0.0, 0.0])),
            "lonEdge": ("nEdges", np.array([5.0, 25.0])),
            "latEdge": ("nEdges", np.array([0.0, 0.0])),
        }
    )


class TestSpatialFilter:
    """filter_by_spatial_extent must use the coordinates that match the data's spatial dimension."""

    def test_vertex_data_filtered_by_vertex_coords(self) -> None:
        """
        This test asserts vertex data is masked with the vertex coordinates and a vertex-length mask.
        """
        ds = _vertex_dataset()
        data = xr.DataArray(np.ones(3), dims=["nVertices"])
        out, mask = MPASGeographicUtils.filter_by_spatial_extent(
            data, ds, 10, 20, -5, 5
        )
        assert mask.shape == (3,)
        assert np.array_equal(np.isfinite(out.values), [False, True, False])

    def test_edge_data_filtered_by_edge_coords(self) -> None:
        """
        This test asserts edge data is masked with the edge coordinates.
        """
        ds = _vertex_dataset()
        data = xr.DataArray(np.ones(2), dims=["nEdges"])
        out, mask = MPASGeographicUtils.filter_by_spatial_extent(data, ds, 0, 10, -5, 5)
        assert np.array_equal(np.isfinite(out.values), [True, False])
        assert mask.shape == (2,)

    def test_vertex_data_without_vertex_coords_raises(self) -> None:
        """
        This test asserts a clear ValueError when vertex data is filtered but the dataset has no vertex coordinates.
        """
        ds = _vertex_dataset().drop_vars(["lonVertex", "latVertex"])
        with pytest.raises(ValueError, match="lonVertex"):
            MPASGeographicUtils.filter_by_spatial_extent(
                xr.DataArray(np.ones(3), dims=["nVertices"]), ds, 0, 1, 0, 1
            )

    def test_cell_data_unchanged(self) -> None:
        """
        This test asserts cell data keeps the previous behaviour.
        """
        ds = _vertex_dataset()
        out, mask = MPASGeographicUtils.filter_by_spatial_extent(
            xr.DataArray(np.ones(4), dims=["nCells"]), ds, 5, 25, -5, 5
        )
        assert np.array_equal(np.isfinite(out.values), [False, True, True, False])


class TestProcessor2D:
    """2D processor behaviours: edge variables and vertex below-ground masking."""

    def test_edge_variable_rejected(self, tmp_path: Path) -> None:
        """
        This test asserts that requesting an nEdges variable raises a clear ValueError instead of failing late.
        """
        grid = tmp_path / "grid.nc"
        xr.Dataset({"lonCell": ("nCells", np.zeros(3))}).to_netcdf(grid)
        p = MPAS2DProcessor(str(grid), verbose=False)
        p.dataset = xr.Dataset({"u": (("Time", "nEdges"), np.ones((1, 5)))})
        with pytest.raises(ValueError, match="nEdges"):
            p.get_2d_variable_data("u", 0)

    def test_cells_to_vertices_average(self) -> None:
        """
        This test checks the cell-to-vertex averaging helper, including a missing (index 0) cell.
        """
        out = MPAS2DProcessor._cells_to_vertices(
            np.array([1.0, 3.0, 5.0]), np.array([[1, 2, 3], [1, 3, 0]])
        )
        assert np.allclose(out, [3.0, 3.0])

    @needs_data
    def test_vertex_field_below_ground_masked(self) -> None:
        """
        This test asserts that vorticity at 850 hPa has below-ground vertices set to NaN and records the masking, while mask_below_ground=False leaves it untouched.
        """
        p = MPAS2DProcessor(str(GRID), verbose=False)
        p.load_2d_data(str(DIAG_DIR))
        masked = p.get_2d_variable_data("vorticity_850hPa", 1)
        raw = p.get_2d_variable_data("vorticity_850hPa", 1, mask_below_ground=False)
        assert masked.attrs.get("below_ground_masked_cells", 0) > 0
        assert np.isnan(masked.values).sum() > np.isnan(raw.values).sum()


class TestMetadata:
    """Metadata and unit conventions."""

    @pytest.mark.parametrize("name", ["theta", "theta_base", "th2m", "theta_e"])
    def test_theta_displayed_in_kelvin(self, name: str) -> None:
        """
        This test asserts potential-temperature fields keep K as the display unit.
        """
        assert UnitConverter.get_display_units(name, "K") == "K"

    def test_temperature_still_celsius(self) -> None:
        """
        This test asserts real temperatures keep their Celsius display.
        """
        assert UnitConverter.get_display_units("t2m", "K") == "°C"
        assert UnitConverter.get_display_units("tslb", "K") == "°C"

    def test_theta_levels_cover_data(self) -> None:
        """
        This test asserts the theta preset is in K and spans realistic tropospheric through stratospheric values.
        """
        m = MPASFileMetadata.get_3d_variable_metadata("theta")
        assert m["units"] == "K"
        assert min(m["levels"]) <= 270 and max(m["levels"]) >= 1300

    @pytest.mark.parametrize("name", ["re_cloud", "re_ice", "re_snow"])
    def test_effective_radius_units_consistent(self, name: str) -> None:
        """
        This test asserts the registry says metres (the data unit) and the display converts to microns.
        """
        assert MPASFileMetadata.get_variable_metadata(name)["units"] == "m"
        m = MPASFileMetadata.get_2d_variable_metadata(name)
        assert m["original_units"] == "m" and m["units"] == "microns"

    def test_ni_units_normalise(self) -> None:
        """
        This test asserts the data unit string 'nb kg^{-1}' normalises to the registry unit and the preset includes zero.
        """
        assert UnitConverter._normalize_unit_string("nb kg^{-1}") == "1/kg"
        assert MPASFileMetadata.get_variable_metadata("ni")["units"] == "1/kg"
        assert 0 in MPASFileMetadata.get_3d_variable_metadata("ni")["levels"]

    def test_surface_pressure_levels_cover_high_terrain(self) -> None:
        """
        This test asserts the surface-pressure preset reaches below 655 hPa.
        """
        levels = MPASFileMetadata.get_2d_variable_metadata("surface_pressure")["levels"]
        assert min(levels) <= 540 and max(levels) >= 1040

    @pytest.mark.parametrize("name", ["uReconstructZonal", "uReconstructMeridional"])
    def test_ureconstruct_registered(self, name: str) -> None:
        """
        This test asserts the mixed-case reconstructed-wind variables are found by the registry.
        """
        m = MPASFileMetadata.get_variable_metadata(name)
        assert m["units"] == "m/s"
        assert m["long_name"] != name

    def test_unknown_variable_falls_back_to_attrs(self) -> None:
        """
        This test asserts an unknown variable takes units and long_name from the supplied data array.
        """
        arr = xr.DataArray(
            np.ones(3), attrs={"units": "J kg^{-1}", "long_name": "Mystery energy"}
        )
        m = MPASFileMetadata.get_variable_metadata("mystery_var", arr)
        assert m["units"] == "J kg^{-1}" and m["long_name"] == "Mystery energy"

    def test_registry_unit_aliases_match_data_units(self) -> None:
        """
        This test asserts common data-file unit spellings normalise to the registry spellings.
        """
        n = UnitConverter._normalize_unit_string
        assert n("kg m^{-3}") == "kg/m^3"
        assert n("m3 m^{-3}") == "m^3/m^3"


class TestHelpers:
    """File and geography helpers."""

    def test_spatial_resolution_nearest_neighbour(self) -> None:
        """
        This test asserts the resolution estimate of a 2-degree regular grid is close to 2 degrees (nearest-neighbour spacing) irrespective of point ordering.
        """
        lon, lat, _, _ = _latlon_mesh(2.0)
        order = np.random.default_rng(0).permutation(lon.size)
        res = MPASGeographicUtils.calculate_spatial_resolution(
            lon[order], lat[order], sample_size=5000
        )
        assert 1.0 < res < 2.5

    def test_load_config_missing_raises(self) -> None:
        """
        This test asserts a missing configuration file raises FileNotFoundError rather than returning defaults.
        """
        with pytest.raises(FileNotFoundError):
            FileManager.load_config_file("/nonexistent/config.yaml")

    def test_load_config_invalid_raises(self, tmp_path: Path) -> None:
        """
        This test asserts an unparsable configuration file raises a ValueError mentioning the file.
        """
        bad = tmp_path / "bad.yaml"
        bad.write_text("grid_file: [unclosed\n")
        with pytest.raises(ValueError, match="bad.yaml"):
            FileManager.load_config_file(str(bad))

    def test_validate_input_files_custom_pattern(self, tmp_path: Path) -> None:
        """
        This test asserts validate_input_files honours a custom file_pattern on the configuration.
        """
        grid = tmp_path / "grid.nc"
        grid.write_bytes(b"")
        data = tmp_path / "data"
        data.mkdir()
        (data / "wiso.2024-09-17_00.00.00.nc").write_bytes(b"")

        class Cfg:
            grid_file = str(grid)
            data_dir = str(data)
            file_pattern = "wiso.*.nc"

        assert FileManager.validate_input_files(Cfg()) is True  # type: ignore[arg-type]
        Cfg.file_pattern = "nomatch*.nc"
        assert FileManager.validate_input_files(Cfg()) is False  # type: ignore[arg-type]


class TestCubicKDTreeRemap:
    """The KDTree gridder accepts 'cubic' (used by --interpolation cubic)."""

    def test_cubic_matches_smooth_field(self: "TestCubicKDTreeRemap") -> None:
        """
        This test asserts cubic remapping of a smooth field reproduces it on the interior of the target grid and that an unknown method is still rejected.
        """
        from mpasdiag.processing.remapping import remap_mpas_to_latlon

        rng = np.random.default_rng(0)
        lon = rng.uniform(-40.0, 40.0, 4000)
        lat = rng.uniform(-30.0, 30.0, 4000)
        values = np.sin(np.radians(lon)) * np.cos(np.radians(lat))

        result = remap_mpas_to_latlon(
            values, lon, lat, -20.0, 20.0, -15.0, 15.0, resolution=1.0, method="cubic"
        )
        lon2d, lat2d = np.meshgrid(result["lon"].values, result["lat"].values)
        expected = np.sin(np.radians(lon2d)) * np.cos(np.radians(lat2d))

        assert np.all(np.isfinite(result.values))
        assert np.nanmax(np.abs(result.values - expected)) < 1e-3

        with pytest.raises(ValueError, match="cubic"):
            remap_mpas_to_latlon(values, lon, lat, method="spline")

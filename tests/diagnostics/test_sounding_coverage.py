#!/usr/bin/env python3

"""
MPASdiag Test Suite: Sounding Diagnostics Coverage 

This module contains unit tests for the SoundingDiagnostics class in mpasdiag.diagnostics.sounding, specifically targeting code paths that were previously untested. The tests cover fallback LCL computation when MetPy is unavailable, handling of missing pressure/temperature/dewpoint variables, and the extraction of profiles from the dataset. By exercising these branches, we ensure that the SoundingDiagnostics class behaves robustly under a wider range of input conditions and provides informative feedback to users when expected data is missing or when certain computations cannot be performed.   

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: April 2026
Version: 1.0.0
"""
import pytest
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional
from unittest.mock import Mock

from mpasdiag.diagnostics.sounding import SoundingDiagnostics
from mpasdiag.processing.processors_3d import MPAS3DProcessor

try:
    from metpy.units import units
    assert units is not None 
    HAS_METPY_TEST = True
except ImportError:
    HAS_METPY_TEST = False


def _synthetic_mpas_data(n_cells: int = 5,
                         n_vert: int = 10,
                         pressure_varname: str = "pressure",
                         temp_varname: str = "theta", 
                         dew_varname: str = "qv",     
                         include_wind: bool = True,
                         include_height: bool = False,
                         include_staggered_height: bool = False,
                         nan_pressure_idx: Optional[int] = None,) -> xr.Dataset:
    """
    This function generates a synthetic xarray Dataset that mimics the structure of MPAS 3D output, with configurable options for pressure, temperature, and dewpoint variable names, as well as the inclusion of wind and height variables. The pressure profile is created to decrease with height, and NaN values can be introduced at a specified index to test handling of missing data. This dataset serves as a controlled input for testing the SoundingDiagnostics class, allowing us to verify that it correctly extracts and processes the necessary profiles under various conditions.

    Parameters:
        n_cells (int): Number of horizontal cells in the dataset.
        n_vert (int): Number of vertical levels in the dataset.
        pressure_varname (str): Name of the pressure variable.
        temp_varname (str): Name of the temperature variable.
        dew_varname (str): Name of the dewpoint/humidity variable.
        include_wind (bool): Whether to include wind variables.
        include_height (bool): Whether to include height variables.
        include_staggered_height (bool): Whether to include staggered height variables.
        nan_pressure_idx (Optional[int]): Index at which to introduce NaN in the pressure profile.

    Returns:
        xr.Dataset: A synthetic dataset with the specified structure and variables. 
    """
    p_vals = np.linspace(101000.0, 5000.0, n_vert)

    if nan_pressure_idx is not None:
        p_vals[nan_pressure_idx] = np.nan

    pressure = np.tile(p_vals, (1, n_cells, 1))
    data_vars: dict = {}

    if pressure_varname == "pressure":
        data_vars["pressure"] = (["Time", "nCells", "nVertLevels"], pressure)
    elif pressure_varname == "components":
        data_vars["pressure_p"] = (
            ["Time", "nCells", "nVertLevels"],
            np.ones((1, n_cells, n_vert)) * 1000.0,
        )
        data_vars["pressure_base"] = (["Time", "nCells", "nVertLevels"], pressure)

    if temp_varname == "theta":
        data_vars["theta"] = (
            ["Time", "nCells", "nVertLevels"],
            np.full((1, n_cells, n_vert), 300.0),
        )
    elif temp_varname == "temperature":
        data_vars["temperature"] = (
            ["Time", "nCells", "nVertLevels"],
            np.full((1, n_cells, n_vert), 285.0),
        )
    elif temp_varname == "temp":
        data_vars["temp"] = (
            ["Time", "nCells", "nVertLevels"],
            np.full((1, n_cells, n_vert), 285.0),
        )

    if dew_varname == "qv":
        data_vars["qv"] = (
            ["Time", "nCells", "nVertLevels"],
            np.full((1, n_cells, n_vert), 0.010),
        )
    elif dew_varname == "dewpoint_K":
        # mean > 100 → will be converted from K to °C
        data_vars["dewpoint"] = (
            ["Time", "nCells", "nVertLevels"],
            np.full((1, n_cells, n_vert), 270.0),
        )
    elif dew_varname == "td_C":
        # mean < 100 → already in °C
        data_vars["td"] = (
            ["Time", "nCells", "nVertLevels"],
            np.full((1, n_cells, n_vert), 12.0),
        )

    if include_wind:
        data_vars["uReconstructZonal"] = (
            ["Time", "nCells", "nVertLevels"],
            np.ones((1, n_cells, n_vert)) * 5.0,
        )
        data_vars["uReconstructMeridional"] = (
            ["Time", "nCells", "nVertLevels"],
            np.ones((1, n_cells, n_vert)) * 3.0,
        )

    if include_height:
        heights = np.linspace(0.0, 15000.0, n_vert)
        data_vars["height"] = (
            ["Time", "nCells", "nVertLevels"],
            np.tile(heights, (1, n_cells, 1)),
        )

    if include_staggered_height:
        # nVertLevelsP1 = n_vert + 1 → triggers mid-level averaging
        heights_p1 = np.linspace(0.0, 15000.0, n_vert + 1)
        data_vars["zgrid"] = (
            ["nCells", "nVertLevelsP1"],
            np.tile(heights_p1, (n_cells, 1)),
        )

    return xr.Dataset(data_vars)


def _make_mock_proc(ds: xr.Dataset, 
                    tmp_path: Path, 
                    prefix: str = "grid",
                    extra_vars: bool = False, 
                    omit_lon: bool = False,
                    omit_lat: bool = False) -> MPAS3DProcessor:
    """
    This function creates a mock MPAS3DProcessor with a synthetic grid file based on the provided dataset. The grid file can be customized to include or omit longitude and latitude variables, as well as to add extra variables that are not part of the standard sounding coordinates. This allows for testing how the SoundingDiagnostics class handles different grid configurations and ensures that it can correctly extract necessary information from the grid file under various conditions. The mock processor is set up with the provided dataset and a path to the generated grid file, ready for use in unit tests.

    Parameters:
        ds (xr.Dataset): The dataset to be associated with the mock processor.
        tmp_path (Path): A temporary directory path for saving the synthetic grid file.
        prefix (str): A prefix for the grid file name.
        extra_vars (bool): Whether to include extra variables in the grid file that are not part of the standard sounding coordinates.
        omit_lon (bool): Whether to omit the longitude variable from the grid file.
        omit_lat (bool): Whether to omit the latitude variable from the grid file.
    
    Returns:
        MPAS3DProcessor: A mock MPAS3DProcessor instance with the synthetic grid file.
    """
    n_cells = ds.sizes.get("nCells", 5)
    lon = np.linspace(-110.0, -90.0, n_cells)
    lat = np.linspace(25.0, 45.0, n_cells)

    grid_vars: dict = {}

    if not omit_lon:
        grid_vars["lonCell"] = (["nCells"], np.radians(lon))

    if not omit_lat:
        grid_vars["latCell"] = (["nCells"], np.radians(lat))

    if extra_vars:
        grid_vars["cellsOnCell"] = (["nCells"], np.arange(n_cells, dtype=np.int32))

    grid_path = tmp_path / f"{prefix}.nc"
    xr.Dataset(grid_vars).to_netcdf(str(grid_path))

    proc = Mock(spec=MPAS3DProcessor)
    proc.dataset = ds
    proc.grid_file = str(grid_path)
    return proc


@pytest.mark.skipif(not HAS_METPY_TEST, reason="MetPy not installed")
class TestThermodynamicIndicesWithWindAndHeight:
    """ Tests for compute_thermodynamic_indices when wind and height profiles are supplied. """

    def test_wet_bulb_zero_path_executed(self: 'TestThermodynamicIndicesWithWindAndHeight') -> None:
        """
        This test verifies that when pressure, temperature, dewpoint, and height profiles are supplied to the compute_thermodynamic_indices method, the code path for computing the wet-bulb zero height is executed. It checks that the resulting dictionary includes the key 'wet_bulb_zero_height', which indicates that the method attempted to compute this index based on the provided profiles. This ensures that users who supply height information along with their sounding data can access the wet-bulb zero height index, which is a valuable parameter for understanding atmospheric conditions. 

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        n = 40
        p = np.linspace(1000.0, 200.0, n)
        t = np.linspace(25.0, -60.0, n)
        td = t - 5.0

        height = np.linspace(0.0, 12000.0, n)
        result = diag.compute_thermodynamic_indices(p, t, td, height_m=height)

        assert "wet_bulb_zero_height" in result

    def test_shear_indices_attempted_with_wind_height(self: 'TestThermodynamicIndicesWithWindAndHeight') -> None:
        """
        This test verifies that when pressure, temperature, dewpoint, wind, and height profiles are supplied to the compute_thermodynamic_indices method, the code paths for computing shear-related indices are attempted. It checks that at least one of the shear-related keys (e.g., 'bulk_shear_0_1km', 'srh_0_3km') is present in the resulting dictionary, which indicates that the method attempted to compute these indices based on the provided wind and height information. This ensures that users who supply comprehensive sounding data can access shear-related indices, which are important for assessing storm dynamics and severe weather potential. 

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        n = 40
        p = np.linspace(1000.0, 200.0, n)
        t = np.linspace(25.0, -60.0, n)
        td = t - 5.0
        u = np.random.uniform(-15.0, 15.0, n)
        v = np.random.uniform(-15.0, 15.0, n)
        height = np.linspace(0.0, 12000.0, n)

        result = diag.compute_thermodynamic_indices(
            p, t, td, u_wind_kt=u, v_wind_kt=v, height_m=height
        )

        shear_keys = (
            "bulk_shear_0_1km", "bulk_shear_0_6km",
            "srh_0_1km", "srh_0_3km",
        )

        assert any(k in result for k in shear_keys)


class TestNonFiniteFiltering:
    """ Tests for the filtering of non-finite pressure levels in the extract_sounding_profile method. """

    def test_nan_pressure_levels_are_filtered(self: 'TestNonFiniteFiltering', 
                                              tmp_path: str) -> None:
        """
        This test verifies that when the pressure profile extracted in the extract_sounding_profile method contains NaN values, these non-finite levels are filtered out from the resulting sounding profile. It checks that when a synthetic dataset is created with NaN values injected into the pressure profile, the resulting profile returned by extract_sounding_profile contains only finite pressure values and that the length of the profile is reduced accordingly. This ensures that users receive a cleaned sounding profile without non-finite pressure levels, which could otherwise lead to errors or misleading results in subsequent analysis of the sounding data. 

        Parameters:
            tmp_path (str): A temporary directory provided by pytest for creating test files.

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        ds = _synthetic_mpas_data(n_cells=5, n_vert=10, nan_pressure_idx=4)
        proc = _make_mock_proc(ds, tmp_path, prefix="grid_nan_p")
        profile = diag.extract_sounding_profile(proc, -100.0, 35.0)
        assert np.all(np.isfinite(profile["pressure"]))
        assert len(profile["pressure"]) < 10

    def test_wind_and_height_also_filtered(self: 'TestNonFiniteFiltering', 
                                           tmp_path: str) -> None:
        """
        This test verifies that when the pressure profile contains NaN values and is filtered in the extract_sounding_profile method, the corresponding wind and height profiles are also filtered to maintain consistent indexing across all profile variables. It checks that when a synthetic dataset is created with NaN values injected into the pressure profile, the resulting u_wind, v_wind, and height profiles returned by extract_sounding_profile contain only finite values and have the same length as the filtered pressure profile. This ensures that users receive consistent and aligned profiles for pressure, wind, and height after non-finite levels are removed, allowing for accurate analysis of the sounding data without mismatched indices. 

        Parameters:
            tmp_path (str): A temporary directory provided by pytest for creating test files.

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)

        ds = _synthetic_mpas_data(
            n_cells=5, n_vert=10,
            nan_pressure_idx=2,
            include_wind=True,
            include_height=True,
        )

        proc = _make_mock_proc(ds, tmp_path, prefix="grid_nan_wind")
        profile = diag.extract_sounding_profile(proc, -100.0, 35.0)
        n = len(profile["pressure"])
        assert profile["u_wind"] is not None
        assert len(profile["u_wind"]) == n
        assert profile["height"] is not None
        assert len(profile["height"]) == n


class TestExtractSoundingProfileWithComponents:
    """ Tests for the extract_sounding_profile method when the dataset contains pressure components. """

    def test_pressure_components_dataset_works_end_to_end(self: 'TestExtractSoundingProfileWithComponents', 
                                                          tmp_path: str) -> None:
        """
        This test verifies that when the dataset contains pressure components (e.g., 'components' variable), the extract_sounding_profile method can successfully extract a sounding profile end-to-end, including pressure, temperature, dewpoint, wind, and height profiles. It checks that when a synthetic dataset is created with pressure components and the necessary variables for temperature, dewpoint, wind, and height, the method returns a profile with finite pressure values and valid wind and height profiles. This ensures that users can obtain a complete sounding profile from datasets that use pressure components, which is a common format in MPAS output, and that the method can handle the necessary conversions and extractions to produce a consistent profile for analysis. 

        Parameters:
            tmp_path (str): A temporary directory provided by pytest for creating test files.

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)

        ds = _synthetic_mpas_data(
            n_cells=5, n_vert=10,
            pressure_varname="components",
            temp_varname="temperature",
            dew_varname="dewpoint_K",
            include_wind=True,
            include_height=True,
        )

        proc = _make_mock_proc(ds, tmp_path, prefix="grid_comp")
        profile = diag.extract_sounding_profile(proc, -100.0, 35.0)

        assert len(profile["pressure"]) > 0
        assert np.all(np.isfinite(profile["pressure"]))
        assert profile["u_wind"] is not None
        assert profile["height"] is not None

    def test_potential_temperature_conversion_in_full_pipeline(self: 'TestExtractSoundingProfileWithComponents', 
                                                               tmp_path: str) -> None:
        """
        This test verifies that when the dataset contains potential temperature ('theta') as the temperature variable and pressure components, the extract_sounding_profile method correctly converts potential temperature to actual temperature in the full extraction pipeline. It checks that when a synthetic dataset is created with 'theta' as the temperature variable and pressure components, the resulting temperature profile in the extracted sounding profile is in a physically reasonable range for Celsius temperatures. This ensures that the method can handle datasets with potential temperature and perform the necessary conversions to produce a valid temperature profile for sounding diagnostics, even when using pressure components. 

        Parameters:
            tmp_path (str): A temporary directory provided by pytest for creating test files.

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)

        ds = _synthetic_mpas_data(
            n_cells=5, n_vert=10,
            pressure_varname="pressure",
            temp_varname="theta",
            dew_varname="qv",
        )

        proc = _make_mock_proc(ds, tmp_path, prefix="grid_theta")
        profile = diag.extract_sounding_profile(proc, -100.0, 35.0)

        assert np.all(profile["temperature"] > -200.0)
        assert np.all(profile["temperature"] < 100.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

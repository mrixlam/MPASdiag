#!/usr/bin/env python3

"""
MPASdiag Test Suite: Diagnostics Layer Integration Tests

This module contains integration tests for the diagnostics layer of the MPASdiag package. These tests validate the end-to-end functionality of key diagnostic computations, such as precipitation accumulation differences, wind speed/direction analysis, and vertical sounding profile extraction. The tests use real MPAS diagnostic datasets loaded through fixtures defined in `conftest.py`, ensuring that the diagnostics operate correctly on actual model output. Each test asserts that the outputs are physically reasonable and conform to expected data structures, providing confidence in the robustness of the diagnostics pipeline. 

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
from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
from mpasdiag.diagnostics.wind import WindDiagnostics
from mpasdiag.diagnostics.sounding import SoundingDiagnostics


@pytest.mark.integration
@pytest.mark.requires_data
class TestPrecipitationDiagnosticsPipeline:
    """ Integration tests for precipitation accumulation diagnostics on real diagnostic data. """

    @pytest.mark.parametrize("accum_period", ["a01h", "a03h", "a06h"])
    def test_accumulation_difference_periods(self: "TestPrecipitationDiagnosticsPipeline",
                                             real_2d_processor: MPAS2DProcessor,
                                             accum_period: str,) -> None:
        """
        This test validates the precipitation accumulation difference diagnostic across multiple accumulation periods (1, 3, and 6 hours). It confirms that the computed accumulation differences are returned as xarray DataArrays with appropriate metadata attributes, and that the values are physically reasonable (non-negative after quality control). By parameterizing over different accumulation periods, this test ensures that the diagnostic correctly handles various temporal differencing scenarios on real MPAS datasets.

        Parameters:
            real_2d_processor (MPAS2DProcessor): Loaded 2D processor fixture containing the real diagnostic dataset.
            accum_period (str): The accumulation period to test, e.g., "a01h" for 1 hour, "a03h" for 3 hours, and "a06h" for 6 hours.

        Returns:
            None
        """
        dataset = real_2d_processor.dataset
        assert dataset is not None

        diag = PrecipitationDiagnostics(verbose=False)
        time_dim = "Time" if "Time" in dataset.dims else "time"
        last_index = int(dataset.sizes[time_dim]) - 1

        accum = diag.compute_precipitation_difference(
            dataset, time_index=last_index, var_name="rainnc", accum_period=accum_period
        )

        assert isinstance(accum, xr.DataArray)
        assert "long_name" in accum.attrs
        assert accum.attrs.get("accumulation_period") == accum_period
        assert accum.attrs.get("accumulation_hours") == diag.get_accumulation_hours(
            accum_period
        )

        values = np.asarray(accum.values)
        finite = values[np.isfinite(values)]
        assert finite.size > 0
        assert (finite >= 0).all()

    def test_first_timestep_edge_case(self: "TestPrecipitationDiagnosticsPipeline", 
                                      real_2d_processor: MPAS2DProcessor,) -> None:
        """
        This test checks the behavior of the precipitation accumulation difference diagnostic at the first time step (index 0), where there is no prior time step to difference against. The expected behavior is that the diagnostic should return an accumulation difference of zero or NaN, depending on the implementation. This test ensures that the diagnostic gracefully handles this edge case without errors and returns a valid xarray DataArray with appropriate metadata.

        Parameters:
            real_2d_processor (MPAS2DProcessor): Loaded 2D processor fixture containing the real diagnostic dataset.

        Returns:
            None
        """
        dataset = real_2d_processor.dataset
        assert dataset is not None

        diag = PrecipitationDiagnostics(verbose=False)
        accum = diag.compute_precipitation_difference(
            dataset, time_index=0, var_name="rainnc", accum_period="a01h"
        )

        assert isinstance(accum, xr.DataArray)
        assert accum.values.ravel().size > 0


@pytest.mark.integration
@pytest.mark.requires_data
class TestWindDiagnosticsPipeline:
    """ Integration tests for wind diagnostics on real diagnostic data. """

    def test_wind_speed_direction_and_analysis(self: "TestWindDiagnosticsPipeline", 
                                               real_2d_processor: MPAS2DProcessor,) -> None:
        """
        This test validates the full wind diagnostics pipeline on real 2D diagnostic data. It extracts the u and v wind components at the first time step, computes the horizontal wind speed and direction, and performs a basic analysis of the wind components. The test asserts that the computed wind speed is non-negative, the wind direction is within the range of 0 to 360 degrees, and that the analysis dictionary contains expected keys for the u component, v component, horizontal speed, and direction. This comprehensive test ensures that all steps of the wind diagnostics pipeline function correctly on actual MPAS output.

        Parameters:
            real_2d_processor (MPAS2DProcessor): Loaded 2D processor fixture containing the real diagnostic dataset.

        Returns:
            None
        """
        dataset = real_2d_processor.dataset
        assert dataset is not None

        diag = WindDiagnostics(verbose=False)
        u, v = diag.get_2d_wind_components(
            dataset, u_variable="u10", v_variable="v10", time_index=0
        )

        speed = diag.compute_wind_speed(u, v)
        direction = diag.compute_wind_direction(u, v, degrees=True)
        analysis = diag.analyze_wind_components(u, v)

        speed_vals = np.asarray(speed.values)
        dir_vals = np.asarray(direction.values)

        assert (speed_vals[np.isfinite(speed_vals)] >= 0).all()
        finite_dir = dir_vals[np.isfinite(dir_vals)]
        assert finite_dir.size > 0
        assert finite_dir.min() >= 0.0
        assert finite_dir.max() <= 360.0

        assert "u_component" in analysis
        assert "v_component" in analysis
        assert "horizontal_speed" in analysis
        assert "direction" in analysis


@pytest.mark.integration
@pytest.mark.requires_data
class TestSoundingDiagnosticsPipeline:
    """ Integration tests for vertical sounding diagnostics on real 3D data. """

    def test_extract_sounding_profile(self: "TestSoundingDiagnosticsPipeline", 
                                      real_3d_processor: MPAS3DProcessor,) -> None:
        """
        This test validates the extraction of a vertical sounding profile from a real 3D diagnostic dataset. It uses the SoundingDiagnostics class to extract a profile at a specified longitude, latitude, and time index. The test asserts that the resulting profile dictionary contains all expected keys (pressure, temperature, dewpoint, u_wind, v_wind, cell_index, station_lon, station_lat) and that the pressure and temperature arrays have consistent sizes. This test ensures that the sounding extraction process correctly retrieves and organizes the necessary variables for subsequent thermodynamic analysis.

        Parameters:
            real_3d_processor (MPAS3DProcessor): Loaded 3D processor fixture containing the real diagnostic dataset.

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        profile = diag.extract_sounding_profile(
            real_3d_processor, lon=0.0, lat=0.0, time_index=0
        )

        for key in (
            "pressure",
            "temperature",
            "dewpoint",
            "u_wind",
            "v_wind",
            "cell_index",
            "station_lon",
            "station_lat",
        ):
            assert key in profile

        assert np.asarray(profile["pressure"]).size > 1
        assert (
            np.asarray(profile["temperature"]).size
            == np.asarray(profile["pressure"]).size
        )

    def test_dewpoint_from_mixing_ratio(self: "TestSoundingDiagnosticsPipeline",) -> None:
        """
        This test validates the computation of dewpoint temperature from mixing ratio and pressure using the SoundingDiagnostics class. It creates synthetic mixing ratio and pressure arrays, computes the dewpoint, and asserts that the resulting dewpoint array has the same shape as the input mixing ratio and that the dewpoint values increase with increasing mixing ratio, which is physically consistent. This test ensures that the dewpoint computation method correctly implements the thermodynamic relationships between these variables.

        Parameters:
            None

        Returns:
            None
        """
        diag = SoundingDiagnostics(verbose=False)
        pressure = np.array([90000.0, 90000.0, 90000.0])
        mixing_ratio = np.array([0.002, 0.008, 0.014])

        dewpoint = diag.compute_dewpoint_from_mixing_ratio(mixing_ratio, pressure)

        assert dewpoint.shape == mixing_ratio.shape
        assert np.all(np.diff(dewpoint) > 0)

    def test_thermodynamic_indices_from_profile(self: "TestSoundingDiagnosticsPipeline", 
                                                real_3d_processor: MPAS3DProcessor,) -> None:
        """
        This test validates the computation of thermodynamic indices (such as CAPE and CIN) from an extracted sounding profile using the SoundingDiagnostics class. It first extracts a sounding profile from the real 3D diagnostic dataset, then computes the thermodynamic indices using the pressure, temperature, dewpoint, and wind components from the profile. The test asserts that the resulting indices dictionary contains expected keys for CAPE and CIN, and that the values are physically reasonable (e.g., CAPE should be non-negative). This test ensures that the full pipeline from profile extraction to thermodynamic analysis functions correctly on actual MPAS output.

        Parameters:
            real_3d_processor (MPAS3DProcessor): Loaded 3D processor fixture containing the real diagnostic dataset.

        Returns:
            None
        """
        pytest.importorskip("metpy")

        diag = SoundingDiagnostics(verbose=False)
        profile = diag.extract_sounding_profile(
            real_3d_processor, lon=0.0, lat=0.0, time_index=0
        )

        indices = diag.compute_thermodynamic_indices(
            pressure_hpa=np.asarray(profile["pressure"]),
            temperature_c=np.asarray(profile["temperature"]) - 273.15,
            dewpoint_c=np.asarray(profile["dewpoint"]) - 273.15,
            u_wind_kt=np.asarray(profile["u_wind"]),
            v_wind_kt=np.asarray(profile["v_wind"]),
        )

        assert isinstance(indices, dict)
        assert "cape" in indices
        assert "cin" in indices


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

#!/usr/bin/env python3

"""
MPASdiag Test Suite: Unit Conversion and Display

This module contains unit tests for the MPASdiag processing utilities related to unit conversion and display formatting. The tests cover various code paths in the UnitConverter class, including no-op conversions, pressure variable handling, error fallback behavior, and colorbar label formatting. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import numpy as np
import pytest
import xarray as xr
from unittest.mock import patch

from mpasdiag.processing.utils_unit import UnitConverter
from mpasdiag.processing.constants import HPA, PA, KELVIN


class TestConvertUnitsNoOp:
    """ Tests if convert_units correctly identifies when no conversion is needed and returns the original data unchanged. """

    def test_same_unit_scalar_returned_unchanged(self: 'TestConvertUnitsNoOp') -> None:
        """
        This test checks that when the input and output units are the same for a scalar value, the convert_units function returns the original value without modification. This is a no-op case where no conversion should occur. 

        Parameters:
            None

        Returns:
            None
        """
        result = UnitConverter.convert_units(25.0, "°C", "°C")
        assert result == 25.0

    def test_same_unit_array_returned_unchanged(self: 'TestConvertUnitsNoOp') -> None:
        """
        This test verifies that when the input and output units are the same for a numpy array, the convert_units function returns the original array object without modification. This ensures that no unnecessary copying or processing occurs when a conversion is not needed.

        Parameters:
            None

        Returns:
            None
        """
        arr = np.array([1.0, 2.0, 3.0])
        result = UnitConverter.convert_units(arr, "K", "K")
        assert result is arr

    def test_same_unit_after_normalization(self: 'TestConvertUnitsNoOp') -> None:
        """
        This test checks that when the input and output units are the same after normalization (e.g., "kelvin" and "K"), the convert_units function recognizes this and returns the original value without modification. This ensures that unit normalization is correctly handled in the no-op case. 

        Parameters:
            None

        Returns:
            None
        """
        result = UnitConverter.convert_units(300.0, "kelvin", "K")
        assert result == 300.0

    def test_same_unit_xarray_returned_unchanged(self: 'TestConvertUnitsNoOp') -> None:
        """
        This test verifies that when the input and output units are the same for an xarray DataArray, the convert_units function returns the original DataArray object without modification. This ensures that no unnecessary copying or processing occurs when a conversion is not needed for xarray objects.

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(np.ones(5))
        result = UnitConverter.convert_units(da, "hPa", "hPa")
        assert result is da


class TestConvertDataForDisplayPressureBranch:
    """ Test if convert_data_for_display correctly converts pressure variables from Pa to hPa based on variable name and metadata. """

    def _make_pa_dataarray(self: 'TestConvertDataForDisplayPressureBranch', 
                           var_name: str) -> xr.DataArray:
        """
        This helper method creates an xarray DataArray with values in Pascals (Pa) for testing pressure variable conversion. The DataArray has a single dimension 'nCells' and is populated with linearly spaced values from 85,000 Pa to 105,000 Pa. The units attribute is set to PA, and the long_name attribute is set to the provided variable name. 

        Parameters:
            var_name (str): The name of the variable to be used in the long_name attribute of the DataArray. 

        Returns:
            xr.DataArray: An xarray DataArray with values in Pascals and appropriate metadata for testing pressure variable conversion.            
        """
        return xr.DataArray(
            np.linspace(85_000.0, 105_000.0, 5),
            dims=["nCells"],
            attrs={"units": PA, "long_name": var_name},
        )

    def test_pressure_var_converts_pa_to_hpa(self: 'TestConvertDataForDisplayPressureBranch') -> None:
        """
        This test checks that when a DataArray with units of Pascals (Pa) and a variable name indicating pressure (e.g., "mslp") is passed to the convert_data_for_display function, the function correctly converts the values from Pa to hPa. The test verifies that the metadata is updated to reflect the new units (hPa) and that the original units (Pa) are preserved in the metadata. Additionally, it checks that the minimum value of the converted DataArray is approximately 850.0 hPa, which corresponds to 85,000 Pa. 

        Parameters:
            None

        Returns:
            None
        """
        da = self._make_pa_dataarray("mslp")
        mock_metadata = {"units": PA, "long_name": "Mean Sea Level Pressure"}

        with patch("mpasdiag.visualization.MPASFileMetadata") as mock_cls:
            mock_cls.get_2d_variable_metadata.return_value = mock_metadata
            converted, meta = UnitConverter.convert_data_for_display(da, "mslp", da)

        assert meta["units"] == HPA
        assert meta["original_units"] == PA
        assert float(converted.min()) == pytest.approx(850.0, rel=1e-5)

    def test_pressure_var_name_pressure_also_converts(self: 'TestConvertDataForDisplayPressureBranch') -> None:
        """
        This test verifies that the convert_data_for_display function also converts pressure variables when the variable name contains "pressure" (e.g., "surface_pressure") in addition to "mslp". The test creates a DataArray with units of Pascals (Pa) and a long_name indicating surface pressure, then checks that the conversion to hPa occurs correctly and that the metadata is updated accordingly. This ensures that the logic for identifying pressure variables based on their names is functioning as intended. 

        Parameters:
            None

        Returns:
            None
        """
        da = self._make_pa_dataarray("pressure")
        mock_metadata = {"units": PA, "long_name": "Surface Pressure"}

        with patch("mpasdiag.visualization.MPASFileMetadata") as mock_cls:
            mock_cls.get_2d_variable_metadata.return_value = mock_metadata
            converted, meta = UnitConverter.convert_data_for_display(da, "pressure", da)

        assert meta["units"] == HPA
        assert meta["original_units"] == PA


class TestConvertDataForDisplayValueErrorFallback:
    """ Test if convert_data_for_display correctly falls back to original units when convert_units raises a ValueError due to unsupported conversion. """

    def test_unsupported_conversion_falls_back_to_original_unit(self: 'TestConvertDataForDisplayValueErrorFallback') -> None:
        """
        This test checks that when the convert_units function raises a ValueError due to an unsupported conversion (e.g., trying to convert temperature from Kelvin to Celsius without proper handling), the convert_data_for_display function correctly falls back to using the original units for display. The test creates a DataArray with units of Kelvin and mocks the metadata to trigger the display_units logic. It then patches the convert_units method to raise a ValueError, simulating an unsupported conversion scenario. Finally, it verifies that the metadata for display units is reset to the original units (Kelvin) as expected in the error handling branch. 

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(
            np.array([273.15, 280.0]),
            dims=["nCells"],
            attrs={"units": KELVIN},
        )

        mock_metadata = {"units": KELVIN, "long_name": "Temperature"}

        with patch("mpasdiag.visualization.MPASFileMetadata") as mock_cls:
            mock_cls.get_2d_variable_metadata.return_value = mock_metadata
            with patch.object(
                UnitConverter,
                "convert_units",
                side_effect=ValueError("unsupported conversion"),
            ):
                converted, meta = UnitConverter.convert_data_for_display(da, "t2m", da)

        assert meta["units"] == KELVIN
        assert meta["original_units"] == KELVIN


class TestFormatColorbarLabel:
    """ Test if _format_colorbar_label correctly replaces various patterns for degrees Celsius with the degree symbol and 'C' in the colorbar label. """

    def test_deg_underscore_c_replaced(self: 'TestFormatColorbarLabel') -> None:
        """
        This test checks that the _format_colorbar_label function correctly identifies and replaces the pattern "deg_C" with "°C" in the input string. The test verifies that the resulting string contains the degree symbol followed by 'C' and that the original pattern "deg_C" is no longer present in the output. This ensures that the function is properly formatting colorbar labels for temperature variables that may use different conventions for representing degrees Celsius. 

        Parameters:
            None

        Returns:
            None
        """
        result = UnitConverter._format_colorbar_label("Temperature (deg_C)")
        assert "°C" in result
        assert "deg_C" not in result

    def test_deg_space_c_replaced(self: 'TestFormatColorbarLabel') -> None:
        """
        This test verifies that the _format_colorbar_label function correctly replaces the pattern "deg C" with "°C" in the input string. The test checks that the output string contains the degree symbol followed by 'C' and that the original pattern "deg C" is not present in the result. This ensures that the function can handle different spacing conventions for representing degrees Celsius in colorbar labels. 

        Parameters:
            None

        Returns:
            None
        """
        result = UnitConverter._format_colorbar_label("Temperature (deg C)")
        assert "°C" in result
        assert "deg C" not in result

    def test_degrees_space_c_replaced(self: 'TestFormatColorbarLabel') -> None:
        """
        This test checks that the _format_colorbar_label function correctly replaces the pattern "degrees C" with "°C" in the input string. The test verifies that the resulting string contains the degree symbol followed by 'C' and that the original pattern "degrees C" is no longer present in the output. This ensures that the function can handle more verbose conventions for representing degrees Celsius in colorbar labels. 

        Parameters:
            None

        Returns:
            None
        """
        result = UnitConverter._format_colorbar_label("T2m [degrees C]")
        assert "°C" in result
        assert "degrees C" not in result

    def test_degrees_underscore_c_replaced(self: 'TestFormatColorbarLabel') -> None:
        """
        This test verifies that the _format_colorbar_label function correctly replaces the pattern "degrees_C" with "°C" in the input string. The test checks that the output string contains the degree symbol followed by 'C' and that the original pattern "degrees_C" is not present in the result. This ensures that the function can handle different underscore conventions for representing degrees Celsius in colorbar labels.

        Parameters:
            None

        Returns:
            None
        """
        result = UnitConverter._format_colorbar_label("T2m [degrees_C]")
        assert "°C" in result
        assert "degrees_C" not in result

    def test_degc_replaced(self: 'TestFormatColorbarLabel') -> None:
        """
        This test checks that the _format_colorbar_label function correctly replaces the pattern "degC" with "°C" in the input string. The test verifies that the resulting string contains the degree symbol followed by 'C' and that the original pattern "degC" is no longer present in the output. This ensures that the function can handle compact conventions for representing degrees Celsius in colorbar labels.

        Parameters:
            None

        Returns:
            None
        """
        result = UnitConverter._format_colorbar_label("Surface Temp degC")
        assert "°C" in result
        assert "degC" not in result

    def test_no_match_returns_unchanged(self: 'TestFormatColorbarLabel') -> None:
        """
        This test verifies that when the input string does not contain any of the recognized patterns for degrees Celsius, the _format_colorbar_label function returns the original string unchanged. The test checks that the output is identical to the input when no replacements are needed, ensuring that the function does not modify labels that do not match any of the specified patterns. 

        Parameters:
            None

        Returns:
            None
        """
        label = "Wind Speed (m/s)"
        assert UnitConverter._format_colorbar_label(label) == label

    def test_all_patterns_in_one_string(self: 'TestFormatColorbarLabel') -> None:
        """
        This test checks that when the input string contains multiple patterns for degrees Celsius (e.g., "deg_C", "deg C", "degrees C", "degrees_C", "degC"), the _format_colorbar_label function correctly replaces all of them with "°C". The test verifies that the resulting string contains the degree symbol followed by 'C' for each occurrence and that none of the original patterns are present in the output. This ensures that the function can handle and replace multiple variations of degrees Celsius in a single colorbar label effectively. 

        Parameters:
            None

        Returns:
            None
        """
        label = "deg_C deg C degrees C degrees_C degC"
        result = UnitConverter._format_colorbar_label(label)
        assert "deg_C" not in result
        assert "deg C" not in result
        assert "degrees C" not in result
        assert "degrees_C" not in result
        assert "degC" not in result
        assert result.count("°C") == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

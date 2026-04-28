#!/usr/bin/env python3

"""
MPASdiag Test Suite: MPASFileMetadata Validation and Fallbacks

This module contains unit tests for the MPASFileMetadata class, specifically targeting the handling of variable metadata and the behavior of fallback mechanisms when certain metadata attributes are missing. The tests cover scenarios such as the use of standard_name as a fallback for long_name, the assignment of default visualization settings for unknown variables, and the retrieval of available variables from the metadata. The tests are designed to ensure that the MPASFileMetadata class behaves as expected in various edge cases, providing robustness and reliability in the processing of MPAS data. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import numpy as np
import pytest
import xarray as xr

from mpasdiag.processing.utils_metadata import MPASFileMetadata


class TestGetVariableMetadataStandardNameFallback:
    """ Test if standard_name is used as a fallback for long_name when long_name is missing. """

    def test_standard_name_used_when_long_name_absent(self: 'TestGetVariableMetadataStandardNameFallback') -> None:
        """
        This test verifies that when a DataArray is provided to MPASFileMetadata.get_variable_metadata without a long_name attribute but with a standard_name attribute, the resulting metadata dictionary correctly uses the standard_name value as the long_name. This ensures that users can still get meaningful long_name values in the metadata even if the original DataArray does not include a long_name, as long as it has a standard_name. The test creates a DataArray with only a standard_name and checks that the returned metadata contains the expected long_name derived from the standard_name.

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(
            np.ones(5),
            dims=["nCells"],
            attrs={"units": "K", "standard_name": "air_temperature"},
        )

        meta = MPASFileMetadata.get_variable_metadata("t2m", da)
        assert meta["long_name"] == "air_temperature"

    def test_long_name_takes_precedence_over_standard_name(self: 'TestGetVariableMetadataStandardNameFallback') -> None:
        """
        This test checks that if both long_name and standard_name attributes are present in the DataArray, the long_name is used in the metadata instead of the standard_name. This ensures that the more descriptive long_name is preferred when available, while still allowing for a fallback to standard_name when long_name is missing. The test creates a DataArray with both attributes and verifies that the long_name value is what appears in the resulting metadata. 

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(
            np.ones(5),
            dims=["nCells"],
            attrs={"long_name": "2-metre Temperature", "standard_name": "air_temperature"},
        )

        meta = MPASFileMetadata.get_variable_metadata("t2m", da)
        assert meta["long_name"] == "2-metre Temperature"

    def test_standard_name_fallback_on_unknown_var(self: 'TestGetVariableMetadataStandardNameFallback') -> None:
        """
        This test verifies that when an unknown variable name is provided to MPASFileMetadata.get_variable_metadata along with a DataArray that has a standard_name attribute but no long_name, the standard_name is correctly used as the long_name in the resulting metadata. This ensures that even for variables that are not recognized by name, users can still get meaningful metadata based on the attributes of the provided DataArray. The test creates a DataArray with a standard_name and checks that the returned metadata contains the expected long_name derived from the standard_name, despite the variable name being unknown.

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(
            np.ones(3),
            dims=["nCells"],
            attrs={"standard_name": "surface_air_pressure"},
        )

        meta = MPASFileMetadata.get_variable_metadata("unknown_var_xyz", da)
        assert meta["long_name"] == "surface_air_pressure"

    def test_neither_long_name_nor_standard_name_leaves_default(self: 'TestGetVariableMetadataStandardNameFallback') -> None:
        """
        This test checks that if a DataArray is provided without either long_name or standard_name attributes, the resulting metadata does not have a long_name or has it set to a default value (e.g., an empty string). This ensures that the get_variable_metadata method does not produce misleading metadata when no descriptive attributes are available. The test creates a DataArray with no long_name or standard_name and verifies that the long_name in the resulting metadata is either missing or set to a default value. 

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(np.ones(3), dims=["nCells"], attrs={"units": "Pa"})
        meta = MPASFileMetadata.get_variable_metadata("mslp", da)
        assert "long_name" in meta
        assert meta["long_name"] != ""


class TestGetVariableMetadataVisualizationFallback:
    """ Tests for the fallback behavior of visualization metadata when include_visualization=True is passed to get_variable_metadata. """

    def test_unknown_var_gets_viridis_colormap(self: 'TestGetVariableMetadataVisualizationFallback') -> None:
        """
        This test verifies that when an unknown variable name is provided to MPASFileMetadata.get_variable_metadata with include_visualization set to True, the resulting metadata includes a colormap attribute set to "viridis". This ensures that even for variables that are not recognized, users can still get a default visualization setting that allows them to visualize the data without needing to specify a custom colormap. The test calls get_variable_metadata with an unknown variable name and checks that the colormap in the returned metadata is "viridis". 

        Parameters:
            None

        Returns:
            None
        """
        meta = MPASFileMetadata.get_variable_metadata(
            "completely_unknown_variable", include_visualization=True
        )

        assert meta["colormap"] == "viridis"
        assert meta["levels"] is None
        assert meta["spatial_dims"] == 2

    def test_known_var_does_not_use_viridis_fallback(self: 'TestGetVariableMetadataVisualizationFallback') -> None:
        """
        This test checks that when a known variable name is provided to MPASFileMetadata.get_variable_metadata with include_visualization set to True, the resulting metadata does not use the "viridis" colormap as a fallback, but instead uses the appropriate colormap for that variable (if defined). This ensures that the fallback to "viridis" only occurs for unknown variables and does not override the intended visualization settings for recognized variables. The test calls get_variable_metadata with a known variable name (e.g., "t2m") and checks that the colormap in the returned metadata is not "viridis". 

        Parameters:
            None

        Returns:
            None
        """
        meta = MPASFileMetadata.get_variable_metadata("t2m", include_visualization=True)
        assert meta["colormap"] != "viridis"

    def test_unknown_var_without_include_viz_has_no_colormap(self: 'TestGetVariableMetadataVisualizationFallback') -> None:
        """
        This test verifies that when an unknown variable name is provided to MPASFileMetadata.get_variable_metadata without include_visualization set to True, the resulting metadata does not include a colormap attribute. This ensures that the visualization fallback only applies when explicitly requested by the user, and that metadata for unknown variables does not contain irrelevant visualization settings when include_visualization is False. The test calls get_variable_metadata with an unknown variable name and checks that the colormap attribute is not present in the returned metadata. 

        Parameters:
            None

        Returns:
            None
        """
        meta = MPASFileMetadata.get_variable_metadata(
            "completely_unknown_variable", include_visualization=False
        )

        assert "colormap" not in meta

    def test_unknown_var_with_data_array_and_viz(self: 'TestGetVariableMetadataVisualizationFallback') -> None:
        """
        This test checks that when an unknown variable name is provided to MPASFileMetadata.get_variable_metadata along with a DataArray that has a standard_name attribute, and include_visualization is set to True, the resulting metadata includes the "viridis" colormap as a fallback and also uses the standard_name as the long_name. This ensures that even for unknown variables, users can get meaningful metadata for both visualization and descriptive purposes when a DataArray with attributes is provided. The test creates a DataArray with a standard_name, calls get_variable_metadata with an unknown variable name and include_visualization=True, and checks that the colormap is "viridis" and the long_name is derived from the standard_name. 

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(
            np.ones(4),
            dims=["nCells"],
            attrs={"units": "m/s", "standard_name": "eastward_wind"},
        )

        meta = MPASFileMetadata.get_variable_metadata(
            "my_custom_wind_var", da, include_visualization=True
        )
        
        assert meta["colormap"] == "viridis"
        assert meta["long_name"] == "eastward_wind"


class TestGetAvailableVariables:
    """ Tests for MPASFileMetadata.get_available_variables. """

    def test_returns_list(self: 'TestGetAvailableVariables') -> None:
        """
        This test verifies that the get_available_variables method of MPASFileMetadata returns a list of variable names. This ensures that the method is providing the expected data type for the available variables, which is important for users who will be using this list to check for variable availability or to iterate over available variables. The test calls get_available_variables and checks that the result is an instance of list. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASFileMetadata.get_available_variables()
        assert isinstance(result, list)

    def test_list_is_nonempty(self: 'TestGetAvailableVariables') -> None:
        """
        This test checks that the list returned by get_available_variables is not empty, indicating that there are indeed variables available in the metadata. This is important to ensure that the metadata is properly populated and that users have access to variable information. The test calls get_available_variables and asserts that the length of the resulting list is greater than zero. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASFileMetadata.get_available_variables()
        assert len(result) > 0

    def test_contains_common_surface_vars(self: 'TestGetAvailableVariables') -> None:
        """
        This test verifies that the list of available variables returned by get_available_variables includes common surface variables such as "t2m", "mslp", "u10", "v10", and "rainnc". This ensures that users can expect to find these commonly used variables in the metadata, which is important for typical use cases involving surface meteorological data. The test calls get_available_variables and checks that each of the specified variable names is present in the resulting list. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASFileMetadata.get_available_variables()
        for var in ("t2m", "mslp", "u10", "v10", "rainnc"):
            assert var in result, f"Expected '{var}' in available variables"

    def test_contains_upper_air_vars(self: 'TestGetAvailableVariables') -> None:
        """
        This test checks that the list of available variables returned by get_available_variables includes common upper air variables such as "t500hPa" and "t850hPa". This ensures that users can expect to find these important upper air variables in the metadata, which is crucial for analyses involving atmospheric profiles and dynamics. The test calls get_available_variables and asserts that at least one variable containing "500hPa" or "850hPa" is present in the resulting list.

        Parameters:
            None

        Returns:
            None
        """
        result = MPASFileMetadata.get_available_variables()
        assert any("500hPa" in v or "850hPa" in v for v in result)

    def test_all_entries_are_strings(self: 'TestGetAvailableVariables') -> None:
        """
        This test verifies that all entries in the list returned by get_available_variables are strings, which is important for ensuring that the variable names are in the expected format and can be used reliably in subsequent processing steps. The test calls get_available_variables and checks that each entry in the resulting list is an instance of str. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASFileMetadata.get_available_variables()
        assert all(isinstance(v, str) for v in result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

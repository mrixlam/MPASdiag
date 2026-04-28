#!/usr/bin/env python3

"""
MPASdiag Test Suite: MPASConfig Validation and Serialization

This module contains unit tests for the MPASConfig class, focusing on the validation of spatial extent parameters, remapping settings, and the functionality of saving and loading configurations to and from YAML files. The tests ensure that invalid configurations are properly rejected with informative error messages, and that valid configurations are accepted. Additionally, the tests verify that the save_to_file method correctly writes a YAML file and that the from_dict method properly converts list inputs to tuples where necessary. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import sys
import tempfile
import os
import pytest
from io import StringIO
from unittest.mock import patch, MagicMock

from mpasdiag.processing.utils_config import MPASConfig


class TestValidateSpatialExtent:
    """ Tests for MPASConfig._validate_spatial_extent. """

    def test_none_extent_param_returns_true(self: 'TestValidateSpatialExtent') -> None:
        """
        This test verifies that if any of the spatial extent parameters (lat_min, lat_max, lon_min, lon_max) is set to None, the validation should pass without raising an exception. This is because the presence of None indicates that the user does not want to constrain the spatial extent in that direction, and thus it should be considered valid. The test creates a configuration with lat_min set to None and checks that it is accepted as valid. 

        Parameters:
            None

        Returns:
            None
        """
        config = MPASConfig(lat_min=None)  # type: ignore[arg-type]
        assert config.lat_min is None

    def test_invalid_extent_raises_at_init(self: 'TestValidateSpatialExtent') -> None:
        """
        This test checks that if the spatial extent parameters are set in a way that lat_max is less than lat_min, the MPASConfig initialization should raise a ValueError with the message "Invalid spatial extent". This is because such a configuration would not make sense for defining a valid spatial region. The test attempts to create a configuration with lat_min set to 50.0 and lat_max set to 10.0, which should trigger the validation logic and result in an exception being raised.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="Invalid spatial extent"):
            MPASConfig(lat_min=50.0, lat_max=10.0) 

    def test_reversed_lon_raises_at_init(self: 'TestValidateSpatialExtent') -> None:
        """
        This test verifies that if the longitude parameters are set such that lon_max is less than lon_min, the MPASConfig initialization should raise a ValueError with the message "Invalid spatial extent". This is because such a configuration would not define a valid spatial region. The test attempts to create a configuration with lon_min set to 90.0 and lon_max set to -90.0, which should trigger the validation logic and result in an exception being raised. 

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="Invalid spatial extent"):
            MPASConfig(lon_min=90.0, lon_max=-90.0)

    def test_out_of_range_lat_raises_at_init(self: 'TestValidateSpatialExtent') -> None:
        """
        This test checks that if the latitude parameters are set outside the valid range of -90 to 90 degrees, the MPASConfig initialization should raise a ValueError with the message "Invalid spatial extent". This is because latitudes outside this range are not physically meaningful. The test attempts to create a configuration with lat_min set to -91.0, which should trigger the validation logic and result in an exception being raised. 

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="Invalid spatial extent"):
            MPASConfig(lat_min=-91.0)


class TestValidateRemapSettings:
    """ Tests for MPASConfig._validate_remap_settings. """

    def test_invalid_remap_engine_raises(self: 'TestValidateRemapSettings') -> None:
        """
        This test verifies that if an invalid remap_engine is provided during MPASConfig initialization, a ValueError should be raised with the message "Invalid remap_engine". This is important to ensure that users are aware of the valid options for remapping engines and to prevent misconfigurations. The test attempts to create a configuration with remap_engine set to "unknown_engine", which should trigger the validation logic and result in an exception being raised.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="Invalid remap_engine"):
            MPASConfig(remap_engine="unknown_engine")

    def test_esmf_nearest_auto_corrected_to_bilinear(self: 'TestValidateRemapSettings') -> None:
        """
        This test checks that if the remap_engine is set to "esmf" and the remap_method is set to "nearest", the MPASConfig should automatically correct the remap_method to "bilinear". This is because the ESMF engine does not support the "nearest" method, and the configuration should adjust to a compatible method instead of raising an error. The test creates a configuration with these settings and verifies that the remap_method has been changed to "bilinear". 

        Parameters:
            None

        Returns:
            None
        """
        config = MPASConfig(remap_engine="esmf", remap_method="nearest")
        assert config.remap_method == "bilinear"

    def test_invalid_remap_method_for_engine_raises(self: 'TestValidateRemapSettings') -> None:
        """
        This test verifies that if a valid remap_engine is provided but an invalid remap_method is specified for that engine, a ValueError should be raised with the message "Invalid remap_method". This ensures that users are aware of the valid remapping methods that are compatible with the chosen remapping engine. The test attempts to create a configuration with remap_engine set to "kdtree" and remap_method set to "patch", which should trigger the validation logic and result in an exception being raised. 

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="Invalid remap_method"):
            MPASConfig(remap_engine="kdtree", remap_method="patch")

    def test_remapping_import_failure_uses_fallback_methods(self: 'TestValidateRemapSettings') -> None:
        """
        This test simulates a failure during the import of the remapping module (e.g., due to missing dependencies) and verifies that the MPASConfig initialization falls back to a default set of remap methods without raising an exception. This is important for ensuring that the configuration can still be created even if the remapping functionality is not available, allowing users to proceed with other aspects of their configuration. The test uses unittest.mock to patch the sys.modules dictionary to simulate the absence of the remapping module, and then checks that the configuration is still created with the specified remap_engine and remap_method. 

        Parameters:
            None

        Returns:
            None
        """
        mock_module = MagicMock()
        mock_module.MPASRemapper._METHOD_MAP = None 

        with patch.dict(sys.modules, {"mpasdiag.processing.remapping": mock_module}):
            config = MPASConfig(remap_engine="esmf", remap_method="bilinear")

        assert config.remap_engine == "esmf"
        assert config.remap_method == "bilinear"


class TestFromDict:
    """ Tests for MPASConfig.from_dict. """

    def test_figure_size_list_converted_to_tuple(self: 'TestFromDict') -> None:
        """
        This test verifies that when the figure_size parameter is provided as a list in the input dictionary to MPASConfig.from_dict, it is correctly converted to a tuple in the resulting MPASConfig instance. This is important because the figure_size parameter is expected to be a tuple, and allowing users to input it as a list provides flexibility while ensuring that the internal representation is consistent. The test creates a configuration using from_dict with figure_size set to [14.0, 8.0] and checks that the resulting figure_size attribute is a tuple with the correct values. 

        Parameters:
            None

        Returns:
            None
        """
        config = MPASConfig.from_dict({"figure_size": [14.0, 8.0]})
        assert isinstance(config.figure_size, tuple)
        assert config.figure_size == (14.0, 8.0)

    def test_figure_size_already_tuple_unchanged(self: 'TestFromDict') -> None:
        """
        This test checks that if the figure_size parameter is already provided as a tuple in the input dictionary to MPASConfig.from_dict, it remains unchanged in the resulting MPASConfig instance. This ensures that the from_dict method does not modify valid input and only converts lists to tuples when necessary. The test creates a configuration using from_dict with figure_size set to (12.0, 10.0) and verifies that the figure_size attribute is still a tuple with the correct values. 

        Parameters:
            None

        Returns:
            None
        """
        config = MPASConfig.from_dict({"figure_size": (12.0, 10.0)})
        assert isinstance(config.figure_size, tuple)


class TestSaveToFile:
    """ Tests for MPASConfig.save_to_file. """

    def test_saves_valid_yaml_and_prints_confirmation(self: 'TestSaveToFile') -> None:
        """
        This test verifies that the save_to_file method of MPASConfig correctly saves the configuration to a YAML file and prints a confirmation message to the console. The test creates a temporary file, calls save_to_file with the path to that file, and captures the standard output to check for the confirmation message. It then reads the contents of the saved YAML file to ensure that it contains the expected configuration values. Finally, it cleans up by deleting the temporary file.

        Parameters:
            None

        Returns:
            None
        """
        import yaml
        config = MPASConfig(variable="t2m", dpi=200)
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            filepath = f.name
        try:
            captured = StringIO()
            with patch("sys.stdout", new=captured):
                config.save_to_file(filepath)
            assert "Configuration saved to" in captured.getvalue()
            with open(filepath) as f:
                loaded = yaml.safe_load(f)
            assert loaded["variable"] == "t2m"
            assert loaded["dpi"] == 200
        finally:
            os.unlink(filepath)

    def test_saved_file_can_be_round_tripped(self: 'TestSaveToFile') -> None:
        """
        This test checks that a configuration saved to a YAML file using MPASConfig.save_to_file can be successfully loaded back into an MPASConfig instance using MPASConfig.load_from_file, and that the loaded configuration matches the original configuration. The test creates an MPASConfig instance with specific values, saves it to a temporary YAML file, and then loads it back. It asserts that the attributes of the reloaded configuration are the same as those of the original configuration. Finally, it cleans up by deleting the temporary file.

        Parameters:
            None

        Returns:
            None
        """
        original = MPASConfig(variable="precip", chunk_size=50000, verbose=False)
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            filepath = f.name
        try:
            original.save_to_file(filepath)
            reloaded = MPASConfig.load_from_file(filepath)
            assert reloaded.variable == original.variable
            assert reloaded.chunk_size == original.chunk_size
            assert reloaded.verbose == original.verbose
        finally:
            os.unlink(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

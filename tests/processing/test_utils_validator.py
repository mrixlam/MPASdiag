#!/usr/bin/env python3

"""
MPASdiag Test Suite: Data Validation

This module contains unit tests for the DataValidator class in mpasdiag.processing.utils_validator. The tests cover various scenarios for validating coordinate arrays and data arrays, including checks for valid and invalid inputs, edge cases, and integration with real MPAS data fixtures. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import numpy as np
import pytest
from typing import Tuple

from mpasdiag.processing.utils_validator import DataValidator


class TestValidateCoordinates:
    """ Test if DataValidator.validate_coordinates correctly identifies valid and invalid coordinate arrays. """

    def test_valid_normal(self: 'TestValidateCoordinates') -> None:
        """
        This test verifies that the validation method correctly identifies normal longitude and latitude arrays as valid. It uses longitude values ranging from -120 to -80 degrees and latitude values ranging from 25 to 55 degrees, which are typical for many geographic datasets. The test asserts that the validation method returns True for these coordinate arrays. 

        Parameters:
            None
        
        Returns:
            None
        """
        lon = np.linspace(-120.0, -80.0, 50)
        lat = np.linspace(25.0, 55.0, 50)
        assert DataValidator.validate_coordinates(lon, lat) is True

    def test_valid_boundary_values(self: 'TestValidateCoordinates') -> None:
        """
        This test verifies that the validation method correctly identifies longitude and latitude arrays that contain boundary values as valid. It uses longitude values at the extreme ends of the valid range (-180, 0, 180) and latitude values at their extremes (-90, 0, 90). The test asserts that these boundary values are accepted as valid coordinates by the validation method. 

        Parameters:
            None
        
        Returns:
            None
        """
        lon = np.array([-180.0, 0.0, 180.0])
        lat = np.array([-90.0, 0.0, 90.0])
        assert DataValidator.validate_coordinates(lon, lat) is True

    def test_valid_single_element(self: 'TestValidateCoordinates') -> None:
        """
        This test checks that the validation method correctly identifies longitude and latitude arrays with a single valid element as valid. It uses a single longitude value of 45.0 and a single latitude value of 30.0, which are well within the acceptable ranges for geographic coordinates. The test asserts that the validation method returns True for these single-element coordinate arrays. 

        Parameters:
            None
        
        Returns:
            None
        """
        lon = np.array([45.0])
        lat = np.array([30.0])
        assert DataValidator.validate_coordinates(lon, lat) is True

    def test_valid_zeros(self: 'TestValidateCoordinates') -> None:
        """
        This test verifies that the validation method correctly identifies longitude and latitude arrays that consist entirely of zeros as valid. While an array of zeros may not represent a meaningful geographic location, it is technically within the valid range for both longitude and latitude. The test asserts that the validation method returns True for these zero-valued coordinate arrays. 

        Parameters:
            None
        
        Returns:
            None
        """
        lon = np.zeros(10)
        lat = np.zeros(10)
        assert DataValidator.validate_coordinates(lon, lat) is True

    def test_invalid_length_mismatch(self: 'TestValidateCoordinates') -> None:
        """
        This test checks that the validation method correctly identifies longitude and latitude arrays of different lengths as invalid. It uses a longitude array with 10 elements and a latitude array with 15 elements, which should trigger a length mismatch error. The test asserts that the validation method returns False for these mismatched coordinate arrays. 

        Parameters:
            None
        
        Returns:
            None
        """
        lon = np.linspace(-90.0, 90.0, 10)
        lat = np.linspace(-45.0, 45.0, 15)
        assert DataValidator.validate_coordinates(lon, lat) is False

    def test_invalid_nan_in_lon(self: 'TestValidateCoordinates') -> None:
        """
        This test verifies that the validation method correctly identifies longitude arrays containing NaN values as invalid. It uses a longitude array with a NaN value in the middle and a corresponding latitude array with valid values. The presence of NaN in the longitude array should cause the validation method to return False, indicating that the coordinates are not valid. 

        Parameters:
            None
        
        Returns:
            None
        """
        lon = np.array([0.0, np.nan, 45.0])
        lat = np.array([10.0, 20.0, 30.0])
        assert DataValidator.validate_coordinates(lon, lat) is False

    def test_invalid_nan_in_lat(self: 'TestValidateCoordinates') -> None:
        """
        This test checks that the validation method correctly identifies latitude arrays containing NaN values as invalid. It uses a longitude array with valid values and a latitude array with a NaN value in the middle. The presence of NaN in the latitude array should cause the validation method to return False, indicating that the coordinates are not valid.

        Parameters:
            None

        Returns:
            None
        """
        lon = np.array([0.0, 45.0, 90.0])
        lat = np.array([10.0, np.nan, 30.0])
        assert DataValidator.validate_coordinates(lon, lat) is False

    def test_invalid_inf_in_lon(self: 'TestValidateCoordinates') -> None:
        """
        This test verifies that the validation method correctly identifies longitude arrays containing infinite values as invalid. It uses a longitude array with an infinite value and a valid latitude array. The test asserts that the validation method returns False for these coordinate arrays. 

        Parameters:
            None

        Returns:
            None
        """
        lon = np.array([0.0, np.inf, 45.0])
        lat = np.array([10.0, 20.0, 30.0])
        assert DataValidator.validate_coordinates(lon, lat) is False

    def test_invalid_inf_in_lat(self: 'TestValidateCoordinates') -> None:
        """
        This test checks that the validation method correctly identifies latitude arrays containing infinite values as invalid. It uses a longitude array with valid values and a latitude array with an infinite value. The presence of an infinite value in the latitude array should cause the validation method to return False, indicating that the coordinates are not valid.

        Parameters:
            None

        Returns:
            None
        """
        lon = np.array([0.0, 45.0, 90.0])
        lat = np.array([10.0, -np.inf, 30.0])
        assert DataValidator.validate_coordinates(lon, lat) is False

    def test_invalid_lon_exceeds_upper(self: 'TestValidateCoordinates') -> None:
        """
        This test verifies that the validation method correctly identifies longitude arrays containing values that exceed the upper limit of 180 degrees as invalid. It uses a longitude array with a value of 181.0, which is outside the valid range for longitude, and a valid latitude array. The test asserts that the validation method returns False for these coordinate arrays.

        Parameters:
            None

        Returns:
            None
        """
        lon = np.array([0.0, 45.0, 181.0])
        lat = np.array([10.0, 20.0, 30.0])
        assert DataValidator.validate_coordinates(lon, lat) is False

    def test_invalid_lon_exceeds_lower(self: 'TestValidateCoordinates') -> None:
        """
        This test checks that the validation method correctly identifies longitude arrays containing values that exceed the lower limit of -180 degrees as invalid. It uses a longitude array with a value of -181.0, which is outside the valid range for longitude, and a valid latitude array. The presence of a value less than -180 in the longitude array should cause the validation method to return False, indicating that the coordinates are not valid. 

        Parameters:
            None

        Returns:
            None
        """
        lon = np.array([-181.0, 45.0, 90.0])
        lat = np.array([10.0, 20.0, 30.0])
        assert DataValidator.validate_coordinates(lon, lat) is False

    def test_invalid_lat_exceeds_upper(self: 'TestValidateCoordinates') -> None:
        """
        This test verifies that the validation method correctly identifies latitude arrays containing values that exceed the upper limit of 90 degrees as invalid. It uses a latitude array with a value of 91.0, which is outside the valid range for latitude, and a valid longitude array. The presence of a value greater than 90 in the latitude array should cause the validation method to return False, indicating that the coordinates are not valid.

        Parameters:
            None

        Returns:
            None
        """
        lon = np.array([0.0, 45.0, 90.0])
        lat = np.array([10.0, 20.0, 91.0])
        assert DataValidator.validate_coordinates(lon, lat) is False

    def test_invalid_lat_exceeds_lower(self: 'TestValidateCoordinates') -> None:
        """
        This test checks that the validation method correctly identifies latitude arrays containing values that exceed the lower limit of -90 degrees as invalid. It uses a latitude array with a value of -91.0, which is outside the valid range for latitude, and a valid longitude array. The presence of a value less than -90 in the latitude array should cause the validation method to return False, indicating that the coordinates are not valid.

        Parameters:
            None

        Returns:
            None
        """
        lon = np.array([0.0, 45.0, 90.0])
        lat = np.array([-91.0, 20.0, 30.0])
        assert DataValidator.validate_coordinates(lon, lat) is False


class TestValidateDataArray:
    """ Tests for DataValidator.validate_data_array. """

    def test_valid_normal_data(self: 'TestValidateDataArray') -> None:
        """
        This test verifies that the validation method correctly identifies a normal data array with finite values as valid. It uses a data array with values ranging from 1.0 to 5.0. The test asserts that the validation method returns True for this data array and that there are no issues reported.

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = DataValidator.validate_data_array(data)
        assert result["valid"] is True
        assert result["issues"] == []

    def test_stats_keys_present(self: 'TestValidateDataArray') -> None:
        """
        This test verifies that the validation method correctly includes all expected keys in the stats dictionary. It uses a data array with values ranging from 0.0 to 100.0 and asserts that the keys "total_points", "finite_points", "finite_percentage", "min", "max", "mean", "std", and "median" are present in the stats dictionary.

        Parameters:
            None

        Returns:
            None
        """
        data = np.linspace(0.0, 100.0, 50)
        result = DataValidator.validate_data_array(data)
        assert "valid" in result
        assert "issues" in result
        assert "stats" in result
        stats = result["stats"]
        for key in ("total_points", "finite_points", "finite_percentage", "min", "max", "mean", "std", "median"):
            assert key in stats, f"Missing stats key: {key}"

    def test_stats_values_correct(self: 'TestValidateDataArray') -> None:
        """
        This test checks that the statistical values calculated by the validation method are correct for a known data array. It uses a data array with values from 1.0 to 5.0 and asserts that the total points, finite points, finite percentage, minimum, maximum, mean, standard deviation, and median are all calculated correctly according to the expected values.

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = DataValidator.validate_data_array(data)
        stats = result["stats"]
        assert stats["total_points"] == 5
        assert stats["finite_points"] == 5
        assert stats["finite_percentage"] == pytest.approx(100.0)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(5.0)
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["std"] == pytest.approx(float(np.std(data)))
        assert stats["median"] == pytest.approx(3.0)

    def test_all_nan_data(self: 'TestValidateDataArray') -> None:
        """
        This test verifies that the validation method correctly identifies a data array with all NaN values as invalid. It uses a data array with 10 NaN values and asserts that the validation method returns False and reports the issue "No finite values found".

        Parameters:
            None

        Returns:
            None
        """
        data = np.full(10, np.nan)
        result = DataValidator.validate_data_array(data)
        assert result["valid"] is False
        assert "No finite values found" in result["issues"]

    def test_all_inf_data(self: 'TestValidateDataArray') -> None:
        """
        This test verifies that the validation method correctly identifies a data array with all infinite values as invalid. It uses a data array with values [inf, -inf, inf] and asserts that the validation method returns False and reports the issue "No finite values found".

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([np.inf, -np.inf, np.inf])
        result = DataValidator.validate_data_array(data)
        assert result["valid"] is False
        assert "No finite values found" in result["issues"]

    def test_mixed_nan_finite(self: 'TestValidateDataArray') -> None:
        """
        This test verifies that the validation method correctly handles a data array with a mix of NaN and finite values. It uses a data array with 5 NaN values followed by 5 finite values and asserts that the validation method returns True, correctly counts the finite points, and calculates the finite percentage.

        Parameters:
            None

        Returns:
            None
        """
        finite_part = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data = np.concatenate([np.full(5, np.nan), finite_part])
        result = DataValidator.validate_data_array(data)
        assert result["valid"] is True
        assert result["stats"]["finite_points"] == 5
        assert result["stats"]["finite_percentage"] == pytest.approx(50.0)

    def test_min_val_violated(self: 'TestValidateDataArray') -> None:
        """
        This test verifies that the validation method correctly identifies a data array with values below the specified minimum value as invalid. It uses a data array with values [-5.0, 1.0, 2.0, 3.0] and a minimum value of 0.0, and asserts that the validation method returns False and reports the issue "Minimum value violated".

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([-5.0, 1.0, 2.0, 3.0])
        result = DataValidator.validate_data_array(data, min_val=0.0)
        assert result["valid"] is False
        assert any("Minimum value" in issue for issue in result["issues"])

    def test_max_val_violated(self: 'TestValidateDataArray') -> None:
        """
        This test verifies that the validation method correctly identifies a data array with values above the specified maximum value as invalid. It uses a data array with values [1.0, 50.0, 200.0] and a maximum value of 100.0, and asserts that the validation method returns False and reports the issue "Maximum value violated".

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([1.0, 50.0, 200.0])
        result = DataValidator.validate_data_array(data, max_val=100.0)
        assert result["valid"] is False
        assert any("Maximum value" in issue for issue in result["issues"])

    def test_thresholds_not_violated(self: 'TestValidateDataArray') -> None:
        """
        This test verifies that the validation method correctly identifies a data array with values within the specified minimum and maximum values as valid. It uses a data array with values from 10.0 to 90.0 and asserts that the validation method returns True and reports no issues.

        Parameters:
            None

        Returns:
            None
        """
        data = np.linspace(10.0, 90.0, 20)
        result = DataValidator.validate_data_array(data, min_val=-10.0, max_val=200.0)
        assert result["valid"] is True
        assert result["issues"] == []

    def test_no_thresholds_no_threshold_issues(self: 'TestValidateDataArray') -> None:
        """
        This test verifies that the validation method correctly identifies a data array with no specified minimum or maximum values as having no threshold-related issues. It uses a data array with values [1.0, 2.0, 3.0, 4.0, 5.0] and asserts that the validation method returns True and reports no issues related to thresholds.

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = DataValidator.validate_data_array(data)
        assert not any("below expected" in issue for issue in result["issues"])
        assert not any("above expected" in issue for issue in result["issues"])

    def test_constant_data(self: 'TestValidateDataArray') -> None:
        """
        This test verifies that the validation method correctly identifies a data array with all identical values as invalid. It uses a data array with 10 identical values (1.0) and asserts that the validation method returns False and reports the issues "All values are identical" and "Zero standard deviation".

        Parameters:
            None

        Returns:
            None
        """
        data = np.ones(10)
        result = DataValidator.validate_data_array(data)
        assert result["valid"] is False
        assert "All values are identical" in result["issues"]
        assert "Zero standard deviation" in result["issues"]

    def test_2d_array_input(self: 'TestValidateDataArray') -> None:
        """
        This test verifies that the validation method correctly handles a 2D data array. It uses a 5x5 data array with values from 1.0 to 25.0 and asserts that the validation method returns True and reports the correct total number of points.

        Parameters:
            None

        Returns:
            None
        """
        data = np.linspace(1.0, 25.0, 25).reshape(5, 5)
        result = DataValidator.validate_data_array(data)
        assert result["valid"] is True
        assert result["stats"]["total_points"] == 25

    def test_single_element_finite(self: 'TestValidateDataArray') -> None:
        """
        This test verifies that the validation method correctly identifies a data array with a single finite value as invalid. It uses a data array with a single value (42.0) and asserts that the validation method returns False and reports the issue "All values are identical".

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([42.0])
        result = DataValidator.validate_data_array(data)
        assert result["valid"] is False
        assert "All values are identical" in result["issues"]

    def test_valid_is_false_when_any_issue(self: 'TestValidateDataArray') -> None:
        """
        This test verifies that the validation method correctly identifies a data array as invalid when any issue is present. It uses a data array with values [-10.0, 1.0, 2.0] and a minimum value of 0.0, which should trigger a "Minimum value violated" issue. The test asserts that the validation method returns False for this data array.

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([-10.0, 1.0, 2.0])
        result = DataValidator.validate_data_array(data, min_val=0.0)
        assert result["valid"] is False

    def test_valid_is_true_no_issues(self: 'TestValidateDataArray') -> None:
        """
        This test verifies that the validation method correctly identifies a data array as valid when no issues are present. It uses a data array with values from 5.0 to 50.0 and asserts that the validation method returns True and reports no issues.

        Parameters:
            None

        Returns:
            None
        """
        data = np.linspace(5.0, 50.0, 30)
        result = DataValidator.validate_data_array(data, min_val=0.0, max_val=100.0)
        assert result["valid"] is True
        assert result["issues"] == []


class TestDataValidatorIntegration:
    """ Integration tests using real MPAS coordinate fixtures from conftest. """

    def test_real_mpas_coordinates(self: 'TestDataValidatorIntegration', 
                                   mpas_coordinates: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        This test verifies that the validation method correctly handles real MPAS coordinates. It uses the provided MPAS coordinates fixture and asserts that the validation method returns True.

        Parameters:
            mpas_coordinates (Tuple[np.ndarray, np.ndarray]): A tuple containing longitude and latitude arrays.

        Returns:
            None
        """
        lon, lat = mpas_coordinates
        assert DataValidator.validate_coordinates(lon, lat) is True

    def test_real_mpas_precip_data(self: 'TestDataValidatorIntegration', 
                                   mpas_precip_data: np.ndarray) -> None:
        """
        This test verifies that the validation method correctly handles real MPAS precipitation data. It uses the provided MPAS precipitation data fixture and asserts that the validation method returns a dictionary with the expected keys.

        Parameters:
            mpas_precip_data (np.ndarray): An array containing MPAS precipitation data.

        Returns:
            None
        """
        result = DataValidator.validate_data_array(mpas_precip_data)
        assert isinstance(result, dict)
        assert "valid" in result
        assert "stats" in result

    def test_validator_methods_are_static(self: 'TestDataValidatorIntegration') -> None:
        """
        This test verifies that the validator methods are static. It checks that the methods can be called without an instance of the class and that they return the expected types.

        Parameters:
            None

        Returns:
            None
        """
        assert callable(DataValidator.validate_coordinates)
        assert callable(DataValidator.validate_data_array)
        lon = np.array([0.0, 45.0])
        lat = np.array([10.0, 30.0])
        assert isinstance(DataValidator.validate_coordinates(lon, lat), bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

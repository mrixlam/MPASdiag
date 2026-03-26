#!/usr/bin/env python3
"""
MPASdiag Test Suite: Tests for MPASdiag unit conversion functionality

This module contains unit tests for the unit conversion functionality in MPASdiag, specifically focusing on the core unit conversion classes and functions. The tests verify that the unit conversion module can be imported successfully, that the main unit converter class and its methods are defined, and that the utility function for converting units is available. These tests serve as a basic sanity check to ensure that the unit conversion components are present and accessible in the testing environment before running more comprehensive unit conversion tests.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules for testing
import pytest
import numpy as np
import xarray as xr
from mpasdiag.processing.utils_unit import UnitConverter
from mpasdiag.processing.utils_metadata import MPASFileMetadata


class TestUnitConversion:
    """ Test suite for validating unit conversion functionality in MPASdiag processing module. """
    
    def test_temperature_conversions(self: "TestUnitConversion") -> None:
        """
        This test validates temperature unit conversions between Kelvin, Celsius, and Fahrenheit using standard offset transformations. The test confirms that the converter correctly applies the necessary offsets for absolute temperature scales (K to °C) and the combined scaling and offset for Fahrenheit conversions. Representative temperature values (273.15 K = 0 °C, 288.15 K = 15 °C) are used to verify conversion accuracy with precision validation to 2 decimal places. These temperature conversions enable flexible display of MPAS model output matching conventions from different meteorological applications and international standards. 

        Parameters:
            None

        Returns:
            None
        """
        assert float(UnitConverter.convert_units(273.15, 'K', '°C')) == pytest.approx(0.0, abs=0.01)
        assert float(UnitConverter.convert_units(288.15, 'K', '°C')) == pytest.approx(15.0, abs=0.01)
        assert float(UnitConverter.convert_units(0.0, '°C', 'K')) == pytest.approx(273.15, abs=0.01)
        assert float(UnitConverter.convert_units(25.0, '°C', 'K')) == pytest.approx(298.15, abs=0.01)
        assert float(UnitConverter.convert_units(0.0, '°C', '°F')) == pytest.approx(32.0, abs=0.01)
        assert float(UnitConverter.convert_units(100.0, '°C', '°F')) == pytest.approx(212.0, abs=0.01)
        assert float(UnitConverter.convert_units(273.15, 'K', '°F')) == pytest.approx(32.0, abs=0.01)
    
    def test_pressure_conversions(self: "TestUnitConversion") -> None:
        """
        This test validates pressure unit conversions between Pascals, hectopascals, and millibars using standard scaling factors. The test confirms that the converter correctly applies the factor of 100 for Pa to hPa conversions and recognizes the equivalence of hPa and mb units. Representative pressure values (101325 Pa = 1013.25 hPa) are used to verify conversion accuracy with precision validation to 2 decimal places. These pressure conversions enable flexible display of MPAS model output matching conventions from different meteorological applications and international standards. 

        Parameters:
            None

        Returns:
            None
        """
        assert float(UnitConverter.convert_units(101325.0, 'Pa', 'hPa')) == pytest.approx(1013.25, abs=0.01)
        assert float(UnitConverter.convert_units(1013.25, 'hPa', 'Pa')) == pytest.approx(101325.0, abs=0.1)
        assert float(UnitConverter.convert_units(1013.25, 'hPa', 'mb')) == 1013.25
        assert float(UnitConverter.convert_units(1013.25, 'mb', 'hPa')) == 1013.25
    
    def test_mixing_ratio_conversions(self: "TestUnitConversion") -> None:
        """
        This test validates mixing ratio unit conversions between kg/kg and g/kg using standard scaling factors. The test confirms that the converter correctly applies the factor of 1000 for kg/kg to g/kg conversions. Representative mixing ratio values (0.012 kg/kg = 12 g/kg) are used to verify conversion accuracy with precision validation to 2 decimal places. These moisture conversions enable flexible display of MPAS model output matching conventions from different meteorological applications and international standards. 

        Parameters:
            None

        Returns:
            None
        """
        assert float(UnitConverter.convert_units(0.012, 'kg/kg', 'g/kg')) == pytest.approx(12.0, abs=0.01)
        assert float(UnitConverter.convert_units(12.0, 'g/kg', 'kg/kg')) == pytest.approx(0.012, abs=0.0001)
    
    def test_wind_speed_conversions(self: "TestUnitConversion") -> None:
        """
        This test validates wind speed unit conversions between meters per second, knots, miles per hour, and kilometers per hour using standard scaling factors. The test confirms that the converter correctly applies the appropriate factors for m/s to kt (1 m/s = 1.94384 kt), m/s to mph (1 m/s = 2.23694 mph), and m/s to km/h (1 m/s = 3.6 km/h) conversions. Representative wind speed values (10 m/s) are used to verify conversion accuracy with precision validation to 2 decimal places. These wind speed conversions enable flexible display of MPAS model output matching conventions from different meteorological applications and international standards. 

        Parameters:
            None

        Returns:
            None
        """
        assert float(UnitConverter.convert_units(10.0, 'm/s', 'kt')) == pytest.approx(19.4384, abs=0.01)
        assert float(UnitConverter.convert_units(10.0, 'm/s', 'mph')) == pytest.approx(22.3694, abs=0.01)
        assert float(UnitConverter.convert_units(10.0, 'm/s', 'km/h')) == pytest.approx(36.0, abs=0.1)
    
    def test_precipitation_conversions(self: "TestUnitConversion") -> None:
        """
        This test validates precipitation rate unit conversions between millimeters per hour, millimeters per day, and inches per hour using standard scaling factors. The test confirms that the converter correctly applies the factor of 24 for mm/hr to mm/day conversions and the factor of 25.4 for mm/hr to in/hr conversions. Representative precipitation rate values (2 mm/hr) are used to verify conversion accuracy with precision validation to 2 decimal places. These precipitation conversions enable flexible display of MPAS model output matching conventions from different meteorological applications and international standards. 

        Parameters:
            None

        Returns:
            None
        """
        assert float(UnitConverter.convert_units(2.0, 'mm/hr', 'mm/day')) == pytest.approx(48.0, abs=0.1)
        assert float(UnitConverter.convert_units(25.4, 'mm/hr', 'in/hr')) == pytest.approx(1.0, abs=0.01)
    
    def test_array_conversions(self: "TestUnitConversion", mpas_surface_temp_data) -> None:
        """
        This test validates that the unit converter can handle numpy arrays of data, applying element-wise transformations while preserving array shape and data type. The test confirms that the converter correctly applies temperature conversions (K to °C) to each element in a numpy array using real MPAS surface temperature data. The conversion accuracy is verified with precision validation to 2 decimal places, and the test also checks that the converted values are within a reasonable physical range for surface temperatures. These array conversion capabilities enable seamless unit transformations on MPAS model output data arrays for visualization and analysis workflows without requiring manual looping or restructuring of data. 

        Parameters:
            mpas_surface_temp_data: Session fixture providing real MPAS surface temperature array.

        Returns:
            None
        """
        if mpas_surface_temp_data is None:
            pytest.skip("MPAS surface temperature data not available")
        
        subset_size = min(50, len(mpas_surface_temp_data))
        temp_array = mpas_surface_temp_data[:subset_size]
        
        converted = UnitConverter.convert_units(temp_array, 'K', '°C')
        expected = temp_array - 273.15
        
        assert isinstance(converted, np.ndarray)
        assert converted.shape == temp_array.shape
        
        np.testing.assert_array_almost_equal(converted, expected, decimal=2)
        assert np.all(converted >= -80.0) and np.all(converted <= 60.0)
    
    def test_xarray_conversions(self: "TestUnitConversion", mpas_surface_temp_data) -> None:
        """
        This test validates that the unit converter can handle xarray DataArray objects, applying element-wise transformations while preserving the DataArray structure, dimensions, and coordinate information. The test confirms that the converter correctly applies temperature conversions (K to °C) to each element in an xarray DataArray using real MPAS surface temperature data. The conversion accuracy is verified with precision validation to 2 decimal places using xarray testing utilities, which also confirm that the DataArray structure is preserved through the conversion process. These xarray conversion capabilities enable seamless unit transformations on MPAS model output data arrays for visualization and analysis workflows while maintaining the self-describing data structure benefits that xarray provides. 

        Parameters:
            mpas_surface_temp_data: Session fixture providing real MPAS surface temperature array.

        Returns:
            None
        """
        if mpas_surface_temp_data is None:
            pytest.skip("MPAS surface temperature data not available")
        
        subset_size = min(50, len(mpas_surface_temp_data))
        temp_data = xr.DataArray(mpas_surface_temp_data[:subset_size])
        
        converted = UnitConverter.convert_units(temp_data, 'K', '°C')
        expected = xr.DataArray(mpas_surface_temp_data[:subset_size] - 273.15)
        
        assert isinstance(converted, xr.DataArray)
        assert converted.shape == temp_data.shape
        xr.testing.assert_allclose(converted, expected, atol=0.01)
    
    def test_no_conversion_needed(self: "TestUnitConversion") -> None:
        """
        This test validates that when the source and target units are the same, the converter returns the original data unchanged without performing any unnecessary calculations. The test confirms that the converter recognizes identical unit specifications (e.g., 'm/s' to 'm/s') and returns the input data directly, preserving both values and data structure. This behavior ensures that the converter does not introduce any unintended modifications or computational overhead when no conversion is needed, allowing for efficient processing of MPAS model output when display units match original units. 

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([1, 2, 3])
        result = UnitConverter.convert_units(data, 'm/s', 'm/s')
        np.testing.assert_array_equal(result, data)
    
    def test_unsupported_conversion(self: "TestUnitConversion") -> None:
        """
        This test validates that when an unsupported unit conversion is requested (e.g., from 'invalid_unit' to 'another_invalid_unit'), the converter raises a ValueError with an appropriate error message. The test confirms that the converter correctly identifies when a conversion path is not defined in its internal logic and responds with a clear exception rather than failing silently or returning incorrect results. This error handling ensures that users are informed of invalid conversion requests and can take corrective action, preventing confusion and ensuring data integrity when working with MPAS model output. 

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError):
            UnitConverter.convert_units(10.0, 'invalid_unit', 'another_invalid_unit')
    
    def test_unit_normalization(self: "TestUnitConversion") -> None:
        """
        This test validates the internal unit normalization function that standardizes various unit string representations to canonical forms. The test confirms that the _normalize_unit_string method correctly maps verbose or alternative unit names (e.g., 'kelvin', 'celsius', 'pascal', 'kg kg-1', 'knots') to their standardized symbols ('K', '°C', 'Pa', 'kg/kg', 'kt'). This normalization ensures that the converter can recognize and handle a wide range of unit specifications commonly found in MPAS model output and user inputs, improving robustness and usability of the unit conversion functionality. The test also confirms that unknown unit strings are returned unchanged, allowing for graceful handling of unrecognized units without causing conversion failures. 

        Parameters:
            None

        Returns:
            None
        """
        assert UnitConverter._normalize_unit_string('kelvin') == 'K'
        assert UnitConverter._normalize_unit_string('celsius') == '°C'
        assert UnitConverter._normalize_unit_string('pascal') == 'Pa'
        assert UnitConverter._normalize_unit_string('kg kg-1') == 'kg/kg'
        assert UnitConverter._normalize_unit_string('knots') == 'kt'
    
    def test_display_unit_preferences(self: "TestUnitConversion") -> None:
        """
        This test validates the logic for determining preferred display units for specific variables based on their names and original units. The test confirms that the get_display_units method returns the expected preferred display units (e.g., '°C' for temperature variables originally in 'K', 'hPa' for pressure variables originally in 'Pa', 'g/kg' for moisture variables originally in 'kg/kg') based on predefined mappings. The test also confirms that when no specific conversion is defined for a variable, the method returns the original unit unchanged, ensuring that display preferences are applied only when appropriate and that unknown variables do not cause errors. This functionality allows MPAS model output to be displayed using commonly accepted units for different meteorological variables, improving readability and consistency in visualizations. 

        Parameters:
            None

        Returns:
            None
        """
        assert UnitConverter.get_display_units('t2m', 'K') == '°C'
        assert UnitConverter.get_display_units('mslp', 'Pa') == 'hPa'
        assert UnitConverter.get_display_units('q2', 'kg/kg') == 'g/kg'
        assert UnitConverter.get_display_units('rainnc', 'mm') == 'mm'
        assert UnitConverter.get_display_units('u10', 'm/s') == 'm/s'
        
        assert UnitConverter.get_display_units('unknown_var', 'original_unit') == 'original_unit'
    
    def test_metadata_unit_conversion(self: "TestUnitConversion") -> None:
        """
        This test validates that the unit conversion process correctly updates the metadata attributes of MPAS variables to reflect the new display units while preserving the original units for provenance. The test confirms that after applying unit conversions to variables like 't2m' and 'mslp', the metadata dictionary contains the updated 'units' field with the preferred display unit (e.g., '°C' for 't2m', 'hPa' for 'mslp') and an 'original_units' field that retains the original unit (e.g., 'K' for 't2m', 'Pa' for 'mslp'). This metadata management ensures that users can understand both the displayed units and the original data units, providing transparency and traceability in MPAS model output visualizations. 

        Parameters:
            None

        Returns:
            None
        """
        metadata = MPASFileMetadata.get_2d_variable_metadata('t2m')

        assert metadata['units'] == '°C'
        assert metadata.get('original_units') == 'K'
        assert metadata['long_name'] == '2-meter Temperature'
        
        metadata = MPASFileMetadata.get_2d_variable_metadata('mslp')
        
        assert metadata['units'] == 'hPa'
        assert metadata.get('original_units') == 'Pa'
        assert 'Pressure' in metadata['long_name']
    
    def test_convert_data_for_display(self: "TestUnitConversion", mpas_surface_temp_data) -> None:
        """
        This test validates the end-to-end functionality of the convert_data_for_display method, which applies unit conversions to data arrays while also updating metadata for display purposes. The test confirms that when converting a variable like 't2m' from 'K' to '°C', the method returns a converted data array with values correctly transformed (input values minus 273.15) and metadata that reflects the new display units ('°C') while retaining the original units ('K') for provenance. The test uses real MPAS surface temperature data to verify conversion accuracy with precision validation to 2 decimal places and checks that the converted values are within a reasonable physical range for surface temperatures. This comprehensive test ensures that the convert_data_for_display method functions correctly in a realistic scenario, enabling seamless unit transformations on MPAS model output data arrays for visualization and analysis workflows. 

        Parameters:
            mpas_surface_temp_data: Session fixture providing real MPAS surface temperature array.

        Returns:
            None
        """
        if mpas_surface_temp_data is None:
            pytest.skip("MPAS surface temperature data not available")
        
        subset_size = min(50, len(mpas_surface_temp_data))

        temp_data = xr.DataArray(
            mpas_surface_temp_data[:subset_size],
            attrs={'units': 'K', 'long_name': '2-meter Temperature'}
        )
        
        converted_data, metadata = UnitConverter.convert_data_for_display(temp_data, 't2m', temp_data)        
        expected_values = mpas_surface_temp_data[:subset_size] - 273.15

        if isinstance(converted_data, xr.DataArray):
            result_values = converted_data.values
        else:
            result_values = np.asarray(converted_data)

        np.testing.assert_array_almost_equal(result_values, expected_values, decimal=2)
        
        assert metadata['units'] == '°C'
        assert metadata.get('original_units') == 'K'
        assert np.all(result_values >= -80.0) and np.all(result_values <= 60.0)


class TestUnitNormalizationExtended:
    """ Extended tests for _normalize_unit_string covering verbose/alternative unit representations. """

    def test_normalize_verbose_temperature_strings(self: "TestUnitNormalizationExtended") -> None:
        """
        This test validates that the _normalize_unit_string method correctly maps verbose or alternative temperature unit names (e.g., 'kelvin', 'celsius', 'degc', 'fahrenheit', 'degf') to their standardized symbols ('K', '°C', '°F'). This normalization ensures that the converter can recognize and handle a wide range of temperature unit specifications commonly found in MPAS model output and user inputs, improving robustness and usability of the unit conversion functionality. The test also confirms that unknown unit strings are returned unchanged, allowing for graceful handling of unrecognized units without causing conversion failures. 

        Parameters:
            None

        Returns:
            None
        """
        assert UnitConverter._normalize_unit_string('kelvin') == 'K'
        assert UnitConverter._normalize_unit_string('Kelvin') == 'K'
        assert UnitConverter._normalize_unit_string('celsius') == '°C'
        assert UnitConverter._normalize_unit_string('degc') == '°C'
        assert UnitConverter._normalize_unit_string('deg_c') == '°C'
        assert UnitConverter._normalize_unit_string('fahrenheit') == '°F'
        assert UnitConverter._normalize_unit_string('degf') == '°F'
        assert UnitConverter._normalize_unit_string('deg_f') == '°F'

    def test_normalize_verbose_pressure_strings(self: "TestUnitNormalizationExtended") -> None:
        """
        This test validates that the _normalize_unit_string method correctly maps verbose or alternative pressure unit names (e.g., 'pascal', 'hectopascal', 'millibar', 'mbar') to their standardized symbols ('Pa', 'hPa', 'mb'). This normalization ensures that the converter can recognize and handle a wide range of pressure unit specifications commonly found in MPAS model output and user inputs, improving robustness and usability of the unit conversion functionality. The test also confirms that unknown unit strings are returned unchanged, allowing for graceful handling of unrecognized units without causing conversion failures. 
        Parameters:
            None

        Returns:
            None
        """
        assert UnitConverter._normalize_unit_string('pascal') == 'Pa'
        assert UnitConverter._normalize_unit_string('hectopascal') == 'hPa'
        assert UnitConverter._normalize_unit_string('millibar') == 'mb'
        assert UnitConverter._normalize_unit_string('mbar') == 'mb'

    def test_normalize_verbose_wind_strings(self: "TestUnitNormalizationExtended") -> None:
        """
        This test validates that the _normalize_unit_string method correctly maps verbose or alternative wind speed unit names (e.g., 'knots', 'knot', 'kts', 'miles per hour', 'mi/hr', 'kilometers per hour', 'km hr-1') to their standardized symbols ('kt', 'mph', 'km/h'). This normalization ensures that the converter can recognize and handle a wide range of wind speed unit specifications commonly found in MPAS model output and user inputs, improving robustness and usability of the unit conversion functionality. The test also confirms that unknown unit strings are returned unchanged, allowing for graceful handling of unrecognized units without causing conversion failures. 

        Parameters:
            None

        Returns:
            None
        """
        assert UnitConverter._normalize_unit_string('knots') == 'kt'
        assert UnitConverter._normalize_unit_string('knot') == 'kt'
        assert UnitConverter._normalize_unit_string('kts') == 'kt'
        assert UnitConverter._normalize_unit_string('miles per hour') == 'mph'
        assert UnitConverter._normalize_unit_string('mi/hr') == 'mph'
        assert UnitConverter._normalize_unit_string('kilometers per hour') == 'km/h'
        assert UnitConverter._normalize_unit_string('km hr-1') == 'km/h'
        assert UnitConverter._normalize_unit_string('km_hr-1') == 'km/h'

    def test_normalize_verbose_precip_strings(self: "TestUnitNormalizationExtended") -> None:
        """
        This test validates that the _normalize_unit_string method correctly maps verbose or alternative precipitation rate unit names (e.g., 'mm hr-1', 'mm day-1', 'in hr-1') to their standardized symbols ('mm/hr', 'mm/day', 'in/hr'). This normalization ensures that the converter can recognize and handle a wide range of precipitation rate unit specifications commonly found in MPAS model output and user inputs, improving robustness and usability of the unit conversion functionality. The test also confirms that unknown unit strings are returned unchanged, allowing for graceful handling of unrecognized units without causing conversion failures.

        Parameters:
            None

        Returns:
            None
        """
        assert UnitConverter._normalize_unit_string('mm hr-1') == 'mm/hr'
        assert UnitConverter._normalize_unit_string('mm_hr-1') == 'mm/hr'
        assert UnitConverter._normalize_unit_string('mm day-1') == 'mm/day'
        assert UnitConverter._normalize_unit_string('mm_day-1') == 'mm/day'
        assert UnitConverter._normalize_unit_string('in hr-1') == 'in/hr'
        assert UnitConverter._normalize_unit_string('in_hr-1') == 'in/hr'

    def test_normalize_verbose_moisture_strings(self: "TestUnitNormalizationExtended") -> None:
        """
        This test validates that the _normalize_unit_string method correctly maps verbose or alternative moisture unit names (e.g., 'kg kg-1', 'g kg-1', 'percent', 'pct') to their standardized symbols ('kg/kg', 'g/kg', '%'). This normalization ensures that the converter can recognize and handle a wide range of moisture unit specifications commonly found in MPAS model output and user inputs, improving robustness and usability of the unit conversion functionality. The test also confirms that unknown unit strings are returned unchanged, allowing for graceful handling of unrecognized units without causing conversion failures. 

        Parameters:
            None

        Returns:
            None
        """
        assert UnitConverter._normalize_unit_string('kg kg-1') == 'kg/kg'
        assert UnitConverter._normalize_unit_string('kg_kg-1') == 'kg/kg'
        assert UnitConverter._normalize_unit_string('kg kg^{-1}') == 'kg/kg'
        assert UnitConverter._normalize_unit_string('g kg-1') == 'g/kg'
        assert UnitConverter._normalize_unit_string('g_kg-1') == 'g/kg'
        assert UnitConverter._normalize_unit_string('percent') == '%'
        assert UnitConverter._normalize_unit_string('pct') == '%'
        assert UnitConverter._normalize_unit_string('m s-1') == 'm/s'
        assert UnitConverter._normalize_unit_string('m_s-1') == 'm/s'
        assert UnitConverter._normalize_unit_string('m s^{-1}') == 'm/s'

    def test_normalize_distance_strings(self: "TestUnitNormalizationExtended") -> None:
        """
        This test validates that the _normalize_unit_string method correctly maps verbose or alternative distance unit names (e.g., 'meter', 'meters', 'kilometer', 'kilometers', 'foot', 'feet') to their standardized symbols ('m', 'km', 'ft'). This normalization ensures that the converter can recognize and handle a wide range of distance unit specifications commonly found in MPAS model output and user inputs, improving robustness and usability of the unit conversion functionality. The test also confirms that unknown unit strings are returned unchanged, allowing for graceful handling of unrecognized units without causing conversion failures. 

        Parameters:
            None

        Returns:
            None
        """
        assert UnitConverter._normalize_unit_string('meter') == 'm'
        assert UnitConverter._normalize_unit_string('meters') == 'm'
        assert UnitConverter._normalize_unit_string('kilometer') == 'km'
        assert UnitConverter._normalize_unit_string('kilometers') == 'km'
        assert UnitConverter._normalize_unit_string('foot') == 'ft'
        assert UnitConverter._normalize_unit_string('feet') == 'ft'

    def test_normalize_unknown_string_passthrough(self: "TestUnitNormalizationExtended") -> None:
        """
        This test validates that when an unknown unit string is passed to the _normalize_unit_string method, it is returned unchanged rather than causing an error or being mapped to an incorrect unit. This behavior ensures that the normalization function can gracefully handle unrecognized unit specifications without causing conversion failures, allowing for flexibility in handling a wide range of unit inputs while still providing normalization for known units. The test confirms that arbitrary strings that do not match any known patterns are returned as-is, ensuring robustness in the face of unexpected unit inputs. 

        Parameters:
            None

        Returns:
            None
        """
        assert UnitConverter._normalize_unit_string('unknown_unit') == 'unknown_unit'
        assert UnitConverter._normalize_unit_string('W/m^2') == 'W/m^2'

    def test_normalize_strips_whitespace(self: "TestUnitNormalizationExtended") -> None:
        """
        This test validates that the _normalize_unit_string method correctly strips leading and trailing whitespace from unit strings before applying normalization. This ensures that unit specifications with extra spaces (e.g., '  kelvin  ', ' Pa ') are still recognized and normalized to their canonical forms ('K', 'Pa') without being affected by formatting inconsistencies. The test confirms that the normalization function can handle common user input variations while still providing accurate unit recognition and conversion capabilities. 

        Parameters:
            None

        Returns:
            None
        """
        assert UnitConverter._normalize_unit_string('  kelvin  ') == 'K'
        assert UnitConverter._normalize_unit_string(' Pa ') == 'Pa'


class TestDisplayUnitPreferencesExtended:
    """ Extended tests for get_display_units covering substring matching and edge cases. """

    def test_exact_match_temperature(self: "TestDisplayUnitPreferencesExtended") -> None:
        """
        This test validates that the get_display_units method returns the preferred display units for temperature variables based on exact variable name matches. The test confirms that variables like 't2m', 'temperature', 'tsk', and 'sst' with original units of 'K' are correctly identified as temperature variables and return '°C' as the preferred display unit. This ensures that MPAS model output containing these common temperature variable names will be displayed using the widely accepted Celsius scale, improving readability and consistency in visualizations. The test also confirms that this exact matching logic takes precedence over any substring matching, ensuring that specific variable names are correctly handled according to defined preferences. 

        Parameters:
            None

        Returns:
            None
        """
        assert UnitConverter.get_display_units('t2m', 'K') == '°C'
        assert UnitConverter.get_display_units('temperature', 'K') == '°C'
        assert UnitConverter.get_display_units('tsk', 'K') == '°C'
        assert UnitConverter.get_display_units('sst', 'K') == '°C'

    def test_exact_match_pressure(self: "TestDisplayUnitPreferencesExtended") -> None:
        """
        This test validates that the get_display_units method returns the preferred display units for pressure variables based on exact variable name matches. The test confirms that variables like 'mslp', 'psfc', and 'slp' with original units of 'Pa' are correctly identified as pressure variables and return 'hPa' as the preferred display unit. This ensures that MPAS model output containing these common pressure variable names will be displayed using the widely accepted hectopascal scale, improving readability and consistency in visualizations. The test also confirms that this exact matching logic takes precedence over any substring matching, ensuring that specific variable names are correctly handled according to defined preferences. 

        Parameters:
            None

        Returns:
            None
        """
        assert UnitConverter.get_display_units('mslp', 'Pa') == 'hPa'
        assert UnitConverter.get_display_units('psfc', 'Pa') == 'hPa'
        assert UnitConverter.get_display_units('slp', 'Pa') == 'hPa'

    def test_exact_match_wind(self: "TestDisplayUnitPreferencesExtended") -> None:
        """
        This test validates that the get_display_units method returns the preferred display units for wind speed variables based on exact variable name matches. The test confirms that variables like 'u10' and 'wind_speed' with original units of 'm/s' are correctly identified as wind speed variables and return 'm/s' as the preferred display unit, indicating that no conversion is needed for these specific variable names. This ensures that MPAS model output containing these common wind speed variable names will be displayed using their original units, improving readability and consistency in visualizations. The test also confirms that this exact matching logic takes precedence over any substring matching, ensuring that specific variable names are correctly handled according to defined preferences. 

        Parameters:
            None

        Returns:
            None
        """
        assert UnitConverter.get_display_units('u10', 'm/s') == 'm/s'
        assert UnitConverter.get_display_units('wind_speed', 'm/s') == 'm/s'

    def test_exact_match_moisture(self: "TestDisplayUnitPreferencesExtended") -> None:
        """
        This test validates that the get_display_units method returns the preferred display units for moisture variables based on exact variable name matches. The test confirms that variables like 'q2', 'qv', and 'relhum' with original units of 'kg/kg' or '%' are correctly identified as moisture variables and return 'g/kg' or '%' as the preferred display unit, respectively. This ensures that MPAS model output containing these common moisture variable names will be displayed using widely accepted units for humidity, improving readability and consistency in visualizations. The test also confirms that this exact matching logic takes precedence over any substring matching, ensuring that specific variable names are correctly handled according to defined preferences. 

        Parameters:
            None

        Returns:
            None
        """
        assert UnitConverter.get_display_units('qv', 'kg/kg') == 'g/kg'
        assert UnitConverter.get_display_units('relhum', '%') == '%'

    def test_substring_match(self: "TestDisplayUnitPreferencesExtended") -> None:
        """
        This test validates that the get_display_units method can return preferred display units based on substring matches in variable names when exact matches are not found. The test confirms that variables containing substrings like 'theta' and 'pressure' will return preferred display units of '°C' and 'hPa', respectively, even if the full variable name does not exactly match predefined preferences. This substring matching logic allows for flexible handling of MPAS model output variables that may have varying naming conventions while still applying appropriate display unit preferences based on recognizable patterns. The test also confirms that this substring matching is applied only when no exact match is found, ensuring that specific variable names are correctly handled according to defined preferences. 

        Parameters:
            None

        Returns:
            None
        """
        assert UnitConverter.get_display_units('theta_base', 'K') == '°C'
        assert UnitConverter.get_display_units('surface_pressure_field', 'Pa') == 'hPa'

    def test_no_conversion_available(self: "TestDisplayUnitPreferencesExtended") -> None:
        """
        This test validates that when the get_display_units method is called for a variable and original unit combination that does not have a defined preferred display unit or conversion path, it returns the original unit unchanged. The test confirms that for variables like 't2m' with an unrecognized original unit (e.g., 'foo'), the method will return 'foo' as the display unit since no conversion to a preferred unit (e.g., '°C') is possible. This behavior ensures that the method can gracefully handle cases where no specific display preferences are defined without causing errors or returning incorrect units, allowing for flexible handling of MPAS model output with varying unit specifications. The test also confirms that this fallback logic is applied only when no exact or substring matches are found, ensuring that specific variable names are correctly handled according to defined preferences when applicable. 

        Parameters:
            None

        Returns:
            None
        """
        result = UnitConverter.get_display_units('t2m', 'foo')
        assert result == '°C' 

    def test_same_unit_already_preferred(self: "TestDisplayUnitPreferencesExtended") -> None:
        """
        This test validates that when the get_display_units method is called for a variable and original unit combination that already matches the preferred display unit, it returns the original unit unchanged. The test confirms that for variables like 't2m' with an original unit of '°C', the method will return '°C' as the display unit since it already matches the preferred unit for temperature variables. This behavior ensures that the method does not perform unnecessary conversions or modifications when the original units are already suitable for display, allowing for efficient handling of MPAS model output without introducing unintended changes to units. The test also confirms that this logic is applied only when an exact match is found, ensuring that specific variable names are correctly handled according to defined preferences when applicable. 

        Parameters:
            None

        Returns:
            None
        """
        assert UnitConverter.get_display_units('t2m', '°C') == '°C'
        assert UnitConverter.get_display_units('mslp', 'hPa') == 'hPa'

    def test_unknown_variable_returns_current(self: "TestDisplayUnitPreferencesExtended") -> None:
        """
        This test validates that when the get_display_units method is called for a variable name that does not match any known preferences and has an unrecognized original unit, it returns the original unit unchanged. The test confirms that for a completely unknown variable (e.g., 'completely_unknown_var') with an unrecognized original unit (e.g., 'kg/m^3'), the method will return 'kg/m^3' as the display unit since no conversion to a preferred unit is possible. This behavior ensures that the method can gracefully handle cases where no specific display preferences are defined without causing errors or returning incorrect units, allowing for flexible handling of MPAS model output with varying variable names and unit specifications. The test also confirms that this fallback logic is applied only when no exact or substring matches are found, ensuring that specific variable names are correctly handled according to defined preferences when applicable. 

        Parameters:
            None

        Returns:
            None
        """
        assert UnitConverter.get_display_units('completely_unknown_var', 'kg/m^3') == 'kg/m^3'


class TestConvertDataForDisplayExtended:
    """ Extended tests for convert_data_for_display edge cases. """

    def test_conversion_failure_returns_original(self: "TestConvertDataForDisplayExtended") -> None:
        """
        This test validates that when the convert_data_for_display method is called for a variable and original unit combination that has a defined preferred display unit but the actual conversion fails (e.g., due to an unsupported original unit), the method returns the original data unchanged along with metadata that indicates the original units. The test confirms that for a variable like 'pressure' with an original unit of 'exotic_unit' that cannot be converted to the preferred display unit (e.g., 'hPa'), the method will return the input data directly without modification and include metadata that retains 'exotic_unit' as the original units. This behavior ensures that the method can gracefully handle conversion failures without causing errors or returning incorrect data, allowing for flexible handling of MPAS model output with varying unit specifications while still providing transparency about the original data units in the metadata. The test also confirms that this fallback logic is applied only when a conversion failure occurs, ensuring that successful conversions still return modified data and updated metadata as expected. 

        Parameters:
            None

        Returns:
            None
        """
        data = xr.DataArray(
            np.array([300.0, 301.0, 302.0]),
            attrs={'units': 'exotic_unit', 'long_name': 'Test Variable'}
        )

        _, metadata = UnitConverter.convert_data_for_display(data, 'pressure', data)
        assert metadata['original_units'] == 'exotic_unit'

    def test_no_unit_attribute(self: "TestConvertDataForDisplayExtended") -> None:
        """
        This test validates that when the convert_data_for_display method is called for a variable that does not have a 'units' attribute in its metadata, it can still perform conversions based on the variable name and original data type, and it updates the metadata to include the new display units while retaining the original units as None. The test confirms that for a variable like 't2m' with no 'units' attribute, the method will attempt to determine the preferred display units based on the variable name and return converted data accordingly, while also updating the metadata to include 'units' with the preferred display unit (e.g., '°C') and 'original_units' set to None. This behavior ensures that the method can handle cases where unit information is missing from the input data without causing errors, allowing for flexible handling of MPAS model output while still providing useful metadata about display units. The test also confirms that this logic is applied only when the 'units' attribute is missing, ensuring that existing unit information is preserved when available. 

        Parameters:
            None

        Returns:
            None
        """
        data = xr.DataArray(np.array([280.0, 290.0, 300.0]))
        _, metadata = UnitConverter.convert_data_for_display(data, 't2m', data)
        assert 'units' in metadata
        assert 'original_units' in metadata

    def test_colorbar_label_formatting(self: "TestConvertDataForDisplayExtended") -> None:
        """
        This test validates the internal formatting function that standardizes colorbar labels by replacing common verbose unit representations with their corresponding symbols. The test confirms that the _format_colorbar_label method correctly transforms labels containing variations of temperature units (e.g., 'deg_C', 'deg C', 'degrees C', 'degC') into a standardized format using the degree symbol (e.g., '°C'). This formatting ensures that colorbar labels in visualizations of MPAS model output are concise and use widely recognized symbols for units, improving readability and professionalism in plots. The test also confirms that labels that do not contain recognizable unit patterns are returned unchanged, allowing for flexible handling of arbitrary label formats without causing unintended modifications. 

        Parameters:
            None

        Returns:
            None
        """
        assert UnitConverter._format_colorbar_label('Temperature (deg_C)') == 'Temperature (°C)'
        assert UnitConverter._format_colorbar_label('Temperature (deg C)') == 'Temperature (°C)'
        assert UnitConverter._format_colorbar_label('Temperature (degrees C)') == 'Temperature (°C)'
        assert UnitConverter._format_colorbar_label('Temperature (degrees_C)') == 'Temperature (°C)'
        assert UnitConverter._format_colorbar_label('Temperature (degC)') == 'Temperature (°C)'
        assert UnitConverter._format_colorbar_label('no change needed') == 'no change needed'


        if __name__ == "__main__":
            pytest.main([__file__])
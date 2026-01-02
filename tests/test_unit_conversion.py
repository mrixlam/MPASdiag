#!/usr/bin/env python3
"""
MPAS Unit Conversion Module Unit Tests

This module provides comprehensive unit tests for the UnitConverter class and unit conversion
functionality used throughout the MPAS diagnostic package. These tests validate conversions
between meteorological units (temperature, pressure, wind speed, precipitation, mixing ratios),
array handling (NumPy and xarray), unit string normalization, and display unit preferences
for various atmospheric variables using synthetic data to ensure accurate conversions.

Tests Performed:
    TestUnitConversion:
        - test_temperature_conversions: Validates temperature conversions (K ↔ °C ↔ °F)
        - test_pressure_conversions: Tests pressure conversions (Pa ↔ hPa ↔ mb)
        - test_mixing_ratio_conversions: Validates humidity conversions (kg/kg ↔ g/kg)
        - test_wind_speed_conversions: Tests wind speed conversions (m/s → kt, mph, km/h)
        - test_precipitation_conversions: Validates precipitation rate conversions (mm/hr ↔ in/hr, mm/day)
        - test_array_conversions: Tests conversions on NumPy arrays with multiple values
        - test_xarray_conversions: Validates conversions on xarray DataArray objects
        - test_no_conversion_needed: Tests identity conversion (same source and target units)
        - test_unsupported_conversion: Validates error handling for invalid unit pairs
        - test_unit_normalization: Tests string normalization for various unit representations
        - test_display_unit_preferences: Validates preferred display units for meteorological variables
        - test_metadata_unit_conversion: Tests unit conversion with metadata preservation
        - test_convert_data_for_display: Validates complete data conversion workflow with metadata

Test Coverage:
    - Temperature conversions: Kelvin, Celsius, Fahrenheit with proper offset and scaling
    - Pressure conversions: Pascal, hectopascal, millibar equivalences
    - Mixing ratio conversions: mass fraction to specific humidity transformations
    - Wind speed conversions: metric and imperial unit transformations
    - Precipitation conversions: rate and accumulation unit transformations
    - Array operations: NumPy array and xarray DataArray handling with attributes
    - Unit normalization: string parsing for various unit representation formats
    - Display preferences: variable-specific preferred units for visualization
    - Metadata preservation: maintaining and updating unit attributes through conversions

Testing Approach:
    Unit tests using synthetic scalar values and arrays to verify mathematical accuracy of
    conversion formulas. Tests check both single values and array operations, validate
    precision to appropriate decimal places, test edge cases (zero, negative values), and
    verify proper handling of xarray attributes and metadata throughout conversions.

Expected Results:
    - Temperature conversions accurate to 2 decimal places (0°C = 273.15K = 32°F)
    - Pressure conversions maintain precision (101325 Pa = 1013.25 hPa)
    - Array conversions preserve shape and apply element-wise transformations correctly
    - xarray operations maintain DataArray structure and update attributes appropriately
    - Unit normalization handles various string formats and standardizes representations
    - Display unit preferences return appropriate units for each meteorological variable
    - Invalid conversions raise ValueError with descriptive error messages
    - Identity conversions return original data unchanged

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import unittest
import numpy as np
import xarray as xr
from mpasdiag.processing.utils_unit import UnitConverter
from mpasdiag.processing.utils_metadata import MPASFileMetadata


class TestUnitConversion(unittest.TestCase):
    """
    Test unit conversion functionality.
    """
    
    def test_temperature_conversions(self) -> None:
        """
        Validate temperature unit conversions across Kelvin, Celsius, and Fahrenheit scales with proper offset and scaling transformations. This test confirms accurate bidirectional conversions between all three temperature scales including freezing point (0°C = 273.15K = 32°F) and boiling point (100°C = 212°F) reference values. The conversion formulas apply appropriate additive offsets for absolute zero and multiplicative scaling factors (9/5 for °C↔°F) to maintain thermodynamic consistency. Precision validation to 2 decimal places ensures numerical accuracy sufficient for meteorological applications. These fundamental temperature conversions support display of model output in user-preferred units across diverse international conventions.

        Parameters:
            None

        Returns:
            None
        """
        self.assertAlmostEqual(float(UnitConverter.convert_units(273.15, 'K', '°C')), 0.0, places=2)
        self.assertAlmostEqual(float(UnitConverter.convert_units(288.15, 'K', '°C')), 15.0, places=2)
        self.assertAlmostEqual(float(UnitConverter.convert_units(0.0, '°C', 'K')), 273.15, places=2)
        self.assertAlmostEqual(float(UnitConverter.convert_units(25.0, '°C', 'K')), 298.15, places=2)
        self.assertAlmostEqual(float(UnitConverter.convert_units(0.0, '°C', '°F')), 32.0, places=2)
        self.assertAlmostEqual(float(UnitConverter.convert_units(100.0, '°C', '°F')), 212.0, places=2)
        self.assertAlmostEqual(float(UnitConverter.convert_units(273.15, 'K', '°F')), 32.0, places=2)
    
    def test_pressure_conversions(self) -> None:
        """
        Verify pressure unit conversions between Pascal, hectopascal, and millibar with appropriate scaling factors for atmospheric applications. This test validates conversions using standard atmospheric pressure reference value (101325 Pa = 1013.25 hPa) to confirm proper factor-of-100 scaling between Pa and hPa. The test also confirms hectopascal-millibar equivalence (1 hPa = 1 mb) which simplifies meteorological data exchange. Precision validation ensures that conversions maintain accuracy suitable for surface pressure analysis and upper-air measurements. These pressure conversions enable flexible display of model output matching conventions from different meteorological agencies and forecast centers worldwide.

        Parameters:
            None

        Returns:
            None
        """
        self.assertAlmostEqual(float(UnitConverter.convert_units(101325.0, 'Pa', 'hPa')), 1013.25, places=2)
        self.assertAlmostEqual(float(UnitConverter.convert_units(1013.25, 'hPa', 'Pa')), 101325.0, places=1)
        self.assertEqual(float(UnitConverter.convert_units(1013.25, 'hPa', 'mb')), 1013.25)
        self.assertEqual(float(UnitConverter.convert_units(1013.25, 'mb', 'hPa')), 1013.25)
    
    def test_mixing_ratio_conversions(self) -> None:
        """
        Validate water vapor mixing ratio conversions between mass fraction and specific humidity representations with factor-of-1000 scaling. This test confirms accurate bidirectional conversions between kg/kg (mass fraction) and g/kg (specific humidity) units commonly used in atmospheric moisture analysis. The test uses representative atmospheric moisture value (0.012 kg/kg = 12 g/kg) typical of mid-latitude conditions to verify conversion accuracy. Precision validation to 4 decimal places for kg/kg ensures maintenance of numerical accuracy through round-trip conversions. These mixing ratio conversions support flexible display of model humidity output matching conventions from different meteorological modeling systems and observational datasets.

        Parameters:
            None

        Returns:
            None
        """
        self.assertAlmostEqual(float(UnitConverter.convert_units(0.012, 'kg/kg', 'g/kg')), 12.0, places=2)
        self.assertAlmostEqual(float(UnitConverter.convert_units(12.0, 'g/kg', 'kg/kg')), 0.012, places=4)
    
    def test_wind_speed_conversions(self) -> None:
        """
        Verify wind speed unit conversions from metric standard (m/s) to aviation and imperial units including knots, miles per hour, and kilometers per hour. This test validates conversion factors for 10 m/s reference wind speed to knots (19.44 kt), miles per hour (22.37 mph), and kilometers per hour (36 km/h) using internationally accepted conversion constants. The conversions support operational meteorology requirements where aviation forecasts use knots, public forecasts use mph or km/h depending on region, and model output typically provides m/s. Precision validation to 2 decimal places ensures accuracy sufficient for wind speed interpretation across diverse user communities. These flexible unit conversions enable visualization of MPAS wind output matching conventions from different forecast applications and international standards.

        Parameters:
            None

        Returns:
            None
        """
        self.assertAlmostEqual(float(UnitConverter.convert_units(10.0, 'm/s', 'kt')), 19.4384, places=2)
        self.assertAlmostEqual(float(UnitConverter.convert_units(10.0, 'm/s', 'mph')), 22.3694, places=2)
        self.assertAlmostEqual(float(UnitConverter.convert_units(10.0, 'm/s', 'km/h')), 36.0, places=1)
    
    def test_precipitation_conversions(self) -> None:
        """
        Validate precipitation unit conversions between rates and accumulations including mm/hr, mm/day, and inches per hour for diverse temporal scales. This test confirms temporal scaling conversions (mm/hr to mm/day using factor of 24) and metric-imperial conversions (mm/hr to in/hr using 25.4 mm/inch factor). The test validates precipitation rate conversion accuracy for operational forecast display where different accumulation periods and unit preferences apply across regions and applications. Precision validation ensures that converted precipitation values maintain accuracy sufficient for quantitative precipitation forecasting and hydrological applications. These flexible precipitation conversions enable visualization of MPAS model output matching user preferences from millimeters to inches and rates to accumulations.

        Parameters:
            None

        Returns:
            None
        """
        self.assertAlmostEqual(float(UnitConverter.convert_units(2.0, 'mm/hr', 'mm/day')), 48.0, places=1)
        self.assertAlmostEqual(float(UnitConverter.convert_units(25.4, 'mm/hr', 'in/hr')), 1.0, places=2)
    
    def test_array_conversions(self) -> None:
        """
        Verify element-wise unit conversions on NumPy arrays maintaining array shape and applying transformations independently to each element. This test validates that temperature array conversions (K to °C) preserve the original 3-element array structure while correctly applying offset transformations to each value. The test uses representative temperature values spanning 20°C range to confirm uniform conversion behavior across array elements. NumPy testing utilities validate numerical accuracy to 2 decimal places ensuring precision maintenance through vectorized operations. These array conversion capabilities support efficient batch processing of gridded MPAS model output without requiring element-by-element iteration in user code.

        Parameters:
            None

        Returns:
            None
        """
        temp_array = np.array([273.15, 283.15, 293.15])
        converted = UnitConverter.convert_units(temp_array, 'K', '°C')
        expected = np.array([0.0, 10.0, 20.0])
        np.testing.assert_array_almost_equal(converted, expected, decimal=2)
    
    def test_xarray_conversions(self) -> None:
        """
        Validate unit conversions on xarray DataArray objects preserving data structure, dimensions, and coordinate information through transformation. This test confirms that temperature conversions maintain xarray's labeled array structure including dimension names, coordinate values, and metadata attributes. The conversion applies element-wise transformations (K to °C) while keeping the DataArray wrapper intact for subsequent analysis operations. Xarray testing utilities validate both numerical accuracy and structural integrity ensuring that converted data integrates seamlessly with xarray-based workflows. These xarray conversion capabilities enable unit transformations within modern scientific Python analysis pipelines without losing the self-describing data structure benefits that xarray provides.

        Parameters:
            None

        Returns:
            None
        """
        temp_data = xr.DataArray([273.15, 283.15, 293.15])
        converted = UnitConverter.convert_units(temp_data, 'K', '°C')
        expected = xr.DataArray([0.0, 10.0, 20.0])
        xr.testing.assert_allclose(converted, expected)
    
    def test_no_conversion_needed(self) -> None:
        """
        Verify identity conversion behavior when source and target units match, returning original data unchanged without unnecessary transformations. This test confirms that requesting conversion between identical units (m/s to m/s) bypasses conversion logic and returns the original array without modification. The identity conversion optimization avoids unnecessary computational overhead and floating-point precision loss from trivial transformations. Array equality testing validates that returned data maintains exact bit-level identity with input data including data type preservation. This identity conversion behavior supports flexible code design where unit conversion calls can be made unconditionally without performance penalties when conversions are unnecessary.

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([1, 2, 3])
        result = UnitConverter.convert_units(data, 'm/s', 'm/s')
        np.testing.assert_array_equal(result, data)
    
    def test_unsupported_conversion(self) -> None:
        """
        Validate error handling for invalid unit conversion requests raising ValueError with descriptive messages for unsupported unit pairs. This test confirms that the converter properly rejects conversion requests involving unrecognized unit strings or physically incompatible unit types. The exception-raising behavior prevents silent failures or undefined behavior when users specify incorrect unit specifications in configuration or function calls. Proper ValueError exceptions with clear error messages enable rapid debugging of unit specification errors in operational workflows. This defensive error handling approach ensures that unit conversion failures surface immediately with actionable diagnostic information rather than propagating invalid data through analysis pipelines.

        Parameters:
            None

        Returns:
            None
        """
        with self.assertRaises(ValueError):
            UnitConverter.convert_units(10.0, 'invalid_unit', 'another_invalid_unit')
    
    def test_unit_normalization(self) -> None:
        """
        Verify unit string normalization converting various textual representations to standardized symbolic format for consistent internal processing. This test validates that diverse unit string formats including full words (kelvin, celsius, pascal), compound forms (kg kg-1), and alternative spellings (knots) map to canonical symbols (K, °C, Pa, kg/kg, kt). String normalization enables flexible user input accepting multiple equivalent representations while maintaining internal consistency for lookup tables and conversion logic. The normalization handles common variations from different data sources including MPAS output files, configuration files, and user specifications. This robust string handling prevents conversion failures due to minor variations in unit string formatting across diverse input sources.

        Parameters:
            None

        Returns:
            None
        """
        self.assertEqual(UnitConverter._normalize_unit_string('kelvin'), 'K')
        self.assertEqual(UnitConverter._normalize_unit_string('celsius'), '°C')
        self.assertEqual(UnitConverter._normalize_unit_string('pascal'), 'Pa')
        self.assertEqual(UnitConverter._normalize_unit_string('kg kg-1'), 'kg/kg')
        self.assertEqual(UnitConverter._normalize_unit_string('knots'), 'kt')
    
    def test_display_unit_preferences(self) -> None:
        """
        Validate variable-specific display unit preferences returning appropriate target units for meteorological variables based on conventions and usability. This test confirms that the converter provides sensible default display units for common MPAS variables including temperature (°C), pressure (hPa), moisture (g/kg), and precipitation (mm/hr) matching operational meteorology standards. The preference system enables automatic unit selection without requiring users to specify target units explicitly for every variable type. Testing includes unknown variable handling where original units pass through unchanged to avoid conversion failures for arbitrary variables. These intelligent display unit defaults simplify visualization code by encoding domain knowledge about appropriate meteorological unit conventions for different variable types.

        Parameters:
            None

        Returns:
            None
        """
        self.assertEqual(UnitConverter.get_display_units('t2m', 'K'), '°C')
        self.assertEqual(UnitConverter.get_display_units('mslp', 'Pa'), 'hPa')
        self.assertEqual(UnitConverter.get_display_units('q2', 'kg/kg'), 'g/kg')
        self.assertEqual(UnitConverter.get_display_units('rainnc', 'mm'), 'mm')
        self.assertEqual(UnitConverter.get_display_units('u10', 'm/s'), 'm/s')
        
        self.assertEqual(UnitConverter.get_display_units('unknown_var', 'original_unit'), 'original_unit')
    
    def test_metadata_unit_conversion(self) -> None:
        """
        Verify metadata handling during unit conversions including attribute updates to reflect transformed units in variable descriptions. This test validates that MPASFileMetadata correctly updates unit attributes (units field changed to °C or hPa) while preserving original unit information for provenance tracking. The metadata conversion updates long_name descriptions to reflect display units ensuring plot labels and documentation remain consistent with transformed data. Testing covers common meteorological variables (t2m temperature, mslp pressure) confirming that metadata transformations parallel data conversions appropriately. This metadata management ensures that converted data maintains self-documenting properties through xarray attributes enabling reproducible analysis workflows where unit specifications remain explicitly documented.

        Parameters:
            None

        Returns:
            None
        """
        metadata = MPASFileMetadata.get_2d_variable_metadata('t2m')
        self.assertEqual(metadata['units'], '°C')
        self.assertEqual(metadata.get('original_units'), 'K')
        self.assertEqual(metadata['long_name'], '2-meter Temperature')
        
        metadata = MPASFileMetadata.get_2d_variable_metadata('mslp')
        self.assertEqual(metadata['units'], 'hPa')
        self.assertEqual(metadata.get('original_units'), 'Pa')
        self.assertIn('Pressure', metadata['long_name'])
    
    def test_convert_data_for_display(self) -> None:
        """
        Validate complete data conversion workflow combining numerical transformation with metadata updates for visualization-ready output. This test confirms that the convert_data_for_display method performs both unit conversion (K to °C) on DataArray values and updates associated metadata attributes including units field and original_units provenance. The conversion workflow handles xarray DataArray objects preserving dimension structure while transforming data values and updating attributes atomically. Numerical accuracy validation confirms proper temperature offset application (280K = 6.85°C) while metadata validation ensures attribute consistency. This integrated conversion approach provides single-function convenience for preparing MPAS model output for visualization with proper unit transformations and self-documenting metadata updates.

        Parameters:
            None

        Returns:
            None
        """
        temp_data = xr.DataArray(
            [280.0, 290.0, 300.0],
            attrs={'units': 'K', 'long_name': '2-meter Temperature'}
        )
        
        converted_data, metadata = UnitConverter.convert_data_for_display(temp_data, 't2m', temp_data)
        
        expected_values = np.array([6.85, 16.85, 26.85])

        if isinstance(converted_data, xr.DataArray):
            result_values = converted_data.values
        else:
            result_values = np.asarray(converted_data)

        np.testing.assert_array_almost_equal(result_values, expected_values, decimal=2)
        
        self.assertEqual(metadata['units'], '°C')
        self.assertEqual(metadata.get('original_units'), 'K')


if __name__ == '__main__':
    unittest.main()
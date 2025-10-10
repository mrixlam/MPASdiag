#!/usr/bin/env python3

"""
Unit tests for unit conversion functionality.
"""

import unittest
import numpy as np
import xarray as xr
from mpas_analysis.visualization import (
    UnitConverter,
    MPASFileMetadata
)


class TestUnitConversion(unittest.TestCase):
    """Test unit conversion functionality."""
    
    def test_temperature_conversions(self):
        """Test temperature unit conversions."""
        self.assertAlmostEqual(UnitConverter.convert_units(273.15, 'K', '°C'), 0.0, places=2)
        self.assertAlmostEqual(UnitConverter.convert_units(288.15, 'K', '°C'), 15.0, places=2)
        self.assertAlmostEqual(UnitConverter.convert_units(0.0, '°C', 'K'), 273.15, places=2)
        self.assertAlmostEqual(UnitConverter.convert_units(25.0, '°C', 'K'), 298.15, places=2)
        self.assertAlmostEqual(UnitConverter.convert_units(0.0, '°C', '°F'), 32.0, places=2)
        self.assertAlmostEqual(UnitConverter.convert_units(100.0, '°C', '°F'), 212.0, places=2)
        self.assertAlmostEqual(UnitConverter.convert_units(273.15, 'K', '°F'), 32.0, places=2)
    
    def test_pressure_conversions(self):
        """Test pressure unit conversions."""
        self.assertAlmostEqual(UnitConverter.convert_units(101325.0, 'Pa', 'hPa'), 1013.25, places=2)
        self.assertAlmostEqual(UnitConverter.convert_units(1013.25, 'hPa', 'Pa'), 101325.0, places=1)
        self.assertEqual(UnitConverter.convert_units(1013.25, 'hPa', 'mb'), 1013.25)
        self.assertEqual(UnitConverter.convert_units(1013.25, 'mb', 'hPa'), 1013.25)
    
    def test_mixing_ratio_conversions(self):
        """Test mixing ratio conversions."""
        self.assertAlmostEqual(UnitConverter.convert_units(0.012, 'kg/kg', 'g/kg'), 12.0, places=2)
        self.assertAlmostEqual(UnitConverter.convert_units(12.0, 'g/kg', 'kg/kg'), 0.012, places=4)
    
    def test_wind_speed_conversions(self):
        """Test wind speed conversions."""
        self.assertAlmostEqual(UnitConverter.convert_units(10.0, 'm/s', 'kt'), 19.4384, places=2)
        self.assertAlmostEqual(UnitConverter.convert_units(10.0, 'm/s', 'mph'), 22.3694, places=2)
        self.assertAlmostEqual(UnitConverter.convert_units(10.0, 'm/s', 'km/h'), 36.0, places=1)
    
    def test_precipitation_conversions(self):
        """Test precipitation conversions."""
        self.assertAlmostEqual(UnitConverter.convert_units(2.0, 'mm/hr', 'mm/day'), 48.0, places=1)
        self.assertAlmostEqual(UnitConverter.convert_units(25.4, 'mm/hr', 'in/hr'), 1.0, places=2)
    
    def test_array_conversions(self):
        """Test conversions with numpy arrays."""
        temp_array = np.array([273.15, 283.15, 293.15])
        converted = UnitConverter.convert_units(temp_array, 'K', '°C')
        expected = np.array([0.0, 10.0, 20.0])
        np.testing.assert_array_almost_equal(converted, expected, decimal=2)
    
    def test_xarray_conversions(self):
        """Test conversions with xarray DataArrays."""
        temp_data = xr.DataArray([273.15, 283.15, 293.15])
        converted = UnitConverter.convert_units(temp_data, 'K', '°C')
        expected = xr.DataArray([0.0, 10.0, 20.0])
        xr.testing.assert_allclose(converted, expected)
    
    def test_no_conversion_needed(self):
        """Test when no conversion is needed."""
        data = np.array([1, 2, 3])
        result = UnitConverter.convert_units(data, 'm/s', 'm/s')
        np.testing.assert_array_equal(result, data)
    
    def test_unsupported_conversion(self):
        """Test error handling for unsupported conversions."""
        with self.assertRaises(ValueError):
            UnitConverter.convert_units(10.0, 'invalid_unit', 'another_invalid_unit')
    
    def test_unit_normalization(self):
        """Test unit string normalization."""
        self.assertEqual(UnitConverter._normalize_unit_string('kelvin'), 'K')
        self.assertEqual(UnitConverter._normalize_unit_string('celsius'), '°C')
        self.assertEqual(UnitConverter._normalize_unit_string('pascal'), 'Pa')
        self.assertEqual(UnitConverter._normalize_unit_string('kg kg-1'), 'kg/kg')
        self.assertEqual(UnitConverter._normalize_unit_string('knots'), 'kt')
    
    def test_display_unit_preferences(self):
        """Test display unit preferences."""
        self.assertEqual(UnitConverter.get_display_units('t2m', 'K'), '°C')
        self.assertEqual(UnitConverter.get_display_units('mslp', 'Pa'), 'hPa')
        self.assertEqual(UnitConverter.get_display_units('q2', 'kg/kg'), 'g/kg')
        self.assertEqual(UnitConverter.get_display_units('rainnc', 'mm'), 'mm/hr')
        self.assertEqual(UnitConverter.get_display_units('u10', 'm/s'), 'm/s')
        
        self.assertEqual(UnitConverter.get_display_units('unknown_var', 'original_unit'), 'original_unit')
    
    def test_metadata_unit_conversion(self):
        """Test metadata with automatic unit conversion."""
        metadata = MPASFileMetadata.get_2d_variable_metadata('t2m')
        self.assertEqual(metadata['units'], '°C')
        self.assertEqual(metadata.get('original_units'), 'K')
        self.assertIn('°C', metadata['long_name'])
        
        metadata = MPASFileMetadata.get_2d_variable_metadata('mslp')
        self.assertEqual(metadata['units'], 'hPa')
        self.assertEqual(metadata.get('original_units'), 'Pa')
        self.assertIn('hPa', metadata['long_name'])
    
    def test_convert_data_for_display(self):
        """Test the convert_data_for_display function."""
        temp_data = xr.DataArray(
            [280.0, 290.0, 300.0],
            attrs={'units': 'K', 'long_name': '2-meter Temperature'}
        )
        
        converted_data, metadata = UnitConverter.convert_data_for_display(temp_data, 't2m', temp_data)
        
        expected_values = np.array([6.85, 16.85, 26.85])  
        np.testing.assert_array_almost_equal(converted_data.values, expected_values, decimal=2)
        
        self.assertEqual(metadata['units'], '°C')
        self.assertEqual(metadata.get('original_units'), 'K')


if __name__ == '__main__':
    unittest.main()
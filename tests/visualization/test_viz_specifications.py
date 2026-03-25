#!/usr/bin/env python3
"""
MPASdiag Test Suite: Visualization Functionality

This test suite verifies the presence and importability of visualization modules and classes within the MPASdiag package. It ensures that key plotting components for cross-sections, wind, surface, and precipitation visualizations are accessible and can be imported without errors. The tests focus on confirming the existence of the expected classes and modules, which is critical for downstream plotting functionality to work correctly. By validating the import paths and class definitions, this suite helps catch packaging or refactoring issues that could break visualization features. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries 
import os
import sys
import pytest
import matplotlib

matplotlib.use('Agg')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestCrossSectionVisualization:
    """ Test vertical cross-section plotting functionality. """
    
    def test_import_cross_section_module(self: "TestCrossSectionVisualization") -> None:
        """
        This test verifies that the cross-section visualization module can be imported without errors. Successful import indicates that the module is correctly packaged and available in the runtime environment, which is essential for any functionality that relies on it.

        Parameters:
            None

        Returns:
            None: Verified by asserting the imported module is not None.
        """         
        from mpasdiag.visualization import cross_section
        assert cross_section is not None
    
    def test_cross_section_plotter_class_exists(self: "TestCrossSectionVisualization") -> None:
        """
        This test checks that the `MPASVerticalCrossSectionPlotter` class is defined and can be imported from the cross-section visualization module. The presence of this class is crucial for implementing vertical cross-section plotting functionality, and its importability confirms that it is correctly exposed by the module.

        Parameters:
            None

        Returns:
            None: Confirmed by asserting the class symbol is not None.
        """        
        from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
        assert MPASVerticalCrossSectionPlotter is not None


class TestWindVisualization:
    """ Test wind plotting functionality. """
    
    def test_import_wind_module(self: "TestWindVisualization") -> None:
        """
        This test ensures that the wind visualization module can be imported successfully. The ability to import this module without errors is a fundamental requirement for any wind plotting features to function, as it indicates that the module is correctly included in the package and does not have import-related issues.

        Parameters:
            None

        Returns:
            None: Verified by asserting the imported module is not None.
        """
        
        from mpasdiag.visualization import wind
        assert wind is not None
    
    def test_wind_plotter_class_exists(self: "TestWindVisualization") -> None:
        """
        This test verifies that the `MPASWindPlotter` class is defined and can be imported from the wind visualization module. The existence of this class is essential for implementing wind plotting functionality, and its successful import confirms that it is properly exposed by the module for use in visualization tasks.

        Parameters:
            None

        Returns:
            None: Confirmed by asserting the class symbol is not None.
        """        
        from mpasdiag.visualization.wind import MPASWindPlotter
        assert MPASWindPlotter is not None


class TestSurfaceVisualization:
    """ Test surface plotting functionality. """
    
    def test_import_surface_module(self: "TestSurfaceVisualization") -> None:
        """
        This test ensures that the surface visualization module can be imported successfully. The ability to import this module without errors is a fundamental requirement for any surface plotting features to function, as it indicates that the module is correctly included in the package and does not have import-related issues.

        Parameters:
            None

        Returns:
            None: Verified by asserting the imported module is not None.
        """        
        from mpasdiag.visualization import surface
        assert surface is not None
    
    def test_surface_plotter_class_exists(self: "TestSurfaceVisualization") -> None:
        """
        This test verifies that the `MPASSurfacePlotter` class is defined and can be imported from the surface visualization module. The existence of this class is essential for implementing surface plotting functionality, and its successful import confirms that it is properly exposed by the module for use in visualization tasks.

        Parameters:
            None

        Returns:
            None: Confirmed by asserting the class symbol is not None.
        """        
        from mpasdiag.visualization.surface import MPASSurfacePlotter
        assert MPASSurfacePlotter is not None


class TestPrecipitationVisualization:
    """ Test precipitation plotting functionality. """
    
    def test_import_precipitation_module(self: "TestPrecipitationVisualization") -> None:
        """
        This test ensures that the precipitation visualization module can be imported successfully. The ability to import this module without errors is a fundamental requirement for any precipitation plotting features to function, as it indicates that the module is correctly included in the package and does not have import-related issues.

        Parameters:
            None

        Returns:
            None: Verified by asserting the imported module is not None.
        """        
        from mpasdiag.visualization import precipitation
        assert precipitation is not None
    
    def test_precipitation_plotter_class_exists(self: "TestPrecipitationVisualization") -> None:
        """
        This test verifies that the `MPASPrecipitationPlotter` class is defined and can be imported from the precipitation visualization module. The existence of this class is essential for implementing precipitation plotting functionality, and its successful import confirms that it is properly exposed by the module for use in visualization tasks.

        Parameters:
            None

        Returns:
            None: Confirmed by asserting the class symbol is not None.
        """        
        from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter
        assert MPASPrecipitationPlotter is not None


class TestBaseVisualizer:
    """ Test base visualizer functionality. """
    
    def test_import_base_visualizer_module(self: "TestBaseVisualizer") -> None:
        """
        This test ensures that the base visualizer module can be imported successfully. The ability to import this module without errors is a fundamental requirement for any base visualizer features to function, as it indicates that the module is correctly included in the package and does not have import-related issues.

        Parameters:
            None

        Returns:
            None: Verified by asserting the module import returns a value.
        """        
        from mpasdiag.visualization import base_visualizer
        assert base_visualizer is not None
    
    def test_base_visualizer_class_exists(self: "TestBaseVisualizer") -> None:
        """
        This test verifies that the `MPASVisualizer` class is defined and can be imported from the base visualizer module. The existence of this class is essential for providing common visualization functionality that other plotter classes may inherit from, and its successful import confirms that it is properly exposed by the module for use in visualization tasks.

        Parameters:
            None

        Returns:
            None: Verified via assertion that the class symbol is not None.
        """        
        from mpasdiag.visualization.base_visualizer import MPASVisualizer
        assert MPASVisualizer is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

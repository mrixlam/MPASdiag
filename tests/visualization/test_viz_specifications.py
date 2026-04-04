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
import inspect
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
        import importlib
        import types
        
        cross_section = importlib.import_module("mpasdiag.visualization.cross_section")

        assert isinstance(cross_section, types.ModuleType)
        assert hasattr(cross_section, "__file__")

        assert hasattr(cross_section, "MPASVerticalCrossSectionPlotter")
        plotter_cls = getattr(cross_section, "MPASVerticalCrossSectionPlotter")
        assert isinstance(plotter_cls, type)
    
    def test_cross_section_plotter_class_exists(self: "TestCrossSectionVisualization") -> None:
        """
        This test checks that the `MPASVerticalCrossSectionPlotter` class is defined and can be imported from the cross-section visualization module. The presence of this class is crucial for implementing vertical cross-section plotting functionality, and its importability confirms that it is correctly exposed by the module.

        Parameters:
            None

        Returns:
            None: Confirmed by asserting the class symbol is not None.
        """        
        from mpasdiag.visualization.cross_section import MPASVerticalCrossSectionPlotter
        assert isinstance(MPASVerticalCrossSectionPlotter, type)

        public_methods = [
            name for name, member in inspect.getmembers(MPASVerticalCrossSectionPlotter, predicate=inspect.isfunction)
            if not name.startswith('_')
        ]

        assert public_methods, "MPASVerticalCrossSectionPlotter should have at least one public method"


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
        import importlib
        import types

        wind = importlib.import_module("mpasdiag.visualization.wind")

        assert isinstance(wind, types.ModuleType)
        assert hasattr(wind, "__file__")

        assert hasattr(wind, "MPASWindPlotter")
        plotter_cls = getattr(wind, "MPASWindPlotter")
        assert isinstance(plotter_cls, type)
    
    def test_wind_plotter_class_exists(self: "TestWindVisualization") -> None:
        """
        This test verifies that the `MPASWindPlotter` class is defined and can be imported from the wind visualization module. The existence of this class is essential for implementing wind plotting functionality, and its successful import confirms that it is properly exposed by the module for use in visualization tasks.

        Parameters:
            None

        Returns:
            None: Confirmed by asserting the class symbol is not None.
        """        
        from mpasdiag.visualization.wind import MPASWindPlotter
        assert isinstance(MPASWindPlotter, type)

        public_methods = [
            name for name, member in inspect.getmembers(MPASWindPlotter, predicate=inspect.isfunction)
            if not name.startswith('_')
        ]

        assert public_methods, "MPASWindPlotter should have at least one public method"

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
        import importlib
        import types

        surface = importlib.import_module("mpasdiag.visualization.surface")

        assert isinstance(surface, types.ModuleType)
        assert hasattr(surface, "__file__")

        assert hasattr(surface, "MPASSurfacePlotter")
        plotter_cls = getattr(surface, "MPASSurfacePlotter")
        assert isinstance(plotter_cls, type)
        
    def test_surface_plotter_class_exists(self: "TestSurfaceVisualization") -> None:
        """
        This test verifies that the `MPASSurfacePlotter` class is defined and can be imported from the surface visualization module. The existence of this class is essential for implementing surface plotting functionality, and its successful import confirms that it is properly exposed by the module for use in visualization tasks.

        Parameters:
            None

        Returns:
            None: Confirmed by asserting the class symbol is not None.
        """        
        from mpasdiag.visualization.surface import MPASSurfacePlotter
        assert isinstance(MPASSurfacePlotter, type)

        public_methods = [
            name for name, _ in inspect.getmembers(MPASSurfacePlotter, predicate=inspect.isfunction)
            if not name.startswith('_')
        ]

        assert public_methods, "MPASSurfacePlotter should have at least one public method"


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
        import importlib
        import types

        precipitation = importlib.import_module("mpasdiag.visualization.precipitation")

        assert isinstance(precipitation, types.ModuleType)
        assert hasattr(precipitation, "__file__")
        assert hasattr(precipitation, "MPASPrecipitationPlotter")

        plotter_cls = getattr(precipitation, "MPASPrecipitationPlotter")
        assert isinstance(plotter_cls, type)        
    
    def test_precipitation_plotter_class_exists(self: "TestPrecipitationVisualization") -> None:
        """
        This test verifies that the `MPASPrecipitationPlotter` class is defined and can be imported from the precipitation visualization module. The existence of this class is essential for implementing precipitation plotting functionality, and its successful import confirms that it is properly exposed by the module for use in visualization tasks.

        Parameters:
            None

        Returns:
            None: Confirmed by asserting the class symbol is not None.
        """        
        from mpasdiag.visualization.precipitation import MPASPrecipitationPlotter
        assert isinstance(MPASPrecipitationPlotter, type)

        public_methods = [
            name for name, member in inspect.getmembers(MPASPrecipitationPlotter, predicate=inspect.isfunction)
            if not name.startswith('_')
        ]

        assert public_methods, "MPASPrecipitationPlotter should have at least one public method"


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
        import numpy as np
        from mpasdiag.visualization import base_visualizer
        assert base_visualizer is not None
        assert hasattr(base_visualizer, "__file__")

        assert hasattr(base_visualizer, "MPASVisualizer")
        MPASVisualizer = getattr(base_visualizer, "MPASVisualizer")
        assert isinstance(MPASVisualizer, type)

        assert hasattr(MPASVisualizer, "convert_to_numpy")
        arr = MPASVisualizer.convert_to_numpy((1, 2, 3))

        assert isinstance(arr, np.ndarray)
        assert np.all(arr == np.array([1, 2, 3]))        
    
    def test_base_visualizer_class_exists(self: "TestBaseVisualizer") -> None:
        """
        This test verifies that the `MPASVisualizer` class is defined and can be imported from the base visualizer module. The existence of this class is essential for providing common visualization functionality that other plotter classes may inherit from, and its successful import confirms that it is properly exposed by the module for use in visualization tasks.

        Parameters:
            None

        Returns:
            None: Verified via assertion that the class symbol is not None.
        """        
        import numpy as np
        from mpasdiag.visualization.base_visualizer import MPASVisualizer
        assert isinstance(MPASVisualizer, type)
        assert hasattr(MPASVisualizer, "convert_to_numpy")

        method = getattr(MPASVisualizer, "convert_to_numpy")
        assert inspect.isfunction(method) or inspect.ismethod(method)

        for data in [(1, 2, 3), [4, 5, 6], np.array([7, 8, 9])]:
            arr = MPASVisualizer.convert_to_numpy(data)
            assert isinstance(arr, np.ndarray)
            assert np.allclose(arr, np.array(data))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

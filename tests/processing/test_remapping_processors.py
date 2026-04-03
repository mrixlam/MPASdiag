#!/usr/bin/env python3
"""
MPASdiag Test Suite: Tests for MPASdiag remapping functionality

This module contains integration tests for the remapping functionality in MPASdiag, specifically focusing on the end-to-end remapping process using real MPAS data. The tests verify that the remapping functions can successfully remap data defined on a real MPAS grid to a regular lat-lon grid, and that the main remapping class can be used

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules for testing
import pytest
import xesmf as xe

if xe is not None:
    XESMF_AVAILABLE = True
else:
    XESMF_AVAILABLE = False

REMAPPING_AVAILABLE = True


class TestMPAS2DProcessor:
    """ Test 2D data processor functionality and API surface. """
    
    def test_import_2d_processor(self: "TestMPAS2DProcessor") -> None:
        """
        This test verifies that the `MPAS2DProcessor` class can be imported successfully from the `processors_2d` module. Successful import indicates that the module is correctly structured, dependencies are satisfied, and there are no syntax errors preventing the class from being defined. This is a fundamental test to ensure that subsequent tests can run against the processor class. 

        Parameters:
            self ("TestMPAS2DProcessor"): Test instance (unused).

        Returns:
            None: Assertions validate import success.
        """
        from mpasdiag.processing.processors_2d import MPAS2DProcessor

        public_methods = [method for method in dir(MPAS2DProcessor) if not method.startswith('_')]
        expected_methods = [
            'find_diagnostic_files',
            'load_2d_data',
            'add_spatial_coordinates',
            'extract_2d_coordinates_for_variable',
            'get_2d_variable_data',
            'get_available_variables'
        ]
        for method in expected_methods:
            assert method in public_methods, f"Expected method '{method}' not found in MPAS2DProcessor"
    
    def test_2d_processor_has_base_class(self: "TestMPAS2DProcessor") -> None:
        """
        This test checks that the `MPAS2DProcessor` class is a subclass of `MPASBaseProcessor`, ensuring it inherits common functionality and adheres to the expected interface for processors in MPASdiag. Subclassing the base processor allows for consistent behavior across different processor types and enables shared utilities to operate on either 2D or 3D processors. 

        Parameters:
            self ("TestMPAS2DProcessor"): Test instance (unused).

        Returns:
            None: Assertion verifies subclass relationship.
        """
        from mpasdiag.processing.processors_2d import MPAS2DProcessor
        from mpasdiag.processing.base import MPASBaseProcessor
        assert issubclass(MPAS2DProcessor, MPASBaseProcessor)
    
    def test_2d_processor_has_init_method(self: "TestMPAS2DProcessor") -> None:
        """
        This test inspects the constructor of the `MPAS2DProcessor` class to confirm that it accepts expected parameters such as `grid_file` and `verbose`. The presence of these parameters in the constructor signature is important for users who need to initialize the processor with specific grid information and control verbosity. By using the `inspect` module, the test checks that these parameters are defined without actually invoking the constructor, thus avoiding any side effects or file I/O during testing. 

        Parameters:
            self ("TestMPAS2DProcessor"): Test instance (unused).

        Returns:
            None: Assertion inspects the constructor signature.
        """
        from mpasdiag.processing.processors_2d import MPAS2DProcessor
        import inspect
        sig = inspect.signature(MPAS2DProcessor.__init__)
        params = list(sig.parameters.keys())
        assert 'grid_file' in params
        assert 'verbose' in params
    
    def test_2d_processor_has_find_diagnostic_files_method(self: "TestMPAS2DProcessor") -> None:
        """
        This test verifies that the `MPAS2DProcessor` class has a method named `find_diagnostic_files`, which is essential for locating diagnostic files needed for processing. The presence of this method is crucial for the processor to function correctly, as it allows users to specify how diagnostic files are discovered. The test checks both the existence of the method and that it is callable, ensuring that the API is intact and ready for use by higher-level routines without performing any actual file I/O.  

        Parameters:
            self ("TestMPAS2DProcessor"): Test instance (unused).
        Returns:
            None: Assertion checks callable attribute.
        """
        from mpasdiag.processing.processors_2d import MPAS2DProcessor
        assert hasattr(MPAS2DProcessor, 'find_diagnostic_files')
        assert callable(MPAS2DProcessor.find_diagnostic_files)
    
    def test_2d_processor_has_load_2d_data_method(self: "TestMPAS2DProcessor") -> None:
        """
        This test confirms that the `MPAS2DProcessor` class exposes a method named `load_2d_data`, which is responsible for reading 2D fields from diagnostic files. The presence of this method is essential for the processor to function as it allows users to access variable data needed for plotting and diagnostics. The test checks that the method exists and is callable, ensuring that the API is available for use by higher-level routines without performing any actual file I/O during testing.  

        Parameters:
            self ("TestMPAS2DProcessor"): Test instance (unused).

        Returns:
            None: Assertion validates method availability.
        """
        from mpasdiag.processing.processors_2d import MPAS2DProcessor
        assert hasattr(MPAS2DProcessor, 'load_2d_data')
        assert callable(MPAS2DProcessor.load_2d_data)
    
    def test_2d_processor_has_add_spatial_coordinates_method(self: "TestMPAS2DProcessor") -> None:
        """
        This test checks that the `MPAS2DProcessor` class has a method named `add_spatial_coordinates`, which is used to add spatial coordinate variables (e.g., longitude and latitude) to the processor's internal dataset. This method is important for ensuring that the processor can provide necessary coordinate information for remapping and plotting. The test verifies that the method exists and is callable, confirming that the API supports adding spatial coordinates as expected without performing any actual data manipulation during testing. 

        Parameters:
            self ("TestMPAS2DProcessor"): Test instance (unused).

        Returns:
            None: Assertion ensures attribute exists and is callable.
        """
        from mpasdiag.processing.processors_2d import MPAS2DProcessor
        assert hasattr(MPAS2DProcessor, 'add_spatial_coordinates')
        assert callable(MPAS2DProcessor.add_spatial_coordinates)
    
    def test_2d_processor_has_extract_2d_coordinates_method(self: "TestMPAS2DProcessor") -> None:
        """
        This test verifies that the `MPAS2DProcessor` class has a method named `extract_2d_coordinates_for_variable`, which is responsible for extracting 2D coordinate arrays (longitude and latitude) for a given variable. This functionality is crucial for remapping and horizontal plotting of 2D diagnostics. The test checks that the method exists and is callable, ensuring that the API provides the necessary interface for coordinate extraction without performing any actual data manipulation during testing. 

        Parameters:
            self ("TestMPAS2DProcessor"): Test instance (unused).
        Returns:
            None: Assertion verifies method existence.
        """
        from mpasdiag.processing.processors_2d import MPAS2DProcessor
        assert hasattr(MPAS2DProcessor, 'extract_2d_coordinates_for_variable')
        assert callable(MPAS2DProcessor.extract_2d_coordinates_for_variable)
    
    def test_2d_processor_has_get_2d_variable_data_method(self: "TestMPAS2DProcessor") -> None:
        """
        This test confirms that the `MPAS2DProcessor` class exposes a method named `get_2d_variable_data`, which is used to extract 2D variable data arrays at specific time indices. This method is essential for users who need to access the actual data values for plotting and diagnostics. The test checks that the method exists and is callable, ensuring that the API provides the necessary interface for data retrieval without performing any actual file I/O during testing. 

        Parameters:
            self ("TestMPAS2DProcessor"): Test instance (unused).

        Returns:
            None: Assertion validates method availability.
        """
        from mpasdiag.processing.processors_2d import MPAS2DProcessor
        assert hasattr(MPAS2DProcessor, 'get_2d_variable_data')
        assert callable(MPAS2DProcessor.get_2d_variable_data)
    
    def test_2d_processor_has_get_available_variables_method(self: "TestMPAS2DProcessor") -> None:
        """
        This test checks that the `MPAS2DProcessor` class has a method named `get_available_variables`, which is used to retrieve a list of variable names available in the processor's dataset. This method is important for users to understand what variables can be accessed and processed. The test verifies that the method exists and is callable, ensuring that the API provides a way to query available variables without performing any actual data manipulation during testing. 

        Parameters:
            self ("TestMPAS2DProcessor"): Test instance (unused).

        Returns:
            None: Assertion verifies method presence.
        """
        from mpasdiag.processing.processors_2d import MPAS2DProcessor
        assert hasattr(MPAS2DProcessor, 'get_available_variables')
        assert callable(MPAS2DProcessor.get_available_variables)


class TestMPAS3DProcessor:
    """ Test 3D data processor functionality and API surface. """
    
    def test_import_3d_processor(self: "TestMPAS3DProcessor") -> None:
        """
        This test verifies that the `MPAS3DProcessor` class can be imported successfully from the `processors_3d` module. Successful import indicates that the module is correctly structured, dependencies are satisfied, and there are no syntax errors preventing the class from being defined. This is a fundamental test to ensure that subsequent tests can run against the processor class. 

        Parameters:
            self ("TestMPAS3DProcessor"): Test instance (unused).

        Returns:
            None: Assertion validates importability.
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor

        public_methods = [method for method in dir(MPAS3DProcessor) if not method.startswith('_')]

        expected_methods = [
            'find_mpasout_files',
            'load_3d_data',
            'get_3d_variable_data',
            'get_vertical_levels',
            'extract_2d_coordinates_for_variable'
        ]

        for method in expected_methods:
            assert method in public_methods, f"Expected method '{method}' not found in MPAS3DProcessor"
    
    def test_3d_processor_has_base_class(self: "TestMPAS3DProcessor") -> None:
        """
        This test checks that the `MPAS3DProcessor` class is a subclass of `MPASBaseProcessor`, ensuring it inherits common functionality and adheres to the expected interface for processors in MPASdiag. Subclassing the base processor allows for consistent behavior across different processor types and enables shared utilities to operate on either 2D or 3D processors. 

        Parameters:
            self ("TestMPAS3DProcessor"): Test instance (unused).

        Returns:
            None: Assertion confirms subclass relationship.
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        from mpasdiag.processing.base import MPASBaseProcessor
        assert issubclass(MPAS3DProcessor, MPASBaseProcessor)
    
    def test_3d_processor_has_init_method(self: "TestMPAS3DProcessor") -> None:
        """
        This test inspects the constructor of the `MPAS3DProcessor` class to confirm that it accepts expected parameters such as `grid_file` and `verbose`. The presence of these parameters in the constructor signature is important for users who need to initialize the processor with specific grid information and control verbosity. By using the `inspect` module, the test checks that these parameters are defined without actually invoking the constructor, thus avoiding any side effects or file I/O during testing. 

        Parameters:
            self ("TestMPAS3DProcessor"): Test instance (unused).

        Returns:
            None: Assertion inspects constructor signature.
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        import inspect
        sig = inspect.signature(MPAS3DProcessor.__init__)
        params = list(sig.parameters.keys())
        assert 'grid_file' in params
        assert 'verbose' in params
    
    def test_3d_processor_has_find_mpasout_files_method(self: "TestMPAS3DProcessor") -> None:
        """
        This test verifies that the `MPAS3DProcessor` class has a method named `find_mpasout_files`, which is essential for locating MPAS output files needed for processing 3D diagnostics. The presence of this method is crucial for the processor to function correctly, as it allows users to specify how MPAS output files are discovered. The test checks both the existence of the method and that it is callable, ensuring that the API is intact and ready for use by higher-level routines without performing any actual file I/O. 

        Parameters:
            self ("TestMPAS3DProcessor"): Test instance (unused).

        Returns:
            None: Assertion validates method availability.
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        assert hasattr(MPAS3DProcessor, 'find_mpasout_files')
        assert callable(MPAS3DProcessor.find_mpasout_files)
    
    def test_3d_processor_has_load_3d_data_method(self: "TestMPAS3DProcessor") -> None:
        """
        This test confirms that the `MPAS3DProcessor` class exposes a method named `load_3d_data`, which is responsible for reading 3D fields from MPAS output files. The presence of this method is essential for the processor to function as it allows users to access variable data needed for plotting and diagnostics. The test checks that the method exists and is callable, ensuring that the API is available for use by higher-level routines without performing any actual file I/O during testing. 

        Parameters:
            self ("TestMPAS3DProcessor"): Test instance (unused).

        Returns:
            None: Assertion validates method presence.
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        assert hasattr(MPAS3DProcessor, 'load_3d_data')
        assert callable(MPAS3DProcessor.load_3d_data)
    
    def test_3d_processor_has_get_3d_variable_data_method(self: "TestMPAS3DProcessor") -> None:
        """
        This test ensures that the `MPAS3DProcessor` class has a method named `get_3d_variable_data`, which is used to extract 3D variable data arrays at specific time and vertical indices. This method is essential for users who need to access the actual data values for plotting and diagnostics. The test checks that the method exists and is callable, ensuring that the API provides the necessary interface for data retrieval without performing any actual file I/O during testing. 

        Parameters:
            self ("TestMPAS3DProcessor"): Test instance (unused).

        Returns:
            None: Assertion validates method availability.
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        assert hasattr(MPAS3DProcessor, 'get_3d_variable_data')
        assert callable(MPAS3DProcessor.get_3d_variable_data)
    
    def test_3d_processor_has_get_vertical_levels_method(self: "TestMPAS3DProcessor") -> None:
        """
        This test checks that the `MPAS3DProcessor` class has a method named `get_vertical_levels`, which is responsible for retrieving the vertical level information (e.g., pressure levels, height levels) associated with 3D variables. This functionality is crucial for users who need to understand the vertical structure of the data for remapping and vertical plotting. The test verifies that the method exists and is callable, ensuring that the API provides the necessary interface for accessing vertical level information without performing any actual data manipulation during testing. 

        Parameters:
            self ("TestMPAS3DProcessor"): Test instance (unused).

        Returns:
            None: Assertion confirms method presence.
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        assert hasattr(MPAS3DProcessor, 'get_vertical_levels')
        assert callable(MPAS3DProcessor.get_vertical_levels)
    
    def test_3d_processor_has_extract_2d_coordinates_method(self: "TestMPAS3DProcessor") -> None:
        """
        This test verifies that the `MPAS3DProcessor` class has a method named `extract_2d_coordinates_for_variable`, which is responsible for extracting 2D coordinate arrays (longitude and latitude) for a given variable. This functionality is crucial for remapping and horizontal plotting of 3D-derived diagnostics. The test checks that the method exists and is callable, ensuring that the API provides the necessary interface for coordinate extraction without performing any actual data manipulation during testing. 

        Parameters:
            self ("TestMPAS3DProcessor"): Test instance (unused).

        Returns:
            None: Assertion verifies method presence.
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        assert hasattr(MPAS3DProcessor, 'extract_2d_coordinates_for_variable')
        assert callable(MPAS3DProcessor.extract_2d_coordinates_for_variable)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

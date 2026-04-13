#!/usr/bin/env python3
"""
MPASdiag Test Suite: Tests for MPAS2DProcessor and MPAS3DProcessor API surface

This module contains unit tests for the MPASdiag processing parallel wrappers, specifically focusing on the `ParallelPrecipitationProcessor`, `ParallelSurfaceProcessor`, `ParallelWindProcessor`, and `ParallelCrossSectionProcessor` classes. The tests cover the functionality of batch processing methods that utilize parallel execution to generate plots and maps from MPAS model output. Additionally, tests for the `_process_parallel_results` function are included to ensure that it correctly summarizes the outcomes of parallel tasks. The test suite uses pytest fixtures to set up temporary directories and mock objects, allowing for isolated testing of the processing logic without relying on actual data files or parallel execution environments. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules for testing
import inspect
import pytest
import xesmf as xe

if xe is not None:
    XESMF_AVAILABLE = True
else:
    XESMF_AVAILABLE = False

REMAPPING_AVAILABLE = True


class TestMPAS2DProcessor:
    """ Test 2D data processor class hierarchy, constructor, and API surface. """

    def test_2d_processor_class_hierarchy_and_constructor(self: "TestMPAS2DProcessor") -> None:
        """
        This test verifies that MPAS2DProcessor is a subclass of MPASBaseProcessor and that its constructor accepts both `grid_file` and `verbose` parameters. 

        Parameters:
            self ("TestMPAS2DProcessor"): Test instance.

        Returns:
            None
        """
        from mpasdiag.processing.processors_2d import MPAS2DProcessor
        from mpasdiag.processing.base import MPASBaseProcessor

        assert issubclass(MPAS2DProcessor, MPASBaseProcessor), \
            "MPAS2DProcessor must inherit from MPASBaseProcessor"

        sig = inspect.signature(MPAS2DProcessor.__init__)
        params = list(sig.parameters.keys())
        assert 'grid_file' in params, "__init__ must accept 'grid_file'"
        assert 'verbose' in params, "__init__ must accept 'verbose'"

    def test_2d_processor_required_methods(self: "TestMPAS2DProcessor") -> None:
        """
        This test verifies that all required public methods exist on MPAS2DProcessor and are callable.

        Parameters:
            self ("TestMPAS2DProcessor"): Test instance.

        Returns:
            None
        """
        from mpasdiag.processing.processors_2d import MPAS2DProcessor

        required_methods = [
            'find_diagnostic_files',
            'load_2d_data',
            'add_spatial_coordinates',
            'extract_2d_coordinates_for_variable',
            'get_2d_variable_data',
            'get_available_variables',
        ]
        for method in required_methods:
            assert hasattr(MPAS2DProcessor, method), \
                f"MPAS2DProcessor is missing required method '{method}'"
            assert callable(getattr(MPAS2DProcessor, method)), \
                f"MPAS2DProcessor.{method} must be callable"


class TestMPAS3DProcessor:
    """ Test 3D data processor class hierarchy, constructor, and API surface. """

    def test_3d_processor_class_hierarchy_and_constructor(self: "TestMPAS3DProcessor") -> None:
        """
        This test verifies that MPAS3DProcessor is a subclass of MPASBaseProcessor and that its constructor accepts both `grid_file` and `verbose` parameters. It checks the class hierarchy to ensure that MPAS3DProcessor inherits from the base processor class, which provides common functionality for all processors. Additionally, it inspects the constructor signature to confirm that it can accept the necessary parameters for initializing the processor with a grid file and verbosity settings. 

        Parameters:
            self ("TestMPAS3DProcessor"): Test instance.

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        from mpasdiag.processing.base import MPASBaseProcessor

        assert issubclass(MPAS3DProcessor, MPASBaseProcessor), \
            "MPAS3DProcessor must inherit from MPASBaseProcessor"

        sig = inspect.signature(MPAS3DProcessor.__init__)
        params = list(sig.parameters.keys())
        assert 'grid_file' in params, "__init__ must accept 'grid_file'"
        assert 'verbose' in params, "__init__ must accept 'verbose'"

    def test_3d_processor_required_methods(self: "TestMPAS3DProcessor") -> None:
        """
        This test verifies that all required public methods exist on MPAS3DProcessor and are callable. It checks for the presence of methods related to finding files, loading data, extracting variable data, and handling vertical levels, which are essential for processing 3D MPAS model output.

        Parameters:
            self ("TestMPAS3DProcessor"): Test instance.

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor

        required_methods = [
            'find_mpasout_files',
            'load_3d_data',
            'get_3d_variable_data',
            'get_vertical_levels',
            'extract_2d_coordinates_for_variable',
        ]
        for method in required_methods:
            assert hasattr(MPAS3DProcessor, method), \
                f"MPAS3DProcessor is missing required method '{method}'"
            assert callable(getattr(MPAS3DProcessor, method)), \
                f"MPAS3DProcessor.{method} must be callable"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

#!/usr/bin/env python3

"""
MPASdiag Test Suite: Module Imports and Basic Functionality

This test suite module focuses on verifying that key modules within the `mpasdiag` package can be imported successfully and that basic attributes and structures are present. The tests cover CLI modules, processing modules, visualization modules, constants, and simple diagnostic class attributes. These tests serve as a sanity check to ensure that the package structure is intact and that critical components are available for use in more complex tests and real-world usage.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries 
import pytest


class TestCLIImports:
    """ Test that CLI modules can be imported and expose expected entrypoints. """
    
    def test_import_cli_main(self: "TestCLIImports") -> None:
        """
        This test verifies that the main CLI module (`mpasdiag.cli`) can be imported and that it provides a `main` function, which is the expected entrypoint for command-line execution. This ensures that the CLI component of the package is available and properly structured for use in scripts and command-line contexts.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag import cli
        assert hasattr(cli, 'main')
    
    def test_import_cli_unified(self: "TestCLIImports") -> None:
        """
        This test checks that the `cli_unified` module within `mpasdiag.processing` can be imported and that it exposes the `MPASUnifiedCLI` class, which is expected to provide a unified command-line interface for processing tasks. This ensures that the unified CLI component is available and properly structured for use in command-line contexts.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing import cli_unified
        assert hasattr(cli_unified, 'MPASUnifiedCLI')


class TestProcessingImports:
    """ Test that processing modules can be imported and expose expected classes. """
    
    def test_import_base(self: "TestProcessingImports") -> None:
        """
        This test verifies that the base processing module (`mpasdiag.processing.base`) can be imported and that it exposes the `MPASBaseProcessor` class, which is expected to provide foundational processing utilities and structure for other processors. This ensures that the base processing component is available and properly structured for use in scripts and other modules.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing import base
        assert hasattr(base, 'MPASBaseProcessor')
    
    def test_import_processors_2d(self: "TestProcessingImports") -> None:
        """
        This test checks that the 2D processor module (`mpasdiag.processing.processors_2d`) can be imported and that it exposes the `MPAS2DProcessor` class, which is expected to provide processing utilities for 2D surface diagnostics. This ensures that the 2D processing component is available and properly structured for use in scripts and other modules.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing import processors_2d
        assert hasattr(processors_2d, 'MPAS2DProcessor')
    
    def test_import_processors_3d(self: "TestProcessingImports") -> None:
        """
        This test verifies that the 3D processor module (`mpasdiag.processing.processors_3d`) can be imported and that it exposes the `MPAS3DProcessor` class, which is expected to provide processing utilities for volumetric diagnostics. This ensures that the 3D processing component is available and properly structured for use in scripts and other modules.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing import processors_3d
        assert hasattr(processors_3d, 'MPAS3DProcessor')
    
    def test_import_remapping(self: "TestProcessingImports") -> None:
        """
        This test ensures that the remapping utilities module (`mpasdiag.processing.remapping`) can be imported and that it exposes the `MPASRemapper` class, which is expected to provide spatial remapping tools for processing tasks. This confirms that the remapping component is available and properly structured for use in scripts and other modules.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing import remapping
        assert hasattr(remapping, 'MPASRemapper')
    
    def test_import_parallel(self: "TestProcessingImports") -> None:
        """
        This test verifies that the `parallel` module within `mpasdiag.processing` can be imported and is available for use. The test performs a minimal import and asserts the module object is not None, ensuring that parallel processing utilities are accessible.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing import parallel
        assert parallel is not None
    
    def test_import_parallel_wrappers(self: "TestProcessingImports") -> None:
        """
        This test ensures that the `parallel_wrappers` module within `mpasdiag.processing` can be imported and is available for use. The test performs a minimal import and asserts the module object is not None, confirming that helper wrappers for parallel execution are accessible.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing import parallel_wrappers
        assert parallel_wrappers is not None
    
    def test_import_data_cache(self: "TestProcessingImports") -> None:
        """
        This test confirms that the `data_cache` module within `mpasdiag.processing` can be imported and is available for caching processing results. The test performs a minimal import and asserts the module object is defined, ensuring that caching utilities are accessible.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing import data_cache
        assert data_cache is not None


class TestUtilityModules:
    """ Test utility modules can be imported and expose expected functions or classes. """
    
    def test_import_utils_config(self: "TestUtilityModules") -> None:
        """
        This test ensures that the `utils_config` module within `mpasdiag.processing` can be imported and is available for use. The test performs a minimal import and asserts the module object is not None, confirming that configuration utilities are accessible.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing import utils_config
        assert utils_config is not None
    
    def test_import_utils_logger(self: "TestUtilityModules") -> None:
        """
        This test ensures that the `utils_logger` module within `mpasdiag.processing` can be imported and is available for use. The test performs a minimal import and asserts the module object is not None, confirming that logging utilities are accessible.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing import utils_logger
        assert utils_logger is not None
    
    def test_import_utils_file(self: "TestUtilityModules") -> None:
        """
        This test ensures that the `utils_file` module within `mpasdiag.processing` can be imported and is available for use. The test performs a minimal import and asserts the module object is not None, confirming that filesystem helpers are accessible.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing import utils_file
        assert utils_file is not None
    
    def test_import_utils_unit(self: "TestUtilityModules") -> None:
        """
        This test ensures that the `utils_unit` module within `mpasdiag.processing` can be imported and is available for use. The test performs a minimal import and asserts the module object is not None, confirming that unit conversion utilities and constants are accessible.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing import utils_unit
        assert utils_unit is not None
    
    def test_import_utils_geog(self: "TestUtilityModules") -> None:
        """
        This test ensures that the `utils_geog` module within `mpasdiag.processing` can be imported and is available for use. The test performs a minimal import and asserts the module object is not None, confirming that geographic utilities for coordinate handling and spatial queries are accessible.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing import utils_geog
        assert utils_geog is not None
    
    def test_import_utils_parser(self: "TestUtilityModules") -> None:
        """
        This test ensures that the `utils_parser` module within `mpasdiag.processing` can be imported and is available for use. The test performs a minimal import and asserts the module object is not None, confirming that argument parsing helpers needed by CLI components are accessible.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing import utils_parser
        assert utils_parser is not None
    
    def test_import_utils_validator(self: "TestUtilityModules") -> None:
        """
        This test ensures that the `utils_validator` module within `mpasdiag.processing` can be imported and is available for use. The test performs a minimal import and asserts the module object is not None, confirming that input validation helpers used by processors and visualizers are accessible.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing import utils_validator
        assert utils_validator is not None


class TestVisualizationImports:
    """ Test visualization modules can be imported and expose expected classes or functions. """
    
    def test_import_base_visualizer(self: "TestVisualizationImports") -> None:
        """
        This test ensures that the `base_visualizer` module within `mpasdiag.visualization` can be imported and is available for use. The test performs a minimal import and asserts the module object is not None, confirming that the base visualizer utilities are accessible.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.visualization import base_visualizer
        assert hasattr(base_visualizer, 'MPASVisualizer')
    
    def test_import_surface(self: "TestVisualizationImports") -> None:
        """
        This test ensures that the `surface` module within `mpasdiag.visualization` can be imported and is available for use. The test performs a minimal import and asserts the module object is not None, confirming that the 2D surface plotting utilities are accessible.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.visualization import surface
        assert hasattr(surface, 'MPASSurfacePlotter')
    
    def test_import_cross_section(self: "TestVisualizationImports") -> None:
        """
        This test ensures that the `cross_section` module within `mpasdiag.visualization` can be imported and is available for use. The test performs a minimal import and asserts the module object is not None, confirming that vertical cross-section plotting utilities are accessible.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.visualization import cross_section
        assert hasattr(cross_section, 'MPASVerticalCrossSectionPlotter')
    
    def test_import_wind_viz(self: "TestVisualizationImports") -> None:
        """
        This test ensures that the `wind` module within `mpasdiag.visualization` can be imported and is available for use. The test performs a minimal import and asserts the module object is not None, confirming that wind visualization utilities are accessible.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.visualization import wind
        assert hasattr(wind, 'MPASWindPlotter')
    
    def test_import_precipitation_viz(self: "TestVisualizationImports") -> None:
        """
        This test ensures that the `precipitation` module within `mpasdiag.visualization` can be imported and is available for use. The test performs a minimal import and asserts the module object is not None, confirming that precipitation visualization utilities are accessible.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.visualization import precipitation
        assert hasattr(precipitation, 'MPASPrecipitationPlotter')
    
    def test_import_styling(self: "TestVisualizationImports") -> None:
        """
        This test ensures that the `styling` module within `mpasdiag.visualization` can be imported and is available for use. The test performs a minimal import and asserts the module object is not None, confirming that plot styling utilities are accessible.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.visualization import styling
        assert styling is not None


class TestConstants:
    """ This class verifies that the `constants` module can be imported and that key physical and unit constants are defined. """
    
    def test_import_constants(self: "TestConstants") -> None:
        """
        This test ensures that the `constants` module within `mpasdiag.processing` can be imported and is available for use. The test performs a minimal import and asserts the module object is not None, confirming that physical and unit constants are accessible.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing import constants
        assert constants is not None
    
    def test_physical_constants_exist(self: "TestConstants") -> None:
        """
        This test verifies that key physical constants such as `MM` (millimeters), `KELVIN` (absolute temperature), and `M_PER_S` (meters per second) are defined in the `constants` module. These constants are commonly used in processing and visualization code, and their presence ensures that dependent code can rely on consistent values for these fundamental parameters.

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.constants as const
        assert hasattr(const, 'MM')
        assert hasattr(const, 'KELVIN')
        assert hasattr(const, 'M_PER_S')
    
    def test_unit_conversions_exist(self: "TestConstants") -> None:
        """
        This test verifies that key unit conversion constants such as `KILOMETER` and `METER` are defined in the `constants` module and have the expected literal values. These constants are commonly used in processing and visualization code, and their presence ensures that dependent code can rely on consistent values for these fundamental units.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.constants import KILOMETER, METER
        assert KILOMETER == 'km'
        assert METER == 'm'


class TestDiagnosticBasics:
    """ This class tests basic attributes of diagnostic classes, such as the presence and behavior of a `verbose` flag in the `WindDiagnostics` and `PrecipitationDiagnostics` classes. """
    
    def test_wind_diagnostics_verbose_flag(self: "TestDiagnosticBasics") -> None:
        """
        This test confirms that the `WindDiagnostics` class respects the `verbose` constructor argument used to enable or disable diagnostic verbosity. The test constructs both verbose and quiet instances and asserts the `verbose` attribute matches the provided flag. The presence of this flag allows users to control the amount of diagnostic output produced during processing, which can be helpful for debugging or reducing noise in normal operation.

        Parameters:
                None

            Returns:
                None
            """
        from mpasdiag.diagnostics.wind import WindDiagnostics
        
        diag_verbose = WindDiagnostics(verbose=True)
        diag_quiet = WindDiagnostics(verbose=False)
        
        assert diag_verbose.verbose is True
        assert diag_quiet.verbose is False
    
    def test_precipitation_diagnostics_verbose_flag(self: "TestDiagnosticBasics") -> None:
        """
        This test confirms that the `PrecipitationDiagnostics` class respects the `verbose` constructor argument used to enable or disable diagnostic verbosity. The test constructs both verbose and quiet instances and asserts the `verbose` attribute matches the provided flag. The presence of this flag allows users to control the amount of diagnostic output produced during processing, which can be helpful for debugging or reducing noise in normal operation.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
        
        diag_verbose = PrecipitationDiagnostics(verbose=True)
        diag_quiet = PrecipitationDiagnostics(verbose=False)
        
        assert diag_verbose.verbose is True
        assert diag_quiet.verbose is False


class TestPackageStructure:
    """ This class verifies that the top-level `mpasdiag` package can be imported and that core subpackages are accessible. """
    
    def test_package_import(self: "TestPackageStructure") -> None:
        """
        This test verifies that the top-level `mpasdiag` package can be imported successfully. The test performs a minimal import and asserts that the package object is not None, confirming that the package is available for use and that the basic import structure is intact.

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag
        assert mpasdiag is not None
    
    def test_version_exists(self: "TestPackageStructure") -> None:
        """
        This test checks that the `__version__` attribute is defined on the main `mpasdiag` package object. The presence of a version attribute is important for users to identify the version of the package they are using, and it is commonly used in logging, error messages, and when reporting issues. This test ensures that the version information is properly defined and accessible.

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag
        assert hasattr(mpasdiag, '__version__')
    
    def test_subpackages_accessible(self: "TestPackageStructure") -> None:
        """
        This test verifies that the core subpackages `processing`, `visualization`, and `diagnostics` are accessible as attributes of the main `mpasdiag` package. This ensures that users can access these key components directly from the top-level package, confirming that the package structure is properly defined and that subpackage imports are working as expected.

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag
        assert hasattr(mpasdiag, 'processing')
        assert hasattr(mpasdiag, 'visualization')
        assert hasattr(mpasdiag, 'diagnostics')


class TestUtilityFunctions:
    """ This class tests utility functions and classes provided by the `utils_logger` module, specifically the `MPASLogger` class. """
    
    def test_setup_logger(self: "TestUtilityFunctions") -> None:
        """
        This test verifies the creation of a `MPASLogger` instance and checks that it exposes standard logging methods (`info`, `warning`, `error`). This confirms that the logging helper constructs a usable logger for downstream code.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.utils_logger import MPASLogger

        logger = MPASLogger('test_logger', verbose=False)
        assert isinstance(logger, MPASLogger)
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'error')

    def test_setup_logger_verbose(self: "TestUtilityFunctions") -> None:
        """
        This test verifies that the `MPASLogger` class respects the `verbose` constructor argument used to enable or disable verbose logging. The test constructs both verbose and quiet logger instances and asserts that the `verbose` attribute matches the provided flag. This ensures that users can control the verbosity of logging output when using the `MPASLogger` utility.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.utils_logger import MPASLogger
        
        logger_verbose = MPASLogger('test_verbose', verbose=True)
        logger_quiet = MPASLogger('test_quiet', verbose=False)
        assert logger_verbose.verbose is True
        assert logger_quiet.verbose is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

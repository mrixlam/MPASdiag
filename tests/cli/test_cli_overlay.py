#!/usr/bin/env python3
"""
MPASdiag Test Suite: CLI Overlay Analysis Tests

This module contains tests for the overlay analysis functionality of the MPASdiag CLI. It verifies that different overlay types (e.g., 'precip_wind', 'temp_pressure') are handled correctly by the `run_analysis` method and that the expected results are produced when sample data are available. The tests also check that custom overlay types fall back to default handling and that the overlay analysis can be executed with an attached logger.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load pytest for testing and os for file path checks
import os
import pytest
from pathlib import Path

class TestOverlayAndCompositeAnalysis:
    """ Test overlay and composite analysis paths in the CLI. """
    
    def test_overlay_analysis_without_type(self: "TestOverlayAndCompositeAnalysis", grid_file: str, test_data_dir: str) -> None:
        """
        This test verifies that the overlay analysis path can be executed without specifying a special overlay type, ensuring that the default handling is correct. It constructs an `MPASConfig` with an `overlay_type` that does not match the special cases and checks that `run_analysis` returns `True` when sample data are available. The test is skipped if the required sample data files are not present in the test environment to avoid false failures.

        Parameters:
            self (TestOverlayAndCompositeAnalysis): The test instance.
            grid_file: Session fixture providing path to MPAS grid file.
            test_data_dir: Session fixture providing path to test data directory.

        Returns:
            None: The test asserts `run_analysis` returns `True` on success.
        """
        import pytest
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        if grid_file is None:
            pytest.skip("Test data files not available")
        
        data_dir = str(Path(test_data_dir) / 'u120k' / 'diag')
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file=grid_file,
            data_dir=data_dir,
            analysis_type='overlay',
            overlay_type='other_type',  
            time_index=0,
            output_dir='output/test_overlay_other',
            verbose=True
        )
        
        result = cli.run_analysis(config)
        
        assert result is True

    def test_run_overlay_analysis_precip_wind(self: "TestOverlayAndCompositeAnalysis", grid_file: str, test_data_dir: str) -> None:
        """
        This test verifies that the 'precip_wind' overlay type is correctly processed by the `run_analysis` method. It constructs an `MPASConfig` with `overlay_type='precip_wind'` and checks that the analysis runs successfully, returning `True`. The test is skipped if the required sample data files are not available to ensure that it only fails when there is an actual issue with the code rather than missing data.

        Parameters:
            self (TestOverlayAndCompositeAnalysis): The test instance.
            grid_file: Session fixture providing path to MPAS grid file.
            test_data_dir: Session fixture providing path to test data directory.

        Returns:
            None: The test asserts the boolean result returned by `run_analysis`.
        """
        import pytest
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        if grid_file is None:
            pytest.skip("Test data files not available")
        
        data_dir = str(Path(test_data_dir) / 'u120k' / 'diag')
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file=grid_file,
            data_dir=data_dir,
            analysis_type='overlay',
            overlay_type='precip_wind',
            time_index=0,
            output_dir='output/test_overlay_precip_wind',
            verbose=True  
        )
        
        result = cli.run_analysis(config)
        
        assert result is True
    
    def test_run_overlay_analysis_temp_pressure(self: "TestOverlayAndCompositeAnalysis", grid_file: str, test_data_dir: str) -> None:
        """
        This test verifies that the 'temp_pressure' overlay type is correctly processed by the `run_analysis` method. It constructs an `MPASConfig` with `overlay_type='temp_pressure'` and checks that the analysis runs successfully, returning `True`. The test is skipped if the required sample data files are not available to ensure that it only fails when there is an actual issue with the code rather than missing data.

        Parameters:
            self (TestOverlayAndCompositeAnalysis): The test instance.
            grid_file: Session fixture providing path to MPAS grid file.
            test_data_dir: Session fixture providing path to test data directory.

        Returns:
            None: Assertions validate successful completion of the analysis.
        """
        import pytest
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        if grid_file is None:
            pytest.skip("Test data files not available")
        
        data_dir = str(Path(test_data_dir) / 'u120k' / 'diag')
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file=grid_file,
            data_dir=data_dir,
            analysis_type='overlay',
            overlay_type='temp_pressure',
            time_index=0,
            output_dir='output/test_overlay_temp_pressure',
            verbose=True
        )
        
        result = cli.run_analysis(config)
        
        assert result is True


class TestOverlayTypePaths:
    """ Test different overlay type paths in _run_overlay_analysis. """
    
    def test_overlay_other_type_fallthrough(self: "TestOverlayTypePaths") -> None:
        """
        This test verifies that when an `overlay_type` is provided that does not match the special cases (e.g., 'precip_wind' or 'temp_pressure'), the `run_analysis` method still completes successfully. It constructs an `MPASConfig` with a custom `overlay_type` and checks that `run_analysis` returns `True`. The test is skipped if the required sample data files are not available to ensure that it only fails when there is an actual issue with the code rather than missing data.

        Parameters:
            self (TestOverlayTypePaths): The test instance.

        Returns:
            None: The test asserts that `run_analysis` returns True when
            the overlay fallback path completes successfully.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        
        if not os.path.exists('data/grids/x1.40962.init.nc'):
            pytest.skip("Test data not available")
        
        cli = MPASUnifiedCLI()

        config = MPASConfig(
            grid_file='data/grids/x1.40962.init.nc',
            data_dir='data/u120k/diag',
            analysis_type='overlay',
            overlay_type='custom_type', 
            time_index=0,
            output_dir='output/test_overlay_other',
            verbose=True
        )
        
        result = cli.run_analysis(config)
        assert result is True


class TestOverlayWithLogger:
    """ Test overlay analysis with logger. """
    
    def test_overlay_precip_wind_with_logger(self: "TestOverlayWithLogger") -> None:
        """
        This test verifies that the 'precip_wind' overlay type can be executed successfully when an `MPASLogger` is attached to the CLI. It constructs an `MPASConfig` for the 'precip_wind' overlay type, attaches a logger, and checks that `run_analysis` returns `True`. The test is skipped if the required sample data files are not available to ensure that it only fails when there is an actual issue with the code rather than missing data.

        Parameters:
            self (TestOverlayWithLogger): The test instance.

        Returns:
            None: The test asserts that `run_analysis` returns True when
            the overlay workflow executes successfully with data present.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        from mpasdiag.processing.utils_logger import MPASLogger
        
        if not os.path.exists('data/grids/x1.40962.init.nc'):
            pytest.skip("Test data not available")
        
        cli = MPASUnifiedCLI()
        cli.logger = MPASLogger('test', verbose=True)
        
        config = MPASConfig(
            grid_file='data/grids/x1.40962.init.nc',
            data_dir='data/u120k/diag',
            analysis_type='overlay',
            overlay_type='precip_wind',
            time_index=0,
            output_dir='output/test_overlay_precip_wind_logger',
            verbose=True
        )
        
        result = cli.run_analysis(config)
        assert result is True
    
    def test_overlay_temp_pressure_with_logger(self: "TestOverlayWithLogger") -> None:
        """
        This test verifies that the 'temp_pressure' overlay type can be executed successfully when an `MPASLogger` is attached to the CLI. It constructs an `MPASConfig` for the 'temp_pressure' overlay type, attaches a logger, and checks that `run_analysis` returns `True`. The test is skipped if the required sample data files are not available to ensure that it only fails when there is an actual issue with the code rather than missing data.

        Parameters:
            self (TestOverlayWithLogger): The test instance.

        Returns:
            None: The test asserts that `run_analysis` returns True when
            data exist and the overlay completes successfully.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        from mpasdiag.processing.utils_logger import MPASLogger
        
        if not os.path.exists('data/grids/x1.40962.init.nc'):
            pytest.skip("Test data not available")
        
        cli = MPASUnifiedCLI()
        cli.logger = MPASLogger('test', verbose=True)
        
        config = MPASConfig(
            grid_file='data/grids/x1.40962.init.nc',
            data_dir='data/u120k/diag',
            analysis_type='overlay',
            overlay_type='temp_pressure',
            time_index=0,
            output_dir='output/test_overlay_temp_pressure_logger',
            verbose=True
        )
        
        result = cli.run_analysis(config)
        assert result is True
    
    def test_overlay_other_type_with_logger(self: "TestOverlayWithLogger") -> None:
        """
        This test verifies that a custom `overlay_type` can be executed successfully when an `MPASLogger` is attached to the CLI. It constructs an `MPASConfig` with a custom `overlay_type`, attaches a logger, and checks that `run_analysis` returns `True`. The test is skipped if the required sample data files are not available to ensure that it only fails when there is an actual issue with the code rather than missing data.

        Parameters:
            self (TestOverlayWithLogger): The test instance.

        Returns:
            None: The test asserts that `run_analysis` returns True for
            custom overlay types when sample data are available.
        """
        from mpasdiag.processing.cli_unified import MPASUnifiedCLI
        from mpasdiag.processing.utils_config import MPASConfig
        from mpasdiag.processing.utils_logger import MPASLogger
        
        if not os.path.exists('data/grids/x1.40962.init.nc'):
            pytest.skip("Test data not available")
        
        cli = MPASUnifiedCLI()
        cli.logger = MPASLogger('test', verbose=True)
        
        config = MPASConfig(
            grid_file='data/grids/x1.40962.init.nc',
            data_dir='data/u120k/diag',
            analysis_type='overlay',
            overlay_type='custom_overlay', 
            time_index=0,
            output_dir='output/test_overlay_custom_logger',
            verbose=True
        )
        
        result = cli.run_analysis(config)
        assert result is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

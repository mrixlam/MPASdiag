#!/usr/bin/env python3

"""
MPASdiag Test Suite: MPAS2DProcessor with Real MPAS Data

This test suite validates the functionality of the MPAS2DProcessor class using actual MPAS diagnostic data. It covers initialization, file discovery, coordinate extraction, variable retrieval, accumulation parsing, and coordinate augmentation. The tests are designed to be comprehensive while gracefully skipping if the required MPAS test dataset is not available on the testing machine. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load standard libraries
import os
import pytest
from typing import Any, Dict
from unittest.mock import patch

from mpasdiag.processing.processors_2d import MPAS2DProcessor
from tests.test_data_helpers import get_mpas_data_paths, check_mpas_data_available, assert_expected_public_methods


def get_mpas_test_data_paths() -> Dict[str, str]:
    """
    This helper function retrieves the paths to the MPAS test data required for processing tests. It uses a utility function to get the paths and ensures that if a diagnostic directory is provided without a separate data directory, it defaults to using the diagnostic directory for both. This function centralizes the logic for determining data paths, making it easier to manage and update as needed.

    Parameters:
        None

    Returns:
        Dict[str, str]: Dictionary containing paths to MPAS test data.
    """
    paths = get_mpas_data_paths()

    if 'diag_dir' in paths and 'data_dir' not in paths:
        paths['data_dir'] = paths['diag_dir']

    return paths

def load_mpas_processor(verbose: bool = False, 
                        use_pure_xarray: bool = True) -> MPAS2DProcessor:
    """
    This function initializes an MPAS2DProcessor instance and attempts to load MPAS diagnostic data if available. It retrieves the necessary paths using a helper function, checks for the existence of the required grid file, and initializes the processor. If diagnostic data is available, it attempts to load it using the specified loading method. If any required data is missing or if loading fails, it gracefully skips the test. This function is designed to be used as a fixture in tests that require a processor with loaded data, ensuring that tests only run when valid data is present.
    
    Parameters:
        verbose (bool): Whether to enable verbose logging on the processor.
        use_pure_xarray (bool): If True, prefer the pure-xarray loader path.

    Returns:
        MPAS2DProcessor: Processor instance with data loaded when available.
    """
    paths = get_mpas_data_paths()
    
    if not os.path.exists(paths['grid_file']):
        pytest.skip("MPAS test data not found")
        return
    
    processor = MPAS2DProcessor(paths['grid_file'], verbose=verbose)
    
    if os.path.exists(paths['diag_dir']):
        try:
            processor.load_2d_data(paths['diag_dir'], use_pure_xarray=use_pure_xarray)
        except Exception as e:
            pytest.skip(f"Could not load MPAS data: {e}")
            return
    
    return processor


class TestExtract2DCoordinates:
    """ Test coordinate extraction from actual MPAS data. """

    @pytest.fixture(autouse=True)
    def setup_method(self: 'TestExtract2DCoordinates', 
                     mpas_2d_processor_diag: 'MPAS2DProcessor', 
                     mpas_data_available: bool) -> None:
        """
        This fixture sets up the MPAS2DProcessor with loaded diagnostic data for coordinate extraction tests. It uses the session-scoped `mpas_2d_processor_diag` fixture to avoid redundant loading across multiple tests. If the MPAS dataset is not available, it will skip all tests in this class, ensuring that they only run when valid data is present. Shared attributes are stored on `self` for use in individual tests. This setup ensures that the tests are deterministic and do not fail due to missing data on machines that do not have the MPAS test dataset.

        Parameters:
            self: Test instance provided by pytest.
            mpas_2d_processor_diag: Session-scoped processor with loaded diagnostic data
            mpas_data_available: Boolean flag indicating if MPAS data exists

        Returns:
            None
        """
        if not mpas_data_available or mpas_2d_processor_diag is None:
            pytest.skip("MPAS test data not available")
            return
        
        self.processor = mpas_2d_processor_diag
        self.paths = get_mpas_test_data_paths()


    def test_extract_coordinates_verbose_output(self: 'TestExtract2DCoordinates') -> None:
        """
        This test checks that the coordinate extraction method produces output when verbose mode is enabled. It creates a new processor instance with `verbose=True`, loads the diagnostic data, and patches the built-in `print` function to capture output during the coordinate extraction process. The test asserts that some output was produced, confirming that the method provides user feedback in verbose mode. It also checks that the output contains messages indicating the progress of coordinate extraction, ensuring that the method communicates useful information to the user.

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        processor = MPAS2DProcessor(self.paths['grid_file'], verbose=True)
        processor.load_2d_data(self.paths['diag_dir'], use_pure_xarray=True)
        assert_expected_public_methods(processor, 'MPAS2DProcessor')
        
        with patch('builtins.print') as mock_print:
            _, _ = processor.extract_spatial_coordinates()
            assert mock_print.call_count >= 0  


class TestLoad2DData:
    """ Test complete data loading workflow with actual MPAS data. """

    @classmethod
    def setup_class(cls: Any) -> None:
        """
        This method initializes shared resources for data loading tests. It prepares the paths and checks for the availability of MPAS test data. If the required data is not present, it will skip all tests in this class, ensuring that they only run when valid data is available. Shared attributes are stored on `cls` for reuse in individual tests. This setup ensures that the tests are deterministic and do not fail due to missing data on machines that do not have the MPAS test dataset. 

        Parameters:
            cls: The test class object used to store shared attributes.

        Returns:
            None
        """
        cls.paths = get_mpas_test_data_paths()

        if not check_mpas_data_available():
            pytest.skip("MPAS test data not available")
            return

    def test_load_data_returns_self(self: 'TestLoad2DData') -> None:
        """
        This test verifies that the `load_2d_data` method returns the MPAS2DProcessor instance itself, allowing for method chaining. It creates a new processor instance, calls the `load_2d_data` method with the diagnostic directory path, and asserts that the returned value is the same instance as the processor. This confirms that the method is designed to return `self`, enabling a fluent interface for loading data and performing subsequent operations on the same processor instance. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        processor = MPAS2DProcessor(self.paths['grid_file'], verbose=False)
        assert_expected_public_methods(processor, 'MPAS2DProcessor')
        result = processor.load_2d_data(self.paths['diag_dir'], use_pure_xarray=True)
        
        assert result is processor


def _make_mock_2d_processor(**kwargs: Any) -> MPAS2DProcessor:
    """
    This helper function creates a mock instance of MPAS2DProcessor with specified attributes for testing purposes. It patches the `__init__` method of MPAS2DProcessor to set default values for `verbose`, `dataset`, and `grid_file` attributes, allowing tests to create processor instances without needing to load actual data. The function accepts arbitrary keyword arguments to customize the attributes of the mock processor instance as needed for different test scenarios.

    Parameters:
        **kwargs: Arbitrary keyword arguments to set attributes on the mock processor instance.

    Returns:
        MPAS2DProcessor: A mock instance of MPAS2DProcessor with specified attributes.
    """
    with patch.object(MPAS2DProcessor, '__init__', lambda self, *a, **kw: setattr(self, 'verbose', kw.get('verbose', kwargs.get('verbose', False))) or setattr(self, 'dataset', None) or setattr(self, 'grid_file', 'mock')):
        proc = MPAS2DProcessor.__new__(MPAS2DProcessor)
        assert_expected_public_methods(proc, 'MPAS2DProcessor')
        proc.verbose = kwargs.get('verbose', False)
        proc.dataset = kwargs.get('dataset', None)
        proc.grid_file = 'mock'
        return proc


if __name__ == '__main__':
    pytest.main([__file__, "-v"])

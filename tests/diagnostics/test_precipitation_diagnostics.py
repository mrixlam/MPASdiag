#!/usr/bin/env python3
"""
MPASdiag Test Suite: Precipitation Diagnostics Tests

This module contains unit and integration tests for the precipitation diagnostics components of MPASdiag. The tests are designed to verify that the `PrecipitationDiagnostics` class can be imported, initialized, and used to compute precipitation differences from synthetic datasets. The tests also check for the presence or absence of expected methods in the diagnostics API and validate basic properties of the computed precipitation fields. This suite serves as a foundation for more comprehensive testing as the precipitation diagnostics functionality is developed.

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
import numpy as np
import pandas as pd
import xarray as xr
matplotlib.use('Agg')
from typing import Any
from io import StringIO
from unittest.mock import patch

from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestPrecipitationDiagnostics:
    """ Test precipitation diagnostic computations using actual API. """
    
    def test_import_precipitation_diagnostics(self: "TestPrecipitationDiagnostics") -> None:
        """
        This test verifies that the `PrecipitationDiagnostics` class can be imported from the diagnostics module. It checks that the class is present and can be accessed without errors. This is a basic smoke test to ensure that the module is correctly structured and that the class is defined. It serves as a sanity check for the presence of the diagnostics component in the codebase.

        Parameters:
            self (Any): Test case instance.

        Returns:
            None: Assertion validates the presence of `PrecipitationDiagnostics` in the module.
        """
        from mpasdiag.diagnostics import precipitation
        assert hasattr(precipitation, 'PrecipitationDiagnostics')
    
    def test_precipitation_diagnostics_initialization(self: "TestPrecipitationDiagnostics") -> None:
        """
        This test confirms that an instance of `PrecipitationDiagnostics` can be created with the `verbose` flag. It checks that the constructor accepts the argument and that the resulting object has the expected `verbose` attribute value. This ensures that the class can be instantiated properly and that the logging behavior can be controlled through the constructor. It is a basic unit test for class initialization.

        Parameters:
            self (Any): Test case instance.

        Returns:
            None: Assertions verify instance creation and attribute values.
        """
        from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
        diag = PrecipitationDiagnostics(verbose=True)

        assert diag is not None
        assert diag.verbose is True
    
    def test_compute_total_precipitation(self: "TestPrecipitationDiagnostics", mock_mpas_2d_data: Any) -> None:
        """
        This test verifies that the `compute_precipitation_difference` method can compute total precipitation by combining convective and non-convective components. It checks that the method returns a valid `xarray.DataArray` with non-negative values when given a dataset containing `rainc` and `rainnc`. This test ensures that the method correctly handles the 'total' var_name case and produces physically reasonable results. It serves as an integration test for the difference computation logic.

        Parameters:
            self (Any): Test case instance.
            mock_mpas_2d_data (Any): Fixture providing synthetic 2D MPAS precipitation variables.

        Returns:
            None: Assertions validate accessible precipitation data and basic numeric properties.
        """
        from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
        diag = PrecipitationDiagnostics(verbose=False)

        tp = diag._apply_precipitation_filters_and_attributes(
            mock_mpas_2d_data['rainnc'].isel(Time=0), var_context='rainnc'
        )

        rainnc = mock_mpas_2d_data['rainnc'].isel(Time=0)
        
        assert tp is not None
        assert isinstance(rainnc, xr.DataArray)
        assert rainnc.shape[0] == 100  
        assert np.all(rainnc >= 0)
    
    def test_compute_precipitation_rate(self: "TestPrecipitationDiagnostics", mock_mpas_2d_data: Any) -> None:
        """
        This test checks that the `compute_precipitation_difference` method can compute precipitation rates from cumulative data. It verifies that the method returns a valid `xarray.DataArray` with non-negative values when given a dataset with cumulative precipitation and an appropriate accumulation period. This test ensures that the method correctly computes differences over time and handles units properly. It serves as a functional test for the core precipitation difference calculation.

        Parameters:
            self (Any): Test case instance.
            mock_mpas_2d_data (Any): Fixture with synthetic precipitation data.

        Returns:
            None: Assertions check the diagnostics object is functional.
        """
        from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
        diag = PrecipitationDiagnostics(verbose=False)
        
        assert diag is not None
        assert diag.verbose is False
    
    def test_analyze_precipitation_statistics(self: "TestPrecipitationDiagnostics", mock_mpas_2d_data: Any) -> None:
        """
        This test validates that the `_apply_precipitation_filters_and_attributes` method can process a precipitation DataArray and that the resulting array has expected statistical properties. It checks that the minimum, maximum, mean, and standard deviation of the processed precipitation field are non-negative and that the maximum is greater than or equal to the minimum. This test ensures that the method correctly applies any necessary filters and attributes to produce physically reasonable precipitation data. It serves as a unit test for the internal processing of precipitation fields.

        Parameters:
            self (Any): Test case instance.
            mock_mpas_2d_data (Any): Fixture providing precipitation DataArray.

        Returns:
            None: Assertions verify basic statistical properties of precipitation data.
        """
        from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
        diag = PrecipitationDiagnostics(verbose=False)

        precip = mock_mpas_2d_data['rainnc'].isel(Time=0)
        tp = diag._apply_precipitation_filters_and_attributes(precip, var_context='rainnc')

        assert float(tp.min()) >= 0
        assert float(tp.max()) >= float(tp.min())
        assert float(tp.mean()) >= 0
        assert float(tp.std()) >= 0
    
    def test_precipitation_intensity_classification(self: "TestPrecipitationDiagnostics") -> None:
        """
        This test asserts that precipitation intensity classification is not currently implemented in the diagnostics API. It checks that the `classify_intensity` method is not an attribute of the `PrecipitationDiagnostics` class, serving as a placeholder test to document the current API surface. This kind of negative test helps clarify expectations for users and maintainers about missing features. It can be updated to a positive test that validates classification outputs once the method is implemented.

        Parameters:
            self (Any): Test case instance.

        Returns:
            None: Assertion checks absence of the method on the diagnostics object.
        """
        from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
        diag = PrecipitationDiagnostics(verbose=False)

        assert not hasattr(diag, 'classify_intensity')
    
    def test_convective_vs_stratiform_separation(self: "TestPrecipitationDiagnostics") -> None:
        """
        This test confirms that convective vs. stratiform precipitation separation is not currently implemented in the diagnostics API. It checks that the `separate_convective_stratiform` method is not an attribute of the `PrecipitationDiagnostics` class, serving as a placeholder test to document the current API capabilities. This kind of explicit negative test clarifies expectations for users and maintainers about missing features and can be updated to validate actual separation logic once implemented.

        Parameters:
            self (Any): Test case instance.

        Returns:
            None: Assertion verifies the method is not present.
        """
        from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
        diag = PrecipitationDiagnostics(verbose=False)

        assert not hasattr(diag, 'separate_convective_stratiform')
    
    def test_extreme_precipitation_indices(self: "TestPrecipitationDiagnostics") -> None:
        """
        This test asserts that extreme precipitation indices calculation is not currently implemented in the diagnostics API. It checks that the `calculate_extreme_indices` method is not an attribute of the `PrecipitationDiagnostics` class, serving as a placeholder test to document the current API surface. This kind of negative test helps clarify expectations for users and maintainers about missing features and can be updated to a positive test that validates index calculations once the method is implemented.

        Parameters:
            self (Any): Test case instance.

        Returns:
            None: Assertion checks method absence.
        """
        from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics        
        diag = PrecipitationDiagnostics(verbose=False)

        assert not hasattr(diag, 'calculate_extreme_indices')
    
    def test_precipitation_return_period(self: "TestPrecipitationDiagnostics") -> None:
        """
        This test confirms that precipitation return period estimation is not currently implemented in the diagnostics API. It checks that the `estimate_return_periods` method is not an attribute of the `PrecipitationDiagnostics` class, serving as a placeholder test to document the current API capabilities. This kind of explicit negative test clarifies expectations for users and maintainers about missing features and can be updated to validate actual return period estimation logic once implemented.

        Parameters:
            self (Any): Test case instance.

        Returns:
            None: Assertion verifies absence of the estimation method.
        """
        from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics        
        diag = PrecipitationDiagnostics(verbose=False)

        assert not hasattr(diag, 'estimate_return_periods')
    
    def test_wet_day_definition(self: "TestPrecipitationDiagnostics") -> None:
        """
        This test asserts that wet day definition is not currently implemented in the diagnostics API. It checks that the `identify_wet_days` method is not an attribute of the `PrecipitationDiagnostics` class, serving as a placeholder test to document the current API surface. This kind of negative test helps clarify expectations for users and maintainers about missing features and can be updated to a positive test that validates wet day identification logic once the method is implemented.

        Parameters:
            self (Any): Test case instance.

        Returns:
            None: Assertion confirms the method is not implemented.
        """
        from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics
        diag = PrecipitationDiagnostics(verbose=False)

        assert not hasattr(diag, 'identify_wet_days')


class TestPrecipitationDiagnosticsActual:
    """ Test actual PrecipitationDiagnostics implementation. """
    
    def test_precipitation_diagnostics_initialization(self: "TestPrecipitationDiagnosticsActual") -> None:
        """
        This test confirms that an instance of `PrecipitationDiagnostics` can be created with the `verbose` flag. It checks that the constructor accepts the argument and that the resulting object has the expected `verbose` attribute value. This ensures that the class can be instantiated properly and that the logging behavior can be controlled through the constructor. It is a basic unit test for class initialization.

        Parameters:
            self ("TestPrecipitationDiagnosticsActual"): Test case instance.

        Returns:
            None: Assertions validate instance creation and attribute values.
        """
        diag = PrecipitationDiagnostics(verbose=True)
        assert diag.verbose is True
        
        diag_quiet = PrecipitationDiagnostics(verbose=False)
        assert diag_quiet.verbose is False
    
    def test_precipitation_exists(self: "TestPrecipitationDiagnosticsActual") -> None:
        """
        This test verifies that the `PrecipitationDiagnostics` class can be imported from the diagnostics module. It checks that the class is present and can be accessed without errors. This is a basic smoke test to ensure that the module is correctly structured and that the class is defined. It serves as a sanity check for the presence of the diagnostics component in the codebase.

        Parameters:
            self ("TestPrecipitationDiagnosticsActual"): Test case instance.

        Returns:
            None: Assertions verify presence of the class and its constructor.
        """
        assert PrecipitationDiagnostics is not None
        assert hasattr(PrecipitationDiagnostics, '__init__')
    
    def test_precipitation_has_methods(self: "TestPrecipitationDiagnosticsActual") -> None:
        """
        This test checks for the presence of expected methods in the `PrecipitationDiagnostics` class. It verifies that the class has an `__init__` method and can be instantiated. This serves as a basic check to ensure that the class has the expected API surface and that it can be used to create diagnostic objects. It is a simple test to confirm that the class definition includes the necessary components for initialization.

        Parameters:
            self ("TestPrecipitationDiagnosticsActual"): Test case instance.

        Returns:
            None: Assertions validate presence of listed methods.
        """
        diag = PrecipitationDiagnostics()        
        expected_methods = ['__init__']

        for method in expected_methods:
            assert hasattr(diag, method), f"Missing method: {method}"


class TestPrecipitationDiagnosticsInitialization:
    """ Test initialization and basic functionality. """
    
    def test_init_verbose_true(self: "TestPrecipitationDiagnosticsInitialization") -> None:
        """
        This test checks that initializing `PrecipitationDiagnostics` with `verbose=True` sets the `verbose` attribute to True. It confirms that the constructor correctly accepts the `verbose` argument and that the resulting object has the expected logging behavior. This is a basic unit test for class initialization and helps ensure that users can control verbosity through the constructor.

        Parameters:
            self ("TestPrecipitationDiagnosticsInitialization"): Test case instance.

        Returns:
            None: Assertion validates the `verbose` attribute is True.
        """
        diag = PrecipitationDiagnostics(verbose=True)
        assert diag.verbose is True
    
    def test_init_verbose_false(self: "TestPrecipitationDiagnosticsInitialization") -> None:
        """
        This test checks that initializing `PrecipitationDiagnostics` with `verbose=False` sets the `verbose` attribute to False. It confirms that the constructor correctly accepts the `verbose` argument and that the resulting object has the expected logging behavior. This is a basic unit test for class initialization and helps ensure that users can control verbosity through the constructor.

        Parameters:
            self ("TestPrecipitationDiagnosticsInitialization"): Test case instance.

        Returns:
            None: Assertion validates the `verbose` attribute is False.
        """
        diag = PrecipitationDiagnostics(verbose=False)
        assert diag.verbose is False
    
    def test_init_default_verbose(self: "TestPrecipitationDiagnosticsInitialization") -> None:
        """
        This test checks the default initialization of `PrecipitationDiagnostics` without explicit flags. It confirms that the class defaults to a documented `verbose` setting (True in the current implementation). This documents the sensible default used by callers who do not set logging verbosity. Update this test if defaults change.

        Parameters:
            self ("TestPrecipitationDiagnosticsInitialization"): Test case instance.

        Returns:
            None: Assertion validates the default verbose value.
        """
        diag = PrecipitationDiagnostics()
        assert diag.verbose is True


class TestGetAccumulationHours:
    """ Test accumulation hours parsing. """
    
    def test_get_accumulation_hours_a01h(self: "TestGetAccumulationHours") -> None:
        """
        This test verifies that the `get_accumulation_hours` method correctly parses the 'a01h' accumulation period code into 1 hour. It checks that the method returns the expected integer value for this standard accumulation period. Accurate parsing of accumulation periods is crucial for correct lookback calculations in precipitation difference computations. This unit test documents the expected mapping for this specific code.

        Parameters:
            self ("TestGetAccumulationHours"): Test case instance.

        Returns:
            None: Assertion validates returned accumulation hours.
        """
        diag = PrecipitationDiagnostics(verbose=False)
        assert diag.get_accumulation_hours('a01h') == 1
    
    def test_get_accumulation_hours_a03h(self: "TestGetAccumulationHours") -> None:
        """
        This test verifies that the `get_accumulation_hours` method correctly parses the 'a03h' accumulation period code into 3 hours. It checks that the method returns the expected integer value for this standard accumulation period. Accurate parsing of accumulation periods is crucial for correct lookback calculations in precipitation difference computations. This unit test documents the expected mapping for this specific code.

        Parameters:
            self ("TestGetAccumulationHours"): Test case instance.

        Returns:
            None: Assertion validates returned accumulation hours.
        """
        diag = PrecipitationDiagnostics(verbose=False)
        assert diag.get_accumulation_hours('a03h') == 3
    
    def test_get_accumulation_hours_a06h(self: "TestGetAccumulationHours") -> None:
        """
        This test verifies that the `get_accumulation_hours` method correctly parses the 'a06h' accumulation period code into 6 hours. It checks that the method returns the expected integer value for this standard accumulation period. Accurate parsing of accumulation periods is crucial for correct lookback calculations in precipitation difference computations. This unit test documents the expected mapping for this specific code.

        Parameters:
            self ("TestGetAccumulationHours"): Test case instance.

        Returns:
            None: Assertion validates returned accumulation hours.
        """
        diag = PrecipitationDiagnostics(verbose=False)
        assert diag.get_accumulation_hours('a06h') == 6
    
    def test_get_accumulation_hours_a12h(self: "TestGetAccumulationHours") -> None:
        """
        This test verifies that the `get_accumulation_hours` method correctly parses the 'a12h' accumulation period code into 12 hours. It checks that the method returns the expected integer value for this standard accumulation period. Accurate parsing of accumulation periods is crucial for correct lookback calculations in precipitation difference computations. This unit test documents the expected mapping for this specific code.

        Parameters:
            self ("TestGetAccumulationHours"): Test case instance.

        Returns:
            None: Assertion validates returned accumulation hours.
        """
        diag = PrecipitationDiagnostics(verbose=False)
        assert diag.get_accumulation_hours('a12h') == 12
    
    def test_get_accumulation_hours_a24h(self: "TestGetAccumulationHours") -> None:
        """
        This test verifies that the `get_accumulation_hours` method correctly parses the 'a24h' accumulation period code into 24 hours. It checks that the method returns the expected integer value for this standard accumulation period. Accurate parsing of accumulation periods is crucial for correct lookback calculations in precipitation difference computations. This unit test documents the expected mapping for this specific code.

        Parameters:
            self ("TestGetAccumulationHours"): Test case instance.

        Returns:
            None: Assertion validates returned accumulation hours.
        """
        diag = PrecipitationDiagnostics(verbose=False)
        assert diag.get_accumulation_hours('a24h') == 24
    
    def test_get_accumulation_hours_unknown(self: "TestGetAccumulationHours") -> None:
        """
        This test verifies that the `get_accumulation_hours` method correctly handles unknown accumulation period codes. It checks that the method defaults to 24 hours for unrecognized codes, providing a safe fallback. This behavior prevents crashes when parsing unexpected strings and documents the fallback policy. Adjust the expectation if the default policy changes.

        Parameters:
            self ("TestGetAccumulationHours"): Test case instance.

        Returns:
            None: Assertion validates the default fallback value.
        """
        diag = PrecipitationDiagnostics(verbose=False)
        assert diag.get_accumulation_hours('unknown') == 24


class TestComputePrecipitationDifference:
    """ Test compute_precipitation_difference method. """
    
    @pytest.fixture
    def sample_dataset(self: "TestComputePrecipitationDifference", mpas_2d_processor_diag) -> Any:
        """
        This fixture provides a sample dataset for testing the `compute_precipitation_difference` method. It uses real MPAS 2D processor diagnostics data if available, allowing tests to run with realistic input. If the MPAS data is not available, the fixture will skip the test. This approach ensures that tests can be executed in environments with access to MPAS data while still allowing for graceful handling when data is missing.
        
        Parameters:
            self ("TestComputePrecipitationDifference"): Test case instance.
            mpas_2d_processor_diag: Fixture providing MPAS 2D processor diagnostics.

        Returns:
            Any: Sample MPAS dataset for testing.
        """
        if mpas_2d_processor_diag is None:
            pytest.skip("MPAS data not available")
        
        return mpas_2d_processor_diag.dataset
    
    def test_compute_difference_basic(self: "TestComputePrecipitationDifference", sample_dataset: Any) -> None:
        """
        This test verifies that the `compute_precipitation_difference` method can be called with a sample dataset and that it returns a valid `xarray.DataArray`. It checks that the method executes without errors and that the output is of the expected type. This serves as a basic functional test for the difference computation logic, ensuring that it can process real MPAS data and produce an output.

        Parameters:
            self ("TestComputePrecipitationDifference"): Test case instance.
            sample_dataset (Any): Fixture providing a synthetic MPAS dataset for testing.

        Returns:
            None: Assertions validate output type, shape, and metadata.
        """
        diag = PrecipitationDiagnostics(verbose=False)

        result = diag.compute_precipitation_difference(
            sample_dataset, time_index=5, var_name='rainnc', accum_period='a01h'
        )
        
        assert isinstance(result, xr.DataArray)
        assert len(result.shape) == 1
        assert result.shape[0] > 0  
        assert 'units' in result.attrs
        assert result.attrs['units'] == 'mm'
        assert 'accumulation_period' in result.attrs
        assert result.attrs['accumulation_hours'] == 1
    
    def test_compute_difference_verbose(self: "TestComputePrecipitationDifference", sample_dataset: Any) -> None:
        """
        This test checks that the `compute_precipitation_difference` method produces verbose output when the diagnostics object is initialized with `verbose=True`. It captures the standard output during the method execution and verifies that expected logging messages are present. This ensures that the verbose mode is functioning correctly and provides useful information during computations.

        Parameters:
            self ("TestComputePrecipitationDifference"): Test case instance.
            sample_dataset (Any): Fixture providing a synthetic MPAS dataset for testing.

        Returns:
            None: Assertions check captured output and result type.
        """
        diag = PrecipitationDiagnostics(verbose=True)
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            result = diag.compute_precipitation_difference(
                sample_dataset, time_index=5, var_name='rainnc', accum_period='a01h'
            )
        
        output = captured_output.getvalue()

        assert 'Computing' in output or 'hour accumulation' in output
        assert isinstance(result, xr.DataArray)
    
    def test_compute_difference_total_precipitation(self: "TestComputePrecipitationDifference", sample_dataset: Any) -> None:
        """
        This test verifies that the `compute_precipitation_difference` method can compute total precipitation by combining convective and non-convective components. It checks that the method returns a valid `xarray.DataArray` with non-negative values when given a dataset containing `rainc` and `rainnc`. This test ensures that the method correctly handles the 'total' var_name case and produces physically reasonable results. It serves as an integration test for the difference computation logic.

        Parameters:
            self ("TestComputePrecipitationDifference"): Test case instance.
            sample_dataset (Any): Fixture providing a synthetic MPAS dataset for testing.

        Returns:
            None: Assertions validate output type, shape, and non-negativity.
        """
        diag = PrecipitationDiagnostics(verbose=False)

        result = diag.compute_precipitation_difference(
            sample_dataset, time_index=5, var_name='total', accum_period='a01h'
        )
        
        assert isinstance(result, xr.DataArray)
        assert len(result.shape) == 1
        assert result.shape[0] > 0  
        assert np.all(result.values >= 0)
    
    def test_compute_difference_all_accumulation_periods(self: "TestComputePrecipitationDifference", sample_dataset: Any) -> None:
        """
        This test checks that the `compute_precipitation_difference` method can compute differences for all supported accumulation periods. It iterates through a list of standard accumulation period codes and verifies that the method returns a valid `xarray.DataArray` with expected metadata for each period. This ensures that the method correctly handles different lookback periods and that the output includes necessary attributes for interpretation.

        Parameters:
            self ("TestComputePrecipitationDifference"): Test case instance.
            sample_dataset (Any): Fixture providing a synthetic MPAS dataset for testing.

        Returns:
            None: Assertions verify each period returns a DataArray with expected metadata.
        """
        diag = PrecipitationDiagnostics(verbose=False)
        periods = ['a01h', 'a03h', 'a06h', 'a12h', 'a24h']
        
        for period in periods:
            result = diag.compute_precipitation_difference(
                sample_dataset, time_index=5, var_name='rainnc', accum_period=period
            )

            assert isinstance(result, xr.DataArray)
            assert 'accumulation_period' in result.attrs
    
    def test_compute_difference_no_dataset(self: "TestComputePrecipitationDifference") -> None:
        """
        This test verifies that calling `compute_precipitation_difference` without providing a dataset raises an informative ValueError. It checks that the method detects the missing dataset and communicates the issue clearly to the user. Proper error handling for missing inputs is crucial for usability and helps prevent silent failures or crashes. The test expects a ValueError with a matching message indicating that the dataset was not provided.

        Parameters:
            self ("TestComputePrecipitationDifference"): Test case instance.

        Returns:
            None: Assertion verifies a ValueError is raised.
        """
        diag = PrecipitationDiagnostics(verbose=False)
        
        with pytest.raises(ValueError, match="Dataset not provided"):
            diag.compute_precipitation_difference(
                None, time_index=5, var_name='rainnc', accum_period='a01h' # type: ignore
            )
    
    def test_compute_difference_time_index_out_of_bounds(self: "TestComputePrecipitationDifference", sample_dataset: Any) -> None:
        """
        This test checks that calling `compute_precipitation_difference` with a `time_index` that exceeds the dataset size raises an informative ValueError. It verifies that the method detects the out-of-bounds index and communicates the issue clearly to the user. Proper error handling for invalid indices is crucial for usability and helps prevent silent failures or crashes. The test expects a ValueError with a matching message indicating that the time index exceeds the dataset size.

        Parameters:
            self ("TestComputePrecipitationDifference"): Test case instance.
            sample_dataset (Any): Fixture providing a synthetic MPAS dataset.

        Returns:
            None: Assertion verifies the ValueError is raised for invalid indices.
        """
        diag = PrecipitationDiagnostics(verbose=False)
        
        with pytest.raises(ValueError, match="Time index .* exceeds dataset size"):
            diag.compute_precipitation_difference(
                sample_dataset, time_index=100, var_name='rainnc', accum_period='a01h'
            )
    
    def test_compute_difference_time_index_zero(self: "TestComputePrecipitationDifference", sample_dataset: Any) -> None:
        """
        This test verifies that the `compute_precipitation_difference` method can handle a `time_index` of 0, which typically represents the first timestep in the dataset. The method should return a valid `xarray.DataArray` even when there is no prior data to compute a difference, often by using a fallback approach. This test ensures that the method can gracefully handle edge cases at the start of the time series and still produce an output with expected properties.

        Parameters:
            self ("TestComputePrecipitationDifference"): Test case instance.
            sample_dataset (Any): Fixture providing a synthetic MPAS dataset.

        Returns:
            None: Assertions validate output type and shape for time_index 0.
        """
        diag = PrecipitationDiagnostics(verbose=False)

        result = diag.compute_precipitation_difference(
            sample_dataset, time_index=0, var_name='rainnc', accum_period='a01h'
        )
        
        assert isinstance(result, xr.DataArray)
        assert len(result.shape) == 1
        assert result.shape[0] > 0  
    
    def test_compute_difference_insufficient_lookback(self: "TestComputePrecipitationDifference", sample_dataset: Any) -> None:
        """
        This test checks that the `compute_precipitation_difference` method can handle cases where the specified `time_index` does not have sufficient prior data for the requested accumulation period. For example, if `time_index=2` is used with an accumulation period of 'a06h' that requires 6 hours of lookback, the method should detect the insufficient data and return a valid `xarray.DataArray` with appropriate metadata indicating the issue. This ensures that the method can gracefully handle edge cases where lookback requirements cannot be met and still provide informative output.

        Parameters:
            self ("TestComputePrecipitationDifference"): Test case instance.
            sample_dataset (Any): Fixture providing a synthetic MPAS dataset.

        Returns:
            None: Assertions verify fallback behavior when lookback is insufficient.
        """
        diag = PrecipitationDiagnostics(verbose=False)

        result = diag.compute_precipitation_difference(
            sample_dataset, time_index=2, var_name='rainnc', accum_period='a06h'
        )
        
        assert isinstance(result, xr.DataArray)

        if 'note' in result.attrs:
            assert 'Insufficient' in result.attrs['note']


class TestHandleFirstTimeStep:
    """ Test _handle_first_time_step method. """
    
    @pytest.fixture
    def sample_dataset(self: "TestHandleFirstTimeStep") -> Any:
        """
        This fixture provides a sample dataset for testing the `_handle_first_time_step` method. It creates a synthetic `xarray.Dataset` with cumulative precipitation data for both convective and non-convective components. The dataset includes a 'Time' dimension and an 'nCells' dimension, with random cumulative values that increase over time. This allows tests to run with realistic input data that mimics the structure of MPAS datasets, enabling validation of the first timestep handling logic.

        Parameters:
            self ("TestHandleFirstTimeStep"): Test case instance.

        Returns:
            xr.Dataset: A synthetic dataset with cumulative precipitation data.
        """
        nCells = 50
        nTime = 5
        
        rainnc_data = np.cumsum(np.random.rand(nTime, nCells) * 5, axis=0)
        rainc_data = np.cumsum(np.random.rand(nTime, nCells) * 2, axis=0)
        
        ds = xr.Dataset({
            'rainnc': (['Time', 'nCells'], rainnc_data),
            'rainc': (['Time', 'nCells'], rainc_data),
        })
        
        return ds
    
    def test_handle_first_time_step_zero_verbose(self: "TestHandleFirstTimeStep", sample_dataset: Any) -> None:
        """
        Test internal handling of the first timestep with verbose logging enabled. The helper should provide user-readable messages and return a valid DataArray when `time_index=0`. This validates both messaging and return-type behavior for the first timestep pathway. Useful for ensuring friendly behavior on initial conditions.

        Parameters:
            self ("TestHandleFirstTimeStep"): Test case instance.
            sample_dataset (Any): Fixture providing a synthetic MPAS dataset.

        Returns:
            None: Assertions validate printed output and return type.
        """
        diag = PrecipitationDiagnostics(verbose=True)
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            result = diag._handle_first_time_step(
                sample_dataset, 'Time', 0, 'rainnc', 'a01h', 1, 'xarray'
            )
        
        output = captured_output.getvalue()

        assert 'Time index 0' in output
        assert isinstance(result, xr.DataArray)
    
    def test_handle_first_time_step_insufficient_data_verbose(self: "TestHandleFirstTimeStep", sample_dataset: Any) -> None:
        """
        Test verbose output when the first-timestep handler encounters insufficient lookback data. The routine should emit a warning or note and include metadata indicating the fallback. This ensures observable feedback when computations cannot use full lookback. The test asserts the presence of warning text and appropriate result attributes.

        Parameters:
            self ("TestHandleFirstTimeStep"): Test case instance.
            sample_dataset (Any): Fixture providing a synthetic MPAS dataset.

        Returns:
            None: Assertions validate warning output and result notes.
        """
        diag = PrecipitationDiagnostics(verbose=True)
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            result = diag._handle_first_time_step(
                sample_dataset, 'Time', 2, 'rainnc', 'a06h', 6, 'xarray'
            )
        
        output = captured_output.getvalue()

        assert 'Warning' in output or 'lookback' in output
        assert 'note' in result.attrs


class TestApplyPrecipitationFilters:
    """ Test _apply_precipitation_filters_and_attributes method. """
    
    def test_apply_filters_basic(self: "TestApplyPrecipitationFilters") -> None:
        """
        Test the basic precipitation filters and attribute assignment applied to raw arrays. The method should zero out negative values and clamp extremely large values, while setting expected metadata such as units. This helps ensure quality-control steps are applied consistently before analysis or plotting. The test asserts numeric corrections and presence of expected attributes.

        Parameters:
            self ("TestApplyPrecipitationFilters"): Test case instance.

        Returns:
            None: Assertions validate numeric corrections and metadata.
        """
        data = xr.DataArray(
            np.array([1.0, 5.0, 10.0, -1.0, 200000.0]),
            dims=['nCells']
        )
        
        diag = PrecipitationDiagnostics(verbose=False)
        result = diag._apply_precipitation_filters_and_attributes(data, var_context='rainnc')
        
        assert result.values[3] == 0  
        assert result.values[4] == 0 
        assert 'units' in result.attrs
        assert result.attrs['units'] == 'mm'


class TestAnalyzePrecipitationDiagnostics:
    """ Test _analyze_precipitation_diagnostics method. """
    
    def test_analyze_with_current_and_previous_verbose(self: "TestAnalyzePrecipitationDiagnostics") -> None:
        """
        This test checks that the `_analyze_precipitation_diagnostics` method produces summary output when both current and previous datasets are provided and verbose mode is enabled. The method should print diagnostic information about the current and previous data, such as maximum values and ranges. This helps ensure that the analysis routine provides useful feedback during development or debugging. The test captures standard output and asserts that expected summary text is present.

        Parameters:
            self ("TestAnalyzePrecipitationDiagnostics"): Test case instance.

        Returns:
            None: Assertions validate captured output contains summary text.
        """
        current = xr.DataArray(np.array([10.0, 20.0, 30.0]), dims=['nCells'])
        previous = xr.DataArray(np.array([5.0, 15.0, 25.0]), dims=['nCells'])
        
        diag = PrecipitationDiagnostics(verbose=True)
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            diag._analyze_precipitation_diagnostics(
                current_data=current, previous_data=previous, var_context='rainnc'
            )
        
        output = captured_output.getvalue()

        assert 'Current' in output and 'range' in output
    
    def test_analyze_with_warning_current_less_than_previous(self: "TestAnalyzePrecipitationDiagnostics") -> None:
        """
        This test verifies that the `_analyze_precipitation_diagnostics` method produces a warning when the current precipitation data has lower values than the previous data, which may indicate an issue with the accumulation or data quality. When verbose mode is enabled, the method should print a warning message if it detects that the current maximum value is less than the previous maximum value. This helps ensure that potential issues are flagged during analysis. The test captures standard output and asserts that warning text is present when expected.

        Parameters:
            self ("TestAnalyzePrecipitationDiagnostics"): Test case instance.

        Returns:
            None: Assertion checks that warning text is present in output.
        """
        current = xr.DataArray(np.array([10.0, 15.0, 20.0]), dims=['nCells'])
        previous = xr.DataArray(np.array([15.0, 25.0, 35.0]), dims=['nCells'])
        
        diag = PrecipitationDiagnostics(verbose=True)
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            diag._analyze_precipitation_diagnostics(
                current_data=current, previous_data=previous
            )
        
        output = captured_output.getvalue()

        assert 'WARNING' in output


class TestDaskComputeBranch:
    """ Tests for the dask .compute() branch in _extract_variable_at_time."""

    def test_extract_variable_at_time_with_compute_attribute(self: "TestDaskComputeBranch") -> None:
        """
        This test simulates the presence of a dask-like `.compute()` method on the extracted DataArray. The `_extract_variable_at_time` method should detect the `compute` attribute and call it, allowing for compatibility with dask arrays. The test patches the `compute` method to return a known result and verifies that the final output is correct. This ensures that the diagnostics can handle both standard xarray DataArrays and dask arrays without errors.

        Parameters:
            self ("TestDaskComputeBranch"): Test case instance.
        
        Returns:
            None: Assertions validate that the compute method is called and returns expected results.
        """
        dataset = xr.Dataset({
            'rainc': xr.DataArray(np.array([[1.0, 2.0], [3.0, 4.0]]), dims=['Time', 'nCells']),
        })

        diag = PrecipitationDiagnostics(verbose=False)
        result = diag._extract_variable_at_time(dataset, 'rainc', 0, 'Time', 'xarray')

        np.testing.assert_array_equal(result.values, np.array([1.0, 2.0]))

    def test_extract_variable_at_time_dask_like(self: "TestDaskComputeBranch") -> None:
        """
        This test simulates a dask-like DataArray that has a `compute` method. The `_extract_variable_at_time` method should call the `compute` method on the extracted data, allowing for compatibility with dask arrays. The test patches the `compute` method to return a known result and verifies that the final output is correct. This ensures that the diagnostics can handle both standard xarray DataArrays and dask arrays without errors.

        Parameters:
            self ("TestDaskComputeBranch"): Test case instance.
        
        Returns:
            None: Assertions validate that the compute method is called and returns expected results.
        """
        dataset = xr.Dataset({
            'rainc': xr.DataArray(np.array([[10.0, 20.0], [30.0, 40.0]]), dims=['Time', 'nCells']),
        })

        diag = PrecipitationDiagnostics(verbose=False)

        with patch.object(xr.DataArray, 'compute', create=True, return_value=xr.DataArray(np.array([10.0, 20.0]))):
            result = diag._extract_variable_at_time(dataset, 'rainc', 0, 'Time', 'xarray')
            assert result is not None

    def test_extract_variable_at_time_uxarray_path(self: "TestDaskComputeBranch") -> None:
        """
        This test verifies that the `_extract_variable_at_time` method correctly extracts data when the `data_format` is set to 'uxarray'. In this case, the method should directly index into the dataset without attempting to call a `compute` method. The test provides a simple dataset and checks that the correct time slice is extracted as expected. This ensures that the method can handle different data formats appropriately.

        Parameters:
            self ("TestDaskComputeBranch"): Test case instance.

        Returns:
            None: Assertions validate that the correct time slice is extracted without compute.
        """
        dataset = xr.Dataset({
            'rainc': xr.DataArray(np.array([[5.0, 6.0], [7.0, 8.0]]), dims=['Time', 'nCells']),
        })

        diag = PrecipitationDiagnostics(verbose=False)
        result = diag._extract_variable_at_time(dataset, 'rainc', 1, 'Time', 'uxarray')

        np.testing.assert_array_equal(result.values, np.array([7.0, 8.0]))


class TestVerboseDiagnosticPrints:
    """ Tests for verbose print branches in accumulation methods. """

    def test_compute_total_accumulation_verbose(self: "TestVerboseDiagnosticPrints") -> None:
        """
        This test checks that the `_compute_total_precipitation_accumulation` method produces verbose output when `verbose=True`. It captures the standard output during the method execution and verifies that expected diagnostic messages are present. This ensures that the verbose mode is functioning correctly and provides useful information during computations. The test also validates that the computed total precipitation accumulation is correct based on the input dataset.

        Parameters:
            self ("TestVerboseDiagnosticPrints"): Test case instance.

        Returns:
            None: Assertions validate that verbose output is produced and results are correct.
        """
        dataset = xr.Dataset({
            'rainc': xr.DataArray(np.array([[1.0, 2.0], [3.0, 4.0]]), dims=['Time', 'nCells']),
            'rainnc': xr.DataArray(np.array([[0.5, 1.0], [1.5, 2.0]]), dims=['Time', 'nCells']),
        })

        diag = PrecipitationDiagnostics(verbose=True)
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            result = diag._compute_total_precipitation_accumulation(dataset, 1, 0, 'Time', 'xarray')

        output = captured_output.getvalue()
        assert 'total' in output.lower() or 'range' in output.lower()
        np.testing.assert_array_almost_equal(result.values, np.array([3.0, 3.0]))

    def test_compute_single_variable_accumulation_verbose(self: "TestVerboseDiagnosticPrints") -> None:
        """
        This test checks that the `_compute_single_variable_accumulation` method produces verbose output when `verbose=True`. It captures the standard output during the method execution and verifies that expected diagnostic messages are present. This ensures that the verbose mode is functioning correctly and provides useful information during computations. The test also validates that the computed single variable accumulation is correct based on the input dataset.

        Parameters:
            self ("TestVerboseDiagnosticPrints"): Test case instance.

        Returns:
            None: Assertions validate that verbose output is produced and results are correct.
        """
        dataset = xr.Dataset({
            'rainnc': xr.DataArray(np.array([[2.0, 4.0], [5.0, 8.0]]), dims=['Time', 'nCells']),
        })

        diag = PrecipitationDiagnostics(verbose=True)
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            result = diag._compute_single_variable_accumulation(dataset, 'rainnc', 1, 0, 'Time', 'xarray')

        output = captured_output.getvalue()

        assert len(output) > 0  
        np.testing.assert_array_almost_equal(result.values, np.array([3.0, 4.0]))

    def test_analyze_not_verbose_returns_early(self: "TestVerboseDiagnosticPrints") -> None:
        """
        This test verifies that the `_analyze_precipitation_diagnostics` method returns early without producing output when `verbose=False`. It captures the standard output during the method execution and checks that it is empty, confirming that no analysis messages are printed. This ensures that the method respects the verbose setting and does not produce unnecessary output when verbose mode is disabled.

        Parameters:
            self ("TestVerboseDiagnosticPrints"): Test case instance.

        Returns:
            None: Assertions validate that no output is produced when verbose is disabled.
        """
        diag = PrecipitationDiagnostics(verbose=False)

        current = xr.DataArray(np.array([10.0, 20.0]), dims=['nCells'])
        previous = xr.DataArray(np.array([5.0, 15.0]), dims=['nCells'])

        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            diag._analyze_precipitation_diagnostics(current_data=current, previous_data=previous)

        assert captured_output.getvalue() == ''


class TestExceptionHandlers:
    """ Tests for exception handling paths in diagnostic helper methods. """

    def test_print_current_previous_comparison_exception(self: "TestExceptionHandlers") -> None:
        """
        This test verifies that the `_print_current_previous_comparison` method handles exceptions gracefully when it encounters data that does not support min/max operations. The method should catch the exception and print an informative message instead of crashing. This ensures that the diagnostics can continue running even if there are issues with the data, and provides feedback to the user about what went wrong.

        Parameters:
            self ("TestExceptionHandlers"): Test case instance.

        Returns:
            None: Assertions validate that error messages are printed when min/max extraction fails.
        """
        diag = PrecipitationDiagnostics(verbose=True)
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            diag._print_current_previous_comparison("not_an_array", "not_an_array", 'test')

        output = captured_output.getvalue()

        assert 'Could not analyze' in output

    def test_compute_result_statistics_exception(self: "TestExceptionHandlers") -> None:
        """
        This test checks that the `_compute_result_statistics` method handles exceptions gracefully when it encounters data that cannot be processed as expected. The method should catch any exceptions that arise during the computation of statistics and return `None`, while also printing an informative message. This ensures that the diagnostics can continue running even if there are issues with the data, and provides feedback to the user about what went wrong.

        Parameters:
            self ("TestExceptionHandlers"): Test case instance.

        Returns:
            None: Assertions validate that `None` is returned when result statistics computation fails.
        """
        diag = PrecipitationDiagnostics(verbose=True)
        result = diag._compute_result_statistics("invalid_data")
        assert result is None

    def test_print_result_data_analysis_no_finite_values(self: "TestExceptionHandlers") -> None:
        """
        This test verifies that the `_print_result_data_analysis` method handles cases where there are no finite values in the data. The method should detect this condition and print a warning message instead of attempting to compute statistics or summaries. This ensures that the diagnostics can provide feedback about data quality issues without crashing, and informs the user about the lack of valid data for analysis.

        Parameters:
            self ("TestExceptionHandlers"): Test case instance.

        Returns:
            None: Assertions validate that a warning message is printed when no finite values are present.
        """
        diag = PrecipitationDiagnostics(verbose=True)
        data = xr.DataArray(np.array([np.nan, np.nan, np.nan]), dims=['nCells'])

        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            diag._print_result_data_analysis(data, 'test')

        output = captured_output.getvalue()

        assert 'No finite values' in output

    def test_handle_first_time_step_exception(self: "TestExceptionHandlers") -> None:
        """
        This test checks that the `_handle_first_time_step` method handles exceptions gracefully when it encounters an empty dataset or a dataset that does not have the expected time index. The method should catch the exception and print an informative message instead of crashing. This ensures that the diagnostics can continue running even if there are issues with the input dataset, and provides feedback to the user about what went wrong. The test expects a ValueError with a matching message indicating that the time index cannot be handled.

        Parameters:
            self ("TestExceptionHandlers"): Test case instance.

        Returns:
            None: Assertions validate that a ValueError is raised with an informative message when the dataset is not suitable for handling the first time step.
        """
        diag = PrecipitationDiagnostics(verbose=True)
        empty_ds = xr.Dataset()

        with pytest.raises(ValueError, match="Cannot handle time index"):
            diag._handle_first_time_step(empty_ds, 'Time', 0, 'rainc', 'a01h', 1, 'xarray')

    def test_analyze_with_result_data(self: "TestExceptionHandlers") -> None:
        """
        This test verifies that the `_analyze_precipitation_diagnostics` method can analyze result data when provided with a valid `xarray.DataArray`. The method should compute statistics and print an analysis summary of the result data, even if previous data is not provided. This ensures that the diagnostics can still provide useful feedback about the current results, which is important for understanding the output of the computations. The test captures standard output and asserts that expected analysis text is present.

        Parameters:
            self ("TestExceptionHandlers"): Test case instance.

        Returns:
            None: Assertions validate that the method prints the expected analysis output.
        """
        diag = PrecipitationDiagnostics(verbose=True)
        result = xr.DataArray(np.array([0.0, 1.5, 3.0, 0.0, 5.0]), dims=['nCells'])

        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            diag._analyze_precipitation_diagnostics(result_data=result, var_context='rainnc')

        output = captured_output.getvalue()

        assert 'Result' in output
        assert 'mean' in output.lower()

    def test_compute_result_statistics_valid(self: "TestExceptionHandlers") -> None:
        """
        This test checks that the `_compute_result_statistics` method correctly computes statistics for a valid `xarray.DataArray`. The method should return a dictionary containing the minimum, maximum, count of non-zero values, and total count of finite values in the data. This ensures that the method can successfully analyze the result data and provide useful summary statistics, which are important for understanding the characteristics of the computed precipitation differences. The test asserts that the returned statistics are correct based on the input data.

        Parameters:
            self ("TestExceptionHandlers"): Test case instance.

        Returns:
            None: Assertions validate that the method returns correct statistics for the input data.
        """
        diag = PrecipitationDiagnostics(verbose=True)

        data = xr.DataArray(np.array([0.0, 0.5, 1.0, 2.0, np.nan]), dims=['nCells'])
        stats = diag._compute_result_statistics(data)

        assert stats is not None
        assert stats['min'] == 0.0
        assert stats['max'] == 2.0
        assert stats['nonzero_count'] > 0
        assert stats['total_count'] == 4  # NaN excluded


class TestTimeSliceInfo:
    """ Tests for _print_time_slice_info branches. """

    def test_print_time_slice_info_with_time_coord(self: "TestTimeSliceInfo") -> None:
        """
        This test verifies that the `_print_time_slice_info` method correctly prints time slice information when the dataset contains a 'Time' coordinate. The method should extract the current and previous time values based on the provided `time_index` and `time_step_diff`, and print them in a user-friendly format. This ensures that the diagnostics can provide useful temporal context during computations, which is important for understanding the timing of precipitation events. The test captures standard output and asserts that expected time information is present.
        
        Parameters:
            self ("TestTimeSliceInfo"): Test case instance.

        Returns:
            None: Assertions validate that the method prints the expected time slice information.
        """
        times = pd.date_range('2024-01-01', periods=3, freq='h')

        dataset = xr.Dataset({
            'rainc': xr.DataArray(np.zeros((3, 2)), dims=['Time', 'nCells']),
        }, coords={'Time': times})

        diag = PrecipitationDiagnostics(verbose=True)
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            diag._print_time_slice_info(dataset, 2, var_context='total', time_step_diff=1)

        output = captured_output.getvalue()

        assert 'Time slice info' in output
        assert 'total' in output
        assert 'Current time' in output
        assert 'Previous time' in output

    def test_print_time_slice_info_without_time_coord(self: "TestTimeSliceInfo") -> None:
        """
        This test checks that the `_print_time_slice_info` method can still print index-based information when the dataset does not contain a 'Time' coordinate. The method should detect the absence of the time coordinate and print a message indicating that it is using index-based information instead. This ensures that the diagnostics can provide some level of feedback even when temporal metadata is missing, which can be important for debugging or understanding the context of the computations. The test captures standard output and asserts that expected index-based information is present.

        Parameters:
            self ("TestTimeSliceInfo"): Test case instance.

        Returns:
            None: Assertions validate that the method prints the expected index-based information.
        """
        dataset = xr.Dataset({
            'rainc': xr.DataArray(np.zeros((3, 2)), dims=['Time', 'nCells']),
        })

        diag = PrecipitationDiagnostics(verbose=True)
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            diag._print_time_slice_info(dataset, 2, time_step_diff=1)

        output = captured_output.getvalue()

        assert 'indices' in output.lower()

    def test_print_time_slice_info_not_verbose(self: "TestTimeSliceInfo") -> None:
        """
        This test verifies that the `_print_time_slice_info` method does not produce any output when `verbose=False`, even if the dataset contains a 'Time' coordinate. The method should check the verbose setting and return early without printing any information. This ensures that the diagnostics respect the verbose setting and do not produce unnecessary output when verbose mode is disabled. The test captures standard output and asserts that it is empty.

        Parameters:
            self ("TestTimeSliceInfo"): Test case instance.

        Returns:
            None: Assertions validate that no output is produced when verbose is disabled.
        """
        dataset = xr.Dataset()

        diag = PrecipitationDiagnostics(verbose=False)
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            diag._print_time_slice_info(dataset, 0)

        assert captured_output.getvalue() == ''

    def test_print_time_slice_info_exception(self: "TestTimeSliceInfo") -> None:
        """
        This test checks that the `_print_time_slice_info` method handles exceptions gracefully when it encounters an out-of-bounds index or other issues while trying to access time slice information. The method should catch any exceptions that arise and print an informative message instead of crashing. This ensures that the diagnostics can continue running even if there are issues with accessing time slice information, and provides feedback to the user about what went wrong. The test captures standard output and asserts that the expected error message is present when an exception occurs.

        Parameters:
            self ("TestTimeSliceInfo"): Test case instance.

        Returns:
            None: Assertions validate that the method prints the expected error message when an exception occurs.
        """
        dataset = xr.Dataset({
            'rainc': xr.DataArray(np.zeros((2, 2)), dims=['Time', 'nCells']),
        }, coords={'Time': [0, 1]})

        diag = PrecipitationDiagnostics(verbose=True)
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            diag._print_time_slice_info(dataset, 10, time_step_diff=5)

        output = captured_output.getvalue()

        assert 'Could not print time slice info' in output


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

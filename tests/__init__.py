#!/usr/bin/env python3
"""
MPAS Analysis Package Test Suite Runner

This module provides the test runner and test suite orchestration for the comprehensive
MPAS Analysis package test collection. The runner executes all unit tests, integration
tests, and functional tests across the package, providing detailed test results, dependency
verification, and comprehensive test reporting. This module serves as the central entry
point for running the complete test suite with proper test discovery, execution, and
result summarization.

Tests Performed:
    Test Module Discovery and Execution:
        - test_data_processing: Data processing utilities and processor class tests
        - test_visualization: Visualization classes and plotting functionality tests
        - test_utils: Utility functions and helper method tests
        - Dynamic test module loading with error handling
        - Automatic test discovery from imported modules
        - Test suite aggregation across multiple test files
    
    Test Result Reporting:
        - Total test count tracking and reporting
        - Pass/fail/error/skip status categorization
        - Individual test failure and error details with tracebacks
        - Skipped test tracking with skip reasons
        - Success rate calculation and percentage reporting
        - Visual test result summary with status indicators

Test Coverage:
    - Test suite orchestration: module discovery, test loading, suite aggregation
    - Dependency verification: core dependencies (numpy, pandas, xarray, matplotlib)
    - Optional dependency checking: uxarray, cartopy, psutil with graceful handling
    - Test execution: unittest TextTestRunner with verbose output and buffer control
    - Result collection: test counts, failures, errors, skipped tests
    - Summary generation: formatted output, status indicators, success metrics
    - Exit code management: proper return codes for CI/CD integration
    - Module imports: dynamic import with error handling and fallback
    - Test discovery: automatic test method discovery from modules
    - Error reporting: detailed failure and error information with tracebacks

Testing Approach:
    Integration test runner using unittest framework to orchestrate comprehensive test
    execution across all package test modules. Dynamic module imports with try/except
    error handling provide graceful degradation when modules unavailable. TextTestRunner
    with verbosity=2 provides detailed test execution feedback. Result summarization
    extracts pass/fail/error/skip counts from TestResult object. Dependency checking
    validates availability before test execution. Exit codes support CI/CD integration.

Expected Results:
    - Test runner imports all test modules successfully or reports missing modules
    - Core dependencies (numpy, pandas, xarray, matplotlib) available before testing
    - Optional dependencies checked with warnings for missing packages
    - Test suite aggregates all discovered tests from imported modules
    - Test execution completes for all loaded tests with detailed output
    - Test results collected and categorized (passed, failed, errors, skipped)
    - Summary displays total counts, individual test status, and success rate
    - Failed tests listed with test names for debugging
    - Error tests listed with test names and error indicators
    - Skipped tests listed with reasons for skipping
    - Success rate calculated as percentage of passed tests
    - Exit code 0 when all tests pass, 1 when failures or errors occur
    - Visual indicators (✅ ❌ ⚠️ 🎉) enhance result readability
    - CI/CD integration supported through proper exit code handling

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import unittest
import sys
import os
from pathlib import Path

package_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_dir))

from tests.test_data_processing import *
from tests.test_visualization import *
from tests.test_utils import *


def run_all_tests() -> unittest.TestResult:
    """
    Execute all test modules in the MPAS Analysis package test suite and collect results. This function dynamically discovers and loads test modules, aggregates them into a unified test suite, and executes all tests using the unittest framework. The function handles import errors gracefully, providing warnings for modules that cannot be loaded. Test execution uses verbose output with buffering enabled to provide detailed feedback during the test run. The collected results include information about passed tests, failures, errors, and skipped tests.

    Parameters:
        None

    Returns:
        unittest.TestResult: Test results object containing comprehensive pass/fail/error/skip information, test counts, and detailed traceback data for failures and errors.
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    test_modules = [
        'tests.test_data_processing',
        'tests.test_visualization', 
        'tests.test_utils'
    ]
    
    for module_name in test_modules:
        try:
            module = __import__(module_name, fromlist=[''])
            suite.addTests(loader.loadTestsFromModule(module))
        except ImportError as e:
            print(f"Warning: Could not import {module_name}: {e}")
    
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    return result


def print_test_summary(result: unittest.TestResult) -> None:
    """
    Display a formatted summary of test execution results with detailed statistics and categorization. This function analyzes the TestResult object to extract and display comprehensive test metrics including total test count, passed tests, failures, errors, and skipped tests. The summary includes individual listings of failed and error tests with their names for debugging purposes. Success rate is calculated as a percentage of passed tests relative to total tests executed. Visual indicators (emojis) enhance readability and provide quick status assessment of the test run.

    Parameters:
        result (unittest.TestResult): Test results object from test run containing all execution data, failure tracebacks, error information, and skip reasons.

    Returns:
        None
    """
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    
    passed = total_tests - failures - errors - skipped
    
    print(f"Total tests run: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    
    if failures > 0:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if errors > 0:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    if skipped > 0:
        print("\nSKIPPED:")
        for test, reason in result.skipped:
            print(f"  - {test}: {reason}")
    
    success_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    if failures == 0 and errors == 0:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed or had errors")


if __name__ == '__main__':
    print("Running MPAS Analysis Package Tests")
    print("="*50)
    
    try:
        import numpy
        import pandas
        import xarray
        import matplotlib
        print("✅ Core dependencies available")
    except ImportError as e:
        print(f"❌ Missing core dependency: {e}")
        sys.exit(1)
    
    optional_deps = {
        'uxarray': 'UXarray for unstructured grids',
        'cartopy': 'Cartopy for cartographic projections',
        'psutil': 'PSUtil for system information'
    }
    
    for dep, description in optional_deps.items():
        try:
            __import__(dep)
            print(f"✅ {description}")
        except ImportError:
            print(f"⚠️  {description} (optional, some tests may be skipped)")
    
    print("\nStarting test execution...\n")
    
    result = run_all_tests()
    print_test_summary(result)
    
    if result.failures or result.errors:
        sys.exit(1)
    else:
        sys.exit(0)
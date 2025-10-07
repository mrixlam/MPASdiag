#!/usr/bin/env python3

"""
Test runner for MPAS Analysis Package

This script runs all unit tests for the MPAS Analysis package and provides
a comprehensive test report.

Author: Rubaiat Islam
"""

import unittest
import sys
import os
from pathlib import Path

package_dir = Path(__file__).parent.parent
sys.path.insert(0, str(package_dir))

from tests.test_data_processing import *
from tests.test_visualization import *
from tests.test_utils import *


def run_all_tests():
    """Run all tests and return results."""
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


def print_test_summary(result):
    """Print a summary of test results."""
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
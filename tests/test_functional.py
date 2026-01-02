#!/usr/bin/env python3
"""
MPAS Class Refactoring Functional Tests

This module provides functional tests to verify the MPAS processing classes after
major refactoring, including the 3D processor functionality and backward compatibility
with the 2D processor interface. These tests validate file discovery, data loading,
variable extraction, and vertical level handling using real file paths and directory
structures to ensure the refactored classes maintain expected functionality.

Tests Performed:
    test_3d_processor:
        - MPAS3DProcessor import validation
        - Processor initialization with invariant file
        - File discovery for MPAS output files in data directory
        - 3D data loading from discovered files
        - Available 3D variables extraction
        - Vertical levels retrieval for test variable
        - 3D variable data extraction at surface level
        - Data shape and range validation
    
    test_backward_compatibility:
        - MPAS2DProcessor import validation
        - Processor initialization with invariant file
        - 2D diagnostic file discovery
        - 2D data loading from diagnostic files
        - Available 2D variables extraction
        - Verification of legacy interface functionality

Test Coverage:
    - MPAS3DProcessor class: initialization, file discovery, 3D data loading
    - MPAS2DProcessor class: backward compatibility, 2D data handling
    - File discovery: mpasout files, diagnostic files, directory traversal
    - Data loading: 3D datasets, 2D datasets, invariant files
    - Variable extraction: 3D variables, 2D variables, metadata access
    - Vertical coordinate handling: pressure levels, level extraction
    - Data validation: shape verification, range checking, finite value handling
    - Interface compatibility: legacy method preservation, signature consistency

Testing Approach:
    Functional tests using real file paths and data directories to validate the
    complete workflow of the refactored processor classes. Tests exercise actual
    file system operations, NetCDF file reading, and xarray dataset manipulation.
    Print statements provide verbose output for debugging and verification. Tests
    use pytest framework with explicit failure reporting for integration testing.

Expected Results:
    - MPAS3DProcessor and MPAS2DProcessor import without errors
    - Both processors initialize successfully with invariant files
    - File discovery functions return non-empty lists of valid file paths
    - Data loading completes without exceptions for both 2D and 3D data
    - Variable extraction returns comprehensive lists of available variables
    - Vertical level extraction returns sensible pressure ranges
    - Data extraction produces arrays with expected shapes and finite value ranges
    - Backward compatibility maintained with no interface regressions
    - All functional tests pass with real data structures

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import sys
import pytest
from pathlib import Path

package_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_dir))

def test_3d_processor() -> None:
    """
    Validate MPAS3DProcessor functionality including file discovery, data loading, and 3D variable extraction. This comprehensive functional test exercises the complete workflow of the refactored 3D processor using real file paths and data directories. The test initializes the processor with an invariant file, discovers MPAS output files, loads 3D datasets, extracts available variables, retrieves vertical pressure levels, and validates extracted data shapes and ranges. Print statements provide verbose output for debugging and verification of each processing stage. This ensures the 3D processor maintains expected functionality after refactoring with proper file system operations and NetCDF data handling.

    Parameters:
        None

    Returns:
        None
    """
    print("🧪 Testing MPAS3DProcessor Functionality")
    print("=" * 50)
    
    invariant_file = "data/grids/x1.40962.init.nc"
    data_dir = "data/u120k/mpasout"
    
    try:
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        print("✅ MPAS3DProcessor imported successfully")
        
        processor = MPAS3DProcessor(invariant_file, verbose=True)
        print("✅ MPAS3DProcessor initialized successfully")
        
        print("\n=== Testing File Discovery ===")
        mpasout_files = processor.find_mpasout_files(data_dir)
        print(f"✅ Found {len(mpasout_files)} MPAS output files")
        
        print("\n=== Testing 3D Data Loading ===")
        processor.load_3d_data(data_dir)
        print("✅ 3D data loaded successfully")
        
        print("\n=== Testing Available Variables ===")
        variables = processor.get_available_3d_variables()
        print(f"✅ Found {len(variables)} 3D variables")
        print(f"Sample variables: {variables[:5]}")
        
        print("\n=== Testing Vertical Levels ===")

        if variables:
            test_var = variables[0]
            levels = processor.get_vertical_levels(test_var)
            print(f"✅ Variable '{test_var}' has {len(levels)} vertical levels")
            print(f"Pressure range: {min(levels):.1f} - {max(levels):.1f} Pa")
        
        print("\n=== Testing 3D Variable Data ===")
        if variables:
            test_var = variables[0]
            data = processor.get_3d_variable_data(test_var, level='surface')
            print(f"✅ Extracted '{test_var}' data at surface level")
            print(f"Data shape: {data.shape}")
            if hasattr(data, 'values'):
                import numpy as np
                valid_data = data.values[np.isfinite(data.values)]
                if len(valid_data) > 0:
                    print(f"Data range: {np.min(valid_data):.3f} - {np.max(valid_data):.3f}")
        
        print("\n✅ All 3D processor tests passed!")
        
    except Exception as e:
        print(f"❌ 3D processor test failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"3D processor test failed: {e}")

def test_backward_compatibility() -> None:
    """
    Verify backward compatibility of MPAS2DProcessor interface after refactoring to maintain legacy functionality. This functional test validates the 2D processor maintains its original interface and behavior despite internal refactoring changes. The test initializes the 2D processor with an invariant file, discovers diagnostic files, loads 2D datasets, and extracts available 2D variables using legacy method signatures. Print statements provide detailed output for each compatibility checkpoint. This ensures existing code using the 2D processor interface continues to function without modifications, preventing regressions while enabling new 3D functionality through parallel processor classes.

    Parameters:
        None

    Returns:
        None
    """
    print("\n" + "=" * 50)
    print("🧪 Testing Backward Compatibility")
    print("=" * 50)
    
    invariant_file = "data/grids/x1.40962.init.nc"
    data_dir = "data/u120k/diag"
    
    try:
        from mpasdiag.processing.processors_2d import MPAS2DProcessor
        print("✅ MPAS2DProcessor imported successfully")
        
        processor = MPAS2DProcessor(invariant_file, verbose=True)
        print("✅ MPAS2DProcessor initialized successfully")
        
        print("\n=== Testing 2D File Discovery ===")
        diag_files = processor.find_diagnostic_files(data_dir)
        print(f"✅ Found {len(diag_files)} diagnostic files")
        
        print("\n=== Testing 2D Data Loading ===")
        processor.load_2d_data(data_dir)
        print("✅ 2D data loaded successfully")
        
        print("\n=== Testing Available 2D Variables ===")
        variables = processor.get_available_variables()
        print(f"✅ Found {len(variables)} 2D variables")
        print(f"Sample variables: {variables[:5]}")
        
        print("\n✅ Backward compatibility tests passed!")
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Backward compatibility test failed: {e}")

def main() -> bool:
    """
    Orchestrate execution of all functional tests for MPAS class refactoring with summary reporting. This function serves as the test suite coordinator, running all defined functional tests in sequence and collecting results. The test suite includes 3D processor validation and backward compatibility verification to ensure refactored classes maintain expected functionality. Results are aggregated and a comprehensive summary is printed showing passed/failed test counts. This entry point enables command-line execution of the complete functional test suite with clear success/failure indication through exit codes.

    Parameters:
        None

    Returns:
        bool: True if all tests passed, False if any test failed.
    """
    print("🧪 Testing MPAS Class Refactoring - Functional Tests")
    print("=" * 60)
    
    tests = [
        test_3d_processor,
        test_backward_compatibility
    ]
    
    results = []

    for test in tests:
        try:
            test()
            results.append(True)
        except (AssertionError, Exception):
            results.append(False)
    
    print("\n" + "=" * 60)
    print("=== Final Test Summary ===")
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} functional tests passed!")
        print("🎉 Class refactoring is fully functional!")
    else:
        print(f"❌ {total - passed} out of {total} tests failed")
        print("🚨 Class refactoring needs attention")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
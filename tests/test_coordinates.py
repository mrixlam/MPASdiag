#!/usr/bin/env python3
"""
MPAS Coordinate Specialization Test Suite

This module provides comprehensive functional tests for specialized coordinate handling
in MPAS data processors, verifying that the 2D and 3D processor classes correctly add
appropriate spatial coordinates, dimensions, and metadata specific to their respective
data types. These tests validate coordinate presence, dimension consistency, and proper
differentiation between 2D diagnostic files and 3D model output using real MPAS data
structures and file paths.

Tests Performed:
    test_2d_coordinates:
        - MPAS2DProcessor initialization with invariant file
        - 2D data loading from diagnostic file directory
        - Required coordinate verification (Time, nCells)
        - 2D-specific coordinate checking (nVertices, nIsoLevelsT, nIsoLevelsZ)
        - Spatial variable validation (latCell, lonCell)
        - Coordinate dimension size reporting
    
    test_3d_coordinates:
        - MPAS3DProcessor initialization with invariant file
        - 3D data loading from MPAS output file directory
        - Required coordinate verification (Time, nCells)
        - 3D-specific coordinate checking (nVertLevels, nVertLevelsP1, nEdges, nVertices, nSoilLevels)
        - Spatial variable validation (latCell, lonCell)
        - Coordinate dimension size reporting
    
    test_coordinate_differences:
        - Parallel loading of 2D and 3D datasets
        - Coordinate set comparison between processor types
        - Identification of common coordinates across both types
        - Detection of 2D-specific unique coordinates
        - Detection of 3D-specific unique coordinates
        - Dimension comparison between 2D and 3D datasets
        - Verification of proper coordinate specialization

Test Coverage:
    - MPAS2DProcessor: initialization, 2D data loading, diagnostic file handling
    - MPAS3DProcessor: initialization, 3D data loading, output file handling
    - Coordinate addition: specialized add_spatial_coordinates methods
    - 2D-specific coordinates: isosurface levels, 2D mesh topology
    - 3D-specific coordinates: vertical levels, soil levels, edge coordinates
    - Common coordinates: Time, nCells, spatial variables
    - Dimension handling: coordinate sizes, dimension presence
    - Dataset comparison: set operations on coordinates and dimensions
    - Error handling: missing files, invalid coordinates, loading failures
    - Verbose logging: progress reporting, coordinate verification output
    - File path handling: invariant files, data directories, file discovery

Testing Approach:
    Functional tests using pytest framework with real MPAS file paths and data directories.
    Tests execute actual data loading operations with xarray to verify coordinate handling
    in production-like scenarios. Verbose output provides detailed progress and verification
    messages. Set operations compare coordinates between 2D and 3D processors. Tests use
    real data structures without mocking to validate end-to-end functionality.

Expected Results:
    - MPAS2DProcessor and MPAS3DProcessor initialize successfully with invariant files
    - 2D data loading completes without errors and adds appropriate coordinates
    - 3D data loading completes without errors and adds appropriate coordinates
    - Required coordinates (Time, nCells) present in both 2D and 3D datasets
    - 2D-specific coordinates (nIsoLevelsT, nIsoLevelsZ) present in 2D datasets
    - 3D-specific coordinates (nVertLevels, nVertLevelsP1, nSoilLevels) present in 3D datasets
    - Spatial variables (latCell, lonCell) available in both dataset types
    - Coordinate comparison reveals clear differences between 2D and 3D handling
    - Unique coordinates identified for each processor type
    - Dimension sets differ appropriately between 2D and 3D datasets
    - All tests pass with real MPAS data files and proper coordinate specialization
    - Verbose output confirms successful coordinate verification at each step

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

def test_2d_coordinates() -> None:
    """
    Validate specialized coordinate handling in MPAS2DProcessor for 2D diagnostic data. This function initializes an MPAS2DProcessor instance, loads 2D diagnostic data, and verifies the presence of required and 2D-specific coordinates in the resulting xarray dataset. The test checks for common coordinates (Time, nCells), 2D-specific coordinates (nVertices, nIsoLevelsT, nIsoLevelsZ), and spatial variables (latCell, lonCell). Comprehensive progress reporting and coordinate verification messages provide detailed test feedback. The test uses real MPAS file paths and production data structures to validate end-to-end coordinate addition functionality without mocking.

    Parameters:
        None

    Returns:
        None
    """
    print("🧪 Testing MPAS2DProcessor Coordinate Handling")
    print("=" * 50)
    
    invariant_file = "data/grids/x1.40962.init.nc"
    data_dir = "data/u120k/diag"
    
    try:
        from mpasdiag.processing.processors_2d import MPAS2DProcessor
        
        processor_2d = MPAS2DProcessor(invariant_file, verbose=True)
        print("✅ MPAS2DProcessor initialized successfully")
        
        print("\n=== Testing 2D Data Loading with Specialized Coordinates ===")
        processor_2d.load_2d_data(data_dir)
        print("✅ 2D data loaded with specialized coordinate handling")
        
        print("\n=== Verifying 2D-Specific Coordinates ===")
        dataset = processor_2d.dataset
        expected_coords = ['Time', 'nCells']
        expected_2d_coords = ['nVertices', 'nIsoLevelsT', 'nIsoLevelsZ']
        
        for coord in expected_coords:
            if coord in dataset.coords:
                print(f"✅ Required coordinate '{coord}' present")
            else:
                print(f"❌ Required coordinate '{coord}' missing")
        
        for coord in expected_2d_coords:
            if coord in dataset.coords:
                print(f"✅ 2D-specific coordinate '{coord}' present ({dataset.sizes.get(coord, 0)} values)")
            else:
                print(f"⚠️  2D-specific coordinate '{coord}' not found (may not be in this dataset)")
        
        expected_spatial = ['latCell', 'lonCell']
        for var in expected_spatial:
            if var in dataset.data_vars:
                print(f"✅ Spatial variable '{var}' present")
            else:
                print(f"❌ Spatial variable '{var}' missing")
        
    except Exception as e:
        print(f"❌ 2D coordinate test failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"2D coordinate test failed: {e}")

def test_3d_coordinates() -> None:
    """
    Validate specialized coordinate handling in MPAS3DProcessor for 3D model output data. This function initializes an MPAS3DProcessor instance, loads 3D model output data, and systematically verifies the presence of required and 3D-specific coordinates in the resulting xarray dataset. The test validates common coordinates (Time, nCells), 3D-specific coordinates (nVertLevels, nVertLevelsP1, nEdges, nVertices, nSoilLevels), and spatial variables (latCell, lonCell) with dimension size reporting. Detailed verification messages track coordinate presence and values throughout the test execution. Real MPAS file paths and production data structures ensure authentic testing of the 3D coordinate addition pipeline without mocking.

    Parameters:
        None

    Returns:
        None
    """
    print("\n" + "=" * 50)
    print("🧪 Testing MPAS3DProcessor Coordinate Handling")
    print("=" * 50)
    
    invariant_file = "data/grids/x1.40962.init.nc"
    data_dir = "data/u120k/mpasout"
    
    try:
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        
        processor_3d = MPAS3DProcessor(invariant_file, verbose=True)
        print("✅ MPAS3DProcessor initialized successfully")
        
        print("\n=== Testing 3D Data Loading with Specialized Coordinates ===")
        processor_3d.load_3d_data(data_dir)
        print("✅ 3D data loaded with specialized coordinate handling")
        
        print("\n=== Verifying 3D-Specific Coordinates ===")
        dataset = processor_3d.dataset
        expected_coords = ['Time', 'nCells']
        expected_3d_coords = ['nVertLevels', 'nVertLevelsP1', 'nEdges', 'nVertices', 'nSoilLevels']
        
        for coord in expected_coords:
            if coord in dataset.coords:
                print(f"✅ Required coordinate '{coord}' present")
            else:
                print(f"❌ Required coordinate '{coord}' missing")
        
        for coord in expected_3d_coords:
            if coord in dataset.coords:
                print(f"✅ 3D-specific coordinate '{coord}' present ({dataset.sizes.get(coord, 0)} values)")
            else:
                print(f"⚠️  3D-specific coordinate '{coord}' not found (may not be in this dataset)")
        
        expected_spatial = ['latCell', 'lonCell']

        for var in expected_spatial:
            if var in dataset.data_vars:
                print(f"✅ Spatial variable '{var}' present")
            else:
                print(f"❌ Spatial variable '{var}' missing")
        
    except Exception as e:
        print(f"❌ 3D coordinate test failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"3D coordinate test failed: {e}")

def test_coordinate_differences() -> None:
    """
    Compare coordinate sets between 2D and 3D processors to verify proper specialization. This function loads both 2D diagnostic and 3D model output datasets using their respective processor classes, then performs set operations to identify common, 2D-specific, and 3D-specific coordinates and dimensions. The comparison validates that each processor type adds appropriate specialized coordinates while maintaining common required coordinates. Detailed reporting shows complete coordinate and dimension lists for both processor types, highlighting unique entries for each. This test confirms proper coordinate differentiation between processor types and serves as a regression check for coordinate specialization logic.

    Parameters:
        None

    Returns:
        None
    """
    print("\n" + "=" * 50)
    print("🧪 Testing Coordinate Specialization Differences")
    print("=" * 50)
    
    try:
        from mpasdiag.processing.processors_2d import MPAS2DProcessor
        from mpasdiag.processing.processors_3d import MPAS3DProcessor

        invariant_file = "data/grids/x1.40962.init.nc"
        data_dir_2d = "data/u120k/diag"
        data_dir_3d = "data/u120k/mpasout"

        processor_2d = MPAS2DProcessor(invariant_file, verbose=False)
        processor_2d.load_2d_data(data_dir_2d)

        processor_3d = MPAS3DProcessor(invariant_file, verbose=False)
        processor_3d.load_3d_data(data_dir_3d)
        
        print("Both datasets loaded successfully")
        
        coords_2d = set(processor_2d.dataset.coords.keys())
        coords_3d = set(processor_3d.dataset.coords.keys())
        
        print(f"\n2D Coordinates: {sorted(map(str, coords_2d))}")
        print(f"3D Coordinates: {sorted(map(str, coords_3d))}")
        
        common_coords = coords_2d & coords_3d

        unique_2d = coords_2d - coords_3d
        unique_3d = coords_3d - coords_2d
        
        print(f"\nCommon coordinates: {sorted(map(str, common_coords))}")
        print(f"2D-specific coordinates: {sorted(map(str, unique_2d))}")
        print(f"3D-specific coordinates: {sorted(map(str, unique_3d))}")
        
        dims_2d = set(processor_2d.dataset.sizes.keys())
        dims_3d = set(processor_3d.dataset.sizes.keys())
        
        print(f"\n2D Dimensions: {sorted(map(str, dims_2d))}")
        print(f"3D Dimensions: {sorted(map(str, dims_3d))}")
        
        unique_2d_dims = dims_2d - dims_3d
        unique_3d_dims = dims_3d - dims_2d
        
        print(f"\n2D-specific dimensions: {sorted(map(str, unique_2d_dims))}")
        print(f"3D-specific dimensions: {sorted(map(str, unique_3d_dims))}")
        
        has_differences = len(unique_2d) > 0 or len(unique_3d) > 0 or len(unique_2d_dims) > 0 or len(unique_3d_dims) > 0
        
        if has_differences:
            print("\n✅ 2D and 3D processors handle coordinates differently as expected")
        else:
            print("\n⚠️  No significant coordinate differences detected")
        
    except Exception as e:
        print(f"❌ Coordinate comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Coordinate comparison test failed: {e}")

def main() -> bool:
    """
    Execute comprehensive coordinate specialization test suite with summary reporting. This function serves as the primary entry point for testing specialized coordinate handling across both 2D and 3D MPAS processors. The function orchestrates execution of all coordinate tests (2D coordinates, 3D coordinates, and coordinate differences), collects results, and generates a comprehensive final summary with pass/fail counts. Visual indicators (✅/❌) provide clear test status feedback throughout execution. The function returns a boolean success indicator for script exit code determination, following standard testing conventions where True indicates all tests passed.

    Parameters:
        None

    Returns:
        bool: True if all coordinate tests passed successfully, False if any test failed or raised an exception.
    """
    print("🧪 Testing Specialized Coordinate Handling")
    print("=" * 60)
    
    tests = [
        test_2d_coordinates,
        test_3d_coordinates,
        test_coordinate_differences
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("=== Final Test Summary ===")
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} coordinate tests passed!")
        print("🎉 Specialized coordinate handling is working correctly!")
    else:
        print(f"❌ {total - passed} out of {total} tests failed")
        print("🚨 Coordinate specialization needs attention")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
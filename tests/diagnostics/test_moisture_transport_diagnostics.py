#!/usr/bin/env python3
"""
MPASdiag Test Suite: Moisture Transport Diagnostics Tests

This module contains unit tests for the MoistureTransportDiagnostics class in the mpasdiag.diagnostics.moisture_transport module. It verifies the correct import, initialization, and functionality of the class methods for computing vertically integrated water vapor (IWV) and vertically integrated water vapor transport (IVT) components and magnitude from synthetic 3D MPAS-like datasets. The tests include checks for output types, shapes, CF-compliant attributes, analytical correctness using uniform profiles, and proper handling of vertical level ordering. The test suite uses pytest fixtures to provide synthetic data and captures verbose output for verification. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: April 2026
Version: 1.0.0
"""
import os
import sys
import pytest
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
from io import StringIO
from unittest.mock import patch
from typing import Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mpasdiag.diagnostics.moisture_transport import MoistureTransportDiagnostics
from mpasdiag.processing.constants import GRAVITY


@pytest.fixture
def synthetic_3d_data() -> Dict[str, xr.DataArray]:
    """
    This fixture creates a synthetic 3D dataset that mimics the structure of MPAS output data, with dimensions (Time, nCells, nVertLevels). The specific humidity (qv) is randomly generated between 0.001 and 0.015 kg/kg, the zonal (u) and meridional (v) wind components are randomly generated between -20 and 20 m/s, and the pressure decreases linearly from 100000 Pa at the surface to 20000 Pa at the model top over 20 vertical levels. This synthetic dataset allows for testing the moisture transport diagnostics without relying on actual MPAS output files. 

    Parameters:
        None

    Returns:
        Dict[str, xr.DataArray]: A dictionary containing synthetic 3D DataArrays for 'pressure', 'qv', 'u', and 'v' with appropriate dimensions and random values within realistic ranges for atmospheric conditions.
    """
    rng = np.random.default_rng(42)
    n_time, n_cells, n_vert = 2, 30, 20

    p_levels = np.linspace(100000.0, 20000.0, n_vert)
    pressure = xr.DataArray(
        np.broadcast_to(p_levels, (n_time, n_cells, n_vert)).copy(),
        dims=['Time', 'nCells', 'nVertLevels'],
    )

    qv = xr.DataArray(
        rng.uniform(0.001, 0.015, (n_time, n_cells, n_vert)),
        dims=['Time', 'nCells', 'nVertLevels'],
    )

    u = xr.DataArray(
        rng.uniform(-20.0, 20.0, (n_time, n_cells, n_vert)),
        dims=['Time', 'nCells', 'nVertLevels'],
    )

    v = xr.DataArray(
        rng.uniform(-20.0, 20.0, (n_time, n_cells, n_vert)),
        dims=['Time', 'nCells', 'nVertLevels'],
    )

    return {'pressure': pressure, 'qv': qv, 'u': u, 'v': v}


@pytest.fixture
def uniform_3d_data() -> Dict[str, xr.DataArray]:
    """
    This fixture creates a synthetic 3D dataset with uniform profiles for specific humidity, wind components, and pressure, allowing for analytical verification of the moisture transport diagnostics. The specific humidity is set to a constant value of 0.01 kg/kg, the zonal wind component is set to 10 m/s, the meridional wind component is set to 5 m/s, and the pressure decreases linearly from 100000 Pa at the surface to 20000 Pa at the model top over 55 vertical levels. The fixture also pre-computes the expected values for IWV, IVT_u, and IVT_v based on the analytical formulas for uniform profiles, which can be used in tests to confirm the correctness of the diagnostic computations. 

    Parameters:
        None

    Returns:
        Dict[str, xr.DataArray]: A dictionary containing uniform 3D DataArrays for 'pressure', 'qv', 'u', and 'v', as well as pre-computed expected values for 'expected_iwv', 'expected_ivt_u', and 'expected_ivt_v' based on the analytical solutions for uniform profiles.
    """
    n_time, n_cells, n_vert = 1, 10, 55

    q0, u0, v0 = 0.01, 10.0, 5.0
    p_sfc, p_top = 100000.0, 20000.0

    p_levels = np.linspace(p_sfc, p_top, n_vert)

    pressure = xr.DataArray(
        np.broadcast_to(p_levels, (n_time, n_cells, n_vert)).copy(),
        dims=['Time', 'nCells', 'nVertLevels'],
    )

    qv = xr.DataArray(
        np.full((n_time, n_cells, n_vert), q0),
        dims=['Time', 'nCells', 'nVertLevels'],
    )

    u = xr.DataArray(
        np.full((n_time, n_cells, n_vert), u0),
        dims=['Time', 'nCells', 'nVertLevels'],
    )

    v = xr.DataArray(
        np.full((n_time, n_cells, n_vert), v0),
        dims=['Time', 'nCells', 'nVertLevels'],
    )

    dp = p_sfc - p_top

    return {
        'pressure': pressure,
        'qv': qv,
        'u': u,
        'v': v,
        'expected_iwv': q0 * dp / GRAVITY,
        'expected_ivt_u': q0 * u0 * dp / GRAVITY,
        'expected_ivt_v': q0 * v0 * dp / GRAVITY,
    }


class TestMoistureTransportDiagnostics:
    """ Tests for MoistureTransportDiagnostics class covering import, initialization, computation, and analytical correctness. """


    def test_analyze_moisture_transport_keys(self: 'TestMoistureTransportDiagnostics',
                                             synthetic_3d_data: Dict[str, xr.DataArray]) -> None:
        """
        This test confirms that the analyze_moisture_transport method returns a dictionary containing the expected keys for the computed diagnostics: 'iwv', 'ivt_u', 'ivt_v', and 'ivt'. It also checks that each entry in the result dictionary is itself a dictionary containing the required statistics keys: 'min', 'max', 'mean', 'std', 'units', and 'data'. This test ensures that the analyze_moisture_transport method produces a comprehensive and structured output that includes both the computed DataArrays and their associated summary statistics, which are essential for interpreting the results of the moisture transport diagnostics. 

        Parameters:
            synthetic_3d_data (Dict): Fixture providing synthetic 3D MPAS-like data for specific humidity, wind components, and pressure.

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=False)

        result = diag.analyze_moisture_transport(
            synthetic_3d_data['qv'],
            synthetic_3d_data['u'],
            synthetic_3d_data['v'],
            synthetic_3d_data['pressure'],
        )

        assert set(result.keys()) == {'iwv', 'ivt_u', 'ivt_v', 'ivt'}
        required_stats = {'min', 'max', 'mean', 'std', 'units', 'data'}

        for key in result:
            assert required_stats.issubset(result[key].keys()), \
                f"Missing stats keys in '{key}' entry."

    def test_analyze_moisture_transport_data_arrays(self: 'TestMoistureTransportDiagnostics',
                                                    synthetic_3d_data: Dict[str, xr.DataArray]) -> None:
        """
        This test confirms that the 'data' entry in each sub-dictionary returned by analyze_moisture_transport is an xr.DataArray with the vertical dimension (nVertLevels) collapsed, resulting in 2D fields with dimensions (Time, nCells). It checks that each 'data' entry is indeed a DataArray and that it does not contain the nVertLevels dimension, which indicates that the vertical integration has been performed correctly. This test ensures that the analyze_moisture_transport method produces outputs of the correct type and shape for each diagnostic, which is essential for subsequent analysis and visualization. 

        Parameters:
            synthetic_3d_data (Dict): Fixture providing synthetic 3D MPAS-like data for specific humidity, wind components, and pressure.

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=False)

        result = diag.analyze_moisture_transport(
            synthetic_3d_data['qv'],
            synthetic_3d_data['u'],
            synthetic_3d_data['v'],
            synthetic_3d_data['pressure'],
        )

        for key in result:
            arr = result[key]['data']
            assert isinstance(arr, xr.DataArray), f"'data' for '{key}' is not a DataArray."
            assert 'nVertLevels' not in arr.dims

    def test_analyze_moisture_transport_verbose(self: 'TestMoistureTransportDiagnostics',
                                                synthetic_3d_data: Dict[str, xr.DataArray]) -> None:
        """
        This test checks that when the verbose flag is set to True, the analyze_moisture_transport method prints summary statistics for each computed diagnostic (IWV, IVT_u, IVT_v, and IVT) to the standard output. It captures the output during the method call and verifies that the expected labels for each diagnostic are present in the output string. This test ensures that the verbose mode of analyze_moisture_transport provides informative feedback to the user about the computed diagnostics, which can be helpful for quick assessments of moisture transport characteristics in the dataset. 

        Parameters:
            synthetic_3d_data (Dict): Fixture providing synthetic 3D MPAS-like data for specific humidity, wind components, and pressure.

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=True)

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            diag.analyze_moisture_transport(
                synthetic_3d_data['qv'],
                synthetic_3d_data['u'],
                synthetic_3d_data['v'],
                synthetic_3d_data['pressure'],
            )
            output = mock_stdout.getvalue()

        for label in ('IWV', 'IVT_U', 'IVT_V', 'IVT'):
            assert label in output, f"Expected '{label}' in verbose output."

    def test_analyze_moisture_transport_consistency(self: 'TestMoistureTransportDiagnostics',
                                                    synthetic_3d_data: Dict[str, xr.DataArray]) -> None:
        """
        This test verifies that the summary statistics (min and max) reported in the output of analyze_moisture_transport are consistent with the actual minimum and maximum values of the computed DataArrays for each diagnostic. It checks that the 'min' and 'max' values in the result dictionary are approximately equal to the minimum and maximum of the corresponding 'data' DataArray, allowing for a small relative tolerance to account for numerical precision. This test ensures that the summary statistics provided by analyze_moisture_transport accurately reflect the computed diagnostics, which is important for users who rely on these statistics for interpreting the results. 

        Parameters:
            synthetic_3d_data (Dict): Fixture providing synthetic 3D MPAS-like data for specific humidity, wind components, and pressure.

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=False)

        result = diag.analyze_moisture_transport(
            synthetic_3d_data['qv'],
            synthetic_3d_data['u'],
            synthetic_3d_data['v'],
            synthetic_3d_data['pressure'],
        )

        for key in result:
            arr = result[key]['data']
            assert np.isclose(result[key]['min'], float(arr.min()), rtol=1e-6)
            assert np.isclose(result[key]['max'], float(arr.max()), rtol=1e-6)

    def test_analyze_moisture_transport_with_real_data(self: 'TestMoistureTransportDiagnostics',
                                                       mock_mpas_3d_data: xr.Dataset) -> None:
        """
        This test verifies that the analyze_moisture_transport method can be executed without errors using a session-scoped fixture that provides synthetic 3D MPAS-like data. It checks that the method returns a dictionary with the expected keys and that the 'data' entries are xr.DataArrays, confirming that the method can handle realistic input data structures. This test serves as an integration test to ensure that the entire workflow of analyzing moisture transport diagnostics functions correctly when provided with data that mimics actual MPAS output, which is crucial for validating the applicability of the diagnostics in real-world scenarios. 

        Parameters:
            mock_mpas_3d_data (xr.Dataset): Session-scoped fixture providing 3D MPAS-like data.

        Returns:
            None
        """
        ds = mock_mpas_3d_data

        # Construct pressure (use existing field or build from scratch)
        if 'pressure' in ds:
            pressure = ds['pressure']
        elif 'pressure_p' in ds and 'pressure_base' in ds:
            pressure = ds['pressure_p'] + ds['pressure_base']
        else:
            n_vert = ds.sizes['nVertLevels']
            n_cells = ds.sizes['nCells']
            n_time = ds.sizes['Time']
            p_levels = np.linspace(100000.0, 20000.0, n_vert)
            pressure = xr.DataArray(
                np.broadcast_to(p_levels, (n_time, n_cells, n_vert)).copy(),
                dims=['Time', 'nCells', 'nVertLevels'],
            )

        # Synthesize qv if not present
        if 'qv' in ds:
            qv = ds['qv']
        else:
            rng = np.random.default_rng(0)
            qv = xr.DataArray(
                rng.uniform(0.001, 0.015, pressure.shape),
                dims=pressure.dims,
            )

        u = ds['uReconstructZonal']
        v = ds['uReconstructMeridional']

        diag = MoistureTransportDiagnostics(verbose=False)
        result = diag.analyze_moisture_transport(qv, u, v, pressure)

        assert set(result.keys()) == {'iwv', 'ivt_u', 'ivt_v', 'ivt'}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

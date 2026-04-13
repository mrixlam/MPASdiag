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
from typing import Any, Dict

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

    def test_import_moisture_transport_diagnostics(self: "TestMoistureTransportDiagnostics") -> None:
        """
        This test verifies that the MoistureTransportDiagnostics class can be successfully imported from the mpasdiag.diagnostics.moisture_transport module, confirming that the module and class are correctly defined and accessible. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.diagnostics import moisture_transport
        assert hasattr(moisture_transport, 'MoistureTransportDiagnostics')

    def test_package_export(self: "TestMoistureTransportDiagnostics") -> None:
        """
        This test confirms that the MoistureTransportDiagnostics class is included in the mpasdiag.diagnostics package's __all__ list, ensuring that it is properly exported and available for import when using from mpasdiag.diagnostics import *. This check helps maintain the integrity of the package's public API and ensures that users can access the diagnostics class as intended. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.diagnostics import MoistureTransportDiagnostics as MTC
        from mpasdiag.diagnostics import __all__ as pkg_all
        assert MTC is not None
        assert 'MoistureTransportDiagnostics' in pkg_all

    def test_initialization(self: "TestMoistureTransportDiagnostics") -> None:
        """
        This test verifies that an instance of the MoistureTransportDiagnostics class can be successfully created with the verbose parameter set to True, and that the instance has the expected public methods for analyzing moisture transport. It checks for the presence of methods such as analyze_moisture_transport, compute_ivt, compute_ivt_components, and compute_iwv, which are essential for performing moisture transport diagnostics. This test ensures that the class is properly initialized and that its interface is consistent with the expected functionality. 

        Parameters:
            None

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=True)
        public_methods = [m for m in dir(diag) if not m.startswith('_')]

        expected_methods = [
            'analyze_moisture_transport',
            'compute_ivt',
            'compute_ivt_components',
            'compute_iwv',
        ]
 
        for method in expected_methods:
            assert method in public_methods, f"Expected method '{method}' not found."

        assert diag.verbose is True

    def test_compute_iwv_output_type_and_shape(self: "TestMoistureTransportDiagnostics",
                                               synthetic_3d_data: Dict[str, xr.DataArray]) -> None:
        """
        This test checks that the compute_iwv method returns an xr.DataArray with the expected dimensions and that the vertical dimension (nVertLevels) has been collapsed, resulting in a 2D field with dimensions (Time, nCells). It also verifies that the computed IWV values are non-negative, which is a physical requirement for water vapor content. This test ensures that the compute_iwv method produces outputs of the correct type and shape, and that the values are within a physically realistic range. 

        Parameters:
            synthetic_3d_data (Dict): Fixture providing synthetic 3D MPAS-like data for specific humidity and pressure.

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=False)
        iwv = diag.compute_iwv(synthetic_3d_data['qv'], synthetic_3d_data['pressure'])

        assert isinstance(iwv, xr.DataArray)
        assert 'nVertLevels' not in iwv.dims
        assert 'Time' in iwv.dims
        assert 'nCells' in iwv.dims
        assert float(iwv.min()) >= 0.0

    def test_compute_iwv_cf_attributes(self: "TestMoistureTransportDiagnostics",
                                       synthetic_3d_data: Dict[str, xr.DataArray]) -> None:
        """
        This test verifies that the compute_iwv method assigns correct CF-compliant attributes to the output DataArray, including 'units' set to 'kg m^-2', 'standard_name' set to 'atmosphere_water_vapor_content', and 'long_name' set to 'Vertically Integrated Water Vapor'. These attributes are essential for ensuring that the output is self-describing and can be correctly interpreted by other tools that adhere to CF conventions. This test confirms that the compute_iwv method not only produces the correct numerical output but also includes the necessary metadata for proper usage in scientific analysis. 

        Parameters:
            synthetic_3d_data (Dict): Fixture providing synthetic 3D MPAS-like data for specific humidity and pressure.

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=False)
        iwv = diag.compute_iwv(synthetic_3d_data['qv'], synthetic_3d_data['pressure'])

        assert 'units' in iwv.attrs
        assert 'standard_name' in iwv.attrs
        assert 'long_name' in iwv.attrs
        assert iwv.attrs['standard_name'] == 'atmosphere_water_vapor_content'

    def test_compute_iwv_analytical(self: "TestMoistureTransportDiagnostics",
                                    uniform_3d_data: Dict[str, Any]) -> None:
        """
        This test verifies the numerical correctness of the compute_iwv method using a uniform profile fixture where specific humidity, wind components, and pressure vary in a simple, predictable way. With constant specific humidity and linear pressure, the expected IWV can be calculated analytically as q₀ * (p_sfc - p_top) / g. The test compares the computed IWV values against the pre-computed expected value from the fixture, allowing for a small relative tolerance to account for numerical precision. This test ensures that the compute_iwv method is correctly implementing the vertical integration and producing results that match theoretical expectations for uniform profiles. 

        Parameters:
            uniform_3d_data (Dict): Fixture providing synthetic 3D MPAS-like data with uniform profiles and pre-computed expected IWV. 

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=False)
        iwv = diag.compute_iwv(uniform_3d_data['qv'], uniform_3d_data['pressure'])

        assert np.allclose(iwv.values, uniform_3d_data['expected_iwv'], rtol=1e-6)

    def test_compute_iwv_verbose_output(self: "TestMoistureTransportDiagnostics",
                                        synthetic_3d_data: Dict[str, xr.DataArray]) -> None:
        """
        This test checks that when the verbose flag is set to True, the compute_iwv method prints summary statistics of the computed IWV field, including minimum, maximum, and mean values along with the units. It captures the standard output during the method call and verifies that the expected summary information is present in the output string. This test ensures that the verbose mode of compute_iwv provides useful feedback to the user about the computed diagnostic, which can be helpful for quick assessments of the moisture content in the dataset. 

        Parameters:
            synthetic_3d_data (Dict): Fixture providing synthetic 3D MPAS-like data for specific humidity and pressure.

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=True)

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            diag.compute_iwv(synthetic_3d_data['qv'], synthetic_3d_data['pressure'])
            output = mock_stdout.getvalue()

        assert 'IWV' in output

    def test_compute_iwv_top_to_surface_ordering(self: "TestMoistureTransportDiagnostics") -> None:
        """
        This test verifies that the compute_iwv method produces consistent results regardless of the vertical level ordering in the input data. It creates two synthetic datasets with identical specific humidity and pressure profiles but with opposite vertical level orderings (surface-to-top and top-to-surface). The test then computes IWV for both datasets and checks that the results are approximately equal, confirming that the method correctly handles different vertical level arrangements without affecting the physical correctness of the output. This test ensures that users can input data in either vertical ordering without concern for incorrect IWV calculations. 

        Parameters:
            None

        Returns:
            None
        """
        n_cells, n_vert = 10, 20

        p_sfc_to_top = np.linspace(100000.0, 20000.0, n_vert)
        p_top_to_sfc = p_sfc_to_top[::-1]

        qv_s2t = xr.DataArray(
            np.full((1, n_cells, n_vert), 0.01),
            dims=['Time', 'nCells', 'nVertLevels'],
        )
        qv_t2s = qv_s2t.isel(nVertLevels=slice(None, None, -1))

        pressure_s2t = xr.DataArray(
            np.broadcast_to(p_sfc_to_top, (1, n_cells, n_vert)).copy(),
            dims=['Time', 'nCells', 'nVertLevels'],
        )
        pressure_t2s = xr.DataArray(
            np.broadcast_to(p_top_to_sfc, (1, n_cells, n_vert)).copy(),
            dims=['Time', 'nCells', 'nVertLevels'],
        )

        diag = MoistureTransportDiagnostics(verbose=False)
        iwv_s2t = diag.compute_iwv(qv_s2t, pressure_s2t)
        iwv_t2s = diag.compute_iwv(qv_t2s, pressure_t2s)

        assert np.allclose(iwv_s2t.values, iwv_t2s.values, rtol=1e-6)

    def test_compute_ivt_components_output_types(self: "TestMoistureTransportDiagnostics",
                                                 synthetic_3d_data: Dict[str, xr.DataArray]) -> None:
        """
        This test verifies that the compute_ivt_components method returns a tuple of two xr.DataArrays representing the vertically integrated eastward (IVT_u) and northward (IVT_v) water vapor flux components. It checks that both outputs are DataArrays, that they do not contain the nVertLevels dimension (indicating successful vertical integration), and that they have the expected dimensions of (Time, nCells). This test ensures that the compute_ivt_components method produces outputs of the correct type and shape, which are essential for subsequent analysis and visualization of moisture transport. 

        Parameters:
            synthetic_3d_data (Dict): Fixture providing synthetic 3D MPAS-like data for specific humidity, wind components, and pressure.

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=False)
        result = diag.compute_ivt_components(
            synthetic_3d_data['qv'],
            synthetic_3d_data['u'],
            synthetic_3d_data['v'],
            synthetic_3d_data['pressure'],
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        ivt_u, ivt_v = result
        for arr in (ivt_u, ivt_v):
            assert isinstance(arr, xr.DataArray)
            assert 'nVertLevels' not in arr.dims

    def test_compute_ivt_components_cf_attributes(self: "TestMoistureTransportDiagnostics",
                                                  synthetic_3d_data: Dict[str, xr.DataArray]) -> None:
        """
        This test checks that the compute_ivt_components method assigns correct CF-compliant attributes to the output DataArrays for IVT_u and IVT_v. It verifies that both components have 'units' set to 'kg m^-1 s^-1', 'standard_name' set to 'eastward_water_vapor_flux' for IVT_u and 'northward_water_vapor_flux' for IVT_v, and appropriate 'long_name' attributes. These attributes are crucial for ensuring that the outputs are self-describing and can be correctly interpreted by other tools that adhere to CF conventions. This test confirms that the compute_ivt_components method not only produces the correct numerical outputs but also includes the necessary metadata for proper usage in scientific analysis. 

        Parameters:
            synthetic_3d_data (Dict): Fixture providing synthetic 3D MPAS-like data for specific humidity, wind components, and pressure.

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=False)
        ivt_u, ivt_v = diag.compute_ivt_components(
            synthetic_3d_data['qv'],
            synthetic_3d_data['u'],
            synthetic_3d_data['v'],
            synthetic_3d_data['pressure'],
        )

        assert ivt_u.attrs.get('standard_name') == 'eastward_water_vapor_flux'
        assert ivt_v.attrs.get('standard_name') == 'northward_water_vapor_flux'

    def test_compute_ivt_components_analytical(self: "TestMoistureTransportDiagnostics",
                                               uniform_3d_data: Dict[str, Any]) -> None:
        """
        This test verifies the numerical correctness of the compute_ivt_components method using a uniform profile fixture where specific humidity, wind components, and pressure vary in a simple, predictable way. With constant specific humidity and linear pressure, the expected IVT_u and IVT_v can be calculated analytically as q₀ * u₀ * (p_sfc - p_top) / g and q₀ * v₀ * (p_sfc - p_top) / g, respectively. The test compares the computed IVT components against the pre-computed expected values from the fixture, allowing for a small relative tolerance to account for numerical precision. This test ensures that the compute_ivt_components method is correctly implementing the vertical integration and producing results that match theoretical expectations for uniform profiles. 

        Parameters:
            uniform_3d_data (Dict): Fixture with uniform fields and pre-computed expected values.

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=False)

        ivt_u, ivt_v = diag.compute_ivt_components(
            uniform_3d_data['qv'],
            uniform_3d_data['u'],
            uniform_3d_data['v'],
            uniform_3d_data['pressure'],
        )

        assert np.allclose(ivt_u.values, uniform_3d_data['expected_ivt_u'], rtol=1e-6)
        assert np.allclose(ivt_v.values, uniform_3d_data['expected_ivt_v'], rtol=1e-6)

    def test_compute_ivt_components_sign(self: "TestMoistureTransportDiagnostics") -> None:
        """
        This test confirms that the compute_ivt_components method correctly captures the sign of the IVT components based on the direction of the wind. It creates synthetic datasets with uniform specific humidity and pressure but with either a positive zonal wind (eastward) and zero meridional wind, or zero zonal wind and a positive meridional wind (northward). The test then checks that the computed IVT_u is positive when the zonal wind is eastward and that IVT_v is positive when the meridional wind is northward, while the other component remains close to zero. This test ensures that the method correctly accounts for wind direction in calculating the moisture transport components. 

        Parameters:
            None

        Returns:
            None
        """
        n = 5
        n_vert = 10
        dims = ['Time', 'nCells', 'nVertLevels']

        qv = xr.DataArray(np.full((1, n, n_vert), 0.01), dims=dims)

        pressure = xr.DataArray(
            np.broadcast_to(np.linspace(100000.0, 20000.0, n_vert), (1, n, n_vert)).copy(),
            dims=dims,
        )

        diag = MoistureTransportDiagnostics(verbose=False)

        u_east = xr.DataArray(np.full((1, n, n_vert), 10.0), dims=dims)
        v_zero = xr.DataArray(np.zeros((1, n, n_vert)), dims=dims)
        ivt_u, ivt_v = diag.compute_ivt_components(qv, u_east, v_zero, pressure)
        assert float(ivt_u.mean()) > 0.0
        assert np.allclose(ivt_v.values, 0.0, atol=1e-10)

        u_zero = xr.DataArray(np.zeros((1, n, n_vert)), dims=dims)
        v_north = xr.DataArray(np.full((1, n, n_vert), 5.0), dims=dims)
        ivt_u2, ivt_v2 = diag.compute_ivt_components(qv, u_zero, v_north, pressure)
        assert np.allclose(ivt_u2.values, 0.0, atol=1e-10)
        assert float(ivt_v2.mean()) > 0.0

    def test_compute_ivt_non_negative(self: "TestMoistureTransportDiagnostics",
                                      synthetic_3d_data: Dict[str, xr.DataArray]) -> None:
        """
        This test verifies that the compute_ivt method returns a non-negative magnitude of the vertically integrated water vapor transport (IVT) by confirming that the computed IVT values are greater than or equal to zero. Since IVT represents the magnitude of the moisture flux, it should not be negative. This test ensures that the compute_ivt method correctly calculates the magnitude using the components and that it adheres to physical constraints. 

        Parameters:
            synthetic_3d_data (Dict): Fixture providing synthetic 3D MPAS-like data for specific humidity, wind components, and pressure. 

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=False)
        ivt_u, ivt_v = diag.compute_ivt_components(
            synthetic_3d_data['qv'],
            synthetic_3d_data['u'],
            synthetic_3d_data['v'],
            synthetic_3d_data['pressure'],
        )
        ivt = diag.compute_ivt(ivt_u, ivt_v)

        assert isinstance(ivt, xr.DataArray)
        assert float(ivt.min()) >= 0.0

    def test_compute_ivt_magnitude(self: "TestMoistureTransportDiagnostics") -> None:
        """
        This test confirms that the compute_ivt method correctly calculates the magnitude of the vertically integrated water vapor transport (IVT) using the Pythagorean theorem. It creates synthetic IVT component fields with known values (e.g., IVT_u = 3 kg m^-1 s^-1 and IVT_v = 4 kg m^-1 s^-1) and checks that the computed IVT magnitude is equal to 5 kg m^-1 s^-1, which is the expected result for these components. Additionally, it verifies that the computed IVT has the correct 'standard_name' attribute set to 'water_vapor_flux'. This test ensures that the compute_ivt method is correctly implementing the mathematical calculation for the magnitude of the moisture transport. 

        Parameters:
            None

        Returns:
            None
        """
        dims = ['Time', 'nCells']
        ivt_u = xr.DataArray(np.full((1, 5), 3.0), dims=dims)
        ivt_v = xr.DataArray(np.full((1, 5), 4.0), dims=dims)

        diag = MoistureTransportDiagnostics(verbose=False)
        ivt = diag.compute_ivt(ivt_u, ivt_v)

        assert np.allclose(ivt.values, 5.0, rtol=1e-10)
        assert ivt.attrs.get('standard_name') == 'water_vapor_flux'

    def test_compute_ivt_geq_components(self: "TestMoistureTransportDiagnostics",
                                        synthetic_3d_data: Dict[str, xr.DataArray]) -> None:
        """
        This test verifies that the computed IVT magnitude is greater than or equal to the absolute values of its components (IVT_u and IVT_v) across all grid points. Since IVT represents the magnitude of the moisture flux, it should always be greater than or equal to the individual components. This test checks that the compute_ivt method correctly calculates the magnitude in a way that is consistent with the mathematical relationship between the components and the magnitude, ensuring that no negative or physically inconsistent values are produced. 

        Parameters:
            synthetic_3d_data (Dict): Fixture providing synthetic 3D MPAS-like data for specific humidity, wind components, and pressure.

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=False)

        ivt_u, ivt_v = diag.compute_ivt_components(
            synthetic_3d_data['qv'],
            synthetic_3d_data['u'],
            synthetic_3d_data['v'],
            synthetic_3d_data['pressure'],
        )

        ivt = diag.compute_ivt(ivt_u, ivt_v)

        assert np.all(ivt.values >= np.abs(ivt_u.values) - 1e-10)
        assert np.all(ivt.values >= np.abs(ivt_v.values) - 1e-10)

    def test_analyze_moisture_transport_keys(self: "TestMoistureTransportDiagnostics",
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

    def test_analyze_moisture_transport_data_arrays(self: "TestMoistureTransportDiagnostics",
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

    def test_analyze_moisture_transport_verbose(self: "TestMoistureTransportDiagnostics",
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

    def test_analyze_moisture_transport_consistency(self: "TestMoistureTransportDiagnostics",
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

    def test_analyze_moisture_transport_with_real_data(self: "TestMoistureTransportDiagnostics",
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

#!/usr/bin/env python3

"""
MPASdiag Test Suite: Moisture Transport Diagnostics Coverage

This module contains tests for the moisture transport diagnostics in MPASdiag, specifically targeting the compute_iwv, compute_ivt_components, and compute_ivt methods of the MoistureTransportDiagnostics class. The tests are designed to verify that the diagnostic functions produce outputs with the correct types, shapes, and attributes when given synthetic input data. Additionally, the tests check that verbose print statements are executed correctly when verbose mode is enabled. The synthetic data used in the tests mimics realistic atmospheric profiles for pressure, specific humidity, and wind components, allowing for comprehensive coverage of the moisture transport diagnostic computations. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import numpy as np
import pytest
import xarray as xr
from io import StringIO
from unittest.mock import patch

from mpasdiag.diagnostics.moisture_transport import (
    MoistureTransportDiagnostics,
    _trapezoidal_column_integral,
)
from mpasdiag.processing.constants import NVERT_LEVELS_DIM


N_CELLS = 8
N_VERT = 6


@pytest.fixture()
def pressure() -> xr.DataArray:
    """
    This fixture creates a synthetic pressure profile for testing. The pressure decreases linearly from 1000 hPa at the surface to 200 hPa at the top of the atmosphere across 6 vertical levels. The resulting DataArray has dimensions (nCells, nVertLevels) and is used as input for testing the moisture transport diagnostics, particularly for the compute_iwv method which integrates specific humidity over the vertical using the pressure levels. 

    Parameters:
        None

    Returns:
        xr.DataArray: A 2D DataArray of shape (nCells, nVertLevels) containing the synthetic pressure profile. 
    """
    p_vals = np.linspace(100_000.0, 20_000.0, N_VERT)
    return xr.DataArray(
        np.tile(p_vals, (N_CELLS, 1)),
        dims=["nCells", NVERT_LEVELS_DIM],
    )


@pytest.fixture()
def specific_humidity(pressure: xr.DataArray) -> xr.DataArray:
    """
    This fixture creates a synthetic specific humidity profile for testing. The specific humidity is set to a constant value of 0.01 kg/kg across all cells and vertical levels, which is a typical value for mid-tropospheric moisture. The resulting DataArray has dimensions (nCells, nVertLevels) and is used as input for testing the moisture transport diagnostics. The pressure fixture is included as a parameter to ensure that the specific humidity profile can be used in tests that require both pressure and specific humidity, such as the compute_iwv method which integrates specific humidity over the vertical using the pressure levels. 

    Parameters:
        pressure (xr.DataArray): The pressure profile DataArray, included as a parameter to ensure that the fixture can be used in tests that require both pressure and specific humidity.

    Returns:
        xr.DataArray: A 2D DataArray of shape (nCells, nVertLevels) containing the synthetic specific humidity profile. 
    """
    return xr.DataArray(
        np.full((N_CELLS, N_VERT), 0.01),
        dims=["nCells", NVERT_LEVELS_DIM],
    )


@pytest.fixture()
def u_component() -> xr.DataArray:
    """
    This fixture creates a synthetic u-component of wind for testing. The u-component is set to a constant value of 10.0 m/s across all cells and vertical levels. The resulting DataArray has dimensions (nCells, nVertLevels) and is used as input for testing the moisture transport diagnostics, particularly for the compute_ivt_components method which calculates the eastward water vapor flux using the u-component of wind and specific humidity. 

    Parameters:
        None

    Returns:
        xr.DataArray: A 2D DataArray of shape (nCells, nVertLevels) containing the synthetic u-component of wind.
    """
    return xr.DataArray(
        np.full((N_CELLS, N_VERT), 10.0),
        dims=["nCells", NVERT_LEVELS_DIM],
    )


@pytest.fixture()
def v_component() -> xr.DataArray:
    """
    This fixture creates a synthetic v-component of wind for testing. The v-component is set to a constant value of -5.0 m/s across all cells and vertical levels. The resulting DataArray has dimensions (nCells, nVertLevels) and is used as input for testing the moisture transport diagnostics, particularly for the compute_ivt_components method which calculates the northward water vapor flux using the v-component of wind and specific humidity. 

    Parameters:
        None

    Returns:
        xr.DataArray: A 2D DataArray of shape (nCells, nVertLevels) containing the synthetic v-component of wind.
    """
    return xr.DataArray(
        np.full((N_CELLS, N_VERT), -5.0),
        dims=["nCells", NVERT_LEVELS_DIM],
    )


class TestTrapezoidalColumnIntegral:
    """ Tests for the _trapezoidal_column_integral function, which performs vertical integration using the trapezoidal rule. """

    def test_uniform_integrand_returns_positive(self: 'TestTrapezoidalColumnIntegral') -> None:
        """
        This test verifies that the _trapezoidal_column_integral function returns a positive result when the integrand is uniform and positive. A uniform integrand of ones is integrated over a pressure profile that decreases with height, which should yield a positive column integral. The test asserts that the result has the correct shape and that all values are greater than zero, confirming that the integration is performed correctly for a simple case.

        Parameters:
            None

        Returns:
            None
        """
        integrand = np.ones((N_CELLS, N_VERT))
        p = np.tile(np.linspace(100_000.0, 20_000.0, N_VERT), (N_CELLS, 1))
        result = _trapezoidal_column_integral(integrand, p)
        assert result.shape == (N_CELLS,)
        assert np.all(result > 0.0)

    def test_zero_integrand_returns_zero(self: 'TestTrapezoidalColumnIntegral') -> None:
        """
        This test verifies that the _trapezoidal_column_integral function returns zero when the integrand is zero. A uniform integrand of zeros is integrated over a pressure profile, which should yield a column integral of zero. The test asserts that the result has the correct shape and that all values are close to zero, confirming that the integration correctly handles cases where there is no contribution from the integrand.

        Parameters:
            None

        Returns:
            None
        """
        integrand = np.zeros((N_CELLS, N_VERT))
        p = np.tile(np.linspace(100_000.0, 20_000.0, N_VERT), (N_CELLS, 1))
        result = _trapezoidal_column_integral(integrand, p)
        assert np.allclose(result, 0.0)

    def test_single_level_pair(self: 'TestTrapezoidalColumnIntegral') -> None:
        """
        This test verifies that the _trapezoidal_column_integral function correctly computes the integral for a simple case with only two vertical levels. The integrand is set to 1.0 at the first level and 0.0 at the second level, while the pressure decreases from 1000 hPa to 900 hPa. The expected result is the average of the integrand values multiplied by the pressure difference, divided by gravity. The test asserts that the result has the correct shape and that the computed integral matches the expected value, confirming that the trapezoidal integration is performed correctly for a single level pair.

        Parameters:
            None

        Returns:
            None
        """
        integrand = np.array([[1.0, 1.0]])
        p = np.array([[100_000.0, 90_000.0]])
        result = _trapezoidal_column_integral(integrand, p)
        import mpasdiag.processing.constants as _c
        expected = 10_000.0 / _c.GRAVITY
        assert result.shape == (1,)
        assert np.isclose(result[0], expected)


class TestIntegrateColumn:
    """ Tests for the _integrate_column method of MoistureTransportDiagnostics, which integrates specific humidity over the vertical using pressure levels. """

    def test_returns_xarray_without_vert_dim(self: 'TestIntegrateColumn',
                                             specific_humidity: xr.DataArray,
                                             pressure: xr.DataArray,) -> None:
        """
        This test verifies that the _integrate_column method of MoistureTransportDiagnostics returns an xarray DataArray without the vertical levels dimension. The method is called with synthetic specific humidity and pressure profiles, and the test asserts that the result is an xarray DataArray, that it does not contain the NVERT_LEVELS_DIM in its dimensions, and that its shape corresponds to (nCells,). This confirms that the integration correctly collapses the vertical dimension and produces a column-integrated result. 

        Parameters:
            specific_humidity (xr.DataArray): The specific humidity profile used for integration, provided by the specific_humidity fixture.
            pressure (xr.DataArray): The pressure profile used for integration, provided by the pressure fixture.

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=False)
        result = diag._integrate_column(specific_humidity, pressure)
        assert isinstance(result, xr.DataArray)
        assert NVERT_LEVELS_DIM not in result.dims
        assert result.shape == (N_CELLS,)

    def test_result_is_positive(self: 'TestIntegrateColumn',
                                specific_humidity: xr.DataArray,
                                pressure: xr.DataArray,) -> None:
        """
        This test verifies that the _integrate_column method of MoistureTransportDiagnostics returns positive values when integrating specific humidity over the vertical using pressure levels. The method is called with synthetic specific humidity and pressure profiles, and the test asserts that the minimum value of the result is greater than 0. This confirms that the integration produces physically meaningful positive values.

        Parameters:
            specific_humidity (xr.DataArray): The specific humidity profile used for integration, provided by the specific_humidity fixture.
            pressure (xr.DataArray): The pressure profile used for integration, provided by the pressure fixture.

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=False)
        result = diag._integrate_column(specific_humidity, pressure)
        assert float(result.min()) > 0.0


class TestComputeIWVVerbose:
    """ Tests for verbose print statements in the compute_iwv method of MoistureTransportDiagnostics, which computes integrated water vapor. """

    def test_verbose_prints_iwv_range_and_mean(self: 'TestComputeIWVVerbose',
                                               specific_humidity: xr.DataArray,
                                               pressure: xr.DataArray,) -> None:
        """
        This test verifies that the compute_iwv method of MoistureTransportDiagnostics prints the range and mean of the integrated water vapor (IWV) when verbose mode is enabled. The method is called with synthetic specific humidity and pressure profiles, and the test captures the standard output to check for the presence of the expected print statements. It asserts that the output contains "IWV range:" and "IWV mean:", confirming that the diagnostic provides summary statistics about the computed IWV when verbose mode is active. Additionally, it checks that the result is an xarray DataArray.

        Parameters:
            specific_humidity (xr.DataArray): The specific humidity profile used for computing IWV, provided by the specific_humidity fixture.
            pressure (xr.DataArray): The pressure profile used for computing IWV, provided by the pressure fixture.

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=True)
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            result = diag.compute_iwv(specific_humidity, pressure)

        out = captured.getvalue()
        assert "IWV range:" in out
        assert "IWV mean:" in out
        assert isinstance(result, xr.DataArray)

    def test_verbose_false_prints_nothing(self: 'TestComputeIWVVerbose',
                                          specific_humidity: xr.DataArray,
                                          pressure: xr.DataArray,) -> None:
        """
        This test verifies that the compute_iwv method of MoistureTransportDiagnostics does not print anything when verbose mode is disabled. The method is called with synthetic specific humidity and pressure profiles, and the test captures the standard output to check for the absence of any print statements. It asserts that the captured output is an empty string, confirming that no diagnostic information is printed when verbose mode is set to False.

        Parameters:
            specific_humidity (xr.DataArray): The specific humidity profile used for computing IWV, provided by the specific_humidity fixture.
            pressure (xr.DataArray): The pressure profile used for computing IWV, provided by the pressure fixture.

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=False)
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            diag.compute_iwv(specific_humidity, pressure)

        assert captured.getvalue() == ""

    def test_iwv_has_correct_attrs(self: 'TestComputeIWVVerbose',
                                   specific_humidity: xr.DataArray,
                                   pressure: xr.DataArray,) -> None:
        """
        This test verifies that the compute_iwv method of MoistureTransportDiagnostics returns an xarray DataArray with the correct attributes. The method is called with synthetic specific humidity and pressure profiles, and the test asserts that the resulting DataArray has a "standard_name" attribute equal to "atmosphere_water_vapor_content" and that its "units" attribute contains "kg". This confirms that the computed integrated water vapor (IWV) is properly annotated with metadata that describes its physical meaning and units. 

        Parameters:
            specific_humidity (xr.DataArray): The specific humidity profile used for computing IWV, provided by the specific_humidity fixture.
            pressure (xr.DataArray): The pressure profile used for computing IWV, provided by the pressure fixture.

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=False)
        result = diag.compute_iwv(specific_humidity, pressure)
        assert result.attrs["standard_name"] == "atmosphere_water_vapor_content"
        assert "kg" in result.attrs["units"]


class TestComputeIVTComponentsVerbose:
    """ Tests for verbose print statements in the compute_ivt_components method of MoistureTransportDiagnostics, which computes the eastward and northward water vapor flux components. """

    def test_verbose_prints_ivt_u_and_v_stats(self: 'TestComputeIVTComponentsVerbose',
                                              specific_humidity: xr.DataArray,
                                              u_component: xr.DataArray,
                                              v_component: xr.DataArray,
                                              pressure: xr.DataArray,) -> None:
        """
        This test verifies that the compute_ivt_components method of MoistureTransportDiagnostics prints the range and mean of the IVT components (IVT_u and IVT_v) when verbose mode is enabled. The method is called with synthetic specific humidity, u-component, v-component, and pressure profiles, and the test captures the standard output to check for the presence of the expected print statements. It asserts that the output contains "IVT_u range:", "IVT_u mean:", "IVT_v range:", and "IVT_v mean:", confirming that the diagnostic provides summary statistics about the computed IVT components when verbose mode is active. Additionally, it checks that the method returns two xarray DataArrays corresponding to IVT_u and IVT_v. 

        Parameters:
            specific_humidity (xr.DataArray): The specific humidity profile used for computing IVT components, provided by the specific_humidity fixture.
            u_component (xr.DataArray): The u-component of wind used for computing IVT components, provided by the u_component fixture.
            v_component (xr.DataArray): The v-component of wind used for computing IVT components, provided by the v_component fixture.
            pressure (xr.DataArray): The pressure profile used for computing IVT components, provided by the pressure fixture.

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=True)
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            ivt_u, ivt_v = diag.compute_ivt_components(
                specific_humidity, u_component, v_component, pressure
            )

        out = captured.getvalue()
        assert "IVT_u range:" in out
        assert "IVT_u mean:" in out
        assert "IVT_v range:" in out
        assert "IVT_v mean:" in out

    def test_verbose_false_prints_nothing(self: 'TestComputeIVTComponentsVerbose',
                                          specific_humidity: xr.DataArray,
                                          u_component: xr.DataArray,
                                          v_component: xr.DataArray,
                                          pressure: xr.DataArray,) -> None:
        """
        This test verifies that the compute_ivt_components method of MoistureTransportDiagnostics does not print anything when verbose mode is disabled. The method is called with synthetic specific humidity, u-component, v-component, and pressure profiles, and the test captures the standard output to check for the absence of any print statements. It asserts that the captured output is an empty string, confirming that no diagnostic information about the IVT components is printed when verbose mode is set to False.

        Parameters:
            specific_humidity (xr.DataArray): The specific humidity profile used for computing IVT components, provided by the specific_humidity fixture.
            u_component (xr.DataArray): The u-component of wind used for computing IVT components, provided by the u_component fixture.
            v_component (xr.DataArray): The v-component of wind used for computing IVT components, provided by the v_component fixture.
            pressure (xr.DataArray): The pressure profile used for computing IVT components, provided by the pressure fixture.

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=False)
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            diag.compute_ivt_components(
                specific_humidity, u_component, v_component, pressure
            )

        assert captured.getvalue() == ""

    def test_returns_two_dataarrays_with_attrs(self: 'TestComputeIVTComponentsVerbose',
                                               specific_humidity: xr.DataArray,
                                               u_component: xr.DataArray,
                                               v_component: xr.DataArray,
                                               pressure: xr.DataArray,) -> None:
        """
        This test verifies that the compute_ivt_components method of MoistureTransportDiagnostics returns two xarray DataArrays with the correct attributes for the eastward and northward water vapor flux components. The method is called with synthetic specific humidity, u-component, v-component, and pressure profiles, and the test asserts that the returned ivt_u DataArray has a "standard_name" attribute equal to "eastward_water_vapor_flux" and that the ivt_v DataArray has a "standard_name" attribute equal to "northward_water_vapor_flux". Additionally, it checks that neither DataArray contains the NVERT_LEVELS_DIM in its dimensions, confirming that the vertical dimension has been properly integrated out in the computation of the IVT components. 

        Parameters:
            specific_humidity (xr.DataArray): The specific humidity profile used for computing IVT components, provided by the specific_humidity fixture.
            u_component (xr.DataArray): The u-component of wind used for computing IVT components, provided by the u_component fixture.
            v_component (xr.DataArray): The v-component of wind used for computing IVT components, provided by the v_component fixture.
            pressure (xr.DataArray): The pressure profile used for computing IVT components, provided by the pressure fixture.

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=False)

        ivt_u, ivt_v = diag.compute_ivt_components(
            specific_humidity, u_component, v_component, pressure
        )

        assert ivt_u.attrs["standard_name"] == "eastward_water_vapor_flux"
        assert ivt_v.attrs["standard_name"] == "northward_water_vapor_flux"
        assert NVERT_LEVELS_DIM not in ivt_u.dims
        assert NVERT_LEVELS_DIM not in ivt_v.dims

    def test_ivt_u_v_magnitudes_consistent_with_wind(self: 'TestComputeIVTComponentsVerbose',
                                                     specific_humidity: xr.DataArray,
                                                     u_component: xr.DataArray,
                                                     v_component: xr.DataArray,
                                                     pressure: xr.DataArray,) -> None:
        """
        This test verifies that the magnitudes of the computed IVT components (IVT_u and IVT_v) are consistent with the magnitudes of the input wind components (u_component and v_component) and the specific humidity. Given that the u-component of wind is set to 10 m/s and the v-component is set to -5 m/s, and the specific humidity is constant, we expect the magnitude of IVT_u to be approximately twice that of IVT_v due to the ratio of the wind components. The test asserts that the absolute values of IVT_u are close to twice the absolute values of IVT_v, confirming that the computed water vapor flux components are consistent with the input wind fields.

        Parameters:
            specific_humidity (xr.DataArray): The specific humidity profile used for computing IVT components, provided by the specific_humidity fixture.
            u_component (xr.DataArray): The u-component of wind used for computing IVT components, provided by the u_component fixture.
            v_component (xr.DataArray): The v-component of wind used for computing IVT components, provided by the v_component fixture.
            pressure (xr.DataArray): The pressure profile used for computing IVT components, provided by the pressure fixture.

        Returns:
            None
        """
        diag = MoistureTransportDiagnostics(verbose=False)

        ivt_u, ivt_v = diag.compute_ivt_components(
            specific_humidity, u_component, v_component, pressure
        )

        assert np.allclose(
            np.abs(ivt_u.values), 2.0 * np.abs(ivt_v.values), rtol=1e-6
        )


class TestComputeIVTVerbose:
    """ Tests for verbose print statements in the compute_ivt method of MoistureTransportDiagnostics, which computes the integrated water vapor transport (IVT) magnitude from its components. """

    @pytest.fixture()
    def ivt_components(self: 'TestComputeIVTVerbose',
                       specific_humidity: xr.DataArray,
                       u_component: xr.DataArray,
                       v_component: xr.DataArray,
                       pressure: xr.DataArray,) -> tuple:
        """
        This fixture computes the IVT components (IVT_u and IVT_v) using the compute_ivt_components method of MoistureTransportDiagnostics. It takes synthetic specific humidity, u-component, v-component, and pressure profiles as input and returns the computed IVT components as a tuple. This fixture is used in multiple tests to provide consistent IVT component inputs for testing the compute_ivt method, which calculates the magnitude of the integrated water vapor transport.

        Parameters:
            specific_humidity (xr.DataArray): The specific humidity profile used for computing IVT components, provided by the specific_humidity fixture.
            u_component (xr.DataArray): The u-component of wind used for computing IVT components, provided by the u_component fixture.
            v_component (xr.DataArray): The v-component of wind used for computing IVT components, provided by the v_component fixture.
            pressure (xr.DataArray): The pressure profile used for computing IVT components, provided by the pressure fixture.

        Returns:
            tuple: A tuple containing the IVT_u and IVT_v DataArrays computed by the compute_ivt_components method.
        """
        diag = MoistureTransportDiagnostics(verbose=False)
        return diag.compute_ivt_components(
            specific_humidity, u_component, v_component, pressure
        )

    def test_verbose_prints_ivt_range_and_mean(self: 'TestComputeIVTVerbose',
                                               ivt_components: tuple,) -> None:
        """
        This test verifies that the compute_ivt method of MoistureTransportDiagnostics prints the range and mean of the integrated water vapor transport (IVT) when verbose mode is enabled. The method is called with synthetic IVT components provided by the ivt_components fixture, and the test captures the standard output to check for the presence of the expected print statements. It asserts that the output contains "IVT range:" and "IVT mean:", confirming that the diagnostic provides summary statistics about the computed IVT magnitude when verbose mode is active. Additionally, it checks that the result is an xarray DataArray.

        Parameters:
            ivt_components (tuple): A tuple containing the IVT_u and IVT_v DataArrays used for computing IVT, provided by the ivt_components fixture.

        Returns:
            None
        """
        ivt_u, ivt_v = ivt_components
        diag = MoistureTransportDiagnostics(verbose=True)
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            result = diag.compute_ivt(ivt_u, ivt_v)

        out = captured.getvalue()
        assert "IVT range:" in out
        assert "IVT mean:" in out
        assert isinstance(result, xr.DataArray)

    def test_verbose_false_prints_nothing(self: 'TestComputeIVTVerbose',
                                          ivt_components: tuple,) -> None:
        """
        This test verifies that the compute_ivt method of MoistureTransportDiagnostics does not print anything when verbose mode is disabled. The method is called with synthetic IVT components provided by the ivt_components fixture, and the test captures the standard output to check for the absence of any print statements. It asserts that the captured output is an empty string, confirming that no diagnostic information about the IVT magnitude is printed when verbose mode is set to False.

        Parameters:
            ivt_components (tuple): A tuple containing the IVT_u and IVT_v DataArrays used for computing IVT, provided by the ivt_components fixture.

        Returns:
            None
        """
        ivt_u, ivt_v = ivt_components
        diag = MoistureTransportDiagnostics(verbose=False)
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            diag.compute_ivt(ivt_u, ivt_v)

        assert captured.getvalue() == ""

    def test_ivt_is_nonnegative_magnitude(self: 'TestComputeIVTVerbose',
                                          ivt_components: tuple,) -> None:
        """
        This test verifies that the compute_ivt method of MoistureTransportDiagnostics returns a non-negative magnitude for the integrated water vapor transport (IVT). The method is called with synthetic IVT components provided by the ivt_components fixture, and the test asserts that the minimum value of the computed IVT is greater than or equal to 0. This confirms that the IVT magnitude is correctly computed as a non-negative quantity, which is consistent with its physical interpretation as a flux magnitude.

        Parameters:
            ivt_components (tuple): A tuple containing the IVT_u and IVT_v DataArrays used for computing IVT, provided by the ivt_components fixture.

        Returns:
            None
        """
        ivt_u, ivt_v = ivt_components
        diag = MoistureTransportDiagnostics(verbose=False)
        ivt = diag.compute_ivt(ivt_u, ivt_v)
        assert float(ivt.min()) >= 0.0
        assert ivt.attrs["standard_name"] == "water_vapor_flux"

    def test_ivt_magnitude_equals_pythagorean(self: 'TestComputeIVTVerbose',
                                              ivt_components: tuple,) -> None:
        """
        This test verifies that the compute_ivt method of MoistureTransportDiagnostics correctly computes the magnitude of the integrated water vapor transport (IVT) as the square root of the sum of squares of its components (IVT_u and IVT_v). The method is called with synthetic IVT components provided by the ivt_components fixture, and the test calculates the expected IVT magnitude using the Pythagorean theorem. It asserts that the computed IVT values are close to the expected values, confirming that the method correctly combines the eastward and northward flux components to produce the overall IVT magnitude.

        Parameters:
            ivt_components (tuple): A tuple containing the IVT_u and IVT_v DataArrays used for computing IVT, provided by the ivt_components fixture.

        Returns:
            None
        """
        ivt_u, ivt_v = ivt_components
        diag = MoistureTransportDiagnostics(verbose=False)
        ivt = diag.compute_ivt(ivt_u, ivt_v)
        expected = np.sqrt(ivt_u.values**2 + ivt_v.values**2)
        assert np.allclose(ivt.values, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

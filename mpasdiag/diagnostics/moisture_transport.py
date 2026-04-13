#!/usr/bin/env python3

"""
MPASdiag Core Diagnostics Module: Moisture Transport Diagnostics

This module defines the MoistureTransportDiagnostics class, which provides methods to compute vertically integrated water vapor (IWV) and vertically integrated water vapor transport (IVT) from MPAS 3D output data. The class includes methods for performing column integration using the trapezoidal rule in pressure coordinates, as well as convenience methods for computing both IWV and IVT components in a single call. The diagnostics computed by this class are essential for analyzing moisture transport patterns and assessing model performance in simulating atmospheric moisture fluxes. The class is designed to work with xarray DataArrays, allowing for seamless integration with MPASdiag's data processing workflows. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: April 2026
Version: 1.0.0
"""
import numpy as np
import xarray as xr
from typing import Tuple, Dict, Any

from mpasdiag.processing.constants import (
    GRAVITY, KG_PER_M2, KG_PER_M_PER_S, NVERT_LEVELS_DIM
)


def _trapezoidal_column_integral(q_arr: np.ndarray, 
                                 p_arr: np.ndarray) -> np.ndarray:
    """
    This function performs the column integration of a given quantity (q_arr) over the vertical dimension in pressure coordinates using the trapezoidal rule. It calculates the pressure thickness (dp) between model levels and applies the trapezoidal rule to compute the integral of q_arr weighted by dp. The resulting column-integrated quantity is then divided by gravity to convert from mass flux to a column-integrated value. The function assumes that the vertical dimension is the last axis of the input arrays and that p_arr represents total atmospheric pressure in Pascals at each model level. The output is an array with the vertical dimension removed, containing the column-integrated quantity in units consistent with q_arr and p_arr.

    Parameters:
        q_arr (np.ndarray): Array of the quantity to integrate, with the vertical dimension as the last axis.
        p_arr (np.ndarray): Array of total atmospheric pressure in Pascals with the same shape as q_arr.

    Returns:
        np.ndarray: Array with the vertical axis removed, containing the column-integrated quantity divided by gravity.
    """
    dp_abs = np.abs(np.diff(p_arr, axis=-1))
    q_mid  = (q_arr[..., :-1] + q_arr[..., 1:]) / 2.0
    return np.sum(q_mid * dp_abs, axis=-1) / GRAVITY


class MoistureTransportDiagnostics:
    """ Computes vertically integrated water vapor (IWV) and vertically integrated water vapor transport (IVT) for MPAS 3D output data. """

    def __init__(self: "MoistureTransportDiagnostics",
                 verbose: bool = True) -> None:
        """
        This is the constructor for the MoistureTransportDiagnostics class. It initializes an instance of the class with an optional verbose flag that controls whether detailed output messages are printed during the moisture transport calculations. The verbose flag is stored as an instance variable and can be used by other methods in the class to determine whether to print diagnostic information about the computed IWV and IVT fields, such as their ranges and means. By default, verbose mode is enabled, allowing users to receive immediate feedback on the results of their moisture transport analyses.

        Parameters:
            verbose (bool): Enable verbose output messages during moisture transport calculations (default: True).

        Returns:
            None
        """
        self.verbose = verbose

    def _integrate_column(self: "MoistureTransportDiagnostics",
                          integrand: xr.DataArray,
                          pressure: xr.DataArray) -> xr.DataArray:
        """
        This function performs the column integration of a given quantity (integrand) over the vertical dimension in pressure coordinates using the trapezoidal rule. It applies the _trapezoidal_column_integral function to compute the integral of the integrand weighted by the pressure thickness (dp) between model levels. The resulting column-integrated quantity is then divided by gravity to convert from mass flux to a column-integrated value. The function uses xarray's apply_ufunc to apply the integration across the vertical dimension while maintaining compatibility with dask for parallelized computations. The output is an xarray DataArray with the vertical dimension removed, containing the column-integrated quantity in units consistent with the input integrand and pressure.

        Parameters:
            integrand (xr.DataArray): Quantity to integrate, defined at each model level (must share the nVertLevels dimension with pressure).
            pressure (xr.DataArray): Total atmospheric pressure in Pascals at each model level.

        Returns:
            xr.DataArray: Column-integrated quantity divided by gravity, with the vertical dimension removed.
        """
        return xr.apply_ufunc(
            _trapezoidal_column_integral,
            integrand,
            pressure,
            input_core_dims=[[NVERT_LEVELS_DIM], [NVERT_LEVELS_DIM]],
            output_core_dims=[[]],
            dask='parallelized',
            output_dtypes=[float],
        )

    def compute_iwv(self: "MoistureTransportDiagnostics",
                    specific_humidity: xr.DataArray,
                    pressure: xr.DataArray) -> xr.DataArray:
        """
        This function computes the vertically integrated water vapor (IWV) by integrating the specific humidity over the vertical dimension in pressure coordinates. It applies the _integrate_column method to the specific humidity and pressure fields to calculate IWV in kg m⁻². The resulting DataArray is assigned CF-compliant attributes, including units of kg m⁻² and standard_name 'atmosphere_water_vapor_content'. If verbose mode is enabled, the function prints the range and mean of the computed IWV to provide immediate feedback on the results. This diagnostic is essential for analyzing moisture content in the atmosphere and can be used to identify features such as atmospheric rivers in model simulations. 

        Parameters:
            specific_humidity (xr.DataArray): Specific humidity in kg kg⁻¹ at each model level with a nVertLevels dimension.
            pressure (xr.DataArray): Total atmospheric pressure in Pascals at each model level, with the same nVertLevels dimension as specific_humidity. 

        Returns:
            xr.DataArray: Vertically integrated water vapor in kg m⁻² with CF-compliant attributes.
        """
        iwv = self._integrate_column(specific_humidity, pressure)

        iwv.attrs.update({
            'units': KG_PER_M2,
            'standard_name': 'atmosphere_water_vapor_content',
            'long_name': 'Vertically Integrated Water Vapor',
        })

        if self.verbose:
            iwv_min = float(iwv.min())
            iwv_max = float(iwv.max())
            iwv_mean = float(iwv.mean())
            print(f"IWV range: {iwv_min:.2f} to {iwv_max:.2f} kg/m²")
            print(f"IWV mean:  {iwv_mean:.2f} kg/m²")

        return iwv

    def compute_ivt_components(self: "MoistureTransportDiagnostics",
                               specific_humidity: xr.DataArray,
                               u_component: xr.DataArray,
                               v_component: xr.DataArray,
                               pressure: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        This function computes the vertically integrated water vapor transport (IVT) components by integrating the product of specific humidity and horizontal wind components over the vertical dimension in pressure coordinates. It calculates the eastward (IVT_u) and northward (IVT_v) components of IVT by applying the _integrate_column method to the products of specific humidity with the zonal (u_component) and meridional (v_component) wind fields, respectively. The resulting DataArrays are assigned CF-compliant attributes, including units of kg m⁻¹ s⁻¹ and standard_names 'eastward_water_vapor_flux' and 'northward_water_vapor_flux'. If verbose mode is enabled, the function prints the range and mean of both IVT components to provide immediate feedback on the results. These diagnostics are essential for analyzing moisture transport patterns and can be used to identify features such as atmospheric rivers in model simulations. 

        Parameters:
            specific_humidity (xr.DataArray): Specific humidity in kg kg⁻¹ at each model level with a nVertLevels dimension.
            u_component (xr.DataArray): Zonal (eastward) wind component in m s⁻¹ at each model level with a nVertLevels dimension.
            v_component (xr.DataArray): Meridional (northward) wind component in m s⁻¹ at each model level with a nVertLevels dimension.
            pressure (xr.DataArray): Total atmospheric pressure in Pascals at each model level with a nVertLevels dimension.

        Returns:
            Tuple[xr.DataArray, xr.DataArray]: Two DataArrays containing the vertically integrated eastward (IVT_u) and northward (IVT_v) water vapor flux components in kg m⁻¹ s⁻¹ with CF-compliant attributes.
        """
        ivt_u = self._integrate_column(specific_humidity * u_component, pressure)
        ivt_v = self._integrate_column(specific_humidity * v_component, pressure)

        ivt_u.attrs.update({
            'units': KG_PER_M_PER_S,
            'standard_name': 'eastward_water_vapor_flux',
            'long_name': 'Vertically Integrated Eastward Water Vapor Flux',
        })

        ivt_v.attrs.update({
            'units': KG_PER_M_PER_S,
            'standard_name': 'northward_water_vapor_flux',
            'long_name': 'Vertically Integrated Northward Water Vapor Flux',
        })

        if self.verbose:
            print(f"IVT_u range: {float(ivt_u.min()):.2f} to {float(ivt_u.max()):.2f} kg/(m·s)")
            print(f"IVT_u mean:  {float(ivt_u.mean()):.2f} kg/(m·s)")
            print(f"IVT_v range: {float(ivt_v.min()):.2f} to {float(ivt_v.max()):.2f} kg/(m·s)")
            print(f"IVT_v mean:  {float(ivt_v.mean()):.2f} kg/(m·s)")

        return ivt_u, ivt_v

    def compute_ivt(self: "MoistureTransportDiagnostics",
                    ivt_u: xr.DataArray,
                    ivt_v: xr.DataArray) -> xr.DataArray:
        """
        This function computes the total vertically integrated water vapor transport (IVT) magnitude by combining the eastward (IVT_u) and northward (IVT_v) components. It calculates the IVT magnitude using the Pythagorean theorem, taking the square root of the sum of squares of IVT_u and IVT_v. The resulting DataArray is assigned CF-compliant attributes, including units of kg m⁻¹ s⁻¹ and standard_name 'water_vapor_flux'. If verbose mode is enabled, the function prints the range and mean of the computed IVT magnitude to provide immediate feedback on the results. This diagnostic is essential for analyzing moisture transport patterns and can be used to identify features such as atmospheric rivers in model simulations. 

        Parameters:
            ivt_u (xr.DataArray): Vertically integrated eastward water vapor flux in kg m⁻¹ s⁻¹, as returned by compute_ivt_components.
            ivt_v (xr.DataArray): Vertically integrated northward water vapor flux in kg m⁻¹ s⁻¹, as returned by compute_ivt_components.

        Returns:
            xr.DataArray: Total vertically integrated water vapor flux magnitude in kg m⁻¹ s⁻¹ with CF-compliant attributes.
        """
        ivt = xr.apply_ufunc(np.sqrt, ivt_u**2 + ivt_v**2, keep_attrs=False, dask='parallelized')

        ivt.attrs.update({
            'units': KG_PER_M_PER_S,
            'standard_name': 'water_vapor_flux',
            'long_name': 'Vertically Integrated Water Vapor Flux',
        })

        if self.verbose:
            ivt_min = float(ivt.min())
            ivt_max = float(ivt.max())
            ivt_mean = float(ivt.mean())
            print(f"IVT range: {ivt_min:.2f} to {ivt_max:.2f} kg/(m·s)")
            print(f"IVT mean:  {ivt_mean:.2f} kg/(m·s)")

        return ivt

    def analyze_moisture_transport(self: "MoistureTransportDiagnostics",
                                   specific_humidity: xr.DataArray,
                                   u_component: xr.DataArray,
                                   v_component: xr.DataArray,
                                   pressure: xr.DataArray) -> Dict[str, Any]:
        """
        This function provides a convenience method to compute all moisture transport diagnostics (IWV, IVT_u, IVT_v, and IVT) in a single call. It takes the specific humidity, zonal and meridional wind components, and pressure fields as inputs, and returns a dictionary containing the computed diagnostics along with their summary statistics (min, max, mean, std) and units. The function first computes IWV using the compute_iwv method, then computes the IVT components using compute_ivt_components, and finally calculates the total IVT magnitude using compute_ivt. If verbose mode is enabled, it prints a summary of the computed diagnostics for quick assessment of the results. This method is designed to streamline the analysis of moisture transport patterns in MPAS model output data. 

        Parameters:
            specific_humidity (xr.DataArray): Specific humidity in kg kg⁻¹ at each model level with a nVertLevels dimension.
            u_component (xr.DataArray): Zonal (eastward) wind component in m s⁻¹ at each model level.
            v_component (xr.DataArray): Meridional (northward) wind component in m s⁻¹ at each model level.
            pressure (xr.DataArray): Total atmospheric pressure in Pascals at each model level.

        Returns:
            Dict[str, Any]: A dictionary containing the computed diagnostics (IWV, IVT_u, IVT_v, IVT) along with their summary statistics and units.
        """
        # Use a silent copy of self to avoid duplicate verbose prints from sub-methods
        _saved_verbose = self.verbose
        self.verbose = False

        iwv = self.compute_iwv(specific_humidity, pressure)
        ivt_u, ivt_v = self.compute_ivt_components(specific_humidity, u_component, v_component, pressure)
        ivt = self.compute_ivt(ivt_u, ivt_v)

        self.verbose = _saved_verbose

        analysis: Dict[str, Any] = {
            'iwv': {
                'data': iwv,
                'min': float(iwv.min()),
                'max': float(iwv.max()),
                'mean': float(iwv.mean()),
                'std': float(iwv.std()),
                'units': KG_PER_M2,
            },
            'ivt_u': {
                'data': ivt_u,
                'min': float(ivt_u.min()),
                'max': float(ivt_u.max()),
                'mean': float(ivt_u.mean()),
                'std': float(ivt_u.std()),
                'units': KG_PER_M_PER_S,
            },
            'ivt_v': {
                'data': ivt_v,
                'min': float(ivt_v.min()),
                'max': float(ivt_v.max()),
                'mean': float(ivt_v.mean()),
                'std': float(ivt_v.std()),
                'units': KG_PER_M_PER_S,
            },
            'ivt': {
                'data': ivt,
                'min': float(ivt.min()),
                'max': float(ivt.max()),
                'mean': float(ivt.mean()),
                'std': float(ivt.std()),
                'units': KG_PER_M_PER_S,
            },
        }

        if self.verbose:
            print("Moisture Transport Analysis:")
            for name, stats in analysis.items():
                print(f"  {name.upper()}: {stats['min']:.2f} to {stats['max']:.2f} {stats['units']} "
                      f"(mean: {stats['mean']:.2f}, std: {stats['std']:.2f})")

        return analysis

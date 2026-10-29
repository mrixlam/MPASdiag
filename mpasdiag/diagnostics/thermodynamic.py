#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Core Diagnostics Module: Thermodynamic Diagnostics

This module provides tools for computing thermodynamic diagnostics from MPAS model output, with a focus on the density potential temperature (theta_rho) following the formulation of Emanuel (1994).

The density potential temperature is defined as:

    theta_rho = theta * (1 + qv/eps) / (1 + qt)

where theta is potential temperature (K), qv is the water vapor mixing ratio (kg/kg), qt is the total water mixing ratio (kg/kg), and eps = Rd/Rv ≈ 0.622 is the ratio of the gas constants for dry air and water vapor. This is the exact form; it equals the virtual potential temperature when there is no condensate (qt = qv).

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: March 2026
Version: 1.0.0
"""

import numpy as np
import xarray as xr
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from mpasdiag.processing.utils_logger import get_logger
from mpasdiag.diagnostics._vertical import (
    as_dataarray,
    interpolate_to_pressure,
    is_lazy,
)

logger = get_logger(__name__)


# Physical constants needed for thermodynamic diagnostics calculations
RD = 287.04  # Gas constant for dry air         [J kg-1 K-1]
RV = 461.6  # Gas constant for water vapor     [J kg-1 K-1]
EPSILON = RD / RV  # Rd / Rv  ≈ 0.622
RV_OVER_RD = RV / RD  # Rv / Rd  ≈ 1.61
CP = 3.5 * RD  # Specific heat of dry air [J kg-1 K-1]; Rd/cp = 2/7 (MPAS)
P0 = 1.0e5  # Reference pressure                [Pa]
G = 9.81  # Gravitational acceleration        [m s-2]

# Default MPAS variable names for thermodynamic fields
DEFAULT_THETA_VAR = "theta"
DEFAULT_QV_VAR = "qv"
DEFAULT_QC_VAR = "qc"
DEFAULT_QR_VAR = "qr"
DEFAULT_QI_VAR = "qi"
DEFAULT_QS_VAR = "qs"
DEFAULT_QG_VAR = "qg"

# List of standard condensate variable names to check for in datasets
CONDENSATE_VARS = [
    DEFAULT_QC_VAR,
    DEFAULT_QR_VAR,
    DEFAULT_QI_VAR,
    DEFAULT_QS_VAR,
    DEFAULT_QG_VAR,
]

# Units for density potential temperature
THETA_RHO_UNITS = "K"


class ThermodynamicDiagnostics:
    """Helper class for computing thermodynamic diagnostics from MPAS model output."""

    def __init__(self: "ThermodynamicDiagnostics", verbose: bool = True) -> None:
        """
        This initializer sets up the ThermodynamicDiagnostics class with an optional verbosity flag that controls the level of detail printed during computations. When verbose is True, the class will print summary statistics for input fields and computed diagnostics to help users understand the range and distribution of values, which can be useful for debugging and interpretation.

        Parameters:
            verbose (bool): If True, print detailed diagnostic messages during computations (default: True).

        Returns:
            None
        """
        # Store the verbose flag for use in other methods
        self.verbose = verbose

    def compute_density_potential_temperature(
        self: "ThermodynamicDiagnostics",
        theta: Any,
        qv: Any,
        qt: Optional[Any] = None,
        qc: Optional[Any] = None,
        qr: Optional[Any] = None,
        qi: Optional[Any] = None,
        qs: Optional[Any] = None,
        qg: Optional[Any] = None,
    ) -> Any:
        """
        This method computes the density potential temperature (theta_rho) from the potential temperature (theta), water vapor mixing ratio (qv), and optionally the total water mixing ratio (qt) or individual condensate species (qc, qr, qi, qs, qg). If qt is not provided directly, it will be computed as the sum of qv and any available condensate species. The exact relation theta_rho = theta * (1 + qv/eps) / (1 + qt), eps = Rd/Rv, is used, which reduces to the virtual potential temperature of compute_virtual_potential_temperature when there is no condensate (qt = qv); the earlier linearised form theta * (1 + (Rv/Rd) qv - qt) is not used. Inputs may be numpy- or dask-backed DataArrays or plain arrays: dask inputs give a lazy result and plain arrays give a plain array. The method returns theta_rho with CF-compliant attributes (DataArrays) and prints summary statistics for the inputs and result if verbosity is enabled.

        Parameters:
            theta (Union[xr.DataArray, np.ndarray]): Potential temperature in Kelvin.
            qv (Union[xr.DataArray, np.ndarray]): Water vapor mixing ratio in kg/kg.
            qt (Optional[Union[xr.DataArray, np.ndarray]]): Total water mixing ratio in kg/kg. If not provided, it will be computed from qv and condensate species.
            qc (Optional[Union[xr.DataArray, np.ndarray]]): Cloud water mixing ratio in kg/kg (default: None).
            qr (Optional[Union[xr.DataArray, np.ndarray]]): Rain water mixing ratio in kg/kg (default: None).
            qi (Optional[Union[xr.DataArray, np.ndarray]]): Cloud ice mixing ratio in kg/kg (default: None).
            qs (Optional[Union[xr.DataArray, np.ndarray]]): Snow mixing ratio in kg/kg (default: None).
            qg (Optional[Union[xr.DataArray, np.ndarray]]): Graupel mixing ratio in kg/kg (default: None).

        Returns:
            Union[xr.DataArray, np.ndarray]: Density potential temperature in Kelvin, a DataArray with CF-compliant attributes or a plain array when no input is a DataArray.
        """
        # Check that required inputs are provided
        if theta is None or qv is None:
            raise ValueError(
                "Both 'theta' (potential temperature) and 'qv' (water vapor "
                "mixing ratio) must be provided."
            )

        # Plain arrays are computed through DataArrays and returned as plain arrays
        inputs = (theta, qv, qt, qc, qr, qi, qs, qg)
        return_array = not any(isinstance(x, xr.DataArray) for x in inputs)
        theta = as_dataarray(theta)
        qv = as_dataarray(qv)
        qt, qc, qr, qi, qs, qg = (
            None if x is None else as_dataarray(x) for x in (qt, qc, qr, qi, qs, qg)
        )

        # If qt is not provided, compute it from qv and any available condensate species
        if qt is None:
            qt = self.compute_total_water_mixing_ratio(qv, qc, qr, qi, qs, qg)

        # If verbose, print summary statistics for the inputs
        verbose = self.verbose and not is_lazy(theta)
        if verbose:
            self._print_input_summary(theta, qv, qt)

        # Compute density potential temperature (exact form, consistent with theta_v)
        theta_rho: xr.DataArray = theta * (1.0 + qv / EPSILON) / (1.0 + qt)

        # Add CF-compliant attributes to the result
        theta_rho.attrs.update(
            {
                "units": THETA_RHO_UNITS,
                "standard_name": "density_potential_temperature",
                "long_name": "Density Potential Temperature (θρ)",
                "reference": "Emanuel, K. A., 1994: Atmospheric Convection. "
                "Oxford University Press, 580 pp.",
            }
        )

        # If verbose, print summary statistics for the result
        if verbose:
            self._print_result_summary(theta_rho)

        # Return the computed density potential temperature
        return theta_rho.values if return_array else theta_rho

    def compute_density_potential_temperature_from_dataset(
        self: "ThermodynamicDiagnostics",
        dataset: xr.Dataset,
        time_index: int = 0,
        level: Optional[Union[str, int, float]] = None,
        theta_var: str = DEFAULT_THETA_VAR,
        qv_var: str = DEFAULT_QV_VAR,
        condensate_vars: Optional[List[str]] = None,
        data_type: str = "xarray",
    ) -> xr.DataArray:
        """
        This method computes the density potential temperature (theta_rho) directly from an MPAS dataset by extracting the necessary fields (potential temperature, water vapor mixing ratio, and optionally condensate species) at a specified time index and vertical level. The method handles both xarray and uxarray datasets, allowing for flexible data access patterns. A float level is a pressure in Pa and is interpolated per column, linearly in log(pressure), using each column's own total pressure (NaN where the pressure lies below the ground or above the model top, with 'actual_pressure_level' recorded in the attrs). It also includes error handling for missing variables and invalid level specifications. The resulting theta_rho is returned with CF-compliant attributes, and summary statistics are printed if verbosity is enabled.

        Parameters:
            dataset (xr.Dataset): MPAS dataset containing the necessary variables.
            time_index (int): Time index to select from the dataset (default: 0).
            level (Optional[Union[str, int, float]]): Vertical level specification (index, pressure in Pa, or 'surface'/'top') or None for full 3D (default: None).
            theta_var (str): Name of the potential temperature variable in the dataset (default: 'theta').
            qv_var (str): Name of the water vapor mixing ratio variable in the dataset (default: 'qv').
            condensate_vars (Optional[List[str]]): List of condensate variable names to extract (default: None, which uses all standard condensate variables).
            data_type (str): Type of dataset ('xarray' or 'uxarray') to determine extraction method (default: 'xarray').

        Returns:
            xr.DataArray: Density potential temperature in Kelvin.
        """
        from mpasdiag.processing.utils_datetime import MPASDateTimeUtils

        # Make sure the dataset is provided
        if dataset is None:
            raise ValueError("Dataset not provided. Pass a valid MPAS dataset.")

        # Set default condensate variable names if not provided
        if condensate_vars is None:
            condensate_vars = list(CONDENSATE_VARS)

        # Validate time parameters and get the time dimension name
        time_dim, validated_time_index, _ = MPASDateTimeUtils.validate_time_parameters(
            dataset, time_index, self.verbose
        )

        # Extract theta and check for missing variable
        theta = self._extract_field(
            dataset,
            theta_var,
            time_dim,
            validated_time_index,
            level,
            data_type,
        )

        # Extract qv and check for missing variable
        qv = self._extract_field(
            dataset,
            qv_var,
            time_dim,
            validated_time_index,
            level,
            data_type,
        )

        # Initialize a dictionary to hold any extracted condensate fields
        condensates: Dict[str, xr.DataArray] = {}

        # Loop through the list of condensate variable names and extract those that are present in the dataset.
        for cvar in condensate_vars:
            if cvar in dataset.data_vars:
                condensates[cvar] = self._extract_field(
                    dataset,
                    cvar,
                    time_dim,
                    validated_time_index,
                    level,
                    data_type,
                )
            elif self.verbose:
                logger.debug(
                    "Condensate variable '%s' not in dataset — treating as zero", cvar
                )

        # Compute and return the density potential temperature using the extracted fields
        return self.compute_density_potential_temperature(
            theta=theta,
            qv=qv,
            qc=condensates.get("qc"),
            qr=condensates.get("qr"),
            qi=condensates.get("qi"),
            qs=condensates.get("qs"),
            qg=condensates.get("qg"),
        )

    def compute_virtual_potential_temperature(
        self: "ThermodynamicDiagnostics",
        theta: Any,
        qv: Any,
    ) -> Any:
        """
        This method computes the virtual potential temperature (theta_v) from the potential temperature (theta) and water vapor mixing ratio (qv) using the exact mixing-ratio formula theta_v = theta * (1 + qv/eps) / (1 + qv) with eps = Rd/Rv, which is approximately theta * (1 + 0.608 qv); the earlier theta * (1 + (Rv/Rd) * qv) form overstated the vapour effect by a factor of about 2.6 and is not a virtual-temperature relation. Here qv is the MPAS water vapour mixing ratio (kg/kg), not specific humidity. This diagnostic accounts for the buoyancy effect of water vapor without considering condensate loading, making it a useful comparison to the density potential temperature. Inputs may be numpy- or dask-backed DataArrays (the result is then lazy) or plain arrays (a plain array is returned). The method returns theta_v with CF-compliant attributes (DataArrays) and prints summary statistics if verbosity is enabled.

        Parameters:
            theta (Union[xr.DataArray, np.ndarray]): Potential temperature in Kelvin.
            qv (Union[xr.DataArray, np.ndarray]): Water vapor mixing ratio in kg/kg.

        Returns:
            Union[xr.DataArray, np.ndarray]: Virtual potential temperature in Kelvin, a DataArray with CF-compliant attributes or a plain array when no input is a DataArray.
        """
        # Check that required inputs are provided
        if theta is None or qv is None:
            raise ValueError("Both 'theta' and 'qv' must be provided.")

        return_array = not (
            isinstance(theta, xr.DataArray) or isinstance(qv, xr.DataArray)
        )
        theta, qv = as_dataarray(theta), as_dataarray(qv)

        # Compute virtual potential temperature without condensate loading
        theta_v: xr.DataArray = theta * (1.0 + qv / EPSILON) / (1.0 + qv)

        # Add CF-compliant attributes to the result
        theta_v.attrs.update(
            {
                "units": THETA_RHO_UNITS,
                "standard_name": "virtual_potential_temperature",
                "long_name": "virtual potential temperature",
                "formula": "theta * (1 + qv/eps) / (1 + qv), eps = Rd/Rv, qv = water vapour mixing ratio",
            }
        )

        # If verbose, print summary statistics for the result (skipped for lazy results)
        if self.verbose and not is_lazy(theta_v):
            tv_min, tv_max = float(theta_v.min()), float(theta_v.max())
            tv_mean = float(theta_v.mean())
            logger.debug(
                "Virtual potential temperature range: %.2f to %.2f K (mean: %.2f K)",
                tv_min,
                tv_max,
                tv_mean,
            )

        # Return the computed virtual potential temperature
        return theta_v.values if return_array else theta_v

    def compute_total_water_mixing_ratio(
        self: "ThermodynamicDiagnostics",
        qv: Any,
        qc: Optional[Any] = None,
        qr: Optional[Any] = None,
        qi: Optional[Any] = None,
        qs: Optional[Any] = None,
        qg: Optional[Any] = None,
    ) -> Any:
        """
        This method computes the total water mixing ratio (qt) as the sum of the water vapor mixing ratio (qv) and any available condensate species (qc, qr, qi, qs, qg). If a particular condensate species is not provided, it is treated as zero. Inputs may be numpy- or dask-backed DataArrays (the result is then lazy) or plain arrays (a plain array is returned). The method returns qt with CF-compliant attributes (DataArrays) and prints summary statistics for the inputs and result if verbosity is enabled.

        Parameters:
            qv (Union[xr.DataArray, np.ndarray]): Water vapor mixing ratio in kg/kg.
            qc (Optional[Union[xr.DataArray, np.ndarray]]): Cloud water mixing ratio in kg/kg (default: None).
            qr (Optional[Union[xr.DataArray, np.ndarray]]): Rain water mixing ratio in kg/kg (default: None).
            qi (Optional[Union[xr.DataArray, np.ndarray]]): Cloud ice mixing ratio in kg/kg (default: None).
            qs (Optional[Union[xr.DataArray, np.ndarray]]): Snow mixing ratio in kg/kg (default: None).
            qg (Optional[Union[xr.DataArray, np.ndarray]]): Graupel mixing ratio in kg/kg (default: None).

        Returns:
            Union[xr.DataArray, np.ndarray]: Total water mixing ratio in kg/kg, a DataArray with CF-compliant attributes or a plain array when no input is a DataArray.
        """
        return_array = not any(
            isinstance(x, xr.DataArray) for x in (qv, qc, qr, qi, qs, qg)
        )

        # Start with qv as the base for qt
        qt = as_dataarray(qv).copy(deep=True)

        # Initialize a list to keep track of which condensate species are included
        species_names: List[str] = []

        # Loop through the condensate species and add them to qt if they are provided
        for name, species in [
            ("qc", qc),
            ("qr", qr),
            ("qi", qi),
            ("qs", qs),
            ("qg", qg),
        ]:
            if species is not None:
                qt = qt + as_dataarray(species)
                species_names.append(name)

        # Add CF-compliant attributes to the total water mixing ratio
        qt.attrs.update(
            {
                "units": "kg/kg",
                "standard_name": "total_water_mixing_ratio",
                "long_name": "total water mixing ratio (qv + condensate)",
                "included_species": ", ".join(["qv"] + species_names),
            }
        )

        # If verbose, print summary statistics for the total water mixing ratio (skipped for lazy results)
        if self.verbose and not is_lazy(qt):
            qt_min, qt_max = float(qt.min()), float(qt.max())
            qt_mean = float(qt.mean())
            logger.debug(
                "Total water mixing ratio (qt) range: %.6f to %.6f kg/kg (mean: %.6f kg/kg)",
                qt_min,
                qt_max,
                qt_mean,
            )
            logger.debug(
                "  Included species: qv, %s",
                ", ".join(species_names) if species_names else "none",
            )

        # Return the computed total water mixing ratio
        return qt.values if return_array else qt

    def analyze_density_potential_temperature(
        self: "ThermodynamicDiagnostics",
        theta_rho: xr.DataArray,
        theta: Optional[xr.DataArray] = None,
    ) -> Dict[str, Any]:
        """
        This method provides a diagnostic analysis of the density potential temperature (theta_rho) by computing summary statistics such as minimum, maximum, mean, and standard deviation. If the potential temperature (theta) is also provided, it computes the perturbation (theta_rho - theta) to assess the buoyancy contribution of moisture and provides summary statistics for that as well. The results are returned in a dictionary format with appropriate units and descriptions, and if verbosity is enabled, the statistics are printed in a readable format.

        Parameters:
            theta_rho (xr.DataArray): Density potential temperature in Kelvin.
            theta (Optional[xr.DataArray]): Potential temperature in Kelvin for comparison (default: None).

        Returns:
            Dict[str, Any]: A dictionary containing summary statistics for theta_rho and, if theta is provided, the perturbation (theta_rho - theta) with their respective units and descriptions.
        """
        # Initialize the analysis dictionary to store results
        analysis: Dict[str, Any] = {}

        # Compute summary statistics for theta_rho
        analysis["theta_rho"] = {
            "min": float(theta_rho.min()),
            "max": float(theta_rho.max()),
            "mean": float(theta_rho.mean()),
            "std": float(theta_rho.std()),
            "units": THETA_RHO_UNITS,
        }

        # If potential temperature is provided, compute the perturbation and its statistics
        if theta is not None:
            perturbation = theta_rho - theta
            analysis["perturbation"] = {
                "min": float(perturbation.min()),
                "max": float(perturbation.max()),
                "mean": float(perturbation.mean()),
                "std": float(perturbation.std()),
                "units": THETA_RHO_UNITS,
                "description": "theta_rho - theta (buoyancy contribution of moisture)",
            }

        # If verbose, print the analysis results in a readable format
        if self.verbose:
            tr = analysis["theta_rho"]
            logger.info("Density Potential Temperature Analysis:")
            logger.info(
                "  theta_rho: %.2f to %.2f K (mean: %.2f, std: %.2f)",
                tr["min"],
                tr["max"],
                tr["mean"],
                tr["std"],
            )
            if "perturbation" in analysis:
                p = analysis["perturbation"]
                logger.info(
                    "  Perturbation (theta_rho - theta): %.4f to %.4f K (mean: %.4f)",
                    p["min"],
                    p["max"],
                    p["mean"],
                )

        # Return the analysis results as a dictionary
        return analysis

    def compute_cold_pool_strength(
        self: "ThermodynamicDiagnostics",
        theta_rho: xr.DataArray,
        theta_rho_env: Optional[Union[float, xr.DataArray]] = None,
        cold_pool_depth: float = 1000.0,
        g: float = G,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        This method computes the cold-pool strength parameter (C) based on the buoyancy perturbation of the density potential temperature relative to an environmental reference value. This diagnostic is inspired by the Rotunno-Klemp-Weisman (RKW) theory for squall line dynamics, where the cold-pool strength is a key factor in determining the balance between low-level inflow and vertical shear.

        The buoyancy perturbation is defined as:

            B = g * (theta_rho - theta_rho_env) / theta_rho_env   [m s-2]

        and the cold-pool strength parameter is approximated by assuming
        a uniform buoyancy deficit over an effective depth H:

            C = sqrt(-2 * min(B, 0) * H)                          [m s-1]

        C is zero wherever theta_rho >= theta_rho_env (no cold pool).

        Parameters:
            theta_rho (xr.DataArray): Density potential temperature in Kelvin.
            theta_rho_env (Optional[Union[float, xr.DataArray]]): Environmental reference value for theta_rho. If None, the domain mean of theta_rho will be used (default: None).
            cold_pool_depth (float): Effective depth of the cold pool in meters (default: 1000 m).
            g (float): Gravitational acceleration in m/s² (default: 9.81 m/s²).

        Returns:
            Tuple[xr.DataArray, xr.DataArray]: A tuple containing the cold-pool strength parameter (C) in m/s and the buoyancy perturbation (B) in m/s², both as xarray DataArrays with appropriate attributes.
        """
        # Determine the environmental reference value for theta_rho
        if theta_rho_env is None:
            env_ref: Union[float, xr.DataArray] = float(theta_rho.mean())
            if self.verbose:
                logger.debug(
                    "Cold-pool computation: using domain-mean theta_rho_env = %.2f K",
                    env_ref,
                )
        elif isinstance(theta_rho_env, (int, float)):
            env_ref = float(theta_rho_env)
        else:
            env_ref = theta_rho_env

        # Compute the buoyancy perturbation relative to the environmental reference
        buoyancy = g * (theta_rho - env_ref) / env_ref

        # Add CF-compliant attributes to the buoyancy field
        buoyancy.attrs.update(
            {
                "units": "m s-2",
                "standard_name": "air_buoyancy",
                "long_name": "Buoyancy perturbation g*(theta_rho - theta_rho_env)/theta_rho_env",
            }
        )

        # Compute the cold-pool strength: C = sqrt(-2 * min(B, 0) * H)
        negative_b = xr.apply_ufunc(np.minimum, buoyancy, 0.0, keep_attrs=False)
        cold_pool_strength = xr.apply_ufunc(
            np.sqrt, -2.0 * negative_b * cold_pool_depth, keep_attrs=False
        )

        # Add CF-compliant attributes to the cold-pool strength field
        cold_pool_strength.attrs.update(
            {
                "units": "m s-1",
                "standard_name": "cold_pool_strength",
                "long_name": f"RKW cold-pool strength C (H={cold_pool_depth:.0f} m)",
            }
        )

        # If verbose, print summary statistics for the buoyancy and cold-pool strength fields
        if self.verbose:
            b_min = float(buoyancy.min())
            b_max = float(buoyancy.max())
            c_max = float(cold_pool_strength.max())
            cold_pool_mask = cold_pool_strength > 0
            n_cold = int(cold_pool_mask.sum())
            c_mean_cold = (
                float(cold_pool_strength.where(cold_pool_mask).mean())
                if n_cold > 0
                else 0.0
            )
            logger.info("Cold-pool diagnostics (H = %.0f m):", cold_pool_depth)
            logger.info("  Buoyancy B range: %.4f to %.4f m/s²", b_min, b_max)
            logger.info(
                "  Cold-pool grid points: %s / %s",
                f"{n_cold:,}",
                f"{theta_rho.size:,}",
            )
            logger.info("  Cold-pool strength C max: %.2f m/s", c_max)
            if n_cold > 0:
                logger.info("  Mean C in cold-pool region: %.2f m/s", c_mean_cold)

        # Return the cold-pool strength and buoyancy fields as a tuple
        return cold_pool_strength, buoyancy

    def _extract_field(
        self: "ThermodynamicDiagnostics",
        dataset: xr.Dataset,
        var_name: str,
        time_dim: str,
        time_index: int,
        level: Optional[Union[str, int, float]],
        data_type: str,
    ) -> xr.DataArray:
        """
        This helper method extracts a specified variable from the dataset at a given time index and optionally at a specified vertical level. It handles both xarray and uxarray datasets, checking for the presence of the variable and applying appropriate selection methods based on the dataset type. If a vertical level is specified, it resolves the level to an index and selects that level from the data. The method also includes error handling for missing variables and invalid level specifications, ensuring that users receive informative messages when issues arise.

        Parameters:
            dataset (xr.Dataset): Source dataset containing the variable.
            var_name (str): Name of the variable to extract.
            time_dim (str): Name of the time dimension in the dataset.
            time_index (int): Time index to select.
            level (Optional[Union[str, int, float]]): Vertical level specification (index, pressure in Pa, or 'surface'/'top') or None for full 3D.
            data_type (str): Type of dataset ('xarray' or 'uxarray') to determine extraction method.

        Returns:
            xr.DataArray: Extracted variable at the specified time and level.
        """
        # Check if the variable exists in the dataset
        if var_name not in dataset.data_vars:
            raise ValueError(
                f"Variable '{var_name}' not found in dataset. "
                f"Available: {list(dataset.data_vars)[:20]}"
            )

        # Extract the variable at the specified time index
        if data_type == "uxarray" and hasattr(dataset, "__getitem__"):
            data = dataset[var_name][time_index]
        else:
            if type(dataset) is not xr.Dataset and isinstance(dataset, xr.Dataset):
                plain_ds = xr.Dataset(
                    dict(dataset.data_vars), coords=dataset.coords, attrs=dataset.attrs
                )
            else:
                plain_ds = dataset
            data = plain_ds[var_name].isel({time_dim: time_index})

        # A pressure level is interpolated in every column separately (log-p); NaN below ground / above the model top
        if (
            level is not None
            and isinstance(level, (float, np.floating))
            and not isinstance(level, bool)
            and self._resolve_vertical_dim(data) is not None
        ):
            return interpolate_to_pressure(
                dataset, var_name, float(level), time_dim, time_index
            )

        # If a vertical level is specified, resolve it to an index and select that level
        if level is not None:
            vert_dim = self._resolve_vertical_dim(data)
            if vert_dim is not None:
                level_idx = self._resolve_level_index(
                    dataset, var_name, level, time_dim, time_index, vert_dim
                )
                data = data.isel({vert_dim: level_idx})

        # If the data is still a dask array, compute it to get the actual values
        if hasattr(data, "compute"):
            data = cast(Any, data).compute()

        # Return the extracted data array
        return data

    @staticmethod
    def _resolve_vertical_dim(data: xr.DataArray) -> Optional[str]:
        """
        This helper method checks the dimensions of the provided DataArray to identify the vertical dimension, which is typically named 'nVertLevels' or 'nVertLevelsP1' in MPAS datasets. If a standard vertical dimension is found, its name is returned; otherwise, None is returned, indicating that no vertical dimension could be identified. This method allows for flexible handling of datasets with different vertical dimension naming conventions while providing a clear mechanism for identifying the vertical coordinate when present.

        Parameters:
            data (xr.DataArray): DataArray to inspect.

        Returns:
            Optional[str]: Vertical dimension name or None.
        """
        # Check for common vertical dimension names
        for dim in ("nVertLevels", "nVertLevelsP1"):
            if dim in data.dims:
                return dim

        # If no standard vertical dimension is found, return None
        return None

    def _resolve_level_index(
        self: "ThermodynamicDiagnostics",
        dataset: xr.Dataset,
        var_name: str,
        level: Union[str, int, float],
        time_dim: str,
        time_index: int,
        vert_dim: str,
    ) -> int:
        """
        This helper method resolves a vertical level specification to an index for selecting data from the dataset. The level specification can be an integer index, a float representing pressure in Pascals, or a string indicating 'surface' or 'top'. The method includes error handling for invalid specifications and checks that the resolved index is within the bounds of the available vertical levels in the dataset. A pressure level cannot be reduced to one model level index (the model level at a given pressure differs from column to column), so a float level raises here; _extract_field handles pressure levels by per-column log-pressure interpolation instead.

        Parameters:
            dataset (xr.Dataset): Dataset containing the vertical coordinate information.
            var_name (str): Name of the variable for error messages.
            level (Union[str, int, float]): Vertical level specification.
            time_dim (str): Name of the time dimension for selecting pressure fields if needed.
            time_index (int): Time index for selecting pressure fields if needed.
            vert_dim (str): Name of the vertical dimension in the dataset.

        Returns:
            int: Resolved vertical level index.
        """
        # Get the maximum number of vertical levels from the dataset for bounds checking
        max_levels = dataset.sizes.get(vert_dim, 0)

        # If the level is an integer, use it directly as an index (with bounds checking)
        if isinstance(level, (int, np.integer)) and not isinstance(level, bool):
            if level >= max_levels:
                raise ValueError(
                    f"Level {level} exceeds available levels ({max_levels}) "
                    f"for '{var_name}'."
                )
            return int(level)

        # Pressure levels are interpolated per column in _extract_field and never reduced to a model level index
        if isinstance(level, (float, np.floating)) and not isinstance(level, bool):
            raise ValueError(
                f"Pressure level {level} Pa for '{var_name}' has no single model level index; "
                f"use _extract_field, which interpolates to the pressure in every column."
            )

        # Handle string specifications for surface and top levels
        if isinstance(level, str):
            if level.lower() == "surface":
                return 0
            if level.lower() == "top":
                return int(max_levels) - 1
            raise ValueError(f"Unknown level specification: '{level}'")

        # If we reach this point, the level specification type is invalid
        raise ValueError(f"Invalid level specification type: {type(level)}")

    @staticmethod
    def _vert_dim(da: xr.DataArray) -> Optional[str]:
        """
        This helper method checks the dimensions of the provided DataArray to identify the vertical dimension, which is typically named 'nVertLevels' or 'nVertLevelsP1' in MPAS datasets. If a standard vertical dimension is found, its name is returned; otherwise, None is returned, indicating that no vertical dimension could be identified. This method allows for flexible handling of datasets with different vertical dimension naming conventions while providing a clear mechanism for identifying the vertical coordinate when present.

        Parameters:
            da (xr.DataArray): DataArray to inspect.

        Returns:
            Optional[str]: Vertical dimension name or None.
        """
        # Check for common vertical dimension names
        for name in ("nVertLevels", "nVertLevelsP1"):
            if name in da.dims:
                return name

        # If no standard vertical dimension is found, return None
        return None

    def _print_input_summary(
        self: "ThermodynamicDiagnostics",
        theta: xr.DataArray,
        qv: xr.DataArray,
        qt: xr.DataArray,
    ) -> None:
        """
        This helper method prints summary statistics for the input fields (potential temperature, water vapor mixing ratio, and total water mixing ratio) used in the computation of density potential temperature. It checks for the presence of a vertical dimension to provide context on the range of potential temperature values across the full column versus surface level, and it reports the overall range for each input variable. This information can help users understand the characteristics of the input data and identify any potential issues or interesting features before performing the calculation.

        Parameters:
            theta (xr.DataArray): Potential temperature in Kelvin.
            qv (xr.DataArray): Water vapor mixing ratio in kg/kg.
            qt (xr.DataArray): Total water mixing ratio in kg/kg.

        Returns:
            None
        """
        # If not verbose, skip printing the input summary
        if not self.verbose:
            return

        # Check for the presence of a vertical dimension
        vdim = self._vert_dim(theta)
        is_3d = vdim is not None

        logger.debug("Computing density potential temperature (Emanuel 1994):")

        # Print the range of potential temperature, distinguishing between full column and surface level if 3D
        if is_3d:
            logger.debug(
                "  theta range (full column) : %.2f to %.2f K",
                float(theta.min()),
                float(theta.max()),
            )
            sfc = theta.isel({vdim: 0})
            logger.debug(
                "  theta range (surface lev) : %.2f to %.2f K",
                float(sfc.min()),
                float(sfc.max()),
            )
        else:
            logger.debug(
                "  theta range : %.2f to %.2f K",
                float(theta.min()),
                float(theta.max()),
            )

        # Print the range of water vapor mixing ratio
        logger.debug(
            "  qv range    : %.6f to %.6f kg/kg",
            float(qv.min()),
            float(qv.max()),
        )

        # Print the range of total water mixing ratio
        logger.debug(
            "  qt range    : %.6f to %.6f kg/kg",
            float(qt.min()),
            float(qt.max()),
        )

        # Print the Rv/Rd ratio used in the calculation
        logger.debug("  Rv/Rd       : %.4f", RV_OVER_RD)

        # If 3D, note that the full-column stats include all vertical levels
        if is_3d:
            n = theta.sizes[vdim]
            logger.debug(
                "  Note: full-column stats span all %d vertical levels "
                "including the stratosphere (theta can reach 1 000+ K there).",
                n,
            )

    def _print_result_summary(
        self: "ThermodynamicDiagnostics", theta_rho: xr.DataArray
    ) -> None:
        """
        This helper method prints summary statistics for the computed density potential temperature (theta_rho). It checks for the presence of a vertical dimension to provide context on the range of theta_rho values across the full column versus surface level, and it reports the overall range, mean, and standard deviation for theta_rho. This information can help users understand the characteristics of the computed density potential temperature and identify any potential issues or interesting features in the result.

        Parameters:
            theta_rho (xr.DataArray): Computed density potential temperature.

        Returns:
            None
        """
        # If not verbose, skip printing the result summary
        if not self.verbose:
            return

        # Check for the presence of a vertical dimension
        vdim = self._vert_dim(theta_rho)
        is_3d = vdim is not None

        if is_3d:
            # Print the range of theta_rho for the full column
            logger.debug(
                "  theta_rho range (full column) : %.2f to %.2f K",
                float(theta_rho.min()),
                float(theta_rho.max()),
            )

            # Extract the surface level of theta_rho for additional statistics
            sfc = theta_rho.isel({vdim: 0})

            # Print the range of theta_rho at the surface level
            logger.debug(
                "  theta_rho range (surface lev) : %.2f to %.2f K",
                float(sfc.min()),
                float(sfc.max()),
            )

            # Print the mean and standard deviation of theta_rho at the surface level
            logger.debug("  theta_rho mean  (surface lev) : %.2f K", float(sfc.mean()))
            logger.debug("  theta_rho std   (surface lev) : %.4f K", float(sfc.std()))
        else:
            # Print the range, mean, and standard deviation of theta_rho for 2D data
            tr_min, tr_max = float(theta_rho.min()), float(theta_rho.max())
            logger.debug("  theta_rho range: %.2f to %.2f K", tr_min, tr_max)
            logger.debug("  theta_rho mean : %.2f K", float(theta_rho.mean()))
            logger.debug("  theta_rho std  : %.4f K", float(theta_rho.std()))

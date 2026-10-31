#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Core Diagnostics Module: Sounding Diagnostics

This module provides functionality to extract vertical sounding profiles from MPAS model output at specified geographic locations and time indices, and to compute a comprehensive set of thermodynamic and severe weather indices using MetPy. The SoundingDiagnostics class includes methods for finding the nearest grid cell to a target location, extracting key atmospheric variables (pressure, temperature, dewpoint, wind) at all vertical levels, and calculating indices such as CAPE, CIN, lifted index, K-index, and more. The module also includes error handling to ensure robust computation of indices even when some profile data may be missing or invalid. This allows users to analyze the thermodynamic structure of the atmosphere in MPAS simulations and assess severe weather potential at specific locations.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: March 2026
Version: 1.0.0
"""

# Load standard libraries and type hints
import contextlib
import importlib.util
import sys
import warnings

import numpy as np
import xarray as xr
from scipy.spatial import KDTree as cKDTree
from typing import Dict, Any, Iterator, Optional, Tuple, cast

# Import MPASdiag utilities and base classes
from mpasdiag.processing.utils_datetime import MPASDateTimeUtils
from mpasdiag.processing.utils_geog import MPASGeographicUtils
from mpasdiag.processing.processors_3d import MPAS3DProcessor
from mpasdiag.processing.base import MPASBaseProcessor
from mpasdiag.processing.constants import (
    P0_REF_PA,
    KAPPA,
    EPSILON_RD_RV,
    NETCDF_ENGINE,
)
from mpasdiag.processing.utils_logger import get_logger

logger = get_logger(__name__)

_NEAREST_CELL_TREE_CACHE: Dict[Any, Any] = {}
_NEAREST_CELL_TREE_CACHE_MAX: int = 4
_GRID_COORD_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
_GRID_COORD_CACHE_MAX: int = 4

_SOUNDING_COORD_NAMES = [
    "lonCell",
    "longitude",
    "lon",
    "latCell",
    "latitude",
    "lat",
]

# metpy.calc and metpy.units take about a second to import, which every CLI process, worker server and
# MPI rank would otherwise pay even when no sounding is computed, so they are imported on first use
HAS_METPY = importlib.util.find_spec("metpy") is not None


def __getattr__(name: str) -> Any:
    """
    This module-level attribute hook imports MetPy the first time the module attributes 'mpcalc' (metpy.calc) or 'munits' (the MetPy unit registry) are read, and stores them as ordinary module attributes so later reads, and tests that patch them, see plain values.

    Parameters:
        name (str): The attribute being read.

    Returns:
        Any: The metpy.calc module or the MetPy unit registry.
    """
    if name not in ("mpcalc", "munits"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import metpy.calc
    from metpy.units import units

    globals().update(mpcalc=metpy.calc, munits=units)
    return globals()[name]


@contextlib.contextmanager
def _quiet_out_of_bounds() -> Iterator[None]:
    """
    This context manager silences MetPy's "Interpolation point out of data bounds" warning, which MetPy emits when an index is undefined for the profile (for example a layer deeper than the sounding); the result is NaN, which the callers report as None, so the warning would only repeat that on the console.

    Returns:
        Iterator[None]: Context manager yielding nothing.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Interpolation point out of data bounds"
        )
        yield


def _load_metpy() -> Tuple[Any, Any]:
    """
    This helper returns metpy.calc and the MetPy unit registry, or (None, None) when MetPy is not installed or is installed but fails to import (for example a pint/numpy version mismatch), so the caller falls back to the MetPy-free estimates instead of raising.

    Returns:
        Tuple[Any, Any]: (metpy.calc, unit registry), or (None, None) when MetPy is unusable.
    """
    if not HAS_METPY:
        return None, None
    try:
        return _metpy_calc(), _metpy_units()
    except ImportError as exc:
        logger.warning("MetPy is installed but could not be imported: %s", exc)
        return None, None


def _metpy_calc() -> Any:
    """
    This helper returns metpy.calc through the module attribute, importing MetPy on first use.

    Returns:
        Any: The metpy.calc module, or whatever a test patched in its place.
    """
    return sys.modules[__name__].mpcalc


def _metpy_units() -> Any:
    """
    This helper returns the MetPy unit registry through the module attribute, importing MetPy on first use.

    Returns:
        Any: The MetPy unit registry, or whatever a test patched in its place.
    """
    return sys.modules[__name__].munits


class SoundingDiagnostics:
    """Computes sounding-related diagnostics for MPAS model output, including profile extraction, thermodynamic index calculation, and data quality checks."""

    def __init__(self: "SoundingDiagnostics", verbose: bool = True) -> None:
        """
        This function initializes the SoundingDiagnostics class with optional verbose output. The verbose flag controls whether detailed messages about the sounding extraction process and index calculations are printed to the console. When enabled, it provides insights into the nearest cell selection, profile characteristics, and any issues encountered during computations. This can be helpful for debugging and understanding the results, especially when working with complex MPAS datasets.

        Parameters:
            verbose (bool): Enable verbose output messages during sounding extraction (default: True).

        Returns:
            None
        """
        # Store the verbose flag for use in other methods
        self.verbose = verbose

    def extract_sounding_profile(
        self: "SoundingDiagnostics",
        processor: "MPAS3DProcessor",
        lon: float,
        lat: float,
        time_index: int = 0,
        cell_index: Optional[int] = None,
        station_coords: Optional[Tuple[float, float]] = None,
        max_distance_km: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        This function extracts the sounding profile at a specified longitude, latitude, and time index from the MPAS dataset using the provided processor. It first checks that the dataset is loaded, then retrieves the grid coordinates and finds the nearest cell to the target location. It extracts the pressure profile, temperature profile (converting from potential temperature if necessary), dewpoint profile (computing from mixing ratio if necessary), wind profiles (u and v components), and optionally the height profile (taken from the dataset, or from 'zgrid' in the grid file averaged to the mass levels, since mpasout files do not hold it; it is MSL and the severe-weather indices convert it to above ground level). The requested point is validated (finite longitude, latitude within [-90, 90]) and, after the nearest-cell search, the great-circle distance to the chosen cell is compared with a threshold: max_distance_km when given, otherwise three times the distance from the chosen cell to its nearest neighbouring cell. A point farther away than that (for example outside a regional domain) raises a ValueError naming the requested point, the cell and the distance, instead of silently returning the profile of a far-away boundary cell. The function sorts the profiles from surface to top and filters out any non-finite values. Finally, it returns a dictionary containing all extracted variables along with metadata about the station location and time index. This allows users to analyze the vertical structure of the atmosphere at specific locations in their MPAS simulations. The nearest-cell search depends only on the grid and the station, never on the timestep, so batch callers (one sounding per timestep at a fixed station) can resolve the cell once and pass cell_index together with station_coords; the function then skips loading the full grid coordinates and the KDTree query entirely for that call.

        Parameters:
            processor (MPAS3DProcessor): The MPAS data processor instance containing the dataset and grid information.
            lon (float): Target longitude in degrees for sounding extraction.
            lat (float): Target latitude in degrees for sounding extraction.
            time_index (int): Time index to extract the profile from (default: 0).
            cell_index (Optional[int]): Pre-resolved nearest-cell index; skips the grid-coordinate load and nearest-cell search when given together with station_coords (default: None).
            station_coords (Optional[Tuple[float, float]]): Pre-resolved (station_lon, station_lat) of the cell at cell_index in degrees; required for the fast path because the station metadata otherwise comes from the loaded grid coordinates (default: None).
            max_distance_km (Optional[float]): Largest allowed distance in km between the requested point and the cell used; None uses three times the local cell spacing. On the fast path the default check is skipped (the cell was resolved and checked by the caller) and only an explicit max_distance_km is enforced (default: None).

        Returns:
            Dict[str, Any]: A dictionary containing the extracted sounding profile with keys such as 'pressure', 'temperature', 'dewpoint', 'u_wind', 'v_wind', 'height', 'surface_height' (terrain height in m, or None when unavailable; pass it as surface_height_m to compute_thermodynamic_indices), 'station_lon', 'station_lat', 'requested_lon', 'requested_lat', 'distance_km', 'cell_index', and 'time_index'.
        """
        # Raise an error if the dataset is not loaded in the processor
        if processor.dataset is None:
            raise ValueError("Dataset not loaded. Call load_3d_data() first.")

        # Get the plain xarray dataset from the processor for direct variable access
        ds = MPASBaseProcessor._get_plain_dataset(processor.dataset)

        # Validate the time index and get the corresponding time dimension name and validated time index
        time_dim, validated_time_index, _ = MPASDateTimeUtils.validate_time_parameters(
            processor.dataset, time_index, self.verbose
        )

        # Validate the requested point and resolve the cell (checking its distance from the point)
        cell_idx, station_lon, station_lat, dist_km = self._resolve_station(
            processor, lon, lat, cell_index, station_coords, max_distance_km
        )

        # Print nearest cell information if verbose mode is enabled
        if self.verbose:
            logger.debug(
                "Nearest cell %d: (%.4f°, %.4f°), distance %.1f km from target",
                cell_idx,
                station_lon,
                station_lat,
                dist_km,
            )

        # Extract pressure profile, converting from Pa to hPa for consistency with meteorological conventions
        pressure_pa = self._extract_pressure_profile(
            ds, time_dim, validated_time_index, cell_idx
        )
        pressure_hpa = pressure_pa / 100.0

        # Extract temperature profile, converting from potential temperature if necessary
        temperature_c = self._extract_temperature_profile(
            ds, time_dim, validated_time_index, cell_idx, pressure_pa
        )

        # Extract dewpoint profile if available, or compute from mixing ratio if possible
        dewpoint_c = self._extract_dewpoint_profile(
            ds, time_dim, validated_time_index, cell_idx, pressure_pa
        )

        # Extract wind profiles if available
        u_kt, v_kt = self._extract_wind_profiles(
            ds, time_dim, validated_time_index, cell_idx
        )

        # Extract height profile if available; real MPAS output carries zgrid only in the static grid file
        height_m = self._extract_height_profile(
            ds, time_dim, validated_time_index, cell_idx
        )
        if height_m is None:
            height_m = self._extract_height_profile_from_grid(
                getattr(processor, "grid_file", None),
                cell_idx,
                len(pressure_hpa),
            )

        # Terrain height (MSL) of the cell, used as the ground reference of the above-ground layers
        surface_height_m = self._extract_surface_height(
            ds, getattr(processor, "grid_file", None), cell_idx
        )

        # Sort profiles surface-to-top and drop levels with non-finite P or T
        (
            pressure_hpa,
            temperature_c,
            dewpoint_c,
            u_kt,
            v_kt,
            height_m,
        ) = self._order_and_clean_profiles(
            pressure_hpa, temperature_c, dewpoint_c, u_kt, v_kt, height_m
        )

        # Print profile summary if verbose mode is enabled
        if self.verbose:
            logger.debug(
                "Sounding profile: %d levels, P range %.1f–%.1f hPa, T range %.1f–%.1f °C",
                len(pressure_hpa),
                pressure_hpa[-1],
                pressure_hpa[0],
                float(np.nanmin(temperature_c)),
                float(np.nanmax(temperature_c)),
            )

        # Return the extracted profile and metadata as a dictionary
        return {
            "pressure": pressure_hpa,
            "temperature": temperature_c,
            "dewpoint": dewpoint_c,
            "u_wind": u_kt,
            "v_wind": v_kt,
            "height": height_m,
            "surface_height": surface_height_m,
            "station_lon": station_lon,
            "station_lat": station_lat,
            "requested_lon": float(lon),
            "requested_lat": float(lat),
            "distance_km": float(dist_km),
            "cell_index": cell_idx,
            "time_index": validated_time_index,
        }

    def _resolve_station(
        self: "SoundingDiagnostics",
        processor: "MPAS3DProcessor",
        lon: float,
        lat: float,
        cell_index: Optional[int],
        station_coords: Optional[Tuple[float, float]],
        max_distance_km: Optional[float],
    ) -> Tuple[int, float, float, float]:
        """
        This function validates the requested point (finite longitude, latitude within [-90, 90], non-negative max_distance_km), resolves the cell to use (a pre-resolved cell_index with station_coords, or the nearest cell of the grid) and checks the great-circle distance from the point to that cell. The threshold is max_distance_km when given, otherwise three times the distance from the chosen cell to its nearest neighbouring cell; a larger distance raises a ValueError naming the requested point, the cell and the distance. On the pre-resolved fast path the grid is not loaded, so only an explicit max_distance_km is enforced.

        Parameters:
            processor (MPAS3DProcessor): Processor holding the grid file.
            lon (float): Requested longitude in degrees.
            lat (float): Requested latitude in degrees.
            cell_index (Optional[int]): Pre-resolved cell index, or None.
            station_coords (Optional[Tuple[float, float]]): Pre-resolved (lon, lat) of that cell, or None.
            max_distance_km (Optional[float]): Largest allowed distance in km, or None for the default.

        Returns:
            Tuple[int, float, float, float]: Cell index, cell longitude, cell latitude and distance to the requested point in km.
        """
        if not (np.isfinite(lon) and np.isfinite(lat)):
            raise ValueError(f"Requested location ({lon}, {lat}) must be finite.")
        if not -90.0 <= lat <= 90.0:
            raise ValueError(f"Requested latitude {lat} is outside [-90, 90] degrees.")
        if max_distance_km is not None and not (
            np.isfinite(max_distance_km) and max_distance_km >= 0
        ):
            raise ValueError("max_distance_km must be a non-negative number.")

        if cell_index is not None and station_coords is not None:
            cell_idx = int(cell_index)
            station_lon, station_lat = float(station_coords[0]), float(
                station_coords[1]
            )
            dist_km = self._haversine_km(lon, lat, station_lon, station_lat)
            if max_distance_km is not None and dist_km > max_distance_km:
                raise ValueError(
                    f"Requested point ({lon:.4f}, {lat:.4f}) is {dist_km:.1f} km from "
                    f"the nearest cell {cell_idx} ({station_lon:.4f}, {station_lat:.4f}), "
                    f"more than max_distance_km={max_distance_km:g}."
                )
        else:
            grid_lon, grid_lat = self._load_grid_coordinates(processor)
            cell_idx = self._find_nearest_cell(grid_lon, grid_lat, lon, lat)

            # Get the longitude and latitude of the nearest cell for metadata
            station_lon = float(grid_lon[cell_idx])
            station_lat = float(grid_lat[cell_idx])

            # Reject points that are not covered by the grid (e.g. outside a regional domain)
            dist_km = self._haversine_km(lon, lat, station_lon, station_lat)
            if max_distance_km is not None:
                threshold_km = float(max_distance_km)
                threshold_desc = f"max_distance_km={threshold_km:g}."
            else:
                spacing_km = self._nearest_neighbor_distance_km(
                    grid_lon, grid_lat, cell_idx
                )
                threshold_km = 3.0 * spacing_km
                threshold_desc = (
                    f"3 x the local cell spacing ({spacing_km:.1f} km) = {threshold_km:.1f} km; "
                    "the point lies outside the model domain."
                )
            if dist_km > threshold_km:
                raise ValueError(
                    f"Requested point ({lon:.4f}, {lat:.4f}) is {dist_km:.1f} km from "
                    f"the nearest cell {cell_idx} ({station_lon:.4f}, {station_lat:.4f}), "
                    f"more than {threshold_desc}"
                )

        return cell_idx, station_lon, station_lat, dist_km

    @staticmethod
    def _order_and_clean_profiles(
        pressure_hpa: np.ndarray,
        temperature_c: np.ndarray,
        dewpoint_c: np.ndarray,
        u_kt: Optional[np.ndarray],
        v_kt: Optional[np.ndarray],
        height_m: Optional[np.ndarray],
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        This function sorts the sounding profiles from surface to top by descending pressure and drops any levels with non-finite pressure or temperature values. It takes the pressure, temperature, dewpoint, wind, and height profiles as input, sorts them according to pressure, and then filters out any levels where pressure or temperature is not finite. The function ensures that all profiles remain aligned after sorting and filtering. This is important for accurate index calculations later on, as non-finite values can cause errors in the computations. The output is the cleaned and sorted profiles ready for further analysis.

        Parameters:
            pressure_hpa (np.ndarray): Pressure profile in hPa.
            temperature_c (np.ndarray): Temperature profile in °C.
            dewpoint_c (np.ndarray): Dewpoint profile in °C.
            u_kt (Optional[np.ndarray]): Zonal wind profile in knots, or None.
            v_kt (Optional[np.ndarray]): Meridional wind profile in knots, or None.
            height_m (Optional[np.ndarray]): Geometric height profile in m, or None.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]: The (pressure, temperature, dewpoint, u_wind, v_wind, height) profiles sorted surface-to-top with non-finite levels removed.
        """

        def _take(arr: Optional[np.ndarray], idx: np.ndarray) -> Optional[np.ndarray]:
            """
            This helper function takes an array and an index array, returning the indexed values if the array is not None, or None if the input array is None. This is used to apply the same sorting and filtering indices to the optional wind and height profiles without needing to check for their presence multiple times.

            Parameters:
                arr (Optional[np.ndarray]): The input array to index, or None.
                idx (np.ndarray): The indices to take from the array.

            Returns:
                Optional[np.ndarray]: The indexed array if input is not None, otherwise None.
            """
            return arr[idx] if arr is not None else None

        # Sort surface-to-top by descending pressure
        sort_idx = np.argsort(pressure_hpa)[::-1]
        pressure_hpa = pressure_hpa[sort_idx]
        temperature_c = temperature_c[sort_idx]
        dewpoint_c = dewpoint_c[sort_idx]
        u_kt = _take(u_kt, sort_idx)
        v_kt = _take(v_kt, sort_idx)
        height_m = _take(height_m, sort_idx)

        # Drop levels with non-finite pressure or temperature
        valid = np.isfinite(pressure_hpa) & np.isfinite(temperature_c)

        if not np.all(valid):
            pressure_hpa = pressure_hpa[valid]
            temperature_c = temperature_c[valid]
            dewpoint_c = dewpoint_c[valid]
            u_kt = _take(u_kt, valid)
            v_kt = _take(v_kt, valid)
            height_m = _take(height_m, valid)

        return pressure_hpa, temperature_c, dewpoint_c, u_kt, v_kt, height_m

    def compute_dewpoint_from_mixing_ratio(
        self: "SoundingDiagnostics",
        mixing_ratio: np.ndarray,
        pressure_pa: np.ndarray,
    ) -> np.ndarray:
        """
        This function computes the dewpoint temperature in degrees Celsius from the water vapour mixing ratio (in kg/kg) and total pressure (in Pa) using the Magnus formula. It first calculates the vapour pressure from the mixing ratio and total pressure, then applies the Magnus formula to convert vapour pressure to dewpoint temperature. The function ensures that the input arrays are treated as floating-point values for accurate calculations and clips the vapour pressure to avoid issues with logarithm of zero. This allows for deriving the dewpoint profile when only mixing ratio and pressure are available in the MPAS dataset.

        Parameters:
            mixing_ratio (np.ndarray): Water vapour mixing ratio in kg/kg.
            pressure_pa (np.ndarray): Total pressure in Pa.

        Returns:
            np.ndarray: Dewpoint temperature in °C.
        """
        # Ensure inputs are treated as floating-point arrays for accurate calculations
        mixing_ratio_f64 = np.asarray(mixing_ratio, dtype=np.float64)
        pressure_f64 = np.asarray(pressure_pa, dtype=np.float64)

        # Vapour pressure: e = q * p / (epsilon + q)
        vapour_pressure = np.clip(
            mixing_ratio_f64 * pressure_f64 / (EPSILON_RD_RV + mixing_ratio_f64),
            1e-10,
            None,
        )

        # Magnus formula: compute dewpoint in °C from vapour pressure in Pa
        ln_ratio = np.log(vapour_pressure / 611.2)
        dewpoint_c = 243.5 * ln_ratio / (17.67 - ln_ratio)

        # Return the computed dewpoint profile in degrees Celsius
        return cast(np.ndarray, dewpoint_c)

    def compute_thermodynamic_indices(
        self: "SoundingDiagnostics",
        pressure_hpa: np.ndarray,
        temperature_c: np.ndarray,
        dewpoint_c: np.ndarray,
        u_wind_kt: Optional[np.ndarray] = None,
        v_wind_kt: Optional[np.ndarray] = None,
        height_m: Optional[np.ndarray] = None,
        surface_height_m: Optional[float] = None,
    ) -> Dict[str, Optional[float]]:
        """
        This function computes a comprehensive set of thermodynamic and severe weather indices from the provided sounding profile variables using MetPy's calculation functions. It calculates indices such as CAPE, CIN, lifted index, K-index, total totals, showalter index, cross totals, precipitable water, wet bulb zero height (in meters above the terrain when surface_height_m is given, otherwise above the lowest profile level, i.e. approximately above ground level, not MSL), and various shear-related indices if wind data is available. The function uses helper methods to safely call MetPy functions and handle any exceptions that may arise during calculations, returning None for indices that cannot be computed due to data limitations. The inputs are validated first: profiles of different lengths, pressures outside the hPa range (any value above 1100 or not positive, e.g. Pa) and temperatures that look like Kelvin (above 80 degC) raise a ValueError instead of producing meaningless numbers; a profile ordered top-first (ascending pressure) is reordered surface-first; a dewpoint above the temperature is clipped to the temperature with a warning. This allows for robust analysis of the sounding profile's thermodynamic structure and severe weather potential even when some profile data may be missing or invalid.

        Parameters:
            pressure_hpa (np.ndarray): Pressure profile in hPa.
            temperature_c (np.ndarray): Temperature profile in °C.
            dewpoint_c (np.ndarray): Dewpoint profile in °C.
            u_wind_kt (Optional[np.ndarray]): U-component of wind profile in knots (optional).
            v_wind_kt (Optional[np.ndarray]): V-component of wind profile in knots (optional).
            height_m (Optional[np.ndarray]): Height profile in meters (optional).
            surface_height_m (Optional[float]): Terrain height in meters in the same datum as height_m (optional); when given and below the lowest profile level, the wet bulb zero height and the 0-1 km / 0-6 km layers are measured from it (a surface level carrying the lowest level's wind is added), otherwise they are measured from the lowest profile level.

        Returns:
            Dict[str, Optional[float]]: A dictionary containing computed indices such as 'cape', 'cin', 'lifted_index', 'k_index', etc., with float values on success or None for indices that could not be computed.
        """
        # Validate and normalise the input profiles (raises ValueError for unusable input)
        (
            pressure_hpa,
            temperature_c,
            dewpoint_c,
            u_wind_kt,
            v_wind_kt,
            height_m,
        ) = self._validate_profile_inputs(
            pressure_hpa, temperature_c, dewpoint_c, u_wind_kt, v_wind_kt, height_m
        )

        # Define the list of all indices to compute
        _ALL_KEYS = [
            "cape",
            "cin",
            "sbcape",
            "sbcin",
            "mlcape",
            "mlcin",
            "mucape",
            "mucin",
            "lifted_index",
            "dcape",
            "lcl_pressure",
            "lcl_temperature",
            "lfc_pressure",
            "el_pressure",
            "k_index",
            "total_totals",
            "showalter_index",
            "cross_totals",
            "precipitable_water",
            "wet_bulb_zero_height",
            "bulk_shear_0_1km",
            "bulk_shear_0_6km",
            "srh_0_1km",
            "srh_0_3km",
            "stp",
            "scp",
            "sweat_index",
        ]

        # Initialize the result dictionary with all keys set to None by default
        result: Dict[str, Optional[float]] = dict.fromkeys(_ALL_KEYS, None)

        # Create local aliases for MetPy calculation functions and units
        metpy_calc, metpy_units = _load_metpy()

        # If MetPy is not available, compute a fallback LCL estimate and skip the rest of the indices
        if metpy_calc is None:
            self._compute_fallback_lcl(pressure_hpa, temperature_c, dewpoint_c, result)
            return result

        # Compute the parcel profile using the surface pressure, temperature, and dewpoint
        try:
            pressure_metpy = pressure_hpa * metpy_units.hPa
            temperature_metpy = temperature_c * metpy_units.degC
            dewpoint_metpy = dewpoint_c * metpy_units.degC
            parcel_profile = cast(
                Any,
                metpy_calc.parcel_profile(
                    pressure_metpy, temperature_metpy[0], dewpoint_metpy[0]
                ),
            ).to("degC")

        # Catch any exceptions that occur during the base profile computation
        except Exception as exc:
            if self.verbose:
                logger.warning("MetPy base profile computation failed: %s", exc)
            return result

        # --- Lifting condensation level (LCL) ---
        result["lcl_pressure"], result["lcl_temperature"] = self._safe_pair(
            metpy_calc.lcl, pressure_metpy[0], temperature_metpy[0], dewpoint_metpy[0]
        )

        # --- Level of free convection (LFC) ---
        result["lfc_pressure"], _ = self._safe_pair(
            metpy_calc.lfc, pressure_metpy, temperature_metpy, dewpoint_metpy
        )

        # --- Equilibrium level (EL) ---
        result["el_pressure"], _ = self._safe_pair(
            metpy_calc.el, pressure_metpy, temperature_metpy, dewpoint_metpy
        )

        # --- Surface-based CAPE/CIN ---
        result["sbcape"], result["sbcin"] = self._safe_pair(
            metpy_calc.surface_based_cape_cin,
            pressure_metpy,
            temperature_metpy,
            dewpoint_metpy,
        )

        # --- Surface-based CAPE/CIN fallback for backward compatibility ---
        if result["sbcape"] is None:
            result["sbcape"], result["sbcin"] = self._safe_pair(
                metpy_calc.cape_cin,
                pressure_metpy,
                temperature_metpy,
                dewpoint_metpy,
                parcel_profile,
            )

        # --- For backward compatibility, also store surface-based CAPE/CIN under 'cape' and 'cin' keys ---
        result["cape"], result["cin"] = result["sbcape"], result["sbcin"]

        # --- Mixed layer CAPE/CIN ---
        result["mlcape"], result["mlcin"] = self._safe_pair(
            metpy_calc.mixed_layer_cape_cin,
            pressure_metpy,
            temperature_metpy,
            dewpoint_metpy,
        )

        # --- Most unstable CAPE/CIN (using the level with the highest equivalent potential temperature) ---
        result["mucape"], result["mucin"] = self._safe_pair(
            metpy_calc.most_unstable_cape_cin,
            pressure_metpy,
            temperature_metpy,
            dewpoint_metpy,
        )

        # --- Downdraft CAPE (MetPy returns a 3-tuple; only the first element, DCAPE, is kept) ---
        result["dcape"], _ = self._safe_pair(
            metpy_calc.downdraft_cape, pressure_metpy, temperature_metpy, dewpoint_metpy
        )

        # --- Scalar stability indices ---
        result["lifted_index"] = self._safe_scalar(
            metpy_calc.lifted_index, pressure_metpy, temperature_metpy, parcel_profile
        )

        # --- K-index ---
        result["k_index"] = self._safe_scalar(
            metpy_calc.k_index, pressure_metpy, temperature_metpy, dewpoint_metpy
        )

        # --- Total totals index ---
        result["total_totals"] = self._safe_scalar(
            metpy_calc.total_totals_index,
            pressure_metpy,
            temperature_metpy,
            dewpoint_metpy,
        )

        # --- Showalter index ---
        result["showalter_index"] = self._safe_scalar(
            metpy_calc.showalter_index,
            pressure_metpy,
            temperature_metpy,
            dewpoint_metpy,
        )

        # --- Cross totals index ---
        result["cross_totals"] = self._safe_scalar(
            metpy_calc.cross_totals, pressure_metpy, temperature_metpy, dewpoint_metpy
        )

        # --- Precipitable water ---
        result["precipitable_water"] = self._safe_scalar(
            metpy_calc.precipitable_water, pressure_metpy, dewpoint_metpy, to_unit="mm"
        )

        # --- Wet-bulb zero height (above the terrain when known, else above the lowest level) ---
        if height_m is not None:
            lowest_m = float(np.asarray(height_m, dtype=np.float64)[0])
            ground_m = self._ground_height(surface_height_m, lowest_m)
            result["wet_bulb_zero_height"] = self._compute_wet_bulb_zero(
                pressure_metpy,
                temperature_metpy,
                dewpoint_metpy,
                height_m,
                surface_height_m=ground_m,
            )

        # --- Shear and storm-relative indices (require wind and height) ---
        if u_wind_kt is not None and v_wind_kt is not None and height_m is not None:
            self._compute_shear_indices(
                result,
                pressure_hpa,
                temperature_c,
                dewpoint_c,
                u_wind_kt,
                v_wind_kt,
                height_m,
                surface_height_m=surface_height_m,
            )

        # Return the dictionary of computed indices
        return result

    def _validate_profile_inputs(
        self: "SoundingDiagnostics",
        pressure_hpa: Any,
        temperature_c: Any,
        dewpoint_c: Any,
        u_wind_kt: Any,
        v_wind_kt: Any,
        height_m: Any,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        This function validates the sounding profiles handed to compute_thermodynamic_indices and returns them as aligned float arrays ordered surface-first. Profiles of different lengths, a pressure that is not in hPa (any value above 1100 or not positive, which is what Pa or a zero fill looks like) and temperatures or dewpoints that look like Kelvin (above 80 degC) raise a ValueError, because MetPy would otherwise return silently wrong numbers or all None. A profile ordered top-first (pressure increasing) is reordered to decreasing pressure together with the winds and heights, and a dewpoint above the temperature is clipped to the temperature with a warning.

        Parameters:
            pressure_hpa (Any): Pressure profile in hPa.
            temperature_c (Any): Temperature profile in degC.
            dewpoint_c (Any): Dewpoint profile in degC.
            u_wind_kt (Any): U-component of wind in knots, or None.
            v_wind_kt (Any): V-component of wind in knots, or None.
            height_m (Any): Height profile in meters, or None.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]: The validated (pressure, temperature, dewpoint, u_wind, v_wind, height) profiles.
        """

        def _arr(x: Any) -> Optional[np.ndarray]:
            """
            This helper converts an optional profile to a one-dimensional float64 array, or returns None for a missing profile.

            Parameters:
                x (Any): Profile values or None.

            Returns:
                Optional[np.ndarray]: One-dimensional float array, or None.
            """
            return None if x is None else np.asarray(x, dtype=np.float64).ravel()

        p = cast(np.ndarray, _arr(pressure_hpa))
        t = cast(np.ndarray, _arr(temperature_c))
        td = cast(np.ndarray, _arr(dewpoint_c))
        u, v, z = _arr(u_wind_kt), _arr(v_wind_kt), _arr(height_m)

        # All profiles must describe the same levels
        sizes = {"pressure": p.size, "temperature": t.size, "dewpoint": td.size}
        for name, arr in (("u_wind", u), ("v_wind", v), ("height", z)):
            if arr is not None:
                sizes[name] = arr.size
        if len(set(sizes.values())) > 1:
            raise ValueError(
                f"Sounding profiles must have the same length, got {sizes}"
            )

        # Pressure must be in hPa and temperatures in degC
        finite_p = p[np.isfinite(p)]
        if finite_p.size and (finite_p.max() > 1100.0 or finite_p.min() <= 0.0):
            raise ValueError(
                "pressure_hpa must be in hPa (0 < p <= 1100); got range "
                f"{finite_p.min():g} to {finite_p.max():g}. Convert Pa to hPa first."
            )
        for name, arr in (("temperature_c", t), ("dewpoint_c", td)):
            finite = arr[np.isfinite(arr)]
            if finite.size and finite.max() > 80.0:
                raise ValueError(
                    f"{name} must be in degC; got a maximum of {finite.max():g}, "
                    "which looks like Kelvin."
                )

        # Order surface-first (decreasing pressure) when the profile came top-first
        finite_idx = np.nonzero(np.isfinite(p))[0]
        if np.any(np.diff(p[finite_idx]) > 0):
            order = np.argsort(-np.where(np.isfinite(p), p, -np.inf), kind="stable")
            p, t, td = p[order], t[order], td[order]
            u = None if u is None else u[order]
            v = None if v is None else v[order]
            z = None if z is None else z[order]

        # A dewpoint cannot exceed the temperature
        excess = np.isfinite(t) & np.isfinite(td) & (td > t)
        if np.any(excess):
            logger.warning(
                "Dewpoint exceeds temperature at %d level(s); clipping to temperature",
                int(excess.sum()),
            )
            td = np.where(excess, t, td)

        return p, t, td, u, v, z

    @staticmethod
    def _ground_height(
        surface_height_m: Optional[float], lowest_level_m: float
    ) -> float:
        """
        This helper returns the height of the ground used as the reference of the above-ground layers: the terrain height when it is known, finite and not above the lowest profile level, otherwise the height of the lowest profile level itself.

        Parameters:
            surface_height_m (Optional[float]): Terrain height in meters, or None when unknown.
            lowest_level_m (float): Height of the lowest profile level in meters.

        Returns:
            float: The reference ground height in meters.
        """
        if (
            surface_height_m is not None
            and np.isfinite(surface_height_m)
            and surface_height_m < lowest_level_m
        ):
            return float(surface_height_m)
        return float(lowest_level_m)

    @staticmethod
    def potential_to_actual_temperature(
        theta: np.ndarray,
        pressure_pa: np.ndarray,
    ) -> np.ndarray:
        """
        This function converts potential temperature (in K) to actual (sensible) temperature (in K) using the formula T = θ * (p / p0)^kappa, where p0 is the reference pressure (1000 hPa) and kappa is the Poisson constant (R_d / c_p). The function ensures that the input arrays are treated as floating-point values for accurate calculations. This allows for deriving the actual temperature profile when only potential temperature and pressure are available in the MPAS dataset.

        Parameters:
            theta (np.ndarray): Potential temperature profile in K.
            pressure_pa (np.ndarray): Pressure profile in Pa.

        Returns:
            np.ndarray: Actual (sensible) temperature in K.
        """
        # Return the actual temperature profile in K by applying the Poisson equation
        return cast(
            np.ndarray,
            np.asarray(theta, dtype=np.float64)
            * (np.asarray(pressure_pa, dtype=np.float64) / P0_REF_PA) ** KAPPA,
        )

    def _load_grid_coordinates(
        self: "SoundingDiagnostics", processor: "MPAS3DProcessor"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function loads the longitude and latitude coordinates of the cell centers from the MPAS grid file using xarray. It checks for common variable names for longitude and latitude in the grid dataset, and extracts them as 1D arrays. If the coordinates are in radians, it converts them to degrees. Finally, it ensures that longitudes are in the range [-180, 180] degrees before returning the longitude and latitude arrays. This is used to find the nearest cell to a target location when extracting sounding profiles.

        Parameters:
            processor (MPAS3DProcessor): The MPAS data processor instance containing the grid file.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Longitude and latitude arrays in degrees.
        """
        grid_file = getattr(processor, "grid_file", None)
        cache_key = grid_file if isinstance(grid_file, str) else None

        if cache_key is not None and cache_key in _GRID_COORD_CACHE:
            return _GRID_COORD_CACHE[cache_key]

        open_kwargs = self._coord_open_kwargs(processor.grid_file)

        with xr.open_dataset(
            processor.grid_file, **open_kwargs, engine=NETCDF_ENGINE
        ) as grid_ds:
            lon = self._select_coordinate(
                grid_ds, ("lonCell", "longitude", "lon"), "longitude"
            )
            lat = self._select_coordinate(
                grid_ds, ("latCell", "latitude", "lat"), "latitude"
            )
            lon_attrs = next(
                (
                    grid_ds[n].attrs
                    for n in ("lonCell", "longitude", "lon")
                    if n in grid_ds
                ),
                None,
            )
            lat_attrs = next(
                (
                    grid_ds[n].attrs
                    for n in ("latCell", "latitude", "lat")
                    if n in grid_ds
                ),
                None,
            )

        # Convert from radians to degrees if necessary, using the shared units-aware detection
        lon, lat = MPASGeographicUtils.coordinates_to_degrees(
            lon, lat, lon_attrs=lon_attrs, lat_attrs=lat_attrs
        )

        # Ensure longitudes are in the range [-180, 180]
        lon = ((lon + 180) % 360) - 180

        # Cache for reuse by subsequent soundings on the same grid in this worker
        if cache_key is not None:
            if len(_GRID_COORD_CACHE) >= _GRID_COORD_CACHE_MAX:
                _GRID_COORD_CACHE.clear()
            _GRID_COORD_CACHE[cache_key] = (lon, lat)

        # Return the longitude and latitude arrays in degrees
        return lon, lat

    @staticmethod
    def _coord_open_kwargs(grid_file: Any) -> dict:
        """
        This helper function determines the appropriate keyword arguments to pass to xarray's open_dataset function when loading the grid file for coordinate extraction. It always disables time decoding since the grid file should not contain time coordinates, and it optionally drops non-coordinate variables if they are present in the grid dataset. This is done by opening the grid file in a temporary context, checking for the presence of coordinate variables, and constructing a list of variables to drop if they are not among the expected coordinate names. This can help reduce memory usage and speed up loading when the grid file contains many non-coordinate variables that are not needed for nearest-cell lookup.

        Parameters:
            grid_file (Any): Path to the MPAS grid file to open.

        Returns:
            dict: Keyword arguments for xr.open_dataset (always disables time decoding, optionally drops non-coordinate variables).
        """
        open_kwargs: dict = {"decode_times": False}

        try:
            with xr.open_dataset(
                grid_file, decode_times=False, engine=NETCDF_ENGINE
            ) as probe:
                all_vars = list(probe.data_vars)
            drop = [v for v in all_vars if v not in _SOUNDING_COORD_NAMES]
            if drop:
                open_kwargs["drop_variables"] = drop
        except Exception:
            pass

        return open_kwargs

    @staticmethod
    def _select_coordinate(
        grid_ds: xr.Dataset, candidates: Tuple[str, ...], kind: str
    ) -> np.ndarray:
        """
        This function selects the coordinate variable from the grid dataset by checking a list of candidate names in order. It looks for the candidate names in both the coordinates and data variables of the dataset, and returns the values of the first match as a 1D array. If none of the candidates are found, it raises a ValueError indicating that the specified coordinate could not be found in the grid file. This allows for flexible handling of different grid file conventions where longitude and latitude may be named differently.

        Parameters:
            grid_ds (xr.Dataset): The opened grid dataset to search.
            candidates (Tuple[str, ...]): Candidate variable names to try in order.
            kind (str): Human-readable coordinate name ('longitude'/'latitude') for the error message.

        Returns:
            np.ndarray: The matched coordinate variable's values raveled to 1D.
        """
        for name in candidates:
            if name in grid_ds.coords or name in grid_ds.data_vars:
                return cast(np.ndarray, grid_ds[name].values.ravel())
        raise ValueError(f"Cannot find {kind} coordinate in grid file")

    @staticmethod
    def _find_nearest_cell(
        grid_lon: np.ndarray,
        grid_lat: np.ndarray,
        target_lon: float,
        target_lat: float,
    ) -> int:
        """
        This function finds the index of the nearest cell in the grid to a target longitude and latitude using a KDTree for efficient spatial querying. It converts the grid coordinates and target location from spherical (longitude, latitude) to Cartesian (x, y, z) coordinates on the unit sphere, then queries a KDTree of the grid points to find the nearest neighbor. The KDTree depends only on the model grid, so it is built once and cached (keyed by a cheap grid fingerprint) and reused across every call for that grid -- in a batch of soundings this turns a per-file O(n_cells log n_cells) tree build into a single cheap query. The function returns the index of the nearest cell in the grid, which can then be used to extract the sounding profile at that location.

        Parameters:
            grid_lon (np.ndarray): 1D array of grid longitudes in degrees.
            grid_lat (np.ndarray): 1D array of grid latitudes in degrees.
            target_lon (float): Target longitude in degrees.
            target_lat (float): Target latitude in degrees.

        Returns:
            int: Index of the nearest cell in the grid.
        """
        tree = SoundingDiagnostics._get_nearest_cell_tree(grid_lon, grid_lat)

        # Convert the target longitude and latitude from degrees to radians
        tgt_lon_r = np.radians(target_lon)
        tgt_lat_r = np.radians(target_lat)

        # Convert the target longitude and latitude to Cartesian coordinates on the unit sphere
        tgt_cart = np.array(
            [
                [
                    np.cos(tgt_lat_r) * np.cos(tgt_lon_r),
                    np.cos(tgt_lat_r) * np.sin(tgt_lon_r),
                    np.sin(tgt_lat_r),
                ]
            ]
        )

        # Query the cached tree with the target point to find the nearest neighbor
        _, nearest_idx = tree.query(tgt_cart)

        # Return the index of the nearest cell as an integer
        return int(nearest_idx[0])

    @staticmethod
    def _get_nearest_cell_tree(grid_lon: np.ndarray, grid_lat: np.ndarray) -> Any:
        """
        This helper returns a KDTree built over the model grid's cell centres on the unit sphere, reusing a cached tree whenever the same grid is queried again. It computes a cheap fingerprint of the grid (shape plus endpoint and summed coordinates) so repeated soundings on the identical grid -- the common batch case of one Skew-T per timestep at a fixed station -- avoid rebuilding the tree, which is the dominant cost of nearest-cell lookup. The cache is bounded; when it is full it is cleared before inserting a new tree so a long-lived worker that processes several distinct grids cannot leak memory.

        Parameters:
            grid_lon (np.ndarray): 1D array of grid longitudes in degrees.
            grid_lat (np.ndarray): 1D array of grid latitudes in degrees.

        Returns:
            Any: A scipy KDTree over the grid cell centres in unit-sphere Cartesian coordinates.
        """
        grid_lon = np.asarray(grid_lon)
        grid_lat = np.asarray(grid_lat)

        key = (
            grid_lon.shape,
            float(grid_lon[0]),
            float(grid_lon[-1]),
            float(grid_lat[0]),
            float(grid_lat[-1]),
            float(grid_lon.sum()),
            float(grid_lat.sum()),
        )

        tree = _NEAREST_CELL_TREE_CACHE.get(key)
        if tree is not None:
            return tree

        lon_r = np.radians(grid_lon)
        lat_r = np.radians(grid_lat)

        cart = np.column_stack(
            [
                np.cos(lat_r) * np.cos(lon_r),
                np.cos(lat_r) * np.sin(lon_r),
                np.sin(lat_r),
            ]
        )

        tree = cKDTree(cart)

        if len(_NEAREST_CELL_TREE_CACHE) >= _NEAREST_CELL_TREE_CACHE_MAX:
            _NEAREST_CELL_TREE_CACHE.clear()
        _NEAREST_CELL_TREE_CACHE[key] = tree

        return tree

    @staticmethod
    def _nearest_neighbor_distance_km(
        grid_lon: np.ndarray, grid_lat: np.ndarray, cell_idx: int
    ) -> float:
        """
        This helper returns the great-circle distance in kilometres from one grid cell to its nearest neighbouring cell centre, which is the local cell spacing used as the scale of the "is the requested point inside the domain" check. It queries the cached KDTree of the grid for the two nearest points of the cell (the cell itself and its closest neighbour) and converts the unit-sphere chord length to a great-circle arc on a sphere of radius 6371 km.

        Parameters:
            grid_lon (np.ndarray): 1D array of grid longitudes in degrees.
            grid_lat (np.ndarray): 1D array of grid latitudes in degrees.
            cell_idx (int): Index of the cell.

        Returns:
            float: Distance to the nearest neighbouring cell centre in km (infinite for a single-cell grid).
        """
        if len(grid_lon) < 2:
            return float("inf")

        tree = SoundingDiagnostics._get_nearest_cell_tree(grid_lon, grid_lat)
        lon_r, lat_r = np.radians(grid_lon[cell_idx]), np.radians(grid_lat[cell_idx])
        point = np.array(
            [
                [
                    np.cos(lat_r) * np.cos(lon_r),
                    np.cos(lat_r) * np.sin(lon_r),
                    np.sin(lat_r),
                ]
            ]
        )
        chord, _ = tree.query(point, k=2)
        return float(6371.0 * 2.0 * np.arcsin(min(float(chord[0, 1]) / 2.0, 1.0)))

    @staticmethod
    def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """
        This function calculates the great-circle distance between two points on the Earth's surface specified by their longitude and latitude using the Haversine formula. The input coordinates are in degrees, and the output distance is returned in kilometers. This is used to compute the distance from the target location to the nearest grid cell when extracting sounding profiles, providing a measure of how close the extracted profile is to the desired location.

        Parameters:
            lon1 (float): Longitude of the first point in degrees.
            lat1 (float): Latitude of the first point in degrees.
            lon2 (float): Longitude of the second point in degrees.
            lat2 (float): Latitude of the second point in degrees.

        Returns:
            float: Distance between the two points in kilometers.
        """
        # Convert input coordinates from degrees to radians
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        # Calculate the differences in latitude and longitude
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Haversine formula to calculate the great-circle distance
        haversine_term = (
            np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        )

        # Return the distance in kilometers
        return float(6371.0 * 2 * np.arcsin(np.sqrt(haversine_term)))

    @staticmethod
    def _layer_covered(
        height_m: np.ndarray, u_wind: np.ndarray, v_wind: np.ndarray, depth_m: float
    ) -> bool:
        """
        This helper reports whether a sounding profile supports a layer-based index (bulk shear or storm-relative helicity) over the lowest `depth_m` metres. The profile must reach at least that far above its lowest level, and the winds inside the layer must be finite. MetPy silently clips the layer to the available data and returns 0.0 for missing winds, so without this check a shallow or wind-less profile would report a misleading number instead of None.

        Parameters:
            height_m (np.ndarray): Height profile in metres, ordered from the surface upward.
            u_wind (np.ndarray): Zonal wind profile (any unit).
            v_wind (np.ndarray): Meridional wind profile (any unit).
            depth_m (float): Layer depth in metres above the lowest level.

        Returns:
            bool: True if the layer is fully covered by finite heights and winds, False otherwise.
        """
        heights = np.asarray(height_m, dtype=np.float64)
        finite_heights = heights[np.isfinite(heights)]

        if finite_heights.size < 2:
            return False

        bottom = float(finite_heights.min())

        if float(finite_heights.max()) - bottom < 0.999 * depth_m:
            return False

        in_layer = np.isfinite(heights) & (heights <= bottom + depth_m)
        winds = np.stack(
            [np.asarray(u_wind, dtype=np.float64), np.asarray(v_wind, dtype=np.float64)]
        )

        return bool(np.isfinite(winds[:, in_layer]).all())

    @staticmethod
    def _finite_or_none(value: Any) -> Optional[float]:
        """
        This helper converts a scalar-like value (a Python or NumPy number, or a zero-dimensional array) to a float and returns None when the result is not finite, so that every sounding index uses the same convention: a number when defined, None when undefined. Anything that cannot be converted to a scalar float also returns None.

        Parameters:
            value (Any): Scalar-like value, typically the magnitude of a MetPy quantity.

        Returns:
            Optional[float]: The value as a float, or None if it is NaN, infinite or not a scalar.
        """
        try:
            number = float(np.asarray(value, dtype=np.float64).reshape(()))
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    @staticmethod
    def _safe_scalar(
        func: Any,
        *args: Any,
        to_unit: Optional[str] = None,
    ) -> Optional[float]:
        """
        This function safely calls a MetPy calculation function that returns a single quantity (e.g., lifted index), handling any exceptions that may occur during the calculation. It attempts to call the provided function with the given arguments, optionally converting the result to a specified unit before extracting its magnitude. If any step of the process fails (e.g., due to insufficient profile levels, invalid values, or issues with unit conversion), it catches the exception and returns None instead of raising an error; a NaN result (undefined index) is also returned as None. This allows for robust computation of indices where some calculations may not be possible due to data limitations, while still providing valid results for other indices that can be computed successfully.

        Parameters:
            func: MetPy calculation function that returns a single quantity (e.g., ``mpcalc.lifted_index``).
            *args: Positional arguments forwarded to *func*.
            to_unit: Optional unit string for conversion (e.g., 'K' or 'J/kg').

        Returns:
            Optional[float]: Computed index value on success, None on any failure.
        """
        try:
            # Compute the value using the provided function and arguments
            with _quiet_out_of_bounds():
                computed_val = func(*args)

            # If a unit conversion is requested, convert the result to the specified unit
            if to_unit is not None:
                computed_val = computed_val.to(to_unit)

            # Extract the magnitude of the result
            magnitude = computed_val.magnitude

            # Extract the magnitude as a float
            value = (
                float(magnitude.item())
                if hasattr(magnitude, "item")
                else float(magnitude)
            )

            # MetPy signals "not defined" (e.g. no LFC/EL) with NaN; report that as None
            return value if np.isfinite(value) else None

        # Catch any exceptions that occur during the calculation and return None
        except Exception:
            return None

    @staticmethod
    def _safe_pair(
        func: Any,
        *args: Any,
        to_unit: Optional[str] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        This function safely calls a MetPy calculation function that returns a pair of quantities (e.g., CAPE and CIN), handling any exceptions that may occur during the calculation. It attempts to call the provided function with the given arguments, optionally converting both results to a specified unit before extracting their magnitudes. If any step of the process fails (e.g., due to insufficient profile levels, invalid values, or issues with unit conversion), it catches the exception and returns (None, None) instead of raising an error. Elements that MetPy reports as NaN (for example no LFC or EL), or that are not scalars, are returned as None individually, and for functions returning more than two values (downdraft_cape) only the first two are used. This allows for robust computation of paired indices where some calculations may not be possible due to data limitations, while still providing valid results for other indices that can be computed successfully.

        Parameters:
            func: MetPy calculation function that returns a pair of quantities (e.g., ``mpcalc.surface_based_cape_cin``).
            *args: Positional arguments forwarded to *func*.
            to_unit: Optional unit string for conversion (e.g., 'J/kg').

        Returns:
            Tuple[Optional[float], Optional[float]]: Pair of float values on success, (None, None) on any failure.
        """
        try:
            # Compute the result using the provided function and arguments; only the first two elements are used (MetPy's downdraft_cape returns three)
            with _quiet_out_of_bounds():
                func_result = func(*args)
            first_result, second_result = func_result[0], func_result[1]

            # If a unit conversion is requested, convert both results to the specified unit
            if to_unit is not None:
                first_result, second_result = first_result.to(
                    to_unit
                ), second_result.to(to_unit)

            # Extract the magnitudes as floats, mapping NaN (undefined) and non-scalar extras (e.g. downdraft profiles) to None
            def _as_float(quantity: Any) -> Optional[float]:
                magnitude = np.asarray(quantity.magnitude, dtype=np.float64)
                if magnitude.ndim != 0 or not np.isfinite(magnitude):
                    return None
                return float(magnitude)

            return _as_float(first_result), _as_float(second_result)

        # Catch any exceptions that occur during the calculation and return (None, None)
        except Exception:
            return None, None

    def _compute_fallback_lcl(
        self: "SoundingDiagnostics",
        pressure_hpa: np.ndarray,
        temperature_c: np.ndarray,
        dewpoint_c: np.ndarray,
        result: Dict[str, Optional[float]],
    ) -> None:
        """
        This function computes a fallback estimate for the Lifting Condensation Level (LCL) pressure and temperature using a simplified formula based on the surface temperature and dewpoint. It is used when MetPy is not available to provide at least some basic thermodynamic information about the sounding profile. The formula estimates the LCL temperature as the surface dewpoint minus a factor times the difference between surface temperature and dewpoint, and then calculates the LCL pressure using the Poisson equation. The resulting LCL pressure and temperature are stored in the provided result dictionary under the keys 'lcl_pressure' and 'lcl_temperature'. This allows users to still obtain an estimate of the LCL even without MetPy, although it may be less accurate than the full calculation.

        Parameters:
            pressure_hpa (np.ndarray): Pressure profile in hPa.
            temperature_c (np.ndarray): Temperature profile in °C.
            dewpoint_c (np.ndarray): Dewpoint profile in °C.
            result (Dict[str, Optional[float]]): Output dictionary to store 'lcl_pressure' and 'lcl_temperature' (modified in-place).

        Returns:
            None
        """
        try:
            # Use the surface temperature and dewpoint (first level of the profile) for the LCL estimation
            surface_temp_c = temperature_c[0]
            surface_dewpoint_c = dewpoint_c[0]

            # Estimate the LCL temperature using a simplified formula based on surface temperature and dewpoint
            lcl_temperature = surface_dewpoint_c - (
                0.001296 * surface_dewpoint_c + 0.1963
            ) * (surface_temp_c - surface_dewpoint_c)

            # Compute the LCL pressure using the Poisson equation with the estimated LCL temperature
            lcl_pressure = pressure_hpa[0] * (
                (lcl_temperature + 273.15) / (surface_temp_c + 273.15)
            ) ** (1.0 / KAPPA)

            # Store the fallback LCL pressure and temperature in the result dictionary
            result["lcl_pressure"] = float(lcl_pressure)
            result["lcl_temperature"] = float(lcl_temperature)
        except Exception:
            pass

        # Print a message about the fallback LCL computation if verbose mode is enabled
        if self.verbose:
            logger.warning(
                "MetPy not available – full index computation skipped. "
                "Install metpy>=1.3.0 for comprehensive thermodynamic indices."
            )

    @staticmethod
    def _compute_wet_bulb_zero(
        pressure_metpy: Any,
        temperature_metpy: Any,
        dewpoint_metpy: Any,
        height_m: np.ndarray,
        surface_height_m: float = 0.0,
    ) -> Optional[float]:
        """
        This function computes the height of the wet bulb zero isotherm (the level where the wet bulb temperature reaches 0°C) using MetPy's wet bulb temperature calculation. It takes pressure, temperature, and dewpoint profiles with MetPy units, along with the corresponding height profile in meters. The function calculates the wet bulb temperature at each level, converts it to degrees Celsius, and then finds the first crossing of the 0°C isotherm. If a crossing is found, it performs a linear interpolation between the two levels that bracket the crossing to estimate the height of the wet bulb zero. If any step of the process fails (e.g., due to insufficient profile levels or invalid values), it returns None. Levels with a non-finite wet bulb temperature or height are discarded before the crossing search so that NaN cannot create a spurious sign change. The returned height is above ground level: surface_height_m (the height of the lowest profile level, in the same reference as height_m) is subtracted from the interpolated MSL height; pass 0 to obtain the MSL height. This provides an important diagnostic for understanding freezing level conditions in the sounding profile.

        Parameters:
            pressure_metpy (Any): Pressure profile with MetPy units (e.g., hPa).
            temperature_metpy (Any): Temperature profile with MetPy units (e.g., °C).
            dewpoint_metpy (Any): Dewpoint profile with MetPy units (e.g., °C).
            height_m (np.ndarray): Height profile in meters corresponding to the pressure levels.
            surface_height_m (float): Reference height in meters (same datum as height_m) subtracted from the result to give height above ground (default: 0.0, i.e. no subtraction).

        Returns:
            Optional[float]: Height of the wet bulb zero isotherm in meters above the reference height, or None if computation failed.
        """
        # Create local alias for MetPy calculation functions
        metpy_calc: Any = _metpy_calc()

        try:
            # Ensure the height profile is treated as a floating-point array for accurate calculations
            height_arr = np.asarray(height_m, dtype=np.float64)

            # Calculate the wet bulb temperature profile using MetPy's calculation function
            wet_bulb_metpy = cast(
                Any,
                metpy_calc.wet_bulb_temperature(
                    pressure_metpy, temperature_metpy, dewpoint_metpy
                ),
            )

            # Convert the wet bulb temperature to degrees Celsius for zero crossing detection
            wet_bulb_celsius = np.asarray(
                wet_bulb_metpy.to("degC").magnitude, dtype=np.float64
            )

            # Keep only levels where both the wet bulb temperature and the height are finite
            valid = np.isfinite(wet_bulb_celsius) & np.isfinite(height_arr)
            wet_bulb_valid = wet_bulb_celsius[valid]
            height_valid = height_arr[valid]

            # A profile without vertical extent cannot locate the crossing height
            if height_valid.size < 2 or np.ptp(height_valid) <= 0:
                return None

            # Find the layers (between adjacent valid levels) that bracket a 0°C crossing
            crossings = np.nonzero(
                (wet_bulb_valid[:-1] * wet_bulb_valid[1:] <= 0)
                & (wet_bulb_valid[:-1] != wet_bulb_valid[1:])
            )[0]

            if len(crossings) > 0:
                # Get the index of the first crossing of the 0°C isotherm in the wet bulb temperature profile
                crossing_idx = crossings[0]

                # Compute the fraction of the way between the two levels where the wet bulb zero crossing occurs for linear interpolation
                frac = -wet_bulb_valid[crossing_idx] / (
                    wet_bulb_valid[crossing_idx + 1] - wet_bulb_valid[crossing_idx]
                )

                # Return the interpolated height of the wet bulb zero isotherm in meters above the reference height
                return float(
                    height_valid[crossing_idx]
                    + frac
                    * (height_valid[crossing_idx + 1] - height_valid[crossing_idx])
                    - surface_height_m
                )

        # Catch any exceptions that occur during the wet bulb zero computation and return None
        except Exception:
            pass
        return None

    def _compute_shear_indices(
        self: "SoundingDiagnostics",
        result: Dict[str, Optional[float]],
        pressure_hpa: np.ndarray,
        temperature_c: np.ndarray,
        dewpoint_c: np.ndarray,
        u_wind_kt: np.ndarray,
        v_wind_kt: np.ndarray,
        height_m: np.ndarray,
        surface_height_m: Optional[float] = None,
    ) -> None:
        """
        This function computes shear-related indices such as bulk shear, storm-relative helicity, significant tornado parameter (STP), supercell composite parameter (SCP), and the SWEAT index using MetPy's calculation functions. It first converts the wind components from knots to meters per second and ensures that all input profiles are treated as floating-point values with appropriate units. It then calculates the bulk shear for specified depth ranges and the storm-relative helicity for specified depth ranges relative to the Bunkers right-moving storm motion (MetPy bunkers_storm_motion, so the values are comparable with the usual SRH and are not ground-relative), and uses these along with CAPE and LCL information to compute STP and SCP if the necessary dependencies are available. Finally, it computes the SWEAT index using the full profile of pressure, temperature, dewpoint, wind speed and meteorological wind direction (degrees, as MetPy requires). The STP LCL height is obtained by interpolating height linearly in log-pressure to the LCL pressure and is expressed above the terrain when surface_height_m is given, otherwise above the lowest profile level. The layers (0-1, 0-3, 0-6 km) start at the same ground height; when the terrain lies below the lowest profile level a surface level carrying the lowest level's wind and a hydrostatically extrapolated pressure is added to the layer profiles. Each calculation is wrapped in a try-except block to handle any potential issues gracefully, allowing the function to return valid indices even if some calculations fail due to data limitations or invalid values. The computed indices are stored in the provided result dictionary under their respective keys.

        Parameters:
            result (Dict[str, Optional[float]]): Output dictionary to store computed indices (modified in-place).
            pressure_hpa (np.ndarray): Pressure profile in hPa.
            temperature_c (np.ndarray): Temperature profile in °C.
            dewpoint_c (np.ndarray): Dewpoint profile in °C.
            u_wind_kt (np.ndarray): U-component of wind in knots.
            v_wind_kt (np.ndarray): V-component of wind in knots.
            height_m (np.ndarray): Height profile in meters.
            surface_height_m (Optional[float]): Terrain height in meters in the datum of height_m, or None to start the layers at the lowest profile level (default: None).

        Returns:
            None
        """
        # Create local aliases for MetPy calculation functions and units
        metpy_units: Any = _metpy_units()

        try:
            kt_to_ms = 1.0 / 1.94384

            # Convert wind components from knots to meters per second
            u_ms = (
                np.asarray(u_wind_kt, dtype=np.float64) * kt_to_ms * metpy_units("m/s")
            )
            v_ms = (
                np.asarray(v_wind_kt, dtype=np.float64) * kt_to_ms * metpy_units("m/s")
            )

            # Ensure the height profile is treated as a floating-point array for accurate calculations and convert to MetPy units
            height_metpy = np.asarray(height_m, dtype=np.float64) * metpy_units.m

            # Convert pressure to MetPy units for the shear calculations
            pressure_metpy = pressure_hpa * metpy_units.hPa

            # Layer profiles: start at the terrain when it lies below the lowest level
            p_lay, h_lay, u_lay, v_lay = self._build_shear_layer_profiles(
                pressure_hpa,
                temperature_c,
                u_wind_kt,
                v_wind_kt,
                height_m,
                surface_height_m,
            )
            u_lay_ms = u_lay * kt_to_ms * metpy_units("m/s")
            v_lay_ms = v_lay * kt_to_ms * metpy_units("m/s")
            h_lay_metpy = h_lay * metpy_units.m
            p_lay_metpy = p_lay * metpy_units.hPa

        # Catch any exceptions that occur during the wind unit setup and skip shear-related calculations if it fails
        except Exception as exc:
            if self.verbose:
                logger.warning("MetPy wind unit setup failed: %s", exc)
            return

        layer = (p_lay_metpy, u_lay_ms, v_lay_ms, h_lay_metpy)
        self._compute_bulk_shear_indices(result, layer, h_lay, u_lay, v_lay)

        storm_u, storm_v = self._bunkers_right_mover(layer)
        self._compute_srh_indices(result, layer, h_lay, u_lay, v_lay, storm_u, storm_v)

        self._compute_stp_index(result, pressure_hpa, h_lay, height_metpy)
        self._compute_scp_index(result)
        self._compute_sweat_index(
            result, pressure_metpy, temperature_c, dewpoint_c, u_ms, v_ms
        )

    def _build_shear_layer_profiles(
        self: "SoundingDiagnostics",
        pressure_hpa: np.ndarray,
        temperature_c: np.ndarray,
        u_wind_kt: np.ndarray,
        v_wind_kt: np.ndarray,
        height_m: np.ndarray,
        surface_height_m: Optional[float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This function builds the layer profiles used by the shear calculations. The profiles start at the terrain when it lies below the lowest profile level, in which case a surface level carrying the lowest level's wind and a hydrostatically extrapolated pressure is prepended; otherwise the input profiles are returned unchanged as floating-point arrays.

        Parameters:
            pressure_hpa (np.ndarray): Pressure profile in hPa.
            temperature_c (np.ndarray): Temperature profile in °C.
            u_wind_kt (np.ndarray): U-component of wind in knots.
            v_wind_kt (np.ndarray): V-component of wind in knots.
            height_m (np.ndarray): Height profile in meters.
            surface_height_m (Optional[float]): Terrain height in meters in the datum of height_m, or None.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Pressure (hPa), height (m), u-wind (knots) and v-wind (knots) layer profiles.
        """
        p_lay = np.asarray(pressure_hpa, dtype=np.float64)
        h_lay = np.asarray(height_m, dtype=np.float64)
        u_lay = np.asarray(u_wind_kt, dtype=np.float64)
        v_lay = np.asarray(v_wind_kt, dtype=np.float64)
        ground_m = self._ground_height(surface_height_m, float(h_lay[0]))
        if ground_m < h_lay[0]:
            t0_k = float(np.asarray(temperature_c, dtype=np.float64)[0]) + 273.15
            p_sfc = p_lay[0] * np.exp(9.80665 * (h_lay[0] - ground_m) / (287.04 * t0_k))
            p_lay = np.r_[p_sfc, p_lay]
            h_lay = np.r_[ground_m, h_lay]
            u_lay = np.r_[u_lay[0], u_lay]
            v_lay = np.r_[v_lay[0], v_lay]
        return p_lay, h_lay, u_lay, v_lay

    def _compute_bulk_shear_indices(
        self: "SoundingDiagnostics",
        result: Dict[str, Optional[float]],
        layer: Tuple[Any, Any, Any, Any],
        h_lay: np.ndarray,
        u_lay: np.ndarray,
        v_lay: np.ndarray,
    ) -> None:
        """
        This function computes the 0-1 km and 0-6 km bulk shear magnitudes in knots with MetPy and stores them in the result dictionary. A layer that the profile does not cover (or that has missing winds) has no defined shear and is skipped, as is any layer whose computation fails.

        Parameters:
            result (Dict[str, Optional[float]]): Output dictionary to store computed indices (modified in-place).
            layer (Tuple[Any, Any, Any, Any]): MetPy-unit layer pressure, u-wind, v-wind and height profiles.
            h_lay (np.ndarray): Layer height profile in meters.
            u_lay (np.ndarray): Layer u-wind profile in knots.
            v_lay (np.ndarray): Layer v-wind profile in knots.

        Returns:
            None
        """
        metpy_calc: Any = _metpy_calc()
        metpy_units: Any = _metpy_units()
        p_lay_metpy, u_lay_ms, v_lay_ms, h_lay_metpy = layer

        for depth_m, key in [(1000, "bulk_shear_0_1km"), (6000, "bulk_shear_0_6km")]:
            # A layer the profile does not cover (or with missing winds) has no defined shear
            if not self._layer_covered(h_lay, u_lay, v_lay, depth_m):
                continue
            try:
                # Calculate the bulk shear for the specified depth range using MetPy's calculation function
                bulk_shear_u, bulk_shear_v = cast(
                    Any,
                    metpy_calc.bulk_shear(
                        p_lay_metpy,
                        u_lay_ms,
                        v_lay_ms,
                        height=h_lay_metpy,
                        depth=depth_m * metpy_units.m,
                    ),
                )

                # Store the magnitude of the bulk shear in knots (None when not finite)
                result[key] = self._finite_or_none(
                    metpy_calc.wind_speed(bulk_shear_u, bulk_shear_v)
                    .to("knots")
                    .magnitude
                )

            # Catch any exceptions that occur during the bulk shear computation and skip it
            except Exception:
                pass

    def _bunkers_right_mover(
        self: "SoundingDiagnostics",
        layer: Tuple[Any, Any, Any, Any],
    ) -> Tuple[Any, Any]:
        """
        This function determines the Bunkers right-moving storm motion from the layer profiles with MetPy. Both components are None when the motion cannot be determined or is not finite, which leaves the storm-relative helicity undefined.

        Parameters:
            layer (Tuple[Any, Any, Any, Any]): MetPy-unit layer pressure, u-wind, v-wind and height profiles.

        Returns:
            Tuple[Any, Any]: Storm motion u and v components with units, or (None, None).
        """
        metpy_calc: Any = _metpy_calc()
        p_lay_metpy, u_lay_ms, v_lay_ms, h_lay_metpy = layer

        # Storm motion: the Bunkers right-mover (None when it cannot be determined, which leaves SRH undefined)
        storm_u: Any = None
        storm_v: Any = None
        try:
            right_mover, _, _ = cast(
                Any,
                metpy_calc.bunkers_storm_motion(
                    p_lay_metpy, u_lay_ms, v_lay_ms, h_lay_metpy
                ),
            )
            storm_u, storm_v = right_mover[0], right_mover[1]
            if not (np.isfinite(storm_u.magnitude) and np.isfinite(storm_v.magnitude)):
                storm_u = storm_v = None
        except Exception:
            storm_u = storm_v = None
        return storm_u, storm_v

    def _compute_srh_indices(
        self: "SoundingDiagnostics",
        result: Dict[str, Optional[float]],
        layer: Tuple[Any, Any, Any, Any],
        h_lay: np.ndarray,
        u_lay: np.ndarray,
        v_lay: np.ndarray,
        storm_u: Any,
        storm_v: Any,
    ) -> None:
        """
        This function computes the 0-1 km and 0-3 km storm-relative helicity relative to the Bunkers right-mover with MetPy and stores them in the result dictionary. Nothing is computed when the storm motion is unavailable or the layer is not covered by the profile, because MetPy returns 0.0 (not NaN) in those cases.

        Parameters:
            result (Dict[str, Optional[float]]): Output dictionary to store computed indices (modified in-place).
            layer (Tuple[Any, Any, Any, Any]): MetPy-unit layer pressure, u-wind, v-wind and height profiles.
            h_lay (np.ndarray): Layer height profile in meters.
            u_lay (np.ndarray): Layer u-wind profile in knots.
            v_lay (np.ndarray): Layer v-wind profile in knots.
            storm_u (Any): Storm motion u component with units, or None.
            storm_v (Any): Storm motion v component with units, or None.

        Returns:
            None
        """
        metpy_calc: Any = _metpy_calc()
        metpy_units: Any = _metpy_units()
        _, u_lay_ms, v_lay_ms, h_lay_metpy = layer

        for depth_m, key in [(1000, "srh_0_1km"), (3000, "srh_0_3km")]:
            # MetPy returns 0.0 (not NaN) for missing winds or a profile shallower than the layer, so check first
            if storm_u is None or not self._layer_covered(h_lay, u_lay, v_lay, depth_m):
                continue
            try:
                # Calculate the storm-relative helicity (relative to the Bunkers right-mover) for the specified depth range
                _, _, srh = cast(
                    Any,
                    metpy_calc.storm_relative_helicity(
                        h_lay_metpy,
                        u_lay_ms,
                        v_lay_ms,
                        depth=depth_m * metpy_units.m,
                        storm_u=storm_u,
                        storm_v=storm_v,
                    ),
                )

                # Store the storm-relative helicity (None when not finite, like the other indices)
                result[key] = self._finite_or_none(srh.magnitude)

            # Catch any exceptions that occur during the storm-relative helicity computation and skip it
            except Exception:
                pass

    def _compute_stp_index(
        self: "SoundingDiagnostics",
        result: Dict[str, Optional[float]],
        pressure_hpa: np.ndarray,
        h_lay: np.ndarray,
        height_metpy: Any,
    ) -> None:
        """
        This function computes the Significant Tornado Parameter from surface-based CAPE, the LCL height, 0-1 km storm-relative helicity and 0-6 km bulk shear already stored in the result dictionary. The LCL height is interpolated linearly in log-pressure and expressed above the lowest layer level. Nothing is stored when a dependency is missing or the computation fails.

        Parameters:
            result (Dict[str, Optional[float]]): Output dictionary holding the dependencies and receiving 'stp' (modified in-place).
            pressure_hpa (np.ndarray): Pressure profile in hPa.
            h_lay (np.ndarray): Layer height profile in meters.
            height_metpy (Any): Height profile with MetPy units.

        Returns:
            None
        """
        metpy_calc: Any = _metpy_calc()
        metpy_units: Any = _metpy_units()

        try:
            # Check the dependencies for Significant Tornado Parameter: surface-based CAPE, LCL pressure, 0-1 km storm-relative helicity, and 0-6 km bulk shear
            stp_deps = [
                result.get("sbcape"),
                result.get("lcl_pressure"),
                result.get("srh_0_1km"),
                result.get("bulk_shear_0_6km"),
            ]

            if all(v is not None for v in stp_deps):
                # Calculate the height of the surface above ground level in meters for the STP calculation
                ground_height_m = float(h_lay[0])

                # Interpolate the height at the LCL pressure linearly in log-pressure (profile is ordered surface to top, so reverse for ascending log-p)
                log_p = np.log(np.asarray(pressure_hpa, dtype=np.float64))
                lcl_interp_m = float(
                    np.interp(
                        np.log(float(cast(float, stp_deps[1]))),
                        log_p[::-1],
                        np.asarray(height_metpy.magnitude, dtype=np.float64)[::-1],
                    )
                )

                # Calculate the height of the LCL above the surface in meters for the STP calculation
                lcl_height_m = lcl_interp_m - ground_height_m

                # Only attempt to compute STP if all dependencies are available, otherwise skip it
                result["stp"] = self._safe_scalar(
                    metpy_calc.significant_tornado,
                    stp_deps[0] * metpy_units("J/kg"),
                    lcl_height_m * metpy_units.m,
                    stp_deps[2] * metpy_units("m^2/s^2"),
                    stp_deps[3] * metpy_units.knots,
                )

        # Catch any exceptions that occur during the STP computation and skip it
        except Exception:
            pass

    def _compute_scp_index(
        self: "SoundingDiagnostics",
        result: Dict[str, Optional[float]],
    ) -> None:
        """
        This function computes the Supercell Composite Parameter from the most unstable CAPE, 0-3 km storm-relative helicity and 0-6 km bulk shear already stored in the result dictionary. Nothing is stored when a dependency is missing or the computation fails.

        Parameters:
            result (Dict[str, Optional[float]]): Output dictionary holding the dependencies and receiving 'scp' (modified in-place).

        Returns:
            None
        """
        metpy_calc: Any = _metpy_calc()
        metpy_units: Any = _metpy_units()

        try:
            # Check the dependencies for Supercell Composite Parameter: most unstable CAPE, 0-3 km storm-relative helicity, and 0-6 km bulk shear
            scp_deps = [
                result.get("mucape"),
                result.get("srh_0_3km"),
                result.get("bulk_shear_0_6km"),
            ]

            # Only attempt to compute SCP if all dependencies are available, otherwise skip it
            if all(v is not None for v in scp_deps):
                result["scp"] = self._safe_scalar(
                    metpy_calc.supercell_composite,
                    scp_deps[0] * metpy_units("J/kg"),
                    scp_deps[1] * metpy_units("m^2/s^2"),
                    scp_deps[2] * metpy_units.knots,
                )

        # Catch any exceptions that occur during the STP and SCP computations and skip them
        except Exception:
            pass

    def _compute_sweat_index(
        self: "SoundingDiagnostics",
        result: Dict[str, Optional[float]],
        pressure_metpy: Any,
        temperature_c: np.ndarray,
        dewpoint_c: np.ndarray,
        u_ms: Any,
        v_ms: Any,
    ) -> None:
        """
        This function computes the SWEAT index from the full profile of pressure, temperature, dewpoint, wind speed (knots) and meteorological wind direction, and stores it in the result dictionary. Nothing is stored when the computation fails.

        Parameters:
            result (Dict[str, Optional[float]]): Output dictionary receiving 'sweat_index' (modified in-place).
            pressure_metpy (Any): Pressure profile with MetPy units.
            temperature_c (np.ndarray): Temperature profile in °C.
            dewpoint_c (np.ndarray): Dewpoint profile in °C.
            u_ms (Any): U-component of wind in m/s with MetPy units.
            v_ms (Any): V-component of wind in m/s with MetPy units.

        Returns:
            None
        """
        metpy_calc: Any = _metpy_calc()
        metpy_units: Any = _metpy_units()

        try:
            # Generate the full profile of temperature and dewpoint with units for SWEAT calculation
            temperature_metpy = temperature_c * metpy_units.degC
            dewpoint_metpy = dewpoint_c * metpy_units.degC

            # SWEAT is defined with wind speeds in knots, and MetPy sums the raw magnitudes, so convert explicitly
            result["sweat_index"] = self._safe_scalar(
                metpy_calc.sweat_index,
                pressure_metpy,
                temperature_metpy,
                dewpoint_metpy,
                metpy_calc.wind_speed(u_ms, v_ms).to("knots"),
                metpy_calc.wind_direction(u_ms, v_ms),
            )

        # Catch any exceptions that occur during the SWEAT index computation and skip it
        except Exception:
            pass

    def _extract_pressure_profile(
        self: "SoundingDiagnostics",
        ds: xr.Dataset,
        time_dim: str,
        time_idx: int,
        cell_idx: int,
    ) -> np.ndarray:
        """
        This function extracts the pressure profile in hPa from the dataset for a given time index and cell index. It first checks for a direct 'pressure' variable, and if not found, it looks for 'pressure_p' and 'pressure_base' variables to compute the total pressure. If neither method is successful, it raises a ValueError indicating that the necessary pressure information is missing from the dataset. The extracted pressure values are returned as a 1D numpy array in hPa, ensuring that they are treated as floating-point values for accurate calculations in subsequent steps.

        Parameters:
            ds (xr.Dataset): The xarray Dataset containing the sounding data.
            time_dim (str): The name of the time dimension in the dataset.
            time_idx (int): The index along the time dimension to extract.
            cell_idx (int): The index of the cell for which to extract the profile.

        Returns:
            np.ndarray: Pressure profile in hPa as a 1D array.
        """
        if "pressure" in ds:
            # Extract the pressure values for the given time and cell index
            pressure_vals = ds["pressure"].isel(
                {time_dim: time_idx, "nCells": cell_idx}
            )

            # Return the pressure profile in hPa
            return cast(
                np.ndarray, np.asarray(pressure_vals.values, dtype=np.float64).ravel()
            )

        if "pressure_p" in ds and "pressure_base" in ds:
            # Extract the pressure perturbation for the given time and cell index
            pressure_perturbation = ds["pressure_p"].isel(
                {time_dim: time_idx, "nCells": cell_idx}
            )

            # Extract the base pressure for the given time and cell index
            base_pressure = ds["pressure_base"].isel(
                {time_dim: time_idx, "nCells": cell_idx}
            )

            # Return the total pressure profile in hPa
            return cast(
                np.ndarray,
                np.asarray(
                    (pressure_perturbation + base_pressure).values, dtype=np.float64
                ).ravel(),
            )

        # Raise an error if no pressure variable is found in the dataset
        raise ValueError(
            "Cannot determine pressure: dataset lacks 'pressure', "
            "'pressure_p', and 'pressure_base' variables."
        )

    def _extract_temperature_profile(
        self: "SoundingDiagnostics",
        ds: xr.Dataset,
        time_dim: str,
        time_idx: int,
        cell_idx: int,
        pressure_pa: np.ndarray,
    ) -> np.ndarray:
        """
        This function extracts the temperature profile in °C from the dataset for a given time index and cell index. It first checks for direct temperature variables (e.g., 'temperature', 'temp', 't') and returns their values converted from Kelvin to Celsius if found. If no direct temperature variable is available, it looks for potential temperature variables (e.g., 'theta', 'potential_temperature') and converts them to actual temperature using the provided pressure profile. If neither method is successful, it raises a ValueError indicating that the necessary temperature information is missing from the dataset. The extracted temperature values are returned as a 1D numpy array in °C, ensuring that they are treated as floating-point values for accurate calculations in subsequent steps.

        Parameters:
            ds (xr.Dataset): The xarray Dataset containing the sounding data.
            time_dim (str): The name of the time dimension in the dataset.
            time_idx (int): The index along the time dimension to extract.
            cell_idx (int): The index of the cell for which to extract the profile.
            pressure_pa (np.ndarray): Pressure profile in Pa, used for potential temperature conversion if needed.

        Returns:
            np.ndarray: Temperature profile in °C as a 1D array.
        """
        for name in ("temperature", "temp", "t"):
            if name in ds and "nVertLevels" in ds[name].dims:
                # Extract the temperature values for the given time and cell index
                data = ds[name].isel({time_dim: time_idx, "nCells": cell_idx})

                # Convert temperature to a numpy array of floats
                temperature_k = np.asarray(data.values, dtype=np.float64).ravel()

                # Convert from Kelvin to Celsius if the mean temperature is above 100 K
                return cast(np.ndarray, temperature_k - 273.15)  # K → °C

        for name in ("theta", "potential_temperature"):
            if name in ds and "nVertLevels" in ds[name].dims:
                # Extract the potential temperature values for the given time and cell index
                data = ds[name].isel({time_dim: time_idx, "nCells": cell_idx})

                # Convert potential temperature to a numpy array of floats
                potential_temp_k = np.asarray(data.values, dtype=np.float64).ravel()

                # Convert potential temperature to actual temperature
                temperature_k = self.potential_to_actual_temperature(
                    potential_temp_k, pressure_pa
                )

                # If verbose mode is enabled, print a message about converting potential temperature to actual temperature
                if self.verbose:
                    logger.debug(
                        "Converted '%s' (potential temp) to actual temperature", name
                    )

                # Return the temperature profile in °C
                return cast(np.ndarray, temperature_k - 273.15)

        # Raise an error if no temperature or potential temperature variable is found
        raise ValueError("Cannot find temperature or theta variable in dataset.")

    def _extract_dewpoint_profile(
        self: "SoundingDiagnostics",
        ds: xr.Dataset,
        time_dim: str,
        time_idx: int,
        cell_idx: int,
        pressure_pa: np.ndarray,
    ) -> np.ndarray:
        """
        This function extracts the dewpoint profile in °C from the dataset for a given time index and cell index. It first checks for direct dewpoint variables (e.g., 'dewpoint', 'td', 'dew_point_temperature') and returns their values converted from Kelvin to Celsius if found. If no direct dewpoint variable is available, it looks for mixing ratio or specific humidity variables (e.g., 'qv', 'specific_humidity') and computes the dewpoint temperature from them using the provided pressure profile. If neither method is successful, it returns an array of NaN values with the same shape as the pressure profile, indicating that the dewpoint could not be determined. The extracted or computed dewpoint values are returned as a 1D numpy array in °C, ensuring that they are treated as floating-point values for accurate calculations in subsequent steps.

        Parameters:
            ds (xr.Dataset): The xarray Dataset containing the sounding data.
            time_dim (str): The name of the time dimension in the dataset.
            time_idx (int): The index along the time dimension to extract.
            cell_idx (int): The index of the cell for which to extract the profile.
            pressure_pa (np.ndarray): Pressure profile in Pa, used for dewpoint computation if needed.

        Returns:
            np.ndarray: Dewpoint profile in °C as a 1D array, or NaN array if dewpoint cannot be determined.
        """
        for name in ("dewpoint", "td", "dew_point_temperature"):
            if name in ds and "nVertLevels" in ds[name].dims:
                # Extract the dewpoint values for the given time and cell index
                data = ds[name].isel({time_dim: time_idx, "nCells": cell_idx})
                dewpoint_vals = np.asarray(data.values, dtype=np.float64).ravel()

                # Convert from Kelvin to Celsius if the mean dewpoint is above 100 K
                if np.nanmean(dewpoint_vals) > 100:
                    dewpoint_vals = dewpoint_vals - 273.15

                # Return the dewpoint profile in °C
                return cast(np.ndarray, dewpoint_vals)

        for name in (
            "qv",
            "q_vapor",
            "scalars_qv",
            "specific_humidity",
            "vapor_mixing_ratio",
        ):
            if name in ds and "nVertLevels" in ds[name].dims:
                # Extract the mixing ratio or specific humidity values for the given time and cell index
                data = ds[name].isel({time_dim: time_idx, "nCells": cell_idx})

                # Convert mixing ratio or specific humidity to a non-negative numpy array
                qv = np.asarray(data.values, dtype=np.float64).ravel()
                qv = np.clip(qv, 0, None)

                # A true specific humidity q is converted to mixing ratio w = q / (1 - q); MPAS qv is already a mixing ratio
                if name == "specific_humidity":
                    qv = qv / (1.0 - np.clip(qv, 0, 0.99))

                # Print a message about computing dewpoint from mixing ratio if verbose mode is enabled
                if self.verbose:
                    logger.debug("Computing dewpoint from '%s' mixing ratio", name)

                # Compute dewpoint from mixing ratio and pressure
                return self.compute_dewpoint_from_mixing_ratio(qv, pressure_pa)

        # If no dewpoint or mixing ratio variables are found, return NaN
        if self.verbose:
            logger.warning("No moisture variable found; dewpoint set to NaN")

        # Return NaN array if dewpoint cannot be determined
        return cast(np.ndarray, np.full_like(pressure_pa, np.nan))

    def _extract_wind_profiles(
        self: "SoundingDiagnostics",
        ds: xr.Dataset,
        time_dim: str,
        time_idx: int,
        cell_idx: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        This function extracts the u and v wind component profiles in knots from the dataset for a given time index and cell index. It checks for common variable names for zonal and meridional wind components (e.g., 'uReconstructZonal', 'uzonal', 'u' for u-component and 'uReconstructMeridional', 'umeridional', 'v' for v-component) that have the expected dimensions. If found, it extracts their values, converts them from meters per second to knots, and returns them as 1D numpy arrays. If no suitable wind variables are found, it returns (None, None) and optionally prints a warning message if verbose mode is enabled. The extracted wind profiles are treated as floating-point values for accurate calculations in subsequent steps.

        Parameters:
            ds (xr.Dataset): The xarray Dataset containing the sounding data.
            time_dim (str): The name of the time dimension in the dataset.
            time_idx (int): The index along the time dimension to extract.
            cell_idx (int): The index of the cell for which to extract the profile.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: u and v wind profiles in knots as 1D arrays, or (None, None) if no suitable wind variables are found.
        """
        # Conversion factor from meters per second to knots
        ms_to_kt = 1.94384

        # Initialize u and v wind data as None, to be populated if suitable variables are found
        u_data: Optional[np.ndarray] = None
        v_data: Optional[np.ndarray] = None

        # Check for u-component wind variables and extract if found
        for u_name in ("uReconstructZonal", "uzonal", "u"):
            if (
                u_name in ds
                and "nVertLevels" in ds[u_name].dims
                and "nCells" in ds[u_name].dims
            ):
                raw = ds[u_name].isel({time_dim: time_idx, "nCells": cell_idx})
                u_data = np.asarray(raw.values, dtype=np.float64).ravel() * ms_to_kt
                break

        # Check for v-component wind variables and extract if found
        for v_name in ("uReconstructMeridional", "umeridional", "v"):
            if (
                v_name in ds
                and "nVertLevels" in ds[v_name].dims
                and "nCells" in ds[v_name].dims
            ):
                raw = ds[v_name].isel({time_dim: time_idx, "nCells": cell_idx})
                v_data = np.asarray(raw.values, dtype=np.float64).ravel() * ms_to_kt
                break

        # Print a warning if no wind variables were found for the sounding profile
        if u_data is None and v_data is None and self.verbose:
            logger.warning("No wind variables found for sounding")

        # Return the extracted u and v wind profiles in knots
        return u_data, v_data

    def _terrain_height_from_dataset(
        self: "SoundingDiagnostics",
        source: Any,
        cell_idx: int,
        use_ter: bool = True,
    ) -> Optional[float]:
        """
        This function reads the terrain height of one cell from an xarray dataset, trying the zgrid level-0 interface and then, when use_ter is True, 'ter'.

        Parameters:
            source (Any): An xarray dataset.
            cell_idx (int): Index of the cell whose terrain height is wanted.
            use_ter (bool): Whether to fall back to the 'ter' variable when there is no zgrid (default: True).

        Returns:
            Optional[float]: Terrain height in metres, or None.
        """
        if "zgrid" in source and "nCells" in source["zgrid"].dims:
            zg = source["zgrid"].isel({"nCells": cell_idx})
            for dim in zg.dims:
                if dim != "nVertLevelsP1":
                    zg = zg.isel({dim: 0})
            if "nVertLevelsP1" in zg.dims and zg.sizes["nVertLevelsP1"] > 1:
                return self._finite_or_none(zg.isel({"nVertLevelsP1": 0}).values)
        if use_ter and "ter" in source and "nCells" in source["ter"].dims:
            ter = source["ter"].isel({"nCells": cell_idx})
            for dim in ter.dims:
                ter = ter.isel({dim: 0})
            return self._finite_or_none(ter.values)
        return None

    def _extract_surface_height(
        self: "SoundingDiagnostics",
        ds: xr.Dataset,
        grid_file: Any,
        cell_idx: int,
    ) -> Optional[float]:
        """
        This function returns the terrain height (MSL, in metres) of one cell from the same source as the height profile: the lowest 'zgrid' interface (level 0 of nVertLevelsP1) of the dataset when it has zgrid, otherwise the static grid file's zgrid or 'ter'. A 'ter' variable in the data files is only the last resort, because MPAS output streams can carry it filled with zeros while the heights (from the grid file's zgrid) are above mean sea level, which would put the ground at sea level everywhere. None is returned when nothing is available or the value is not finite, in which case the index code measures above-ground layers from the lowest mass level instead.

        Parameters:
            ds (xr.Dataset): The dataset loaded by the processor.
            grid_file (Any): Path of the MPAS grid file, or None.
            cell_idx (int): Index of the cell whose terrain height is wanted.

        Returns:
            Optional[float]: Terrain height in metres above mean sea level, or None if unavailable.
        """

        try:
            value = self._terrain_height_from_dataset(ds, cell_idx, use_ter=False)
            if value is None and grid_file is not None:
                with xr.open_dataset(
                    grid_file, decode_times=False, engine=NETCDF_ENGINE
                ) as grid_ds:
                    value = self._terrain_height_from_dataset(grid_ds, cell_idx)
            if value is None:
                value = self._terrain_height_from_dataset(ds, cell_idx)
            return value
        except Exception:
            return None

    def _extract_height_profile_from_grid(
        self: "SoundingDiagnostics",
        grid_file: Any,
        cell_idx: int,
        n_levels: int,
    ) -> Optional[np.ndarray]:
        """
        This function derives the mass-level height profile of one cell from the 'zgrid' variable of the MPAS grid (static) file, which is where real MPAS runs keep the geometric height because mpasout and diag files do not contain it. 'zgrid' holds the heights of the nVertLevelsP1 layer interfaces above mean sea level; the heights of the nVertLevels mass levels are the averages of adjacent interfaces, the standard MPAS destaggering. The result is in the same order as the pressure and temperature profiles (level 0 is the lowest level) and is MSL, which the index code converts to above ground level by subtracting the lowest level. The profile is rejected (None) when its length differs from n_levels, so it cannot be misaligned with the other profiles, and None is also returned if the file is unavailable or has no zgrid.

        Parameters:
            grid_file (Any): Path of the MPAS grid file, or None.
            cell_idx (int): Index of the cell whose profile is wanted.
            n_levels (int): Number of mass levels of the sounding the height must align with.

        Returns:
            Optional[np.ndarray]: Height of the mass levels above mean sea level in metres, or None if it cannot be derived.
        """
        if grid_file is None:
            return None

        try:
            with xr.open_dataset(
                grid_file, decode_times=False, engine=NETCDF_ENGINE
            ) as grid_ds:
                if "zgrid" not in grid_ds or "nCells" not in grid_ds["zgrid"].dims:
                    return None
                interfaces = grid_ds["zgrid"].isel({"nCells": cell_idx})
                for dim in interfaces.dims:
                    if dim not in ("nVertLevelsP1", "nVertLevels"):
                        interfaces = interfaces.isel({dim: 0})
                values = np.asarray(interfaces.values, dtype=np.float64).ravel()
        except Exception:
            if self.verbose:
                logger.warning("Failed to read zgrid from grid file '%s'", grid_file)
            return None

        if len(values) == n_levels + 1:
            values = 0.5 * (values[:-1] + values[1:])

        if len(values) != n_levels:
            return None

        return cast(np.ndarray, values)

    def _extract_height_profile(
        self: "SoundingDiagnostics",
        ds: xr.Dataset,
        time_dim: str,
        time_idx: int,
        cell_idx: int,
    ) -> Optional[np.ndarray]:
        """
        This function extracts the height profile in meters from the dataset for a given time index and cell index. It checks for common variable names that could represent height (e.g., 'zgrid', 'height', 'height_agl') and attempts to extract their values. If the variable has a staggered grid with nVertLevelsP1, it performs mid-level averaging to convert it to nVertLevels. The extracted height values are returned as a 1D numpy array in meters. If no suitable height variable is found or if any error occurs during extraction, it returns None and optionally prints a warning message if verbose mode is enabled. The height profile is treated as floating-point values for accurate calculations in subsequent steps.

        Parameters:
            ds (xr.Dataset): The xarray Dataset containing the sounding data.
            time_dim (str): The name of the time dimension in the dataset.
            time_idx (int): The index along the time dimension to extract.
            cell_idx (int): The index of the cell for which to extract the profile.

        Returns:
            Optional[np.ndarray]: Height profile in meters as a 1D array, or None if no suitable height variable is found or an error occurs during extraction.
        """
        # Check for common height variable names and attempt to extract the height profile
        for name in ("zgrid", "height", "height_agl"):
            # Check if the variable exists in the dataset
            if name not in ds:
                continue

            try:
                # Build the selection dictionary for isel, starting with the cell index
                isel_dict: Dict[str, int] = {"nCells": cell_idx}

                # If the height variable has a time dimension, include it in the selection
                if time_dim in ds[name].dims:
                    isel_dict[time_dim] = time_idx

                # Extract the raw height values for the specified time and cell index
                raw = ds[name].isel(isel_dict)
                height_vals = np.asarray(raw.values, dtype=np.float64).ravel()

                # Staggered grid: nVertLevelsP1 → nVertLevels via mid-level averaging
                vert_dim = "nVertLevels"
                nlevs = ds.sizes.get(vert_dim, None)

                # If the height variable has one more level than the pressure/temperature variables, perform mid-level averaging
                if nlevs is not None and len(height_vals) == nlevs + 1:
                    height_vals = 0.5 * (height_vals[:-1] + height_vals[1:])

                # Return the extracted height profile in meters
                return cast(np.ndarray, height_vals)

            # Catch any exceptions that occur during height extraction and return None
            except Exception:
                if self.verbose:
                    logger.warning("Failed to extract height profile from '%s'", name)
                continue
        return None

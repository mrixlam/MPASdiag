#!/usr/bin/env python3

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
import numpy as np
import xarray as xr
from types import ModuleType
from scipy.spatial import KDTree as cKDTree
from typing import TYPE_CHECKING, Dict, Any, Optional, Tuple, cast

# Import MPASdiag utilities and base classes
from mpasdiag.processing.utils_datetime import MPASDateTimeUtils
from mpasdiag.processing.processors_3d import MPAS3DProcessor
from mpasdiag.processing.base import MPASBaseProcessor
from mpasdiag.processing.constants import P0_REF_PA, KAPPA, EPSILON_RD_RV

if TYPE_CHECKING:
    pass

_mpcalc: Optional[ModuleType] = None
_munits: Any = None

try:
    import metpy.calc as _mpcalc  # type: ignore[no-redef]
    from metpy.units import units as _munits  # type: ignore[no-redef]
    HAS_METPY = True
    mpcalc: Any = _mpcalc
    munits: Any = _munits
except ImportError:
    HAS_METPY = False


class SoundingDiagnostics:
    """ Computes sounding-related diagnostics for MPAS model output, including profile extraction, thermodynamic index calculation, and data quality checks. """

    def __init__(self: 'SoundingDiagnostics', 
                 verbose: bool = True) -> None:
        """
        This function initializes the SoundingDiagnostics class with optional verbose output. The verbose flag controls whether detailed messages about the sounding extraction process and index calculations are printed to the console. When enabled, it provides insights into the nearest cell selection, profile characteristics, and any issues encountered during computations. This can be helpful for debugging and understanding the results, especially when working with complex MPAS datasets. 

        Parameters:
            verbose (bool): Enable verbose output messages during sounding extraction (default: True).

        Returns:
            None
        """
        # Store the verbose flag for use in other methods
        self.verbose = verbose

    def extract_sounding_profile(self: 'SoundingDiagnostics', 
                                 processor: 'MPAS3DProcessor', 
                                 lon: float, 
                                 lat: float, 
                                 time_index: int = 0,) -> Dict[str, Any]:
        """
        This function extracts the sounding profile at a specified longitude, latitude, and time index from the MPAS dataset using the provided processor. It first checks that the dataset is loaded, then retrieves the grid coordinates and finds the nearest cell to the target location. It extracts the pressure profile, temperature profile (converting from potential temperature if necessary), dewpoint profile (computing from mixing ratio if necessary), wind profiles (u and v components), and optionally height profile if available. The function sorts the profiles from surface to top and filters out any non-finite values. Finally, it returns a dictionary containing all extracted variables along with metadata about the station location and time index. This allows users to analyze the vertical structure of the atmosphere at specific locations in their MPAS simulations. 

        Parameters:
            processor (MPAS3DProcessor): The MPAS data processor instance containing the dataset and grid information.
            lon (float): Target longitude in degrees for sounding extraction.
            lat (float): Target latitude in degrees for sounding extraction.
            time_index (int): Time index to extract the profile from (default: 0). 

        Returns:
            Dict[str, Any]: A dictionary containing the extracted sounding profile with keys such as 'pressure', 'temperature', 'dewpoint', 'u_wind', 'v_wind', 'height', 'station_lon', 'station_lat', 'cell_index', and 'time_index'.
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

        # Identify the nearest cell to the target longitude and latitude
        grid_lon, grid_lat = self._load_grid_coordinates(processor)
        cell_idx = self._find_nearest_cell(grid_lon, grid_lat, lon, lat)

        # Get the longitude and latitude of the nearest cell for metadata
        station_lon = float(grid_lon[cell_idx])
        station_lat = float(grid_lat[cell_idx])

        # Print nearest cell information if verbose mode is enabled
        if self.verbose:
            dist_km = self._haversine_km(lon, lat, station_lon, station_lat)
            print(f"Nearest cell {cell_idx}: ({station_lon:.4f}°, {station_lat:.4f}°), "
                  f"distance {dist_km:.1f} km from target")

        # Extract pressure profile, converting from Pa to hPa for consistency with meteorological conventions
        pressure_pa = self._extract_pressure_profile(ds, time_dim, validated_time_index, cell_idx)
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

        # Extract height profile if available
        height_m = self._extract_height_profile(ds, time_dim, validated_time_index, cell_idx)

        # Sort profiles from surface to top based on pressure 
        sort_idx = np.argsort(pressure_hpa)[::-1]

        # Sort pressure profile from surface to top
        pressure_hpa = pressure_hpa[sort_idx]

        # Temperature is required, so sort it along with pressure
        temperature_c = temperature_c[sort_idx]

        # Dewpoint is optional, so only sort if it exists
        dewpoint_c = dewpoint_c[sort_idx]

        # u_kt is optional, so only sort if it exists 
        if u_kt is not None:
            u_kt = u_kt[sort_idx]

        # v_kt is optional, so only sort if it exists 
        if v_kt is not None:
            v_kt = v_kt[sort_idx]

        # Height is optional, so only sort if it exists
        if height_m is not None:
            height_m = height_m[sort_idx]

        # Filter out any non-finite values from the profiles
        valid = np.isfinite(pressure_hpa) & np.isfinite(temperature_c)

        if not np.all(valid):
            # Filter pressure only if it has valid values
            pressure_hpa = pressure_hpa[valid]

            # Filter temperature only if it has valid values
            temperature_c = temperature_c[valid]

            # Filter dewpoint only if it has valid values
            dewpoint_c = dewpoint_c[valid]

            # u_kt is optional, so only filter if it exists 
            if u_kt is not None:
                u_kt = u_kt[valid]

            # v_kt is optional, so only filter if it exists 
            if v_kt is not None:
                v_kt = v_kt[valid]

            # Height is optional, so only filter if it exists
            if height_m is not None:
                height_m = height_m[valid]

        # Print profile summary if verbose mode is enabled
        if self.verbose:
            print(f"Sounding profile: {len(pressure_hpa)} levels, "
                  f"P range {pressure_hpa[-1]:.1f}–{pressure_hpa[0]:.1f} hPa, "
                  f"T range {np.nanmin(temperature_c):.1f}–{np.nanmax(temperature_c):.1f} °C")

        # Return the extracted profile and metadata as a dictionary
        return {
            'pressure': pressure_hpa,
            'temperature': temperature_c,
            'dewpoint': dewpoint_c,
            'u_wind': u_kt,
            'v_wind': v_kt,
            'height': height_m,
            'station_lon': station_lon,
            'station_lat': station_lat,
            'cell_index': cell_idx,
            'time_index': validated_time_index,
        }

    def compute_dewpoint_from_mixing_ratio(self: 'SoundingDiagnostics', 
                                           mixing_ratio: np.ndarray, 
                                           pressure_pa: np.ndarray,) -> np.ndarray:
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
        vapour_pressure = np.clip(mixing_ratio_f64 * pressure_f64 / (EPSILON_RD_RV + mixing_ratio_f64), 1e-10, None)

        # Magnus formula: compute dewpoint in °C from vapour pressure in Pa
        ln_ratio = np.log(vapour_pressure / 611.2)
        dewpoint_c = 243.5 * ln_ratio / (17.67 - ln_ratio)

        # Return the computed dewpoint profile in degrees Celsius
        return dewpoint_c

    def compute_thermodynamic_indices(self: 'SoundingDiagnostics', 
                                      pressure_hpa: np.ndarray, 
                                      temperature_c: np.ndarray, 
                                      dewpoint_c: np.ndarray, 
                                      u_wind_kt: Optional[np.ndarray] = None, 
                                      v_wind_kt: Optional[np.ndarray] = None, 
                                      height_m: Optional[np.ndarray] = None,) -> Dict[str, Optional[float]]:
        """
        This function computes a comprehensive set of thermodynamic and severe weather indices from the provided sounding profile variables using MetPy's calculation functions. It calculates indices such as CAPE, CIN, lifted index, K-index, total totals, showalter index, cross totals, precipitable water, wet bulb zero height, and various shear-related indices if wind data is available. The function uses helper methods to safely call MetPy functions and handle any exceptions that may arise during calculations, returning None for indices that cannot be computed due to data limitations. This allows for robust analysis of the sounding profile's thermodynamic structure and severe weather potential even when some profile data may be missing or invalid. 

        Parameters:
            pressure_hpa (np.ndarray): Pressure profile in hPa.
            temperature_c (np.ndarray): Temperature profile in °C.
            dewpoint_c (np.ndarray): Dewpoint profile in °C.
            u_wind_kt (Optional[np.ndarray]): U-component of wind profile in knots (optional).
            v_wind_kt (Optional[np.ndarray]): V-component of wind profile in knots (optional).
            height_m (Optional[np.ndarray]): Height profile in meters (optional). 

        Returns:
            Dict[str, Optional[float]]: A dictionary containing computed indices such as 'cape', 'cin', 'lifted_index', 'k_index', etc., with float values on success or None for indices that could not be computed. 
        """
        # Define the list of all indices to compute
        _ALL_KEYS = [
            'cape', 'cin', 'sbcape', 'sbcin', 'mlcape', 'mlcin',
            'mucape', 'mucin', 'lifted_index', 'dcape',
            'lcl_pressure', 'lcl_temperature', 'lfc_pressure', 'el_pressure',
            'k_index', 'total_totals', 'showalter_index', 'cross_totals',
            'precipitable_water', 'wet_bulb_zero_height',
            'bulk_shear_0_1km', 'bulk_shear_0_6km',
            'srh_0_1km', 'srh_0_3km', 'stp', 'scp', 'sweat_index',
        ]
        
        # Initialize the result dictionary with all keys set to None by default
        result: Dict[str, Optional[float]] = dict.fromkeys(_ALL_KEYS, None)

        # If MetPy is not available, compute a fallback LCL estimate and skip the rest of the indices
        if not HAS_METPY:
            self._compute_fallback_lcl(
                pressure_hpa, temperature_c, dewpoint_c, result)
            return result

        # Create local aliases for MetPy calculation functions and units
        metpy_calc: Any = mpcalc
        metpy_units: Any = munits

        # Compute the parcel profile using the surface pressure, temperature, and dewpoint
        try:
            pressure_metpy = pressure_hpa * metpy_units.hPa
            temperature_metpy = temperature_c * metpy_units.degC
            dewpoint_metpy = dewpoint_c * metpy_units.degC
            parcel_profile = cast(Any, metpy_calc.parcel_profile(pressure_metpy, temperature_metpy[0], dewpoint_metpy[0])).to('degC')

        # Catch any exceptions that occur during the base profile computation
        except Exception as exc:
            if self.verbose:
                print(f"MetPy base profile computation failed: {exc}")
            return result

        # --- Lifting condensation level (LCL) ---
        result['lcl_pressure'], result['lcl_temperature'] = (
            self._safe_pair(metpy_calc.lcl, pressure_metpy[0], temperature_metpy[0], dewpoint_metpy[0]))

        # --- Level of free convection (LFC) ---
        result['lfc_pressure'], _ = self._safe_pair(metpy_calc.lfc, pressure_metpy, temperature_metpy, dewpoint_metpy)

        # --- Equilibrium level (EL) ---
        result['el_pressure'], _ = self._safe_pair(metpy_calc.el, pressure_metpy, temperature_metpy, dewpoint_metpy)

        # --- Surface-based CAPE/CIN ---
        result['sbcape'], result['sbcin'] = (
            self._safe_pair(metpy_calc.surface_based_cape_cin, pressure_metpy, temperature_metpy, dewpoint_metpy))

        # --- Surface-based CAPE/CIN fallback for backward compatibility ---
        if result['sbcape'] is None:
            result['sbcape'], result['sbcin'] = (
                self._safe_pair(metpy_calc.cape_cin, pressure_metpy, temperature_metpy, dewpoint_metpy, parcel_profile))

        # --- For backward compatibility, also store surface-based CAPE/CIN under 'cape' and 'cin' keys ---
        result['cape'], result['cin'] = result['sbcape'], result['sbcin']

        # --- Mixed layer CAPE/CIN ---
        result['mlcape'], result['mlcin'] = (
            self._safe_pair(metpy_calc.mixed_layer_cape_cin, pressure_metpy, temperature_metpy, dewpoint_metpy))

        # --- Most unstable CAPE/CIN (using the level with the highest equivalent potential temperature) ---
        result['mucape'], result['mucin'] = (
            self._safe_pair(metpy_calc.most_unstable_cape_cin, pressure_metpy, temperature_metpy, dewpoint_metpy))

        # --- Downdraft CAPE ---
        result['dcape'], _ = self._safe_pair(metpy_calc.downdraft_cape, pressure_metpy, temperature_metpy, dewpoint_metpy)

        # --- Scalar stability indices ---
        result['lifted_index'] = self._safe_scalar(
            metpy_calc.lifted_index, pressure_metpy, temperature_metpy, parcel_profile)

        # --- K-index ---
        result['k_index'] = self._safe_scalar(metpy_calc.k_index, pressure_metpy, temperature_metpy, dewpoint_metpy)

        # --- Total totals index ---
        result['total_totals'] = self._safe_scalar(
            metpy_calc.total_totals_index, pressure_metpy, temperature_metpy, dewpoint_metpy)

        # --- Showalter index ---
        result['showalter_index'] = self._safe_scalar(
            metpy_calc.showalter_index, pressure_metpy, temperature_metpy, dewpoint_metpy)

        # --- Cross totals index ---
        result['cross_totals'] = self._safe_scalar(
            metpy_calc.cross_totals, pressure_metpy, temperature_metpy, dewpoint_metpy)

        # --- Precipitable water ---
        result['precipitable_water'] = self._safe_scalar(
            metpy_calc.precipitable_water, pressure_metpy, dewpoint_metpy, to_unit='mm')

        # --- Wet-bulb zero height ---
        if height_m is not None:
            result['wet_bulb_zero_height'] = self._compute_wet_bulb_zero(
                pressure_metpy, temperature_metpy, dewpoint_metpy, height_m)

        # --- Shear and storm-relative indices (require wind and height) ---
        if (u_wind_kt is not None and v_wind_kt is not None
                and height_m is not None):
            self._compute_shear_indices(
                result, pressure_hpa, temperature_c, dewpoint_c,
                u_wind_kt, v_wind_kt, height_m)

        # Return the dictionary of computed indices
        return result

    @staticmethod
    def potential_to_actual_temperature(theta: np.ndarray, 
                                        pressure_pa: np.ndarray,) -> np.ndarray:
        """
        This function converts potential temperature (in K) to actual (sensible) temperature (in K) using the formula T = θ * (p / p0)^kappa, where p0 is the reference pressure (1000 hPa) and kappa is the Poisson constant (R_d / c_p). The function ensures that the input arrays are treated as floating-point values for accurate calculations. This allows for deriving the actual temperature profile when only potential temperature and pressure are available in the MPAS dataset. 

        Parameters:
            theta (np.ndarray): Potential temperature profile in K.
            pressure_pa (np.ndarray): Pressure profile in Pa. 

        Returns:
            np.ndarray: Actual (sensible) temperature in K.
        """
        # Return the actual temperature profile in K by applying the Poisson equation
        return np.asarray(theta, dtype=np.float64) * (
            np.asarray(pressure_pa, dtype=np.float64) / P0_REF_PA
        ) ** KAPPA

    def _load_grid_coordinates(self: 'SoundingDiagnostics', 
                               processor: 'MPAS3DProcessor') -> Tuple[np.ndarray, np.ndarray]:
        """
        This function loads the longitude and latitude coordinates of the cell centers from the MPAS grid file using xarray. It checks for common variable names for longitude and latitude in the grid dataset, and extracts them as 1D arrays. If the coordinates are in radians, it converts them to degrees. Finally, it ensures that longitudes are in the range [-180, 180] degrees before returning the longitude and latitude arrays. This is used to find the nearest cell to a target location when extracting sounding profiles. 

        Parameters:
            processor (MPAS3DProcessor): The MPAS data processor instance containing the grid file.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Longitude and latitude arrays in degrees.
        """
        # Only load lon/lat coordinate variables from the grid file
        _SOUNDING_COORD_NAMES = ['lonCell', 'longitude', 'lon', 'latCell', 'latitude', 'lat']
        open_kwargs: dict = {'decode_times': False}

        try:
            with xr.open_dataset(processor.grid_file, decode_times=False) as probe:
                all_vars = list(probe.data_vars)
            drop = [v for v in all_vars if v not in _SOUNDING_COORD_NAMES]
            if drop:
                open_kwargs['drop_variables'] = drop
        except Exception:
            pass
        
        with xr.open_dataset(processor.grid_file, **open_kwargs) as grid_ds:
            # Check for common longitude variable names and extract the longitude array
            for lon_name in ('lonCell', 'longitude', 'lon'):
                if lon_name in grid_ds.coords or lon_name in grid_ds.data_vars:
                    lon = grid_ds[lon_name].values.ravel()
                    break
            else:
                raise ValueError("Cannot find longitude coordinate in grid file")

            # Check for common latitude variable names and extract the latitude array
            for lat_name in ('latCell', 'latitude', 'lat'):
                if lat_name in grid_ds.coords or lat_name in grid_ds.data_vars:
                    lat = grid_ds[lat_name].values.ravel()
                    break
            else:
                raise ValueError("Cannot find latitude coordinate in grid file")

        # Convert from radians to degrees if necessary
        if np.nanmax(np.abs(lat)) <= np.pi:
            lon = np.degrees(lon)
            lat = np.degrees(lat)

        # Ensure longitudes are in the range [-180, 180]
        lon = ((lon + 180) % 360) - 180

        # Return the longitude and latitude arrays in degrees
        return lon, lat

    @staticmethod
    def _find_nearest_cell(grid_lon: np.ndarray, 
                           grid_lat: np.ndarray,
                           target_lon: float,
                           target_lat: float,) -> int:
        """
        This function finds the index of the nearest cell in the grid to a target longitude and latitude using a KDTree for efficient spatial querying. It first converts the grid coordinates and target location from spherical (longitude, latitude) to Cartesian (x, y, z) coordinates on the unit sphere, then builds a KDTree from the grid points and queries it with the target point to find the nearest neighbor. The function returns the index of the nearest cell in the grid, which can then be used to extract the sounding profile at that location. 

        Parameters:
            grid_lon (np.ndarray): 1D array of grid longitudes in degrees.
            grid_lat (np.ndarray): 1D array of grid latitudes in degrees.
            target_lon (float): Target longitude in degrees.
            target_lat (float): Target latitude in degrees. 

        Returns:
            int: Index of the nearest cell in the grid.
        """
        # Convert the grid longitude and latitude from degrees to radians
        lon_r = np.radians(grid_lon)
        lat_r = np.radians(grid_lat)

        # Convert the grid longitude and latitude to Cartesian coordinates on the unit sphere
        cart = np.column_stack([
            np.cos(lat_r) * np.cos(lon_r),
            np.cos(lat_r) * np.sin(lon_r),
            np.sin(lat_r),
        ])

        # Convert the target longitude and latitude from degrees to radians 
        tgt_lon_r = np.radians(target_lon)
        tgt_lat_r = np.radians(target_lat)

        # Convert the target longitude and latitude to Cartesian coordinates on the unit sphere
        tgt_cart = np.array([[
            np.cos(tgt_lat_r) * np.cos(tgt_lon_r),
            np.cos(tgt_lat_r) * np.sin(tgt_lon_r),
            np.sin(tgt_lat_r),
        ]])

        # Build a KDTree from the grid points and query it with the target point to find the nearest neighbor
        tree = cKDTree(cart)
        _, nearest_idx = tree.query(tgt_cart)

        # Return the index of the nearest cell as an integer
        return int(nearest_idx[0])

    @staticmethod
    def _haversine_km(lon1: float, 
                      lat1: float, 
                      lon2: float, 
                      lat2: float) -> float:
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
        haversine_term = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

        # Return the distance in kilometers
        return float(6371.0 * 2 * np.arcsin(np.sqrt(haversine_term)))

    @staticmethod
    def _safe_scalar(func: Any, 
                     *args: Any, 
                     to_unit: Optional[str] = None,) -> Optional[float]:
        """
        This function safely calls a MetPy calculation function that returns a single quantity (e.g., lifted index), handling any exceptions that may occur during the calculation. It attempts to call the provided function with the given arguments, optionally converting the result to a specified unit before extracting its magnitude. If any step of the process fails (e.g., due to insufficient profile levels, invalid values, or issues with unit conversion), it catches the exception and returns None instead of raising an error. This allows for robust computation of indices where some calculations may not be possible due to data limitations, while still providing valid results for other indices that can be computed successfully. 

        Parameters:
            func: MetPy calculation function that returns a single quantity (e.g., ``mpcalc.lifted_index``).
            *args: Positional arguments forwarded to *func*.
            to_unit: Optional unit string for conversion (e.g., 'K' or 'J/kg'). 

        Returns:
            Optional[float]: Computed index value on success, None on any failure.
        """
        try:
            # Compute the value using the provided function and arguments
            computed_val = func(*args)

            # If a unit conversion is requested, convert the result to the specified unit
            if to_unit is not None:
                computed_val = computed_val.to(to_unit)

            # Extract the magnitude of the result
            magnitude = computed_val.magnitude

            # Extract the magnitude as a float
            return float(magnitude.item()) if hasattr(magnitude, 'item') else float(magnitude)

        # Catch any exceptions that occur during the calculation and return None
        except Exception:
            return None

    @staticmethod
    def _safe_pair(func: Any, 
                   *args: Any, 
                   to_unit: Optional[str] = None,) -> Tuple[Optional[float], Optional[float]]:
        """
        This function safely calls a MetPy calculation function that returns a pair of quantities (e.g., CAPE and CIN), handling any exceptions that may occur during the calculation. It attempts to call the provided function with the given arguments, optionally converting both results to a specified unit before extracting their magnitudes. If any step of the process fails (e.g., due to insufficient profile levels, invalid values, or issues with unit conversion), it catches the exception and returns (None, None) instead of raising an error. This allows for robust computation of paired indices where some calculations may not be possible due to data limitations, while still providing valid results for other indices that can be computed successfully. 

        Parameters:
            func: MetPy calculation function that returns a pair of quantities (e.g., ``mpcalc.surface_based_cape_cin``).
            *args: Positional arguments forwarded to *func*.
            to_unit: Optional unit string for conversion (e.g., 'J/kg').

        Returns:
            Tuple[Optional[float], Optional[float]]: Pair of float values on success, (None, None) on any failure.
        """
        try:
            # Compute the pair of values using the provided function and arguments
            first_result, second_result = func(*args)

            # If a unit conversion is requested, convert both results to the specified unit
            if to_unit is not None:
                first_result, second_result = first_result.to(to_unit), second_result.to(to_unit)

            # Extract the magnitudes of both results and return them as floats
            return float(first_result.magnitude), float(second_result.magnitude)

        # Catch any exceptions that occur during the calculation and return (None, None)
        except Exception:
            return None, None

    def _compute_fallback_lcl(self: 'SoundingDiagnostics', 
                              pressure_hpa: np.ndarray, 
                              temperature_c: np.ndarray, 
                              dewpoint_c: np.ndarray, 
                              result: Dict[str, Optional[float]],) -> None:
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
            lcl_temperature = surface_dewpoint_c - (0.001296 * surface_dewpoint_c + 0.1963) * (surface_temp_c - surface_dewpoint_c)

            # Compute the LCL pressure using the Poisson equation with the estimated LCL temperature
            lcl_pressure = pressure_hpa[0] * (
                (lcl_temperature + 273.15) / (surface_temp_c + 273.15)) ** (1.0 / KAPPA)

            # Store the fallback LCL pressure and temperature in the result dictionary
            result['lcl_pressure'] = float(lcl_pressure)
            result['lcl_temperature'] = float(lcl_temperature)
        except Exception:
            pass

        # Print a message about the fallback LCL computation if verbose mode is enabled
        if self.verbose:
            print("MetPy not available – full index computation skipped. "
                  "Install metpy>=1.3.0 for comprehensive thermodynamic indices.")

    @staticmethod
    def _compute_wet_bulb_zero(pressure_metpy: Any,
                               temperature_metpy: Any,
                               dewpoint_metpy: Any,
                               height_m: np.ndarray,) -> Optional[float]:
        """
        This function computes the height of the wet bulb zero isotherm (the level where the wet bulb temperature reaches 0°C) using MetPy's wet bulb temperature calculation. It takes pressure, temperature, and dewpoint profiles with MetPy units, along with the corresponding height profile in meters. The function calculates the wet bulb temperature at each level, converts it to degrees Celsius, and then finds the first crossing of the 0°C isotherm. If a crossing is found, it performs a linear interpolation between the two levels that bracket the crossing to estimate the height of the wet bulb zero. If any step of the process fails (e.g., due to insufficient profile levels or invalid values), it returns None. This provides an important diagnostic for understanding freezing level conditions in the sounding profile. 

        Parameters:
            p: Pressure profile with MetPy units.
            T: Temperature profile with MetPy units.
            Td: Dewpoint profile with MetPy units.
            height_m: Height profile in meters (AGL).

        Returns:
            Optional[float]: Height of the wet bulb zero isotherm in meters, or None if computation failed.
        """
        # Create local alias for MetPy calculation functions
        metpy_calc: Any = mpcalc

        try:
            # Ensure the height profile is treated as a floating-point array for accurate calculations
            height_arr = np.asarray(height_m, dtype=np.float64)

            # Calculate the wet bulb temperature profile using MetPy's calculation function
            wet_bulb_metpy = cast(Any, metpy_calc.wet_bulb_temperature(pressure_metpy, temperature_metpy, dewpoint_metpy))

            # Convert the wet bulb temperature to degrees Celsius for zero crossing detection
            wet_bulb_celsius = np.asarray(wet_bulb_metpy.to('degC').magnitude, dtype=np.float64)

            # Find the indices where the wet bulb temperature crosses 0°C
            crossings = np.nonzero(np.diff(np.sign(wet_bulb_celsius)))[0]

            if len(crossings) > 0:
                # Get the index of the first crossing of the 0°C isotherm in the wet bulb temperature profile
                crossing_idx = crossings[0]

                # Compute the fraction of the way between the two levels where the wet bulb zero crossing occurs for linear interpolation
                frac = -wet_bulb_celsius[crossing_idx] / (wet_bulb_celsius[crossing_idx + 1] - wet_bulb_celsius[crossing_idx])

                # Return the interpolated height of the wet bulb zero isotherm in meters
                return float(height_arr[crossing_idx] + frac * (height_arr[crossing_idx + 1] - height_arr[crossing_idx]))
            
        # Catch any exceptions that occur during the wet bulb zero computation and return None
        except Exception:
            pass
        return None

    def _compute_shear_indices(self: 'SoundingDiagnostics', 
                               result: Dict[str, Optional[float]], 
                               pressure_hpa: np.ndarray, 
                               temperature_c: np.ndarray, 
                               dewpoint_c: np.ndarray, 
                               u_wind_kt: np.ndarray, 
                               v_wind_kt: np.ndarray, 
                               height_m: np.ndarray,) -> None:
        """
        This function computes shear-related indices such as bulk shear, storm-relative helicity, significant tornado parameter (STP), supercell composite parameter (SCP), and the SWEAT index using MetPy's calculation functions. It first converts the wind components from knots to meters per second and ensures that all input profiles are treated as floating-point values with appropriate units. It then calculates the bulk shear for specified depth ranges, storm-relative helicity for specified depth ranges, and uses these along with CAPE and LCL information to compute STP and SCP if the necessary dependencies are available. Finally, it computes the SWEAT index using the full profile of pressure, temperature, dewpoint, and wind. Each calculation is wrapped in a try-except block to handle any potential issues gracefully, allowing the function to return valid indices even if some calculations fail due to data limitations or invalid values. The computed indices are stored in the provided result dictionary under their respective keys. 

        Parameters:
            result (Dict[str, Optional[float]]): Output dictionary to store computed indices (modified in-place).
            pressure_hpa (np.ndarray): Pressure profile in hPa.
            temperature_c (np.ndarray): Temperature profile in °C.
            dewpoint_c (np.ndarray): Dewpoint profile in °C.
            u_wind_kt (np.ndarray): U-component of wind in knots.
            v_wind_kt (np.ndarray): V-component of wind in knots.
            height_m (np.ndarray): Height profile in meters.

        Returns:
            None
        """
        # Create local aliases for MetPy calculation functions and units
        metpy_calc: Any = mpcalc
        metpy_units: Any = munits

        try:
            kt_to_ms = 1.0 / 1.94384

            # Convert wind components from knots to meters per second
            u_ms = np.asarray(u_wind_kt, dtype=np.float64) * kt_to_ms * metpy_units('m/s')
            v_ms = np.asarray(v_wind_kt, dtype=np.float64) * kt_to_ms * metpy_units('m/s')

            # Ensure the height profile is treated as a floating-point array for accurate calculations and convert to MetPy units
            height_metpy = np.asarray(height_m, dtype=np.float64) * metpy_units.m

            # Convert pressure to MetPy units for the shear calculations
            pressure_metpy = pressure_hpa * metpy_units.hPa

        # Catch any exceptions that occur during the wind unit setup and skip shear-related calculations if it fails
        except Exception as exc:
            if self.verbose:
                print(f"MetPy wind unit setup failed: {exc}")
            return

        for depth_m, key in [(1000, 'bulk_shear_0_1km'),
                             (6000, 'bulk_shear_0_6km')]:
            try:
                # Calculate the bulk shear for the specified depth range using MetPy's calculation function
                bulk_shear_u, bulk_shear_v = cast(Any, metpy_calc.bulk_shear(
                    pressure_metpy, u_ms, v_ms, height=height_metpy,
                    depth=depth_m * metpy_units.m))

                # Store the magnitude of the bulk shear in knots in the result dictionary
                result[key] = float(
                    metpy_calc.wind_speed(bulk_shear_u, bulk_shear_v).to('knots').magnitude)

            # Catch any exceptions that occur during the bulk shear computation and skip it
            except Exception:
                pass

        for depth_m, key in [(1000, 'srh_0_1km'),
                             (3000, 'srh_0_3km')]:
            try:
                # Calculate the storm-relative helicity for the specified depth range using MetPy's calculation function
                _, _, srh = cast(Any, metpy_calc.storm_relative_helicity(
                    height_metpy, u_ms, v_ms, depth=depth_m * metpy_units.m))

                # Store the magnitude of the storm-relative helicity in the result dictionary
                result[key] = float(srh.magnitude)

            # Catch any exceptions that occur during the storm-relative helicity computation and skip it
            except Exception:
                pass

        try:
            # Check the dependencies for Significant Tornado Parameter: surface-based CAPE, LCL pressure, 0-1 km storm-relative helicity, and 0-6 km bulk shear
            stp_deps = [result.get('sbcape'), result.get('lcl_pressure'),
                        result.get('srh_0_1km'), result.get('bulk_shear_0_6km')]

            if all(v is not None for v in stp_deps):
                # Calculate the height of the surface above ground level in meters for the STP calculation
                surface_height_m = float(height_metpy[0].magnitude)

                # Find the index of the level closest to the LCL pressure for the STP calculation
                lcl_idx = int(np.argmin(
                    np.abs(pressure_hpa - stp_deps[1])))  # type: ignore[arg-type]

                # Calculate the height of the LCL above the surface in meters for the STP calculation
                lcl_height_m = float(height_metpy[lcl_idx].magnitude) - surface_height_m

                # Only attempt to compute STP if all dependencies are available, otherwise skip it
                result['stp'] = self._safe_scalar(
                    metpy_calc.significant_tornado,
                    stp_deps[0] * metpy_units('J/kg'), lcl_height_m * metpy_units.m,
                    stp_deps[2] * metpy_units('m^2/s^2'), stp_deps[3] * metpy_units.knots)

        # Catch any exceptions that occur during the STP computation and skip it
        except Exception:
            pass

        try:
            # Check the dependencies for Supercell Composite Parameter: most unstable CAPE, 0-3 km storm-relative helicity, and 0-6 km bulk shear
            scp_deps = [result.get('mucape'), result.get('srh_0_3km'),
                        result.get('bulk_shear_0_6km')]

            # Only attempt to compute SCP if all dependencies are available, otherwise skip it
            if all(v is not None for v in scp_deps):
                result['scp'] = self._safe_scalar(
                    metpy_calc.supercell_composite,
                    scp_deps[0] * metpy_units('J/kg'),
                    scp_deps[1] * metpy_units('m^2/s^2'),
                    scp_deps[2] * metpy_units.knots)

        # Catch any exceptions that occur during the STP and SCP computations and skip them
        except Exception:
            pass

        try:
            # Generate the full profile of temperature and dewpoint with units for SWEAT calculation
            temperature_metpy = temperature_c * metpy_units.degC
            dewpoint_metpy = dewpoint_c * metpy_units.degC

            # Rebuild the temperature profile with units for SWEAT calculation
            result['sweat_index'] = self._safe_scalar(
                metpy_calc.sweat_index, pressure_metpy, temperature_metpy, dewpoint_metpy, u_ms, v_ms)

        # Catch any exceptions that occur during the SWEAT index computation and skip it
        except Exception:
            pass

    def _extract_pressure_profile(self: 'SoundingDiagnostics', 
                                  ds: xr.Dataset, 
                                  time_dim: str, 
                                  time_idx: int, 
                                  cell_idx: int) -> np.ndarray:
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
        if 'pressure' in ds:
            # Extract the pressure values for the given time and cell index
            pressure_vals = ds['pressure'].isel({time_dim: time_idx, 'nCells': cell_idx})

            # Return the pressure profile in hPa
            return np.asarray(pressure_vals.values, dtype=np.float64).ravel()

        if 'pressure_p' in ds and 'pressure_base' in ds:
            # Extract the pressure perturbation for the given time and cell index
            pressure_perturbation = ds['pressure_p'].isel({time_dim: time_idx, 'nCells': cell_idx})

            # Extract the base pressure for the given time and cell index
            base_pressure = ds['pressure_base'].isel({time_dim: time_idx, 'nCells': cell_idx})

            # Return the total pressure profile in hPa
            return np.asarray((pressure_perturbation + base_pressure).values, dtype=np.float64).ravel()

        # Raise an error if no pressure variable is found in the dataset
        raise ValueError("Cannot determine pressure: dataset lacks 'pressure', "
                         "'pressure_p', and 'pressure_base' variables.")

    def _extract_temperature_profile(self: 'SoundingDiagnostics',
                                     ds: xr.Dataset, 
                                     time_dim: str, 
                                     time_idx: int, 
                                     cell_idx: int, 
                                     pressure_pa: np.ndarray,) -> np.ndarray:
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
        for name in ('temperature', 'temp', 't'):
            if name in ds and 'nVertLevels' in ds[name].dims:
                # Extract the temperature values for the given time and cell index
                data = ds[name].isel({time_dim: time_idx, 'nCells': cell_idx})

                # Convert temperature to a numpy array of floats
                temperature_k = np.asarray(data.values, dtype=np.float64).ravel()

                # Convert from Kelvin to Celsius if the mean temperature is above 100 K
                return temperature_k - 273.15  # K → °C

        for name in ('theta', 'potential_temperature'):
            if name in ds and 'nVertLevels' in ds[name].dims:
                # Extract the potential temperature values for the given time and cell index
                data = ds[name].isel({time_dim: time_idx, 'nCells': cell_idx})

                # Convert potential temperature to a numpy array of floats
                potential_temp_k = np.asarray(data.values, dtype=np.float64).ravel()

                # Convert potential temperature to actual temperature
                temperature_k = self.potential_to_actual_temperature(potential_temp_k, pressure_pa)

                # If verbose mode is enabled, print a message about converting potential temperature to actual temperature
                if self.verbose:
                    print(f"Converted '{name}' (potential temp) to actual temperature")

                # Return the temperature profile in °C
                return temperature_k - 273.15

        # Raise an error if no temperature or potential temperature variable is found
        raise ValueError("Cannot find temperature or theta variable in dataset.")

    def _extract_dewpoint_profile(self: 'SoundingDiagnostics',
                                  ds: xr.Dataset, 
                                  time_dim: str, 
                                  time_idx: int, 
                                  cell_idx: int, 
                                  pressure_pa: np.ndarray,) -> np.ndarray:
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
        for name in ('dewpoint', 'td', 'dew_point_temperature'):
            if name in ds and 'nVertLevels' in ds[name].dims:
                # Extract the dewpoint values for the given time and cell index
                data = ds[name].isel({time_dim: time_idx, 'nCells': cell_idx})
                dewpoint_vals = np.asarray(data.values, dtype=np.float64).ravel()

                # Convert from Kelvin to Celsius if the mean dewpoint is above 100 K
                if np.nanmean(dewpoint_vals) > 100:
                    dewpoint_vals = dewpoint_vals - 273.15

                # Return the dewpoint profile in °C
                return dewpoint_vals

        for name in ('qv', 'q_vapor', 'scalars_qv', 'specific_humidity', 'vapor_mixing_ratio'):
            if name in ds and 'nVertLevels' in ds[name].dims:
                # Extract the mixing ratio or specific humidity values for the given time and cell index
                data = ds[name].isel({time_dim: time_idx, 'nCells': cell_idx})

                # Convert mixing ratio or specific humidity to a non-negative numpy array 
                qv = np.asarray(data.values, dtype=np.float64).ravel()
                qv = np.clip(qv, 0, None)

                # Print a message about computing dewpoint from mixing ratio if verbose mode is enabled
                if self.verbose:
                    print(f"Computing dewpoint from '{name}' mixing ratio")
                
                # Compute dewpoint from mixing ratio and pressure
                return self.compute_dewpoint_from_mixing_ratio(qv, pressure_pa)

        # If no dewpoint or mixing ratio variables are found, return NaN
        if self.verbose:
            print("Warning: No moisture variable found; dewpoint set to NaN.")

        # Return NaN array if dewpoint cannot be determined
        return np.full_like(pressure_pa, np.nan)

    def _extract_wind_profiles(self: 'SoundingDiagnostics',
                               ds: xr.Dataset,
                               time_dim: str,
                               time_idx: int,
                               cell_idx: int,) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
        for u_name in ('uReconstructZonal', 'uzonal', 'u'):
            if u_name in ds and 'nVertLevels' in ds[u_name].dims and 'nCells' in ds[u_name].dims:
                raw = ds[u_name].isel({time_dim: time_idx, 'nCells': cell_idx})
                u_data = np.asarray(raw.values, dtype=np.float64).ravel() * ms_to_kt
                break
        
        # Check for v-component wind variables and extract if found 
        for v_name in ('uReconstructMeridional', 'umeridional', 'v'):
            if v_name in ds and 'nVertLevels' in ds[v_name].dims and 'nCells' in ds[v_name].dims:
                raw = ds[v_name].isel({time_dim: time_idx, 'nCells': cell_idx})
                v_data = np.asarray(raw.values, dtype=np.float64).ravel() * ms_to_kt
                break
        
        # Print a warning if no wind variables were found for the sounding profile
        if u_data is None and v_data is None and self.verbose:
            print("Warning: No wind variables found for sounding.")

        # Return the extracted u and v wind profiles in knots
        return u_data, v_data

    def _extract_height_profile(self: 'SoundingDiagnostics', 
                                ds: xr.Dataset, 
                                time_dim: str, 
                                time_idx: int, 
                                cell_idx: int,) -> Optional[np.ndarray]:
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
        for name in ('zgrid', 'height', 'height_agl'):
            # Check if the variable exists in the dataset
            if name not in ds:
                continue

            try:
                # Build the selection dictionary for isel, starting with the cell index
                isel_dict: Dict[str, int] = {'nCells': cell_idx}

                # If the height variable has a time dimension, include it in the selection
                if time_dim in ds[name].dims:
                    isel_dict[time_dim] = time_idx

                # Extract the raw height values for the specified time and cell index
                raw = ds[name].isel(isel_dict)
                height_vals = np.asarray(raw.values, dtype=np.float64).ravel()

                # Staggered grid: nVertLevelsP1 → nVertLevels via mid-level averaging
                vert_dim = 'nVertLevels'
                nlevs = ds.sizes.get(vert_dim, None)

                # If the height variable has one more level than the pressure/temperature variables, perform mid-level averaging
                if nlevs is not None and len(height_vals) == nlevs + 1:
                    height_vals = 0.5 * (height_vals[:-1] + height_vals[1:])

                # Return the extracted height profile in meters
                return height_vals

            # Catch any exceptions that occur during height extraction and return None 
            except Exception:
                if self.verbose:
                    print(f"Warning: Failed to extract height profile from '{name}'.")
                continue
        return None

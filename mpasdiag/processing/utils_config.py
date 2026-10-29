#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Core Processing Module: Configuration Management Utilities

This module defines the MPASConfig dataclass, which encapsulates all configuration parameters for MPAS analysis and plotting. It provides methods for validating spatial extent parameters, converting the configuration to and from dictionary format, and saving/loading configurations to YAML files. This structured approach to configuration management ensures that all parameters are organized, validated, and easily accessible throughout the processing workflow.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, fields

from .utils_logger import get_logger
from .utils_path import safe_resolve_within

logger = get_logger(__name__)

SUPPORTED_OUTPUT_FORMATS = ("png", "pdf", "svg", "eps", "jpg")
SUPPORTED_ACCUMULATION_PERIODS = ("a01h", "a03h", "a06h", "a12h", "a24h")
PRECIPITATION_PLOT_TYPES = ("scatter", "contourf")
PRECIPITATION_UNITS = ("mm", "in", "cm")
SURFACE_PLOT_TYPES = ("scatter", "contour", "contourf", "pcolormesh")
WIND_PLOT_TYPES = ("barbs", "arrows", "streamlines")
VERTICAL_COORDS = ("pressure", "modlev", "height")
CROSS_PLOT_STYLES = ("contourf", "contour", "pcolormesh")
COLORBAR_EXTENDS = ("both", "min", "max", "neither")
INTERPOLATION_METHODS = ("nearest", "linear", "cubic")

# YAML keys spelled like the command-line flags, mapped to the MPASConfig field they set
CONFIG_KEY_ALIASES = {
    "formats": "output_formats",
    "accumulation": "accumulation_period",
    "batch_all": "batch_mode",
    "threshold": "precip_threshold",
    "units": "precip_units",
    "subsample": "subsample_factor",
    "interpolation": "interpolation_method",
    "lon": "sounding_lon",
    "lat": "sounding_lat",
    "max_distance_km": "sounding_max_distance_km",
}

_TRUE_STRINGS = ("true", "yes")
_FALSE_STRINGS = ("false", "no")
_PRECIPITATION_ANALYSIS_TYPES = ("precipitation", "precip", "rain")
_SURFACE_ANALYSIS_TYPES = ("surface", "surf", "2d")


def _validate_config_path(
    filepath: str,
    base_dir: Optional[str],
    *,
    must_exist: bool,
) -> Path:
    """
    Resolve a user-supplied config path and confirm it is a YAML file contained within the trusted base directory before any filesystem access. Config paths can originate from CLI arguments, so this guards against path-traversal escapes (e.g. '../../etc/passwd' or absolute paths outside the working tree).

    Parameters:
        filepath (str): Raw config path supplied by the caller (relative or absolute).
        base_dir (Optional[str]): Directory the path must stay within. Defaults to the current working directory when None.
        must_exist (bool): When True (loading), require the path to be an existing regular file.

    Returns:
        Path: The resolved, validated absolute path.
    """
    return safe_resolve_within(
        filepath,
        base_dir,
        allowed_suffixes=(".yaml", ".yml"),
        must_exist=must_exist,
    )


@dataclass
class MPASConfig:
    """Configuration class for MPAS analysis parameters. The color limits clim_min and clim_max and the contour levels in levels are expressed in the plotted (display) units of the variable, for example degrees Celsius for t2m or hPa for mslp, after the automatic unit conversion; they are never converted themselves. The vector_color and vector_alpha settings give the color and opacity of the wind barbs and arrows (streamlines keep the speed colormap but honor the opacity)."""

    grid_file: str = ""
    data_dir: str = ""
    output_dir: str = ""

    base_dir: Optional[str] = None

    lat_min: Optional[float] = None
    lat_max: Optional[float] = None
    lon_min: Optional[float] = None
    lon_max: Optional[float] = None

    variable: str = "rainnc"
    accumulation_period: str = "a01h"

    plot_type: str = "scatter"
    time_index: int = 0
    title: Optional[str] = None
    clim_min: Optional[float] = None
    clim_max: Optional[float] = None
    output: Optional[str] = None
    figure_size: Tuple[float, float] = (12.0, 10.0)
    grid_resolution: Optional[float] = None
    regrid_method: str = "linear"

    projection: str = "PlateCarree"
    central_longitude: Optional[float] = None
    central_latitude: Optional[float] = None
    proj_kwargs: Optional[Dict[str, Any]] = None

    remap_engine: str = "kdtree"
    remap_method: str = "nearest"

    u_variable: str = "u10"
    v_variable: str = "v10"
    wind_level: str = "surface"
    wind_plot_type: str = "barbs"
    subsample_factor: int = -1
    wind_scale: Optional[float] = None
    show_background: bool = False
    background_colormap: str = "viridis"

    colormap: str = "default"
    figure_width: float = 10.0
    figure_height: float = 9.0
    dpi: int = 100
    output_formats: Optional[List[str]] = None

    use_pure_xarray: bool = False
    batch_mode: bool = False
    verbose: bool = False
    quiet: bool = False
    log_level: Optional[str] = None
    log_file: Optional[str] = None

    chunk_size: Optional[int] = None
    parallel: bool = False
    workers: Optional[int] = None

    analysis_type: Optional[str] = None
    time_start: Optional[int] = None
    time_end: Optional[int] = None

    precip_threshold: Optional[float] = None
    precip_units: str = "mm"

    interpolation_method: Optional[str] = None
    contour_levels: Optional[int] = None
    levels: Optional[List[float]] = None

    vector_color: str = "black"
    vector_alpha: float = 1.0

    start_lon: Optional[float] = None
    start_lat: Optional[float] = None
    end_lon: Optional[float] = None
    end_lat: Optional[float] = None
    vertical_coord: str = "pressure"
    num_points: int = 100
    max_height: Optional[float] = None
    plot_style: str = "contourf"
    extend: str = "both"

    overlay_type: Optional[str] = None
    primary_variable: Optional[str] = None
    secondary_variable: Optional[str] = None
    variables: Optional[List[str]] = None
    pressure_variable: str = "mslp"
    primary_colormap: Optional[str] = None
    secondary_colormap: Optional[str] = None
    transparency: float = 0.7
    contour_overlay: bool = False

    sounding_lon: Optional[float] = None
    sounding_lat: Optional[float] = None
    sounding_max_distance_km: Optional[float] = None
    show_indices: bool = False
    show_parcel: bool = False

    def __post_init__(self: "MPASConfig") -> None:
        """
        This method is automatically called after the dataclass is initialized. It performs validation and sets default values for certain parameters if they are not provided. Specifically, it checks if the output_formats parameter is None and sets it to a default list containing "png". It also calls the _validate_spatial_extent method to ensure that the spatial extent parameters (latitude and longitude bounds) are valid. If the spatial extent parameters are invalid, it raises a ValueError with an appropriate message. This post-initialization step ensures that the configuration object is in a consistent and valid state before it is used in any processing or plotting operations.

        Parameters:
            None

        Returns:
            None
        """
        if self.output_formats is None:
            self.output_formats = ["png"]

        self._coerce_booleans()
        self.revalidate()

    def revalidate(self: "MPASConfig") -> None:
        """
        This method revalidates the configuration parameters after the dataclass has been initialized. It checks the spatial extent parameters, remap settings, log level, numeric ranges, categorical choices (output formats, accumulation period, plot types), and the time selection to ensure that they are all valid. If any of these validations fail, it raises a ValueError with an appropriate message indicating the specific issue. This method can be called at any time to ensure that the configuration remains valid, especially if any parameters have been modified after initialization.

        Parameters:
            None

        Returns:
            None
        """
        # A subsample factor of 0 means "automatic", the same as -1
        if (
            isinstance(self.subsample_factor, int)
            and not isinstance(self.subsample_factor, bool)
            and self.subsample_factor == 0
        ):
            self.subsample_factor = -1

        extent_error = self._spatial_extent_error()
        if extent_error is not None:
            raise ValueError(extent_error)

        self._validate_remap_settings()
        self._validate_log_level()
        self._validate_numeric_ranges()
        self._validate_choices()
        self._validate_time_selection()

    def _coerce_booleans(self: "MPASConfig") -> None:
        """
        This method type-checks every boolean field of the configuration. Real booleans pass unchanged, the strings true/false/yes/no (any case, surrounding whitespace ignored) are converted, and any other value raises a ValueError naming the field, so a quoted YAML value such as show_background: "false" is neither silently truthy nor silently ignored.

        Parameters:
            None

        Returns:
            None
        """
        for f in fields(self):
            if f.type is not bool:
                continue
            value = getattr(self, f.name)
            if isinstance(value, bool):
                continue
            text = value.strip().lower() if isinstance(value, str) else None
            if text in _TRUE_STRINGS:
                setattr(self, f.name, True)
            elif text in _FALSE_STRINGS:
                setattr(self, f.name, False)
            else:
                raise ValueError(
                    f"{f.name} must be true or false (also accepted: yes/no), "
                    f"got {value!r}"
                )

    def resolved_base_dir(self: "MPASConfig") -> Path:
        """
        This method resolves the base directory for the configuration. If the base_dir attribute is set, it converts it to a Path object and resolves it to an absolute path. If base_dir is None, it defaults to the current working directory (Path.cwd()) and resolves that to an absolute path. This method ensures that any file paths used in the configuration are resolved relative to a trusted base directory, which helps prevent path traversal vulnerabilities.

        Parameters:
            None

        Returns:
            Path: The resolved base directory.
        """
        return (Path(self.base_dir) if self.base_dir else Path.cwd()).resolve()

    def _validate_numeric_ranges(self: "MPASConfig") -> None:
        """
        This method validates the numeric parameters of the configuration to ensure they fall within their allowed ranges. It checks parameters such as dpi, time_index (any integer; negative values are Python-style indices resolved against the dataset size once it is known), num_points, chunk_size, subsample_factor (-1 or 0 for automatic, where 0 is normalised to -1 by revalidate, or an integer >= 1), contour_levels, transparency and vector_alpha (within [0, 1]), the clim_min/clim_max ordering, and workers (if specified). If any of these parameters are out of their defined ranges or are not numeric types (int or float), it raises a ValueError with an appropriate message indicating the invalid parameter and its allowed range. This validation step is crucial for preventing runtime errors and ensuring that the configuration parameters are set to reasonable values for processing and plotting operations.

        Parameters:
            None

        Returns:
            None
        """

        def _check(name: str, value: Any, low: float, high: float) -> None:
            """
            This helper function checks if a numeric value falls within a specified range. If the value is None, it returns without performing any checks. If the value is not an int or float (or is a bool), it raises a ValueError. If the value is outside the specified range [low, high], it raises a ValueError indicating the allowed range.

            Parameters:
                name (str): The name of the parameter being checked (for error messages).
                value (Any): The value of the parameter to check.
                low (float): The lower bound of the allowed range.
                high (float): The upper bound of the allowed range.

            Returns:
                None
            """
            if value is None:
                return
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be a number, got {value!r}")
            if not (low <= value <= high):
                raise ValueError(
                    f"{name} ({value}) out of allowed range [{low}, {high}]"
                )

        _check("dpi", self.dpi, 10, 2_000)
        _check("time_index", self.time_index, -10_000_000, 10_000_000)
        _check("num_points", self.num_points, 1, 1_000_000)
        _check("chunk_size", self.chunk_size, 1, 1_000_000_000)
        _check("subsample_factor", self.subsample_factor, -1, 1_000_000)
        _check("contour_levels", self.contour_levels, 1, 10_000)
        _check("sounding_max_distance_km", self.sounding_max_distance_km, 0.0, 40_000.0)
        _check("transparency", self.transparency, 0.0, 1.0)
        _check("vector_alpha", self.vector_alpha, 0.0, 1.0)

        self._validate_subsample_and_limits()

        if self.workers is not None:
            _check("workers", self.workers, 1, 100_000)

        if self.figure_size is not None:
            if len(self.figure_size) != 2:
                raise ValueError("figure_size must have exactly two values")
            for dim in self.figure_size:
                _check("figure_size", dim, 0.1, 1_000.0)

    def _validate_subsample_and_limits(self: "MPASConfig") -> None:
        """
        This method validates the cross-field numeric rule that clim_min must not exceed clim_max when both color limits are specified, raising a ValueError naming the offending values. A wind subsample factor of 0 is not an error: revalidate normalises it to -1 (automatic) before this check runs.

        Parameters:
            None

        Returns:
            None
        """
        if (
            self.clim_min is not None
            and self.clim_max is not None
            and self.clim_min > self.clim_max
        ):
            raise ValueError(
                f"clim_min ({self.clim_min}) must not be greater than "
                f"clim_max ({self.clim_max})"
            )

    def _validate_log_level(self: "MPASConfig") -> None:
        """
        This method validates the log_level parameter of the configuration. It checks if log_level is None, in which case it simply returns without performing any validation (as it may be unspecified). If log_level is specified, it converts it to uppercase (if it's a string) and checks if it is one of the allowed log levels: "DEBUG", "INFO", "WARNING", or "ERROR". If the provided log_level is not valid, it raises a ValueError with a message indicating the invalid value and the allowed options. If the log_level is valid, it updates the log_level attribute to the uppercase version for consistency. This validation ensures that any logging operations using this configuration will have a valid log level set, preventing potential issues with logging output.

        Parameters:
            None

        Returns:
            None
        """
        if self.log_level is None:
            return

        allowed = {"DEBUG", "INFO", "WARNING", "ERROR"}
        level = self.log_level.upper() if isinstance(self.log_level, str) else None

        if level not in allowed:
            raise ValueError(
                f"Invalid log_level '{self.log_level}'. Must be one of: {sorted(allowed)}"
            )

        self.log_level = level

    def _spatial_extent_error(self: "MPASConfig") -> Optional[str]:
        """
        This method checks the spatial extent parameters (lon_min, lon_max, lat_min, lat_max) and returns a human-readable description of the first problem found, or None when the extent is valid. A bound left as None means unspecified and is resolved from the data extent at run time, so only the specified bounds are checked. Longitudes may lie anywhere in [-360, 360] so that dateline-crossing boxes (e.g. 150 to 210) and 0 to 360 domains are accepted, but lon_max must exceed lon_min and the longitude span must not exceed 360 degrees. Latitudes must lie in [-90, 90] with lat_max greater than lat_min. Every message names the offending values so users can correct them directly.

        Parameters:
            None

        Returns:
            Optional[str]: Description of the invalid extent naming the bad values, or None when the extent is valid or unspecified.
        """
        for name in ("lon_min", "lon_max", "lat_min", "lat_max"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return f"Invalid spatial extent: {name} must be a number, got {value!r}"

        for names, low, high, label in (
            (("lon_min", "lon_max"), -360.0, 360.0, "longitude"),
            (("lat_min", "lat_max"), -90.0, 90.0, "latitude"),
        ):
            for name in names:
                value = getattr(self, name)
                if value is not None and not (low <= value <= high):
                    return (
                        f"Invalid spatial extent: {name}={value} is outside the "
                        f"allowed {label} range [{int(low)}, {int(high)}]"
                    )

        if (
            self.lon_min is not None
            and self.lon_max is not None
            and self.lon_max <= self.lon_min
        ):
            return (
                f"Invalid spatial extent: lon_max ({self.lon_max}) must be greater "
                f"than lon_min ({self.lon_min})"
            )

        if (
            self.lon_min is not None
            and self.lon_max is not None
            and self.lon_max - self.lon_min > 360.0
        ):
            return (
                f"Invalid spatial extent: longitude span "
                f"{self.lon_max - self.lon_min} (lon_min={self.lon_min}, "
                f"lon_max={self.lon_max}) exceeds 360 degrees"
            )

        if (
            self.lat_min is not None
            and self.lat_max is not None
            and self.lat_max <= self.lat_min
        ):
            return (
                f"Invalid spatial extent: lat_max ({self.lat_max}) must be greater "
                f"than lat_min ({self.lat_min})"
            )

        return None

    def _validate_spatial_extent(self: "MPASConfig") -> bool:
        """
        This method validates the spatial extent parameters (latitude and longitude bounds) of the configuration and reports the result as a boolean. Unspecified (None) bounds are valid because they are resolved from the data extent at run time. Specified bounds must satisfy the same rules the plotters accept: longitudes within [-360, 360] with lon_max greater than lon_min and a span of at most 360 degrees (so 0 to 360 and dateline-crossing boxes such as 150 to 210 are valid), and latitudes within [-90, 90] with lat_max greater than lat_min. The detailed reason for a failure is available from _spatial_extent_error, which revalidate uses to raise a descriptive ValueError.

        Parameters:
            None

        Returns:
            bool: True if spatial extent parameters are valid or unspecified, False if invalid bounds detected.
        """
        return self._spatial_extent_error() is None

    def _validate_choices(self: "MPASConfig") -> None:
        """
        This method validates the categorical configuration parameters against the choices supported by the command-line interface: output formats, accumulation period, plot type (restricted per analysis type for precipitation and surface runs), wind plot type, vertical coordinate, cross-section plot style, and colorbar extension. Each unknown value raises a ValueError that names the offending value and lists the supported options, so a typo in a YAML file is reported before any data is loaded instead of failing deep inside a plotter.

        Parameters:
            None

        Returns:
            None
        """

        def _require(name: str, value: Any, allowed: Tuple[str, ...]) -> None:
            """
            This helper function raises a ValueError naming the parameter, the invalid value, and the supported options when the value is not one of the allowed choices.

            Parameters:
                name (str): The name of the parameter being checked (for error messages).
                value (Any): The value of the parameter to check.
                allowed (Tuple[str, ...]): The supported values for the parameter.

            Returns:
                None
            """
            if value not in allowed:
                raise ValueError(
                    f"Invalid {name} '{value}'. Must be one of: {', '.join(allowed)}"
                )

        for fmt in self.output_formats or []:
            _require("output format", fmt, SUPPORTED_OUTPUT_FORMATS)

        _require(
            "accumulation_period",
            self.accumulation_period,
            SUPPORTED_ACCUMULATION_PERIODS,
        )

        if self.analysis_type in _PRECIPITATION_ANALYSIS_TYPES:
            _require("plot_type", self.plot_type, PRECIPITATION_PLOT_TYPES)
        else:
            _require("plot_type", self.plot_type, SURFACE_PLOT_TYPES)

        _require("precip_units", self.precip_units, PRECIPITATION_UNITS)
        _require("wind_plot_type", self.wind_plot_type, WIND_PLOT_TYPES)
        _require("vertical_coord", self.vertical_coord, VERTICAL_COORDS)
        _require("plot_style", self.plot_style, CROSS_PLOT_STYLES)
        _require("extend", self.extend, COLORBAR_EXTENDS)
        if self.interpolation_method is not None:
            _require(
                "interpolation_method", self.interpolation_method, INTERPOLATION_METHODS
            )

    def _validate_time_selection(self: "MPASConfig") -> None:
        """
        This method validates the optional time range (time_start, time_end) used for batch processing. Both values must be non-negative integers when specified, and time_end must not be smaller than time_start because the range is inclusive. Whether the range lies within the loaded dataset can only be checked once the data are open, so that check is performed by the command-line interface at run time.

        Parameters:
            None

        Returns:
            None
        """
        for name in ("time_start", "time_end"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"{name} must be a non-negative integer, got {value!r}"
                )

        if (
            self.time_start is not None
            and self.time_end is not None
            and self.time_end < self.time_start
        ):
            raise ValueError(
                f"Invalid time range: time_end ({self.time_end}) is before "
                f"time_start ({self.time_start})"
            )

    def _validate_remap_settings(self: "MPASConfig") -> None:
        """
        This method validates the remapping settings specified in the configuration. It checks if the remap_engine parameter is set to a known value (either "kdtree" or "esmf") and raises a ValueError if it is not. If the remap_engine is "esmf" and the remap_method is set to "nearest", it automatically changes the remap_method to "nearest_s2d" since "nearest" is not a valid method for ESMF. Then, it checks if the remap_method is valid for the specified remap_engine by comparing it against a predefined set of allowed methods for each engine. If the remap_method is not valid for the chosen remap_engine, it raises a ValueError with an appropriate message indicating the allowed methods. This validation ensures that the remapping settings are consistent and compatible with the underlying remapping libraries, preventing errors during the remapping process.

        Parameters:
            None

        Returns:
            None
        """
        _KDTREE_METHODS = {"nearest", "linear"}

        try:
            from .remapping import MPASRemapper

            _ESMF_METHODS = set(MPASRemapper._METHOD_MAP.keys())
        except Exception:
            _ESMF_METHODS = {
                "bilinear",
                "conservative",
                "conservative_normed",
                "patch",
                "nearest_s2d",
                "nearest_d2s",
            }
        _KNOWN_ENGINES = {"kdtree", "esmf"}

        if self.remap_engine not in _KNOWN_ENGINES:
            raise ValueError(
                f"Invalid remap_engine '{self.remap_engine}'. "
                f"Must be one of: {sorted(_KNOWN_ENGINES)}"
            )

        if self.remap_engine == "esmf" and self.remap_method == "nearest":
            self.remap_method = "bilinear"

        allowed = _KDTREE_METHODS if self.remap_engine == "kdtree" else _ESMF_METHODS

        if self.remap_method not in allowed:
            raise ValueError(
                f"Invalid remap_method '{self.remap_method}' for "
                f"remap_engine='{self.remap_engine}'. "
                f"Allowed methods: {sorted(allowed)}"
            )

    def to_dict(self: "MPASConfig") -> Dict[str, Any]:
        """
        This method converts the MPASConfig dataclass instance into a dictionary format. It uses the asdict function from the dataclasses module to create a dictionary representation of the dataclass. Additionally, it iterates through the dictionary items and checks if any values are tuples (such as figure_size) and converts them to lists, since YAML serialization does not support tuples. This method ensures that all configuration parameters are in a format suitable for saving to a YAML file or for other uses where a dictionary representation is needed.

        Parameters:
            None

        Returns:
            Dict[str, Any]: Dictionary containing all configuration parameters with tuple values converted to lists.
        """
        config_dict = asdict(self)
        for key, value in config_dict.items():
            if isinstance(value, tuple):
                config_dict[key] = list(value)
        return config_dict

    @staticmethod
    def _apply_aliases(config_dict: Dict[str, Any], known: set) -> Dict[str, Any]:
        """
        This helper renames YAML keys that use the command-line flag spelling (for example 'formats', 'accumulation', 'subsample') to the MPASConfig field they set, expands figure_width/figure_height into figure_size, and expands time_range into time_start/time_end. A canonical key present in the same mapping always wins over its alias.

        Parameters:
            config_dict (Dict[str, Any]): Raw configuration mapping.
            known (set): Names of the declared MPASConfig fields.

        Returns:
            Dict[str, Any]: A new mapping with aliases resolved.
        """
        result = dict(config_dict)
        for alias, field_name in CONFIG_KEY_ALIASES.items():
            if alias in result and alias not in known:
                value = result.pop(alias)
                result.setdefault(field_name, value)
        time_range = result.pop("time_range", None)
        if time_range is not None:
            if not isinstance(time_range, (list, tuple)) or len(time_range) != 2:
                raise ValueError(f"time_range must be [start, end], got {time_range!r}")
            result.setdefault("time_start", time_range[0])
            result.setdefault("time_end", time_range[1])
        if "figure_size" not in result and (
            "figure_width" in result or "figure_height" in result
        ):
            default_w, default_h = 12.0, 10.0
            result["figure_size"] = (
                result.get("figure_width", default_w),
                result.get("figure_height", default_h),
            )
        return result

    @classmethod
    def from_dict(cls: type["MPASConfig"], config_dict: Dict[str, Any]) -> "MPASConfig":
        """
        This class method creates an instance of MPASConfig from a given dictionary of configuration parameters. It takes a dictionary as input, which should contain keys that match the attribute names of the MPASConfig dataclass. The method checks if the 'figure_size' key is present in the dictionary and if its value is a list; if so, it converts it to a tuple since the MPASConfig class expects figure_size to be a tuple. Finally, it uses the unpacking operator (**) to pass the dictionary items as keyword arguments to the MPASConfig constructor, which will initialize the instance with the provided configuration values. This method allows for easy creation of configuration objects from dictionaries, such as those loaded from YAML files or other sources.

        Parameters:
            config_dict (Dict[str, Any]): Dictionary containing configuration parameters with keys matching MPASConfig attributes.

        Returns:
            MPASConfig: An instance of MPASConfig initialized with the provided configuration parameters.
        """
        if not isinstance(config_dict, dict):
            raise ValueError(
                f"Configuration must be a mapping of parameters, got "
                f"{type(config_dict).__name__}"
            )

        # Filter to declared dataclass fields so an untrusted YAML with unknown or
        # extra keys is a warning, not a TypeError-level crash (audit finding F12).
        known = {f.name for f in fields(cls)}
        config_dict = cls._apply_aliases(config_dict, known)
        unknown = set(config_dict) - known
        if unknown:
            logger.warning(
                "Ignoring unknown configuration keys: %s", ", ".join(sorted(unknown))
            )
        filtered = {k: v for k, v in config_dict.items() if k in known}

        if "figure_size" in filtered and isinstance(filtered["figure_size"], list):
            filtered["figure_size"] = tuple(filtered["figure_size"])
        return cls(**filtered)

    def save_to_file(
        self: "MPASConfig", filepath: str, base_dir: Optional[str] = None
    ) -> None:
        """
        This method saves the current configuration parameters of the MPASConfig instance to a YAML file at the specified filepath. It first converts the configuration to a dictionary format using the to_dict method, which also handles any necessary conversions (such as tuples to lists). Then, it opens the specified file in write mode and uses the yaml.dump function to write the configuration dictionary to the file in a human-readable format with proper indentation. After successfully saving the configuration, it prints a confirmation message indicating where the configuration has been saved. This method allows users to easily export their current configuration settings for later use or sharing with others.

        Parameters:
            filepath (str): Path to YAML file where configuration should be saved. May be absolute or relative, but must resolve inside base_dir.
            base_dir (Optional[str]): Trusted directory the path must stay within. Defaults to the current working directory.

        Returns:
            None
        """
        safe_path = _validate_config_path(filepath, base_dir, must_exist=False)
        config_dict = self.to_dict()

        with open(safe_path, "w") as yaml_file:
            yaml.dump(config_dict, yaml_file, default_flow_style=False, indent=2)

        logger.info("Configuration saved to: %s", safe_path)

    @classmethod
    def load_from_file(
        cls: type["MPASConfig"], filepath: str, base_dir: Optional[str] = None
    ) -> "MPASConfig":
        """
        This class method loads configuration parameters from a specified YAML file and creates an instance of MPASConfig with those parameters. It opens the YAML file in read mode and uses the yaml.safe_load function to parse the contents of the file into a dictionary. Then, it calls the from_dict class method to convert the loaded dictionary into an MPASConfig instance, which will be initialized with the parameters from the file. This method allows users to easily load previously saved configurations or configurations shared by others, ensuring that all parameters are correctly set up for use in analysis and plotting operations.

        Parameters:
            filepath (str): Path to YAML file containing configuration parameters. May be absolute or relative, but must resolve inside base_dir.
            base_dir (Optional[str]): Trusted directory the path must stay within. Defaults to the current working directory.

        Returns:
            MPASConfig: An instance of MPASConfig initialized with parameters loaded from the specified YAML file.
        """
        return cls.from_dict(cls.load_dict_from_file(filepath, base_dir))

    @classmethod
    def load_dict_from_file(
        cls: type["MPASConfig"], filepath: str, base_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        This class method reads a YAML configuration file and returns its contents as a plain dictionary without constructing or validating an MPASConfig. The command-line interface uses it to layer explicitly passed flags on top of the file (explicit flags > YAML > defaults), which requires knowing exactly which keys the file set instead of seeing every dataclass default. The path is confined to the trusted base directory exactly as in load_from_file, and the parsed document must be a mapping (an empty file yields an empty dictionary).

        Parameters:
            filepath (str): Path to the YAML file. May be absolute or relative, but must resolve inside base_dir.
            base_dir (Optional[str]): Trusted directory the path must stay within. Defaults to the current working directory.

        Returns:
            Dict[str, Any]: The configuration keys and values defined in the file.
        """
        safe_path = _validate_config_path(filepath, base_dir, must_exist=True)

        with open(safe_path, "r") as yaml_file:
            config_dict = yaml.safe_load(yaml_file)

        if config_dict is None:
            return {}
        if not isinstance(config_dict, dict):
            raise ValueError(
                f"Configuration must be a mapping of parameters, got "
                f"{type(config_dict).__name__}"
            )
        return config_dict

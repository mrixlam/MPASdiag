#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag - MPAS Model Output Analysis and Visualization Package

A comprehensive Python package for analyzing and visualizing MPAS model output, providing professional-quality diagnostic tools for meteorological research. The full public API is available directly from the top-level package::

    import mpasdiag as md

    cfg = md.MPASConfig(...)
    bounds = md.GeographicBounds(...)
    plotter = md.MPASSurfacePlotter(...)

Symbols are resolved lazily (PEP 562): ``import mpasdiag`` itself stays cheap and free of heavy optional dependencies (matplotlib, cartopy, esmpy); those are only imported the first time a symbol that needs them is accessed. The sub-packages remain importable as ``md.processing``, ``md.visualization`` and ``md.diagnostics`` for code that prefers explicit namespaces.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

import importlib
from typing import TYPE_CHECKING, Any, List

__version__ = "1.0.0"
__author__ = "Rubaiat Islam"
__email__ = "mrislam@ucar.edu"
__institution__ = "Mesoscale & Microscale Meteorology Laboratory, NCAR"

# Define the public API of the package
_META = ("__version__", "__author__", "__email__", "__institution__")

# Real sub-packages, importable as ``md.processing`` etc.
_SUBPACKAGES = ("processing", "visualization", "diagnostics")

# Convenience aliases to nested modules (e.g. ``md.constants``).
_SUBMODULE_ALIASES = {
    "constants": "mpasdiag.processing.constants",
}

# Public symbols and their source modules.
_SOURCES = {
    "mpasdiag.processing": (
        "MPASBaseProcessor",
        "clear_grid_cache",
        "collective_grid_load",
        "MPAS2DProcessor",
        "MPAS3DProcessor",
        "MPASFileMetadata",
        "UnitConverter",
        "MPASGeographicUtils",
        "GeographicBounds",
        "MPASDateTimeUtils",
        "MPASConfig",
        "MPASLogger",
        "get_logger",
        "FileManager",
        "print_system_info",
        "DataValidator",
        "PerformanceMonitor",
        "ArgumentParser",
        "MPASDataCache",
        "CachedVariable",
        "MPASParallelManager",
        "TaskResult",
        "ParallelStats",
        "LoadBalanceStrategy",
        "ErrorPolicy",
        "MPASRemapper",
        "remap_mpas_to_latlon",
        "remap_mpas_to_latlon_with_masking",
        "build_remapped_valid_mask",
        "create_target_grid",
        "dispatch_remap",
    ),
    "mpasdiag.visualization": (
        "MPASVisualizer",
        "MPASVisualizationStyle",
        "WindPlotStyle",
        "TransectLineStyle",
        "MPASSurfacePlotter",
        "SurfaceMapStyle",
        "create_surface_plot",
        "MPASVerticalCrossSectionPlotter",
        "CrossSectionStyle",
        "MPASPrecipitationPlotter",
        "PrecipitationMapStyle",
        "PrecipitationRenderStyle",
        "OverlayColorSpec",
        "MPASWindPlotter",
        "MPASSkewTPlotter",
    ),
    "mpasdiag.diagnostics": (
        "PrecipitationDiagnostics",
        "WindDiagnostics",
        "SoundingDiagnostics",
        "MoistureTransportDiagnostics",
        "ThermodynamicDiagnostics",
    ),
    "mpasdiag.processing.cli_unified": (
        "MPASUnifiedCLI",
        "main",
    ),
    "mpasdiag.processing.parallel_wrappers": (
        "ParallelPrecipitationProcessor",
        "ParallelSurfaceProcessor",
        "ParallelWindProcessor",
        "ParallelCrossSectionProcessor",
        "ParallelSkewTProcessor",
        "SurfaceBatchStyle",
        "WindBatchStyle",
        "CrossSectionBatchStyle",
        "RemapConfig",
        "auto_batch_processor",
    ),
    "mpasdiag.processing.constants": (
        "GRAVITY",
        "HPA",
        "PA",
        "KELVIN",
        "NVERT_LEVELS_DIM",
    ),
    "mpasdiag.processing.remapping": ("ESMPY_AVAILABLE",),
}

_SYMBOL_SOURCES: "dict[str, str]" = {}

for _module, _names in _SOURCES.items():
    for _name in _names:
        _SYMBOL_SOURCES.setdefault(_name, _module)

__all__ = [
    *_META,
    *_SUBPACKAGES,
    *_SUBMODULE_ALIASES,
    *sorted(_SYMBOL_SOURCES),
]


def __getattr__(name: str) -> Any:
    """
    This function is called when an attribute lookup on the module fails. It implements lazy loading of sub-packages and symbols, as described in the module docstring.

    Parameters:
        name (str): The name of the attribute being accessed.

    Returns:
        Any: The value of the requested attribute, which may be a sub-package, a module alias, or a symbol from the public API.
    """
    if name in _SUBPACKAGES:
        value = importlib.import_module(f"{__name__}.{name}")
    elif name in _SUBMODULE_ALIASES:
        value = importlib.import_module(_SUBMODULE_ALIASES[name])
    elif name in _SYMBOL_SOURCES:
        source = importlib.import_module(_SYMBOL_SOURCES[name])
        value = getattr(source, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value  # cache so __getattr__ is not called again
    return value


def __dir__() -> List[str]:
    """
    This function is called when dir() is used on the module. It returns a sorted list of all public symbols, including sub-packages and module aliases, as defined in __all__. This ensures that IDEs and interactive environments can discover all available attributes of the module, even those that are lazily loaded.

    Parameters:
        None

    Returns:
        List[str]: A sorted list of public symbols available in the module.
    """
    return sorted(__all__)


if TYPE_CHECKING:  # help IDEs and static type checkers resolve ``md.<symbol>``
    from mpasdiag.processing import (
        ArgumentParser,
        build_remapped_valid_mask,
        CachedVariable,
        clear_grid_cache,
        collective_grid_load,
        create_target_grid,
        DataValidator,
        dispatch_remap,
        ErrorPolicy,
        FileManager,
        GeographicBounds,
        get_logger,
        LoadBalanceStrategy,
        MPAS2DProcessor,
        MPAS3DProcessor,
        MPASBaseProcessor,
        MPASConfig,
        MPASDataCache,
        MPASDateTimeUtils,
        MPASFileMetadata,
        MPASGeographicUtils,
        MPASLogger,
        MPASParallelManager,
        MPASRemapper,
        ParallelStats,
        PerformanceMonitor,
        print_system_info,
        remap_mpas_to_latlon,
        remap_mpas_to_latlon_with_masking,
        TaskResult,
        UnitConverter,
    )
    from mpasdiag.visualization import (
        CrossSectionStyle,
        create_surface_plot,
        MPASPrecipitationPlotter,
        MPASSkewTPlotter,
        MPASSurfacePlotter,
        MPASVerticalCrossSectionPlotter,
        MPASVisualizationStyle,
        MPASVisualizer,
        MPASWindPlotter,
        OverlayColorSpec,
        PrecipitationMapStyle,
        PrecipitationRenderStyle,
        SurfaceMapStyle,
        TransectLineStyle,
        WindPlotStyle,
    )
    from mpasdiag.diagnostics import (
        MoistureTransportDiagnostics,
        PrecipitationDiagnostics,
        SoundingDiagnostics,
        ThermodynamicDiagnostics,
        WindDiagnostics,
    )
    from mpasdiag.processing.cli_unified import main, MPASUnifiedCLI
    from mpasdiag.processing.parallel_wrappers import (
        auto_batch_processor,
        CrossSectionBatchStyle,
        ParallelCrossSectionProcessor,
        ParallelPrecipitationProcessor,
        ParallelSkewTProcessor,
        ParallelSurfaceProcessor,
        ParallelWindProcessor,
        RemapConfig,
        SurfaceBatchStyle,
        WindBatchStyle,
    )
    from mpasdiag.processing.constants import (
        GRAVITY,
        HPA,
        KELVIN,
        NVERT_LEVELS_DIM,
        PA,
    )
    from mpasdiag.processing.remapping import ESMPY_AVAILABLE

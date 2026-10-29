#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Core Diagnostics Module: Vertical Helpers

This private module holds the vertical-selection helpers shared by the diagnostics classes. Its main entry point interpolates a 3D MPAS field to a single pressure surface independently in every column, reusing the same vectorized log-pressure kernel that MPAS3DProcessor.get_3d_variable_data uses so that diagnostics and the processing layer return identical values for a requested pressure level.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: March 2026
Version: 1.0.0
"""

import numpy as np
import xarray as xr
from typing import Any, Optional, cast

from mpasdiag.processing.processors_3d import MPAS3DProcessor

MIN_PLAUSIBLE_PRESSURE_PA = 1100.0


def plain_dataset(dataset: Any) -> xr.Dataset:
    """
    This helper returns a plain xarray Dataset for datasets that subclass xr.Dataset (for example uxarray datasets), rebuilding it from its data variables, coordinates and attributes so that ordinary positional selection works, and returns genuine xarray Datasets unchanged.

    Parameters:
        dataset (Any): An xarray Dataset or a subclass of it.

    Returns:
        xr.Dataset: A plain xarray Dataset.
    """
    if type(dataset) is not xr.Dataset and isinstance(dataset, xr.Dataset):
        return xr.Dataset(
            dict(dataset.data_vars), coords=dataset.coords, attrs=dataset.attrs
        )
    return cast(xr.Dataset, dataset)


def check_pressure_level(level: float, var_name: str) -> None:
    """
    This helper validates a pressure-level request in Pascals and raises a ValueError for values that are not plausible pressures (below 1100 Pa), which are almost always hPa values passed by mistake and would silently select a layer near the model top.

    Parameters:
        level (float): Requested pressure in Pa.
        var_name (str): Variable name used in the error message.

    Returns:
        None
    """
    if not np.isfinite(level) or float(level) < MIN_PLAUSIBLE_PRESSURE_PA:
        raise ValueError(
            f"Float level {level} is interpreted as a pressure in Pa, but "
            f"{level} Pa is near the top of the atmosphere; for '{var_name}' "
            f"pass an int for a model level index or the pressure in Pa "
            f"(e.g. 85000.0 for 850 hPa)."
        )


def interpolate_to_pressure(
    dataset: Any,
    var_name: str,
    level: float,
    time_dim: str,
    time_index: int,
    field: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    This function interpolates a 3D field to a pressure surface separately in every column, linearly in log(pressure) between the two model levels that bracket the requested pressure in that column, using the total pressure (pressure_p + pressure_base) of that column and the shared kernel MPAS3DProcessor._interpolate_columns_log_p. Columns whose pressure range does not contain the requested pressure (below ground or above the model top) are NaN rather than clamped to the nearest model level. Fields on interface levels (nVertLevelsP1) are averaged to the mass levels first, the standard MPAS destaggering. The result carries the variable attributes plus 'selected_level', 'level_index' (-1 because no single model level is selected), 'actual_pressure_level' and 'vertical_interpolation'. The earlier behaviour of picking the model level whose domain-mean pressure was nearest and applying it to every column, which could be wrong by tens of hPa at individual columns, is intentionally gone.

    Parameters:
        dataset (Any): Dataset holding the field plus 'pressure_p' and 'pressure_base'.
        var_name (str): Name of the field to interpolate.
        level (float): Target pressure in Pa.
        time_dim (str): Name of the time dimension.
        time_index (int): Validated time index.
        field (Optional[xr.DataArray]): Pre-selected field at this time (with its vertical dimension); read from dataset when None.

    Returns:
        xr.DataArray: The field on the pressure surface with the vertical dimension removed.
    """
    check_pressure_level(level, var_name)

    if "pressure_p" not in dataset or "pressure_base" not in dataset:
        raise ValueError(
            "Cannot find pressure level - pressure data not available "
            "('pressure_p' and 'pressure_base' are required)."
        )

    ds = plain_dataset(dataset)
    total_pressure = (
        ds["pressure_p"].isel({time_dim: time_index})
        + ds["pressure_base"].isel({time_dim: time_index})
    ).compute()

    if field is None:
        field = ds[var_name].isel({time_dim: time_index})
    field = field.compute() if hasattr(field, "compute") else field

    vertical_dim = "nVertLevelsP1" if "nVertLevelsP1" in field.dims else "nVertLevels"
    horizontal_dims = [d for d in field.dims if d != vertical_dim]

    pressure_vdim = (
        "nVertLevels" if "nVertLevels" in total_pressure.dims else "nVertLevelsP1"
    )
    if sorted(horizontal_dims) != sorted(
        d for d in total_pressure.dims if d != pressure_vdim
    ):
        raise ValueError(
            f"Cannot interpolate '{var_name}' to pressure: its horizontal dimensions "
            f"{horizontal_dims} differ from those of the pressure field"
        )

    values = field.transpose(*horizontal_dims, vertical_dim).values
    if vertical_dim == "nVertLevelsP1" and pressure_vdim == "nVertLevels":
        values = 0.5 * (values[..., :-1] + values[..., 1:])

    pressure = total_pressure.transpose(*horizontal_dims, pressure_vdim).values
    n_levels = pressure.shape[-1]

    result = MPAS3DProcessor._interpolate_columns_log_p(
        values.reshape(-1, n_levels), pressure.reshape(-1, n_levels), float(level)
    ).reshape(values.shape[:-1])

    if np.issubdtype(field.dtype, np.floating):
        result = result.astype(field.dtype)

    out = field.isel({vertical_dim: 0}, drop=True).copy(data=result)
    out.attrs = dict(ds[var_name].attrs) if var_name in ds else dict(field.attrs)
    out.attrs["selected_level"] = level
    out.attrs["level_index"] = -1
    out.attrs["actual_pressure_level"] = f"{level:.1f} Pa"
    out.attrs["vertical_interpolation"] = "linear in log(pressure), per column"
    return out


def as_dataarray(values: Any) -> xr.DataArray:
    """
    This helper returns the input as an xarray DataArray: a DataArray (numpy or dask backed) is returned unchanged, anything array-like is wrapped without dimension names so that the diagnostics can treat every input the same way.

    Parameters:
        values (Any): A DataArray or an array-like of numbers.

    Returns:
        xr.DataArray: The input as a DataArray.
    """
    if isinstance(values, xr.DataArray):
        return values
    arr = np.asarray(values)
    return xr.DataArray(arr, dims=[f"dim_{i}" for i in range(arr.ndim)])


def is_lazy(values: xr.DataArray) -> bool:
    """
    This helper reports whether a DataArray is backed by a dask array, so that verbose statistics are not allowed to force a computation of a lazy result.

    Parameters:
        values (xr.DataArray): DataArray to inspect.

    Returns:
        bool: True if the DataArray holds a chunked (dask) array.
    """
    return values.chunks is not None

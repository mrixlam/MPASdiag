#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Test Suite: Thermodynamic Diagnostics

This module contains unit tests for the ThermodynamicDiagnostics class in the MPASdiag package. The tests cover the initialization of the diagnostics class, the computation of density potential temperature from both direct inputs and from datasets, and the handling of various edge cases such as missing variables and different vertical dimension naming conventions. The tests also verify that the verbose mode correctly prints input and result summaries when enabled, and that it does not print anything when disabled.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: March 2026
Version: 1.0.0
"""

import matplotlib

matplotlib.use("Agg")
import math
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mpasdiag.diagnostics.thermodynamic import (
    ThermodynamicDiagnostics,
    RV_OVER_RD,
    CONDENSATE_VARS,
    THETA_RHO_UNITS,
)


def _make_3d_dataset(
    n_time: int = 2,
    n_cells: int = 6,
    n_vert: int = 4,
    include_condensates: bool = True,
    include_pressure: bool = True,
    vert_name: str = "nVertLevels",
) -> xr.Dataset:
    """
    This helper function creates a synthetic 3D dataset with dimensions (Time, nCells, nVertLevels) containing 'theta' and 'qv' variables, along with optional condensate and pressure variables. The values are generated randomly within typical atmospheric ranges. This dataset is designed to test the functionality of the thermodynamic diagnostics when provided with a full 3D dataset that includes a vertical dimension.

    Parameters:
        n_time (int): Number of time steps (default: 2)
        n_cells (int): Number of horizontal cells (default: 6)
        n_vert (int): Number of vertical levels (default: 4)
        include_condensates (bool): Whether to include condensate variables (default: True)
        include_pressure (bool): Whether to include pressure variables (default: True)
        vert_name (str): Name of the vertical dimension (default: 'nVertLevels')

    Returns:
        xr.Dataset: A synthetic dataset with the specified variables and dimensions.
    """
    times = pd.date_range("2024-01-01", periods=n_time, freq="h")
    rng = np.random.default_rng(42)

    theta_vals = 300.0 + rng.uniform(-5.0, 5.0, size=(n_time, n_cells, n_vert))
    qv_vals = rng.uniform(0.001, 0.018, size=(n_time, n_cells, n_vert))

    data_vars = {
        "theta": xr.DataArray(
            theta_vals,
            dims=["Time", "nCells", vert_name],
            coords={"Time": times},
        ),
        "qv": xr.DataArray(
            qv_vals,
            dims=["Time", "nCells", vert_name],
            coords={"Time": times},
        ),
    }

    if include_condensates:
        for cv in CONDENSATE_VARS:
            data_vars[cv] = xr.DataArray(
                rng.uniform(0.0, 1e-4, size=(n_time, n_cells, n_vert)),
                dims=["Time", "nCells", vert_name],
                coords={"Time": times},
            )

    if include_pressure:
        base_profile = np.linspace(100000.0, 50000.0, n_vert)
        pb = np.tile(base_profile, (n_time, n_cells, 1))
        pp = np.full((n_time, n_cells, n_vert), 100.0)
        data_vars["pressure_base"] = xr.DataArray(
            pb,
            dims=["Time", "nCells", vert_name],
        )
        data_vars["pressure_p"] = xr.DataArray(
            pp,
            dims=["Time", "nCells", vert_name],
        )

    return xr.Dataset(data_vars)


def _make_2d_dataset() -> xr.Dataset:
    """
    This helper function creates a synthetic 2D dataset with dimensions (Time, nCells) containing 'theta' and 'qv' variables. The values are generated as constant arrays for simplicity. This dataset is designed to test the functionality of the thermodynamic diagnostics when provided with a 2D dataset that lacks a vertical dimension, ensuring that the diagnostics can still be computed correctly in this case.

    Parameters:
        None

    Returns:
        xr.Dataset: A synthetic 2D dataset with 'theta' and 'qv' variables.
    """
    times = pd.date_range("2024-01-01", periods=2, freq="h")
    n_cells = 6

    return xr.Dataset(
        {
            "theta": xr.DataArray(
                np.full((2, n_cells), 300.0),
                dims=["Time", "nCells"],
                coords={"Time": times},
            ),
            "qv": xr.DataArray(
                np.full((2, n_cells), 0.01),
                dims=["Time", "nCells"],
                coords={"Time": times},
            ),
        }
    )


class _UxLikeDataset(xr.Dataset):
    """A simple subclass of xr.Dataset to test handling of non-plain Dataset types in the diagnostics code."""

    __slots__ = ()


@pytest.fixture
def diag_silent() -> ThermodynamicDiagnostics:
    """
    This fixture provides a ThermodynamicDiagnostics instance with verbose=False for use in tests that require silent operation. It allows testing of the diagnostics functionality without print statements, ensuring that the diagnostics can be computed without verbose output when desired.

    Parameters:
        None

    Returns:
        ThermodynamicDiagnostics: An instance of the diagnostics class with verbose output disabled.
    """
    return ThermodynamicDiagnostics(verbose=False)


@pytest.fixture
def diag_verbose() -> ThermodynamicDiagnostics:
    """
    This fixture provides a ThermodynamicDiagnostics instance with verbose=True for use in tests that require verbose output. It allows testing of the diagnostics functionality with print statements enabled, ensuring that the diagnostics can provide detailed input and result summaries when desired.

    Parameters:
        None

    Returns:
        ThermodynamicDiagnostics: An instance of the diagnostics class with verbose output enabled.
    """
    return ThermodynamicDiagnostics(verbose=True)


@pytest.fixture
def ds_3d() -> xr.Dataset:
    """
    This fixture provides a synthetic 3D dataset for use in tests that require three-dimensional data. It allows testing of the diagnostics functionality with a dataset that includes a vertical dimension, enabling validation of the diagnostics methods that operate on 3D data.

    Parameters:
        None

    Returns:
        xr.Dataset: A synthetic 3D dataset with 'theta' and 'qv' variables.
    """
    return _make_3d_dataset()


@pytest.fixture
def ds_2d() -> xr.Dataset:
    """
    This fixture provides a synthetic 2D dataset for use in tests that require two-dimensional data. It allows testing of the diagnostics functionality with a dataset that lacks a vertical dimension, enabling validation of the diagnostics methods that operate on 2D data.

    Parameters:
        None

    Returns:
        xr.Dataset: A synthetic 2D dataset with 'theta' and 'qv' variables.
    """
    return _make_2d_dataset()


class TestInit:
    """Test ThermodynamicDiagnostics initialization."""

    def test_default_verbose_true(self: "TestInit") -> None:
        """
        This test verifies that the default value of the verbose parameter in ThermodynamicDiagnostics is True. It creates an instance of the diagnostics class without specifying the verbose argument and asserts that the verbose attribute is set to True, ensuring that the diagnostics will provide detailed output by default.

        Parameters:
            None

        Returns:
            None
        """
        diag = ThermodynamicDiagnostics()
        assert diag.verbose is True

    def test_verbose_false(self: "TestInit") -> None:
        """
        This test verifies that the verbose parameter in ThermodynamicDiagnostics can be set to False during initialization. It creates an instance of the diagnostics class with verbose=False and asserts that the verbose attribute is set to False, ensuring that the diagnostics can operate in silent mode when desired.

        Parameters:
            None

        Returns:
            None
        """
        diag = ThermodynamicDiagnostics(verbose=False)
        assert diag.verbose is False


class TestComputeDensityPotentialTemperature:
    """Test compute_density_potential_temperature direct API."""

    def test_raises_when_theta_none(
        self: "TestComputeDensityPotentialTemperature",
        diag_silent: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature method raises a ValueError when the theta argument is None. It provides a valid qv DataArray and asserts that the error message contains the word "theta", ensuring that the method correctly identifies missing required input.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.

        Returns:
            None
        """
        qv = xr.DataArray(np.array([0.01]), dims=["x"])
        with pytest.raises(ValueError, match="theta"):
            diag_silent.compute_density_potential_temperature(theta=None, qv=qv)

    def test_raises_when_qv_none(
        self: "TestComputeDensityPotentialTemperature",
        diag_silent: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature method raises a ValueError when the qv argument is None. It provides a valid theta DataArray and asserts that the error message contains the word "qv", ensuring that the method correctly identifies missing required input.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.

        Returns:
            None
        """
        theta = xr.DataArray(np.array([300.0]), dims=["x"])
        with pytest.raises(ValueError, match="qv"):
            diag_silent.compute_density_potential_temperature(theta=theta, qv=None)

    def test_computes_correct_values_without_condensates(
        self: "TestComputeDensityPotentialTemperature",
        diag_silent: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature method computes correct values based on the formula θ_ρ = θ * (1 + (Rv/Rd - 1) * qv) when no condensate variables are provided. It provides valid theta and qv DataArrays, computes the density potential temperature, and asserts that the computed values match the expected values calculated from the formula. It also checks that the resulting DataArray has the correct units and standard_name attributes, and that the reference attribute includes "Emanuel", confirming that the method correctly computes density potential temperature without condensates.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.

        Returns:
            None
        """
        theta = xr.DataArray(np.array([300.0, 280.0]), dims=["x"])
        qv = xr.DataArray(np.array([0.01, 0.005]), dims=["x"])

        result = diag_silent.compute_density_potential_temperature(theta, qv)

        # qt == qv → theta_rho = theta * (1 + (Rv/Rd - 1) * qv)
        expected = theta.values * (1.0 + RV_OVER_RD * qv.values - qv.values)
        assert np.allclose(result.values, expected)
        assert result.attrs["units"] == THETA_RHO_UNITS
        assert result.attrs["standard_name"] == "density_potential_temperature"
        assert "Emanuel" in result.attrs["reference"]

    def test_computes_with_explicit_qt(
        self: "TestComputeDensityPotentialTemperature",
        diag_silent: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature method computes correct values when an explicit total water mixing ratio (qt) is provided. It provides valid theta, qv, and qt DataArrays, computes the density potential temperature, and asserts that the computed values match the expected formula that accounts for the total water mixing ratio. This test ensures that the method correctly uses the provided qt value instead of calculating it from qv and condensates when qt is explicitly given.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.

        Returns:
            None
        """
        theta = xr.DataArray(np.array([300.0]), dims=["x"])
        qv = xr.DataArray(np.array([0.01]), dims=["x"])
        qt = xr.DataArray(np.array([0.012]), dims=["x"])  # qv + condensate

        result = diag_silent.compute_density_potential_temperature(theta, qv, qt=qt)
        expected = 300.0 * (1.0 + RV_OVER_RD * 0.01 - 0.012)
        assert np.allclose(result.values, expected)

    def test_computes_with_individual_condensates(
        self: "TestComputeDensityPotentialTemperature",
        diag_silent: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature method computes correct values when individual condensate variables are provided. It provides valid theta, qv, and individual condensate DataArrays and asserts that the computed values match the expected formula that accounts for the total water mixing ratio calculated from the sum of qv and all condensates. This test ensures that the method correctly sums the individual condensate variables to compute qt when qt is not explicitly provided, and uses this qt value in the density potential temperature calculation.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.

        Returns:
            None
        """
        theta = xr.DataArray(np.array([300.0]), dims=["x"])
        qv = xr.DataArray(np.array([0.01]), dims=["x"])
        qc = xr.DataArray(np.array([0.0005]), dims=["x"])
        qr = xr.DataArray(np.array([0.0001]), dims=["x"])
        qi = xr.DataArray(np.array([0.00005]), dims=["x"])
        qs = xr.DataArray(np.array([0.00002]), dims=["x"])
        qg = xr.DataArray(np.array([0.00001]), dims=["x"])

        result = diag_silent.compute_density_potential_temperature(
            theta,
            qv,
            qc=qc,
            qr=qr,
            qi=qi,
            qs=qs,
            qg=qg,
        )
        qt_expected = 0.01 + 0.0005 + 0.0001 + 0.00005 + 0.00002 + 0.00001
        expected = 300.0 * (1.0 + RV_OVER_RD * 0.01 - qt_expected)
        assert np.allclose(result.values, expected)

    def test_verbose_runs_print_helpers_2d(
        self: "TestComputeDensityPotentialTemperature",
        diag_verbose: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature method runs correctly in verbose mode with 2D input. It exercises the "no vertical dim" branch in the print helpers by providing theta and qv DataArrays with only 'nCells' dimension, and asserts that the method computes the density potential temperature without errors and returns a result with the expected shape. This test ensures that the verbose mode correctly handles 2D input and prints the appropriate diagnostic information.

        Parameters:
            diag_verbose (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=True.

        Returns:
            None
        """
        # 2D input exercises the "no vertical dim" branch in print helpers.
        theta = xr.DataArray(np.full((3,), 300.0), dims=["nCells"])
        qv = xr.DataArray(np.full((3,), 0.01), dims=["nCells"])
        result = diag_verbose.compute_density_potential_temperature(theta, qv)
        assert result.shape == (3,)

    def test_verbose_runs_print_helpers_3d(
        self: "TestComputeDensityPotentialTemperature",
        diag_verbose: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature method runs correctly in verbose mode with 3D input. It exercises the "has vertical dim" branch in the print helpers by providing theta and qv DataArrays with 'nCells' and 'nVertLevels' dimensions, and asserts that the method computes the density potential temperature without errors and returns a result with the expected shape. This test ensures that the verbose mode correctly handles 3D input and prints the appropriate diagnostic information, including summaries of the vertical dimension.

        Parameters:
            diag_verbose (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=True.

        Returns:
            None
        """
        # 3D input exercises the "has vertical dim" branch in print helpers.
        theta = xr.DataArray(
            np.full((4, 3), 300.0),
            dims=["nCells", "nVertLevels"],
        )
        qv = xr.DataArray(
            np.full((4, 3), 0.01),
            dims=["nCells", "nVertLevels"],
        )
        result = diag_verbose.compute_density_potential_temperature(theta, qv)
        assert result.shape == (4, 3)


class TestComputeFromDataset:
    """Test compute_density_potential_temperature_from_dataset."""

    def test_raises_when_dataset_none(
        self: "TestComputeFromDataset",
        diag_silent: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature_from_dataset method raises a ValueError when the dataset argument is None. It asserts that the error message contains the phrase "Dataset not provided", ensuring that the method correctly identifies the missing dataset input and provides an informative error message.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.

        Returns:
            None
        """
        with pytest.raises(ValueError, match="Dataset not provided"):
            diag_silent.compute_density_potential_temperature_from_dataset(
                dataset=None,
            )

    def test_full_3d_without_level(
        self: "TestComputeFromDataset",
        diag_silent: ThermodynamicDiagnostics,
        ds_3d: xr.Dataset,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature_from_dataset method computes correct values when provided with a full 3D dataset without specifying a vertical level. It asserts that the resulting DataArray has both 'nVertLevels' and 'nCells' dimensions, confirming that the method correctly processes the 3D dataset and returns a result that retains the vertical dimension when no specific level is requested. This test ensures that the method correctly handles 3D datasets and maintains the vertical structure when no level selection is made.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.
            ds_3d (xr.Dataset): A synthetic 3D dataset containing the necessary variables for computation.

        Returns:
            None
        """
        result = diag_silent.compute_density_potential_temperature_from_dataset(
            ds_3d,
            time_index=0,
        )
        assert "nVertLevels" in result.dims
        assert "nCells" in result.dims

    def test_with_int_level(
        self: "TestComputeFromDataset",
        diag_silent: ThermodynamicDiagnostics,
        ds_3d: xr.Dataset,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature_from_dataset method computes correct values when a specific integer vertical level index is provided. It asserts that the resulting DataArray does not have the 'nVertLevels' dimension, confirming that the method correctly selects the specified vertical level from the dataset and returns a 2D result with only the 'nCells' dimension. This test ensures that the method correctly handles level selection by index and returns the expected dimensionality in the output.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.
            ds_3d (xr.Dataset): A synthetic 3D dataset containing the necessary variables for computation.

        Returns:
            None
        """
        result = diag_silent.compute_density_potential_temperature_from_dataset(
            ds_3d,
            time_index=0,
            level=1,
        )
        assert "nVertLevels" not in result.dims

    def test_with_surface_level(
        self: "TestComputeFromDataset",
        diag_silent: ThermodynamicDiagnostics,
        ds_3d: xr.Dataset,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature_from_dataset method computes correct values when the 'surface' level is specified. It asserts that the resulting DataArray does not have the 'nVertLevels' dimension, confirming that the method correctly identifies and selects the surface level from the dataset, returning a 2D result with only the 'nCells' dimension. This test ensures that the method correctly handles level selection by name and returns the expected dimensionality in the output when 'surface' is specified.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.
            ds_3d (xr.Dataset): A synthetic 3D dataset containing the necessary variables for computation.

        Returns:
            None
        """
        result = diag_silent.compute_density_potential_temperature_from_dataset(
            ds_3d,
            time_index=0,
            level="surface",
        )
        assert "nVertLevels" not in result.dims

    def test_with_top_level(
        self: "TestComputeFromDataset",
        diag_silent: ThermodynamicDiagnostics,
        ds_3d: xr.Dataset,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature_from_dataset method computes correct values when the 'top' level is specified. It asserts that the resulting DataArray does not have the 'nVertLevels' dimension, confirming that the method correctly identifies and selects the top level from the dataset, returning a 2D result with only the 'nCells' dimension. This test ensures that the method correctly handles level selection by name and returns the expected dimensionality in the output when 'top' is specified.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.
            ds_3d (xr.Dataset): A synthetic 3D dataset containing the necessary variables for computation.

        Returns:
            None
        """
        result = diag_silent.compute_density_potential_temperature_from_dataset(
            ds_3d,
            time_index=0,
            level="top",
        )
        assert "nVertLevels" not in result.dims

    def test_with_pressure_level(
        self: "TestComputeFromDataset",
        diag_verbose: ThermodynamicDiagnostics,
        ds_3d: xr.Dataset,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature_from_dataset method computes correct values when a specific pressure level is provided. It asserts that the resulting DataArray does not have the 'nVertLevels' dimension, confirming that the method correctly resolves the specified pressure level to the corresponding vertical index and returns a 2D result with only the 'nCells' dimension. This test ensures that the method correctly handles level selection by pressure value and returns the expected dimensionality in the output.

        Parameters:
            diag_verbose (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=True.
            ds_3d (xr.Dataset): A synthetic 3D dataset containing the necessary variables for computation.

        Returns:
            None
        """
        result = diag_verbose.compute_density_potential_temperature_from_dataset(
            ds_3d,
            time_index=0,
            level=85000.0,
        )
        assert "nVertLevels" not in result.dims

    def test_missing_var_raises(
        self: "TestComputeFromDataset",
        diag_silent: ThermodynamicDiagnostics,
        ds_3d: xr.Dataset,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature_from_dataset method raises a ValueError when a required variable is missing from the dataset. It asserts that the error message contains the phrase "not found in dataset", ensuring that the method correctly identifies missing variables and provides an informative error message. This test ensures that the method has robust error handling for cases where the input dataset does not contain all necessary variables for computation.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.
            ds_3d (xr.Dataset): A synthetic 3D dataset containing the necessary variables for computation.

        Returns:
            None
        """
        with pytest.raises(ValueError, match="not found in dataset"):
            diag_silent.compute_density_potential_temperature_from_dataset(
                ds_3d,
                theta_var="nope",
            )

    def test_missing_condensates_treated_as_zero(
        self: "TestComputeFromDataset",
        diag_verbose: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature_from_dataset method correctly handles the case when condensate variables are missing from the dataset by treating them as zero. It asserts that the computed density potential temperature values match the expected values calculated from the formula that assumes no condensates (qt == qv), confirming that the method correctly defaults to zero for missing condensate variables and computes the density potential temperature accordingly. This test ensures that the method can still produce valid results even when condensate information is not available in the input dataset.

        Parameters:
            diag_verbose (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=True.

        Returns:
            None
        """
        ds = _make_3d_dataset(include_condensates=False, include_pressure=False)
        result = diag_verbose.compute_density_potential_temperature_from_dataset(
            ds,
            time_index=0,
        )
        # With no condensates, qt == qv → known closed form
        theta = ds["theta"].isel(Time=0).values
        qv = ds["qv"].isel(Time=0).values
        expected = theta * (1.0 + RV_OVER_RD * qv - qv)
        assert np.allclose(result.values, expected)

    def test_time_index_clamped_when_out_of_range(
        self: "TestComputeFromDataset",
        diag_silent: ThermodynamicDiagnostics,
        ds_3d: xr.Dataset,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature_from_dataset method correctly handles the case when the provided time_index is out of range by clamping it to the valid range of available time steps in the dataset. It asserts that the method does not raise an error and successfully computes the density potential temperature using the clamped time index, ensuring that the method has robust handling for out-of-range time indices and can still produce valid results. This test is important to confirm that the method can gracefully handle user input errors related to time indexing without crashing, and that it provides a reasonable fallback behavior by using the nearest valid time index.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.
            ds_3d (xr.Dataset): A synthetic 3D dataset containing the necessary variables for computation.

        Returns:
            None
        """
        # n_time=2 → time_index=99 is clamped to 1; should not raise.
        result = diag_silent.compute_density_potential_temperature_from_dataset(
            ds_3d,
            time_index=99,
        )
        assert result is not None

    def test_uxarray_data_type_branch(
        self: "TestComputeFromDataset",
        diag_silent: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature_from_dataset method correctly handles the case when the data_type argument is set to 'uxarray', which takes a different extraction branch in the code. It asserts that the method successfully extracts the necessary variables and computes the density potential temperature without raising an error, ensuring that the method can handle this specific data type and execute the appropriate code path for variable extraction. This test is important to confirm that the method's handling of different data types is functioning as intended and does not lead to errors when 'uxarray' is specified.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.

        Returns:
            None
        """
        # data_type='uxarray' takes a different extraction branch
        # (positional [time_index] indexing).
        ds = _make_3d_dataset(include_condensates=False, include_pressure=False)
        result = diag_silent.compute_density_potential_temperature_from_dataset(
            ds,
            time_index=0,
            data_type="uxarray",
        )
        assert result is not None

    def test_uxarray_subclass_dataset_unwrapped(
        self: "TestComputeFromDataset",
        diag_silent: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature_from_dataset method correctly handles the case when the input dataset is a subclass of xr.Dataset, which requires unwrapping to a plain Dataset for variable extraction. It asserts that the method successfully extracts the necessary variables and computes the density potential temperature without raising an error, ensuring that the method can handle subclassed datasets and correctly unwrap them to access the required data. This test is important to confirm that the method's handling of different dataset types is robust and does not lead to errors when encountering subclasses of xr.Dataset.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.

        Returns:
            None
        """
        # A subclass of xr.Dataset hits the "unwrap to plain Dataset" branch
        # of _extract_field (data_type='xarray') and _pressure_to_level_index.
        base = _make_3d_dataset(include_condensates=False)
        ux_ds = _UxLikeDataset(
            dict(base.data_vars),
            coords=base.coords,
            attrs=base.attrs,
        )
        result = diag_silent.compute_density_potential_temperature_from_dataset(
            ux_ds,
            time_index=0,
            level=85000.0,
        )
        assert result is not None

    def test_explicit_condensate_vars_list(
        self: "TestComputeFromDataset",
        diag_silent: ThermodynamicDiagnostics,
        ds_3d: xr.Dataset,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature_from_dataset method correctly handles the case when an explicit list of condensate variables is provided. It asserts that the method successfully extracts the specified condensate variables and computes the density potential temperature without raising an error. This test ensures that the method can correctly use a user-provided list of condensate variable names to compute the total water mixing ratio (qt) when qt is not explicitly given, and that it can compute the density potential temperature accordingly. This test is important to confirm that the method's handling of explicit condensate variable lists is functioning as intended and does not lead to errors when a user provides a custom list of condensate variables.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.
            ds_3d (xr.Dataset): A synthetic 3D dataset containing the necessary variables for computation.

        Returns:
            None
        """
        # Restricting condensate_vars exercises the explicit-list branch.
        result = diag_silent.compute_density_potential_temperature_from_dataset(
            ds_3d,
            time_index=0,
            condensate_vars=["qc", "qr"],
        )
        assert result is not None


class TestVirtualPotentialTemperature:
    """Test compute_virtual_potential_temperature."""

    def test_raises_when_theta_or_qv_none(
        self: "TestVirtualPotentialTemperature",
        diag_silent: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the compute_virtual_potential_temperature method raises a ValueError when either the theta or qv argument is None. It asserts that the error message contains the phrase "theta and qv", ensuring that the method correctly identifies missing required inputs and provides an informative error message. This test ensures that the method has robust error handling for cases where essential input variables are not provided, preventing silent failures and guiding users to provide the necessary data for computation.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.

        Returns:
            None
        """
        with pytest.raises(ValueError):
            diag_silent.compute_virtual_potential_temperature(theta=None, qv=None)

    def test_computes_correct_values(
        self: "TestVirtualPotentialTemperature",
        diag_silent: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the compute_virtual_potential_temperature method computes correct values based on the formula θ_v = θ * (1 + Rv/Rd * qv). It provides valid theta and qv DataArrays, computes the virtual potential temperature, and asserts that the computed values match the expected values calculated from the formula. It also checks that the resulting DataArray has the correct units and standard_name attributes, confirming that the method correctly computes virtual potential temperature with the appropriate metadata. This test ensures that the method's implementation of the virtual potential temperature formula is correct and that it returns results with the expected attributes.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.

        Returns:
            None
        """
        theta = xr.DataArray(np.array([300.0, 290.0]), dims=["x"])
        qv = xr.DataArray(np.array([0.01, 0.005]), dims=["x"])
        result = diag_silent.compute_virtual_potential_temperature(theta, qv)
        expected = theta.values * (1.0 + RV_OVER_RD * qv.values)
        assert np.allclose(result.values, expected)
        assert result.attrs["units"] == THETA_RHO_UNITS
        assert result.attrs["standard_name"] == "virtual_potential_temperature"

    def test_verbose_runs(
        self: "TestVirtualPotentialTemperature", diag_verbose: ThermodynamicDiagnostics
    ) -> None:
        """
        This test verifies that the compute_virtual_potential_temperature method runs correctly in verbose mode. It provides valid theta and qv DataArrays, computes the virtual potential temperature, and asserts that the method executes without errors and returns a result. This test ensures that the verbose mode of the method does not interfere with the computation and that it can successfully compute virtual potential temperature while providing detailed diagnostic output.

        Parameters:
            diag_verbose (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=True.

        Returns:
            None
        """
        theta = xr.DataArray(np.array([300.0]), dims=["x"])
        qv = xr.DataArray(np.array([0.01]), dims=["x"])
        result = diag_verbose.compute_virtual_potential_temperature(theta, qv)
        assert result is not None


class TestTotalWaterMixingRatio:
    """Test compute_total_water_mixing_ratio."""

    def test_only_qv_returns_copy(
        self: "TestTotalWaterMixingRatio",
        diag_silent: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the compute_total_water_mixing_ratio method returns a copy of the input qv DataArray when no condensate variables are provided. It provides a valid qv DataArray, computes the total water mixing ratio, and asserts that the computed values match the input qv values, confirming that the method correctly defaults to using qv as the total water mixing ratio when no condensates are given. It also checks that the resulting DataArray attributes indicate that only 'qv' is included in the total and that the units are correct, ensuring that the method returns appropriate metadata even in this simplified case.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.

        Returns:
            None
        """
        qv = xr.DataArray(np.array([0.01, 0.005]), dims=["x"])
        result = diag_silent.compute_total_water_mixing_ratio(qv)
        assert np.allclose(result.values, qv.values)
        assert result.attrs["included_species"] == "qv"
        assert result.attrs["units"] == "kg/kg"

    def test_sums_all_condensates(
        self: "TestTotalWaterMixingRatio",
        diag_verbose: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the compute_total_water_mixing_ratio method correctly sums the input qv and all provided condensate variables to compute the total water mixing ratio. It provides valid qv and individual condensate DataArrays, computes the total water mixing ratio, and asserts that the computed values match the expected sum of qv and all condensates. It also checks that the resulting DataArray attributes indicate which species are included in the total and that the units are correct, ensuring that the method returns accurate results with appropriate metadata when multiple condensate variables are provided.

        Parameters:
            diag_verbose (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=True.

        Returns:
            None
        """
        qv = xr.DataArray(np.array([0.01]), dims=["x"])
        qc = xr.DataArray(np.array([0.0005]), dims=["x"])
        qr = xr.DataArray(np.array([0.0001]), dims=["x"])
        qi = xr.DataArray(np.array([0.00005]), dims=["x"])
        qs = xr.DataArray(np.array([0.00002]), dims=["x"])
        qg = xr.DataArray(np.array([0.00001]), dims=["x"])

        result = diag_verbose.compute_total_water_mixing_ratio(
            qv,
            qc=qc,
            qr=qr,
            qi=qi,
            qs=qs,
            qg=qg,
        )
        expected = 0.01 + 0.0005 + 0.0001 + 0.00005 + 0.00002 + 0.00001
        assert np.allclose(result.values, expected)
        for name in ("qc", "qr", "qi", "qs", "qg"):
            assert name in result.attrs["included_species"]


class TestAnalyze:
    """Test analyze_density_potential_temperature."""

    def test_without_theta(
        self: "TestAnalyze", diag_silent: ThermodynamicDiagnostics
    ) -> None:
        """
        This test verifies that the analyze_density_potential_temperature method returns an analysis dictionary containing summary statistics for theta_rho when only the theta_rho DataArray is provided. It asserts that the analysis includes keys for 'theta_rho' with expected summary statistics (min, max, mean, std, units) and does not include a 'perturbation' key, confirming that the method correctly analyzes the density potential temperature without reference to a perturbation when theta is not given.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.

        Returns:
            None
        """
        theta_rho = xr.DataArray(np.linspace(295.0, 305.0, 10), dims=["x"])
        analysis = diag_silent.analyze_density_potential_temperature(theta_rho)
        assert "theta_rho" in analysis
        assert "perturbation" not in analysis
        assert set(analysis["theta_rho"]) >= {"min", "max", "mean", "std", "units"}

    def test_with_theta_adds_perturbation(
        self: "TestAnalyze",
        diag_verbose: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the analyze_density_potential_temperature method includes a 'perturbation' analysis when both theta_rho and theta DataArrays are provided. It asserts that the analysis dictionary contains a 'perturbation' key with summary statistics (mean, units) and a description of the buoyancy contribution, confirming that the method correctly computes and analyzes the perturbation of density potential temperature relative to theta and provides detailed diagnostic information about the perturbation's characteristics.

        Parameters:
            diag_verbose (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=True.

        Returns:
            None
        """
        theta = xr.DataArray(np.full(10, 300.0), dims=["x"])
        theta_rho = xr.DataArray(np.full(10, 301.0), dims=["x"])
        analysis = diag_verbose.analyze_density_potential_temperature(
            theta_rho,
            theta=theta,
        )
        assert "perturbation" in analysis
        assert np.isclose(analysis["perturbation"]["mean"], 1.0)
        assert "buoyancy contribution" in analysis["perturbation"]["description"]


class TestColdPoolStrength:
    """Test compute_cold_pool_strength."""

    def test_with_explicit_float_env(
        self: "TestColdPoolStrength",
        diag_silent: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the compute_cold_pool_strength method correctly computes the cold pool strength and buoyancy when an explicit float value for the environmental potential temperature is provided. It asserts that the computed cold pool strength is positive for grid points where theta_rho is colder than the specified environmental temperature, and zero for grid points where theta_rho is warmer. It also checks that the resulting buoyancy and cold pool strength DataArrays have the correct units, confirming that the method correctly computes these diagnostics based on the provided environmental reference temperature.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.

        Returns:
            None
        """
        theta_rho = xr.DataArray(np.array([298.0, 300.0, 302.0]), dims=["x"])
        c, b = diag_silent.compute_cold_pool_strength(
            theta_rho,
            theta_rho_env=300.0,
            cold_pool_depth=1000.0,
        )
        # Element 0 is colder than env → buoyancy negative, C > 0
        # Element 2 is warmer than env → C == 0 there
        assert c.values[0] > 0.0
        assert math.isclose(c.values[2], 0.0)
        assert b.attrs["units"] == "m s-2"
        assert c.attrs["units"] == "m s-1"

    def test_env_none_uses_domain_mean(
        self: "TestColdPoolStrength",
        diag_verbose: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the compute_cold_pool_strength method correctly uses the domain mean of theta_rho as the environmental potential temperature when theta_rho_env is None. It asserts that the computed cold pool strength is positive for grid points where theta_rho is colder than the domain mean, and that the buoyancy values are centered around zero, confirming that the method correctly identifies cold pool regions based on the relative temperature compared to the domain mean.

        Parameters:
            diag_verbose (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=True.

        Returns:
            None
        """
        theta_rho = xr.DataArray(np.array([298.0, 300.0, 302.0]), dims=["x"])
        c, b = diag_verbose.compute_cold_pool_strength(theta_rho)
        # Mean is 300; element 0 is below → cold pool
        assert c.values[0] > 0.0
        assert float(b.mean()) == pytest.approx(0.0, abs=1e-6)

    def test_env_dataarray(
        self: "TestColdPoolStrength",
        diag_silent: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the compute_cold_pool_strength method correctly handles the case when theta_rho_env is provided as a DataArray. It asserts that the method computes the cold pool strength and buoyancy based on the provided environmental reference DataArray, and that the computed cold pool strength is positive for grid points where theta_rho is colder than the corresponding values in theta_rho_env, confirming that the method can handle DataArray inputs for the environmental reference and compute diagnostics accordingly.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.

        Returns:
            None
        """
        theta_rho = xr.DataArray(np.array([298.0, 300.0]), dims=["x"])
        env = xr.DataArray(np.array([299.0, 299.0]), dims=["x"])
        c, _ = diag_silent.compute_cold_pool_strength(
            theta_rho,
            theta_rho_env=env,
        )
        assert c.values[0] > 0.0
        assert math.isclose(c.values[1], 0.0)

    def test_no_cold_pool_anywhere_verbose(
        self: "TestColdPoolStrength",
        diag_verbose: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the compute_cold_pool_strength method correctly identifies the case when there are no cold pool grid points (i.e., all theta_rho values are warmer than the environmental reference) and that it handles this scenario without errors in verbose mode. It asserts that the computed cold pool strength is zero for all grid points, confirming that the method correctly computes a cold pool strength of zero when there are no colder-than-environment grid points, and that it does so while providing verbose diagnostic output without encountering issues related to having no cold pool points.

        Parameters:
            diag_verbose (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=True.

        Returns:
            None
        """
        # All warmer than env → no cold-pool grid points; covers branch where
        # n_cold == 0 in the verbose summary.
        theta_rho = xr.DataArray(np.array([301.0, 302.0, 303.0]), dims=["x"])
        c, _ = diag_verbose.compute_cold_pool_strength(
            theta_rho,
            theta_rho_env=300.0,
        )
        assert math.isclose(float(c.max()), 0.0)


class TestResolveLevelIndex:
    """Test private level-index resolution branches via compute_from_dataset."""

    def test_int_level_out_of_range_raises(
        self: "TestResolveLevelIndex",
        diag_silent: ThermodynamicDiagnostics,
        ds_3d: xr.Dataset,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature_from_dataset method raises a ValueError when an integer level index is provided that exceeds the available vertical levels in the dataset. It asserts that the error message contains the phrase "exceeds available levels", ensuring that the method correctly identifies out-of-range integer level indices and provides an informative error message. This test ensures that the method has robust error handling for invalid integer level inputs and can guide users to provide valid level indices within the bounds of the dataset's vertical dimension.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.
            ds_3d (xr.Dataset): A synthetic 3D dataset containing the necessary variables for computation.

        Returns:
            None
        """
        with pytest.raises(ValueError, match="exceeds available levels"):
            diag_silent.compute_density_potential_temperature_from_dataset(
                ds_3d,
                level=99,
            )

    def test_unknown_string_level_raises(
        self: "TestResolveLevelIndex",
        diag_silent: ThermodynamicDiagnostics,
        ds_3d: xr.Dataset,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature_from_dataset method raises a ValueError when an unknown string level specification is provided. It asserts that the error message contains the phrase "Unknown level specification", ensuring that the method correctly identifies invalid string level inputs and provides an informative error message. This test ensures that the method has robust error handling for invalid string level specifications and can guide users to provide valid level names or indices.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.
            ds_3d (xr.Dataset): A synthetic 3D dataset containing the necessary variables for computation.

        Returns:
            None
        """
        with pytest.raises(ValueError, match="Unknown level specification"):
            diag_silent.compute_density_potential_temperature_from_dataset(
                ds_3d,
                level="middle",
            )

    def test_invalid_level_type_raises(
        self: "TestResolveLevelIndex",
        diag_silent: ThermodynamicDiagnostics,
        ds_3d: xr.Dataset,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature_from_dataset method raises a ValueError when an invalid type of level specification is provided (e.g., a list instead of an integer or string). It asserts that the error message contains the phrase "Invalid level specification", ensuring that the method correctly identifies invalid level input types and provides an informative error message. This test ensures that the method has robust error handling for level specifications of incorrect types and can guide users to provide valid level inputs.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.
            ds_3d (xr.Dataset): A synthetic 3D dataset containing the necessary variables for computation.

        Returns:
            None
        """
        with pytest.raises(ValueError, match="Invalid level specification"):
            diag_silent.compute_density_potential_temperature_from_dataset(
                ds_3d,
                level=[0, 1],  # type: ignore[arg-type]
            )

    def test_pressure_level_without_pressure_vars_raises(
        self: "TestResolveLevelIndex",
        diag_silent: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature_from_dataset method raises a ValueError when a pressure level specification is provided but the dataset does not contain the necessary pressure variables to resolve the pressure level. It asserts that the error message contains the phrase "pressure_p", ensuring that the method correctly identifies the absence of required pressure variables and provides an informative error message. This test ensures that the method has robust error handling for cases where a pressure level is requested but cannot be resolved due to missing data in the dataset.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.

        Returns:
            None
        """
        ds = _make_3d_dataset(include_condensates=False, include_pressure=False)
        with pytest.raises(ValueError, match="pressure_p"):
            diag_silent.compute_density_potential_temperature_from_dataset(
                ds,
                level=85000.0,
            )

    def test_2d_data_with_level_skips_vertical_selection(
        self: "TestResolveLevelIndex",
        diag_silent: ThermodynamicDiagnostics,
        ds_2d: xr.Dataset,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature_from_dataset method correctly handles the case when a level selection is made on a dataset that does not have a vertical dimension (i.e., it is 2D). It asserts that the method ignores the level selection and successfully computes the density potential temperature, returning a result with only the 'nCells' dimension. This test ensures that the method can gracefully handle cases where a user specifies a level selection for a dataset that does not contain vertical levels, without raising an error and while still producing valid output.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.
            ds_2d (xr.Dataset): A synthetic 2D dataset containing the necessary variables for computation.

        Returns:
            None
        """
        # No vertical dim → level is ignored; result has nCells only.
        result = diag_silent.compute_density_potential_temperature_from_dataset(
            ds_2d,
            level="surface",
        )
        assert "nCells" in result.dims

    def test_nvertlevelsp1_dim_handled(
        self: "TestResolveLevelIndex",
        diag_silent: ThermodynamicDiagnostics,
    ) -> None:
        """
        This test verifies that the compute_density_potential_temperature_from_dataset method correctly handles datasets that have a vertical dimension named 'nVertLevelsP1' instead of 'nVertLevels'. It asserts that the method can successfully resolve the vertical dimension and compute the density potential temperature without raising an error, and that the resulting DataArray does not have the 'nVertLevelsP1' dimension, confirming that the method correctly processes datasets with this alternative vertical dimension naming convention. This test ensures that the method is flexible in handling different dataset structures and can still perform computations correctly when the vertical dimension is named differently.

        Parameters:
            diag_silent (ThermodynamicDiagnostics): An instance of the diagnostics class with verbose=False.

        Returns:
            None
        """
        ds = _make_3d_dataset(
            include_condensates=False,
            vert_name="nVertLevelsP1",
        )
        result = diag_silent.compute_density_potential_temperature_from_dataset(
            ds,
            time_index=0,
            level=70000.0,
        )
        assert "nVertLevelsP1" not in result.dims


class TestStaticHelpers:
    """Test the static vertical-dimension helpers directly."""

    def test_resolve_vertical_dim_finds_nvertlevels(self: "TestStaticHelpers") -> None:
        """
        This test verifies that the _resolve_vertical_dim static method correctly identifies the 'nVertLevels' dimension in a DataArray. It asserts that when a DataArray with dimensions including 'nVertLevels' is passed to the method, it returns 'nVertLevels' as the identified vertical dimension, confirming that the method can successfully resolve this common vertical dimension name. This test ensures that the method's logic for identifying the 'nVertLevels' dimension is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(np.zeros((2, 3)), dims=["nCells", "nVertLevels"])
        assert ThermodynamicDiagnostics._resolve_vertical_dim(da) == "nVertLevels"

    def test_resolve_vertical_dim_finds_nvertlevelsp1(
        self: "TestStaticHelpers",
    ) -> None:
        """
        This test verifies that the _resolve_vertical_dim static method correctly identifies the 'nVertLevelsP1' dimension in a DataArray. It asserts that when a DataArray with dimensions including 'nVertLevelsP1' is passed to the method, it returns 'nVertLevelsP1' as the identified vertical dimension, confirming that the method can successfully resolve this alternative vertical dimension name. This test ensures that the method's logic for identifying the 'nVertLevelsP1' dimension is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(np.zeros((2, 3)), dims=["nCells", "nVertLevelsP1"])
        assert ThermodynamicDiagnostics._resolve_vertical_dim(da) == "nVertLevelsP1"

    def test_resolve_vertical_dim_returns_none(self: "TestStaticHelpers") -> None:
        """
        This test verifies that the _resolve_vertical_dim static method correctly returns None when no vertical dimension is present in a DataArray. It asserts that when a DataArray with dimensions that do not include 'nVertLevels' or 'nVertLevelsP1' is passed to the method, it returns None, confirming that the method can correctly identify the absence of recognized vertical dimensions and return an appropriate value. This test ensures that the method's logic for resolving vertical dimensions includes proper handling for cases where no valid vertical dimension is found.

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(np.zeros(3), dims=["nCells"])
        assert ThermodynamicDiagnostics._resolve_vertical_dim(da) is None

    def test_vert_dim_finds_nvertlevels(self: "TestStaticHelpers") -> None:
        """
        This test verifies that the _vert_dim static method correctly identifies the 'nVertLevels' dimension in a DataArray. It asserts that when a DataArray with dimensions including 'nVertLevels' is passed to the method, it returns 'nVertLevels' as the identified vertical dimension, confirming that the method can successfully identify this common vertical dimension name. This test ensures that the method's logic for identifying the 'nVertLevels' dimension is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(np.zeros((2, 3)), dims=["nCells", "nVertLevels"])
        assert ThermodynamicDiagnostics._vert_dim(da) == "nVertLevels"

    def test_vert_dim_finds_nvertlevelsp1(self: "TestStaticHelpers") -> None:
        """
        This test verifies that the _vert_dim static method correctly identifies the 'nVertLevelsP1' dimension in a DataArray. It asserts that when a DataArray with dimensions including 'nVertLevelsP1' is passed to the method, it returns 'nVertLevelsP1' as the identified vertical dimension, confirming that the method can successfully identify this alternative vertical dimension name. This test ensures that the method's logic for identifying the 'nVertLevelsP1' dimension is functioning as intended.

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(np.zeros((2, 3)), dims=["nCells", "nVertLevelsP1"])
        assert ThermodynamicDiagnostics._vert_dim(da) == "nVertLevelsP1"

    def test_vert_dim_returns_none(self: "TestStaticHelpers") -> None:
        """
        This test verifies that the _vert_dim static method correctly returns None when no vertical dimension is present in a DataArray. It asserts that when a DataArray with dimensions that do not include 'nVertLevels' or 'nVertLevelsP1' is passed to the method, it returns None, confirming that the method can correctly identify the absence of recognized vertical dimensions and return an appropriate value. This test ensures that the method's logic for identifying vertical dimensions includes proper handling for cases where no valid vertical dimension is found.

        Parameters:
            None

        Returns:
            None
        """
        da = xr.DataArray(np.zeros(3), dims=["nCells"])
        assert ThermodynamicDiagnostics._vert_dim(da) is None

    def test_print_helpers_silent_early_return(self: "TestStaticHelpers") -> None:
        """
        This test verifies that the _print_input_summary and _print_result_summary helper methods in the ThermodynamicDiagnostics class return early without performing any operations when the verbose attribute is set to False. It asserts that when these methods are called with valid input DataArrays while verbose=False, they do not raise any errors and simply return without printing any summaries, confirming that the methods correctly implement an early return mechanism to avoid unnecessary computations and output when verbose mode is disabled. This test ensures that the diagnostics class can operate efficiently in silent mode without attempting to generate summaries or print output.

        Parameters:
            None

        Returns:
            None
        """
        # verbose=False → both print helpers should return early without error.
        diag = ThermodynamicDiagnostics(verbose=False)
        theta = xr.DataArray(np.full(3, 300.0), dims=["x"])
        qv = xr.DataArray(np.full(3, 0.01), dims=["x"])
        qt = xr.DataArray(np.full(3, 0.012), dims=["x"])
        diag._print_input_summary(theta, qv, qt)
        diag._print_result_summary(theta)

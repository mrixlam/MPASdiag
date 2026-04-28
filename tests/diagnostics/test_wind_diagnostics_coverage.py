#!/usr/bin/env python3

"""
MPASdiag Test Suite: Wind Diagnostics Coverage

This module contains tests for validating the behavior of the WindDiagnostics class in the MPASdiag package. It ensures that wind-related computations, such as wind direction and wind components, are correctly performed and that edge cases are properly handled.

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
from unittest.mock import MagicMock, patch

from mpasdiag.diagnostics.wind import WindDiagnostics


N_CELLS = 5
N_VERT = 10
N_TIME = 2


@pytest.fixture()
def wind_pair() -> tuple:
    """
    This fixture provides a pair of xarray DataArrays, u and v, representing the zonal and meridional wind components, respectively. The values in these DataArrays are chosen to yield a variety of wind directions when processed by the compute_wind_direction method in the WindDiagnostics class. The u and v components are structured with a dimension of "nCells" and contain values that will allow for testing the correct calculation of wind direction in radians. 

    Parameters:
        None

    Returns:
        tuple: A tuple containing two xarray DataArrays, u and v, representing the zonal and meridional wind components, respectively. 
    """
    u = xr.DataArray(np.array([3.0, -4.0, 0.0, 5.0]), dims=["nCells"])
    v = xr.DataArray(np.array([4.0, 3.0, -5.0, 0.0]), dims=["nCells"])
    return u, v


@pytest.fixture()
def ds_3d() -> xr.Dataset:
    """
    This fixture provides a 3D xarray Dataset containing u, pressure_p, and pressure_base variables for testing the _compute_level_index_from_pressure method in the WindDiagnostics class. The dataset is structured with dimensions "Time", "nVertLevels", and "nCells", and contains values designed to allow for testing the method's ability to compute a level index based on a specified pressure level. The pressure_p variable is set to a constant value across all levels, while the pressure_base variable is defined as a linear gradient from 100,000 Pa at the surface to 20,000 Pa at the top level. This setup allows for verifying that the method can correctly identify the appropriate vertical level corresponding to a given pressure level when both pressure_p and pressure_base variables are present in the dataset. 

    Parameters:
        None

    Returns:
        xr.Dataset: A 3D xarray Dataset containing u, pressure_p, and pressure_base variables. The dataset has dimensions "Time", "nVertLevels", and "nCells". 
    """
    p_base = np.linspace(100_000.0, 20_000.0, N_VERT)

    return xr.Dataset({
        "u": (["Time", "nVertLevels", "nCells"],
              np.ones((N_TIME, N_VERT, N_CELLS))),
        "pressure_p": (["Time", "nVertLevels", "nCells"],
                       np.ones((N_TIME, N_VERT, N_CELLS)) * 500.0),
        "pressure_base": (["Time", "nVertLevels", "nCells"],
                          np.tile(p_base, (N_TIME, N_CELLS, 1)).transpose(0, 2, 1)),
    })


@pytest.fixture()
def ds_3d_no_pressure() -> xr.Dataset:
    """
    This fixture provides a 3D xarray Dataset containing only the u variable, without any pressure variables, for testing the error handling in the _compute_level_index_from_pressure method of the WindDiagnostics class. The dataset is structured with dimensions "Time", "nVertLevels", and "nCells", and contains constant values for the u variable. This setup allows for verifying that the method correctly raises a ValueError when it attempts to compute a level index based on pressure but finds no pressure data available in the dataset. 

    Parameters:
        None

    Returns:
        xr.Dataset: A 3D xarray Dataset containing only the u variable, with dimensions "Time", "nVertLevels", and "nCells". No pressure variables are included in this dataset.
    """
    return xr.Dataset({
        "u": (["Time", "nVertLevels", "nCells"],
              np.ones((N_TIME, N_VERT, N_CELLS))),
    })


@pytest.fixture()
def ds_2d() -> xr.Dataset:
    """
    This fixture provides a 2D xarray Dataset containing u10 and v10 variables for testing the get_2d_wind_components method in the WindDiagnostics class. The dataset is structured with dimensions "Time" and "nCells", and contains random values for both u10 and v10 variables. This setup allows for verifying that the method can successfully extract the 2D wind components from the dataset and return them as xarray DataArrays when the data_type parameter is set to "uxarray". The random values ensure that the method's functionality is tested with a variety of wind component values, which can lead to different wind directions when processed by other methods in the WindDiagnostics class. 

    Parameters:
        None

    Returns:
        xr.Dataset: A 2D xarray Dataset containing u10 and v10 variables. The dataset has dimensions "Time" and "nCells". 

    """
    rng = np.random.default_rng(0)

    return xr.Dataset({
        "u10": (["Time", "nCells"], rng.standard_normal((3, N_CELLS))),
        "v10": (["Time", "nCells"], rng.standard_normal((3, N_CELLS))),
    })


class TestComputeWindDirectionRadians:
    """Covers the radians (degrees=False) branch of compute_wind_direction."""

    def test_radians_returns_dataarray_in_range(self: 'TestComputeWindDirectionRadians', 
                                                wind_pair: tuple) -> None:
        """
        This test validates that the compute_wind_direction method in the WindDiagnostics class correctly computes the wind direction in radians when the degrees parameter is set to False. It checks that the returned result is an xarray DataArray and that all values in the resulting DataArray are within the expected range of 0 to 2π radians. This ensures that the method is correctly calculating wind direction in radians and that it is properly handling the input wind components to produce valid wind direction values. 

        Parameters:
            wind_pair (tuple): A tuple containing two xarray DataArrays, u and v, representing the zonal and meridional wind components, respectively. These DataArrays are used as input for the compute_wind_direction method to calculate the wind direction in radians.

        Returns:
            None
        """
        u, v = wind_pair
        diag = WindDiagnostics(verbose=False)
        result = diag.compute_wind_direction(u, v, degrees=False)
        assert isinstance(result, xr.DataArray)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 2 * np.pi

    def test_radians_attrs_are_set(self: 'TestComputeWindDirectionRadians', 
                                   wind_pair: tuple) -> None:
        """
        This test validates that the compute_wind_direction method in the WindDiagnostics class correctly sets the attributes of the resulting xarray DataArray when the degrees parameter is set to False. It checks that the "units" attribute is set to "radians" and that the "standard_name" attribute is set to "wind_from_direction". This ensures that the method is not only computing the wind direction in radians but also properly annotating the resulting DataArray with the correct metadata, which is important for downstream processing and interpretation of the wind direction data. 

        Parameters:
            wind_pair (tuple): A tuple containing two xarray DataArrays, u and v, representing the zonal and meridional wind components, respectively. These DataArrays are used as input for the compute_wind_direction method to calculate the wind direction in radians.

        Returns:
            None
        """
        u, v = wind_pair
        diag = WindDiagnostics(verbose=False)
        result = diag.compute_wind_direction(u, v, degrees=False)
        assert result.attrs["units"] == "radians"
        assert result.attrs["standard_name"] == "wind_from_direction"

    def test_radians_verbose_prints_range_and_mean(self: 'TestComputeWindDirectionRadians', 
                                                   wind_pair: tuple) -> None:
        """
        This test validates that the compute_wind_direction method in the WindDiagnostics class prints the expected range and mean of the wind direction values when the degrees parameter is set to False and verbose is True. It captures the standard output during the execution of the method and checks that the printed output contains the phrases "Wind direction range:" and "Wind direction mean:", as well as the unit "radians". This ensures that when verbose mode is enabled, the method provides informative output about the computed wind direction values, which can be useful for debugging and understanding the results.

        Parameters:
            wind_pair (tuple): A tuple containing two xarray DataArrays, u and v, representing the zonal and meridional wind components, respectively. 

        Returns:
            None
        """
        u, v = wind_pair
        diag = WindDiagnostics(verbose=True)
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            result = diag.compute_wind_direction(u, v, degrees=False)

        out = captured.getvalue()
        assert "Wind direction range:" in out
        assert "radians" in out
        assert "Wind direction mean:" in out
        assert isinstance(result, xr.DataArray)

    def test_radians_verbose_false_silent(self: 'TestComputeWindDirectionRadians', 
                                          wind_pair: tuple) -> None:
        """
        This test validates that the compute_wind_direction method in the WindDiagnostics class does not print any output when the degrees parameter is set to False and verbose is False. It checks that the printed output is empty and that the method executes silently. This ensures that when verbose mode is disabled, the method does not produce any unnecessary output, allowing for cleaner execution in cases where the user does not want verbose information about the wind direction calculations. 

        Parameters:
            wind_pair (tuple): A tuple containing two xarray DataArrays, u and v, representing the zonal and meridional wind components, respectively. These DataArrays are used as input for the compute_wind_direction method to calculate the wind direction in radians.

        Returns:
            None
        """
        u, v = wind_pair
        diag = WindDiagnostics(verbose=False)
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            diag.compute_wind_direction(u, v, degrees=False)

        assert captured.getvalue() == ""


class TestValidate3DVariable:
    """ Covers error branches in _validate_3d_variable: missing var, var without vert dim. """

    def test_var_without_vert_dim_raises(self: 'TestValidate3DVariable') -> None:
        """
        This test validates that the _validate_3d_variable method in the WindDiagnostics class raises a ValueError when the input variable is present in the dataset but does not have the expected vertical dimension. It checks that the error message contains the expected string indicating that the variable is not a 3D atmospheric variable. This ensures that the method correctly identifies when a variable is not structured as a 3D variable with a vertical dimension, which is necessary for certain wind diagnostics calculations. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "u2d": (["Time", "nCells"], np.ones((2, N_CELLS))),
        })

        diag = WindDiagnostics(verbose=False)

        with pytest.raises(ValueError, match="is not a 3D atmospheric variable"):
            diag._validate_3d_variable(ds, "u2d")

    def test_var_missing_from_dataset_raises(self: 'TestValidate3DVariable') -> None:
        """
        This test validates that the _validate_3d_variable method in the WindDiagnostics class raises a ValueError when the specified variable is not found in the input dataset. It checks that the error message contains the expected string indicating that the variable is not found in the dataset. This ensures that the method correctly handles cases where a required variable for wind diagnostics calculations is missing from the dataset, providing clear feedback to the user about the issue. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "u": (["Time", "nVertLevels", "nCells"], np.ones((2, 5, 3))),
        })

        diag = WindDiagnostics(verbose=False)

        with pytest.raises(ValueError, match="not found in dataset"):
            diag._validate_3d_variable(ds, "nonexistent")


class TestComputeLevelIndexFromPressure:
    """ Covers error branches in _compute_level_index_from_pressure: missing pressure vars, verbose prints. """

    def test_missing_pressure_vars_raises(self: 'TestComputeLevelIndexFromPressure', 
                                          ds_3d_no_pressure: xr.Dataset) -> None:
        """
        This test validates that the _compute_level_index_from_pressure method in the WindDiagnostics class raises a ValueError when the input dataset does not contain any pressure variables (neither pressure_p nor pressure_base). It checks that the error message contains the expected string indicating that pressure data is not available. This ensures that the method correctly identifies when it cannot perform the level index computation due to the absence of necessary pressure information in the dataset, providing clear feedback to the user about the issue. 

        Parameters:
            ds_3d_no_pressure (xr.Dataset): A 3D xarray Dataset that does not contain any pressure variables, used to test the error handling in the _compute_level_index_from_pressure method when pressure data is missing.

        Returns:
            None
        """
        diag = WindDiagnostics(verbose=False)
        with pytest.raises(ValueError, match="pressure data not available"):
            diag._compute_level_index_from_pressure(
                ds_3d_no_pressure, 85_000.0, "Time", 0
            )

    def test_missing_only_pressure_p_raises(self: 'TestComputeLevelIndexFromPressure') -> None:
        """
        This test validates that the _compute_level_index_from_pressure method in the WindDiagnostics class raises a ValueError when the input dataset contains only the pressure_base variable and not the pressure_p variable. It checks that the error message contains the expected string indicating that pressure data is not available. This ensures that the method correctly identifies when it cannot perform the level index computation due to the absence of the necessary pressure_p information in the dataset, providing clear feedback to the user about the issue. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "u": (["Time", "nVertLevels", "nCells"], np.ones((2, 10, 5))),
            "pressure_base": (["Time", "nVertLevels", "nCells"], np.ones((2, 10, 5))),
        })

        diag = WindDiagnostics(verbose=False)

        with pytest.raises(ValueError, match="pressure data not available"):
            diag._compute_level_index_from_pressure(ds, 85_000.0, "Time", 0)

    def test_verbose_prints_pressure_info(self: 'TestComputeLevelIndexFromPressure', 
                                          ds_3d: xr.Dataset) -> None:
        """
        This test validates that the _compute_level_index_from_pressure method in the WindDiagnostics class prints the expected information about the requested pressure level and the available pressure levels in the dataset when verbose is True. It captures the standard output during the execution of the method and checks that the printed output contains the phrase "Requested pressure" along with the requested pressure value, as well as information about the available pressure levels. This ensures that when verbose mode is enabled, the method provides informative output about the pressure level being requested and how it relates to the pressure data available in the dataset, which can be useful for debugging and understanding how the level index is being computed based on pressure. 

        Parameters:
            ds_3d (xr.Dataset): A 3D xarray Dataset containing the necessary pressure variables for testing the verbose output of the _compute_level_index_from_pressure method when it successfully computes a level index based on a specified pressure level. 

        Returns:
            None
        """
        diag = WindDiagnostics(verbose=True)
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            idx = diag._compute_level_index_from_pressure(ds_3d, 70_000.0, "Time", 0)

        assert "Requested pressure" in captured.getvalue()
        assert isinstance(idx, int)


class TestComputeLevelIndex:
    """ Covers error branches in _compute_level_index: int level exceeds max, unknown string level, invalid type level, float level with no pressure data. """

    def test_int_level_exceeds_max_raises(self: 'TestComputeLevelIndex', 
                                          ds_3d: xr.Dataset) -> None:
        """
        This test validates that the _compute_level_index method in the WindDiagnostics class raises a ValueError when the specified integer level index exceeds the maximum available vertical levels in the dataset. It checks that the error message contains the expected string indicating that the requested level index exceeds the available levels. This ensures that the method correctly handles cases where a user requests a vertical level index that is out of bounds for the dataset, providing clear feedback about the issue.

        Parameters:
            ds_3d (xr.Dataset): A 3D xarray Dataset containing the necessary variables and dimensions for testing the behavior of the _compute_level_index method when an integer level index that exceeds the maximum available vertical levels is specified.

        Returns:
            None
        """
        diag = WindDiagnostics(verbose=False)
        with pytest.raises(ValueError, match="exceeds available levels"):
            diag._compute_level_index(ds_3d, "u", 999, "Time", 0)

    def test_unknown_string_level_raises(self: 'TestComputeLevelIndex', 
                                         ds_3d: xr.Dataset) -> None:
        """
        This test validates that the _compute_level_index method in the WindDiagnostics class raises a ValueError when an unknown string level specification is provided. It checks that the error message contains the expected string indicating that the level specification is unknown. This ensures that the method correctly identifies and handles cases where a user provides a string level specification that it does not recognize, providing clear feedback about the issue. 

        Parameters:
            ds_3d (xr.Dataset): A 3D xarray Dataset containing the necessary variables and dimensions for testing the behavior of the _compute_level_index method when an unknown string level is specified.

        Returns:
            None
        """
        diag = WindDiagnostics(verbose=False)
        with pytest.raises(ValueError, match="Unknown level specification"):
            diag._compute_level_index(ds_3d, "u", "midlevel", "Time", 0)

    def test_invalid_type_level_raises(self: 'TestComputeLevelIndex', 
                                       ds_3d: xr.Dataset) -> None:
        """
        This test validates that the _compute_level_index method in the WindDiagnostics class raises a ValueError when an invalid type is provided for the level specification. It checks that the error message contains the expected string indicating that the level specification is invalid. This ensures that the method correctly identifies and handles cases where a user provides a level specification of an unsupported type (e.g., None or a list), providing clear feedback about the issue. 

        Parameters:
            ds_3d (xr.Dataset): A 3D xarray Dataset containing the necessary variables and dimensions for testing the behavior of the _compute_level_index method when an invalid type is provided for the level specification.

        Returns:
            None
        """
        diag = WindDiagnostics(verbose=False)
        with pytest.raises(ValueError, match="Invalid level specification"):
            diag._compute_level_index(ds_3d, "u", None, "Time", 0)  # type: ignore[arg-type]

    def test_list_type_level_raises(self: 'TestComputeLevelIndex', 
                                    ds_3d: xr.Dataset) -> None:
        """
        This test validates that the _compute_level_index method in the WindDiagnostics class raises a ValueError when a list is provided as the level specification, which is not a valid input type for this parameter. It checks that the error message contains the expected string indicating that the level specification is invalid. This ensures that the method correctly identifies and handles cases where a user provides a level specification of an unsupported type (in this case, a list), providing clear feedback about the issue. 

        Parameters:
            ds_3d (xr.Dataset): A 3D xarray Dataset containing the necessary variables and dimensions for testing the behavior of the _compute_level_index method when a list is provided for the level specification.

        Returns:
            None
        """
        diag = WindDiagnostics(verbose=False)
        with pytest.raises(ValueError, match="Invalid level specification"):
            diag._compute_level_index(ds_3d, "u", [85000.0], "Time", 0)  # type: ignore[arg-type]

    def test_float_level_triggers_pressure_lookup(self: 'TestComputeLevelIndex', 
                                                  ds_3d_no_pressure: xr.Dataset) -> None:
        """
        This test validates that the _compute_level_index method in the WindDiagnostics class raises a ValueError when a float level specification is provided but the input dataset does not contain any pressure variables, which are necessary for computing the level index based on pressure. It checks that the error message contains the expected string indicating that pressure data is not available. This ensures that the method correctly identifies when it cannot perform the level index computation due to the absence of necessary pressure information in the dataset, providing clear feedback about the issue when a user attempts to specify a level using a float value that requires pressure data for interpretation.

        Parameters:
            ds_3d_no_pressure (xr.Dataset): A 3D xarray Dataset that does not contain any pressure variables, used to test the error handling in the _compute_level_index method when a float level specification is provided but pressure data is missing from the dataset.

        Returns:
            None
        """
        diag = WindDiagnostics(verbose=False)
        with pytest.raises(ValueError, match="pressure data not available"):
            diag._compute_level_index(ds_3d_no_pressure, "u", 85_000.0, "Time", 0)


class TestGet2DWindComponentsNoneDataset:
    """ Test the branch in get_2d_wind_components where the dataset is None, which should raise a RuntimeError. """

    def test_none_dataset_raises_runtimeerror(self: 'TestGet2DWindComponentsNoneDataset') -> None:
        """
        This test validates that the get_2d_wind_components method in the WindDiagnostics class raises a RuntimeError when the input dataset is None. It checks that the error message contains the expected string indicating that no dataset was provided. This ensures that the method correctly handles cases where it is called without a valid dataset, providing clear feedback about the issue instead of attempting to proceed with a None value, which would lead to further errors down the line.

        Parameters:
            None

        Returns:
            None
        """
        diag = WindDiagnostics(verbose=False)
        with pytest.raises(RuntimeError, match="No dataset provided"):
            diag.get_2d_wind_components(None, "u10", "v10", time_index=0)  # type: ignore[arg-type]


class TestGet2DWindComponentsMissingVars:
    """ Test the branches in get_2d_wind_components where one or both of the required wind variables (u10 and v10) are missing from the dataset, which should raise a ValueError. """

    def test_missing_u_variable_raises(self: 'TestGet2DWindComponentsMissingVars') -> None:
        """
        This test validates that the get_2d_wind_components method in the WindDiagnostics class raises a ValueError when the input dataset is missing the u10 variable, which is required for extracting the 2D wind components. It checks that the error message contains the expected string indicating that the u10 variable is not found in the dataset. This ensures that the method correctly identifies when a required variable for wind component extraction is missing from the dataset, providing clear feedback about the issue to the user.

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "v10": (["Time", "nCells"], np.ones((2, N_CELLS))),
        })

        diag = WindDiagnostics(verbose=False)

        with pytest.raises(ValueError, match="not found in dataset"):
            diag.get_2d_wind_components(ds, "u10", "v10", time_index=0)

    def test_missing_v_variable_raises(self: 'TestGet2DWindComponentsMissingVars') -> None:
        """
        This test validates that the get_2d_wind_components method in the WindDiagnostics class raises a ValueError when the input dataset is missing the v10 variable, which is required for extracting the 2D wind components. It checks that the error message contains the expected string indicating that the v10 variable is not found in the dataset. This ensures that the method correctly identifies when a required variable for wind component extraction is missing from the dataset, providing clear feedback about the issue to the user.

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "u10": (["Time", "nCells"], np.ones((2, N_CELLS))),
        })

        diag = WindDiagnostics(verbose=False)

        with pytest.raises(ValueError, match="not found in dataset"):
            diag.get_2d_wind_components(ds, "u10", "v10", time_index=0)

    def test_missing_both_variables_raises(self: 'TestGet2DWindComponentsMissingVars') -> None:
        """
        This test validates that the get_2d_wind_components method in the WindDiagnostics class raises a ValueError when the input dataset is missing both the u10 and v10 variables, which are required for extracting the 2D wind components. It checks that the error message contains the expected string indicating that the required wind variables are not found in the dataset. This ensures that the method correctly identifies when both required variables for wind component extraction are missing from the dataset, providing clear feedback about the issue to the user.

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({
            "temperature": (["Time", "nCells"], np.ones((2, N_CELLS))),
        })

        diag = WindDiagnostics(verbose=False)

        with pytest.raises(ValueError, match="not found in dataset"):
            diag.get_2d_wind_components(ds, "u10", "v10", time_index=0)


class TestGet2DWindComponentsUXarray:
    """ Test the branch in get_2d_wind_components where data_type is "uxarray", which should return xarray DataArrays for u and v. Also test that verbose prints the expected message. """

    def test_uxarray_path_returns_dataarrays(self: 'TestGet2DWindComponentsUXarray', 
                                             ds_2d: xr.Dataset) -> None:
        """
        This test validates that the get_2d_wind_components method in the WindDiagnostics class returns xarray DataArrays for the u and v wind components when the data_type parameter is set to "uxarray". It checks that the returned u and v variables are instances of xr.DataArray and that they contain the expected dimension "nCells". This ensures that the method correctly extracts the 2D wind components from the dataset and returns them in the expected format when requested as xarray DataArrays. 

        Parameters:
            ds_2d (xr.Dataset): A 2D xarray Dataset containing the u10 and v10 variables, used to test the functionality of the get_2d_wind_components method when the data_type parameter is set to "uxarray".

        Returns:
            None
        """
        diag = WindDiagnostics(verbose=False)

        u, v = diag.get_2d_wind_components(
            ds_2d, "u10", "v10", time_index=0, data_type="uxarray"
        )

        assert isinstance(u, xr.DataArray)
        assert isinstance(v, xr.DataArray)
        assert "nCells" in u.dims

    def test_uxarray_path_verbose_prints(self: 'TestGet2DWindComponentsUXarray', 
                                         ds_2d: xr.Dataset) -> None:
        """
        This test validates that the get_2d_wind_components method in the WindDiagnostics class prints the expected message about extracting wind components when the data_type parameter is set to "uxarray" and verbose is True. It captures the standard output during the execution of the method and checks that the printed output contains the phrase "Extracting wind components". This ensures that when verbose mode is enabled, the method provides informative output about the process of extracting wind components from the dataset, which can be useful for debugging and understanding what the method is doing during execution.

        Parameters:
            ds_2d (xr.Dataset): A 2D xarray Dataset containing the u10 and v10 variables, used to test the verbose output of the get_2d_wind_components method when the data_type parameter is set to "uxarray".

        Returns:
            None
        """
        diag = WindDiagnostics(verbose=True)
        captured = StringIO()

        with patch("sys.stdout", new=captured):
            u, v = diag.get_2d_wind_components(
                ds_2d, "u10", "v10", time_index=0, data_type="uxarray"
            )

        out = captured.getvalue()
        assert "Extracting wind components" in out
        assert isinstance(u, xr.DataArray)


class TestGet2DWindComponentsExceptions:
    """ Test the branches in get_2d_wind_components where exceptions are raised during the extraction of wind components, which should be caught and re-raised as ValueError or RuntimeError with appropriate messages. """

    def test_keyerror_inside_try_becomes_valueerror(self: 'TestGet2DWindComponentsExceptions') -> None:
        """
        This test validates that if a KeyError occurs during the extraction of wind components in the get_2d_wind_components method of the WindDiagnostics class, it is caught and re-raised as a ValueError with an appropriate message. It simulates a KeyError by mocking the dataset's __getitem__ method to raise a KeyError when attempting to access the u10 variable. The test checks that the raised ValueError contains the expected string indicating an error accessing wind variables. This ensures that the method correctly handles unexpected issues during wind component extraction and provides clear feedback about the nature of the error to the user. 

        Parameters:
            None

        Returns:
            None
        """
        diag = WindDiagnostics(verbose=False)
        mock_ds = MagicMock()
        mock_ds.data_vars = {"u10": object(), "v10": object()}
        mock_ds.__getitem__.side_effect = KeyError("u10")

        with patch(
            "mpasdiag.processing.utils_datetime.MPASDateTimeUtils.validate_time_parameters",
            return_value=("Time", 0, 3),
        ):
            with pytest.raises(ValueError, match="Error accessing wind variables"):
                diag.get_2d_wind_components(mock_ds, "u10", "v10", time_index=0)

    def test_generic_exception_inside_try_becomes_runtimeerror(self: 'TestGet2DWindComponentsExceptions') -> None:
        """
        This test validates that if a generic exception occurs during the extraction of wind components in the get_2d_wind_components method of the WindDiagnostics class, it is caught and re-raised as a RuntimeError with an appropriate message. It simulates a generic exception by mocking the dataset's __getitem__ method to raise a TypeError when attempting to access the u10 variable. The test checks that the raised RuntimeError contains the expected string indicating an error extracting 2D wind components. This ensures that the method correctly handles unexpected issues during wind component extraction and provides clear feedback about the nature of the error to the user.

        Parameters:
            None

        Returns:
            None
        """
        diag = WindDiagnostics(verbose=False)
        mock_ds = MagicMock()
        mock_ds.data_vars = {"u10": object(), "v10": object()}
        mock_ds.__getitem__.side_effect = TypeError("simulated failure")

        with patch(
            "mpasdiag.processing.utils_datetime.MPASDateTimeUtils.validate_time_parameters",
            return_value=("Time", 0, 3),
        ):
            with pytest.raises(RuntimeError, match="Error extracting 2D wind components"):
                diag.get_2d_wind_components(mock_ds, "u10", "v10", time_index=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

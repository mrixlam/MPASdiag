#!/usr/bin/env python3

"""
MPASdiag Test Suite: Sounding Diagnostics

This test suite focuses on the SoundingDiagnostics class, which provides methods for extracting sounding profiles from MPAS datasets and computing various thermodynamic indices. The tests cover the core functionalities of the class, including finding the nearest grid cell to a target location, computing dewpoint temperatures from mixing ratios, converting potential temperature to actual temperature, and calculating a range of thermodynamic indices. Additionally, the tests verify the behavior of the extract_sounding_profile method under various conditions, including edge cases such as missing moisture or wind variables. The use of fixtures allows for reusable setup of synthetic datasets and mock processors to facilitate comprehensive testing of the diagnostics functionality. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: March 2026
Version: 1.0.0
"""
# Load standard libraries
import pytest
import numpy as np
import xarray as xr
from pathlib import Path
from unittest.mock import Mock

from mpasdiag.diagnostics.sounding import SoundingDiagnostics
from mpasdiag.processing.processors_3d import MPAS3DProcessor

@pytest.fixture
def diag() -> SoundingDiagnostics:
    """
    This fixture provides a standard SoundingDiagnostics instance for testing. It is used in multiple test cases to ensure consistency across tests that require an instance of the diagnostics class. The verbose flag is set to False for this fixture, meaning that it will not print additional information during execution, which is suitable for most unit tests where we want to focus on functionality rather than output. 

    Parameters:
        None

    Returns:
        SoundingDiagnostics: An instance of the SoundingDiagnostics class with verbose mode disabled.
    """
    return SoundingDiagnostics(verbose=False)


@pytest.fixture
def diag_verbose() -> SoundingDiagnostics:
    """
    This fixture provides a SoundingDiagnostics instance with verbose mode enabled. It is intended for tests that want to verify the output printed during the execution of certain methods, such as extract_sounding_profile. By setting verbose to True, this instance will print additional information about the nearest cell and the sounding profile extraction process, which can be captured and asserted in tests that check for correct logging behavior. 

    Parameters:
        None

    Returns:
        SoundingDiagnostics: An instance of the SoundingDiagnostics class with verbose mode enabled.
    """
    return SoundingDiagnostics(verbose=True)


@pytest.fixture
def sample_grid_coords() -> tuple[np.ndarray, np.ndarray]:
    """
    This fixture generates synthetic longitude and latitude coordinates for a grid of 100 cells spread across the globe. The coordinates are randomly generated within the valid ranges for longitude (-180 to 180 degrees) and latitude (-90 to 90 degrees). A fixed random seed is used to ensure reproducibility of the generated coordinates across test runs. These coordinates can be used in tests that require a sample grid for nearest-cell search or other spatial operations. 

    Parameters:
        None

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays of longitude and latitude coordinates for the grid cells.
    """
    np.random.seed(42)
    lon = np.random.uniform(-180, 180, 100)
    lat = np.random.uniform(-90, 90, 100)
    return lon, lat


@pytest.fixture
def synthetic_3d_dataset() -> xr.Dataset:
    """
    This fixture creates a synthetic 3D MPAS dataset with dimensions for time, cells, and vertical levels. The dataset includes variables for pressure, potential temperature (theta), specific humidity (qv), and wind components (u and v). The pressure decreases with height, the potential temperature increases slightly with height, and the specific humidity decreases with height, which are typical profiles in the atmosphere. Random noise is added to the potential temperature and specific humidity to make the dataset more realistic. This synthetic dataset can be used in tests that require a sample MPAS dataset for profile extraction or thermodynamic calculations. 

    Parameters:
        None

    Returns:
        xr.Dataset: A synthetic 3D MPAS dataset.
    """
    n_cells = 50
    n_vert = 30
    n_time = 2

    np.random.seed(0)
    p_levels = np.linspace(101000, 5000, n_vert)
    pressure = np.tile(p_levels, (n_time, n_cells, 1))

    theta = np.linspace(300, 350, n_vert)
    theta_3d = np.tile(theta, (n_time, n_cells, 1)) + np.random.normal(0, 1, (n_time, n_cells, n_vert))

    qv = np.linspace(0.015, 0.0001, n_vert)
    qv_3d = np.tile(qv, (n_time, n_cells, 1)) * np.random.uniform(0.8, 1.2, (n_time, n_cells, n_vert))
    qv_3d = np.clip(qv_3d, 0, None)

    u_wind = np.random.uniform(-20, 20, (n_time, n_cells, n_vert))
    v_wind = np.random.uniform(-20, 20, (n_time, n_cells, n_vert))

    ds = xr.Dataset({
        'pressure': (['Time', 'nCells', 'nVertLevels'], pressure),
        'theta': (['Time', 'nCells', 'nVertLevels'], theta_3d),
        'qv': (['Time', 'nCells', 'nVertLevels'], qv_3d),
        'uReconstructZonal': (['Time', 'nCells', 'nVertLevels'], u_wind),
        'uReconstructMeridional': (['Time', 'nCells', 'nVertLevels'], v_wind),
    })

    return ds


@pytest.fixture
def mock_processor(synthetic_3d_dataset: xr.Dataset, 
                   tmp_path: Path) -> MPAS3DProcessor:
    """
    This fixture creates a mock MPAS3DProcessor instance that includes a synthetic 3D dataset and a grid file. The grid file is created as a tiny NetCDF file with longitude and latitude coordinates for 50 cells, which are randomly generated within typical ranges. The mock processor has the dataset and grid_file attributes set, allowing it to be used in tests that require an MPAS3DProcessor instance without needing to load actual data from disk. This approach enables testing of the SoundingDiagnostics class in isolation with controlled inputs. 

    Parameters:
        synthetic_3d_dataset (xr.Dataset): A synthetic 3D MPAS dataset.
        tmp_path (pathlib.Path): A temporary directory for creating the grid file.

    Returns:
        MPAS3DProcessor: A mock MPAS3DProcessor instance.
    """
    n_cells = 50
    np.random.seed(42)
    lon = np.linspace(-110, -90, n_cells)
    lat = np.linspace(25, 45, n_cells)

    grid_ds = xr.Dataset({
        'lonCell': (['nCells'], np.radians(lon)),
        'latCell': (['nCells'], np.radians(lat)),
    })

    grid_path = tmp_path / 'grid.nc'
    grid_ds.to_netcdf(str(grid_path))

    proc = Mock(spec=MPAS3DProcessor)
    proc.dataset = synthetic_3d_dataset
    proc.grid_file = str(grid_path)
    return proc


class TestThermodynamicIndices:
    """ Tests for the compute_thermodynamic_indices method of the SoundingDiagnostics class. """

    def test_returns_dict_keys(self: 'TestThermodynamicIndices', 
                               diag: 'SoundingDiagnostics') -> None:
        """
        This test verifies that the compute_thermodynamic_indices method returns a dictionary containing all the expected keys for various thermodynamic indices. A small array of pressure, temperature, and dewpoint values is created, and the method is called to compute the indices. The test asserts that the output is a dictionary and that it contains all the expected keys, such as 'cape', 'cin', 'lifted_index', 'k_index', and others. This ensures that the method is producing a comprehensive set of diagnostics that can be used for further analysis or visualization. 

        Parameters:
            diag (SoundingDiagnostics): An instance of the SoundingDiagnostics class.

        Returns:
            None
        """
        p = np.linspace(1000, 200, 30)
        t = np.linspace(25, -50, 30)
        td = t - 5  

        result = diag.compute_thermodynamic_indices(p, t, td)
        assert isinstance(result, dict)

        expected_keys = (
            'cape', 'cin', 'sbcape', 'sbcin', 'mlcape', 'mlcin',
            'mucape', 'mucin', 'lifted_index', 'dcape',
            'lcl_pressure', 'lcl_temperature', 'lfc_pressure', 'el_pressure',
            'k_index', 'total_totals', 'showalter_index', 'cross_totals',
            'precipitable_water', 'wet_bulb_zero_height',
            'bulk_shear_0_1km', 'bulk_shear_0_6km',
            'srh_0_1km', 'srh_0_3km', 'stp', 'scp', 'sweat_index',
        )

        for key in expected_keys:
            assert key in result

    def test_lcl_computed(self: 'TestThermodynamicIndices', 
                          diag: 'SoundingDiagnostics') -> None:
        """
        This test checks that the compute_thermodynamic_indices method successfully computes the LCL pressure and that it is a positive value. A small array of pressure, temperature, and dewpoint values is created, and the method is called to compute the indices. The test asserts that the 'lcl_pressure' key in the result is not None and that its value is greater than zero, confirming that the method can compute the LCL pressure correctly based on the input profiles. This is important because the LCL is a fundamental parameter in many thermodynamic analyses and serves as a basis for other indices. 

        Parameters:
            diag (SoundingDiagnostics): An instance of the SoundingDiagnostics class.

        Returns:
            None
        """
        p = np.linspace(1000, 200, 30)
        t = np.linspace(25, -50, 30)
        td = t - 10
        result = diag.compute_thermodynamic_indices(p, t, td)
        assert result['lcl_pressure'] is not None
        assert result['lcl_pressure'] > 0

    def test_wind_indices_none_without_wind(self: 'TestThermodynamicIndices', 
                                            diag: 'SoundingDiagnostics') -> None:
        """
        This test verifies that the compute_thermodynamic_indices method returns None for wind-related indices when no wind information is available. A small array of pressure, temperature, and dewpoint values is created, and the method is called to compute the indices. The test asserts that the indices related to bulk shear, storm-relative helicity (SRH), significant tornado parameter (STP), supercell composite parameter (SCP), and SWEAT index are all None, confirming that the method correctly handles cases where wind data is not provided and does not attempt to compute indices that require wind information. This ensures that the method can gracefully handle incomplete input profiles without producing erroneous results. 

        Parameters:
            diag (SoundingDiagnostics): An instance of the SoundingDiagnostics class.

        Returns:
            None
        """
        p = np.linspace(1000, 200, 30)
        t = np.linspace(25, -50, 30)
        td = t - 5
        result = diag.compute_thermodynamic_indices(p, t, td)
        for key in ('bulk_shear_0_1km', 'bulk_shear_0_6km',
                     'srh_0_1km', 'srh_0_3km', 'stp', 'scp', 'sweat_index'):
            assert result[key] is None


class TestExtractSoundingProfile:
    """ Tests for the extract_sounding_profile method of the SoundingDiagnostics class. """

    def test_basic_extraction(self: 'TestExtractSoundingProfile', 
                              diag: 'SoundingDiagnostics', 
                              mock_processor: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the extract_sounding_profile method successfully extracts a sounding profile containing pressure, temperature, and dewpoint data for a given location. The method is called with a mock processor that includes a synthetic dataset, and the test asserts that the returned profile contains the expected keys and that the lengths of the pressure, temperature, and dewpoint arrays are consistent. This confirms that the method can extract complete profiles from the dataset based on spatial coordinates and that it correctly computes the necessary variables for thermodynamic analysis. 

        Parameters:
            diag (SoundingDiagnostics): An instance of the SoundingDiagnostics class.
            mock_processor: A mock processor object to simulate data extraction.

        Returns:
            None
        """
        profile = diag.extract_sounding_profile(mock_processor, -100.0, 35.0, time_index=0)
        assert 'pressure' in profile
        assert 'temperature' in profile
        assert 'dewpoint' in profile
        assert len(profile['pressure']) > 0
        assert len(profile['temperature']) == len(profile['pressure'])
        assert len(profile['dewpoint']) == len(profile['pressure'])

    def test_pressure_sorted_descending(self: 'TestExtractSoundingProfile', 
                                        diag: 'SoundingDiagnostics', 
                                        mock_processor: 'MPAS3DProcessor') -> None:
        """
        This test checks that the pressure levels in the extracted sounding profile are sorted in descending order, which is a standard convention for atmospheric profiles. The extract_sounding_profile method is called with a mock processor and specific coordinates, and the test asserts that the first pressure value (surface) is greater than the last pressure value (upper level). This ensures that the method correctly organizes the profile data in a way that is consistent with typical atmospheric sounding conventions, which is important for accurate interpretation and analysis of the profile. 

        Parameters:
            diag (SoundingDiagnostics): An instance of the SoundingDiagnostics class.
            mock_processor: A mock processor object to simulate data extraction.

        Returns:
            None
        """
        profile = diag.extract_sounding_profile(mock_processor, -100.0, 35.0)
        p = profile['pressure']
        assert p[0] > p[-1]

    def test_station_metadata(self: 'TestExtractSoundingProfile', 
                              diag: 'SoundingDiagnostics', 
                              mock_processor: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the extract_sounding_profile method correctly includes station metadata in the returned profile. The method is called with a mock processor and specific coordinates, and the test asserts that the profile contains the expected keys for station longitude, latitude, and cell index. It also checks that the cell index is an integer, ensuring that the method provides accurate and properly typed metadata for the sounding profile. 

        Parameters:
            diag (SoundingDiagnostics): An instance of the SoundingDiagnostics class.
            mock_processor: A mock processor object to simulate data extraction.

        Returns:
            None
        """
        profile = diag.extract_sounding_profile(mock_processor, -100.0, 35.0)
        assert isinstance(profile['cell_index'], int)
        assert 'station_lon' in profile
        assert 'station_lat' in profile
        assert 'cell_index' in profile

    def test_wind_components_present(self: 'TestExtractSoundingProfile', 
                                     diag: 'SoundingDiagnostics', 
                                     mock_processor: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the extract_sounding_profile method correctly includes wind components in the returned profile. The method is called with a mock processor and specific coordinates, and the test asserts that the profile contains the expected keys for u and v wind components. It also checks that the lengths of the wind component arrays match the length of the pressure array, ensuring that the method provides accurate and properly sized wind data. This is important for confirming that the method can extract complete profiles that include both thermodynamic and kinematic information when available. 

        Parameters:
            diag (SoundingDiagnostics): An instance of the SoundingDiagnostics class.
            mock_processor: A mock processor object to simulate data extraction.

        Returns:
            None
        """
        profile = diag.extract_sounding_profile(mock_processor, -100.0, 35.0)
        assert profile['u_wind'] is not None
        assert profile['v_wind'] is not None
        assert len(profile['u_wind']) == len(profile['pressure'])
        assert len(profile['v_wind']) == len(profile['pressure'])

    def test_wind_in_knots(self: 'TestExtractSoundingProfile', 
                            diag: 'SoundingDiagnostics', 
                            mock_processor: 'MPAS3DProcessor') -> None:
        """
        This test checks that the wind components in the extracted sounding profile are converted to knots and that their magnitudes are within a reasonable range. The extract_sounding_profile method is called with a mock processor and specific coordinates, and the test asserts that the maximum absolute value of the u wind component is greater than 1 knot (to ensure it's non-trivial) and less than 100 knots (to ensure it's not impossibly large). This helps confirm that the method is correctly converting wind speeds to knots and that the synthetic data used in the test is producing realistic wind values. 

        Parameters:
            diag (SoundingDiagnostics): An instance of the SoundingDiagnostics class.
            mock_processor: A mock processor object to simulate data extraction.

        Returns:
            None
        """
        profile = diag.extract_sounding_profile(mock_processor, -100.0, 35.0)
        u_max = np.nanmax(np.abs(profile['u_wind']))
        assert u_max > 1.0  
        assert u_max < 100 

    def test_different_time_indices(self: 'TestExtractSoundingProfile', 
                                    diag: 'SoundingDiagnostics', 
                                    mock_processor: 'MPAS3DProcessor') -> None:
        """
        This test verifies that the extract_sounding_profile method can extract profiles for different time indices without errors and that the lengths of the extracted profiles are consistent. The method is called twice with the same coordinates but different time indices, and the test asserts that the lengths of the temperature and pressure arrays in both profiles are the same. This ensures that the method can handle temporal variations in the dataset and that it consistently extracts profiles with the correct dimensions regardless of the time index specified. 

        Parameters:
            diag (SoundingDiagnostics): An instance of the SoundingDiagnostics class.
            mock_processor: A mock processor object to simulate data extraction.

        Returns:
            None
        """
        p0 = diag.extract_sounding_profile(mock_processor, -100.0, 35.0, time_index=0)
        p1 = diag.extract_sounding_profile(mock_processor, -100.0, 35.0, time_index=1)
        assert len(p0['temperature']) == len(p1['temperature'])
        assert len(p0['pressure']) == len(p1['pressure'])

    def test_verbose_output(self: 'TestExtractSoundingProfile', 
                            diag_verbose: 'SoundingDiagnostics', 
                            mock_processor: 'MPAS3DProcessor', 
                            capsys) -> None:
        """
        This test verifies that the extract_sounding_profile method produces verbose output when the diagnostics instance is initialized with verbose mode enabled. The method is called with a mock processor and specific coordinates, and the test captures the standard output using the capsys fixture. The test asserts that the captured output contains specific strings indicating that the nearest cell was found and that the sounding profile was extracted, confirming that the verbose mode is functioning correctly and providing useful information during execution. 

        Parameters:
            diag_verbose (SoundingDiagnostics): An instance of the SoundingDiagnostics class with verbose output enabled.
            mock_processor: A mock processor object to simulate data extraction.
            capsys: A pytest fixture to capture stdout and stderr.

        Returns:
            None
        """
        diag_verbose.extract_sounding_profile(mock_processor, -100.0, 35.0)
        captured = capsys.readouterr()
        assert "Nearest cell" in captured.out
        assert "Sounding profile" in captured.out


class TestEdgeCases:
    """ Tests for edge cases in the extract_sounding_profile method of the SoundingDiagnostics class. """

    def test_missing_moisture_variable(self: 'TestEdgeCases', 
                                       diag: 'SoundingDiagnostics', 
                                       tmp_path: 'Path', 
                                       mock_processor: 'MPAS3DProcessor') -> None:
        """
        This test checks the behavior of the extract_sounding_profile method when the dataset does not contain any moisture variable (e.g., specific humidity). A synthetic dataset is created with pressure and potential temperature variables but without any moisture variable. The method is called with a mock processor that includes this dataset, and the test asserts that the dewpoint values in the extracted profile are all NaN, confirming that the method correctly handles cases where moisture information is unavailable and does not produce erroneous dewpoint values. This ensures that the method can gracefully handle incomplete datasets without crashing or providing misleading results. 

        Parameters:
            diag (SoundingDiagnostics): An instance of the SoundingDiagnostics class.
            tmp_path: A pytest fixture providing a temporary directory for file operations.

        Returns:
            None
        """
        n_cells, n_vert = 10, 5

        ds = xr.Dataset({
            'pressure': (['Time', 'nCells', 'nVertLevels'],
                         np.linspace(101000, 5000, n_vert).reshape(1, 1, n_vert).repeat(n_cells, axis=1)),
            'theta': (['Time', 'nCells', 'nVertLevels'],
                      np.full((1, n_cells, n_vert), 300.0)),
            'uReconstructZonal': (['Time', 'nCells', 'nVertLevels'],
                                   np.zeros((1, n_cells, n_vert))),
            'uReconstructMeridional': (['Time', 'nCells', 'nVertLevels'],
                                       np.zeros((1, n_cells, n_vert))),
        })

        lon = np.linspace(-10, 10, n_cells)
        lat = np.linspace(-10, 10, n_cells)

        grid_ds = xr.Dataset({
            'lonCell': (['nCells'], np.radians(lon)),
            'latCell': (['nCells'], np.radians(lat)),
        })

        grid_path = tmp_path / 'grid_no_moisture.nc'
        grid_ds.to_netcdf(str(grid_path))

        proc = Mock(spec=MPAS3DProcessor)
        proc.dataset = ds
        proc.grid_file = str(grid_path)

        profile = diag.extract_sounding_profile(proc, 0.0, 0.0)
        assert np.all(np.isnan(profile['dewpoint']))

    def test_missing_wind_variables(self: 'TestEdgeCases', 
                                    diag: 'SoundingDiagnostics', 
                                    tmp_path: 'Path', 
                                    mock_processor: 'MPAS3DProcessor') -> None:
        """
        This test checks the behavior of the extract_sounding_profile method when the dataset does not contain any wind variables (e.g., u and v components). A synthetic dataset is created with pressure and potential temperature variables but without any wind variables. The method is called with a mock processor that includes this dataset, and the test asserts that the u_wind and v_wind values in the extracted profile are None, confirming that the method correctly handles cases where wind information is unavailable and does not produce erroneous wind values. This ensures that the method can gracefully handle incomplete datasets without crashing or providing misleading results. 

        Parameters:
            diag (SoundingDiagnostics): An instance of the SoundingDiagnostics class.
            tmp_path: A pytest fixture providing a temporary directory for file operations.

        Returns:
            None
        """
        n_cells, n_vert = 10, 5

        ds = xr.Dataset({
            'pressure': (['Time', 'nCells', 'nVertLevels'],
                         np.linspace(101000, 5000, n_vert).reshape(1, 1, n_vert).repeat(n_cells, axis=1)),
            'theta': (['Time', 'nCells', 'nVertLevels'],
                      np.full((1, n_cells, n_vert), 300.0)),
        })

        lon = np.linspace(-10, 10, n_cells)
        lat = np.linspace(-10, 10, n_cells)

        grid_ds = xr.Dataset({
            'lonCell': (['nCells'], np.radians(lon)),
            'latCell': (['nCells'], np.radians(lat)),
        })

        grid_path = tmp_path / 'grid_no_wind.nc'
        grid_ds.to_netcdf(str(grid_path))

        proc = Mock(spec=MPAS3DProcessor)
        proc.dataset = ds
        proc.grid_file = str(grid_path)

        profile = diag.extract_sounding_profile(proc, 0.0, 0.0)

        assert profile['u_wind'] is None
        assert profile['v_wind'] is None


if __name__ == "__main__":
    pytest.main([__file__])

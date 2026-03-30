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
def mock_processor(synthetic_3d_dataset: xr.Dataset, tmp_path: Path) -> MPAS3DProcessor:
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


class TestFindNearestCell:
    """ Tests for the _find_nearest_cell method of the SoundingDiagnostics class. """

    def test_exact_match(self: "TestFindNearestCell", diag: SoundingDiagnostics) -> None:
        """
        This test checks that the _find_nearest_cell method correctly identifies the index of a cell when the target longitude and latitude exactly match one of the cell coordinates. A small array of longitude and latitude values is created, and the method is called with a target location that matches one of the cells. The test asserts that the returned index corresponds to the correct cell, confirming that the method can handle exact matches accurately. 

        Parameters:
            diag (SoundingDiagnostics): An instance of the SoundingDiagnostics class.

        Returns:
            None
        """
        lon = np.array([0.0, 10.0, 20.0])
        lat = np.array([0.0, 10.0, 20.0])
        idx = SoundingDiagnostics._find_nearest_cell(lon, lat, 10.0, 10.0)
        assert idx == 1

    def test_closest_cell_returned(self: "TestFindNearestCell", diag: SoundingDiagnostics, sample_grid_coords: tuple[np.ndarray, np.ndarray]) -> None:
        """
        This test verifies that the _find_nearest_cell method returns a valid index for a target location that does not exactly match any cell coordinates. Using the sample grid coordinates provided by the fixture, the test calls the method with a target location at (0.0, 0.0) and asserts that the returned index is within the valid range of cell indices. This test ensures that the method can find the nearest cell even when there is no exact match, which is a common scenario in real-world applications where target locations may not align perfectly with grid points. 

        Parameters:
            diag (SoundingDiagnostics): An instance of the SoundingDiagnostics class.
            sample_grid_coords (tuple[np.ndarray, np.ndarray]): A tuple containing arrays of longitude and latitude coordinates for the grid cells.

        Returns:
            None
        """
        lon, lat = sample_grid_coords
        idx = SoundingDiagnostics._find_nearest_cell(lon, lat, 0.0, 0.0)
        assert 0 <= idx < len(lon)

    def test_different_targets(self: "TestFindNearestCell", sample_grid_coords: tuple[np.ndarray, np.ndarray]) -> None:
        """
        This test checks that the _find_nearest_cell method can handle different target locations and still return valid indices. Using the sample grid coordinates, the test calls the method with two different target locations: one in the western hemisphere and one in the eastern hemisphere. The test asserts that the returned indices are valid integers within the range of cell indices, confirming that the method can find nearest cells for a variety of target locations across the globe. 

        Parameters:
            sample_grid_coords (tuple[np.ndarray, np.ndarray]): A tuple containing arrays of longitude and latitude coordinates for the grid cells.

        Returns:
            None
        """
        lon, lat = sample_grid_coords
        idx1 = SoundingDiagnostics._find_nearest_cell(lon, lat, -100.0, 40.0)
        idx2 = SoundingDiagnostics._find_nearest_cell(lon, lat, 100.0, -40.0)
        assert isinstance(idx1, int)
        assert isinstance(idx2, int)


class TestDewpointComputation:
    """ Tests for the compute_dewpoint_from_mixing_ratio method of the SoundingDiagnostics class. """

    def test_positive_mixing_ratio(self: "TestDewpointComputation", diag: SoundingDiagnostics) -> None:
        """
        This test verifies that the compute_dewpoint_from_mixing_ratio method returns a finite dewpoint temperature when given a positive mixing ratio. A small array of mixing ratio values and corresponding pressure levels is created, and the method is called to compute the dewpoint temperature. The test asserts that the output has the correct shape and that all dewpoint temperatures are finite. Additionally, it checks that the dewpoint temperature is higher for higher mixing ratios, which is consistent with physical expectations. 

        Parameters:
            diag (SoundingDiagnostics): An instance of the SoundingDiagnostics class.

        Returns:
            None
        """
        qv = np.array([0.010, 0.005, 0.001])
        p = np.array([100000.0, 80000.0, 50000.0])
        td = diag.compute_dewpoint_from_mixing_ratio(qv, p)
        assert td.shape == qv.shape
        assert np.all(np.isfinite(td))
        assert td[0] > td[2]  

    def test_zero_mixing_ratio(self: "TestDewpointComputation", diag: SoundingDiagnostics) -> None:
        """
        This test checks that the compute_dewpoint_from_mixing_ratio method returns a finite dewpoint temperature when given a zero mixing ratio, which represents an extremely dry atmosphere. A small array with a zero mixing ratio and a typical pressure level is created, and the method is called to compute the dewpoint temperature. The test asserts that the output has the correct shape and that the dewpoint temperature is finite, even though it should be very low due to the lack of moisture. This ensures that the method can handle edge cases without producing non-physical results. 

        Parameters:
            diag (SoundingDiagnostics): An instance of the SoundingDiagnostics class.

        Returns:
            None
        """
        qv = np.array([0.0])
        p = np.array([100000.0])
        td = diag.compute_dewpoint_from_mixing_ratio(qv, p)
        assert td.shape == (1,)
        assert np.isfinite(td[0])

    def test_output_in_celsius_range(self: "TestDewpointComputation", diag: SoundingDiagnostics) -> None:
        """
        This test ensures that the compute_dewpoint_from_mixing_ratio method returns dewpoint temperatures that are within a realistic range in degrees Celsius. A small array with a typical mixing ratio and pressure level is created, and the method is called to compute the dewpoint temperature. The test asserts that the computed dewpoint temperature falls within a reasonable range (e.g., -80°C to 40°C), which is consistent with physical expectations for atmospheric conditions. This test helps confirm that the method is producing physically meaningful results. 

        Parameters:
            diag (SoundingDiagnostics): An instance of the SoundingDiagnostics class.

        Returns:
            None
        """
        qv = np.array([0.020])
        p = np.array([101325.0])
        td = diag.compute_dewpoint_from_mixing_ratio(qv, p)
        assert -80 < td[0] < 40


class TestPotentialToActual:
    """ Tests for the potential_to_actual_temperature method of the SoundingDiagnostics class. """

    def test_surface_conversion(self: "TestPotentialToActual") -> None:
        """
        This test verifies that the potential_to_actual_temperature method correctly converts potential temperature to actual temperature at the reference pressure level (1000 hPa). A small array with a potential temperature value is created, and the method is called with a pressure level of 100000 Pa (1000 hPa). The test asserts that the computed actual temperature is approximately equal to the input potential temperature, confirming that the method correctly handles the conversion at the reference level where no change in temperature should occur. 
        
        Parameters:
            None

        Returns:
            None
        """
        theta = np.array([300.0])
        p = np.array([100000.0])
        T = SoundingDiagnostics.potential_to_actual_temperature(theta, p)
        np.testing.assert_allclose(T, theta, atol=0.01)

    def test_upper_level_colder(self: "TestPotentialToActual") -> None:
        """
        This test checks that the potential_to_actual_temperature method produces a lower actual temperature at a higher altitude (lower pressure) compared to the surface. A small array with the same potential temperature value is created, and the method is called with two different pressure levels: one representing the surface (1000 hPa) and one representing an upper level (500 hPa). The test asserts that the computed actual temperature at 500 hPa is lower than the actual temperature at 1000 hPa, which is consistent with the physical expectation that temperature decreases with height in the atmosphere. 

        Parameters:
            None

        Returns:
            None
        """
        theta = np.array([300.0, 300.0])
        p = np.array([100000.0, 50000.0])
        T = SoundingDiagnostics.potential_to_actual_temperature(theta, p)
        assert T[1] < T[0]

    def test_array_shapes_preserved(self: "TestPotentialToActual") -> None:
        """
        This test ensures that the potential_to_actual_temperature method preserves the shape of the input arrays when performing the conversion. A 2D array of potential temperature values and a corresponding 2D array of pressure levels are created, and the method is called to compute the actual temperature. The test asserts that the output actual temperature has the same shape as the input potential temperature, confirming that the method can handle multi-dimensional inputs without altering their structure. This is important for ensuring compatibility with larger datasets where multiple profiles may be processed simultaneously. 

        Parameters:
            None

        Returns:
            None
        """
        theta = np.ones((5, 3)) * 300.0
        p = np.ones((5, 3)) * 80000.0
        T = SoundingDiagnostics.potential_to_actual_temperature(theta, p)
        assert T.shape == (5, 3)


class TestThermodynamicIndices:
    """ Tests for the compute_thermodynamic_indices method of the SoundingDiagnostics class. """

    def test_returns_dict_keys(self: "TestThermodynamicIndices", diag: SoundingDiagnostics) -> None:
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

    def test_lcl_computed(self: "TestThermodynamicIndices", diag: SoundingDiagnostics) -> None:
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

    def test_wind_indices_none_without_wind(self: "TestThermodynamicIndices", diag: SoundingDiagnostics) -> None:
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

    def test_basic_extraction(self: "TestExtractSoundingProfile", diag: SoundingDiagnostics, mock_processor: MPAS3DProcessor) -> None:
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

    def test_pressure_sorted_descending(self: "TestExtractSoundingProfile", diag: SoundingDiagnostics, mock_processor: MPAS3DProcessor) -> None:
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

    def test_station_metadata(self: "TestExtractSoundingProfile", diag: SoundingDiagnostics, mock_processor: MPAS3DProcessor) -> None:
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

    def test_wind_components_present(self: "TestExtractSoundingProfile", diag: SoundingDiagnostics, mock_processor: MPAS3DProcessor) -> None:
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

    def test_wind_in_knots(self: "TestExtractSoundingProfile", diag: SoundingDiagnostics, mock_processor: MPAS3DProcessor) -> None:
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

    def test_different_time_indices(self: "TestExtractSoundingProfile", diag: SoundingDiagnostics, mock_processor: MPAS3DProcessor) -> None:
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

    def test_dataset_not_loaded_raises(self: "TestExtractSoundingProfile", diag: SoundingDiagnostics) -> None:
        """
        This test checks that the extract_sounding_profile method raises a ValueError when the dataset is not loaded in the processor. A mock processor is created with its dataset attribute set to None, and the method is called with specific coordinates. The test asserts that a ValueError is raised with a message indicating that the dataset is not loaded, confirming that the method correctly handles cases where the necessary data is unavailable and provides informative error messages to guide users in troubleshooting. 

        Parameters:
            diag (SoundingDiagnostics): An instance of the SoundingDiagnostics class.

        Returns:
            None
        """
        proc = Mock(spec=MPAS3DProcessor)
        proc.dataset = None
        with pytest.raises(ValueError, match="not loaded"):
            diag.extract_sounding_profile(proc, 0.0, 0.0)

    def test_verbose_output(self: "TestExtractSoundingProfile", diag_verbose: SoundingDiagnostics, mock_processor: MPAS3DProcessor, capsys) -> None:
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

    def test_missing_moisture_variable(self: "TestEdgeCases", diag: SoundingDiagnostics, tmp_path: Path, mock_processor: MPAS3DProcessor) -> None:
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

    def test_missing_wind_variables(self: "TestEdgeCases", diag: SoundingDiagnostics, tmp_path: Path, mock_processor: MPAS3DProcessor) -> None:
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
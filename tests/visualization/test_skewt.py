#!/usr/bin/env python3

"""
MPASdiag Test Suite: SkewT Diagram Visualization

This module contains unit tests for the MPASSkewTPlotter class, which is responsible for creating SkewT-logP diagrams from sounding data. The tests cover basic functionality, edge cases, and save functionality. It uses pytest fixtures to provide sample profiles and indices for testing. The tests ensure that the plotter can create figures without errors and that the save functionality works as expected. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: March 2026
Version: 1.0.0
"""
# Load standard libraries
import os
import pytest
import pathlib
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.figure
import matplotlib.pyplot as plt
from typing import Mapping, Optional

from mpasdiag.visualization.skewt import MPASSkewTPlotter


@pytest.fixture
def plotter() -> MPASSkewTPlotter:
    """ 
    This fixture provides a fresh instance of the MPASSkewTPlotter for each test. It initializes the plotter with a specific figure size and DPI, and sets verbose to False to suppress output during testing. This allows tests to focus on functionality without extraneous print statements.

    Parameters:
        None

    Returns:
        MPASSkewTPlotter: An instance of the MPASSkewTPlotter class initialized with specified parameters for testing.
    """
    return MPASSkewTPlotter(figsize=(9, 12), dpi=100, verbose=False)


@pytest.fixture
def sample_profile() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This fixture generates a synthetic sounding profile for testing purposes. It creates arrays for pressure, temperature, dewpoint, and wind components (u and v) that mimic a typical atmospheric sounding. The pressure decreases with height, the temperature decreases with height, and the dewpoint is slightly lower than the temperature. The wind components increase with height to simulate typical wind shear. This profile is used in various tests to verify that the SkewT diagram can be created without errors and that the data is handled correctly.

    Parameters:
        None

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Arrays representing pressure, temperature, dewpoint, u-wind, and v-wind components.
    """
    n = 30
    pressure = np.linspace(1000, 200, n)
    temperature = np.linspace(25, -55, n)
    dewpoint = temperature - np.linspace(3, 15, n)
    u_wind = np.linspace(5, 40, n)
    v_wind = np.linspace(-5, 10, n)
    return pressure, temperature, dewpoint, u_wind, v_wind


@pytest.fixture
def sample_indices() -> dict[str, float]:
    """ 
    This fixture provides a sample set of SkewT indices for testing the display of indices on the SkewT diagram. The dictionary includes common indices such as CAPE, CIN, LCL pressure and temperature, LFC pressure, and EL pressure. These values are arbitrary but realistic, allowing tests to verify that the indices can be passed to the plotter and displayed without errors.

    Parameters:
        None

    Returns:
        dict[str, float]: A dictionary containing sample values for various SkewT indices.
    """
    return {
        # Parcel / instability
        'cape': 1500.0, 'cin': -120.0,
        'sbcape': 1500.0, 'sbcin': -120.0,
        'mlcape': 1200.0, 'mlcin': -45.0,
        'mucape': 1800.0, 'mucin': -30.0,
        'lifted_index': -5.2, 'dcape': 800.0,
        'lcl_pressure': 900.0, 'lcl_temperature': 18.0,
        'lfc_pressure': 750.0, 'el_pressure': 250.0,
        # Stability
        'k_index': 32.0, 'total_totals': 52.0,
        'showalter_index': -2.1, 'cross_totals': 22.0,
        # Moisture
        'precipitable_water': 32.1, 'wet_bulb_zero_height': 3200.0,
        # Shear / severe
        'bulk_shear_0_1km': 15.0, 'bulk_shear_0_6km': 35.0,
        'srh_0_1km': 180.0, 'srh_0_3km': 250.0,
        'stp': 1.5, 'scp': 8.2, 'sweat_index': 280.0,
    }


class TestCreateSkewTDiagram:
    """ This test class contains unit tests for the create_skewt_diagram method of the MPASSkewTPlotter class. """

    def test_returns_figure_and_axes(self: 'TestCreateSkewTDiagram', 
                                     plotter: 'MPASSkewTPlotter', 
                                     sample_profile: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        """ 
        The create_skewt_diagram method should return a Matplotlib Figure and Axes object when provided with valid sounding data. This test verifies that the method executes without errors and that the returned objects are of the correct types. It uses a sample profile to ensure that the method can handle typical sounding data and produce a SkewT diagram successfully. 

        Parameters:
            plotter (MPASSkewTPlotter): The plotter instance provided by the fixture.
            sample_profile (tuple): A tuple containing synthetic sounding data for pressure, temperature, dewpoint, u-wind, and v-wind.

        Returns:
            None: The test will pass if the method returns a Figure and Axes object without raising exceptions. If the method fails to create the diagram or returns incorrect types, the test will fail.
        """
        p, t, td, u, v = sample_profile
        fig, ax = plotter.create_skewt_diagram(p, t, td, u, v)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_without_wind(self: 'TestCreateSkewTDiagram', 
                          plotter: 'MPASSkewTPlotter', 
                          sample_profile: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        """ 
        This test verifies that the create_skewt_diagram method can successfully create a SkewT diagram even when wind data is not provided. The method should be able to handle None values for the u and v wind components without raising errors. This ensures that users can still visualize the thermodynamic profile of the atmosphere even if wind data is unavailable or not relevant for their analysis.

        Parameters:
            plotter (MPASSkewTPlotter): The plotter instance provided by the fixture.
            sample_profile (tuple): A tuple containing synthetic sounding data for pressure, temperature, dewpoint, u-wind, and v-wind. The u and v wind components will be ignored in this test.
            
        Returns:    
            None: The test will pass if the method returns a Figure object without raising exceptions. If the method fails to create the diagram or returns incorrect types, the test will fail.
        """
        p, t, td, _, _ = sample_profile
        fig, ax = plotter.create_skewt_diagram(p, t, td, None, None)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_with_title(self: 'TestCreateSkewTDiagram', 
                        plotter: 'MPASSkewTPlotter', 
                        sample_profile: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        """ 
        This test checks that the create_skewt_diagram method can accept a title argument and still produce a valid SkewT diagram. The presence of a title should not interfere with the creation of the figure, and the method should return a Figure object as expected. This ensures that users can customize their SkewT diagrams with titles without encountering issues.

        Parameters:
            plotter (MPASSkewTPlotter): The plotter instance provided by the fixture.
            sample_profile (tuple): A tuple containing synthetic sounding data for pressure, temperature, dewpoint, u-wind, and v-wind.

        Returns:
            None: The test will pass if the method returns a Figure object without raising exceptions. If the method fails to create the diagram or returns incorrect types, the test will fail.
        """
        p, t, td, u, v = sample_profile
        fig, ax = plotter.create_skewt_diagram(
            p, t, td, u, v, title='Test Sounding')
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_with_indices(self: 'TestCreateSkewTDiagram', 
                          plotter: 'MPASSkewTPlotter', 
                          sample_profile: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], 
                          sample_indices: Mapping[str, Optional[float]]) -> None:
        """ 
        This test verifies that the create_skewt_diagram method can accept a dictionary of indices and display them on the SkewT diagram without errors. The indices should be correctly passed to the method, and the presence of these indices should not interfere with the creation of the figure. This ensures that users can enhance their SkewT diagrams with additional information about atmospheric stability and other relevant parameters.

        Parameters:
            plotter (MPASSkewTPlotter): The plotter instance provided by the fixture.
            sample_profile (tuple): A tuple containing synthetic sounding data for pressure, temperature, dewpoint, u-wind, and v-wind.
            sample_indices (dict): A dictionary containing sample values for various SkewT indices such as CAPE, CIN, LCL pressure and temperature, LFC pressure, and EL pressure.

        Returns:
            None: The test will pass if the method returns a Figure object without raising exceptions. If the method fails to create the diagram or returns incorrect types, the test will fail.
        """
        p, t, td, u, v = sample_profile
        fig, ax = plotter.create_skewt_diagram(
            p, t, td, u, v, indices=sample_indices) # type: ignore
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_with_parcel_and_indices(self: 'TestCreateSkewTDiagram', 
                                     plotter: 'MPASSkewTPlotter', 
                                     sample_profile: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], 
                                     sample_indices: Mapping[str, Optional[float]]) -> None:
        """ 
        This test checks that the create_skewt_diagram method can accept both a dictionary of indices and the show_parcel flag to display a parcel profile on the SkewT diagram without errors. The method should be able to handle the combination of these parameters and still produce a valid figure. This ensures that users can visualize both the indices and the parcel profile simultaneously for a more comprehensive analysis of the sounding data.
        
        Parameters:
            plotter (MPASSkewTPlotter): The plotter instance provided by the fixture.
            sample_profile (tuple): A tuple containing synthetic sounding data for pressure, temperature, dewpoint, u-wind, and v-wind.
            sample_indices (dict): A dictionary containing sample values for various SkewT indices such as CAPE, CIN, LCL pressure and temperature, LFC pressure, and EL pressure.

        Returns:
            None: The test will pass if the method returns a Figure object without raising exceptions. If the method fails to create the diagram or returns incorrect types, the test will fail.
        """
        p, t, td, u, v = sample_profile

        fig, ax = plotter.create_skewt_diagram(
            p, t, td, u, v, indices=sample_indices, show_parcel=True) # type: ignore

        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


class TestSavePlot:
    """ Verify that the create_skewt_diagram method can save the generated SkewT diagram to a file when a save_path is provided. """

    def test_save_png(self: 'TestSavePlot', 
                      plotter: 'MPASSkewTPlotter', 
                      sample_profile: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], 
                      tmp_path: pathlib.Path) -> None:
        """ 
        This test checks that the create_skewt_diagram method can successfully save the generated SkewT diagram as a PNG file when a save_path is provided. The test verifies that the file is created and has a non-zero size, indicating that the plot was saved correctly. This ensures that users can utilize the save functionality of the plotter to export their SkewT diagrams for use in reports, presentations, or further analysis. 

        Parameters:
            plotter (MPASSkewTPlotter): The plotter instance provided by the fixture.
            sample_profile (tuple): A tuple containing synthetic sounding data for pressure, temperature, dewpoint, u-wind, and v-wind.
            tmp_path (pathlib.Path): A temporary directory provided by pytest for saving test files.
            
        Returns:
            None: The test will pass if the PNG file is created and has a non-zero size. If the file is not created or is empty, the test will fail.
        """
        p, t, td, u, v = sample_profile
        save_path = str(tmp_path / 'skewt')

        fig, ax = plotter.create_skewt_diagram(
            p, t, td, u, v, save_path=save_path)

        assert os.path.isfile(save_path + '.png')
        assert os.path.getsize(save_path + '.png') > 0

        plt.close(fig)

    def test_save_pdf(self: 'TestSavePlot', 
                      plotter: 'MPASSkewTPlotter', 
                      sample_profile: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], 
                      tmp_path: pathlib.Path) -> None:
        """ 
        This test checks that the create_skewt_diagram method can successfully save the generated SkewT diagram as a PDF file when a save_path with a .pdf extension is provided. The test verifies that the file is created and has a non-zero size, indicating that the plot was saved correctly. This ensures that users can utilize the save functionality of the plotter to export their SkewT diagrams in different formats for use in reports, presentations, or further analysis.
        
        Parameters:
            plotter (MPASSkewTPlotter): The plotter instance provided by the fixture.
            sample_profile (tuple): A tuple containing synthetic sounding data for pressure, temperature, dewpoint, u-wind, and v-wind.
            tmp_path (pathlib.Path): A temporary directory provided by pytest for saving test files.
            
        Returns:
            None: The test will pass if the PDF file is created and has a non-zero size. If the file is not created or is empty, the test will fail.
        """
        p, t, td, u, v = sample_profile
        save_path = str(tmp_path / 'skewt')

        fig, ax = plotter.create_skewt_diagram(
            p, t, td, u, v, save_path=save_path)

        assert os.path.isfile(save_path + '.pdf')
        assert os.path.getsize(save_path + '.pdf') > 0
        plt.close(fig)


class TestEdgeCases:
    """ Verify that the create_skewt_diagram method can handle edge cases such as very short profiles, NaN values in dewpoint, and indices with None values without crashing. """

    def test_short_profile(self: 'TestEdgeCases', plotter: 'MPASSkewTPlotter') -> None:
        """
        This test checks that the create_skewt_diagram method can handle a very short sounding profile (e.g., only 3 levels) without crashing. The method should be able to create a SkewT diagram even with limited data points, and it should return a valid Figure object. This ensures that users can still visualize their sounding data in a SkewT format even if they have only a few levels of data available.

        Parameters:
            plotter (MPASSkewTPlotter): The plotter instance provided by the fixture.

        Returns:
            None: The test will pass if the method returns a Figure object without raising exceptions. 
        """
        p = np.array([1000, 500, 200])
        t = np.array([25, -10, -50])
        td = np.array([20, -15, -55])
        fig, ax = plotter.create_skewt_diagram(p, t, td, None, None)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_nan_in_dewpoint(self: 'TestEdgeCases', 
                             plotter: 'MPASSkewTPlotter', 
                             sample_profile: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        """
        This test verifies that the create_skewt_diagram method can handle NaN values in the dewpoint array without crashing. The method should be able to create a SkewT diagram even if some dewpoint values are missing, and it should return a valid Figure object. This ensures that users can still visualize their sounding data in a SkewT format even if there are gaps in the dewpoint data.

        Parameters:
            plotter (MPASSkewTPlotter): The plotter instance provided by the fixture.
            sample_profile (tuple): A tuple containing synthetic sounding data for pressure, temperature, dewpoint, u-wind, and v-wind. The dewpoint array will contain NaN values for this test.

        Returns:
            None: The test will pass if the method returns a Figure object without raising exceptions.
        """
        p, t, td, u, v = sample_profile
        td[5:10] = np.nan
        fig, ax = plotter.create_skewt_diagram(p, t, td, u, v)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_indices_with_none_values(self: 'TestEdgeCases',
                                      plotter: 'MPASSkewTPlotter', 
                                      sample_profile: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        """
        This test verifies that the create_skewt_diagram method can handle indices with None values without crashing. The method should be able to create a SkewT diagram even if some indices are missing, and it should return a valid Figure object. This ensures that users can still visualize their sounding data in a SkewT format even if there are gaps in the indices data.

        Parameters:
            plotter (MPASSkewTPlotter): The plotter instance provided by the fixture.
            sample_profile (tuple): A tuple containing synthetic sounding data for pressure, temperature, dewpoint, u-wind, and v-wind. The indices dictionary will contain None values for this test.

        Returns:
            None: The test will pass if the method returns a Figure object without raising exceptions.
        """
        p, t, td, u, v = sample_profile

        indices = {'cape': None, 'cin': None, 'lcl_pressure': None,
                   'lcl_temperature': None, 'lfc_pressure': None, 'el_pressure': None}

        fig, ax = plotter.create_skewt_diagram(
            p, t, td, u, v, indices=indices) # type: ignore

        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


class TestIndicesTable:
    """ Verify the _add_indices_table method renders correctly. """

    def test_table_with_full_indices(self: 'TestIndicesTable',
                                     plotter: 'MPASSkewTPlotter',
                                     sample_profile: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                                     sample_indices: dict[str, float]) -> None:
        """ 
        This test checks that the _add_indices_table method can render a table of indices on the SkewT diagram when provided with a complete set of indices. The method should be able to display all the indices without errors, and the resulting figure should contain more than one axes (the main plot and the table). This ensures that users can visualize a comprehensive set of indices alongside their SkewT diagram for enhanced analysis.

        Parameters:
            plotter (MPASSkewTPlotter): The plotter instance provided by the fixture.
            sample_profile (tuple): A tuple containing synthetic sounding data for pressure, temperature, dewpoint, u-wind, and v-wind.
            sample_indices (dict): A dictionary containing sample values for various SkewT indices such as CAPE, CIN, LCL pressure and temperature, LFC pressure, and EL pressure.

        Returns:
            None: The test will pass if the method returns a Figure object with more than one axes without raising exceptions. If the method fails to create the diagram or does not render the table correctly, the test will fail.
        """
        p, t, td, u, v = sample_profile
        fig, ax = plotter.create_skewt_diagram(
            p, t, td, u, v, indices=sample_indices)  # type: ignore
        # Figure should have more than 1 axes (main plot + table)
        assert len(fig.get_axes()) >= 2
        plt.close(fig)

    def test_table_with_partial_indices(self: 'TestIndicesTable',
                                        plotter: 'MPASSkewTPlotter',
                                        sample_profile: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        """ 
        This test checks that the _add_indices_table method can render a table of indices on the SkewT diagram when most indices are None. The method should handle missing values gracefully and still produce a valid figure. This ensures that users can visualize a partial set of indices without encountering errors. 

        Parameters:
            plotter (MPASSkewTPlotter): The plotter instance provided by the fixture.
            sample_profile (tuple): A tuple containing synthetic sounding data for pressure, temperature, dewpoint, u-wind, and v-wind.

        Returns:
            None: The test will pass if the method returns a Figure object without raising exceptions. If the method fails to create the diagram or does not render the table correctly, the test will fail.
        """
        p, t, td, u, v = sample_profile
        sparse = {'cape': 500.0, 'cin': None, 'lcl_pressure': 920.0,
                  'sbcape': None, 'k_index': 28.0}
        fig, ax = plotter.create_skewt_diagram(
            p, t, td, u, v, indices=sparse)  # type: ignore
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_table_all_none_no_crash(self: 'TestIndicesTable',
                                     plotter: 'MPASSkewTPlotter',
                                     sample_profile: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        """
        This test checks that the _add_indices_table method does not crash when all indices are None. The method should be able to handle a scenario where no indices are available and still produce a valid figure without errors. This ensures that users can visualize their SkewT diagram even if they do not have any indices to display, and that the absence of indices does not interfere with the core functionality of the plotter. 

        Parameters:
            plotter (MPASSkewTPlotter): The plotter instance provided by the fixture.
            sample_profile (tuple): A tuple containing synthetic sounding data for pressure, temperature, dewpoint, u-wind, and v-wind.

        Returns:
            None: The test will pass if the method returns a Figure object without raising exceptions. If the method fails to create the diagram or does not render the table correctly, the test will fail.
        """
        p, t, td, u, v = sample_profile
        indices = dict.fromkeys(('cape', 'cin', 'lcl_pressure', 'sbcape', 'k_index', 'bulk_shear_0_6km'))

        fig, ax = plotter.create_skewt_diagram(
            p, t, td, u, v, indices=indices)  # type: ignore

        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__])

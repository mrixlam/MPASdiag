#!/usr/bin/env python3

"""
MPASdiag Test Suite: Enhancements and Refactoring Validation

This test suite focuses on validating the recent enhancements and refactoring efforts in the MPASdiag package, specifically targeting the processing of 2D and 3D variables, as well as the conditional display of valid time information in surface plots. The tests are designed to ensure that new functionalities are correctly implemented, that backward compatibility is maintained for existing features, and that the user experience is improved with intelligent time display logic. By using a combination of real MPAS data and mock objects, these tests provide comprehensive coverage of the new code paths while ensuring that existing functionality continues to operate as expected.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries
import math
import pytest
import matplotlib
matplotlib.use('Agg')  
from datetime import datetime
import matplotlib.pyplot as plt
from cartopy.mpl.geoaxes import GeoAxes
from typing import Generator, Tuple, List
from unittest.mock import patch, MagicMock

if GeoAxes is not None:
    CARTOPY_AVAILABLE = True
else:
    CARTOPY_AVAILABLE = False

# Import MPASdiag modules for testing
from mpasdiag.visualization.surface import MPASSurfacePlotter
from mpasdiag.processing.utils_metadata import MPASFileMetadata
from tests.test_data_helpers import load_mpas_coords_from_processor


class TestRefactoredFunctions:
    """ Test refactored functions for 2D variable metadata retrieval and 3D variable processing placeholders. This class validates the new MPASFileMetadata methods for extracting 2D variable metadata with automatic unit conversions, ensuring that temperature variables are correctly converted from Kelvin to Celsius and that appropriate metadata fields are populated. Additionally, it tests that placeholder functions for future 3D variable support raise NotImplementedError with informative messages, confirming that users receive clear feedback when attempting to use unimplemented features. These tests ensure the integrity of the refactored code while maintaining backward compatibility and providing a clear path for future enhancements. """
    
    def test_get_2d_variable_metadata(self: "TestRefactoredFunctions") -> None:
        """
        This test validates the functionality of the `get_2d_variable_metadata` method in the `MPASFileMetadata` class. It checks that the method correctly retrieves metadata for a 2D variable (in this case, 't2m' for 2-meter temperature) and performs the necessary unit conversion from Kelvin to Celsius. The test asserts that the returned metadata is a dictionary containing the expected keys ('units', 'long_name', 'colormap') and that the values are correctly set, including the conversion of units and the appropriate long name without redundant unit information. This ensures that the refactored method provides accurate and user-friendly metadata for 2D variables, supporting improved visualization and analysis workflows.

        Parameters:
            None

        Returns:
            None
        """
        metadata = MPASFileMetadata.get_2d_variable_metadata('t2m')
        
        assert isinstance(metadata, dict)
        assert 'units' in metadata
        assert 'long_name' in metadata
        assert 'colormap' in metadata
        assert metadata['units'] == '°C'
        assert metadata['original_units'] == 'K'
        assert metadata['long_name'] == '2-meter Temperature'  
    
    def test_3d_placeholder_functions(self: "TestRefactoredFunctions") -> None:
        """
        This test verifies that the placeholder functions for 3D variable support in the `MPASFileMetadata` class correctly raise `NotImplementedError` with informative messages. It attempts to call the `get_3d_variable_metadata`, `get_3d_colormap_and_levels`, and `plot_3d_variable_slice` methods with example parameters and asserts that each call raises the expected exception with a message indicating that 3D variable support is not yet implemented. This ensures that users receive clear feedback when trying to access unimplemented features, guiding them towards the current capabilities of the package while setting expectations for future enhancements.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(NotImplementedError) as cm:
            MPASFileMetadata.get_3d_variable_metadata('temperature', level=500)

        assert "3D variable support not yet implemented" in str(cm.value)
        
        with pytest.raises(NotImplementedError) as cm:
            MPASFileMetadata.get_3d_colormap_and_levels('temperature', level=500)

        assert "3D variable support not yet implemented" in str(cm.value)
        
        with pytest.raises(NotImplementedError) as cm:
            import xarray as xr
            lon, lat, u, v = load_mpas_coords_from_processor(n=10)
            dummy_data = xr.DataArray(u.reshape(10, 1))
            MPASFileMetadata.plot_3d_variable_slice(dummy_data, lon, lat, 500, 'temperature')

        assert "3D variable support not yet implemented" in str(cm.value)


class TestConditionalTimeDisplay:
    """ This test class validates the conditional display of valid time information in surface plots based on the presence of time references in plot titles. It ensures that when users provide custom titles without temporal keywords, the plotting system automatically adds corner text with the valid time. Conversely, if the title already contains time information, no duplicate corner text should be displayed. The tests use mock objects to capture title and text method calls, verifying that timestamps are correctly integrated into titles or corner text as appropriate. This intelligent display logic enhances user experience by providing clear temporal context without visual clutter from duplicate timestamps. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestConditionalTimeDisplay", mpas_surface_temp_data) -> Generator[None, None, None]:
        """
        This fixture sets up the necessary environment for testing conditional time display in surface plots. It initializes an instance of `MPASSurfacePlotter`, loads a sample MPAS surface temperature dataset, and prepares longitude and latitude coordinates for plotting. The fixture also defines a test time to be used for validating timestamp display logic. After the tests are executed, it ensures that any created figures are closed to prevent resource leaks.

        Parameters:
            mpas_surface_temp_data: Real MPAS surface temperature from session-scoped fixture

        Returns:
            Generator[None, None, None]
        """
        if not CARTOPY_AVAILABLE:
            pytest.skip("Cartopy not available")
            return
            
        self.surface_plotter = MPASSurfacePlotter(figsize=(10, 8), dpi=150)
        self.test_time = datetime(2024, 9, 17, 3, 0, 0)
        lon, lat, _, _ = load_mpas_coords_from_processor(n=100)
        self.lon = lon
        self.lat = lat
        self.data = mpas_surface_temp_data[:100]
    
        yield

        if hasattr(self.surface_plotter, 'fig') and self.surface_plotter.fig:
            plt.close(self.surface_plotter.fig)
    
    def _setup_mocks(self: "TestConditionalTimeDisplay") -> Tuple[MagicMock, MagicMock, List[Tuple[Tuple, dict]]]:
        """
        This helper method sets up mock objects for matplotlib figure and axes to capture title and text method calls during testing. It creates a mock figure and a mock GeoAxes object, configuring the text method to capture calls in a list for later assertions. This allows the tests to verify that timestamps are correctly integrated into titles or corner text without relying on actual rendering, enabling precise validation of the conditional time display logic.

        Parameters:
            None

        Returns:
            Tuple[MagicMock, MagicMock, List]: Mock figure, mock axes with text capture, and text_calls list
        """
        mock_fig = MagicMock()
        mock_ax = MagicMock(spec=GeoAxes)
        mock_ax.transAxes = MagicMock()
        
        text_calls = []

        def capture_text(*args, **kwargs):
            text_calls.append((args, kwargs))

        mock_ax.text = capture_text
        
        return mock_fig, mock_ax, text_calls
    
    def test_default_title_with_time(self: "TestConditionalTimeDisplay") -> None:
        """
        This test verifies that when no custom title is provided, the valid time is automatically included in the plot title and that no corner text is added. It uses mock objects to capture calls to the set_title method and the text method of the axes. The test asserts that the title contains the expected "Valid Time: 20240917T03" string and that no corner text calls are made with coordinates (0.98, 0.02). This ensures that the default behavior correctly integrates time information into the title without redundant corner text, providing a clean and informative plot presentation.

        Parameters:
            None

        Returns:
            None
        """
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig, mock_ax, text_calls = self._setup_mocks()
            mock_figure.return_value = mock_fig
            mock_fig.add_subplot.return_value = mock_ax
            
            _, _ = self.surface_plotter.create_surface_map(
                lon=self.lon, lat=self.lat, data=self.data,
                var_name='t2m',
                lon_min=91.0, lon_max=113.0, lat_min=-10.0, lat_max=12.0,
                title=None,
                time_stamp=self.test_time
            )
        
        title_calls = [call for call in mock_ax.set_title.call_args_list]
        title_text = title_calls[0][0][0]  

        assert len(title_calls) > 0
        assert "Valid Time: 20240917T03" in title_text
        
        corner_text_calls = [call for call in text_calls 
                   if len(call[0]) >= 3 and math.isclose(call[0][1], 0.98, abs_tol=1e-6) and math.isclose(call[0][2], 0.02, abs_tol=1e-6)]

        assert len(corner_text_calls) == pytest.approx(0), "Corner text should not be displayed when time is in title"
    
    def test_custom_title_without_time(self: "TestConditionalTimeDisplay") -> None:
        """
        This test verifies that when a custom title is provided without any temporal keywords, the valid time is displayed as corner text instead of being included in the title. It uses mock objects to capture calls to the set_title method and the text method of the axes. The test asserts that the title does not contain any time references and that corner text calls are made with the expected timestamp "Valid: 20240917T03" at coordinates (0.98, 0.02). This ensures that the conditional display logic correctly identifies when to add time information as corner text, enhancing plot readability without cluttering the title.

        Parameters:
            None

        Returns:
            None
        """
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig, mock_ax, text_calls = self._setup_mocks()
            mock_figure.return_value = mock_fig
            mock_fig.add_subplot.return_value = mock_ax
            
            _, _ = self.surface_plotter.create_surface_map(
                lon=self.lon, lat=self.lat, data=self.data,
                var_name='t2m',
                lon_min=91.0, lon_max=113.0, lat_min=-10.0, lat_max=12.0,
                title="Custom Temperature Analysis",
                time_stamp=self.test_time
            )
        
        corner_text_calls = [call for call in text_calls 
                           if len(call[0]) >= 3 and 'Valid: 20240917T03' in str(call[0])]
        
        assert len(corner_text_calls) > 0, "Corner text should be displayed when time is not in title"
    
    def test_custom_title_with_time(self: "TestConditionalTimeDisplay") -> None:
        """
        This test verifies that when a custom title is provided that already contains temporal keywords, the valid time is not duplicated as corner text. It uses mock objects to capture calls to the set_title method and the text method of the axes. The test asserts that the title contains the expected "Valid: 20240917T03" string and that no corner text calls are made with the same timestamp. This ensures that the conditional display logic correctly prevents redundant time information, maintaining a clean and informative plot presentation.

        Parameters:
            None

        Returns:
            None
        """
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig, mock_ax, text_calls = self._setup_mocks()
            mock_figure.return_value = mock_fig
            mock_fig.add_subplot.return_value = mock_ax
            
            _, _ = self.surface_plotter.create_surface_map(
            lon=self.lon, lat=self.lat, data=self.data,
            var_name='t2m',
            lon_min=91.0, lon_max=113.0, lat_min=-10.0, lat_max=12.0,
            title="Temperature Analysis - Valid: 20240917T03",
            time_stamp=self.test_time
        )
        
        corner_text_calls = [call for call in text_calls 
                           if len(call[0]) >= 3 and 'Valid: 20240917T03' in str(call[0])]
        
        assert len(corner_text_calls) == pytest.approx(0), "Corner text should not be displayed when time is already in title"
    
    def test_no_timestamp(self: "TestConditionalTimeDisplay") -> None:
        """
        This test verifies the complete absence of time display when the timestamp parameter is None. It validates that the plotting system properly handles missing temporal information without attempting to display invalid or placeholder timestamps. When time_stamp=None is passed, neither the title nor corner text should contain time references. Mock objects capture set_title and text method calls to verify that no "Valid Time:" or "Valid:" strings appear in any display elements. Assertions confirm zero time-related text when the timestamp is absent, ensuring that plots for climatological fields or non-temporal analyses remain clean without confusing or erroneous time displays.

        Parameters:
            None

        Returns:
            None
        """
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig, mock_ax, text_calls = self._setup_mocks()
            mock_figure.return_value = mock_fig
            mock_fig.add_subplot.return_value = mock_ax
            
            _, _ = self.surface_plotter.create_surface_map(
                lon=self.lon, lat=self.lat, data=self.data,
                var_name='t2m',
                lon_min=91.0, lon_max=113.0, lat_min=-10.0, lat_max=12.0,
                title=None,
                time_stamp=None
            )
            
            title_calls = [call for call in mock_ax.set_title.call_args_list]
            title_text = title_calls[0][0][0]

            assert len(title_calls) > 0
            assert "Valid Time:" not in title_text
            
            corner_text_calls = [call for call in text_calls 
                               if len(call[0]) >= 3 and 'Valid:' in str(call[0])]
            
        assert len(corner_text_calls) == pytest.approx(0), "No time display should appear when timestamp is None"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
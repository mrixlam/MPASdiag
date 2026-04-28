#!/usr/bin/env python3

"""
MPASdiag Test Suite: Skew-T Plotter Coverage

This test suite is designed to cover specific lines of the MPASSkewTPlotter class that were not previously tested, ensuring comprehensive coverage of the codebase. The tests focus on the verbose output functionality when saving plots and the exception handling in the parcel profile plotting method. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
import numpy as np
import pytest
from io import StringIO
from unittest.mock import MagicMock, patch

from mpasdiag.visualization.skewt import MPASSkewTPlotter


PRESSURE = np.array([1000.0, 850.0, 700.0, 500.0, 300.0])
TEMPERATURE = np.array([25.0, 15.0, 5.0, -10.0, -35.0])
DEWPOINT = np.array([20.0, 10.0, 0.0, -15.0, -40.0])


class TestCreateSkewTSaveVerbose:
    """ Test if the create_skewt_diagram method correctly prints the saved path when verbose is True, and does not print when verbose is False. """

    def test_verbose_prints_saved_path(self: 'TestCreateSkewTSaveVerbose', tmp_path) -> None:
        """
        This test verifies that when the MPASSkewTPlotter is initialized with verbose=True, the create_skewt_diagram method prints the path where the plot is saved after successfully saving the plot. It mocks the internal methods to avoid actual file I/O and captures the standard output to check for the expected print statement. 

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSkewTPlotter(verbose=True)
        mock_fig = MagicMock()
        mock_ax = MagicMock()

        skewt_path = str(tmp_path / 'test_skewt.png')
        with patch.object(plotter, "_create_metpy_skewt", return_value=(mock_fig, mock_ax)):
            with patch("mpasdiag.visualization.skewt.MPASVisualizationStyle.save_plot"):
                with patch.object(plotter, "add_timestamp_and_branding"):
                    captured = StringIO()
                    with patch("sys.stdout", new=captured):
                        fig, ax = plotter.create_skewt_diagram(
                            PRESSURE, TEMPERATURE, DEWPOINT,
                            save_path=skewt_path,
                        )

        out = captured.getvalue()
        assert "Skew-T diagram saved to:" in out
        assert skewt_path in out
        assert fig is mock_fig
        assert ax is mock_ax

    def test_verbose_false_no_print_after_save(self: 'TestCreateSkewTSaveVerbose', tmp_path) -> None:
        """
        This test checks that when the MPASSkewTPlotter is initialized with verbose=False, the create_skewt_diagram method does not print any output to the console after saving the plot. It mocks the necessary methods to prevent actual file I/O and captures the standard output to confirm that it remains empty.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSkewTPlotter(verbose=False)
        mock_fig = MagicMock()
        mock_ax = MagicMock()

        with patch.object(plotter, "_create_metpy_skewt", return_value=(mock_fig, mock_ax)):
            with patch("mpasdiag.visualization.skewt.MPASVisualizationStyle.save_plot"):
                with patch.object(plotter, "add_timestamp_and_branding"):
                    captured = StringIO()
                    with patch("sys.stdout", new=captured):
                        plotter.create_skewt_diagram(
                            PRESSURE, TEMPERATURE, DEWPOINT,
                            save_path=str(tmp_path / 'test_skewt.png'),
                        )

        assert captured.getvalue() == ""

    def test_no_save_path_skips_save_and_print(self: 'TestCreateSkewTSaveVerbose') -> None:
        """
        This test verifies that when the MPASSkewTPlotter is initialized with verbose=True but no save_path is provided, the create_skewt_diagram method does not attempt to save the plot and does not print any output. It mocks the necessary methods to prevent actual file I/O and captures the standard output to confirm that it remains empty.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSkewTPlotter(verbose=True)
        mock_fig = MagicMock()
        mock_ax = MagicMock()

        with patch.object(plotter, "_create_metpy_skewt", return_value=(mock_fig, mock_ax)):
            with patch("mpasdiag.visualization.skewt.MPASVisualizationStyle.save_plot") as mock_save:
                with patch.object(plotter, "add_timestamp_and_branding"):
                    captured = StringIO()
                    with patch("sys.stdout", new=captured):
                        plotter.create_skewt_diagram(
                            PRESSURE, TEMPERATURE, DEWPOINT,
                            save_path=None,
                        )

        mock_save.assert_not_called()
        assert "saved to" not in captured.getvalue()


class TestPlotParcelProfileException:
    """ Test if the _plot_parcel_profile method correctly handles exceptions during parcel profile calculation, printing a warning when verbose is True and remaining silent when verbose is False. """

    def test_exception_verbose_prints_warning(self: 'TestPlotParcelProfileException') -> None:
        """
        This test checks that when the MPASSkewTPlotter is initialized with verbose=True, if an exception occurs during the parcel profile calculation in the _plot_parcel_profile method, it catches the exception and prints a warning message to the console that includes the exception message. It mocks the mpcalc.parcel_profile function to raise an exception and captures the standard output to verify that the expected warning is printed. 

        Parameters:
            None

        Returns:
            None
        """ 
        plotter = MPASSkewTPlotter(verbose=True)
        skew = MagicMock()
        pressure_hpa = MagicMock()
        temp_degc = MagicMock()
        dewpoint_degc = MagicMock()

        with patch(
            "mpasdiag.visualization.skewt.mpcalc.parcel_profile",
            side_effect=Exception("profile calculation failed"),
        ):
            captured = StringIO()
            with patch("sys.stdout", new=captured):
                plotter._plot_parcel_profile(
                    skew, pressure_hpa, temp_degc, dewpoint_degc, indices=None
                )

        out = captured.getvalue()
        assert "Warning: parcel profile plotting failed:" in out
        assert "profile calculation failed" in out

    def test_exception_verbose_false_silent(self: 'TestPlotParcelProfileException') -> None:
        """
        This test checks that when the MPASSkewTPlotter is initialized with verbose=False, if an exception occurs during the parcel profile calculation in the _plot_parcel_profile method, it catches the exception and does not print any warning message to the console. It mocks the mpcalc.parcel_profile function to raise an exception and captures the standard output to verify that it remains empty.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSkewTPlotter(verbose=False)
        skew = MagicMock()
        pressure_hpa = MagicMock()
        temp_degc = MagicMock()
        dewpoint_degc = MagicMock()

        with patch(
            "mpasdiag.visualization.skewt.mpcalc.parcel_profile",
            side_effect=RuntimeError("oops"),
        ):
            captured = StringIO()
            with patch("sys.stdout", new=captured):
                plotter._plot_parcel_profile(
                    skew, pressure_hpa, temp_degc, dewpoint_degc, indices=None
                )

        assert captured.getvalue() == ""

    def test_exception_with_indices_still_caught(self: 'TestPlotParcelProfileException') -> None:
        """
        This test checks that when the MPASSkewTPlotter is initialized with verbose=True, if an exception occurs during the parcel profile calculation in the _plot_parcel_profile method, it catches the exception and prints a warning message to the console that includes the exception message, even when indices are provided. It mocks the mpcalc.parcel_profile function to raise an exception and captures the standard output to verify that the expected warning is printed.

        Parameters:
            None

        Returns:
            None
        """
        plotter = MPASSkewTPlotter(verbose=True)
        skew = MagicMock()
        pressure_hpa = MagicMock()
        temp_degc = MagicMock()
        dewpoint_degc = MagicMock()
        indices = {"cape": 1500.0, "cin": -50.0}

        with patch(
            "mpasdiag.visualization.skewt.mpcalc.parcel_profile",
            side_effect=ValueError("bad profile"),
        ):
            captured = StringIO()
            with patch("sys.stdout", new=captured):
                plotter._plot_parcel_profile(
                    skew, pressure_hpa, temp_degc, dewpoint_degc, indices=indices
                )

        assert "Warning: parcel profile plotting failed:" in captured.getvalue()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

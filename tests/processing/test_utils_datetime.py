"""Tests for mpasdiag/processing/utils_datetime.py."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from datetime import datetime
from io import StringIO
from unittest.mock import patch

from mpasdiag.processing.utils_datetime import MPASDateTimeUtils


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def time_dataset() -> xr.Dataset:
    """Dataset with capital-T 'Time' coordinate, 5 hourly steps."""
    times = pd.date_range("2024-01-15T06:00:00", periods=5, freq="1h")
    return xr.Dataset(coords={"Time": times})


@pytest.fixture
def lowercase_time_dataset() -> xr.Dataset:
    """Dataset with lowercase 'time' coordinate, 3 hourly steps."""
    times = pd.date_range("2024-01-15T00:00:00", periods=3, freq="1h")
    return xr.Dataset(coords={"time": times})


@pytest.fixture
def no_time_dataset() -> xr.Dataset:
    """Dataset without any Time coordinate."""
    return xr.Dataset({"temperature": xr.DataArray(np.ones(10), dims=["nCells"])})


@pytest.fixture
def strftime_time_mock():
    """MagicMock dataset where Time.values[0] is a pd.Timestamp (has strftime) — triggers line 111."""
    from unittest.mock import MagicMock
    ts = pd.Timestamp("2024-01-15T06:00:00")
    mock_time = MagicMock()
    mock_time.__len__ = MagicMock(return_value=1)
    mock_time.values = [ts]
    mock_ds = MagicMock()
    mock_ds.Time = mock_time
    return mock_ds


# ---------------------------------------------------------------------------
# TestParseFileDatetimes
# ---------------------------------------------------------------------------

class TestParseFileDatetimes:
    """Tests for MPASDateTimeUtils.parse_file_datetimes."""

    def test_valid_filenames_parsed(self) -> None:
        files = [
            "/data/diag.2024-01-15_06.00.00.nc",
            "/data/diag.2024-01-15_12.00.00.nc",
        ]
        result = MPASDateTimeUtils.parse_file_datetimes(files, verbose=False)
        assert len(result) == 2
        assert result[0] == datetime(2024, 1, 15, 6, 0, 0)
        assert result[1] == datetime(2024, 1, 15, 12, 0, 0)

    def test_no_match_silent_generates_synthetic(self) -> None:
        files = ["no_date_in_this_filename.nc"]
        result = MPASDateTimeUtils.parse_file_datetimes(files, verbose=False)
        assert len(result) == 1
        assert result[0] == datetime(2000, 1, 1)

    def test_no_match_verbose_prints_warning(self) -> None:
        files = ["no_date_in_this_filename.nc"]
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            result = MPASDateTimeUtils.parse_file_datetimes(files, verbose=True)
        assert len(result) == 1
        assert "Could not parse datetime" in captured.getvalue()

    def test_invalid_datetime_match_silent_fallback(self) -> None:
        # Month 13 matches regex but fails datetime() → ValueError path
        files = ["/data/diag.2024-13-01_00.00.00.nc"]
        result = MPASDateTimeUtils.parse_file_datetimes(files, verbose=False)
        assert len(result) == 1
        assert result[0] == datetime(2000, 1, 1)

    def test_invalid_datetime_match_verbose_prints_warning(self) -> None:
        files = ["/data/diag.2024-13-01_00.00.00.nc"]
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            result = MPASDateTimeUtils.parse_file_datetimes(files, verbose=True)
        assert "Invalid datetime" in captured.getvalue()
        assert len(result) == 1


# ---------------------------------------------------------------------------
# TestValidateTimeParameters
# ---------------------------------------------------------------------------

class TestValidateTimeParameters:
    """Tests for MPASDateTimeUtils.validate_time_parameters."""

    def test_none_dataset_raises(self) -> None:
        with pytest.raises(ValueError):
            MPASDateTimeUtils.validate_time_parameters(None, 0)

    def test_valid_index_returns_correct_tuple(self, time_dataset: xr.Dataset) -> None:
        dim, idx, size = MPASDateTimeUtils.validate_time_parameters(time_dataset, 2)
        assert dim == "Time"
        assert idx == 2
        assert size == 5

    def test_out_of_range_index_clamped_silent(self, time_dataset: xr.Dataset) -> None:
        dim, idx, size = MPASDateTimeUtils.validate_time_parameters(time_dataset, 99, verbose=False)
        assert idx == size - 1

    def test_out_of_range_index_clamped_verbose(self, time_dataset: xr.Dataset) -> None:
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            dim, idx, size = MPASDateTimeUtils.validate_time_parameters(time_dataset, 99, verbose=True)
        assert idx == size - 1
        assert "exceeds" in captured.getvalue()

    def test_lowercase_time_dim_used(self, lowercase_time_dataset: xr.Dataset) -> None:
        dim, idx, size = MPASDateTimeUtils.validate_time_parameters(lowercase_time_dataset, 0)
        assert dim == "time"
        assert size == 3


# ---------------------------------------------------------------------------
# TestExtractTimeStr
# ---------------------------------------------------------------------------

class TestExtractTimeStr:
    """Tests for MPASDateTimeUtils._extract_time_str."""

    def test_no_time_attr_returns_none(self, no_time_dataset: xr.Dataset) -> None:
        result = MPASDateTimeUtils._extract_time_str(no_time_dataset, 0)
        assert result is None

    def test_time_index_out_of_range_returns_none(self, time_dataset: xr.Dataset) -> None:
        result = MPASDateTimeUtils._extract_time_str(time_dataset, 999)
        assert result is None

    def test_timestamp_uses_strftime_path(self, strftime_time_mock) -> None:
        # pd.Timestamp has .strftime → line 111
        time_val = strftime_time_mock.Time.values[0]
        assert hasattr(time_val, "strftime"), "Mock must yield a pd.Timestamp"
        result = MPASDateTimeUtils._extract_time_str(strftime_time_mock, 0)
        assert result == "20240115T06"

    def test_numpy_datetime64_uses_pd_to_datetime(self, time_dataset: xr.Dataset) -> None:
        # numpy datetime64 has no .strftime → line 113
        time_val = time_dataset.Time.values[0]
        assert not hasattr(time_val, "strftime"), "Fixture must contain numpy datetime64"
        result = MPASDateTimeUtils._extract_time_str(time_dataset, 0)
        assert result == "20240115T06"


# ---------------------------------------------------------------------------
# TestLogTimeInfo
# ---------------------------------------------------------------------------

class TestLogTimeInfo:
    """Tests for MPASDateTimeUtils._log_time_info."""

    def test_silent_mode_produces_no_output(self) -> None:
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            MPASDateTimeUtils._log_time_info(False, 0, "20240115T06", "precip")
        assert captured.getvalue() == ""

    def test_time_str_present_no_context(self) -> None:
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            MPASDateTimeUtils._log_time_info(True, 0, "20240115T06", "")
        output = captured.getvalue()
        assert "corresponds to: 20240115T06" in output
        assert "using variable" not in output

    def test_time_str_present_with_context(self) -> None:
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            MPASDateTimeUtils._log_time_info(True, 0, "20240115T06", "precip")
        assert "using variable: precip" in captured.getvalue()

    def test_error_present_no_context(self) -> None:
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            MPASDateTimeUtils._log_time_info(True, 2, None, "", error=RuntimeError("oops"))
        output = captured.getvalue()
        assert "Using time index 2" in output
        assert "oops" in output
        assert "using variable" not in output

    def test_error_present_with_context(self) -> None:
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            MPASDateTimeUtils._log_time_info(True, 2, None, "wind", error=RuntimeError("oops"))
        output = captured.getvalue()
        assert "using variable: wind" in output
        assert "oops" in output

    def test_no_time_str_no_error_no_context(self) -> None:
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            MPASDateTimeUtils._log_time_info(True, 3, None, "")
        output = captured.getvalue()
        assert "Using time index 3" in output
        assert "time coordinate not available" in output
        assert "using variable" not in output

    def test_no_time_str_no_error_with_context(self) -> None:
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            MPASDateTimeUtils._log_time_info(True, 3, None, "temperature")
        assert "using variable: temperature" in captured.getvalue()


# ---------------------------------------------------------------------------
# TestGetTimeInfo
# ---------------------------------------------------------------------------

class TestGetTimeInfo:
    """Tests for MPASDateTimeUtils.get_time_info."""

    def test_exception_in_extract_returns_fallback(self, time_dataset: xr.Dataset) -> None:
        with patch.object(
            MPASDateTimeUtils, "_extract_time_str", side_effect=RuntimeError("forced")
        ):
            result = MPASDateTimeUtils.get_time_info(time_dataset, 0, verbose=False)
        assert result == "time_0"

    def test_exception_in_extract_verbose_logs_error(self, time_dataset: xr.Dataset) -> None:
        captured = StringIO()
        with patch("sys.stdout", new=captured):
            with patch.object(
                MPASDateTimeUtils, "_extract_time_str", side_effect=RuntimeError("forced")
            ):
                result = MPASDateTimeUtils.get_time_info(time_dataset, 0, verbose=True)
        assert result == "time_0"
        assert "forced" in captured.getvalue()

    def test_valid_dataset_returns_time_string(self, time_dataset: xr.Dataset) -> None:
        result = MPASDateTimeUtils.get_time_info(time_dataset, 0, verbose=False)
        assert result == "20240115T06"

    def test_no_time_coord_returns_fallback(self, no_time_dataset: xr.Dataset) -> None:
        result = MPASDateTimeUtils.get_time_info(no_time_dataset, 0, verbose=False)
        assert result == "time_0"


# ---------------------------------------------------------------------------
# TestGetTimeRange
# ---------------------------------------------------------------------------

class TestGetTimeRange:
    """Tests for MPASDateTimeUtils.get_time_range."""

    def test_none_dataset_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be None"):
            MPASDateTimeUtils.get_time_range(None)  # type: ignore[arg-type]

    def test_no_time_coord_raises(self, no_time_dataset: xr.Dataset) -> None:
        with pytest.raises(ValueError, match="Time coordinate"):
            MPASDateTimeUtils.get_time_range(no_time_dataset)

    def test_valid_dataset_returns_start_end(self, time_dataset: xr.Dataset) -> None:
        start, end = MPASDateTimeUtils.get_time_range(time_dataset)
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)
        assert start < end
        assert start == datetime(2024, 1, 15, 6, 0, 0)
        assert end == datetime(2024, 1, 15, 10, 0, 0)


# ---------------------------------------------------------------------------
# TestFormatTimeForFilename
# ---------------------------------------------------------------------------

class TestFormatTimeForFilename:
    """Tests for MPASDateTimeUtils.format_time_for_filename."""

    def setup_method(self) -> None:
        self.dt = datetime(2024, 1, 15, 6, 30, 45)

    def test_mpas_format(self) -> None:
        assert MPASDateTimeUtils.format_time_for_filename(self.dt, "mpas") == "2024-01-15_06.30.45"

    def test_iso_format(self) -> None:
        assert MPASDateTimeUtils.format_time_for_filename(self.dt, "iso") == "20240115T063045"

    def test_compact_format(self) -> None:
        assert MPASDateTimeUtils.format_time_for_filename(self.dt, "compact") == "2024011506"

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown format_type"):
            MPASDateTimeUtils.format_time_for_filename(self.dt, "invalid")


# ---------------------------------------------------------------------------
# TestParseTimeFromString
# ---------------------------------------------------------------------------

class TestParseTimeFromString:
    """Tests for MPASDateTimeUtils.parse_time_from_string."""

    def test_default_patterns_mpas_format(self) -> None:
        result = MPASDateTimeUtils.parse_time_from_string("2024-01-15_06.30.45")
        assert result == datetime(2024, 1, 15, 6, 30, 45)

    def test_default_patterns_iso_compact_format(self) -> None:
        result = MPASDateTimeUtils.parse_time_from_string("20240115T063045")
        assert result == datetime(2024, 1, 15, 6, 30, 45)

    def test_default_patterns_standard_format(self) -> None:
        result = MPASDateTimeUtils.parse_time_from_string("2024-01-15 06:30:45")
        assert result == datetime(2024, 1, 15, 6, 30, 45)

    def test_custom_pattern_used(self) -> None:
        result = MPASDateTimeUtils.parse_time_from_string("15/01/2024", ["%d/%m/%Y"])
        assert result == datetime(2024, 1, 15)

    def test_unparseable_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not parse"):
            MPASDateTimeUtils.parse_time_from_string("not-a-date", ["%Y%m%d"])

    def test_none_patterns_uses_defaults(self) -> None:
        result = MPASDateTimeUtils.parse_time_from_string("2024-01-15_06.30.45", None)
        assert result == datetime(2024, 1, 15, 6, 30, 45)


# ---------------------------------------------------------------------------
# TestGetTimeBounds
# ---------------------------------------------------------------------------

class TestGetTimeBounds:
    """Tests for MPASDateTimeUtils.get_time_bounds."""

    def test_none_dataset_returns_none_none(self) -> None:
        result = MPASDateTimeUtils.get_time_bounds(None, 0)  # type: ignore[arg-type]
        assert result == (None, None)

    def test_no_bounds_variable_returns_none_none(self, time_dataset: xr.Dataset) -> None:
        result = MPASDateTimeUtils.get_time_bounds(time_dataset, 0)
        assert result == (None, None)

    def test_time_bnds_variable_returns_bounds(self) -> None:
        bounds = np.array([
            [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")],
            [np.datetime64("2024-01-02"), np.datetime64("2024-01-03")],
        ], dtype="datetime64[ns]")
        ds = xr.Dataset({"time_bnds": xr.DataArray(bounds, dims=["time", "bnds"])})
        start, end = MPASDateTimeUtils.get_time_bounds(ds, 0)
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)
        assert start < end

    def test_Time_bnds_variable_returns_bounds(self) -> None:
        bounds = np.array([
            [np.datetime64("2024-03-01"), np.datetime64("2024-03-02")],
        ], dtype="datetime64[ns]")
        ds = xr.Dataset({"Time_bnds": xr.DataArray(bounds, dims=["Time", "bnds"])})
        start, end = MPASDateTimeUtils.get_time_bounds(ds, 0)
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)

    def test_index_error_in_bounds_returns_none_none(self) -> None:
        # Only 1 row; requesting time_index=99 causes IndexError → continue → (None, None)
        bounds = np.array(
            [[np.datetime64("2024-01-01"), np.datetime64("2024-01-02")]],
            dtype="datetime64[ns]",
        )
        ds = xr.Dataset({"time_bnds": xr.DataArray(bounds, dims=["time", "bnds"])})
        result = MPASDateTimeUtils.get_time_bounds(ds, 99)
        assert result == (None, None)


# ---------------------------------------------------------------------------
# TestCalculateTimeDelta
# ---------------------------------------------------------------------------

class TestCalculateTimeDelta:
    """Tests for MPASDateTimeUtils.calculate_time_delta."""

    def test_none_dataset_raises(self) -> None:
        with pytest.raises(ValueError):
            MPASDateTimeUtils.calculate_time_delta(None)  # type: ignore[arg-type]

    def test_no_time_coord_raises(self, no_time_dataset: xr.Dataset) -> None:
        with pytest.raises(ValueError):
            MPASDateTimeUtils.calculate_time_delta(no_time_dataset)

    def test_single_time_step_raises(self) -> None:
        times = pd.date_range("2024-01-01", periods=1, freq="1h")
        ds = xr.Dataset(coords={"Time": times})
        with pytest.raises(ValueError, match="at least 2"):
            MPASDateTimeUtils.calculate_time_delta(ds)

    def test_valid_dataset_returns_correct_timedelta(self) -> None:
        times = pd.date_range("2024-01-01", periods=5, freq="6h")
        ds = xr.Dataset(coords={"Time": times})
        delta = MPASDateTimeUtils.calculate_time_delta(ds)
        assert isinstance(delta, pd.Timedelta)
        assert delta == pd.Timedelta("6h")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

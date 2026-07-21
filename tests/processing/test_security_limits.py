#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
Tests for the security-hardening helpers added in the pre-release audit:

This module contains unit tests for the security-hardening helpers introduced in the pre-release audit of the MPASdiag package. The tests focus on verifying the functionality of the enforce_size_limits function and the safe_resolve_within function, ensuring that they correctly enforce input size limits and path containment, respectively.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: June 2026
Version: 1.0.0
"""

import pytest
from typing import Any

from mpasdiag.processing.utils_validator import DataValidator
from mpasdiag.processing.utils_path import (
    safe_resolve_within,
    sanitize_filename_component,
    safe_label,
    safe_plot_text,
)
from mpasdiag.processing.utils_config import MPASConfig
from mpasdiag.processing import constants
from pathlib import Path


class TestEnforceSizeLimits:
    """Generous, configurable input-size caps (MPAS-001)."""

    def test_within_limits_passes(self: "TestEnforceSizeLimits") -> None:
        """
        This test verifies that the enforce_size_limits function does not raise an exception when the provided values for source grid cells, target grid points, non-zero remap weight entries, and cross-section interpolation points are all well below their respective default limits. It ensures that valid inputs are accepted without any issues.

        Parameters:
            None

        Returns:
            None
        """
        # Well below every default limit -> no exception.
        DataValidator.enforce_size_limits(
            n_src=1000, n_tgt=2000, nnz=5000, num_points=100
        )

    def test_none_values_are_skipped(self: "TestEnforceSizeLimits") -> None:
        """
        This test checks that the enforce_size_limits function correctly handles None values for its parameters. When None is passed for any of the parameters (n_src, n_tgt, nnz, num_points), the function should skip the corresponding checks and not raise any exceptions. This allows callers to only specify the dimensions relevant to their allocation without being forced to provide values for all parameters.

        Parameters:
            None

        Returns:
            None
        """
        # Passing nothing relevant must never raise.
        DataValidator.enforce_size_limits()

    def test_source_cells_over_default_raises(self: "TestEnforceSizeLimits") -> None:
        """
        This test verifies that the enforce_size_limits function raises a ValueError when the number of source grid cells (n_src) exceeds the default limit defined in constants.MAX_SOURCE_CELLS. It ensures that the function correctly enforces the safety limit for source grid cells and provides an appropriate error message indicating the issue.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="source grid cells"):
            DataValidator.enforce_size_limits(n_src=constants.MAX_SOURCE_CELLS + 1)

    def test_target_points_over_default_raises(self: "TestEnforceSizeLimits") -> None:
        """
        This test checks that the enforce_size_limits function raises a ValueError when the number of target grid points (n_tgt) exceeds the default limit defined in constants.MAX_TARGET_POINTS. It ensures that the function correctly enforces the safety limit for target grid points and provides an appropriate error message indicating the issue.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="target grid points"):
            DataValidator.enforce_size_limits(n_tgt=constants.MAX_TARGET_POINTS + 1)

    def test_nnz_over_default_raises(self: "TestEnforceSizeLimits") -> None:
        """
        This test verifies that the enforce_size_limits function raises a ValueError when the number of non-zero remap weight entries (nnz) exceeds the default limit defined in constants.MAX_WEIGHTS_NNZ. It ensures that the function correctly enforces the safety limit for non-zero remap weight entries and provides an appropriate error message indicating the issue.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="weight entries"):
            DataValidator.enforce_size_limits(nnz=constants.MAX_WEIGHTS_NNZ + 1)

    def test_num_points_over_default_raises(self: "TestEnforceSizeLimits") -> None:
        """
        This test checks that the enforce_size_limits function raises a ValueError when the number of cross-section interpolation points (num_points) exceeds the default limit defined in constants.MAX_NUM_POINTS. It ensures that the function correctly enforces the safety limit for cross-section interpolation points and provides an appropriate error message indicating the issue.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="cross-section points"):
            DataValidator.enforce_size_limits(num_points=constants.MAX_NUM_POINTS + 1)

    def test_negative_value_raises(self: "TestEnforceSizeLimits") -> None:
        """
        This test checks that the enforce_size_limits function raises a ValueError when a negative value is provided for the number of source grid points (n_src). It ensures that the function correctly enforces the safety limit for non-negative values and provides an appropriate error message indicating the issue.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="negative"):
            DataValidator.enforce_size_limits(n_src=-1)

    def test_env_override_relaxes_limit(
        self: "TestEnforceSizeLimits", monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        This test checks that the enforce_size_limits function allows an override of the default limit for the number of source grid points (n_src) through an environment variable (MPASDIAG_MAX_SOURCE_CELLS). It ensures that the function correctly respects the environment variable and allows values above the default limit when the override is set.

        Parameters:
            monkeypatch: pytest.MonkeyPatch

        Returns:
            None
        """
        # An override above the requested size lets an otherwise-rejected input pass.
        monkeypatch.setenv(
            "MPASDIAG_MAX_SOURCE_CELLS", str(constants.MAX_SOURCE_CELLS + 10)
        )
        DataValidator.enforce_size_limits(n_src=constants.MAX_SOURCE_CELLS + 5)

    def test_env_override_can_tighten_limit(
        self: "TestEnforceSizeLimits", monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        This test checks that the enforce_size_limits function respects an environment variable (MPASDIAG_MAX_NUM_POINTS) that tightens the default limit for the number of cross-section interpolation points (num_points). It ensures that the function correctly enforces the tightened limit and raises a ValueError when the limit is exceeded.

        Parameters:
            monkeypatch: pytest.MonkeyPatch

        Returns:
            None
        """
        monkeypatch.setenv("MPASDIAG_MAX_NUM_POINTS", "10")
        with pytest.raises(ValueError, match="cross-section points"):
            DataValidator.enforce_size_limits(num_points=11)

    def test_invalid_env_override_raises(
        self: "TestEnforceSizeLimits", monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        This test checks that the enforce_size_limits function raises a ValueError when the environment variable (MPASDIAG_MAX_SOURCE_CELLS) is set to a non-integer value. It ensures that the function correctly validates the environment variable and raises an appropriate error message indicating that the value must be a positive integer.

        Parameters:
            monkeypatch: pytest.MonkeyPatch

        Returns:
            None
        """
        monkeypatch.setenv("MPASDIAG_MAX_SOURCE_CELLS", "not-a-number")
        with pytest.raises(ValueError, match="positive integer"):
            DataValidator.enforce_size_limits(n_src=1)

    def test_nonpositive_env_override_raises(
        self: "TestEnforceSizeLimits", monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        This test checks that the enforce_size_limits function raises a ValueError when the environment variable (MPASDIAG_MAX_SOURCE_CELLS) is set to a non-positive value. It ensures that the function correctly validates the environment variable and raises an appropriate error message indicating that the value must be a positive integer.

        Parameters:
            monkeypatch: pytest.MonkeyPatch

        Returns:
            None
        """
        monkeypatch.setenv("MPASDIAG_MAX_SOURCE_CELLS", "0")
        with pytest.raises(ValueError, match="positive integer"):
            DataValidator.enforce_size_limits(n_src=1)


class TestSafeResolveWithin:
    """Path-traversal containment guard (MPAS-005)."""

    def test_valid_relative_path_resolves_inside_base(
        self: "TestSafeResolveWithin", tmp_path: Path
    ) -> None:
        """
        This test verifies that the safe_resolve_within function correctly resolves a valid relative file path within a specified base directory. It ensures that the resolved path is an absolute path that remains within the base directory and does not raise any exceptions.

        Parameters:
            tmp_path: Path - A temporary directory provided by pytest for testing.

        Returns:
            None
        """
        resolved = safe_resolve_within("weights.nc", str(tmp_path))
        assert resolved == (tmp_path / "weights.nc").resolve()

    def test_traversal_escape_rejected(
        self: "TestSafeResolveWithin", tmp_path: Path
    ) -> None:
        """
        This test checks that the safe_resolve_within function raises a ValueError when a path traversal escape is attempted. It ensures that the function correctly identifies and rejects paths that attempt to access files outside the specified base directory, such as '../../etc/passwd'.

        Parameters:
            tmp_path: Path - A temporary directory provided by pytest for testing.

        Returns:
            None
        """
        with pytest.raises(ValueError, match="outside"):
            safe_resolve_within("../../etc/passwd", str(tmp_path))

    def test_absolute_path_outside_rejected(
        self: "TestSafeResolveWithin", tmp_path: Path
    ) -> None:
        """
        This test verifies that the safe_resolve_within function raises a ValueError when an absolute file path outside the specified base directory is provided. It ensures that the function correctly identifies and rejects absolute paths that do not reside within the base directory, such as '/etc/hosts'.

        Parameters:
            tmp_path: Path - A temporary directory provided by pytest for testing.

        Returns:
            None
        """
        with pytest.raises(ValueError, match="outside"):
            safe_resolve_within("/etc/hosts", str(tmp_path))

    def test_suffix_enforced(self: "TestSafeResolveWithin", tmp_path: Path) -> None:
        """
        This test checks that the safe_resolve_within function raises a ValueError when a file path with a disallowed suffix is provided. It ensures that the function correctly enforces the allowed file extensions specified in the allowed_suffixes parameter and raises an appropriate error message when the file's suffix does not match any of the allowed extensions.

        Parameters:
            tmp_path: Path - A temporary directory provided by pytest for testing.

        Returns:
            None
        """
        with pytest.raises(ValueError, match="extensions"):
            safe_resolve_within("weights.txt", str(tmp_path), allowed_suffixes=(".nc",))

    def test_suffix_allowed_passes(
        self: "TestSafeResolveWithin", tmp_path: Path
    ) -> None:
        """
        This test verifies that the safe_resolve_within function correctly allows a file path with an allowed suffix. It ensures that when a file path with a suffix that matches one of the allowed extensions specified in the allowed_suffixes parameter is provided, the function resolves the path without raising any exceptions.

        Parameters:
            tmp_path: Path - A temporary directory provided by pytest for testing.

        Returns:
            None
        """
        resolved = safe_resolve_within(
            "weights.nc", str(tmp_path), allowed_suffixes=(".nc",)
        )
        assert resolved.suffix == ".nc"

    def test_must_exist_raises_for_missing(
        self: "TestSafeResolveWithin", tmp_path: Path
    ) -> None:
        """
        This test verifies that the safe_resolve_within function raises a FileNotFoundError when a file path that must exist is missing. It ensures that the function correctly identifies and raises an error for non-existent files when the must_exist parameter is set to True.

        Parameters:
            tmp_path: Path - A temporary directory provided by pytest for testing.

        Returns:
            None
        """
        with pytest.raises(FileNotFoundError):
            safe_resolve_within("missing.nc", str(tmp_path), must_exist=True)

    def test_must_exist_passes_for_present(
        self: "TestSafeResolveWithin", tmp_path: Path
    ) -> None:
        """
        This test checks that the safe_resolve_within function correctly resolves a file path that must exist when the file is present. It ensures that when a file path is provided for a file that exists in the specified base directory and the must_exist parameter is set to True, the function resolves the path without raising any exceptions.

        Parameters:
            tmp_path: Path - A temporary directory provided by pytest for testing.

        Returns:
            None
        """
        target = tmp_path / "present.nc"
        target.write_bytes(b"")
        resolved = safe_resolve_within("present.nc", str(tmp_path), must_exist=True)
        assert resolved == target.resolve()


class TestExtendedSizeLimits:
    """Extended input-size caps for testing (MPAS-001)."""

    def test_cell_vertices_over_default_raises(self: "TestExtendedSizeLimits") -> None:
        """
        This test verifies that the enforce_size_limits function raises a ValueError when the number of cell vertices (nv) exceeds the default limit defined in constants.MAX_CELL_VERTICES. It ensures that the function correctly enforces the safety limit for cell vertices and provides an appropriate error message indicating the issue.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="cell vertices"):
            DataValidator.enforce_size_limits(nv=constants.MAX_CELL_VERTICES + 1)

    def test_workers_over_default_raises(self: "TestExtendedSizeLimits") -> None:
        """
        This test verifies that the enforce_size_limits function raises a ValueError when the number of worker processes (n_workers) exceeds the default limit defined in constants.MAX_WORKERS. It ensures that the function correctly enforces the safety limit for worker processes and provides an appropriate error message indicating the issue.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="worker processes"):
            DataValidator.enforce_size_limits(n_workers=constants.MAX_WORKERS + 1)

    def test_input_files_over_default_raises(self: "TestExtendedSizeLimits") -> None:
        """
        This test verifies that the enforce_size_limits function raises a ValueError when the number of input files (n_files) exceeds the default limit defined in constants.MAX_INPUT_FILES. It ensures that the function correctly enforces the safety limit for input files and provides an appropriate error message indicating the issue.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="input files"):
            DataValidator.enforce_size_limits(n_files=constants.MAX_INPUT_FILES + 1)

    def test_new_limits_env_override(
        self: "TestExtendedSizeLimits", monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        This test checks that the enforce_size_limits function allows an override of the default limit for the number of cell vertices (nv) through an environment variable (MPASDIAG_MAX_CELL_VERTICES). It ensures that the function correctly respects the environment variable and allows values above the default limit when the override is set.

        Parameters:
            monkeypatch: pytest.MonkeyPatch - Fixture for modifying environment variables.

        Returns:
            None
        """
        monkeypatch.setenv("MPASDIAG_MAX_CELL_VERTICES", "5")
        with pytest.raises(ValueError, match="cell vertices"):
            DataValidator.enforce_size_limits(nv=6)


class TestSanitizeFilenameComponent:
    """Filename-component sanitization (F2): no separators or traversal survive."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("../../etc/passwd", "etc_passwd"),
            ("foo/bar", "foo_bar"),
            ("a\\b", "a_b"),
            ("x\x00y", "x_y"),
            ("te\nst", "te_st"),
            ("..", "output"),
            (".", "output"),
            ("", "output"),
            (".hidden", "hidden"),
            ("rainnc", "rainnc"),
        ],
    )
    def test_sanitize(
        self: "TestSanitizeFilenameComponent", raw: str, expected: str
    ) -> None:
        """
        This test verifies that the sanitize_filename_component function correctly sanitizes various raw filename components by removing or replacing disallowed characters and patterns. It ensures that the sanitized output matches the expected result and does not contain any path separators or traversal sequences.

        Parameters:
            raw: str - The raw filename component to be sanitized.
            expected: str - The expected sanitized output.

        Returns:
            None
        """
        result = sanitize_filename_component(raw)
        assert result == expected
        assert "/" not in result and "\\" not in result
        assert ".." not in result

    def test_custom_fallback(self: "TestSanitizeFilenameComponent") -> None:
        """
        This test checks that the sanitize_filename_component function correctly uses a custom fallback value when the sanitized result is empty or invalid. It ensures that when the input string is such that it results in an empty sanitized output, the function returns the specified fallback value instead.

        Parameters:
            None

        Returns:
            None
        """
        assert sanitize_filename_component("///", fallback="var") == "var"


class TestMessageAndPlotTextSanitizers:
    """Log/plot text sanitization for the LLM-agent threat model (F15/F16)."""

    def test_safe_label_strips_control_chars(
        self: "TestMessageAndPlotTextSanitizers",
    ) -> None:
        """
        This test verifies that the safe_label function correctly removes control characters (such as newlines, tabs, and null bytes) from the input string. It ensures that the sanitized output is free of any control characters that could potentially disrupt logging or plotting.

        Parameters:
            None

        Returns:
            None
        """
        assert safe_label("a\nb\tc\x00d") == "a b c d"

    def test_safe_label_truncates(self: "TestMessageAndPlotTextSanitizers") -> None:
        """
        This test checks that the safe_label function correctly truncates long input strings to a specified maximum length. It ensures that when the input string exceeds the max_len parameter, the function returns a truncated version of the string with an appropriate indication (e.g., "(truncated)") appended to it.

        Parameters:
            None

        Returns:
            None
        """
        out = safe_label("x" * 500, max_len=50)
        assert out.endswith("(truncated)") and len(out) <= 70

    def test_safe_plot_text_removes_mathtext(
        self: "TestMessageAndPlotTextSanitizers",
    ) -> None:
        """
        This test verifies that the safe_plot_text function correctly removes LaTeX mathtext expressions (enclosed in dollar signs) from the input string. It ensures that any mathematical expressions intended for plotting are stripped out, leaving only the plain text content.

        Parameters:
            None

        Returns:
            None
        """
        assert "$" not in safe_plot_text("temp $\\frac{1}{0}$ K")


class TestConfigValidationIntegrity:
    """Merged-config revalidation and numeric bounds (F10/F11/F12)."""

    def test_numeric_bounds_reject_abusive_dpi(
        self: "TestConfigValidationIntegrity",
    ) -> None:
        """
        This test checks that the MPASConfig class raises a ValueError when an excessively high DPI (dots per inch) value is specified. It ensures that the configuration validation enforces reasonable bounds for the DPI setting, preventing configurations that could lead to performance issues or resource exhaustion.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="dpi"):
            MPASConfig(dpi=1_000_000)

    def test_auto_sentinels_are_accepted(
        self: "TestConfigValidationIntegrity",
    ) -> None:
        """
        This test verifies that the MPASConfig class accepts sentinel values for certain configuration parameters, such as subsample_factor and time_index. It ensures that the configuration validation allows these sentinel values to be used without raising any exceptions, indicating that they are valid inputs for the respective parameters.

        Parameters:
            None

        Returns:
            None
        """
        MPASConfig(subsample_factor=-1)
        MPASConfig(subsample_factor=0)
        MPASConfig(time_index=-1)

    def test_numeric_bounds_reject_negative_workers(
        self: "TestConfigValidationIntegrity",
    ) -> None:
        """
        This test checks that the MPASConfig class raises a ValueError when a negative number of worker processes is specified. It ensures that the configuration validation enforces non-negative bounds for the number of workers, preventing configurations that could lead to runtime errors or unexpected behavior.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="workers"):
            MPASConfig(workers=-4)

    def test_revalidate_catches_merged_bad_remap_engine(
        self: "TestConfigValidationIntegrity",
    ) -> None:
        """
        This test verifies that the MPASConfig class raises a ValueError when an invalid remap engine is specified. It ensures that the configuration validation correctly identifies unsupported remap engine values and raises an appropriate error message.

        Parameters:
            None

        Returns:
            None
        """
        cfg = MPASConfig()
        cfg.remap_engine = "totally-bogus"
        with pytest.raises(ValueError, match="remap_engine"):
            cfg.revalidate()

    def test_revalidate_catches_merged_bad_extent(
        self: "TestConfigValidationIntegrity",
    ) -> None:
        """
        This test checks that the MPASConfig class raises a ValueError when the specified spatial extent is invalid (i.e., when lat_min is greater than lat_max). It ensures that the configuration validation correctly identifies and rejects configurations with inconsistent spatial bounds.

        Parameters:
            None

        Returns:
            None
        """
        cfg = MPASConfig()
        cfg.lat_min = 50.0
        cfg.lat_max = 10.0
        with pytest.raises(ValueError, match="spatial extent"):
            cfg.revalidate()

    def test_from_dict_filters_unknown_keys(
        self: "TestConfigValidationIntegrity",
    ) -> None:
        """
        This test verifies that the MPASConfig class correctly filters out unknown keys when creating a configuration instance from a dictionary. It ensures that only recognized configuration keys are retained, while any unrecognized keys are ignored without raising an error.

        Parameters:
            None

        Returns:
            None
        """
        cfg = MPASConfig.from_dict(
            {"variable": "t2m", "an_unknown_key": 123, "another": "x"}
        )
        assert cfg.variable == "t2m"

    def test_from_dict_rejects_non_mapping(
        self: "TestConfigValidationIntegrity",
    ) -> None:
        """
        This test checks that the MPASConfig class raises a ValueError when attempting to create a configuration instance from a non-mapping object (e.g., a list). It ensures that the from_dict method enforces the requirement that the input must be a dictionary or mapping type.

        Parameters:
            None

        Returns:
            None
        """
        not_a_mapping: Any = ["not", "a", "dict"]
        with pytest.raises(ValueError, match="mapping"):
            MPASConfig.from_dict(not_a_mapping)

    def test_resolved_base_dir_defaults_to_cwd(
        self: "TestConfigValidationIntegrity",
    ) -> None:
        """
        This test verifies that the resolved_base_dir method of the MPASConfig class defaults to the current working directory when no explicit base_dir is provided. It ensures that the method correctly resolves the base directory to the current working directory in the absence of a user-specified value.

        Parameters:
            None

        Returns:
            None
        """
        assert MPASConfig().resolved_base_dir() == Path.cwd().resolve()

    def test_resolved_base_dir_uses_explicit_value(
        self: "TestConfigValidationIntegrity", tmp_path: Path
    ) -> None:
        """
        This test checks that the resolved_base_dir method of the MPASConfig class correctly uses an explicitly provided base_dir value. It ensures that when a specific base directory is set, the method resolves and returns that directory instead of defaulting to the current working directory.

        Parameters:
            None

        Returns:
            None
        """
        cfg = MPASConfig(base_dir=str(tmp_path))
        assert cfg.resolved_base_dir() == tmp_path.resolve()


class TestWorkerCountClamp:
    """Fork-bomb guard on the multiprocessing pool size (F8)."""

    def test_excessive_request_is_clamped(self: "TestWorkerCountClamp") -> None:
        """
        This test verifies that the _resolve_worker_count method of the MPASParallelManager class correctly clamps an excessively high requested worker count to the maximum allowed value defined in constants.MAX_WORKERS. It ensures that the method enforces the upper limit on the number of worker processes to prevent resource exhaustion or potential fork-bomb scenarios.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.parallel import MPASParallelManager

        assert (
            MPASParallelManager._resolve_worker_count(100_000) == constants.MAX_WORKERS
        )

    def test_nonpositive_request_floors_at_one(self: "TestWorkerCountClamp") -> None:
        """
        This test checks that the _resolve_worker_count method of the MPASParallelManager class correctly floors a non-positive requested worker count to a minimum of one. It ensures that the method enforces a lower limit on the number of worker processes, preventing configurations that would result in zero or negative workers.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.parallel import MPASParallelManager

        assert MPASParallelManager._resolve_worker_count(-3) == 1
        assert MPASParallelManager._resolve_worker_count(0) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

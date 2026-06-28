#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
Tests for the security-hardening helpers added in the pre-release audit:

This module contains unit tests for the security-hardening helpers introduced in the pre-release audit of the MPASdiag package. The tests focus on verifying the functionality of the enforce_size_limits function and the safe_resolve_within function, ensuring that they correctly enforce input size limits and path containment, respectively.
"""

import pytest

from mpasdiag.processing.utils_validator import DataValidator
from mpasdiag.processing.utils_path import safe_resolve_within
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

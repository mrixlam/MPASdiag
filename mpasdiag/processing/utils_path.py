#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

"""
MPASdiag Core Processing Module: Path Containment Utilities

This module provides a single, dependency-free helper for safely resolving a user- or caller-supplied file path against a trusted base directory. It guards against path-traversal escapes (e.g. ``'../../etc/passwd'`` or absolute paths outside the working tree) and optionally enforces an allowed file-extension set. It is shared by the configuration loader and the remapping weights cache so that both use the same proven containment logic (security audit findings MPAS-005). This module intentionally imports nothing from the rest of the package, keeping it free of import cycles so any module may depend on it.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: November 2025
Version: 1.0.0
"""

from pathlib import Path
from typing import Iterable, Optional, Union


def safe_resolve_within(
    filepath: Union[str, Path],
    base_dir: Optional[Union[str, Path]],
    *,
    allowed_suffixes: Optional[Iterable[str]] = None,
    must_exist: bool = False,
) -> Path:
    """
    This function resolves a user-supplied file path against a trusted base directory, ensuring that the resolved path does not escape the base directory and optionally enforcing allowed file extensions. It also checks for the existence of the file if required.

    Parameters:
        filepath (Union[str, Path]): Raw path supplied by the caller (relative or absolute).
        base_dir (Optional[Union[str, Path]]): Directory the path must stay within. Defaults to the current working directory when None.
        allowed_suffixes (Optional[Iterable[str]]): If given, the resolved path's suffix (case-insensitive) must be one of these (e.g. ``(".yaml", ".yml")``).
        must_exist (bool): When True, require the resolved path to be an existing regular file.

    Returns:
        Path: The resolved, validated absolute path.
    """
    base = (Path(base_dir) if base_dir else Path.cwd()).resolve()
    resolved = (base / filepath).resolve()

    if not resolved.is_relative_to(base):
        raise ValueError(f"Refusing to access path outside '{base}': {filepath}")

    if allowed_suffixes is not None:
        allowed = {suffix.lower() for suffix in allowed_suffixes}
        if resolved.suffix.lower() not in allowed:
            raise ValueError(
                f"Path must have one of {sorted(allowed)} extensions: {filepath}"
            )

    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    return resolved

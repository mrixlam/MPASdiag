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

import re
from pathlib import Path
from typing import Iterable, Optional, Union

_UNSAFE_COMPONENT_CHARS = re.compile(r"[/\\\x00-\x1f\x7f]")


def sanitize_filename_component(name: str, *, fallback: str = "output") -> str:
    """
    This function sanitizes a string to be used as a safe filename component. It replaces unsafe characters with underscores, collapses multiple dots and underscores, and strips leading/trailing dots and whitespace. If the resulting string is empty or invalid, it returns a fallback value.

    Parameters:
        name (str): Raw, possibly attacker-influenced token.
        fallback (str): Value returned when ``name`` sanitizes to empty.

    Returns:
        str: A safe filename component.
    """
    cleaned = _UNSAFE_COMPONENT_CHARS.sub("_", str(name))
    # Collapse any run of dots so ``..`` (and longer) can never survive.
    cleaned = re.sub(r"\.{2,}", "_", cleaned)
    # Collapse runs of underscores introduced by substitution for tidy names.
    cleaned = re.sub(r"_{2,}", "_", cleaned)
    # A leading dot/underscore would create a hidden/ugly file; strip them.
    cleaned = cleaned.strip(". _\t")
    if not cleaned or cleaned in {".", ".."}:
        return fallback
    return cleaned


def safe_label(text: object, *, max_len: int = 200) -> str:
    """
    This function sanitizes untrusted text for safe rendering in labels or titles. It removes control characters, collapses whitespace, and truncates the string to a specified maximum length. This helps prevent issues with rendering and ensures that the text is safe to display.

    Parameters:
        text (object): Value to render safely (coerced with ``str``).
        max_len (int): Maximum length before truncation.

    Returns:
        str: A single-line, control-character-free, length-bounded string.
    """
    s = re.sub(r"[\x00-\x1f\x7f]", " ", str(text))
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[:max_len] + "…(truncated)"
    return s


def safe_plot_text(text: object, *, max_len: int = 300) -> str:
    """
    This function sanitizes untrusted text for safe rendering in matplotlib labels or titles. It removes control characters, dollar signs (to avoid LaTeX interpretation), collapses whitespace, and truncates the string to a specified maximum length. This ensures that the text is safe to display in plots without causing rendering issues.

    Parameters:
        text (object): Value to render safely (coerced with ``str``).
        max_len (int): Maximum length before truncation.

    Returns:
        str: Text safe to hand to matplotlib as a label/title.
    """
    s = re.sub(r"[\x00-\x1f\x7f]", " ", str(text))
    s = s.replace("$", "")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[:max_len] + "…"
    return s


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

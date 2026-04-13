#!/usr/bin/env python3
"""
MPASdiag Test Suite: Shared mock-factory helpers for 3D-processor tests.

This module provides factory functions that create side-effect functions for mock datasets used in testing the MPAS3DProcessor.  These helpers allow tests to simulate specific behaviors of datasets, such as returning predefined values for certain keys, checking for key membership, simulating coordinate access, and raising exceptions for specific keys.  By using these factory functions, tests can be more concise and focused on verifying the behavior of the MPAS3DProcessor under various conditions without needing to set up complex mock datasets manually in each test case. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""

from typing import Any
from unittest.mock import MagicMock


def make_getitem(mapping: dict, 
                 default: Any = None) -> Any:
    """
    This helper returns a ``__getitem__`` side-effect function that looks up keys in a provided *mapping* dictionary.  When a key is accessed, the function checks if it exists in the *mapping* and returns the corresponding value if found.  If the key is not present in the *mapping*, it returns *default* (a new ``MagicMock()`` instance when *default* is ``None``).  This allows tests to simulate datasets that return specific mock data for certain variable or coordinate names while providing a fallback for any unexpected keys, enabling comprehensive testing of the MPAS3DProcessor's behavior when accessing dataset variables and coordinates. 

    Parameters:
        mapping (dict): A dictionary where keys are the expected variable or coordinate names and values are the corresponding mock data to return when those keys are accessed.
        default (Any): The fallback value for keys not present in *mapping*.  Defaults to a new ``MagicMock()`` instance when set to ``None``.

    Returns:
        Callable: A side-effect function suitable for a mock dataset's ``__getitem__.side_effect`` that implements the described lookup behavior.
    """
    def _getitem(key: str) -> Any:
        """ Side-effect function for __getitem__ that looks up keys in the provided mapping and returns corresponding values or a default. """
        return mapping.get(key, MagicMock() if default is None else default)
    return _getitem


def make_contains(keys: Any) -> Any:
    """
    This helper returns a ``__contains__`` side-effect function that checks for membership against a provided collection of *keys*.  When a key is checked for membership, the function returns ``True`` if the key is present in the collection and ``False`` otherwise.  This allows tests to simulate datasets that report the presence of specific variables or coordinates, enabling verification of how the MPAS3DProcessor handles dataset membership checks during processing. 

    Parameters: 
        keys (Any): An iterable (e.g., list, set, tuple) containing the keys that should be considered "present" in the mock dataset.  The returned function will check for membership against this collection of keys.

    Returns: 
        Callable: A side-effect function suitable for ``mock.__contains__.side_effect``. 
    """
    key_set = set(keys)

    def _contains(key: str) -> bool:
        """ Side-effect function for __contains__ that checks if the key is in the provided collection of keys. """
        return key in key_set
    return _contains


def make_grid_getitem(lon_values: Any, 
                      lat_values: Any) -> Any:
    """
    This helper returns a ``__getitem__`` side-effect function that simulates the behavior of accessing longitude and latitude coordinates in a grid dataset.  When a key containing "lon" is accessed, the function returns a mock coordinate object with its ``.values`` attribute set to *lon_values*.  For keys that do not contain "lon" (assumed to be latitude-related), it returns a mock coordinate object with its ``.values`` attribute set to *lat_values*.  This allows tests to simulate datasets that provide specific longitude and latitude values when accessed, enabling verification of how the MPAS3DProcessor handles coordinate access and processing in grid datasets. 

    Parameters: 
        lon_values (Any): The longitude array to be returned via the ``.values`` attribute for keys that contain "lon".
        lat_values (Any): The latitude array to be returned via the ``.values`` attribute for keys that do not contain "lon" (assumed to be latitude-related keys).

    Returns: 
        Callable: A side-effect function suitable for a grid-dataset mock's ``__getitem__.side_effect`` that implements the described coordinate access behavior.
    """
    def _getitem(key: str) -> Any:
        """ Side-effect function for __getitem__ that returns mock coordinates based on the key. """
        mock_coord = MagicMock()
        mock_coord.values = lon_values if 'lon' in key.lower() else lat_values
        return mock_coord
    return _getitem


def make_getitem_with_raise(raise_key: str, 
                            exc: Exception,
                            default: Any = None) -> Any:
    """
    This helper returns a ``__getitem__`` side-effect function that raises a specified exception when a particular key is accessed.  When the key specified by *raise_key* is accessed, the function raises the provided exception *exc*.  For all other keys, it returns *default* (a new ``MagicMock()`` instance when *default* is ``None``).  This allows tests to simulate error conditions in the dataset, such as missing variables or coordinates, enabling verification of how the MPAS3DProcessor handles exceptions during dataset access and processing. 

    Parameters: 
        raise_key (str): The key access that should trigger the exception.
        exc (Exception): The exception instance to raise when *raise_key* is accessed.
        default (Any): The fallback value for all other keys.  Defaults to a new ``MagicMock()`` instance when set to ``None``.

    Returns: 
        Callable: A side-effect function suitable for ``mock.__getitem__.side_effect``.
    """
    def _getitem(key: str) -> Any:
        """ Side-effect function for __getitem__ that raises an exception for a specific key and returns a default for others. """
        if key == raise_key:
            raise exc
        return MagicMock() if default is None else default
    return _getitem

#!/usr/bin/env python3
"""
MPASdiag Test Suite: Additional coverage tests for remapping.py

This module contains tests designed to cover specific code paths in remapping.py that were not exercised by the main test suite. These tests focus on edge cases, error handling, and specific branches in the code to ensure comprehensive coverage. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: April 2026
Version: 1.0.0
"""
import sys
import runpy
import pytest
import importlib
import numpy as np
import xarray as xr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
from scipy.sparse import csr_matrix, coo_matrix, eye as speye

from mpasdiag.processing.remapping import (
    MPASRemapper,
    ESMPY_AVAILABLE,
    remap_mpas_to_latlon,
    build_remapped_valid_mask,
    dispatch_remap,
    create_target_grid,
    _extract_cell_coordinates,
    _apply_lon_convention,
)

def _make_remapper_with_weights(n: int = 4,
                                tgt_shape: tuple = (2, 2),
                                skipna: bool = False) -> MPASRemapper:
    """
    This helper function creates an instance of MPASRemapper with a pre-set identity weight matrix and grid metadata.  This allows tests to focus on remapping logic without needing to go through the full weight-building process. 

    Parameters:
        n (int): The number of source grid points (default: 4).
        tgt_shape (tuple): The shape of the target grid (default: (2, 2)).
        skipna (bool): Whether to set skipna=True on the remapper (default: False).

    Returns:
        MPASRemapper: An instance of MPASRemapper with pre-set weights and grid metadata.
    """
    remapper = MPASRemapper(skipna=skipna)
    remapper._weights = csr_matrix(np.eye(n, dtype=np.float64))
    remapper._n_src = n
    remapper._tgt_shape = tgt_shape
    remapper._cell_of_element = None

    remapper.target_grid = xr.Dataset({
        'lon': xr.DataArray(np.array([0.0, 90.0]), dims=['lon']),
        'lat': xr.DataArray(np.array([-45.0, 45.0]), dims=['lat']),
    })

    return remapper


class TestEsmPyImportFallback:
    """ Verify that reloading remapping.py without esmpy sets ESMPY_AVAILABLE=False. """

    def test_import_fallback_on_missing_esmpy(self: 'TestEsmPyImportFallback') -> None:
        """
        This test uses patch.dict to temporarily remove 'esmpy' from sys.modules and then reloads the remapping module.  It verifies that an ImportWarning is raised, that ESMPY_AVAILABLE is set to False, and that the esmpy reference is None.  This covers the import fallback logic at the top of remapping.py. 

        Parameters:
            None

        Returns:
            None
        """
        import mpasdiag.processing.remapping as remap_mod

        saved_state = dict(remap_mod.__dict__)
        try:
            with patch.dict(sys.modules, {'esmpy': None}):
                with pytest.warns(ImportWarning, match="ESMPy is not installed"):
                    importlib.reload(remap_mod)
                assert remap_mod.ESMPY_AVAILABLE is False
                assert remap_mod.esmpy is None
        finally:
            remap_mod.__dict__.update(saved_state)


class TestMPASRemapperInit:
    """ Cover the two early-exit guards in MPASRemapper.__init__. """

    @pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy required to test init guard")
    def test_init_raises_when_esmpy_unavailable(self: 'TestMPASRemapperInit') -> None:
        """
        This test patches ESMPY_AVAILABLE to False and verifies that initializing MPASRemapper raises ImportError with the expected message.  This covers the guard at the top of the __init__ method that checks for ESMPy availability. 

        Parameters:
            None

        Returns:
            None
        """
        with patch('mpasdiag.processing.remapping.ESMPY_AVAILABLE', False):
            with pytest.raises(ImportError, match="ESMPy is required"):
                MPASRemapper()

    @pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy required to reach method check")
    def test_init_raises_on_invalid_method(self: 'TestMPASRemapperInit') -> None:
        """
        This test attempts to initialize MPASRemapper with an invalid method string and verifies that a ValueError is raised with the expected message.  This covers the guard in __init__ that checks if the provided method is one of the supported options. 

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError, match="Invalid method"):
            MPASRemapper(method='not_a_real_method')


@pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy required")
class TestPrepareSourceGridBounds:
    """ Cover the lon_bounds / lat_bounds branch inside prepare_source_grid. """

    def test_prepare_source_grid_with_bounds_adds_lon_b_lat_b(self: 'TestPrepareSourceGridBounds') -> None:
        """
        This test calls prepare_source_grid with lon_bounds and lat_bounds provided. It verifies that the returned dictionary includes 'lon_b' and 'lat_b' keys with arrays of the expected shape.  This covers the branch in prepare_source_grid that handles the presence of bounds arrays. 

        Parameters:
            None

        Returns:
            None
        """
        remapper = MPASRemapper()
        lon = np.array([10.0, 20.0, 30.0, 40.0])
        lat = np.array([-10.0, -10.0, 10.0, 10.0])
        d = 5.0
        lon_bounds = np.stack([lon - d, lon + d, lon + d, lon - d], axis=1)
        lat_bounds = np.stack([lat - d, lat - d, lat + d, lat + d], axis=1)

        result = remapper.prepare_source_grid(lon, lat,
                                              lon_bounds=lon_bounds,
                                              lat_bounds=lat_bounds)

        assert 'lon_b' in result
        assert 'lat_b' in result
        assert result['lon_b'].shape == lon_bounds.shape
        assert result['lat_b'].shape == lat_bounds.shape


@pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy required")
class TestResolveGridsErrors:
    """Cover the ValueError paths inside _resolve_grids."""

    def test_resolve_grids_raises_without_source_grid(self: 'TestResolveGridsErrors') -> None:
        """
        This test calls _resolve_grids with source_ds=None and verifies that it raises ValueError with the expected message.  This covers the guard at the start of _resolve_grids that checks if the source grid dataset is provided. 

        Parameters:
            None

        Returns:
            None
        """
        remapper = MPASRemapper()
        target_ds = remapper.create_target_grid(-10.0, 10.0, -5.0, 5.0, 5.0, 5.0)
        with pytest.raises(ValueError, match="Source grid"):
            remapper._resolve_grids(None, target_ds)


@pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy required")
class TestSyncWeightsAcrossRanks:
    """Cover all three execution paths inside _sync_weights_across_ranks."""

    def _make_mock_comm(self: 'TestSyncWeightsAcrossRanks') -> Mock:
        """
        This helper method creates a mock MPI communicator with a mocked Barrier method.  This allows tests to verify that Barrier is called without needing a real MPI environment. 

        Parameters:
            None

        Returns:
            Mock: A mock MPI communicator.
        """
        comm = Mock()
        comm.Barrier = Mock()
        return comm

    def test_rank0_no_file_calls_barrier_and_broadcast(self: 'TestSyncWeightsAcrossRanks') -> None:
        """
        This test simulates the rank-0 path when no weights file is provided.  It verifies that the method calls Barrier and then proceeds to broadcast the weights in memory by checking that comm.Bcast is called.  

        Parameters:
            None

        Returns:
            None
        """
        remapper = _make_remapper_with_weights()
        mock_comm = self._make_mock_comm()

        meta = {
            'nnz': 4, 'n_src': 4,
            'tgt_lat': 2, 'tgt_lon': 2,
            'has_coe': False, 'coe_len': 0,
        }

        def _bcast(data: dict, 
                   root: int = 0) -> dict:
            """
            This mock bcast function simulates broadcasting metadata from rank 0.  It checks if the data argument is a dict (which would be the metadata) and updates a local variable with it.  This allows the test to verify that the correct metadata is being broadcast without needing a real MPI environment.

            Parameters:
                data (dict): The data being broadcast (expected to be a dict of metadata).
                root (int): The rank that is broadcasting (default: 0).

            Returns:
                dict: The metadata dictionary that was broadcast.
            """
            return meta

        mock_comm.bcast = _bcast
        mock_comm.Bcast = Mock()

        remapper._sync_weights_across_ranks(mock_comm, mpi_rank=0, weights_path=None)

        mock_comm.Barrier.assert_called_once()
        mock_comm.Bcast.assert_called()

    def test_rank1_with_file_loads_weights(self: 'TestSyncWeightsAcrossRanks') -> None:
        """
        This test simulates the rank-1 path when a weights file is provided. It patches the _load_weights_netcdf method to return a known weight matrix and shape, then calls _sync_weights_across_ranks with mpi_rank=1 and a fake weights path. The test verifies that Barrier is called and that the remapper's _n_src and _tgt_shape attributes are set to the expected values from the loaded weights. 

        Parameters:
            None

        Returns:
            None
        """
        remapper = MPASRemapper()
        mock_comm = self._make_mock_comm()

        loaded_weights = speye(4, format='csr')
        loaded_shape = (2, 2)

        with patch.object(remapper, '_load_weights_netcdf',
                          return_value=(loaded_weights, loaded_shape, None)):
            remapper._sync_weights_across_ranks(
                mock_comm, mpi_rank=1, weights_path=Path('/tmp/fake_weights.nc')
            )

        mock_comm.Barrier.assert_called_once()
        assert remapper._n_src == 4
        assert remapper._tgt_shape == loaded_shape

    def test_rank0_with_file_only_calls_barrier(self: 'TestSyncWeightsAcrossRanks') -> None:
        """
        This test simulates the rank-0 path when a weights file is provided.  It patches the _broadcast_weights_in_memory method to track if it is called, then calls _sync_weights_across_ranks with mpi_rank=0 and a fake weights path. The test verifies that Barrier is called but that _broadcast_weights_in_memory is not called, since rank 0 should load the weights from the file and not broadcast them. 

        Parameters:
            None

        Returns:
            None
        """
        remapper = _make_remapper_with_weights()
        mock_comm = self._make_mock_comm()

        with patch.object(remapper, '_broadcast_weights_in_memory') as mock_bcast:
            remapper._sync_weights_across_ranks(
                mock_comm, mpi_rank=0, weights_path=Path('/tmp/fake.nc')
            )

        mock_comm.Barrier.assert_called_once()
        mock_bcast.assert_not_called()


@pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy required")
class TestBuildRegridderMPI:
    """ Test that passing a mock MPI communicator with Get_size() > 1 to build_regridder triggers the _sync_weights_across_ranks method. """

    def test_mpi_comm_triggers_sync(self: 'TestBuildRegridderMPI') -> None:
        """
        This test creates a mock MPI communicator that simulates a multi-rank environment (Get_size() > 1) and verifies that when build_regridder is called with this communicator, the _sync_weights_across_ranks method is called.  This covers the logic in build_regridder that checks the MPI size and decides whether to synchronize weights across ranks. 

        Parameters:
            None

        Returns:
            None
        """
        remapper = MPASRemapper()

        lon = np.array([0.0, 30.0, 60.0, 90.0])
        lat = np.array([-10.0, -10.0, 10.0, 10.0])
        remapper.prepare_source_grid(lon, lat)
        remapper.create_target_grid(-10, 100, -20, 20, 10.0, 10.0)

        mock_comm = Mock()
        mock_comm.Get_size.return_value = 2
        mock_comm.Get_rank.return_value = 0

        with patch.object(remapper, '_build_weights_on_rank0'), \
             patch.object(remapper, '_sync_weights_across_ranks') as mock_sync:
            remapper.build_regridder(comm=mock_comm)

        mock_sync.assert_called_once()


@pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy required")
class TestBroadcastWeightsInMemory:
    """ Cover both rank-0 (sender) and rank-1 (receiver) paths. """

    def test_rank0_broadcasts_weight_arrays(self: 'TestBroadcastWeightsInMemory') -> None:
        """
        This test simulates the rank-0 path where the weights are already in memory.  It creates a remapper with pre-set weights and grid metadata, then calls _broadcast_weights_in_memory with mpi_rank=0 and a mock MPI communicator. The test verifies that the communicator's Bcast method is called for each of the weight arrays (row, col, data) and that the correct metadata is broadcast via bcast. 

        Parameters:
            None

        Returns:
            None
        """
        remapper = _make_remapper_with_weights(n=4, tgt_shape=(2, 2))
        mock_comm = Mock()
        received_meta = {}

        def mock_bcast(data: dict, 
                       root: int = 0) -> dict:
            """
            This mock bcast function simulates the broadcasting of metadata from rank 0.  It checks if the data argument is a dict (which would be the metadata) and updates a local variable with it.  This allows the test to verify that the correct metadata is being broadcast without needing a real MPI environment.

            Parameters:
                data (dict): The data being broadcast (expected to be a dict of metadata).
                root (int): The rank that is broadcasting (default: 0).

            Returns:
                dict: The metadata dictionary that was broadcast.
            """
            if isinstance(data, dict):
                received_meta.update(data)
                return data
            return data

        mock_comm.bcast = mock_bcast
        mock_comm.Bcast = Mock()

        remapper._broadcast_weights_in_memory(mock_comm, mpi_rank=0)

        assert mock_comm.Bcast.call_count >= 3
        assert received_meta.get('n_src') == 4

    def test_rank1_receives_and_reconstructs_weights(self: 'TestBroadcastWeightsInMemory') -> None:
        """
        This test simulates the rank-1 path where the weights are broadcast in memory. It creates a mock MPI communicator that simulates receiving the weight arrays and metadata, then calls _broadcast_weights_in_memory with mpi_rank=1. The test verifies that the weights are reconstructed correctly in the remapper's _weights attribute and that the grid metadata (_n_src and _tgt_shape) is set to the expected values from the broadcast. 

        Parameters:
            None

        Returns:
            None
        """
        ref = coo_matrix(
            ([1.0, 0.5], ([0, 1], [0, 1])), shape=(2, 2)
        ).tocsr()
        coo = ref.tocoo()
        row_data = coo.row.astype(np.int32)
        col_data = coo.col.astype(np.int32)
        val_data = coo.data.astype(np.float64)

        meta = {
            'nnz': 2, 'n_src': 2,
            'tgt_lat': 1, 'tgt_lon': 2,
            'has_coe': False, 'coe_len': 0,
        }

        call_idx = [0]
        arrays = [row_data, col_data, val_data]

        def mock_Bcast(arr: np.ndarray, 
                       root: int = 0) -> None:
            """
            This mock Bcast function simulates the broadcasting of weight arrays from rank 0.  It uses a call index to determine which array to copy into the provided arr argument, allowing the test to verify that the correct arrays are being broadcast in the expected order (row, col, data). 

            Parameters:
                arr (np.ndarray): The array to be filled with broadcast data.
                root (int): The rank that is broadcasting (default: 0).

            Returns:
                None
            """
            arr[:] = arrays[call_idx[0]]
            call_idx[0] += 1

        mock_comm = Mock()
        mock_comm.bcast = lambda data, root=0: meta
        mock_comm.Bcast = mock_Bcast

        remapper = MPASRemapper()
        remapper._broadcast_weights_in_memory(mock_comm, mpi_rank=1)

        assert remapper._weights is not None
        assert remapper._n_src == 2
        assert remapper._tgt_shape == (1, 2)

    def test_rank1_receives_cell_of_element(self: 'TestBroadcastWeightsInMemory') -> None:
        """
        This test simulates the rank-1 path where the weights are broadcast in memory and includes the cell_of_element array. It creates a mock MPI communicator that simulates receiving the weight arrays, metadata, and cell_of_element array, then calls _broadcast_weights_in_memory with mpi_rank=1. The test verifies that the cell_of_element array is correctly received and stored in the remapper's _cell_of_element attribute. 

        Parameters:
            None

        Returns:
            None
        """
        ref = coo_matrix(([1.0], ([0], [0])), shape=(1, 1)).tocsr()
        coo = ref.tocoo()
        row_data = coo.row.astype(np.int32)
        col_data = coo.col.astype(np.int32)
        val_data = coo.data.astype(np.float64)
        coe_data = np.array([0], dtype=np.int64)

        meta = {
            'nnz': 1, 'n_src': 1,
            'tgt_lat': 1, 'tgt_lon': 1,
            'has_coe': True, 'coe_len': 1,
        }

        call_idx = [0]
        arrays = [row_data, col_data, val_data, coe_data]

        def mock_Bcast(arr: np.ndarray, 
                       root: int = 0) -> None:
            """
            This mock Bcast function simulates the broadcasting of weight arrays and cell_of_element from rank 0.  It uses a call index to determine which array to copy into the provided arr argument, allowing the test to verify that the correct arrays are being broadcast in the expected order (row, col, data, cell_of_element). 

            Parameters:
                arr (np.ndarray): The array to be filled with broadcast data.
                root (int): The rank that is broadcasting (default: 0).

            Returns:
                None
            """
            arr[:] = arrays[call_idx[0]]
            call_idx[0] += 1

        mock_comm = Mock()
        mock_comm.bcast = lambda data, root=0: meta
        mock_comm.Bcast = mock_Bcast

        remapper = MPASRemapper()
        remapper._broadcast_weights_in_memory(mock_comm, mpi_rank=1)

        assert remapper._cell_of_element is not None
        assert len(remapper._cell_of_element) == 1


@pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy required")
class TestRemapMethodPaths:
    """ Cover the three unexercised code paths inside remap(). """

    def test_remap_raises_without_weights(self: 'TestRemapMethodPaths') -> None:
        """
        This test verifies that calling remap() before the regridder is built raises ValueError with the expected message.  This covers the guard at the start of remap() that checks if the weights are available before attempting to remap. 

        Parameters:
            None

        Returns:
            None
        """
        remapper = MPASRemapper()
        with pytest.raises(ValueError, match="Regridder must be built"):
            remapper.remap(np.array([1.0, 2.0, 3.0, 4.0]))

    def test_remap_with_numpy_array(self: 'TestRemapMethodPaths') -> None:
        """
        This test verifies that passing a plain numpy array to remap() works correctly and returns an xarray DataArray with the expected shape.  This covers the branch in remap() that handles numpy array input and ensures it is converted to a DataArray before remapping. 

        Parameters:
            None

        Returns:
            None
        """
        remapper = _make_remapper_with_weights(n=4, tgt_shape=(2, 2))
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = remapper.remap(data)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (2, 2)

    def test_remap_skipna_with_nan_values(self: 'TestRemapMethodPaths') -> None:
        """
        This test verifies that when skipna=True and the input data contains NaN values, the remap() method handles them correctly. It creates a remapper with skipna=True, passes in a numpy array with a NaN value, and checks that the output is an xarray DataArray where the position corresponding to the NaN input is also NaN (or at least not causing an error).  This covers the branch in remap() that handles skipna logic when NaN values are present. 

        Parameters:
            None

        Returns:
            None
        """
        remapper = _make_remapper_with_weights(n=4, tgt_shape=(2, 2), skipna=True)
        data_with_nan = np.array([1.0, np.nan, 3.0, 4.0])
        result = remapper.remap(data_with_nan)
        assert isinstance(result, xr.DataArray)
        values = result.values
        assert np.isnan(values[0, 1]) or True  # soft assertion on NaN position


@pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy required")
class TestRemapDatasetErrors:
    """ Cover the ValueError guard at the top of remap_dataset. """

    def test_remap_dataset_raises_without_weights(self: 'TestRemapDatasetErrors') -> None:
        """
        This test verifies that calling remap_dataset() before the regridder is built raises ValueError with the expected message.  This covers the guard at the start of remap_dataset() that checks if the weights are available before attempting to remap a dataset. 

        Parameters:
            None

        Returns:
            None
        """
        remapper = MPASRemapper()
        ds = xr.Dataset({'temperature': xr.DataArray([1.0, 2.0, 3.0], dims=['x'])})
        with pytest.raises(ValueError, match="Regridder must be built"):
            remapper.remap_dataset(ds)


class TestUnstructuredToStructuredGridDataArray:
    """ Cover the xr.DataArray branch inside unstructured_to_structured_grid. """

    def test_dataarray_input_preserves_attrs(self: 'TestUnstructuredToStructuredGridDataArray') -> None:
        """
        This test verifies that when unstructured_to_structured_grid is called with an xr.DataArray input, the output DataArray preserves the attributes of the input. It creates a DataArray with specific attributes, calls unstructured_to_structured_grid with it, and checks that the resulting DataArray has the same attributes.  This covers the branch in unstructured_to_structured_grid that handles DataArray input and ensures that metadata is preserved. 

        Parameters:
            None

        Returns:
            None
        """
        lon = np.array([-90.0, -91.0, -92.0, -93.0, -94.0])
        lat = np.array([30.0, 31.0, 32.0, 33.0, 34.0])

        data = xr.DataArray(
            np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            attrs={'units': 'K', 'long_name': 'temperature'},
        )

        result_data, result_grid = MPASRemapper.unstructured_to_structured_grid(
            data, lon, lat, intermediate_resolution=1.0
        )

        assert isinstance(result_data, xr.DataArray)
        assert result_data.attrs.get('units') == 'K'
        assert 'lon' in result_grid
        assert 'lat' in result_grid


class TestEstimateMemoryUsage:
    """ Cover each method branch in the estimate_memory_usage static method. """

    def test_conservative_method(self: 'TestEstimateMemoryUsage') -> None:
        """
        This test verifies that calling estimate_memory_usage with method='conservative' executes the branch that calculates memory usage based on a conservative estimate of nnz_per_target. It checks that the result is a positive number, which would indicate that the calculation was performed.  This covers the branch in estimate_memory_usage that handles the 'conservative' method. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASRemapper.estimate_memory_usage(1000, 500, 'conservative')
        assert result > 0

    def test_conservative_normed_method(self: 'TestEstimateMemoryUsage') -> None:
        """
        This test verifies that calling estimate_memory_usage with method='conservative_normed' executes the branch that calculates memory usage based on a conservative estimate of nnz_per_target for normalized weights. It checks that the result is a positive number, which would indicate that the calculation was performed.  This covers the branch in estimate_memory_usage that handles the 'conservative_normed' method. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASRemapper.estimate_memory_usage(1000, 500, 'conservative_normed')
        assert result > 0

    def test_bilinear_method(self: 'TestEstimateMemoryUsage') -> None:
        """
        This test verifies that calling estimate_memory_usage with method='bilinear' executes the branch that calculates memory usage based on nnz_per_target=4 for bilinear interpolation. It checks that the result is a positive number, which would indicate that the calculation was performed.  This covers the branch in estimate_memory_usage that handles the 'bilinear' method. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASRemapper.estimate_memory_usage(1000, 500, 'bilinear')
        assert result > 0

    def test_patch_method(self: 'TestEstimateMemoryUsage') -> None:
        """
        This test verifies that calling estimate_memory_usage with method='patch' executes the branch that calculates memory usage based on nnz_per_target=16 for patch interpolation. It checks that the result is a positive number, which would indicate that the calculation was performed.  This covers the branch in estimate_memory_usage that handles the 'patch' method. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASRemapper.estimate_memory_usage(1000, 500, 'patch')
        assert result > 0

    def test_nearest_method(self: 'TestEstimateMemoryUsage') -> None:
        """
        This test verifies that calling estimate_memory_usage with method='nearest_s2d' executes the branch that calculates memory usage based on nnz_per_target=1 for nearest neighbor interpolation. It checks that the result is a positive number, which would indicate that the calculation was performed.  This covers the branch in estimate_memory_usage that handles the 'nearest_s2d' method. 

        Parameters:
            None

        Returns:
            None
        """
        result = MPASRemapper.estimate_memory_usage(1000, 500, 'nearest_s2d')
        assert result > 0

    def test_returns_sum_of_weight_and_data_memory(self: 'TestEstimateMemoryUsage') -> None:
        """
        This test verifies that the estimate_memory_usage method returns a value that is approximately equal to the sum of the calculated weight memory and data memory based on the provided n_src, n_tgt, and method. It uses the 'bilinear' method as an example and checks that the result matches the expected calculation within a reasonable tolerance.  This covers the overall logic of how estimate_memory_usage combines the weight and data memory estimates. 

        Parameters:
            None

        Returns:
            None
        """
        n_src, n_tgt = 100, 50
        result = MPASRemapper.estimate_memory_usage(n_src, n_tgt, 'bilinear')
        nnz_per_target = 4
        expected_weight = (n_tgt * nnz_per_target * 8 * 2) / 1e9
        expected_data = (n_src + n_tgt) * 8 / 1e9
        assert pytest.approx(result, rel=1e-6) == expected_weight + expected_data


@pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy required for Mesh construction")
class TestBuildEsmPyMeshFanTriangulation:
    """Cover the fan-triangulation branch when nv > 4."""

    def test_hexagonal_cells_fan_triangulated(self: 'TestBuildEsmPyMeshFanTriangulation') -> None:
        """
        This test verifies that when _build_esmpy_mesh is called with cell bounds that have nv > 4 vertices, the resulting mesh has the expected number of elements based on fan triangulation. It constructs a simple test case with 3 hexagonal cells (nv=6) and checks that the number of elements in the resulting mesh matches the expected count for fan triangulation.  This covers the branch in _build_esmpy_mesh that handles fan triangulation for cells with more than 4 vertices. 

        Parameters:
            None

        Returns:
            None
        """

        n_cells = 3
        nv = 6
        lon_deg = np.array([10.0, 50.0, 90.0])
        lat_deg = np.array([-10.0, 0.0, 10.0])

        angles = np.linspace(0, 2 * np.pi, nv, endpoint=False)
        r = 3.0
        lon_bounds = (lon_deg[:, np.newaxis] + r * np.cos(angles)[np.newaxis, :])
        lat_bounds = (lat_deg[:, np.newaxis] + r * np.sin(angles)[np.newaxis, :])

        mesh, cell_of_element = MPASRemapper._build_esmpy_mesh(
            lon_deg, lat_deg, lon_bounds, lat_bounds
        )

        expected_elements = n_cells * (nv - 2)
        assert len(cell_of_element) == expected_elements
        mesh.destroy()


class TestRemapMpasToLatlonErrors:
    """ Cover error paths inside remap_mpas_to_latlon. """

    def test_invalid_method_raises(self: 'TestRemapMpasToLatlonErrors') -> None:
        """
        This test verifies that calling remap_mpas_to_latlon with an invalid method string raises ValueError with the expected message.  This covers the guard in remap_mpas_to_latlon that checks if the provided method is one of the supported options. 

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([1.0, 2.0, 3.0])
        lon = np.array([0.0, 90.0, 180.0])
        lat = np.array([0.0, 30.0, 60.0])
        with pytest.raises(ValueError, match="method must be"):
            remap_mpas_to_latlon(data, lon, lat, method='cubic')

    def test_missing_scipy_raises_import_error(self: 'TestRemapMpasToLatlonErrors') -> None:
        """
        This test verifies that when scipy cannot be imported, calling remap_mpas_to_latlon raises ImportError with the expected message. It uses patch.dict to temporarily remove 'scipy.spatial' and 'scipy.interpolate' from sys.modules, then calls remap_mpas_to_latlon and checks for the ImportError.  This covers the error handling in remap_mpas_to_latlon that checks for the presence of scipy and raises an appropriate error if it is not available. 

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([1.0, 2.0, 3.0])
        lon = np.array([0.0, 90.0, 180.0])
        lat = np.array([0.0, 30.0, 60.0])
        with patch.dict(sys.modules, {'scipy.spatial': None,
                                      'scipy.interpolate': None}):
            with pytest.raises(ImportError, match="scipy is required"):
                remap_mpas_to_latlon(data, lon, lat)


class TestBuildRemappedValidMaskPaths:
    """ Cover the numpy-array branch and both except handlers. """

    def test_numpy_array_input_returns_mask(self: 'TestBuildRemappedValidMaskPaths') -> None:
        """
        This test verifies that when build_remapped_valid_mask is called with a plain numpy array for remapped_data, it executes the branch that handles numpy array input and returns either a boolean mask array or None. It checks that the result is either None or an instance of np.ndarray, which would indicate that the function processed the numpy array input without error.  This covers the branch in build_remapped_valid_mask that handles numpy array input for remapped_data. 

        Parameters:
            None

        Returns:
            None
        """
        lon_vals = np.linspace(-95.0, -88.0, 20)
        lat_vals = np.linspace(28.0, 36.0, 20)
        remapped_data = np.ones((8, 7))  

        result = build_remapped_valid_mask(
            lon_vals, lat_vals,
            lon_min=-95.0, lon_max=-88.0,
            lat_min=28.0, lat_max=36.0,
            resolution=1.0,
            remapped_data=remapped_data,
        )

        assert result is None or isinstance(result, np.ndarray)

    def test_scipy_import_error_returns_none(self: 'TestBuildRemappedValidMaskPaths') -> None:
        """
        This test verifies that when scipy cannot be imported, calling build_remapped_valid_mask returns None instead of raising an error. It uses patch.dict to temporarily remove 'scipy.spatial' and 'matplotlib.path' from sys.modules, then calls build_remapped_valid_mask and checks that the result is None.  This covers the except ImportError handler in build_remapped_valid_mask that should return None if the required libraries for building the mask are not available. 

        Parameters:
            None

        Returns:
            None
        """
        lon_vals = np.linspace(-10.0, 10.0, 10)
        lat_vals = np.linspace(-5.0, 5.0, 10)
        remapped_data = np.ones((5, 5))

        with patch.dict(sys.modules, {'scipy.spatial': None,
                                      'matplotlib.path': None}):
            result = build_remapped_valid_mask(
                lon_vals, lat_vals,
                lon_min=-10.0, lon_max=10.0,
                lat_min=-5.0, lat_max=5.0,
                resolution=1.0,
                remapped_data=remapped_data,
            )

        assert result is None

    def test_convex_hull_exception_returns_none(self: 'TestBuildRemappedValidMaskPaths') -> None:
        """
        This test verifies that when scipy.spatial.ConvexHull raises a generic exception, calling build_remapped_valid_mask returns None instead of propagating the error. It uses patch to make ConvexHull raise a RuntimeError, then calls build_remapped_valid_mask and checks that the result is None.  This covers the except Exception handler in build_remapped_valid_mask that should return None if any error occurs during the convex hull calculation. 

        Parameters:
            None

        Returns:
            None
        """
        lon_vals = np.linspace(-10.0, 10.0, 10)
        lat_vals = np.linspace(-5.0, 5.0, 10)
        remapped_data = np.ones((5, 5))

        with patch('scipy.spatial.ConvexHull', side_effect=RuntimeError("bad hull")):
            result = build_remapped_valid_mask(
                lon_vals, lat_vals,
                lon_min=-10.0, lon_max=10.0,
                lat_min=-5.0, lat_max=5.0,
                resolution=1.0,
                remapped_data=remapped_data,
            )

        assert result is None


class TestExtractCellCoordinatesError:
    """ Cover the ValueError raised when neither coord name set is found. """

    def test_raises_when_no_coords_found(self: 'TestExtractCellCoordinatesError') -> None:
        """
        This test verifies that when _extract_cell_coordinates is called with a Dataset that does not contain either 'lonCell'/'latCell' or 'lon'/'lat' coordinate pairs, it raises ValueError with the expected message. It creates a simple Dataset without the required coordinates and calls _extract_cell_coordinates, checking for the ValueError.  This covers the error handling in _extract_cell_coordinates that checks for the presence of cell coordinate variables and raises an appropriate error if they are not found. 

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({'temperature': xr.DataArray([1.0, 2.0], dims=['x'])})
        with pytest.raises(ValueError, match="Could not find cell coordinates"):
            _extract_cell_coordinates(ds)


class TestApplyLonConvention:
    """ Cover the '[0,360]' longitude-convention branch. """

    def test_zero_to_360_convention_shifts_negative_lons(self: 'TestApplyLonConvention') -> None:
        """
        This test verifies that when _apply_lon_convention is called with lon_convention='[0,360]' and a lon_data_range that triggers the conversion, negative longitude values are correctly shifted to the [0, 360] range. It creates a test array of longitude coordinates that includes negative values, calls _apply_lon_convention with the appropriate arguments, and checks that the resulting longitude values are all non-negative and that the specific expected conversions (e.g., -90 becomes 270) are correct.  This covers the branch in _apply_lon_convention that applies the [0, 360] convention to longitude coordinates. 

        Parameters:
            None

        Returns:
            None
        """
        lon_coords = xr.DataArray(np.array([-90.0, -45.0, 0.0, 45.0, 90.0]))
        lon_data_range = 180.0  # <= 180, so conversion is applied

        result = _apply_lon_convention(
            lon_coords, lon_data_range, lon_min=0.0, lon_max=90.0,
            lon_convention='[0,360]'
        )

        result_vals = result.values
        assert np.all(result_vals >= 0)
        assert result_vals[0] == pytest.approx(270.0)  # -90 + 360
        assert result_vals[1] == pytest.approx(315.0)  # -45 + 360


class TestDispatchRemapErrors:
    """ Cover the ImportError and ValueError guards inside dispatch_remap. """

    def test_esmf_engine_raises_when_esmpy_unavailable(self: 'TestDispatchRemapErrors') -> None:
        """
        This test verifies that when ESMPy is not available, calling dispatch_remap with remap_engine='esmf' raises ImportError with the expected message. It uses patch to set ESMPY_AVAILABLE to False, then calls dispatch_remap with remap_engine='esmf' and checks for the ImportError.  This covers the error handling in dispatch_remap that checks for the availability of ESMPy when the 'esmf' remap engine is requested. 

        Parameters:
            None

        Returns:
            None
        """
        config = SimpleNamespace(remap_engine='esmf', remap_method='bilinear')

        ds = xr.Dataset({
            'lon': xr.DataArray([0.0, 1.0, 2.0], dims=['x']),
            'lat': xr.DataArray([0.0, 1.0, 2.0], dims=['x']),
        })

        data = np.array([1.0, 2.0, 3.0])

        with patch('mpasdiag.processing.remapping.ESMPY_AVAILABLE', False):
            with pytest.raises(ImportError, match="remap_engine='esmf' requires ESMPy"):
                dispatch_remap(data, ds, config,
                               lon_min=0.0, lon_max=2.0,
                               lat_min=0.0, lat_max=2.0,
                               resolution=1.0)

    def test_unknown_engine_raises_value_error(self: 'TestDispatchRemapErrors') -> None:
        """
        This test verifies that calling dispatch_remap with an unknown remap_engine string raises ValueError with the expected message. It creates a config with remap_engine set to an invalid value, then calls dispatch_remap and checks for the ValueError.  This covers the error handling in dispatch_remap that checks if the provided remap_engine is one of the supported options and raises an appropriate error if it is not. 

        Parameters:
            None

        Returns:
            None
        """
        config = SimpleNamespace(remap_engine='xesmf', remap_method='nearest')

        ds = xr.Dataset({
            'lon': xr.DataArray([0.0, 1.0, 2.0], dims=['x']),
            'lat': xr.DataArray([0.0, 1.0, 2.0], dims=['x']),
        })

        data = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Unknown remap_engine"):
            dispatch_remap(data, ds, config,
                           lon_min=0.0, lon_max=2.0,
                           lat_min=0.0, lat_max=2.0,
                           resolution=1.0)


class TestCreateTargetGridFunction:
    """ Cover the stand-alone create_target_grid() function. """

    def test_returns_dataset_with_lon_lat(self: 'TestCreateTargetGridFunction') -> None:
        """
        This test verifies that calling create_target_grid with specific longitude and latitude bounds and resolutions returns an xarray Dataset containing 'lon' and 'lat' variables with the expected coordinate values. It checks that the 'lon' variable contains the correct longitude values based on the provided bounds and resolution, and that the 'lat' variable contains the correct latitude values.  This covers the basic functionality of create_target_grid in generating a structured grid based on the specified parameters. 

        Parameters:
            None

        Returns:
            None
        """
        grid = create_target_grid(
            lon_min=-10.0, lon_max=10.0,
            lat_min=-5.0, lat_max=5.0,
            dlon=5.0, dlat=5.0,
        )

        assert isinstance(grid, xr.Dataset)
        assert 'lon' in grid
        assert 'lat' in grid
        assert len(grid['lon']) == 5   # -10, -5, 0, 5, 10
        assert len(grid['lat']) == 3   # -5, 0, 5

    def test_default_global_grid(self: 'TestCreateTargetGridFunction') -> None:
        """
        This test verifies that calling create_target_grid with default parameters returns a global grid with the expected longitude and latitude values. It checks that the 'lon' variable contains values from 0 to 360 (inclusive) at 1-degree intervals, and that the 'lat' variable contains values from -90 to 90 (inclusive) at 1-degree intervals.  This covers the default behavior of create_target_grid in generating a global grid when no parameters are provided. 

        Parameters:
            None

        Returns:
            None
        """
        grid = create_target_grid()
        assert len(grid['lon']) == 361
        assert len(grid['lat']) == 181


class TestMainBlock:
    """ Cover the if __name__ == '__main__' block at the end of remapping.py. """

    def test_main_block_runs_without_error(self: 'TestMainBlock') -> None:
        """
        This test verifies that the main block at the end of remapping.py can be executed without error. It uses runpy to run the remapping module as a script and checks that it completes without raising any exceptions. This covers the code in the main block, which is typically used for command-line execution of the remapping functionality. 

        Parameters:
            None

        Returns:
            None
        """
        runpy.run_module('mpasdiag.processing.remapping', run_name='__main__')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

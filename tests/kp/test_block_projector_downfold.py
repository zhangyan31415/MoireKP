from __future__ import annotations

import numpy as np

from kp.blocks.downfold import DownfoldingOptions, downfold_blocks, downfold_from_projectors


def _unitary_from_seed(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    unitary, _ = np.linalg.qr(raw)
    return np.asarray(unitary, dtype=np.complex128)


def _assemble_dense_projectors(
    block_indices: list[np.ndarray],
    block_vecs: list[np.ndarray],
    low_bands_by_block: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    full_dim = max(int(np.max(indices)) for indices in block_indices) + 1
    low_dim = sum(len(bands) for bands in low_bands_by_block)
    high_dim = sum(vec.shape[1] - len(bands) for vec, bands in zip(block_vecs, low_bands_by_block))
    u_low = np.zeros((full_dim, low_dim), dtype=np.complex128)
    u_high = np.zeros((full_dim, high_dim), dtype=np.complex128)

    low_col = 0
    high_col = 0
    for indices, vec, low_bands in zip(block_indices, block_vecs, low_bands_by_block):
        low_bands_array = np.array(low_bands, dtype=int)
        high_bands = np.delete(np.arange(vec.shape[1]), low_bands_array)
        for band in low_bands_array:
            u_low[indices, low_col] = vec[:, band]
            low_col += 1
        for band in high_bands:
            u_high[indices, high_col] = vec[:, band]
            high_col += 1
    return u_low, u_high


def _fixed_schur_block_local_reference(
    ham: np.ndarray,
    block_indices: list[np.ndarray],
    block_vecs: list[np.ndarray],
    low_bands_by_block: list[list[int]],
    *,
    e_ref: float,
) -> np.ndarray:
    low_slices: list[slice] = []
    high_slices: list[slice] = []
    low_offset = 0
    high_offset = 0
    low_columns: list[np.ndarray] = []
    high_columns: list[np.ndarray] = []
    for vec, low_bands in zip(block_vecs, low_bands_by_block):
        low_bands_array = np.array(low_bands, dtype=int)
        high_bands = np.delete(np.arange(vec.shape[1]), low_bands_array)
        low_slices.append(slice(low_offset, low_offset + len(low_bands_array)))
        high_slices.append(slice(high_offset, high_offset + len(high_bands)))
        low_offset += len(low_bands_array)
        high_offset += len(high_bands)
        low_columns.append(vec[:, low_bands_array])
        high_columns.append(vec[:, high_bands])

    h_pp = np.zeros((low_offset, low_offset), dtype=np.complex128)
    h_ph = np.zeros((low_offset, high_offset), dtype=np.complex128)
    h_hh = np.zeros((high_offset, high_offset), dtype=np.complex128)
    for row, row_indices in enumerate(block_indices):
        for col, col_indices in enumerate(block_indices):
            ham_block = ham[np.ix_(row_indices, col_indices)]
            h_pp[low_slices[row], low_slices[col]] = (
                low_columns[row].conj().T @ ham_block @ low_columns[col]
            )
            h_ph[low_slices[row], high_slices[col]] = (
                low_columns[row].conj().T @ ham_block @ high_columns[col]
            )
            h_hh[high_slices[row], high_slices[col]] = (
                high_columns[row].conj().T @ ham_block @ high_columns[col]
            )

    return downfold_blocks(
        h_pp,
        h_ph,
        h_hh,
        DownfoldingOptions(method="fixed_schur", e_ref=e_ref),
    ).heff


def test_fixed_schur_dense_path_matches_block_local_reference_for_gamma_like_blocks() -> None:
    rng = np.random.default_rng(20240615)
    block_indices = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    block_vecs = [_unitary_from_seed(31, 3), _unitary_from_seed(47, 3)]
    low_bands_by_block = [[0], [0]]
    local_eigs = [
        np.array([-0.08, 0.42, 0.91]),
        np.array([-0.04, 0.50, 1.03]),
    ]

    ham = np.zeros((6, 6), dtype=np.complex128)
    for indices, vec, eigs in zip(block_indices, block_vecs, local_eigs):
        ham[np.ix_(indices, indices)] = vec @ np.diag(eigs) @ vec.conj().T
    coupling = 0.025 * (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    ham[np.ix_(block_indices[0], block_indices[1])] = coupling
    ham[np.ix_(block_indices[1], block_indices[0])] = coupling.conj().T

    u_low, u_high = _assemble_dense_projectors(block_indices, block_vecs, low_bands_by_block)
    e_ref = 0.12

    dense = downfold_from_projectors(
        ham,
        u_low,
        u_high,
        DownfoldingOptions(method="fixed_schur", e_ref=e_ref),
    ).heff
    block_local = _fixed_schur_block_local_reference(
        ham,
        block_indices,
        block_vecs,
        low_bands_by_block,
        e_ref=e_ref,
    )

    np.testing.assert_allclose(dense, block_local, atol=1e-12)


def test_block_projector_groups_match_dense_projectors() -> None:
    from kp.blocks.blocks import _assemble_projector_groups_from_block_eigenvectors
    from kp.blocks.downfold import downfold_from_projector_groups

    rng = np.random.default_rng(20260615)
    block_indices = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    block_vecs = [_unitary_from_seed(31, 3), _unitary_from_seed(47, 3)]
    low_bands_by_block = [[0], [0]]
    local_eigs = [
        np.array([-0.08, 0.42, 0.91]),
        np.array([-0.04, 0.50, 1.03]),
    ]

    ham = np.zeros((6, 6), dtype=np.complex128)
    for indices, vec, eigs in zip(block_indices, block_vecs, local_eigs):
        ham[np.ix_(indices, indices)] = vec @ np.diag(eigs) @ vec.conj().T
    coupling = 0.02 * (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    ham[np.ix_(block_indices[0], block_indices[1])] = coupling
    ham[np.ix_(block_indices[1], block_indices[0])] = coupling.conj().T

    u_low, u_high = _assemble_dense_projectors(block_indices, block_vecs, low_bands_by_block)
    options = DownfoldingOptions(method="fixed_schur", e_ref=0.12)
    dense = downfold_from_projectors(ham, u_low, u_high, options).heff

    groups_low, groups_high = _assemble_projector_groups_from_block_eigenvectors(
        np.asarray(block_vecs, dtype=object),
        block_indices,
        low_bands_by_block,
        include_high=True,
    )
    grouped = downfold_from_projector_groups(ham, groups_low, groups_high, options).heff

    np.testing.assert_allclose(grouped, dense, atol=1e-12)


def test_fixed_schur_preserves_off_block_low_high_coupling() -> None:
    block_indices = [np.array([0, 1]), np.array([2, 3])]
    block_vecs = [np.eye(2, dtype=np.complex128), np.eye(2, dtype=np.complex128)]
    low_bands_by_block = [[0], [0]]
    ham = np.zeros((4, 4), dtype=np.complex128)
    ham[1, 1] = 2.0
    ham[3, 3] = 3.0
    local_low_high = 0.20 + 0.05j
    off_block_low_high = -0.12 + 0.08j
    ham[0, 1] = local_low_high
    ham[1, 0] = local_low_high.conjugate()
    ham[2, 1] = off_block_low_high
    ham[1, 2] = off_block_low_high.conjugate()
    e_ref = 0.0

    u_low, u_high = _assemble_dense_projectors(block_indices, block_vecs, low_bands_by_block)
    dense = downfold_from_projectors(
        ham,
        u_low,
        u_high,
        DownfoldingOptions(method="fixed_schur", e_ref=e_ref),
    ).heff
    block_local = _fixed_schur_block_local_reference(
        ham,
        block_indices,
        block_vecs,
        low_bands_by_block,
        e_ref=e_ref,
    )
    block_diagonal_ham = ham.copy()
    block_diagonal_ham[np.ix_(block_indices[0], block_indices[1])] = 0.0
    block_diagonal_ham[np.ix_(block_indices[1], block_indices[0])] = 0.0
    dropped_coupling = downfold_from_projectors(
        block_diagonal_ham,
        u_low,
        u_high,
        DownfoldingOptions(method="fixed_schur", e_ref=e_ref),
    ).heff

    expected_off_diagonal = local_low_high * (-0.5) * off_block_low_high.conjugate()
    np.testing.assert_allclose(dense, block_local, atol=1e-12)
    np.testing.assert_allclose(dense[0, 1], expected_off_diagonal, atol=1e-12)
    assert abs(dense[0, 1]) > 1e-3
    assert abs(dropped_coupling[0, 1]) < 1e-12


def test_procrustes_aligned_low_vectors_are_not_treated_as_diagonal_selectors() -> None:
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    aligned_vec = np.array(
        [
            [inv_sqrt2, 0.0, inv_sqrt2, 0.0],
            [1j * inv_sqrt2, 0.0, -1j * inv_sqrt2, 0.0],
            [0.0, inv_sqrt2, 0.0, inv_sqrt2],
            [0.0, inv_sqrt2, 0.0, -inv_sqrt2],
        ],
        dtype=np.complex128,
    )
    block_indices = [np.array([0, 1, 2, 3])]
    block_vecs = [aligned_vec]
    low_bands_by_block = [[0, 1]]
    ham = aligned_vec @ np.diag([-0.11, 0.07, 1.20, 1.65]) @ aligned_vec.conj().T
    e_ref = 0.02

    u_low, u_high = _assemble_dense_projectors(block_indices, block_vecs, low_bands_by_block)
    dense = downfold_from_projectors(
        ham,
        u_low,
        u_high,
        DownfoldingOptions(method="fixed_schur", e_ref=e_ref),
    ).heff
    block_local = _fixed_schur_block_local_reference(
        ham,
        block_indices,
        block_vecs,
        low_bands_by_block,
        e_ref=e_ref,
    )
    diagonal_selector = downfold_from_projectors(
        ham,
        np.eye(4, dtype=np.complex128)[:, :2],
        np.eye(4, dtype=np.complex128)[:, 2:],
        DownfoldingOptions(method="fixed_schur", e_ref=e_ref),
    ).heff

    assert np.count_nonzero(np.abs(aligned_vec[:, 0]) > 1e-12) > 1
    assert np.count_nonzero(np.abs(aligned_vec[:, 1]) > 1e-12) > 1
    np.testing.assert_allclose(dense, block_local, atol=1e-12)
    assert not np.allclose(dense, diagonal_selector, atol=1e-6)

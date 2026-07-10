from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from tapw.symmetry.periodic_gauge import (
    build_projected_boundary_operator,
    load_boundary_operators,
    projected_basis_hash,
    save_boundary_operators,
)


def _structure_frame(group_sizes: tuple[int, int]) -> pd.DataFrame:
    rows = []
    atom_type = 0
    for group, size in enumerate(group_sizes):
        for index in range(size):
            rows.append(
                {
                    "twist_group": group,
                    "atom_type": atom_type,
                    "species": f"X{group}",
                    "orb_name": "s1",
                    "orb_num": 1,
                    "x": 0.17 * (index + 1) + 0.31 * group,
                    "y": -0.11 * index + 0.23 * group,
                }
            )
            atom_type += 1
    return pd.DataFrame(rows)


@pytest.mark.parametrize("group_sizes", [(1, 1), (1, 2), (1, 3)])
@pytest.mark.parametrize("spinful", [False, True])
def test_projected_boundary_operator_supports_unequal_group_dimensions(group_sizes, spinful):
    frame = _structure_frame(group_sizes)
    spinless_dim = int(sum(group_sizes))
    dim = spinless_dim * (2 if spinful else 1)
    projector = scipy.sparse.identity(dim, dtype=np.complex128, format="csr")
    shift = np.array([0.7, -0.2])

    boundary, diagnostics = build_projected_boundary_operator(
        structure_df=frame,
        g_matrix=projector,
        reciprocal_shift=shift,
        spinful=spinful,
    )

    positions = frame[["x", "y"]].to_numpy(dtype=float)
    expected_spinless = np.exp(1.0j * (positions @ shift))
    expected = np.tile(expected_spinless, 2) if spinful else expected_spinless
    assert boundary.shape == (dim, dim)
    np.testing.assert_allclose(boundary.diagonal(), expected)
    assert diagnostics["atomic_spinless_dim"] == spinless_dim
    assert diagnostics["projected_dim"] == dim


def test_projected_boundary_operator_uses_full_projector_not_g_block_relabel():
    frame = _structure_frame((1, 2))
    raw = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 0.0, np.sqrt(2.0)],
        ],
        dtype=np.complex128,
    ) / np.sqrt(2.0)
    projector = scipy.sparse.csr_matrix(raw)
    shift = np.array([0.4, 0.1])

    boundary, _diagnostics = build_projected_boundary_operator(
        structure_df=frame,
        g_matrix=projector,
        reciprocal_shift=shift,
        spinful=False,
    )

    atomic_phase = scipy.sparse.diags(
        np.exp(-1.0j * (frame[["x", "y"]].to_numpy(dtype=float) @ shift)),
        format="csr",
    )
    expected = (projector @ atomic_phase @ projector.conj().T).conj().T
    np.testing.assert_allclose(boundary.toarray(), expected.toarray())


def test_boundary_operator_maps_k_plus_b_eigenvectors_back_to_k_basis():
    frame = _structure_frame((1, 2))
    projector = scipy.sparse.identity(3, dtype=np.complex128, format="csr")
    shift = np.array([0.6, -0.15])
    boundary, _ = build_projected_boundary_operator(
        structure_df=frame,
        g_matrix=projector,
        reciprocal_shift=shift,
        spinful=False,
    )
    phase_plus = boundary.conj().T.toarray()
    h0 = np.array(
        [[0.2, 0.1j, 0.03], [-0.1j, 0.7, 0.08j], [0.03, -0.08j, 1.1]],
        dtype=np.complex128,
    )
    h_plus = phase_plus @ h0 @ phase_plus.conj().T
    _e0, u0 = np.linalg.eigh(h0)
    _e1, u_plus = np.linalg.eigh(h_plus)

    sewing_overlap = u0.conj().T @ boundary @ u_plus

    np.testing.assert_allclose(np.abs(sewing_overlap), np.eye(3), atol=1.0e-12)


def test_boundary_operator_pack_round_trips_sparse_matrices_and_identity(tmp_path: Path):
    frame = _structure_frame((1, 3))
    projector = scipy.sparse.identity(4, dtype=np.complex128, format="csr")
    b1 = np.array([1.0, 0.0])
    b2 = np.array([0.2, 0.8])
    op1, _ = build_projected_boundary_operator(
        structure_df=frame,
        g_matrix=projector,
        reciprocal_shift=b1,
        spinful=False,
    )
    op2, _ = build_projected_boundary_operator(
        structure_df=frame,
        g_matrix=projector,
        reciprocal_shift=b2,
        spinful=False,
    )
    basis_hash = projected_basis_hash(
        structure_df=frame,
        g_matrix=projector,
        valley="Gamma",
        q_shell=4,
        spinful=False,
    )
    path = tmp_path / "boundary_sewing.npz"
    metadata = {
        "identity_schema": "moirekp.artifact-identity.v1",
        "input_hash": "input-a",
        "config_hash": "config-a",
        "basis_hash": basis_hash,
        "package_version": "0.1.0",
        "schema_version": 1,
        "reciprocal_basis": [b1.tolist(), b2.tolist()],
        "wavefunction_hashes": {"wavefunctions_vbm.npy": "wave-a"},
    }

    save_boundary_operators(path, {"b1": op1, "b2": op2}, metadata)
    operators, loaded = load_boundary_operators(path)

    np.testing.assert_allclose(operators["b1"].toarray(), op1.toarray())
    np.testing.assert_allclose(operators["b2"].toarray(), op2.toarray())
    assert loaded == metadata
    with np.load(path, allow_pickle=False) as payload:
        assert json.loads(str(payload["metadata_json"].item()))["basis_hash"] == basis_hash

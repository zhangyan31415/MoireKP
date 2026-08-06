"""Symmetry-cyclic structural generators for complete response envelopes.

This module works before :class:`RawPolynomialSeed` materialization.  It only
chooses sector/orbital/harmonic supports; every authored momentum monomial of a
chosen support is retained for the existing symbolic projector and rank
reducer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class StructuralGeneratorPlan:
    """Certified pre-materialization selection for one symmetry group."""

    selected_term_indices: tuple[int, ...]
    orbit_representative_term_indices: tuple[int, ...]
    full_structural_support_count: int
    selected_generator_count: int
    full_logical_term_count: int
    materialized_term_count: int
    full_closure_rank: int
    selected_closure_rank: int
    maximum_omitted_closure_residual: float
    blocks: tuple[Mapping[str, Any], ...]

    def artifact(self) -> dict[str, Any]:
        return {
            "compiler": "exact_structural_cyclic_generators_v1",
            "full_structural_support_count": int(
                self.full_structural_support_count
            ),
            "selected_generator_count": int(self.selected_generator_count),
            "full_logical_term_count": int(self.full_logical_term_count),
            "materialized_term_count": int(self.materialized_term_count),
            "full_closure_rank": int(self.full_closure_rank),
            "selected_closure_rank": int(self.selected_closure_rank),
            "maximum_omitted_closure_residual": float(
                self.maximum_omitted_closure_residual
            ),
            "selected_term_indices": [
                int(value) for value in self.selected_term_indices
            ],
            "orbit_representative_term_indices": [
                int(value) for value in self.orbit_representative_term_indices
            ],
            "blocks": [dict(block) for block in self.blocks],
        }


def _family_key(term: Any) -> tuple[str, ...]:
    metadata = dict(getattr(term, "registry_metadata", {}) or {})
    return (
        str(metadata.get("term_name", getattr(term, "tag", ""))),
        str(metadata.get("term_kind", getattr(term, "tag", ""))),
        str(getattr(term, "tag", "")),
        str(metadata.get("term_space_policy", "")),
    )


def _structural_key(term: Any) -> tuple[Any, ...]:
    key = term.key
    return (
        int(key.layer_from),
        int(key.layer_to),
        int(key.orbital_from),
        int(key.orbital_to),
        float(key.p[0]),
        float(key.p[1]),
    )


def _structural_matrix(
    structural_key: tuple[Any, ...],
    *,
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    n_orb1: int,
    n_orb2: int,
    p_tolerance: float = 1.0e-5,
) -> sparse.csr_matrix:
    layer_from, layer_to, orbital_from, orbital_to, px, py = structural_key
    qsets = {1: Q_set1, 2: Q_set2}
    n_orbs = {1: int(n_orb1), 2: int(n_orb2)}
    q_rows = qsets[int(layer_from)]
    q_cols = qsets[int(layer_to)]
    orbital_from = int(orbital_from) - 1
    orbital_to = int(orbital_to) - 1
    if not 0 <= orbital_from < n_orbs[int(layer_from)]:
        raise ValueError("structural generator orbital_from is out of range")
    if not 0 <= orbital_to < n_orbs[int(layer_to)]:
        raise ValueError("structural generator orbital_to is out of range")
    q1_count = int(Q_set1.shape[0])
    dim = q1_count * int(n_orb1) + int(Q_set2.shape[0]) * int(n_orb2)

    def global_index(layer: int, q_index: int, orbital: int) -> int:
        if layer == 1:
            return int(orbital * q1_count + q_index)
        return int(q1_count * int(n_orb1) + orbital * Q_set2.shape[0] + q_index)

    p_vector = np.asarray((px, py), dtype=float)
    rows: list[int] = []
    columns: list[int] = []
    for row_q_index, q_row in enumerate(q_rows):
        matches = np.flatnonzero(
            np.linalg.norm(q_row - p_vector - q_cols, axis=1)
            < float(p_tolerance)
        )
        row = global_index(int(layer_from), row_q_index, orbital_from)
        for col_q_index in matches:
            rows.append(row)
            columns.append(
                global_index(int(layer_to), int(col_q_index), orbital_to)
            )
    matrix = sparse.coo_matrix(
        (
            np.ones(len(rows), dtype=np.complex128),
            (
                np.asarray(rows, dtype=np.int64),
                np.asarray(columns, dtype=np.int64),
            ),
        ),
        shape=(dim, dim),
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def _real_column(matrix: sparse.spmatrix) -> sparse.csc_matrix:
    value = sparse.csr_matrix(matrix, dtype=np.complex128).reshape((-1, 1))
    return sparse.vstack((value.real, value.imag), format="csc")


def _complex_direction_columns(matrix: sparse.spmatrix) -> sparse.csc_matrix:
    value = sparse.csr_matrix(matrix, dtype=np.complex128)
    return sparse.hstack(
        (_real_column(value), _real_column(1.0j * value)), format="csc"
    )


def _cyclic_columns(matrix: sparse.spmatrix, group: Any) -> sparse.csc_matrix:
    source_matrix = sparse.csr_matrix(matrix, dtype=np.complex128)
    columns: list[sparse.csc_matrix] = []
    for element in group.elements:
        internal = sparse.csr_matrix(element.internal_u, dtype=np.complex128)
        source = source_matrix.conjugate() if bool(element.antiunitary) else source_matrix
        transformed = (internal @ source @ internal.getH()).tocsr()
        columns.append(_complex_direction_columns(transformed))
        columns.append(_complex_direction_columns(transformed.getH()))
    return sparse.hstack(columns, format="csc")


def _coordinate_residual_norms(
    *,
    gram: np.ndarray,
    overlaps: np.ndarray,
    coordinates: np.ndarray,
    column_norms_squared: np.ndarray,
) -> np.ndarray:
    cross = np.sum(coordinates * overlaps, axis=0)
    reconstructed = np.sum(coordinates * (gram @ coordinates), axis=0)
    residual_squared = column_norms_squared - 2.0 * cross + reconstructed
    residual_squared = np.maximum(residual_squared, 0.0)
    return np.sqrt(residual_squared)


def _select_block_generators(
    matrices: Sequence[sparse.csr_matrix],
    *,
    group: Any,
    maximum_action_error_bound: float,
) -> tuple[tuple[int, ...], int, int, float, bool]:
    cyclic_blocks = tuple(_cyclic_columns(matrix, group) for matrix in matrices)
    cyclic = sparse.hstack(cyclic_blocks, format="csc")
    gram = np.asarray((cyclic.T @ cyclic).toarray(), dtype=float)
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    spectral_scale = max(float(np.max(eigenvalues, initial=0.0)), 1.0)
    eigenvalue_tolerance = max(
        512.0
        * np.finfo(float).eps
        * max(gram.shape)
        * spectral_scale,
        (8.0 * float(maximum_action_error_bound)) ** 2,
    )
    retained = eigenvalues > eigenvalue_tolerance
    if not np.any(retained):
        return tuple(range(len(matrices))), 0, 0, 0.0, False
    coordinates = (
        np.sqrt(eigenvalues[retained])[:, np.newaxis]
        * eigenvectors[:, retained].T
    )
    tolerance = max(
        512.0
        * np.finfo(float).eps
        * max(coordinates.shape)
        * max(float(np.linalg.norm(coordinates, ord=2)), 1.0),
        8.0 * float(maximum_action_error_bound),
    )

    selected: list[int] = []
    orthonormal = np.empty((coordinates.shape[0], 0), dtype=float)
    offset = 0
    for support_index, block in enumerate(cyclic_blocks):
        width = int(block.shape[1])
        block_coordinates = coordinates[:, offset : offset + width]
        offset += width
        residual = block_coordinates
        if orthonormal.shape[1]:
            residual = residual - orthonormal @ (orthonormal.T @ residual)
        singular_values = np.linalg.svd(residual, compute_uv=False)
        residual_scale = max(float(np.max(singular_values, initial=0.0)), 1.0)
        rank_tolerance = max(
            512.0
            * np.finfo(float).eps
            * max(residual.shape)
            * residual_scale,
            8.0 * float(maximum_action_error_bound),
        )
        added_rank = int(np.count_nonzero(singular_values > rank_tolerance))
        if added_rank == 0:
            continue
        selected.append(int(support_index))
        left, values, _right = np.linalg.svd(residual, full_matrices=False)
        new_directions = left[:, values > rank_tolerance]
        augmented = np.hstack((orthonormal, new_directions))
        orthonormal, _r = np.linalg.qr(augmented, mode="reduced")

    full_rank = int(np.count_nonzero(retained))
    selected_rank = int(orthonormal.shape[1])
    selected_residual = coordinates
    if selected_rank:
        selected_residual = coordinates - orthonormal @ (orthonormal.T @ coordinates)
    maximum_residual = float(
        np.max(np.linalg.norm(selected_residual, axis=0), initial=0.0)
    )
    if selected_rank != full_rank or maximum_residual > tolerance:
        return tuple(range(len(matrices))), full_rank, full_rank, 0.0, False
    return tuple(selected), full_rank, selected_rank, maximum_residual, True


def compile_structural_generator_plan(
    term_records: Sequence[tuple[int, Any]],
    *,
    group: Any,
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    n_orb1: int,
    n_orb2: int,
    maximum_action_error_bound: float = 0.0,
    optimize_zero_harmonic: bool = True,
) -> StructuralGeneratorPlan:
    """Select symmetry-cyclic structural supports within exact envelopes."""

    records = tuple((int(index), term) for index, term in term_records)
    qset1 = np.asarray(Q_set1, dtype=float)
    qset2 = np.asarray(Q_set2, dtype=float)
    if qset1.ndim != 2 or qset1.shape[1:] != (2,):
        raise ValueError("Q_set1 must have shape (N,2)")
    if qset2.ndim != 2 or qset2.shape[1:] != (2,):
        raise ValueError("Q_set2 must have shape (N,2)")

    terms_by_support: dict[tuple[tuple[str, ...], tuple[Any, ...]], list[tuple[int, Any]]] = {}
    passthrough_indices: list[int] = []
    for term_index, term in records:
        family = _family_key(term)
        structural_key = _structural_key(term)
        case_derived = family[0].startswith(
            ("gamma_case_", "k_case_", "m_case_")
        )
        zero_harmonic = bool(
            float(structural_key[-2]) == 0.0
            and float(structural_key[-1]) == 0.0
        )
        if (
            family[-1] != "complete"
            or not case_derived
            or (zero_harmonic and not bool(optimize_zero_harmonic))
        ):
            passthrough_indices.append(term_index)
            continue
        terms_by_support.setdefault((family, structural_key), []).append(
            (term_index, term)
        )

    blocks: dict[
        tuple[tuple[str, ...], tuple[tuple[int, int], ...]],
        list[tuple[tuple[Any, ...], list[tuple[int, Any]]]],
    ] = {}
    for (family, structural_key), support_terms in terms_by_support.items():
        envelope = tuple(
            sorted(
                {
                    (int(term.key.Mz), int(term.key.Mz_star))
                    for _index, term in support_terms
                }
            )
        )
        blocks.setdefault((family, envelope), []).append(
            (structural_key, support_terms)
        )

    selected_indices = set(passthrough_indices)
    orbit_representative_indices: set[int] = set()
    block_artifacts: list[Mapping[str, Any]] = []
    full_support_count = len(terms_by_support) + len(passthrough_indices)
    selected_generator_count = len(passthrough_indices)
    full_closure_rank = 0
    selected_closure_rank = 0
    maximum_residual = 0.0
    for (family, envelope), supports in sorted(
        blocks.items(), key=lambda item: repr(item[0])
    ):
        supports = sorted(supports, key=lambda item: repr(item[0]))
        matrices = tuple(
            _structural_matrix(
                structural_key,
                Q_set1=qset1,
                Q_set2=qset2,
                n_orb1=int(n_orb1),
                n_orb2=int(n_orb2),
            )
            for structural_key, _terms in supports
        )
        if any(matrix.nnz == 0 for matrix in matrices):
            selected_local = tuple(range(len(supports)))
            block_full_rank = 2 * len(supports)
            block_selected_rank = block_full_rank
            block_residual = 0.0
            certified = False
        else:
            (
                selected_local,
                block_full_rank,
                block_selected_rank,
                block_residual,
                certified,
            ) = _select_block_generators(
                matrices,
                group=group,
                maximum_action_error_bound=float(maximum_action_error_bound),
            )
        for local_index in selected_local:
            selected_support_indices = {
                term_index for term_index, _term in supports[local_index][1]
            }
            selected_indices.update(selected_support_indices)
            if certified:
                orbit_representative_indices.update(selected_support_indices)
        selected_generator_count += len(selected_local)
        full_closure_rank += int(block_full_rank)
        selected_closure_rank += int(block_selected_rank)
        maximum_residual = max(maximum_residual, float(block_residual))
        block_artifacts.append(
            {
                "family": list(family),
                "allowed_monomials": [list(value) for value in envelope],
                "full_structural_support_count": int(len(supports)),
                "selected_generator_count": int(len(selected_local)),
                "selected_structural_keys": [
                    list(supports[index][0]) for index in selected_local
                ],
                "full_closure_rank": int(block_full_rank),
                "selected_closure_rank": int(block_selected_rank),
                "maximum_omitted_closure_residual": float(block_residual),
                "certified": bool(certified),
            }
        )

    selected = tuple(sorted(selected_indices))
    return StructuralGeneratorPlan(
        selected_term_indices=selected,
        orbit_representative_term_indices=tuple(
            sorted(orbit_representative_indices)
        ),
        full_structural_support_count=int(full_support_count),
        selected_generator_count=int(selected_generator_count),
        full_logical_term_count=int(len(records)),
        materialized_term_count=int(len(selected)),
        full_closure_rank=int(full_closure_rank),
        selected_closure_rank=int(selected_closure_rank),
        maximum_omitted_closure_residual=float(maximum_residual),
        blocks=tuple(block_artifacts),
    )


__all__ = ["StructuralGeneratorPlan", "compile_structural_generator_plan"]

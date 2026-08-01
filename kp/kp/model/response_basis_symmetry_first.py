"""Symmetry-first compilation helpers for finite-harmonic response seeds."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import time

import numpy as np
import scipy.linalg
from scipy import sparse

from .response_basis import (
    CandidateResponseSet,
    NullClassificationPolicy,
    PolynomialCoordinateBasis,
    RawPolynomialSeed,
    ResponseChannel,
    _adjoint_polynomial,
    _channel_sparse_vector,
    _coefficient_norm,
    _coefficient_vector_to_response_map,
    _propagated_error_components,
)
from .response_basis_factorized import compile_factorized_raw_seed_action
from .response_basis_factorized import compile_factorized_group_element_actions
from .response_basis_fixed_compiler import compile_generator_fixed_vocabulary


@dataclass(frozen=True)
class RawRealVocabulary:
    """Sparse global real coefficient columns for ordered complex seeds."""

    matrix: sparse.csc_matrix
    channel_ids: tuple[str, ...]


@dataclass(frozen=True)
class SymmetryFirstFixedSpace:
    rank: int
    physical_basis: sparse.csc_matrix
    fixed_vocabulary_coordinates: Any
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class _SymbolicGroupContext:
    element: Any
    column_routes: tuple[tuple[np.ndarray, np.ndarray], ...]
    monomial_pullbacks: Mapping[tuple[int, int], Mapping[tuple[int, int], complex]]


@dataclass(frozen=True)
class _SymbolicSeedAtoms:
    entries: tuple[tuple[int, int, int, int, complex], ...]


def _column_routes(matrix: sparse.spmatrix) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    value = sparse.csc_matrix(matrix, dtype=np.complex128)
    return tuple(
        (
            np.asarray(value.indices[value.indptr[index] : value.indptr[index + 1]], dtype=np.int64),
            np.asarray(value.data[value.indptr[index] : value.indptr[index + 1]], dtype=np.complex128),
        )
        for index in range(value.shape[1])
    )


def _seed_symbolic_atoms(seed: RawPolynomialSeed) -> _SymbolicSeedAtoms:
    entries: list[tuple[int, int, int, int, complex]] = []
    for (r, s), matrix in seed.coefficients.items():
        coo = sparse.coo_matrix(matrix, dtype=np.complex128)
        entries.extend(
            (
                int(r),
                int(s),
                int(row),
                int(col),
                complex(value),
            )
            for row, col, value in zip(coo.row, coo.col, coo.data)
        )
    return _SymbolicSeedAtoms(entries=tuple(entries))


def _symbolic_vector_from_accumulated(
    accumulated: Mapping[tuple[int, int, int, int], complex],
    *,
    coordinate: PolynomialCoordinateBasis,
    dim: int,
    group_scale: float,
) -> sparse.csc_matrix:
    hermitian: dict[tuple[int, int, int, int], complex] = {}
    for (r, s, row, col), value in accumulated.items():
        direct_key = (r, s, row, col)
        adjoint_key = (s, r, col, row)
        scaled = 0.5 * group_scale * complex(value)
        hermitian[direct_key] = hermitian.get(direct_key, 0.0j) + scaled
        hermitian[adjoint_key] = hermitian.get(adjoint_key, 0.0j) + np.conjugate(scaled)

    monomial_index = {key: index for index, key in enumerate(coordinate.monomials)}
    block = len(coordinate.monomials) * dim * dim
    real_entries: dict[int, float] = {}
    for (r, s, row, col), value in hermitian.items():
        base = monomial_index[(r, s)] * dim * dim + row * dim + col
        if value.real != 0.0:
            real_entries[base] = real_entries.get(base, 0.0) + float(value.real)
        if value.imag != 0.0:
            imag_row = block + base
            real_entries[imag_row] = real_entries.get(imag_row, 0.0) + float(value.imag)
    nonzero = [(row, value) for row, value in real_entries.items() if value != 0.0]
    if not nonzero:
        return sparse.csc_matrix((2 * block, 1), dtype=np.float64)
    rows, values = zip(*sorted(nonzero))
    return sparse.csc_matrix(
        (
            np.asarray(values, dtype=np.float64),
            (np.asarray(rows, dtype=np.int64), np.zeros(len(rows), dtype=np.int64)),
        ),
        shape=(2 * block, 1),
    )


def _symbolic_reynolds_pair(
    seed: RawPolynomialSeed,
    atoms: _SymbolicSeedAtoms,
    *,
    coordinate: PolynomialCoordinateBasis,
    contexts: Sequence[_SymbolicGroupContext],
) -> tuple[sparse.csc_matrix, sparse.csc_matrix]:
    """Project one authored real/imag pair using scalar polynomial/matrix atoms."""

    dim = int(seed.dim)
    accumulated_unitary: dict[tuple[int, int, int, int], complex] = {}
    accumulated_antiunitary: dict[tuple[int, int, int, int], complex] = {}
    for context in contexts:
        element = context.element
        routes = context.column_routes
        pullbacks = context.monomial_pullbacks
        accumulated = (
            accumulated_antiunitary
            if bool(element.antiunitary)
            else accumulated_unitary
        )
        for r, s, source_row, source_col, raw_value in atoms.entries:
            left_rows, left_values = routes[source_row]
            right_rows, right_values = routes[source_col]
            for pulled_monomial, polynomial_value in pullbacks[(r, s)].items():
                transformed_monomial = tuple(int(value) for value in pulled_monomial)
                transformed_value = raw_value * complex(polynomial_value)
                if bool(element.antiunitary):
                    transformed_monomial = (
                        transformed_monomial[1],
                        transformed_monomial[0],
                    )
                    transformed_value = np.conjugate(transformed_value)
                for target_row, left_value in zip(left_rows, left_values):
                    for target_col, right_value in zip(right_rows, right_values):
                        key = (
                            transformed_monomial[0],
                            transformed_monomial[1],
                            int(target_row),
                            int(target_col),
                        )
                        contribution = (
                            transformed_value
                            * complex(left_value)
                            * np.conjugate(complex(right_value))
                        )
                        accumulated[key] = accumulated.get(key, 0.0j) + contribution

    group_scale = 1.0 / float(len(contexts))
    all_keys = set(accumulated_unitary).union(accumulated_antiunitary)
    accumulated_real = {
        key: accumulated_unitary.get(key, 0.0j)
        + accumulated_antiunitary.get(key, 0.0j)
        for key in all_keys
    }
    accumulated_imag = {
        key: 1.0j
        * (
            accumulated_unitary.get(key, 0.0j)
            - accumulated_antiunitary.get(key, 0.0j)
        )
        for key in all_keys
    }
    return (
        _symbolic_vector_from_accumulated(
            accumulated_real,
            coordinate=coordinate,
            dim=dim,
            group_scale=group_scale,
        ),
        _symbolic_vector_from_accumulated(
            accumulated_imag,
            coordinate=coordinate,
            dim=dim,
            group_scale=group_scale,
        ),
    )


def _symbolic_reynolds_batch(
    seeds: Sequence[RawPolynomialSeed],
    atoms_by_seed: Sequence[_SymbolicSeedAtoms],
    *,
    coordinate: PolynomialCoordinateBasis,
    contexts: Sequence[_SymbolicGroupContext],
) -> sparse.csc_matrix:
    """Project all authored complex seeds through sparse atom operators at once."""

    ordered = tuple(seeds)
    if len(ordered) != len(tuple(atoms_by_seed)):
        raise ValueError("symbolic seed/atom batches must have the same length")
    dim = int(ordered[0].dim)
    complex_block = len(coordinate.monomials) * dim * dim
    monomial_index = {key: index for index, key in enumerate(coordinate.monomials)}
    raw_rows: list[int] = []
    raw_columns: list[int] = []
    raw_values: list[complex] = []
    for seed_index, atoms in enumerate(atoms_by_seed):
        for r, s, row, col, value in atoms.entries:
            raw_rows.append(
                monomial_index[(r, s)] * dim * dim + row * dim + col
            )
            raw_columns.append(seed_index)
            raw_values.append(value)
    raw = sparse.coo_matrix(
        (
            np.asarray(raw_values, dtype=np.complex128),
            (
                np.asarray(raw_rows, dtype=np.int64),
                np.asarray(raw_columns, dtype=np.int64),
            ),
        ),
        shape=(complex_block, len(ordered)),
    ).tocsc()
    raw.sum_duplicates()
    raw.eliminate_zeros()
    source_atom_rows = np.unique(raw.indices)
    raw_conjugate = raw.conjugate().tocsc()
    unitary_sum = sparse.csc_matrix(raw.shape, dtype=np.complex128)
    antiunitary_sum = sparse.csc_matrix(raw.shape, dtype=np.complex128)

    for context in contexts:
        transform_rows: list[int] = []
        transform_columns: list[int] = []
        transform_values: list[complex] = []
        antiunitary = bool(context.element.antiunitary)
        for source_atom in source_atom_rows:
            source_monomial_index, matrix_offset = divmod(
                int(source_atom),
                dim * dim,
            )
            source_row, source_col = divmod(matrix_offset, dim)
            monomial = coordinate.monomials[source_monomial_index]
            left_rows, left_values = context.column_routes[source_row]
            right_rows, right_values = context.column_routes[source_col]
            for pulled_monomial, polynomial_value in context.monomial_pullbacks[
                monomial
            ].items():
                target_monomial = tuple(int(value) for value in pulled_monomial)
                scalar = complex(polynomial_value)
                if antiunitary:
                    target_monomial = (target_monomial[1], target_monomial[0])
                    scalar = np.conjugate(scalar)
                target_monomial_offset = monomial_index[target_monomial] * dim * dim
                for target_row, left_value in zip(left_rows, left_values):
                    for target_col, right_value in zip(right_rows, right_values):
                        transform_rows.append(
                            target_monomial_offset
                            + int(target_row) * dim
                            + int(target_col)
                        )
                        transform_columns.append(int(source_atom))
                        transform_values.append(
                            scalar
                            * complex(left_value)
                            * np.conjugate(complex(right_value))
                        )
        transform = sparse.coo_matrix(
            (
                np.asarray(transform_values, dtype=np.complex128),
                (
                    np.asarray(transform_rows, dtype=np.int64),
                    np.asarray(transform_columns, dtype=np.int64),
                ),
            ),
            shape=(complex_block, complex_block),
        ).tocsc()
        transform.sum_duplicates()
        transform.eliminate_zeros()
        if antiunitary:
            antiunitary_sum = (antiunitary_sum + transform @ raw_conjugate).tocsc()
        else:
            unitary_sum = (unitary_sum + transform @ raw).tocsc()

    scale = 1.0 / float(len(contexts))

    def hermitian_project_batch(value: sparse.csc_matrix) -> sparse.csc_matrix:
        coo = value.tocoo()
        source_monomials, matrix_offsets = np.divmod(
            np.asarray(coo.row, dtype=np.int64),
            dim * dim,
        )
        matrix_rows, matrix_cols = np.divmod(matrix_offsets, dim)
        adjoint_monomials = np.asarray(
            [
                monomial_index[
                    (
                        coordinate.monomials[int(index)][1],
                        coordinate.monomials[int(index)][0],
                    )
                ]
                for index in source_monomials
            ],
            dtype=np.int64,
        )
        adjoint_rows = (
            adjoint_monomials * dim * dim + matrix_cols * dim + matrix_rows
        )
        adjoint = sparse.coo_matrix(
            (
                np.conjugate(np.asarray(coo.data, dtype=np.complex128)),
                (adjoint_rows, np.asarray(coo.col, dtype=np.int64)),
            ),
            shape=value.shape,
        ).tocsc()
        projected_value = (0.5 * scale * (value + adjoint)).tocsc()
        projected_value.sum_duplicates()
        projected_value.eliminate_zeros()
        return projected_value

    real_complex = hermitian_project_batch(unitary_sum + antiunitary_sum)
    imag_complex = hermitian_project_batch(1.0j * (unitary_sum - antiunitary_sum))

    def real_stack(value: sparse.csc_matrix) -> sparse.csc_matrix:
        return sparse.vstack([value.real, value.imag], format="csc")

    real_columns = real_stack(real_complex)
    imag_columns = real_stack(imag_complex)
    grouped = sparse.hstack([real_columns, imag_columns], format="csc")
    interleaved = np.empty(2 * len(ordered), dtype=np.int64)
    interleaved[0::2] = np.arange(len(ordered), dtype=np.int64)
    interleaved[1::2] = len(ordered) + np.arange(len(ordered), dtype=np.int64)
    projected = grouped[:, interleaved].tocsc()
    projected.eliminate_zeros()
    projected.sort_indices()
    return projected


def _independent_projected_columns(
    projected: sparse.csc_matrix,
    *,
    relative_error_bounds: np.ndarray | None = None,
    confirm_factor: float = 100.0,
) -> tuple[tuple[int, ...], tuple[Mapping[str, Any], ...]]:
    norms = np.sqrt(np.asarray(projected.power(2).sum(axis=0)).ravel())
    relative_errors = (
        np.zeros(projected.shape[1], dtype=np.float64)
        if relative_error_bounds is None
        else np.asarray(relative_error_bounds, dtype=np.float64)
    )
    if relative_errors.shape != (projected.shape[1],):
        raise ValueError("relative error bounds must match projected columns")
    if np.any(~np.isfinite(relative_errors)) or np.any(relative_errors < 0.0):
        raise ValueError("relative error bounds must be finite and nonnegative")
    active = [index for index, norm in enumerate(norms) if norm > 0.0]
    active_array = np.asarray(active, dtype=np.int64)
    normalized_matrix = projected[:, active_array].tocsc() @ sparse.diags(
        1.0 / norms[active_array],
        format="csc",
    )
    incidence = normalized_matrix.copy()
    incidence.data = np.ones(incidence.nnz, dtype=np.float64)
    adjacency = (incidence.T @ incidence).tocsr()
    adjacency.data = np.ones(adjacency.nnz, dtype=np.float64)
    _component_count, labels = sparse.csgraph.connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    components: dict[int, list[int]] = {}
    for local_index, label in enumerate(labels):
        components.setdefault(int(label), []).append(active[local_index])
    local_by_global = {global_index: local_index for local_index, global_index in enumerate(active)}

    selected: list[int] = []
    proofs: list[Mapping[str, Any]] = []
    for component_index, indices in enumerate(sorted(components.values(), key=tuple)):
        local_indices = np.asarray(
            [local_by_global[index] for index in indices],
            dtype=np.int64,
        )
        vectors = normalized_matrix[:, local_indices].tocsc()
        active_rows = np.unique(vectors.indices)
        dense = np.asarray(vectors[active_rows, :].toarray(), dtype=np.float64)
        q_factor, r_factor, pivots = scipy.linalg.qr(
            dense,
            pivoting=True,
            mode="economic",
            check_finite=False,
        )
        rank_diagonal = np.abs(np.diag(r_factor))
        roundoff_tolerance = (
            np.finfo(float).eps
            * max(dense.shape)
            * float(np.linalg.norm(dense, ord="fro"))
            if dense.size
            else 0.0
        )
        tolerance = max(
            roundoff_tolerance,
            float(confirm_factor)
            * float(np.max(relative_errors[indices], initial=0.0)),
        )
        rank = int(np.count_nonzero(rank_diagonal > tolerance))
        if rank:
            chosen = [indices[int(value)] for value in pivots[:rank]]
        else:
            chosen = []
        certified_chosen = list(chosen)
        if rank:
            selected_orthogonal = q_factor[:, :rank]
            residuals = np.linalg.norm(
                dense
                - selected_orthogonal
                @ (selected_orthogonal.T @ dense),
                axis=0,
            )
        else:
            residuals = np.linalg.norm(dense, axis=0)
        condition_roundoff = (
            float(
                rank_diagonal[0]
                / rank_diagonal[rank - 1]
                * np.finfo(float).eps
            )
            if rank and rank_diagonal[rank - 1] > 0.0
            else 0.0
        )
        chosen_set = set(chosen)
        for local_index, global_index in enumerate(indices):
            if global_index in chosen_set:
                continue
            own_floor = float(relative_errors[global_index] + condition_roundoff)
            if float(residuals[local_index]) > own_floor:
                certified_chosen.append(global_index)
        selected.extend(certified_chosen)
        proofs.append(
            {
                "component": int(component_index),
                "channel_indices": [int(value) for value in indices],
                "selected_indices": [int(value) for value in certified_chosen],
                "rank": int(len(certified_chosen)),
                "rrqr_proposed_rank": int(rank),
                "active_atom_rows": int(active_rows.size),
                "rank_tolerance": float(tolerance),
            }
        )
    return tuple(sorted(selected)), tuple(proofs)


def _seed_component_coefficients(
    seed: RawPolynomialSeed,
    component: str,
) -> dict[tuple[int, int], sparse.csr_matrix]:
    phase = 1.0 + 0.0j if component == "real" else 1.0j
    return {
        (int(r), int(s)): sparse.csr_matrix(matrix * phase)
        for (r, s), matrix in seed.coefficients.items()
    }


def _vectorize(
    coefficients: dict[tuple[int, int], sparse.csr_matrix],
    *,
    coordinate: PolynomialCoordinateBasis,
    dim: int,
    channel_id: str,
) -> sparse.csc_matrix:
    return _channel_sparse_vector(
        SimpleNamespace(channel_id=channel_id, coefficients=coefficients),
        coordinate,
        dim,
    )


def _validate_seeds(seeds: Sequence[RawPolynomialSeed]) -> tuple[RawPolynomialSeed, ...]:
    ordered = tuple(seeds)
    if not ordered:
        raise ValueError("raw real vocabulary requires at least one seed")
    dim = ordered[0].dim
    if any(seed.dim != dim for seed in ordered):
        raise ValueError("raw real vocabulary seeds do not share one matrix dimension")
    if len({seed.seed_id for seed in ordered}) != len(ordered):
        raise ValueError("raw real vocabulary seed ids must be unique")
    return ordered


def build_raw_real_vocabulary(
    seeds: Sequence[RawPolynomialSeed],
    *,
    coordinate: PolynomialCoordinateBasis,
) -> RawRealVocabulary:
    ordered = _validate_seeds(seeds)
    channel_ids = tuple(
        f"{seed.seed_id}:{component}"
        for seed in ordered
        for component in ("real", "imag")
    )
    columns = [
        _vectorize(
            _seed_component_coefficients(seed, component),
            coordinate=coordinate,
            dim=seed.dim,
            channel_id=f"{seed.seed_id}:{component}",
        )
        for seed in ordered
        for component in ("real", "imag")
    ]
    return RawRealVocabulary(
        matrix=sparse.hstack(columns, format="csc"),
        channel_ids=channel_ids,
    )


def build_raw_adjoint_image(
    seeds: Sequence[RawPolynomialSeed],
    *,
    coordinate: PolynomialCoordinateBasis,
) -> sparse.csc_matrix:
    ordered = _validate_seeds(seeds)
    columns = []
    for seed in ordered:
        for component in ("real", "imag"):
            channel_id = f"{seed.seed_id}:{component}"
            coefficients = _seed_component_coefficients(seed, component)
            columns.append(
                _vectorize(
                    _adjoint_polynomial(coefficients),
                    coordinate=coordinate,
                    dim=seed.dim,
                    channel_id=f"{channel_id}:adjoint",
                )
            )
    return sparse.hstack(columns, format="csc")


def compile_symmetry_first_fixed_space(
    seeds: Sequence[RawPolynomialSeed],
    *,
    coordinate: PolynomialCoordinateBasis,
    factorized_actions: Mapping[str, Any],
    group: Any | None = None,
    internal_actions_by_word: Mapping[tuple[str, ...], sparse.spmatrix] | None = None,
    factorized_group_artifact: Mapping[str, Any] | None = None,
) -> SymmetryFirstFixedSpace:
    ordered = _validate_seeds(seeds)
    if not factorized_actions and group is None:
        raise ValueError("symmetry-first fixed space requires generator actions")
    channel_ids = tuple(
        f"{seed.seed_id}:{component}"
        for seed in ordered
        for component in ("real", "imag")
    )
    if group is not None:
        context_started = time.perf_counter()
        if internal_actions_by_word is None:
            internal_actions, action_artifact = compile_factorized_group_element_actions(
                group=group,
                factorized_generators=factorized_actions,
            )
        else:
            if factorized_group_artifact is None:
                raise ValueError(
                    "precompiled factorized group actions require their certification artifact"
                )
            internal_actions = {
                tuple(str(value) for value in word): sparse.csr_matrix(matrix)
                for word, matrix in internal_actions_by_word.items()
            }
            action_artifact = dict(factorized_group_artifact)
        from . import response_basis as _response_basis

        contexts = tuple(
            _SymbolicGroupContext(
                element=element,
                column_routes=_column_routes(
                    internal_actions[
                        tuple(str(value) for value in element.canonical_word)
                    ]
                ),
                monomial_pullbacks=_response_basis._monomial_pullback_table(
                    element,
                    coordinate,
                ),
            )
            for element in group.elements
        )
        group_context_seconds = time.perf_counter() - context_started
        atoms_started = time.perf_counter()
        atoms_by_seed = tuple(_seed_symbolic_atoms(seed) for seed in ordered)
        seed_atoms_seconds = time.perf_counter() - atoms_started
        projection_started = time.perf_counter()
        projected = _symbolic_reynolds_batch(
            ordered,
            atoms_by_seed,
            coordinate=coordinate,
            contexts=contexts,
        )
        symbolic_projection_seconds = time.perf_counter() - projection_started
        rank_started = time.perf_counter()
        projected_norms = np.sqrt(
            np.asarray(projected.power(2).sum(axis=0)).ravel()
        )
        internal_error = float(action_artifact.get("maximum_action_error_bound", 0.0))
        absolute_error_bounds: list[float] = []
        for seed in ordered:
            raw_norm = _coefficient_norm(seed.coefficients)
            components = _propagated_error_components(
                raw_norm,
                dim=seed.dim,
                degree=coordinate.max_degree,
                group=group,
            )
            if internal_error > 0.0:
                components["factorized_internal_action"] = float(
                    (2.0 * internal_error + internal_error**2)
                    * max(raw_norm, np.finfo(float).tiny)
                )
            error_bound = float(sum(components.values()))
            absolute_error_bounds.extend((error_bound, error_bound))
        relative_error_bounds = np.asarray(absolute_error_bounds, dtype=np.float64) / np.maximum(
            projected_norms,
            np.finfo(float).tiny,
        )
        preliminary_selected, preliminary_proofs = _independent_projected_columns(
            projected,
        )
        reduced_local, certified_proofs = _independent_projected_columns(
            projected[:, preliminary_selected].tocsc(),
            relative_error_bounds=relative_error_bounds[
                np.asarray(preliminary_selected, dtype=np.int64)
            ],
            confirm_factor=NullClassificationPolicy().confirm_factor,
        )
        selected = tuple(preliminary_selected[index] for index in reduced_local)
        proofs = (
            {
                "stage": "machine_rank_preselection",
                "components": list(preliminary_proofs),
                "retained": int(len(preliminary_selected)),
            },
            {
                "stage": "per_column_propagated_error_certification",
                "components": list(certified_proofs),
                "retained": int(len(selected)),
            },
        )
        rank_selection_seconds = time.perf_counter() - rank_started
        physical_basis = projected[:, selected].tocsc()
        selector = sparse.csc_matrix(
            (
                np.ones(len(selected), dtype=np.float64),
                (np.asarray(selected, dtype=np.int64), np.arange(len(selected), dtype=np.int64)),
            ),
            shape=(projected.shape[1], len(selected)),
        )
        return SymmetryFirstFixedSpace(
            rank=int(len(selected)),
            physical_basis=physical_basis,
            fixed_vocabulary_coordinates=selector,
            metadata={
                "compiler": "closed_symbolic_atom_reynolds_v1",
                "logical_channel_ids": list(channel_ids),
                "selected_channel_ids": [channel_ids[index] for index in selected],
                "projected_channel_count": int(projected.shape[1]),
                "retained_channel_count": int(len(selected)),
                "rank_proofs": list(proofs),
                "factorized_group_actions": dict(action_artifact),
                "timings_seconds": {
                    "group_context": float(group_context_seconds),
                    "seed_atoms": float(seed_atoms_seconds),
                    "symbolic_projection": float(symbolic_projection_seconds),
                    "rank_selection": float(rank_selection_seconds),
                },
            },
        )
    vocabulary = build_raw_real_vocabulary(ordered, coordinate=coordinate)
    generator_actions = {
        str(name): compile_factorized_raw_seed_action(
            seeds=ordered,
            factorized_action=action,
        )
        for name, action in factorized_actions.items()
    }
    generator_images = {
        name: (vocabulary.matrix @ action).tocsc()
        for name, action in generator_actions.items()
    }
    generator_images["adjoint"] = build_raw_adjoint_image(
        ordered,
        coordinate=coordinate,
    )
    antiunitary_parities = {
        str(name): bool(action.antiunitary)
        for name, action in factorized_actions.items()
    }
    antiunitary_parities["adjoint"] = True
    fixed = compile_generator_fixed_vocabulary(
        vocabulary.matrix,
        generator_images=generator_images,
        antiunitary_parities=antiunitary_parities,
        logical_channel_ids=vocabulary.channel_ids,
        materialize_ambient_vectors=False,
    )
    coordinates = fixed.fixed_vocabulary_coordinates
    physical_basis = sparse.csc_matrix(vocabulary.matrix @ coordinates)
    physical_basis.eliminate_zeros()
    physical_basis.sort_indices()
    return SymmetryFirstFixedSpace(
        rank=int(fixed.rank),
        physical_basis=physical_basis,
        fixed_vocabulary_coordinates=coordinates,
        metadata=fixed.metadata,
    )


def compile_symbolic_atom_candidate_group(
    seeds: Sequence[RawPolynomialSeed],
    *,
    coordinate: PolynomialCoordinateBasis,
    group: Any,
    factorized_actions: Mapping[str, Any],
    internal_actions_by_word: Mapping[tuple[str, ...], sparse.spmatrix] | None = None,
    factorized_group_artifact: Mapping[str, Any] | None = None,
) -> CandidateResponseSet:
    """Compile only independent finite-p channels through symbolic atoms."""

    ordered = _validate_seeds(seeds)
    compiled = compile_symmetry_first_fixed_space(
        ordered,
        coordinate=coordinate,
        factorized_actions=factorized_actions,
        group=group,
        internal_actions_by_word=internal_actions_by_word,
        factorized_group_artifact=factorized_group_artifact,
    )
    logical = {
        f"{seed.seed_id}:{component}": (seed, component)
        for seed in ordered
        for component in ("real", "imag")
    }
    selected_ids = tuple(str(value) for value in compiled.metadata["selected_channel_ids"])
    action_artifact = dict(compiled.metadata["factorized_group_actions"])
    policy = NullClassificationPolicy()
    channels: list[ResponseChannel] = []
    materialize_started = time.perf_counter()
    for column_index, channel_id in enumerate(selected_ids):
        seed, component = logical[channel_id]
        coefficients = _coefficient_vector_to_response_map(
            compiled.physical_basis[:, column_index],
            coordinate=coordinate,
            dim=seed.dim,
        )
        response_norm = _coefficient_norm(coefficients)
        raw_norm = _coefficient_norm(seed.coefficients)
        error_components = _propagated_error_components(
            raw_norm,
            dim=seed.dim,
            degree=coordinate.max_degree,
            group=group,
        )
        internal_error = float(action_artifact.get("maximum_action_error_bound", 0.0))
        if internal_error > 0.0:
            error_components["factorized_internal_action"] = float(
                (2.0 * internal_error + internal_error**2)
                * max(raw_norm, np.finfo(float).tiny)
            )
        error_bound = float(sum(error_components.values()))
        channels.append(
            ResponseChannel(
                channel_id=channel_id,
                seed_id=str(seed.seed_id),
                component=str(component),
                coefficients=coefficients,
                unnormalized_norm=response_norm,
                propagated_error_bound=error_bound,
                classification=policy.classify(response_norm, error_bound),
                response_scale=response_norm,
                support_component=str(seed.support_component),
                metadata={
                    **dict(seed.metadata),
                    "symbolic_atom_compiler": "closed_symbolic_atom_reynolds_v1",
                },
                error_bound_components=error_components,
            )
        )
    channel_materialization_seconds = time.perf_counter() - materialize_started
    logical_count = 2 * len(ordered)
    retained_count = len(channels)
    return CandidateResponseSet(
        coordinate=coordinate,
        channels=tuple(channels),
        dim=int(ordered[0].dim),
        group_artifact=group.artifact(),
        null_policy=policy,
        adjoint_artifact={
            "certification": "closed_symbolic_atom_reynolds_v1",
            "logical_candidate_channel_count": int(logical_count),
            "authored_ordered_seed_count": int(len(ordered)),
            "adjoint_orbit_descriptor_count": int(len(ordered)),
            "hermitian_ambient_channel_count": int(logical_count),
            "physically_materialized_projected_channel_count": int(retained_count),
            "physically_compiled_representative_channel_count": int(retained_count),
            "adjoint_certified_dropped_channel_count": int(logical_count - retained_count),
            "rank_proofs": list(compiled.metadata["rank_proofs"]),
            "factorized_group_actions": action_artifact,
            "reynolds_internal_action": action_artifact,
            "timings_seconds": {
                **dict(compiled.metadata["timings_seconds"]),
                "channel_materialization": float(channel_materialization_seconds),
            },
        },
    )


__all__ = [
    "RawRealVocabulary",
    "SymmetryFirstFixedSpace",
    "build_raw_adjoint_image",
    "build_raw_real_vocabulary",
    "compile_symbolic_atom_candidate_group",
    "compile_symmetry_first_fixed_space",
]

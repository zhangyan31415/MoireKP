"""Graded polynomial primitives for case-derived response compilation.

The public response basis is expressed in one global polynomial coordinate,
while authored continuum terms are naturally filtered by their total momentum
degree.  This module contains the small, exact degree representations used by
the filtered compiler; it deliberately does not encode material-specific term
profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import time
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.linalg
from scipy import sparse

from .response_basis import (
    CandidateResponseSet,
    FiniteGroupElement,
    NullClassificationPolicy,
    PolynomialCoordinateBasis,
    RawPolynomialSeed,
    ResponseChannel,
    _coefficient_norm,
    _coefficient_vector_to_response_map,
    _monomial_pullback_table,
    _propagated_error_components,
)
from .response_basis_factorized import (
    FactorizedTermActionError,
    compile_factorized_group_element_actions,
    compile_factorized_raw_seed_action,
)
from .response_basis_adjoint import (
    JointAdjointSeed,
    JointAdjointSeedKey,
    canonicalize_joint_adjoint_seeds,
    certify_joint_adjoint_coefficients,
)
from .response_basis_symmetry_first import (
    _SymbolicGroupContext,
    _column_routes,
    _seed_symbolic_atoms,
    _symbolic_reynolds_batch,
    _validate_seeds,
    build_raw_real_vocabulary,
)
from .response_basis_local_fixed import (
    LOCAL_FIXED_GRAY_ZONE_FACTOR_V1,
    LOCAL_REYNOLDS_ALGORITHM_V1,
    LocalResponseCompilationError,
    build_local_action_blocks,
    summarize_local_action_blocks,
    solve_local_reynolds_block,
)


class GradedCompilationUnavailable(RuntimeError):
    """A filtered-rank certificate could not be established."""

    def __init__(
        self,
        reason: str,
        *,
        certificate: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(str(reason))
        self.reason = str(reason)
        self.certificate = dict(certificate or {})


@dataclass(frozen=True)
class FilteredSeedDescriptor:
    """The nominal homogeneous head of one Q-centred polynomial seed.

    ``RawPolynomialSeed.coefficients`` may contain every degree below the
    authored order because expanding ``(k-Q)^r (k-Q)^*^s`` in the global
    coordinate is triangular.  The nominal degree must therefore come from the
    term key, never from a scan that assumes every coefficient has one degree.
    """

    seed_index: int
    seed_id: str
    term_index: int | None
    nominal_monomial: tuple[int, int]
    nominal_degree: int
    actual_degrees: tuple[int, ...]
    top_coefficients: sparse.csr_matrix
    structural_support_id: str


@dataclass(frozen=True)
class GradedColumnSelection:
    """Certified owner-column selection from a filtered real vocabulary."""

    selected_owner_indices: tuple[int, ...]
    logical_to_selected_coordinates: sparse.csc_matrix
    residual_error_bounds: tuple[float, ...]
    degree_proofs: tuple[Mapping[str, Any], ...]

    @property
    def selected_indices(self) -> tuple[int, ...]:
        return self.selected_owner_indices


@dataclass(frozen=True)
class StructuralOrbit:
    """One closed orbit of actual matrix/Q-pair support directions."""

    orbit_index: int
    support_ids: tuple[str, ...]
    representative_seed_indices: tuple[int, ...]


@dataclass(frozen=True)
class StructuralCompilationPlan:
    """Target-independent structural partition of an authored seed envelope."""

    representative_seed_indices: tuple[int, ...]
    duplicate_seed_groups: tuple[tuple[int, ...], ...]
    structural_orbits: tuple[StructuralOrbit, ...]
    zero_seed_indices: tuple[int, ...]
    structural_probe_seed_indices: tuple[int, ...]
    family_degree_limits: Mapping[str, int]


def _homogeneous_monomials(
    coordinate: PolynomialCoordinateBasis,
    degree: int,
) -> tuple[tuple[int, int], ...]:
    requested = int(degree)
    if requested < 0 or requested > int(coordinate.max_degree):
        raise ValueError(
            "homogeneous degree must lie within the declared polynomial basis; "
            f"got degree={requested}, max_degree={coordinate.max_degree}"
        )
    return tuple(
        monomial
        for monomial in coordinate.monomials
        if sum(monomial) == requested
    )


def compile_homogeneous_momentum_action(
    element: FiniteGroupElement,
    *,
    coordinate: PolynomialCoordinateBasis,
    degree: int,
) -> sparse.csc_matrix:
    """Return the degree-``d`` principal action of a momentum pullback.

    Columns are input monomials and rows are output monomials, both in the
    ordering frozen by :class:`PolynomialCoordinateBasis`.  If the polynomial
    origin is not fixed by ``element``, the full affine pullback also contains
    lower-degree terms.  Those terms belong to the strict lower part of the
    degree filtration and are intentionally omitted here; the returned
    principal block remains exact.
    """

    basis = _homogeneous_monomials(coordinate, degree)
    index = {monomial: position for position, monomial in enumerate(basis)}
    pullbacks = _monomial_pullback_table(element, coordinate)
    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    requested = int(degree)
    for source_index, source in enumerate(basis):
        for target, coefficient in pullbacks[source].items():
            target_degree = sum(target)
            if target_degree > requested:
                raise ValueError(
                    "affine momentum pullback increased total polynomial degree"
                )
            if target_degree != requested or complex(coefficient) == 0.0j:
                continue
            # ``_apply_group_element`` performs the polynomial pullback first
            # and then conjugates an antiunitary coefficient while exchanging
            # w and wbar.  Keep that convention here so callers can uniformly
            # interpret the result as C z (unitary) or C conjugate(z)
            # (antiunitary), without a second monomial swap during realifying.
            output = (target[1], target[0]) if element.antiunitary else target
            value = (
                complex(coefficient).conjugate()
                if element.antiunitary
                else complex(coefficient)
            )
            rows.append(index[output])
            columns.append(source_index)
            values.append(value)
    action = sparse.coo_matrix(
        (
            values,
            (rows, columns),
        ),
        shape=(len(basis), len(basis)),
        dtype=complex,
    ).tocsc()
    action.sum_duplicates()
    action.eliminate_zeros()
    action.sort_indices()
    return action


def _canonical_sparse_direction(
    matrix: sparse.spmatrix,
) -> sparse.csr_matrix:
    """Normalize one nonzero sparse matrix direction deterministically."""

    value = sparse.csr_matrix(matrix, dtype=np.complex128)
    value.sum_duplicates()
    value.eliminate_zeros()
    value.sort_indices()
    if value.nnz == 0:
        return value
    first = complex(value.data[0])
    if first == 0.0j or not np.isfinite(first.real) or not np.isfinite(first.imag):
        raise GradedCompilationUnavailable(
            "a structural support contains a non-finite leading coefficient",
            certificate={"check": "structural_direction_normalization"},
        )
    value = sparse.csr_matrix(value / first, dtype=np.complex128)
    value.sum_duplicates()
    value.eliminate_zeros()
    value.sort_indices()
    return value


def _sparse_direction_digest(matrix: sparse.spmatrix) -> str:
    value = _canonical_sparse_direction(matrix).tocoo()
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(value.row, dtype="<i8").tobytes())
    digest.update(np.asarray(value.col, dtype="<i8").tobytes())
    data = np.asarray(value.data, dtype="<c16").copy()
    data.real[data.real == 0.0] = 0.0
    data.imag[data.imag == 0.0] = 0.0
    digest.update(data.tobytes())
    return digest.hexdigest()


def _physical_seed_digest(
    seed: RawPolynomialSeed,
    *,
    coordinate: PolynomialCoordinateBasis,
) -> str:
    """Hash one complete polynomial direction using canonical CSR buffers.

    Two seeds receive the same digest exactly when their coefficient stacks
    differ only by one nonzero complex scalar.  Monomials and sparse structure
    are encoded explicitly, while whole CSR buffers are hashed at once rather
    than iterating over every nonzero in Python.
    """

    matrices: list[tuple[tuple[int, int], sparse.csr_matrix]] = []
    leading: complex | None = None
    for monomial in coordinate.monomials:
        raw_matrix = seed.coefficients.get(monomial)
        if raw_matrix is None:
            continue
        matrix = sparse.csr_matrix(raw_matrix, dtype=np.complex128, copy=False)
        if (
            not bool(matrix.has_canonical_format)
            or not bool(matrix.has_sorted_indices)
            or np.any(matrix.data == 0.0j)
        ):
            matrix = sparse.csr_matrix(matrix, dtype=np.complex128, copy=True)
            matrix.sum_duplicates()
            matrix.eliminate_zeros()
            matrix.sort_indices()
        if matrix.nnz == 0:
            continue
        if leading is None:
            leading = complex(matrix.data[0])
            if (
                leading == 0.0j
                or not np.isfinite(leading.real)
                or not np.isfinite(leading.imag)
            ):
                raise GradedCompilationUnavailable(
                    f"seed {seed.seed_id!r} has an invalid physical leading coefficient",
                    certificate={"seed_id": str(seed.seed_id)},
                )
        matrices.append((monomial, matrix))

    digest = hashlib.sha256()
    digest.update(b"physical-polynomial-direction-csr-v2")
    digest.update(np.asarray((seed.dim, seed.dim), dtype="<i8").tobytes())
    if leading is None:
        digest.update(b"zero-polynomial-direction")
        return digest.hexdigest()

    for monomial, matrix in matrices:
        digest.update(
            np.asarray(
                (int(monomial[0]), int(monomial[1]), int(matrix.nnz)),
                dtype="<i8",
            ).tobytes()
        )
        digest.update(np.asarray(matrix.indptr, dtype="<i8").tobytes())
        digest.update(np.asarray(matrix.indices, dtype="<i8").tobytes())
        normalized = np.asarray(matrix.data / leading, dtype="<c16").copy()
        normalized.real[normalized.real == 0.0] = 0.0
        normalized.imag[normalized.imag == 0.0] = 0.0
        digest.update(normalized.tobytes())
    return digest.hexdigest()


def describe_filtered_seed(
    seed: RawPolynomialSeed,
    *,
    coordinate: PolynomialCoordinateBasis,
    seed_index: int = 0,
) -> FilteredSeedDescriptor:
    """Describe a seed without mistaking its triangular tail for its order."""

    term_key = seed.metadata.get("term_key")
    if not isinstance(term_key, Mapping):
        raise GradedCompilationUnavailable(
            f"graded seed {seed.seed_id!r} has no term_key metadata",
            certificate={"seed_id": str(seed.seed_id), "check": "term_key"},
        )
    try:
        nominal = (int(term_key["Mz"]), int(term_key["Mz_star"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise GradedCompilationUnavailable(
            f"graded seed {seed.seed_id!r} has an invalid nominal monomial",
            certificate={"seed_id": str(seed.seed_id), "check": "nominal_monomial"},
        ) from exc
    if min(nominal) < 0 or sum(nominal) > int(coordinate.max_degree):
        raise GradedCompilationUnavailable(
            f"graded seed {seed.seed_id!r} lies outside the polynomial coordinate",
            certificate={
                "seed_id": str(seed.seed_id),
                "nominal_monomial": list(nominal),
                "coordinate_max_degree": int(coordinate.max_degree),
            },
        )

    coordinate_monomials = set(coordinate.monomials)
    actual_degrees: set[int] = set()
    for raw_monomial, raw_matrix in seed.coefficients.items():
        monomial = (int(raw_monomial[0]), int(raw_monomial[1]))
        if monomial not in coordinate_monomials:
            raise GradedCompilationUnavailable(
                f"graded seed {seed.seed_id!r} contains an undeclared monomial",
                certificate={
                    "seed_id": str(seed.seed_id),
                    "monomial": list(monomial),
                },
            )
        matrix = sparse.csr_matrix(raw_matrix, dtype=np.complex128)
        if matrix.nnz:
            degree = int(sum(monomial))
            actual_degrees.add(degree)
            if degree > sum(nominal):
                raise GradedCompilationUnavailable(
                    f"graded seed {seed.seed_id!r} increases its nominal degree",
                    certificate={
                        "seed_id": str(seed.seed_id),
                        "nominal_degree": int(sum(nominal)),
                        "observed_degree": degree,
                    },
                )

    top = sparse.csr_matrix(
        seed.coefficients.get(
            nominal,
            sparse.csr_matrix((seed.dim, seed.dim), dtype=np.complex128),
        ),
        dtype=np.complex128,
    )
    top.sum_duplicates()
    top.eliminate_zeros()
    top.sort_indices()
    term_index_value = seed.metadata.get("term_index")
    term_index = None if term_index_value is None else int(term_index_value)
    return FilteredSeedDescriptor(
        seed_index=int(seed_index),
        seed_id=str(seed.seed_id),
        term_index=term_index,
        nominal_monomial=nominal,
        nominal_degree=int(sum(nominal)),
        actual_degrees=tuple(sorted(actual_degrees)),
        top_coefficients=top,
        # Family, tag, authored p label, and maximum order are provenance.
        # Actual top matrix/Q-pair amplitudes alone define the structural fiber.
        structural_support_id=_sparse_direction_digest(top),
    )


def projected_row_degrees(
    *,
    coordinate: PolynomialCoordinateBasis,
    dim: int,
) -> np.ndarray:
    """Map rows of the symbolic real stack to total polynomial degree."""

    matrix_dim = int(dim)
    if matrix_dim <= 0:
        raise ValueError("projected row degrees require a positive matrix dimension")
    complex_degrees = np.repeat(
        np.asarray([sum(item) for item in coordinate.monomials], dtype=np.int64),
        matrix_dim * matrix_dim,
    )
    result = np.concatenate([complex_degrees, complex_degrees])
    result.setflags(write=False)
    return result


def _seed_family(seed: RawPolynomialSeed) -> str:
    for key in ("term_name", "family", "tag"):
        value = str(seed.metadata.get(key, "")).strip()
        if value:
            return value
    return str(seed.support_component)


def _seed_provenance(
    seed: RawPolynomialSeed,
    descriptor: FilteredSeedDescriptor,
) -> dict[str, Any]:
    return {
        "seed_id": str(seed.seed_id),
        "term_index": descriptor.term_index,
        "family": _seed_family(seed),
        "support_component": str(seed.support_component),
        "nominal_monomial": list(descriptor.nominal_monomial),
        "nominal_degree": int(descriptor.nominal_degree),
        "term_space_policy": str(seed.metadata.get("term_space_policy", "")),
    }


def build_structural_compilation_plan(
    seeds: Sequence[RawPolynomialSeed],
    *,
    coordinate: PolynomialCoordinateBasis,
    group: Any,
    internal_actions_by_word: Mapping[tuple[str, ...], sparse.spmatrix],
    maximum_action_error_bound: float = 0.0,
    descriptors: Sequence[FilteredSeedDescriptor] | None = None,
    compute_cyclic_orbits: bool = True,
) -> StructuralCompilationPlan:
    """Deduplicate physical seeds and close actual support fibers into orbits.

    The probe is independent of polynomial degree and family.  It discovers
    only how orbital/Q/harmonic matrix directions transform; every configured
    polynomial degree is still projected and ranked separately downstream.
    """

    ordered = _validate_seeds(seeds)
    described = (
        tuple(descriptors)
        if descriptors is not None
        else tuple(
            describe_filtered_seed(seed, coordinate=coordinate, seed_index=index)
            for index, seed in enumerate(ordered)
        )
    )
    if len(described) != len(ordered):
        raise ValueError("structural descriptors must match authored seeds")

    duplicate_by_direction: dict[str, list[int]] = {}
    zero_seed_indices: list[int] = []
    for index, seed in enumerate(ordered):
        nonzero = any(
            sparse.csr_matrix(matrix).nnz
            for matrix in seed.coefficients.values()
        )
        if not nonzero:
            zero_seed_indices.append(index)
            continue
        duplicate_by_direction.setdefault(
            _physical_seed_digest(seed, coordinate=coordinate), []
        ).append(index)
    duplicate_groups = tuple(
        tuple(indices)
        for _digest, indices in sorted(
            duplicate_by_direction.items(), key=lambda item: tuple(item[1])
        )
    )
    representative_indices = tuple(
        group_indices[0] for group_indices in duplicate_groups
    )

    family_degree_limits: dict[str, int] = {}
    for seed, descriptor in zip(ordered, described):
        family = _seed_family(seed)
        family_degree_limits[family] = max(
            int(descriptor.nominal_degree),
            family_degree_limits.get(family, -1),
        )

    support_seed_indices: dict[str, list[int]] = {}
    for seed_index in representative_indices:
        support_seed_indices.setdefault(
            described[seed_index].structural_support_id, []
        ).append(seed_index)
    support_ids = tuple(sorted(support_seed_indices))
    if not support_ids:
        return StructuralCompilationPlan(
            representative_seed_indices=(),
            duplicate_seed_groups=duplicate_groups,
            structural_orbits=(),
            zero_seed_indices=tuple(zero_seed_indices),
            structural_probe_seed_indices=(),
            family_degree_limits=dict(sorted(family_degree_limits.items())),
        )

    if not bool(compute_cyclic_orbits):
        return StructuralCompilationPlan(
            representative_seed_indices=representative_indices,
            duplicate_seed_groups=duplicate_groups,
            structural_orbits=tuple(
                StructuralOrbit(
                    orbit_index=orbit_index,
                    support_ids=(support_id,),
                    representative_seed_indices=tuple(
                        sorted(support_seed_indices[support_id])
                    ),
                )
                for orbit_index, support_id in enumerate(support_ids)
            ),
            zero_seed_indices=tuple(zero_seed_indices),
            structural_probe_seed_indices=tuple(
                min(
                    support_seed_indices[support_id],
                    key=lambda index: (described[index].nominal_degree, index),
                )
                for support_id in support_ids
            ),
            family_degree_limits=dict(sorted(family_degree_limits.items())),
        )

    authored_support_matrices = tuple(
        _canonical_sparse_direction(
            described[support_seed_indices[support_id][0]].top_coefficients
        )
        for support_id in support_ids
    )
    dim = int(ordered[0].dim)
    image_columns: list[sparse.csc_matrix] = []
    image_sources: list[int] = []
    closure_digests_by_source: list[set[str]] = [set() for _ in support_ids]
    for source_index, source_matrix in enumerate(authored_support_matrices):
        for element in group.elements:
            word = tuple(str(value) for value in element.canonical_word)
            try:
                internal = sparse.csr_matrix(
                    internal_actions_by_word[word], dtype=np.complex128
                )
            except KeyError as exc:
                raise GradedCompilationUnavailable(
                    "structural action is missing a finite-group word",
                    certificate={"missing_word": list(word)},
                ) from exc
            source = (
                source_matrix.conjugate()
                if bool(element.antiunitary)
                else source_matrix
            )
            transformed = (internal @ source @ internal.getH()).tocsr()
            for image in (transformed, transformed.getH()):
                image = sparse.csr_matrix(image, dtype=np.complex128)
                image.sum_duplicates()
                image.eliminate_zeros()
                image.sort_indices()
                norm = float(
                    np.sqrt(np.vdot(image.data, image.data).real)
                )
                if not np.isfinite(norm) or norm <= 0.0:
                    raise GradedCompilationUnavailable(
                        "symmetry/adjoint mapped a structural support to zero",
                        certificate={
                            "word": list(word),
                            "source_support_id": support_ids[source_index],
                        },
                    )
                normalized = sparse.csr_matrix(image / norm)
                closure_digests_by_source[source_index].add(
                    _sparse_direction_digest(normalized)
                )
                image_columns.append(
                    normalized.reshape((dim * dim, 1), order="C").tocsc()
                )
                image_sources.append(int(source_index))

    cyclic_images = sparse.hstack(image_columns, format="csc")
    gram_matrix = (cyclic_images.getH() @ cyclic_images).tocsc()
    gram_matrix.sum_duplicates()
    gram_matrix.eliminate_zeros()
    gram = gram_matrix.tocoo()
    adjacency_rows = list(range(len(support_ids)))
    adjacency_columns = list(range(len(support_ids)))
    for image_row, image_col, overlap in zip(gram.row, gram.col, gram.data):
        overlap_value = complex(overlap)
        if not np.isfinite(overlap_value.real) or not np.isfinite(
            overlap_value.imag
        ):
            raise GradedCompilationUnavailable(
                "structural cyclic-image Gram matrix is non-finite",
                certificate={"check": "structural_cyclic_image_gram"},
            )
        if overlap_value == 0.0j:
            continue
        source = image_sources[int(image_row)]
        target = image_sources[int(image_col)]
        adjacency_rows.extend((source, target))
        adjacency_columns.extend((target, source))
    adjacency_matrix = sparse.coo_matrix(
        (
            np.ones(len(adjacency_rows), dtype=np.int8),
            (
                np.asarray(adjacency_rows, dtype=np.int64),
                np.asarray(adjacency_columns, dtype=np.int64),
            ),
        ),
        shape=(len(support_ids), len(support_ids)),
    ).tocsr()
    adjacency_matrix.sum_duplicates()
    component_count, labels = sparse.csgraph.connected_components(
        adjacency_matrix,
        directed=False,
        return_labels=True,
    )
    orbits: list[StructuralOrbit] = []
    for orbit_index in range(int(component_count)):
        support_positions = np.flatnonzero(labels == orbit_index)
        authored_orbit_support_ids = tuple(
            support_ids[int(value)] for value in support_positions
        )
        closure_digests: set[str] = set()
        for value in support_positions:
            closure_digests.update(closure_digests_by_source[int(value)])
        virtual_support_ids = tuple(
            f"virtual:{digest}"
            for digest in sorted(closure_digests.difference(support_ids))
        )
        orbit_support_ids = authored_orbit_support_ids + virtual_support_ids
        orbit_seed_indices = tuple(
            sorted(
                seed_index
                for support_id in authored_orbit_support_ids
                for seed_index in support_seed_indices[support_id]
            )
        )
        orbits.append(
            StructuralOrbit(
                orbit_index=int(orbit_index),
                support_ids=orbit_support_ids,
                representative_seed_indices=orbit_seed_indices,
            )
        )
    return StructuralCompilationPlan(
        representative_seed_indices=representative_indices,
        duplicate_seed_groups=duplicate_groups,
        structural_orbits=tuple(orbits),
        zero_seed_indices=tuple(zero_seed_indices),
        structural_probe_seed_indices=tuple(
            min(
                support_seed_indices[support_id],
                key=lambda index: (described[index].nominal_degree, index),
            )
            for support_id in support_ids
        ),
        family_degree_limits=dict(sorted(family_degree_limits.items())),
    )


def _structural_route_closure_diagnostic(
    structural_plan: StructuralCompilationPlan,
    *,
    descriptors: Sequence[FilteredSeedDescriptor],
    coordinate: PolynomialCoordinateBasis,
    contexts: Sequence[_SymbolicGroupContext],
) -> dict[str, Any]:
    """Return conservative closed-owner block bounds without building actions."""

    described = tuple(descriptors)
    authored_support_ids = {
        described[index].structural_support_id
        for index in structural_plan.representative_seed_indices
    }
    closed_direction_count = int(
        sum(len(orbit.support_ids) for orbit in structural_plan.structural_orbits)
    )
    authored_block_dimensions: list[int] = []
    closed_upper_block_dimensions: list[int] = []
    for orbit in structural_plan.structural_orbits:
        seed_indices = tuple(int(value) for value in orbit.representative_seed_indices)
        authored_block_dimensions.append(2 * len(seed_indices))
        maximum_degree = max(
            (described[index].nominal_degree for index in seed_indices),
            default=0,
        )
        monomial_count = sum(
            int(sum(monomial) <= maximum_degree)
            for monomial in coordinate.monomials
        )
        closed_upper_block_dimensions.append(
            2 * len(orbit.support_ids) * monomial_count
        )

    monomial_route_nonzero_count = 0
    lower_degree_edge_count = 0
    for context in contexts:
        for source, targets in context.monomial_pullbacks.items():
            source_degree = int(sum(source))
            for target, coefficient in targets.items():
                if complex(coefficient) == 0.0j:
                    continue
                monomial_route_nonzero_count += 1
                lower_degree_edge_count += int(sum(target) < source_degree)

    return {
        "schema_version": "structural-route-closure-diagnostic-v1",
        "exact_closed_owner_actions_available": False,
        "authored_real_owner_count": int(
            2 * len(structural_plan.representative_seed_indices)
        ),
        "closed_structural_direction_count": closed_direction_count,
        "virtual_structural_direction_count": int(
            closed_direction_count - len(authored_support_ids)
        ),
        "structural_block_count": int(len(structural_plan.structural_orbits)),
        "authored_real_owner_block_dimensions": authored_block_dimensions,
        "maximum_authored_real_owner_block_dimension": int(
            max(authored_block_dimensions, default=0)
        ),
        "closed_real_owner_upper_bound_block_dimensions": (
            closed_upper_block_dimensions
        ),
        "maximum_closed_real_owner_upper_bound_block_dimension": int(
            max(closed_upper_block_dimensions, default=0)
        ),
        "closed_real_owner_upper_bound": int(sum(closed_upper_block_dimensions)),
        "monomial_route_nonzero_count": int(monomial_route_nonzero_count),
        "lower_degree_monomial_route_edge_count": int(lower_degree_edge_count),
    }


def _homogeneous_action_certificates(
    *,
    group: Any,
    coordinate: PolynomialCoordinateBasis,
    contexts: Sequence[_SymbolicGroupContext],
    degrees: Sequence[int],
) -> dict[int, Mapping[str, Any]]:
    """Compile and bind every actual homogeneous action to projection contexts."""

    if len(contexts) != len(group.elements):
        raise GradedCompilationUnavailable(
            "homogeneous actions and symbolic contexts have different group sizes"
        )
    certificates: dict[int, Mapping[str, Any]] = {}
    for degree in sorted(set(int(value) for value in degrees)):
        element_records: list[dict[str, Any]] = []
        maximum_residual = 0.0
        for element, context in zip(group.elements, contexts):
            action = compile_homogeneous_momentum_action(
                element,
                coordinate=coordinate,
                degree=degree,
            ).tocsc()
            basis = _homogeneous_monomials(coordinate, degree)
            basis_index = {monomial: index for index, monomial in enumerate(basis)}
            rows: list[int] = []
            columns: list[int] = []
            values: list[complex] = []
            for column, source in enumerate(basis):
                for target, coefficient in context.monomial_pullbacks[source].items():
                    if sum(target) != degree:
                        continue
                    output = (target[1], target[0]) if element.antiunitary else target
                    value = (
                        complex(coefficient).conjugate()
                        if element.antiunitary
                        else complex(coefficient)
                    )
                    rows.append(basis_index[output])
                    columns.append(column)
                    values.append(value)
            context_action = sparse.coo_matrix(
                (values, (rows, columns)), shape=action.shape, dtype=np.complex128
            ).tocsc()
            residual = float(sparse.linalg.norm(action - context_action))
            maximum_residual = max(maximum_residual, residual)
            digest = hashlib.sha256()
            digest.update(np.asarray(action.shape, dtype="<i8").tobytes())
            digest.update(np.asarray(action.indptr, dtype="<i8").tobytes())
            digest.update(np.asarray(action.indices, dtype="<i8").tobytes())
            digest.update(np.asarray(action.data, dtype="<c16").tobytes())
            element_records.append(
                {
                    "word": [str(value) for value in element.canonical_word],
                    "shape": [int(action.shape[0]), int(action.shape[1])],
                    "rank": int(np.linalg.matrix_rank(action.toarray())),
                    "context_residual": residual,
                    "action_hash": digest.hexdigest(),
                }
            )
        if maximum_residual > 64.0 * np.finfo(float).eps * max(1, degree + 1):
            raise GradedCompilationUnavailable(
                "homogeneous action disagrees with the exact projection context",
                certificate={
                    "degree": int(degree),
                    "maximum_context_residual": maximum_residual,
                },
            )
        certificates[int(degree)] = {
            "degree": int(degree),
            "dimension": int(degree + 1),
            "maximum_context_residual": float(maximum_residual),
            "elements": element_records,
        }
    return certificates


def _connected_column_components(
    value: sparse.csc_matrix,
    columns: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    """Return exact-support components, with indices local to ``value``."""

    if columns.size == 0:
        return ()
    selected = value[:, columns].tocsc()
    incidence = selected.copy()
    incidence.data = np.ones(incidence.nnz, dtype=np.bool_)
    bipartite = sparse.bmat(
        ((None, incidence), (incidence.T, None)),
        format="csr",
    )
    _count, bipartite_labels = sparse.csgraph.connected_components(
        bipartite,
        directed=False,
        return_labels=True,
    )
    labels = bipartite_labels[selected.shape[0] :]
    grouped: dict[int, list[int]] = {}
    for local_index, label in enumerate(labels):
        grouped.setdefault(int(label), []).append(int(columns[local_index]))
    return tuple(
        tuple(indices)
        for indices in sorted(grouped.values(), key=lambda item: tuple(item))
    )


def _certified_component_elimination(
    block: sparse.csc_matrix,
    component: Sequence[int],
    *,
    error_bounds: np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Choose residual pivots and certify cancellation of one degree block."""

    indices = np.asarray(tuple(component), dtype=np.int64)
    vectors = block[:, indices].tocsc()
    active_rows = np.unique(vectors.indices)
    dense = np.asarray(vectors[active_rows, :].toarray(), dtype=np.float64)
    norms = np.linalg.norm(dense, axis=0)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise GradedCompilationUnavailable(
            "a graded incidence component contains a non-finite or empty column",
            certificate={"component": [int(value) for value in indices]},
        )
    normalized = dense / norms[np.newaxis, :]
    _q_factor, r_factor, permutation = scipy.linalg.qr(
        normalized,
        pivoting=True,
        mode="economic",
        check_finite=False,
    )
    diagonal = np.abs(np.diag(r_factor))
    machine_tolerance = float(
        np.finfo(float).eps
        * max(normalized.shape)
        * max(float(np.linalg.norm(normalized, ord="fro")), 1.0)
    )
    proposed_rank = int(np.count_nonzero(diagonal > machine_tolerance))
    proposed_rank = max(1, min(proposed_rank, len(indices)))
    pivot_local = [int(value) for value in permutation[:proposed_rank]]

    # A machine-rank proposal is only accepted when every cancelled top-degree
    # residual lies inside its propagated absolute error.  Otherwise retain the
    # offending owner as another pivot and retry.  This is conservative and
    # never assumes that a top-degree dependency also holds below it.
    while True:
        pivot_set = set(pivot_local)
        nonpivot_local = [
            index for index in range(len(indices)) if index not in pivot_set
        ]
        if not nonpivot_local:
            solution = np.zeros((len(pivot_local), 0), dtype=np.float64)
            new_errors = np.zeros(0, dtype=np.float64)
            residual_norms = np.zeros(0, dtype=np.float64)
            solve_roundoff = np.zeros(0, dtype=np.float64)
            break
        pivot_positions = np.asarray(pivot_local, dtype=np.int64)
        nonpivot_positions = np.asarray(nonpivot_local, dtype=np.int64)
        pivot_dense = dense[:, pivot_positions]
        nonpivot_dense = dense[:, nonpivot_positions]
        normalized_solution, _residues, numerical_rank, _singular = (
            scipy.linalg.lstsq(
                normalized[:, pivot_positions],
                normalized[:, nonpivot_positions],
                cond=None,
                check_finite=False,
                lapack_driver="gelsy",
            )
        )
        # RRQR selected pivots after normalizing every degree-head column.  The
        # dependency solve must use that same normalization; otherwise a valid
        # pivot whose physical coefficient is merely small can be reclassified
        # as rank deficient by ``gelsy``.  Convert the normalized coefficients
        # back to the original residual coordinates before cancellation:
        #
        #   A_N = A_P diag(1 / n_P) X_norm diag(n_N).
        solution = (
            normalized_solution
            * norms[nonpivot_positions][np.newaxis, :]
            / norms[pivot_positions][:, np.newaxis]
        )
        if int(numerical_rank) != len(pivot_local) or np.any(~np.isfinite(solution)):
            raise GradedCompilationUnavailable(
                "graded owner pivots are not numerically full rank",
                certificate={
                    "component": [int(value) for value in indices],
                    "pivot_count": int(len(pivot_local)),
                    "reported_rank": int(numerical_rank),
                },
            )
        top_residual = nonpivot_dense - pivot_dense @ solution
        residual_norms = np.linalg.norm(top_residual, axis=0)
        operation_scale = (
            np.linalg.norm(nonpivot_dense, axis=0)
            + norms[pivot_positions] @ np.abs(solution)
        )
        solve_roundoff = (
            np.finfo(float).eps
            * float(64 * max(1, max(dense.shape)))
            * np.maximum(operation_scale, np.finfo(float).tiny)
        )
        pivot_errors = error_bounds[
            indices[np.asarray(pivot_local, dtype=np.int64)]
        ]
        nonpivot_errors = error_bounds[
            indices[np.asarray(nonpivot_local, dtype=np.int64)]
        ]
        new_errors = (
            nonpivot_errors
            + np.abs(solution).T @ pivot_errors
            + solve_roundoff
        )
        failed = np.flatnonzero(residual_norms > new_errors)
        if failed.size == 0:
            break
        # Deterministic fail-closed enlargement.  Keeping an owner is always
        # safe; silently accepting the uncertified cancellation is not.
        added_local = nonpivot_local[int(failed[0])]
        pivot_local.append(int(added_local))

    pivot_indices = tuple(int(indices[index]) for index in pivot_local)
    nonpivot_indices = tuple(int(indices[index]) for index in nonpivot_local)
    proof = {
        "channel_indices": [int(value) for value in indices],
        "selected_owner_indices": [int(value) for value in pivot_indices],
        "active_atom_rows": int(active_rows.size),
        "rrqr_proposed_rank": int(proposed_rank),
        "certified_rank": int(len(pivot_indices)),
        "machine_rank_tolerance": float(machine_tolerance),
        "maximum_cancelled_residual": float(np.max(residual_norms, initial=0.0)),
        "maximum_cancelled_error_bound": float(np.max(new_errors, initial=0.0)),
        "maximum_solve_roundoff_bound": float(
            np.max(solve_roundoff, initial=0.0)
        ),
    }
    return pivot_indices, nonpivot_indices, solution, new_errors, proof


def select_filtered_independent_columns(
    projected: sparse.spmatrix,
    *,
    row_degrees: Sequence[int] | np.ndarray,
    nominal_degrees: Sequence[int] | np.ndarray,
    absolute_error_bounds: Sequence[float] | np.ndarray,
    structural_blocks: Sequence[Sequence[int]] | None = None,
) -> GradedColumnSelection:
    """Select original projected columns by descending-degree residual rank.

    A column whose degree-``d`` head is dependent is *not* discarded.  Its
    cancelled residual is carried to degree ``d-1`` where it may become a new
    pivot.  Thus the algorithm does not require downward closure of the seed
    labels or of their homogeneous heads.
    """

    matrix = sparse.csc_matrix(projected, dtype=np.float64)
    rows = np.asarray(row_degrees, dtype=np.int64)
    nominal = np.asarray(nominal_degrees, dtype=np.int64)
    errors = np.asarray(absolute_error_bounds, dtype=np.float64)
    if rows.shape != (matrix.shape[0],):
        raise ValueError("row degrees must match projected matrix rows")
    if nominal.shape != (matrix.shape[1],):
        raise ValueError("nominal degrees must match projected matrix columns")
    if errors.shape != (matrix.shape[1],):
        raise ValueError("absolute error bounds must match projected matrix columns")
    if np.any(rows < 0) or np.any(nominal < 0):
        raise ValueError("graded degrees must be nonnegative")
    if np.any(~np.isfinite(errors)) or np.any(errors < 0.0):
        raise ValueError("graded error bounds must be finite and nonnegative")
    if structural_blocks is not None:
        flattened = [int(value) for block in structural_blocks for value in block]
        if sorted(flattened) != list(range(matrix.shape[1])):
            raise ValueError(
                "structural blocks must be a disjoint cover of projected columns"
            )

    norms = np.sqrt(np.asarray(matrix.power(2).sum(axis=0)).ravel())
    active_global = np.flatnonzero(norms > 0.0).astype(np.int64)
    relative_errors = errors[active_global] / np.maximum(
        norms[active_global], np.finfo(float).tiny
    )
    residual = matrix[:, active_global].tocsc() @ sparse.diags(
        1.0 / norms[active_global], format="csc"
    )
    active_owners = active_global.copy()
    active_nominal = nominal[active_global].copy()
    selected: list[int] = []
    proofs: list[Mapping[str, Any]] = []
    reconstruction_errors = np.zeros(matrix.shape[1], dtype=np.float64)

    if active_owners.size:
        leakage_squared = np.zeros(active_owners.size, dtype=np.float64)
        coo = residual.tocoo()
        leaked = rows[coo.row] > active_nominal[coo.col]
        if np.any(leaked):
            np.add.at(
                leakage_squared,
                coo.col[leaked],
                np.asarray(coo.data[leaked], dtype=np.float64) ** 2,
            )
        leakage = np.sqrt(leakage_squared)
        leakage_roundoff = (
            np.finfo(float).eps * float(64 * max(1, matrix.shape[0]))
        )
        failed = np.flatnonzero(leakage > relative_errors + leakage_roundoff)
        if failed.size:
            local = int(failed[0])
            raise GradedCompilationUnavailable(
                "a projected seed contains coefficients above its nominal degree",
                certificate={
                    "owner_index": int(active_owners[local]),
                    "nominal_degree": int(active_nominal[local]),
                    "leakage_norm": float(leakage[local]),
                    "error_bound": float(relative_errors[local]),
                },
            )

    maximum_degree = int(max(rows.max(initial=0), nominal.max(initial=0)))
    for degree in range(maximum_degree, -1, -1):
        if active_owners.size == 0:
            proofs.append(
                {
                    "stage": "filtered_degree_elimination",
                    "degree": int(degree),
                    "active_input": 0,
                    "active_output": 0,
                    "selected_owner_indices": [],
                    "components": [],
                }
            )
            continue
        degree_rows = np.flatnonzero(rows == degree).astype(np.int64)
        block = residual[degree_rows, :].tocsc()
        block_norms = np.sqrt(np.asarray(block.power(2).sum(axis=0)).ravel())
        roundoff_floor = np.finfo(float).eps * float(
            64 * max(1, block.shape[0])
        )
        nonzero_columns = np.flatnonzero(
            block_norms > relative_errors + roundoff_floor
        ).astype(np.int64)
        components = _connected_column_components(block, nonzero_columns)

        pivot_local_all: list[int] = []
        eliminations: list[
            tuple[tuple[int, ...], tuple[int, ...], np.ndarray, np.ndarray]
        ] = []
        component_proofs: list[Mapping[str, Any]] = []
        for component in components:
            pivots, nonpivots, solution, new_errors, component_proof = (
                _certified_component_elimination(
                    block,
                    component,
                    error_bounds=relative_errors,
                )
            )
            pivot_local_all.extend(pivots)
            eliminations.append((pivots, nonpivots, solution, new_errors))
            component_proofs.append(component_proof)

        pivot_set = set(pivot_local_all)
        keep_local = [
            index for index in range(active_owners.size) if index not in pivot_set
        ]
        new_by_old = {old: new for new, old in enumerate(keep_local)}
        transform_rows = list(keep_local)
        transform_columns = [new_by_old[value] for value in keep_local]
        transform_values = [1.0] * len(keep_local)
        next_errors = relative_errors[np.asarray(keep_local, dtype=np.int64)].copy()
        for pivots, nonpivots, solution, new_errors in eliminations:
            for nonpivot_position, nonpivot in enumerate(nonpivots):
                next_errors[new_by_old[nonpivot]] = float(
                    new_errors[nonpivot_position]
                )
                for pivot_position, pivot in enumerate(pivots):
                    value = -float(solution[pivot_position, nonpivot_position])
                    if value != 0.0:
                        transform_rows.append(int(pivot))
                        transform_columns.append(int(new_by_old[nonpivot]))
                        transform_values.append(value)
        transform = sparse.coo_matrix(
            (
                np.asarray(transform_values, dtype=np.float64),
                (
                    np.asarray(transform_rows, dtype=np.int64),
                    np.asarray(transform_columns, dtype=np.int64),
                ),
            ),
            shape=(active_owners.size, len(keep_local)),
        ).tocsc()
        next_residual = (residual @ transform).tocsc()
        next_residual.sum_duplicates()
        next_residual.eliminate_zeros()

        # The just-certified degree head is now represented by selected owner
        # columns.  Remove it explicitly so only the strict lower filtration is
        # presented to the next stage, charging every removed bit to the bound.
        next_coo = next_residual.tocoo()
        removed = rows[next_coo.row] == degree
        removed_squared = np.zeros(len(keep_local), dtype=np.float64)
        if np.any(removed):
            np.add.at(
                removed_squared,
                next_coo.col[removed],
                np.asarray(next_coo.data[removed], dtype=np.float64) ** 2,
            )
        next_errors += np.sqrt(removed_squared)
        retained = ~removed
        residual = sparse.coo_matrix(
            (
                np.asarray(next_coo.data[retained], dtype=np.float64),
                (
                    np.asarray(next_coo.row[retained], dtype=np.int64),
                    np.asarray(next_coo.col[retained], dtype=np.int64),
                ),
            ),
            shape=next_residual.shape,
        ).tocsc()
        residual.sum_duplicates()
        residual.eliminate_zeros()
        residual.sort_indices()

        stage_selected = [int(active_owners[index]) for index in pivot_local_all]
        selected.extend(stage_selected)
        active_owners = active_owners[np.asarray(keep_local, dtype=np.int64)]
        active_nominal = active_nominal[np.asarray(keep_local, dtype=np.int64)]
        relative_errors = next_errors
        proofs.append(
            {
                "stage": "filtered_degree_elimination",
                "degree": int(degree),
                "active_input": int(len(keep_local) + len(pivot_local_all)),
                "active_output": int(len(keep_local)),
                "selected_owner_indices": stage_selected,
                "certified_zero_head_count": int(
                    block.shape[1] - len(nonzero_columns)
                ),
                "components": list(component_proofs),
            }
        )

    if residual.nnz:
        raise GradedCompilationUnavailable(
            "graded residual remained after the degree-zero stage",
            certificate={
                "remaining_owner_indices": [int(value) for value in active_owners],
                "remaining_nnz": int(residual.nnz),
            },
        )
    reconstruction_errors[active_owners] = relative_errors * norms[active_owners]
    selected_owners = tuple(sorted(set(int(value) for value in selected)))
    selector = sparse.csc_matrix(
        (
            np.ones(len(selected_owners), dtype=np.float64),
            (
                np.asarray(selected_owners, dtype=np.int64),
                np.arange(len(selected_owners), dtype=np.int64),
            ),
        ),
        shape=(matrix.shape[1], len(selected_owners)),
    )
    return GradedColumnSelection(
        selected_owner_indices=selected_owners,
        logical_to_selected_coordinates=selector,
        residual_error_bounds=tuple(
            float(value) for value in reconstruction_errors
        ),
        degree_proofs=tuple(proofs),
    )


def _select_filtered_independent_column_blocks(
    projected_blocks: Sequence[sparse.spmatrix],
    *,
    owner_index_blocks: Sequence[Sequence[int]],
    nominal_degree_blocks: Sequence[Sequence[int] | np.ndarray],
    error_bound_blocks: Sequence[Sequence[float] | np.ndarray],
    row_degrees: Sequence[int] | np.ndarray,
) -> tuple[GradedColumnSelection, tuple[int, ...], Mapping[str, Any]]:
    """Apply global filtered owner semantics through one degree slice at a time.

    Projection remains partitioned into structural blocks.  The selector never
    forms their full authored-column hstack: at degree ``d`` it assembles only
    the rows of that homogeneous slice, applies the accumulated cancellation
    coordinates, and then releases the slice.  Sorting by the authored owner
    index before every slice makes the result identical to one global filtered
    selection, including deterministic RRQR owner choices.
    """

    blocks = tuple(
        sparse.csc_matrix(block, dtype=np.float64) for block in projected_blocks
    )
    owner_blocks = tuple(
        np.asarray(tuple(values), dtype=np.int64) for values in owner_index_blocks
    )
    nominal_blocks = tuple(
        np.asarray(values, dtype=np.int64) for values in nominal_degree_blocks
    )
    error_blocks = tuple(
        np.asarray(values, dtype=np.float64) for values in error_bound_blocks
    )
    block_count = len(blocks)
    if not (
        len(owner_blocks)
        == len(nominal_blocks)
        == len(error_blocks)
        == block_count
    ):
        raise ValueError("projected block metadata must have matching lengths")
    rows = np.asarray(row_degrees, dtype=np.int64)
    if rows.ndim != 1 or np.any(rows < 0):
        raise ValueError("row degrees must be a nonnegative vector")
    for block, owners, nominal, errors in zip(
        blocks, owner_blocks, nominal_blocks, error_blocks
    ):
        if block.shape[0] != len(rows):
            raise ValueError("projected blocks must share the declared rows")
        expected = (block.shape[1],)
        if owners.shape != expected or nominal.shape != expected:
            raise ValueError("projected block owner/degree metadata is misaligned")
        if errors.shape != expected:
            raise ValueError("projected block error metadata is misaligned")
        if np.any(nominal < 0):
            raise ValueError("graded degrees must be nonnegative")
        if np.any(~np.isfinite(errors)) or np.any(errors < 0.0):
            raise ValueError("graded error bounds must be finite and nonnegative")

    if blocks:
        concatenated_owners = np.concatenate(owner_blocks)
        concatenated_nominal = np.concatenate(nominal_blocks)
        concatenated_errors = np.concatenate(error_blocks)
        concatenated_norms = np.concatenate(
            [
                np.sqrt(np.asarray(block.power(2).sum(axis=0)).ravel())
                for block in blocks
            ]
        )
    else:
        concatenated_owners = np.empty(0, dtype=np.int64)
        concatenated_nominal = np.empty(0, dtype=np.int64)
        concatenated_errors = np.empty(0, dtype=np.float64)
        concatenated_norms = np.empty(0, dtype=np.float64)
    if len(np.unique(concatenated_owners)) != len(concatenated_owners):
        raise ValueError("global filtered owner indices must be unique")
    authored_order = np.argsort(concatenated_owners, kind="stable")
    concatenated_to_source = np.empty(len(authored_order), dtype=np.int64)
    concatenated_to_source[authored_order] = np.arange(
        len(authored_order), dtype=np.int64
    )
    source_owner_indices = concatenated_owners[authored_order]
    nominal = concatenated_nominal[authored_order]
    errors = concatenated_errors[authored_order]
    norms = concatenated_norms[authored_order]
    source_count = len(source_owner_indices)

    leakage_squared = np.zeros(source_count, dtype=np.float64)
    concatenated_offset = 0
    for block in blocks:
        coo = block.tocoo()
        source_columns = concatenated_to_source[
            concatenated_offset + np.asarray(coo.col, dtype=np.int64)
        ]
        leaked = rows[np.asarray(coo.row, dtype=np.int64)] > nominal[source_columns]
        if np.any(leaked):
            np.add.at(
                leakage_squared,
                source_columns[leaked],
                np.asarray(coo.data[leaked], dtype=np.float64) ** 2,
            )
        concatenated_offset += block.shape[1]

    eligible_sources = np.flatnonzero(norms > 0.0).astype(np.int64)
    eligible_relative_errors = errors[eligible_sources] / np.maximum(
        norms[eligible_sources], np.finfo(float).tiny
    )
    leakage = np.sqrt(leakage_squared[eligible_sources]) / np.maximum(
        norms[eligible_sources], np.finfo(float).tiny
    )
    leakage_roundoff = np.finfo(float).eps * float(64 * max(1, len(rows)))
    failed = np.flatnonzero(
        leakage > eligible_relative_errors + leakage_roundoff
    )
    if failed.size:
        local = int(failed[0])
        source = int(eligible_sources[local])
        raise GradedCompilationUnavailable(
            "a projected seed contains coefficients above its nominal degree",
            certificate={
                "owner_index": int(source_owner_indices[source]),
                "nominal_degree": int(nominal[source]),
                "leakage_norm": float(leakage[local]),
                "error_bound": float(eligible_relative_errors[local]),
            },
        )

    normalization = sparse.csc_matrix(
        (
            1.0 / norms[eligible_sources],
            (
                eligible_sources,
                np.arange(len(eligible_sources), dtype=np.int64),
            ),
        ),
        shape=(source_count, len(eligible_sources)),
    )
    active_owners = np.empty(0, dtype=np.int64)
    active_nominal = np.empty(0, dtype=np.int64)
    relative_errors = np.empty(0, dtype=np.float64)
    selected: list[int] = []
    proofs: list[Mapping[str, Any]] = []
    reconstruction_errors = np.zeros(source_count, dtype=np.float64)
    slice_artifacts: list[dict[str, Any]] = []
    slice_assembly_seconds = 0.0
    component_discovery_seconds = 0.0
    component_elimination_seconds = 0.0
    stage_transform_build_seconds = 0.0
    current_degree_transform_seconds = 0.0
    lower_degree_transform_seconds = 0.0
    lower_degree_transform_calls = 0
    selection_started = time.perf_counter()
    maximum_degree = int(max(rows.max(initial=0), nominal.max(initial=0)))
    authored_residual_slices: dict[int, sparse.csc_matrix] = {}
    for degree in range(maximum_degree, -1, -1):
        degree_rows = np.flatnonzero(rows == degree).astype(np.int64)
        assembly_started = time.perf_counter()
        if blocks:
            authored_slice = sparse.hstack(
                [block[degree_rows, :] for block in blocks], format="csc"
            )[:, authored_order]
        else:
            authored_slice = sparse.csc_matrix(
                (len(degree_rows), 0), dtype=np.float64
            )
        residual_slice = (authored_slice @ normalization).tocsc()
        residual_slice.sum_duplicates()
        residual_slice.eliminate_zeros()
        authored_residual_slices[int(degree)] = residual_slice
        assembly_seconds = time.perf_counter() - assembly_started
        slice_assembly_seconds += assembly_seconds
        slice_artifacts.append(
            {
                "degree": int(degree),
                "authored_row_count": int(len(degree_rows)),
                "authored_column_count": int(source_count),
                "authored_nonzero_count": int(authored_slice.nnz),
                "active_column_count": int(residual_slice.shape[1]),
                "active_nonzero_count": int(residual_slice.nnz),
                "assembly_seconds": float(assembly_seconds),
            }
        )
    deferred_relative_errors = eligible_relative_errors.copy()
    eligible_nominal = nominal[eligible_sources]
    for degree, authored_residual_slice in authored_residual_slices.items():
        pending = np.flatnonzero(eligible_nominal < degree).astype(np.int64)
        if pending.size:
            deferred_relative_errors[pending] += np.sqrt(
                np.asarray(
                    authored_residual_slice[:, pending].power(2).sum(axis=0)
                ).ravel()
            )
    residual_slices = {
        degree: sparse.csc_matrix(
            (authored_residual_slices[degree].shape[0], 0),
            dtype=np.float64,
        )
        for degree in range(maximum_degree, -1, -1)
    }
    for degree in range(maximum_degree, -1, -1):
        new_positions = np.flatnonzero(
            nominal[eligible_sources] == degree
        ).astype(np.int64)
        if new_positions.size:
            new_owners = eligible_sources[new_positions]
            combined_owners = np.concatenate((active_owners, new_owners))
            merged_order = np.argsort(combined_owners, kind="stable")
            for lower_degree, carried_slice in tuple(residual_slices.items()):
                new_slice = authored_residual_slices[lower_degree][
                    :, new_positions
                ].tocsc()
                merged_slice = sparse.hstack(
                    (carried_slice, new_slice), format="csc"
                )[:, merged_order]
                merged_slice.sum_duplicates()
                merged_slice.eliminate_zeros()
                residual_slices[lower_degree] = merged_slice
            active_owners = combined_owners[merged_order]
            active_nominal = np.concatenate(
                (active_nominal, nominal[new_owners])
            )[merged_order]
            relative_errors = np.concatenate(
                (relative_errors, deferred_relative_errors[new_positions])
            )[merged_order]
        if active_owners.size == 0:
            proofs.append(
                {
                    "stage": "filtered_degree_elimination",
                    "degree": int(degree),
                    "active_input": 0,
                    "active_output": 0,
                    "selected_owner_indices": [],
                    "components": [],
                }
            )
            continue
        block = residual_slices.pop(int(degree))
        block_norms = np.sqrt(np.asarray(block.power(2).sum(axis=0)).ravel())
        roundoff_floor = np.finfo(float).eps * float(
            64 * max(1, block.shape[0])
        )
        nonzero_columns = np.flatnonzero(
            block_norms > relative_errors + roundoff_floor
        ).astype(np.int64)
        components_started = time.perf_counter()
        components = _connected_column_components(block, nonzero_columns)
        component_discovery_seconds += time.perf_counter() - components_started

        pivot_local_all: list[int] = []
        eliminations: list[
            tuple[tuple[int, ...], tuple[int, ...], np.ndarray, np.ndarray]
        ] = []
        component_proofs: list[Mapping[str, Any]] = []
        elimination_started = time.perf_counter()
        for component in components:
            pivots, nonpivots, solution, new_errors, component_proof = (
                _certified_component_elimination(
                    block,
                    component,
                    error_bounds=relative_errors,
                )
            )
            pivot_local_all.extend(pivots)
            eliminations.append((pivots, nonpivots, solution, new_errors))
            component_proofs.append(component_proof)
        component_elimination_seconds += time.perf_counter() - elimination_started

        transform_build_started = time.perf_counter()
        pivot_set = set(pivot_local_all)
        keep_local = [
            index for index in range(active_owners.size) if index not in pivot_set
        ]
        new_by_old = {old: new for new, old in enumerate(keep_local)}
        transform_rows = list(keep_local)
        transform_columns = [new_by_old[value] for value in keep_local]
        transform_values = [1.0] * len(keep_local)
        next_errors = relative_errors[np.asarray(keep_local, dtype=np.int64)].copy()
        for pivots, nonpivots, solution, new_errors in eliminations:
            for nonpivot_position, nonpivot in enumerate(nonpivots):
                next_errors[new_by_old[nonpivot]] = float(
                    new_errors[nonpivot_position]
                )
                for pivot_position, pivot in enumerate(pivots):
                    value = -float(solution[pivot_position, nonpivot_position])
                    if value != 0.0:
                        transform_rows.append(int(pivot))
                        transform_columns.append(int(new_by_old[nonpivot]))
                        transform_values.append(value)
        stage_transform = sparse.coo_matrix(
            (
                np.asarray(transform_values, dtype=np.float64),
                (
                    np.asarray(transform_rows, dtype=np.int64),
                    np.asarray(transform_columns, dtype=np.int64),
                ),
            ),
            shape=(active_owners.size, len(keep_local)),
        ).tocsc()
        stage_transform_build_seconds += (
            time.perf_counter() - transform_build_started
        )
        current_transform_started = time.perf_counter()
        next_degree_block = (block @ stage_transform).tocsc()
        next_degree_block.sum_duplicates()
        next_degree_block.eliminate_zeros()
        current_degree_transform_seconds += (
            time.perf_counter() - current_transform_started
        )
        next_errors += np.sqrt(
            np.asarray(next_degree_block.power(2).sum(axis=0)).ravel()
        )
        for lower_degree, lower_slice in tuple(residual_slices.items()):
            lower_transform_started = time.perf_counter()
            next_slice = (lower_slice @ stage_transform).tocsc()
            next_slice.sum_duplicates()
            next_slice.eliminate_zeros()
            residual_slices[lower_degree] = next_slice
            lower_degree_transform_seconds += (
                time.perf_counter() - lower_transform_started
            )
            lower_degree_transform_calls += 1

        stage_selected = [int(active_owners[index]) for index in pivot_local_all]
        selected.extend(stage_selected)
        active_owners = active_owners[np.asarray(keep_local, dtype=np.int64)]
        active_nominal = active_nominal[np.asarray(keep_local, dtype=np.int64)]
        relative_errors = next_errors
        proofs.append(
            {
                "stage": "filtered_degree_elimination",
                "degree": int(degree),
                "active_input": int(len(keep_local) + len(pivot_local_all)),
                "active_output": int(len(keep_local)),
                "selected_owner_indices": stage_selected,
                "certified_zero_head_count": int(
                    block.shape[1] - len(nonzero_columns)
                ),
                "components": list(component_proofs),
            }
        )

    reconstruction_errors[active_owners] = (
        relative_errors * norms[active_owners]
    )
    selected_sources = tuple(sorted(set(int(value) for value in selected)))
    selector = sparse.csc_matrix(
        (
            np.ones(len(selected_sources), dtype=np.float64),
            (
                np.asarray(selected_sources, dtype=np.int64),
                np.arange(len(selected_sources), dtype=np.int64),
            ),
        ),
        shape=(source_count, len(selected_sources)),
    )
    selection = GradedColumnSelection(
        selected_owner_indices=selected_sources,
        logical_to_selected_coordinates=selector,
        residual_error_bounds=tuple(
            float(value) for value in reconstruction_errors
        ),
        degree_proofs=tuple(proofs),
    )
    artifact = {
        "strategy": "deferred_owner_degree_slices_v2",
        "single_pass_filtered_selection": True,
        "global_authored_projected_matrix_materialized": False,
        "structural_projection_block_count": int(block_count),
        "input_real_column_count": int(source_count),
        "input_nonzero_count": int(sum(block.nnz for block in blocks)),
        "retained_real_column_count": int(len(selected_sources)),
        "selected_source_indices": [int(value) for value in selected_sources],
        "selected_global_owner_indices": [
            int(source_owner_indices[index]) for index in selected_sources
        ],
        "degree_slices": slice_artifacts,
        "maximum_degree_slice_row_count": int(
            max((item["authored_row_count"] for item in slice_artifacts), default=0)
        ),
        "maximum_degree_slice_column_count": int(
            max(
                (item["authored_column_count"] for item in slice_artifacts),
                default=0,
            )
        ),
        "maximum_degree_slice_nonzero_count": int(
            max(
                (item["authored_nonzero_count"] for item in slice_artifacts),
                default=0,
            )
        ),
        "initial_residual_slice_total_nonzero_count": int(
            sum(item["active_nonzero_count"] for item in slice_artifacts)
        ),
        "slice_assembly_seconds": float(slice_assembly_seconds),
        "component_discovery_seconds": float(component_discovery_seconds),
        "component_elimination_seconds": float(component_elimination_seconds),
        "stage_transform_build_seconds": float(stage_transform_build_seconds),
        "current_degree_transform_seconds": float(
            current_degree_transform_seconds
        ),
        "lower_degree_transform_seconds": float(lower_degree_transform_seconds),
        "lower_degree_transform_call_count": int(lower_degree_transform_calls),
        "selection_seconds": float(time.perf_counter() - selection_started),
        "rank_proofs": list(proofs),
    }
    return selection, tuple(int(value) for value in source_owner_indices), artifact


def _joint_adjoint_key_artifact(key: JointAdjointSeedKey) -> dict[str, Any]:
    return {
        "sector_from": str(key.sector_from),
        "sector_to": str(key.sector_to),
        "orbital_from": int(key.orbital_from),
        "orbital_to": int(key.orbital_to),
        "monomial": [int(value) for value in key.monomial],
        "center": {
            "serialized_little_endian_hex": str(
                key.center.serialized_little_endian_hex
            ),
            "coordinate_convention": str(key.center.coordinate_convention),
        },
        "harmonic_id": str(key.harmonic_id),
        "adjoint_harmonic_id": str(key.adjoint_harmonic_id),
    }


def _certified_hermitian_owner_action(
    certified_raw_adjoint: Any,
    *,
    physical_seeds: Sequence[RawPolynomialSeed],
) -> sparse.csc_matrix:
    """Compile raw adjunction in the physical real/imag owner vocabulary."""

    seeds = tuple(physical_seeds)
    index_by_seed_id = {
        str(seed.seed_id): index for index, seed in enumerate(seeds)
    }
    if len(index_by_seed_id) != len(seeds):
        raise ValueError("physical adjoint seed ids must be unique")
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    covered_seed_ids: set[str] = set()
    for orbit in certified_raw_adjoint.orbits:
        if bool(orbit.requires_numeric_certification):
            raise LocalResponseCompilationError(
                "uncertified_adjoint_route",
                certificate={
                    "representative_seed_id": str(orbit.representative_seed_id),
                    "certification_status": str(orbit.certification_status),
                },
            )
        member_ids = tuple(str(value) for value in orbit.member_seed_ids)
        if any(seed_id not in index_by_seed_id for seed_id in member_ids):
            raise LocalResponseCompilationError(
                "adjoint_route_missing_physical_seed",
                certificate={"member_seed_ids": member_ids},
            )
        covered_seed_ids.update(member_ids)
        if bool(orbit.self_adjoint):
            if len(member_ids) != 1:
                raise RuntimeError("self-adjoint orbit must contain exactly one seed")
            source = index_by_seed_id[member_ids[0]]
            rows.extend((2 * source, 2 * source + 1))
            columns.extend((2 * source, 2 * source + 1))
            values.extend((1.0, -1.0))
            continue
        if len(member_ids) != 2:
            raise RuntimeError("non-self-adjoint orbit must contain exactly two seeds")
        left = index_by_seed_id[member_ids[0]]
        right = index_by_seed_id[member_ids[1]]
        rows.extend((2 * right, 2 * left, 2 * right + 1, 2 * left + 1))
        columns.extend((2 * left, 2 * right, 2 * left + 1, 2 * right + 1))
        values.extend((1.0, 1.0, -1.0, -1.0))
    if covered_seed_ids != set(index_by_seed_id):
        raise LocalResponseCompilationError(
            "adjoint_routes_incomplete",
            certificate={
                "missing_seed_ids": tuple(sorted(set(index_by_seed_id) - covered_seed_ids)),
            },
        )
    dimension = 2 * len(seeds)
    action = sparse.coo_matrix(
        (
            np.asarray(values, dtype=np.float64),
            (
                np.asarray(rows, dtype=np.int64),
                np.asarray(columns, dtype=np.int64),
            ),
        ),
        shape=(dimension, dimension),
    ).tocsc()
    action.sum_duplicates()
    action.eliminate_zeros()
    action.sort_indices()
    column_nnz = np.diff(action.indptr)
    if not np.array_equal(column_nnz, np.ones(dimension, dtype=column_nnz.dtype)):
        raise RuntimeError("certified adjoint action is not a signed permutation")
    involution = (action @ action - sparse.eye(dimension, format="csc")).tocsc()
    involution.eliminate_zeros()
    if involution.nnz:
        raise RuntimeError("certified adjoint action does not square to identity")
    return action


def _fixed_column_error_components(
    coordinates: np.ndarray,
    *,
    component_owner_indices: Sequence[int],
    error_components_by_seed: Sequence[Mapping[str, float]],
    absolute_errors_by_seed: Sequence[float],
) -> tuple[dict[str, float], ...]:
    owner_indices = tuple(int(value) for value in component_owner_indices)
    matrix = np.asarray(coordinates, dtype=np.float64)
    records: list[dict[str, float]] = []
    for fixed_index in range(matrix.shape[1]):
        record: dict[str, float] = {}
        for local_index, owner in enumerate(owner_indices):
            weight = abs(float(matrix[local_index, fixed_index]))
            if weight == 0.0:
                continue
            seed_index = owner // 2
            source_components = error_components_by_seed[seed_index]
            source_total = float(sum(source_components.values()))
            source_envelope = float(absolute_errors_by_seed[seed_index])
            for name, value in source_components.items():
                record[str(name)] = record.get(str(name), 0.0) + weight * float(value)
            if source_envelope > source_total:
                record["duplicate_source_envelope"] = (
                    record.get("duplicate_source_envelope", 0.0)
                    + weight * (source_envelope - source_total)
                )
        records.append(dict(sorted(record.items())))
    return tuple(records)


def _compile_graded_candidate_group_local_fixed(
    seeds: Sequence[RawPolynomialSeed],
    *,
    coordinate: PolynomialCoordinateBasis,
    group: Any,
    factorized_actions: Mapping[str, Any],
    internal_actions_by_word: Mapping[tuple[str, ...], sparse.spmatrix] | None,
    factorized_group_artifact: Mapping[str, Any] | None,
    joint_adjoint_keys: Mapping[str, JointAdjointSeedKey] | None,
) -> CandidateResponseSet:
    """Compile the graded physical space from local generator/H fixed spaces."""

    function_started = time.perf_counter()
    ordered = _validate_seeds(seeds)
    descriptors = tuple(
        describe_filtered_seed(seed, coordinate=coordinate, seed_index=index)
        for index, seed in enumerate(ordered)
    )
    action_started = time.perf_counter()
    if internal_actions_by_word is None:
        internal_actions, action_artifact = compile_factorized_group_element_actions(
            group=group,
            factorized_generators=factorized_actions,
        )
    else:
        if factorized_group_artifact is None:
            raise LocalResponseCompilationError(
                "missing_factorized_group_artifact",
                certificate={"check": "factorized_group_artifact"},
            )
        internal_actions = {
            tuple(str(value) for value in word): sparse.csr_matrix(matrix)
            for word, matrix in internal_actions_by_word.items()
        }
        action_artifact = dict(factorized_group_artifact)
    action_seconds = time.perf_counter() - action_started
    internal_error = float(action_artifact.get("maximum_action_error_bound", 0.0))

    error_started = time.perf_counter()
    error_components_by_seed: list[dict[str, float]] = []
    absolute_errors_by_seed: list[float] = []
    raw_adjoint_errors_by_seed: list[float] = []
    for seed in ordered:
        raw_norm = _coefficient_norm(seed.coefficients)
        components = _propagated_error_components(
            raw_norm,
            dim=seed.dim,
            degree=coordinate.max_degree,
            group=group,
        )
        raw_adjoint_errors_by_seed.append(
            float(
                sum(components.values())
                + float(seed.metadata.get("coordinate_conversion_error_bound", 0.0))
            )
        )
        if internal_error > 0.0:
            components["factorized_internal_action"] = float(
                (2.0 * internal_error + internal_error**2)
                * max(raw_norm, np.finfo(float).tiny)
            )
        error_components_by_seed.append(components)
        absolute_errors_by_seed.append(float(sum(components.values())))
    error_seconds = time.perf_counter() - error_started

    structural_started = time.perf_counter()
    structural_plan = build_structural_compilation_plan(
        ordered,
        coordinate=coordinate,
        group=group,
        internal_actions_by_word=internal_actions,
        maximum_action_error_bound=internal_error,
        descriptors=descriptors,
        compute_cyclic_orbits=False,
    )
    structural_seconds = time.perf_counter() - structural_started
    if not structural_plan.representative_seed_indices:
        raise LocalResponseCompilationError(
            "empty_physical_seed_vocabulary",
            certificate={"zero_seed_indices": structural_plan.zero_seed_indices},
        )
    if joint_adjoint_keys is None:
        raise LocalResponseCompilationError(
            "missing_certified_adjoint_routes",
            certificate={"physical_seed_count": len(structural_plan.representative_seed_indices)},
        )

    adjoint_started = time.perf_counter()
    ordered_seed_ids = {str(seed.seed_id) for seed in ordered}
    if {str(seed_id) for seed_id in joint_adjoint_keys} != ordered_seed_ids:
        raise ValueError("graded joint-adjoint keys must cover every authored seed exactly")
    physical_seed_indices = tuple(
        int(indices[0]) for indices in structural_plan.duplicate_seed_groups
    )
    physical_seeds = tuple(ordered[index] for index in physical_seed_indices)
    provisional_raw_adjoint = canonicalize_joint_adjoint_seeds(
        [
            JointAdjointSeed(
                str(seed.seed_id),
                joint_adjoint_keys[str(seed.seed_id)],
            )
            for seed in physical_seeds
        ]
    )
    certified_raw_adjoint = certify_joint_adjoint_coefficients(
        provisional_raw_adjoint,
        {str(seed.seed_id): seed.coefficients for seed in physical_seeds},
        absolute_error_bounds_by_seed={
            str(ordered[index].seed_id): float(raw_adjoint_errors_by_seed[index])
            for index in physical_seed_indices
        },
    )
    hermitian_action = _certified_hermitian_owner_action(
        certified_raw_adjoint,
        physical_seeds=physical_seeds,
    )
    adjoint_seconds = time.perf_counter() - adjoint_started

    vocabulary_started = time.perf_counter()
    vocabulary = build_raw_real_vocabulary(physical_seeds, coordinate=coordinate)
    owner_indices = tuple(
        owner
        for seed_index in physical_seed_indices
        for owner in (2 * seed_index, 2 * seed_index + 1)
    )
    nominal_degrees = tuple(
        degree
        for seed_index in physical_seed_indices
        for degree in (
            descriptors[seed_index].nominal_degree,
            descriptors[seed_index].nominal_degree,
        )
    )
    duplicates_by_representative = {
        int(indices[0]): tuple(int(value) for value in indices)
        for indices in structural_plan.duplicate_seed_groups
    }
    column_error_bounds = np.asarray(
        [
            max(absolute_errors_by_seed[index] for index in duplicates_by_representative[seed_index])
            for seed_index in physical_seed_indices
            for _component in ("real", "imag")
        ],
        dtype=np.float64,
    )
    provenance_by_owner = {
        owner: {
            "owner": owner,
            "seed_index": owner // 2,
            "component": ("real", "imag")[owner % 2],
            "seed_id": str(ordered[owner // 2].seed_id),
            "term_index": descriptors[owner // 2].term_index,
            "nominal_degree": descriptors[owner // 2].nominal_degree,
            "harmonic_adjoint": _joint_adjoint_key_artifact(
                joint_adjoint_keys[str(ordered[owner // 2].seed_id)]
            ),
        }
        for owner in owner_indices
    }
    vocabulary_seconds = time.perf_counter() - vocabulary_started

    raw_rank_started = time.perf_counter()
    row_degrees = projected_row_degrees(
        coordinate=coordinate,
        dim=ordered[0].dim,
    )
    raw_owner_selection, raw_owner_map, raw_owner_artifact = (
        _select_filtered_independent_column_blocks(
            (vocabulary.matrix,),
            owner_index_blocks=(owner_indices,),
            nominal_degree_blocks=(nominal_degrees,),
            error_bound_blocks=(column_error_bounds,),
            row_degrees=row_degrees,
        )
    )
    if len(raw_owner_selection.selected_owner_indices) != len(owner_indices):
        raise LocalResponseCompilationError(
            "raw_owner_vocabulary_not_injective",
            certificate={
                "raw_owner_count": len(owner_indices),
                "certified_rank": len(raw_owner_selection.selected_owner_indices),
                "selected_owner_indices": tuple(
                    int(raw_owner_map[index])
                    for index in raw_owner_selection.selected_owner_indices
                ),
            },
        )
    raw_rank_seconds = time.perf_counter() - raw_rank_started

    generator_started = time.perf_counter()
    generator_actions: dict[str, sparse.csc_matrix] = {}
    antiunitary_parities: dict[str, bool] = {}
    generator_error_bounds: dict[str, float] = {}
    if factorized_actions:
        for name in sorted(str(value) for value in factorized_actions):
            action = factorized_actions[name]
            try:
                generator_actions[name] = compile_factorized_raw_seed_action(
                    seeds=physical_seeds,
                    factorized_action=action,
                )
            except FactorizedTermActionError as error:
                raise LocalResponseCompilationError(
                    "missing_certified_generator_route",
                    certificate={"generator": name, "message": str(error)},
                ) from error
            antiunitary_parities[name] = bool(action.antiunitary)
            generator_error_bounds[name] = float(
                max(
                    internal_error,
                    getattr(action, "matrix_certification_bound", 0.0),
                    getattr(action, "phase_certification_bound", 0.0),
                )
            )
    elif len(tuple(group.elements)) == 1 and not bool(group.elements[0].antiunitary):
        generator_actions["identity"] = sparse.eye(
            len(owner_indices), format="csc", dtype=np.float64
        )
        antiunitary_parities["identity"] = False
        generator_error_bounds["identity"] = 0.0
    else:
        raise LocalResponseCompilationError(
            "missing_certified_generator_routes",
            certificate={"group_element_count": len(tuple(group.elements))},
        )
    generator_seconds = time.perf_counter() - generator_started

    component_started = time.perf_counter()
    action_blocks = build_local_action_blocks(
        nominal_degrees=nominal_degrees,
        logical_owner_indices=owner_indices,
        generator_actions=generator_actions,
        generator_error_bounds=generator_error_bounds,
        antiunitary_parities=antiunitary_parities,
        hermitian_action=hermitian_action,
    )
    action_block_diagnostic = summarize_local_action_blocks(action_blocks)
    component_seconds = time.perf_counter() - component_started
    group_words = tuple(
        tuple(str(value) for value in element.canonical_word)
        for element in group.elements
    )

    local_solve_started = time.perf_counter()
    fixed_block_records: list[dict[str, Any]] = []
    component_artifacts: list[dict[str, Any]] = []
    total_fixed_rank = 0
    total_fixed_nnz = 0
    owner_position = {
        int(owner): position for position, owner in enumerate(owner_indices)
    }
    for block in action_blocks:
        solve_started = time.perf_counter()
        result = solve_local_reynolds_block(
            block,
            group_words=group_words,
        )
        solve_seconds = time.perf_counter() - solve_started
        component_artifacts.append(
            {
                "component_index": int(block.block_index),
                "raw_dimension": len(block.owner_indices),
                "independent_dimension": int(
                    result.certification_metadata["independent_dimension"]
                ),
                "fixed_rank": int(result.fixed_rank),
                "generator_action_nnz": dict(
                    result.certification_metadata["generator_action_nnz"]
                ),
                "action_block_count": 1,
                "maximum_action_block_dimension": len(block.owner_indices),
                "solve_seconds": float(solve_seconds),
                "fallback_reason": None,
            }
        )
        if result.fixed_rank == 0:
            continue
        fixed_coordinates = sparse.csc_matrix(
            np.asarray(result.fixed_vocabulary_coordinates, dtype=np.float64)
        )
        positions = np.asarray(
            [owner_position[int(owner)] for owner in block.owner_indices],
            dtype=np.int64,
        )
        fixed_columns = (
            vocabulary.matrix[:, positions] @ fixed_coordinates
        ).tocsc()
        fixed_columns.sum_duplicates()
        fixed_columns.eliminate_zeros()
        fixed_columns.sort_indices()
        fixed_error_components = _fixed_column_error_components(
            np.asarray(result.fixed_vocabulary_coordinates, dtype=np.float64),
            component_owner_indices=block.owner_indices,
            error_components_by_seed=error_components_by_seed,
            absolute_errors_by_seed=absolute_errors_by_seed,
        )
        selected_owners = tuple(int(value) for value in result.selected_global_owner_indices)
        fixed_block_records.append(
            {
                "component_index": int(block.block_index),
                "projected": fixed_columns,
                "owner_indices": selected_owners,
                "nominal_degrees": np.asarray(
                    [descriptors[owner // 2].nominal_degree for owner in selected_owners],
                    dtype=np.int64,
                ),
                "error_bounds": np.asarray(
                    [sum(values.values()) for values in fixed_error_components],
                    dtype=np.float64,
                ),
                "error_components": fixed_error_components,
                "fixed_coordinates": np.asarray(
                    result.fixed_vocabulary_coordinates, dtype=np.float64
                ),
                "component_owner_indices": block.owner_indices,
                "result": result,
            }
        )
        total_fixed_rank += int(result.fixed_rank)
        total_fixed_nnz += int(fixed_columns.nnz)
    local_solve_seconds = time.perf_counter() - local_solve_started
    if not fixed_block_records:
        raise LocalResponseCompilationError(
            "empty_local_fixed_space",
            certificate={"component_count": len(action_blocks)},
        )

    location_by_owner: dict[int, tuple[dict[str, Any], int]] = {}
    for record in fixed_block_records:
        for local_index, owner in enumerate(record["owner_indices"]):
            if int(owner) in location_by_owner:
                raise RuntimeError("local fixed owner selected by more than one component")
            location_by_owner[int(owner)] = (record, int(local_index))
    selected_global_owners = tuple(sorted(location_by_owner))

    policy = NullClassificationPolicy()
    channel_ids = tuple(
        f"{seed.seed_id}:{component}"
        for seed in ordered
        for component in ("real", "imag")
    )
    materialize_started = time.perf_counter()
    channels: list[ResponseChannel] = []
    for owner in selected_global_owners:
        record, local_index = location_by_owner[owner]
        seed_index = owner // 2
        component_name = ("real", "imag")[owner % 2]
        seed = ordered[seed_index]
        column = record["projected"][:, local_index].tocsc()
        coefficients = _coefficient_vector_to_response_map(
            column,
            coordinate=coordinate,
            dim=seed.dim,
        )
        response_norm = _coefficient_norm(coefficients)
        components_error = dict(record["error_components"][local_index])
        error_bound = float(sum(components_error.values()))
        fixed_coordinate_column = np.asarray(
            record["fixed_coordinates"][:, local_index], dtype=np.float64
        )
        source_provenance: list[dict[str, Any]] = []
        source_coordinates: list[dict[str, Any]] = []
        for local_owner, coefficient in zip(
            record["component_owner_indices"], fixed_coordinate_column
        ):
            value = float(coefficient)
            if value == 0.0:
                continue
            source_seed_index = int(local_owner) // 2
            source_coordinates.append(
                {
                    "owner": int(local_owner),
                    "channel_id": channel_ids[int(local_owner)],
                    "coefficient": value,
                }
            )
            source_provenance.extend(
                _seed_provenance(ordered[index], descriptors[index])
                for index in duplicates_by_representative[source_seed_index]
            )
        channels.append(
            ResponseChannel(
                channel_id=channel_ids[owner],
                seed_id=str(seed.seed_id),
                component=component_name,
                coefficients=coefficients,
                unnormalized_norm=response_norm,
                propagated_error_bound=error_bound,
                classification=policy.classify(response_norm, error_bound),
                response_scale=response_norm,
                support_component=str(seed.support_component),
                metadata={
                    **dict(seed.metadata),
                    "symbolic_atom_compiler": LOCAL_REYNOLDS_ALGORITHM_V1,
                    "graded_nominal_degree": int(descriptors[seed_index].nominal_degree),
                    "graded_structural_support_id": str(
                        descriptors[seed_index].structural_support_id
                    ),
                    "graded_source_provenance": source_provenance,
                    "local_fixed_owner_coordinates": source_coordinates,
                    "local_fixed_component_index": int(record["component_index"]),
                },
                error_bound_components=components_error,
            )
        )
    materialize_seconds = time.perf_counter() - materialize_started

    raw_rank_proofs = tuple(
        {
            **dict(proof),
            "selection_scope": "raw_owner_injectivity",
            "local_to_global_owner_indices": [int(value) for value in raw_owner_map],
            "selected_global_owner_indices": [
                int(raw_owner_map[index])
                for index in proof.get("selected_owner_indices", [])
            ],
        }
        for proof in raw_owner_selection.degree_proofs
    )
    raw_owner_artifact = {
        **dict(raw_owner_artifact),
        "certificate": "full_column_rank",
        "raw_owner_count": len(owner_indices),
        "certified_rank": len(raw_owner_selection.selected_owner_indices),
        "rank_proofs": list(raw_rank_proofs),
        "post_projection_rank_reduction_performed": False,
    }
    logical_count = 2 * len(ordered)
    physical_real_count = 2 * len(physical_seed_indices)
    adjoint_orbit_descriptor_count = int(len(certified_raw_adjoint.orbits))
    hermitian_ambient_channel_count = int(
        sum(len(orbit.channel_ids) for orbit in certified_raw_adjoint.orbits)
    )
    structural_support_count = len(
        {
            descriptors[index].structural_support_id
            for index in structural_plan.representative_seed_indices
        }
    )
    return CandidateResponseSet(
        coordinate=coordinate,
        channels=tuple(channels),
        dim=int(ordered[0].dim),
        group_artifact=group.artifact(),
        null_policy=policy,
        adjoint_artifact={
            "certification": "local_small_matrix_reynolds_v1",
            "local_fixed_algorithm": LOCAL_REYNOLDS_ALGORITHM_V1,
            "local_fixed_gray_zone_factor": LOCAL_FIXED_GRAY_ZONE_FACTOR_V1,
            "logical_candidate_channel_count": int(logical_count),
            "authored_ordered_seed_count": int(len(ordered)),
            "physical_seed_count": int(len(physical_seed_indices)),
            "physical_real_owner_count": int(physical_real_count),
            "adjoint_orbit_descriptor_count": adjoint_orbit_descriptor_count,
            "hermitian_ambient_channel_count": hermitian_ambient_channel_count,
            "physically_materialized_projected_channel_count": len(channels),
            "physically_compiled_representative_channel_count": len(channels),
            "adjoint_certified_dropped_channel_count": int(
                logical_count - hermitian_ambient_channel_count
            ),
            "structural_support_count": int(structural_support_count),
            "structural_orbit_count": int(len(structural_plan.structural_orbits)),
            "duplicate_seed_groups": [
                [int(value) for value in values]
                for values in structural_plan.duplicate_seed_groups
            ],
            "zero_seed_indices": [int(value) for value in structural_plan.zero_seed_indices],
            "local_components": component_artifacts,
            "local_component_count": len(component_artifacts),
            "maximum_local_component_raw_dimension": max(
                (record["raw_dimension"] for record in component_artifacts), default=0
            ),
            "maximum_local_component_independent_dimension": max(
                (record["independent_dimension"] for record in component_artifacts), default=0
            ),
            "maximum_local_action_block_dimension": max(
                (
                    record["maximum_action_block_dimension"]
                    for record in component_artifacts
                ),
                default=0,
            ),
            "local_action_block_diagnostic": action_block_diagnostic,
            "local_fixed_rank": int(total_fixed_rank),
            "local_fixed_nonzero_count": int(total_fixed_nnz),
            "retained_real_column_count": len(channels),
            "reynolds_columns_avoided": int(physical_real_count),
            "omitted_reynolds_columns": int(physical_real_count - len(channels)),
            "raw_owner_injectivity_certificate": raw_owner_artifact,
            "rank_proofs": list(raw_rank_proofs),
            "factorized_group_actions": dict(action_artifact),
            "fallback_used": False,
            "timings_seconds": {
                "factorized_group_actions": float(action_seconds),
                "error_bound_construction": float(error_seconds),
                "structural_plan": float(structural_seconds),
                "raw_adjoint_certification": float(adjoint_seconds),
                "raw_vocabulary": float(vocabulary_seconds),
                "raw_owner_injectivity_certificate": float(raw_rank_seconds),
                "generator_actions": float(generator_seconds),
                "action_block_construction": float(component_seconds),
                "local_fixed_solve": float(local_solve_seconds),
                "channel_materialization": float(materialize_seconds),
                "function_body": float(time.perf_counter() - function_started),
            },
        },
    )


def _compile_graded_candidate_group_reynolds(
    seeds: Sequence[RawPolynomialSeed],
    *,
    coordinate: PolynomialCoordinateBasis,
    group: Any,
    factorized_actions: Mapping[str, Any],
    internal_actions_by_word: Mapping[tuple[str, ...], sparse.spmatrix] | None = None,
    factorized_group_artifact: Mapping[str, Any] | None = None,
    joint_adjoint_keys: Mapping[str, JointAdjointSeedKey] | None = None,
) -> CandidateResponseSet:
    """Compile a candidate group with filtered owner-column elimination."""

    function_started = time.perf_counter()
    descriptor_started = time.perf_counter()
    ordered = _validate_seeds(seeds)
    descriptors = tuple(
        describe_filtered_seed(seed, coordinate=coordinate, seed_index=index)
        for index, seed in enumerate(ordered)
    )
    descriptor_seconds = time.perf_counter() - descriptor_started
    started = time.perf_counter()
    if internal_actions_by_word is None:
        internal_actions, action_artifact = compile_factorized_group_element_actions(
            group=group,
            factorized_generators=factorized_actions,
        )
    else:
        if factorized_group_artifact is None:
            raise GradedCompilationUnavailable(
                "precompiled graded group actions lack their certification artifact",
                certificate={"check": "factorized_group_artifact"},
            )
        internal_actions = {
            tuple(str(value) for value in word): sparse.csr_matrix(matrix)
            for word, matrix in internal_actions_by_word.items()
        }
        action_artifact = dict(factorized_group_artifact)
    from . import response_basis as _response_basis

    try:
        contexts = tuple(
            _SymbolicGroupContext(
                element=element,
                column_routes=_column_routes(
                    internal_actions[
                        tuple(str(value) for value in element.canonical_word)
                    ]
                ),
                monomial_pullbacks=_response_basis._monomial_pullback_table(
                    element, coordinate
                ),
            )
            for element in group.elements
        )
    except KeyError as exc:
        raise GradedCompilationUnavailable(
            "graded group action is missing a finite-group word",
            certificate={"missing_word": list(exc.args[0])},
        ) from exc
    group_context_seconds = time.perf_counter() - started

    internal_error = float(action_artifact.get("maximum_action_error_bound", 0.0))
    error_components_by_seed: list[dict[str, float]] = []
    absolute_errors_by_seed: list[float] = []
    raw_adjoint_errors_by_seed: list[float] = []
    error_bounds_started = time.perf_counter()
    for seed in ordered:
        raw_norm = _coefficient_norm(seed.coefficients)
        components = _propagated_error_components(
            raw_norm,
            dim=seed.dim,
            degree=coordinate.max_degree,
            group=group,
        )
        raw_adjoint_errors_by_seed.append(
            float(
                sum(components.values())
                + float(
                    seed.metadata.get(
                        "coordinate_conversion_error_bound",
                        0.0,
                    )
                )
            )
        )
        if internal_error > 0.0:
            components["factorized_internal_action"] = float(
                (2.0 * internal_error + internal_error**2)
                * max(raw_norm, np.finfo(float).tiny)
            )
        error_components_by_seed.append(components)
        error = float(sum(components.values()))
        absolute_errors_by_seed.append(error)
    error_bound_seconds = time.perf_counter() - error_bounds_started

    structural_started = time.perf_counter()
    structural_plan = build_structural_compilation_plan(
        ordered,
        coordinate=coordinate,
        group=group,
        internal_actions_by_word=internal_actions,
        maximum_action_error_bound=internal_error,
        descriptors=descriptors,
    )
    structural_seconds = time.perf_counter() - structural_started
    structural_route_closure_diagnostic = _structural_route_closure_diagnostic(
        structural_plan,
        descriptors=descriptors,
        coordinate=coordinate,
        contexts=contexts,
    )
    certified_raw_adjoint = None
    raw_adjoint_seconds = 0.0
    if joint_adjoint_keys is not None:
        raw_adjoint_started = time.perf_counter()
        ordered_seed_ids = {str(seed.seed_id) for seed in ordered}
        if {str(seed_id) for seed_id in joint_adjoint_keys} != ordered_seed_ids:
            raise ValueError(
                "graded joint-adjoint keys must cover every authored seed exactly"
            )
        physical_seed_indices = tuple(
            int(indices[0])
            for indices in structural_plan.duplicate_seed_groups
        )
        physical_seeds = tuple(ordered[index] for index in physical_seed_indices)
        provisional_raw_adjoint = canonicalize_joint_adjoint_seeds(
            [
                JointAdjointSeed(
                    str(seed.seed_id),
                    joint_adjoint_keys[str(seed.seed_id)],
                )
                for seed in physical_seeds
            ]
        )
        certified_raw_adjoint = certify_joint_adjoint_coefficients(
            provisional_raw_adjoint,
            {
                str(seed.seed_id): seed.coefficients
                for seed in physical_seeds
            },
            absolute_error_bounds_by_seed={
                str(ordered[index].seed_id): float(
                    raw_adjoint_errors_by_seed[index]
                )
                for index in physical_seed_indices
            },
        )
        raw_adjoint_seconds = time.perf_counter() - raw_adjoint_started
    homogeneous_started = time.perf_counter()
    homogeneous_certificates = _homogeneous_action_certificates(
        group=group,
        coordinate=coordinate,
        contexts=contexts,
        degrees=[
            descriptors[index].nominal_degree
            for index in structural_plan.representative_seed_indices
        ],
    )
    homogeneous_seconds = time.perf_counter() - homogeneous_started
    duplicates_by_representative = {
        group_indices[0]: group_indices
        for group_indices in structural_plan.duplicate_seed_groups
    }
    row_degrees = projected_row_degrees(
        coordinate=coordinate,
        dim=ordered[0].dim,
    )
    seed_atoms_seconds = 0.0
    projection_seconds = 0.0
    reduction_seconds = 0.0
    projection_batches: list[dict[str, Any]] = []
    orbit_artifacts: list[dict[str, Any]] = []
    projected_block_records: list[dict[str, Any]] = []
    orbit_hstack_seconds = 0.0
    for orbit in structural_plan.structural_orbits:
        by_degree: dict[int, list[int]] = {}
        for seed_index in orbit.representative_seed_indices:
            by_degree.setdefault(descriptors[seed_index].nominal_degree, []).append(
                seed_index
            )
        orbit_blocks: list[sparse.csc_matrix] = []
        orbit_owner_indices: list[int] = []
        orbit_nominal_degrees: list[int] = []
        orbit_error_bounds: list[float] = []
        orbit_batches: list[dict[str, Any]] = []
        orbit_candidate_nonzero_count = 0
        for degree, seed_indices in sorted(by_degree.items()):
            batch = tuple(ordered[index] for index in seed_indices)
            atoms_started = time.perf_counter()
            batch_atoms = tuple(_seed_symbolic_atoms(seed) for seed in batch)
            seed_atoms_seconds += time.perf_counter() - atoms_started
            projection_started = time.perf_counter()
            projected_batch = _symbolic_reynolds_batch(
                batch,
                batch_atoms,
                coordinate=coordinate,
                contexts=contexts,
            )
            batch_seconds = time.perf_counter() - projection_started
            projection_seconds += batch_seconds
            orbit_blocks.append(projected_batch)
            for seed_index in seed_indices:
                duplicate_indices = duplicates_by_representative[seed_index]
                error_bound = max(
                    absolute_errors_by_seed[index] for index in duplicate_indices
                )
                owners = (2 * seed_index, 2 * seed_index + 1)
                orbit_owner_indices.extend(owners)
                orbit_nominal_degrees.extend((degree, degree))
                orbit_error_bounds.extend((error_bound, error_bound))
            orbit_candidate_nonzero_count += int(projected_batch.nnz)
            batch_artifact = {
                "orbit_index": int(orbit.orbit_index),
                "degree": int(degree),
                "complex_seed_count": int(len(seed_indices)),
                "real_column_count": int(projected_batch.shape[1]),
                "ambient_row_count": int(projected_batch.shape[0]),
                "nonzero_count": int(projected_batch.nnz),
                "seconds": float(batch_seconds),
            }
            orbit_batches.append(batch_artifact)
            projection_batches.append(batch_artifact)
        if not orbit_owner_indices:
            continue
        orbit_hstack_started = time.perf_counter()
        orbit_projected = sparse.hstack(orbit_blocks, format="csc")
        orbit_hstack_seconds += time.perf_counter() - orbit_hstack_started
        projected_block_records.append(
            {
                "orbit_index": int(orbit.orbit_index),
                "projected": orbit_projected,
                "owner_indices": tuple(
                    int(value) for value in orbit_owner_indices
                ),
                "nominal_degrees": np.asarray(
                    orbit_nominal_degrees, dtype=np.int64
                ),
                "error_bounds": np.asarray(
                    orbit_error_bounds, dtype=np.float64
                ),
            }
        )
        orbit_artifacts.append(
            {
                "orbit_index": int(orbit.orbit_index),
                "support_ids": list(orbit.support_ids),
                "representative_seed_indices": [
                    int(value) for value in orbit.representative_seed_indices
                ],
                "candidate_real_column_count": int(len(orbit_owner_indices)),
                "candidate_nonzero_count": int(orbit_candidate_nonzero_count),
                "retained_real_column_count": 0,
                "selected_global_owner_indices": [],
                "selection_scope": "global_owner_degree_slices_v1",
                "degree_batches": orbit_batches,
            }
        )
    reduction_started = time.perf_counter()
    global_selection, global_owner_map, global_owner_artifact = (
        _select_filtered_independent_column_blocks(
            [record["projected"] for record in projected_block_records],
            owner_index_blocks=[
                record["owner_indices"] for record in projected_block_records
            ],
            nominal_degree_blocks=[
                record["nominal_degrees"] for record in projected_block_records
            ],
            error_bound_blocks=[
                record["error_bounds"] for record in projected_block_records
            ],
            row_degrees=row_degrees,
        )
    )
    reduction_seconds = time.perf_counter() - reduction_started
    selection_scope = str(global_owner_artifact["strategy"])
    selected_lookup_started = time.perf_counter()
    column_locations_by_owner: dict[int, tuple[dict[str, Any], int]] = {}
    for record in projected_block_records:
        for local_index, global_owner in enumerate(record["owner_indices"]):
            column_locations_by_owner[int(global_owner)] = (
                record,
                int(local_index),
            )
    selected_global_owners = tuple(
        int(global_owner_map[index])
        for index in global_selection.selected_owner_indices
    )
    selected_records: list[dict[str, Any]] = []
    for global_owner in selected_global_owners:
        block_record, local_index = column_locations_by_owner[global_owner]
        selected_records.append(
            {
                "global_owner": int(global_owner),
                "seed_index": int(global_owner) // 2,
                "component_index": int(global_owner) % 2,
                "orbit_index": int(block_record["orbit_index"]),
                "column": block_record["projected"][:, local_index].tocsc(),
            }
        )
    selected_lookup_seconds = time.perf_counter() - selected_lookup_started
    selected_by_orbit: dict[int, list[int]] = {}
    for record in selected_records:
        selected_by_orbit.setdefault(int(record["orbit_index"]), []).append(
            int(record["global_owner"])
        )
    for artifact in orbit_artifacts:
        selected_owners = selected_by_orbit.get(int(artifact["orbit_index"]), [])
        artifact["retained_real_column_count"] = int(len(selected_owners))
        artifact["selected_global_owner_indices"] = selected_owners
        artifact["selection_scope"] = selection_scope
    all_rank_proofs = tuple(
        {
            **dict(proof),
            "selection_scope": selection_scope,
            "local_to_global_owner_indices": [
                int(value) for value in global_owner_map
            ],
            "selected_global_owner_indices": [
                int(global_owner_map[index])
                for index in proof.get("selected_owner_indices", [])
            ],
        }
        for proof in global_selection.degree_proofs
    )
    global_owner_artifact = {
        **dict(global_owner_artifact),
        "rank_proofs": list(all_rank_proofs),
        "owner_column_materialization_policy": (
            "selected_global_owner_only_v1"
        ),
        "materialized_owner_column_count": int(len(selected_records)),
        "avoided_owner_column_materialization_count": int(
            len(global_owner_map) - len(selected_records)
        ),
    }
    channel_ids = tuple(
        f"{seed.seed_id}:{component}"
        for seed in ordered
        for component in ("real", "imag")
    )

    materialize_started = time.perf_counter()
    policy = NullClassificationPolicy()
    channels: list[ResponseChannel] = []
    for record in selected_records:
        seed_index = int(record["seed_index"])
        component_index = int(record["component_index"])
        channel_index = int(record["global_owner"])
        seed = ordered[seed_index]
        component = ("real", "imag")[component_index]
        coefficients = _coefficient_vector_to_response_map(
            record["column"],
            coordinate=coordinate,
            dim=seed.dim,
        )
        response_norm = _coefficient_norm(coefficients)
        components = dict(error_components_by_seed[seed_index])
        duplicate_indices = duplicates_by_representative[seed_index]
        duplicate_error = max(
            absolute_errors_by_seed[index] for index in duplicate_indices
        )
        own_error = float(sum(components.values()))
        if duplicate_error > own_error:
            components["duplicate_source_envelope"] = float(
                duplicate_error - own_error
            )
        error_bound = float(sum(components.values()))
        provenance = [
            _seed_provenance(ordered[index], descriptors[index])
            for index in duplicate_indices
        ]
        channels.append(
            ResponseChannel(
                channel_id=channel_ids[channel_index],
                seed_id=str(seed.seed_id),
                component=component,
                coefficients=coefficients,
                unnormalized_norm=response_norm,
                propagated_error_bound=error_bound,
                classification=policy.classify(response_norm, error_bound),
                response_scale=response_norm,
                support_component=str(seed.support_component),
                metadata={
                    **dict(seed.metadata),
                    "symbolic_atom_compiler": "graded_filtered_reynolds_v1",
                    "graded_nominal_degree": int(
                        descriptors[seed_index].nominal_degree
                    ),
                    "graded_structural_support_id": str(
                        descriptors[seed_index].structural_support_id
                    ),
                    "graded_structural_orbit_index": int(record["orbit_index"]),
                    "graded_source_provenance": provenance,
                    "graded_duplicate_seed_indices": [
                        int(value) for value in duplicate_indices
                    ],
                },
                error_bound_components=components,
            )
        )
    materialize_seconds = time.perf_counter() - materialize_started
    logical_count = 2 * len(ordered)
    retained_count = len(channels)
    if certified_raw_adjoint is None:
        adjoint_orbit_descriptor_count = int(
            len(structural_plan.representative_seed_indices)
        )
        hermitian_ambient_channel_count = int(logical_count)
        adjoint_certified_dropped_channel_count = int(
            logical_count - retained_count
        )
    else:
        adjoint_orbit_descriptor_count = int(
            len(certified_raw_adjoint.orbits)
        )
        hermitian_ambient_channel_count = int(
            sum(
                len(orbit.channel_ids)
                for orbit in certified_raw_adjoint.orbits
            )
        )
        adjoint_certified_dropped_channel_count = int(
            2 * len(structural_plan.representative_seed_indices)
            - hermitian_ambient_channel_count
        )
    structural_support_count = int(
        len(
            {
                descriptors[index].structural_support_id
                for index in structural_plan.representative_seed_indices
            }
        )
    )
    return CandidateResponseSet(
        coordinate=coordinate,
        channels=tuple(channels),
        dim=int(ordered[0].dim),
        group_artifact=group.artifact(),
        null_policy=policy,
        adjoint_artifact={
            "certification": "graded_filtered_reynolds_v1",
            "logical_candidate_channel_count": int(logical_count),
            "authored_ordered_seed_count": int(len(ordered)),
            "physical_seed_count": int(
                len(structural_plan.representative_seed_indices)
            ),
            "structural_support_count": structural_support_count,
            "closed_structural_direction_count": int(
                sum(
                    len(orbit.support_ids)
                    for orbit in structural_plan.structural_orbits
                )
            ),
            "structural_route_closure_diagnostic": (
                structural_route_closure_diagnostic
            ),
            "structural_orbit_count": int(len(structural_plan.structural_orbits)),
            "structural_probe_seed_indices": [
                int(value)
                for value in structural_plan.structural_probe_seed_indices
            ],
            "structural_probe_degrees": [
                int(descriptors[value].nominal_degree)
                for value in structural_plan.structural_probe_seed_indices
            ],
            "structural_probe_max_degree": int(
                max(
                    (
                        descriptors[value].nominal_degree
                        for value in structural_plan.structural_probe_seed_indices
                    ),
                    default=0,
                )
            ),
            "family_degree_limits": dict(structural_plan.family_degree_limits),
            "duplicate_seed_groups": [
                [int(value) for value in group_indices]
                for group_indices in structural_plan.duplicate_seed_groups
            ],
            "zero_seed_indices": [
                int(value) for value in structural_plan.zero_seed_indices
            ],
            "structural_orbits": orbit_artifacts,
            "projection_batches": projection_batches,
            "maximum_projected_batch_complex_seed_count": int(
                max(
                    (item["complex_seed_count"] for item in projection_batches),
                    default=0,
                )
            ),
            "maximum_projected_batch_real_column_count": int(
                max(
                    (item["real_column_count"] for item in projection_batches),
                    default=0,
                )
            ),
            "maximum_orbit_hstack_real_column_count": int(
                max(
                    (
                        item["candidate_real_column_count"]
                        for item in orbit_artifacts
                    ),
                    default=0,
                )
            ),
            "maximum_orbit_hstack_nonzero_count": int(
                max(
                    (item["candidate_nonzero_count"] for item in orbit_artifacts),
                    default=0,
                )
            ),
            "orbit_hstack_materialized": True,
            "global_authored_projected_matrix_avoided": True,
            "global_filtered_owner_certificate": dict(global_owner_artifact),
            "homogeneous_degree_actions": {
                str(degree): dict(certificate)
                for degree, certificate in homogeneous_certificates.items()
            },
            "adjoint_orbit_descriptor_count": adjoint_orbit_descriptor_count,
            "hermitian_ambient_channel_count": hermitian_ambient_channel_count,
            "physically_materialized_projected_channel_count": int(retained_count),
            "physically_compiled_representative_channel_count": int(retained_count),
            "adjoint_certified_dropped_channel_count": (
                adjoint_certified_dropped_channel_count
            ),
            "rank_proofs": list(all_rank_proofs),
            "factorized_group_actions": dict(action_artifact),
            "reynolds_internal_action": dict(action_artifact),
            "timings_seconds": {
                "group_context": float(group_context_seconds),
                "seed_validation_and_descriptors": float(
                    descriptor_seconds
                ),
                "error_bound_construction": float(error_bound_seconds),
                "structural_plan": float(structural_seconds),
                "raw_adjoint_certification": float(raw_adjoint_seconds),
                "homogeneous_actions": float(homogeneous_seconds),
                "seed_atoms": float(seed_atoms_seconds),
                "symbolic_projection": float(projection_seconds),
                "orbit_hstack": float(orbit_hstack_seconds),
                "graded_rank_selection": float(reduction_seconds),
                "global_degree_slice_assembly": float(
                    global_owner_artifact["slice_assembly_seconds"]
                ),
                "selected_owner_column_lookup": float(
                    selected_lookup_seconds
                ),
                "channel_materialization": float(materialize_seconds),
                "function_body": float(
                    time.perf_counter() - function_started
                ),
            },
        },
    )


def compile_graded_candidate_group(
    seeds: Sequence[RawPolynomialSeed],
    *,
    coordinate: PolynomialCoordinateBasis,
    group: Any,
    factorized_actions: Mapping[str, Any],
    internal_actions_by_word: Mapping[tuple[str, ...], sparse.spmatrix] | None = None,
    factorized_group_artifact: Mapping[str, Any] | None = None,
    joint_adjoint_keys: Mapping[str, JointAdjointSeedKey] | None = None,
    force_reynolds: bool = False,
) -> CandidateResponseSet:
    """Compile p=0 by local Reynolds; route finite-p directly to the legacy path."""

    def ambient_reynolds(
        fallback_reason: str | None = None,
    ) -> CandidateResponseSet:
        compiled = _compile_graded_candidate_group_reynolds(
            seeds,
            coordinate=coordinate,
            group=group,
            factorized_actions=factorized_actions,
            internal_actions_by_word=internal_actions_by_word,
            factorized_group_artifact=factorized_group_artifact,
            joint_adjoint_keys=joint_adjoint_keys,
        )
        if fallback_reason is None:
            return compiled
        return replace(
            compiled,
            adjoint_artifact={
                **dict(compiled.adjoint_artifact),
                "fallback_used": True,
                "fallback_reason": str(fallback_reason),
            },
        )

    if bool(force_reynolds):
        return ambient_reynolds()
    if joint_adjoint_keys is None:
        return ambient_reynolds("missing_certified_adjoint_routes")
    if (
        not factorized_actions
        and not (
            len(tuple(group.elements)) == 1
            and not bool(group.elements[0].antiunitary)
        )
    ):
        return ambient_reynolds("missing_certified_generator_routes")
    try:
        return _compile_graded_candidate_group_local_fixed(
            seeds,
            coordinate=coordinate,
            group=group,
            factorized_actions=factorized_actions,
            internal_actions_by_word=internal_actions_by_word,
            factorized_group_artifact=factorized_group_artifact,
            joint_adjoint_keys=joint_adjoint_keys,
        )
    except LocalResponseCompilationError as error:
        if error.reason not in {
            "missing_certified_generator_route",
            "missing_certified_generator_routes",
            "raw_owner_vocabulary_not_injective",
        }:
            raise
        return ambient_reynolds(error.reason)


__all__ = [
    "FilteredSeedDescriptor",
    "GradedColumnSelection",
    "GradedCompilationUnavailable",
    "StructuralCompilationPlan",
    "StructuralOrbit",
    "build_structural_compilation_plan",
    "compile_graded_candidate_group",
    "compile_homogeneous_momentum_action",
    "describe_filtered_seed",
    "projected_row_degrees",
    "select_filtered_independent_columns",
]

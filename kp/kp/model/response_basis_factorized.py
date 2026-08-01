"""Direct term-space actions from certified factorized symmetries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse

from ..identity import hash_array


class FactorizedTermActionError(ValueError):
    """The authored term vocabulary is not closed under a factorized action."""


def _phase_aligned_frobenius_residual(
    reference: np.ndarray,
    candidate: sparse.spmatrix | np.ndarray,
) -> float:
    reference_array = np.asarray(reference, dtype=np.complex128)
    candidate_array = (
        np.asarray(candidate.toarray(), dtype=np.complex128)
        if sparse.issparse(candidate)
        else np.asarray(candidate, dtype=np.complex128)
    )
    overlap = complex(np.vdot(candidate_array, reference_array))
    phase = overlap / abs(overlap) if abs(overlap) > 0.0 else 1.0 + 0.0j
    return float(np.linalg.norm(reference_array - phase * candidate_array, ord="fro"))


def compile_factorized_group_element_actions(
    *,
    group: Any,
    factorized_generators: Mapping[str, Any],
) -> tuple[dict[tuple[str, ...], sparse.csr_matrix], dict[str, Any]]:
    """Compose certified sparse internal matrices for every finite-group word.

    The authored Fourier seed vocabulary need not itself be group closed.  These
    matrices therefore act in the physical Hamiltonian coefficient space, not
    in the authored seed index space, and can be used by a complete Reynolds
    average without materializing numerical off-route tails in the dense
    exactified matrices.
    """

    from ..symmetry.factorized_action import materialize_factorized_matrix

    elements = tuple(getattr(group, "elements", ()))
    if not elements:
        raise FactorizedTermActionError(
            "factorized group-element compilation requires a nonempty finite group"
        )
    generator_elements = {
        str(element.canonical_word[0]): element
        for element in elements
        if len(tuple(element.canonical_word)) == 1
    }
    actions = {str(name): action for name, action in factorized_generators.items()}
    required_names = {
        str(name)
        for element in elements
        for name in tuple(element.canonical_word)
    }
    missing = sorted(required_names - set(actions))
    if missing:
        raise FactorizedTermActionError(
            f"certified factorized generators are missing group words: {missing}"
        )
    dim = int(np.asarray(elements[0].internal_u).shape[0])
    if any(np.asarray(element.internal_u).shape != (dim, dim) for element in elements):
        raise FactorizedTermActionError(
            "finite-group internal matrices do not share one square dimension"
        )

    sparse_generators: dict[str, sparse.csr_matrix] = {}
    generator_bounds: dict[str, float] = {}
    generator_residuals: dict[str, float] = {}
    for name in sorted(required_names):
        action = actions[name]
        element = generator_elements.get(name)
        if element is None:
            raise FactorizedTermActionError(
                f"finite group has no canonical one-generator element for {name!r}"
            )
        if str(getattr(action, "name", "")) != name:
            raise FactorizedTermActionError(
                f"factorized artifact name {getattr(action, 'name', None)!r} "
                f"does not match generator {name!r}"
            )
        if bool(action.antiunitary) != bool(element.antiunitary):
            raise FactorizedTermActionError(
                f"factorized generator {name!r} has the wrong antiunitary parity"
            )
        if tuple(int(value) for value in action.q_permutation) != tuple(
            int(value) for value in element.q_permutation
        ):
            raise FactorizedTermActionError(
                f"factorized generator {name!r} has the wrong Q permutation"
            )
        if tuple(int(value) for value in action.sector_permutation) != tuple(
            int(value) for value in element.sector_permutation
        ):
            raise FactorizedTermActionError(
                f"factorized generator {name!r} has the wrong sector permutation"
            )
        expected_forward = np.linalg.inv(
            np.asarray(element.k_pullback, dtype=np.float64)
        )
        actual_forward = np.asarray(action.k_forward, dtype=np.float64)
        map_bound = float(
            128.0
            * np.finfo(np.float64).eps
            * max(1.0, np.linalg.norm(expected_forward, ord=2))
        )
        if not np.allclose(
            actual_forward,
            expected_forward,
            rtol=0.0,
            atol=map_bound,
        ):
            raise FactorizedTermActionError(
                f"factorized generator {name!r} has the wrong Cartesian k action"
            )
        reconstructed = materialize_factorized_matrix(action)
        if reconstructed.shape != (dim, dim):
            raise FactorizedTermActionError(
                f"factorized generator {name!r} has shape {reconstructed.shape}, "
                f"expected {(dim, dim)}"
            )
        residual = _phase_aligned_frobenius_residual(
            np.asarray(element.internal_u, dtype=np.complex128),
            reconstructed,
        )
        bound = float(
            action.off_route_residual
            + np.sqrt(max(1, len(action.q_phases)))
            * action.phase_alignment_residual
            + action.matrix_certification_bound
            + 128.0
            * np.finfo(np.float64).eps
            * max(1.0, np.linalg.norm(element.internal_u, ord="fro"))
        )
        if residual > bound:
            raise FactorizedTermActionError(
                f"factorized generator {name!r} does not match the active action: "
                f"residual={residual:.6e}, bound={bound:.6e}"
            )
        matrix = sparse.csr_matrix(reconstructed, dtype=np.complex128)
        matrix.eliminate_zeros()
        matrix.sort_indices()
        sparse_generators[name] = matrix
        generator_bounds[name] = bound
        generator_residuals[name] = residual

    identity = sparse.eye(dim, format="csr", dtype=np.complex128)
    action_by_word: dict[tuple[str, ...], sparse.csr_matrix] = {}
    element_records: list[dict[str, Any]] = []
    maximum_bound = 0.0
    maximum_residual = 0.0
    maximum_nnz = 0
    algebra_residual = float(getattr(group, "algebra_residual", 0.0))
    roundoff = float(
        128.0
        * np.finfo(np.float64).eps
        * max(1, dim)
        * max(1.0, np.sqrt(float(dim)))
    )
    for element in elements:
        word = tuple(str(value) for value in element.canonical_word)
        composed = identity.copy()
        accumulated_bound = 0.0
        for name in word:
            action = actions[name]
            inner = composed.conjugate() if bool(action.antiunitary) else composed
            composed = (sparse_generators[name] @ inner).tocsr()
            composed.eliminate_zeros()
            composed.sort_indices()
            accumulated_bound += float(generator_bounds[name] + roundoff)
        residual = _phase_aligned_frobenius_residual(
            np.asarray(element.internal_u, dtype=np.complex128),
            composed,
        )
        bound = float(
            accumulated_bound
            + (len(word) + 1)
            * (
                algebra_residual * max(1.0, np.sqrt(float(dim)))
                + roundoff
            )
        )
        if residual > bound:
            raise FactorizedTermActionError(
                f"factorized sparse word {word!r} does not match the finite-group "
                f"element: residual={residual:.6e}, bound={bound:.6e}"
            )
        action_by_word[word] = composed
        maximum_bound = max(maximum_bound, bound)
        maximum_residual = max(maximum_residual, residual)
        maximum_nnz = max(maximum_nnz, int(composed.nnz))
        element_records.append(
            {
                "canonical_word": list(word),
                "nnz": int(composed.nnz),
                "phase_aligned_residual": residual,
                "absolute_error_bound": bound,
            }
        )
    return action_by_word, {
        "compiler": "factorized_sparse_group_elements_v1",
        "generator_names": sorted(required_names),
        "generator_action_hashes": {
            name: str(actions[name].artifact_hash) for name in sorted(required_names)
        },
        "generator_match_residuals": dict(sorted(generator_residuals.items())),
        "generator_match_bounds": dict(sorted(generator_bounds.items())),
        "element_count": int(len(action_by_word)),
        "maximum_element_nnz": int(maximum_nnz),
        "maximum_action_match_residual": float(maximum_residual),
        "maximum_action_error_bound": float(maximum_bound),
        "dense_internal_transform_count": 0,
        "elements": element_records,
    }


def compile_joint_route_group_element_actions(
    *,
    group: Any,
    joint_route_generators: Mapping[str, Any],
    joint_artifact_hash: str,
) -> tuple[dict[tuple[str, ...], sparse.csr_matrix], dict[str, Any]]:
    """Compose group words from certified sparse joint route blocks."""

    from ..symmetry.joint_exactification import materialize_block_route_action

    elements = tuple(getattr(group, "elements", ()))
    if not elements:
        raise FactorizedTermActionError(
            "joint-route group-element compilation requires a nonempty finite group"
        )
    artifact_hash = str(joint_artifact_hash)
    if len(artifact_hash) != 64 or any(
        character not in "0123456789abcdef" for character in artifact_hash
    ):
        raise FactorizedTermActionError("joint route artifact hash must be SHA-256")
    actions = {
        str(name): action for name, action in joint_route_generators.items()
    }
    required_names = {
        str(name)
        for element in elements
        for name in tuple(element.canonical_word)
    }
    missing = sorted(required_names - set(actions))
    if missing:
        raise FactorizedTermActionError(
            f"certified joint route generators are missing group words: {missing}"
        )
    generator_elements = {
        str(element.canonical_word[0]): element
        for element in elements
        if len(tuple(element.canonical_word)) == 1
    }
    dim = int(np.asarray(elements[0].internal_u).shape[0])
    if any(np.asarray(element.internal_u).shape != (dim, dim) for element in elements):
        raise FactorizedTermActionError(
            "finite-group internal matrices do not share one square dimension"
        )

    sparse_generators: dict[str, sparse.csr_matrix] = {}
    generator_bounds: dict[str, float] = {}
    generator_residuals: dict[str, float] = {}
    generator_hashes: dict[str, str] = {}
    roundoff = float(
        128.0
        * np.finfo(np.float64).eps
        * max(1, dim)
        * max(1.0, np.sqrt(float(dim)))
    )
    for name in sorted(required_names):
        action = actions[name]
        element = generator_elements.get(name)
        if element is None:
            raise FactorizedTermActionError(
                f"finite group has no canonical one-generator element for {name!r}"
            )
        if str(getattr(action, "name", "")) != name:
            raise FactorizedTermActionError(
                f"joint route artifact name {getattr(action, 'name', None)!r} "
                f"does not match generator {name!r}"
            )
        if bool(action.antiunitary) != bool(element.antiunitary):
            raise FactorizedTermActionError(
                f"joint route generator {name!r} has the wrong antiunitary parity"
            )
        reconstructed = materialize_block_route_action(action)
        if reconstructed.shape != (dim, dim):
            raise FactorizedTermActionError(
                f"joint route generator {name!r} has shape {reconstructed.shape}, "
                f"expected {(dim, dim)}"
            )
        residual = _phase_aligned_frobenius_residual(
            np.asarray(element.internal_u, dtype=np.complex128),
            reconstructed,
        )
        bound = float(
            getattr(action, "unitarity_certification_bound", 0.0)
            + float(getattr(group, "algebra_residual", 0.0))
            * max(1.0, np.sqrt(float(dim)))
            + roundoff
        )
        if residual > bound:
            raise FactorizedTermActionError(
                f"joint route generator {name!r} does not match the active action: "
                f"residual={residual:.6e}, bound={bound:.6e}"
            )
        matrix = sparse.csr_matrix(reconstructed, dtype=np.complex128)
        matrix.eliminate_zeros()
        matrix.sort_indices()
        sparse_generators[name] = matrix
        generator_bounds[name] = bound
        generator_residuals[name] = residual
        generator_hashes[name] = hash_array(reconstructed)

    identity = sparse.eye(dim, format="csr", dtype=np.complex128)
    action_by_word: dict[tuple[str, ...], sparse.csr_matrix] = {}
    element_records: list[dict[str, Any]] = []
    maximum_bound = 0.0
    maximum_residual = 0.0
    maximum_nnz = 0
    algebra_residual = float(getattr(group, "algebra_residual", 0.0))
    for element in elements:
        word = tuple(str(value) for value in element.canonical_word)
        composed = identity.copy()
        accumulated_bound = 0.0
        for name in word:
            action = actions[name]
            inner = composed.conjugate() if bool(action.antiunitary) else composed
            composed = (sparse_generators[name] @ inner).tocsr()
            composed.eliminate_zeros()
            composed.sort_indices()
            accumulated_bound += float(generator_bounds[name] + roundoff)
        residual = _phase_aligned_frobenius_residual(
            np.asarray(element.internal_u, dtype=np.complex128),
            composed,
        )
        bound = float(
            accumulated_bound
            + (len(word) + 1)
            * (algebra_residual * max(1.0, np.sqrt(float(dim))) + roundoff)
        )
        if residual > bound:
            raise FactorizedTermActionError(
                f"joint route sparse word {word!r} does not match the finite-group "
                f"element: residual={residual:.6e}, bound={bound:.6e}"
            )
        action_by_word[word] = composed
        maximum_bound = max(maximum_bound, bound)
        maximum_residual = max(maximum_residual, residual)
        maximum_nnz = max(maximum_nnz, int(composed.nnz))
        element_records.append(
            {
                "canonical_word": list(word),
                "nnz": int(composed.nnz),
                "phase_aligned_residual": residual,
                "absolute_error_bound": bound,
            }
        )
    return action_by_word, {
        "compiler": "joint_route_sparse_group_elements_v1",
        "joint_artifact_hash": artifact_hash,
        "generator_names": sorted(required_names),
        "generator_action_hashes": dict(sorted(generator_hashes.items())),
        "generator_match_residuals": dict(sorted(generator_residuals.items())),
        "generator_match_bounds": dict(sorted(generator_bounds.items())),
        "element_count": int(len(action_by_word)),
        "maximum_element_nnz": int(maximum_nnz),
        "maximum_action_match_residual": float(maximum_residual),
        "maximum_action_error_bound": float(maximum_bound),
        "dense_internal_transform_count": 0,
        "elements": element_records,
    }


@dataclass(frozen=True)
class _OrderedSeed:
    seed_id: str
    support_component: str
    layer_from: int
    layer_to: int
    orbital_from: int
    orbital_to: int
    r: int
    s: int
    q_pairs: tuple[tuple[int, int], ...]

    @classmethod
    def from_raw(
        cls,
        seed: Any,
        *,
        q_counts: Sequence[int],
        n_orb: Sequence[int],
    ) -> "_OrderedSeed":
        metadata = getattr(seed, "metadata", {})
        term_key = metadata.get("term_key") if isinstance(metadata, Mapping) else None
        if not isinstance(term_key, Mapping):
            raise FactorizedTermActionError(
                f"factorized action seed {getattr(seed, 'seed_id', '<unknown>')!r} lacks term_key"
            )
        p = np.asarray(term_key.get("p"), dtype=np.float64)
        layer_from = int(term_key["layer_from"])
        layer_to = int(term_key["layer_to"])
        if p.shape != (2,) or not np.all(np.isfinite(p)):
            raise FactorizedTermActionError("factorized term action seed has invalid p")
        sector_count = len(tuple(q_counts))
        if len(tuple(n_orb)) != sector_count:
            raise FactorizedTermActionError("factorized q_counts/n_orb layout is inconsistent")
        source_sector = layer_from - 1
        target_sector = layer_to - 1
        if not (0 <= source_sector < sector_count and 0 <= target_sector < sector_count):
            raise FactorizedTermActionError("factorized term action seed layer is out of range")
        orbital_from = int(term_key["orbital_from"]) - 1
        orbital_to = int(term_key["orbital_to"]) - 1
        if not (
            0 <= orbital_from < int(n_orb[source_sector])
            and 0 <= orbital_to < int(n_orb[target_sector])
        ):
            raise FactorizedTermActionError("factorized term action seed orbital is out of range")
        counts = tuple(int(value) for value in q_counts)
        orbitals = tuple(int(value) for value in n_orb)
        basis_offsets = np.cumsum(
            (0, *(counts[index] * orbitals[index] for index in range(sector_count - 1)))
        ).astype(np.int64)
        expected_dim = int(sum(counts[index] * orbitals[index] for index in range(sector_count)))

        def decode(index: int) -> tuple[int, int, int]:
            raw_index = int(index)
            for sector in range(sector_count):
                start = int(basis_offsets[sector])
                stop = start + counts[sector] * orbitals[sector]
                if start <= raw_index < stop:
                    local = raw_index - start
                    return sector, local // counts[sector], local % counts[sector]
            raise FactorizedTermActionError(
                f"factorized seed matrix index {raw_index} exceeds dimension {expected_dim}"
            )

        q_pairs: set[tuple[int, int]] = set()
        for matrix in getattr(seed, "coefficients", {}).values():
            if tuple(matrix.shape) != (expected_dim, expected_dim):
                raise FactorizedTermActionError(
                    f"factorized seed matrix shape {matrix.shape} does not match "
                    f"the certified basis dimension {expected_dim}"
                )
            rows, columns = matrix.nonzero()
            for row, column in zip(rows, columns):
                row_sector, row_orbital, row_q = decode(int(row))
                col_sector, col_orbital, col_q = decode(int(column))
                if (
                    row_sector != source_sector
                    or col_sector != target_sector
                    or row_orbital != orbital_from
                    or col_orbital != orbital_to
                ):
                    raise FactorizedTermActionError(
                        f"factorized seed {getattr(seed, 'seed_id', '<unknown>')!r} "
                        "matrix support disagrees with its term_key"
                    )
                q_pairs.add((int(row_q), int(col_q)))
        return cls(
            seed_id=str(seed.seed_id),
            support_component=str(seed.support_component),
            layer_from=layer_from,
            layer_to=layer_to,
            orbital_from=orbital_from,
            orbital_to=orbital_to,
            r=int(term_key["Mz"]),
            s=int(term_key["Mz_star"]),
            q_pairs=tuple(sorted(q_pairs)),
        )

    @property
    def lookup_key(self) -> tuple[Any, ...]:
        return (
            self.support_component,
            self.layer_from,
            self.layer_to,
            self.orbital_from,
            self.orbital_to,
            self.r,
            self.s,
            self.q_pairs,
        )


def _routed_q_support_and_phase(
    seed: _OrderedSeed,
    *,
    factorized_action: Any,
    cleanup_bound: float,
) -> tuple[tuple[tuple[int, int], ...], complex]:
    """Route one seed's exact Q-pair support and certify its scalar phase ratio."""

    if not seed.q_pairs:
        raise FactorizedTermActionError(
            f"factorized physical seed {seed.seed_id!r} has empty Q-pair support"
        )
    q_counts = tuple(int(value) for value in factorized_action.q_counts)
    q_offsets = np.cumsum((0, *q_counts[:-1])).astype(np.int64)
    sector_permutation = tuple(int(value) for value in factorized_action.sector_permutation)
    q_permutation = tuple(int(value) for value in factorized_action.q_permutation)
    q_phases = tuple(complex(value) for value in factorized_action.q_phases)
    source_left = seed.layer_from - 1
    source_right = seed.layer_to - 1
    target_left = sector_permutation[source_left]
    target_right = sector_permutation[source_right]
    routed: set[tuple[int, int]] = set()
    phase_values: list[complex] = []
    for row_q, col_q in seed.q_pairs:
        row_global = int(q_offsets[source_left]) + int(row_q)
        col_global = int(q_offsets[source_right]) + int(col_q)
        target_row_global = q_permutation[row_global]
        target_col_global = q_permutation[col_global]
        target_row_q = target_row_global - int(q_offsets[target_left])
        target_col_q = target_col_global - int(q_offsets[target_right])
        if not (
            0 <= target_row_q < q_counts[target_left]
            and 0 <= target_col_q < q_counts[target_right]
        ):
            raise FactorizedTermActionError(
                f"factorized Q route for seed {seed.seed_id!r} leaves its target sectors"
            )
        routed.add((int(target_row_q), int(target_col_q)))
        phase_values.append(q_phases[row_global] * np.conjugate(q_phases[col_global]))
    phase_reference = complex(phase_values[0])
    phase_spread = max(abs(value - phase_reference) for value in phase_values)
    phase_constancy_bound = float(
        4.0 * max(cleanup_bound, np.finfo(np.float64).eps)
        * max(1.0, abs(phase_reference))
    )
    if phase_spread > phase_constancy_bound:
        raise FactorizedTermActionError(
            f"factorized Q phase ratio for seed {seed.seed_id!r} is not constant on "
            f"its support: spread={phase_spread:.6e}, bound={phase_constancy_bound:.6e}"
        )
    return tuple(sorted(routed)), complex(sum(phase_values) / len(phase_values))


def _poly_multiply(
    left: Mapping[tuple[int, int], complex],
    right: Mapping[tuple[int, int], complex],
) -> dict[tuple[int, int], complex]:
    out: dict[tuple[int, int], complex] = {}
    for (lr, ls), left_value in left.items():
        for (rr, rs), right_value in right.items():
            key = (int(lr + rr), int(ls + rs))
            out[key] = out.get(key, 0.0j) + complex(left_value) * complex(right_value)
    return {key: value for key, value in out.items() if value != 0.0j}


def _poly_power(
    base: Mapping[tuple[int, int], complex], exponent: int
) -> dict[tuple[int, int], complex]:
    out: dict[tuple[int, int], complex] = {(0, 0): 1.0 + 0.0j}
    for _ in range(int(exponent)):
        out = _poly_multiply(out, base)
    return out


def _relative_monomial_image(
    r: int,
    s: int,
    *,
    k_forward: Sequence[Sequence[float]],
    antiunitary: bool,
) -> dict[tuple[int, int], complex]:
    pullback = np.linalg.inv(np.asarray(k_forward, dtype=np.float64))
    a = 0.5 * (pullback[0, 0] + pullback[1, 1]) + 0.5j * (
        pullback[1, 0] - pullback[0, 1]
    )
    b = 0.5 * (pullback[0, 0] - pullback[1, 1]) + 0.5j * (
        pullback[1, 0] + pullback[0, 1]
    )
    w_map = {(1, 0): complex(a), (0, 1): complex(b)}
    wbar_map = {(0, 1): complex(np.conjugate(a)), (1, 0): complex(np.conjugate(b))}
    polynomial = _poly_multiply(_poly_power(w_map, r), _poly_power(wbar_map, s))
    if not antiunitary:
        return polynomial
    conjugated: dict[tuple[int, int], complex] = {}
    for (out_r, out_s), value in polynomial.items():
        key = (out_s, out_r)
        conjugated[key] = conjugated.get(key, 0.0j) + np.conjugate(value)
    return {key: value for key, value in conjugated.items() if value != 0.0j}


def _logical_resolution(
    logical_channels: Sequence[Any],
    physical_channel_ids: Sequence[str],
    physical_channel_aliases: Mapping[str, Any] | None,
) -> tuple[dict[tuple[str, str], tuple[int, int] | None], dict[str, Any]]:
    index_by_id = {str(channel_id): index for index, channel_id in enumerate(physical_channel_ids)}
    if len(index_by_id) != len(tuple(physical_channel_ids)):
        raise FactorizedTermActionError("physical channel ids must be unique")
    aliases: dict[str, tuple[str, int]] = {}
    for key, value in (physical_channel_aliases or {}).items():
        if isinstance(value, (tuple, list)) and len(value) == 2:
            representative, sign = str(value[0]), int(value[1])
        else:
            representative, sign = str(value), 1
        if sign not in {-1, 1}:
            raise FactorizedTermActionError(
                f"physical channel alias {key!r} has invalid sign {sign}"
            )
        aliases[str(key)] = (representative, sign)
    logical_by_id = {str(channel.channel_id): channel for channel in logical_channels}
    resolution: dict[tuple[str, str], tuple[int, int] | None] = {}
    for channel in logical_channels:
        key = (str(channel.seed_id), str(channel.component))
        if str(channel.classification) == "structural_zero":
            resolution[key] = None
            continue
        representative = str(
            channel.metadata.get("adjoint_representative_channel_id", channel.channel_id)
        )
        representative, alias_sign = aliases.get(representative, (representative, 1))
        if representative not in index_by_id:
            raise FactorizedTermActionError(
                f"logical channel {channel.channel_id!r} resolves to missing physical "
                f"channel {representative!r}"
            )
        resolution[key] = (
            int(index_by_id[representative]),
            int(channel.metadata.get("adjoint_sign", 1)) * int(alias_sign),
        )
    return resolution, logical_by_id


def compile_factorized_real_channel_action(
    *,
    seeds: Sequence[Any],
    logical_channels: Sequence[Any],
    physical_channel_ids: Sequence[str],
    factorized_action: Any,
    physical_channel_aliases: Mapping[str, Any] | None = None,
) -> sparse.csc_matrix:
    """Compile ``g(V) = V A_g`` directly in the Hermitian real channel basis."""

    blocks = tuple(np.asarray(value, dtype=np.complex128) for value in factorized_action.orbital_blocks)
    sector_permutation = tuple(int(value) for value in factorized_action.sector_permutation)
    n_orb = tuple(int(value) for value in factorized_action.n_orb)
    q_counts = tuple(int(value) for value in factorized_action.q_counts)
    ordered = tuple(
        _OrderedSeed.from_raw(seed, q_counts=q_counts, n_orb=n_orb)
        for seed in seeds
    )
    by_seed_id = {seed.seed_id: seed for seed in ordered}
    if len(by_seed_id) != len(ordered):
        raise FactorizedTermActionError("ordered seed ids must be unique")
    seed_by_key: dict[tuple[Any, ...], _OrderedSeed] = {}
    for seed in ordered:
        if not seed.q_pairs:
            continue
        if seed.lookup_key in seed_by_key:
            raise FactorizedTermActionError(
                f"ordered seed vocabulary has duplicate key {seed.lookup_key!r}"
            )
        seed_by_key[seed.lookup_key] = seed
    resolution, logical_by_id = _logical_resolution(
        logical_channels,
        physical_channel_ids,
        physical_channel_aliases,
    )
    physical_ids = tuple(str(value) for value in physical_channel_ids)
    max_degree = max((seed.r + seed.s for seed in ordered), default=0)
    block_scale = max(
        1.0,
        *(float(np.linalg.norm(block, ord=2)) for block in blocks),
    )
    k_condition = float(
        np.linalg.cond(np.asarray(factorized_action.k_forward, dtype=np.float64))
    )
    arithmetic_bound = float(
        128.0
        * np.finfo(np.float64).eps
        * max(1, max_degree + 1)
        * block_scale**2
        * max(1.0, k_condition)
    )
    certification_bound = float(
        max(
            getattr(factorized_action, "matrix_certification_bound", 0.0),
            getattr(factorized_action, "phase_certification_bound", 0.0),
        )
    )
    coefficient_cleanup_bound = max(arithmetic_bound, certification_bound)

    entries: dict[tuple[int, int], float] = {}
    for input_index, channel_id in enumerate(physical_ids):
        if channel_id not in logical_by_id:
            raise FactorizedTermActionError(
                f"physical channel {channel_id!r} is absent from logical channel metadata"
            )
        input_channel = logical_by_id[channel_id]
        input_seed = by_seed_id[str(input_channel.seed_id)]
        source_left = input_seed.layer_from - 1
        source_right = input_seed.layer_to - 1
        target_left = sector_permutation[source_left]
        target_right = sector_permutation[source_right]
        left_block = blocks[source_left]
        right_block = blocks[source_right]
        expected_left_shape = (n_orb[target_left], n_orb[source_left])
        expected_right_shape = (n_orb[target_right], n_orb[source_right])
        if left_block.shape != expected_left_shape:
            raise FactorizedTermActionError(
                f"factorized orbital block {source_left} has shape {left_block.shape}, "
                f"expected {expected_left_shape}"
            )
        if right_block.shape != expected_right_shape:
            raise FactorizedTermActionError(
                f"factorized orbital block {source_right} has shape {right_block.shape}, "
                f"expected {expected_right_shape}"
            )
        routed_q_pairs, q_phase_ratio = _routed_q_support_and_phase(
            input_seed,
            factorized_action=factorized_action,
            cleanup_bound=coefficient_cleanup_bound,
        )
        polynomial = _relative_monomial_image(
            input_seed.r,
            input_seed.s,
            k_forward=factorized_action.k_forward,
            antiunitary=bool(factorized_action.antiunitary),
        )
        ordered_image: dict[str, complex] = {}
        for (out_r, out_s), polynomial_value in polynomial.items():
            for orbital_from in range(left_block.shape[0]):
                left = complex(left_block[orbital_from, input_seed.orbital_from])
                if left == 0.0j:
                    continue
                for orbital_to in range(right_block.shape[0]):
                    orbital_value = left * np.conjugate(
                        right_block[orbital_to, input_seed.orbital_to]
                    )
                    coefficient = (
                        complex(q_phase_ratio)
                        * complex(polynomial_value)
                        * complex(orbital_value)
                    )
                    if abs(coefficient) <= coefficient_cleanup_bound:
                        continue
                    output_key = (
                        input_seed.support_component,
                        target_left + 1,
                        target_right + 1,
                        orbital_from,
                        orbital_to,
                        int(out_r),
                        int(out_s),
                        routed_q_pairs,
                    )
                    output_seed = seed_by_key.get(output_key)
                    if output_seed is None:
                        raise FactorizedTermActionError(
                            f"factorized action sends {input_seed.seed_id!r} outside the "
                            f"authored seed vocabulary at {output_key!r}"
                        )
                    ordered_image[output_seed.seed_id] = (
                        ordered_image.get(output_seed.seed_id, 0.0j) + coefficient
                    )

        if str(input_channel.component) == "real":
            component_phase = 1.0 + 0.0j
        elif str(input_channel.component) == "imag":
            component_phase = -1.0j if bool(factorized_action.antiunitary) else 1.0j
        else:
            raise FactorizedTermActionError(
                f"unsupported physical channel component {input_channel.component!r}"
            )
        for output_seed_id, raw_coefficient in ordered_image.items():
            coefficient = component_phase * raw_coefficient
            for component, real_value in (
                ("real", float(np.real(coefficient))),
                ("imag", float(np.imag(coefficient))),
            ):
                if abs(real_value) <= coefficient_cleanup_bound:
                    continue
                target = resolution[(output_seed_id, component)]
                if target is None:
                    # Hermitian projection annihilates the imaginary channel of
                    # a self-adjoint ordered seed.  Dense orbital blocks can
                    # legitimately produce such raw coefficients even though
                    # their projected response is exactly zero.
                    continue
                output_index, sign = target
                entry_key = (output_index, input_index)
                entries[entry_key] = entries.get(entry_key, 0.0) + float(sign) * real_value

    nonzero = [
        (row, column, value)
        for (row, column), value in entries.items()
        if value != 0.0
    ]
    size = len(physical_ids)
    if not nonzero:
        return sparse.csc_matrix((size, size), dtype=np.float64)
    rows, columns, values = zip(*nonzero)
    action = sparse.coo_matrix(
        (
            np.asarray(values, dtype=np.float64),
            (np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)),
        ),
        shape=(size, size),
    ).tocsc()
    action.sum_duplicates()
    action.eliminate_zeros()
    action.sort_indices()
    return action


@dataclass(frozen=True)
class _RawRealChannel:
    channel_id: str
    seed_id: str
    component: str
    classification: str = "confirmed_nonzero"
    metadata: Mapping[str, Any] = field(default_factory=dict)


def compile_factorized_raw_seed_action(
    *,
    seeds: Sequence[Any],
    factorized_action: Any,
) -> sparse.csc_matrix:
    """Compile a generator action on ordered raw seed real/imag directions."""

    logical_channels = tuple(
        _RawRealChannel(
            channel_id=f"{seed.seed_id}:{component}",
            seed_id=str(seed.seed_id),
            component=component,
        )
        for seed in seeds
        for component in ("real", "imag")
    )
    return compile_factorized_real_channel_action(
        seeds=seeds,
        logical_channels=logical_channels,
        physical_channel_ids=tuple(channel.channel_id for channel in logical_channels),
        factorized_action=factorized_action,
    )


__all__ = [
    "FactorizedTermActionError",
    "compile_factorized_group_element_actions",
    "compile_factorized_raw_seed_action",
    "compile_factorized_real_channel_action",
    "compile_joint_route_group_element_actions",
]

"""Target-independent canonicalization of joint Hermitian-adjoint seed orbits.

This module deliberately does not import the response compiler.  Integration should
happen before ``RawPolynomialSeed`` matrices are expanded: describe every logical
ordered seed with :class:`JointAdjointSeedKey`, certify closure with
``canonicalize_joint_adjoint_seeds``, and compile only each orbit representative.
The returned mappings retain the full logical candidate-channel accounting, including
the sign relating an adjoint seed's imaginary channel to its representative.

Suggested ``response_basis.py`` integration sequence:

1. Build keys from the complete logical term vocabulary.  Harmonic adjoints must come
   from explicit action metadata; do not infer them by parsing operation or term names.
2. Canonicalize before ``raw_polynomial_seed_from_term_key`` and expand only the
   representative keys and their returned real/imag channel ids.
3. Before dropping a partner, independently certify that its raw polynomial
   coefficients equal the representative's coefficient-space adjoint within the
   propagated algebra error bound.  Metadata closure alone is not that numeric proof.
4. Preserve every logical candidate in the candidate artifact through ``mappings``;
   only the compiled work and retained basis generators are deduplicated.

No key, orbit, or mapping contains Heff, fit indices, regularization, or a band window.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse


class AdjointClosureError(ValueError):
    """Raised when logical seed support is not closed under Hermitian adjoint."""


class AdjointCertificationError(ValueError):
    """Raised when metadata adjoints disagree with raw polynomial coefficients."""


CANONICAL_CENTER_CONVENTION_V1 = "dimensionless_global_polynomial__little_endian_f64_v1"
ADJOINT_CERTIFICATION_STATUS_PROVISIONAL = "provisional_metadata_only"
ADJOINT_CERTIFICATION_STATUS_CONFIRMED = "certified_raw_coefficient_adjoint_v1"


@dataclass(frozen=True)
class CanonicalDimensionlessCenter:
    """Serialized identity for a center in the global dimensionless coordinates."""

    serialized_little_endian_hex: str
    coordinate_convention: str = CANONICAL_CENTER_CONVENTION_V1

    def __post_init__(self) -> None:
        if self.coordinate_convention != CANONICAL_CENTER_CONVENTION_V1:
            raise ValueError(
                f"unsupported center coordinate convention {self.coordinate_convention!r}"
            )
        serialized = self.serialized_little_endian_hex
        if not isinstance(serialized, str):
            raise TypeError("center identity must be canonical little-endian float64 hex")
        try:
            payload = bytes.fromhex(serialized)
        except ValueError as exc:
            raise ValueError(
                "center identity must be canonical little-endian float64 hex"
            ) from exc
        if len(payload) != 16 or payload.hex() != serialized:
            raise ValueError(
                "center identity must be canonical little-endian float64 hex"
            )
        values = np.frombuffer(payload, dtype="<f8")
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError("canonical dimensionless center must contain two finite values")

    @classmethod
    def from_dimensionless(
        cls,
        values: Sequence[float],
    ) -> "CanonicalDimensionlessCenter":
        array = np.asarray(values, dtype="<f8")
        if array.shape != (2,) or not np.all(np.isfinite(array)):
            raise ValueError("dimensionless center must contain two finite coordinates")
        return cls(np.ascontiguousarray(array, dtype="<f8").tobytes().hex())

    @property
    def values(self) -> tuple[float, float]:
        values = np.frombuffer(
            bytes.fromhex(self.serialized_little_endian_hex), dtype="<f8"
        )
        return float(values[0]), float(values[1])


@dataclass(frozen=True)
class JointAdjointSeedKey:
    sector_from: str
    sector_to: str
    orbital_from: int
    orbital_to: int
    monomial: tuple[int, int]
    center: CanonicalDimensionlessCenter
    harmonic_id: str
    adjoint_harmonic_id: str

    def __post_init__(self) -> None:
        if not self.sector_from or not self.sector_to:
            raise ValueError("joint adjoint sectors must be non-empty")
        if int(self.orbital_from) < 0 or int(self.orbital_to) < 0:
            raise ValueError("joint adjoint orbital indices must be non-negative")
        if len(self.monomial) != 2 or any(int(value) < 0 for value in self.monomial):
            raise ValueError("joint adjoint monomial must contain two non-negative exponents")
        if not isinstance(self.center, CanonicalDimensionlessCenter):
            raise TypeError(
                "joint adjoint center must be a CanonicalDimensionlessCenter; "
                "raw float centers are not stable seed identities"
            )
        if not self.harmonic_id or not self.adjoint_harmonic_id:
            raise ValueError("harmonic and adjoint-harmonic ids must be explicit and non-empty")

    def adjoint(self) -> "JointAdjointSeedKey":
        r, s = self.monomial
        return JointAdjointSeedKey(
            sector_from=self.sector_to,
            sector_to=self.sector_from,
            orbital_from=int(self.orbital_to),
            orbital_to=int(self.orbital_from),
            monomial=(int(s), int(r)),
            center=self.center,
            harmonic_id=self.adjoint_harmonic_id,
            adjoint_harmonic_id=self.harmonic_id,
        )

    def canonical_sort_key(self) -> tuple[object, ...]:
        return (
            self.sector_from,
            self.sector_to,
            int(self.orbital_from),
            int(self.orbital_to),
            int(self.monomial[0]),
            int(self.monomial[1]),
            self.center.coordinate_convention,
            self.center.serialized_little_endian_hex,
            self.harmonic_id,
            self.adjoint_harmonic_id,
        )


@dataclass(frozen=True)
class JointAdjointSeed:
    seed_id: str
    key: JointAdjointSeedKey

    def __post_init__(self) -> None:
        if not self.seed_id:
            raise ValueError("joint adjoint seed requires a non-empty seed id")


@dataclass(frozen=True)
class JointAdjointOrbit:
    representative_seed_id: str
    representative_key: JointAdjointSeedKey
    member_seed_ids: tuple[str, ...]
    self_adjoint: bool
    channel_ids: tuple[str, ...]
    certification_status: str = ADJOINT_CERTIFICATION_STATUS_PROVISIONAL
    requires_numeric_certification: bool = True
    raw_adjoint_residual: float | None = None
    certification_bound: float | None = None


@dataclass(frozen=True)
class AdjointChannelMapping:
    seed_id: str
    component: str
    channel_id: str | None
    sign: int
    structural_zero: bool
    certification_status: str = ADJOINT_CERTIFICATION_STATUS_PROVISIONAL
    requires_numeric_certification: bool = True


@dataclass(frozen=True)
class JointAdjointCanonicalization:
    orbits: tuple[JointAdjointOrbit, ...]
    mappings: tuple[AdjointChannelMapping, ...]

    @property
    def representative_seed_ids(self) -> tuple[str, ...]:
        return tuple(orbit.representative_seed_id for orbit in self.orbits)

    @property
    def channel_ids(self) -> tuple[str, ...]:
        return tuple(channel_id for orbit in self.orbits for channel_id in orbit.channel_ids)

    def mapping_for(self, seed_id: str, component: str) -> AdjointChannelMapping:
        for mapping in self.mappings:
            if mapping.seed_id == str(seed_id) and mapping.component == str(component):
                return mapping
        raise KeyError(f"no joint-adjoint mapping for {seed_id!r}:{component}")


def canonicalize_joint_adjoint_seeds(
    seeds: Sequence[JointAdjointSeed],
) -> JointAdjointCanonicalization:
    """Certify adjoint closure and choose one deterministic seed per joint orbit."""

    by_id: dict[str, JointAdjointSeed] = {}
    by_key: dict[JointAdjointSeedKey, JointAdjointSeed] = {}
    for seed in seeds:
        if seed.seed_id in by_id:
            raise ValueError(f"duplicate joint adjoint seed id {seed.seed_id!r}")
        if seed.key in by_key:
            raise ValueError(
                f"duplicate joint adjoint key for {seed.seed_id!r} and {by_key[seed.key].seed_id!r}"
            )
        by_id[seed.seed_id] = seed
        by_key[seed.key] = seed

    visited: set[JointAdjointSeedKey] = set()
    orbit_records: list[JointAdjointOrbit] = []
    mapping_records: list[AdjointChannelMapping] = []
    ordered_seeds = sorted(seeds, key=lambda seed: seed.key.canonical_sort_key())
    for seed in ordered_seeds:
        if seed.key in visited:
            continue
        adjoint_key = seed.key.adjoint()
        self_adjoint = adjoint_key == seed.key
        if not self_adjoint and adjoint_key not in by_key:
            raise AdjointClosureError(
                f"missing adjoint partner for seed {seed.seed_id!r}; "
                "orbital, sector, monomial, center, and explicit harmonic metadata must all close"
            )
        representative_key = min(
            (seed.key, adjoint_key),
            key=lambda key: key.canonical_sort_key(),
        )
        representative = by_key[representative_key]
        real_channel_id = f"{representative.seed_id}:real"
        if self_adjoint:
            imag_channel_id = f"{representative.seed_id}:imag"
            orbit_records.append(
                JointAdjointOrbit(
                    representative_seed_id=representative.seed_id,
                    representative_key=representative.key,
                    member_seed_ids=(representative.seed_id,),
                    self_adjoint=True,
                    channel_ids=(real_channel_id, imag_channel_id),
                )
            )
            mapping_records.extend(
                (
                    AdjointChannelMapping(
                        seed_id=representative.seed_id,
                        component="real",
                        channel_id=real_channel_id,
                        sign=1,
                        structural_zero=False,
                    ),
                    AdjointChannelMapping(
                        seed_id=representative.seed_id,
                        component="imag",
                        channel_id=imag_channel_id,
                        sign=1,
                        structural_zero=False,
                    ),
                )
            )
            visited.add(representative.key)
            continue

        partner = by_key[representative.key.adjoint()]
        imag_channel_id = f"{representative.seed_id}:imag"
        orbit_records.append(
            JointAdjointOrbit(
                representative_seed_id=representative.seed_id,
                representative_key=representative.key,
                member_seed_ids=(representative.seed_id, partner.seed_id),
                self_adjoint=False,
                channel_ids=(real_channel_id, imag_channel_id),
            )
        )
        for member, orientation in ((representative, 1), (partner, -1)):
            mapping_records.extend(
                (
                    AdjointChannelMapping(
                        seed_id=member.seed_id,
                        component="real",
                        channel_id=real_channel_id,
                        sign=1,
                        structural_zero=False,
                    ),
                    AdjointChannelMapping(
                        seed_id=member.seed_id,
                        component="imag",
                        channel_id=imag_channel_id,
                        sign=orientation,
                        structural_zero=False,
                    ),
                )
            )
        visited.update((representative.key, partner.key))

    return JointAdjointCanonicalization(
        orbits=tuple(orbit_records),
        mappings=tuple(mapping_records),
    )


def _validated_coefficient_map(
    coefficients: Mapping[tuple[int, int], np.ndarray | sparse.spmatrix],
) -> dict[tuple[int, int], np.ndarray | sparse.csr_matrix]:
    validated: dict[tuple[int, int], np.ndarray | sparse.csr_matrix] = {}
    shape: tuple[int, int] | None = None
    for raw_key, raw_matrix in coefficients.items():
        if len(raw_key) != 2:
            raise ValueError("polynomial coefficient key must contain two exponents")
        key = int(raw_key[0]), int(raw_key[1])
        if min(key) < 0:
            raise ValueError("polynomial exponents must be non-negative")
        matrix = (
            sparse.csr_matrix(raw_matrix, dtype=np.complex128)
            if sparse.issparse(raw_matrix)
            else np.asarray(raw_matrix, dtype=np.complex128)
        )
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"polynomial coefficient must be square, got {matrix.shape}")
        if shape is None:
            shape = matrix.shape
        elif matrix.shape != shape:
            raise ValueError("polynomial coefficient matrices must have a common shape")
        finite_values = matrix.data if sparse.issparse(matrix) else matrix
        if not np.all(np.isfinite(finite_values)):
            raise ValueError("polynomial coefficient contains non-finite values")
        validated[key] = (
            matrix.copy().tocsr()
            if sparse.issparse(matrix)
            else np.array(matrix, dtype=np.complex128, copy=True)
        )
    return validated


def _coefficient_adjoint(
    coefficients: Mapping[tuple[int, int], np.ndarray | sparse.spmatrix],
) -> dict[tuple[int, int], np.ndarray | sparse.spmatrix]:
    return {
        (int(s), int(r)): (
            sparse.csr_matrix(matrix, dtype=np.complex128).getH().tocsr()
            if sparse.issparse(matrix)
            else np.asarray(matrix, dtype=np.complex128).conj().T
        )
        for (r, s), matrix in coefficients.items()
    }


def _matrix_frobenius_norm(matrix: np.ndarray | sparse.spmatrix) -> float:
    if sparse.issparse(matrix):
        values = sparse.csr_matrix(matrix, dtype=np.complex128).data
        return float(np.sqrt(np.vdot(values, values).real))
    return float(np.linalg.norm(np.asarray(matrix, dtype=np.complex128), ord="fro"))


def _coefficient_norm(
    coefficients: Mapping[tuple[int, int], np.ndarray | sparse.spmatrix],
) -> float:
    return float(
        np.sqrt(
            sum(
                _matrix_frobenius_norm(matrix) ** 2
                for matrix in coefficients.values()
            )
        )
    )


def _coefficient_residual(
    actual: Mapping[tuple[int, int], np.ndarray | sparse.spmatrix],
    expected: Mapping[tuple[int, int], np.ndarray | sparse.spmatrix],
) -> float:
    keys = set(actual) | set(expected)
    shape = next(
        (matrix.shape for matrix in (*actual.values(), *expected.values())),
        None,
    )
    if shape is None:
        return 0.0
    use_sparse = any(sparse.issparse(value) for value in (*actual.values(), *expected.values()))
    zero = (
        sparse.csr_matrix(shape, dtype=np.complex128)
        if use_sparse
        else np.zeros(shape, dtype=np.complex128)
    )
    return float(
        np.sqrt(
            sum(
                float(
                    _matrix_frobenius_norm(actual.get(key, zero) - expected.get(key, zero))
                )
                ** 2
                for key in keys
            )
        )
    )


def _coefficient_adjoint_residual(
    actual: Mapping[tuple[int, int], np.ndarray | sparse.spmatrix],
    representative: Mapping[tuple[int, int], np.ndarray | sparse.spmatrix],
) -> float:
    """Return ``||actual - representative^dagger||`` without copying a map."""

    keys = set(actual)
    keys.update((int(s), int(r)) for r, s in representative)
    shape = next(
        (matrix.shape for matrix in (*actual.values(), *representative.values())),
        None,
    )
    if shape is None:
        return 0.0
    use_sparse = any(
        sparse.issparse(value)
        for value in (*actual.values(), *representative.values())
    )
    if use_sparse:
        zero = sparse.csr_matrix(shape, dtype=np.complex128)
        total = 0.0
        for r_value, s_value in keys:
            direct = sparse.csr_matrix(
                actual.get((r_value, s_value), zero),
                dtype=np.complex128,
            )
            source = sparse.csr_matrix(
                representative.get((s_value, r_value), zero),
                dtype=np.complex128,
            )
            total += _matrix_frobenius_norm(direct - source.getH()) ** 2
        return float(np.sqrt(total))
    zero_dense = np.zeros(shape, dtype=np.complex128)
    return float(
        np.sqrt(
            sum(
                _matrix_frobenius_norm(
                    np.asarray(
                        actual.get((r_value, s_value), zero_dense),
                        dtype=np.complex128,
                    )
                    - np.asarray(
                        representative.get((s_value, r_value), zero_dense),
                        dtype=np.complex128,
                    ).conj().T
                )
                ** 2
                for r_value, s_value in keys
            )
        )
    )


def certify_joint_adjoint_coefficients(
    canonicalization: JointAdjointCanonicalization,
    coefficients_by_seed: Mapping[
        str,
        Mapping[tuple[int, int], np.ndarray | sparse.spmatrix],
    ],
    *,
    absolute_error_bound: float | None = None,
    absolute_error_bounds_by_seed: Mapping[str, float] | None = None,
    relative_tolerance: float = 32.0 * np.finfo(np.float64).eps,
) -> JointAdjointCanonicalization:
    """Certify metadata adjoint orbits against global raw coefficient tensors.

    Metadata closure alone is provisional.  Only this coefficient-space check may
    drop a partner or classify a self-adjoint imaginary channel as structural zero.
    """

    if (absolute_error_bound is None) == (absolute_error_bounds_by_seed is None):
        raise ValueError(
            "raw adjoint certification requires exactly one absolute error-bound mode"
        )
    if relative_tolerance < 0.0:
        raise ValueError("adjoint certification tolerances must be non-negative")
    if absolute_error_bound is not None and absolute_error_bound < 0.0:
        raise ValueError("adjoint certification tolerances must be non-negative")
    raw_coefficients = {
        str(seed_id): coefficients
        for seed_id, coefficients in coefficients_by_seed.items()
    }
    required_seed_ids = {
        seed_id
        for orbit in canonicalization.orbits
        for seed_id in orbit.member_seed_ids
    }
    if absolute_error_bounds_by_seed is not None:
        per_seed_bounds = {
            str(seed_id): float(value)
            for seed_id, value in absolute_error_bounds_by_seed.items()
        }
        if set(per_seed_bounds) != required_seed_ids:
            raise ValueError(
                "per-seed raw adjoint error bounds must cover every orbit member exactly"
            )
        if any(
            not np.isfinite(value) or value < 0.0
            for value in per_seed_bounds.values()
        ):
            raise ValueError("adjoint certification tolerances must be finite and non-negative")
    else:
        per_seed_bounds = None
    certified_orbits: list[JointAdjointOrbit] = []
    certified_mappings: list[AdjointChannelMapping] = []
    representative_by_seed_id = {
        seed_id: orbit.representative_seed_id
        for orbit in canonicalization.orbits
        for seed_id in orbit.member_seed_ids
    }
    mappings_by_orbit = {
        orbit.representative_seed_id: []
        for orbit in canonicalization.orbits
    }
    for mapping in canonicalization.mappings:
        mappings_by_orbit[representative_by_seed_id[mapping.seed_id]].append(
            mapping
        )
    for orbit in canonicalization.orbits:
        missing = [
            seed_id
            for seed_id in orbit.member_seed_ids
            if seed_id not in raw_coefficients
        ]
        if missing:
            raise AdjointCertificationError(
                f"missing raw polynomial coefficients for adjoint orbit members {missing}"
            )
        validated = {
            seed_id: _validated_coefficient_map(raw_coefficients[seed_id])
            for seed_id in orbit.member_seed_ids
        }
        representative = validated[orbit.representative_seed_id]
        if orbit.self_adjoint:
            actual = representative
        else:
            partner_id = next(
                seed_id
                for seed_id in orbit.member_seed_ids
                if seed_id != orbit.representative_seed_id
            )
            actual = validated[partner_id]
        scale = max(_coefficient_norm(actual), _coefficient_norm(representative))
        orbit_absolute_bound = (
            float(absolute_error_bound)
            if per_seed_bounds is None
            else float(
                sum(per_seed_bounds[seed_id] for seed_id in orbit.member_seed_ids)
            )
        )
        bound = float(orbit_absolute_bound + relative_tolerance * scale)
        residual = _coefficient_adjoint_residual(actual, representative)
        if residual > bound:
            raise AdjointCertificationError(
                f"raw coefficient adjoint residual {residual:.6e} for orbit "
                f"{orbit.representative_seed_id!r} exceeds certification bound {bound:.6e}"
            )
        channel_ids = orbit.channel_ids[:1] if orbit.self_adjoint else orbit.channel_ids
        certified_orbits.append(
            replace(
                orbit,
                channel_ids=channel_ids,
                certification_status=ADJOINT_CERTIFICATION_STATUS_CONFIRMED,
                requires_numeric_certification=False,
                raw_adjoint_residual=residual,
                certification_bound=bound,
            )
        )
        for mapping in mappings_by_orbit[orbit.representative_seed_id]:
            if orbit.self_adjoint and mapping.component == "imag":
                mapping = replace(
                    mapping,
                    channel_id=None,
                    sign=0,
                    structural_zero=True,
                )
            certified_mappings.append(
                replace(
                    mapping,
                    certification_status=ADJOINT_CERTIFICATION_STATUS_CONFIRMED,
                    requires_numeric_certification=False,
                )
            )
    return JointAdjointCanonicalization(
        orbits=tuple(certified_orbits),
        mappings=tuple(certified_mappings),
    )


def hermitian_channel_coefficients(
    coefficients: Mapping[tuple[int, int], np.ndarray],
    *,
    component: str,
) -> dict[tuple[int, int], np.ndarray]:
    """Apply ``(X+X†)/2`` to either a real or ``i X`` polynomial seed."""

    if component not in {"real", "imag"}:
        raise ValueError("Hermitian channel component must be 'real' or 'imag'")
    out: dict[tuple[int, int], np.ndarray] = {}
    direct_factor = 0.5 if component == "real" else 0.5j
    adjoint_factor = 0.5 if component == "real" else -0.5j
    shape: tuple[int, int] | None = None
    for (r_value, s_value), value in coefficients.items():
        r = int(r_value)
        s = int(s_value)
        if r < 0 or s < 0:
            raise ValueError("polynomial exponents must be non-negative")
        matrix = np.asarray(value, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"polynomial coefficient must be square, got {matrix.shape}")
        if shape is None:
            shape = matrix.shape
        elif matrix.shape != shape:
            raise ValueError("polynomial coefficient matrices must have a common shape")
        direct_key = (r, s)
        adjoint_key = (s, r)
        if direct_key not in out:
            out[direct_key] = np.zeros_like(matrix)
        if adjoint_key not in out:
            out[adjoint_key] = np.zeros_like(matrix)
        out[direct_key] += direct_factor * matrix
        out[adjoint_key] += adjoint_factor * matrix.conj().T
    return out

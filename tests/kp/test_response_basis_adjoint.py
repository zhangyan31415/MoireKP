from __future__ import annotations

import numpy as np
import pytest

from kp.model.response_basis_adjoint import (
    AdjointCertificationError,
    AdjointClosureError,
    CanonicalDimensionlessCenter,
    JointAdjointSeed,
    JointAdjointSeedKey,
    canonicalize_joint_adjoint_seeds,
    certify_joint_adjoint_coefficients,
    hermitian_channel_coefficients,
)


def _key(
    *,
    sector_from: str = "layer-1",
    sector_to: str = "layer-1",
    orbital_from: int = 0,
    orbital_to: int = 1,
    monomial: tuple[int, int] = (0, 0),
    center: tuple[float, float] = (0.0, 0.0),
    harmonic_id: str = "zero",
    adjoint_harmonic_id: str = "zero",
) -> JointAdjointSeedKey:
    return JointAdjointSeedKey(
        sector_from=sector_from,
        sector_to=sector_to,
        orbital_from=orbital_from,
        orbital_to=orbital_to,
        monomial=monomial,
        center=CanonicalDimensionlessCenter.from_dimensionless(center),
        harmonic_id=harmonic_id,
        adjoint_harmonic_id=adjoint_harmonic_id,
    )


def test_joint_adjoint_swaps_orbitals_sectors_and_monomial_but_preserves_center() -> None:
    key = _key(
        sector_from="A",
        sector_to="B",
        orbital_from=2,
        orbital_to=5,
        monomial=(3, 1),
        center=(0.25, -0.5),
        harmonic_id="plus-g1",
        adjoint_harmonic_id="minus-g1",
    )

    adjoint = key.adjoint()

    assert adjoint.sector_from == "B"
    assert adjoint.sector_to == "A"
    assert adjoint.orbital_from == 5
    assert adjoint.orbital_to == 2
    assert adjoint.monomial == (1, 3)
    assert adjoint.center.values == (0.25, -0.5)
    assert adjoint.harmonic_id == "minus-g1"
    assert adjoint.adjoint_harmonic_id == "plus-g1"
    assert adjoint.adjoint() == key


def test_e12_orbit_produces_exactly_two_hermitian_real_channels() -> None:
    e12_key = _key(orbital_from=0, orbital_to=1)
    result = canonicalize_joint_adjoint_seeds(
        [
            JointAdjointSeed("E12", e12_key),
            JointAdjointSeed("E21", e12_key.adjoint()),
        ]
    )

    assert len(result.orbits) == 1
    assert result.representative_seed_ids == ("E12",)
    assert result.channel_ids == ("E12:real", "E12:imag")
    assert result.mapping_for("E12", "real").sign == 1
    assert result.mapping_for("E12", "imag").sign == 1
    assert result.mapping_for("E21", "real").sign == 1
    assert result.mapping_for("E21", "imag").sign == -1

    e12 = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    real = hermitian_channel_coefficients({(0, 0): e12}, component="real")[(0, 0)]
    imag = hermitian_channel_coefficients({(0, 0): e12}, component="imag")[(0, 0)]
    np.testing.assert_allclose(real, [[0.0, 0.5], [0.5, 0.0]])
    np.testing.assert_allclose(imag, [[0.0, 0.5j], [-0.5j, 0.0]])
    assert np.linalg.matrix_rank(
        np.column_stack(
            [
                np.concatenate((real.real.ravel(), real.imag.ravel())),
                np.concatenate((imag.real.ravel(), imag.imag.ravel())),
            ]
        )
    ) == 2


def test_nonself_orbit_uses_explicit_harmonic_adjoint_and_opposite_imag_signs() -> None:
    forward = _key(
        sector_from="A",
        sector_to="B",
        orbital_from=1,
        orbital_to=3,
        monomial=(2, 0),
        center=(0.125, 0.25),
        harmonic_id="g",
        adjoint_harmonic_id="minus-g",
    )
    result = canonicalize_joint_adjoint_seeds(
        [
            JointAdjointSeed("forward", forward),
            JointAdjointSeed("backward", forward.adjoint()),
        ]
    )

    real_signs = {
        result.mapping_for("forward", "real").sign,
        result.mapping_for("backward", "real").sign,
    }
    imag_signs = {
        result.mapping_for("forward", "imag").sign,
        result.mapping_for("backward", "imag").sign,
    }
    assert real_signs == {1}
    assert imag_signs == {-1, 1}
    assert len(result.channel_ids) == 2


def test_self_adjoint_seed_keeps_real_and_marks_imag_structural_zero() -> None:
    diagonal = _key(
        orbital_from=2,
        orbital_to=2,
        monomial=(1, 1),
        harmonic_id="zero",
        adjoint_harmonic_id="zero",
    )

    provisional = canonicalize_joint_adjoint_seeds(
        [JointAdjointSeed("E33_abs_w2", diagonal)]
    )

    provisional_orbit = provisional.orbits[0]
    provisional_imag = provisional.mapping_for("E33_abs_w2", "imag")
    assert provisional_orbit.certification_status == "provisional_metadata_only"
    assert provisional_orbit.requires_numeric_certification is True
    assert provisional.channel_ids == (
        "E33_abs_w2:real",
        "E33_abs_w2:imag",
    )
    assert provisional_imag.structural_zero is False

    diagonal_matrix = np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=np.complex128)
    result = certify_joint_adjoint_coefficients(
        provisional,
        {"E33_abs_w2": {(1, 1): diagonal_matrix}},
        absolute_error_bound=1.0e-14,
    )

    assert result.representative_seed_ids == ("E33_abs_w2",)
    assert result.channel_ids == ("E33_abs_w2:real",)
    real_mapping = result.mapping_for("E33_abs_w2", "real")
    imag_mapping = result.mapping_for("E33_abs_w2", "imag")
    assert real_mapping.channel_id == "E33_abs_w2:real"
    assert real_mapping.structural_zero is False
    assert imag_mapping.channel_id is None
    assert imag_mapping.sign == 0
    assert imag_mapping.structural_zero is True
    assert result.orbits[0].requires_numeric_certification is False
    assert result.orbits[0].raw_adjoint_residual == pytest.approx(0.0)


def test_metadata_only_nonself_orbit_requires_raw_coefficient_certification() -> None:
    e12_key = _key(orbital_from=0, orbital_to=1)
    provisional = canonicalize_joint_adjoint_seeds(
        [
            JointAdjointSeed("E12", e12_key),
            JointAdjointSeed("E21", e12_key.adjoint()),
        ]
    )

    assert provisional.orbits[0].requires_numeric_certification is True
    assert all(mapping.structural_zero is False for mapping in provisional.mappings)

    e12 = np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    e21_wrong = np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.complex128)
    with pytest.raises(AdjointCertificationError, match="raw coefficient adjoint residual"):
        certify_joint_adjoint_coefficients(
            provisional,
            {"E12": {(0, 0): e12}, "E21": {(0, 0): e21_wrong}},
            absolute_error_bound=1.0e-14,
        )


def test_raw_adjoint_certification_sums_per_seed_bounds_within_each_orbit() -> None:
    forward_key = _key(orbital_from=0, orbital_to=1)
    provisional = canonicalize_joint_adjoint_seeds(
        [
            JointAdjointSeed("forward", forward_key),
            JointAdjointSeed("backward", forward_key.adjoint()),
        ]
    )
    forward = np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    backward = np.asarray([[0.0, 0.0], [1.09, 0.0]], dtype=np.complex128)

    certified = certify_joint_adjoint_coefficients(
        provisional,
        {
            "forward": {(0, 0): forward},
            "backward": {(0, 0): backward},
        },
        absolute_error_bounds_by_seed={
            "forward": 0.0455,
            "backward": 0.0455,
        },
        relative_tolerance=0.0,
    )

    assert certified.orbits[0].raw_adjoint_residual == pytest.approx(0.09)
    assert certified.orbits[0].certification_bound == pytest.approx(0.091)


def test_center_identity_requires_canonical_dimensionless_serialization() -> None:
    with pytest.raises(TypeError, match="CanonicalDimensionlessCenter"):
        JointAdjointSeedKey(
            sector_from="A",
            sector_to="A",
            orbital_from=0,
            orbital_to=0,
            monomial=(0, 0),
            center=(0.125, -0.25),  # type: ignore[arg-type]
            harmonic_id="zero",
            adjoint_harmonic_id="zero",
        )

    center = CanonicalDimensionlessCenter.from_dimensionless((0.125, -0.25))
    assert center.values == (0.125, -0.25)
    assert len(center.serialized_little_endian_hex) == 32
    with pytest.raises(ValueError, match="canonical little-endian float64"):
        CanonicalDimensionlessCenter("0.125,-0.25")


def test_nonself_seed_without_adjoint_partner_fails_closed() -> None:
    with pytest.raises(AdjointClosureError, match="missing adjoint partner"):
        canonicalize_joint_adjoint_seeds([JointAdjointSeed("E12", _key())])


def test_same_orbital_with_unswapped_monomial_is_not_self_adjoint() -> None:
    key = _key(orbital_from=0, orbital_to=0, monomial=(2, 1))

    with pytest.raises(AdjointClosureError, match="missing adjoint partner"):
        canonicalize_joint_adjoint_seeds([JointAdjointSeed("E11_w2_wbar", key)])

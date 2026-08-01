from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from kp.blocks import (
    GammaRoutingError,
    GammaRoutingThresholds,
    GammaRowLayout,
    assemble_gamma_routed_projectors,
    build_gamma_routed_frames,
)
from kp.projection_handoff import (
    ExplicitLegacyBasisSpec,
    GammaRoutedBasisSpec,
    certify_gamma_routed_basis_spec,
    load_gamma_routed_basis_spec,
    save_gamma_routed_basis_spec,
)
from kp.projection_selection import CandidateRejectionReason
from kp.identity import PROJECTION_BASIS_HANDOFF_VERSION, hash_array
from kp.symmetry import projection as projection_mod
from kp.symmetry.candidate_certificate import (
    CandidateOperationInput,
    CandidateSymmetryThresholds,
    verify_candidate_certificate_envelope,
)
from kp.symmetry.joint_exactification import (
    MagneticGenerator,
    MagneticPresentation,
    MagneticRelation,
)


def _thresholds() -> GammaRoutingThresholds:
    return GammaRoutingThresholds(
        energy_same_ev=1.0e-8,
        energy_different_ev=1.0e-5,
        capture_zero_fraction=1.0e-8,
        capture_loss_max=1.0e-8,
        local_action_isometry=1.0e-10,
        off_route_leakage=1.0e-10,
        closure_residual=1.0e-10,
        route_zero_gap=1.0e-8,
        route_covariance=1.0e-10,
        projector_residual=1.0e-10,
        anchor_sigma_min=1.0e-8,
        max_rank=4,
        max_iterations=8,
    )


def _layout() -> GammaRowLayout:
    return GammaRowLayout.build(
        qsets=(np.zeros((1, 2)), np.zeros((1, 2))),
        num_layer_list=(1, 1),
        num_orb_per_layer_list=((1,), (1,)),
        spin_convention="all",
        source_basis_hash="source-basis-a",
    )


def _uncertified_spec() -> GammaRoutedBasisSpec:
    layout = _layout()
    routed = build_gamma_routed_frames(
        (np.arange(4, dtype=float),),
        (np.eye(4, dtype=np.complex128),),
        joint_band_indices=(0, 1),
        layout=layout,
        thresholds=_thresholds(),
        require_complete_clusters=False,
    )
    heff = np.asarray(
        [np.diag([0.25, 0.75]), np.diag([0.5, 1.0])],
        dtype=np.complex128,
    )
    return GammaRoutedBasisSpec.create(
        artifact_identity={
            "identity_schema": "moirekp.artifact-identity.v1",
            "input_hash": "1" * 64,
            "config_hash": "2" * 64,
            "basis_hash": "3" * 64,
            "package_version": "0.1.0",
            "schema_version": 2,
            "k_indices_hash": "replaced-by-writer",
            "heff_hash": "replaced-by-writer",
        },
        layout=layout,
        thresholds=_thresholds(),
        k_indices=(4, 7),
        routed_frames=(routed, routed),
        authoritative_heff=heff,
        heff_k_indices=(4, 7),
        closure_certificate_hashes=("4" * 64, "7" * 64),
        routing_certificate_hashes=("5" * 64, "8" * 64),
        source_hamiltonian_hash="a" * 64,
        ordered_q_hashes=layout.ordered_qset_hashes,
        raw_action_package_hash="0" * 64,
        candidate_certificate_hash="0" * 64,
        candidate_input_identity_hash="0" * 64,
    )


def _spec() -> GammaRoutedBasisSpec:
    base = _uncertified_spec()
    pairs = ((4, 4), (7, 7))
    presentation = MagneticPresentation(
        generators=(MagneticGenerator("E", False),),
        relations=(
            MagneticRelation("E^2", lhs=("E", "E"), rhs=(), central_phase=1.0),
        ),
        central_phases=(1.0,),
        source="gamma_handoff_test",
    )
    return certify_gamma_routed_basis_spec(
        candidate_id="gamma-routed-test",
        artifact_identity={
            **base.artifact_identity,
            "basis_hash": base.base_basis_hash,
        },
        layout=base.layout,
        thresholds=base.thresholds,
        k_indices=base.k_indices,
        routed_frames=tuple(base.routed_frames_for_k(k) for k in base.k_indices),
        authoritative_heff=base.authoritative_heff,
        heff_k_indices=base.heff_k_indices,
        closure_certificate_hashes=base.closure_certificate_hashes,
        routing_certificate_hashes=base.routing_certificate_hashes,
        source_hamiltonian_hash=base.source_hamiltonian_hash,
        ordered_q_hashes=base.ordered_q_hashes,
        raw_action_package_hash=base.raw_action_package_hash,
        operations={
            "E": CandidateOperationInput(
                name="E",
                antiunitary=False,
                d_full=np.eye(base.layout.full_dimension, dtype=np.complex128),
                pairs=pairs,
            )
        },
        exactified_actions={"E": np.eye(base.model_dim, dtype=np.complex128)},
        presentation=presentation,
        required_pairs={"E": pairs},
        candidate_thresholds=CandidateSymmetryThresholds.uniform(1.0e-10),
    )


def _rewrite_npz(
    path: Path,
    *,
    remove: tuple[str, ...] = (),
    **updates: np.ndarray,
) -> None:
    with np.load(path, allow_pickle=False) as payload:
        copied = {name: np.array(payload[name], copy=True) for name in payload.files}
    for name in remove:
        copied.pop(name)
    copied.update(updates)
    np.savez(path, **copied)


def _write_routed_project_artifacts(project_dir: Path, spec: GammaRoutedBasisSpec) -> None:
    project_dir.mkdir()
    save_gamma_routed_basis_spec(project_dir / "basis.npz", spec)
    np.save(project_dir / "heff.npy", spec.authoritative_heff)
    np.savez(
        project_dir / "wavefunctions.npz",
        wavefunctions=np.repeat(
            np.eye(spec.model_dim, dtype=np.complex128)[None, :, :],
            len(spec.k_indices),
            axis=0,
        ),
        k_indices=np.asarray(spec.k_indices, dtype=np.int64),
        **{
            name: np.asarray(value)
            for name, value in spec.artifact_identity.items()
        },
    )


def _write_legacy_project_artifacts(project_dir: Path) -> None:
    project_dir.mkdir()
    heff = np.eye(2, dtype=np.complex128)[None, :, :]
    wavefunctions = np.eye(2, dtype=np.complex128)[None, :, :]
    k_indices = np.asarray([0], dtype=np.int64)
    identity = {
        "identity_schema": "moirekp.artifact-identity.v1",
        "input_hash": "input-legacy",
        "config_hash": "config-legacy",
        "basis_hash": "basis-legacy",
        "package_version": "0.1.0",
        "schema_version": 2,
        "k_indices_hash": hash_array(k_indices),
        "heff_hash": hash_array(heff),
    }
    common = {
        "projection_basis_kind": np.asarray("explicit_legacy"),
        "projection_basis_handoff_version": np.asarray(
            PROJECTION_BASIS_HANDOFF_VERSION
        ),
        "gauge_mode": np.asarray("manual"),
        "symmetry_adapted_frame_json": np.asarray(""),
        "wavefunctions_hash": np.asarray(hash_array(wavefunctions)),
        "spin_operator_hash": np.asarray("absent"),
        **{name: np.asarray(value) for name, value in identity.items()},
    }
    np.save(project_dir / "heff.npy", heff)
    np.savez(
        project_dir / "basis.npz",
        nlow_state_list=np.asarray([[0], [1]], dtype=object),
        norb_fix_list=np.asarray([[[[0, 1.0]]], [[[1, 1.0]]]], dtype=object),
        **common,
    )
    np.savez(
        project_dir / "wavefunctions.npz",
        wavefunctions=wavefunctions,
        k_indices=k_indices,
        **common,
    )


def test_gamma_routed_handoff_roundtrip_is_numeric_and_assembles_identically(
    tmp_path: Path,
) -> None:
    path = tmp_path / "basis.npz"
    original = _spec()

    save_gamma_routed_basis_spec(path, original)

    with np.load(path, allow_pickle=False) as payload:
        assert str(payload["projection_basis_kind"].item()) == "gamma_routed"
        assert all(not payload[name].dtype.hasobject for name in payload.files)
    restored = load_gamma_routed_basis_spec(path)
    assert restored.projection_basis_kind == "gamma_routed"
    assert restored.k_indices == (4, 7)
    assert restored.heff_k_indices == (4, 7)
    np.testing.assert_array_equal(restored.frames, original.frames)
    np.testing.assert_array_equal(restored.authoritative_heff, original.authoritative_heff)
    for k_index in original.k_indices:
        expected, _ = assemble_gamma_routed_projectors(
            original.routed_frames_for_k(k_index),
            layout=original.layout,
            thresholds=original.thresholds,
            include_high=False,
        )
        actual, high = restored.assemble_for_k(k_index, include_high=True)
        np.testing.assert_array_equal(actual, expected)
        assert high is not None
        np.testing.assert_allclose(actual.conj().T @ high, 0.0, atol=1.0e-12)


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        ("object_discriminator", "projection_basis_kind"),
        ("extra_object", "attacker_object"),
        ("missing", "k_indices"),
    ],
)
def test_gamma_routed_schema_boundary_rejects_as_typed_identity_error(
    tmp_path: Path,
    mutation: str,
    field: str,
) -> None:
    path = tmp_path / "basis.npz"
    save_gamma_routed_basis_spec(path, _spec())
    if mutation == "object_discriminator":
        _rewrite_npz(
            path,
            **{field: np.asarray(["gamma_routed"], dtype=object)},
        )
    elif mutation == "extra_object":
        _rewrite_npz(path, **{field: np.asarray([{"payload": 1}], dtype=object)})
    else:
        _rewrite_npz(path, remove=(field,))

    with pytest.raises(GammaRoutingError) as rejected:
        load_gamma_routed_basis_spec(path)

    assert rejected.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("joint_band_indices", np.asarray([0.5, 1.5], dtype=np.float64)),
        ("group_ranks", np.asarray([1, 1], dtype=np.int32)),
        ("group_offsets", np.asarray([0, 1, 2], dtype=np.float64)),
        ("k_indices", np.asarray([4, 7], dtype=np.int32)),
        ("heff_k_indices", np.asarray([4, 7], dtype=np.int32)),
        ("frames", None),
        ("reference_frames", None),
        ("authoritative_heff", None),
        ("route_gaps", None),
    ],
)
def test_gamma_routed_schema_rejects_noncanonical_disk_dtype(
    tmp_path: Path,
    field: str,
    replacement: np.ndarray | None,
) -> None:
    path = tmp_path / "basis.npz"
    save_gamma_routed_basis_spec(path, _spec())
    if replacement is None:
        with np.load(path, allow_pickle=False) as payload:
            original = np.array(payload[field], copy=True)
        replacement = original.astype(
            np.float32 if field == "route_gaps" else np.complex64
        )
    _rewrite_npz(path, **{field: replacement})

    with pytest.raises(GammaRoutingError) as rejected:
        load_gamma_routed_basis_spec(path)

    assert rejected.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


@pytest.mark.parametrize(
    "legacy_field",
    [
        "nlow_state_list",
        "norb_fix_list",
        "gauge_mode",
        "symmetry_adapted_frame_json",
    ],
)
def test_gamma_routed_archive_rejects_legacy_variant_fields(
    tmp_path: Path,
    legacy_field: str,
) -> None:
    path = tmp_path / "basis.npz"
    save_gamma_routed_basis_spec(path, _spec())
    value = (
        np.asarray([[0]], dtype=object)
        if legacy_field in {"nlow_state_list", "norb_fix_list"}
        else np.asarray("legacy")
    )
    _rewrite_npz(path, **{legacy_field: value})

    with pytest.raises(GammaRoutingError) as rejected:
        load_gamma_routed_basis_spec(path)

    assert rejected.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


@pytest.mark.parametrize(
    "field",
    [
        "frames",
        "layout_hash",
        "routing_thresholds_hash",
        "source_hamiltonian_hash",
        "ordered_q_hashes",
        "raw_action_package_hash",
        "candidate_certificate_hash",
    ],
)
def test_gamma_routed_handoff_rejects_tampered_identity(
    tmp_path: Path,
    field: str,
) -> None:
    path = tmp_path / "basis.npz"
    save_gamma_routed_basis_spec(path, _spec())
    with np.load(path, allow_pickle=False) as payload:
        value = np.array(payload[field], copy=True)
    if field == "frames":
        value.flat[0] += 0.125
    elif value.shape == ():
        value = np.asarray(str(value.item()) + "-tampered")
    else:
        value[0] = "f" * 64
    _rewrite_npz(path, **{field: value})

    with pytest.raises(GammaRoutingError) as rejected:
        load_gamma_routed_basis_spec(path)
    assert rejected.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


@pytest.mark.parametrize(
    "mutation",
    ["status", "certificate_hash", "state_hash", "action_hash"],
)
def test_gamma_routed_handoff_rejects_tampered_candidate_certificate(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / "basis.npz"
    save_gamma_routed_basis_spec(path, _spec())
    with np.load(path, allow_pickle=False) as payload:
        envelope = json.loads(str(payload["candidate_certificate_envelope_json"].item()))
    if mutation == "status":
        envelope["certificate_payload"]["status"] = "failed"
    elif mutation == "certificate_hash":
        envelope["certificate_hash"] = "0" * 64
    elif mutation == "state_hash":
        envelope["input_identity_payload"]["states"][0]["u_low_hash"] = "0" * 64
    else:
        envelope["input_identity_payload"]["operations"][0]["raw_action_hash"] = (
            "0" * 64
        )
    _rewrite_npz(
        path,
        candidate_certificate_envelope_json=np.asarray(
            json.dumps(envelope, separators=(",", ":"), sort_keys=True)
        ),
    )

    with pytest.raises(GammaRoutingError) as rejected:
        load_gamma_routed_basis_spec(path)

    assert rejected.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


def test_gamma_routed_handoff_rejects_missing_required_k() -> None:
    spec = _spec()

    with pytest.raises(GammaRoutingError) as rejected:
        spec.require_k_indices((4, 9))

    assert rejected.value.reason is CandidateRejectionReason.HANDOFF_K_COVERAGE


@pytest.mark.parametrize("field", ["authoritative_heff", "heff_hash", "heff_k_indices"])
def test_gamma_routed_handoff_rejects_wrong_authoritative_heff(
    tmp_path: Path,
    field: str,
) -> None:
    path = tmp_path / "basis.npz"
    save_gamma_routed_basis_spec(path, _spec())
    with np.load(path, allow_pickle=False) as payload:
        value = np.array(payload[field], copy=True)
    if field == "authoritative_heff":
        value.flat[0] += 0.25
    elif field == "heff_hash":
        value = np.asarray("wrong-heff")
    else:
        value[1] = 8
    _rewrite_npz(path, **{field: value})

    with pytest.raises(GammaRoutingError) as rejected:
        load_gamma_routed_basis_spec(path)
    assert rejected.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


def test_explicit_legacy_basis_spec_retains_stable_discriminator() -> None:
    legacy = ExplicitLegacyBasisSpec(
        artifact_identity={"basis_hash": "legacy-a"},
        nlow_state_list=[[0], [1]],
        resolved_norb_fix_list=[[[[0, 1.0]]], [[[1, 1.0]]]],
        gauge_mode="manual",
        frame_artifact=None,
        model_dim=2,
    )

    assert legacy.projection_basis_kind == "explicit_legacy"
    assert legacy.nlow_state_list == [[0], [1]]


def test_gamma_routed_spec_detects_in_memory_heff_replacement() -> None:
    spec = _spec()
    changed = np.array(spec.authoritative_heff, copy=True)
    changed[0, 0, 0] += 1.0

    with pytest.raises(GammaRoutingError) as rejected:
        replace(spec, authoritative_heff=changed)

    assert rejected.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


def test_gamma_routed_factory_binds_task7_candidate_certificate() -> None:
    base = _spec()
    presentation = MagneticPresentation(
        generators=(MagneticGenerator("E", False),),
        relations=(
            MagneticRelation("E^2", lhs=("E", "E"), rhs=(), central_phase=1.0),
        ),
        central_phases=(1.0,),
        source="gamma_handoff_test",
    )
    pairs = ((4, 4), (7, 7))

    certified = certify_gamma_routed_basis_spec(
        candidate_id="gamma-routed-test",
        artifact_identity=base.artifact_identity,
        layout=base.layout,
        thresholds=base.thresholds,
        k_indices=base.k_indices,
        routed_frames=tuple(base.routed_frames_for_k(k) for k in base.k_indices),
        authoritative_heff=base.authoritative_heff,
        heff_k_indices=base.heff_k_indices,
        closure_certificate_hashes=base.closure_certificate_hashes,
        routing_certificate_hashes=base.routing_certificate_hashes,
        source_hamiltonian_hash=base.source_hamiltonian_hash,
        ordered_q_hashes=base.ordered_q_hashes,
        raw_action_package_hash=base.raw_action_package_hash,
        operations={
            "E": CandidateOperationInput(
                name="E",
                antiunitary=False,
                d_full=np.eye(base.layout.full_dimension, dtype=np.complex128),
                pairs=pairs,
            )
        },
        exactified_actions={"E": np.eye(base.model_dim, dtype=np.complex128)},
        presentation=presentation,
        required_pairs={"E": pairs},
        candidate_thresholds=CandidateSymmetryThresholds.uniform(1.0e-10),
    )

    assert len(certified.candidate_certificate_hash) == 64
    assert len(certified.candidate_input_identity_hash) == 64
    envelope = verify_candidate_certificate_envelope(
        certified.candidate_certificate_envelope
    )
    assert envelope["candidate_id"] == certified.candidate_id
    assert set(
        record["k_index"]
        for record in envelope["input_identity_payload"]["states"]
    ) == set(certified.k_indices)


def test_gamma_routed_save_rejects_arbitrary_candidate_hashes(tmp_path: Path) -> None:
    with pytest.raises(GammaRoutingError) as rejected:
        save_gamma_routed_basis_spec(
            tmp_path / "basis.npz",
            _uncertified_spec(),
        )

    assert rejected.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


def test_gamma_routed_factory_rejects_failed_task7_candidate() -> None:
    base = _spec()
    presentation = MagneticPresentation(
        generators=(MagneticGenerator("E", False),),
        relations=(
            MagneticRelation("E^2", lhs=("E", "E"), rhs=(), central_phase=1.0),
        ),
        central_phases=(1.0,),
        source="gamma_handoff_test",
    )

    with pytest.raises(GammaRoutingError) as rejected:
        certify_gamma_routed_basis_spec(
            candidate_id="gamma-routed-test",
            artifact_identity=base.artifact_identity,
            layout=base.layout,
            thresholds=base.thresholds,
            k_indices=base.k_indices,
            routed_frames=tuple(base.routed_frames_for_k(k) for k in base.k_indices),
            authoritative_heff=base.authoritative_heff,
            heff_k_indices=base.heff_k_indices,
            closure_certificate_hashes=base.closure_certificate_hashes,
            routing_certificate_hashes=base.routing_certificate_hashes,
            source_hamiltonian_hash=base.source_hamiltonian_hash,
            ordered_q_hashes=base.ordered_q_hashes,
            raw_action_package_hash=base.raw_action_package_hash,
            operations={
                "E": CandidateOperationInput(
                    name="E",
                    antiunitary=False,
                    d_full=np.eye(base.layout.full_dimension, dtype=np.complex128),
                    pairs=((4, 4),),
                )
            },
            exactified_actions={"E": np.eye(base.model_dim, dtype=np.complex128)},
            presentation=presentation,
            required_pairs={"E": ((4, 4), (7, 7))},
            candidate_thresholds=CandidateSymmetryThresholds.uniform(1.0e-10),
        )

    assert rejected.value.reason is CandidateRejectionReason.CANDIDATE_SYMMETRY_FAILED


def test_kp_symm_states_consume_persisted_frames_and_heff_without_reprojection() -> None:
    spec = _spec()
    ctx = SimpleNamespace(
        required_k=[7, 4],
        q_count=spec.layout.q_count,
        full_dim=spec.layout.full_dimension,
        hamk_source_by_k={
            4: np.diag(np.arange(spec.layout.full_dimension)),
            7: np.diag(np.arange(spec.layout.full_dimension) + 10.0),
        },
    )

    with patch.object(
        projection_mod,
        "_states_for_resolved_anchors",
        side_effect=AssertionError("routed handoff must not reproject"),
    ):
        _states, source, target, first = projection_mod._states_from_gamma_routed_handoff(
            ctx,
            spec,
        )

    for k_index in ctx.required_k:
        expected_u, _ = spec.assemble_for_k(k_index, include_high=False)
        np.testing.assert_array_equal(source[k_index].u_low, expected_u)
        np.testing.assert_array_equal(
            source[k_index].heff,
            spec.authoritative_heff_for_k(k_index),
        )
        assert target[k_index] is source[k_index]
    assert first is source[7]


def test_project_handoff_loader_rejects_wrong_companion_heff_with_typed_reason(
    tmp_path: Path,
) -> None:
    spec = _spec()
    project_dir = tmp_path / "projection"
    _write_routed_project_artifacts(project_dir, spec)
    changed = np.array(spec.authoritative_heff, copy=True)
    changed[0, 0, 0] += 1.0
    np.save(project_dir / "heff.npy", changed)

    with pytest.raises(GammaRoutingError) as rejected:
        projection_mod._load_persisted_projection_basis_handoff(project_dir)

    assert rejected.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


def test_project_handoff_loader_retains_versioned_explicit_legacy_compatibility(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "projection"
    _write_legacy_project_artifacts(project_dir)

    loaded = projection_mod._load_persisted_projection_basis_handoff(project_dir)

    assert isinstance(loaded, ExplicitLegacyBasisSpec)
    assert loaded.projection_basis_kind == "explicit_legacy"
    assert loaded.nlow_state_list == [[0], [1]]


@pytest.mark.parametrize(
    "routed_field",
    ["frames", "layout_payload", "authoritative_heff", "joint_band_indices"],
)
def test_explicit_legacy_archive_rejects_routed_variant_fields(
    tmp_path: Path,
    routed_field: str,
) -> None:
    project_dir = tmp_path / "projection"
    _write_legacy_project_artifacts(project_dir)
    basis_path = project_dir / "basis.npz"
    with np.load(basis_path, allow_pickle=True) as payload:
        copied = {name: np.array(payload[name], copy=True) for name in payload.files}
    copied[routed_field] = np.asarray([0], dtype=np.int64)
    np.savez(basis_path, **copied)

    with pytest.raises(GammaRoutingError) as rejected:
        projection_mod._load_persisted_projection_basis_handoff(project_dir)

    assert rejected.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY

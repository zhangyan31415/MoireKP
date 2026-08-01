from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from kp import projection_handoff as handoff_mod
from kp.blocks import (
    GammaRoutingError,
    GammaRoutingThresholds,
    GammaRowLayout,
    assemble_gamma_routed_projectors,
    build_gamma_routed_frames,
)
from kp.blocks.gamma_layout import (
    certify_gamma_raw_action,
    gamma_certified_action_route_contract,
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
from kp.low_energy_selection import CandidateMetrics
from kp.selection_artifact import (
    CertificationEvidence,
    ResolvedCandidateIdentity,
    SelectionArtifact,
    SelectionBindingError,
    SelectionFailureCode,
    SelectionIdentity,
    SelectionInputIdentity,
    SelectionMetricEvidence,
    build_certified_gamma_selection_identity,
    gamma_routed_ordered_q_identity_hash,
    verify_certified_gamma_selection_artifact,
    verify_certified_gamma_selection_identity,
)
from kp.symmetry import candidate_certificate as candidate_certificate_mod
from kp.symmetry import projection as projection_mod
from kp.symmetry.candidate_certificate import (
    CandidateOperationInput,
    CandidateSymmetryThresholds,
    candidate_raw_action_package_hash,
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


def _kpoints() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [0.25, -0.125]], dtype=np.float64)


def _uncertified_spec(*, kpoints: np.ndarray | None = None) -> GammaRoutedBasisSpec:
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
        kpoints=_kpoints() if kpoints is None else kpoints,
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
        kpoints=base.kpoints,
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


def test_exact_action_drift_keeps_raw_package_but_changes_candidate_and_handoff() -> None:
    base = _uncertified_spec()
    pairs = ((4, 4), (7, 7))
    presentation = MagneticPresentation(
        generators=(MagneticGenerator("E", False),),
        relations=(
            MagneticRelation("E^2", lhs=("E", "E"), rhs=(), central_phase=1.0),
        ),
        central_phases=(1.0,),
        source="gamma_handoff_exact_action_identity_test",
    )
    operations = {
        "E": CandidateOperationInput(
            name="E",
            antiunitary=False,
            d_full=np.eye(base.layout.full_dimension, dtype=np.complex128),
            pairs=pairs,
        )
    }
    required_pairs = {"E": pairs}
    raw_package_hash = candidate_raw_action_package_hash(
        operations=operations,
        presentation=presentation,
        required_pairs=required_pairs,
    )

    def certify(exact: np.ndarray) -> GammaRoutedBasisSpec:
        return certify_gamma_routed_basis_spec(
            candidate_id="gamma-routed-exact-drift",
            artifact_identity={
                **base.artifact_identity,
                "basis_hash": base.base_basis_hash,
            },
            layout=base.layout,
            thresholds=base.thresholds,
            k_indices=base.k_indices,
            kpoints=base.kpoints,
            routed_frames=tuple(
                base.routed_frames_for_k(k) for k in base.k_indices
            ),
            authoritative_heff=base.authoritative_heff,
            heff_k_indices=base.heff_k_indices,
            closure_certificate_hashes=base.closure_certificate_hashes,
            routing_certificate_hashes=base.routing_certificate_hashes,
            source_hamiltonian_hash=base.source_hamiltonian_hash,
            ordered_q_hashes=base.ordered_q_hashes,
            raw_action_package_hash=raw_package_hash,
            operations=operations,
            exactified_actions={"E": exact},
            presentation=presentation,
            required_pairs=required_pairs,
            candidate_thresholds=CandidateSymmetryThresholds.uniform(3.0),
        )

    positive = certify(np.eye(base.model_dim, dtype=np.complex128))
    negative = certify(-np.eye(base.model_dim, dtype=np.complex128))

    assert positive.raw_action_package_hash == negative.raw_action_package_hash
    assert positive.raw_action_package_hash == raw_package_hash
    assert positive.candidate_input_identity_hash != negative.candidate_input_identity_hash
    assert positive.candidate_certificate_hash != negative.candidate_certificate_hash
    assert positive.handoff_identity_hash != negative.handoff_identity_hash


def test_gamma_handoff_recomputes_and_binds_factorized_route_contract() -> None:
    base = _uncertified_spec()
    pairs = ((4, 4), (7, 7))
    presentation = MagneticPresentation(
        generators=(MagneticGenerator("E", False),),
        relations=(
            MagneticRelation("E^2", lhs=("E", "E"), rhs=(), central_phase=1.0),
        ),
        central_phases=(1.0,),
        source="gamma_handoff_route_identity_test",
    )
    full_action = np.eye(base.layout.full_dimension, dtype=np.complex128)
    certified_action = certify_gamma_raw_action(
        name="E",
        full_action=full_action,
        layout=base.layout,
        q_permutations=((0,), (0,)),
        sector_map=(0, 1),
        antiunitary=False,
        thresholds=base.thresholds,
        tapw_source_basis_hash=base.layout.tapw_source_basis_hash,
    )
    route_contract = gamma_certified_action_route_contract(
        certified_action,
        layout=base.layout,
        thresholds=base.thresholds,
        full_action=full_action,
    )

    def certify(route: dict[str, object]) -> GammaRoutedBasisSpec:
        operations = {
            "E": CandidateOperationInput(
                name="E",
                antiunitary=False,
                d_full=full_action,
                pairs=pairs,
                route_contract=route,
            )
        }
        raw_hash = candidate_raw_action_package_hash(
            operations=operations,
            presentation=presentation,
            required_pairs={"E": pairs},
        )
        return certify_gamma_routed_basis_spec(
            candidate_id="gamma-routed-route-contract",
            artifact_identity={**base.artifact_identity, "basis_hash": base.base_basis_hash},
            layout=base.layout,
            thresholds=base.thresholds,
            k_indices=base.k_indices,
            kpoints=base.kpoints,
            routed_frames=tuple(base.routed_frames_for_k(k) for k in base.k_indices),
            authoritative_heff=base.authoritative_heff,
            heff_k_indices=base.heff_k_indices,
            closure_certificate_hashes=base.closure_certificate_hashes,
            routing_certificate_hashes=base.routing_certificate_hashes,
            source_hamiltonian_hash=base.source_hamiltonian_hash,
            ordered_q_hashes=base.ordered_q_hashes,
            raw_action_package_hash=raw_hash,
            operations=operations,
            exactified_actions={"E": np.eye(base.model_dim, dtype=np.complex128)},
            presentation=presentation,
            required_pairs={"E": pairs},
            candidate_thresholds=CandidateSymmetryThresholds.uniform(1.0e-10),
            certified_gamma_actions=(certified_action,),
        )

    certified = certify(route_contract)
    assert certified.raw_action_package_hash != "0" * 64

    drifted = dict(route_contract)
    drifted["sector_map"] = [1, 0]
    with pytest.raises(GammaRoutingError, match="route contract differs") as rejected:
        certify(drifted)
    assert rejected.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


def _selection_input(
    spec: GammaRoutedBasisSpec,
    **changes: str,
) -> SelectionInputIdentity:
    values = {
        "selection_mode": "auto",
        "frozen_target_window_hash": "b" * 64,
        "validation_k_indices_hash": "c" * 64,
        "ordered_q_hash": gamma_routed_ordered_q_identity_hash(spec),
        "source_hamiltonian_hash": spec.source_hamiltonian_hash,
        "action_package_hash": spec.raw_action_package_hash,
        "row_layout_hash": spec.layout.layout_hash,
        "selection_policy_hash": "d" * 64,
    }
    values.update(changes)
    return SelectionInputIdentity.create(**values)


def _selection_metrics(spec: GammaRoutedBasisSpec) -> CandidateMetrics:
    return CandidateMetrics(
        candidate_id=spec.candidate_id,
        dimension=spec.model_dim,
        band_rms_mev=0.1,
        band_max_mev=0.2,
        subspace_overlap=0.999,
        symmetry_residual=1.0e-10,
        symmetry_leakage=2.0e-10,
        structural_failure=None,
    )


def _selection_artifact(spec: GammaRoutedBasisSpec) -> SelectionArtifact:
    identity = build_certified_gamma_selection_identity(
        selection_input=_selection_input(spec),
        handoff=spec,
        metrics=_selection_metrics(spec),
    )
    return SelectionArtifact.certified(
        transaction_id="gamma-selection",
        identity=identity,
        payload_manifest_hash="e" * 64,
        projection_handoff=spec,
    )


def _write_selection_marker(project_dir: Path, artifact: SelectionArtifact) -> None:
    (project_dir / "selection_artifact.json").write_text(
        json.dumps(artifact.to_dict(), sort_keys=True),
        encoding="utf-8",
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
    np.testing.assert_array_equal(restored.kpoints, original.kpoints)
    assert restored.kpoints_hash == hash_array(original.kpoints)
    assert restored.artifact_identity["kpoints_hash"] == restored.kpoints_hash
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
        ("missing", "kpoints"),
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
        ("kpoints", None),
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
            np.float32 if field in {"route_gaps", "kpoints"} else np.complex64
        )
    _rewrite_npz(path, **{field: replacement})

    with pytest.raises(GammaRoutingError) as rejected:
        load_gamma_routed_basis_spec(path)

    assert rejected.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("joint_band_indices", np.asarray([[0, 1]], dtype=np.int64)),
        ("group_ranks", np.asarray([[1, 1]], dtype=np.int64)),
        ("group_offsets", np.asarray([[0, 1, 2]], dtype=np.int64)),
        ("k_indices", np.asarray([[4, 7]], dtype=np.int64)),
        ("heff_k_indices", np.asarray([[4, 7]], dtype=np.int64)),
    ],
)
def test_gamma_routed_schema_rejects_noncanonical_integer_shape(
    tmp_path: Path,
    field: str,
    replacement: np.ndarray,
) -> None:
    path = tmp_path / "basis.npz"
    save_gamma_routed_basis_spec(path, _spec())
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
        "kpoints",
        "kpoints_hash",
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
    if field in {"frames", "kpoints"}:
        value.flat[0] += 0.125
    elif field == "kpoints_hash":
        value = np.asarray("f" * 64)
    elif value.shape == ():
        value = np.asarray(str(value.item()) + "-tampered")
    else:
        value[0] = "f" * 64
    _rewrite_npz(path, **{field: value})

    with pytest.raises(GammaRoutingError) as rejected:
        load_gamma_routed_basis_spec(path)
    assert rejected.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


def test_gamma_routed_kpoints_order_binds_basis_and_handoff_identity() -> None:
    original = _uncertified_spec()
    reordered = _uncertified_spec(kpoints=_kpoints()[::-1])

    assert original.kpoints_hash == hash_array(_kpoints())
    assert reordered.kpoints_hash == hash_array(_kpoints()[::-1])
    assert reordered.kpoints_hash != original.kpoints_hash
    assert reordered.artifact_identity["basis_hash"] != original.artifact_identity["basis_hash"]
    assert reordered.handoff_identity_hash != original.handoff_identity_hash


@pytest.mark.parametrize(
    "bad_kpoints",
    [
        np.asarray([0.0, 0.25], dtype=np.float64),
        np.zeros((1, 2), dtype=np.float64),
        np.zeros((2, 3), dtype=np.float64),
        np.asarray([[0.0, 0.0], [np.nan, 0.0]], dtype=np.float64),
    ],
)
def test_gamma_routed_create_rejects_invalid_kpoints_shape_or_values(
    bad_kpoints: np.ndarray,
) -> None:
    with pytest.raises(GammaRoutingError) as rejected:
        _uncertified_spec(kpoints=bad_kpoints)

    assert rejected.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY
    assert "kpoints" in str(rejected.value)


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


@pytest.mark.parametrize("missing_field", ["operations", "states"])
def test_gamma_routed_handoff_rejects_rehashed_incomplete_certificate_payload(
    tmp_path: Path,
    missing_field: str,
) -> None:
    path = tmp_path / "basis.npz"
    original = _spec()
    envelope = original.candidate_certificate_envelope
    del envelope["certificate_payload"][missing_field]
    certificate_hash = candidate_certificate_mod._candidate_certificate_payload_hash(
        envelope["certificate_payload"]
    )
    envelope["certificate_hash"] = certificate_hash
    envelope_json = json.dumps(envelope, separators=(",", ":"), sort_keys=True)
    identity = dict(original.artifact_identity)
    identity["basis_hash"] = handoff_mod._bound_routed_basis_hash(
        base_basis_hash=original.base_basis_hash,
        layout_hash=original.layout.layout_hash,
        thresholds_hash=original.thresholds.identity_hash,
        k_indices_hash=str(identity["k_indices_hash"]),
        kpoints_hash=original.kpoints_hash,
        frame_hash=original.frame_hash,
        reference_frame_hash=original.reference_frame_hash,
        heff_hash=original.heff_hash,
        closure_certificate_hashes=original.closure_certificate_hashes,
        routing_certificate_hashes=original.routing_certificate_hashes,
        source_hamiltonian_hash=original.source_hamiltonian_hash,
        ordered_q_hashes=original.ordered_q_hashes,
        raw_action_package_hash=original.raw_action_package_hash,
        candidate_certificate_hash=certificate_hash,
        candidate_input_identity_hash=original.candidate_input_identity_hash,
    )

    with pytest.raises(GammaRoutingError) as rejected:
        tampered = replace(
            original,
            artifact_identity=identity,
            candidate_certificate_envelope_json=envelope_json,
            candidate_certificate_hash=certificate_hash,
        )
        save_gamma_routed_basis_spec(path, tampered)
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


@pytest.mark.parametrize(
    "updates",
    [
        {"k_indices": (4.5, 7.5), "heff_k_indices": (4.5, 7.5)},
        {"heff_k_indices": (4.0, 7.0)},
        {"joint_band_indices": (0.0, 1.0)},
        {"group_ranks": (True, 1)},
        {"group_offsets": (0, 1.0, 2)},
    ],
)
def test_gamma_routed_spec_rejects_non_integral_in_memory_metadata(
    updates: dict[str, tuple[object, ...]],
) -> None:
    with pytest.raises(GammaRoutingError) as rejected:
        replace(_spec(), **updates)

    assert rejected.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY
    assert "strict integers" in str(rejected.value)


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
        kpoints=base.kpoints,
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
            kpoints=base.kpoints,
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


def test_project_handoff_loader_rejects_unknown_discriminator_with_typed_reason(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "projection"
    _write_legacy_project_artifacts(project_dir)
    basis_path = project_dir / "basis.npz"
    with np.load(basis_path, allow_pickle=True) as payload:
        copied = {name: np.array(payload[name], copy=True) for name in payload.files}
    copied["projection_basis_kind"] = np.asarray("future_projection_kind")
    np.savez(basis_path, **copied)

    with pytest.raises(GammaRoutingError) as rejected:
        projection_mod._load_persisted_projection_basis_handoff(project_dir)

    assert rejected.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


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


def test_certified_selection_identity_is_built_from_real_gamma_handoff() -> None:
    spec = _spec()
    selection_input = _selection_input(spec)

    identity = build_certified_gamma_selection_identity(
        selection_input=selection_input,
        handoff=spec,
        metrics=_selection_metrics(spec),
    )

    assert identity.resolved_candidate.basis_handoff_hash == spec.handoff_identity_hash
    assert identity.resolved_candidate.authoritative_heff_hash == spec.heff_hash
    assert identity.resolved_candidate.heff_k_indices_hash == hash_array(
        np.asarray(spec.heff_k_indices, dtype=np.int64)
    )
    assert identity.certification_evidence is not None
    assert (
        identity.certification_evidence.symmetry_certificate_hash
        == spec.candidate_certificate_hash
    )
    assert verify_certified_gamma_selection_identity(identity, spec) is identity

    artifact = SelectionArtifact.certified(
        transaction_id="gamma-selection",
        identity=identity,
        payload_manifest_hash="e" * 64,
        projection_handoff=spec,
    )
    assert artifact.identity is identity
    assert verify_certified_gamma_selection_artifact(artifact, spec) is artifact


def test_kp_symm_loads_current_certified_gamma_selection_identity(
    tmp_path: Path,
) -> None:
    spec = _spec()
    project_dir = tmp_path / "projection"
    project_dir.mkdir()
    artifact = _selection_artifact(spec)
    _write_selection_marker(project_dir, artifact)

    selection_identity_hash = (
        projection_mod._load_current_gamma_selection_identity_hash(
            project_dir,
            spec,
        )
    )

    assert artifact.identity is not None
    assert selection_identity_hash == artifact.identity.selection_identity_hash


def test_kp_symm_rejects_symlinked_current_gamma_selection_marker(
    tmp_path: Path,
) -> None:
    spec = _spec()
    project_dir = tmp_path / "projection"
    project_dir.mkdir()
    outside_marker = tmp_path / "outside-selection.json"
    outside_marker.write_text(
        json.dumps(_selection_artifact(spec).to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    (project_dir / "selection_artifact.json").symlink_to(outside_marker)

    with pytest.raises(SelectionBindingError):
        projection_mod._load_current_gamma_selection_identity_hash(
            project_dir,
            spec,
        )


@pytest.mark.parametrize("invalid_marker", ["missing", "malformed", "pending"])
def test_kp_symm_rejects_invalid_current_gamma_selection(
    tmp_path: Path,
    invalid_marker: str,
) -> None:
    spec = _spec()
    project_dir = tmp_path / "projection"
    project_dir.mkdir()
    marker = project_dir / "selection_artifact.json"
    if invalid_marker == "malformed":
        marker.write_text("{not-json", encoding="utf-8")
    elif invalid_marker == "pending":
        pending = SelectionArtifact.pending(
            transaction_id="gamma-selection",
            selection_input_identity_hash="f" * 64,
        )
        marker.write_text(json.dumps(pending.to_dict()), encoding="utf-8")

    with pytest.raises(SelectionBindingError):
        projection_mod._load_current_gamma_selection_identity_hash(
            project_dir,
            spec,
        )


def test_canonical_symmetry_metadata_binds_one_selection_identity_per_operation(
    tmp_path: Path,
) -> None:
    selection_identity_hash = "f" * 64
    matrix_path = tmp_path / "exactified_E.npy"
    np.save(matrix_path, np.eye(2, dtype=np.complex128))
    summary = {
        "artifact_identity": {},
        "valley": "G",
        "spin": "all",
        "tolerance": 1.0e-10,
        "full_dim": 2,
        "low_dim": 2,
        "operations": [
            {
                "name": "E",
                "operation": "E",
                "antiunitary": False,
                "pairs": [],
                "status": "certified",
                "matrix_file": str(matrix_path),
            }
        ],
    }

    projection_mod._bind_selection_identity_to_summary(
        summary,
        selection_identity_hash,
    )
    projection_mod._write_canonical_symmetry_outputs(tmp_path, summary)

    with np.load(tmp_path / "representations.npz", allow_pickle=False) as payload:
        metadata = json.loads(str(payload["__metadata_json__"].item()))
    assert metadata["selection_identity_hash"] == selection_identity_hash
    assert metadata["operations"][0]["selection_identity_hash"] == (
        selection_identity_hash
    )


def test_kp_symm_validates_routed_selection_before_invalidating_outputs(
    tmp_path: Path,
) -> None:
    class _StopAfterPreflight(RuntimeError):
        pass

    spec = _spec()
    events: list[str] = []
    run_cfg = SimpleNamespace(
        symm_cfg={"output_dir": "symm"},
        project_cfg={"out_dir": "projection"},
        cfg_dir=tmp_path,
    )
    ctx = SimpleNamespace(output_dir=tmp_path / "symm")

    def load_handoff(_project_dir: Path) -> GammaRoutedBasisSpec:
        events.append("handoff")
        return spec

    def load_selection(
        _project_dir: Path,
        handoff: GammaRoutedBasisSpec,
    ) -> str:
        assert handoff is spec
        events.append("selection")
        return "f" * 64

    def stop_at_first_invalidation(_output_dir: Path) -> None:
        assert events == ["handoff", "selection"]
        raise _StopAfterPreflight

    with (
        patch.object(
            projection_mod,
            "_load_projection_run_config",
            return_value=run_cfg,
        ),
        patch.object(
            projection_mod,
            "_load_persisted_projection_basis_handoff",
            side_effect=load_handoff,
        ),
        patch.object(
            projection_mod,
            "_build_projection_run_context",
            return_value=ctx,
        ) as build_context,
        patch.object(
            projection_mod,
            "_load_current_gamma_selection_identity_hash",
            side_effect=load_selection,
        ),
        patch.object(
            projection_mod,
            "_invalidate_stale_canonical_symmetry_outputs",
            side_effect=stop_at_first_invalidation,
        ),
        pytest.raises(_StopAfterPreflight),
    ):
        projection_mod.run_symmetry_projection_from_config("unused.yaml")
    assert build_context.call_args.kwargs["create_output_dir"] is False
    assert build_context.call_args.kwargs["require_nlow_state_list"] is False


@pytest.mark.parametrize(
    "changed_input",
    (
        {"ordered_q_hash": "1" * 64},
        {"source_hamiltonian_hash": "2" * 64},
        {"action_package_hash": "3" * 64},
        {"row_layout_hash": "4" * 64},
        {"selection_mode": "explicit"},
    ),
)
def test_certified_selection_builder_rejects_input_handoff_mismatch(
    changed_input: dict[str, str],
) -> None:
    spec = _spec()

    with pytest.raises(SelectionBindingError) as rejected:
        build_certified_gamma_selection_identity(
            selection_input=_selection_input(spec, **changed_input),
            handoff=spec,
            metrics=_selection_metrics(spec),
        )

    assert rejected.value.failure_code is SelectionFailureCode.IDENTITY_MISMATCH


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("basis_handoff_hash", "1" * 64),
        ("authoritative_heff_hash", "2" * 64),
        ("heff_k_indices_hash", "3" * 64),
    ),
)
def test_gamma_selection_verifier_rejects_coherent_resolved_identity_drift(
    field: str,
    value: str,
) -> None:
    spec = _spec()
    good = build_certified_gamma_selection_identity(
        selection_input=_selection_input(spec),
        handoff=spec,
        metrics=_selection_metrics(spec),
    )
    evidence = good.certification_evidence
    assert evidence is not None
    resolved_values = good.resolved_candidate.to_dict()
    resolved_values.pop("resolved_candidate_hash")
    resolved_values[field] = value
    resolved = ResolvedCandidateIdentity.create(**resolved_values)
    if field == "basis_handoff_hash":
        metric = evidence.metric_evidence
        metric_evidence = SelectionMetricEvidence.create(
            candidate_id=metric.candidate_id,
            frozen_target_window_hash=metric.frozen_target_window_hash,
            validation_k_indices_hash=metric.validation_k_indices_hash,
            basis_handoff_hash=value,
            metrics=metric.metrics,
        )
        evidence = CertificationEvidence.create(
            metric_evidence=metric_evidence,
            symmetry_certificate_hash=evidence.symmetry_certificate_hash,
            symmetry_input_identity_hash=evidence.symmetry_input_identity_hash,
        )
    drifted = SelectionIdentity.create(
        selection_input=good.selection_input,
        selection_policy_hash=good.selection_policy_hash,
        resolved_candidate=resolved,
        certification_evidence=evidence,
    )

    with pytest.raises(SelectionBindingError) as rejected:
        verify_certified_gamma_selection_identity(drifted, spec)

    assert rejected.value.failure_code is SelectionFailureCode.IDENTITY_MISMATCH


@pytest.mark.parametrize(
    "certificate_field",
    ("symmetry_certificate_hash", "symmetry_input_identity_hash"),
)
def test_gamma_selection_verifier_rejects_certificate_identity_drift(
    certificate_field: str,
) -> None:
    spec = _spec()
    good = build_certified_gamma_selection_identity(
        selection_input=_selection_input(spec),
        handoff=spec,
        metrics=_selection_metrics(spec),
    )
    evidence = good.certification_evidence
    assert evidence is not None
    evidence_values = {
        "metric_evidence": evidence.metric_evidence,
        "symmetry_certificate_hash": evidence.symmetry_certificate_hash,
        "symmetry_input_identity_hash": evidence.symmetry_input_identity_hash,
    }
    evidence_values[certificate_field] = "f" * 64
    drifted = SelectionIdentity.create(
        selection_input=good.selection_input,
        selection_policy_hash=good.selection_policy_hash,
        resolved_candidate=good.resolved_candidate,
        certification_evidence=CertificationEvidence.create(**evidence_values),
    )

    with pytest.raises(SelectionBindingError):
        verify_certified_gamma_selection_identity(drifted, spec)


def test_gamma_certified_artifact_cannot_bypass_handoff_verification() -> None:
    spec = _spec()
    identity = build_certified_gamma_selection_identity(
        selection_input=_selection_input(spec),
        handoff=spec,
        metrics=_selection_metrics(spec),
    )

    with pytest.raises(SelectionBindingError):
        SelectionArtifact.certified(
            transaction_id="missing-handoff",
            identity=identity,
            payload_manifest_hash="e" * 64,
        )


def test_explicit_legacy_handoff_is_never_built_as_certified_selection() -> None:
    spec = _spec()
    legacy = ExplicitLegacyBasisSpec(
        artifact_identity=spec.artifact_identity,
        nlow_state_list=[[0], [1]],
        resolved_norb_fix_list=[[[[0, 1.0]]], [[[1, 1.0]]]],
        gauge_mode="manual",
        frame_artifact=None,
        model_dim=2,
    )

    with pytest.raises(SelectionBindingError):
        build_certified_gamma_selection_identity(
            selection_input=_selection_input(spec),
            handoff=legacy,
            metrics=_selection_metrics(spec),
        )

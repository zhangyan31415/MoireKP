from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kp import blocks as blocks_mod
from kp import projection_handoff as handoff_mod
from kp import selection_artifact as selection_artifact_mod
from kp.basis.selection import AutoGaugeConfig
from kp.blocks import GammaRoutingError, GammaRowLayout
from kp.identity import (
    IDENTITY_SCHEMA,
    KP_VERSION,
    PROJECTION_BASIS_SCHEMA_VERSION,
    hash_array,
    hash_mapping,
)
from kp.low_energy_selection import CandidateMetrics, SelectionThresholds
from kp.projection_selection import CandidateRejectionReason
from kp.selection_artifact import (
    CertificationStatus,
    SelectionArtifactStore,
    SelectionInputIdentity,
)
from kp.selection_orchestration import (
    CaseSelectionInputs,
    begin_case_selection,
    resolve_case_selection,
)


_COMMON_ARCHIVE_KEYS = frozenset(
    {
        "projection_basis_kind",
        "projection_basis_handoff_version",
        "identity_schema",
        "input_hash",
        "config_hash",
        "basis_hash",
        "package_version",
        "schema_version",
        "k_indices_hash",
        "heff_hash",
        "source_hamiltonian_hash",
        "kpoints_hash",
        "layout_payload",
        "anchor_spec_payload",
        "candidate_id",
        "action_package_hash",
        "candidate_certificate_hash",
        "candidate_input_identity_hash",
        "k_indices",
        "local_frames",
    }
)


def _digest(label: str) -> str:
    return hash_mapping({"label": label})


def _aab_common_project(project_dir: Path):
    basis_type = getattr(handoff_mod, "GammaCommonAnchorBasisSpec")
    qset = np.asarray([[0.0, 0.0]], dtype=np.float64)
    layout = GammaRowLayout.build(
        qsets=(qset, qset),
        num_layer_list=(1, 2),
        num_orb_per_layer_list=((2,), (2, 2)),
        spin_convention="all",
        source_basis_hash=_digest("source-basis"),
    )
    low_rows = (2, 8, 4, 10)
    remaining_rows = tuple(row for row in range(12) if row not in low_rows)
    eigenvectors = np.eye(12, dtype=np.complex128)[:, low_rows + remaining_rows]
    eigenvalues = np.arange(-4.0, 8.0, dtype=np.float64)
    anchor = blocks_mod.build_gamma_common_anchor_spec(
        reference_eigenvalues_by_q=(eigenvalues,),
        reference_eigenvectors_by_q=(eigenvectors,),
        joint_band_indices=(0, 1, 2, 3),
        layout=layout,
        gauge_config=AutoGaugeConfig(reference_q_index=0),
    )
    frames = blocks_mod.build_gamma_common_anchor_frames(
        eigenvalues_by_q=(eigenvalues,),
        eigenvectors_by_q=(eigenvectors,),
        layout=layout,
        anchor_spec=anchor,
    )
    local_frames = np.stack(
        [np.stack(frames.local_frames_by_q, axis=0)],
        axis=0,
    )
    k_indices = (4,)
    kpoints = np.asarray([[0.125, -0.25]], dtype=np.float64)
    heff = np.asarray(
        [np.diag([-0.4, -0.1, 0.2, 0.7])],
        dtype=np.complex128,
    )
    spec = basis_type.create(
        candidate_id="aab-common-four",
        artifact_identity={
            "identity_schema": IDENTITY_SCHEMA,
            "input_hash": _digest("input"),
            "config_hash": _digest("config"),
            "basis_hash": _digest("unbound-basis"),
            "package_version": KP_VERSION,
            "schema_version": PROJECTION_BASIS_SCHEMA_VERSION,
            "k_indices_hash": hash_array(np.asarray(k_indices, dtype=np.int64)),
            "heff_hash": hash_array(heff),
        },
        layout=layout,
        anchor_spec=anchor,
        k_indices=k_indices,
        local_frames=local_frames,
        authoritative_heff=heff,
        kpoints=kpoints,
        source_hamiltonian_hash=_digest("source-hamiltonian"),
        action_package_hash=_digest("actions"),
        candidate_certificate_hash=_digest("candidate-certificate"),
        candidate_input_identity_hash=_digest("candidate-input"),
    )
    project_dir.mkdir(parents=True, exist_ok=True)
    np.save(project_dir / "heff.npy", heff)
    np.save(project_dir / "kpoints.npy", kpoints)
    handoff_mod.save_gamma_common_anchor_basis_spec(project_dir / "basis.npz", spec)
    return spec


def _rewrite_npz(path: Path, **replacement: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload.update(replacement)
    np.savez(path, **payload)


def _recreate_common_handoff(original, *, anchor_spec=None, local_frames=None):
    return type(original).create(
        candidate_id=original.candidate_id,
        artifact_identity=original.artifact_identity,
        layout=original.layout,
        anchor_spec=(original.anchor_spec if anchor_spec is None else anchor_spec),
        k_indices=original.k_indices,
        local_frames=(
            original.local_frames if local_frames is None else local_frames
        ),
        authoritative_heff=original.authoritative_heff,
        kpoints=original.kpoints,
        source_hamiltonian_hash=original.source_hamiltonian_hash,
        action_package_hash=original.action_package_hash,
        candidate_certificate_hash=original.candidate_certificate_hash,
        candidate_input_identity_hash=original.candidate_input_identity_hash,
    )


def test_gamma_common_anchor_roundtrip_is_numeric_minimal_and_keeps_owner_views(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    original = _aab_common_project(project_dir)

    with np.load(project_dir / "basis.npz", allow_pickle=False) as archive:
        assert frozenset(archive.files) == _COMMON_ARCHIVE_KEYS
        assert str(archive["projection_basis_kind"].item()) == "gamma_common_anchor"
        assert all(not archive[name].dtype.hasobject for name in archive.files)
        assert archive["local_frames"].dtype == np.dtype(np.complex128)
    restored = handoff_mod.load_gamma_common_anchor_basis_spec(
        project_dir / "basis.npz"
    )

    np.testing.assert_array_equal(restored.local_frames, original.local_frames)
    np.testing.assert_array_equal(
        restored.authoritative_heff,
        original.authoritative_heff,
    )
    np.testing.assert_array_equal(restored.kpoints, original.kpoints)
    assert restored.anchor_spec.to_payload() == original.anchor_spec.to_payload()
    assert restored.physical_layer_anchor_counts == (0, 2, 2)
    assert restored.source_qset_anchor_counts == (0, 4)
    assert restored.active_model_layers == (1, 2)
    assert restored.model_group_ranks == (2, 2)
    assert restored.model_group_qset_indices == (1, 1)


@pytest.mark.parametrize("fault", ("zero", "duplicate_column"))
def test_gamma_common_anchor_rejects_nonorthonormal_local_frames(
    tmp_path: Path,
    fault: str,
) -> None:
    original = _aab_common_project(tmp_path / fault)
    damaged = np.array(original.local_frames, copy=True)
    if fault == "zero":
        damaged[...] = 0.0
    else:
        damaged[..., 1] = damaged[..., 0]

    with pytest.raises(GammaRoutingError) as rejected:
        _recreate_common_handoff(original, local_frames=damaged)

    assert rejected.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


@pytest.mark.parametrize(
    ("field", "value_factory"),
    (
        ("same_q_dimension", lambda spec, layout: spec.same_q_dimension + 1),
        ("reference_q_index", lambda spec, layout: layout.q_count),
        (
            "physical_layer_count",
            lambda spec, layout: sum(layout.num_layer_list) + 1,
        ),
        ("source_qset_count", lambda spec, layout: 3),
    ),
)
def test_gamma_common_anchor_rejects_anchor_layout_metadata_drift(
    tmp_path: Path,
    field: str,
    value_factory,
) -> None:
    original = _aab_common_project(tmp_path / field)
    payload = original.anchor_spec.to_payload()
    payload[field] = value_factory(original.anchor_spec, original.layout)
    identity_payload = dict(payload)
    identity_payload.pop("identity_hash")
    payload["identity_hash"] = hash_mapping(identity_payload)
    drifted_anchor = blocks_mod.GammaCommonAnchorSpec.from_payload(payload)

    with pytest.raises(GammaRoutingError) as rejected:
        _recreate_common_handoff(original, anchor_spec=drifted_anchor)

    assert rejected.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


@pytest.mark.parametrize(
    "drift",
    (
        "local_frames",
        "layout_payload",
        "anchor_spec_payload",
        "heff.npy",
        "kpoints.npy",
    ),
)
def test_gamma_common_anchor_loader_rejects_any_bound_payload_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    project_dir = tmp_path / drift.replace(".", "-")
    _aab_common_project(project_dir)
    basis_path = project_dir / "basis.npz"
    if drift == "local_frames":
        with np.load(basis_path, allow_pickle=False) as archive:
            value = np.array(archive[drift], copy=True)
        value.flat[0] += 0.125
        _rewrite_npz(basis_path, local_frames=value)
    elif drift in {"layout_payload", "anchor_spec_payload"}:
        with np.load(basis_path, allow_pickle=False) as archive:
            value = json.loads(str(archive[drift].item()))
        if drift == "layout_payload":
            value["uniform_orbital_count"] = float(
                value["uniform_orbital_count"]
            )
        else:
            value["identity_hash"] = "f" * 64
        _rewrite_npz(
            basis_path,
            **{
                drift: np.asarray(
                    json.dumps(value, sort_keys=True, separators=(",", ":"))
                )
            },
        )
    else:
        path = project_dir / drift
        value = np.load(path, allow_pickle=False)
        changed = np.array(value, copy=True)
        changed.flat[0] += 0.125
        np.save(path, changed)

    with pytest.raises(GammaRoutingError) as rejected:
        handoff_mod.load_gamma_common_anchor_basis_spec(basis_path)

    assert rejected.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


def test_auto_resolution_accepts_common_anchor_without_routing_certificate(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    _aab_common_project(project_dir)
    handoff = handoff_mod.load_gamma_common_anchor_basis_spec(
        project_dir / "basis.npz"
    )
    selection_input = SelectionInputIdentity.create(
        selection_mode="auto",
        frozen_target_window_hash=_digest("target"),
        validation_k_indices_hash=_digest("validation-k"),
        ordered_q_hash=(
            selection_artifact_mod.gamma_common_anchor_ordered_q_identity_hash(
                handoff
            )
        ),
        source_hamiltonian_hash=handoff.source_hamiltonian_hash,
        action_package_hash=handoff.action_package_hash,
        row_layout_hash=handoff.layout.layout_hash,
        selection_policy_hash=_digest("policy"),
    )
    metrics = CandidateMetrics(
        candidate_id=handoff.candidate_id,
        dimension=handoff.model_dim,
        band_rms_mev=0.1,
        band_max_mev=0.2,
        subspace_overlap=0.999,
        symmetry_residual=1.0e-10,
        symmetry_leakage=2.0e-10,
        structural_failure=None,
    )
    store = SelectionArtifactStore(tmp_path / "selection")
    session = begin_case_selection(
        store=store,
        selection_input=selection_input,
        transaction_id="common-anchor-auto",
    )
    result = resolve_case_selection(
        CaseSelectionInputs.auto(
            selection_input=selection_input,
            candidates=(metrics,),
            thresholds=SelectionThresholds(1.0, 2.0, 0.99, 1.0e-8, 1.0e-8),
            projection_handoff=handoff,
            payloads={
                name: (project_dir / name).read_bytes()
                for name in ("basis.npz", "heff.npy", "kpoints.npy")
            },
        ),
        session=session,
    )

    assert result.status is CertificationStatus.CERTIFIED
    assert result.identity is not None
    resolved = result.identity.resolved_candidate
    assert resolved.projection_basis_kind == "gamma_common_anchor"
    assert resolved.basis_handoff_hash == handoff.artifact_identity["basis_hash"]
    assert (
        resolved.authoritative_heff_hash
        == handoff.artifact_identity["heff_hash"]
    )
    assert (
        resolved.heff_k_indices_hash
        == handoff.artifact_identity["k_indices_hash"]
    )
    assert not hasattr(handoff, "routing_certificate_hashes")

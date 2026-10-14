from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from kp.basis.selection import AutoGaugeConfig, GaugeAnchorReport
from kp.blocks import (
    GammaRowLayout,
    build_gamma_common_anchor_frames,
    build_gamma_common_anchor_spec,
)
from kp.identity import (
    IDENTITY_SCHEMA,
    KP_VERSION,
    PROJECTION_BASIS_SCHEMA_VERSION,
    hash_mapping,
)
from kp.low_energy_selection import CandidateMetrics
from kp.projection_handoff import (
    GammaCommonAnchorBasisSpec,
    GammaRoutedBasisSpec,
    save_gamma_common_anchor_basis_spec,
)
from kp.selection_artifact import (
    CertificationStatus,
    SelectionArtifact,
    SelectionArtifactStore,
    SelectionInputIdentity,
    build_certified_gamma_selection_identity,
    gamma_common_anchor_ordered_q_identity_hash,
)
from kp.symmetry import projection as projection_mod
from kp.symmetry.factorized_action import (
    load_factorized_actions,
    materialize_factorized_matrix,
)
from kp.symmetry.q_canonicalization import CanonicalQResult
from tests.kp.test_gamma_projection_handoff import (
    _certified_q2_dual_frame_spec,
    _write_routed_project_artifacts,
)


def _digest(label: str) -> str:
    return hash_mapping({"label": label})


def _common_q2_handoff(project_dir: Path) -> GammaCommonAnchorBasisSpec:
    qset = np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    layout = GammaRowLayout.build(
        qsets=(qset, qset),
        num_layer_list=(1, 1),
        num_orb_per_layer_list=((1,), (1,)),
        spin_convention="all",
        source_basis_hash=_digest("symmetry-common-source-basis"),
    )
    eigenvalues = tuple(
        np.asarray([-2.0, -1.0, 1.0, 2.0], dtype=np.float64)
        for _ in range(layout.q_count)
    )
    eigenvectors = tuple(
        np.eye(layout.same_q_dimension, dtype=np.complex128)
        for _ in range(layout.q_count)
    )
    anchor = build_gamma_common_anchor_spec(
        reference_eigenvalues_by_q=eigenvalues,
        reference_eigenvectors_by_q=eigenvectors,
        joint_band_indices=(0, 1),
        layout=layout,
        gauge_config=AutoGaugeConfig(reference_q_index=0),
    )
    frames = build_gamma_common_anchor_frames(
        eigenvalues_by_q=eigenvalues,
        eigenvectors_by_q=eigenvectors,
        layout=layout,
        anchor_spec=anchor,
    )
    local_frames = np.stack(
        [np.stack(frames.local_frames_by_q, axis=0)],
        axis=0,
    )
    heff = np.asarray(
        [np.diag([-0.4, -0.4, 0.2, 0.2])],
        dtype=np.complex128,
    )
    kpoints = np.asarray([[0.0, 0.0]], dtype=np.float64)
    handoff = GammaCommonAnchorBasisSpec.create(
        candidate_id="common-q2-symmetry",
        artifact_identity={
            "identity_schema": IDENTITY_SCHEMA,
            "input_hash": _digest("symmetry-common-input"),
            "config_hash": _digest("symmetry-common-config"),
            "package_version": KP_VERSION,
            "schema_version": PROJECTION_BASIS_SCHEMA_VERSION,
        },
        layout=layout,
        anchor_spec=anchor,
        k_indices=(4,),
        local_frames=local_frames,
        authoritative_heff=heff,
        kpoints=kpoints,
        source_hamiltonian_hash=_digest("symmetry-common-hamiltonian"),
        action_package_hash=_digest("symmetry-common-actions"),
        candidate_certificate_hash=_digest("symmetry-common-certificate"),
        candidate_input_identity_hash=_digest("symmetry-common-candidate-input"),
    )
    project_dir.mkdir(parents=True)
    save_gamma_common_anchor_basis_spec(project_dir / "basis.npz", handoff)
    np.save(project_dir / "heff.npy", heff)
    np.save(project_dir / "kpoints.npy", kpoints)
    return handoff


def _pending_common_anchor_selection(
    project_dir: Path,
    handoff: GammaCommonAnchorBasisSpec,
) -> SelectionArtifact:
    selection_input = SelectionInputIdentity.create(
        selection_mode="auto",
        frozen_target_window_hash=_digest("symmetry-common-target"),
        validation_k_indices_hash=_digest("symmetry-common-validation-k"),
        ordered_q_hash=gamma_common_anchor_ordered_q_identity_hash(handoff),
        source_hamiltonian_hash=handoff.source_hamiltonian_hash,
        action_package_hash=handoff.action_package_hash,
        row_layout_hash=handoff.layout.layout_hash,
        selection_policy_hash=_digest("symmetry-common-policy"),
    )
    metrics = CandidateMetrics(
        candidate_id=handoff.candidate_id,
        dimension=handoff.model_dim,
        band_rms_mev=0.1,
        band_max_mev=0.2,
        subspace_overlap=0.9,
        symmetry_residual=None,
        symmetry_leakage=None,
    )
    identity = build_certified_gamma_selection_identity(
        selection_input=selection_input,
        handoff=handoff,
        metrics=metrics,
    )
    pending_payload = {
        "schema": "kp.gamma-post-selection-symmetry-pending.v2",
        "status": "pending",
        "candidate_id": handoff.candidate_id,
        "candidate_dimension": handoff.model_dim,
        "nlow_state_list": [list(handoff.anchor_spec.joint_band_indices)],
        "mode": "gamma",
        "spin": handoff.layout.spin_scope,
        "ordered_q_hash": gamma_common_anchor_ordered_q_identity_hash(handoff),
        "qset_hashes": list(handoff.layout.ordered_qset_hashes),
        "selection_input_identity_hash": selection_input.selection_input_identity_hash,
        "resolved_candidate_hash": identity.resolved_candidate.resolved_candidate_hash,
        "basis_handoff_hash": identity.resolved_candidate.basis_handoff_hash,
    }
    store = SelectionArtifactStore(project_dir)
    initial = SelectionArtifact.pending(
        transaction_id="common-anchor-pending-symmetry",
        selection_input_identity_hash=selection_input.selection_input_identity_hash,
    )
    with store.begin(initial) as transaction:
        manifest_hash = transaction.stage_payloads(
            {
                "basis.npz": (project_dir / "basis.npz").read_bytes(),
                "pending_symmetry.json": json.dumps(
                    pending_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            }
        )
        artifact = SelectionArtifact.pending_symmetry(
            transaction_id=initial.transaction_id,
            identity=identity,
            payload_manifest_hash=manifest_hash,
        )
        transaction.publish(artifact)
    return artifact


def _q_swap_full_action(layout: GammaRowLayout) -> np.ndarray:
    by_address = {
        (
            address.q_index,
            address.source_group,
            address.physical_layer,
            address.spin_index,
            address.orbital,
        ): address.full_row
        for address in layout.addresses_by_full_row
    }
    action = np.zeros(
        (layout.full_dimension, layout.full_dimension),
        dtype=np.complex128,
    )
    for source in layout.addresses_by_full_row:
        target_key = (
            1 - source.q_index,
            source.source_group,
            source.physical_layer,
            source.spin_index,
            source.orbital,
        )
        action[by_address[target_key], source.full_row] = 1.0
    return action


def _repeated_qset_owner_handoff() -> GammaCommonAnchorBasisSpec:
    qset = np.asarray([[0.0, -3.0], [0.0, 3.0]], dtype=np.float64)
    layout = GammaRowLayout.build(
        qsets=(qset.copy(), qset.copy()),
        num_layer_list=(1, 2),
        num_orb_per_layer_list=((1,), (1, 1)),
        spin_convention="all",
        source_basis_hash=_digest("repeated-owner-source-basis"),
    )
    eigenvalues = tuple(
        np.arange(layout.same_q_dimension, dtype=np.float64)
        for _ in range(layout.q_count)
    )
    eigenvectors = tuple(
        np.eye(layout.same_q_dimension, dtype=np.complex128)
        for _ in range(layout.q_count)
    )
    anchor = build_gamma_common_anchor_spec(
        reference_eigenvalues_by_q=eigenvalues,
        reference_eigenvectors_by_q=eigenvectors,
        joint_band_indices=(1, 2),
        layout=layout,
        gauge_config=AutoGaugeConfig(reference_q_index=0),
    )
    frames = build_gamma_common_anchor_frames(
        eigenvalues_by_q=eigenvalues,
        eigenvectors_by_q=eigenvectors,
        layout=layout,
        anchor_spec=anchor,
    )
    local_frames = np.stack(
        [np.stack(frames.local_frames_by_q, axis=0)],
        axis=0,
    )
    heff = np.zeros((1, 4, 4), dtype=np.complex128)
    return GammaCommonAnchorBasisSpec.create(
        candidate_id="common-repeated-qset-owner",
        artifact_identity={
            "identity_schema": IDENTITY_SCHEMA,
            "input_hash": _digest("repeated-owner-input"),
            "config_hash": _digest("repeated-owner-config"),
            "package_version": KP_VERSION,
            "schema_version": PROJECTION_BASIS_SCHEMA_VERSION,
        },
        layout=layout,
        anchor_spec=anchor,
        k_indices=(4,),
        local_frames=local_frames,
        authoritative_heff=heff,
        kpoints=np.zeros((1, 2), dtype=np.float64),
        source_hamiltonian_hash=_digest("repeated-owner-hamiltonian"),
        action_package_hash=_digest("repeated-owner-actions"),
        candidate_certificate_hash=_digest("repeated-owner-certificate"),
        candidate_input_identity_hash=_digest("repeated-owner-candidate-input"),
    )


def _reflection_full_action(layout: GammaRowLayout) -> np.ndarray:
    by_address = {
        (
            address.q_index,
            address.source_group,
            address.physical_layer,
            address.spin_index,
            address.orbital,
        ): address.full_row
        for address in layout.addresses_by_full_row
    }
    action = np.zeros(
        (layout.full_dimension, layout.full_dimension),
        dtype=np.complex128,
    )
    for source in layout.addresses_by_full_row:
        target_key = (
            1 - source.q_index,
            source.source_group,
            source.physical_layer,
            source.spin_index,
            source.orbital,
        )
        action[by_address[target_key], source.full_row] = 1.0
    return action


def test_kp_symm_consumes_common_anchor_authorities_and_owner_views(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "common-project"
    expected = _common_q2_handoff(project_dir)

    loaded = projection_mod._load_persisted_projection_basis_handoff(project_dir)

    assert isinstance(loaded, GammaCommonAnchorBasisSpec)
    np.testing.assert_array_equal(loaded.authoritative_heff, expected.authoritative_heff)
    np.testing.assert_array_equal(loaded.kpoints, expected.kpoints)
    ctx = SimpleNamespace(
        required_k=[4],
        q_count=loaded.layout.q_count,
        full_dim=loaded.layout.full_dimension,
        hamk_source_by_k={
            4: np.zeros(
                (loaded.layout.full_dimension, loaded.layout.full_dimension),
                dtype=np.complex128,
            )
        },
    )
    consume_persisted_gamma = getattr(
        projection_mod,
        "_states_from_persisted_gamma_handoff",
    )
    _states, source, target, first = consume_persisted_gamma(ctx, loaded)
    expected_frame = loaded.assemble_model_for_k(4)
    np.testing.assert_array_equal(source[4].u_low, expected_frame)
    np.testing.assert_array_equal(source[4].heff, loaded.authoritative_heff_for_k(4))
    assert target[4] is source[4]
    assert first is source[4]

    report = projection_mod._gauge_report_from_gamma_common_anchor_handoff(loaded)
    assert report.gauge_mode == "auto_scdm"
    assert report.metric["projection_basis_kind"] == "gamma_common_anchor"
    assert report.metric["authoritative_frame"] == "common_anchor"
    assert report.state_selection_quality["owner_specs"] == [
        owner.to_payload() for owner in loaded.anchor_spec.owner_specs
    ]
    serialized = json.dumps(report.to_dict(), sort_keys=True).lower()
    assert "routed" not in serialized
    assert "dual" not in serialized


def test_kp_symm_accepts_identity_bound_pending_common_anchor_selection(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "common-project"
    handoff = _common_q2_handoff(project_dir)
    pending = _pending_common_anchor_selection(project_dir, handoff)

    binding = projection_mod._load_current_gamma_selection_binding(
        project_dir,
        handoff,
    )

    assert binding.artifact == pending
    assert binding.artifact.certification_status is CertificationStatus.PENDING_SYMMETRY
    assert binding.artifact.identity is not None
    assert (
        projection_mod._load_current_gamma_selection_identity_hash(
            project_dir,
            handoff,
        )
        == binding.artifact.identity.selection_identity_hash
    )


def test_kp_symm_certifies_pending_common_anchor_with_measured_metrics(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "common-project"
    handoff = _common_q2_handoff(project_dir)
    pending = _pending_common_anchor_selection(project_dir, handoff)
    binding = projection_mod._load_current_gamma_selection_binding(
        project_dir,
        handoff,
    )
    raw = np.eye(handoff.model_dim, dtype=np.complex128)
    exact = raw.copy()
    summary = {
        "operations": [
            {
                "name": "C2",
                "pairs": [
                    {
                        "raw": {
                            "d_unitarity_error": 2.0e-8,
                            "subspace_leakage": 3.0e-8,
                            "heff_covariance_residual": 4.0e-8,
                        },
                        "full_space_covariance_residual": 5.0e-8,
                    }
                ],
            }
        ]
    }

    certified = projection_mod._certify_pending_gamma_selection(
        binding,
        summary=summary,
        raw_low_matrices={"C2": raw},
        exact_matrices={"C2": exact},
    )

    assert certified is not None
    assert certified.certification_status is CertificationStatus.CERTIFIED
    assert certified.metrics is not None
    assert certified.metrics.symmetry_residual == 5.0e-8
    assert certified.metrics.symmetry_leakage == 3.0e-8
    assert pending.identity is not None
    assert certified.identity is not None
    assert certified.identity.selection_identity_hash != pending.identity.selection_identity_hash
    assert summary["selection_identity_hash"] == certified.identity.selection_identity_hash
    assert summary["operations"][0]["selection_identity_hash"] == (
        certified.identity.selection_identity_hash
    )

    binding.store.promote_pending_symmetry(certified)
    assert binding.store.load_current(require_certified=True) == certified


def test_common_anchor_global_frame_projects_and_factorizes_the_physical_q_route(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "common-project"
    _common_q2_handoff(project_dir)
    handoff = projection_mod._load_persisted_projection_basis_handoff(project_dir)
    assert isinstance(handoff, GammaCommonAnchorBasisSpec)
    ctx = SimpleNamespace(
        required_k=[4],
        q_count=handoff.layout.q_count,
        full_dim=handoff.layout.full_dimension,
        hamk_source_by_k={
            4: np.zeros(
                (handoff.layout.full_dimension, handoff.layout.full_dimension),
                dtype=np.complex128,
            )
        },
    )
    consume_persisted_gamma = getattr(
        projection_mod,
        "_states_from_persisted_gamma_handoff",
    )
    _states, source, target, _first = consume_persisted_gamma(ctx, handoff)
    full_action = _q_swap_full_action(handoff.layout)

    raw, _polar, rows = projection_mod._project_operation(
        operation="C2",
        antiunitary=False,
        d_full=full_action,
        states=source,
        pairs=[(4, 4)],
        tolerance=1.0e-13,
        target_states=target,
        source_states=source,
    )

    global_frame = handoff.assemble_model_for_k(4)
    expected_projected = global_frame.conj().T @ full_action @ global_frame
    np.testing.assert_array_equal(raw[0], expected_projected)
    assert rows[0]["raw"]["subspace_leakage"] == 0.0

    qset = np.asarray(handoff.layout.ordered_qsets[0], dtype=np.float64)
    q_geometry = CanonicalQResult(
        raw_q={"L1": qset, "L2": qset},
        canonical_q={"L1": qset, "L2": qset},
        artifact={
            "sector_order": ("L1", "L2"),
            "canonical_closure_max": 0.0,
            "status": "certified",
        },
    )
    items = [
        {
            "source_sector": sector,
            "source_q_index": source_q,
            "target_sector": sector,
            "target_q_index": 1 - source_q,
        }
        for sector in ("L1", "L2")
        for source_q in range(2)
    ]
    metadata = {
        "kp_symm_exactification": {
            "n_orb": list(handoff.model_group_ranks),
        },
        "operations": [
            {
                "name": "C2",
                "antiunitary": False,
                "model_action": {
                    "k_map": {"type": "negation"},
                    "q_map": {"type": "negation"},
                    "sector_map": "identity",
                },
                "model_basis_action": {"items": items},
            }
        ],
    }

    package, arrays = projection_mod._factorized_response_action_package(
        metadata,
        {"C2": raw[0]},
        q_geometry=q_geometry,
    )

    assert package["status"] == "certified"
    factorized = load_factorized_actions(package, arrays)["C2"]
    assert factorized.q_permutation == (1, 0, 3, 2)
    assert factorized.sector_permutation == (0, 1)
    assert factorized.q_counts == (2, 2)
    assert factorized.n_orb == handoff.model_group_ranks
    assert factorized.off_route_residual == 0.0
    assert factorized.q_covariance_residual == 0.0
    assert factorized.phase_alignment_residual == 0.0
    np.testing.assert_array_equal(
        materialize_factorized_matrix(factorized),
        expected_projected,
    )


def test_factorized_common_anchor_repeats_q_geometry_by_owner_qset_index(
    monkeypatch,
) -> None:
    handoff = _repeated_qset_owner_handoff()
    assert handoff.active_model_layers == (1, 2)
    assert handoff.model_group_ranks == (1, 1)
    assert handoff.model_group_qset_indices == (1, 1)
    np.testing.assert_array_equal(
        np.asarray(handoff.layout.ordered_qsets[0]),
        np.asarray(handoff.layout.ordered_qsets[1]),
    )
    model_frame, heff = handoff.model_state_for_k(4)
    state = projection_mod.ProjectionState(
        hamk=np.zeros(
            (handoff.layout.full_dimension, handoff.layout.full_dimension),
            dtype=np.complex128,
        ),
        heff=heff,
        u_low=model_frame,
    )
    full_action = _reflection_full_action(handoff.layout)
    raw, _polar, rows = projection_mod._project_operation(
        operation="C2",
        antiunitary=False,
        d_full=full_action,
        states={4: state},
        pairs=[(4, 4)],
        tolerance=1.0e-13,
        target_states={4: state},
        source_states={4: state},
    )
    assert rows[0]["raw"]["subspace_leakage"] == 0.0
    qset0 = np.asarray(handoff.layout.ordered_qsets[0], dtype=np.float64)
    qset1 = np.asarray(handoff.layout.ordered_qsets[1], dtype=np.float64)
    q_geometry = CanonicalQResult(
        raw_q={"L1": qset0, "L2": qset1},
        canonical_q={"L1": qset0, "L2": qset1},
        artifact={
            "sector_order": ("L1", "L2"),
            "canonical_closure_max": 0.0,
            "status": "certified",
        },
    )
    projected_support = [
        {
            "source_sector": owner,
            "source_q_index": source_q,
            "target_sector": owner,
            "target_q_index": 1 - source_q,
        }
        for owner in ("L1", "L2")
        for source_q in range(2)
    ]
    metadata = {
        "project_basis": {
            "projection_basis_kind": "gamma_common_anchor",
            "model_group_qset_indices": list(
                handoff.model_group_qset_indices
            ),
            "model_group_ranks": list(handoff.model_group_ranks),
        },
        "kp_symm_exactification": {
            "n_orb": list(handoff.model_group_ranks),
        },
        "operations": [
            {
                "name": "C2",
                "antiunitary": False,
                "model_action": {
                    "k_map": {"type": "reflection", "axis_deg": 0.0},
                    "q_map": {"type": "reflection", "axis_deg": 0.0},
                    "sector_map": "identity",
                },
                "model_basis_action": {"items": projected_support},
            }
        ],
    }
    captured: dict[str, object] = {}

    def resolve_owner_geometry(actual_metadata, *, q_geometry):
        assert actual_metadata is metadata
        assert q_geometry is not None
        project_basis = actual_metadata["project_basis"]
        source_qset_indices = tuple(
            int(value)
            for value in project_basis["model_group_qset_indices"]
        )
        n_orb = tuple(
            int(value) for value in project_basis["model_group_ranks"]
        )
        sector_order = tuple(q_geometry.artifact["sector_order"])
        source_q_vectors = tuple(
            np.asarray(q_geometry.canonical_q[name], dtype=np.float64)
            for name in sector_order
        )
        q_vectors = tuple(
            source_q_vectors[source_index]
            for source_index in source_qset_indices
        )
        q_counts = tuple(int(value.shape[0]) for value in q_vectors)
        captured.update(
            {
                "source_qset_indices": source_qset_indices,
                "q_counts": q_counts,
                "n_orb": n_orb,
            }
        )
        return {
            "q_vectors": q_vectors,
            "q_counts": q_counts,
            "n_orb": n_orb,
            "source_qset_indices": source_qset_indices,
        }

    monkeypatch.setattr(
        projection_mod,
        "_resolve_factorized_owner_geometry",
        resolve_owner_geometry,
        raising=False,
    )

    package, arrays = projection_mod._factorized_response_action_package(
        metadata,
        {"C2": raw[0]},
        q_geometry=q_geometry,
    )

    assert captured == {
        "source_qset_indices": (1, 1),
        "q_counts": (2, 2),
        "n_orb": (1, 1),
    }
    assert package["status"] == "certified"
    factorized = load_factorized_actions(package, arrays)["C2"]
    assert factorized.q_permutation == (1, 0, 3, 2)
    assert factorized.sector_permutation == (0, 1)
    assert factorized.q_counts == (2, 2)
    assert factorized.n_orb == handoff.model_group_ranks
    assert factorized.q_covariance_residual == 0.0
    assert factorized.off_route_residual == 0.0
    assert factorized.phase_alignment_residual == 0.0
    np.testing.assert_array_equal(
        materialize_factorized_matrix(factorized),
        raw[0],
    )


def test_factorized_common_anchor_keeps_two_sector_layout_for_one_owner() -> None:
    qset0 = np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    qset1 = np.asarray([[0.0, -2.0], [0.0, 2.0]], dtype=np.float64)
    q_geometry = CanonicalQResult(
        raw_q={"L1": qset0, "L2": qset1},
        canonical_q={"L1": qset0, "L2": qset1},
        artifact={
            "sector_order": ("L1", "L2"),
            "canonical_closure_max": 0.0,
            "status": "certified",
        },
    )
    metadata = {
        "project_basis": {
            "projection_basis_kind": "gamma_common_anchor",
            "model_group_ranks": [2],
            "model_group_qset_indices": [1],
        },
        "kp_symm_exactification": {"n_orb": [0, 2]},
    }

    resolved = projection_mod._resolve_factorized_owner_geometry(
        metadata,
        q_geometry=q_geometry,
    )

    assert resolved["n_orb"] == (0, 2)
    assert resolved["q_counts"] == (2, 2)
    assert resolved["source_qset_indices"] == (0, 1)
    np.testing.assert_array_equal(resolved["q_vectors"][0], qset0)
    np.testing.assert_array_equal(resolved["q_vectors"][1], qset1)


def test_kp_symm_dispatches_common_anchor_without_legacy_or_routed_assumptions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "common-project"
    handoff = _repeated_qset_owner_handoff()
    project_dir.mkdir(parents=True)
    save_gamma_common_anchor_basis_spec(project_dir / "basis.npz", handoff)
    np.save(project_dir / "heff.npy", handoff.authoritative_heff)
    np.save(project_dir / "kpoints.npy", handoff.kpoints)
    output_dir = tmp_path / "symm"
    qset0 = np.asarray(handoff.layout.ordered_qsets[0], dtype=np.float64)
    qset1 = np.asarray(handoff.layout.ordered_qsets[1], dtype=np.float64)
    run_cfg = SimpleNamespace(
        cfg_path="common.yaml",
        cfg_dir=tmp_path,
        valley="Gamma",
        spin="all",
        tolerance=1.0e-10,
        material={},
        symm_cfg={"output_dir": "symm"},
        project_cfg={"out_dir": "common-project"},
        canonical_layout=True,
    )
    ctx = SimpleNamespace(
        config=run_cfg,
        output_dir=output_dir,
        required_k=[4],
        q_count=handoff.layout.q_count,
        full_dim=handoff.layout.full_dimension,
        hamk_source_by_k={
            4: np.zeros(
                (handoff.layout.full_dimension, handoff.layout.full_dimension),
                dtype=np.complex128,
            )
        },
        q1=qset0,
        q2=qset1,
        q_model1=qset0,
        q_model2=qset1,
        mode="gamma",
        nlow_state_list=[],
        num_layer_list=list(handoff.layout.num_layer_list),
        num_orb_per_layer_list=[[1], [1, 1]],
        orb0=handoff.layout.same_q_dimension,
        q_rotation_deg=0.0,
        frame_inference=None,
        selected_gauge_states=None,
    )
    model_frame, model_heff = handoff.model_state_for_k(4)
    state = projection_mod.ProjectionState(
        hamk=ctx.hamk_source_by_k[4],
        heff=model_heff,
        u_low=model_frame,
    )
    persisted_states = ({4: state}, {4: state}, {4: state}, state)
    selection_binding = SimpleNamespace(
        name="pending-common-anchor-selection",
        artifact=SimpleNamespace(
            certification_status=CertificationStatus.PENDING_SYMMETRY,
        ),
    )
    owner_payloads = [
        owner.to_payload() for owner in handoff.anchor_spec.owner_specs
    ]
    report = GaugeAnchorReport(
        gauge_mode="auto_scdm",
        resolved_norb_fix_list=[],
        selections=[],
        metric={
            "type": "persisted_gamma_common_anchor_basis_handoff",
            "projection_basis_kind": "gamma_common_anchor",
            "authoritative_frame": "common_anchor",
        },
        state_selection_quality={
            "status": "persisted_gamma_common_anchor",
            "owner_specs": owner_payloads,
        },
        gauge_anchor_quality={
            "status": "persisted_common_anchor",
        },
        symmetry_closure_quality={
            "status": "persisted_gamma_common_anchor_basis",
            "symmetry_adapted_frame": None,
        },
    )
    calls: list[str] = []
    exactification_capture: dict[str, object] = {}
    sentinel = {"status": "dispatcher-common-anchor"}

    def build_context(_run_cfg, **kwargs):
        assert kwargs["create_output_dir"] is False
        assert kwargs["require_nlow_state_list"] is False
        calls.append("context")
        return ctx

    def consume_common(actual_ctx, actual_handoff):
        assert actual_ctx is ctx
        assert actual_handoff is handoff
        calls.append("states")
        return persisted_states

    def common_report(actual_handoff):
        assert actual_handoff is handoff
        calls.append("report")
        return report

    def append_operations(
        actual_ctx,
        *,
        summary,
        source_states,
        target_states,
        n_orb_for_exactification,
        compute_heff_covariance,
    ):
        assert actual_ctx is ctx
        assert source_states is persisted_states[1]
        assert target_states is persisted_states[2]
        assert n_orb_for_exactification == handoff.model_group_ranks
        assert compute_heff_covariance is True
        project_basis = summary["project_basis"]
        assert project_basis["projection_basis_kind"] == "gamma_common_anchor"
        assert project_basis["authoritative_frame"] == "common_anchor"
        assert project_basis["model_group_ranks"] == list(
            handoff.model_group_ranks
        )
        assert project_basis["model_group_qset_indices"] == list(
            handoff.model_group_qset_indices
        )
        serialized = json.dumps(project_basis, sort_keys=True).lower()
        assert "routed" not in serialized
        assert "dual" not in serialized
        calls.append("append")
        return {}

    def finish_exactification(
        actual_ctx,
        *,
        summary,
        raw_low_matrices,
        n_orb,
        gamma_selection_binding,
    ):
        assert actual_ctx is ctx
        assert raw_low_matrices == {}
        assert gamma_selection_binding is selection_binding
        exactification_capture["n_orb"] = n_orb
        exactification_capture["summary"] = summary
        calls.append("exactify")
        return sentinel

    monkeypatch.setattr(
        projection_mod,
        "_states_from_persisted_gamma_handoff",
        consume_common,
        raising=False,
    )
    monkeypatch.setattr(
        projection_mod,
        "_gauge_report_from_gamma_common_anchor_handoff",
        common_report,
        raising=False,
    )
    with (
        patch.object(
            projection_mod,
            "_load_projection_run_config",
            return_value=run_cfg,
        ),
        patch.object(
            projection_mod,
            "_load_persisted_projection_basis_handoff",
            return_value=handoff,
        ),
        patch.object(
            projection_mod,
            "_load_current_gamma_selection_binding",
            side_effect=lambda *_args, **_kwargs: (
                calls.append("selection") or selection_binding
            ),
        ),
        patch.object(
            projection_mod,
            "_build_projection_run_context",
            side_effect=build_context,
        ),
        patch.object(
            projection_mod,
            "_resolve_symmetry_project_identity",
            side_effect=lambda actual_ctx, *, handoff: (
                calls.append("identity") or dict(handoff.artifact_identity)
            ),
        ),
        patch.object(
            projection_mod,
            "_sector_orbital_counts",
            side_effect=AssertionError(
                "common-anchor exactification ranks must come from owner views"
            ),
        ),
        patch.object(
            projection_mod,
            "_append_projected_operation_summaries",
            side_effect=append_operations,
        ),
        patch.object(
            projection_mod,
            "_exactify_and_write_projection_summary",
            side_effect=finish_exactification,
        ),
    ):
        result = projection_mod.run_symmetry_projection_from_config(
            "unused.yaml"
        )

    assert result is sentinel
    required_calls = {
        "selection",
        "context",
        "states",
        "report",
        "identity",
        "append",
        "exactify",
    }
    assert required_calls.issubset(calls)
    position = {name: calls.index(name) for name in required_calls}
    assert position["selection"] < position["context"]
    assert position["context"] < position["states"]
    assert position["context"] < position["report"]
    assert position["states"] < position["append"]
    assert position["report"] < position["append"]
    assert position["identity"] < position["append"]
    assert position["append"] < position["exactify"]
    assert exactification_capture["n_orb"] == handoff.model_group_ranks


def test_kp_symm_keeps_legacy_routed_handoff_readable(tmp_path: Path) -> None:
    project_dir = tmp_path / "routed-project"
    expected = _certified_q2_dual_frame_spec()
    _write_routed_project_artifacts(project_dir, expected)
    np.save(project_dir / "kpoints.npy", expected.kpoints)

    loaded = projection_mod._load_persisted_projection_basis_handoff(project_dir)

    assert isinstance(loaded, GammaRoutedBasisSpec)
    assert loaded.handoff_identity_hash == expected.handoff_identity_hash

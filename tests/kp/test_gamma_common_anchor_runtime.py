from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import kp.gamma_auto_runtime as runtime
from kp.basis.selection import AutoGaugeConfig
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
from kp.low_energy_selection import SelectionThresholds
from kp.projection_handoff import (
    GammaCommonAnchorBasisSpec,
    load_gamma_common_anchor_basis_spec,
    load_gamma_routed_basis_spec,
)
from kp.selection_artifact import SelectionInputIdentity
from tests.kp.test_gamma_projection_handoff import (
    _certified_q2_dual_frame_spec,
)


def _digest(label: str) -> str:
    return hash_mapping({"label": label})


def _common_anchor_handoff() -> GammaCommonAnchorBasisSpec:
    qset = np.zeros((1, 2), dtype=np.float64)
    layout = GammaRowLayout.build(
        qsets=(qset, qset),
        num_layer_list=(1, 1),
        num_orb_per_layer_list=((1,), (1,)),
        spin_convention="all",
        source_basis_hash=_digest("runtime-source-basis"),
    )
    eigenvalues = np.asarray([-2.0, -1.0, 1.0, 2.0], dtype=np.float64)
    eigenvectors = np.eye(4, dtype=np.complex128)
    anchor = build_gamma_common_anchor_spec(
        reference_eigenvalues_by_q=(eigenvalues,),
        reference_eigenvectors_by_q=(eigenvectors,),
        joint_band_indices=(0, 2),
        layout=layout,
        gauge_config=AutoGaugeConfig(reference_q_index=0),
    )
    frames = build_gamma_common_anchor_frames(
        eigenvalues_by_q=(eigenvalues,),
        eigenvectors_by_q=(eigenvectors,),
        layout=layout,
        anchor_spec=anchor,
    )
    local_frames = np.stack(
        [np.stack(frames.local_frames_by_q, axis=0)],
        axis=0,
    )
    heff = np.asarray([np.diag([-0.4, 0.2])], dtype=np.complex128)
    kpoints = np.asarray([[0.125, -0.25]], dtype=np.float64)
    return GammaCommonAnchorBasisSpec.create(
        candidate_id="common-anchor-runtime",
        artifact_identity={
            "identity_schema": IDENTITY_SCHEMA,
            "input_hash": _digest("runtime-input"),
            "config_hash": _digest("runtime-config"),
            "package_version": KP_VERSION,
            "schema_version": PROJECTION_BASIS_SCHEMA_VERSION,
        },
        layout=layout,
        anchor_spec=anchor,
        k_indices=(4,),
        local_frames=local_frames,
        authoritative_heff=heff,
        kpoints=kpoints,
        source_hamiltonian_hash=_digest("runtime-source-hamiltonian"),
        action_package_hash=_digest("runtime-actions"),
        candidate_certificate_hash=_digest("runtime-candidate-certificate"),
        candidate_input_identity_hash=_digest("runtime-candidate-input"),
    )


def _write_common_authorities(
    directory: Path,
    handoff: GammaCommonAnchorBasisSpec,
    basis_bytes: bytes,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "basis.npz").write_bytes(basis_bytes)
    np.save(directory / "heff.npy", handoff.authoritative_heff)
    np.save(directory / "kpoints.npy", handoff.kpoints)


def test_gamma_runtime_serializes_common_anchor_with_external_authorities(
    tmp_path: Path,
) -> None:
    handoff = _common_anchor_handoff()

    basis_bytes = runtime._serialize_gamma_handoff(handoff)
    _write_common_authorities(tmp_path, handoff, basis_bytes)
    restored = load_gamma_common_anchor_basis_spec(tmp_path / "basis.npz")

    assert restored.projection_basis_kind == "gamma_common_anchor"
    np.testing.assert_array_equal(restored.local_frames, handoff.local_frames)
    np.testing.assert_array_equal(
        restored.authoritative_heff,
        handoff.authoritative_heff,
    )
    np.testing.assert_array_equal(restored.kpoints, handoff.kpoints)


def test_gamma_runtime_projects_spin_from_common_anchor_model_frame() -> None:
    handoff = _common_anchor_handoff()
    model_frame = handoff.assemble_model_for_k(4)
    spin_eigenvalue = {"up": 1.0, "down": -1.0}
    spin_diagonal = np.asarray(
        [
            spin_eigenvalue[address.spin_label]
            for address in handoff.layout.addresses_by_full_row
        ]
    )
    expected = model_frame.conj().T @ (
        spin_diagonal[:, np.newaxis] * model_frame
    )

    actual = runtime._gamma_projected_spin_operator(handoff)

    np.testing.assert_allclose(actual[0], expected, rtol=0.0, atol=1.0e-14)


def test_gamma_runtime_serializer_keeps_legacy_routed_roundtrip(
    tmp_path: Path,
) -> None:
    handoff = _certified_q2_dual_frame_spec()

    basis_bytes = runtime._serialize_gamma_handoff(handoff)
    basis_path = tmp_path / "basis.npz"
    basis_path.write_bytes(basis_bytes)
    restored = load_gamma_routed_basis_spec(basis_path)

    assert restored.projection_basis_kind == "gamma_routed"
    assert restored.handoff_identity_hash == handoff.handoff_identity_hash
    np.testing.assert_array_equal(
        restored.authoritative_heff,
        handoff.authoritative_heff,
    )


def test_gamma_runtime_finalize_commits_and_forwards_common_anchor_handoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    handoff = _common_anchor_handoff()
    selection_input = SelectionInputIdentity.create(
        selection_mode="auto",
        frozen_target_window_hash=_digest("runtime-target"),
        validation_k_indices_hash=_digest("runtime-validation-k"),
        ordered_q_hash=_digest("runtime-ordered-q"),
        source_hamiltonian_hash=handoff.source_hamiltonian_hash,
        action_package_hash=handoff.action_package_hash,
        row_layout_hash=handoff.layout.layout_hash,
        selection_policy_hash=_digest("runtime-selection-policy"),
    )
    thresholds = SelectionThresholds(
        band_rms_mev=1.0,
        band_max_mev=2.0,
        subspace_overlap=0.99,
        symmetry_residual=1.0e-8,
        symmetry_leakage=1.0e-8,
    )
    selection_preparation = SimpleNamespace(
        selection_input=selection_input,
        config=SimpleNamespace(selection_thresholds=thresholds),
    )
    preparation = runtime.GammaAutomaticRuntimePreparation(
        selection_preparation=selection_preparation,
        output_directory=tmp_path,
    )
    evaluation = SimpleNamespace(
        selection_input=selection_input,
        candidates=(),
        handoff=handoff,
    )
    captured: dict[str, object] = {}
    sentinel = object()

    monkeypatch.setattr(
        runtime,
        "evaluate_gamma_automatic_selection",
        lambda actual: evaluation,
    )

    def capture_resolution(canonical_inputs, *, session):
        captured["inputs"] = canonical_inputs
        captured["session"] = session
        return sentinel

    monkeypatch.setattr(runtime, "resolve_case_selection", capture_resolution)
    session = object()

    result = runtime.finalize_gamma_automatic_runtime(
        preparation,
        session=session,
    )

    assert result is sentinel
    assert captured["session"] is session
    assert captured["inputs"].projection_handoff is handoff
    restored = load_gamma_common_anchor_basis_spec(tmp_path / "basis.npz")
    assert restored.projection_basis_kind == "gamma_common_anchor"
    np.testing.assert_array_equal(
        np.load(tmp_path / "heff.npy", allow_pickle=False),
        handoff.authoritative_heff,
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / "kpoints.npy", allow_pickle=False),
        handoff.kpoints,
    )

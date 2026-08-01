from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import kp.cli as cli
import kp.gamma_auto_runtime as runtime
from kp.selection_artifact import SelectionInputIdentity
from kp.selection_artifact import CertificationStatus, SelectionArtifactStore
from kp.projection_handoff import load_gamma_routed_basis_spec
from kp.identity import load_projection_artifact_identity
from kp.symmetry import projection as projection_mod


def _selection_input() -> SelectionInputIdentity:
    return SelectionInputIdentity.create(
        selection_mode="auto",
        frozen_target_window_hash="1" * 64,
        validation_k_indices_hash="2" * 64,
        ordered_q_hash="3" * 64,
        source_hamiltonian_hash="4" * 64,
        action_package_hash="5" * 64,
        row_layout_hash="6" * 64,
        selection_policy_hash="7" * 64,
    )


def _write_dispatch_config(
    tmp_path: Path,
    *,
    spin: str = "all",
    mode: str = "Gamma",
    valley: str = "Gamma",
) -> Path:
    path = tmp_path / "case.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "case": {
                    "profile": "Gamma",
                    "q_shell": "q00",
                    "output_root": "outputs",
                },
                "material": {"spin": spin},
                "project": {
                    "mode": mode,
                    "out_dir": "projection",
                    "workers": 1,
                    "selection": {"mode": "auto"},
                },
                "symm": {"valley": valley, "spin": spin},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("mode", "valley"),
    (("General", "G1"), ("GammaK", "GammaK"), ("G", "G")),
)
def test_gamma_auto_dispatch_requires_exact_gamma_family(
    tmp_path: Path,
    mode: str,
    valley: str,
) -> None:
    cfg_path = _write_dispatch_config(tmp_path, mode=mode, valley=valley)

    assert not cli._gamma_auto_requested(cfg_path)


def _strict_auto_selection_payload() -> dict[str, object]:
    return {
        "mode": "auto",
        "routing_thresholds": {
            "energy_same_ev": 1.0e-9,
            "energy_different_ev": 1.0e-5,
            "capture_zero_fraction": 1.0e-9,
            "capture_loss_max": 1.0e-9,
            "local_action_isometry": 1.0e-10,
            "off_route_leakage": 1.0e-10,
            "closure_residual": 1.0e-10,
            "route_zero_gap": 1.0e-8,
            "route_covariance": 1.0e-10,
            "projector_residual": 1.0e-10,
            "anchor_sigma_min": 1.0e-8,
            "max_rank": 4,
            "max_iterations": 8,
        },
        "selection_thresholds": {
            "band_rms_mev": 1.0e-6,
            "band_max_mev": 1.0e-6,
            "subspace_overlap": 1.0 - 1.0e-10,
            "symmetry_residual": 1.0e-9,
            "symmetry_leakage": 1.0e-9,
        },
        "candidate_symmetry_thresholds": {
            "raw_h_leakage": 1.0e-10,
            "exactification_distance": 1.0e-10,
            "intertwining_residual": 1.0e-10,
            "heff_covariance_residual": 1.0e-10,
            "relation_residual": 1.0e-10,
            "antiunitary_square_residual": 1.0e-10,
            "exact_action_unitarity_residual": 1.0e-10,
            "projection_orthonormality_residual": 1.0e-10,
            "heff_hermiticity_residual": 1.0e-10,
            "raw_h_action_unitarity_residual": 1.0e-10,
        },
        "exactification": {
            "enabled": True,
            "max_rms_correction": 1.0e-8,
            "max_route_correction": 1.0e-8,
            "central_branch_margin": 1.0e-6,
            "max_iterations": 8,
            "condition_limit": 1.0e8,
        },
        "target_window": {
            "edge": "valence",
            "band_count": 4,
            "validation_k_indices": [0],
            "energy_reference_ev": 0.0,
            "degeneracy_tolerance_mev": 1.0e-6,
        },
        "candidate_seed_band_indices": [[0, 1, 2, 3]],
        "reference_k_index": 0,
        "downfold": {
            "method": "first_order",
            "e_ref": None,
            "pole_warning_mev": 10.0,
            "pole_danger_mev": 1.0,
            "fail_on_near_pole": True,
            "compute_pole_diagnostics": True,
            "compute_condition_number": False,
        },
    }


def _write_packed_gamma_runtime_case(
    tmp_path: Path,
    *,
    include_basis_hash: bool = True,
    top_level_basis_hash: object = "a" * 64,
    operation_basis_hashes: tuple[object | None, object | None] | None = None,
    declared_k_pairs: list[list[int]] | None = None,
) -> Path:
    source = np.diag([-2.0, -1.0, -2.0, -1.0]).astype(np.complex128)
    np.save(tmp_path / "hamk.npy", source[None, :, :])
    np.save(tmp_path / "q1.npy", np.zeros((1, 2), dtype=np.float64))
    np.save(tmp_path / "q2.npy", np.zeros((1, 2), dtype=np.float64))
    np.save(tmp_path / "kpoints.npy", np.asarray([[0.0, 0.0, 0.0]]))
    identity2 = np.eye(2, dtype=np.complex128)
    tr = np.block(
        [
            [np.zeros((2, 2), dtype=np.complex128), identity2],
            [-identity2, np.zeros((2, 2), dtype=np.complex128)],
        ]
    )
    phase = np.exp(1j * np.pi / 3.0)
    c3z = np.diag([phase, phase, phase.conjugate(), phase.conjugate()])
    matrix_entries = [
        {
            "key": "TR",
            "operation": "TR",
            "source_valley": "Gamma",
            "target_valley": "Gamma",
            "antiunitary": True,
            "k_map": {"type": "negation"},
            "q_map": {"type": "negation"},
            "sector_map": "identity",
        },
        {
            "key": "C3z",
            "operation": "C3z",
            "source_valley": "Gamma",
            "target_valley": "Gamma",
            "antiunitary": False,
            "k_map": {"type": "rotation", "angle_deg": 120.0},
            "q_map": {"type": "rotation", "angle_deg": 120.0},
            "sector_map": "identity",
        },
    ]
    if operation_basis_hashes is None:
        operation_basis_hashes = (
            ("a" * 64, "a" * 64)
            if include_basis_hash
            else (None, None)
        )
    for entry, basis_hash in zip(
        matrix_entries, operation_basis_hashes, strict=True
    ):
        if basis_hash is not None:
            entry["basis_hash"] = basis_hash
    metadata = {
        "schema": "test-packed-gamma-runtime.v1",
        "basis_order": "spin->group->q->layer->orbital",
        "matrices": matrix_entries,
    }
    if include_basis_hash:
        metadata["basis_hash"] = top_level_basis_hash
    if declared_k_pairs is not None:
        for row in metadata["matrices"]:
            row["k_pairs"] = declared_k_pairs
    symmetry_dir = tmp_path / "tapw_symmetry"
    symmetry_dir.mkdir()
    import json

    np.savez(
        symmetry_dir / "representations.npz",
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        TR=tr,
        C3z=c3z,
    )
    cfg_path = tmp_path / "runtime.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "case": {
                    "profile": "Gamma",
                    "q_shell": "q00",
                    "output_root": "outputs",
                },
                "material": {
                    "hamk_file": "hamk.npy",
                    "qset1_file": "q1.npy",
                    "qset2_file": "q2.npy",
                    "kpoints_file": "kpoints.npy",
                    "spin": "all",
                    "energy_unit": "eV",
                    "num_layer_list": [1, 1],
                    "num_orb_per_layer": [1],
                },
                "plot": {"mode": "Gamma", "hamk_index": 0},
                "kpath": {"tmat": np.eye(3).tolist()},
                "project": {
                    "mode": "Gamma",
                    "out_dir": "projection",
                    "workers": 3,
                    "selection": _strict_auto_selection_payload(),
                },
                "symm": {
                    "enable": True,
                    "valley": "Gamma",
                    "spin": "all",
                    "tolerance": 1.0e-10,
                    "tapw_symmetry_dir": "tapw_symmetry",
                    "operations": ["TR", "C3z"],
                },
            }
        ),
        encoding="utf-8",
    )
    return cfg_path


def test_gamma_auto_project_dispatches_before_legacy_and_publishes_pending_after_prepare(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_dispatch_config(tmp_path)
    events: list[str] = []
    selection_input = _selection_input()
    preparation = SimpleNamespace(
        selection_preparation=SimpleNamespace(selection_input=selection_input),
        output_directory=tmp_path / "projection",
    )
    session = SimpleNamespace(close=lambda: events.append("close"))
    resolved = object()

    def prepare(*_args, **_kwargs):
        events.append("prepare")
        return preparation

    def begin(**kwargs):
        assert kwargs["selection_input"] is selection_input
        events.append("pending")
        return session

    def finalize(received, *, session):
        assert received is preparation
        events.append("evaluate")
        return resolved

    monkeypatch.setattr(runtime, "prepare_gamma_automatic_runtime", prepare)
    monkeypatch.setattr(runtime, "finalize_gamma_automatic_runtime", finalize)
    monkeypatch.setattr(cli, "begin_case_selection", begin)
    monkeypatch.setattr(
        cli,
        "_cmd_project_from_config_impl",
        lambda *_args, **_kwargs: pytest.fail("Gamma auto entered legacy projection"),
    )

    result = cli.cmd_project_from_config(str(cfg_path))

    assert result is resolved
    assert events == ["prepare", "pending", "evaluate", "close"]


def test_gamma_auto_inspect_preview_uses_real_prepare_identity_without_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_dispatch_config(tmp_path)
    selection_input = _selection_input()
    preparation = SimpleNamespace(
        selection_preparation=SimpleNamespace(selection_input=selection_input),
        output_directory=tmp_path / "projection",
    )
    monkeypatch.setattr(
        runtime,
        "prepare_gamma_automatic_runtime",
        lambda *_args, **_kwargs: preparation,
    )
    monkeypatch.setattr(
        runtime,
        "finalize_gamma_automatic_runtime",
        lambda *_args, **_kwargs: pytest.fail("inspect evaluated candidates"),
    )

    preview = cli._preview_cli_selection_from_config(str(cfg_path))

    assert preview.selection_input is selection_input
    assert preview.identity is None
    assert preview.status.value == "PENDING"


def test_gamma_auto_dispatch_rejects_non_spinful_gamma_before_legacy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_dispatch_config(tmp_path, spin="up")
    monkeypatch.setattr(
        cli,
        "_cmd_project_from_config_impl",
        lambda *_args, **_kwargs: pytest.fail("unsupported Gamma auto entered legacy"),
    )

    with pytest.raises(ValueError, match="spin.*all"):
        cli.cmd_project_from_config(str(cfg_path))


def test_gamma_runtime_config_uses_strict_selection_payload_without_defaults() -> None:
    payload = {"mode": "auto"}

    with pytest.raises(ValueError, match="routing_thresholds"):
        runtime._load_strict_gamma_auto_config(payload)


def test_sampled_k_route_keeps_every_exact_hit_including_duplicate_gamma() -> None:
    kpoints = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        dtype=np.float64,
    )

    pairs = runtime._infer_sampled_k_pairs(
        operation="mirror",
        kpoints=kpoints,
        k_map={
            "type": "reflection",
            "axis_deg": 0.0,
            "reflection_axis_convention": "mirror_axis_deg",
        },
    )

    assert pairs == ((0, 0), (3, 0), (1, 1), (0, 3), (3, 3))


@pytest.mark.parametrize(
    ("kpoints", "k_map", "message"),
    [
        (
            np.asarray([[1.0, 0.0]], dtype=np.float64),
            {"type": "negation"},
            "no exact sampled k-pairs",
        ),
        (
            np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
            {"type": "rotation", "angle_deg": np.degrees(5.0e-9)},
            "gray zone",
        ),
        (
            np.asarray([[0.0, 0.0], [np.nan, 0.0]], dtype=np.float64),
            {"type": "identity"},
            "finite",
        ),
        (
            np.asarray([[0.0, 0.0]], dtype=np.float64),
            {"type": "affine"},
            "unsupported",
        ),
        (
            np.asarray([[0.0, 0.0]], dtype=np.float64),
            {"type": "rotation", "angle_deg": "120"},
            "finite numeric",
        ),
    ],
)
def test_sampled_k_route_fails_closed_for_unresolved_geometry(
    kpoints: np.ndarray,
    k_map: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        runtime._infer_sampled_k_pairs(
            operation="test",
            kpoints=kpoints,
            k_map=k_map,
        )


def test_gamma_physical_frontend_loads_layout_without_legacy_nlow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    np.save(
        tmp_path / "hamk.npy",
        np.diag([-2.0, -1.0, 1.0, 2.0]).astype(np.complex128)[None, :, :],
    )
    np.save(tmp_path / "q1.npy", np.zeros((1, 2), dtype=np.float64))
    np.save(tmp_path / "q2.npy", np.zeros((1, 2), dtype=np.float64))
    cfg_path = tmp_path / "physical.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "case": {
                    "profile": "Gamma",
                    "q_shell": "q00",
                    "output_root": "outputs",
                },
                "material": {
                    "hamk_file": "hamk.npy",
                    "qset1_file": "q1.npy",
                    "qset2_file": "q2.npy",
                    "spin": "all",
                    "energy_unit": "eV",
                    "num_layer_list": [1, 1],
                    "num_orb_per_layer": [1],
                },
                "project": {
                    "mode": "Gamma",
                    "downfold_method": "first_order",
                    "selection": {"mode": "auto"},
                },
                "symm": {"enable": True, "valley": "Gamma", "spin": "all"},
            }
        ),
        encoding="utf-8",
    )
    run_cfg = projection_mod._load_projection_run_config(
        str(cfg_path), developer_outputs=False
    )
    monkeypatch.setattr(
        projection_mod,
        "_normalize_nlow_state_list",
        lambda *_args, **_kwargs: pytest.fail("auto frontend normalized legacy nlow"),
    )

    physical = projection_mod._load_projection_physical_arrays_and_layout(run_cfg)

    assert physical.hamk.shape == (1, 4, 4)
    assert physical.num_layer_list == (1, 1)
    assert physical.num_orb_per_layer_list == ((1,), (1,))
    assert physical.mode == "gamma"


def test_gamma_runtime_materializes_real_packed_actions_routes_and_presentation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_packed_gamma_runtime_case(tmp_path)
    sentinel = object()
    captured: dict[str, object] = {}

    def prepare(inputs, config, *, workers):
        captured["inputs"] = inputs
        captured["config"] = config
        captured["workers"] = workers
        return sentinel

    monkeypatch.setattr(runtime, "prepare_gamma_automatic_selection", prepare)

    prepared = runtime.prepare_gamma_automatic_runtime(cfg_path)

    assert prepared.selection_preparation is sentinel
    assert captured["workers"] == 3
    inputs = captured["inputs"]
    assert inputs.k_indices == (0,)
    np.testing.assert_array_equal(inputs.kpoints, np.zeros((1, 2)))
    assert inputs.tapw_source_basis_hash == "a" * 64
    assert tuple(operation.name for operation in inputs.operations) == ("TR", "C3z")
    assert all(operation.q_permutations == ((0,), (0,)) for operation in inputs.operations)
    assert all(operation.sector_map == (0, 1) for operation in inputs.operations)
    assert all(operation.pairs == ((0, 0),) for operation in inputs.operations)
    assert tuple(generator.name for generator in inputs.presentation.generators) == (
        "TR",
        "C3z",
    )


@pytest.mark.parametrize("workers", [None, 0, -1, True, 1.5])
def test_gamma_runtime_requires_explicit_positive_integer_project_workers(
    tmp_path: Path,
    workers: object,
) -> None:
    cfg_path = _write_packed_gamma_runtime_case(tmp_path)
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if workers is None:
        del payload["project"]["workers"]
    else:
        payload["project"]["workers"] = workers
    cfg_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="project.workers.*strict positive integer"):
        runtime.prepare_gamma_automatic_runtime(cfg_path)


def test_gamma_runtime_requires_manifest_source_basis_identity(
    tmp_path: Path,
) -> None:
    cfg_path = _write_packed_gamma_runtime_case(
        tmp_path,
        include_basis_hash=False,
    )

    with pytest.raises(ValueError, match="basis_hash"):
        runtime.prepare_gamma_automatic_runtime(cfg_path)


def test_gamma_runtime_accepts_common_selected_operation_basis_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operation_basis_hash = "b" * 64
    cfg_path = _write_packed_gamma_runtime_case(
        tmp_path,
        include_basis_hash=False,
        operation_basis_hashes=(operation_basis_hash, operation_basis_hash),
    )
    captured: dict[str, object] = {}

    def prepare(inputs, _config, *, workers):
        captured["inputs"] = inputs
        captured["workers"] = workers
        return object()

    monkeypatch.setattr(runtime, "prepare_gamma_automatic_selection", prepare)

    runtime.prepare_gamma_automatic_runtime(cfg_path)

    assert captured["inputs"].tapw_source_basis_hash == operation_basis_hash
    assert captured["workers"] == 3


@pytest.mark.parametrize(
    ("operation_basis_hashes", "message"),
    (
        (("b" * 64, None), "basis_hash"),
        (("b" * 64, "c" * 64), "common basis_hash"),
        (("0" * 64, "0" * 64), "non-placeholder"),
        ((int("1" * 64), int("1" * 64)), "lowercase SHA-256 string"),
    ),
)
def test_gamma_runtime_rejects_invalid_selected_operation_basis_identity(
    tmp_path: Path,
    operation_basis_hashes: tuple[object | None, object | None],
    message: str,
) -> None:
    cfg_path = _write_packed_gamma_runtime_case(
        tmp_path,
        include_basis_hash=False,
        operation_basis_hashes=operation_basis_hashes,
    )

    with pytest.raises(ValueError, match=message):
        runtime.prepare_gamma_automatic_runtime(cfg_path)


def test_gamma_runtime_rejects_top_level_and_operation_basis_identity_mismatch(
    tmp_path: Path,
) -> None:
    cfg_path = _write_packed_gamma_runtime_case(
        tmp_path,
        operation_basis_hashes=("b" * 64, "b" * 64),
    )

    with pytest.raises(ValueError, match="top-level.*basis_hash"):
        runtime.prepare_gamma_automatic_runtime(cfg_path)


def test_gamma_runtime_rejects_non_string_top_level_basis_identity(
    tmp_path: Path,
) -> None:
    numeric_hash = int("1" * 64)
    cfg_path = _write_packed_gamma_runtime_case(
        tmp_path,
        top_level_basis_hash=numeric_hash,
        operation_basis_hashes=("1" * 64, "1" * 64),
    )

    with pytest.raises(ValueError, match="lowercase SHA-256 string"):
        runtime.prepare_gamma_automatic_runtime(cfg_path)


def test_gamma_runtime_rejects_packed_k_pairs_that_disagree_with_sampled_route(
    tmp_path: Path,
) -> None:
    cfg_path = _write_packed_gamma_runtime_case(
        tmp_path,
        declared_k_pairs=[[0, 1]],
    )

    with pytest.raises(ValueError, match="disagrees with actual sampled k-set"):
        runtime.prepare_gamma_automatic_runtime(cfg_path)


def test_gamma_auto_project_end_to_end_commits_identical_canonical_and_staged_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_packed_gamma_runtime_case(tmp_path)
    monkeypatch.setattr(
        cli,
        "_cmd_project_from_config_impl",
        lambda *_args, **_kwargs: pytest.fail("Gamma auto entered legacy projection"),
    )

    result = cli.cmd_project_from_config(str(cfg_path))

    assert result.status is CertificationStatus.CERTIFIED
    project_dir = tmp_path / "projection"
    staged = SelectionArtifactStore(project_dir).load_current_payloads()
    assert set(staged) == {
        "basis.npz",
        "heff.npy",
        "kpoints.npy",
        "wavefunctions.npz",
    }
    for name, content in staged.items():
        assert (project_dir / name).read_bytes() == content
    handoff = load_gamma_routed_basis_spec(project_dir / "basis.npz")
    assert handoff.kpoints_hash == runtime.hash_array(
        np.load(project_dir / "kpoints.npy", allow_pickle=False)
    )
    identity = load_projection_artifact_identity(project_dir)
    assert identity["basis_hash"] == handoff.artifact_identity["basis_hash"]


def test_gamma_auto_wavefunctions_embed_loader_compatible_routed_spin_operator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kp.symm_rep import _load_spin_operator

    cfg_path = _write_packed_gamma_runtime_case(tmp_path)
    monkeypatch.setattr(
        cli,
        "_cmd_project_from_config_impl",
        lambda *_args, **_kwargs: pytest.fail("Gamma auto entered legacy projection"),
    )

    cli.cmd_project_from_config(str(cfg_path))

    project_dir = tmp_path / "projection"
    wavefunctions_path = project_dir / "wavefunctions.npz"
    handoff = load_gamma_routed_basis_spec(project_dir / "basis.npz")
    spin_signs = np.asarray(
        [
            1.0 if address.spin_label == "up" else -1.0
            for address in handoff.layout.addresses_by_full_row
        ],
        dtype=np.complex128,
    )
    expected_rows = []
    for k_index in handoff.k_indices:
        routed_frame, _ = handoff.assemble_for_k(k_index, include_high=False)
        expected_rows.append(
            routed_frame.conj().T
            @ (spin_signs[:, np.newaxis] * routed_frame)
        )
    expected = np.stack(expected_rows, axis=0)

    with np.load(wavefunctions_path, allow_pickle=False) as archive:
        assert str(np.asarray(archive["spin_convention"]).item()) == "all"
        assert archive["spin_operator"].shape == (
            len(handoff.k_indices),
            handoff.model_dim,
            handoff.model_dim,
        )
        np.testing.assert_allclose(archive["spin_operator"], expected, atol=1.0e-12)
        np.testing.assert_allclose(
            archive["spin_operator"],
            archive["spin_operator"].conj().transpose(0, 2, 1),
            atol=1.0e-12,
        )

    loaded = _load_spin_operator(
        wavefunctions_path,
        len(handoff.k_indices),
        handoff.model_dim,
    )
    assert loaded is not None
    np.testing.assert_allclose(loaded, expected, atol=1.0e-12)


def test_gamma_auto_canonical_write_failure_leaves_pending_and_never_certified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_packed_gamma_runtime_case(tmp_path)

    def fail_install(output_directory: Path, payloads: dict[str, bytes]) -> None:
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / "basis.npz").write_bytes(payloads["basis.npz"])
        raise OSError("injected canonical write failure")

    monkeypatch.setattr(runtime, "_install_canonical_payloads", fail_install)

    with pytest.raises(OSError, match="injected canonical write failure"):
        cli.cmd_project_from_config(str(cfg_path))

    project_dir = tmp_path / "projection"
    marker = SelectionArtifactStore(project_dir).load_current()
    assert marker.certification_status is CertificationStatus.PENDING
    assert not (project_dir / "heff.npy").exists()


def test_gamma_auto_inspect_command_prepares_real_identity_without_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_packed_gamma_runtime_case(tmp_path)
    monkeypatch.setattr(
        runtime,
        "evaluate_gamma_automatic_selection",
        lambda *_args, **_kwargs: pytest.fail("inspect evaluated Gamma candidates"),
    )
    monkeypatch.setattr(
        cli,
        "plot_eigs_scatter",
        lambda *_args, out, **_kwargs: Path(out).write_text("plot", encoding="utf-8"),
    )
    monkeypatch.setattr(
        cli,
        "plot_inspect_band_and_qblock",
        lambda *_args, out, **_kwargs: Path(out).write_text("plot", encoding="utf-8"),
    )

    result = cli.cmd_plot_from_config(str(cfg_path))

    assert result.status is CertificationStatus.PENDING
    assert result.identity is None
    assert result.selection_input.selection_mode == "auto"
    assert not (tmp_path / "projection" / "selection_artifact.json").exists()


def test_gamma_runtime_payloads_bind_handoff_heff_kpoints_and_wavefunctions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heff = np.asarray([np.diag([-1.0, 1.0])], dtype=np.complex128)
    kpoints = np.asarray([[0.0, 0.0]], dtype=np.float64)
    identity = {
        "identity_schema": "moirekp.artifact-identity.v1",
        "input_hash": "1" * 64,
        "config_hash": "2" * 64,
        "basis_hash": "3" * 64,
        "package_version": "0.1.0",
        "schema_version": 2,
        "k_indices_hash": "4" * 64,
        "heff_hash": "5" * 64,
        "source_hamiltonian_hash": "6" * 64,
        "kpoints_hash": runtime.hash_array(kpoints),
    }
    handoff = SimpleNamespace(
        authoritative_heff=heff,
        k_indices=(0,),
        kpoints=kpoints,
        kpoints_hash=runtime.hash_array(kpoints),
        artifact_identity=identity,
    )
    monkeypatch.setattr(
        runtime,
        "_serialize_gamma_handoff",
        lambda _handoff: b"basis-bytes",
    )
    projected_spin = np.asarray([np.diag([1.0, -1.0])], dtype=np.complex128)
    monkeypatch.setattr(
        runtime,
        "_gamma_projected_spin_operator",
        lambda _handoff: projected_spin,
    )

    payloads = runtime._gamma_project_payloads(handoff, kpoints)

    assert set(payloads) == {
        "basis.npz",
        "heff.npy",
        "kpoints.npy",
        "wavefunctions.npz",
    }
    assert payloads["basis.npz"] == b"basis-bytes"
    np.testing.assert_allclose(runtime._load_npy_bytes(payloads["heff.npy"]), heff)
    np.testing.assert_allclose(
        runtime._load_npy_bytes(payloads["kpoints.npy"]), kpoints
    )
    with np.load(runtime._bytes_buffer(payloads["wavefunctions.npz"]), allow_pickle=False) as archive:
        assert str(archive["kpoints_hash"].item()) == runtime.hash_array(kpoints)
        np.testing.assert_array_equal(archive["k_indices"], np.asarray([0]))
        assert str(archive["spin_convention"].item()) == "all"
        np.testing.assert_array_equal(archive["spin_operator"], projected_spin)
        np.testing.assert_allclose(
            archive["wavefunctions"].conj().transpose(0, 2, 1)
            @ archive["wavefunctions"],
            np.eye(2)[None, :, :],
        )

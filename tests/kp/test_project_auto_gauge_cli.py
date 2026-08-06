from __future__ import annotations

import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import yaml

import kp.cli as cli
from kp.symmetry.projection import resolve_symmetry_validated_project_gauge
from kp.basis.selection import GaugeAnchorReport
from kp.basis.symmetry_gauge import (
    SymmetryAdaptedBasisFrame,
    SymmetryAdaptedInternalFrame,
)
from kp.selection_artifact import CertificationStatus, load_selection_artifact
from kp.low_energy_selection import (
    CandidateMetrics,
    CandidateSelectionError,
    FrozenBandCandidate,
    SelectionThresholds,
    select_projection_candidate,
)
from kp.projection_selection import (
    CandidateRejected,
    CandidateRejectionReason,
    TargetWindowSpec,
    evaluate_fixed_target_window,
    resolve_target_window,
)


def _write_tiny_project_config(tmp_path: Path, project: dict) -> Path:
    hamk = np.diag([0.0, 10.0, 1.0, 11.0, 100.0, 110.0, 101.0, 111.0]).astype(
        np.complex128
    )
    np.save(tmp_path / "hamk.npy", hamk[np.newaxis, :, :])
    np.save(tmp_path / "q1.npy", np.array([[0.0, 0.0]], dtype=float))
    np.save(tmp_path / "q2.npy", np.array([[0.0, 0.0]], dtype=float))
    np.save(tmp_path / "kpoints.npy", np.zeros((1, 3), dtype=float))
    (tmp_path / "bands.txt").write_text("0.0 1.0\n", encoding="utf-8")

    cfg = {
        "case": {"profile": "K1", "q_shell": "q06", "output_root": "outputs"},
        "material": {
            "name": "tiny",
            "hamk_file": "hamk.npy",
            "qset1_file": "q1.npy",
            "qset2_file": "q2.npy",
            "kpoints_file": "kpoints.npy",
            "band_file": "bands.txt",
            "spin": "up",
            "energy_unit": "eV",
            "num_layers": 2,
            "num_layer_list": [1, 1],
            "num_orb_per_layer": [2],
        },
        "plot": {"hamk_index": 0},
        "kpath": {"tmat": np.eye(3).tolist()},
        "project": project,
    }
    cfg_path = tmp_path / "source.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _write_tiny_project_config_with_symm(tmp_path: Path, project: dict, symm: dict) -> Path:
    cfg_path = _write_tiny_project_config(tmp_path, project)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["symm"] = symm
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _write_cached_symmetry_basis(symm_dir: Path, report: dict) -> None:
    symm_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"project_basis": report}
    np.savez_compressed(
        symm_dir / "representations.npz",
        __metadata_json__=np.asarray(json.dumps(metadata, sort_keys=True)),
        TR=np.eye(2, dtype=np.complex128),
    )
    (symm_dir / "residuals.csv").write_text("operation,status\nTR,ok\n", encoding="utf-8")
    (symm_dir / "summary.md").write_text("# KP Symmetry Projection\n", encoding="utf-8")


def _write_tiny_tapw_c3_source(symm_dir: Path) -> None:
    rep_dir = symm_dir / "representations" / "K1"
    rep_dir.mkdir(parents=True)
    raw_h = np.eye(8, dtype=np.complex128)
    np.savez(rep_dir / "C3_rawH.npz", matrix=raw_h)
    manifest = {
        "operations": {
            "K1": {
                "C3": {
                    "antiunitary": False,
                    "raw_h_operator_file": "K1/C3_rawH.npz",
                    "k_pairs": [[0, 0]],
                    "k_map": {"type": "rotation", "angle_deg": 120.0},
                    "q_map": {"type": "rotation", "angle_deg": 120.0},
                    "sector_map": "identity",
                }
            }
        }
    }
    (symm_dir / "representations" / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def _explicit_project_config() -> dict:
    return {
        "mode": "K1",
        "workers": 1,
        "downfold_method": "first_order",
        "out_dir": "project",
        "nlow_state_list": [[0], [0]],
        "norb_fix_list": [[[[0, 1.0]]], [[[0, 1.0]]]],
    }


def test_project_plot_axis_uses_inline_kpath_without_kpath_in(tmp_path: Path) -> None:
    cfg = {
        "kpath": {
            "labels": ["Gamma", "M", "K", "Gamma"],
            "points_per_segment": 2,
            "coordinates": {
                "Gamma": [0.0, 0.0],
                "M": [0.5, 0.0],
                "K": [1.0 / 3.0, 1.0 / 3.0],
            },
            "tmat": np.eye(3).tolist(),
        }
    }

    axis = cli._kpath_axis_from_config(
        cfg,
        cfg_dir=str(tmp_path / "relocated"),
        project_indices=list(range(7)),
        row_count=7,
    )

    assert np.asarray(axis["x_values"]).shape == (7,)
    assert len(axis["x_ticks"]) == 4
    assert axis["x_ticklabels"] == [r"$\Gamma$", "M", "K", r"$\Gamma$"]


@pytest.mark.parametrize("valley", ["K1", "M1"])
def test_non_gamma_auto_freezes_selected_bands_before_project_materialization(
    monkeypatch,
    tmp_path: Path,
    valley: str,
) -> None:
    """K/M auto selection must resolve bands before the explicit projector core."""

    cfg_path = _write_tiny_project_config(
        tmp_path,
        {
            "mode": valley,
            "workers": 1,
            "downfold_method": "first_order",
            "out_dir": "project",
            "selection": {
                "mode": "auto",
                "validation_bands": 1,
                "thresholds": {
                    "band_rms_mev": 2000.0,
                    "band_max_mev": 2000.0,
                },
            },
        },
    )
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["case"]["profile"] = valley
    cfg["material"]["efermi"] = 0.0
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    observed: dict[str, list[list[int]]] = {}

    class ReachedProjectMaterializer(RuntimeError):
        pass

    def stop_at_project_materializer(
        candidate_cfg_path: str,
        candidate_overrides=None,
    ):
        assert candidate_overrides is not None, (
            f"{valley} automatic selection must supply temporary project overrides"
        )
        assert candidate_overrides.get("nlow_state_list") not in (None, []), (
            f"{valley} automatic selection must freeze nlow_state_list before materialization"
        )
        with Path(candidate_cfg_path).open("r", encoding="utf-8") as handle:
            normalized = cli.normalize_case_config(
                yaml.safe_load(handle),
                config_path=candidate_cfg_path,
            )
        project_cfg = cli._apply_project_overrides(
            dict(normalized["project"]),
            candidate_overrides,
        )
        frozen = cli._normalize_nlow_state_list(project_cfg)
        assert len(frozen) == 2
        assert sum(len(row) for row in frozen) > 0
        observed["nlow_state_list"] = frozen
        raise ReachedProjectMaterializer

    monkeypatch.setattr(
        cli,
        "_cmd_project_from_config_impl",
        stop_at_project_materializer,
    )
    monkeypatch.setattr(
        cli,
        "prepare_symmetry_validated_project_gauge",
        lambda *_args, **_kwargs: SimpleNamespace(gauge_report=object()),
    )
    monkeypatch.setattr(
        cli,
        "_non_gamma_symmetry_validation_evidence",
        lambda _preparation: SimpleNamespace(
            symmetry_residual=0.001,
            symmetry_leakage=0.002,
            certificate_hash="a" * 64,
            input_identity_hash="b" * 64,
            report={"status": "validated"},
        ),
    )

    with pytest.raises(ReachedProjectMaterializer):
        cli.cmd_project_from_config(str(cfg_path))

    persisted = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert "nlow_state_list" not in persisted["project"]
    assert observed["nlow_state_list"]


def test_non_gamma_auto_policy_identity_records_partial_sector_band_semantics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_tiny_project_config(
        tmp_path,
        {
            "mode": "K1",
            "selection": {"mode": "auto", "validation_bands": 1},
        },
    )
    captured: dict[str, object] = {}
    real_builder = cli.build_selection_policy_hash

    def capture_policy(**kwargs):
        captured.update(kwargs)
        return real_builder(**kwargs)

    monkeypatch.setattr(cli, "build_selection_policy_hash", capture_policy)

    cli._prepare_cli_selection_request(str(cfg_path))

    envelope = captured["candidate_envelope_config"]
    assert isinstance(envelope, dict)
    assert envelope["partial_sector_band_metric_policy"] == (
        "diagnostic_when_candidate_layer_is_inactive"
    )


def test_non_gamma_band_error_preserves_spinless_single_band_dispersion() -> None:
    target = resolve_target_window(
        np.array([[0.0], [0.10]]),
        TargetWindowSpec(
            edge="conduction",
            band_count=1,
            validation_k_indices=(0, 1),
            energy_reference_ev=0.0,
            degeneracy_tolerance_mev=1.0e-6,
        ),
    )
    metrics = evaluate_fixed_target_window(target, np.array([[0.0], [0.25]]))

    assert metrics.rms_error_mev == pytest.approx(np.sqrt(0.5) * 150.0)
    assert metrics.maximum_abs_error_mev == pytest.approx(150.0)


def test_non_gamma_partial_sector_band_error_is_diagnostic_but_full_sector_is_hard() -> None:
    partial_candidates = (
        FrozenBandCandidate("k-b", ((), (), (22,)), ()),
        FrozenBandCandidate("k-b-large", ((), (), (22, 23)), ()),
    )
    full_candidates = (
        FrozenBandCandidate("bilayer", ((22,), (22,)), ()),
    )
    metrics = CandidateMetrics(
        candidate_id="k-b",
        dimension=18,
        band_rms_mev=14.5,
        band_max_mev=42.0,
        subspace_overlap=0.61,
        symmetry_residual=None,
        symmetry_leakage=None,
    )
    thresholds = SelectionThresholds(10.0, 30.0, 0.05, 1.0, 1.0)

    diagnostic_names = cli._non_gamma_diagnostic_metric_names(partial_candidates)
    decision = select_projection_candidate(
        [metrics],
        thresholds,
        allow_pending_symmetry=True,
        diagnostic_metric_names=diagnostic_names,
    )

    assert decision.selected.candidate_id == "k-b"
    assert diagnostic_names == ("band_rms_mev", "band_max_mev")
    assert cli._non_gamma_diagnostic_metric_names(full_candidates) == ()
    with pytest.raises(CandidateSelectionError, match="hard metric"):
        select_projection_candidate(
            [metrics],
            thresholds,
            allow_pending_symmetry=True,
            diagnostic_metric_names=cli._non_gamma_diagnostic_metric_names(
                full_candidates
            ),
        )


@pytest.mark.parametrize("spin", ["up", "down"])
def test_non_gamma_spin_slice_collapses_only_certified_paired_target(
    monkeypatch,
    spin: str,
) -> None:
    candidate = FrozenBandCandidate("spin-sliced", ((0,), (0,)), ())
    hamk = np.diag([0.0, 1.0]).astype(np.complex128)[np.newaxis, ...]

    monkeypatch.setattr(
        cli,
        "resolve_project_gauge_anchors",
        lambda _ham, *_args, nlow_state_list, **_kwargs: (
            [list(row) for row in nlow_state_list],
            GaugeAnchorReport(
                gauge_mode="auto_scdm",
                resolved_norb_fix_list=[list(row) for row in nlow_state_list],
                selections=[],
                metric={"type": "orthonormal"},
                state_selection_quality={"status": "ok"},
                gauge_anchor_quality={"sigma_min": 1.0, "condition_number": 1.0},
                symmetry_closure_quality={"status": "not_evaluated"},
                warnings=[],
            ),
        ),
    )
    monkeypatch.setattr(
        cli,
        "project_heff_full",
        lambda *_args, **_kwargs: (
            None,
            np.array([0.00005, 1.00005], dtype=float),
        ),
    )

    result = cli._evaluate_non_gamma_automatic_candidates(
        cfg_path="case.yaml",
        cfg={"material": {"efermi": -1.0}},
        candidates=(candidate,),
        hamk=hamk,
        q1=np.zeros((1, 2)),
        q2=np.zeros((1, 2)),
        num_layer_list=[1, 1],
        orb0=1,
        num_orb_per_layer_list=[[1], [1]],
        spin=spin,
        mode="M1",
        method="first_order",
        e_ref=None,
        reference_rows=[np.array([0.0, 0.0001, 1.0, 1.0001])],
        reference_k_index=0,
        selection_cfg={
            "validation_indices": [0],
            "validation_bands": 2,
            "spin_pair_tolerance_mev": 2.0,
            "thresholds": {
                "band_rms_mev": 0.01,
                "band_max_mev": 0.01,
                "subspace_overlap": 0.5,
            },
        },
        project_cfg={"gauge": "auto", "target": "conduction"},
    )

    assert result.candidate.candidate_id == "spin-sliced"
    assert result.decision.selected.band_rms_mev == pytest.approx(0.0, abs=1.0e-10)
    assert result.decision.selected.band_max_mev == pytest.approx(0.0, abs=1.0e-10)


def test_non_gamma_spin_slice_does_not_blindly_decimate_dense_bands() -> None:
    row = np.array([0.0, 0.0005, 0.0009, 0.0014, 0.0018, 0.0023])

    normalized = cli._spin_consistent_non_gamma_reference_rows(
        [row],
        validation_k_indices=[0],
        spin="up",
        edge="conduction",
        band_count=2,
        pair_tolerance_mev=2.0,
        pair_isolation_ratio=0.25,
    )

    assert len(normalized) == 1
    np.testing.assert_array_equal(normalized[0], row)


def test_non_gamma_spin_mapping_rejects_dense_opposite_spin_ambiguity() -> None:
    reference = np.array([0.0, 0.0010, 0.00105, 0.0020])
    wrong_spin_guide = np.array([0.001025])

    with pytest.raises(CandidateRejected) as caught:
        cli._monotone_spin_reference_subset(
            reference,
            wrong_spin_guide,
            edge="conduction",
            band_count=1,
            energy_reference_ev=-1.0,
        )

    assert caught.value.reason is CandidateRejectionReason.AMBIGUOUS_ENERGY_CLUSTER


@pytest.mark.parametrize(
    ("spin", "small_nlow", "large_nlow"),
    [
        pytest.param(
            "down",
            ((22,), (22,), ()),
            ((22, 23), (22,), ()),
            id="aab-k1-a-down",
        ),
        pytest.param(
            "up",
            ((), (), (22,)),
            ((), (), (22, 23)),
            id="aab-k1-b-up",
        ),
    ],
)
def test_aab_k1_spin_target_is_frozen_from_largest_envelope(
    monkeypatch,
    spin: str,
    small_nlow: tuple[tuple[int, ...], ...],
    large_nlow: tuple[tuple[int, ...], ...],
) -> None:
    candidates = (
        FrozenBandCandidate("small", small_nlow, ()),
        FrozenBandCandidate("largest", large_nlow, ()),
    )
    hamk = np.diag([0.0, 1.0]).astype(np.complex128)[np.newaxis, ...]

    monkeypatch.setattr(
        cli,
        "resolve_project_gauge_anchors",
        lambda _ham, *_args, nlow_state_list, **_kwargs: (
            [list(row) for row in nlow_state_list],
            GaugeAnchorReport(
                gauge_mode="auto_scdm",
                resolved_norb_fix_list=[list(row) for row in nlow_state_list],
                selections=[],
                metric={"type": "orthonormal"},
                state_selection_quality={"status": "ok"},
                gauge_anchor_quality={"sigma_min": 1.0, "condition_number": 1.0},
                symmetry_closure_quality={"status": "not_evaluated"},
                warnings=[],
            ),
        ),
    )

    def fake_project(*_args, nlow_state_list, **_kwargs):
        is_large = any(23 in row for row in nlow_state_list)
        values = [0.4, 1.4, 3.0] if is_large else [0.4, 1.4]
        return None, np.asarray(values, dtype=float)

    monkeypatch.setattr(cli, "project_heff_full", fake_project)

    result = cli._evaluate_non_gamma_automatic_candidates(
        cfg_path="case.yaml",
        cfg={"material": {"efermi": -1.0}},
        candidates=candidates,
        hamk=hamk,
        q1=np.zeros((1, 2)),
        q2=np.zeros((1, 2)),
        num_layer_list=[1, 2],
        orb0=1,
        num_orb_per_layer_list=[[1], [1, 1]],
        spin=spin,
        mode="K1",
        method="first_order",
        e_ref=None,
        reference_rows=[np.array([0.0, 0.4, 1.0, 1.4])],
        reference_k_index=0,
        selection_cfg={
            "validation_indices": [0],
            "validation_bands": 2,
            "thresholds": {
                "band_rms_mev": 0.01,
                "band_max_mev": 0.01,
                "subspace_overlap": 0.5,
            },
        },
        project_cfg={"gauge": "auto", "target": "conduction"},
    )

    assert result.candidate.candidate_id == "small"
    assert result.candidate.nlow_state_list == small_nlow
    assert result.decision.selected.band_rms_mev == pytest.approx(0.0, abs=1.0e-10)
    assert result.decision.selected.band_max_mev == pytest.approx(0.0, abs=1.0e-10)
    assert result.decision.diagnostic_metric_names == (
        "band_rms_mev",
        "band_max_mev",
    )


def test_non_gamma_anchor_threshold_prefers_public_name_and_accepts_legacy_name() -> None:
    public = cli._non_gamma_selection_thresholds(
        {"thresholds": {"gauge_anchor_sigma_min": 0.75}}
    )
    legacy = cli._non_gamma_selection_thresholds(
        {"thresholds": {"subspace_overlap": 0.25}}
    )

    assert public.subspace_overlap == pytest.approx(0.75)
    assert legacy.subspace_overlap == pytest.approx(0.25)


def test_non_gamma_anchor_threshold_rejects_conflicting_public_and_legacy_names() -> None:
    with pytest.raises(ValueError, match="gauge_anchor_sigma_min.*subspace_overlap"):
        cli._non_gamma_selection_thresholds(
            {
                "thresholds": {
                    "gauge_anchor_sigma_min": 0.75,
                    "subspace_overlap": 0.25,
                }
            }
        )


def test_non_gamma_validation_band_override_is_independent_of_model_band_window(
    tmp_path: Path,
) -> None:
    cfg_path = _write_tiny_project_config(
        tmp_path,
        {
            "mode": "K1",
            "selection": {"mode": "auto", "validation_bands": 6},
        },
    )
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"] = {
        "fit": {"method": "linear", "bands": 8},
        "bands": {"plot": {"top_bands": 8}},
    }
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    normalized = cli.normalize_case_config(raw, config_path=cfg_path)
    selection_cfg = normalized["project"]["selection"]

    assert selection_cfg == {"mode": "auto", "validation_bands": 6}
    assert normalized["fit"]["bands"] == 8
    assert normalized["bands"]["plot"]["top_bands"] == 8
    assert cli._non_gamma_selection_validation_band_count(
        normalized,
        selection_cfg,
    ) == 6


def test_non_gamma_candidate_validation_uses_reference_k_and_leaves_symmetry_pending(
    monkeypatch,
) -> None:
    candidates = (
        FrozenBandCandidate("small", ((0,), ()), ()),
        FrozenBandCandidate("large", ((0,), (0,)), ()),
    )
    hamk = np.stack(
        [
            np.diag([1.0, 2.0]).astype(np.complex128),
            np.diag([9.0, 10.0]).astype(np.complex128),
        ]
    )
    anchor_inputs: list[np.ndarray] = []

    def fake_anchors(reference_ham, *_args, nlow_state_list, **_kwargs):
        anchor_inputs.append(np.array(reference_ham, copy=True))
        return [list(row) for row in nlow_state_list], GaugeAnchorReport(
            gauge_mode="auto_scdm",
            resolved_norb_fix_list=[list(row) for row in nlow_state_list],
            selections=[],
            metric={"type": "orthonormal"},
            state_selection_quality={"status": "ok"},
            gauge_anchor_quality={"sigma_min": 1.0, "condition_number": 1.0},
            symmetry_closure_quality={"status": "not_evaluated"},
            warnings=[],
        )

    monkeypatch.setattr(cli, "resolve_project_gauge_anchors", fake_anchors)
    def fake_project(*_args, nlow_state_list, **_kwargs):
        dimension = sum(len(row) for row in nlow_state_list)
        return (
            np.zeros((dimension, dimension), dtype=np.complex128),
            np.arange(dimension, dtype=float),
            np.eye(dimension, dtype=np.complex128),
            None,
        )

    monkeypatch.setattr(cli, "project_heff_full", fake_project)
    monkeypatch.setattr(
        cli,
        "prepare_symmetry_validated_project_gauge",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("K/M preselection must not prepare source symmetry")
        ),
        raising=False,
    )
    result = cli._evaluate_non_gamma_automatic_candidates(
        cfg_path="case.yaml",
        cfg={},
        candidates=candidates,
        hamk=hamk,
        q1=np.zeros((1, 2)),
        q2=np.zeros((1, 2)),
        num_layer_list=[1, 1],
        orb0=1,
        num_orb_per_layer_list=[[1], [1]],
        spin="all",
        mode="K1",
        method="first_order",
        e_ref=None,
        reference_rows=[np.array([0.0]), np.array([0.0])],
        reference_k_index=0,
        selection_cfg={
            "validation_indices": [1],
            "validation_bands": 1,
            "thresholds": {
                "band_rms_mev": 1.0,
                "band_max_mev": 1.0,
                "subspace_overlap": 0.5,
                "symmetry_residual": 0.01,
                "symmetry_leakage": 0.01,
            },
        },
        project_cfg={"gauge": "auto", "target": "conduction"},
    )

    assert result.candidate.candidate_id == "small"
    assert result.decision.selected.symmetry_residual is None
    assert result.decision.selected.symmetry_leakage is None
    assert result.symmetry_evidence.report["status"] == "post_selection_pending"
    assert anchor_inputs
    np.testing.assert_allclose(
        anchor_inputs[0],
        cli._selected_spin_project_input(hamk[0], "all"),
    )


def test_non_gamma_automatic_materialization_uses_symmetry_canonical_frame(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_tiny_project_config_with_symm(
        tmp_path,
        {
            "mode": "K1",
            "workers": 1,
            "downfold_method": "first_order",
            "out_dir": "project",
            "gauge": "auto",
            "selection": {"mode": "auto"},
        },
        {
            "enable": True,
            "tapw_symmetry_dir": "tapw_symmetry",
            "operations": ["C3"],
        },
    )
    (tmp_path / "tapw_symmetry").mkdir()
    identity_internal = SymmetryAdaptedInternalFrame(
        status="applied",
        unitary=np.eye(1, dtype=np.complex128),
        primary_operation="C3",
        eigenvalues=(1.0 + 0.0j,),
    )
    frame = SymmetryAdaptedBasisFrame(
        status="applied",
        full_unitary=np.eye(2, dtype=np.complex128),
        sector_frames=(
            ("L1", 1, 1, identity_internal),
            ("L2", 1, 1, identity_internal),
        ),
    ).artifact()
    selected = [[[[0, 1.0]]], [[[0, 1.0]]]]
    report = GaugeAnchorReport(
        gauge_mode="auto_scdm",
        resolved_norb_fix_list=selected,
        selections=[],
        metric={"type": "orthonormal", "basis_is_orthonormal": True},
        state_selection_quality={"status": "selected"},
        gauge_anchor_quality={"sigma_min": 1.0, "condition_number": 1.0},
        symmetry_closure_quality={
            "status": "validated",
            "selected_candidate_id": "selected",
            "symmetry_adapted_frame": frame,
        },
        warnings=[],
    )
    resolver_calls: list[dict] = []

    def fake_symmetry_resolver(_cfg_path, *, project_config=None):
        resolver_calls.append(dict(project_config or {}))
        return report

    monkeypatch.setattr(
        cli,
        "resolve_symmetry_validated_project_gauge",
        fake_symmetry_resolver,
    )
    monkeypatch.setattr(
        cli,
        "resolve_project_gauge_anchors",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("automatic materialization must use source symmetry")
        ),
    )

    result = cli._cmd_project_from_config_impl(
        str(cfg_path),
        {
            "nlow_state_list": [[0], [0]],
            "_automatic_selection_materialization": True,
        },
    )

    assert result.candidate_dimension == 2
    assert len(resolver_calls) == 1
    assert resolver_calls[0]["nlow_state_list"] == [[0], [0]]
    assert result.symmetry_certificate_hash is None
    assert result.symmetry_input_identity_hash is None
    with np.load(tmp_path / "project" / "basis.npz", allow_pickle=True) as payload:
        persisted = json.loads(str(payload["symmetry_adapted_frame_json"].item()))
    assert persisted["status"] == "applied"
    assert persisted["frame_hash"] == frame["frame_hash"]


def test_non_gamma_symmetry_evidence_uses_measured_worst_values() -> None:
    report = GaugeAnchorReport(
        gauge_mode="auto_scdm",
        resolved_norb_fix_list=[[0], [1]],
        selections=[],
        metric={"type": "orthonormal"},
        state_selection_quality={"status": "ok"},
        gauge_anchor_quality={"sigma_min": 1.0, "condition_number": 1.0},
        symmetry_closure_quality={
            "status": "validated",
            "selected_candidate_id": "frame-1",
            "metrics": [
                {
                    "candidate_id": "frame-1",
                    "exactification_distance_by_op": {"C3": 0.003, "T": 0.004},
                    "support_off_by_op": {"C3": 0.005, "T": 0.002},
                    "metadata": {"status": "evaluated"},
                }
            ],
        },
        warnings=[],
    )
    context = SimpleNamespace(
        config=SimpleNamespace(valley="K1", spin="all"),
        mode="k1",
        nlow_state_list=[[0], [1]],
        q_model1=np.zeros((1, 2)),
        q_model2=np.zeros((1, 2)),
        operation_payloads={
            "C3": {
                "action": SimpleNamespace(matrix=np.eye(2, dtype=np.complex128)),
                "entry": {"_pairs": [(0, 0)]},
                "antiunitary": False,
            }
        },
    )

    evidence = cli._non_gamma_symmetry_validation_evidence(
        SimpleNamespace(gauge_report=report, context=context)
    )

    assert evidence.symmetry_residual == pytest.approx(0.004)
    assert evidence.symmetry_leakage == pytest.approx(0.005)
    assert (
        evidence.report["subspace_metric_semantics"]
        == "gauge_anchor_sigma_min"
    )
    assert len(evidence.certificate_hash) == 64
    assert len(evidence.input_identity_hash) == 64


def test_non_gamma_symmetry_evidence_serializes_infinite_ranking_sentinels() -> None:
    report = GaugeAnchorReport(
        gauge_mode="auto_scdm",
        resolved_norb_fix_list=[[0], [1]],
        selections=[],
        metric={"type": "orthonormal"},
        state_selection_quality={"status": "ok"},
        gauge_anchor_quality={"sigma_min": 1.0, "condition_number": 1.0},
        symmetry_closure_quality={
            "status": "validated",
            "selected_candidate_id": "frame-1",
            "candidate_rankings": [
                {
                    "candidate_id": "frame-1",
                    # Missing optional phase/support metrics use infinity only
                    # as an internal ordering sentinel.
                    "sort_key": [0, float("inf"), float("inf"), 1, 0, 0.003, "frame-1"],
                }
            ],
            "metrics": [
                {
                    "candidate_id": "frame-1",
                    "exactification_distance_by_op": {"C3": 0.003},
                    "support_off_by_op": {"C3": 0.005},
                    "metadata": {"status": "evaluated"},
                }
            ],
        },
        warnings=[],
    )
    context = SimpleNamespace(
        config=SimpleNamespace(valley="K1", spin="all"),
        mode="k1",
        nlow_state_list=[[0], [1]],
        q_model1=np.zeros((1, 2)),
        q_model2=np.zeros((1, 2)),
        operation_payloads={
            "C3": {
                "action": SimpleNamespace(matrix=np.eye(2, dtype=np.complex128)),
                "entry": {"_pairs": [(0, 0)]},
                "antiunitary": False,
            }
        },
    )

    evidence = cli._non_gamma_symmetry_validation_evidence(
        SimpleNamespace(gauge_report=report, context=context)
    )

    json.dumps(evidence.report, sort_keys=True, allow_nan=False)
    assert evidence.report["symmetry_closure_quality"]["candidate_rankings"][0][
        "sort_key"
    ][1:3] == ["Infinity", "Infinity"]
    assert len(evidence.certificate_hash) == 64


def test_non_gamma_selection_uses_anchor_quality_without_full_h_eigensystem(
    monkeypatch,
) -> None:
    candidates = (
        FrozenBandCandidate("a-bad", ((1,), ()), ()),
        FrozenBandCandidate("b-good", ((0, 1), ()), ()),
    )
    hamk = np.diag([0.0, 1.0]).astype(np.complex128)[np.newaxis, ...]

    def fake_anchors(_reference_ham, *_args, nlow_state_list, **_kwargs):
        sigma_min = 0.25 if nlow_state_list[0] == [1] else 0.90
        return [list(row) for row in nlow_state_list], GaugeAnchorReport(
            gauge_mode="auto_scdm",
            resolved_norb_fix_list=[list(row) for row in nlow_state_list],
            selections=[],
            metric={"type": "orthonormal"},
            state_selection_quality={"status": "ok"},
            gauge_anchor_quality={
                "sigma_min": sigma_min,
                "condition_number": 1.0 / sigma_min,
            },
            symmetry_closure_quality={"status": "not_evaluated"},
            warnings=[],
        )

    monkeypatch.setattr(cli, "resolve_project_gauge_anchors", fake_anchors)
    def fake_project(*_args, nlow_state_list, **_kwargs):
        dimension = sum(len(row) for row in nlow_state_list)
        # The automatic selector only needs the projected band energies.  A
        # two-item result makes accidental use of candidate eigenvectors fail.
        return (None, np.arange(dimension, dtype=float))

    monkeypatch.setattr(cli, "project_heff_full", fake_project)

    monkeypatch.setattr(
        cli,
        "_non_gamma_projector_basis",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("automatic K/M selection must not rebuild U_low")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "prepare_symmetry_validated_project_gauge",
        lambda *_args, **_kwargs: SimpleNamespace(gauge_report=object()),
    )
    monkeypatch.setattr(
        cli,
        "_non_gamma_symmetry_validation_evidence",
        lambda _preparation: SimpleNamespace(
            symmetry_residual=0.0,
            symmetry_leakage=0.0,
            certificate_hash="a" * 64,
            input_identity_hash="b" * 64,
            report={"status": "validated"},
        ),
    )
    monkeypatch.setattr(
        np.linalg,
        "eigh",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("automatic K/M selection must not diagonalize full H")
        ),
    )

    result = cli._evaluate_non_gamma_automatic_candidates(
        cfg_path="case.yaml",
        cfg={"material": {"efermi": -0.1}},
        candidates=candidates,
        hamk=hamk,
        q1=np.zeros((1, 2)),
        q2=np.zeros((1, 2)),
        num_layer_list=[1, 1],
        orb0=1,
        num_orb_per_layer_list=[[1], [1]],
        spin="all",
        mode="K1",
        method="first_order",
        e_ref=None,
        reference_rows=[np.array([0.0, 1.0])],
        reference_k_index=0,
        selection_cfg={
            "validation_indices": [0],
            "validation_bands": 1,
            "efermi": -0.1,
            "target_degeneracy_tolerance_mev": 1.0e-6,
            "thresholds": {
                "band_rms_mev": 1.0,
                "band_max_mev": 1.0,
                "subspace_overlap": 0.5,
                "symmetry_residual": 0.01,
                "symmetry_leakage": 0.01,
            },
        },
        project_cfg={"gauge": "auto", "target": "conduction"},
    )

    assert result.candidate.candidate_id == "b-good"
    overlap_by_id = {metric.candidate_id: metric.subspace_overlap for metric in result.metrics}
    assert overlap_by_id == {
        "a-bad": pytest.approx(0.25),
        "b-good": pytest.approx(0.90),
    }


def test_non_gamma_fixed_window_structurally_rejects_small_candidate(
    monkeypatch,
) -> None:
    candidates = (
        FrozenBandCandidate("small", ((0,), ()), ()),
        FrozenBandCandidate("large", ((0, 1), ()), ()),
    )
    hamk = np.diag([0.0, 1.0, 2.0]).astype(np.complex128)[np.newaxis, ...]
    monkeypatch.setattr(
        cli,
        "resolve_project_gauge_anchors",
        lambda _ham, *_args, nlow_state_list, **_kwargs: (
            [list(row) for row in nlow_state_list],
            GaugeAnchorReport(
                gauge_mode="auto_scdm",
                resolved_norb_fix_list=[list(row) for row in nlow_state_list],
                selections=[],
                metric={"type": "orthonormal"},
                state_selection_quality={"status": "ok"},
                gauge_anchor_quality={"sigma_min": 1.0, "condition_number": 1.0},
                symmetry_closure_quality={"status": "not_evaluated"},
                warnings=[],
            ),
        ),
    )

    def fake_project(*_args, nlow_state_list, **_kwargs):
        dimension = sum(len(row) for row in nlow_state_list)
        return (
            np.zeros((dimension, dimension), dtype=np.complex128),
            np.arange(dimension, dtype=float),
            np.eye(dimension, dtype=np.complex128),
            None,
        )

    monkeypatch.setattr(cli, "project_heff_full", fake_project)
    monkeypatch.setattr(
        cli,
        "_non_gamma_projector_basis",
        lambda *_args, nlow_state_list, **_kwargs: np.eye(
            3, sum(len(row) for row in nlow_state_list), dtype=np.complex128
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "prepare_symmetry_validated_project_gauge",
        lambda *_args, **_kwargs: SimpleNamespace(gauge_report=object()),
    )
    monkeypatch.setattr(
        cli,
        "_non_gamma_symmetry_validation_evidence",
        lambda _preparation: SimpleNamespace(
            symmetry_residual=0.0,
            symmetry_leakage=0.0,
            certificate_hash="a" * 64,
            input_identity_hash="b" * 64,
            report={"status": "validated"},
        ),
    )

    result = cli._evaluate_non_gamma_automatic_candidates(
        cfg_path="case.yaml",
        cfg={"material": {"efermi": -1.0}},
        candidates=candidates,
        hamk=hamk,
        q1=np.zeros((1, 2)),
        q2=np.zeros((1, 2)),
        num_layer_list=[1, 1],
        orb0=1,
        num_orb_per_layer_list=[[1], [1]],
        spin="all",
        mode="K1",
        method="first_order",
        e_ref=None,
        reference_rows=[np.array([0.0, 1.0, 2.0])],
        reference_k_index=0,
        selection_cfg={
            "validation_indices": [0],
            "validation_bands": 2,
            "efermi": -1.0,
            "target_degeneracy_tolerance_mev": 1.0e-6,
            "thresholds": {
                "band_rms_mev": 1.0,
                "band_max_mev": 1.0,
                "subspace_overlap": 0.5,
                "symmetry_residual": 0.01,
                "symmetry_leakage": 0.01,
            },
        },
        project_cfg={"gauge": "auto", "target": "conduction"},
    )

    rejected = next(metric for metric in result.metrics if metric.candidate_id == "small")
    assert rejected.structural_failure is not None
    assert CandidateRejectionReason.INSUFFICIENT_CANDIDATE_BANDS.value in rejected.structural_failure
    assert result.candidate.candidate_id == "large"


def test_non_gamma_fixed_window_propagates_typed_boundary_degeneracy(
    monkeypatch,
) -> None:
    candidate = FrozenBandCandidate("one", ((0,), ()), ())
    monkeypatch.setattr(
        cli,
        "resolve_project_gauge_anchors",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("candidate loop must not start for an ambiguous target")
        ),
    )

    with pytest.raises(CandidateRejected) as caught:
        cli._evaluate_non_gamma_automatic_candidates(
            cfg_path="case.yaml",
            cfg={"material": {"efermi": -1.0}},
            candidates=(candidate,),
            hamk=np.diag([0.0, 0.0, 1.0]).astype(np.complex128)[np.newaxis, ...],
            q1=np.zeros((1, 2)),
            q2=np.zeros((1, 2)),
            num_layer_list=[1, 1],
            orb0=1,
            num_orb_per_layer_list=[[1], [1]],
            spin="all",
            mode="K1",
            method="first_order",
            e_ref=None,
            reference_rows=[np.array([0.0, 0.0, 1.0])],
            reference_k_index=0,
            selection_cfg={
                "validation_indices": [0],
                "validation_bands": 1,
                "efermi": -1.0,
                "target_degeneracy_tolerance_mev": 1.0e-3,
            },
            project_cfg={"gauge": "auto", "target": "conduction"},
        )

    assert caught.value.reason is CandidateRejectionReason.TARGET_BOUNDARY_DEGENERACY


@pytest.mark.parametrize(
    "error",
    [ValueError("bad value"), IndexError("bad index"), RuntimeError("boom"), OSError("io")],
)
def test_non_gamma_unexpected_candidate_error_propagates(monkeypatch, error) -> None:
    monkeypatch.setattr(
        cli,
        "resolve_project_gauge_anchors",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(type(error), match=str(error)):
        cli._evaluate_non_gamma_automatic_candidates(
            cfg_path="case.yaml",
            cfg={"material": {"efermi": -1.0}},
            candidates=(FrozenBandCandidate("one", ((0,), ()), ()),),
            hamk=np.diag([0.0, 1.0]).astype(np.complex128)[np.newaxis, ...],
            q1=np.zeros((1, 2)),
            q2=np.zeros((1, 2)),
            num_layer_list=[1, 1],
            orb0=1,
            num_orb_per_layer_list=[[1], [1]],
            spin="all",
            mode="K1",
            method="first_order",
            e_ref=None,
            reference_rows=[np.array([0.0, 1.0])],
            reference_k_index=0,
            selection_cfg={
                "validation_indices": [0],
                "validation_bands": 1,
                "efermi": -1.0,
                "target_degeneracy_tolerance_mev": 1.0e-6,
            },
            project_cfg={"gauge": "auto", "target": "conduction"},
        )


def test_non_gamma_sparse_action_identity_never_materializes_dense() -> None:
    class NoDenseCSR(sp.csr_matrix):
        def toarray(self, *args, **kwargs):
            raise AssertionError("sparse action must not be materialized")

    report = GaugeAnchorReport(
        gauge_mode="auto_scdm",
        resolved_norb_fix_list=[[0], [1]],
        selections=[],
        metric={"type": "orthonormal"},
        state_selection_quality={"status": "ok"},
        gauge_anchor_quality={"sigma_min": 1.0, "condition_number": 1.0},
        symmetry_closure_quality={
            "status": "validated",
            "selected_candidate_id": "frame-1",
            "metrics": [
                {
                    "candidate_id": "frame-1",
                    "exactification_distance_by_op": {"C3": 0.003},
                    "support_off_by_op": {"C3": 0.005},
                    "metadata": {"status": "evaluated"},
                }
            ],
        },
        warnings=[],
    )
    sparse_action = NoDenseCSR(sp.eye(100_000, format="csr", dtype=np.complex128))
    context = SimpleNamespace(
        config=SimpleNamespace(valley="K1", spin="all"),
        mode="k1",
        nlow_state_list=[[0], [1]],
        q_model1=np.zeros((1, 2)),
        q_model2=np.zeros((1, 2)),
        operation_payloads={
            "C3": {
                "action": SimpleNamespace(matrix=sparse_action),
                "entry": {"_pairs": [(0, 0)]},
                "antiunitary": False,
            }
        },
    )

    evidence = cli._non_gamma_symmetry_validation_evidence(
        SimpleNamespace(gauge_report=report, context=context)
    )

    assert len(evidence.input_identity_hash) == 64


def test_non_gamma_auto_rejects_materializer_dimension_mismatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_tiny_project_config(
        tmp_path,
        {
            "mode": "K1",
            "workers": 1,
            "downfold_method": "first_order",
            "out_dir": "project",
            "selection": {"mode": "auto"},
        },
    )
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["material"]["efermi"] = 0.0
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    metric = CandidateMetrics("small", 1, 0.0, 0.0, 1.0, None, None)
    thresholds = SelectionThresholds(1.0, 1.0, 0.5, 0.01, 0.01)
    candidate = FrozenBandCandidate("small", ((0,), ()), ())
    expected_evidence = SimpleNamespace(
        certificate_hash="a" * 64,
        input_identity_hash="b" * 64,
    )
    monkeypatch.setattr(
        cli,
        "_prepare_non_gamma_automatic_selection",
        lambda *_args, **_kwargs: SimpleNamespace(
            candidate=candidate,
            decision=select_projection_candidate(
                (metric,), thresholds, allow_pending_symmetry=True
            ),
            metrics=(metric,),
            thresholds=thresholds,
            q_count=1,
            symmetry_evidence=expected_evidence,
        ),
    )
    monkeypatch.setattr(
        cli,
        "_cmd_project_from_config_impl",
        lambda *_args, **_kwargs: SimpleNamespace(
            candidate_id="materialized",
            candidate_dimension=2,
            candidate_fiber_rank=1,
            q_count=1,
            resolved_nlow_state_list=((0,), ()),
            basis_handoff_hash="1" * 64,
            authoritative_heff_hash="2" * 64,
            heff_k_indices_hash="3" * 64,
            certification_hash=None,
            symmetry_certificate_hash=None,
            symmetry_input_identity_hash=None,
        ),
    )

    with pytest.raises(ValueError, match="materialized.*dimension"):
        cli.cmd_project_from_config(str(cfg_path))


def test_inspect_and_explicit_project_share_input_but_only_project_finalizes(
    monkeypatch,
    tmp_path: Path,
) -> None:
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
    cfg_path = _write_tiny_project_config(tmp_path, _explicit_project_config())

    inspect = cli.cmd_plot_from_config(str(cfg_path))
    project = cli.cmd_project_from_config(str(cfg_path))

    assert inspect.status is CertificationStatus.PENDING
    assert inspect.identity is None
    assert project.status is CertificationStatus.UNVERIFIED_OVERRIDE
    assert project.identity is not None
    assert (
        inspect.selection_input.selection_input_identity_hash
        == project.selection_input.selection_input_identity_hash
    )
    marker = load_selection_artifact(tmp_path / "project" / "selection_artifact.json")
    assert marker == project.artifact


def test_active_indices_is_recorded_as_explicit_unverified_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cli,
        "plot_eigs_scatter",
        lambda *_args, out, **_kwargs: Path(out).write_text("plot", encoding="utf-8"),
    )
    cfg_path = _write_tiny_project_config(tmp_path, _explicit_project_config())
    monkeypatch.setattr(
        cli,
        "_cmd_project_from_config_impl",
        lambda *_args, **_kwargs: cli._ExplicitProjectSelectionMaterialization(
            candidate_id="active-indices-0-1",
            candidate_dimension=2,
            basis_handoff_hash="1" * 64,
            authoritative_heff_hash="2" * 64,
            heff_k_indices_hash="3" * 64,
        ),
    )

    result = cli.cmd_project_from_config(str(cfg_path), {"active_indices": "0,1"})

    assert result.status is CertificationStatus.UNVERIFIED_OVERRIDE
    assert result.selection_input.selection_mode == "explicit"
    assert result.identity is not None
    assert result.identity.certification_evidence is None


def test_later_project_failure_leaves_pending_instead_of_previous_final(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cli,
        "plot_eigs_scatter",
        lambda *_args, out, **_kwargs: Path(out).write_text("plot", encoding="utf-8"),
    )
    cfg_path = _write_tiny_project_config(tmp_path, _explicit_project_config())
    cli.cmd_project_from_config(str(cfg_path))

    def fail_projection(*_args, **_kwargs):
        raise RuntimeError("injected projection failure")

    monkeypatch.setattr(cli, "project_heff_full", fail_projection)
    with pytest.raises(RuntimeError, match="injected projection failure"):
        cli.cmd_project_from_config(str(cfg_path))

    marker = load_selection_artifact(tmp_path / "project" / "selection_artifact.json")
    assert marker.certification_status is CertificationStatus.PENDING


def test_keyboard_interrupt_releases_pending_selection_lock(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_tiny_project_config(tmp_path, _explicit_project_config())
    real_begin = cli.begin_case_selection
    retained_sessions = []

    def retain_session(**kwargs):
        session = real_begin(**kwargs)
        retained_sessions.append(session)
        return session

    def interrupt_project(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "begin_case_selection", retain_session)
    monkeypatch.setattr(cli, "_cmd_project_from_config_impl", interrupt_project)
    with pytest.raises(KeyboardInterrupt):
        cli.cmd_project_from_config(str(cfg_path))

    project_dir = tmp_path / "project"
    marker = load_selection_artifact(project_dir / "selection_artifact.json")
    assert marker.certification_status is CertificationStatus.PENDING
    assert retained_sessions
    with (project_dir / ".selection-artifact.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def test_interrupt_at_resolver_handoff_releases_pending_selection_lock(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_tiny_project_config(tmp_path, _explicit_project_config())
    real_begin = cli.begin_case_selection
    retained_sessions = []

    def retain_session(**kwargs):
        session = real_begin(**kwargs)
        retained_sessions.append(session)
        return session

    monkeypatch.setattr(cli, "begin_case_selection", retain_session)
    monkeypatch.setattr(
        cli,
        "_cmd_project_from_config_impl",
        lambda *_args, **_kwargs: cli._ExplicitProjectSelectionMaterialization(
            candidate_id="explicit-resolver-interrupt",
            candidate_dimension=2,
            basis_handoff_hash="1" * 64,
            authoritative_heff_hash="2" * 64,
            heff_k_indices_hash="3" * 64,
        ),
    )

    def interrupt_resolver(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "resolve_case_selection", interrupt_resolver)
    with pytest.raises(KeyboardInterrupt):
        cli.cmd_project_from_config(str(cfg_path))

    project_dir = tmp_path / "project"
    assert retained_sessions
    with (project_dir / ".selection-artifact.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def test_project_auto_requires_source_symmetry(tmp_path: Path) -> None:
    cfg_path = _write_tiny_project_config(
        tmp_path,
        {
            "mode": "K1",
            "workers": 1,
            "downfold_method": "first_order",
            "out_dir": "project",
            "nlow_state_list": [[0], [0]],
            "gauge": {"method": "auto_scdm", "validate_with_symmetry": True},
        },
    )

    with pytest.raises(ValueError, match="auto gauge.*symm.tapw_symmetry_dir"):
        cli.cmd_project_from_config(str(cfg_path))

    assert not (tmp_path / "project" / "heff.npy").exists()


def test_symmetry_gauge_resolver_needs_no_project_artifacts_and_writes_nothing(tmp_path: Path) -> None:
    cfg_path = _write_tiny_project_config_with_symm(
        tmp_path,
        {
            "mode": "K1",
            "workers": 1,
            "downfold_method": "first_order",
            "out_dir": "project",
            "nlow_state_list": [[0], [0]],
            "gauge": "auto",
        },
        {
            "enable": True,
            "valley": "K1",
            "spin": "up",
            "tapw_symmetry_dir": "tapw_symmetry",
            "output_dir": "symm",
            "operations": ["C3"],
            "tolerance": 1.0e-8,
        },
    )
    _write_tiny_tapw_c3_source(tmp_path / "tapw_symmetry")

    report = resolve_symmetry_validated_project_gauge(str(cfg_path))

    assert report.gauge_mode == "auto_scdm"
    assert report.symmetry_closure_quality["status"] == "validated"
    assert not (tmp_path / "project").exists()
    assert not (tmp_path / "symm").exists()


def test_project_auto_uses_symmetry_validated_resolved_anchors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli,
        "plot_eigs_scatter",
        lambda *_args, out, **_kwargs: Path(out).write_text("plot", encoding="utf-8"),
    )
    selected = [[[[1, 1.0]]], [[[1, 1.0]]]]

    report = {
        "gauge_mode": "auto_scdm",
        "resolved_norb_fix_list": selected,
        "selections": [],
        "metric": {"type": "orthonormal", "basis_is_orthonormal": True},
        "state_selection_quality": {"status": "not_evaluated"},
        "gauge_anchor_quality": {"status": "ok", "sigma_min": 1.0, "condition_number": 1.0},
        "symmetry_closure_quality": {
            "status": "validated",
            "selected_candidate_id": "test_candidate",
        },
        "warnings": [],
    }

    calls: list[str] = []
    resolved_project_configs: list[dict] = []

    def fake_resolver(cfg_path: str, *, project_config=None):
        calls.append(cfg_path)
        resolved_project_configs.append(dict(project_config or {}))
        return GaugeAnchorReport(**report)

    def fail_full_symm(*_args, **_kwargs):
        raise AssertionError("kp project must not run the full kp symm writer")

    monkeypatch.setattr(cli, "resolve_symmetry_validated_project_gauge", fake_resolver, raising=False)
    monkeypatch.setattr(cli, "run_symmetry_projection_from_config", fail_full_symm)

    cfg_path = _write_tiny_project_config_with_symm(
        tmp_path,
        {
            "mode": "K1",
            "workers": 1,
            "downfold_method": "first_order",
            "out_dir": "project",
            "nlow_state_list": [[0], [0]],
            "gauge": {"method": "auto_scdm", "validate_with_symmetry": True},
        },
        {
            "enable": True,
            "tapw_symmetry_dir": "tapw_symmetry",
            "output_dir": "symm",
            "operations": ["TR"],
        },
    )
    (tmp_path / "tapw_symmetry").mkdir()

    cli.cmd_project_from_config(str(cfg_path), {"k_indices": "0"})

    assert calls == [str(cfg_path)]
    assert resolved_project_configs[0]["k_indices"] == "0"
    with np.load(tmp_path / "project" / "basis.npz", allow_pickle=True) as payload:
        assert payload["norb_fix_list"].tolist() == selected
    assert "symmetry_closure_quality: `validated`" in (tmp_path / "project" / "basis.md").read_text(encoding="utf-8")


def test_project_auto_does_not_trust_unbound_cached_symmetry_basis(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli,
        "plot_eigs_scatter",
        lambda *_args, out, **_kwargs: Path(out).write_text("plot", encoding="utf-8"),
    )
    stale_selected = [[[[1, 1.0]]], [[[1, 1.0]]]]
    resolved_selected = [[[[0, 1.0]]], [[[0, 1.0]]]]
    symm_dir = tmp_path / "symm"
    symm_dir.mkdir()
    report = {
        "gauge_mode": "auto_scdm",
        "resolved_norb_fix_list": stale_selected,
        "selections": [],
        "metric": {"type": "orthonormal", "basis_is_orthonormal": True},
        "state_selection_quality": {"status": "not_evaluated"},
        "gauge_anchor_quality": {"status": "ok", "sigma_min": 1.0, "condition_number": 1.0},
        "symmetry_closure_quality": {
            "status": "validated",
            "selected_candidate_id": "cached_candidate",
        },
        "warnings": [],
    }
    _write_cached_symmetry_basis(symm_dir, report)

    resolved_report = dict(report)
    resolved_report["resolved_norb_fix_list"] = resolved_selected
    resolved_report["symmetry_closure_quality"] = {
        "status": "validated",
        "selected_candidate_id": "fresh_candidate",
    }

    def fake_resolver(_cfg_path: str, *, project_config=None):
        return GaugeAnchorReport(**resolved_report)

    def fail_symm(*_args, **_kwargs):
        raise AssertionError("kp project must not run the full kp symm writer")

    monkeypatch.setattr(cli, "resolve_symmetry_validated_project_gauge", fake_resolver, raising=False)
    monkeypatch.setattr(cli, "run_symmetry_projection_from_config", fail_symm)

    cfg_path = _write_tiny_project_config_with_symm(
        tmp_path,
        {
            "mode": "K1",
            "workers": 1,
            "downfold_method": "first_order",
            "out_dir": "project",
            "nlow_state_list": [[0], [0]],
            "gauge": "auto",
        },
        {
            "enable": True,
            "tapw_symmetry_dir": "tapw_symmetry",
            "output_dir": "symm",
            "operations": ["TR"],
        },
    )
    (tmp_path / "tapw_symmetry").mkdir()

    cli.cmd_project_from_config(str(cfg_path))

    with np.load(tmp_path / "project" / "basis.npz", allow_pickle=True) as basis_payload:
        assert basis_payload["norb_fix_list"].tolist() == resolved_selected
    assert "symmetry_closure_quality: `validated`" in (tmp_path / "project" / "basis.md").read_text(encoding="utf-8")


def test_project_auto_resolves_symmetry_without_running_full_symm(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli,
        "plot_eigs_scatter",
        lambda *_args, out, **_kwargs: Path(out).write_text("plot", encoding="utf-8"),
    )

    selected = [[[[1, 1.0]]], [[[1, 1.0]]]]
    report = {
        "gauge_mode": "auto_scdm",
        "resolved_norb_fix_list": selected,
        "selections": [],
        "metric": {"type": "orthonormal", "basis_is_orthonormal": True},
        "state_selection_quality": {"status": "not_evaluated"},
        "gauge_anchor_quality": {"status": "ok", "sigma_min": 1.0, "condition_number": 1.0},
        "symmetry_closure_quality": {
            "status": "validated",
            "selected_candidate_id": "resolved_candidate",
        },
        "warnings": [],
    }

    def fake_resolver(_cfg_path: str, *, project_config=None):
        return GaugeAnchorReport(**report)

    def fail_symm(*_args, **_kwargs):
        raise AssertionError("kp project must not run the full kp symm writer")

    monkeypatch.setattr(cli, "resolve_symmetry_validated_project_gauge", fake_resolver, raising=False)
    monkeypatch.setattr(cli, "run_symmetry_projection_from_config", fail_symm)

    cfg_path = _write_tiny_project_config_with_symm(
        tmp_path,
        {
            "mode": "K1",
            "workers": 1,
            "downfold_method": "first_order",
            "out_dir": "project",
            "nlow_state_list": [[0], [0]],
            "gauge": "auto",
        },
        {
            "enable": True,
            "tapw_symmetry_dir": "tapw_symmetry",
            "output_dir": "symm",
            "operations": ["TR"],
        },
    )
    (tmp_path / "tapw_symmetry").mkdir()

    cli.cmd_project_from_config(str(cfg_path))

    with np.load(tmp_path / "project" / "basis.npz", allow_pickle=True) as basis_payload:
        assert basis_payload["norb_fix_list"].tolist() == selected
    assert "symmetry_closure_quality: `validated`" in (tmp_path / "project" / "basis.md").read_text(encoding="utf-8")


def test_project_rejects_manual_norb_fix_with_auto_gauge(tmp_path: Path) -> None:
    cfg_path = _write_tiny_project_config(
        tmp_path,
        {
            "mode": "K1",
            "workers": 1,
            "downfold_method": "first_order",
            "out_dir": "project",
            "nlow_state_list": [[0], [0]],
            "norb_fix_list": [[[[0, 1.0]]], [[[0, 1.0]]]],
            "gauge": "auto",
        },
    )

    with np.testing.assert_raises_regex(ValueError, "manual norb_fix_list.*gauge"):
        cli.cmd_project_from_config(str(cfg_path))


def test_project_requires_auto_gauge_or_manual_norb_fix_list(tmp_path: Path) -> None:
    cfg_path = _write_tiny_project_config(
        tmp_path,
        {
            "mode": "K1",
            "workers": 1,
            "downfold_method": "first_order",
            "out_dir": "project",
            "nlow_state_list": [[0], [0]],
        },
    )

    with pytest.raises(ValueError, match="auto gauge.*symm.tapw_symmetry_dir"):
        cli.cmd_project_from_config(str(cfg_path))

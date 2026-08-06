from __future__ import annotations

from dataclasses import replace
import re
import threading

import numpy as np
import pytest

import kp.gamma_auto_producer as producer_mod
from kp.blocks import GammaRowLayout
from kp.config.case import normalize_case_config
from kp.gamma_auto_producer import (
    GammaAutomaticProducerInputs,
    GammaAutomaticSelectionConfig,
    GammaRawOperationSpec,
    evaluate_gamma_automatic_selection,
    prepare_gamma_automatic_selection,
)
from kp.identity import hash_array
from kp.low_energy_selection import (
    CandidateSelectionError,
    CandidateSelectionFailureCode,
    SelectionThresholds,
)
from kp.symmetry.joint_exactification import (
    MagneticGenerator,
    MagneticPresentation,
    MagneticRelation,
)


def _normalized_project(
    *,
    selection: dict[str, object],
    downfold_method: str = "first_order",
    e_ref: float | None = None,
) -> dict[str, object]:
    project: dict[str, object] = {
        "mode": "Gamma",
        "target": "valence",
        "efermi": 0.0,
        "workers": 1,
        "downfold_method": downfold_method,
        "selection": selection,
    }
    if e_ref is not None:
        project["e_ref"] = e_ref
    normalized = normalize_case_config(
        {
            "case": {
                "profile": "Gamma",
                "q_shell": "q00",
                "output_root": "outputs",
            },
            "valley": "Gamma",
            "material": {"spin": "all", "num_layer_list": [1, 1]},
            "project": project,
        },
        config_path="/tmp/gamma-bare-auto.yaml",
    )
    return dict(normalized["project"])


def _automatic_config(
    *,
    max_dimension: int,
    degeneracy_tolerance: float = 1.0e-3,
) -> GammaAutomaticSelectionConfig:
    project = _normalized_project(
        selection={
            "mode": "auto",
            "max_dimension": max_dimension,
            "degeneracy_tolerance": degeneracy_tolerance,
            "validation_indices": [1],
            "validation_bands": 2,
        }
    )
    return GammaAutomaticSelectionConfig.from_normalized_project_config(project)


def _identity_presentation() -> MagneticPresentation:
    return MagneticPresentation(
        generators=(MagneticGenerator("E", False),),
        relations=(
            MagneticRelation(
                "E^2",
                lhs=("E", "E"),
                rhs=(),
                central_phase=1.0,
            ),
        ),
        central_phases=(1.0,),
        source="gamma_bare_auto_selection_test",
    )


def _inputs(
    *,
    reference_energies: tuple[float, float, float, float] = (
        -0.30,
        -0.1000,
        -0.0995,
        0.20,
    ),
    qset: np.ndarray | None = None,
) -> GammaAutomaticProducerInputs:
    # Neither the representative k nor the representative Q is row zero.
    kpoints = np.asarray(
        [[0.40, 0.0], [0.02, 0.0], [0.20, 0.0]],
        dtype=np.float64,
    )
    if qset is None:
        qset = np.asarray([[1.0, 0.0], [0.10, 0.0]], dtype=np.float64)
    else:
        qset = np.asarray(qset, dtype=np.float64)
    source_basis_hash = hash_array(np.arange(8, dtype=np.int64))
    layout = GammaRowLayout.build(
        qsets=(qset, qset),
        num_layer_list=(1, 1),
        num_orb_per_layer_list=((1,), (1,)),
        spin_convention="all",
        source_basis_hash=source_basis_hash,
    )

    hamiltonians: list[np.ndarray] = []
    q1_energies_by_k = (
        (-0.30, -0.100, -0.050, 0.20),
        reference_energies,
        (-0.1004, -0.1002, -0.1000, 0.20),
    )
    for shift, q1_energies in zip(
        (0.03, 0.00, -0.02),
        q1_energies_by_k,
        strict=True,
    ):
        matrix = np.zeros(
            (layout.full_dimension, layout.full_dimension),
            dtype=np.complex128,
        )
        q0_rows = layout.same_q_full_rows(0)
        q1_rows = layout.same_q_full_rows(1)
        matrix[np.ix_(q0_rows, q0_rows)] = np.diag(
            np.asarray([-0.80, -0.70, 0.40, 0.50]) + shift
        )
        matrix[np.ix_(q1_rows, q1_rows)] = np.diag(
            np.asarray(q1_energies) + shift
        )
        hamiltonians.append(matrix)

    dimension = layout.full_dimension
    return GammaAutomaticProducerInputs(
        source_hamiltonians=np.stack(hamiltonians, axis=0),
        k_indices=(0, 1, 2),
        kpoints=kpoints,
        qsets=(qset, qset),
        num_layer_list=(1, 1),
        num_orb_per_layer_list=((1,), (1,)),
        tapw_source_basis_hash=source_basis_hash,
        operations=(
            GammaRawOperationSpec(
                name="E",
                full_action=np.eye(dimension, dtype=np.complex128),
                antiunitary=False,
                q_permutations=((0, 1), (0, 1)),
                sector_map=(0, 1),
                pairs=((0, 0), (1, 1), (2, 2)),
            ),
        ),
        presentation=_identity_presentation(),
    )


def test_bare_public_gamma_auto_materializes_a_versioned_universal_policy() -> None:
    project = _normalized_project(
        selection={"mode": "auto"},
        downfold_method="fixed_schur",
        e_ref=-0.75,
    )

    config = GammaAutomaticSelectionConfig.from_normalized_project_config(project)
    policy = config.policy_payload()

    assert project["selection"] == {"mode": "auto"}
    assert re.fullmatch(r"kp\.gamma-auto-[a-z0-9-]+\.v[1-9][0-9]*", policy["schema"])
    assert "candidate_seed_band_indices" not in policy
    assert "reference_k_index" not in policy
    assert config.max_dimension > 0
    assert config.degeneracy_tolerance >= 0.0
    assert config.selection_thresholds.band_rms_mev == pytest.approx(3.0)
    assert config.selection_thresholds.band_max_mev == pytest.approx(3.0)
    assert config.target_window_spec.edge == "valence"
    assert config.target_window_spec.energy_reference_ev == pytest.approx(0.0)
    assert config.downfold.options.method == "fixed_schur"
    assert config.downfold.options.e_ref == pytest.approx(-0.75)


def test_public_max_dimension_is_capped_by_actual_local_dimension() -> None:
    project = _normalized_project(
        selection={
            "mode": "auto",
            "degeneracy_tolerance": 1.0e-3,
            "validation_indices": [1],
            "validation_bands": 2,
        }
    )
    config = GammaAutomaticSelectionConfig.from_normalized_project_config(project)

    preparation = prepare_gamma_automatic_selection(_inputs(), config)

    assert config.max_dimension > preparation.layout.same_q_dimension
    assert preparation.effective_max_dimension == preparation.layout.same_q_dimension
    assert preparation.candidate_seed_band_indices == ((1, 2), (0, 1, 2))


def test_public_gamma_auto_rejects_hand_written_candidate_seeds() -> None:
    project = _normalized_project(
        selection={
            "mode": "auto",
            "candidate_seed_band_indices": [[0, 1]],
        }
    )

    with pytest.raises(ValueError, match="candidate_seed_band_indices"):
        GammaAutomaticSelectionConfig.from_normalized_project_config(project)


def test_reference_point_and_complete_cluster_envelope_come_from_actual_case() -> None:
    preparation = prepare_gamma_automatic_selection(
        _inputs(),
        _automatic_config(max_dimension=3),
    )

    assert preparation.reference_point.k_index == 1
    assert preparation.reference_point.q_indices == (1, 1)
    assert preparation.candidate_seed_band_indices == (
        (1, 2),
        (0, 1, 2),
    )
    assert tuple(map(len, preparation.candidate_seed_band_indices)) == (2, 3)
    assert all(
        len(seed) <= preparation.config.max_dimension
        for seed in preparation.candidate_seed_band_indices
    )


def test_reference_q_requires_a_unique_minimum_norm() -> None:
    tied_qset = np.asarray([[-0.10, 0.0], [0.10, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="unique.*minimum|minimum.*unique"):
        prepare_gamma_automatic_selection(
            _inputs(qset=tied_qset),
            _automatic_config(max_dimension=3),
        )


def test_max_dimension_smaller_than_first_complete_cluster_fails_typed() -> None:
    with pytest.raises(CandidateSelectionError) as captured:
        prepare_gamma_automatic_selection(
            _inputs(),
            _automatic_config(max_dimension=1),
        )

    assert captured.value.failure_code is CandidateSelectionFailureCode.NO_CANDIDATES
    assert "max_dimension" in str(captured.value)


def test_generated_candidate_envelope_is_bound_into_selection_policy_identity() -> None:
    clustered = prepare_gamma_automatic_selection(
        _inputs(),
        _automatic_config(max_dimension=3),
    )
    split = prepare_gamma_automatic_selection(
        _inputs(reference_energies=(-0.30, -0.1000, -0.0800, 0.20)),
        _automatic_config(max_dimension=3),
    )

    assert clustered.config.policy_payload() == split.config.policy_payload()
    assert clustered.candidate_seed_band_indices != split.candidate_seed_band_indices
    assert (
        clustered.selection_input.selection_policy_hash
        != split.selection_input.selection_policy_hash
    )


def test_public_gamma_auto_materializes_only_the_frozen_reference_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation = prepare_gamma_automatic_selection(
        _inputs(),
        replace(
            _automatic_config(max_dimension=3),
            selection_thresholds=SelectionThresholds(
                band_rms_mev=1000.0,
                band_max_mev=1000.0,
                subspace_overlap=0.01,
                symmetry_residual=1.0,
                symmetry_leakage=1.0,
            ),
        ),
    )
    assert len(preparation.candidate_seed_band_indices) == 2
    materialized: list[tuple[int, ...]] = []
    real_materialize = producer_mod._evaluate_candidate

    def record_materialization(*, seed, **kwargs):
        materialized.append(tuple(seed))
        return real_materialize(seed=seed, **kwargs)

    monkeypatch.setattr(
        producer_mod,
        "_evaluate_candidate",
        record_materialization,
    )

    result = evaluate_gamma_automatic_selection(preparation)

    assert materialized == [result.handoff.anchor_spec.joint_band_indices]


def test_local_gamma_eigensystems_parallelize_over_k_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    layout = GammaRowLayout.build(
        qsets=inputs.qsets,
        num_layer_list=inputs.num_layer_list,
        num_orb_per_layer_list=inputs.num_orb_per_layer_list,
        spin_convention="all",
        source_basis_hash=inputs.tapw_source_basis_hash,
    )
    real_eigh = np.linalg.eigh
    rendezvous = threading.Barrier(len(inputs.k_indices), timeout=5.0)
    worker_threads: set[int] = set()

    def synchronized_eigh(matrix: np.ndarray):
        worker_threads.add(threading.get_ident())
        rendezvous.wait()
        return real_eigh(matrix)

    monkeypatch.setattr(producer_mod.np.linalg, "eigh", synchronized_eigh)

    values, _vectors = producer_mod._local_eigensystems(
        inputs.source_hamiltonians,
        layout,
        hermiticity_tolerance=1.0e-12,
        workers=len(inputs.k_indices),
    )

    assert len(values) == len(inputs.k_indices)
    assert len(worker_threads) == len(inputs.k_indices)

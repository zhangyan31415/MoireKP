from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kp.model import candidates


def _scan_inputs(monkeypatch, tmp_path: Path):
    heff = np.zeros((2, 1, 1), dtype=complex)
    config = SimpleNamespace(
        fit_selection_metadata={
            "mode": "auto_low_energy",
            "fit_candidate_sets": [
                {"name": "first", "indices": [0]},
                {"name": "second", "indices": [1]},
            ],
        },
        raw={"fit": {"disable_residual_augmented_fit": True}},
        n_orb=(1, 0), harmonics_config={}, heff_eig_file=None,
        band_indices=None, band_slice=None, band_plot_config={},
    )
    moire = SimpleNamespace(Q_set1=np.zeros((1, 2)), Q_set2=np.empty((0, 2)))
    monkeypatch.setattr(candidates.configuration, "_load_kpoints", lambda _cfg: np.zeros((2, 2)))
    monkeypatch.setattr(candidates.configuration, "_load_heff_in_model_basis", lambda _cfg: heff)
    monkeypatch.setattr(candidates.validation, "_current_heff_support_hamiltonians", lambda *_args, **_kwargs: heff)
    return dict(
        moire_config=moire, model_config=config, output_dir=tmp_path,
        log_path=tmp_path / "selected.log", verbose=False, progress=False,
    )


@pytest.mark.parametrize("error", [AttributeError("missing action"), ValueError("metadata mismatch")])
def test_candidate_scan_propagates_program_and_metadata_errors(monkeypatch, tmp_path, error):
    inputs = _scan_inputs(monkeypatch, tmp_path)
    visited = []

    def execute(**kwargs):
        visited.append(kwargs["name"])
        raise error

    monkeypatch.setattr(candidates, "_execute_fit_candidate", execute)
    with pytest.raises(type(error), match=str(error)) as exc:
        candidates._run_auto_low_energy_fit_candidate_scan(**inputs)
    assert exc.value is error
    assert visited == ["first"]


def test_candidate_scan_continues_after_explicit_numerical_rejection(monkeypatch, tmp_path):
    inputs = _scan_inputs(monkeypatch, tmp_path)
    visited = []

    def execute(**kwargs):
        visited.append(kwargs["name"])
        if kwargs["name"] == "first":
            raise candidates.CandidateRejected("declared candidate constraint failed")
        kwargs["record"].update(
            status="ok", plot_rms_mev=1.0, plot_max_mev=2.0,
            all_rms_mev=1.0, all_max_mev=2.0,
        )
        return {}, inputs["moire_config"], inputs["model_config"], tmp_path / "second.log"

    monkeypatch.setattr(candidates, "_execute_fit_candidate", execute)
    monkeypatch.setattr(candidates, "_model_config_for_fit_candidate", lambda *_args, **_kwargs: inputs["model_config"])
    monkeypatch.setattr(candidates, "_moire_config_for_fit_indices", lambda *_args, **_kwargs: inputs["moire_config"])
    monkeypatch.setattr(candidates, "_run_model_pipeline", lambda *_args, **_kwargs: {})
    result, _, _ = candidates._run_auto_low_energy_fit_candidate_scan(**inputs)
    assert visited == ["first", "second"]
    records = result["auto_fit_candidate_scan"]["candidates"]
    assert [row["status"] for row in records] == ["failed", "ok"]
    assert records[0]["error"] == "declared candidate constraint failed"

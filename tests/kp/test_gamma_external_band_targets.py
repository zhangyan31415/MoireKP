from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

import kp.gamma_auto_producer as producer
import kp.gamma_auto_runtime as runtime
from kp.gamma_auto_producer import (
    GammaAutomaticProducerInputs,
    evaluate_gamma_automatic_selection,
    prepare_gamma_automatic_selection,
)
from tests.kp.test_gamma_auto_runtime import _write_packed_gamma_runtime_case
from tests.kp.test_gamma_common_anchor_producer import (
    _common_config,
    _common_inputs,
)


def _full_spectra(inputs: GammaAutomaticProducerInputs) -> np.ndarray:
    return np.stack(
        [np.linalg.eigvalsh(matrix) for matrix in inputs.source_hamiltonians],
        axis=0,
    )


def _with_external_spectra(
    inputs: GammaAutomaticProducerInputs,
    spectra: np.ndarray | None = None,
) -> GammaAutomaticProducerInputs:
    return GammaAutomaticProducerInputs(
        **{
            **inputs.__dict__,
            "external_target_band_spectra": (
                _full_spectra(inputs) if spectra is None else spectra
            ),
        }
    )


def test_runtime_loads_sorted_external_band_spectra_and_passes_them_to_producer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_packed_gamma_runtime_case(tmp_path)
    spectra = np.asarray([[-2.0, -1.0, 0.5, 1.5]], dtype=np.float64)
    np.savetxt(tmp_path / "bands.txt", spectra)
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    payload["material"]["band_file"] = "bands.txt"
    cfg_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    captured: dict[str, object] = {}

    def capture(inputs, _config, *, workers):
        captured["inputs"] = inputs
        captured["workers"] = workers
        return object()

    monkeypatch.setattr(runtime, "prepare_gamma_automatic_selection", capture)

    runtime.prepare_gamma_automatic_runtime(cfg_path)

    actual = captured["inputs"]
    np.testing.assert_array_equal(actual.external_target_band_spectra, spectra)
    assert captured["workers"] == 3


@pytest.mark.parametrize(
    ("rows", "message"),
    (
        ([[0.0, 1.0], [0.1, 1.1]], "k-point coverage"),
        ([[0.0, np.nan]], "finite"),
        ([[1.0, 0.0]], "sorted"),
    ),
)
def test_runtime_rejects_invalid_external_band_spectra(
    tmp_path: Path,
    rows: list[list[float]],
    message: str,
) -> None:
    path = tmp_path / "bands.txt"
    np.savetxt(path, np.asarray(rows, dtype=np.float64))

    with pytest.raises(ValueError, match=message):
        runtime._load_external_target_band_spectra(path, nk=1)


def test_prepare_never_diagonalizes_the_full_source_hamiltonian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _with_external_spectra(_common_inputs())
    real_eigh = np.linalg.eigh

    def guarded_eigh(matrix, *args, **kwargs):
        if np.shape(matrix) == inputs.source_hamiltonians.shape[1:]:
            raise AssertionError("automatic Gamma attempted a full-H eigh")
        return real_eigh(matrix, *args, **kwargs)

    monkeypatch.setattr(producer.np.linalg, "eigh", guarded_eigh)
    preparation = prepare_gamma_automatic_selection(inputs, _common_config())

    assert preparation.target_values.shape == (1, 8)
    assert not hasattr(preparation, "target_vectors")


def test_band_error_uses_external_target_spectra() -> None:
    base = _common_inputs()
    shifted = _full_spectra(base) - 0.1
    inputs = _with_external_spectra(base, shifted)
    config = replace(
        _common_config(),
        selection_thresholds=replace(
            _common_config().selection_thresholds,
            band_rms_mev=200.0,
            band_max_mev=200.0,
            subspace_overlap=0.01,
        ),
    )

    result = evaluate_gamma_automatic_selection(
        prepare_gamma_automatic_selection(inputs, config)
    )

    assert result.candidates[0].band_rms_mev == pytest.approx(100.0)
    assert result.candidates[0].band_max_mev == pytest.approx(100.0)


def test_candidate_overlap_uses_representative_k_and_reference_q_capture() -> None:
    base = _common_inputs()
    rotated_hamiltonians = np.array(base.source_hamiltonians, copy=True)
    cosine = 0.6
    sine = np.sqrt(1.0 - cosine**2)
    rotation = np.asarray(
        [
            [cosine, 0.0, -sine, 0.0],
            [0.0, cosine, 0.0, -sine],
            [sine, 0.0, cosine, 0.0],
            [0.0, sine, 0.0, cosine],
        ],
        dtype=np.complex128,
    )
    layout = producer.GammaRowLayout.build(
        qsets=base.qsets,
        num_layer_list=base.num_layer_list,
        num_orb_per_layer_list=base.num_orb_per_layer_list,
        spin_convention="all",
        source_basis_hash=base.tapw_source_basis_hash,
    )
    for k_position in range(rotated_hamiltonians.shape[0]):
        for q_index in range(layout.q_count):
            if k_position == 0 and q_index == 0:
                continue
            rows = layout.same_q_full_rows(q_index)
            local = rotated_hamiltonians[k_position][np.ix_(rows, rows)]
            rotated_hamiltonians[k_position][np.ix_(rows, rows)] = (
                rotation @ local @ rotation.conj().T
            )
    inputs = GammaAutomaticProducerInputs(
        **{
            **base.__dict__,
            "source_hamiltonians": rotated_hamiltonians,
            "external_target_band_spectra": np.stack(
                [np.linalg.eigvalsh(matrix) for matrix in rotated_hamiltonians]
            ),
        }
    )
    config = replace(
        _common_config(),
        selection_thresholds=replace(
            _common_config().selection_thresholds,
            subspace_overlap=0.01,
        ),
    )

    result = evaluate_gamma_automatic_selection(
        prepare_gamma_automatic_selection(inputs, config)
    )

    assert result.candidates[0].subspace_overlap == pytest.approx(1.0)

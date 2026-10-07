from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kp.model import operator_runtime as operator_runtime_module  # noqa: E402
from kp.model.core import (  # noqa: E402
    ContinuumModel,
    ContinuumModelBuilder,
    ContinuumTerm,
    ContinuumTermKey,
    MoireConfig,
    compute_bands,
)


class _DenseC3Generator:
    def __init__(self) -> None:
        angle = 2.0 * np.pi / 3.0
        block = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.complex128,
        )
        self.c3 = np.block(
            [
                [block, np.zeros((2, 2), dtype=np.complex128)],
                [np.zeros((2, 2), dtype=np.complex128), block],
            ]
        )

    def get_operator(self, name: str, param=None) -> np.ndarray:
        assert name == "C3z"
        power = 1 if param is None else int(param)
        return np.linalg.matrix_power(self.c3, power)


def _dense_c3_models() -> tuple[MoireConfig, ContinuumModel, ContinuumModel]:
    q1 = np.zeros((1, 2), dtype=float)
    q2 = np.zeros((1, 2), dtype=float)
    operation = {
        "name": "C3z",
        "antiunitary": False,
        "k_map": {"type": "rotation", "angle_deg": 120.0},
        "q_map": {"type": "rotation", "angle_deg": 120.0},
        "sector_map": "identity",
    }
    config = MoireConfig(
        Q_set1=q1,
        Q_set2=q2,
        n_orb1=2,
        n_orb2=2,
        nlow_state=[2, 2],
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.5, np.sqrt(3.0) / 2.0]),
        kpoints=np.array([[0.11, -0.07], [0.23, 0.19], [-0.17, 0.29]], dtype=float),
        kpoints_fit=np.array([[0.11, -0.07]], dtype=float),
        save_eigvecs=True,
        symmetry_gen=_DenseC3Generator(),
    )
    supported = ContinuumModel()
    legacy = ContinuumModel()
    term_specs = [
        (ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.0, 0.0)), 0.7, 0.11),
        (ContinuumTermKey(1, 0, 1, 1, 2, 1, (0.0, 0.0)), -0.3, 0.17),
        (ContinuumTermKey(0, 1, 2, 2, 1, 2, (0.0, 0.0)), 0.4, -0.09),
        (ContinuumTermKey(1, 1, 1, 2, 1, 2, (0.0, 0.0)), 0.2, 0.13),
    ]
    for key, real, imag in term_specs:
        sparse_basis = ContinuumModelBuilder.make_Y_basis_function(key, q1, q2, 2, 2)

        def dense_only(k: np.ndarray, basis=sparse_basis) -> np.ndarray:
            return basis(k)

        supported.terms[key] = ContinuumTerm(
            key=key,
            Y_basis=sparse_basis,
            r_value_real=real,
            r_value_imag=imag,
            active=True,
            tag="intra" if key.layer_from == key.layer_to else "inter",
            symmetry_ops=[dict(operation)],
        )
        legacy.terms[key] = ContinuumTerm(
            key=key,
            Y_basis=dense_only,
            r_value_real=real,
            r_value_imag=imag,
            active=True,
            tag="intra" if key.layer_from == key.layer_to else "inter",
            symmetry_ops=[dict(operation)],
        )
    return config, supported, legacy


def _assert_eigensystems_equivalent(
    compiled: tuple[np.ndarray, np.ndarray],
    legacy: tuple[np.ndarray, np.ndarray],
) -> None:
    compiled_values, compiled_vectors = compiled
    legacy_values, legacy_vectors = legacy
    np.testing.assert_allclose(compiled_values, legacy_values, atol=1.0e-11, rtol=0.0)
    for index in range(compiled_values.shape[0]):
        reconstructed_compiled = (
            compiled_vectors[index]
            @ np.diag(compiled_values[index])
            @ compiled_vectors[index].conj().T
        )
        reconstructed_legacy = (
            legacy_vectors[index]
            @ np.diag(legacy_values[index])
            @ legacy_vectors[index].conj().T
        )
        np.testing.assert_allclose(
            reconstructed_compiled,
            reconstructed_legacy,
            atol=1.0e-11,
            rtol=0.0,
        )


def test_compute_bands_dense_c3_compiled_runtime_matches_legacy() -> None:
    config, supported, legacy = _dense_c3_models()
    compiled_hamiltonians: list[np.ndarray] = []
    legacy_hamiltonians: list[np.ndarray] = []
    compiled_runtime: list[object | None] = []
    legacy_runtime: list[object | None] = []

    compiled_result = compute_bands(
        config,
        supported,
        config.kpoints,
        return_eigvecs=True,
        hamiltonians_out=compiled_hamiltonians,
        compiled_runtime_out=compiled_runtime,
    )
    legacy_result = compute_bands(
        config,
        legacy,
        config.kpoints,
        return_eigvecs=True,
        hamiltonians_out=legacy_hamiltonians,
        compiled_runtime_out=legacy_runtime,
    )

    assert compiled_runtime and compiled_runtime[0] is not None
    assert legacy_runtime == [None]
    np.testing.assert_allclose(compiled_hamiltonians, legacy_hamiltonians, atol=1.0e-11, rtol=0.0)
    _assert_eigensystems_equivalent(compiled_result, legacy_result)


def test_compute_bands_compiled_runtime_preserves_fixed_schur_reduction() -> None:
    config, supported, legacy = _dense_c3_models()
    config.keep_indices = np.array([0, 1], dtype=int)
    config.remove_indices = np.array([2, 3], dtype=int)
    compiled_hamiltonians: list[np.ndarray] = []
    legacy_hamiltonians: list[np.ndarray] = []
    compiled_runtime: list[object | None] = []

    compiled_values = compute_bands(
        config,
        supported,
        config.kpoints,
        return_eigvecs=False,
        hamiltonians_out=compiled_hamiltonians,
        compiled_runtime_out=compiled_runtime,
    )
    legacy_values = compute_bands(
        config,
        legacy,
        config.kpoints,
        return_eigvecs=False,
        hamiltonians_out=legacy_hamiltonians,
    )

    assert compiled_runtime and compiled_runtime[0] is not None
    np.testing.assert_allclose(compiled_hamiltonians, legacy_hamiltonians, atol=1.0e-11, rtol=0.0)
    np.testing.assert_allclose(compiled_values, legacy_values, atol=1.0e-11, rtol=0.0)


def test_compiled_response_matrix_reconstructs_multi_k_runtime_hamiltonians() -> None:
    config, supported, _legacy = _dense_c3_models()
    runtime_terms = list(supported.terms.values())
    runtime = operator_runtime_module.CompiledOperatorRuntime.from_terms(
        runtime_terms,
        config,
        dim=4,
    )

    response = operator_runtime_module.compiled_operator_response_matrix(
        runtime.recipe,
        config.kpoints,
        dim=4,
        term_count=len(runtime_terms),
    )
    coefficients = np.empty(2 * len(runtime_terms), dtype=float)
    coefficients[0::2] = runtime.r_real
    coefficients[1::2] = runtime.r_imag
    reconstructed = np.asarray(response @ coefficients).reshape(12, 12)

    expected = np.zeros((12, 12), dtype=np.complex128)
    for index, kpoint in enumerate(config.kpoints):
        block = slice(4 * index, 4 * (index + 1))
        expected[block, block] = runtime.hamiltonian(kpoint)

    assert response.shape == (12 * 12, 2 * len(runtime_terms))
    np.testing.assert_allclose(reconstructed, expected, atol=1.0e-11, rtol=0.0)


def test_compiled_response_matrix_preserves_dynamic_hermitianization() -> None:
    q1 = np.zeros((1, 2), dtype=float)
    q2 = np.zeros((1, 2), dtype=float)
    key = ContinuumTermKey(1, 0, 1, 2, 1, 1, (0.0, 0.0))
    term = ContinuumTerm(
        key=key,
        Y_basis=ContinuumModelBuilder.make_Y_basis_function(key, q1, q2, 1, 1),
        r_value_real=0.37,
        r_value_imag=-0.21,
        active=True,
        tag="inter",
        symmetry_ops=[],
    )
    term._moire_needs_hermitize_real = True
    term._moire_needs_hermitize_imag = True
    term._moire_hermitize_flags_inconsistent = True
    config = MoireConfig(
        Q_set1=q1,
        Q_set2=q2,
        n_orb1=1,
        n_orb2=1,
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.5, np.sqrt(3.0) / 2.0]),
        kpoints=np.array([[0.2, -0.1], [-0.3, 0.4]], dtype=float),
    )
    runtime = operator_runtime_module.CompiledOperatorRuntime.from_terms(
        [term],
        config,
        dim=2,
    )

    response = operator_runtime_module.compiled_operator_response_matrix(
        runtime.recipe,
        config.kpoints,
        dim=2,
        term_count=1,
    )
    reconstructed = np.asarray(response @ np.array([0.37, -0.21])).reshape(4, 4)
    expected = np.zeros((4, 4), dtype=np.complex128)
    for index, kpoint in enumerate(config.kpoints):
        block = slice(2 * index, 2 * (index + 1))
        expected[block, block] = runtime.hamiltonian(kpoint)

    np.testing.assert_allclose(reconstructed, expected, atol=1.0e-12, rtol=0.0)

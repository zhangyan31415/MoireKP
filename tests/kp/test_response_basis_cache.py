from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from scipy import sparse

import kp.model.response_basis as response_basis_module
from kp.model.response_basis import (
    CompiledResponseBasis,
    PolynomialCoordinateBasis,
    RawPolynomialSeed,
    compile_candidate_responses,
    identity_finite_group,
)
from kp.model.response_basis_cache import (
    PersistentResponseBasisCache,
    ResponseBasisCacheCorruptionError,
    TargetDependentCacheKeyError,
    target_independent_basis_key,
)


def _toy_basis(*, value: float = 1.0) -> CompiledResponseBasis:
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=[0.0, 0.0],
        reciprocal_basis=[[2.0, 0.0], [0.0, 4.0]],
        max_degree=0,
    )
    matrix = sparse.csr_matrix(np.diag([value, 0.0]).astype(np.complex128))
    candidates = compile_candidate_responses(
        [RawPolynomialSeed("E11", {(0, 0): matrix}, "diag", {})],
        coordinate=coordinate,
        group=identity_finite_group(2),
    )
    return CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload={
            "basis_layout": {"order": "toy", "dim": 2},
            "q_vectors": np.zeros((1, 2), dtype=np.float64),
            "basis_ordering": np.arange(2, dtype=np.int64),
            "n_orb": np.asarray([2, 0], dtype=np.int64),
            "exactified_matrices": {"identity": np.eye(2, dtype=np.complex128)},
            "k_pullbacks": {"identity": np.eye(2, dtype=np.float64)},
            "q_permutations": {"identity": np.arange(1, dtype=np.int64)},
            "sector_permutations": {"identity": np.asarray([0, 1], dtype=np.int64)},
            "dtype": "complex128",
            "compiler_version": "cache-test-v1",
        },
        reduce=True,
    )


def _compile_identity(*, marker: int = 1) -> dict[str, object]:
    coordinate_basis = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=[0.0, 0.0],
        reciprocal_basis=[[1.0, 0.0], [0.0, 1.0]],
        max_degree=0,
    )
    identity_group = identity_finite_group(2)
    identity_u = np.eye(2, dtype=np.complex128)
    element = {
        "canonical_word": [],
        "alternate_words": [],
        "antiunitary": False,
        "canonical_k_map": [[1, 0], [0, 1]],
        "k_pullback": [[1.0, 0.0], [0.0, 1.0]],
        "q_permutation": [0],
        "sector_permutation": [0, 1],
        "alternate_word_residual": 0.0,
        "unitarity_residual": 0.0,
    }
    coordinate = coordinate_basis.metadata()
    return {
        "identity": {
            "basis_layout": {"order": "toy", "dim": 2},
            "basis_ordering": np.asarray([0, 1], dtype=np.int64),
            "q_vectors": {"qset1": np.asarray([[float(marker - 1), 0.0]]), "qset2": np.empty((0, 2))},
            "n_orb": np.asarray([2, 0], dtype=np.int64),
            "exactified_matrices": {"group_0_element_0": identity_u},
            "antiunitary_parities": {"group_0_element_0": False},
            "canonical_k_maps": {"group_0_element_0": np.eye(2, dtype=np.int64)},
            "k_pullbacks": {"group_0_element_0": np.eye(2, dtype=np.float64)},
            "q_permutations": {"group_0_element_0": np.asarray([0], dtype=np.int64)},
            "sector_permutations": {"group_0_element_0": np.asarray([0, 1], dtype=np.int64)},
            "group_limits": [{"max_group_size": 256, "max_word_length": 32}],
            "factorized_action_hashes": [{}],
            "joint_route_action_hashes": [
                {
                    "joint_artifact_hash": None,
                    "generator_matrix_hashes": {},
                }
            ],
            "polynomial_coordinate": coordinate,
            "dtype": "complex128",
            "response_normalization": "reynolds_mean__hermitian_half_sum__dimensionless_k_v1",
            "coefficient_normalization": "response_norm_times_coefficient_over_1eV_v1",
            "regularization_normalization": "ridge_on_dimensionless_response_amplitude_v1",
            "compiler_version": "cache-test-v1",
            "term_templates": [],
            "structural_preselection": {
                "compiler": "cache-test-v1",
                "groups": [],
            },
            "local_reynolds_response_compiler": (
                response_basis_module._local_reynolds_cache_identity(
                    coordinate=coordinate_basis,
                    groups=[identity_group],
                    seeds_by_group=[[]],
                    factorized_actions_by_group=[{}],
                )
            ),
        },
        "coordinate": coordinate,
        "groups": [{
            "max_group_size": 256,
            "max_word_length": 32,
            "algebra_residual": 0.0,
            "elements": [element],
        }],
        "group_internal_u": [[identity_u]],
        "seeds": [{
            "seed_id": "E11",
            "support_component": "diag",
            "metadata": {},
            "coefficients": {(0, 0): np.diag([1.0, 0.0]).astype(np.complex128)},
        }],
        "reduce": True,
    }


def _rewrite_archive(path: Path, transform) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    transform(arrays)
    np.savez_compressed(path, **arrays)


def test_target_independent_basis_key_is_deterministic_for_canonical_arrays() -> None:
    native = _compile_identity()
    big_endian = _compile_identity()
    big_endian["identity"]["basis_ordering"] = np.asarray([0, 1], dtype=">i8")
    big_endian["identity"]["q_vectors"]["qset1"] = np.asarray([[0.0, 0.0]], dtype=">f8")
    big_endian["identity"]["q_vectors"]["qset2"] = np.empty((0, 2), dtype=">f8")

    assert target_independent_basis_key(native) == target_independent_basis_key(big_endian)
    assert target_independent_basis_key(native) != target_independent_basis_key(
        _compile_identity(marker=2)
    )


def test_target_independent_basis_key_includes_factorized_action_hashes() -> None:
    native = _compile_identity()
    factorized = _compile_identity()
    factorized["identity"]["factorized_action_hashes"][0]["C3z"] = "a" * 64

    assert target_independent_basis_key(native) != target_independent_basis_key(factorized)


def test_target_independent_basis_key_includes_joint_route_action_hashes() -> None:
    native = _compile_identity()
    routed = _compile_identity()
    routed["identity"]["joint_route_action_hashes"][0] = {
        "joint_artifact_hash": "b" * 64,
        "generator_matrix_hashes": {"C2": "c" * 64},
    }

    assert target_independent_basis_key(native) != target_independent_basis_key(routed)


@pytest.mark.parametrize("payload", [{}, {"identity": {}}])
def test_target_independent_basis_key_rejects_empty_or_incomplete_payload(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="incomplete|missing"):
        target_independent_basis_key(payload)


def test_target_independent_basis_key_rejects_unknown_fields() -> None:
    payload = _compile_identity()
    payload["unversioned_future_input"] = {"silently": "ignored"}

    with pytest.raises(ValueError, match="unexpected.*unversioned_future_input"):
        target_independent_basis_key(payload)


@pytest.mark.parametrize(
    "field",
    [
        "heff",
        "heff_file",
        "target_hamiltonians",
        "fit_indices",
        "response_fit_indices",
        "regularization",
        "band_window",
        "bandwindow",
        "fit_pivots",
        "fitted_coefficients",
    ],
)
def test_target_independent_basis_key_rejects_target_dependent_fields(field: str) -> None:
    payload = _compile_identity()
    payload["nested"] = {field: [1, 2, 3]}

    with pytest.raises(TargetDependentCacheKeyError, match=field):
        target_independent_basis_key(payload)


def test_cache_identity_certifies_group_actions_element_by_element_modulo_phase() -> None:
    payload = _compile_identity()
    payload["identity"]["exactified_matrices"]["group_0_element_0"] *= np.exp(0.37j)

    target_independent_basis_key(payload)

    payload["identity"]["exactified_matrices"]["group_0_element_0"] = np.asarray(
        [[0.0, 1.0], [1.0, 0.0]],
        dtype=np.complex128,
    )
    with pytest.raises(ValueError, match="group_0_element_0.*phase"):
        target_independent_basis_key(payload)


def test_cache_identity_certifies_group_discrete_action_and_element_order() -> None:
    payload = _compile_identity()
    payload["groups"][0]["elements"][0]["antiunitary"] = True

    with pytest.raises(ValueError, match="group_0_element_0.*antiunitary"):
        target_independent_basis_key(payload)


def test_cache_identity_certifies_actual_polynomial_k_pullback() -> None:
    payload = _compile_identity()
    payload["groups"][0]["elements"][0]["k_pullback"][0][1] = 1.0e-12

    with pytest.raises(ValueError, match="group_0_element_0.*k_pullback"):
        target_independent_basis_key(payload)


def test_persistent_cache_cold_compiles_once_then_warm_loads_frozen_basis(
    tmp_path: Path,
) -> None:
    basis = _toy_basis()
    payload = _compile_identity()
    key = target_independent_basis_key(payload)
    compile_calls = 0

    def compile_basis() -> CompiledResponseBasis:
        nonlocal compile_calls
        compile_calls += 1
        return basis

    cold_cache = PersistentResponseBasisCache(tmp_path / "basis-cache")
    cold = cold_cache.get_or_compile(key, compile_basis)

    assert cold.basis_hash == basis.basis_hash
    assert compile_calls == 1
    assert cold_cache.stats()["cold_compile_count"] == 1
    assert cold_cache.stats()["warm_load_count"] == 0

    warm_cache = PersistentResponseBasisCache(tmp_path / "basis-cache")
    warm = warm_cache.get_or_compile(
        key,
        lambda: pytest.fail("a persistent warm hit must not invoke the compiler"),
    )

    assert warm.basis_hash == basis.basis_hash
    np.testing.assert_allclose(
        warm.hamiltonians([[0.0, 0.0]], [2.5]),
        basis.hamiltonians([[0.0, 0.0]], [2.5]),
    )
    assert warm_cache.stats()["cold_compile_count"] == 0
    assert warm_cache.stats()["warm_load_count"] == 1
    assert warm_cache.stats()["warm_load_count_by_key"] == {key: 1}


def test_cache_artifact_contains_no_cache_directory_or_machine_path(tmp_path: Path) -> None:
    cache_dir = tmp_path / "host-specific" / "basis-cache"
    cache = PersistentResponseBasisCache(cache_dir)
    path = cache.store(target_independent_basis_key(_compile_identity()), _toy_basis())

    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(np.asarray(archive["cache_metadata_json"]).item()))

    assert str(cache_dir) not in json.dumps(metadata, sort_keys=True)
    assert metadata["basis_hash"] == _toy_basis().basis_hash
    assert not list(cache_dir.glob("*.tmp*"))


@pytest.mark.parametrize("tamper", ["semantics", "basis_hash"])
def test_cache_rejects_semantics_or_basis_hash_mismatch(
    tmp_path: Path,
    tamper: str,
) -> None:
    cache = PersistentResponseBasisCache(tmp_path / "basis-cache")
    payload = _compile_identity()
    key = target_independent_basis_key(payload)
    path = cache.store(key, _toy_basis())

    def corrupt(arrays: dict[str, np.ndarray]) -> None:
        metadata = json.loads(str(np.asarray(arrays["cache_metadata_json"]).item()))
        if tamper == "semantics":
            metadata["response_semantics"] = "legacy_frozen_v1"
        else:
            metadata["basis_hash"] = "0" * 64
        arrays["cache_metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        )

    _rewrite_archive(path, corrupt)

    with pytest.raises(ResponseBasisCacheCorruptionError, match=tamper.replace("_", " ")):
        cache.load(key)


def test_corrupt_cache_fails_closed_without_recompiling(tmp_path: Path) -> None:
    cache = PersistentResponseBasisCache(tmp_path / "basis-cache")
    payload = _compile_identity()
    key = target_independent_basis_key(payload)
    path = cache.store(key, _toy_basis())
    path.write_bytes(b"not-an-npz")
    compile_calls = 0

    def must_not_compile() -> CompiledResponseBasis:
        nonlocal compile_calls
        compile_calls += 1
        return _toy_basis()

    with pytest.raises(ResponseBasisCacheCorruptionError, match="could not be read"):
        cache.get_or_compile(key, must_not_compile)

    assert compile_calls == 0
    assert cache.stats()["corrupt_load_count"] == 1


def test_failed_atomic_rewrite_preserves_previous_valid_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = PersistentResponseBasisCache(tmp_path / "basis-cache")
    payload = _compile_identity()
    key = target_independent_basis_key(payload)
    basis = _toy_basis()
    cache.store(key, basis)

    import kp.model.response_basis_cache as cache_module

    def fail_during_write(file_object, **arrays) -> None:
        file_object.write(b"partial")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(cache_module.np, "savez_compressed", fail_during_write)

    with pytest.raises(OSError, match="interrupted write"):
        cache.store(key, basis)

    restored = cache.load(key)
    assert restored.basis_hash == basis.basis_hash
    assert not list((tmp_path / "basis-cache").glob("*.tmp*"))


def test_existing_same_key_with_different_basis_hash_fails_closed(tmp_path: Path) -> None:
    cache = PersistentResponseBasisCache(tmp_path / "basis-cache")
    payload = _compile_identity()
    key = target_independent_basis_key(payload)
    first = _toy_basis(value=1.0)
    conflicting = _toy_basis(value=2.0)
    assert first.basis_hash != conflicting.basis_hash
    cache.store(key, first)

    with pytest.raises(ResponseBasisCacheCorruptionError, match="different basis hash"):
        cache.store(key, conflicting)

    assert cache.load(key).basis_hash == first.basis_hash


def test_same_key_concurrent_processes_cold_compile_once(tmp_path: Path) -> None:
    cache_dir = tmp_path / "basis-cache"
    count_path = tmp_path / "compile-count"
    script = textwrap.dedent(
        f"""
        import time
        from pathlib import Path
        from tests.kp.test_response_basis_cache import _compile_identity, _toy_basis
        from kp.model.response_basis_cache import (
            PersistentResponseBasisCache,
            target_independent_basis_key,
        )

        cache = PersistentResponseBasisCache({str(cache_dir)!r})
        count_path = Path({str(count_path)!r})
        def compile_basis():
            with count_path.open("a", encoding="utf-8") as handle:
                handle.write("compile\\n")
            time.sleep(0.5)
            return _toy_basis()
        cache.get_or_compile(
            target_independent_basis_key(_compile_identity()), compile_basis
        )
        """
    )
    processes = [
        subprocess.Popen([sys.executable, "-c", script])
        for _ in range(2)
    ]
    return_codes = [process.wait(timeout=20) for process in processes]

    assert return_codes == [0, 0]
    assert count_path.read_text(encoding="utf-8").splitlines() == ["compile"]

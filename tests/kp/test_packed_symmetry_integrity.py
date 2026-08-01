from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kp.model.symmetry import MatrixSymmetryGenerator, load_symmetry_source
from kp.symmetry.canonical_target import certify_fixed_target_exactification
from kp.symmetry.joint_exactification import (
    BlockRouteAction,
    JointExactificationConfig,
    MagneticGenerator,
    MagneticPresentation,
    MagneticRelation,
    certify_fixed_target_joint_result,
    materialize_block_route_action,
)


def _write_fixed_target_package(
    directory: Path,
    *,
    operation: str = "C2",
    include_power_relation: bool = True,
    c3_central_sign: int = -1,
    block_dimension: int = 1,
) -> Path:
    if block_dimension <= 0:
        raise ValueError("block_dimension must be positive")
    if operation == "C2":
        action_value = -1.0 + 0.0j
        relation = MagneticRelation("C2^2", ("C2", "C2"), (), 1.0)
        central_phases = (1.0,)
        root_order = 4
        k_map = {"type": "reflection", "axis_deg": 0.0}
        q_map = {"type": "reflection", "axis_deg": 0.0}
    elif operation == "C3z":
        if c3_central_sign == -1:
            action_value = 0.5 - 0.8660254037844386j
            central_phases = (1.0, -1.0)
        elif c3_central_sign == 1:
            action_value = -0.5 + 0.8660254037844386j
            central_phases = (1.0,)
        else:
            raise ValueError("c3_central_sign must be +1 or -1")
        relation = MagneticRelation(
            "C3z^3",
            ("C3z",) * 3,
            (),
            float(c3_central_sign),
        )
        root_order = 24
        k_map = {"type": "rotation", "angle_deg": 120.0}
        q_map = {"type": "rotation", "angle_deg": 120.0}
    else:
        raise ValueError(f"unsupported test operation {operation!r}")
    action = BlockRouteAction(
        operation,
        False,
        (0,),
        (block_dimension,),
        (
            np.diag(
                [
                    action_value if index % 2 == 0 else np.conjugate(action_value)
                    for index in range(block_dimension)
                ]
            ).astype(np.complex128),
        ),
    )
    actions = {operation: action}
    presentation = MagneticPresentation(
        generators=(MagneticGenerator(operation, False),),
        relations=(relation,) if include_power_relation else (),
        central_phases=central_phases,
        source="packed-fixed-target-integrity-test",
    )
    config = JointExactificationConfig()
    certificate = certify_fixed_target_exactification(
        {operation: materialize_block_route_action(action)},
        actions,
        root_order=root_order,
        max_rms_correction=config.max_rms_correction,
        max_route_correction=config.max_route_correction,
    )
    result = certify_fixed_target_joint_result(
        actions,
        presentation,
        config=config,
        provenance={"kind": "persisted_project_basis_fixed_target"},
        fixed_target_certificate=certificate,
    )
    metadata = {
        "strict_metadata": True,
        "requires_model_exactification": False,
        "exactification_owner": "kp_symm",
        "kp_symm_exactification": {
            "status": "exactified",
            "matrix_source": "kp_symm_exactified_action",
            "fixed_target_exactification": dict(certificate),
            "joint_certification": {
                "fixed_target_exactification": dict(certificate),
            },
            "joint_block_representation": dict(result.artifact_metadata),
        },
        "operations": [
            {
                "name": operation,
                "operation": operation,
                "matrix_file": "representations.npz",
                "matrix_array_key": operation,
                "matrix_kind": "continuum_internal_rep_exact",
                "matrix_source": "kp_symm_exactified_action",
                "antiunitary": False,
                "k_map": k_map,
                "q_map": q_map,
                "sector_map": "identity",
                "spin_map": "from_kp_symm_output",
                "valley_map": "identity",
                "source_matrix_role": "raw_h_sewing_action",
                "source_gauge": "raw_saved_TAPW",
                "target_role": "continuum_internal_rep",
                "gauge_correction": {"kind": "none"},
                "antiunitary_convention": "none",
                "status": "exactified",
                "exactification_status": "exactified",
                "source_matrix_projection_report": {
                    "source_matrix_role": "raw_h_sewing_action",
                    "target_role": "continuum_internal_rep",
                    "support_resolution": {
                        "matrix_kind": "action",
                        "matrix_source": "raw_h_action_projection",
                    },
                    "report": {"status": "exactified"},
                },
            }
        ],
    }
    path = directory / "representations.npz"
    np.savez_compressed(
        path,
        **{operation: materialize_block_route_action(action)},
        __metadata_json__=np.asarray(json.dumps(metadata, sort_keys=True)),
        **dict(result.artifact_arrays),
    )
    return path


def _rewrite_array(path: Path, key: str, value: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays[key] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def _rewrite_fixed_target_hash(path: Path) -> None:
    from kp.symmetry.joint_exactification import _joint_artifact_hash

    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    metadata = json.loads(str(arrays["__metadata_json__"].item()))
    exactification = metadata["kp_symm_exactification"]
    joint = exactification["joint_block_representation"]
    joint["fixed_target_exactification"]["certificate"][
        "target_matrix_hashes"
    ]["C2"] = "0" * 64
    joint["report"]["fixed_target_exactification"]["target_matrix_hashes"][
        "C2"
    ] = "0" * 64
    exactification["fixed_target_exactification"]["target_matrix_hashes"][
        "C2"
    ] = "0" * 64
    exactification["joint_certification"]["fixed_target_exactification"][
        "target_matrix_hashes"
    ]["C2"] = "0" * 64
    route_arrays = {
        key: value
        for key, value in arrays.items()
        if key.startswith("__joint_block_representation_")
    }
    unsigned = dict(joint)
    unsigned.pop("artifact_hash")
    joint["artifact_hash"] = _joint_artifact_hash(unsigned, route_arrays)
    arrays["__metadata_json__"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)


def _rewrite_fixed_target_layout(path: Path) -> None:
    from kp.symmetry.joint_exactification import _joint_artifact_hash

    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    metadata = json.loads(str(arrays["__metadata_json__"].item()))
    exactification = metadata["kp_symm_exactification"]
    joint = exactification["joint_block_representation"]
    joint["fixed_target_exactification"]["certificate"]["route_layout"][
        "C2"
    ]["fiber_permutation"] = [1]
    joint["report"]["fixed_target_exactification"]["route_layout"]["C2"][
        "fiber_permutation"
    ] = [1]
    exactification["fixed_target_exactification"]["route_layout"]["C2"][
        "fiber_permutation"
    ] = [1]
    exactification["joint_certification"]["fixed_target_exactification"][
        "route_layout"
    ]["C2"]["fiber_permutation"] = [1]
    route_arrays = {
        key: value
        for key, value in arrays.items()
        if key.startswith("__joint_block_representation_")
    }
    unsigned = dict(joint)
    unsigned.pop("artifact_hash")
    joint["artifact_hash"] = _joint_artifact_hash(unsigned, route_arrays)
    arrays["__metadata_json__"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)


def _rewrite_top_level_fixed_target_hash(path: Path) -> None:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    metadata = json.loads(str(arrays["__metadata_json__"].item()))
    metadata["kp_symm_exactification"]["fixed_target_exactification"][
        "target_matrix_hashes"
    ]["C2"] = "0" * 64
    arrays["__metadata_json__"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)


def _rewrite_c3_central_phase(path: Path) -> None:
    from kp.symmetry.joint_exactification import _joint_artifact_hash

    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    metadata = json.loads(str(arrays["__metadata_json__"].item()))
    joint = metadata["kp_symm_exactification"]["joint_block_representation"]
    joint["presentation"]["relations"][0]["central_phase"] = [1.0, 0.0]
    route_arrays = {
        key: value
        for key, value in arrays.items()
        if key.startswith("__joint_block_representation_")
    }
    unsigned = dict(joint)
    unsigned.pop("artifact_hash")
    joint["artifact_hash"] = _joint_artifact_hash(unsigned, route_arrays)
    arrays["__metadata_json__"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)


def test_packed_fixed_target_loads_when_dense_routes_and_certificate_agree(
    tmp_path: Path,
) -> None:
    _write_fixed_target_package(tmp_path)

    source = load_symmetry_source(
        {"type": "kp_symm_output", "path": str(tmp_path)},
        base=tmp_path,
        expected_dim=1,
    )

    assert source.generator is not None
    assert np.array_equal(
        source.generator.get_operator("C2"),
        np.asarray([[-1.0]], dtype=np.complex128),
    )


def test_packed_fixed_target_rejects_dense_matrix_not_matching_joint_routes(
    tmp_path: Path,
) -> None:
    path = _write_fixed_target_package(tmp_path)
    _rewrite_array(path, "C2", np.asarray([[1.0]], dtype=np.complex128))

    with pytest.raises(ValueError, match="certified joint.*C2"):
        load_symmetry_source(
            {"type": "kp_symm_output", "path": str(tmp_path)},
            base=tmp_path,
            expected_dim=1,
        )


def test_packed_fixed_target_rejects_certificate_for_different_target(
    tmp_path: Path,
) -> None:
    path = _write_fixed_target_package(tmp_path)
    _rewrite_fixed_target_hash(path)

    with pytest.raises(ValueError, match="fixed-target.*hash"):
        load_symmetry_source(
            {"type": "kp_symm_output", "path": str(tmp_path)},
            base=tmp_path,
            expected_dim=1,
        )


def test_packed_fixed_target_rejects_certificate_for_different_route_layout(
    tmp_path: Path,
) -> None:
    path = _write_fixed_target_package(tmp_path)
    _rewrite_fixed_target_layout(path)

    with pytest.raises(ValueError, match="fixed-target.*validation"):
        load_symmetry_source(
            {"type": "kp_symm_output", "path": str(tmp_path)},
            base=tmp_path,
            expected_dim=1,
        )


def test_packed_fixed_target_rejects_inconsistent_top_level_certificate(
    tmp_path: Path,
) -> None:
    path = _write_fixed_target_package(tmp_path)
    _rewrite_top_level_fixed_target_hash(path)

    with pytest.raises(ValueError, match="fixed-target.*top-level"):
        load_symmetry_source(
            {"type": "kp_symm_output", "path": str(tmp_path)},
            base=tmp_path,
            expected_dim=1,
        )


def test_packed_c3_derived_powers_use_certified_relation_without_rounding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_fixed_target_package(tmp_path, operation="C3z")
    source = load_symmetry_source(
        {"type": "kp_symm_output", "path": str(tmp_path)},
        base=tmp_path,
        expected_dim=1,
    )
    assert source.generator is not None
    generator = source.generator
    base = generator.get_operator("C3z", 1)
    adjoint = np.array(base.conjugate().T, copy=True)
    derivation = source.metadata["c3_power_derivation"]
    assert derivation["status"] == "certified"
    assert derivation["numeric_inverse_or_matrix_power_performed"] is False
    assert (
        derivation["relation"]["certification_source"]
        == "joint_block_representation.presentation"
    )
    assert len(derivation["relation"]["joint_artifact_hash"]) == 64

    noisy_square = base @ base
    assert noisy_square.tobytes(order="C") != (-adjoint).tobytes(order="C")

    def fail_inverse(_matrix: np.ndarray) -> np.ndarray:
        raise AssertionError("certified C3 powers must not call np.linalg.inv")

    monkeypatch.setattr(np.linalg, "inv", fail_inverse)

    assert generator.get_operator("C3z", -1).tobytes(order="C") == adjoint.tobytes(
        order="C"
    )
    assert generator.get_operator("C3z", 2).tobytes(order="C") == (
        -adjoint
    ).tobytes(order="C")
    assert generator.get_operator("C3z", -2).tobytes(order="C") == (
        -base
    ).tobytes(order="C")


def test_packed_c3_without_certified_cubic_relation_fails_closed(
    tmp_path: Path,
) -> None:
    _write_fixed_target_package(
        tmp_path,
        operation="C3z",
        include_power_relation=False,
    )

    with pytest.raises(ValueError, match="C3z.*certified cubic power relation"):
        load_symmetry_source(
            {"type": "kp_symm_output", "path": str(tmp_path)},
            base=tmp_path,
            expected_dim=1,
        )


def test_packed_spinless_c3_uses_positive_certified_central_phase(
    tmp_path: Path,
) -> None:
    _write_fixed_target_package(
        tmp_path,
        operation="C3z",
        c3_central_sign=1,
    )
    source = load_symmetry_source(
        {"type": "kp_symm_output", "path": str(tmp_path)},
        base=tmp_path,
        expected_dim=1,
    )
    assert source.generator is not None
    generator = source.generator
    base = generator.get_operator("C3z", 1)
    adjoint = np.array(base.conjugate().T, copy=True)

    assert generator.get_operator("C3z", 2).tobytes(order="C") == adjoint.tobytes(
        order="C"
    )
    assert generator.get_operator("C3z", -2).tobytes(order="C") == base.tobytes(
        order="C"
    )


def test_certified_c3_derived_powers_keep_structural_zero_bits_positive(
    tmp_path: Path,
) -> None:
    _write_fixed_target_package(
        tmp_path,
        operation="C3z",
        block_dimension=2,
    )
    source = load_symmetry_source(
        {"type": "kp_symm_output", "path": str(tmp_path)},
        base=tmp_path,
        expected_dim=2,
    )
    assert source.generator is not None
    generator = source.generator
    base = generator.get_operator("C3z", 1)
    expected_by_power = {
        -1: np.array(base.conjugate().T, dtype=np.complex128, order="C"),
        2: np.negative(
            np.array(base.conjugate().T, dtype=np.complex128, order="C")
        ),
        -2: np.negative(np.array(base, dtype=np.complex128, order="C")),
    }

    for power, expected in expected_by_power.items():
        actual = generator.get_operator("C3z", power)
        structural_zero = actual == 0.0
        assert np.all(actual.real[structural_zero].view(np.uint64) == 0)
        assert np.all(actual.imag[structural_zero].view(np.uint64) == 0)
        assert actual[~structural_zero].tobytes(order="C") == expected[
            expected != 0.0
        ].tobytes(order="C")


def test_packed_c3_tampered_central_phase_is_rejected(tmp_path: Path) -> None:
    path = _write_fixed_target_package(tmp_path, operation="C3z")
    _rewrite_c3_central_phase(path)

    with pytest.raises(ValueError, match="relation certification"):
        load_symmetry_source(
            {"type": "kp_symm_output", "path": str(tmp_path)},
            base=tmp_path,
            expected_dim=1,
        )


def test_legacy_c3_generator_keeps_explicit_numeric_power_path() -> None:
    base = np.asarray([[0.5 - 0.8660254037844386j]], dtype=np.complex128)
    generator = MatrixSymmetryGenerator({"C3z": base}, {"operations": []})

    assert generator.get_operator("C3z", 2).tobytes(order="C") == (
        base @ base
    ).tobytes(order="C")
    assert generator.metadata["c3_power_derivation"] == {
        "status": "legacy_numeric",
        "reason": "no_certified_joint_cubic_power_relation",
        "numeric_inverse_or_matrix_power_performed": True,
    }

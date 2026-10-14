from __future__ import annotations

import pytest
import numpy as np

from kp.basis.symmetry_gauge import (
    SymmetryAdaptedBasisFrame,
    SymmetryAdaptedInternalFrame,
    symmetry_adapted_frame_hash,
)


def _basis_artifact():
    internal = SymmetryAdaptedInternalFrame(
        status="resolved", unitary=np.eye(1, dtype=complex),
        primary_operation="C3z", eigenvalues=(1+0j,),
    )
    basis = SymmetryAdaptedBasisFrame(
        status="resolved", full_unitary=np.eye(1, dtype=complex),
        sector_frames=(("L1", 1, 1, internal),),
    )
    return basis.artifact()


@pytest.mark.parametrize("version", [None, "obsolete-format"])
def test_symmetry_frame_rejects_missing_or_unsupported_schema(version):
    artifact = _basis_artifact()
    if version is None:
        artifact.pop("version")
    else:
        artifact["version"] = version
    artifact["frame_hash"] = symmetry_adapted_frame_hash(artifact)
    with pytest.raises(ValueError, match="format"):
        SymmetryAdaptedBasisFrame.from_artifact(artifact)


def test_symmetry_frame_rejects_unknown_basis_ordering():
    artifact = _basis_artifact()
    artifact["basis_ordering"] = "unknown-ordering"
    artifact["frame_hash"] = symmetry_adapted_frame_hash(artifact)
    with pytest.raises(ValueError, match="ordering"):
        SymmetryAdaptedBasisFrame.from_artifact(artifact)


def test_symmetry_frame_requires_content_hash():
    artifact = _basis_artifact()
    artifact.pop("frame_hash")
    with pytest.raises(ValueError, match="hash"):
        SymmetryAdaptedBasisFrame.from_artifact(artifact)

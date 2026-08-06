from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from numbers import Integral, Number, Real
from typing import Any, List, Mapping, Sequence, Tuple, Literal

import numpy as np
import scipy
import scipy.linalg
import scipy.optimize

from .downfold import (
    DownfoldingOptions,
    downfold_from_projector_groups,
    downfold_from_projectors,
    projector_groups_from_block_columns,
    set_projector_blas_threads as _set_downfold_projector_blas_threads,
)
from kp.basis.selection import (
    AutoGaugeConfig,
    AutoGaugeSelection,
    GaugeAnchorReport,
    select_anchor_rows_qrcp,
)
from kp.identity import hash_array, hash_mapping
from kp.projection_selection import CandidateRejectionReason

from .gamma_layout import GammaRoutingError, GammaRowLayout

PROJECTOR_BLAS_THREADS = 8
_PROJECTOR_BLAS_SCOPE_DEPTH: ContextVar[int] = ContextVar(
    "projector_blas_scope_depth",
    default=0,
)


@dataclass(frozen=True)
class ProjectGaugeAnchorCandidate:
    candidate_id: str
    resolved_norb_fix_list: list[Any]
    report: GaugeAnchorReport
    priority: int = 0


def _record_tuple(value: Any, *, context: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{context} must be a sequence, not a string")
    try:
        return tuple(value)
    except TypeError as error:
        raise ValueError(f"{context} must be a sequence") from error


def _record_int(value: Any, *, context: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context} must be a strict integer")
    return int(value)


def _record_float(value: Any, *, context: str) -> float:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context} must be a real number")
    return float(value)


def _record_complex(value: Any, *, context: str) -> complex:
    if not isinstance(value, Number) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context} must be a numeric coefficient")
    return complex(value)


def _record_references(value: Any) -> tuple[tuple[tuple[int, complex], ...], ...]:
    references: list[tuple[tuple[int, complex], ...]] = []
    for raw_reference in _record_tuple(value, context="Gamma model resolved references"):
        terms: list[tuple[int, complex]] = []
        for raw_term in _record_tuple(raw_reference, context="Gamma model reference column"):
            term = _record_tuple(raw_term, context="Gamma model reference term")
            if len(term) != 2:
                raise ValueError("Gamma model reference terms must contain exactly row and coefficient")
            terms.append(
                (
                    _record_int(term[0], context="Gamma model reference row"),
                    _record_complex(term[1], context="Gamma model reference coefficient"),
                )
            )
        references.append(tuple(terms))
    return tuple(references)


@dataclass(frozen=True)
class GammaModelAnchorSpec:
    """Immutable SCDM anchor contract for one automatic-Gamma candidate."""

    joint_band_indices: tuple[int, ...]
    group_ranks: tuple[int, int]
    model_column_order: tuple[tuple[int, int], ...]
    resolved_reference_terms: tuple[tuple[tuple[int, complex], ...], ...]
    selected_rows: tuple[int, ...]
    reference_q_index: int
    layout_hash: str
    same_q_dimension: int
    min_sigma: float
    max_condition: float
    projector_tolerance: float
    orthonormality_tolerance: float
    selection_sigma_min: float
    selection_condition_number: float
    reference_singular_values: tuple[float, ...]
    reference_sigma_min: float
    reference_condition_number: float
    reference_eigenvalues: tuple[float, ...]
    reference_projector_hash: str
    warnings: tuple[str, ...]
    identity_hash: str

    SCHEMA = "kp.gamma-model-anchor-spec.v1"

    def __post_init__(self) -> None:
        raw_bands = _record_tuple(self.joint_band_indices, context="Gamma model joint bands")
        raw_ranks = _record_tuple(self.group_ranks, context="Gamma model group ranks")
        bands, ranks, expected_order = _gamma_joint_contract(raw_bands, raw_ranks)
        raw_order = _record_tuple(self.model_column_order, context="Gamma model column order")
        order = tuple(
            tuple(
                _record_int(value, context="Gamma model column-order index")
                for value in _record_tuple(pair, context="Gamma model column-order pair")
            )
            for pair in raw_order
        )
        if any(len(pair) != 2 for pair in order) or order != expected_order:
            raise ValueError("Gamma model anchor joint-band/group/model-column contract is not canonical")
        references = _record_references(self.resolved_reference_terms)
        if len(references) != len(bands):
            raise ValueError("Gamma model anchor must contain one resolved reference per joint column")
        selected_rows = tuple(
            _record_int(row, context="Gamma model selected row")
            for row in _record_tuple(self.selected_rows, context="Gamma model selected rows")
        )
        if (
            len(selected_rows) != len(bands)
            or len(set(selected_rows)) != len(bands)
        ):
            raise ValueError("Gamma model anchor selected rows must uniquely cover the joint rank")
        reference_q_index = _record_int(
            self.reference_q_index,
            context="Gamma model reference_q_index",
        )
        if reference_q_index < 0:
            raise ValueError("Gamma model anchor reference_q_index must be a non-negative integer")
        same_q_dimension = _record_int(
            self.same_q_dimension,
            context="Gamma model same_q_dimension",
        )
        if same_q_dimension <= 0:
            raise ValueError("Gamma model anchor same_q_dimension must be a positive integer")
        if any(row < 0 or row >= same_q_dimension for row in selected_rows):
            raise ValueError("Gamma model anchor selected row is outside same_q_dimension")
        if not isinstance(self.layout_hash, str) or not self.layout_hash:
            raise ValueError("Gamma model anchor layout_hash must be nonempty")
        if not isinstance(self.reference_projector_hash, str) or not self.reference_projector_hash:
            raise ValueError("Gamma model anchor reference_projector_hash must be nonempty")
        scalar_names = (
            "min_sigma",
            "max_condition",
            "projector_tolerance",
            "orthonormality_tolerance",
            "selection_sigma_min",
            "selection_condition_number",
            "reference_sigma_min",
            "reference_condition_number",
        )
        scalar_values = tuple(
            _record_float(getattr(self, name), context=f"Gamma model {name}")
            for name in scalar_names
        )
        reference_singular_values = tuple(
            _record_float(value, context="Gamma model reference singular value")
            for value in _record_tuple(
                self.reference_singular_values,
                context="Gamma model reference singular values",
            )
        )
        reference_eigenvalues = tuple(
            _record_float(value, context="Gamma model reference eigenvalue")
            for value in _record_tuple(
                self.reference_eigenvalues,
                context="Gamma model reference eigenvalues",
            )
        )
        warnings = _record_tuple(self.warnings, context="Gamma model warnings")
        if any(not isinstance(value, str) for value in warnings):
            raise ValueError("Gamma model warnings must contain strings")
        object.__setattr__(self, "joint_band_indices", bands)
        object.__setattr__(self, "group_ranks", ranks)
        object.__setattr__(self, "model_column_order", order)
        object.__setattr__(self, "resolved_reference_terms", references)
        object.__setattr__(self, "selected_rows", selected_rows)
        object.__setattr__(self, "reference_q_index", reference_q_index)
        object.__setattr__(self, "same_q_dimension", same_q_dimension)
        for name, value in zip(scalar_names, scalar_values):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "reference_singular_values", reference_singular_values)
        object.__setattr__(self, "reference_eigenvalues", reference_eigenvalues)
        object.__setattr__(self, "warnings", warnings)
        all_numeric_values = scalar_values + reference_singular_values + reference_eigenvalues
        if any(not np.isfinite(value) for value in all_numeric_values):
            raise ValueError("Gamma model anchor quality/energy values must be finite")
        if (
            self.min_sigma <= 0.0
            or self.max_condition < 1.0
            or self.projector_tolerance <= 0.0
            or self.orthonormality_tolerance <= 0.0
            or self.selection_sigma_min < self.min_sigma
            or self.selection_condition_number > self.max_condition
            or self.reference_sigma_min < self.min_sigma
            or self.reference_condition_number > self.max_condition
        ):
            raise ValueError("Gamma model anchor quality metrics violate their configured thresholds")
        if len(reference_singular_values) != len(bands) or len(reference_eigenvalues) != len(bands):
            raise ValueError("Gamma model anchor singular-value/energy metrics must cover the joint rank")
        _gamma_reference_matrix(references, row_count=same_q_dimension)
        if not isinstance(self.identity_hash, str) or not self.identity_hash:
            raise ValueError("Gamma model anchor identity_hash must be nonempty")
        actual = hash_mapping(self._identity_payload())
        if actual != self.identity_hash:
            raise ValueError(
                f"Gamma model anchor identity hash mismatch: {self.identity_hash} != {actual}"
            )

    @property
    def reference_q(self) -> int:
        return self.reference_q_index

    @property
    def resolved_norb_fix_list(self) -> list[Any]:
        """Return the legacy two-owner anchor shape without mutable aliases."""

        resolved: list[Any] = []
        cursor = 0
        for rank in self.group_ranks:
            owner_refs: list[Any] = []
            for terms in self.resolved_reference_terms[cursor : cursor + rank]:
                owner_refs.append(_format_auto_reference_terms(list(terms)))
            resolved.append(owner_refs)
            cursor += rank
        return resolved

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "joint_band_indices": list(self.joint_band_indices),
            "group_ranks": list(self.group_ranks),
            "model_column_order": [list(value) for value in self.model_column_order],
            "resolved_reference_terms": _gamma_reference_payload(self.resolved_reference_terms),
            "selected_rows": list(self.selected_rows),
            "reference_q_index": self.reference_q_index,
            "layout_hash": self.layout_hash,
            "same_q_dimension": self.same_q_dimension,
            "min_sigma": self.min_sigma,
            "max_condition": self.max_condition,
            "projector_tolerance": self.projector_tolerance,
            "orthonormality_tolerance": self.orthonormality_tolerance,
            "selection_sigma_min": self.selection_sigma_min,
            "selection_condition_number": self.selection_condition_number,
            "reference_singular_values": list(self.reference_singular_values),
            "reference_sigma_min": self.reference_sigma_min,
            "reference_condition_number": self.reference_condition_number,
            "reference_eigenvalues": list(self.reference_eigenvalues),
            "reference_projector_hash": self.reference_projector_hash,
            "warnings": list(self.warnings),
        }

    def to_payload(self) -> dict[str, Any]:
        payload = self._identity_payload()
        actual = hash_mapping(payload)
        if actual != self.identity_hash:
            raise ValueError(
                f"Gamma model anchor identity hash mismatch: {self.identity_hash} != {actual}"
            )
        payload["identity_hash"] = self.identity_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "GammaModelAnchorSpec":
        if not isinstance(payload, Mapping):
            raise TypeError("Gamma model anchor payload must be a mapping")
        identity_keys = {
            "schema",
            "joint_band_indices",
            "group_ranks",
            "model_column_order",
            "resolved_reference_terms",
            "selected_rows",
            "reference_q_index",
            "layout_hash",
            "same_q_dimension",
            "min_sigma",
            "max_condition",
            "projector_tolerance",
            "orthonormality_tolerance",
            "selection_sigma_min",
            "selection_condition_number",
            "reference_singular_values",
            "reference_sigma_min",
            "reference_condition_number",
            "reference_eigenvalues",
            "reference_projector_hash",
            "warnings",
        }
        expected = identity_keys | {"identity_hash"}
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        if missing or unknown:
            raise ValueError(
                "Gamma model anchor payload keys mismatch "
                f"(missing={missing}, unknown={unknown})"
            )
        if payload["schema"] != cls.SCHEMA:
            raise ValueError(f"unsupported Gamma model anchor schema {payload['schema']!r}")
        canonical = {key: payload[key] for key in identity_keys}
        expected_hash = payload["identity_hash"]
        if not isinstance(expected_hash, str) or not expected_hash:
            raise ValueError("Gamma model anchor identity_hash must be a nonempty string")
        actual_hash = hash_mapping(canonical)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Gamma model anchor identity hash mismatch: {expected_hash} != {actual_hash}"
            )
        raw_references = payload["resolved_reference_terms"]
        if not isinstance(raw_references, list):
            raise ValueError("Gamma model resolved_reference_terms must be a list")
        references: list[tuple[tuple[int, complex], ...]] = []
        for terms in raw_references:
            if not isinstance(terms, list) or not terms:
                raise ValueError("Gamma model reference columns must be nonempty lists")
            parsed: list[tuple[int, complex]] = []
            for term in terms:
                if not isinstance(term, list) or len(term) != 3:
                    raise ValueError("Gamma model reference terms must be [row, real, imag]")
                parsed.append(
                    (
                        term[0],
                        complex(
                            _record_float(term[1], context="Gamma model reference coefficient real part"),
                            _record_float(term[2], context="Gamma model reference coefficient imaginary part"),
                        ),
                    )
                )
            references.append(tuple(parsed))
        spec = cls(
            joint_band_indices=payload["joint_band_indices"],
            group_ranks=payload["group_ranks"],
            model_column_order=payload["model_column_order"],
            resolved_reference_terms=tuple(references),
            selected_rows=payload["selected_rows"],
            reference_q_index=payload["reference_q_index"],
            layout_hash=payload["layout_hash"],
            same_q_dimension=payload["same_q_dimension"],
            min_sigma=payload["min_sigma"],
            max_condition=payload["max_condition"],
            projector_tolerance=payload["projector_tolerance"],
            orthonormality_tolerance=payload["orthonormality_tolerance"],
            selection_sigma_min=payload["selection_sigma_min"],
            selection_condition_number=payload["selection_condition_number"],
            reference_singular_values=payload["reference_singular_values"],
            reference_sigma_min=payload["reference_sigma_min"],
            reference_condition_number=payload["reference_condition_number"],
            reference_eigenvalues=payload["reference_eigenvalues"],
            reference_projector_hash=payload["reference_projector_hash"],
            warnings=payload["warnings"],
            identity_hash=expected_hash,
        )
        if spec._identity_payload() != canonical:
            raise ValueError("Gamma model anchor payload is not in canonical numeric form")
        return spec


@dataclass(frozen=True)
class GammaModelFrames:
    """Compact per-Q SCDM frames and their full-space alignment evidence."""

    anchor_spec_identity_hash: str
    layout_hash: str
    joint_band_indices: tuple[int, ...]
    group_ranks: tuple[int, int]
    local_frames_by_q: tuple[np.ndarray, ...]
    alignment_unitaries_by_q: tuple[np.ndarray, ...]
    alignment_singular_values_by_q: tuple[tuple[float, ...], ...]
    orthonormality_residuals_by_q: tuple[float, ...]
    projector_residuals_by_q: tuple[float, ...]
    frame_hashes_by_q: tuple[str, ...]
    alignment_hashes_by_q: tuple[str, ...]
    identity_hash: str

    SCHEMA = "kp.gamma-model-frames.v1"

    def __post_init__(self) -> None:
        raw_bands = _record_tuple(self.joint_band_indices, context="Gamma model frame joint bands")
        raw_ranks = _record_tuple(self.group_ranks, context="Gamma model frame group ranks")
        bands, ranks, _ = _gamma_joint_contract(raw_bands, raw_ranks)
        local_frames = _record_tuple(self.local_frames_by_q, context="Gamma model local frames")
        alignments = _record_tuple(
            self.alignment_unitaries_by_q,
            context="Gamma model alignment unitaries",
        )
        singular_values_by_q = tuple(
            tuple(
                _record_float(value, context="Gamma model alignment singular value")
                for value in _record_tuple(values, context="Gamma model alignment singular values")
            )
            for values in _record_tuple(
                self.alignment_singular_values_by_q,
                context="Gamma model Q alignment singular values",
            )
        )
        orthonormality_residuals = tuple(
            _record_float(value, context="Gamma model orthonormality residual")
            for value in _record_tuple(
                self.orthonormality_residuals_by_q,
                context="Gamma model orthonormality residuals",
            )
        )
        projector_residuals = tuple(
            _record_float(value, context="Gamma model projector residual")
            for value in _record_tuple(
                self.projector_residuals_by_q,
                context="Gamma model projector residuals",
            )
        )
        frame_hashes = _record_tuple(self.frame_hashes_by_q, context="Gamma model frame hashes")
        alignment_hashes = _record_tuple(
            self.alignment_hashes_by_q,
            context="Gamma model alignment hashes",
        )
        if any(not isinstance(value, str) for value in frame_hashes + alignment_hashes):
            raise ValueError("Gamma model frame/alignment hashes must contain strings")
        object.__setattr__(self, "joint_band_indices", bands)
        object.__setattr__(self, "group_ranks", ranks)
        object.__setattr__(self, "local_frames_by_q", local_frames)
        object.__setattr__(self, "alignment_unitaries_by_q", alignments)
        object.__setattr__(self, "alignment_singular_values_by_q", singular_values_by_q)
        object.__setattr__(self, "orthonormality_residuals_by_q", orthonormality_residuals)
        object.__setattr__(self, "projector_residuals_by_q", projector_residuals)
        object.__setattr__(self, "frame_hashes_by_q", frame_hashes)
        object.__setattr__(self, "alignment_hashes_by_q", alignment_hashes)
        q_count = len(local_frames)
        lengths = (
            q_count,
            len(alignments),
            len(singular_values_by_q),
            len(orthonormality_residuals),
            len(projector_residuals),
            len(frame_hashes),
            len(alignment_hashes),
        )
        if q_count <= 0 or len(set(lengths)) != 1:
            raise ValueError("Gamma model frame arrays and quality records must have one row per Q")
        frozen_frames: list[np.ndarray] = []
        frozen_alignments: list[np.ndarray] = []
        for q_index in range(q_count):
            frame = np.array(local_frames[q_index], dtype=np.complex128, copy=True, order="C")
            alignment = np.array(
                alignments[q_index], dtype=np.complex128, copy=True, order="C"
            )
            if frame.ndim != 2 or frame.shape[1] != len(bands):
                raise ValueError(f"Gamma model Q {q_index} frame does not have the joint column rank")
            if alignment.shape != (len(bands), len(bands)):
                raise ValueError(f"Gamma model Q {q_index} alignment shape mismatch")
            if not np.all(np.isfinite(frame)) or not np.all(np.isfinite(alignment)):
                raise ValueError(f"Gamma model Q {q_index} frame data contain NaN or Inf")
            if hash_array(frame) != frame_hashes[q_index]:
                raise ValueError(f"Gamma model Q {q_index} frame hash mismatch")
            if hash_array(alignment) != alignment_hashes[q_index]:
                raise ValueError(f"Gamma model Q {q_index} alignment hash mismatch")
            singular_values = singular_values_by_q[q_index]
            if len(singular_values) != len(bands) or any(not np.isfinite(value) for value in singular_values):
                raise ValueError(f"Gamma model Q {q_index} alignment quality is invalid")
            residuals = (
                orthonormality_residuals[q_index],
                projector_residuals[q_index],
            )
            if any(not np.isfinite(value) or value < 0.0 for value in residuals):
                raise ValueError(f"Gamma model Q {q_index} residual quality is invalid")
            frame.setflags(write=False)
            alignment.setflags(write=False)
            frozen_frames.append(frame)
            frozen_alignments.append(alignment)
        object.__setattr__(self, "local_frames_by_q", tuple(frozen_frames))
        object.__setattr__(self, "alignment_unitaries_by_q", tuple(frozen_alignments))
        if any(
            not isinstance(value, str) or not value
            for value in (
                self.anchor_spec_identity_hash,
                self.layout_hash,
                self.identity_hash,
            )
        ):
            raise ValueError("Gamma model frame identities must be nonempty")
        actual = hash_mapping(self._identity_payload())
        if actual != self.identity_hash:
            raise ValueError(f"Gamma model frame identity hash mismatch: {self.identity_hash} != {actual}")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "anchor_spec_identity_hash": self.anchor_spec_identity_hash,
            "layout_hash": self.layout_hash,
            "joint_band_indices": list(self.joint_band_indices),
            "group_ranks": list(self.group_ranks),
            "frame_hashes_by_q": list(self.frame_hashes_by_q),
            "alignment_hashes_by_q": list(self.alignment_hashes_by_q),
            "alignment_singular_values_by_q": [
                list(value) for value in self.alignment_singular_values_by_q
            ],
            "orthonormality_residuals_by_q": list(self.orthonormality_residuals_by_q),
            "projector_residuals_by_q": list(self.projector_residuals_by_q),
        }


@dataclass(frozen=True)
class GammaModelOwnerSpec:
    """One model-sector owner assigned independently of anchor support gauge."""

    lambda_index: int
    rank: int
    source_group: int
    qset_index: int
    physical_layers: tuple[int, ...]
    reference_columns: tuple[int, ...]

    def __post_init__(self) -> None:
        integer_fields = ("lambda_index", "rank", "source_group", "qset_index")
        values = {
            name: _record_int(getattr(self, name), context=f"Gamma common-anchor owner {name}")
            for name in integer_fields
        }
        if values["lambda_index"] < 0 or values["source_group"] < 0 or values["qset_index"] < 0:
            raise ValueError("Gamma common-anchor owner indices must be non-negative")
        if values["rank"] <= 0:
            raise ValueError("Gamma common-anchor owner rank must be positive")
        physical_layers = tuple(
            _record_int(value, context="Gamma common-anchor owner physical layer")
            for value in _record_tuple(
                self.physical_layers,
                context="Gamma common-anchor owner physical layers",
            )
        )
        reference_columns = tuple(
            _record_int(value, context="Gamma common-anchor owner reference column")
            for value in _record_tuple(
                self.reference_columns,
                context="Gamma common-anchor owner reference columns",
            )
        )
        if (
            not physical_layers
            or any(value < 0 for value in physical_layers)
            or len(set(physical_layers)) != len(physical_layers)
        ):
            raise ValueError("Gamma common-anchor owner physical layers must be unique and non-negative")
        if (
            len(reference_columns) != values["rank"]
            or len(set(reference_columns)) != values["rank"]
            or any(value < 0 for value in reference_columns)
        ):
            raise ValueError("Gamma common-anchor owner columns must uniquely cover its rank")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "physical_layers", physical_layers)
        object.__setattr__(self, "reference_columns", reference_columns)

    def to_payload(self) -> dict[str, Any]:
        return {
            "lambda_index": self.lambda_index,
            "rank": self.rank,
            "source_group": self.source_group,
            "qset_index": self.qset_index,
            "physical_layers": list(self.physical_layers),
            "reference_columns": list(self.reference_columns),
        }


@dataclass(frozen=True)
class GammaCommonAnchorSpec:
    """Common reference-orbital contract shared by every Gamma Q fibre."""

    joint_band_indices: tuple[int, ...]
    resolved_reference_terms: tuple[tuple[tuple[int, complex], ...], ...]
    selected_rows: tuple[int, ...]
    reference_physical_layers: tuple[int, ...]
    reference_source_groups: tuple[int, ...]
    owner_specs: tuple[GammaModelOwnerSpec, ...]
    reference_q_index: int
    layout_hash: str
    same_q_dimension: int
    physical_layer_count: int
    source_qset_count: int
    min_sigma: float
    max_condition: float
    projector_tolerance: float
    orthonormality_tolerance: float
    selection_sigma_min: float
    selection_condition_number: float
    reference_singular_values: tuple[float, ...]
    reference_sigma_min: float
    reference_condition_number: float
    reference_eigenvalues: tuple[float, ...]
    reference_projector_hash: str
    warnings: tuple[str, ...]
    identity_hash: str

    SCHEMA = "kp.gamma-common-anchor-spec.v1"

    def __post_init__(self) -> None:
        bands = _gamma_common_band_indices(self.joint_band_indices)
        rank = len(bands)
        references = _record_references(self.resolved_reference_terms)
        selected_rows = tuple(
            _record_int(row, context="Gamma common-anchor selected row")
            for row in _record_tuple(self.selected_rows, context="Gamma common-anchor selected rows")
        )
        physical_layers = tuple(
            _record_int(value, context="Gamma common-anchor reference physical layer")
            for value in _record_tuple(
                self.reference_physical_layers,
                context="Gamma common-anchor reference physical layers",
            )
        )
        source_groups = tuple(
            _record_int(value, context="Gamma common-anchor reference source group")
            for value in _record_tuple(
                self.reference_source_groups,
                context="Gamma common-anchor reference source groups",
            )
        )
        if any(len(value) != rank for value in (references, selected_rows, physical_layers, source_groups)):
            raise ValueError("Gamma common-anchor column metadata must cover the selected rank")
        if len(set(selected_rows)) != rank:
            raise ValueError("Gamma common-anchor selected rows must be unique")
        reference_q_index = _record_int(
            self.reference_q_index,
            context="Gamma common-anchor reference_q_index",
        )
        same_q_dimension = _record_int(
            self.same_q_dimension,
            context="Gamma common-anchor same_q_dimension",
        )
        physical_layer_count = _record_int(
            self.physical_layer_count,
            context="Gamma common-anchor physical_layer_count",
        )
        source_qset_count = _record_int(
            self.source_qset_count,
            context="Gamma common-anchor source_qset_count",
        )
        if reference_q_index < 0:
            raise ValueError("Gamma common-anchor reference_q_index must be non-negative")
        if same_q_dimension <= 0 or physical_layer_count <= 0 or source_qset_count <= 0:
            raise ValueError("Gamma common-anchor dimensions must be positive")
        if any(row < 0 or row >= same_q_dimension for row in selected_rows):
            raise ValueError("Gamma common-anchor selected row lies outside the local dimension")
        if any(value < 0 or value >= physical_layer_count for value in physical_layers):
            raise ValueError("Gamma common-anchor physical-layer metadata is out of range")
        if any(value < 0 or value >= source_qset_count for value in source_groups):
            raise ValueError("Gamma common-anchor source-group metadata is out of range")
        owners = tuple(self.owner_specs)
        if not owners or any(not isinstance(owner, GammaModelOwnerSpec) for owner in owners):
            raise ValueError("Gamma common-anchor requires one or more model owners")
        if tuple(owner.lambda_index for owner in owners) != tuple(range(len(owners))):
            raise ValueError("Gamma common-anchor owner lambda indices must be canonical")
        owner_columns = tuple(column for owner in owners for column in owner.reference_columns)
        if owner_columns != tuple(range(rank)):
            raise ValueError("Gamma common-anchor owners must canonically partition reference columns")
        for owner in owners:
            if owner.source_group >= source_qset_count or owner.qset_index >= source_qset_count:
                raise ValueError("Gamma common-anchor owner source/Q-set index is out of range")
            for column in owner.reference_columns:
                if physical_layers[column] not in owner.physical_layers:
                    raise ValueError("Gamma common-anchor owner does not cover its physical-layer column")
                if source_groups[column] != owner.source_group:
                    raise ValueError("Gamma common-anchor owner source group disagrees with its columns")
        scalar_names = (
            "min_sigma",
            "max_condition",
            "projector_tolerance",
            "orthonormality_tolerance",
            "selection_sigma_min",
            "selection_condition_number",
            "reference_sigma_min",
            "reference_condition_number",
        )
        scalars = {
            name: _record_float(getattr(self, name), context=f"Gamma common-anchor {name}")
            for name in scalar_names
        }
        singular_values = tuple(
            _record_float(value, context="Gamma common-anchor reference singular value")
            for value in _record_tuple(
                self.reference_singular_values,
                context="Gamma common-anchor reference singular values",
            )
        )
        eigenvalues = tuple(
            _record_float(value, context="Gamma common-anchor reference eigenvalue")
            for value in _record_tuple(
                self.reference_eigenvalues,
                context="Gamma common-anchor reference eigenvalues",
            )
        )
        warnings = tuple(str(value) for value in _record_tuple(self.warnings, context="Gamma common-anchor warnings"))
        if (
            len(singular_values) != rank
            or len(eigenvalues) != rank
            or any(not np.isfinite(value) for value in tuple(scalars.values()) + singular_values + eigenvalues)
            or scalars["min_sigma"] <= 0.0
            or scalars["max_condition"] < 1.0
            or scalars["projector_tolerance"] <= 0.0
            or scalars["orthonormality_tolerance"] <= 0.0
            or scalars["selection_sigma_min"] < scalars["min_sigma"]
            or scalars["selection_condition_number"] > scalars["max_condition"]
            or scalars["reference_sigma_min"] < scalars["min_sigma"]
            or scalars["reference_condition_number"] > scalars["max_condition"]
        ):
            raise ValueError("Gamma common-anchor quality metrics are invalid")
        if not isinstance(self.layout_hash, str) or not self.layout_hash:
            raise ValueError("Gamma common-anchor layout hash must be nonempty")
        if not isinstance(self.reference_projector_hash, str) or not self.reference_projector_hash:
            raise ValueError("Gamma common-anchor reference projector hash must be nonempty")
        _gamma_reference_matrix(references, row_count=same_q_dimension)
        object.__setattr__(self, "joint_band_indices", bands)
        object.__setattr__(self, "resolved_reference_terms", references)
        object.__setattr__(self, "selected_rows", selected_rows)
        object.__setattr__(self, "reference_physical_layers", physical_layers)
        object.__setattr__(self, "reference_source_groups", source_groups)
        object.__setattr__(self, "owner_specs", owners)
        object.__setattr__(self, "reference_q_index", reference_q_index)
        object.__setattr__(self, "same_q_dimension", same_q_dimension)
        object.__setattr__(self, "physical_layer_count", physical_layer_count)
        object.__setattr__(self, "source_qset_count", source_qset_count)
        for name, value in scalars.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "reference_singular_values", singular_values)
        object.__setattr__(self, "reference_eigenvalues", eigenvalues)
        object.__setattr__(self, "warnings", warnings)
        if not isinstance(self.identity_hash, str) or hash_mapping(self._identity_payload()) != self.identity_hash:
            raise ValueError("Gamma common-anchor identity hash mismatch")

    @property
    def physical_layer_anchor_counts(self) -> tuple[int, ...]:
        return tuple(self.reference_physical_layers.count(index) for index in range(self.physical_layer_count))

    @property
    def source_qset_anchor_counts(self) -> tuple[int, ...]:
        return tuple(
            sum(owner.rank for owner in self.owner_specs if owner.qset_index == index)
            for index in range(self.source_qset_count)
        )

    @property
    def active_model_layers(self) -> tuple[int, ...]:
        return tuple(layer for owner in self.owner_specs for layer in owner.physical_layers)

    @property
    def model_group_ranks(self) -> tuple[int, ...]:
        return tuple(owner.rank for owner in self.owner_specs)

    @property
    def model_group_qset_indices(self) -> tuple[int, ...]:
        return tuple(owner.qset_index for owner in self.owner_specs)

    @property
    def continuum_sector_ranks(self) -> tuple[int, int]:
        """Return the two-sector rank layout consumed by symm/model.

        Two active owners already define the two continuum sectors, even when
        both owners draw Q geometry from the same source Q set.  A single
        active owner is placed in its source-Q-set slot and the other
        continuum sector remains present with zero rank.
        """

        ranks = self.model_group_ranks
        if len(ranks) == 2:
            return int(ranks[0]), int(ranks[1])
        if len(ranks) == 1 and self.source_qset_count == 2:
            owner = self.owner_specs[0]
            resolved = [0, 0]
            resolved[int(owner.qset_index)] = int(owner.rank)
            return int(resolved[0]), int(resolved[1])
        raise ValueError(
            "Gamma common-anchor model owners cannot be represented by the "
            "two-sector continuum layout: "
            f"owner_ranks={list(ranks)}, source_qset_count={self.source_qset_count}"
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "joint_band_indices": list(self.joint_band_indices),
            "resolved_reference_terms": _gamma_reference_payload(self.resolved_reference_terms),
            "selected_rows": list(self.selected_rows),
            "reference_physical_layers": list(self.reference_physical_layers),
            "reference_source_groups": list(self.reference_source_groups),
            "owner_specs": [owner.to_payload() for owner in self.owner_specs],
            "reference_q_index": self.reference_q_index,
            "layout_hash": self.layout_hash,
            "same_q_dimension": self.same_q_dimension,
            "physical_layer_count": self.physical_layer_count,
            "source_qset_count": self.source_qset_count,
            "min_sigma": self.min_sigma,
            "max_condition": self.max_condition,
            "projector_tolerance": self.projector_tolerance,
            "orthonormality_tolerance": self.orthonormality_tolerance,
            "selection_sigma_min": self.selection_sigma_min,
            "selection_condition_number": self.selection_condition_number,
            "reference_singular_values": list(self.reference_singular_values),
            "reference_sigma_min": self.reference_sigma_min,
            "reference_condition_number": self.reference_condition_number,
            "reference_eigenvalues": list(self.reference_eigenvalues),
            "reference_projector_hash": self.reference_projector_hash,
            "warnings": list(self.warnings),
        }

    def to_payload(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["identity_hash"] = self.identity_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "GammaCommonAnchorSpec":
        """Restore one canonical, hash-bound common-anchor specification."""

        if not isinstance(payload, Mapping):
            raise TypeError("Gamma common-anchor payload must be a mapping")
        identity_keys = {
            "schema",
            "joint_band_indices",
            "resolved_reference_terms",
            "selected_rows",
            "reference_physical_layers",
            "reference_source_groups",
            "owner_specs",
            "reference_q_index",
            "layout_hash",
            "same_q_dimension",
            "physical_layer_count",
            "source_qset_count",
            "min_sigma",
            "max_condition",
            "projector_tolerance",
            "orthonormality_tolerance",
            "selection_sigma_min",
            "selection_condition_number",
            "reference_singular_values",
            "reference_sigma_min",
            "reference_condition_number",
            "reference_eigenvalues",
            "reference_projector_hash",
            "warnings",
        }
        expected_keys = identity_keys | {"identity_hash"}
        missing = sorted(expected_keys - set(payload))
        unknown = sorted(set(payload) - expected_keys)
        if missing or unknown:
            raise ValueError(
                "Gamma common-anchor payload keys mismatch "
                f"(missing={missing}, unknown={unknown})"
            )
        if payload["schema"] != cls.SCHEMA:
            raise ValueError(
                f"unsupported Gamma common-anchor schema {payload['schema']!r}"
            )
        canonical = {key: payload[key] for key in identity_keys}
        expected_hash = payload["identity_hash"]
        if not isinstance(expected_hash, str) or not expected_hash:
            raise ValueError(
                "Gamma common-anchor identity_hash must be a nonempty string"
            )
        if hash_mapping(canonical) != expected_hash:
            raise ValueError("Gamma common-anchor identity hash mismatch")

        raw_references = payload["resolved_reference_terms"]
        if not isinstance(raw_references, list):
            raise ValueError(
                "Gamma common-anchor resolved_reference_terms must be a list"
            )
        references: list[tuple[tuple[int, complex], ...]] = []
        for terms in raw_references:
            if not isinstance(terms, list) or not terms:
                raise ValueError(
                    "Gamma common-anchor reference columns must be nonempty lists"
                )
            parsed: list[tuple[int, complex]] = []
            for term in terms:
                if not isinstance(term, list) or len(term) != 3:
                    raise ValueError(
                        "Gamma common-anchor reference terms must be [row, real, imag]"
                    )
                parsed.append(
                    (
                        term[0],
                        complex(
                            _record_float(
                                term[1],
                                context="Gamma common-anchor coefficient real part",
                            ),
                            _record_float(
                                term[2],
                                context="Gamma common-anchor coefficient imaginary part",
                            ),
                        ),
                    )
                )
            references.append(tuple(parsed))

        raw_owners = payload["owner_specs"]
        if not isinstance(raw_owners, list) or not raw_owners:
            raise ValueError("Gamma common-anchor owner_specs must be a nonempty list")
        owner_keys = {
            "lambda_index",
            "rank",
            "source_group",
            "qset_index",
            "physical_layers",
            "reference_columns",
        }
        owners: list[GammaModelOwnerSpec] = []
        for raw_owner in raw_owners:
            if not isinstance(raw_owner, Mapping) or set(raw_owner) != owner_keys:
                raise ValueError("Gamma common-anchor owner payload keys mismatch")
            owners.append(
                GammaModelOwnerSpec(
                    lambda_index=raw_owner["lambda_index"],
                    rank=raw_owner["rank"],
                    source_group=raw_owner["source_group"],
                    qset_index=raw_owner["qset_index"],
                    physical_layers=raw_owner["physical_layers"],
                    reference_columns=raw_owner["reference_columns"],
                )
            )

        spec = cls(
            joint_band_indices=payload["joint_band_indices"],
            resolved_reference_terms=tuple(references),
            selected_rows=payload["selected_rows"],
            reference_physical_layers=payload["reference_physical_layers"],
            reference_source_groups=payload["reference_source_groups"],
            owner_specs=tuple(owners),
            reference_q_index=payload["reference_q_index"],
            layout_hash=payload["layout_hash"],
            same_q_dimension=payload["same_q_dimension"],
            physical_layer_count=payload["physical_layer_count"],
            source_qset_count=payload["source_qset_count"],
            min_sigma=payload["min_sigma"],
            max_condition=payload["max_condition"],
            projector_tolerance=payload["projector_tolerance"],
            orthonormality_tolerance=payload["orthonormality_tolerance"],
            selection_sigma_min=payload["selection_sigma_min"],
            selection_condition_number=payload["selection_condition_number"],
            reference_singular_values=payload["reference_singular_values"],
            reference_sigma_min=payload["reference_sigma_min"],
            reference_condition_number=payload["reference_condition_number"],
            reference_eigenvalues=payload["reference_eigenvalues"],
            reference_projector_hash=payload["reference_projector_hash"],
            warnings=payload["warnings"],
            identity_hash=expected_hash,
        )
        if spec._identity_payload() != canonical:
            raise ValueError(
                "Gamma common-anchor payload is not in canonical numeric form"
            )
        return spec


@dataclass(frozen=True)
class GammaCommonAnchorFrames:
    """Per-Q frames aligned to one common Gamma reference basis."""

    anchor_spec_identity_hash: str
    layout_hash: str
    joint_band_indices: tuple[int, ...]
    model_group_ranks: tuple[int, ...]
    local_frames_by_q: tuple[np.ndarray, ...]
    alignment_unitaries_by_q: tuple[np.ndarray, ...]
    alignment_singular_values_by_q: tuple[tuple[float, ...], ...]
    orthonormality_residuals_by_q: tuple[float, ...]
    projector_residuals_by_q: tuple[float, ...]
    frame_hashes_by_q: tuple[str, ...]
    alignment_hashes_by_q: tuple[str, ...]
    identity_hash: str

    SCHEMA = "kp.gamma-common-anchor-frames.v1"

    def __post_init__(self) -> None:
        bands = _gamma_common_band_indices(self.joint_band_indices)
        ranks = tuple(
            _record_int(value, context="Gamma common-anchor model rank")
            for value in _record_tuple(self.model_group_ranks, context="Gamma common-anchor model ranks")
        )
        if not ranks or any(value <= 0 for value in ranks) or sum(ranks) != len(bands):
            raise ValueError("Gamma common-anchor model ranks must be positive and sum to total rank")
        frames = tuple(self.local_frames_by_q)
        alignments = tuple(self.alignment_unitaries_by_q)
        singular_values_by_q = tuple(tuple(float(value) for value in values) for values in self.alignment_singular_values_by_q)
        orthonormality = tuple(float(value) for value in self.orthonormality_residuals_by_q)
        projector = tuple(float(value) for value in self.projector_residuals_by_q)
        frame_hashes = tuple(self.frame_hashes_by_q)
        alignment_hashes = tuple(self.alignment_hashes_by_q)
        q_count = len(frames)
        if q_count <= 0 or len({q_count, len(alignments), len(singular_values_by_q), len(orthonormality), len(projector), len(frame_hashes), len(alignment_hashes)}) != 1:
            raise ValueError("Gamma common-anchor frame evidence must contain one row per Q")
        frozen_frames: list[np.ndarray] = []
        frozen_alignments: list[np.ndarray] = []
        for q_index, (raw_frame, raw_alignment) in enumerate(zip(frames, alignments)):
            frame = np.array(raw_frame, dtype=np.complex128, copy=True, order="C")
            alignment = np.array(raw_alignment, dtype=np.complex128, copy=True, order="C")
            if frame.ndim != 2 or frame.shape[1] != len(bands) or alignment.shape != (len(bands), len(bands)):
                raise ValueError(f"Gamma common-anchor Q {q_index} frame/alignment shape mismatch")
            if hash_array(frame) != frame_hashes[q_index] or hash_array(alignment) != alignment_hashes[q_index]:
                raise ValueError(f"Gamma common-anchor Q {q_index} frame/alignment hash mismatch")
            if len(singular_values_by_q[q_index]) != len(bands):
                raise ValueError(f"Gamma common-anchor Q {q_index} singular-value rank mismatch")
            frame.setflags(write=False)
            alignment.setflags(write=False)
            frozen_frames.append(frame)
            frozen_alignments.append(alignment)
        object.__setattr__(self, "joint_band_indices", bands)
        object.__setattr__(self, "model_group_ranks", ranks)
        object.__setattr__(self, "local_frames_by_q", tuple(frozen_frames))
        object.__setattr__(self, "alignment_unitaries_by_q", tuple(frozen_alignments))
        object.__setattr__(self, "alignment_singular_values_by_q", singular_values_by_q)
        object.__setattr__(self, "orthonormality_residuals_by_q", orthonormality)
        object.__setattr__(self, "projector_residuals_by_q", projector)
        object.__setattr__(self, "frame_hashes_by_q", frame_hashes)
        object.__setattr__(self, "alignment_hashes_by_q", alignment_hashes)
        if not isinstance(self.identity_hash, str) or hash_mapping(self._identity_payload()) != self.identity_hash:
            raise ValueError("Gamma common-anchor frame identity hash mismatch")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "anchor_spec_identity_hash": self.anchor_spec_identity_hash,
            "layout_hash": self.layout_hash,
            "joint_band_indices": list(self.joint_band_indices),
            "model_group_ranks": list(self.model_group_ranks),
            "frame_hashes_by_q": list(self.frame_hashes_by_q),
            "alignment_hashes_by_q": list(self.alignment_hashes_by_q),
            "alignment_singular_values_by_q": [list(values) for values in self.alignment_singular_values_by_q],
            "orthonormality_residuals_by_q": list(self.orthonormality_residuals_by_q),
            "projector_residuals_by_q": list(self.projector_residuals_by_q),
        }


@contextmanager
def _bounded_blas_threads():
    depth = _PROJECTOR_BLAS_SCOPE_DEPTH.get()
    token = _PROJECTOR_BLAS_SCOPE_DEPTH.set(depth + 1)
    try:
        if depth > 0:
            yield
            return
        try:
            from threadpoolctl import threadpool_limits
        except Exception:  # pragma: no cover - optional runtime dependency
            scope = nullcontext()
        else:
            scope = threadpool_limits(limits=PROJECTOR_BLAS_THREADS, user_api="blas")
        with scope:
            yield
    finally:
        _PROJECTOR_BLAS_SCOPE_DEPTH.reset(token)


def set_projector_blas_threads(threads: int) -> None:
    global PROJECTOR_BLAS_THREADS
    PROJECTOR_BLAS_THREADS = max(1, int(threads))
    _set_downfold_projector_blas_threads(PROJECTOR_BLAS_THREADS)


def _hermitize(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (mat + mat.conj().T)


def _project_sz_from_low_groups(
    *,
    full_dim: int,
    low_groups: Any,
    spin: Literal["up", "down", "all"],
    spin_operator_sign: int | None = None,
) -> np.ndarray:
    n_columns = int(low_groups.n_columns)
    if n_columns == 0:
        return np.zeros((0, 0), dtype=np.complex128)
    spin_norm = str(spin).lower()
    if spin_norm != "all":
        sign = 1 if spin_operator_sign is None else (1 if int(spin_operator_sign) >= 0 else -1)
        return np.eye(n_columns, dtype=np.complex128) * float(sign)
    if int(full_dim) % 2 != 0:
        raise ValueError("spin='all' Sz projection requires an even full Hamiltonian dimension")
    half = int(full_dim) // 2
    projected = np.zeros((n_columns, n_columns), dtype=np.complex128)
    for group in low_groups.groups:
        rows = np.asarray(group.rows, dtype=np.intp)
        cols = np.asarray(group.cols, dtype=np.intp)
        local = np.asarray(group.local, dtype=np.complex128)
        signs = np.where(rows < half, 1.0, -1.0).astype(np.complex128)
        projected[np.ix_(cols, cols)] += local.conj().T @ (signs[:, np.newaxis] * local)
    return _hermitize(projected)


def _extract_square_block(matrix: np.ndarray, index: np.ndarray) -> np.ndarray:
    idx = np.asarray(index, dtype=np.intp)
    if idx.ndim != 1:
        raise ValueError("block index must be one-dimensional")
    if idx.size == 0:
        return np.zeros((0, 0), dtype=np.asarray(matrix).dtype)
    start = int(idx[0])
    stop = start + int(idx.size)
    if np.array_equal(idx, np.arange(start, stop, dtype=np.intp)):
        return matrix[start:stop, start:stop]
    return matrix[np.ix_(idx, idx)]


def _complement_indices(size: int, selected: list[int] | np.ndarray) -> np.ndarray:
    selected_arr = np.asarray(selected, dtype=np.intp)
    if selected_arr.size == 0:
        return np.arange(size, dtype=np.intp)
    mask = np.ones(size, dtype=bool)
    mask[selected_arr] = False
    return np.nonzero(mask)[0]


def _hermitian_eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with _bounded_blas_threads():
        return scipy.linalg.eigh(
            np.asarray(matrix, dtype=np.complex128),
            check_finite=False,
            overwrite_a=True,
            driver="evr",
        )


def _hermitian_eigh_columns(matrix: np.ndarray, columns: list[int] | None) -> tuple[np.ndarray, np.ndarray]:
    if columns is None:
        return _hermitian_eigh(matrix)
    if not columns:
        size = int(np.asarray(matrix).shape[0])
        return np.zeros(size, dtype=float), np.zeros((size, size), dtype=np.complex128)
    size = int(np.asarray(matrix).shape[0])
    unique_cols = sorted({int(col) for col in columns})
    if unique_cols[0] < 0 or unique_cols[-1] >= size:
        raise IndexError(f"eigenvector column request {unique_cols} outside block size {size}")
    lo, hi = unique_cols[0], unique_cols[-1]
    with _bounded_blas_threads():
        eig_window, vec_window = scipy.linalg.eigh(
            np.asarray(matrix, dtype=np.complex128),
            subset_by_index=(lo, hi),
            check_finite=False,
            overwrite_a=True,
            driver="evr",
        )
    eig = np.zeros(size, dtype=float)
    vec = np.zeros((size, size), dtype=np.complex128)
    if len(unique_cols) == hi - lo + 1:
        eig[lo : hi + 1] = eig_window
        vec[:, lo : hi + 1] = vec_window
    else:
        cols = np.asarray(unique_cols, dtype=np.intp)
        window_cols = cols - lo
        eig[cols] = eig_window[window_cols]
        vec[:, cols] = vec_window[:, window_cols]
    return eig, vec


def _cached_hermitian_eigh_columns(
    *,
    hamk: np.ndarray,
    block: np.ndarray,
    block_indices: np.ndarray,
    columns: list[int] | None,
    cache: dict[Any, tuple[np.ndarray, np.ndarray]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reuse an unaligned block eigensystem within one projection run."""

    if cache is None:
        return _hermitian_eigh_columns(block, columns)
    array = np.asarray(hamk)
    storage_key = (
        int(array.__array_interface__["data"][0]),
        tuple(int(value) for value in array.shape),
        tuple(int(value) for value in array.strides),
        array.dtype.str,
    )
    column_key = None if columns is None else tuple(sorted({int(col) for col in columns}))
    key = (
        storage_key,
        tuple(int(value) for value in np.asarray(block_indices, dtype=np.intp)),
        column_key,
    )
    cached = cache.get(key)
    if cached is None:
        eig, vec = _hermitian_eigh_columns(block, columns)
        cache[key] = (np.array(eig, copy=True), np.array(vec, copy=True))
        return eig, vec
    eig, vec = cached
    # Anchor alignment mutates selected columns, so every caller needs its own
    # working copy while the cache retains the unaligned eigensystem.
    return np.array(eig, copy=True), np.array(vec, copy=True)


def align_eigenstates(U_low: np.ndarray, Phi_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align low-energy eigenstates to a reference basis using Procrustes via SVD.

    Parameters
    - U_low: (M, N) eigenvector matrix (columns are states to align)
    - Phi_ref: (M, N) reference vectors (columns)

    Returns
    - U_aligned: (M, N) aligned eigenvectors
    - V: (N, N) unitary rotation applied in the low-energy subspace
    """

    # Overlap O = Phi_ref^† U_low
    O = Phi_ref.conj().T @ U_low
    # SVD: O = X Σ Y^†
    X, s, Yh = np.linalg.svd(O, full_matrices=False)
    # Optimal unitary: V = Y X^†
    V = Yh.conj().T @ X.conj().T
    # Apply rotation in subspace
    U_aligned = U_low @ V
    return U_aligned, V


def _is_reference_pair(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and not isinstance(value[0], (list, tuple))
    )


def _layer_reference_entries(layer_refs: Any, n_bands: int) -> list[Any]:
    if n_bands == 1 and _is_reference_pair(layer_refs):
        return [layer_refs]
    if isinstance(layer_refs, (list, tuple)):
        return list(layer_refs)
    return [layer_refs]


def _reference_coef_to_complex(value: Any) -> complex:
    if isinstance(value, Mapping):
        return complex(float(value.get("real", 0.0)), float(value.get("imag", 0.0)))
    return complex(value)


def _parse_reference_terms(reference: Any, *, context: str) -> list[tuple[int, complex]]:
    raw_items = [reference] if _is_reference_pair(reference) else reference
    if not isinstance(raw_items, (list, tuple)):
        raw_items = [raw_items]

    terms: list[tuple[int, complex]] = []
    for item in raw_items:
        if _is_reference_pair(item):
            idxc, coef = item
            terms.append((int(idxc), _reference_coef_to_complex(coef)))
        else:
            terms.append((int(item), complex(1.0)))
    if not terms:
        raise ValueError(f"{context}: empty reference in norb_fix_list")
    return terms


def _canonical_resolved_anchor_key(norb_fix_list: Any) -> tuple[Any, ...]:
    layers: list[Any] = []
    for layer_index, layer_refs in enumerate(norb_fix_list):
        entries = _layer_reference_entries(layer_refs, 1 if _is_reference_pair(layer_refs) else len(layer_refs))
        layer_key: list[Any] = []
        for ref_index, reference in enumerate(entries):
            terms = _parse_reference_terms(reference, context=f"canonical anchor layer {layer_index} reference {ref_index}")
            term_key = tuple(
                sorted(
                    (
                        int(row),
                        round(float(np.real(coef)), 14),
                        round(float(np.imag(coef)), 14),
                    )
                    for row, coef in terms
                )
            )
            layer_key.append(term_key)
        layers.append(tuple(layer_key))
    return tuple(layers)


def _resolve_reference_index(
    idx: int,
    *,
    block_dim: int,
    context: str,
    allow_layer_global: bool = False,
    layer: int | None = None,
    layer_block_dim: int | None = None,
    total_layers: int | None = None,
) -> int:
    if 0 <= idx < block_dim:
        return idx

    if allow_layer_global:
        if layer is None or layer_block_dim is None or total_layers is None:
            raise ValueError(f"{context}: layer-global reference resolution is missing layer metadata")
        total_dim = int(layer_block_dim) * int(total_layers)
        if 0 <= idx < total_dim:
            owner_layer = int(idx) // int(layer_block_dim)
            if owner_layer != int(layer):
                raise ValueError(
                    f"{context}: reference index {idx} belongs to layer {owner_layer}, "
                    f"not layer {layer}"
                )
            local_idx = int(idx) - owner_layer * int(layer_block_dim)
            if 0 <= local_idx < block_dim:
                return local_idx
        raise ValueError(
            f"{context}: reference index {idx} is outside local block dimension {block_dim} "
            f"and combined same-Q dimension {total_dim}"
        )

    raise ValueError(f"{context}: reference index {idx} is outside block dimension {block_dim}")


def _reference_terms_for_layer(
    *,
    nlow_state_list: Any,
    norb_fix_list: Any,
    layer: int,
    block_dim: int,
    context: str,
    allow_layer_global: bool = False,
    layer_for_global: int | None = None,
    layer_block_dim: int | None = None,
    total_layers: int | None = None,
) -> tuple[list[int], list[list[tuple[int, complex]]]]:
    if layer >= len(nlow_state_list):
        raise IndexError(f"{context}: missing nlow_state_list entry for layer {layer}")

    bands = [int(band) for band in nlow_state_list[layer]]
    if not bands:
        return [], []
    if layer >= len(norb_fix_list):
        raise IndexError(f"{context}: missing norb_fix_list entry for layer {layer}")

    ref_entries = _layer_reference_entries(norb_fix_list[layer], len(bands))
    if len(ref_entries) != len(bands):
        raise ValueError(
            f"{context}: nlow_state_list layer {layer} has {len(bands)} bands but "
            f"norb_fix_list layer {layer} has {len(ref_entries)} references"
        )

    resolved: list[list[tuple[int, complex]]] = []
    for ref_idx, reference in enumerate(ref_entries):
        ref_context = f"{context} reference {ref_idx}"
        terms = []
        for raw_idx, coef in _parse_reference_terms(reference, context=ref_context):
            terms.append(
                (
                    _resolve_reference_index(
                        int(raw_idx),
                        block_dim=block_dim,
                        context=ref_context,
                        allow_layer_global=allow_layer_global,
                        layer=layer_for_global,
                        layer_block_dim=layer_block_dim,
                        total_layers=total_layers,
                    ),
                    coef,
                )
            )
        resolved.append(terms)
    return bands, resolved


def _align_selected_eigenstates(
    vec: np.ndarray,
    bands: list[int],
    references: list[list[tuple[int, complex]]],
    *,
    context: str,
) -> None:
    if len(bands) != len(references):
        raise ValueError(f"{context}: band/reference length mismatch")
    if not bands:
        return

    for band in bands:
        if band < -vec.shape[1] or band >= vec.shape[1]:
            raise IndexError(f"{context}: low-state band index {band} outside block dimension {vec.shape[1]}")

    phi_ref = np.zeros((vec.shape[0], len(bands)), dtype=complex)
    for col_idx, terms in enumerate(references):
        col = np.zeros(phi_ref.shape[0], dtype=complex)
        for idxc, coef in terms:
            col[idxc] += coef
        norm = np.linalg.norm(col)
        if norm <= 0.0:
            raise ValueError(f"{context}: reference {col_idx} has zero norm after resolving norb_fix_list")
        phi_ref[:, col_idx] = col / norm

    u_low = vec[:, np.array(bands, dtype=int)]
    u_aligned, _ = align_eigenstates(u_low, phi_ref)
    vec[:, bands] = u_aligned


def _is_auto_token(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"auto", "auto_scdm"}


def _gauge_requests_auto(gauge_config: Any) -> bool:
    if _is_auto_token(gauge_config):
        return True
    if isinstance(gauge_config, dict):
        method = gauge_config.get("method", gauge_config.get("mode"))
        return _is_auto_token(method)
    return False


def _auto_gauge_config(gauge_config: Any) -> AutoGaugeConfig:
    if gauge_config is None or _is_auto_token(gauge_config):
        return AutoGaugeConfig()
    if not isinstance(gauge_config, dict):
        raise ValueError(f"project.gauge must be 'auto' or a mapping, got {gauge_config!r}")
    method = str(gauge_config.get("method", "auto_scdm")).lower()
    if method != "auto_scdm":
        raise ValueError(f"Unsupported project.gauge.method={method!r}; expected 'auto_scdm'")
    anchor_scope = str(gauge_config.get("anchor_scope", "per_sector")).lower()
    if anchor_scope != "per_sector":
        raise ValueError("Phase 1 auto gauge only supports project.gauge.anchor_scope='per_sector'")
    candidate_pool = str(gauge_config.get("candidate_pool", "all")).lower()
    if candidate_pool != "all":
        raise ValueError("Phase 1 auto gauge only supports project.gauge.candidate_pool='all'")
    projected_anchors = bool(gauge_config.get("projected_anchors", False))
    if projected_anchors:
        raise ValueError("project.gauge.projected_anchors=true is not supported in Phase 1")
    return AutoGaugeConfig(
        method=method,
        anchor_scope=anchor_scope,
        candidate_pool=candidate_pool,
        projected_anchors=projected_anchors,
        min_sigma=float(gauge_config.get("min_sigma", 1.0e-6)),
        max_condition=float(gauge_config.get("max_condition", 1.0e6)),
        reference_q_index=int(gauge_config.get("reference_q_index", gauge_config.get("ref_q_index", 0))),
        basis_is_orthonormal=bool(gauge_config.get("basis_is_orthonormal", True)),
    )


def _manual_gauge_report(norb_fix_list: Any) -> GaugeAnchorReport:
    return GaugeAnchorReport(
        gauge_mode="manual_norb_fix_list",
        resolved_norb_fix_list=norb_fix_list,
        selections=[],
        metric={"type": "orthonormal", "basis_is_orthonormal": True},
        state_selection_quality={
            "status": "not_evaluated",
            "reason": "manual gauge anchors were provided",
        },
        gauge_anchor_quality={
            "status": "manual",
            "sigma_min": None,
            "condition_number": None,
        },
        symmetry_closure_quality={
            "status": "not_available",
            "subspace_leakage": None,
            "reason": "symmetry validation is not available during gauge resolution",
        },
    )


def _auto_reference_terms(row: int) -> list[tuple[int, complex]]:
    return [(int(row), 1.0 + 0.0j)]


def _format_auto_reference_terms(terms: list[tuple[int, complex]]) -> list[list[Any]]:
    out: list[list[Any]] = []
    for row, coef in terms:
        value = complex(coef)
        if abs(value.imag) < 1.0e-14:
            coef_out: Any = float(value.real)
        elif abs(value.real) < 1.0e-14 and abs(value.imag - 1.0) < 1.0e-14:
            coef_out = "1j"
        elif abs(value.real) < 1.0e-14 and abs(value.imag + 1.0) < 1.0e-14:
            coef_out = "-1j"
        else:
            coef_out = {"real": float(value.real), "imag": float(value.imag)}
        out.append([int(row), coef_out])
    return out


def _format_complex_for_report(value: complex) -> Any:
    return _format_auto_reference_terms([(0, complex(value))])[0][1]


def _reference_overlap_scores(u_low: np.ndarray, references: list[list[tuple[int, complex]]]) -> np.ndarray:
    u = np.asarray(u_low, dtype=np.complex128)
    scores = np.zeros((len(references), u.shape[1]), dtype=float)
    for ref_pos, terms in enumerate(references):
        overlap = np.zeros(u.shape[1], dtype=np.complex128)
        norm_sq = 0.0
        for row, coef in terms:
            overlap += np.conjugate(complex(coef)) * u[int(row), :]
            norm_sq += abs(complex(coef)) ** 2
        if norm_sq <= 0.0:
            raise ValueError("auto gauge produced a zero-norm reference")
        scores[ref_pos, :] = np.abs(overlap) / np.sqrt(norm_sq)
    return scores


def _reference_overlap_singular_values(u_low: np.ndarray, references_by_band: list[list[tuple[int, complex]]]) -> np.ndarray:
    u = np.asarray(u_low, dtype=np.complex128)
    overlap = np.zeros((len(references_by_band), u.shape[1]), dtype=np.complex128)
    for ref_pos, terms in enumerate(references_by_band):
        norm_sq = 0.0
        for row, coef in terms:
            overlap[ref_pos, :] += np.conjugate(complex(coef)) * u[int(row), :]
            norm_sq += abs(complex(coef)) ** 2
        if norm_sq <= 0.0:
            raise ValueError("auto gauge produced a zero-norm reference")
        overlap[ref_pos, :] /= np.sqrt(norm_sq)
    return np.linalg.svd(overlap, compute_uv=False)


def _same_segment(row: int, partner: int, segments: list[tuple[int, int]]) -> bool:
    for start, stop in segments:
        if start <= int(row) < stop:
            return start <= int(partner) < stop
    return False


def _segment_index_for_row(row: int, segments: list[tuple[int, int]]) -> int | None:
    for index, (start, stop) in enumerate(segments):
        if int(start) <= int(row) < int(stop):
            return int(index)
    return None


def _gamma_reference_sort_key(
    terms: list[tuple[int, complex]],
    *,
    segments: list[tuple[int, int]],
) -> tuple[int, int, int, int]:
    """Order Gamma references in the continuum model frame.

    Gamma same-Q blocks are laid out as spin blocks, with physical layers inside
    each spin block.  The model frame expects physical layer sectors first, and
    spin partners inside each sector.  QRCP row order is not stable enough for
    that semantic ordering, especially in nearly degenerate Gamma subspaces.
    """

    if not terms:
        return (0, 0, 0, 0)
    total_segments = max(len(segments), 1)
    spin_count = 2 if total_segments % 2 == 0 else 1
    layer_count = max(total_segments // spin_count, 1)
    layer_indices: list[int] = []
    spin_indices: list[int] = []
    rows: list[int] = []
    for row, _coef in terms:
        row_int = int(row)
        rows.append(row_int)
        segment_index = _segment_index_for_row(row_int, segments)
        if segment_index is None:
            continue
        spin_indices.append(int(segment_index // layer_count))
        layer_indices.append(int(segment_index % layer_count))
    layer_key = -max(layer_indices) if layer_indices else 0
    spin_key = -max(spin_indices) if spin_indices else 0
    span_key = len(set(layer_indices)) if layer_indices else 0
    row_key = min(rows)
    return (layer_key, spin_key, span_key, row_key)


def _same_spin_layer_exchange_candidates(
    row: int,
    *,
    segments: list[tuple[int, int]],
) -> list[int]:
    segment_index = _segment_index_for_row(row, segments)
    if segment_index is None or len(segments) < 2 or len(segments) % 2 != 0:
        return []
    layer_count = len(segments) // 2
    spin_index = int(segment_index) // layer_count
    layer_index = int(segment_index) % layer_count
    candidates: list[int] = []
    for other_layer in range(layer_count):
        if other_layer == layer_index:
            continue
        other_segment = spin_index * layer_count + other_layer
        start, stop = segments[other_segment]
        candidates.extend(range(int(start), int(stop)))
    return candidates


def _order_gamma_references_for_model_basis(
    references: list[list[tuple[int, complex]]],
    *,
    segments: list[tuple[int, int]],
) -> list[list[tuple[int, complex]]]:
    return [
        list(ref)
        for ref in sorted(
            references,
            key=lambda ref: _gamma_reference_sort_key(ref, segments=segments),
        )
    ]


def _gamma_same_q_row_segments(
    block_dim: int,
    num_layer_list: list[int],
    num_orb_per_layer_list: list[list[int]],
    *,
    spin: Literal["up", "down", "all"],
) -> list[tuple[int, int]]:
    widths = [int(width) for group in num_orb_per_layer_list for width in group]
    total_layers = int(sum(int(n) for n in num_layer_list))
    if len(widths) != total_layers:
        if total_layers <= 0:
            return [(0, int(block_dim))]
        spin_count = 2 if spin == "all" else 1
        if int(block_dim) % (spin_count * total_layers) != 0:
            return [(0, int(block_dim))]
        widths = [int(block_dim) // (spin_count * total_layers)] * total_layers
    per_spin_dim = int(sum(widths))
    spin_count = 2 if spin == "all" else 1
    if int(block_dim) != spin_count * per_spin_dim:
        return [(0, int(block_dim))]
    segments: list[tuple[int, int]] = []
    for spin_index in range(spin_count):
        offset = spin_index * per_spin_dim
        cursor = offset
        for width in widths:
            segments.append((cursor, cursor + int(width)))
            cursor += int(width)
    return segments


def _physical_layer_widths(
    num_layer_list: list[int],
    num_orb_per_layer_list: list[list[int]],
    *,
    fallback_width: int,
) -> list[int]:
    widths = [int(width) for group in num_orb_per_layer_list for width in group]
    total_layers = int(sum(int(n) for n in num_layer_list))
    if len(widths) == total_layers:
        return widths
    return [int(fallback_width) for _ in range(total_layers)]


def _local_layer_row_segments(
    block_dim: int,
    layer_width: int,
    *,
    spin: Literal["up", "down", "all"],
) -> list[tuple[int, int]]:
    if spin != "all":
        return [(0, int(block_dim))]
    width = int(layer_width)
    if width > 0 and int(block_dim) == 2 * width:
        return [(0, width), (width, 2 * width)]
    if int(block_dim) % 2 == 0:
        half = int(block_dim) // 2
        return [(0, half), (half, int(block_dim))]
    return [(0, int(block_dim))]


def _complete_gamma_spinful_reference_terms(
    u_low: np.ndarray,
    selected_rows: list[int],
    *,
    segments: list[tuple[int, int]],
    leverage_rtol: float = 1.0e-2,
    magnitude_rtol: float = 5.0e-2,
) -> tuple[list[list[tuple[int, complex]]], list[dict[str, Any]], list[str]]:
    u = np.asarray(u_low, dtype=np.complex128)
    leverage = np.real(np.sum(np.abs(u) ** 2, axis=1))
    selected = {int(row) for row in selected_rows}
    references: list[list[tuple[int, complex]]] = []
    details: list[dict[str, Any]] = []
    warnings: list[str] = []
    for row in selected_rows:
        row = int(row)
        row_mag = np.abs(u[row, :])
        row_norm = max(float(np.linalg.norm(row_mag)), 1.0e-15)
        candidates: list[tuple[float, int, complex, float, float]] = []
        adjacent_rows = [row - 1, row + 1]
        segment_index = _segment_index_for_row(row, segments)
        if segment_index is None:
            partner_groups = [adjacent_rows]
        else:
            start, stop = segments[segment_index]
            same_segment_rows = [
                candidate
                for candidate in range(int(start), int(stop))
                if candidate not in adjacent_rows and candidate != row
            ]
            partner_groups = [adjacent_rows + same_segment_rows, _same_spin_layer_exchange_candidates(row, segments=segments)]
        partner_scope = "same_segment"
        for scope_index, partner_rows in enumerate(partner_groups):
            scope_candidates: list[tuple[float, int, complex, float, float]] = []
            for partner in partner_rows:
                if partner < 0 or partner >= u.shape[0] or partner in selected:
                    continue
                if scope_index == 0 and not _same_segment(row, partner, segments):
                    continue
                lev_ref = max(float(abs(leverage[row])), float(abs(leverage[partner])), 1.0e-15)
                leverage_rel = float(abs(leverage[row] - leverage[partner]) / lev_ref)
                if leverage_rel > float(leverage_rtol):
                    continue
                partner_mag = np.abs(u[partner, :])
                magnitude_rel = float(np.linalg.norm(row_mag - partner_mag) / row_norm)
                if magnitude_rel > float(magnitude_rtol):
                    continue
                for phase in (1.0 + 0.0j, -1.0 + 0.0j, 1.0j, -1.0j):
                    overlap = (u[row, :] + np.conjugate(phase) * u[partner, :]) / np.sqrt(2.0)
                    score = float(np.linalg.norm(overlap))
                    scope_candidates.append((score, int(partner), complex(phase), leverage_rel, magnitude_rel))
            if scope_candidates:
                candidates = scope_candidates
                partner_scope = "same_segment" if scope_index == 0 else "same_spin_layer_exchange"
                break
        if not candidates:
            references.append(_auto_reference_terms(row))
            warnings.append(f"auto gauge gamma spinful row {row} did not find a safe adjacent chiral partner")
            details.append({"row": int(row), "partner_row": None, "partner_phase": None, "pair_score": None})
            continue
        score, partner, phase, leverage_rel, magnitude_rel = sorted(
            candidates,
            key=lambda item: (
                -float(item[0]),
                0 if abs(int(item[1]) - row) == 1 else 1,
                abs(int(item[1]) - row),
                int(item[1]),
                float(np.real(item[2])),
                float(np.imag(item[2])),
            ),
        )[0]
        references.append([(int(row), 1.0 + 0.0j), (int(partner), phase)])
        details.append(
            {
                "row": int(row),
                "partner_row": int(partner),
                "partner_phase": _format_complex_for_report(phase),
                "pair_score": float(score),
                "partner_scope": partner_scope,
                "leverage_relative_difference": float(leverage_rel),
                "row_magnitude_relative_difference": float(magnitude_rel),
            }
        )
    return references, details, warnings


def _complete_adjacent_chiral_reference_terms(
    u_low: np.ndarray,
    selected_rows: list[int],
    *,
    segments: list[tuple[int, int]],
    conjugate: bool = False,
    primary: Literal["selected", "higher"] = "selected",
    partner_policy: Literal["best_adjacent", "lower"] = "best_adjacent",
    leverage_rtol: float | None = None,
    magnitude_rtol: float | None = None,
) -> tuple[list[list[tuple[int, complex]]], list[dict[str, Any]], list[str]]:
    u = np.asarray(u_low, dtype=np.complex128)
    leverage = np.real(np.sum(np.abs(u) ** 2, axis=1))
    selected = {int(row) for row in selected_rows}
    references: list[list[tuple[int, complex]]] = []
    details: list[dict[str, Any]] = []
    warnings: list[str] = []
    for row in selected_rows:
        row = int(row)
        row_mag = np.abs(u[row, :])
        row_norm = max(float(np.linalg.norm(row_mag)), 1.0e-15)
        segment_index = _segment_index_for_row(row, segments)
        phase = -1.0j if segment_index is None or int(segment_index) % 2 == 0 else 1.0j
        if conjugate:
            phase = np.conjugate(phase)
        choices: list[tuple[float, int, float, float, list[tuple[int, complex]]]] = []
        partner_rows = (row - 1,) if partner_policy == "lower" else (row - 1, row + 1)
        for partner in partner_rows:
            if partner < 0 or partner >= u.shape[0] or partner in selected:
                continue
            if not _same_segment(row, partner, segments):
                continue
            lev_ref = max(float(abs(leverage[row])), float(abs(leverage[partner])), 1.0e-15)
            leverage_rel = float(abs(leverage[row] - leverage[partner]) / lev_ref)
            if leverage_rtol is not None and leverage_rel > float(leverage_rtol):
                continue
            partner_mag = np.abs(u[partner, :])
            magnitude_rel = float(np.linalg.norm(row_mag - partner_mag) / row_norm)
            if magnitude_rtol is not None and magnitude_rel > float(magnitude_rtol):
                continue
            if primary == "higher" and int(partner) > int(row):
                terms = [(int(partner), 1.0 + 0.0j), (int(row), complex(phase))]
            elif primary == "higher":
                terms = [(int(row), 1.0 + 0.0j), (int(partner), complex(phase))]
            elif primary == "selected":
                terms = [(int(row), 1.0 + 0.0j), (int(partner), complex(phase))]
            else:
                raise ValueError(f"unknown adjacent chiral primary policy {primary!r}")
            overlap = np.zeros(u.shape[1], dtype=np.complex128)
            for term_row, coef in terms:
                overlap += np.conjugate(complex(coef)) * u[int(term_row), :]
            overlap /= np.sqrt(sum(abs(complex(coef)) ** 2 for _term_row, coef in terms))
            choices.append((float(np.linalg.norm(overlap)), int(partner), leverage_rel, magnitude_rel, terms))
        if not choices:
            references.append(_auto_reference_terms(row))
            warnings.append(f"auto gauge row {row} did not find a safe adjacent chiral partner")
            details.append({"row": int(row), "partner_row": None, "partner_phase": None, "pair_score": None})
            continue
        score, partner, leverage_rel, magnitude_rel, terms = sorted(
            choices,
            key=lambda item: (-float(item[0]), abs(int(item[1]) - row), int(item[1])),
        )[0]
        references.append(terms)
        details.append(
            {
                "row": int(row),
                "partner_row": int(partner),
                "partner_phase": _format_complex_for_report(complex(terms[1][1])),
                "pair_score": float(score),
                "partner_scope": "adjacent_same_segment",
                "primary_policy": primary,
                "partner_policy": partner_policy,
                "leverage_relative_difference": float(leverage_rel),
                "row_magnitude_relative_difference": float(magnitude_rel),
            }
        )
    return references, details, warnings


def _assign_anchor_references_to_bands(
    u_low: np.ndarray,
    references: list[list[tuple[int, complex]]],
) -> tuple[list[list[tuple[int, complex]]], list[float]]:
    if not references:
        return [], []
    u = np.asarray(u_low, dtype=np.complex128)
    if len(references) != u.shape[1]:
        raise ValueError(
            f"anchor assignment requires one reference per low-state column, "
            f"got {len(references)} references for {u.shape[1]} columns"
        )
    scores = _reference_overlap_scores(u, references)
    tie_break = np.asarray([min(int(row) for row, _coef in ref) for ref in references], dtype=float)[:, np.newaxis]
    scale = max(float(np.max(np.abs(scores))), 1.0)
    cost = -scores + 1.0e-14 * scale * tie_break / max(float(u.shape[0]), 1.0)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
    assigned: list[list[tuple[int, complex]] | None] = [None] * u.shape[1]
    assigned_scores: list[float | None] = [None] * u.shape[1]
    for ref_pos, band_pos in zip(row_ind, col_ind):
        assigned[int(band_pos)] = references[int(ref_pos)]
        assigned_scores[int(band_pos)] = float(scores[int(ref_pos), int(band_pos)])
    if any(ref is None for ref in assigned):
        raise ValueError("anchor assignment failed to cover all low-state columns")
    return (
        [list(ref) for ref in assigned if ref is not None],
        [float(score) for score in assigned_scores if score is not None],
    )


@dataclass(frozen=True)
class _GammaReferenceResolution:
    selection: AutoGaugeSelection
    references_by_band: tuple[tuple[tuple[int, complex], ...], ...]
    assigned_scores: tuple[float, ...]
    completion_details: tuple[Mapping[str, Any], ...]
    completion_warnings: tuple[str, ...]
    reference_singular_values: tuple[float, ...]
    reference_sigma_min: float
    reference_condition_number: float


def _resolve_gamma_reference_core(
    u_low: np.ndarray,
    *,
    config: AutoGaugeConfig,
    segments: list[tuple[int, int]] | None,
) -> _GammaReferenceResolution:
    """Resolve and certify one complete Gamma SCDM reference set."""

    u = np.asarray(u_low, dtype=np.complex128)
    if u.ndim != 2 or min(u.shape) <= 0:
        raise ValueError(f"Gamma model reference eigenspace must be nonempty and 2D, got {u.shape}")
    if not np.all(np.isfinite(u)):
        raise ValueError("Gamma model reference eigenspace contains NaN or Inf")
    selection = select_anchor_rows_qrcp(
        u,
        n_anchors=u.shape[1],
        basis_is_orthonormal=config.basis_is_orthonormal,
    )
    if selection.sigma_min < config.min_sigma:
        raise ValueError(
            f"auto gauge sigma_min={selection.sigma_min:.3e} below min_sigma={config.min_sigma:.3e}"
        )
    if selection.condition_number > config.max_condition:
        raise ValueError(
            f"auto gauge condition_number={selection.condition_number:.3e} exceeds "
            f"max_condition={config.max_condition:.3e}"
        )

    raw = [_auto_reference_terms(int(row)) for row in selection.selected_rows]
    completion_details: list[dict[str, Any]] = []
    completion_warnings: list[str] = []
    if segments is None:
        references, assigned_scores = _assign_anchor_references_to_bands(u, raw)
    else:
        completed, completion_details, completion_warnings = _complete_gamma_spinful_reference_terms(
            u,
            selection.selected_rows,
            segments=segments,
        )
        references = _order_gamma_references_for_model_basis(completed, segments=segments)
        scores = _reference_overlap_scores(u, references)
        assigned_scores = [float(np.max(scores[index, :])) for index in range(scores.shape[0])]

    singular_values = _reference_overlap_singular_values(u, references)
    sigma_min = float(np.min(singular_values)) if singular_values.size else 0.0
    sigma_max = float(np.max(singular_values)) if singular_values.size else 0.0
    condition = float("inf") if sigma_min <= 0.0 else float(sigma_max / sigma_min)
    if sigma_min < config.min_sigma:
        raise ValueError(
            f"auto gauge reference sigma_min={sigma_min:.3e} below min_sigma={config.min_sigma:.3e}"
        )
    if condition > config.max_condition:
        raise ValueError(
            f"auto gauge reference condition_number={condition:.3e} exceeds "
            f"max_condition={config.max_condition:.3e}"
        )
    return _GammaReferenceResolution(
        selection=selection,
        references_by_band=tuple(
            tuple((int(row), complex(coef)) for row, coef in reference)
            for reference in references
        ),
        assigned_scores=tuple(float(value) for value in assigned_scores),
        completion_details=tuple(dict(value) for value in completion_details),
        completion_warnings=tuple(str(value) for value in completion_warnings),
        reference_singular_values=tuple(float(value) for value in singular_values.tolist()),
        reference_sigma_min=sigma_min,
        reference_condition_number=condition,
    )


def _gamma_model_config(gauge_config: Any) -> AutoGaugeConfig:
    config = gauge_config if isinstance(gauge_config, AutoGaugeConfig) else _auto_gauge_config(gauge_config)
    if (
        config.method != "auto_scdm"
        or config.anchor_scope != "per_sector"
        or config.candidate_pool != "all"
        or config.projected_anchors
    ):
        raise ValueError("Gamma model frames require the supported AutoGaugeConfig SCDM policy")
    if not config.basis_is_orthonormal:
        raise ValueError("Gamma model frames require basis_is_orthonormal=True")
    if (
        not np.isfinite(float(config.min_sigma))
        or float(config.min_sigma) <= 0.0
        or not np.isfinite(float(config.max_condition))
        or float(config.max_condition) < 1.0
    ):
        raise ValueError("Gamma model gauge quality thresholds must be finite and positive")
    if (
        not isinstance(config.reference_q_index, Integral)
        or isinstance(config.reference_q_index, (bool, np.bool_))
    ):
        raise ValueError("Gamma model reference_q_index must be an integer")
    return config


def _gamma_joint_contract(
    joint_band_indices: Sequence[int],
    group_ranks: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, int], tuple[tuple[int, int], ...]]:
    if any(
        not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
        for value in joint_band_indices
    ):
        raise ValueError("Gamma model joint band indices must be strict integers")
    bands = tuple(int(value) for value in joint_band_indices)
    if not bands or len(set(bands)) != len(bands) or any(value < 0 for value in bands):
        raise ValueError("Gamma model joint band indices must be nonempty, unique, and non-negative")
    if len(group_ranks) != 2 or any(
        not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
        for value in group_ranks
    ):
        raise ValueError("Gamma model frames require exactly two strict group ranks")
    ranks = tuple(int(value) for value in group_ranks)
    if any(value <= 0 for value in ranks) or sum(ranks) != len(bands):
        raise ValueError("Gamma model group ranks must be positive and sum to the joint rank")
    order = tuple(
        (int(group), int(orbital))
        for group, rank in enumerate(ranks)
        for orbital in range(rank)
    )
    return bands, (ranks[0], ranks[1]), order


def _gamma_common_band_indices(joint_band_indices: Sequence[int]) -> tuple[int, ...]:
    if any(
        not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
        for value in joint_band_indices
    ):
        raise ValueError("Gamma common-anchor band indices must be strict integers")
    bands = tuple(int(value) for value in joint_band_indices)
    if not bands or len(set(bands)) != len(bands) or any(value < 0 for value in bands):
        raise ValueError(
            "Gamma common-anchor band indices must be nonempty, unique, and non-negative"
        )
    return bands


def _validate_gamma_model_eigensystems(
    *,
    eigenvalues_by_q: Sequence[np.ndarray],
    eigenvectors_by_q: Sequence[np.ndarray],
    layout: GammaRowLayout,
    joint_band_indices: tuple[int, ...],
    orthonormality_tolerance: float,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    if len(eigenvalues_by_q) != layout.q_count or len(eigenvectors_by_q) != layout.q_count:
        raise ValueError(
            "Gamma model eigensystems must provide exactly one eigenvalue/vector pair per layout Q"
        )
    values_out: list[np.ndarray] = []
    vectors_out: list[np.ndarray] = []
    columns = np.asarray(joint_band_indices, dtype=np.intp)
    for q_index, (raw_values, raw_vectors) in enumerate(zip(eigenvalues_by_q, eigenvectors_by_q)):
        values = np.asarray(raw_values, dtype=np.float64)
        vectors = np.asarray(raw_vectors, dtype=np.complex128)
        if values.ndim != 1 or vectors.ndim != 2:
            raise ValueError(f"Gamma model Q {q_index} eigensystem must be a vector and a matrix")
        if vectors.shape[0] != layout.same_q_dimension or vectors.shape[1] != values.size:
            raise ValueError(
                f"Gamma model Q {q_index} eigensystem shape {values.shape}/{vectors.shape} "
                f"does not match local dimension {layout.same_q_dimension}"
            )
        max_band = int(np.max(columns))
        if max_band >= values.size:
            raise IndexError(
                f"Gamma model joint band {max_band} outside Q {q_index} eigensystem size {values.size}"
            )
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(vectors)):
            raise ValueError(f"Gamma model Q {q_index} eigensystem contains NaN or Inf")
        selected = vectors[:, columns]
        gram_residual = float(
            np.linalg.norm(selected.conj().T @ selected - np.eye(columns.size), ord="fro")
        )
        if gram_residual > orthonormality_tolerance:
            raise ValueError(
                f"Gamma model Q {q_index} selected eigenspace orthonormality residual "
                f"{gram_residual:.3e} exceeds {orthonormality_tolerance:.3e}"
            )
        values_out.append(values)
        vectors_out.append(vectors)
    return tuple(values_out), tuple(vectors_out)


def _gamma_reference_matrix(
    references: Sequence[Sequence[tuple[int, complex]]],
    *,
    row_count: int,
) -> np.ndarray:
    phi = np.zeros((int(row_count), len(references)), dtype=np.complex128)
    for column, terms in enumerate(references):
        for row, coef in terms:
            if int(row) < 0 or int(row) >= int(row_count):
                raise IndexError(f"Gamma model reference row {row} outside local dimension {row_count}")
            value = complex(coef)
            if not np.isfinite(value.real) or not np.isfinite(value.imag):
                raise ValueError("Gamma model reference coefficient contains NaN or Inf")
            phi[int(row), int(column)] += value
        norm = float(np.linalg.norm(phi[:, int(column)]))
        if norm <= 0.0:
            raise ValueError(f"Gamma model reference column {column} has zero norm")
        phi[:, int(column)] /= norm
    return phi


def _gamma_reference_payload(
    references: Sequence[Sequence[tuple[int, complex]]],
) -> list[list[list[float | int]]]:
    return [
        [[int(row), float(complex(coef).real), float(complex(coef).imag)] for row, coef in terms]
        for terms in references
    ]


def _common_anchor_source_group_orbits(
    source_group_orbits: Sequence[Sequence[int]] | None,
    *,
    source_group_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Normalize the source-group partition used only for model ownership.

    With no narrower contract, all Gamma source groups form the common model
    owner fiber.  Explicit partitions can still require separate owners.
    """

    if source_group_orbits is None:
        return (tuple(range(int(source_group_count))),)
    try:
        raw_orbits = tuple(tuple(orbit) for orbit in source_group_orbits)
    except TypeError as error:
        raise ValueError("Gamma common-anchor source-group orbits must be sequences") from error
    normalized: list[tuple[int, ...]] = []
    for orbit in raw_orbits:
        if not orbit or any(
            isinstance(group, (bool, np.bool_)) or not isinstance(group, Integral)
            for group in orbit
        ):
            raise ValueError(
                "Gamma common-anchor source-group orbits must contain strict integers"
            )
        groups = tuple(sorted(int(group) for group in orbit))
        if len(set(groups)) != len(groups):
            raise ValueError("Gamma common-anchor source-group orbit contains duplicates")
        normalized.append(groups)
    normalized.sort(key=lambda orbit: orbit[0])
    flattened = tuple(group for orbit in normalized for group in orbit)
    if flattened != tuple(range(int(source_group_count))):
        raise ValueError(
            "Gamma common-anchor source-group orbits must canonically partition all groups"
        )
    return tuple(normalized)


def build_gamma_common_anchor_spec(
    *,
    reference_eigenvalues_by_q: Sequence[np.ndarray],
    reference_eigenvectors_by_q: Sequence[np.ndarray],
    joint_band_indices: Sequence[int],
    layout: GammaRowLayout,
    gauge_config: AutoGaugeConfig | Mapping[str, Any] | str | None = None,
    projector_tolerance: float = 1.0e-10,
    orthonormality_tolerance: float = 1.0e-10,
    source_group_orbits: Sequence[Sequence[int]] | None = None,
) -> GammaCommonAnchorSpec:
    """Select one canonical orbital anchor pattern at a representative Gamma Q."""

    if not isinstance(layout, GammaRowLayout):
        raise TypeError("Gamma common-anchor construction requires a GammaRowLayout")
    config = _gamma_model_config(gauge_config)
    bands = _gamma_common_band_indices(joint_band_indices)
    tolerances = (float(projector_tolerance), float(orthonormality_tolerance))
    if any(not np.isfinite(value) or value <= 0.0 for value in tolerances):
        raise ValueError(
            "Gamma common-anchor projector/orthonormality tolerances must be finite and positive"
        )
    values_by_q, vectors_by_q = _validate_gamma_model_eigensystems(
        eigenvalues_by_q=reference_eigenvalues_by_q,
        eigenvectors_by_q=reference_eigenvectors_by_q,
        layout=layout,
        joint_band_indices=bands,
        orthonormality_tolerance=tolerances[1],
    )
    reference_q = int(config.reference_q_index)
    if reference_q < 0 or reference_q >= layout.q_count:
        raise IndexError(
            f"Gamma common-anchor reference_q_index={reference_q} outside available Q range "
            f"0..{layout.q_count - 1}"
        )
    columns = np.asarray(bands, dtype=np.intp)
    u_reference = vectors_by_q[reference_q][:, columns]
    widths = [
        [int(layout.uniform_orbital_count)] * int(layer_count)
        for layer_count in layout.num_layer_list
    ]
    segments = _gamma_same_q_row_segments(
        layout.same_q_dimension,
        list(layout.num_layer_list),
        widths,
        spin="all",
    )
    try:
        resolution = _resolve_gamma_reference_core(
            u_reference,
            config=config,
            segments=segments,
        )
    except GammaRoutingError:
        raise
    except ValueError as error:
        raise GammaRoutingError(
            CandidateRejectionReason.PROJECTOR_FRAME_RANK,
            str(error),
        ) from error
    selected_rows = tuple(int(row) for row in resolution.selection.selected_rows)
    group_orbits = _common_anchor_source_group_orbits(
        source_group_orbits,
        source_group_count=len(layout.ordered_qsets),
    )
    resolved_references: list[
        tuple[set[int], set[int], tuple[tuple[int, complex], ...]]
    ] = []
    for reference in resolution.references_by_band:
        addresses = tuple(
            layout.rows_by_q[reference_q][int(row)] for row, _coefficient in reference
        )
        physical_layers = {int(address.physical_layer) for address in addresses}
        source_groups = {int(address.source_group) for address in addresses}
        resolved_references.append((physical_layers, source_groups, reference))

    pure_owner_support = all(
        len(physical_layers) == 1 and len(source_groups) == 1
        for physical_layers, source_groups, _reference in resolved_references
    )
    if pure_owner_support:
        resolved_references.sort(
            key=lambda item: (
                next(iter(item[0])),
                next(iter(item[1])),
            )
        )
        references = tuple(item[2] for item in resolved_references)
        reference_physical_layers = [
            next(iter(item[0])) for item in resolved_references
        ]
        reference_source_groups = [
            next(iter(item[1])) for item in resolved_references
        ]
    else:
        mixed_supports = tuple(
            tuple(sorted(source_groups))
            for _physical_layers, source_groups, _reference in resolved_references
        )
        matching_orbits = tuple(
            orbit
            for orbit in group_orbits
            if len(orbit) > 1 and all(support == orbit for support in mixed_supports)
        )
        if len(matching_orbits) != 1:
            raise GammaRoutingError(
                CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                "Gamma common-anchor reference orbital spans multiple physical layers "
                "or source groups without one matching symmetry owner contract",
            )
        owner_orbit = matching_orbits[0]
        if any(int(layout.num_layer_list[group]) != 1 for group in owner_orbit):
            raise GammaRoutingError(
                CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                "Gamma common-anchor mixed reference owner contract is ambiguous for "
                "multi-layer source groups",
            )
        if len(resolved_references) % len(owner_orbit) != 0:
            raise GammaRoutingError(
                CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                "Gamma common-anchor mixed reference rank is incompatible with its "
                "source-group symmetry orbit",
            )
        references = tuple(item[2] for item in resolved_references)
        rank_per_group = len(references) // len(owner_orbit)
        reference_source_groups = [
            group
            for group in owner_orbit
            for _column in range(rank_per_group)
        ]
        physical_offsets = tuple(
            sum(int(value) for value in layout.num_layer_list[:group])
            for group in range(len(layout.num_layer_list))
        )
        reference_physical_layers = [
            physical_offsets[group] for group in reference_source_groups
        ]
    phi = _gamma_reference_matrix(references, row_count=layout.same_q_dimension)
    singular_values = np.asarray(
        resolution.reference_singular_values,
        dtype=np.float64,
    )
    sigma_min = float(resolution.reference_sigma_min)
    condition = float(resolution.reference_condition_number)

    owner_specs: list[GammaModelOwnerSpec] = []
    ordered_physical_layers = tuple(dict.fromkeys(reference_physical_layers))
    for physical_layer in ordered_physical_layers:
        reference_columns = tuple(
            index
            for index, column_layer in enumerate(reference_physical_layers)
            if int(column_layer) == physical_layer
        )
        source_groups = {
            int(reference_source_groups[index]) for index in reference_columns
        }
        if len(source_groups) != 1:
            raise GammaRoutingError(
                CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                "Gamma common-anchor physical-layer owner spans multiple source groups",
            )
        source_group = next(iter(source_groups))
        owner_specs.append(
            GammaModelOwnerSpec(
                lambda_index=len(owner_specs),
                rank=len(reference_columns),
                source_group=source_group,
                qset_index=source_group,
                physical_layers=(physical_layer,),
                reference_columns=reference_columns,
            )
        )

    selected_eigenvalues = tuple(float(values_by_q[reference_q][band]) for band in bands)
    payload = {
        "schema": GammaCommonAnchorSpec.SCHEMA,
        "joint_band_indices": list(bands),
        "resolved_reference_terms": _gamma_reference_payload(references),
        "selected_rows": list(selected_rows),
        "reference_physical_layers": list(reference_physical_layers),
        "reference_source_groups": list(reference_source_groups),
        "owner_specs": [owner.to_payload() for owner in owner_specs],
        "reference_q_index": reference_q,
        "layout_hash": layout.layout_hash,
        "same_q_dimension": layout.same_q_dimension,
        "physical_layer_count": sum(int(value) for value in layout.num_layer_list),
        "source_qset_count": len(layout.ordered_qsets),
        "min_sigma": float(config.min_sigma),
        "max_condition": float(config.max_condition),
        "projector_tolerance": tolerances[0],
        "orthonormality_tolerance": tolerances[1],
        "selection_sigma_min": float(resolution.selection.sigma_min),
        "selection_condition_number": float(resolution.selection.condition_number),
        "reference_singular_values": [float(value) for value in singular_values.tolist()],
        "reference_sigma_min": sigma_min,
        "reference_condition_number": condition,
        "reference_eigenvalues": list(selected_eigenvalues),
        "reference_projector_hash": hash_array(u_reference @ u_reference.conj().T),
        "warnings": list(resolution.completion_warnings)
        + list(resolution.selection.warnings),
    }
    return GammaCommonAnchorSpec(
        joint_band_indices=bands,
        resolved_reference_terms=references,
        selected_rows=selected_rows,
        reference_physical_layers=tuple(reference_physical_layers),
        reference_source_groups=tuple(reference_source_groups),
        owner_specs=tuple(owner_specs),
        reference_q_index=reference_q,
        layout_hash=layout.layout_hash,
        same_q_dimension=layout.same_q_dimension,
        physical_layer_count=sum(int(value) for value in layout.num_layer_list),
        source_qset_count=len(layout.ordered_qsets),
        min_sigma=float(config.min_sigma),
        max_condition=float(config.max_condition),
        projector_tolerance=tolerances[0],
        orthonormality_tolerance=tolerances[1],
        selection_sigma_min=float(resolution.selection.sigma_min),
        selection_condition_number=float(resolution.selection.condition_number),
        reference_singular_values=tuple(float(value) for value in singular_values.tolist()),
        reference_sigma_min=sigma_min,
        reference_condition_number=condition,
        reference_eigenvalues=selected_eigenvalues,
        reference_projector_hash=payload["reference_projector_hash"],
        warnings=tuple(payload["warnings"]),
        identity_hash=hash_mapping(payload),
    )


def build_gamma_common_anchor_frames(
    *,
    eigenvalues_by_q: Sequence[np.ndarray],
    eigenvectors_by_q: Sequence[np.ndarray],
    layout: GammaRowLayout,
    anchor_spec: GammaCommonAnchorSpec,
) -> GammaCommonAnchorFrames:
    """Align each local selected eigenspace to the shared common-anchor basis."""

    if not isinstance(anchor_spec, GammaCommonAnchorSpec):
        raise TypeError("Gamma common-anchor frames require a GammaCommonAnchorSpec")
    anchor_spec.to_payload()
    if (
        not isinstance(layout, GammaRowLayout)
        or layout.layout_hash != anchor_spec.layout_hash
        or layout.same_q_dimension != anchor_spec.same_q_dimension
    ):
        raise ValueError("Gamma common-anchor frame layout/spec identity mismatch")
    bands = _gamma_common_band_indices(anchor_spec.joint_band_indices)
    _, vectors_by_q = _validate_gamma_model_eigensystems(
        eigenvalues_by_q=eigenvalues_by_q,
        eigenvectors_by_q=eigenvectors_by_q,
        layout=layout,
        joint_band_indices=bands,
        orthonormality_tolerance=anchor_spec.orthonormality_tolerance,
    )
    phi = _gamma_reference_matrix(
        anchor_spec.resolved_reference_terms,
        row_count=layout.same_q_dimension,
    )
    columns = np.asarray(bands, dtype=np.intp)
    frames: list[np.ndarray] = []
    alignments: list[np.ndarray] = []
    singular_values_by_q: list[tuple[float, ...]] = []
    orthonormality_residuals: list[float] = []
    projector_residuals: list[float] = []
    for q_index, vectors in enumerate(vectors_by_q):
        u_joint = vectors[:, columns]
        singular_values = np.linalg.svd(phi.conj().T @ u_joint, compute_uv=False)
        sigma_min = float(np.min(singular_values)) if singular_values.size else 0.0
        sigma_max = float(np.max(singular_values)) if singular_values.size else 0.0
        condition = float("inf") if sigma_min <= 0.0 else float(sigma_max / sigma_min)
        if sigma_min < anchor_spec.min_sigma or condition > anchor_spec.max_condition:
            raise GammaRoutingError(
                CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                f"Gamma common-anchor Q {q_index} coverage failed: "
                f"sigma_min={sigma_min:.3e}, condition_number={condition:.3e}",
            )
        frame, alignment = align_eigenstates(u_joint, phi)
        orthonormality_residual = float(
            max(
                np.linalg.norm(frame.conj().T @ frame - np.eye(len(bands)), ord="fro"),
                np.linalg.norm(alignment.conj().T @ alignment - np.eye(len(bands)), ord="fro"),
            )
        )
        projector_residual = float(
            np.linalg.norm(frame @ frame.conj().T - u_joint @ u_joint.conj().T, ord="fro")
        )
        if orthonormality_residual > anchor_spec.orthonormality_tolerance:
            raise GammaRoutingError(
                CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                f"Gamma common-anchor Q {q_index} orthonormality residual exceeds "
                f"{anchor_spec.orthonormality_tolerance:.3e}",
            )
        if projector_residual > anchor_spec.projector_tolerance:
            raise GammaRoutingError(
                CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                f"Gamma common-anchor Q {q_index} projector residual exceeds "
                f"{anchor_spec.projector_tolerance:.3e}",
            )
        frames.append(np.asarray(frame, dtype=np.complex128))
        alignments.append(np.asarray(alignment, dtype=np.complex128))
        singular_values_by_q.append(tuple(float(value) for value in singular_values.tolist()))
        orthonormality_residuals.append(orthonormality_residual)
        projector_residuals.append(projector_residual)
    frame_hashes = tuple(hash_array(value) for value in frames)
    alignment_hashes = tuple(hash_array(value) for value in alignments)
    payload = {
        "schema": GammaCommonAnchorFrames.SCHEMA,
        "anchor_spec_identity_hash": anchor_spec.identity_hash,
        "layout_hash": layout.layout_hash,
        "joint_band_indices": list(bands),
        "model_group_ranks": list(anchor_spec.model_group_ranks),
        "frame_hashes_by_q": list(frame_hashes),
        "alignment_hashes_by_q": list(alignment_hashes),
        "alignment_singular_values_by_q": [list(values) for values in singular_values_by_q],
        "orthonormality_residuals_by_q": orthonormality_residuals,
        "projector_residuals_by_q": projector_residuals,
    }
    return GammaCommonAnchorFrames(
        anchor_spec_identity_hash=anchor_spec.identity_hash,
        layout_hash=layout.layout_hash,
        joint_band_indices=bands,
        model_group_ranks=anchor_spec.model_group_ranks,
        local_frames_by_q=tuple(frames),
        alignment_unitaries_by_q=tuple(alignments),
        alignment_singular_values_by_q=tuple(singular_values_by_q),
        orthonormality_residuals_by_q=tuple(orthonormality_residuals),
        projector_residuals_by_q=tuple(projector_residuals),
        frame_hashes_by_q=frame_hashes,
        alignment_hashes_by_q=alignment_hashes,
        identity_hash=hash_mapping(payload),
    )


def build_gamma_model_anchor_spec(
    *,
    reference_eigenvalues_by_q: Sequence[np.ndarray],
    reference_eigenvectors_by_q: Sequence[np.ndarray],
    joint_band_indices: Sequence[int],
    group_ranks: Sequence[int],
    layout: GammaRowLayout,
    gauge_config: AutoGaugeConfig | Mapping[str, Any] | str | None = None,
    projector_tolerance: float = 1.0e-10,
    orthonormality_tolerance: float = 1.0e-10,
) -> GammaModelAnchorSpec:
    """Resolve one deterministic full-rank SCDM model gauge from eigensystems."""

    if not isinstance(layout, GammaRowLayout):
        raise TypeError("Gamma model anchor construction requires a GammaRowLayout")
    config = _gamma_model_config(gauge_config)
    bands, ranks, model_order = _gamma_joint_contract(joint_band_indices, group_ranks)
    tolerances = (float(projector_tolerance), float(orthonormality_tolerance))
    if any(not np.isfinite(value) or value <= 0.0 for value in tolerances):
        raise ValueError("Gamma model projector/orthonormality tolerances must be finite and positive")
    values_by_q, vectors_by_q = _validate_gamma_model_eigensystems(
        eigenvalues_by_q=reference_eigenvalues_by_q,
        eigenvectors_by_q=reference_eigenvectors_by_q,
        layout=layout,
        joint_band_indices=bands,
        orthonormality_tolerance=tolerances[1],
    )
    reference_q = int(config.reference_q_index)
    if reference_q < 0 or reference_q >= layout.q_count:
        raise IndexError(
            f"Gamma model reference_q_index={reference_q} outside available Q range 0..{layout.q_count - 1}"
        )
    u_reference = vectors_by_q[reference_q][:, np.asarray(bands, dtype=np.intp)]
    widths = [
        [int(layout.uniform_orbital_count)] * int(layer_count)
        for layer_count in layout.num_layer_list
    ]
    segments = _gamma_same_q_row_segments(
        layout.same_q_dimension,
        list(layout.num_layer_list),
        widths,
        spin="all",
    )
    resolution = _resolve_gamma_reference_core(u_reference, config=config, segments=segments)
    references = resolution.references_by_band
    phi = _gamma_reference_matrix(references, row_count=layout.same_q_dimension)
    if np.linalg.matrix_rank(phi.conj().T @ u_reference) != len(bands):
        raise ValueError("Gamma model fixed anchors are rank deficient in the reference joint space")
    selected_eigenvalues = tuple(float(values_by_q[reference_q][band]) for band in bands)
    projector_hash = hash_array(u_reference @ u_reference.conj().T)
    payload = {
        "schema": GammaModelAnchorSpec.SCHEMA,
        "joint_band_indices": list(bands),
        "group_ranks": list(ranks),
        "model_column_order": [list(value) for value in model_order],
        "resolved_reference_terms": _gamma_reference_payload(references),
        "selected_rows": list(resolution.selection.selected_rows),
        "reference_q_index": reference_q,
        "layout_hash": layout.layout_hash,
        "same_q_dimension": layout.same_q_dimension,
        "min_sigma": float(config.min_sigma),
        "max_condition": float(config.max_condition),
        "projector_tolerance": tolerances[0],
        "orthonormality_tolerance": tolerances[1],
        "selection_sigma_min": float(resolution.selection.sigma_min),
        "selection_condition_number": float(resolution.selection.condition_number),
        "reference_singular_values": list(resolution.reference_singular_values),
        "reference_sigma_min": resolution.reference_sigma_min,
        "reference_condition_number": resolution.reference_condition_number,
        "reference_eigenvalues": list(selected_eigenvalues),
        "reference_projector_hash": projector_hash,
        "warnings": list(resolution.completion_warnings) + list(resolution.selection.warnings),
    }
    return GammaModelAnchorSpec(
        joint_band_indices=bands,
        group_ranks=ranks,
        model_column_order=model_order,
        resolved_reference_terms=references,
        selected_rows=tuple(int(value) for value in resolution.selection.selected_rows),
        reference_q_index=reference_q,
        layout_hash=layout.layout_hash,
        same_q_dimension=layout.same_q_dimension,
        min_sigma=float(config.min_sigma),
        max_condition=float(config.max_condition),
        projector_tolerance=tolerances[0],
        orthonormality_tolerance=tolerances[1],
        selection_sigma_min=float(resolution.selection.sigma_min),
        selection_condition_number=float(resolution.selection.condition_number),
        reference_singular_values=resolution.reference_singular_values,
        reference_sigma_min=resolution.reference_sigma_min,
        reference_condition_number=resolution.reference_condition_number,
        reference_eigenvalues=selected_eigenvalues,
        reference_projector_hash=projector_hash,
        warnings=tuple(payload["warnings"]),
        identity_hash=hash_mapping(payload),
    )


def validate_gamma_model_anchor_reference(
    *,
    reference_eigenvalues_by_q: Sequence[np.ndarray],
    reference_eigenvectors_by_q: Sequence[np.ndarray],
    layout: GammaRowLayout,
    anchor_spec: GammaModelAnchorSpec,
) -> None:
    """Validate the eigensystem used to create an existing anchor contract."""

    if not isinstance(anchor_spec, GammaModelAnchorSpec):
        raise TypeError("Gamma model reference validation requires a GammaModelAnchorSpec")
    anchor_spec.to_payload()
    if (
        not isinstance(layout, GammaRowLayout)
        or layout.layout_hash != anchor_spec.layout_hash
        or layout.same_q_dimension != anchor_spec.same_q_dimension
    ):
        raise ValueError("Gamma model reference layout/spec identity mismatch")
    bands, ranks, order = _gamma_joint_contract(
        anchor_spec.joint_band_indices,
        anchor_spec.group_ranks,
    )
    if ranks != anchor_spec.group_ranks or order != anchor_spec.model_column_order:
        raise ValueError("Gamma model reference joint-band/model-column contract mismatch")
    values_by_q, vectors_by_q = _validate_gamma_model_eigensystems(
        eigenvalues_by_q=reference_eigenvalues_by_q,
        eigenvectors_by_q=reference_eigenvectors_by_q,
        layout=layout,
        joint_band_indices=bands,
        orthonormality_tolerance=anchor_spec.orthonormality_tolerance,
    )
    reference_q = anchor_spec.reference_q_index
    if reference_q < 0 or reference_q >= layout.q_count:
        raise IndexError(
            f"Gamma model reference_q_index={reference_q} outside available Q range 0..{layout.q_count - 1}"
        )
    columns = np.asarray(bands, dtype=np.intp)
    actual_energies = values_by_q[reference_q][columns]
    expected_energies = np.asarray(anchor_spec.reference_eigenvalues, dtype=np.float64)
    if not np.array_equal(actual_energies, expected_energies):
        raise ValueError("Gamma model anchor reference eigenvalues mismatch")
    u_reference = vectors_by_q[reference_q][:, columns]
    actual_projector_hash = hash_array(u_reference @ u_reference.conj().T)
    if actual_projector_hash != anchor_spec.reference_projector_hash:
        raise ValueError("Gamma model anchor reference projector mismatch")


def build_gamma_model_frames(
    *,
    eigenvalues_by_q: Sequence[np.ndarray],
    eigenvectors_by_q: Sequence[np.ndarray],
    layout: GammaRowLayout,
    anchor_spec: GammaModelAnchorSpec,
) -> GammaModelFrames:
    """Materialize compact model frames with one full-U(N) alignment per Q."""

    if not isinstance(anchor_spec, GammaModelAnchorSpec):
        raise TypeError("Gamma model frame construction requires a GammaModelAnchorSpec")
    anchor_spec.to_payload()
    if not isinstance(layout, GammaRowLayout) or layout.layout_hash != anchor_spec.layout_hash:
        raise ValueError("Gamma model frame layout/spec identity mismatch")
    bands, ranks, order = _gamma_joint_contract(anchor_spec.joint_band_indices, anchor_spec.group_ranks)
    if order != anchor_spec.model_column_order:
        raise ValueError("Gamma model anchor model-column order is inconsistent with group ranks")
    _, vectors_by_q = _validate_gamma_model_eigensystems(
        eigenvalues_by_q=eigenvalues_by_q,
        eigenvectors_by_q=eigenvectors_by_q,
        layout=layout,
        joint_band_indices=bands,
        orthonormality_tolerance=anchor_spec.orthonormality_tolerance,
    )
    phi = _gamma_reference_matrix(
        anchor_spec.resolved_reference_terms,
        row_count=layout.same_q_dimension,
    )
    columns = np.asarray(bands, dtype=np.intp)
    frames: list[np.ndarray] = []
    alignments: list[np.ndarray] = []
    singular_values_by_q: list[tuple[float, ...]] = []
    orthonormality_residuals: list[float] = []
    projector_residuals: list[float] = []
    for q_index, vectors in enumerate(vectors_by_q):
        u_joint = vectors[:, columns]
        singular_values = np.linalg.svd(phi.conj().T @ u_joint, compute_uv=False)
        sigma_min = float(np.min(singular_values)) if singular_values.size else 0.0
        sigma_max = float(np.max(singular_values)) if singular_values.size else 0.0
        condition = float("inf") if sigma_min <= 0.0 else float(sigma_max / sigma_min)
        if sigma_min < anchor_spec.min_sigma or condition > anchor_spec.max_condition:
            raise ValueError(
                f"Gamma model Q {q_index} fixed-anchor quality failed: "
                f"sigma_min={sigma_min:.3e}, condition_number={condition:.3e}"
            )
        model_frame, alignment = align_eigenstates(u_joint, phi)
        orthonormality_residual = float(
            np.linalg.norm(model_frame.conj().T @ model_frame - np.eye(len(bands)), ord="fro")
        )
        alignment_residual = float(
            np.linalg.norm(alignment.conj().T @ alignment - np.eye(len(bands)), ord="fro")
        )
        projector_residual = float(
            np.linalg.norm(
                model_frame @ model_frame.conj().T - u_joint @ u_joint.conj().T,
                ord="fro",
            )
        )
        if max(orthonormality_residual, alignment_residual) > anchor_spec.orthonormality_tolerance:
            raise ValueError(
                f"Gamma model Q {q_index} frame/alignment orthonormality residual exceeds "
                f"{anchor_spec.orthonormality_tolerance:.3e}"
            )
        if projector_residual > anchor_spec.projector_tolerance:
            raise ValueError(
                f"Gamma model Q {q_index} projector residual {projector_residual:.3e} exceeds "
                f"{anchor_spec.projector_tolerance:.3e}"
            )
        frames.append(np.asarray(model_frame, dtype=np.complex128))
        alignments.append(np.asarray(alignment, dtype=np.complex128))
        singular_values_by_q.append(tuple(float(value) for value in singular_values.tolist()))
        orthonormality_residuals.append(orthonormality_residual)
        projector_residuals.append(projector_residual)

    frame_hashes = tuple(hash_array(value) for value in frames)
    alignment_hashes = tuple(hash_array(value) for value in alignments)
    payload = {
        "schema": GammaModelFrames.SCHEMA,
        "anchor_spec_identity_hash": anchor_spec.identity_hash,
        "layout_hash": layout.layout_hash,
        "joint_band_indices": list(bands),
        "group_ranks": list(ranks),
        "frame_hashes_by_q": list(frame_hashes),
        "alignment_hashes_by_q": list(alignment_hashes),
        "alignment_singular_values_by_q": [list(value) for value in singular_values_by_q],
        "orthonormality_residuals_by_q": orthonormality_residuals,
        "projector_residuals_by_q": projector_residuals,
    }
    return GammaModelFrames(
        anchor_spec_identity_hash=anchor_spec.identity_hash,
        layout_hash=layout.layout_hash,
        joint_band_indices=bands,
        group_ranks=ranks,
        local_frames_by_q=tuple(frames),
        alignment_unitaries_by_q=tuple(alignments),
        alignment_singular_values_by_q=tuple(singular_values_by_q),
        orthonormality_residuals_by_q=tuple(orthonormality_residuals),
        projector_residuals_by_q=tuple(projector_residuals),
        frame_hashes_by_q=frame_hashes,
        alignment_hashes_by_q=alignment_hashes,
        identity_hash=hash_mapping(payload),
    )


def _selection_dict(
    *,
    scope: str,
    bands: list[int],
    selection,
    layer: int | None = None,
    q_index: int | None = None,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "layer": None if layer is None else int(layer),
        "q_index": None if q_index is None else int(q_index),
        "bands": [int(band) for band in bands],
        "selected_rows": [int(row) for row in selection.selected_rows],
        "method": selection.method,
        "rank": int(selection.rank),
        "singular_values": [float(value) for value in selection.singular_values],
        "sigma_min": float(selection.sigma_min),
        "condition_number": float(selection.condition_number),
        "selected_leverage_scores": [float(value) for value in selection.leverage_scores],
        "top_leverage_rows": [
            {
                "row": int(candidate.row),
                "leverage": float(candidate.leverage),
                "selected": bool(candidate.selected),
                "label": candidate.label,
            }
            for candidate in selection.candidates
        ],
        "warnings": list(selection.warnings),
    }


def _auto_gauge_report(
    *,
    resolved_norb_fix_list: list[Any],
    selections: list[dict[str, Any]],
    warnings: list[str],
    config: AutoGaugeConfig,
) -> GaugeAnchorReport:
    sigma_values = [
        float(row.get("reference_sigma_min", row["sigma_min"]))
        for row in selections
        if row.get("reference_sigma_min", row.get("sigma_min")) is not None
    ]
    cond_values = [
        float(row.get("reference_condition_number", row["condition_number"]))
        for row in selections
        if row.get("reference_condition_number", row.get("condition_number")) is not None
    ]
    sigma_min = min(sigma_values) if sigma_values else None
    condition_number = max(cond_values) if cond_values else None
    return GaugeAnchorReport(
        gauge_mode="auto_scdm",
        resolved_norb_fix_list=resolved_norb_fix_list,
        selections=selections,
        metric={
            "type": "orthonormal" if config.basis_is_orthonormal else "overlap_metric",
            "basis_is_orthonormal": bool(config.basis_is_orthonormal),
            "formula": "diag(U U^dagger)" if config.basis_is_orthonormal else "diag(S^1/2 U U^dagger S^1/2)",
        },
        state_selection_quality={
            "status": "not_evaluated",
            "reason": "auto gauge fixes anchors for the configured nlow_state_list only",
        },
        gauge_anchor_quality={
            "status": "ok",
            "sigma_min": sigma_min,
            "condition_number": condition_number,
            "min_sigma": float(config.min_sigma),
            "max_condition": float(config.max_condition),
        },
        symmetry_closure_quality={
            "status": "not_available",
            "subspace_leakage": None,
            "reason": "symmetry validation is not available during gauge resolution",
        },
        warnings=warnings,
    )


def _resolved_from_references(
    owners: list[tuple[int, int]],
    references_by_band: list[list[tuple[int, complex]]],
    *,
    total_layers: int,
) -> list[Any]:
    resolved: list[Any] = [[] for _ in range(total_layers)]
    for (layer, _band_pos), terms in zip(owners, references_by_band):
        resolved[int(layer)].append(_format_auto_reference_terms(terms))
    return resolved


def _parse_formatted_complex(value: Any) -> complex:
    if isinstance(value, Mapping):
        return complex(float(value.get("real", 0.0)), float(value.get("imag", 0.0)))
    if isinstance(value, str):
        if value == "1j":
            return 1.0j
        if value == "-1j":
            return -1.0j
        return complex(value)
    return complex(value)


def _parse_formatted_reference_terms(value: Any) -> list[tuple[int, complex]]:
    if value is None or value == []:
        return []
    if isinstance(value, (int, np.integer)):
        return [(int(value), 1.0 + 0.0j)]
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError(f"cannot parse formatted reference terms from {value!r}")
    if value and not isinstance(value[0], (list, tuple)):
        row = int(value[0])
        coef = _parse_formatted_complex(value[1]) if len(value) > 1 else 1.0 + 0.0j
        return [(row, coef)]
    terms: list[tuple[int, complex]] = []
    for term in value:
        if not isinstance(term, (list, tuple)) or not term:
            raise ValueError(f"cannot parse formatted reference term {term!r}")
        row = int(term[0])
        coef = _parse_formatted_complex(term[1]) if len(term) > 1 else 1.0 + 0.0j
        terms.append((row, coef))
    return terms


def _references_by_layer_from_resolved(resolved: list[Any], *, total_layers: int) -> list[list[list[tuple[int, complex]]]]:
    out: list[list[list[tuple[int, complex]]]] = [[] for _ in range(total_layers)]
    for layer in range(total_layers):
        if layer >= len(resolved) or resolved[layer] in (None, []):
            continue
        if not isinstance(resolved[layer], list):
            raise ValueError(f"resolved anchors for layer {layer} must be a list")
        out[layer] = [_parse_formatted_reference_terms(anchor) for anchor in resolved[layer]]
    return out


def _source_group_layer_ranges(num_layer_list: list[int]) -> list[tuple[int, int, int]]:
    ranges: list[tuple[int, int, int]] = []
    cursor = 0
    for group_index, n_layers in enumerate(num_layer_list):
        n_layers = int(n_layers)
        ranges.append((int(group_index), cursor, cursor + n_layers))
        cursor += n_layers
    return ranges


def _translate_gamma_reference_terms_between_layers(
    terms: list[tuple[int, complex]],
    *,
    source_layer: int,
    target_layer: int,
    segments: list[tuple[int, int]],
) -> list[tuple[int, complex]]:
    if not terms:
        return []
    if len(segments) < 2 or len(segments) % 2 != 0:
        raise ValueError("gamma source-group translation requires spin-resolved layer segments")
    layer_count = len(segments) // 2
    if source_layer < 0 or source_layer >= layer_count or target_layer < 0 or target_layer >= layer_count:
        raise ValueError("gamma source-group translation layer index is outside segment layout")
    translated: list[tuple[int, complex]] = []
    for row, coef in terms:
        row = int(row)
        segment_index = _segment_index_for_row(row, segments)
        if segment_index is None:
            raise ValueError(f"gamma source-group translation row {row} is outside segment layout")
        spin_index = int(segment_index) // layer_count
        layer_index = int(segment_index) % layer_count
        if layer_index != int(source_layer):
            raise ValueError(
                f"gamma source-group translation expected row {row} on layer {source_layer}, "
                f"got layer {layer_index}"
            )
        start, stop = segments[int(segment_index)]
        local_offset = row - int(start)
        target_segment = spin_index * layer_count + int(target_layer)
        target_start, target_stop = segments[target_segment]
        if local_offset < 0 or int(target_start) + local_offset >= int(target_stop):
            raise ValueError(
                f"gamma source-group translation local row offset {local_offset} from layer {source_layer} "
                f"does not fit target layer {target_layer}"
            )
        translated.append((int(target_start) + int(local_offset), complex(coef)))
    return translated


def _gamma_source_group_translated_references(
    resolved: list[Any],
    owners: list[tuple[int, int]],
    *,
    num_layer_list: list[int],
    segments: list[tuple[int, int]],
) -> tuple[list[list[tuple[int, complex]]], list[dict[str, Any]]]:
    total_layers = int(sum(int(n) for n in num_layer_list))
    by_layer = _references_by_layer_from_resolved(resolved, total_layers=total_layers)
    changed = False
    details: list[dict[str, Any]] = []
    for group_index, start, stop in _source_group_layer_ranges(num_layer_list):
        active_layers = [layer for layer in range(start, stop) if by_layer[layer]]
        if len(active_layers) < 2:
            continue
        template_layer = int(active_layers[0])
        template_refs = by_layer[template_layer]
        for target_layer in active_layers[1:]:
            if len(by_layer[target_layer]) != len(template_refs):
                raise ValueError(
                    f"gamma source-group translation requires equal anchor counts inside source group {group_index}"
                )
            translated_refs: list[list[tuple[int, complex]]] = []
            for band_pos, terms in enumerate(template_refs):
                translated = _translate_gamma_reference_terms_between_layers(
                    terms,
                    source_layer=template_layer,
                    target_layer=int(target_layer),
                    segments=segments,
                )
                translated_refs.append(translated)
                details.append(
                    {
                        "source_group": int(group_index),
                        "template_layer": int(template_layer),
                        "target_layer": int(target_layer),
                        "band_position": int(band_pos),
                        "source_terms": _format_auto_reference_terms(terms),
                        "translated_terms": _format_auto_reference_terms(translated),
                    }
                )
            by_layer[int(target_layer)] = translated_refs
            changed = True
    if not changed:
        raise ValueError("gamma source-group translation did not change any anchors")
    references_by_band: list[list[tuple[int, complex]]] = []
    for layer, band_pos in owners:
        layer = int(layer)
        band_pos = int(band_pos)
        if layer >= len(by_layer) or band_pos >= len(by_layer[layer]):
            raise ValueError("gamma source-group translation does not cover all low-state owners")
        references_by_band.append(list(by_layer[layer][band_pos]))
    return references_by_band, details


def _gamma_candidate_from_references(
    *,
    candidate_id: str,
    u_low: np.ndarray,
    owners: list[tuple[int, int]],
    bands_flat: list[int],
    selection,
    references_by_band: list[list[tuple[int, complex]]],
    assigned_scores: list[float],
    total_layers: int,
    config: AutoGaugeConfig,
    ref_q: int,
    reference_ordering: str,
    reference_score_mode: str,
    anchor_completion: list[dict[str, Any]] | None = None,
    warnings: list[str] | None = None,
    priority: int = 0,
) -> ProjectGaugeAnchorCandidate:
    reference_singular_values = _reference_overlap_singular_values(u_low, references_by_band)
    reference_sigma_min = float(np.min(reference_singular_values)) if reference_singular_values.size else 0.0
    reference_sigma_max = float(np.max(reference_singular_values)) if reference_singular_values.size else 0.0
    reference_condition = float("inf") if reference_sigma_min <= 0.0 else float(reference_sigma_max / reference_sigma_min)
    if reference_sigma_min < config.min_sigma:
        raise ValueError(
            f"auto gauge candidate {candidate_id!r} reference sigma_min={reference_sigma_min:.3e} "
            f"below min_sigma={config.min_sigma:.3e}"
        )
    if reference_condition > config.max_condition:
        raise ValueError(
            f"auto gauge candidate {candidate_id!r} reference condition_number={reference_condition:.3e} "
            f"exceeds max_condition={config.max_condition:.3e}"
        )
    resolved = _resolved_from_references(owners, references_by_band, total_layers=total_layers)
    selection_row = _selection_dict(
        scope="gamma_same_q",
        bands=bands_flat,
        selection=selection,
        q_index=ref_q,
    )
    selection_row["candidate_id"] = candidate_id
    selection_row["resolved_references_by_band"] = [
        _format_auto_reference_terms(terms) for terms in references_by_band
    ]
    selection_row["assigned_reference_scores"] = [float(score) for score in assigned_scores]
    selection_row["reference_ordering"] = reference_ordering
    selection_row["reference_score_mode"] = reference_score_mode
    selection_row["reference_singular_values"] = [float(value) for value in reference_singular_values.tolist()]
    selection_row["reference_sigma_min"] = float(reference_sigma_min)
    selection_row["reference_condition_number"] = float(reference_condition)
    selection_row["anchor_completion"] = [] if anchor_completion is None else list(anchor_completion)
    report = _auto_gauge_report(
        resolved_norb_fix_list=resolved,
        selections=[selection_row],
        warnings=[] if warnings is None else list(warnings),
        config=config,
    )
    return ProjectGaugeAnchorCandidate(
        candidate_id=candidate_id,
        resolved_norb_fix_list=resolved,
        report=report,
        priority=int(priority),
    )


def _non_gamma_candidate_from_kind(
    *,
    candidate_id: str,
    kind: str,
    h_vec_blk: list[np.ndarray],
    q_count: int,
    nlow_state_list: list[list[int]],
    layer_widths: list[int],
    total_layers: int,
    spin: Literal["up", "down", "all"],
    config: AutoGaugeConfig,
    ref_q: int,
    priority: int,
) -> ProjectGaugeAnchorCandidate | None:
    resolved: list[Any] = [[] for _ in range(total_layers)]
    selections: list[dict[str, Any]] = []
    warnings: list[str] = []
    any_layer = False
    template_references: list[list[tuple[int, complex]]] | None = None
    for layer, bands in enumerate(nlow_state_list):
        layer_bands = [int(band) for band in bands]
        if not layer_bands:
            continue
        any_layer = True
        block_index = int(layer) * int(q_count) + int(ref_q)
        vec = np.asarray(h_vec_blk[block_index], dtype=np.complex128)
        u_low = vec[:, np.asarray(layer_bands, dtype=np.intp)]
        selection = select_anchor_rows_qrcp(
            u_low,
            n_anchors=len(layer_bands),
            basis_is_orthonormal=config.basis_is_orthonormal,
        )
        if selection.sigma_min < config.min_sigma:
            raise ValueError(
                f"auto gauge candidate {candidate_id!r} layer {layer} sigma_min={selection.sigma_min:.3e} "
                f"below min_sigma={config.min_sigma:.3e}"
            )
        if selection.condition_number > config.max_condition:
            raise ValueError(
                f"auto gauge candidate {candidate_id!r} layer {layer} condition_number="
                f"{selection.condition_number:.3e} exceeds max_condition={config.max_condition:.3e}"
            )
        segments = _local_layer_row_segments(
            u_low.shape[0],
            layer_widths[int(layer)] if int(layer) < len(layer_widths) else u_low.shape[0],
            spin=spin,
        )
        completion_details: list[dict[str, Any]] = []
        if kind == "raw":
            references = [_auto_reference_terms(int(row)) for row in selection.selected_rows]
            reference_score_mode = "assigned_overlap"
        elif kind == "best_overlap":
            references, completion_details, completion_warnings = _complete_gamma_spinful_reference_terms(
                u_low,
                selection.selected_rows,
                segments=segments,
            )
            warnings.extend(completion_warnings)
            reference_score_mode = "best_local_pair_overlap"
        elif kind == "spin_chiral":
            references, completion_details, completion_warnings = _complete_adjacent_chiral_reference_terms(
                u_low,
                selection.selected_rows,
                segments=segments,
                conjugate=False,
                primary="selected",
            )
            warnings.extend(completion_warnings)
            reference_score_mode = "spin_chiral_adjacent_pair"
        elif kind == "spin_chiral_conjugate":
            references, completion_details, completion_warnings = _complete_adjacent_chiral_reference_terms(
                u_low,
                selection.selected_rows,
                segments=segments,
                conjugate=True,
                primary="selected",
            )
            warnings.extend(completion_warnings)
            reference_score_mode = "spin_chiral_adjacent_pair_conjugate"
        elif kind == "spin_chiral_high_row":
            references, completion_details, completion_warnings = _complete_adjacent_chiral_reference_terms(
                u_low,
                selection.selected_rows,
                segments=segments,
                conjugate=False,
                primary="higher",
            )
            warnings.extend(completion_warnings)
            reference_score_mode = "spin_chiral_adjacent_pair_high_row_primary"
        elif kind == "spin_chiral_high_row_model_frame":
            references, completion_details, completion_warnings = _complete_adjacent_chiral_reference_terms(
                u_low,
                selection.selected_rows,
                segments=segments,
                conjugate=False,
                primary="higher",
            )
            references = _order_gamma_references_for_model_basis(references, segments=segments)
            warnings.extend(completion_warnings)
            reference_score_mode = "spin_chiral_adjacent_pair_high_row_model_frame"
        elif kind == "spin_chiral_high_row_lower_template":
            if template_references is None:
                references, completion_details, completion_warnings = _complete_adjacent_chiral_reference_terms(
                    u_low,
                    selection.selected_rows,
                    segments=segments,
                    conjugate=False,
                    primary="higher",
                    partner_policy="lower",
                )
                if spin == "all":
                    references = _order_gamma_references_for_model_basis(references, segments=segments)
                template_references = [list(ref) for ref in references]
                warnings.extend(completion_warnings)
            else:
                references = [list(ref) for ref in template_references]
                completion_details = [
                    {
                        "template_source": "first_nonempty_layer",
                        "template_reference_index": int(index),
                    }
                    for index, _ref in enumerate(references)
                ]
            reference_score_mode = "spin_chiral_high_row_lower_template"
        elif kind == "spin_chiral_high_row_conjugate":
            references, completion_details, completion_warnings = _complete_adjacent_chiral_reference_terms(
                u_low,
                selection.selected_rows,
                segments=segments,
                conjugate=True,
                primary="higher",
            )
            warnings.extend(completion_warnings)
            reference_score_mode = "spin_chiral_adjacent_pair_high_row_primary_conjugate"
        elif kind == "spin_chiral_high_row_conjugate_model_frame":
            references, completion_details, completion_warnings = _complete_adjacent_chiral_reference_terms(
                u_low,
                selection.selected_rows,
                segments=segments,
                conjugate=True,
                primary="higher",
            )
            references = _order_gamma_references_for_model_basis(references, segments=segments)
            warnings.extend(completion_warnings)
            reference_score_mode = "spin_chiral_adjacent_pair_high_row_conjugate_model_frame"
        else:
            raise ValueError(f"unknown non-gamma auto gauge candidate kind {kind!r}")

        if kind.endswith("_model_frame") and spin == "all":
            references_by_band = [list(ref) for ref in references]
            scores = _reference_overlap_scores(u_low, references_by_band)
            assigned_scores = [float(np.max(scores[index, :])) for index in range(scores.shape[0])]
            reference_ordering = "local_model_frame"
        else:
            references_by_band, assigned_scores = _assign_anchor_references_to_bands(u_low, references)
            reference_ordering = "overlap_assignment"
        reference_singular_values = _reference_overlap_singular_values(u_low, references_by_band)
        reference_sigma_min = float(np.min(reference_singular_values)) if reference_singular_values.size else 0.0
        reference_sigma_max = float(np.max(reference_singular_values)) if reference_singular_values.size else 0.0
        reference_condition = float("inf") if reference_sigma_min <= 0.0 else float(reference_sigma_max / reference_sigma_min)
        if reference_sigma_min < config.min_sigma:
            raise ValueError(
                f"auto gauge candidate {candidate_id!r} layer {layer} reference sigma_min={reference_sigma_min:.3e} "
                f"below min_sigma={config.min_sigma:.3e}"
            )
        if reference_condition > config.max_condition:
            raise ValueError(
                f"auto gauge candidate {candidate_id!r} layer {layer} reference condition_number="
                f"{reference_condition:.3e} exceeds max_condition={config.max_condition:.3e}"
            )
        resolved[int(layer)] = [_format_auto_reference_terms(terms) for terms in references_by_band]
        selection_row = _selection_dict(
            scope="physical_layer",
            layer=int(layer),
            bands=layer_bands,
            selection=selection,
            q_index=ref_q,
        )
        selection_row["candidate_id"] = candidate_id
        selection_row["resolved_references_by_band"] = [
            _format_auto_reference_terms(terms) for terms in references_by_band
        ]
        selection_row["assigned_reference_scores"] = [float(score) for score in assigned_scores]
        selection_row["reference_ordering"] = reference_ordering
        selection_row["reference_score_mode"] = reference_score_mode
        selection_row["reference_singular_values"] = [float(value) for value in reference_singular_values.tolist()]
        selection_row["reference_sigma_min"] = float(reference_sigma_min)
        selection_row["reference_condition_number"] = float(reference_condition)
        selection_row["anchor_completion"] = completion_details
        selections.append(selection_row)
        warnings.extend(selection.warnings)
    if not any_layer:
        return None
    report = _auto_gauge_report(
        resolved_norb_fix_list=resolved,
        selections=selections,
        warnings=warnings,
        config=config,
    )
    return ProjectGaugeAnchorCandidate(
        candidate_id=candidate_id,
        resolved_norb_fix_list=resolved,
        report=report,
        priority=int(priority),
    )


def resolve_project_gauge_anchors(
    hamk_reference: np.ndarray,
    q_count: int,
    orb_per_layer0: int,
    num_layer_list: List[int],
    *,
    spin: Literal["up", "down", "all"] = "up",
    Qlayer_list: List[List[np.ndarray]] | None = None,
    num_orb_per_layer_list: List[List[int]] | None = None,
    nlow_state_list: List[List[int]] | None = None,
    norb_fix_list: Any = None,
    gauge_config: Any = None,
    mode: str = "gamma",
    eigensystem_cache: dict[Any, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[list[Any], GaugeAnchorReport]:
    """Resolve manual or automatic gauge anchors to legacy norb_fix_list format."""

    if nlow_state_list is None:
        raise ValueError("project.nlow_state_list must be provided before resolving gauge anchors")
    total_layers = int(sum(int(n) for n in num_layer_list))
    if len(nlow_state_list) != total_layers:
        raise ValueError(
            f"project.nlow_state_list must have {total_layers} physical-layer rows "
            f"(sum(num_layer_list)); got {len(nlow_state_list)}"
        )

    auto_from_norb = _is_auto_token(norb_fix_list)
    auto_from_gauge = _gauge_requests_auto(gauge_config)
    has_manual = norb_fix_list is not None and not auto_from_norb
    if has_manual and auto_from_gauge:
        raise ValueError("manual norb_fix_list cannot be combined with project.gauge auto")
    if has_manual:
        if len(norb_fix_list) != total_layers:
            raise ValueError(
                f"project.norb_fix_list must have {total_layers} physical-layer rows "
                f"(sum(num_layer_list)); got {len(norb_fix_list)}"
            )
        return norb_fix_list, _manual_gauge_report(norb_fix_list)
    if not (auto_from_norb or auto_from_gauge):
        raise ValueError(
            "project.norb_fix_list is missing; write project.gauge: auto or provide manual norb_fix_list anchors"
        )

    config = _auto_gauge_config(gauge_config)
    mode_lower = str(mode).lower()
    ref_q = int(config.reference_q_index)
    if ref_q < 0 or ref_q >= int(q_count):
        raise IndexError(f"project.gauge.reference_q_index={ref_q} outside available Q range 0..{int(q_count) - 1}")
    if Qlayer_list is None:
        Qlayer_list = [[np.arange(q_count) for _ in range(n)] for n in num_layer_list]
    if num_orb_per_layer_list is None:
        num_orb_per_layer_list = [[int(orb_per_layer0) for _ in range(n)] for n in num_layer_list]

    _, h_vec_blk, _, _ = get_H_block(
        np.asarray(hamk_reference, dtype=np.complex128),
        Qlayer_list,
        num_layer_list,
        num_orb_per_layer_list,
        nlow_state_list,
        [],
        spin=spin,
        mode=mode_lower,
        selected_bands_by_layer=nlow_state_list,
        eigensystem_cache=eigensystem_cache,
    )
    resolved: list[Any] = [[] for _ in range(total_layers)]
    selections: list[dict[str, Any]] = []
    warnings: list[str] = []

    if mode_lower == "gamma":
        vec = np.asarray(h_vec_blk[ref_q], dtype=np.complex128)
        bands_flat: list[int] = []
        owners: list[tuple[int, int]] = []
        for layer, bands in enumerate(nlow_state_list):
            for band_pos, band in enumerate(bands):
                bands_flat.append(int(band))
                owners.append((int(layer), int(band_pos)))
        if bands_flat:
            if len(set(bands_flat)) != len(bands_flat):
                raise ValueError("project.gauge auto cannot resolve duplicate gamma low-state band indices")
            u_low = vec[:, np.asarray(bands_flat, dtype=np.intp)]
            segments: list[tuple[int, int]] | None = None
            if spin == "all":
                segments = _gamma_same_q_row_segments(
                    u_low.shape[0],
                    num_layer_list,
                    num_orb_per_layer_list,
                    spin=spin,
                )
            reference = _resolve_gamma_reference_core(u_low, config=config, segments=segments)
            selection = reference.selection
            references_by_band = [list(terms) for terms in reference.references_by_band]
            assigned_scores = list(reference.assigned_scores)
            completion_details = [dict(value) for value in reference.completion_details]
            reference_singular_values = reference.reference_singular_values
            reference_sigma_min = reference.reference_sigma_min
            reference_condition = reference.reference_condition_number
            warnings.extend(reference.completion_warnings)
            for (layer, _band_pos), terms in zip(owners, references_by_band):
                resolved[layer].append(_format_auto_reference_terms(terms))
            selections.append(
                _selection_dict(
                    scope="gamma_same_q",
                    bands=bands_flat,
                    selection=selection,
                    q_index=ref_q,
                )
            )
            selections[-1]["resolved_references_by_band"] = [
                _format_auto_reference_terms(terms) for terms in references_by_band
            ]
            selections[-1]["assigned_reference_scores"] = [float(score) for score in assigned_scores]
            selections[-1]["reference_ordering"] = "gamma_model_frame" if spin == "all" else "overlap_assignment"
            selections[-1]["reference_score_mode"] = "row_max_overlap" if spin == "all" else "assigned_overlap"
            selections[-1]["reference_singular_values"] = [float(value) for value in reference_singular_values]
            selections[-1]["reference_sigma_min"] = float(reference_sigma_min)
            selections[-1]["reference_condition_number"] = float(reference_condition)
            selections[-1]["anchor_completion"] = completion_details
            warnings.extend(selection.warnings)
    else:
        for layer, bands in enumerate(nlow_state_list):
            layer_bands = [int(band) for band in bands]
            if not layer_bands:
                continue
            block_index = int(layer) * int(q_count) + ref_q
            vec = np.asarray(h_vec_blk[block_index], dtype=np.complex128)
            u_low = vec[:, np.asarray(layer_bands, dtype=np.intp)]
            selection = select_anchor_rows_qrcp(
                u_low,
                n_anchors=len(layer_bands),
                basis_is_orthonormal=config.basis_is_orthonormal,
            )
            if selection.sigma_min < config.min_sigma:
                raise ValueError(
                    f"auto gauge layer {layer} sigma_min={selection.sigma_min:.3e} "
                    f"below min_sigma={config.min_sigma:.3e}"
                )
            if selection.condition_number > config.max_condition:
                raise ValueError(
                    f"auto gauge layer {layer} condition_number={selection.condition_number:.3e} "
                    f"exceeds max_condition={config.max_condition:.3e}"
                )
            references = [_auto_reference_terms(int(row)) for row in selection.selected_rows]
            references_by_band, assigned_scores = _assign_anchor_references_to_bands(u_low, references)
            reference_singular_values = _reference_overlap_singular_values(u_low, references_by_band)
            reference_sigma_min = float(np.min(reference_singular_values)) if reference_singular_values.size else 0.0
            reference_sigma_max = float(np.max(reference_singular_values)) if reference_singular_values.size else 0.0
            reference_condition = (
                float("inf") if reference_sigma_min <= 0.0 else float(reference_sigma_max / reference_sigma_min)
            )
            if reference_sigma_min < config.min_sigma:
                raise ValueError(
                    f"auto gauge layer {layer} reference sigma_min={reference_sigma_min:.3e} "
                    f"below min_sigma={config.min_sigma:.3e}"
                )
            if reference_condition > config.max_condition:
                raise ValueError(
                    f"auto gauge layer {layer} reference condition_number={reference_condition:.3e} "
                    f"exceeds max_condition={config.max_condition:.3e}"
                )
            resolved[layer] = [_format_auto_reference_terms(terms) for terms in references_by_band]
            selections.append(
                _selection_dict(
                    scope="physical_layer",
                    layer=layer,
                    bands=layer_bands,
                    selection=selection,
                    q_index=ref_q,
                )
            )
            selections[-1]["resolved_references_by_band"] = [
                _format_auto_reference_terms(terms) for terms in references_by_band
            ]
            selections[-1]["assigned_reference_scores"] = [float(score) for score in assigned_scores]
            selections[-1]["reference_singular_values"] = [float(value) for value in reference_singular_values.tolist()]
            selections[-1]["reference_sigma_min"] = float(reference_sigma_min)
            selections[-1]["reference_condition_number"] = float(reference_condition)
            warnings.extend(selection.warnings)

    sigma_values = [
        float(row.get("reference_sigma_min", row["sigma_min"]))
        for row in selections
        if row.get("reference_sigma_min", row.get("sigma_min")) is not None
    ]
    cond_values = [
        float(row.get("reference_condition_number", row["condition_number"]))
        for row in selections
        if row.get("reference_condition_number", row.get("condition_number")) is not None
    ]
    sigma_min = min(sigma_values) if sigma_values else None
    condition_number = max(cond_values) if cond_values else None
    report = GaugeAnchorReport(
        gauge_mode="auto_scdm",
        resolved_norb_fix_list=resolved,
        selections=selections,
        metric={
            "type": "orthonormal" if config.basis_is_orthonormal else "overlap_metric",
            "basis_is_orthonormal": bool(config.basis_is_orthonormal),
            "formula": "diag(U U^dagger)" if config.basis_is_orthonormal else "diag(S^1/2 U U^dagger S^1/2)",
        },
        state_selection_quality={
            "status": "not_evaluated",
            "reason": "auto gauge fixes anchors for the configured nlow_state_list only",
        },
        gauge_anchor_quality={
            "status": "ok",
            "sigma_min": sigma_min,
            "condition_number": condition_number,
            "min_sigma": float(config.min_sigma),
            "max_condition": float(config.max_condition),
        },
        symmetry_closure_quality={
            "status": "not_available",
            "subspace_leakage": None,
            "reason": "symmetry validation is not available during gauge resolution",
        },
        warnings=warnings,
    )
    return resolved, report


def resolve_project_gauge_anchor_candidates(
    hamk_reference: np.ndarray,
    q_count: int,
    orb_per_layer0: int,
    num_layer_list: List[int],
    *,
    spin: Literal["up", "down", "all"] = "up",
    Qlayer_list: List[List[np.ndarray]] | None = None,
    num_orb_per_layer_list: List[List[int]] | None = None,
    nlow_state_list: List[List[int]] | None = None,
    norb_fix_list: Any = None,
    gauge_config: Any = None,
    mode: str = "gamma",
    eigensystem_cache: dict[Any, tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[ProjectGaugeAnchorCandidate]:
    """Return all finite-basis auto-gauge candidates usable by symmetry validation.

    Manual anchors intentionally produce a single candidate.  Auto gauge always
    keeps the first candidate identical to :func:`resolve_project_gauge_anchors`
    so callers without symmetry data keep the established structural behavior.
    """

    resolved, report = resolve_project_gauge_anchors(
        hamk_reference,
        q_count,
        orb_per_layer0,
        num_layer_list,
        spin=spin,
        Qlayer_list=Qlayer_list,
        num_orb_per_layer_list=num_orb_per_layer_list,
        nlow_state_list=nlow_state_list,
        norb_fix_list=norb_fix_list,
        gauge_config=gauge_config,
        mode=mode,
        eigensystem_cache=eigensystem_cache,
    )
    auto_from_norb = _is_auto_token(norb_fix_list)
    auto_from_gauge = _gauge_requests_auto(gauge_config)
    mode_lower = str(mode).lower()
    primary_id = (
        "manual_norb_fix_list"
        if not (auto_from_norb or auto_from_gauge)
        else "gamma_model_frame"
        if mode_lower == "gamma" and spin == "all"
        else "qrcp_overlap_assignment"
    )
    candidates = [ProjectGaugeAnchorCandidate(primary_id, resolved, report, priority=0)]
    if report.gauge_mode != "auto_scdm":
        return candidates
    if nlow_state_list is None:
        return candidates

    config = _auto_gauge_config(gauge_config)
    total_layers = int(sum(int(n) for n in num_layer_list))
    ref_q = int(config.reference_q_index)
    if Qlayer_list is None:
        Qlayer_list = [[np.arange(q_count) for _ in range(n)] for n in num_layer_list]
    if num_orb_per_layer_list is None:
        num_orb_per_layer_list = [[int(orb_per_layer0) for _ in range(n)] for n in num_layer_list]
    seen = {_canonical_resolved_anchor_key(resolved)}
    if mode_lower != "gamma":
        _, h_vec_blk, _, _ = get_H_block(
            np.asarray(hamk_reference, dtype=np.complex128),
            Qlayer_list,
            num_layer_list,
            num_orb_per_layer_list,
            nlow_state_list,
            [],
            spin=spin,
            mode=mode_lower,
            selected_bands_by_layer=nlow_state_list,
            eigensystem_cache=eigensystem_cache,
        )
        layer_widths = _physical_layer_widths(
            num_layer_list,
            num_orb_per_layer_list,
            fallback_width=orb_per_layer0,
        )
        candidate_defs = [
            ("local_completed_overlap_assignment", "best_overlap", 1),
            ("local_spin_chiral_high_row_lower_template_assignment", "spin_chiral_high_row_lower_template", 2),
            ("local_spin_chiral_high_row_model_frame_assignment", "spin_chiral_high_row_model_frame", 3),
            (
                "local_spin_chiral_high_row_conjugate_model_frame_assignment",
                "spin_chiral_high_row_conjugate_model_frame",
                4,
            ),
            ("local_spin_chiral_high_row_assignment", "spin_chiral_high_row", 5),
            ("local_spin_chiral_high_row_conjugate_assignment", "spin_chiral_high_row_conjugate", 6),
            ("local_spin_chiral_assignment", "spin_chiral", 7),
            ("local_spin_chiral_conjugate_assignment", "spin_chiral_conjugate", 8),
            ("qrcp_delta_overlap_assignment", "raw", 9),
        ]
        for candidate_id, kind, priority in candidate_defs:
            try:
                candidate = _non_gamma_candidate_from_kind(
                    candidate_id=candidate_id,
                    kind=kind,
                    h_vec_blk=h_vec_blk,
                    q_count=q_count,
                    nlow_state_list=nlow_state_list,
                    layer_widths=layer_widths,
                    total_layers=total_layers,
                    spin=spin,
                    config=config,
                    ref_q=ref_q,
                    priority=priority,
                )
            except ValueError:
                continue
            if candidate is None:
                continue
            key = _canonical_resolved_anchor_key(candidate.resolved_norb_fix_list)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
        return candidates
    if spin != "all":
        return candidates
    _, h_vec_blk, _, _ = get_H_block(
        np.asarray(hamk_reference, dtype=np.complex128),
        Qlayer_list,
        num_layer_list,
        num_orb_per_layer_list,
        nlow_state_list,
        [],
        spin=spin,
        mode=mode_lower,
        selected_bands_by_layer=nlow_state_list,
        eigensystem_cache=eigensystem_cache,
    )
    vec = np.asarray(h_vec_blk[ref_q], dtype=np.complex128)
    bands_flat: list[int] = []
    owners: list[tuple[int, int]] = []
    for layer, bands in enumerate(nlow_state_list):
        for band_pos, band in enumerate(bands):
            bands_flat.append(int(band))
            owners.append((int(layer), int(band_pos)))
    if not bands_flat:
        return candidates
    u_low = vec[:, np.asarray(bands_flat, dtype=np.intp)]
    selection = select_anchor_rows_qrcp(
        u_low,
        n_anchors=len(bands_flat),
        basis_is_orthonormal=config.basis_is_orthonormal,
    )
    segments = _gamma_same_q_row_segments(
        u_low.shape[0],
        num_layer_list,
        num_orb_per_layer_list,
        spin=spin,
    )

    raw_references = [_auto_reference_terms(int(row)) for row in selection.selected_rows]
    completed_references, completion_details, completion_warnings = _complete_gamma_spinful_reference_terms(
        u_low,
        selection.selected_rows,
        segments=segments,
    )

    candidate_defs: list[
        tuple[
            str,
            list[list[tuple[int, complex]]],
            str,
            list[dict[str, Any]],
            list[str],
            int,
            bool,
            str,
        ]
    ] = []
    try:
        translated_references, translation_details = _gamma_source_group_translated_references(
            resolved,
            owners,
            num_layer_list=[int(n) for n in num_layer_list],
            segments=segments,
        )
        candidate_defs.append(
            (
                "gamma_source_group_translated_model_frame",
                translated_references,
                "source_group_translated_model_frame",
                translation_details,
                list(report.warnings) + list(selection.warnings),
                1,
                True,
                "source_group_layer_translation",
            )
        )
    except ValueError:
        pass
    candidate_defs.extend(
        [
        (
            "gamma_completed_overlap_assignment",
            completed_references,
            "overlap_assignment",
            completion_details,
            list(completion_warnings) + list(selection.warnings),
            2,
            False,
            "assigned_overlap",
        ),
        (
            "qrcp_delta_overlap_assignment",
            raw_references,
            "overlap_assignment",
            [],
            list(selection.warnings),
            3,
            False,
            "assigned_overlap",
        ),
        ]
    )
    for (
        candidate_id,
        references,
        ordering,
        completion,
        candidate_warnings,
        priority,
        preassigned,
        reference_score_mode,
    ) in candidate_defs:
        try:
            if preassigned:
                references_by_band = [list(ref) for ref in references]
                scores = _reference_overlap_scores(u_low, references_by_band)
                assigned_scores = [float(np.max(scores[index, :])) for index in range(scores.shape[0])]
            else:
                references_by_band, assigned_scores = _assign_anchor_references_to_bands(u_low, references)
            candidate = _gamma_candidate_from_references(
                candidate_id=candidate_id,
                u_low=u_low,
                owners=owners,
                bands_flat=bands_flat,
                selection=selection,
                references_by_band=references_by_band,
                assigned_scores=assigned_scores,
                total_layers=total_layers,
                config=config,
                ref_q=ref_q,
                reference_ordering=ordering,
                reference_score_mode=reference_score_mode,
                anchor_completion=completion,
                warnings=candidate_warnings,
                priority=priority,
            )
        except ValueError:
            continue
        key = _canonical_resolved_anchor_key(candidate.resolved_norb_fix_list)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return candidates


def _physical_layer_entry_index(nlow_state_list: Any, num_layer_arr: np.ndarray, group_index: int, layer_in_group: int) -> int:
    total_layers = int(np.sum(num_layer_arr))
    if len(nlow_state_list) != total_layers:
        raise ValueError(
            f"nlow_state_list must have {total_layers} physical-layer rows "
            f"(sum(num_layer_list)); got {len(nlow_state_list)}. "
            "Use [] for layers that do not contribute."
        )
    return int(np.sum(num_layer_arr[:group_index]) + layer_in_group)


def _source_group_band_lists(nlow_state_list: Any, num_layer_list: List[int]) -> list[list[int]]:
    rows = [[int(band) for band in row] for row in nlow_state_list]
    total_layers = int(sum(num_layer_list))
    if len(rows) != total_layers:
        raise ValueError(
            f"nlow_state_list must have {total_layers} physical-layer rows "
            f"(sum(num_layer_list)); got {len(rows)}. "
            "Use [] for layers that do not contribute."
        )
    grouped: list[list[int]] = []
    offset = 0
    for n_layers in num_layer_list:
        bands: list[int] = []
        for local in range(int(n_layers)):
            bands.extend(rows[offset + local])
        grouped.append(bands)
        offset += int(n_layers)
    return grouped


def _get_H_block_impl(
    Hamk_list: np.ndarray,
    Qlayer_list: List[np.ndarray],
    num_layer_list: List[int],
    num_orb_per_layer_list: List[List[int]],
    nlow_state_list: List[int],
    norb_fix_list: List[int],
    *,
    spin: Literal["up", "down", "all"] = "up",
    mode: Literal["gamma", "K1", "K2"] = "gamma",
    selected_bands_by_layer: list[list[int]] | None = None,
    eigensystem_cache: dict[Any, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble Hamiltonian sub-blocks and diagonalize per-Q selection.

    This ports the core indexing logic from the notebook's get_H_block
    to extract same-Q blocks across layers, handle spins, and compute
    eigenvalues/eigenvectors per block.

    Returns
    - H_GM_diag_eig: array of eigenvalues per block (list -> array)
    - H_GM_diag_eig_vec: array of eigenvectors per block (list -> array)
    - H_diag_block: list/array of extracted Hamiltonian blocks
    - U_new_block: reserved compatibility output for unitary transforms (empty)
    """

    # Inputs can be large dense arrays
    hamk = Hamk_list

    # Normalize list-like shapes
    num_layer_arr = np.array(num_layer_list)
    num_orb_per_layer = [np.array(x) for x in num_orb_per_layer_list]
    total_layers = int(num_layer_arr.sum())
    if len(nlow_state_list) != total_layers:
        raise ValueError(
            f"nlow_state_list must have {total_layers} physical-layer rows "
            f"(sum(num_layer_list)); got {len(nlow_state_list)}. "
            "Use [] for layers that do not contribute."
        )
    if norb_fix_list and len(norb_fix_list) != total_layers:
        raise ValueError(
            f"norb_fix_list must have {total_layers} physical-layer rows "
            f"(sum(num_layer_list)); got {len(norb_fix_list)}."
        )
    if selected_bands_by_layer is not None and len(selected_bands_by_layer) != total_layers:
        raise ValueError(
            f"selected_bands_by_layer must have {total_layers} physical-layer rows "
            f"(sum(num_layer_list)); got {len(selected_bands_by_layer)}."
        )
    # Number of Q points per group
    num_q_list = [np.array([len(q_layer) for q_layer in q_group]) for q_group in Qlayer_list]

    H_diag_block: List[np.ndarray] = []
    H_GM_diag_eig: List[np.ndarray] = []
    H_GM_diag_eig_vec: List[np.ndarray] = []
    U_new_block: List[np.ndarray] = []

    # helper for spin handling
    def apply_spin(idx: np.ndarray) -> np.ndarray:
        if spin == "all":
            return np.concatenate((idx, idx + hamk.shape[1] // 2))
        if spin == "down":
            return idx + hamk.shape[1] // 2
        return idx

    # Expected orbitals per layer (assuming group 0 orbitals define block width)
    orb_per_layer_0 = int(num_orb_per_layer[0][0])
    q_count_0 = int(num_q_list[0][0])

    if mode == "gamma":
        # Same-Q across all layers grouped together
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:  # pragma: no cover
            tqdm = lambda x, total=None, miniters=None: x  # fallback

        # for iqx in tqdm(range(q_count_0), total=q_count_0, miniters=1):
        for iqx in range(q_count_0):
            same_q_index_parts: List[np.ndarray] = []
            for ilx in range(len(num_layer_arr)):
                for j in range(num_layer_arr[ilx]):
                    iqx_ = iqx * num_layer_arr[ilx] + j
                    base = np.arange(iqx_ * orb_per_layer_0, (iqx_ + 1) * orb_per_layer_0)
                    # index2 = np.array([15, 18, 20, 16, 19, 21, 24, 26, 25, 27, 14, 22, 17, 23])-14
                    # iqx2 = index2[iqx] * num_layer_arr[ilx] + j
                    # base_2 = np.arange(iqx2 * orb_per_layer_0, (iqx2 + 1) * orb_per_layer_0)
                    # if ilx != 0:
                    #     print("base_2 = ",base_2)
                    #     print("base = ",base)
                    shift = q_count_0 * orb_per_layer_0
                    if ilx == 0:
                        same_q_index_parts.append(base + shift * num_layer_arr[:ilx].sum())
                    else:
                        same_q_index_parts.append(base + shift * num_layer_arr[:ilx].sum())
            # print(same_q_index_parts)
            same_q_index = np.concatenate(same_q_index_parts)

            expected = int(num_layer_arr.sum()) * orb_per_layer_0
            if same_q_index.shape[0] != expected:
                raise ValueError(
                    f"Mismatch in block size: got {same_q_index.shape[0]} vs expected {expected}"
                )

            same_q_index = apply_spin(same_q_index)
            block = _extract_square_block(hamk, same_q_index)
            # print(block[10,20])
            selected_bands: list[int] | None = None
            if selected_bands_by_layer is not None:
                selected_bands = sorted(
                    {
                        int(band)
                        for layer_bands in selected_bands_by_layer
                        for band in layer_bands
                    }
                )
            eig, vec = _cached_hermitian_eigh_columns(
                hamk=hamk,
                block=block,
                block_indices=same_q_index,
                columns=selected_bands,
                cache=eigensystem_cache,
            )
            # print(np.sort(eig)[:5],np.linalg.norm(block))
            if nlow_state_list and norb_fix_list:
                bands_flat: list[int] = []
                ref_flat: list[list[tuple[int, complex]]] = []
                for layer in range(len(nlow_state_list)):
                    layer_context = f"mode gamma q {iqx} layer {layer}"
                    bands_layer, refs_layer = _reference_terms_for_layer(
                        nlow_state_list=nlow_state_list,
                        norb_fix_list=norb_fix_list,
                        layer=layer,
                        block_dim=vec.shape[0],
                        context=layer_context,
                    )
                    bands_flat.extend(bands_layer)
                    ref_flat.extend(refs_layer)
                if bands_flat:
                    if len(set(bands_flat)) != len(bands_flat):
                        raise ValueError(f"mode gamma q {iqx}: duplicate low-state band indices are ambiguous")
                    _align_selected_eigenstates(
                        vec,
                        bands_flat,
                        ref_flat,
                        context=f"mode gamma q {iqx}",
                    )

            H_diag_block.append(block)
            H_GM_diag_eig.append(eig)

            H_GM_diag_eig_vec.append(vec)

        H_GM_diag_eig = np.array(H_GM_diag_eig, dtype=object)
        H_GM_diag_eig_vec = np.array(H_GM_diag_eig_vec, dtype=object)
        H_diag_block = np.array(H_diag_block, dtype=object)

    else:
        # Per-layer, per-Q blocks
        for ilx in range(len(num_layer_arr)):
            for jj in range(num_layer_arr[ilx]):
                for iq in range(int(num_q_list[ilx][0])):
                    iqx = iq * num_layer_arr[ilx] + jj
                    base = np.arange(iqx * orb_per_layer_0, (iqx + 1) * orb_per_layer_0)
                    shift = q_count_0 * orb_per_layer_0
                    same_q_index = base + shift * num_layer_arr[:ilx].sum()
                    same_q_index = apply_spin(same_q_index)
                    # print(f"ilx = {ilx}, j = {jj}, iq = {iq}, iqx = {iqx}, base = {base}, shift = {shift}, same_q_index = {same_q_index}")
                    block = _extract_square_block(hamk, same_q_index)
                    entry_layer = _physical_layer_entry_index(nlow_state_list, num_layer_arr, int(ilx), int(jj))
                    selected_bands = None if selected_bands_by_layer is None else [int(band) for band in selected_bands_by_layer[entry_layer]]
                    eig, vec = _cached_hermitian_eigh_columns(
                        hamk=hamk,
                        block=block,
                        block_indices=same_q_index,
                        columns=selected_bands,
                        cache=eigensystem_cache,
                    )

                    if nlow_state_list and norb_fix_list:
                        layer_for_global = int(num_layer_arr[:ilx].sum() + jj)
                        context = f"mode {mode} layer {entry_layer} q {iq}"
                        bands_flat, ref_flat = _reference_terms_for_layer(
                            nlow_state_list=nlow_state_list,
                            norb_fix_list=norb_fix_list,
                            layer=entry_layer,
                            block_dim=vec.shape[0],
                            context=context,
                            allow_layer_global=spin != "all" and vec.shape[0] == orb_per_layer_0,
                            layer_for_global=layer_for_global,
                            layer_block_dim=orb_per_layer_0,
                            total_layers=int(num_layer_arr.sum()),
                        )
                        _align_selected_eigenstates(
                            vec,
                            bands_flat,
                            ref_flat,
                            context=context,
                        )



                    H_diag_block.append(block)
                    H_GM_diag_eig.append(eig)
                    H_GM_diag_eig_vec.append(vec)
    return (
        np.array(H_GM_diag_eig, dtype=object),
        np.array(H_GM_diag_eig_vec, dtype=object),
        np.array(H_diag_block, dtype=object),
        np.array(U_new_block, dtype=object),
    )


@wraps(_get_H_block_impl)
def get_H_block(*args, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build all requested blocks under one BLAS thread-limit scope."""

    with _bounded_blas_threads():
        return _get_H_block_impl(*args, **kwargs)


def calculate_energy_lists(
    H_GM_diag_eig_vec,
    nlow_state_list,
    norb_fix_list,
    Qlayer_list,
    num_orb_per_layer_list,
    mode="gamma",
    *,
    include_high: bool = True,
):
    """
    Calculate low energy and high energy lists from the given eigenvector matrix.

    Parameters:
    H_GM_diag_eig_vec (numpy.ndarray): Eigenvector matrix.
    nlow_state_list: List of low energy state indices per layer.
    norb_fix_list: List of fixed orbital indices per layer.
    Qlayer_list: List of Q layers.
    num_orb_per_layer_list: Number of orbitals per layer.
    mode: "gamma" or "K1"/"K2" - determines indexing structure.

    Returns:
    tuple: Low energy list (numpy.ndarray), High energy list (numpy.ndarray).
    """
    # Normalize mode to lowercase
    mode_lower = mode.lower() if isinstance(mode, str) else mode

    theta_rot = 0
    U_low_proj = None
    layer_low_blocks = []
    layer_row_dims = []
    layer_low_col_dims = []
    Qlayer1, Qlayer2 = Qlayer_list
    Qlayer1 = Qlayer1[0]
    Qlayer2 = Qlayer2[0]
    phase = 1

    # Convert H_GM_diag_eig_vec to list if it's an array
    if isinstance(H_GM_diag_eig_vec, np.ndarray):
        H_vec_list = H_GM_diag_eig_vec.tolist()
    else:
        H_vec_list = H_GM_diag_eig_vec

    # Ensure all elements are numpy arrays
    H_vec_list = [np.asarray(v, dtype=np.complex128) for v in H_vec_list]

    for ilayer in range(len(nlow_state_list)):
        nlow_state = nlow_state_list[ilayer]
        Qlayer = Qlayer1 if ilayer == 0 else Qlayer2
        q_count_layer = len(Qlayer)
        vec_probe_idx = 0 if mode_lower == "gamma" else ilayer * len(Qlayer1)
        if vec_probe_idx >= len(H_vec_list):
            raise IndexError(f"vec_probe_idx {vec_probe_idx} >= len(H_vec_list) {len(H_vec_list)} for ilayer={ilayer}")
        vec_probe = np.asarray(H_vec_list[vec_probe_idx], dtype=np.complex128)
        row_dim = vec_probe.shape[0] * q_count_layer
        layer_row_dims.append(row_dim)
        layer_low_col_dims.append(len(nlow_state) * q_count_layer)
        if nlow_state == []:
            layer_low_blocks.append(np.zeros((row_dim, 0), dtype=np.complex128))
            continue
        layer_low = np.zeros((row_dim, len(nlow_state) * q_count_layer), dtype=np.complex128)

        bands_arr = np.asarray(nlow_state, dtype=np.intp)
        col_base = np.arange(len(bands_arr), dtype=np.intp) * q_count_layer
        for i in range(q_count_layer):
            # Index calculation depends on mode
            if mode_lower == "gamma":
                # gamma mode: structure is [Q0_all_layers, Q1_all_layers, ...]
                # But get_H_block returns [Q0, Q1, ...] for gamma mode
                vec_idx = i
            else:
                # non-gamma mode: structure is [layer0_Q0, layer0_Q1, ..., layer0_Qn, layer1_Q0, ...]
                # Index = layer_offset + Q_index
                # Assuming Qlayer1 and Qlayer2 have same length for simplicity
                layer_offset = ilayer * len(Qlayer1)
                vec_idx = layer_offset + i

            if vec_idx >= len(H_vec_list):
                raise IndexError(f"vec_idx {vec_idx} >= len(H_vec_list) {len(H_vec_list)} for ilayer={ilayer}, i={i}")

            vec = np.asarray(H_vec_list[vec_idx], dtype=np.complex128)
            row_start = i * vec.shape[0]
            row_stop = row_start + vec.shape[0]
            layer_low[row_start:row_stop, col_base + i] = vec[:, bands_arr]
        layer_low_blocks.append(layer_low)

    total_low_cols = sum(layer_low_col_dims)
    U_low_proj = np.zeros((sum(layer_row_dims), total_low_cols), dtype=np.complex128)
    row_offset = 0
    col_offset = 0
    for block, row_dim, col_dim in zip(layer_low_blocks, layer_row_dims, layer_low_col_dims):
        if col_dim:
            U_low_proj[row_offset:row_offset + row_dim, col_offset:col_offset + col_dim] = block
        row_offset += row_dim
        col_offset += col_dim

    def gamma_full_row_order(block_dim: int, q_count: int, orb0: int) -> np.ndarray:
        """Map q-major same-Q block rows back to the full Hamiltonian row order."""
        group_layer_counts = [len(group) for group in num_orb_per_layer_list]
        total_layers = sum(group_layer_counts)
        per_spin_dim = total_layers * orb0
        if block_dim == 2 * per_spin_dim:
            spin_count = 2
        elif block_dim == per_spin_dim:
            spin_count = 1
        else:
            raise ValueError(
                f"Cannot infer gamma row order from block_dim={block_dim}, "
                f"orb0={orb0}, layers={group_layer_counts}"
            )

        order = []
        spin_full_offset = q_count * per_spin_dim
        for spin in range(spin_count):
            q_block_spin_offset = spin * per_spin_dim
            for group_index, layer_count in enumerate(group_layer_counts):
                layer_offset = sum(group_layer_counts[:group_index])
                for local in range(q_count * layer_count):
                    q_index = local // layer_count
                    layer_in_group = local % layer_count
                    row_base = (
                        q_index * block_dim
                        + q_block_spin_offset
                        + (layer_offset + layer_in_group) * orb0
                    )
                    order.extend(range(row_base, row_base + orb0))
        if spin_count == 2:
            down_start = q_count * per_spin_dim
            if order[down_start:down_start + orb0] == order[:orb0]:
                raise AssertionError("gamma row-order construction duplicated spin sectors")
        return np.asarray(order, dtype=int)

    # The following reordering is only for gamma mode. get_H_block returns rows
    # as q-major same-Q blocks; full Hamk rows are grouped by source sector.
    gamma_row_order = None
    if mode_lower == "gamma":
        n_g = len(Qlayer1)
        orb0 = num_orb_per_layer_list[0][0]
        block_dim0 = np.asarray(H_vec_list[0], dtype=np.complex128).shape[0]
        gamma_row_order = gamma_full_row_order(block_dim0, n_g, int(orb0))
        U_low_proj = U_low_proj[gamma_row_order]

    # print("shape of Uproj = ",np.shape(U_low_proj))
    if not include_high:
        return U_low_proj, None

    U_high_proj = None
    layer_high_blocks = []
    layer_high_col_dims = []
    for ilayer in range(len(nlow_state_list)):
        nlow_state = nlow_state_list[ilayer]
        Qlayer = Qlayer1 if ilayer == 0 else Qlayer2
        q_count_layer = len(Qlayer)
        vec_probe_idx = 0 if mode_lower == "gamma" else ilayer * len(Qlayer1)
        vec_probe = np.asarray(H_vec_list[vec_probe_idx], dtype=np.complex128)
        high_col_count = vec_probe.shape[1] - len(nlow_state)
        high_block = np.zeros(
            (vec_probe.shape[0] * q_count_layer, high_col_count * q_count_layer),
            dtype=np.complex128,
        )
        high_bands = _complement_indices(vec_probe.shape[1], nlow_state)
        for i in range(len(Qlayer)):
            # Index calculation depends on mode
            if mode_lower == "gamma":
                vec_idx = i
            else:
                # non-gamma mode: structure is [layer0_Q0, layer0_Q1, ..., layer0_Qn, layer1_Q0, ...]
                layer_offset = ilayer * len(Qlayer1)
                vec_idx = layer_offset + i

            if vec_idx >= len(H_vec_list):
                raise IndexError(f"vec_idx {vec_idx} >= len(H_vec_list) {len(H_vec_list)} for ilayer={ilayer}, i={i}")

            # Ensure vec is a numpy array
            vec = np.asarray(H_vec_list[vec_idx], dtype=np.complex128)
            vecpart = vec[:, high_bands]
            # for j in range(vecpart.shape[1]):
            #     vecpart[:,j] = vecpart[:,j] * ((vecpart[norb_fix,j] / np.abs(vecpart[norb_fix,j])) ** (-1)) * np.exp(-1j * 2 * theta_rot * np.pi / 180)
            row_start = i * vec.shape[0]
            row_stop = row_start + vec.shape[0]
            col_start = i * vecpart.shape[1]
            col_stop = col_start + vecpart.shape[1]
            high_block[row_start:row_stop, col_start:col_stop] = vecpart
        layer_high_blocks.append(high_block)
        layer_high_col_dims.append(high_block.shape[1])
    total_high_cols = sum(layer_high_col_dims)
    U_high_proj = np.zeros((sum(layer_row_dims), total_high_cols), dtype=np.complex128)
    row_offset = 0
    col_offset = 0
    for block, row_dim, col_dim in zip(layer_high_blocks, layer_row_dims, layer_high_col_dims):
        if col_dim:
            U_high_proj[row_offset:row_offset + row_dim, col_offset:col_offset + col_dim] = block
        row_offset += row_dim
        col_offset += col_dim

    # The following reordering is only for gamma mode
    if mode_lower == "gamma":
        U_high_proj = U_high_proj[gamma_row_order]

    # print("shape of U_high_proj = ",np.shape(U_high_proj))
    # ULowEnergyList = np.array(ULowEnergyList)
    # UHighEnergyList = np.array(UHighEnergyList)

    # return ULowEnergyList, UHighEnergyList
    return U_low_proj, U_high_proj


def _assemble_projectors_from_block_eigenvectors(
    H_GM_diag_eig_vec,
    idx_list: list[np.ndarray],
    nlow_state_list,
    *,
    include_high: bool,
):
    """Assemble block eigenvectors in the original full-Hamiltonian row order."""
    H_vec_list = [np.asarray(v, dtype=np.complex128) for v in H_GM_diag_eig_vec.tolist()]
    if len(H_vec_list) != len(idx_list):
        raise ValueError(f"projector block count mismatch: {len(H_vec_list)} vectors vs {len(idx_list)} index blocks")
    if not idx_list:
        raise ValueError("projector assembly requires at least one index block")

    full_dim = max(int(np.max(idx)) for idx in idx_list) + 1
    q_count = len(idx_list) // len(nlow_state_list)
    if q_count * len(nlow_state_list) != len(idx_list):
        raise ValueError("index blocks must be ordered by layer then Q")

    bands_by_layer = [[int(band) for band in layer_bands] for layer_bands in nlow_state_list]
    low_offsets: list[int] = []
    low_dim = 0
    for bands in bands_by_layer:
        low_offsets.append(low_dim)
        low_dim += len(bands) * q_count
    U_low = np.zeros((full_dim, low_dim), dtype=np.complex128)
    U_high = None
    if include_high:
        high_dim = sum(vec.shape[1] - len(nlow_state_list[i // q_count]) for i, vec in enumerate(H_vec_list))
        U_high = np.zeros((full_dim, high_dim), dtype=np.complex128)

    high_col = 0
    for block_idx, (vec, idx) in enumerate(zip(H_vec_list, idx_list)):
        layer = block_idx // q_count
        q_index = block_idx % q_count
        bands = bands_by_layer[layer]
        if bands:
            for band_slot, band in enumerate(bands):
                col = low_offsets[layer] + band_slot * q_count + q_index
                U_low[idx, col] = vec[:, band]
        if U_high is not None:
            high_bands = np.delete(np.arange(vec.shape[1]), bands)
            U_high[idx, high_col : high_col + len(high_bands)] = vec[:, high_bands]
            high_col += len(high_bands)

    return U_low, U_high


def _assemble_projector_groups_from_block_eigenvectors(
    H_GM_diag_eig_vec,
    idx_list: list[np.ndarray],
    nlow_state_list,
    *,
    include_high: bool,
):
    """Assemble block-sparse projector groups in the projected basis order."""
    H_vec_list = [np.asarray(v, dtype=np.complex128) for v in H_GM_diag_eig_vec.tolist()]
    if len(H_vec_list) != len(idx_list):
        raise ValueError(f"projector block count mismatch: {len(H_vec_list)} vectors vs {len(idx_list)} index blocks")
    if not idx_list:
        raise ValueError("projector assembly requires at least one index block")

    bands_by_layer = [[int(band) for band in layer_bands] for layer_bands in nlow_state_list]
    q_count = len(idx_list) // len(bands_by_layer)
    if q_count * len(bands_by_layer) != len(idx_list):
        raise ValueError("index blocks must be ordered by layer then Q")

    low_offsets: list[int] = []
    low_dim = 0
    for bands in bands_by_layer:
        low_offsets.append(low_dim)
        low_dim += len(bands) * q_count

    low_locals: list[np.ndarray] = []
    low_cols: list[np.ndarray] = []
    high_locals: list[np.ndarray] = []
    high_cols: list[np.ndarray] = []
    high_dim = 0

    for block_idx, vec in enumerate(H_vec_list):
        layer = block_idx // q_count
        q_index = block_idx % q_count
        bands = bands_by_layer[layer]
        if bands:
            bands_arr = np.asarray(bands, dtype=np.intp)
            cols = np.asarray(
                [low_offsets[layer] + band_slot * q_count + q_index for band_slot in range(len(bands))],
                dtype=np.intp,
            )
            low_locals.append(np.ascontiguousarray(vec[:, bands_arr], dtype=np.complex128))
            low_cols.append(cols)
        else:
            low_locals.append(np.zeros((vec.shape[0], 0), dtype=np.complex128))
            low_cols.append(np.zeros(0, dtype=np.intp))

        if include_high:
            high_bands = _complement_indices(vec.shape[1], bands)
            high_locals.append(np.ascontiguousarray(vec[:, high_bands], dtype=np.complex128))
            high_cols.append(np.arange(high_dim, high_dim + high_bands.size, dtype=np.intp))
            high_dim += int(high_bands.size)

    low_groups = projector_groups_from_block_columns(
        idx_list,
        low_locals,
        low_cols,
        n_columns=low_dim,
    )
    if not include_high:
        return low_groups, None
    high_groups = projector_groups_from_block_columns(
        idx_list,
        high_locals,
        high_cols,
        n_columns=high_dim,
    )
    return low_groups, high_groups


def _flatten_band_groups(nlow_state_list: Any) -> tuple[list[list[int]], list[int]]:
    bands_by_group = [[int(band) for band in group] for group in nlow_state_list]
    flat = [band for group in bands_by_group for band in group]
    if len(flat) != len(set(flat)):
        raise ValueError("Gamma same-Q projector bands must be unique across sector groups")
    return bands_by_group, flat


def _assemble_gamma_projectors_from_block_eigenvectors(
    H_GM_diag_eig_vec,
    idx_list: list[np.ndarray],
    nlow_state_list,
    *,
    include_high: bool,
):
    """Assemble Gamma same-Q projectors without duplicating the full row space.

    Gamma blocks contain all active sectors/layers for a given Q. A nested
    nlow_state_list such as [[0, 1], [2, 3]] therefore describes sector labels
    inside the same block, not independent block row spaces.
    """
    H_vec_list = [np.asarray(v, dtype=np.complex128) for v in H_GM_diag_eig_vec.tolist()]
    if len(H_vec_list) != len(idx_list):
        raise ValueError(f"Gamma projector block count mismatch: {len(H_vec_list)} vectors vs {len(idx_list)} index blocks")
    if not idx_list:
        raise ValueError("Gamma projector assembly requires at least one index block")

    bands_by_group, flat_bands = _flatten_band_groups(nlow_state_list)
    q_count = len(idx_list)
    full_dim = max(int(np.max(idx)) for idx in idx_list) + 1

    low_offsets: list[int] = []
    low_dim = 0
    for bands in bands_by_group:
        low_offsets.append(low_dim)
        low_dim += len(bands) * q_count

    U_low = np.zeros((full_dim, low_dim), dtype=np.complex128)
    high_dim = 0
    high_bands_by_q: list[np.ndarray] = []
    if include_high:
        for vec in H_vec_list:
            high_bands = _complement_indices(vec.shape[1], flat_bands)
            high_bands_by_q.append(high_bands)
            high_dim += int(high_bands.size)
        U_high = np.zeros((full_dim, high_dim), dtype=np.complex128)
    else:
        U_high = None

    high_col = 0
    for q_index, (vec, idx) in enumerate(zip(H_vec_list, idx_list)):
        idx = np.asarray(idx, dtype=np.intp)
        for group_index, bands in enumerate(bands_by_group):
            for band_slot, band in enumerate(bands):
                col = low_offsets[group_index] + band_slot * q_count + q_index
                U_low[idx, col] = vec[:, band]
        if U_high is not None:
            high_bands = high_bands_by_q[q_index]
            U_high[idx, high_col : high_col + high_bands.size] = vec[:, high_bands]
            high_col += int(high_bands.size)

    return U_low, U_high


def _assemble_gamma_projector_groups_from_block_eigenvectors(
    H_GM_diag_eig_vec,
    idx_list: list[np.ndarray],
    nlow_state_list,
    *,
    include_high: bool,
):
    """Block-sparse variant of _assemble_gamma_projectors_from_block_eigenvectors."""
    H_vec_list = [np.asarray(v, dtype=np.complex128) for v in H_GM_diag_eig_vec.tolist()]
    if len(H_vec_list) != len(idx_list):
        raise ValueError(f"Gamma projector block count mismatch: {len(H_vec_list)} vectors vs {len(idx_list)} index blocks")
    if not idx_list:
        raise ValueError("Gamma projector assembly requires at least one index block")

    bands_by_group, flat_bands = _flatten_band_groups(nlow_state_list)
    q_count = len(idx_list)

    low_offsets: list[int] = []
    low_dim = 0
    for bands in bands_by_group:
        low_offsets.append(low_dim)
        low_dim += len(bands) * q_count

    low_locals: list[np.ndarray] = []
    low_cols: list[np.ndarray] = []
    high_locals: list[np.ndarray] = []
    high_cols: list[np.ndarray] = []
    high_dim = 0

    for q_index, vec in enumerate(H_vec_list):
        cols: list[int] = []
        local_band_order: list[int] = []
        for group_index, bands in enumerate(bands_by_group):
            for band_slot, band in enumerate(bands):
                cols.append(low_offsets[group_index] + band_slot * q_count + q_index)
                local_band_order.append(int(band))
        low_cols.append(np.asarray(cols, dtype=np.intp))
        low_locals.append(np.ascontiguousarray(vec[:, np.asarray(local_band_order, dtype=np.intp)], dtype=np.complex128))

        if include_high:
            high_bands = _complement_indices(vec.shape[1], flat_bands)
            high_locals.append(np.ascontiguousarray(vec[:, high_bands], dtype=np.complex128))
            high_cols.append(np.arange(high_dim, high_dim + high_bands.size, dtype=np.intp))
            high_dim += int(high_bands.size)

    low_groups = projector_groups_from_block_columns(
        idx_list,
        low_locals,
        low_cols,
        n_columns=low_dim,
    )
    if not include_high:
        return low_groups, None
    high_groups = projector_groups_from_block_columns(
        idx_list,
        high_locals,
        high_cols,
        n_columns=high_dim,
    )
    return low_groups, high_groups


def project_heff_full(
    hamk_full: np.ndarray,
    q_count: int,
    orb_per_layer0: int,
    num_layer_list: List[int],
    *,
    spin: Literal["up", "down", "all"] = "up",
    bands: List[int] | List[List[int]] = None,
    comps: List[int] | List[List[int]] = None,
    # Optional path: call get_H_block directly to reuse alignment logic.
    Qlayer_list: List[List[np.ndarray]] | None = None,
    num_orb_per_layer_list: List[List[int]] | None = None,
    nlow_state_list: List[List[int]] | None = None,
    norb_fix_list: List[List[List[Tuple[int, complex]]]] | None = None,
    second_order: bool = True,
    E_ref: float | None = None,
    mode = "Gamma",
    downfold_method: str | None = None,
    pole_warning_mev: float = 10.0,
    pole_danger_mev: float = 1.0,
    fail_on_near_pole: bool = False,
    compute_pole_diagnostics: bool = False,
    compute_condition_number: bool = False,
    return_diagnostics: bool = False,
    return_spin_operator: bool = False,
    spin_operator_sign: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project full H(k) to Heff(k).

    一阶：Heff = U_low^† H U_low
    二阶（Löwdin/SW 下折）：Heff = H00 + H01 (E_ref I - H11)^{-1} H10

    参数
    - hamk_full: (N, N) 该 k 点的全空间厄米矩阵
    - q_count: Q 的个数（如 len(Q_set1)）
    - orb_per_layer0: 单自旋、每层、每 Q 的轨道数（用于索引）
    - num_layer_list: 每个“侧/谷”的层数，如 [1,1]
    - spin: 'up'|'down'|'all'（影响索引映射）
    - bands: 每个 Q 选取的“低能带”索引（可平铺或嵌套）；长度 = m_bands
    - comps: 每个低能带希望对齐的“参考全局轨道”索引（与 bands 对应；可选）
    - second_order: False 则一阶；True 则按二阶下折
    - E_ref: 二阶中的参考能量，若 None 则取 H00 本征值均值

    返回
    - Heff(k) (M, M)；其本征值 heig (M,) 与本征矢 hvec (M, M)
    """
    # if bands is None:
    #     raise ValueError("bands (nlow_state_list) must be provided")

    # 将 bands 摊平（comps 另行处理，因其可能含复系数组合）
    # def _flatten(x):
    #     if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
    #         out = []
    #         for s in x:
    #             out.extend(list(s))
    #         return list(out)
    #     return list(x)

    # bands_flat = np.array(_flatten(bands), dtype=int)

    # m_bands = bands_flat.size
    # Normalize mode to lowercase for consistent comparison
    mode_lower = mode.lower() if isinstance(mode, str) else mode
    total_layers = int(sum(int(n) for n in num_layer_list))
    if nlow_state_list is None:
        raise ValueError("nlow_state_list must be provided")
    if len(nlow_state_list) != total_layers:
        raise ValueError(
            f"nlow_state_list must have {total_layers} physical-layer rows "
            f"(sum(num_layer_list)); got {len(nlow_state_list)}. "
            "Use [] for layers that do not contribute."
        )
    has_anchor_list = bool(norb_fix_list)
    if has_anchor_list and len(norb_fix_list) != total_layers:
        raise ValueError(
            f"norb_fix_list must have {total_layers} physical-layer rows "
            f"(sum(num_layer_list)); got {len(norb_fix_list)}."
        )
    if mode_lower == "gamma":
        m_bands = sum(len(layer_bands) for layer_bands in nlow_state_list)
    else:
        nonempty_bands = [layer_bands for layer_bands in nlow_state_list if len(layer_bands) > 0]
        if nonempty_bands:
            m_bands = sum(len(layer_bands) for layer_bands in nlow_state_list)
        else:
            raise ValueError("For non-gamma mode, nlow_state_list must contain at least one active band")
    if m_bands == 0:
        raise ValueError("Empty bands list for projection")

    # 构建每个 Q 的全局索引
    num_layer_arr = np.array(num_layer_list)
    shift = q_count * orb_per_layer0
    idx_list: List[np.ndarray] = []

    if mode_lower == "gamma":
        # gamma mode: each Q combines all layers
        for iqx in range(q_count):
            parts = []
            for ilx in range(len(num_layer_arr)):
                for j in range(num_layer_arr[ilx]):
                    iqx_ = iqx * num_layer_arr[ilx] + j
                    base = np.arange(iqx_ * orb_per_layer0, (iqx_ + 1) * orb_per_layer0)
                    parts.append(base + shift * num_layer_arr[:ilx].sum())
            same_q_index = np.concatenate(parts)
            if spin == "all":
                same_q_index = np.concatenate((same_q_index, same_q_index + hamk_full.shape[0] // 2))
            elif spin == "down":
                same_q_index = same_q_index + hamk_full.shape[0] // 2
            idx_list.append(same_q_index)
    else:
        # non-gamma mode: each (layer, Q) pair is a separate block
        # Structure: [layer0_Q0, layer0_Q1, ..., layer0_Qn, layer1_Q0, layer1_Q1, ..., layer1_Qn]
        for ilx in range(len(num_layer_arr)):
            for j in range(num_layer_arr[ilx]):
                for iqx in range(q_count):
                    iqx_ = iqx * num_layer_arr[ilx] + j
                    base = np.arange(iqx_ * orb_per_layer0, (iqx_ + 1) * orb_per_layer0)
                    same_q_index = base + shift * num_layer_arr[:ilx].sum()
                    if spin == "all":
                        same_q_index = np.concatenate((same_q_index, same_q_index + hamk_full.shape[0] // 2))
                    elif spin == "down":
                        same_q_index = same_q_index + hamk_full.shape[0] // 2
                    idx_list.append(same_q_index)

    method = (downfold_method or ("fixed_schur" if second_order else "first_order")).lower()
    include_high = method != "first_order"

    if Qlayer_list is None:
        Qlayer_list = [[np.arange(q_count) for _ in range(n)] for n in num_layer_list]
    if num_orb_per_layer_list is None:
        num_orb_per_layer_list = [[int(orb_per_layer0) for _ in range(n)] for n in num_layer_list]

    H_eig_blk, H_vec_blk, _, _ = get_H_block(
        hamk_full,
        Qlayer_list,
        num_layer_list,
        num_orb_per_layer_list,
        nlow_state_list,
        norb_fix_list if has_anchor_list else [],
        spin=spin,
        mode=mode_lower,
        selected_bands_by_layer=None if include_high else nlow_state_list,
    )

    options = DownfoldingOptions(
        method=method,
        e_ref=E_ref,
        pole_warning_mev=float(pole_warning_mev),
        pole_danger_mev=float(pole_danger_mev),
        fail_on_near_pole=bool(fail_on_near_pole),
        compute_pole_diagnostics=bool(compute_pole_diagnostics) or bool(fail_on_near_pole),
        compute_condition_number=bool(compute_condition_number),
    )
    if mode_lower == "gamma":
        low_groups, high_groups = _assemble_gamma_projector_groups_from_block_eigenvectors(
            H_vec_blk,
            idx_list,
            _source_group_band_lists(nlow_state_list, num_layer_list),
            include_high=include_high,
        )
        result = downfold_from_projector_groups(hamk_full, low_groups, high_groups, options)
    else:
        low_groups, high_groups = _assemble_projector_groups_from_block_eigenvectors(
            H_vec_blk,
            idx_list,
            nlow_state_list,
            include_high=include_high,
        )
        result = downfold_from_projector_groups(hamk_full, low_groups, high_groups, options)

    Heff = result.heff
    heig, hvec = _hermitian_eigh(Heff)
    extras: list[Any] = []
    if return_diagnostics:
        extras.append(result)
    if return_spin_operator:
        extras.append(
            _project_sz_from_low_groups(
                full_dim=int(hamk_full.shape[0]),
                low_groups=low_groups,
                spin=spin,
                spin_operator_sign=spin_operator_sign,
            )
        )
    if extras:
        return (Heff, heig, hvec, *extras)
    return Heff, heig, hvec

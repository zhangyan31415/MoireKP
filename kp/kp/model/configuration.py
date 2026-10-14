from __future__ import annotations
import ast
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse
import yaml
from .band_degeneracy import DEFAULT_DEGENERACY_TOL_EV
from .schema import (
    canonical_operation_name_for_valley,
    canonical_source_operation_name_for_valley,
    effective_operation_metadata_for_valley,
    validate_model_config,
)
from .symmetry import load_symmetry_source
from ..config.case import normalize_case_config
from ..identity import (
    PROJECTION_ARTIFACT_IDENTITY_FIELDS,
    exactified_operation_provenance_is_complete,
    hash_array,
    load_projection_artifact_identity,
    load_projection_k_indices,
    require_identity_fields,
    require_matching_identity,
)
from ..projection_handoff import (
    GAMMA_ROUTED_ONLY_BASIS_FIELDS,
    load_gamma_common_anchor_basis_spec,
    load_gamma_routed_basis_spec,
)
from ..selection_artifact import (
    CertificationStatus,
    SelectionArtifactStore,
    verify_certified_generic_selection_artifact,
    verify_certified_gamma_selection_artifact,
)
from ..symmetry.action_schema import (
    SOURCE_MATRIX_SEMANTICS,
    allows_inferred_action_metadata,
)
from ..symmetry.geometry import (
    bM_candidates_from_q_distances,
    canonical_bM_pair_from_candidates,
    sector_qset,
    sectors_with_q_offsets,
)
from .core import (
    MoireConfig,
    generate_kpath_from_file,
    load_Q_sets_from_gvec_files,
    reciprocal_Tmat_from_Tmat,
    rot,
)
from .model_selection import (
    FamilyOrders,
    ModelSelectionConfig,
    parse_model_selection_config,
)


DEFAULT_MAX_ORDER = {"Kinect": 2, "intra": 0, "inter": 0}



HARMONICS_USER_ALIASES = {
    "intralayer": "intra",
    "interlayer": "inter",
}



MAX_ORDER_USER_ALIASES = {
    "kinetic": "Kinect",
    "intralayer": "intra",
    "interlayer": "inter",
}



SYMMETRY_MAP_USER_ALIASES = {
    "kinetic": "Kinect",
    "onsite": "Onsite",
    "intralayer": "intra",
    "interlayer": "inter",
}



MAX_DERIVATIVE_ORDER_USER_ALIASES = {
    "intralayer_zero": "moire_intra_zero",
    "intralayer_nonzero": "moire_intra_nonzero",
    "interlayer_zero": "tunneling_zero",
    "interlayer_nonzero": "tunneling_nonzero",
}



GAMMA_ORDER_FAMILIES = {
    "tunneling_zero": "inter",
    "tunneling_nonzero": "inter",
    "moire_intra_zero": "intra",
    "moire_intra_nonzero": "intra",
    "kinetic": "Kinect",
}



SOURCE_META_KEYS = (
    "source_matrix_role",
    "source_gauge",
    "target_role",
    "gauge_correction",
    "antiunitary_convention",
    "spin_map",
    "valley_map",
)



_SUPPORTED_MODEL_PROFILES = (
    "linear/low",
    "linear/high",
    "nonlinear/low",
    "nonlinear/high",
)



@dataclass
class ConfiguredModel:
    path: Path
    raw: dict[str, Any]
    source_config: Path
    source_raw: dict[str, Any]
    qset1_file: Path
    qset2_file: Path
    kpoints_file: Path | None
    heff_file: Path
    heff_eig_file: Path | None
    output_dir: Path
    rotation_deg: float
    fit_indices: list[int]
    fit_selection_metadata: dict[str, Any]
    band_indices: list[int] | None
    n_orb: tuple[int, int]
    nlow_state: list[int]
    bM_config: dict[str, Any]
    harmonics_config: dict[str, Any]
    max_order: dict[str, int]
    symmetry_map: dict[str, list[dict[str, Any]]]
    coeff_tol: float
    coeff_prune_threshold: float
    compare_to_heff: bool
    response_semantics: str = "complete_linear"
    response_regularization: float = 0.0
    response_fit_objective: dict[str, Any] = field(
        default_factory=lambda: {"mode": "equal_matrix"}
    )
    kpath_config: dict[str, Any] = field(default_factory=dict)
    valley_model: dict[str, Any] = field(default_factory=dict)
    symmetry_source_config: dict[str, Any] = field(default_factory=dict)
    symmetry_source_metadata: dict[str, Any] = field(default_factory=dict)
    model_basis_gauge: np.ndarray | None = field(default=None, repr=False)
    sectors_config: list[dict[str, Any]] = field(default_factory=list)
    term_templates: list[dict[str, Any]] = field(default_factory=list)
    term_template_metadata: dict[str, Any] = field(default_factory=dict)
    output_config: dict[str, Any] = field(default_factory=dict)
    band_slice: list[int] | None = None
    band_plot_config: dict[str, Any] = field(default_factory=dict)
    bM_diagnostics: dict[str, Any] = field(default_factory=dict)
    harmonics_diagnostics: dict[str, Any] = field(default_factory=dict)
    automatic_harmonic_selection: dict[str, Any] = field(default_factory=dict)
    orbital_count_metadata: dict[str, Any] = field(default_factory=dict)
    validation_config: dict[str, Any] = field(default_factory=dict)
    band_refinement_config: dict[str, Any] = field(default_factory=dict)
    null_channel_abs_tol: float = 0.0
    null_channel_rel_tol: float = 0.0
    artifact_identity: dict[str, Any] = field(default_factory=dict)
    project_k_indices: list[int] = field(default_factory=list)
    kpoints_from_projection: bool = False
    model_selection_config: ModelSelectionConfig | None = None
    model_selection_excluded_term_keys: tuple[Any, ...] = ()
    requested_profiles: tuple[str, ...] = ()
    fit_weighting: str = "automatic"
    fit_method_config: dict[str, Any] = field(default_factory=dict)



def _canonical_public_fit_method(
    fit: Mapping[str, Any],
    *,
    heff_shape: Sequence[int],
) -> dict[str, Any]:
    method_raw = fit.get("method")
    if method_raw is None:
        return {}
    method = str(method_raw).strip().lower()
    if method not in {"linear", "nonlinear"}:
        raise ValueError("fit.method must be 'linear' or 'nonlinear'")
    if method == "linear":
        nonlinear_only = [
            key for key in ("band_kpoints", "band_loss_weight", "max_steps") if key in fit
        ]
        if nonlinear_only:
            raise ValueError(
                f"fit.{nonlinear_only[0]} is only valid for fit.method: nonlinear"
            )
    incompatible = [
        key
        for key in (
            "mode",
            "indices",
            "objective",
            "refine_bands",
            "weighting",
            "model_selection",
        )
        if key in fit
    ]
    if incompatible:
        raise ValueError(
            "fit.method cannot be mixed with incompatible fit keys: " + ", ".join(incompatible)
        )
    allowed = {
        "method",
        "kpoints",
        "bands",
        "one_sided_weight",
        "two_sided_weight",
    }
    if method == "nonlinear":
        allowed.update({"band_kpoints", "band_loss_weight", "max_steps", "degeneracy_tol_ev"})
    unsupported = sorted(str(key) for key in fit if key not in allowed)
    if unsupported:
        raise ValueError(
            "fit.method does not accept advanced fit keys: "
            + ", ".join(unsupported)
        )
    if len(heff_shape) != 3 or int(heff_shape[-1]) != int(heff_shape[-2]):
        raise ValueError(f"fit.method requires Heff with shape (Nk,dim,dim), got {tuple(heff_shape)}")
    n_kpoints = int(heff_shape[0])
    dimension = int(heff_shape[-1])

    def checked_indices(value: Any, *, name: str) -> list[int]:
        if value is None:
            raise ValueError(f"{name} is required")
        indices = _as_int_list(value, name=name)
        if not indices:
            raise ValueError(f"{name} must be a non-empty list")
        if len(set(indices)) != len(indices):
            raise ValueError(f"{name} must contain unique row indices")
        if any(index < 0 or index >= n_kpoints for index in indices):
            raise ValueError(f"{name} entries must be in 0..{n_kpoints - 1}")
        return [int(value) for value in indices]

    hamiltonian_kpoints = checked_indices(fit.get("kpoints"), name="fit.kpoints")
    requested_bands = int(fit.get("bands", 0))
    if requested_bands <= 0 or requested_bands > dimension:
        raise ValueError(
            f"fit.bands must be positive and not exceed Hamiltonian dimension {dimension}"
        )

    def checked_weight(key: str, *, positive: bool = False) -> float:
        if key not in fit:
            raise ValueError(f"fit.{key} is required")
        value = float(fit[key])
        valid = value > 0.0 if positive else value >= 0.0
        if not np.isfinite(value) or not valid:
            condition = "positive" if positive else "non-negative"
            raise ValueError(f"fit.{key} must be finite and {condition}")
        return value

    one_sided_weight = checked_weight("one_sided_weight")
    two_sided_weight = checked_weight("two_sided_weight")
    band_kpoints: list[int] | None = None
    band_loss_weight: float | None = None
    max_steps: int | None = None
    degeneracy_tol_ev = float(fit.get("degeneracy_tol_ev", DEFAULT_DEGENERACY_TOL_EV))
    if not np.isfinite(degeneracy_tol_ev) or degeneracy_tol_ev <= 0.0:
        raise ValueError("fit.degeneracy_tol_ev must be positive and finite")
    if method == "nonlinear":
        raw_band_kpoints = fit.get("band_kpoints")
        if isinstance(raw_band_kpoints, str) and raw_band_kpoints.strip().lower() == "all":
            band_kpoints = list(range(int(heff_shape[0])))
        else:
            band_kpoints = checked_indices(raw_band_kpoints, name="fit.band_kpoints")
        band_loss_weight = checked_weight("band_loss_weight", positive=True)
        max_steps = int(fit.get("max_steps", 30))
        if max_steps <= 0:
            raise ValueError("fit.max_steps must be positive")
    return {
        "method": method,
        "hamiltonian_kpoints": [int(value) for value in hamiltonian_kpoints],
        "band_kpoints": band_kpoints,
        "requested_bands": requested_bands,
        "one_sided_weight": one_sided_weight,
        "two_sided_weight": two_sided_weight,
        "band_loss_weight": band_loss_weight,
        "max_steps": max_steps,
        **({"degeneracy_tol_ev": degeneracy_tol_ev} if "degeneracy_tol_ev" in fit else {}),
        "normalization": "dimension_mean_square",
    }



def _parse_requested_model_profiles(raw_profiles: Any) -> tuple[str, ...]:
    """Normalize optional public model profile identifiers."""

    if raw_profiles is None:
        return ()
    if not isinstance(raw_profiles, Sequence) or isinstance(raw_profiles, (str, bytes)):
        raise ValueError("model.profiles must be a non-empty list of solver/quality names")
    profiles = tuple(str(value).strip().lower() for value in raw_profiles)
    if not profiles:
        raise ValueError("model.profiles must not be empty")
    if any(not profile for profile in profiles):
        raise ValueError("model.profiles entries must not be empty")
    if len(set(profiles)) != len(profiles):
        raise ValueError("model.profiles contains duplicate profile names")
    unsupported = [profile for profile in profiles if profile not in _SUPPORTED_MODEL_PROFILES]
    if unsupported:
        raise ValueError(
            f"unsupported model profile(s) {unsupported}; supported profiles are "
            f"{list(_SUPPORTED_MODEL_PROFILES)}"
        )
    return profiles



def _parse_fit_weighting(raw_weighting: Any) -> str:
    """Normalize the public automatic-fit weighting policy."""

    if raw_weighting in (None, ""):
        return "automatic"
    weighting = str(raw_weighting).strip().lower()
    if weighting == "auto":
        weighting = "automatic"
    if weighting not in {"automatic", "uniform"}:
        raise ValueError("fit.weighting must be 'automatic' or 'uniform'")
    return weighting



def _resolve_path(value: str | Path | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()



def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data



SOURCE_SEMANTICS: dict[str, Any] = {
    "source_matrix_role": "raw_h_sewing_action",
    "source_gauge": "raw_saved_TAPW",
    "target_role": "continuum_internal_rep",
    "gauge_correction": {"kind": "none"},
    "spin_map": "from_kp_symm_output",
    "valley_map": "identity",
}



def _as_int_list(value: Any, *, name: str) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a list of integers, got {value!r}")
    return [int(item) for item in value]



def _num_layer_list_from_material(material: Mapping[str, Any]) -> list[int]:
    raw = material.get("num_layer_list")
    if raw is not None:
        layers = _as_int_list(raw, name="source_config.material.num_layer_list")
        if not layers or any(value <= 0 for value in layers):
            raise ValueError(f"source_config material.num_layer_list must contain positive integers, got {raw!r}")
        return layers
    num_layers = int(material.get("num_layers", 2))
    if num_layers <= 0:
        raise ValueError(f"source_config material.num_layers must be positive, got {num_layers}")
    if num_layers == 1:
        return [1, 0]
    return [1, num_layers - 1]



def _layer_groups_metadata(num_layer_list: Sequence[int], values: Sequence[int]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    layer_start = 1
    for group_index, layer_count_raw in enumerate(num_layer_list):
        layer_count = int(layer_count_raw)
        layer_stop = layer_start + layer_count
        group_values = [int(item) for item in values[layer_start - 1 : layer_stop - 1]]
        groups.append(
            {
                "qset": f"qset{group_index + 1}",
                "source_group": group_index + 1,
                "layers": list(range(layer_start, layer_stop)),
                "values": group_values,
                "total": int(sum(group_values)),
            }
        )
        layer_start = layer_stop
    return groups



def _resolve_layerwise_counts(
    values: Sequence[int],
    *,
    name: str,
    num_layer_list: Sequence[int],
    default_from: str | None = None,
    prefer_active_layer_sectors: bool = False,
) -> tuple[list[int], dict[str, Any]]:
    raw_values = [int(value) for value in values]
    qset_count = 2
    total_layers = int(sum(int(value) for value in num_layer_list))
    metadata: dict[str, Any] = {
        "raw": list(raw_values),
        "num_layer_list": [int(value) for value in num_layer_list],
        "total_layers": total_layers,
    }
    if default_from is not None:
        metadata["default_from"] = default_from

    if len(raw_values) == qset_count and (total_layers == qset_count or default_from is not None):
        metadata.update(
            {
                "input_kind": "default_from_n_orb" if default_from is not None else "qset",
                "resolved_qset": list(raw_values),
            }
        )
        return raw_values, metadata
    if len(raw_values) != total_layers:
        raise ValueError(
            f"{name} must have {total_layers} physical-layer entries "
            f"(sum(source_config material.num_layer_list)); got {len(raw_values)} values: {raw_values}"
        )
    if len(num_layer_list) != qset_count:
        raise ValueError(
            f"{name} physical-layer counts require exactly two source groups for qset1/qset2 resolution, "
            f"got source_config material.num_layer_list={list(num_layer_list)}"
        )

    groups = _layer_groups_metadata(num_layer_list, raw_values)
    qset_values = [int(group["total"]) for group in groups]
    active_layer_values = [int(value) for value in raw_values if int(value) > 0]
    resolved_model_sectors = (
        active_layer_values
        if prefer_active_layer_sectors and len(active_layer_values) == qset_count
        else qset_values
    )
    metadata.update(
        {
            "input_kind": "default_from_n_orb" if default_from is not None else "physical_layer",
            "resolved_qset": list(qset_values),
            "resolved_model_sectors": list(resolved_model_sectors),
            "prefer_active_layer_sectors": bool(prefer_active_layer_sectors),
            "groups": groups,
        }
    )
    return resolved_model_sectors, metadata



def _resolve_projection_qset_counts(
    values: Sequence[int],
    *,
    num_layer_list: Sequence[int],
) -> tuple[list[int], dict[str, Any]]:
    """Resolve routed projection ranks that already belong to source qsets."""

    raw_values = [int(value) for value in values]
    if len(num_layer_list) != 2 or len(raw_values) != 2:
        raise ValueError(
            "projection Gamma group_ranks must contain one count for each of "
            f"qset1/qset2; got ranks={raw_values}, num_layer_list={list(num_layer_list)}"
        )
    if any(value < 0 for value in raw_values) or sum(raw_values) <= 0:
        raise ValueError(
            "projection Gamma sector ranks must be non-negative with at least "
            f"one active sector, got {raw_values}"
        )
    return raw_values, {
        "raw": list(raw_values),
        "num_layer_list": [int(value) for value in num_layer_list],
        "total_layers": int(sum(int(value) for value in num_layer_list)),
        "input_kind": "projection_qset",
        "resolved_qset": list(raw_values),
        "resolved_model_sectors": list(raw_values),
    }



def _infer_k_sectors_from_layerwise_counts(
    n_orb_resolution: Mapping[str, Any],
    *,
    valley_model: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if str(valley_model.get("valley_type", "")) != "K":
        return []
    if str(n_orb_resolution.get("input_kind", "")) != "physical_layer":
        return []
    groups = n_orb_resolution.get("groups", [])
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        return []

    sectors: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        qset = str(group.get("qset", ""))
        layers = group.get("layers", [])
        values = group.get("values", [])
        if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
            continue
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for layer, value in zip(layers, values):
            n_orb = int(value)
            if n_orb <= 0:
                continue
            sectors.append({"name": f"L{int(layer)}", "qset": qset, "n_orb": n_orb})
    return sectors



def _model_section(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    section = raw.get("model", {})
    if not isinstance(section, Mapping):
        raise ValueError("model section must be a mapping")
    return section



def _canonical_response_fit_objective(
    raw_objective: Any,
    *,
    response_semantics: str,
    target_bands: str,
    default_min_bands: int,
) -> dict[str, Any]:
    """Validate the target-dependent complete-response fitting objective."""
    if raw_objective in (None, {}):
        return {"mode": "equal_matrix"}
    if not isinstance(raw_objective, Mapping):
        raise ValueError("fit.objective must be a mapping when provided")
    objective = dict(raw_objective)
    mode = str(objective.get("mode", "equal_matrix")).strip().lower()
    if mode in {"equal_matrix", "equal_matrix"}:
        return {"mode": "equal_matrix"}
    if mode != "target_spectral_linear":
        raise ValueError(
            "fit.objective.mode must be 'equal_matrix' or 'target_spectral_linear'"
        )
    if response_semantics != "complete_linear":
        raise ValueError(
            "fit.objective.mode='target_spectral_linear' requires "
            "model.response_semantics='complete_linear'"
        )
    reference = str(
        objective.get("target_reference", "current_heff_support_mask")
    ).strip().lower()
    if reference != "current_heff_support_mask":
        raise ValueError(
            "target_spectral_linear currently requires "
            "fit.objective.target_reference='current_heff_support_mask'"
        )
    edge = str(objective.get("edge", target_bands)).strip().lower()
    if edge not in {"top", "bottom"}:
        raise ValueError("fit.objective.edge must be 'top' or 'bottom'")
    if edge != target_bands:
        raise ValueError(
            "fit.objective.edge must match model.target_bands so fitting and validation "
            "use the same band edge"
        )
    window_raw = objective.get("window", {})
    if not isinstance(window_raw, Mapping):
        raise ValueError("fit.objective.window must be a mapping")
    window = dict(window_raw)
    window_mode = str(
        window.get("mode", "auto_gap_degeneracy_safe")
    ).strip().lower()
    degeneracy_tol_mev = float(window.get("degeneracy_tol_mev", 0.1))
    if not np.isfinite(degeneracy_tol_mev) or degeneracy_tol_mev < 0.0:
        raise ValueError("fit.objective.window.degeneracy_tol_mev must be non-negative")
    if window_mode == "fixed_count_degeneracy_safe":
        bands = int(window.get("bands", 0))
        if bands <= 0:
            raise ValueError(
                "fit.objective.window.bands must be positive for fixed_count_degeneracy_safe"
            )
        canonical_window = {
            "mode": window_mode,
            "bands": bands,
            "degeneracy_tol_mev": degeneracy_tol_mev,
        }
    elif window_mode == "auto_gap_degeneracy_safe":
        min_bands = int(window.get("min_bands", default_min_bands))
        max_bands = int(window.get("max_bands", max(min_bands, 2 * default_min_bands)))
        if min_bands <= 0 or max_bands < min_bands:
            raise ValueError(
                "fit.objective.window requires 0 < min_bands <= max_bands"
            )
        canonical_window = {
            "mode": window_mode,
            "min_bands": min_bands,
            "max_bands": max_bands,
            "degeneracy_tol_mev": degeneracy_tol_mev,
        }
    else:
        raise ValueError(
            "fit.objective.window.mode must be 'fixed_count_degeneracy_safe' "
            "or 'auto_gap_degeneracy_safe'"
        )
    floor = float(objective.get("floor", 0.05))
    alpha = float(objective.get("alpha", 1.0))
    if not np.isfinite(floor) or not 0.0 < floor <= 1.0:
        raise ValueError("fit.objective.floor must satisfy 0 < floor <= 1")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("fit.objective.alpha must be non-negative")
    normalization = str(
        objective.get("normalization", "mean_trace_per_dimension")
    ).strip().lower()
    if normalization != "mean_trace_per_dimension":
        raise ValueError(
            "fit.objective.normalization must be 'mean_trace_per_dimension'"
        )
    projector_raw = objective.get("two_sided_projector")
    projector: dict[str, Any] | None = None
    if projector_raw is not None:
        if not isinstance(projector_raw, Mapping):
            raise ValueError("fit.objective.two_sided_projector must be a mapping")
        enabled_raw = projector_raw.get("enabled", True)
        if not isinstance(enabled_raw, (bool, np.bool_)):
            raise ValueError("fit.objective.two_sided_projector.enabled must be boolean")
        enabled = bool(enabled_raw)
        weight = float(projector_raw.get("weight", 0.0))
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(
                "fit.objective.two_sided_projector.weight must be finite and non-negative"
            )
        if enabled and weight <= 0.0:
            raise ValueError(
                "fit.objective.two_sided_projector.weight must be positive when enabled"
            )
        if enabled:
            projector = {"enabled": True, "weight": weight}
    canonical = {
        "mode": "target_spectral_linear",
        "target_reference": reference,
        "edge": edge,
        "window": canonical_window,
        "floor": floor,
        "alpha": alpha,
        "normalization": normalization,
    }
    if projector is not None:
        canonical["two_sided_projector"] = projector
    return canonical



def _get_path_value(raw: Mapping[str, Any], key: str) -> Any:
    if key in raw:
        return raw[key]
    input_section = raw.get("input", {})
    if isinstance(input_section, Mapping) and key in input_section:
        return input_section[key]
    return None



def _projection_layerwise_orbital_counts(
    *,
    heff_file: Path,
    project: Mapping[str, Any],
    num_layer_list: Sequence[int],
) -> tuple[list[int] | None, str | None]:
    """Return orbital counts recorded by the projection step."""

    total_layers = int(sum(int(value) for value in num_layer_list))
    rows: Any = None
    source: str | None = None
    basis_file = heff_file.parent / "basis.npz"
    if basis_file.is_file():
        with np.load(basis_file, allow_pickle=False) as discriminator:
            basis_files = frozenset(discriminator.files)
            basis_kind = "explicit_layer"
            if "projection_basis_kind" in discriminator.files:
                basis_kind_raw = np.asarray(discriminator["projection_basis_kind"])
                if basis_kind_raw.shape != ():
                    raise ValueError("projection basis kind must be a scalar")
                basis_kind = str(basis_kind_raw.item())
        if basis_kind == "gamma_routed":
            handoff = load_gamma_routed_basis_spec(basis_file)
            return [int(rank) for rank in handoff.group_ranks], "projection_gamma_routed"
        if basis_kind == "gamma_common_anchor":
            handoff = load_gamma_common_anchor_basis_spec(basis_file)
            return (
                [int(rank) for rank in handoff.continuum_sector_ranks],
                "projection_gamma_common_anchor",
            )
        if basis_kind != "explicit_layer":
            raise ValueError(f"unsupported projection_basis_kind: {basis_kind!r}")
        routed_fields = sorted(basis_files.intersection(GAMMA_ROUTED_ONLY_BASIS_FIELDS))
        if routed_fields:
            raise ValueError(
                "explicit layer basis contains routed-only fields: "
                + ", ".join(routed_fields)
            )
        with np.load(basis_file, allow_pickle=True) as basis:
            if "nlow_state_list" in basis.files:
                rows = basis["nlow_state_list"].tolist()
                source = "projection_basis"
    if rows is None:
        project_rows = project.get("nlow_state_list")
        if project_rows not in (None, []):
            rows = project_rows
            source = "source_project"
    if rows is None:
        return None, None
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError(f"projection nlow_state_list must be a sequence, got {type(rows).__name__}")
    if len(rows) != total_layers:
        raise ValueError(
            "projection nlow_state_list must contain one entry per physical layer: "
            f"expected {total_layers}, got {len(rows)}"
        )
    counts: list[int] = []
    for layer_index, row in enumerate(rows, start=1):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError(
                "projection nlow_state_list entries must be band-index lists; "
                f"layer {layer_index} has {type(row).__name__}"
            )
        counts.append(len(row))
    return counts, source



def _default_symmetry_map_from_valley_model(valley_model: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    names = valley_model.get("allowed_internal_symmetries", [])
    if names is None:
        names = []
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        raise ValueError("valley_model.allowed_internal_symmetries must be a list")
    ops = [{"name": str(name)} for name in names]
    return {"Kinect": list(ops), "Onsite": list(ops), "intra": list(ops), "inter": list(ops)}



def _operation_rows_from_symmetry_source(
    symmetry_source: Mapping[str, Any],
    valley_model: Mapping[str, Any],
) -> list[dict[str, Any]]:
    operations = symmetry_source.get("operations", [])
    if isinstance(operations, Mapping):
        operations = [
            {**(dict(value) if isinstance(value, Mapping) else {}), "name": str(key)}
            for key, value in operations.items()
        ]
    if operations is None:
        operations = []
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)):
        raise ValueError("symmetry_source.operations must be a list or mapping when provided")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for operation in operations:
        if isinstance(operation, Mapping):
            raw_name = operation.get("name", operation.get("operation"))
        else:
            raw_name = operation
        if raw_name is None:
            continue
        user_name = str(raw_name)
        canonical_name = canonical_operation_name_for_valley(user_name, valley_model)
        if canonical_name in seen:
            continue
        seen.add(canonical_name)
        row = {
            "name": canonical_name,
            **effective_operation_metadata_for_valley(user_name, valley_model),
        }
        if user_name != canonical_name:
            row["user_name"] = user_name
        rows.append(row)
    return rows



def _default_symmetry_map_from_source_or_valley(
    symmetry_source: Mapping[str, Any],
    valley_model: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    ops = _operation_rows_from_symmetry_source(symmetry_source, valley_model)
    if not ops:
        return _default_symmetry_map_from_valley_model(valley_model)
    return {
        "Kinect": [dict(op) for op in ops],
        "Onsite": [dict(op) for op in ops],
        "intra": [dict(op) for op in ops],
        "inter": [dict(op) for op in ops],
    }



def _valley_type_from_label(label: str) -> str:
    if label == "Gamma":
        return "Gamma"
    if label.startswith("K"):
        return "K"
    if label.startswith("M"):
        return "M"
    raise ValueError(f"Unsupported valley label {label!r}; expected Gamma, K*, or M*")



def _spin_convention_from_short(spin: str, valley_type: str) -> str:
    if spin in {"spinful", "all"}:
        return "spinful"
    if spin in {"spin_up_projected", "up"}:
        return "spin_up_projected"
    if spin in {"spin_down_projected", "down"}:
        return "spin_down_projected"
    if spin == "spinless_effective":
        return "spinless_effective"
    raise ValueError(
        f"Unsupported spin value {spin!r}; use spinful, spin_up_projected, "
        "spin_down_projected, or spinless_effective"
    )



def _default_internal_symmetries(valley_type: str, spin_convention: str) -> list[str]:
    if valley_type == "Gamma":
        return ["C3z", "TR", "C2"] if spin_convention == "spinful" else ["C3z", "C2"]
    if valley_type == "K":
        return ["C3z", "C2T"]
    if valley_type == "M":
        return ["TR", "C2"]
    raise ValueError(f"Unsupported valley_type {valley_type!r}")



def _normalize_short_model_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(raw)

    short_valley = out.get("valley")
    valley_model = out.get("valley_model")
    if short_valley is not None:
        valley_label = str(short_valley)
        valley_type = _valley_type_from_label(valley_label)
        spin = str(out.get("spin", "spinful" if valley_type == "Gamma" else "spinless"))
        spin_convention = _spin_convention_from_short(spin, valley_type)
        out["valley_model"] = {
            "lattice": "hexagonal",
            "system": "bilayer",
            "valley_type": valley_type,
            "mode": "single_valley",
            "active_valleys": [valley_label],
            "spin_convention": spin_convention,
            "allowed_internal_symmetries": _default_internal_symmetries(valley_type, spin_convention),
            "external_sewing_symmetries": [],
        }
    elif not isinstance(valley_model, Mapping):
        out["valley_model"] = {}

    return out



def _normalize_alias_mapping(
    raw: Any,
    *,
    aliases: Mapping[str, str],
    section_name: str,
) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{section_name} must be a mapping")
    out = dict(raw)
    for alias, canonical in aliases.items():
        if alias not in out:
            continue
        if canonical in out and out[canonical] != out[alias]:
            raise ValueError(
                f"{section_name} uses both {canonical!r} and alias {alias!r}; keep only one spelling"
            )
        out[canonical] = out.pop(alias)
    return out



def _normalize_model_user_aliases(model: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(model)
    if "harmonics" in out:
        out["harmonics"] = _normalize_alias_mapping(
            out.get("harmonics"),
            aliases=HARMONICS_USER_ALIASES,
            section_name="model.harmonics",
        )
    if "max_order" in out:
        out["max_order"] = _normalize_alias_mapping(
            out.get("max_order"),
            aliases=MAX_ORDER_USER_ALIASES,
            section_name="model.max_order",
        )
    if "max_derivative_order" in out:
        out["max_derivative_order"] = _normalize_alias_mapping(
            out.get("max_derivative_order"),
            aliases=MAX_DERIVATIVE_ORDER_USER_ALIASES,
            section_name="model.max_derivative_order",
        )
    if "symmetry_map" in out:
        out["symmetry_map"] = _normalize_alias_mapping(
            out.get("symmetry_map"),
            aliases=SYMMETRY_MAP_USER_ALIASES,
            section_name="model.symmetry_map",
        )
    return out



def _normalize_user_symmetry_names(raw: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    valley_model = out.get("valley_model", {})
    if not isinstance(valley_model, Mapping):
        return out
    valley_model_out = dict(valley_model)
    allowed = valley_model_out.get("allowed_internal_symmetries")
    if isinstance(allowed, Sequence) and not isinstance(allowed, (str, bytes)):
        valley_model_out["allowed_internal_symmetries"] = [
            canonical_operation_name_for_valley(str(name), valley_model_out) for name in allowed
        ]
        valley_model_out["allowed_internal_symmetries_user"] = [str(name) for name in allowed]
    out["valley_model"] = valley_model_out

    model = out.get("model", {})
    if not isinstance(model, Mapping):
        return out
    model_out = dict(model)
    if "symmetry_map" not in model_out:
        out["model"] = model_out
        return out
    symmetry_map = model_out.get("symmetry_map", {})
    if isinstance(symmetry_map, Mapping):
        normalized_map: dict[str, list[dict[str, Any]]] = {}
        for tag, operations in symmetry_map.items():
            rows: list[dict[str, Any]] = []
            if isinstance(operations, Sequence) and not isinstance(operations, (str, bytes)):
                for operation in operations:
                    if isinstance(operation, Mapping):
                        row = dict(operation)
                        if "name" in row:
                            user_name = str(row["name"])
                            canonical_name = canonical_operation_name_for_valley(user_name, valley_model_out)
                            row.setdefault("user_name", user_name)
                            row["name"] = canonical_name
                            row.update(effective_operation_metadata_for_valley(user_name, valley_model_out))
                        rows.append(row)
                    else:
                        user_name = str(operation)
                        canonical_name = canonical_operation_name_for_valley(user_name, valley_model_out)
                        rows.append(
                            {
                                "user_name": user_name,
                                "name": canonical_name,
                                **effective_operation_metadata_for_valley(user_name, valley_model_out),
                            }
                        )
            normalized_map[str(tag)] = rows
        model_out["symmetry_map"] = normalized_map
    out["model"] = model_out
    return out



def _default_symmetry_source(
    raw: Mapping[str, Any],
    source_raw: Mapping[str, Any],
    *,
    source_base: Path,
) -> dict[str, Any]:
    source = raw.get("symmetry_source", {})
    if source is None:
        source = {}
    if isinstance(source, (str, Path)):
        out: dict[str, Any] = {"path": str(source)}
    elif isinstance(source, Mapping):
        out = dict(source)
    else:
        raise ValueError("symmetry_source must be a path string or mapping when provided")

    if out.get("path") and not out.get("type"):
        out["type"] = "kp_symm_output"
    if not out.get("type") and out.get("source") == "toy":
        out["type"] = "toy_generator"
    if not out.get("type") and not out.get("path"):
        symm = source_raw.get("symm", {})
        if isinstance(symm, Mapping) and symm.get("output_dir"):
            out["type"] = "kp_symm_output"
            out["inferred_from_source_config"] = True
    if out.get("type") == "kp_symm_output" and not out.get("path"):
        symm = source_raw.get("symm", {})
        if isinstance(symm, Mapping) and symm.get("output_dir"):
            resolved = _resolve_path(symm.get("output_dir"), source_base)
            if resolved is not None:
                out["path"] = str(resolved)
                out["inferred_from_source_config"] = True
    if out.get("type") == "kp_symm_output":
        out.setdefault("use", "raw")
        if isinstance(raw.get("valley_model"), Mapping):
            out.setdefault("valley_model", dict(raw["valley_model"]))
        symm = source_raw.get("symm", {})
        if "operations" not in out and isinstance(symm, Mapping) and symm.get("operations"):
            out["operations"] = list(symm.get("operations", []))
    return out



def _source_matrix_file(source_name: str, use: str, matrix_kind: str) -> str:
    if matrix_kind != "action":
        raise ValueError("kp_symm_output production symmetry_source.matrix_kind must be 'action'")
    return f"{source_name}_low_{use}.npy"



def _default_source_operation_name(name: str) -> str:
    if name in {"C3z"}:
        return "C3"
    if name in {"TR"}:
        return "TR"
    return name



def _matrix_file_stem(source_operation: str) -> str:
    return source_operation



def _complete_kp_symm_operation_entry(
    operation: Any,
    valley_model: Mapping[str, Any],
    *,
    use: str,
    matrix_kind: str,
    allow_inferred: bool = False,
) -> dict[str, Any]:
    row = dict(operation) if isinstance(operation, Mapping) else {"name": str(operation)}
    user_name = str(row.get("name", row.get("operation", "")))
    name = canonical_source_operation_name_for_valley(user_name, valley_model)
    source_operation = str(row.get("operation", _default_source_operation_name(name)))
    if matrix_kind != "action":
        raise ValueError("kp_symm_output production symmetry_source.matrix_kind must be 'action'")
    required_fields = (
        "antiunitary",
        "k_map",
        "q_map",
        "sector_map",
        "source_matrix_role",
        "source_gauge",
        "target_role",
        "gauge_correction",
        "antiunitary_convention",
    )
    missing_action = [field for field in required_fields if field not in row]
    if allow_inferred and "q_map" in missing_action and "k_map" in row:
        missing_action.remove("q_map")
    if missing_action:
        raise ValueError(
            f"symmetry_source operation {user_name!r} requires explicit action metadata when matrix_kind is set; "
            f"missing {missing_action}"
        )
    completed = {
        "name": name,
        "operation": source_operation,
        "matrix_file": _source_matrix_file(_matrix_file_stem(source_operation), use, matrix_kind),
        **effective_operation_metadata_for_valley(user_name, valley_model),
    }
    completed.update(row)
    completed["name"] = name
    completed.setdefault("operation", source_operation)
    if allow_inferred and "q_map" not in completed and "k_map" in completed:
        completed["q_map"] = copy.deepcopy(completed["k_map"])
        completed["inferred_fields"] = sorted(set([*completed.get("inferred_fields", []), "q_map"]))
    return completed



def _normalize_kp_symm_source(raw: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    valley_model = out.get("valley_model", {})
    symmetry_source = out.get("symmetry_source", {})
    if not isinstance(valley_model, Mapping) or not isinstance(symmetry_source, Mapping):
        return out
    if str(symmetry_source.get("type", "none")) != "kp_symm_output":
        return out
    source_out = dict(symmetry_source)
    operations = source_out.get("operations")
    if operations is None:
        out["symmetry_source"] = source_out
        return out
    if isinstance(operations, Mapping):
        operations = [
            {**(dict(value) if isinstance(value, Mapping) else {}), "name": str(key)}
            for key, value in operations.items()
        ]
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)):
        raise ValueError("symmetry_source.operations must be a list or mapping when provided")
    use = str(source_out.get("use", "raw"))
    matrix_kind_raw = source_out.get("matrix_kind", source_out.get("kind"))
    allow_inferred = allows_inferred_action_metadata(source_out)
    operation_defaults = {
        key: copy.deepcopy(source_out[key])
        for key in (
            "source_matrix_role",
            "source_gauge",
            "target_role",
            "gauge_correction",
            "antiunitary_convention",
        )
        if key in source_out
    }
    if matrix_kind_raw is None:
        normalized_operations: list[Any] = []
        for operation in operations:
            if isinstance(operation, Mapping):
                row = dict(operation)
                user_name = str(row.get("name", row.get("operation", "")))
                if user_name:
                    row["name"] = canonical_source_operation_name_for_valley(user_name, valley_model)
                    row.update(effective_operation_metadata_for_valley(user_name, valley_model))
                normalized_operations.append(row)
            else:
                user_name = str(operation)
                normalized_operations.append(
                    {
                        "name": canonical_source_operation_name_for_valley(user_name, valley_model),
                        **effective_operation_metadata_for_valley(user_name, valley_model),
                    }
                )
        source_out["operations"] = normalized_operations
    else:
        matrix_kind = str(matrix_kind_raw)
        source_out["operations"] = [
            _complete_kp_symm_operation_entry(
                {**operation_defaults, **(dict(operation) if isinstance(operation, Mapping) else {"name": str(operation)})},
                valley_model,
                use=use,
                matrix_kind=matrix_kind,
                allow_inferred=allow_inferred,
            )
            for operation in operations
        ]
    out["symmetry_source"] = source_out
    return out



def _max_derivative_order_values(
    model: Mapping[str, Any],
    *,
    valley_model: Mapping[str, Any] | None = None,
    n_orb: Sequence[int] | None = None,
) -> dict[str, int]:
    family_orders = {**DEFAULT_MAX_ORDER, **{str(key): int(value) for key, value in dict(model.get("max_order", {})).items()}}
    values = dict(family_orders)
    for semantic_key, family_key in GAMMA_ORDER_FAMILIES.items():
        values.setdefault(semantic_key, int(family_orders.get(family_key, 0)))
    explicit = model.get("max_derivative_order", {})
    if explicit is None:
        explicit = {}
    if not isinstance(explicit, Mapping):
        raise ValueError("model.max_derivative_order must be a mapping when provided")
    for key, value in explicit.items():
        values[str(key)] = int(value)
    return values



def _default_auto_low_energy_order_config(valley_model: Mapping[str, Any] | None) -> dict[str, Any]:
    valley_type = str((valley_model or {}).get("valley_type", "")).strip()
    if valley_type == "Gamma":
        return {
            "profile": "gamma_compact_ladder_start",
            "max_order": {"Kinect": 6, "intra": 4, "inter": 6},
            "max_derivative_order": {
                "moire_intra_zero": 0,
                "moire_intra_nonzero": 4,
                "tunneling_zero": 6,
                "tunneling_nonzero": 6,
            },
        }
    if valley_type == "M":
        return {
            "profile": "m_release_baseline",
            "max_order": {"Kinect": 10, "intra": 4, "inter": 6},
            "max_derivative_order": {},
        }
    if valley_type == "K":
        return {
            "profile": "k_release_baseline",
            "max_order": {"Kinect": 6, "intra": 4, "inter": 4},
            "max_derivative_order": {},
        }
    return {
        "profile": "generic_release_baseline",
        "max_order": {"Kinect": 6, "intra": 4, "inter": 4},
        "max_derivative_order": {},
    }



def _reflect_vector(kvec: np.ndarray, axis_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(axis_deg))
    axis = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    return 2.0 * axis * float(np.dot(axis, kvec)) - np.asarray(kvec, dtype=float)



def _rotate_k_map_to_model_frame(k_map: Any, *, rotation_deg: float) -> Any:
    if not isinstance(k_map, Mapping):
        return k_map
    out = dict(k_map)
    if bool(out.get("in_model_frame", False)):
        return out
    map_type = str(out.get("type", "")).lower()
    if map_type in {"reflection", "mirror", "mirror_axis"} and "axis_deg" in out:
        out["axis_deg"] = float(out["axis_deg"]) + float(rotation_deg)
    return out



def _is_accepted_support_exactification_provenance(provenance: Any) -> bool:
    return (
        isinstance(provenance, Mapping)
        and provenance.get("source") == "support_exactification"
        and bool(provenance.get("accepted_by_user"))
    )



def _symmetry_operation_index(metadata: Mapping[str, Any], *, rotation_deg: float) -> dict[str, dict[str, Any]]:
    operations = metadata.get("operations", [])
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)):
        return {}
    index: dict[str, dict[str, Any]] = {}
    for record in operations:
        if not isinstance(record, Mapping):
            continue
        enriched = dict(record)
        resolved_action = record.get("internal_resolved_action")
        if isinstance(resolved_action, Mapping) and not _is_accepted_support_exactification_provenance(
            resolved_action.get("provenance")
        ):
            resolved_action = None
        if not isinstance(resolved_action, Mapping):
            resolved_action = record.get("model_action")
        if isinstance(resolved_action, Mapping):
            for key in ("antiunitary", "k_map", "q_map", "sector_map"):
                if key in resolved_action:
                    enriched[key] = copy.deepcopy(resolved_action[key])
        else:
            enriched["k_map"] = _rotate_k_map_to_model_frame(record.get("k_map"), rotation_deg=rotation_deg)
            if "q_map" in record:
                enriched["q_map"] = _rotate_k_map_to_model_frame(record.get("q_map"), rotation_deg=rotation_deg)
        keys = [record.get("name"), record.get("operation")]
        for key in keys:
            if key is None:
                continue
            index[str(key)] = enriched
    return index



def _enrich_symmetry_map(
    symmetry_map: Mapping[str, Sequence[Mapping[str, Any]]],
    symmetry_metadata: Mapping[str, Any],
    *,
    rotation_deg: float,
    require_action_metadata: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    op_index = _symmetry_operation_index(symmetry_metadata, rotation_deg=rotation_deg)
    out: dict[str, list[dict[str, Any]]] = {}
    for tag, operations in symmetry_map.items():
        rows: list[dict[str, Any]] = []
        for operation in operations:
            if not isinstance(operation, Mapping):
                rows.append({"name": str(operation)})
                continue
            row = dict(operation)
            name = str(row.get("name", ""))
            if name and name in op_index:
                source_row = dict(op_index[name])
                merged = dict(source_row)
                for key, value in row.items():
                    if key not in {"antiunitary", "k_map", "q_map", "sector_map"}:
                        merged[key] = value
                row = merged
            if require_action_metadata:
                _require_resolved_operation_action(row)
            rows.append(row)
        out[str(tag)] = rows
    return out



def _require_resolved_operation_action(operation: Mapping[str, Any]) -> None:
    name = str(operation.get("name", ""))
    missing = [field for field in ("antiunitary", "k_map", "q_map", "sector_map") if field not in operation]
    if missing:
        raise ValueError(f"Operation {name!r} requires explicit resolved action metadata; missing {missing}")



def _default_kpath_config(raw: Mapping[str, Any], *, source_base: Path) -> dict[str, Any]:
    kpath = raw.get("kpath", {})
    if kpath is None:
        kpath = {}
    if not isinstance(kpath, Mapping):
        raise ValueError("kpath section must be a mapping")
    return dict(kpath)



def _rotation_deg_from_symmetry_manifest(raw: Mapping[str, Any], *, base: Path) -> float | None:
    symmetry_source = raw.get("symmetry_source", {})
    if not isinstance(symmetry_source, Mapping) or str(symmetry_source.get("type", "")) != "kp_symm_output":
        return None
    path_raw = symmetry_source.get("path")
    if path_raw is None:
        return None
    path = Path(str(path_raw))
    if not path.is_absolute():
        path = (base / path).resolve()
    for filename in ("manifest.json", "summary.json"):
        manifest_path = path / filename
        if not manifest_path.exists():
            continue
        manifest = _load_yaml(manifest_path)
        frame = manifest.get("frame", {})
        if not isinstance(frame, Mapping):
            continue
        q_transform = frame.get("q_transform", {})
        if isinstance(q_transform, Mapping) and "rotation_deg" in q_transform:
            return float(q_transform["rotation_deg"])
        k_transform = frame.get("k_transform", {})
        if isinstance(k_transform, Mapping) and "rotation_deg" in k_transform:
            return float(k_transform["rotation_deg"])
    representations = path / "representations.npz"
    if representations.exists():
        with np.load(representations, allow_pickle=False) as payload:
            if "__metadata_json__" in payload.files:
                metadata = json.loads(str(payload["__metadata_json__"].item()))
                if isinstance(metadata, Mapping):
                    frame = metadata.get("frame", {})
                    if isinstance(frame, Mapping):
                        q_transform = frame.get("q_transform", {})
                        if isinstance(q_transform, Mapping) and "rotation_deg" in q_transform:
                            return float(q_transform["rotation_deg"])
                        k_transform = frame.get("k_transform", {})
                        if isinstance(k_transform, Mapping) and "rotation_deg" in k_transform:
                            return float(k_transform["rotation_deg"])
        return None
    return None






def _load_symmetry_manifest(raw: Mapping[str, Any], *, base: Path) -> dict[str, Any]:
    symmetry_source = raw.get("symmetry_source", {})
    if not isinstance(symmetry_source, Mapping) or str(symmetry_source.get("type", "")) != "kp_symm_output":
        return {}
    path_raw = symmetry_source.get("path")
    if path_raw is None:
        return {}
    path = Path(str(path_raw))
    if not path.is_absolute():
        path = (base / path).resolve()
    for filename in ("manifest.json", "summary.json"):
        manifest_path = path / filename
        if manifest_path.exists():
            manifest = _load_yaml(manifest_path)
            return dict(manifest) if isinstance(manifest, Mapping) else {}
    representations = path / "representations.npz"
    if representations.exists():
        with np.load(representations, allow_pickle=False) as payload:
            if "__metadata_json__" in payload.files:
                metadata = json.loads(str(payload["__metadata_json__"].item()))
                return dict(metadata) if isinstance(metadata, Mapping) else {}
            raise ValueError(
                f"packed kp symm artifact lacks metadata: {representations}; rerun `kp symm`"
            )
    return {}



def _load_model_artifact_identity(
    heff_file: str | Path,
    symmetry_source: Mapping[str, Any],
    *,
    base: str | Path | None = None,
) -> dict[str, Any]:
    if str(symmetry_source.get("type", "")) != "kp_symm_output":
        return {}
    source_path = symmetry_source.get("path")
    if source_path in (None, ""):
        raise ValueError("kp model requires symmetry_source.path for kp_symm_output")
    symmetry_dir = Path(str(source_path)).expanduser()
    if not symmetry_dir.is_absolute():
        if base is None:
            raise ValueError("Relative symmetry_source.path requires a config base directory")
        symmetry_dir = Path(base) / symmetry_dir
    symmetry_dir = symmetry_dir.resolve()
    representations = symmetry_dir / "representations.npz"
    if not representations.is_file():
        raise FileNotFoundError(
            f"kp model requires canonical kp symm output: {representations}"
        )
    with np.load(representations, allow_pickle=False) as payload:
        if "__metadata_json__" not in payload.files:
            raise ValueError(f"kp model symmetry pack lacks __metadata_json__: {representations}")
        metadata = json.loads(str(payload["__metadata_json__"].item()))
        matrix_keys = {str(name) for name in payload.files if not str(name).startswith("__")}
    if not isinstance(metadata, Mapping):
        raise ValueError(f"kp model symmetry metadata must be a mapping: {representations}")
    symmetry_identity_raw = metadata.get("artifact_identity")
    if not isinstance(symmetry_identity_raw, Mapping):
        raise ValueError(
            f"kp model symmetry pack lacks artifact_identity; rerun kp symm: {representations}"
        )
    symmetry_identity = require_identity_fields(
        symmetry_identity_raw,
        PROJECTION_ARTIFACT_IDENTITY_FIELDS,
        "kp model symmetry pack",
    )
    heff_path = Path(heff_file).resolve()
    project_dir = heff_path.parent
    canonical_heff = (project_dir / "heff.npy").resolve()
    if heff_path != canonical_heff:
        raise ValueError(f"kp model requires canonical projection/heff.npy, got {heff_path}")
    project_identity = load_projection_artifact_identity(project_dir)
    require_matching_identity(
        project_identity,
        symmetry_identity,
        PROJECTION_ARTIFACT_IDENTITY_FIELDS,
        "kp model projection/symmetry",
    )
    operations = metadata.get("operations", [])
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)):
        raise ValueError(f"kp model symmetry pack operations must be a list: {representations}")
    for operation in operations:
        if not isinstance(operation, Mapping):
            raise ValueError(f"kp model symmetry pack operation metadata must be a mapping: {operation!r}")
        name = str(operation.get("name", operation.get("operation", "")))
        require_matching_identity(
            project_identity,
            operation,
            ("basis_hash", "k_indices_hash", "heff_hash"),
            f"kp model symmetry operation {name!r}",
        )
        if not _operation_matrix_is_exactified(operation):
            raise ValueError(
                f"kp model symmetry operation {name!r} lacks complete exactification provenance"
            )
        matrix_key = str(operation.get("matrix_array_key", name))
        if matrix_key not in matrix_keys:
            raise ValueError(
                f"kp model symmetry operation {name!r} references missing matrix array {matrix_key!r}"
            )
    return project_identity



def _first_symlink_component(path: Path) -> Path | None:
    current = Path(path.anchor)
    for part in path.parts:
        if part == path.anchor:
            continue
        current = current / part
        if current.is_symlink():
            return current
    return None



def _preflight_certified_model_selection(config: ConfiguredModel) -> Any:
    """Require the kp-symm-certified project choice for automatic bases."""

    valley_model = config.valley_model
    valley_type = (
        str(valley_model.get("valley_type", "")).strip().casefold()
        if isinstance(valley_model, Mapping)
        else ""
    )
    source_raw = getattr(config, "source_raw", {})
    source_project = (
        source_raw.get("project", {}) if isinstance(source_raw, Mapping) else {}
    )
    selection = (
        source_project.get("selection")
        if isinstance(source_project, Mapping)
        else None
    )
    selection_mode = (
        selection.get("mode", "") if isinstance(selection, Mapping) else selection
    )
    automatic_non_gamma = (
        valley_type in {"k", "m"}
        and str(selection_mode or "").strip().casefold() == "auto"
    )
    if valley_type != "gamma" and not automatic_non_gamma:
        return None

    project_dir = Path(config.heff_file).resolve().parent
    basis_path = project_dir / "basis.npz"
    generic_artifact_identity: dict[str, str] = {}
    with np.load(basis_path, allow_pickle=False) as basis_payload:
        basis_kind = "explicit_layer"
        if "projection_basis_kind" in basis_payload.files:
            basis_kind_raw = np.asarray(basis_payload["projection_basis_kind"])
            if basis_kind_raw.shape != ():
                raise ValueError("projection basis kind must be a scalar")
            basis_kind = str(basis_kind_raw.item())
        if automatic_non_gamma:
            for field in ("basis_hash", "heff_hash", "k_indices_hash"):
                if field not in basis_payload.files:
                    raise ValueError(
                        f"automatic K/M project basis lacks {field}"
                    )
                raw_value = np.asarray(basis_payload[field])
                if raw_value.shape != ():
                    raise ValueError(
                        f"automatic K/M project basis {field} must be a scalar"
                    )
                generic_artifact_identity[field] = str(raw_value.item())
    if basis_kind == "explicit_layer" and not automatic_non_gamma:
        return None
    allowed_basis_kinds = (
        {"explicit_layer"}
        if automatic_non_gamma
        else {"gamma_common_anchor", "gamma_routed"}
    )
    if basis_kind not in allowed_basis_kinds:
        raise ValueError(f"unsupported projection_basis_kind: {basis_kind!r}")

    marker_path = project_dir / "selection_artifact.json"
    if marker_path.is_symlink():
        raise ValueError("kp model projection selection marker must not be a symlink")
    if not marker_path.is_file():
        raise FileNotFoundError(
            f"kp model requires certified projection selection marker: {marker_path}"
        )
    artifact = SelectionArtifactStore(project_dir).load_current(
        require_certified=True
    )
    if artifact.certification_status is not CertificationStatus.CERTIFIED:
        raise ValueError(
            "kp model projection selection is not CERTIFIED: "
            f"{artifact.certification_status.value}"
        )
    if artifact.identity is None:
        raise ValueError("kp model CERTIFIED projection selection lacks identity")
    selection_identity_hash = artifact.identity.selection_identity_hash

    symmetry_source = config.symmetry_source_config
    if str(symmetry_source.get("type", "")) != "kp_symm_output":
        raise ValueError(
            "kp model certified selection requires a canonical kp_symm_output package"
        )
    source_path = symmetry_source.get("path")
    if source_path in (None, ""):
        raise ValueError(
            "kp model certified selection requires symmetry_source.path"
        )
    symmetry_dir = Path(str(source_path)).expanduser()
    if not symmetry_dir.is_absolute():
        symmetry_dir = Path(config.path).resolve().parent / symmetry_dir
    symlink_component = _first_symlink_component(symmetry_dir)
    if symlink_component is not None:
        raise ValueError(
            "kp model symmetry_source.path component must not be a symlink: "
            f"{symlink_component}"
        )
    symmetry_dir = symmetry_dir.resolve()
    config.symmetry_source_config = {
        **dict(symmetry_source),
        "path": str(symmetry_dir),
    }
    representations = symmetry_dir / "representations.npz"
    if representations.is_symlink():
        raise ValueError("kp model symmetry package must not be a symlink")
    if not representations.is_file():
        raise FileNotFoundError(
            f"kp model requires canonical kp symm output: {representations}"
        )
    with np.load(representations, allow_pickle=False) as payload:
        if "__metadata_json__" not in payload.files:
            raise ValueError(
                f"kp model symmetry pack lacks __metadata_json__: {representations}"
            )
        metadata = json.loads(str(payload["__metadata_json__"].item()))
    if not isinstance(metadata, Mapping):
        raise ValueError(
            f"kp model symmetry metadata must be a mapping: {representations}"
        )
    if metadata.get("selection_identity_hash") != selection_identity_hash:
        raise ValueError(
            "kp model projection/symmetry selection_identity_hash mismatch"
        )
    operations = metadata.get("operations")
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)):
        raise ValueError(
            f"kp model symmetry pack operations must be a list: {representations}"
        )
    for operation in operations:
        if not isinstance(operation, Mapping):
            raise ValueError(
                "kp model symmetry operation metadata must be a mapping"
            )
        if operation.get("selection_identity_hash") != selection_identity_hash:
            name = str(operation.get("name", operation.get("operation", "")))
            raise ValueError(
                "kp model symmetry operation "
                f"{name!r} selection_identity_hash mismatch"
            )

    if automatic_non_gamma:
        heff = np.load(config.heff_file, mmap_mode="r", allow_pickle=False)
        if heff.ndim != 3 or heff.shape[-2] != heff.shape[-1]:
            raise ValueError(
                "automatic K/M projection/heff.npy must be a square matrix stack"
            )
        verify_certified_generic_selection_artifact(
            artifact,
            artifact_identity=generic_artifact_identity,
            candidate_dimension=int(heff.shape[-1]),
        )
    elif basis_kind == "gamma_common_anchor":
        handoff = load_gamma_common_anchor_basis_spec(project_dir / "basis.npz")
        verify_certified_gamma_selection_artifact(artifact, handoff)
    else:
        handoff = load_gamma_routed_basis_spec(project_dir / "basis.npz")
        verify_certified_gamma_selection_artifact(artifact, handoff)
    return artifact



def _apply_project_k_indices(kpoints: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    points = _validate_kpoints(kpoints)
    if not indices:
        return points[:0]
    max_index = max(int(index) for index in indices)
    if max_index >= len(points):
        raise IndexError(
            f"projection k_indices contains {max_index}, but the source k-point path has {len(points)} rows"
        )
    return np.asarray(points[[int(index) for index in indices]], dtype=float)



def _symmetry_operations_for_default_templates(raw: Mapping[str, Any], *, base: Path) -> list[Mapping[str, Any]]:
    operations: list[Mapping[str, Any]] = []
    symmetry_source = raw.get("symmetry_source", {})
    if isinstance(symmetry_source, Mapping):
        raw_operations = symmetry_source.get("operations", [])
        if isinstance(raw_operations, Mapping):
            raw_operations = [
                {**(dict(value) if isinstance(value, Mapping) else {}), "name": str(key)}
                for key, value in raw_operations.items()
            ]
        if isinstance(raw_operations, Sequence) and not isinstance(raw_operations, (str, bytes)):
            operations.extend(dict(item) for item in raw_operations if isinstance(item, Mapping))
    manifest_operations = _load_symmetry_manifest(raw, base=base).get("operations", [])
    if isinstance(manifest_operations, Sequence) and not isinstance(manifest_operations, (str, bytes)):
        operations.extend(dict(item) for item in manifest_operations if isinstance(item, Mapping))
    return operations



def _rotation_deg_from_config(raw: Mapping[str, Any], *, base: Path) -> float:
    if "coordinate_frame" in raw:
        raise ValueError("coordinate_frame is not a supported model input; use the kp_symm manifest frame")
    unsupported_frame_keys: list[str] = []
    if "rotation_deg" in raw:
        unsupported_frame_keys.append("rotation_deg")
    model = raw.get("model", {})
    if isinstance(model, Mapping):
        if "rotation_deg" in model:
            unsupported_frame_keys.append("model.rotation_deg")
        if "q_rotation_deg" in model:
            unsupported_frame_keys.append("model.q_rotation_deg")
    kpath = raw.get("kpath", {})
    if isinstance(kpath, Mapping) and "phase_deg" in kpath:
        unsupported_frame_keys.append("kpath.phase_deg")
    if unsupported_frame_keys:
        labels = ", ".join(unsupported_frame_keys)
        raise ValueError(f"Model frame rotation must come from the kp_symm manifest; remove {labels}")
    manifest_rotation = _rotation_deg_from_symmetry_manifest(raw, base=base)
    if manifest_rotation is not None:
        return manifest_rotation
    symmetry_source = raw.get("symmetry_source", {})
    if isinstance(symmetry_source, Mapping) and str(symmetry_source.get("type", "")) == "toy_generator":
        return 0.0
    raise ValueError(
        "Configured model YAML requires a kp_symm symmetry manifest frame rotation. "
        "Rerun `kp symm` with automatic frame inference; do not hand-write model frame rotations."
    )



def load_model_config(path: str | Path) -> ConfiguredModel:
    from . import fit_sampling
    from . import harmonic_selection

    cfg_path = Path(path).resolve()
    raw = normalize_case_config(_load_yaml(cfg_path), config_path=cfg_path)
    base = cfg_path.parent
    raw = dict(raw)
    fit_defaulted = "fit" not in raw
    if fit_defaulted:
        raw["fit"] = {"mode": "auto_low_energy"}
    validation = raw.get("validation", {})
    if validation is None:
        validation = {}
    if not isinstance(validation, Mapping):
        raise ValueError("validation section must be a mapping when provided")
    raw["validation"] = dict(validation)

    if raw.get("source_config") not in (None, ""):
        raise ValueError("source_config is not supported in release-only KP configs; use one case YAML.")
    if all(isinstance(raw.get(section), Mapping) for section in ("material", "project", "plot")):
        source_config = cfg_path
        source_raw = dict(raw)
    else:
        raise ValueError("Configured model YAML requires inline material/project/plot sections")
    source_base = source_config.parent
    material = source_raw.get("material", {})
    project = source_raw.get("project", {})
    plot = source_raw.get("plot", {})
    if not isinstance(material, Mapping) or not isinstance(project, Mapping) or not isinstance(plot, Mapping):
        raise ValueError("source_config must contain mapping sections: material, project, plot")
    num_layer_list = _num_layer_list_from_material(material)

    qset1_file = _resolve_path(material.get("qset1_file"), source_base)
    qset2_file = _resolve_path(material.get("qset2_file"), source_base)
    if qset1_file is None or qset2_file is None:
        raise ValueError("source_config material must provide qset1_file and qset2_file")

    raw = _normalize_short_model_config(raw)

    heff_file = _resolve_path(_get_path_value(raw, "heff_file"), base)
    if heff_file is None:
        project_out = _resolve_path(project.get("out_dir"), source_base)
        if project_out is None:
            raise ValueError("heff_file is missing and source_config project.out_dir is unavailable")
        heff_file = project_out / "heff.npy"
    heff_eig_file = _resolve_path(_get_path_value(raw, "heff_eig_file"), base)
    if heff_eig_file is not None:
        raise ValueError("heff_eig_file is not supported in release-only KP configs; use projection/heff.npy.")
    try:
        project_k_indices = load_projection_k_indices(heff_file)
    except FileNotFoundError:
        heff_rows = int(np.load(heff_file, mmap_mode="r", allow_pickle=False).shape[0])
        project_k_indices = list(range(heff_rows))

    kpoints_file = _resolve_path(_get_path_value(raw, "kpoints_file"), base)
    kpoints_from_projection = False
    if kpoints_file is None:
        projection_kpoints = heff_file.parent / "kpoints.npy"
        if projection_kpoints.is_file():
            kpoints_file = projection_kpoints
            kpoints_from_projection = True

    output_section = raw.get("output", {})
    if output_section is None:
        output_section = {}
    if not isinstance(output_section, Mapping):
        raise ValueError("output section must be a mapping")
    output_dir = _resolve_path(output_section.get("dir", "model"), base)
    if output_dir is None:
        raise ValueError("Failed to resolve output directory")

    model_raw = _normalize_model_user_aliases(_model_section(raw))
    user_explicit_max_order = "max_order" in model_raw
    user_explicit_max_derivative_order = "max_derivative_order" in model_raw
    valley_model_raw = raw.get("valley_model", {})
    if not isinstance(valley_model_raw, Mapping):
        valley_model_raw = {}
    symmetry_source = _default_symmetry_source(raw, source_raw, source_base=source_base)
    raw["symmetry_source"] = symmetry_source
    kpath_config = _default_kpath_config(raw, source_base=source_base)
    raw["kpath"] = kpath_config
    fit_for_defaults = raw.get("fit", {})
    if not isinstance(fit_for_defaults, Mapping):
        fit_for_defaults = {}
    auto_low_energy_mode = str(fit_for_defaults.get("mode", "")).strip().lower() == "auto_low_energy"
    automatic_defaults_meta: dict[str, bool] = {
        "fit": True,
    } if fit_defaulted else {}
    if "harmonics" not in model_raw:
        automatic_defaults_meta["harmonics"] = True
    if automatic_defaults_meta:
        model_raw["automatic_defaults"] = dict(automatic_defaults_meta)
    if auto_low_energy_mode:
        defaults_meta: dict[str, bool] = {}
        if "max_order" not in model_raw and "max_derivative_order" not in model_raw:
            order_defaults = _default_auto_low_energy_order_config(
                raw.get("valley_model", {}) if isinstance(raw.get("valley_model", {}), Mapping) else {}
            )
            model_raw["max_order"] = dict(order_defaults["max_order"])
            if order_defaults.get("max_derivative_order"):
                model_raw["max_derivative_order"] = dict(order_defaults["max_derivative_order"])
            model_raw["auto_low_energy_order_profile"] = {
                "profile": str(order_defaults["profile"]),
                "source": "internal_default",
            }
            defaults_meta["max_order"] = True
        if defaults_meta:
            model_raw["auto_low_energy_defaults"] = defaults_meta
    if "max_order" not in model_raw:
        model_raw["max_order"] = dict(DEFAULT_MAX_ORDER)
    raw["model"] = model_raw
    raw = _normalize_user_symmetry_names(raw)
    raw = _normalize_kp_symm_source(raw)
    model_raw = _normalize_model_user_aliases(_model_section(raw))
    if "symmetry_map" not in model_raw:
        valley_model_normalized = raw.get("valley_model", {})
        symmetry_source_normalized = raw.get("symmetry_source", {})
        if not _operation_rows_from_symmetry_source(
            symmetry_source_normalized if isinstance(symmetry_source_normalized, Mapping) else {},
            valley_model_normalized if isinstance(valley_model_normalized, Mapping) else {},
        ):
            manifest_operations = _symmetry_operations_for_default_templates(raw, base=base)
            if manifest_operations:
                symmetry_source_normalized = {
                    **(dict(symmetry_source_normalized) if isinstance(symmetry_source_normalized, Mapping) else {}),
                    "operations": [dict(item) for item in manifest_operations if isinstance(item, Mapping)],
                }
        model_raw["symmetry_map"] = _default_symmetry_map_from_source_or_valley(
            symmetry_source_normalized if isinstance(symmetry_source_normalized, Mapping) else {},
            valley_model_normalized if isinstance(valley_model_normalized, Mapping) else {},
        )
        raw["model"] = model_raw
        raw = _normalize_user_symmetry_names(raw)
    rotation_deg = _rotation_deg_from_config(raw, base=base)

    model = _model_section(raw)
    model_raw = _normalize_model_user_aliases(model)
    valley_type_for_counts = str(
        (
            raw.get("valley_model", {})
            if isinstance(raw.get("valley_model", {}), Mapping)
            else {}
        ).get("valley_type", "")
    )
    prefer_active_layer_sectors = valley_type_for_counts in {"K", "M"}
    projection_n_orb_input, projection_n_orb_source = _projection_layerwise_orbital_counts(
        heff_file=heff_file,
        project=project,
        num_layer_list=num_layer_list,
    )
    explicit_n_orb = "n_orb" in model or "n_orb1" in model or "n_orb2" in model
    if "n_orb" in model:
        n_orb_raw = model["n_orb"]
    elif "n_orb1" in model or "n_orb2" in model:
        n_orb_raw = [model.get("n_orb1", 2), model.get("n_orb2", 2)]
    elif projection_n_orb_input is not None:
        n_orb_raw = projection_n_orb_input
    else:
        raise ValueError(
            "model.n_orb was omitted, but projection orbital metadata is unavailable; "
            "rerun kp project to create projection/basis.npz or set model.n_orb explicitly "
            "for an incomplete projection artifact"
        )
    n_orb_input = _as_int_list(n_orb_raw, name="model.n_orb")
    projection_uses_qset_counts = projection_n_orb_source in {
        "projection_gamma_common_anchor",
        "projection_gamma_routed",
    }
    if not explicit_n_orb and projection_uses_qset_counts:
        n_orb_values, n_orb_resolution = _resolve_projection_qset_counts(
            n_orb_input,
            num_layer_list=num_layer_list,
        )
    else:
        n_orb_values, n_orb_resolution = _resolve_layerwise_counts(
            n_orb_input,
            name="model.n_orb",
            num_layer_list=num_layer_list,
            prefer_active_layer_sectors=prefer_active_layer_sectors,
        )
    if projection_n_orb_input is not None:
        if projection_uses_qset_counts:
            projection_n_orb_values, _ = _resolve_projection_qset_counts(
                projection_n_orb_input,
                num_layer_list=num_layer_list,
            )
            projection_conflict = n_orb_values != projection_n_orb_values
        else:
            projection_n_orb_values, _ = _resolve_layerwise_counts(
                projection_n_orb_input,
                name="projection nlow_state_list",
                num_layer_list=num_layer_list,
                prefer_active_layer_sectors=prefer_active_layer_sectors,
            )
            projection_conflict = (
                n_orb_input != projection_n_orb_input
                or n_orb_values != projection_n_orb_values
            )
        if explicit_n_orb and projection_conflict:
            raise ValueError(
                "model.n_orb conflicts with projection orbital counts: "
                f"configured {n_orb_input}, projection {projection_n_orb_input}"
            )
        n_orb_resolution["source"] = projection_n_orb_source
    elif explicit_n_orb:
        n_orb_resolution["source"] = "model_config"
    nlow_state_default_from_n_orb = "nlow_state" not in model
    nlow_state_input = _as_int_list(model.get("nlow_state", n_orb_input), name="model.nlow_state")
    nlow_state, nlow_state_resolution = _resolve_layerwise_counts(
        nlow_state_input,
        name="model.nlow_state",
        num_layer_list=num_layer_list,
        default_from="model.n_orb" if nlow_state_default_from_n_orb else None,
        prefer_active_layer_sectors=prefer_active_layer_sectors,
    )
    if len(n_orb_values) != 2:
        raise ValueError(f"model.n_orb must resolve to two qset counts, got {n_orb_values}")
    if len(nlow_state) != 2:
        raise ValueError(f"model.nlow_state must resolve to two qset counts, got {nlow_state}")
    model_raw["n_orb"] = list(n_orb_values)
    model_raw["nlow_state"] = list(nlow_state)
    model_raw["n_orb_resolution"] = n_orb_resolution
    model_raw["nlow_state_resolution"] = nlow_state_resolution
    if "groups" in n_orb_resolution:
        model_raw["n_orb_layerwise"] = n_orb_resolution["raw"]
        model_raw["n_orb_resolved_qset"] = n_orb_resolution["resolved_qset"]
    if "groups" in nlow_state_resolution:
        model_raw["nlow_state_layerwise"] = nlow_state_resolution["raw"]
        model_raw["nlow_state_resolved_qset"] = nlow_state_resolution["resolved_qset"]
    raw["model"] = model_raw
    model = _model_section(raw)
    orbital_count_metadata = {
        "num_layer_list": [int(value) for value in num_layer_list],
        "total_layers": int(sum(int(value) for value in num_layer_list)),
        "n_orb": n_orb_resolution,
        "nlow_state": nlow_state_resolution,
    }
    max_order_values = _max_derivative_order_values(
        model,
        valley_model=raw.get("valley_model", {}) if isinstance(raw.get("valley_model", {}), Mapping) else {},
        n_orb=n_orb_values,
    )

    fit = raw.get("fit", {})
    obsolete_shell_keys = {"shell_subspace_weight", "shell_subspace_max_shells", "shell_subspace_fraction", "shell_subspace_decay"}.intersection(fit)
    if obsolete_shell_keys:
        raise ValueError(f"unsupported fit keys: {sorted(obsolete_shell_keys)}")
    if not isinstance(fit, Mapping):
        raise ValueError("fit section must be a mapping")
    heff_shape = tuple(int(value) for value in np.load(heff_file, mmap_mode="r").shape)
    fit_method_config = _canonical_public_fit_method(fit, heff_shape=heff_shape)
    if fit_method_config:
        if (
            not user_explicit_max_order
            or not isinstance(model.get("max_order"), Mapping)
            or not model.get("max_order")
        ):
            raise ValueError("fit.method requires explicit model.max_order")
        if model.get("profiles") is not None:
            raise ValueError("fit.method cannot be combined with model.profiles")
        if user_explicit_max_derivative_order:
            raise ValueError(
                "fit.method uses model.max_order; remove advanced model.max_derivative_order"
            )
    requested_profiles = _parse_requested_model_profiles(model.get("profiles"))
    fit_weighting = _parse_fit_weighting(fit.get("weighting"))
    model_selection_config = (
        None
        if fit_method_config
        else parse_model_selection_config(
            fit.get("model_selection"),
            maximum_orders=FamilyOrders(
                kinetic=int(max_order_values.get("Kinect", 0)),
                intra=int(max_order_values.get("intra", 0)),
                inter=int(max_order_values.get("inter", 0)),
            ),
        )
    )
    plot_target = str(plot.get("target", "valence")).strip().lower()
    default_target_bands = (
        "bottom"
        if plot_target in {"conduction", "cbm", "bottom"}
        else "top"
    )
    target_bands = str(
        model.get(
            "target_bands",
            raw.get("target_bands", default_target_bands),
        )
    ).strip().lower()
    if target_bands not in {"top", "bottom"}:
        raise ValueError("model.target_bands must be 'top' or 'bottom'")
    model_raw["target_bands"] = target_bands
    response_semantics = str(model.get("response_semantics", "complete_linear")).strip()
    if response_semantics != "complete_linear":
        raise ValueError(
            "model.response_semantics must be 'complete_linear'; "
            f"unsupported value {response_semantics!r}"
        )
    model_raw["response_semantics"] = response_semantics
    raw["model"] = model_raw

    auto_low_energy_refine_indices: list[int] | None = None
    if fit_method_config:
        fit_indices = [int(value) for value in fit_method_config["hamiltonian_kpoints"]]
        fit_selection_metadata = {
            "mode": "manual",
            "source": "fit.kpoints",
            "selected_indices": list(fit_indices),
        }
    elif fit.get("indices") is not None:
        fit_indices = _as_int_list(fit.get("indices", []), name="fit.indices")
        fit_selection_metadata = {
            "mode": "manual",
            "source": "fit.indices",
            "selected_indices": [int(index) for index in fit_indices],
        }
    elif fit.get("mode") is not None and str(fit.get("mode")).strip().lower() in {"auto", "auto_compact"}:
        fit_kpoints_all = _load_kpoints_from_inputs(
            kpoints_file=kpoints_file,
            kpath_config=kpath_config,
            base=base,
            rotation_deg=rotation_deg,
            kpoints_from_projection=kpoints_from_projection,
        )
        if not kpoints_from_projection:
            fit_kpoints_all = _apply_project_k_indices(fit_kpoints_all, project_k_indices)
        fit_indices, fit_selection_metadata = _select_auto_fit_indices(fit_kpoints_all, fit)
    elif fit.get("mode") is not None and str(fit.get("mode")).strip().lower() == "auto_low_energy":
        fit_kpoints_all = _load_kpoints_from_inputs(
            kpoints_file=kpoints_file,
            kpath_config=kpath_config,
            base=base,
            rotation_deg=rotation_deg,
            kpoints_from_projection=kpoints_from_projection,
        )
        if not kpoints_from_projection:
            fit_kpoints_all = _apply_project_k_indices(fit_kpoints_all, project_k_indices)
        max_points = int(fit.get("max_points", 2))
        initial_points = int(fit.get("initial_points", 2))
        initial_fit_indices = (
            _as_int_list(fit.get("initial_fit_indices"), name="fit.initial_fit_indices")
            if fit.get("initial_fit_indices") is not None
            else None
        )
        fit_indices, fit_selection_metadata = fit_sampling._select_adaptive_fit_indices(
            fit_kpoints_all,
            initial_points=initial_points,
            max_points=max_points,
            initial_indices=initial_fit_indices,
        )
        explicit_refine_indices = fit.get("refine_indices", fit.get("refinement_indices"))
        if explicit_refine_indices is not None:
            auto_low_energy_refine_indices = _as_int_list(
                explicit_refine_indices,
                name="fit.refine_indices",
            )
        else:
            auto_low_energy_refine_indices = fit_sampling._auto_low_energy_refinement_indices(
                fit_kpoints_all,
                base_indices=fit_indices,
                max_points=int(fit.get("max_refine_points", fit.get("refine_max_points", 8))),
            )
        fit_selection_metadata["refine_indices"] = [int(index) for index in auto_low_energy_refine_indices]
    else:
        fit_indices = []
        fit_selection_metadata = {
            "mode": "manual",
            "source": "missing_fit_indices",
            "selected_indices": [],
        }

    bands = raw.get("bands", {})
    if bands is None:
        bands = {}
    if not isinstance(bands, Mapping):
        raise ValueError("bands section must be a mapping")
    band_indices_raw = bands.get("indices")
    band_indices = None if band_indices_raw is None else _as_int_list(band_indices_raw, name="bands.indices")
    band_slice = None if bands.get("band_slice") is None else _as_int_list(bands.get("band_slice"), name="bands.band_slice")
    band_plot = bands.get("plot", {})
    if band_plot is None:
        band_plot = {}
    if not isinstance(band_plot, Mapping):
        raise ValueError("bands.plot section must be a mapping when provided")

    harmonic_selection_report: dict[str, Any] | None = None
    _, harmonic_selection_cfg = harmonic_selection._harmonic_selection_options(fit)
    harmonics_defaulted = bool(model.get("automatic_defaults", {}).get("harmonics", False))
    if harmonics_defaulted:
        thresholds_raw = harmonic_selection_cfg.get("thresholds", {})
        if thresholds_raw is None:
            thresholds_raw = {}
        if not isinstance(thresholds_raw, Mapping):
            raise ValueError("fit.harmonic_selection.thresholds must be a mapping when provided")
        threshold_values = harmonic_selection._harmonic_selection_threshold_values(thresholds_raw)
        Q_set1_for_selection, Q_set2_for_selection = load_Q_sets_from_gvec_files(
            qset1_file,
            qset2_file,
            rotation_deg=rotation_deg,
        )
        heff_for_selection = _select_rows(np.load(heff_file, mmap_mode="r"), band_indices)
        sample_indices = harmonic_selection._evenly_spaced_sample_indices(
            int(heff_for_selection.shape[0]),
            int(harmonic_selection_cfg.get("sample_kpoints", 7)),
        )
        if sample_indices is None:
            heff_scan = np.asarray(heff_for_selection)
        else:
            heff_scan = np.asarray(heff_for_selection)[np.asarray(sample_indices, dtype=int)]
        n_primary_for_selection = int(sum(n_orb_values))
        dim_for_selection = int(heff_for_selection.shape[-1])
        plot_bands_for_selection = int(
            harmonic_selection_cfg.get(
                "plot_bands",
                fit.get("weighted_fit_bands", min(dim_for_selection, max(n_primary_for_selection, 10))),
            )
        )
        configured_harmonic_search = bool(harmonic_selection_cfg)
        max_shell_for_selection = (
            int(harmonic_selection_cfg.get("max_shell", 5))
            if configured_harmonic_search
            else None
        )
        candidate_pairs_for_selection = (
            harmonic_selection._harmonic_candidate_pairs_from_config(
                harmonic_selection_cfg.get("candidate_pairs"),
                max_shell=int(max_shell_for_selection),
                search=str(harmonic_selection_cfg.get("search", "ladder")),
            )
            if configured_harmonic_search
            else None
        )
        harmonic_selection_report = harmonic_selection._run_harmonic_ablation_selection(
            heff_scan,
            Q_set1_for_selection,
            Q_set2_for_selection,
            n_orb=(int(n_orb_values[0]), int(n_orb_values[1])),
            target_bands=target_bands,
            primary_bands=n_primary_for_selection,
            plot_bands=plot_bands_for_selection,
            max_shell=max_shell_for_selection,
            thresholds=threshold_values,
            candidate_pairs=candidate_pairs_for_selection,
            tol=float(harmonic_selection_cfg.get("tol", 1.0e-6)),
        )
        harmonic_selection_report["sample_indices"] = sample_indices
        quality_selected = harmonic_selection._high_accuracy_harmonic_recommendation_candidate(harmonic_selection_report)
        selected_harmonics = harmonic_selection._low_cost_harmonic_recommendation_candidate(harmonic_selection_report)
        if selected_harmonics is None:
            raise ValueError(
                "automatic low-cost harmonic selection found no candidate with "
                "plot RMS <= 1 meV and mean subspace overlap > 0.9; "
                "set model.harmonics explicitly or enlarge fit.harmonic_selection.max_shell"
            )
        harmonic_selection_report["quality_selected"] = quality_selected
        harmonic_selection_report["selected"] = dict(selected_harmonics)
        harmonic_selection_report["selection_status"] = "selected_low_cost_relaxed_accuracy"
        harmonic_selection_report["selection_profile"] = "low_cost"
        model_raw["harmonics"] = {
            "intra": {"count": int(selected_harmonics["intra_shells"])},
            "inter": {"count": int(selected_harmonics["inter_shells"])},
        }
        raw["model"] = model_raw
        model = _model_section(raw)

    harmonics = model.get("harmonics", {})
    if not isinstance(harmonics, Mapping):
        raise ValueError("model.harmonics must be a mapping")
    harmonic_count_limits = _harmonic_count_limits(harmonics)
    symmetry_map = model.get("symmetry_map", {})
    if not isinstance(symmetry_map, Mapping):
        raise ValueError("model.symmetry_map must be a mapping")

    validate_model_config(raw, nlow_state=nlow_state)

    valley_model = raw.get("valley_model", {})
    symmetry_source = raw.get("symmetry_source", {})
    sectors = raw.get("sectors", [])
    if sectors is None:
        sectors = []
    if not isinstance(sectors, Sequence) or isinstance(sectors, (str, bytes)):
        raise ValueError("sectors must be a list when provided")
    if not sectors:
        sectors = _infer_k_sectors_from_layerwise_counts(
            n_orb_resolution,
            valley_model=valley_model if isinstance(valley_model, Mapping) else {},
        )
    _validate_sector_orbital_counts(sectors, n_orb_values)
    term_templates = model.get("term_templates", raw.get("term_templates"))
    term_template_metadata: dict[str, Any] = {"input_kind": "explicit"}
    if term_templates is None or term_templates == []:
        valley_type = str(valley_model.get("valley_type", ""))
        if valley_type not in _CASE_DERIVED_GENERATORS:
            raise ValueError(f"no complete response vocabulary for valley_type={valley_type!r}")
        term_templates = []
        term_template_metadata = {
            "input_kind": "default", "valley_type": valley_type,
            "n_orb": [int(n_orb_values[0]), int(n_orb_values[1])],
            "profiles": [], "generator": _CASE_DERIVED_GENERATORS[valley_type],
            "term_space_policies": ["complete"],
        }
    if not isinstance(term_templates, Sequence) or isinstance(term_templates, (str, bytes)):
        raise ValueError("term_templates must be a list when provided")
    term_templates = [dict(item) for item in term_templates]
    if response_semantics == "complete_linear":
        invalid_policies = [
            str(item.get("name", item.get("source", "term")))
            for item in term_templates
            if item.get("term_space_policy")
            not in {"complete", "explicit_reduced", "orbit_representative"}
        ]
        if invalid_policies:
            raise ValueError(
                "complete_linear requires term_space_policy: complete, explicit_reduced, "
                "or orbit_representative "
                f"for every template; missing/invalid templates: {invalid_policies}"
            )

    symmetry_source_metadata = {}
    if isinstance(symmetry_source, Mapping) and bool(symmetry_source.get("inferred_from_source_config", False)):
        symmetry_source_metadata["inferred_from_source_config"] = True

    fit_mode = str(fit.get("mode", "")).strip().lower()
    coeff_prune_default = 1.0e-4 if fit_mode == "auto_low_energy" else 0.0
    coeff_prune_threshold = float(fit.get("coeff_prune_threshold", coeff_prune_default))
    if coeff_prune_threshold < 0.0:
        raise ValueError(f"fit.coeff_prune_threshold must be non-negative, got {coeff_prune_threshold}")
    null_channel_abs_tol = float(fit.get("null_channel_abs_tol", 0.0))
    if null_channel_abs_tol < 0.0:
        raise ValueError(f"fit.null_channel_abs_tol must be non-negative, got {null_channel_abs_tol}")
    null_channel_rel_tol = float(fit.get("null_channel_rel_tol", 0.0))
    if null_channel_rel_tol < 0.0:
        raise ValueError(f"fit.null_channel_rel_tol must be non-negative, got {null_channel_rel_tol}")
    response_regularization = float(fit.get("regularization", 0.0))
    if response_regularization < 0.0:
        raise ValueError(f"fit.regularization must be non-negative, got {response_regularization}")
    response_fit_objective = _canonical_response_fit_objective(
        fit.get("objective"),
        response_semantics=response_semantics,
        target_bands=target_bands,
        default_min_bands=max(1, int(sum(n_orb_values))),
    )
    if fit_method_config:
        response_fit_objective = {
            "mode": "normalized_low_energy_linear",
            "target_reference": "current_heff_support_mask",
            "edge": target_bands,
            "window": {
                "mode": "fixed_count_degeneracy_safe",
                "bands": int(fit_method_config["requested_bands"]),
                "degeneracy_tol_mev": 0.1,
            },
            "one_sided_weight": float(fit_method_config["one_sided_weight"]),
            "two_sided_weight": float(fit_method_config["two_sided_weight"]),
            "normalization": "dimension_mean_square",
        }
    if fit_weighting == "uniform":
        if response_fit_objective.get("mode") != "equal_matrix":
            raise ValueError(
                "fit.weighting='uniform' conflicts with a weighted fit.objective; "
                "remove fit.objective or use equal_matrix"
            )
        response_fit_objective = {"mode": "equal_matrix"}
    output_profile = str(output_section.get("profile", "release")).strip().lower()
    if output_profile in {"", "default"}:
        output_profile = "release"
    if output_profile not in {"release", "debug"}:
        raise ValueError(f"output.profile must be 'release' or 'debug', got {output_profile!r}")
    output_section = dict(output_section)
    output_section["profile"] = output_profile

    refine_raw = fit.get("refine_bands", {})
    if refine_raw is True:
        refine_config = {"enabled": True}
    elif refine_raw in (False, None):
        refine_config = {"enabled": False}
    elif isinstance(refine_raw, Mapping):
        refine_config = dict(refine_raw)
    else:
        raise ValueError("fit.refine_bands must be a mapping or boolean when provided")
    if str(fit.get("mode", "")).strip().lower() == "auto_low_energy":
        heff_shape = tuple(int(value) for value in np.load(heff_file, mmap_mode="r").shape)
        if len(heff_shape) != 3 or heff_shape[-1] != heff_shape[-2]:
            raise ValueError(f"auto_low_energy requires Heff with shape (Nk,dim,dim), got {heff_shape}")
        dim_for_windows = int(heff_shape[-1])
        n_primary = int(sum(n_orb_values))
        gap_tolerance_mev = float(fit.get("gap_tolerance_mev", 0.1))
        primary_band_slice = fit_sampling._auto_low_energy_band_slice(
            dim_for_windows,
            n_primary,
            target_bands,
        )
        windows = {
            "primary": {
                "role": "primary",
                "n_bands": n_primary,
                "band_slice": list(primary_band_slice),
                "boundary_gap_mev": None,
                "state": "pending_current_support_target",
                "use_for_loss": True,
                "reference": "pending_current_heff_support_mask",
            },
            "expanded": [],
            "reference": "pending_current_heff_support_mask",
        }
        weighted_fit_bands = int(fit.get("weighted_fit_bands", fit.get("fit_bands", min(dim_for_windows, max(n_primary, 10)))))
        if weighted_fit_bands < n_primary:
            raise ValueError(
                "fit.weighted_fit_bands must be at least the primary low-energy dimension "
                f"sum(model.n_orb)={n_primary}, got {weighted_fit_bands}"
            )
        if weighted_fit_bands > dim_for_windows:
            raise ValueError(
                f"fit.weighted_fit_bands={weighted_fit_bands} exceeds Hamiltonian dimension {dim_for_windows}"
            )
        weighted_band_slice = fit_sampling._auto_low_energy_band_slice(
            dim_for_windows,
            weighted_fit_bands,
            target_bands,
        )
        weighted_fit_window = {
            "role": "weighted_fit",
            "n_bands": weighted_fit_bands,
            "band_slice": list(weighted_band_slice),
            "boundary_gap_mev": None,
            "state": "pending_current_support_target",
            "use_for_loss": True,
            "reference": "pending_current_heff_support_mask",
        }
        windows["weighted_fit"] = weighted_fit_window
        primary_band_slice = list(primary_band_slice)
        weighted_band_slice = list(weighted_band_slice)
        auto_refine_defaults = {
            "enabled": True,
            "mode": "auto_low_energy",
            "target_bands": target_bands,
            "auto_windows": windows,
            "auto_harmonic_selection": {"enabled": False},
            "band_slice": list(weighted_fit_window["band_slice"]),
            "align": target_bands,
            "solver": str(fit.get("refinement_solver", "linear_low_subspace")),
            "indices": [int(index) for index in (auto_low_energy_refine_indices or fit_indices)],
            "variable_tags": ["Kinect", "Onsite", "intra"],
            "components": ["real"],
            "refinement_candidates": [
                {
                    "name": "A_intra_real",
                    "variable_tags": ["Kinect", "Onsite", "intra"],
                    "components": ["real"],
                    "role": "default_compact",
                },
                {
                    "name": "B_add_inter_real",
                    "variable_tags": ["Kinect", "Onsite", "intra", "inter"],
                    "components": ["real"],
                    "role": "accept_only_if_pareto_better",
                },
                {
                    "name": "C_add_imag",
                    "variable_tags": ["Kinect", "Onsite", "intra", "inter"],
                    "components": ["real", "imag"],
                    "role": "accept_only_if_pareto_better",
                },
            ],
            "regularization": float(fit.get("linear_regularization", fit.get("regularization", 0.3))),
            "band_sigma_mev": float(fit.get("band_sigma_mev", 1.0)),
            "normalize_band_loss": True,
            "weighted_band_loss": {
                "enabled": True,
                "primary_bands": int(n_primary),
                "fit_bands": int(weighted_fit_bands),
                "decay": float(fit.get("band_weight_decay", 0.75)),
                "floor": float(fit.get("band_weight_floor", 0.45)),
                "normalize_mean": True,
            },
            "coefficient_weight": float(fit.get("coefficient_weight", 0.02)),
            "max_nfev": int(fit.get("max_nfev", 8)),
            "use_fit_kpoints": bool(fit.get("use_fit_kpoints", False)),
            "max_variables": int(fit.get("max_variables", 900)),
            "subspace_loss": {
                "enabled": bool(fit.get("subspace_loss_enabled", False)),
                "mode": "principal_angles",
                "weight": float(fit.get("subspace_weight", 2.0)),
                "band_slice": weighted_band_slice,
                "normalize": True,
                "gap_tolerance_mev": gap_tolerance_mev,
            },
            "shell_projected_matrix_loss": {
                "enabled": True,
                "weight": float(fit.get("shell_projected_matrix_weight", 1.0)),
                "max_shells": int(fit.get("shell_projected_matrix_max_shells", 3)),
                "window": {
                    "mode": "fixed_fraction",
                    "center_fraction": float(
                        fit.get("shell_projected_matrix_fraction", 0.5)
                    ),
                    "gap_tolerance_mev": float(fit.get("shell_projected_matrix_gap_tolerance_mev", gap_tolerance_mev)),
                },
                "shell_decay": float(fit.get("shell_projected_matrix_shell_decay", 0.75)),
                            },
            "low_subspace_matrix_loss": {
                "enabled": bool(fit.get("low_subspace_matrix_loss_enabled", False)),
                "weight": float(fit.get("low_subspace_matrix_weight", 0.25)),
                "sigma_mev": float(fit.get("low_subspace_matrix_sigma_mev", 10.0)),
                "band_slice": weighted_band_slice,
                "normalize": True,
                "gap_tolerance_mev": gap_tolerance_mev,
            },
            "matrix_loss": {
                "enabled": True,
                "mode": "block_normalized",
                "weight": float(fit.get("raw_matrix_weight", 0.0)),
                "sigma_mev": float(fit.get("raw_matrix_sigma_mev", 10.0)),
                "blocks": {"kinetic_diagonal": 1.0, "intralayer": 1.0, "interlayer": 0.5},
            },
            "acceptance_guard": {
                "enabled": True,
                "profile": "low_energy",
                "selection": "best_validation_window",
                "line_search_alphas": list(fit.get("acceptance_line_search_alphas", [0.5])),
                "guard_all_bands": bool(fit.get("acceptance_guard_all_bands", False)),
                "max_rms_increase_mev": float(fit.get("acceptance_max_rms_increase_mev", 999.0)),
                "max_max_increase_mev": float(fit.get("acceptance_max_max_increase_mev", 999.0)),
                "max_all_band_rms_increase_mev": float(fit.get("acceptance_max_all_band_rms_increase_mev", 3.0)),
                "max_all_band_max_increase_mev": float(fit.get("acceptance_max_all_band_max_increase_mev", 5.0)),
            },
        }
        if refine_config.get("enabled", False):
            merged = dict(auto_refine_defaults)
            merged.update(refine_config)
            refine_config = merged
        else:
            refine_config = auto_refine_defaults
    if fit_weighting == "uniform":
        weighted_band_loss = dict(refine_config.get("weighted_band_loss", {}))
        weighted_band_loss.update(
            {
                "enabled": False,
                "reason": "fit.weighting=uniform",
            }
        )
        refine_config["weighted_band_loss"] = weighted_band_loss
    if requested_profiles and not any(
        profile.startswith("nonlinear/") for profile in requested_profiles
    ):
        refine_config["enabled"] = False
    if fit_method_config:
        if fit_method_config["method"] == "linear":
            refine_config = {"enabled": False}
        else:
            refine_config = {
                "enabled": True,
                "mode": "public_nonlinear",
                "reference": "current_heff_support_mask",
                "hamiltonian_kpoints": list(
                    fit_method_config["hamiltonian_kpoints"]
                ),
                "band_kpoints": list(fit_method_config["band_kpoints"] or []),
                "target_bands": target_bands,
                "bands": int(fit_method_config["requested_bands"]),
                "one_sided_weight": float(fit_method_config["one_sided_weight"]),
                "two_sided_weight": float(fit_method_config["two_sided_weight"]),
                "band_loss_weight": float(fit_method_config["band_loss_weight"]),
                "max_steps": int(fit_method_config["max_steps"]),
                **({"degeneracy_tol_ev": float(fit_method_config["degeneracy_tol_ev"])}
                   if "degeneracy_tol_ev" in fit_method_config else {}),
            }

    return ConfiguredModel(
        path=cfg_path,
        raw=dict(raw),
        source_config=source_config,
        source_raw=dict(source_raw),
        qset1_file=qset1_file,
        qset2_file=qset2_file,
        kpoints_file=kpoints_file,
        kpoints_from_projection=kpoints_from_projection,
        heff_file=heff_file,
        heff_eig_file=heff_eig_file,
        project_k_indices=project_k_indices,
        output_dir=output_dir,
        rotation_deg=rotation_deg,
        fit_indices=fit_indices,
        fit_selection_metadata=fit_selection_metadata,
        band_indices=band_indices,
        n_orb=(int(n_orb_values[0]), int(n_orb_values[1])),
        nlow_state=nlow_state,
        bM_config=dict(model.get("bM", {})),
        harmonics_config=dict(harmonics),
        max_order=max_order_values,
        symmetry_map={str(key): list(value) for key, value in dict(symmetry_map).items()},
        coeff_tol=float(fit.get("coeff_tol", 1.0e-6)),
        null_channel_abs_tol=null_channel_abs_tol,
        null_channel_rel_tol=null_channel_rel_tol,
        coeff_prune_threshold=coeff_prune_threshold,
        compare_to_heff=bool(bands.get("compare_to_heff", True)),
        response_semantics=response_semantics,
        response_regularization=response_regularization,
        response_fit_objective=response_fit_objective,
        kpath_config=dict(kpath_config),
        valley_model=dict(valley_model),
        symmetry_source_config=dict(symmetry_source) if isinstance(symmetry_source, Mapping) else {},
        symmetry_source_metadata=symmetry_source_metadata,
        sectors_config=[dict(item) for item in sectors],
        term_templates=[dict(item) for item in term_templates],
        term_template_metadata=term_template_metadata,
        output_config=dict(output_section),
        band_slice=band_slice,
        band_plot_config=dict(band_plot),
        automatic_harmonic_selection=(
            dict(harmonic_selection_report) if harmonic_selection_report is not None else {}
        ),
        orbital_count_metadata=orbital_count_metadata,
        validation_config=dict(raw.get("validation", {})),
        band_refinement_config=refine_config,
        model_selection_config=model_selection_config,
        requested_profiles=requested_profiles,
        fit_weighting=fit_weighting,
        fit_method_config=fit_method_config,
    )



def evaluate_vector_expression(value: Any, variables: Mapping[str, Any]) -> np.ndarray:
    """Evaluate a small vector expression such as ``bM1 + bM2`` or ``rot(q1, 240)``."""
    if isinstance(value, np.ndarray):
        return _as_vector(value)
    if isinstance(value, (list, tuple)):
        return _as_vector(np.array(value, dtype=float))
    if isinstance(value, Mapping):
        if "expr" not in value:
            raise ValueError(f"Vector mapping must contain expr, got {value!r}")
        value = value["expr"]
    if not isinstance(value, str):
        raise ValueError(f"Unsupported vector expression {value!r}")
    expr = ast.parse(value, mode="eval")
    result = _eval_ast(expr.body, variables)
    return _as_vector(result)



def _as_vector(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2,):
        raise ValueError(f"Expected a 2-vector, got shape {arr.shape}: {value!r}")
    return arr



def _as_scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.shape != ():
        raise ValueError(f"Expected scalar, got shape {arr.shape}: {value!r}")
    return float(arr)



def _eval_ast(node: ast.AST, variables: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(f"Unknown vector expression symbol: {node.id}")
        return variables[node.id]
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant in vector expression: {node.value!r}")
    if isinstance(node, ast.List):
        return np.array([_as_scalar(_eval_ast(elt, variables)) for elt in node.elts], dtype=float)
    if isinstance(node, ast.Tuple):
        return np.array([_as_scalar(_eval_ast(elt, variables)) for elt in node.elts], dtype=float)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_ast(node.operand, variables)
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left, variables)
        right = _eval_ast(node.right, variables)
        if isinstance(node.op, ast.Add):
            return np.asarray(left) + np.asarray(right)
        if isinstance(node.op, ast.Sub):
            return np.asarray(left) - np.asarray(right)
        if isinstance(node.op, ast.Mult):
            return np.asarray(left) * np.asarray(right)
        if isinstance(node.op, ast.Div):
            return np.asarray(left) / np.asarray(right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "rot" and len(node.args) == 2:
            vec = evaluate_vector_expression_ast(node.args[0], variables)
            angle = _as_scalar(_eval_ast(node.args[1], variables))
            return rot(vec, angle)
        if node.func.id == "norm" and len(node.args) == 1:
            vec = evaluate_vector_expression_ast(node.args[0], variables)
            return float(np.linalg.norm(vec))
    raise ValueError(f"Unsupported vector expression syntax: {ast.dump(node)}")



def evaluate_vector_expression_ast(node: ast.AST, variables: Mapping[str, Any]) -> np.ndarray:
    return _as_vector(_eval_ast(node, variables))



def _harmonic_filter(kind: str, indices: Sequence[int], *, sign: float = 1.0) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": kind, "indices": [int(index) for index in indices]}
    if sign != 1.0:
        out["sign"] = float(sign)
    return out



def _harmonic_count_limits(harmonics: Mapping[str, Any]) -> dict[str, int]:
    limits: dict[str, int] = {}
    for kind in ("intra", "inter"):
        raw = harmonics.get(kind)
        if _is_auto_harmonic_spec(raw):
            limits[kind] = _auto_harmonic_count(raw, name=f"model.harmonics.{kind}")
        elif isinstance(raw, Mapping):
            limits[kind] = max((int(key) for key in raw.keys()), default=0)
    return limits


def _term_template_row(
    name: str,
    source: str,
    sector_pairs: Sequence[Sequence[int]],
    orbital_pairs: Any,
    *,
    max_order: int | None = None,
    max_order_from: str | None = None,
    harmonics: Any | None = None,
    harmonic_filter: Mapping[str, Any] | None = None,
    monomial_constraints: Mapping[str, Any] | None = None,
    term_space_policy: str = "explicit_reduced",
) -> dict[str, Any]:
    if max_order is not None and max_order_from is not None:
        raise ValueError(f"{name}: use either max_order or max_order_from, not both")
    if harmonics is not None and harmonic_filter is not None:
        raise ValueError(f"{name}: use either harmonics or harmonic_filter, not both")

    def _sector_ref(value: Any) -> int | str:
        if isinstance(value, (int, np.integer)):
            return int(value)
        return str(value)

    row: dict[str, Any] = {
        "name": name,
        "source": source,
        "sector_pairs": [[_sector_ref(i), _sector_ref(j)] for i, j in sector_pairs],
        "orbital_pairs": copy.deepcopy(orbital_pairs),
        "term_space_policy": str(term_space_policy),
    }
    if max_order is not None:
        row["max_order"] = int(max_order)
    if max_order_from is not None:
        row["max_order_from"] = str(max_order_from)
    if harmonics is not None:
        row["harmonics"] = copy.deepcopy(harmonics)
    if harmonic_filter is not None:
        row["harmonic_filter"] = copy.deepcopy(dict(harmonic_filter))
    if monomial_constraints is not None:
        row["monomial_constraints"] = copy.deepcopy(dict(monomial_constraints))
    return row


_CASE_DERIVED_GENERATORS = {
    "Gamma": "gamma_case_derived",
    "K": "k_case_derived",
    "M": "m_case_derived",
}



_GAMMA_CASE_DERIVED_GENERATOR = _CASE_DERIVED_GENERATORS["Gamma"]


def _term_name_fragment(value: Any) -> str:
    text = str(value).strip()
    out = "".join(ch if ch.isalnum() else "_" for ch in text)
    out = "_".join(part for part in out.split("_") if part)
    return out or "sector"



def _sector_qset_slot(sector: Mapping[str, Any]) -> int:
    qset = str(sector.get("qset", ""))
    if qset == "qset1":
        return 1
    if qset == "qset2":
        return 2
    raise ValueError(f"Unsupported sector qset {qset!r}; expected qset1 or qset2")



def _active_template_sectors(
    sectors: Sequence[Mapping[str, Any]] | None,
    n_orb: tuple[int, int],
) -> list[dict[str, Any]]:
    if not sectors:
        return []
    active: list[dict[str, Any]] = []
    seen_qsets: set[str] = set()
    for raw in sectors:
        if not isinstance(raw, Mapping):
            continue
        sector = dict(raw)
        qset = str(sector.get("qset", ""))
        slot = _sector_qset_slot(sector)
        if int(n_orb[slot - 1]) <= 0:
            continue
        if qset in seen_qsets:
            raise ValueError(
                "Default sector-aware term generation currently expects at most one active sector per qset; "
                "use explicit term_templates for multiple sectors sharing a qset."
            )
        seen_qsets.add(qset)
        sector.setdefault("name", f"L{slot}")
        active.append(sector)
    return active


def _template_inter_harmonic_vectors(
    template: Mapping[str, Any],
    inter_harmonics_map: Mapping[int, np.ndarray],
) -> list[np.ndarray]:
    raw = template.get("harmonics", "inter")
    sign = float(template.get("harmonic_sign", 1.0))
    indices = None
    if isinstance(raw, Mapping):
        sign = float(raw.get("sign", sign))
        indices = raw.get("indices")
        raw = raw.get("kind", "inter")
    if str(raw) != "inter":
        return []
    items = sorted(inter_harmonics_map.items()) if indices is None else [
        (int(index), inter_harmonics_map[int(index)])
        for index in indices
        if int(index) in inter_harmonics_map
    ]
    return [sign * np.asarray(vector, dtype=float) for _index, vector in items]



def _raw_q_support_count_for_sector_pair(
    *,
    sector_from: Mapping[str, Any],
    sector_to: Mapping[str, Any],
    harmonic_vectors: Sequence[np.ndarray],
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    tol: float = 1.0e-5,
) -> int:
    q_from = sector_qset(sector_from, Q_set1, Q_set2)
    q_to = sector_qset(sector_to, Q_set1, Q_set2)
    count = 0
    for p_vector in harmonic_vectors:
        diff = q_from[:, None, :] - np.asarray(p_vector, dtype=float) - q_to[None, :, :]
        count += int(np.count_nonzero(np.linalg.norm(diff, axis=2) < tol))
    return count



def _orient_k_sector_inter_templates_by_raw_support(
    templates: Sequence[Mapping[str, Any]],
    *,
    valley_model: Mapping[str, Any],
    n_orb: tuple[int, int],
    sectors: Sequence[Mapping[str, Any]],
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    inter_harmonics_map: Mapping[int, np.ndarray],
    term_template_metadata: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out = [dict(row) for row in templates]
    if str(valley_model.get("valley_type", "")) != "K":
        return out, []
    if str(term_template_metadata.get("input_kind", "")) != "default":
        return out, []
    if "k_sector_aware" not in list(term_template_metadata.get("profiles", [])):
        return out, []
    active = _active_template_sectors(sectors, n_orb)
    candidate_pairs = [
        (sector_from, sector_to)
        for sector_from in active
        for sector_to in active
        if str(sector_from.get("qset")) != str(sector_to.get("qset"))
    ]
    if not candidate_pairs:
        return out, []

    by_name = {str(sector.get("name")): sector for sector in active}
    diagnostics: list[dict[str, Any]] = []
    for row in out:
        if str(row.get("source", "")) != "tunneling":
            continue
        harmonic_vectors = _template_inter_harmonic_vectors(row, inter_harmonics_map)
        if not harmonic_vectors:
            continue
        scores = [
            (
                _raw_q_support_count_for_sector_pair(
                    sector_from=sector_from,
                    sector_to=sector_to,
                    harmonic_vectors=harmonic_vectors,
                    Q_set1=Q_set1,
                    Q_set2=Q_set2,
                ),
                str(sector_from["name"]),
                str(sector_to["name"]),
            )
            for sector_from, sector_to in candidate_pairs
        ]
        current_pairs = row.get("sector_pairs", [])
        if not isinstance(current_pairs, Sequence) or isinstance(current_pairs, (str, bytes)):
            current_pairs = []
        current_support = 0
        for pair in current_pairs:
            if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
                continue
            sector_from = by_name.get(str(pair[0]))
            sector_to = by_name.get(str(pair[1]))
            if sector_from is None or sector_to is None:
                continue
            current_support += _raw_q_support_count_for_sector_pair(
                sector_from=sector_from,
                sector_to=sector_to,
                harmonic_vectors=harmonic_vectors,
                Q_set1=Q_set1,
                Q_set2=Q_set2,
            )
        best_support = max((score[0] for score in scores), default=0)
        best_pairs = [[from_name, to_name] for support, from_name, to_name in scores if support == best_support and support > 0]
        changed = bool(best_pairs) and best_support > current_support
        if changed:
            row["sector_pairs"] = best_pairs
            if len(best_pairs) == 1:
                row["name"] = (
                    f"inter_{_term_name_fragment(best_pairs[0][0])}"
                    f"_to_{_term_name_fragment(best_pairs[0][1])}"
                )
        diagnostics.append(
            {
                "template": str(row.get("name", "inter")),
                "current_support": int(current_support),
                "best_support": int(best_support),
                "scores": [
                    {"sector_pair": [from_name, to_name], "support": int(support)}
                    for support, from_name, to_name in scores
                ],
                "selected_sector_pairs": copy.deepcopy(row.get("sector_pairs", [])),
                "changed": changed,
            }
        )
    return out, diagnostics


def _build_variables(Q_set1: np.ndarray, bM1: np.ndarray, bM2: np.ndarray, model: Mapping[str, Any]) -> dict[str, Any]:
    variables: dict[str, Any] = {
        "zero": np.zeros(2, dtype=float),
        "bM1": np.asarray(bM1, dtype=float),
        "bM2": np.asarray(bM2, dtype=float),
        "sqrt3": float(np.sqrt(3.0)),
        "pi": float(np.pi),
        "bM_norm": float(np.linalg.norm(bM1)),
    }
    variables["q1"] = variables["bM_norm"] * np.array([0.0, 1.0 / np.sqrt(3.0)], dtype=float)
    variables["q3"] = rot(variables["q1"], 240.0)

    custom = model.get("vectors", {})
    if custom is None:
        return variables
    if not isinstance(custom, Mapping):
        raise ValueError("model.vectors must be a mapping")
    for key, expr in custom.items():
        variables[str(key)] = evaluate_vector_expression(expr, variables)
    return variables



def _harmonics_map(raw: Mapping[str, Any], variables: Mapping[str, Any]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for key, value in raw.items():
        out[int(key)] = evaluate_vector_expression(value, variables)
    return out



def _is_auto_harmonic_spec(raw: Any) -> bool:
    if isinstance(raw, int):
        return True
    if isinstance(raw, Mapping):
        return bool(raw.get("auto", False)) or "count" in raw
    return False



def _auto_harmonic_count(raw: Any, *, name: str) -> int:
    if isinstance(raw, int):
        count = raw
    elif isinstance(raw, Mapping):
        if "count" not in raw:
            raise ValueError(f"{name} auto harmonics requires count")
        count = int(raw["count"])
    else:
        raise ValueError(f"{name} auto harmonics must be an integer or mapping, got {raw!r}")
    if count < 0:
        raise ValueError(f"{name} harmonic count must be non-negative, got {count}")
    return count



def _c3_orbit_vectors(vector: np.ndarray) -> list[np.ndarray]:
    base = [np.asarray(vector, dtype=float), rot(vector, 120.0), rot(vector, 240.0)]
    return base + [-v for v in base]



def _vector_key(vector: np.ndarray, tol: float) -> tuple[int, int]:
    return tuple(np.rint(np.asarray(vector, dtype=float) / tol).astype(int).tolist())



def _unique_vector_records(vectors: list[np.ndarray], *, tol: float, bM1: np.ndarray, bM2: np.ndarray) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for vector in vectors:
        arr = np.asarray(vector, dtype=float)
        key = _vector_key(arr, tol)
        entry = grouped.setdefault(key, {"sum": np.zeros(2, dtype=float), "pair_count": 0})
        entry["sum"] += arr
        entry["pair_count"] += 1

    basis = np.column_stack([bM1, bM2])
    records: list[dict[str, Any]] = []
    for entry in grouped.values():
        vector = entry["sum"] / float(entry["pair_count"])
        lattice_index = None
        try:
            coeff = np.linalg.solve(basis, vector)
            rounded = np.rint(coeff).astype(int)
            if np.linalg.norm(vector - basis @ rounded) <= 10.0 * tol:
                lattice_index = [int(rounded[0]), int(rounded[1])]
        except np.linalg.LinAlgError:
            lattice_index = None
        records.append(
            {
                "vector": vector,
                "norm": float(np.linalg.norm(vector)),
                "pair_count": int(entry["pair_count"]),
                "lattice_index": lattice_index,
            }
        )
    return records



def _default_sectors(config: ConfiguredModel) -> list[dict[str, Any]]:
    if config.sectors_config:
        return [dict(item) for item in config.sectors_config]
    return [
        {"name": "L1", "qset": "qset1", "n_orb": config.n_orb[0]},
        {"name": "L2", "qset": "qset2", "n_orb": config.n_orb[1]},
    ]



def _validate_sector_orbital_counts(
    sectors: Sequence[Mapping[str, Any]],
    n_orb_values: Sequence[int],
) -> None:
    expected_by_qset = {"qset1": int(n_orb_values[0]), "qset2": int(n_orb_values[1])}
    for sector in sectors:
        qset_name = str(sector.get("qset", "qset1"))
        if qset_name not in expected_by_qset or "n_orb" not in sector:
            continue
        actual = int(sector["n_orb"])
        expected = expected_by_qset[qset_name]
        if actual != expected:
            name = str(sector.get("name", qset_name))
            raise ValueError(
                f"sectors.{name}.n_orb={actual} is inconsistent with model.n_orb for {qset_name} "
                f"({expected})"
            )



def _sector_offset(sector: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(sector.get("q_offset", [0.0, 0.0]), dtype=float)



_HERMITIAN_HARMONIC_ACTION: dict[str, Any] = {"name": "HermitianPair", "type": "negation"}



def _orbit_member_indices(
    records: list[dict[str, Any]],
    seed: np.ndarray,
    *,
    tol: float,
    actions: Sequence[Mapping[str, Any]] = (_HERMITIAN_HARMONIC_ACTION,),
) -> list[int]:
    targets = _harmonic_orbit_vectors(seed, actions, tol=20.0 * tol)
    indices: list[int] = []
    for idx, record in enumerate(records):
        vector = record["vector"]
        if any(np.linalg.norm(vector - target) <= 20.0 * tol for target in targets):
            indices.append(idx)
    return indices



def _representative_score(vector: np.ndarray) -> tuple[float, float, float]:
    theta = float(np.mod(np.arctan2(vector[1], vector[0]), 2.0 * np.pi))
    sector = np.pi / 3.0
    reduced = float(np.mod(theta, sector))
    wedge_distance = min(reduced, sector - reduced)
    circular_angle = min(theta, 2.0 * np.pi - theta)
    if wedge_distance < 1.0e-8:
        wedge_distance = 0.0
    if circular_angle < 1.0e-8:
        circular_angle = 0.0
    return (
        round(wedge_distance, 12),
        round(circular_angle, 12),
        round(theta, 12),
        round(float(np.linalg.norm(vector)), 12),
    )



def _select_star_representative(records: list[dict[str, Any]], indices: list[int]) -> int:
    return min(indices, key=lambda idx: _representative_score(records[idx]["vector"]))



def _candidate_vectors(kind: str, Q_set1: np.ndarray, Q_set2: np.ndarray) -> list[np.ndarray]:
    if kind == "intra":
        return [Q[i] - Q[j] for Q in (Q_set1, Q_set2) for i in range(len(Q)) for j in range(len(Q))]
    if kind == "inter":
        return [Q_set2[j] - Q_set1[i] for i in range(len(Q_set1)) for j in range(len(Q_set2))]
    raise ValueError(f"Unsupported harmonic kind: {kind}")



def _support_count_for_sector_pair(
    *,
    sector_from: Mapping[str, Any],
    sector_to: Mapping[str, Any],
    p_vector: np.ndarray,
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    tol: float,
) -> int:
    q_from = sector_qset(sector_from, Q_set1, Q_set2)
    q_to = sector_qset(sector_to, Q_set1, Q_set2)
    off_from = _sector_offset(sector_from)
    off_to = _sector_offset(sector_to)
    count = 0
    shifted_to = q_to + off_to
    for q in q_from + off_from:
        delta = np.linalg.norm(q - p_vector - shifted_to, axis=1)
        count += int(np.count_nonzero(delta <= tol))
    return count



def _model_q_pair_support_count_for_sector_pair(
    *,
    sector_from: Mapping[str, Any],
    sector_to: Mapping[str, Any],
    p_vector: np.ndarray,
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    tol: float,
) -> int:
    """Count support in the exact Q convention consumed by the model core."""

    q_from = sector_qset(sector_from, Q_set1, Q_set2)
    q_to = sector_qset(sector_to, Q_set1, Q_set2)
    count = 0
    for q in q_from:
        delta = np.linalg.norm(q - p_vector - q_to, axis=1)
        count += int(np.count_nonzero(delta <= tol))
    return count



def _signed_case_harmonic_records(
    mapping: Mapping[int, np.ndarray],
    *,
    kind: str,
    tol: float,
) -> list[dict[str, Any]]:
    """Return one authored representative for each ``{p, -p}`` orbit.

    Every response seed is passed through the unconditional Hermitian
    projector, which supplies the reversed ``-p`` matrix support.  Authoring
    both signs here therefore doubles the symbolic/rank workload without
    enlarging the physical response span.  Keep the missing sign as explicit
    provenance so the downstream adjoint certificate remains auditable.
    """

    grouped: dict[tuple[tuple[int, int], tuple[int, int]], dict[str, Any]] = {}
    for raw_id, raw_vector in sorted(mapping.items(), key=lambda item: int(item[0])):
        vector = np.asarray(raw_vector, dtype=float)
        if vector.shape != (2,) or not np.all(np.isfinite(vector)):
            raise ValueError(
                f"Case-derived {kind} harmonic {raw_id!r} must be a finite two-vector"
            )
        direct_key = _vector_key(vector, tol)
        adjoint_key = _vector_key(-vector, tol)
        orbit_key = tuple(sorted((direct_key, adjoint_key)))
        record = grouped.setdefault(
            orbit_key,
            {
                "kind": str(kind),
                "vector": vector.copy(),
                "source_harmonic_id": int(raw_id),
                "source_orbit_harmonic_ids": [],
            },
        )
        source_ids = record["source_orbit_harmonic_ids"]
        if int(raw_id) not in source_ids:
            source_ids.append(int(raw_id))
    ordered = sorted(
        grouped.values(),
        key=lambda record: (
            round(float(np.linalg.norm(record["vector"])) / tol),
            round(float(record["vector"][0]) / tol),
            round(float(record["vector"][1]) / tol),
            int(record["source_harmonic_id"]),
        ),
    )
    records: list[dict[str, Any]] = []
    for index, record in enumerate(ordered, start=1):
        harmonic_id = f"case:{kind}:{index}"
        vector = np.asarray(record["vector"], dtype=float)
        adjoint = -vector
        orbit_vectors = [vector.tolist()]
        if not np.allclose(adjoint, vector, rtol=0.0, atol=tol):
            orbit_vectors.append(adjoint.tolist())
        records.append(
            {
                **record,
                "id": harmonic_id,
                "adjoint_harmonic_id": harmonic_id,
                "adjoint_generation": "hermitian_projection",
                "adjoint_vector": adjoint.tolist(),
                "source_orbit_vectors": orbit_vectors,
                "source": "case_q_pair_support",
            }
        )
    return records



def _case_derived_term_templates(
    *,
    valley_type: str,
    sectors: Sequence[Mapping[str, Any]],
    n_orb: tuple[int, int],
    max_order: Mapping[str, int],
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    intra_harmonics_map: Mapping[int, np.ndarray],
    inter_harmonics_map: Mapping[int, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valley_type = str(valley_type)
    generator = _CASE_DERIVED_GENERATORS.get(valley_type)
    if generator is None:
        raise ValueError(
            f"Case-derived response generation does not support valley_type={valley_type!r}"
        )
    name_prefix = f"{valley_type.lower()}_case"
    active = _active_template_sectors(sectors, n_orb)
    if not active:
        raise ValueError(
            f"{valley_type} case-derived response space requires at least one active sector"
        )
    required_orders = {
        "Kinect": int(max_order.get("Kinect", 0)),
        "intra": int(max_order.get("intra", 0)),
        "inter": int(max_order.get("inter", 0)),
    }
    if any(value < 0 for value in required_orders.values()):
        raise ValueError(
            f"{valley_type} case-derived response orders must be non-negative, got {required_orders}"
        )

    q_scale = max(
        float(np.max(np.linalg.norm(np.asarray(Q_set1, dtype=float), axis=1)))
        if np.asarray(Q_set1).size
        else 0.0,
        float(np.max(np.linalg.norm(np.asarray(Q_set2, dtype=float), axis=1)))
        if np.asarray(Q_set2).size
        else 0.0,
        1.0,
    )
    tol = max(1.0e-10, q_scale * 1.0e-8)
    harmonic_candidates = {
        "intra": _signed_case_harmonic_records(
            intra_harmonics_map,
            kind="intra",
            tol=tol,
        ),
        "inter": _signed_case_harmonic_records(
            inter_harmonics_map,
            kind="inter",
            tol=tol,
        ),
    }
    templates: list[dict[str, Any]] = []
    included_harmonics: list[dict[str, Any]] = []
    omitted_harmonics: list[dict[str, Any]] = []

    for sector in active:
        sector_name = str(sector["name"])
        pair = [[sector_name, sector_name]]
        templates.extend(
            [
                _term_template_row(
                    f"{name_prefix}_kinetic",
                    "diagonal_kp",
                    pair,
                    "all",
                    max_order=required_orders["Kinect"],
                    term_space_policy="complete",
                ),
                _term_template_row(
                    f"{name_prefix}_onsite",
                    "onsite",
                    pair,
                    "all",
                    max_order=0,
                    term_space_policy="complete",
                ),
            ]
        )
        supported: list[dict[str, Any]] = []
        for record in harmonic_candidates["intra"]:
            support_count = _model_q_pair_support_count_for_sector_pair(
                sector_from=sector,
                sector_to=sector,
                p_vector=np.asarray(record["vector"], dtype=float),
                Q_set1=Q_set1,
                Q_set2=Q_set2,
                tol=tol,
            )
            diagnostic = {
                **record,
                "vector": np.asarray(record["vector"], dtype=float).tolist(),
                "sector_pair": [sector_name, sector_name],
            }
            if support_count <= 0:
                omitted_harmonics.append(
                    {**diagnostic, "reason": "no_q_pair_support"}
                )
                continue
            literal = {**record, "support_count": int(support_count)}
            supported.append(literal)
            included_harmonics.append(
                {**diagnostic, "support_count": int(support_count)}
            )
        if supported:
            row = _term_template_row(
                f"{name_prefix}_intra",
                "moire_potential",
                pair,
                "all",
                max_order=required_orders["intra"],
                term_space_policy="complete",
            )
            row["harmonic_records"] = supported
            templates.append(row)

    for sector_from in active:
        for sector_to in active:
            if str(sector_from["qset"]) == str(sector_to["qset"]):
                continue
            from_name = str(sector_from["name"])
            to_name = str(sector_to["name"])
            supported = []
            for record in harmonic_candidates["inter"]:
                support_count = _model_q_pair_support_count_for_sector_pair(
                    sector_from=sector_from,
                    sector_to=sector_to,
                    p_vector=np.asarray(record["vector"], dtype=float),
                    Q_set1=Q_set1,
                    Q_set2=Q_set2,
                    tol=tol,
                )
                diagnostic = {
                    **record,
                    "vector": np.asarray(record["vector"], dtype=float).tolist(),
                    "sector_pair": [from_name, to_name],
                }
                if support_count <= 0:
                    omitted_harmonics.append(
                        {**diagnostic, "reason": "no_q_pair_support"}
                    )
                    continue
                literal = {**record, "support_count": int(support_count)}
                supported.append(literal)
                included_harmonics.append(
                    {**diagnostic, "support_count": int(support_count)}
                )
            if supported:
                pair_label = "_".join(
                    sorted(
                        (
                            _term_name_fragment(from_name),
                            _term_name_fragment(to_name),
                        )
                    )
                )
                row = _term_template_row(
                    f"{name_prefix}_inter_{pair_label}",
                    "tunneling",
                    [[from_name, to_name]],
                    "all",
                    max_order=required_orders["inter"],
                    term_space_policy="complete",
                )
                row["harmonic_records"] = supported
                templates.append(row)

    return templates, {
        "generator": generator,
        "valley_type": valley_type,
        "generation_status": "materialized",
        "n_orb": [int(n_orb[0]), int(n_orb[1])],
        "active_sectors": [
            {
                "name": str(sector["name"]),
                "qset": str(sector["qset"]),
                "n_orb": int(n_orb[_sector_qset_slot(sector) - 1]),
            }
            for sector in active
        ],
        "orders": required_orders,
        "tolerance": float(tol),
        "included_harmonics": included_harmonics,
        "omitted_harmonics": omitted_harmonics,
        "template_count": int(len(templates)),
    }



def _case_derived_gamma_term_templates(
    *,
    sectors: Sequence[Mapping[str, Any]],
    n_orb: tuple[int, int],
    max_order: Mapping[str, int],
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    intra_harmonics_map: Mapping[int, np.ndarray],
    inter_harmonics_map: Mapping[int, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _case_derived_term_templates(
        valley_type="Gamma",
        sectors=sectors,
        n_orb=n_orb,
        max_order=max_order,
        Q_set1=Q_set1,
        Q_set2=Q_set2,
        intra_harmonics_map=intra_harmonics_map,
        inter_harmonics_map=inter_harmonics_map,
    )



def _harmonic_action_key(action: Mapping[str, Any]) -> tuple[Any, ...]:
    from . import artifacts

    return (
        str(action.get("type", "")),
        round(float(action.get("angle_deg", 0.0)), 12),
        round(float(action.get("axis_deg", 0.0)), 12),
        json.dumps(artifacts._json_safe(action.get("sector_map")), sort_keys=True),
    )



def _named_operation_harmonic_action(name: Any) -> dict[str, Any] | None:
    operation_name = str(name)
    if operation_name in {"C3z", "C3"}:
        return {"name": "C3z", "type": "rotation", "angle_deg": 120.0}
    if operation_name in {"TR", "C2", "C2z"}:
        return {"name": operation_name, "type": "negation"}
    return None



def _operation_to_harmonic_action(operation: Mapping[str, Any]) -> dict[str, Any] | None:
    q_map = operation.get("q_map", operation.get("k_map"))
    if not isinstance(q_map, Mapping):
        for key in ("name", "operation"):
            if key in operation:
                action = _named_operation_harmonic_action(operation[key])
                if action is not None:
                    return action
        return None
    map_type = str(q_map.get("type", "")).lower()
    if map_type not in {"identity", "negation", "rotation", "reflection", "mirror"}:
        return None
    action = {"name": str(operation.get("name", map_type)), "type": map_type}
    for key in ("angle_deg", "axis_deg"):
        if key in q_map:
            action[key] = float(q_map[key])
    if "sector_map" in operation:
        action["sector_map"] = copy.deepcopy(operation["sector_map"])
    return action



def _harmonic_actions_from_operations(operations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = [dict(_HERMITIAN_HARMONIC_ACTION)]
    seen = {_harmonic_action_key(actions[0])}
    for operation in operations:
        if not isinstance(operation, Mapping):
            continue
        action = _operation_to_harmonic_action(operation)
        if action is None:
            continue
        key = _harmonic_action_key(action)
        if key not in seen:
            actions.append(action)
            seen.add(key)
    return actions



def _sector_pair_specs(raw: Any, kind: str, sectors: Sequence[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    by_name = {str(sector["name"]): sector for sector in sectors}
    raw_pairs = raw.get("sector_pairs") if isinstance(raw, Mapping) else None
    if raw_pairs is not None:
        pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for pair in raw_pairs:
            if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
                raise ValueError(f"{kind} sector_pairs entries must be [from, to], got {pair!r}")
            pairs.append((by_name[str(pair[0])], by_name[str(pair[1])]))
        return pairs
    if kind == "intra":
        return [(sector, sector) for sector in sectors]
    if kind == "inter":
        if len(sectors) < 2:
            raise ValueError("inter auto harmonics require at least two sectors")
        return [(sectors[1], sectors[0])]
    raise ValueError(f"Unsupported harmonic kind: {kind}")



def _support_harmonic_records(
    *,
    raw: Any,
    kind: str,
    sectors: Sequence[Mapping[str, Any]],
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    bM1: np.ndarray,
    bM2: np.ndarray,
    tol: float,
) -> list[dict[str, Any]]:
    basis = np.column_stack([bM1, bM2])
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for sector_from, sector_to in _sector_pair_specs(raw, kind, sectors):
        q_from = sector_qset(sector_from, Q_set1, Q_set2) + _sector_offset(sector_from)
        q_to = sector_qset(sector_to, Q_set1, Q_set2) + _sector_offset(sector_to)
        pair_label = [str(sector_from["name"]), str(sector_to["name"])]
        for qf in q_from:
            for qt in q_to:
                vector = np.asarray(qf - qt, dtype=float)
                key = _vector_key(vector, tol)
                entry = grouped.setdefault(
                    key,
                    {"sum": np.zeros(2, dtype=float), "pair_count": 0, "support_count": 0, "sector_pairs": []},
                )
                entry["sum"] += vector
                entry["pair_count"] += 1
                entry["support_count"] += _support_count_for_sector_pair(
                    sector_from=sector_from,
                    sector_to=sector_to,
                    p_vector=vector,
                    Q_set1=Q_set1,
                    Q_set2=Q_set2,
                    tol=tol,
                )
                if pair_label not in entry["sector_pairs"]:
                    entry["sector_pairs"].append(pair_label)

    records: list[dict[str, Any]] = []
    for entry in grouped.values():
        vector = entry["sum"] / float(entry["pair_count"])
        lattice_index = None
        shell = None
        try:
            coeff = np.linalg.solve(basis, vector)
            rounded = np.rint(coeff).astype(int)
            if np.linalg.norm(vector - basis @ rounded) <= 10.0 * tol:
                lattice_index = [int(rounded[0]), int(rounded[1])]
                shell = int(max(abs(rounded[0]), abs(rounded[1]), abs(rounded[0] + rounded[1])))
        except np.linalg.LinAlgError:
            lattice_index = None
            shell = None
        records.append(
            {
                "vector": vector,
                "norm": float(np.linalg.norm(vector)),
                "pair_count": int(entry["pair_count"]),
                "support_count": int(entry["support_count"]),
                "lattice_index": lattice_index,
                "shell": shell,
                "sector_pairs": entry["sector_pairs"],
            }
        )
    records.sort(key=lambda item: (round(float(item["norm"]) / tol), _representative_score(item["vector"])))
    return records



def _harmonic_orbit_record_indices(
    records: Sequence[Mapping[str, Any]],
    seed: np.ndarray,
    *,
    actions: Sequence[Mapping[str, Any]],
    tol: float,
) -> list[int]:
    targets = _harmonic_orbit_vectors(seed, actions, tol=20.0 * tol)
    return [
        idx
        for idx, record in enumerate(records)
        if any(np.linalg.norm(np.asarray(record["vector"], dtype=float) - target) <= 20.0 * tol for target in targets)
    ]



def _auto_harmonics_from_support(
    *,
    raw: Any,
    kind: str,
    count: int,
    sectors: Sequence[Mapping[str, Any]],
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    bM1: np.ndarray,
    bM2: np.ndarray,
    symmetry_operations: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    if count == 0:
        return {}, {"kind": kind, "selected": [], "count": 0}
    b_norm = max(float(np.linalg.norm(bM1)), float(np.linalg.norm(bM2)), 1.0)
    tol = max(1.0e-8, b_norm * 1.0e-6)
    records = _support_harmonic_records(
        raw=raw,
        kind=kind,
        sectors=sectors,
        Q_set1=Q_set1,
        Q_set2=Q_set2,
        bM1=bM1,
        bM2=bM2,
        tol=tol,
    )
    actions = _harmonic_actions_from_operations(symmetry_operations)
    assigned: set[int] = set()
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        if idx in assigned:
            continue
        member_indices = _harmonic_orbit_record_indices(records, record["vector"], actions=actions, tol=tol)
        assigned.update(member_indices)
        rep_idx = _select_star_representative(records, member_indices)
        rep = dict(records[rep_idx])
        rep["member_count"] = int(len(member_indices))
        rep["orbit_vectors"] = [np.asarray(records[i]["vector"], dtype=float) for i in member_indices]
        rep["pair_count"] = int(sum(records[i]["pair_count"] for i in member_indices))
        rep["support_count"] = int(sum(records[i]["support_count"] for i in member_indices))
        rep["orbit_generators"] = [str(action["name"]) for action in actions]
        candidates.append(rep)
        for member_idx in member_indices:
            if member_idx != rep_idx:
                rejected.append(
                    {
                        "vector": np.asarray(records[member_idx]["vector"], dtype=float).tolist(),
                        "reason": "duplicate_symmetry_orbit",
                        "representative": np.asarray(rep["vector"], dtype=float).tolist(),
                        "orbit_generators": list(rep["orbit_generators"]),
                    }
                )
    candidates.sort(key=lambda item: (round(float(item["norm"]) / tol), _representative_score(item["vector"])))
    if count > len(candidates):
        raise ValueError(f"Requested {count} {kind} harmonics but only found {len(candidates)} support orbits")

    selected = candidates[:count]
    harmonic_map = {idx + 1: np.asarray(item["vector"], dtype=float) for idx, item in enumerate(selected)}
    selected_rows = []
    for idx, item in enumerate(selected, start=1):
        selected_rows.append(
            {
                "index": idx,
                "vector": np.asarray(item["vector"], dtype=float).tolist(),
                "norm": float(item["norm"]),
                "pair_count": int(item["pair_count"]),
                "member_count": int(item["member_count"]),
                "orbit_size": int(item["member_count"]),
                "orbit_generators": list(item["orbit_generators"]),
                "orbit_vectors": [np.asarray(vector, dtype=float).tolist() for vector in item["orbit_vectors"]],
                "lattice_index": item["lattice_index"],
                "shell": item["shell"],
                "sector_pair": item["sector_pairs"][0] if len(item["sector_pairs"]) == 1 else None,
                "sector_pairs": item["sector_pairs"],
                "support_count": int(item["support_count"]),
            }
        )
    return harmonic_map, {
        "kind": kind,
        "count": count,
        "generation": "qset_support_symmetry_orbit",
        "selection_rule": "qset_support_plus_operation_action_orbits",
        "orbit_generators": [str(action["name"]) for action in actions],
        "tolerance": tol,
        "selected": selected_rows,
        "rejected": rejected,
    }



def _auto_harmonics_from_q_sets(
    *,
    kind: str,
    count: int,
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    bM1: np.ndarray,
    bM2: np.ndarray,
    tol: float | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    if count == 0:
        return {}, {"kind": kind, "selected": [], "count": 0}
    b_norm = max(float(np.linalg.norm(bM1)), float(np.linalg.norm(bM2)), 1.0)
    tol = float(tol if tol is not None else max(1.0e-8, b_norm * 1.0e-6))
    records = _unique_vector_records(_candidate_vectors(kind, Q_set1, Q_set2), tol=tol, bM1=bM1, bM2=bM2)
    records.sort(key=lambda item: (round(float(item["norm"]) / tol), _representative_score(item["vector"])))

    assigned: set[int] = set()
    stars: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        if idx in assigned:
            continue
        member_indices = _orbit_member_indices(records, record["vector"], tol=tol)
        assigned.update(member_indices)
        rep_idx = _select_star_representative(records, member_indices)
        rep = records[rep_idx]
        stars.append(
            {
                "vector": np.asarray(rep["vector"], dtype=float),
                "norm": float(rep["norm"]),
                "pair_count": int(sum(records[i]["pair_count"] for i in member_indices)),
                "member_count": int(len(member_indices)),
                "lattice_index": rep["lattice_index"],
                "orbit_generators": [str(_HERMITIAN_HARMONIC_ACTION["name"])],
            }
        )

    stars.sort(key=lambda item: (round(float(item["norm"]) / tol), _representative_score(item["vector"])))
    selected = stars[:count]
    if len(selected) < count:
        raise ValueError(f"Requested {count} {kind} harmonics but only found {len(selected)} Q-difference stars")

    harmonic_map = {idx + 1: np.asarray(item["vector"], dtype=float) for idx, item in enumerate(selected)}
    selected_rows = []
    for idx, item in enumerate(selected, start=1):
        selected_rows.append(
            {
                "index": idx,
                "vector": np.asarray(item["vector"], dtype=float).tolist(),
                "norm": float(item["norm"]),
                "pair_count": int(item["pair_count"]),
                "member_count": int(item["member_count"]),
                "orbit_size": int(item["member_count"]),
                "orbit_generators": list(item["orbit_generators"]),
                "lattice_index": item["lattice_index"],
            }
        )
    return harmonic_map, {
        "kind": kind,
        "count": count,
        "generation": "symmetry_orbit",
        "orbit_generators": [str(_HERMITIAN_HARMONIC_ACTION["name"])],
        "tolerance": tol,
        "selected": selected_rows,
    }



def _apply_harmonic_action(vector: np.ndarray, action: Mapping[str, Any]) -> np.ndarray:
    action_type = str(action.get("type", "")).lower()
    if action_type == "identity":
        return np.asarray(vector, dtype=float)
    if action_type == "negation":
        return -np.asarray(vector, dtype=float)
    if action_type == "rotation":
        return rot(np.asarray(vector, dtype=float), float(action.get("angle_deg", 0.0)))
    if action_type in {"reflection", "mirror"}:
        return _reflect_vector(np.asarray(vector, dtype=float), float(action.get("axis_deg", 0.0)))
    raise ValueError(f"Unsupported harmonic orbit action: {action!r}")



def _harmonic_orbit_vectors(seed: np.ndarray, actions: Sequence[Mapping[str, Any]], *, tol: float) -> list[np.ndarray]:
    orbit: list[np.ndarray] = []
    pending = [np.asarray(seed, dtype=float)]
    while pending:
        vector = pending.pop()
        if any(np.linalg.norm(vector - existing) <= tol for existing in orbit):
            continue
        orbit.append(vector)
        for action in actions:
            pending.append(_apply_harmonic_action(vector, action))
    orbit.sort(key=_representative_score)
    return orbit



_AUTO_VALLEY_HARMONIC_PROFILES: dict[tuple[str, str], dict[str, Any]] = {
    ("K", "inter"): {
        "selection_rule": "K_inter_geometry",
        "orbit_generators": [{"name": "C3z", "type": "rotation", "angle_deg": 120.0}],
        "seeds": ["q1", "-2.0 * q1", "q1 + bM2", "2.0 * bM2 + q3"],
    },
    ("M", "inter"): {
        "selection_rule": "M_inter_qset_support",
        "generation": "qset_support_symmetry_orbit",
        "first_selected_source": "qset_offset",
    },
    ("K", "intra"): {
        "selection_rule": "K_intra_geometry",
        "orbit_generators": [{"name": "C3z", "type": "rotation", "angle_deg": 120.0}],
        "seeds": ["[0.0, 0.0]", "-bM1", "bM1 + bM2", "2.0 * bM1"],
    },
    ("M", "intra"): {
        "selection_rule": "M_intra_geometry",
        "orbit_generators": [{"name": "C2", "type": "reflection", "axis_deg": 0.0}],
        "seeds": ["bM1", "bM2", "-bM1 + bM2", "bM1 + bM2", "-2.0 * bM1 + bM2", "2.0 * bM1"],
    },
}



def _variables_for_valley_harmonic_profile(profile: Mapping[str, Any], variables: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(variables)
    extra_variables = profile.get("variables", {})
    if extra_variables is None:
        return out
    if not isinstance(extra_variables, Mapping):
        raise ValueError(f"valley harmonic profile variables must be a mapping, got {extra_variables!r}")
    for key, expr in extra_variables.items():
        out[str(key)] = evaluate_vector_expression(expr, out)
    return out



def _candidate_orbits_from_valley_profile(
    profile: Mapping[str, Any],
    variables: Mapping[str, Any],
    *,
    tol: float,
) -> list[dict[str, Any]]:
    seed_exprs = profile.get("seeds", [])
    if not isinstance(seed_exprs, Sequence) or isinstance(seed_exprs, (str, bytes)):
        raise ValueError(f"valley harmonic profile seeds must be a list, got {seed_exprs!r}")
    raw_actions = profile.get("orbit_generators", [])
    if not isinstance(raw_actions, Sequence) or isinstance(raw_actions, (str, bytes)):
        raise ValueError(f"valley harmonic profile orbit_generators must be a list, got {raw_actions!r}")
    actions = [dict(action) for action in raw_actions if isinstance(action, Mapping)]
    if len(actions) != len(raw_actions):
        raise ValueError(f"valley harmonic profile orbit_generators must contain mappings, got {raw_actions!r}")
    scoped_variables = _variables_for_valley_harmonic_profile(profile, variables)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for expr in seed_exprs:
        seed = evaluate_vector_expression(expr, scoped_variables)
        orbit = _harmonic_orbit_vectors(seed, actions, tol=tol)
        orbit_key = tuple(sorted(_vector_key(np.asarray(vector, dtype=float), tol) for vector in orbit))
        duplicate = orbit_key in seen
        if not duplicate:
            seen.add(orbit_key)
        rows.append(
            {
                "seed_expr": expr,
                "representative": np.asarray(seed, dtype=float),
                "orbit": orbit,
                "duplicate": duplicate,
                "orbit_generators": [str(action.get("name", action.get("type", ""))) for action in actions],
            }
        )
    return rows



def _auto_valley_harmonics(
    *,
    raw: Any,
    kind: str,
    count: int,
    valley_model: Mapping[str, Any] | None,
    sectors: Sequence[Mapping[str, Any]],
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    bM1: np.ndarray,
    bM2: np.ndarray,
    variables: Mapping[str, Any],
    symmetry_operations: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, np.ndarray], dict[str, Any]] | None:
    valley_type = str((valley_model or {}).get("valley_type", ""))
    profile = _AUTO_VALLEY_HARMONIC_PROFILES.get((valley_type, kind))
    if profile is None:
        return None
    if profile.get("generation") == "qset_support_symmetry_orbit":
        model_q_sectors = [
            {**dict(sector), "q_offset": [0.0, 0.0]}
            for sector in sectors
        ]
        harmonic_map, diagnostics = _auto_harmonics_from_support(
            raw=raw,
            kind=kind,
            count=count,
            sectors=model_q_sectors,
            Q_set1=Q_set1,
            Q_set2=Q_set2,
            bM1=bM1,
            bM2=bM2,
            symmetry_operations=symmetry_operations,
        )
        diagnostics["selection_rule"] = str(profile["selection_rule"])
        diagnostics["q_convention"] = "model_qsets_no_additional_sector_offset"
        selected = diagnostics.get("selected", [])
        if selected and profile.get("first_selected_source") is not None:
            selected[0]["source"] = str(profile["first_selected_source"])
        return harmonic_map, diagnostics
    profile_variables = {**dict(variables), "bM1": np.asarray(bM1, dtype=float), "bM2": np.asarray(bM2, dtype=float)}
    b_norm = max(float(np.linalg.norm(bM1)), float(np.linalg.norm(bM2)), 1.0)
    tol = max(1.0e-8, b_norm * 1.0e-8)
    candidate_rows = _candidate_orbits_from_valley_profile(profile, profile_variables, tol=tol)
    candidates = [row for row in candidate_rows if not bool(row["duplicate"])]
    if count > len(candidates):
        raise ValueError(f"{valley_type} {kind} auto harmonics found only {len(candidates)} unique geometry orbits, got {count}")
    selected = candidates[:count]
    harmonic_map = {idx + 1: np.asarray(row["representative"], dtype=float) for idx, row in enumerate(selected)}
    return harmonic_map, {
        "kind": kind,
        "count": count,
        "generation": "valley_geometry_orbit",
        "selection_rule": str(profile.get("selection_rule", f"{valley_type}_{kind}_geometry")),
        "selected": [
            {
                "index": idx,
                "vector": np.asarray(vector, dtype=float).tolist(),
                "norm": float(np.linalg.norm(vector)),
                "pair_count": None,
                "member_count": int(len(selected[idx - 1]["orbit"])),
                "orbit_size": int(len(selected[idx - 1]["orbit"])),
                "orbit_generators": list(selected[idx - 1]["orbit_generators"]),
                "orbit_vectors": [np.asarray(item, dtype=float).tolist() for item in selected[idx - 1]["orbit"]],
                "seed": str(selected[idx - 1]["seed_expr"]),
                "lattice_index": None,
            }
            for idx, vector in harmonic_map.items()
        ],
    }



def _resolve_harmonics_maps(
    harmonics: Mapping[str, Any],
    variables: Mapping[str, Any],
    *,
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    bM1: np.ndarray,
    bM2: np.ndarray,
    sectors: list[dict[str, Any]] | None = None,
    valley_model: Mapping[str, Any] | None = None,
    symmetry_map: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[str, Any]]:
    diagnostics: dict[str, Any] = {"auto_used": False}
    maps: dict[str, dict[int, np.ndarray]] = {}
    for kind in ("intra", "inter"):
        raw = harmonics.get(kind, {})
        if _is_auto_harmonic_spec(raw):
            count = _auto_harmonic_count(raw, name=f"harmonics.{kind}")
            use_explicit_sector_pairs = (
                kind == "inter"
                and bool(sectors)
                and isinstance(raw, Mapping)
                and raw.get("sector_pairs") is not None
            )
            if use_explicit_sector_pairs:
                maps[kind], diagnostics[kind] = _auto_harmonics_from_support(
                    raw=raw,
                    kind=kind,
                    count=count,
                    sectors=sectors,
                    Q_set1=Q_set1,
                    Q_set2=Q_set2,
                    bM1=bM1,
                    bM2=bM2,
                    symmetry_operations=list((symmetry_map or {}).get(kind, [])),
                )
            else:
                valley_harmonics = _auto_valley_harmonics(
                    raw=raw,
                    kind=kind,
                    count=count,
                    valley_model=valley_model,
                    sectors=sectors or [],
                    Q_set1=Q_set1,
                    Q_set2=Q_set2,
                    bM1=bM1,
                    bM2=bM2,
                    variables=variables,
                    symmetry_operations=list((symmetry_map or {}).get(kind, [])),
                )
                if valley_harmonics is not None:
                    maps[kind], diagnostics[kind] = valley_harmonics
                elif sectors:
                    maps[kind], diagnostics[kind] = _auto_harmonics_from_support(
                        raw=raw,
                        kind=kind,
                        count=count,
                        sectors=sectors,
                        Q_set1=Q_set1,
                        Q_set2=Q_set2,
                        bM1=bM1,
                        bM2=bM2,
                        symmetry_operations=list((symmetry_map or {}).get(kind, [])),
                    )
                else:
                    maps[kind], diagnostics[kind] = _auto_harmonics_from_q_sets(
                        kind=kind,
                        count=count,
                        Q_set1=Q_set1,
                        Q_set2=Q_set2,
                        bM1=bM1,
                        bM2=bM2,
                    )
            diagnostics["auto_used"] = True
        else:
            maps[kind] = _harmonics_map(dict(raw), variables)
            diagnostics[kind] = {
                "kind": kind,
                "count": len(maps[kind]),
                "harmonics_source": "explicit_config",
                "selected": [
                    {
                        "index": int(index),
                        "vector": np.asarray(vector, dtype=float).tolist(),
                        "norm": float(np.linalg.norm(vector)),
                        "pair_count": None,
                        "member_count": None,
                        "lattice_index": None,
                        "p_label": None,
                        "support_count": None,
                        "support_weight_norm": None,
                        "chosen": True,
                        "rejected_reason": None,
                    }
                    for index, vector in sorted(maps[kind].items())
                ],
            }
    return maps["intra"], maps["inter"], diagnostics



def _load_kpoints(config: ConfiguredModel) -> np.ndarray:
    if config.kpoints_file is not None:
        kpoints = _validate_kpoints(np.load(config.kpoints_file, allow_pickle=False))
        if bool(getattr(config, "kpoints_from_projection", False)):
            heff_rows = int(np.load(config.heff_file, mmap_mode="r", allow_pickle=False).shape[0])
            if len(kpoints) != heff_rows:
                raise ValueError(
                    "projection kpoints/Heff row mismatch: "
                    f"{len(kpoints)} != {heff_rows} ({config.kpoints_file})"
                )
            identity = getattr(config, "artifact_identity", {})
            stored_hash = require_identity_fields(
                identity,
                ("kpoints_hash",),
                f"KP projection k-points {config.kpoints_file}",
            )["kpoints_hash"]
            actual_hash = hash_array(kpoints)
            if actual_hash != stored_hash:
                raise ValueError(
                    f"KP projection k-points {config.kpoints_file} kpoints_hash mismatch: "
                    f"{stored_hash} != {actual_hash}"
                )
            return np.asarray([rot(point, float(config.rotation_deg)) for point in kpoints], dtype=float)
        return _apply_project_k_indices(kpoints, config.project_k_indices)

    if not config.kpath_config:
        raise ValueError("Configured model requires kpoints_file or kpath section")
    kpath = config.kpath_config
    file_path = _resolve_path(kpath.get("file"), config.path.parent)
    if file_path is None:
        raise ValueError("kpath.file is required when kpoints_file is absent")
    tmat = np.asarray(kpath.get("tmat"), dtype=float)
    phase_deg = float(config.rotation_deg)
    segment_points = kpath.get("segment_points")
    output_file = _resolve_path(kpath.get("output_file"), config.path.parent)
    generated = generate_kpath_from_file(
        Tmat=tmat,
        file_path=file_path,
        phase_deg=phase_deg,
        segment_points=None if segment_points is None else int(segment_points),
        output_file_path=output_file,
    )
    return _apply_project_k_indices(generated.kpoints_2d, config.project_k_indices)



def _load_kpoints_from_inputs(
    *,
    kpoints_file: Path | None,
    kpath_config: Mapping[str, Any],
    base: Path,
    rotation_deg: float,
    kpoints_from_projection: bool = False,
) -> np.ndarray:
    if kpoints_file is not None:
        points = _validate_kpoints(np.load(kpoints_file, allow_pickle=False))
        if kpoints_from_projection:
            return np.asarray([rot(point, float(rotation_deg)) for point in points], dtype=float)
        return points
    if not kpath_config:
        raise ValueError("fit.mode=auto_compact requires kpoints_file or kpath section")
    file_path = _resolve_path(kpath_config.get("file"), base)
    if file_path is None:
        raise ValueError("kpath.file is required when fit.mode=auto_compact uses generated k-points")
    tmat = np.asarray(kpath_config.get("tmat"), dtype=float)
    segment_points = kpath_config.get("segment_points")
    output_file = _resolve_path(kpath_config.get("output_file"), base)
    generated = generate_kpath_from_file(
        Tmat=tmat,
        file_path=file_path,
        phase_deg=float(rotation_deg),
        segment_points=None if segment_points is None else int(segment_points),
        output_file_path=output_file,
    )
    return _validate_kpoints(generated.kpoints_2d)



def _select_auto_fit_indices(
    kpoints: np.ndarray,
    fit_config: Mapping[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    mode = str(fit_config.get("mode", "auto_compact")).strip().lower()
    if mode == "auto":
        mode = "auto_compact"
    if mode != "auto_compact":
        raise ValueError(f"Unsupported fit.mode {fit_config.get('mode')!r}; expected 'auto_compact'")
    arr = _validate_kpoints(kpoints)
    candidate_count = int(arr.shape[0])
    if candidate_count <= 0:
        raise ValueError("fit.mode=auto_compact requires at least one k-point")
    max_points = int(fit_config.get("max_points", 4))
    if max_points <= 0:
        raise ValueError(f"fit.max_points must be positive for auto_compact, got {max_points}")
    count = min(max_points, candidate_count)
    if count == 1:
        selected = [0]
    else:
        selected = sorted({int(round(i * (candidate_count - 1) / (count - 1))) for i in range(count)})
        if len(selected) < count:
            for idx in range(candidate_count):
                if idx not in selected:
                    selected.append(idx)
                if len(selected) == count:
                    break
            selected.sort()
    metadata = {
        "mode": "auto_compact",
        "source": "auto_compact_path_spacing",
        "requested_max_points": max_points,
        "candidate_count": candidate_count,
        "selected_indices": [int(idx) for idx in selected],
    }
    return [int(idx) for idx in selected], metadata



def _validate_kpoints(kpoints: Any) -> np.ndarray:
    arr = np.asarray(kpoints, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"kpoints must have shape (Nk,2), got {arr.shape}")
    return arr



def _select_rows(arr: np.ndarray, indices: Sequence[int] | None) -> np.ndarray:
    if indices is None:
        return np.asarray(arr)
    idx = np.asarray(list(indices), dtype=int)
    return np.asarray(arr)[idx]



def _block_diag_heff(heff_list: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    if not indices:
        raise ValueError("fit.indices must contain at least one k-point index")
    blocks = [np.asarray(heff_list[int(i)]) for i in indices]
    return scipy.linalg.block_diag(*blocks)



def _symmetry_artifact_path(config: ConfiguredModel) -> Path | None:
    if config.symmetry_source_config.get("type") != "kp_symm_output":
        return None
    raw_path = config.symmetry_source_config.get("path")
    if raw_path is None:
        return None
    path = Path(str(raw_path))
    return path if path.is_absolute() else (config.path.parent / path).resolve()






def _load_q_sets_from_symmetry_artifact(config: ConfiguredModel) -> tuple[np.ndarray, np.ndarray] | None:
    artifact_path = _symmetry_artifact_path(config)
    if artifact_path is None:
        return None
    representations = artifact_path / "representations.npz"
    canonical_keys = (
        "__q_model_canonical_layer1__",
        "__q_model_canonical_layer2__",
    )
    raw_keys = (
        "__q_model_raw_layer1__",
        "__q_model_raw_layer2__",
    )
    if representations.is_file():
        with np.load(representations, allow_pickle=False) as payload:
            present_canonical = tuple(key in payload.files for key in canonical_keys)
            present_raw = tuple(key in payload.files for key in raw_keys)
            if any(present_canonical) and not all(present_canonical):
                raise ValueError(
                    f"kp symm package contains an incomplete canonical Q pair: {representations}"
                )
            if any(present_raw) and not all(present_raw):
                raise ValueError(f"kp symm package contains an incomplete raw Q pair: {representations}")
            if all(present_canonical):
                if not all(present_raw) or "__metadata_json__" not in payload.files:
                    raise ValueError(
                        f"kp symm canonical Q package lacks raw Q or metadata: {representations}"
                    )
                q1 = np.asarray(payload[canonical_keys[0]], dtype=np.float64)
                q2 = np.asarray(payload[canonical_keys[1]], dtype=np.float64)
                raw_q1 = np.asarray(payload[raw_keys[0]], dtype=np.float64)
                raw_q2 = np.asarray(payload[raw_keys[1]], dtype=np.float64)
                metadata = json.loads(str(payload["__metadata_json__"].item()))
                q_model = metadata.get("q_model", {}) if isinstance(metadata, Mapping) else {}
                canonicalization = (
                    q_model.get("canonicalization", {}) if isinstance(q_model, Mapping) else {}
                )
                if not isinstance(canonicalization, Mapping) or canonicalization.get("status") not in {
                    "certified",
                    "not_needed",
                }:
                    raise ValueError(
                        f"kp symm canonical Q package is not certified: {representations}"
                    )
                sector_order = tuple(canonicalization.get("sector_order", ()))
                if sector_order != ("L1", "L2"):
                    raise ValueError(
                        "kp symm canonical Q package has unsupported sector ordering: "
                        f"{sector_order!r}"
                    )
                from ..symmetry.q_canonicalization import q_geometry_hash

                expected_raw_hash = str(canonicalization.get("raw_q_hash", ""))
                expected_canonical_hash = str(canonicalization.get("canonical_q_hash", ""))
                actual_raw_hash = q_geometry_hash(
                    {"L1": raw_q1, "L2": raw_q2}, sector_order
                )
                actual_canonical_hash = q_geometry_hash(
                    {"L1": q1, "L2": q2}, sector_order
                )
                if actual_raw_hash != expected_raw_hash:
                    raise ValueError("kp symm raw Q hash does not match the packed arrays")
                if actual_canonical_hash != expected_canonical_hash:
                    raise ValueError("kp symm canonical Q hash does not match the packed arrays")
                for name, array in (("layer1", q1), ("layer2", q2)):
                    if array.ndim != 2 or array.shape[1] != 2 or not np.all(np.isfinite(array)):
                        raise ValueError(
                            f"kp symm canonical Q {name} must be a finite (N, 2) array"
                        )
                return q1, q2
    # q_model.files are raw-coordinate diagnostics. Only the certified packed
    # canonical arrays above can supply the exactified model coordinates.
    return None



def _load_model_q_sets(config: ConfiguredModel) -> tuple[np.ndarray, np.ndarray]:
    artifact_q_sets = _load_q_sets_from_symmetry_artifact(config)
    if artifact_q_sets is not None:
        return artifact_q_sets
    if (
        str(config.symmetry_source_config.get("type", "")) == "kp_symm_output"
        and str(config.response_semantics) == "complete_linear"
    ):
        raise ValueError(
            "complete_linear requires canonical Q arrays from the exactified symmetry package; "
            "rerun `kp symm` with the current code"
        )
    return load_Q_sets_from_gvec_files(
        config.qset1_file,
        config.qset2_file,
        rotation_deg=config.rotation_deg,
    )



def _heff_in_model_basis(
    heff: np.ndarray,
    model_basis_gauge: np.ndarray | None,
) -> np.ndarray:
    """Return Hamiltonians in the response basis used by model symmetries."""

    if model_basis_gauge is None:
        return heff
    array = np.asarray(heff, dtype=np.complex128)
    gauge = np.asarray(model_basis_gauge, dtype=np.complex128)
    if array.ndim != 3 or array.shape[1:] != gauge.shape:
        raise ValueError(
            "model response gauge and Heff shapes differ: "
            f"heff={array.shape}, gauge={gauge.shape}"
        )
    return np.einsum(
        "ab,kbc,cd->kad",
        gauge.conjugate().T,
        array,
        gauge,
        optimize=True,
    )



def _load_heff_in_model_basis(config: ConfiguredModel) -> np.ndarray:
    return _heff_in_model_basis(
        np.load(config.heff_file, mmap_mode="r"),
        getattr(config, "model_basis_gauge", None),
    )



def _operation_matrix_is_exactified(record: Mapping[str, Any]) -> bool:
    return exactified_operation_provenance_is_complete(record)



def _requires_model_side_exactification(metadata: Mapping[str, Any]) -> bool:
    if bool(metadata.get("requires_model_exactification", False)):
        return True
    operations = metadata.get("operations", [])
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)) or not operations:
        return False
    return not all(isinstance(record, Mapping) and _operation_matrix_is_exactified(record) for record in operations)



def build_moire_config_from_file(path: str | Path) -> tuple[MoireConfig, ConfiguredModel]:
    from . import validation

    config = load_model_config(path)
    config.artifact_identity = _load_model_artifact_identity(
        config.heff_file,
        config.symmetry_source_config,
        base=config.path.parent,
    )
    heff_list = np.load(config.heff_file, mmap_mode="r")
    Q_set1, Q_set2 = _load_model_q_sets(config)
    fit_heff_list = heff_list
    auto_low_energy_fit = (
        str(config.fit_selection_metadata.get("mode", "")).strip().lower()
        == "auto_low_energy"
    )
    spectral_support_fit = (
        str(config.response_fit_objective.get("mode", "")).strip().lower()
        in {"target_spectral_linear", "normalized_low_energy_linear"}
        and str(
            config.response_fit_objective.get(
                "target_reference", "current_heff_support_mask"
            )
        ).strip().lower()
        == "current_heff_support_mask"
    )
    if auto_low_energy_fit or spectral_support_fit:
        fit_heff_list = validation._current_heff_support_hamiltonians(
            heff_list,
            qset1=Q_set1,
            qset2=Q_set2,
            n_orb=config.n_orb,
            harmonics_config=config.harmonics_config,
        )
    if auto_low_energy_fit:
        validation._retarget_auto_low_energy_windows_to_current_support(
            config,
            _select_rows(fit_heff_list, config.band_indices),
        )

    bM1, bM2, bM_diagnostics = _build_bM_vectors(Q_set1, Q_set2, config)
    config.bM_diagnostics = bM_diagnostics
    model_section = _model_section(config.raw)
    variables = _build_variables(Q_set1, bM1, bM2, model_section)
    harmonics = config.harmonics_config
    sectors = sectors_with_q_offsets(
        _default_sectors(config),
        Q_set1=Q_set1,
        Q_set2=Q_set2,
        bM1=bM1,
        bM2=bM2,
    )
    expected_dim = len(Q_set1) * config.n_orb[0] + len(Q_set2) * config.n_orb[1]
    loaded_symmetry = load_symmetry_source(
        config.symmetry_source_config,
        base=config.path.parent,
        expected_dim=expected_dim,
    )
    source_type = str(config.symmetry_source_config.get("type", "none"))
    if source_type == "kp_symm_output" and _requires_model_side_exactification(
        loaded_symmetry.metadata
    ):
        raise ValueError(
            "kp_symm_output must provide exactified continuum matrices. "
            "Rerun `kp symm` so the manifest contains matrix_kind='continuum_internal_rep_exact' "
            "and matrix_source='kp_symm_exactified_action'."
        )
    if config.symmetry_source_metadata:
        loaded_symmetry.metadata = {
            **config.symmetry_source_metadata,
            **loaded_symmetry.metadata,
        }
        if hasattr(loaded_symmetry.generator, "metadata") and isinstance(
            loaded_symmetry.generator.metadata,
            dict,
        ):
            loaded_symmetry.generator.metadata.update(
                config.symmetry_source_metadata
            )
    config.symmetry_source_metadata = loaded_symmetry.metadata
    enriched_symmetry_map = _enrich_symmetry_map(
        config.symmetry_map,
        loaded_symmetry.metadata,
        rotation_deg=config.rotation_deg,
    )
    config.symmetry_map = enriched_symmetry_map
    intra, inter, harmonics_diagnostics = _resolve_harmonics_maps(
        harmonics,
        variables,
        Q_set1=Q_set1,
        Q_set2=Q_set2,
        bM1=bM1,
        bM2=bM2,
        sectors=sectors,
        valley_model=config.valley_model,
        symmetry_map=enriched_symmetry_map,
    )
    config.harmonics_diagnostics = harmonics_diagnostics
    case_generator = config.term_template_metadata.get("generator")
    if case_generator in set(_CASE_DERIVED_GENERATORS.values()):
        valley_type = str(config.valley_model.get("valley_type", ""))
        expected_generator = _CASE_DERIVED_GENERATORS.get(valley_type)
        if config.response_semantics != "complete_linear":
            raise ValueError(
                "Case-derived response generation requires complete_linear"
            )
        if case_generator != expected_generator:
            raise ValueError(
                "Case-derived response generator metadata is inconsistent with "
                f"valley_type={valley_type!r}: got {case_generator!r}, "
                f"expected {expected_generator!r}"
            )
        if config.term_templates:
            raise ValueError(
                "Case-derived response generation cannot replace explicit term_templates"
            )
        generated_templates, generator_diagnostics = (
            _case_derived_term_templates(
                valley_type=valley_type,
                sectors=sectors,
                n_orb=config.n_orb,
                max_order=config.max_order,
                Q_set1=Q_set1,
                Q_set2=Q_set2,
                intra_harmonics_map=intra,
                inter_harmonics_map=inter,
            )
        )
        if not generated_templates:
            raise ValueError(
                f"{valley_type} case-derived response generation produced no physical candidates"
            )
        config.term_templates = generated_templates
        config.term_template_metadata = {
            **dict(config.term_template_metadata),
            **generator_diagnostics,
            "term_space_policies": ["complete"],
        }
    if case_generator in set(_CASE_DERIVED_GENERATORS.values()):
        term_templates = [dict(row) for row in config.term_templates]
        inter_direction_diagnostics: list[dict[str, Any]] = []
    else:
        term_templates, inter_direction_diagnostics = _orient_k_sector_inter_templates_by_raw_support(
            config.term_templates,
            valley_model=config.valley_model,
            n_orb=config.n_orb,
            sectors=sectors,
            Q_set1=Q_set1,
            Q_set2=Q_set2,
            inter_harmonics_map=inter,
            term_template_metadata=config.term_template_metadata,
        )
    config.term_templates = term_templates
    if inter_direction_diagnostics:
        config.term_template_metadata = dict(config.term_template_metadata)
        config.term_template_metadata["inter_sector_pair_resolution"] = inter_direction_diagnostics
    kpoints_all = _load_kpoints(config)
    band_kpoints = _select_rows(kpoints_all, config.band_indices)
    fit_kpoints = _select_rows(kpoints_all, config.fit_indices)
    config.model_basis_gauge = getattr(
        loaded_symmetry.generator,
        "model_basis_gauge",
        None,
    )
    fit_heff_list = _heff_in_model_basis(
        fit_heff_list,
        config.model_basis_gauge,
    )

    moire_config = MoireConfig(
        Q_set1=Q_set1,
        Q_set2=Q_set2,
        n_orb1=config.n_orb[0],
        n_orb2=config.n_orb[1],
        nlow_state=list(config.nlow_state),
        bM1=bM1,
        bM2=bM2,
        intra_harmonics_map=intra,
        inter_harmonics_map=inter,
        max_order=dict(config.max_order),
        symmetry_map=dict(enriched_symmetry_map),
        kpoints=band_kpoints,
        kpoints_fit=fit_kpoints,
        heff=_block_diag_heff(fit_heff_list, config.fit_indices),
        coeff_tol=config.coeff_tol,
        null_channel_abs_tol=config.null_channel_abs_tol,
        null_channel_rel_tol=config.null_channel_rel_tol,
        response_semantics=config.response_semantics,
        response_regularization=config.response_regularization,
        response_fit_indices=list(config.fit_indices),
        response_band_window=list(config.band_slice or []),
        response_fit_objective=dict(config.response_fit_objective),
        output_dir=config.output_dir,
    )
    if loaded_symmetry.generator is not None:
        moire_config.symmetry_gen = loaded_symmetry.generator
    moire_config.term_templates = [dict(item) for item in config.term_templates]
    moire_config.sectors = sectors
    moire_config.symmetry_source_metadata = loaded_symmetry.metadata
    moire_config.bM_diagnostics = bM_diagnostics
    moire_config.orbital_count_metadata = dict(config.orbital_count_metadata)
    setattr(moire_config, "_harmonics_diagnostics", harmonics_diagnostics)
    return moire_config, config



def _bM_candidate_rows(candidates: list[np.ndarray], *, max_rows: int = 24) -> list[dict[str, Any]]:
    rows = []
    for vector in candidates:
        arr = np.asarray(vector, dtype=float)
        norm = float(np.linalg.norm(arr))
        if norm <= 1.0e-12:
            continue
        rows.append({"vector": arr.tolist(), "norm": norm, "angle_deg": float(np.degrees(np.arctan2(arr[1], arr[0])))})
    rows.sort(key=lambda item: (item["norm"], item["angle_deg"]))
    return rows[:max_rows]



def _reconstruction_diagnostics(
    *,
    vectors: list[np.ndarray],
    bM1: np.ndarray,
    bM2: np.ndarray,
    tol: float,
) -> dict[str, Any]:
    basis = np.column_stack([bM1, bM2])
    errors: list[float] = []
    failures = 0
    checked = 0
    for vector in vectors:
        arr = np.asarray(vector, dtype=float)
        if np.linalg.norm(arr) <= 1.0e-12:
            continue
        checked += 1
        coeff = np.linalg.solve(basis, arr)
        rounded = np.rint(coeff)
        error = float(np.linalg.norm(arr - basis @ rounded))
        errors.append(error)
        if error > tol:
            failures += 1
    if errors:
        max_error = float(np.max(errors))
        median_error = float(np.median(errors))
    else:
        max_error = 0.0
        median_error = 0.0
    return {
        "checked_count": int(checked),
        "max_error": max_error,
        "median_error": median_error,
        "failure_count": int(failures),
        "failure_fraction": float(failures / checked) if checked else 0.0,
        "tolerance": float(tol),
    }



def _bM_geometry(bM1: np.ndarray, bM2: np.ndarray) -> dict[str, Any]:
    n1 = float(np.linalg.norm(bM1))
    n2 = float(np.linalg.norm(bM2))
    dot = float(np.dot(bM1, bM2) / (n1 * n2)) if n1 > 0 and n2 > 0 else 1.0
    return {
        "b1": np.asarray(bM1, dtype=float).tolist(),
        "b2": np.asarray(bM2, dtype=float).tolist(),
        "norms": [n1, n2],
        "angle_deg": float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))),
    }



def _bM_candidates_from_tmat(Tmat: Any, *, rotation_deg: float) -> list[np.ndarray]:
    reciprocal = reciprocal_Tmat_from_Tmat(np.asarray(Tmat, dtype=float))
    g1 = rot(reciprocal[0, :2], rotation_deg)
    g2 = rot(reciprocal[1, :2], rotation_deg)
    candidates: list[np.ndarray] = []
    for m in range(-2, 3):
        for n in range(-2, 3):
            if m == 0 and n == 0:
                continue
            candidates.append(m * g1 + n * g2)
    return candidates



def _build_bM_vectors(Q_set1: np.ndarray, Q_set2: np.ndarray, config: ConfiguredModel) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw = config.bM_config
    angle = float(raw.get("angle_deg", 60.0))
    source = str(raw.get("source", "auto")).lower()
    if raw.get("infer_from_q", False):
        source = "q_distance"
    if source == "auto" and "bM1" in raw:
        source = "explicit"
    if not raw:
        source = "q_distance"
    elif source == "auto":
        source = "tmat" if config.kpath_config.get("tmat") is not None else "q_distance"
    if source in {"q", "q_distance", "q_distances"}:
        candidates = bM_candidates_from_q_distances(Q_set1, Q_set2)
        bM1, bM2 = canonical_bM_pair_from_candidates(candidates, angle_deg=angle)
        b_norm = max(float(np.linalg.norm(bM1)), float(np.linalg.norm(bM2)), 1.0)
        validation = raw.get("validation", {})
        if validation is None:
            validation = {}
        if not isinstance(validation, Mapping):
            raise ValueError("model.bM.validation must be a mapping")
        tol = float(validation.get("max_error", max(1.0e-8, b_norm * 1.0e-6)))
        max_failure_fraction = float(validation.get("max_failure_fraction", 0.0))
        reconstruction = _reconstruction_diagnostics(vectors=candidates, bM1=bM1, bM2=bM2, tol=tol)
        diagnostics = {
            "source": "q_distance",
            **_bM_geometry(bM1, bM2),
            "candidates": _bM_candidate_rows(candidates),
            "reconstruction": reconstruction,
        }
        if reconstruction["failure_fraction"] > max_failure_fraction:
            raise ValueError(
                "q_distance bM reconstruction validation failed: "
                f"failure_fraction={reconstruction['failure_fraction']:.3g} > {max_failure_fraction:.3g}, "
                f"max_error={reconstruction['max_error']:.3g}"
            )
        if config.kpath_config.get("tmat") is not None:
            tmat_candidates = _bM_candidates_from_tmat(config.kpath_config["tmat"], rotation_deg=config.rotation_deg)
            t_bM1, t_bM2 = canonical_bM_pair_from_candidates(tmat_candidates, angle_deg=angle)
            diagnostics["comparison_with_tmat"] = {
                "b1_delta_norm": float(np.linalg.norm(bM1 - t_bM1)),
                "b2_delta_norm": float(np.linalg.norm(bM2 - t_bM2)),
                "tmat_b1": t_bM1.tolist(),
                "tmat_b2": t_bM2.tolist(),
            }
        return bM1, bM2, diagnostics
    if source == "tmat":
        if config.kpath_config.get("tmat") is None:
            raise ValueError("model.bM.source=tmat requires kpath.tmat")
        candidates = _bM_candidates_from_tmat(config.kpath_config["tmat"], rotation_deg=config.rotation_deg)
        bM1, bM2 = canonical_bM_pair_from_candidates(candidates, angle_deg=angle)
        diagnostics = {
            "source": "tmat",
            **_bM_geometry(bM1, bM2),
            "candidates": _bM_candidate_rows(candidates),
            "comparison_with_tmat": {"b1_delta_norm": 0.0, "b2_delta_norm": 0.0, "tmat_b1": bM1.tolist(), "tmat_b2": bM2.tolist()},
        }
        return bM1, bM2, diagnostics
    variables = {"zero": np.zeros(2), "pi": float(np.pi), "sqrt3": float(np.sqrt(3.0))}
    if "bM1" not in raw:
        raise ValueError("model.bM must provide bM1 or source q_distance/tmat")
    bM1 = evaluate_vector_expression(raw["bM1"], variables)
    variables["bM1"] = bM1
    if "bM2" in raw:
        bM2 = evaluate_vector_expression(raw["bM2"], variables)
    else:
        bM2 = rot(bM1, float(raw.get("angle_deg", 60.0)))
    return bM1, bM2, {"source": "explicit", **_bM_geometry(bM1, bM2), "candidates": []}

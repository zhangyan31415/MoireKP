from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


PHYSICAL_TR_NAMES = {"TR"}
PHYSICAL_C2_NAMES = {"C2", "C2z"}
K_SINGLE_ALLOWED_INTERNAL = {"C3z", "C2T"}
M_SPINLESS_ALLOWED_INTERNAL = {"TR", "C2", "C2T"}
CANONICAL_INTERNAL_NAMES = {"C3z", "C2", "TR", "C2T"}
ALLOWED_GENERATION_MODES = {"representation_invariant"}
M_EFFECTIVE_OPERATION_ALIASES = {
    "TR_eff": {"canonical": "TR", "effective_name": "TR_eff"},
    "C2_eff": {"canonical": "C2", "effective_name": "C2_eff"},
    "C2TR_eff": {"canonical": "C2T", "effective_name": "C2TR_eff", "derived_from": ["C2", "TR"]},
}


def is_m_spinless_effective(valley_model: Mapping[str, Any] | None) -> bool:
    valley_model = valley_model or {}
    return (
        str(valley_model.get("valley_type", "")) == "M"
        and str(valley_model.get("mode", "")) == "single_valley"
        and str(valley_model.get("spin_convention", "")) == "spinless_effective"
    )


def effective_operation_metadata_for_valley(
    name: str,
    valley_model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not is_m_spinless_effective(valley_model):
        return {}
    raw = str(name)
    alias = M_EFFECTIVE_OPERATION_ALIASES.get(raw)
    if alias is None:
        canonical = canonical_operation_name_for_valley(raw, valley_model)
        for legacy_name, legacy_meta in M_EFFECTIVE_OPERATION_ALIASES.items():
            if str(legacy_meta["canonical"]) == canonical:
                alias = {"operation_alias": legacy_name, **legacy_meta}
                break
    else:
        alias = {"operation_alias": raw, **alias}
    if alias is None:
        return {}
    out = {
        "operation_alias": str(alias["operation_alias"]),
        "canonical_physical_operation": str(alias["canonical"]),
        "physical_parent": str(alias["canonical"]),
        "representation_level": "effective_single_spin",
        "effective_name": str(alias["effective_name"]),
        "approximation": {"kind": "spin_SU2_effective_block"},
    }
    if "derived_from" in alias:
        out["derived_from"] = list(alias["derived_from"])
    return out


def canonical_operation_name_for_valley(name: str, valley_model: Mapping[str, Any] | None = None) -> str:
    raw = str(name)
    valley_model = valley_model or {}
    if is_m_spinless_effective(valley_model) and raw in M_EFFECTIVE_OPERATION_ALIASES:
        return str(M_EFFECTIVE_OPERATION_ALIASES[raw]["canonical"])
    valley_type = str(valley_model.get("valley_type", ""))
    mode = str(valley_model.get("mode", ""))
    spin_convention = str(valley_model.get("spin_convention", ""))
    if raw == "C2T" and valley_type == "K" and mode == "single_valley" and spin_convention in {"spin_up_projected", "spin_down_projected"}:
        return "C2T"
    return raw


def canonical_source_operation_name_for_valley(name: str, valley_model: Mapping[str, Any] | None = None) -> str:
    raw = str(name)
    return canonical_operation_name_for_valley(raw, valley_model)


def canonical_vector_key(vector: Sequence[float], *, tol: float = 1.0e-9) -> tuple[int, int]:
    arr = np.asarray(vector, dtype=float)
    if arr.shape != (2,):
        raise ValueError(f"canonical_vector_key expects a 2-vector, got shape {arr.shape}")
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}")
    return tuple(np.rint(arr / float(tol)).astype(int).tolist())


def symmetry_operation_names(symmetry_map: Mapping[str, Any], *, valley_model: Mapping[str, Any] | None = None) -> list[str]:
    names: list[str] = []
    for ops in symmetry_map.values():
        if not isinstance(ops, Sequence) or isinstance(ops, (str, bytes)):
            continue
        for op in ops:
            if isinstance(op, Mapping) and "name" in op:
                names.append(canonical_operation_name_for_valley(str(op["name"]), valley_model))
            elif isinstance(op, str):
                names.append(canonical_operation_name_for_valley(op, valley_model))
    return names


def _model_section(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    model = raw.get("model", {})
    if not isinstance(model, Mapping):
        raise ValueError("model section must be a mapping")
    return model


def _operation_set_from_source(
    symmetry_source: Mapping[str, Any],
    *,
    valley_model: Mapping[str, Any] | None = None,
) -> set[str]:
    operations = symmetry_source.get("operations", [])
    out: set[str] = set()
    if isinstance(operations, Mapping):
        iterable = []
        for key, value in operations.items():
            if isinstance(value, Mapping):
                row = dict(value)
                row.setdefault("name", str(key))
                iterable.append(row)
            else:
                iterable.append({"name": str(key)})
        operations = iterable
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)):
        return out
    for op in operations:
        if isinstance(op, Mapping):
            if "name" in op:
                out.add(canonical_source_operation_name_for_valley(str(op["name"]), valley_model))
        elif isinstance(op, str):
            out.add(canonical_source_operation_name_for_valley(op, valley_model))
    return out


def validate_model_config(raw: Mapping[str, Any], *, nlow_state: Sequence[int]) -> None:
    model = _model_section(raw)
    symmetry_map = model.get("symmetry_map", {})
    if not isinstance(symmetry_map, Mapping):
        raise ValueError("model.symmetry_map must be a mapping")

    valley_model = raw.get("valley_model")
    if not isinstance(valley_model, Mapping):
        raise ValueError("model config requires explicit valley_model")
    sym_names = set(symmetry_operation_names(symmetry_map, valley_model=valley_model))
    lattice = str(valley_model.get("lattice", ""))
    system = str(valley_model.get("system", ""))
    valley_type = str(valley_model.get("valley_type", ""))
    mode = str(valley_model.get("mode", ""))
    spin_convention = str(valley_model.get("spin_convention", ""))
    if lattice != "hexagonal":
        raise ValueError(f"Only lattice=hexagonal is currently supported, got {lattice!r}")
    if system != "bilayer":
        raise ValueError(f"Only system=bilayer is currently supported, got {system!r}")
    if valley_type not in {"Gamma", "K", "M"}:
        raise ValueError(f"valley_model.valley_type must be Gamma/K/M, got {valley_type!r}")
    if mode not in {"single_valley", "valley_pair", "triple_valley"}:
        raise ValueError(f"valley_model.mode must be single_valley/valley_pair/triple_valley, got {mode!r}")
    unsupported_ops = sym_names - CANONICAL_INTERNAL_NAMES
    if unsupported_ops:
        raise ValueError(f"Unsupported internal symmetry operation names: {sorted(unsupported_ops)}")
    symmetry_source = raw.get("symmetry_source", {})
    if symmetry_source is None:
        symmetry_source = {}
    if not isinstance(symmetry_source, Mapping):
        raise ValueError("symmetry_source must be a mapping when provided")
    source_type = str(symmetry_source.get("type", "none"))
    source_ops = _operation_set_from_source(symmetry_source, valley_model=valley_model)

    if sym_names and source_type == "none":
        raise ValueError("model.symmetry_map is non-empty; provide symmetry_source")

    if valley_type == "K" and mode == "single_valley":
        forbidden = (sym_names & PHYSICAL_TR_NAMES) | (sym_names & PHYSICAL_C2_NAMES)
        if forbidden:
            raise ValueError(
                "K single_valley cannot use physical TR/C2 as internal symmetry; "
                "use valley_pair with external_sewing_symmetries or provide C2T/C3z internal operations."
            )
        invalid = sym_names - K_SINGLE_ALLOWED_INTERNAL
        if invalid:
            raise ValueError(f"K single_valley internal symmetries must be C3z/C2T, got {sorted(invalid)}")

    if valley_type == "M" and mode == "single_valley" and spin_convention == "spinless_effective":
        invalid = sym_names - M_SPINLESS_ALLOWED_INTERNAL
        if invalid and source_type != "kp_symm_output":
            raise ValueError(f"M spinless_effective internal symmetries must be TR/C2/C2T, got {sorted(invalid)}")

    if valley_type == "M" and mode == "triple_valley":
        active_valleys = list(valley_model.get("active_valleys", []))
        if sorted(active_valleys) != ["M1", "M2", "M3"]:
            raise ValueError("M triple_valley requires active_valleys: [M1, M2, M3]")
        if "C3z" in sym_names and source_type != "kp_symm_output":
            raise ValueError("M triple_valley C3z requires kp_symm_output or explicit triple-valley template")

    if source_type == "toy_generator":
        if not bool(symmetry_source.get("allow", False)):
            raise ValueError("toy_generator requires explicit symmetry_source.allow: true")
        template = symmetry_source.get("basis_template")
        if any(name in sym_names for name in {"C3z", "C2", "C2T", "TR"}) and not template:
            raise ValueError("toy_generator symmetry operations require explicit basis_template")
        if "C2T" in sym_names:
            raise ValueError("C2T toy generator is not supported; use kp_symm_output matrices")
        if (
            sym_names & PHYSICAL_TR_NAMES
            and any(int(n) % 2 for n in nlow_state)
            and not (valley_type == "M" and mode == "single_valley" and spin_convention == "spinless_effective")
        ):
            raise ValueError(
                "TR toy generator requires explicit spin/Kramers pair basis or a kp_symm_output representation; "
                "M single-spin effective TR requires an effective_single_spin representation."
            )
        if ({"C2"} & sym_names) and str(template) not in {"Gamma_four_orbital", "M_spinless_layer_exchange"}:
            raise ValueError("C2 toy generator requires Gamma_four_orbital or M_spinless_layer_exchange basis_template")
    elif source_type == "kp_symm_output":
        if not symmetry_source.get("path"):
            raise ValueError("symmetry_source.type=kp_symm_output requires path")
        missing = sym_names - source_ops
        if missing and source_ops:
            raise ValueError(f"symmetry_source is missing matrices for operations: {sorted(missing)}")
    elif source_type != "none":
        raise ValueError(f"Unsupported symmetry_source.type: {source_type}")

    if "exactification" in symmetry_source:
        raise ValueError(
            "symmetry_source source matrix projection is internal; remove symmetry_source.exactification from model config"
        )

    term_templates = model.get("term_templates", raw.get("term_templates", []))
    if term_templates is None:
        term_templates = []
    if not isinstance(term_templates, Sequence) or isinstance(term_templates, (str, bytes)):
        raise ValueError("term_templates must be a list when provided")
    for idx, template in enumerate(term_templates):
        if not isinstance(template, Mapping):
            raise ValueError(f"term_templates[{idx}] must be a mapping")
        monomial_constraints = template.get("monomial_constraints")
        if monomial_constraints is not None and not isinstance(monomial_constraints, Mapping):
            raise ValueError(f"term_templates[{idx}].monomial_constraints must be a mapping")
        if "legacy_monomial_filter" in template or "monomial_filter" in template:
            raise ValueError("monomial_filter is not supported by the model schema; use monomial_constraints")
        generation_mode = str(template.get("generation_mode", "")).strip()
        if generation_mode:
            if generation_mode not in ALLOWED_GENERATION_MODES:
                allowed = ", ".join(sorted(ALLOWED_GENERATION_MODES))
                raise ValueError(f"term_templates[{idx}].generation_mode must be one of: {allowed}; got {generation_mode!r}")
        else:
            lower_name = str(template.get("name", "")).lower()
            if "legacy" in lower_name:
                raise ValueError(f"term_templates[{idx}] uses a legacy-style name; use a neutral term name")

    bM = model.get("bM", {})
    if isinstance(bM, Mapping):
        source = str(bM.get("source", "auto"))
        if bM.get("infer_from_q", False):
            source = "q_distance"
        if valley_type in {"K", "M"} and source in {"q", "q_distance", "q_distances"}:
            sectors = raw.get("sectors", [])
            has_offsets = (
                isinstance(sectors, Sequence)
                and not isinstance(sectors, (str, bytes))
                and all(isinstance(sector, Mapping) and "q_offset" in sector for sector in sectors)
                and len(sectors) > 0
            )
            if not has_offsets:
                raise ValueError("K/M q_distance bM requires explicit sectors with q_offset; prefer bM.source: tmat")

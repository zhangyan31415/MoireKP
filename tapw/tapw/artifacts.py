"""Canonical artifact names shared by TAPW writers and readers."""

from __future__ import annotations


def _normalized_band_edge(kind) -> str:
    text = str(kind).strip().lower()
    if text in {"vbm", "valence"}:
        return "vbm"
    if text in {"cbm", "conduction"}:
        return "cbm"
    raise ValueError(f"Unsupported band edge: {kind!r}")


def _normalized_kind(kind: str) -> str:
    value = str(kind)
    if "_" not in value:
        return value.upper() if value.lower() in {"cbm", "vbm"} else value.lower()
    prefix, rest = value.split("_", 1)
    return f"{prefix.lower()}_{rest.upper()}"


def run_dir_name(compute_cfg, *, symmetrized: bool = False) -> str:
    if not bool(getattr(compute_cfg, "TAPW", True)):
        return "direct"
    name = f"Q_shell_{getattr(compute_cfg, 'n_g')}"
    if symmetrized:
        return name + "_symm"
    return name


def band_output_filename(kind: str, *, valley_flag: str, suffix: str = "", tapw: bool = True) -> str:
    kind = _normalized_kind(kind)
    suffix = str(suffix)
    if tapw:
        return f"band_{kind}_{valley_flag}_valley{suffix}.txt"
    return f"band_{kind}{suffix}.txt"


def array_output_filename(kind: str, *, valley_flag: str, suffix: str = "", tapw: bool = True) -> str:
    kind = _normalized_kind(kind)
    suffix = str(suffix)
    if tapw:
        return f"{kind}_{valley_flag}_valley{suffix}.npy"
    return f"{kind}{suffix}.npy"


def berry_flux_output_filename(kind: str, *, valley_flag: str, suffix: str = "", tapw: bool = True) -> str:
    kind = _normalized_kind(kind)
    suffix = str(suffix)
    if tapw:
        return f"berry_flux_{kind}_{valley_flag}_valley{suffix}.npy"
    return f"berry_flux_{kind}{suffix}.npy"


def chern_summary_output_filename(kind: str, *, valley_flag: str, suffix: str = "", tapw: bool = True) -> str:
    kind = _normalized_kind(kind)
    suffix = str(suffix)
    if tapw:
        return f"chern_summary_{kind}_{valley_flag}_valley{suffix}.txt"
    return f"chern_summary_{kind}{suffix}.txt"


def raw_memmap_filename(kind: str, *, valley_flag: str, suffix: str = "") -> str:
    kind = str(kind).lower()
    suffix = str(suffix)
    return f"{kind}_raw_{valley_flag}_valley{suffix}.npy"


def chern_flux_filename(*, band_type: str, valley_flag: str, band_label: str, suffix: str = "") -> str:
    band_type = str(band_type).upper()
    suffix = str(suffix)
    return f"berry_flux_{band_type}_{valley_flag}_valley{suffix}_{band_label}.npy"


def chern_summary_filename(
    *,
    band_type: str | None = None,
    valley_flag: str | None = None,
    suffix: str = "",
    tapw: bool = True,
) -> str:
    if band_type is None or valley_flag is None:
        return "chern_summary.json"
    band_type = str(band_type).upper()
    suffix = str(suffix)
    if tapw:
        return f"chern_summary_{band_type}_{valley_flag}_valley{suffix}.json"
    return f"chern_summary_{band_type}{suffix}.json"


def gvec_output_filename(*, n_g, valley_flag: str, layer: int) -> str:
    return f"g_vec_list_{n_g}_{valley_flag}_{int(layer)}layer.npy"


def canonical_qshell_name(n_g) -> str:
    text = str(n_g).strip()
    if text.lower().startswith("q"):
        text = text[1:]
    return f"q{int(text):02d}"


def canonical_profile_name(valley_flag: str, *, spin="spinful", profile: str | None = None) -> str:
    if profile not in (None, ""):
        return str(profile)
    base = str(valley_flag)
    spin_text = str(spin).strip().lower()
    if spin is True or spin_text in {"true", "all", "spinful", "both", "full"}:
        return base
    if spin is False or spin_text in {"false", "spinless", "spinless_effective"}:
        return f"{base}_spinless"
    if spin_text in {"up", "spin_up", "spin_up_projected"}:
        return f"{base}_up"
    if spin_text in {"down", "spin_down", "spin_down_projected"}:
        return f"{base}_down"
    return f"{base}_{spin_text.replace('-', '_')}"


def canonical_band_filename(kind: str, edge_or_group) -> str:
    kind_text = str(kind).strip().lower()
    if kind_text == "energies":
        return f"energies_{_normalized_band_edge(edge_or_group)}.txt"
    if kind_text == "wavefunctions":
        return f"wavefunctions_{_normalized_band_edge(edge_or_group)}.npy"
    if kind_text in {"hamiltonian", "hamiltonian_k", "hamk"}:
        return "hamiltonian_k.npy"
    if kind_text in {"g_vectors", "gvec", "qset"}:
        return f"g_vectors_group{int(edge_or_group)}.npy"
    if kind_text == "kpoints":
        return "kpoints.npy"
    raise ValueError(f"Unsupported canonical band artifact kind: {kind!r}")


def _canonical_range_token(value) -> str:
    number = float(value)
    text = f"{number:.12g}"
    if "e" in text.lower():
        text = f"{number:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return text.replace("-", "m").replace(".", "p")


def canonical_topology_grid_name(
    num_k1: int,
    num_k2: int,
    *,
    range_b1=(-0.5, 0.5),
    range_b2=(-0.5, 0.5),
) -> str:
    base = f"grid{int(num_k1)}x{int(num_k2)}"
    r1 = (float(range_b1[0]), float(range_b1[1]))
    r2 = (float(range_b2[0]), float(range_b2[1]))
    return (
        f"{base}_b1_{_canonical_range_token(r1[0])}_{_canonical_range_token(r1[1])}"
        f"_b2_{_canonical_range_token(r2[0])}_{_canonical_range_token(r2[1])}"
    )


def canonical_topology_band_token(index: int) -> str:
    value = int(index)
    return f"m{abs(value)}" if value < 0 else str(value)


def canonical_topology_band_label(indices) -> str:
    values = [int(value) for value in indices]
    if not values:
        raise ValueError("At least one band index is required")
    tokens = [canonical_topology_band_token(value) for value in values]
    prefix = "band" if len(tokens) == 1 else "bands"
    return f"{prefix}_{'_'.join(tokens)}"

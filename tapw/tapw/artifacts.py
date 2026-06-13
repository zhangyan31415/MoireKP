"""Canonical artifact names shared by TAPW writers and readers."""


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

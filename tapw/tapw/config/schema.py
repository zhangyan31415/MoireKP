from dataclasses import asdict, dataclass, field
from typing import Any, List, Dict, Optional, Sequence, Tuple, Union
import yaml
from pathlib import Path
import os

from ..artifacts import canonical_qshell_name


_GridLike = Union[int, Sequence[int]]
_SymmetrizeSpec = Union[bool, List[str]]
CANONICAL_HAMILTONIAN_SYMMETRY_OPERATIONS = ("C3z", "C2", "C2T", "TR")
LEGACY_HAMILTONIAN_SYMMETRY_OPERATION_ALIASES = {"T": "TR"}
VALLEY_LABEL_TO_INT = {
    "K1": 1,
    "K2": 2,
    "K1_120": 11,
    "K1_240": 12,
    "GAMMA": 5,
    "Γ": 5,
    "M": 3,
    "M1": 31,
    "M2": 32,
    "M3": 33,
    "X": 41,
    "Y": 42,
}


def _parse_valley(value) -> int:
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    if text.lstrip("+-").isdigit():
        return int(text)
    key = text.upper()
    if key not in VALLEY_LABEL_TO_INT:
        raise ValueError(f"Unknown TAPW valley {value!r}; use labels such as K1, Gamma, M1 or an integer id.")
    return VALLEY_LABEL_TO_INT[key]


def _parse_q_shell(value) -> int:
    text = str(value).strip()
    if text.lower().startswith("q"):
        text = text[1:]
    return int(text)


def _copy_section(value) -> dict[str, Any]:
    return dict(value or {})


_CANONICAL_SYSTEM_KEYS = {
    "output",
    "structure",
    "hamiltonian",
    "overlap",
    "orbitals",
    "twist_index",
    "layers",
    "spin",
}
_CANONICAL_SYSTEM_REQUIRED_KEYS = _CANONICAL_SYSTEM_KEYS - {"overlap"}
_RELEASE_TOP_LEVEL_KEYS = {
    "system",
    "case",
    "twist",
    "paths",
    "bands",
    "symmetry",
    "topology",
    "field",
    "cluster",
    "symmetry_analysis",
    # Kept here so existing targeted error messages remain stable.
    "compute",
    "output_layout",
    "slab",
}


@dataclass(frozen=True)
class SystemInputConfig:
    """Deeply immutable normalized TAPW source input shared by every workflow."""

    source_kind: str
    output: Path
    structure: Path
    hamiltonian: Path
    overlap: Optional[Path]
    orbitals: Optional[Tuple[Tuple[str, str], ...]]
    twist_index: int
    layers: Tuple[int, ...]
    spin: bool
    explicit_bravais: Optional[str]

    def __post_init__(self) -> None:
        if self.source_kind not in {"canonical_structure", "legacy_openmx"}:
            raise ValueError(f"Unsupported system input source_kind={self.source_kind!r}.")
        for key in ("output", "structure", "hamiltonian"):
            if not isinstance(getattr(self, key), Path):
                raise TypeError(f"system_input.{key} must be a resolved pathlib.Path.")
        if self.overlap is not None and not isinstance(self.overlap, Path):
            raise TypeError("system_input.overlap must be a resolved pathlib.Path or None.")
        if self.source_kind == "canonical_structure" and not self.orbitals:
            raise ValueError("Canonical system.orbitals must be a non-empty species-to-orbitals mapping.")
        if self.source_kind == "legacy_openmx" and self.orbitals is not None:
            raise ValueError("Legacy OpenMX orbitals are source-owned and must not be synthesized in system_input.")
        if self.twist_index < 1:
            raise ValueError("system.twist_index must be a positive integer.")
        if not self.layers or any(value < 1 for value in self.layers):
            raise ValueError("system.layers must be a non-empty list of positive integers.")
        if not isinstance(self.spin, bool):
            raise ValueError("system.spin must be true or false.")

    @property
    def orbital_mapping(self) -> Optional[Dict[str, str]]:
        return None if self.orbitals is None else dict(self.orbitals)


def _resolved_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _normalize_canonical_system(
    config_dict: dict[str, Any],
    *,
    config_dir: Path,
) -> SystemInputConfig | None:
    if "system" not in config_dict:
        return None
    conflicting = [name for name in ("case", "twist", "paths") if name in config_dict]
    if conflicting:
        raise ValueError(
            "Canonical system input cannot be mixed with legacy section(s): "
            f"{conflicting}"
        )
    raw = config_dict.get("system")
    if not isinstance(raw, dict):
        raise ValueError("system must be a mapping.")
    unknown = sorted(set(raw) - _CANONICAL_SYSTEM_KEYS)
    if unknown:
        raise ValueError(f"system contains unknown field(s): {unknown}")
    missing = sorted(_CANONICAL_SYSTEM_REQUIRED_KEYS - set(raw))
    if missing:
        raise ValueError(f"system is missing required field(s): {missing}")
    orbitals = raw.get("orbitals")
    if not isinstance(orbitals, dict) or not orbitals:
        raise ValueError("system.orbitals must be a non-empty species-to-orbitals mapping.")
    layers = raw.get("layers")
    if not isinstance(layers, list):
        raise ValueError("system.layers must be a list of positive integers.")
    if not isinstance(raw.get("spin"), bool):
        raise ValueError("system.spin must be true or false.")
    return SystemInputConfig(
        source_kind="canonical_structure",
        output=_resolved_path(raw["output"], config_dir),
        structure=_resolved_path(raw["structure"], config_dir),
        hamiltonian=_resolved_path(raw["hamiltonian"], config_dir),
        overlap=None if raw.get("overlap") in (None, "") else _resolved_path(raw["overlap"], config_dir),
        orbitals=tuple(sorted((str(species), str(spec)) for species, spec in orbitals.items())),
        twist_index=int(raw["twist_index"]),
        layers=tuple(int(value) for value in layers),
        spin=raw["spin"],
        explicit_bravais=None,
    )


def _reject_keys(section: dict[str, Any], keys: set[str], *, section_name: str) -> None:
    present = sorted(keys & set(section))
    if present:
        raise ValueError(f"{section_name} contains removed release parameter(s): {present}")


def _section_enabled(section: dict[str, Any]) -> bool:
    return bool(section) and bool(section.get("enable", True))


def _require_release_workflow_fields(section: dict[str, Any], *, section_name: str) -> None:
    if not section:
        return
    if not _section_enabled(section):
        return
    missing = [key for key in ("valley", "q_shell") if section.get(key) in (None, "")]
    if missing:
        raise ValueError(f"{section_name} enabled workflow requires field(s): {missing}")


def _normalise_release_kpath(kpath: Any) -> dict[str, Any] | None:
    if kpath in (None, ""):
        return None
    if not isinstance(kpath, dict):
        raise ValueError("bands.kpath must be a mapping with labels, points_per_segment, and coordinates.")
    labels = kpath.get("labels")
    coordinates = kpath.get("coordinates")
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)) or len(labels) < 2:
        raise ValueError("bands.kpath.labels must contain at least two labels.")
    if not isinstance(coordinates, dict) or not coordinates:
        raise ValueError("bands.kpath.coordinates must map each label to a two- or three-component coordinate.")
    normalised_labels = [str(label) for label in labels]
    missing = [label for label in normalised_labels if label not in coordinates]
    if missing:
        raise ValueError(f"bands.kpath.coordinates is missing label(s): {missing}")
    normalised_coordinates: dict[str, list[float]] = {}
    for label in normalised_labels:
        raw = coordinates[label]
        values = [float(value) for value in raw]
        if len(values) == 2:
            values.append(0.0)
        if len(values) != 3:
            raise ValueError(f"bands.kpath.coordinates.{label} must have two or three values.")
        normalised_coordinates[label] = values
    points = int(kpath.get("points_per_segment", 40))
    if points <= 0:
        raise ValueError("bands.kpath.points_per_segment must be positive.")
    return {
        "labels": normalised_labels,
        "points_per_segment": points,
        "coordinates": normalised_coordinates,
    }


def _apply_release_section_to_mapping(target: dict[str, Any], section: dict[str, Any], *, workflow: str) -> None:
    """Map release-facing workflow fields onto the internal ComputeConfig names."""
    if not section:
        return
    if "valley" in section:
        target["valleys"] = [_parse_valley(section["valley"])]
    if "q_shell" in section:
        target["n_g"] = _parse_q_shell(section["q_shell"])
    if "efermi" in section:
        target["efermi"] = float(section["efermi"])
    if "num_processes" in section:
        target["num_processes"] = int(section["num_processes"])
    if "blas_threads" in section:
        target["blas_threads"] = int(section["blas_threads"])
    if "num_bands" in section:
        target["num_bands_cal"] = int(section["num_bands"])
    if "num_bands_cal" in section:
        target["num_bands_cal"] = int(section["num_bands_cal"])
    if "save_hamiltonian" in section:
        target["hamk_save"] = bool(section["save_hamiltonian"])
    if "save_wavefunctions" in section:
        target["eig_vec_cal"] = bool(section["save_wavefunctions"])
    if "use_sparse_dot_mkl" in section:
        target["use_sparse_dot_mkl"] = bool(section["use_sparse_dot_mkl"])
    if workflow == "chern":
        mesh = _copy_section(section.get("mesh"))
        if mesh:
            if "n_b1" in mesh:
                target["num_k1"] = int(mesh["n_b1"])
            if "n_b2" in mesh:
                target["num_k2"] = int(mesh["n_b2"])
        band_type, indices = _primary_topology_band_request(section)
        if indices is not None:
            target["chern_band_indices"] = indices
            target["band_type"] = band_type


def _apply_release_section_to_object(target, section: dict[str, Any], *, workflow: str) -> None:
    updates: dict[str, Any] = {}
    _apply_release_section_to_mapping(updates, dict(section or {}), workflow=workflow)
    for key, value in updates.items():
        setattr(target, key, value)


def _band_type_from_release_sector(sector) -> str:
    normalized = str(sector).strip().lower()
    if normalized in {"valence", "vbm"}:
        return "VBM"
    if normalized in {"conduction", "cbm"}:
        return "CBM"
    raise ValueError(f"Unknown band sector {sector!r}; use valence/VBM or conduction/CBM.")


def _primary_topology_band_request(topology: dict[str, Any]) -> tuple[str, Optional[list[int]]]:
    """Return the first requested topology bandset as internal `(band_type, indices)`."""
    bands = dict(topology.get("bands", {}) or {})
    task_refs: list[Any] = []
    for key in ("berry_curvature", "quantum_geometry", "wcc"):
        value = topology.get(key)
        if value is None:
            continue
        if isinstance(value, dict):
            task_refs.append(value)
        elif isinstance(value, list):
            task_refs.extend(value)
        else:
            task_refs.append(value)
    if not task_refs and bands:
        task_refs.append(next(iter(bands)))
    if not task_refs:
        return "BOTH", None
    first = task_refs[0]
    if isinstance(first, str):
        spec = dict(bands.get(first, {}))
    elif isinstance(first, dict):
        ref = first.get("bands", first.get("bandset"))
        spec = dict(bands.get(ref, {})) if isinstance(ref, str) else dict(first)
    else:
        raise ValueError("topology observable entries must be band-set names or mappings.")
    if "indices" not in spec:
        raise ValueError("topology bandset requires indices.")
    if "band_type" in spec:
        raise ValueError("topology bandsets use sector: valence/conduction; band_type is not supported.")
    band_type = _band_type_from_release_sector(spec.get("sector", ""))
    return band_type, [int(value) for value in spec["indices"]]


def _copy_release_topology_config(raw_topology: dict[str, Any]) -> dict[str, Any]:
    """Return release-facing topology config and reject removed compatibility shapes."""
    topology = dict(raw_topology or {})
    removed = {"bandsets", "observables"} & set(topology)
    if removed:
        raise ValueError(f"topology contains removed release parameter(s): {sorted(removed)}")
    return topology


def normalize_symmetrize_hamiltonian(value) -> _SymmetrizeSpec:
    """Normalize the Hamiltonian-symmetrization request from release configs."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        raise ValueError(
            "compute.symmetrize_hamiltonian must be false, true, or a list of canonical "
            "operation names such as [C3z, C2T]. Use C3z, not C3."
        )
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("compute.symmetrize_hamiltonian operation list must not be empty; use false.")
        normalized = []
        allowed = set(CANONICAL_HAMILTONIAN_SYMMETRY_OPERATIONS)
        for item in value:
            if not isinstance(item, str):
                raise ValueError("compute.symmetrize_hamiltonian operation names must be strings.")
            op = LEGACY_HAMILTONIAN_SYMMETRY_OPERATION_ALIASES.get(item.strip(), item.strip())
            if op not in allowed:
                raise ValueError(
                    f"Invalid Hamiltonian symmetrization operation {op!r}; use canonical names "
                    f"{list(CANONICAL_HAMILTONIAN_SYMMETRY_OPERATIONS)}. Use C3z, not C3; use TR, not T."
                )
            if op not in normalized:
                normalized.append(op)
        return normalized
    raise ValueError(
        "compute.symmetrize_hamiltonian must be false, true, or a list of canonical operation names."
    )


def hamiltonian_symmetrization_requested(value) -> bool:
    return normalize_symmetrize_hamiltonian(value) is not False


def _coerce_legacy_num_chern(num_chern: _GridLike) -> Tuple[int, int]:
    """Normalize legacy `num_chern` input to an explicit `(num_k1, num_k2)` pair."""
    if isinstance(num_chern, (list, tuple)):
        if len(num_chern) != 2:
            raise ValueError("num_chern as a list/tuple must contain exactly two integers, e.g. [30, 50]")
        return int(num_chern[0]), int(num_chern[1])
    return int(num_chern), int(num_chern)


def resolve_chern_grid_shape(num_chern: _GridLike, num_k1: Optional[int] = None, num_k2: Optional[int] = None) -> tuple:
    """Resolve the fractional Chern/Wilson-loop vertex grid dimensions.

    `num_chern` accepts the legacy scalar square-grid form (`40`) and also a
    two-entry list/tuple (`[30, 50]`). Explicit `num_k1`/`num_k2` still take
    precedence when present.
    """
    legacy_k1, legacy_k2 = _coerce_legacy_num_chern(num_chern)
    k1 = int(num_k1 if num_k1 is not None else legacy_k1)
    k2 = int(num_k2 if num_k2 is not None else legacy_k2)
    if k1 < 2 or k2 < 2:
        raise ValueError("Chern/Wilson-loop grids require num_k1 >= 2 and num_k2 >= 2")
    return k1, k2


def format_chern_grid_suffix(num_k1: int, num_k2: int) -> str:
    """Return a stable output suffix for a Chern-mode grid."""
    num_k1 = int(num_k1)
    num_k2 = int(num_k2)
    if num_k1 == num_k2:
        return "_2d_{0}".format(num_k1)
    return "_2d_{0}x{1}".format(num_k1, num_k2)


@dataclass
class TwistConfig:
    """Configuration for twisted materials"""
    twist_index_m: int
    bravais: str = "hex"  # Bravais lattice type: "hex", "square", or "rect"
    num_layers: int = 2
    type_structure: List[int] = field(default_factory=lambda: [2, 2])
    twist_layer: List[int] = field(default_factory=lambda: [1, 1])
    spin: bool = True

    def __post_init__(self):
        allowed = {"hex", "square", "rect"}
        if self.bravais not in allowed:
            raise ValueError(f"Invalid bravais lattice type: {self.bravais}. Must be one of {sorted(allowed)}")

@dataclass
class PathConfig:
    """Configuration for file paths"""
    H_file: str
    input_file: str
    output_dir: str
    kpath_in: Optional[str] = None
    kpath_out: Optional[str] = None
    S_file: Optional[str] = None

    def __post_init__(self):
        return None

    @staticmethod
    def _resolve_path(path_value: str, base_dir: Path) -> str:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return str(path.resolve())
        return str((base_dir / path).resolve())

    def normalize(self, base_dir: Path) -> None:
        """Resolve relative path fields against the configuration file directory."""
        base_dir = Path(base_dir).expanduser()
        if not base_dir.is_absolute():
            base_dir = Path.cwd() / base_dir

        self.H_file = self._resolve_path(self.H_file, base_dir)
        self.input_file = self._resolve_path(self.input_file, base_dir)
        self.output_dir = self._resolve_path(self.output_dir, base_dir)
        if self.kpath_in is not None:
            self.kpath_in = self._resolve_path(self.kpath_in, base_dir)
        if self.kpath_out is not None:
            self.kpath_out = self._resolve_path(self.kpath_out, base_dir)
        if self.S_file is not None:
            self.S_file = self._resolve_path(self.S_file, base_dir)

@dataclass
class ClusterConfig:
    """Configuration for clustering parameters with fixed values"""
    layer_eps: float = 0.5
    layer_min_samples: int = 1
    sublayer_eps: float = 0.5
    sublayer_min_samples: int = 1
    atom_eps: float = 0.3
    atom_min_samples: int = 2
    period: float = 2 * 3.14159
    k_max: int = 2

    def __post_init__(self):
        # These parameters are fixed and should not be changed
        return None

@dataclass
class SymmetryAnalysisConfig:
    """Configuration for the optional TAPW symmetry-analysis mode."""

    enable: bool = False
    valleys: Optional[List[int]] = None
    tolerance: float = 1.0e-2
    spglib_symprec: Optional[float] = None
    validation: str = "full"
    output_dir: str = "symmetry_analysis"
    debug: bool = False
    developer_outputs: bool = False

    def __post_init__(self):
        mode = str(self.validation).strip().lower().replace("-", "_")
        if mode != "full":
            raise ValueError(
                f"Invalid symmetry.validation={self.validation!r}. Release symmetry requires 'full' covariance validation."
            )
        self.validation = mode


@dataclass
class OutputLayoutConfig:
    """Optional release-facing output layout."""

    style: str = "legacy"
    root: str = ""
    profile: Optional[str] = None
    q_shell: Optional[str] = None

    @staticmethod
    def _resolve_path(path_value: str, base_dir: Path) -> str:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return str(path.resolve())
        return str((base_dir / path).resolve())

    def normalize(self, base_dir: Path) -> None:
        self.style = str(self.style).strip().lower()
        if self.root:
            self.root = self._resolve_path(self.root, base_dir)

    def validate(self) -> None:
        allowed = {"legacy", "canonical_v1"}
        if self.style not in allowed:
            raise ValueError(f"Invalid output_layout.style={self.style!r}. Must be one of {sorted(allowed)}")
        if self.style == "canonical_v1" and not self.root:
            raise ValueError("output_layout.root is required when output_layout.style=canonical_v1")

@dataclass
class ComputeConfig:
    """Configuration for computation parameters"""
    valleys: List[int] = field(default_factory=lambda: [31, 32, 33])  # List of valleys to calculate
    mode: str = "band"  # Calculation mode: "band" for band structure, "chern" for Chern number
    efermi: float = -0.17
    n_g: int = 6
    num_processes: int = 50
    blas_threads: int = 1  # BLAS/OpenMP threads per worker; 0 leaves thread pools unrestricted.
    parallel_impl: str = "joblib"  # "joblib" or "mp"
    parallel_backend: str = "loky"  # joblib backend: "loky" (spawn) or "multiprocessing" (fork on Linux)
    tapw_auto_fork: bool = False  # Optional: for TAPW on POSIX, upgrade default joblib/loky k-loop to mp/fork.
    vec_store: str = "memory"  # "memory" or "memmap" (recommended for large k-mesh + wavefunctions)
    memmap_dir: Optional[str] = None  # If set, store memmap outputs here; otherwise use output path
    kpoint_chunk_id: int = 0  # For job-array sharding: 0-based chunk index
    kpoint_chunk_count: int = 1  # For job-array sharding: total number of chunks
    fast_getk: bool = True  # Faster CSR build in Getk_super_gauge_sparse (recommended)
    use_sparse_dot_mkl: bool = True  # Use sparse_dot_mkl for TAPW sparse projection.
    eigensolver: str = "scipy"  # non-TAPW generalized solver: "scipy" or "slepc" (SLEPc tuning is internal)
    slepc_eps_type: str = "krylovschur"  # e.g. "krylovschur", "jd", "lapack" (small problems)
    slepc_st_type: str = "sinvert"  # spectral transform: "sinvert" is typical for interior eigenvalues
    slepc_ksp_type: str = "preonly"  # linear solver for ST; "preonly" + "lu" is robust but memory-heavy
    slepc_pc_type: str = "lu"  # preconditioner: "lu" / "ilu" / "gamg" etc.
    slepc_factor_mat_solver_type: str = ""  # e.g. "mumps", "superlu_dist"
    slepc_tol: float = 1e-8
    slepc_max_it: int = 5000
    slepc_comm: str = "self"  # "self" (COMM_SELF, k-point workers) or "world" (COMM_WORLD, MPI parallel per k-point)
    slepc_make_hermitian: bool = True  # symmetrize H(k),S(k) as (A+A^H)/2 to satisfy GHEP assumptions
    slepc_spd_shift: float = 0.0  # optional diagonal shift added to S(k) to improve definiteness (e.g. 1e-10)
    num_bands_cal: int = 50
    num_chern: _GridLike = 40  # Legacy scalar square-grid size, or a 2-entry list like [num_k1, num_k2].
    num_k1: Optional[int] = None  # Fractional reciprocal-grid points along kappa1 (falls back to num_chern).
    num_k2: Optional[int] = None  # Fractional reciprocal-grid points along kappa2 (falls back to num_chern).
    chern_band_indices: Optional[List[int]] = None  # Band indices used by mode='chern'; negative indices are allowed.
    band_type: str = "BOTH"  # Band subset to save: "CBM", "VBM", or "BOTH"
    hamk_save: bool = False
    TAPW: bool = True
    eigsh_cal: bool = True
    symmetrize_hamiltonian: _SymmetrizeSpec = False
    ge: bool = False
    eig_vec_cal: bool = True
    valley: int = field(init=False)  # Current valley being calculated
    valley_flag: str = field(init=False)  # Valley flag for display
    solve_flag: str = field(init=False)  # Solver flag
    eq_flag: str = field(init=False)  # Equation flag
    symm_flag: str = field(init=False)  # Symmetry flag
    Electric_field_in_eVpA: Optional[float] = None  # 电场强度 (eV/Å)
    zero_potential_layers: Optional[List[int]] = None  # 选择的层数（用于确定零势能面）
    Inner_symmetrical_Electric_Field: bool = False  # 是否加内对称电场
    orthogonal_basis: bool = False  # 是否使用正交基底，正交时S矩阵可以省略

    def validate(self) -> None:
        """Validate runtime-related options that may be overridden after YAML load."""
        allowed_modes = {"band", "chern", "symmetry"}
        if self.mode not in allowed_modes:
            raise ValueError(
                f"Invalid mode={self.mode!r}. Must be one of {sorted(allowed_modes)}"
            )
        if self.mode == "chern" and self.ge:
            raise ValueError(
                "Chern mode with ge=true is not supported in this release because saved generalized "
                "eigenvectors require overlap-metric Berry phases, not Euclidean overlaps."
            )

        allowed_parallel_impl = {"joblib", "mp"}
        if self.parallel_impl not in allowed_parallel_impl:
            raise ValueError(
                f"Invalid parallel_impl={self.parallel_impl!r}. Must be one of {sorted(allowed_parallel_impl)}"
            )
        allowed_backend = {"loky", "multiprocessing"}
        if self.parallel_backend not in allowed_backend:
            raise ValueError(
                f"Invalid parallel_backend={self.parallel_backend!r}. Must be one of {sorted(allowed_backend)}"
            )
        allowed_vec_store = {"memory", "memmap"}
        if self.vec_store not in allowed_vec_store:
            raise ValueError(
                f"Invalid vec_store={self.vec_store!r}. Must be one of {sorted(allowed_vec_store)}"
            )
        if self.kpoint_chunk_count < 1:
            raise ValueError("kpoint_chunk_count must be >= 1")
        if not (0 <= self.kpoint_chunk_id < self.kpoint_chunk_count):
            raise ValueError(
                f"kpoint_chunk_id must be in [0, {self.kpoint_chunk_count - 1}], got {self.kpoint_chunk_id}"
            )
        if self.num_processes < 1:
            raise ValueError("num_processes must be >= 1")
        if self.blas_threads < 0:
            raise ValueError("blas_threads must be >= 0")
        self.get_chern_grid_shape()

        allowed_eigensolver = {"scipy", "slepc"}
        if self.eigensolver not in allowed_eigensolver:
            raise ValueError(
                f"Invalid eigensolver={self.eigensolver!r}. Must be one of {sorted(allowed_eigensolver)}"
            )
        if not self.TAPW:
            if not self.ge:
                raise ValueError("non-TAPW band calculations require ge=true.")
            if self.eig_vec_cal:
                raise ValueError("non-TAPW band calculations currently require eig_vec_cal=false.")
            if self.eigensolver == "slepc":
                allowed_slepc_comm = {"self", "world"}
                if self.slepc_comm not in allowed_slepc_comm:
                    raise ValueError(
                        f"Invalid slepc_comm={self.slepc_comm!r}. Must be one of {sorted(allowed_slepc_comm)}"
                    )

                # SLEPc is MPI-based; mixing it with forked Python workers is unsafe.
                # We only support joblib+loky (spawn) for now.
                if self.parallel_impl != "joblib" or self.parallel_backend != "loky":
                    raise ValueError(
                        "eigensolver='slepc' requires parallel_impl='joblib' and parallel_backend='loky' "
                        "(spawn). Do not use multiprocessing/fork with MPI libraries."
                    )
                if self.slepc_comm == "world":
                    # In COMM_WORLD mode we expect the user to launch with MPI (srun/mpiexec -n N).
                    # Do not also spawn local workers or do k-point chunking; it can deadlock or duplicate work.
                    if self.num_processes != 1:
                        raise ValueError("slepc_comm='world' requires num_processes=1 (no k-point multiprocessing).")
                    if self.kpoint_chunk_count != 1 or self.kpoint_chunk_id != 0:
                        raise ValueError("slepc_comm='world' requires kpoint_chunk_count=1 and kpoint_chunk_id=0.")
        self.symmetrize_hamiltonian = normalize_symmetrize_hamiltonian(self.symmetrize_hamiltonian)
        if self.symmetrize_hamiltonian is not False and not self.TAPW:
            raise ValueError("compute.symmetrize_hamiltonian requires TAPW=true.")

    def get_chern_grid_shape(self) -> tuple:
        """Return the explicit fractional Chern-grid shape `(num_k1, num_k2)`."""
        return resolve_chern_grid_shape(
            num_chern=self.num_chern,
            num_k1=self.num_k1,
            num_k2=self.num_k2,
        )

    def get_chern_grid_suffix(self) -> str:
        """Return the filename suffix used for Chern-mode raw/final outputs."""
        num_k1, num_k2 = self.get_chern_grid_shape()
        return format_chern_grid_suffix(num_k1, num_k2)

    @staticmethod
    def valley_label(valley: int) -> str:
        valley_flag = {
            1: "K1", 2: "K2",
            11: "K1_120", 12: "K1_240",
            # Square/rect high-symmetry points
            5: "Gamma",
            3: "M",
            41: "X",
            42: "Y",
            31: "M1", 32: "M2", 33: "M3"
        }
        return valley_flag.get(valley, "unknown")

    def set_valley(self, valley: int) -> None:
        """Set the active valley and refresh derived display flags."""
        self.valley = int(valley)
        self.valley_flag = self.valley_label(self.valley)

    def __post_init__(self):
        self.validate()
        # Solver mapping
        solve_flag = {True: "eigsh", False: "lapack"}
        # Equation mapping
        eq_flag = {True: "ge", False: "st"}
        # Symmetry mapping
        symm_flag = {True: "symm", False: "nsymm"}

        # Set initial valley if not already set
        if not hasattr(self, 'valley') and self.valleys:
            self.valley = self.valleys[0]

        # Set flags
        self.valley_flag = self.valley_label(self.valley)
        self.solve_flag = solve_flag[self.eigsh_cal]
        self.eq_flag = eq_flag[self.ge]
        self.symm_flag = symm_flag[self.symmetrize_hamiltonian is not False]
        
@dataclass
class Config:
    """Main configuration class"""
    twist: TwistConfig
    paths: PathConfig
    compute: ComputeConfig
    system_input: SystemInputConfig
    symmetry_analysis: SymmetryAnalysisConfig = field(default_factory=SymmetryAnalysisConfig)
    output_layout: Optional[OutputLayoutConfig] = None
    case: Dict[str, Any] = field(default_factory=dict)
    bands: Dict[str, Any] = field(default_factory=dict)
    symmetry: Dict[str, Any] = field(default_factory=dict)
    topology: Dict[str, Any] = field(default_factory=dict)
    kpath: Optional[Dict[str, Any]] = None
    cluster: ClusterConfig = field(default_factory=ClusterConfig)  # Use default values if not provided
    field_config: Dict[str, Any] = field(default_factory=dict)
    resolved_structure_input: Optional[Any] = field(default=None, init=False, repr=False, compare=False)

    def apply_workflow_section(self, mode: Optional[str] = None) -> None:
        """Apply release-facing workflow section fields to the internal runtime config."""
        mode = str(mode or self.compute.mode)
        section_by_mode = {
            "band": ("bands", self.bands),
            "chern": ("topology", self.topology),
            "symmetry": ("symmetry", self.symmetry),
        }
        if mode not in section_by_mode:
            return
        _, section = section_by_mode[mode]
        section = dict(section or {})
        self.compute.mode = mode
        if mode == "chern":
            # Topology often reuses band diagonalization controls. Use bands as defaults,
            # then let topology override valley/q-shell/mesh/bandsets.
            _apply_release_section_to_object(self.compute, self.bands, workflow="band")
        _apply_release_section_to_object(self.compute, section, workflow=mode)
        if self.compute.valleys:
            self.compute.set_valley(self.compute.valleys[0])
        self.compute.validate()
        if self.output_layout is not None and getattr(self.output_layout, "style", "") == "canonical_v1":
            self.output_layout.q_shell = canonical_qshell_name(self.compute.n_g)

    def validate(self) -> None:
        """Validate cross-section configuration constraints."""
        self.compute.validate()
        if self.output_layout is not None:
            self.output_layout.validate()
        if (
            self.compute.mode in {"band", "chern"}
            and not self.compute.orthogonal_basis
            and not self.paths.S_file
        ):
            raise ValueError(
                "paths.S_file is required for non-orthogonal band/chern calculations. "
                "Set compute.orthogonal_basis=true for orthogonal bases or use mode='symmetry'."
            )
        if self.compute.mode == "band" and self.kpath is None and not self.paths.kpath_in:
            raise ValueError("TAPW band workflow requires bands.kpath in release configs.")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file"""
        config_path = Path(yaml_path).expanduser()
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        config_dir = config_path.parent

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        if not isinstance(config_dict, dict):
            raise ValueError("TAPW configuration root must be a mapping.")
        unknown_top_level = sorted(set(config_dict) - _RELEASE_TOP_LEVEL_KEYS)
        if unknown_top_level:
            raise ValueError(f"TAPW config contains unknown top-level section(s): {unknown_top_level}")
        system_input = _normalize_canonical_system(config_dict, config_dir=config_dir)
        if 'slab' in config_dict:
            raise ValueError("The release TAPW package does not support slab configuration.")
        if config_dict.get("output_layout") is not None:
            raise ValueError("output_layout is not supported in release-only TAPW configs; use case.output_root.")
        if config_dict.get("compute") not in (None, {}):
            raise ValueError("Top-level compute is not supported in release-only TAPW configs; use bands/symmetry/topology.")
        case_raw = _copy_section(config_dict.get("case"))
        if system_input is not None:
            case_raw = {"output_root": str(system_input.output)}
        bands_raw = _copy_section(config_dict.get("bands"))
        symmetry_raw = _copy_section(config_dict.get("symmetry"))
        field_raw = _copy_section(config_dict.get("field"))
        topology_config = _copy_release_topology_config(config_dict.get("topology", {}) or {})
        release_sections_present = bool(case_raw or bands_raw or symmetry_raw or topology_config)
        if not release_sections_present:
            raise ValueError("Release TAPW configs require case plus at least one of bands, symmetry, or topology.")
        if case_raw.get("output_root") in (None, ""):
            raise ValueError("case.output_root is required in release-only TAPW configs.")
        for section_name, section in (
            ("bands", bands_raw),
            ("symmetry", symmetry_raw),
            ("topology", topology_config),
        ):
            _require_release_workflow_fields(section, section_name=section_name)
        if not any(_section_enabled(section) for section in (bands_raw, symmetry_raw, topology_config)):
            raise ValueError("Release TAPW configs require at least one enabled workflow section.")
        forbidden_user_keys = {"band_type", "eigensolver", "orthogonal_basis", "num_layers"}
        _reject_keys(bands_raw, forbidden_user_keys, section_name="bands")
        _reject_keys(symmetry_raw, forbidden_user_keys, section_name="symmetry")
        _reject_keys(topology_config, forbidden_user_keys, section_name="topology")
        mesh = topology_config.get("mesh")
        if isinstance(mesh, dict):
            _reject_keys(mesh, forbidden_user_keys, section_name="topology.mesh")
        for bandset_name, bandset in dict(topology_config.get("bands", {}) or {}).items():
            if isinstance(bandset, dict):
                _reject_keys(dict(bandset), forbidden_user_keys, section_name=f"topology.bands.{bandset_name}")
        kpath_config = _normalise_release_kpath(bands_raw.get("kpath"))

        compute_raw: dict[str, Any] = {}
        if release_sections_present:
            if _section_enabled(bands_raw):
                compute_raw.setdefault("mode", "band")
            elif _section_enabled(topology_config):
                compute_raw.setdefault("mode", "chern")
            elif _section_enabled(symmetry_raw):
                compute_raw.setdefault("mode", "symmetry")
            else:
                compute_raw.setdefault("mode", "band")
            compute_raw.setdefault("TAPW", True)
            if _section_enabled(bands_raw):
                _apply_release_section_to_mapping(compute_raw, bands_raw, workflow="band")
            if "n_g" not in compute_raw:
                fallback_section = topology_config if _section_enabled(topology_config) else symmetry_raw
                fallback_workflow = "chern" if _section_enabled(topology_config) else "symmetry"
                _apply_release_section_to_mapping(compute_raw, fallback_section, workflow=fallback_workflow)
            if "zero_potential_layers" in field_raw:
                compute_raw["zero_potential_layers"] = field_raw["zero_potential_layers"]
            if "electric_field_eVpA" in field_raw:
                compute_raw["Electric_field_in_eVpA"] = float(field_raw["electric_field_eVpA"])
            if "inner_symmetric" in field_raw:
                compute_raw["Inner_symmetrical_Electric_Field"] = bool(field_raw["inner_symmetric"])
        removed = {'gpu', 'gpu_index', 'delay_time'} & set(compute_raw)
        if removed:
            raise ValueError(f"The release TAPW package does not support GPU options: {sorted(removed)}")
        removed_symmetry = {'C3_H', 'M_valley_D3_H'} & set(compute_raw)
        if removed_symmetry:
            raise ValueError(
                f"Removed TAPW Hamiltonian symmetrization option(s) {sorted(removed_symmetry)}. "
                "Use compute.symmetrize_hamiltonian instead."
            )
        if compute_raw.get('TAPW', True) and 'n_g' not in compute_raw:
            raise ValueError(
                "Enabled TAPW workflows require q_shell. "
                "Automatic q-shell inference from twist.twist_angle is not supported."
            )

        twist_raw = dict(config_dict.get('twist', {}) or {})
        explicit_legacy_bravais = twist_raw.get("bravais") if "bravais" in twist_raw else None
        if system_input is not None:
            twist_raw = {
                "twist_index_m": system_input.twist_index,
                "twist_layer": list(system_input.layers),
                "spin": system_input.spin,
            }
        if "num_layers" in twist_raw:
            raise ValueError("twist.num_layers is not supported in release-only TAPW configs; use twist.twist_layer.")
        if "num_layers" not in twist_raw and "twist_layer" in twist_raw:
            twist_raw["num_layers"] = int(sum(int(value) for value in twist_raw["twist_layer"]))
        twist_config = TwistConfig(**twist_raw)

        paths_raw = dict(config_dict.get('paths', {}) or {})
        if system_input is not None:
            paths_raw = {
                "H_file": str(system_input.hamiltonian),
                "S_file": None if system_input.overlap is None else str(system_input.overlap),
                "input_file": str(system_input.structure),
                "output_dir": str(system_input.output),
            }
        if paths_raw.get("kpath_out") not in (None, ""):
            raise ValueError("paths.kpath_out is not supported in release-only TAPW configs; it is inferred from case/output.")
        if "output_dir" not in paths_raw and case_raw.get("output_root") not in (None, ""):
            paths_raw["output_dir"] = case_raw["output_root"]
        paths_config = PathConfig(**paths_raw)
        paths_config.normalize(config_dir)
        if system_input is None:
            system_input = SystemInputConfig(
                source_kind="legacy_openmx",
                output=Path(paths_config.output_dir),
                structure=Path(paths_config.input_file),
                hamiltonian=Path(paths_config.H_file),
                overlap=None if paths_config.S_file is None else Path(paths_config.S_file),
                orbitals=None,
                twist_index=int(twist_config.twist_index_m),
                layers=tuple(int(value) for value in twist_config.twist_layer),
                spin=bool(twist_config.spin),
                explicit_bravais=None if explicit_legacy_bravais is None else str(explicit_legacy_bravais),
            )
        if "orthogonal_basis" not in compute_raw and paths_raw.get("S_file") in (None, ""):
            compute_raw["orthogonal_basis"] = True
        compute_config = ComputeConfig(**compute_raw)
        compute_config.topology = topology_config
        symmetry_analysis_raw = dict(config_dict.get('symmetry_analysis', {}) or {})
        if symmetry_raw:
            if "enable" in symmetry_raw:
                symmetry_analysis_raw["enable"] = bool(symmetry_raw["enable"])
            if "valley" in symmetry_raw:
                symmetry_analysis_raw["valleys"] = [_parse_valley(symmetry_raw["valley"])]
            for key in ("tolerance", "spglib_symprec", "validation", "debug", "developer_outputs"):
                if key in symmetry_raw:
                    symmetry_analysis_raw[key] = symmetry_raw[key]
        symmetry_analysis_config = SymmetryAnalysisConfig(**symmetry_analysis_raw)
        output_layout_config = OutputLayoutConfig(
            style="canonical_v1",
            root=str(paths_config.output_dir),
            q_shell=canonical_qshell_name(compute_config.n_g),
        )
        # Use default cluster config if not provided
        cluster_config = ClusterConfig(**config_dict.get('cluster', {})) if 'cluster' in config_dict else ClusterConfig()
        
        config_obj = cls(
            twist=twist_config,
            paths=paths_config,
            compute=compute_config,
            symmetry_analysis=symmetry_analysis_config,
            output_layout=output_layout_config,
            case=case_raw,
            bands=bands_raw,
            symmetry=symmetry_raw,
            topology=topology_config,
            kpath=kpath_config,
            cluster=cluster_config,
            system_input=system_input,
            field_config=field_raw,
        )
        # Propagate twist bravais to compute for downstream logic
        config_obj.compute.bravais = config_obj.twist.bravais
        config_obj.release_sections_present = release_sections_present
        if release_sections_present:
            config_obj.apply_workflow_section(config_obj.compute.mode)
        config_obj.validate()
        return config_obj

    def save_yaml(self, yaml_path: str):
        """Save a release-facing YAML that can be loaded again without compatibility fields."""
        destination = Path(yaml_path).expanduser()
        if not destination.is_absolute():
            destination = Path.cwd() / destination
        destination_dir = destination.parent

        def portable(path_value: str) -> str:
            return os.path.relpath(str(Path(path_value)), start=str(destination_dir))

        config_dict: dict[str, Any] = {}
        if self.system_input.source_kind == "canonical_structure":
            orbitals = self.system_input.orbital_mapping
            if orbitals is None:
                raise ValueError("Canonical system_input is missing orbital metadata.")
            config_dict["system"] = {
                "output": portable(self.system_input.output),
                "structure": portable(self.system_input.structure),
                "hamiltonian": portable(self.system_input.hamiltonian),
                "orbitals": orbitals,
                "twist_index": int(self.system_input.twist_index),
                "layers": list(self.system_input.layers),
                "spin": bool(self.system_input.spin),
            }
            if self.system_input.overlap is not None:
                config_dict["system"]["overlap"] = portable(self.system_input.overlap)
        else:
            case = dict(self.case or {})
            case["output_root"] = portable(self.paths.output_dir)
            config_dict["case"] = case
            config_dict["twist"] = {
                "twist_index_m": int(self.twist.twist_index_m),
                "twist_layer": list(self.twist.twist_layer),
                "spin": bool(self.twist.spin),
            }
            if self.system_input.explicit_bravais is not None:
                config_dict["twist"]["bravais"] = self.system_input.explicit_bravais
            paths = {
                "H_file": portable(self.paths.H_file),
                "input_file": portable(self.paths.input_file),
            }
            if self.paths.S_file:
                paths["S_file"] = portable(self.paths.S_file)
            if self.paths.kpath_in:
                paths["kpath_in"] = portable(self.paths.kpath_in)
            config_dict["paths"] = paths
        for name in ("bands", "symmetry", "topology"):
            section = dict(getattr(self, name, {}) or {})
            if section:
                config_dict[name] = section
        if self.field_config:
            config_dict["field"] = dict(self.field_config)
        config_dict["cluster"] = asdict(self.cluster)
        config_dict["symmetry_analysis"] = asdict(self.symmetry_analysis)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config_dict, handle, sort_keys=False)

    def update_ng(self):
        """Reject legacy automatic n_g inference in release configs."""
        raise ValueError(
            "Automatic n_g inference from twist.twist_angle is not supported. "
            "Set compute.n_g explicitly."
        )

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Sequence, Tuple, Union
import yaml
from pathlib import Path


_GridLike = Union[int, Sequence[int]]
_SymmetrizeSpec = Union[bool, List[str]]
CANONICAL_HAMILTONIAN_SYMMETRY_OPERATIONS = ("C3z", "C2", "C2T", "TR")
LEGACY_HAMILTONIAN_SYMMETRY_OPERATION_ALIASES = {"T": "TR"}


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
    kpath_in: str
    kpath_out: str
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
        self.kpath_in = self._resolve_path(self.kpath_in, base_dir)
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
    output_dir: str = "symmetry_analysis"
    debug: bool = False
    developer_outputs: bool = False

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
    use_sparse_dot_mkl: bool = False  # Use sparse_dot_mkl for g@H@g^H (can help or hurt depending on sizes/threads)
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
    symmetry_analysis: SymmetryAnalysisConfig = field(default_factory=SymmetryAnalysisConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)  # Use default values if not provided

    def validate(self) -> None:
        """Validate cross-section configuration constraints."""
        self.compute.validate()
        if (
            self.compute.mode in {"band", "chern"}
            and not self.compute.orthogonal_basis
            and not self.paths.S_file
        ):
            raise ValueError(
                "paths.S_file is required for non-orthogonal band/chern calculations. "
                "Set compute.orthogonal_basis=true for orthogonal bases or use mode='symmetry'."
            )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file"""
        config_path = Path(yaml_path).expanduser()
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        config_dir = config_path.parent

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        if 'slab' in config_dict:
            raise ValueError("The release TAPW package does not support slab configuration.")
        compute_raw = config_dict.get('compute', {})
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
                "TAPW configurations require compute.n_g. "
                "Automatic n_g inference from twist.twist_angle is not supported by Config.from_yaml."
            )
        
        twist_config = TwistConfig(**config_dict.get('twist', {}))
        paths_config = PathConfig(**config_dict.get('paths', {}))
        paths_config.normalize(config_dir)
        compute_config = ComputeConfig(**compute_raw)
        symmetry_analysis_config = SymmetryAnalysisConfig(**config_dict.get('symmetry_analysis', {}))
        # Use default cluster config if not provided
        cluster_config = ClusterConfig(**config_dict.get('cluster', {})) if 'cluster' in config_dict else ClusterConfig()
        
        config_obj = cls(
            twist=twist_config,
            paths=paths_config,
            compute=compute_config,
            symmetry_analysis=symmetry_analysis_config,
            cluster=cluster_config
        )
        # Propagate twist bravais to compute for downstream logic
        config_obj.compute.bravais = config_obj.twist.bravais
        config_obj.validate()
        return config_obj

    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'twist': self.twist.__dict__,
            'paths': {k: v for k, v in self.paths.__dict__.items() if not k.startswith('_')},
            'compute': {k: v for k, v in self.compute.__dict__.items() if k != 'valley'},
            'symmetry_analysis': self.symmetry_analysis.__dict__,
            # Don't save cluster config as it uses fixed values
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def update_ng(self):
        """Reject legacy automatic n_g inference in release configs."""
        raise ValueError(
            "Automatic n_g inference from twist.twist_angle is not supported. "
            "Set compute.n_g explicitly."
        )

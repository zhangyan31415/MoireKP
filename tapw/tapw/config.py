from dataclasses import dataclass, field
from typing import List, Dict, Optional, Sequence, Tuple, Union
import os
import yaml
from pathlib import Path


_GridLike = Union[int, Sequence[int]]


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
        self.H_file = os.path.abspath(self.H_file)
        if self.S_file is not None:
            self.S_file = os.path.abspath(self.S_file)
        self.output_dir = os.path.abspath(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        # 新增：如果不是正交基底，S_file 必须存在
        # from .config import ComputeConfig
        import inspect
        # 尝试获取调用栈中的Config对象
        frame = inspect.currentframe()
        while frame:
            local_vars = frame.f_locals
            if 'self' in local_vars and hasattr(local_vars['self'], 'compute'):
                compute = local_vars['self'].compute
                break
            frame = frame.f_back
        else:
            compute = None
        if compute is not None and not getattr(compute, 'orthogonal_basis', False):
            if self.S_file is None or not os.path.isfile(self.S_file):
                raise ValueError("S_file must be provided and exist when not using orthogonal basis.")

@dataclass
class SlabConfig:
    """Configuration for slab calculations"""
    nslab: int = 7  # Number of layers in the slab
    ijmax: int = 1  # Maximum interlayer distance (for twisted materials)
    slab_direction: List[int] = field(default_factory=lambda: [0, 1, 0])  # Slab normal direction vector [x,y,z]
    analyze_surface: bool = True  # Whether to analyze surface character
    surface_threshold: float = 0.5  # Threshold for identifying surface states
    kpath_slab_in: str = "KPATH_SLAB.in"  # Slab k-path input file
    kpath_slab_out: str = "KPATH_SLAB.out"  # Slab k-path output file
    
    def __post_init__(self):
        """Validate slab direction vector"""
        if len(self.slab_direction) != 3:
            raise ValueError("slab_direction must be a 3-element vector [x, y, z]")
        
        # Convert to list of integers
        self.slab_direction = [int(x) for x in self.slab_direction]
        
        # Check that exactly one component is non-zero
        non_zero_count = sum(1 for x in self.slab_direction if x != 0)
        if non_zero_count != 1:
            raise ValueError("slab_direction must have exactly one non-zero component (e.g., [0,1,0] for y-direction)")
        
        # Get direction index and name for convenience
        self.direction_index = self.slab_direction.index(max(self.slab_direction, key=abs))
        direction_names = ['x', 'y', 'z']
        self.direction_name = direction_names[self.direction_index]
        
        print(f"Slab direction: {self.slab_direction} ({self.direction_name}-direction, index {self.direction_index})")

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
        pass

@dataclass
class SymmetryAnalysisConfig:
    """Configuration for the optional TAPW symmetry-analysis mode."""

    enable: bool = False
    valleys: Optional[List[int]] = None
    tolerance: float = 1.0e-2
    output_dir: str = "symmetry_analysis"
    debug: bool = False

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
    tapw_auto_fork: bool = False  # Opt-in: for TAPW on POSIX, upgrade default joblib/loky k-loop to mp/fork.
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
    band_type: str = "BOTH"  # Band subset to save: "CBM", "VBM", or "BOTH"
    gpu: bool = False
    gpu_index: List[int] = field(default_factory=lambda: [0, 1])
    delay_time: int = 0  # seconds; stagger start a little to reduce I/O spikes (max delay is ~(workers-1)*delay_time)
    hamk_save: bool = False
    TAPW: bool = True
    eigsh_cal: bool = True
    C3_H: bool = False
    M_valley_D3_H: bool = False  # For hex TAPW M valleys, add the single-M C2 projection after the C3 orbit average.
    ge: bool = False
    eig_vec_cal: bool = True
    valley: int = field(init=False)  # Current valley being calculated
    valley_flag: str = field(init=False)  # Valley flag for display
    solve_flag: str = field(init=False)  # Solver flag
    eq_flag: str = field(init=False)  # Equation flag
    symm_flag: str = field(init=False)  # Symmetry flag
    gpu_num: int = field(init=False)  # Number of GPUs
    Electric_field_in_eVpA: Optional[float] = None  # 电场强度 (eV/Å)
    zero_potential_layers: Optional[List[int]] = None  # 选择的层数（用于确定零势能面）
    Inner_symmetrical_Electric_Field: bool = False  # 是否加内对称电场
    orthogonal_basis: bool = False  # 是否使用正交基底，正交时S矩阵可以省略

    def validate(self) -> None:
        """Validate runtime-related options that may be overridden after YAML load."""
        allowed_modes = {"band", "chern", "slab", "symmetry"}
        if self.mode not in allowed_modes:
            raise ValueError(
                f"Invalid mode={self.mode!r}. Must be one of {sorted(allowed_modes)}"
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
                if self.eig_vec_cal:
                    raise ValueError("eigensolver='slepc' currently supports eig_vec_cal=false only.")

                if self.slepc_comm == "world":
                    # In COMM_WORLD mode we expect the user to launch with MPI (srun/mpiexec -n N).
                    # Do not also spawn local workers or do k-point chunking; it can deadlock or duplicate work.
                    if self.num_processes != 1:
                        raise ValueError("slepc_comm='world' requires num_processes=1 (no k-point multiprocessing).")
                    if self.kpoint_chunk_count != 1 or self.kpoint_chunk_id != 0:
                        raise ValueError("slepc_comm='world' requires kpoint_chunk_count=1 and kpoint_chunk_id=0.")

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

    def __post_init__(self):
        self.validate()

        # Valley mapping
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
        self.valley_flag = valley_flag.get(self.valley, "unknown")
        self.solve_flag = solve_flag[self.eigsh_cal]
        self.eq_flag = eq_flag[self.ge]
        self.symm_flag = symm_flag[self.C3_H]
        
        # Set GPU number
        self.gpu_num = len(self.gpu_index) #if self.gpu else 0

        # Set GPU environment if using GPU
        if self.gpu:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(idx) for idx in self.gpu_index)
            print(f"Using GPU with index {self.gpu_index}")

@dataclass
class Config:
    """Main configuration class"""
    twist: TwistConfig
    paths: PathConfig
    compute: ComputeConfig
    symmetry_analysis: SymmetryAnalysisConfig = field(default_factory=SymmetryAnalysisConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)  # Use default values if not provided
    slab: Optional[SlabConfig] = None  # Slab configuration (only used when mode="slab")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        twist_config = TwistConfig(**config_dict.get('twist', {}))
        paths_config = PathConfig(**config_dict.get('paths', {}))
        compute_config = ComputeConfig(**config_dict.get('compute', {}))
        symmetry_analysis_config = SymmetryAnalysisConfig(**config_dict.get('symmetry_analysis', {}))
        # Use default cluster config if not provided
        cluster_config = ClusterConfig(**config_dict.get('cluster', {})) if 'cluster' in config_dict else ClusterConfig()
        
        # Load slab config if mode is "slab" or if slab section exists
        slab_config = None
        if (compute_config.mode == "slab"):
            slab_config = SlabConfig(**config_dict.get('slab', {}))
        
        config_obj = cls(
            twist=twist_config,
            paths=paths_config,
            compute=compute_config,
            symmetry_analysis=symmetry_analysis_config,
            cluster=cluster_config,
            slab=slab_config
        )
        # Propagate twist bravais to compute for downstream logic
        config_obj.compute.bravais = config_obj.twist.bravais
        # 如果config.yaml没有n_g字段，则自动调用update_ng
        if 'n_g' not in config_dict.get('compute', {}):
            config_obj.update_ng()
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
        
        # Add slab config if it exists
        if self.slab is not None:
            config_dict['slab'] = self.slab.__dict__
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def update_ng(self):
        """Update n_g based on twist angle"""
        angle = float(self.twist.twist_angle)
        if angle > 9:
            self.compute.n_g = 3
        elif angle > 6:
            # self.compute.efermi = -0.17
            self.compute.n_g = 6
        elif angle > 4:
            # self.compute.efermi = -0.172
            self.compute.n_g = 7
        elif angle > 2.8:
            # self.compute.efermi = -0.178
            self.compute.n_g = 8
        elif angle > 2:
            self.compute.efermi = -0.18
            self.compute.n_g = 9
        elif angle > 1:
            self.compute.efermi = -0.18
            self.compute.n_g = 10 

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os
import yaml
from pathlib import Path

@dataclass
class TwistConfig:
    """Configuration for twisted materials"""
    twist_index_m: int
    num_layers: int = 2
    type_structure: List[int] = field(default_factory=lambda: [2, 2])
    twist_layer: List[int] = field(default_factory=lambda: [1, 1])
    spin: bool = True

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
class ComputeConfig:
    """Configuration for computation parameters"""
    valleys: List[int] = field(default_factory=lambda: [31, 32, 33])  # List of valleys to calculate
    mode: str = "band"  # Calculation mode: "band" for band structure, "chern" for Chern number
    efermi: float = -0.17
    n_g: int = 6
    num_processes: int = 50
    num_bands_cal: int = 50
    num_chern: int = 40
    band_type: str = "CBM"  # Band type to analyze: "CBM" for conduction band minimum, "VBM" for valence band maximum
    gpu: bool = False
    gpu_index: List[int] = field(default_factory=lambda: [0, 1])
    delay_time: int = 4
    hamk_save: bool = False
    TAPW: bool = True
    eigsh_cal: bool = True
    C3_H: bool = False
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
    def __post_init__(self):
        # Valley mapping
        valley_flag = {
            1: "K1", 2: "K2", 
            11: "K1_120", 12: "K1_240", 
            5: "Gamma",
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
            cluster=cluster_config,
            slab=slab_config
        )
        # 如果config.yaml没有n_g字段，则自动调用update_ng
        if 'n_g' not in config_dict.get('compute', {}):
            config_obj.update_ng()
        return config_obj

    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'twist': self.twist.__dict__,
            'paths': {k: v for k, v in self.paths.__dict__.items() if not k.startswith('_')},
            'compute': {k: v for k, v in self.compute.__dict__.items() if k != 'valley'}
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
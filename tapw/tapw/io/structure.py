import numpy as np
import re
from collections import defaultdict
import os
import tempfile


import pandas as pd
import scipy.linalg
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import scipy.sparse as sp
import scipy
import pandas as pd
from scipy.spatial import cKDTree

from ..reporting import ensure_reporter

# Plotting is optional for non-plot workflows; keep matplotlib import failure from breaking core logic.
try:
    import matplotlib.pyplot as plt
    import matplotlib
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
except Exception as _matplotlib_import_error:  # pragma: no cover
    plt = None
    matplotlib = None
    inset_axes = None


def reciprocal_from_Tmat(Tmat: np.ndarray) -> np.ndarray:
    """
    Compute reciprocal lattice matrix from a real-space lattice matrix.

    Convention in this codebase:
    - `Tmat` is a (3,3) matrix whose *row vectors* are real-space lattice vectors a1,a2,a3.
    - A fractional coordinate `r_frac` is converted to Cartesian by `r_cart = Tmat.T @ r_frac`.

    We return `B_row` as a (3,3) matrix whose *row vectors* are reciprocal vectors b1,b2,b3,
    satisfying b_i · a_j = 2π δ_ij.
    """
    Tmat = np.asarray(Tmat, dtype=np.float64)
    if Tmat.shape != (3, 3):
        raise ValueError(f"Tmat must be shape (3,3), got {Tmat.shape}")

    # Convert to column-vector convention, compute reciprocal, then convert back.
    # A_col has columns a1,a2,a3; B_col has columns b1,b2,b3.
    A_col = Tmat.T
    B_col = 2 * np.pi * np.linalg.inv(A_col).T
    B_row = B_col.T
    return B_row


class OpenMXFile:
    def __init__(self, file_path, twist_index, spin):
        self.file_path = file_path
        self.twist_index = twist_index
        # Bravais lattice of the moiré supercell. Default is "hex" to preserve legacy behaviour.
        # The square moiré workflow relies on automatic detection from supercell (a1,a2) as fallback.
        self.bravais = "hex"
        self._bravais_forced = False

        self.twist_angle = None  # degrees (float)
        self.twist_angle_deg = None  # degrees (float), explicit name for downstream consistency
        self.num_unit_cell = None
        self.spin = spin

        self.Tmat = np.zeros((3, 3), dtype=np.float64)
        self.reciprocal_Tmat = np.zeros((3, 3), dtype=np.float64)
        self.atoms_number = 0
        self.species_coordinates_unit = None
        self.species_coordinates = []
        self.sorted_species_coordinates = []
        self.atom_basis = {}
        self.species_count = {}
        self.orbitals_count = {}
        self.permutation_matrix = None

        self.unit_cell_atom_orb = {} # 
        self.df = None

        # Optional override from env var (cannot rely on main.py passing parameters).
        bravais_env = os.environ.get("TAPW_BRAVAIS")
        if bravais_env:
            self.set_bravais(bravais_env)

        self.calc_twist_angle()
        self.calc_num_unit_cell()
        self.parse_file()

        # Fallback automatic detection from the parsed moiré supercell lattice.
        prev_bravais = self.bravais
        self._auto_detect_bravais_from_Tmat()
        if self.bravais != prev_bravais:
            self.calc_twist_angle()
            self.calc_num_unit_cell()

        # Propagate detected bravais for other components (e.g. LayeredLatticeAnalyzer) without
        # changing main.py call signatures.
        os.environ.setdefault("TAPW_BRAVAIS", self.bravais)

        self.sort_atoms_by_z()
        # self.compute_permutation_matrix()
        self.count_species()
        self.count_orbitals()
        self.get_dataframe()


    def set_bravais(self, bravais: str):
        """Public setter for bravais lattice type ("hex" or "square")."""
        if bravais is None:
            return
        bravais_norm = str(bravais).strip().lower()
        if bravais_norm in {"hex", "hexagonal"}:
            self.bravais = "hex"
        elif bravais_norm in {"square", "sq"}:
            self.bravais = "square"
        else:
            raise ValueError(f"Unsupported bravais='{bravais}'. Expected 'hex' or 'square'.")
        self._bravais_forced = True

    def _auto_detect_bravais_from_Tmat(self):
        """
        Fallback automatic detection (only when bravais is not forced by user/env).

        Rule:
        - Use supercell in-plane vectors a1,a2 from Tmat (row vectors).
        - If a1 ⟂ a2 and |a1|==|a2| within 1e-3 relative tolerance -> square.
        """
        if self._bravais_forced:
            return
        if not isinstance(self.Tmat, np.ndarray) or self.Tmat.shape != (3, 3):
            return
        a1 = np.asarray(self.Tmat[0, :2], dtype=np.float64)
        a2 = np.asarray(self.Tmat[1, :2], dtype=np.float64)
        n1 = float(np.linalg.norm(a1))
        n2 = float(np.linalg.norm(a2))
        if n1 == 0.0 or n2 == 0.0:
            return
        cos_abs = float(np.abs(np.dot(a1, a2)) / (n1 * n2))
        rel_len = float(np.abs(n1 - n2) / n1)
        if cos_abs < 1.0e-3 and rel_len < 1.0e-3:
            self.bravais = "square"

    def calc_twist_angle(self):
        """
        计算扭转角度。
        """
        m = int(self.twist_index)
        if m < 1:
            raise ValueError(f"twist_index must be >= 1, got {self.twist_index}")

        if getattr(self, "bravais", "hex") == "square":
            # (m, m+1) family for square moiré:
            #   theta = 2 * arctan( 1 / (2*m + 1) )   [radians]
            theta_rad = 2.0 * np.arctan(1.0 / (2.0 * m + 1.0))
            theta_deg = float(np.degrees(theta_rad))
        else:
            # Legacy (hex) commensurate family
            cos_ang = (3 * m ** 2 + 3 * m + 0.5) / (3 * m ** 2 + 3 * m + 1)
            cos_ang = float(np.clip(cos_ang, -1.0, 1.0))
            theta_deg = float(np.degrees(np.arccos(cos_ang)))

        self.twist_angle_deg = theta_deg
        self.twist_angle = theta_deg

    def calc_num_unit_cell(self):
        """
        计算单位晶胞的数量。
        """
        m = int(self.twist_index)
        if getattr(self, "bravais", "hex") == "square":
            # Square coincidence lattice index for tan(theta/2)=1/(2m+1): Σ = (2m+1)^2 + 1
            self.num_unit_cell = int((2 * m + 1) ** 2 + 1)
        else:
            self.num_unit_cell = int(3 * m ** 2 + 3 * m + 1)

    def parse_file(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()

        # Flags to identify sections
        in_species_def = False
        in_atoms_coords = False
        in_unit_vectors = False

        for line in lines:
            line = line.strip()

            # Parse Definition.of.Atomic.Species
            if line.startswith("<Definition.of.Atomic.Species"):
                in_species_def = True
                continue

            if in_species_def:
                if line.endswith(">"):
                    in_species_def = False
                    continue
                else:
                    # 示例行: Mo    Mo7.0-s3p2d1      Mo_PBE19
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 3:
                        species = parts[0]
                        orbitals = parts[1]
                        self.atom_basis[species] = orbitals
                continue

            # Parse Atoms.Number
            if line.startswith("Atoms.Number"):
                parts = line.split()
                if len(parts) >= 2:
                    self.atoms_number = int(parts[1])
                continue

            # Parse Atoms.SpeciesAndCoordinates.Unit
            if line.startswith("Atoms.SpeciesAndCoordinates.Unit"):
                parts = line.split()
                if len(parts) >= 2:
                    self.species_coordinates_unit = parts[1]
                continue

            # Parse Atoms.SpeciesAndCoordinates
            if line.startswith("<Atoms.SpeciesAndCoordinates"):
                in_atoms_coords = True
                continue
            if in_atoms_coords:
                if line.endswith(">"):
                    in_atoms_coords = False
                    continue
                else:
                    # 示例行:
                    # 1     Te    0.8301890   0.1202700   0.1948840    8.0  8.0
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 7:

                        atom = {
                            'original_index': int(parts[0]),
                            'species': parts[1],
                            'r': np.array([float(part) for part in parts[2:5]]),
                            'x': float(parts[2]),
                            'y': float(parts[3]),
                            'z': float(parts[4]),
                            'orb_num': 0,
                            'orb_name': '',
                            'orb_global_index': 0,
                        }
                        self.species_coordinates.append(atom)
                continue

            # Parse Atoms.UnitVectors
            if line.startswith("<Atoms.UnitVectors"):
                in_unit_vectors = True
                self.Tmat = []
                continue
            if in_unit_vectors:
                if line.endswith(">"):
                    in_unit_vectors = False
                    continue
                else:
                    # 示例行: 21.4328000     0.0000000     0.0000000
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 3:
                        vector = [float(part) for part in parts[:3]]
                        self.Tmat.append(vector)
                continue

        # 转换单位向量为 numpy 数组
        if self.Tmat:
            self.Tmat = np.array(self.Tmat)
            self.reciprocal_Tmat = reciprocal_from_Tmat(self.Tmat)

        # 转换坐标为直角坐标
        if self.species_coordinates_unit[0].upper() == 'F':
            for i, atom in enumerate(self.species_coordinates):
                cart_coords = self.frac_to_cart_real(self.Tmat, atom['r'])
                self.species_coordinates[i]['x'] = float(cart_coords[0])
                self.species_coordinates[i]['y'] = float(cart_coords[1])
                self.species_coordinates[i]['z'] = float(cart_coords[2])
        # 删除 'r' 字段
        for i in range(len(self.species_coordinates)):
            self.species_coordinates[i].pop('r', None)
        

    def cart_to_frac_real(self, Amat, pos_cart):
        return np.dot(np.linalg.inv(Amat.T), np.array(pos_cart))

    def frac_to_cart_real(self, Amat, pos_frac):
        return np.dot(Amat.T, np.array(pos_frac))

    def sort_atoms_by_z(self):
        """
        按 z 轴对原子进行排序，并存储排序后的列表。
        """
        self.sorted_species_coordinates = sorted(self.species_coordinates, key=lambda atom: atom['z'])

    def compute_permutation_matrix(self):
        """
        计算排序前后原子索引变换的置换矩阵。
        """
        # 创建一个从原始索引到排序后索引的映射
        original_indices = [atom['original_index'] for atom in self.species_coordinates]
        sorted_indices = [atom['original_index'] for atom in self.sorted_species_coordinates]

        # 创建一个字典，键为原始索引，值为排序后的位置
        index_mapping = {original: sorted_pos for sorted_pos, original in enumerate(sorted_indices)}

        # 初始化置换矩阵
        N = self.atoms_number
        P = np.zeros((N, N), dtype=int)

        for new_pos, original in enumerate(original_indices):
            sorted_pos = index_mapping[original]
            if sorted_pos < N and new_pos < N:
                P[new_pos, sorted_pos] = 1

        self.permutation_matrix = P

    def count_species(self):
        """
        统计每种元素的原子数量。
        """
        species_count = defaultdict(int)
        for atom in self.species_coordinates:
            species_count[atom['species']] += 1
        self.species_count = dict(species_count)

    def count_orbitals(self):
        """
        计算每种元素的轨道数量，包括可选的 s、p、d 和 f 轨道。
        
        如果某个元素没有定义任何轨道，将抛出一个 ValueError。
        
        返回:
        - orbitals_count: dict
        """
        orbitals_count = {}
        for species, definition in self.atom_basis.items():
            # 示例定义: Te7.0-s3p2d2f1 或 Mo7.0-s3p2d1 或 Mo7.0-s3 或 Cu6.0H-s2p2d1 或 Cu6.0S-s2p1d1 等
            # 正则表达式匹配 s、p、d、f 轨道的数量，p、d、f 为可选
            # 注意：元素符号后可能包含字母后缀（如 H、S 等）
            match = re.match(
                r'^[A-Za-z]+[\d\.]+[A-Za-z]*(?:-s(?P<s>\d+))?(?:p(?P<p>\d+))?(?:d(?P<d>\d+))?(?:f(?P<f>\d+))?$',
                definition
            )

            if match:
                s = int(match.group('s')) if match.group('s') else 0
                p = int(match.group('p')) if match.group('p') else 0
                d = int(match.group('d')) if match.group('d') else 0
                f = int(match.group('f')) if match.group('f') else 0

                # 检查至少有一个轨道类型被定义
                if s == 0 and p == 0 and d == 0 and f == 0:
                    raise ValueError(f"元素 {species} 的轨道定义中没有任何轨道类型被定义。定义内容: '{definition}'")

                # 计算总轨道数
                total_orbitals = s * 1 + p * 3 + d * 5 + f * 7
                orbitals_count[species] = total_orbitals
            else:
                # 如果轨道定义格式不匹配，则抛出错误
                raise ValueError(f"无法解析元素 {species} 的轨道定义。定义内容: '{definition}'")

        self.orbitals_count = orbitals_count
        global_orbital_counter = 0
        for i,atom in enumerate(self.species_coordinates):
            atom['orb_num'] = self.orbitals_count[atom['species']]
            atom['orb_name'] = self.atom_basis[atom['species']]
            atom['orb_global_index'] = list(range(global_orbital_counter, global_orbital_counter + atom['orb_num']))
            global_orbital_counter += atom['orb_num']
            self.species_coordinates[i] = atom

    def get_dataframe(self):
        """
        返回一个 Pandas DataFrame，包含所有原子的信息。
        """
        import pandas as pd

        df = pd.DataFrame(self.species_coordinates)
        # return df
        self.df = df
    

    
    def get_sorted_atoms(self):
        """
        返回排序后的原子列表。
        """
        return self.sorted_species_coordinates

    def get_original_atoms(self):
        """
        返回原始的原子列表。
        """
        return self.species_coordinates

    def display_properties(self, reporter=None):
        """
        Print all user-facing structure properties.
        """
        reporter = ensure_reporter(reporter)
        reporter.stage("Structure overview", "Parsed moire supercell and source orbital metadata.")
        reporter.kv("Bravais (moire supercell)", self.bravais)
        reporter.kv("Twist angle from twist_index_m (deg)", self.twist_angle)
        reporter.array("Moire lattice vectors (Angstrom)", self.Tmat)
        reporter.array("Moire reciprocal vectors (1/Angstrom)", self.reciprocal_Tmat)
        if self.bravais == "square":
            b1 = np.asarray(self.reciprocal_Tmat[0][:2], dtype=np.float64)
            b2 = np.asarray(self.reciprocal_Tmat[1][:2], dtype=np.float64)
            reporter.section("Square reciprocal self-check")
            reporter.kv("b1 dot b2", f"{float(np.dot(b1, b2)):.6e}")
            reporter.kv("|b1| - |b2|", f"{float(np.linalg.norm(b1) - np.linalg.norm(b2)):.6e}")

            reporter.section("Square moire BZ high-symmetry points (1/Angstrom)")
            Gamma = np.array([0.0, 0.0])
            X = 0.5 * b1
            Y = 0.5 * b2
            M = 0.5 * (b1 + b2)
            reporter.kv("Gamma", Gamma)
            reporter.kv("X", X)
            reporter.kv("Y", Y)
            reporter.kv("M", M)
        reporter.kv("Atom count", self.atoms_number)
        reporter.kv("Coordinate unit", self.species_coordinates_unit)
        # print("\n=== 按 z 轴排序前的前5个原子 ===")
        # for atom in self.species_coordinates[:5]:
        #     print(atom)
        # print("\n=== 按 z 轴排序后的前5个原子 ===")
        # for atom in self.sorted_species_coordinates[:5]:
        #     print(atom)
        reporter.kv("Species counts", self.species_count)
        reporter.kv("Orbital counts", self.orbitals_count)
        # print(self.permutation_matrix.T@self.permutation_matrix)
        reporter.kv("Atomic basis", self.atom_basis)


class LayeredLatticeAnalyzer:
    def __init__(self, input_data, num_layers, type_structure, twist_layer):
        """
        Initializes the LayeredLatticeAnalyzer with input data and number of layers.

        Args:
            input_data (list of dicts): Each dict should have 'original_index', 'species', and 'r' (tuple/list of x, y, z).
            num_layers (int): Number of layers to separate the atoms into based on z-coordinate.
            type_structure (list of int): Type of structure (1 for graphene-like, 2 for MoTe2-like.)
            twist_layer (list of int): The layer index where the twist occurs. e.g. [1, 3] for twist between layer 1 and the other 3 layers above.
        """
        self.input_data = input_data
        self.num_layers = num_layers
        self.df = None
        self.lattice_vectors = {}      # {layer: [a1, a2]}
        self.reciprocal_vectors = {}   # {layer: [b1, b2]}
        self.type_structure = type_structure #[1,1] length = num_layers
        self.twist_layer = twist_layer
        self.layer_nearest_vectors = {}  # {layer: [nearest_vectors]}
        self.layer_all_basis_vectors = {}  # {layer: [a1,a2,a3,...]}
        # Bravais lattice type hook (read from env var to avoid changing main.py signatures).
        bravais_env = os.environ.get("TAPW_BRAVAIS", "hex")
        bravais_norm = str(bravais_env).strip().lower()
        self.bravais = "square" if bravais_norm in {"square", "sq"} else "hex"
        

    def process(self,TAPW=True):
        """
        Runs the processing workflow to derive lattice vectors and reciprocal lattice vectors.
        """
        self.load_data()
        if TAPW:
            self.separate_layers()
            self.compute_lattice_vectors()

    def load_data(self):
        """
        Loads input data into a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing 'original_index', 'species', 'x', 'y', 'z'.
        """
        data = []
        for item in self.input_data:
            index = item['original_index']
            species = item['species']
            x, y, z = item['x'],item['y'],item['z']
            orb_num = item['orb_num']
            orb_name = item['orb_name']
            orb_global_index = item['orb_global_index']
            data.append({'original_index': index, 'species': species, 'x': x, 'y': y, 'z': z, 'orb_num': orb_num, 'orb_name': orb_name, 'orb_global_index': orb_global_index})
        self.df = pd.DataFrame(data)
        print(f"Loaded {len(self.df)} atoms into DataFrame.")

    def separate_layers1(self):
        """
        Separates atoms into specified number of layers based on their z-coordinate.

        Adds:
            'layer' column to self.df.
        """
        if self.df is None:
            raise ValueError("DataFrame is empty. Please load data first.")
        
        self.df = self.df.copy()
        self.df = self.df.sort_values(by='z').reset_index(drop=True)
        self.df['layer'] = pd.qcut(self.df['z'], q=self.num_layers, labels=False)
        print(f"Separated into {self.num_layers} layers with {self.df.shape[0]} atoms.")
        for i in range(self.num_layers):
            print(f"Layer {i}: {self.df[self.df['layer'] == i].shape[0]} atoms.")

    def separate_layers(self):
        """
        Separates atoms into the specified number of layers based on their z-coordinate using K-Means clustering.

        Adds:
            'layer' column to self.df.
        """
        def assign_twist_groups(layer_indices, twist_layer):
            """
            layer_indices: 已经按z均值排序后的层编号（如[0,1,2,3]）
            twist_layer: 例如[1,2,1]
            返回：每个层编号对应的组号
            """
            group_labels = []
            current = 0
            for group, count in enumerate(twist_layer):
                for _ in range(count):
                    group_labels.append(group)
                    current += 1
            return {layer: group_labels[i] for i, layer in enumerate(layer_indices)}
        
        
        if self.df is None:
            raise ValueError("DataFrame is empty. Please load data first.")
        
        # Extract z coordinates and reshape for clustering
        z_coords = self.df['z'].values.reshape(-1, 1)
        
        # Initialize K-Means with the desired number of clusters (layers)
        kmeans = KMeans(n_clusters=self.num_layers, random_state=0, n_init='auto')
        
        # Fit K-Means and predict cluster labels
        labels = kmeans.fit_predict(z_coords)
        
        # Assign cluster labels to the DataFrame
        self.df['layer'] = labels
        
        # Calculate the mean z-coordinate for each layer to sort layers from bottom to top
        layer_means = self.df.groupby('layer')['z'].mean().sort_values().index.tolist()
        
        # Create a mapping from old labels to new labels sorted by mean z-coordinate
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(layer_means)}
        
        # Apply the mapping to ensure layers are ordered from bottom to top
        self.df['layer'] = self.df['layer'].map(label_mapping)
        # self.df['layer'] = self.df['layer'].apply(lambda x: 0 if x < self.twist_layer[0] else 1)
        group_mapping = assign_twist_groups(sorted(label_mapping.values()), self.twist_layer)
        self.df['layer'] = self.df['layer'].map(group_mapping)
        for i, item in enumerate(self.input_data):
            self.input_data[i]['layer'] = self.df.loc[i, 'layer']
        # Print summary of layer separation
        # print(f"Separated into 2 layers with {self.df.shape[0]} atoms.")
        print(f"Separated into {len(self.twist_layer)} layers with {self.df.shape[0]} atoms.")
        for i in range(len(self.twist_layer)):
            num_atoms = self.df[self.df['layer'] == i].shape[0]
            print(f"Layer {i}: {num_atoms} atoms.")

    

    def compute_lattice_vectors(self):
        """
        Computes lattice vectors for each layer and stores them in self.lattice_vectors.
        """
        if 'layer' not in self.df.columns:
            raise ValueError("Layers are not separated. Please run separate_layers() first.")
        twist_angle_list = []
        current_angle = 0
        for layer in range(len(self.twist_layer)):
            # layer_atoms = self.df[self.df['layer'] == layer][['x', 'y', 'z']].values
            # if len(layer_atoms) < 2:
            #     print(f"Layer {layer}: Not enough atoms to determine lattice vectors.")
            #     continue
            if getattr(self, "bravais", "hex") == "square":
                a1, a2 = self._determine_lattice_vectors_square(layer)
            else:
                a1, a2 = self._determine_lattice_vectors(layer)
            self.lattice_vectors[layer] = [a1, a2]
            twist_angle_list.append(np.degrees(np.arctan2(a1[1],a1[0])))

            area = a1[0]*a2[1] - a1[1]*a2[0]
            if area == 0:
                # print(f"Layer {layer}: Lattice vectors are collinear. Cannot compute reciprocal vectors.")
                # continue
                raise ValueError(f"Layer {layer}: Lattice vectors are collinear. Cannot compute reciprocal vectors.")
            
            b1 = (2 * np.pi / area) * np.array([a2[1], -a2[0]])
            b2 = (2 * np.pi / area) * np.array([-a1[1], a1[0]])
            self.reciprocal_vectors[layer] = [b1, b2]

            print(f"\n=======      Layer {layer}: Lattice vectors (Angstrom)          ===== ")
            print(np.array([a1, a2]))
            print(f"\n======= Layer {layer}: Reciprocal lattice vectors (1/Angstrom) ===== ")
            print(np.array([b1, b2]))
            if getattr(self, "bravais", "hex") == "square":
                print("\n======= [square] a1/a2 self-check (monolayer) ===== ")
                print(f"a1·a2 = {float(np.dot(a1, a2)):.6e}")
                print(f"|a1|-|a2| = {float(np.linalg.norm(a1) - np.linalg.norm(a2)):.6e}")
            # print(f"Layer {layer}: Reciprocal lattice vectors computed.")

            # print(f"Layer {layer}: Lattice vectors computed.")
            # print(f"layer = {layer}, a1 = {a1}, a2 = {a2}")

        

            
            

        twist_angle_list = np.diff(twist_angle_list)
        print("\n======================= calculated twist angle ========================\n")
        for i in range(len(twist_angle_list)):
            print(f"The twist angle between layer {i} and layer {i+1} is {twist_angle_list[i]}°")
        print(f"\n======================= end lattice_vectors computation ========================")
        # exit()

    # useless
    def compute_reciprocal_vectors(self):
        """
        Computes reciprocal lattice vectors for each layer and stores them in self.reciprocal_vectors.
        """
        if not self.lattice_vectors:
            raise ValueError("Lattice vectors not computed. Please run compute_lattice_vectors() first.")
        
        for layer, vectors in self.lattice_vectors.items():
            a1, a2 = vectors
            # Calculate reciprocal lattice vectors
            area = a1[0]*a2[1] - a1[1]*a2[0]
            if area == 0:
                print(f"Layer {layer}: Lattice vectors are collinear. Cannot compute reciprocal vectors.")
                continue
            b1 = (2 * np.pi / area) * np.array([a2[1], -a2[0]])
            b2 = (2 * np.pi / area) * np.array([-a1[1], a1[0]])
            self.reciprocal_vectors[layer] = [b1, b2]
            print(f"Layer {layer}: Reciprocal lattice vectors computed.")

    def plot_lattice1(self, layer, xlim=[0,100], ylim=[0,100],periodic = [-20,40]):
        """
        Plots the atomic distribution and lattice vectors for a specified layer.

        Args:
            layer (int): The layer number to plot.
        """
        if layer not in self.lattice_vectors:
            raise ValueError(f"Lattice vectors for layer {layer} not found. Please compute them first.")
        
        layer_atoms = self.df[self.df['layer'] == layer][['x', 'y']].values
        a1, a2 = self.lattice_vectors[layer]
        
        # plt.figure(figsize=(8,8))
        # plt.scatter(layer_atoms[:,0], layer_atoms[:,1], s=10, label='Atomic Positions', alpha=0.6)

        fig,ax = plt.subplots(figsize=(8,8))
        ax.scatter(layer_atoms[:,0], layer_atoms[:,1], s=1, label='Atomic Positions')


        
        origin = np.mean(layer_atoms, axis=0)
        origin = np.array([0,0])
        
        # Plot lattice vectors
        # plt.arrow(origin[0], origin[1], a1[0], a1[1], head_width=0.2, head_length=0.2, fc='r', ec='r', label='Lattice Vector a1')
        # plt.arrow(origin[0], origin[1], a2[0], a2[1], head_width=0.2, head_length=0.2, fc='b', ec='b', label='Lattice Vector a2')
        ax.arrow(origin[0], origin[1], a1[0], a1[1], head_width=0., head_length=0., fc='r', ec='r', label='Lattice Vector a1')
        ax.arrow(origin[0], origin[1], a2[0], a2[1], head_width=0., head_length=0., fc='b', ec='b', label='Lattice Vector a2')
        # Optionally, plot additional lattice vectors to show periodicity
        # For visualization purposes, plot multiples of a1 and a2
        # multiples = [1, -1, 2, -2]
        # for m in multiples:
        #     plt.arrow(origin[0], origin[1], m*a1[0], m*a1[1], head_width=0.2, head_length=0.2, fc='r', ec='r', alpha=0.3)
        #     plt.arrow(origin[0], origin[1], m*a2[0], m*a2[1], head_width=0.2, head_length=0.2, fc='b', ec='b', alpha=0.3) 
        for i in range(*periodic):
            for j in range(*periodic):
                origin_new = origin + i*a1 + j*a2
                # plt.arrow(origin_new[0], origin_new[1], a1[0], a1[1], head_width=0., head_length=0., fc='r', ec='r', alpha=0.3)
                # plt.arrow(origin_new[0], origin_new[1], a2[0], a2[1], head_width=0., head_length=0., fc='b', ec='b', alpha=0.3)
                ax.arrow(origin_new[0], origin_new[1], a1[0], a1[1], head_width=0., head_length=0., fc='r', ec='r', alpha=0.3)
                ax.arrow(origin_new[0], origin_new[1], a2[0], a2[1], head_width=0., head_length=0., fc='b', ec='b', alpha=0.3)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        # plt.show()
        
        #set xlim and ylim
        # plt.xlim(-10,40)
        # plt.ylim(0,40)
        # #set aspect
        # plt.gca().set_aspect('equal', adjustable='box')
        
        # plt.xlabel('X (Å)')
        # plt.ylabel('Y (Å)')
        # plt.title(f'Layer {layer} Atomic Distribution with Lattice Vectors')
        # plt.legend()
        # plt.axis('equal')
        plt.show()

    def plot_lattice1(self, layer, xlim=None, ylim=None, distance_threshold=None):
        """
        Plots the atomic distribution and lattice vectors for a specified layer.
        
        Args:
            layer (int): The layer number to plot.
            xlim (list): Limits for the x-axis.
            ylim (list): Limits for the y-axis.
            distance_threshold (float): Distance threshold to determine which periodic vectors to plot.
        """
        if layer not in self.lattice_vectors:
            raise ValueError(f"Lattice vectors for layer {layer} not found. Please compute them first.")
        
        # 获取指定层的原子坐标
        layer_atoms = self.df[self.df['layer'] == layer][['x', 'y']].values
        a1, a2 = self.lattice_vectors[layer]
        
        # 创建绘图
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(layer_atoms[:, 0], layer_atoms[:, 1], s=1, label='Atomic Positions', color='black', alpha=0.6)
        
        # 定义距离阈值
        # threshold = distance_threshold
        threshold = np.max(np.linalg.norm(layer_atoms, axis=1))*0.1 if distance_threshold is None else distance_threshold
        
        # 计算需要的最大平移次数
        max_i = int(np.ceil(threshold / np.linalg.norm(a1))) + 1
        max_j = int(np.ceil(threshold / np.linalg.norm(a2))) + 1
        
        # 生成 i 和 j 的范围
        i_values = np.arange(-max_i, max_i + 1)
        j_values = np.arange(-max_j, max_j + 1)
        ii, jj = np.meshgrid(i_values, j_values)
        
        # 计算所有可能的平移向量
        translation_vectors = ii.flatten()[:, np.newaxis] * a1 + jj.flatten()[:, np.newaxis] * a2  # Shape: (M, 2)
        
        # 构建 KDTree 以加速距离计算
        tree = cKDTree(layer_atoms)
        
        # 查询每个平移向量到最近原子的距离
        distances, _ = tree.query(translation_vectors, k=1)
        print(f"min distance = {np.min(distances)}")
        print(f"max distance = {np.max(distances)}")
        print(f"mean distance = {np.mean(distances)}")

        # 选择满足距离阈值的平移向量
        mask = distances < threshold*0 + 3.5
        selected_translations = translation_vectors[mask]
        
        # 准备 a1 和 a2 的向量数据
        U_a1 = np.tile(a1[0], len(selected_translations))  # Shape: (N,)
        V_a1 = np.tile(a1[1], len(selected_translations))  # Shape: (N,)
        
        U_a2 = np.tile(a2[0], len(selected_translations))  # Shape: (N,)
        V_a2 = np.tile(a2[1], len(selected_translations))  # Shape: (N,)
        
        # 使用 quiver 批量绘制 a1 向量
        ax.quiver(
            selected_translations[:, 0],
            selected_translations[:, 1],
            U_a1,
            V_a1,
            angles='xy',
            scale_units='xy',
            scale=1,
            color='r',
            alpha=0.3,
            width=0.002,
            label='Lattice Vector a1'
        )
        
        # 使用 quiver 批量绘制 a2 向量
        ax.quiver(
            selected_translations[:, 0],
            selected_translations[:, 1],
            U_a2,
            V_a2,
            angles='xy',
            scale_units='xy',
            scale=1,
            color='b',
            alpha=0.3,
            width=0.002,
            label='Lattice Vector a2'
        )
        
        # 设置绘图范围和比例
        xlim = [np.min(layer_atoms[:, 0]) - 5, np.max(layer_atoms[:, 0]) + 5] if xlim is None else xlim
        ylim = [np.min(layer_atoms[:, 1]) - 5, np.max(layer_atoms[:, 1]) + 5] if ylim is None else ylim
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_title(f'Layer {layer} Atomic Distribution with Lattice Vectors')
        ax.legend(['Atomic Positions', 'Lattice Vector a1', 'Lattice Vector a2'])
        plt.show()

    def plot_lattice(self, layer=None, xlim=None, ylim=None, distance_threshold=None, save_path=None, save=False):
        """
        Plots the atomic distribution and lattice vectors for a specified layer, or for all layers if `layer` is None.
        
        Args:
            layer (int or None): The layer number to plot. If None, plots all layers in separate subplots.
            xlim (list): Limits for the x-axis.
            ylim (list): Limits for the y-axis.
            distance_threshold (float): Distance threshold to determine which periodic vectors to plot.
            save_path (str or None): Path to save the plot. If None, does not save the plot.
            save (bool): Whether to save the plot.
        """
        layers_to_plot = [layer] if layer is not None else sorted(self.lattice_vectors.keys())
        num_layers = len(layers_to_plot)
        
        # 设置子图布局
        rows = (num_layers + 1) // 2  # 行数
        cols = 2 if num_layers > 1 else 1  # 列数
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        axes = np.array(axes).reshape(-1)  # 将 axes 转为一维数组，以便索引

        for idx, layer in enumerate(layers_to_plot):
            ax = axes[idx] if num_layers is not None else axes
            if layer not in self.lattice_vectors:
                raise ValueError(f"Lattice vectors for layer {layer} not found. Please compute them first.")
            
            # 获取指定层的原子坐标
            layer_atoms = self.df[self.df['layer'] == layer][['x', 'y']].values
            a1, a2 = self.lattice_vectors[layer]
            
            # 绘制原子位置
            ax.scatter(layer_atoms[:, 0], layer_atoms[:, 1], s=1, label='Atomic Positions', color='black', alpha=0.6)
            
            # 设置距离阈值
            threshold = np.max(np.linalg.norm(layer_atoms, axis=1)) * 1.3 if distance_threshold is None else distance_threshold
            
            # 计算平移次数
            max_i = int(np.ceil(threshold / np.linalg.norm(a1))) + 1
            max_j = int(np.ceil(threshold / np.linalg.norm(a2))) + 1
            
            # 生成平移向量
            i_values = np.arange(-max_i, max_i + 1)
            j_values = np.arange(-max_j, max_j + 1)
            ii, jj = np.meshgrid(i_values, j_values)
            translation_vectors = ii.flatten()[:, np.newaxis] * a1 + jj.flatten()[:, np.newaxis] * a2
            
            # 使用 KDTree 加速距离计算
            tree = cKDTree(layer_atoms)
            distances, _ = tree.query(translation_vectors, k=1)
            mask = distances < threshold*0 + 3.5
            
            selected_translations = translation_vectors[mask]
            
            # 准备 a1 和 a2 的向量数据
            U_a1 = np.tile(a1[0], len(selected_translations))
            V_a1 = np.tile(a1[1], len(selected_translations))
            U_a2 = np.tile(a2[0], len(selected_translations))
            V_a2 = np.tile(a2[1], len(selected_translations))
            
            # 批量绘制 a1 和 a2 的向量
            ax.quiver(selected_translations[:, 0], selected_translations[:, 1], U_a1, V_a1,
                    angles='xy', scale_units='xy', scale=1, color='r', alpha=0.3, width=0.002,
                    label='Lattice Vector a1')
            ax.quiver(selected_translations[:, 0], selected_translations[:, 1], U_a2, V_a2,
                    angles='xy', scale_units='xy', scale=1, color='b', alpha=0.3, width=0.002,
                    label='Lattice Vector a2')
            
            # 设置坐标轴范围
            xlim_layer = [np.min(layer_atoms[:, 0]) - 5, np.max(layer_atoms[:, 0]) + 5] if xlim is None else xlim
            ylim_layer = [np.min(layer_atoms[:, 1]) - 5, np.max(layer_atoms[:, 1]) + 5] if ylim is None else ylim
            ax.set_xlim(xlim_layer)
            ax.set_ylim(ylim_layer)
            ax.set_aspect('equal')
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_title(f'Layer {layer} Atomic Distribution with Lattice Vectors')
            
            # 添加图例
            ax.legend(['Atomic Positions', 'Lattice Vector a1', 'Lattice Vector a2'])

        # 移除多余的空子图
        for extra_ax in axes[num_layers:]:
            extra_ax.axis('off')

        plt.tight_layout()
        if save:
            if save_path is None:
                raise ValueError("Please provide a save_path to save the plot.")
            plt.savefig(save_path + '/lattice.pdf', dpi=300, bbox_inches='tight')

        plt.show()

    def plot_nearest_vectors_phase(self, layer=None, xlim=None, ylim=None, distance_threshold=None, save_path=None, save=False):
        """
        Plots all vectors or vectors of a specific layer with color coding based on class and z-deviation.

        Args:
            layer (int or None): If None, plots all layers with subplots. Otherwise, plots the specified layer.
            xlim (list): Limits for the x-axis.
            ylim (list): Limits for the y-axis.
            distance_threshold (float): Distance threshold to determine which periodic vectors to plot.
            save_path (str or None): Path to save the plot. If None, does not save the plot.
            save (bool): Whether to save the plot.
        """
        # 根据层数决定子图布局
        if layer is None:
            num_layers = len(self.twist_layer)  # 假设 self.num_layers 已定义
            rows = (num_layers + 1) // 2  # 行数
            cols = 2 if num_layers > 1 else 1  # 列数
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
            axes = np.array(axes).reshape(-1)  # 将 axes 转为一维数组，以便索引
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            axes = [ax]

        for idx, ax in enumerate(axes):
            if layer is not None and idx >=1:
                ax.axis('off')  # 只绘制指定的层，其他子图关闭

            # 根据层选择数据
            if layer is None:
                current_layer = idx
                vectors = self.lattice_vectors.get(current_layer, [])
                if not vectors:
                    ax.axis('off')
                    continue
            else:
                current_layer = layer
                vectors = self.lattice_vectors.get(current_layer, [])
                if not vectors:
                    ax.axis('off')
                    continue
            vectors = self.layer_nearest_vectors.get(current_layer, [])
            all_basis_vectors = self.layer_all_basis_vectors.get(current_layer, [])
            basis_vectors = self.lattice_vectors.get(current_layer, [])
            # if not vectors.any() or not all_basis_vectors.any() or not basis_vectors.any():
            #     raise ValueError(f"basis vectors for layer {current_layer} not found. Please compute them first.")
            all_basis_vectors_angle = np.degrees(np.arctan2(all_basis_vectors[:,1],all_basis_vectors[:,0]))%360
            vectors_angle = np.degrees(np.arctan2(vectors[:,1],vectors[:,0]))%360
            ax.quiver(0, 0, basis_vectors[0][0], basis_vectors[0][1], angles='xy', scale_units='xy', scale=1, color='r', width=0.004,alpha=0.7, label='Lattice Vector a1')
            ax.quiver(0, 0, basis_vectors[1][0], basis_vectors[1][1], angles='xy', scale_units='xy', scale=1, color='b', width=0.004,alpha=0.7, label='Lattice Vector a2')
            ax.scatter(all_basis_vectors[:,0], all_basis_vectors[:,1],  s=10, label='Nearst Vectors')
            max_norm = np.linalg.norm(basis_vectors[0]) * 1.2
            # Add polar grid lines every 60 and 30 degrees

            for angle in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
                rad = np.radians(angle)
                linestyle = '-' if angle % 60 == 0 else '--'
                linewidth = 1 if angle % 60 == 0 else 0.5
                color = 'black' if angle % 60 == 0 else 'gray'
                ax.plot([0, max_norm * np.cos(rad)], [0, max_norm * np.sin(rad)],
                        color=color, linestyle=linestyle, linewidth=linewidth)

            # Add concentric circles
            circle_norm_list = np.arange(0, max_norm*1.1, 0.5)
            for circle_norm in circle_norm_list:
                circle = plt.Circle((0, 0), circle_norm, color='black', fill=False, linestyle='--', linewidth=0.5)
                ax.add_artist(circle)

            # Label average angle for each class
            for i, vec in enumerate(all_basis_vectors):
                angle = np.degrees(np.arctan2(vec[1], vec[0])) % 360
                vec_text = self.rot_z(vec[:2], np.pi/30) * 1.05
                ax.text(vec_text[0], vec_text[1], f'{angle:.4f}°', color='red', fontsize=8)

            # Add histogram in the lower right
            ax_hist = inset_axes(ax, width="24%", height="8%", loc='lower right')
            ax_hist.hist(vectors_angle, bins=np.arange(0, 365, 20), color='gray', edgecolor='black')
            # Set ticks and labels 120 degrees apart, 60 minor ticks
            ax_hist.set_xticks(np.arange(0, 361, 120))
            ax_hist.set_xticks(np.arange(0, 361, 60), minor=True)
            ax_hist.set_xlabel('Angle (degrees)', fontsize=8)
            ax_hist.set_ylabel('Count', fontsize=8)
            ax_hist.set_title('Angle Distribution', fontsize=8)
            ax_hist.tick_params(axis='both', which='major', labelsize=8)

            # Set labels and title
            ax.set_xlabel(r'Vec$_x$')
            ax.set_ylabel(r'Vec$_y$')
            ax.set_title(f'Clustering of Vectors and Lattice Basis (Layer {current_layer})')

            # Add legend
            ax.legend(fontsize=8, loc='upper right')
            ax.set_xlim(-max_norm*1.2, max_norm*1.2)
            ax.set_ylim(-max_norm*1.2, max_norm*1.2)
            ax.set_aspect('equal')
            

        plt.tight_layout()
        if save:
            if save_path is None:
                raise ValueError("Please provide a save_path to save the plot.")
            plt.savefig(save_path + '/nearest_vectors_phase.pdf', dpi=300, bbox_inches='tight')
        plt.show()
	            

    def _determine_lattice_vectors_square(
        self,
        layer,
        neighbor_k: int = 12,
        round_tol: float = 1e-3,
        match_tol: float = 1e-2,
        score_threshold: float = 0.5,
    ):
        """
        Determine two in-plane translation vectors (a1,a2) for a square Bravais lattice.

        This branch is intentionally free of hex-specific assumptions (no 60°/C3 hard-coding).
        It searches short neighbour vectors and scores each candidate by translation invariance:
        shifting points by the candidate should map most points onto other points.

        Side effects (for plotting/debug compatibility):
        - Populates `self.layer_nearest_vectors[layer]` and `self.layer_all_basis_vectors[layer]`.
        """
        if self.df is None:
            raise ValueError("DataFrame is empty. Please load data first.")

        layer_df = self.df[self.df["layer"] == layer]
        if layer_df.empty:
            raise ValueError(f"Layer {layer}: no atoms found.")

        # Use the dominant species to reduce basis interference; fallback to all atoms if too few.
        species_counts = layer_df["species"].value_counts()
        dominant_species = species_counts.index[0]
        coords = layer_df[layer_df["species"] == dominant_species][["x", "y"]].values
        if coords.shape[0] < 4:
            coords = layer_df[["x", "y"]].values
        coords = np.asarray(coords, dtype=np.float64)
        if coords.shape[0] < 4:
            raise ValueError(f"Layer {layer}: not enough atoms to determine lattice vectors.")

        tree = cKDTree(coords)

        k = min(int(neighbor_k) + 1, coords.shape[0])
        _, idxs = tree.query(coords, k=k)

        # Candidate neighbour vectors (2D).
        neigh = coords[idxs[:, 1:].reshape(-1)]
        origin = np.repeat(coords, k - 1, axis=0)
        cand = neigh - origin
        cand = np.vstack([cand, -cand])

        lengths = np.linalg.norm(cand, axis=1)
        valid = lengths > 1e-6
        cand = cand[valid]
        lengths = lengths[valid]
        if cand.shape[0] == 0:
            raise ValueError(f"Layer {layer}: failed to build neighbour vectors.")

        # Store for plotting/debug.
        self.layer_nearest_vectors[layer] = cand

        # Deduplicate candidates by rounding (avoids scoring identical vectors many times).
        round_tol = float(round_tol) if round_tol and round_tol > 0 else 1e-3
        keys = np.round(cand / round_tol).astype(int)
        uniq = {}
        for v, l, kvec in zip(cand, lengths, keys):
            key = (int(kvec[0]), int(kvec[1]))
            if key == (0, 0):
                continue
            prev = uniq.get(key)
            if prev is None or l < prev[1]:
                uniq[key] = (v, l)

        if not uniq:
            raise ValueError(f"Layer {layer}: failed to deduplicate neighbour vectors.")

        unique_vectors = np.asarray([item[0] for item in uniq.values()], dtype=np.float64)
        unique_lengths = np.asarray([item[1] for item in uniq.values()], dtype=np.float64)
        sort_idx = np.argsort(unique_lengths)
        unique_vectors = unique_vectors[sort_idx]
        unique_lengths = unique_lengths[sort_idx]

        match_tol = float(match_tol) if match_tol and match_tol > 0 else 1e-2

        def match_score(v: np.ndarray) -> float:
            shifted = coords + v
            d, _ = tree.query(shifted, k=1)
            return float(np.mean(d < match_tol))

        # Pick a1: shortest vector with good translation-invariance score.
        score_threshold = float(score_threshold)
        max_scan = min(200, unique_vectors.shape[0])
        a1 = None
        a1_score = -1.0
        for v in unique_vectors[:max_scan]:
            s = match_score(v)
            if s > a1_score:
                a1_score = s
                a1 = v
            if s >= score_threshold:
                a1_score = s
                a1 = v
                break
        if a1 is None:
            raise ValueError(f"Layer {layer}: failed to pick a1 for square bravais.")

        # Pick a2: linearly independent vector (prefer ~90°) with good score.
        n1 = float(np.linalg.norm(a1))
        max_scan2 = min(400, unique_vectors.shape[0])
        a2 = None
        a2_score = -1.0
        best_cos = 1.0
        for v in unique_vectors[:max_scan2]:
            n = float(np.linalg.norm(v))
            if n == 0.0:
                continue
            cos = abs(float(np.dot(v, a1)) / (n * n1))
            if cos > 0.95:
                continue
            s = match_score(v)
            if (s > a2_score) or (np.isclose(s, a2_score) and cos < best_cos):
                a2_score = s
                a2 = v
                best_cos = cos
            if s >= score_threshold and cos < 0.2:
                a2_score = s
                a2 = v
                best_cos = cos
                break
        if a2 is None:
            raise ValueError(f"Layer {layer}: failed to pick a2 for square bravais.")

        a1 = np.asarray(a1, dtype=np.float64)
        a2 = np.asarray(a2, dtype=np.float64)
        
        theta = 45 if self.type_structure[layer] == 1 else 0
        scale = np.sqrt(2) if self.type_structure[layer] == 1 else 1
        a1 = self.rot_z(a1,np.radians(theta))*scale
        a2 = self.rot_z(a2,np.radians(theta))*scale
        

        # Ensure right-handed orientation in xy plane.
        area = a1[0] * a2[1] - a1[1] * a2[0]
        if area < 0:
            a2 = -a2
            area = -area
        if area == 0:
            raise ValueError(f"Layer {layer}: a1/a2 are collinear after selection.")
        # --- force a1 angle into [0, 90) by rotating (a1,a2) together by 90° multiples ---
        ang = (np.degrees(np.arctan2(a1[1], a1[0])) + 360.0) % 360.0
        k = int(ang // 90.0)          # 0,1,2,3
        k = (-k) & 3                  # rotate by -90*k (mod 4), now k in 0..3

        if k:
            def rot90_xy(v, k):
                x, y = v[0], v[1]
                if k == 1:  v[0], v[1] = -y, x
                elif k == 2: v[0], v[1] = -x, -y
                elif k == 3: v[0], v[1] = y, -x

            a1 = a1.copy(); a2 = a2.copy()
            rot90_xy(a1, k)
            rot90_xy(a2, k)

        # For plotting: show four symmetry-related basis vectors.
        self.layer_all_basis_vectors[layer] = np.array([a1, a2, -a1, -a2], dtype=np.float64)

        print(f"\n======= Layer {layer}: [square] translation match scores ===== ")
        print(f"a1_score = {a1_score:.3f}, a2_score = {a2_score:.3f}")

        return a1, a2

    def _determine_lattice_vectors_old(self, layer, factor=1.1, tolerance=10):
        """
        Determines the lattice vectors for a single layer using geometric and statistical analysis.

        Args:
            layer_atoms (np.ndarray): Array of shape (N, 2) containing x and y coordinates of atoms in the layer.
            factor (float): Factor to select candidate vectors based on length.
            tolerance (float): Tolerance in degrees for angle clustering.
            type_structure (int): 1 for graphene-like, 2 for MoTe2-like.

        Returns:
            tuple: Two lattice vectors a1 and a2 as numpy arrays.
        """
        # layer_atoms = self.df[self.df['layer'] == layer][['x', 'y', 'z']].values
        # vectors = self._compute_pairwise_vectors(layer_atoms)
        # candidate_vectors, candidate_lengths = self._select_candidate_vectors(vectors, factor)
        # candidate_angles = self._compute_vector_angles(candidate_vectors)
        # hist, bin_edges = self._plot_angle_histogram(candidate_angles, bin_size=5)
        # peak_angles = self._identify_peak_angles(hist, bin_edges, num_peaks=6)
        # lattice_vectors = self._determine_lattice_vectors_from_peaks(
        #     candidate_vectors, candidate_angles, peak_angles, tolerance, type_structure=type_structure
        # )

        lattice_vectors = self._determine_lattice_vectors_from_peaks(
            layer, tolerance
        )
        if len(lattice_vectors) < 2:
            print(f"Layer: Not enough lattice vectors determined.")
            return np.array([0,0]), np.array([0,0])
        else:
            print(f"Layer: Lattice vectors determined in function _detemine_lattice_vectors.")
            print(f"lattice_vectors = {lattice_vectors}")
            return lattice_vectors[0], lattice_vectors[1]

    def _compute_pairwise_vectors1(self, coords):
        """
        Computes all unique pairwise vectors between atoms.

        Args:
            coords (np.ndarray): Array of shape (N, 2) containing x and y coordinates.

        Returns:
            np.ndarray: Array of shape (M, 2) containing pairwise vectors.
        """
        num_atoms = len(coords)
        vectors = []
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                vector = coords[j] - coords[i]
                vectors.append(vector)
        return np.array(vectors)
    
    def _compute_pairwise_vectors(self, coords):
        """
        Computes all unique pairwise vectors between atoms using NumPy broadcasting for speed.

        Args:
            coords (np.ndarray): Array of shape (N, 2) containing x and y coordinates.

        Returns:
            np.ndarray: Array of shape (M, 2) containing unique pairwise vectors.
        """
        # Compute pairwise differences with broadcasting
        diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        
        # Keep only the upper triangular part (i < j), as (j - i) is just the negative of (i - j)
        # indices = np.triu_indices(len(coords), k=1)
        # pairwise_vectors = diffs[indices]

        # Create a mask to exclude the diagonal (i != j)
        mask = ~np.eye(len(coords), dtype=bool)

        # Apply the mask to keep both upper and lower triangular parts, excluding the diagonal
        pairwise_vectors = diffs[mask]
        
        return pairwise_vectors


    def _select_candidate_vectors(self, vectors, factor=1.1, max_z=0.3):
        """
        Selects candidate vectors based on their lengths and z-distance.

        Args:
            vectors (np.ndarray): Array of shape (M, 3) containing pairwise vectors (x, y, z).
            factor (float): Upper bound multiplier for selecting vectors based on length.
            max_z (float): Maximum allowed absolute z-distance to retain the vector.

        Returns:
            tuple: (candidate_vectors, candidate_lengths)
                - candidate_vectors (np.ndarray): Filtered vectors meeting length and z-distance criteria.
                - candidate_lengths (np.ndarray): Lengths of the filtered vectors.
        """
        # Calculate the full 3D lengths of the vectors
        lengths = np.linalg.norm(vectors, axis=1)
        
        # Calculate the absolute z-distances of the vectors
        z_dists = np.abs(vectors[:, 2])
        
        # Identify vectors with |z| <= max_z
        valid_indices = np.where(z_dists <= max_z)[0]
        valid_lengths = lengths[valid_indices]
        
        if len(valid_lengths) == 0:
            raise ValueError(f"No vectors found with |z| <= {max_z} Å.")
        
        # Determine the minimum length among the valid vectors
        min_length = np.min(valid_lengths)
        print(f"\nMinimum bond length (with |z| <= {max_z} Å): {min_length:.2f} Å.")
        
        # Define the lower and upper bounds for length filtering
        lower_bound = min_length * 0.95
        upper_bound = factor * min_length * 1.05
        
        # Select vectors that meet both length and z-distance criteria
        candidate_indices = np.where(
            (lengths >= lower_bound) &
            (lengths <= upper_bound) &
            (z_dists <= max_z)
        )[0]
        
        candidate_vectors = vectors[candidate_indices]
        candidate_lengths = lengths[candidate_indices]
        
        print(f"Selected {len(candidate_vectors)} candidate vectors with lengths between {lower_bound:.2f} Å and {upper_bound:.2f} Å and |z| <= {max_z} Å.")
        print(f"Minimum candidate length = {np.min(candidate_lengths):.2f} Å, Maximum candidate length = {np.max(candidate_lengths):.2f} Å.")
        
        return candidate_vectors, candidate_lengths

    def _compute_vector_angles(self, vectors):
        """
        Computes the angles of vectors in degrees, modulo 180.

        Args:
            vectors (np.ndarray): Array of shape (M, 2) containing vectors.

        Returns:
            np.ndarray: Array of angles in degrees.
        """
        angles = np.degrees(np.arctan2(vectors[:,1], vectors[:,0]))
        angles = np.mod(angles, 360)  # Equivalent directions
        return angles

    def _plot_angle_histogram(self, angles, bin_size=5):
        """
        Plots a histogram of vector angles.

        Args:
            angles (np.ndarray): Array of angles in degrees.
            bin_size (int): Size of each bin in degrees.

        Returns:
            tuple: (hist, bin_edges)
        """
        bins = np.arange(0, 365, bin_size)  # 0 to 180 degrees
        hist, bin_edges = np.histogram(angles, bins=bins)
        
        # plt.figure(figsize=(8,6))
        # plt.bar(bin_edges[:-1], hist, width=bin_size, edgecolor='k', align='edge')
        # plt.xlabel('Angle (degrees)')
        # plt.ylabel('Count')
        # plt.title('Angle Histogram of Candidate Vectors')
        # plt.show()
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].bar(bin_edges[:-1], hist, width=bin_size, edgecolor='k', align='edge')
        ax[0].set_xlabel('Angle (degrees)')
        ax[0].set_ylabel('Count')
        ax[0].set_title('Angle Histogram of Candidate Vectors')
        ax[1].scatter(np.cos(np.radians(angles)),np.sin(np.radians(angles)),s=1)
        ax[1].set_xlabel('cos(angle)')
        ax[1].set_ylabel('sin(angle)')
        ax[1].set_title('Angle Scatter of Candidate Vectors')
        ax[1].set_aspect('equal')
        ax[1].set_xlim(-1.05,1.05)
        ax[1].set_ylim(-1.05,1.05)
        plt.show()
        
        return hist, bin_edges

    def _identify_peak_angles(self, hist, bin_edges, num_peaks=6):
        """
        Identifies peak angles from the histogram.

        Args:
            hist (np.ndarray): Histogram counts.
            bin_edges (np.ndarray): Edges of the histogram bins.
            num_peaks (int): Number of peak angles to identify.

        Returns:
            list: List of peak angles in degrees.
        """
        peak_indices = hist.argsort()[-num_peaks:][::-1]
        peak_angles = bin_edges[:-1][peak_indices] + (bin_edges[1] - bin_edges[0])/2
        print(f"Identified peak angles: {peak_angles}")
        return peak_angles.tolist()

    def _determine_lattice_vectors_from_peaks1(self, vectors, angles, peak_angles, tolerance=10,type_structure=1):
        """
        Determines lattice vectors based on peak angles.

        Args:
            vectors (np.ndarray): Candidate vectors.
            angles (np.ndarray): Angles of candidate vectors.
            peak_angles (list): Identified peak angles.
            tolerance (float): Tolerance in degrees for matching vectors to peaks.
            type_structure (int): 1 for graphene-like, 2 for MoTe2-like.

        Returns:
            list: List of lattice vectors as numpy arrays.
        """
        lattice_vectors = []
        lattice_vectors_angle = []
        for angle in peak_angles:
            lower = angle - tolerance
            upper = angle + tolerance
            # Handle wrap-around
            mask = (angles >= lower) & (angles <= upper)
            selected_vectors = vectors[mask]
            if len(selected_vectors) == 0:
                print(f"No vectors found within {tolerance} degrees of {angle} degrees.")
                continue
            avg_vector = np.mean(selected_vectors, axis=0)
            print(f"Peak angle: {angle:.2f}°, Average vector: {avg_vector}")
            print(f"Vector lengths: {np.mean(np.linalg.norm(selected_vectors, axis=1))}")
            lattice_vectors.append(avg_vector)
            angle = np.arctan2(avg_vector[1],avg_vector[0])%360
            lattice_vectors_angle.append(angle)
            print(f"Peak angle: {angle:.2f}°, Average vector: {avg_vector}")
        sort_index = np.argsort(lattice_vectors_angle)
        lattice_vectors = [lattice_vectors[i][:2] for i in sort_index]
        # lattice_vectors = [] 
        # avg_vector = np.mean(np.linalg.norm(selected_vectors, axis=1))
        if type_structure == 1:
            # a1 = np.array([avg_vector, 0]) * np.sqrt(3)
            # a1 = rot_z(a1,np.min(peak_angles)*np.pi/180+np.pi/6)
            a1 = self.rot_z(lattice_vectors[0],np.pi/6) * np.sqrt(3)
            a2 = self.rot_z(a1,np.pi/3)
            print("type structure 1")
            print(f"lattice_vectors = {a1}, {a2}")
            print(f"min peak angle = {np.min(peak_angles)}")
        else:
            # a1 = np.array([avg_vector, 0])
            # a1 = rot_z(a1,np.min(peak_angles)*np.pi/180)
            a1 = lattice_vectors[0]
            a2 = self.rot_z(a1,np.pi/3)
            print("type structure 2")
            print(f"lattice_vectors = {a1}, {a2}")
            print(f"min peak angle = {np.min(peak_angles)}")
        
        return np.array([a1, a2])

    def _determine_lattice_vectors(self, layer, tolerance=5,factor=1.1):
        """
        Determines lattice vectors based on clustered candidate vectors using DBSCAN.

        Args:
            vectors (np.ndarray): Candidate vectors (M, 3).
            angles (np.ndarray): Angles of candidate vectors in degrees (M,).
            tolerance (float): Tolerance in degrees for DBSCAN clustering.
            type_structure (int): 1 for graphene-like, 2 for MoTe2-like structures.

        Returns:
            np.ndarray: Array containing two lattice vectors [a1, a2].
        """
        # Step 1: Cluster the candidate vectors based on angle %120 degrees
        layer_atoms = self.df[self.df['layer'] == layer][['x', 'y', 'z']].values
        vectors = self._compute_pairwise_vectors(layer_atoms)
        candidate_vectors, candidate_lengths = self._select_candidate_vectors(vectors, factor)
        self.layer_nearest_vectors[layer] = candidate_vectors
        vectors = candidate_vectors
        angles = np.arctan2(vectors[:,1], vectors[:,0])
        # angles = np.degrees(angles) 
        # radians = np.radians(angles)
        angles_mod120 = angles % (2 * np.pi / 3)
        # radians_mod120 = np.radians(angles_mod120)
        # unit_vectors = np.vstack((np.cos(radians_mod120), np.sin(radians_mod120))).T
        unit_vectors = np.array([np.cos(angles), np.sin(angles)]).T

        db = DBSCAN(eps=np.radians(tolerance)*0 + 0.2, min_samples=5)
        # db.fit(unit_vectors)
        # labels = db.labels_

        labels = db.fit_predict(unit_vectors)



        # Remove noise
        valid_mask = labels != -1
        valid_labels = labels[valid_mask]
        valid_vectors = vectors[valid_mask]
        valid_angles_mod120 = angles_mod120[valid_mask]

        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        num_clusters = len(unique_labels)

        # Check if number of clusters is multiple of 3
        if num_clusters % 3 != 0:
            print(f"Number of clusters ({num_clusters}) is not a multiple of 3.")
            print(f"Unique labels: {unique_labels}")
            raise ValueError(f"Number of clusters ({num_clusters}) is not a multiple of 3.")

        # Determine m (m=1 or m=2)
        m = num_clusters // 3
        if m not in [1, 2]:
            raise ValueError(f"Detected m={m}, expected m=1 or m=2.")
        # print(f"Detected clusters: {num_clusters} (m={m})")
        # Sort clusters by size and select top 3m clusters
        sorted_indices = np.argsort(counts)[::-1]
        top_labels = unique_labels[sorted_indices]
        top_counts = counts[sorted_indices]

        # Check cluster size consistency (within 20%)
        if np.max(top_counts) / np.min(top_counts) > 1.4:
            raise ValueError("Selected clusters have significantly different sizes.")
            # print("Selected clusters have significantly different sizes.")
            # if m != 2:
            #     raise ValueError("There are maybe something wrong with the clustering.")
            # m = 1
            # tap_labels = top_labels[:3*m]
            # top_counts = top_counts[:3*m]
        

        # Step 2: Assign clusters to classes
        if m == 1:
            # Single class
            class_assignments = {label: 0 for label in top_labels}
        else:
            # m=2, further cluster the top_labels into 2 classes using DBSCAN on their average angles
            cluster_avg_angles = []
            for label in top_labels:
                cluster_angles = angles_mod120[labels == label]
                #保留5位小数
                # print(f"cluster_angles = {np.round(cluster_angles*180/np.pi,5)}")
                if np.max(cluster_angles)-np.min(cluster_angles) > 10/180*np.pi:
                    shift = -10/180*np.pi
                    cluster_angles = cluster_angles - shift
                    cluster_angles = np.mod(cluster_angles, 2*np.pi/3)+shift
                    # print(f"cluster_angles after shift = {np.round(cluster_angles*180/np.pi,5)}")
                if np.max(cluster_angles)-np.min(cluster_angles) > 10/180*np.pi:
                    raise ValueError(f"cluster_angles = {np.round(cluster_angles*180/np.pi,5)}")
                avg_angle = np.mean(cluster_angles)
                if avg_angle > np.pi*2/3-1/180*np.pi:
                    avg_angle = avg_angle - 2*np.pi/3
                cluster_avg_angles.append(avg_angle)
            cluster_avg_angles = np.array(cluster_avg_angles)*180/np.pi
            print(f"cluster_avg_angles = {cluster_avg_angles}")

            # Convert average angles to unit vectors for clustering
            radians_avg_angles = np.radians(cluster_avg_angles)
            unit_vectors_avg_angles = np.vstack((np.cos(radians_avg_angles), np.sin(radians_avg_angles))).T
            db_final = DBSCAN(eps=np.radians(tolerance*0.1), min_samples=1)
            db_final.fit(unit_vectors_avg_angles)
            final_labels = db_final.labels_

            # Remove noise
            final_mask = final_labels != -1
            final_unique_labels, final_counts = np.unique(final_labels[final_mask], return_counts=True)
            
            if len(final_unique_labels) not in [1, 2]:
                raise ValueError(f"After secondary clustering, number of classes ({len(final_unique_labels)}) is not 1 or 2.")

            if len(final_unique_labels) == 2:
                # Check if angle difference is ~60 degrees
                class_angles = []
                for flabel in final_unique_labels:
                    class_angles.append(np.mean(cluster_avg_angles[final_labels == flabel]))
                class_angles = np.array(class_angles)
                angle_diff = np.abs(class_angles[0] - class_angles[1]) % 120
                if not np.isclose(angle_diff, 60, atol=10):
                    raise ValueError(f"Angle difference between classes is {angle_diff:.2f} degrees, expected ~60 degrees.")

                # Assign clusters to classes
                class_assignments = {}
                for i, flabel in enumerate(final_unique_labels):
                    class_assignments[top_labels[i]] = i
            else:
                # Only one class
                class_assignments = {label: 0 for label in top_labels}

        # Step 3: Compute average vectors for each selected cluster
        basis_vectors = []
        for label in top_labels:
            # class_id = class_assignments[label]
            cluster_vectors = vectors[labels == label]
            avg_vector = np.mean(cluster_vectors, axis=0)
            basis_vectors.append(avg_vector)
        basis_vectors = np.array(basis_vectors)
        self.layer_all_basis_vectors[layer] = basis_vectors

        # Step 4: Rotate basis vectors to 0-theta range
        # theta = 120 if m ==1 else 60
        theta = 60
        rotated_basis_vectors = []
        angles = np.array([np.degrees(np.arctan2(vec[1], vec[0])) for vec in basis_vectors]) % theta
        print(f"angles = {angles}")
        if np.max(angles) - np.min(angles) > 10:
            shift = 10
            angles = angles - shift
            angles = np.mod(angles, theta) + shift
        if np.mean(angles) > 59:
            angles = angles - 60
        print(f"angles after shift = {angles}")
        if np.max(angles) - np.min(angles) > 10:
            raise ValueError(f"angles = {angles}")
        
        for i,vec in enumerate(basis_vectors):
            # angle = np.degrees(np.arctan2(vec[1], vec[0])) % theta
            angle = angles[i]
            norm = np.linalg.norm(vec[:2])
            rotated_vec = norm * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
            rotated_basis_vectors.append(rotated_vec)
        rotated_basis_vectors = np.array(rotated_basis_vectors)
        print(f"rotated_basis_vectors = {rotated_basis_vectors}")

        # Compute final basis vectors by averaging
        # theta = 30 if m ==2 else 0
        # scale = np.sqrt(3) if m ==2 else 1
        theta = 30 if self.type_structure[layer] == 1 else 0
        scale = np.sqrt(3) if self.type_structure[layer] == 1 else 1
        if np.degrees(np.arctan2(rotated_basis_vectors[0,1], rotated_basis_vectors[0,0])) > 30*0.95:
            theta = - theta

        final_a1 = np.mean(rotated_basis_vectors, axis=0)
        final_a1 = self.rot_z(final_a1, np.radians(theta))*scale
        final_a2 = self.rot_z(final_a1, np.pi / 3)  # Rotate a1 by 60 degrees

        self.lattice_vectors[layer] = np.array([final_a1, final_a2])
        # print("rotated_basis_vectors = ", rotated_basis_vectors)
        # print("theta = ", theta)
        # print(f"Final basis vectors:\na1 = {final_a1}\na2 = {final_a2}")
        # exit()
        return final_a1, final_a2


    def _determine_lattice_vectors_from_clustered_peaks(self, vectors, angles, tolerance=np.radians(5), type_structure=1):
        """
        Determines lattice vectors based on clustered candidate vectors using DBSCAN.

        Args:
            vectors (np.ndarray): Candidate vectors (M, 3).
            angles (np.ndarray): Angles of candidate vectors in radians (M,).
            tolerance (float): Tolerance in radians for DBSCAN clustering.
            type_structure (int): 1 for graphene-like, 2 for MoTe2-like structures.

        Returns:
            np.ndarray: Array containing two lattice vectors [a1, a2].
        """
        # Step 1: Cluster the candidate vectors based on angle % (2*pi/3) radians (120 degrees)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        angles = np.deg2rad(angles)
        angles_mod120 = np.mod(angles, 2 * np.pi / 3)
        unit_vectors = np.vstack((np.cos(angles_mod120), np.sin(angles_mod120))).T

        # DBSCAN clustering
        db = DBSCAN(eps=0.2, min_samples=5)  # eps set to 0.2 radians
        labels = db.fit_predict(unit_vectors)

        # Remove noise
        valid_mask = labels != -1
        valid_labels = labels[valid_mask]
        valid_vectors = vectors[valid_mask]
        valid_angles_mod120 = angles_mod120[valid_mask]

        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        num_clusters = len(unique_labels)

        # Check if number of clusters is multiple of 3
        if num_clusters % 3 != 0:
            print(f"Number of clusters ({num_clusters}) is not a multiple of 3.")
            print(f"Unique labels: {unique_labels}")
            raise ValueError(f"Number of clusters ({num_clusters}) is not a multiple of 3.")

        # Determine m (m=1 or m=2)
        m = num_clusters // 3
        if m not in [1, 2]:
            raise ValueError(f"Detected m={m}, expected m=1 or m=2.")

        # Sort clusters by size and select top 3m clusters
        sorted_indices = np.argsort(counts)[::-1]
        top_labels = unique_labels[sorted_indices[:3 * m]]
        top_counts = counts[sorted_indices[:3 * m]]

        # Check cluster size consistency (within 20%)
        if np.max(top_counts) / np.min(top_counts) > 1.2:
            raise ValueError("Selected clusters have significantly different sizes.")

        # Step 2: Assign clusters to classes
        if m == 1:
            # Single class
            class_assignments = {label: 0 for label in top_labels}
        else:
            # m=2, further cluster the top_labels into 2 classes using DBSCAN on their average angles
            cluster_avg_angles = []
            for label in top_labels:
                cluster_angles = angles_mod120[labels == label]
                avg_angle = np.mean(cluster_angles)
                cluster_avg_angles.append(avg_angle)
            cluster_avg_angles = np.array(cluster_avg_angles)

            # Convert average angles to unit vectors for clustering
            unit_vectors_avg_angles = np.vstack((np.cos(cluster_avg_angles), np.sin(cluster_avg_angles))).T

            db_final = DBSCAN(eps=tolerance, min_samples=2)
            db_final.fit(unit_vectors_avg_angles)
            final_labels = db_final.labels_

            # Remove noise
            final_mask = final_labels != -1
            final_unique_labels, final_counts = np.unique(final_labels[final_mask], return_counts=True)

            if len(final_unique_labels) not in [1, 2]:
                raise ValueError(f"After secondary clustering, number of classes ({len(final_unique_labels)}) is not 1 or 2.")

            if len(final_unique_labels) == 2:
                # Check if angle difference is ~60 degrees (pi/3 radians)
                class_angles = []
                for flabel in final_unique_labels:
                    class_angles.append(np.mean(cluster_avg_angles[final_labels == flabel]))
                class_angles = np.array(class_angles)
                angle_diff = np.abs(class_angles[0] - class_angles[1]) % (2 * np.pi / 3)
                if not np.isclose(angle_diff, np.pi / 3, atol=np.radians(10)):
                    raise ValueError(f"Angle difference between classes is {np.degrees(angle_diff):.2f} degrees, expected ~60 degrees.")

                # Assign clusters to classes
                class_assignments = {}
                for i, flabel in enumerate(final_unique_labels):
                    class_assignments[top_labels[i]] = i
            else:
                # Only one class
                class_assignments = {label: 0 for label in top_labels}

        # Step 3: Compute average vectors for each selected cluster
        basis_vectors = []
        for label in top_labels:
            class_id = class_assignments[label]
            cluster_vectors = vectors[labels == label]
            avg_vector = np.mean(cluster_vectors, axis=0)
            basis_vectors.append(avg_vector)
        basis_vectors = np.array(basis_vectors)

        # Step 4: Rotate basis vectors to 0-theta range
        theta = 2 * np.pi / 3 if m == 1 else np.pi / 3  # 120 degrees or 60 degrees
        rotated_basis_vectors = []
        for vec in basis_vectors:
            angle = np.arctan2(vec[1], vec[0]) % theta
            norm = np.linalg.norm(vec[:2])
            rotated_vec = norm * np.array([np.cos(angle), np.sin(angle)])
            rotated_basis_vectors.append(rotated_vec)
        rotated_basis_vectors = np.array(rotated_basis_vectors)

        # Compute final basis vectors by averaging
        final_a1 = np.mean(rotated_basis_vectors, axis=0)
        final_a2 = self.rot_z(final_a1, np.pi / 3)  # Rotate a1 by 60 degrees

        print(f"Final basis vectors:\na1 = {final_a1}\na2 = {final_a2}")

        # Step 5: Plotting
        # Compute all vectors' norm and angles
        vec_norms = np.linalg.norm(vectors[:, :2], axis=1)
        vec_angles_rad = np.arctan2(vectors[:, 1], vectors[:, 0])

        # Compute z deviations
        vec_z = vectors[:, 2]
        mean_z = np.mean(vec_z)
        z_deviation = np.abs(vec_z - mean_z)

        # Prepare selected vectors
        selected_mask = np.isin(labels, top_labels)
        selected_vectors = vectors[selected_mask]
        selected_angles_mod120 = angles_mod120[selected_mask]
        selected_z_deviation = z_deviation[selected_mask]

        # Assign class labels to selected vectors
        if m == 1:
            class_labels = np.zeros(len(selected_vectors), dtype=int)
        else:
            # Assign classes based on class_assignments
            cluster_labels = labels[selected_mask]
            class_labels = np.array([class_assignments[label] for label in cluster_labels])

        # Define colors for classes
        num_classes = len(np.unique(class_labels))
        cmap = plt.cm.get_cmap('viridis', num_classes)
        colors = cmap(class_labels)

        # Adjust color intensity based on z deviation (higher deviation -> darker)
        z_norm = selected_z_deviation / np.max(selected_z_deviation) if np.max(selected_z_deviation) != 0 else selected_z_deviation
        # Modify colors: decrease brightness based on z deviation
        colors_rgb = colors[:, :3] * (1 - 0.5 * z_norm[:, np.newaxis]) + 0.5 * z_norm[:, np.newaxis]  # Mix with white
        colors = colors_rgb

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot all vectors in light gray
        all_x = vec_norms * np.cos(vec_angles_rad)
        all_y = vec_norms * np.sin(vec_angles_rad)
        ax.scatter(all_x, all_y, c='lightgray', s=10, alpha=0.5, label='All Vectors')

        # Plot selected vectors with colors
        selected_x = selected_vectors[:, 0]
        selected_y = selected_vectors[:, 1]
        scatter = ax.scatter(selected_x, selected_y, c=colors, s=20, label='Selected Vectors')

        # Plot basis vectors
        ax.quiver(0, 0, final_a1[0], final_a1[1], angles='xy', scale_units='xy', scale=1,
                color='red', alpha=0.7, width=0.005, label='a1')
        ax.quiver(0, 0, final_a2[0], final_a2[1], angles='xy', scale_units='xy', scale=1,
                color='blue', alpha=0.7, width=0.005, label='a2')

        # Add polar grid lines every 60 and 30 degrees
        max_norm = np.max(vec_norms) * 1.1
        for angle_rad in [0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6,
                        np.pi, 7 * np.pi / 6, 4 * np.pi / 3, 3 * np.pi / 2, 5 * np.pi / 3, 11 * np.pi / 6]:
            linestyle = '-' if np.isclose(angle_rad % (2 * np.pi / 6), 0, atol=1e-3) else '--'
            linewidth = 1 if np.isclose(angle_rad % (2 * np.pi / 6), 0, atol=1e-3) else 0.5
            color = 'black' if np.isclose(angle_rad % (2 * np.pi / 6), 0, atol=1e-3) else 'gray'
            ax.plot([0, max_norm * np.cos(angle_rad)], [0, max_norm * np.sin(angle_rad)],
                    color=color, linestyle=linestyle, linewidth=linewidth)

        # Label average angle for each class
        if m >= 1:
            if m == 1:
                class_avg_angle = np.arctan2(final_a1[1], final_a1[0]) % (2 * np.pi / 3)
                class_avg_angle_deg = np.degrees(class_avg_angle)
                ax.text(final_a1[0], final_a1[1], f'{class_avg_angle_deg:.1f}°', color='red',
                        fontsize=12, ha='center', va='center')
            else:
                for i in range(num_classes):
                    # Compute average vector for class i
                    class_vectors = basis_vectors[class_labels == i]
                    avg_vector = np.mean(class_vectors, axis=0)
                    avg_angle = np.arctan2(avg_vector[1], avg_vector[0]) % (np.pi / 3)
                    avg_angle_deg = np.degrees(avg_angle)
                    ax.text(avg_vector[0], avg_vector[1],
                            f'{avg_angle_deg:.1f}°', color='red' if i == 0 else 'blue',
                            fontsize=12, ha='center', va='center')

        # Add histogram in the lower right
        ax_hist = inset_axes(ax, width="30%", height="30%", loc='lower right')
        ax_hist.hist(angles_mod120, bins=np.linspace(0, 2 * np.pi / 3, 7), color='gray', edgecolor='black')
        ax_hist.set_xlabel('Angle (radians)', fontsize=8)
        ax_hist.set_ylabel('Count', fontsize=8)
        ax_hist.set_title('Angle Distribution', fontsize=10)
        ax_hist.tick_params(axis='both', which='major', labelsize=8)

        # Set labels and title
        ax.set_xlabel('norm(vec) * cos(angle)')
        ax.set_ylabel('norm(vec) * sin(angle)')
        ax.set_title('Clustering of Candidate Vectors and Lattice Basis')

        # Add legend
        ax.legend()

        # Show plot
        plt.show()

        return np.array([final_a1, final_a2])

    def rot_z(self,vec,theta):
        """
        逆时针旋转theta角度
        """
        R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        return np.dot(R,vec)

    def get_reciprocal_vectors(self, layer):
        """
        Retrieves the reciprocal lattice vectors for a specified layer.

        Args:
            layer (int): The layer number.

        Returns:
            list: [b1, b2] reciprocal lattice vectors.
        """
        if layer not in self.reciprocal_vectors:
            raise ValueError(f"Reciprocal vectors for layer {layer} not found.")
        return self.reciprocal_vectors[layer]

    def get_lattice_vectors(self, layer):
        """
        Retrieves the real-space lattice vectors for a specified layer.

        Args:
            layer (int): The layer number.

        Returns:
            list: [a1, a2] real-space lattice vectors.
        """
        if layer not in self.lattice_vectors:
            raise ValueError(f"Lattice vectors for layer {layer} not found.")
        return self.lattice_vectors[layer]


class LayeredLatticeAnalyzerSpglib:
    """
    A simplified LayeredLatticeAnalyzer using spglib for automatic primitive cell detection.
    
    This class separates atoms into layers and uses spglib to identify the lattice vectors
    for each layer, avoiding complex DBSCAN-based geometric analysis.
    
    Key simplifications:
    - No manual bravais type specification (spglib detects automatically)
    - No type_structure for graphene-like vs MoTe2-like (spglib handles)
    - No candidate vector storage (DBSCAN-specific)
    """
    
    def __init__(self, input_data, num_layers, twist_layer, Tmat=None):
        """
        Initializes the LayeredLatticeAnalyzerSpglib with input data and number of layers.

        Args:
            input_data (list of dicts): Each dict should have 'original_index', 'species', and coordinates.
            num_layers (int): Number of layers to separate the atoms into based on z-coordinate.
            twist_layer (list of int): Layer grouping, e.g. [1, 3] means 1 bottom layer + 3 top layers.
            Tmat (np.ndarray): Initial moiré supercell lattice matrix (3,3) for reference.
        """
        self.input_data = input_data
        self.num_layers = num_layers
        self.twist_layer = twist_layer
        self.Tmat = Tmat if Tmat is not None else np.eye(3)
        
        # Core data structures
        self.df = None
        self.lattice_vectors = {}      # {layer: [a1, a2]}
        self.reciprocal_vectors = {}   # {layer: [b1, b2]}
        
    def process(self, TAPW=True):
        """
        Runs the processing workflow to derive lattice vectors and reciprocal lattice vectors.
        """
        self.load_data()
        if TAPW:
            self.separate_layers()
            self.compute_lattice_vectors_spglib()
    
    def load_data(self):
        """
        Loads input data into a pandas DataFrame.
        """
        data = []
        for item in self.input_data:
            index = item['original_index']
            species = item['species']
            x, y, z = item['x'], item['y'], item['z']
            orb_num = item['orb_num']
            orb_name = item['orb_name']
            orb_global_index = item['orb_global_index']
            data.append({
                'original_index': index, 'species': species, 
                'x': x, 'y': y, 'z': z, 
                'orb_num': orb_num, 'orb_name': orb_name, 
                'orb_global_index': orb_global_index
            })
        self.df = pd.DataFrame(data)
        print(f"[Spglib] Loaded {len(self.df)} atoms into DataFrame.")
    
    def separate_layers(self):
        """
        Separates atoms into the specified number of layers based on their z-coordinate using K-Means clustering.
        """
        def assign_twist_groups(layer_indices, twist_layer):
            """Assign layers to twist groups."""
            group_labels = []
            for group, count in enumerate(twist_layer):
                for _ in range(count):
                    group_labels.append(group)
            return {layer: group_labels[i] for i, layer in enumerate(layer_indices)}
        
        if self.df is None:
            raise ValueError("DataFrame is empty. Please load data first.")
        
        # Extract z coordinates and reshape for clustering
        z_coords = self.df['z'].values.reshape(-1, 1)
        
        # Initialize K-Means with the desired number of clusters (layers)
        kmeans = KMeans(n_clusters=self.num_layers, random_state=0, n_init='auto')
        
        # Fit K-Means and predict cluster labels
        labels = kmeans.fit_predict(z_coords)
        self.df['layer'] = labels
        
        # Sort layers by mean z-coordinate
        layer_means = self.df.groupby('layer')['z'].mean().sort_values().index.tolist()
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(layer_means)}
        self.df['layer'] = self.df['layer'].map(label_mapping)
        
        # Assign twist groups
        group_mapping = assign_twist_groups(sorted(label_mapping.values()), self.twist_layer)
        self.df['layer'] = self.df['layer'].map(group_mapping)
        
        # Update input_data
        for i, item in enumerate(self.input_data):
            self.input_data[i]['layer'] = self.df.loc[i, 'layer']
        
        print(f"[Spglib] Separated into {len(self.twist_layer)} twist groups with {self.df.shape[0]} atoms.")
        for i in range(len(self.twist_layer)):
            num_atoms = self.df[self.df['layer'] == i].shape[0]
            print(f"[Spglib] Layer {i}: {num_atoms} atoms.")
    
    def compute_lattice_vectors_spglib(self, symprec=4e-1):
        """
        Computes lattice vectors for each layer using spglib's primitive cell detection.
        
        Args:
            symprec (float): Symmetry precision for spglib (default: 1e-1 Å)
        """
        try:
            import spglib
        except ImportError:
            raise ImportError("spglib is required for LayeredLatticeAnalyzerSpglib. Install with: pip install spglib")
        
        if 'layer' not in self.df.columns:
            raise ValueError("Layers are not separated. Please run separate_layers() first.")
        
        # Create element to atomic number mapping
        from ase.data import chemical_symbols, atomic_numbers
        symbol_to_Z = {sym: Z for Z, sym in enumerate(chemical_symbols)}
        
        twist_angle_list = []
        
        for layer in range(len(self.twist_layer)):
            layer_df = self.df[self.df['layer'] == layer]
            
            if layer_df.empty:
                raise ValueError(f"Layer {layer}: no atoms found.")
            
            # Build supercell for this layer
            # Use moiré supercell lattice as initial guess
            latS = self.Tmat.copy()
            latS[2, 2] = 50.0  # Large z-direction for 2D material
            
            # Get atom positions and convert to fractional coordinates
            posS_cart = layer_df[['x', 'y', 'z']].values
            posS_frac = np.linalg.solve(latS.T, posS_cart.T).T
            
            # Get atomic numbers
            species_list = layer_df['species'].values
            numS = np.array([symbol_to_Z.get(s, atomic_numbers.get(s, 0)) for s in species_list])
            
            # Call spglib to find primitive cell
            cellS = (latS, posS_frac, numS)
            # Print summary only (not full cellS with all atom positions)
            print(f"[Spglib] Layer {layer}: {len(numS)} atoms, lattice shape {latS.shape}, positions shape {posS_frac.shape}")
            prim = spglib.standardize_cell(
                cellS, 
                to_primitive=True, 
                no_idealize=True, 
                symprec=symprec
            )
            
            if prim is None:
                print(f"[Spglib] Warning: Layer {layer} primitive cell detection failed with symprec={symprec}")
                print(f"[Spglib] Attempting with larger symprec=0.5...")
                prim = spglib.standardize_cell(cellS, to_primitive=True, no_idealize=True, symprec=0.5)
                
                if prim is None:
                    raise RuntimeError(
                        f"Layer {layer}: spglib failed to identify primitive cell.\n"
                        f"Try adjusting symprec or use the original LayeredLatticeAnalyzer."
                    )
            
            latP, posP, numP = prim
            
            # Extract 2D lattice vectors (in-plane only)
            a1 = latP[0, :2]
            a2 = latP[1, :2]
            
            self.lattice_vectors[layer] = [a1, a2]
            twist_angle_list.append(np.degrees(np.arctan2(a1[1], a1[0])))
            
            # Compute reciprocal lattice vectors
            area = a1[0] * a2[1] - a1[1] * a2[0]
            if area == 0:
                raise ValueError(f"Layer {layer}: Lattice vectors are collinear. Cannot compute reciprocal vectors.")
            
            b1 = (2 * np.pi / area) * np.array([a2[1], -a2[0]])
            b2 = (2 * np.pi / area) * np.array([-a1[1], a1[0]])
            self.reciprocal_vectors[layer] = [b1, b2]
            
            # Print results
            print(f"\n[Spglib] ======= Layer {layer}: Lattice vectors (Angstrom) =====")
            print(np.array([a1, a2]))
            print(f"[Spglib] ======= Layer {layer}: Reciprocal lattice vectors (1/Angstrom) =====")
            print(np.array([b1, b2]))
            
            # Self-check (useful for any lattice type)
            print(f"[Spglib] ======= Lattice self-check =====")
            print(f"a1·a2 = {float(np.dot(a1, a2)):.6e}  (0 = orthogonal)")
            print(f"|a1| = {float(np.linalg.norm(a1)):.6f} Å")
            print(f"|a2| = {float(np.linalg.norm(a2)):.6f} Å")
            print(f"|a1|/|a2| = {float(np.linalg.norm(a1)/np.linalg.norm(a2)):.6f}")
            print(f"[Spglib] Primitive cell: {len(numP)} atoms (from {len(numS)} in supercell layer)")
        
        # Calculate twist angles
        twist_angle_list = np.diff(twist_angle_list)
        print("\n[Spglib] ======================= Calculated twist angle ========================")
        for i in range(len(twist_angle_list)):
            print(f"[Spglib] The twist angle between layer {i} and layer {i+1} is {twist_angle_list[i]:.4f}°")
        print(f"[Spglib] ======================= End lattice_vectors computation ========================\n")
    
    def plot_lattice(self, layer=None, xlim=None, ylim=None, distance_threshold=None, save_path=None, save=False):
        """
        Plots the atomic distribution and lattice vectors for a specified layer.
        Shows all periodic repetitions of the primitive cell within the supercell.
        """
        if plt is None:
            raise ImportError("matplotlib is required for plotting.")
        
        layers_to_plot = [layer] if layer is not None else sorted(self.lattice_vectors.keys())
        num_layers = len(layers_to_plot)
        
        rows = (num_layers + 1) // 2
        cols = 2 if num_layers > 1 else 1
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        axes = np.array(axes).reshape(-1)
        
        for idx, layer in enumerate(layers_to_plot):
            ax = axes[idx]
            
            if layer not in self.lattice_vectors:
                raise ValueError(f"Lattice vectors for layer {layer} not found.")
            
            layer_df = self.df[self.df['layer'] == layer]
            layer_atoms = layer_df[['x', 'y']].values
            a1, a2 = self.lattice_vectors[layer]
            
            # Draw moiré supercell boundary (from Tmat)
            # Tmat rows are lattice vectors: [a1_super, a2_super, a3_super]
            a1_super = self.Tmat[0, :2]  # In-plane x,y components
            a2_super = self.Tmat[1, :2]
            
            # Draw supercell as a parallelogram
            supercell_corners = np.array([
                [0, 0],
                a1_super,
                a1_super + a2_super,
                a2_super,
                [0, 0]  # Close the loop
            ])
            ax.plot(supercell_corners[:, 0], supercell_corners[:, 1], 
                   'k-', linewidth=2.5, label='Moiré Supercell', zorder=1)  # Lower zorder to stay below legend
            
            # Plot atoms by species with different colors
            # Use z-coordinate to vary point size for overlapping xy positions
            species_list = layer_df['species'].unique()
            colors = plt.get_cmap('tab10')
            
            # Get z-range for this layer to normalize sizes
            z_min, z_max = layer_df['z'].min(), layer_df['z'].max()
            z_range = z_max - z_min if z_max > z_min else 1.0
            
            for i, species in enumerate(species_list):
                species_mask = layer_df['species'] == species
                species_data = layer_df[species_mask]
                species_atoms = species_data[['x', 'y']].values
                species_z = species_data['z'].values
                
                # Vary size based on z: higher z → larger points
                if z_range > 0.01:  # If there's significant z variation
                    z_normalized = (species_z - z_min) / z_range  # 0 to 1
                    sizes = 15 + 25 * z_normalized  # Size from 15 to 40
                    alphas = 0.5 + 0.4 * z_normalized  # Alpha from 0.5 to 0.9
                    
                    # Plot with size/alpha encoding z
                    for j, (x, y, s, a) in enumerate(zip(species_atoms[:, 0], species_atoms[:, 1], sizes, alphas)):
                        ax.scatter(x, y, s=s, color=colors(i % 10), alpha=a, 
                                 edgecolors='black', linewidths=0.5,
                                 label=species if j == 0 else None)  # Only label first point
                else:
                    # No z variation, use uniform size
                    ax.scatter(species_atoms[:, 0], species_atoms[:, 1], 
                             s=20, color=colors(i % 10), alpha=0.7, 
                             label=species, edgecolors='black', linewidths=0.5)
            
            # Determine threshold for plotting periodic cells
            if distance_threshold is None:
                distance_threshold = np.max(np.linalg.norm(layer_atoms, axis=1)) * 1.3
            
            # Calculate how many periods to show
            max_i = int(np.ceil(distance_threshold / np.linalg.norm(a1))) + 1
            max_j = int(np.ceil(distance_threshold / np.linalg.norm(a2))) + 1
            
            # Generate periodic lattice vectors using KDTree for filtering
            i_values = np.arange(-max_i, max_i + 1)
            j_values = np.arange(-max_j, max_j + 1)
            ii, jj = np.meshgrid(i_values, j_values)
            
            # All possible translation vectors
            translation_vectors = ii.flatten()[:, np.newaxis] * a1 + jj.flatten()[:, np.newaxis] * a2
            
            # Use KDTree to find translations close to atoms
            tree = cKDTree(layer_atoms)
            distances, _ = tree.query(translation_vectors, k=1)
            
            # Select translations near atoms (within a threshold)
            mask = distances < 3.5  # Same threshold as original
            selected_translations = translation_vectors[mask]
            
            # Prepare vector arrays for batch plotting
            U_a1 = np.tile(a1[0], len(selected_translations))
            V_a1 = np.tile(a1[1], len(selected_translations))
            U_a2 = np.tile(a2[0], len(selected_translations))
            V_a2 = np.tile(a2[1], len(selected_translations))
            
            # Plot all a1 vectors (red) - only label the first one
            if len(selected_translations) > 0:
                # Plot first vector with label
                ax.quiver(
                    selected_translations[0, 0], selected_translations[0, 1],
                    a1[0], a1[1],
                    angles='xy', scale_units='xy', scale=1,
                    color='r', alpha=0.5, width=0.003,
                    label='a1'
                )
                # Plot rest without label
                if len(selected_translations) > 1:
                    ax.quiver(
                        selected_translations[1:, 0], selected_translations[1:, 1],
                        U_a1[1:], V_a1[1:],
                        angles='xy', scale_units='xy', scale=1,
                        color='r', alpha=0.3, width=0.002
                    )
            
            # Plot all a2 vectors (blue) - only label the first one
            if len(selected_translations) > 0:
                # Plot first vector with label
                ax.quiver(
                    selected_translations[0, 0], selected_translations[0, 1],
                    a2[0], a2[1],
                    angles='xy', scale_units='xy', scale=1,
                    color='b', alpha=0.5, width=0.003,
                    label='a2'
                )
                # Plot rest without label
                if len(selected_translations) > 1:
                    ax.quiver(
                        selected_translations[1:, 0], selected_translations[1:, 1],
                        U_a2[1:], V_a2[1:],
                        angles='xy', scale_units='xy', scale=1,
                        color='b', alpha=0.3, width=0.002
                    )
            
            # Set limits
            if xlim is None:
                xlim_layer = [np.min(layer_atoms[:, 0]) - 5, np.max(layer_atoms[:, 0]) + 5]
            else:
                xlim_layer = xlim
            
            if ylim is None:
                ylim_layer = [np.min(layer_atoms[:, 1]) - 5, np.max(layer_atoms[:, 1]) + 5]
            else:
                ylim_layer = ylim
            
            ax.set_xlim(xlim_layer)
            ax.set_ylim(ylim_layer)
            ax.set_aspect('equal')
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            
            # Count atoms in this layer
            n_atoms_layer = len(layer_df)
            ax.set_title(f'Layer {layer} ({n_atoms_layer} atoms)')
            
            # Legend with solid white background to avoid overlap
            legend = ax.legend(loc='upper right', fontsize=8, framealpha=0.8, ncol=2, 
                             fancybox=True, shadow=False, frameon=True)
            legend.set_zorder(100)  # Put legend on top of everything
        
        # Hide extra subplots
        for extra_ax in axes[num_layers:]:
            extra_ax.axis('off')
        
        plt.tight_layout()
        
        if save:
            if save_path is None:
                raise ValueError("Please provide a save_path to save the plot.")
            plt.savefig(f'{save_path}/lattice.pdf', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_nearest_vectors_phase(self, **kwargs):
        """Compatibility hook; the spglib reader does not implement this backend."""
        print("[Spglib] plot_nearest_vectors_phase is not implemented in spglib version.")
        print("[Spglib] Use the original LayeredLatticeAnalyzer if this functionality is needed.")
    
    def get_reciprocal_vectors(self, layer):
        """Retrieves the reciprocal lattice vectors for a specified layer."""
        if layer not in self.reciprocal_vectors:
            raise ValueError(f"Reciprocal vectors for layer {layer} not found.")
        return self.reciprocal_vectors[layer]
    
    def get_lattice_vectors(self, layer):
        """Retrieves the real-space lattice vectors for a specified layer."""
        if layer not in self.lattice_vectors:
            raise ValueError(f"Lattice vectors for layer {layer} not found.")
        return self.lattice_vectors[layer]


class StructureProcessor:
    """
    A class to process atomic structures by clustering atoms into layers, sublayers, and atom types,
    and aligning their coordinates based on phase information.

    Attributes:
        input_data (list of dict): List containing atom data with 'original_index', 'species', and 'r' (coordinates).
        num_layers (int): Number of layers to divide the structure into.
        monolayer_reciprocal_list (np.ndarray): Reciprocal lattice vectors for each layer.
        layer_eps (float): DBSCAN epsilon parameter for layer clustering.
        layer_min_samples (int): DBSCAN min_samples parameter for layer clustering.
        sublayer_eps (float): DBSCAN epsilon parameter for sublayer clustering.
        sublayer_min_samples (int): DBSCAN min_samples parameter for sublayer clustering.
        atom_eps (float): DBSCAN epsilon parameter for atom type clustering.
        atom_min_samples (int): DBSCAN min_samples parameter for atom type clustering.
        period (float): Phase period, default is 2π.
        k_max (int): Maximum integer multiples of 2π to consider for coordinate alignment.
        df (pd.DataFrame): DataFrame containing processed atomic data.
        phase1 (np.ndarray): Phase data corresponding to reciprocal lattice vector b1.
        phase2 (np.ndarray): Phase data corresponding to reciprocal lattice vector b2.
    """

    def __init__(self, input_data, num_layers, monolayer_reciprocal_list, twist_layer,
                 layer_eps=0.5, layer_min_samples=1,
                 sublayer_eps=0.5, sublayer_min_samples=1,
                 atom_eps=0.3, atom_min_samples=5,
                 period=2*np.pi, k_max=2,
                 spin=None, twist_index=None, Tmat=None, reciprocal_Tmat=None):
        """
        Initializes the StructureProcessor with necessary parameters.

        Parameters:
            input_data (list of dict): List containing atom data with 'original_index', 'species', and 'r' (coordinates).
            num_layers (int): Number of layers to divide the structure into.
            monolayer_reciprocal_list (list of lists): Reciprocal lattice vectors for each layer, e.g., [[b1, b2], [b1, b2], ...].
            twist_layer (list of int): The layer index where the twist occurs. e.g. [1, 3] for twist between layer 1 and the other 3 layers above.
            layer_eps (float): DBSCAN epsilon parameter for layer clustering.
            layer_min_samples (int): DBSCAN min_samples parameter for layer clustering.
            sublayer_eps (float): DBSCAN epsilon parameter for sublayer clustering.
            sublayer_min_samples (int): DBSCAN min_samples parameter for sublayer clustering.
            atom_eps (float): DBSCAN epsilon parameter for atom type clustering.
            atom_min_samples (int): DBSCAN min_samples parameter for atom type clustering.
            period (float): Phase period, default is 2π.
            k_max (int): Maximum integer multiples of 2π to consider for coordinate alignment.
        """
        self.input_data = input_data
        self.num_layers = num_layers
        self.monolayer_reciprocal_list = monolayer_reciprocal_list
        self.twist_layer = twist_layer
        self.layer_eps = layer_eps
        self.layer_min_samples = layer_min_samples
        self.sublayer_eps = sublayer_eps
        self.sublayer_min_samples = sublayer_min_samples
        self.atom_eps = atom_eps
        self.atom_min_samples = atom_min_samples
        self.period = period
        self.k_max = k_max
        self.df = None
        self.phase1 = None
        self.phase2 = None
        self.transformed_index_matrix = None

        self.spin = spin
        self.twist_index = twist_index
        self.Tmat = Tmat
        self.reciprocal_Tmat = reciprocal_Tmat
        
        self.load_data()
        print("Data loaded.")

    def load_data(self):
        """
        Loads input data into a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing 'original_index', 'species', 'x', 'y', 'z'.
        """
        data = []
        for item in self.input_data:
            index = item['original_index']
            species = item['species']
            x, y, z = item['x'], item['y'], item['z']
            orb_num = item['orb_num']
            orb_name = item['orb_name']
            global_orb_index = item['orb_global_index']
            try:
                layer_index = item['layer']
            except:
                layer_index = 0
            
            
            data.append({'original_index': index, 'species': species, 'x': x, 'y': y, 'z': z, 'orb_num': orb_num, 'orb_name': orb_name, 'orb_global_index': global_orb_index, 'layer': layer_index})

        self.df = pd.DataFrame(data)
        print(f"Loaded {len(self.df)} atoms into DataFrame.")

    def separate_layers1(self):
        """
        Separates atoms into specified number of layers based on their z-coordinate.

        Adds:
            'layer' column to self.df.
        """
        self.df = self.df.copy()
        self.df = self.df.sort_values(by='z').reset_index(drop=True)
        self.df['layer'] = pd.qcut(self.df['z'], q=self.num_layers, labels=False)
        print(f"Separated into {self.num_layers} layers with {self.df.shape[0]} atoms.")

    def separate_layers(self):
        """
        Separates atoms into the specified number of layers based on their z-coordinate using K-Means clustering.

        Adds:
            'layer' column to self.df.
        """
        def assign_twist_groups(layer_indices, twist_layer):
            """
            layer_indices: 已经按z均值排序后的层编号（如[0,1,2,3]）
            twist_layer: 例如[1,2,1]
            返回：每个层编号对应的组号
            """
            group_labels = []
            current = 0
            for group, count in enumerate(twist_layer):
                for _ in range(count):
                    group_labels.append(group)
                    current += 1
            return {layer: group_labels[i] for i, layer in enumerate(layer_indices)}

        if self.df is None:
            raise ValueError("DataFrame is empty. Please load data first.")
        
        # Extract z coordinates and reshape for clustering
        z_coords = self.df['z'].values.reshape(-1, 1)
        
        # Initialize K-Means with the desired number of clusters (layers)
        kmeans = KMeans(n_clusters=self.num_layers, random_state=0, n_init='auto')
        
        # Fit K-Means and predict cluster labels
        labels = kmeans.fit_predict(z_coords)
        
        # Assign cluster labels to the DataFrame
        self.df['layer'] = labels
        
        # Calculate the mean z-coordinate for each layer to sort layers from bottom to top
        layer_means = self.df.groupby('layer')['z'].mean().sort_values().index.tolist()
        
        # Create a mapping from old labels to new labels sorted by mean z-coordinate
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(layer_means)}
        
        # Apply the mapping to ensure layers are ordered from bottom to top
        self.df['layer'] = self.df['layer'].map(label_mapping)
        # self.df['layer'] = self.df['layer'].apply(lambda x: 0 if x < self.twist_layer[0] else 1)
        group_mapping = assign_twist_groups(sorted(label_mapping.values()), self.twist_layer)
        self.df['layer'] = self.df['layer'].map(group_mapping)
        for i, item in enumerate(self.input_data):
            self.input_data[i]['layer'] = self.df.loc[i, 'layer']
        # Print summary of layer separation
        # print(f"Separated into 2 layers with {self.df.shape[0]} atoms.")
        print(f"Separated into {len(self.twist_layer)} layers with {self.df.shape[0]} atoms.")
        for i in range(len(self.twist_layer)):
            num_atoms = self.df[self.df['layer'] == i].shape[0]
            print(f"Layer {i}: {num_atoms} atoms.")


    def cluster_sublayers(self):
        """
        Clusters atoms within each layer into sublayers using DBSCAN based on z-coordinate.

        Adds:
            'sublayer' column to self.df.
        """
        self.df = self.df.copy()
        self.df['sublayer'] = -1  # Initialize with -1
        layers = self.df['layer'].unique()
        for layer in layers:
            layer_mask = self.df['layer'] == layer
            z = self.df.loc[layer_mask, 'z'].values.reshape(-1, 1)
            db = DBSCAN(eps=self.sublayer_eps, min_samples=self.sublayer_min_samples)
            labels = db.fit_predict(z)
            self.df.loc[layer_mask, 'sublayer'] = labels
            num_atoms = len(labels)
            num_sublayers = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"Layer {layer}: {num_atoms} atoms clustered into {num_sublayers} sublayers.")

    def cluster_atom_types(self, plot=False):
        """
        Clusters atoms within each sublayer into atom types using DBSCAN based on phase data.

        Parameters:
            plot (bool): Whether to plot clustering results for each sublayer.

        Adds:
            'atom_type' column to self.df.
        """
        self.df = self.df.copy()
        self.df['atom_type'] = -1  # Initialize with -1
        current_label = 0  # Global atom type counter
        layers = sorted(self.df['layer'].unique())
        
        for layer in layers:
            sublayers = sorted(self.df[self.df['layer'] == layer]['sublayer'].unique())
            for sublayer in sublayers:
                print(f"Clustering Layer {layer}, Sublayer {sublayer} with {self.df[(self.df['layer'] == layer) & (self.df['sublayer'] == sublayer)].shape[0]} atoms.")
                mask = (self.df['layer'] == layer) & (self.df['sublayer'] == sublayer)
                if mask.sum() == 0:
                    raise ValueError(f"No atoms found for layer {layer}, sublayer {sublayer}")
                
                indices = self.df[mask].index

                # Normalize phase data to [0, period)
                phase_normalized1 = self.phase1[mask] % self.period
                phase_normalized2 = self.phase2[mask] % self.period

                # Convert phases to sine and cosine components for circular data handling
                theta1 = (phase_normalized1 / self.period) * 2 * np.pi
                theta2 = (phase_normalized2 / self.period) * 2 * np.pi
                phase_cos1 = np.cos(theta1)
                phase_sin1 = np.sin(theta1)
                phase_cos2 = np.cos(theta2)
                phase_sin2 = np.sin(theta2)
                phase_features = np.array([phase_cos1, phase_sin1, phase_cos2, phase_sin2]).T  # Shape: (n_atoms, 4)

                # Perform DBSCAN clustering
                db = DBSCAN(eps=self.atom_eps, min_samples=self.atom_min_samples)
                labels = db.fit_predict(phase_features)
                unique_labels = sorted(set(labels))
                print(f"Unique labels in clustering: {unique_labels}")

                for label in unique_labels:
                    if label == -1:
                        # Noise points, do not assign a new atom_type
                        continue
                    clustered_indices = indices[labels == label]
                    self.df.loc[clustered_indices, 'atom_type'] = current_label
                    current_label += 1

                if plot:
                    self.plot_atom_type_clustering(layer, sublayer, phase_normalized1, phase_normalized2, labels)

        # Check for unclustered atoms
        if (self.df['atom_type'] == -1).any():
            num_noise = (self.df['atom_type'] == -1).sum()
            print(f"Warning: {num_noise} atoms were identified as noise and not clustered into any atom_type.")

        # Verify that each atom_type corresponds to a single species
        atom_types = self.df['atom_type'].unique()
        for atom_type in atom_types:
            if atom_type == -1:
                continue
            species = self.df[self.df['atom_type'] == atom_type]['species'].unique()
            if len(species) != 1:
                raise ValueError(f"Atom type {atom_type} corresponds to multiple species: {species}")
        
        #sort by layer,sublayer,atom_type
        print("==========================================================================")
        print("================ Sorting by layer, sublayer, atom_type... ================")
        print("==========================================================================")
        self.df = self.df.sort_values(by=['layer','sublayer','atom_type']).reset_index(drop=True)
        print(self.df)

        
    def plot_atom_type_clustering(self, layer, sublayer, phase_normalized1, phase_normalized2, labels):
        """
        Plots the clustering results for a specific layer and sublayer.

        Parameters:
            layer (int): Layer index.
            sublayer (int): Sublayer index.
            phase_normalized1 (np.ndarray): Normalized phase1 data for the sublayer.
            phase_normalized2 (np.ndarray): Normalized phase2 data for the sublayer.
            labels (np.ndarray): Cluster labels from DBSCAN.
        """
        unique_labels_sorted = sorted(set(labels))
        cmap = matplotlib.colormaps['viridis']
        colors = cmap(np.linspace(0, 1, len(unique_labels_sorted)))

        fig, ax = plt.subplots(1, len(self.twist_layer), figsize=(12, 4))
        kx = np.arange(len(phase_normalized1))

        for i, label in enumerate(unique_labels_sorted):
            mask_atom = (labels == label)
            if label == -1:
                # Noise points
                color = 'k'
                label_name = 'Noise'
            else:
                color = colors[i]
                label_name = f'Cluster {label}'
            ax[0].scatter(kx[mask_atom], phase_normalized1[mask_atom], color=color, label=label_name, s=5)
            ax[1].scatter(kx[mask_atom], phase_normalized2[mask_atom], color=color, label=label_name, s=5)

        for i in range(len(self.twist_layer)):
            ax[i].axhline(0, color='black', linestyle='--')
            ax[i].axhline(self.period, color='black', linestyle='--')
            ax[i].set_xlabel('Atom Index')
            ax[i].set_ylabel('Phase Value')
            ax[i].set_title(f'Clustering of Phase Data\nLayer {layer}, Sublayer {sublayer}')
            ax[i].legend(fontsize='8')
            ax[i].set_ylim(0, self.period)
            ax[i].grid(True)

        plt.tight_layout()
        plt.show()

    def align_coordinates(self):
        """
        Aligns atom coordinates based on their atom_type and reciprocal lattice vectors.

        Adds:
            'delta_tau_x', 'delta_tau_y', 'shifted_x', 'shifted_y' columns to self.df.
        """
        self.df = self.df.copy()
        self.df['delta_tau_x'] = 0.0
        self.df['delta_tau_y'] = 0.0
        self.df['shifted_x'] = self.df['x']
        self.df['shifted_y'] = self.df['y']
        phase1 = self.df['phase1'].values
        phase2 = self.df['phase2'].values

        atom_types = self.df['atom_type'].unique()
        for atom_type in atom_types:
            if atom_type == -1:
                # Skip noise points
                continue
            mask = self.df['atom_type'] == atom_type
            if mask.sum() == 0:
                continue

            layer_index = self.df.loc[mask, 'layer'].values[0]
            b1, b2 = self.monolayer_reciprocal_list[layer_index]
            A = np.array([b1, b2])  # Shape: (2, 2)
            try:
                A_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                raise ValueError(f"Reciprocal lattice vectors for layer {layer_index} are linearly dependent and cannot be inverted.")

            # Calculate circular average phase
            phase1_avg = np.arctan2(np.mean(np.sin(phase1[mask])), np.mean(np.cos(phase1[mask]))) % self.period
            phase2_avg = np.arctan2(np.mean(np.sin(phase2[mask])), np.mean(np.cos(phase2[mask]))) % self.period

            indices = self.df[mask].index
            phase1_subset = phase1[mask]
            phase2_subset = phase2[mask]

            # Compute phase differences
            c1 = (phase1_avg - phase1_subset) % self.period
            c2 = (phase2_avg - phase2_subset) % self.period
            c = np.vstack([c1, c2])  # Shape: (2, n_atoms)

            # Initial displacement
            delta_tau0 = A_inv @ c  # Shape: (2, n_atoms)

            for i, idx in enumerate(indices):
                delta_tau_initial = delta_tau0[:, i]

                # Find the shift that minimizes the displacement norm
                min_norm = np.inf
                best_shift = None
                for k1 in range(-self.k_max, self.k_max + 1):
                    for k2 in range(-self.k_max, self.k_max + 1):
                        k = np.array([k1, k2])
                        delta = delta_tau_initial + A_inv @ (self.period * k)
                        norm = np.linalg.norm(delta)
                        if norm < min_norm:
                            min_norm = norm
                            best_shift = delta

                # Record displacement
                self.df.at[idx, 'delta_tau_x'] = best_shift[0]
                self.df.at[idx, 'delta_tau_y'] = best_shift[1]

                # Apply displacement
                self.df.at[idx, 'shifted_x'] = self.df.at[idx, 'x'] + best_shift[0]
                self.df.at[idx, 'shifted_y'] = self.df.at[idx, 'y'] + best_shift[1]

    def calculate_transformed_matrix(self):
        """
        Calculates the transformed index matrix based on the shifted coordinates.

        Returns:
            scipy.sparse.csr_matrix: Transformed index matrix.
        """
        # # 创建一个从原始索引到排序后索引的映射
        # original_indices = [atom['original_index'] for atom in self.species_coordinates]
        # sorted_indices = [atom['original_index'] for atom in self.sorted_species_coordinates]

        # # 创建一个字典，键为原始索引，值为排序后的位置
        # index_mapping = {original: sorted_pos for sorted_pos, original in enumerate(sorted_indices)}

        # # 初始化置换矩阵
        # N = self.atoms_number
        # P = np.zeros((N, N), dtype=int)

        # for new_pos, original in enumerate(original_indices):
        #     sorted_pos = index_mapping[original]
        #     if sorted_pos < N and new_pos < N:
        #         P[new_pos, sorted_pos] = 1

        # self.permutation_matrix = P

        #create a mapping from original index to sorted index
        # self.df = self.df.sort_values(by='atom_type')
        sorted_indices = self.df['orb_global_index'].explode().astype(int).tolist()
        sorted_indices = np.array(sorted_indices)
        df_original = self.df.copy()
        df_original = df_original.sort_values(by='original_index')
        original_indices = df_original['orb_global_index'].explode().astype(int).tolist()
        original_indices = np.array(original_indices)
        
        # transformed_index_matrix = sp.csr_matrix((np.ones(len(sorted_indices)), (sorted_indices, original_indices)), shape=(len(sorted_indices), len(original_indices))).toarray().T
        num_wann = len(sorted_indices)
        if self.spin:
            # self.transformed_index_matrix = scipy.linalg.block_diag(transformed_index_matrix, transformed_index_matrix)
            self.transformed_index_matrix = sp.csr_matrix((np.ones(num_wann*2), (np.concatenate([original_indices, original_indices+num_wann]), np.concatenate([sorted_indices, sorted_indices+num_wann]))), shape=(num_wann*2, num_wann*2))#.toarray().T
        else:
            self.transformed_index_matrix = sp.csr_matrix((np.ones(num_wann), (original_indices, sorted_indices)), shape=(num_wann, num_wann))
        self.transformed_index_matrix = sp.csr_matrix(self.transformed_index_matrix)
        print("=================== calculated transformed index matrix ===================")
        print(f"================== new orb_global_index = transformed_index_matrix{np.shape(self.transformed_index_matrix)} * old orb_global_index ===============")
        print("==================    new ord : atom_type   ===============================")

    
    def plot_clusters_loc(self, save_path=None, save=False):
        """
        Plots the clustering results in real space (shifted and original coordinates).
        save_path (str): Path to save the plot.
        save (bool): Whether to save the plot.
        """
        fig, ax = plt.subplots(1, 2, figsize=(15, 4))
        unique_labels = sorted(self.df['atom_type'].unique())
        colors = plt.get_cmap('tab10', len(unique_labels))

        for i, label in enumerate(unique_labels):
            mask = self.df['atom_type'] == label
            if label == -1:
                # Noise points
                ax[0].scatter(self.df.loc[mask, 'shifted_x'], self.df.loc[mask, 'shifted_y'], 
                              color='k', label='Noise', s=5)
                ax[1].scatter(self.df.loc[mask, 'x'], self.df.loc[mask, 'y'], 
                              color='k', label='Noise', s=5)
            else:
                species = self.df.loc[mask, 'species'].values[0]
                layer = self.df.loc[mask, 'layer'].values[0]
                sublayer = self.df.loc[mask, 'sublayer'].values[0]
                ax[0].scatter(self.df.loc[mask, 'shifted_x'], self.df.loc[mask, 'shifted_y'], 
                              color=colors(label % 10), 
                              label=f'Atom Type {label} ({species}) Layer {layer} Sublayer {sublayer}', 
                              s=5)
                ax[1].scatter(self.df.loc[mask, 'x'], self.df.loc[mask, 'y'], 
                              color=colors(label % 10), 
                              label=f'Atom Type {label} ({species}) Layer {layer} Sublayer {sublayer}', 
                              s=5)

        for i in range(2):
            ax[i].set_aspect('equal')
            ax[i].set_xlabel('X (Å)')
            ax[i].set_ylabel('Y (Å)')
            ax[i].legend(fontsize='8', loc='upper right')
            # ax[i].grid(True)
        ax[0].set_title('Shifted Coordinates')
        ax[1].set_title('Original Coordinates')

        plt.tight_layout()
        if save:
            if save_path is None:
                raise ValueError("Please provide a save path for the plot.")
            plt.savefig(save_path+'/culster_loc.pdf', dpi=300)
        plt.show()

    def plot_clusters_phase(self, save_path=None, save=False):
        """
        Plots the clustering results in phase space.
        save_path (str): Path to save the plot.
        save (bool): Whether to save the plot.
        """
        # phase1_deg = (self.phase1 * 180 / np.pi) % 360
        # phase2_deg = (self.phase2 * 180 / np.pi) % 360
        # period_deg = (self.period * 180 / np.pi) % 360

        phase1_deg = (self.df.phase1.values * 180 / np.pi) % 360
        phase2_deg = (self.df.phase2.values * 180 / np.pi) % 360
        period_deg = (self.period * 180 / np.pi) % 360


        fig, ax = plt.subplots(1, len(self.twist_layer), figsize=(15, 4))
        unique_labels = sorted(self.df['atom_type'].unique())
        colors = plt.get_cmap('tab10', len(unique_labels))

        kx = np.arange(self.df.shape[0])
        for i, label in enumerate(unique_labels):
            mask = self.df['atom_type'] == label
            if label == -1:
                # Noise points
                ax[0].scatter(kx[mask], phase1_deg[mask], color='k', label='Noise', s=5)
                ax[1].scatter(kx[mask], phase2_deg[mask], color='k', label='Noise', s=5)
            else:
                species = self.df.loc[mask, 'species'].values[0]
                layer = self.df.loc[mask, 'layer'].values[0]
                sublayer = self.df.loc[mask, 'sublayer'].values[0]
                ax[0].scatter(kx[mask], phase1_deg[mask], color=colors(label % 10), 
                              label=f'Atom Type {label} ({species}) Layer {layer} Sublayer {sublayer}', 
                              s=5)
                ax[1].scatter(kx[mask], phase2_deg[mask], color=colors(label % 10), 
                              label=f'Atom Type {label} ({species}) Layer {layer} Sublayer {sublayer}', 
                              s=5)

        for i in range(2):
            ax[i].axhline(0, color='black', linestyle='--')
            ax[i].axhline(period_deg, color='black', linestyle='--')
            ax[i].set_xlabel('Atom Index (sorted by z-coordinate)')
            ax[i].set_ylabel('Phase (Degrees)')
            ax[i].set_title(f'Phase Clustering Results b{i+1}')
            ax[i].legend(fontsize='8', loc='upper right')
            ax[i].grid(True)
            ax[i].yaxis.set_major_locator(plt.MultipleLocator(60))
            ax[i].set_ylim(-10, 370)

        plt.tight_layout()
        if save:
            if save_path is None:
                raise ValueError("Please provide a save path for the plot.")
            plt.savefig(save_path+'/culster_phase.pdf', dpi=300)
        plt.show()

    def process(self):
        """
        Executes the full processing workflow: loading data, separating layers, clustering sublayers,
        clustering atom types, aligning coordinates, and printing essential information.
        """
        # self.load_data()
        # print("Data loaded.")
        self.separate_layers()
        print("Layers separated.")
        self.cluster_sublayers()
        print("Sublayers clustered.")
        self.compute_phase()
        print("Phase data computed.")
        self.cluster_atom_types(plot=False)  # Set plot=True to visualize clustering per sublayer
        print("Atom types clustered.")
        self.align_coordinates()
        print("Coordinates aligned.")
        self.calculate_transformed_matrix()
        print("Transformed index matrix calculated.")
        self.print_summary()
        # self.df.sort_values(by=['atom_type'], inplace=True)

    def compute_phase(self):
        """
        Computes phase1 and phase2 based on reciprocal lattice vectors and atom positions.
        """
        self.phase1 = np.zeros(self.df.shape[0])
        self.phase2 = np.zeros(self.df.shape[0])
        for i in range(len(self.twist_layer)):
            b1, b2 = self.monolayer_reciprocal_list[i]
            mask = self.df['layer'] == i
            self.phase1[mask] = (self.df.loc[mask, 'x'] * b1[0] + self.df.loc[mask, 'y'] * b1[1]) % self.period
            self.phase2[mask] = (self.df.loc[mask, 'x'] * b2[0] + self.df.loc[mask, 'y'] * b2[1]) % self.period
        self.df['phase1'] = self.phase1
        self.df['phase2'] = self.phase2
        # print("Phase data computed.")

    def print_summary(self):
        """
        Prints a summary of the clustering results, including the number of layers, sublayers,
        atom types, and atom counts.
        """
        print("\n--- Clustering Summary ---")
        total_layers = self.df['layer'].nunique()
        print(f"Total Layers: {total_layers}")
        for layer in sorted(self.df['layer'].unique()):
            layer_df = self.df[self.df['layer'] == layer]
            num_sublayers = layer_df['sublayer'].nunique()
            print(f"\nLayer {layer}: {num_sublayers} Sublayers")
            for sublayer in sorted(layer_df['sublayer'].unique()):
                sublayer_df = layer_df[layer_df['sublayer'] == sublayer]
                atom_types = sublayer_df['atom_type'].unique()
                print(f"  Sublayer {sublayer}:")
                for atom_type in sorted(atom_types):
                    count = (sublayer_df['atom_type'] == atom_type).sum()
                    species = sublayer_df.loc[sublayer_df['atom_type'] == atom_type, 'species'].values
                    species = species[0] if len(species) > 0 else "Unknown"
                    if atom_type == -1:
                        label = "Noise"
                    else:
                        label = f"Atom Type {atom_type}"
                    print(f"    {label} ({species}): {count} atoms")
        print("--- End of Summary ---\n")


class StructureProcessorSpglib:
    """
    StructureProcessor variant that uses spglib to:
    - determine per-(twist)layer in-plane lattice vectors and reciprocal vectors
    - assign a spglib-derived `basis_id` (mapping_to_primitive) for each atom
    - derive `atom_type` from (layer, sublayer, basis_id) while preserving `sublayer`

    Notes:
    - `pos_frac` is *not* wrapped; it is passed to spglib as-is.
    - A sanity check enforces that each `basis_id` maps to exactly one chemical species
      inside the same layer group.
    """

    def __init__(
        self,
        input_data,
        num_layers,
        twist_layer,
        layer_eps=0.5,
        layer_min_samples=1,
        sublayer_eps=0.5,
        sublayer_min_samples=1,
        atom_eps=0.3,
        atom_min_samples=5,
        period=2 * np.pi,
        k_max=2,
        spin=None,
        twist_index=None,
        Tmat=None,
        reciprocal_Tmat=None,
        symprec: float = 10e-1,
        spglib_z_lattice: float = 50.0,
        reporter=None,
    ):
        self.input_data = input_data
        self.num_layers = num_layers
        self.monolayer_reciprocal_list = {}
        self.twist_layer = twist_layer
        self.layer_eps = layer_eps
        self.layer_min_samples = layer_min_samples
        self.sublayer_eps = sublayer_eps
        self.sublayer_min_samples = sublayer_min_samples
        self.atom_eps = atom_eps
        self.atom_min_samples = atom_min_samples
        self.period = period
        self.k_max = k_max
        self.df = None
        self.phase1 = None
        self.phase2 = None
        self.transformed_index_matrix = None

        self.spin = spin
        self.twist_index = twist_index
        self.Tmat = Tmat
        self.reciprocal_Tmat = reciprocal_Tmat

        self.symprec = float(symprec)
        self.spglib_z_lattice = float(spglib_z_lattice)
        self.layer_lattice_vectors = {}  # {layer: [a1, a2]}
        self.reporter = ensure_reporter(reporter)

        self.load_data()
        self.reporter.step("Structure table", "loaded atom coordinates into a DataFrame")

    def load_data(self):
        """
        Loads input data into a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing 'original_index', 'species', 'x', 'y', 'z'.
        """
        data = []
        for item in self.input_data:
            index = item['original_index']
            species = item['species']
            x, y, z = item['x'], item['y'], item['z']
            orb_num = item['orb_num']
            orb_name = item['orb_name']
            global_orb_index = item['orb_global_index']
            try:
                layer_index = item['layer']
            except Exception:
                layer_index = 0

            data.append({
                'original_index': index,
                'species': species,
                'x': x,
                'y': y,
                'z': z,
                'orb_num': orb_num,
                'orb_name': orb_name,
                'orb_global_index': global_orb_index,
                'layer': layer_index,
            })

        self.df = pd.DataFrame(data)
        self.reporter.kv("Loaded atoms", len(self.df))

    def separate_layers(self):
        """
        Separates atoms into physical layers (phys_layer) and assigns twist groups (twist_group).
        
        Physical layers (phys_layer): 0..N-1, ordered from bottom to top by z-coordinate.
        Twist groups (twist_group): 0..G-1, where G = len(twist_layer_counts).
        Each twist_group contains twist_layer_counts[gid] physical layers.
        
        Adds:
            'phys_layer': Physical layer index (0..N-1)
            'twist_group': Twist group index (0..G-1)
            'layer': Alias for phys_layer (for backward compatibility)
        """
        if self.df is None:
            raise ValueError("DataFrame is empty. Please load data first.")
        
        # Validate twist_layer_counts
        twist_layer_counts = self.twist_layer
        if sum(twist_layer_counts) != self.num_layers:
            raise ValueError(
                f"sum(twist_layer_counts)={sum(twist_layer_counts)} != num_layers={self.num_layers}. "
                f"twist_layer_counts={twist_layer_counts} must sum to num_layers."
            )
        
        # Extract z coordinates and reshape for clustering
        z_coords = self.df['z'].values.reshape(-1, 1)
        
        # Initialize K-Means with the desired number of clusters (physical layers)
        kmeans = KMeans(n_clusters=self.num_layers, random_state=0, n_init='auto')
        
        # Fit K-Means and predict cluster labels
        labels = kmeans.fit_predict(z_coords)
        self.df['phys_layer'] = labels
        
        # Calculate the mean z-coordinate for each layer to sort layers from bottom to top
        layer_means = self.df.groupby('phys_layer')['z'].mean().sort_values().index.tolist()
        
        # Create a mapping from old labels to new labels sorted by mean z-coordinate
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(layer_means)}
        
        # Apply the mapping to ensure layers are ordered from bottom to top
        self.df['phys_layer'] = self.df['phys_layer'].map(label_mapping)
        
        # Validate phys_layer covers 0..N-1
        unique_phys_layers = sorted(self.df['phys_layer'].unique())
        if unique_phys_layers != list(range(self.num_layers)):
            raise ValueError(
                f"phys_layer must cover 0..{self.num_layers-1}, got {unique_phys_layers}"
            )
        
        # Assign twist groups based on twist_layer_counts
        # group0 covers phys_layer 0..(count0-1)
        # group1 covers phys_layer count0..(count0+count1-1)
        # etc.
        self.df['twist_group'] = -1
        phys_layer_to_group = {}
        phys_layer_idx = 0
        for group_id, count in enumerate(twist_layer_counts):
            for _ in range(count):
                phys_layer_to_group[phys_layer_idx] = group_id
                phys_layer_idx += 1
        
        self.df['twist_group'] = self.df['phys_layer'].map(phys_layer_to_group)
        
        # Validate twist_group covers 0..G-1
        n_groups = len(twist_layer_counts)
        unique_groups = sorted(self.df['twist_group'].unique())
        if unique_groups != list(range(n_groups)):
            raise ValueError(
                f"twist_group must cover 0..{n_groups-1}, got {unique_groups}"
            )
        
        # Set 'layer' = 'phys_layer' for backward compatibility
        self.df['layer'] = self.df['phys_layer']
        
        # Propagate labels to input_data for downstream compatibility
        for i, _item in enumerate(self.input_data):
            self.input_data[i]['phys_layer'] = int(self.df.loc[i, 'phys_layer'])
            self.input_data[i]['twist_group'] = int(self.df.loc[i, 'twist_group'])
            self.input_data[i]['layer'] = int(self.df.loc[i, 'layer'])  # = phys_layer
        
        self.reporter.stage("Preprocessing", "Assign layers, source orientation groups, and TAPW projector ordering.")
        self.reporter.kv("Physical layers", self.num_layers)
        self.reporter.kv("Source orientation groups", n_groups)
        self.reporter.kv("Atoms", self.df.shape[0])
        self.reporter.kv("Twist layer counts", twist_layer_counts)
        for phys_layer in range(self.num_layers):
            num_atoms = len(self.df[self.df['phys_layer'] == phys_layer])
            group_id = phys_layer_to_group[phys_layer]
            self.reporter.step(f"Physical layer {phys_layer}", f"source orientation group {group_id}, {num_atoms} atoms")
        for group_id in range(n_groups):
            num_atoms = len(self.df[self.df['twist_group'] == group_id])
            self.reporter.step(f"Source orientation group {group_id}", f"{num_atoms} atoms")

    def cluster_sublayers(self):
        """
        Clusters atoms within each layer into sublayers using DBSCAN based on z-coordinate.

        Adds:
            'sublayer' column to self.df.
        """
        self.df = self.df.copy()
        self.df['sublayer'] = -1  # Initialize with -1
        layers = self.df['layer'].unique()
        for layer in layers:
            layer_mask = self.df['layer'] == layer
            z = self.df.loc[layer_mask, 'z'].values.reshape(-1, 1)
            db = DBSCAN(eps=self.sublayer_eps, min_samples=self.sublayer_min_samples)
            labels = db.fit_predict(z)
            self.df.loc[layer_mask, 'sublayer'] = labels
            num_atoms = len(labels)
            num_sublayers = len(set(labels)) - (1 if -1 in labels else 0)
            self.reporter.step(f"Layer {layer} sublayer clustering", f"{num_atoms} atoms -> {num_sublayers} sublayers")

    def compute_phase(self):
        """
        Computes phase1 and phase2 based on reciprocal lattice vectors and atom positions.
        
        Uses twist_group's reciprocal vectors (shared by all physical layers in the group),
        but computes phase for each physical layer separately.
        """
        self.phase1 = np.zeros(self.df.shape[0])
        self.phase2 = np.zeros(self.df.shape[0])
        n_groups = len(self.twist_layer)
        for twist_group in range(n_groups):
            # Get reciprocal vectors for this twist_group (shared by all phys_layers in the group)
            b1, b2 = self.monolayer_reciprocal_list[twist_group]
            # Compute phase for all atoms in this twist_group (all physical layers)
            mask = self.df['twist_group'] == twist_group
            self.phase1[mask] = (self.df.loc[mask, 'x'] * b1[0] + self.df.loc[mask, 'y'] * b1[1]) % self.period
            self.phase2[mask] = (self.df.loc[mask, 'x'] * b2[0] + self.df.loc[mask, 'y'] * b2[1]) % self.period
        self.df['phase1'] = self.phase1
        self.df['phase2'] = self.phase2

    def align_coordinates(self):
        """
        Aligns atom coordinates based on their atom_type and reciprocal lattice vectors.

        Adds:
            'delta_tau_x', 'delta_tau_y', 'shifted_x', 'shifted_y' columns to self.df.
        """
        self.df = self.df.copy()
        self.df['delta_tau_x'] = 0.0
        self.df['delta_tau_y'] = 0.0
        self.df['shifted_x'] = self.df['x']
        self.df['shifted_y'] = self.df['y']
        phase1 = self.df['phase1'].values
        phase2 = self.df['phase2'].values

        atom_types = self.df['atom_type'].unique()
        for atom_type in atom_types:
            if atom_type == -1:
                continue
            mask = self.df['atom_type'] == atom_type
            if mask.sum() == 0:
                continue

            # Use twist_group to get reciprocal vectors (shared by all phys_layers in the group)
            twist_group = self.df.loc[mask, 'twist_group'].values[0]
            b1, b2 = self.monolayer_reciprocal_list[twist_group]
            A = np.array([b1, b2])  # Shape: (2, 2)
            try:
                A_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                raise ValueError(
                    f"Reciprocal lattice vectors for layer {layer_index} are linearly dependent and cannot be inverted."
                )

            phase1_avg = np.arctan2(np.mean(np.sin(phase1[mask])), np.mean(np.cos(phase1[mask]))) % self.period
            phase2_avg = np.arctan2(np.mean(np.sin(phase2[mask])), np.mean(np.cos(phase2[mask]))) % self.period

            indices = self.df[mask].index
            phase1_subset = phase1[mask]
            phase2_subset = phase2[mask]

            c1 = (phase1_avg - phase1_subset) % self.period
            c2 = (phase2_avg - phase2_subset) % self.period
            c = np.vstack([c1, c2])  # Shape: (2, n_atoms)

            delta_tau0 = A_inv @ c  # Shape: (2, n_atoms)

            for i, idx in enumerate(indices):
                delta_tau_initial = delta_tau0[:, i]

                min_norm = np.inf
                best_shift = None
                for k1 in range(-self.k_max, self.k_max + 1):
                    for k2 in range(-self.k_max, self.k_max + 1):
                        k = np.array([k1, k2])
                        delta = delta_tau_initial + A_inv @ (self.period * k)
                        norm = np.linalg.norm(delta)
                        if norm < min_norm:
                            min_norm = norm
                            best_shift = delta

                self.df.at[idx, 'delta_tau_x'] = best_shift[0]
                self.df.at[idx, 'delta_tau_y'] = best_shift[1]

                self.df.at[idx, 'shifted_x'] = self.df.at[idx, 'x'] + best_shift[0]
                self.df.at[idx, 'shifted_y'] = self.df.at[idx, 'y'] + best_shift[1]

    def calculate_transformed_matrix(self):
        """
        Calculates the transformed index matrix based on the shifted coordinates.
        """
        sorted_indices = self.df['orb_global_index'].explode().astype(int).tolist()
        sorted_indices = np.array(sorted_indices)
        df_original = self.df.copy()
        df_original = df_original.sort_values(by='original_index')
        original_indices = df_original['orb_global_index'].explode().astype(int).tolist()
        original_indices = np.array(original_indices)

        num_wann = len(sorted_indices)
        if self.spin:
            self.transformed_index_matrix = sp.csr_matrix(
                (
                    np.ones(num_wann * 2),
                    (
                        np.concatenate([original_indices, original_indices + num_wann]),
                        np.concatenate([sorted_indices, sorted_indices + num_wann]),
                    ),
                ),
                shape=(num_wann * 2, num_wann * 2),
            )
        else:
            self.transformed_index_matrix = sp.csr_matrix(
                (np.ones(num_wann), (original_indices, sorted_indices)), shape=(num_wann, num_wann)
            )
        self.transformed_index_matrix = sp.csr_matrix(self.transformed_index_matrix)
        self.reporter.stage("Orbital reordering", "Map OpenMX orbital order to TAPW atom-type order.")
        self.reporter.kv("Transform matrix shape", self.transformed_index_matrix.shape)
        self.reporter.kv("Ordering", "new orb_global_index = transformed_index_matrix * old orb_global_index")
        self.reporter.kv("New order", "atom_type")

    def plot_clusters_loc(self, save_path=None, save=False):
        """
        Plots the clustering results in real space (shifted and original coordinates).
        """
        if plt is None:
            raise ImportError("matplotlib is required for plotting.")

        fig, ax = plt.subplots(1, 2, figsize=(15, 4))
        unique_labels = sorted(self.df['atom_type'].unique())
        colors = plt.get_cmap('tab10', len(unique_labels))

        for _i, label in enumerate(unique_labels):
            mask = self.df['atom_type'] == label
            if label == -1:
                ax[0].scatter(self.df.loc[mask, 'shifted_x'], self.df.loc[mask, 'shifted_y'], color='k', label='Noise', s=5)
                ax[1].scatter(self.df.loc[mask, 'x'], self.df.loc[mask, 'y'], color='k', label='Noise', s=5)
            else:
                species = self.df.loc[mask, 'species'].values[0]
                layer = self.df.loc[mask, 'layer'].values[0]
                sublayer = self.df.loc[mask, 'sublayer'].values[0]
                ax[0].scatter(
                    self.df.loc[mask, 'shifted_x'],
                    self.df.loc[mask, 'shifted_y'],
                    color=colors(label % 10),
                    label=f'Atom Type {label} ({species}) Layer {layer} Sublayer {sublayer}',
                    s=5,
                )
                ax[1].scatter(
                    self.df.loc[mask, 'x'],
                    self.df.loc[mask, 'y'],
                    color=colors(label % 10),
                    label=f'Atom Type {label} ({species}) Layer {layer} Sublayer {sublayer}',
                    s=5,
                )

        for i in range(2):
            ax[i].set_aspect('equal')
            ax[i].set_xlabel('X (Å)')
            ax[i].set_ylabel('Y (Å)')
            ax[i].legend(fontsize='8', loc='upper right')
        ax[0].set_title('Shifted Coordinates')
        ax[1].set_title('Original Coordinates')

        plt.tight_layout()
        if save:
            if save_path is None:
                raise ValueError("Please provide a save path for the plot.")
            plt.savefig(save_path + '/culster_loc.pdf', dpi=300)
        plt.show()

    def plot_clusters_phase(self, save_path=None, save=False):
        """
        Plots the clustering results in phase space.
        """
        if plt is None or matplotlib is None:
            raise ImportError("matplotlib is required for plotting.")

        phase1_deg = (self.df.phase1.values * 180 / np.pi) % 360
        phase2_deg = (self.df.phase2.values * 180 / np.pi) % 360
        period_deg = (self.period * 180 / np.pi) % 360

        # Phase plots should always have 2 subplots (b1 and b2), not based on twist_group count
        fig, ax = plt.subplots(1, 2, figsize=(15, 4))
        unique_labels = sorted(self.df['atom_type'].unique())
        colors = plt.get_cmap('tab10', len(unique_labels))

        kx = np.arange(self.df.shape[0])
        for _i, label in enumerate(unique_labels):
            mask = self.df['atom_type'] == label
            if label == -1:
                ax[0].scatter(kx[mask], phase1_deg[mask], color='k', label='Noise', s=5)
                ax[1].scatter(kx[mask], phase2_deg[mask], color='k', label='Noise', s=5)
            else:
                species = self.df.loc[mask, 'species'].values[0]
                layer = self.df.loc[mask, 'layer'].values[0]
                sublayer = self.df.loc[mask, 'sublayer'].values[0]
                ax[0].scatter(
                    kx[mask],
                    phase1_deg[mask],
                    color=colors(label % 10),
                    label=f'Atom Type {label} ({species}) Layer {layer} Sublayer {sublayer}',
                    s=5,
                )
                ax[1].scatter(
                    kx[mask],
                    phase2_deg[mask],
                    color=colors(label % 10),
                    label=f'Atom Type {label} ({species}) Layer {layer} Sublayer {sublayer}',
                    s=5,
                )

        for i in range(2):
            ax[i].axhline(0, color='black', linestyle='--')
            ax[i].axhline(period_deg, color='black', linestyle='--')
            ax[i].set_xlabel('Atom Index (sorted by z-coordinate)')
            ax[i].set_ylabel('Phase (Degrees)')
            ax[i].set_title(f'Phase Clustering Results b{i+1}')
            ax[i].legend(fontsize='8', loc='upper right')
            ax[i].grid(True)
            ax[i].yaxis.set_major_locator(plt.MultipleLocator(60))
            ax[i].set_ylim(-10, 370)

        plt.tight_layout()
        if save:
            if save_path is None:
                raise ValueError("Please provide a save path for the plot.")
            plt.savefig(save_path + '/culster_phase.pdf', dpi=300)
        plt.show()

    def print_summary(self):
        """
        Prints a summary of the clustering results, including the number of layers, sublayers,
        atom types, and atom counts.
        """
        reporter = getattr(self, "reporter", None)
        reporter = ensure_reporter(reporter)
        reporter.stage("Layer and atom-type summary", "Final atom-type grouping used by the TAPW projector.")
        total_layers = self.df['layer'].nunique()
        reporter.kv("Total layers", total_layers)
        for layer in sorted(self.df['layer'].unique()):
            layer_df = self.df[self.df['layer'] == layer]
            num_sublayers = layer_df['sublayer'].nunique()
            reporter.line(f"  Layer {layer}: {num_sublayers} sublayers")
            for sublayer in sorted(layer_df['sublayer'].unique()):
                sublayer_df = layer_df[layer_df['sublayer'] == sublayer]
                atom_types = sublayer_df['atom_type'].unique()
                reporter.line(f"    Sublayer {sublayer}:")
                for atom_type in sorted(atom_types):
                    count = (sublayer_df['atom_type'] == atom_type).sum()
                    species = sublayer_df.loc[sublayer_df['atom_type'] == atom_type, 'species'].values
                    species = species[0] if len(species) > 0 else "Unknown"
                    label = "Noise" if atom_type == -1 else f"Atom type {atom_type}"
                    reporter.line(f"      {label} ({species}): {count} atoms")

    @staticmethod
    def _get_mapping_to_primitive(dataset):
        # Prefer attribute interface; dict interface is deprecated in newer spglib.
        mapping = getattr(dataset, "mapping_to_primitive", None)
        if mapping is None:
            mapping = getattr(dataset, "std_mapping_to_primitive", None)
        if mapping is None:
            raise KeyError("spglib symmetry dataset missing mapping_to_primitive/std_mapping_to_primitive")
        return np.asarray(mapping, dtype=int)

    @staticmethod
    def _species_to_atomic_numbers(species_list):
        try:
            from ase.data import atomic_numbers
        except Exception as exc:
            raise ImportError("ase is required for spglib-based structure processing.") from exc

        nums = []
        unknown = set()
        for s in species_list:
            sym = str(s).strip()
            Z = atomic_numbers.get(sym, 0)
            if Z == 0:
                unknown.add(sym)
            nums.append(Z)
        if unknown:
            raise ValueError(f"Unknown chemical symbols for spglib/ase mapping: {sorted(unknown)}")
        return np.asarray(nums, dtype=int)

    @staticmethod
    def _raise_for_spglib_stderr(stderr_text: str, context: str) -> None:
        if "Primitive lattice cleaning is incomplete" not in stderr_text:
            return
        raise RuntimeError(
            f"{context}: spglib reported incomplete primitive-lattice cleaning. "
            "This usually means paths.input_file contains relaxed/distorted coordinates, "
            "so spglib cannot recover the primitive basis. Use a rigid/reference OpenMX "
            "input file for paths.input_file. Its atom count, original atom order, and "
            "orbital order must correspond one-to-one to paths.H_file and paths.S_file."
        )

    @classmethod
    def _call_spglib_checked(cls, func, context: str):
        """Run a spglib call while converting known C-stderr failures to hard errors."""
        stderr_fd = 2
        saved_stderr = os.dup(stderr_fd)
        try:
            with tempfile.TemporaryFile(mode="w+b") as stderr_tmp:
                os.dup2(stderr_tmp.fileno(), stderr_fd)
                try:
                    result = func()
                finally:
                    os.dup2(saved_stderr, stderr_fd)
                stderr_tmp.seek(0)
                stderr_text = stderr_tmp.read().decode("utf-8", errors="replace")
        finally:
            os.close(saved_stderr)
        cls._raise_for_spglib_stderr(stderr_text, context)
        if stderr_text:
            os.write(stderr_fd, stderr_text.encode("utf-8", errors="replace"))
        return result

    def _moire_cell_count_from_twist_index(self) -> int | None:
        if self.twist_index is None:
            return None
        m = int(self.twist_index)
        if m < 1:
            return None
        if self.Tmat is not None:
            a1 = np.asarray(self.Tmat[0, :2], dtype=np.float64)
            a2 = np.asarray(self.Tmat[1, :2], dtype=np.float64)
            n1 = float(np.linalg.norm(a1))
            n2 = float(np.linalg.norm(a2))
            if n1 > 0.0 and n2 > 0.0:
                cos_abs = float(np.abs(np.dot(a1, a2)) / (n1 * n2))
                rel_len = float(np.abs(n1 - n2) / n1)
                if cos_abs < 1.0e-3 and rel_len < 1.0e-3:
                    return int((2 * m + 1) ** 2 + 1)
        return int(3 * m ** 2 + 3 * m + 1)

    def _validate_primitive_basis_count(self, twist_group: int, group_df, mapping: np.ndarray) -> None:
        moire_cells = self._moire_cell_count_from_twist_index()
        if moire_cells is None or moire_cells <= 0:
            return
        atom_count = len(group_df)
        if atom_count % moire_cells != 0:
            raise RuntimeError(
                f"Twist group {twist_group}: atom count {atom_count} is not divisible by the "
                f"commensurate moire cell count {moire_cells}. Check paths.input_file and twist_index_m."
            )
        expected_basis = atom_count // moire_cells
        actual_basis = int(np.unique(mapping).size)
        if actual_basis != expected_basis:
            raise RuntimeError(
                f"Twist group {twist_group}: spglib primitive basis assignment produced {actual_basis} "
                f"basis ids, expected {expected_basis} from atom_count={atom_count} and "
                f"moire_cell_count={moire_cells}. This usually means paths.input_file uses relaxed "
                "coordinates rather than a rigid/reference structure. Use a rigid/reference OpenMX "
                "input file whose atom count, original atom order, and orbital order match paths.H_file "
                "and paths.S_file one-to-one."
            )

    def _compute_layer_lattice_and_basis_spglib(self, twist_group: int):
        """
        Compute lattice vectors and basis_id for a twist_group using spglib.
        
        All physical layers in the same twist_group share the same lattice vectors.
        This method aggregates all atoms from all physical layers in the given twist_group.
        
        Args:
            twist_group: Twist group index (0..G-1)
        """
        try:
            import spglib
        except Exception as exc:
            raise ImportError("spglib is required for StructureProcessorSpglib. Install with: pip install spglib") from exc

        if self.Tmat is None:
            raise ValueError("Tmat is required for spglib processing (moiré supercell lattice).")

        # Aggregate all atoms from all physical layers in this twist_group
        group_df = self.df[self.df["twist_group"] == twist_group]
        if group_df.empty:
            raise ValueError(f"Twist group {twist_group}: no atoms found.")

        latS = np.asarray(self.Tmat, dtype=np.float64).copy()
        if latS.shape != (3, 3):
            raise ValueError(f"Tmat must be shape (3,3), got {latS.shape}")

        # Enforce a large out-of-plane lattice constant for quasi-2D materials.
        latS[2, 2] = max(self.spglib_z_lattice, float(latS[2, 2]))

        pos_cart = group_df[["x", "y", "z"]].values.astype(np.float64, copy=False)
        pos_frac = np.linalg.solve(latS.T, pos_cart.T).T  # no wrap on purpose
        numS = self._species_to_atomic_numbers(group_df["species"].values)

        cellS = (latS, pos_frac, numS)
        # Report summary only (not full cellS with all atom positions).
        self.reporter.stage(f"Source orientation group {twist_group} primitive cell", "Spglib primitive-cell summary.")
        self.reporter.kv("Atoms", len(numS))
        self.reporter.kv("Lattice shape", latS.shape)
        self.reporter.kv("Positions shape", pos_frac.shape)
        dataset = self._call_spglib_checked(
            lambda: spglib.get_symmetry_dataset(cellS, symprec=self.symprec),
            f"Twist group {twist_group}: spglib.get_symmetry_dataset",
        )
        if dataset is None:
            raise RuntimeError(
                f"Twist group {twist_group}: spglib.get_symmetry_dataset failed with symprec={self.symprec}. "
                "Use a rigid/reference OpenMX input file for paths.input_file. Its atom count, "
                "original atom order, and orbital order must correspond one-to-one to paths.H_file "
                "and paths.S_file."
            )

        mapping = self._get_mapping_to_primitive(dataset)
        if mapping.shape[0] != numS.shape[0]:
            raise RuntimeError(
                f"Twist group {twist_group}: mapping_to_primitive length mismatch "
                f"({mapping.shape[0]} vs {numS.shape[0]})."
            )
        self._validate_primitive_basis_count(twist_group, group_df, mapping)

        # Sanity check: each basis_id should correspond to a single atomic number.
        for basis_id in np.unique(mapping):
            zs = np.unique(numS[mapping == basis_id])
            if zs.size != 1:
                bad_species = [group_df["species"].values[i] for i in np.where(mapping == basis_id)[0]]
                raise ValueError(
                    f"Twist group {twist_group}: basis_id={basis_id} maps to multiple species {zs.tolist()} "
                    f"({sorted(set(map(str, bad_species)))}) — try smaller symprec."
                )

        # Write back using original indices to avoid accidental reordering bugs.
        idx = group_df.index.to_numpy()
        if "basis_id" not in self.df.columns:
            self.df["basis_id"] = -1
        self.df.loc[idx, "basis_id"] = mapping

        # Primitive lattice for reciprocal vectors.
        prim = self._call_spglib_checked(
            lambda: spglib.standardize_cell(cellS, to_primitive=True, no_idealize=True, symprec=self.symprec),
            f"Twist group {twist_group}: spglib.standardize_cell",
        )
        if prim is None:
            # Fallback tries: progressively relax symprec.
            for symprec_try in (max(self.symprec * 2.0, 5e-2), 1e-1, 5e-1):
                prim = self._call_spglib_checked(
                    lambda symprec_try=symprec_try: spglib.standardize_cell(
                        cellS, to_primitive=True, no_idealize=True, symprec=symprec_try
                    ),
                    f"Twist group {twist_group}: spglib.standardize_cell",
                )
                if prim is not None:
                    break
        if prim is None:
            raise RuntimeError(
                f"Twist group {twist_group}: spglib.standardize_cell(to_primitive=True) failed. "
                "Use a rigid/reference OpenMX input file for paths.input_file. Its atom count, "
                "original atom order, and orbital order must correspond one-to-one to paths.H_file "
                "and paths.S_file."
            )

        latP, _, _ = prim
        a1 = np.asarray(latP[0, :2], dtype=np.float64)
        a2 = np.asarray(latP[1, :2], dtype=np.float64)
        
        # Store lattice vectors for this twist_group (shared by all physical layers in the group)
        self.layer_lattice_vectors[twist_group] = [a1, a2]
        self.reporter.array("Primitive lattice vectors (Angstrom)", np.array([a1, a2]))

        area = float(a1[0] * a2[1] - a1[1] * a2[0])
        if area == 0.0:
            raise ValueError(f"Twist group {twist_group}: lattice vectors are collinear; cannot compute reciprocal vectors.")
        b1 = (2 * np.pi / area) * np.array([a2[1], -a2[0]], dtype=np.float64)
        b2 = (2 * np.pi / area) * np.array([-a1[1], a1[0]], dtype=np.float64)
        
        # Store reciprocal vectors for this twist_group (shared by all physical layers in the group)
        self.monolayer_reciprocal_list[twist_group] = [b1, b2]

        self.reporter.array("Reciprocal vectors (1/Angstrom)", np.array([b1, b2]))
        
        # Also store for each physical layer in this group (for backward compatibility)
        phys_layers_in_group = sorted(group_df['phys_layer'].unique())
        for phys_layer in phys_layers_in_group:
            self.layer_lattice_vectors[phys_layer] = [a1, a2]
            self.monolayer_reciprocal_list[phys_layer] = [b1, b2]

    def _assign_atom_types_from_basis(self):
        if "basis_id" not in self.df.columns:
            raise ValueError("basis_id not found; run spglib basis assignment first.")
        if "sublayer" not in self.df.columns:
            raise ValueError("sublayer not found; run cluster_sublayers() first.")

        key_df = self.df[["layer", "sublayer", "basis_id"]].astype(int)
        atom_type = pd.factorize(pd.MultiIndex.from_frame(key_df))[0]
        self.df = self.df.copy()
        self.df["atom_type"] = atom_type

        # Sanity check: each atom_type should correspond to a single species.
        for at in np.unique(atom_type):
            species = self.df.loc[self.df["atom_type"] == at, "species"].unique()
            if len(species) != 1:
                raise ValueError(f"atom_type {at} corresponds to multiple species: {species}")

        # Match legacy behaviour: sort by phys_layer/sublayer/atom_type (layer=phys_layer for compatibility)
        self.df = self.df.sort_values(by=["phys_layer", "sublayer", "atom_type"]).reset_index(drop=True)

    def process(self):
        """
        Full workflow using spglib for lattice and basis assignment.
        """
        self.separate_layers()
        self.reporter.step("Layer separation", "complete")
        self.cluster_sublayers()
        self.reporter.step("Sublayer clustering", "complete")

        # spglib: per-twist_group lattice + basis_id mapping
        # All physical layers in the same twist_group share the same lattice vectors
        n_groups = len(self.twist_layer)
        for twist_group in range(n_groups):
            self._compute_layer_lattice_and_basis_spglib(twist_group)
        
        # Calculate and print twist angles between twist_groups
        if n_groups >= 2:
            twist_angle_list = []
            for twist_group in range(n_groups):
                a1, a2 = self.layer_lattice_vectors[twist_group]
                # Use a1 direction as the orientation angle
                angle = np.degrees(np.arctan2(a1[1], a1[0]))
                twist_angle_list.append(angle)
            
            # Calculate relative twist angles between adjacent groups
            twist_angle_diffs = np.diff(twist_angle_list)
            # Normalize to [-180, 180] range
            twist_angle_diffs = np.mod(twist_angle_diffs + 180, 360) - 180
            
            self.reporter.stage("Twist-angle check", "Compare primitive-cell orientations between source orientation groups.")
            for i in range(len(twist_angle_diffs)):
                self.reporter.kv(f"Group {i} to group {i+1} twist angle (deg)", f"{twist_angle_diffs[i]:.4f}")

        self._assign_atom_types_from_basis()
        self.reporter.step("Atom-type assignment", "based on layer, sublayer, and primitive-cell basis id")

        self.compute_phase()
        self.reporter.step("Phase coordinates", "computed")
        self.align_coordinates()
        self.reporter.step("Aligned coordinates", "computed")
        self.calculate_transformed_matrix()
        self.reporter.step("Orbital reorder matrix", "computed")
        self.print_summary()

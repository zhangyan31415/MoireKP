# =============================================================================
# >>> SECTION: 00. Module Overview & Public API
# =============================================================================
# >>> SPLIT_HINT: move this section into __init__.py
from __future__ import annotations

"""
Moire k·p continuum model (single-file module).

This file is a structural refactor of `kp/configs/mgi2_G/src/moire.py` into an import-safe,
reproducible single-file module. The numerical core (Y_basis, symmetry operators,
symmetrization, assembly, eigensolve, coefficient extraction) is preserved as-is.

Most common entrypoints:
- `build_model(config) -> ContinuumModel`
- `compute_coefficients(config, model) -> (model, diagnostics)`
- `compute_bands(config, model, kpoints) -> eigvals` (optionally eigvecs)
- `run_end_to_end(config) -> dict`

Minimal usage (pseudo-code):
    cfg = MoireConfig(Q_set1=..., Q_set2=..., n_orb1=..., n_orb2=..., bM1=..., bM2=...,
                     intra_harmonics_map=..., inter_harmonics_map=..., max_order=...,
                     symmetry_map=..., kpoints=...)
    model = build_model(cfg)
    model, diag = compute_coefficients(cfg, model)
    eigvals = compute_bands(cfg, model, cfg.kpoints)
"""

__all__ = [
    "MoireConfig",
    "KPath",
    "build_model",
    "compute_coefficients",
    "compute_bands",
    "run_end_to_end",
    "setup_logging",
    "parse_kpath_in",
    "generate_kpath",
    "generate_kpath_from_vertices",
    "generate_kpath_from_file",
    "generate_kpath_from_symbols",
    "load_Q_sets_from_gvec_files",
    "make_Q_sets_from_gvec_lists",
    "select_kpoints",
    "SymmetryGenerator",
    "ContinuumModelBuilder",
    "ContinuumModel",
    "ContinuumTermKey",
    "ContinuumTerm",
    "KPathGenerator",
    "timing_decorator_factory",
]

# =============================================================================
# >>> SECTION: 01. Imports
# =============================================================================
# >>> SPLIT_HINT: move this section into imports.py

# --- stdlib ---
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

# --- third-party ---
import numpy as np
import scipy
import scipy.linalg
from joblib import Parallel, delayed
import psutil
from scipy import sparse
from tqdm import tqdm

# Plotting is optional; imports are kept local to plotting section when possible.


def _summarize_symmetry_ops(symm: Sequence[Mapping[str, Any]] | Sequence[Any]) -> str:
    names: list[str] = []
    for op in symm:
        if isinstance(op, Mapping):
            name = str(op.get("name", op.get("operation", "?")))
            matrix_kind = op.get("matrix_kind")
            source = op.get("source")
            suffix = []
            if source:
                suffix.append(str(source))
            if matrix_kind:
                suffix.append(str(matrix_kind))
            if suffix:
                name = f"{name}({','.join(suffix)})"
            names.append(name)
        else:
            names.append(str(op))
    return "[" + ", ".join(names) + "]"


def _summarize_coefficients(values: Sequence[Any] | np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.complex128).ravel()
    if arr.size == 0:
        return "count=0"
    abs_arr = np.abs(arr)
    return (
        f"count={arr.size}, nonzero={int(np.count_nonzero(abs_arr > 0.0))}, "
        f"max_abs={float(np.max(abs_arr)):.6g}, median_abs={float(np.median(abs_arr)):.6g}"
    )

# =============================================================================
# >>> SECTION: 02. Constants & Global Toggles
# =============================================================================
# >>> SPLIT_HINT: move this section into constants.py

hartree = 27.2113845
CANONICAL_P_TOL = 1.0e-10

# Symmetrization caches (shared by ContinuumModelBuilder static methods).
# Key structure is internal; safe to clear between runs by calling `clear_symmetry_caches()`.
SYMMETRIZE_GLOBAL_CACHE: dict = {}
SYMMETRIZE_MONOMIAL_OP_CACHE: dict = {}
SYMMETRIZE_MONOMIAL_OP_VALIDATED: set = set()
SYMMETRIZE_COMPOSED_OP_CACHE: dict = {}
SYMMETRIZE_COMPOSED_OP_VALIDATED: set = set()

def clear_symmetry_caches() -> None:
    """Clear module-level symmetrization caches."""
    SYMMETRIZE_GLOBAL_CACHE.clear()
    SYMMETRIZE_MONOMIAL_OP_CACHE.clear()
    SYMMETRIZE_MONOMIAL_OP_VALIDATED.clear()
    SYMMETRIZE_COMPOSED_OP_CACHE.clear()
    SYMMETRIZE_COMPOSED_OP_VALIDATED.clear()

# =============================================================================
# >>> SECTION: 03. Logging
# =============================================================================
# >>> SPLIT_HINT: move this section into logging_utils.py

logger = logging.getLogger(__name__)

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging (optional)."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# =============================================================================
# >>> SECTION: 04. Small Utilities
# =============================================================================
# >>> SPLIT_HINT: move this section into utils.py

def timing_decorator_factory(process_id):
    def timing_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if process_id == 0:
                start_time = time.time()
                process = psutil.Process()
                mem_before = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB

                result = func(*args, **kwargs)

                mem_after = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB
                end_time = time.time()
                duration = end_time - start_time
                mem_peak = mem_after - mem_before

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}] Function '{func.__name__}' executed in {duration:.6f} seconds, Memory peak: {mem_peak:.6f} GB")
                sys.stdout.flush() 
            else:
                result = func(*args, **kwargs)
            return result
        return wrapper
    return timing_decorator

class KPathGenerator:
    def __init__(self, Amat):
        self.Amat = Amat
        self.astar, self.bstar, self.cstar = self.calculate_reciprocal_vectors(Amat)

        self.x_ticks = []
        self.labels_ticks = []
        self.kpoints = []
        self.high_symmetry_points = []
        self.labels = []
        self.segment_points = 0

    @staticmethod
    def _normalize_label(label: str) -> str:
        lab = label.strip()
        if lab.upper() in {"GAMMA", "\\GAMMA", "G", "Γ"} or lab.lower() == "gamma":
            return r"$\Gamma$"
        return label

    @staticmethod
    def calculate_reciprocal_vectors(Amat):
        a, b, c = Amat
        vol = np.dot(a, np.cross(b, c))
        astar = 2 * np.pi * np.cross(b, c) / vol
        bstar = 2 * np.pi * np.cross(c, a) / vol
        cstar = 2 * np.pi * np.cross(a, b) / vol
        return astar, bstar, cstar
    
    def generate_kpath_from_high_symmetry_points(
        self,
        high_symmetry_points: np.ndarray,
        labels: Sequence[str],
        segment_points: int,
        output_file_path: str | Path | None = None,
    ) -> None:
        """
        Generate k-path by interpolating between paired high-symmetry points.

        Notes
        -----
        The input points follow the same convention as VASPKIT/OpenMX KPATH files:
        a sequence of endpoints where every two consecutive points define one segment:
          (P0 -> P1), (P2 -> P3), ...
        """
        self.segment_points = int(segment_points)
        self.high_symmetry_points = np.asarray(high_symmetry_points, dtype=float)
        self.labels = [self._normalize_label(str(lab)) for lab in labels]

        # 生成kpath
        Amat_reciprocal = np.array([self.astar, self.bstar, self.cstar])
        x = 0.0
        num_high_symmetry_points = len(self.high_symmetry_points)
        self.x_ticks = []
        self.labels_ticks = []
        self.kpoints = []

        f = None
        if output_file_path is not None:
            output_file_path = Path(output_file_path)
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            f = output_file_path.open("w", encoding="utf-8")
        try:
            self.x_ticks.append(x)
            self.labels_ticks.append(self.labels[0])
            for i in range(int(num_high_symmetry_points / 2)):
                delta = self.distance(
                    self.direct_cart_real(Amat_reciprocal, self.high_symmetry_points[2 * i + 1]),
                    self.direct_cart_real(Amat_reciprocal, self.high_symmetry_points[2 * i]),
                ) / self.segment_points

                for j in range(self.segment_points):
                    fraction = 1.0 * j / self.segment_points
                    interpolated_point = (1.0 - fraction) * self.high_symmetry_points[2 * i] + fraction * self.high_symmetry_points[2 * i + 1]

                    if f is not None:
                        f.write(f"{interpolated_point[0]:>10.6f} {interpolated_point[1]:>10.6f} {interpolated_point[2]:>10.6f} {x:>10.6f}\n")
                    self.kpoints.append(np.append(interpolated_point, x))
                    x += delta

                if i < int(num_high_symmetry_points / 2) - 1 and self.labels[2 * i + 2] != self.labels[2 * i + 1]:
                    if f is not None:
                        f.write(
                            f"{self.high_symmetry_points[2 * i + 1][0]:>10.6f} {self.high_symmetry_points[2 * i + 1][1]:>10.6f} {self.high_symmetry_points[2 * i + 1][2]:>10.6f} {x:>10.6f}\n"
                        )
                    self.kpoints.append(np.append(self.high_symmetry_points[2 * i + 1], x))

                if i < int(num_high_symmetry_points / 2) - 1:
                    if self.labels[2 * i + 2] == self.labels[2 * i + 1]:
                        self.x_ticks.append(x)
                        self.labels_ticks.append(self.labels[2 * i + 1])
                    else:
                        self.x_ticks.append(x)
                        self.labels_ticks.append(f"{self.labels[2 * i + 1]}|{self.labels[2 * i + 2]}")

            self.x_ticks.append(x)
            self.labels_ticks.append(self.labels[-1])
            if f is not None:
                f.write(
                    f"{self.high_symmetry_points[2 * i + 1][0]:>10.6f} {self.high_symmetry_points[2 * i + 1][1]:>10.6f} {self.high_symmetry_points[2 * i + 1][2]:>10.6f} {x:>10.6f}\n"
                )
            self.kpoints.append(np.append(self.high_symmetry_points[2 * i + 1], x))
            self.kpoints = np.array(self.kpoints)
        finally:
            if f is not None:
                f.close()

    def read_and_generate_kpath(self, file_path: str | Path, output_file_path: str | Path | None = None) -> None:
        """Read a VASPKIT/OpenMX-style KPATH file and generate kpoints (optionally writing an output file)."""
        file_path = Path(file_path)
        lines = file_path.read_text(encoding="utf-8").splitlines()

        self.segment_points = int(lines[1])
        high_symmetry_points: list[np.ndarray] = []
        labels: list[str] = []

        for i in range(4, len(lines)):
            parts = lines[i].split()
            coordinates = np.array([float(coord) for coord in parts[:3]])
            if len(coordinates) == 3:
                label = parts[-1] if parts else ""
                high_symmetry_points.append(coordinates)
                labels.append(label)

        self.generate_kpath_from_high_symmetry_points(
            np.array(high_symmetry_points),
            labels,
            self.segment_points,
            output_file_path=output_file_path,
        )

    @staticmethod
    def distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    @staticmethod
    def cart_direct_real(Amat, pos_cart):
        return np.dot(np.linalg.inv(Amat.T), np.array(pos_cart))

    @staticmethod
    def direct_cart_real(Amat, pos_direct):
        return np.dot(Amat.T, np.array(pos_direct))

    @staticmethod
    def generate_chern_kmesh(num):
        num_k = num ** 2
        kpoints = np.zeros((num_k, 3))
        for i in range(num):
            for j in range(num):
                kpoints[i * num + j] = np.array([i / (num - 1), j / (num - 1), 0])
        return kpoints

    @staticmethod
    def generate_foundamental_kmesh(num, a1, a2):
        num_k = num ** 2
        kpoints = np.zeros((num_k, 3))
        for i in range(num):
            for j in range(num):
                kpoints[i * num + j] = i / (num - 1) * a1 + j / (num - 1) * a2
        return kpoints

    @staticmethod
    def generate_foundamental_kmesh_kpBC(delta_a1, delta_a2, num, BM):
        # num = int(np.linalg.norm(a2)/delta+1e-3)+1
        num_k = num ** 3
        kpoints = np.zeros((num_k, 3))
        for i in range(num):
            for j in range(num):
                kpoints[i * num + j] = (i-int(num/2))*delta_a1 + (j-int(num/2))*delta_a2
        # kpoints = kpoints@np.linalg.inv(BM)
        return kpoints

def rot(vec,theta):
    theta = theta/180*np.pi
    rot_mat = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return np.dot(rot_mat,vec)

def generate_orb(classname,l1,l2, max_M_sum, max_p_order,orb_list,intra_harmonics_map, symm=[{"name":"TR"}]):
    """
    classname: same_spin_diag, same_spin_offdiag, diff_spin_diag, diff_spin_offdiag
    max_M_sum: 最大 M_sum
    max_p_order: 最大 p_order
    """
    key_list = []
    if classname == "same_spin_diag":
        for a in range(1,orb_list[l1-1]+1):
            for b in range(1,orb_list[l2-1]+1):
                if (a-b)%2 != 0:
                    continue
                for M_sum in range(0, max_M_sum+1):
                    for Mz in range(0, M_sum+1):
                        Mz_star = M_sum - Mz
                        for p_order in range(1, 2):
                            p = intra_harmonics_map[p_order]
                            key = ContinuumTermKey(Mz, Mz_star, l1, l2, a, b, tuple(p))
                            key_list.append(key)
    elif classname == "same_spin_offdiag":
        for a in range(1,orb_list[l1-1]+1):
            for b in range(1,orb_list[l2-1]+1):
                if (a-b)%2 != 0:
                    continue
                for M_sum in range(0, max_M_sum+1):
                    for Mz in range(0, M_sum+1):
                        Mz_star = M_sum - Mz
                        for p_order in range(2, max_p_order+1):
                            p = intra_harmonics_map[p_order]
                            key = ContinuumTermKey(Mz, Mz_star, l1, l2, a, b, tuple(p))
                            key_list.append(key)
    elif classname == "diff_spin_diag":
        for a in range(1,orb_list[l1-1]+1):
            for b in range(1,orb_list[l2-1]+1):
                if (a-b)%2 == 0:
                    continue
                for M_sum in range(0, max_M_sum+1):
                    for Mz in range(0, M_sum+1):
                        Mz_star = M_sum - Mz
                        for p_order in range(1, 2):
                            p = intra_harmonics_map[p_order]
                            key = ContinuumTermKey(Mz, Mz_star, l1, l2, a, b, tuple(p))
                            key_list.append(key)
    elif classname == "diff_spin_offdiag":
        for a in range(1,orb_list[l1-1]+1):
            for b in range(1,orb_list[l2-1]+1):
                if (a-b)%2 == 0:
                    continue
                for M_sum in range(0, max_M_sum+1):
                    for Mz in range(0, M_sum+1):
                        Mz_star = M_sum - Mz
                        for p_order in range(2, max_p_order+1):
                            p = intra_harmonics_map[p_order]
                            key = ContinuumTermKey(Mz, Mz_star, l1, l2, a, b, tuple(p))
                            key_list.append(key)
    elif classname == "Kinect":
        for a in range(1,orb_list[l1-1]+1):
            b = a
            for M_sum in range(0, max_M_sum+1):
                for Mz in range(0, M_sum+1):
                    Mz_star = M_sum - Mz
                    if Mz + Mz_star == 0 or (Mz - Mz_star) % 3 != 0 or Mz_star>Mz:
                        continue
                    for p_order in range(1, 2):
                        p = intra_harmonics_map[p_order]
                        key = ContinuumTermKey(Mz, Mz_star, l1, l2, a, b, tuple(p))
                        key_list.append(key)
    elif classname == "Onsite":
        for a in range(1,orb_list[l1-1]+1):
            b = a
            key = ContinuumTermKey(0, 0, l1, l2, a, b, (0.0, 0.0))
            key_list.append(key)
    else:
        raise ValueError("Invalid classname. Choose from 'same_spin_diag', 'same_spin_offdiag', 'diff_spin_diag', 'diff_spin_offdiag'.")
    
    #根据对称性删除一些key
    def get_opposite_spin(a):
        return 2-(a-1)%2+(a-1)//2*2
    for symm_op in symm:
        if symm_op["name"] == "TR":

            key_list_new = []
            
            spin_up_dn_list = []
            spin_up_dn_list_diag = []
            spin_up_dn_list_offdiag = []
            
            for a in range(1,orb_list[l1-1]+1):
                for b in range(1,orb_list[l2-1]+1):
                    if (a-b)%2 == 1:
                        spin_up_dn_list.append((a,b))
            for a,b in spin_up_dn_list:
                a_, b_ = get_opposite_spin(a), get_opposite_spin(b)
                if a_ > a or a < b:
                    spin_up_dn_list_diag.append((a,b))
                if a < b-1 or (b%2 !=0 and b-a ==1):
                    spin_up_dn_list_offdiag.append((a,b))
            
            for key in key_list:
                a, b, p = key.orbital_from, key.orbital_to, key.p
                if a%2 == 0 and b%2 == 0: #spin down
                    continue
                elif a%2 == 1 and b%2 == 1: #spin up
                    if a <= b and (np.sum(np.abs(np.array(p))) < 1e-5 and classname not in  ["Kinect", "Onsite"]):
                        continue
                elif (a-b)%2 == 1: #spin up - spin down
                    if (a,b) in spin_up_dn_list_diag and np.sum(np.abs(np.array(p))) < 1e-5:
                        continue
                    if (a,b) in spin_up_dn_list_offdiag and np.sum(np.abs(np.array(p))) > 1e-5:
                        continue
                # print(f"a = {a}, b = {b}, p = {p}")
                key_list_new.append(key)
            return key_list_new
    
    return key_list

def make_hashable(obj: Any) -> Any:
    """
    将对象转换为可哈希的版本。
    - 对于 list 和 tuple，递归将每个元素转换为 tuple。
    - 对于 dict，将其转换为 sorted 的 (key, value) tuple。
    - 对于 numpy 数组，使用 .tobytes()（也可以用 tuple(obj.tolist())）
    - 其它类型保持不变。
    """
    if isinstance(obj, (list, tuple)):
        return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, np.ndarray):
        # 使用 tobytes() 保证数组的位表示是唯一的  
        return obj.tobytes()
    elif isinstance(obj, ContinuumTerm):
        # 利用 term 的 key，该 key 需是可哈希的（比如 @dataclass(frozen=True)）
        return make_hashable(obj.key)
    # elif isinstance(obj, int):
    #     return obj
    # elif isinstance(obj, float):
    #     return obj
    # #如果是None，直接返回
    # elif obj is None:
    #     return 0
    # elif isinstance(obj, str):
    #     return obj
    # elif isinstance(obj, complex):
    #     return (obj.real, obj.imag)
    # elif hasattr(obj, '__hash__'):
    #     # 如果对象有 __hash__ 方法，则直接返回其哈希值
    #     return hash(obj)
    else:
        return obj
        raise TypeError(f"Unsupported type for hashing: {type(obj)}")

def get_function_hash(fn: Callable) -> int:
    try:
        closure = fn.__closure__
        if closure is not None:
            closure_values = tuple(make_hashable(c.cell_contents) for c in closure)
        else:
            closure_values = None
        defaults = make_hashable(fn.__defaults__)
        return hash((fn.__code__.co_code, defaults, closure_values))
    except AttributeError:
        # 如果函数没有 __code__ 属性，则退化处理
        return id(fn)

def truncate_indices(Q_set1: np.ndarray, Q_set2: np.ndarray, n_orb1: int, n_orb2: int, cutoff_shells: int, spin: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    按“圈数”对平面波基做截断。

    参数
    ----
    Q_sets : list of (n_Q_i, 2) ndarray  
        每层的 Q 向量列表，Q_sets[i].shape == (n_Q_i, 2)  
    orbitals_per_layer : list of int  
        每层的轨道数 len(orbitals_per_layer)==len(Q_sets)  
    cutoff_shells : int  
        截断的圈数（包含中心的第 0 圈），例如 cutoff_shells=3 表示保留半径最小的 4 个壳：第 0 圈 + 前 3 圈  
    spin : bool  
        是否包含自旋。如果 True，则每个轨道先放完所有 up 的 Q，再放所有 down 的 Q  

    返回
    ----
    keep_idx, remove_idx : ndarray of int  
        分别为需要保留和平面波截断后丢弃的全局基函数下标  
    """
    keep = []
    remove = []
    offset = 0
    Q_sets = [Q_set1, Q_set2]
    orbitals_per_layer = [n_orb1, n_orb2]

    for Q, n_orb in zip(Q_sets, orbitals_per_layer):
        # 计算这一层各 Q 的径向距离，找到“壳”半径
        r = np.linalg.norm(Q, axis=1)
        shells = np.sort(np.unique(np.round(r, 6)))
        # 阈值半径：第 cutoff_shells 个半径（如果索引越界，就取最大壳）
        if cutoff_shells < len(shells):
            r_thresh = shells[cutoff_shells]
        else:
            r_thresh = shells[-1]
        mask = r <= r_thresh+1e-5  # True 表示保留

        n_Q = Q.shape[0]
        for orb in range(n_orb):
            if spin:
                # up 态
                for iq in range(n_Q):
                    idx = offset + iq
                    (keep if mask[iq] else remove).append(idx)
                offset += n_Q
                # down 态
                for iq in range(n_Q):
                    idx = offset + iq
                    (keep if mask[iq] else remove).append(idx)
                offset += n_Q
            else:
                # 无自旋
                for iq in range(n_Q):
                    idx = offset + iq
                    (keep if mask[iq] else remove).append(idx)
                offset += n_Q

    return np.array(keep, dtype=int), np.array(remove, dtype=int)

def truncate_Q_indices(Q_set1: np.ndarray, cutoff_shells: int) -> tuple[np.ndarray, np.ndarray]:
    """
    按“圈数”对平面波基做截断。

    参数
    ----
    Q_set1 : (n_Q1, 2) ndarray
        第一层的 Q 向量列表，Q_set1.shape == (n_Q1, 2)
    cutoff_shells : int  
        截断的圈数（包含中心的第 0 圈），例如 cutoff_shells=3 表示保留半径最小的 4 个壳：第 0 圈 + 前 3 圈  

    返回
    ----
    keep1_indices, remove1_indices : ndarray of int  
        第一层需要保留和平面波截断后丢弃的全局基函数下标
    """
    keep1 = []
    remove1 = []
    
    # 计算第一层各 Q 的径向距离，找到“壳”半径
    r1 = np.linalg.norm(Q_set1, axis=1)
    shells1 = np.sort(np.unique(np.round(r1, 6)))
    # 阈值半径：第 cutoff_shells 个半径（如果索引越界，就取最大壳）
    if cutoff_shells < len(shells1):
        r_thresh1 = shells1[cutoff_shells]
    else:
        r_thresh1 = shells1[-1]
    mask1 = r1 <= r_thresh1+1e-5  # True 表示保留


    n_Q1 = Q_set1.shape[0]

    for iq in range(n_Q1):
        idx = iq
        (keep1 if mask1[iq] else remove1).append(idx)



    return np.array(keep1, dtype=int), np.array(remove1, dtype=int)

# =============================================================================
# >>> SECTION: 05. Data Structures
# =============================================================================
# >>> SPLIT_HINT: move this section into types.py

@dataclass(frozen=True)
class ContinuumTermKey:
    """
    唯一标识连续模型中一项 term 的指标

    属性：
      Mz, Mz_star: 多项式阶数（满足 Mz+Mz_star <= max_order）
      layer_from, layer_to: 层号（1 或 2）
      orbital_from, orbital_to: 轨道编号（从1开始）
      p: 跃迁动量，以 tuple 表示，如 (px, py)
    """
    Mz: int
    Mz_star: int
    layer_from: int
    layer_to: int
    orbital_from: int
    orbital_to: int
    p: Tuple[float, float]
    
    def __post_init__(self):
        # 但 dataclass(frozen=True) 下不能直接赋值；可以用 object.__setattr__
        p0 = round(float(self.p[0]) / CANONICAL_P_TOL) * CANONICAL_P_TOL
        p1 = round(float(self.p[1]) / CANONICAL_P_TOL) * CANONICAL_P_TOL
        object.__setattr__(self, 'p', (p0, p1))

@dataclass
class ContinuumTerm:
    """
    表示连续模型中的一项 term

    属性：
      key: 唯一标识 term 的指标
      Y_basis: 一个函数，输入 k 返回基函数矩阵（维度由各层 Q 数和轨道数决定）
      r_value_real, r_value_imag: 分别对应原基和 i×原基的系数（待求解）
      active: 正交化后标记该项是否为线性无关
      tag: "Kinect", "intra", "inter"，用于区分 onsite（Kinect且Mz=Mz_star=0）与耦合项
      symmetry_ops: 对称操作列表，每个元素为字典，如 {"name": "C3z", "params": 1}
    """
    key: ContinuumTermKey
    Y_basis: Callable[[np.ndarray], np.ndarray]
    r_value_real: Any = 0
    r_value_imag: Any = 0
    active: bool = False
    tag: str = "intra"
    symmetry_ops: List[Dict[str, Any]] = field(default_factory=list)
    registry_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MoireConfig:
    """
    Configuration container for building/fitting/solving the moire continuum model.

    This object intentionally keeps I/O (paths) and in-memory arrays in one place to
    make experiments reproducible.

    Key array shapes:
      - Q_set1: (N1, 2)
      - Q_set2: (N2, 2)
      - kpoints: (Nk, 2)
      - heff: (Nk_fit*dim, Nk_fit*dim) when constructed as block-diagonal
    """

    # --- inputs (in-memory) ---
    Q_set1: np.ndarray | None = None
    Q_set2: np.ndarray | None = None
    n_orb1: int = 2
    n_orb2: int = 2
    nlow_state: List[int] | None = None

    bM1: np.ndarray | None = None
    bM2: np.ndarray | None = None
    intra_harmonics_map: Dict[int, np.ndarray] = field(default_factory=dict)
    inter_harmonics_map: Dict[int, np.ndarray] = field(default_factory=dict)
    max_order: Dict[str, int] = field(default_factory=lambda: {"Kinect": 10, "intra": 4, "inter": 4})
    symmetry_map: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    symmetry_gen: Any | None = None
    symmetry_source_metadata: Dict[str, Any] = field(default_factory=dict)
    sectors: List[Dict[str, Any]] = field(default_factory=list)
    term_templates: List[Dict[str, Any]] = field(default_factory=list)
    bM_diagnostics: Dict[str, Any] = field(default_factory=dict)

    # --- k sampling / fitting ---
    kpoints: np.ndarray | None = None
    kpoints_fit: np.ndarray | None = None
    heff: np.ndarray | None = None
    coeff_tol: float = 1e-6

    # --- band reduction (optional Schur complement) ---
    keep_indices: np.ndarray | None = None
    remove_indices: np.ndarray | None = None

    # --- toggles ---
    use_cache: bool = True
    profile_light: bool = False
    eigvals_only: bool = True
    n_jobs: int = 1

    # --- outputs ---
    output_dir: str | Path | None = None
    save_hamiltonians: bool = False
    save_eigvecs: bool = False
    log_level: int = logging.INFO

    # --- optional: file-based inputs / provenance (not required by core numerics) ---
    Tmat: np.ndarray | None = None
    phase_deg: float = 0.0
    kpath_file: str | Path | None = None
    kpath_out_file: str | Path | None = None
    kpath_segment_points: int | None = None
    gvec_file_layer1: str | Path | None = None
    gvec_file_layer2: str | Path | None = None
    Q_rotation_deg: float | None = None


@dataclass(frozen=True)
class KPath:
    """
    Container for a k-path produced from high-symmetry points.

    Attributes
    ----------
    kpoints_frac:
        Shape (Nk, 3). Fractional coordinates in reciprocal basis as provided by the input KPATH.
    x:
        Shape (Nk,). Cumulative distance (in cartesian reciprocal space units) used for plotting.
    kpoints_2d:
        Shape (Nk, 2). 2D k-points after applying reciprocal-lattice transform + in-plane rotation.
    x_ticks / labels_ticks:
        Tick positions/labels for plotting along the path.
    """

    kpoints_frac: np.ndarray
    x: np.ndarray
    kpoints_2d: np.ndarray
    x_ticks: List[float]
    labels_ticks: List[str]

# =============================================================================
# >>> SECTION: 06. Symmetry Layer
# =============================================================================
# >>> SPLIT_HINT: move this section into symmetry.py

class SymmetryGenerator:
    """
    根据输入的 Q 数据生成对称操作矩阵，其维度与基函数矩阵一致。
    """

    def __init__(self, Qlayer1: np.ndarray, Qlayer2: np.ndarray, nlow_state: List[int], basis_template: str | None = None):
        self.Qlayer1 = Qlayer1
        self.Qlayer2 = Qlayer2
        self.nlow_state = nlow_state
        self.basis_template = basis_template
        self.Q_set = np.concatenate([Qlayer1, Qlayer2], axis=0)

        # 预计算所有操作矩阵并缓存
        self.cached_operators = {}
        # self._cache_all_operators()

    def _cache_all_operators(self):
        """
        预计算并缓存所有的对称操作矩阵。
        这里考虑 C3z 有参数，因此我们缓存一个字典存储不同参数的 C3z 操作矩阵。
        """
        # 预缓存C3z矩阵（假设params为0，1，2）
        for params in range(3):
            self.cached_operators[f'C3z_{params}'] = self.get_C3z_operator(params)
        
        self.cached_operators['C2T'] = self.get_C2T_operator()

    def rotation_matrix(self, theta: float) -> np.ndarray:
        """生成二维旋转矩阵"""
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])

    def get_C3z_operator(self, params: int) -> np.ndarray:
        """
        生成 C3z 对称操作的投影矩阵（示例代码，维度与各层 Q 数和轨道数匹配）
        这里利用输入的 Qlayer 与 nlow_state 生成 block_diag 矩阵
        """
        if self.basis_template not in {"Gamma_four_orbital", "K_notebook"}:
            raise ValueError("C3z toy generator requires Gamma_four_orbital or K_notebook basis_template; use kp_symm_output for production.")
        gamma_template_phases = [np.exp(1j * np.pi / 3)]
        q1norm = np.max(np.linalg.norm(self.Q_set, axis=1)) - np.min(np.linalg.norm(self.Q_set, axis=1))
        C3_matrix = []
        for i in range(2):
            Qlayer = self.Qlayer1 if i == 0 else self.Qlayer2
            num_low_orb = self.nlow_state[i]
            for j in range(num_low_orb):
                value = gamma_template_phases[j % len(gamma_template_phases)]
                matrix = np.array([
                    [value if np.linalg.norm(ii - self.rotation_matrix(np.deg2rad(120)) @ jj) < q1norm/60 else 0
                     for jj in Qlayer]
                    for ii in Qlayer
                ])
                C3_matrix.append(matrix)
        C3_proj_matrix = scipy.linalg.block_diag(*C3_matrix)
        # C3_temp = np.load("/data/work/zy/software/TAPW_tmdc/dft_relax_from_mlff/3.48_same/2soc/Q_shell_7/band_data/C3_matrix.npy")
        # if params == 2:
        #     C3_temp = C3_temp @ C3_temp
        # if params == 0:
        #     C3_temp = np.eye(C3_temp.shape[0], dtype=complex)
        # if np.sum(np.abs(C3_proj_matrix@C3_proj_matrix - C3_proj_matrix.T)) > 1e-8:
        # if np.sum(np.abs(C3_proj_matrix - C3_temp.T)) > 1e-8:
        #     raise ValueError("C3z operator not orthogonal.")
        if params == 2:
            C3_proj_matrix = C3_proj_matrix @ C3_proj_matrix
        if params == -1:
            C3_proj_matrix = np.linalg.inv(C3_proj_matrix)
        if params == -2:
            C3_proj_matrix = np.linalg.inv(C3_proj_matrix) @ np.linalg.inv(C3_proj_matrix)
        return C3_proj_matrix

    def get_time_reversal_matrix(self) -> np.ndarray:
        """
        构造仅含自旋部分的时间反演酉算符矩阵 (i * sigma_y)。
        
        假设每个层对轨道与自旋的排列顺序是：
          - 对第 n 个轨道：
            先遍历该轨道的 “down” 自旋在所有 Q 的分量，
            再遍历该轨道的 “up” 自旋在所有 Q 的分量，
          - 然后再切换到 (n+1)-th 轨道，重复上述 down->up。

        记法：对第 i 层，
          nQ = len(Qlayer_i),
          num_low_orb = self.nlow_state[i]
          注意：这里的 nlow_state 在本脚本中约定为“包含自旋的低能态数”，
               即 num_low_orb = 2 * (physical_orbital_count)。
          则该层总维度为 dim_layer = nQ * num_low_orb。

        注意：这里只构造 spin-space 上的 i*sigma_y 块，
             未包含复共轭 K，也未做 Q->-Q 的交换。

        Returns
        -------
        T_proj_matrix : np.ndarray
            时间反演(自旋部分)在整个多层空间的投影矩阵（分块对角拼接）。
        """

        if any(int(n) % 2 for n in self.nlow_state):
            raise ValueError(
                "TR toy generator requires explicit spin/Kramers pair basis or a kp_symm_output representation; "
                "for spinless effective TR use operation name TR_eff with explicit matrix convention."
            )

        # 先写好 i*sigma_y 在基 (down, up) 下的 2x2 矩阵：
        #   i*sigma_y = [[0, -1],
        #                [1,  0]]
        # 表示下->-上, 上->下
        spin_block = np.array([
            [0, -1],
            [1,  0]
        ], dtype=complex)

        # 存储每一层的矩阵块
        T_blocks = []

        # 两层循环（如需更多层可在此扩展）
        for i in range(2):
            if i == 0:
                spin_block = np.array([
                    [0, -1],
                    [1,  0]
                ], dtype=complex)
            else:
                spin_block = -np.array([
                    [0, 1],
                    [-1,  0]
                ], dtype=complex)
            Qlayer = self.Qlayer1 if i == 0 else self.Qlayer2
            nQ = len(Qlayer)
            num_low_orb = self.nlow_state[i]

            # 该层总维度 = (包含自旋的) num_low_orb * nQ
            dim_layer = int(num_low_orb) * nQ
            T_matrix_layer = np.zeros((dim_layer, dim_layer), dtype=complex)

            # 定义一个 index 函数，用来返回 (orb, spin, q) 在矩阵里的行列号
            # 这里 spin=0 表示 down, spin=1 表示 up
            # “先遍历该轨道 down 在所有Q, 再该轨道 up 在所有Q”，然后下个轨道
            def idx(orb, spin, q):
                # 每个轨道有 2*nQ 维度 (down block + up block)
                # orb_offset = orb*(2*nQ)
                # spin_offset = spin*(nQ)
                # return orb_offset + spin_offset + q
                return orb*(2*nQ) + spin*nQ + q

            # 在该层内部构造时间反演(自旋部分)的耦合
            for orb_i in range(int(num_low_orb) // 2):
                for q_i in range(nQ):
                    for q_j in range(nQ):
                        if np.sum(np.abs(Qlayer[q_i] + Qlayer[q_j])) < 1e-8:
                            # (down, up) => 用 spin_block 表示
                            row_down = idx(orb_i, 0, q_i)  # down
                            col_down = row_down
                            row_up = idx(orb_i, 1, q_j)    # up
                            col_up = row_up

                            # 根据 2x2 子矩阵 spin_block 填充：
                            # spin_block[0,0]  ->  (down, down)
                            # spin_block[0,1]  ->  (down, up)
                            # spin_block[1,0]  ->  (up, down)
                            # spin_block[1,1]  ->  (up, up)
                            T_matrix_layer[row_down, col_down] = spin_block[0, 0]
                            T_matrix_layer[row_down, col_up]   = spin_block[0, 1]
                            T_matrix_layer[row_up,   col_down] = spin_block[1, 0]
                            T_matrix_layer[row_up,   col_up]   = spin_block[1, 1]

            T_blocks.append(T_matrix_layer)

        # 分块对角拼接两层
        T_proj_matrix = scipy.linalg.block_diag(*T_blocks)

        return T_proj_matrix

    def get_time_reversal_matrix_effective(self) -> np.ndarray:
        """
        Spinless effective time-reversal sewing matrix D for the antiunitary operator D K.

        This matches Q -> -Q within each layer and keeps orbital labels fixed.
        """
        T_blocks = []
        for i in range(2):
            Qlayer = self.Qlayer1 if i == 0 else self.Qlayer2
            nQ = len(Qlayer)
            num_low_orb = int(self.nlow_state[i])
            dim_layer = num_low_orb * nQ
            T_matrix_layer = np.zeros((dim_layer, dim_layer), dtype=complex)

            def idx(orb, q):
                return orb * nQ + q

            for orb_i in range(num_low_orb):
                for q_i in range(nQ):
                    matched = False
                    for q_j in range(nQ):
                        if np.sum(np.abs(Qlayer[q_i] + Qlayer[q_j])) < 1e-8:
                            T_matrix_layer[idx(orb_i, q_i), idx(orb_i, q_j)] = 1.0
                            matched = True
                    if not matched:
                        raise ValueError("TR_eff toy generator requires Q -> -Q matching within each layer")
            T_blocks.append(T_matrix_layer)

        return scipy.linalg.block_diag(*T_blocks)
    
    def get_C2T_operator(self) -> np.ndarray:
        """
        Build the single-valley antiunitary twofold action matrix.
        """
        Qset = self.Q_set
        q1norm = np.min(np.linalg.norm(Qset, axis=1))
        mat = np.zeros((len(Qset), len(Qset)), dtype=complex)
        if self.nlow_state[0] != self.nlow_state[1] or len(self.Qlayer1) != len(self.Qlayer2) or self.nlow_state[0] != 1:
            raise ValueError("Different number of low energy states or Q points or not 1 low energy state per layer. Not supported C2T.")
        
        for i in range(2):
            Qlayer_i = self.Qlayer1 if i == 0 else self.Qlayer2
            for j in range(2):
                Qlayer_j = self.Qlayer1 if j == 0 else self.Qlayer2
                R_y = np.array([[1,0],[0,-1]])
                for ii in range(len(Qlayer_i)):
                    for jj in range(len(Qlayer_j)):
                        if np.linalg.norm(Qlayer_i[ii] - R_y @ Qlayer_j[jj]) < q1norm/10:
                            mat[ii + i*len(Qlayer_i), jj + j*len(Qlayer_j)] = 1
        return mat.T

    def get_C2_operator(self, qtol: float | None = None) -> np.ndarray:
        """
        Construct a twofold layer-exchange unitary matrix.

        The in-plane action is supplied as metadata by configured models; this
        fallback generator keeps the historical layer/orbital matrix template.

        基底顺序假定与 get_C3z_operator 一致：
        [ layer1: orb0(Qs), orb1(Qs), ..., layer2: orb0(Qs), orb1(Qs), ... ]，
        其中每个 “orbj(Qs)” 是一个大小 nQ 的子块（先固定轨道，再遍历该轨道的全部 Q）。

        参数
        ----
        qtol : float | None
            Q 匹配容差；默认使用 (max|Q|-min|Q|)/60（与 C3z 实现保持一致）。

        返回
        ----
        U_C2 : np.ndarray (complex)
            The two-sector representation matrix.
        """
        import numpy as np
        import scipy.linalg

        # ---- 基本量与容差 ----
        Q1 = np.asarray(self.Qlayer1, dtype=float)
        Q2 = np.asarray(self.Qlayer2, dtype=float)
        nQ1, nQ2 = len(Q1), len(Q2)
        m1, m2 = int(self.nlow_state[0]), int(self.nlow_state[1])

        if m1 != m2:
            raise ValueError(f"两层的 num_low_orb 不一致: {m1} vs {m2}")
        if nQ1 != nQ2:
            raise ValueError(f"两层的 Q 数不一致: {nQ1} vs {nQ2}")

        m, nQ = m1, nQ1
        if self.basis_template not in {"Gamma_four_orbital", "M_spinless_layer_exchange"}:
            raise ValueError("C2 toy generator requires Gamma_four_orbital or M_spinless_layer_exchange basis_template")

        if qtol is None:
            qset = getattr(self, "Q_set", None)
            if qset is None:
                qnorms = np.linalg.norm(np.vstack([Q1, Q2]), axis=1) if (nQ1 + nQ2) else np.array([0.0])
            else:
                qnorms = np.linalg.norm(np.asarray(qset, dtype=float), axis=1)
            qtol = (np.max(qnorms) - np.min(qnorms)) / 60.0 if len(qnorms) else 1e-12

        c2_inplane = np.array([[1.0, 0.0],
                               [0.0, -1.0]], dtype=float)

        # ---- 构造层间 Q 的置换矩阵：P12 把 layer2 的 Q 旋到 layer1 ----
        def build_perm(Q_src, Q_tgt, A, tol):
            P = np.zeros((len(Q_src), len(Q_tgt)), dtype=complex)
            for i, qi in enumerate(Q_src):
                Aq = (A @ Q_tgt.T).T  # 所有目标一次性变换
                d = np.linalg.norm(qi - Aq, axis=1)
                # 小于阈值则认为匹配；允许多对一时取最接近者
                if np.any(d < tol):
                    j = int(np.argmin(d))
                    P[i, j] = 1.0
            return P

        P12 = build_perm(Q1, Q2, c2_inplane, qtol)  # map layer2 to layer1

        if self.basis_template == "M_spinless_layer_exchange":
            if m != 1:
                raise ValueError("M_spinless_layer_exchange C2 toy template requires one low-energy orbital per layer")
            M12 = P12
            M21 = P12.conj().T
            Z = np.zeros((nQ, nQ), dtype=complex)
            return np.block([[Z, M12], [M21, Z]])

        # ---- 轨道内部的 σ_x 交换（按相邻成对：0↔1, 2↔3, ...）----
        if m % 2 != 0:
            raise ValueError(f"期望每层的轨道数为偶数（成对交换），当前 m={m}")
        OrbX = np.zeros((m, m), dtype=complex)
        for p in range(0, m, 2):
            OrbX[p, p+1] = -1.0
            OrbX[p+1, p] = 1.0

        # 层间映射块：M12 = OrbX ⊗ P12
        M12 = np.kron(OrbX, P12)      # shape: (m*nQ, m*nQ)
        M21 = -M12.conj().T            # 保证酉性；令 U = [[0, M12],[M21, 0]]

        Z = np.zeros((m*nQ, m*nQ), dtype=complex)
        # top = np.hstack([Z,   M12])
        # bot = np.hstack([M21, Z  ])
        U_C2 = np.block([[Z, M12],
                         [M21, Z]])

        # （可选）数值自检：U^†U≈I, U^2≈I
        # I_full = np.eye(U_C2.shape[0], dtype=complex)
        # assert np.allclose(U_C2.conj().T @ U_C2, I_full, atol=1e-10)
        # assert np.allclose(U_C2 @ U_C2, I_full, atol=1e-10)

        return U_C2


    def get_operator(self, name: str, params: Any) -> np.ndarray:
        """
        根据对称操作名称返回变换后的 k（此处保持不变）和操作矩阵 D。
        根据名称调用相应函数。
        """
        # 检查缓存中是否存在对应的操作矩阵
        if name == 'C3z':
            operator_name = f'C3z_{params}'  # 对于 C3z 操作，使用参数来区分
        else:
            operator_name = name
        
        if operator_name not in self.cached_operators:
            if name == "C3z":
                D = self.get_C3z_operator(params)
            elif name == "C2T":
                D = self.get_C2T_operator()
            elif name == "TR":
                D = self.get_time_reversal_matrix()
            elif name == "TR_eff":
                D = self.get_time_reversal_matrix_effective()
            elif name == "C2":
                D = self.get_C2_operator()
            elif name == "C2_eff":
                D = self.get_C2_operator()
            elif name == "C2TR_eff":
                D = self.get_C2_operator() @ self.get_time_reversal_matrix_effective()
            else:
                raise ValueError(f"Unknown symmetry operation: {name}")
            self.cached_operators[operator_name] = D
        return self.cached_operators[operator_name]

# =============================================================================
# >>> SECTION: 07. Continuum Model Core
# =============================================================================
# >>> SPLIT_HINT: move this section into model.py

class ContinuumModel:
    """
    存储所有 term 的集合，并提供组装连续模型哈密顿量的方法。

    组装公式：
       H_cont(k) = Σ_{term active} [ r_value_real * Y_symm(k) + r_value_imag * (i*Y_symm(k)) ]
    """
    def __init__(self):
        self.terms: Dict[ContinuumTermKey, ContinuumTerm] = {}
    
    def add_term(
        self,
        key: ContinuumTermKey,
        Y_basis: Callable[[np.ndarray], np.ndarray],
        tag: str = "intra",
        symmetry_ops: List[Dict[str, Any]] = None,
        registry_metadata: Dict[str, Any] | None = None,
    ):
        if symmetry_ops is None:
            symmetry_ops = []
        if key in self.terms:
            print(f"Warning: Term {key} already exists, overwriting.")
        self.terms[key] = ContinuumTerm(
            key,
            Y_basis,
            tag=tag,
            symmetry_ops=symmetry_ops,
            registry_metadata=dict(registry_metadata or {}),
        )
    
    def update_term_coefficients(self, key: ContinuumTermKey, r_real: complex, r_imag: complex):
        if key not in self.terms:
            raise ValueError(f"Term {key} does not exist.")
        self.terms[key].r_value_real = r_real
        self.terms[key].r_value_imag = r_imag

    def assemble_hamiltonian(self, k: np.ndarray, symmetry_gen: any, use_cache=True) -> np.ndarray:
        """
        对所有 active term 组装哈密顿量
        """
        H_cont = 0
        # 注意：这里组装时调用的是 builder 中封装的对称化方法（由 builder 实例调用）
        # 本方法仅简单地遍历各 term
        for term in self.terms.values():
            # print(f"Assembling term {term.key}... ativated: {term.active}")
            if term.active:
                if term.r_value_real is None or term.r_value_imag is None:
                    raise ValueError(f"Term {term.key} coefficients not assigned!")
                # 使用外部封装好的对称化函数（注意：此处仅作为占位，实际由 builder 调用）
                Y_symm, Y_symm_imag = ContinuumModelBuilder.symmetrize_Y_and_iY_basis_static(
                    term.Y_basis, k, term.symmetry_ops, symmetry_gen, term, use_cache
                )
                contribution = term.r_value_real * Y_symm + term.r_value_imag * Y_symm_imag
                if term.tag == "onsite":
                    print(f"Adding onsite energy {term.r_value_real} {term.r_value_imag} to term {term.key}")
                    print(f"Y basis for onsite term {term.key}:\n{Y_symm}")
                H_cont += contribution
        return H_cont

# =============================================================================
# >>> SECTION: 08. Builder / Pipeline
# =============================================================================
# >>> SPLIT_HINT: move this section into builder.py

class ContinuumModelBuilder:
    """
    封装连续模型构建、基函数生成、对称化、正交化与系数提取等接口，
    对外只暴露 build_terms()、compute_coefficients() 与 assemble_hamiltonian() 等接口。

    构造时需要传入：
      Q_set1, Q_set2: 两层的 Q 点（二维数组）
      n_orb1, n_orb2: 两层轨道数
      bM1, bM2: 倒格基矢（二维向量）
      intra_harmonics_map, inter_harmonics_map: 跃迁 harmonic 映射（字典）
      max_order: dict，给出 "Kinect","intra","inter" 对应的 Mz+Mz_star 最大阶数
      symmetry_gen: 一个 SymmetryGenerator 实例（根据 Q 数据生成对称操作矩阵）
    """
    _SYMMETRIZE_GLOBAL_CACHE = SYMMETRIZE_GLOBAL_CACHE
    _SYMMETRIZE_MONOMIAL_OP_CACHE = SYMMETRIZE_MONOMIAL_OP_CACHE
    _SYMMETRIZE_MONOMIAL_OP_VALIDATED = SYMMETRIZE_MONOMIAL_OP_VALIDATED
    _SYMMETRIZE_COMPOSED_OP_CACHE = SYMMETRIZE_COMPOSED_OP_CACHE
    _SYMMETRIZE_COMPOSED_OP_VALIDATED = SYMMETRIZE_COMPOSED_OP_VALIDATED
    _SYMMETRIZE_ORBIT_CACHE: Dict[Tuple[int, Tuple[float, float], Tuple[Tuple[str, Any], ...]], Any] = {}
    _KZ_POW_CACHE: Dict[Tuple[int, Tuple[float, float]], np.ndarray] = {}
    _SYMM_ANTIUNITARY_OPS = frozenset({"TR", "TR_eff", "C2T", "C2TR_eff"})
    _SYMM_UNITARY_OPS = frozenset({"C2", "C2_eff"})
    _SYMM_VALIDATE_MONOMIAL = True
    _SYMM_VALIDATE_SPARSE = True
    _SYMM_USE_SPARSE_BASIS = True
    _SYMM_MONOMIAL_CLEANUP_TOL = 1.0e-6
    _SYMM_SPARSE_VALIDATED = False
    
    def __init__(self, Q_set1: np.ndarray, Q_set2: np.ndarray,
                 n_orb1: int, n_orb2: int,
                 bM1: np.ndarray, bM2: np.ndarray,
                 intra_harmonics_map: Dict[int, np.ndarray],
                 inter_harmonics_map: Dict[int, np.ndarray],
                 max_order: Dict[str, int],
                 symmetry_gen: SymmetryGenerator,
                 symmetry_map: Dict[str, List[Dict[str, Any]]] = None,
                 term_templates: List[Dict[str, Any]] | None = None,
                 sectors: List[Dict[str, Any]] | None = None):
        self.Q_set1 = Q_set1
        self.Q_set2 = Q_set2
        self.n_orb1 = n_orb1
        self.n_orb2 = n_orb2
        self.bM1 = bM1
        self.bM2 = bM2
        self.intra_harmonics_map = intra_harmonics_map
        self.inter_harmonics_map = inter_harmonics_map
        self.max_order = max_order
        self.symmetry_gen = symmetry_gen
        self.term_templates = list(term_templates or [])
        self.sectors = list(sectors or [
            {"name": "L1", "qset": "qset1", "n_orb": int(n_orb1)},
            {"name": "L2", "qset": "qset2", "n_orb": int(n_orb2)},
        ])
        self._sector_name_to_slot = self._build_sector_name_to_slot(self.sectors)
        # 如果没有传入 symmetry_map，则使用默认设置
        self.symmetry_map = symmetry_map if symmetry_map is not None else {"Onsite": [], "Kinect": [], "intra": [], "inter": []}
        self.model = ContinuumModel()

    @staticmethod
    def _build_sector_name_to_slot(sectors: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for sector in sectors:
            name = str(sector.get("name"))
            qset = str(sector.get("qset", ""))
            if qset == "qset1":
                mapping[name] = 1
            elif qset == "qset2":
                mapping[name] = 2
            else:
                raise ValueError(f"Unsupported sector qset {qset!r}; expected qset1 or qset2")
        return mapping

    @staticmethod
    def _normalised_sector_map(raw: Any, sector_names: Sequence[str] | None = None) -> dict[str, str]:
        if isinstance(raw, Mapping):
            return {str(k): str(v) for k, v in raw.items()}
        if isinstance(raw, str):
            if raw == "layer_exchange":
                if sector_names and len(sector_names) == 2:
                    return {str(sector_names[0]): str(sector_names[1]), str(sector_names[1]): str(sector_names[0])}
                return {"L1": "L2", "L2": "L1"}
            if raw == "identity":
                return {}
        return {}

    @staticmethod
    def _uses_full_bilayer_block(sym_ops: Sequence[Mapping[str, Any]], sector_names: Sequence[str] | None = None) -> bool:
        for sym in sym_ops:
            if not isinstance(sym, Mapping):
                continue
            name = str(sym.get("name", ""))
            antiunitary = bool(sym.get("antiunitary", name in {"TR", "TR_eff", "C2T", "C2TR_eff"}))
            if not antiunitary:
                continue
            sector_map = ContinuumModelBuilder._normalised_sector_map(sym.get("sector_map", "identity"), sector_names)
            names = list(sector_names or ["L1", "L2"])
            if len(names) == 2 and sector_map.get(names[0]) == names[1] and sector_map.get(names[1]) == names[0]:
                return True
        return False
    
    @staticmethod
    def make_Y_basis_function_(key: ContinuumTermKey,
                              Q_set1: np.ndarray, Q_set2: np.ndarray,
                              n_orb1: int, n_orb2: int, tol: float = 1e-5) -> Callable[[np.ndarray], np.ndarray]:
        """
        根据 term 的 key 与各层 Q 数据生成 Y_basis 函数

        实现公式：
          [Y(k)]_{row,col} = δ(row∈layer1,a1) δ(col∈layer2,a2)
                             × (k-Q)_z^{Mz} (k-Q)_z^*^{Mz_star}
                             ，当 Q' 满足 Q = Q'+p（tol 容差）时取值，否则 0
        注意：内部使用 get_global_index() 确保排列顺序正确
        """
        p_vec = np.array(key.p)
        Mz = key.Mz
        Mz_star = key.Mz_star
        l1 = key.layer_from
        l2 = key.layer_to
        a1 = key.orbital_from - 1  # 转为 0 开始
        a2 = key.orbital_to - 1

        def Y_func(k: np.ndarray) -> np.ndarray:
            dim = Q_set1.shape[0] * n_orb1 + Q_set2.shape[0] * n_orb2
            Y = np.zeros((dim, dim), dtype=complex)
            # 选择对应层的 Q 集合
            if l1 == 1:
                Q_rows = Q_set1
            else:
                Q_rows = Q_set2
            if l2 == 1:
                Q_cols = Q_set1
            else:
                Q_cols = Q_set2
            # 遍历 Q_rows
            for i, Q in enumerate(Q_rows):
                row_index = ContinuumModelBuilder.get_global_index(l1, i, a1, Q_set1, Q_set2, n_orb1, n_orb2)
                k_minus_Q = k - Q
                kz = k_minus_Q[0] + 1j*k_minus_Q[1]
                kz_star = np.conjugate(kz)
                for j, Qp in enumerate(Q_cols):
                    if np.linalg.norm(Q - p_vec - Qp) < tol:
                        col_index = ContinuumModelBuilder.get_global_index(l2, j, a2, Q_set1, Q_set2, n_orb1, n_orb2)
                        Y[row_index, col_index] = (kz**Mz) * (kz_star**Mz_star)

            # if l1 == l2 and a1 == a2 and np.linalg.norm(p_vec) < tol:
            #     if l1 == 1:
            #         Q_rows = Q_set2
            #     else:
            #         Q_rows = Q_set1
            #     if l2 == 1:
            #         Q_cols = Q_set2
            #     else:
            #         Q_cols = Q_set1
            #     # 遍历 Q_rows
            #     for i, Q in enumerate(Q_rows):
            #         row_index = ContinuumModelBuilder.get_global_index(l1+1, i, a1, Q_set1, Q_set2, n_orb1, n_orb2)
            #         k_minus_Q = k - Q
            #         kz = k_minus_Q[0] + 1j*k_minus_Q[1]
            #         kz_star = np.conjugate(kz)
            #         for j, Qp in enumerate(Q_cols):
            #             if np.linalg.norm(Q - p_vec - Qp) < tol:
            #                 col_index = ContinuumModelBuilder.get_global_index(l2+1, j, a2, Q_set1, Q_set2, n_orb1, n_orb2)
            #                 Y[row_index, col_index] = (kz**Mz) * (kz_star**Mz_star)            
                        
            if l1 == l2 and a1 == a2 and Mz != Mz_star and np.linalg.norm(p_vec) < tol:
                return Y + Y.conj().T
            # elif l1 == l2 and a1 == a2 and np.linalg.norm(p_vec) > tol:
            #     return Y + Y.conj().T
            else:
                return Y
        return Y_func

    @staticmethod
    def make_Y_basis_function(key: ContinuumTermKey,
                          Q_set1: np.ndarray, Q_set2: np.ndarray,
                          n_orb1: int, n_orb2: int, tol: float = 1e-5) -> Callable[[np.ndarray], np.ndarray]:
        """
        根据 term 的 key 与各层 Q 数据生成 Y_basis 函数

        实现公式：
        [Y(k)]_{row,col} = δ(row∈layer1,a1) δ(col∈layer2,a2)
                            × (k-Q)_z^{Mz} (k-Q)_z^*^{Mz_star}
                            ，当 Q' 满足 Q = Q'+p（tol 容差）时取值，否则 0
        注意：内部使用 get_global_index() 确保排列顺序正确
        """
        p_vec = np.array(key.p)
        Mz = key.Mz
        Mz_star = key.Mz_star
        l1 = key.layer_from
        l2 = key.layer_to
        a1 = key.orbital_from - 1  # 转为 0 开始
        a2 = key.orbital_to - 1
        # 总维度与 Q 集合在 term 生命周期内不变，可在闭包外预计算
        dim = Q_set1.shape[0] * n_orb1 + Q_set2.shape[0] * n_orb2
        Q_rows = Q_set1 if l1 == 1 else Q_set2
        Q_cols = Q_set1 if l2 == 1 else Q_set2

        row_indices = np.array(
            [
                ContinuumModelBuilder.get_global_index(l1, i, a1, Q_set1, Q_set2, n_orb1, n_orb2)
                for i in range(Q_rows.shape[0])
            ],
            dtype=int,
        )
        col_indices = np.array(
            [
                ContinuumModelBuilder.get_global_index(l2, j, a2, Q_set1, Q_set2, n_orb1, n_orb2)
                for j in range(Q_cols.shape[0])
            ],
            dtype=int,
        )

        diff_Q = Q_rows[:, None, :] - p_vec - Q_cols[None, :, :]
        mask = np.linalg.norm(diff_Q, axis=2) < tol
        i_idx, j_idx = np.nonzero(mask)
        sparse_rows = row_indices[i_idx]
        sparse_cols = col_indices[j_idx]
        sparse_row_q_idx = i_idx.astype(int, copy=False)
        if sparse_rows.size:
            pairs = sparse_rows.astype(np.int64) * np.int64(dim) + sparse_cols.astype(np.int64)
            sparse_unique = np.unique(pairs).size == pairs.size
        else:
            sparse_unique = True

        p_norm = float(np.linalg.norm(p_vec))
        hermitize_in_basis = (l1 == l2 and a1 == a2 and Mz != Mz_star and p_norm < tol)
        # 该分支理论上只会发生在对角结构（p=0 且 Q_set 无重复）；若不满足则回退到稠密构造以保证严格等价
        can_sparse_hermitize = (not hermitize_in_basis) or (sparse_rows.size == 0 or np.all(sparse_rows == sparse_cols))

        def eval_sparse(k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            # 缓存每个 (Q_rows, k) 的 kz^m（m=0..max(Mz,Mz_star)），多 term 复用，避免重复做复幂
            k_key = (float(k[0]), float(k[1]))
            pow_key = (id(Q_rows), k_key)
            kz_pows = ContinuumModelBuilder._KZ_POW_CACHE.get(pow_key)
            needed = max(Mz, Mz_star)
            if kz_pows is None or kz_pows.shape[0] <= needed:
                diff = k - Q_rows
                kz = diff[:, 0] + 1j * diff[:, 1]
                kz_pows = np.empty((needed + 1, kz.shape[0]), dtype=complex)
                kz_pows[0] = 1.0
                if needed >= 1:
                    kz_pows[1] = kz
                    for m in range(2, needed + 1):
                        kz_pows[m] = kz_pows[m - 1] * kz
                ContinuumModelBuilder._KZ_POW_CACHE[pow_key] = kz_pows
            value = kz_pows[Mz] * np.conjugate(kz_pows[Mz_star])
            vals = value[sparse_row_q_idx]
            if hermitize_in_basis:
                # 对角情形：Y + Y† -> 2*Re(Y)
                vals = vals + np.conjugate(vals)
            return sparse_rows, sparse_cols, vals

        def Y_func(k: np.ndarray) -> np.ndarray:
            Y = np.zeros((dim, dim), dtype=complex)
            if sparse_rows.size:
                rows, cols, vals = eval_sparse(k)
                # 若存在重复索引（通常不会），+= 的高级索引会不安全；此处保守用 add.at
                np.add.at(Y, (rows, cols), vals)
            if hermitize_in_basis and not can_sparse_hermitize:
                # 极少数（非对角）情况：退回原始定义，严格实现 Y + Y†
                return Y + Y.conjugate().T
            return Y

        # 仅在结构允许且无需额外 Y+Y† 展开时，暴露稀疏评估接口供对称化快速路径使用
        if can_sparse_hermitize:
            Y_func.eval_sparse = eval_sparse  # type: ignore[attr-defined]
            Y_func._moire_sparse_dim = dim  # type: ignore[attr-defined]
            Y_func._moire_sparse_unique = sparse_unique  # type: ignore[attr-defined]
        else:
            Y_func._moire_sparse_dim = dim  # type: ignore[attr-defined]
        return Y_func
    
    @staticmethod
    def get_global_index(layer: int, q_index: int, orb: int,
                         Q_set1: np.ndarray, Q_set2: np.ndarray,
                         n_orb1: int, n_orb2: int) -> int:
        """
        根据层号、Q 点索引与轨道号返回全局矩阵索引
        """
        if layer == 1:
            return Q_set1.shape[0] * orb + q_index
        elif layer == 2:
            return Q_set1.shape[0] * n_orb1 + Q_set2.shape[0] * orb + q_index
        else:
            raise ValueError("Layer must be 1 or 2.")

    @staticmethod
    def _extract_monomial_matrix(op_matrix: np.ndarray, tol: float | None = None) -> tuple[np.ndarray, np.ndarray] | None:
        """
        若 op_matrix 为 monomial matrix（每行/每列仅一个非零元），返回 (perm, vals)：
        - perm[i] = 第 i 行非零元所在列
        - vals[i] = op_matrix[i, perm[i]]
        TAPW/kp 投影后的矩阵可能带很小的数值泄漏。只有当矩阵相对 Frobenius
        残差足够接近 monomial 时才 snap 到 monomial 快路径；真正 dense 的表示
        会返回 None 并回退到稠密乘法。
        """
        if tol is None:
            tol = ContinuumModelBuilder._SYMM_MONOMIAL_CLEANUP_TOL
        if op_matrix.ndim != 2 or op_matrix.shape[0] != op_matrix.shape[1]:
            return None
        n = op_matrix.shape[0]
        abs_op = np.abs(op_matrix)
        perm = np.argmax(abs_op, axis=1).astype(int)
        if len(set(int(index) for index in perm)) != n:
            return None
        vals = op_matrix[np.arange(n), perm]
        if np.any(np.abs(vals) <= tol):
            return None
        snapped = np.zeros_like(op_matrix)
        snapped[np.arange(n), perm] = vals
        denom = float(np.linalg.norm(op_matrix))
        if denom == 0.0:
            return None
        if float(np.linalg.norm(op_matrix - snapped) / denom) > tol:
            return None
        return perm, vals

    @staticmethod
    def _get_monomial_op(symmetry_gen: Any, op_name: str, param: Any, tol: float | None = None) -> tuple[np.ndarray, np.ndarray] | None:
        cache_key = (id(symmetry_gen), op_name, param)
        cache = ContinuumModelBuilder._SYMMETRIZE_MONOMIAL_OP_CACHE
        if cache_key in cache:
            return cache[cache_key]
        op_matrix = symmetry_gen.get_operator(op_name, param)
        mono = ContinuumModelBuilder._extract_monomial_matrix(op_matrix, tol=tol)
        cache[cache_key] = mono
        return mono

    @staticmethod
    def _validate_monomial_op_once(symmetry_gen: Any, op_name: str, param: Any, mono: tuple[np.ndarray, np.ndarray] | None) -> bool:
        if not ContinuumModelBuilder._SYMM_VALIDATE_MONOMIAL or mono is None:
            return True
        validate_key = (id(symmetry_gen), op_name, param)
        validated = ContinuumModelBuilder._SYMMETRIZE_MONOMIAL_OP_VALIDATED
        if validate_key in validated:
            return True

        perm, vals = mono
        n = perm.shape[0]
        rng = np.random.default_rng(0)
        Y0 = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))

        if op_name == "C3z":
            D = symmetry_gen.get_operator(op_name, param)
            D_inv = symmetry_gen.get_operator(op_name, -param)
            dense = D @ Y0 @ D_inv
            fast = vals[:, None] * Y0[perm, :]
            fast = fast[:, perm] * (1.0 / vals)[None, :]
        elif op_name in ContinuumModelBuilder._SYMM_ANTIUNITARY_OPS:
            D = symmetry_gen.get_operator(op_name, param)
            dense = D @ Y0.conj() @ D.conj().T
            fast = vals[:, None] * Y0.conj()[perm, :]
            fast = fast[:, perm] * np.conjugate(vals)[None, :]
        else:
            D = symmetry_gen.get_operator(op_name, param)
            dense = D @ Y0 @ D.conj().T
            fast = vals[:, None] * Y0[perm, :]
            fast = fast[:, perm] * np.conjugate(vals)[None, :]

        ok = np.allclose(fast, dense, atol=max(1.0e-10, 20.0 * ContinuumModelBuilder._SYMM_MONOMIAL_CLEANUP_TOL), rtol=0.0)
        if not ok:
            ContinuumModelBuilder._SYMMETRIZE_MONOMIAL_OP_CACHE[validate_key] = None
        validated.add(validate_key)
        return ok

    @staticmethod
    def _apply_symmetry_op_to_matrix(Y: np.ndarray, op_name: str, param: Any, symmetry_gen: Any) -> np.ndarray:
        if symmetry_gen is None:
            raise ValueError("symmetry_gen must be provided when sym_ops is non-empty.")

        if op_name == "C3z":
            mono = ContinuumModelBuilder._get_monomial_op(symmetry_gen, op_name, param)
            if mono is not None and ContinuumModelBuilder._validate_monomial_op_once(symmetry_gen, op_name, param, mono):
                perm, vals = mono
                inv_vals = 1.0 / vals
                Y = Y[np.ix_(perm, perm)]
                Y *= vals[:, None]
                Y *= inv_vals[None, :]
                return Y
            D = symmetry_gen.get_operator(op_name, param)
            D_inv = symmetry_gen.get_operator(op_name, -param)
            return D @ Y @ D_inv

        if op_name in ContinuumModelBuilder._SYMM_ANTIUNITARY_OPS:
            mono = ContinuumModelBuilder._get_monomial_op(symmetry_gen, op_name, param)
            if mono is not None and ContinuumModelBuilder._validate_monomial_op_once(symmetry_gen, op_name, param, mono):
                perm, vals = mono
                Y = Y[np.ix_(perm, perm)]
                np.conjugate(Y, out=Y)
                Y *= vals[:, None]
                Y *= np.conjugate(vals)[None, :]
                return Y
            D = symmetry_gen.get_operator(op_name, param)
            return D @ Y.conj() @ D.conj().T

        if op_name in ContinuumModelBuilder._SYMM_UNITARY_OPS:
            mono = ContinuumModelBuilder._get_monomial_op(symmetry_gen, op_name, param)
            if mono is not None and ContinuumModelBuilder._validate_monomial_op_once(symmetry_gen, op_name, param, mono):
                perm, vals = mono
                Y = Y[np.ix_(perm, perm)]
                Y *= vals[:, None]
                Y *= np.conjugate(vals)[None, :]
                return Y
            D = symmetry_gen.get_operator(op_name, param)
            return D @ Y @ D.conj().T

        raise ValueError(f"Unknown symmetry operation: {op_name}")

    @staticmethod
    def _get_composed_symmetry_action(
        symmetry_gen: Any, op_seq_applied: Tuple[Tuple[str, Any], ...]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None:
        """
        将一串对称操作（按其在 apply_symm 中的实际应用顺序）合成为一个 monomial operator：
        返回 (perm, vals, inv_vals, is_anti_total)，用于一次性实现
          Y -> U · (Y 或 Y*) · U^{-1}。

        若某一步无法走 monomial 快速路径，则返回 None（外层会回退到逐步应用，保证结果正确）。
        """
        cache_key = (id(symmetry_gen), op_seq_applied)
        cache = ContinuumModelBuilder._SYMMETRIZE_COMPOSED_OP_CACHE
        if cache_key in cache:
            return cache[cache_key]

        anti_ops = ContinuumModelBuilder._SYMM_ANTIUNITARY_OPS
        is_anti_total = False

        perm_total: np.ndarray | None = None
        vals_total: np.ndarray | None = None

        for op_name, param in op_seq_applied:
            is_anti_op = op_name in anti_ops
            mono = ContinuumModelBuilder._get_monomial_op(symmetry_gen, op_name, param)
            if mono is None or not ContinuumModelBuilder._validate_monomial_op_once(symmetry_gen, op_name, param, mono):
                cache[cache_key] = None
                return None

            perm_op, vals_op = mono
            if perm_total is None:
                n = perm_op.shape[0]
                perm_total = np.arange(n, dtype=int)
                vals_total = np.ones(n, dtype=complex)

            if is_anti_op:
                vals_total = np.conjugate(vals_total)

            perm_total = perm_total[perm_op]
            vals_total = vals_op * vals_total[perm_op]
            is_anti_total = (is_anti_total ^ is_anti_op)

        inv_vals_total = 1.0 / vals_total
        composed = (perm_total, vals_total, inv_vals_total, is_anti_total)
        cache[cache_key] = composed
        return composed

    @staticmethod
    def _validate_composed_symmetry_action_once(
        symmetry_gen: Any,
        op_seq_applied: Tuple[Tuple[str, Any], ...],
        composed: tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None,
    ) -> bool:
        if not ContinuumModelBuilder._SYMM_VALIDATE_MONOMIAL:
            return True
        validate_key = (id(symmetry_gen), op_seq_applied)
        validated = ContinuumModelBuilder._SYMMETRIZE_COMPOSED_OP_VALIDATED
        if validate_key in validated:
            return True
        validated.add(validate_key)

        if composed is None:
            return False

        perm, vals, inv_vals, is_anti_total = composed
        n = perm.shape[0]
        rng = np.random.default_rng(1)
        Y0 = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))

        # 顺序应用（与 apply_symm 一致：对矩阵逐步施加每个 op）
        Y_seq = Y0
        for op_name, param in op_seq_applied:
            Y_seq = ContinuumModelBuilder._apply_symmetry_op_to_matrix(Y_seq, op_name, param, symmetry_gen)

        # 合并后一次性应用
        Y_fast = Y0.conj() if is_anti_total else Y0
        Y_fast = vals[:, None] * Y_fast[perm, :]
        Y_fast = Y_fast[:, perm] * inv_vals[None, :]

        ok = np.allclose(Y_fast, Y_seq, atol=max(1.0e-10, 20.0 * ContinuumModelBuilder._SYMM_MONOMIAL_CLEANUP_TOL), rtol=0.0)
        if not ok:
            ContinuumModelBuilder._SYMMETRIZE_COMPOSED_OP_CACHE[validate_key] = None
        return ok

    @staticmethod
    def _apply_k_map_to_vector(kvec: np.ndarray, operation: Mapping[str, Any], *, power: int | None = None) -> np.ndarray:
        name = str(operation.get("name", ""))
        k_map = operation.get("k_map")
        if not isinstance(k_map, Mapping):
            raise ValueError(f"Operation {name!r} requires explicit k_map metadata")
        map_type = str(k_map.get("type", "")).lower()
        if map_type == "rotation":
            angle = float(k_map.get("angle_deg", 0.0))
            step = 1 if power is None else int(power)
            return rot(np.asarray(kvec, dtype=float), -angle * step)
        if map_type == "reflection":
            axis_deg = float(k_map.get("axis_deg", 0.0))
            theta = np.deg2rad(axis_deg)
            axis = np.array([np.cos(theta), np.sin(theta)], dtype=float)
            vec = np.asarray(kvec, dtype=float)
            return 2.0 * axis * float(np.dot(axis, vec)) - vec
        if map_type == "negation":
            return -np.asarray(kvec, dtype=float)
        if map_type == "identity":
            return np.asarray(kvec, dtype=float)
        raise ValueError(f"Unsupported k_map.type {k_map.get('type')!r} for operation {name!r}")

    @staticmethod
    def _generate_symmetry_orbit(base_k: np.ndarray, sym_ops: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], List[List[Tuple[str, Any]]]]:
        points = [base_k.copy()]
        op_seqs: List[List[Tuple[str, Any]]] = [[]]

        for op in sym_ops:
            op_name = op["name"]
            new_pts: List[np.ndarray] = []
            new_ops: List[List[Tuple[str, Any]]] = []
            for pt, seq in zip(points, op_seqs):
                if any(s[0] == op_name for s in seq):
                    continue
                if op_name == "C3z":
                    for n in (1, 2):
                        new_pts.append(ContinuumModelBuilder._apply_k_map_to_vector(pt, op, power=n))
                        new_ops.append(seq + [(op_name, n)])
                else:
                    new_pts.append(ContinuumModelBuilder._apply_k_map_to_vector(pt, op))
                    new_ops.append(seq + [(op_name, None)])
            points += new_pts
            op_seqs += new_ops

        return points, op_seqs

    @staticmethod
    def _get_symmetry_orbit_actions_cached(
        k: np.ndarray, sym_ops: List[Dict[str, Any]], symmetry_gen: Any
    ) -> List[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool] | None, Tuple[Tuple[str, Any], ...], bool]]:
        """
        缓存某个 (k, sym_ops) 下的 orbit 以及每个 orbit 元素的 composed monomial action（如可用）。

        返回列表元素：
        (kk, action, op_seq_applied, is_anti)
        - kk: orbit k 点
        - action: (perm, inv_perm, vals, inv_vals, is_anti_total) 或 None（需要逐步回退）
        - op_seq_applied: 逐步回退时用的操作序列（已按实际应用顺序排列，即 reversed(op_seq)）
        - is_anti: 该 orbit 元素对应的 antiunitary 总奇偶（用于 iY 的符号与统计）
        """
        k_key = tuple(float(x) for x in k)
        sym_ops_key = tuple(
            (
                op["name"],
                op.get("params", None),
                json.dumps(op.get("k_map", {}), sort_keys=True, default=str),
            )
            for op in sym_ops
        )
        cache_key = (id(symmetry_gen), k_key, sym_ops_key)
        cached = ContinuumModelBuilder._SYMMETRIZE_ORBIT_CACHE.get(cache_key)
        if cached is not None:
            return cached

        symm_points, op_seqs = ContinuumModelBuilder._generate_symmetry_orbit(k, sym_ops)
        anti_ops = ContinuumModelBuilder._SYMM_ANTIUNITARY_OPS
        orbit_actions: List[
            Tuple[
                np.ndarray,
                Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool] | None,
                Tuple[Tuple[str, Any], ...],
                bool,
            ]
        ] = []

        for kk, op_seq in zip(symm_points, op_seqs):
            if not op_seq:
                orbit_actions.append((kk, None, tuple(), False))
                continue

            op_seq_applied = tuple(reversed(op_seq))
            composed = ContinuumModelBuilder._get_composed_symmetry_action(symmetry_gen, op_seq_applied)
            if composed is not None and ContinuumModelBuilder._validate_composed_symmetry_action_once(symmetry_gen, op_seq_applied, composed):
                perm, vals, inv_vals, is_anti_total = composed
                inv_perm = np.empty_like(perm)
                inv_perm[perm] = np.arange(perm.shape[0], dtype=int)
                orbit_actions.append((kk, (perm, inv_perm, vals, inv_vals, is_anti_total), tuple(), bool(is_anti_total)))
            else:
                is_anti = (sum(1 for op_name, _ in op_seq if op_name in anti_ops) % 2) == 1
                orbit_actions.append((kk, None, op_seq_applied, bool(is_anti)))

        ContinuumModelBuilder._SYMMETRIZE_ORBIT_CACHE[cache_key] = orbit_actions
        return orbit_actions

    @staticmethod
    def symmetrize_Y_basis_static(Y_basis: Callable[[np.ndarray], np.ndarray],
                                  k: np.ndarray,
                                  sym_ops: List[Dict[str, Any]],
                                  symmetry_gen: Any = None,
                                  term = ContinuumTerm,
                                  use_cache=False) -> np.ndarray:
        """
        静态版本的对称化函数（当不需要调用 symmetry_gen 时可传 None）
        如果 symmetry_gen 不为 None，则调用 symmetry_gen.get_operator()。
        对称平均后 Hermitian 化。
        """
        if ContinuumModelBuilder._SYMMETRIZE_GLOBAL_CACHE is None:
            ContinuumModelBuilder._SYMMETRIZE_GLOBAL_CACHE = {}

        cache_key = None
        if use_cache:
            Y_id = get_function_hash(Y_basis)
            k_key = tuple(float(x) for x in k)
            sym_ops_key = tuple(
                (
                    op["name"],
                    op.get("params", None),
                    json.dumps(op.get("k_map", {}), sort_keys=True, default=str),
                )
                for op in sym_ops
            )
            term_key = None
            if isinstance(term, ContinuumTerm) and term.key is not None:
                term_key = (
                    term.key.Mz, term.key.Mz_star,
                    term.key.layer_from, term.key.layer_to,
                    term.key.orbital_from, term.key.orbital_to,
                    term.key.p
                )
            cache_key = (id(symmetry_gen), Y_id, k_key, sym_ops_key, term_key)
            cached = ContinuumModelBuilder._SYMMETRIZE_GLOBAL_CACHE.get(cache_key)
            if cached is not None:
                return cached

        use_sparse = (
            ContinuumModelBuilder._SYMM_USE_SPARSE_BASIS
            and hasattr(Y_basis, "eval_sparse")
            and symmetry_gen is not None
        )
        dim = getattr(Y_basis, "_moire_sparse_dim", None)

        if sym_ops and symmetry_gen is None:
            raise ValueError("symmetry_gen must be provided when sym_ops is non-empty.")

        orbit_actions = (
            ContinuumModelBuilder._get_symmetry_orbit_actions_cached(k, sym_ops, symmetry_gen)
            if sym_ops
            else [(k, None, tuple(), False)]
        )

        Y_symm = np.zeros((dim, dim), dtype=complex) if (use_sparse and isinstance(dim, int)) else None
        for kk, action, op_seq_applied, _is_anti in orbit_actions:
            if use_sparse and Y_symm is not None and op_seq_applied == tuple():
                rows, cols, vals0 = Y_basis.eval_sparse(kk)
                if action is not None:
                    perm, inv_perm, vals, inv_vals, is_anti_total = action
                    rr = inv_perm[rows]
                    cc = inv_perm[cols]
                    vv = np.conjugate(vals0) if is_anti_total else vals0
                    vv = vv * vals[rr] * inv_vals[cc]
                    np.add.at(Y_symm, (rr, cc), vv)
                else:
                    np.add.at(Y_symm, (rows, cols), vals0)
                continue

            # 稠密回退路径（保持原始语义）
            Y_part = Y_basis(kk)
            if action is not None:
                perm, _inv_perm, vals, inv_vals, is_anti_total = action
                Y_part = Y_part[np.ix_(perm, perm)]
                if is_anti_total:
                    np.conjugate(Y_part, out=Y_part)
                Y_part *= vals[:, None]
                Y_part *= inv_vals[None, :]
            elif op_seq_applied:
                for op_name, param in op_seq_applied:
                    Y_part = ContinuumModelBuilder._apply_symmetry_op_to_matrix(Y_part, op_name, param, symmetry_gen)

            if Y_symm is None:
                Y_symm = np.zeros_like(Y_part)
            Y_symm += Y_part

        if not np.allclose(Y_symm, Y_symm.conjugate().T):
            Y_symm = (Y_symm + Y_symm.conjugate().T)

        if use_cache and cache_key is not None:
            ContinuumModelBuilder._SYMMETRIZE_GLOBAL_CACHE[cache_key] = Y_symm

        return Y_symm

    @staticmethod
    def symmetrize_Y_and_iY_basis_static(
        Y_basis: Callable[[np.ndarray], np.ndarray],
        k: np.ndarray,
        sym_ops: List[Dict[str, Any]],
        symmetry_gen: Any = None,
        term = ContinuumTerm,
        use_cache: bool = False,
        out_real: np.ndarray | None = None,
        out_imag: np.ndarray | None = None,
        out_anti: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        同时计算 S[Y] 与 S[iY]，避免对同一 term/k 重复做一遍对称化。

        对于 unitary g:   g(iY) = i·g(Y)
        对于 antiunitary g: g(iY) = -i·g(Y)   （因为 antiunitary 会做复共轭，i -> -i）
        因而：
          S[iY](k) = Σ_g s(g)·i·g(Y)(k),
        其中 s(g)=+1 (unitary), s(g)=-1 (antiunitary)。
        """
        if ContinuumModelBuilder._SYMMETRIZE_GLOBAL_CACHE is None:
            ContinuumModelBuilder._SYMMETRIZE_GLOBAL_CACHE = {}

        cache_key = None
        if use_cache:
            Y_id = get_function_hash(Y_basis)
            k_key = tuple(float(x) for x in k)
            sym_ops_key = tuple(
                (
                    op["name"],
                    op.get("params", None),
                    json.dumps(op.get("k_map", {}), sort_keys=True, default=str),
                )
                for op in sym_ops
            )
            term_key = None
            if isinstance(term, ContinuumTerm) and term.key is not None:
                term_key = (
                    term.key.Mz, term.key.Mz_star,
                    term.key.layer_from, term.key.layer_to,
                    term.key.orbital_from, term.key.orbital_to,
                    term.key.p
                )
            cache_key = ("pair", id(symmetry_gen), Y_id, k_key, sym_ops_key, term_key)
            cached = ContinuumModelBuilder._SYMMETRIZE_GLOBAL_CACHE.get(cache_key)
            if cached is not None:
                return cached

        use_sparse = (
            ContinuumModelBuilder._SYMM_USE_SPARSE_BASIS
            and hasattr(Y_basis, "eval_sparse")
            and symmetry_gen is not None
        )
        dim = getattr(Y_basis, "_moire_sparse_dim", None)

        if sym_ops and symmetry_gen is None:
            raise ValueError("symmetry_gen must be provided when sym_ops is non-empty.")

        orbit_actions = (
            ContinuumModelBuilder._get_symmetry_orbit_actions_cached(k, sym_ops, symmetry_gen)
            if sym_ops
            else [(k, None, tuple(), False)]
        )

        def compute_pair(use_sparse_path: bool, allow_out: bool = True) -> Tuple[np.ndarray, np.ndarray]:
            use_out = (
                allow_out
                and (not use_cache)
                and out_real is not None
                and out_imag is not None
                and out_anti is not None
                and out_real.shape == out_imag.shape == out_anti.shape
                and out_real.ndim == 2
                and out_real.shape[0] == out_real.shape[1]
            )

            if use_out:
                Y_symm_local = out_real
                Y_anti_local = out_anti
                Y_symm_local.fill(0)
                Y_anti_local.fill(0)
            else:
                Y_symm_local = np.zeros((dim, dim), dtype=complex) if isinstance(dim, int) else None
                Y_anti_local = np.zeros((dim, dim), dtype=complex) if isinstance(dim, int) else None

            for kk, action, op_seq_applied, is_anti in orbit_actions:
                if use_sparse_path and Y_symm_local is not None and op_seq_applied == tuple():
                    use_add_at = not getattr(Y_basis, "_moire_sparse_unique", True)
                    rows, cols, vals0 = Y_basis.eval_sparse(kk)
                    if action is not None:
                        perm, inv_perm, vals, inv_vals, is_anti_total = action
                        rr = inv_perm[rows]
                        cc = inv_perm[cols]
                        vv = np.conjugate(vals0) if is_anti_total else vals0
                        vv = vv * vals[rr] * inv_vals[cc]
                        if use_add_at:
                            np.add.at(Y_symm_local, (rr, cc), vv)
                        else:
                            Y_symm_local[rr, cc] += vv
                        if is_anti_total:
                            if use_add_at:
                                np.add.at(Y_anti_local, (rr, cc), vv)
                            else:
                                Y_anti_local[rr, cc] += vv
                    else:
                        if use_add_at:
                            np.add.at(Y_symm_local, (rows, cols), vals0)
                        else:
                            Y_symm_local[rows, cols] += vals0
                    continue

                Y_part = Y_basis(kk)
                is_anti_total = is_anti
                if action is not None:
                    perm, _inv_perm, vals, inv_vals, is_anti_total = action
                    Y_part = Y_part[np.ix_(perm, perm)]
                    if is_anti_total:
                        np.conjugate(Y_part, out=Y_part)
                    Y_part *= vals[:, None]
                    Y_part *= inv_vals[None, :]
                elif op_seq_applied:
                    for op_name, param in op_seq_applied:
                        Y_part = ContinuumModelBuilder._apply_symmetry_op_to_matrix(Y_part, op_name, param, symmetry_gen)

                if Y_symm_local is None:
                    Y_symm_local = np.zeros_like(Y_part)
                    Y_anti_local = np.zeros_like(Y_part)
                Y_symm_local += Y_part
                if is_anti_total:
                    Y_anti_local += Y_part

            if use_out:
                Y_symm_i_local = out_imag
                Y_symm_i_local[:] = Y_symm_local
            else:
                Y_symm_i_local = Y_symm_local.copy()
            Y_symm_i_local -= Y_anti_local
            Y_symm_i_local -= Y_anti_local
            Y_symm_i_local *= 1j

            needs_herm_real = not np.allclose(Y_symm_local, Y_symm_local.conjugate().T)
            needs_herm_imag = not np.allclose(Y_symm_i_local, Y_symm_i_local.conjugate().T)

            # 记录是否触发过 Hermitian 化：用于后续更快的“直接组装”路径（不影响默认计算结果）
            if isinstance(term, ContinuumTerm):
                if not hasattr(term, "_moire_needs_hermitize_real"):
                    term._moire_needs_hermitize_real = bool(needs_herm_real)
                elif bool(getattr(term, "_moire_needs_hermitize_real")) != bool(needs_herm_real):
                    term._moire_hermitize_flags_inconsistent = True
                if not hasattr(term, "_moire_needs_hermitize_imag"):
                    term._moire_needs_hermitize_imag = bool(needs_herm_imag)
                elif bool(getattr(term, "_moire_needs_hermitize_imag")) != bool(needs_herm_imag):
                    term._moire_hermitize_flags_inconsistent = True

            if needs_herm_real:
                if use_out:
                    Y_symm_local[:] = (Y_symm_local + Y_symm_local.conjugate().T)
                else:
                    Y_symm_local = (Y_symm_local + Y_symm_local.conjugate().T)
            if needs_herm_imag:
                if use_out:
                    Y_symm_i_local[:] = (Y_symm_i_local + Y_symm_i_local.conjugate().T)
                else:
                    Y_symm_i_local = (Y_symm_i_local + Y_symm_i_local.conjugate().T)

            return Y_symm_local, Y_symm_i_local

        if use_sparse and ContinuumModelBuilder._SYMM_VALIDATE_SPARSE and not ContinuumModelBuilder._SYMM_SPARSE_VALIDATED and sym_ops:
            Y_sparse, Y_sparse_i = compute_pair(True, allow_out=False)
            Y_dense, Y_dense_i = compute_pair(False, allow_out=False)
            ok = np.allclose(Y_sparse, Y_dense, atol=1e-10, rtol=0.0) and np.allclose(Y_sparse_i, Y_dense_i, atol=1e-10, rtol=0.0)
            if not ok:
                ContinuumModelBuilder._SYMM_USE_SPARSE_BASIS = False
                ContinuumModelBuilder._SYMM_SPARSE_VALIDATED = True
                Y_symm, Y_symm_i = Y_dense, Y_dense_i
            else:
                ContinuumModelBuilder._SYMM_SPARSE_VALIDATED = True
                Y_symm, Y_symm_i = Y_sparse, Y_sparse_i
        else:
            Y_symm, Y_symm_i = compute_pair(use_sparse, allow_out=True)

        if use_cache and cache_key is not None:
            ContinuumModelBuilder._SYMMETRIZE_GLOBAL_CACHE[cache_key] = (Y_symm, Y_symm_i)

        return Y_symm, Y_symm_i

    @staticmethod
    def add_symmetrized_term_to_matrix_static(
        H_out: np.ndarray,
        Y_basis: Callable[[np.ndarray], np.ndarray],
        k: np.ndarray,
        sym_ops: List[Dict[str, Any]],
        r_value_real: float,
        r_value_imag: float,
        symmetry_gen: Any = None,
        term=ContinuumTerm,
    ) -> None:
        """
        将某个 term 在 k 点的贡献直接累加到 H_out，避免先构造 (Y_symm, Y_symm_i) 两个稠密矩阵。

        严格等价于：
          Y_symm, Y_symm_i = symmetrize_Y_and_iY_basis_static(...)
          H_out += r_real * Y_symm + r_imag * Y_symm_i

        推导：对每个 orbit 元素 g，
          unitary:   g(iY)= i g(Y)
          antiunitary: g(iY)= -i g(Y)
        因而每个 g 的权重为：
          w_g = r_real + i * s(g) * r_imag,  s(g)=+1(unitary), -1(antiunitary).
        """
        if r_value_real == 0.0 and r_value_imag == 0.0:
            return
        if sym_ops and symmetry_gen is None:
            raise ValueError("symmetry_gen must be provided when sym_ops is non-empty.")

        # Hermitian 化语义必须与 symmetrize_Y_and_iY_basis_static 完全一致：
        # - 对 Y_symm 与 Y_symm_i 分别做一次 “若非 Hermitian 则 Y<-Y+Y†” 的条件修正。
        #   这一步的触发与否通常只取决于 term/key（与 k 无关）；若检测到不同 k 下不一致，
        #   则该 term 回退到稠密路径（保证严格等价）。
        needs_herm_real = False
        needs_herm_imag = False
        inconsistent = False
        if isinstance(term, ContinuumTerm):
            inconsistent = bool(getattr(term, "_moire_hermitize_flags_inconsistent", False))
            if hasattr(term, "_moire_needs_hermitize_real"):
                needs_herm_real = bool(getattr(term, "_moire_needs_hermitize_real"))
            if hasattr(term, "_moire_needs_hermitize_imag"):
                needs_herm_imag = bool(getattr(term, "_moire_needs_hermitize_imag"))

        if inconsistent or (isinstance(term, ContinuumTerm) and (not hasattr(term, "_moire_needs_hermitize_real") or not hasattr(term, "_moire_needs_hermitize_imag"))):
            # 保守回退：直接用原函数构造 (Y_symm, Y_symm_i) 再加到 H_out。
            # 该分支只在开发/异常情况下触发，不影响默认正确性。
            Y_symm, Y_symm_i = ContinuumModelBuilder.symmetrize_Y_and_iY_basis_static(
                Y_basis, k, sym_ops, symmetry_gen=symmetry_gen, term=term, use_cache=False
            )
            if r_value_real != 0.0:
                H_out += r_value_real * Y_symm
            if r_value_imag != 0.0:
                H_out += r_value_imag * Y_symm_i
            return

        use_sparse = (
            ContinuumModelBuilder._SYMM_USE_SPARSE_BASIS
            and hasattr(Y_basis, "eval_sparse")
            and symmetry_gen is not None
        )

        orbit_actions = (
            ContinuumModelBuilder._get_symmetry_orbit_actions_cached(k, sym_ops, symmetry_gen)
            if sym_ops
            else [(k, None, tuple(), False)]
        )

        for kk, action, op_seq_applied, is_anti in orbit_actions:
            if use_sparse and op_seq_applied == tuple():
                use_add_at = not getattr(Y_basis, "_moire_sparse_unique", True)
                rows, cols, vals0 = Y_basis.eval_sparse(kk)
                if action is not None:
                    perm, inv_perm, vals, inv_vals, is_anti_total = action
                    rr = inv_perm[rows]
                    cc = inv_perm[cols]
                    vv = np.conjugate(vals0) if is_anti_total else vals0
                    vv = vv * vals[rr] * inv_vals[cc]
                else:
                    rr, cc, vv = rows, cols, vals0
                    is_anti_total = False

                # base: r_real * S[Y] + r_imag * S[iY] 的逐 orbit 累加
                s = -1.0 if is_anti_total else 1.0
                weight = r_value_real + (1j * s) * r_value_imag
                if use_add_at:
                    np.add.at(H_out, (rr, cc), weight * vv)
                else:
                    H_out[rr, cc] += weight * vv

                # conditional hermitianization: add transpose contributions if needed
                if (needs_herm_real or needs_herm_imag) and (r_value_real != 0.0 or r_value_imag != 0.0):
                    wt = (r_value_real if needs_herm_real else 0.0) + ((-1j * s) * r_value_imag if needs_herm_imag else 0.0)
                    if wt != 0.0:
                        vvH = np.conjugate(vv)
                        if use_add_at:
                            np.add.at(H_out, (cc, rr), wt * vvH)
                        else:
                            H_out[cc, rr] += wt * vvH
                continue

            # 稠密回退路径（保证正确性；通常不会触发）
            Y_part = Y_basis(kk)
            is_anti_total = is_anti
            if action is not None:
                perm, _inv_perm, vals, inv_vals, is_anti_total = action
                Y_part = Y_part[np.ix_(perm, perm)]
                if is_anti_total:
                    np.conjugate(Y_part, out=Y_part)
                Y_part *= vals[:, None]
                Y_part *= inv_vals[None, :]
            elif op_seq_applied:
                for op_name, param in op_seq_applied:
                    Y_part = ContinuumModelBuilder._apply_symmetry_op_to_matrix(Y_part, op_name, param, symmetry_gen)

            s = -1.0 if is_anti_total else 1.0
            weight = r_value_real + (1j * s) * r_value_imag
            H_out += weight * Y_part

            if needs_herm_real or needs_herm_imag:
                wt = (r_value_real if needs_herm_real else 0.0) + ((-1j * s) * r_value_imag if needs_herm_imag else 0.0)
                if wt != 0.0:
                    H_out += wt * Y_part.conjugate().T

        # 与原逻辑一致：不在此处强制 Hermitian 化（原实现是在 Y_symm/Y_symm_i 层面做 condition-allclose 再修正）

    @staticmethod
    def symmetrize_Y_basis_static_(Y_basis: Callable[[np.ndarray], np.ndarray],
                                k: np.ndarray,
                                sym_ops: List[Dict[str, Any]],
                                symmetry_gen: Any = None,
                                term=ContinuumTerm) -> np.ndarray:
        """
        静态版本的对称化函数（当不需要调用 symmetry_gen 时可传 None）
        如果 symmetry_gen 不为 None，则调用 symmetry_gen.get_operator()。
        对称平均后 Hermitian 化。
        
        这里所有矩阵操作均使用 scipy.sparse 中的稀疏矩阵，最后转换为 np.array 返回。
        """
        
        def rot(k, theta):
            """二维旋转"""
            theta = np.deg2rad(theta)
            return np.array([np.cos(theta)*k[0] - np.sin(theta)*k[1],
                            np.sin(theta)*k[0] + np.cos(theta)*k[1]])
        
        def gen_symm_k(base_k: np.ndarray) -> Tuple[List[np.ndarray], List[List[Tuple[str, Any]]]]:
            """生成对称操作后的 k 点及对应的操作序列"""
            points = [base_k.copy()]
            op_seqs = [[]]
            
            # 定义各对称操作对应的 k 变换
            op_actions = {
                'C3z': lambda k, n: rot(k, -120*n),
                'C2T': lambda k: np.array([k[0], -k[1]]),
                'C2': lambda k: np.array([-k[0], k[1]]),
                'TR': lambda k: -k
            }
            
            for op in sym_ops:
                op_name = op["name"]
                new_pts, new_ops = [], []
                for pt, seq in zip(points, op_seqs):
                    # 避免重复应用相同操作
                    if any(s[0] == op_name for s in seq):
                        continue
                    if op_name == 'C3z':
                        # 对于 C3z，生成两个旋转点（120°, 240°）
                        for n in [1, 2]:
                            new_pt = op_actions[op_name](pt, n)
                            new_pts.append(new_pt)
                            new_ops.append(seq + [(op_name, n)])
                    else:
                        new_pt = op_actions[op_name](pt)
                        new_pts.append(new_pt)
                        new_ops.append(seq + [(op_name, None)])
                points += new_pts
                op_seqs += new_ops
            
            return points, op_seqs
        
        def apply_symm(YY: Callable[[np.ndarray], np.ndarray], kk: np.ndarray, op_seq: List[Tuple[str, Any]]):
            """
            应用对称操作序列的逆操作：
            Y_symm = (1/N)Σ_g D(g) Y(g^{-1}k) D(g)^†
            这里所有操作均使用稀疏矩阵，D(g)^† 使用 getH()（即共轭转置）。
            """
            # 将 Y_basis 的结果转换为稀疏矩阵
            Y = sparse.csr_matrix(YY(kk))
            # 依次对逆序的操作进行变换
            for op in reversed(op_seq):
                op_name, param = op
                if op_name == 'C3z':
                    D = sparse.csr_matrix(symmetry_gen.get_operator(op_name, param))
                    Y = D @ Y @ D.getH()
                elif op_name in ['C2T', 'TR']:
                    D = sparse.csr_matrix(symmetry_gen.get_operator(op_name, param))
                    Y = D @ Y.conjugate() @ D.getH()
                else:
                    raise ValueError(f"Unknown symmetry operation: {op_name}")
            return Y
        
        # 生成对称操作下的 k 点与对应的操作序列
        symm_points, op_seqs = gen_symm_k(k)
        # 累加所有对称化后的矩阵
        Y_symm = None
        for kk, op_seq in zip(symm_points, op_seqs):
            Y_part = apply_symm(Y_basis, kk, op_seq)
            if Y_symm is None:
                Y_symm = Y_part
            else:
                Y_symm = Y_symm + Y_part
        # Hermitian 化：取矩阵与其共轭转置的平均
        Y_symm = (Y_symm + Y_symm.getH()) / 2
        # 转换为密集数组返回
        return Y_symm.toarray()

    # @timing_decorator_factory(0)
    def stack_Y_for_term(self, term: ContinuumTerm, k_points: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        对于 term，在各个 k 点下分别计算对称化后的两部分矩阵：
          - real 部分：直接 symmetrize_Y_basis
          - imag 部分：先乘 i，再 symmetrize_Y_basis
        最后以 block_diag 拼接后返回。
        """
        # Y_real_list = [self.symmetrize_Y_basis(term.Y_basis, k) for k in k_points]
        # Y_imag_list = [self.symmetrize_Y_basis(lambda k: 1j * term.Y_basis(k), k) for k in k_points]
        Y_pairs = [
            ContinuumModelBuilder.symmetrize_Y_and_iY_basis_static(
                term.Y_basis, k, term.symmetry_ops, symmetry_gen=self.symmetry_gen, term=term
            )
            for k in k_points
        ]
        Y_real_list = [pair[0] for pair in Y_pairs]
        Y_imag_list = [pair[1] for pair in Y_pairs]
        # if term.tag == "Kinect":
        #     print(f"Y_real_list for term {term.key}:\n{np.sum(np.abs(np.array(Y_real_list)))}")
        #     print(f"Y_imag_list for term {term.key}:\n{np.sum(np.abs(np.array(Y_imag_list)))}")
            # print(f"Y_imag_list for term {term.key}:\n{Y_imag_list[0]}")
        return (scipy.linalg.block_diag(*Y_real_list),
                scipy.linalg.block_diag(*Y_imag_list))

    def symmetrize_Y_basis(self, Y_basis: Callable[[np.ndarray], np.ndarray], k: np.ndarray) -> np.ndarray:
        """
        对给定 Y_basis 进行对称化，调用内部的 symmetry_gen
        """
        return ContinuumModelBuilder.symmetrize_Y_basis_static(Y_basis, k, 
                                                                sym_ops=[],  # 外部调用时每个 term 自带对称操作
                                                                symmetry_gen=self.symmetry_gen)

    # 以下正交化与系数求解函数与之前类似，仅不再暴露为全局函数

    @timing_decorator_factory(0)
    def orthogonalize_hermitian_matrices_(self, matlist: List[np.ndarray], tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        print(f"orthogonalize_hermitian_matrices num of matlist old: {len(matlist)} dim: {matlist[0].shape}")
        orthogonallist = []
        includinglist = []
        # mat_traceless_list = [mat - np.trace(mat)/len(mat)*np.eye(len(mat)) for mat in matlist]
        mat_traceless_list = matlist
        for i, M in tqdm(enumerate(mat_traceless_list)):
            U = M.copy()
            for Q in orthogonallist:
                Q_dagger = np.conj(Q).T
                projection = np.trace(Q_dagger @ M) / np.trace(Q_dagger @ Q)
                U -= projection * Q
            norm = np.linalg.norm(U)
            if norm > tol:
                orthogonallist.append(U / norm)
                includinglist.append(i)
        return np.array(orthogonallist), np.array(includinglist, dtype=int)
    
    @timing_decorator_factory(0)
    def orthogonalize_hermitian_matrices(self,matlist: List[np.ndarray], tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        print(f"orthogonalize_hermitian_matrices num of matlist new: {len(matlist)} dim: {matlist[0].shape}")
        orthonormal_list = []
        including_list = []
        flattened = [mat.flatten() for mat in matlist]
        for i, v in tqdm(enumerate(flattened)):
            U = v.copy()
            for w in orthonormal_list:
                # 利用 np.vdot 计算内积（假设 w 已归一化）
                projection = np.vdot(w, v)
                U -= projection * w
            norm = np.linalg.norm(U)
            if norm > tol:
                orthonormal_list.append(U / norm)
                including_list.append(i)
        # 还原形状
        n = matlist[0].shape[0]
        orthonormal_matrices = [v.reshape(n, n) for v in orthonormal_list]
        return np.array(orthonormal_matrices), np.array(including_list, dtype=int)

    @timing_decorator_factory(0)
    def get_orthogonalized_terms_subset_by_part_(self, keys: List[ContinuumTermKey], k_points: List[np.ndarray],
                                                 tol: float = 1e-8, part: str = "real", tag: str = None) -> Tuple[List[ContinuumTermKey], List[np.ndarray], np.ndarray, np.ndarray]:
        initialterms = []
        for key in keys:
            mat_real, mat_imag = self.stack_Y_for_term(self.model.terms[key], k_points)
            initialterms.append(mat_real if part=="real" else mat_imag)
        initialterms = np.array(initialterms)
        initalterms_copy = initialterms
        print("sum abs of initialterms", [np.sum(np.abs(initialterm)) for initialterm in initialterms])
        onsite_energy_list = []
        if tag == "inter":
            print("sum abs of initialterms", [np.trace(initialterm) for initialterm in initialterms])
        if tag == "Onsite" or tag == "Kinect":
            # print("tag", tag)
            # print("trace of initialterms", [np.trace(initialterm) for initialterm in initialterms])
            for i, key in enumerate(keys):
                l1, l2 = key.layer_from, key.layer_to
                orb1, orb2 = key.orbital_from, key.orbital_to
                # index_start = self.get_global_index(l1, 0, orb1-1, self.Q_set1, self.Q_set2, self.n_orb1, self.n_orb2)
                # index_end = self.get_global_index(l2, len(self.Q_set2)-1, orb2-1, self.Q_set1, self.Q_set2, self.n_orb1, self.n_orb2)
                for idx, term in enumerate(initialterms):
                    for ikx, kx in enumerate(k_points):
                        index_start = self.get_global_index(l1, 0, orb1-1, self.Q_set1, self.Q_set2, self.n_orb1, self.n_orb2) + ikx*(len(self.Q_set1)*self.n_orb1 + len(self.Q_set2)*self.n_orb2)
                        index_end = self.get_global_index(l2, len(self.Q_set2), orb2-1, self.Q_set1, self.Q_set2, self.n_orb1, self.n_orb2) + ikx*(len(self.Q_set1)*self.n_orb1 + len(self.Q_set2)*self.n_orb2)
                        # print("index_start", index_start)
                        # print("index_end", index_end)
                        index_start = 0 + ikx*60
                        index_end = 60 + ikx*60
                        onsite_energy = np.trace(term[index_start:index_end, index_start:index_end])/(index_end-index_start)
                        initialterms[idx][index_start:index_end, index_start:index_end] -= onsite_energy*np.eye(index_end-index_start)
                        onsite_energy_list.append(onsite_energy)
        # for idx, term in enumerate(initialterms):
        #     for ikx, kx in enumerate(k_points):
        #         initialterms[idx][0+ikx*60:60+ikx*60, 0+ikx*60:60+ikx*60] -= np.trace(initialterms[idx][0+ikx*60:60+ikx*60, 0+ikx*60:60+ikx*60])/(60)*np.eye(60)
        finalterms, includinglist = self.orthogonalize_hermitian_matrices(initialterms, tol=tol)
        if tag == "Onsite" and part == "real":
            includinglist = np.arange(len(keys))
            finalterms = initalterms_copy
            print("trace of finalterms", [np.trace(finalterm) for finalterm in finalterms])
        for i, key in enumerate(keys):
            if not self.model.terms[key].active:
                self.model.terms[key].active = (i in includinglist)
        return keys, initialterms, finalterms, includinglist

    def compute_coefficients_by_tag_(self, heff: np.ndarray, k_points: List[np.ndarray], tol: float = 1e-8) -> Dict[str, Dict[str, np.ndarray]]:
        tag_groups: Dict[str, List[ContinuumTermKey]] = {}
        for key, term in self.model.terms.items():
            tag_groups.setdefault(term.tag, []).append(key)
        coeffs_by_tag = {}
        for tag, keys in tag_groups.items():
            print(f"Processing tag '{tag}' with {len(keys)} terms. Time: {time.strftime('%H:%M:%S', time.localtime())}")
            group_coeffs = {}
            for part in ["real", "imag"]:
                grp_keys, initialterms, finalterms, includinglist = self.get_orthogonalized_terms_subset_by_part(keys, k_points, tol=tol, part=part, tag=tag)
                print(f"  {len(includinglist)} terms included for {part} part. Time: {time.strftime('%H:%M:%S', time.localtime())}")
                if len(finalterms) == 0:
                    group_coeffs[part] = np.array([])
                    continue
                transfermat = np.array([[np.trace(finalterms[i] @ initialterms[includinglist[j]])
                                          for j in range(len(includinglist))]
                                         for i in range(len(finalterms))])

                if tag != "Onsite":
                    rhs = np.array([np.trace(heff @ finalterm) for finalterm in finalterms])
                    coeffs = np.linalg.inv(transfermat) @ rhs
                    for idx, grp_idx in enumerate(includinglist):
                        key = grp_keys[grp_idx]
                        term = self.model.terms[key]
                        c = coeffs[idx]
                        # 对于 （onsite）项单独处理
                        # if term.tag == "Onsite":
                        #     # onsite = np.trace(heff)/len(heff)
                        #     onsite = onsite_energy_list[idx]
                            # c = onsite
                        if part=="real":
                            term.r_value_real = c
                        else:
                            term.r_value_imag = c
                else:
                    if part == "imag" and len(includinglist) > 0:
                        raise ValueError("Onsite energy should be added to real part.")
                    coeffs = []
                    for idx, grp_idx in enumerate(includinglist):
                        key = grp_keys[grp_idx]
                        l1, l2 = key.layer_from, key.layer_to
                        orb1, orb2 = key.orbital_from, key.orbital_to
                        index_start = self.get_global_index(l1, 0, orb1-1, self.Q_set1, self.Q_set2, self.n_orb1, self.n_orb2) 
                        index_end = self.get_global_index(l2, len(self.Q_set2), orb2-1, self.Q_set1, self.Q_set2, self.n_orb1, self.n_orb2)
                        
                        coeffs_i = np.trace(heff @ finalterms[idx])/((index_end-index_start)*len(k_points))
                        print(f"onsite energy for term {key}: {coeffs_i}",np.trace(finalterms[idx]))
                        term = self.model.terms[key]
                        term.r_value_real = coeffs_i
                        term.r_value_imag = 0
                        coeffs.append(coeffs_i)
                    
                group_coeffs[part] = coeffs
                print(f"Updated {part} coefficients for tag '{tag}': {_summarize_coefficients(coeffs)}")
            coeffs_by_tag[tag] = group_coeffs
        return coeffs_by_tag

    @timing_decorator_factory(0)
    def get_orthogonalized_terms_subset_old(self, keys: List[ContinuumTermKey], k_points: List[np.ndarray],
                                        tol: float = 1e-8, tag: str = None) -> Tuple[List[ContinuumTermKey], List[np.ndarray], np.ndarray, np.ndarray]:
        """
        对模型中指定 keys 的项，在 k_points 下采样后，
        对于每个 term同时采样 real 与 imag 两部分（分别由 stack_Y_for_term 返回），
        将这两部分都添加到 initialterms 中（顺序为 term1_real, term1_imag, term2_real, term2_imag, ...）。
        
        对于 onsite 或 Kinect 类项（tag=="Onsite"或tag=="Kinect"），还会在每个 k 点下对该部分做去迹处理，
        并将对应子矩阵的 trace 用于 onsite 能量的校正。
        
        返回：
        keys: 原 keys 列表（顺序不变）
        initialterms: 每个 term 得到的 block_diag 拼接矩阵（总数为 2*N）
        finalterms: 经过正交化后的矩阵数组
        includinglist: 正交化中被认为是线性独立的矩阵的原始索引数组
        """
        initialterms = []
        initialterms_copy = []
        for key in tqdm(keys):
            mat_real, mat_imag = self.stack_Y_for_term(self.model.terms[key], k_points)
            l1, l2 = key.layer_from, key.layer_to
            orb1, orb2 = key.orbital_from, key.orbital_to
            n_orb1, n_orb2 = self.n_orb1, self.n_orb2
            Q_set1 = self.Q_set1
            Q_set2 = self.Q_set2
            H_dim = len(Q_set1)*self.n_orb1 + len(Q_set2)*self.n_orb2
            
            Qlayer = Q_set1 if l1 == 1 else Q_set2
            # 若是 onsite 或 Kinect 项，则对每个 k 点对应的子块进行校正
            if tag in ("Kinect"): # l1 = l2 and orb1 = orb2
                exchange_antiunitary_flag = ContinuumModelBuilder._uses_full_bilayer_block(
                    self.symmetry_map[tag],
                    [str(sector.get("name")) for sector in self.sectors],
                )

                for ik, _ in enumerate(k_points):
                    # 计算子块索引（这里假设每个 k 点 block 的尺寸为 block_dim）
                    # block_dim = self.Q_set1.shape[0]*self.n_orb1 + self.Q_set2.shape[0]*self.n_orb2
                    # idx_start = ik * block_dim
                    # idx_end = (ik+1) * block_dim
                    if exchange_antiunitary_flag:
                        idx_start = ik * H_dim
                        idx_end = (ik+1) * H_dim
                    else:
                        idx_start = ik * H_dim + self.get_global_index(l1, 0, orb1-1, Q_set1, Q_set2, n_orb1, n_orb2)
                        idx_end = ik * H_dim + self.get_global_index(l1, len(Qlayer), orb1-1, Q_set1, Q_set2, n_orb1, n_orb2)
                    block_dim = np.abs(idx_end - idx_start)
                    if idx_end < idx_start:
                        raise ValueError(f"Invalid index range: {idx_start} to {idx_end}")
                    onsite_energy = np.trace(mat_real[idx_start:idx_end, idx_start:idx_end]) / block_dim
                    # 去除子块的平均值
                    mat_real[idx_start:idx_end, idx_start:idx_end] -= onsite_energy * np.eye(block_dim)
                    # 同时记录 onsite 能量（这里可扩展存入 term 中，此处仅作示例）
            initialterms.append(mat_real)
            initialterms.append(mat_imag)
        initialterms = np.array(initialterms)
        initialterms_copy = initialterms.copy()
        finalterms, includinglist = self.orthogonalize_hermitian_matrices(initialterms, tol=tol)
        # 更新每个 term 的 active 标志：如果 term 对应的两个矩阵中至少有一个被保留，则该 term 保持 active
        if tag == "Onsite":
            finalterms = initialterms_copy
            includinglist = np.arange(len(finalterms))
        for i, key in enumerate(keys):
            idx1, idx2 = 2*i, 2*i+1
            self.model.terms[key].active = (idx1 in includinglist or idx2 in includinglist)
        return keys, initialterms, finalterms, includinglist
    
    @timing_decorator_factory(0)
    def get_orthogonalized_terms_subset(self, keys: List[ContinuumTermKey], k_points: List[np.ndarray],
                                        tol: float = 1e-8, tag: str = None) -> Tuple[List[ContinuumTermKey], List[np.ndarray], np.ndarray, np.ndarray]:
        """
        对模型中指定 keys 的项，在 k_points 下采样后，
        对于每个 term同时采样 real 与 imag 两部分（分别由 stack_Y_for_term 返回），
        将这两部分都添加到 initialterms 中（顺序为 term1_real, term1_imag, term2_real, term2_imag, ...）。
        
        对于 onsite 或 Kinect 类项（tag=="Onsite"或tag=="Kinect"），还会在每个 k 点下对该部分做去迹处理，
        并将对应子矩阵的 trace 用于 onsite 能量的校正。
        
        返回：
        keys: 原 keys 列表（顺序不变）
        initialterms: 每个 term 得到的 block_diag 拼接矩阵（总数为 2*N）
        finalterms: 经过正交化后的矩阵数组
        includinglist: 正交化中被认为是线性独立的矩阵的原始索引数组
        """
        
        # 内部定义处理单个 key 的函数
        def _process_single_term(key):
            # 采样获得实部与虚部矩阵
            mat_real, mat_imag = self.stack_Y_for_term(self.model.terms[key], k_points)
            l1, l2 = key.layer_from, key.layer_to
            orb1, orb2 = key.orbital_from, key.orbital_to
            n_orb1, n_orb2 = self.n_orb1, self.n_orb2
            Q_set1 = self.Q_set1
            Q_set2 = self.Q_set2
            H_dim = len(Q_set1) * self.n_orb1 + len(Q_set2) * self.n_orb2
            

            Qlayer = Q_set1 if l1 == 1 else Q_set2
            
            # 若是 Kinect 项，则对每个 k 点对应的子块进行校正
            if tag in ("Kinect",):
                exchange_antiunitary_flag = ContinuumModelBuilder._uses_full_bilayer_block(
                    self.symmetry_map[tag],
                    [str(sector.get("name")) for sector in self.sectors],
                )
                for ik, _ in enumerate(k_points):
                    # if exchange_antiunitary_flag:
                        # idx_start = ik * H_dim
                        # idx_end = (ik + 1) * H_dim
                    # else:
                        # idx_start = ik * H_dim + self.get_global_index(l1, 0, orb1 - 1, Q_set1, Q_set2, n_orb1, n_orb2)
                        # idx_end = ik * H_dim + self.get_global_index(l1, len(Qlayer), orb1 - 1, Q_set1, Q_set2, n_orb1, n_orb2)
                    # block_dim = np.abs(idx_end - idx_start)
                    # if idx_end < idx_start:
                    #     raise ValueError(f"Invalid index range: {idx_start} to {idx_end}")
                    # onsite_energy = np.trace(mat_real[idx_start:idx_end, idx_start:idx_end]) / block_dim
                    # mat_real[idx_start:idx_end, idx_start:idx_end] -= onsite_energy * np.eye(block_dim)
                    sub_block = self.get_mat_blocks([mat_real], key, len(k_points))[0]
                    block_dim = sub_block.shape[0]
                    onsite_energy = np.trace(sub_block) / block_dim
                    mat_real -= onsite_energy * np.eye(mat_real.shape[0])
            # 返回该 key 对应的两个矩阵
            return mat_real, mat_imag

        # 并行处理 keys，使用所有 CPU 核心
        time_start = time.time()
        results = Parallel(n_jobs=1)(
            delayed(_process_single_term)(key) for key in tqdm(keys, desc="Processing terms")
        )
        time_end = time.time()
        print(f"Time elapsed for processing terms: {time_end - time_start:.2f} s")
        # 组合结果：每个 key 返回的两个矩阵依次放入 initialterms 列表
        initialterms = []
        for mat_real, mat_imag in results:
            initialterms.append(mat_real)
            initialterms.append(mat_imag)
            # print(f"shape of mat_real: {mat_real.shape}, mat_imag: {mat_imag.shape}")
        initialterms = np.array(initialterms)
        initialterms_copy = initialterms.copy()
        
        subgroup_0 = (keys[0].layer_from, keys[0].layer_to, keys[0].orbital_from, keys[0].orbital_to)
        for key in keys:
            subgroup = (key.layer_from, key.layer_to, key.orbital_from, key.orbital_to)
            if subgroup != subgroup_0:
                raise ValueError("Different subgroups in keys.")

        initialterms = np.array(self.get_mat_blocks(initialterms, keys[0], len(k_points)))
        initialterms_copy = initialterms.copy()
        
        # 正交化
        finalterms, includinglist = self.orthogonalize_hermitian_matrices(initialterms, tol=tol)

        # 如果 tag 为 "Onsite"，则不进行正交化，直接保留所有初始矩阵
        if tag == "Onsite":
            finalterms = initialterms_copy
            includinglist = np.arange(len(finalterms))
        
        # 更新每个 term 的 active 标志：如果该 term 对应的两个矩阵中至少有一个被保留，则 active 为 True
        for i, key in enumerate(keys):
            idx1, idx2 = 2 * i, 2 * i + 1
            self.model.terms[key].active = (idx1 in includinglist or idx2 in includinglist)
        
        return keys, initialterms, finalterms, includinglist
    
    
    def compute_coeffs_extreme(self,finalterms, initialterms, includinglist, heff):
        """
        极致向量化实现：
        transfermat[i, j] = trace(finalterms[i] @ initialterms[includinglist[j]])
        rhs[i] = trace(heff @ finalterms[i])
        
        具体方法：
        - 将 finalterms 重塑为 (m, n*n)
        - 选取 initialterms[includinglist] 后对每个矩阵先取转置，再重塑为 (k, n*n)
        - 利用 F @ I_T.T 计算 transfermat
        - 利用 F @ (heff.T).ravel() 计算 rhs
        
        返回求解出的系数 coeffs。
        """
        m, n, _ = finalterms.shape
        # 选取包含项
        init_included = initialterms[includinglist]  # shape (k, n, n)
        # 将 finalterms 重塑为 (m, n*n)
        F = finalterms.reshape(m, -1)
        # 对初始矩阵先取转置，再重塑为 (k, n*n)
        I_T = init_included.transpose(0, 2, 1).reshape(len(includinglist), -1)
        # 利用矩阵乘法计算 transfermat (m x k)
        transfermat = F @ I_T.T
        # 计算 rhs：heff 部分先转置后展平
        H_T = heff.T.ravel()
        rhs = F @ H_T
        # 求解线性方程组
        coeffs = np.linalg.solve(transfermat, rhs)
        return coeffs
    
    def compute_coeffs_extreme_(self, finalterms, initialterms, includinglist, heff):
        """
        极致向量化实现：利用 np.einsum 一次性计算 transfermat 与 rhs。
        
        transfermat[i, j] = trace( finalterms[i] @ initialterms[includinglist[j]] )
                        = sum_{p,q} finalterms[i, p, q] * initialterms[includinglist[j], q, p]
        
        rhs[i] = trace( heff @ finalterms[i] )
            = sum_{p,q} heff[p,q] * finalterms[i, q, p]
        """
        init_terms_included = initialterms[includinglist]  # shape (k, n, n)
        # 计算 transfermat：使用 einsum 直接得到形状 (m, k)
        transfermat = np.einsum('ipq,jqp->ij', finalterms, init_terms_included)
        # 计算 rhs：这里 finalterms[i] 的转置后乘以 heff
        rhs = np.einsum('pq,iqp->i', heff, finalterms)
        # 利用线性求解（比 np.linalg.inv 更稳定）
        coeffs = np.linalg.solve(transfermat, rhs)
        return coeffs
    
    
    def compute_coeffs_diag_only_(self,finalterms, initialterms, includinglist, heff,
                           reweight_diag=True, only_diag=False, use_lstsq=False):
        """
        reweight_diag=True: 对角权重设为 w_diag（其中 (0,0)=0.5，其余平分0.5）；
        only_diag=False:    非对角权重=1（不改变非对角的贡献）
        use_lstsq=False:    若方程非方阵或病态，建议设 True 用最小二乘
        """
        m, n, _ = finalterms.shape
        k = len(includinglist)
        init_included = initialterms[includinglist]          # (k,n,n)

        # 1) 构造权重矩阵 W
        # w_diag = np.full(n, 0.5/(n-1), dtype=float); w_diag[0] = 0.5
        nq = 19
        nk = (n//nq)//2
        w_diag = np.eye(nq, nq)
        w_diag[0,0] = 1e10
        w_diag[range(1,7),range(1,7)] = 1e10
        w_diag[range(1,4),range(1,4)] = 1e10
        w_diag[range(7,13),range(7,13)] = 1e10
        w_diag[range(13,19),range(13,19)] = 1e10
        w_diag[range(13,16),range(13,16)] = 1e10
        D_mat = np.zeros((nq,nq))
        D_mat[range(nq),index_new]=1
        w_diag = D_mat.T @ w_diag @ D_mat
        # w_diag = np.kron(np.eye(2), w_diag)
        # w_diag = np.tile(w_diag, (2,2))
        w_diag = np.block([[w_diag,w_diag],
                           [w_diag,w_diag]])
        W = np.tile(w_diag, (nk,nk))          # 若只想看对角，关掉非对角
        # print("shape of W:", W.shape, "shape of init_included:", np.array(init_included).shape,"nk = ",nk)
        # print("nonezero:", np.count_nonzero(W),np.nonzero(W),np.count_nonzero(init_included[0]),np.nonzero(init_included[0]))

        # 2) 展平
        F   = finalterms.reshape(m, -1)                      # vec(F_i) 行堆
        I_T = init_included.transpose(0, 2, 1).reshape(k, -1)# vec(I_j^T) 行堆
        H_T = heff.T.ravel()                                 # vec(H^T)

        # 3) 逐元素加权（关键一步）
        w_flat = W.reshape(-1)
        Fw = F * w_flat[None, :]                             # 对 F 的每个元素乘 w_ab

        # 4) 组装并求解
        transfermat = Fw @ I_T.T                             # (m×k)
        rhs        = Fw @ H_T                                # (m,)

        # if use_lstsq or (m != k):
        #     coeffs, *_ = np.linalg.lstsq(transfermat, rhs, rcond=None)
        # else:
        coeffs = np.linalg.solve(transfermat, rhs)
        
        H_rec = sum(coeffs[j] * init_included[j] for j in range(k))
        temp = np.zeros_like(heff)
        temp[H_rec.nonzero()] = heff[H_rec.nonzero()]
        diag = (H_rec-temp)/heff[H_rec.nonzero()]
        resid = np.linalg.norm(diag)
        print(np.sort(np.diag(diag).real)[:10],np.argsort(np.diag(diag).real)[:10])
        print(f'residual: {resid}')
        print("======compute_coeffs_diag_only_======")
        return coeffs
    
    
    
    @timing_decorator_factory(0)
    def compute_coefficients_by_tag_(self, heff: np.ndarray, k_points: List[np.ndarray], tol: float = 1e-8) -> Dict[str, Dict[str, np.ndarray]]:
        """
        对模型中不同 tag（例如 "Onsite", "Kinect", "intra", "inter"）的项分组求解系数，
        对于每组项，采用 get_orthogonalized_terms_subset 得到初始采样矩阵和正交化后的矩阵。
        注意：每个 term 对应两个矩阵（real 与 imag），
        最终解出的系数向量 x 为长度为 2*N 的向量，
        每个 term 的系数组合为 r = x[2*i] + i*x[2*i+1].
        
        对于 onsite 项（tag=="Onsite"）单独处理：直接使用 onsite 能量（例如取 heff 的 trace 平均）。
        
        返回一个字典，键为 tag，值为一个字典，包含 key "coeffs" 对应各项复数系数（按 keys 顺序）。
        """
        tag_groups: Dict[str, List[ContinuumTermKey]] = {}
        for key, term in self.model.terms.items():
            tag_groups.setdefault(term.tag, []).append(key)
        coeffs_by_tag = {}
        for tag, keys in tag_groups.items():
            # print("="*100)
            print("\n"+"="*100)
            print(f"Processing tag '{tag}' with {len(keys)} terms. Time: {time.strftime('%H:%M:%S', time.localtime())}")
            # 对于每组项，获取正交化结果（同时处理 real 和 imag 部分）
            grp_keys, initialterms, finalterms, includinglist = self.get_orthogonalized_terms_subset(keys, k_points, tol=tol, tag=tag)
            print(f"  {len(includinglist)} terms included. Time: {time.strftime('%H:%M:%S', time.localtime())}")
            # total = len(finalterms)  # 此处应为2*N
            if len(finalterms) == 0:
                coeffs_by_tag[tag] = {"coeffs": np.array([])}
                continue
            coeffs_print = []
            if tag == "Onsite":
                exchange_antiunitary_flag = ContinuumModelBuilder._uses_full_bilayer_block(
                    self.symmetry_map[tag],
                    [str(sector.get("name")) for sector in self.sectors],
                )
                H_dim = len(Q_set1)*self.n_orb1 + len(Q_set2)*self.n_orb2
                # 对于 onsite 项，直接使用 onsite 能量作为系数
                coeffs = []
                Qlayer = self.Q_set1 if grp_keys[0].layer_from == 1 else self.Q_set2
                
                for i in tqdm(range(len(keys))):
                    idx_real = includinglist.tolist().index(2*i) if 2*i in includinglist.tolist() else None
                    idx_imag = includinglist.tolist().index(2*i+1) if (2*i+1) in includinglist.tolist() else None
                    # l1, l2 = grp_keys[grp_idx].layer_from, grp_keys[grp_idx].layer_to
                    # orb1, orb2 = grp_keys[grp_idx].orbital_from, grp_keys[grp_idx].orbital_to
                    # n_orb1, n_orb2 = self.n_orb1, self.n_orb2
                    # Q_set1, Q_set2 = self.Q_set1, self.Q_set2
                    # if exchange_antiunitary_flag:
                    #     idx_start = 0
                    #     idx_end = H_dim
                    # else:
                    #     idx_start = self.get_global_index(l1, 0, orb1-1, Q_set1, Q_set2, n_orb1, n_orb2)
                    #     idx_end = self.get_global_index(l1, len(Q_set1), orb1-1, Q_set1, Q_set2, n_orb1, n_orb2)
                    block_dim = H_dim if exchange_antiunitary_flag else len(Qlayer)
                    # if idx_end < idx_start:
                    #     raise ValueError(f"Invalid index range: {idx_start} to {idx_end}")
                    print("sum abs of finalterms", [np.sum(np.abs(finalterm)) for finalterm in finalterms])
                    coeffs_i = np.trace(heff @ finalterms[idx_real]) / (block_dim*len(k_points))
                    coeffs.append(coeffs_i)
                    self.model.terms[keys[i]].r_value_real = coeffs_i
                    self.model.terms[keys[i]].r_value_imag = 0
                    coeffs_print.append(coeffs_i)
            else:
                # 构造 transfer matrix T: T[i,j] = trace( finalterms[i] @ initialterms[includinglist[j]] )
                print(f'before transfermat time: {time.strftime("%H:%M:%S", time.localtime())}')
                # transfermat = np.array([[np.trace(finalterms[i] @ initialterms[includinglist[j]]) 
                #                 for j in range(len(includinglist))] 
                #             for i in range(len(finalterms))])
                # # 构造右侧向量 b: b[i] = trace(heff @ finalterms[i])
                # rhs = np.array([np.trace(heff @ finalterm) for finalterm in finalterms])
                # coeffs = np.linalg.inv(transfermat) @ rhs
                coeffs = self.compute_coeffs_extreme(finalterms, initialterms, includinglist, heff)
                print(f'after transfermat time: {time.strftime("%H:%M:%S", time.localtime())}')
                # 对每个 term，组合其两个系数
                
                for i in tqdm(range(len(keys))):
                    idx_real = includinglist.tolist().index(2*i) if 2*i in includinglist.tolist() else None
                    idx_imag = includinglist.tolist().index(2*i+1) if (2*i+1) in includinglist.tolist() else None
                    r_real = coeffs[idx_real] if idx_real is not None else 0
                    r_imag = coeffs[idx_imag] if idx_imag is not None else 0
                    r = r_real + 1j*r_imag
                    coeffs_print.append(r)
                    self.model.terms[keys[i]].r_value_real = r_real
                    self.model.terms[keys[i]].r_value_imag = r_imag
            print(f"Updated coefficients for tag '{tag}': {_summarize_coefficients(coeffs_print)}")
            print("="*100)
            coeffs_by_tag[tag] = {"coeffs": np.array(coeffs)}
        return coeffs_by_tag

    # def get_mat_blocks(self, mat_list: List[np.ndarray], subgroup: Tuple[int, int, int, int], num_kpoints: int = 1) -> List[np.ndarray]:
    def get_mat_blocks(self, mat_list: List[np.ndarray], key: ContinuumTermKey, num_kpoints: int = 1) -> List[np.ndarray]:
        """
        从 mat_list 中提取子块，子块的索引由 subgroup 指定。
        subgroup 为 (layer_from, layer_to, orbital_from, orbital_to) 的一个 tuple。

        如果是对角块（layer_from == layer_to 且 orbital_from == orbital_to），直接提取子块。
        如果不是，则认为是非对角块，此时构造一个 2×2 的块矩阵：
        [[0, A],
        [B, 0]]
        其中 A 是从 (layer_from, orbital_from) 到 (layer_to, orbital_to) 的子块，
        而 B 则是 (layer_to, orbital_to) 到 (layer_from, orbital_from) 的子块。
        """
        # 解包 subgroup 信息
        l1, l2, orb1, orb2 = key.layer_from, key.layer_to, key.orbital_from, key.orbital_to
        # l1, l2, orb1, orb2 = subgroup
        n_orb1, n_orb2 = self.n_orb1, self.n_orb2
        Q_set1, Q_set2 = self.Q_set1, self.Q_set2
        H_dim = len(Q_set1)*n_orb1 + len(Q_set2)*n_orb2
        
        tag, symm = self.model.terms[key].tag, self.model.terms[key].symmetry_ops

        
        if H_dim * num_kpoints != mat_list[0].shape[0]:
            print(f"mat_list[0].shape[0]: {mat_list[0].shape[0]}, H_dim: {H_dim}, num_kpoints: {num_kpoints}")
            raise ValueError("Mismatched matrix shape and H_dim.")

        # 根据层号确定 Q 集合
        Qlayer1 = Q_set1 if l1 == 1 else Q_set2
        Qlayer2 = Q_set1 if l2 == 1 else Q_set2

        # 计算全局索引范围
        idx_start = self.get_global_index(l1, 0, orb1-1, Q_set1, Q_set2, n_orb1, n_orb2)
        idx_end   = self.get_global_index(l1, len(Qlayer1), orb1-1, Q_set1, Q_set2, n_orb1, n_orb2)
        idy_start = self.get_global_index(l2, 0, orb2-1, Q_set1, Q_set2, n_orb1, n_orb2)
        idy_end   = self.get_global_index(l2, len(Qlayer2), orb2-1, Q_set1, Q_set2, n_orb1, n_orb2)
        idx_inc = np.arange(idx_start, idx_end, dtype=int)
        idy_inc = np.arange(idy_start, idy_end, dtype=int)
        # print("="*100)
        # print(f"idx_inc: {idx_inc}, idy_inc: {idy_inc}")
        symm_ops = [sym["name"] for sym in symm]
        if 'TR' in symm_ops:
            # print(f"TR symmetry detected for tag '{tag}'")
            l1_prime, l2_prime = l1, l2
            orb1_prime, orb2_prime = 2-(orb1-1)%2+(orb1-1)//2*2, 2-(orb2-1)%2+(orb2-1)//2*2
            idx_start_prime = self.get_global_index(l1_prime, 0, orb1_prime-1, Q_set1, Q_set2, n_orb1, n_orb2)
            idx_end_prime   = self.get_global_index(l1_prime, len(Qlayer1), orb1_prime-1, Q_set1, Q_set2, n_orb1, n_orb2)
            idy_start_prime = self.get_global_index(l2_prime, 0, orb2_prime-1, Q_set1, Q_set2, n_orb1, n_orb2)
            idy_end_prime   = self.get_global_index(l2_prime, len(Qlayer2), orb2_prime-1, Q_set1, Q_set2, n_orb1, n_orb2)
            idx_inc_prime = np.arange(idx_start_prime, idx_end_prime, dtype=int)
            idy_inc_prime = np.arange(idy_start_prime, idy_end_prime, dtype=int)
            idx_inc = np.concatenate((idx_inc, idx_inc_prime))
            idy_inc = np.concatenate((idy_inc, idy_inc_prime))
        idx_inc = np.sort(idx_inc)
        idy_inc = np.sort(idy_inc)
        if not np.array_equal(np.sort(idx_inc), np.sort(idy_inc)):
            # 1) 拼起来
            combined = np.concatenate((idx_inc, idy_inc))
            # 2) 去重并排序（也可以用 np.unique，它本身就会排序并去重）
            combined = np.unique(combined)
            # 3) 赋回
            idx_inc = combined.copy()
            idy_inc = combined.copy()
        # if orb1 != orb2:
        #     print("-"*100)
        #     print(f"idx_inc: {idx_inc}, idy_inc: {idy_inc}")
        #     print(f"symm: {symm}")
        mat_blocks = []
        for mat in mat_list:
            block_list = []
            for k in range(num_kpoints):
                # idx_start_k = idx_start + k*H_dim
                # idx_end_k = idx_end + k*H_dim
                # idy_start_k = idy_start + k*H_dim
                # idy_end_k = idy_end + k*H_dim
                # if l1 == l2 and orb1 == orb2:
                #     # 对角块：直接截取
                #     block = mat[idx_start_k:idx_end_k, idy_start_k:idy_end_k]
                # else:
                #     # 非对角块：需要组合两个方向的子块
                #     # A：从 (l1,orb1) 到 (l2,orb2)
                #     A = mat[idx_start_k:idx_end_k, idy_start_k:idy_end_k]
                #     # B：从 (l2,orb2) 到 (l1,orb1)
                #     B = mat[idy_start_k:idy_end_k, idx_start_k:idx_end_k]
                #     block = np.block([[np.zeros_like(A), A], 
                #                            [B, np.zeros_like(B)]])
                idx_inc_k = idx_inc + k*H_dim
                idy_inc_k = idy_inc + k*H_dim
                block = mat[np.ix_(idx_inc_k, idy_inc_k)]
                block_list.append(block)
            mat_blocks.append(scipy.linalg.block_diag(*block_list))

        return mat_blocks
        
            
    @timing_decorator_factory(0)
    def compute_coefficients_by_tag(self, heff: np.ndarray, k_points: List[np.ndarray], tol: float = 1e-8) -> Dict[str, Dict[Any, np.ndarray]]:
        """
        先按照 tag 对模型中的 term 进行分组，
        然后在每个 tag 内再按照 (layer_from, layer_to, orbital_from, orbital_to) 进行子分组，
        分别求解每个子组的系数。

        返回的字典结构为：
        {
            tag1: { subgroup1: coeffs_array, subgroup2: coeffs_array, ... },
            tag2: { subgroup1: coeffs_array, ... },
            ...
        }
        """
        # 按 tag 分组
        tag_groups: Dict[str, List[ContinuumTermKey]] = {}
        for key, term in self.model.terms.items():
            tag_groups.setdefault(term.tag, []).append(key)
        
        coeffs_by_tag = {}
        
        # 遍历每个 tag 组
        for tag, keys in tag_groups.items():
            # if tag not in ['Kinect', 'Onsite']:
            #     continue
            symm = self.symmetry_map.get(tag, [])
            print("\n" + "="*100)
            print(
                f"Processing tag '{tag}' with {len(keys)} terms, "
                f"symmetry={_summarize_symmetry_ops(symm)}. "
                f"Time: {time.strftime('%H:%M:%S', time.localtime())}"
            )
            
            # 在当前 tag 组内按 (layer_from, layer_to, orbital_from, orbital_to) 分组
            subgroup_dict: Dict[Tuple[int, int, int, int], List[ContinuumTermKey]] = {}
            for key in keys:
                subgroup = (key.layer_from, key.layer_to, key.orbital_from, key.orbital_to)
                subgroup_dict.setdefault(subgroup, []).append(key)
            
            coeffs_by_subgroup = {}
            
            # 遍历每个子组
            for subgroup, sub_keys in subgroup_dict.items():
                print(f"  Processing subgroup {subgroup} with {len(sub_keys)} terms. Time: {time.strftime('%H:%M:%S', time.localtime())}")
                # 获取正交化结果（同时处理 real 与 imag 部分）
                generation_modes = {
                    str(self.model.terms[key].registry_metadata.get("generation_mode", "representation_invariant"))
                    for key in sub_keys
                }
                orthogonalize_subset = self.get_orthogonalized_terms_subset
                post_block_legacy_subset = False
                if len(generation_modes) == 1 and generation_modes <= {"explicit_legacy", "notebook_compatibility"}:
                    orthogonalize_subset = self.get_orthogonalized_terms_subset_old
                    post_block_legacy_subset = True
                grp_keys, initialterms, finalterms, includinglist = orthogonalize_subset(sub_keys, k_points, tol=tol, tag=tag)
                if post_block_legacy_subset:
                    initialterms = np.array(self.get_mat_blocks(list(initialterms), sub_keys[0], len(k_points)))
                    finalterms = np.array(self.get_mat_blocks(list(finalterms), sub_keys[0], len(k_points)))
                print(f"    {len(includinglist)} terms included after orthogonalization. Time: {time.strftime('%H:%M:%S', time.localtime())}")
                if len(includinglist) == 0 or np.asarray(finalterms).size == 0:
                    for key in sub_keys:
                        self.model.terms[key].r_value_real = 0.0
                        self.model.terms[key].r_value_imag = 0.0
                    coeffs_by_subgroup[subgroup] = np.array([], dtype=complex)
                    print(f"  No independent terms for subgroup {subgroup}; coefficients set to zero.")
                    continue
                
                coeffs_print = []
                # heff_block = []
                # for ik in enumerate(k_points):
                #     heff_block.append(self.get_mat_blocks([heff[ik]], subgroup)[0])
                # heff_block = scipy.linalg.block_diag(*heff_block)
                
                heff_block = self.get_mat_blocks([heff], sub_keys[0], len(k_points))[0]
                print("rank of initialterms[0]", np.linalg.matrix_rank(initialterms[0]),"shape of initialterms[0]", initialterms[0].shape)
                if np.linalg.matrix_rank(initialterms[0]) < initialterms[0].shape[0]:
                    print(f"Warning: initialterms[0] is not full rank. Rank: {np.linalg.matrix_rank(initialterms[0])}, Shape: {initialterms[0].shape}")
                    # print(f"initialterms[0]: {initialterms[0]}")
                    
                # 对于 onsite 类型单独处理
                if tag == "Onsite":
                    H_dim = len(self.Q_set1)*self.n_orb1 + len(self.Q_set2)*self.n_orb2
                    coeffs = []
                    Qlayer = self.Q_set1 if grp_keys[0].layer_from == 1 else self.Q_set2
                    for i in tqdm(range(len(sub_keys))):
                        # 找到对应的 real 部分在 includinglist 中的索引（若不存在，则返回 None）
                        try:
                            idx_real = includinglist.tolist().index(2*i)
                        except ValueError:
                            idx_real = None
                        try:
                            idx_imag = includinglist.tolist().index(2*i+1)
                        except ValueError:
                            idx_imag = None
                        # block_dim = H_dim if exchange_antiunitary_flag else len(Qlayer)
                        block_dim = np.shape(heff_block)[0]
                        H_Kinect_list = []
                        for k in k_points:
                            H_Kinect=0
                            for key, term in self.model.terms.items():
                                if term.tag == "Kinect":
                                    H_Kinect += term.r_value_real*self.symmetrize_Y_basis_static(term.Y_basis, k, term.symmetry_ops, self.symmetry_gen, term)
                            H_Kinect_list.append(H_Kinect)
                        H_Kinect_list = scipy.linalg.block_diag(*H_Kinect_list)
                        H_Kinect_list = self.get_mat_blocks([H_Kinect_list], sub_keys[0], len(k_points))[0]
                        heff_block = heff_block - H_Kinect_list
                        # if block_dim != len(finalterms[idx_real]):
                        #     raise ValueError(f"Block dimension mismatch: {block_dim} vs {len(finalterms[idx_real])}")
                        coeffs_i = np.trace(heff_block @ finalterms[idx_real]) / (block_dim) if idx_real is not None else 0
                        coeffs_i = np.real(coeffs_i)
                        coeffs.append(coeffs_i)
                        # 更新每个 term 的系数
                        self.model.terms[sub_keys[i]].r_value_real = coeffs_i
                        self.model.terms[sub_keys[i]].r_value_imag = 0
                        coeffs_print.append(coeffs_i)
                else:
                    # 对非 Onsite 项，构造 transfer matrix 并求解
                    print(f'    before transfer matrix computation time: {time.strftime("%H:%M:%S", time.localtime())}')
                    if tag == "Kinect":
                        heff_block = heff_block - np.eye(heff_block.shape[0]) * np.trace(heff_block) / heff_block.shape[0]
                        # initialterms = np.array([initialterms[i] - np.eye(initialterms[i].shape[0]) * np.trace(initialterms[i]) / initialterms[i].shape[0] for i in range(len(initialterms))])
                    # if tag ==  "inter":
                    #     coeffs = self.compute_coeffs_diag_only_(finalterms, initialterms, includinglist, heff_block)
                    # else:
                    coeffs = self.compute_coeffs_extreme(finalterms, initialterms, includinglist, heff_block)
                    
                    
                    coeffs = np.real(coeffs)
                    # print(f"diag of heff_block", np.diag(heff_block))
                    # print(f"diag of initialterms", np.diag(initialterms[0]))
                    # print(f"diag of finalterms", np.diag(finalterms[0]))
                    print(f'    after transfer matrix computation time: {time.strftime("%H:%M:%S", time.localtime())}')
                    for i in tqdm(range(len(sub_keys))):
                        try:
                            idx_real = includinglist.tolist().index(2*i)
                        except ValueError:
                            idx_real = None
                        try:
                            idx_imag = includinglist.tolist().index(2*i+1)
                        except ValueError:
                            idx_imag = None
                        r_real = coeffs[idx_real] if idx_real is not None else 0.0
                        r_imag = coeffs[idx_imag] if idx_imag is not None else 0.0
                        r = r_real + 1j*r_imag
                        coeffs_print.append(r)
                        self.model.terms[sub_keys[i]].r_value_real = r_real
                        self.model.terms[sub_keys[i]].r_value_imag = r_imag
                print(f"  Updated coefficients for subgroup {subgroup}: {_summarize_coefficients(coeffs_print)}")
                coeffs_by_subgroup[subgroup] = np.array(coeffs)
            print("="*100)
            coeffs_by_tag[tag] = coeffs_by_subgroup
        return coeffs_by_tag


    def _default_term_templates(self) -> List[Dict[str, Any]]:
        return [
            {"name": "kinetic", "source": "diagonal_kp", "sector_pairs": "same", "orbital_pairs": "diagonal", "max_order": self.max_order.get("Kinect", 0)},
            {"name": "intra", "source": "moire_potential", "sector_pairs": "same", "orbital_pairs": "all", "harmonics": "intra", "max_order": self.max_order.get("intra", 0)},
            {"name": "inter", "source": "tunneling", "sector_pairs": [[2, 1], [1, 2]], "orbital_pairs": "all", "harmonics": "inter", "max_order": self.max_order.get("inter", 0)},
        ]

    def _sector_slot_from_ref(self, ref: Any) -> int:
        if isinstance(ref, (int, np.integer)):
            slot = int(ref)
            if slot not in {1, 2}:
                raise ValueError(f"sector reference must resolve to slot 1 or 2, got {slot}")
            return slot
        name = str(ref)
        if name not in self._sector_name_to_slot:
            raise ValueError(f"Unknown sector reference {name!r}; known sectors: {sorted(self._sector_name_to_slot)}")
        return int(self._sector_name_to_slot[name])

    def _sector_name_from_slot(self, slot: int) -> str:
        for sector in self.sectors:
            if self._sector_slot_from_ref(sector.get("name")) == int(slot):
                return str(sector.get("name"))
        return f"L{int(slot)}"

    def _sector_pairs_from_template(self, template: Mapping[str, Any]) -> List[Tuple[int, int]]:
        raw = template.get("sector_pairs", "same")
        if raw == "same":
            pairs: List[Tuple[int, int]] = []
            seen: set[Tuple[int, int]] = set()
            for sector in self.sectors:
                slot = self._sector_slot_from_ref(sector.get("name"))
                pair = (slot, slot)
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
            return pairs
        pairs = []
        for pair in raw:
            if len(pair) != 2:
                raise ValueError(f"sector_pairs entries must have length 2, got {pair!r}")
            pairs.append((self._sector_slot_from_ref(pair[0]), self._sector_slot_from_ref(pair[1])))
        return pairs

    def _orbital_pairs_from_template(self, template: Mapping[str, Any], l_from: int, l_to: int) -> List[Tuple[int, int]]:
        n_from = self.n_orb1 if l_from == 1 else self.n_orb2
        n_to = self.n_orb1 if l_to == 1 else self.n_orb2
        raw = template.get("orbital_pairs", "all")
        if raw == "diagonal":
            return [(a, a) for a in range(1, min(n_from, n_to) + 1)]
        if raw == "all":
            return [(a, b) for a in range(1, n_from + 1) for b in range(1, n_to + 1)]
        return [(int(pair[0]), int(pair[1])) for pair in raw]

    def _harmonic_records_from_template(self, template: Mapping[str, Any]) -> List[Dict[str, Any]]:
        source = str(template.get("source", ""))
        if source in {"diagonal_kp", "onsite"}:
            return [
                {
                    "id": None,
                    "kind": "none",
                    "vector": np.zeros(2, dtype=float),
                    "source": "implicit_zero",
                }
            ]
        raw = template.get("harmonics", "intra" if source == "moire_potential" else "inter")
        sign = float(template.get("harmonic_sign", 1.0))
        indices = None
        if isinstance(raw, Mapping):
            sign = float(raw.get("sign", sign))
            indices = raw.get("indices")
            raw = raw.get("kind", "intra" if source == "moire_potential" else "inter")
        harmonic_name = str(raw)
        mapping = self.intra_harmonics_map if harmonic_name == "intra" else self.inter_harmonics_map
        if indices is None:
            items = sorted(mapping.items())
        else:
            items = [(int(index), mapping[int(index)]) for index in indices]
        source_label = str(template.get("harmonics_source", ""))
        if not source_label:
            lower_name = str(template.get("name", "")).lower()
            if "notebook" in lower_name:
                source_label = "legacy_notebook_explicit"
            elif "legacy" in lower_name:
                source_label = "explicit_legacy"
            elif indices is None:
                source_label = "explicit_full_mapping"
            else:
                source_label = "explicit_indices"
        return [
            {
                "id": int(key),
                "kind": harmonic_name,
                "vector": sign * np.asarray(value, dtype=float),
                "source": source_label,
            }
            for key, value in items
        ]

    @staticmethod
    def _template_generation_mode(template: Mapping[str, Any] | None) -> str:
        if template is None:
            return "representation_invariant"
        explicit_mode = str(template.get("generation_mode", "")).strip()
        if explicit_mode:
            return explicit_mode
        lower_name = str(template.get("name", "")).lower()
        legacy_filter = str(template.get("legacy_monomial_filter", template.get("monomial_filter", "")))
        if legacy_filter:
            return "notebook_compatibility"
        if "notebook" in lower_name or "legacy" in lower_name:
            return "explicit_legacy"
        return "representation_invariant"

    @staticmethod
    def _monomial_constraints(template: Mapping[str, Any] | None, source: str) -> Dict[str, Any]:
        constraints: Dict[str, Any] = {}
        if template is None:
            return constraints
        raw = template.get("monomial_constraints", {})
        if raw is None:
            raw = {}
        if raw and not isinstance(raw, Mapping):
            raise ValueError(f"monomial_constraints must be a mapping, got {raw!r}")
        constraints.update(dict(raw))
        monomial_filter = str(template.get("legacy_monomial_filter", template.get("monomial_filter", "")))
        if monomial_filter == "notebook_kinetic_c3_diag":
            legacy = {
                "exclude_m_sum_zero": True,
                "difference_mod": 3,
                "difference_residue": 0,
                "require_mz_ge_mz_star": True,
            }
            for key, value in legacy.items():
                constraints.setdefault(key, value)
        if source == "diagonal_kp":
            constraints.setdefault("exclude_m_sum_zero", True)
        return constraints

    def _monomial_orders(self, max_order: int, source: str, template: Mapping[str, Any] | None = None) -> List[Tuple[int, int]]:
        orders: List[Tuple[int, int]] = []
        constraints = self._monomial_constraints(template, source)
        for M_sum in range(0, int(max_order) + 1):
            for Mz in range(0, M_sum + 1):
                Mz_star = M_sum - Mz
                if constraints.get("exclude_m_sum_zero", False) and M_sum == 0:
                    continue
                difference_mod = constraints.get("difference_mod")
                if difference_mod is not None:
                    residue = int(constraints.get("difference_residue", 0))
                    if (Mz - Mz_star) % int(difference_mod) != residue:
                        continue
                if bool(constraints.get("require_mz_ge_mz_star", False)) and Mz_star > Mz:
                    continue
                orders.append((Mz, Mz_star))
        if source == "diagonal_kp":
            return orders
        return orders or [(0, 0)]

    def build_terms(self):
        """
        Template-driven term generation. Material-specific orbital filters must live in
        term_templates, not hard-coded Python branches.
        """
        templates = self.term_templates or self._default_term_templates()
        for template in templates:
            source = str(template.get("source", ""))
            tag = str(template.get("tag", "Kinect" if source == "diagonal_kp" else ("Onsite" if source == "onsite" else ("intra" if source == "moire_potential" else "inter"))))
            sym_ops = self.symmetry_map.get(tag, [])
            max_order = int(template.get("max_order", self.max_order.get(tag, 0)))
            generation_mode = self._template_generation_mode(template)
            if generation_mode == "notebook_compatibility":
                generated_by = "notebook_compatibility"
            elif generation_mode == "explicit_legacy":
                generated_by = "explicit_legacy"
            else:
                generated_by = "representation_invariant_generator"
            for harmonic in self._harmonic_records_from_template(template):
                p_val = np.asarray(harmonic["vector"], dtype=float)
                for Mz, Mz_star in self._monomial_orders(max_order, source, template):
                    for l_from, l_to in self._sector_pairs_from_template(template):
                        for a, b in self._orbital_pairs_from_template(template, l_from, l_to):
                            key = ContinuumTermKey(Mz, Mz_star, l_from, l_to, a, b, tuple(p_val))
                            Y_func = ContinuumModelBuilder.make_Y_basis_function(key, self.Q_set1, self.Q_set2, self.n_orb1, self.n_orb2)
                            registry_metadata = {
                                "term_name": str(template.get("name", tag)),
                                "term_kind": "kinetic" if source == "diagonal_kp" else ("onsite" if source == "onsite" else ("intra" if source == "moire_potential" else "inter")),
                                "sector_pair": [self._sector_name_from_slot(l_from), self._sector_name_from_slot(l_to)],
                                "orbital_pair": [int(a), int(b)],
                                "harmonic_id": harmonic["id"],
                                "harmonic_kind": harmonic["kind"],
                                "harmonic_vector": np.asarray(harmonic["vector"], dtype=float).tolist(),
                                "harmonics_source": str(harmonic["source"]),
                                "monomial": {"Mz": int(Mz), "Mz_star": int(Mz_star)},
                                "symmetry_orbit_id": None,
                                "generation_mode": generation_mode,
                                "generated_by": generated_by,
                                "coefficient_unit": "eV",
                                "coefficient_role": "fitted",
                            }
                            self.model.add_term(
                                key,
                                Y_func,
                                tag=tag,
                                symmetry_ops=sym_ops,
                                registry_metadata=registry_metadata,
                            )
        print("All terms generated from term_templates.")

    def get_term_metadata(self, key):
        """
        根据给定的 ContinuumTermKey，返回对应 term 的 tag 和 symmetry_ops。
        """
        try:
            term = self.model.terms[key]
        except KeyError:
            raise KeyError(f"No term found for key {key!r}")
        # # 如果 term 是字典
        # if isinstance(term, dict):
        #     return term["tag"], term["symmetry_ops"]
        # 如果 term 是对象
        return term.tag, term.symmetry_ops
    
    def get_model(self) -> ContinuumModel:
        return self.model

# =============================================================================
# >>> SECTION: 09. I/O Helpers
# =============================================================================
# >>> SPLIT_HINT: move this section into io.py

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        # 将键转换为字符串，防止循环引用
        return {str(key): convert_to_serializable(value) for key, value in obj.items()}
    else:
        return str(obj)


def save_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    """Save a Python object to JSON (numpy arrays will be converted to lists)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=convert_to_serializable)

def save_npy(path: str | Path, arr: np.ndarray) -> None:
    """Save a numpy array to .npy, creating parents as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def parse_kpath_in(file_path: str | Path) -> Tuple[int, np.ndarray, List[str]]:
    """
    Parse a VASPKIT/OpenMX-style `KPATH_*.in` file.

    Returns
    -------
    segment_points:
        int, number of interpolation points per segment.
    high_symmetry_points:
        np.ndarray of shape (Nh, 3), fractional coordinates (in reciprocal basis).
    labels:
        list[str] of length Nh, labels for each high-symmetry point.
    """
    file_path = Path(file_path)
    lines = file_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 5:
        raise ValueError(f"KPATH file too short: {file_path}")

    try:
        segment_points = int(lines[1])
    except Exception as e:
        raise ValueError(f"Failed to parse segment_points from line 2 of {file_path}: {lines[1]!r}") from e

    high_symmetry_points: list[np.ndarray] = []
    labels: list[str] = []
    for i in range(4, len(lines)):
        parts = lines[i].split()
        if len(parts) < 3:
            continue
        try:
            coordinates = np.array([float(coord) for coord in parts[:3]], dtype=float)
        except Exception:
            continue
        if coordinates.shape != (3,):
            continue
        label = parts[-1] if parts else ""
        high_symmetry_points.append(coordinates)
        labels.append(label)

    if not high_symmetry_points:
        raise ValueError(f"No high-symmetry points found in {file_path}")

    return segment_points, np.array(high_symmetry_points, dtype=float), labels


def _pairwise_path(points: Sequence[Sequence[float]], labels: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """Convert a vertex list (P0,P1,P2,...) into paired endpoints (P0,P1, P1,P2, ...)."""
    if len(points) != len(labels):
        raise ValueError(f"points and labels must have same length, got {len(points)} and {len(labels)}.")
    if len(points) < 2:
        raise ValueError("Need at least 2 points to build a k-path.")
    pts: list[np.ndarray] = []
    labs: list[str] = []
    for i in range(len(points) - 1):
        pts.append(np.asarray(points[i], dtype=float))
        pts.append(np.asarray(points[i + 1], dtype=float))
        labs.append(str(labels[i]))
        labs.append(str(labels[i + 1]))
    return np.array(pts, dtype=float), labs


def reciprocal_Tmat_from_Tmat(Tmat: np.ndarray) -> np.ndarray:
    """
    Compute reciprocal lattice vectors from direct lattice `Tmat`.

    Matches the convention used in the original script:
        reciprocal_Tmat = inv(Tmat).T * 2*pi
    """
    Tmat = np.asarray(Tmat, dtype=float)
    if Tmat.shape != (3, 3):
        raise ValueError(f"Tmat must have shape (3,3), got {Tmat.shape}.")
    return np.linalg.inv(Tmat).T * (2.0 * np.pi)


def _transform_kpoints_frac_to_model_2d(
    kpoints_frac: np.ndarray,
    *,
    reciprocal_Tmat: np.ndarray,
    phase_deg: float,
) -> np.ndarray:
    """
    Transform fractional kpoints (reciprocal basis) into the 2D model coordinates.

    This reproduces the original script's loop:
      k_dft[:2] = ([b1[:2], b2[:2]].T @ k_frac[:2])
      k_model[:2] = rot(k_dft[:2], phase_deg)
    """
    kpoints_frac = np.asarray(kpoints_frac, dtype=float)
    if kpoints_frac.ndim != 2 or kpoints_frac.shape[1] < 2:
        raise ValueError(f"kpoints_frac must have shape (Nk, >=2), got {kpoints_frac.shape}.")

    reciprocal_Tmat = np.asarray(reciprocal_Tmat, dtype=float)
    if reciprocal_Tmat.shape != (3, 3):
        raise ValueError(f"reciprocal_Tmat must have shape (3,3), got {reciprocal_Tmat.shape}.")

    mat2 = np.array([reciprocal_Tmat[0][:2], reciprocal_Tmat[1][:2]]).T
    out = np.zeros((kpoints_frac.shape[0], 2), dtype=float)
    k_dft = np.zeros(2, dtype=float)
    for i in range(kpoints_frac.shape[0]):
        k_dft[:] = (mat2 @ kpoints_frac[i, :2]).T
        out[i, :] = rot(k_dft, phase_deg)
    return out


def generate_kpath(
    *,
    Tmat: np.ndarray,
    high_symmetry_points: np.ndarray,
    labels: Sequence[str],
    segment_points: int,
    phase_deg: float = 0.0,
    reciprocal_Tmat: np.ndarray | None = None,
    output_file_path: str | Path | None = None,
) -> KPath:
    """
    Generate a k-path from paired high-symmetry endpoints.

    Parameters
    ----------
    Tmat:
        Direct lattice matrix, shape (3,3), where rows are the real-space lattice vectors.
    high_symmetry_points:
        Shape (Nh, 3), fractional coordinates in reciprocal basis. Must be even-length:
        (P0->P1), (P2->P3), ...
    labels:
        List of labels of length Nh.
    segment_points:
        Number of points per segment (matches VASPKIT `KPATH.in` line 2).
    phase_deg:
        In-plane rotation applied after converting to cartesian reciprocal coordinates.
    reciprocal_Tmat:
        Optional explicit reciprocal matrix. If None, computed from `Tmat` as inv(Tmat).T*2*pi.
    output_file_path:
        If provided, writes a `kpath.out` style file with columns (kx ky kz x).
    """
    Tmat = np.asarray(Tmat, dtype=float)
    if Tmat.shape != (3, 3):
        raise ValueError(f"Tmat must have shape (3,3), got {Tmat.shape}.")
    if reciprocal_Tmat is None:
        reciprocal_Tmat = reciprocal_Tmat_from_Tmat(Tmat)

    gen = KPathGenerator(Tmat)
    gen.generate_kpath_from_high_symmetry_points(
        np.asarray(high_symmetry_points, dtype=float),
        labels,
        int(segment_points),
        output_file_path=output_file_path,
    )

    kpoints4 = np.asarray(gen.kpoints, dtype=float)
    if kpoints4.ndim != 2 or kpoints4.shape[1] != 4:
        raise ValueError(f"Unexpected KPathGenerator.kpoints shape {kpoints4.shape}; expected (Nk,4).")
    kpoints_frac = kpoints4[:, :3]
    x = kpoints4[:, 3]
    kpoints_2d = _transform_kpoints_frac_to_model_2d(kpoints_frac, reciprocal_Tmat=reciprocal_Tmat, phase_deg=float(phase_deg))
    return KPath(
        kpoints_frac=kpoints_frac,
        x=x,
        kpoints_2d=kpoints_2d,
        x_ticks=list(gen.x_ticks),
        labels_ticks=list(gen.labels_ticks),
    )


def generate_kpath_from_vertices(
    *,
    Tmat: np.ndarray,
    vertices: Sequence[Sequence[float]],
    labels: Sequence[str],
    segment_points: int,
    phase_deg: float = 0.0,
    reciprocal_Tmat: np.ndarray | None = None,
    output_file_path: str | Path | None = None,
) -> KPath:
    """
    User-friendly wrapper: generate k-path from vertices (P0,P1,P2,...) instead of paired endpoints.
    """
    paired_points, paired_labels = _pairwise_path(vertices, labels)
    return generate_kpath(
        Tmat=np.asarray(Tmat, dtype=float),
        high_symmetry_points=paired_points,
        labels=paired_labels,
        segment_points=int(segment_points),
        phase_deg=float(phase_deg),
        reciprocal_Tmat=reciprocal_Tmat,
        output_file_path=output_file_path,
    )


def generate_kpath_from_file(
    *,
    Tmat: np.ndarray,
    file_path: str | Path,
    phase_deg: float = 0.0,
    segment_points: int | None = None,
    reciprocal_Tmat: np.ndarray | None = None,
    output_file_path: str | Path | None = None,
) -> KPath:
    """
    Generate k-path from a `KPATH_*.in` file (VASPKIT/OpenMX-style), and return 2D kpoints.

    `segment_points` can override the value stored in the file (line 2).
    """
    sp, high_symmetry_points, labels = parse_kpath_in(file_path)
    if segment_points is None:
        segment_points = sp
    return generate_kpath(
        Tmat=np.asarray(Tmat, dtype=float),
        high_symmetry_points=high_symmetry_points,
        labels=labels,
        segment_points=int(segment_points),
        phase_deg=float(phase_deg),
        reciprocal_Tmat=reciprocal_Tmat,
        output_file_path=output_file_path,
    )


_DEFAULT_HEX_HIGH_SYM: Dict[str, Tuple[float, float, float]] = {
    "G": (0.0, 0.0, 0.0),
    "Γ": (0.0, 0.0, 0.0),
    "GAMMA": (0.0, 0.0, 0.0),
    "Gamma": (0.0, 0.0, 0.0),
    "M": (0.5, 0.0, 0.0),
    "K": (1.0 / 3.0, 1.0 / 3.0, 0.0),
}


def generate_kpath_from_symbols(
    *,
    Tmat: np.ndarray,
    symbols: Sequence[str],
    segment_points: int,
    phase_deg: float = 0.0,
    coords: Mapping[str, Sequence[float]] | None = None,
    reciprocal_Tmat: np.ndarray | None = None,
    output_file_path: str | Path | None = None,
) -> KPath:
    """
    Generate k-path from a high-symmetry label sequence like ('Gamma','M','K','Gamma').

    The label sequence is expanded into paired endpoints (P0->P1, P1->P2, ...),
    matching the VASPKIT `KPATH.in` convention.
    """
    if coords is None:
        coords = _DEFAULT_HEX_HIGH_SYM

    points: list[Sequence[float]] = []
    labels: list[str] = []
    for s in symbols:
        if s in coords:
            points.append(coords[s])
            labels.append(s)
            continue
        # try a few normalizations
        s2 = s.strip()
        if s2 in coords:
            points.append(coords[s2])
            labels.append(s2)
            continue
        sU = s2.upper()
        if sU in coords:
            points.append(coords[sU])
            labels.append(s2)
            continue
        raise ValueError(f"Unknown high-symmetry symbol {s!r}. Provide `coords=` mapping to define it.")

    paired_points, paired_labels = _pairwise_path(points, labels)
    return generate_kpath(
        Tmat=np.asarray(Tmat, dtype=float),
        high_symmetry_points=paired_points,
        labels=paired_labels,
        segment_points=int(segment_points),
        phase_deg=float(phase_deg),
        reciprocal_Tmat=reciprocal_Tmat,
        output_file_path=output_file_path,
    )


def make_Q_sets_from_gvec_lists(
    Qlayer1: np.ndarray,
    Qlayer2: np.ndarray,
    *,
    rotation_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (Q_set1, Q_set2) from two `g_vec_list_*.npy` arrays.

    Reproduces the original script:
        Q_set = rot(mean(Qlayer) - Qlayer[i], rotation_deg)
    """
    Qlayer1 = np.asarray(Qlayer1, dtype=float)
    Qlayer2 = np.asarray(Qlayer2, dtype=float)
    if Qlayer1.ndim != 2 or Qlayer1.shape[1] != 2:
        raise ValueError(f"Qlayer1 must have shape (N,2), got {Qlayer1.shape}.")
    if Qlayer2.ndim != 2 or Qlayer2.shape[1] != 2:
        raise ValueError(f"Qlayer2 must have shape (N,2), got {Qlayer2.shape}.")

    c1 = np.mean(Qlayer1, axis=0)
    c2 = np.mean(Qlayer2, axis=0)
    Q_set1 = np.array([rot(c1 - Qlayer1[i], rotation_deg) for i in range(len(Qlayer1))], dtype=float)
    Q_set2 = np.array([rot(c2 - Qlayer2[i], rotation_deg) for i in range(len(Qlayer2))], dtype=float)
    return Q_set1, Q_set2


def load_Q_sets_from_gvec_files(
    file_layer1: str | Path,
    file_layer2: str | Path,
    *,
    rotation_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load `g_vec_list_*.npy` for two layers and build (Q_set1, Q_set2)."""
    file_layer1 = Path(file_layer1)
    file_layer2 = Path(file_layer2)
    if not file_layer1.exists():
        raise FileNotFoundError(file_layer1)
    if not file_layer2.exists():
        raise FileNotFoundError(file_layer2)
    Qlayer1 = np.load(file_layer1)
    Qlayer2 = np.load(file_layer2)
    return make_Q_sets_from_gvec_lists(Qlayer1, Qlayer2, rotation_deg=float(rotation_deg))


def infer_bM_vectors_from_Q_set1(Q_set1: np.ndarray, *, angle_deg: float = 60.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Infer (bM1, bM2) from Q_set1 using the original script convention.

    Original script:
      bM1 = [min(norm(Q_set1[1:])), 0]
      bM2 = rot(bM1, 60)
    """
    Q_set1 = np.asarray(Q_set1, dtype=float)
    if Q_set1.ndim != 2 or Q_set1.shape[1] != 2:
        raise ValueError(f"Q_set1 must have shape (N,2), got {Q_set1.shape}.")
    if Q_set1.shape[0] < 2:
        raise ValueError("Q_set1 must contain at least 2 points (including the zero vector).")
    mag = np.linalg.norm(Q_set1[1:], axis=1)
    b = float(np.min(mag))
    bM1 = np.array([b, 0.0], dtype=float)
    bM2 = rot(bM1, float(angle_deg))
    return bM1, bM2


def select_kpoints(kpoints: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """Select a subset of kpoints by integer indices (useful for fixed k sampling / fitting)."""
    kpoints = np.asarray(kpoints)
    if kpoints.ndim != 2 or kpoints.shape[1] != 2:
        raise ValueError(f"kpoints must have shape (Nk,2), got {kpoints.shape}.")
    idx = np.asarray(list(indices), dtype=int)
    if idx.ndim != 1:
        raise ValueError(f"indices must be 1D, got shape {idx.shape}.")
    if idx.size == 0:
        return np.empty((0, 2), dtype=float)
    if idx.min() < 0 or idx.max() >= kpoints.shape[0]:
        raise ValueError(f"indices out of range for kpoints of length {kpoints.shape[0]}: {idx}")
    return kpoints[idx]

# =============================================================================
# >>> SECTION: 10. High-level User Functions
# =============================================================================
# >>> SPLIT_HINT: move this section into api.py

@dataclass
class _BandState:
    active_terms: List[Tuple[ContinuumTerm, Callable[[np.ndarray], np.ndarray], List[Dict[str, Any]], float, float]]
    symmetry_gen: SymmetryGenerator
    keep: np.ndarray
    remove: np.ndarray
    dim_full: int
    profile_light: bool

def _validate_Q_sets(Q_set1: np.ndarray, Q_set2: np.ndarray) -> None:
    if Q_set1.ndim != 2 or Q_set2.ndim != 2:
        raise ValueError(f"Q_set1/Q_set2 must be 2D arrays, got shapes {Q_set1.shape} and {Q_set2.shape}.")
    if Q_set1.shape[1] != 2 or Q_set2.shape[1] != 2:
        raise ValueError(
            f"Q_set1/Q_set2 must have shape (N, 2) in this 2D continuum model; got {Q_set1.shape} and {Q_set2.shape}."
        )

def build_model(config: MoireConfig) -> ContinuumModel:
    """
    Build a `ContinuumModel` from `config` (terms + symmetry metadata).

    Parameters
    ----------
    config: MoireConfig
        Requires at minimum: Q_set1, Q_set2, n_orb1, n_orb2, bM1, bM2,
        intra_harmonics_map, inter_harmonics_map, max_order.

    Returns
    -------
    ContinuumModel
        A model with populated `terms` (coefficients are NOT fitted here).
    """
    if config.Q_set1 is None or config.Q_set2 is None:
        raise ValueError("config.Q_set1 and config.Q_set2 must be provided (in-memory arrays).")
    if config.bM1 is None or config.bM2 is None:
        raise ValueError("config.bM1 and config.bM2 must be provided (2D reciprocal vectors).")
    Q_set1 = np.asarray(config.Q_set1, dtype=float)
    Q_set2 = np.asarray(config.Q_set2, dtype=float)
    _validate_Q_sets(Q_set1, Q_set2)
    nlow_state = config.nlow_state if config.nlow_state is not None else [int(config.n_orb1), int(config.n_orb2)]
    if len(nlow_state) != 2:
        raise ValueError(f"config.nlow_state must have length 2, got {nlow_state}.")

    basis_template = config.symmetry_source_metadata.get("basis_template") if isinstance(config.symmetry_source_metadata, dict) else None
    symmetry_gen = config.symmetry_gen if config.symmetry_gen is not None else SymmetryGenerator(Q_set1, Q_set2, nlow_state, basis_template=basis_template)
    builder = ContinuumModelBuilder(
        Q_set1, Q_set2, int(config.n_orb1), int(config.n_orb2),
        np.asarray(config.bM1, dtype=float), np.asarray(config.bM2, dtype=float),
        dict(config.intra_harmonics_map), dict(config.inter_harmonics_map),
        dict(config.max_order),
        symmetry_gen,
        dict(config.symmetry_map) if config.symmetry_map else None,
        list(config.term_templates),
        list(config.sectors),
    )
    builder.build_terms()
    model = builder.get_model()
    # Attach context for developer convenience (not required for numeric semantics).
    setattr(model, "_moire_symmetry_gen", symmetry_gen)
    setattr(model, "_moire_builder", builder)
    return model

def compute_coefficients(config: MoireConfig, model: ContinuumModel) -> Tuple[ContinuumModel, Any]:
    """
    Fit term coefficients using `heff` and update `model` in-place.

    Parameters
    ----------
    config: MoireConfig
        Requires `heff` (block-diagonal effective Hamiltonian) and `kpoints_fit`.
    model: ContinuumModel
        A model returned by `build_model` (terms already built).

    Returns
    -------
    (model, diagnostics)
        `model` is the same object with `r_value_real/r_value_imag` populated;
        `diagnostics` is the return value of `ContinuumModelBuilder.compute_coefficients_by_tag`.
    """
    if config.Q_set1 is None or config.Q_set2 is None:
        raise ValueError("config.Q_set1 and config.Q_set2 must be provided.")
    if config.bM1 is None or config.bM2 is None:
        raise ValueError("config.bM1 and config.bM2 must be provided.")
    if config.heff is None:
        raise ValueError("config.heff must be provided for coefficient fitting.")
    if config.kpoints_fit is None:
        raise ValueError("config.kpoints_fit must be provided for coefficient fitting.")

    Q_set1 = np.asarray(config.Q_set1, dtype=float)
    Q_set2 = np.asarray(config.Q_set2, dtype=float)
    nlow_state = config.nlow_state if config.nlow_state is not None else [int(config.n_orb1), int(config.n_orb2)]
    basis_template = config.symmetry_source_metadata.get("basis_template") if isinstance(config.symmetry_source_metadata, dict) else None
    symmetry_gen = config.symmetry_gen if config.symmetry_gen is not None else SymmetryGenerator(Q_set1, Q_set2, nlow_state, basis_template=basis_template)
    builder = ContinuumModelBuilder(
        Q_set1, Q_set2, int(config.n_orb1), int(config.n_orb2),
        np.asarray(config.bM1, dtype=float), np.asarray(config.bM2, dtype=float),
        dict(config.intra_harmonics_map), dict(config.inter_harmonics_map),
        dict(config.max_order),
        symmetry_gen,
        dict(config.symmetry_map) if config.symmetry_map else None,
        list(config.term_templates),
    )
    builder.model = model
    heff = np.asarray(config.heff)
    kpts = np.asarray(config.kpoints_fit, dtype=float)
    diagnostics = builder.compute_coefficients_by_tag(heff, kpts, tol=float(config.coeff_tol))
    return model, diagnostics

def _prepare_band_state(config: MoireConfig, model: ContinuumModel) -> _BandState:
    if config.Q_set1 is None or config.Q_set2 is None:
        raise ValueError("config.Q_set1 and config.Q_set2 must be provided.")
    Q_set1 = np.asarray(config.Q_set1, dtype=float)
    Q_set2 = np.asarray(config.Q_set2, dtype=float)
    _validate_Q_sets(Q_set1, Q_set2)
    nlow_state = config.nlow_state if config.nlow_state is not None else [int(config.n_orb1), int(config.n_orb2)]
    basis_template = config.symmetry_source_metadata.get("basis_template") if isinstance(config.symmetry_source_metadata, dict) else None
    symmetry_gen = config.symmetry_gen if config.symmetry_gen is not None else SymmetryGenerator(Q_set1, Q_set2, nlow_state, basis_template=basis_template)

    dim_full = len(Q_set1) * int(config.n_orb1) + len(Q_set2) * int(config.n_orb2)

    # keep/remove (optional)
    if config.keep_indices is None:
        keep = np.arange(dim_full, dtype=int)
    else:
        keep = np.asarray(config.keep_indices, dtype=int)
    remove = np.asarray(config.remove_indices, dtype=int) if config.remove_indices is not None else np.array([], dtype=int)

    # Preserve term iteration order (insertion order).
    active_terms: List[Tuple[ContinuumTerm, Callable[[np.ndarray], np.ndarray], List[Dict[str, Any]], float, float]] = []
    for term in model.terms.values():
        if not getattr(term, "active", True):
            continue
        if term.r_value_real is None or term.r_value_imag is None:
            raise ValueError(f"Term {term.key} coefficients not assigned!")
        active_terms.append((term, term.Y_basis, term.symmetry_ops, float(term.r_value_real), float(term.r_value_imag)))

    return _BandState(active_terms=active_terms, symmetry_gen=symmetry_gen, keep=keep, remove=remove, dim_full=dim_full, profile_light=bool(config.profile_light))

def compute_bands(
    config: MoireConfig,
    model: ContinuumModel,
    kpoints: np.ndarray,
    *,
    return_eigvecs: bool | None = None,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues along a k-path / k-mesh.

    Parameters
    ----------
    config: MoireConfig
        Controls caching/profiling and optional Schur complement reduction via keep/remove.
    model: ContinuumModel
        Model with fitted coefficients (or manually assigned coefficients).
    kpoints: np.ndarray
        Shape (Nk, 2).
    return_eigvecs: bool | None
        If True, also returns eigenvectors per k. If None, uses config.save_eigvecs.

    Returns
    -------
    eigvals: np.ndarray
        Shape (Nk, dim_kept).
    (eigvals, eigvecs): Tuple[np.ndarray, np.ndarray]
        If requested, eigvecs has shape (Nk, dim_kept, dim_kept).
    """
    state = _prepare_band_state(config, model)
    kpts = np.asarray(kpoints, dtype=float)
    if kpts.ndim != 2:
        raise ValueError(f"kpoints must be a 2D array, got shape {kpts.shape}.")
    if kpts.shape[1] != 2:
        raise ValueError(f"kpoints must have shape (Nk, 2) (kx,ky), got {kpts.shape}.")

    want_vecs = bool(config.save_eigvecs) if return_eigvecs is None else bool(return_eigvecs)
    nk = kpts.shape[0]
    dim_kept = int(state.keep.size)
    eigvals_out = np.empty((nk, dim_kept), dtype=float)
    eigvecs_out = np.empty((nk, dim_kept, dim_kept), dtype=complex) if want_vecs else None

    for i in range(nk):
        _H_kept, w, v, _counts, _prof = _compute_one_k(i, kpts[i], state)
        eigvals_out[i] = w
        if want_vecs and eigvecs_out is not None:
            eigvecs_out[i] = v

    if want_vecs and eigvecs_out is not None:
        return eigvals_out, eigvecs_out
    return eigvals_out

def _compute_one_k(i: int, k: np.ndarray, state: _BandState):
    """Internal single-k routine matching the original script logic (no parallelism)."""
    t_total_start = time.perf_counter()
    t_loop = 0.0
    t_symm = 0.0
    t_schur = 0.0
    t_eig = 0.0

    H_cont = np.zeros((state.dim_full, state.dim_full), dtype=complex)

    t_loop_start = time.perf_counter()
    for term, Y_basis, sym_ops, r_value_real, r_value_imag in state.active_terms:
        t_symm_start = time.perf_counter()
        ContinuumModelBuilder.add_symmetrized_term_to_matrix_static(
            H_cont,
            Y_basis,
            k,
            sym_ops,
            r_value_real,
            r_value_imag,
            symmetry_gen=state.symmetry_gen,
            term=term,
        )
        t_symm += time.perf_counter() - t_symm_start
    t_loop = time.perf_counter() - t_loop_start

    t_schur_start = time.perf_counter()
    keep = state.keep
    remove = state.remove
    H00 = H_cont[np.ix_(keep, keep)]
    if remove.size == 0:
        H = H00
    else:
        H01 = H_cont[np.ix_(keep, remove)]
        H10 = H_cont[np.ix_(remove, keep)]
        H11 = H_cont[np.ix_(remove, remove)]
        energy = np.max(np.linalg.eigvalsh(H00))
        H = H00 + H01 @ np.linalg.inv(energy * np.eye(len(H11)) - H11) @ H10
    t_schur = time.perf_counter() - t_schur_start

    t_eig_start = time.perf_counter()
    w, v = scipy.linalg.eigh(H, check_finite=False)
    t_eig = time.perf_counter() - t_eig_start

    t_total = time.perf_counter() - t_total_start
    counts = [len(state.active_terms), 0, 0]
    profile = None
    if state.profile_light:
        profile = {
            "t_total": t_total,
            "t_loop": t_loop,
            "t_symm": t_symm,
            "t_schur": t_schur,
            "t_eig": t_eig,
        }
    return H, w, v, counts, profile

def run_end_to_end(config: MoireConfig) -> Dict[str, Any]:
    """
    Convenience pipeline: build -> fit -> bands.

    Returns a results dict containing at least:
      - model
      - eigvals
      - diagnostics (if coefficient fitting ran)
    """
    setup_logging(config.log_level)
    model = build_model(config)
    diagnostics = None
    if config.heff is not None and config.kpoints_fit is not None:
        model, diagnostics = compute_coefficients(config, model)
    if config.kpoints is None:
        raise ValueError("config.kpoints must be provided for band computation.")
    eigvals = compute_bands(config, model, config.kpoints, return_eigvecs=config.save_eigvecs)
    results: Dict[str, Any] = {"model": model, "eigvals": eigvals, "diagnostics": diagnostics}

    out_dir = Path(config.output_dir) if config.output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_npy(out_dir / "eigvals.npy", eigvals[0] if isinstance(eigvals, tuple) else eigvals)
        if diagnostics is not None:
            save_json(out_dir / "diagnostics.json", diagnostics)
    return results

# =============================================================================
# >>> SECTION: 10b. Plotting Helpers (optional)
# =============================================================================
# >>> SPLIT_HINT: move this section into plotting.py

plt = None
LineCollection = None
MultipleLocator = None
interp1d = None

def _require_plotting() -> None:
    """Lazy-import plotting dependencies on first use."""
    global plt, LineCollection, MultipleLocator, interp1d
    if plt is not None and LineCollection is not None and interp1d is not None:
        return
    try:
        import matplotlib.pyplot as _plt
        from matplotlib.collections import LineCollection as _LineCollection
        from matplotlib.ticker import MultipleLocator as _MultipleLocator
        from scipy.interpolate import interp1d as _interp1d
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Plotting dependencies (matplotlib/scipy.interpolate) are required for plotting helpers."
        ) from e
    plt = _plt
    LineCollection = _LineCollection
    MultipleLocator = _MultipleLocator
    interp1d = _interp1d

def smooth_curve(x, y, num_points=1000):
    """
    Smooth a curve using interpolation.

    Parameters:
        x (array-like): Array of x coordinates.
        y (array-like): Array of y coordinates.
        num_points (int): Number of points for interpolation (default: 1000).

    Returns:
        array-like: Smoothed y coordinates.
    """
    _require_plotting()
    f = interp1d(x, y, kind='cubic')
    x_smooth = np.linspace(x.min(), x.max(), num_points)
    return x_smooth, f(x_smooth)

def plot_band_structure_new_(ax, x, bands, values,lw, label, linestyle, offset, num,loc,vmin,vmax,cmap='viridis_r',bar_show=False,**kwargs):
    """
    Plot a band structure with values represented by color.

    Parameters:
        x (array-like): Array of x coordinates.
        bands (array-like): 2D array of y coordinates representing the bands.
        values (array-like): Array of values corresponding to each point in bands.

    Returns:
        None
    """
    _require_plotting()
    # Create line segments for each band
    segments_list = []
    for band in bands.T:
        points = np.array([x, band]).T.reshape(-1, 1, 2)
        segments_list.append(np.concatenate([points[:-1], points[1:]], axis=1))

    # Create a colormap and normalize
    cmap = plt.get_cmap(cmap)
    # cmap = plt.get_cmap('viridis')
    if vmin == 0 and vmax == 1:
        norm = plt.Normalize(np.min(values), np.max(values))
    else:
        norm = plt.Normalize(vmin, vmax)
    print("shape of segments_list",np.shape(segments_list))
    # Create subplots
    # fig, ax = plt.subplots()

    # Loop through each set of line segments and create LineCollection
    # for i, segments in enumerate(segments_list):
    #     if i == offset:
    #         print("shit")
    #         lc = LineCollection(segments, cmap=cmap, norm=norm, label=label)
    #         lc = LineCollection(segments, cmap=cmap, norm=norm)
    #         lc.set_array(values[i])
    #         lc.set_linewidth(1)
    #         line = ax.add_collection(lc)
    #     elif i < offset + num:
    #         lc = LineCollection(segments, cmap=cmap, norm=norm)
    #         lc.set_array(values[i])
    #         lc.set_linewidth(1)
    #         ax.add_collection(lc)
        # Loop through each set of line segments and create LineCollection
    for i, segments in enumerate(segments_list):
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(values[:, i])  # 设置每个点的值
        lc.set_linewidth(lw)
        ax.add_collection(lc)
        if i == offset:
            lc.set_label(label)  # 设置标签

    # for i in range(num):
    #     if i == num-1:
    #         ax.plot(x, bands[:,i+offset], color=cmap(norm(np.mean(values[:,i+offset]))), linewidth=lw, linestyle=linestyle, label=label)
    #     else:
    #         ax.plot(x, bands[:,i+offset], color=cmap(norm(np.mean(values[:,i+offset]))), linewidth=lw, linestyle=linestyle)
    

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if bar_show:
        cbar = plt.colorbar(sm, ax=ax)
        # cbar.set_label('Values')
        print(ax.get_position())
        # cbar.ax.set_position(loc)
        # 设置颜色条的刻度间隔
        cbar.locator = MultipleLocator(0.02)
        cbar.update_ticks()

def plot_band_structure_new(ax, x, bands, values, lw, label, linestyle, offset, num, loc, vmin=0, vmax=1, cmap='viridis_r',bar_show=False,**kwargs):
    """
    Plot a band structure with values represented by color.

    Parameters:
        ax (matplotlib.axes.Axes): The axes on which to plot.
        x (array-like): Array of x coordinates.
        bands (array-like): 2D array of y coordinates representing the bands.
        values (array-like): Array of values corresponding to each point in bands.
        lw (float): Line width for the bands.
        label (str): Label for the plot.
        linestyle (str): Line style for the bands.
        offset (float): Offset value for shifting the bands.
        num (int): Number of bands to plot.
        loc (str): Location for the legend.
        vmin (float): Minimum value for normalization.
        vmax (float): Maximum value for normalization.
        cmap (str): Colormap for representing values.

    Returns:
        None
    """
    _require_plotting()
    # Ensure values are in the correct shape
    values = np.array(values)

    # Create a colormap and normalize
    cmap = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin, vmax)

    for i in range(num):
        band = bands[:, i] + offset
        proj = values[:, i]

        # Create line segments for each band
        points = np.array([x, band]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a LineCollection for the current band
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(proj)
        lc.set_linewidth(lw)
        lc.set_linestyle(linestyle)
        if i == 0:
            lc.set_label(label)
        ax.add_collection(lc)

    # Add colorbar
    if bar_show:
        cbar = plt.colorbar(lc, ax=ax)
    # cbar.set_label('Projection Intensity')

    # Set labels and title
    # ax.set_xlabel('k-point')
    # ax.set_ylabel('Energy (eV)')
    ax.set_title(label)

def plot_band(ax, kx1, band, c="dodgerblue", lw=1.5, linestyle='-', label='', offset=2000, num=200, smooth=False,num_points=1000,values=[],
              proj=False,loc=[0.9,0,0.02,0.5],vmin=0,vmax=1,cmap='viridis_r',sca=False,bar_show=False,**kwargs):
    """
    Plot a band structure.

    Parameters:
        ax (Axes): Matplotlib Axes object to plot on.
        kx1 (array-like): Array of x coordinates.
        band (array-like): 2D array of y coordinates representing the bands.
        c (str): Color of the curve (default: 'dodgerblue').
        lw (float): Line width (default: 1.5).
        linestyle (str): Line style (default: '-').
        label (str): Label for the curve (default: 'TAPW').
        offset (int): Offset index (default: 2000).
        num (int): Number of curves to plot (default: 200).
        smooth (bool): Whether to apply smoothing (default: False).

    Returns:
        None
    """
    _require_plotting()
    # for i in range(num):
        # y = band[:, offset + i]
        # if smooth:
        #     y = smooth_curve(kx1, y, num_points=num_points)
        # if i == 0:
        #     ax.plot(kx1, y, c=c, linestyle=linestyle, lw=lw, label=label)
        # else:
        #     ax.plot(kx1, y, c=c, linestyle=linestyle, lw=lw)
    if smooth:
        bands_smooth = []
        for i in range(np.shape(band)[1]):
            kx1_smooth, y_smooth = smooth_curve(kx1, band[:, i], num_points=num_points)
            bands_smooth.append(y_smooth)
        bands = np.array(bands_smooth).T
        kx = kx1_smooth
    else:
        bands = band
        kx = kx1
    
    if proj:
        plot_band_structure_new(ax, kx, bands, values, lw, label,linestyle,offset,num,loc,vmin,vmax,cmap=cmap,bar_show=bar_show,**kwargs)
    else:
        if sca:
            for i in range(num):
                if len(label)>0:
                    if i == num-1:
                        ax.scatter(kx, bands[:,i+offset], c=c, s=lw, label=label, **kwargs)
                    else:
                        ax.scatter(kx, bands[:,i+offset], c=c, s=lw, **kwargs) 
                else:
                    ax.scatter(kx, bands[:,i+offset], c=c, s=lw, **kwargs)
        else:
            for i in range(num):
                if len(label)>0:
                    if i == num-1:
                        ax.plot(kx, bands[:,i+offset], c=c, linestyle=linestyle, lw=lw, label=label, **kwargs)
                    else:
                        ax.plot(kx, bands[:,i+offset], c=c, linestyle=linestyle, lw=lw, **kwargs) 
                else:
                    ax.plot(kx, bands[:,i+offset], c=c, linestyle=linestyle, lw=lw, **kwargs)

def set_pic(ax,ymin,ymax,ylim = True,meV=True,xticks =np.array([0, 0.125, 0.197, 0.341]),legend_show=True,transparent_flag = False,
            xlabels = [r'${\Gamma}_{\mathrm{M}}$', r'${\mathrm{M}}_{\mathrm{M}}$', r'${\mathrm{K}}_{\mathrm{M}}$', r'${\Gamma}_{\mathrm{M}}$'], 
            title = '4.41 Bilayer tMoTe2 w SOC G vally',save = False,savepath = '',legend_fontsize = 9):
    _require_plotting()
    # Add reference lines
    ax.plot([0, np.max(xticks)], [0, 0], c='black', lw=0.6, linestyle='-',alpha=0.4)
    for i in range(len(xticks)-2):
        ax.plot([xticks[i+1], xticks[i+1]], [ymin-0.1, ymax + 0.1], lw=0.6,c='black', linestyle='-',alpha=0.4)
    # ax.plot([0.125, 0.125], [ymin-0.1, ymax + 0.1], lw=1,c='black', linestyle='--')
    # ax.plot([0.197, 0.197], [ymin-0.1, ymax + 0.1], lw=1,c='black', linestyle='--')
    # ax.plot([0.341, 0.341], [emin, emax + 0.01], c='blue', linestyle='--')

    # Set y-axis limits
    if ylim:
        ax.set_ylim(ymin, ymax)

    # Set x-axis limits and ticks
    ax.set_xlim(np.min(xticks), np.max(xticks))
    ax.set_xticks(xticks)
    print(xticks)
    print(xlabels)
    scale = 1.5
    ax.set_xticklabels(xlabels,fontsize=11*scale,fontfamily='Times New Roman')  # Bold x-axis tick labels
    if meV:
        # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x*1000)))
        ax.set_ylabel('Energy (meV)', fontsize=12*scale,fontfamily='Times New Roman')  # Bold y-axis label
    else:
        ax.set_ylabel('Energy (eV)', fontsize=12*scale,fontfamily='Times New Roman')  # Bold y-axis label 

    # ax.set_yticks(fontproperties = 'Times New Roman', size = 12)
    # ax.tick_params(axis='y', labelsize=12, labelfamily='Times New Roman')
    y1_label = ax.get_yticklabels() 
    [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
    [y1_label_temp.set_fontsize(11*scale) for y1_label_temp in y1_label]
    [y1_label_temp.set_position((0.01, y1_label_temp.get_position()[1])) for y1_label_temp in y1_label]  # Adjust the x position



    # Set plot title
    ax.set_title(title,fontsize=12*scale,fontfamily= 'Times New Roman')
    if legend_show:
        legend = ax.legend(loc='upper right',fontsize=legend_fontsize*scale)
        labelss = legend.get_texts()
        [label.set_fontname('Times New Roman') for label in labelss]

    # Show the plot
    if save:
        ax.set_facecolor('none')
        plt.savefig(savepath, dpi=300,bbox_inches='tight',transparent=transparent_flag)
    plt.show()

# =============================================================================
# >>> SECTION: 11. Self-test & Example
# =============================================================================
# >>> SPLIT_HINT: move this section into self_test.py

def example_config() -> MoireConfig:
    """
    Return a runnable config for `main()` sanity run.

    Preference order:
      1) Use the local `kp/configs/mgi2_G/plots_mgi2_5_Gamma/*` files if present.
      2) Fallback to a tiny synthetic config if the files are missing.
    """

    def _synthetic() -> MoireConfig:
        Q1 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
        Q2 = np.array([[0.0, 0.0], [-1.0, 0.0], [0.0, -1.0]], dtype=float)
        bM1 = np.array([1.0, 0.0], dtype=float)
        bM2 = rot(bM1, 60)
        intra = {1: bM1 * 0.0, 2: -bM1}
        inter = {1: bM1 * 0.0, 2: -bM1}
        max_order = {"Kinect": 1, "intra": 1, "inter": 1}
        symmetry_map = {"Onsite": [], "Kinect": [], "intra": [], "inter": []}
        kpoints = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]], dtype=float)
        return MoireConfig(
            Q_set1=Q1,
            Q_set2=Q2,
            n_orb1=2,
            n_orb2=2,
            bM1=bM1,
            bM2=bM2,
            intra_harmonics_map=intra,
            inter_harmonics_map=inter,
            max_order=max_order,
            symmetry_map=symmetry_map,
            kpoints=kpoints,
            use_cache=False,
            profile_light=True,
            log_level=logging.INFO,
        )

    try:
        # cfg_dir = Path(__file__).resolve().parent.parent  # kp/configs/mgi2_G
        cfg_dir = Path("/Users/xtz/code/TAPW_tmdc/tMgI2/moirekp/kp/configs/mgi2_G")
        plots_dir = cfg_dir / "plots_mgi2_5_Gamma"
        g1 = plots_dir / "g_vec_list_5_Gamma_1layer.npy"
        g2 = plots_dir / "g_vec_list_5_Gamma_2layer.npy"
        kpath_in = plots_dir / "KPATH_GMKG.in"
        if not (g1.exists() and g2.exists() and kpath_in.exists()):
            return _synthetic()

        phase_deg = 210.0
        Q_set1, Q_set2 = load_Q_sets_from_gvec_files(g1, g2, rotation_deg=phase_deg)
        bM1, bM2 = infer_bM_vectors_from_Q_set1(Q_set1, angle_deg=60.0)

        intra = {1: bM1 * 0.0, 2: -bM1}
        inter = {1: bM1 * 0.0, 2: -bM1}

        # Original script Tmat (direct lattice, rows are vectors).
        Tmat = np.array(
            [
                [39.4387956162, 0.0, 0.0],
                [-19.7193978074, 34.1549988985, 0.0],
                [0.0, 0.0, 50.0],
            ],
            dtype=float,
        )
        kpath_out = plots_dir / "kpath.out"
        kpath = generate_kpath_from_file(
            Tmat=Tmat,
            file_path=kpath_in,
            phase_deg=phase_deg,
            output_file_path=kpath_out,
        )

        max_order = {"Kinect": 10, "intra": 4, "inter": 4}
        symmetry_map = {
            "Kinect": [{"name": "TR"}, {"name": "C2"}],
            "Onsite": [{"name": "TR"}, {"name": "C2"}],
            "intra": [{"name": "C3z"}, {"name": "TR"}, {"name": "C2"}],
            "inter": [{"name": "C3z"}, {"name": "TR"}, {"name": "C2"}],
        }

        return MoireConfig(
            Q_set1=Q_set1,
            Q_set2=Q_set2,
            n_orb1=2,
            n_orb2=2,
            nlow_state=[2, 2],
            bM1=bM1,
            bM2=bM2,
            intra_harmonics_map=intra,
            inter_harmonics_map=inter,
            max_order=max_order,
            symmetry_map=symmetry_map,
            kpoints=kpath.kpoints_2d,
            use_cache=False,
            profile_light=True,
            log_level=logging.INFO,
            Tmat=Tmat,
            phase_deg=phase_deg,
            kpath_file=kpath_in,
            kpath_out_file=kpath_out,
            gvec_file_layer1=g1,
            gvec_file_layer2=g2,
        )
    except Exception:
        # Keep `main()` robust even if local files are missing/misconfigured.
        return _synthetic()


def self_test(*, k_index: int = 0) -> None:
    """Sanity run: build terms, assemble H(k), run eigh, and report basics."""
    cfg = example_config()
    setup_logging(cfg.log_level)

    model = build_model(cfg)

    # Assign synthetic coefficients so at least some terms contribute.
    rng = np.random.default_rng(0)
    for term in model.terms.values():
        term.active = True
        term.r_value_real = float(rng.normal(scale=0.1))
        term.r_value_imag = float(rng.normal(scale=0.1))

    if cfg.kpoints is None:
        raise ValueError("example_config() returned config.kpoints=None")
    if not (0 <= int(k_index) < len(cfg.kpoints)):
        raise ValueError(f"k_index={k_index} out of range for kpoints of length {len(cfg.kpoints)}")
    k0 = np.asarray(cfg.kpoints[int(k_index)], dtype=float)
    symmetry_gen = getattr(model, "_moire_symmetry_gen", None)
    if symmetry_gen is None:
        basis_template = cfg.symmetry_source_metadata.get("basis_template") if isinstance(cfg.symmetry_source_metadata, dict) else None
        symmetry_gen = SymmetryGenerator(np.asarray(cfg.Q_set1), np.asarray(cfg.Q_set2), [cfg.n_orb1, cfg.n_orb2], basis_template=basis_template)

    # 1) Validate `assemble_hamiltonian` path.
    H_full = model.assemble_hamiltonian(k0, symmetry_gen=symmetry_gen, use_cache=cfg.use_cache)
    herm_err_full = float(np.max(np.abs(H_full - H_full.conjugate().T)))
    w_full, _v_full = scipy.linalg.eigh(H_full, check_finite=False)

    logger.info(f"dim={H_full.shape[0]}")
    logger.info(f"Hermitian check: max|H-H†| = {herm_err_full:.3e}")
    logger.info(f"eigvals[:5] = {w_full[:5]}")

    # 2) Also run the same internal routine used by `compute_bands` (profiling-friendly).
    state = _prepare_band_state(cfg, model)
    _H_kept, _w_kept, _v_kept, _counts, prof = _compute_one_k(0, k0, state)
    if prof is not None:
        logger.info(f"profile: {prof}")

def main() -> None:
    """Entry point for a small, dependency-free sanity run."""
    self_test()
    cfg = example_config()              # 现在会优先用你真实的 mgi2_Gamma 文件
    model = build_model(cfg)

    heff_list = np.load("/Users/xtz/code/TAPW_tmdc/tMgI2/moirekp/kp/configs/mgi2_G/plots_mgi2_5_Gamma/heff_list.npy")
    k_proj = [0, 40]
    cfg.kpoints_fit = select_kpoints(cfg.kpoints, k_proj)
    cfg.heff = scipy.linalg.block_diag(*(heff_list[k_proj]))

    model, diag = compute_coefficients(cfg, model)
    eigvals = compute_bands(cfg, model, cfg.kpoints)   # 这里才是整条 kpath


if __name__ == "__main__":
    main()

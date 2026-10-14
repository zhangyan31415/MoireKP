import numpy as np
from pathlib import Path
from typing import List, Tuple


def format_kpoint_label(label: str) -> str:
    """格式化k点标签，处理特殊情况如G_M -> Γ_M或'Gamma Valley'"""
    # 处理常见的k点标签
    label_map = {
        'Gamma': r'{\Gamma}',
        'G': r'{\Gamma}',
        'K': r'{\mathrm{K}}',
        'M': r'{\mathrm{M}}',
        'X': r'{\mathrm{X}}',
        'Y': r'{\mathrm{Y}}'
    }
    # Sort keys by length, descending, to match 'Gamma' before 'G'
    sorted_keys = sorted(label_map.keys(), key=len, reverse=True)

    # 1. 处理带下划线的标签（例如 G_M -> Γ_M）
    if '_' in label:
        main_label, subscript = label.split('_', 1)
        main_label = label_map.get(main_label, r'{\mathrm{' + main_label + r'}}')
        return fr'${main_label}_\mathrm{{{subscript}}}$'

    # 2. 处理带空格的复杂标签 (例如 'Gamma Valley', "K' point")
    parts = label.split(' ')
    formatted_parts = []
    for part in parts:
        if not part: continue # Skip empty parts that might result from multiple spaces

        # 检查部分是否以已知的k点符号开头
        found_match = False
        for key in sorted_keys:
            if part.startswith(key):
                rest_of_part = part[len(key):]
                # '代表prime，在LaTeX中用'表示
                if rest_of_part == "'":
                    formatted_parts.append(f"{label_map[key]}'")
                else: # 其他情况，用\mathrm包裹
                    formatted_parts.append(f"{label_map[key]}" + (r"\mathrm{" + rest_of_part + "}" if rest_of_part else ""))
                found_match = True
                break

        if not found_match:
            # 如果没有匹配，则将整个部分视为普通文本
            formatted_parts.append(r'\mathrm{' + part + '}')

    return f'${" ".join(formatted_parts)}$'.replace(' ', r'\ ')


def parse_kpath_labels(lines: List[str]) -> Tuple[List[str], List[int]]:
    """
    解析VASPKIT格式的K路径文件

    Args:
        lines: KPATH.in文件的行列表

    Returns:
        Tuple[List[str], List[int]]:
            - 不重复的高对称点标签列表（如果同一位置有多个标签，用'|'连接）
            - 对应的tick索引位置
    """
    # 读取每段路径的k点数
    N = int(lines[1].strip())

    # 提取所有标签和坐标
    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 4:
            try:
                float(parts[0]), float(parts[1]), float(parts[2])
                labels.append(parts[3].strip())
            except ValueError:
                continue

    # 计算每个标签对应的tick位置
    raw_idx = []
    idx = 0
    for i in range(len(labels)):
        if (i + 1) > 1 and (i + 1) % 2 == 0:  # 偶数行，idx增加N
            idx += N
        raw_idx.append(idx)

    # 合并同一位置的标签
    ticks = []
    groups = {}
    for lab, t in zip(labels, raw_idx):
        if t not in groups:
            ticks.append(t)
            groups[t] = [lab]
        else:
            if lab not in groups[t]:
                groups[t].append(lab)

    # 生成最终标签列表，使用format_kpoint_label处理每个标签
    unique_labels = []
    for t in ticks:
        formatted_labels = [format_kpoint_label(label) for label in groups[t]]
        unique_labels.append('|'.join(formatted_labels))

    return unique_labels, ticks


def read_kpath(kpath_in: str, kpath_out: str) -> Tuple[List[str], np.ndarray, List[int]]:
    """
    读取KPATH.in和KPATH.out文件

    Args:
        kpath_in: KPATH.in文件路径，用于获取高对称点标签和位置
        kpath_out: KPATH.out文件路径，用于获取实际k点坐标

    Returns:
        Tuple[List[str], np.ndarray, List[int]]:
            - 格式化后的标签列表
            - 用于画图的x轴坐标
            - tick索引位置列表
    """
    # 读取KPATH.in获取高对称点和标签
    with open(kpath_in) as f:
        lines = f.readlines()
    labels, ticks_idx = parse_kpath_labels(lines)

    # 读取KPATH.out获取x坐标
    kpoints_data = np.loadtxt(kpath_out)
    x_coords = kpoints_data[:, 3]  # 第4列是用于画图的x坐标

    return labels, x_coords, ticks_idx


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
    except ValueError as e:
        raise ValueError(f"Failed to parse segment_points from line 2 of {file_path}: {lines[1]!r}") from e

    high_symmetry_points: list[np.ndarray] = []
    labels: list[str] = []
    for i in range(4, len(lines)):
        parts = lines[i].split()
        if len(parts) < 3:
            continue
        try:
            coordinates = np.array([float(coord) for coord in parts[:3]], dtype=float)
        except ValueError:
            continue
        if coordinates.shape != (3,):
            continue
        label = parts[-1] if parts else ""
        high_symmetry_points.append(coordinates)
        labels.append(label)

    if not high_symmetry_points:
        raise ValueError(f"No high-symmetry points found in {file_path}")

    return segment_points, np.array(high_symmetry_points, dtype=float), labels


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
    def _format_label(label):
        text = str(label)
        if text.upper() in ['GAMMA', '\\GAMMA', 'G', 'Γ']:
            return r'$\Gamma$'
        return text

    @staticmethod
    def calculate_reciprocal_vectors(Amat):
        a, b, c = Amat
        vol = np.dot(a, np.cross(b, c))
        astar = 2 * np.pi * np.cross(b, c) / vol
        bstar = 2 * np.pi * np.cross(c, a) / vol
        cstar = 2 * np.pi * np.cross(a, b) / vol
        return astar, bstar, cstar

    def read_and_generate_kpath(self, file_path: str, output_file_path: str) -> None:
        # 读取kpath文件
        with open(file_path, 'r') as input_file:
            lines = input_file.readlines()

        self.segment_points = int(lines[1])
        self.high_symmetry_points = []
        self.labels = []

        for i in range(4, len(lines)):
            coordinates = np.array([float(coord) for coord in lines[i].split()[:3]])
            if len(coordinates) == 3:
                label = lines[i].split()[-1]
                label = self._format_label(label)
                self.high_symmetry_points.append(coordinates)
                self.labels.append(label)

        self.high_symmetry_points = np.array(self.high_symmetry_points)

        # 生成kpath
        Amat_reciprocal = np.array([self.astar, self.bstar, self.cstar])
        x = 0.0
        num_high_symmetry_points = len(self.high_symmetry_points)
        self.x_ticks = []
        self.labels_ticks = []
        self.kpoints = []

        Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, 'w') as output_file:
            self.x_ticks.append(x)
            self.labels_ticks.append(self.labels[0])
            for i in range(int(num_high_symmetry_points / 2)):
                delta = self.distance(
                    self.direct_cart_real(Amat_reciprocal, self.high_symmetry_points[2 * i + 1]),
                    self.direct_cart_real(Amat_reciprocal, self.high_symmetry_points[2 * i])
                ) / self.segment_points

                for j in range(self.segment_points):
                    fraction = 1.0 * j / self.segment_points
                    interpolated_point = (1.0 - fraction) * self.high_symmetry_points[2 * i] + fraction * self.high_symmetry_points[2 * i + 1]

                    output_file.write(f'{interpolated_point[0]:>10.6f} {interpolated_point[1]:>10.6f} {interpolated_point[2]:>10.6f} {x:>10.6f}\n')
                    self.kpoints.append(np.append(interpolated_point, x))
                    x += delta

                if i < int(num_high_symmetry_points / 2) - 1 and self.labels[2 * i + 2] != self.labels[2 * i + 1]:
                    output_file.write(f'{self.high_symmetry_points[2 * i + 1][0]:>10.6f} {self.high_symmetry_points[2 * i + 1][1]:>10.6f} {self.high_symmetry_points[2 * i + 1][2]:>10.6f} {x:>10.6f}\n')
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
            output_file.write(f'{self.high_symmetry_points[2 * i + 1][0]:>10.6f} {self.high_symmetry_points[2 * i + 1][1]:>10.6f} {self.high_symmetry_points[2 * i + 1][2]:>10.6f} {x:>10.6f}\n')
            self.kpoints.append(np.append(self.high_symmetry_points[2 * i + 1], x))

    def generate_from_config(self, kpath_config, output_file_path: str) -> None:
        labels = [str(label) for label in kpath_config.get("labels", [])]
        coordinates = dict(kpath_config.get("coordinates", {}) or {})
        self.segment_points = int(kpath_config.get("points_per_segment", 40))
        if len(labels) < 2:
            raise ValueError("kpath.labels must contain at least two labels")
        if self.segment_points <= 0:
            raise ValueError("kpath.points_per_segment must be positive")

        points = []
        for label in labels:
            if label not in coordinates:
                raise ValueError(f"kpath.coordinates is missing label {label!r}")
            point = np.asarray(coordinates[label], dtype=float).ravel()
            if point.size == 2:
                point = np.array([point[0], point[1], 0.0], dtype=float)
            if point.size != 3:
                raise ValueError(f"kpath coordinate for {label!r} must have two or three values")
            points.append(point)
        self.high_symmetry_points = np.asarray(points, dtype=float)
        self.labels = [self._format_label(label) for label in labels]

        Amat_reciprocal = np.array([self.astar, self.bstar, self.cstar])
        x = 0.0
        self.x_ticks = [x]
        self.labels_ticks = [self.labels[0]]
        self.kpoints = []

        Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, 'w') as output_file:
            for iseg in range(len(points) - 1):
                start = self.high_symmetry_points[iseg]
                stop = self.high_symmetry_points[iseg + 1]
                delta = self.distance(
                    self.direct_cart_real(Amat_reciprocal, stop),
                    self.direct_cart_real(Amat_reciprocal, start),
                ) / self.segment_points
                for j in range(self.segment_points):
                    if iseg > 0 and j == 0:
                        # The shared endpoint was written by the previous segment.
                        # Advance to this segment's first new point.
                        x += delta
                        continue
                    fraction = float(j) / float(self.segment_points)
                    interpolated_point = (1.0 - fraction) * start + fraction * stop
                    output_file.write(
                        f'{interpolated_point[0]:>10.6f} {interpolated_point[1]:>10.6f} '
                        f'{interpolated_point[2]:>10.6f} {x:>10.6f}\n'
                    )
                    self.kpoints.append(np.append(interpolated_point, x))
                    x += delta
                output_file.write(f'{stop[0]:>10.6f} {stop[1]:>10.6f} {stop[2]:>10.6f} {x:>10.6f}\n')
                self.kpoints.append(np.append(stop, x))
                self.x_ticks.append(x)
                self.labels_ticks.append(self.labels[iseg + 1])

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

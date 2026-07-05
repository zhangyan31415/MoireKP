import numpy as np
from pathlib import Path

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

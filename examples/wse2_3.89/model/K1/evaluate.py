from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Run this standalone evaluator with:
#   python evaluate.py
#
# Edit HIGH_SYMMETRY_POINTS, KPATH, POINTS_PER_SEGMENT, WINDOW_BANDS,
# and WINDOW_EDGE to change the path or plotted energy window.
# Coordinates are fractional_model_basis.

# =========================
# User-editable settings
# =========================

HIGH_SYMMETRY_POINTS = {
    "G": [
        0.0,
        0.0
    ],
    "K": [
        0.33333333333685333,
        -0.6666666666731859
    ],
    "M": [
        5.0846766450304235e-12,
        -0.5000000000048896
    ]
}

KPATH = ['G', 'M', 'K', 'G']
POINTS_PER_SEGMENT = 10

# All bands are drawn. These settings only control the plotted y-range.
WINDOW_BANDS = None
WINDOW_EDGE = 'all'

BM1 = [0.1500040225446449, 0.0]
BM2 = [0.07500201127232248, 0.1299072941935161]

OUT_KPOINTS = "kpoints.npy"
OUT_BANDS = "bands.npy"
OUT_BAND_PLOT = "bands.pdf"
OUT_KDIST = "kdist.npy"
OUT_TICKS = "kpath_ticks.json"

# Standalone topology. Keep this off by default for quick band checks; set it
# to True and tune the grid/band set below when you want model BC/QGT/WCC.
RUN_BANDS = True
RUN_TOPOLOGY = False

TOPO_N_B1 = 31
TOPO_N_B2 = 31
TOPO_RANGE_B1 = (-0.5, 0.5)
TOPO_RANGE_B2 = (-0.5, 0.5)
TOPO_BAND_SETS = {
    "vbm2": {"sector": "valence", "indices": [-1, -2]},
}
CALC_BERRY_CURVATURE = True
CALC_QUANTUM_GEOMETRY = True
CALC_WCC = False
WCC_LOOP = "b2"
WCC_BOUNDARY_SINGULAR_VALUE_TOL = 1.0e-8
WCC_BOUNDARY_ISOMETRY_TOL = 1.0e-3
TOPOLOGY_OUTPUT_DIR = "outputs/topology"

# End user-editable settings


class StandaloneModel:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.data = np.load(self.root / "model_data.npz", allow_pickle=False)
        self.response_semantics = str(np.asarray(self.data["response_semantics"]).item())
        if self.response_semantics != "complete_linear":
            raise ValueError(f"unsupported response_semantics: {self.response_semantics!r}")
        self.dim = int(np.asarray(self.data["dimension_dim"]).item())
        self.hermitianize_before_eigvalsh = bool(
            np.asarray(self.data["runtime_hermitianize_before_eigvalsh"]).item()
        )
        self._required = [
            "basis_metadata_json",
            "basis_entry_channel",
            "basis_entry_monomial",
            "basis_entry_row",
            "basis_entry_col",
            "basis_entry_value",
            "fitted_coefficients",
        ]
        missing = [key for key in self._required if key not in self.data.files]
        if missing:
            raise KeyError(f"model_data.npz is missing complete-response keys: {missing}")
        metadata = json.loads(str(np.asarray(self.data["basis_metadata_json"]).item()))
        coordinate = metadata["coordinate"]
        self.response_origin = np.asarray(coordinate["origin"], dtype=float)
        self.response_scale = float(coordinate["scale"])
        self.response_monomials = np.asarray(coordinate["monomials"], dtype=np.int64)
        self.response_coefficients = np.asarray(self.data["fitted_coefficients"], dtype=float)
        entry_arrays = [
            np.asarray(self.data[name])
            for name in (
                "basis_entry_channel",
                "basis_entry_monomial",
                "basis_entry_row",
                "basis_entry_col",
                "basis_entry_value",
            )
        ]
        if len({array.size for array in entry_arrays}) != 1:
            raise ValueError("complete-response frozen entry arrays have inconsistent lengths")
        if self.response_origin.shape != (2,) or self.response_scale <= 0.0:
            raise ValueError("invalid complete-response polynomial coordinate metadata")
        return

    def hamiltonian(self, k):
        k = np.asarray(k, dtype=float)
        if k.shape != (2,):
            raise ValueError(f"k must have shape (2,), got {k.shape}")
        channel = np.asarray(self.data["basis_entry_channel"], dtype=np.int64)
        monomial_index = np.asarray(self.data["basis_entry_monomial"], dtype=np.int64)
        rows = np.asarray(self.data["basis_entry_row"], dtype=np.int64)
        cols = np.asarray(self.data["basis_entry_col"], dtype=np.int64)
        values = np.asarray(self.data["basis_entry_value"], dtype=np.complex128)
        exponents = self.response_monomials[monomial_index]
        u = (k - self.response_origin) / self.response_scale
        w = u[0] + 1j * u[1]
        monomial = (w ** exponents[:, 0]) * (np.conjugate(w) ** exponents[:, 1])
        h = np.zeros((self.dim, self.dim), dtype=np.complex128)
        np.add.at(
            h,
            (rows, cols),
            self.response_coefficients[channel] * values * monomial,
        )
        return h

    def _hamiltonian_for_eigvalsh(self, k):
        h = self.hamiltonian(k)
        if self.hermitianize_before_eigvalsh:
            return 0.5 * (h + h.conj().T)
        return h

    def bands(self, kpoints, band_slice=None):
        kpoints = np.asarray(kpoints, dtype=float)
        if kpoints.ndim != 2 or kpoints.shape[1] != 2:
            raise ValueError(f"kpoints must have shape (Nk, 2), got {kpoints.shape}")
        out = np.empty((kpoints.shape[0], self.dim), dtype=float)
        for i, k in enumerate(kpoints):
            out[i] = np.linalg.eigvalsh(self._hamiltonian_for_eigvalsh(k))
        if band_slice is not None:
            if len(band_slice) != 2:
                raise ValueError(f"band_slice must be [start, stop], got {band_slice!r}")
            out = out[:, int(band_slice[0]):int(band_slice[1])]
        return out

    def eigensystem(self, k):
        h = self._hamiltonian_for_eigvalsh(k)
        return np.linalg.eigh(h)


def load_model(root="."):
    return StandaloneModel(root)


def _resolve_path(root, path):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(root).resolve() / path


def _load_kpoints(path, root):
    arr = np.load(_resolve_path(root, path), allow_pickle=False)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"kpoints must have shape (Nk, 2), got {arr.shape}")
    return arr


def _generate_kpath():
    if len(KPATH) < 2:
        raise ValueError("KPATH must contain at least two labels")
    b1 = np.asarray(BM1, dtype=float)
    b2 = np.asarray(BM2, dtype=float)
    if b1.shape != (2,) or b2.shape != (2,):
        raise ValueError("BM1 and BM2 must each contain two numbers")
    points = {label: np.asarray(value, dtype=float) for label, value in HIGH_SYMMETRY_POINTS.items()}
    missing = [label for label in KPATH if label not in points]
    if missing:
        raise ValueError(f"KPATH contains labels missing from HIGH_SYMMETRY_POINTS: {missing}")
    kpoints = []
    kdist = []
    tick_positions = [0.0]
    distance = 0.0
    for iseg, (start_label, stop_label) in enumerate(zip(KPATH[:-1], KPATH[1:])):
        start = points[start_label]
        stop = points[stop_label]
        if start.shape != (2,) or stop.shape != (2,):
            raise ValueError("high-symmetry points must be two-component coordinates")
        n = int(POINTS_PER_SEGMENT)
        if n <= 0:
            raise ValueError("POINTS_PER_SEGMENT must be positive")
        for i in range(n):
            if iseg > 0 and i == 0:
                continue
            t = i / float(n)
            k = (1.0 - t) * start + t * stop
            if kpoints:
                prev = kpoints[-1]
                dk = k - prev
                distance += float(np.linalg.norm(dk[0] * b1 + dk[1] * b2))
            kpoints.append(k)
            kdist.append(distance)
        if not np.allclose(kpoints[-1], stop):
            dk = stop - kpoints[-1]
            distance += float(np.linalg.norm(dk[0] * b1 + dk[1] * b2))
            kpoints.append(stop)
            kdist.append(distance)
        tick_positions.append(distance)
    return np.asarray(kpoints, dtype=float), np.asarray(kdist, dtype=float), {"labels": list(KPATH), "positions": tick_positions}


def _kdist_from_kpoints(kpoints):
    kpoints = np.asarray(kpoints, dtype=float)
    if kpoints.ndim != 2 or kpoints.shape[1] != 2:
        raise ValueError(f"kpoints must have shape (Nk, 2), got {kpoints.shape}")
    kdist = np.zeros(kpoints.shape[0], dtype=float)
    for i in range(1, kpoints.shape[0]):
        kdist[i] = kdist[i - 1] + float(np.linalg.norm(kpoints[i] - kpoints[i - 1]))
    return kdist


def _ticks_for_reference_kpath(kdist):
    segments = len(KPATH) - 1
    if segments <= 0:
        return {"labels": [], "positions": []}
    expected = segments * int(POINTS_PER_SEGMENT) + 1
    if int(kdist.shape[0]) != expected:
        return {"labels": [], "positions": []}
    indices = [i * int(POINTS_PER_SEGMENT) for i in range(segments + 1)]
    return {"labels": list(KPATH), "positions": [float(kdist[i]) for i in indices]}


def _band_window(dim):
    if WINDOW_BANDS is not None:
        count = max(1, min(int(WINDOW_BANDS), int(dim)))
        edge = str(WINDOW_EDGE).strip().lower()
        if edge == "bottom":
            start, stop = 0, count
        elif edge in {"top", "all"}:
            start, stop = int(dim) - count, int(dim)
        else:
            raise ValueError(f"WINDOW_EDGE must be 'top', 'bottom', or 'all', got {WINDOW_EDGE!r}")
    else:
        start, stop = 0, int(dim)
    if start < 0 or stop > dim or start >= stop:
        raise ValueError(f"Invalid band window [{start}, {stop}] for dim={dim}")
    return start, stop


def _plot_bands(kdist, bands, ticks, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipped band plot")
        return None

    bands = np.asarray(bands, dtype=float)
    start, stop = _band_window(bands.shape[1])
    window = bands[:, start:stop]
    edge = str(WINDOW_EDGE).strip().lower()
    if edge == "bottom":
        reference = float(np.min(window[:, 0]))
        ylabel = "Energy - E_bottom (eV)"
    elif edge in {"top", "slice"}:
        reference = float(np.max(window[:, -1]))
        ylabel = "Energy - E_top (eV)"
    else:
        reference = 0.0
        ylabel = "Energy (eV)"
    shifted = bands - reference
    window_shifted = window - reference
    fig, ax = plt.subplots(figsize=(3.0, 5.0))
    for iband in range(shifted.shape[1]):
        ax.plot(kdist, shifted[:, iband], color="#1f77b4", linewidth=1.05, marker="o", markersize=2.3)
    if ticks.get("positions") and ticks.get("labels"):
        ax.set_xticks([float(item) for item in ticks["positions"]])
        ax.set_xticklabels([str(item) for item in ticks["labels"]])
        for tick in ticks["positions"]:
            ax.axvline(float(tick), color="0.86", linewidth=0.75, zorder=0)
    else:
        ax.set_xlabel("k-point index")
    ax.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--", alpha=0.8, zorder=0)
    ax.set_ylabel(ylabel)
    ymin = float(np.min(window_shifted))
    ymax = float(np.max(window_shifted))
    pad = max(0.004, 0.08 * (ymax - ymin if ymax > ymin else 1.0))
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(axis="y", color="0.88", linewidth=0.65)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _float_token(value):
    text = f"{float(value):.6g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


def _topology_grid_id():
    return (
        f"grid{int(TOPO_N_B1)}x{int(TOPO_N_B2)}"
        f"_b1_{_float_token(TOPO_RANGE_B1[0])}_{_float_token(TOPO_RANGE_B1[1])}"
        f"_b2_{_float_token(TOPO_RANGE_B2[0])}_{_float_token(TOPO_RANGE_B2[1])}"
    )


def _model_reciprocal_basis(model):
    if "model_reciprocal_basis" not in model.data.files:
        raise KeyError(
            "Topology requires model_reciprocal_basis in model_data.npz; "
            "regenerate this standalone package with kp model"
        )
    reciprocal_basis = np.asarray(model.data["model_reciprocal_basis"], dtype=float)
    if reciprocal_basis.shape != (2, 2):
        raise ValueError(
            f"model_reciprocal_basis must have shape (2, 2), got {reciprocal_basis.shape}"
        )
    determinant = float(np.linalg.det(reciprocal_basis))
    if not np.all(np.isfinite(reciprocal_basis)) or abs(determinant) < 1.0e-12:
        raise ValueError("model_reciprocal_basis must be finite and nonsingular")
    return reciprocal_basis


def _topology_mesh(model):
    n1 = int(TOPO_N_B1)
    n2 = int(TOPO_N_B2)
    if n1 < 2 or n2 < 2:
        raise ValueError("TOPO_N_B1 and TOPO_N_B2 must be at least 2")
    b1_values = np.linspace(float(TOPO_RANGE_B1[0]), float(TOPO_RANGE_B1[1]), n1)
    b2_values = np.linspace(float(TOPO_RANGE_B2[0]), float(TOPO_RANGE_B2[1]), n2)
    fractional_points = np.asarray([[x, y] for x in b1_values for y in b2_values], dtype=float)
    points = fractional_points @ _model_reciprocal_basis(model)
    return b1_values, b2_values, points


def _resolve_topology_band_indices(dim, spec):
    if not isinstance(spec, dict):
        raise ValueError(f"band-set spec must be a dict, got {spec!r}")
    sector = str(spec.get("sector", "valence")).strip().lower()
    raw_indices = list(spec.get("indices", []))
    if not raw_indices:
        raise ValueError("topology band-set indices cannot be empty")
    resolved = []
    for raw in raw_indices:
        idx = int(raw)
        if sector == "valence":
            resolved_idx = dim + idx if idx < 0 else idx
        elif sector == "conduction":
            resolved_idx = idx if idx >= 0 else dim + idx
        else:
            raise ValueError(f"Unsupported topology sector {sector!r}")
        if resolved_idx < 0 or resolved_idx >= dim:
            raise ValueError(f"Resolved band index {resolved_idx} is outside 0..{dim - 1}")
        resolved.append(resolved_idx)
    return sorted(set(resolved))


def _eigensystem_on_grid(model, points, n1, n2):
    eigvals = np.empty((n1, n2, model.dim), dtype=float)
    eigvecs = np.empty((n1, n2, model.dim, model.dim), dtype=np.complex128)
    for flat_index, k in enumerate(points):
        i = flat_index // n2
        j = flat_index % n2
        vals, vecs = model.eigensystem(k)
        eigvals[i, j] = vals
        eigvecs[i, j] = vecs
    return eigvals, eigvecs


def _projectors_for_bandset(eigvecs, band_indices):
    n1, n2, dim, _ = eigvecs.shape
    projectors = np.empty((n1, n2, dim, dim), dtype=np.complex128)
    for i in range(n1):
        for j in range(n2):
            u = eigvecs[i, j][:, band_indices]
            projectors[i, j] = u @ u.conj().T
    return projectors


def _finite_difference(arr, values, axis, index):
    i, j = index
    if axis == 0:
        if i == 0:
            return (arr[1, j] - arr[0, j]) / float(values[1] - values[0])
        if i == arr.shape[0] - 1:
            return (arr[i, j] - arr[i - 1, j]) / float(values[i] - values[i - 1])
        return (arr[i + 1, j] - arr[i - 1, j]) / float(values[i + 1] - values[i - 1])
    if j == 0:
        return (arr[i, 1] - arr[i, 0]) / float(values[1] - values[0])
    if j == arr.shape[1] - 1:
        return (arr[i, j] - arr[i, j - 1]) / float(values[j] - values[j - 1])
    return (arr[i, j + 1] - arr[i, j - 1]) / float(values[j + 1] - values[j - 1])


def _berry_and_qgt_from_projectors(projectors, b1_values, b2_values):
    n1, n2 = projectors.shape[:2]
    berry = np.empty((n1, n2), dtype=float)
    qgt = np.empty((n1, n2, 4), dtype=float)
    for i in range(n1):
        for j in range(n2):
            p = projectors[i, j]
            dp1 = _finite_difference(projectors, b1_values, 0, (i, j))
            dp2 = _finite_difference(projectors, b2_values, 1, (i, j))
            berry[i, j] = float(np.real(1j * np.trace(p @ (dp1 @ dp2 - dp2 @ dp1))))
            g11 = float(0.5 * np.real(np.trace(dp1 @ dp1)))
            g22 = float(0.5 * np.real(np.trace(dp2 @ dp2)))
            g12 = float(0.5 * np.real(np.trace(dp1 @ dp2)))
            qgt[i, j] = [g11 + g22, g11, g22, g12]
    return berry, qgt


def _geometry_to_cartesian(berry_fractional, qgt_fractional, reciprocal_basis):
    reciprocal_basis = np.asarray(reciprocal_basis, dtype=float)
    if reciprocal_basis.shape != (2, 2):
        raise ValueError(
            f"reciprocal_basis must have shape (2, 2), got {reciprocal_basis.shape}"
        )
    determinant = float(np.linalg.det(reciprocal_basis))
    if not np.all(np.isfinite(reciprocal_basis)) or abs(determinant) < 1.0e-12:
        raise ValueError("reciprocal_basis must be finite and nonsingular")

    berry_cartesian = np.asarray(berry_fractional, dtype=float) / determinant
    qgt_fractional = np.asarray(qgt_fractional, dtype=float)
    inverse_basis = np.linalg.inv(reciprocal_basis)
    metric_fractional = np.empty(qgt_fractional.shape[:-1] + (2, 2), dtype=float)
    metric_fractional[..., 0, 0] = qgt_fractional[..., 1]
    metric_fractional[..., 1, 1] = qgt_fractional[..., 2]
    metric_fractional[..., 0, 1] = qgt_fractional[..., 3]
    metric_fractional[..., 1, 0] = qgt_fractional[..., 3]
    metric_cartesian = np.einsum(
        "ab,...bc,dc->...ad",
        inverse_basis,
        metric_fractional,
        inverse_basis,
        optimize=True,
    )
    qgt_cartesian = np.empty_like(qgt_fractional)
    qgt_cartesian[..., 0] = np.trace(metric_cartesian, axis1=-2, axis2=-1)
    qgt_cartesian[..., 1] = metric_cartesian[..., 0, 0]
    qgt_cartesian[..., 2] = metric_cartesian[..., 1, 1]
    qgt_cartesian[..., 3] = metric_cartesian[..., 0, 1]
    return berry_cartesian, qgt_cartesian


def _save_grid_table(path, b1_values, b2_values, values, columns):
    rows = []
    for i, x in enumerate(b1_values):
        for j, y in enumerate(b2_values):
            value = np.asarray(values[i, j], dtype=float).ravel()
            rows.append([float(x), float(y), *[float(v) for v in value]])
    header = "kappa1 kappa2 " + " ".join(columns)
    np.savetxt(path, np.asarray(rows, dtype=float), header=header)


def _plot_grid_scalar(path, b1_values, b2_values, values, colorbar_label):
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(f"matplotlib is not installed; skipped {path.name}")
        return None

    arr = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    image = ax.imshow(
        arr.T,
        origin="lower",
        extent=[float(b1_values[0]), float(b1_values[-1]), float(b2_values[0]), float(b2_values[-1])],
        aspect="equal",
        interpolation="bilinear",
    )
    ax.set_xlabel(r"$k_1/|b_M|$")
    ax.set_ylabel(r"$k_2/|b_M|$")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _sewing_permutation(model, shift):
    required = ["basis_qset_id", "basis_q_index", "basis_orbital", "basis_q_vector"]
    missing = [key for key in required if key not in model.data.files]
    if missing:
        raise KeyError(
            "WCC requires basis metadata in model_data.npz; regenerate this standalone package with a newer kp model export. "
            f"Missing: {missing}"
        )
    qset_id = np.asarray(model.data["basis_qset_id"], dtype=np.int64)
    orbital = np.asarray(model.data["basis_orbital"], dtype=np.int64)
    qvec = np.asarray(model.data["basis_q_vector"], dtype=float)
    shift = np.asarray(shift, dtype=float)
    target_by_source = np.full(model.dim, -1, dtype=np.int64)
    used = set()
    for source in range(model.dim):
        wanted = qvec[source] + shift
        matches = np.where(
            (qset_id == qset_id[source])
            & (orbital == orbital[source])
            & (np.linalg.norm(qvec - wanted[None, :], axis=1) < 1.0e-7)
        )[0]
        if matches.size == 0:
            continue
        if matches.size != 1:
            raise ValueError(
                f"WCC boundary sewing is ambiguous for row {source} with shift={shift.tolist()}"
            )
        target = int(matches[0])
        if target in used:
            raise ValueError("WCC boundary sewing is not one-to-one")
        used.add(target)
        target_by_source[source] = target
    return target_by_source


def _apply_sewing(matrix, target_by_source):
    out = np.zeros_like(matrix)
    supported = np.asarray(target_by_source) >= 0
    out[np.asarray(target_by_source)[supported], :] = matrix[supported, :]
    return out


def _wilson_link(overlap, *, boundary=False):
    overlap = np.asarray(overlap, dtype=np.complex128)
    if not np.isfinite(overlap).all():
        raise ValueError("WCC overlap contains nonfinite values")
    left, singular_values, right_h = np.linalg.svd(overlap, full_matrices=False)
    minimum = float(np.min(singular_values)) if singular_values.size else 0.0
    if minimum <= 0 or minimum < float(WCC_BOUNDARY_SINGULAR_VALUE_TOL):
        label = "boundary link" if boundary else "Wilson link"
        raise ValueError(
            f"WCC {label} is rank deficient: minimum singular value={minimum:.6e}, "
            f"required>={float(WCC_BOUNDARY_SINGULAR_VALUE_TOL):.6e}"
        )
    if boundary:
        error = float(np.max(np.abs(singular_values - 1.0)))
        if error > float(WCC_BOUNDARY_ISOMETRY_TOL):
            raise ValueError(
                "WCC boundary link is not isometric in the selected band subspace: "
                f"max|sigma-1|={error:.6e}, required<={float(WCC_BOUNDARY_ISOMETRY_TOL):.6e}. "
                "Increase the Q cutoff or revise the selected band subspace."
            )
    return left @ right_h


def _wilson_wcc(model, eigvecs, band_indices, loop):
    loop = str(loop).strip().lower()
    if loop not in {"b1", "b2"}:
        raise ValueError(f"WCC_LOOP must be 'b1' or 'b2', got {loop!r}")
    for tolerance in (WCC_BOUNDARY_SINGULAR_VALUE_TOL, WCC_BOUNDARY_ISOMETRY_TOL):
        if not np.isfinite(tolerance) or float(tolerance) <= 0:
            raise ValueError("WCC link tolerances must be finite and positive")
    axis = 0 if loop == "b1" else 1
    loop_range = TOPO_RANGE_B1 if axis == 0 else TOPO_RANGE_B2
    span = float(loop_range[1]) - float(loop_range[0])
    if not np.isfinite(span) or span == 0 or not np.isclose(span, np.rint(span), rtol=0, atol=1.0e-12):
        raise ValueError("WCC loop endpoints must differ by a nonzero integer reciprocal period")
    reciprocal_basis = _model_reciprocal_basis(model)
    # H uses k-Q: an endpoint channel Q maps to Q-(k_end-k_start).
    target_by_source = _sewing_permutation(model, -span * reciprocal_basis[axis])
    paths = np.moveaxis(np.asarray(eigvecs), axis, 0)
    if paths.shape[0] < 2:
        raise ValueError("WCC requires at least two loop vertices")
    branches = []
    for fixed in range(paths.shape[1]):
        frames = paths[:, fixed, :, :][..., band_indices]
        product = np.eye(len(band_indices), dtype=np.complex128)
        for index in range(len(frames) - 1):
            overlap = frames[index + 1].conj().T @ frames[index]
            product = _wilson_link(overlap) @ product
        endpoint = _apply_sewing(frames[-1], target_by_source)
        closing_overlap = frames[0].conj().T @ endpoint
        product = _wilson_link(closing_overlap, boundary=True) @ product
        phases = np.sort((np.angle(np.linalg.eigvals(product)) / (2.0 * np.pi)) % 1.0)
        branches.append(phases)
    return np.asarray(branches, dtype=float)


def _plot_wcc(path, sweep_values, branches):
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(f"matplotlib is not installed; skipped {path.name}")
        return None

    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    for branch in range(branches.shape[1]):
        ax.scatter(sweep_values, branches[:, branch], s=8, color="#4a4f55")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$k/|b_M|$")
    ax.set_ylabel(r"$\theta/2\pi$")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _run_topology(root, model):
    reciprocal_basis = _model_reciprocal_basis(model)
    b1_values, b2_values, points = _topology_mesh(model)
    grid_dir = _resolve_path(root, Path(TOPOLOGY_OUTPUT_DIR) / _topology_grid_id())
    grid_dir.mkdir(parents=True, exist_ok=True)
    eigvals, eigvecs = _eigensystem_on_grid(model, points, int(TOPO_N_B1), int(TOPO_N_B2))
    np.save(grid_dir / "eigvals.npy", eigvals.reshape((-1, model.dim)))
    for name, spec in TOPO_BAND_SETS.items():
        band_indices = _resolve_topology_band_indices(model.dim, spec)
        projectors = _projectors_for_bandset(eigvecs, band_indices)
        berry_fractional, qgt_fractional = _berry_and_qgt_from_projectors(
            projectors,
            b1_values,
            b2_values,
        )
        berry, qgt = _geometry_to_cartesian(
            berry_fractional,
            qgt_fractional,
            reciprocal_basis,
        )
        if CALC_BERRY_CURVATURE:
            txt = grid_dir / f"berry_curvature_{name}.txt"
            _save_grid_table(txt, b1_values, b2_values, berry[:, :, None], ["omega"])
            _plot_grid_scalar(grid_dir / f"berry_curvature_{name}.pdf", b1_values, b2_values, berry, r"$\Omega$")
        if CALC_QUANTUM_GEOMETRY:
            txt = grid_dir / f"quantum_geometry_{name}.txt"
            _save_grid_table(txt, b1_values, b2_values, qgt, ["trace_g", "g11", "g22", "g12"])
            _plot_grid_scalar(grid_dir / f"quantum_geometry_{name}.pdf", b1_values, b2_values, qgt[:, :, 0], r"$\mathrm{Tr}\,g$")
        if CALC_WCC:
            branches = _wilson_wcc(model, eigvecs, band_indices, WCC_LOOP)
            sweep_values = b2_values if str(WCC_LOOP).lower() == "b1" else b1_values
            txt = grid_dir / f"wcc_{name}_loop_{str(WCC_LOOP).lower()}.txt"
            np.savetxt(
                txt,
                np.column_stack([sweep_values, branches]),
                header="sweep " + " ".join(f"branch{i}" for i in range(branches.shape[1])),
            )
            _plot_wcc(grid_dir / f"wcc_{name}_loop_{str(WCC_LOOP).lower()}.pdf", sweep_values, branches)
    print(f"wrote topology outputs: {grid_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate standalone NumPy continuum model bands.")
    parser.add_argument("--model-root", default=".", help="Standalone package root")
    parser.add_argument("--kpoints", default=None, help="Optional .npy kpoints file with shape (Nk, 2)")
    parser.add_argument("--out", default=None, help="Output .npy file for computed bands")
    args = parser.parse_args(argv)

    root = Path(args.model_root).resolve()
    model = load_model(root)
    if RUN_BANDS:
        if args.kpoints is None:
            if "reference_kpoints" in model.data.files:
                kpoints = np.asarray(model.data["reference_kpoints"], dtype=float)
                kdist = _kdist_from_kpoints(kpoints)
                ticks = _ticks_for_reference_kpath(kdist)
            else:
                kpoints, kdist, ticks = _generate_kpath()
        else:
            kpoints = _load_kpoints(args.kpoints, root)
            kdist = _kdist_from_kpoints(kpoints)
            ticks = {"labels": [], "positions": []}
        bands = model.bands(kpoints, band_slice=None)
        np.save(_resolve_path(root, OUT_KPOINTS), kpoints)
        np.save(_resolve_path(root, args.out or OUT_BANDS), bands)
        np.save(_resolve_path(root, OUT_KDIST), kdist)
        with _resolve_path(root, OUT_TICKS).open("w", encoding="utf-8") as handle:
            json.dump(ticks, handle, indent=2, allow_nan=False)
            handle.write("\n")
        plot_path = _plot_bands(kdist, bands, ticks, _resolve_path(root, OUT_BAND_PLOT))
        print(f"wrote {kpoints.shape[0]} k-points and {bands.shape[1]} bands")
        if plot_path is not None:
            print(f"wrote band plot: {plot_path}")
    if RUN_TOPOLOGY:
        _run_topology(root, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

from .artifacts import array_output_filename
from .config import format_chern_grid_suffix, resolve_chern_grid_shape


plt.rc("font", family="Times New Roman")
plt.rc("mathtext", fontset="stix")


VALLEY_MAP = {
    1: "K1",
    2: "K2",
    5: "Gamma",
    11: "K1_120",
    12: "K1_240",
    31: "M1",
    32: "M2",
    33: "M3",
}

_DEFAULT_SAVEFIG_KWARGS = {
    "dpi": 300,
    "transparent": False,
    "facecolor": "white",
    "edgecolor": "none",
}


def load_config(config_path):
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)


def build_fractional_axis(num_k):
    """Fractional reciprocal coordinate along one moire reciprocal direction."""
    return np.linspace(-0.5, 0.5, int(num_k), endpoint=True)


def build_fractional_vertex_mesh(num_k1, num_k2):
    """Return the fractional vertex mesh `(kappa1, kappa2)` with `indexing='ij'`."""
    kappa1_values = build_fractional_axis(num_k1)
    kappa2_values = build_fractional_axis(num_k2)
    kappa1_mesh, kappa2_mesh = np.meshgrid(kappa1_values, kappa2_values, indexing="ij")
    return np.stack((kappa1_mesh, kappa2_mesh), axis=-1)


def build_fractional_plaquette_center_mesh(num_k1, num_k2):
    """Return plaquette centers in fractional coordinates for Berry-flux data."""
    vertex_mesh = build_fractional_vertex_mesh(num_k1, num_k2)
    return 0.25 * (
        vertex_mesh[:-1, :-1]
        + vertex_mesh[1:, :-1]
        + vertex_mesh[:-1, 1:]
        + vertex_mesh[1:, 1:]
    )


def build_fractional_interior_mesh(num_k1, num_k2):
    """Return the interior fractional mesh used by centered-difference QGT fields."""
    return build_fractional_vertex_mesh(num_k1, num_k2)[1:-1, 1:-1]


def build_axis_edges_from_centers(center_values):
    """Build cell edges for a uniformly spaced 1D center grid."""
    center_values = np.asarray(center_values, dtype=float)
    if center_values.ndim != 1 or center_values.size < 1:
        raise ValueError("center_values must be a 1D array with at least one entry.")
    if center_values.size == 1:
        half_step = 0.5
        return np.array([center_values[0] - half_step, center_values[0] + half_step], dtype=float)

    edges = np.empty(center_values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (center_values[:-1] + center_values[1:])
    edges[0] = center_values[0] - 0.5 * (center_values[1] - center_values[0])
    edges[-1] = center_values[-1] + 0.5 * (center_values[-1] - center_values[-2])
    return edges


def build_fractional_edge_mesh_from_centers(center_mesh):
    """Convert a regular center mesh `(Nk1, Nk2, 2)` into a cell-edge mesh."""
    kappa1_centers = np.asarray(center_mesh[:, 0, 0], dtype=float)
    kappa2_centers = np.asarray(center_mesh[0, :, 1], dtype=float)
    kappa1_edges = build_axis_edges_from_centers(kappa1_centers)
    kappa2_edges = build_axis_edges_from_centers(kappa2_centers)
    kappa1_edge_mesh, kappa2_edge_mesh = np.meshgrid(kappa1_edges, kappa2_edges, indexing="ij")
    return np.stack((kappa1_edge_mesh, kappa2_edge_mesh), axis=-1)


def flatten_coordinate_mesh(coordinate_mesh):
    """Flatten a `(Nk1, Nk2, 2)` mesh in the same order as `field.flatten()`."""
    return np.ascontiguousarray(coordinate_mesh).reshape(-1, coordinate_mesh.shape[-1])


def fractional_mesh_to_cartesian(coordinate_mesh, basis_2d):
    """Convert fractional coordinates `kappa @ B` to Cartesian coordinates."""
    flat = flatten_coordinate_mesh(coordinate_mesh)
    cart_flat = np.dot(flat, basis_2d)
    return cart_flat.reshape(coordinate_mesh.shape[0], coordinate_mesh.shape[1], 2)


def build_plot_basis(b_phys_2d):
    """Normalized reciprocal basis used only for plotting axes."""
    scale = float(np.linalg.norm(b_phys_2d[0]))
    if np.isclose(scale, 0.0):
        scale = 1.0
    return b_phys_2d / scale


def flatten_output_coordinate_meshes(fractional_mesh, cartesian_mesh, plot_mesh):
    """Return flattened fractional / physical-cartesian / plot-normalized coordinates."""
    frac_points = flatten_coordinate_mesh(fractional_mesh)
    cart_points = flatten_coordinate_mesh(cartesian_mesh)
    # `plot_points` share the figure-axis normalization: k / |b1|.
    plot_points = flatten_coordinate_mesh(plot_mesh)
    return frac_points, cart_points, plot_points


def compute_fractional_spacings(num_k1, num_k2):
    """Return `(delta_kappa1, delta_kappa2, axis1, axis2)` for the stored vertex grid."""
    axis1 = build_fractional_axis(num_k1)
    axis2 = build_fractional_axis(num_k2)
    if len(axis1) < 2 or len(axis2) < 2:
        raise ValueError("Need at least 2 grid points along each fractional axis.")
    return float(axis1[1] - axis1[0]), float(axis2[1] - axis2[0]), axis1, axis2


def reshape_wavefunction_grid(band_vec, num_k1, num_k2):
    """Reshape the flattened TAPW eigenvector array onto a rectangular vertex grid."""
    expected = int(num_k1) * int(num_k2)
    if band_vec.shape[0] != expected:
        raise ValueError(
            "Wavefunction file contains {0} k-points, but {1} were expected for a {2}x{3} grid.".format(
                band_vec.shape[0],
                expected,
                num_k1,
                num_k2,
            )
        )
    return band_vec.reshape(int(num_k1), int(num_k2), band_vec.shape[-2], band_vec.shape[-1])


def _phase_from_overlap(overlap, eps=1e-14):
    phase = np.ones_like(overlap, dtype=np.complex128)
    mask = np.abs(overlap) >= eps
    phase[mask] = overlap[mask] / np.abs(overlap[mask])
    return phase


def compute_berry_flux_single_band(band_grid, band_index, eps=1e-14):
    """FHS plaquette phase for a single band on a rectangular vertex grid."""
    band_slice = band_grid[:, :, :, band_index]
    v_k = band_slice[:-1, :-1]
    v_k1 = band_slice[1:, :-1]
    v_k2 = band_slice[:-1, 1:]
    v_k1k2 = band_slice[1:, 1:]

    overlap_k1 = np.sum(np.conj(v_k) * v_k1, axis=-1)
    overlap_k2 = np.sum(np.conj(v_k) * v_k2, axis=-1)
    overlap_k1_at_k2 = np.sum(np.conj(v_k2) * v_k1k2, axis=-1)
    overlap_k2_at_k1 = np.sum(np.conj(v_k1) * v_k1k2, axis=-1)

    ux = _phase_from_overlap(overlap_k1, eps=eps)
    uy = _phase_from_overlap(overlap_k2, eps=eps)
    ux_at_k2 = _phase_from_overlap(overlap_k1_at_k2, eps=eps)
    uy_at_k1 = _phase_from_overlap(overlap_k2_at_k1, eps=eps)

    plaquette = ux * uy_at_k1 / (ux_at_k2 * uy)
    return np.angle(plaquette)


def compute_berry_flux_multiband(band_grid, band_indices, eps=1e-14):
    """FHS plaquette phase for an occupied subspace on a rectangular vertex grid."""
    band_sub = band_grid[:, :, :, band_indices]
    v_k = band_sub[:-1, :-1]
    v_k1 = band_sub[1:, :-1]
    v_k2 = band_sub[:-1, 1:]
    v_k1k2 = band_sub[1:, 1:]

    def overlap(v1, v2):
        return np.einsum("...ia,...ib->...ab", np.conj(v1), v2)

    det_k1 = np.linalg.det(overlap(v_k, v_k1))
    det_k2 = np.linalg.det(overlap(v_k, v_k2))
    det_k1_at_k2 = np.linalg.det(overlap(v_k2, v_k1k2))
    det_k2_at_k1 = np.linalg.det(overlap(v_k1, v_k1k2))

    ux = _phase_from_overlap(det_k1, eps=eps)
    uy = _phase_from_overlap(det_k2, eps=eps)
    ux_at_k2 = _phase_from_overlap(det_k1_at_k2, eps=eps)
    uy_at_k1 = _phase_from_overlap(det_k2_at_k1, eps=eps)

    plaquette = ux * uy_at_k1 / (ux_at_k2 * uy)
    return np.angle(plaquette)


def berry_flux_to_cartesian_density(berry_flux, b_phys_2d, delta_kappa1, delta_kappa2):
    """Convert plaquette Berry flux to Cartesian Berry-curvature density."""
    area = float(np.linalg.det(b_phys_2d)) * float(delta_kappa1) * float(delta_kappa2)
    if np.isclose(area, 0.0):
        raise ValueError("det(B_phys_2d) * delta_kappa1 * delta_kappa2 is zero.")
    return berry_flux / area


def _projector_from_subspace(subspace):
    return np.einsum("...ma,...na->...mn", subspace, np.conj(subspace))


def _trace_two_projectors(subspace_left, subspace_right):
    overlap = np.conj(subspace_left.T) @ subspace_right
    return np.trace(overlap @ np.conj(overlap.T))


def _trace_three_projectors(subspace_a, subspace_b, subspace_c):
    overlap_ab = np.conj(subspace_a.T) @ subspace_b
    overlap_bc = np.conj(subspace_b.T) @ subspace_c
    overlap_ca = np.conj(subspace_c.T) @ subspace_a
    return np.trace(overlap_ab @ overlap_bc @ overlap_ca)


def compute_qgt_fields(band_grid, band_indices, delta_kappa1, delta_kappa2):
    r"""Compute the rigorous fractional-coordinate QGT without materializing a full projector grid.

    Instead of building dense `dim_h x dim_h` projector matrices on the full grid,
    this expands the finite-difference projector formula into traces of two- and
    three-projector products:

        Tr[P_a P_b] = Tr[(V_a^\dagger V_b)(V_b^\dagger V_a)]
        Tr[P_a P_b P_c] = Tr[(V_a^\dagger V_b)(V_b^\dagger V_c)(V_c^\dagger V_a)]

    so the calculation stays exact while only manipulating `n_occ x n_occ`
    overlap matrices.
    """
    if not band_indices:
        raise ValueError("band_indices must contain at least one band.")
    if band_grid.shape[0] < 3 or band_grid.shape[1] < 3:
        raise ValueError("QGT evaluation requires at least a 3x3 vertex grid for centered differences.")

    subspace_grid = np.asarray(band_grid[:, :, :, band_indices], dtype=np.complex128)
    num_occ = subspace_grid.shape[-1]
    num_i = subspace_grid.shape[0] - 2
    num_j = subspace_grid.shape[1] - 2
    g11_frac = np.empty((num_i, num_j), dtype=np.float64)
    g12_frac = np.empty((num_i, num_j), dtype=np.float64)
    g22_frac = np.empty((num_i, num_j), dtype=np.float64)
    omega12_frac = np.empty((num_i, num_j), dtype=np.float64)

    for i in range(1, subspace_grid.shape[0] - 1):
        for j in range(1, subspace_grid.shape[1] - 1):
            center_subspace = subspace_grid[i, j]
            plus_k1 = subspace_grid[i + 1, j]
            minus_k1 = subspace_grid[i - 1, j]
            plus_k2 = subspace_grid[i, j + 1]
            minus_k2 = subspace_grid[i, j - 1]

            trace_k1_pm = _trace_two_projectors(plus_k1, minus_k1)
            trace_k2_pm = _trace_two_projectors(plus_k2, minus_k2)
            trace_cross = (
                _trace_two_projectors(plus_k1, plus_k2)
                - _trace_two_projectors(plus_k1, minus_k2)
                - _trace_two_projectors(minus_k1, plus_k2)
                + _trace_two_projectors(minus_k1, minus_k2)
            )
            trace_commutator = (
                _trace_three_projectors(center_subspace, plus_k1, plus_k2)
                - _trace_three_projectors(center_subspace, plus_k1, minus_k2)
                - _trace_three_projectors(center_subspace, minus_k1, plus_k2)
                + _trace_three_projectors(center_subspace, minus_k1, minus_k2)
                - _trace_three_projectors(center_subspace, plus_k2, plus_k1)
                + _trace_three_projectors(center_subspace, plus_k2, minus_k1)
                + _trace_three_projectors(center_subspace, minus_k2, plus_k1)
                - _trace_three_projectors(center_subspace, minus_k2, minus_k1)
            )

            out_i = i - 1
            out_j = j - 1
            g11_frac[out_i, out_j] = (num_occ - trace_k1_pm.real) / (4.0 * delta_kappa1 * delta_kappa1)
            g12_frac[out_i, out_j] = trace_cross.real / (8.0 * delta_kappa1 * delta_kappa2)
            g22_frac[out_i, out_j] = (num_occ - trace_k2_pm.real) / (4.0 * delta_kappa2 * delta_kappa2)
            omega12_frac[out_i, out_j] = (-1j * trace_commutator / (4.0 * delta_kappa1 * delta_kappa2)).real

    return {
        "g11_frac": g11_frac,
        "g12_frac": g12_frac,
        "g22_frac": g22_frac,
        "omega12_frac": omega12_frac,
    }


def transform_metric_components_to_cartesian(g11_frac, g12_frac, g22_frac, b_phys_2d):
    """Transform a symmetric fractional metric tensor into Cartesian coordinates."""
    g_frac = np.empty(g11_frac.shape + (2, 2), dtype=np.float64)
    g_frac[..., 0, 0] = g11_frac
    g_frac[..., 0, 1] = g12_frac
    g_frac[..., 1, 0] = g12_frac
    g_frac[..., 1, 1] = g22_frac

    b_inverse = np.linalg.inv(b_phys_2d)
    left = np.einsum("ab,...bc->...ac", b_inverse, g_frac)
    g_cart = np.einsum("...ab,cb->...ac", left, b_inverse)
    return g_cart[..., 0, 0], g_cart[..., 0, 1], g_cart[..., 1, 1]


def transform_qgt_to_cartesian(qgt_fields, b_phys_2d):
    gxx_cart, gxy_cart, gyy_cart = transform_metric_components_to_cartesian(
        qgt_fields["g11_frac"],
        qgt_fields["g12_frac"],
        qgt_fields["g22_frac"],
        b_phys_2d,
    )
    det_b = float(np.linalg.det(b_phys_2d))
    if np.isclose(det_b, 0.0):
        raise ValueError("det(B_phys_2d) is zero.")
    omega_xy_cart = qgt_fields["omega12_frac"] / det_b
    qgt_cart = dict(qgt_fields)
    qgt_cart["gxx_cart"] = gxx_cart
    qgt_cart["gxy_cart"] = gxy_cart
    qgt_cart["gyy_cart"] = gyy_cart
    qgt_cart["trace_g_cart"] = gxx_cart + gyy_cart
    qgt_cart["omega_xy_cart"] = omega_xy_cart
    return qgt_cart


def moire_bz_area(b_phys_2d):
    """Return the moire Brillouin-zone area in Cartesian reciprocal units."""
    return abs(float(np.linalg.det(b_phys_2d)))


def scale_cartesian_geometric_field(field_cart, b_phys_2d):
    """Scale an Angstrom^2 Cartesian geometric field into a dimensionless mBZ-normalized field."""
    return np.asarray(field_cart, dtype=float) * (moire_bz_area(b_phys_2d) / (2.0 * np.pi))


def integrate_cartesian_field_over_bz(scalar_field_cart, b_phys_2d, delta_kappa1, delta_kappa2):
    """Approximate a Cartesian scalar-field BZ integral from a uniform fractional-grid sum."""
    area_element = moire_bz_area(b_phys_2d) * float(delta_kappa1) * float(delta_kappa2)
    return float(np.sum(np.asarray(scalar_field_cart, dtype=float)) * area_element)


def integrate_cartesian_pseudoscalar_over_bz(pseudoscalar_field_cart, b_phys_2d, delta_kappa1, delta_kappa2):
    """Approximate an oriented Cartesian 2-form / pseudoscalar BZ integral."""
    oriented_area_element = float(np.linalg.det(b_phys_2d)) * float(delta_kappa1) * float(delta_kappa2)
    return float(np.sum(np.asarray(pseudoscalar_field_cart, dtype=float)) * oriented_area_element)


def compute_trace_condition_diagnostic(trace_g, omega_xy, eps=1e-14):
    trace_g = np.asarray(trace_g, dtype=float)
    omega_xy = np.asarray(omega_xy, dtype=float)
    if trace_g.shape != omega_xy.shape:
        raise ValueError(
            "Cannot compute trace-condition diagnostic: trace_g and omega_xy must have the same shape, "
            "got {0} and {1}.".format(trace_g.shape, omega_xy.shape)
        )

    finite_mask = np.isfinite(trace_g) & np.isfinite(omega_xy)
    valid_trace_g = trace_g[finite_mask]
    valid_abs_omega = np.abs(omega_xy[finite_mask])
    if valid_trace_g.size == 0:
        warning = "No finite trace_g/omega_xy points remain after filtering; delta_tr and delta_g are undefined."
        return {
            "mean_trace_g": np.nan,
            "mean_abs_omega": np.nan,
            "rms_residual": np.nan,
            "max_abs_residual": np.nan,
            "rms_trace_g_fluctuation": np.nan,
            "num_points": 0,
            "warning": warning,
            "delta_tr": np.nan,
            "delta_g": np.nan,
        }

    residual = valid_trace_g - valid_abs_omega
    mean_trace_g = float(np.mean(valid_trace_g))
    mean_abs_omega = float(np.mean(valid_abs_omega))
    rms_residual = float(np.sqrt(np.mean(residual**2)))
    max_abs_residual = float(np.max(np.abs(residual)))
    rms_trace_g_fluctuation = float(np.sqrt(np.mean((valid_trace_g - mean_trace_g) ** 2)))

    warning_messages = []
    delta_tr = np.nan
    delta_g = np.nan
    if mean_abs_omega <= float(eps):
        warning_messages.append("mean_abs_omega <= eps; delta_tr is undefined.")
    else:
        delta_tr = float(rms_residual / mean_abs_omega)
    if mean_trace_g <= float(eps):
        warning_messages.append("mean_trace_g <= eps; delta_g is undefined.")
    else:
        delta_g = float(rms_trace_g_fluctuation / mean_trace_g)

    return {
        "mean_trace_g": mean_trace_g,
        "mean_abs_omega": mean_abs_omega,
        "rms_residual": rms_residual,
        "max_abs_residual": max_abs_residual,
        "rms_trace_g_fluctuation": rms_trace_g_fluctuation,
        "num_points": int(valid_trace_g.size),
        "warning": " ".join(warning_messages),
        "delta_tr": delta_tr,
        "delta_g": delta_g,
    }


def trace_g_plot_title(title_prefix, trace_g_cart, b_phys_2d, delta_kappa1, delta_kappa2):
    trace_g_integral = integrate_cartesian_field_over_bz(trace_g_cart, b_phys_2d, delta_kappa1, delta_kappa2)
    return (
        title_prefix
        + "\n"
        + rf"$\frac{{1}}{{2\pi}}\int_{{\mathrm{{BZ}}}}\mathrm{{Tr}}\,g\,d^2k \approx {trace_g_integral / (2.0 * np.pi):.4f}$"
    )


def berry_curvature_density_plot_title(title_prefix, omega_xy_cart, b_phys_2d, delta_kappa1, delta_kappa2):
    omega_integral = integrate_cartesian_pseudoscalar_over_bz(
        omega_xy_cart,
        b_phys_2d,
        delta_kappa1,
        delta_kappa2,
    )
    chern_number = omega_integral / (2.0 * np.pi)
    return (
        title_prefix
        + "\n"
        + rf"$C \approx {chern_number:.4f}$"
    )


def _save_table(output_path, data, header):
    txt_path = output_path.replace(".pdf", ".txt")
    np.savetxt(txt_path, data, header=header, fmt="%15.8f")
    print("Saved data to {0}".format(txt_path))


def _format_trace_condition_value(value):
    return "{0:.8f}".format(float(value))


def _format_band_indices(band_indices):
    return ",".join(str(index) for index in band_indices)


def _finite_cartesian_integral_or_nan(field_cart, integrator, b_phys_2d, delta_kappa1, delta_kappa2):
    field_cart = np.asarray(field_cart, dtype=float)
    if not np.all(np.isfinite(field_cart)):
        return np.nan
    return integrator(field_cart, b_phys_2d, delta_kappa1, delta_kappa2)


def _finite_scaled_integral_or_nan(field_scaled, delta_kappa1, delta_kappa2):
    field_scaled = np.asarray(field_scaled, dtype=float)
    if not np.all(np.isfinite(field_scaled)):
        return np.nan
    return float(np.sum(field_scaled) * float(delta_kappa1) * float(delta_kappa2))


def save_trace_condition_output(
    output_prefix,
    band_indices,
    trace_g_cart,
    omega_xy_cart,
    b_phys_2d,
    delta_kappa1,
    delta_kappa2,
    eps=1e-14,
):
    trace_g_scaled = scale_cartesian_geometric_field(trace_g_cart, b_phys_2d)
    omega_xy_scaled = scale_cartesian_geometric_field(omega_xy_cart, b_phys_2d)
    diagnostic = compute_trace_condition_diagnostic(trace_g_scaled, omega_xy_scaled, eps=eps)
    band_indices = list(band_indices)
    mode = "subspace" if len(band_indices) > 1 else "single_band"
    delta_key = "delta_tr_subspace" if mode == "subspace" else "delta_tr"
    definition = (
        "delta_tr_subspace = sqrt(<(Tr g_sub - |Omega_sub|)^2>) / <|Omega_sub|>"
        if mode == "subspace"
        else "delta_tr = sqrt(<(Tr g - |Omega|)^2>) / <|Omega|>"
    )
    subspace_warning = (
        "This is a subspace trace-condition diagnostic, not a single-band ideal-Chern-band diagnostic."
    )
    integral_trace_g = _finite_scaled_integral_or_nan(
        trace_g_scaled,
        delta_kappa1,
        delta_kappa2,
    )
    integral_abs_omega = _finite_scaled_integral_or_nan(
        np.abs(omega_xy_scaled),
        delta_kappa1,
        delta_kappa2,
    )
    chern_from_qgt_omega = _finite_scaled_integral_or_nan(
        omega_xy_scaled,
        delta_kappa1,
        delta_kappa2,
    )
    if np.isfinite(chern_from_qgt_omega):
        chern_from_qgt_omega *= np.sign(float(np.linalg.det(b_phys_2d))) or 1.0

    warning_text = diagnostic["warning"]
    if mode == "subspace":
        warning_text = subspace_warning if not warning_text else subspace_warning + " " + warning_text

    lines = [
        "# Quantum-geometry trace-condition diagnostic",
        "# Tr g(k) = g_xx(k) + g_yy(k)",
        "# <Tr g> = <Tr g(k)>_BZ",
        "# absOmega(k) = |Omega_xy(k)|",
        "# rms_trace_condition_residual = sqrt(<(Tr g - |Omega|)^2>)",
        "# rms_trace_g_fluctuation = sqrt(<(Tr g - <Tr g>)^2>)",
        "# " + definition,
        "# delta_g = sqrt(<(Tr g - <Tr g>)^2>) / <Tr g>",
        "# mode: {0}".format(mode),
    ]
    if mode == "subspace":
        lines.append("# WARNING: {0}".format(subspace_warning))
    lines.extend(
        [
            "mode {0}".format(mode),
            "band_indices {0}".format(_format_band_indices(band_indices)),
            "",
            "{0} {1}".format(delta_key, _format_trace_condition_value(diagnostic["delta_tr"])),
            "delta_g {0}".format(_format_trace_condition_value(diagnostic["delta_g"])),
            "mean_trace_g_scaled {0}".format(_format_trace_condition_value(diagnostic["mean_trace_g"])),
            "mean_abs_omega_scaled {0}".format(_format_trace_condition_value(diagnostic["mean_abs_omega"])),
            "rms_residual_scaled {0}".format(_format_trace_condition_value(diagnostic["rms_residual"])),
            "max_abs_residual_scaled {0}".format(_format_trace_condition_value(diagnostic["max_abs_residual"])),
            "rms_trace_g_fluctuation_scaled {0}".format(
                _format_trace_condition_value(diagnostic["rms_trace_g_fluctuation"])
            ),
            "num_points {0}".format(diagnostic["num_points"]),
            "integral_trace_g_over_2pi {0}".format(_format_trace_condition_value(integral_trace_g)),
            "integral_abs_omega_over_2pi {0}".format(_format_trace_condition_value(integral_abs_omega)),
            "chern_from_qgt_omega {0}".format(_format_trace_condition_value(chern_from_qgt_omega)),
        ]
    )
    if warning_text:
        lines.append("warning {0}".format(warning_text))

    trace_condition_path = output_prefix + "_trace_condition.txt"
    with open(trace_condition_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("Saved trace-condition diagnostic to {0}".format(trace_condition_path))
    return trace_condition_path


def field_axis_labels():
    """Axis labels for the normalized plotting basis `B_plot = B_phys / |b1|`."""
    return r"$k_x / |b_1|$", r"$k_y / |b_1|$"


def field_colorbar_label(field_kind):
    labels = {
        "berry_flux": r"$\Phi_{\mathrm{Berry}}\ \mathrm{[rad]}$",
        "berry_curvature_density_cart": r"$\Omega_{xy}\ (\mathrm{\AA}^2)$",
        "berry_curvature_density_scaled": r"$(A_{\mathrm{mBZ}}/2\pi)\,\Omega_{xy}$",
        "trace_g_cart": r"$\mathrm{Tr}\, g\ (\mathrm{\AA}^2)$",
        "trace_g_scaled": r"$(A_{\mathrm{mBZ}}/2\pi)\,\mathrm{Tr}\, g$",
        "omega_xy_cart": r"$\Omega_{xy}\ (\mathrm{\AA}^2)$",
        "omega_xy_scaled": r"$(A_{\mathrm{mBZ}}/2\pi)\,\Omega_{xy}$",
    }
    if field_kind not in labels:
        raise ValueError("Unknown field_kind={0}".format(field_kind))
    return labels[field_kind]


def default_savefig_kwargs():
    return dict(_DEFAULT_SAVEFIG_KWARGS)


def _configure_filled_contour_rendering(ax, contour):
    """Reduce vector-PDF seam artifacts for filled contours across matplotlib versions."""
    contour.set_edgecolor("face")
    contour.set_linewidth(0.0)
    contour.set_antialiased(False)

    collections = getattr(contour, "collections", None)
    if collections is not None:
        for collection in collections:
            collection.set_edgecolor("face")
            collection.set_linewidth(0.0)
            collection.set_antialiased(False)
            collection.set_rasterized(True)
        return

    # Matplotlib >= 3.10 no longer exposes `QuadContourSet.collections`.
    # Rasterize the filled contour artist by z-order while keeping axes/text vectorized.
    contour.set_zorder(-5)
    ax.set_rasterization_zorder(-4)


def plot_scalar_field(plot_mesh, scalar_field, output_path, title, colorbar_label, data_columns=None, header=None, save_table=True):
    values = np.asarray(scalar_field)
    if values.ndim != 2:
        raise ValueError("plot_scalar_field expects a 2D scalar field.")

    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    contour = ax.contourf(
        plot_mesh[:, :, 0],
        plot_mesh[:, :, 1],
        values,
        levels=192,
        cmap="viridis",
        antialiased=False,
    )
    _configure_filled_contour_rendering(ax, contour)
    colorbar = fig.colorbar(contour, ax=ax)
    if getattr(colorbar, "solids", None) is not None:
        colorbar.solids.set_edgecolor("face")
        colorbar.solids.set_linewidth(0.0)
    colorbar.set_label(colorbar_label, fontsize=12)
    ax.set_aspect("equal")
    xlabel, ylabel = field_axis_labels()
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=16)
    fig.patch.set_alpha(1.0)
    ax.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(output_path, **default_savefig_kwargs())
    plt.close(fig)

    if save_table:
        if data_columns is None or header is None:
            raise ValueError("data_columns and header are required when save_table=True.")
        _save_table(output_path, data_columns, header)


def save_qgt_outputs(
    output_prefix,
    fractional_mesh,
    cartesian_mesh,
    plot_mesh,
    qgt_fields,
    title_prefix,
    b_phys_2d,
    delta_kappa1,
    delta_kappa2,
    band_indices=None,
):
    qgt_fields = dict(qgt_fields)
    # `kx_cart`, `ky_cart` use the physical reciprocal basis (1/Angstrom).
    # `kx_plot`, `ky_plot` follow the plotted axes and are normalized by |b1|.
    frac_points, cart_points, plot_points = flatten_output_coordinate_meshes(
        fractional_mesh,
        cartesian_mesh,
        plot_mesh,
    )
    # `*_cart` fields retain the physical Angstrom^2 normalization. `*_scaled`
    # fields are dimensionless and better suited for cross-twist-angle comparison.
    qgt_fields["trace_g_scaled"] = scale_cartesian_geometric_field(qgt_fields["trace_g_cart"], b_phys_2d)
    qgt_fields["omega_xy_scaled"] = scale_cartesian_geometric_field(qgt_fields["omega_xy_cart"], b_phys_2d)

    table = np.column_stack(
        [
            frac_points[:, 0],
            frac_points[:, 1],
            cart_points[:, 0],
            cart_points[:, 1],
            plot_points[:, 0],
            plot_points[:, 1],
            qgt_fields["g11_frac"].reshape(-1),
            qgt_fields["g12_frac"].reshape(-1),
            qgt_fields["g22_frac"].reshape(-1),
            qgt_fields["omega12_frac"].reshape(-1),
            qgt_fields["gxx_cart"].reshape(-1),
            qgt_fields["gxy_cart"].reshape(-1),
            qgt_fields["gyy_cart"].reshape(-1),
            qgt_fields["trace_g_cart"].reshape(-1),
            qgt_fields["trace_g_scaled"].reshape(-1),
            qgt_fields["omega_xy_cart"].reshape(-1),
            qgt_fields["omega_xy_scaled"].reshape(-1),
        ]
    )
    txt_path = output_prefix + ".txt"
    np.savetxt(
        txt_path,
        table,
        header=(
            "kappa1 kappa2 kx_cart ky_cart kx_plot ky_plot "
            "g11_frac g12_frac g22_frac omega12_frac "
            "gxx_cart gxy_cart gyy_cart trace_g_cart trace_g_scaled "
            "omega_xy_cart omega_xy_scaled"
        ),
        fmt="%15.8f",
    )
    print("Saved QGT data to {0}".format(txt_path))
    plot_scalar_field(
        plot_mesh,
        qgt_fields["trace_g_scaled"],
        output_prefix + "_trace_g_scaled.pdf",
        trace_g_plot_title(title_prefix, qgt_fields["trace_g_cart"], b_phys_2d, delta_kappa1, delta_kappa2),
        field_colorbar_label("trace_g_scaled"),
        save_table=False,
    )
    if band_indices is not None:
        try:
            save_trace_condition_output(
                output_prefix,
                band_indices,
                qgt_fields["trace_g_cart"],
                qgt_fields["omega_xy_cart"],
                b_phys_2d,
                delta_kappa1,
                delta_kappa2,
            )
        except ValueError:
            raise
        except Exception as exc:
            print(
                "[WARN] Skipping trace-condition diagnostic for {0}: {1}".format(
                    output_prefix,
                    exc,
                )
            )


def parse_lattice_vectors_from_openmx(openmx_path):
    """Read the physical reciprocal basis B_phys from openmx.dat, in 1/Angstrom."""
    if not os.path.exists(openmx_path):
        raise FileNotFoundError("openmx.dat 文件未找到: {0}".format(openmx_path))
    with open(openmx_path, "r") as handle:
        lines = handle.readlines()

    start = None
    end = None
    for idx, line in enumerate(lines):
        if "<Atoms.UnitVectors" in line:
            start = idx + 1
        if "Atoms.UnitVectors>" in line:
            end = idx
            break
    if start is None or end is None or end - start < 3:
        raise ValueError("openmx.dat 文件中未找到合法的 Atoms.UnitVectors 区块: {0}".format(openmx_path))

    lattice = np.array([[float(x) for x in lines[start + i].split()[:3]] for i in range(3)], dtype=float)
    return 2.0 * np.pi * np.linalg.inv(lattice.T)


def parallel_transport_path(vecs_path):
    for i in range(len(vecs_path) - 1):
        overlap = np.conj(vecs_path[i].T) @ vecs_path[i + 1]
        u_mat, _, v_dag = np.linalg.svd(overlap)
        vecs_path[i + 1] = vecs_path[i + 1] @ (u_mat @ v_dag)
    return vecs_path


def wilson_loop(vecs_occ_path):
    n_path, _, n_occ = vecs_occ_path.shape
    wilson = np.eye(n_occ, dtype=np.complex128)
    for i in range(n_path):
        overlap = np.conj(vecs_occ_path[(i + 1) % n_path].T) @ vecs_occ_path[i]
        wilson = overlap @ wilson
    phases = np.angle(np.linalg.eigvals(wilson)) / (2.0 * np.pi)
    return np.sort(phases) % 1.0


def _loop_indices_without_periodic_duplicate(kappa_values):
    if len(kappa_values) > 1 and np.isclose(kappa_values[-1] - kappa_values[0], 1.0):
        return list(range(len(kappa_values) - 1))
    return list(range(len(kappa_values)))


def wcc_sweep_axis_label(direction):
    if direction == "ky":
        return r"$\kappa_1$"
    if direction == "kx":
        return r"$\kappa_2$"
    raise ValueError("direction must be 'kx' or 'ky'")


def sweep_wcc(eig_vec_grid, occ_bands, kappa1_values, kappa2_values, direction="ky"):
    """Sweep Wilson loops along one fractional axis while fixing the other."""
    if direction == "ky":
        sweep_values = np.asarray(kappa1_values)
        # Legacy compatibility: keep the stored periodic endpoint in the Wilson-loop path.
        loop_indices = list(range(len(kappa2_values)))
        sweep_count = eig_vec_grid.shape[0]
    else:
        sweep_values = np.asarray(kappa2_values)
        # Legacy compatibility: keep the stored periodic endpoint in the Wilson-loop path.
        loop_indices = list(range(len(kappa1_values)))
        sweep_count = eig_vec_grid.shape[1]

    all_wcc = []
    for fixed_index in range(sweep_count):
        vecs = []
        for loop_index in loop_indices:
            if direction == "ky":
                eigvec = eig_vec_grid[fixed_index, loop_index]
            else:
                eigvec = eig_vec_grid[loop_index, fixed_index]
            vecs.append(eigvec[:, occ_bands])
        all_wcc.append(wilson_loop(np.stack(vecs, axis=0)))
    return sweep_values, np.stack(all_wcc, axis=0)


def plot_wcc(kappa_sweep_values, wcc_branches, output_path, direction, title=None):
    fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
    for branch_index in range(wcc_branches.shape[1]):
        ax.scatter(kappa_sweep_values, wcc_branches[:, branch_index], c="black", s=2)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(float(np.min(kappa_sweep_values)), float(np.max(kappa_sweep_values)))
    ax.set_ylabel(r"$\theta / 2\pi$", fontsize=16)
    ax.set_xlabel(wcc_sweep_axis_label(direction), fontsize=16)
    if title:
        ax.set_title(title, fontsize=16)
    fig.patch.set_alpha(1.0)
    ax.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(output_path, **default_savefig_kwargs())
    plt.close(fig)

    header = "kappa_1" if direction == "ky" else "kappa_2"
    for branch_index in range(wcc_branches.shape[1]):
        header += " wcc_band_{0}".format(branch_index)
    txt_path = output_path.replace(".pdf", ".txt")
    np.savetxt(
        txt_path,
        np.hstack([kappa_sweep_values.reshape(-1, 1), wcc_branches]),
        header=header,
        fmt="%12.6f",
    )
    print("Saved WCC data to {0}".format(txt_path))


def _resolve_chern_grid_from_config(config):
    compute_config = config.get("compute", {})
    num_chern = compute_config.get("num_chern", 40)
    return resolve_chern_grid_shape(
        num_chern=num_chern,
        num_k1=compute_config.get("num_k1"),
        num_k2=compute_config.get("num_k2"),
    )


def _grid_label(num_k1, num_k2):
    if int(num_k1) == int(num_k2):
        return str(int(num_k1))
    return "{0}x{1}".format(int(num_k1), int(num_k2))


def _resolve_band_indices(raw_indices, num_bands):
    resolved = []
    for band_index in raw_indices:
        actual = band_index if band_index >= 0 else num_bands + band_index
        if actual < 0 or actual >= num_bands:
            raise IndexError("带号 {0} 超出范围，有效范围: 0 ~ {1} 或负数索引。".format(band_index, num_bands - 1))
        resolved.append(actual)
    return resolved


def _candidate_suffixes(num_k1, num_k2, num_chern):
    suffixes = [format_chern_grid_suffix(num_k1, num_k2)]
    legacy_k1, legacy_k2 = resolve_chern_grid_shape(num_chern)
    legacy_suffix = format_chern_grid_suffix(legacy_k1, legacy_k2)
    if legacy_suffix not in suffixes:
        suffixes.append(legacy_suffix)
    suffixes.extend(["_2d", "_2D", "_chern", "_Chern", "_CHERN", ""])
    seen = []
    for suffix in suffixes:
        if suffix not in seen:
            seen.append(suffix)
    return seen


def _locate_wavefunction_file(output_dir, band_type, valley_str, num_k1, num_k2, num_chern):
    suffixes = _candidate_suffixes(num_k1, num_k2, num_chern)
    seen = set()
    for suffix in suffixes:
        candidate = os.path.join(
            output_dir,
            array_output_filename(f"vec_{band_type}", valley_flag=valley_str, suffix=suffix, tapw=True),
        )
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate
    prefixes = [
        "vec_{0}_{1}_valley".format(band_type, valley_str),
        "vec_{0}_valley".format(valley_str),
    ]
    for prefix in prefixes:
        for suffix in suffixes:
            candidate = os.path.join(output_dir, prefix + suffix + ".npy")
            if candidate in seen:
                continue
            if os.path.exists(candidate):
                return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="TAPW Chern number, Wilson-loop, and QGT post-processing")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("-b", "--band", type=int, nargs="+", help="Band index list for Berry/QGT outputs")
    parser.add_argument("-wb", "--wcc-bands", type=int, nargs="+", help="Band index list for WCC")
    parser.add_argument(
        "-wd",
        "--wcc-direction",
        type=str,
        default="ky",
        choices=["kx", "ky"],
        help="Wilson-loop direction along the stored fractional grid (default: ky)",
    )
    parser.add_argument("-o", "--output-dir", type=str, default="./", help="Output directory for topology files")
    parser.add_argument(
        "-v",
        "--valley",
        type=int,
        default=1,
        help="Valley number (1:K1, 2:K2, 5:Gamma, 11:K1_120, 12:K1_240, 31:M1, 32:M2, 33:M3)",
    )
    parser.add_argument("-bt", "--band-type", type=str, default="VBM", help="Band type (default: VBM)")
    args = parser.parse_args()

    if not args.band and not args.wcc_bands:
        parser.error("No action requested, add --band or --wcc-bands")

    valley_str = VALLEY_MAP.get(args.valley, "valley{0}".format(args.valley))
    config = load_config(args.config)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    num_k1, num_k2 = _resolve_chern_grid_from_config(config)
    num_chern = config.get("compute", {}).get("num_chern", num_k1)
    grid_label = _grid_label(num_k1, num_k2)
    band_type = args.band_type if args.band_type else config.get("compute", {}).get("band_type", "VBM")

    input_file = config["paths"].get("input_file", "openmx.dat")
    openmx_path = input_file if os.path.isabs(input_file) else os.path.join(config["paths"]["output_dir"], input_file)
    if not os.path.exists(openmx_path):
        print("[ERROR] openmx.dat 文件未找到: {0}".format(openmx_path))
        sys.exit(1)

    try:
        b_phys = parse_lattice_vectors_from_openmx(openmx_path)
        b_phys_2d = np.asarray(b_phys[:2, :2], dtype=float)
        b_plot = build_plot_basis(b_phys_2d)
    except Exception as exc:
        print("[ERROR] 解析 openmx.dat 失败: {0}".format(exc))
        sys.exit(1)

    vec_file = _locate_wavefunction_file(output_dir, band_type, valley_str, num_k1, num_k2, num_chern)
    if vec_file is None:
        print(
            "[ERROR] 波函数文件未找到，请检查 {0} 下的 vec_{1}_{2}_valley*.npy 或 vec_{2}_valley*.npy 是否存在。".format(
                output_dir,
                band_type,
                valley_str,
            )
        )
        sys.exit(1)

    band_vec = np.load(vec_file)
    try:
        band_vec_grid = reshape_wavefunction_grid(band_vec, num_k1, num_k2)
    except Exception as exc:
        print("[ERROR] 波函数文件形状不正确: {0}".format(exc))
        sys.exit(1)

    delta_kappa1, delta_kappa2, kappa1_values, kappa2_values = compute_fractional_spacings(num_k1, num_k2)
    # `kappa1`, `kappa2` are dimensionless fractional moire reciprocal coordinates.
    plaquette_fractional_mesh = build_fractional_plaquette_center_mesh(num_k1, num_k2)
    # `kx_cart`, `ky_cart` come from the physical reciprocal basis and are in 1/Angstrom.
    # `kx_plot`, `ky_plot` use the same k / |b1| normalization as the figure axes.
    plaquette_cart_mesh = fractional_mesh_to_cartesian(plaquette_fractional_mesh, b_phys_2d)
    plaquette_plot_mesh = fractional_mesh_to_cartesian(plaquette_fractional_mesh, b_plot)
    interior_fractional_mesh = build_fractional_interior_mesh(num_k1, num_k2)
    interior_cart_mesh = fractional_mesh_to_cartesian(interior_fractional_mesh, b_phys_2d)
    interior_plot_mesh = fractional_mesh_to_cartesian(interior_fractional_mesh, b_plot)
    fig_suffix = "{0}_{1}_{2}".format(band_type, valley_str, grid_label)

    if args.band:
        try:
            resolved_band_indices = _resolve_band_indices(args.band, band_vec_grid.shape[-1])
        except IndexError as exc:
            print("[ERROR] {0}".format(exc))
            sys.exit(1)

        for raw_band_index, band_index in zip(args.band, resolved_band_indices):
            berry_flux = compute_berry_flux_single_band(band_vec_grid, band_index)
            berry_curvature_density_cart = berry_flux_to_cartesian_density(
                berry_flux,
                b_phys_2d,
                delta_kappa1,
                delta_kappa2,
            )
            berry_curvature_density_scaled = scale_cartesian_geometric_field(
                berry_curvature_density_cart,
                b_phys_2d,
            )
            chern_number = np.sum(berry_flux) / (2.0 * np.pi)

            frac_points, cart_points, plot_points = flatten_output_coordinate_meshes(
                plaquette_fractional_mesh,
                plaquette_cart_mesh,
                plaquette_plot_mesh,
            )

            bc_table = np.column_stack(
                [
                    frac_points[:, 0],
                    frac_points[:, 1],
                    cart_points[:, 0],
                    cart_points[:, 1],
                    plot_points[:, 0],
                    plot_points[:, 1],
                    berry_flux.reshape(-1),
                    berry_curvature_density_cart.reshape(-1),
                    berry_curvature_density_scaled.reshape(-1),
                ]
            )
            bc_path = os.path.join(output_dir, "bc_band_{0}_{1}.txt".format(raw_band_index, fig_suffix))
            np.savetxt(
                bc_path,
                bc_table,
                header=(
                    "kappa1_center kappa2_center kx_cart_center ky_cart_center "
                    "kx_plot_center ky_plot_center "
                    "berry_flux berry_curvature_density_cart berry_curvature_density_scaled"
                ),
                fmt="%15.8f",
            )
            plot_scalar_field(
                plaquette_plot_mesh,
                berry_curvature_density_scaled,
                os.path.join(
                    output_dir,
                    "berry_curvature_density_band_{0}_{1}_scaled.pdf".format(raw_band_index, fig_suffix),
                ),
                berry_curvature_density_plot_title(
                    "BC for Band {0} in {1}".format(raw_band_index, valley_str),
                    berry_curvature_density_cart,
                    b_phys_2d,
                    delta_kappa1,
                    delta_kappa2,
                ),
                field_colorbar_label("berry_curvature_density_scaled"),
                save_table=False,
            )

            qgt_fields = transform_qgt_to_cartesian(
                compute_qgt_fields(band_vec_grid, [band_index], delta_kappa1, delta_kappa2),
                b_phys_2d,
            )
            print(
                "[INFO] Band {0}: computed QGT on interior {1}x{2} grid.".format(
                    raw_band_index,
                    qgt_fields["g11_frac"].shape[0],
                    qgt_fields["g11_frac"].shape[1],
                )
            )
            save_qgt_outputs(
                os.path.join(output_dir, "qgt_band_{0}_{1}".format(raw_band_index, fig_suffix)),
                interior_fractional_mesh,
                interior_cart_mesh,
                interior_plot_mesh,
                qgt_fields,
                "Tr g for Band {0} in {1}".format(raw_band_index, valley_str),
                b_phys_2d,
                delta_kappa1,
                delta_kappa2,
                band_indices=[raw_band_index],
            )

            with open(
                os.path.join(output_dir, "chern_{0}_{1}_band{2}.txt".format(band_type, valley_str, raw_band_index)),
                "w",
            ) as handle:
                handle.write(
                    "Chern number from Berry flux for band {0} ({1}): {2:.8f}\n".format(
                        raw_band_index,
                        band_type,
                        chern_number,
                    )
                )
            print(
                "[INFO] Band {0}: Chern number = {1:.8f}, BC table -> {2}, scaled Berry density -> {3}".format(
                    raw_band_index,
                    chern_number,
                    bc_path,
                    os.path.join(
                        output_dir,
                        "berry_curvature_density_band_{0}_{1}_scaled.pdf".format(raw_band_index, fig_suffix),
                    ),
                )
            )

        if len(resolved_band_indices) > 1:
            band_str = "_".join(str(i) for i in args.band)
            berry_flux_multiband = compute_berry_flux_multiband(band_vec_grid, resolved_band_indices)
            berry_curvature_density_multiband = berry_flux_to_cartesian_density(
                berry_flux_multiband,
                b_phys_2d,
                delta_kappa1,
                delta_kappa2,
            )
            berry_curvature_density_scaled_multiband = scale_cartesian_geometric_field(
                berry_curvature_density_multiband,
                b_phys_2d,
            )
            chern_number = np.sum(berry_flux_multiband) / (2.0 * np.pi)

            frac_points, cart_points, plot_points = flatten_output_coordinate_meshes(
                plaquette_fractional_mesh,
                plaquette_cart_mesh,
                plaquette_plot_mesh,
            )

            bc_multiband_path = os.path.join(output_dir, "bc_bands_{0}_{1}.txt".format(band_str, fig_suffix))
            np.savetxt(
                bc_multiband_path,
                np.column_stack(
                    [
                        frac_points[:, 0],
                        frac_points[:, 1],
                        cart_points[:, 0],
                        cart_points[:, 1],
                        plot_points[:, 0],
                        plot_points[:, 1],
                        berry_flux_multiband.reshape(-1),
                        berry_curvature_density_multiband.reshape(-1),
                        berry_curvature_density_scaled_multiband.reshape(-1),
                    ]
                ),
                header=(
                    "kappa1_center kappa2_center kx_cart_center ky_cart_center "
                    "kx_plot_center ky_plot_center "
                    "berry_flux berry_curvature_density_cart berry_curvature_density_scaled"
                ),
                fmt="%15.8f",
            )
            plot_scalar_field(
                plaquette_plot_mesh,
                berry_curvature_density_scaled_multiband,
                os.path.join(
                    output_dir,
                    "berry_curvature_density_bands_{0}_{1}_scaled.pdf".format(band_str, fig_suffix),
                ),
                berry_curvature_density_plot_title(
                    "BC for Bands {0} in {1}".format(band_str, valley_str),
                    berry_curvature_density_multiband,
                    b_phys_2d,
                    delta_kappa1,
                    delta_kappa2,
                ),
                field_colorbar_label("berry_curvature_density_scaled"),
                save_table=False,
            )

            qgt_fields = transform_qgt_to_cartesian(
                compute_qgt_fields(band_vec_grid, resolved_band_indices, delta_kappa1, delta_kappa2),
                b_phys_2d,
            )
            print(
                "[INFO] Bands {0}: computed QGT on interior {1}x{2} grid.".format(
                    band_str,
                    qgt_fields["g11_frac"].shape[0],
                    qgt_fields["g11_frac"].shape[1],
                )
            )
            save_qgt_outputs(
                os.path.join(output_dir, "qgt_bands_{0}_{1}".format(band_str, fig_suffix)),
                interior_fractional_mesh,
                interior_cart_mesh,
                interior_plot_mesh,
                qgt_fields,
                "Tr g for Bands {0} in {1}".format(band_str, valley_str),
                b_phys_2d,
                delta_kappa1,
                delta_kappa2,
                band_indices=args.band,
            )

            with open(
                os.path.join(output_dir, "chern_{0}_{1}_bands_{2}.txt".format(band_type, valley_str, band_str)),
                "w",
            ) as handle:
                handle.write(
                    "Chern number from Berry flux for bands {0} ({1}): {2:.8f}\n".format(
                        band_str,
                        band_type,
                        chern_number,
                    )
                )
            print(
                "[INFO] Bands {0}: Chern number = {1:.8f}, BC table -> {2}, scaled Berry density -> {3}".format(
                    band_str,
                    chern_number,
                    bc_multiband_path,
                    os.path.join(
                        output_dir,
                        "berry_curvature_density_bands_{0}_{1}_scaled.pdf".format(band_str, fig_suffix),
                    ),
                )
            )

    if args.wcc_bands:
        try:
            occ_bands = _resolve_band_indices(args.wcc_bands, band_vec_grid.shape[-1])
        except IndexError as exc:
            print("[ERROR] {0}".format(exc))
            sys.exit(1)

        sweep_values, wcc_branches = sweep_wcc(
            band_vec_grid,
            occ_bands,
            kappa1_values,
            kappa2_values,
            direction=args.wcc_direction,
        )
        wcc_bands_str = "_".join(str(i) for i in args.wcc_bands)
        wcc_img = os.path.join(output_dir, "wcc_{0}_{1}_{2}.pdf".format(args.wcc_direction, wcc_bands_str, fig_suffix))
        plot_wcc(
            sweep_values,
            wcc_branches,
            wcc_img,
            args.wcc_direction,
            title="WCC bands {0}".format(args.wcc_bands),
        )
        print(
            "[INFO] WCC for bands {0} in direction {1} saved to {2}".format(
                args.wcc_bands,
                args.wcc_direction,
                wcc_img,
            )
        )


if __name__ == "__main__":
    main()

import numpy as np

from tapw.workflows.band import BandStructureCalculator
from tapw import chern_post
from tapw.config import resolve_chern_grid_shape


def _expected_flattened_pairs(values1, values2):
    return np.array([[v1, v2] for v1 in values1 for v2 in values2], dtype=float)


def _random_orthonormal_grid(seed, shape, dim_h, num_bands):
    rng = np.random.default_rng(seed)
    grid = np.empty(shape + (dim_h, num_bands), dtype=np.complex128)
    for index in np.ndindex(shape):
        raw = rng.normal(size=(dim_h, num_bands)) + 1j * rng.normal(size=(dim_h, num_bands))
        q_mat, _ = np.linalg.qr(raw)
        grid[index] = q_mat[:, :num_bands]
    return grid


def _reference_qgt_from_projectors(band_grid, band_indices, delta_kappa1, delta_kappa2):
    projector_grid = np.einsum(
        "...ma,...na->...mn",
        band_grid[:, :, :, band_indices],
        np.conj(band_grid[:, :, :, band_indices]),
    )
    projector_center = projector_grid[1:-1, 1:-1]
    d1_projector = (projector_grid[2:, 1:-1] - projector_grid[:-2, 1:-1]) / (2.0 * delta_kappa1)
    d2_projector = (projector_grid[1:-1, 2:] - projector_grid[1:-1, :-2]) / (2.0 * delta_kappa2)

    return {
        "g11_frac": 0.5 * np.einsum("...ij,...ji->...", d1_projector, d1_projector).real,
        "g12_frac": 0.5 * np.einsum("...ij,...ji->...", d1_projector, d2_projector).real,
        "g22_frac": 0.5 * np.einsum("...ij,...ji->...", d2_projector, d2_projector).real,
        "omega12_frac": (
            -1j
            * np.einsum(
                "...ij,...ji->...",
                projector_center,
                np.matmul(d1_projector, d2_projector) - np.matmul(d2_projector, d1_projector),
            )
        ).real,
    }


def test_fractional_mesh_flatten_order_matches_ndarray_flatten_order():
    vertex_mesh = chern_post.build_fractional_vertex_mesh(4, 5)
    vertex_pairs = chern_post.flatten_coordinate_mesh(vertex_mesh)
    assert vertex_pairs.shape == (20, 2)
    assert np.allclose(
        vertex_pairs,
        _expected_flattened_pairs(
            np.linspace(-0.5, 0.5, 4),
            np.linspace(-0.5, 0.5, 5),
        ),
    )

    plaquette_mesh = chern_post.build_fractional_plaquette_center_mesh(4, 5)
    plaquette_pairs = chern_post.flatten_coordinate_mesh(plaquette_mesh)
    assert plaquette_pairs.shape == (12, 2)
    assert np.allclose(
        plaquette_pairs,
        _expected_flattened_pairs(
            np.array([-1.0 / 3.0, 0.0, 1.0 / 3.0]),
            np.array([-0.375, -0.125, 0.125, 0.375]),
        ),
    )

    interior_mesh = chern_post.build_fractional_interior_mesh(4, 5)
    interior_pairs = chern_post.flatten_coordinate_mesh(interior_mesh)
    assert interior_pairs.shape == (6, 2)
    assert np.allclose(
        interior_pairs,
        _expected_flattened_pairs(
            np.array([-1.0 / 6.0, 1.0 / 6.0]),
            np.array([-0.25, 0.0, 0.25]),
        ),
    )


def test_build_axis_edges_from_centers_matches_expected_half_step_edges():
    edges = chern_post.build_axis_edges_from_centers(np.array([-0.25, 0.0, 0.25]))
    assert np.allclose(edges, np.array([-0.375, -0.125, 0.125, 0.375]))


def test_resolve_chern_grid_shape_accepts_legacy_scalar_and_two_entry_list():
    assert resolve_chern_grid_shape(40) == (40, 40)
    assert resolve_chern_grid_shape([30, 50]) == (30, 50)


def test_explicit_num_k1_num_k2_override_num_chern_list():
    assert resolve_chern_grid_shape([30, 50], num_k1=12, num_k2=18) == (12, 18)


def test_candidate_suffixes_accept_num_chern_list():
    suffixes = chern_post._candidate_suffixes(30, 50, [30, 50])
    assert "_2d_30x50" in suffixes


def test_rectangular_wavefunction_reshape_preserves_flatten_order():
    num_k1 = 3
    num_k2 = 5
    dim_h = 2
    num_bands = 4
    raw = np.arange(num_k1 * num_k2 * dim_h * num_bands, dtype=float).reshape(
        num_k1 * num_k2,
        dim_h,
        num_bands,
    )
    reshaped = chern_post.reshape_wavefunction_grid(raw, num_k1, num_k2)

    assert reshaped.shape == (num_k1, num_k2, dim_h, num_bands)
    assert np.array_equal(reshaped[0, 0], raw[0])
    assert np.array_equal(reshaped[1, 0], raw[num_k2])
    assert np.array_equal(reshaped[2, 4], raw[-1])


def test_square_generate_kmesh_uses_legacy_flatten_order_for_backward_compatibility():
    calculator = BandStructureCalculator.__new__(BandStructureCalculator)

    kpoints = calculator.generate_kmesh(3, 3)

    assert kpoints.shape == (9, 3)
    assert np.allclose(kpoints[0], np.array([-0.5, -0.5, 0.0]))
    assert np.allclose(kpoints[1], np.array([0.0, -0.5, 0.0]))
    assert np.allclose(kpoints[2], np.array([0.5, -0.5, 0.0]))
    assert np.allclose(kpoints[3], np.array([-0.5, 0.0, 0.0]))


def test_single_band_berry_flux_supports_rectangular_grids():
    band_grid = np.zeros((3, 5, 2, 1), dtype=np.complex128)
    band_grid[..., 0, 0] = 1.0

    berry_flux = chern_post.compute_berry_flux_single_band(band_grid, 0)

    assert berry_flux.shape == (2, 4)
    assert np.allclose(berry_flux, 0.0)


def test_qgt_matches_projector_reference_for_single_band():
    band_grid = _random_orthonormal_grid(seed=0, shape=(5, 4), dim_h=6, num_bands=2)
    delta_kappa1 = 0.2
    delta_kappa2 = 0.25

    qgt_fast = chern_post.compute_qgt_fields(band_grid, [0], delta_kappa1, delta_kappa2)
    qgt_reference = _reference_qgt_from_projectors(band_grid, [0], delta_kappa1, delta_kappa2)

    for key in qgt_reference:
        assert np.allclose(qgt_fast[key], qgt_reference[key], atol=1e-10)


def test_qgt_is_invariant_under_local_subspace_rotations():
    band_grid = _random_orthonormal_grid(seed=1, shape=(5, 5), dim_h=7, num_bands=3)
    rotated_grid = band_grid.copy()

    rng = np.random.default_rng(2)
    for index in np.ndindex(rotated_grid.shape[:2]):
        raw = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        unitary, _ = np.linalg.qr(raw)
        subspace = rotated_grid[index][:, :2]
        rotated_grid[index][:, :2] = subspace @ unitary

    qgt_original = chern_post.compute_qgt_fields(band_grid, [0, 1], 0.25, 0.25)
    qgt_rotated = chern_post.compute_qgt_fields(rotated_grid, [0, 1], 0.25, 0.25)

    for key in qgt_original:
        assert np.allclose(qgt_original[key], qgt_rotated[key], atol=1e-10)


def test_qgt_does_not_materialize_projectors_for_the_full_k_grid(monkeypatch):
    band_grid = _random_orthonormal_grid(seed=3, shape=(5, 4), dim_h=8, num_bands=2)
    original = chern_post._projector_from_subspace

    def guarded_projector(subspace):
        if np.asarray(subspace).ndim > 2:
            raise AssertionError("compute_qgt_fields should not materialize a full projector grid.")
        return original(subspace)

    monkeypatch.setattr(chern_post, "_projector_from_subspace", guarded_projector)

    chern_post.compute_qgt_fields(band_grid, [0], 0.2, 0.25)


def test_cartesian_metric_transform_matches_orthogonal_basis_scaling():
    g11_frac = np.array([[4.0]])
    g12_frac = np.array([[2.0]])
    g22_frac = np.array([[9.0]])
    b_phys_2d = np.array([[2.0, 0.0], [0.0, 3.0]])

    gxx_cart, gxy_cart, gyy_cart = chern_post.transform_metric_components_to_cartesian(
        g11_frac,
        g12_frac,
        g22_frac,
        b_phys_2d,
    )

    assert np.allclose(gxx_cart, [[1.0]])
    assert np.allclose(gxy_cart, [[1.0 / 3.0]])
    assert np.allclose(gyy_cart, [[1.0]])


def test_wcc_axis_labels_use_fractional_coordinates():
    assert chern_post.wcc_sweep_axis_label("ky") == r"$\kappa_1$"
    assert chern_post.wcc_sweep_axis_label("kx") == r"$\kappa_2$"


def test_sweep_wcc_keeps_periodic_endpoint_like_legacy_behavior(monkeypatch):
    eig_vec_grid = _random_orthonormal_grid(seed=4, shape=(3, 4), dim_h=5, num_bands=2)
    kappa1_values = np.linspace(-0.5, 0.5, 3)
    kappa2_values = np.linspace(-0.5, 0.5, 4)
    path_lengths = []

    def fake_wilson_loop(vecs_occ_path):
        path_lengths.append(vecs_occ_path.shape[0])
        return np.zeros(vecs_occ_path.shape[-1], dtype=float)

    monkeypatch.setattr(chern_post, "wilson_loop", fake_wilson_loop)

    chern_post.sweep_wcc(eig_vec_grid, [0, 1], kappa1_values, kappa2_values, direction="ky")
    chern_post.sweep_wcc(eig_vec_grid, [0, 1], kappa1_values, kappa2_values, direction="kx")

    assert path_lengths[:3] == [4, 4, 4]
    assert path_lengths[3:] == [3, 3, 3, 3]


def test_field_labels_expose_normalization_and_units():
    xlabel, ylabel = chern_post.field_axis_labels()
    assert "|b_1|" in xlabel
    assert "|b_1|" in ylabel
    assert "cart" not in xlabel
    assert "cart" not in ylabel
    assert "cart" not in chern_post.field_colorbar_label("berry_curvature_density_cart")
    assert "cart" not in chern_post.field_colorbar_label("trace_g_cart")
    assert r"\AA" in chern_post.field_colorbar_label("berry_curvature_density_cart")
    assert r"\AA" in chern_post.field_colorbar_label("trace_g_cart")
    assert "rad" in chern_post.field_colorbar_label("berry_flux")


def test_trace_g_bz_integral_uses_cartesian_area_element():
    trace_g_cart = np.full((2, 3), 2.0)
    b_phys_2d = np.array([[2.0, 0.0], [0.0, 3.0]])

    integral = chern_post.integrate_cartesian_field_over_bz(trace_g_cart, b_phys_2d, 0.25, 0.5)

    assert np.isclose(integral, 9.0)


def test_trace_g_title_reports_bz_integral_value():
    trace_g_cart = np.full((2, 3), 2.0)
    b_phys_2d = np.array([[2.0, 0.0], [0.0, 3.0]])

    title = chern_post.trace_g_plot_title("QGT for Band -1 in K1", trace_g_cart, b_phys_2d, 0.25, 0.5)

    assert "Trace[g]" in title
    assert "BZ" in title
    assert "9.0000" in title


def test_berry_curvature_integral_uses_oriented_cartesian_area_element():
    omega_xy_cart = np.full((2, 3), 2.0)
    b_phys_2d = np.array([[2.0, 0.0], [0.0, -3.0]])

    integral = chern_post.integrate_cartesian_pseudoscalar_over_bz(omega_xy_cart, b_phys_2d, 0.25, 0.5)

    assert np.isclose(integral, -9.0)


def test_berry_curvature_title_reports_bz_integral_and_chern():
    omega_xy_cart = np.full((2, 3), 2.0)
    b_phys_2d = np.array([[2.0, 0.0], [0.0, 3.0]])

    title = chern_post.berry_curvature_density_plot_title(
        "Berry Curvature Density for Band -1 in K1",
        omega_xy_cart,
        b_phys_2d,
        0.25,
        0.5,
    )

    assert "Berry Curvature Density" in title
    assert "BZ" in title
    assert "9.0000" in title
    assert "1.4324" in title


def test_default_savefig_is_opaque_white_background():
    kwargs = chern_post.default_savefig_kwargs()
    assert kwargs["transparent"] is False
    assert kwargs["facecolor"] == "white"


def test_plot_scalar_field_supports_current_matplotlib_contour_api(tmp_path):
    plot_mesh = chern_post.build_fractional_plaquette_center_mesh(4, 5)
    scalar_field = np.arange(12, dtype=float).reshape(3, 4)
    output_path = tmp_path / "field.pdf"
    data_columns = np.column_stack(
        [
            chern_post.flatten_coordinate_mesh(plot_mesh),
            scalar_field.reshape(-1),
        ]
    )

    chern_post.plot_scalar_field(
        plot_mesh,
        scalar_field,
        str(output_path),
        "test",
        "units",
        data_columns,
        "kappa1 kappa2 value",
    )

    assert output_path.exists()
    assert output_path.with_suffix(".txt").exists()


def test_plot_wcc_exports_fractional_sweep_axis_name(tmp_path):
    output_path = tmp_path / "wcc_test.pdf"
    chern_post.plot_wcc(
        np.array([-0.5, 0.0, 0.5]),
        np.array([[0.1], [0.2], [0.3]]),
        str(output_path),
        "ky",
        title="test",
    )

    header = output_path.with_suffix(".txt").read_text().splitlines()[0]
    assert "kappa_1" in header
    assert "k_x" not in header
    assert "k_y" not in header

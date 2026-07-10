import numpy as np
import pytest
import scipy.sparse
from types import SimpleNamespace

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


def test_fractional_mesh_accepts_topology_ranges():
    vertex_mesh = chern_post.build_fractional_vertex_mesh(3, 5, (0.0, 0.5), (-0.5, 0.5))
    vertex_pairs = chern_post.flatten_coordinate_mesh(vertex_mesh)

    assert np.allclose(
        vertex_pairs,
        _expected_flattened_pairs(
            np.array([0.0, 0.25, 0.5]),
            np.array([-0.5, -0.25, 0.0, 0.25, 0.5]),
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


def test_topology_mesh_overrides_legacy_num_chern():
    cfg = {
        "compute": {"num_chern": 31},
        "topology": {
            "mesh": {
                "b1": 21,
                "b2": 41,
                "range_b1": [0.0, 0.5],
                "range_b2": [-0.5, 0.5],
            }
        },
    }

    assert chern_post._resolve_topology_mesh(cfg) == (21, 41, (0.0, 0.5), (-0.5, 0.5))


def test_wcc_loop_requires_full_period_axis():
    with pytest.raises(ValueError, match="span one reciprocal period"):
        chern_post._validate_wcc_loop_range("b1", (0.0, 0.5), (-0.5, 0.5))

    chern_post._validate_wcc_loop_range("b2", (0.0, 0.5), (-0.5, 0.5))


def test_wcc_loop_uses_legacy_xy_storage_mapping_without_manifest():
    assert chern_post._wcc_direction_from_loop("b1", grid_order="legacy_xy") == "ky"
    assert chern_post._wcc_direction_from_loop("b2", grid_order="legacy_xy") == "kx"


def test_wcc_loop_uses_canonical_ij_storage_mapping_with_manifest():
    assert chern_post._wcc_direction_from_loop("b1", grid_order="ij") == "kx"
    assert chern_post._wcc_direction_from_loop("b2", grid_order="ij") == "ky"


def test_wcc_g_vector_lookup_resolves_relative_topology_dir(tmp_path, monkeypatch):
    qshell_dir = tmp_path / "Q_shell_8"
    topo_dir = qshell_dir / "topo"
    topo_dir.mkdir(parents=True)
    group1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    group2 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=float)
    np.save(qshell_dir / "g_vec_list_8_Gamma_1layer.npy", group1)
    np.save(qshell_dir / "g_vec_list_8_Gamma_2layer.npy", group2)

    monkeypatch.chdir(topo_dir)
    groups, files = chern_post._load_wcc_g_vectors(".", "Gamma")

    assert files == [
        str(qshell_dir / "g_vec_list_8_Gamma_1layer.npy"),
        str(qshell_dir / "g_vec_list_8_Gamma_2layer.npy"),
    ]
    assert np.allclose(groups[0], group1)
    assert np.allclose(groups[1], group2)


def test_topology_grid_output_dir_separates_nondefault_ranges(tmp_path):
    config = {
        "compute": {"num_chern": 31},
        "topology": {
            "mesh": {
                "n_b1": 21,
                "n_b2": 41,
                "range_b1": [0.0, 0.5],
                "range_b2": [-0.5, 0.5],
            }
        },
    }
    output_dir = tmp_path / "topology"

    grid_dir = chern_post._topology_grid_output_dir(output_dir, config)

    assert grid_dir == output_dir / "grid21x41_b1_0p0_0p5_b2_m0p5_0p5"


def test_topology_task_output_dir_places_grid_under_case_topology_dir(tmp_path):
    config = {
        "topology": {
            "mesh": {
                "n_b1": 31,
                "n_b2": 31,
                "range_b1": [-0.5, 0.5],
                "range_b2": [-0.5, 0.5],
            }
        },
    }
    case_dir = tmp_path / "Gamma" / "q04"

    assert chern_post._topology_task_output_dir(case_dir, config) == case_dir / "topology"
    assert chern_post._topology_grid_output_dir(
        chern_post._topology_task_output_dir(case_dir, config),
        config,
    ) == case_dir / "topology" / "grid31x31_b1_m0p5_0p5_b2_m0p5_0p5"


def test_topology_tasks_resolve_named_band_sets():
    cfg = {
        "topology": {
            "bands": {
                "vbm2": {"sector": "valence", "indices": [-1, -2]},
                "cbm2": {"sector": "conduction", "indices": [0, 1]},
            },
            "berry_curvature": [{"bands": "vbm2"}],
            "quantum_geometry": [{"bands": "cbm2"}],
            "wcc": [{"bands": "vbm2", "loop": "b2"}],
        }
    }

    band_tasks, wcc_tasks = chern_post._resolve_topology_tasks(cfg)

    assert {"label": "vbm2", "band_type": "VBM", "indices": [-1, -2], "bc": True, "qgt": False} in band_tasks
    assert {"label": "cbm2", "band_type": "CBM", "indices": [0, 1], "bc": False, "qgt": True} in band_tasks
    assert wcc_tasks == [{"label": "vbm2", "band_type": "VBM", "indices": [-1, -2], "loop": "b2"}]




def test_topology_tasks_reject_removed_bandsets_and_observables():
    cfg = {
        "topology": {
            "bandsets": {
                "vbm2": {"sector": "valence", "indices": [-1, -2]},
                "cbm2": {"sector": "conduction", "indices": [0, 1]},
            },
            "observables": {
                "berry_curvature": ["vbm2"],
                "quantum_geometry": ["cbm2"],
                "wcc": [{"bands": "vbm2", "loop": "b1"}],
            },
        }
    }

    with pytest.raises(ValueError, match="bandsets|observables"):
        chern_post._resolve_topology_tasks(cfg)


def test_topology_task_dispatch_defaults_to_config_valley(monkeypatch, tmp_path):
    cfg = {
        "topology": {
            "valley": "Gamma",
            "mesh": {"n_b1": 3, "n_b2": 3, "range_b1": [-0.5, 0.5], "range_b2": [-0.5, 0.5]},
            "bands": {"vbm2": {"sector": "valence", "indices": [-1, -2]}},
            "berry_curvature": [{"bands": "vbm2"}],
        },
    }
    calls = []

    def fake_main(argv=None, **_kwargs):
        calls.append(list(argv))

    monkeypatch.setattr(chern_post, "main", fake_main)

    args = SimpleNamespace(output_dir=str(tmp_path), valley=None, wcc_sewing="auto", wcc_sewing_atol=1.0e-6)
    chern_post._dispatch_topology_tasks("config.yaml", args, cfg)

    assert calls
    assert calls[0][calls[0].index("--valley") + 1] == "5"
    assert calls[0][calls[0].index("--topology-band-label") + 1] == "vbm2"


def test_topology_task_dispatch_explicit_valley_overrides_config(monkeypatch, tmp_path):
    cfg = {
        "compute": {"valley": 5},
        "topology": {
            "mesh": {"n_b1": 3, "n_b2": 3, "range_b1": [-0.5, 0.5], "range_b2": [-0.5, 0.5]},
            "bands": {"vbm2": {"sector": "valence", "indices": [-1, -2]}},
            "berry_curvature": [{"bands": "vbm2"}],
        },
    }
    calls = []

    def fake_main(argv=None, **_kwargs):
        calls.append(list(argv))

    monkeypatch.setattr(chern_post, "main", fake_main)

    args = SimpleNamespace(output_dir=str(tmp_path), valley=31, wcc_sewing="auto", wcc_sewing_atol=1.0e-6)
    chern_post._dispatch_topology_tasks("config.yaml", args, cfg)

    assert calls
    assert calls[0][calls[0].index("--valley") + 1] == "31"
    assert calls[0][calls[0].index("--topology-band-label") + 1] == "vbm2"


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


def test_square_generate_kmesh_uses_ij_flatten_order():
    calculator = BandStructureCalculator.__new__(BandStructureCalculator)

    kpoints = calculator.generate_kmesh(3, 3)

    assert kpoints.shape == (9, 3)
    assert np.allclose(kpoints[0], np.array([-0.5, -0.5, 0.0]))
    assert np.allclose(kpoints[1], np.array([-0.5, 0.0, 0.0]))
    assert np.allclose(kpoints[2], np.array([-0.5, 0.5, 0.0]))
    assert np.allclose(kpoints[3], np.array([0.0, -0.5, 0.0]))


def test_rotation_helpers_raise_value_error_instead_of_successful_system_exit():
    from tapw.geometry import rotations

    with pytest.raises(ValueError, match="p orbital"):
        rotations.get_orb_map_p(rotations.x, rotations.y, rotations.z, ndim=2, orbi=1)


def test_chern_post_resolves_relative_input_file_like_tapw_run(tmp_path, monkeypatch):
    config_dir = tmp_path / "case"
    config_dir.mkdir()
    input_file = config_dir / "openmx.dat"
    input_file.write_text(
        "Atoms.UnitVectors.Unit Ang\n<Atoms.UnitVectors\n1 0 0\n0 1 0\n0 0 1\nAtoms.UnitVectors>\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "run-output"
    output_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                "  input_file: openmx.dat",
                f"  output_dir: {output_dir}",
                "compute:",
                "  num_chern: 2",
            ]
        ),
        encoding="utf-8",
    )

    seen = {}

    def fake_parse_lattice(path):
        seen["openmx_path"] = path
        raise RuntimeError("stop after path resolution")

    monkeypatch.setattr(chern_post, "parse_lattice_vectors_from_openmx", fake_parse_lattice)

    with pytest.raises(SystemExit):
        chern_post.main(["--config", str(config_path), "--output-dir", str(output_dir), "-b", "0"])

    assert seen["openmx_path"] == str(input_file.resolve())


def test_chern_post_rejects_generalized_eigenvectors_before_file_lookup(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                "  input_file: missing-openmx.dat",
                "compute:",
                "  ge: true",
                "  num_chern: 2",
            ]
        ),
        encoding="utf-8",
    )

    def fail_if_called(path):
        raise AssertionError("ge=true should fail before parsing lattice vectors")

    monkeypatch.setattr(chern_post, "parse_lattice_vectors_from_openmx", fail_if_called)

    with pytest.raises(SystemExit) as excinfo:
        chern_post.main(["--config", str(config_path), "--output-dir", str(tmp_path), "-b", "0"])

    assert excinfo.value.code == 1


def test_chern_post_config_tasks_write_canonical_topology_artifact_names(tmp_path):
    config_dir = tmp_path / "case"
    config_dir.mkdir()
    input_file = config_dir / "openmx.dat"
    input_file.write_text(
        "Atoms.UnitVectors.Unit Ang\n<Atoms.UnitVectors\n1 0 0\n0 1 0\n0 0 1\nAtoms.UnitVectors>\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    grid_dir = output_root / "grid4x4_b1_m0p5_0p5_b2_m0p5_0p5"
    grid_dir.mkdir(parents=True)

    vec_grid = np.zeros((4, 4, 2, 2), dtype=np.complex128)
    vec_grid[:, :, 0, 0] = 1.0
    vec_grid[:, :, 1, 1] = 1.0
    wavefunction_path = grid_dir / "wavefunctions_vbm.npy"
    np.save(wavefunction_path, vec_grid.reshape(16, 2, 2))
    from tapw import __version__ as tapw_version
    from tapw.identity import IDENTITY_SCHEMA, hash_file, hash_mapping
    from tapw.symmetry.periodic_gauge import (
        BOUNDARY_SEWING_SCHEMA,
        BOUNDARY_SEWING_SCHEMA_VERSION,
        save_boundary_operators,
    )

    wavefunction_hashes = {wavefunction_path.name: hash_file(wavefunction_path)}
    save_boundary_operators(
        grid_dir / "boundary_sewing.npz",
        {
            "b1": scipy.sparse.identity(2, dtype=np.complex128, format="csr"),
            "b2": scipy.sparse.identity(2, dtype=np.complex128, format="csr"),
        },
        {
            "schema": BOUNDARY_SEWING_SCHEMA,
            "identity_schema": IDENTITY_SCHEMA,
            "input_hash": hash_mapping({"wavefunction_hashes": wavefunction_hashes}),
            "config_hash": "config-a",
            "basis_hash": "basis-a",
            "package_version": tapw_version,
            "schema_version": BOUNDARY_SEWING_SCHEMA_VERSION,
            "matrix_dimension": 2,
            "reciprocal_basis": [[2.0 * np.pi, 0.0], [0.0, 2.0 * np.pi]],
            "wavefunction_hashes": wavefunction_hashes,
        },
    )

    config_path = config_dir / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                "  input_file: openmx.dat",
                "twist:",
                "  spin: false",
                "topology:",
                "  mesh:",
                "    n_b1: 4",
                "    n_b2: 4",
                "    range_b1: [-0.5, 0.5]",
                "    range_b2: [-0.5, 0.5]",
                "  bands:",
                "    vbm2:",
                "      sector: valence",
                "      indices: [-1, -2]",
                "  berry_curvature:",
                "    - bands: vbm2",
                "  quantum_geometry:",
                "    - bands: vbm2",
                "  wcc:",
                "    - bands: vbm2",
                "      loop: b2",
            ]
        ),
        encoding="utf-8",
    )

    chern_post.main(["--config", str(config_path), "--output-dir", str(output_root)])

    assert (grid_dir / "berry_curvature_vbm2.txt").is_file()
    assert (grid_dir / "berry_curvature_vbm2.pdf").is_file()
    assert (grid_dir / "quantum_geometry_vbm2.txt").is_file()
    assert (grid_dir / "quantum_geometry_vbm2.pdf").is_file()
    assert (grid_dir / "wcc_vbm2_loop_b2.txt").is_file()
    assert (grid_dir / "wcc_vbm2_loop_b2.pdf").is_file()
    assert (grid_dir / "chern_summary.json").is_file()
    assert not (grid_dir / "manifest.json").exists()
    assert not (grid_dir / "chern_vbm2.txt").exists()
    assert not list(grid_dir.glob("bc_bands_*"))
    assert not list(grid_dir.glob("qgt_bands_*"))


def test_single_band_berry_flux_supports_rectangular_grids():
    band_grid = np.zeros((3, 5, 2, 1), dtype=np.complex128)
    band_grid[..., 0, 0] = 1.0

    berry_flux = chern_post.compute_berry_flux_single_band(band_grid, 0)

    assert berry_flux.shape == (2, 4)
    assert np.allclose(berry_flux, 0.0)


def test_single_band_berry_flux_changes_sign_when_grid_orientation_is_swapped():
    band_grid = _random_orthonormal_grid(seed=42, shape=(4, 5), dim_h=3, num_bands=1)

    berry_flux = chern_post.compute_berry_flux_single_band(band_grid, 0)
    swapped_flux = chern_post.compute_berry_flux_single_band(band_grid.transpose(1, 0, 2, 3), 0)

    assert np.max(np.abs(berry_flux)) > 1.0e-6
    assert np.allclose(swapped_flux, -berry_flux.T, atol=1.0e-12)


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


def test_wcc_boundary_shift_uses_positive_loop_closing_vector():
    basis = np.array([[2.0, 0.5], [0.0, 3.0]])

    assert np.allclose(chern_post._wcc_boundary_shift("ky", basis), [2.0, 0.5])
    assert np.allclose(chern_post._wcc_boundary_shift("kx", basis), [0.0, 3.0])


def test_wcc_boundary_shift_prefers_g_vector_basis_for_sewing():
    g_vectors = [
        np.array(
            [
                [0.0, 0.0],
                [0.0, 0.28908146],
                [0.0, 0.57816292],
                [-0.25035189, -0.14454073],
                [-0.25035189, 0.14454073],
                [0.25035189, 0.14454073],
            ]
        ),
        np.array(
            [
                [0.0, 0.0],
                [0.0, 0.28908146],
                [0.0, 0.57816292],
                [-0.25035189, -0.14454073],
                [-0.25035189, 0.14454073],
                [0.25035189, 0.14454073],
            ]
        ),
    ]
    openmx_basis = np.array([[0.25035189, 0.14454073], [0.0, 0.28908146]])

    shift = chern_post._wcc_boundary_shift("ky", openmx_basis, g_vectors)

    assert np.allclose(shift, [0.25035189, 0.14454073])


def test_chern_post_wcc_sewing_defaults_to_auto():
    parser = chern_post.build_parser()

    args = parser.parse_args(["--config", "config.yaml", "-wb", "-1", "-2"])

    assert args.wcc_sewing == "auto"


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


def test_boundary_sewing_shifts_g_blocks_for_wilson_loop():
    g_vectors = [np.array([[0.0, 0.0], [1.0, 0.0]])]
    sewing = chern_post.build_boundary_sewing(g_vectors, np.array([-1.0, 0.0]), dim_h=4)
    start = np.array([[1.0], [0.0], [0.0], [0.0]], dtype=np.complex128)
    end = np.array([[0.0], [0.0], [1.0], [0.0]], dtype=np.complex128)

    naked_overlap = np.conj(start.T) @ end
    sewn_overlap = np.conj(start.T) @ chern_post.apply_boundary_sewing(end, sewing)

    assert np.allclose(naked_overlap, [[0.0]])
    assert np.allclose(sewn_overlap, [[1.0]])
    assert sewing.matched_blocks == 1
    assert sewing.missing_blocks == 1


def test_boundary_sewing_respects_spin_block_row_order():
    g_vectors = [np.array([[0.0, 0.0], [1.0, 0.0]])]
    sewing = chern_post.build_boundary_sewing(
        g_vectors,
        np.array([-1.0, 0.0]),
        dim_h=8,
        spin_blocks=2,
    )
    vec = np.zeros((8, 1), dtype=np.complex128)
    vec[6, 0] = 1.0

    sewn = chern_post.apply_boundary_sewing(vec, sewing)

    assert sewn[4, 0] == 1.0
    assert np.count_nonzero(sewn) == 1
    assert sewing.matched_blocks == 2
    assert sewing.missing_blocks == 2


def test_canonical_boundary_operator_is_bound_to_wavefunction_file(tmp_path):
    from tapw import __version__ as tapw_version
    from tapw.identity import IDENTITY_SCHEMA, hash_file, hash_mapping
    from tapw.symmetry.periodic_gauge import (
        BOUNDARY_SEWING_SCHEMA,
        BOUNDARY_SEWING_SCHEMA_VERSION,
        save_boundary_operators,
    )

    wavefunction_path = tmp_path / "wavefunctions_vbm.npy"
    np.save(wavefunction_path, np.eye(2, dtype=np.complex128)[None, :, :])
    wavefunction_hashes = {wavefunction_path.name: hash_file(wavefunction_path)}
    operators = {
        "b1": scipy.sparse.diags([1.0, -1.0], dtype=np.complex128, format="csr"),
        "b2": scipy.sparse.identity(2, dtype=np.complex128, format="csr"),
    }
    reciprocal_basis = np.array([[1.0, 0.0], [0.2, 0.8]])
    save_boundary_operators(
        tmp_path / "boundary_sewing.npz",
        operators,
        {
            "schema": BOUNDARY_SEWING_SCHEMA,
            "identity_schema": IDENTITY_SCHEMA,
            "input_hash": hash_mapping({"wavefunction_hashes": wavefunction_hashes}),
            "config_hash": "config-a",
            "basis_hash": "basis-a",
            "package_version": tapw_version,
            "schema_version": BOUNDARY_SEWING_SCHEMA_VERSION,
            "matrix_dimension": 2,
            "reciprocal_basis": reciprocal_basis.tolist(),
            "wavefunction_hashes": wavefunction_hashes,
        },
    )

    loaded = chern_post._load_required_boundary_operator(
        tmp_path,
        loop="b1",
        wavefunction_path=wavefunction_path,
        reciprocal_basis=reciprocal_basis,
        expected_dim=2,
    )

    np.testing.assert_allclose(loaded.toarray(), operators["b1"].toarray())
    np.save(wavefunction_path, np.ones((1, 2, 2), dtype=np.complex128))
    with pytest.raises(ValueError, match="wavefunction hash"):
        chern_post._load_required_boundary_operator(
            tmp_path,
            loop="b1",
            wavefunction_path=wavefunction_path,
            reciprocal_basis=reciprocal_basis,
            expected_dim=2,
        )


def test_wilson_loop_rejects_rank_deficient_boundary_link():
    vectors = np.zeros((2, 2, 1), dtype=np.complex128)
    vectors[:, 0, 0] = 1.0
    boundary = scipy.sparse.diags([0.0, 1.0], dtype=np.complex128, format="csr")

    with pytest.raises(ValueError, match="boundary link is rank deficient"):
        chern_post.wilson_loop(
            vectors,
            boundary_sewing=boundary,
            boundary_singular_value_tol=1.0e-8,
        )


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

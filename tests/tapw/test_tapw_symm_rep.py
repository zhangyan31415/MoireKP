import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse
import yaml


def _write_legacy_inputs(tmp_path: Path):
    band_dir = tmp_path / "band"
    symmetry_dir = tmp_path / "symmetry"
    rawh_dir = symmetry_dir / "representations" / "Gamma"
    band_dir.mkdir(parents=True)
    rawh_dir.mkdir(parents=True)

    hamk = np.diag([-3.0, -1.0, -1.0, 1.0, 1.0, 3.0]).astype(np.complex128)
    c2 = np.diag([1.0, -1.0, -1.0, 1.0, 1.0, -1.0]).astype(np.complex128)
    tr = np.eye(6, dtype=np.complex128)

    np.save(band_dir / "hamk_Gamma_valley.npy", hamk)
    scipy.sparse.save_npz(rawh_dir / "C2_rawH.npz", scipy.sparse.csr_matrix(c2))
    scipy.sparse.save_npz(rawh_dir / "TR_rawH.npz", scipy.sparse.csr_matrix(tr))
    return band_dir, symmetry_dir


def _write_release_symm_rep_config(tmp_path: Path, *, output_root: Path | None = None) -> Path:
    config_path = tmp_path / "config.yaml"
    payload = {
        "case": {"output_root": str(output_root or (tmp_path / "outputs"))},
        "twist": {"bravais": "hex", "twist_index_m": 3, "twist_layer": [1, 1], "spin": True},
        "paths": {
            "H_file": str(tmp_path / "H.dat"),
            "S_file": str(tmp_path / "S.dat"),
            "input_file": str(tmp_path / "openmx.dat"),
        },
        "symmetry": {
            "valley": "Gamma",
            "q_shell": 4,
            "efermi": -4.055365,
            "tolerance": 0.02,
            "spglib_symprec": 0.05,
            "representation": {
                "points": {
                    "Gamma": [0.0, 0.0],
                    "M": [0.5, 0.0],
                    "K": [1.0 / 3.0, 1.0 / 3.0],
                },
                "valence_count": 12,
                "conduction_count": 8,
                "num_bands": 50,
                "degeneracy_tol": 1.0e-3,
            },
        },
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_symm_rep_groups_degenerate_blocks_and_writes_outputs(tmp_path):
    from tapw.workflows.symm_rep import run_symm_rep

    band_dir, symmetry_dir = _write_legacy_inputs(tmp_path)
    output_dir = tmp_path / "rep"

    result = run_symm_rep(
        band_dir=band_dir,
        symmetry_dir=symmetry_dir,
        output_dir=output_dir,
        fermi_energy=0.0,
        valence_count=3,
        conduction_count=3,
        degeneracy_tol=1.0e-6,
        overwrite=False,
    )

    assert result.output_dir == output_dir
    assert (output_dir / "high_symmetry_wavefunctions.npz").is_file()
    assert (output_dir / "band_representations.npz").is_file()
    assert (output_dir / "characters.csv").is_file()
    assert (output_dir / "bands.csv").is_file()
    assert (output_dir / "summary.md").is_file()

    with np.load(output_dir / "high_symmetry_wavefunctions.npz", allow_pickle=False) as payload:
        assert payload["points"].tolist() == ["Gamma"]
        assert np.allclose(payload["Gamma_energies"], [-3.0, -1.0, -1.0, 1.0, 1.0, 3.0])
        assert payload["Gamma_eigenvectors"].shape == (6, 6)
        assert payload["Gamma_selected_indices"].tolist() == [0, 1, 2, 3, 4, 5]

    rows = list(csv.DictReader((output_dir / "characters.csv").open(newline="", encoding="utf-8")))
    c2_valence_doublet = [
        row
        for row in rows
        if row["point"] == "Gamma"
        and row["sector"] == "valence"
        and row["band_indices"] == "1 2"
        and row["operation"] == "C2"
    ]
    assert len(c2_valence_doublet) == 1
    assert float(c2_valence_doublet[0]["trace_real"]) == pytest.approx(-2.0)
    assert float(c2_valence_doublet[0]["unitarity_residual"]) == pytest.approx(0.0)

    bands = list(csv.DictReader((output_dir / "bands.csv").open(newline="", encoding="utf-8")))
    assert [row["spin_label"] for row in bands if row["sector"] == "valence"] == ["up", "up", "up"]
    assert [row["spin_label"] for row in bands if row["sector"] == "conduction"] == ["down", "down", "down"]

    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "## Gamma" in summary
    assert "### Valence" in summary
    assert "### Conduction" in summary
    assert "### Symmetry Representations" in summary
    assert "### D_block Matrices" not in summary
    assert "| valence | 1 | 1 2 | -1.000000, -1.000000 | C2 |  | (-1[↑1.000,↓0.000], -1[↑1.000,↓0.000]) | `" in summary
    assert "| valence | 1 | 1 2 | -1.000000, -1.000000 | C2 | false | -2.000000 | 0 |" in summary


def test_select_band_indices_returns_empty_sector_for_zero_count():
    from tapw.workflows.symm_rep import select_band_indices

    selected = select_band_indices(
        np.array([-3.0, -2.0, -1.0, 0.5, 1.0]),
        0.0,
        valence_count=0,
        conduction_count=2,
    )

    assert selected["valence"].tolist() == []
    assert selected["conduction"].tolist() == [3, 4]


def test_symm_rep_phase_label_uses_spinful_c3_omega():
    from tapw.workflows.symm_rep import _phase_label

    assert _phase_label(np.exp(1.0j * np.pi / 3.0)) == "ω"
    assert _phase_label(np.exp(2.0j * np.pi / 3.0)) == "ω²"
    assert _phase_label(np.exp(-2.0j * np.pi / 3.0)) == "ω⁴"
    assert _phase_label(np.exp(-1.0j * np.pi / 3.0)) == "ω⁵"
    assert _phase_label(-1.0 + 0.0j) == "-1"


def test_symm_rep_polar_unitary_part_removes_projection_scale():
    from tapw.workflows.symm_rep import _unitarity_residual, polar_unitary_part

    raw = np.diag([0.8, 1.2]).astype(np.complex128)
    unitary, distance = polar_unitary_part(raw)

    assert np.allclose(unitary, np.eye(2))
    assert _unitarity_residual(unitary) == pytest.approx(0.0)
    assert _unitarity_residual(raw) > 0.0
    assert distance > 0.0


def test_symm_rep_uses_configured_slice_for_3d_hamk(tmp_path):
    from tapw.workflows.symm_rep import run_symm_rep

    band_dir, symmetry_dir = _write_legacy_inputs(tmp_path)
    hamk_stack = np.stack(
        [
            np.diag([-2.0, -1.0, 1.0, 2.0, 3.0, 4.0]),
            np.diag([-4.0, -3.0, 3.0, 4.0, 5.0, 6.0]),
        ]
    ).astype(np.complex128)
    np.save(band_dir / "hamk_Gamma_valley.npy", hamk_stack)

    output_dir = tmp_path / "rep-3d"
    run_symm_rep(
        band_dir=band_dir,
        symmetry_dir=symmetry_dir,
        output_dir=output_dir,
        fermi_energy=0.0,
        valence_count=2,
        conduction_count=2,
        hamiltonian_index=1,
    )

    with np.load(output_dir / "high_symmetry_wavefunctions.npz", allow_pickle=False) as payload:
        assert payload["Gamma_hamiltonian_index"].item() == 1
        assert payload["Gamma_hamiltonian_count"].item() == 2
        assert np.allclose(payload["Gamma_energies"], [-4.0, -3.0, 3.0, 4.0, 5.0, 6.0])

    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "Selected Hamiltonian slice: `1` of `2`." in summary


def test_symm_rep_loads_packed_canonical_representations(tmp_path):
    from tapw.workflows.symm_rep import run_symm_rep

    band_dir, symmetry_dir = _write_legacy_inputs(tmp_path)
    legacy_dir = symmetry_dir / "representations"
    for path in legacy_dir.rglob("*_rawH.npz"):
        path.unlink()
    c2 = scipy.sparse.csr_matrix(np.diag([1.0, -1.0, -1.0, 1.0, 1.0, -1.0]).astype(np.complex128))
    np.savez_compressed(
        symmetry_dir / "representations.npz",
        C2_data=c2.data,
        C2_indices=c2.indices,
        C2_indptr=c2.indptr,
        C2_shape=np.array(c2.shape, dtype=int),
    )

    output_dir = tmp_path / "rep-packed"
    run_symm_rep(
        band_dir=band_dir,
        symmetry_dir=symmetry_dir,
        output_dir=output_dir,
        fermi_energy=0.0,
        valence_count=3,
        conduction_count=3,
    )

    rows = list(csv.DictReader((output_dir / "characters.csv").open(newline="", encoding="utf-8")))
    assert {row["operation"] for row in rows} == {"C2"}
    assert {row["antiunitary"] for row in rows} == {"false"}


def test_unscoped_packed_operations_apply_only_to_profile_point(tmp_path):
    from tapw.workflows.symm_rep import load_raw_h_symmetry_operations

    symmetry_dir = tmp_path / "outputs" / "Gamma" / "q04" / "symmetry"
    symmetry_dir.mkdir(parents=True)
    c2 = scipy.sparse.identity(2, dtype=np.complex128, format="csr")
    np.savez_compressed(
        symmetry_dir / "representations.npz",
        C2_data=c2.data,
        C2_indices=c2.indices,
        C2_indptr=c2.indptr,
        C2_shape=np.array(c2.shape, dtype=int),
    )

    operations = load_raw_h_symmetry_operations(symmetry_dir, {"Gamma", "K"})

    assert [op.operation for op in operations["Gamma"]] == ["C2"]
    assert operations["K"] == []


def test_manifest_source_actions_are_available_for_each_requested_point(tmp_path):
    from tapw.workflows.symm_rep import load_raw_h_symmetry_operations

    symmetry_dir = tmp_path / "outputs" / "Gamma" / "q04" / "symmetry"
    representations_dir = symmetry_dir / "representations"
    representations_dir.mkdir(parents=True)
    c3 = scipy.sparse.identity(2, dtype=np.complex128, format="csr")
    np.savez_compressed(
        symmetry_dir / "representations.npz",
        C3z_data=c3.data,
        C3z_indices=c3.indices,
        C3z_indptr=c3.indptr,
        C3z_shape=np.array(c3.shape, dtype=int),
    )
    (representations_dir / "manifest.json").write_text(
        json.dumps(
            {
                "matrices": [
                    {
                        "operation": "C3z",
                        "antiunitary": False,
                        "packed_matrix_file": "../representations.npz",
                        "packed_matrix_key": "C3z",
                        "source_action": {
                            "antiunitary": False,
                            "k_map": {"type": "rotation", "angle_deg": 120.0},
                            "q_map": {"type": "rotation", "angle_deg": 120.0},
                            "sector_map": "identity",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    operations = load_raw_h_symmetry_operations(symmetry_dir, {"Gamma", "K"})

    assert [op.operation for op in operations["Gamma"]] == ["C3z"]
    assert [op.operation for op in operations["K"]] == ["C3z"]
    assert operations["K"][0].source_action["k_map"]["type"] == "rotation"


def test_packed_metadata_json_is_enough_for_source_actions(tmp_path):
    from tapw.workflows.symm_rep import load_raw_h_symmetry_operations

    symmetry_dir = tmp_path / "outputs" / "Gamma" / "q04" / "symmetry"
    symmetry_dir.mkdir(parents=True)
    c3 = scipy.sparse.identity(2, dtype=np.complex128, format="csr")
    metadata = {
        "schema": "tapw.raw_h_representations.v1",
        "storage": "scipy_csr_components_v1",
        "basis": "tapw_projected",
        "basis_order": "spin_outermost; group -> g_index -> atom_type -> orbital",
        "matrices": [
            {
                "key": "C3z",
                "operation": "C3z",
                "antiunitary": False,
                "axis_deg": None,
                "source_valley": "Gamma",
                "target_valley": "Gamma",
                "role": "internal",
                "supported": True,
                "residual_H_raw": 0.0,
                "status": "exact",
                "source_action": {
                    "antiunitary": False,
                    "k_map": {"type": "rotation", "angle_deg": 120.0},
                    "q_map": {"type": "rotation", "angle_deg": 120.0},
                    "sector_map": "identity",
                },
            }
        ],
    }
    np.savez_compressed(
        symmetry_dir / "representations.npz",
        C3z_data=c3.data,
        C3z_indices=c3.indices,
        C3z_indptr=c3.indptr,
        C3z_shape=np.array(c3.shape, dtype=int),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True), dtype=str),
    )

    operations = load_raw_h_symmetry_operations(symmetry_dir, {"Gamma", "K"})

    assert [op.operation for op in operations["Gamma"]] == ["C3z"]
    assert [op.operation for op in operations["K"]] == ["C3z"]
    assert operations["K"][0].source_action["k_map"] == {"type": "rotation", "angle_deg": 120.0}


def test_partial_reciprocal_sewing_allows_finite_g_truncation():
    from tapw.workflows.symm_rep import _sewing_matrix_from_shift

    matrix = _sewing_matrix_from_shift(
        g_vectors_by_group=[np.array([[0.0, 0.0], [1.0, 0.0]])],
        reciprocal_basis=np.eye(2),
        reciprocal_shift_coeffs=np.array([1, 0]),
        dim=2,
        spin_blocks=1,
    )

    assert matrix.shape == (2, 2)
    assert matrix[1, 0] == pytest.approx(1.0)
    assert matrix[0, 1] == pytest.approx(0.0)


def test_symm_rep_sewing_default_tolerance_allows_large_moire_roundoff():
    from tapw.workflows.symm_rep import _sewing_matrix_from_shift

    matrix = _sewing_matrix_from_shift(
        g_vectors_by_group=[np.array([[0.0, 0.0], [1.0, 0.0]])],
        reciprocal_basis=np.array([[1.0004, 0.0], [0.0, 1.0]]),
        reciprocal_shift_coeffs=np.array([1, 0]),
        dim=2,
        spin_blocks=1,
    )
    too_loose_matrix = _sewing_matrix_from_shift(
        g_vectors_by_group=[np.array([[0.0, 0.0], [1.0, 0.0]])],
        reciprocal_basis=np.array([[1.0006, 0.0], [0.0, 1.0]]),
        reciprocal_shift_coeffs=np.array([1, 0]),
        dim=2,
        spin_blocks=1,
    )

    assert matrix[1, 0] == pytest.approx(1.0)
    assert too_loose_matrix[1, 0] == pytest.approx(0.0)


def test_config_sewing_context_snaps_reciprocal_basis_to_actual_g_vectors():
    from tapw import symm_rep
    from tapw.chern_post import build_boundary_sewing

    g_vectors = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    calculator = SimpleNamespace(
        TAPW_parameters=SimpleNamespace(g_vec_list_K1=g_vectors, g_vec_list_K2=g_vectors.copy()),
        structure=SimpleNamespace(
            reciprocal_Tmat=np.array(
                [
                    [1.0001, 0.0, 0.0],
                    [0.0, 0.9999, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
        ),
    )
    config = SimpleNamespace(twist=SimpleNamespace(spin=True))

    context = symm_rep._sewing_context_from_calculator(calculator, config)

    np.testing.assert_allclose(context["reciprocal_basis"], np.eye(2), atol=1.0e-12)
    sewing = build_boundary_sewing(
        context["g_vectors_by_group"],
        np.array([1.0, 0.0]) @ context["reciprocal_basis"],
        dim_h=16,
        spin_blocks=context["spin_blocks"],
    )
    assert sewing.matched_blocks == 8


def test_symm_rep_falls_back_to_saved_vec_and_band_files_without_hamk(tmp_path):
    from tapw.workflows.symm_rep import run_symm_rep

    band_dir, symmetry_dir = _write_legacy_inputs(tmp_path)
    (band_dir / "hamk_Gamma_valley.npy").unlink()
    vbm_energies = np.array([[-4.0, -3.0, -2.0, -1.0], [-4.5, -3.5, -2.5, -1.5]])
    cbm_energies = np.array([[1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5]])
    basis = np.eye(6, dtype=np.complex128)
    vbm_vec = np.stack([basis[:, [0, 1, 2, 3]], basis[:, [1, 2, 3, 4]]])
    cbm_vec = np.stack([basis[:, [4, 5, 0, 1]], basis[:, [5, 0, 1, 2]]])
    np.savetxt(band_dir / "band_VBM_Gamma_valley.txt", vbm_energies)
    np.savetxt(band_dir / "band_CBM_Gamma_valley.txt", cbm_energies)
    np.save(band_dir / "vec_VBM_Gamma_valley.npy", vbm_vec)
    np.save(band_dir / "vec_CBM_Gamma_valley.npy", cbm_vec)

    output_dir = tmp_path / "rep-saved"
    result = run_symm_rep(
        band_dir=band_dir,
        symmetry_dir=symmetry_dir,
        output_dir=output_dir,
        valence_count=2,
        conduction_count=2,
        hamiltonian_index=1,
    )

    assert result.fermi_energy == pytest.approx(0.0)
    with np.load(output_dir / "high_symmetry_wavefunctions.npz", allow_pickle=False) as payload:
        assert payload["Gamma_source_kind"].item() == "saved_wavefunctions"
        assert payload["Gamma_hamiltonian_index"].item() == 1
        assert payload["Gamma_selected_indices"].tolist() == [2, 3, 0, 1]
        assert np.allclose(payload["Gamma_energies"], [-2.5, -1.5, 1.5, 2.5])
        assert payload["Gamma_eigenvectors"].shape == (6, 4)

    bands = list(csv.DictReader((output_dir / "bands.csv").open(newline="", encoding="utf-8")))
    assert [(row["sector"], row["band_index"], row["energy"]) for row in bands] == [
        ("valence", "3", "-1.500000000000"),
        ("valence", "2", "-2.500000000000"),
        ("conduction", "0", "1.500000000000"),
        ("conduction", "1", "2.500000000000"),
    ]
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "Source: saved wavefunctions" in summary


def test_symm_rep_can_filter_points(tmp_path):
    from tapw.workflows.symm_rep import run_symm_rep

    band_dir, symmetry_dir = _write_legacy_inputs(tmp_path)
    np.save(band_dir / "hamk_M1_valley.npy", np.diag([-2.0, -1.0, 1.0, 2.0, 3.0, 4.0]))

    output_dir = tmp_path / "rep-filtered"
    run_symm_rep(
        band_dir=band_dir,
        symmetry_dir=symmetry_dir,
        output_dir=output_dir,
        fermi_energy=0.0,
        valence_count=2,
        conduction_count=2,
        points=["Gamma"],
    )

    with np.load(output_dir / "high_symmetry_wavefunctions.npz", allow_pickle=False) as payload:
        assert payload["points"].tolist() == ["Gamma"]
        assert "M1_energies" not in payload.files


def test_symm_rep_writes_from_in_memory_point_sources(tmp_path):
    from tapw.workflows.symm_rep import run_symm_rep_from_point_sources

    _band_dir, symmetry_dir = _write_legacy_inputs(tmp_path)
    basis = np.eye(6, dtype=np.complex128)
    point_sources = {
        "Gamma": {
            "source_kind": "computed_config_points",
            "source_index": 0,
            "source_count": 1,
            "sectors": {
                "valence": {
                    "energies": np.array([-2.0, -1.0]),
                    "vectors": basis[:, [0, 1]],
                    "band_indices": np.array([0, 1]),
                },
                "conduction": {
                    "energies": np.array([1.0, 2.0]),
                    "vectors": basis[:, [3, 4]],
                    "band_indices": np.array([2, 3]),
                },
            },
        }
    }

    output_dir = tmp_path / "rep-memory"
    run_symm_rep_from_point_sources(
        point_sources=point_sources,
        symmetry_dir=symmetry_dir,
        output_dir=output_dir,
        fermi_energy=0.0,
        degeneracy_tol=1.0e-6,
    )

    assert (output_dir / "summary.md").is_file()
    with np.load(output_dir / "high_symmetry_wavefunctions.npz", allow_pickle=False) as payload:
        assert payload["Gamma_source_kind"].item() == "computed_config_points"
        assert payload["Gamma_selected_indices"].tolist() == [0, 1, 2, 3]


def test_symm_rep_diagonalizes_sz_inside_degenerate_blocks(tmp_path):
    from tapw.workflows.symm_rep import run_symm_rep_from_point_sources

    symmetry_dir = tmp_path / "symmetry"
    rawh_dir = symmetry_dir / "representations" / "Gamma"
    rawh_dir.mkdir(parents=True)
    scipy.sparse.save_npz(
        rawh_dir / "C2_rawH.npz",
        scipy.sparse.csr_matrix(np.diag([1.0, -1.0]).astype(np.complex128)),
    )

    mixed_vectors = np.array(
        [
            [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
            [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)],
        ],
        dtype=np.complex128,
    )
    point_sources = {
        "Gamma": {
            "source_kind": "computed_config_points",
            "sectors": {
                "valence": {
                    "energies": np.array([-1.0, -1.0]),
                    "vectors": mixed_vectors,
                    "band_indices": np.array([0, 1]),
                },
            },
        }
    }

    output_dir = tmp_path / "rep-spin-gauge"
    run_symm_rep_from_point_sources(
        point_sources=point_sources,
        symmetry_dir=symmetry_dir,
        output_dir=output_dir,
        fermi_energy=0.0,
        degeneracy_tol=1.0e-6,
    )

    with np.load(output_dir / "band_representations.npz", allow_pickle=False) as payload:
        d_block = payload["Gamma__valence__block0__C2"]

    assert d_block[0, 0] == pytest.approx(1.0)
    assert d_block[1, 1] == pytest.approx(-1.0)
    assert d_block[0, 1] == pytest.approx(0.0)
    assert d_block[1, 0] == pytest.approx(0.0)

    bands = list(csv.DictReader((output_dir / "bands.csv").open(newline="", encoding="utf-8")))
    spin_weights = sorted(
        (float(row["spin_up_weight"]), float(row["spin_down_weight"]))
        for row in bands
    )
    assert spin_weights == pytest.approx([(0.0, 1.0), (1.0, 0.0)])


def test_symm_rep_config_resolves_canonical_paths_and_fractional_points(tmp_path):
    from tapw.symm_rep import resolve_config_request

    config_path = _write_release_symm_rep_config(tmp_path)
    rawh_path = tmp_path / "outputs" / "Gamma" / "q04" / "symmetry" / "representations.npz"
    rawh_path.parent.mkdir(parents=True)
    np.savez_compressed(rawh_path, C3z_data=np.array([1.0]), C3z_indices=np.array([0]), C3z_indptr=np.array([0, 1]), C3z_shape=np.array([1, 1]))

    request = resolve_config_request(config_path)

    assert request.symmetry_dir == rawh_path.parent
    assert request.output_dir == tmp_path / "outputs" / "Gamma" / "q04" / "symm_rep"
    assert request.fermi_energy == pytest.approx(-4.055365)
    assert request.valence_count == 12
    assert request.conduction_count == 8
    assert request.num_bands == 50
    assert request.degeneracy_tol == pytest.approx(1.0e-3)
    assert request.points == {"Gamma": (0.0, 0.0, 0.0), "M": (0.5, 0.0, 0.0), "K": (1.0 / 3.0, 1.0 / 3.0, 0.0)}


def test_symm_rep_config_missing_rawh_errors_without_running_symmetry(tmp_path):
    from tapw.symm_rep import resolve_config_request

    config_path = _write_release_symm_rep_config(tmp_path)

    with pytest.raises(FileNotFoundError, match="Run `tapw symm -c"):
        resolve_config_request(config_path)


def test_symm_rep_cli_accepts_config_only(monkeypatch, tmp_path):
    from tapw import symm_rep

    config_path = _write_release_symm_rep_config(tmp_path)
    calls = []

    monkeypatch.setattr(
        symm_rep,
        "run_configured_symm_rep",
        lambda config_path, **kwargs: calls.append((config_path, kwargs)) or (tmp_path / "out"),
    )

    code = symm_rep.main(["-c", str(config_path), "--valence-count", "6", "--conduction-count", "4", "--overwrite"])

    assert code == 0
    assert len(calls) == 1
    assert calls[0][0] == config_path
    assert calls[0][1]["valence_count"] == 6
    assert calls[0][1]["conduction_count"] == 4
    assert calls[0][1]["degeneracy_tol"] is None
    assert calls[0][1]["overwrite"] is True
    assert callable(calls[0][1]["progress"])


def test_run_configured_symm_rep_solves_points_and_writes_outputs(monkeypatch, tmp_path):
    from tapw import symm_rep

    config_path = _write_release_symm_rep_config(tmp_path)
    rawh_path = tmp_path / "outputs" / "Gamma" / "q04" / "symmetry" / "representations.npz"
    rawh_path.parent.mkdir(parents=True)
    np.savez_compressed(rawh_path, C3z_data=np.array([1.0]), C3z_indices=np.array([0]), C3z_indptr=np.array([0, 1]), C3z_shape=np.array([1, 1]))
    calls = []

    def fake_calculate(request, *, progress=None, timings=None):
        calls.append(("calculate", request.points))
        if progress is not None:
            progress("fake calculate")
        return {"Gamma": {"sectors": {"valence": {"energies": np.array([-1.0]), "vectors": np.ones((1, 1)), "band_indices": np.array([0])}}}}

    def fake_write(**kwargs):
        calls.append(("write", kwargs["symmetry_dir"], kwargs["output_dir"], kwargs["fermi_energy"], kwargs["degeneracy_tol"], kwargs["overwrite"]))
        return type("Result", (), {"output_dir": kwargs["output_dir"]})()

    monkeypatch.setattr(symm_rep, "calculate_config_point_sources", fake_calculate)
    monkeypatch.setattr(symm_rep, "run_symm_rep_from_point_sources", fake_write)

    messages = []
    output_dir = symm_rep.run_configured_symm_rep(config_path, overwrite=True, progress=messages.append)

    assert output_dir == tmp_path / "outputs" / "Gamma" / "q04" / "symm_rep"
    assert messages == [
        f"Resolved config: {config_path.resolve()}",
        f"Using raw-H representations: {rawh_path}",
        f"Writing symmetry representations to: {tmp_path / 'outputs' / 'Gamma' / 'q04' / 'symm_rep'}",
        "Band window: num_bands=50, valence_count=12, conduction_count=8, degeneracy_tol=0.001",
        "Computing high-symmetry point wavefunctions for: Gamma, K, M",
        "fake calculate",
        "Projecting raw-H actions into band subspaces.",
        f"Finished: {tmp_path / 'outputs' / 'Gamma' / 'q04' / 'symm_rep'}",
    ]
    assert calls == [
        ("calculate", {"Gamma": (0.0, 0.0, 0.0), "M": (0.5, 0.0, 0.0), "K": (1.0 / 3.0, 1.0 / 3.0, 0.0)}),
        ("write", rawh_path.parent, tmp_path / "outputs" / "Gamma" / "q04" / "symm_rep", -4.055365, 1.0e-3, True),
    ]


def test_config_point_solve_fails_without_auto_expanding_num_bands(monkeypatch, tmp_path):
    from tapw import symm_rep

    config_path = _write_release_symm_rep_config(tmp_path)
    calls = []

    class FakeCalculator:
        def __init__(self, config):
            self.config = config

        def calculate_band_01(self, _coords, _point_index):
            calls.append(int(self.config.num_bands_cal))
            energies = np.array([-6.0, -5.0, -4.0, -3.0])
            vectors = np.eye(128, len(energies), dtype=np.complex128)
            return energies, vectors, None, None

    monkeypatch.setattr(symm_rep, "_build_point_calculator", lambda config: FakeCalculator(config.compute))

    request = symm_rep.ConfigSymmRepRequest(
        config_path=config_path,
        symmetry_dir=tmp_path / "symmetry",
        output_dir=tmp_path / "out",
        fermi_energy=0.0,
        num_bands=50,
        valence_count=2,
        conduction_count=2,
        degeneracy_tol=1.0e-3,
        points={"Gamma": (0.0, 0.0, 0.0)},
    )

    with pytest.raises(RuntimeError, match=r"symmetry\.representation\.num_bands.*larger than 50"):
        symm_rep.calculate_config_point_sources(request)

    assert calls == [50]


def test_config_point_sources_uses_fresh_loky_point_workers(monkeypatch, tmp_path):
    from tapw import symm_rep

    config_path = _write_release_symm_rep_config(tmp_path)
    calls = []

    def fail_build(_config):
        raise AssertionError("parent process should not build a calculator for joblib/loky point workers")

    def fake_loky_runner(*, request, workers, blas_threads):
        calls.append(
            {
                "points": request.points,
                "workers": workers,
                "blas_threads": blas_threads,
            }
        )
        return [
            {
                "label": "Gamma",
                "coords": (0.0, 0.0, 0.0),
                "eig": np.array([-2.0, -1.0, 1.0, 2.0]),
                "vec": np.eye(4, dtype=np.complex128),
                "sewing_context": None,
                "setup_seconds": 0.2,
                "solve_seconds": 1.0,
                "bands": 4,
            },
            {
                "label": "K",
                "coords": (1.0 / 3.0, 1.0 / 3.0, 0.0),
                "eig": np.array([-3.0, -1.5, 1.5, 3.0]),
                "vec": np.eye(4, dtype=np.complex128),
                "sewing_context": None,
                "setup_seconds": 0.3,
                "solve_seconds": 1.1,
                "bands": 4,
            },
        ]

    monkeypatch.setattr(symm_rep, "_build_point_calculator", fail_build)
    monkeypatch.setattr(symm_rep, "_run_loky_point_workers", fake_loky_runner)

    request = symm_rep.ConfigSymmRepRequest(
        config_path=config_path,
        symmetry_dir=tmp_path / "symmetry",
        output_dir=tmp_path / "out",
        fermi_energy=0.0,
        num_bands=4,
        valence_count=1,
        conduction_count=1,
        degeneracy_tol=1.0e-3,
        points={"Gamma": (0.0, 0.0, 0.0), "K": (1.0 / 3.0, 1.0 / 3.0, 0.0)},
        cache_dir=tmp_path / "cache",
    )

    point_sources = symm_rep.calculate_config_point_sources(request)

    assert len(calls) == 1
    assert calls[0]["points"] == {"Gamma": (0.0, 0.0, 0.0), "K": (1.0 / 3.0, 1.0 / 3.0, 0.0)}
    assert calls[0]["workers"] == 2
    assert calls[0]["blas_threads"] == 1
    assert point_sources["Gamma"]["sectors"]["valence"]["energies"].tolist() == [-1.0]
    assert point_sources["K"]["sectors"]["conduction"]["energies"].tolist() == [1.5]


def test_config_point_sources_reuses_high_symmetry_cache(monkeypatch, tmp_path):
    from tapw import symm_rep

    config_path = _write_release_symm_rep_config(tmp_path)
    build_calls = []

    class FakeCalculator:
        def __init__(self, config):
            self.config = config
            self.result = {}

        def calculate_band_01(self, coords, point_index):
            build_calls.append((tuple(coords), point_index))
            eig = np.array([-2.0, -1.0, 1.0, 2.0], dtype=float)
            vec = np.eye(4, dtype=np.complex128)
            return eig, vec, None, None

    monkeypatch.setattr(symm_rep, "_build_point_calculator", lambda config: FakeCalculator(config.compute))

    request = symm_rep.ConfigSymmRepRequest(
        config_path=config_path,
        symmetry_dir=tmp_path / "symmetry",
        output_dir=tmp_path / "out",
        fermi_energy=0.0,
        num_bands=4,
        valence_count=1,
        conduction_count=1,
        degeneracy_tol=1.0e-3,
        points={"Gamma": (0.0, 0.0, 0.0)},
        cache_dir=tmp_path / "cache",
    )

    first = symm_rep.calculate_config_point_sources(request)
    assert build_calls == [((0.0, 0.0, 0.0), 0)]
    assert (tmp_path / "cache" / "high_symmetry_wavefunctions.npz").is_file()

    def fail_if_called(_config):
        raise AssertionError("calculator should not be rebuilt when high-symmetry cache is valid")

    monkeypatch.setattr(symm_rep, "_build_point_calculator", fail_if_called)
    second = symm_rep.calculate_config_point_sources(request)

    assert second["Gamma"]["sectors"]["valence"]["energies"].tolist() == first["Gamma"]["sectors"]["valence"]["energies"].tolist()
    assert second["Gamma"]["sectors"]["conduction"]["band_indices"].tolist() == [2]


def test_configured_symm_rep_no_overwrite_checks_output_dir_before_solving(monkeypatch, tmp_path):
    from tapw import symm_rep

    config_path = _write_release_symm_rep_config(tmp_path)
    rawh_path = tmp_path / "outputs" / "Gamma" / "q04" / "symmetry" / "representations.npz"
    rawh_path.parent.mkdir(parents=True)
    np.savez_compressed(rawh_path, C3z_data=np.array([1.0]), C3z_indices=np.array([0]), C3z_indptr=np.array([0, 1]), C3z_shape=np.array([1, 1]))
    output_dir = tmp_path / "outputs" / "Gamma" / "q04" / "symm_rep"
    output_dir.mkdir(parents=True)
    (output_dir / "old.txt").write_text("old", encoding="utf-8")

    def fail_if_called(_request, *, progress=None):
        raise AssertionError("point solve should not start when output_dir is blocked")

    monkeypatch.setattr(symm_rep, "calculate_config_point_sources", fail_if_called)

    with pytest.raises(FileExistsError, match="already exists and is not empty"):
        symm_rep.run_configured_symm_rep(config_path, overwrite=False)


def test_antiunitary_projection_uses_conjugated_eigenvectors():
    from tapw.workflows.symm_rep import project_operation_to_subspace

    vector = np.array([[1.0], [1.0j]], dtype=np.complex128) / np.sqrt(2.0)
    raw_action = np.eye(2, dtype=np.complex128)

    unitary = project_operation_to_subspace(raw_action, vector, antiunitary=False)
    antiunitary = project_operation_to_subspace(raw_action, vector, antiunitary=True)

    assert unitary[0, 0] == pytest.approx(1.0 + 0.0j)
    assert antiunitary[0, 0] == pytest.approx(0.0 + 0.0j)


def test_projection_orthonormalizes_nonorthogonal_degenerate_block():
    from tapw.workflows.symm_rep import project_operation_to_subspace

    overlap = -0.422777490301 - 0.01123923592023j
    vectors = np.array(
        [
            [1.0, overlap],
            [0.0, np.sqrt(1.0 - abs(overlap) ** 2)],
        ],
        dtype=np.complex128,
    )
    raw_action = -np.eye(2, dtype=np.complex128)

    projected = project_operation_to_subspace(raw_action, vectors, antiunitary=False)

    assert np.linalg.det(projected) == pytest.approx(1.0 + 0.0j, abs=1.0e-12)
    assert projected == pytest.approx(-np.eye(2, dtype=np.complex128), abs=1.0e-12)


def test_spin_labels_from_spin_halves():
    from tapw.workflows.symm_rep import spin_weights_and_label

    up_vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    down_vector = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128)
    mixed_vector = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.complex128) / np.sqrt(2.0)
    spinless_vector = np.array([1.0, 0.0, 0.0], dtype=np.complex128)

    up_weight, down_weight, label = spin_weights_and_label(up_vector)
    assert (up_weight, down_weight) == pytest.approx((1.0, 0.0))
    assert label == "up"

    up_weight, down_weight, label = spin_weights_and_label(down_vector)
    assert (up_weight, down_weight) == pytest.approx((0.0, 1.0))
    assert label == "down"

    up_weight, down_weight, label = spin_weights_and_label(mixed_vector)
    assert (up_weight, down_weight) == pytest.approx((0.5, 0.5))
    assert label == "mixed"

    assert spin_weights_and_label(spinless_vector) == (None, None, "spinless")


def test_symmetry_rep_label_preserves_spin_gauge_for_repeated_phase():
    from tapw.workflows.symm_rep import symmetry_rep_label

    projected = np.array(
        [
            [-1.0, 1.0e-6],
            [1.0e-6, -1.0],
        ],
        dtype=np.complex128,
    )
    spin_diagonal_vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.complex128,
    )

    label = symmetry_rep_label(projected, spin_diagonal_vectors)

    assert "-1[↑1.000,↓0.000]" in label
    assert "-1[↑0.000,↓1.000]" in label


def test_symm_rep_cli_smoke_writes_files(tmp_path, capsys):
    from tapw import symm_rep

    band_dir, symmetry_dir = _write_legacy_inputs(tmp_path)
    output_dir = tmp_path / "cli-output"

    code = symm_rep.main(
        [
            "--band-dir",
            str(band_dir),
            "--symmetry-dir",
            str(symmetry_dir),
            "--output-dir",
            str(output_dir),
            "--fermi-energy",
            "0.0",
            "--valence-count",
            "3",
            "--conduction-count",
            "3",
        ]
    )

    assert code == 0
    assert (output_dir / "summary.md").is_file()
    assert (output_dir / "characters.csv").is_file()
    stdout = capsys.readouterr().out
    assert "[tapw symm-rep] Start" in stdout
    assert "[tapw symm-rep] Complete" in stdout
    assert "summary          :" in stdout
    assert "time breakdown   :" in stdout
    assert "total           :" in stdout

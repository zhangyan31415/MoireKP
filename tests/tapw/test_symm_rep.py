import csv
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse


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
    assert "| valence | 1 | 1 2 | -1.000000000000, -1.000000000000 | C2 | false | -2.000000 | 0 |" in summary


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


def test_antiunitary_projection_uses_conjugated_eigenvectors():
    from tapw.workflows.symm_rep import project_operation_to_subspace

    vector = np.array([[1.0], [1.0j]], dtype=np.complex128) / np.sqrt(2.0)
    raw_action = np.eye(2, dtype=np.complex128)

    unitary = project_operation_to_subspace(raw_action, vector, antiunitary=False)
    antiunitary = project_operation_to_subspace(raw_action, vector, antiunitary=True)

    assert unitary[0, 0] == pytest.approx(1.0 + 0.0j)
    assert antiunitary[0, 0] == pytest.approx(0.0 + 0.0j)


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


def test_symm_rep_cli_smoke_writes_files(tmp_path):
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

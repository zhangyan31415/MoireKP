import csv
from pathlib import Path

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
    assert "| valence | 1 | 1 2 | -1.000000000000, -1.000000000000 | C2 | (-1↑, -1↑) |" in summary
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
    assert calls == [
        (
            config_path,
            {
                "valence_count": 6,
                "conduction_count": 4,
                "degeneracy_tol": None,
                "overwrite": True,
            },
        )
    ]


def test_run_configured_symm_rep_solves_points_and_writes_outputs(monkeypatch, tmp_path):
    from tapw import symm_rep

    config_path = _write_release_symm_rep_config(tmp_path)
    rawh_path = tmp_path / "outputs" / "Gamma" / "q04" / "symmetry" / "representations.npz"
    rawh_path.parent.mkdir(parents=True)
    np.savez_compressed(rawh_path, C3z_data=np.array([1.0]), C3z_indices=np.array([0]), C3z_indptr=np.array([0, 1]), C3z_shape=np.array([1, 1]))
    calls = []

    def fake_calculate(request):
        calls.append(("calculate", request.points))
        return {"Gamma": {"sectors": {"valence": {"energies": np.array([-1.0]), "vectors": np.ones((1, 1)), "band_indices": np.array([0])}}}}

    def fake_write(**kwargs):
        calls.append(("write", kwargs["symmetry_dir"], kwargs["output_dir"], kwargs["fermi_energy"], kwargs["degeneracy_tol"], kwargs["overwrite"]))
        return type("Result", (), {"output_dir": kwargs["output_dir"]})()

    monkeypatch.setattr(symm_rep, "calculate_config_point_sources", fake_calculate)
    monkeypatch.setattr(symm_rep, "run_symm_rep_from_point_sources", fake_write)

    output_dir = symm_rep.run_configured_symm_rep(config_path, overwrite=True)

    assert output_dir == tmp_path / "outputs" / "Gamma" / "q04" / "symm_rep"
    assert calls == [
        ("calculate", {"Gamma": (0.0, 0.0, 0.0), "M": (0.5, 0.0, 0.0), "K": (1.0 / 3.0, 1.0 / 3.0, 0.0)}),
        ("write", rawh_path.parent, tmp_path / "outputs" / "Gamma" / "q04" / "symm_rep", -4.055365, 1.0e-3, True),
    ]


def test_config_point_solve_expands_until_valence_and_conduction_are_available(monkeypatch, tmp_path):
    from tapw import symm_rep

    config_path = _write_release_symm_rep_config(tmp_path)
    calls = []

    class FakeCalculator:
        def __init__(self, config):
            self.config = config

        def calculate_band_01(self, _coords, _point_index):
            calls.append(int(self.config.num_bands_cal))
            if self.config.num_bands_cal < 80:
                energies = np.array([-6.0, -5.0, -4.0, -3.0])
            else:
                energies = np.array([-6.0, -5.0, -1.0, -0.5, 0.5, 1.0])
            vectors = np.eye(128, len(energies), dtype=np.complex128)
            return energies, vectors, None, None

    monkeypatch.setattr(symm_rep, "_build_point_calculator", lambda config: FakeCalculator(config.compute))

    request = symm_rep.ConfigSymmRepRequest(
        config_path=config_path,
        symmetry_dir=tmp_path / "symmetry",
        output_dir=tmp_path / "out",
        fermi_energy=0.0,
        valence_count=2,
        conduction_count=2,
        degeneracy_tol=1.0e-3,
        points={"Gamma": (0.0, 0.0, 0.0)},
    )

    point_sources = symm_rep.calculate_config_point_sources(request)

    assert calls == [50, 100]
    sectors = point_sources["Gamma"]["sectors"]
    assert sectors["valence"]["energies"].tolist() == [-1.0, -0.5]
    assert sectors["conduction"]["energies"].tolist() == [0.5, 1.0]


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

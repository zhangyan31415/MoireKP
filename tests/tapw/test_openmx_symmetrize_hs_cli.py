from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "openmx_symmetrize_hs.py"
PYTHON_WRAPPER_PATH = REPO_ROOT / "scripts" / "openmx_symm_hs_python.py"
SCFOUT_WRAPPER_PATH = REPO_ROOT / "scripts" / "openmx_symm_hs_scfout.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("openmx_symmetrize_hs", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_script(path: Path, name: str):
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))


def _write_openmx_dat(
    path: Path,
    *,
    p_basis: bool = False,
    spin_polarization: str = "off",
    spin_orbit: bool = False,
    nc_spin_tail: str | None = None,
) -> None:
    basis = "X7.0-p1" if p_basis else "X7.0-s1"
    spin_tail = nc_spin_tail if nc_spin_tail is not None else "1.0 1.0"
    path.write_text(
        f"""System.Name openmx
scf.SpinPolarization {spin_polarization}
scf.SpinOrbit.Coupling {'on' if spin_orbit else 'off'}
Atoms.UnitVectors.Unit Ang
<Atoms.UnitVectors
  1.0000000000 0.0000000000 0.0000000000
  0.0000000000 1.0000000000 0.0000000000
  0.0000000000 0.0000000000 1.0000000000
Atoms.UnitVectors>
Species.Number 1
<Definition.of.Atomic.Species
X {basis} X_PBE19
Definition.of.Atomic.Species>
Atoms.Number 2
Atoms.SpeciesAndCoordinates.Unit FRAC
<Atoms.SpeciesAndCoordinates
  1 X 0.0000000000 0.0000000000 0.0000000000 {spin_tail}
  2 X 0.5000000000 0.0000000000 0.0000000000 {spin_tail}
Atoms.SpeciesAndCoordinates>
""",
        encoding="utf-8",
    )


def _write_sparse(path: Path, label: str, nwann: int, entries: list[tuple[int, int, int, int, int, float, float]]) -> None:
    lines = [
        f" ! Sparse format of {label}",
        f" {len(entries)} ! Number of non-zeros lines of {label}mnR",
        f" {nwann} ! Number of orbitals",
        " 1 ! Number of R points",
    ]
    for rx, ry, rz, row, col, real, imag in entries:
        lines.append(f"{rx:5d}{ry:5d}{rz:5d}{row:6d}{col:6d}{real:16.8f}{imag:16.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_python_wrapper_uses_current_directory_and_explicit_h_s_files(tmp_path, monkeypatch):
    wrapper = _load_script(PYTHON_WRAPPER_PATH, "openmx_symm_hs_python_wrapper")
    monkeypatch.chdir(tmp_path)
    _write_openmx_dat(tmp_path / "my_openmx.dat")
    (tmp_path / "my_H.dat").write_text("H\n", encoding="utf-8")
    (tmp_path / "my_S.dat").write_text("S\n", encoding="utf-8")
    captured: list[list[str]] = []
    structure_text: list[str] = []

    def fake_main(argv):
        captured.append(list(argv))
        structure_dir = Path(argv[argv.index("--input-dir") + 1])
        structure_text.append((structure_dir / "openmx.dat").read_text(encoding="utf-8"))
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        output_dir.mkdir(parents=True)
        for name in ("H_sym.dat", "S_sym.dat", "H_sym.npz", "S_sym.npz", "symmetrization_report.json"):
            (output_dir / name).write_text(name, encoding="utf-8")
        return 0

    monkeypatch.setattr(wrapper.openmx_symmetrize_hs, "main", fake_main)

    assert wrapper.main(["--openmx", "my_openmx.dat", "--h", "my_H.dat", "--s", "my_S.dat"]) == 0

    argv = captured[0]
    assert "System.Name openmx" in structure_text[0]
    assert Path(argv[argv.index("--h-file") + 1]) == tmp_path / "my_H.dat"
    assert Path(argv[argv.index("--s-file") + 1]) == tmp_path / "my_S.dat"
    for name in ("H_symm.dat", "S_symm.dat", "H_symm.npz", "S_symm.npz", "symmetrization_report.json"):
        assert (tmp_path / name).is_file()
    for name in ("H_sym.dat", "S_sym.dat", "H_sym.npz", "S_sym.npz"):
        assert not (tmp_path / name).exists()
    assert not (tmp_path / "openmx_symm_python_out").exists()


def test_scfout_wrapper_uses_current_directory_and_explicit_scfout(tmp_path, monkeypatch, capsys):
    wrapper = _load_script(SCFOUT_WRAPPER_PATH, "openmx_symm_hs_scfout_wrapper")
    monkeypatch.chdir(tmp_path)
    _write_openmx_dat(tmp_path / "my_openmx.dat")
    scfout = tmp_path / "case.scfout"
    scfout.write_text("mock", encoding="utf-8")
    binary = tmp_path / "analysis_symm_hs"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    captured_transport: list[list[str]] = []
    captured_native: list[list[str]] = []
    structure_text: list[str] = []

    def fake_transport_main(argv):
        captured_transport.append(list(argv))
        structure_dir = Path(argv[argv.index("--input-dir") + 1])
        structure_text.append((structure_dir / "openmx.dat").read_text(encoding="utf-8"))
        Path(argv[argv.index("--export-transports-json") + 1]).write_text(
            json.dumps(
                {
                    "spglib_symprec": 1.0e-3,
                    "spglib_operation_count": 2,
                    "spin_mode": "spinor",
                    "spinful": True,
                    "natoms": 1,
                    "nwann": 2,
                    "max_atom_mapping_residual": 0.0,
                    "magnetic_symmetry": {
                        "mode": "auto-magnetic",
                        "applied": True,
                        "reason": "magnetic_moments_present",
                        "kept_unitary_operation_indices": [0],
                        "kept_primed_operation_indices": [1],
                        "filtered_operation_indices": [],
                    },
                    "operations": [
                        {
                            "index": 0,
                            "antiunitary": False,
                            "rotation_frac": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            "translation_frac": [0, 0, 0],
                        },
                        {
                            "index": 1,
                            "antiunitary": True,
                            "rotation_frac": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
                            "translation_frac": [0.999997, 0.999999, 0],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return 0

    class FakeRun:
        returncode = 0

    def fake_run(cmd, check=False):
        captured_native.append(list(cmd))
        native_out = Path(cmd[cmd.index("--output-dir") + 1])
        native_out.mkdir(parents=True, exist_ok=True)
        for name in ("H.dat", "S.dat", "H.npz", "S.npz", "symmetrization_report.json"):
            (native_out / name).write_text(name, encoding="utf-8")
        (tmp_path / "temporal_12345.input").write_text("temporary", encoding="utf-8")
        return FakeRun()

    monkeypatch.setattr(wrapper.openmx_symmetrize_hs, "main", fake_transport_main)
    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)

    assert wrapper.main(["--openmx", "my_openmx.dat", "--scfout", str(scfout), "--binary", str(binary), "--format", "both"]) == 0

    assert "System.Name openmx" in structure_text[0]
    assert Path(captured_transport[0][captured_transport[0].index("--export-transports-json") + 1]) == tmp_path / "symmetry_transports.json"
    native_cmd = captured_native[0]
    assert native_cmd[native_cmd.index("--output-format") + 1] == "both"
    assert native_cmd[native_cmd.index("--cutoff") + 1] == "1e-7"
    for name in ("H_symm.dat", "S_symm.dat", "H_symm.npz", "S_symm.npz", "symmetrization_report.json"):
        assert (tmp_path / name).is_file()
    for name in ("H.dat", "S.dat", "H.npz", "S.npz"):
        assert not (tmp_path / name).exists()
    assert not (tmp_path / "temporal_12345.input").exists()
    stdout = capsys.readouterr().out
    assert "[openmx-symm] Symmetry analysis" in stdout
    assert "[openmx-symm] Symmetry" in stdout
    assert "operations    : spglib=2, kept=2, unitary=1, antiunitary=1" in stdout
    assert "magnetic      : mode=auto-magnetic, applied=True, reason=magnetic_moments_present" in stdout
    assert "kept unitary  : [0]" in stdout
    assert "kept primed   : [1]" in stdout
    assert "1    1       antiunitary  +1" in stdout
    assert "t(frac)=(0, 0, 0)" in stdout
    assert "[openmx-symm] Native symmetrizer" in stdout
    assert "binary      :" in stdout
    assert "scfout      :" in stdout
    assert "output dir  :" not in stdout
    assert "[openmx-symm] Outputs" in stdout
    assert "H_symm.npz" in stdout
    assert "[openmx-symm] Finished" in stdout


def test_toy_s_orbital_symmetrization_reduces_covariance_residual_and_preserves_hermiticity(tmp_path):
    tool = _load_tool()
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _write_openmx_dat(input_dir / "openmx.dat")

    # Inversion swaps the two atoms. The diagonal onsite terms intentionally violate that symmetry.
    _write_sparse(
        input_dir / "H.dat",
        "H",
        2,
        [
            (0, 0, 0, 1, 1, 1.0, 0.0),
            (0, 0, 0, 2, 2, 3.0, 0.0),
            (0, 0, 0, 1, 2, 0.2, 0.0),
            (0, 0, 0, 2, 1, 0.2, 0.0),
        ],
    )
    _write_sparse(
        input_dir / "S.dat",
        "S",
        2,
        [
            (0, 0, 0, 1, 1, 1.0, 0.0),
            (0, 0, 0, 2, 2, 1.2, 0.0),
        ],
    )

    code = tool.main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--symprec",
            "1e-5",
            "--diagnostics",
            "full",
        ]
    )

    assert code == 0
    report = json.loads((output_dir / "symmetrization_report.json").read_text(encoding="utf-8"))
    assert report["spglib_operation_count"] >= 2
    assert report["H"]["symmetry_covariance_residual_after"] < report["H"]["symmetry_covariance_residual_before"]
    assert report["H"]["hermiticity_residual_after"] < 1.0e-12

    result = tool.read_openmx_sparse(output_dir / "H_sym.dat")
    h0 = result.blocks[(0, 0, 0)].toarray()
    np.testing.assert_allclose(h0, h0.conj().T, atol=1.0e-12)
    assert h0[0, 0].real == pytest.approx(h0[1, 1].real)


def test_p_orbital_rotation_block_has_expected_dimension_and_mismatched_orbitals_raise(tmp_path):
    tool = _load_tool()
    structure_path = tmp_path / "openmx.dat"
    _write_openmx_dat(structure_path, p_basis=True)
    structure = tool.parse_openmx_structure(structure_path)
    operations = tool.build_symmetry_operations(structure, symprec=1.0e-5)

    op = next(item for item in operations if not np.allclose(item.rotation_cart, np.eye(3)))
    transports = tool.build_operation_transports(structure, [op], atom_tol=1.0e-5)

    assert transports[0].atom_mappings
    assert all(block.shape == (3, 3) for block in transports[0].orbital_blocks.values())

    soc_path = tmp_path / "openmx_soc.dat"
    _write_openmx_dat(soc_path, p_basis=True, spin_polarization="NC", spin_orbit=True)
    soc_structure = tool.parse_openmx_structure(soc_path)
    soc_transports = tool.build_operation_transports(soc_structure, [op], atom_tol=1.0e-5)
    assert soc_structure.spinful is True
    assert soc_structure.nwann == 2 * soc_structure.nwann_spinless
    assert all(block.shape == (6, 6) for block in soc_transports[0].orbital_blocks.values())

    structure.atoms[1].orbital_spec = "X7.0-s1"
    with pytest.raises(ValueError, match="Orbital block mismatch"):
        tool.build_operation_transports(structure, [op], atom_tol=1.0e-5)


def test_collinear_magnetic_without_soc_uses_spin_channel_block_diagonal_basis(tmp_path):
    tool = _load_tool()
    structure_path = tmp_path / "openmx_collinear.dat"
    _write_openmx_dat(structure_path, p_basis=True, spin_polarization="on", spin_orbit=False)
    structure = tool.parse_openmx_structure(structure_path)
    operations = tool.build_symmetry_operations(structure, symprec=1.0e-5)
    op = next(item for item in operations if not np.allclose(item.rotation_cart, np.eye(3)))

    transports = tool.build_operation_transports(structure, [op], atom_tol=1.0e-5)

    assert structure.spinful is True
    assert structure.spin_mode == "collinear"
    assert structure.nwann == 2 * structure.nwann_spinless
    spinless_block = tool.orbital_rotation_block("X7.0-p1", op.rotation_cart)
    expected = np.kron(np.eye(2), spinless_block)
    np.testing.assert_allclose(transports[0].orbital_blocks[0], expected, atol=1.0e-12)


def test_export_transports_json_preserves_soc_spin_major_rotation_blocks(tmp_path):
    tool = _load_tool()
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    _write_openmx_dat(input_dir / "openmx.dat", p_basis=True, spin_polarization="NC", spin_orbit=True)
    structure = tool.parse_openmx_structure(input_dir / "openmx.dat")
    operations = tool.build_symmetry_operations(structure, symprec=1.0e-5)
    transports = tool.build_operation_transports(structure, operations, atom_tol=1.0e-5)
    output = tmp_path / "symmetry_transports.json"

    tool.write_symmetry_transports_json(output, structure, operations, transports, symprec=1.0e-5)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "openmx_hs_symmetry_transports/v1"
    assert payload["spinful"] is True
    assert payload["nwann"] == 12
    assert payload["atom_orbital_counts"] == [3, 3]
    assert payload["atom_offsets"] == [0, 3]
    assert len(payload["operations"]) == len(operations)
    non_identity_index = next(i for i, op in enumerate(operations) if not np.allclose(op.rotation_cart, np.eye(3)))
    exported = np.asarray(
        [
            [complex(value[0], value[1]) for value in row]
            for row in payload["operations"][non_identity_index]["orbital_blocks"]["0"]
        ],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(exported, transports[non_identity_index].orbital_blocks[0], atol=1.0e-12)
    assert exported.shape == (6, 6)


def test_magnetic_unitary_filter_rejects_operations_that_flip_axial_moment(tmp_path):
    tool = _load_tool()
    structure_path = tmp_path / "openmx_soc.dat"
    _write_openmx_dat(
        structure_path,
        spin_polarization="NC",
        spin_orbit=True,
        nc_spin_tail="2.0 1.0 0.0 0.0 0.0 0.0 0 on",
    )
    structure = tool.parse_openmx_structure(structure_path)
    identity = tool.SymmetryOperation(
        0,
        np.eye(3),
        np.zeros(3),
        np.eye(3),
        np.zeros(3),
    )
    c2_like_flips_z_axial_moment = tool.SymmetryOperation(
        1,
        np.diag([1.0, -1.0, -1.0]),
        np.zeros(3),
        np.diag([1.0, -1.0, -1.0]),
        np.zeros(3),
    )
    transports = tool.build_operation_transports(
        structure,
        [identity, c2_like_flips_z_axial_moment],
        atom_tol=1.0e-5,
    )

    filtered_operations, filtered_transports, report = tool.filter_magnetic_unitary_operations(
        structure,
        [identity, c2_like_flips_z_axial_moment],
        transports,
        magmom_tol=1.0e-8,
    )

    assert [op.index for op in filtered_operations] == [0]
    assert [item.operation.index for item in filtered_transports] == [0]
    assert report["mode"] == "unitary"
    assert report["input_operation_count"] == 2
    assert report["kept_operation_count"] == 1
    assert report["filtered_operation_indices"] == [1]


def test_antiunitary_time_reversal_projection_conjugates_spinful_block():
    tool = _load_tool()
    identity = tool.SymmetryOperation(0, np.eye(3), np.zeros(3), np.eye(3), np.zeros(3))
    unitary = tool.OperationTransport(
        identity,
        [0],
        [np.zeros(3, dtype=int)],
        {0: np.eye(2, dtype=np.complex128)},
        0.0,
        antiunitary=False,
    )
    antiunitary = tool.OperationTransport(
        identity,
        [0],
        [np.zeros(3, dtype=int)],
        {0: tool.time_reversal_spin_major_block(1)},
        0.0,
        antiunitary=True,
    )
    sigma_z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    projected = tool.symmetrize_atom_blocks({((0, 0, 0), 0, 0): sigma_z}, [unitary, antiunitary])

    np.testing.assert_allclose(projected[((0, 0, 0), 0, 0)], np.zeros((2, 2)), atol=1.0e-12)


def test_auto_magnetic_filter_keeps_primed_operations_as_antiunitary(tmp_path):
    tool = _load_tool()
    structure_path = tmp_path / "openmx_soc.dat"
    _write_openmx_dat(
        structure_path,
        spin_polarization="NC",
        spin_orbit=True,
        nc_spin_tail="2.0 1.0 0.0 0.0 0.0 0.0 0 on",
    )
    structure = tool.parse_openmx_structure(structure_path)
    identity = tool.SymmetryOperation(0, np.eye(3), np.zeros(3), np.eye(3), np.zeros(3))
    c2_like_needs_time_reversal = tool.SymmetryOperation(
        1,
        np.diag([1.0, -1.0, -1.0]),
        np.zeros(3),
        np.diag([1.0, -1.0, -1.0]),
        np.zeros(3),
    )
    transports = tool.build_operation_transports(
        structure,
        [identity, c2_like_needs_time_reversal],
        atom_tol=1.0e-5,
    )

    filtered_operations, filtered_transports, report = tool.apply_magnetic_symmetry_filter(
        structure,
        [identity, c2_like_needs_time_reversal],
        transports,
        mode="auto",
        magmom_tol=1.0e-8,
    )

    assert [op.index for op in filtered_operations] == [0, 1]
    assert [item.antiunitary for item in filtered_transports] == [False, True]
    assert report["mode"] == "auto-magnetic"
    assert report["kept_operation_count"] == 2
    assert report["kept_primed_operation_indices"] == [1]


def test_cli_dry_run_can_export_transports_json(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    transport_json = tmp_path / "symmetry_transports.json"
    input_dir.mkdir()
    _write_openmx_dat(input_dir / "openmx.dat")

    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--export-transports-json",
            str(transport_json),
            "--dry-run",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert run.returncode == 0, run.stderr
    assert not output_dir.exists()
    payload = json.loads(transport_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "openmx_hs_symmetry_transports/v1"
    assert payload["spglib_operation_count"] >= 2


def test_cli_dry_run_and_output_files(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _write_openmx_dat(input_dir / "openmx.dat")
    _write_sparse(input_dir / "H.dat", "H", 2, [(0, 0, 0, 1, 1, 1.0, 0.0), (0, 0, 0, 2, 2, 1.0, 0.0)])
    _write_sparse(input_dir / "S.dat", "S", 2, [(0, 0, 0, 1, 1, 1.0, 0.0), (0, 0, 0, 2, 2, 1.0, 0.0)])

    dry = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert dry.returncode == 0
    assert not output_dir.exists()

    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--matrix-workers",
            "2",
            "--block-workers",
            "2",
            "--text-parser",
            "loadtxt",
            "--write-npz-cache",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert run.returncode == 0, run.stderr
    assert (output_dir / "H_sym.dat").is_file()
    assert (output_dir / "S_sym.dat").is_file()
    assert (output_dir / "H_sym.npz").is_file()
    assert (output_dir / "S_sym.npz").is_file()
    assert (output_dir / "symmetrization_report.json").is_file()
    first_data = [line for line in (output_dir / "H_sym.dat").read_text(encoding="utf-8").splitlines()[4:] if line.strip()][0]
    assert int(first_data.split()[3]) >= 1
    assert int(first_data.split()[4]) >= 1
    tool = _load_tool()
    dat_blocks = tool.read_openmx_sparse(output_dir / "H_sym.dat").blocks
    npz_blocks = tool.read_openmx_sparse_npz(output_dir / "H_sym.npz", nwann=2).blocks
    assert set(npz_blocks) == set(dat_blocks)
    for key in dat_blocks:
        np.testing.assert_allclose(npz_blocks[key].toarray(), dat_blocks[key].toarray(), atol=1.0e-12)


def test_loadtxt_parser_matches_line_parser_with_duplicate_entries(tmp_path):
    tool = _load_tool()
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    _write_openmx_dat(input_dir / "openmx.dat", spin_polarization="NC", spin_orbit=True)
    _write_sparse(
        input_dir / "H.dat",
        "H",
        4,
        [
            (0, 0, 0, 1, 1, 1.0, 0.0),
            (0, 0, 0, 1, 1, 0.5, 0.25),
            (0, 0, 0, 3, 4, 0.2, -0.1),
            (-1, 0, 0, 4, 2, -0.3, 0.4),
        ],
    )
    structure = tool.parse_openmx_structure(input_dir / "openmx.dat")

    line_blocks = tool.read_openmx_atom_blocks(input_dir / "H.dat", structure, text_parser="line").atom_blocks
    loadtxt_blocks = tool.read_openmx_atom_blocks(input_dir / "H.dat", structure, text_parser="loadtxt").atom_blocks

    assert set(loadtxt_blocks) == set(line_blocks)
    for key in line_blocks:
        np.testing.assert_allclose(loadtxt_blocks[key], line_blocks[key], atol=1.0e-14)


def test_block_worker_transform_matches_serial(tmp_path):
    tool = _load_tool()
    structure_path = tmp_path / "openmx.dat"
    _write_openmx_dat(structure_path)
    structure = tool.parse_openmx_structure(structure_path)
    transports = tool.build_operation_transports(
        structure,
        tool.build_symmetry_operations(structure, symprec=1.0e-5),
        atom_tol=1.0e-5,
    )
    canonical_blocks = {
        ((0, 0, 0), 0, 0): np.array([[1.0 + 0.1j]], dtype=np.complex128),
        ((0, 0, 0), 0, 1): np.array([[0.2 - 0.3j]], dtype=np.complex128),
    }
    action_cache = tool.SymmetryActionCache(transports)

    serial = tool.transform_canonical_atom_blocks(
        canonical_blocks,
        transports,
        action_cache=action_cache,
        block_workers=1,
    )
    threaded = tool.transform_canonical_atom_blocks(
        canonical_blocks,
        transports,
        action_cache=action_cache,
        block_workers=2,
    )

    assert set(threaded) == set(serial)
    for key in serial:
        np.testing.assert_allclose(threaded[key], serial[key], atol=1.0e-14)


def test_fused_hermitian_canonicalization_matches_two_step():
    tool = _load_tool()
    atom_blocks = {
        ((0, 0, 0), 0, 0): np.array([[1.0 + 0.2j, 0.4], [0.1j, 2.0]], dtype=np.complex128),
        ((1, 0, 0), 0, 1): np.array([[0.2 - 0.3j]], dtype=np.complex128),
        ((-1, 0, 0), 1, 0): np.array([[0.8 + 0.5j]], dtype=np.complex128),
        ((0, 1, 0), 1, 0): np.array([[0.7 - 0.1j]], dtype=np.complex128),
    }

    two_step = tool.canonicalize_hermitian_blocks(tool.hermitize_atom_blocks(atom_blocks))
    fused = tool.hermitize_canonical_atom_blocks(atom_blocks)

    assert set(fused) == set(two_step)
    for key in two_step:
        np.testing.assert_allclose(fused[key], two_step[key], atol=1.0e-14)


def test_soc_cli_uses_spin_major_openmx_indices(tmp_path):
    input_dir = tmp_path / "soc_in"
    output_dir = tmp_path / "soc_out"
    input_dir.mkdir()
    _write_openmx_dat(input_dir / "openmx.dat", spin_polarization="NC", spin_orbit=True)
    _write_sparse(
        input_dir / "H.dat",
        "H",
        4,
        [
            (0, 0, 0, 1, 1, 1.0, 0.0),
            (0, 0, 0, 2, 2, 3.0, 0.0),
            (0, 0, 0, 3, 3, 1.0, 0.0),
            (0, 0, 0, 4, 4, 3.0, 0.0),
        ],
    )
    _write_sparse(
        input_dir / "S.dat",
        "S",
        4,
        [
            (0, 0, 0, 1, 1, 1.0, 0.0),
            (0, 0, 0, 2, 2, 1.0, 0.0),
            (0, 0, 0, 3, 3, 1.0, 0.0),
            (0, 0, 0, 4, 4, 1.0, 0.0),
        ],
    )

    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--matrix-workers",
            "2",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert run.returncode == 0, run.stderr
    report = json.loads((output_dir / "symmetrization_report.json").read_text(encoding="utf-8"))
    assert report["spinful"] is True
    assert report["nwann"] == 4
    result = _load_tool().read_openmx_sparse(output_dir / "H_sym.dat")
    h0 = result.blocks[(0, 0, 0)].toarray()
    np.testing.assert_allclose(h0, h0.conj().T, atol=1.0e-12)
    assert {int(line.split()[3]) for line in (output_dir / "H_sym.dat").read_text(encoding="utf-8").splitlines()[4:]} <= {1, 2, 3, 4}


def test_collinear_spin_inputs_are_parsed_as_spinful_block_diagonal(tmp_path):
    tool = _load_tool()
    path = tmp_path / "openmx.dat"
    _write_openmx_dat(path, spin_polarization="on")

    structure = tool.parse_openmx_structure(path)

    assert structure.spinful is True
    assert structure.spin_mode == "collinear"
    assert structure.nwann == 2 * structure.nwann_spinless

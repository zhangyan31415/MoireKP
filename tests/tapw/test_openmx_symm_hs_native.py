from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "openmx_symmetrize_hs.py"
NATIVE_SOURCE = REPO_ROOT / "scripts" / "openmx_analysis_symm_hs.cpp"


def _load_tool():
    spec = importlib.util.spec_from_file_location("openmx_symmetrize_hs", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_native_dense_value_insertion_avoids_eager_matrix_allocation():
    source = NATIVE_SOURCE.read_text(encoding="utf-8")
    start = source.index("void add_dense_value(")
    end = source.index("#ifdef OPENMX_ANALYSIS_BUILD", start)
    body = source[start:end]

    assert "try_emplace" in body
    assert "blocks.emplace(key, Matrix(rows_dim, cols_dim))" not in body


def test_native_openmx_loader_fills_dense_blocks_without_scalar_map_lookups():
    source = NATIVE_SOURCE.read_text(encoding="utf-8")
    start = source.index("MatrixSet load_openmx_scfout(")
    end = source.index("merge_block_maps(data.h_blocks", start)
    body = source[start:end]

    assert "get_or_create_block" in source
    assert "add_dense_value(" not in body


def test_native_openmx_loader_supports_collinear_spin_switch_as_block_diagonal_spinful():
    source = NATIVE_SOURCE.read_text(encoding="utf-8")
    start = source.index("MatrixSet load_openmx_scfout(")
    end = source.index("merge_block_maps(data.h_blocks", start)
    body = source[start:end]

    assert "SpinP_switch == 1" in body
    assert "Collinear SpinP_switch==1 is not supported" not in body
    assert "data.spinful = SpinP_switch == 1 || SpinP_switch == 3" in body
    assert "accumulate_openmx_neighbor_task(tasks[idx], SpinP_switch" in body


def _write_openmx_dat(path: Path) -> None:
    path.write_text(
        """System.Name openmx
scf.SpinPolarization off
scf.SpinOrbit.Coupling off
Atoms.UnitVectors.Unit Ang
<Atoms.UnitVectors
  1.0000000000 0.0000000000 0.0000000000
  0.0000000000 1.0000000000 0.0000000000
  0.0000000000 0.0000000000 1.0000000000
Atoms.UnitVectors>
Species.Number 1
<Definition.of.Atomic.Species
X X7.0-s1 X_PBE19
Definition.of.Atomic.Species>
Atoms.Number 2
Atoms.SpeciesAndCoordinates.Unit FRAC
<Atoms.SpeciesAndCoordinates
  1 X 0.0000000000 0.0000000000 0.0000000000 1.0 1.0
  2 X 0.5000000000 0.0000000000 0.0000000000 1.0 1.0
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


def test_native_mock_mode_compiles_and_matches_python_symmetrized_toy_output(tmp_path):
    tool = _load_tool()
    input_dir = tmp_path / "in"
    python_out = tmp_path / "python"
    native_out = tmp_path / "native"
    input_dir.mkdir()
    _write_openmx_dat(input_dir / "openmx.dat")
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
    assert tool.main(["--input-dir", str(input_dir), "--output-dir", str(python_out), "--symprec", "1e-5"]) == 0

    structure = tool.parse_openmx_structure(input_dir / "openmx.dat")
    operations = tool.build_symmetry_operations(structure, symprec=1.0e-5)
    transports = tool.build_operation_transports(structure, operations, atom_tol=1.0e-5)
    transport_json = tmp_path / "symmetry_transports.json"
    tool.write_symmetry_transports_json(transport_json, structure, operations, transports, symprec=1.0e-5)
    mock_input = tmp_path / "mock_blocks.json"
    mock_input.write_text(
        json.dumps(
            {
                "spinful": False,
                "atom_orbital_counts": [1, 1],
                "h_blocks": [
                    {"r": [0, 0, 0], "atom_i": 0, "atom_j": 0, "rows": [0], "cols": [0], "values": [[1.0, 0.0]]},
                    {"r": [0, 0, 0], "atom_i": 1, "atom_j": 1, "rows": [0], "cols": [0], "values": [[3.0, 0.0]]},
                    {"r": [0, 0, 0], "atom_i": 0, "atom_j": 1, "rows": [0], "cols": [0], "values": [[0.2, 0.0]]},
                    {"r": [0, 0, 0], "atom_i": 1, "atom_j": 0, "rows": [0], "cols": [0], "values": [[0.2, 0.0]]},
                ],
                "s_blocks": [
                    {"r": [0, 0, 0], "atom_i": 0, "atom_j": 0, "rows": [0], "cols": [0], "values": [[1.0, 0.0]]},
                    {"r": [0, 0, 0], "atom_i": 1, "atom_j": 1, "rows": [0], "cols": [0], "values": [[1.2, 0.0]]},
                ],
            }
        ),
        encoding="utf-8",
    )
    binary = tmp_path / "analysis_symm_hs"
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-fopenmp", str(NATIVE_SOURCE), "-o", str(binary)],
        check=True,
        text=True,
        capture_output=True,
    )

    run = subprocess.run(
        [
            str(binary),
            "--mock-input",
            str(mock_input),
            "--symmetry-json",
            str(transport_json),
            "--output-dir",
            str(native_out),
            "--cutoff",
            "1e-10",
            "--diagnostics",
            "basic",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert run.returncode == 0, run.stderr
    assert "[1/8] load symmetry JSON" in run.stdout
    assert "[4/8] symmetrize H" in run.stdout
    assert "[8/8] write report" in run.stdout
    assert "finished in" in run.stdout
    native_h = tool.read_openmx_sparse(native_out / "H.dat").blocks
    native_s = tool.read_openmx_sparse(native_out / "S.dat").blocks
    python_h = tool.read_openmx_sparse(python_out / "H_sym.dat").blocks
    python_s = tool.read_openmx_sparse(python_out / "S_sym.dat").blocks
    assert set(native_h) == set(python_h)
    assert set(native_s) == set(python_s)
    for key in python_h:
        np.testing.assert_allclose(native_h[key].toarray(), python_h[key].toarray(), atol=1.0e-8)
    for key in python_s:
        np.testing.assert_allclose(native_s[key].toarray(), python_s[key].toarray(), atol=1.0e-8)


def test_native_mock_mode_applies_antiunitary_complex_conjugation(tmp_path):
    tool = _load_tool()
    native_out = tmp_path / "native_antiunitary"
    transport_json = tmp_path / "symmetry_transports.json"

    def cmatrix(matrix):
        return [[[float(value.real), float(value.imag)] for value in row] for row in matrix]

    identity = np.eye(2, dtype=np.complex128)
    time_reversal = np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.complex128)
    transport_json.write_text(
        json.dumps(
            {
                "schema": "openmx_hs_symmetry_transports/v1",
                "spglib_symprec": 1.0e-5,
                "spglib_operation_count": 2,
                "spinful": True,
                "natoms": 1,
                "nwann": 2,
                "nwann_spinless": 1,
                "atom_orbital_counts": [1],
                "atom_offsets": [0],
                "operations": [
                    {
                        "index": 0,
                        "antiunitary": False,
                        "rotation_frac": np.eye(3).tolist(),
                        "atom_mappings": [0],
                        "image_shifts": [[0, 0, 0]],
                        "orbital_blocks": {"0": cmatrix(identity)},
                    },
                    {
                        "index": 1,
                        "antiunitary": True,
                        "rotation_frac": np.eye(3).tolist(),
                        "atom_mappings": [0],
                        "image_shifts": [[0, 0, 0]],
                        "orbital_blocks": {"0": cmatrix(time_reversal)},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    mock_input = tmp_path / "mock_blocks.json"
    mock_input.write_text(
        json.dumps(
            {
                "spinful": True,
                "atom_orbital_counts": [1],
                "h_blocks": [
                    {
                        "r": [0, 0, 0],
                        "atom_i": 0,
                        "atom_j": 0,
                        "rows": [0, 1],
                        "cols": [0, 1],
                        "values": [[1.0, 0.0], [-1.0, 0.0]],
                    }
                ],
                "s_blocks": [
                    {
                        "r": [0, 0, 0],
                        "atom_i": 0,
                        "atom_j": 0,
                        "rows": [0, 1],
                        "cols": [0, 1],
                        "values": [[1.0, 0.0], [1.0, 0.0]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    binary = tmp_path / "analysis_symm_hs"
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-fopenmp", str(NATIVE_SOURCE), "-o", str(binary)],
        check=True,
        text=True,
        capture_output=True,
    )

    run = subprocess.run(
        [
            str(binary),
            "--mock-input",
            str(mock_input),
            "--symmetry-json",
            str(transport_json),
            "--output-dir",
            str(native_out),
            "--cutoff",
            "1e-10",
            "--diagnostics",
            "basic",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert run.returncode == 0, run.stderr
    assert tool.read_openmx_sparse(native_out / "H.dat").blocks == {}
    native_s = tool.read_openmx_sparse(native_out / "S.dat").blocks[(0, 0, 0)].toarray()
    np.testing.assert_allclose(native_s, np.eye(2), atol=1.0e-12)
    report = json.loads((native_out / "symmetrization_report.json").read_text(encoding="utf-8"))
    assert report["diagnostics"] == "basic"
    assert report["H"]["output_nonzero"] == 0
    assert report["S"]["output_nonzero"] == 2


def test_native_mock_mode_writes_npz_output_compatible_with_python_reader(tmp_path):
    tool = _load_tool()
    input_dir = tmp_path / "in"
    python_out = tmp_path / "python"
    native_out = tmp_path / "native_npz"
    input_dir.mkdir()
    _write_openmx_dat(input_dir / "openmx.dat")
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
    assert (
        tool.main(
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(python_out),
                "--symprec",
                "1e-5",
                "--write-npz-cache",
            ]
        )
        == 0
    )

    structure = tool.parse_openmx_structure(input_dir / "openmx.dat")
    operations = tool.build_symmetry_operations(structure, symprec=1.0e-5)
    transports = tool.build_operation_transports(structure, operations, atom_tol=1.0e-5)
    transport_json = tmp_path / "symmetry_transports.json"
    tool.write_symmetry_transports_json(transport_json, structure, operations, transports, symprec=1.0e-5)
    mock_input = tmp_path / "mock_blocks.json"
    mock_input.write_text(
        json.dumps(
            {
                "spinful": False,
                "atom_orbital_counts": [1, 1],
                "h_blocks": [
                    {"r": [0, 0, 0], "atom_i": 0, "atom_j": 0, "rows": [0], "cols": [0], "values": [[1.0, 0.0]]},
                    {"r": [0, 0, 0], "atom_i": 1, "atom_j": 1, "rows": [0], "cols": [0], "values": [[3.0, 0.0]]},
                    {"r": [0, 0, 0], "atom_i": 0, "atom_j": 1, "rows": [0], "cols": [0], "values": [[0.2, 0.0]]},
                    {"r": [0, 0, 0], "atom_i": 1, "atom_j": 0, "rows": [0], "cols": [0], "values": [[0.2, 0.0]]},
                ],
                "s_blocks": [
                    {"r": [0, 0, 0], "atom_i": 0, "atom_j": 0, "rows": [0], "cols": [0], "values": [[1.0, 0.0]]},
                    {"r": [0, 0, 0], "atom_i": 1, "atom_j": 1, "rows": [0], "cols": [0], "values": [[1.2, 0.0]]},
                ],
            }
        ),
        encoding="utf-8",
    )
    binary = tmp_path / "analysis_symm_hs"
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-fopenmp", str(NATIVE_SOURCE), "-o", str(binary)],
        check=True,
        text=True,
        capture_output=True,
    )

    run = subprocess.run(
        [
            str(binary),
            "--mock-input",
            str(mock_input),
            "--symmetry-json",
            str(transport_json),
            "--output-dir",
            str(native_out),
            "--cutoff",
            "1e-10",
            "--output-format",
            "npz",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert run.returncode == 0, run.stderr
    assert not (native_out / "H.dat").exists()
    assert not (native_out / "S.dat").exists()
    native_h = tool.read_openmx_sparse_npz(native_out / "H.npz", nwann=2).blocks
    native_s = tool.read_openmx_sparse_npz(native_out / "S.npz", nwann=2).blocks
    python_h = tool.read_openmx_sparse_npz(python_out / "H_sym.npz", nwann=2).blocks
    python_s = tool.read_openmx_sparse_npz(python_out / "S_sym.npz", nwann=2).blocks
    assert set(native_h) == set(python_h)
    assert set(native_s) == set(python_s)
    for key in python_h:
        np.testing.assert_allclose(native_h[key].toarray(), python_h[key].toarray(), atol=1.0e-8)
    for key in python_s:
        np.testing.assert_allclose(native_s[key].toarray(), python_s[key].toarray(), atol=1.0e-8)

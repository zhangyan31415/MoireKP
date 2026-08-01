from pathlib import Path
import os

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import write

from tapw.config import Config
from tapw.io.structure import OpenMXFile, infer_2d_bravais, resolve_structure_input


def _write_sparse_source(path: Path, dimension: int) -> None:
    diagonal = np.arange(dimension, dtype=np.int64)
    np.savez(
        path,
        **{
            "(0, 0, 0)_row": diagonal,
            "(0, 0, 0)_col": diagonal,
            "(0, 0, 0)_val": np.ones(dimension, dtype=np.complex128),
        },
    )


def _write_system_case(
    tmp_path: Path,
    *,
    symbols=("Te", "Mo"),
    orbitals=None,
    spin=False,
    matrix_dimension=4,
    structure_format="vasp",
) -> Path:
    orbitals = orbitals or {"Mo": "s1", "Te": "p1"}
    atoms = Atoms(
        symbols=list(symbols),
        positions=[[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]],
        cell=[[3.0, 0.0, 0.0], [1.5, 2.598076211, 0.0], [0.0, 0.0, 20.0]],
        pbc=[True, True, False],
    )
    structure_path = tmp_path / ("POSCAR" if structure_format == "vasp" else "structure.cif")
    write(structure_path, atoms, format=structure_format)
    _write_sparse_source(tmp_path / "H.npz", matrix_dimension)
    _write_sparse_source(tmp_path / "S.npz", matrix_dimension)
    payload = {
        "system": {
            "output": "outputs",
            "structure": structure_path.name,
            "hamiltonian": "H.npz",
            "overlap": "S.npz",
            "orbitals": orbitals,
            "twist_index": 1,
            "layers": [1, 1],
            "spin": spin,
        },
        "symmetry": {"valley": "Gamma", "q_shell": 1},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


@pytest.mark.parametrize("structure_format", ["vasp", "cif"])
def test_resolved_structure_preserves_ase_site_order_and_compact_orbitals(tmp_path, structure_format):
    config = Config.from_yaml(str(_write_system_case(tmp_path, structure_format=structure_format)))

    resolved = resolve_structure_input(config)

    assert [atom["species"] for atom in resolved.structure.species_coordinates] == ["Te", "Mo"]
    assert [atom["original_index"] for atom in resolved.structure.species_coordinates] == [1, 2]
    assert resolved.structure.orbitals_count == {"Mo": 1, "Te": 3}
    assert resolved.expected_basis_dimension == 4
    assert resolved.bravais == "hex"
    assert resolved.structure.reciprocal_Tmat.shape == (3, 3)
    assert len(resolved.source_identity) == 64


def test_resolved_structure_counts_spin_in_hs_basis_validation(tmp_path):
    config = Config.from_yaml(str(_write_system_case(tmp_path, spin=True, matrix_dimension=8)))

    resolved = resolve_structure_input(config)

    assert resolved.expected_basis_dimension == 8


def test_resolved_structure_rejects_missing_or_extra_orbital_species(tmp_path):
    config_path = _write_system_case(tmp_path, orbitals={"Mo": "s1", "I": "p1"})
    config = Config.from_yaml(str(config_path))

    with pytest.raises(ValueError, match=r"orbitals.*missing.*Te.*extra.*I"):
        resolve_structure_input(config)


def test_resolved_structure_rejects_hamiltonian_and_overlap_dimension_mismatch(tmp_path):
    config_path = _write_system_case(tmp_path, matrix_dimension=4)
    _write_sparse_source(tmp_path / "S.npz", 5)
    config = Config.from_yaml(str(config_path))

    with pytest.raises(ValueError, match=r"H/S basis dimension mismatch.*4.*5"):
        resolve_structure_input(config)


def test_resolved_structure_rejects_source_dimension_incompatible_with_orbitals(tmp_path):
    config = Config.from_yaml(str(_write_system_case(tmp_path, matrix_dimension=5)))

    with pytest.raises(ValueError, match=r"source basis dimension.*5.*expected.*4"):
        resolve_structure_input(config)


def test_structure_identity_changes_with_site_order_or_orbital_mapping(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    third_dir = tmp_path / "third"
    for directory in (first_dir, second_dir, third_dir):
        directory.mkdir()
    first = resolve_structure_input(Config.from_yaml(str(_write_system_case(first_dir))))
    reordered = resolve_structure_input(
        Config.from_yaml(str(_write_system_case(second_dir, symbols=("Mo", "Te"))))
    )
    changed_orbitals = resolve_structure_input(
        Config.from_yaml(
            str(
                _write_system_case(
                    third_dir,
                    orbitals={"Mo": "p1", "Te": "s1"},
                    matrix_dimension=4,
                )
            )
        )
    )

    assert first.source_identity != reordered.source_identity
    assert first.source_identity != changed_orbitals.source_identity


def test_infer_2d_bravais_is_pure_and_fail_closed(monkeypatch):
    monkeypatch.setenv("TAPW_BRAVAIS", "square")
    hex_cell = np.array([[3.0, 0.0, 0.0], [1.5, 2.598076211, 0.0], [0.0, 0.0, 20.0]])
    square_cell = np.diag([3.0, 3.0, 20.0])

    assert infer_2d_bravais(hex_cell) == "hex"
    assert infer_2d_bravais(square_cell) == "square"
    assert infer_2d_bravais(np.diag([3.0, 4.0, 20.0])) == "rect"
    assert os.environ["TAPW_BRAVAIS"] == "square"
    with pytest.raises(ValueError, match="Cannot infer supported 2D Bravais"):
        infer_2d_bravais(np.array([[3.0, 0.0, 0.0], [0.7, 3.4, 0.0], [0.0, 0.0, 20.0]]))


def test_legacy_openmx_structure_does_not_read_or_write_global_bravais(tmp_path, monkeypatch):
    monkeypatch.delenv("TAPW_BRAVAIS", raising=False)
    source = tmp_path / "openmx.dat"
    source.write_text(
        """<Definition.of.Atomic.Species
Te Te7.0-s1 Te_PBE
Definition.of.Atomic.Species>
Atoms.Number 1
Atoms.SpeciesAndCoordinates.Unit Ang
<Atoms.SpeciesAndCoordinates
1 Te 0.0 0.0 1.0 0.0 0.0
Atoms.SpeciesAndCoordinates>
<Atoms.UnitVectors
3.0 0.0 0.0
1.5 2.598076211 0.0
0.0 0.0 20.0
Atoms.UnitVectors>
""",
        encoding="utf-8",
    )

    structure = OpenMXFile(source, twist_index=1, spin=False, bravais="hex")

    assert structure.bravais == "hex"
    assert "TAPW_BRAVAIS" not in os.environ

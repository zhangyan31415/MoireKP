from pathlib import Path
import os
import json
from dataclasses import replace

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import write

from tapw.config import Config
from tapw.io.hr import SPARSE_NPZ_METADATA_KEY, SPARSE_NPZ_SCHEMA
from tapw.io.structure import (
    OpenMXFile,
    infer_2d_bravais,
    load_structure_from_config,
    resolve_structure_input,
)


def _write_sparse_source(
    path: Path,
    dimension: int,
    *,
    metadata_dimension=None,
    metadata_schema=SPARSE_NPZ_SCHEMA,
    metadata_version=1,
    coordinate_indices=None,
) -> None:
    diagonal = np.asarray(
        np.arange(dimension, dtype=np.int64) if coordinate_indices is None else coordinate_indices,
        dtype=np.int64,
    )
    payload = {
        "(0, 0, 0)_row": diagonal,
        "(0, 0, 0)_col": diagonal,
        "(0, 0, 0)_val": np.ones(len(diagonal), dtype=np.complex128),
    }
    if metadata_dimension is not None:
        payload[SPARSE_NPZ_METADATA_KEY] = np.asarray(
            json.dumps(
                {
                    "schema": metadata_schema,
                    "schema_version": metadata_version,
                    "basis_dimension": int(metadata_dimension),
                },
                sort_keys=True,
            ),
            dtype=str,
        )
    np.savez(path, **payload)


def _write_sparse_dat(path: Path, dimension: int, coordinate_indices=None) -> None:
    indices = list(range(dimension)) if coordinate_indices is None else list(coordinate_indices)
    lines = [
        "! sparse source",
        f"{len(indices)} ! nonzero",
        f"{dimension} ! basis dimension",
        "1 ! R points",
    ]
    lines.extend(
        f"0 0 0 {index + 1} {index + 1} 1.0 0.0"
        for index in indices
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_system_case(
    tmp_path: Path,
    *,
    symbols=("Te", "Mo"),
    orbitals=None,
    spin=False,
    matrix_dimension=4,
    structure_format="vasp",
    cell=None,
    overlap=True,
) -> Path:
    orbitals = orbitals or {"Mo": "s1", "Te": "p1"}
    atoms = Atoms(
        symbols=list(symbols),
        positions=[[float(index), float(index), float(index + 1)] for index in range(len(symbols))],
        cell=(
            [[3.0, 0.0, 0.0], [1.5, 2.598076211, 0.0], [0.0, 0.0, 20.0]]
            if cell is None
            else cell
        ),
        pbc=[True, True, False],
    )
    structure_path = tmp_path / ("POSCAR" if structure_format == "vasp" else "structure.cif")
    write(structure_path, atoms, format=structure_format)
    _write_sparse_source(tmp_path / "H.npz", matrix_dimension)
    if overlap:
        _write_sparse_source(tmp_path / "S.npz", matrix_dimension)
    payload = {
        "system": {
            "output": "outputs",
            "structure": structure_path.name,
            "hamiltonian": "H.npz",
            "orbitals": orbitals,
            "twist_index": 1,
            "layers": [1, 1],
            "spin": spin,
        },
        "symmetry": {"valley": "Gamma", "q_shell": 1},
    }
    if overlap:
        payload["system"]["overlap"] = "S.npz"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _write_openmx_case(tmp_path: Path, *, square: bool = True, explicit_bravais=None) -> Path:
    a2 = "0.0 3.0 0.0" if square else "1.5 2.598076211 0.0"
    (tmp_path / "openmx.dat").write_text(
        f"""<Definition.of.Atomic.Species
Te Te7.0-s1 Te_PBE
Definition.of.Atomic.Species>
Atoms.Number 1
Atoms.SpeciesAndCoordinates.Unit Ang
<Atoms.SpeciesAndCoordinates
1 Te 0.0 0.0 1.0 0.0 0.0
Atoms.SpeciesAndCoordinates>
<Atoms.UnitVectors
3.0 0.0 0.0
{a2}
0.0 0.0 20.0
Atoms.UnitVectors>
""",
        encoding="utf-8",
    )
    _write_sparse_dat(tmp_path / "H.dat", 1)
    _write_sparse_dat(tmp_path / "S.dat", 1)
    twist = {"twist_index_m": 1, "twist_layer": [1, 1], "spin": False}
    if explicit_bravais is not None:
        twist["bravais"] = explicit_bravais
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "case": {"output_root": "outputs"},
                "twist": twist,
                "paths": {
                    "input_file": "openmx.dat",
                    "H_file": "H.dat",
                    "S_file": "S.dat",
                },
                "symmetry": {"valley": "Gamma", "q_shell": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
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


def test_resolved_structure_rejects_unspun_source_for_spinful_system(tmp_path):
    config_path = _write_system_case(tmp_path, spin=True, matrix_dimension=4)
    _write_sparse_source(tmp_path / "H.npz", 4, metadata_dimension=4)
    _write_sparse_source(tmp_path / "S.npz", 4, metadata_dimension=4)
    config = Config.from_yaml(str(config_path))

    with pytest.raises(ValueError, match=r"source basis dimension.*4.*expected.*8"):
        resolve_structure_input(config)


def test_canonical_structure_rejects_rectangular_bravais_metric(tmp_path):
    config = Config.from_yaml(
        str(
            _write_system_case(
                tmp_path,
                cell=[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 20.0]],
            )
        )
    )

    with pytest.raises(ValueError, match="Cannot infer supported 2D Bravais"):
        resolve_structure_input(config)


def test_infer_2d_bravais_uses_full_3d_vectors_for_tilted_cells():
    tilted_hex = np.array(
        [
            [3.0, 0.0, 0.0],
            [1.5, 2.25, 1.2990381057],
            [0.0, -10.0, 17.3205080767],
        ]
    )
    tilted_square = np.array(
        [
            [3.0, 0.0, 1.0],
            [-1.0 / 3.0, 4.0 * np.sqrt(5.0) / 3.0, 1.0],
            [-6.0, 0.0, 18.0],
        ]
    )

    assert infer_2d_bravais(tilted_hex) == "hex"
    assert infer_2d_bravais(tilted_square) == "square"


def test_metadata_less_npz_allows_structurally_zero_tail_with_expected_dimension(tmp_path):
    config_path = _write_system_case(tmp_path, matrix_dimension=4)
    _write_sparse_source(tmp_path / "H.npz", 4, coordinate_indices=[0])
    _write_sparse_source(tmp_path / "S.npz", 4, coordinate_indices=[0])
    config = Config.from_yaml(str(config_path))

    resolved = resolve_structure_input(config)

    assert resolved.hamiltonian_basis_dimension == 4
    assert resolved.overlap_basis_dimension == 4
    assert resolved.hamiltonian_dimension_provenance == "expected_with_legacy_coordinate_bounds"
    assert resolved.overlap_dimension_provenance == "expected_with_legacy_coordinate_bounds"


@pytest.mark.parametrize("bad_index", [-1, 4])
def test_metadata_less_npz_rejects_coordinates_outside_expected_dimension(tmp_path, bad_index):
    config_path = _write_system_case(tmp_path, matrix_dimension=4)
    _write_sparse_source(tmp_path / "H.npz", 4, coordinate_indices=[bad_index])

    with pytest.raises(ValueError, match=r"index outside \[0, 4\)"):
        resolve_structure_input(Config.from_yaml(str(config_path)))


def test_versioned_npz_basis_dimension_is_exact_and_must_match_expected(tmp_path):
    config_path = _write_system_case(tmp_path, matrix_dimension=4)
    _write_sparse_source(tmp_path / "H.npz", 4, metadata_dimension=5, coordinate_indices=[0])
    _write_sparse_source(tmp_path / "S.npz", 4, metadata_dimension=5, coordinate_indices=[0])
    config = Config.from_yaml(str(config_path))

    with pytest.raises(ValueError, match=r"source basis dimension.*5.*expected.*4"):
        resolve_structure_input(config)


@pytest.mark.parametrize(
    ("metadata_schema", "metadata_version", "message"),
    [
        ("tapw.sparse-realspace.unknown", 1, "Unsupported sparse NPZ schema"),
        (SPARSE_NPZ_SCHEMA, 2, "Unsupported sparse NPZ schema version"),
    ],
)
def test_versioned_npz_rejects_unknown_schema_or_version(
    tmp_path,
    metadata_schema,
    metadata_version,
    message,
):
    config_path = _write_system_case(tmp_path, matrix_dimension=4)
    _write_sparse_source(
        tmp_path / "H.npz",
        4,
        metadata_dimension=4,
        metadata_schema=metadata_schema,
        metadata_version=metadata_version,
    )

    with pytest.raises(ValueError, match=message):
        resolve_structure_input(Config.from_yaml(str(config_path)))


def test_dat_header_basis_dimension_is_exact(tmp_path):
    config_path = _write_system_case(tmp_path, matrix_dimension=4)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["system"]["hamiltonian"] = "H.dat"
    payload["system"]["overlap"] = "S.dat"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    _write_sparse_dat(tmp_path / "H.dat", 4, coordinate_indices=[0])
    _write_sparse_dat(tmp_path / "S.dat", 4, coordinate_indices=[0])

    resolved = resolve_structure_input(Config.from_yaml(str(config_path)))

    assert resolved.hamiltonian_basis_dimension == 4
    assert resolved.overlap_basis_dimension == 4
    assert resolved.hamiltonian_dimension_provenance == "dat_header"
    assert resolved.overlap_dimension_provenance == "dat_header"


def test_dat_headers_reject_hamiltonian_overlap_dimension_mismatch(tmp_path):
    config_path = _write_system_case(tmp_path, matrix_dimension=4)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["system"]["hamiltonian"] = "H.dat"
    payload["system"]["overlap"] = "S.dat"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    _write_sparse_dat(tmp_path / "H.dat", 4, coordinate_indices=[0])
    _write_sparse_dat(tmp_path / "S.dat", 5, coordinate_indices=[0])

    with pytest.raises(ValueError, match=r"H/S basis dimension mismatch.*4.*5"):
        resolve_structure_input(Config.from_yaml(str(config_path)))


def test_source_identity_records_exact_dimension_provenance(tmp_path):
    config_path = _write_system_case(tmp_path, matrix_dimension=4)
    _write_sparse_source(tmp_path / "H.npz", 4, metadata_dimension=4, coordinate_indices=[0])
    _write_sparse_source(tmp_path / "S.npz", 4, metadata_dimension=4, coordinate_indices=[0])

    resolved = resolve_structure_input(Config.from_yaml(str(config_path)))

    assert resolved.hamiltonian_dimension_provenance == "versioned_npz_metadata"
    assert resolved.overlap_dimension_provenance == "versioned_npz_metadata"
    assert resolved.identity_components["hamiltonian_basis_dimension"] == 4
    assert resolved.identity_components["hamiltonian_dimension_provenance"] == "versioned_npz_metadata"


def test_resolved_structure_optional_overlap_has_none_dimension_hash_and_provenance(tmp_path):
    config = Config.from_yaml(str(_write_system_case(tmp_path, overlap=False)))

    resolved = resolve_structure_input(config)

    assert resolved.overlap_basis_dimension is None
    assert resolved.overlap_dimension_provenance is None
    assert resolved.identity_components["overlap_hash"] is None
    assert resolved.identity_components["overlap_basis_dimension"] is None
    assert resolved.identity_components["overlap_dimension_provenance"] is None


def test_resolved_structure_rejects_missing_or_extra_orbital_species(tmp_path):
    config_path = _write_system_case(tmp_path, orbitals={"Mo": "s1", "I": "p1"})
    config = Config.from_yaml(str(config_path))

    with pytest.raises(ValueError, match=r"orbitals.*missing.*Te.*extra.*I"):
        resolve_structure_input(config)


def test_resolved_structure_rejects_hamiltonian_and_overlap_dimension_mismatch(tmp_path):
    config_path = _write_system_case(tmp_path, matrix_dimension=4)
    _write_sparse_source(tmp_path / "H.npz", 4, metadata_dimension=4, coordinate_indices=[0])
    _write_sparse_source(tmp_path / "S.npz", 5, metadata_dimension=5, coordinate_indices=[0])
    config = Config.from_yaml(str(config_path))

    with pytest.raises(ValueError, match=r"H/S basis dimension mismatch.*4.*5"):
        resolve_structure_input(config)


def test_resolved_structure_rejects_source_dimension_incompatible_with_orbitals(tmp_path):
    config_path = _write_system_case(tmp_path, matrix_dimension=5)
    _write_sparse_source(tmp_path / "H.npz", 5, metadata_dimension=5, coordinate_indices=[0])
    _write_sparse_source(tmp_path / "S.npz", 5, metadata_dimension=5, coordinate_indices=[0])
    config = Config.from_yaml(str(config_path))

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


def test_raw_h_input_identity_includes_canonical_orbitals_and_resolved_structure(tmp_path):
    from tapw.workflows.symmetry import _source_input_hash

    config_path = _write_system_case(tmp_path)
    first = Config.from_yaml(str(config_path))
    second = Config.from_yaml(str(config_path))
    second.system_input = replace(
        second.system_input,
        orbitals=(("Mo", "p1"), ("Te", "s1")),
    )

    assert _source_input_hash(first) != _source_input_hash(second)


def test_infer_2d_bravais_is_pure_and_fail_closed(monkeypatch):
    monkeypatch.setenv("TAPW_BRAVAIS", "square")
    hex_cell = np.array([[3.0, 0.0, 0.0], [1.5, 2.598076211, 0.0], [0.0, 0.0, 20.0]])
    square_cell = np.diag([3.0, 3.0, 20.0])

    assert infer_2d_bravais(hex_cell) == "hex"
    assert infer_2d_bravais(square_cell) == "square"
    with pytest.raises(ValueError, match="Cannot infer supported 2D Bravais"):
        infer_2d_bravais(np.diag([3.0, 4.0, 20.0]))
    assert os.environ["TAPW_BRAVAIS"] == "square"
    with pytest.raises(ValueError, match="Cannot infer supported 2D Bravais"):
        infer_2d_bravais(np.array([[3.0, 0.0, 0.0], [0.7, 3.4, 0.0], [0.0, 0.0, 20.0]]))


@pytest.mark.parametrize("bravais", ["hex", "rect"])
def test_legacy_openmx_structure_does_not_read_or_write_global_bravais(tmp_path, monkeypatch, bravais):
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

    structure = OpenMXFile(source, twist_index=1, spin=False, bravais=bravais)

    assert structure.bravais == bravais
    assert "TAPW_BRAVAIS" not in os.environ


def test_shared_structure_loader_records_canonical_resolution_on_config(tmp_path):
    config = Config.from_yaml(str(_write_system_case(tmp_path)))

    structure = load_structure_from_config(config)

    assert structure is config.resolved_structure_input.structure
    assert config.resolved_structure_input.source_identity
    assert config.twist.bravais == "hex"
    assert config.compute.bravais == "hex"


def test_legacy_square_without_explicit_bravais_matches_direct_auto_detection(tmp_path):
    config = Config.from_yaml(str(_write_openmx_case(tmp_path, square=True)))

    direct = OpenMXFile(
        config.system_input.structure,
        twist_index=config.system_input.twist_index,
        spin=config.system_input.spin,
        bravais=None,
    )
    shared = load_structure_from_config(config)

    assert config.system_input.explicit_bravais is None
    assert direct.bravais == "square"
    assert shared.bravais == direct.bravais
    assert config.resolved_structure_input.structure is shared
    assert config.resolved_structure_input.identity_components["site_order"] == ["Te"]
    assert config.resolved_structure_input.identity_components["basis_order"] == ["site:1:Te:orbital:0"]
    from tapw.identity import hash_file, hash_mapping
    from tapw.workflows.symmetry import _source_input_hash

    historical = hash_mapping(
        {
            "H_file": hash_file(config.system_input.hamiltonian),
            "S_file": hash_file(config.system_input.overlap),
            "input_file": hash_file(config.system_input.structure),
        }
    )
    assert config.resolved_structure_input.source_identity == historical
    assert _source_input_hash(config) == historical


def test_equivalent_canonical_and_legacy_sources_share_internal_site_basis_mapping(tmp_path):
    canonical_dir = tmp_path / "canonical"
    legacy_dir = tmp_path / "legacy"
    canonical_dir.mkdir()
    legacy_dir.mkdir()
    canonical_path = _write_system_case(
        canonical_dir,
        symbols=("Te",),
        orbitals={"Te": "s1"},
        matrix_dimension=1,
        cell=np.diag([3.0, 3.0, 20.0]),
    )
    legacy_path = _write_openmx_case(legacy_dir, square=True)

    canonical_config = Config.from_yaml(str(canonical_path))
    legacy_config = Config.from_yaml(str(legacy_path))
    load_structure_from_config(canonical_config)
    load_structure_from_config(legacy_config)

    canonical_mapping = canonical_config.resolved_structure_input.identity_components
    legacy_mapping = legacy_config.resolved_structure_input.identity_components
    assert canonical_mapping["site_order"] == legacy_mapping["site_order"]
    assert canonical_mapping["basis_order"] == legacy_mapping["basis_order"]


@pytest.mark.parametrize("source_kind", ["canonical", "legacy"])
def test_five_public_workflow_loads_share_source_site_and_basis_identity(tmp_path, source_kind):
    if source_kind == "canonical":
        config_path = _write_system_case(tmp_path)
    else:
        config_path = _write_openmx_case(tmp_path, square=True)

    contracts = []
    for _workflow in ("run", "symm", "symm-rep", "topo", "orbital"):
        config = Config.from_yaml(str(config_path))
        load_structure_from_config(config)
        resolved = config.resolved_structure_input
        contracts.append(
            (
                resolved.source_identity,
                resolved.identity_components["site_order"],
                resolved.identity_components["basis_order"],
            )
        )

    assert all(contract == contracts[0] for contract in contracts[1:])


def test_canonical_save_reload_preserves_resolved_source_identity(tmp_path):
    source_dir = tmp_path / "source"
    saved_dir = tmp_path / "saved"
    source_dir.mkdir()
    saved_dir.mkdir()
    config = Config.from_yaml(str(_write_system_case(source_dir)))
    original = resolve_structure_input(config)
    saved_path = saved_dir / "config.yaml"

    config.save_yaml(str(saved_path))
    reloaded = resolve_structure_input(Config.from_yaml(str(saved_path)))

    assert reloaded.source_identity == original.source_identity


@pytest.mark.parametrize("bravais", ["square", "rect"])
def test_shared_structure_loader_passes_explicit_bravais_to_legacy_factory(tmp_path, bravais):
    config_path = tmp_path / "config.yaml"
    for name in ("openmx.dat", "H.dat", "S.dat"):
        (tmp_path / name).write_text("placeholder\n", encoding="utf-8")
    config_path.write_text(
        yaml.safe_dump(
            {
                "case": {"output_root": "outputs"},
                "twist": {
                    "bravais": bravais,
                    "twist_index_m": 2,
                    "twist_layer": [1, 1],
                    "spin": False,
                },
                "paths": {
                    "input_file": "openmx.dat",
                    "H_file": "H.dat",
                    "S_file": "S.dat",
                },
                "symmetry": {"valley": "Gamma", "q_shell": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = Config.from_yaml(str(config_path))
    calls = []

    class FakeLegacyStructure:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.bravais = kwargs["bravais"]

    structure = load_structure_from_config(config, legacy_factory=FakeLegacyStructure)

    assert structure.bravais == bravais
    assert calls == [
        {
            "file_path": config.system_input.structure,
            "twist_index": 2,
            "spin": False,
            "bravais": bravais,
        }
    ]

from pathlib import Path

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import write

from tapw import chern_post, cli, orbital_analysis_tool, symm_rep
from tapw.io.structure import load_structure_from_config as real_structure_loader


class _WorkflowBoundaryReached(RuntimeError):
    pass


class _SilentLogger:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None


def _write_all_workflow_config(tmp_path: Path, *, source_kind: str) -> Path:
    if source_kind == "canonical":
        atoms = Atoms(
            symbols=["Te"],
            positions=[[0.0, 0.0, 1.0]],
            cell=np.diag([3.0, 3.0, 20.0]),
            pbc=[True, True, False],
        )
        write(tmp_path / "POSCAR", atoms, format="vasp")
        for name in ("H.npz", "S.npz"):
            np.savez(
                tmp_path / name,
                **{
                    "(0, 0, 0)_row": np.array([0], dtype=np.int32),
                    "(0, 0, 0)_col": np.array([0], dtype=np.int32),
                    "(0, 0, 0)_val": np.array([1.0 + 0.0j]),
                },
            )
        source = {"system": {
            "output": "outputs",
            "structure": "POSCAR",
            "hamiltonian": "H.npz",
            "overlap": "S.npz",
            "orbitals": {"Te": "s1"},
            "twist_index": 1,
            "layers": [1, 1],
            "spin": False,
        }}
    else:
        (tmp_path / "openmx.dat").write_text(
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
0.0 3.0 0.0
0.0 0.0 20.0
Atoms.UnitVectors>
""",
            encoding="utf-8",
        )
        sparse_dat = "! source\n1\n1\n1\n0 0 0 1 1 1.0 0.0\n"
        for name in ("H.dat", "S.dat"):
            (tmp_path / name).write_text(sparse_dat, encoding="utf-8")
        source = {
            "case": {"output_root": "outputs"},
            "twist": {"twist_index_m": 1, "twist_layer": [1, 1], "spin": False},
            "paths": {
                "input_file": "openmx.dat",
                "H_file": "H.dat",
                "S_file": "S.dat",
            },
        }
    payload = {
        **source,
        "bands": {
            "valley": "Gamma",
            "q_shell": 1,
            "num_processes": 1,
            "kpath": {
                "labels": ["G", "M"],
                "points_per_segment": 1,
                "coordinates": {"G": [0.0, 0.0], "M": [0.5, 0.0]},
            },
        },
        "symmetry": {
            "valley": "Gamma",
            "q_shell": 1,
            "efermi": 0.0,
            "num_processes": 1,
            "num_bands": 1,
            "representation": {
                "points": {"G": [0.0, 0.0]},
                "degeneracy_tol": 0.002,
            },
        },
        "topology": {
            "valley": "Gamma",
            "q_shell": 1,
            "num_processes": 1,
            "mesh": {
                "n_b1": 2,
                "n_b2": 2,
                "range_b1": [-0.5, 0.5],
                "range_b2": [-0.5, 0.5],
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _contract(config):
    resolved = config.resolved_structure_input
    return (
        resolved.source_identity,
        tuple(resolved.identity_components["site_order"]),
        tuple(resolved.identity_components["basis_order"]),
    )


@pytest.mark.parametrize("source_kind", ["canonical", "legacy"])
def test_all_public_workflow_entries_consume_same_resolved_structure_contract(
    tmp_path,
    monkeypatch,
    source_kind,
):
    config_path = _write_all_workflow_config(tmp_path, source_kind=source_kind)
    records = {}

    def capture(label):
        def loader(config, **kwargs):
            real_structure_loader(config, **kwargs)
            records[label] = _contract(config)
            raise _WorkflowBoundaryReached(label)

        return loader

    monkeypatch.setattr(cli, "setup_logging", lambda *_args, **_kwargs: _SilentLogger())
    monkeypatch.setattr(cli, "shutdown_parallel_runtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "_mpi_world_rank_size", lambda: (0, 1))
    for label, entry in (
        ("run", cli.main_calc),
        ("symm", cli.main_symm),
        ("topo", cli.main_topo),
    ):
        monkeypatch.setattr(cli, "load_structure_from_config", capture(label))
        assert entry(["-c", str(config_path)], finalize=False) == 1

    monkeypatch.setattr(chern_post, "load_structure_from_config", capture("chern-post"))
    with pytest.raises(_WorkflowBoundaryReached, match="chern-post"):
        chern_post.main(["-c", str(config_path), "-b", "0"])

    request = symm_rep.ConfigSymmRepRequest(
        config_path=config_path,
        symmetry_dir=tmp_path / "symmetry",
        output_dir=tmp_path / "symm_rep",
        fermi_energy=0.0,
        num_bands=1,
        degeneracy_tol=0.002,
        points={"G": (0.0, 0.0, 0.0)},
        identity_schema="test.identity.v1",
        input_hash="input",
        config_hash="config",
        basis_hash="basis",
        package_version="test",
        schema_version=1,
        cache_enabled=False,
    )
    monkeypatch.setattr(symm_rep, "resolve_config_request", lambda _path: request)
    monkeypatch.setattr(symm_rep, "load_structure_from_config", capture("symm-rep"))
    with pytest.raises(_WorkflowBoundaryReached, match="symm-rep"):
        symm_rep.main(["-c", str(config_path)])

    result_dir = tmp_path / "orbital-result"
    result_dir.mkdir()
    np.savetxt(result_dir / "band_CBM_Gamma_valley.txt", np.array([[0.0]]))
    np.save(result_dir / "vec_CBM_Gamma_valley.npy", np.ones((1, 1, 1), dtype=np.complex128))
    monkeypatch.setattr(
        orbital_analysis_tool,
        "load_structure_from_config",
        capture("orbital"),
    )
    with pytest.raises(SystemExit) as exc_info:
        orbital_analysis_tool.main(
            [str(result_dir), "-c", str(config_path), "--band", "CBM", "--quiet"]
        )
    assert exc_info.value.code == 1

    expected_labels = {"run", "symm", "topo", "chern-post", "symm-rep", "orbital"}
    assert set(records) == expected_labels
    reference = records["run"]
    assert all(records[label] == reference for label in expected_labels)

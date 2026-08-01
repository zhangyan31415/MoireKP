from __future__ import annotations

import builtins
import copy
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

import kp.cli as cli


class _GuardedHamk:
    ndim = 3
    shape = (3, 4, 4)

    def __init__(self) -> None:
        self.accessed: list[int] = []

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError(f"unexpected hamk key {key!r}")
        self.accessed.append(key)
        if key == 2:
            raise AssertionError("project read an unselected k-point")
        return np.eye(4, dtype=np.complex128) * float(key + 1)


def _canonical_case_cfg(
    cfg: dict,
    *,
    tmp_path: Path | None = None,
    nk: int | None = None,
) -> dict:
    out = copy.deepcopy(cfg)
    out.setdefault("case", {"profile": "K1", "q_shell": "q06", "output_root": "outputs"})
    if tmp_path is not None:
        if nk is None:
            raise ValueError("nk is required when creating project reference artifacts")
        material = out.setdefault("material", {})
        material.setdefault("band_file", "bands.txt")
        material.setdefault("kpoints_file", "kpoints.npy")
        out.setdefault("kpath", {"tmat": np.eye(3).tolist()})

        def source_path(field: str, default: str) -> Path:
            raw = Path(str(material.get(field, default)))
            resolved = raw if raw.is_absolute() else tmp_path / raw
            resolved.parent.mkdir(parents=True, exist_ok=True)
            return resolved

        np.save(
            source_path("hamk_file", "hamk.npy"),
            np.zeros((nk, 4, 4), dtype=np.complex128),
        )
        np.save(
            source_path("qset1_file", "q1.npy"),
            np.zeros((1, 2), dtype=float),
        )
        np.save(
            source_path("qset2_file", "q2.npy"),
            np.zeros((1, 2), dtype=float),
        )
        (tmp_path / "bands.txt").write_text(
            "".join("0.0 1.0\n" for _ in range(nk)),
            encoding="utf-8",
        )
        np.save(tmp_path / "kpoints.npy", np.zeros((nk, 3), dtype=float))
    return out


def test_project_spin_slice_honors_k_indices_before_materializing(monkeypatch, tmp_path: Path) -> None:
    guarded = _GuardedHamk()
    q = np.zeros((1, 2), dtype=float)
    projected_blocks: list[np.ndarray] = []

    monkeypatch.setattr(cli, "_load_hamk_with_energy_unit", lambda *_args, **_kwargs: guarded)
    monkeypatch.setattr(cli, "load_Q_sets", lambda *_args, **_kwargs: (q, q.copy()))

    def fake_project_heff_full(ham, *_args, **_kwargs):
        block = np.asarray(ham)
        projected_blocks.append(block.copy())
        assert block.shape == (2, 2)
        return (
            np.eye(2, dtype=np.complex128),
            np.array([0.0, 1.0], dtype=float),
            np.eye(2, dtype=np.complex128),
            SimpleNamespace(hermiticity_residual=0.0),
        )

    monkeypatch.setattr(cli, "project_heff_full", fake_project_heff_full)
    monkeypatch.setattr(cli, "plot_eigs_scatter", lambda *_args, out, **_kwargs: Path(out).write_text("plot"))
    real_import = builtins.__import__

    def fail_parallel_import(name, *args, **kwargs):
        if name in {"joblib", "tqdm"}:
            raise AssertionError(f"workers=1 should not import {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_parallel_import)

    cfg = {
        "material": {
            "hamk_file": "unused.npy",
            "qset1_file": "unused_q1.npy",
            "qset2_file": "unused_q2.npy",
            "spin": "up",
            "energy_unit": "eV",
            "num_layers": 2,
            "num_orb_per_layer": [1],
        },
        "plot": {"hamk_index": 0},
        "project": {
            "mode": "K1",
            "workers": 1,
            "k_indices": [1],
            "downfold_method": "first_order",
            "out_dir": "project",
            "nlow_state_list": [[0], [0]],
            "norb_fix_list": [[[[0, 1.0]]], [[[0, 1.0]]]],
        },
    }
    cfg_path = tmp_path / "source.yaml"
    cfg_path.write_text(
        yaml.safe_dump(_canonical_case_cfg(cfg, tmp_path=tmp_path, nk=3)),
        encoding="utf-8",
    )

    request = cli._prepare_cli_selection_request(str(cfg_path))
    source = np.load(tmp_path / "unused.npy", mmap_mode="r", allow_pickle=False)
    assert request.selection_input.source_hamiltonian_hash == cli.hash_array(source)
    assert guarded.accessed == []
    cli.cmd_project_from_config(str(cfg_path))

    assert guarded.accessed == [0, 1]
    assert len(projected_blocks) == 1
    wavefunctions = np.load(tmp_path / "project" / "wavefunctions.npz")
    np.testing.assert_array_equal(wavefunctions["k_indices"], np.array([1], dtype=int))
    with np.load(tmp_path / "project" / "basis.npz", allow_pickle=True) as basis:
        for field in (
            "identity_schema",
            "input_hash",
            "config_hash",
            "basis_hash",
            "package_version",
            "schema_version",
            "k_indices_hash",
            "heff_hash",
        ):
            assert field in basis.files
            assert field in wavefunctions.files
            assert np.asarray(basis[field]).item() == np.asarray(wavefunctions[field]).item()


def test_selection_identity_rejects_fortran_order_source(tmp_path: Path) -> None:
    cfg = {
        "material": {
            "hamk_file": "hamk.npy",
            "qset1_file": "q1.npy",
            "qset2_file": "q2.npy",
            "spin": "up",
            "energy_unit": "eV",
            "num_layers": 2,
            "num_orb_per_layer": [1],
        },
        "project": {
            "mode": "K1",
            "workers": 1,
            "downfold_method": "first_order",
            "out_dir": "project",
            "nlow_state_list": [[0], [0]],
            "norb_fix_list": [[[[0, 1.0]]], [[[0, 1.0]]]],
        },
    }
    cfg_path = tmp_path / "source.yaml"
    cfg_path.write_text(
        yaml.safe_dump(_canonical_case_cfg(cfg, tmp_path=tmp_path, nk=2)),
        encoding="utf-8",
    )
    np.save(
        tmp_path / "hamk.npy",
        np.asfortranarray(np.zeros((2, 4, 4), dtype=np.complex128)),
    )

    with pytest.raises(ValueError, match="C-contiguous"):
        cli._prepare_cli_selection_request(str(cfg_path))


def test_project_records_fixed_spin_without_saving_operator(monkeypatch, tmp_path: Path) -> None:
    class FullPathHamk:
        ndim = 3
        shape = (1, 4, 4)

        def __getitem__(self, key):
            if not isinstance(key, int):
                raise TypeError(f"unexpected hamk key {key!r}")
            return np.eye(4, dtype=np.complex128)

    q = np.zeros((1, 2), dtype=float)
    seen: dict[str, object] = {}

    monkeypatch.setattr(cli, "_load_hamk_with_energy_unit", lambda *_args, **_kwargs: FullPathHamk())
    monkeypatch.setattr(cli, "load_Q_sets", lambda *_args, **_kwargs: (q, q.copy()))

    def fake_project_heff_full(*_args, **_kwargs):
        seen.update(_kwargs)
        return (
            np.eye(2, dtype=np.complex128),
            np.array([0.0, 1.0], dtype=float),
            np.eye(2, dtype=np.complex128),
            SimpleNamespace(hermiticity_residual=0.0),
        )

    monkeypatch.setattr(cli, "project_heff_full", fake_project_heff_full)
    monkeypatch.setattr(cli, "plot_eigs_scatter", lambda *_args, out, **_kwargs: Path(out).write_text("plot"))

    cfg = {
        "material": {
            "hamk_file": "unused.npy",
            "qset1_file": "unused_q1.npy",
            "qset2_file": "unused_q2.npy",
            "spin": "down",
            "energy_unit": "eV",
            "num_layers": 2,
            "num_orb_per_layer": [1],
        },
        "plot": {"hamk_index": 0},
        "project": {
            "mode": "K1",
            "workers": 1,
            "downfold_method": "first_order",
            "out_dir": "project",
            "nlow_state_list": [[0], [0]],
            "norb_fix_list": [[[[0, 1.0]]], [[[0, 1.0]]]],
        },
    }
    cfg_path = tmp_path / "source.yaml"
    cfg_path.write_text(
        yaml.safe_dump(_canonical_case_cfg(cfg, tmp_path=tmp_path, nk=1)),
        encoding="utf-8",
    )

    cli.cmd_project_from_config(str(cfg_path))

    assert "return_spin_operator" not in seen
    assert "spin_operator_sign" not in seen
    assert not (tmp_path / "project" / "spin_operator.npy").exists()
    with np.load(tmp_path / "project" / "wavefunctions.npz", allow_pickle=False) as payload:
        assert np.asarray(payload["spin_convention"]).item() == "down"
        assert "spin_operator" not in payload.files


def test_project_embeds_mixed_spin_operator_in_wavefunctions_archive(monkeypatch, tmp_path: Path) -> None:
    class FullPathHamk:
        ndim = 3
        shape = (1, 4, 4)

        def __getitem__(self, key):
            if not isinstance(key, int):
                raise TypeError(f"unexpected hamk key {key!r}")
            return np.eye(4, dtype=np.complex128)

    q = np.zeros((1, 2), dtype=float)
    projected_spin = np.diag([0.75, -0.75]).astype(np.complex128)
    seen: dict[str, object] = {}

    monkeypatch.setattr(cli, "_load_hamk_with_energy_unit", lambda *_args, **_kwargs: FullPathHamk())
    monkeypatch.setattr(cli, "load_Q_sets", lambda *_args, **_kwargs: (q, q.copy()))

    def fake_project_heff_full(*_args, **kwargs):
        seen.update(kwargs)
        return (
            np.eye(2, dtype=np.complex128),
            np.array([0.0, 1.0], dtype=float),
            np.eye(2, dtype=np.complex128),
            SimpleNamespace(hermiticity_residual=0.0),
            projected_spin,
        )

    monkeypatch.setattr(cli, "project_heff_full", fake_project_heff_full)
    monkeypatch.setattr(cli, "plot_eigs_scatter", lambda *_args, out, **_kwargs: Path(out).write_text("plot"))

    cfg = {
        "material": {
            "hamk_file": "unused.npy",
            "qset1_file": "unused_q1.npy",
            "qset2_file": "unused_q2.npy",
            "spin": "all",
            "energy_unit": "eV",
            "num_layers": 2,
            "num_orb_per_layer": [1],
        },
        "plot": {"hamk_index": 0},
        "project": {
            "mode": "K1",
            "workers": 1,
            "downfold_method": "first_order",
            "out_dir": "project",
            "nlow_state_list": [[0], [0]],
            "norb_fix_list": [[[[0, 1.0]]], [[[0, 1.0]]]],
        },
    }
    cfg_path = tmp_path / "source.yaml"
    cfg_path.write_text(
        yaml.safe_dump(_canonical_case_cfg(cfg, tmp_path=tmp_path, nk=1)),
        encoding="utf-8",
    )

    cli.cmd_project_from_config(str(cfg_path))

    assert seen["return_spin_operator"] is True
    assert "spin_operator_sign" not in seen
    assert not (tmp_path / "project" / "spin_operator.npy").exists()
    with np.load(tmp_path / "project" / "wavefunctions.npz", allow_pickle=False) as payload:
        assert np.asarray(payload["spin_convention"]).item() == "all"
        np.testing.assert_allclose(payload["spin_operator"], projected_spin[None, :, :])


def test_project_resolves_tapw_band_manifest_inputs(monkeypatch, tmp_path: Path) -> None:
    band_dir = tmp_path / "tapw" / "outputs" / "K1" / "q06" / "band"
    band_dir.mkdir(parents=True)
    manifest_path = band_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "tapw_band_outputs/v1",
                "files": {
                    "hamiltonian_k": "hamiltonian_k.npy",
                    "g_vectors_group1": "g_vectors_group1.npy",
                    "g_vectors_group2": "g_vectors_group2.npy",
                    "kpoints": "kpoints.npy",
                    "energies_vbm": "energies_vbm.txt",
                },
            }
        ),
        encoding="utf-8",
    )
    hamk = np.zeros((1, 4, 4), dtype=np.complex128)
    q = np.zeros((1, 2), dtype=float)
    np.save(band_dir / "hamiltonian_k.npy", hamk)
    np.save(band_dir / "g_vectors_group1.npy", q)
    np.save(band_dir / "g_vectors_group2.npy", q)
    np.save(band_dir / "kpoints.npy", np.zeros((1, 3), dtype=float))
    (band_dir / "energies_vbm.txt").write_text("0.0 1.0\n", encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_load_hamk(path, *_args, **_kwargs):
        seen["hamk"] = Path(path)
        return hamk

    def fake_load_qsets(qset1, qset2):
        seen["qset1"] = Path(qset1)
        seen["qset2"] = Path(qset2)
        return q, q.copy()

    monkeypatch.setattr(cli, "_load_hamk_with_energy_unit", fake_load_hamk)
    monkeypatch.setattr(cli, "load_Q_sets", fake_load_qsets)
    monkeypatch.setattr(
        cli,
        "project_heff_full",
        lambda *_args, **_kwargs: (
            np.eye(2, dtype=np.complex128),
            np.array([0.0, 1.0], dtype=float),
            np.eye(2, dtype=np.complex128),
            SimpleNamespace(hermiticity_residual=0.0),
        ),
    )
    monkeypatch.setattr(cli, "plot_eigs_scatter", lambda *_args, out, **_kwargs: Path(out).write_text("plot"))

    cfg = {
        "material": {
            "tapw_band_manifest": "tapw/outputs/K1/q06/band/manifest.json",
            "spin": "up",
            "energy_unit": "eV",
            "num_layers": 2,
            "num_orb_per_layer": [1],
        },
        "plot": {"hamk_index": 0},
        "kpath": {"tmat": np.eye(3).tolist()},
        "project": {
            "mode": "K1",
            "workers": 1,
            "k_indices": [0],
            "downfold_method": "first_order",
            "out_dir": "project",
            "nlow_state_list": [[0], [0]],
            "norb_fix_list": [[[[0, 1.0]]], [[[0, 1.0]]]],
        },
    }
    cfg_path = tmp_path / "source.yaml"
    cfg_path.write_text(
        yaml.safe_dump(_canonical_case_cfg(cfg, tmp_path=tmp_path, nk=1)),
        encoding="utf-8",
    )

    cli.cmd_project_from_config(str(cfg_path))

    assert seen == {
        "hamk": band_dir / "hamiltonian_k.npy",
        "qset1": band_dir / "g_vectors_group1.npy",
        "qset2": band_dir / "g_vectors_group2.npy",
    }


def test_active_indices_summary_flattens_sparse_layer_bands() -> None:
    assert cli._active_indices_from_project_cfg({"nlow_state_list": [[], [], [22]]}) == [22]
    assert cli._active_indices_from_project_cfg({"nlow_state_list": [[22], [22], []]}) == [22]


def test_plot_spin_down_uses_single_spin_block_indexing(monkeypatch, tmp_path: Path) -> None:
    q = np.zeros((1, 2), dtype=float)
    hamk = np.zeros((1, 4, 4), dtype=np.complex128)
    seen: dict[str, object] = {}

    monkeypatch.setattr(cli, "_load_hamk_with_energy_unit", lambda *_args, **_kwargs: hamk)
    monkeypatch.setattr(cli, "load_Q_sets", lambda *_args, **_kwargs: (q, q.copy()))

    def fake_get_H_block(ham, *_args, spin: str, **_kwargs):
        seen["shape"] = np.asarray(ham).shape
        seen["spin"] = spin
        eigs = np.empty(2, dtype=object)
        eigs[0] = np.array([-0.2])
        eigs[1] = np.array([-0.1])
        vecs = np.empty(2, dtype=object)
        vecs[0] = np.eye(1, dtype=np.complex128)
        vecs[1] = np.eye(1, dtype=np.complex128)
        return eigs, vecs, np.empty(0, dtype=object), np.empty(0, dtype=object)

    monkeypatch.setattr(cli, "get_H_block", fake_get_H_block)
    monkeypatch.setattr(cli, "plot_eigs_scatter", lambda *_args, out, **_kwargs: Path(out).write_text("plot"))

    cfg = {
        "material": {
            "hamk_file": "unused.npy",
            "qset1_file": "unused_q1.npy",
            "qset2_file": "unused_q2.npy",
            "spin": "down",
            "energy_unit": "eV",
            "num_layers": 2,
            "num_layer_list": [1, 1],
            "num_orb_per_layer": [1],
        },
        "plot": {
            "mode": "K1",
            "hamk_index": 0,
            "out": "plot.png",
            "nlow_state_list": [[0], [0]],
            "norb_fix_list": [[[[0, 1.0]]], [[[0, 1.0]]]],
        },
    }
    cfg_path = tmp_path / "source.yaml"
    cfg_path.write_text(
        yaml.safe_dump(_canonical_case_cfg(cfg, tmp_path=tmp_path, nk=1)),
        encoding="utf-8",
    )

    cli.cmd_plot_from_config(str(cfg_path))

    assert seen == {"shape": (2, 2), "spin": "up"}


def test_project_full_path_does_not_emit_k_indices_selection_file(monkeypatch, tmp_path: Path) -> None:
    class FullPathHamk:
        ndim = 3
        shape = (2, 4, 4)

        def __getitem__(self, key):
            if not isinstance(key, int):
                raise TypeError(f"unexpected hamk key {key!r}")
            return np.eye(4, dtype=np.complex128) * float(key + 1)

    q = np.zeros((1, 2), dtype=float)

    monkeypatch.setattr(cli, "_load_hamk_with_energy_unit", lambda *_args, **_kwargs: FullPathHamk())
    monkeypatch.setattr(cli, "load_Q_sets", lambda *_args, **_kwargs: (q, q.copy()))
    monkeypatch.setattr(
        cli,
        "project_heff_full",
        lambda *_args, **_kwargs: (
            np.eye(2, dtype=np.complex128),
            np.array([0.0, 1.0], dtype=float),
            np.eye(2, dtype=np.complex128),
            SimpleNamespace(hermiticity_residual=0.0),
        ),
    )
    monkeypatch.setattr(cli, "plot_eigs_scatter", lambda *_args, out, **_kwargs: Path(out).write_text("plot"))

    cfg = {
        "material": {
            "hamk_file": "unused.npy",
            "qset1_file": "unused_q1.npy",
            "qset2_file": "unused_q2.npy",
            "spin": "up",
            "energy_unit": "eV",
            "num_layers": 2,
            "num_orb_per_layer": [1],
        },
        "plot": {"hamk_index": 0},
        "project": {
            "mode": "K1",
            "workers": 1,
            "downfold_method": "first_order",
            "out_dir": "project",
            "nlow_state_list": [[0], [0]],
            "norb_fix_list": [[[[0, 1.0]]], [[[0, 1.0]]]],
        },
    }
    cfg_path = tmp_path / "source.yaml"
    cfg_path.write_text(
        yaml.safe_dump(_canonical_case_cfg(cfg, tmp_path=tmp_path, nk=2)),
        encoding="utf-8",
    )

    cli.cmd_project_from_config(str(cfg_path))

    assert np.loadtxt(tmp_path / "project" / "eigvals.txt").shape == (2, 2)
    assert not (tmp_path / "project" / "k_indices.npy").exists()
    wavefunctions = np.load(tmp_path / "project" / "wavefunctions.npz")
    np.testing.assert_array_equal(wavefunctions["k_indices"], np.array([0, 1], dtype=int))


def test_parallel_project_uses_worker_initializer_without_hamk_payload(monkeypatch, tmp_path: Path) -> None:
    class FullPathHamk:
        ndim = 3
        shape = (9, 4, 4)

        def __getitem__(self, key):
            if not isinstance(key, int):
                raise TypeError(f"unexpected hamk key {key!r}")
            return np.eye(4, dtype=np.complex128) * float(key + 1)

    q = np.zeros((1, 2), dtype=float)
    hamk = FullPathHamk()
    seen: dict[str, object] = {"projected": []}

    monkeypatch.setattr(cli, "_load_hamk_with_energy_unit", lambda *_args, **_kwargs: hamk)
    monkeypatch.setattr(cli, "load_hamk", lambda *_args, **_kwargs: hamk)
    monkeypatch.setattr(cli, "load_Q_sets", lambda *_args, **_kwargs: (q, q.copy()))

    def fake_project_heff_full(ham, *_args, **_kwargs):
        value = float(np.asarray(ham)[0, 0].real)
        seen["projected"].append(value)
        return (
            np.eye(2, dtype=np.complex128) * value,
            np.array([value, value + 1.0], dtype=float),
            np.eye(2, dtype=np.complex128),
            SimpleNamespace(hermiticity_residual=0.0),
        )

    class FakeParallel:
        def __init__(self, **kwargs):
            seen["parallel_kwargs"] = kwargs

        def __call__(self, tasks):
            init = seen["parallel_kwargs"]["initializer"]
            initargs = seen["parallel_kwargs"]["initargs"]
            seen["initializer_context"] = initargs[0]
            init(*initargs)
            outputs = []
            for func, args, kwargs in tasks:
                seen.setdefault("batches", []).append(list(args[0]))
                outputs.append(func(*args, **kwargs))
            return outputs

    def fake_delayed(func):
        def wrap(*args, **kwargs):
            return func, args, kwargs
        return wrap

    fake_joblib = ModuleType("joblib")
    fake_joblib.Parallel = FakeParallel
    fake_joblib.delayed = fake_delayed
    fake_tqdm = ModuleType("tqdm")

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            seen["progress_total"] = kwargs.get("total")
            seen["progress_unit"] = kwargs.get("unit")
            seen["progress_leave"] = kwargs.get("leave")
            seen["progress_updates"] = []

        def update(self, value):
            seen["progress_updates"].append(value)

        def close(self):
            seen["progress_closed"] = True

    fake_tqdm.tqdm = FakeTqdm
    monkeypatch.setitem(sys.modules, "joblib", fake_joblib)
    monkeypatch.setitem(sys.modules, "tqdm", fake_tqdm)
    monkeypatch.setattr(cli, "project_heff_full", fake_project_heff_full)
    monkeypatch.setattr(cli, "plot_eigs_scatter", lambda *_args, out, **_kwargs: Path(out).write_text("plot"))

    cfg = {
        "material": {
            "hamk_file": "unused.npy",
            "qset1_file": "unused_q1.npy",
            "qset2_file": "unused_q2.npy",
            "spin": "up",
            "energy_unit": "eV",
            "num_layers": 2,
            "num_orb_per_layer": [1],
        },
        "plot": {"hamk_index": 0},
        "project": {
            "mode": "K1",
            "workers": 4,
            "k_indices": list(range(9)),
            "downfold_method": "first_order",
            "out_dir": "project",
            "nlow_state_list": [[0], [0]],
            "norb_fix_list": [[[[0, 1.0]]], [[[0, 1.0]]]],
        },
    }
    cfg_path = tmp_path / "source.yaml"
    cfg_path.write_text(
        yaml.safe_dump(_canonical_case_cfg(cfg, tmp_path=tmp_path, nk=9)),
        encoding="utf-8",
    )

    cli.cmd_project_from_config(str(cfg_path))

    assert seen["parallel_kwargs"]["n_jobs"] == 4
    assert seen["parallel_kwargs"]["return_as"] == "generator"
    assert "hamk" not in seen["initializer_context"]
    assert seen["batches"] == [[0, 1], [2, 3], [4, 5], [6, 7], [8]]
    assert seen["progress_total"] == 9
    assert seen["progress_unit"] == "k"
    assert seen["progress_leave"] is True
    assert seen["progress_updates"] == [2, 2, 2, 2, 1]
    assert seen["progress_closed"] is True
    assert np.loadtxt(tmp_path / "project" / "eigvals.txt")[:, 0].tolist() == [
        float(i + 1) for i in range(9)
    ]


def test_project_cli_accepts_k_indices_override(monkeypatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "source.yaml"
    cfg_path.write_text("project: {}\n", encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_project(path: str, overrides: dict[str, object] | None = None) -> None:
        seen["path"] = path
        seen["overrides"] = overrides

    monkeypatch.setattr(cli, "cmd_project_from_config", fake_project)

    cli.main([
        "project",
        "-c",
        str(cfg_path),
        "--k-indices",
        "2,0",
        "--fail-on-near-pole",
        "--compute-pole-diagnostics",
        "--compute-condition-number",
    ])

    assert seen["path"] == str(cfg_path)
    assert seen["overrides"]["k_indices"] == "2,0"
    assert seen["overrides"]["fail_on_near_pole"] is True
    assert seen["overrides"]["compute_pole_diagnostics"] is True
    assert seen["overrides"]["compute_condition_number"] is True


def test_project_blas_threads_auto_scales_with_worker_count(monkeypatch) -> None:
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 128)

    assert cli._project_blas_threads({}, 1) == 8
    assert cli._project_blas_threads({}, 16) == 8
    assert cli._project_blas_threads({}, 32) == 4
    assert cli._project_blas_threads({}, 64) == 2
    assert cli._project_blas_threads({"blas_threads": 3}, 64) == 3


def test_project_cli_leaves_diagnostics_opt_in_by_default(monkeypatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "source.yaml"
    cfg_path.write_text("project: {}\n", encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_project(_path: str, overrides: dict[str, object] | None = None) -> None:
        seen["overrides"] = overrides

    monkeypatch.setattr(cli, "cmd_project_from_config", fake_project)

    cli.main(["project", "-c", str(cfg_path)])

    assert seen["overrides"]["fail_on_near_pole"] is None
    assert seen["overrides"]["compute_pole_diagnostics"] is None
    assert seen["overrides"]["compute_condition_number"] is None


def test_project_passes_downfold_diagnostic_options_when_supported(monkeypatch, tmp_path: Path) -> None:
    class FullPathHamk:
        ndim = 3
        shape = (1, 4, 4)

        def __getitem__(self, key):
            if not isinstance(key, int):
                raise TypeError(f"unexpected hamk key {key!r}")
            return np.eye(4, dtype=np.complex128)

    q = np.zeros((1, 2), dtype=float)
    seen: list[tuple[bool, bool, bool]] = []

    monkeypatch.setattr(cli, "_load_hamk_with_energy_unit", lambda *_args, **_kwargs: FullPathHamk())
    monkeypatch.setattr(cli, "load_Q_sets", lambda *_args, **_kwargs: (q, q.copy()))

    def fake_project_heff_full(
        *_args,
        compute_pole_diagnostics: bool,
        compute_condition_number: bool,
        **_kwargs,
    ):
        seen.append((
            compute_pole_diagnostics,
            compute_condition_number,
            bool(_kwargs["fail_on_near_pole"]),
        ))
        return (
            np.eye(2, dtype=np.complex128),
            np.array([0.0, 1.0], dtype=float),
            np.eye(2, dtype=np.complex128),
            SimpleNamespace(hermiticity_residual=0.0),
        )

    monkeypatch.setattr(cli, "project_heff_full", fake_project_heff_full)
    monkeypatch.setattr(cli, "plot_eigs_scatter", lambda *_args, out, **_kwargs: Path(out).write_text("plot"))

    cfg = {
        "material": {
            "hamk_file": "unused.npy",
            "qset1_file": "unused_q1.npy",
            "qset2_file": "unused_q2.npy",
            "spin": "up",
            "energy_unit": "eV",
            "num_layers": 2,
            "num_orb_per_layer": [1],
        },
        "plot": {"hamk_index": 0},
        "project": {
            "mode": "K1",
            "workers": 1,
            "downfold_method": "first_order",
            "out_dir": "project-default",
            "nlow_state_list": [[0], [0]],
            "norb_fix_list": [[[[0, 1.0]]], [[[0, 1.0]]]],
        },
    }
    cfg_path = tmp_path / "source.yaml"
    cfg_path.write_text(
        yaml.safe_dump(_canonical_case_cfg(cfg, tmp_path=tmp_path, nk=1)),
        encoding="utf-8",
    )
    cli.cmd_project_from_config(str(cfg_path))

    cfg["project"]["compute_pole_diagnostics"] = True
    cfg["project"]["compute_condition_number"] = True
    cfg["project"]["out_dir"] = "project-enabled"
    cfg_path.write_text(
        yaml.safe_dump(_canonical_case_cfg(cfg, tmp_path=tmp_path, nk=1)),
        encoding="utf-8",
    )
    cli.cmd_project_from_config(str(cfg_path))

    cfg["project"]["compute_pole_diagnostics"] = False
    cfg["project"]["compute_condition_number"] = False
    cfg["project"]["fail_on_near_pole"] = True
    cfg["project"]["out_dir"] = "project-fail-on-near-pole"
    cfg_path.write_text(
        yaml.safe_dump(_canonical_case_cfg(cfg, tmp_path=tmp_path, nk=1)),
        encoding="utf-8",
    )
    cli.cmd_project_from_config(str(cfg_path))

    assert seen == [(False, False, False), (True, True, False), (False, False, True)]

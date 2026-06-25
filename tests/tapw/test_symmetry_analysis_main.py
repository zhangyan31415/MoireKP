from pathlib import Path
from types import SimpleNamespace

import numpy as np

import tapw.cli as main_mod


class _FakeCompute(SimpleNamespace):
    def validate(self):
        return None


class _FakeLogger:
    def info(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


def _make_config(tmp_path: Path, *, mode: str, symmetry_enable: bool = False):
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    s_file = tmp_path / "S.dat"
    s_file.write_text("S\n", encoding="utf-8")

    compute = _FakeCompute(
        mode=mode,
        TAPW=True,
        valleys=[5],
        valley=5,
        n_g=6,
        num_processes=1,
        orthogonal_basis=False,
        symmetrize_hamiltonian=False,
        eigensolver="scipy",
        slepc_comm="self",
    )
    config = SimpleNamespace(
        twist=SimpleNamespace(
            twist_index_m=6,
            spin=False,
            num_layers=2,
            twist_layer=[1, 1],
        ),
        paths=SimpleNamespace(
            input_file=str(tmp_path / "openmx.dat"),
            output_dir=str(output_dir),
            H_file=str(tmp_path / "H.dat"),
            S_file=str(s_file),
            kpath_in=str(tmp_path / "KPATH.in"),
            kpath_out=str(tmp_path / "KPATH.out"),
        ),
        compute=compute,
        cluster=SimpleNamespace(
            layer_eps=0.5,
            layer_min_samples=1,
            sublayer_eps=0.5,
            sublayer_min_samples=1,
            atom_eps=0.3,
            atom_min_samples=2,
            period=2 * 3.14159,
            k_max=2,
        ),
        symmetry_analysis=SimpleNamespace(
            enable=symmetry_enable,
            valleys=None,
            tolerance=1.0e-2,
            output_dir="symmetry_analysis",
            debug=False,
        ),
    )
    config.validate = lambda: None
    return config


def _patch_main_dependencies(monkeypatch, config, events):
    args = SimpleNamespace(
        config="config.yaml",
        twist_index=None,
        output_dir=None,
        valleys=None,
        mode=None,
        n_g=None,
        num_chern=None,
        num_k1=None,
        num_k2=None,
        num_processes=None,
        blas_threads=None,
        parallel_impl=None,
        parallel_backend=None,
        vec_store=None,
        memmap_dir=None,
        kpoint_chunk_id=None,
        kpoint_chunk_count=None,
        developer_outputs=False,
        verbose=False,
    )
    monkeypatch.setattr(main_mod.Config, "from_yaml", lambda _: config)
    monkeypatch.setattr(main_mod, "_mpi_world_rank_size", lambda: (0, 1))
    monkeypatch.setattr(main_mod, "setup_logging", lambda *_args, **_kwargs: _FakeLogger())

    class FakeOpenMXFile:
        def __init__(self, *args, **kwargs):
            self.sorted_species_coordinates = []
            self.Tmat = np.eye(3)
            self.reciprocal_Tmat = np.eye(3)
            self.spin = False

        def display_properties(self):
            return None

    class FakeProcessor:
        def __init__(self, *args, **kwargs):
            self.transformed_index_matrix = None
            self.Tmat = np.eye(3)
            self.reciprocal_Tmat = np.eye(3)

        def process(self):
            events.append("processor.process")

        def plot_clusters_loc(self, save, save_path):
            events.append(("plot.loc", save, str(save_path)))

        def plot_clusters_phase(self, save, save_path):
            events.append(("plot.phase", save, str(save_path)))

    class FakeHrHandler:
        def __init__(self, *args, **kwargs):
            events.append(("hr.init", kwargs.get("file_name"), kwargs.get("npz_file_name")))

        def get_hr_sparse(self):
            return {}

    class FakeKPathGenerator:
        def __init__(self, *args, **kwargs):
            self.kpoints = [np.zeros(4)]

        def read_and_generate_kpath(self, kpath_in, kpath_out):
            events.append(("kpath", str(kpath_in), str(kpath_out)))

    class FakeRunner:
        def __init__(self, **kwargs):
            events.append(("runner.init", sorted(kwargs)))

        def run(self):
            events.append("runner.run")
            return {
                "summary": {
                    "operations": {"Gamma": [{"operation": "C3z", "supported": True}]},
                    "minimal_generators": {"Gamma": ["C3z"]},
                }
            }

    monkeypatch.setattr(main_mod, "OpenMXFile", FakeOpenMXFile)
    monkeypatch.setattr(main_mod, "StructureProcessorSpglib", FakeProcessor)
    monkeypatch.setattr(main_mod, "HrSparseHandler", FakeHrHandler)
    monkeypatch.setattr(main_mod, "KPathGenerator", FakeKPathGenerator)
    monkeypatch.setattr(main_mod, "SymmetryAnalysisRunner", FakeRunner, raising=False)
    return args


def test_symmetry_mode_dispatches_to_symmetry_runner_without_band_calculation(monkeypatch, tmp_path):
    events = []
    config = _make_config(tmp_path, mode="symmetry", symmetry_enable=False)
    args = _patch_main_dependencies(monkeypatch, config, events)

    class FailingCalculator:
        def __init__(self, *args, **kwargs):
            raise AssertionError("band calculator should not be constructed in symmetry mode")

    monkeypatch.setattr(main_mod, "BandStructureCalculator", FailingCalculator)

    main_mod.run_calc(args)

    assert "runner.run" in events
    assert not any(isinstance(event, tuple) and event[0] == "hr.init" and event[1] == str(Path(config.paths.S_file)) for event in events)
    assert not any(event == "kpath" or (isinstance(event, tuple) and event[0] == "kpath") for event in events)


def test_normal_band_mode_still_uses_existing_band_calculator(monkeypatch, tmp_path):
    events = []
    config = _make_config(tmp_path, mode="band", symmetry_enable=False)
    args = _patch_main_dependencies(monkeypatch, config, events)

    class FakeCalculator:
        def __init__(self, *args, **kwargs):
            self.valley_flag = "Gamma"
            self.uses_hamiltonian_symmetrization = False
            events.append("band.init")

        def run_calculation(self, path):
            events.append(("band.run", str(path)))

    monkeypatch.setattr(main_mod, "BandStructureCalculator", FakeCalculator)

    main_mod.run_calc(args)

    assert "band.init" in events
    assert any(isinstance(event, tuple) and event[0] == "band.run" for event in events)
    assert "runner.run" not in events


def test_band_mode_can_run_symmetry_analysis_after_normal_calculation(monkeypatch, tmp_path):
    events = []
    config = _make_config(tmp_path, mode="band", symmetry_enable=True)
    args = _patch_main_dependencies(monkeypatch, config, events)

    class FakeCalculator:
        def __init__(self, *args, **kwargs):
            self.valley_flag = "Gamma"
            self.uses_hamiltonian_symmetrization = False
            events.append("band.init")

        def run_calculation(self, path):
            events.append(("band.run", str(path)))

    monkeypatch.setattr(main_mod, "BandStructureCalculator", FakeCalculator)

    main_mod.run_calc(args)

    assert any(isinstance(event, tuple) and event[0] == "band.run" for event in events)
    assert "runner.run" in events


def test_band_mode_runs_symmetry_analysis_when_hamiltonian_symmetrization_is_requested(monkeypatch, tmp_path):
    events = []
    config = _make_config(tmp_path, mode="band", symmetry_enable=False)
    config.compute.symmetrize_hamiltonian = ["C3z"]
    args = _patch_main_dependencies(monkeypatch, config, events)

    class FakeCalculator:
        def __init__(self, *args, **kwargs):
            self.valley_flag = "Gamma"
            self.uses_hamiltonian_symmetrization = True
            events.append("band.init")

        def run_calculation(self, path):
            events.append(("band.run", str(path)))

    monkeypatch.setattr(main_mod, "BandStructureCalculator", FakeCalculator)

    main_mod.run_calc(args)

    assert "runner.run" in events
    assert any(isinstance(event, tuple) and event[0] == "band.run" for event in events)
    assert any(
        isinstance(event, tuple)
        and event[0] == "band.run"
        and event[1].endswith("Q_shell_6_symm")
        for event in events
    )

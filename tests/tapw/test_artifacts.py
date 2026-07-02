from types import SimpleNamespace

import json
import numpy as np

from tapw.artifacts import (
    array_output_filename,
    band_output_filename,
    canonical_band_filename,
    canonical_profile_name,
    canonical_qshell_name,
    canonical_topology_band_label,
    canonical_topology_grid_name,
    chern_flux_filename,
    chern_summary_filename,
    gvec_output_filename,
    raw_memmap_filename,
    run_dir_name,
)
from tapw.workflows.band import BandStructureCalculator


def test_run_dir_name_uses_direct_for_notapw():
    assert run_dir_name(SimpleNamespace(TAPW=False, n_g=8), symmetrized=False) == "direct"
    assert run_dir_name(SimpleNamespace(TAPW=False, n_g=8), symmetrized=True) == "direct"


def test_run_dir_name_uses_qshell_for_tapw():
    assert run_dir_name(SimpleNamespace(TAPW=True, n_g=8), symmetrized=False) == "Q_shell_8"


def test_run_dir_name_marks_symmetrized_tapw():
    assert run_dir_name(SimpleNamespace(TAPW=True, n_g=8), symmetrized=True) == "Q_shell_8_symm"


def test_band_output_filename_uses_tapw_valley_and_notapw_direct_names():
    assert band_output_filename("vbm", valley_flag="M1", suffix="", tapw=True) == "band_VBM_M1_valley.txt"
    assert band_output_filename("CBM", valley_flag="Gamma", suffix="_2d", tapw=True) == "band_CBM_Gamma_valley_2d.txt"
    assert band_output_filename("VBM", valley_flag="M1", suffix="", tapw=False) == "band_VBM.txt"
    assert band_output_filename("CBM", valley_flag="Gamma", suffix="_2d", tapw=False) == "band_CBM_2d.txt"


def test_array_output_filename_uses_tapw_valley_and_notapw_direct_names():
    assert array_output_filename("vec_vbm", valley_flag="M1", suffix="", tapw=True) == "vec_VBM_M1_valley.npy"
    assert array_output_filename("hamk", valley_flag="Gamma", suffix="_2d", tapw=True) == "hamk_Gamma_valley_2d.npy"
    assert array_output_filename("vec_CBM", valley_flag="M1", suffix="", tapw=False) == "vec_CBM.npy"
    assert array_output_filename("hamk", valley_flag="Gamma", suffix="_2d", tapw=False) == "hamk_2d.npy"


def test_raw_memmap_filename_uses_canonical_raw_prefix():
    assert raw_memmap_filename("eig", valley_flag="M1", suffix="") == "eig_raw_M1_valley.npy"
    assert raw_memmap_filename("vec", valley_flag="Gamma", suffix="_2d") == "vec_raw_Gamma_valley_2d.npy"


def test_gvec_output_filename_includes_ng_valley_and_layer():
    assert gvec_output_filename(n_g=8, valley_flag="M1", layer=1) == "g_vec_list_8_M1_1layer.npy"
    assert gvec_output_filename(n_g=12, valley_flag="Gamma", layer=2) == "g_vec_list_12_Gamma_2layer.npy"


def test_chern_artifact_filenames_include_valley_and_grid_suffix():
    assert (
        chern_flux_filename(
            band_type="CBM",
            valley_flag="M1",
            suffix="_2d_30x50",
            band_label="band0",
        )
        == "berry_flux_CBM_M1_valley_2d_30x50_band0.npy"
    )
    assert (
        chern_summary_filename(
            band_type="CBM",
            valley_flag="M1",
            suffix="_2d_30x50",
        )
        == "chern_summary_CBM_M1_valley_2d_30x50.json"
    )


def test_canonical_profile_names_default_spinful_and_mark_nondefault_spin():
    assert canonical_profile_name("K1", spin="all") == "K1"
    assert canonical_profile_name("K1", spin="spinful") == "K1"
    assert canonical_profile_name("K1", spin=True) == "K1"
    assert canonical_profile_name("K1", spin="up") == "K1_up"
    assert canonical_profile_name("K1", spin="spin_up_projected") == "K1_up"
    assert canonical_profile_name("K1", spin="spinless_effective") == "K1_spinless"
    assert canonical_profile_name("K1", profile="K1_custom") == "K1_custom"


def test_canonical_qshell_and_file_names_are_user_facing():
    assert canonical_qshell_name(6) == "q06"
    assert canonical_qshell_name("q7") == "q07"
    assert canonical_band_filename("energies", "vbm") == "energies_vbm.txt"
    assert canonical_band_filename("wavefunctions", "CBM") == "wavefunctions_cbm.npy"
    assert canonical_band_filename("hamiltonian", None) == "hamiltonian_k.npy"
    assert canonical_band_filename("g_vectors", 2) == "g_vectors_group2.npy"


def test_canonical_topology_band_labels_avoid_dash_collisions():
    assert canonical_topology_band_label([-1]) == "band_m1"
    assert canonical_topology_band_label([-1, -2]) == "bands_m1_m2"
    assert canonical_topology_band_label([0, 2]) == "bands_0_2"


def test_canonical_topology_grid_name_includes_nondefault_ranges():
    assert canonical_topology_grid_name(31, 31) == "grid31x31"
    assert (
        canonical_topology_grid_name(21, 41, range_b1=(0.0, 0.5), range_b2=(-0.5, 0.5))
        == "grid21x41_b1_0p0_0p5_b2_m0p5_0p5"
    )
    assert (
        canonical_topology_grid_name(21, 41, range_b1=(-0.5, 0.0), range_b2=(-0.5, 0.5))
        == "grid21x41_b1_m0p5_0p0_b2_m0p5_0p5"
    )


def test_band_calculation_writes_canonical_manifest_and_filenames(tmp_path):
    target_dir = tmp_path / "outputs" / "K1" / "q06"
    calc = BandStructureCalculator.__new__(BandStructureCalculator)
    calc.config = SimpleNamespace(
        TAPW=True,
        n_g=6,
        mode="band",
        band_type="BOTH",
        eig_vec_cal=True,
        hamk_save=True,
        vec_store="memory",
        efermi=0.5,
        kpoint_chunk_id=0,
        kpoint_chunk_count=1,
        eigensolver="scipy",
        output_layout=SimpleNamespace(style="canonical_v1"),
    )
    calc.valley_flag = "K1"
    calc.kpath_config = SimpleNamespace(kpoints=np.zeros((2, 3), dtype=float))
    calc.reporter = None
    calc.TAPW_parameters = SimpleNamespace(
        g_vec_list_K1=np.array([[0.0, 0.0]], dtype=float),
        g_vec_list_K2=np.array([[1.0, 0.0]], dtype=float),
    )
    calc.result = {}

    def fake_parallel(_kpoints, **_kwargs):
        calc.result["eig"] = np.array([[-1.0, -0.5, 1.0], [-0.8, -0.4, 1.2]], dtype=float)
        calc.result["vec"] = np.ones((2, 2, 3), dtype=np.complex128)
        calc.result["hamk"] = np.zeros((2, 2, 2), dtype=np.complex128)

    calc.parallel_calculate_band_01 = fake_parallel

    calc.calculate_band_structure(str(target_dir))

    band_dir = target_dir / "band"
    assert (band_dir / "energies_vbm.txt").is_file()
    assert (band_dir / "energies_cbm.txt").is_file()
    assert (band_dir / "wavefunctions_vbm.npy").is_file()
    assert (band_dir / "wavefunctions_cbm.npy").is_file()
    assert (band_dir / "hamiltonian_k.npy").is_file()
    assert (band_dir / "g_vectors_group1.npy").is_file()
    assert (band_dir / "g_vectors_group2.npy").is_file()
    assert (band_dir / "kpoints.npy").is_file()

    manifest = json.loads((band_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "tapw_band_outputs/v1"
    assert manifest["files"]["hamiltonian_k"] == "hamiltonian_k.npy"
    assert manifest["files"]["g_vectors_group1"] == "g_vectors_group1.npy"
    assert manifest["files"]["energies_vbm"] == "energies_vbm.txt"


def test_chern_calculation_writes_canonical_topology_files(tmp_path):
    target_dir = tmp_path / "outputs" / "K1" / "q06"
    calc = BandStructureCalculator.__new__(BandStructureCalculator)

    class _Config(SimpleNamespace):
        def get_chern_grid_shape(self):
            return 3, 3

        def get_chern_grid_suffix(self):
            return "_2d_3"

    calc.config = _Config(
        TAPW=True,
        mode="chern",
        band_type="VBM",
        eig_vec_cal=True,
        ge=False,
        kpoint_chunk_count=1,
        chern_band_indices=[-1, -2],
        output_layout=SimpleNamespace(style="canonical_v1"),
    )
    calc.valley_flag = "K1"
    calc.reporter = None
    calc.generate_kmesh = lambda *_args: np.zeros((9, 3), dtype=float)

    vec_grid = np.zeros((3, 3, 2, 2), dtype=np.complex128)
    vec_grid[:, :, 0, 0] = 1.0
    vec_grid[:, :, 1, 1] = 1.0

    def fake_calculate_band_structure(_path, _kpoints):
        calc.result = {"vec": vec_grid.reshape(9, 2, 2)}

    calc.calculate_band_structure = fake_calculate_band_structure

    calc.calculate_chern(str(target_dir))

    topology_dir = target_dir / "topology" / "grid3x3"
    assert (topology_dir / "berry_flux_band_m1.npy").is_file()
    assert (topology_dir / "berry_flux_band_m2.npy").is_file()
    assert (topology_dir / "berry_flux_bands_m1_m2.npy").is_file()
    assert (topology_dir / "chern_summary.json").is_file()
    manifest = json.loads((topology_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "tapw_topology_grid/v1"
    assert manifest["grid_order"] == "ij"
    assert manifest["files"]["chern_summary"] == "chern_summary.json"
    assert "berry_flux_bands_m1_m2" in manifest["files"]

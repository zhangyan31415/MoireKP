from types import SimpleNamespace

from tapw.artifacts import (
    array_output_filename,
    band_output_filename,
    gvec_output_filename,
    raw_memmap_filename,
    run_dir_name,
)


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

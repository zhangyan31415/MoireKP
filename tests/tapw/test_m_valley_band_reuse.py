from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tapw.workflows.band import BandStructureCalculator

from tapw.workflows.band import requested_hamiltonian_symmetry_operations








def test_save_band_static_metadata_writes_valley_specific_g_vectors(tmp_path: Path):
    calculator = BandStructureCalculator.__new__(BandStructureCalculator)
    calculator.config = SimpleNamespace(TAPW=True, n_g=4)
    calculator.valley_flag = "M2"
    calculator.uses_hamiltonian_symmetrization = False
    calculator.TAPW_parameters = SimpleNamespace(
        g_vec_list_K1=np.array([[1.0, 2.0]], dtype=float),
        g_vec_list_K2=np.array([[3.0, 4.0]], dtype=float),
    )

    calculator.save_band_static_metadata(str(tmp_path))

    g1 = tmp_path / "band" / "g_vectors_group1.npy"
    g2 = tmp_path / "band" / "g_vectors_group2.npy"
    assert g1.exists()
    assert g2.exists()
    assert np.allclose(np.load(g1), calculator.TAPW_parameters.g_vec_list_K1)
    assert np.allclose(np.load(g2), calculator.TAPW_parameters.g_vec_list_K2)


def test_switch_m_valley_reuses_cached_parameters_and_resets_run_state():
    calculator = BandStructureCalculator.__new__(BandStructureCalculator)
    calculator.use_M_valley_threefold_symm = True
    calculator.config = SimpleNamespace(valley=31)
    calculator.result = {"eig": np.array([[1.0]])}
    calculator._progress_dir = "/tmp/progress"
    calculator.valley_flag = "M1"
    calculator.structure = SimpleNamespace(
        spin=False,
        df=pd.DataFrame({"orb_num": [1, 2]}),
    )
    calculator._sorted_wann = np.zeros((3, 3), dtype=float)
    calculator._num_wann = 3
    calculator._ef_onsite_orb = np.array([9.0, 9.0, 9.0], dtype=float)

    params_31 = SimpleNamespace(electric_field_onsite=np.array([1.0, 2.0], dtype=float))
    params_32 = SimpleNamespace(electric_field_onsite=np.array([3.0, 4.0], dtype=float))
    calculator._m_valley_parameters = {31: params_31, 32: params_32}
    calculator.TAPW_parameters = params_31

    BandStructureCalculator.switch_m_valley(calculator, 32)

    assert calculator.config.valley == 32
    assert calculator.valley_flag == "M2"
    assert calculator.TAPW_parameters is params_32
    assert calculator.result == {}
    assert calculator._progress_dir is None
    assert np.array_equal(calculator._ef_onsite_orb, np.array([3.0, 4.0, 4.0], dtype=float))


def test_switch_m_valley_requires_prebuilt_threefold_state():
    calculator = BandStructureCalculator.__new__(BandStructureCalculator)
    calculator.use_M_valley_threefold_symm = False

    with pytest.raises(ValueError):
        BandStructureCalculator.switch_m_valley(calculator, 32)


def test_requested_symmetry_operations_use_current_valley_not_stale_flag():
    cfg = SimpleNamespace(
        TAPW=True,
        valley=32,
        valley_flag="M1",
        symmetrize_hamiltonian=True,
        resolved_hamiltonian_symmetry_operations_by_valley={
            "M1": ["C3z"],
            "M2": ["C3z", "C2"],
        },
    )

    assert requested_hamiltonian_symmetry_operations(cfg) == ["C3z", "C2"]

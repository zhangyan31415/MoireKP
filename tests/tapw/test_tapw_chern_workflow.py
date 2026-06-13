from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tapw.artifacts import (
    array_output_filename,
    berry_flux_output_filename,
    chern_summary_filename,
    chern_summary_output_filename,
)
from tapw.config import format_chern_grid_suffix
from tapw.workflows.band import BandStructureCalculator
from tapw import chern_post


def _config(*, eig_vec_cal=True, chunk_count=1, band_type="CBM", grid=(3, 2)):
    num_k1, num_k2 = grid
    return SimpleNamespace(
        eig_vec_cal=eig_vec_cal,
        kpoint_chunk_count=chunk_count,
        band_type=band_type,
        TAPW=True,
        ge=False,
        get_chern_grid_shape=lambda: (num_k1, num_k2),
        get_chern_grid_suffix=lambda: format_chern_grid_suffix(num_k1, num_k2),
    )


def _constant_wavefunctions(num_kpoints, *, dim_h=2, num_bands=1):
    wavefunctions = np.zeros((num_kpoints, dim_h, num_bands), dtype=np.complex128)
    wavefunctions[:, 0, :] = 1.0
    return wavefunctions


def _calculator(config):
    calculator = BandStructureCalculator.__new__(BandStructureCalculator)
    calculator.config = config
    calculator.valley_flag = "K1"
    return calculator


def test_calculate_chern_writes_berry_flux_and_chern_summary(tmp_path):
    config = _config(grid=(3, 2))
    calculator = _calculator(config)
    seen = {}
    wavefunctions = _constant_wavefunctions(6)

    def fake_calculate_band_structure(path, kpoints):
        seen["kpoints"] = np.asarray(kpoints)
        calculator.result = {
            "eig": np.zeros((len(kpoints), 1), dtype=float),
            "vec": wavefunctions,
        }

    calculator.calculate_band_structure = fake_calculate_band_structure

    result = calculator.calculate_chern(str(tmp_path))

    berry_flux_path = tmp_path / "topo" / "berry_flux_CBM_K1_valley_2d_3x2_band0.npy"
    summary_path = tmp_path / "topo" / "chern_summary_CBM_K1_valley_2d_3x2.json"
    assert result is not NotImplemented
    assert result["grid_order"] == "ij"
    assert result["berry_flux_file"] == berry_flux_path.name
    assert result["chern_number"] == pytest.approx(0.0)
    assert berry_flux_path.exists()
    assert summary_path.exists()
    assert np.load(berry_flux_path).shape == (2, 1)
    assert result["summary_file"] == summary_path.name

    expected_kpoints = np.array(
        [
            [-0.5, -0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )
    np.testing.assert_allclose(seen["kpoints"], expected_kpoints)


def test_calculate_chern_requires_wavefunctions_enabled(tmp_path):
    calculator = _calculator(_config(eig_vec_cal=False))

    def fail_if_called(path, kpoints):
        raise AssertionError("calculate_band_structure should not run without eig_vec_cal")

    calculator.calculate_band_structure = fail_if_called

    with pytest.raises(ValueError, match="eig_vec_cal=true"):
        calculator.calculate_chern(str(tmp_path))


def test_calculate_chern_rejects_unfinalized_kpoint_chunks(tmp_path):
    calculator = _calculator(_config(chunk_count=2))

    def fail_if_called(path, kpoints):
        raise AssertionError("chunked Chern should fail before launching k-point work")

    calculator.calculate_band_structure = fail_if_called

    with pytest.raises(ValueError, match="unfinished k-point chunks"):
        calculator.calculate_chern(str(tmp_path))


def test_calculate_chern_rejects_generalized_eigenvectors(tmp_path):
    config = _config()
    config.ge = True
    calculator = _calculator(config)

    def fail_if_called(path, kpoints):
        raise AssertionError("generalized Chern should fail before launching k-point work")

    calculator.calculate_band_structure = fail_if_called

    with pytest.raises(ValueError, match="ge=true"):
        calculator.calculate_chern(str(tmp_path))


def test_chern_summary_filename_includes_valley_band_type_and_grid():
    assert (
        chern_summary_filename(band_type="CBM", valley_flag="K1", suffix="_2d_3x2")
        == "chern_summary_CBM_K1_valley_2d_3x2.json"
    )


def test_calculate_chern_fails_when_wavefunction_file_is_missing(tmp_path):
    calculator = _calculator(_config())

    def fake_calculate_band_structure(path, kpoints):
        (Path(path) / "topo").mkdir(parents=True)

    calculator.calculate_band_structure = fake_calculate_band_structure

    with pytest.raises(RuntimeError, match="wavefunctions"):
        calculator.calculate_chern(str(tmp_path))


def test_chern_flux_output_rejects_invalid_band_index(tmp_path):
    num_k1, num_k2 = 3, 2
    suffix = format_chern_grid_suffix(num_k1, num_k2)
    np.save(
        tmp_path / array_output_filename("vec_CBM", valley_flag="K1", suffix=suffix, tapw=True),
        _constant_wavefunctions(num_k1 * num_k2),
    )

    with pytest.raises(IndexError, match="1"):
        chern_post.write_chern_flux_outputs(
            str(tmp_path),
            "CBM",
            "K1",
            num_k1,
            num_k2,
            [1],
        )

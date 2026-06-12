import subprocess
import sys
from pathlib import Path

import numpy as np


def _expected_legacy_reorder(array: np.ndarray, grid_size: int) -> np.ndarray:
    reshaped = array.reshape(grid_size, grid_size, *array.shape[1:])
    axes = [1, 0] + list(range(2, reshaped.ndim))
    return reshaped.transpose(axes).reshape(array.shape)


def test_script_auto_detects_current_directory_and_reorders_in_place(tmp_path):
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "reorder_legacy_chern_vec.py"
    )

    source = tmp_path / "vec_CBM_Gamma_valley_2d_31.npy"
    original = np.arange(4 * 2 * 3, dtype=np.float64).reshape(4, 2, 3)
    np.save(source, original)
    band_source = tmp_path / "band_CBM_Gamma_valley_2d_31.txt"
    original_band = np.arange(4 * 5, dtype=np.float64).reshape(4, 5)
    np.savetxt(band_source, original_band)

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr

    backup_candidates = sorted(tmp_path.glob("vec_CBM_Gamma_valley_2d_31.bak_*.npy"))
    assert len(backup_candidates) == 1
    assert np.array_equal(np.load(backup_candidates[0]), original)

    band_backup_candidates = sorted(tmp_path.glob("band_CBM_Gamma_valley_2d_31.bak_*.txt"))
    assert len(band_backup_candidates) == 1
    assert np.array_equal(np.loadtxt(band_backup_candidates[0]), original_band)

    rewritten = np.load(source)
    expected = _expected_legacy_reorder(original, grid_size=2)
    assert np.array_equal(rewritten, expected)

    rewritten_band = np.loadtxt(band_source)
    expected_band = _expected_legacy_reorder(original_band, grid_size=2)
    assert np.array_equal(rewritten_band, expected_band)


def test_script_skips_non_square_wavefunction_files_in_auto_mode(tmp_path):
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "reorder_legacy_chern_vec.py"
    )

    square = tmp_path / "vec_CBM_Gamma_valley_2d_31.npy"
    square_array = np.arange(4 * 2 * 3, dtype=np.float64).reshape(4, 2, 3)
    np.save(square, square_array)
    np.savetxt(tmp_path / "band_CBM_Gamma_valley_2d_31.txt", np.arange(20, dtype=np.float64).reshape(4, 5))

    nonsquare = tmp_path / "vec_CBM_Gamma_valley_2d_121x31.npy"
    nonsquare_array = np.arange(6 * 2 * 3, dtype=np.float64).reshape(6, 2, 3)
    np.save(nonsquare, nonsquare_array)

    completed = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Skip:" in completed.stdout
    assert np.array_equal(np.load(nonsquare), nonsquare_array)

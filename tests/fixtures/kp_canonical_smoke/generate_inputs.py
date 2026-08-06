from __future__ import annotations

from pathlib import Path

import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parent
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)

    hamk = np.zeros((3, 4, 4), dtype=np.complex128)
    diagonals = [
        [-0.30, -0.10, 0.20, 0.40],
        [-0.24, -0.08, 0.24, 0.43],
        [-0.18, -0.05, 0.29, 0.47],
    ]
    for idx, diag in enumerate(diagonals):
        hamk[idx] = np.diag(diag)

    np.save(data / "hamiltonian_k.npy", hamk)
    np.save(data / "g_vectors_group1.npy", np.array([[0.0, 0.0]], dtype=float))
    np.save(data / "g_vectors_group2.npy", np.array([[0.1, 0.0]], dtype=float))
    np.save(
        data / "kpoints.npy",
        np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=float),
    )
    np.savetxt(data / "band_energies.txt", np.linalg.eigvalsh(hamk), fmt="%.16e")


if __name__ == "__main__":
    main()

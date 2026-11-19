"""Restore the KP input from the archived reduced trilayer Hamiltonian."""
from pathlib import Path
import shutil

import numpy as np


def main():
    directory = Path(__file__).resolve().parent
    output = directory.parent / "output/kp_input"
    representations = output / "source_symmetry/representations"
    representations.mkdir(parents=True, exist_ok=True)
    with np.load(directory / "projected_input.npz", allow_pickle=False) as data:
        heff = data["heff"]
        rows, dimension, _ = heff.shape
        hamiltonian = np.zeros((rows, 2 * dimension, 2 * dimension), dtype=heff.dtype)
        hamiltonian[:, :dimension, :dimension] = heff
        np.save(output / "hamiltonian_k.npy", hamiltonian)
        for name in ("kpoints", "g_vectors_group1", "g_vectors_group2"):
            np.save(output / f"{name}.npy", data[name])
    shutil.copyfile(directory / "energies_vbm.txt", output / "energies_vbm.txt")
    shutil.copyfile(directory / "symmetry_manifest.json", representations / "manifest.json")
    with np.load(directory / "source_symmetry.npz", allow_pickle=False) as data:
        np.save(representations / "C3z_raw_H.npy", data["C3z_raw_H"])


if __name__ == "__main__":
    main()

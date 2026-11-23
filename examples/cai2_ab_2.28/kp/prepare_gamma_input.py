"""Rebuild the six-state Gamma input from orthonormal shell-5 TAPW matrices."""
import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, solve


def molecular_columns(frame, nq):
    """The TAPW order is spin, layer, Q, atomic orbital."""
    block, rank = frame.shape
    indices = np.arange(block * nq).reshape(4, nq, -1).transpose(1, 0, 2)
    columns = np.zeros((block * nq, rank * nq), dtype=complex)
    for i, rows in enumerate(indices.reshape(nq, block)):
        columns[np.ix_(rows, np.arange(rank*i, rank*(i+1)))] = frame
    return columns


def effective_matrix(h, trial, energy):
    shifted = np.array(-h, order="F")
    shifted.flat[::len(h) + 1] += energy
    solutions = solve(shifted, trial, assume_a="her", check_finite=False)
    green = trial.conj().T @ solutions
    green = (green + green.conj().T) / 2
    inverse = solve(green, np.eye(len(green)), assume_a="her")
    unnormalized = solutions @ inverse
    values, vectors = eigh(unnormalized.conj().T @ unnormalized)
    if values.min() <= 0:
        raise ValueError("The molecular embedding has a singular norm matrix.")
    normalization = (vectors / np.sqrt(values)) @ vectors.conj().T
    result = energy * np.eye(len(green)) - normalization @ inverse @ normalization
    return (result + result.conj().T) / 2


def main():
    case = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("band_dir", type=Path, help="Gamma q05 TAPW band output")
    parser.add_argument("output", type=Path, help="New model-input directory")
    args = parser.parse_args()
    source = args.band_dir
    supplied = case / "input/Gamma"
    q = np.load(source / "g_vectors_group1.npy")
    k = np.load(source / "kpoints.npy")
    np.testing.assert_allclose(q, np.load(source / "g_vectors_group2.npy"), atol=1e-9)
    np.testing.assert_allclose(q, np.load(supplied / "g_vectors_group1.npy"), atol=1e-9)
    np.testing.assert_allclose(k, np.load(supplied / "kpoints.npy"), atol=1e-9)
    with np.load(case / "kp/molecular_frame_Gamma.npz") as data:
        trial = molecular_columns(data["F0"], len(q))
    matrices = np.load(source / "hamiltonian_k.npy", mmap_mode="r")
    energy = float(np.loadtxt(source / "energies_vbm.txt")[0, -6:].mean())
    dim = trial.shape[1]
    if matrices.shape != (len(k), trial.shape[0], trial.shape[0]):
        raise ValueError("Unexpected TAPW Hamiltonian dimensions.")
    # The configured two-group interface orders spin first. Only the first
    # group is retained; the uncoupled auxiliary group is outside the fit.
    active = [spin*dim + 3*i + a for i in range(len(q)) for a in range(3) for spin in range(2)]
    auxiliary = np.setdiff1d(np.arange(2*dim), active)
    args.output.mkdir(parents=True, exist_ok=False)
    output = np.lib.format.open_memmap(args.output / "hamiltonian_k.npy", mode="w+",
                                       dtype=complex, shape=(len(k), 2*dim, 2*dim))
    bands = []
    for i,h in enumerate(matrices):
        effective = effective_matrix(h, trial, energy)
        output[i][np.ix_(active,active)] = effective
        output[i,auxiliary,auxiliary] = 10.0
        bands.append(np.linalg.eigvalsh(effective))
    output.flush()
    np.savetxt(args.output / "energies_vbm.txt", bands)


if __name__ == "__main__":
    main()

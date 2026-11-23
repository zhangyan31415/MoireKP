"""Recompute the Gamma energy and wavefunction targets from orthonormal TAPW output."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, solve


def molecular_columns(frame, nq):
    block, rank = frame.shape
    rows = np.arange(block*nq).reshape(4, nq, -1).transpose(1, 0, 2)
    p = np.zeros((block*nq, rank*nq), complex)
    for i, rr in enumerate(rows.reshape(nq,block)):
        p[np.ix_(rr, np.arange(i*rank, (i+1)*rank))] = frame
    return p


def wave_target(h, energies, columns, ref):
    # Orthonormalized reconstruction at E_ref, in the archived molecular frame.
    shifted = np.array(-h, order="F")
    shifted.flat[::len(h)+1] += ref
    x = solve(shifted, columns, assume_a="her", overwrite_a=True, check_finite=False)
    green = columns.conj().T @ x
    j = solve((green+green.conj().T)/2, x.conj().T, assume_a="her").conj().T
    e, u = eigh(j.conj().T @ j)
    if e[0] <= 0:
        raise ValueError("The molecular embedding has a singular norm matrix.")
    lift = j @ ((u/np.sqrt(e)) @ u.conj().T)
    vals, psi = eigh(h, subset_by_value=(energies[-1]-1e-7, energies[0]+1e-7),
                     check_finite=False)
    np.testing.assert_allclose(vals[::-1], energies, atol=1e-7, rtol=0)
    return psi[:,::-1].conj().T @ lift


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("band_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    model = folder.parent/"model/Gamma"
    rows = json.loads((model/"MODEL.json").read_text())["train_wave_indices"]
    band = args.band_dir
    q = np.load(band/"g_vectors_group1.npy")
    np.testing.assert_allclose(q, np.load(band/"g_vectors_group2.npy"), atol=1e-8, rtol=0)
    with np.load(model/"GEOMETRIES.npz") as z:
        np.testing.assert_allclose(q,z["q"],atol=1e-8,rtol=0)
        k = np.load(band/"kpoints.npy")[:,:2] @ z["B"]
    with np.load(folder/"input_Gamma.npz") as z:
        np.testing.assert_allclose(k,z["k_cartesian"],atol=1e-8,rtol=0)
        seed = z["initial_controls"]
    with np.load(model/"FIXED_ORBITALS.npz") as z:
        p = molecular_columns(z["reference_vectors"],len(q))
    he = np.load(band/"hamiltonian_k.npy",mmap_mode="r")
    native = np.loadtxt(band/"energies_vbm.txt")[:,-8:][:,::-1]
    assert he.shape == (61,len(p),len(p)) and native.shape == (61,8)
    ref = float(native[0,:6].mean())
    cross = [wave_target(he[i],native[i,:6],p,ref) for i in rows]
    with args.output.open("xb") as f:
        np.savez_compressed(f,k_cartesian=k,target_valence_eV=native-native[0,0],
                            wave_rows=rows,tapw_cross=cross,initial_controls=seed)


if __name__=="__main__":
    main()

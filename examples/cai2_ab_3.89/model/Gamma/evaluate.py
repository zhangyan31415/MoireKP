"""NumPy-only fixed-angle CaI2 AB Γ electron Hamiltonian in eV."""
from pathlib import Path
import runpy

import numpy as np


class Model:
    """Momentums are Cartesian inverse angstroms, energies relative to the Γ VBM."""
    def __init__(self, folder):
        folder = Path(folder)
        self.local = runpy.run_path(str(folder/"model.py"))["CaI2Gamma"](folder)
        self.theta = self.local.info["angle_deg"]
        with np.load(folder/"GEOMETRIES.npz",allow_pickle=False) as g:
            self.B, self.q = g["B"],g["q"]
        self.dim = self.local.dimension*len(self.q)
        self.zero = np.linalg.eigvalsh(self.raw_hamiltonian(np.zeros(2)))[-1]

    def raw_hamiltonian(self, k):
        return self.local.hamiltonian(k,self.theta,self.B,self.q,carrier="electron")

    def hamiltonian(self, k):
        return self.raw_hamiltonian(k)-self.zero*np.eye(self.dim)

    def bands(self, kpoints):
        return np.array([np.linalg.eigvalsh(self.raw_hamiltonian(k))-self.zero for k in kpoints])


def load_model(folder=None):
    return Model(Path(__file__).resolve().parent if folder is None else folder)

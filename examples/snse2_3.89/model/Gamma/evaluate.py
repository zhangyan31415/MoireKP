"""NumPy evaluator for the fixed-Q four-state polynomial continuum model.

Input k is physical Cartesian momentum in inverse angstrom; output is eV.
"""
from pathlib import Path
import numpy as np
class Model:
    def __init__(self,folder):
        with np.load(Path(folder)/'polynomial.npz') as z:
            self.monomials=z['monomials'];self.matrices=z['matrices'];self.origin=z['origin'];self.scale=float(z['scale'])
        self.dim=self.matrices.shape[-1]
    def hamiltonian(self,k):
        k=np.asarray(k,float)
        if k.shape!=(2,):raise ValueError('Expected physical Cartesian [kx,ky] in inverse angstrom')
        p=(k-self.origin)/self.scale;w=p[0]+1j*p[1];powers=w**self.monomials[:,0]*w.conjugate()**self.monomials[:,1]
        h=np.einsum('m,mij->ij',powers,self.matrices)
        return (h+h.conj().T)/2
    def bands(self,kpoints):return np.array([np.linalg.eigvalsh(self.hamiltonian(k)) for k in kpoints])
def load_model(folder=None):return Model(Path(__file__).resolve().parent if folder is None else folder)

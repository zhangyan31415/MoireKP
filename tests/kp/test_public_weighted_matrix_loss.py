"""Independent dense-loss checks for the public sparse row compiler."""
from types import SimpleNamespace
import numpy as np
from kp.model.response_basis import CompiledResponseBasis,NormalizedLowEnergyWeightingSpec,build_normalized_low_energy_weighting

def test_masked_target_weighted_rows_match_dense_loss_and_subspace_gauge():
    target=np.array([[[.7,.4,.2],[.4,-.6,.5],[.2,.5,.1]]],complex)
    mask=np.array([[1,1,0],[1,1,1],[0,1,1]])
    target=target*mask
    ops=np.array([np.diag([1.,-1.,0.]),[[0.,1.,0.],[1.,0.,0.],[0.,0.,0.]]],complex)
    support=np.flatnonzero(np.any(abs(ops)>0,axis=0));design=ops.reshape(2,-1)[:,support].T
    spec=NormalizedLowEnergyWeightingSpec(n_bands=2,one_sided_weight=30.,two_sided_weight=300.)
    w=build_normalized_low_energy_weighting(target,spec)
    a,b=CompiledResponseBasis._normalized_low_energy_complex_rows(SimpleNamespace(dim=3),design,support,target,w)
    v=w.eigenvectors[0][:,w.selected_mask[0]]
    def dense(c):
        delta=np.einsum('a,aij->ij',c,ops)-target[0]
        return np.linalg.norm(delta)**2/9+30*np.linalg.norm(delta@v)**2/6+300*np.linalg.norm(v.conj().T@delta@v)**2/4
    # The omitted inactive rows contribute a constant only.
    c0=np.array([0.,0.]);base=dense(c0)-np.linalg.norm(a@c0-b)**2
    for c in [np.array([.3,-.2]),np.array([1.1,.7])]:
        np.testing.assert_allclose(dense(c)-np.linalg.norm(a@c-b)**2,base,atol=1e-10)
    rotation=np.array([[1.,-1j],[-1j,1.]])/np.sqrt(2)
    changed=w.eigenvectors.copy();changed[0][:,w.selected_mask[0]]=v@rotation
    rotated=SimpleNamespace(spec=spec,eigenvectors=changed,selected_mask=w.selected_mask)
    ar,br=CompiledResponseBasis._normalized_low_energy_complex_rows(SimpleNamespace(dim=3),design,support,target,rotated)
    np.testing.assert_allclose(ar.conj().T@ar,a.conj().T@a,atol=1e-10)
    np.testing.assert_allclose(ar.conj().T@br,a.conj().T@b,atol=1e-10)
    plain=build_normalized_low_energy_weighting(target,NormalizedLowEnergyWeightingSpec(n_bands=2,one_sided_weight=0.,two_sided_weight=0.))
    ap,bp=CompiledResponseBasis._normalized_low_energy_complex_rows(SimpleNamespace(dim=3),design,support,target,plain)
    np.testing.assert_allclose(ap,design/3)
    np.testing.assert_allclose(bp,target.reshape(-1)[support]/3)

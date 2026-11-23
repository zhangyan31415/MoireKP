"""Repeat the fixed-angle, 20-parameter Gamma refinement of AB CaI2 at 3.89 degrees."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from scipy.linalg import eigh
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

def reciprocal_stars(b,nstars):
    stars=[b[0],b[1]-b[0],-b[1],b[0]+b[1],b[1]-2*b[0],b[0]-2*b[1],2*b[0],2*(b[1]-b[0]),-2*b[1]]
    return np.array(stars[:3*nstars])


_PACKED={}


def packed(ops):
    key=id(ops)
    if len(ops)>1 and key in _PACKED and _PACKED[key][0] is ops:return _PACKED[key][1:]
    c=np.array([a['polynomial'] for a in ops]);v=np.array([a['fourier'] for a in ops])
    if len(ops)>1:
        if len(_PACKED)>8:_PACKED.clear()
        _PACKED[key]=(ops,c,v)
    return c,v


@lru_cache(maxsize=64)
def _edges(qbytes,bbytes,ng):
    q=np.frombuffer(qbytes,dtype=np.float64).reshape(-1,2);b=np.frombuffer(bbytes,dtype=np.float64).reshape(2,2)
    delta=q[:,None]-q[None]
    return [np.argwhere(np.max(abs(delta-g),axis=-1)<1e-7).T for g in reciprocal_stars(b,ng//3)]


def matrices(ops,q,b,k):
    q=np.asarray(q);p=q+np.asarray(k);n=len(q)
    phi=np.array([np.ones(n),p[:,0],p[:,1],p[:,0]**2,p[:,0]*p[:,1],p[:,1]**2]).T
    cp,vp=packed(ops)
    h=np.zeros((len(ops),6*n,6*n),complex);blocks=np.einsum('qm,pmab->pqab',phi,cp)
    for iq in range(n):h[:,6*iq:6*iq+6,6*iq:6*iq+6]=blocks[:,iq]
    for j,(a,c) in enumerate(_edges(np.asarray(q,dtype=np.float64).tobytes(),np.asarray(b,dtype=np.float64).tobytes(),vp.shape[1])):
        if not len(a):continue
        mid=(p[a]+p[c])/2;x,y=mid.T;basis=np.array([np.ones(len(a)),x,y,x*x,x*y,y*y]).T
        gx,gy=reciprocal_stars(b,vp.shape[1]//3)[j]
        basis[:,3:]-=np.array([gx*gx,gx*gy,gy*gy])/4
        block=np.einsum('cm,pmab->pcab',basis,vp[:,j])
        ra=6*a[:,None]+np.arange(6);rc=6*c[:,None]+np.arange(6)
        h[:,ra[:,:,None],rc[:,None,:]]+=block
        h[:,rc[:,:,None],ra[:,None,:]]+=block.conj().swapaxes(-1,-2)
    return h


def sampled_operators(ops, q, B, k, scale):
    responses = matrices(ops, q, B, k)*scale[:,None,None]
    nz = np.nonzero(np.any(abs(responses)>1e-13,axis=0))
    values = responses[:,nz[0],nz[1]]
    return nz, values, [csr_matrix((v,nz),shape=responses.shape[-2:]) for v in values]


def train_point(data, x, wave, dim):
    nz, values, csr = data
    H = np.zeros((dim,dim),complex)
    H[nz] = x @ values
    e, v = eigh(H, check_finite=False, subset_by_index=None if wave is not None else (0,7))
    u = v[:,:8]
    act = np.stack([op@u for op in csr])
    de = np.einsum("ia,dia->ad",u.conj(),act).real
    if wave is None:
        return e[:8], de, [], []
    proj = wave @ v
    r, j = [], []
    for pair in (0,2,4):
        sl = np.arange(pair,pair+2)
        other = np.delete(np.arange(dim),sl)
        cross = v[:,other].conj().T @ act[:,:,sl]
        gaps = e[sl][None,:] - e[other][:,None]
        derivative = np.einsum("ao,doi,oi->dai",proj[sl][:,other],cross,gaps/(gaps**2+1e-10),optimize=True)
        matching = proj[np.ix_(sl,sl)]
        r.append(32*(1-np.linalg.norm(matching)**2/2))
        j.append(-32*np.einsum("ai,dai->d",matching.conj(),derivative).real)
    outside = v[:,6:].conj().T @ act[:,:,:6]
    gaps = e[None,:6] - e[6:,None]
    derivative = np.einsum("ao,doi,oi->dai",proj[:,6:],outside,gaps/(gaps**2+1e-10),optimize=True)
    block = proj[:,:6]
    loss = np.eye(6)-block@block.conj().T
    jd = derivative@block.conj().T
    jd = -(jd+jd.conj().swapaxes(-1,-2))
    r.extend(8*np.r_[loss.real.ravel(),loss.imag.ravel()])
    j.extend(8*np.r_[jd.real.reshape(len(x),-1).T,jd.imag.reshape(len(x),-1).T])
    return e[:8], de, r, j


class Objective:
    def __init__(self, pool, cache, train, waves, target, principal, dim):
        self.pool = pool
        self.cache = cache
        self.train = train
        self.waves = waves
        self.target = target
        self.principal = principal
        self.dim = dim
        self.last = None
        self.calls = 0

    def __call__(self, x):
        if self.last is not None and np.array_equal(self.last[0], x):
            return self.last[1:]
        work = lambda k: train_point(self.cache[k], x, self.waves.get(k), self.dim)
        outputs = list(self.pool.map(work, self.train))
        origin, jac0 = outputs[0][0][:2].mean(), outputs[0][1][:2].mean(axis=0)
        r, j = [], []
        for k, (e, de, rp, jp) in zip(self.train, outputs):
            delta = (e-origin-self.target[k])*1000
            g = (de-jac0)*1000
            delta[6:] *= .25
            g[6:] *= .25
            r.extend([*delta,*rp])
            j.extend([*g,*jp])
        ev, vec = np.linalg.eigh(np.einsum("d,tdab->tab",x,self.principal))
        angle = np.argmin(ev[:,0])
        margin = ev[angle,0]
        r.append(1000*min(0.,margin-.08))
        u = vec[angle,:,0]
        j.append(1000*np.einsum("i,dij,j->d",u.conj(),self.principal[angle],u).real if margin<.08 else np.zeros_like(x))
        r, j = np.asarray(r), np.asarray(j)
        self.last = x.copy(), r, j
        self.calls += 1
        if self.calls == 1 or self.calls % 10 == 0:
            print(f"Call {self.calls}: loss {0.5*(r@r):.8g}",flush=True)
        return r, j



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path, nargs="?", default=Path(__file__).with_name("kp_Gamma.yaml"))
    parser.add_argument("--workers", type=int, default=1, help="parallel training points")
    parser.add_argument("--output", type=Path, required=True, help="new export directory")
    parser.add_argument("--input", type=Path, help="alternative training NPZ (default: input in configuration)")
    args = parser.parse_args()
    inputs = args.configuration.resolve().parent
    config = yaml.safe_load(args.configuration.read_text())
    fit_settings = config["fit"]
    model = inputs / config["model_directory"]
    train = fit_settings["kpoints"]
    with np.load(args.input or inputs / config["input"]) as z:
        initial, k, target = z["initial_controls"], z["k_cartesian"], -z["target_valence_eV"]
        np.testing.assert_array_equal(z["wave_rows"], fit_settings["wavefunction_kpoints"])
        waves = dict(zip(z["wave_rows"], z["tapw_cross"]))
    with np.load(model / "GEOMETRIES.npz") as z:
        q, B = z["q"], z["B"]
    with np.load(model / "FIXED_OPERATOR_TEMPLATES.npz") as z:
        bank = dict(z)
    with np.load(model / "MATERIAL_CONTROLS.npz") as z:
        names, kinds = z["names"], z["kinds"]
    scale = np.select([kinds == "quadratic", np.isin(kinds, ["linear", "gradient"])],
                      [4., .3], default=.03)
    ops = [{kind: bank[kind][i] for kind in ("polynomial", "fourier")} for i in range(len(scale))]
    cache = {i: sampled_operators(ops,q,B,k[i],scale) for i in train}
    angles = np.linspace(0,np.pi,65)
    directions = np.c_[np.cos(angles)**2, np.sin(angles)*np.cos(angles), np.sin(angles)**2]
    principal = np.einsum("tm,dmab->tdab",directions,bank["polynomial"][:,3:])*scale[None,:,None,None]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        objective = Objective(pool,cache,train,waves,target,principal,6*len(q))
        fit = least_squares(lambda x: objective(x)[0], initial/scale,
                            jac=lambda x: objective(x)[1], x_scale="jac", ftol=fit_settings["ftol"], max_nfev=fit_settings["max_nfev"])
    coeff = fit.x*scale
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/"MATERIAL_CONTROLS.npz",controls=coeff[None],names=names,kinds=kinds)
    np.savez_compressed(args.output/"COEFFICIENTS.npz",
                        polynomial_controls=np.einsum("d,dmab->mab",coeff,bank["polynomial"])[None],
                        fourier_controls=np.einsum("d,dgmab->gmab",coeff,bank["fourier"])[None])
    print(f"{fit.nfev} evaluations; final loss {fit.cost:.10f}; {fit.message}")


if __name__ == "__main__":
    main()

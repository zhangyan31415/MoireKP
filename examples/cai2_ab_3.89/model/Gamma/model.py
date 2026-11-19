"""NumPy-only local Gamma operator. Angle controls are contracted before NQS use."""
from pathlib import Path
import json
import numpy as np
ORDER=np.array([0,1,1,2,2,2])

class LocalGammaModel:
    def __init__(self,folder=None):
        self.folder=Path(folder) if folder else Path(__file__).resolve().parent;self.info=json.loads((self.folder/'MODEL.json').read_text());z=np.load(self.folder/'COEFFICIENTS.npz');self.C=z['polynomial_controls'];self.V=z['fourier_controls'];self.dimension=self.C.shape[-1]

    def weights(self,theta):
        lo,hi=self.info['angle_domain_deg']
        upper=min(hi+.006,self.info.get('nominal_export_max_angle_deg',hi+.006))
        if not np.isfinite(theta) or not lo-.006<=theta<=upper+1e-10:raise ValueError('Angle outside the declared 2.28-5.09 degree domain')
        if len(self.C)==1:return np.ones(1)
        if self.info['angle_law']=='bernstein_sin':
            lo,hi=self.info['angle_domain_deg']
            a,b=np.sin(np.deg2rad([lo,hi])/2)
            t=np.clip((np.sin(np.deg2rad(theta)/2)-a)/(b-a),0,1)
            return np.array([(1-t)**2,2*t*(1-t),t*t])
        if self.info['angle_law']=='original_rational_spline':
            ends=np.array([2.2810596097298803,9.430007907896421]);f=lambda t:np.sin(np.deg2rad(t)/2)**2/(np.sin(np.deg2rad(t)/2)**2+np.sin(np.deg2rad(3.)/2)**2);a,b=f(ends);u=(f(theta)-a)/(b-a);inner=(f(np.array([(2.875894633632971+3.149657426389616)/2,(4.408455007944072+5.085847808123473)/2]))-a)/(b-a)
        else:
            a,b=np.sin(np.deg2rad([lo,hi])/2);u=np.clip((np.sin(np.deg2rad(theta)/2)-a)/(b-a),0,1);inner=[.35,.65]
        knots=np.r_[np.zeros(4),inner,np.ones(4)];values=np.array([float(knots[i]<=u<knots[i+1]) for i in range(9)])
        for degree in range(1,4):
            out=np.zeros(9-degree)
            for i in range(len(out)):
                left=knots[i+degree]-knots[i];right=knots[i+degree+1]-knots[i+1]
                if left>0:out[i]+=(u-knots[i])/left*values[i]
                if right>0:out[i]+=(knots[i+degree+1]-u)/right*values[i+1]
            values=out
        if u==1:values=np.r_[np.zeros(5),1.]
        return values

    def coefficients(self,theta,B,carrier='hole'):
        if carrier not in ['hole','electron']:raise ValueError('Use hole/electron carrier')
        w=self.weights(theta);C=np.einsum('c,cmab->mab',w,self.C);V=np.einsum('c,cgmab->gmab',w,self.V);b1,b2=np.asarray(B);star=np.array([b1,b2-b1,-b2,b1+b2,b2-2*b1,b1-2*b2,2*b1,2*(b2-b1),-2*b2])[:len(V)];both=np.r_[V,V.conj().swapaxes(-1,-2)];G=np.r_[star,-star];T=both.copy()
        for j,(gx,gy) in enumerate(G):
            f=both[j];T[j,0]=f[0]+gx*f[1]/2+gy*f[2]/2;T[j,1]=f[1]+gx*f[3]+gy*f[4]/2;T[j,2]=f[2]+gx*f[4]/2+gy*f[5]
        mask=np.max(abs(T),axis=(1,2,3))>1e-14;sign=1 if carrier=='hole' else -1;return sign*C,sign*T[mask],G[mask]

    def hamiltonian(self,k,theta,B,q,carrier='hole'):
        q=np.asarray(q);p=q+np.asarray(k);n=len(q);d=self.dimension;C,T,G=self.coefficients(theta,B,carrier);phi=np.array([np.ones(n),p[:,0],p[:,1],p[:,0]**2,p[:,0]*p[:,1],p[:,1]**2]);h=np.zeros((n,d,n,d),complex);local=np.einsum('mq,mab->qab',phi,C)
        for i in range(n):h[i,:,i,:]=local[i]
        delta=q[:,None]-q[None]
        for g,t in zip(G,T):
            ii,jj=np.nonzero(np.max(abs(delta-g),axis=-1)<1e-7)
            for i,j in zip(ii,jj):h[i,:,j,:]+=np.einsum('m,mab->ab',phi[:,j],t)
        return h.reshape(n*d,n*d)

    def nqs_profile(self,theta,B,length_A=None,carrier='hole'):
        B=np.asarray(B);C,T,G=self.coefficients(theta,B,carrier);L=float(length_A) if length_A else 4*np.pi/(np.sqrt(3)*np.linalg.norm(B[0]));scale=L**ORDER
        return dict(polynomial_meV=C*1000/scale[:,None,None],fourier_meV=T*1000/scale[None,:,None,None],G_dimensionless=G*L,length_A=L,primitive_lattice=2*np.pi*np.linalg.inv(B).T/L,internal_components=self.dimension,carrier=carrier,monomial_order=['1','px','py','px2','pxpy','py2'],fourier_ordering='right normal; drift and divergence corrections already included',includes_both_Fourier_stars=True)

    def real_parameters(self,theta):
        z=np.load(self.folder/'MATERIAL_CONTROLS.npz');return self.weights(theta)@z['controls']

CaI2Gamma=LocalGammaModel

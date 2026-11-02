#!/usr/bin/env python3
"""Overlay current and archived bands, without changing their energy zero."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from verify_results import load_bands

ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(7.2,3.4),sharey=True)
for ax,valley in zip(axes,['K1','K2']):
    current=load_bands(ROOT/f'outputs/{valley}_spinless/q06/band')*1000
    old=np.sort(np.loadtxt(ROOT/f'reference/bands_{valley}.csv.gz',delimiter=','),axis=1)*1000
    x=np.load(ROOT/f'outputs/{valley}_spinless/q06/band/kpoints.npy')[:,3]
    active=np.any(abs(old)<=180,axis=0)
    ax.plot(x,current[:,active],color='#24292F',lw=.8)
    ax.plot(x[::3],old[::3,active],'.',color='#C43B3B',ms=1.8)
    for tick in x[[20,40]]: ax.axvline(tick,color='#CED4DA',lw=.5)
    ax.axhline(0,color='#8A939B',lw=.5,ls='--')
    ax.set(ylim=(-150,150),xlim=(x[0],x[-1]),xticks=x[[0,20,40,60]],xticklabels=[r'$\Gamma$','M','K',r'$\Gamma$'],
           title=f'{valley}: full graphene + hBN TAPW basis',ylabel='Energy (meV)')
    ax.plot([],[],color='#24292F',lw=.8,label='Current release')
    ax.plot([],[],'.',color='#C43B3B',ms=3,label='Reference TAPW')
    ax.legend(fontsize=7,frameon=False,loc='lower left')
fig.tight_layout()
(ROOT/'verification').mkdir(exist_ok=True)
for ext in ['pdf','png']:
    fig.savefig(ROOT/f'verification/band_comparison.{ext}',dpi=300)
plt.close(fig)

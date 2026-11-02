#!/usr/bin/env python3
"""Check the two archived band paths and export actual TAPW basis metadata."""
import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent


def load_bands(directory):
    parts = [np.loadtxt(directory/f'energies_{edge}.txt', ndmin=2)
             for edge in ['vbm', 'cbm'] if (directory/f'energies_{edge}.txt').is_file()]
    if not parts:
        raise FileNotFoundError(f'No band energies under {directory}')
    return np.sort(np.concatenate(parts, axis=1), axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--energy-tolerance-mev', type=float, default=0.1)
    args = parser.parse_args()
    output = ROOT/'verification'
    output.mkdir(exist_ok=True)
    cells = np.array([[float(v) for v in l.split()] for l in (ROOT/'inputs/POSCAR').read_text().splitlines()[2:5]])
    b_m = (2*np.pi*np.linalg.inv(cells).T)[:2,:2]
    reports, records = {}, []
    passed = True
    for valley in ['K1', 'K2']:
        directory = ROOT/f'outputs/{valley}_spinless/q06/band'
        bands = load_bands(directory)
        reference = np.sort(np.loadtxt(ROOT/f'reference/bands_{valley}.csv.gz', delimiter=',', ndmin=2), axis=1)
        kpoints = np.load(directory/'kpoints.npy', allow_pickle=False)
        if bands.shape != (61,202) or reference.shape != bands.shape or len(kpoints) != 61:
            raise ValueError(f'Unexpected band/k-path shape for {valley}: {bands.shape}, {kpoints.shape}')
        if not np.isfinite(bands).all():
            raise ValueError(f'Nonfinite band energies for {valley}')
        reference_path = np.loadtxt(ROOT/'reference/KPATH_reference.txt', ndmin=2)
        if kpoints.shape != reference_path.shape:
            raise ValueError(f'K-path data shape differs from the archive for {valley}.')
        path_error = float(np.max(abs(kpoints-reference_path)))
        if path_error > 1e-6:
            raise ValueError(f'The sampled band path differs from the archive for {valley}.')
        delta = bands-reference
        low = abs(reference) <= .15
        maximum = float(np.max(abs(delta))*1000)
        entry = {'shape': list(bands.shape), 'energy_alignment': 'none',
                 'max_kpath_table_difference':path_error,
                 'max_abs_energy_difference_mev': maximum,
                 'rms_energy_difference_mev': float(np.sqrt(np.mean(delta**2))*1000),
                 'low_energy_window_ev': [-.15,.15],
                 'low_energy_max_abs_difference_mev': float(np.max(abs(delta[low]))*1000),
                 'low_energy_rms_difference_mev': float(np.sqrt(np.mean(delta[low]**2))*1000),
                 'groups': []}
        passed = passed and maximum <= args.energy_tolerance_mev
        for group, material, expected in [(1,'hBN',27),(2,'graphene',37)]:
            vectors = np.load(directory/f'g_vectors_group{group}.npy', allow_pickle=False)
            old = np.loadtxt(ROOT/f'reference/g_vectors_{valley}_group{group}.csv', delimiter=',')
            if vectors.shape != (expected,2):
                raise ValueError(f'Unexpected {material} vector count: {vectors.shape}')
            center = vectors.mean(axis=0)
            radii = np.linalg.norm(vectors-center,axis=1)
            shell = np.empty(expected,dtype=int)
            shell_radii = []
            for i in np.argsort(radii):
                if not shell_radii or abs(radii[i]-shell_radii[-1]) > .1*np.linalg.norm(b_m[0]):
                    shell_radii.append(float(radii[i]))
                shell[i] = len(shell_radii)
            if len(shell_radii) != 6:
                raise ValueError(f'{material} basis does not contain six radial shells.')
            displacement = center-old.mean(axis=0)
            distances = np.linalg.norm((vectors-displacement)[:,None,:]-old[None,:,:],axis=2)
            residual = float(np.max(np.min(distances,axis=1)))
            # Report the small old primitive-fit origin offset separately from shape.
            passed = passed and residual < 1e-7 and np.linalg.norm(displacement) < 1e-4
            entry['groups'].append({'material':material,'physical_layers': [0] if group==1 else [1,2],
                                    'vector_count':expected, 'orbitals_per_primitive_cell_per_layer':2,
                                    'valley_center_Ainv':center.tolist(),
                                    'radial_shell_radii_Ainv':shell_radii,
                                    'shell_populations':[int(np.count_nonzero(shell==s)) for s in range(1,7)],
                                    'archived_common_offset_Ainv':displacement.tolist(),
                                    'max_set_residual_after_offset_Ainv':residual})
            coefficients = vectors@np.linalg.inv(b_m)
            for i,(g,q,coeff,s) in enumerate(zip(vectors,center-vectors,coefficients,shell),1):
                records.append([valley,material,i,int(s),*coeff,*g,*q])
        reports[valley] = entry
    summary = {'scope':'Archived hBN/R2G band-path reproduction; not a geometry-mesh or cutoff convergence study.',
               'passed':bool(passed),'energy_tolerance_mev':args.energy_tolerance_mev,'valleys':reports,
               'artifact_verification':{'python':platform.python_version(),'numpy':np.__version__},
               'log_paths':['verification/K1.log','verification/K2.log'],
               'provenance_note':'This report describes artifact verification, not the execution history of the band calculations. Machine-local execution logs are not part of the public data bundle.'}
    try:
        summary['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
        summary['working_tree_has_uncommitted_changes']=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        summary['source_commit']=None
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (output/'selected_vectors.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['valley','material','index','shell','m1','m2','Gx_Ainv','Gy_Ainv','Qx_Ainv','Qy_Ainv'])
        writer.writerows(records)
    print(json.dumps(summary,indent=2))
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())

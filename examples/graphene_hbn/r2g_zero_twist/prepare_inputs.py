#!/usr/bin/env python3
"""Prepare an archived orthogonal SK example without changing its matrix units/order."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parent


def write_rigid_reference(source, destination):
    """Reconstruct two ideal graphene sheets without changing source atom order."""
    from ase.io import read, write
    atoms = read(source)
    reference = atoms.copy()
    carbon = np.flatnonzero(np.asarray(atoms.get_chemical_symbols()) == 'C')
    n = int(round(np.sqrt(len(carbon)/4)))
    if not n or len(carbon) != 4*n*n:
        raise ValueError('Expected two graphene layers with two sublattices per primitive cell.')
    lattice = atoms.cell.array[:2,:2]/n
    z = atoms.positions[carbon,2]
    split = (z.min()+z.max())/2
    sublattice = np.arange(len(carbon)) % 2
    report = []
    for layer, mask in enumerate([z < split, z >= split], 1):
        if int(mask.sum()) != 2*n*n:
            raise ValueError('Unequal graphene layer populations in the reference input.')
        indices = carbon[mask]
        frac = atoms.positions[indices,:2] @ np.linalg.inv(lattice)
        candidates = []
        for delta in [np.array([1/3,2/3]), np.array([2/3,1/3])]:
            residual = frac-sublattice[mask,None]*delta
            shift = np.angle(np.mean(np.exp(2j*np.pi*residual),axis=0))/(2*np.pi)
            cells = np.rint(residual-shift).astype(int)
            ideal = cells+shift+sublattice[mask,None]*delta
            displacement = (ideal-frac)@lattice
            candidates.append((float(np.mean(np.sum(displacement**2,axis=1))),cells,ideal,displacement))
        _, cells, ideal, displacement = min(candidates,key=lambda item:item[0])
        for sub in [0,1]:
            if len(np.unique(cells[sublattice[mask]==sub] % n,axis=0)) != n*n:
                raise ValueError('Rigid-reference assignment contains missing or duplicate primitive cells.')
        reference.positions[indices,:2] = ideal@lattice
        reference.positions[indices,2] = z[mask].mean()
        report.append({'layer':layer,'z_A':float(z[mask].mean()),
                       'max_inplane_change_A':float(np.linalg.norm(displacement,axis=1).max())})
    write(destination,reference,format='vasp',direct=False,sort=False)
    destination = Path(destination)
    destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines())+'\n')
    return report


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def convert_sparse_hamiltonian(source, destination):
    source, destination = Path(source), Path(destination)
    with source.open() as stream:
        description = stream.readline().strip()
        nonzeros = int(stream.readline().split()[0])
        dimension = int(stream.readline().split()[0])
        nr = int(stream.readline().split()[0])
    records = np.loadtxt(source, skiprows=4, ndmin=2)
    if records.shape != (nonzeros, 7):
        raise ValueError(f'Sparse record count/column mismatch: header={nonzeros}, data={records.shape}')
    if not np.isfinite(records).all():
        raise ValueError('Nonfinite sparse matrix entries.')
    integer = records[:, :5].astype(np.int64)
    if not np.array_equal(integer, records[:, :5]):
        raise ValueError('Real-space and orbital indices must be integers.')
    if dimension < 1 or integer[:, 3:].min() < 1 or integer[:, 3:].max() > dimension:
        raise ValueError('Sparse orbital indices exceed the declared source dimension.')
    rvecs = np.unique(integer[:, :3], axis=0)
    if len(rvecs) != nr:
        raise ValueError('Real-space block count differs from the source header.')
    metadata = {'schema': 'tapw.sparse-realspace.v1', 'schema_version': 1,
                'basis_dimension': dimension, 'basis_order': 'source', 'energy_unit': 'eV',
                'source_sha256': sha256(source), 'source_description': description}
    payload = {'__tapw_sparse_metadata_json__': np.asarray(json.dumps(metadata, sort_keys=True))}
    for r in rvecs:
        mask = np.all(integer[:, :3] == r, axis=1)
        key = str(tuple(int(v) for v in r))
        payload[key+'_row'] = integer[mask, 3]-1
        payload[key+'_col'] = integer[mask, 4]-1
        payload[key+'_val'] = records[mask, 5]+1j*records[mask, 6]
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return {'source': source.name, 'source_sha256': metadata['source_sha256'],
            'basis_dimension': dimension, 'nonzero_records': nonzeros, 'realspace_blocks': nr,
            'energy_unit': 'eV', 'unit_conversion': 'none', 'basis_order': 'source',
            'prepared_sha256': sha256(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Reference source directory containing POSCAR, H.dat and tapw/Q_shell_6.')
    args = parser.parse_args()
    source = args.source.resolve()
    for name in ['POSCAR', 'H.dat']:
        if not (source/name).is_file():
            raise FileNotFoundError(source/name)
    inputs, reference = ROOT/'inputs', ROOT/'reference'
    inputs.mkdir(exist_ok=True); reference.mkdir(exist_ok=True)
    shutil.copy2(source/'POSCAR', inputs/'POSCAR_relaxed')
    rigid_report = write_rigid_reference(inputs/'POSCAR_relaxed',inputs/'POSCAR_rigid')
    report = convert_sparse_hamiltonian(source/'H.dat', inputs/'H.npz')
    report['structure_sha256'] = sha256(inputs/'POSCAR_relaxed')
    report['source_structure'] = 'POSCAR'
    report['reference_structure'] = 'POSCAR_rigid'
    report['reference_structure_sha256'] = sha256(inputs/'POSCAR_rigid')
    report['rigid_reference_layers'] = rigid_report
    report['reference_files'] = {}
    for valley in ['K1', 'K2']:
        old = source/f'tapw/Q_shell_6/band_data/band_data_{valley}_valley.txt'
        if not old.is_file():
            raise FileNotFoundError(old)
        bands = np.loadtxt(old, ndmin=2)
        if bands.shape != (61, 202) or not np.isfinite(bands).all():
            raise ValueError(f'Unexpected archived band data: {old}, {bands.shape}')
        np.savetxt(reference/f'bands_{valley}.csv.gz', bands, delimiter=',', fmt='%.12e')
        report['reference_files'][old.relative_to(source).as_posix()] = sha256(old)
        for group in [1, 2]:
            old_g = source/f'tapw/Q_shell_6/g_vec_list_6_{valley}_{group}layer.npy'
            vectors = np.load(old_g, allow_pickle=False)
            np.savetxt(reference/f'g_vectors_{valley}_group{group}.csv', vectors, delimiter=',', fmt='%.16e')
            report['reference_files'][old_g.relative_to(source).as_posix()] = sha256(old_g)
    old_path = source/'tapw/KPATH.out'
    shutil.copy2(old_path, reference/'KPATH_reference.txt')
    report['reference_files'][old_path.relative_to(source).as_posix()] = sha256(old_path)
    (inputs/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

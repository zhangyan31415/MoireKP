"""SIESTA HSX -> canonical TAPW real-space input, using optional sisl.

HSX 1/2 is read through sisl's public API.  sisl returns H-Ef*S in eV;
absolute export restores Ef exactly once.  All exported orbitals use TAPW's
positive Cartesian real harmonics and blocked spin ordering.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile
import warnings
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
from scipy import sparse

from .hr import HrSparseHandler

_M_ORDER = {0: (0,), 1: (1, -1, 0), 2: (0, 2, -2, 1, -1), 3: (0, 1, -1, 2, -2, 3, -3)}
_NAMES = {0: ('s',), 1: ('px', 'py', 'pz'), 2: ('dz2', 'dx2-y2', 'dxy', 'dxz', 'dyz'),
          3: ('fz3', 'fxz2', 'fyz2', 'fz(x2-y2)', 'fxyz', 'fx(x2-3y2)', 'fy(3x2-y2)')}


@dataclass(frozen=True)
class OrbitalLayout:
    permutation: np.ndarray  # target index -> source index
    phases: np.ndarray
    specification: str


def orbital_layout(orbitals: Sequence[dict[str, int]]) -> OrbitalLayout:
    """Map complete SIESTA (n,l,zeta,m) shells to TAPW's real basis."""
    if not orbitals:
        raise ValueError('No orbital metadata; self-describing HSX 1/2 is required.')
    shells: dict[tuple[int, int, int], dict[int, int]] = defaultdict(dict)
    for index, orb in enumerate(orbitals):
        values = []
        for key in ('n', 'l', 'zeta', 'm'):
            val = orb.get(key)
            if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, np.integer)):
                raise ValueError(f'Orbital {index}: integer {key} metadata required.')
            values.append(int(val))
        n, l, zeta, m = values
        if l not in _M_ORDER or n < 1 or zeta < 1 or abs(m) > l:
            raise ValueError(f'Unsupported orbital quantum numbers: {orb}; supported shells are s/p/d/f.')
        group = shells[(l, n, zeta)]
        if m in group:
            raise ValueError(f'Duplicate orbital in shell (l,n,zeta)={(l,n,zeta)}: m={m}.')
        group[m] = index
    permutation, phases = [], []
    counts = [0, 0, 0, 0]
    for (l, n, zeta), entries in sorted(shells.items()):
        if set(entries) != set(range(-l, l+1)):
            raise ValueError(f'Incomplete angular shell (l,n,zeta)={(l,n,zeta)}.')
        counts[l] += 1
        for m in _M_ORDER[l]:
            permutation.append(entries[m])
            phases.append((-1)**abs(m))
    specification = ''.join(f'{label}{n}' for label, n in zip('spdf', counts) if n)
    return OrbitalLayout(np.asarray(permutation, dtype=np.int64), np.asarray(phases, dtype=float), specification)


def realspace_blocks(matrix, offsets, permutation, phases) -> dict:
    """Convert public supercell CSR at Gamma, preserving R and basis phases."""
    mat = sparse.coo_matrix(matrix, dtype=np.complex128)
    offsets = np.asarray(offsets)
    permutation = np.asarray(permutation)
    phases = np.asarray(phases, dtype=float)
    if permutation.ndim != 1 or permutation.dtype.kind not in 'iu':
        raise ValueError('Orbital permutation must be a one-dimensional integer array.')
    n = len(permutation)
    if (offsets.ndim != 2 or offsets.shape[1] != 3 or n == 0
            or mat.shape != (n, n * len(offsets))):
        raise ValueError('Supercell matrix shape does not match basis dimension and lattice offsets.')
    if not np.isfinite(offsets).all() or not np.equal(offsets, np.rint(offsets)).all():
        raise ValueError('Lattice offsets must be finite integer triples.')
    if np.any(offsets < -(2**63)) or np.any(offsets >= 2**63):
        raise ValueError('Lattice offsets exceed the supported integer range.')
    offsets = offsets.astype(np.int64)
    if len({tuple(r) for r in offsets}) != len(offsets):
        raise ValueError('Lattice offsets must be unique.')
    if not np.array_equal(np.sort(permutation), np.arange(n)):
        raise ValueError('Invalid orbital permutation.')
    if phases.shape != (n,) or not np.isin(phases, [-1., 1.]).all():
        raise ValueError('Orbital phases must be +1 or -1.')
    if not np.isfinite(mat.data).all():
        raise ValueError('Hamiltonian/overlap entries must be finite.')
    with np.errstate(over='ignore', invalid='ignore'):
        mat.sum_duplicates()
    if not np.isfinite(mat.data).all():
        raise ValueError('Coalesced Hamiltonian/overlap entries must be finite.')
    inv = np.empty(n, dtype=np.int64)
    inv[permutation.astype(np.int64)] = np.arange(n)
    rows = inv[mat.row]
    cols = inv[mat.col % n]
    cells = mat.col // n
    values = mat.data * phases[rows] * phases[cols]
    blocks = {}
    for cell in np.unique(cells):
        pick = (cells == cell) & (values != 0)
        if not np.any(pick):
            continue
        r, c, v = rows[pick], cols[pick], values[pick]
        order = np.lexsort((c, r))
        blocks[tuple(map(int, offsets[cell]))] = {'row': r[order], 'col': c[order], 'val': v[order]}
    return blocks


def _double_blocks(up: dict, down: dict, n: int) -> dict:
    out = {}
    for r in sorted(set(up) | set(down)):
        records = [(up.get(r), 0), (down.get(r), n)]
        out[r] = {key: np.concatenate([v[key] + shift if key != 'val' else v[key]
                                      for v, shift in records if v is not None])
                  for key in ('row', 'col', 'val')}
    return out


def _check_hermitian(blocks: dict, dimension: int, name: str, tolerance: float = 1e-7) -> float:
    maximum = 0.
    for r, block in blocks.items():
        opposite = blocks.get(tuple(-x for x in r))
        a = sparse.coo_matrix((block['val'], (block['row'], block['col'])), shape=(dimension, dimension)).tocsr()
        b = sparse.csr_matrix(a.shape, dtype=complex) if opposite is None else sparse.coo_matrix(
            (opposite['val'], (opposite['row'], opposite['col'])), shape=a.shape).tocsr()
        error = a-b.conjugate().T
        residual = float(np.max(np.abs(error.data), initial=0))
        maximum = max(maximum, residual)
        scale = max(1., float(np.max(np.abs(a.data), initial=0)), float(np.max(np.abs(b.data), initial=0)))
        if not np.isfinite(residual) or not np.isfinite(scale):
            raise ValueError(f'{name} Hermiticity residual must be finite.')
        if residual > tolerance * scale:
            raise ValueError(f'{name} is not Hermitian across R={r}: residual {residual:.3g}.')
    return maximum


def _basis_layout(geometry, species_map=None):
    from ase.data import atomic_numbers
    species_map = dict(species_map or {})
    permutation, phases, sites, basis = [], [], [], []
    species_specs = {}
    scalar_offset = 0
    for site, atom in enumerate(geometry.atoms):
        tag = str(atom.tag)
        symbol = species_map.get(tag, tag)
        if symbol not in atomic_numbers or atomic_numbers[symbol] <= 0:
            raise ValueError(f'Unresolved SIESTA species {tag!r}; supply --species {tag}=ELEMENT.')
        if int(atom.Z) <= 0:
            raise ValueError(f'Ghost or unresolved species {tag!r} is not supported.')
        if int(atom.Z) != atomic_numbers[symbol]:
            raise ValueError(f'Atomic identity for {tag!r} is inconsistent.')
        orbitals = [{'n': o.n, 'l': o.l, 'm': o.m, 'zeta': o.zeta} for o in atom.orbitals]
        layout = orbital_layout(orbitals)
        # Validate before converting numpy integer scalars for portable JSON.
        orbitals = [{key: int(value) for key, value in orb.items()} for orb in orbitals]
        if symbol in species_specs and species_specs[symbol] != layout.specification:
            raise ValueError(f'Different per-site basis specifications for {symbol}; canonical species mapping is ambiguous.')
        species_specs[symbol] = layout.specification
        # Equal counts but different n/zeta sets must also not masquerade as one radial basis.
        signature = sorted((o['n'],o['l'],o['zeta'],o['m']) for o in orbitals)
        previous = next((entry for entry in sites if entry['element'] == symbol), None)
        if previous is not None and previous['source_species'] != tag:
            raise ValueError(f'Different source species for {symbol} cannot be collapsed to one radial basis; '
                             'HSX quantum-number counts do not establish identical radial functions or pseudopotentials.')
        if previous is not None and previous['source_quantum_numbers'] != signature:
            raise ValueError(f'Different radial basis identities for {symbol} on different sites.')
        sites.append({'site_index': site, 'source_species': tag, 'element': symbol,
                      'orbital_specification': layout.specification, 'source_quantum_numbers': signature,
                      'position_angstrom': np.asarray(geometry.xyz[site], float).tolist()})
        for local_target, local_source in enumerate(layout.permutation):
            orb = orbitals[int(local_source)]
            basis.append({'target_index': len(basis), 'source_index': scalar_offset+int(local_source),
                          'site_index': site, **orb, 'phase': int(layout.phases[local_target]),
                          'target_name': _NAMES[orb['l']][_M_ORDER[orb['l']].index(orb['m'])]})
        permutation.extend((layout.permutation+scalar_offset).tolist())
        phases.extend(layout.phases.tolist())
        scalar_offset += len(orbitals)
    if scalar_offset != geometry.no:
        raise ValueError('Geometry orbital metadata does not match matrix basis dimension.')
    unknown = set(species_map)-{s['source_species'] for s in sites}
    if unknown:
        raise ValueError(f'Unused species overrides: {sorted(unknown)}')
    return np.asarray(permutation, dtype=np.int64), np.asarray(phases), sites, basis, species_specs


def convert_hamiltonian(hamiltonian, *, fermi_ev: float, energy_reference='absolute', species_map=None):
    """Convert an sisl Hamiltonian which is already referenced to its Ef.

    This function uses public Hk/Sk supercell APIs; private sparse spin components
    are deliberately not decoded a second time. It performs no diagonalization.
    """
    if energy_reference not in ('absolute', 'fermi'):
        raise ValueError('energy_reference must be absolute or fermi.')
    if not np.isfinite(fermi_ev):
        raise ValueError('A finite HSX Fermi energy is required.')
    h = hamiltonian.copy()
    if energy_reference == 'absolute':
        h.shift(float(fermi_ev))
    geom = h.geometry
    cell = np.asarray(geom.cell, dtype=float)
    if cell.shape != (3,3) or not np.isfinite(cell).all() or abs(np.linalg.det(cell)) < 1e-10:
        raise ValueError('A finite nonsingular 3D cell is required.')
    if not np.isfinite(geom.xyz).all():
        raise ValueError('Atomic positions must be finite.')
    perm, phase, sites, basis, specs = _basis_layout(geom, species_map)
    offsets = np.asarray(geom.lattice.sc_off)
    kwargs = dict(k=(0.,0.,0.), gauge='lattice', format='sc:csr', dtype=np.complex128)
    spin = h.spin
    n = geom.no
    if spin.is_unpolarized:
        mode='unpolarized';dimension=n
        H=realspace_blocks(h.Hk(**kwargs), offsets, perm, phase)
        S=realspace_blocks(h.Sk(**kwargs), offsets, perm, phase)
    elif spin.is_polarized:
        mode='polarized';dimension=2*n
        up=realspace_blocks(h.Hk(spin=0, **kwargs), offsets, perm, phase)
        down=realspace_blocks(h.Hk(spin=1, **kwargs), offsets, perm, phase)
        H=_double_blocks(up, down, n)
        scalar_s=realspace_blocks(h.Sk(**kwargs), offsets, perm, phase)
        S=_double_blocks(scalar_s, scalar_s, n)
    elif spin.is_noncolinear or spin.is_spinorbit:
        mode='spinorbit' if spin.is_spinorbit else 'noncollinear';dimension=2*n
        gather=np.concatenate([2*perm, 2*perm+1]);signs=np.tile(phase, 2)
        H=realspace_blocks(h.Hk(**kwargs), offsets, gather, signs)
        S=realspace_blocks(h.Sk(**kwargs), offsets, gather, signs)
    else:
        raise ValueError(f'Unsupported SIESTA spin mode {spin}; Nambu/particle-hole input is not supported.')
    # A valid one-level H can become identically zero at its Fermi reference.
    # The explicit basis dimension is retained in the sparse NPZ metadata.
    if not S:
        raise ValueError('Empty overlap matrix.')
    residuals={'H': _check_hermitian(H, dimension, 'H'), 'S': _check_hermitian(S, dimension, 'S')}
    metadata={'schema':'tapw.siesta-import.v1', 'basis_dimension':int(dimension),
              'scalar_basis_dimension':int(n),'spin_mode':mode,'spin':mode!='unpolarized',
              'basis_order':'source',
              'spin_order':'blocked_up_then_down' if mode!='unpolarized' else 'scalar',
              'hamiltonian_unit':'eV','overlap_unit':'dimensionless','length_unit':'angstrom',
              'energy_reference':energy_reference,'source_fermi_ev':float(fermi_ev),
              'energy_zero_ev':0. if energy_reference=='absolute' else float(fermi_ev),
              'sisl_read_energy_convention':'H_file - Ef*S; absolute export restores Ef*S once',
              'fourier_convention':'sum_R M(R) exp(+2*pi*i*k_fractional.R)',
              'orbital_convention':'TAPW positive Cartesian real harmonics; permutation and (-1)^abs(m) phases',
              'cell_angstrom':cell.tolist(),'sites':sites,'scalar_basis':basis,
              'system_orbitals':specs,'hermiticity_max_entry_residual':residuals,
              'translation_vectors':sorted([list(r) for r in set(H)|set(S)]),
              'scope':'matrix and basis import; no moire geometry, low-energy model, or physical quantum geometry is inferred'}
    return H,S,metadata


def publish_directory(stage: Path, output: Path) -> None:
    """Reserve a fresh directory exclusively; publish the completion metadata last."""
    output.mkdir()  # Fails for existing directories and all symlinks, including races.
    linked = []
    directories = [output]
    try:
        entries = list(stage.rglob('*'))
        if any(path.is_symlink() for path in entries):
            raise ValueError('Publication staging tree must not contain symbolic links.')
        for source in sorted((p for p in entries if p.is_dir()), key=lambda p: len(p.parts)):
            dest = output / source.relative_to(stage)
            dest.mkdir()
            directories.append(dest)
        def publication_order(path):
            completion = (2 if path.name == 'preparation.json' else
                          1 if path.name.endswith('_import.json') or path.name == 'symmetrization_report.json' else 0)
            return completion, str(path.relative_to(stage))
        files = sorted((p for p in entries if p.is_file()), key=publication_order)
        for source in files:
            dest = output / source.relative_to(stage)
            os.link(source, dest)  # Same filesystem; atomic no-overwrite per file.
            linked.append((source, dest))
    except BaseException:
        for source, dest in linked:
            if dest.exists() and source.samefile(dest):
                dest.unlink()
        for directory in reversed(directories):
            try:
                directory.rmdir()  # Do not remove any concurrently added foreign files.
            except OSError:
                pass
        raise


def _source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def import_siesta(source, output, *, energy_reference='absolute', species_map=None, write_dat=False):
    """Read HSX v1/2 into a new reserved directory; completion metadata is last."""
    try:
        import sisl
    except ImportError as exc:
        raise ImportError('SIESTA import requires the optional sisl dependency: install moirekp[siesta].') from exc
    from ase import Atoms
    from ase.io import write
    source=Path(source).expanduser().resolve()
    requested=Path(output).expanduser().absolute()
    output=requested.parent.resolve()/requested.name
    if not source.is_file() or source.suffix.upper()!='.HSX':
        raise ValueError('Input must be an existing SIESTA .HSX file (version 1 or 2).')
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output already exists: {output}; choose a new output directory.')
    before=source.stat()
    source_digest=_source_hash(source)
    reader=sisl.get_sile(str(source))
    version=reader.version
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer)) or version not in (1,2):
        raise ValueError(f'HSX version {version} is not supported; regenerate a self-describing HSX 1/2 with SIESTA >=5.')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fermi=reader.read_fermi_level()
        ham=reader.read_hamiltonian()
    if fermi is None or not np.isfinite(fermi):
        raise ValueError('HSX does not contain a usable Fermi energy; use a completed SCF output.')
    if caught:
        details='; '.join(str(w.message) for w in caught)
        raise ValueError('SIESTA reader reported incomplete or ambiguous input: '+details)
    if ham is None:
        raise ValueError('No Hamiltonian was read from the HSX input.')
    after=source.stat()
    if ((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) !=
            (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
            or _source_hash(source) != source_digest):
        raise ValueError('HSX source changed while being read; use a completed, stable output.')
    H,S,metadata=convert_hamiltonian(ham,fermi_ev=float(fermi),energy_reference=energy_reference,species_map=species_map)
    metadata.update(source_hsx=str(source),source_sha256=source_digest,hsx_version=int(version),
                    sisl_version=importlib.metadata.version('sisl'))
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.'+output.name+'-',dir=output.parent))
    try:
        for name,blocks in [('H',H),('S',S)]:
            writer=HrSparseHandler();writer.hr_sparse=blocks
            writer.save_to_npz(temp/(name+'.npz'),basis_dimension=metadata['basis_dimension'],
                               basis_order='source')
            if write_dat:
                with (temp/(name+'.dat')).open('w') as f:
                    count=sum(len(v['val']) for v in blocks.values())
                    f.write(f'SIESTA import {name}; '+('eV' if name=='H' else 'dimensionless')+'\n')
                    f.write(f'{count}\n{metadata["basis_dimension"]}\n{len(blocks)}\n')
                    for r,v in sorted(blocks.items()):
                        for i,j,z in zip(v['row'],v['col'],v['val']):
                            f.write(f'{r[0]} {r[1]} {r[2]} {i+1} {j+1} {z.real:.17g} {z.imag:.17g}\n')
        atoms=Atoms(symbols=[site['element'] for site in metadata['sites']],positions=[site['position_angstrom'] for site in metadata['sites']],cell=metadata['cell_angstrom'],pbc=True)
        write(str(temp/'structure.extxyz'),atoms,format='extxyz')
        (temp/'siesta_import.json').write_text(json.dumps(metadata,indent=2)+'\n')
        import yaml
        fragment={'system':{'structure':'structure.extxyz','hamiltonian':'H.npz','overlap':'S.npz',
                            'orbitals':metadata['system_orbitals'],'spin':metadata['spin']}}
        (temp/'system.fragment.yaml').write_text('# Input fragment only: supply output, twist_index and layers for a valid moire calculation.\n# Do not infer moire metadata from a primitive-cell SIESTA validation case.\n'+yaml.safe_dump(fragment,sort_keys=False))
        (temp/'README.md').write_text('SIESTA import. H.npz is in eV; S.npz is dimensionless.\nSee siesta_import.json for the exact energy reference, basis transformation and source hash.\nstructure.extxyz and system.fragment.yaml preserve site order. The YAML is an input fragment, not a runnable moire calculation. Supply actual output/twist_index/layers and workflow parameters.\nNo symmetry averaging, orbital truncation, energy fit, or spatial wrapping has been applied.\n')
        publish_directory(temp, output)
        shutil.rmtree(temp)
    except BaseException:
        shutil.rmtree(temp)
        raise
    return metadata


def main(argv=None, *, prog='tapw import-siesta'):
    parser=argparse.ArgumentParser(prog=prog,description='Import a self-describing SIESTA HSX 1/2 into TAPW H/S and basis files.')
    parser.add_argument('hsx',metavar='INPUT.HSX')
    parser.add_argument('--output',required=True,type=Path,help='New output directory (never overwritten)')
    parser.add_argument('--energy-reference',choices=['absolute','fermi'],default='absolute')
    parser.add_argument('--species',action='append',default=[],metavar='LABEL=ELEMENT',help='Explicit element for a custom SIESTA species label')
    parser.add_argument('--write-dat',action='store_true',help='Also write seven-column H.dat/S.dat')
    args=parser.parse_args(argv)
    mapping={}
    for item in args.species:
        if '=' not in item:parser.error('--species requires LABEL=ELEMENT')
        label,element=item.split('=',1)
        if not label or not element or label in mapping:parser.error('Species labels must be nonempty and unique')
        mapping[label]=element
    try:
        metadata=import_siesta(args.hsx,args.output,energy_reference=args.energy_reference,species_map=mapping,write_dat=args.write_dat)
    except (ValueError,FileNotFoundError,FileExistsError,ImportError) as exc:
        parser.error(str(exc))
    print(f'SIESTA import written to {args.output.resolve()}')
    print(f'Basis: {metadata["basis_dimension"]}, spin: {metadata["spin_mode"]}, H: eV ({metadata["energy_reference"]})')
    return 0


if __name__=='__main__':
    raise SystemExit(main())

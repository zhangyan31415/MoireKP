"""Lossless OpenMX scfout import using the native OpenMX reader."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

from .hr import HrSparseHandler
from .siesta import _check_hermitian, _source_hash, publish_directory


def _validate_source_geometry(structure, source):
    """Reject wrong geometry/site order before attaching orbital metadata."""
    for name, expected, actual in (
        ('cell', structure.lattice, source['cell_angstrom']),
        ('positions', [atom.cart for atom in structure.atoms], source['positions_angstrom']),
    ):
        expected, actual = np.asarray(expected), np.asarray(actual)
        if (actual.shape != expected.shape or not np.isfinite(actual).all()
                or not np.allclose(actual, expected, atol=2e-6, rtol=1e-8)):
            raise ValueError(f'OpenMX scfout {name} do not match the supplied structure/site order.')
    expected_spin = {'none': 0, 'collinear': 1, 'spinor': 3}[structure.spin_mode]
    if source['spin_switch'] != expected_spin:
        raise ValueError('OpenMX scfout spin mode does not match supplied structure.')
    if not np.isfinite(source['source_fermi_ev']):
        raise ValueError('OpenMX scfout Fermi energy must be finite.')


def import_openmx(source, output, *, structure, binary=None, write_dat=False, threads=1):
    """Import H/S without averaging, dropping entries, wrapping, or shifting Ef.

    ``binary`` must be built from scripts/openmx/openmx_analysis_symm_hs.cpp.
    A fresh output directory is required, including when a prior run failed.
    """
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output already exists: {output}')
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError('threads must be a positive integer.')
    source, structure_path = Path(source).resolve(strict=True), Path(structure).resolve(strict=True)
    if binary is None:
        binary = os.environ.get('TAPW_OPENMX_READER') or shutil.which('analysis_symm_hs')
        if binary is None:
            candidate = Path(__file__).resolve().parents[3] / 'build/openmx_import_hs/analysis_symm_hs'
            if candidate.is_file():
                binary = candidate
    if binary is None:
        raise FileNotFoundError('Supply --binary or TAPW_OPENMX_READER; build scripts/openmx/build_openmx_symm_hs.sh first.')
    binary = Path(binary).resolve(strict=True)
    from .hs_symmetry import (parse_openmx_structure, SymmetryOperation,
                             build_operation_transports, write_symmetry_transports_json, _section)
    from ase import Atoms
    from ase.data import atomic_numbers
    from ase.io import write
    import yaml

    geometry = parse_openmx_structure(structure_path)
    if [a.index for a in geometry.atoms] != list(range(1, len(geometry.atoms)+1)):
        raise ValueError('OpenMX site indices must be contiguous and in source order.')
    source_pseudos = {}
    for row in _section(structure_path.read_text().splitlines(), 'Definition.of.Atomic.Species'):
        fields = row.split()
        if len(fields) < 3:
            raise ValueError('Each OpenMX species requires PAO and pseudopotential identifiers.')
        if fields[0] in source_pseudos:
            raise ValueError(f'Duplicate OpenMX species definition: {fields[0]}.')
        source_pseudos[fields[0]] = fields[2]
    species_specs, sites, element_paos, element_pseudos = {}, [], {}, {}
    for i, atom in enumerate(geometry.atoms):
        # Species tags may be arbitrary; the PAO filename identifies the element.
        match = re.fullmatch(r'([A-Z][a-z]?)\d+(?:\.\d+)?-([spdf]\d+(?:[spdf]\d+)*)', atom.orbital_spec)
        if match is None or match[1] not in atomic_numbers:
            raise ValueError(f'Unsupported OpenMX basis identity: {atom.orbital_spec!r}.')
        element, spec = match.groups()
        pseudo = source_pseudos[atom.species]
        if element in element_paos and element_paos[element] != atom.orbital_spec:
            raise ValueError(f'Different per-site PAO basis identities for {element}.')
        if element in element_pseudos and element_pseudos[element] != pseudo:
            raise ValueError(f'Different per-site pseudopotential identities for {element}.')
        element_paos[element] = atom.orbital_spec
        element_pseudos[element] = pseudo
        species_specs[element] = spec
        sites.append({'site_index': i, 'source_species': atom.species, 'element': element,
                      'position_angstrom': atom.cart.tolist(), 'orbital_specification': spec,
                      'source_pao_basis': atom.orbital_spec, 'source_pseudopotential': pseudo})
    source_digest, structure_digest = _source_hash(source), _source_hash(structure_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.'+output.name+'-', dir=output.parent))
    try:
        identity = [SymmetryOperation(0, np.eye(3), np.zeros(3), np.eye(3), np.zeros(3))]
        transports = build_operation_transports(geometry, identity, atom_tol=1e-7)
        transport_path = stage/'identity_transports.json'
        write_symmetry_transports_json(transport_path, geometry, identity, transports, symprec=1e-7)
        command = [str(binary), str(source), '--symmetry-json', str(transport_path),
                   '--output-dir', str(stage), '--output-format', 'both' if write_dat else 'npz',
                   '--cutoff', '0', '--no-symmetrize', '--threads', str(threads)]
        environment = os.environ.copy()
        environment.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with (stage/'native_import.log').open('w') as log:
            # OpenMX's read_scfout writes a fixed temporal_12345.input in cwd.
            # Keep it inside this import's private stage to avoid caller files
            # and concurrent readers interfering with one another.
            run = subprocess.run(command, cwd=stage, stdout=log, stderr=subprocess.STDOUT, env=environment)
        if run.returncode:
            diagnostic = (stage/'native_import.log').read_text(errors='replace')[-6000:]
            raise RuntimeError(f'OpenMX reader exited {run.returncode}:\n{diagnostic}')
        (stage/'temporal_12345.input').unlink(missing_ok=True)
        embedded = json.loads((stage/'scfout_metadata.json').read_text())
        _validate_source_geometry(geometry, embedded)
        residuals = {}
        translations = set()
        for label in ('H', 'S'):
            handler = HrSparseHandler()
            handler.load_from_npz(stage/(label+'.npz'))
            residuals[label] = _check_hermitian(handler.hr_sparse, geometry.nwann, label)
            translations.update(handler.hr_sparse)
            handler.save_to_npz(
                stage/(label+'.npz'),
                basis_dimension=geometry.nwann,
                basis_order='source',
            )
        if _source_hash(source) != source_digest or _source_hash(structure_path) != structure_digest:
            raise RuntimeError('OpenMX input changed while it was being imported.')
        mode = {'none':'unpolarized', 'collinear':'polarized', 'spinor':'spinorbit'}[geometry.spin_mode]
        metadata = {
            'schema':'tapw.openmx-import.v1', 'source_scfout':str(source), 'source_sha256':source_digest,
            'source_structure':str(structure_path), 'source_structure_sha256':structure_digest,
            'native_binary':str(binary), 'native_binary_sha256':_source_hash(binary), 'command':command,
            'threads':threads, 'basis_dimension':geometry.nwann, 'scalar_basis_dimension':geometry.nwann_spinless,
            'spin':geometry.spinful, 'spin_mode':mode, 'spin_order':'blocked_up_then_down' if geometry.spinful else 'scalar',
            'hamiltonian_unit':'eV', 'overlap_unit':'dimensionless', 'length_unit':'angstrom',
            'hartree_to_ev':27.2113862, 'energy_reference':'absolute', 'energy_zero_ev':0.,
            'source_fermi_ev':embedded['source_fermi_ev'], 'cell_angstrom':geometry.lattice.tolist(),
            'sites':sites, 'system_orbitals':species_specs, 'hermiticity_max_entry_residual':residuals,
            'translation_vectors':sorted(map(list, translations)),
            'fourier_convention':'sum_R M(R) exp(+2*pi*i*k_fractional.R)',
            'orbital_convention':'OpenMX positive Cartesian real harmonics; no permutation or phase change',
            'symmetrized':False, 'cutoff':0.,
        }
        write(stage/'structure.extxyz', Atoms(symbols=[s['element'] for s in sites],
              positions=[s['position_angstrom'] for s in sites], cell=geometry.lattice, pbc=True), format='extxyz')
        fragment = {'system':{'structure':'structure.extxyz', 'hamiltonian':'H.npz', 'overlap':'S.npz',
                              'orbitals':species_specs, 'spin':geometry.spinful}}
        (stage/'system.fragment.yaml').write_text('# Input fragment: add output, twist_index, layers, and workflow parameters.\n'+yaml.safe_dump(fragment, sort_keys=False))
        (stage/'openmx_import.json').write_text(json.dumps(metadata, indent=2)+'\n')
        (stage/'README.md').write_text('OpenMX raw import: H.npz in eV, S.npz dimensionless.\nNo symmetry averaging, orbital truncation, coordinate wrapping, or Fermi shift.\nSee openmx_import.json for provenance and basis information.\n')
        publish_directory(stage, output)
        return metadata
    finally:
        shutil.rmtree(stage)

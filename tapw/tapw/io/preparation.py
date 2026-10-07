"""Common preparation interface for localized-orbital DFT H/S matrices.

Backend importers certify source conventions. Symmetry averaging is a separate,
explicit operation and never changes the raw imported pair.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import yaml

from .hr import HrSparseHandler
from .siesta import publish_directory


BACKENDS = ('openmx', 'abacus', 'siesta')
FORMATS = ('npz', 'dat', 'both')


def dispatch(argv=None):
    """Installed CLI entry point, equivalent to the per-backend scripts."""
    import sys
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='tapw prepare-hs',
        description='Prepare OpenMX, ABACUS or SIESTA H/S. Use BACKEND --help for options.')
    parser.add_argument('backend', choices=BACKENDS)
    if not arguments or arguments[0] in ('-h', '--help'):
        parser.parse_args(arguments)
    backend = parser.parse_args(arguments[:1]).backend
    return main(backend, arguments[1:])


def _digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _new_destination(path):
    # Resolve parents but never follow the final component: a dangling symlink
    # is an existing user-owned destination too.
    path = Path(path).expanduser().absolute()
    path = path.parent.resolve() / path.name
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    return path


def _write_pair_matrix(path, blocks, dimension, *, output_format='npz'):
    """Write exact-dimension sparse matrices; text retains double precision."""
    if output_format not in FORMATS:
        raise ValueError(f'Unknown output format {output_format!r}.')
    path = Path(path)
    if output_format in ('npz', 'both'):
        writer = HrSparseHandler()
        writer.hr_sparse = blocks
        writer.save_to_npz(path.with_suffix('.npz'), basis_dimension=int(dimension))
    if output_format in ('dat', 'both'):
        with path.with_suffix('.dat').open('w') as handle:
            label = path.name.split('_')[0]
            count = sum(len(block['val']) for block in blocks.values())
            handle.write(f'{label}; ' + ('eV' if label == 'H' else 'dimensionless') + '\n')
            handle.write(f'{count}\n{dimension}\n{len(blocks)}\n')
            for r, block in sorted(blocks.items()):
                for row, col, value in zip(block['row'], block['col'], block['val']):
                    handle.write(f'{r[0]} {r[1]} {r[2]} {int(row)+1} {int(col)+1} '
                                 f'{value.real:.17g} {value.imag:.17g}\n')


def _metadata(source):
    manifests = [source / f'{backend}_import.json' for backend in BACKENDS
                 if (source / f'{backend}_import.json').is_file()]
    if len(manifests) != 1:
        raise ValueError('Expected exactly one OpenMX, ABACUS or SIESTA import manifest.')
    metadata = json.loads(manifests[0].read_text())
    return metadata, manifests[0]


def _canonical_structure(source, metadata, *, magnetic_moments=None,
                         assume_nonmagnetic=False):
    from ase.io import read
    from . import hs_symmetry as core

    atoms = read(source / 'structure.extxyz')
    specifications = metadata['system_orbitals']
    spin = bool(metadata['spin'])
    moments = magnetic_moments
    if moments is None:
        moments = metadata.get('site_magnetic_moments')
    if moments is None and spin and not assume_nonmagnetic:
        raise ValueError('Spinful symmetry requires site magnetic moments, or an explicit '
                         '--assume-nonmagnetic confirmation for a nonmagnetic system.')
    if moments is not None:
        moments = np.asarray(moments, dtype=float)
        if moments.shape != (len(atoms), 3) or not np.isfinite(moments).all():
            raise ValueError('Magnetic moments must be a finite natoms-by-3 array in original site order.')
        if assume_nonmagnetic and np.max(np.abs(moments), initial=0.) > 1e-8:
            raise ValueError('Nonzero magnetic moments conflict with --assume-nonmagnetic.')
    records = []
    for i, (symbol, position, frac) in enumerate(zip(
            atoms.get_chemical_symbols(), atoms.positions,
            atoms.get_scaled_positions(wrap=False))):
        records.append(core.AtomRecord(i+1, symbol, frac, position, specifications[symbol],
                                      None if moments is None else moments[i]))
    # Magnetic-space-group filtering rotates moments as physical axial vectors.
    # Its matrix action must therefore include the matching SU(2) rotation even
    # for a collinear H; combining fixed spin channels with time reversal would
    # spuriously average away a ferromagnet's exchange splitting.
    mode = 'spinor' if spin else 'none'
    structure = core.OpenMXStructure(source / 'structure.extxyz', atoms.cell.array,
                                    records, specifications, spinful=spin, spin_mode=mode)
    if structure.nwann != int(metadata['basis_dimension']):
        raise ValueError('Structure and orbital metadata do not match imported basis dimension.')
    return structure


def symmetrize_import(source, output, *, output_format='npz', symprec=1e-3,
                      cutoff=0., threads=1, magnetic_moments=None,
                      assume_nonmagnetic=False):
    """Write a separate symmetrized H/S pair from a canonical backend import."""
    from . import hs_symmetry as core

    source, output = Path(source).resolve(), _new_destination(output)
    if output_format not in FORMATS:
        raise ValueError(f'Unknown output format {output_format!r}.')
    if threads < 1 or symprec <= 0 or cutoff < 0 or not np.isfinite([symprec, cutoff]).all():
        raise ValueError('threads/symprec must be positive and cutoff finite and nonnegative.')
    metadata, manifest = _metadata(source)
    structure = _canonical_structure(source, metadata, magnetic_moments=magnetic_moments,
                                     assume_nonmagnetic=assume_nonmagnetic)
    operations = core.build_symmetry_operations(structure, symprec=symprec)
    transports = core.build_operation_transports(structure, operations,
                                                atom_tol=max(symprec, 5e-6))
    operations, transports, magnetic_report = core.apply_magnetic_symmetry_filter(
        structure, operations, transports, mode='auto', magmom_tol=1e-5)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    report = dict(schema='tapw.hs-symmetry.v1', source_manifest_sha256=_digest(manifest),
                  operation_count=len(operations), basis_dimension=structure.nwann,
                  symprec_angstrom=symprec, cutoff=cutoff, threads=threads,
                  magnetic_symmetry=magnetic_report,
                  source_spin_mode=metadata.get('spin_mode'),
                  symmetry_spin_representation='SU2' if structure.spinful else 'scalar',
                  assume_nonmagnetic=assume_nonmagnetic,
                  raw_matrices_modified=False, source_sha256={})
    try:
        cache = core.SymmetryActionCache(transports)
        for name in ('H', 'S'):
            file = source / f'{name}.npz'
            if not file.is_file():
                file = source / f'{name}.dat'
            report['source_sha256'][name] = _digest(file)
            matrix = core.read_openmx_atom_blocks(file, structure)
            with core._blas_thread_context(1):
                blocks, diagnostics = core.symmetrize_atom_block_matrix_set(
                    matrix, structure, transports, cutoff=cutoff, action_cache=cache,
                    block_workers=threads, diagnostics='full')
            triplets = core._atom_block_sparse_triplets(structure, blocks, cutoff)
            canonical = {r: dict(zip(('row', 'col', 'val'), arrays))
                         for r, arrays in triplets.items()}
            _write_pair_matrix(stage / f'{name}_symm', canonical, structure.nwann,
                               output_format=output_format)
            report[name] = diagnostics
            if _digest(file) != report['source_sha256'][name]:
                raise ValueError('Source matrix changed during symmetry processing.')
        core.write_symmetry_transports_json(stage / 'symmetry_transports.json', structure,
            operations, transports, symprec=symprec, magnetic_report=magnetic_report)
        shutil.copy2(source / 'structure.extxyz', stage / 'structure.extxyz')
        extension = 'dat' if output_format == 'dat' else 'npz'
        fragment = {'system': {'structure': 'structure.extxyz',
                    'hamiltonian': f'H_symm.{extension}', 'overlap': f'S_symm.{extension}',
                    'orbitals': metadata['system_orbitals'], 'spin': metadata['spin']}}
        (stage / 'system.fragment.yaml').write_text(yaml.safe_dump(fragment, sort_keys=False))
        (stage / 'symmetrization_report.json').write_text(json.dumps(report, indent=2)+'\n')
        publish_directory(stage, output)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return report


def prepare(backend, source, output, *, output_format='npz', symmetrize=False,
            symprec=1e-3, cutoff=0., threads=1, magnetic_moments=None,
            assume_nonmagnetic=False, **backend_options):
    """Import one backend transactionally, optionally adding a symmetry subdirectory."""
    if backend not in BACKENDS or output_format not in FORMATS:
        raise ValueError('Unknown backend or output format.')
    output = _new_destination(output)
    module = importlib.import_module(f'tapw.io.{backend}')
    importer = getattr(module, f'import_{backend}')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{output.name}-prepare-', dir=output.parent))
    imported = temporary / 'imported'
    try:
        if backend == 'openmx':
            backend_options['threads'] = threads
        metadata = importer(source, imported, write_dat=output_format != 'npz', **backend_options)
        if symmetrize:
            symmetrize_import(imported, imported / 'symmetrized', output_format=output_format,
                symprec=symprec, cutoff=cutoff, threads=threads,
                magnetic_moments=magnetic_moments, assume_nonmagnetic=assume_nonmagnetic)
        if output_format == 'dat':
            for name in ('H', 'S'):
                (imported / f'{name}.npz').unlink()
            fragment_path = imported / 'system.fragment.yaml'
            fragment = yaml.safe_load(fragment_path.read_text())
            fragment['system'].update(hamiltonian='H.dat', overlap='S.dat')
            fragment_path.write_text(yaml.safe_dump(fragment, sort_keys=False))
        report = dict(schema='tapw.dft-preparation.v1', backend=backend,
                      output_format=output_format, symmetrized=symmetrize,
                      basis_dimension=metadata['basis_dimension'],
                      hamiltonian_unit='eV', overlap_unit='dimensionless',
                      raw_cutoff=0., symmetry_cutoff=cutoff if symmetrize else None)
        report['files_sha256'] = {str(file.relative_to(imported)): _digest(file)
                                  for file in sorted(imported.rglob('*')) if file.is_file()}
        (imported / 'preparation.json').write_text(json.dumps(report, indent=2)+'\n')
        (imported / 'README.md').write_text(
            f'{backend.upper()} H/S import: H in eV, S dimensionless. Format: {output_format}.\n'
            f'Source/basis conventions: {backend}_import.json. Checksums: preparation.json.\n'
            'system.fragment.yaml contains matrix/structure fields; add the actual moire '
            'twist_index, layers, output and workflow settings before TAPW.\n'
            + ('Separate symmetry-averaged matrices and diagnostics: symmetrized/.\n' if symmetrize else ''))
        # README is written last above; refresh its manifest checksum.
        report['files_sha256']['README.md'] = _digest(imported / 'README.md')
        (imported / 'preparation.json').write_text(json.dumps(report, indent=2)+'\n')
        publish_directory(imported, output)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return report


def main(backend, argv=None):
    parser = argparse.ArgumentParser(
        prog=f'tapw prepare-hs {backend}',
        description=f'Prepare {backend.upper()} real-space H/S for TAPW.',
    )
    input_help = {
        'openmx': 'Completed OpenMX .scfout',
        'abacus': 'Completed ABACUS case directory',
        'siesta': 'Completed SIESTA .HSX',
    }
    parser.add_argument('--input', required=True, type=Path, help=input_help[backend])
    parser.add_argument('--output', required=True, type=Path, help='New output directory')
    parser.add_argument('--format', choices=FORMATS, default='npz')
    parser.add_argument('--symmetrize', action='store_true', help='Also write symmetrized/H_symm and S_symm')
    parser.add_argument('--symprec', type=float, default=1e-3)
    parser.add_argument('--cutoff', type=float, default=0., help='Cutoff for optional symmetry outputs only')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--magnetic-moments', type=Path, help='JSON natoms-by-3 moments in source site order')
    parser.add_argument('--assume-nonmagnetic', action='store_true', help='Confirm nonmagnetic spinful symmetry')
    if backend == 'openmx':
        parser.add_argument('--structure', required=True, type=Path, help='Matching OpenMX .dat input')
        parser.add_argument('--binary', type=Path, help='Compiled reader with --no-symmetrize support')
    elif backend == 'siesta':
        parser.add_argument('--energy-reference', choices=('absolute', 'fermi'), default='absolute')
        parser.add_argument('--species', action='append', default=[], metavar='LABEL=ELEMENT')
    elif backend == 'abacus':
        parser.add_argument('--nspin', type=int, choices=(1, 2, 4))
        parser.add_argument('--step', type=int)
        parser.add_argument('--converter', type=Path, help='Override the bundled ABACUS CSR utility')
    args = parser.parse_args(argv)
    options = {}
    if backend == 'openmx':
        options.update(structure=args.structure, binary=args.binary)
    elif backend == 'siesta':
        mapping = {}
        for item in args.species:
            if item.count('=') != 1:
                parser.error('--species must be LABEL=ELEMENT')
            key, value = item.split('=')
            if not key or not value or key in mapping:
                parser.error('Species labels/elements must be nonempty and labels unique')
            mapping[key] = value
        options.update(energy_reference=args.energy_reference, species_map=mapping)
    elif backend == 'abacus':
        options = {key: value for key, value in dict(nspin=args.nspin, step=args.step,
                   converter_path=args.converter).items() if value is not None}
    try:
        moments = None if args.magnetic_moments is None else json.loads(args.magnetic_moments.read_text())
        report = prepare(backend, args.input, args.output, output_format=args.format,
            symmetrize=args.symmetrize, symprec=args.symprec, cutoff=args.cutoff,
            threads=args.threads, magnetic_moments=moments,
            assume_nonmagnetic=args.assume_nonmagnetic, **options)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(1, f'{backend} preparation: {exc}\n')
    print(f'Wrote {args.output.resolve()} ({report["basis_dimension"]} basis states; H in eV, S dimensionless)')
    return 0


def symmetry_main(argv=None):
    parser = argparse.ArgumentParser(description='Symmetrize a canonical OpenMX/ABACUS/SIESTA H/S import.')
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--format', choices=FORMATS, default='npz')
    parser.add_argument('--symprec', type=float, default=1e-3)
    parser.add_argument('--cutoff', type=float, default=0.)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--magnetic-moments', type=Path)
    parser.add_argument('--assume-nonmagnetic', action='store_true')
    args = parser.parse_args(argv)
    try:
        moments = None if args.magnetic_moments is None else json.loads(args.magnetic_moments.read_text())
        report = symmetrize_import(args.input, args.output, output_format=args.format,
            symprec=args.symprec, cutoff=args.cutoff, threads=args.threads,
            magnetic_moments=moments, assume_nonmagnetic=args.assume_nonmagnetic)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'H/S symmetry: {exc}\n')
    print(f'Wrote {args.output.resolve()} using {report["operation_count"]} symmetry operations')
    return 0

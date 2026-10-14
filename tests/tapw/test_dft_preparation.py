"""Contracts for backend-neutral preparation and optional symmetry processing."""
from pathlib import Path
import hashlib
import importlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import write

from tapw.io.hr import HrSparseHandler

ROOT = Path(__file__).resolve().parents[2]


def _imported(tmp_path, *, spin=False):
    source = tmp_path / 'source'
    source.mkdir()
    atoms = Atoms('Si2', scaled_positions=[[0, 0, 0], [.5, .5, .5]],
                  cell=np.eye(3) * 4, pbc=True)
    write(source / 'structure.extxyz', atoms)
    n = 4 if spin else 2
    for name, vals in [('H', np.tile([1., 3.], n // 2)), ('S', np.ones(n))]:
        writer = HrSparseHandler()
        writer.hr_sparse = {(0, 0, 0): {'row': np.arange(n), 'col': np.arange(n), 'val': vals}}
        writer.save_to_npz(source / f'{name}.npz', basis_dimension=n)
    meta = dict(schema='tapw.siesta-import.v1', basis_dimension=n,
                scalar_basis_dimension=2, spin=spin,
                spin_mode='spinorbit' if spin else 'unpolarized',
                system_orbitals={'Si': 's1'},
                orbital_convention='TAPW positive Cartesian real harmonics')
    (source / 'siesta_import.json').write_text(json.dumps(meta))
    (source / 'system.fragment.yaml').write_text(yaml.safe_dump({'system': {
        'structure': 'structure.extxyz', 'hamiltonian': 'H.npz', 'overlap': 'S.npz',
        'orbitals': {'Si': 's1'}, 'spin': spin}}))
    return source


def _load(path):
    reader = HrSparseHandler()
    reader.load_from_npz(path)
    return reader.hr_sparse


def test_common_symmetry_preserves_raw_and_writes_versioned_pair(tmp_path):
    tool = importlib.import_module('tapw.io.preparation')
    source = _imported(tmp_path)
    original = hashlib.sha256((source / 'H.npz').read_bytes()).hexdigest()
    output = tmp_path / 'symmetric'
    report = tool.symmetrize_import(source, output, output_format='both')
    assert hashlib.sha256((source / 'H.npz').read_bytes()).hexdigest() == original
    assert report['operation_count'] > 1
    for name in ('H', 'S'):
        assert (output / f'{name}_symm.dat').is_file()
        assert (output / f'{name}_symm.npz').is_file()
    h = _load(output / 'H_symm.npz')[(0, 0, 0)]
    np.testing.assert_allclose(h['val'], [2., 2.], atol=1e-12)
    from tapw.io.hr import read_sparse_npz_metadata
    with np.load(output / 'H_symm.npz') as payload:
        assert read_sparse_npz_metadata(payload)['basis_dimension'] == 2


def test_spinful_symmetry_requires_actual_moments_or_explicit_nonmagnetic(tmp_path):
    tool = importlib.import_module('tapw.io.preparation')
    source = _imported(tmp_path, spin=True)
    with pytest.raises(ValueError, match='magnetic|moment'):
        tool.symmetrize_import(source, tmp_path / 'unknown')
    assert not (tmp_path / 'unknown').exists()
    tool.symmetrize_import(source, tmp_path / 'known', assume_nonmagnetic=True)
    assert (tmp_path / 'known/H_symm.npz').is_file()


def test_symmetry_refuses_existing_output(tmp_path):
    tool = importlib.import_module('tapw.io.preparation')
    source = _imported(tmp_path)
    output = tmp_path / 'existing'
    output.mkdir()
    with pytest.raises(FileExistsError):
        tool.symmetrize_import(source, output)
    assert list(output.iterdir()) == []


def test_shared_symmetry_reader_retains_trailing_zero_basis_states(tmp_path):
    core = importlib.import_module('tapw.io.hs_symmetry')
    writer = HrSparseHandler()
    writer.hr_sparse = {(0, 0, 0): {'row': np.array([0]), 'col': np.array([0]), 'val': np.array([2.])}}
    writer.save_to_npz(tmp_path / 'H.npz', basis_dimension=3)
    matrix = core.read_openmx_sparse_npz(tmp_path / 'H.npz')
    assert matrix.nwann == 3
    with pytest.raises(ValueError, match='dimension'):
        core.read_openmx_sparse_npz(tmp_path / 'H.npz', nwann=2)


@pytest.mark.parametrize('backend', ['openmx', 'siesta', 'abacus'])
def test_backend_entry_point_works_outside_repo(backend, tmp_path):
    script = ROOT / 'scripts' / backend / 'prepare_hs.py'
    result = subprocess.run([sys.executable, str(script), '--help'], cwd=tmp_path,
                            env={**os.environ, 'PYTHONPATH': str(ROOT / 'tapw')},
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '--input' in result.stdout and '--output' in result.stdout
    assert '--format' in result.stdout and '--symmetrize' in result.stdout


def test_prepare_publishes_nested_symmetry_with_requested_format(tmp_path):
    sisl = pytest.importorskip('sisl')
    tool = importlib.import_module('tapw.io.preparation')
    atom = sisl.Atom(14, orbitals=[sisl.AtomicOrbital(n=3, l=0, m=0, zeta=1, R=2.)])
    geom = sisl.Geometry([[0., 0., 0.], [2., 2., 2.]], [atom, atom],
                         lattice=sisl.Lattice([4., 4., 4.]))
    h = sisl.Hamiltonian(geom, orthogonal=False)
    h[0, 0] = [1., 1.]
    h[1, 1] = [3., 1.]
    source = tmp_path / 'input.HSX'
    h.write(source, version=2)
    output = tmp_path / 'prepared'
    report = tool.prepare('siesta', source, output, output_format='dat', symmetrize=True)
    assert (output / 'H.dat').is_file()
    assert not (output / 'H.npz').exists()
    assert (output / 'symmetrized/H_symm.dat').is_file()
    assert not (output / 'symmetrized/H_symm.npz').exists()
    for relative, digest in report['files_sha256'].items():
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == digest
    config = yaml.safe_load((output / 'system.fragment.yaml').read_text())
    assert config['system']['hamiltonian'] == 'H.dat'


def test_directory_publication_rolls_back_its_nested_files_only(tmp_path, monkeypatch):
    from tapw.io.siesta import publish_directory
    stage, output = tmp_path / 'stage', tmp_path / 'result'
    (stage / 'nested').mkdir(parents=True)
    (stage / 'nested/H.npz').write_bytes(b'matrix')
    (stage / 'preparation.json').write_text('{}')
    original_link = os.link
    def fail_manifest(source, destination):
        if Path(source).name == 'preparation.json':
            raise OSError('simulated publication failure')
        original_link(source, destination)
    monkeypatch.setattr(os, 'link', fail_manifest)
    with pytest.raises(OSError, match='simulated'):
        publish_directory(stage, output)
    assert not output.exists()
    assert (stage / 'nested/H.npz').read_bytes() == b'matrix'


@pytest.mark.parametrize('moments, diagonal', [
    ([[0., 0., 1.], [0., 0., 1.]], [-1., -1., 1., 1.]),
    ([[0., 0., 1.], [0., 0., -1.]], [-1., 1., 1., -1.]),
])
def test_magnetic_symmetry_keeps_collinear_exchange_splitting(tmp_path, moments, diagonal):
    tool = importlib.import_module('tapw.io.preparation')
    source = _imported(tmp_path, spin=True)
    metadata_path = source / 'siesta_import.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['spin_mode'] = 'polarized'
    metadata_path.write_text(json.dumps(metadata))
    writer = HrSparseHandler()
    writer.hr_sparse = {(0, 0, 0): dict(row=np.arange(4), col=np.arange(4), val=np.array(diagonal))}
    writer.save_to_npz(source / 'H.npz', basis_dimension=4)
    output = tmp_path / 'magnetic'
    tool.symmetrize_import(source, output, magnetic_moments=moments)
    blocks = _load(output / 'H_symm.npz')
    from scipy.sparse import coo_matrix
    h = sum((coo_matrix((v['val'], (v['row'], v['col'])), shape=(4, 4)).toarray()
             for v in blocks.values()), np.zeros((4, 4), dtype=complex))
    np.testing.assert_allclose(h, np.diag(diagonal), atol=1e-12)


def test_common_preparation_rejects_dangling_output_symlink(tmp_path):
    tool = importlib.import_module('tapw.io.preparation')
    source = _imported(tmp_path)
    target, output = tmp_path / 'target', tmp_path / 'link'
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        tool.symmetrize_import(source, output)
    assert output.is_symlink() and not target.exists()


@pytest.mark.parametrize('backend', ['openmx', 'abacus', 'siesta'])
def test_installed_cli_exposes_common_backend_options(backend, capsys):
    from tapw import cli
    with pytest.raises(SystemExit) as exc:
        cli.main(['prepare-hs', backend, '--help'])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert f'usage: tapw prepare-hs {backend}' in output
    assert '--input' in output and '--symmetrize' in output
    assert {
        'openmx': 'Completed OpenMX .scfout',
        'abacus': 'Completed ABACUS case directory',
        'siesta': 'Completed SIESTA .HSX',
    }[backend] in output


def test_symmetrized_pair_applies_source_to_tapw_permutation(tmp_path):
    """Canonical averaging preserves source order even when dimension metadata exists."""
    from scipy.sparse import csr_matrix
    from tapw.io.preparation import _write_pair_matrix
    blocks = {(0, 0, 0): {'row': np.array([0, 1]), 'col': np.array([0, 1]),
                          'val': np.array([2., 7.])}}
    _write_pair_matrix(tmp_path / 'H_symm', blocks, 2)
    permutation = csr_matrix([[0., 1.], [1., 0.]])
    loaded = HrSparseHandler(file_name='', npz_file_name=str(tmp_path / 'H_symm.npz'),
                              A=permutation, read_from_npz=True).get_hr_sparse()[(0, 0, 0)]
    matrix = csr_matrix((loaded['val'], (loaded['row'], loaded['col'])), shape=(2, 2)).toarray()
    np.testing.assert_allclose(matrix, np.diag([7., 2.]), atol=0, rtol=0)

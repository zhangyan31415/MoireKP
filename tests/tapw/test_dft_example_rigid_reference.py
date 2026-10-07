"""The MoTe2 TAPW example uses the paired rigid reference for projection."""
from pathlib import Path
import subprocess
import sys
import json

import numpy as np
from ase.io import read


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / 'examples/dft_backends/mote2_9.43'


def test_paired_rigid_reference_keeps_site_order_but_differs_from_relaxed():
    relaxed = read(EXAMPLE / 'openmx/prepared/structure.extxyz')
    rigid = read(EXAMPLE / 'rigid/structure.extxyz')
    assert len(rigid) == len(relaxed) == 222
    assert rigid.get_chemical_symbols() == relaxed.get_chemical_symbols()
    assert np.max(np.abs(rigid.positions - relaxed.positions)) > 0.5
    assert np.max(np.abs(rigid.cell.array - relaxed.cell.array)) > 0.2


def test_tapw_runner_exposes_explicit_reference_structure():
    script = ROOT / 'examples/dft_backends/run_tapw.py'
    result = subprocess.run([sys.executable, str(script), '--help'],
                            capture_output=True, text=True, check=True)
    assert '--reference-structure' in result.stdout


def test_tapw_runner_refuses_unmarked_source_order_npz_before_creating_output(tmp_path):
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    (prepared / 'openmx_import.json').write_text(json.dumps({'source_fermi_ev': 0.0}))
    (prepared / 'system.fragment.yaml').write_text(
        'system:\n  structure: structure.extxyz\n  hamiltonian: H.npz\n'
        '  overlap: S.npz\n  orbitals: {Mo: s1}\n  spin: false\n')
    for name in ('H', 'S'):
        np.savez(prepared / f'{name}.npz', **{
            '(0, 0, 0)_row': np.asarray([0]),
            '(0, 0, 0)_col': np.asarray([0]),
            '(0, 0, 0)_val': np.asarray([1.]),
        })
    output = tmp_path / 'must_not_exist'
    result = subprocess.run(
        [sys.executable, str(ROOT / 'examples/dft_backends/run_tapw.py'), 'openmx',
         '--prepared-dir', str(prepared), '--output-dir', str(output)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'basis_order' in result.stderr + result.stdout
    assert not output.exists()

"""Regression tests for lossless OpenMX preparation."""
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def test_native_raw_export_preserves_asymmetry(tmp_path):
    source = ROOT / 'scripts/openmx/openmx_analysis_symm_hs.cpp'
    if not source.exists():
        source = ROOT / 'scripts/openmx_analysis_symm_hs.cpp'
    binary = tmp_path / 'reader'
    subprocess.run(['g++', '-std=c++17', '-O1', str(source), '-o', str(binary)], check=True)
    identity = [[[1., 0.]]]
    symmetry = {'schema':'openmx_hs_symmetry_transports/v1', 'spinful':False,
                'natoms':2, 'nwann':2, 'nwann_spinless':2,
                'atom_orbital_counts':[1,1], 'atom_offsets':[0,1],
                'operations':[{'index':0,'antiunitary':False,'rotation_frac':np.eye(3).tolist(),
                               'atom_mappings':[0,1], 'image_shifts':[[0,0,0],[0,0,0]],
                               'orbital_blocks':{'0':identity,'1':identity}}]}
    (tmp_path/'sym.json').write_text(json.dumps(symmetry))
    def block(i,j,value):
        return {'r':[0,0,0], 'atom_i':i,'atom_j':j,'rows':[0],'cols':[0],'values':[[value,0.]]}
    (tmp_path/'mock.json').write_text(json.dumps({'spinful':False,'atom_orbital_counts':[1,1],
        'h_blocks':[block(0,1,0.200000000123),block(1,0,0.600000000789)],'s_blocks':[block(0,0,1),block(1,1,1)]}))
    for mode in ('raw','sym'):
        command=[str(binary),'--mock-input',str(tmp_path/'mock.json'),'--symmetry-json',str(tmp_path/'sym.json'),
                 '--output-dir',str(tmp_path/mode),'--output-format','both','--cutoff','0']
        if mode == 'raw': command.append('--no-symmetrize')
        run=subprocess.run(command,capture_output=True,text=True)
        assert run.returncode == 0,run.stderr
    with np.load(tmp_path/'raw/H.npz') as raw, np.load(tmp_path/'sym/H.npz') as sym:
        np.testing.assert_allclose(raw['(0, 0, 0)_val'],[.200000000123,.600000000789],rtol=0,atol=1e-16)
        np.testing.assert_allclose(sym['(0, 0, 0)_val'],[.400000000456,.400000000456])

    np.testing.assert_allclose(np.loadtxt(tmp_path/'raw/H.dat',skiprows=4)[:,5], [.200000000123,.600000000789], rtol=0,atol=1e-16)


def test_geometry_validation_rejects_reordered_sites():
    from types import SimpleNamespace
    import pytest
    from tapw.io.openmx import _validate_source_geometry
    structure=SimpleNamespace(lattice=np.eye(3), atoms=[SimpleNamespace(cart=np.array(p)) for p in [[0,0,0],[.2,0,0]]], spin_mode='none')
    metadata={'cell_angstrom':np.eye(3).tolist(),'positions_angstrom':[[.2,0,0],[0,0,0]],'spin_switch':0,'source_fermi_ev':1.}
    with pytest.raises(ValueError,match='positions'):
        _validate_source_geometry(structure,metadata)
    metadata['positions_angstrom'].reverse()
    _validate_source_geometry(structure,metadata)


def test_import_refuses_existing_output(tmp_path):
    import pytest
    from tapw.io.openmx import import_openmx
    output=tmp_path/'output';output.mkdir()
    (output/'precious').write_text('keep')
    with pytest.raises(FileExistsError):
        import_openmx(tmp_path/'absent.scfout',output,structure=tmp_path/'absent.dat')
    assert (output/'precious').read_text()=='keep'


def _write_distinct_species_input(path, basis_b='Mo9.0-s1p1d1', pseudo_b='Mo_PBE19'):
    path.write_text(f'''scf.SpinPolarization off
Atoms.UnitVectors.Unit Ang
<Atoms.UnitVectors
4 0 0
0 4 0
0 0 4
Atoms.UnitVectors>
Species.Number 2
<Definition.of.Atomic.Species
MoA Mo7.0-s1p1d1 Mo_PBE19
MoB {basis_b} {pseudo_b}
Definition.of.Atomic.Species>
Atoms.Number 2
Atoms.SpeciesAndCoordinates.Unit Ang
<Atoms.SpeciesAndCoordinates
1 MoA 0 0 0 7 7
2 MoB 2 2 2 7 7
Atoms.SpeciesAndCoordinates>
''')


def test_import_rejects_same_element_with_distinct_radial_basis(tmp_path):
    import pytest
    from tapw.io.openmx import import_openmx
    structure=tmp_path/'openmx.dat'; source=tmp_path/'openmx.scfout'
    source.write_bytes(b'test input must be rejected before native reading')
    _write_distinct_species_input(structure)
    with pytest.raises(ValueError,match='Different.*PAO.*Mo'):
        import_openmx(source,tmp_path/'output',structure=structure,binary='/bin/true')
    assert not (tmp_path/'output').exists()


def test_import_rejects_same_element_with_distinct_pseudopotentials(tmp_path):
    import pytest
    from tapw.io.openmx import import_openmx
    structure=tmp_path/'openmx.dat'; source=tmp_path/'openmx.scfout'
    source.write_bytes(b'test input must be rejected before native reading')
    _write_distinct_species_input(structure,basis_b='Mo7.0-s1p1d1',pseudo_b='Mo_PBE19_alt')
    with pytest.raises(ValueError,match='Different.*pseudopotential.*Mo'):
        import_openmx(source,tmp_path/'output',structure=structure,binary='/bin/true')
    assert not (tmp_path/'output').exists()


def test_native_reader_scratch_files_are_confined_to_private_stage(tmp_path, monkeypatch):
    import pytest
    from tapw.io.openmx import import_openmx
    structure, source = tmp_path/'openmx.dat', tmp_path/'openmx.scfout'
    _write_distinct_species_input(structure, basis_b='Mo7.0-s1p1d1')
    source.write_bytes(b'fixture')
    binary = tmp_path/'reader'
    binary.write_text('#!/bin/sh\nprintf temporary > temporal_12345.input\nexit 7\n')
    binary.chmod(0o755)
    caller = tmp_path/'caller'; caller.mkdir()
    monkeypatch.chdir(caller)
    with pytest.raises(RuntimeError, match='exited 7'):
        import_openmx(source, tmp_path/'result', structure=structure, binary=binary)
    assert list(caller.iterdir()) == []
    assert not (tmp_path/'result').exists()


def test_reader_build_uses_matching_openmx_length_constant(tmp_path):
    """The reader must invert the writer's conversion, even for legacy BohrR."""
    import os
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'read_scfout.o').touch()
    (source / 'openmx_common.h').write_text('#define BohrR 0.529177249\n')
    compiler = tmp_path / 'compiler'
    compiler.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
    compiler.chmod(0o755)
    capture = tmp_path / 'args'
    env = dict(os.environ, CXX=str(compiler), MKLROOT=str(tmp_path), CAPTURE=str(capture))
    subprocess.run(['bash', str(ROOT / 'scripts/openmx/build_openmx_symm_hs.sh'),
                    str(source), str(tmp_path / 'build')], check=True, env=env,
                   capture_output=True, text=True)
    assert '-DOPENMX_BOHR_TO_ANG=0.529177249' in capture.read_text().splitlines()
    (source / 'openmx_common.h').write_text('/* unknown conversion */\n')
    result = subprocess.run(['bash', str(ROOT / 'scripts/openmx/build_openmx_symm_hs.sh'),
                             str(source), str(tmp_path / 'build')], env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Cannot determine BohrR' in result.stderr

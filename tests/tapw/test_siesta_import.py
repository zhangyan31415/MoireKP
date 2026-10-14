"""SIESTA source-basis and real-space conversion contracts (small tests)."""
import importlib
import numpy as np
import pytest
from scipy import sparse


def importer():
    return importlib.import_module('tapw.io.siesta')


def shell(l, n=2, zeta=1):
    return [dict(n=n, l=l, m=m, zeta=zeta) for m in range(-l, l+1)]


def test_siesta_p_shell_maps_to_positive_cartesian_tapw_order():
    layout = importer().orbital_layout(shell(1))
    assert layout.permutation.tolist() == [2, 0, 1]
    assert layout.phases.tolist() == [-1, -1, 1]
    assert layout.specification == 'p1'


def test_radial_shells_are_counted_and_not_confused_with_principal_n():
    source = shell(1, n=3, zeta=2) + shell(0, n=3) + shell(1, n=3, zeta=1)
    layout = importer().orbital_layout(source)
    assert layout.permutation.tolist() == [3, 6, 4, 5, 2, 0, 1]
    assert layout.specification == 's1p2'
    assert sorted(layout.permutation.tolist()) == list(range(7))


def test_siesta_d_and_f_orders_and_phases():
    d = importer().orbital_layout(shell(2, n=3))
    assert d.permutation.tolist() == [2, 4, 0, 3, 1]
    assert d.phases.tolist() == [1, 1, 1, -1, -1]
    f = importer().orbital_layout(shell(3, n=4))
    assert f.permutation.tolist() == [3, 4, 2, 5, 1, 6, 0]
    assert f.phases.tolist() == [1, -1, -1, 1, 1, -1, -1]


@pytest.mark.parametrize('orbitals', [shell(1)[:-1], shell(1)+shell(1), shell(4,n=5), [dict(n=2,l=1,m=1,zeta=1)]])
def test_incomplete_duplicate_or_unsupported_shells_fail(orbitals):
    with pytest.raises(ValueError):
        importer().orbital_layout(orbitals)


def test_realspace_conversion_keeps_actual_supercell_order_and_complex_phase():
    # Nonlexical cell order: zero, positive, negative. Wrong indexing passes at Gamma.
    offsets = np.array([[0,0,0],[1,0,0],[-1,0,0]])
    h0 = np.array([[1, 2+3j],[2-3j,4]])
    hp = np.array([[.2j, .7+1j],[.3, -.6j]])
    source = sparse.csr_matrix(np.hstack([h0, hp, hp.conj().T]))
    perm = np.array([1,0]);phases = np.array([-1,1])
    blocks = importer().realspace_blocks(source, offsets, perm, phases)
    result = sum(sparse.coo_matrix((v['val'],(v['row'],v['col'])),shape=(2,2)).toarray()*np.exp(2j*np.pi*.19*r[0]) for r,v in blocks.items())
    original = h0 + hp*np.exp(2j*np.pi*.19)+hp.conj().T*np.exp(-2j*np.pi*.19)
    expected = phases[:,None]*original[np.ix_(perm,perm)]*phases[None,:]
    np.testing.assert_allclose(result,expected,atol=1e-13)
    np.testing.assert_allclose(result,result.conj().T,atol=1e-13)


def test_zero_edge_orbitals_retain_declared_dimension(tmp_path):
    mod=importer()
    matrix=sparse.csr_matrix(([3.],([0],[0])),shape=(3,3))
    blocks=mod.realspace_blocks(matrix,np.array([[0,0,0]]),np.arange(3),np.ones(3))
    from tapw.io.hr import HrSparseHandler, SPARSE_NPZ_METADATA_KEY
    h=HrSparseHandler();h.hr_sparse=blocks;h.save_to_npz(tmp_path/'H.npz',basis_dimension=3)
    import json
    with np.load(tmp_path/'H.npz',allow_pickle=False) as archive:
        assert json.loads(str(archive[SPARSE_NPZ_METADATA_KEY].item()))['basis_dimension']==3


def test_nonfinite_matrix_is_rejected():
    with pytest.raises(ValueError,match='finite'):
        importer().realspace_blocks(sparse.csr_matrix([[np.nan]]),np.array([[0,0,0]]),np.array([0]),np.array([1]))


def test_invalid_supercell_columns_are_rejected():
    with pytest.raises(ValueError,match='shape'):
        importer().realspace_blocks(sparse.csr_matrix((2,5)),np.array([[0,0,0],[1,0,0]]),np.arange(2),np.ones(2))


def test_cli_exposes_siesta_import_help_without_running_calculations(capsys):
    from tapw import cli
    with pytest.raises(SystemExit) as exc:
        cli.main(['import-siesta','--help'])
    assert exc.value.code==0
    text=capsys.readouterr().out
    assert 'HSX' in text and '--output' in text


def test_duplicate_sum_overflow_is_rejected():
    matrix=sparse.coo_matrix(([1e308,1e308],([0,0],[0,0])),shape=(1,1))
    with pytest.raises(ValueError,match='finite'):
        importer().realspace_blocks(matrix,np.array([[0,0,0]]),np.array([0]),np.array([1]))


def test_out_of_range_translation_is_rejected():
    with pytest.raises(ValueError,match='range|integer'):
        importer().realspace_blocks(sparse.eye(1),np.array([[1e30,0,0]]),np.array([0]),np.array([1]))


def test_noninteger_permutation_is_rejected():
    with pytest.raises(ValueError,match='permutation'):
        importer().realspace_blocks(sparse.eye(2),np.array([[0,0,0]]),np.array([0.,1.]),np.array([1,1]))


def test_publish_does_not_replace_an_existing_empty_directory(tmp_path):
    stage=tmp_path/'stage';stage.mkdir();(stage/'H.npz').write_bytes(b'new')
    dest=tmp_path/'destination';dest.mkdir()
    with pytest.raises(FileExistsError):
        importer().publish_directory(stage,dest)
    assert dest.is_dir() and list(dest.iterdir())==[]
    assert (stage/'H.npz').read_bytes()==b'new'


def test_publish_creates_new_directory_and_keeps_manifest_last(tmp_path):
    stage=tmp_path/'stage';stage.mkdir();(stage/'H.npz').write_bytes(b'H')
    (stage/'siesta_import.json').write_text('{"schema":"test"}')
    dest=tmp_path/'destination'
    importer().publish_directory(stage,dest)
    assert (dest/'H.npz').read_bytes()==b'H'
    assert (dest/'siesta_import.json').is_file()

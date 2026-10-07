"""Independent ABACUS basis, spin and file import contracts."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.special import lpmv, factorial


def importer():
    return importlib.import_module('tapw.io.abacus')


def test_abacus_real_harmonics_become_positive_cartesian():
    # Associated Legendre functions contain Condon-Shortley signs. Evaluate
    # ABACUS's 0,+1,-1,... basis independently at a generic direction.
    x, y, z = np.array([.2, .5, .8]) / np.linalg.norm([.2, .5, .8])
    phi = np.arctan2(y, x)
    source = []
    for l in range(4):
        c = np.sqrt((2*l+1)/(4*np.pi))
        source.append(c*lpmv(0,l,z))
        for m in range(1,l+1):
            value = c*np.sqrt(2*factorial(l-m)/factorial(l+m))*lpmv(m,l,z)
            source.extend([value*np.cos(m*phi), value*np.sin(m*phi)])
    expected = [1/np.sqrt(4*np.pi), * (np.sqrt(3/(4*np.pi))*np.array([x,y,z])),
                np.sqrt(5/(16*np.pi))*(3*z*z-1), np.sqrt(15/(16*np.pi))*(x*x-y*y),
                np.sqrt(15/(4*np.pi))*x*y, np.sqrt(15/(4*np.pi))*x*z, np.sqrt(15/(4*np.pi))*y*z,
                np.sqrt(7/(16*np.pi))*z*(5*z*z-3),
                np.sqrt(21/(32*np.pi))*x*(5*z*z-1), np.sqrt(21/(32*np.pi))*y*(5*z*z-1),
                np.sqrt(105/(16*np.pi))*z*(x*x-y*y), np.sqrt(105/(4*np.pi))*x*y*z,
                np.sqrt(35/(32*np.pi))*x*(x*x-3*y*y), np.sqrt(35/(32*np.pi))*y*(3*x*x-y*y)]
    layout = importer().orbital_layout([1,1,1,1])
    np.testing.assert_allclose(np.asarray(source)[layout.permutation]*layout.phases, expected, atol=1e-14)
    assert layout.specification == 's1p1d1f1'


def _csr(path, matrix, kind):
    mat = sparse.csr_matrix(matrix)
    values = ' '.join(f'({v.real:.16g},{v.imag:.16g})' if np.iscomplexobj(matrix) else f'{v:.16g}' for v in mat.data)
    path.write_text(f'STEP: 0\nMatrix Dimension of {kind}(R): {mat.shape[0]}\nMatrix number of {kind}(R): 1\n0 0 0 {mat.nnz}\n{values}\n'+
                    ' '.join(map(str,mat.indices))+'\n'+' '.join(map(str,mat.indptr))+'\n')


def case(tmp_path, nspin=1):
    root = tmp_path/'case'; root.mkdir(); out=root/'OUT.test'; out.mkdir()
    (root/'INPUT').write_text(f'INPUT_PARAMETERS\nsuffix test\ncalculation scf\nbasis_type lcao\nnspin {nspin}\n')
    (root/'STRU').write_text('ATOMIC_SPECIES\nSi 28 Si.upf\nNUMERICAL_ORBITAL\nSi.orb\nLATTICE_CONSTANT\n10\nLATTICE_VECTORS\n1 0 0\n0 1 0\n0 0 1\nATOMIC_POSITIONS\nDirect\nSi\n0\n1\n.1 .2 .3 1 1 1\n')
    (root/'Si.orb').write_text('Element Si\nLmax 1\nNumber of Sorbital--> 1\nNumber of Porbital--> 1\nSUMMARY END\n')
    h=np.array([[1.,2,3,4],[2,5,6,7],[3,6,8,9],[4,7,9,10]])
    s=np.eye(4)
    if nspin==4:
        h=np.kron(h,np.eye(2)).astype(complex); h[0,1]=1+2j; h[1,0]=1-2j
        s=np.eye(8).astype(complex)
    _csr(out/'data-HR-sparse_SPIN0.csr',h,'H')
    _csr(out/'data-SR-sparse_SPIN0.csr',s,'S')
    if nspin==2: _csr(out/'data-HR-sparse_SPIN1.csr',h+np.eye(4),'H')
    return root,h


def dense(path):
    from tapw.io.hr import HrSparseHandler
    reader=HrSparseHandler(); reader.load_from_npz(path)
    v=reader.hr_sparse[(0,0,0)]
    return sparse.coo_matrix((v['val'],(v['row'],v['col'])),shape=(reader.basis_dimension,reader.basis_dimension)).toarray()


@pytest.mark.parametrize('nspin',[1,2,4])
def test_real_import_basis_spin_units_structure_and_dat(tmp_path,nspin):
    from ase.io import read
    root,h=case(tmp_path,nspin)
    result=tmp_path/'result'
    meta=importer().import_abacus(root,result,write_dat=True)
    p=np.array([0,2,3,1]); signs=np.array([1,-1,-1,1])
    if nspin==2:
        h=sparse.block_diag([h,h+np.eye(4)]).toarray(); p=np.r_[p,p+4]; signs=np.tile(signs,2)
    elif nspin==4:
        p=np.r_[2*p,2*p+1]; signs=np.tile(signs,2)
    expected=h[np.ix_(p,p)]*signs[:,None]*signs[None,:]*13.605693122994
    np.testing.assert_allclose(dense(result/'H.npz'),expected)
    np.testing.assert_allclose(dense(result/'S.npz'),np.eye(len(p)))
    assert meta['basis_dimension']==len(p)
    assert meta['system_orbitals']=={'Si':'s1p1'}
    assert meta['spin']==(nspin!=1)
    atoms=read(result/'structure.extxyz')
    np.testing.assert_allclose(atoms.get_scaled_positions()[0],[.1,.2,.3])
    assert (result/'H.dat').is_file() and (result/'S.dat').is_file()
    assert json.loads((result/'abacus_import.json').read_text())['source_nspin']==nspin
    with pytest.raises(FileExistsError): importer().import_abacus(root,result)


def test_missing_or_mismatched_basis_rejected_without_output(tmp_path):
    root,_=case(tmp_path)
    (root/'Si.orb').write_text('Element Si\nLmax 0\nNumber of Sorbital--> 1\nSUMMARY END\n')
    with pytest.raises(ValueError,match='dimension'):
        importer().import_abacus(root,tmp_path/'result')
    assert not (tmp_path/'result').exists()
    (root/'Si.orb').unlink()
    with pytest.raises(FileNotFoundError): importer().import_abacus(root,tmp_path/'result')


def test_relaxation_cannot_silently_use_initial_structure(tmp_path):
    root,_=case(tmp_path)
    (root/'INPUT').write_text('INPUT_PARAMETERS\nsuffix test\ncalculation relax\nnspin 1\n')
    with pytest.raises(ValueError,match='scf|nscf'):
        importer().import_abacus(root,tmp_path/'result')


@pytest.mark.parametrize('counts',[[0,0,0,0],[1,-1],[1,1,1,1,1],[True,1],[1,1.2]])
def test_invalid_shell_counts_rejected(counts):
    with pytest.raises(ValueError): importer().orbital_layout(counts)


@pytest.mark.parametrize('nspin',[1,2,4])
def test_h_and_s_may_have_different_translation_support(tmp_path,nspin):
    root,h=case(tmp_path,nspin)
    path=root/'OUT.test/data-HR-sparse_SPIN0.csr'
    text=path.read_text().replace('Matrix number of H(R): 1','Matrix number of H(R): 3')
    for r,value in [(1,.125),(-1,.125)]:
        # A single diagonal hopping on orbital zero and its Hermitian partner;
        # all other H channels and S retain only R=0.
        token=f'({value},0)' if nspin==4 else str(value)
        text+=f'{r} 0 0 1\n{token}\n0\n0 '+ ' '.join(['1']*len(h))+'\n'
    path.write_text(text)
    out=tmp_path/'result'
    importer().import_abacus(root,out)
    from tapw.io.hr import HrSparseHandler
    reader=HrSparseHandler();reader.load_from_npz(out/'H.npz')
    assert set(reader.hr_sparse)=={(0,0,0),(-1,0,0),(1,0,0)}
    np.testing.assert_allclose(reader.hr_sparse[(1,0,0)]['val'],[.125*13.605693122994])
    reader.load_from_npz(out/'S.npz')
    assert set(reader.hr_sparse)=={(0,0,0)}


def test_running_log_basis_mismatch_rejected(tmp_path):
    root,_=case(tmp_path)
    (root/'OUT.test/running_scf.log').write_text('READING ATOM TYPE 1\nAtom label = Si\nL=0, number of zeta = 4\nNumber of atoms for this type = 1\nTOTAL ATOM NUMBER = 1\nNumber of basis (NLOCAL) = 4\n')
    with pytest.raises(ValueError,match='log.*basis|basis.*log'):
        importer().import_abacus(root,tmp_path/'result')


def test_running_log_valid_basis_accepted(tmp_path):
    root,_=case(tmp_path)
    (root/'OUT.test/running_scf.log').write_text('READING ATOM TYPE 1\nAtom label = Si\nL=0, number of zeta = 1\nL=1, number of zeta = 1\nNumber of atoms for this type = 1\nTOTAL ATOM NUMBER = 1\nNumber of basis (NLOCAL) = 4\n')
    meta=importer().import_abacus(root,tmp_path/'result')
    assert meta['basis_dimension']==4


def test_csr_precision_and_tiny_entries_survive_intermediate_dat(tmp_path):
    root,h=case(tmp_path)
    h[0,1]=h[1,0]=1.234567891234e-10
    s=np.eye(4);s[0,1]=s[1,0]=2.819298306181e-6
    _csr(root/'OUT.test/data-HR-sparse_SPIN0.csr',h,'H')
    _csr(root/'OUT.test/data-SR-sparse_SPIN0.csr',s,'S')
    out=tmp_path/'result';importer().import_abacus(root,out)
    assert dense(out/'H.npz')[0,3]==pytest.approx(h[0,1]*13.605693122994,rel=1e-14,abs=0)
    assert dense(out/'S.npz')[0,3]==pytest.approx(s[0,1],rel=1e-14,abs=0)


def test_latest_log_fermi_energy_recorded_without_shifting_h(tmp_path):
    root,h=case(tmp_path)
    (root/'OUT.test/running_scf.log').write_text('READING ATOM TYPE 1\nAtom label = Si\nL=0, number of zeta = 1\nL=1, number of zeta = 1\nNumber of atoms for this type = 1\nTOTAL ATOM NUMBER = 1\nE_Fermi 0.5 6.802846561497\nE_Fermi 1.0 13.605693122994\n')
    out=tmp_path/'result';meta=importer().import_abacus(root,out)
    assert meta['source_fermi_ev']==pytest.approx(13.605693122994)
    assert dense(out/'H.npz')[0,0]==pytest.approx(h[0,0]*13.605693122994)
    meta=importer().import_abacus(root,tmp_path/'step_result',step=0)
    assert meta['source_fermi_ev'] is None


def test_native_abacus_legacy_ev_factor_is_not_misidentified_as_bad_fermi(tmp_path):
    root,_=case(tmp_path)
    (root/'OUT.test/running_scf.log').write_text('READING ATOM TYPE 1\nAtom label = Si\nL=0, number of zeta = 1\nL=1, number of zeta = 1\nNumber of atoms for this type = 1\nTOTAL ATOM NUMBER = 1\nE_Fermi 1.1959246462 16.2713895668\n')
    meta=importer().import_abacus(root,tmp_path/'result')
    assert meta['source_fermi_ev']==pytest.approx(1.1959246462*meta['ry_to_ev'])
    assert meta['source_fermi_reported_ev']==16.2713895668


@pytest.mark.parametrize('x',[1.,-.2,1.2,-1.,-1e-13])
@pytest.mark.parametrize('nspin',[1,2,4])
def test_wrapped_abacus_sites_preserve_original_orbital_gauge_at_non_gamma(tmp_path,x,nspin):
    from ase.io import read
    from tapw.io.hr import HrSparseHandler
    root,_=case(tmp_path,nspin)
    (root/'STRU').write_text(f'ATOMIC_SPECIES\nSi 28 Si.upf\nNUMERICAL_ORBITAL\nSi.orb\nLATTICE_CONSTANT\n10\nLATTICE_VECTORS\n1 0 0\n0 1 0\n0 0 1\nATOMIC_POSITIONS\nDirect\nSi\n0\n2\n.1 .2 .3\n{x} .2 .3\n')
    (root/'Si.orb').write_text('Element Si\nLmax 0\nNumber of Sorbital--> 1\nSUMMARY END\n')
    h=np.array([[1.,.2],[.2,2.]])
    s=np.array([[1.,.05],[.05,1.]])
    hsource=h;ssource=s
    if nspin==4:
        hsource=np.kron(h,np.eye(2)).astype(complex)
        hsource[0,1]=.03j;hsource[1,0]=-.03j
        ssource=np.kron(s,np.eye(2)).astype(complex)
    _csr(root/'OUT.test/data-HR-sparse_SPIN0.csr',hsource,'H')
    _csr(root/'OUT.test/data-SR-sparse_SPIN0.csr',ssource,'S')
    if nspin==2:
        _csr(root/'OUT.test/data-HR-sparse_SPIN1.csr',h+.1*np.eye(2),'H')
        hsource=sparse.block_diag([h,h+.1*np.eye(2)]).toarray()
        ssource=sparse.block_diag([s,s]).toarray()
    elif nspin==4:
        permutation=[0,2,1,3]
        hsource=hsource[np.ix_(permutation,permutation)]
        ssource=ssource[np.ix_(permutation,permutation)]
    original=np.array([[.1,.2,.3],[x,.2,.3]])
    # Independent source convention, copied from ABACUS read_stru.cpp equation.
    wrapped=np.fmod(original+10000.,1.)
    n=np.rint(original-wrapped).astype(int)
    output=tmp_path/'result';meta=importer().import_abacus(root,output)
    np.testing.assert_allclose(read(output/'structure.extxyz').get_scaled_positions(wrap=False),original,atol=1e-12)
    site_ids=np.tile([0,1],2 if nspin!=1 else 1)
    tau_orig=original[site_ids];tau_source=wrapped[site_ids]
    k=np.array([.173,.137,-.081])
    for name,source in [('H',hsource*meta['ry_to_ev']),('S',ssource)]:
        reader=HrSparseHandler();reader.load_from_npz(output/(name+'.npz'))
        result=np.zeros_like(source,dtype=complex)
        for r,b in reader.hr_sparse.items():
            row,col=b['row'],b['col']
            # Include the actual orbital positions on BOTH sides. Preserving
            # only Gamma eigenvalues would not catch an incorrect R correction.
            displacements=np.asarray(r)+tau_orig[col]-tau_orig[row]
            np.add.at(result,(row,col),b['val']*np.exp(2j*np.pi*(displacements@k)))
        expected=source*np.exp(2j*np.pi*((tau_source[None,:,:]-tau_source[:,None,:])@k))
        np.testing.assert_allclose(result,expected,atol=1e-10,rtol=1e-12)
    assert [site['source_wrap_shift'] for site in meta['sites']]==n.tolist()


def test_cartesian_angstrom_boundary_uses_native_abacus_bohr_constant(tmp_path):
    root,_=case(tmp_path)
    # 5.291770 A is exactly ABACUS's cell edge for lat0=10 Bohr,
    # although ASE's modern Bohr conversion makes it slightly below the edge.
    (root/'STRU').write_text('ATOMIC_SPECIES\nSi 28 Si.upf\nNUMERICAL_ORBITAL\nSi.orb\nLATTICE_CONSTANT\n10\nLATTICE_VECTORS\n1 0 0\n0 1 0\n0 0 1\nATOMIC_POSITIONS\nCartesian_angstrom\nSi\n0\n1\n5.291770 0 0\n')
    meta=importer().import_abacus(root,tmp_path/'result')
    assert meta['sites'][0]['source_wrap_shift']==[1,0,0]


def test_canonical_hs_npz_marks_source_atom_basis_order_for_tapw(tmp_path):
    """Angular canonicalization still leaves source atom order, needing TAPW A."""
    from tapw.io.hr import read_sparse_npz_metadata
    root,_=case(tmp_path)
    output=tmp_path/'marked_import'
    importer().import_abacus(root,output,write_dat=True)
    for name in ('H','S'):
        with np.load(output/(name+'.npz'),allow_pickle=False) as payload:
            metadata=read_sparse_npz_metadata(payload)
            assert metadata.get('basis_order')=='source', f'{name} must request TAPW atom-type permutation'

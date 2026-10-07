"""ABACUS LCAO H/S -> TAPW positive Cartesian orbitals.

The separately licensed CSR converter is invoked as an external process. This
module handles the documented STRU/orbital metadata and the canonical basis
transformation; it does not import or embed that GPL converter.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from .hr import HrSparseHandler
from .siesta import _check_hermitian, _source_hash, publish_directory

_ANGULAR = {0: (0,), 1: (1, 2, 0), 2: (0, 3, 4, 1, 2), 3: tuple(range(7))}
_NAMES = {0: ('s',), 1: ('px','py','pz'), 2: ('dz2','dx2-y2','dxy','dxz','dyz'),
          3: ('fz3','fxz2','fyz2','fz(x2-y2)','fxyz','fx(x2-3y2)','fy(3x2-y2)')}
# Use ABACUS's own geometry convention, not ASE's slightly different modern
# Bohr value. Otherwise Cartesian_angstrom wrapping can change at boundaries.
_ABACUS_BOHR_TO_ANGSTROM = 0.5291770


@dataclass(frozen=True)
class OrbitalLayout:
    permutation: np.ndarray
    phases: np.ndarray
    specification: str


def orbital_layout(counts):
    """Map ABACUS l,zeta,(0,+1,-1,...) order to positive Cartesian s/p/d/f."""
    counts = list(counts)
    if (not counts or len(counts)>4 or any(isinstance(n,(bool,np.bool_)) or
            not isinstance(n,(int,np.integer)) or n<0 for n in counts) or not any(counts)):
        raise ValueError('Nonnegative integer s/p/d/f shell counts with a nonempty basis are required.')
    permutation, phases = [], []
    offset = 0
    for l, count in enumerate(counts):
        for _ in range(count):
            for m_index in _ANGULAR[l]:
                permutation.append(offset+m_index)
                phases.append((-1)**((m_index+1)//2))
            offset += 2*l+1
    spec=''.join(f'{label}{count}' for label,count in zip('spdf',counts) if count)
    return OrbitalLayout(np.array(permutation,dtype=np.int64),np.array(phases,dtype=float),spec)


def _lines(path):
    return [line for raw in path.read_text().splitlines() if (line:=re.split(r'#|//',raw,1)[0].strip())]


def _parameters(path):
    values={}
    for line in _lines(path):
        parts=line.split()
        if parts[0].upper()=='INPUT_PARAMETERS': continue
        key=parts[0].lower()
        if key in values: raise ValueError(f'Duplicate INPUT parameter {key}.')
        if len(parts)<2: raise ValueError(f'INPUT parameter {key} has no value.')
        values[key]=parts[1]
    if values.get('calculation','scf').lower() not in ('scf','nscf'):
        raise ValueError('Only fixed-geometry scf/nscf imports are supported; initial STRU is unsafe for relax/MD matrices.')
    if values.get('basis_type','lcao').lower()!='lcao':
        raise ValueError('ABACUS import requires an LCAO basis.')
    return values


def _orbital_counts(path, element):
    text=path.read_text().split('SUMMARY',1)[0]
    match=re.search(r'^\s*Element\s+(\S+)',text,re.M)
    if not match or match.group(1)!=element:
        raise ValueError(f'{path}: orbital Element must match STRU element {element}.')
    lmax=re.search(r'^\s*Lmax\s+(\d+)\s*$',text,re.M)
    if not lmax or int(lmax.group(1))>3:
        raise ValueError(f'{path}: explicit Lmax <= 3 required; only s/p/d/f are supported.')
    entries=re.findall(r'Number\s+of\s+([SPDFGHI])orbital\s*-->\s*(\d+)',text,re.I)
    counts={}
    for label,count in entries:
        label=label.lower()
        if label in counts: raise ValueError(f'{path}: duplicate {label} shell count.')
        counts[label]=int(count)
    expected=list('spdf'[:int(lmax.group(1))+1])
    if set(counts)!=set(expected): raise ValueError(f'{path}: incomplete or inconsistent orbital shell counts.')
    result=[counts[label] for label in expected]
    orbital_layout(result)
    return result


def _read_structure(case, parameters):
    from ase import Atoms
    from ase.data import atomic_numbers
    path=case/parameters.get('stru_file','STRU')
    lines=_lines(path)
    sections={}; active=None
    headings={'ATOMIC_SPECIES','NUMERICAL_ORBITAL','LATTICE_CONSTANT','LATTICE_VECTORS','ATOMIC_POSITIONS',
              'NUMERICAL_DESCRIPTOR','PAW_FILES','LATTICE_PARAMETERS'}
    for line in lines:
        if line in headings:
            if line in sections: raise ValueError(f'{path}: duplicate section {line}.')
            active=line;sections[active]=[]
        elif active is None: raise ValueError(f'{path}: expected STRU section, got {line!r}.')
        else: sections[active].append(line)
    for name in ('ATOMIC_SPECIES','NUMERICAL_ORBITAL','LATTICE_CONSTANT','LATTICE_VECTORS','ATOMIC_POSITIONS'):
        if not sections.get(name): raise ValueError(f'{path}: explicit {name} required.')
    species=[line.split()[0] for line in sections['ATOMIC_SPECIES']]
    if len(set(species))!=len(species) or any(s not in atomic_numbers for s in species):
        raise ValueError('STRU requires unique chemical element labels; custom or repeated species are unsupported.')
    orbs=sections['NUMERICAL_ORBITAL']
    if len(orbs)!=len(species): raise ValueError('One NUMERICAL_ORBITAL file is required per species.')
    orbital_dir=case/parameters.get('orbital_dir','.')
    orbital_paths=[orbital_dir/name for name in orbs]
    counts={s:_orbital_counts(p,s) for s,p in zip(species,orbital_paths)}
    try:
        if len(sections['LATTICE_CONSTANT'])!=1 or len(sections['LATTICE_VECTORS'])!=3: raise ValueError
        a0=float(sections['LATTICE_CONSTANT'][0])*_ABACUS_BOHR_TO_ANGSTROM
        cell=np.array([[float(x) for x in row.split()] for row in sections['LATTICE_VECTORS']])*a0
        if a0<=0 or cell.shape!=(3,3) or not np.isfinite(cell).all() or abs(np.linalg.det(cell))<1e-10: raise ValueError
    except (ValueError,TypeError) as exc: raise ValueError('Finite explicit nonsingular lattice vectors and positive lattice constant required.') from exc
    positions=sections['ATOMIC_POSITIONS']; mode=positions[0].lower(); cursor=1
    if mode not in ('direct','cartesian','cartesian_angstrom','cartesian_au'):
        raise ValueError(f'Unsupported STRU coordinate convention {positions[0]!r}.')
    symbols=[]; coordinates=[]; sites=[]
    try:
        for symbol in species:
            if positions[cursor]!=symbol: raise ValueError('Species order mismatch.')
            natom=int(positions[cursor+2]);cursor+=3
            if natom<=0: raise ValueError('Positive atom counts required.')
            for _ in range(natom):
                coordinate=[float(x) for x in positions[cursor].split()[:3]];cursor+=1
                if len(coordinate)!=3 or not np.isfinite(coordinate).all(): raise ValueError('Invalid position.')
                symbols.append(symbol); coordinates.append(coordinate)
        if cursor!=len(positions): raise ValueError('Extra positions or species sections.')
    except (IndexError,ValueError) as exc: raise ValueError(f'Invalid STRU ATOMIC_POSITIONS: {exc}') from exc
    coordinates=np.array(coordinates)
    original_direct=coordinates.copy() if mode=='direct' else None
    coordinates=coordinates@cell if mode=='direct' else coordinates*{'cartesian':a0,'cartesian_angstrom':1.,'cartesian_au':_ABACUS_BOHR_TO_ANGSTROM}[mode]
    fractional=original_direct if original_direct is not None else np.linalg.solve(cell.T,coordinates.T).T
    # ABACUS read_stru.cpp wraps positions before H(R)/S(R) assembly. Mirror
    # that operation including its floating-point behavior near integer edges.
    source_fractional=np.fmod(fractional+10000.,1.)
    shifts_float=np.rint(fractional-source_fractional)
    if (not np.isfinite(shifts_float).all() or np.any(np.abs(shifts_float)>=2**62)
            or not np.allclose(fractional-source_fractional,shifts_float,rtol=0,atol=1e-9)):
        raise ValueError('Cannot establish ABACUS integer site wrapping shifts from STRU.')
    shifts=shifts_float.astype(np.int64)
    atoms=Atoms(symbols,positions=coordinates,cell=cell,pbc=True)
    permutation=[];phases=[];basis=[];offset=0
    for site,(symbol,position) in enumerate(zip(symbols,coordinates)):
        layout=orbital_layout(counts[symbol]);source_quantum=[]
        for l,count in enumerate(counts[symbol]):
            for zeta in range(1,count+1):
                for m_index in range(2*l+1):
                    m=0 if m_index==0 else ((m_index+1)//2)*(1 if m_index%2 else -1)
                    source_quantum.append({'l':l,'zeta':zeta,'m':m})
        sites.append({'site_index':site,'source_species':symbol,'element':symbol,
                      'position_angstrom':position.tolist(),'orbital_specification':layout.specification,
                      'original_fractional':fractional[site].tolist(),
                      'source_wrapped_fractional':source_fractional[site].tolist(),
                      'source_wrap_shift':shifts[site].tolist()})
        for local,source_index in enumerate(layout.permutation):
            quantum=source_quantum[source_index]
            target_name=_NAMES[quantum['l']][_ANGULAR[quantum['l']].index((0 if quantum['m']==0 else 2*abs(quantum['m'])-(quantum['m']>0)))]
            basis.append({'target_index':len(basis),'source_index':offset+int(source_index),
                          'site_index':site,**quantum,'phase':int(layout.phases[local]),'target_name':target_name})
        permutation.extend((layout.permutation+offset).tolist());phases.extend(layout.phases.tolist());offset+=len(layout.permutation)
    return atoms,np.array(permutation,dtype=np.int64),np.array(phases),sites,basis,{s:orbital_layout(counts[s]).specification for s in species},[path,*orbital_paths]


def _converter_path(explicit=None):
    candidates=[Path(explicit)] if explicit else [Path(__file__).resolve().parents[3]/'scripts/abacus/_vendor/analysis_abacus.py',
                                               Path(sys.prefix)/'share/moirekp/scripts/abacus/_vendor/analysis_abacus.py']
    for candidate in candidates:
        if candidate.is_file(): return candidate.resolve()
    raise FileNotFoundError('ABACUS CSR converter missing; install the companion scripts or pass converter_path.')


def _validate_log_basis(path, sites, specs, scalar_dimension, source_nspin):
    """Cross-check copied orbital files against the actual calculation log."""
    if not path.is_file():
        return
    text=path.read_text()
    records=re.findall(r'READING ATOM TYPE\s+\d+(.*?)(?=READING ATOM TYPE|TOTAL ATOM NUMBER|\Z)',text,re.S)
    observed={}
    for record in records:
        label=re.search(r'Atom label\s*=\s*(\S+)',record)
        entries=re.findall(r'L\s*=\s*(\d+)\s*,\s*number of zeta\s*=\s*(\d+)',record)
        count=re.search(r'Number of atoms for this type\s*=\s*(\d+)',record)
        if not label or not entries or not count:
            raise ValueError(f'{path}: incomplete log basis metadata.')
        symbol=label.group(1)
        if symbol in observed or any(int(l)>3 for l,n in entries):
            raise ValueError(f'{path}: ambiguous or unsupported log basis metadata.')
        counts=[0]*4
        seen=set()
        for l,n in entries:
            l=int(l)
            if l in seen: raise ValueError(f'{path}: duplicate log basis shell.')
            seen.add(l);counts[l]=int(n)
        observed[symbol]=(orbital_layout(counts).specification,int(count.group(1)))
    expected={symbol:(spec,sum(site['element']==symbol for site in sites)) for symbol,spec in specs.items()}
    if observed!=expected:
        raise ValueError(f'{path}: calculation log basis {observed} does not match supplied orbital/STRU metadata {expected}.')
    dimensions=re.findall(r'Number of basis\s*\(NLOCAL\)\s*=\s*(\d+)',text)
    expected_dimension=scalar_dimension*(2 if source_nspin==4 else 1)
    if dimensions and any(int(value)!=expected_dimension for value in dimensions):
        raise ValueError(f'{path}: log basis dimension differs from supplied orbital metadata.')


def _read_dat(path, dimension, permutation, phases, orbital_shifts):
    inverse=np.empty(dimension,dtype=np.int64);inverse[permutation]=np.arange(dimension)
    blocks={};seen=0;source_r=set()
    with path.open() as stream:
        stream.readline();count=int(stream.readline().split()[0]);dim=int(stream.readline().split()[0]);nr=int(stream.readline().split()[0])
        if dim!=dimension: raise ValueError('Converted matrix dimension does not match orbital metadata.')
        for line in stream:
            fields=line.split()
            if len(fields)!=7: raise ValueError('Invalid converted H/S record.')
            r=tuple(map(int,fields[:3]));i,j=int(fields[3])-1,int(fields[4])-1
            source_r.add(r)
            if not (0<=i<dim and 0<=j<dim): raise ValueError('Matrix index outside declared dimension.')
            row,col=inverse[i],inverse[j];value=complex(float(fields[5]),float(fields[6]))*phases[row]*phases[col]
            # Original tau_i = source wrapped tau_i + n_i. Preserve original
            # atom positions by moving each coupling to R' = R + n_i - n_j.
            r=tuple(int(r[d])+int(orbital_shifts[row,d])-int(orbital_shifts[col,d]) for d in range(3))
            if not np.isfinite(value): raise ValueError('Nonfinite H/S matrix value.')
            block=blocks.setdefault(r,{'row':[],'col':[],'val':[]})
            block['row'].append(row);block['col'].append(col);block['val'].append(value);seen+=1
    if seen!=count or len(source_r)>nr: raise ValueError('Converted H/S record count mismatch.')
    for block in blocks.values():
        order=np.lexsort((block['col'],block['row']))
        for key in block: block[key]=np.asarray(block[key],dtype=complex if key=='val' else np.int64)[order]
    return blocks


def _log_fermi(path, step):
    # The SCF log is a sequence of electronic iterations, not a reliable lookup
    # table for an explicitly selected historical CSR ionic step.
    result={'source_fermi_ev':None,'source_fermi_ry':None,'source_fermi_reported_ev':None}
    if step!='latest' or not path.is_file(): return result
    values=[]
    for line in path.read_text().splitlines():
        fields=line.split()
        if len(fields)>=3 and fields[0]=='E_Fermi':
            try: rydberg,ev=map(float,fields[1:3])
            except ValueError as exc: raise ValueError(f'{path}: invalid E_Fermi row.') from exc
            # Native ABACUS uses the historical factor 13.605698 in some
            # versions. Preserve its printed eV separately; canonical Ef uses
            # the same explicit Ry conversion as H, so H-Ef*S stays consistent.
            if not np.isfinite([rydberg,ev]).all() or not np.isclose(rydberg*13.605693122994,ev,rtol=1e-6,atol=1e-7):
                raise ValueError(f'{path}: E_Fermi Ry/eV columns are inconsistent.')
            values.append((rydberg,ev))
    if values:
        rydberg,ev=values[-1]
        result.update(source_fermi_ev=rydberg*13.605693122994,source_fermi_ry=rydberg,source_fermi_reported_ev=ev)
    return result


def _write_dat(path,blocks,dimension,kind):
    with path.open('w') as stream:
        stream.write(f'ABACUS canonical {kind}; '+('eV' if kind=='H' else 'dimensionless')+'\n')
        stream.write(f'{sum(len(v["val"]) for v in blocks.values())}\n{dimension}\n{len(blocks)}\n')
        for r,v in sorted(blocks.items()):
            for i,j,z in zip(v['row'],v['col'],v['val']):
                stream.write(f'{r[0]} {r[1]} {r[2]} {i+1} {j+1} {z.real:.17g} {z.imag:.17g}\n')


def import_abacus(source, output, *, write_dat=False, nspin=None, step='latest', converter_path=None):
    """Import a completed fixed-geometry ABACUS case into a new directory.

    Source must contain INPUT/STRU plus the numerical orbital files. CSR files
    alone cannot establish the angular basis and are deliberately insufficient.
    H keeps ABACUS's absolute energy zero and is converted from Ry to eV once.
    """
    from ase.io import write
    import yaml
    case=Path(source).expanduser().resolve()
    if case.is_file() and case.name=='INPUT': case=case.parent
    if not (case/'INPUT').is_file(): raise ValueError('ABACUS source must be a case directory containing INPUT and STRU.')
    output=Path(output).expanduser().absolute()
    output=output.parent.resolve()/output.name
    if output.exists() or output.is_symlink(): raise FileExistsError(f'Output already exists: {output}.')
    parameters=_parameters(case/'INPUT')
    atoms,perm,phase,sites,basis,specs,source_metadata=_read_structure(case,parameters)
    outdir=case/('OUT.'+parameters.get('suffix','ABACUS'))
    source_files=[case/'INPUT',*source_metadata,*sorted(outdir.glob('*.csr'))]
    log=outdir/('running_'+parameters.get('calculation','scf').lower()+'.log')
    if log.is_file(): source_files.append(log)
    digests={str(p.resolve()):_source_hash(p) for p in source_files}
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.'+output.name+'-',dir=output.parent))
    try:
        raw=temp/'_csr';raw.mkdir()
        command=[sys.executable,str(_converter_path(converter_path)),str(case),'--output-dir',str(raw),
                 '--step',str(step),'--h-cutoff-ev','0','--s-cutoff','0']
        if nspin is not None: command+=['--nspin',str(nspin)]
        process=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        if process.returncode: raise ValueError('ABACUS CSR conversion failed: '+process.stderr.strip())
        rawmeta=json.loads((raw/'matrix_meta.json').read_text())
        scalar_dimension=len(perm);source_nspin=rawmeta['source_nspin'];dimension=rawmeta['output_orbitals']
        if scalar_dimension!=rawmeta['spatial_orbitals']:
            raise ValueError(f'ABACUS orbital metadata dimension {scalar_dimension} differs from matrix dimension {rawmeta["spatial_orbitals"]}.')
        _validate_log_basis(log,sites,specs,scalar_dimension,source_nspin)
        orbital_shifts=np.array([sites[orbital['site_index']]['source_wrap_shift'] for orbital in basis],dtype=np.int64)
        # Converter has already changed interleaved spinors to blocked spin order.
        if source_nspin!=1:
            perm=np.r_[perm,perm+scalar_dimension];phase=np.tile(phase,2)
            orbital_shifts=np.tile(orbital_shifts,(2,1))
        H=_read_dat(raw/'H.dat',dimension,perm,phase,orbital_shifts)
        S=_read_dat(raw/'S.dat',dimension,perm,phase,orbital_shifts)
        if not S: raise ValueError('Empty overlap matrix.')
        residuals={name:_check_hermitian(blocks,dimension,name,tolerance=1e-6) for name,blocks in [('H',H),('S',S)]}
        for filename,digest in digests.items():
            if _source_hash(Path(filename))!=digest: raise ValueError(f'ABACUS source changed during conversion: {filename}.')
        mode={1:'unpolarized',2:'polarized',4:'noncollinear'}[source_nspin]
        metadata={'schema':'tapw.abacus-import.v1','basis_dimension':dimension,'scalar_basis_dimension':scalar_dimension,
                  'spin':source_nspin!=1,'spin_mode':mode,'source_nspin':source_nspin,
                  'spin_order':'scalar' if source_nspin==1 else 'blocked_up_then_down',
                  'hamiltonian_unit':'eV','overlap_unit':'dimensionless','length_unit':'angstrom',
                  'source_native_bohr_to_angstrom':_ABACUS_BOHR_TO_ANGSTROM,
                  'length_conversion_provenance':'ABACUS source_base/constants.h BOHR_TO_A; original unwrapped STRU geometry retained in actual source-native units',
                  'energy_reference':'absolute','energy_zero_ev':0.,'ry_to_ev':13.605693122994,
                  'source_native_ry_to_ev':13.605698,
                  'source_native_ry_to_ev_provenance':'audited ABACUS source_base/constants.h; source_fermi_reported_ev also preserves actual logged eV',
                  **_log_fermi(log,step),
                  'fermi_source':'last E_Fermi Ry column converted with H factor; printed eV kept separately; null if unavailable or historical step requested',
                  'fourier_convention':'sum_R M(R) exp(+2*pi*i*k_fractional.R)',
                  'site_gauge_convention':'original STRU positions retained; source wrapping=fmod(fractional+10000,1); n_i=round(original-wrapped); R_canonical=R_ABACUS+n_i-n_j for H and S',
                  'orbital_convention':'TAPW positive Cartesian real harmonics; ABACUS Condon-Shortley odd-|m| signs removed',
                  'cell_angstrom':atoms.cell.array.tolist(),'sites':sites,'scalar_basis':basis,'system_orbitals':specs,
                  'hermiticity_max_entry_residual':residuals,'translation_vectors':sorted([list(r) for r in set(H)|set(S)]),
                  'source_case':str(case),'source_sha256':digests,'csr_conversion':rawmeta,
                  'scope':'matrix and basis import; fixed geometry only; SCF convergence is not inferred from matrix availability'}
        for name,blocks in [('H',H),('S',S)]:
            writer=HrSparseHandler();writer.hr_sparse=blocks
            # Angular functions are canonical, but atom order remains STRU's.
            # TAPW must still apply its atom-type permutation on NPZ loading.
            writer.save_to_npz(temp/(name+'.npz'),basis_dimension=dimension,basis_order='source')
            if write_dat: _write_dat(temp/(name+'.dat'),blocks,dimension,name)
        write(str(temp/'structure.extxyz'),atoms,format='extxyz')
        fragment={'system':{'structure':'structure.extxyz','hamiltonian':'H.npz','overlap':'S.npz','orbitals':specs,'spin':metadata['spin']}}
        (temp/'system.fragment.yaml').write_text('# Input fragment: supply actual output, twist_index and layers for a moire calculation.\n'+yaml.safe_dump(fragment,sort_keys=False))
        (temp/'README.md').write_text('ABACUS canonical import: H in eV and S dimensionless, with positive Cartesian orbitals and blocked spins.\nSee abacus_import.json for source hashes, exact orbital map and raw CSR metadata.\nThe YAML is an input fragment; provide actual moire metadata and workflow settings.\nNo symmetry averaging or orbital truncation has been applied.\n')
        shutil.rmtree(raw)
        (temp/'abacus_import.json').write_text(json.dumps(metadata,indent=2)+'\n')
        publish_directory(temp,output)
    finally:
        shutil.rmtree(temp)
    return metadata


def main(argv=None, *, prog='tapw import-abacus'):
    parser=argparse.ArgumentParser(prog=prog,description='Import ABACUS case H/S, structure and explicit orbital metadata.')
    parser.add_argument('source',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--write-dat',action='store_true')
    parser.add_argument('--nspin',type=int,choices=(1,2,4))
    parser.add_argument('--step',default='latest')
    parser.add_argument('--converter-path',type=Path)
    args=parser.parse_args(argv)
    try:
        step='latest' if args.step=='latest' else int(args.step)
        metadata=import_abacus(args.source,args.output,write_dat=args.write_dat,nspin=args.nspin,step=step,converter_path=args.converter_path)
    except (ValueError,OSError) as exc: parser.error(str(exc))
    print(f'ABACUS import written to {args.output.resolve()}; basis {metadata["basis_dimension"]}, spin {metadata["spin_mode"]}, H in eV.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())

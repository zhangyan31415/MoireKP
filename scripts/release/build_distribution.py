"""Build current software artifacts and reject research output in the sdist."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', str(ROOT), '--no-deps',
                    '--no-build-isolation', '--wheel-dir', str(out)], check=True,
                   cwd=ROOT)
    # Run the PEP 517 backend outside the project so build/ cannot shadow a tool.
    script = 'import os; os.chdir({root!r}); from setuptools.build_meta import build_sdist; build_sdist({out!r})'
    subprocess.run([sys.executable, '-c', script.format(root=str(ROOT), out=str(out))],
                   check=True, cwd=out)
    wheel = next(out.glob('moirekp-*.whl'))
    source = next(out.glob('moirekp-*.tar.gz'))
    with zipfile.ZipFile(wheel) as archive:
        for package in ['kp', 'tapw']:
            for path in (ROOT / package / package).rglob('*.py'):
                name = path.relative_to(ROOT / package).as_posix()
                if archive.read(name) != path.read_bytes():
                    raise ValueError(f'Wheel/source mismatch: {name}')
        metadata = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        for destination, patterns in metadata['tool']['setuptools']['data-files'].items():
            for pattern in patterns:
                for path in ROOT.glob(pattern):
                    suffix = '/data/' + destination + '/' + path.name
                    matches = [name for name in archive.namelist() if name.endswith(suffix)]
                    if len(matches) != 1 or archive.read(matches[0]) != path.read_bytes():
                        raise ValueError(f'Wheel tool/resource mismatch: {path.relative_to(ROOT)}')
    forbidden = {'paper', 'results', 'validation_runs', 'devtools', 'review_bundles',
                 'review_outputs', 'review_packages', 'build', 'dist', 'tmp', 'output'}
    with tarfile.open(source) as archive:
        members = [m.name.split('/', 1)[1] for m in archive.getmembers()
                   if m.isfile() and '/' in m.name]
        example_inputs = {line.removeprefix('include ').strip()
                          for line in (ROOT / 'MANIFEST.in').read_text().splitlines()
                          if line.startswith('include examples/')}
        invalid = [name for name in members if Path(name).parts[0] in forbidden
                   or any(part.startswith(('outputs', 'stability_', 'current_flow_',
                                            'unweighted_search_', 'runs'))
                          for part in Path(name).parts)
                   or name.endswith(('.log', '.pyc', '.so'))
                   or (name.startswith('examples/') and name not in example_inputs)]
        if invalid:
            raise ValueError(f'Generated/author files in source distribution: {invalid}')
        for member in archive.getmembers():
            if member.isfile() and (member.name.endswith(('.py', '.md', '.toml', '.yaml',
                                                        '.yml', '.sh', '.cpp', '.json'))
                                   or Path(member.name).name == 'PKG-INFO'):
                if b'\x00' in archive.extractfile(member).read():
                    raise ValueError(f'Invalid text file in source distribution: {member.name}')
        for name in ['README.md', 'README.zh.md', 'examples/data-manifest.yaml',
                     'examples/zrs2_3.89/kp/configs/zrs2_3.89_Gamma_spinful_q04.yaml']:
            if name not in members:
                raise ValueError(f'Missing source file: {name}')
    result = dict(status='passed', python=sys.executable, source_files=len(members),
                  artifacts={p.name: dict(bytes=p.stat().st_size,
                             sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                             for p in [wheel, source]})
    (out / 'build_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()

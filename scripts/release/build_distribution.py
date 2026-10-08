"""Build current software artifacts and reject research output in the sdist."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
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
    with tempfile.TemporaryDirectory(prefix='.moirekp-build-', dir=out) as temporary:
        stage = Path(temporary)
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', str(ROOT), '--no-deps',
                        '--no-build-isolation', '--wheel-dir', str(stage)], check=True,
                       cwd=ROOT)
        # Build the source archive outside the checkout so build/ cannot shadow a tool.
        script = 'import os; os.chdir({root!r}); from setuptools.build_meta import build_sdist; build_sdist({out!r})'
        subprocess.run([sys.executable, '-c', script.format(root=str(ROOT), out=str(stage))],
                       check=True, cwd=stage)
        wheels = list(stage.glob('moirekp-*.whl'))
        sources = list(stage.glob('moirekp-*.tar.gz'))
        if len(wheels) != 1 or len(sources) != 1:
            raise ValueError('The build must produce exactly one wheel and one source archive')
        wheel, source = wheels[0], sources[0]
        source_files = verify_distributions(wheel, source)
        wheel = wheel.replace(out / wheel.name)
        source = source.replace(out / source.name)
    result = dict(status='passed', python=sys.executable, source_files=source_files,
                  artifacts={p.name: dict(bytes=p.stat().st_size,
                             sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                             for p in [wheel, source]})
    (out / 'build_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

def verify_distributions(wheel, source):
    with zipfile.ZipFile(wheel) as archive:
        for package in ['kp', 'tapw']:
            for path in (ROOT / package / package).rglob('*.py'):
                if path.relative_to(ROOT / package / package).parts[0] == 'experimental':
                    continue
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
        invalid = [name for name in archive.namelist()
                   if name.startswith(('tests/', 'docs/', 'devtools/', 'kp/experimental/'))
                   or '/share/moirekp/docs/' in name]
        if invalid:
            raise ValueError(f'Internal files in wheel: {invalid}')
    forbidden = {'paper', 'results', 'validation_runs', 'devtools', 'review_bundles',
                 'review_outputs', 'review_packages', 'build', 'dist', 'tmp', 'output',
                 'docs', 'tests'}
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
                   or name.startswith(('kp/kp/experimental/', 'scripts/benchmarks/'))
                   or (name.startswith('examples/zrs2_3.89/') and '_gmkg_' in Path(name).name)
                   or name in {'RELEASE_BLOCKERS.md', 'RELEASE_VALIDATION.md', 'pytest.ini',
                               'scripts/release_gate.sh', 'scripts/release/release_gate.sh',
                               'scripts/benchmark_physical_export.py',
                               'scripts/benchmark_physical_export_suite.py',
                               'examples/data-manifest.yaml', 'tapw/environment.yml'}
                   or (name.startswith('examples/') and name not in example_inputs)]
        if invalid:
            raise ValueError(f'Generated/author files in source distribution: {invalid}')
        for member in archive.getmembers():
            if member.isfile() and (member.name.endswith(('.py', '.md', '.toml', '.yaml',
                                                        '.yml', '.sh', '.cpp', '.json'))
                                   or Path(member.name).name == 'PKG-INFO'):
                if b'\x00' in archive.extractfile(member).read():
                    raise ValueError(f'Invalid text file in source distribution: {member.name}')
        for name in ['README.md', 'README.zh.md',
                     'examples/zrs2_3.89/kp/kp_Gamma.yaml']:
            if name not in members:
                raise ValueError(f'Missing source file: {name}')
    return len(members)

if __name__ == '__main__':
    main()

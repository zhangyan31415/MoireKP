#!/usr/bin/env python3
"""Simple OpenMX H/S symmetrizer wrapper.

Usage:
    python scripts/openmx_symm_hs_python.py --openmx openmx.dat --h H.dat --s S.dat

Run from the target output directory. The script reads the explicitly supplied
OpenMX input file plus H/S sparse files, then writes H_symm.dat, S_symm.dat,
H_symm.npz, S_symm.npz, and symmetrization_report.json in the current directory.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import openmx_symmetrize_hs

OUTPUT_MAP = {
    "H_sym.dat": "H_symm.dat",
    "S_sym.dat": "S_symm.dat",
    "H_sym.npz": "H_symm.npz",
    "S_sym.npz": "S_symm.npz",
    "symmetrization_report.json": "symmetrization_report.json",
}
OUTPUT_FILES = tuple(OUTPUT_MAP.values())


def _resolve_input(path_text: str, cwd: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = cwd / path
    if not path.is_file():
        raise SystemExit(f"Input file not found: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple Python OpenMX H.dat/S.dat symmetrizer. Defaults: cutoff=1e-7, symprec=1e-3, writes DAT+NPZ."
    )
    parser.add_argument("--openmx", required=True, help="Input OpenMX structure/control file, e.g. openmx.dat_rigid or openmx.dat.")
    parser.add_argument("--h", required=True, help="Input Hamiltonian sparse file, e.g. H.dat.")
    parser.add_argument("--s", required=True, help="Input overlap sparse file, e.g. S.dat.")
    parser.add_argument("--force", action="store_true", help="Replace existing output files in the current directory.")
    parser.add_argument("--cutoff", default="1e-7", help="Drop abs(value) <= cutoff. Default: 1e-7.")
    parser.add_argument("--symprec", default="1e-3", help="spglib symmetry tolerance. Default: 1e-3.")
    parser.add_argument("--magnetic-symmetry", choices=("auto", "off", "unitary", "magnetic"), default="auto", help="Magnetic operation filter. Default: auto.")
    parser.add_argument("--magmom-tol", default="1e-5", help="Magnetic moment filter tolerance. Default: 1e-5.")
    parser.add_argument("--block-workers", default="4", help="Thread workers per matrix. Default: 4.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cwd = Path.cwd().resolve()
    openmx_file = _resolve_input(args.openmx, cwd)
    h_file = _resolve_input(args.h, cwd)
    s_file = _resolve_input(args.s, cwd)
    existing = [cwd / name for name in OUTPUT_FILES if (cwd / name).exists()]
    if existing and not args.force:
        names = ", ".join(path.name for path in existing)
        raise SystemExit(f"Output file(s) already exist in current directory: {names}. Use --force to replace them.")
    for path in existing:
        path.unlink()

    temp_parent = Path(tempfile.mkdtemp(prefix=".openmx_symm_hs_python_", dir=cwd))
    temp_case = temp_parent / "case"
    temp_case.mkdir()
    shutil.copy2(openmx_file, temp_case / "openmx.dat")
    temp_dir = temp_parent / "out"
    forwarded = [
        "--input-dir",
        str(temp_case),
        "--output-dir",
        str(temp_dir),
        "--h-file",
        str(h_file),
        "--s-file",
        str(s_file),
        "--cutoff",
        str(args.cutoff),
        "--symprec",
        str(args.symprec),
        "--magnetic-symmetry",
        str(args.magnetic_symmetry),
        "--magmom-tol",
        str(args.magmom_tol),
        "--matrix-workers",
        "2",
        "--block-workers",
        str(args.block_workers),
        "--write-npz-cache",
    ]
    try:
        total_start = time.perf_counter()
        print("[openmx-symm] running Python H/S symmetrizer", flush=True)
        print(f"[openmx-symm] input OpenMX file: {openmx_file}", flush=True)
        print(f"[openmx-symm] input H file: {h_file}", flush=True)
        print(f"[openmx-symm] input S file: {s_file}", flush=True)
        t0 = time.perf_counter()
        rc = openmx_symmetrize_hs.main(forwarded)
        print(f"[openmx-symm] Python symmetrizer exited with code {rc} in {time.perf_counter() - t0:.3f} s", flush=True)
        for source_name, target_name in OUTPUT_MAP.items():
            shutil.move(str(temp_dir / source_name), str(cwd / target_name))
            print(f"[openmx-symm] moved {source_name} -> {target_name}", flush=True)
        print(f"[openmx-symm] finished in {time.perf_counter() - total_start:.3f} s", flush=True)
        return rc
    finally:
        shutil.rmtree(temp_parent, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

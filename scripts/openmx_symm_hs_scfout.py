#!/usr/bin/env python3
"""Simple OpenMX scfout -> symmetrized H/S runner.

Usage:
    python scripts/openmx_symm_hs_scfout.py --openmx openmx.dat --scfout openmx.scfout --binary <analysis_symm_hs>

The wrapper generates symmetry_transports.json from the explicitly supplied
OpenMX input file, then runs the native C++ analysis_symm_hs binary. Outputs
are written to the current directory.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import openmx_symmetrize_hs

OUTPUTS_BY_FORMAT = {
    "npz": ("H_symm.npz", "S_symm.npz"),
    "dat": ("H_symm.dat", "S_symm.dat"),
    "both": ("H_symm.dat", "S_symm.dat", "H_symm.npz", "S_symm.npz"),
}
NATIVE_OUTPUTS_BY_FORMAT = {
    "npz": {"H.npz": "H_symm.npz", "S.npz": "S_symm.npz"},
    "dat": {"H.dat": "H_symm.dat", "S.dat": "S_symm.dat"},
    "both": {"H.dat": "H_symm.dat", "S.dat": "S_symm.dat", "H.npz": "H_symm.npz", "S.npz": "S_symm.npz"},
}
COMMON_OUTPUTS = ("symmetry_transports.json", "symmetrization_report.json")


def _det3(matrix: list[list[object]]) -> float:
    values = [[float(item) for item in row] for row in matrix]
    return (
        values[0][0] * (values[1][1] * values[2][2] - values[1][2] * values[2][1])
        - values[0][1] * (values[1][0] * values[2][2] - values[1][2] * values[2][0])
        + values[0][2] * (values[1][0] * values[2][1] - values[1][1] * values[2][0])
    )


def _fmt_number(value: object, *, zero_tol: float = 1.0e-10) -> str:
    number = float(value)
    rounded = round(number)
    if abs(number - rounded) < zero_tol:
        return str(int(rounded))
    return f"{number:.6g}"


def _wrap_frac(value: object) -> float:
    number = float(value)
    return ((number + 0.5) % 1.0) - 0.5


def _fmt_vector(values: object, *, frac_translation: bool = False) -> str:
    if not isinstance(values, list):
        return str(values)
    items = [_wrap_frac(item) for item in values] if frac_translation else values
    return "(" + ", ".join(_fmt_number(item, zero_tol=1.0e-5) for item in items) + ")"


def _fmt_matrix(matrix: object) -> str:
    if not isinstance(matrix, list):
        return str(matrix)
    return "[" + "; ".join(_fmt_vector(row) for row in matrix) + "]"


def _print_symmetry_summary(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[openmx-symm] warning: could not read symmetry summary from {path}: {exc}", flush=True)
        return

    operations = payload.get("operations", [])
    if not isinstance(operations, list):
        operations = []
    antiunitary_count = sum(1 for op in operations if isinstance(op, dict) and op.get("antiunitary", False))
    unitary_count = len(operations) - antiunitary_count
    print("", flush=True)
    print("[openmx-symm] Symmetry", flush=True)
    print(
        "[openmx-symm]   "
        f"system        : spin_mode={payload.get('spin_mode')}, spinful={payload.get('spinful')}, "
        f"natoms={payload.get('natoms')}, nwann={payload.get('nwann')}",
        flush=True,
    )
    print(
        "[openmx-symm]   "
        f"operations    : spglib={payload.get('spglib_operation_count')}, kept={len(operations)}, "
        f"unitary={unitary_count}, antiunitary={antiunitary_count}",
        flush=True,
    )
    print(
        "[openmx-symm]   "
        f"tolerance     : symprec={payload.get('spglib_symprec')}, "
        f"max_mapping_residual={payload.get('max_atom_mapping_residual')}",
        flush=True,
    )
    magnetic = payload.get("magnetic_symmetry")
    if isinstance(magnetic, dict):
        print(
            "[openmx-symm]   "
            f"magnetic      : mode={magnetic.get('mode')}, applied={magnetic.get('applied')}, "
            f"reason={magnetic.get('reason')}",
            flush=True,
        )
        if "kept_unitary_operation_indices" in magnetic or "kept_primed_operation_indices" in magnetic:
            print(
                "[openmx-symm]   "
                f"kept unitary  : {magnetic.get('kept_unitary_operation_indices', [])}",
                flush=True,
            )
            print(
                "[openmx-symm]   "
                f"kept primed   : {magnetic.get('kept_primed_operation_indices', [])}",
                flush=True,
            )
            print(
                "[openmx-symm]   "
                f"filtered      : {magnetic.get('filtered_operation_indices', [])}",
                flush=True,
            )
    print("[openmx-symm]   operations:", flush=True)
    print("[openmx-symm]     pos  orig    kind         det  translation        rotation_frac", flush=True)
    for position, operation in enumerate(operations):
        if not isinstance(operation, dict):
            continue
        rotation = operation.get("rotation_frac")
        det_text = "?"
        if isinstance(rotation, list) and len(rotation) == 3:
            try:
                det_text = f"{int(round(_det3(rotation))):+d}"
            except Exception:
                det_text = "?"
        op_type = "antiunitary" if operation.get("antiunitary", False) else "unitary"
        print(
            "[openmx-symm]     "
            f"{position:<4} {str(operation.get('index')):<7} {op_type:<12} {det_text:<4} "
            f"t(frac)={_fmt_vector(operation.get('translation_frac'), frac_translation=True):<16} "
            f"R={_fmt_matrix(rotation)}",
            flush=True,
        )


def _resolve_input(path_text: str, cwd: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = cwd / path
    if not path.is_file():
        raise SystemExit(f"Input file not found: {path}")
    return path


def _default_binary() -> str | None:
    env = os.environ.get("OPENMX_SYMM_HS_BIN")
    if env:
        return env
    found = shutil.which("analysis_symm_hs")
    if found:
        return found
    local = Path.cwd() / "build" / "openmx_symm_hs" / "analysis_symm_hs"
    if local.is_file():
        return str(local)
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple C++ OpenMX scfout -> symmetrized H/S runner. Defaults: cutoff=1e-7, output-format=npz."
    )
    parser.add_argument("--openmx", required=True, help="Input OpenMX structure/control file, e.g. openmx.dat_rigid or openmx.dat.")
    parser.add_argument("--binary", help="Path to compiled analysis_symm_hs. Also accepted via OPENMX_SYMM_HS_BIN.")
    parser.add_argument("--scfout", required=True, help="Input OpenMX .scfout file.")
    parser.add_argument("--format", choices=("npz", "dat", "both"), default="npz", help="Output format. Default: npz.")
    parser.add_argument("--threads", type=int, default=48, help="OpenMP threads for native symmetrizer. Default: 48.")
    parser.add_argument("--cutoff", default="1e-7", help="Drop abs(value) <= cutoff. Default: 1e-7.")
    parser.add_argument("--symprec", default="1e-3", help="spglib symmetry tolerance. Default: 1e-3.")
    parser.add_argument("--magnetic-symmetry", choices=("auto", "off", "unitary", "magnetic"), default="auto", help="Magnetic operation filter. Default: auto.")
    parser.add_argument("--magmom-tol", default="1e-5", help="Magnetic moment filter tolerance. Default: 1e-5.")
    parser.add_argument("--force", action="store_true", help="Replace existing output files in the current directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = Path.cwd().resolve()
    output_dir = input_dir
    binary = Path(args.binary or _default_binary() or "").expanduser()
    if not binary.is_file():
        raise SystemExit("analysis_symm_hs binary not found. Pass --binary or set OPENMX_SYMM_HS_BIN.")
    openmx_file = _resolve_input(args.openmx, input_dir)
    scfout = _resolve_input(args.scfout, input_dir)

    output_names = (*COMMON_OUTPUTS, *OUTPUTS_BY_FORMAT[args.format])
    existing = [output_dir / name for name in output_names if (output_dir / name).exists()]
    if existing and not args.force:
        names = ", ".join(path.name for path in existing)
        raise SystemExit(f"Output file(s) already exist in current directory: {names}. Use --force to replace them.")
    for path in existing:
        path.unlink()

    temp_parent = Path(tempfile.mkdtemp(prefix=".openmx_symm_hs_scfout_", dir=output_dir))
    temp_case = temp_parent / "case"
    temp_case.mkdir()
    temp_native = temp_parent / "native_out"
    shutil.copy2(openmx_file, temp_case / "openmx.dat")
    transports = output_dir / "symmetry_transports.json"
    try:
        total_start = time.perf_counter()
        t0 = time.perf_counter()
        print("[openmx-symm] Symmetry analysis", flush=True)
        print(f"[openmx-symm]   openmx      : {openmx_file}", flush=True)
        print(f"[openmx-symm]   output json : {transports}", flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            openmx_symmetrize_hs.main(
                [
                    "--input-dir",
                    str(temp_case),
                    "--output-dir",
                    str(temp_parent / "_dry_run"),
                    "--symprec",
                    str(args.symprec),
                    "--cutoff",
                    str(args.cutoff),
                    "--magnetic-symmetry",
                    str(args.magnetic_symmetry),
                    "--magmom-tol",
                    str(args.magmom_tol),
                    "--export-transports-json",
                    str(transports),
                    "--dry-run",
                ]
            )
        print(f"[openmx-symm]   done        : {time.perf_counter() - t0:.3f} s", flush=True)
        _print_symmetry_summary(transports)

        cmd = [
            str(binary),
            str(scfout),
            "--symmetry-json",
            str(transports),
            "--output-dir",
            str(temp_native),
            "--cutoff",
            str(args.cutoff),
            "--diagnostics",
            "basic",
            "--output-format",
            str(args.format),
            "--threads",
            str(args.threads),
        ]
        print("", flush=True)
        print("[openmx-symm] Native symmetrizer", flush=True)
        print(f"[openmx-symm]   binary      : {binary}", flush=True)
        print(f"[openmx-symm]   scfout      : {scfout}", flush=True)
        print(f"[openmx-symm]   format      : {args.format}", flush=True)
        print(f"[openmx-symm]   cutoff      : {args.cutoff}", flush=True)
        print(f"[openmx-symm]   threads     : {args.threads}", flush=True)
        existing_temporals = {path.resolve() for path in output_dir.glob("temporal_*.input")}
        try:
            t0 = time.perf_counter()
            rc = subprocess.run(cmd, check=False).returncode
            print(f"[openmx-symm]   exit       : code {rc}, {time.perf_counter() - t0:.3f} s", flush=True)
            if rc == 0:
                print("", flush=True)
                print("[openmx-symm] Outputs", flush=True)
                for source_name, target_name in NATIVE_OUTPUTS_BY_FORMAT[args.format].items():
                    shutil.move(str(temp_native / source_name), str(output_dir / target_name))
                    print(f"[openmx-symm]   {target_name}", flush=True)
                shutil.move(str(temp_native / "symmetrization_report.json"), str(output_dir / "symmetrization_report.json"))
                print("[openmx-symm]   symmetrization_report.json", flush=True)
                print(f"[openmx-symm] Finished in {time.perf_counter() - total_start:.3f} s", flush=True)
            return rc
        finally:
            for path in output_dir.glob("temporal_*.input"):
                if path.resolve() not in existing_temporals:
                    path.unlink(missing_ok=True)
    finally:
        shutil.rmtree(temp_parent, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

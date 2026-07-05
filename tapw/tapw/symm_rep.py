"""CLI entry point for TAPW symmetry-representation post-processing."""

from __future__ import annotations

import argparse
from pathlib import Path

from .workflows.symm_rep import run_symm_rep


def build_parser(prog: str = "tapw symm-rep") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Project saved TAPW raw-H symmetry actions into high-symmetry band subspaces.",
    )
    parser.add_argument("--band-dir", required=True, type=Path, help="Directory containing TAPW high-symmetry band outputs.")
    parser.add_argument(
        "--symmetry-dir",
        required=True,
        type=Path,
        help="Directory containing tapw symm raw-H representation outputs.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for symm-rep output files.")
    parser.add_argument("--valence-count", type=int, default=20, help="Number of valence states to report per point.")
    parser.add_argument(
        "--conduction-count",
        type=int,
        default=20,
        help="Number of conduction states to report per point.",
    )
    parser.add_argument(
        "--degeneracy-tol",
        type=float,
        default=2.0e-3,
        help="Energy tolerance for automatic degeneracy grouping.",
    )
    parser.add_argument(
        "--fermi-energy",
        type=float,
        default=None,
        help="Fermi energy separating valence and conduction states.",
    )
    parser.add_argument(
        "--hamiltonian-index",
        type=int,
        default=0,
        help="Slice index to use when hamk_<point>_valley.npy contains multiple Hamiltonians.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing non-empty output directory.")
    return parser


def main(argv=None, *, prog: str = "tapw symm-rep") -> int:
    args = build_parser(prog=prog).parse_args(argv)
    result = run_symm_rep(
        band_dir=args.band_dir,
        symmetry_dir=args.symmetry_dir,
        output_dir=args.output_dir,
        valence_count=args.valence_count,
        conduction_count=args.conduction_count,
        degeneracy_tol=args.degeneracy_tol,
        fermi_energy=args.fermi_energy,
        hamiltonian_index=args.hamiltonian_index,
        overwrite=args.overwrite,
    )
    print(f"Wrote TAPW symmetry representations to {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

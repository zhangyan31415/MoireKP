"""Verify every published k point after a successful model-stage run."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import run_published_examples as replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=tuple(replay.CASES), required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--physical-workers', type=int, default=8)
    args = parser.parse_args()
    baseline = args.root / replay.CASES[args.case][4]
    result = replay.validate(args.output, baseline, args.case,
                             physical_workers=args.physical_workers)
    replay.write(args.report, {'status': 'passed', 'case': args.case,
                              'verification': result,
                              'verification_python': sys.executable})
    print(f'{args.case}: verified {result["kpoints"]} points; dim={result["dimension"]}', flush=True)
    print(args.report, flush=True)


if __name__ == '__main__':
    main()

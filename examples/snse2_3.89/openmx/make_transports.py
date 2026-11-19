"""Generate the six spatial transports for the D3 average of SnSe2 H and S."""
import argparse
from pathlib import Path

import numpy as np

from tapw.io.hs_symmetry import (
    build_operation_transports,
    build_symmetry_operations,
    parse_openmx_structure,
    write_symmetry_transports_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    structure = parse_openmx_structure(args.structure)
    tolerance = 0.01
    operations = build_symmetry_operations(structure, symprec=tolerance)
    if len(operations) != 6 or any(np.linalg.det(op.rotation_cart) < 0.99 for op in operations):
        raise ValueError("Expected six proper D3 rotations in the SOC source structure")
    transports = build_operation_transports(structure, operations, atom_tol=tolerance)
    residual = max(transport.max_mapping_residual for transport in transports)
    if residual >= tolerance:
        raise ValueError(f"D3 site matching exceeded {tolerance} angstrom")
    write_symmetry_transports_json(
        args.output, structure, operations, transports, symprec=tolerance,
        magnetic_report={"source": "SOC from nonmagnetic NSOC restart", "operation_count": len(operations)},
    )
    print(f"Six D3 operations, maximum site-matching residual {residual:.3g} angstrom")


if __name__ == "__main__":
    main()

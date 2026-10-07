#!/usr/bin/env python3
"""Compatibility entry point; shared implementation is tapw.io.hs_symmetry."""
from tapw.io import hs_symmetry as _implementation

globals().update({name: value for name, value in vars(_implementation).items()
                  if not name.startswith('__')})
if __name__ == '__main__':
    raise SystemExit(_implementation.main())
